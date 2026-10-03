# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Main application window — menus, docks, status bar, and keyboard shortcuts.

:class:`MainWindow` is a QMainWindow that wires all Phase 3/4 widgets
together:

* Central widget : :class:`~spriter.ui.canvas.CanvasWidget`
* Left dock      : :class:`~spriter.ui.toolbar.ToolBar`
* Right dock     : :class:`~spriter.ui.color_picker.ColorPicker`
                   :class:`~spriter.ui.layers_panel.LayersPanel`

The window owns the active :class:`~spriter.core.sprite.Sprite` and the
:class:`~spriter.commands.base.CommandStack`.  Undo/redo is routed through
the stack, and every command push invalidates the canvas cache.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QWidget,
)

from spriter.__about__ import __version__

from ..commands.base import CommandStack
from ..core.animation import LoopMode
from ..core.settings import Settings
from ..core.sprite import Sprite
from ..io.gif_io import import_gif
from ..io.png_io import import_png
from ..io.project_io import load as load_project
from ..io.project_io import save as save_project
from ..tools.contiguous_delete import ContiguousDeleteTool
from ..tools.ellipse import EllipseTool
from ..tools.eraser import EraserTool
from ..tools.eyedropper import EyedropperTool
from ..tools.fill import FillTool
from ..tools.line import LineTool
from ..tools.move import MoveTool
from ..tools.pencil import PencilTool
from ..tools.rectangle import RectangleTool
from ..tools.select import RectSelectTool
from ..tools.text import TextTool
from .canvas import CanvasWidget
from .color_picker import ColorPicker
from .layers_panel import LayersPanel
from .main_window_diffusion import DiffusionMixin, _DiffusionWorker
from .main_window_edit import EditMixin
from .main_window_io import IOMixin
from .preferences import PreferencesDialog
from .preview import PreviewWindow
from .timeline import TimelinePanel
from .toolbar import ToolBar


class MainWindow(EditMixin, DiffusionMixin, IOMixin, QMainWindow):
    """Top-level application window.

    Args:
        parent: Optional Qt parent.

    Creating a :class:`MainWindow` automatically opens a new 32×32 sprite.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spriter")
        self.resize(1200, 800)
        from spriter.resources import SPRITE_ICO_BYTES

        _pix = QPixmap()
        _pix.loadFromData(SPRITE_ICO_BYTES)
        self.setWindowIcon(QIcon(_pix))

        self._sprite: Sprite | None = None
        self._settings: Settings = Settings.load()
        self._stack = CommandStack(max_depth=self._settings.max_undo_depth)
        self._current_path: Path | None = None
        self._title_name: str | None = None
        self._unsaved = False
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._do_autosave)
        self._reset_autosave_timer()

        # Widgets (created after the sprite is set up)
        self._canvas: CanvasWidget | None = None
        self._toolbar: ToolBar | None = None
        self._color_picker: ColorPicker | None = None
        self._layers_panel: LayersPanel | None = None
        self._timeline: TimelinePanel | None = None
        self._preview: PreviewWindow | None = None

        # Background worker for diffusion tasks (kept to prevent GC mid-run).
        self._diffusion_worker: _DiffusionWorker | None = None

        # Status-bar labels
        self._status_cursor = QLabel("0, 0")
        self._status_canvas = QLabel("")
        self._status_zoom = QLabel("100%")
        self._status_tool = QLabel("pencil")
        self._status_layer = QLabel("")
        self._init_status_bar()

        # Accept file drops.
        self.setAcceptDrops(True)

        # Build the default project and UI.
        self.new_project(
            self._settings.default_canvas_width,
            self._settings.default_canvas_height,
        )
        self._build_menus()
        self._build_shortcuts()

    @property
    def _unsaved(self) -> bool:
        """Whether the project has unsaved changes."""
        return self._unsaved_flag

    @_unsaved.setter
    def _unsaved(self, value: bool) -> None:
        self._unsaved_flag = bool(value)
        self._update_title()

    def _update_title(self) -> None:
        """Refresh the window title, marking unsaved changes with a leading ``*``."""
        name = self._title_name or "Untitled"
        prefix = "*" if self._unsaved_flag else ""
        self.setWindowTitle(f"Spriter \u2014 {prefix}{name}")

    def _load_imported_sprite(self, sprite: Sprite, title_name: str) -> None:
        """Install an imported/dropped/generated sprite as a new unsaved document."""
        self._sprite = sprite
        self._stack = CommandStack(max_depth=self._settings.max_undo_depth)
        self._current_path = None
        self._title_name = title_name
        self._unsaved = True
        self._rebuild_ui()
        self._status_canvas.setText(f"{sprite.width}\u00d7{sprite.height}")

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    def new_project(self, width: int = 32, height: int = 32) -> None:
        """Create a fresh sprite and rebuild all UI widgets.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
        """
        self._sprite = Sprite(width, height)
        self._sprite.add_layer("Layer 1")
        self._sprite.add_frame()
        self._stack = CommandStack(max_depth=100)
        self._current_path = None
        self._title_name = None
        self._unsaved = False
        self._preview = None  # reset preview on new project

        self._rebuild_ui()
        self._status_canvas.setText(f"{width}×{height}")

    def open_project(self, path: str | None = None) -> None:
        """Load a .spriter project file.

        Args:
            path: File path string.  Opens a file dialog if ``None``.
        """
        # QAction.triggered passes a bool (checked state); treat non-str as None.
        if not isinstance(path, str):
            path = None
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Project",
                self._dialog_dir("open"),
                "Spriter files (*.spriter)",
            )
        if not path:
            return
        self._remember_path(path, "open")
        try:
            sprite = load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not load project:\n{exc}")
            return
        self._sprite = sprite
        self._stack = CommandStack(max_depth=100)
        self._current_path = Path(path)
        self._title_name = Path(path).name
        self._unsaved = False
        self._rebuild_ui()
        w, h = sprite.width, sprite.height
        self._status_canvas.setText(f"{w}×{h}")

    def save_project(self) -> bool:
        """Save to the current path, prompting if none is set.

        Returns:
            True if saved successfully.
        """
        if self._current_path is None:
            return self.save_as_project()
        return self._do_save(self._current_path)

    def save_as_project(self) -> bool:
        """Prompt for a path and save.

        Returns:
            True if saved successfully.
        """
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            self._dialog_dir("save"),
            "Spriter files (*.spriter)",
        )
        if not path:
            return False
        if not path.endswith(".spriter"):
            path += ".spriter"
        self._remember_path(path, "save")
        return self._do_save(Path(path))

    def _do_save(self, path: Path) -> bool:
        try:
            assert self._sprite is not None
            save_project(self._sprite, str(path))
            self._current_path = path
            self._title_name = path.name
            self._unsaved = False
            self._settings.add_recent_file(str(path))
            self._settings.save()
            self._refresh_recent_menu()
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Could not save:\n{exc}")
            return False

    def _do_autosave(self) -> None:
        """Write an autosave copy if the project is dirty and has a path."""
        if self._unsaved and self._current_path and self._sprite:
            try:
                from ..io.project_io import autosave as _autosave

                _autosave(self._sprite, self._current_path)
            except Exception:
                pass

    def _reset_autosave_timer(self) -> None:
        interval = self._settings.autosave_interval_ms
        if interval > 0:
            self._autosave_timer.start(interval)
        else:
            self._autosave_timer.stop()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _rebuild_ui(self) -> None:
        """(Re)create all dock widgets and the central canvas."""
        assert self._sprite is not None

        # Remove any existing dock widgets before recreating them so that
        # repeated calls (e.g. File→New) don't accumulate duplicate panels.
        for dock in list(self.findChildren(QDockWidget)):
            self.removeDockWidget(dock)
            dock.setParent(None)  # type: ignore[arg-type]
            dock.deleteLater()

        # ── Canvas ───────────────────────────────────────────────────
        self._canvas = CanvasWidget(self._sprite, self._stack)
        self._canvas.cursor_moved.connect(self._on_cursor_moved)
        self._canvas.zoom_changed.connect(
            lambda z: self._status_zoom.setText(f"{int(z * 100)}%")
        )
        self._canvas.color_sampled.connect(self._on_color_sampled)
        self.setCentralWidget(self._canvas)
        self._toolbar = ToolBar(keybindings=self._settings.keybindings)
        self._toolbar.tool_changed.connect(self._on_tool_changed)
        self._toolbar.brush_size_changed.connect(self._on_brush_size_changed)
        self._toolbar.opacity_changed.connect(self._on_opacity_changed)
        self._toolbar.grid_toggled.connect(self._on_toolbar_grid_toggled)
        self._toolbar.fit_requested.connect(self._fit)
        self._toolbar.center_requested.connect(self._center_view)
        self._toolbar.sprite_sheet_requested.connect(self._diffusion_generate_sheet)
        if hasattr(self, "_grid_action"):
            self._toolbar._grid_button.setChecked(self._grid_action.isChecked())
        tool_dock = QDockWidget("Tools", self)
        tool_dock.setWidget(self._toolbar)
        tool_dock.setObjectName("tools_dock")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, tool_dock)

        # ── Color picker (right dock, top) ────────────────────────────
        self._color_picker = ColorPicker()
        self._color_picker.foreground_changed.connect(self._on_fg_color_changed)
        color_dock = QDockWidget("Colors", self)
        color_dock.setWidget(self._color_picker)
        color_dock.setObjectName("colors_dock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, color_dock)

        # ── Layers panel (right dock, below colors) ───────────────────
        self._layers_panel = LayersPanel(self._sprite, self._stack)
        self._layers_panel.active_layer_changed.connect(self._on_active_layer_changed)
        self._layers_panel.layer_visibility_changed.connect(
            lambda li, v: self._canvas.invalidate_cache() if self._canvas else None
        )
        self._layers_panel.layers_modified.connect(self._on_layers_modified)
        layers_dock = QDockWidget("Layers", self)
        layers_dock.setWidget(self._layers_panel)
        layers_dock.setObjectName("layers_dock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, layers_dock)

        # ── Timeline panel (bottom dock) ──────────────────────────────
        self._timeline = TimelinePanel(self._sprite, self._stack)
        self._timeline.frame_selected.connect(self._on_timeline_frame_selected)
        self._timeline.frame_duration_changed.connect(
            lambda fi, ms: self._canvas.invalidate_cache() if self._canvas else None
        )
        self._timeline.animation_changed.connect(self._on_animation_changed)
        timeline_dock = QDockWidget("Timeline", self)
        timeline_dock.setWidget(self._timeline)
        timeline_dock.setObjectName("timeline_dock")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, timeline_dock)

        # Select the pencil tool on startup.
        self._on_tool_changed("pencil")
        self._update_layer_status()

    def _top_menu(self, title: str) -> QMenu:
        """Add a top-level menu, narrowing the optional menu bar / menu away."""
        mb = self.menuBar()
        assert mb is not None
        menu = mb.addMenu(title)
        assert menu is not None
        return menu

    def _build_menus(self) -> None:
        # ── File ──────────────────────────────────────────────────────
        file_menu = self._top_menu("&File")
        self._add_action(file_menu, "&New…", self._prompt_new, "Ctrl+N")
        self._add_action(file_menu, "&Open…", self.open_project, "Ctrl+O")
        file_menu.addSeparator()
        self._add_action(file_menu, "&Save", self.save_project, "Ctrl+S")
        self._add_action(file_menu, "Save &As…", self.save_as_project, "Ctrl+Shift+S")
        file_menu.addSeparator()
        # Export sub-menu
        export_menu = QMenu("&Export", self)
        self._add_action(export_menu, "Export Frame as &PNG…", self._export_frame_png)
        self._add_action(
            export_menu, "Export All Frames as PNG…", self._export_all_frames_png
        )
        self._add_action(export_menu, "Export Animated &GIF…", self._export_gif)
        export_menu.addSeparator()
        self._add_action(export_menu, "Export Sprite &Sheet…", self._export_sheet)
        self._add_action(export_menu, "Export Sheet + &Atlas…", self._export_atlas)
        export_menu.addSeparator()
        self._add_action(export_menu, "Export as &ICO…", self._export_ico)
        export_menu.addSeparator()
        self._add_action(export_menu, "Export &Palette…", self._export_palette)
        file_menu.addMenu(export_menu)
        # Import sub-menu
        import_menu = QMenu("&Import", self)
        self._add_action(import_menu, "Import &PNG as Sprite…", self._import_png)
        self._add_action(import_menu, "Import &GIF as Sprite…", self._import_gif)
        self._add_action(import_menu, "Import Sprite &Sheet…", self._import_sheet)
        import_menu.addSeparator()
        self._add_action(import_menu, "Import &Palette…", self._import_palette)
        file_menu.addMenu(import_menu)
        file_menu.addSeparator()
        # Copy / Paste
        self._add_action(file_menu, "&Copy Selection", self._copy_selection, "Ctrl+C")
        self._add_action(
            file_menu, "&Paste from Clipboard", self._paste_clipboard, "Ctrl+V"
        )
        file_menu.addSeparator()
        # Recent files
        self._recent_menu = QMenu("&Recent Files", self)
        file_menu.addMenu(self._recent_menu)
        self._refresh_recent_menu()
        file_menu.addSeparator()
        self._add_action(file_menu, "E&xit", self.close, "Ctrl+Q")

        # ── Edit ──────────────────────────────────────────────────────
        edit_menu = self._top_menu("&Edit")
        self._undo_action = self._add_action(edit_menu, "&Undo", self._undo, "Ctrl+Z")
        self._redo_action = self._add_action(edit_menu, "&Redo", self._redo, "Ctrl+Y")
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Select &All", self._select_all, "Ctrl+A")
        self._add_action(edit_menu, "Select &None", self._select_none, "Ctrl+D")

        # ── View ──────────────────────────────────────────────────────
        view_menu = self._top_menu("&View")
        self._add_action(view_menu, "Zoom &In", self._zoom_in, "Ctrl+=")
        self._add_action(view_menu, "Zoom &Out", self._zoom_out, "Ctrl+-")
        self._add_action(view_menu, "&Fit to Window", self._fit, "Ctrl+Shift+H")
        self._add_action(view_menu, "&Center View", self._center_view, "Ctrl+Shift+C")
        view_menu.addSeparator()
        self._grid_action = self._add_action(
            view_menu, "Show &Grid", self._toggle_grid, "Ctrl+G", checkable=True
        )
        self._grid_action.setChecked(True)

        # ── Layer ─────────────────────────────────────────────────────
        layer_menu = self._top_menu("&Layer")
        self._add_action(layer_menu, "&Add Layer", self._add_layer, "Ctrl+Shift+N")
        self._add_action(layer_menu, "&Delete Layer", self._delete_layer)
        self._add_action(
            layer_menu, "D&uplicate Layer", self._duplicate_layer, "Ctrl+J"
        )
        layer_menu.addSeparator()
        self._add_action(layer_menu, "Merge &Down", self._merge_down, "Ctrl+E")
        self._add_action(layer_menu, "&Flatten Image", self._flatten)
        layer_menu.addSeparator()
        self._add_action(layer_menu, "Re&name Layer\u2026", self._rename_layer)
        layer_menu.addSeparator()
        role_menu = QMenu("Set &Role", self)
        self._add_action(role_menu, "&Normal", self._set_layer_role_normal)
        self._add_action(role_menu, "&Foreground", self._set_layer_role_foreground)
        self._add_action(role_menu, "&Background", self._set_layer_role_background)
        layer_menu.addMenu(role_menu)

        # ── Frame ─────────────────────────────────────────────────────
        frame_menu = self._top_menu("Fr&ame")
        self._add_action(frame_menu, "&Add Frame", self._add_frame)
        self._add_action(frame_menu, "&Delete Frame", self._delete_frame)
        self._add_action(frame_menu, "D&uplicate Frame", self._duplicate_frame)
        frame_menu.addSeparator()
        self._add_action(
            frame_menu, "Move Frame &Left", self._move_frame_left, "Ctrl+Shift+,"
        )
        self._add_action(
            frame_menu, "Move Frame &Right", self._move_frame_right, "Ctrl+Shift+."
        )

        # ── Animation ─────────────────────────────────────────────────
        anim_menu = self._top_menu("&Animation")
        self._add_action(anim_menu, "&Preview…", self._show_preview)
        anim_menu.addSeparator()
        self._add_action(
            anim_menu, "Loop Mode: &Loop", lambda: self._set_loop_mode(LoopMode.LOOP)
        )
        self._add_action(
            anim_menu,
            "Loop Mode: &Ping-Pong",
            lambda: self._set_loop_mode(LoopMode.PING_PONG),
        )
        self._add_action(
            anim_menu,
            "Loop Mode: &One-Shot",
            lambda: self._set_loop_mode(LoopMode.ONE_SHOT),
        )
        anim_menu.addSeparator()
        self._onion_action = self._add_action(
            anim_menu, "&Onion Skinning", self._toggle_onion_skin, checkable=True
        )
        self._add_action(anim_menu, "Onion Skin &Depth\u2026", self._prompt_onion_depth)

        # ── Transform ─────────────────────────────────────────────────
        xform_menu = self._top_menu("&Transform")
        self._add_action(xform_menu, "Flip &Horizontal", self._flip_h)
        self._add_action(xform_menu, "Flip &Vertical", self._flip_v)
        xform_menu.addSeparator()
        self._add_action(xform_menu, "Rotate &90° CW", lambda: self._rotate(90))
        self._add_action(xform_menu, "Rotate 90° &CCW", lambda: self._rotate(-90))
        self._add_action(xform_menu, "Rotate &180°", lambda: self._rotate(180))
        xform_menu.addSeparator()
        self._add_action(xform_menu, "&Canvas Size…", self._prompt_canvas_resize)
        self._add_action(xform_menu, "Crop to S&election", self._crop_to_selection)
        self._add_action(xform_menu, "Auto&crop", self._autocrop)
        self._add_action(xform_menu, "&Scale Image…", self._prompt_scale)
        self._add_action(xform_menu, "Scale Se&lection…", self._prompt_scale_selection)
        xform_menu.addSeparator()
        self._add_action(xform_menu, "Shift / &Offset…", self._prompt_shift)
        self._add_action(xform_menu, "&Outline", self._apply_outline)
        self._add_action(xform_menu, "Replace &Color…", self._prompt_replace_color)
        self._add_action(xform_menu, "&Invert Colors", self._invert_colors)
        self._add_action(
            xform_menu, "Invert Colors in &Selection", self._invert_colors_selection
        )
        xform_menu.addSeparator()
        self._add_action(
            xform_menu, "&Brightness / Contrast…", self._prompt_adjust_brightness
        )
        self._add_action(xform_menu, "H&ue / Saturation…", self._prompt_adjust_hue)
        # ── View additions (Phase 8) ───────────────────────────────────
        view_menu.addSeparator()
        self._sym_h_action = self._add_action(
            view_menu, "Symmetry: &Horizontal", self._toggle_sym_h, checkable=True
        )
        self._sym_v_action = self._add_action(
            view_menu, "Symmetry: &Vertical", self._toggle_sym_v, checkable=True
        )
        view_menu.addSeparator()
        self._tiling_action = self._add_action(
            view_menu, "&Tiling Preview", self._toggle_tiling, checkable=True
        )
        self._add_action(
            view_menu, "Set &Reference Image\u2026", self._set_reference_image
        )
        self._add_action(
            view_menu, "Clear Reference Image", self._clear_reference_image
        )

        # ── Diffusion (optional AI frame prediction) ──────────────────
        diffusion_menu = self._top_menu("&Diffusion")
        self._add_action(
            diffusion_menu, "&Select Model\u2026", self._diffusion_select_model
        )
        self._add_action(
            diffusion_menu,
            "Select &ControlNet\u2026",
            self._diffusion_select_controlnet,
        )
        self._add_action(
            diffusion_menu,
            "Select IP-&Adapter\u2026",
            self._diffusion_select_ip_adapter,
        )
        self._add_action(
            diffusion_menu,
            "Select &Video Model\u2026",
            self._diffusion_select_video_model,
        )
        self._add_action(
            diffusion_menu,
            "Select Video &LoRA\u2026",
            self._diffusion_select_video_lora,
        )
        self._add_action(
            diffusion_menu, "&Download Model\u2026", self._diffusion_download_model
        )
        self._add_action(
            diffusion_menu, "Model &Info\u2026", self._diffusion_model_info
        )
        diffusion_menu.addSeparator()
        self._add_action(
            diffusion_menu,
            "&Restyle Frame\u2026",
            self._diffusion_restyle_frame,
        )
        self._add_action(
            diffusion_menu,
            "&Predict Next Frame\u2026",
            self._diffusion_predict_next_frame,
        )
        self._add_action(
            diffusion_menu,
            "&Animate From Frame\u2026",
            self._diffusion_animate_from_frame,
        )

        # ── Preferences ───────────────────────────────────────────────
        prefs_menu = self._top_menu("&Preferences")
        self._add_action(
            prefs_menu, "&Preferences\u2026", self._open_preferences
        )  # ── Help ──────────────────────────────────────────────────────
        help_menu = self._top_menu("&Help")
        self._add_action(help_menu, "&About…", self._show_about)

    def _add_action(
        self,
        menu,
        text: str,
        slot,
        shortcut: str | None = None,
        checkable: bool = False,
    ) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_shortcuts(self) -> None:
        """Additional canvas-level keyboard shortcuts (respects keybinding settings)."""
        # Build key→tool map from settings keybindings (inverted).
        shortcuts = {v: k for k, v in self._settings.keybindings.items()}
        from PyQt6.QtGui import QShortcut

        for key, tool_name in shortcuts.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(
                lambda n=tool_name: (
                    self._toolbar.select_tool(n) if self._toolbar else None
                )
            )

        # Swap FG/BG colours (X key — Photoshop/Aseprite convention).
        swap_sc = QShortcut(QKeySequence("X"), self)
        swap_sc.activated.connect(
            lambda: self._color_picker.swap_colors() if self._color_picker else None
        )

    def _init_status_bar(self) -> None:
        bar = self.statusBar()
        assert bar is not None
        bar.addWidget(QLabel("Cursor:"))
        bar.addWidget(self._status_cursor)
        bar.addWidget(QLabel("  Canvas:"))
        bar.addWidget(self._status_canvas)
        bar.addWidget(QLabel("  Zoom:"))
        bar.addWidget(self._status_zoom)
        bar.addWidget(QLabel("  Tool:"))
        bar.addWidget(self._status_tool)
        bar.addWidget(QLabel("  Layer:"))
        bar.addWidget(self._status_layer)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_cursor_moved(self, x: int, y: int) -> None:
        self._status_cursor.setText(f"{x}, {y}")

    def _tool_status_text(self, name: str) -> str:
        """Human-readable tool name with its shortcut, e.g. ``"Pencil (B)"``."""
        label = name.replace("_", " ").title()
        key = self._settings.keybindings.get(name, "")
        return f"{label} ({key.upper()})" if key else label

    def _update_layer_status(self) -> None:
        """Show the active layer's name in the status bar."""
        if self._sprite is None or self._layers_panel is None:
            return
        idx = self._layers_panel.active_layer
        layers = self._sprite.layers
        self._status_layer.setText(layers[idx].name if 0 <= idx < len(layers) else "")

    def _on_tool_changed(self, name: str) -> None:
        if self._canvas is None or self._sprite is None:
            return
        self._status_tool.setText(self._tool_status_text(name))
        tool = self._make_tool(name)
        if self._color_picker:
            fg = self._color_picker.foreground
            tool.foreground = fg
        if self._toolbar:
            tool.brush_size = self._toolbar.brush_size
            tool.opacity = self._toolbar.opacity
        if self._layers_panel:
            tool.layer_index = self._layers_panel.active_layer
        tool.frame_index = self._canvas.active_frame
        self._canvas.set_tool(tool)

    def _on_brush_size_changed(self, size: int) -> None:
        if self._canvas and self._canvas._tool:
            self._canvas._tool.brush_size = size

    def _on_opacity_changed(self, opacity: int) -> None:
        if self._canvas and self._canvas._tool:
            self._canvas._tool.opacity = opacity

    def _on_fg_color_changed(self, color: tuple) -> None:
        if self._canvas and self._canvas._tool:
            self._canvas._tool.foreground = color

    def _on_color_sampled(self, color: tuple) -> None:
        """Update the color picker when a tool (e.g. eyedropper) samples a pixel."""
        if self._color_picker:
            self._color_picker.foreground = color

    def _on_active_layer_changed(self, layer_idx: int) -> None:
        if self._canvas:
            self._canvas.active_layer = layer_idx
            if self._canvas._tool:
                self._canvas._tool.layer_index = layer_idx
                self._canvas._tool.cancel()
        self._update_layer_status()
        if self._sprite:
            self._sprite.clear_selection()
            if self._canvas:
                self._canvas.update()

    def _on_layers_modified(self) -> None:
        if self._canvas:
            self._canvas.invalidate_cache()
            if self._layers_panel:
                new_layer = self._layers_panel.active_layer
                self._canvas.active_layer = new_layer
                if self._canvas._tool:
                    self._canvas._tool.layer_index = new_layer
        if self._timeline:
            self._timeline.refresh()
        self._unsaved = True
        self._refresh_undo_redo_labels()

    def _select_all(self) -> None:
        if self._sprite and self._canvas:
            import numpy as np

            h, w = self._sprite.height, self._sprite.width
            self._sprite.selection_mask = np.ones((h, w), dtype=bool)
            self._canvas.update()

    def _select_none(self) -> None:
        if self._sprite and self._canvas:
            self._sprite.clear_selection()
            self._canvas.update()

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    def _prompt_new(self) -> None:
        w, ok1 = QInputDialog.getInt(self, "New Sprite", "Width (px):", 32, 1, 4096)
        if not ok1:
            return
        h, ok2 = QInputDialog.getInt(self, "New Sprite", "Height (px):", 32, 1, 4096)
        if ok2:
            self.new_project(w, h)

    def _refresh_undo_redo_labels(self) -> None:
        """Update Undo/Redo menu item text with the top-of-stack description."""
        desc = self._stack.undo_description
        self._undo_action.setText(f"&Undo: {desc}" if desc else "&Undo")
        desc = self._stack.redo_description
        self._redo_action.setText(f"&Redo: {desc}" if desc else "&Redo")

    def _undo(self) -> None:
        if self._stack.can_undo:
            self._stack.undo()
            if self._canvas and self._sprite and self._sprite.frame_count > 0:
                new_fi = min(self._canvas.active_frame, self._sprite.frame_count - 1)
                self._canvas.active_frame = max(0, new_fi)
            if self._canvas:
                self._canvas.invalidate_cache()
            if self._layers_panel:
                self._layers_panel.refresh()
                if self._canvas:
                    self._canvas.active_layer = self._layers_panel.active_layer
            if self._timeline:
                self._timeline.refresh()
        self._refresh_undo_redo_labels()

    def _redo(self) -> None:
        if self._stack.can_redo:
            self._stack.redo()
            if self._canvas and self._sprite and self._sprite.frame_count > 0:
                new_fi = min(self._canvas.active_frame, self._sprite.frame_count - 1)
                self._canvas.active_frame = max(0, new_fi)
            if self._canvas:
                self._canvas.invalidate_cache()
            if self._layers_panel:
                self._layers_panel.refresh()
                if self._canvas:
                    self._canvas.active_layer = self._layers_panel.active_layer
            if self._timeline:
                self._timeline.refresh()
        self._refresh_undo_redo_labels()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Spriter",
            "Spriter \u2014 Pixel art editor\n\nVersion " + __version__,
        )

    # ------------------------------------------------------------------
    # File-dialog default location
    # ------------------------------------------------------------------

    def _dialog_dir(self, mode: str = "open") -> str:
        """Return the directory file dialogs should open in.

        Defaults to the most recently used location for the given mode
        ("open" for open/import, "save" for save/export).  Returns ``""`` if no
        directory has been recorded yet, which causes Qt to use its platform default.

        Args:
            mode: Either "open" or "save" to determine which directory to use.
        """
        last = (
            self._settings.last_open_directory
            if mode == "open"
            else self._settings.last_save_directory
        )
        if last and Path(last).is_dir():
            return last
        return ""

    def _remember_path(self, path: str, mode: str = "open") -> None:
        """Record *path*'s parent directory and persist settings.

        Args:
            path: File path to remember.
            mode: Either "open" or "save" to determine which directory to update.
        """
        if not path:
            return
        if mode == "open":
            self._settings.remember_open_path(path)
        else:
            self._settings.remember_save_path(path)
        try:
            self._settings.save()
        except Exception:
            pass

    def _remember_directory(self, directory: str, mode: str = "open") -> None:
        """Record *directory* as the last-used location and persist settings.

        Args:
            directory: Directory path to remember.
            mode: Either "open" or "save" to determine which directory to update.
        """
        if not directory:
            return
        if mode == "open":
            self._settings.remember_open_directory(directory)
        else:
            self._settings.remember_save_directory(directory)
        try:
            self._settings.save()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 8: Recent files
    # ------------------------------------------------------------------

    def _refresh_recent_menu(self) -> None:
        """Rebuild the Recent Files sub-menu from current settings."""
        if not hasattr(self, "_recent_menu"):
            return
        self._recent_menu.clear()
        recent = self._settings.recent_files
        if not recent:
            action = QAction("(No recent files)", self)
            action.setEnabled(False)
            self._recent_menu.addAction(action)
            return
        for file_path in recent:
            action = QAction(file_path, self)
            action.triggered.connect(
                lambda checked=False, p=file_path: self.open_project(p)
            )
            self._recent_menu.addAction(action)

    # ------------------------------------------------------------------
    # Phase 8: Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        urls = event.mimeData().urls()
        if not urls:
            return
        url: QUrl = urls[0]
        path = url.toLocalFile()
        lowered = path.lower()
        if lowered.endswith(".spriter"):
            self.open_project(path)
        else:
            # Try importing as a GIF or PNG/image.
            try:
                if lowered.endswith(".gif"):
                    sprite = import_gif(path)
                else:
                    sprite = import_png(path)
            except Exception as exc:
                QMessageBox.critical(self, "Drop Error", str(exc))
                return
            self._load_imported_sprite(sprite, Path(path).name)

    # ------------------------------------------------------------------
    # Phase 8: Symmetry mode
    # ------------------------------------------------------------------

    def _toggle_sym_h(self) -> None:
        if self._canvas:
            self._canvas.symmetry_h = self._sym_h_action.isChecked()

    def _toggle_sym_v(self) -> None:
        if self._canvas:
            self._canvas.symmetry_v = self._sym_v_action.isChecked()

    # ------------------------------------------------------------------
    # Phase 8: Tiling preview
    # ------------------------------------------------------------------

    def _toggle_tiling(self) -> None:
        if self._canvas:
            self._canvas.tiling_preview = self._tiling_action.isChecked()
            self._canvas.update()

    # ------------------------------------------------------------------
    # Phase 8: Reference image
    # ------------------------------------------------------------------

    def _set_reference_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Reference Image",
            self._dialog_dir("open"),
            "Images (*.png *.bmp *.jpg *.jpeg)",
        )
        if not path or self._canvas is None:
            return
        self._remember_path(path, "open")
        try:
            import numpy as np
            from PIL import Image

            img = Image.open(path).convert("RGBA")
            self._canvas.reference_image = np.array(img, dtype=np.uint8)
            self._canvas.update()
        except Exception as exc:
            QMessageBox.critical(self, "Reference Image Error", str(exc))

    def _clear_reference_image(self) -> None:
        if self._canvas:
            self._canvas.reference_image = None
            self._canvas.update()

    # ------------------------------------------------------------------
    # Phase 8: Preferences dialog
    # ------------------------------------------------------------------

    def _open_preferences(self) -> None:
        old_w = self._settings.default_canvas_width
        old_h = self._settings.default_canvas_height
        dlg = PreferencesDialog(self._settings, self)
        if dlg.exec():
            self._settings.save()
            self._reset_autosave_timer()
            # Re-apply the theme live so changes are visible immediately.
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                from . import theme as _theme

                _theme.apply_theme(app, self._settings.theme)
            # Re-apply shortcut bindings.
            self._build_shortcuts()
            # QoL: if the default canvas size changed and the project has only
            # one frame, immediately resize the current canvas to match.
            new_w = self._settings.default_canvas_width
            new_h = self._settings.default_canvas_height
            if (
                self._sprite is not None
                and self._sprite.frame_count == 1
                and (new_w != old_w or new_h != old_h)
            ):
                from ..commands.transform import CanvasResizeCommand

                self._push_transform(CanvasResizeCommand(self._sprite, new_w, new_h))
                self._status_canvas.setText(
                    f"{self._sprite.width}\u00d7{self._sprite.height}"
                )

    # ------------------------------------------------------------------
    # Tool factory
    # ------------------------------------------------------------------

    def _make_tool(self, name: str):
        assert self._sprite is not None
        tools = {
            "pencil": PencilTool,
            "eraser": EraserTool,
            "line": LineTool,
            "rectangle": RectangleTool,
            "ellipse": EllipseTool,
            "fill": FillTool,
            "contiguous_delete": ContiguousDeleteTool,
            "eyedropper": EyedropperTool,
            "select": RectSelectTool,
            "move": MoveTool,
            "text": TextTool,
        }
        cls = tools.get(name, PencilTool)
        return cls(self._sprite, self._stack)  # type: ignore[abstract]

    # ------------------------------------------------------------------
    # Close guard
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._unsaved:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        event.accept()
