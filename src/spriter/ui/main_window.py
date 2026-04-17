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
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from ..commands.base import CommandStack
from ..commands.frame_ops import (
    AddFrameCommand,
    DuplicateFrameCommand,
    RemoveFrameCommand,
)
from ..commands.layer_ops import (
    AddLayerCommand,
    DuplicateLayerCommand,
    FlattenCommand,
    MergeLayerDownCommand,
    RemoveLayerCommand,
)
from ..core.sprite import Sprite
from ..io.project_io import load as load_project
from ..io.project_io import save as save_project
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
from ..commands.transform import (
    AdjustmentCommand,
    CanvasResizeCommand,
    FlipCommand,
    OutlineCommand,
    ReplaceColorCommand,
    RotateCommand,
    ScaleCommand,
    ShiftCommand,
)
from ..core.animation import LoopMode
from .canvas import CanvasWidget
from .color_picker import ColorPicker
from .layers_panel import LayersPanel
from .preview import PreviewWindow
from .timeline import TimelinePanel
from .toolbar import ToolBar


class MainWindow(QMainWindow):
    """Top-level application window.

    Args:
        parent: Optional Qt parent.

    Creating a :class:`MainWindow` automatically opens a new 32×32 sprite.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spriter")
        self.resize(1200, 800)

        self._sprite: Optional[Sprite] = None
        self._stack = CommandStack(max_depth=100)
        self._current_path: Optional[Path] = None
        self._unsaved = False

        # Widgets (created after the sprite is set up)
        self._canvas: Optional[CanvasWidget] = None
        self._toolbar: Optional[ToolBar] = None
        self._color_picker: Optional[ColorPicker] = None
        self._layers_panel: Optional[LayersPanel] = None
        self._timeline: Optional[TimelinePanel] = None
        self._preview: Optional[PreviewWindow] = None

        # Status-bar labels
        self._status_cursor = QLabel("0, 0")
        self._status_canvas = QLabel("")
        self._status_zoom = QLabel("100%")
        self._status_tool = QLabel("pencil")
        self._init_status_bar()

        # Build the default project and UI.
        self.new_project(32, 32)
        self._build_menus()
        self._build_shortcuts()

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
        self._unsaved = False
        self._preview = None  # reset preview on new project

        self._rebuild_ui()
        self._status_canvas.setText(f"{width}×{height}")

    def open_project(self, path: Optional[str] = None) -> None:
        """Load a .spriter project file.

        Args:
            path: File path string.  Opens a file dialog if ``None``.
        """
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Project", "", "Spriter files (*.spriter)"
            )
        if not path:
            return
        try:
            sprite = load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not load project:\n{exc}")
            return
        self._sprite = sprite
        self._stack = CommandStack(max_depth=100)
        self._current_path = Path(path)
        self._unsaved = False
        self._rebuild_ui()
        w, h = sprite.width, sprite.height
        self._status_canvas.setText(f"{w}×{h}")
        self.setWindowTitle(f"Spriter — {Path(path).name}")

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
            self, "Save Project As", "", "Spriter files (*.spriter)"
        )
        if not path:
            return False
        if not path.endswith(".spriter"):
            path += ".spriter"
        return self._do_save(Path(path))

    def _do_save(self, path: Path) -> bool:
        try:
            assert self._sprite is not None
            save_project(self._sprite, str(path))
            self._current_path = path
            self._unsaved = False
            self.setWindowTitle(f"Spriter — {path.name}")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Could not save:\n{exc}")
            return False

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _rebuild_ui(self) -> None:
        """(Re)create all dock widgets and the central canvas."""
        assert self._sprite is not None

        # ── Canvas ───────────────────────────────────────────────────
        self._canvas = CanvasWidget(self._sprite, self._stack)
        self._canvas.cursor_moved.connect(self._on_cursor_moved)
        self._canvas.zoom_changed.connect(
            lambda z: self._status_zoom.setText(f"{int(z * 100)}%")
        )
        self.setCentralWidget(self._canvas)

        # ── Tool bar (left dock) ──────────────────────────────────────
        self._toolbar = ToolBar()
        self._toolbar.tool_changed.connect(self._on_tool_changed)
        self._toolbar.brush_size_changed.connect(self._on_brush_size_changed)
        self._toolbar.opacity_changed.connect(self._on_opacity_changed)
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
            lambda li, v: self._canvas.invalidate_cache()
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
            lambda fi, ms: self._canvas.invalidate_cache()
        )
        timeline_dock = QDockWidget("Timeline", self)
        timeline_dock.setWidget(self._timeline)
        timeline_dock.setObjectName("timeline_dock")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, timeline_dock)

        # Select the pencil tool on startup.
        self._on_tool_changed("pencil")

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # ── File ──────────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")
        self._add_action(file_menu, "&New…", self._prompt_new, "Ctrl+N")
        self._add_action(file_menu, "&Open…", self.open_project, "Ctrl+O")
        file_menu.addSeparator()
        self._add_action(file_menu, "&Save", self.save_project, "Ctrl+S")
        self._add_action(file_menu, "Save &As…", self.save_as_project, "Ctrl+Shift+S")
        file_menu.addSeparator()
        self._add_action(file_menu, "E&xit", self.close, "Ctrl+Q")

        # ── Edit ──────────────────────────────────────────────────────
        edit_menu = mb.addMenu("&Edit")
        self._undo_action = self._add_action(edit_menu, "&Undo", self._undo, "Ctrl+Z")
        self._redo_action = self._add_action(edit_menu, "&Redo", self._redo, "Ctrl+Y")

        # ── View ──────────────────────────────────────────────────────
        view_menu = mb.addMenu("&View")
        self._add_action(view_menu, "Zoom &In", self._zoom_in, "Ctrl+=")
        self._add_action(view_menu, "Zoom &Out", self._zoom_out, "Ctrl+-")
        self._add_action(view_menu, "&Fit to Window", self._fit, "Ctrl+Shift+H")
        view_menu.addSeparator()
        self._grid_action = self._add_action(
            view_menu, "Show &Grid", self._toggle_grid, "Ctrl+G", checkable=True
        )
        self._grid_action.setChecked(True)

        # ── Layer ─────────────────────────────────────────────────────
        layer_menu = mb.addMenu("&Layer")
        self._add_action(layer_menu, "&Add Layer", self._add_layer, "Ctrl+Shift+N")
        self._add_action(layer_menu, "&Delete Layer", self._delete_layer)
        self._add_action(
            layer_menu, "D&uplicate Layer", self._duplicate_layer, "Ctrl+J"
        )
        layer_menu.addSeparator()
        self._add_action(layer_menu, "Merge &Down", self._merge_down, "Ctrl+E")
        self._add_action(layer_menu, "&Flatten Image", self._flatten)

        # ── Frame ─────────────────────────────────────────────────────
        frame_menu = mb.addMenu("Fr&ame")
        self._add_action(frame_menu, "&Add Frame", self._add_frame)
        self._add_action(frame_menu, "&Delete Frame", self._delete_frame)
        self._add_action(frame_menu, "D&uplicate Frame", self._duplicate_frame)

        # ── Animation ─────────────────────────────────────────────────
        anim_menu = mb.addMenu("&Animation")
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

        # ── Transform ─────────────────────────────────────────────────
        xform_menu = mb.addMenu("&Transform")
        self._add_action(xform_menu, "Flip &Horizontal", self._flip_h)
        self._add_action(xform_menu, "Flip &Vertical", self._flip_v)
        xform_menu.addSeparator()
        self._add_action(xform_menu, "Rotate &90° CW", lambda: self._rotate(90))
        self._add_action(xform_menu, "Rotate 90° &CCW", lambda: self._rotate(-90))
        self._add_action(xform_menu, "Rotate &180°", lambda: self._rotate(180))
        xform_menu.addSeparator()
        self._add_action(xform_menu, "&Canvas Size…", self._prompt_canvas_resize)
        self._add_action(xform_menu, "&Scale Image…", self._prompt_scale)
        xform_menu.addSeparator()
        self._add_action(xform_menu, "Shift / &Offset…", self._prompt_shift)
        self._add_action(xform_menu, "&Outline", self._apply_outline)
        self._add_action(xform_menu, "Replace &Color…", self._prompt_replace_color)
        xform_menu.addSeparator()
        self._add_action(
            xform_menu, "&Brightness / Contrast…", self._prompt_adjust_brightness
        )
        self._add_action(xform_menu, "H&ue / Saturation…", self._prompt_adjust_hue)

        # ── Help ──────────────────────────────────────────────────────
        help_menu = mb.addMenu("&Help")
        self._add_action(help_menu, "&About…", self._show_about)

    def _add_action(
        self,
        menu,
        text: str,
        slot,
        shortcut: Optional[str] = None,
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
        """Additional canvas-level keyboard shortcuts."""
        shortcuts = {
            "B": "pencil",
            "E": "eraser",
            "L": "line",
            "R": "rectangle",
            "O": "ellipse",
            "G": "fill",
            "I": "eyedropper",
            "S": "select",
            "M": "move",
            "T": "text",
        }
        from PyQt6.QtGui import QShortcut

        for key, tool_name in shortcuts.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(
                lambda n=tool_name: (
                    self._toolbar.select_tool(n) if self._toolbar else None
                )
            )

    def _init_status_bar(self) -> None:
        bar: QStatusBar = self.statusBar()
        bar.addWidget(QLabel("Cursor:"))
        bar.addWidget(self._status_cursor)
        bar.addWidget(QLabel("  Canvas:"))
        bar.addWidget(self._status_canvas)
        bar.addWidget(QLabel("  Zoom:"))
        bar.addWidget(self._status_zoom)
        bar.addWidget(QLabel("  Tool:"))
        bar.addWidget(self._status_tool)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_cursor_moved(self, x: int, y: int) -> None:
        self._status_cursor.setText(f"{x}, {y}")

    def _on_tool_changed(self, name: str) -> None:
        if self._canvas is None or self._sprite is None:
            return
        self._status_tool.setText(name)
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

    def _on_active_layer_changed(self, layer_idx: int) -> None:
        if self._canvas:
            self._canvas.active_layer = layer_idx
            if self._canvas._tool:
                self._canvas._tool.layer_index = layer_idx

    def _on_layers_modified(self) -> None:
        if self._canvas:
            self._canvas.invalidate_cache()
        if self._timeline:
            self._timeline.refresh()
        self._unsaved = True

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

    def _undo(self) -> None:
        if self._stack.can_undo:
            self._stack.undo()
            if self._canvas:
                self._canvas.invalidate_cache()
            if self._layers_panel:
                self._layers_panel.refresh()

    def _redo(self) -> None:
        if self._stack.can_redo:
            self._stack.redo()
            if self._canvas:
                self._canvas.invalidate_cache()
            if self._layers_panel:
                self._layers_panel.refresh()

    def _zoom_in(self) -> None:
        if self._canvas:
            self._canvas._zoom_step(1)

    def _zoom_out(self) -> None:
        if self._canvas:
            self._canvas._zoom_step(-1)

    def _fit(self) -> None:
        if self._canvas:
            self._canvas.fit_to_window()

    def _toggle_grid(self) -> None:
        if self._canvas:
            self._canvas.show_grid = self._grid_action.isChecked()
            self._canvas.update()

    def _add_layer(self) -> None:
        if self._layers_panel:
            self._layers_panel._add_layer()

    def _delete_layer(self) -> None:
        if self._layers_panel:
            self._layers_panel._remove_layer()

    def _duplicate_layer(self) -> None:
        if self._layers_panel:
            self._layers_panel._duplicate_layer()

    def _merge_down(self) -> None:
        if self._layers_panel:
            self._layers_panel._merge_down()

    def _flatten(self) -> None:
        if self._layers_panel:
            self._layers_panel._flatten()

    def _add_frame(self) -> None:
        if self._sprite is None:
            return
        cmd = AddFrameCommand(self._sprite)
        self._stack.push(cmd)
        if self._canvas:
            self._canvas.invalidate_cache()

    def _delete_frame(self) -> None:
        if self._sprite is None or self._sprite.frame_count <= 1:
            QMessageBox.warning(
                self, "Cannot Delete", "A sprite must have at least one frame."
            )
            return
        fi = self._canvas.active_frame if self._canvas else 0
        cmd = RemoveFrameCommand(self._sprite, fi)
        self._stack.push(cmd)
        if self._canvas:
            new_fi = max(0, fi - 1)
            self._canvas.active_frame = new_fi
            self._canvas.invalidate_cache()

    def _duplicate_frame(self) -> None:
        if self._sprite is None:
            return
        fi = self._canvas.active_frame if self._canvas else 0
        cmd = DuplicateFrameCommand(self._sprite, fi)
        self._stack.push(cmd)
        if self._canvas:
            self._canvas.active_frame = fi + 1
            self._canvas.invalidate_cache()

    def _on_timeline_frame_selected(self, frame_index: int) -> None:
        if self._canvas:
            self._canvas.active_frame = frame_index
            self._canvas.invalidate_cache()
            if self._canvas._tool:
                self._canvas._tool.frame_index = frame_index

    # ------------------------------------------------------------------
    # Animation menu actions
    # ------------------------------------------------------------------

    def _show_preview(self) -> None:
        if self._sprite is None:
            return
        if self._preview is None:
            self._preview = PreviewWindow(self._sprite, self)
        else:
            self._preview.set_sprite(self._sprite)
        self._preview.show()
        self._preview.raise_()

    def _set_loop_mode(self, mode: LoopMode) -> None:
        if self._sprite:
            self._sprite.animation.loop_mode = mode

    def _toggle_onion_skin(self) -> None:
        if self._canvas is None:
            return
        enabled = self._onion_action.isChecked()
        self._canvas.onion_before = 1 if enabled else 0
        self._canvas.onion_after = 1 if enabled else 0
        self._canvas.invalidate_cache()

    # ------------------------------------------------------------------
    # Transform menu actions
    # ------------------------------------------------------------------

    def _active_layer_frame(self):
        li = self._layers_panel.active_layer if self._layers_panel else 0
        fi = self._canvas.active_frame if self._canvas else 0
        return li, fi

    def _push_transform(self, cmd) -> None:
        self._stack.push(cmd)
        if self._canvas:
            self._canvas.invalidate_cache()
        self._unsaved = True

    def _flip_h(self) -> None:
        if self._sprite is None:
            return
        li, fi = self._active_layer_frame()
        self._push_transform(FlipCommand(self._sprite, li, fi, horizontal=True))

    def _flip_v(self) -> None:
        if self._sprite is None:
            return
        li, fi = self._active_layer_frame()
        self._push_transform(FlipCommand(self._sprite, li, fi, horizontal=False))

    def _rotate(self, angle: float) -> None:
        if self._sprite is None:
            return
        li, fi = self._active_layer_frame()
        self._push_transform(RotateCommand(self._sprite, li, fi, angle))

    def _prompt_canvas_resize(self) -> None:
        if self._sprite is None:
            return
        w, ok1 = QInputDialog.getInt(
            self, "Canvas Size", "New width (px):", self._sprite.width, 1, 4096
        )
        if not ok1:
            return
        h, ok2 = QInputDialog.getInt(
            self, "Canvas Size", "New height (px):", self._sprite.height, 1, 4096
        )
        if ok2:
            self._push_transform(CanvasResizeCommand(self._sprite, w, h))
            self._status_canvas.setText(f"{self._sprite.width}×{self._sprite.height}")

    def _prompt_scale(self) -> None:
        if self._sprite is None:
            return
        w, ok1 = QInputDialog.getInt(
            self, "Scale Image", "New width (px):", self._sprite.width, 1, 4096
        )
        if not ok1:
            return
        h, ok2 = QInputDialog.getInt(
            self, "Scale Image", "New height (px):", self._sprite.height, 1, 4096
        )
        if ok2:
            self._push_transform(ScaleCommand(self._sprite, w, h))
            self._status_canvas.setText(f"{self._sprite.width}×{self._sprite.height}")

    def _prompt_shift(self) -> None:
        if self._sprite is None:
            return
        dx, ok1 = QInputDialog.getInt(
            self,
            "Shift",
            "Horizontal offset (px):",
            0,
            -self._sprite.width,
            self._sprite.width,
        )
        if not ok1:
            return
        dy, ok2 = QInputDialog.getInt(
            self,
            "Shift",
            "Vertical offset (px):",
            0,
            -self._sprite.height,
            self._sprite.height,
        )
        if ok2:
            li, fi = self._active_layer_frame()
            self._push_transform(ShiftCommand(self._sprite, li, fi, dx, dy))

    def _apply_outline(self) -> None:
        if self._sprite is None:
            return
        li, fi = self._active_layer_frame()
        self._push_transform(OutlineCommand(self._sprite, li, fi))

    def _prompt_replace_color(self) -> None:
        if self._sprite is None:
            return
        QMessageBox.information(
            self,
            "Replace Color",
            "Replace Color is accessible programmatically via ReplaceColorCommand.",
        )

    def _prompt_adjust_brightness(self) -> None:
        if self._sprite is None:
            return
        val, ok = QInputDialog.getDouble(
            self, "Brightness", "Brightness factor (1.0 = no change):", 1.0, 0.0, 5.0, 2
        )
        if ok:
            li, fi = self._active_layer_frame()
            self._push_transform(
                AdjustmentCommand(self._sprite, li, fi, brightness=val)
            )

    def _prompt_adjust_hue(self) -> None:
        if self._sprite is None:
            return
        val, ok = QInputDialog.getDouble(
            self, "Hue / Saturation", "Hue rotation (degrees):", 0.0, -180.0, 180.0, 1
        )
        if ok:
            li, fi = self._active_layer_frame()
            self._push_transform(AdjustmentCommand(self._sprite, li, fi, hue=val))

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Spriter",
            "Spriter — Pixel art editor\n\nPhases 1-6 implemented.",
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
            "eyedropper": EyedropperTool,
            "select": RectSelectTool,
            "move": MoveTool,
            "text": TextTool,
        }
        cls = tools.get(name, PencilTool)
        return cls(self._sprite, self._stack)

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
