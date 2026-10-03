# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Typing base shared by the :class:`MainWindow` feature mixins.

The main window's slot methods are split across several mixin modules
(:mod:`main_window_edit`, :mod:`main_window_io`, :mod:`main_window_diffusion`).
Those mixins reference state and helper methods that live on the composed
:class:`~spriter.ui.main_window.MainWindow`.  :class:`_MainWindowProtocol`
exists purely so static analysis understands those shared members: at runtime
it is a bare ``object`` subclass, but for type checking it derives from
``QMainWindow`` and declares the shared attributes and cross-mixin helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QLabel, QMainWindow, QMenu

    from ..commands.base import CommandStack
    from ..core.settings import Settings
    from ..core.sprite import Sprite
    from .canvas import CanvasWidget
    from .color_picker import ColorPicker
    from .layers_panel import LayersPanel
    from .main_window_diffusion import _DiffusionWorker
    from .preview import PreviewWindow
    from .timeline import TimelinePanel
    from .toolbar import ToolBar

    _Base = QMainWindow
else:
    _Base = object


class _MainWindowProtocol(_Base):
    """Declares the shared state and helpers the slot mixins rely on.

    Only meaningful to the type checker; carries no behaviour at runtime.
    """

    if TYPE_CHECKING:
        # Shared state (set up by MainWindow.__init__ / _rebuild_ui).
        _sprite: Sprite | None
        _settings: Settings
        _stack: CommandStack
        _current_path: Path | None
        _title_name: str | None
        _unsaved: bool
        _canvas: CanvasWidget | None
        _toolbar: ToolBar | None
        _color_picker: ColorPicker | None
        _layers_panel: LayersPanel | None
        _timeline: TimelinePanel | None
        _preview: PreviewWindow | None
        _diffusion_worker: _DiffusionWorker | None
        _autosave_timer: QTimer
        _status_cursor: QLabel
        _status_canvas: QLabel
        _status_zoom: QLabel
        _status_tool: QLabel
        _status_layer: QLabel
        _grid_action: QAction
        _onion_action: QAction
        _undo_action: QAction
        _redo_action: QAction
        _recent_menu: QMenu

        # Cross-mixin helpers (implemented on MainWindow or a sibling mixin).
        def _rebuild_ui(self) -> None: ...
        def _refresh_undo_redo_labels(self) -> None: ...
        def _active_layer_frame(self) -> tuple[int, int]: ...
        def _dialog_dir(self, mode: str = ...) -> str: ...
        def _remember_path(self, path: str, mode: str = ...) -> None: ...
        def _remember_directory(self, directory: str, mode: str = ...) -> None: ...
        def _load_imported_sprite(self, sprite: Sprite, title_name: str) -> None: ...
