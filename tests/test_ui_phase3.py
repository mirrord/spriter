# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Phase 3 UI smoke tests — widget instantiation and signal/slot wiring.

These tests use the ``qapp`` fixture from ``conftest.py``  (session-scoped
:class:`~PyQt6.QtWidgets.QApplication` with offscreen rendering).

All tests are lightweight: they verify that widgets can be created without
errors, that the expected signals exist, and that core signal→slot wiring
functions correctly.  No pixels are rendered.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# CanvasWidget
# ---------------------------------------------------------------------------


class TestCanvasWidget:
    def test_instantiation(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.ui.canvas import CanvasWidget

        s = Sprite(16, 16)
        s.add_layer()
        s.add_frame()
        widget = CanvasWidget(s, CommandStack())
        assert widget is not None

    def test_default_zoom_is_one(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.ui.canvas import CanvasWidget

        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        widget = CanvasWidget(s, CommandStack())
        assert widget.zoom == 1.0

    def test_zoom_clamped_to_max(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.ui.canvas import CanvasWidget

        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        widget = CanvasWidget(s, CommandStack())
        widget.zoom = 99999.0
        assert widget.zoom == float(CanvasWidget.ZOOM_LEVELS[-1])

    def test_zoom_clamped_to_min(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.ui.canvas import CanvasWidget

        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        widget = CanvasWidget(s, CommandStack())
        widget.zoom = 0.0
        assert widget.zoom == float(CanvasWidget.ZOOM_LEVELS[0])

    def test_zoom_changed_signal(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.ui.canvas import CanvasWidget

        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        widget = CanvasWidget(s, CommandStack())
        received = []
        widget.zoom_changed.connect(received.append)
        widget.zoom = 2.0
        assert received == [2.0]

    def test_cursor_moved_signal_exists(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.ui.canvas import CanvasWidget

        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        widget = CanvasWidget(s, CommandStack())
        assert hasattr(widget, "cursor_moved")

    def test_set_tool(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.tools.pencil import PencilTool
        from spriter.ui.canvas import CanvasWidget

        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        widget = CanvasWidget(s, CommandStack())
        tool = PencilTool(s, CommandStack())
        widget.set_tool(tool)
        assert widget._tool is tool

    def test_invalidate_cache(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.ui.canvas import CanvasWidget

        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        widget = CanvasWidget(s, CommandStack())
        # Pre-warm cache
        _ = widget._get_composite()
        assert widget._composite_cache is not None
        widget.invalidate_cache()
        assert widget._composite_cache is None

    def test_fit_to_window(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.ui.canvas import CanvasWidget

        s = Sprite(32, 32)
        s.add_layer()
        s.add_frame()
        widget = CanvasWidget(s, CommandStack())
        widget.resize(200, 200)
        widget.fit_to_window()
        # After fit, zoom should be > 1.0 since widget > canvas.
        assert widget.zoom > 1.0

    def test_show_grid_default_true(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.ui.canvas import CanvasWidget

        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        widget = CanvasWidget(s, CommandStack())
        assert widget.show_grid is True


# ---------------------------------------------------------------------------
# ToolBar
# ---------------------------------------------------------------------------


class TestToolBar:
    def test_instantiation(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        assert tb is not None

    def test_default_tool_is_pencil(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        assert tb.current_tool == "pencil"

    def test_all_tool_buttons_present(self, qapp):
        from spriter.ui.toolbar import ToolBar, _TOOLS

        tb = ToolBar()
        for name, _ in _TOOLS:
            assert name in tb._buttons

    def test_tool_changed_signal_emitted(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        received = []
        tb.tool_changed.connect(received.append)
        tb.select_tool("eraser")
        assert received == ["eraser"]

    def test_select_tool_updates_current(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        tb.select_tool("line")
        assert tb.current_tool == "line"

    def test_brush_size_default(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        assert tb.brush_size == 1

    def test_opacity_default(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        assert tb.opacity == 255

    def test_tolerance_default(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        assert tb.tolerance == 32

    def test_brush_size_signal(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        received = []
        tb.brush_size_changed.connect(received.append)
        tb._brush_spin.setValue(5)
        assert 5 in received

    def test_unknown_tool_raises(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        with pytest.raises(KeyError):
            tb.select_tool("nonexistent_tool")


# ---------------------------------------------------------------------------
# ColorPicker
# ---------------------------------------------------------------------------


class TestColorPicker:
    def test_instantiation(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        assert cp is not None

    def test_default_foreground(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        assert cp.foreground == (0, 0, 0, 255)

    def test_default_background(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        assert cp.background == (255, 255, 255, 255)

    def test_set_foreground_emits_signal(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        received = []
        cp.foreground_changed.connect(received.append)
        cp.foreground = (255, 0, 0, 255)
        assert received[-1] == (255, 0, 0, 255)

    def test_set_background_emits_signal(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        received = []
        cp.background_changed.connect(received.append)
        cp.background = (0, 0, 255, 255)
        assert received[-1] == (0, 0, 255, 255)

    def test_set_foreground_roundtrip(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.foreground = (128, 64, 32, 200)
        assert cp.foreground == (128, 64, 32, 200)

    def test_set_background_roundtrip(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.background = (10, 20, 30, 100)
        assert cp.background == (10, 20, 30, 100)

    def test_palette_buttons_present(self, qapp):
        from spriter.ui.color_picker import ColorPicker, _DEFAULT_PALETTE

        cp = ColorPicker()
        assert len(cp._palette_buttons) == len(_DEFAULT_PALETTE)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------


class TestMainWindow:
    def test_instantiation(self, qapp):
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert window is not None
        window.close()

    def test_window_title(self, qapp):
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert "Spriter" in window.windowTitle()
        window.close()

    def test_has_canvas(self, qapp):
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert window._canvas is not None
        window.close()

    def test_has_toolbar(self, qapp):
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert window._toolbar is not None
        window.close()

    def test_has_color_picker(self, qapp):
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert window._color_picker is not None
        window.close()

    def test_has_layers_panel(self, qapp):
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert window._layers_panel is not None
        window.close()

    def test_new_project_default_size(self, qapp):
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert window._sprite is not None
        assert window._sprite.width == 32
        assert window._sprite.height == 32
        window.close()

    def test_new_project_custom_size(self, qapp):
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        window.new_project(64, 48)
        assert window._sprite is not None
        assert window._sprite.width == 64
        assert window._sprite.height == 48
        window.close()

    def test_undo_action_shortcut(self, qapp):
        from PyQt6.QtGui import QKeySequence
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert window._undo_action.shortcut() == QKeySequence("Ctrl+Z")
        window.close()

    def test_redo_action_shortcut(self, qapp):
        from PyQt6.QtGui import QKeySequence
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert window._redo_action.shortcut() == QKeySequence("Ctrl+Y")
        window.close()

    def test_grid_action_initially_checked(self, qapp):
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert window._grid_action.isChecked()
        window.close()

    def test_undo_redo_cycle(self, qapp):
        """Pushing a command and undoing it via the main window should work."""
        from spriter.ui.main_window import MainWindow

        window = MainWindow()
        assert window._sprite is not None
        initial_count = window._sprite.layer_count
        # Add a layer via the layers panel
        window._add_layer()
        assert window._sprite.layer_count == initial_count + 1
        window._undo()
        assert window._sprite.layer_count == initial_count
        window._redo()
        assert window._sprite.layer_count == initial_count + 1
        # Clear unsaved flag so closeEvent doesn't show a blocking dialog.
        window._unsaved = False
        window.close()
