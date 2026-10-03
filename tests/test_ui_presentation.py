# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""UI presentation-polish tests.

Covers three small feedback improvements:
  A. Unsaved-state indicator (``*``) in the window title.
  B. Keyboard-shortcut hints in toolbar button tooltips.
  C. Zoom-percentage label rendered on the canvas.
"""

from __future__ import annotations

# ===========================================================================
# A. Unsaved indicator in window title
# ===========================================================================


class TestWindowTitleUnsavedIndicator:
    def test_new_project_title_has_no_asterisk(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        try:
            assert "Spriter" in win.windowTitle()
            assert "*" not in win.windowTitle()
        finally:
            win._unsaved = False
            win.close()

    def test_edit_marks_title_with_asterisk(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        try:
            win._unsaved = True
            assert "*" in win.windowTitle()
        finally:
            win._unsaved = False
            win.close()

    def test_clearing_unsaved_removes_asterisk(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        try:
            win._unsaved = True
            assert "*" in win.windowTitle()
            win._unsaved = False
            assert "*" not in win.windowTitle()
        finally:
            win._unsaved = False
            win.close()

    def test_save_updates_title_with_name_and_no_asterisk(self, qapp, tmp_path):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        try:
            win._unsaved = True
            path = tmp_path / "hero.spriter"
            assert win._do_save(path) is True
            assert "hero.spriter" in win.windowTitle()
            assert "*" not in win.windowTitle()
        finally:
            win._unsaved = False
            win.close()


# ===========================================================================
# B. Shortcut hints in toolbar tooltips
# ===========================================================================


class TestToolbarShortcutTooltips:
    def test_pencil_tooltip_includes_shortcut(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        tip = tb._buttons["pencil"].toolTip()
        assert "Pencil" in tip
        assert "(B)" in tip

    def test_every_tool_tooltip_includes_a_shortcut(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        for name, btn in tb._buttons.items():
            tip = btn.toolTip()
            assert "(" in tip and ")" in tip, f"{name} tooltip missing shortcut: {tip}"


# ===========================================================================
# C. Zoom label on canvas
# ===========================================================================


class TestCanvasZoomLabel:
    def _make_canvas(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.sprite import Sprite
        from spriter.ui.canvas import CanvasWidget

        sprite = Sprite(8, 8)
        sprite.add_layer("Layer 1")
        sprite.add_frame()
        return CanvasWidget(sprite, CommandStack())

    def test_zoom_label_reflects_scale(self, qapp):
        canvas = self._make_canvas(qapp)
        canvas.zoom = 8
        assert canvas._zoom_label_text() == "800%"

    def test_zoom_label_at_unit_scale(self, qapp):
        canvas = self._make_canvas(qapp)
        canvas.zoom = 1
        assert canvas._zoom_label_text() == "100%"


# ===========================================================================
# D. Status-bar tool label with shortcut
# ===========================================================================


class TestStatusBarToolLabel:
    def test_tool_label_includes_name_and_shortcut(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        try:
            win._on_tool_changed("eraser")
            text = win._status_tool.text()
            assert "Eraser" in text
            assert "(E)" in text
        finally:
            win._unsaved = False
            win.close()


# ===========================================================================
# E. Status-bar active-layer badge
# ===========================================================================


class TestStatusBarLayerBadge:
    def test_layer_badge_shows_active_layer_name(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        try:
            assert "Layer 1" in win._status_layer.text()
        finally:
            win._unsaved = False
            win.close()


# ===========================================================================
# F. Toolbar view-control buttons (grid / fit / center)
# ===========================================================================


class TestToolbarViewControls:
    def test_toolbar_exposes_view_control_signals(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        assert hasattr(tb, "grid_toggled")
        assert hasattr(tb, "fit_requested")
        assert hasattr(tb, "center_requested")

    def test_grid_button_emits_state(self, qapp):
        from spriter.ui.toolbar import ToolBar

        tb = ToolBar()
        received: list[bool] = []
        tb.grid_toggled.connect(received.append)
        tb._grid_button.setChecked(False)
        tb._grid_button.clicked.emit(False)
        assert received == [False]


# ===========================================================================
# G. Color-picker swatch hex tooltip
# ===========================================================================


class TestColorSwatchTooltip:
    def test_swatch_tooltip_shows_hex(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.foreground = (255, 0, 0, 255)
        assert "ff0000" in cp._fg_swatch.toolTip().lower()
