# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Palette & colour-picker UX improvement tests.

Features tested:
  A. Palette import/export wired into File menu
  B. 2-D SV square + vertical H strip (Aseprite-style gradient picker)
  C. 16-slot recent-colours row
  D. Palette button right-click context menu (Set / Delete)
"""

from __future__ import annotations

from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Phase A: Palette import/export in File menu
# ===========================================================================


class TestPaletteMenuWiring:
    def _get_submenu_actions(self, win, submenu_title: str):
        """Return action texts from a named sub-menu inside File menu."""
        mb = win.menuBar()
        file_menu = None
        for action in mb.actions():
            if "&File" in action.text() or "File" in action.text():
                file_menu = action.menu()
                break
        assert file_menu is not None, "File menu not found"
        for action in file_menu.actions():
            if action.menu() and submenu_title in action.text():
                return [a.text() for a in action.menu().actions()]
        return []

    def test_import_menu_has_palette_action(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        texts = self._get_submenu_actions(win, "Import")
        assert any(
            "Palette" in t for t in texts
        ), f"No 'Palette' action in Import menu. Found: {texts}"
        win.close()

    def test_export_menu_has_palette_action(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        texts = self._get_submenu_actions(win, "Export")
        assert any(
            "Palette" in t for t in texts
        ), f"No 'Palette' action in Export menu. Found: {texts}"
        win.close()


# ===========================================================================
# Phase B: Gradient picker — _HueStrip and _SVSquare
# ===========================================================================


class TestHueStrip:
    def test_hue_strip_emits_hue_changed_on_click(self, qapp):
        from PyQt6.QtCore import QPoint
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import Qt, QPointF

        from spriter.ui.color_picker import _HueStrip

        strip = _HueStrip()
        strip.show()

        received: List[int] = []
        strip.hue_changed.connect(received.append)

        # Click at top → hue near 0
        h = strip.height()
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(8, 0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        strip.mousePressEvent(event)
        assert len(received) == 1
        assert 0 <= received[0] <= 359

    def test_hue_strip_click_bottom_emits_high_hue(self, qapp):
        from PyQt6.QtCore import QPointF
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.color_picker import _HueStrip

        strip = _HueStrip()
        strip.show()

        received: List[int] = []
        strip.hue_changed.connect(received.append)

        h = strip.height()
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(8, h - 1),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        strip.mousePressEvent(event)
        assert len(received) == 1
        assert received[0] >= 300  # near 359

    def test_set_hue_updates_internal_state(self, qapp):
        from spriter.ui.color_picker import _HueStrip

        strip = _HueStrip()
        strip.set_hue(180)
        assert strip._hue == 180

    def test_drag_continues_emitting(self, qapp):
        from PyQt6.QtCore import QPointF
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.color_picker import _HueStrip

        strip = _HueStrip()
        strip.show()

        received: List[int] = []
        strip.hue_changed.connect(received.append)

        h = strip.height()
        for y in [0, h // 4, h // 2, h - 1]:
            event = QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPointF(8, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            strip.mouseMoveEvent(event)

        assert len(received) == 4


class TestSVSquare:
    def test_sv_square_emits_sv_changed_on_click(self, qapp):
        from PyQt6.QtCore import QPointF
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.color_picker import _SVSquare

        sq = _SVSquare()
        sq.show()

        received: List[Tuple[int, int]] = []
        sq.sv_changed.connect(lambda s, v: received.append((s, v)))

        # Click at top-right corner → high S, high V
        w, h = sq.width(), sq.height()
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(w - 1, 0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        sq.mousePressEvent(event)
        assert len(received) == 1
        s, v = received[0]
        assert s >= 240
        assert v >= 240

    def test_sv_square_bottom_left_low_sv(self, qapp):
        from PyQt6.QtCore import QPointF
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.color_picker import _SVSquare

        sq = _SVSquare()
        sq.show()

        received: List[Tuple[int, int]] = []
        sq.sv_changed.connect(lambda s, v: received.append((s, v)))

        # Click at bottom-left → low S, low V
        w, h = sq.width(), sq.height()
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(0, h - 1),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        sq.mousePressEvent(event)
        assert len(received) == 1
        s, v = received[0]
        assert s <= 15
        assert v <= 15

    def test_set_hue_updates_hue(self, qapp):
        from spriter.ui.color_picker import _SVSquare

        sq = _SVSquare()
        sq.set_hue(120)
        assert sq._hue == 120

    def test_set_sv_updates_state(self, qapp):
        from spriter.ui.color_picker import _SVSquare

        sq = _SVSquare()
        sq.set_sv(200, 100)
        assert sq._s == 200
        assert sq._v == 100


class TestGradientPickerIntegration:
    def test_color_picker_has_sv_square_and_hue_strip(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        assert hasattr(cp, "_sv_square")
        assert hasattr(cp, "_hue_strip")

    def test_hue_strip_click_syncs_h_slider(self, qapp):
        from PyQt6.QtCore import QPointF
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.show()

        h = cp._hue_strip.height()
        # Click at 50% height → should be around hue 180
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(8, h // 2),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        cp._hue_strip.mousePressEvent(event)
        # H slider should have updated to a mid-range value
        assert 100 <= cp._h_slider.value() <= 270

    def test_sv_click_syncs_s_v_sliders(self, qapp):
        from PyQt6.QtCore import QPointF
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.show()

        w = cp._sv_square.width()
        h = cp._sv_square.height()
        # Click at ~75% x, ~25% y → high S (~190), high V (~190)
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(w * 0.75, h * 0.25),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        cp._sv_square.mousePressEvent(event)
        assert cp._s_slider.value() >= 150
        assert cp._v_slider.value() >= 150

    def test_refresh_controls_syncs_sv_square_and_strip(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.foreground = (0, 200, 0, 255)  # hue ~120, high S/V
        # After setting FG, _sv_square and _hue_strip should be in sync
        assert cp._sv_square._hue == cp._h_slider.value()
        assert cp._hue_strip._hue == cp._h_slider.value()


# ===========================================================================
# Phase C: Recent colours row
# ===========================================================================


class TestRecentColors:
    def test_push_recent_appears_in_row(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.push_recent((255, 0, 0, 255))
        assert len(cp._recent_colors) >= 1
        assert cp._recent_colors[0] == (255, 0, 0, 255)

    def test_push_recent_capped_at_16(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        for i in range(20):
            cp.push_recent((i, 0, 0, 255))
        assert len(cp._recent_colors) == 16

    def test_push_recent_newest_first(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        red = (255, 0, 0, 255)
        blue = (0, 0, 255, 255)
        cp.push_recent(red)
        cp.push_recent(blue)
        assert cp._recent_colors[0] == blue
        assert cp._recent_colors[1] == red

    def test_push_recent_deduplicates(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        red = (255, 0, 0, 255)
        cp.push_recent(red)
        cp.push_recent((0, 255, 0, 255))
        cp.push_recent(red)  # push red again
        # red should appear only once (at front)
        assert cp._recent_colors.count(red) == 1
        assert cp._recent_colors[0] == red

    def test_recent_button_click_applies_fg(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        color = (100, 150, 200, 255)
        cp.push_recent(color)

        received: List[Tuple] = []
        cp.foreground_changed.connect(received.append)

        cp._recent_buttons[0].click()
        assert len(received) >= 1
        assert received[-1] == color

    def test_palette_click_pushes_recent(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        initial_count = len(cp._recent_colors)
        # Click first palette button
        cp._palette_buttons[0].click()
        assert len(cp._recent_colors) == initial_count + 1

    def test_hex_edit_pushes_recent(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp._hex_edit.setText("#ff8000ff")
        cp._on_hex_edited()
        assert len(cp._recent_colors) >= 1
        assert cp._recent_colors[0] == (255, 128, 0, 255)

    def test_color_picker_has_recent_buttons_list(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        assert hasattr(cp, "_recent_buttons")
        assert len(cp._recent_buttons) == 16


# ===========================================================================
# Phase D: Palette button right-click context menu
# ===========================================================================


class TestPaletteButtonContextMenu:
    def test_palette_button_is_palette_button_type(self, qapp):
        from spriter.ui.color_picker import ColorPicker, _PaletteButton

        cp = ColorPicker()
        assert isinstance(cp._palette_buttons[0], _PaletteButton)

    def test_palette_button_has_set_color_signal(self, qapp):
        from spriter.ui.color_picker import _PaletteButton

        btn = _PaletteButton(0, (255, 0, 0, 255))
        received: List[int] = []
        btn.set_color_requested.connect(received.append)
        btn.set_color_requested.emit(0)
        assert received == [0]

    def test_palette_button_has_delete_signal(self, qapp):
        from spriter.ui.color_picker import _PaletteButton

        btn = _PaletteButton(0, (255, 0, 0, 255))
        received: List[int] = []
        btn.delete_requested.connect(received.append)
        btn.delete_requested.emit(0)
        assert received == [0]

    def test_set_color_updates_palette_slot(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.foreground = (255, 0, 0, 255)
        initial_len = len(cp._palette_colors)

        # Trigger set_color on slot 0
        cp._on_palette_set_color(0)

        assert cp._palette_colors[0] == (255, 0, 0, 255)
        assert len(cp._palette_colors) == initial_len

    def test_delete_slot_removes_color(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        initial_len = len(cp._palette_colors)

        cp._on_palette_delete(0)

        assert len(cp._palette_colors) == initial_len - 1

    def test_context_menu_actions_contain_expected_labels(self, qapp):
        from PyQt6.QtCore import QPoint

        from spriter.ui.color_picker import _PaletteButton

        btn = _PaletteButton(0, (255, 0, 0, 255))
        btn.show()

        # Build the menu and inspect it without actually showing it
        menu = btn._build_context_menu()
        action_texts = [a.text() for a in menu.actions()]
        assert any(
            "Set" in t for t in action_texts
        ), f"Expected 'Set' action, got {action_texts}"
        assert any(
            "Delete" in t for t in action_texts
        ), f"Expected 'Delete' action, got {action_texts}"

    def test_set_color_via_signal_chain(self, qapp):
        """Clicking 'Set to Foreground Color' in the menu fires the right signal."""
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.foreground = (42, 43, 44, 255)

        received: List[int] = []
        cp._palette_buttons[2].set_color_requested.connect(received.append)
        cp._palette_buttons[2].set_color_requested.emit(2)

        assert received == [2]
        # Now verify the handler updates the color
        cp._on_palette_set_color(2)
        assert cp._palette_colors[2] == (42, 43, 44, 255)
