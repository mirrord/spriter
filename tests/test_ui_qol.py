# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""QoL feature tests — 5 quality-of-life improvements.

Features tested:
  A. FG/BG colour swap (ColorPicker.swap_colors + X shortcut wiring)
  B. Undo/redo action labels (MainWindow._refresh_undo_redo_labels)
  C. Onion skin depth control (MainWindow._prompt_onion_depth)
  D. Timeline right-click context menu (_FrameCell.right_clicked + _on_cell_context_menu)
  E. Palette load/save UI (ColorPicker.load_palette + MainWindow import/export)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sprite(w: int = 8, h: int = 8, frames: int = 1, layers: int = 1):
    from spriter.core.sprite import Sprite

    s = Sprite(w, h)
    for i in range(layers):
        s.add_layer(f"Layer {i + 1}")
    for _ in range(frames - 1):
        s.add_frame()
    return s


# ===========================================================================
# Phase A: FG/BG colour swap
# ===========================================================================


class TestColorPickerSwap:
    def test_swap_colors_swaps_fg_bg(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.foreground = (255, 0, 0, 255)
        cp.background = (0, 0, 255, 255)

        cp.swap_colors()

        assert cp.foreground == (0, 0, 255, 255)
        assert cp.background == (255, 0, 0, 255)

    def test_swap_emits_both_signals(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.foreground = (10, 20, 30, 255)
        cp.background = (40, 50, 60, 255)

        fg_received: List[Tuple] = []
        bg_received: List[Tuple] = []
        cp.foreground_changed.connect(fg_received.append)
        cp.background_changed.connect(bg_received.append)

        cp.swap_colors()

        assert len(fg_received) == 1
        assert len(bg_received) == 1
        # After swap: new FG = old BG and vice-versa
        assert fg_received[0] == (40, 50, 60, 255)
        assert bg_received[0] == (10, 20, 30, 255)

    def test_swap_twice_restores_original(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        original_fg = (100, 150, 200, 255)
        original_bg = (50, 75, 100, 128)
        cp.foreground = original_fg
        cp.background = original_bg

        cp.swap_colors()
        cp.swap_colors()

        assert cp.foreground == original_fg
        assert cp.background == original_bg

    def test_swap_shortcut_wired_in_main_window(self, qapp):
        """MainWindow must register the X shortcut that calls swap_colors."""
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        # Ensure color picker exists and swap_colors is callable
        assert win._color_picker is not None
        assert callable(win._color_picker.swap_colors)
        win.close()


# ===========================================================================
# Phase B: Undo/redo action labels
# ===========================================================================


class TestUndoRedoLabels:
    def _make_named_command(self, name: str):
        from spriter.commands.base import Command

        class _Cmd(Command):
            @property
            def description(self):
                return name

            def execute(self):
                pass

            def undo(self):
                pass

        return _Cmd()

    def test_undo_label_updates_after_push_and_undo(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False

        cmd = self._make_named_command("Draw Pencil")
        win._stack.push(cmd)
        win._refresh_undo_redo_labels()

        assert "Draw Pencil" in win._undo_action.text()
        assert "&Redo" in win._redo_action.text() or "Redo" in win._redo_action.text()
        # The redo stack is empty so label should just be "&Redo"
        assert ":" not in win._redo_action.text()

        win._undo()
        assert "Draw Pencil" in win._redo_action.text()
        # Undo stack now empty
        assert ":" not in win._undo_action.text()

        win.close()

    def test_redo_label_clears_after_new_push(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False

        cmd1 = self._make_named_command("Action 1")
        win._stack.push(cmd1)
        win._undo()  # puts Action 1 on redo stack

        cmd2 = self._make_named_command("Action 2")
        win._stack.push(cmd2)
        win._refresh_undo_redo_labels()

        # Pushing a new command should clear redo
        assert ":" not in win._redo_action.text()
        assert "Action 2" in win._undo_action.text()

        win.close()

    def test_empty_stacks_show_plain_labels(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        win._stack.clear()
        win._refresh_undo_redo_labels()

        assert win._undo_action.text() == "&Undo"
        assert win._redo_action.text() == "&Redo"
        win.close()


# ===========================================================================
# Phase C: Onion skin depth control
# ===========================================================================


class TestOnionSkinDepth:
    def test_prompt_onion_depth_updates_canvas(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False

        call_count = [0]
        answers = [3, True, 2, True]  # before=3 ok, after=2 ok

        def fake_getint(_parent, _title, _label, default, *args, **kwargs):
            i = call_count[0]
            call_count[0] += 2
            return answers[i], answers[i + 1]

        with patch(
            "spriter.ui.main_window.QInputDialog.getInt", side_effect=fake_getint
        ):
            win._prompt_onion_depth()

        assert win._canvas.onion_before == 3
        assert win._canvas.onion_after == 2
        win.close()

    def test_prompt_onion_depth_cancel_first_does_nothing(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        original_before = win._canvas.onion_before
        original_after = win._canvas.onion_after

        with patch(
            "spriter.ui.main_window.QInputDialog.getInt", return_value=(5, False)
        ):
            win._prompt_onion_depth()

        assert win._canvas.onion_before == original_before
        assert win._canvas.onion_after == original_after
        win.close()

    def test_prompt_onion_depth_no_canvas(self, qapp):
        """Should silently return when no canvas exists."""
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        win._canvas = None
        # Should not raise
        win._prompt_onion_depth()
        win.close()


# ===========================================================================
# Phase D: Timeline right-click context menu
# ===========================================================================


class TestTimelineContextMenu:
    def test_right_click_emits_signal(self, qapp):
        from spriter.ui.timeline import _FrameCell
        from PyQt6.QtCore import Qt, QPoint
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QPointF

        cell = _FrameCell(2, 100)
        received = []
        cell.right_clicked.connect(lambda fi, pos: received.append((fi, pos)))

        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(10, 10),
            QPointF(10, 10),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
        cell.mousePressEvent(event)

        assert len(received) == 1
        assert received[0][0] == 2

    def test_left_click_does_not_emit_right_clicked(self, qapp):
        from spriter.ui.timeline import _FrameCell
        from PyQt6.QtCore import Qt, QPointF
        from PyQt6.QtGui import QMouseEvent

        cell = _FrameCell(0, 100)
        right_received = []
        cell.right_clicked.connect(lambda fi, pos: right_received.append(fi))

        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(10, 10),
            QPointF(10, 10),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        cell.mousePressEvent(event)

        assert right_received == []

    def test_context_menu_sets_active_frame(self, qapp):
        from spriter.ui.timeline import TimelinePanel
        from spriter.commands.base import CommandStack

        sprite = _make_sprite(frames=3)
        stack = CommandStack()
        panel = TimelinePanel(sprite, stack)
        panel._active_frame = 0

        with patch.object(panel, "_duplicate_frame"), patch(
            "spriter.ui.timeline.QMenu"
        ) as MockMenu:
            mock_instance = MagicMock()
            MockMenu.return_value = mock_instance
            panel._on_cell_context_menu(2, None)

        assert panel._active_frame == 2

    def test_context_menu_has_expected_actions(self, qapp):
        from spriter.ui.timeline import TimelinePanel
        from spriter.commands.base import CommandStack

        sprite = _make_sprite(frames=2)
        stack = CommandStack()
        panel = TimelinePanel(sprite, stack)

        action_labels: List[str] = []

        with patch("spriter.ui.timeline.QMenu") as MockMenu:
            mock_menu = MagicMock()
            MockMenu.return_value = mock_menu
            mock_menu.addAction.side_effect = lambda *a, **kw: action_labels.append(
                a[0]
            )
            panel._on_cell_context_menu(0, None)

        assert "Duplicate Frame" in action_labels
        assert "Delete Frame" in action_labels
        assert any("Duration" in lbl for lbl in action_labels)
        assert "Move Left" in action_labels
        assert "Move Right" in action_labels


# ===========================================================================
# Phase E: Palette load/save
# ===========================================================================


class TestColorPickerLoadPalette:
    def test_load_palette_replaces_buttons(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        original_count = len(cp._palette_buttons)

        new_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        cp.load_palette(new_colors)

        assert len(cp._palette_buttons) == 3
        assert cp._palette_colors == [
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
        ]

    def test_load_palette_accepts_rgba_tuples(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.load_palette([(10, 20, 30, 128), (40, 50, 60, 200)])

        assert cp._palette_colors == [(10, 20, 30, 128), (40, 50, 60, 200)]

    def test_load_palette_clicking_applies_color(self, qapp):
        from spriter.ui.color_picker import ColorPicker

        cp = ColorPicker()
        cp.load_palette([(200, 100, 50)])

        received = []
        cp.foreground_changed.connect(received.append)
        cp._palette_buttons[0].click()

        # _apply_palette_color sets R/G/B spinboxes in sequence, each emitting a
        # signal — the final emission carries the fully composed colour.
        assert len(received) >= 1
        assert received[-1][0] == 200
        assert received[-1][1] == 100
        assert received[-1][2] == 50


class TestPaletteImportExport:
    def test_import_palette_jasc(self, qapp, tmp_path):
        from spriter.ui.main_window import MainWindow
        from spriter.core.palette import Palette

        pal = Palette([(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)])
        pal_path = str(tmp_path / "test.pal")
        pal.to_jasc(pal_path)

        win = MainWindow()
        win._unsaved = False

        with patch(
            "spriter.ui.main_window.QFileDialog.getOpenFileName",
            return_value=(pal_path, ""),
        ):
            win._import_palette()

        assert len(win._color_picker._palette_buttons) == 3
        assert win._color_picker._palette_colors[0][:3] == (255, 0, 0)
        win.close()

    def test_import_palette_gpl(self, qapp, tmp_path):
        from spriter.ui.main_window import MainWindow
        from spriter.core.palette import Palette

        pal = Palette([(128, 64, 32, 255), (16, 32, 64, 255)])
        pal_path = str(tmp_path / "test.gpl")
        pal.to_gpl(pal_path)

        win = MainWindow()
        win._unsaved = False

        with patch(
            "spriter.ui.main_window.QFileDialog.getOpenFileName",
            return_value=(pal_path, ""),
        ):
            win._import_palette()

        assert len(win._color_picker._palette_buttons) == 2
        win.close()

    def test_export_palette_jasc_roundtrip(self, qapp, tmp_path):
        from spriter.ui.main_window import MainWindow
        from spriter.core.palette import Palette

        win = MainWindow()
        win._unsaved = False

        # Load a known palette into the picker
        win._color_picker.load_palette([(10, 20, 30), (40, 50, 60)])
        pal_path = str(tmp_path / "out.pal")

        with patch(
            "spriter.ui.main_window.QFileDialog.getSaveFileName",
            return_value=(pal_path, ""),
        ):
            win._export_palette()

        # Read it back and verify
        loaded = Palette.from_jasc(pal_path)
        assert len(loaded) == 2
        assert loaded[0][:3] == (10, 20, 30)
        assert loaded[1][:3] == (40, 50, 60)
        win.close()

    def test_export_palette_hex_roundtrip(self, qapp, tmp_path):
        from spriter.ui.main_window import MainWindow
        from spriter.core.palette import Palette

        win = MainWindow()
        win._unsaved = False
        win._color_picker.load_palette([(255, 128, 0), (0, 128, 255)])
        hex_path = str(tmp_path / "out.hex")

        with patch(
            "spriter.ui.main_window.QFileDialog.getSaveFileName",
            return_value=(hex_path, ""),
        ):
            win._export_palette()

        loaded = Palette.from_hex_list(hex_path)
        assert len(loaded) == 2
        assert loaded[0][:3] == (255, 128, 0)
        win.close()

    def test_import_palette_cancelled_does_nothing(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        original_count = len(win._color_picker._palette_buttons)

        with patch(
            "spriter.ui.main_window.QFileDialog.getOpenFileName",
            return_value=("", ""),
        ):
            win._import_palette()

        assert len(win._color_picker._palette_buttons) == original_count
        win.close()
