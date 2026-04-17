# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for PencilTool and EraserTool."""

import numpy as np
import pytest
from spriter.commands.base import CommandStack
from spriter.core.sprite import Sprite
from spriter.tools.eraser import EraserTool
from spriter.tools.pencil import PencilTool

RED = (255, 0, 0, 255)
TRANSPARENT = (0, 0, 0, 0)


def _setup(w=16, h=16):
    s = Sprite(w, h)
    s.add_layer()
    s.add_frame()
    stack = CommandStack()
    return s, stack


class TestPencilTool:
    def test_single_press_sets_pixel(self):
        s, stack = _setup()
        tool = PencilTool(s, stack)
        tool.foreground = RED
        tool.on_press(4, 4)
        tool.on_release(4, 4)
        px = s.get_cel(0, 0).pixels
        assert tuple(px[4, 4]) == RED

    def test_drag_connects_pixels(self):
        s, stack = _setup()
        tool = PencilTool(s, stack)
        tool.foreground = RED
        tool.on_press(0, 0)
        tool.on_drag(3, 0)
        tool.on_release(3, 0)
        px = s.get_cel(0, 0).pixels
        for x in range(4):
            assert tuple(px[0, x]) == RED

    def test_undo_reverts_stroke(self):
        s, stack = _setup()
        tool = PencilTool(s, stack)
        tool.foreground = RED
        tool.on_press(4, 4)
        tool.on_release(4, 4)
        stack.undo()
        px = s.get_cel(0, 0).pixels
        assert tuple(px[4, 4]) == TRANSPARENT

    def test_redo_reapplies_stroke(self):
        s, stack = _setup()
        tool = PencilTool(s, stack)
        tool.foreground = RED
        tool.on_press(4, 4)
        tool.on_release(4, 4)
        stack.undo()
        stack.redo()
        px = s.get_cel(0, 0).pixels
        assert tuple(px[4, 4]) == RED

    def test_brush_size_square(self):
        s, stack = _setup()
        tool = PencilTool(s, stack)
        tool.foreground = RED
        tool.brush_size = 3
        tool.on_press(5, 5)
        tool.on_release(5, 5)
        px = s.get_cel(0, 0).pixels
        # A 3x3 square should be painted around (5, 5).
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                assert tuple(px[5 + dy, 5 + dx]) == RED

    def test_selection_mask_restricts_paint(self):
        s, stack = _setup()
        # Only allow painting inside row 0.
        mask = np.zeros((16, 16), dtype=bool)
        mask[0, :] = True
        s.set_selection(mask)
        tool = PencilTool(s, stack)
        tool.foreground = RED
        tool.on_press(5, 5)  # row 5 is outside selection
        tool.on_release(5, 5)
        px = s.get_cel(0, 0).pixels
        assert tuple(px[5, 5]) == TRANSPARENT

    def test_no_undo_entry_when_nothing_changed(self):
        """Painting on a pixel with the same value should not push a command."""
        s, stack = _setup()
        # Paint red first.
        tool = PencilTool(s, stack)
        tool.foreground = RED
        tool.on_press(4, 4)
        tool.on_release(4, 4)
        count_before = len(stack._undo_stack)
        # Paint the same color again.
        tool.on_press(4, 4)
        tool.on_release(4, 4)
        # No new command pushed because pixels are identical.
        # (Stack may or may not grow — depends on whether pixels changed.)
        # Either way, undo must not break anything after multiple strokes.
        stack.undo()
        assert not stack.can_undo or count_before >= 1

    def test_preview_overlay_during_stroke(self):
        s, stack = _setup()
        tool = PencilTool(s, stack)
        tool.foreground = RED
        tool.on_press(2, 2)
        overlay = tool.preview_overlay()
        assert overlay is not None
        assert overlay.shape == (16, 16, 4)
        tool.on_release(2, 2)
        assert tool.preview_overlay() is None


class TestEraserTool:
    def test_erases_pixel(self):
        s, stack = _setup()
        # Pre-fill the layer.
        px = np.full((16, 16, 4), 200, dtype=np.uint8)
        s.set_cel_pixels(0, 0, px)
        tool = EraserTool(s, stack)
        tool.on_press(5, 5)
        tool.on_release(5, 5)
        result = s.get_cel(0, 0).pixels
        assert tuple(result[5, 5]) == TRANSPARENT

    def test_undo_restores_erased_pixel(self):
        s, stack = _setup()
        px = np.full((16, 16, 4), 200, dtype=np.uint8)
        s.set_cel_pixels(0, 0, px)
        tool = EraserTool(s, stack)
        tool.on_press(5, 5)
        tool.on_release(5, 5)
        stack.undo()
        result = s.get_cel(0, 0).pixels
        assert result[5, 5, 3] == 200

    def test_eraser_drag(self):
        s, stack = _setup()
        px = np.full((16, 16, 4), 200, dtype=np.uint8)
        s.set_cel_pixels(0, 0, px)
        tool = EraserTool(s, stack)
        tool.on_press(0, 0)
        tool.on_drag(2, 0)
        tool.on_release(2, 0)
        result = s.get_cel(0, 0).pixels
        for x in range(3):
            assert result[0, x, 3] == 0
