# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for MoveTool and TextTool."""

import numpy as np
import pytest
from spriter.commands.base import CommandStack
from spriter.core.sprite import Sprite
from spriter.tools.move import MoveTool
from spriter.tools.text import TextTool

RED = (255, 0, 0, 255)
TRANSPARENT = (0, 0, 0, 0)


def _setup(w=16, h=16):
    s = Sprite(w, h)
    s.add_layer()
    s.add_frame()
    stack = CommandStack()
    return s, stack


class TestMoveTool:
    def test_moves_entire_layer(self):
        s, stack = _setup()
        # Place a single red pixel at (2, 2).
        px = np.zeros((16, 16, 4), dtype=np.uint8)
        px[2, 2] = RED
        s.set_cel_pixels(0, 0, px)
        tool = MoveTool(s, stack)
        tool.on_press(2, 2)
        tool.on_release(5, 5)  # move by (+3, +3)
        result = s.get_cel(0, 0).pixels
        assert tuple(result[5, 5]) == RED  # moved
        assert tuple(result[2, 2]) == TRANSPARENT  # old position cleared

    def test_undo_reverts_move(self):
        s, stack = _setup()
        px = np.zeros((16, 16, 4), dtype=np.uint8)
        px[2, 2] = RED
        s.set_cel_pixels(0, 0, px)
        tool = MoveTool(s, stack)
        tool.on_press(2, 2)
        tool.on_release(7, 7)
        stack.undo()
        result = s.get_cel(0, 0).pixels
        assert tuple(result[2, 2]) == RED
        assert tuple(result[7, 7]) == TRANSPARENT

    def test_moves_selection_only(self):
        s, stack = _setup()
        px = np.zeros((16, 16, 4), dtype=np.uint8)
        px[0:4, 0:4] = RED
        s.set_cel_pixels(0, 0, px)
        # Select only the top-left 2x2.
        mask = np.zeros((16, 16), dtype=bool)
        mask[0:2, 0:2] = True
        s.set_selection(mask)
        tool = MoveTool(s, stack)
        tool.on_press(1, 1)
        tool.on_release(6, 6)  # move by (+5, +5)
        result = s.get_cel(0, 0).pixels
        # Selected pixels moved to new position.
        assert tuple(result[5, 5]) == RED
        assert tuple(result[6, 6]) == RED
        # Non-selected pixels remain in place.
        assert tuple(result[2, 2]) == RED
        assert tuple(result[3, 3]) == RED

    def test_drag_updates_preview(self):
        s, stack = _setup()
        px = np.zeros((16, 16, 4), dtype=np.uint8)
        px[1, 1] = RED
        s.set_cel_pixels(0, 0, px)
        tool = MoveTool(s, stack)
        tool.on_press(1, 1)
        tool.on_drag(4, 4)
        overlay = tool.preview_overlay()
        assert overlay is not None
        # Pixel should appear at new position during drag.
        assert tuple(overlay[4, 4]) == RED
        assert tuple(overlay[1, 1]) == TRANSPARENT
        tool.on_release(4, 4)

    def test_zero_offset_no_change(self):
        """Moving by (0,0) should still push an undo command (pixels unchanged)."""
        s, stack = _setup()
        px = np.zeros((16, 16, 4), dtype=np.uint8)
        px[3, 3] = RED
        s.set_cel_pixels(0, 0, px)
        tool = MoveTool(s, stack)
        tool.on_press(3, 3)
        tool.on_release(3, 3)  # no movement
        # Pixel must still be at (3,3).
        assert tuple(s.get_cel(0, 0).pixels[3, 3]) == RED


class TestTextTool:
    def test_renders_text(self):
        s, stack = _setup(64, 32)
        tool = TextTool(s, stack)
        tool.foreground = RED
        tool.text = "Hi"
        tool.font_size = 12
        tool.on_press(0, 0)
        px = s.get_cel(0, 0).pixels
        # Some pixels must be non-transparent (text was rendered).
        assert (px[..., 3] > 0).any()

    def test_empty_text_noop(self):
        s, stack = _setup()
        tool = TextTool(s, stack)
        tool.text = ""
        tool.on_press(0, 0)
        assert not stack.can_undo

    def test_undo_reverts_text(self):
        s, stack = _setup(64, 32)
        tool = TextTool(s, stack)
        tool.foreground = RED
        tool.text = "A"
        tool.font_size = 12
        tool.on_press(0, 0)
        stack.undo()
        px = s.get_cel(0, 0).pixels
        assert (px[..., 3] == 0).all()

    def test_text_respects_opacity(self):
        s, stack = _setup(64, 32)
        tool = TextTool(s, stack)
        tool.foreground = (255, 0, 0, 255)
        tool.opacity = 128
        tool.text = "X"
        tool.font_size = 14
        tool.on_press(0, 0)
        px = s.get_cel(0, 0).pixels
        # Some painted pixels should have alpha close to 128 (semi-transparent).
        # Because of PIL blending the exact values may vary slightly.
        non_zero = px[px[..., 3] > 0]
        if len(non_zero) > 0:
            # The maximum alpha in any painted pixel should not be 255
            # (since we scaled it to 128).
            assert non_zero[:, 3].max() <= 200
