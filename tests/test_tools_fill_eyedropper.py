# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for FillTool and EyedropperTool."""

import numpy as np
import pytest
from spriter.commands.base import CommandStack
from spriter.core.sprite import Sprite
from spriter.tools.eyedropper import EyedropperTool
from spriter.tools.fill import FillTool
from spriter.utils.geometry import draw_rect

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
TRANSPARENT = (0, 0, 0, 0)


def _setup(w=16, h=16):
    s = Sprite(w, h)
    s.add_layer()
    s.add_frame()
    stack = CommandStack()
    return s, stack


class TestFillTool:
    def test_fills_blank_canvas(self):
        s, stack = _setup()
        tool = FillTool(s, stack)
        tool.foreground = RED
        tool.on_press(0, 0)
        px = s.get_cel(0, 0).pixels
        assert (px == np.array(RED, dtype=np.uint8)).all()

    def test_does_not_cross_border(self):
        s, stack = _setup()
        # Draw a red 6×6 box outline, then fill interior with green.
        px_init = np.zeros((16, 16, 4), dtype=np.uint8)
        draw_rect(px_init, 2, 2, 6, 6, RED, filled=False)
        s.set_cel_pixels(0, 0, px_init)
        tool = FillTool(s, stack)
        tool.foreground = GREEN
        tool.on_press(4, 4)
        px = s.get_cel(0, 0).pixels
        assert tuple(px[4, 4]) == GREEN
        # Outer border remains red.
        assert tuple(px[2, 2]) == RED
        # Outside remains transparent.
        assert tuple(px[0, 0]) == TRANSPARENT

    def test_undo_reverts_fill(self):
        s, stack = _setup()
        tool = FillTool(s, stack)
        tool.foreground = RED
        tool.on_press(0, 0)
        stack.undo()
        px = s.get_cel(0, 0).pixels
        assert tuple(px[0, 0]) == TRANSPARENT

    def test_tolerance_fill(self):
        s, stack = _setup(8, 8)
        # Paint a gradient-like stripe of near-red colors.
        px_init = np.zeros((8, 8, 4), dtype=np.uint8)
        for x in range(4):
            px_init[0, x] = (250, 0, 0, 255)  # near-red
        for x in range(4, 8):
            px_init[0, x] = (0, 255, 0, 255)  # green — outside tolerance
        s.set_cel_pixels(0, 0, px_init)
        tool = FillTool(s, stack)
        tool.foreground = RED
        tool.tolerance = 20
        tool.on_press(0, 0)  # seed on (250,0,0,255)
        px = s.get_cel(0, 0).pixels
        for x in range(4):
            assert tuple(px[0, x]) == RED  # near-red pixels replaced
        assert tuple(px[0, 4]) == (0, 255, 0, 255)  # green untouched

    def test_same_color_noop(self):
        s, stack = _setup()
        tool = FillTool(s, stack)
        tool.foreground = TRANSPARENT  # same as blank canvas
        tool.on_press(0, 0)
        assert not stack.can_undo  # nothing was pushed

    def test_selection_restricts_fill(self):
        s, stack = _setup()
        # Only allow filling in the top-left 4×4 quadrant.
        mask = np.zeros((16, 16), dtype=bool)
        mask[:4, :4] = True
        s.set_selection(mask)
        tool = FillTool(s, stack)
        tool.foreground = RED
        tool.on_press(0, 0)
        px = s.get_cel(0, 0).pixels
        assert tuple(px[0, 0]) == RED
        assert tuple(px[5, 5]) == TRANSPARENT


class TestEyedropperTool:
    def test_samples_foreground_color(self):
        s, stack = _setup()
        px = np.zeros((16, 16, 4), dtype=np.uint8)
        px[5, 5] = RED
        s.set_cel_pixels(0, 0, px)
        tool = EyedropperTool(s, stack)
        tool.on_press(5, 5)
        assert tool.foreground == RED

    def test_samples_merged_composites(self):
        """Eyedropper with sample_merged=True reads from composited all-layers."""
        s = Sprite(8, 8)
        s.add_layer("Bottom")
        s.add_layer("Top")
        s.add_frame()
        stack = CommandStack()
        # Bottom layer: red at (2,2).
        bot = np.zeros((8, 8, 4), dtype=np.uint8)
        bot[2, 2] = RED
        s.set_cel_pixels(0, 0, bot)
        # Top layer: transparent.
        tool = EyedropperTool(s, stack)
        tool.sample_merged = True
        tool.on_press(2, 2)
        # Should see red from the bottom layer.
        assert tool.foreground == RED

    def test_samples_single_layer(self):
        s = Sprite(8, 8)
        s.add_layer("Bottom")
        s.add_layer("Top")
        s.add_frame()
        stack = CommandStack()
        bot = np.zeros((8, 8, 4), dtype=np.uint8)
        bot[2, 2] = RED
        s.set_cel_pixels(0, 0, bot)
        tool = EyedropperTool(s, stack)
        tool.sample_merged = False
        tool.layer_index = 1  # top layer (transparent)
        tool.on_press(2, 2)
        assert tool.foreground == TRANSPARENT

    def test_drag_updates_foreground(self):
        s, stack = _setup()
        px = np.zeros((16, 16, 4), dtype=np.uint8)
        px[3, 3] = RED
        px[7, 7] = GREEN
        s.set_cel_pixels(0, 0, px)
        tool = EyedropperTool(s, stack)
        tool.on_press(3, 3)
        assert tool.foreground == RED
        tool.on_drag(7, 7)
        assert tool.foreground == GREEN

    def test_out_of_bounds_does_nothing(self):
        s, stack = _setup()
        tool = EyedropperTool(s, stack)
        tool.foreground = RED
        tool.on_press(100, 100)  # out of bounds
        assert tool.foreground == RED  # unchanged
