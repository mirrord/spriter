# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for RectSelectTool, LassoTool, and MagicWandTool."""

import numpy as np
import pytest
from spriter.commands.base import CommandStack
from spriter.core.sprite import Sprite
from spriter.tools.select import LassoTool, MagicWandTool, RectSelectTool

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)


def _setup(w=16, h=16):
    s = Sprite(w, h)
    s.add_layer()
    s.add_frame()
    stack = CommandStack()
    return s, stack


class TestRectSelectTool:
    def test_creates_rect_mask(self):
        s, stack = _setup()
        tool = RectSelectTool(s, stack)
        tool.on_press(2, 2)
        tool.on_release(6, 6)
        mask = s.selection_mask
        assert mask is not None
        assert mask[2, 2]
        assert mask[6, 6]
        assert not mask[0, 0]
        assert not mask[7, 7]

    def test_mask_shape(self):
        s, stack = _setup()
        tool = RectSelectTool(s, stack)
        tool.on_press(0, 0)
        tool.on_release(15, 15)
        assert s.selection_mask is not None
        assert s.selection_mask.shape == (16, 16)

    def test_inverted_drag_handled(self):
        """Press at bottom-right, release at top-left — same result."""
        s, stack = _setup()
        tool = RectSelectTool(s, stack)
        tool.on_press(6, 6)
        tool.on_release(2, 2)
        mask = s.selection_mask
        assert mask is not None
        assert mask[2, 2] and mask[6, 6]

    def test_undo_clears_selection(self):
        s, stack = _setup()
        tool = RectSelectTool(s, stack)
        tool.on_press(0, 0)
        tool.on_release(5, 5)
        stack.undo()
        assert s.selection_mask is None

    def test_undo_restores_previous_selection(self):
        s, stack = _setup()
        tool = RectSelectTool(s, stack)
        # First selection.
        tool.on_press(0, 0)
        tool.on_release(3, 3)
        first = s.selection_mask.copy()
        # Second selection.
        tool.on_press(5, 5)
        tool.on_release(8, 8)
        stack.undo()
        assert np.array_equal(s.selection_mask, first)


class TestLassoTool:
    def test_triangle_selection(self):
        s, stack = _setup(32, 32)
        tool = LassoTool(s, stack)
        # A large triangle.
        tool.on_press(5, 25)
        tool.on_drag(15, 5)
        tool.on_drag(25, 25)
        tool.on_release(25, 25)
        mask = s.selection_mask
        assert mask is not None
        assert mask.shape == (32, 32)
        # Interior point of the triangle must be selected.
        assert mask[20, 15]

    def test_undo(self):
        s, stack = _setup()
        tool = LassoTool(s, stack)
        tool.on_press(1, 1)
        tool.on_drag(8, 1)
        tool.on_drag(8, 8)
        tool.on_release(1, 8)
        stack.undo()
        assert s.selection_mask is None

    def test_degenerate_fewer_than_3_points(self):
        """A lasso with fewer than 3 points should produce an empty mask."""
        s, stack = _setup()
        tool = LassoTool(s, stack)
        tool.on_press(5, 5)
        tool.on_release(5, 5)
        mask = s.selection_mask
        # Mask is created but expected to be all-false (no area).
        assert mask is not None
        assert not mask.any()


class TestMagicWandTool:
    def test_selects_contiguous_region(self):
        s, stack = _setup(16, 16)
        # Paint a 4×4 red block.
        px = np.zeros((16, 16, 4), dtype=np.uint8)
        px[4:8, 4:8] = RED
        s.set_cel_pixels(0, 0, px)
        tool = MagicWandTool(s, stack)
        tool.tolerance = 0
        tool.on_press(5, 5)  # inside red block
        mask = s.selection_mask
        assert mask is not None
        # Inside selected.
        assert mask[5, 5] and mask[4, 4] and mask[7, 7]
        # Outside (transparent) must not be selected.
        assert not mask[0, 0]

    def test_tolerance_selects_similar_colors(self):
        s, stack = _setup(8, 8)
        px = np.zeros((8, 8, 4), dtype=np.uint8)
        px[0, 0] = (255, 0, 0, 255)
        px[0, 1] = (245, 0, 0, 255)  # within distance 10
        px[0, 2] = (0, 255, 0, 255)  # very different
        s.set_cel_pixels(0, 0, px)
        tool = MagicWandTool(s, stack)
        tool.tolerance = 15
        tool.on_press(0, 0)
        mask = s.selection_mask
        assert mask is not None
        assert mask[0, 0] and mask[0, 1]
        assert not mask[0, 2]

    def test_undo(self):
        s, stack = _setup()
        tool = MagicWandTool(s, stack)
        tool.on_press(0, 0)
        stack.undo()
        assert s.selection_mask is None
