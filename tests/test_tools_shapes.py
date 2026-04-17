# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for shape tools: LineTool, RectangleTool, EllipseTool."""

import numpy as np
import pytest
from spriter.commands.base import CommandStack
from spriter.core.sprite import Sprite
from spriter.tools.ellipse import EllipseTool
from spriter.tools.line import LineTool
from spriter.tools.rectangle import RectangleTool

RED = (255, 0, 0, 255)
TRANSPARENT = (0, 0, 0, 0)


def _setup(w=32, h=32):
    s = Sprite(w, h)
    s.add_layer()
    s.add_frame()
    stack = CommandStack()
    return s, stack


def _pixels(s):
    return s.get_cel(0, 0).pixels


class TestLineTool:
    def test_horizontal_line(self):
        s, stack = _setup()
        tool = LineTool(s, stack)
        tool.foreground = RED
        tool.on_press(0, 5)
        tool.on_release(7, 5)
        px = _pixels(s)
        for x in range(8):
            assert tuple(px[5, x]) == RED

    def test_vertical_line(self):
        s, stack = _setup()
        tool = LineTool(s, stack)
        tool.foreground = RED
        tool.on_press(5, 0)
        tool.on_release(5, 7)
        px = _pixels(s)
        for y in range(8):
            assert tuple(px[y, 5]) == RED

    def test_drag_shows_preview_not_multiple_lines(self):
        """Dragging should redraw the single line, not accumulate lines."""
        s, stack = _setup()
        tool = LineTool(s, stack)
        tool.foreground = RED
        tool.on_press(0, 0)
        tool.on_drag(5, 0)  # horizontal
        tool.on_drag(0, 5)  # now vertical — should erase previous
        tool.on_release(0, 5)
        px = _pixels(s)
        # Vertical line y=0..5 at x=0 should be red.
        for y in range(6):
            assert tuple(px[y, 0]) == RED
        # Previous horizontal drag must not persist.
        assert tuple(px[0, 5]) == TRANSPARENT

    def test_undo_reverts(self):
        s, stack = _setup()
        tool = LineTool(s, stack)
        tool.foreground = RED
        tool.on_press(0, 0)
        tool.on_release(7, 0)
        stack.undo()
        px = _pixels(s)
        assert tuple(px[0, 0]) == TRANSPARENT


class TestRectangleTool:
    def test_outline_rectangle(self):
        s, stack = _setup()
        tool = RectangleTool(s, stack)
        tool.foreground = RED
        tool.filled = False
        tool.on_press(1, 1)
        tool.on_release(5, 5)
        px = _pixels(s)
        # Corners must be painted.
        assert tuple(px[1, 1]) == RED
        assert tuple(px[1, 5]) == RED
        assert tuple(px[5, 1]) == RED
        assert tuple(px[5, 5]) == RED
        # Interior must be transparent.
        assert tuple(px[3, 3]) == TRANSPARENT

    def test_filled_rectangle(self):
        s, stack = _setup()
        tool = RectangleTool(s, stack)
        tool.foreground = RED
        tool.filled = True
        tool.on_press(1, 1)
        tool.on_release(4, 4)
        px = _pixels(s)
        for y in range(1, 5):
            for x in range(1, 5):
                assert tuple(px[y, x]) == RED

    def test_rounded_corners_outline(self):
        s, stack = _setup()
        tool = RectangleTool(s, stack)
        tool.foreground = RED
        tool.filled = False
        tool.corner_radius = 3
        tool.on_press(3, 3)
        tool.on_release(15, 15)
        px = _pixels(s)
        # The exact corners (3,3), (15,3) etc. should NOT be painted (rounded away).
        assert tuple(px[3, 3]) == TRANSPARENT
        assert tuple(px[3, 15]) == TRANSPARENT
        # The midpoints of the top edge must be painted.
        assert tuple(px[3, 9]) == RED

    def test_undo_redo(self):
        s, stack = _setup()
        tool = RectangleTool(s, stack)
        tool.foreground = RED
        tool.filled = True
        tool.on_press(0, 0)
        tool.on_release(3, 3)
        stack.undo()
        px = _pixels(s)
        assert tuple(px[0, 0]) == TRANSPARENT
        stack.redo()
        assert tuple(_pixels(s)[0, 0]) == RED


class TestEllipseTool:
    def test_outline_ellipse_center_transparent(self):
        s, stack = _setup()
        tool = EllipseTool(s, stack)
        tool.foreground = RED
        tool.filled = False
        tool.on_press(6, 6)
        tool.on_release(18, 18)
        px = _pixels(s)
        # Centre of the ellipse (midpoint of bounding box = 12, 12) should be
        # transparent for an outline.
        assert tuple(px[12, 12]) == TRANSPARENT

    def test_filled_ellipse_center_painted(self):
        s, stack = _setup()
        tool = EllipseTool(s, stack)
        tool.foreground = RED
        tool.filled = True
        tool.on_press(4, 4)
        tool.on_release(20, 20)
        px = _pixels(s)
        # Centre must be painted.
        assert tuple(px[12, 12]) == RED

    def test_undo(self):
        s, stack = _setup()
        tool = EllipseTool(s, stack)
        tool.foreground = RED
        tool.filled = True
        tool.on_press(4, 4)
        tool.on_release(12, 12)
        stack.undo()
        px = _pixels(s)
        assert tuple(px[8, 8]) == TRANSPARENT
