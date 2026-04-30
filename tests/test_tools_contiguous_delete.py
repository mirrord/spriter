# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for ContiguousDeleteTool."""

import numpy as np
from spriter.commands.base import CommandStack
from spriter.core.sprite import Sprite
from spriter.tools.contiguous_delete import ContiguousDeleteTool
from spriter.utils.geometry import draw_rect

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
TRANSPARENT = (0, 0, 0, 0)


def _setup(w=16, h=16):
    s = Sprite(w, h)
    s.add_layer()
    s.add_frame()
    return s, CommandStack()


def _solid(w, h, color):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :] = color
    return arr


class TestContiguousDeleteTool:
    def test_deletes_solid_region(self):
        s, stack = _setup()
        s.set_cel_pixels(0, 0, _solid(16, 16, RED))
        tool = ContiguousDeleteTool(s, stack)
        tool.on_press(0, 0)
        px = s.get_cel(0, 0).pixels
        assert (px == np.array(TRANSPARENT, dtype=np.uint8)).all()

    def test_does_not_cross_border(self):
        s, stack = _setup()
        # 6×6 red box outline filled with green interior.
        init = np.zeros((16, 16, 4), dtype=np.uint8)
        draw_rect(init, 2, 2, 6, 6, RED, filled=False)
        # Fill the interior (3..6, 3..6) with green.
        init[3:7, 3:7] = GREEN
        s.set_cel_pixels(0, 0, init)
        tool = ContiguousDeleteTool(s, stack)
        tool.on_press(4, 4)
        px = s.get_cel(0, 0).pixels
        # Green interior is now transparent.
        assert tuple(px[4, 4]) == TRANSPARENT
        assert tuple(px[3, 3]) == TRANSPARENT
        # Red border is preserved.
        assert tuple(px[2, 2]) == RED
        assert tuple(px[7, 7]) == RED

    def test_undo_restores_pixels(self):
        s, stack = _setup()
        s.set_cel_pixels(0, 0, _solid(16, 16, RED))
        tool = ContiguousDeleteTool(s, stack)
        tool.on_press(0, 0)
        stack.undo()
        px = s.get_cel(0, 0).pixels
        assert (px == np.array(RED, dtype=np.uint8)).all()

    def test_click_on_transparent_is_noop(self):
        s, stack = _setup()
        tool = ContiguousDeleteTool(s, stack)
        tool.on_press(0, 0)
        assert not stack.can_undo

    def test_tolerance_deletes_near_colors(self):
        s, stack = _setup(8, 8)
        init = np.zeros((8, 8, 4), dtype=np.uint8)
        for x in range(4):
            init[0, x] = (250, 0, 0, 255)  # near-red, contiguous
        for x in range(4, 8):
            init[0, x] = GREEN  # outside tolerance
        s.set_cel_pixels(0, 0, init)
        tool = ContiguousDeleteTool(s, stack)
        tool.tolerance = 20
        tool.on_press(0, 0)
        px = s.get_cel(0, 0).pixels
        for x in range(4):
            assert tuple(px[0, x]) == TRANSPARENT
        assert tuple(px[0, 4]) == GREEN

    def test_selection_restricts_delete(self):
        s, stack = _setup()
        s.set_cel_pixels(0, 0, _solid(16, 16, RED))
        mask = np.zeros((16, 16), dtype=bool)
        mask[:4, :4] = True
        s.set_selection(mask)
        tool = ContiguousDeleteTool(s, stack)
        tool.on_press(0, 0)
        px = s.get_cel(0, 0).pixels
        assert tuple(px[0, 0]) == TRANSPARENT
        # Outside selection — pixels remain red.
        assert tuple(px[10, 10]) == RED

    def test_eight_connectivity(self):
        s, stack = _setup(4, 4)
        init = np.zeros((4, 4, 4), dtype=np.uint8)
        # Diagonal stripe of red.
        init[0, 0] = RED
        init[1, 1] = RED
        init[2, 2] = RED
        init[3, 3] = RED
        s.set_cel_pixels(0, 0, init)
        tool = ContiguousDeleteTool(s, stack)
        tool.connectivity = 8
        tool.on_press(0, 0)
        px = s.get_cel(0, 0).pixels
        for i in range(4):
            assert tuple(px[i, i]) == TRANSPARENT
