# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for utils/geometry.py drawing primitives."""

import numpy as np
import pytest
from spriter.utils.geometry import (
    draw_ellipse,
    draw_line,
    draw_rect,
    flood_fill,
    get_pixel,
    line_points,
    set_pixel,
)

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
TRANSPARENT = (0, 0, 0, 0)


def blank(w=16, h=16) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


class TestSetGetPixel:
    def test_set_and_get(self):
        px = blank()
        set_pixel(px, 3, 5, RED)
        assert get_pixel(px, 3, 5) == RED

    def test_set_out_of_bounds_is_noop(self):
        px = blank()
        set_pixel(px, 100, 100, RED)  # no error, no mutation
        assert (px == 0).all()

    def test_get_out_of_bounds(self):
        px = blank()
        with pytest.raises(IndexError):
            get_pixel(px, 100, 0)


class TestDrawLine:
    def test_horizontal_line(self):
        px = blank()
        pts = draw_line(px, 0, 0, 7, 0, RED)
        assert len(pts) == 8
        for x in range(8):
            assert get_pixel(px, x, 0) == RED

    def test_vertical_line(self):
        px = blank()
        draw_line(px, 0, 0, 0, 7, RED)
        for y in range(8):
            assert get_pixel(px, 0, y) == RED

    def test_single_point(self):
        px = blank()
        pts = draw_line(px, 4, 4, 4, 4, RED)
        assert pts == [(4, 4)]
        assert get_pixel(px, 4, 4) == RED

    def test_diagonal(self):
        px = blank()
        pts = draw_line(px, 0, 0, 4, 4, RED)
        assert (0, 0) in pts
        assert (4, 4) in pts

    def test_line_points_no_draw(self):
        pts = line_points(0, 0, 3, 0)
        assert pts == [(0, 0), (1, 0), (2, 0), (3, 0)]


class TestDrawRect:
    def test_outline(self):
        px = blank()
        draw_rect(px, 1, 1, 4, 4, RED, filled=False)
        # Corners must be painted.
        assert get_pixel(px, 1, 1) == RED
        assert get_pixel(px, 4, 1) == RED
        assert get_pixel(px, 1, 4) == RED
        assert get_pixel(px, 4, 4) == RED
        # Interior must be empty.
        assert get_pixel(px, 2, 2) == TRANSPARENT

    def test_filled(self):
        px = blank()
        draw_rect(px, 1, 1, 4, 4, RED, filled=True)
        for y in range(1, 5):
            for x in range(1, 5):
                assert get_pixel(px, x, y) == RED
        # Outside must be empty.
        assert get_pixel(px, 0, 0) == TRANSPARENT

    def test_zero_size_is_noop(self):
        px = blank()
        draw_rect(px, 0, 0, 0, 4, RED)
        assert (px == 0).all()


class TestDrawEllipse:
    def test_single_point_rx_ry_zero(self):
        px = blank()
        draw_ellipse(px, 8, 8, 0, 0, RED)
        assert get_pixel(px, 8, 8) == RED

    def test_circle_outline_center_is_empty(self):
        px = blank(32, 32)
        draw_ellipse(px, 16, 16, 5, 5, RED, filled=False)
        # Center should be transparent for an outline-only ellipse.
        assert get_pixel(px, 16, 16) == TRANSPARENT

    def test_circle_filled_center_is_painted(self):
        px = blank(32, 32)
        draw_ellipse(px, 16, 16, 5, 5, RED, filled=True)
        assert get_pixel(px, 16, 16) == RED

    def test_ellipse_axes_painted(self):
        px = blank(32, 32)
        draw_ellipse(px, 15, 15, 5, 3, RED, filled=False)
        # Leftmost and rightmost points of the ellipse should be painted.
        assert get_pixel(px, 15 - 5, 15) == RED
        assert get_pixel(px, 15 + 5, 15) == RED


class TestFloodFill:
    def test_basic_fill(self):
        px = blank(8, 8)
        count = flood_fill(px, 0, 0, RED)
        assert count == 64
        assert (px == np.array(RED, dtype=np.uint8)).all()

    def test_bounded_fill(self):
        px = blank(8, 8)
        # Draw a 3x3 box outline in red, then fill the interior.
        draw_rect(px, 1, 1, 3, 3, RED, filled=False)
        flood_fill(px, 2, 2, GREEN)
        assert get_pixel(px, 2, 2) == GREEN
        # Border pixels remain red.
        assert get_pixel(px, 1, 1) == RED

    def test_same_color_noop(self):
        px = blank(4, 4)
        count = flood_fill(px, 0, 0, TRANSPARENT)
        assert count == 0

    def test_out_of_bounds_seed(self):
        px = blank(4, 4)
        count = flood_fill(px, 100, 100, RED)
        assert count == 0

    def test_4_connectivity(self):
        # Diagonal gap should NOT be crossed with 4-connectivity.
        px = blank(4, 4)
        # Block: set specific pixels to form a diagonal barrier.
        set_pixel(px, 1, 0, RED)
        set_pixel(px, 0, 1, RED)
        flood_fill(px, 2, 0, GREEN, connectivity=4)
        # (0,0) is 4-connected to neither RED nor GREEN region,
        # but it is separated from (2,0) by RED pixels.
        assert get_pixel(px, 0, 0) != GREEN

    def test_8_connectivity(self):
        px = blank(4, 4)
        flood_fill(px, 0, 0, GREEN, connectivity=8)
        assert (px == np.array(GREEN, dtype=np.uint8)).all()

    def test_invalid_connectivity(self):
        px = blank()
        with pytest.raises(ValueError):
            flood_fill(px, 0, 0, RED, connectivity=6)
