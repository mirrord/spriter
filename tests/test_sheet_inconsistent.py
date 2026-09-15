# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for importing sprite sheets with inconsistent frame spacing."""

import numpy as np
import pytest
from PIL import Image


def _blank(h: int, w: int) -> np.ndarray:
    """Transparent RGBA canvas."""
    return np.zeros((h, w, 4), dtype=np.uint8)


def _rect(arr, y0, x0, h, w, color=(255, 0, 0, 255)) -> None:
    """Paint a solid filled rectangle into an RGBA array in place."""
    arr[y0 : y0 + h, x0 : x0 + w] = np.array(color, dtype=np.uint8)


class TestLabelComponents:
    def test_diagonal_pixels_are_one_component(self):
        from spriter.io.spritesheet import _label_components

        mask = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ]
        )
        labels, count = _label_components(mask, connectivity=8)
        assert count == 1
        assert len(set(labels[mask].tolist())) == 1

    def test_diagonal_pixels_split_under_4_connectivity(self):
        from spriter.io.spritesheet import _label_components

        mask = np.array(
            [
                [True, False],
                [False, True],
            ]
        )
        labels, count = _label_components(mask, connectivity=4)
        assert count == 2

    def test_separate_blobs_not_merged(self):
        from spriter.io.spritesheet import _label_components

        mask = np.zeros((5, 9), dtype=bool)
        mask[1:3, 1:3] = True
        mask[1:3, 6:8] = True
        _, count = _label_components(mask, connectivity=8)
        assert count == 2

    def test_empty_mask(self):
        from spriter.io.spritesheet import _label_components

        _, count = _label_components(np.zeros((4, 4), dtype=bool))
        assert count == 0


class TestImportSheetAuto:
    def test_frame_count_and_size_from_max_dims(self):
        from spriter.io.spritesheet import import_sheet_auto

        arr = _blank(40, 60)
        _rect(arr, 2, 2, 6, 6, (255, 0, 0, 255))  # 6x6
        _rect(arr, 4, 20, 10, 8, (0, 255, 0, 255))  # 10x8 (tallest/widest? h=10,w=8)
        _rect(arr, 2, 40, 4, 12, (0, 0, 255, 255))  # 4x12 (widest w=12)
        sprite = import_sheet_auto(arr)
        assert sprite.frame_count == 3
        # Max width across candidates = 12, max height = 10.
        assert sprite.width == 12
        assert sprite.height == 10

    def test_candidates_centered_not_stretched(self):
        from spriter.io.spritesheet import import_sheet_auto

        arr = _blank(30, 40)
        _rect(arr, 1, 1, 4, 4, (255, 0, 0, 255))  # small 4x4
        _rect(arr, 1, 20, 10, 10, (0, 255, 0, 255))  # big 10x10 -> frame size
        sprite = import_sheet_auto(arr)
        assert sprite.width == 10 and sprite.height == 10
        # First candidate (leftmost) is the 4x4 red block, centered in 10x10.
        cel0 = sprite.get_cel(0, 0)
        assert cel0 is not None
        px = cel0.pixels
        # Offset = (10-4)//2 = 3.
        opaque = px[..., 3] > 0
        ys, xs = np.nonzero(opaque)
        assert ys.min() == 3 and ys.max() == 6
        assert xs.min() == 3 and xs.max() == 6
        # Not stretched: exactly 4x4 opaque region, red.
        assert opaque.sum() == 16
        assert np.all(px[3:7, 3:7, :3] == np.array([255, 0, 0], dtype=np.uint8))

    def test_ordering_left_to_right_then_top_to_bottom(self):
        from spriter.io.spritesheet import import_sheet_auto

        arr = _blank(40, 40)
        # Row 0: two blocks, right one placed first in array but should sort by x.
        _rect(arr, 2, 20, 5, 5, (10, 0, 0, 255))  # row0 right
        _rect(arr, 2, 2, 5, 5, (20, 0, 0, 255))  # row0 left
        # Row 1 (lower): one block.
        _rect(arr, 25, 10, 5, 5, (30, 0, 0, 255))  # row1
        sprite = import_sheet_auto(arr)
        assert sprite.frame_count == 3

        def red(fi):
            return int(sprite.get_cel(0, fi).pixels[..., 0].max())

        assert red(0) == 20  # row0 left
        assert red(1) == 10  # row0 right
        assert red(2) == 30  # row1

    def test_no_neighbor_bleed(self):
        from spriter.io.spritesheet import import_sheet_auto

        # An L-shaped sprite whose bbox overlaps a neighbor block's column.
        arr = _blank(20, 20)
        # Component A: L shape spanning cols 1..6, rows 1..6.
        _rect(arr, 1, 1, 6, 2, (255, 0, 0, 255))  # vertical bar
        _rect(arr, 5, 1, 2, 6, (255, 0, 0, 255))  # horizontal foot
        # Component B: sits inside A's bbox corner region but separated.
        _rect(arr, 1, 5, 2, 2, (0, 0, 255, 255))
        sprite = import_sheet_auto(arr)
        # Two components.
        assert sprite.frame_count == 2
        # Neither frame should contain both red and blue.
        for fi in range(2):
            px = sprite.get_cel(0, fi).pixels
            has_red = np.any((px[..., 0] == 255) & (px[..., 3] > 0))
            has_blue = np.any((px[..., 2] == 255) & (px[..., 3] > 0))
            assert not (has_red and has_blue)

    def test_pixel_fidelity_preserves_interior(self):
        from spriter.io.spritesheet import import_sheet_auto

        arr = _blank(20, 20)
        # A 6x6 red block with a distinct green interior pixel.
        _rect(arr, 2, 2, 6, 6, (255, 0, 0, 255))
        arr[4, 4] = np.array([0, 255, 0, 255], dtype=np.uint8)
        sprite = import_sheet_auto(arr)
        assert sprite.frame_count == 1
        px = sprite.get_cel(0, 0).pixels
        # Interior green pixel preserved somewhere in the frame.
        assert np.any((px[..., 1] == 255) & (px[..., 0] == 0) & (px[..., 3] > 0))

    def test_empty_sheet_raises(self):
        from spriter.io.spritesheet import import_sheet_auto

        with pytest.raises(ValueError):
            import_sheet_auto(_blank(16, 16))

    def test_accepts_path(self, tmp_path):
        from spriter.io.spritesheet import import_sheet_auto

        arr = _blank(20, 30)
        _rect(arr, 2, 2, 5, 5)
        _rect(arr, 2, 15, 8, 8)
        path = tmp_path / "sheet.png"
        Image.fromarray(arr, "RGBA").save(str(path))
        sprite = import_sheet_auto(path)
        assert sprite.frame_count == 2
        assert sprite.width == 8 and sprite.height == 8

    def test_opaque_background_sheet(self):
        from spriter.io.spritesheet import import_sheet_auto

        # Fully opaque sheet with a magenta background and two sprites.
        arr = np.zeros((20, 30, 4), dtype=np.uint8)
        arr[..., :3] = np.array([255, 0, 255], dtype=np.uint8)  # bg
        arr[..., 3] = 255
        _rect(arr, 2, 2, 5, 5, (0, 0, 0, 255))
        _rect(arr, 2, 15, 8, 8, (255, 255, 255, 255))
        sprite = import_sheet_auto(arr)
        assert sprite.frame_count == 2
        assert sprite.width == 8 and sprite.height == 8
