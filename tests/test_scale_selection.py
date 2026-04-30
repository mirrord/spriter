# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for :class:`ScaleSelectionCommand`."""

from __future__ import annotations

import numpy as np
import pytest

from spriter.commands.transform import ScaleSelectionCommand
from spriter.core.sprite import Sprite


def _make_sprite_with_block(color=(255, 0, 0, 255)):
    """8x8 sprite with one default layer/frame and a 2x2 red block at (1,1)."""
    sprite = Sprite(8, 8)
    sprite.add_layer("L")
    sprite.add_frame()
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[1:3, 1:3] = color
    sprite.set_cel_pixels(0, 0, pixels)
    return sprite


def test_scale_selection_grows_region():
    sprite = _make_sprite_with_block()
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    sprite.selection_mask = mask

    cmd = ScaleSelectionCommand(sprite, 0, 0, new_width=4, new_height=4)
    cmd.execute()

    pixels = sprite.get_cel(0, 0).pixels
    assert pixels is not None
    # 4x4 red block now anchored at top-left of original bbox (1,1).
    assert np.all(pixels[1:5, 1:5] == [255, 0, 0, 255])
    # Selection mask updated to the new 4x4 region.
    assert sprite.selection_mask is not None
    assert sprite.selection_mask[1:5, 1:5].all()
    assert sprite.selection_mask.sum() == 16


def test_scale_selection_shrinks_region():
    sprite = _make_sprite_with_block()
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:5, 1:5] = True  # 4x4 selection (original block is 2x2 inside it)
    # Fill the 4x4 area with red so shrinking is meaningful.
    pixels = sprite.get_cel(0, 0).pixels.copy()
    pixels[1:5, 1:5] = [0, 255, 0, 255]
    sprite.set_cel_pixels(0, 0, pixels)
    sprite.selection_mask = mask

    cmd = ScaleSelectionCommand(sprite, 0, 0, new_width=2, new_height=2)
    cmd.execute()

    pixels = sprite.get_cel(0, 0).pixels
    assert pixels is not None
    # New 2x2 green block at (1,1).
    assert np.all(pixels[1:3, 1:3] == [0, 255, 0, 255])
    # Pixels outside the new selection but inside the old bbox are cleared.
    assert np.all(pixels[3:5, 1:5] == 0)
    assert np.all(pixels[1:3, 3:5] == 0)
    assert sprite.selection_mask is not None
    assert sprite.selection_mask.sum() == 4


def test_scale_selection_undo_redo_roundtrip():
    sprite = _make_sprite_with_block()
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    sprite.selection_mask = mask

    original_pixels = sprite.get_cel(0, 0).pixels.copy()
    original_mask = mask.copy()

    cmd = ScaleSelectionCommand(sprite, 0, 0, 4, 4)
    cmd.execute()
    cmd.undo()

    assert np.array_equal(sprite.get_cel(0, 0).pixels, original_pixels)
    assert sprite.selection_mask is not None
    assert np.array_equal(sprite.selection_mask, original_mask)

    # Redo via execute() again.
    cmd.execute()
    assert sprite.selection_mask is not None
    assert sprite.selection_mask.sum() == 16


def test_scale_selection_requires_active_selection():
    sprite = _make_sprite_with_block()
    sprite.selection_mask = None
    with pytest.raises(ValueError):
        ScaleSelectionCommand(sprite, 0, 0, 4, 4)


def test_scale_selection_rejects_empty_mask():
    sprite = _make_sprite_with_block()
    sprite.selection_mask = np.zeros((8, 8), dtype=bool)
    with pytest.raises(ValueError):
        ScaleSelectionCommand(sprite, 0, 0, 4, 4)


def test_scale_selection_rejects_non_positive_size():
    sprite = _make_sprite_with_block()
    sprite.selection_mask = np.ones((8, 8), dtype=bool)
    with pytest.raises(ValueError):
        ScaleSelectionCommand(sprite, 0, 0, 0, 4)
    with pytest.raises(ValueError):
        ScaleSelectionCommand(sprite, 0, 0, 4, -1)


def test_scale_selection_clips_to_canvas():
    sprite = _make_sprite_with_block()
    mask = np.zeros((8, 8), dtype=bool)
    mask[6:8, 6:8] = True  # 2x2 selection in bottom-right corner.
    pixels = sprite.get_cel(0, 0).pixels.copy()
    pixels[6:8, 6:8] = [0, 0, 255, 255]
    sprite.set_cel_pixels(0, 0, pixels)
    sprite.selection_mask = mask

    # Try to scale to 8x8 — destination would extend beyond the canvas.
    cmd = ScaleSelectionCommand(sprite, 0, 0, 8, 8)
    cmd.execute()

    # New region clipped to (6:8, 6:8) — only 2x2 in the corner.
    out = sprite.get_cel(0, 0).pixels
    assert out is not None
    assert sprite.selection_mask is not None
    assert sprite.selection_mask[6:8, 6:8].all()
    assert sprite.selection_mask.sum() == 4
    assert np.all(out[6:8, 6:8] == [0, 0, 255, 255])
