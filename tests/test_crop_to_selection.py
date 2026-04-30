# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for :class:`CropToSelectionCommand`."""

from __future__ import annotations

import numpy as np
import pytest

from spriter.commands.transform import CropToSelectionCommand
from spriter.core.sprite import Sprite


def _make_sprite(w: int = 8, h: int = 8, layers: int = 1, frames: int = 1) -> Sprite:
    sprite = Sprite(w, h)
    for i in range(layers):
        sprite.add_layer(f"L{i}")
    for _ in range(frames):
        sprite.add_frame()
    return sprite


def test_crop_basic_reduces_canvas_to_bbox():
    sprite = _make_sprite(8, 8)
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[2:6, 3:7] = (255, 0, 0, 255)
    sprite.set_cel_pixels(0, 0, pixels)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 3:7] = True
    sprite.selection_mask = mask

    cmd = CropToSelectionCommand(sprite)
    cmd.execute()

    assert sprite.width == 4
    assert sprite.height == 4
    out = sprite.get_cel(0, 0).pixels
    assert out is not None
    assert out.shape == (4, 4, 4)
    assert np.all(out == [255, 0, 0, 255])
    # Selection cleared on crop.
    assert sprite.selection_mask is None


def test_crop_uses_bbox_of_non_rectangular_mask():
    sprite = _make_sprite(8, 8)
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[1:4, 2:5] = (10, 20, 30, 255)
    sprite.set_cel_pixels(0, 0, pixels)
    # L-shaped mask whose bbox is rows 1..3, cols 2..4 (3x3).
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 2] = True
    mask[3, 4] = True
    sprite.selection_mask = mask

    cmd = CropToSelectionCommand(sprite)
    cmd.execute()

    assert (sprite.width, sprite.height) == (3, 3)
    out = sprite.get_cel(0, 0).pixels
    assert out is not None
    assert np.array_equal(out, pixels[1:4, 2:5])


def test_crop_applies_to_all_layers_and_frames():
    sprite = _make_sprite(6, 6, layers=2, frames=3)
    # Distinct fill per cel so we can verify each crop independently.
    for li in range(2):
        for fi in range(3):
            buf = np.zeros((6, 6, 4), dtype=np.uint8)
            buf[:, :] = (li * 50, fi * 50, 100, 255)
            sprite.set_cel_pixels(li, fi, buf)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 2:5] = True
    sprite.selection_mask = mask

    cmd = CropToSelectionCommand(sprite)
    cmd.execute()

    assert (sprite.width, sprite.height) == (3, 3)
    for li in range(2):
        for fi in range(3):
            out = sprite.get_cel(li, fi).pixels
            assert out is not None
            assert out.shape == (3, 3, 4)
            assert np.all(out == [li * 50, fi * 50, 100, 255])


def test_crop_undo_redo_roundtrip():
    sprite = _make_sprite(8, 8, layers=2, frames=2)
    originals = {}
    for li in range(2):
        for fi in range(2):
            buf = np.zeros((8, 8, 4), dtype=np.uint8)
            buf[fi : fi + 4, li : li + 4] = (200, 100, 50, 255)
            sprite.set_cel_pixels(li, fi, buf)
            originals[(li, fi)] = buf.copy()
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    sprite.selection_mask = mask
    original_mask = mask.copy()

    cmd = CropToSelectionCommand(sprite)
    cmd.execute()
    assert (sprite.width, sprite.height) == (4, 4)
    assert sprite.selection_mask is None

    cmd.undo()
    assert (sprite.width, sprite.height) == (8, 8)
    assert sprite.selection_mask is not None
    assert np.array_equal(sprite.selection_mask, original_mask)
    for (li, fi), buf in originals.items():
        out = sprite.get_cel(li, fi).pixels
        assert out is not None
        assert np.array_equal(out, buf)

    # Redo via execute() again.
    cmd.execute()
    assert (sprite.width, sprite.height) == (4, 4)
    assert sprite.selection_mask is None


def test_crop_requires_active_selection():
    sprite = _make_sprite()
    sprite.selection_mask = None
    with pytest.raises(ValueError):
        CropToSelectionCommand(sprite)


def test_crop_rejects_empty_mask():
    sprite = _make_sprite()
    sprite.selection_mask = np.zeros((8, 8), dtype=bool)
    with pytest.raises(ValueError):
        CropToSelectionCommand(sprite)


def test_crop_full_canvas_selection_is_noop_in_size():
    sprite = _make_sprite(5, 5)
    pixels = np.arange(5 * 5 * 4, dtype=np.uint8).reshape(5, 5, 4)
    sprite.set_cel_pixels(0, 0, pixels)
    sprite.selection_mask = np.ones((5, 5), dtype=bool)

    cmd = CropToSelectionCommand(sprite)
    cmd.execute()

    assert (sprite.width, sprite.height) == (5, 5)
    out = sprite.get_cel(0, 0).pixels
    assert out is not None
    assert np.array_equal(out, pixels)
