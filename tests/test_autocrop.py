# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for :class:`AutocropCommand`."""

from __future__ import annotations

import numpy as np
import pytest

from spriter.commands.transform import AutocropCommand
from spriter.core.sprite import Sprite


def _make_sprite(w: int = 16, h: int = 16, layers: int = 1, frames: int = 1) -> Sprite:
    sprite = Sprite(w, h)
    for i in range(layers):
        sprite.add_layer(f"L{i}")
    for _ in range(frames):
        sprite.add_frame()
    return sprite


def test_autocrop_single_pixel_shrinks_to_1x1():
    sprite = _make_sprite(16, 16)
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[7, 5] = (10, 20, 30, 255)
    sprite.set_cel_pixels(0, 0, pixels)

    cmd = AutocropCommand(sprite)
    cmd.execute()

    assert (sprite.width, sprite.height) == (1, 1)
    out = sprite.get_cel(0, 0).pixels
    assert out is not None
    assert out.shape == (1, 1, 4)
    assert tuple(int(v) for v in out[0, 0]) == (10, 20, 30, 255)


def test_autocrop_rectangle_bbox():
    sprite = _make_sprite(16, 16)
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[3:9, 4:11] = (255, 0, 0, 255)  # 6 rows × 7 cols
    sprite.set_cel_pixels(0, 0, pixels)

    cmd = AutocropCommand(sprite)
    cmd.execute()

    assert (sprite.width, sprite.height) == (7, 6)
    out = sprite.get_cel(0, 0).pixels
    assert np.all(out == [255, 0, 0, 255])


def test_autocrop_union_across_frames():
    sprite = _make_sprite(16, 16, frames=2)
    # add_frame already called once via _make_sprite (frames=2 ⇒ 2 added → total 2)
    f0 = np.zeros((16, 16, 4), dtype=np.uint8)
    f0[2, 3] = (255, 255, 255, 255)
    sprite.set_cel_pixels(0, 0, f0)
    f1 = np.zeros((16, 16, 4), dtype=np.uint8)
    f1[10, 12] = (255, 255, 255, 255)
    sprite.set_cel_pixels(0, 1, f1)

    cmd = AutocropCommand(sprite)
    cmd.execute()

    # Union bbox: rows 2..10 (9), cols 3..12 (10).
    assert (sprite.width, sprite.height) == (10, 9)
    out0 = sprite.get_cel(0, 0).pixels
    out1 = sprite.get_cel(0, 1).pixels
    # (2,3) → (0,0); (10,12) → (8,9)
    assert tuple(int(v) for v in out0[0, 0]) == (255, 255, 255, 255)
    assert tuple(int(v) for v in out1[8, 9]) == (255, 255, 255, 255)


def test_autocrop_union_across_layers():
    sprite = _make_sprite(16, 16, layers=2)
    a = np.zeros((16, 16, 4), dtype=np.uint8)
    a[1, 1] = (255, 0, 0, 255)
    sprite.set_cel_pixels(0, 0, a)
    b = np.zeros((16, 16, 4), dtype=np.uint8)
    b[14, 13] = (0, 255, 0, 255)
    sprite.set_cel_pixels(1, 0, b)

    cmd = AutocropCommand(sprite)
    cmd.execute()

    # Rows 1..14 (14), cols 1..13 (13).
    assert (sprite.width, sprite.height) == (13, 14)
    a_out = sprite.get_cel(0, 0).pixels
    b_out = sprite.get_cel(1, 0).pixels
    assert tuple(int(v) for v in a_out[0, 0]) == (255, 0, 0, 255)
    assert tuple(int(v) for v in b_out[13, 12]) == (0, 255, 0, 255)


def test_autocrop_ignores_zero_alpha_pixels():
    sprite = _make_sprite(16, 16)
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    # Fully transparent "pixel" with non-zero RGB should not extend bbox.
    pixels[0, 0] = (255, 255, 255, 0)
    pixels[15, 15] = (1, 2, 3, 0)
    pixels[5, 6] = (10, 20, 30, 255)
    sprite.set_cel_pixels(0, 0, pixels)

    cmd = AutocropCommand(sprite)
    cmd.execute()

    assert (sprite.width, sprite.height) == (1, 1)


def test_autocrop_undo_restores_canvas_and_pixels():
    sprite = _make_sprite(16, 16)
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[4:7, 5:9] = (200, 100, 50, 255)
    sprite.set_cel_pixels(0, 0, pixels.copy())
    mask = np.zeros((16, 16), dtype=bool)
    mask[0, 0] = True
    sprite.selection_mask = mask.copy()

    cmd = AutocropCommand(sprite)
    cmd.execute()
    assert (sprite.width, sprite.height) == (4, 3)
    assert sprite.selection_mask is None

    cmd.undo()
    assert (sprite.width, sprite.height) == (16, 16)
    out = sprite.get_cel(0, 0).pixels
    assert out is not None
    assert out.shape == (16, 16, 4)
    assert np.array_equal(out, pixels)
    assert sprite.selection_mask is not None
    assert np.array_equal(sprite.selection_mask, mask)


def test_autocrop_empty_sprite_raises():
    sprite = _make_sprite(8, 8)
    # No pixels written ⇒ all cels None / fully transparent.
    with pytest.raises(ValueError):
        AutocropCommand(sprite)


def test_autocrop_all_transparent_raises():
    sprite = _make_sprite(8, 8)
    sprite.set_cel_pixels(0, 0, np.zeros((8, 8, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        AutocropCommand(sprite)


def test_autocrop_already_tight_raises():
    sprite = _make_sprite(4, 4)
    pixels = np.full((4, 4, 4), 255, dtype=np.uint8)
    sprite.set_cel_pixels(0, 0, pixels)
    with pytest.raises(ValueError):
        AutocropCommand(sprite)


def test_autocrop_description_includes_dimensions():
    sprite = _make_sprite(16, 16)
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[2:5, 3:8] = (255, 255, 255, 255)
    sprite.set_cel_pixels(0, 0, pixels)
    cmd = AutocropCommand(sprite)
    assert "5" in cmd.description and "3" in cmd.description
    assert "Autocrop" in cmd.description
