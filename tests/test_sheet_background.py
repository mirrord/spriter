# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for sprite-sheet background detection and handling."""

import numpy as np

from spriter.core.sprite import Sprite
from spriter.io.spritesheet import (
    detect_background_color,
    remove_background,
    split_background,
)

MAGENTA = (255, 0, 255, 255)


def _sprite_with_bg(width: int = 4, height: int = 4) -> Sprite:
    """4x4 sprite: magenta background with a single red content pixel."""
    s = Sprite(width, height)
    s.add_layer("Background")
    s.add_frame()
    px = np.zeros((height, width, 4), dtype=np.uint8)
    px[..., 0] = 255  # magenta fill
    px[..., 2] = 255
    px[..., 3] = 255
    px[1, 1] = (255, 0, 0, 255)  # red content pixel
    s.set_cel_pixels(0, 0, px)
    return s


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestDetectBackground:
    def test_detects_solid_opaque_background(self):
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[..., 0] = 255
        arr[..., 2] = 255
        arr[..., 3] = 255
        arr[6:10, 6:10] = (0, 128, 0, 255)  # a sprite blob in the middle
        assert detect_background_color(arr) == MAGENTA

    def test_transparent_border_returns_none(self):
        arr = np.zeros((16, 16, 4), dtype=np.uint8)  # fully transparent
        arr[6:10, 6:10] = (255, 0, 0, 255)
        assert detect_background_color(arr) is None

    def test_no_dominant_border_returns_none(self):
        # Border alternates between four colours so none reaches 50%.
        arr = np.zeros((8, 8, 4), dtype=np.uint8)
        arr[..., 3] = 255
        arr[0, ::2] = (255, 0, 0, 255)
        arr[0, 1::2] = (0, 255, 0, 255)
        arr[-1, ::2] = (0, 0, 255, 255)
        arr[-1, 1::2] = (255, 255, 0, 255)
        arr[:, 0] = (10, 20, 30, 255)
        arr[:, -1] = (40, 50, 60, 255)
        assert detect_background_color(arr) is None


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


class TestRemoveBackground:
    def test_background_pixels_become_transparent(self):
        s = _sprite_with_bg()
        remove_background(s, MAGENTA)
        cel = s.get_cel(0, 0).pixels
        # Magenta corner now transparent.
        assert tuple(cel[0, 0]) == (0, 0, 0, 0)
        # Red content preserved.
        assert tuple(cel[1, 1]) == (255, 0, 0, 255)

    def test_applies_to_all_timelines(self):
        s = _sprite_with_bg()
        s.add_timeline("run")
        s.set_active_timeline(1)
        px = np.zeros((4, 4, 4), dtype=np.uint8)
        px[..., 0] = 255
        px[..., 2] = 255
        px[..., 3] = 255
        s.set_cel_pixels(0, 0, px)
        s.set_active_timeline(0)
        remove_background(s, MAGENTA)
        s.set_active_timeline(1)
        assert not np.any(s.get_cel(0, 0).pixels[..., 3])


# ---------------------------------------------------------------------------
# Split into background + foreground
# ---------------------------------------------------------------------------


class TestSplitBackground:
    def test_adds_bottom_background_layer(self):
        s = _sprite_with_bg()
        split_background(s, MAGENTA)
        assert s.layer_count == 2
        assert s.layers[0].name == "Background"
        assert s.layers[1].name == "Foreground"

    def test_background_layer_is_solid_and_foreground_cleared(self):
        s = _sprite_with_bg()
        split_background(s, MAGENTA)
        bg = s.get_cel(0, 0).pixels  # bottom background layer
        fg = s.get_cel(1, 0).pixels  # foreground content layer
        assert np.all(bg == np.array(MAGENTA, dtype=np.uint8))
        # Foreground background pixels removed, content kept.
        assert tuple(fg[0, 0]) == (0, 0, 0, 0)
        assert tuple(fg[1, 1]) == (255, 0, 0, 255)

    def test_split_applies_to_all_timelines(self):
        s = _sprite_with_bg()
        s.add_timeline("run")
        split_background(s, MAGENTA)
        assert s.layer_count == 2
        for ti in range(s.timeline_count):
            s.set_active_timeline(ti)
            for fi in range(s.frame_count):
                assert np.all(
                    s.get_cel(0, fi).pixels == np.array(MAGENTA, dtype=np.uint8)
                )
