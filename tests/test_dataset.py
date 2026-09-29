# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for automatic training-data harvesting (pure, no ML dependencies)."""

from __future__ import annotations

import numpy as np

from spriter.core.sprite import Sprite


def _solid(w, h, color):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:] = color
    return arr


def _sprite_with_frames(frames):
    """Build an 8×8 single-layer sprite whose frames carry *frames* pixels."""
    sprite = Sprite(8, 8)
    sprite.add_layer("Layer")
    for i, pixels in enumerate(frames):
        sprite.add_frame()
        sprite.set_cel_pixels(0, i, pixels)
    return sprite


class TestHarvestFrames:
    def test_collects_non_empty_frames(self):
        from spriter.ai.dataset import harvest_frames

        red = _solid(8, 8, (255, 0, 0, 255))
        blue = _solid(8, 8, (0, 0, 255, 255))
        sprite = _sprite_with_frames([red, blue])
        frames = harvest_frames(sprite)
        assert len(frames) == 2

    def test_skips_transparent_frames(self):
        from spriter.ai.dataset import harvest_frames

        red = _solid(8, 8, (255, 0, 0, 255))
        empty = np.zeros((8, 8, 4), dtype=np.uint8)
        sprite = _sprite_with_frames([red, empty])
        frames = harvest_frames(sprite)
        assert len(frames) == 1

    def test_restores_active_timeline(self):
        from spriter.ai.dataset import harvest_frames

        red = _solid(8, 8, (255, 0, 0, 255))
        sprite = _sprite_with_frames([red])
        sprite.add_timeline("Animation 2")
        sprite.set_active_timeline(1)
        harvest_frames(sprite)
        assert sprite.active_timeline_index == 1

    def test_spans_multiple_timelines(self):
        from spriter.ai.dataset import harvest_frames

        red = _solid(8, 8, (255, 0, 0, 255))
        sprite = _sprite_with_frames([red])
        sprite.add_timeline("Animation 2")
        sprite.set_active_timeline(1)
        sprite.set_cel_pixels(0, 0, _solid(8, 8, (0, 255, 0, 255)))
        sprite.set_active_timeline(0)
        frames = harvest_frames(sprite)
        assert len(frames) == 2


class TestHarvestFramePairs:
    def test_consecutive_pairs(self):
        from spriter.ai.dataset import harvest_frame_pairs

        red = _solid(8, 8, (255, 0, 0, 255))
        green = _solid(8, 8, (0, 255, 0, 255))
        blue = _solid(8, 8, (0, 0, 255, 255))
        sprite = _sprite_with_frames([red, green, blue])
        pairs = harvest_frame_pairs(sprite)
        assert len(pairs) == 2
        assert np.array_equal(pairs[0][0], red)
        assert np.array_equal(pairs[0][1], green)

    def test_skips_identical_pairs(self):
        from spriter.ai.dataset import harvest_frame_pairs

        red = _solid(8, 8, (255, 0, 0, 255))
        sprite = _sprite_with_frames([red, red.copy()])
        assert harvest_frame_pairs(sprite) == []

    def test_skips_pairs_with_empty_frame(self):
        from spriter.ai.dataset import harvest_frame_pairs

        red = _solid(8, 8, (255, 0, 0, 255))
        empty = np.zeros((8, 8, 4), dtype=np.uint8)
        sprite = _sprite_with_frames([red, empty])
        assert harvest_frame_pairs(sprite) == []

    def test_no_pairs_across_timeline_boundary(self):
        from spriter.ai.dataset import harvest_frame_pairs

        red = _solid(8, 8, (255, 0, 0, 255))
        sprite = _sprite_with_frames([red])  # 1 frame, no intra-timeline pair
        sprite.add_timeline("Animation 2")
        sprite.set_active_timeline(1)
        sprite.set_cel_pixels(0, 0, _solid(8, 8, (0, 255, 0, 255)))
        sprite.set_active_timeline(0)
        assert harvest_frame_pairs(sprite) == []
