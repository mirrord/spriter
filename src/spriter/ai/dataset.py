# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Automatic training-data harvesting for the next-frame predictor.

Frame data is gathered straight from the project's own animations so
fine-tuning needs no labelling or configuration: every non-empty composited
frame becomes a training image, and every consecutive pair becomes an
``(input, target)`` transition example.  Nothing here imports ``torch`` or
``diffusers``, so it is unit-testable without the optional ML dependencies.
"""

from __future__ import annotations

import numpy as np

from ..core.compositor import composite_frame
from ..core.sprite import Sprite


def _has_content(frame: np.ndarray) -> bool:
    """Return ``True`` when *frame* has at least one non-transparent pixel."""
    return bool((frame[..., 3] > 0).any())


def harvest_frames(sprite: Sprite) -> list[np.ndarray]:
    """Return every non-empty composited frame across all of *sprite*'s timelines.

    Used to fine-tune the predictor on the project's own appearance/style.  The
    sprite's active timeline is restored before returning.

    Args:
        sprite: The sprite document to harvest from.

    Returns:
        A list of ``H×W×4`` ``uint8`` RGBA frames (transparent frames omitted).
    """
    frames: list[np.ndarray] = []
    original = sprite.active_timeline_index
    try:
        for timeline_index in range(sprite.timeline_count):
            sprite.set_active_timeline(timeline_index)
            for frame_index in range(sprite.frame_count):
                frame = composite_frame(sprite, frame_index)
                if _has_content(frame):
                    frames.append(frame)
    finally:
        sprite.set_active_timeline(original)
    return frames


def harvest_frame_pairs(sprite: Sprite) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(frame, next_frame)`` transition pairs across all timelines.

    Pairs are drawn only within a single timeline (never across an animation
    boundary).  A pair is skipped when either frame is empty or when the two
    frames are identical, since neither teaches a useful transition.  The
    sprite's active timeline is restored before returning.

    Args:
        sprite: The sprite document to harvest from.

    Returns:
        A list of ``(input, target)`` RGBA frame pairs.
    """
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    original = sprite.active_timeline_index
    try:
        for timeline_index in range(sprite.timeline_count):
            sprite.set_active_timeline(timeline_index)
            count = sprite.frame_count
            for frame_index in range(count - 1):
                current = composite_frame(sprite, frame_index)
                nxt = composite_frame(sprite, frame_index + 1)
                if not (_has_content(current) and _has_content(nxt)):
                    continue
                if np.array_equal(current, nxt):
                    continue
                pairs.append((current, nxt))
    finally:
        sprite.set_active_timeline(original)
    return pairs
