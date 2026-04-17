# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Animated GIF export (Phase 7).

Functions
---------
* :func:`export_gif` — export all frames of a sprite as an animated GIF
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from ..core.compositor import composite_frame
from ..core.sprite import Sprite


def export_gif(
    sprite: Sprite,
    path: Union[str, Path],
    *,
    loop: int = 0,
) -> None:
    """Export all frames of *sprite* as an animated GIF.

    Each frame's composited RGBA image is converted to palette (P) mode.
    Per-frame durations from the animation model are used.

    Args:
        sprite: Source sprite document.
        path: Destination ``.gif`` file path.
        loop: Number of animation loops.  ``0`` means loop forever (default).
    """
    path = Path(path)
    if sprite.frame_count == 0:
        raise ValueError("Sprite has no frames to export.")

    composites = [composite_frame(sprite, fi) for fi in range(sprite.frame_count)]
    durations = [sprite.frames[fi].duration_ms for fi in range(sprite.frame_count)]

    # Build a single wide RGBA image containing all frames side-by-side so we
    # can derive one consistent palette for the entire animation.
    all_pixels = np.concatenate(composites, axis=1)
    combined_rgba = Image.fromarray(all_pixels, mode="RGBA")
    # FASTOCTREE is the only quantization method that supports RGBA.
    combined_p = combined_rgba.quantize(colors=255, method=Image.Quantize.FASTOCTREE)

    fw = sprite.width
    fh = sprite.height
    pil_frames: list = []
    for fi in range(sprite.frame_count):
        frame_p = combined_p.crop((fi * fw, 0, (fi + 1) * fw, fh))
        pil_frames.append(frame_p)

    pil_frames[0].save(
        str(path),
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=loop,
        disposal=2,
        optimize=False,
    )
