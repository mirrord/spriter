# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Animated GIF export and import (Phase 7).

Functions
---------
* :func:`export_gif` — export all frames of a sprite as an animated GIF
* :func:`import_gif` — load an animated GIF as a multi-frame Sprite
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageSequence

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


def import_gif(path: Union[str, Path]) -> Sprite:
    """Import an animated GIF as a multi-frame Sprite.

    Each GIF frame becomes a frame on a single ``"Background"`` layer.
    Per-frame display durations from the GIF metadata are preserved
    (defaulting to 100 ms when not specified).  Sub-rectangle frames are
    composited onto the full GIF canvas via Pillow's sequential RGBA
    conversion, so the resulting cels are always sized to match the sprite.

    Args:
        path: Path to the ``.gif`` file.

    Returns:
        A new :class:`~spriter.core.sprite.Sprite` containing one frame per
        GIF frame.

    Raises:
        ValueError: If the GIF contains no frames.
    """
    path = Path(path)
    with Image.open(str(path)) as img:
        frames_rgba: list[np.ndarray] = []
        durations: list[int] = []
        for pil_frame in ImageSequence.Iterator(img):
            rgba = pil_frame.convert("RGBA")
            frames_rgba.append(np.array(rgba, dtype=np.uint8))
            duration = int(pil_frame.info.get("duration", 100) or 100)
            durations.append(max(duration, 1))
        size = img.size

    if not frames_rgba:
        raise ValueError("GIF has no frames to import.")

    w, h = size
    sprite = Sprite(w, h)
    sprite.add_layer("Background")
    for fi, (pixels, dur) in enumerate(zip(frames_rgba, durations)):
        sprite.add_frame(duration_ms=dur)
        sprite.set_cel_pixels(0, fi, pixels)
    return sprite
