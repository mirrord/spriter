# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""PNG export and import helpers (Phase 7).

Functions
---------
* :func:`export_frame`      — write one composited frame as a PNG file
* :func:`export_all_frames` — write every frame as individual numbered PNGs
* :func:`import_png`        — load any Pillow-readable image as a new Sprite
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image

from ..core.compositor import composite_frame
from ..core.frame import Cel
from ..core.layer import Layer
from ..core.sprite import Sprite


def export_frame(
    sprite: Sprite,
    frame_index: int,
    path: Union[str, Path],
) -> None:
    """Export a single composited frame as a PNG file.

    All visible layers are composited before writing.

    Args:
        sprite: Source sprite document.
        frame_index: Index of the frame to export (0-based).
        path: Destination file path.
    """
    path = Path(path)
    composite = composite_frame(sprite, frame_index)
    img = Image.fromarray(composite, mode="RGBA")
    img.save(str(path), format="PNG")


def export_all_frames(
    sprite: Sprite,
    dir_path: Union[str, Path],
    prefix: str = "frame",
) -> List[Path]:
    """Export every frame of *sprite* as individually numbered PNG files.

    Files are named ``{prefix}_0000.png``, ``{prefix}_0001.png``, …

    Args:
        sprite: Source sprite document.
        dir_path: Output directory (created if it does not exist).
        prefix: Filename prefix before the zero-padded frame number.

    Returns:
        Ordered list of :class:`~pathlib.Path` objects that were written.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    pad = len(str(sprite.frame_count - 1)) if sprite.frame_count > 1 else 4
    pad = max(pad, 4)

    paths: List[Path] = []
    for fi in range(sprite.frame_count):
        name = f"{prefix}_{fi:0{pad}d}.png"
        out = dir_path / name
        export_frame(sprite, fi, out)
        paths.append(out)
    return paths


def import_png(path: Union[str, Path]) -> Sprite:
    """Import a PNG (or any Pillow-supported format) as a new single-frame Sprite.

    The image is placed on a single layer named ``"Background"`` in a 1-frame
    sprite sized to match the image.

    Args:
        path: Path to the image file.

    Returns:
        A new :class:`~spriter.core.sprite.Sprite` containing the image.
    """
    path = Path(path)
    img = Image.open(str(path)).convert("RGBA")
    w, h = img.size
    pixels = np.array(img, dtype=np.uint8)

    sprite = Sprite(w, h)
    sprite.add_layer("Background")
    sprite.add_frame()
    sprite.set_cel_pixels(0, 0, pixels)
    return sprite
