# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Sprite-sheet packer and unpacker (Phase 7).

Functions
---------
* :func:`export_sheet`  — pack all frames into a single image file
* :func:`export_atlas`  — pack frames + write a JSON atlas
* :func:`import_sheet`  — split a sprite sheet into frames of a new Sprite

Enums
-----
* :class:`SheetLayout`  — HORIZONTAL, VERTICAL, GRID
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..core.compositor import composite_frame
from ..core.sprite import Sprite


class SheetLayout(Enum):
    """How frames are arranged in the sprite sheet."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    GRID = "grid"


def _get_frame_images(sprite: Sprite) -> List[np.ndarray]:
    """Return composited RGBA arrays for every frame."""
    return [composite_frame(sprite, fi) for fi in range(sprite.frame_count)]


def _sheet_dimensions(
    frame_w: int,
    frame_h: int,
    n_frames: int,
    layout: SheetLayout,
    cols: int,
    padding: int,
) -> Tuple[int, int, int, int]:
    """Return ``(sheet_w, sheet_h, actual_cols, actual_rows)`` for a layout."""
    if layout == SheetLayout.HORIZONTAL:
        actual_cols = n_frames
        actual_rows = 1
    elif layout == SheetLayout.VERTICAL:
        actual_cols = 1
        actual_rows = n_frames
    else:  # GRID
        actual_cols = max(1, cols if cols > 0 else int(n_frames**0.5 + 0.5))
        actual_rows = (n_frames + actual_cols - 1) // actual_cols

    sheet_w = actual_cols * frame_w + (actual_cols + 1) * padding
    sheet_h = actual_rows * frame_h + (actual_rows + 1) * padding
    return sheet_w, sheet_h, actual_cols, actual_rows


def export_sheet(
    sprite: Sprite,
    path: Union[str, Path],
    *,
    layout: SheetLayout = SheetLayout.HORIZONTAL,
    cols: int = 0,
    padding: int = 0,
) -> None:
    """Export all frames of *sprite* as a single sprite-sheet image.

    Args:
        sprite: Source sprite document.
        path: Output image path (format inferred from extension; PNG recommended).
        layout: Frame arrangement — HORIZONTAL, VERTICAL, or GRID.
        cols: Number of columns for GRID layout (0 = auto square).
        padding: Pixel gap between and around each frame.
    """
    path = Path(path)
    if sprite.frame_count == 0:
        raise ValueError("Sprite has no frames to export.")

    fw, fh = sprite.width, sprite.height
    n = sprite.frame_count
    sheet_w, sheet_h, actual_cols, _ = _sheet_dimensions(
        fw, fh, n, layout, cols, padding
    )

    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    frames = _get_frame_images(sprite)

    for fi, frame_pixels in enumerate(frames):
        col = fi % actual_cols
        row = fi // actual_cols
        x = padding + col * (fw + padding)
        y = padding + row * (fh + padding)
        sheet[y : y + fh, x : x + fw] = frame_pixels

    img = Image.fromarray(sheet, mode="RGBA")
    img.save(str(path))


def export_atlas(
    sprite: Sprite,
    sheet_path: Union[str, Path],
    atlas_path: Union[str, Path],
    *,
    layout: SheetLayout = SheetLayout.HORIZONTAL,
    cols: int = 0,
    padding: int = 0,
) -> Dict:
    """Export a sprite sheet and an accompanying JSON atlas.

    The JSON atlas format is compatible with common texture-packer tools::

        {
            "meta": {
                "image": "sheet.png",
                "size": {"w": 128, "h": 16},
                "scale": "1"
            },
            "frames": {
                "frame_0000": {
                    "frame": {"x": 0, "y": 0, "w": 16, "h": 16},
                    "duration": 100
                },
                ...
            }
        }

    Args:
        sprite: Source sprite document.
        sheet_path: Output image path.
        atlas_path: Output JSON path.
        layout: Frame arrangement.
        cols: Grid columns (GRID layout only; 0 = auto).
        padding: Pixel gap around/between frames.

    Returns:
        The atlas data structure that was written to *atlas_path*.
    """
    sheet_path = Path(sheet_path)
    atlas_path = Path(atlas_path)

    fw, fh = sprite.width, sprite.height
    n = sprite.frame_count
    sheet_w, sheet_h, actual_cols, _ = _sheet_dimensions(
        fw, fh, n, layout, cols, padding
    )

    # Build atlas before exporting so we can return it.
    atlas: Dict = {
        "meta": {
            "image": sheet_path.name,
            "size": {"w": sheet_w, "h": sheet_h},
            "scale": "1",
        },
        "frames": {},
    }
    for fi in range(n):
        col = fi % actual_cols
        row = fi // actual_cols
        x = padding + col * (fw + padding)
        y = padding + row * (fh + padding)
        name = f"frame_{fi:04d}"
        atlas["frames"][name] = {
            "frame": {"x": x, "y": y, "w": fw, "h": fh},
            "duration": sprite.frames[fi].duration_ms,
        }

    export_sheet(sprite, sheet_path, layout=layout, cols=cols, padding=padding)
    atlas_path.write_text(
        json.dumps(atlas, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return atlas


def import_sheet(
    path: Union[str, Path],
    frame_width: int,
    frame_height: int,
    *,
    padding: int = 0,
) -> Sprite:
    """Import a sprite sheet image as a new multi-frame Sprite.

    Frames are read left-to-right, top-to-bottom.  Partial cells at the
    right/bottom edges are ignored.

    Args:
        path: Path to the sprite sheet image.
        frame_width: Width of each frame cell in pixels.
        frame_height: Height of each frame cell in pixels.
        padding: Pixel gap between frame cells (same as used during export).

    Returns:
        A new :class:`~spriter.core.sprite.Sprite` with one layer and
        one frame per cell found in the sheet.
    """
    path = Path(path)
    img = Image.open(str(path)).convert("RGBA")
    sheet_w, sheet_h = img.size
    arr = np.array(img, dtype=np.uint8)

    step_x = frame_width + padding
    step_y = frame_height + padding
    cols = (sheet_w - padding) // step_x
    rows = (sheet_h - padding) // step_y

    if cols <= 0 or rows <= 0:
        raise ValueError(
            f"Sheet size {sheet_w}×{sheet_h} is too small for "
            f"frame size {frame_width}×{frame_height} with padding={padding}."
        )

    sprite = Sprite(frame_width, frame_height)
    sprite.add_layer("Background")

    frame_count = cols * rows
    for _ in range(frame_count):
        sprite.add_frame()

    li = 0
    fi = 0
    for row in range(rows):
        for col in range(cols):
            x = padding + col * step_x
            y = padding + row * step_y
            cell = arr[y : y + frame_height, x : x + frame_width].copy()
            sprite.set_cel_pixels(li, fi, cell)
            fi += 1

    return sprite
