# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Sprite-sheet packer and unpacker (Phase 7).

Functions
---------
* :func:`export_sheet`  — pack all frames into a single image file
* :func:`export_atlas`  — pack frames + write a JSON atlas
* :func:`import_sheet`  — split a sprite sheet into frames of a new Sprite
* :func:`estimate_sheet_layout`  — guess frame size + padding from a sheet image

Enums
-----
* :class:`SheetLayout`  — HORIZONTAL, VERTICAL, GRID
"""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

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


# ---------------------------------------------------------------------------
# Dimension estimation
# ---------------------------------------------------------------------------


class EstimatedLayout(NamedTuple):
    """Estimated frame dimensions for a sprite sheet.

    Attributes:
        frame_width:  Estimated width  of an individual frame cell, in pixels.
        frame_height: Estimated height of an individual frame cell, in pixels.
        padding:      Estimated pixel gap between frame cells (0 if none / undetectable).
    """

    frame_width: int
    frame_height: int
    padding: int


# Common pixel-art frame sizes, used as a fallback when the sheet has no
# detectable inter-sprite separation (uniform alpha + uniform colour).
_PREFERRED_SIZES: Tuple[int, ...] = (8, 16, 24, 32, 48, 64, 96, 128)


def _largest_preferred_divisor(extent: int) -> int:
    """Largest pixel-art-friendly divisor of *extent* yielding ≥ 2 frames."""
    if extent <= 0:
        return max(extent, 1)
    for size in sorted(_PREFERRED_SIZES, reverse=True):
        if size < extent and extent % size == 0:
            return size
    # Fall back to the largest divisor ≥ 4 that gives at least 2 frames.
    for size in range(extent // 2, 3, -1):
        if extent % size == 0:
            return size
    return extent


def _period_from_mask(has_content: np.ndarray) -> Optional[int]:
    """Estimate the frame stride along an axis from a boolean content mask.

    Returns the per-frame *stride* (i.e. frame size **including** any
    internal whitespace + inter-frame padding), or ``None`` when the mask
    has no detectable structure (fully True or fully False).

    Algorithm: find rising edges (False→True transitions) of the mask and
    take the modal distance between consecutive rising edges.  This is
    robust to sprites that don't fill their cells (centred sprites with
    transparent margins) — those margins look like inter-frame padding to
    a naive gap-based detector, but the rising-edge cadence recovers the
    true cell stride.
    """
    n = int(has_content.size)
    if n == 0:
        return None

    # Pad with a leading False so a mask that starts True still produces
    # a rising edge at position 0.
    padded = np.concatenate(([False], np.asarray(has_content, dtype=bool)))
    rises = np.flatnonzero(padded[1:] & ~padded[:-1])
    if rises.size == 0:
        # Mask is uniformly empty — no detectable structure.
        return None
    if rises.size == 1:
        # One filled region — assume a single frame spans this axis.
        return n

    diffs = np.diff(rises).tolist()
    stride, count = Counter(diffs).most_common(1)[0]
    if count * 2 < len(diffs):
        return None
    return int(stride)


def estimate_sheet_layout(
    source: Union[str, Path, np.ndarray, Image.Image],
) -> EstimatedLayout:
    """Estimate per-frame dimensions for a sprite sheet.

    Assumes **no padding between frames** (real-world pixel-art sheets pack
    frames flush; any reported dimension already includes whatever in-cell
    whitespace surrounds the sprite).  Detection uses fast NumPy projections
    of an occupancy mask — alpha channel when the sheet has transparency,
    otherwise a corner-sampled background colour.

    Algorithm (single-pass, O(W·H)):
        1. Load the image to an ``H×W×4`` ``uint8`` array.
        2. Build a per-pixel "is content" mask.
        3. Project to ``col_has = mask.any(axis=0)`` and
           ``row_has = mask.any(axis=1)``.
        4. For each axis, take the modal distance between successive
           rising edges of the mask — that is the frame stride.
        5. If an axis has no detectable structure (uniform fill / uniform
           empty), fall back to the largest pixel-art-friendly divisor.

    Padding is always reported as ``0``; the user can override in the UI.

    Args:
        source: Path to an image file, a PIL :class:`~PIL.Image.Image`, or
            a pre-loaded ``H×W×{3,4}`` ``uint8`` NumPy array.

    Returns:
        An :class:`EstimatedLayout` ``(frame_width, frame_height, padding=0)``.
    """
    # --- Load to an RGBA uint8 array -------------------------------------
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
        elif arr.shape[-1] == 3:
            alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=-1)
    elif isinstance(source, Image.Image):
        arr = np.asarray(source.convert("RGBA"))
    else:
        with Image.open(str(source)) as img:
            arr = np.asarray(img.convert("RGBA"))

    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return EstimatedLayout(max(w, 1), max(h, 1), 0)

    # --- Build occupancy mask -------------------------------------------
    alpha = arr[..., 3]
    if alpha.min() < 255:
        mask: np.ndarray = alpha > 0
    else:
        # Use distance from the most common corner colour as the signal.
        corners = [
            tuple(arr[0, 0, :3]),
            tuple(arr[0, -1, :3]),
            tuple(arr[-1, 0, :3]),
            tuple(arr[-1, -1, :3]),
        ]
        bg, bg_count = Counter(corners).most_common(1)[0]
        if bg_count >= 2:
            bg_arr = np.array(bg, dtype=np.uint8)
            mask = (arr[..., :3] != bg_arr).any(axis=-1)
        else:
            # No corner agrees: treat the whole sheet as filled and let
            # the divisor fallback decide.
            mask = np.ones((h, w), dtype=bool)

    col_has = np.asarray(mask.any(axis=0))
    row_has = np.asarray(mask.any(axis=1))

    fw = _period_from_mask(col_has) or _largest_preferred_divisor(w)
    fh = _period_from_mask(row_has) or _largest_preferred_divisor(h)

    fw = max(1, min(fw, w))
    fh = max(1, min(fh, h))

    return EstimatedLayout(int(fw), int(fh), 0)
