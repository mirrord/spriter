# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Sprite-sheet packer and unpacker (Phase 7).

Functions
---------
* :func:`export_sheet`  — pack all frames into a single image file; sprites
  with multiple animation timelines are packed one row per animation
* :func:`export_atlas`  — pack frames + write a JSON atlas
* :func:`import_sheet`  — split a sprite sheet into frames of a new Sprite,
  optionally splitting each grid row into its own animation timeline
* :func:`import_sheet_auto`  — detect irregularly spaced frames and centre them,
  optionally splitting each detected row band into its own animation timeline
* :func:`estimate_sheet_layout`  — guess frame size + padding from a sheet image
* :func:`detect_background_color`  — detect a solid background colour from the
  sheet border
* :func:`remove_background`  — erase a colour to transparency across all cels
* :func:`split_background`  — move a background colour onto its own layer

Enums
-----
* :class:`SheetLayout`  — HORIZONTAL, VERTICAL, GRID
"""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from ..core.compositor import composite_frame
from ..core.frame import Cel
from ..core.layer import Layer
from ..core.sprite import Sprite


class SheetLayout(Enum):
    """How frames are arranged in the sprite sheet."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    GRID = "grid"


def _get_frame_images(sprite: Sprite) -> list[np.ndarray]:
    """Return composited RGBA arrays for every frame of the active timeline."""
    return [composite_frame(sprite, fi) for fi in range(sprite.frame_count)]


def _get_timeline_frame_images(sprite: Sprite, timeline_index: int) -> list[np.ndarray]:
    """Return composited RGBA arrays for every frame of a specific timeline."""
    saved = sprite.active_timeline_index
    sprite.set_active_timeline(timeline_index)
    try:
        images = [composite_frame(sprite, fi) for fi in range(sprite.frame_count)]
    finally:
        sprite.set_active_timeline(saved)
    return images


def _sheet_dimensions(
    frame_w: int,
    frame_h: int,
    n_frames: int,
    layout: SheetLayout,
    cols: int,
    padding: int,
) -> tuple[int, int, int, int]:
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
    path: str | Path,
    *,
    layout: SheetLayout = SheetLayout.HORIZONTAL,
    cols: int = 0,
    padding: int = 0,
) -> None:
    """Export all frames of *sprite* as a single sprite-sheet image.

    Args:
        sprite: Source sprite document.
        path: Output image path (format inferred from extension; PNG recommended).
        layout: Frame arrangement — HORIZONTAL, VERTICAL, or GRID.  Ignored when
            the sprite has multiple animation timelines (each animation is then
            packed onto its own row).
        cols: Number of columns for GRID layout (0 = auto square).
        padding: Pixel gap between and around each frame.
    """
    path = Path(path)

    # Multiple animations → one row per animation.
    if sprite.timeline_count > 1:
        _export_multi_timeline_sheet(sprite, path, padding=padding)
        return

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


def _export_multi_timeline_sheet(
    sprite: Sprite,
    path: Path,
    *,
    padding: int = 0,
) -> None:
    """Pack each animation timeline onto its own row of the sheet.

    Rows correspond to animations (top-to-bottom); the grid width equals the
    longest animation.  Shorter animations are left-aligned and their trailing
    cells left transparent.
    """
    fw, fh = sprite.width, sprite.height
    rows = sprite.timeline_count
    per_row = [
        _get_timeline_frame_images(sprite, ti) for ti in range(rows)
    ]
    cols = max((len(imgs) for imgs in per_row), default=0)
    if cols == 0:
        raise ValueError("Sprite has no frames to export.")

    sheet_w = cols * fw + (cols + 1) * padding
    sheet_h = rows * fh + (rows + 1) * padding
    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)

    for row, imgs in enumerate(per_row):
        for col, frame_pixels in enumerate(imgs):
            x = padding + col * (fw + padding)
            y = padding + row * (fh + padding)
            sheet[y : y + fh, x : x + fw] = frame_pixels

    img = Image.fromarray(sheet, mode="RGBA")
    img.save(str(path))


def export_atlas(
    sprite: Sprite,
    sheet_path: str | Path,
    atlas_path: str | Path,
    *,
    layout: SheetLayout = SheetLayout.HORIZONTAL,
    cols: int = 0,
    padding: int = 0,
) -> dict:
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

    if sprite.timeline_count > 1:
        multi_atlas = _build_multi_timeline_atlas(sprite, sheet_path, padding=padding)
        export_sheet(sprite, sheet_path, layout=layout, cols=cols, padding=padding)
        atlas_path.write_text(
            json.dumps(multi_atlas, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return multi_atlas

    fw, fh = sprite.width, sprite.height
    n = sprite.frame_count
    sheet_w, sheet_h, actual_cols, _ = _sheet_dimensions(
        fw, fh, n, layout, cols, padding
    )

    # Build atlas before exporting so we can return it.
    atlas: dict = {
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


def _build_multi_timeline_atlas(
    sprite: Sprite,
    sheet_path: Path,
    *,
    padding: int = 0,
) -> dict:
    """Build an atlas dict for the row-per-animation sheet layout."""
    fw, fh = sprite.width, sprite.height
    rows = sprite.timeline_count
    timelines = sprite.timelines
    cols = max((tl.frame_count for tl in timelines), default=0)
    sheet_w = cols * fw + (cols + 1) * padding
    sheet_h = rows * fh + (rows + 1) * padding

    atlas: dict = {
        "meta": {
            "image": sheet_path.name,
            "size": {"w": sheet_w, "h": sheet_h},
            "scale": "1",
        },
        "frames": {},
    }
    for row, timeline in enumerate(timelines):
        for fi, frame in enumerate(timeline._frames):
            x = padding + fi * (fw + padding)
            y = padding + row * (fh + padding)
            name = f"{timeline.name}_{fi:04d}"
            atlas["frames"][name] = {
                "frame": {"x": x, "y": y, "w": fw, "h": fh},
                "duration": frame.duration_ms,
                "animation": timeline.name,
            }
    return atlas


def import_sheet(
    path: str | Path,
    frame_width: int,
    frame_height: int,
    *,
    padding: int = 0,
    split_rows: bool = False,
) -> Sprite:
    """Import a sprite sheet image as a new multi-frame Sprite.

    Frames are read left-to-right, top-to-bottom.  Partial cells at the
    right/bottom edges are ignored.

    Args:
        path: Path to the sprite sheet image.
        frame_width: Width of each frame cell in pixels.
        frame_height: Height of each frame cell in pixels.
        padding: Pixel gap between frame cells (same as used during export).
        split_rows: When True, each grid row becomes a separate animation
            timeline instead of a single continuous animation.

    Returns:
        A new :class:`~spriter.core.sprite.Sprite` with one layer.  With
        *split_rows* enabled it has one timeline per row; otherwise a single
        timeline with one frame per cell.
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
    li = 0

    if split_rows and rows > 1:
        for row in range(rows):
            if row == 0:
                sprite.rename_timeline(0, "Animation 1")
            else:
                sprite.add_timeline(f"Animation {row + 1}")
            sprite.set_active_timeline(row)
            while sprite.frame_count < cols:
                sprite.add_frame()
            for col in range(cols):
                x = padding + col * step_x
                y = padding + row * step_y
                cell = arr[y : y + frame_height, x : x + frame_width].copy()
                sprite.set_cel_pixels(li, col, cell)
        sprite.set_active_timeline(0)
        return sprite

    frame_count = cols * rows
    for _ in range(frame_count):
        sprite.add_frame()

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
# Background detection & handling
# ---------------------------------------------------------------------------

BackgroundColor = tuple[int, int, int, int]


def detect_background_color(
    source: str | Path | np.ndarray | Image.Image,
    *,
    coverage: float = 0.5,
) -> BackgroundColor | None:
    """Detect a solid background colour in a sprite sheet.

    Samples the border pixels (outermost rows and columns) and returns the
    modal RGBA colour when it is opaque and accounts for at least *coverage*
    of the border.  Returns ``None`` when the border is already transparent or
    has no dominant colour — i.e. there is no removable background.

    Args:
        source: Path to an image file, a PIL Image, or an ``H×W×{3,4}``
            ``uint8`` NumPy array.
        coverage: Minimum fraction of border pixels the dominant colour must
            occupy to be considered a background.

    Returns:
        The detected background colour as an ``(r, g, b, a)`` tuple, or
        ``None`` if no background is found.
    """
    arr = _load_rgba(source)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return None
    border = np.concatenate(
        [
            arr[0, :, :].reshape(-1, 4),
            arr[-1, :, :].reshape(-1, 4),
            arr[:, 0, :].reshape(-1, 4),
            arr[:, -1, :].reshape(-1, 4),
        ],
        axis=0,
    )
    colors, counts = np.unique(border, axis=0, return_counts=True)
    top = int(np.argmax(counts))
    bg = colors[top]
    if int(bg[3]) == 0:  # border already transparent → nothing to remove
        return None
    if counts[top] / border.shape[0] < coverage:
        return None
    return (int(bg[0]), int(bg[1]), int(bg[2]), int(bg[3]))


def _color_match_mask(pixels: np.ndarray, color: BackgroundColor) -> np.ndarray:
    """Boolean ``H×W`` mask of pixels whose RGB equals *color*."""
    return (
        (pixels[..., 0] == color[0])
        & (pixels[..., 1] == color[1])
        & (pixels[..., 2] == color[2])
    )


def remove_background(sprite: Sprite, color: BackgroundColor) -> None:
    """Make every pixel matching *color* fully transparent, across all cels.

    Args:
        sprite: The sprite to modify in place.
        color: The background colour to erase (RGB channels are matched).
    """
    for timeline in sprite.timelines:
        for cel in timeline._cels.values():
            if cel.pixels is None:
                continue
            match = _color_match_mask(cel.pixels, color)
            if match.any():
                cel.pixels[match] = 0


def split_background(sprite: Sprite, color: BackgroundColor) -> None:
    """Separate the background colour onto its own layer beneath the content.

    The background colour is erased from every existing layer and a new opaque
    "Background" layer filled with *color* is inserted at the bottom of the
    stack (in every timeline).  The imported content layer is renamed
    "Foreground" when it still carries the default import name.

    Args:
        sprite: The sprite to modify in place.
        color: The detected background colour.
    """
    remove_background(sprite, color)
    fill = np.empty((sprite.height, sprite.width, 4), dtype=np.uint8)
    fill[..., 0] = color[0]
    fill[..., 1] = color[1]
    fill[..., 2] = color[2]
    fill[..., 3] = 255
    for timeline in sprite.timelines:
        shifted: dict[tuple[int, int], Cel] = {}
        for (li, fi), cel in timeline._cels.items():
            shifted[(li + 1, fi)] = cel
        for fi in range(len(timeline._frames)):
            shifted[(0, fi)] = Cel(fill.copy())
        timeline._cels = shifted
    sprite._layers.insert(0, Layer("Background"))
    if len(sprite._layers) >= 2 and sprite._layers[1].name == "Background":
        sprite._layers[1].name = "Foreground"


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
_PREFERRED_SIZES: tuple[int, ...] = (8, 16, 24, 32, 48, 64, 96, 128)


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


def _period_from_mask(has_content: np.ndarray) -> int | None:
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


def _load_rgba(source: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    """Load *source* to an ``H×W×4`` ``uint8`` RGBA array."""
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
        elif arr.shape[-1] == 3:
            alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=-1)
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if isinstance(source, Image.Image):
        return np.asarray(source.convert("RGBA"), dtype=np.uint8)
    with Image.open(str(source)) as img:
        return np.asarray(img.convert("RGBA"), dtype=np.uint8)


def _content_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean ``H×W`` mask of "is content" pixels.

    Uses the alpha channel when the sheet has any transparency; otherwise
    falls back to distance from the most common corner colour (background).
    A fully-opaque sheet whose corners disagree is treated as all content.
    """
    h, w = arr.shape[:2]
    alpha = arr[..., 3]
    if alpha.min() < 255:
        return alpha > 0
    corners = [
        tuple(arr[0, 0, :3]),
        tuple(arr[0, -1, :3]),
        tuple(arr[-1, 0, :3]),
        tuple(arr[-1, -1, :3]),
    ]
    bg, bg_count = Counter(corners).most_common(1)[0]
    if bg_count >= 2:
        bg_arr = np.array(bg, dtype=np.uint8)
        return (arr[..., :3] != bg_arr).any(axis=-1)
    return np.ones((h, w), dtype=bool)


def _label_components(
    mask: np.ndarray, *, connectivity: int = 8
) -> tuple[np.ndarray, int]:
    """Label connected components of a boolean mask (pure NumPy union-find).

    Args:
        mask: Boolean ``H×W`` occupancy mask.
        connectivity: ``4`` (orthogonal) or ``8`` (orthogonal + diagonal).

    Returns:
        ``(labels, count)`` where ``labels`` is an ``H×W`` ``int`` array with
        ``0`` for background and ``1..count`` for each component, and
        ``count`` is the number of components found.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    idx = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.nonzero(mask)
    k = ys.size
    if k == 0:
        return np.zeros((h, w), dtype=np.int64), 0
    idx[ys, xs] = np.arange(k)

    parent = np.arange(k, dtype=np.int64)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union_pairs(a: np.ndarray, b: np.ndarray) -> None:
        for ia, ib in zip(a.tolist(), b.tolist()):
            ra, rb = find(ia), find(ib)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    # Orthogonal neighbours (right, down).
    pair = mask[:, :-1] & mask[:, 1:]
    union_pairs(idx[:, :-1][pair], idx[:, 1:][pair])
    pair = mask[:-1, :] & mask[1:, :]
    union_pairs(idx[:-1, :][pair], idx[1:, :][pair])

    if connectivity == 8:
        # Diagonal neighbours (down-right, down-left).
        pair = mask[:-1, :-1] & mask[1:, 1:]
        union_pairs(idx[:-1, :-1][pair], idx[1:, 1:][pair])
        pair = mask[:-1, 1:] & mask[1:, :-1]
        union_pairs(idx[:-1, 1:][pair], idx[1:, :-1][pair])

    roots = np.array([find(i) for i in range(k)], dtype=np.int64)
    uniq, remapped = np.unique(roots, return_inverse=True)
    labels = np.zeros((h, w), dtype=np.int64)
    labels[ys, xs] = remapped + 1
    return labels, int(uniq.size)


def import_sheet_auto(
    source: str | Path | np.ndarray | Image.Image,
    *,
    split_rows: bool = False,
) -> Sprite:
    """Import a sprite sheet with inconsistent frame spacing.

    Detects individual frame candidates via 8-connected component labeling
    of a content mask, sizes every frame to the maximum candidate width and
    height, and centres each candidate (without stretching) on its own
    transparent frame.  Candidates are ordered left-to-right within a row,
    rows top-to-bottom.

    Args:
        source: Path to an image file, a PIL Image, or an ``H×W×{3,4}``
            ``uint8`` NumPy array.
        split_rows: When True, each detected row band becomes a separate
            animation timeline instead of a single continuous animation.

    Returns:
        A new :class:`~spriter.core.sprite.Sprite` with one layer and one
        frame per detected candidate.

    Raises:
        ValueError: If no frame candidates are detected.
    """
    arr = _load_rgba(source)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Sprite sheet image is empty.")

    mask = _content_mask(arr)
    labels, count = _label_components(mask, connectivity=8)
    if count == 0:
        raise ValueError("No sprite frames detected in the sheet.")

    # Bounding box per component (label ids are 1..count).
    boxes: list[tuple[int, int, int, int]] = []  # (y0, y1, x0, x1) inclusive
    for lbl in range(1, count + 1):
        cys, cxs = np.nonzero(labels == lbl)
        boxes.append((int(cys.min()), int(cys.max()), int(cxs.min()), int(cxs.max())))

    frame_w = max(x1 - x0 + 1 for _, _, x0, x1 in boxes)
    frame_h = max(y1 - y0 + 1 for y0, y1, _, _ in boxes)

    # Order: group into rows by vertical overlap, rows top-to-bottom,
    # left-to-right within each row.
    order = sorted(range(count), key=lambda i: (boxes[i][0], boxes[i][2]))
    rows: list[list[int]] = []
    row_bottom: list[int] = []
    for i in order:
        y0, y1, _, _ = boxes[i]
        placed = False
        for r, bottom in enumerate(row_bottom):
            if y0 <= bottom:  # vertical overlap with an existing row band
                rows[r].append(i)
                row_bottom[r] = max(bottom, y1)
                placed = True
                break
        if not placed:
            rows.append([i])
            row_bottom.append(y1)

    # Row bands sorted top-to-bottom, candidates within a band left-to-right.
    bands: list[list[int]] = [
        sorted(rows[r], key=lambda i: boxes[i][2])
        for r in sorted(
            range(len(rows)), key=lambda r: min(boxes[i][0] for i in rows[r])
        )
    ]

    def _cel_for(i: int) -> np.ndarray:
        y0, y1, x0, x1 = boxes[i]
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        crop = arr[y0 : y1 + 1, x0 : x1 + 1].copy()
        # Suppress pixels belonging to a different component (neighbour bleed).
        crop_labels = labels[y0 : y1 + 1, x0 : x1 + 1]
        other = (crop_labels != 0) & (crop_labels != (i + 1))
        crop[other] = 0
        cel = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
        oy = (frame_h - bh) // 2
        ox = (frame_w - bw) // 2
        cel[oy : oy + bh, ox : ox + bw] = crop
        return cel

    sprite = Sprite(frame_w, frame_h)
    sprite.add_layer("Background")

    if split_rows and len(bands) > 1:
        for r, band in enumerate(bands):
            if r == 0:
                sprite.rename_timeline(0, "Animation 1")
            else:
                sprite.add_timeline(f"Animation {r + 1}")
            sprite.set_active_timeline(r)
            while sprite.frame_count < len(band):
                sprite.add_frame()
            for fi, i in enumerate(band):
                sprite.set_cel_pixels(0, fi, _cel_for(i))
        sprite.set_active_timeline(0)
        return sprite

    ordered: list[int] = [i for band in bands for i in band]
    for _ in range(count):
        sprite.add_frame()
    for fi, i in enumerate(ordered):
        sprite.set_cel_pixels(0, fi, _cel_for(i))

    return sprite


def estimate_sheet_layout(
    source: str | Path | np.ndarray | Image.Image,
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
    arr = _load_rgba(source)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return EstimatedLayout(max(w, 1), max(h, 1), 0)

    mask = _content_mask(arr)
    col_has = np.asarray(mask.any(axis=0))
    row_has = np.asarray(mask.any(axis=1))

    fw = _period_from_mask(col_has) or _largest_preferred_divisor(w)
    fh = _period_from_mask(row_has) or _largest_preferred_divisor(h)

    fw = max(1, min(fw, w))
    fh = max(1, min(fh, h))

    return EstimatedLayout(int(fw), int(fh), 0)
