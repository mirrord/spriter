# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Rasterization primitives operating on NumPy RGBA pixel buffers.

All functions mutate the supplied *pixels* array in-place and accept
coordinates in pixel (integer) space.  No bounds-checking is performed on
individual pixel writes — callers should clip coordinates to the canvas
beforehand, or use the clipping variants ending in ``_clipped``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

# A color value is a 4-element sequence (R, G, B, A) with uint8 range 0–255.
Color = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Low-level pixel helpers
# ---------------------------------------------------------------------------


def set_pixel(pixels: np.ndarray, x: int, y: int, color: Color) -> None:
    """Write a single RGBA pixel if (x, y) is within the buffer bounds.

    Args:
        pixels: RGBA uint8 array of shape ``(H, W, 4)``.
        x: Column index.
        y: Row index.
        color: ``(R, G, B, A)`` tuple.
    """
    h, w = pixels.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        pixels[y, x] = color


def get_pixel(pixels: np.ndarray, x: int, y: int) -> Color:
    """Read a single RGBA pixel.

    Args:
        pixels: RGBA uint8 array of shape ``(H, W, 4)``.
        x: Column index.
        y: Row index.

    Returns:
        ``(R, G, B, A)`` tuple.

    Raises:
        IndexError: If (x, y) is outside the buffer bounds.
    """
    h, w = pixels.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise IndexError(f"Pixel ({x}, {y}) is out of bounds for {w}x{h} buffer")
    return tuple(int(v) for v in pixels[y, x])  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Bresenham line
# ---------------------------------------------------------------------------


def draw_line(
    pixels: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Color,
) -> List[Tuple[int, int]]:
    """Draw a 1-pixel-wide line using Bresenham's algorithm.

    Args:
        pixels: RGBA uint8 array of shape ``(H, W, 4)``.
        x0, y0: Start coordinate.
        x1, y1: End coordinate.
        color: ``(R, G, B, A)`` fill color.

    Returns:
        List of ``(x, y)`` pixel positions that were plotted.
    """
    plotted: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        set_pixel(pixels, x, y, color)
        plotted.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return plotted


def line_points(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Return the pixel coordinates for a Bresenham line without drawing.

    Args:
        x0, y0: Start coordinate.
        x1, y1: End coordinate.

    Returns:
        Ordered list of ``(x, y)`` positions.
    """
    pts: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pts


# ---------------------------------------------------------------------------
# Rectangle
# ---------------------------------------------------------------------------


def draw_rect(
    pixels: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: Color,
    *,
    filled: bool = False,
) -> None:
    """Draw an axis-aligned rectangle.

    Args:
        pixels: RGBA uint8 array of shape ``(H, W, 4)``.
        x, y: Top-left corner.
        width: Width in pixels.
        height: Height in pixels.
        color: ``(R, G, B, A)`` fill/stroke color.
        filled: If True, fill the interior; otherwise draw an outline only.
    """
    if width <= 0 or height <= 0:
        return
    x1, y1 = x + width - 1, y + height - 1
    if filled:
        h_buf, w_buf = pixels.shape[:2]
        cx0 = max(0, x)
        cy0 = max(0, y)
        cx1 = min(w_buf, x1 + 1)
        cy1 = min(h_buf, y1 + 1)
        if cx1 > cx0 and cy1 > cy0:
            pixels[cy0:cy1, cx0:cx1] = color
    else:
        draw_line(pixels, x, y, x1, y, color)  # top
        draw_line(pixels, x, y1, x1, y1, color)  # bottom
        draw_line(pixels, x, y, x, y1, color)  # left
        draw_line(pixels, x1, y, x1, y1, color)  # right


# ---------------------------------------------------------------------------
# Ellipse (midpoint algorithm)
# ---------------------------------------------------------------------------


def draw_ellipse(
    pixels: np.ndarray,
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    color: Color,
    *,
    filled: bool = False,
) -> None:
    """Draw an axis-aligned ellipse using the midpoint algorithm.

    Args:
        pixels: RGBA uint8 array of shape ``(H, W, 4)``.
        cx, cy: Centre coordinate.
        rx: Horizontal radius in pixels.
        ry: Vertical radius in pixels.
        color: ``(R, G, B, A)`` fill/stroke color.
        filled: If True, fill the interior.
    """
    if rx < 0 or ry < 0:
        return
    if rx == 0 and ry == 0:
        set_pixel(pixels, cx, cy, color)
        return

    def _plot4(px: int, py: int) -> None:
        if filled:
            draw_line(pixels, cx - px, cy + py, cx + px, cy + py, color)
            if py != 0:
                draw_line(pixels, cx - px, cy - py, cx + px, cy - py, color)
        else:
            set_pixel(pixels, cx + px, cy + py, color)
            set_pixel(pixels, cx - px, cy + py, color)
            set_pixel(pixels, cx + px, cy - py, color)
            set_pixel(pixels, cx - px, cy - py, color)

    x, y = 0, ry
    rx2, ry2 = rx * rx, ry * ry
    tworx2, twory2 = 2 * rx2, 2 * ry2
    p = round(ry2 - rx2 * ry + 0.25 * rx2)
    dx, dy = twory2 * x, tworx2 * y

    # Region 1
    while dx < dy:
        _plot4(x, y)
        x += 1
        dx += twory2
        if p < 0:
            p += ry2 + dx
        else:
            y -= 1
            dy -= tworx2
            p += ry2 + dx - dy

    # Region 2
    p = round(ry2 * (x + 0.5) ** 2 + rx2 * (y - 1) ** 2 - rx2 * ry2)
    while y >= 0:
        _plot4(x, y)
        y -= 1
        dy -= tworx2
        if p > 0:
            p += rx2 - dy
        else:
            x += 1
            dx += twory2
            p += rx2 - dy + dx


# ---------------------------------------------------------------------------
# Flood fill
# ---------------------------------------------------------------------------


def flood_fill(
    pixels: np.ndarray,
    x: int,
    y: int,
    fill_color: Color,
    *,
    connectivity: int = 4,
) -> int:
    """Scanline flood fill.

    Args:
        pixels: RGBA uint8 array of shape ``(H, W, 4)``.
        x, y: Seed coordinate.
        fill_color: ``(R, G, B, A)`` replacement color.
        connectivity: ``4`` (cardinal) or ``8`` (diagonal) neighbour connectivity.

    Returns:
        Number of pixels replaced.

    Raises:
        ValueError: If *connectivity* is not 4 or 8.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    h, w = pixels.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return 0

    target_color = tuple(int(v) for v in pixels[y, x])
    fill = tuple(int(v) for v in fill_color)
    if target_color == fill:
        return 0

    count = 0
    queue: deque[Tuple[int, int]] = deque()
    queue.append((x, y))
    visited = np.zeros((h, w), dtype=bool)
    visited[y, x] = True

    if connectivity == 4:
        neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        neighbours = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

    while queue:
        px, py = queue.popleft()
        current = tuple(int(v) for v in pixels[py, px])
        if current != target_color:
            continue
        pixels[py, px] = fill_color
        count += 1
        for nx, ny in ((px + dx, py + dy) for dx, dy in neighbours):
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))

    return count


# ---------------------------------------------------------------------------
# Rounded rectangle
# ---------------------------------------------------------------------------


def draw_rounded_rect(
    pixels: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: Color,
    *,
    corner_radius: int = 0,
    filled: bool = False,
) -> None:
    """Draw an axis-aligned rectangle with optional rounded corners.

    Args:
        pixels: RGBA uint8 array of shape ``(H, W, 4)``.
        x, y: Top-left corner.
        width: Width in pixels.
        height: Height in pixels.
        color: ``(R, G, B, A)`` fill/stroke color.
        corner_radius: Radius of the corner arcs.  Clamped to
            ``min(width, height) // 2``.
        filled: If True, fill the interior.
    """
    if width <= 0 or height <= 0:
        return
    r = max(0, min(corner_radius, width // 2, height // 2))
    if r == 0:
        draw_rect(pixels, x, y, width, height, color, filled=filled)
        return

    x1, y1 = x + width - 1, y + height - 1

    if filled:
        # Composite: 3 non-overlapping filled rectangles + 4 filled quarter-circles.
        # Central horizontal band.
        draw_rect(pixels, x + r, y, width - 2 * r, height, color, filled=True)
        # Left strip.
        draw_rect(pixels, x, y + r, r, height - 2 * r, color, filled=True)
        # Right strip.
        draw_rect(pixels, x1 - r + 1, y + r, r, height - 2 * r, color, filled=True)
        # Four corner circles (filled).
        draw_ellipse(pixels, x + r, y + r, r, r, color, filled=True)
        draw_ellipse(pixels, x1 - r, y + r, r, r, color, filled=True)
        draw_ellipse(pixels, x + r, y1 - r, r, r, color, filled=True)
        draw_ellipse(pixels, x1 - r, y1 - r, r, r, color, filled=True)
    else:
        # Four straight edges (shortened by r).
        draw_line(pixels, x + r, y, x1 - r, y, color)  # top
        draw_line(pixels, x + r, y1, x1 - r, y1, color)  # bottom
        draw_line(pixels, x, y + r, x, y1 - r, color)  # left
        draw_line(pixels, x1, y + r, x1, y1 - r, color)  # right
        # Four corner arcs (parametric, screen-space: y increases downward).
        _arc_segment(pixels, x + r, y + r, r, 180, 270, color)  # top-left
        _arc_segment(pixels, x1 - r, y + r, r, 270, 360, color)  # top-right
        _arc_segment(pixels, x1 - r, y1 - r, r, 0, 90, color)  # bottom-right
        _arc_segment(pixels, x + r, y1 - r, r, 90, 180, color)  # bottom-left


def _arc_segment(
    pixels: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    start_deg: float,
    end_deg: float,
    color: Color,
) -> None:
    """Plot a circular arc from *start_deg* to *end_deg* (degrees, CW in screen space).

    Uses a parametric approach: ``px = cx + r*cos(θ)``, ``py = cy + r*sin(θ)``
    where ``sin`` is positive downward (standard screen coordinates).

    Args:
        pixels: RGBA uint8 array of shape ``(H, W, 4)``.
        cx, cy: Centre of the arc.
        radius: Radius in pixels.
        start_deg, end_deg: Arc extent in degrees.
        color: Stroke color.
    """
    if radius <= 0:
        return
    steps = max(radius * 2, 8)
    for i in range(steps + 1):
        angle = math.radians(start_deg + (end_deg - start_deg) * i / steps)
        px = round(cx + radius * math.cos(angle))
        py = round(cy + radius * math.sin(angle))
        set_pixel(pixels, px, py, color)


# ---------------------------------------------------------------------------
# Tolerance-based flood fill
# ---------------------------------------------------------------------------


def flood_fill_tolerance(
    pixels: np.ndarray,
    x: int,
    y: int,
    fill_color: Color,
    tolerance: int,
    *,
    connectivity: int = 4,
) -> int:
    """Flood fill that replaces pixels whose color is within *tolerance* of the seed.

    Euclidean distance in RGBA space is used.  A tolerance of ``0`` behaves
    identically to :func:`flood_fill`.

    Args:
        pixels: RGBA uint8 array of shape ``(H, W, 4)``.
        x, y: Seed coordinate.
        fill_color: ``(R, G, B, A)`` replacement color.
        tolerance: Maximum RGBA Euclidean distance from seed color to replace.
        connectivity: ``4`` or ``8``.

    Returns:
        Number of pixels replaced.

    Raises:
        ValueError: If *connectivity* is not 4 or 8.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    h, w = pixels.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return 0

    if tolerance == 0:
        return flood_fill(pixels, x, y, fill_color, connectivity=connectivity)

    target = np.array([int(v) for v in pixels[y, x]], dtype=np.int32)
    fill = tuple(int(v) for v in fill_color)
    if tuple(int(v) for v in pixels[y, x]) == fill:
        return 0

    tol_sq = int(tolerance) ** 2

    def _matches(px: int, py: int) -> bool:
        c = pixels[py, px].astype(np.int32)
        return int(np.sum((c - target) ** 2)) <= tol_sq

    if connectivity == 4:
        neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        neighbours = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

    count = 0
    queue: deque[Tuple[int, int]] = deque()
    queue.append((x, y))
    visited = np.zeros((h, w), dtype=bool)
    visited[y, x] = True

    while queue:
        px, py = queue.popleft()
        if not _matches(px, py):
            continue
        pixels[py, px] = fill_color
        count += 1
        for nx, ny in ((px + dx, py + dy) for dx, dy in neighbours):
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))

    return count


# ---------------------------------------------------------------------------
# Selection mask builders
# ---------------------------------------------------------------------------


def flood_fill_mask(
    pixels: np.ndarray,
    x: int,
    y: int,
    tolerance: int = 0,
    *,
    connectivity: int = 4,
) -> np.ndarray:
    """Return a boolean mask of the region reachable from (x, y) by color similarity.

    Unlike :func:`flood_fill`, this function does not modify *pixels*.

    Args:
        pixels: RGBA uint8 array of shape ``(H, W, 4)``.
        x, y: Seed coordinate.
        tolerance: Maximum RGBA Euclidean distance from seed color.
        connectivity: ``4`` or ``8``.

    Returns:
        Boolean NumPy array of shape ``(H, W)``.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    h, w = pixels.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    if not (0 <= x < w and 0 <= y < h):
        return mask

    target = pixels[y, x].astype(np.int32)
    tol_sq = int(tolerance) ** 2

    def _matches(px: int, py: int) -> bool:
        c = pixels[py, px].astype(np.int32)
        return int(np.sum((c - target) ** 2)) <= tol_sq

    if connectivity == 4:
        neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        neighbours = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

    queue: deque[Tuple[int, int]] = deque()
    queue.append((x, y))
    visited = np.zeros((h, w), dtype=bool)
    visited[y, x] = True

    while queue:
        px, py = queue.popleft()
        if not _matches(px, py):
            continue
        mask[py, px] = True
        for nx, ny in ((px + dx, py + dy) for dx, dy in neighbours):
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))

    return mask


def polygon_mask(
    height: int,
    width: int,
    vertices: List[Tuple[int, int]],
) -> np.ndarray:
    """Create a boolean mask of the interior of a polygon.

    Uses PIL's polygon rasterizer.

    Args:
        height: Mask height in pixels.
        width: Mask width in pixels.
        vertices: Ordered list of ``(x, y)`` polygon vertices.

    Returns:
        Boolean NumPy array of shape ``(height, width)``.
    """
    from PIL import Image, ImageDraw  # local import — PIL is an optional dep here

    img = Image.new("L", (width, height), 0)
    if len(vertices) >= 3:
        draw = ImageDraw.Draw(img)
        draw.polygon(vertices, fill=255)
    return np.array(img, dtype=bool)
