# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Pencil tool — paints with the foreground color."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..utils.geometry import line_points
from .base import Tool


def _remove_l_corners(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Remove pixels that form L-shaped junctions from a stroke.

    For each consecutive triple (A, B, C) where A→B is a diagonal and B→C is
    the perpendicular axis-aligned step, pixel B can be removed without breaking
    visual continuity.  This produces the characteristic pixel-perfect look.
    """
    if len(points) < 3:
        return points

    result = [points[0]]
    for i in range(1, len(points) - 1):
        ax, ay = points[i - 1]
        bx, by = points[i]
        cx, cy = points[i + 1]
        # B is an L-corner when (A→B is diagonal) AND (B→C is axis-aligned
        # in the complementary direction).
        ab_diag = (ax != bx) and (ay != by)
        bc_axis = (bx == cx) or (by == cy)
        if ab_diag and bc_axis:
            continue  # skip this point
        result.append((bx, by))
    result.append(points[-1])
    return result


class PencilTool(Tool):
    """Single-pixel or variable-size brush that paints the foreground color.

    Consecutive drag positions are connected by Bresenham lines so that fast
    gestures leave no gaps.

    Attributes:
        pixel_perfect: When ``True``, L-shaped diagonal corners are removed
            from each stroke segment to produce a cleaner pixel-art line.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last: Optional[Tuple[int, int]] = None
        self.pixel_perfect: bool = False

    def on_press(self, x: int, y: int) -> None:
        w = self._begin_stroke()
        self._paint_at(w, x, y, self._paint_color())
        self._last = (x, y)

    def on_drag(self, x: int, y: int) -> None:
        if self._working is None or self._last is None:
            return
        lx, ly = self._last
        pts = list(line_points(lx, ly, x, y))
        if self.pixel_perfect:
            pts = _remove_l_corners(pts)
        for px, py in pts:
            self._paint_at(self._working, px, py, self._paint_color())
        self._last = (x, y)

    def on_release(self, x: int, y: int) -> None:
        self._commit_stroke("Pencil")
        self._last = None
