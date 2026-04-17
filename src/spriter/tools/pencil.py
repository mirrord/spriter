# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Pencil tool — paints with the foreground color."""

from __future__ import annotations

from typing import Optional, Tuple

from ..utils.geometry import line_points
from .base import Tool


class PencilTool(Tool):
    """Single-pixel or variable-size brush that paints the foreground color.

    Consecutive drag positions are connected by Bresenham lines so that fast
    gestures leave no gaps.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last: Optional[Tuple[int, int]] = None

    def on_press(self, x: int, y: int) -> None:
        w = self._begin_stroke()
        self._paint_at(w, x, y, self._paint_color())
        self._last = (x, y)

    def on_drag(self, x: int, y: int) -> None:
        if self._working is None or self._last is None:
            return
        lx, ly = self._last
        for px, py in line_points(lx, ly, x, y):
            self._paint_at(self._working, px, py, self._paint_color())
        self._last = (x, y)

    def on_release(self, x: int, y: int) -> None:
        self._commit_stroke("Pencil")
        self._last = None
