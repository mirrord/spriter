# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Rectangle tool — draws filled or outlined rectangles, with optional rounded corners."""

from __future__ import annotations

from typing import Optional, Tuple

from ..utils.geometry import draw_rect, draw_rounded_rect
from .base import Tool


class RectangleTool(Tool):
    """Draws an axis-aligned rectangle from press to release.

    Attributes:
        filled: When True the rectangle interior is filled; otherwise only the
            outline is drawn.
        corner_radius: Rounds the corners when > 0.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.filled: bool = False
        self.corner_radius: int = 0
        self._start: Optional[Tuple[int, int]] = None

    def on_press(self, x: int, y: int) -> None:
        self._begin_stroke()
        self._start = (x, y)
        assert self._working is not None
        self._working[:] = self._before  # type: ignore[index]
        self._draw_preview(x, y)

    def on_drag(self, x: int, y: int) -> None:
        if self._working is None or self._start is None or self._before is None:
            return
        self._working[:] = self._before
        self._draw_preview(x, y)

    def on_release(self, x: int, y: int) -> None:
        if self._working is None or self._start is None or self._before is None:
            return
        self._working[:] = self._before
        self._draw_preview(x, y)
        self._commit_stroke("Rectangle")
        self._start = None

    def _draw_preview(self, x: int, y: int) -> None:
        assert self._working is not None and self._start is not None
        x0, y0 = self._start
        rx = min(x0, x)
        ry = min(y0, y)
        rw = abs(x - x0) + 1
        rh = abs(y - y0) + 1
        color = self._paint_color()
        if self.corner_radius > 0:
            draw_rounded_rect(
                self._working,
                rx,
                ry,
                rw,
                rh,
                color,
                corner_radius=self.corner_radius,
                filled=self.filled,
            )
        else:
            draw_rect(self._working, rx, ry, rw, rh, color, filled=self.filled)
