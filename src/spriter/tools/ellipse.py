# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Ellipse tool — draws filled or outlined axis-aligned ellipses."""

from __future__ import annotations

from typing import Optional, Tuple

from ..utils.geometry import draw_ellipse
from .base import Tool


class EllipseTool(Tool):
    """Draws an axis-aligned ellipse bounding-boxed between press and release.

    Attributes:
        filled: When True the ellipse interior is filled.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.filled: bool = False
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
        self._commit_stroke("Ellipse")
        self._start = None

    def _draw_preview(self, x: int, y: int) -> None:
        assert self._working is not None and self._start is not None
        x0, y0 = self._start
        cx = (x0 + x) // 2
        cy = (y0 + y) // 2
        rx = abs(x - x0) // 2
        ry = abs(y - y0) // 2
        color = self._paint_color()
        draw_ellipse(self._working, cx, cy, rx, ry, color, filled=self.filled)
