# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Line tool — draws a Bresenham line between press and release points."""

from __future__ import annotations

from typing import Optional, Tuple

from ..utils.geometry import draw_line
from .base import Tool


class LineTool(Tool):
    """Click-drag line tool.

    The line is previewed live during drag and committed on release.
    Each drag rebuilds the working buffer from the snapshot so the preview
    always shows exactly one line.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._start: Optional[Tuple[int, int]] = None

    def on_press(self, x: int, y: int) -> None:
        self._begin_stroke()
        self._start = (x, y)
        # Draw the initial zero-length line (single dot).
        assert self._working is not None
        self._working[:] = self._before  # type: ignore[index]
        color = self._paint_color()
        draw_line(self._working, x, y, x, y, color)

    def on_drag(self, x: int, y: int) -> None:
        if self._working is None or self._start is None or self._before is None:
            return
        self._working[:] = self._before
        color = self._paint_color()
        draw_line(self._working, self._start[0], self._start[1], x, y, color)

    def on_release(self, x: int, y: int) -> None:
        if self._working is None or self._start is None or self._before is None:
            return
        self._working[:] = self._before
        color = self._paint_color()
        draw_line(self._working, self._start[0], self._start[1], x, y, color)
        self._commit_stroke("Line")
        self._start = None
