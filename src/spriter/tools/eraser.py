# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Eraser tool — sets pixels to fully transparent."""

from __future__ import annotations

from typing import Optional, Tuple

from ..utils.geometry import line_points
from .base import Tool


class EraserTool(Tool):
    """Erases pixels to transparent, respecting the current brush size and shape.

    The eraser always writes ``(0, 0, 0, 0)`` directly regardless of the tool
    opacity setting (opacity controls the *brush* opacity for paint tools; the
    eraser unconditionally removes pixels).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last: Optional[Tuple[int, int]] = None

    def on_press(self, x: int, y: int) -> None:
        w = self._begin_stroke()
        self._paint_at(w, x, y, (0, 0, 0, 0), erase=True)
        self._last = (x, y)

    def on_drag(self, x: int, y: int) -> None:
        if self._working is None or self._last is None:
            return
        lx, ly = self._last
        for px, py in line_points(lx, ly, x, y):
            self._paint_at(self._working, px, py, (0, 0, 0, 0), erase=True)
        self._last = (x, y)

    def on_release(self, x: int, y: int) -> None:
        self._commit_stroke("Eraser")
        self._last = None
