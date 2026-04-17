# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Selection tools: rectangular marquee, freeform lasso, and magic wand."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..commands.base import CommandStack
from ..commands.draw import SetSelectionCommand
from ..core.sprite import Sprite
from ..utils.geometry import flood_fill_mask, polygon_mask
from .base import Tool


class RectSelectTool(Tool):
    """Rectangular marquee selection.

    Draws a rectangular selection mask between the press and release coordinates.
    Pressing on an existing selection replaces it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._start: Optional[Tuple[int, int]] = None
        self._before_mask: Optional[np.ndarray] = None

    def on_press(self, x: int, y: int) -> None:
        self._before_mask = (
            self._sprite.selection_mask.copy()
            if self._sprite.selection_mask is not None
            else None
        )
        self._start = (x, y)

    def on_drag(self, x: int, y: int) -> None:
        pass  # No live-preview for selection in headless mode.

    def on_release(self, x: int, y: int) -> None:
        if self._start is None:
            return
        x0, y0 = self._start
        h, w = self._sprite.height, self._sprite.width
        lx = max(0, min(x0, x))
        ly = max(0, min(y0, y))
        rx = min(w - 1, max(x0, x))
        ry = min(h - 1, max(y0, y))
        mask = np.zeros((h, w), dtype=bool)
        mask[ly : ry + 1, lx : rx + 1] = True
        cmd = SetSelectionCommand(self._sprite, self._before_mask, mask)
        self._stack.push(cmd)
        self._start = None

    # These tools don't draw pixels, so stroke helpers are unused.
    def _begin_stroke(self):  # type: ignore[override]
        raise NotImplementedError("Selection tools do not use _begin_stroke")

    def _commit_stroke(self, description=""):  # type: ignore[override]
        raise NotImplementedError("Selection tools do not use _commit_stroke")


class LassoTool(Tool):
    """Freeform lasso selection.

    Records the drag path as a polygon and converts it to a boolean mask on
    release using PIL scanline rasterization.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._points: List[Tuple[int, int]] = []
        self._before_mask: Optional[np.ndarray] = None

    def on_press(self, x: int, y: int) -> None:
        self._before_mask = (
            self._sprite.selection_mask.copy()
            if self._sprite.selection_mask is not None
            else None
        )
        self._points = [(x, y)]

    def on_drag(self, x: int, y: int) -> None:
        self._points.append((x, y))

    def on_release(self, x: int, y: int) -> None:
        self._points.append((x, y))
        h, w = self._sprite.height, self._sprite.width
        mask = polygon_mask(h, w, self._points)
        cmd = SetSelectionCommand(self._sprite, self._before_mask, mask)
        self._stack.push(cmd)
        self._points = []

    def _begin_stroke(self):  # type: ignore[override]
        raise NotImplementedError

    def _commit_stroke(self, description=""):  # type: ignore[override]
        raise NotImplementedError


class MagicWandTool(Tool):
    """Magic wand — selects a contiguous region by color similarity.

    Attributes:
        tolerance: Maximum RGBA Euclidean distance from the seed pixel to include.
        connectivity: ``4`` or ``8`` neighbour connectivity.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tolerance: int = 32
        self.connectivity: int = 4

    def on_press(self, x: int, y: int) -> None:
        before_mask = (
            self._sprite.selection_mask.copy()
            if self._sprite.selection_mask is not None
            else None
        )
        # Sample from the composited frame so the wand works across layers.
        buf = self._sprite.composite_frame(self.frame_index)
        mask = flood_fill_mask(
            buf, x, y, self.tolerance, connectivity=self.connectivity
        )
        cmd = SetSelectionCommand(self._sprite, before_mask, mask)
        self._stack.push(cmd)

    def on_drag(self, x: int, y: int) -> None:
        pass  # One-shot on press.

    def on_release(self, x: int, y: int) -> None:
        pass

    def _begin_stroke(self):  # type: ignore[override]
        raise NotImplementedError

    def _commit_stroke(self, description=""):  # type: ignore[override]
        raise NotImplementedError
