# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Eyedropper tool — samples a color from the canvas into the foreground slot."""

from __future__ import annotations

import numpy as np

from .base import Tool


class EyedropperTool(Tool):
    """Samples a pixel color from the canvas and stores it as the foreground.

    Attributes:
        sample_merged: When True (default) the color is sampled from the
            composited frame (all visible layers combined).  When False only
            the active layer's pixels are sampled.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sample_merged: bool = True

    def on_press(self, x: int, y: int) -> None:
        self._sample(x, y)

    def on_drag(self, x: int, y: int) -> None:
        self._sample(x, y)

    def on_release(self, x: int, y: int) -> None:
        pass  # Color already set in on_press / on_drag.

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample(self, x: int, y: int) -> None:
        h, w = self._sprite.height, self._sprite.width
        if not (0 <= x < w and 0 <= y < h):
            return
        if self.sample_merged:
            buf = self._sprite.composite_frame(self.frame_index)
        else:
            cel = self._sprite.get_cel(self.layer_index, self.frame_index)
            buf = (
                cel.pixels
                if cel.pixels is not None
                else np.zeros((h, w, 4), dtype=np.uint8)
            )
        r, g, b, a = buf[y, x]
        self.foreground = (int(r), int(g), int(b), int(a))
