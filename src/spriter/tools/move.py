# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Move tool — translates the layer or selection contents by drag offset."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import Tool


class MoveTool(Tool):
    """Moves pixel content by the offset between press and release.

    Behavior:
    - If the sprite has an active **selection mask**, only the selected pixels
      are moved; the vacated area becomes transparent.
    - If there is **no selection**, the entire layer content is moved.

    The moved pixels are alpha-composited over the background at the new
    position (i.e. they are not clipped by the destination content — they
    replace it).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._start: Optional[Tuple[int, int]] = None
        self._floating: Optional[np.ndarray] = None  # pixels being moved
        self._background: Optional[np.ndarray] = None  # layer minus floating pixels

    def on_press(self, x: int, y: int) -> None:
        self._begin_stroke()
        self._start = (x, y)
        assert self._before is not None
        h, w = self._sprite.height, self._sprite.width

        sel = self._sprite.selection_mask
        if sel is not None:
            move_mask = sel
        else:
            move_mask = np.ones((h, w), dtype=bool)

        # Cut the floating pixels out of the source.
        self._floating = np.zeros((h, w, 4), dtype=np.uint8)
        self._floating[move_mask] = self._before[move_mask]

        self._background = self._before.copy()
        self._background[move_mask] = 0

        # Start with background.
        assert self._working is not None
        self._working[:] = self._background

    def on_drag(self, x: int, y: int) -> None:
        if (
            self._working is None
            or self._start is None
            or self._floating is None
            or self._background is None
        ):
            return
        dx = x - self._start[0]
        dy = y - self._start[1]
        self._working[:] = self._background
        _paste_offset(self._working, self._floating, dx, dy)

    def on_release(self, x: int, y: int) -> None:
        if (
            self._working is None
            or self._start is None
            or self._floating is None
            or self._background is None
        ):
            return
        dx = x - self._start[0]
        dy = y - self._start[1]
        self._working[:] = self._background
        _paste_offset(self._working, self._floating, dx, dy)
        # Commit directly — do NOT use _commit_stroke because that would
        # incorrectly re-apply the selection mask (the moved pixels land
        # *outside* the original selection region).
        self._direct_commit("Move")
        self._start = None
        self._floating = None
        self._background = None

    def _direct_commit(self, description: str) -> None:
        """Push a DrawCelCommand without applying selection-mask filtering."""
        if self._before is None or self._working is None:
            return
        from ..commands.draw import DrawCelCommand

        if np.array_equal(self._before, self._working):
            self._before = self._working = None
            return
        cmd = DrawCelCommand(
            self._sprite,
            self.layer_index,
            self.frame_index,
            self._before,
            self._working,
            description,
        )
        self._sprite.set_cel_pixels(self.layer_index, self.frame_index, self._working)
        self._stack.push(cmd, execute=False)
        self._before = self._working = None


def _paste_offset(dst: np.ndarray, src: np.ndarray, dx: int, dy: int) -> None:
    """Alpha-composite *src* shifted by (dx, dy) onto *dst* in-place.

    Out-of-bounds pixels are clipped.

    Args:
        dst: Destination RGBA buffer.
        src: Source RGBA buffer (same shape as *dst*).
        dx: Horizontal offset (positive = right).
        dy: Vertical offset (positive = down).
    """
    h, w = dst.shape[:2]

    # Source region (before offset).
    sx0 = max(0, -dx)
    sy0 = max(0, -dy)
    sx1 = min(w, w - dx)
    sy1 = min(h, h - dy)

    # Destination region (after offset).
    dx0 = max(0, dx)
    dy0 = max(0, dy)
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    if sx1 <= sx0 or sy1 <= sy0:
        return

    src_region = src[sy0:sy1, sx0:sx1].astype(np.float32)
    dst_region = dst[dy0:dy1, dx0:dx1].astype(np.float32)

    sa = src_region[..., 3:4] / 255.0
    da = dst_region[..., 3:4] / 255.0
    out_a = sa + da * (1.0 - sa)
    safe_out_a = np.where(out_a > 0, out_a, 1.0)

    out_rgb = (
        src_region[..., :3] * sa + dst_region[..., :3] * da * (1.0 - sa)
    ) / safe_out_a
    result = np.concatenate([out_rgb, out_a * 255.0], axis=-1)

    dst[dy0:dy1, dx0:dx1] = np.clip(result, 0, 255).astype(np.uint8)
