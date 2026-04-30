# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Drawing commands — record before/after pixel state for undo/redo."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..commands.base import Command
from ..core.sprite import Sprite


class DrawCelCommand(Command):
    """Records a pixel-level edit to a single cel for undo/redo.

    The command snapshots both the *before* and *after* pixel buffers so that
    :meth:`execute` can re-apply the change and :meth:`undo` can revert it.

    Internally only the cropped bounding box of changed pixels is retained, so
    a small edit on a large canvas costs proportionally little memory.

    Args:
        sprite: The owning sprite document.
        layer_index: Index of the layer being edited.
        frame_index: Index of the frame being edited.
        before: Pixel buffer before the edit (RGBA uint8, will be copied).
        after: Pixel buffer after the edit (RGBA uint8, will be copied).
        description: Human-readable label for the Edit menu.
    """

    def __init__(
        self,
        sprite: Sprite,
        layer_index: int,
        frame_index: int,
        before: np.ndarray,
        after: np.ndarray,
        description: str = "Draw",
    ) -> None:
        self._sprite = sprite
        self._layer_index = layer_index
        self._frame_index = frame_index
        self._description = description

        # Crop to the bbox of changed pixels to keep undo memory bounded.
        if before.shape != after.shape:
            # Shape mismatch (e.g. canvas resize) — fall back to full snapshots.
            self._bbox: Optional[tuple] = None
            self._before = before.copy()
            self._after = after.copy()
            return

        diff = np.any(before != after, axis=-1)
        if not diff.any():
            # No change — store empty patch.
            self._bbox = (0, 0, 0, 0)
            self._before = np.empty((0, 0, 4), dtype=np.uint8)
            self._after = np.empty((0, 0, 4), dtype=np.uint8)
            return

        rows = np.any(diff, axis=1)
        cols = np.any(diff, axis=0)
        y0 = int(np.argmax(rows))
        y1 = int(len(rows) - np.argmax(rows[::-1]))
        x0 = int(np.argmax(cols))
        x1 = int(len(cols) - np.argmax(cols[::-1]))
        self._bbox = (x0, y0, x1, y1)
        self._before = before[y0:y1, x0:x1].copy()
        self._after = after[y0:y1, x0:x1].copy()

    @property
    def description(self) -> str:
        return self._description

    def execute(self) -> None:
        self._apply(self._after)

    def undo(self) -> None:
        self._apply(self._before)

    def _apply(self, patch: np.ndarray) -> None:
        if self._bbox is None:
            self._sprite.set_cel_pixels(self._layer_index, self._frame_index, patch)
            return
        # Read current cel, paste the patch into the bbox, write back.
        cel = self._sprite.get_cel(self._layer_index, self._frame_index)
        h, w = self._sprite.height, self._sprite.width
        if cel.pixels is None:
            current = np.zeros((h, w, 4), dtype=np.uint8)
        else:
            current = cel.pixels.copy()
        x0, y0, x1, y1 = self._bbox
        if x1 > x0 and y1 > y0:
            current[y0:y1, x0:x1] = patch
        self._sprite.set_cel_pixels(self._layer_index, self._frame_index, current)


class SetSelectionCommand(Command):
    """Records a change to the sprite's selection mask.

    Args:
        sprite: The owning sprite document.
        before_mask: Selection mask before the operation (may be None).
        after_mask: Selection mask after the operation (may be None).
    """

    def __init__(
        self,
        sprite: Sprite,
        before_mask: Optional[np.ndarray],
        after_mask: Optional[np.ndarray],
    ) -> None:
        self._sprite = sprite
        self._before = before_mask.copy() if before_mask is not None else None
        self._after = after_mask.copy() if after_mask is not None else None

    @property
    def description(self) -> str:
        return "Set Selection"

    def execute(self) -> None:
        if self._after is not None:
            self._sprite.selection_mask = self._after.copy()
        else:
            self._sprite.selection_mask = None

    def undo(self) -> None:
        if self._before is not None:
            self._sprite.selection_mask = self._before.copy()
        else:
            self._sprite.selection_mask = None
