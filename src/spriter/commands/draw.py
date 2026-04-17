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
        self._before = before.copy()
        self._after = after.copy()
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    def execute(self) -> None:
        self._sprite.set_cel_pixels(self._layer_index, self._frame_index, self._after)

    def undo(self) -> None:
        self._sprite.set_cel_pixels(self._layer_index, self._frame_index, self._before)


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
