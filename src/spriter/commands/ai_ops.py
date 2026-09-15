# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""AI-assisted undoable operations.

Commands
--------
* :class:`GenerateFrameCommand` — insert a diffusion-generated frame after the
  current one, placing the generated pixels on the active layer.
"""

from __future__ import annotations

import numpy as np

from ..core.sprite import Sprite
from .base import Command


class GenerateFrameCommand(Command):
    """Insert a new frame carrying diffusion-generated pixels.

    A blank frame is inserted immediately after *after_frame_index*; the
    supplied canvas-sized RGBA *pixels* are written to the cel at
    *layer_index* of that new frame.  Undo removes the inserted frame.

    Args:
        sprite: The owning sprite.
        after_frame_index: Index of the frame the new frame is inserted after.
        layer_index: Layer to place the generated pixels on.
        pixels: Canvas-sized ``H×W×4`` ``uint8`` RGBA array to write.

    Raises:
        ValueError: If *pixels* does not match the canvas dimensions.
    """

    def __init__(
        self,
        sprite: Sprite,
        after_frame_index: int,
        layer_index: int,
        pixels: np.ndarray,
    ) -> None:
        if pixels.shape[:2] != (sprite.height, sprite.width):
            raise ValueError(
                f"pixels shape {pixels.shape[:2]} does not match canvas "
                f"{sprite.width}x{sprite.height}"
            )
        self._sprite = sprite
        self._layer_index = layer_index
        self._insert_index = after_frame_index + 1
        self._pixels = pixels.copy()

    @property
    def description(self) -> str:
        return "Generate Frame"

    def execute(self) -> None:
        self._sprite.add_frame(index=self._insert_index)
        self._sprite.set_cel_pixels(self._layer_index, self._insert_index, self._pixels)

    def undo(self) -> None:
        self._sprite.remove_frame(self._insert_index)
