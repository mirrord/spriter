# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""AI-assisted undoable operations.

Commands
--------
* :class:`GenerateFrameCommand` — insert a diffusion-generated frame after the
  current one, placing the generated pixels on the active layer.
* :class:`GenerateFramesCommand` — insert a sequence of generated frames (e.g.
  extracted from a generated video) after the current one.
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


class GenerateFramesCommand(Command):
    """Insert a sequence of generated frames after the current one.

    Each canvas-sized RGBA array in *frames_pixels* becomes a new frame inserted
    consecutively after *after_frame_index*, with its pixels written to the cel
    at *layer_index*.  Undo removes all inserted frames.

    Args:
        sprite: The owning sprite.
        after_frame_index: Index of the frame the new frames are inserted after.
        layer_index: Layer to place the generated pixels on.
        frames_pixels: Ordered canvas-sized ``H×W×4`` ``uint8`` RGBA arrays.

    Raises:
        ValueError: If *frames_pixels* is empty or any array does not match the
            canvas dimensions.
    """

    def __init__(
        self,
        sprite: Sprite,
        after_frame_index: int,
        layer_index: int,
        frames_pixels: list[np.ndarray],
    ) -> None:
        if not frames_pixels:
            raise ValueError("frames_pixels must contain at least one frame")
        for pixels in frames_pixels:
            if pixels.shape[:2] != (sprite.height, sprite.width):
                raise ValueError(
                    f"pixels shape {pixels.shape[:2]} does not match canvas "
                    f"{sprite.width}x{sprite.height}"
                )
        self._sprite = sprite
        self._layer_index = layer_index
        self._insert_start = after_frame_index + 1
        self._frames = [pixels.copy() for pixels in frames_pixels]

    @property
    def description(self) -> str:
        return "Generate Frames"

    def execute(self) -> None:
        for offset, pixels in enumerate(self._frames):
            index = self._insert_start + offset
            self._sprite.add_frame(index=index)
            self._sprite.set_cel_pixels(self._layer_index, index, pixels)

    def undo(self) -> None:
        for offset in reversed(range(len(self._frames))):
            self._sprite.remove_frame(self._insert_start + offset)
