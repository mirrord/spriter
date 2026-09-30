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
* :class:`GenerateAnimationCommand` — create a new animation timeline from
  generated sprite-sheet frames, sizing the canvas to match.
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


class GenerateAnimationCommand(Command):
    """Create a new animation timeline from generated sprite-sheet frames.

    The canvas is resized to *(width, height)* when it differs from the current
    size (existing content is cropped/padded and restored on undo), then a new
    timeline named *name* is appended and filled with *frames_pixels* on
    *layer_index*.  Undo removes the timeline and restores the canvas.

    Args:
        sprite: The owning sprite.
        name: Name for the new animation timeline.
        frames_pixels: Ordered ``height×width×4`` ``uint8`` RGBA frames.
        width: Target frame/canvas width in pixels.
        height: Target frame/canvas height in pixels.
        layer_index: Layer to place the generated pixels on.

    Raises:
        ValueError: If *frames_pixels* is empty or any array does not match
            *(width, height)*.
    """

    def __init__(
        self,
        sprite: Sprite,
        name: str,
        frames_pixels: list[np.ndarray],
        width: int,
        height: int,
        layer_index: int = 0,
    ) -> None:
        if not frames_pixels:
            raise ValueError("frames_pixels must contain at least one frame")
        for pixels in frames_pixels:
            if pixels.shape[:2] != (height, width):
                raise ValueError(
                    f"pixels shape {pixels.shape[:2]} does not match target "
                    f"{width}x{height}"
                )
        self._sprite = sprite
        self._name = name
        self._frames = [pixels.copy() for pixels in frames_pixels]
        self._width = width
        self._height = height
        self._layer_index = layer_index
        self._old_width: int | None = None
        self._old_height: int | None = None
        self._saved_cels: dict | None = None
        self._timeline_index: int | None = None
        self._prev_active: int | None = None

    @property
    def description(self) -> str:
        return "Generate Animation"

    def execute(self) -> None:
        from .transform import _save_all_cels

        self._prev_active = self._sprite.active_timeline_index
        self._old_width = self._sprite.width
        self._old_height = self._sprite.height
        self._saved_cels = _save_all_cels(self._sprite)
        if (self._width, self._height) != (self._old_width, self._old_height):
            self._sprite.resize_canvas(self._width, self._height)
        timeline = self._sprite.add_timeline(self._name)
        self._timeline_index = self._sprite.timelines.index(timeline)
        self._sprite.set_active_timeline(self._timeline_index)
        for index, pixels in enumerate(self._frames):
            if index >= self._sprite.frame_count:
                self._sprite.add_frame()
            self._sprite.set_cel_pixels(self._layer_index, index, pixels)

    def undo(self) -> None:
        from .transform import _restore_all_cels

        assert self._timeline_index is not None
        assert self._old_width is not None and self._old_height is not None
        assert self._saved_cels is not None and self._prev_active is not None
        self._sprite.remove_timeline(self._timeline_index)
        if (self._width, self._height) != (self._old_width, self._old_height):
            self._sprite.resize_canvas(self._old_width, self._old_height)
        _restore_all_cels(self._sprite, self._saved_cels)
        self._sprite.set_active_timeline(
            min(self._prev_active, self._sprite.timeline_count - 1)
        )
