# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Sprite document model — the root data structure for a Spriter project.

A Sprite owns an ordered list of :class:`Layer` objects and an ordered list
of :class:`Frame` objects.  Pixel data lives in :class:`Cel` objects indexed
by ``(layer_index, frame_index)`` pairs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .animation import Animation
from .frame import Cel, Frame
from .layer import BlendMode, Layer

# Color modes (extensible for future indexed-color support).
ColorMode = str  # "RGBA" is the only mode for Phase 1

CelKey = Tuple[int, int]  # (layer_index, frame_index)


class Sprite:
    """Root document model for a Spriter project.

    Args:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        color_mode: Currently only ``"RGBA"`` is supported.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        color_mode: ColorMode = "RGBA",
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Canvas size must be positive, got {width}x{height}")
        if color_mode != "RGBA":
            raise ValueError(f"Unsupported color mode: {color_mode!r}")
        self.width = width
        self.height = height
        self.color_mode = color_mode
        self._layers: List[Layer] = []
        self._frames: List[Frame] = []
        self._cels: Dict[CelKey, Cel] = {}
        self.selection_mask: Optional[np.ndarray] = None  # bool (H, W) or None
        self.animation: Animation = Animation()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def layers(self) -> List[Layer]:
        """Ordered list of layers (bottom to top)."""
        return list(self._layers)

    @property
    def frames(self) -> List[Frame]:
        """Ordered list of animation frames."""
        return list(self._frames)

    @property
    def layer_count(self) -> int:
        return len(self._layers)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add_layer(
        self,
        name: str = "Layer",
        *,
        index: Optional[int] = None,
        visible: bool = True,
        locked: bool = False,
        opacity: int = 255,
        blend_mode: BlendMode = BlendMode.NORMAL,
    ) -> Layer:
        """Create a new layer and insert it into the stack.

        Args:
            name: Display name.
            index: Position to insert at; appends if None.
            visible: Initial visibility.
            locked: Initial lock state.
            opacity: Initial opacity 0–255.
            blend_mode: Initial blend mode.

        Returns:
            The newly created :class:`Layer`.
        """
        layer = Layer(
            name,
            visible=visible,
            locked=locked,
            opacity=opacity,
            blend_mode=blend_mode,
        )
        if index is None:
            self._layers.append(layer)
        else:
            self._layers.insert(self._clamp_layer_index(index), layer)
        # Blank transparent cels for all existing frames.
        layer_idx = self._layers.index(layer)
        for frame_idx in range(len(self._frames)):
            self._cels[(layer_idx, frame_idx)] = Cel(self._blank_pixels())
        return layer

    def remove_layer(self, index: int) -> Layer:
        """Remove and return the layer at *index*, deleting its cels.

        Args:
            index: Layer index to remove.

        Returns:
            The removed :class:`Layer`.
        """
        self._validate_layer_index(index)
        layer = self._layers.pop(index)
        # Remove cels for the removed layer and re-index higher-layer cels.
        new_cels: Dict[CelKey, Cel] = {}
        for (li, fi), cel in self._cels.items():
            if li == index:
                continue
            new_li = li if li < index else li - 1
            new_cels[(new_li, fi)] = cel
        self._cels = new_cels
        return layer

    def move_layer(self, from_index: int, to_index: int) -> None:
        """Reorder layers by moving the layer at *from_index* to *to_index*.

        Args:
            from_index: Current position.
            to_index: Destination position.
        """
        self._validate_layer_index(from_index)
        to_index = max(0, min(len(self._layers) - 1, to_index))
        if from_index == to_index:
            return
        layer = self._layers.pop(from_index)
        self._layers.insert(to_index, layer)
        # Rebuild cel keys to reflect new order.
        new_cels: Dict[CelKey, Cel] = {}
        for (li, fi), cel in self._cels.items():
            new_li = _reindex(li, from_index, to_index)
            new_cels[(new_li, fi)] = cel
        self._cels = new_cels

    # ------------------------------------------------------------------
    # Frame management
    # ------------------------------------------------------------------

    def add_frame(
        self,
        duration_ms: int = 100,
        *,
        index: Optional[int] = None,
    ) -> Frame:
        """Create a new frame and insert it into the timeline.

        Args:
            duration_ms: Display duration.
            index: Position to insert at; appends if None.

        Returns:
            The newly created :class:`Frame`.
        """
        frame = Frame(duration_ms)
        if index is None:
            self._frames.append(frame)
        else:
            self._frames.insert(index, frame)
        frame_idx = self._frames.index(frame)
        # Blank cels for every existing layer.
        for layer_idx in range(len(self._layers)):
            self._cels[(layer_idx, frame_idx)] = Cel(self._blank_pixels())
        return frame

    def remove_frame(self, index: int) -> Frame:
        """Remove and return the frame at *index*, deleting its cels.

        Args:
            index: Frame index to remove.

        Returns:
            The removed :class:`Frame`.
        """
        self._validate_frame_index(index)
        frame = self._frames.pop(index)
        new_cels: Dict[CelKey, Cel] = {}
        for (li, fi), cel in self._cels.items():
            if fi == index:
                continue
            new_fi = fi if fi < index else fi - 1
            new_cels[(li, new_fi)] = cel
        self._cels = new_cels
        return frame

    def move_frame(self, from_index: int, to_index: int) -> None:
        """Reorder frames by moving *from_index* to *to_index*.

        Args:
            from_index: Current position.
            to_index: Destination position.
        """
        self._validate_frame_index(from_index)
        to_index = max(0, min(len(self._frames) - 1, to_index))
        if from_index == to_index:
            return
        frame_obj = self._frames.pop(from_index)
        self._frames.insert(to_index, frame_obj)
        new_cels: Dict[CelKey, Cel] = {}
        for (li, fi), cel in self._cels.items():
            new_fi = _reindex(fi, from_index, to_index)
            new_cels[(li, new_fi)] = cel
        self._cels = new_cels

    # ------------------------------------------------------------------
    # Cel access
    # ------------------------------------------------------------------

    def get_cel(self, layer_index: int, frame_index: int) -> Cel:
        """Return the :class:`Cel` at the given layer/frame intersection.

        If the cel is linked, the linked frame's cel is returned instead.

        Args:
            layer_index: Index into the layer list.
            frame_index: Index into the frame list.

        Returns:
            The :class:`Cel` for that layer/frame pair.
        """
        self._validate_layer_index(layer_index)
        self._validate_frame_index(frame_index)
        cel = self._cels.get((layer_index, frame_index))
        if cel is None:
            cel = Cel(self._blank_pixels())
            self._cels[(layer_index, frame_index)] = cel
        if cel.is_linked and cel.linked_frame is not None:
            return self._cels.get(
                (layer_index, cel.linked_frame), Cel(self._blank_pixels())
            )
        return cel

    def set_cel_pixels(
        self,
        layer_index: int,
        frame_index: int,
        pixels: np.ndarray,
    ) -> None:
        """Replace the pixel buffer for a specific cel.

        Args:
            layer_index: Index into the layer list.
            frame_index: Index into the frame list.
            pixels: RGBA uint8 array of shape ``(height, width, 4)``.
        """
        self._validate_layer_index(layer_index)
        self._validate_frame_index(frame_index)
        if pixels.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"pixels shape {pixels.shape[:2]} does not match canvas "
                f"{self.width}x{self.height}"
            )
        cel = self._cels.get((layer_index, frame_index))
        if cel is None:
            cel = Cel()
            self._cels[(layer_index, frame_index)] = cel
        cel.pixels = pixels.copy()

    def composite_frame(self, frame_index: int) -> np.ndarray:
        """Composite all visible layers for *frame_index* into a single RGBA image.

        Uses simple alpha-compositing (Porter-Duff over) for NORMAL blend mode.
        Blend modes other than NORMAL are approximated as NORMAL for the initial
        phase.

        Args:
            frame_index: Index into the frame list.

        Returns:
            RGBA uint8 NumPy array of shape ``(height, width, 4)``.
        """
        self._validate_frame_index(frame_index)
        result = np.zeros((self.height, self.width, 4), dtype=np.float32)
        for layer_idx, layer in enumerate(self._layers):
            if not layer.visible:
                continue
            cel = self.get_cel(layer_idx, frame_index)
            if cel.pixels is None:
                continue
            src = cel.pixels.astype(np.float32)
            # Apply layer opacity.
            alpha_scale = layer.opacity / 255.0
            src_a = (src[..., 3] / 255.0) * alpha_scale  # (H, W)
            dst_a = result[..., 3] / 255.0  # (H, W)
            out_a = src_a + dst_a * (1.0 - src_a)
            # Avoid division by zero.
            safe_out_a = np.where(out_a > 0, out_a, 1.0)
            for ch in range(3):
                result[..., ch] = (
                    src[..., ch] * src_a + result[..., ch] * dst_a * (1.0 - src_a)
                ) / safe_out_a
            result[..., 3] = out_a * 255.0
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blank_pixels(self) -> np.ndarray:
        """Return a fully-transparent pixel buffer matching canvas dimensions."""
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def _validate_layer_index(self, index: int) -> None:
        if not (0 <= index < len(self._layers)):
            raise IndexError(
                f"Layer index {index} out of range (0–{len(self._layers) - 1})"
            )

    def _validate_frame_index(self, index: int) -> None:
        if not (0 <= index < len(self._frames)):
            raise IndexError(
                f"Frame index {index} out of range (0–{len(self._frames) - 1})"
            )

    def _clamp_layer_index(self, index: int) -> int:
        return max(0, min(len(self._layers), index))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def set_selection(self, mask: np.ndarray) -> None:
        """Set the active selection to a boolean mask.

        Args:
            mask: Boolean array of shape ``(height, width)``.

        Raises:
            ValueError: If *mask* shape does not match the canvas.
        """
        if mask.shape != (self.height, self.width):
            raise ValueError(
                f"Selection mask shape {mask.shape} does not match canvas "
                f"{self.width}x{self.height}"
            )
        self.selection_mask = mask.astype(bool)

    def clear_selection(self) -> None:
        """Remove the active selection (all pixels become editable)."""
        self.selection_mask = None

    # ------------------------------------------------------------------
    # Canvas / sprite resize
    # ------------------------------------------------------------------

    def resize_canvas(
        self,
        new_width: int,
        new_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        """Resize the canvas, placing existing content at (*offset_x*, *offset_y*).

        Pixels outside the new canvas bounds are discarded.  New regions are
        filled with transparent pixels.

        Args:
            new_width: New canvas width in pixels.
            new_height: New canvas height in pixels.
            offset_x: X position where existing content is placed in the new canvas.
            offset_y: Y position where existing content is placed in the new canvas.
        """
        if new_width <= 0 or new_height <= 0:
            raise ValueError(
                f"Canvas size must be positive, got {new_width}x{new_height}"
            )
        old_w, old_h = self.width, self.height
        new_cels: Dict[CelKey, Cel] = {}
        for key, cel in self._cels.items():
            if cel.pixels is None:
                new_buf = np.zeros((new_height, new_width, 4), dtype=np.uint8)
            else:
                new_buf = np.zeros((new_height, new_width, 4), dtype=np.uint8)
                # Region of old pixels that lands in the new canvas.
                src_x1 = max(0, -offset_x)
                src_y1 = max(0, -offset_y)
                src_x2 = min(old_w, new_width - offset_x)
                src_y2 = min(old_h, new_height - offset_y)
                dst_x1 = src_x1 + offset_x
                dst_y1 = src_y1 + offset_y
                dst_x2 = src_x2 + offset_x
                dst_y2 = src_y2 + offset_y
                if src_x2 > src_x1 and src_y2 > src_y1:
                    new_buf[dst_y1:dst_y2, dst_x1:dst_x2] = cel.pixels[
                        src_y1:src_y2, src_x1:src_x2
                    ]
            new_cels[key] = Cel(new_buf)
        self._cels = new_cels
        self.width = new_width
        self.height = new_height

    def scale_pixels(
        self,
        new_width: int,
        new_height: int,
        *,
        method: str = "nearest",
    ) -> None:
        """Scale all cel pixel buffers to *new_width* × *new_height*.

        Also updates ``self.width`` and ``self.height``.

        Args:
            new_width: Target width in pixels.
            new_height: Target height in pixels.
            method: Resampling filter — ``"nearest"`` (default, good for pixel
                art) or ``"bilinear"``.

        Raises:
            ValueError: If dimensions are not positive.
        """
        if new_width <= 0 or new_height <= 0:
            raise ValueError(
                f"Canvas size must be positive, got {new_width}x{new_height}"
            )
        from PIL import Image as _PILImage

        resample = (
            _PILImage.Resampling.NEAREST
            if method == "nearest"
            else _PILImage.Resampling.BILINEAR
        )
        new_cels: Dict[CelKey, Cel] = {}
        for key, cel in self._cels.items():
            if cel.pixels is None:
                new_buf = np.zeros((new_height, new_width, 4), dtype=np.uint8)
            else:
                pil_img = _PILImage.fromarray(cel.pixels, mode="RGBA")
                pil_img = pil_img.resize((new_width, new_height), resample)
                new_buf = np.array(pil_img, dtype=np.uint8)
            new_cels[key] = Cel(new_buf)
        self._cels = new_cels
        self.width = new_width
        self.height = new_height

    def __repr__(self) -> str:
        return (
            f"Sprite({self.width}x{self.height}, {self.color_mode}, "
            f"{self.layer_count} layers, {self.frame_count} frames)"
        )


def _reindex(idx: int, removed: int, inserted: int) -> int:
    """Compute the new index of element *idx* after moving *removed* to *inserted*."""
    if idx == removed:
        return inserted
    if removed < inserted:
        # Items shift left through the vacated slot.
        if removed < idx <= inserted:
            return idx - 1
    else:
        # Items shift right.
        if inserted <= idx < removed:
            return idx + 1
    return idx
