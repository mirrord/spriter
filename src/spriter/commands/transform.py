# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Pixel-transformation undoable commands (Phase 6).

All commands operate on a single cel identified by ``(layer_index,
frame_index)`` unless noted otherwise.  Each saves the prior pixel state so
:meth:`~spriter.commands.base.Command.undo` can fully restore it.

Commands
--------
* :class:`FlipCommand`          — horizontal or vertical mirror
* :class:`RotateCommand`        — 90° CW / CCW / 180° / arbitrary angle
* :class:`ShiftCommand`         — wrap-around pixel shift
* :class:`OutlineCommand`       — 1-pixel outline around non-transparent pixels
* :class:`ReplaceColorCommand`  — swap one colour for another
* :class:`InvertColorsCommand`  — invert RGB (optionally within selection)
* :class:`AdjustmentCommand`    — brightness / contrast / hue / saturation
* :class:`CanvasResizeCommand`  — resize the canvas (crop / extend every cel)
* :class:`CropToSelectionCommand` — crop every cel to the selection's bbox
* :class:`AutocropCommand`      — crop every cel to the union bbox of opaque pixels
* :class:`ScaleCommand`         — resample all cels to a new size
* :class:`ScaleSelectionCommand` — resample the selected region of one cel
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..commands.base import Command
from ..core.frame import Cel
from ..core.sprite import Sprite

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CelKey = Tuple[int, int]


def _get_pixels(sprite: Sprite, li: int, fi: int) -> np.ndarray:
    """Return a copy of the pixel buffer for the given cel (never None)."""
    cel = sprite.get_cel(li, fi)
    if cel.pixels is None:
        return np.zeros((sprite.height, sprite.width, 4), dtype=np.uint8)
    return cel.pixels.copy()


def _set_pixels(sprite: Sprite, li: int, fi: int, pixels: np.ndarray) -> None:
    sprite.set_cel_pixels(li, fi, pixels)


def _save_all_cels(sprite: Sprite) -> Dict[CelKey, np.ndarray]:
    """Snapshot every cel's pixel buffer (for whole-sprite transforms)."""
    saved: Dict[CelKey, np.ndarray] = {}
    for li in range(sprite.layer_count):
        for fi in range(sprite.frame_count):
            pixels = _get_pixels(sprite, li, fi)
            saved[(li, fi)] = pixels
    return saved


def _restore_all_cels(
    sprite: Sprite,
    saved: Dict[CelKey, np.ndarray],
) -> None:
    for (li, fi), pixels in saved.items():
        sprite.set_cel_pixels(li, fi, pixels)


# ---------------------------------------------------------------------------
# FlipCommand
# ---------------------------------------------------------------------------


class FlipCommand(Command):
    """Mirror the pixels of one cel horizontally or vertically.

    If ``sprite.selection_mask`` is set, only the selected region is flipped
    within the cel; un-selected pixels are preserved.

    Args:
        sprite: The owning sprite.
        layer_index: Layer to operate on.
        frame_index: Frame to operate on.
        horizontal: ``True`` for a left-right flip; ``False`` for top-bottom.
    """

    def __init__(
        self,
        sprite: Sprite,
        layer_index: int,
        frame_index: int,
        *,
        horizontal: bool = True,
    ) -> None:
        self._sprite = sprite
        self._li = layer_index
        self._fi = frame_index
        self._horizontal = horizontal
        self._old_pixels: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        direction = "Horizontal" if self._horizontal else "Vertical"
        return f"Flip {direction}"

    def execute(self) -> None:
        pixels = _get_pixels(self._sprite, self._li, self._fi)
        self._old_pixels = pixels.copy()
        mask = self._sprite.selection_mask
        if mask is not None and np.any(mask):
            # Flip within the bounding box of the selection.
            sel_rows = np.any(mask, axis=1)
            sel_cols = np.any(mask, axis=0)
            r1 = int(np.where(sel_rows)[0][0])
            r2 = int(np.where(sel_rows)[0][-1])
            c1 = int(np.where(sel_cols)[0][0])
            c2 = int(np.where(sel_cols)[0][-1])
            sub = pixels[r1 : r2 + 1, c1 : c2 + 1].copy()
            sub_mask = mask[r1 : r2 + 1, c1 : c2 + 1]
            flipped_sub = np.fliplr(sub) if self._horizontal else np.flipud(sub)
            sub_mask4 = sub_mask[:, :, np.newaxis]
            result_sub = np.where(sub_mask4, flipped_sub, sub)
            result = pixels.copy()
            result[r1 : r2 + 1, c1 : c2 + 1] = result_sub
        else:
            result = np.fliplr(pixels) if self._horizontal else np.flipud(pixels)
        _set_pixels(self._sprite, self._li, self._fi, result)

    def undo(self) -> None:
        assert self._old_pixels is not None
        _set_pixels(self._sprite, self._li, self._fi, self._old_pixels)


# ---------------------------------------------------------------------------
# RotateCommand
# ---------------------------------------------------------------------------


class RotateCommand(Command):
    """Rotate the pixels of one cel.

    Args:
        sprite: The owning sprite.
        layer_index: Layer to operate on.
        frame_index: Frame to operate on.
        angle: Rotation angle.  One of ``90``, ``-90``, ``180``, or any
            arbitrary integer/float value.  For 90/180/-90 the exact
            ``numpy.rot90`` path is used; other values use PIL nearest-neighbour.
    """

    def __init__(
        self,
        sprite: Sprite,
        layer_index: int,
        frame_index: int,
        angle: float = 90,
    ) -> None:
        self._sprite = sprite
        self._li = layer_index
        self._fi = frame_index
        self._angle = angle
        self._old_pixels: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return f"Rotate {self._angle}°"

    def execute(self) -> None:
        pixels = _get_pixels(self._sprite, self._li, self._fi)
        self._old_pixels = pixels.copy()
        ang = self._angle % 360
        if ang == 90:
            result = np.rot90(pixels, k=3)  # 90° CW == -1 anti-clockwise turns
        elif ang == 180:
            result = np.rot90(pixels, k=2)
        elif ang == 270:
            result = np.rot90(pixels, k=1)  # 90° CCW
        else:
            from PIL import Image as _PILImage

            pil_img = _PILImage.fromarray(pixels, mode="RGBA")
            pil_img = pil_img.rotate(
                -self._angle,  # PIL rotates counter-clockwise; negate for CW
                resample=_PILImage.Resampling.NEAREST,
                expand=False,
            )
            result = np.array(pil_img, dtype=np.uint8)
        _set_pixels(self._sprite, self._li, self._fi, result)

    def undo(self) -> None:
        assert self._old_pixels is not None
        _set_pixels(self._sprite, self._li, self._fi, self._old_pixels)


# ---------------------------------------------------------------------------
# ShiftCommand
# ---------------------------------------------------------------------------


class ShiftCommand(Command):
    """Wrap-around shift of a cel's pixels.

    Args:
        sprite: The owning sprite.
        layer_index: Layer to operate on.
        frame_index: Frame to operate on.
        dx: Horizontal shift in pixels (positive = right).
        dy: Vertical shift in pixels (positive = down).
    """

    def __init__(
        self,
        sprite: Sprite,
        layer_index: int,
        frame_index: int,
        dx: int = 0,
        dy: int = 0,
    ) -> None:
        self._sprite = sprite
        self._li = layer_index
        self._fi = frame_index
        self._dx = dx
        self._dy = dy
        self._old_pixels: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return f"Shift ({self._dx}, {self._dy})"

    def execute(self) -> None:
        pixels = _get_pixels(self._sprite, self._li, self._fi)
        self._old_pixels = pixels.copy()
        result = np.roll(pixels, self._dy, axis=0)
        result = np.roll(result, self._dx, axis=1)
        _set_pixels(self._sprite, self._li, self._fi, result)

    def undo(self) -> None:
        assert self._old_pixels is not None
        _set_pixels(self._sprite, self._li, self._fi, self._old_pixels)


# ---------------------------------------------------------------------------
# OutlineCommand
# ---------------------------------------------------------------------------


class OutlineCommand(Command):
    """Generate a 1-pixel outline around non-transparent pixels.

    The *outline_color* is painted on all transparent pixels that are
    4-connected neighbours of at least one opaque pixel.

    Args:
        sprite: The owning sprite.
        layer_index: Layer to operate on.
        frame_index: Frame to operate on.
        outline_color: RGBA colour of the outline (default: opaque black).
    """

    def __init__(
        self,
        sprite: Sprite,
        layer_index: int,
        frame_index: int,
        outline_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
    ) -> None:
        self._sprite = sprite
        self._li = layer_index
        self._fi = frame_index
        self._color = outline_color
        self._old_pixels: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return "Outline"

    def execute(self) -> None:
        pixels = _get_pixels(self._sprite, self._li, self._fi)
        self._old_pixels = pixels.copy()
        opaque = pixels[..., 3] > 0
        result = pixels.copy()
        # Dilate: for each transparent pixel adjacent to an opaque one, paint it.
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(opaque, dy, axis=0)
            shifted = np.roll(shifted, dx, axis=1)
            # Candidate pixels are transparent but neighbour is opaque.
            candidates = shifted & ~opaque
            result[candidates] = self._color
        _set_pixels(self._sprite, self._li, self._fi, result)

    def undo(self) -> None:
        assert self._old_pixels is not None
        _set_pixels(self._sprite, self._li, self._fi, self._old_pixels)


# ---------------------------------------------------------------------------
# ReplaceColorCommand
# ---------------------------------------------------------------------------


class ReplaceColorCommand(Command):
    """Replace all pixels matching *old_color* with *new_color*.

    Comparison uses Euclidean RGBA distance within *tolerance*.
    If ``sprite.selection_mask`` is set, only selected pixels are examined.

    Args:
        sprite: The owning sprite.
        layer_index: Layer to operate on.
        frame_index: Frame to operate on.
        old_color: RGBA colour to replace.
        new_color: RGBA replacement colour.
        tolerance: Maximum Euclidean RGBA distance for a match (0 = exact).
    """

    def __init__(
        self,
        sprite: Sprite,
        layer_index: int,
        frame_index: int,
        old_color: Tuple[int, int, int, int],
        new_color: Tuple[int, int, int, int],
        tolerance: float = 0.0,
    ) -> None:
        self._sprite = sprite
        self._li = layer_index
        self._fi = frame_index
        self._old_color = old_color
        self._new_color = new_color
        self._tolerance = tolerance
        self._old_pixels: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return "Replace Color"

    def execute(self) -> None:
        pixels = _get_pixels(self._sprite, self._li, self._fi)
        self._old_pixels = pixels.copy()
        ref = np.array(self._old_color, dtype=np.float32)
        diff = pixels.astype(np.float32) - ref  # (H, W, 4)
        dist = np.sqrt((diff**2).sum(axis=2))  # (H, W)
        match = dist <= self._tolerance
        mask = self._sprite.selection_mask
        if mask is not None:
            match = match & mask
        result = pixels.copy()
        result[match] = self._new_color
        _set_pixels(self._sprite, self._li, self._fi, result)

    def undo(self) -> None:
        assert self._old_pixels is not None
        _set_pixels(self._sprite, self._li, self._fi, self._old_pixels)


# ---------------------------------------------------------------------------
# InvertColorsCommand
# ---------------------------------------------------------------------------


class InvertColorsCommand(Command):
    """Invert the RGB channels of one cel.

    The alpha channel is preserved, and fully-transparent pixels (alpha == 0)
    are left untouched so transparent regions remain transparent.

    If ``respect_selection`` is True and ``sprite.selection_mask`` is set, only
    pixels inside the selection are inverted.

    Args:
        sprite: The owning sprite.
        layer_index: Layer to operate on.
        frame_index: Frame to operate on.
        respect_selection: If True, restrict the inversion to the current
            selection mask (when one is present).
    """

    def __init__(
        self,
        sprite: Sprite,
        layer_index: int,
        frame_index: int,
        *,
        respect_selection: bool = True,
    ) -> None:
        self._sprite = sprite
        self._li = layer_index
        self._fi = frame_index
        self._respect_selection = respect_selection
        self._old_pixels: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return "Invert Colors"

    def execute(self) -> None:
        pixels = _get_pixels(self._sprite, self._li, self._fi)
        self._old_pixels = pixels.copy()
        # Only touch pixels that are not fully transparent.
        affect = pixels[..., 3] > 0
        if self._respect_selection:
            mask = self._sprite.selection_mask
            if mask is not None:
                affect = affect & mask
        result = pixels.copy()
        result[affect, :3] = 255 - result[affect, :3]
        _set_pixels(self._sprite, self._li, self._fi, result)

    def undo(self) -> None:
        assert self._old_pixels is not None
        _set_pixels(self._sprite, self._li, self._fi, self._old_pixels)


# ---------------------------------------------------------------------------
# AdjustmentCommand
# ---------------------------------------------------------------------------


class AdjustmentCommand(Command):
    """Apply brightness/contrast/hue/saturation adjustments to one cel.

    All factors are multiplicative around 1.0 (neutral):
    - ``brightness=1.5`` → 50% brighter
    - ``contrast=0.8``   → 20% less contrast
    - ``hue=30``         → rotate hue by 30 degrees
    - ``saturation=1.2`` → 20% more saturation

    Args:
        sprite: The owning sprite.
        layer_index: Layer to operate on.
        frame_index: Frame to operate on.
        brightness: Brightness factor (default 1.0 = no change).
        contrast:   Contrast factor (default 1.0 = no change).
        hue:        Hue rotation in degrees (default 0 = no change).
        saturation: Saturation factor (default 1.0 = no change).
    """

    def __init__(
        self,
        sprite: Sprite,
        layer_index: int,
        frame_index: int,
        *,
        brightness: float = 1.0,
        contrast: float = 1.0,
        hue: float = 0.0,
        saturation: float = 1.0,
    ) -> None:
        self._sprite = sprite
        self._li = layer_index
        self._fi = frame_index
        self._brightness = brightness
        self._contrast = contrast
        self._hue = hue
        self._saturation = saturation
        self._old_pixels: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return "Adjust"

    def execute(self) -> None:
        from PIL import Image as _PILImage
        from PIL import ImageEnhance as _Enhance

        pixels = _get_pixels(self._sprite, self._li, self._fi)
        self._old_pixels = pixels.copy()
        img = _PILImage.fromarray(pixels, mode="RGBA")
        # Split alpha for safe colour operations.
        r, g, b, a = img.split()
        rgb = _PILImage.merge("RGB", (r, g, b))

        if self._brightness != 1.0:
            rgb = _Enhance.Brightness(rgb).enhance(self._brightness)
        if self._contrast != 1.0:
            rgb = _Enhance.Contrast(rgb).enhance(self._contrast)
        if self._saturation != 1.0:
            rgb = _Enhance.Color(rgb).enhance(self._saturation)
        if self._hue != 0.0:
            # Vectorised HSV hue rotation.
            rgb_arr = np.array(rgb, dtype=np.float32) / 255.0  # (H, W, 3)
            h_shape = rgb_arr.shape[:2]
            flat = rgb_arr.reshape(-1, 3)  # (N, 3)
            import colorsys

            hsv = np.array(
                [colorsys.rgb_to_hsv(float(r), float(g), float(b)) for r, g, b in flat],
                dtype=np.float32,
            )
            hsv[:, 0] = (hsv[:, 0] + self._hue / 360.0) % 1.0
            out_flat = np.array(
                [colorsys.hsv_to_rgb(float(h), float(s), float(v)) for h, s, v in hsv],
                dtype=np.float32,
            )
            rgb_out = (
                (out_flat.reshape(h_shape[0], h_shape[1], 3) * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )
            rgb = _PILImage.fromarray(rgb_out, "RGB")

        r2, g2, b2 = rgb.split()
        result_img = _PILImage.merge("RGBA", (r2, g2, b2, a))
        result = np.array(result_img, dtype=np.uint8)
        _set_pixels(self._sprite, self._li, self._fi, result)

    def undo(self) -> None:
        assert self._old_pixels is not None
        _set_pixels(self._sprite, self._li, self._fi, self._old_pixels)


# ---------------------------------------------------------------------------
# CanvasResizeCommand
# ---------------------------------------------------------------------------


class CanvasResizeCommand(Command):
    """Resize the canvas, cropping or extending every cel.

    Args:
        sprite: The owning sprite.
        new_width: New canvas width in pixels.
        new_height: New canvas height in pixels.
        offset_x: X position where existing content is placed in the new canvas.
        offset_y: Y position where existing content is placed in the new canvas.
    """

    def __init__(
        self,
        sprite: Sprite,
        new_width: int,
        new_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        self._sprite = sprite
        self._new_width = new_width
        self._new_height = new_height
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._old_width: Optional[int] = None
        self._old_height: Optional[int] = None
        self._saved_cels: Optional[Dict[CelKey, np.ndarray]] = None

    @property
    def description(self) -> str:
        return f"Resize Canvas to {self._new_width}×{self._new_height}"

    def execute(self) -> None:
        self._old_width = self._sprite.width
        self._old_height = self._sprite.height
        self._saved_cels = _save_all_cels(self._sprite)
        self._sprite.resize_canvas(
            self._new_width,
            self._new_height,
            self._offset_x,
            self._offset_y,
        )

    def undo(self) -> None:
        assert self._saved_cels is not None
        assert self._old_width is not None and self._old_height is not None
        # Resize back, then restore pixels.
        self._sprite.resize_canvas(self._old_width, self._old_height)
        _restore_all_cels(self._sprite, self._saved_cels)


# ---------------------------------------------------------------------------
# CropToSelectionCommand
# ---------------------------------------------------------------------------


class CropToSelectionCommand(Command):
    """Crop every cel to the bounding box of the active selection.

    Uses the bounding rectangle of ``sprite.selection_mask`` as the new canvas
    extent.  All layers and frames are cropped consistently so animation and
    layer registration are preserved.  The selection mask is cleared after the
    crop (its pixel coordinates no longer map onto the new canvas) and
    restored on undo.

    Args:
        sprite: The owning sprite.  Must have a non-empty
            :attr:`Sprite.selection_mask`.

    Raises:
        ValueError: If no selection is active or the mask is entirely empty.
    """

    def __init__(self, sprite: Sprite) -> None:
        mask = sprite.selection_mask
        if mask is None or not bool(mask.any()):
            raise ValueError(
                "CropToSelectionCommand requires an active, non-empty selection"
            )
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        self._x0 = int(x_idx[0])
        self._y0 = int(y_idx[0])
        self._w = int(x_idx[-1] - x_idx[0] + 1)
        self._h = int(y_idx[-1] - y_idx[0] + 1)
        self._sprite = sprite
        self._old_width: Optional[int] = None
        self._old_height: Optional[int] = None
        self._saved_cels: Optional[Dict[CelKey, np.ndarray]] = None
        self._saved_mask: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return f"Crop to Selection ({self._w}×{self._h})"

    def execute(self) -> None:
        self._old_width = self._sprite.width
        self._old_height = self._sprite.height
        self._saved_cels = _save_all_cels(self._sprite)
        mask = self._sprite.selection_mask
        self._saved_mask = None if mask is None else mask.copy()
        self._sprite.resize_canvas(self._w, self._h, -self._x0, -self._y0)
        # The bbox occupies the entire new canvas; selection coords are
        # no longer meaningful, so clear it.
        self._sprite.selection_mask = None

    def undo(self) -> None:
        assert self._saved_cels is not None
        assert self._old_width is not None and self._old_height is not None
        self._sprite.resize_canvas(self._old_width, self._old_height)
        _restore_all_cels(self._sprite, self._saved_cels)
        self._sprite.selection_mask = (
            None if self._saved_mask is None else self._saved_mask.copy()
        )


# ---------------------------------------------------------------------------
# AutocropCommand
# ---------------------------------------------------------------------------


class AutocropCommand(Command):
    """Crop the canvas to the union bounding box of all opaque pixels.

    Scans every cel across all layers and frames, finds the smallest axis-aligned
    rectangle that contains every pixel with ``alpha > 0``, and resizes the
    canvas to that rectangle.  All frames and layers are cropped consistently
    so animation and layer registration are preserved.  The selection mask is
    cleared (its pixel coordinates no longer map onto the new canvas) and
    restored on undo.

    Args:
        sprite: The owning sprite.

    Raises:
        ValueError: If the sprite contains no opaque pixels, or if the content
            already fills the entire canvas (nothing to crop).
    """

    def __init__(self, sprite: Sprite) -> None:
        h, w = sprite.height, sprite.width
        any_opaque = np.zeros((h, w), dtype=bool)
        for li in range(sprite.layer_count):
            for fi in range(sprite.frame_count):
                cel = sprite.get_cel(li, fi)
                if cel.pixels is None:
                    continue
                any_opaque |= cel.pixels[..., 3] > 0
        if not bool(any_opaque.any()):
            raise ValueError("AutocropCommand requires at least one opaque pixel")
        rows = np.any(any_opaque, axis=1)
        cols = np.any(any_opaque, axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        x0 = int(x_idx[0])
        y0 = int(y_idx[0])
        new_w = int(x_idx[-1] - x_idx[0] + 1)
        new_h = int(y_idx[-1] - y_idx[0] + 1)
        if x0 == 0 and y0 == 0 and new_w == w and new_h == h:
            raise ValueError("Nothing to crop — content already fills the canvas")
        self._sprite = sprite
        self._x0 = x0
        self._y0 = y0
        self._w = new_w
        self._h = new_h
        self._old_width: Optional[int] = None
        self._old_height: Optional[int] = None
        self._saved_cels: Optional[Dict[CelKey, np.ndarray]] = None
        self._saved_mask: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return f"Autocrop ({self._w}×{self._h})"

    def execute(self) -> None:
        self._old_width = self._sprite.width
        self._old_height = self._sprite.height
        self._saved_cels = _save_all_cels(self._sprite)
        mask = self._sprite.selection_mask
        self._saved_mask = None if mask is None else mask.copy()
        self._sprite.resize_canvas(self._w, self._h, -self._x0, -self._y0)
        # Selection coords no longer map onto the new canvas; clear.
        self._sprite.selection_mask = None

    def undo(self) -> None:
        assert self._saved_cels is not None
        assert self._old_width is not None and self._old_height is not None
        self._sprite.resize_canvas(self._old_width, self._old_height)
        _restore_all_cels(self._sprite, self._saved_cels)
        self._sprite.selection_mask = (
            None if self._saved_mask is None else self._saved_mask.copy()
        )


# ---------------------------------------------------------------------------
# ScaleCommand
# ---------------------------------------------------------------------------


class ScaleCommand(Command):
    """Resample all cels to a new canvas size.

    Args:
        sprite: The owning sprite.
        new_width: Target width in pixels.
        new_height: Target height in pixels.
        method: Resampling method — ``"nearest"`` (default) or ``"bilinear"``.
    """

    def __init__(
        self,
        sprite: Sprite,
        new_width: int,
        new_height: int,
        method: str = "nearest",
    ) -> None:
        self._sprite = sprite
        self._new_width = new_width
        self._new_height = new_height
        self._method = method
        self._old_width: Optional[int] = None
        self._old_height: Optional[int] = None
        self._saved_cels: Optional[Dict[CelKey, np.ndarray]] = None

    @property
    def description(self) -> str:
        return f"Scale to {self._new_width}×{self._new_height}"

    def execute(self) -> None:
        self._old_width = self._sprite.width
        self._old_height = self._sprite.height
        self._saved_cels = _save_all_cels(self._sprite)
        self._sprite.scale_pixels(
            self._new_width, self._new_height, method=self._method
        )

    def undo(self) -> None:
        assert self._saved_cels is not None
        assert self._old_width is not None and self._old_height is not None
        # Restore original size then restore pixels.
        self._sprite.scale_pixels(self._old_width, self._old_height)
        _restore_all_cels(self._sprite, self._saved_cels)


# ---------------------------------------------------------------------------
# ScaleSelectionCommand
# ---------------------------------------------------------------------------


class ScaleSelectionCommand(Command):
    """Resample the selected region of one cel to a new size.

    The bounding box of ``sprite.selection_mask`` is treated as the source.
    Selected pixels are extracted, resampled to ``(new_width, new_height)``,
    and pasted back at the top-left of the original bounding box (clipped to
    the canvas).  The original masked pixels are cleared (made transparent)
    before the scaled pixels are written, and the selection mask is updated
    to cover the new region.

    Args:
        sprite: The owning sprite.
        layer_index: Layer to operate on.
        frame_index: Frame to operate on.
        new_width: Target width of the scaled region in pixels.
        new_height: Target height of the scaled region in pixels.
        method: Resampling method — ``"nearest"`` (default) or ``"bilinear"``.

    Raises:
        ValueError: If there is no active selection, the mask is empty,
            or *new_width* / *new_height* is not positive.
    """

    def __init__(
        self,
        sprite: Sprite,
        layer_index: int,
        frame_index: int,
        new_width: int,
        new_height: int,
        method: str = "nearest",
    ) -> None:
        if new_width <= 0 or new_height <= 0:
            raise ValueError(
                f"Scale size must be positive, got {new_width}x{new_height}"
            )
        if sprite.selection_mask is None or not bool(sprite.selection_mask.any()):
            raise ValueError("ScaleSelectionCommand requires an active selection")
        self._sprite = sprite
        self._li = layer_index
        self._fi = frame_index
        self._new_w = int(new_width)
        self._new_h = int(new_height)
        self._method = method
        self._old_pixels: Optional[np.ndarray] = None
        self._old_mask: Optional[np.ndarray] = None

    @property
    def description(self) -> str:
        return f"Scale Selection to {self._new_w}×{self._new_h}"

    def execute(self) -> None:
        from PIL import Image as _PILImage

        mask = self._sprite.selection_mask
        assert mask is not None  # guarded in __init__
        pixels = _get_pixels(self._sprite, self._li, self._fi)
        self._old_pixels = pixels.copy()
        self._old_mask = mask.copy()

        # Bounding box of the selection.
        sel_rows = np.any(mask, axis=1)
        sel_cols = np.any(mask, axis=0)
        r1 = int(np.where(sel_rows)[0][0])
        r2 = int(np.where(sel_rows)[0][-1])
        c1 = int(np.where(sel_cols)[0][0])
        c2 = int(np.where(sel_cols)[0][-1])

        sub_pixels = pixels[r1 : r2 + 1, c1 : c2 + 1].copy()
        sub_mask = mask[r1 : r2 + 1, c1 : c2 + 1]
        # Zero-out unselected pixels inside the bounding box so they don't
        # contaminate the resampled output.
        sub_pixels[~sub_mask] = 0

        resample = (
            _PILImage.Resampling.BILINEAR
            if self._method == "bilinear"
            else _PILImage.Resampling.NEAREST
        )
        scaled_pixels = np.array(
            _PILImage.fromarray(sub_pixels, mode="RGBA").resize(
                (self._new_w, self._new_h), resample=resample
            ),
            dtype=np.uint8,
        )
        # Resize the mask using nearest neighbour to preserve binary edges.
        mask_img = _PILImage.fromarray(sub_mask.astype(np.uint8) * 255, mode="L")
        scaled_mask = (
            np.array(
                mask_img.resize(
                    (self._new_w, self._new_h),
                    resample=_PILImage.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
            > 0
        )

        h, w = self._sprite.height, self._sprite.width
        result = pixels.copy()
        # Clear original masked pixels.
        result[mask] = 0

        # Compute destination region clipped to the canvas.
        dst_y0 = r1
        dst_x0 = c1
        dst_y1 = min(h, dst_y0 + self._new_h)
        dst_x1 = min(w, dst_x0 + self._new_w)
        src_h = dst_y1 - dst_y0
        src_w = dst_x1 - dst_x0

        new_mask = np.zeros((h, w), dtype=bool)
        if src_h > 0 and src_w > 0:
            src_slice = scaled_pixels[:src_h, :src_w]
            mask_slice = scaled_mask[:src_h, :src_w]
            dst = result[dst_y0:dst_y1, dst_x0:dst_x1]
            dst[mask_slice] = src_slice[mask_slice]
            result[dst_y0:dst_y1, dst_x0:dst_x1] = dst
            new_mask[dst_y0:dst_y1, dst_x0:dst_x1] = mask_slice

        _set_pixels(self._sprite, self._li, self._fi, result)
        self._sprite.selection_mask = new_mask if new_mask.any() else None

    def undo(self) -> None:
        assert self._old_pixels is not None
        assert self._old_mask is not None
        _set_pixels(self._sprite, self._li, self._fi, self._old_pixels)
        self._sprite.selection_mask = self._old_mask.copy()
