# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Compositing engine — flattens visible layers with full blend-mode support.

All blend math is performed in float32 with values normalised to [0, 1].
The final result is returned as a uint8 RGBA array.

Supported blend modes (matching :class:`~spriter.core.layer.BlendMode`):

* NORMAL  — standard Porter-Duff "over"
* MULTIPLY — src × dst
* SCREEN   — 1 − (1 − src)(1 − dst)
* OVERLAY  — context-dependent mix of Multiply and Screen
* DARKEN   — min(src, dst)
* LIGHTEN  — max(src, dst)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .sprite import Sprite

from .layer import BlendMode

# ---------------------------------------------------------------------------
# Buffer pool — reuse large float32 scratch arrays between calls to avoid
# the multi-megabyte allocations that dominate runtime for big canvases.
# ---------------------------------------------------------------------------

_BUFFER_POOL: dict[Tuple[int, int, int], list[np.ndarray]] = {}


def _take_buffer(shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    key = (
        (shape, dtype.str)
        if False
        else (shape[0], shape[1], shape[2] if len(shape) > 2 else 1)
    )
    pool = _BUFFER_POOL.get(key)
    if pool:
        buf = pool.pop()
        if buf.shape == shape and buf.dtype == dtype:
            return buf
    return np.empty(shape, dtype=dtype)


def _return_buffer(buf: np.ndarray) -> None:
    shape = buf.shape
    key = (shape[0], shape[1], shape[2] if len(shape) > 2 else 1)
    pool = _BUFFER_POOL.setdefault(key, [])
    if len(pool) < 4:  # cap to avoid unbounded retention
        pool.append(buf)


def composite_frame(
    sprite: "Sprite",
    frame_index: int,
    *,
    layer_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Composite visible layers for *frame_index* into a single RGBA image.

    Layers are composited bottom-to-top.  Each layer's blend mode and opacity
    are respected.  Invisible layers are skipped.

    Args:
        sprite: The sprite document to composite.
        frame_index: Index of the frame to composite.
        layer_range: Optional ``(start, stop)`` half-open range restricting
            which layer indices are included.  Defaults to all layers.

    Returns:
        RGBA uint8 :class:`numpy.ndarray` of shape ``(height, width, 4)``.

    Raises:
        IndexError: If *frame_index* is out of range.
    """
    sprite._validate_frame_index(frame_index)  # type: ignore[attr-defined]

    h, w = sprite.height, sprite.width
    layers = sprite._layers  # type: ignore[attr-defined]
    if layer_range is None:
        start, stop = 0, len(layers)
    else:
        start, stop = layer_range
        start = max(0, start)
        stop = min(len(layers), stop)

    # Collect the visible cels in the requested range.
    visible: list[tuple[int, "object"]] = []  # (layer_idx, layer)
    for layer_idx in range(start, stop):
        layer = layers[layer_idx]
        if not layer.visible:
            continue
        cel = sprite.get_cel(layer_idx, frame_index)
        if cel.pixels is None:
            continue
        visible.append((layer_idx, layer))

    if not visible:
        return np.zeros((h, w, 4), dtype=np.uint8)

    # Fast-path: a single fully-opaque NORMAL layer with opacity 255.
    # Just return a copy of the cel pixels.
    if len(visible) == 1:
        layer_idx, layer = visible[0]
        if layer.blend_mode == BlendMode.NORMAL and layer.opacity == 255:
            cel = sprite.get_cel(layer_idx, frame_index)
            return cel.pixels.copy()

    # Fast-path: every visible layer uses NORMAL blend with opacity 255.
    # Pillow's Image.alpha_composite is a tight C loop and ~10× faster than
    # the float32 NumPy pipeline on multi-megapixel canvases.
    if all(
        layer.blend_mode == BlendMode.NORMAL and layer.opacity == 255
        for _, layer in visible
    ):
        from PIL import Image

        layer_idx0, _ = visible[0]
        base_pixels = sprite.get_cel(layer_idx0, frame_index).pixels
        base = Image.fromarray(base_pixels, mode="RGBA").copy()
        for layer_idx, _ in visible[1:]:
            top_pixels = sprite.get_cel(layer_idx, frame_index).pixels
            top = Image.fromarray(top_pixels, mode="RGBA")
            base.alpha_composite(top)
        return np.asarray(base, dtype=np.uint8)

    # Generic path — accumulate in float32, but vectorize per-channel math
    # and reuse pooled buffers for the big allocations.
    result = _take_buffer((h, w, 4), np.float32)
    result.fill(0.0)
    src_rgb = _take_buffer((h, w, 3), np.float32)
    src_a = _take_buffer((h, w), np.float32)
    dst_rgb = _take_buffer((h, w, 3), np.float32)
    dst_a = _take_buffer((h, w), np.float32)
    out_a = _take_buffer((h, w), np.float32)
    inv_sa = _take_buffer((h, w), np.float32)
    safe = _take_buffer((h, w), np.float32)

    try:
        for layer_idx, layer in visible:
            cel = sprite.get_cel(layer_idx, frame_index)
            pixels = cel.pixels  # uint8 (H, W, 4)

            # Quick reject: all-zero alpha contributes nothing.
            if not np.any(pixels[..., 3]):
                continue

            alpha_scale = layer.opacity / 255.0
            np.multiply(pixels[..., :3], 1.0 / 255.0, out=src_rgb, dtype=np.float32)
            np.multiply(
                pixels[..., 3], alpha_scale / 255.0, out=src_a, dtype=np.float32
            )

            np.multiply(result[..., :3], 1.0 / 255.0, out=dst_rgb, dtype=np.float32)
            np.multiply(result[..., 3], 1.0 / 255.0, out=dst_a, dtype=np.float32)

            blended_rgb = _blend_rgb(src_rgb, dst_rgb, layer.blend_mode)

            # out_a = src_a + dst_a * (1 - src_a)
            np.subtract(1.0, src_a, out=inv_sa)
            np.multiply(dst_a, inv_sa, out=out_a)
            np.add(out_a, src_a, out=out_a)

            # safe = where(out_a > 0, out_a, 1) — avoid divide-by-zero
            np.copyto(safe, out_a)
            safe[out_a <= 0.0] = 1.0

            # numerator: blended_rgb * src_a[...,None] + dst_rgb * (dst_a*(1-src_a))[...,None]
            # We can compute dst_a*(1-src_a) reusing inv_sa scratch.
            np.multiply(dst_a, inv_sa, out=inv_sa)  # reuse — now holds dst_a*(1-src_a)

            sa3 = src_a[..., None]
            wa3 = inv_sa[..., None]
            # result_rgb = (blended_rgb * sa3 + dst_rgb * wa3) / safe[...,None] * 255
            np.multiply(blended_rgb, sa3, out=blended_rgb)
            np.multiply(dst_rgb, wa3, out=dst_rgb)
            np.add(blended_rgb, dst_rgb, out=blended_rgb)
            np.divide(blended_rgb, safe[..., None], out=blended_rgb)
            np.multiply(blended_rgb, 255.0, out=result[..., :3])

            np.multiply(out_a, 255.0, out=result[..., 3])

        np.clip(result, 0, 255, out=result)
        out = result.astype(np.uint8)
        return out
    finally:
        _return_buffer(result)
        _return_buffer(src_rgb)
        _return_buffer(src_a)
        _return_buffer(dst_rgb)
        _return_buffer(dst_a)
        _return_buffer(out_a)
        _return_buffer(inv_sa)
        _return_buffer(safe)


# ---------------------------------------------------------------------------
# Blend-mode helpers
# ---------------------------------------------------------------------------


def _blend_rgb(
    src: np.ndarray,
    dst: np.ndarray,
    mode: BlendMode,
) -> np.ndarray:
    """Blend two normalised RGB arrays according to *mode*.

    Args:
        src: Source RGB values in [0, 1], shape ``(H, W, 3)``.
        dst: Destination RGB values in [0, 1], shape ``(H, W, 3)``.
        mode: The blend mode to apply.

    Returns:
        Blended RGB values in [0, 1], shape ``(H, W, 3)``.
    """
    if mode == BlendMode.NORMAL:
        return src
    if mode == BlendMode.MULTIPLY:
        return src * dst
    if mode == BlendMode.SCREEN:
        return 1.0 - (1.0 - src) * (1.0 - dst)
    if mode == BlendMode.OVERLAY:
        return np.where(
            dst < 0.5,
            2.0 * src * dst,
            1.0 - 2.0 * (1.0 - src) * (1.0 - dst),
        )
    if mode == BlendMode.DARKEN:
        return np.minimum(src, dst)
    if mode == BlendMode.LIGHTEN:
        return np.maximum(src, dst)
    # Unknown mode — fall back to NORMAL.
    return src
