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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .sprite import Sprite

from .layer import BlendMode


def composite_frame(sprite: "Sprite", frame_index: int) -> np.ndarray:
    """Composite all visible layers for *frame_index* into a single RGBA image.

    Layers are composited bottom-to-top.  Each layer's blend mode and opacity
    are respected.  Invisible layers are skipped.

    Args:
        sprite: The sprite document to composite.
        frame_index: Index of the frame to composite.

    Returns:
        RGBA uint8 :class:`numpy.ndarray` of shape ``(height, width, 4)``.

    Raises:
        IndexError: If *frame_index* is out of range.
    """
    sprite._validate_frame_index(frame_index)  # type: ignore[attr-defined]

    # result stores RGB in [0, 255] and A in [0, 255] as float32 accumulators.
    result = np.zeros((sprite.height, sprite.width, 4), dtype=np.float32)

    for layer_idx, layer in enumerate(sprite._layers):  # type: ignore[attr-defined]
        if not layer.visible:
            continue
        cel = sprite.get_cel(layer_idx, frame_index)
        if cel.pixels is None:
            continue

        src = cel.pixels.astype(np.float32)  # [0, 255]
        src_rgb_norm = src[..., :3] / 255.0  # [0, 1] shape (H, W, 3)
        alpha_scale = layer.opacity / 255.0
        src_a = (src[..., 3] / 255.0) * alpha_scale  # [0, 1] shape (H, W)

        dst_rgb_norm = result[..., :3] / 255.0  # [0, 1]
        dst_a = result[..., 3] / 255.0  # [0, 1]

        # Apply the layer's blend mode to the RGB channels.
        blended_rgb = _blend_rgb(src_rgb_norm, dst_rgb_norm, layer.blend_mode)

        # Porter-Duff "over":  out_a = src_a + dst_a × (1 − src_a)
        out_a = src_a + dst_a * (1.0 - src_a)
        safe_out_a = np.where(out_a > 0.0, out_a, 1.0)  # avoid ÷0

        for ch in range(3):
            result[..., ch] = (
                (
                    blended_rgb[..., ch] * src_a
                    + dst_rgb_norm[..., ch] * dst_a * (1.0 - src_a)
                )
                / safe_out_a
                * 255.0
            )
        result[..., 3] = out_a * 255.0

    return np.clip(result, 0, 255).astype(np.uint8)


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
