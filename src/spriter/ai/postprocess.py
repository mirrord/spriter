# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Post-processing for diffusion-generated frames.

These helpers turn a raw generated RGBA image into a frame that drops cleanly
into a sprite: the flat background colour is keyed out, the remaining content is
cropped to its bounding box, then uniformly scaled down (never up) to fit the
canvas and centred on a transparent buffer.

All functions here are pure NumPy/Pillow and carry no machine-learning
dependencies, so they can be unit-tested without ``torch``/``diffusers``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..io.spritesheet import detect_background_color


def remove_flat_background(rgba: np.ndarray, *, coverage: float = 0.5) -> np.ndarray:
    """Return a copy of *rgba* with a detected flat background keyed out.

    The dominant border colour is detected via
    :func:`~spriter.io.spritesheet.detect_background_color`; every pixel whose
    RGB matches it is made fully transparent.  When no flat background is found
    the image is returned unchanged (as a copy).

    Args:
        rgba: An ``H×W×4`` ``uint8`` RGBA array.
        coverage: Minimum fraction of border pixels the background colour must
            occupy to be treated as removable.

    Returns:
        A new ``H×W×4`` ``uint8`` RGBA array.
    """
    _validate_rgba(rgba)
    out = rgba.copy()
    color = detect_background_color(out, coverage=coverage)
    if color is None:
        return out
    match = (
        (out[..., 0] == color[0])
        & (out[..., 1] == color[1])
        & (out[..., 2] == color[2])
    )
    out[match] = 0
    return out


def autocrop_rgba(rgba: np.ndarray) -> np.ndarray:
    """Crop *rgba* to the bounding box of its non-transparent pixels.

    Args:
        rgba: An ``H×W×4`` ``uint8`` RGBA array.

    Returns:
        The cropped RGBA array.  When the image is fully transparent the
        original array is returned unchanged (as a copy).
    """
    _validate_rgba(rgba)
    opaque = rgba[..., 3] > 0
    if not bool(opaque.any()):
        return rgba.copy()
    rows = np.any(opaque, axis=1)
    cols = np.any(opaque, axis=0)
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    y0, y1 = int(y_idx[0]), int(y_idx[-1]) + 1
    x0, x1 = int(x_idx[0]), int(x_idx[-1]) + 1
    return rgba[y0:y1, x0:x1].copy()


def fit_into_canvas(
    rgba: np.ndarray,
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray:
    """Uniformly scale *rgba* to fit within the canvas and centre it.

    The image is scaled by a single factor that preserves aspect ratio so that
    neither dimension exceeds the canvas.  Content is only ever scaled *down* —
    an image already smaller than the canvas keeps its size.  The result is
    centred on a fully transparent ``canvas_height×canvas_width`` buffer.

    Args:
        rgba: An ``H×W×4`` ``uint8`` RGBA array.
        canvas_width: Target canvas width in pixels.
        canvas_height: Target canvas height in pixels.

    Returns:
        A ``canvas_height×canvas_width×4`` ``uint8`` RGBA array.
    """
    _validate_rgba(rgba)
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError(
            f"Canvas size must be positive, got {canvas_width}x{canvas_height}"
        )
    src_h, src_w = rgba.shape[:2]
    out = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    if src_h == 0 or src_w == 0:
        return out

    scale = min(canvas_width / src_w, canvas_height / src_h, 1.0)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    if (new_w, new_h) != (src_w, src_h):
        img = Image.fromarray(rgba, mode="RGBA").resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        )
        scaled = np.array(img, dtype=np.uint8)
    else:
        scaled = rgba

    off_x = (canvas_width - new_w) // 2
    off_y = (canvas_height - new_h) // 2
    out[off_y : off_y + new_h, off_x : off_x + new_w] = scaled
    return out


def prepare_generated_frame(
    generated_rgba: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    *,
    coverage: float = 0.5,
) -> np.ndarray:
    """Turn a raw generated image into a canvas-sized sprite frame.

    Removes the flat background, crops to the remaining content, then uniformly
    scales it down to fit and centres it on a transparent canvas-sized buffer.

    Args:
        generated_rgba: The raw generated ``H×W×4`` ``uint8`` RGBA image.
        canvas_width: Target canvas width in pixels.
        canvas_height: Target canvas height in pixels.
        coverage: Background-detection coverage threshold.

    Returns:
        A ``canvas_height×canvas_width×4`` ``uint8`` RGBA array ready to become
        a cel.
    """
    keyed = remove_flat_background(generated_rgba, coverage=coverage)
    cropped = autocrop_rgba(keyed)
    return fit_into_canvas(cropped, canvas_width, canvas_height)


def _validate_rgba(rgba: np.ndarray) -> None:
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expected an (H, W, 4) RGBA array")
    if rgba.dtype != np.uint8:
        raise ValueError("expected a uint8 RGBA array")
