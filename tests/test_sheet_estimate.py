# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for :func:`spriter.io.spritesheet.estimate_sheet_layout`.

The estimator assumes **no padding between frames** (real-world pixel-art
sheets almost universally pack frames flush) and infers the per-axis frame
size from the periodicity of the per-axis content signature.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from spriter.core.sprite import Sprite
from spriter.io.spritesheet import (
    EstimatedLayout,
    SheetLayout,
    estimate_sheet_layout,
    export_sheet,
)


def _make_sprite(fw: int, fh: int, n: int, *, margin: int = 2) -> Sprite:
    """Build a sprite with *n* frames; each frame draws a small distinctive
    block surrounded by transparent margins (mimics real-world sheets)."""
    s = Sprite(fw, fh)
    s.add_layer("L0")
    for i in range(n):
        s.add_frame()
        pixels = np.zeros((fh, fw, 4), dtype=np.uint8)
        c = np.array(
            [(i * 53) % 256, (i * 97 + 40) % 256, (i * 31 + 80) % 256, 255],
            dtype=np.uint8,
        )
        pixels[margin : fh - margin, margin : fw - margin] = c
        s.set_cel_pixels(0, i, pixels)
    return s


def test_estimate_horizontal_no_padding(tmp_path):
    s = _make_sprite(16, 16, 4)
    out = tmp_path / "h.png"
    export_sheet(s, out, layout=SheetLayout.HORIZONTAL, padding=0)
    assert estimate_sheet_layout(out) == EstimatedLayout(16, 16, 0)


def test_estimate_vertical_no_padding(tmp_path):
    s = _make_sprite(24, 16, 5)
    out = tmp_path / "v.png"
    export_sheet(s, out, layout=SheetLayout.VERTICAL, padding=0)
    assert estimate_sheet_layout(out) == EstimatedLayout(24, 16, 0)


def test_estimate_grid_no_padding(tmp_path):
    s = _make_sprite(24, 16, 6)
    out = tmp_path / "grid.png"
    export_sheet(s, out, layout=SheetLayout.GRID, cols=3, padding=0)
    assert estimate_sheet_layout(out) == EstimatedLayout(24, 16, 0)


def test_estimate_horizontal_32x32(tmp_path):
    s = _make_sprite(32, 32, 8)
    out = tmp_path / "h32.png"
    export_sheet(s, out, layout=SheetLayout.HORIZONTAL, padding=0)
    assert estimate_sheet_layout(out) == EstimatedLayout(32, 32, 0)


def test_estimate_solid_background_no_alpha(tmp_path):
    """Sheet without transparency: background detected from corner sample."""
    bg = np.array([255, 255, 255], dtype=np.uint8)
    fw, fh, n = 16, 16, 4
    sheet_w = n * fw
    arr = np.broadcast_to(bg, (fh, sheet_w, 3)).copy()
    for i in range(n):
        # Centre a smaller distinctive block inside each cell so the
        # signature has structure within the frame.
        x = i * fw + 4
        arr[4 : fh - 4, x : x + 8] = [(i * 60) % 256, 30, 200]
    out = tmp_path / "solid.png"
    Image.fromarray(arr, mode="RGB").save(str(out))
    assert estimate_sheet_layout(out) == EstimatedLayout(fw, fh, 0)


def test_estimate_with_centered_sprites_in_cells():
    """Realistic case: each 32×32 cell contains a smaller sprite with whitespace.

    The earlier gap-based detector mistook this internal whitespace for
    inter-frame padding; the new periodicity detector sees through it.
    """
    fw, fh, n = 32, 32, 5
    sheet_w = n * fw
    arr = np.zeros((fh, sheet_w, 4), dtype=np.uint8)
    for i in range(n):
        x = i * fw
        # 16×20 sprite centred in a 32×32 cell, distinct colour per frame.
        arr[6:26, x + 8 : x + 24] = [(i * 47) % 256, 80, 160, 255]
    est = estimate_sheet_layout(arr)
    assert est == EstimatedLayout(fw, fh, 0)


def test_estimate_fully_opaque_uniform_does_not_raise():
    """Degenerate input (all one colour) must not raise; returns *something*."""
    arr = np.full((32, 64, 4), 200, dtype=np.uint8)
    est = estimate_sheet_layout(arr)
    assert isinstance(est, EstimatedLayout)
    assert 1 <= est.frame_width <= 64
    assert 1 <= est.frame_height <= 32
    assert est.padding == 0


def test_estimate_accepts_ndarray_and_pil():
    fw, fh, n = 16, 16, 3
    sheet_w = n * fw
    arr = np.zeros((fh, sheet_w, 4), dtype=np.uint8)
    for i in range(n):
        # Centred 8×8 block per cell → transparent margins between sprites.
        x = i * fw + 4
        arr[4:12, x : x + 8] = [(i * 80) % 256, 50, 30, 255]
    est_arr = estimate_sheet_layout(arr)
    est_img = estimate_sheet_layout(Image.fromarray(arr, mode="RGBA"))
    assert est_arr == est_img
    assert est_arr == EstimatedLayout(fw, fh, 0)


def test_estimate_padding_always_zero(tmp_path):
    """Per the new contract the estimator never reports non-zero padding."""
    s = _make_sprite(16, 16, 4)
    out = tmp_path / "p.png"
    # Even when padding *is* present in the source, the estimator returns 0;
    # the user can override in the UI.
    export_sheet(s, out, layout=SheetLayout.HORIZONTAL, padding=2)
    assert estimate_sheet_layout(out).padding == 0


def test_estimate_is_fast_on_large_sheet():
    """Sanity check: estimator should be sub-second on a 2K sheet."""
    import time

    fw = fh = 32
    cols, rows = 64, 64  # 2048 × 2048
    sheet_w, sheet_h = cols * fw, rows * fh
    arr = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    arr[..., 3] = 255
    arr[..., 0] = 128
    t0 = time.perf_counter()
    est = estimate_sheet_layout(arr)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"estimate_sheet_layout too slow: {elapsed:.3f}s"
    assert sheet_w % est.frame_width == 0
    assert sheet_h % est.frame_height == 0
    assert est.padding == 0
