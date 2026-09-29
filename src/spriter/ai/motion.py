# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Pure-NumPy motion estimation for next-frame prediction.

These helpers derive a coarse motion field from two consecutive frames and use
it to extrapolate where content should be in the *next* frame.  The predicted
frame is not meant to be final — it feeds a diffusion ControlNet as structural
guidance (see :mod:`spriter.ai.diffusion`).  Nothing here imports ``torch`` or
``diffusers``, so it is unit-testable without the optional ML dependencies.
"""

from __future__ import annotations

import numpy as np

# Block-matching defaults.  ``block`` is the side of each matched tile and
# ``search`` the maximum per-axis displacement examined, both in pixels.
DEFAULT_BLOCK = 8
DEFAULT_SEARCH = 8

# Weak preference for the smallest displacement that fits equally well; keeps
# flat/ambiguous regions from picking large spurious offsets.
_DISPLACEMENT_BIAS = 0.01


def _luma(rgba: np.ndarray) -> np.ndarray:
    """Return an alpha-weighted luma image (transparent pixels contribute 0)."""
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return luma * alpha


def estimate_flow(
    prev: np.ndarray,
    curr: np.ndarray,
    block: int = DEFAULT_BLOCK,
    search: int = DEFAULT_SEARCH,
) -> np.ndarray:
    """Estimate a per-pixel motion field extrapolating *prev* → *curr* forward.

    Block-matching finds, for each tile of *curr*, the offset into *prev* that
    best explains it.  The content displacement over the pair is the negation of
    that offset; the returned field is that same displacement, so warping *curr*
    by it advances content one more step in the observed direction.

    Args:
        prev: The earlier ``H×W×4`` ``uint8`` RGBA frame.
        curr: The later ``H×W×4`` ``uint8`` RGBA frame (same shape as *prev*).
        block: Side length of each matched tile in pixels.
        search: Maximum per-axis search displacement in pixels.

    Returns:
        An ``H×W×2`` ``float32`` array of ``(dx, dy)`` displacements.
    """
    _validate_pair(prev, curr)
    lp = _luma(prev)
    lc = _luma(curr)
    height, width = lc.shape
    flow = np.zeros((height, width, 2), dtype=np.float32)

    for by in range(0, height, block):
        for bx in range(0, width, block):
            y1 = min(by + block, height)
            x1 = min(bx + block, width)
            ref = lc[by:y1, bx:x1]
            bh, bw = ref.shape
            norm = float(ref.size)
            best_cost = np.inf
            best = (0, 0)
            for dy in range(-search, search + 1):
                sy = by + dy
                if sy < 0 or sy + bh > height:
                    continue
                for dx in range(-search, search + 1):
                    sx = bx + dx
                    if sx < 0 or sx + bw > width:
                        continue
                    cand = lp[sy : sy + bh, sx : sx + bw]
                    cost = float(np.abs(ref - cand).sum())
                    cost += _DISPLACEMENT_BIAS * (abs(dx) + abs(dy)) * norm
                    if cost < best_cost:
                        best_cost = cost
                        best = (dx, dy)
            # Content moved by -offset from prev→curr; extrapolate the same step.
            flow[by:y1, bx:x1, 0] = -best[0]
            flow[by:y1, bx:x1, 1] = -best[1]
    return flow


def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Forward-splat opaque pixels of *frame* by *flow*, leaving holes clear.

    Backward sampling would erase moving objects (the destination cell has zero
    flow), so opaque source pixels are scattered forward to ``p + flow`` instead.
    Uncovered destination pixels stay fully transparent for the diffuser to fill.

    Args:
        frame: An ``H×W×4`` ``uint8`` RGBA frame.
        flow: An ``H×W×2`` ``float32`` displacement field.

    Returns:
        A new ``H×W×4`` ``uint8`` RGBA frame.
    """
    height, width = frame.shape[:2]
    out = np.zeros_like(frame)
    ys, xs = np.nonzero(frame[..., 3] > 0)
    if ys.size == 0:
        return out
    nx = np.round(xs + flow[ys, xs, 0]).astype(np.int64)
    ny = np.round(ys + flow[ys, xs, 1]).astype(np.int64)
    valid = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
    out[ny[valid], nx[valid]] = frame[ys[valid], xs[valid]]
    return out


def predict_next_frame(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Extrapolate the frame after *curr* by warping it along *prev* → *curr*."""
    flow = estimate_flow(prev, curr)
    return warp_frame(curr, flow)


def edge_map(rgba: np.ndarray) -> np.ndarray:
    """Return a normalised ``H×W`` ``uint8`` edge magnitude image.

    A simple central-difference gradient of the alpha-weighted luma; used as the
    ControlNet conditioning image for structural guidance.

    Args:
        rgba: An ``H×W×4`` ``uint8`` RGBA frame.

    Returns:
        An ``H×W`` ``uint8`` array scaled so the strongest edge is 255.
    """
    luma = _luma(rgba)
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    gx[:, 1:-1] = luma[:, 2:] - luma[:, :-2]
    gy[1:-1, :] = luma[2:, :] - luma[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    peak = float(mag.max())
    if peak > 0:
        mag = mag / peak * 255.0
    return mag.astype(np.uint8)


def _validate_pair(prev: np.ndarray, curr: np.ndarray) -> None:
    for name, arr in (("prev", prev), ("curr", curr)):
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError(f"expected {name} to be an (H, W, 4) RGBA array")
    if prev.shape != curr.shape:
        raise ValueError("prev and curr must have the same shape")
