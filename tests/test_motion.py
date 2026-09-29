# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the pure-NumPy motion estimation used by next-frame prediction."""

from __future__ import annotations

import numpy as np


def _block_frame(size=32, block=8, x=4, y=4, color=(255, 0, 0)):
    """An opaque coloured block on a transparent canvas at (x, y)."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[y : y + block, x : x + block, :3] = color
    arr[y : y + block, x : x + block, 3] = 255
    return arr


class TestEstimateFlow:
    def test_detects_rightward_translation(self):
        from spriter.ai.motion import estimate_flow

        prev = _block_frame(x=4, y=4)
        curr = _block_frame(x=8, y=4)  # moved +4 in x
        flow = estimate_flow(prev, curr, block=8, search=8)
        # Flow inside the moved block should point further right (+x).
        fx = flow[8:16, 8:16, 0].mean()
        fy = flow[8:16, 8:16, 1].mean()
        assert fx > 1.0
        assert abs(fy) < 1.0

    def test_static_scene_has_near_zero_flow(self):
        from spriter.ai.motion import estimate_flow

        frame = _block_frame(x=8, y=8)
        flow = estimate_flow(frame, frame, block=8, search=8)
        assert np.abs(flow).max() < 1.0

    def test_shape_matches_input(self):
        from spriter.ai.motion import estimate_flow

        prev = _block_frame()
        curr = _block_frame(x=6)
        flow = estimate_flow(prev, curr)
        assert flow.shape == (32, 32, 2)


class TestWarpFrame:
    def test_forward_splat_moves_content(self):
        from spriter.ai.motion import warp_frame

        frame = _block_frame(size=32, block=8, x=4, y=4)
        flow = np.zeros((32, 32, 2), dtype=np.float32)
        flow[..., 0] = 4  # shift everything +4 in x
        out = warp_frame(frame, flow)
        # Original location cleared, new location opaque.
        assert out[8, 4, 3] == 0
        assert out[8, 12, 3] == 255

    def test_holes_left_transparent(self):
        from spriter.ai.motion import warp_frame

        frame = _block_frame()
        flow = np.zeros((32, 32, 2), dtype=np.float32)
        flow[..., 0] = 100  # push everything off-canvas
        out = warp_frame(frame, flow)
        assert out[..., 3].max() == 0

    def test_out_shape_and_dtype(self):
        from spriter.ai.motion import warp_frame

        frame = _block_frame()
        flow = np.zeros((32, 32, 2), dtype=np.float32)
        out = warp_frame(frame, flow)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8


class TestPredictNextFrame:
    def test_extrapolates_motion(self):
        from spriter.ai.motion import predict_next_frame

        prev = _block_frame(x=4, y=8)
        curr = _block_frame(x=8, y=8)  # moved +4 in x
        pred = predict_next_frame(prev, curr)
        # Predicted block should have advanced further right than curr.
        opaque_cols = np.where(pred[..., 3].any(axis=0))[0]
        assert opaque_cols.size > 0
        assert opaque_cols.min() >= 10  # advanced past the current x=8

    def test_shape_matches_input(self):
        from spriter.ai.motion import predict_next_frame

        prev = _block_frame()
        curr = _block_frame(x=6)
        pred = predict_next_frame(prev, curr)
        assert pred.shape == (32, 32, 4)


class TestEdgeMap:
    def test_flat_region_has_no_edges(self):
        from spriter.ai.motion import edge_map

        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[..., :3] = 128
        arr[..., 3] = 255
        edges = edge_map(arr)
        assert edges.shape == (16, 16)
        assert int(edges.max()) == 0

    def test_block_boundary_produces_edges(self):
        from spriter.ai.motion import edge_map

        frame = _block_frame(size=32, block=8, x=8, y=8, color=(255, 255, 255))
        edges = edge_map(frame)
        assert int(edges.max()) > 0
