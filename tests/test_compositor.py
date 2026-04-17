# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the compositing engine (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from spriter.core.compositor import _blend_rgb, composite_frame
from spriter.core.layer import BlendMode
from spriter.core.sprite import Sprite


def _make_sprite_1x1(r, g, b, a=255) -> Sprite:
    """Return a 1×1 sprite with one fully-opaque layer/frame filled with (r,g,b,a)."""
    s = Sprite(1, 1)
    s.add_layer()
    s.add_frame()
    px = np.array([[[r, g, b, a]]], dtype=np.uint8)
    s.set_cel_pixels(0, 0, px)
    return s


def _make_two_layer_sprite(
    bottom_rgba, top_rgba, top_mode=BlendMode.NORMAL, top_opacity=255
) -> Sprite:
    """1×1 sprite with bottom layer (normal) and top layer with given blend mode."""
    s = Sprite(1, 1)
    bottom = s.add_layer("Bottom")
    s.add_frame()
    px = np.array([[list(bottom_rgba)]], dtype=np.uint8)
    s.set_cel_pixels(0, 0, px)

    top = s.add_layer("Top", blend_mode=top_mode, opacity=top_opacity)
    px2 = np.array([[list(top_rgba)]], dtype=np.uint8)
    s.set_cel_pixels(1, 0, px2)
    return s


# ---------------------------------------------------------------------------
# composite_frame — basic cases
# ---------------------------------------------------------------------------


class TestCompositeFrame:
    def test_returns_correct_shape(self):
        s = Sprite(8, 16)
        s.add_layer()
        s.add_frame()
        result = composite_frame(s, 0)
        assert result.shape == (16, 8, 4)
        assert result.dtype == np.uint8

    def test_transparent_layer(self):
        s = Sprite(4, 4)
        s.add_layer()
        s.add_frame()
        result = composite_frame(s, 0)
        assert np.all(result == 0), "Empty layer should produce transparent output"

    def test_single_opaque_pixel(self):
        s = _make_sprite_1x1(200, 100, 50)
        result = composite_frame(s, 0)
        np.testing.assert_array_equal(result[0, 0], [200, 100, 50, 255])

    def test_invisible_layer_not_composited(self):
        s = _make_sprite_1x1(200, 100, 50)
        s._layers[0].visible = False
        result = composite_frame(s, 0)
        assert result[0, 0, 3] == 0, "Invisible layer should produce transparent output"

    def test_invalid_frame_index_raises(self):
        s = Sprite(4, 4)
        s.add_layer()
        s.add_frame()
        with pytest.raises(IndexError):
            composite_frame(s, 1)

    def test_layer_opacity(self):
        """Half-opacity layer over transparent should produce half-alpha output."""
        s = Sprite(1, 1)
        s.add_layer("L", opacity=128)
        s.add_frame()
        px = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)
        s.set_cel_pixels(0, 0, px)
        result = composite_frame(s, 0)
        # RGB should be present, alpha ≈ 128
        assert (
            abs(int(result[0, 0, 3]) - 128) <= 2
        ), f"Expected ~128 alpha, got {result[0,0,3]}"

    def test_normal_over_compositing(self):
        """Fully-opaque red over fully-opaque blue → only red visible (src on top)."""
        s = _make_two_layer_sprite((0, 0, 255, 255), (255, 0, 0, 255), BlendMode.NORMAL)
        result = composite_frame(s, 0)
        # Top layer fully covers bottom; result should be red.
        np.testing.assert_array_equal(result[0, 0], [255, 0, 0, 255])


# ---------------------------------------------------------------------------
# Blend modes — 1×1 pixel spot-checks
# ---------------------------------------------------------------------------


class TestBlendModes:
    def _blend_1x1(self, src_rgba, dst_rgba, mode) -> np.ndarray:
        """Composite a single top-layer pixel (src) over a bottom pixel (dst)."""
        s = _make_two_layer_sprite(dst_rgba, src_rgba, mode)
        return composite_frame(s, 0)[0, 0]

    def test_multiply_red_over_blue(self):
        result = self._blend_1x1((255, 0, 0, 255), (0, 0, 255, 255), BlendMode.MULTIPLY)
        # MULTIPLY: R=1*0=0, G=0*0=0, B=0*1=0 → black
        np.testing.assert_array_equal(result, [0, 0, 0, 255])

    def test_multiply_white_is_identity(self):
        """Multiplying by white should leave the destination unchanged."""
        result = self._blend_1x1(
            (255, 255, 255, 255), (100, 150, 200, 255), BlendMode.MULTIPLY
        )
        np.testing.assert_array_equal(result, [100, 150, 200, 255])

    def test_multiply_black_yields_black(self):
        result = self._blend_1x1(
            (0, 0, 0, 255), (200, 100, 50, 255), BlendMode.MULTIPLY
        )
        np.testing.assert_array_equal(result, [0, 0, 0, 255])

    def test_screen_red_over_blue(self):
        result = self._blend_1x1((255, 0, 0, 255), (0, 0, 255, 255), BlendMode.SCREEN)
        # SCREEN: R=1-(0)(1)=1, G=1-(1)(1)=0, B=1-(1)(0)=1 → (255,0,255)
        np.testing.assert_array_equal(result, [255, 0, 255, 255])

    def test_screen_black_is_identity(self):
        """Screening with black should leave the destination approximately unchanged."""
        # float32 round-trip: 100/255 * 255 ≈ 99.999 → 99, so allow ±1 tolerance.
        result = self._blend_1x1((0, 0, 0, 255), (100, 150, 200, 255), BlendMode.SCREEN)
        np.testing.assert_allclose(result.astype(int), [100, 150, 200, 255], atol=1)

    def test_darken(self):
        result = self._blend_1x1(
            (200, 100, 50, 255), (100, 200, 150, 255), BlendMode.DARKEN
        )
        # DARKEN: min per channel → (100, 100, 50)
        np.testing.assert_array_equal(result[:3], [100, 100, 50])

    def test_lighten(self):
        result = self._blend_1x1(
            (200, 100, 50, 255), (100, 200, 150, 255), BlendMode.LIGHTEN
        )
        # LIGHTEN: max per channel → (200, 200, 150)
        np.testing.assert_array_equal(result[:3], [200, 200, 150])

    def test_overlay_dark_dst(self):
        """Overlay with dst < 0.5 behaves like Multiply×2."""
        # dst = (64, 64, 64) ≈ 0.25; src = (128, 128, 128) = 0.5
        # overlay = 2 × 0.5 × 0.25 = 0.25 → 64
        result = self._blend_1x1(
            (128, 128, 128, 255), (64, 64, 64, 255), BlendMode.OVERLAY
        )
        expected = int(round(2 * 0.5 * (64 / 255) * 255))
        assert abs(int(result[0]) - expected) <= 2

    def test_overlay_light_dst(self):
        """Overlay with dst > 0.5 behaves like Screen×2 - 1."""
        # dst ≈ 0.75, src = 0.5
        # overlay = 1 - 2*(1-0.5)*(1-0.75) = 1 - 2*0.5*0.25 = 1 - 0.25 = 0.75 → ~191
        result = self._blend_1x1(
            (128, 128, 128, 255), (192, 192, 192, 255), BlendMode.OVERLAY
        )
        expected = int(round((1 - 2 * 0.5 * (1 - 192 / 255)) * 255))
        assert abs(int(result[0]) - expected) <= 2


# ---------------------------------------------------------------------------
# _blend_rgb unit tests
# ---------------------------------------------------------------------------


class TestBlendRgb:
    def _ones(self):
        return np.ones((1, 1, 3), dtype=np.float32)

    def _zeros(self):
        return np.zeros((1, 1, 3), dtype=np.float32)

    def _val(self, v: float):
        return np.full((1, 1, 3), v, dtype=np.float32)

    def test_normal_returns_src(self):
        src = self._val(0.7)
        dst = self._val(0.3)
        result = _blend_rgb(src, dst, BlendMode.NORMAL)
        np.testing.assert_allclose(result, src)

    def test_multiply_zero_src(self):
        result = _blend_rgb(self._zeros(), self._val(0.5), BlendMode.MULTIPLY)
        np.testing.assert_allclose(result, 0.0)

    def test_screen_ones_src(self):
        result = _blend_rgb(self._ones(), self._val(0.5), BlendMode.SCREEN)
        np.testing.assert_allclose(result, 1.0)

    def test_darken_picks_min(self):
        src = self._val(0.3)
        dst = self._val(0.7)
        result = _blend_rgb(src, dst, BlendMode.DARKEN)
        np.testing.assert_allclose(result, 0.3)

    def test_lighten_picks_max(self):
        src = self._val(0.3)
        dst = self._val(0.7)
        result = _blend_rgb(src, dst, BlendMode.LIGHTEN)
        np.testing.assert_allclose(result, 0.7)

    def test_unknown_mode_falls_back_to_normal(self):
        """An unrecognised blend mode should act like NORMAL."""
        src = self._val(0.6)
        result = _blend_rgb(src, self._val(0.2), "unknown_mode")  # type: ignore[arg-type]
        np.testing.assert_allclose(result, src)
