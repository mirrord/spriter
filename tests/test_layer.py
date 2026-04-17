# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for core/layer.py."""

import pytest
from spriter.core.layer import BlendMode, Layer


class TestLayer:
    def test_defaults(self):
        layer = Layer()
        assert layer.name == "Layer"
        assert layer.visible is True
        assert layer.locked is False
        assert layer.opacity == 255
        assert layer.blend_mode == BlendMode.NORMAL

    def test_custom_values(self):
        layer = Layer(
            "Background",
            visible=False,
            locked=True,
            opacity=128,
            blend_mode=BlendMode.MULTIPLY,
        )
        assert layer.name == "Background"
        assert layer.visible is False
        assert layer.locked is True
        assert layer.opacity == 128
        assert layer.blend_mode == BlendMode.MULTIPLY

    def test_opacity_boundary_values(self):
        Layer(opacity=0)
        Layer(opacity=255)

    def test_opacity_invalid_high(self):
        with pytest.raises(ValueError):
            Layer(opacity=256)

    def test_opacity_invalid_low(self):
        with pytest.raises(ValueError):
            Layer(opacity=-1)

    def test_repr(self):
        layer = Layer("Fg")
        assert "Fg" in repr(layer)


class TestBlendMode:
    def test_all_modes_exist(self):
        modes = {m.value for m in BlendMode}
        assert modes == {"normal", "multiply", "screen", "overlay", "darken", "lighten"}
