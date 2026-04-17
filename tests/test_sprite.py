# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for core/sprite.py."""

import numpy as np
import pytest
from spriter.core.layer import BlendMode
from spriter.core.sprite import Sprite, _reindex


class TestSpriteCreation:
    def test_basic_creation(self):
        s = Sprite(16, 16)
        assert s.width == 16
        assert s.height == 16
        assert s.color_mode == "RGBA"
        assert s.layer_count == 0
        assert s.frame_count == 0

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            Sprite(0, 16)
        with pytest.raises(ValueError):
            Sprite(16, 0)

    def test_unsupported_color_mode(self):
        with pytest.raises(ValueError):
            Sprite(8, 8, color_mode="RGB")

    def test_repr(self):
        s = Sprite(32, 32)
        assert "32x32" in repr(s)


class TestLayerManagement:
    def test_add_layer(self):
        s = Sprite(8, 8)
        s.add_frame()
        layer = s.add_layer("Base")
        assert s.layer_count == 1
        assert s.layers[0] is layer
        assert layer.name == "Base"

    def test_add_layer_defaults_to_blank_cels(self):
        s = Sprite(8, 8)
        s.add_frame()
        s.add_layer()
        cel = s.get_cel(0, 0)
        assert cel.pixels is not None
        assert cel.pixels.shape == (8, 8, 4)
        assert (cel.pixels == 0).all()

    def test_add_multiple_layers(self):
        s = Sprite(8, 8)
        s.add_frame()
        s.add_layer("A")
        s.add_layer("B")
        s.add_layer("C")
        assert s.layer_count == 3
        assert [l.name for l in s.layers] == ["A", "B", "C"]

    def test_remove_layer(self):
        s = Sprite(8, 8)
        s.add_frame()
        la = s.add_layer("A")
        s.add_layer("B")
        removed = s.remove_layer(0)
        assert removed is la
        assert s.layer_count == 1
        assert s.layers[0].name == "B"

    def test_remove_layer_out_of_range(self):
        s = Sprite(8, 8)
        with pytest.raises(IndexError):
            s.remove_layer(0)

    def test_move_layer(self):
        s = Sprite(8, 8)
        s.add_frame()
        s.add_layer("A")
        s.add_layer("B")
        s.add_layer("C")
        s.move_layer(0, 2)
        assert [l.name for l in s.layers] == ["B", "C", "A"]

    def test_move_layer_same_index(self):
        s = Sprite(8, 8)
        s.add_frame()
        s.add_layer("A")
        s.move_layer(0, 0)  # no-op
        assert s.layer_count == 1


class TestFrameManagement:
    def test_add_frame(self):
        s = Sprite(8, 8)
        frame = s.add_frame(200)
        assert s.frame_count == 1
        assert frame.duration_ms == 200

    def test_add_frame_creates_cels(self):
        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        assert s.get_cel(0, 0) is not None

    def test_remove_frame(self):
        s = Sprite(8, 8)
        fa = s.add_frame(100)
        s.add_frame(200)
        removed = s.remove_frame(0)
        assert removed is fa
        assert s.frame_count == 1
        assert s.frames[0].duration_ms == 200

    def test_remove_frame_out_of_range(self):
        s = Sprite(8, 8)
        with pytest.raises(IndexError):
            s.remove_frame(0)

    def test_move_frame(self):
        s = Sprite(8, 8)
        s.add_frame(100)
        s.add_frame(200)
        s.add_frame(300)
        s.move_frame(0, 2)
        assert [f.duration_ms for f in s.frames] == [200, 300, 100]


class TestCelAccess:
    def test_set_and_get_cel(self):
        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        px = np.full((8, 8, 4), 128, dtype=np.uint8)
        s.set_cel_pixels(0, 0, px)
        cel = s.get_cel(0, 0)
        assert (cel.pixels == 128).all()

    def test_set_cel_wrong_shape(self):
        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        with pytest.raises(ValueError):
            s.set_cel_pixels(0, 0, np.zeros((4, 4, 4), dtype=np.uint8))

    def test_get_cel_invalid_index(self):
        s = Sprite(8, 8)
        s.add_layer()
        s.add_frame()
        with pytest.raises(IndexError):
            s.get_cel(1, 0)
        with pytest.raises(IndexError):
            s.get_cel(0, 1)


class TestCompositeFrame:
    def test_single_opaque_layer(self):
        s = Sprite(4, 4)
        s.add_layer()
        s.add_frame()
        px = np.full((4, 4, 4), 200, dtype=np.uint8)
        px[..., 3] = 255  # fully opaque
        s.set_cel_pixels(0, 0, px)
        result = s.composite_frame(0)
        assert result.shape == (4, 4, 4)
        assert (result[..., :3] == 200).all()

    def test_invisible_layer_excluded(self):
        s = Sprite(4, 4)
        s.add_layer(visible=False)
        s.add_frame()
        px = np.full((4, 4, 4), 255, dtype=np.uint8)
        s.set_cel_pixels(0, 0, px)
        result = s.composite_frame(0)
        assert (result == 0).all()

    def test_two_layers_composited(self):
        s = Sprite(4, 4)
        s.add_layer("Bottom")
        s.add_layer("Top")
        s.add_frame()
        # Bottom layer: fully opaque red
        bottom = np.zeros((4, 4, 4), dtype=np.uint8)
        bottom[..., 0] = 255
        bottom[..., 3] = 255
        s.set_cel_pixels(0, 0, bottom)
        # Top layer: fully opaque green
        top = np.zeros((4, 4, 4), dtype=np.uint8)
        top[..., 1] = 255
        top[..., 3] = 255
        s.set_cel_pixels(1, 0, top)
        result = s.composite_frame(0)
        # Top layer is fully opaque so it eclipses the bottom.
        assert result[0, 0, 1] == 255  # green channel
        assert result[0, 0, 0] == 0  # red eclipsed


class TestReindex:
    def test_moved_element(self):
        assert _reindex(0, 0, 2) == 2

    def test_shift_left(self):
        # Element at index 1 shifts left when 0 moves to 2.
        assert _reindex(1, 0, 2) == 0

    def test_shift_right(self):
        # Element at index 1 shifts right when 2 moves to 0.
        assert _reindex(1, 2, 0) == 2

    def test_unaffected(self):
        assert _reindex(5, 0, 2) == 5
