# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for core/frame.py."""

import numpy as np
import pytest
from spriter.core.frame import Cel, Frame


class TestFrame:
    def test_default_duration(self):
        frame = Frame()
        assert frame.duration_ms == 100

    def test_custom_duration(self):
        frame = Frame(250)
        assert frame.duration_ms == 250

    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            Frame(0)
        with pytest.raises(ValueError):
            Frame(-1)

    def test_repr(self):
        assert "250" in repr(Frame(250))


class TestCel:
    def _blank(self, w=8, h=8) -> np.ndarray:
        return np.zeros((h, w, 4), dtype=np.uint8)

    def test_empty_cel(self):
        cel = Cel()
        assert cel.is_empty
        assert not cel.is_linked
        assert cel.pixels is None

    def test_cel_with_pixels(self):
        px = self._blank()
        cel = Cel(px)
        assert not cel.is_empty
        assert cel.pixels is not None
        assert cel.pixels.shape == (8, 8, 4)

    def test_cel_linked(self):
        cel = Cel(linked_frame=2)
        assert cel.is_linked
        assert cel.linked_frame == 2
        assert cel.is_empty  # no local pixels

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            Cel(np.zeros((8, 8, 3), dtype=np.uint8))  # wrong channels

    def test_invalid_dtype(self):
        with pytest.raises(ValueError):
            Cel(np.zeros((8, 8, 4), dtype=np.float32))

    def test_clear(self):
        cel = Cel(self._blank())
        cel.clear()
        assert cel.is_empty

    def test_pixels_setter_validation(self):
        cel = Cel()
        with pytest.raises(ValueError):
            cel.pixels = np.zeros((8, 8, 3), dtype=np.uint8)

    def test_repr_empty(self):
        assert "empty" in repr(Cel())

    def test_repr_linked(self):
        assert "linked_frame=3" in repr(Cel(linked_frame=3))

    def test_repr_with_pixels(self):
        assert "8x8" in repr(Cel(self._blank(8, 8)))
