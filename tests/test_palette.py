# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for core/palette.py."""

import pytest
from spriter.core.palette import Palette, _validate_color


class TestPaletteBasics:
    def test_empty_palette(self):
        p = Palette()
        assert len(p) == 0

    def test_add_color(self):
        p = Palette()
        idx = p.add((255, 0, 0, 255))
        assert idx == 0
        assert len(p) == 1
        assert p[0] == (255, 0, 0, 255)

    def test_add_rgb_normalizes_to_rgba(self):
        p = Palette()
        p.add((100, 150, 200))
        assert p[0] == (100, 150, 200, 255)

    def test_add_clamps(self):
        p = Palette()
        p.add((300, -10, 128, 128))
        assert p[0] == (255, 0, 128, 128)

    def test_full_palette_raises(self):
        p = Palette([(i, i, i, 255) for i in range(256)])
        with pytest.raises(ValueError):
            p.add((0, 0, 0, 255))

    def test_remove(self):
        p = Palette([(255, 0, 0, 255), (0, 255, 0, 255)])
        p.remove(0)
        assert len(p) == 1
        assert p[0] == (0, 255, 0, 255)

    def test_move(self):
        p = Palette([(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)])
        p.move(0, 2)
        assert p[2] == (255, 0, 0, 255)

    def test_iter(self):
        colors = [(i, i, i, 255) for i in range(5)]
        p = Palette(colors)
        assert list(p) == colors

    def test_repr(self):
        p = Palette([(0, 0, 0, 255)])
        assert "1 colors" in repr(p)

    def test_sort_by_hue(self):
        # Red hue ~0°, Green ~120°, Blue ~240°
        p = Palette([(0, 0, 255, 255), (255, 0, 0, 255), (0, 255, 0, 255)])
        p.sort_by_hue()
        # After sort: red (≈0°), green (≈120°), blue (≈240°)
        assert p[0] == (255, 0, 0, 255)
        assert p[1] == (0, 255, 0, 255)
        assert p[2] == (0, 0, 255, 255)


class TestPaletteIO:
    def test_jasc_round_trip(self, tmp_path):
        p = Palette([(255, 0, 0, 255), (0, 128, 64, 255)])
        dest = tmp_path / "test.pal"
        p.to_jasc(dest)
        loaded = Palette.from_jasc(dest)
        assert len(loaded) == 2
        assert loaded[0] == (255, 0, 0, 255)
        assert loaded[1] == (0, 128, 64, 255)

    def test_jasc_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.pal"
        bad.write_text("not a pal file\n")
        with pytest.raises(ValueError):
            Palette.from_jasc(bad)

    def test_gpl_round_trip(self, tmp_path):
        p = Palette([(10, 20, 30, 255), (40, 50, 60, 255)])
        dest = tmp_path / "test.gpl"
        p.to_gpl(dest)
        loaded = Palette.from_gpl(dest)
        assert len(loaded) == 2
        assert loaded[0] == (10, 20, 30, 255)
        assert loaded[1] == (40, 50, 60, 255)

    def test_gpl_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.gpl"
        bad.write_text("not a gimp palette\n")
        with pytest.raises(ValueError):
            Palette.from_gpl(bad)

    def test_hex_list_round_trip(self, tmp_path):
        p = Palette([(255, 128, 0, 255), (0, 64, 192, 200)])
        dest = tmp_path / "test.hex"
        p.to_hex_list(dest)
        loaded = Palette.from_hex_list(dest)
        assert len(loaded) == 2
        assert loaded[0] == (255, 128, 0, 255)
        assert loaded[1] == (0, 64, 192, 200)

    def test_hex_list_skips_comments(self, tmp_path):
        dest = tmp_path / "test.hex"
        dest.write_text("# comment\nFF0000FF\n; skip\n00FF00FF\n")
        loaded = Palette.from_hex_list(dest)
        assert len(loaded) == 2


class TestValidateColor:
    def test_rgb_tuple(self):
        assert _validate_color((10, 20, 30)) == (10, 20, 30, 255)

    def test_rgba_tuple(self):
        assert _validate_color((10, 20, 30, 128)) == (10, 20, 30, 128)

    def test_bad_length(self):
        with pytest.raises(ValueError):
            _validate_color((1, 2))  # type: ignore[arg-type]
