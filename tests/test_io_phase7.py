# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Phase 7 tests — Import / Export.

All tests are headless (no Qt required).  They exercise round-trip fidelity
for PNG, GIF, sprite-sheet packing, and atlas generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sprite(w: int = 8, h: int = 8, frames: int = 1, layers: int = 1):
    from spriter.core.sprite import Sprite

    s = Sprite(w, h)
    for i in range(layers):
        s.add_layer(f"Layer {i + 1}")
    for _ in range(frames):
        s.add_frame()
    return s


def _fill_frame(sprite, li: int, fi: int, color=(255, 0, 0, 255)):
    buf = np.zeros((sprite.height, sprite.width, 4), dtype=np.uint8)
    buf[:] = color
    sprite.set_cel_pixels(li, fi, buf)


# ---------------------------------------------------------------------------
# PNG export / import
# ---------------------------------------------------------------------------


class TestPngIO:
    def test_export_frame_creates_file(self, tmp_path):
        from spriter.io.png_io import export_frame

        s = _make_sprite()
        _fill_frame(s, 0, 0, (255, 0, 0, 255))
        out = tmp_path / "frame.png"
        export_frame(s, 0, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_frame_pixel_roundtrip(self, tmp_path):
        """Export then load back and verify pixels are preserved."""
        from PIL import Image
        from spriter.io.png_io import export_frame

        s = _make_sprite(4, 4)
        color = (10, 20, 30, 255)
        _fill_frame(s, 0, 0, color)
        out = tmp_path / "frame.png"
        export_frame(s, 0, out)
        loaded = np.array(Image.open(str(out)).convert("RGBA"), dtype=np.uint8)
        assert loaded.shape == (4, 4, 4)
        assert np.all(loaded == np.array(color, dtype=np.uint8))

    def test_export_all_frames_count(self, tmp_path):
        from spriter.io.png_io import export_all_frames

        s = _make_sprite(frames=3)
        for fi in range(3):
            _fill_frame(s, 0, fi)
        paths = export_all_frames(s, tmp_path / "frames")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    def test_export_all_frames_naming(self, tmp_path):
        from spriter.io.png_io import export_all_frames

        s = _make_sprite(frames=2)
        paths = export_all_frames(s, tmp_path / "out", prefix="spr")
        names = [p.name for p in paths]
        assert names[0] == "spr_0000.png"
        assert names[1] == "spr_0001.png"

    def test_import_png_creates_correct_sprite(self, tmp_path):
        from PIL import Image
        from spriter.io.png_io import import_png

        # Create a 6×4 PNG and import it.
        img = Image.new("RGBA", (6, 4), (128, 64, 32, 255))
        png_path = tmp_path / "test.png"
        img.save(str(png_path))

        sprite = import_png(png_path)
        assert sprite.width == 6
        assert sprite.height == 4
        assert sprite.layer_count == 1
        assert sprite.frame_count == 1

    def test_import_png_pixel_values(self, tmp_path):
        from PIL import Image
        from spriter.io.png_io import import_png

        color = (99, 111, 123, 255)
        img = Image.new("RGBA", (4, 4), color)
        png_path = tmp_path / "c.png"
        img.save(str(png_path))

        sprite = import_png(png_path)
        pixels = sprite.get_cel(0, 0).pixels
        assert pixels is not None
        assert np.all(pixels == np.array(color, dtype=np.uint8))

    def test_import_png_non_rgba_converts(self, tmp_path):
        """Non-RGBA images (e.g. RGB) are converted to RGBA on import."""
        from PIL import Image
        from spriter.io.png_io import import_png

        img = Image.new("RGB", (8, 8), (255, 0, 0))
        path = tmp_path / "rgb.png"
        img.save(str(path))
        sprite = import_png(path)
        pixels = sprite.get_cel(0, 0).pixels
        assert pixels is not None
        assert pixels.shape == (8, 8, 4)


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------


class TestGifIO:
    def test_export_gif_creates_file(self, tmp_path):
        from spriter.io.gif_io import export_gif

        s = _make_sprite(frames=2)
        _fill_frame(s, 0, 0, (255, 0, 0, 255))
        _fill_frame(s, 0, 1, (0, 255, 0, 255))
        out = tmp_path / "anim.gif"
        export_gif(s, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_gif_correct_frame_count(self, tmp_path):
        from PIL import Image
        from spriter.io.gif_io import export_gif

        n = 3
        s = _make_sprite(frames=n)
        # Use distinct colors per frame so Pillow doesn't collapse them.
        colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]
        for fi, color in enumerate(colors):
            _fill_frame(s, 0, fi, color)
        out = tmp_path / "anim.gif"
        export_gif(s, out)
        with Image.open(str(out)) as gif:
            assert gif.n_frames == n

    def test_export_gif_single_frame(self, tmp_path):
        from spriter.io.gif_io import export_gif

        s = _make_sprite(frames=1)
        _fill_frame(s, 0, 0)
        out = tmp_path / "single.gif"
        export_gif(s, out)
        assert out.exists()

    def test_export_gif_no_frames_raises(self, tmp_path):
        from spriter.io.gif_io import export_gif
        from spriter.core.sprite import Sprite

        empty = Sprite(8, 8)
        with pytest.raises(ValueError, match="no frames"):
            export_gif(empty, tmp_path / "bad.gif")


# ---------------------------------------------------------------------------
# Sprite sheet export
# ---------------------------------------------------------------------------


class TestSpriteSheetExport:
    def _make_color_sprite(self, n_frames: int = 3):
        s = _make_sprite(8, 8, frames=n_frames)
        colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]
        for fi in range(n_frames):
            _fill_frame(s, 0, fi, colors[fi % len(colors)])
        return s

    def test_horizontal_sheet_width(self, tmp_path):
        from spriter.io.spritesheet import SheetLayout, export_sheet
        from PIL import Image

        n = 3
        s = self._make_color_sprite(n)
        out = tmp_path / "sheet.png"
        export_sheet(s, out, layout=SheetLayout.HORIZONTAL)
        img = Image.open(str(out))
        assert img.width == 8 * n
        assert img.height == 8

    def test_vertical_sheet_height(self, tmp_path):
        from spriter.io.spritesheet import SheetLayout, export_sheet
        from PIL import Image

        n = 4
        s = self._make_color_sprite(n)
        out = tmp_path / "sheet_v.png"
        export_sheet(s, out, layout=SheetLayout.VERTICAL)
        img = Image.open(str(out))
        assert img.width == 8
        assert img.height == 8 * n

    def test_grid_sheet_dimensions(self, tmp_path):
        from spriter.io.spritesheet import SheetLayout, export_sheet
        from PIL import Image

        n = 6
        s = self._make_color_sprite(n)
        out = tmp_path / "sheet_g.png"
        export_sheet(s, out, layout=SheetLayout.GRID, cols=3)
        img = Image.open(str(out))
        assert img.width == 8 * 3  # 3 cols
        assert img.height == 8 * 2  # 2 rows

    def test_horizontal_sheet_with_padding(self, tmp_path):
        from spriter.io.spritesheet import SheetLayout, export_sheet
        from PIL import Image

        pad = 2
        n = 2
        s = self._make_color_sprite(n)
        out = tmp_path / "padded.png"
        export_sheet(s, out, layout=SheetLayout.HORIZONTAL, padding=pad)
        img = Image.open(str(out))
        # width = n * 8 + (n+1) * pad
        expected_w = n * 8 + (n + 1) * pad
        assert img.width == expected_w

    def test_pixels_at_correct_positions(self, tmp_path):
        """Verify first frame is at (0,0) in horizontal sheet (no padding)."""
        from spriter.io.spritesheet import SheetLayout, export_sheet
        from PIL import Image
        import numpy as np

        s = self._make_color_sprite(2)
        out = tmp_path / "pos.png"
        export_sheet(s, out, layout=SheetLayout.HORIZONTAL)
        arr = np.array(Image.open(str(out)).convert("RGBA"), dtype=np.uint8)
        # Top-left pixel should be red (frame 0).
        assert arr[0, 0, 0] == 255  # R
        assert arr[0, 0, 1] == 0  # G
        # Pixel at x=8 (start of frame 1) should be green.
        assert arr[0, 8, 0] == 0  # R
        assert arr[0, 8, 1] == 255  # G

    def test_export_no_frames_raises(self, tmp_path):
        from spriter.io.spritesheet import export_sheet
        from spriter.core.sprite import Sprite

        empty = Sprite(8, 8)
        with pytest.raises(ValueError, match="no frames"):
            export_sheet(empty, tmp_path / "bad.png")


# ---------------------------------------------------------------------------
# Sprite sheet import
# ---------------------------------------------------------------------------


class TestSpriteSheetImport:
    def test_import_sheet_frame_count(self, tmp_path):
        from spriter.io.spritesheet import SheetLayout, export_sheet, import_sheet

        n = 4
        s = _make_sprite(8, 8, frames=n)
        for fi in range(n):
            _fill_frame(s, 0, fi)
        sheet_path = tmp_path / "sheet.png"
        export_sheet(s, sheet_path, layout=SheetLayout.HORIZONTAL)
        imported = import_sheet(sheet_path, 8, 8)
        assert imported.frame_count == n

    def test_import_sheet_size(self, tmp_path):
        from spriter.io.spritesheet import SheetLayout, export_sheet, import_sheet

        n = 2
        s = _make_sprite(16, 16, frames=n)
        for fi in range(n):
            _fill_frame(s, 0, fi)
        sheet_path = tmp_path / "sheet16.png"
        export_sheet(s, sheet_path, layout=SheetLayout.HORIZONTAL)
        imported = import_sheet(sheet_path, 16, 16)
        assert imported.width == 16
        assert imported.height == 16

    def test_import_sheet_pixel_fidelity(self, tmp_path):
        """Round-trip: packed colors should survive export→import."""
        from spriter.io.spritesheet import SheetLayout, export_sheet, import_sheet

        s = _make_sprite(4, 4, frames=2)
        _fill_frame(s, 0, 0, (200, 100, 50, 255))
        _fill_frame(s, 0, 1, (50, 100, 200, 255))
        sheet_path = tmp_path / "sheet.png"
        export_sheet(s, sheet_path, layout=SheetLayout.HORIZONTAL)
        imported = import_sheet(sheet_path, 4, 4)
        p0 = imported.get_cel(0, 0).pixels
        p1 = imported.get_cel(0, 1).pixels
        assert p0 is not None and p1 is not None
        assert np.all(p0 == np.array([200, 100, 50, 255], dtype=np.uint8))
        assert np.all(p1 == np.array([50, 100, 200, 255], dtype=np.uint8))

    def test_import_sheet_with_padding(self, tmp_path):
        from spriter.io.spritesheet import SheetLayout, export_sheet, import_sheet

        n = 3
        pad = 2
        s = _make_sprite(8, 8, frames=n)
        for fi in range(n):
            _fill_frame(s, 0, fi)
        sheet_path = tmp_path / "padded.png"
        export_sheet(s, sheet_path, layout=SheetLayout.HORIZONTAL, padding=pad)
        imported = import_sheet(sheet_path, 8, 8, padding=pad)
        assert imported.frame_count == n

    def test_import_too_small_raises(self, tmp_path):
        from spriter.io.spritesheet import import_sheet
        from PIL import Image

        img = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
        path = tmp_path / "tiny.png"
        img.save(str(path))
        with pytest.raises(ValueError, match="too small"):
            import_sheet(path, 16, 16)


# ---------------------------------------------------------------------------
# Atlas export
# ---------------------------------------------------------------------------


class TestAtlasExport:
    def test_atlas_json_structure(self, tmp_path):
        from spriter.io.spritesheet import SheetLayout, export_atlas

        n = 2
        s = _make_sprite(8, 8, frames=n)
        for fi in range(n):
            _fill_frame(s, 0, fi)
        sheet_path = tmp_path / "sheet.png"
        atlas_path = tmp_path / "atlas.json"
        atlas = export_atlas(s, sheet_path, atlas_path)

        assert "meta" in atlas
        assert "frames" in atlas
        assert len(atlas["frames"]) == n
        assert "frame_0000" in atlas["frames"]
        assert "frame_0001" in atlas["frames"]

    def test_atlas_frame_positions(self, tmp_path):
        from spriter.io.spritesheet import SheetLayout, export_atlas

        s = _make_sprite(4, 4, frames=2)
        for fi in range(2):
            _fill_frame(s, 0, fi)
        sheet_path = tmp_path / "s.png"
        atlas_path = tmp_path / "a.json"
        atlas = export_atlas(s, sheet_path, atlas_path, layout=SheetLayout.HORIZONTAL)

        f0 = atlas["frames"]["frame_0000"]["frame"]
        f1 = atlas["frames"]["frame_0001"]["frame"]
        assert f0 == {"x": 0, "y": 0, "w": 4, "h": 4}
        assert f1 == {"x": 4, "y": 0, "w": 4, "h": 4}

    def test_atlas_duration_preserved(self, tmp_path):
        from spriter.io.spritesheet import export_atlas

        s = _make_sprite(4, 4, frames=1)
        s.frames[0].duration_ms = 200
        _fill_frame(s, 0, 0)
        atlas = export_atlas(s, tmp_path / "s.png", tmp_path / "a.json")
        assert atlas["frames"]["frame_0000"]["duration"] == 200

    def test_atlas_json_file_written(self, tmp_path):
        from spriter.io.spritesheet import export_atlas

        s = _make_sprite(4, 4, frames=1)
        _fill_frame(s, 0, 0)
        atlas_path = tmp_path / "a.json"
        export_atlas(s, tmp_path / "s.png", atlas_path)
        assert atlas_path.exists()
        data = json.loads(atlas_path.read_text())
        assert "frames" in data

    def test_atlas_with_padding(self, tmp_path):
        from spriter.io.spritesheet import SheetLayout, export_atlas

        pad = 4
        s = _make_sprite(8, 8, frames=2)
        for fi in range(2):
            _fill_frame(s, 0, fi)
        atlas = export_atlas(
            s,
            tmp_path / "s.png",
            tmp_path / "a.json",
            layout=SheetLayout.HORIZONTAL,
            padding=pad,
        )
        f0 = atlas["frames"]["frame_0000"]["frame"]
        f1 = atlas["frames"]["frame_0001"]["frame"]
        # With padding: frame 0 starts at (pad, pad), frame 1 at (pad + 8 + pad, pad)
        assert f0["x"] == pad
        assert f1["x"] == pad + 8 + pad
