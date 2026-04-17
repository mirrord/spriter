# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for io/project_io.py — project file serialization round-trips."""

import numpy as np
import pytest
from spriter.core.frame import Cel
from spriter.core.layer import BlendMode
from spriter.core.sprite import Sprite
from spriter.io.project_io import autosave, load, save


def _make_sprite() -> Sprite:
    """Return a simple 8x8 sprite with 2 layers and 2 frames."""
    s = Sprite(8, 8)
    s.add_layer("Background", opacity=200, blend_mode=BlendMode.NORMAL)
    s.add_layer("Foreground", visible=False, locked=True)
    s.add_frame(100)
    s.add_frame(200)
    # Paint some pixel data into layer 0, frame 0.
    px = np.full((8, 8, 4), 42, dtype=np.uint8)
    s.set_cel_pixels(0, 0, px)
    return s


class TestSaveLoad:
    def test_round_trip_metadata(self, tmp_path):
        s = _make_sprite()
        dest = tmp_path / "test.spriter"
        save(s, dest)
        loaded = load(dest)

        assert loaded.width == 8
        assert loaded.height == 8
        assert loaded.color_mode == "RGBA"
        assert loaded.layer_count == 2
        assert loaded.frame_count == 2

    def test_round_trip_layer_properties(self, tmp_path):
        s = _make_sprite()
        dest = tmp_path / "test.spriter"
        save(s, dest)
        loaded = load(dest)

        assert loaded.layers[0].name == "Background"
        assert loaded.layers[0].opacity == 200
        assert loaded.layers[0].blend_mode == BlendMode.NORMAL

        assert loaded.layers[1].name == "Foreground"
        assert loaded.layers[1].visible is False
        assert loaded.layers[1].locked is True

    def test_round_trip_frame_properties(self, tmp_path):
        s = _make_sprite()
        dest = tmp_path / "test.spriter"
        save(s, dest)
        loaded = load(dest)

        assert loaded.frames[0].duration_ms == 100
        assert loaded.frames[1].duration_ms == 200

    def test_round_trip_pixel_data(self, tmp_path):
        s = _make_sprite()
        dest = tmp_path / "test.spriter"
        save(s, dest)
        loaded = load(dest)

        cel = loaded.get_cel(0, 0)
        assert cel.pixels is not None
        assert (cel.pixels == 42).all()

    def test_overwrite_is_atomic(self, tmp_path):
        """Second save should replace the first without leaving temp files."""
        s = _make_sprite()
        dest = tmp_path / "test.spriter"
        save(s, dest)
        save(s, dest)  # second save
        tmp_file = dest.with_suffix(dest.suffix + "~")
        assert not tmp_file.exists()

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "nonexistent.spriter")

    def test_load_unsupported_version(self, tmp_path):
        import json

        bad = tmp_path / "bad.spriter"
        bad.write_text(json.dumps({"version": 999, "width": 8, "height": 8}))
        with pytest.raises(ValueError, match="version"):
            load(bad)

    def test_linked_cel_round_trip(self, tmp_path):
        s = Sprite(4, 4)
        s.add_layer()
        s.add_frame()
        s.add_frame()
        # Mark frame 1's cel as linked to frame 0.
        s._cels[(0, 1)] = Cel(linked_frame=0)
        dest = tmp_path / "linked.spriter"
        save(s, dest)
        loaded = load(dest)
        cel = loaded._cels.get((0, 1))
        assert cel is not None
        assert cel.is_linked
        assert cel.linked_frame == 0


class TestAutosave:
    def test_autosave_creates_tilde_file(self, tmp_path):
        s = _make_sprite()
        primary = tmp_path / "work.spriter"
        autosave_path = autosave(s, primary)
        assert autosave_path.exists()
        assert str(autosave_path).endswith("~")

    def test_autosave_is_loadable(self, tmp_path):
        s = _make_sprite()
        primary = tmp_path / "work.spriter"
        autosave_path = autosave(s, primary)
        loaded = load(autosave_path)
        assert loaded.width == s.width
        assert loaded.layer_count == s.layer_count
