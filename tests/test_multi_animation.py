# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for multiple animation timelines per project."""

import json

import numpy as np
import pytest

from spriter.commands.animation_ops import (
    AddTimelineCommand,
    RemoveTimelineCommand,
    RenameTimelineCommand,
)
from spriter.commands.base import CommandStack
from spriter.commands.layer_ops import RemoveLayerCommand
from spriter.core.animation import LoopMode
from spriter.core.sprite import Sprite
from spriter.io.project_io import load, save
from spriter.io.spritesheet import export_sheet, import_sheet, import_sheet_auto


def _sprite_two_anims() -> Sprite:
    """8x8 sprite, 1 layer, two timelines with 2 and 3 frames respectively."""
    s = Sprite(8, 8)
    s.add_layer("Layer 1")
    s.add_frame()
    s.add_frame()  # timeline 0 now has 2 frames
    s.add_timeline("run")  # timeline 1: starts with 1 frame
    s.set_active_timeline(1)
    s.add_frame()
    s.add_frame()  # timeline 1 now has 3 frames
    s.set_active_timeline(0)
    return s


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TestTimelineModel:
    def test_default_single_timeline(self):
        s = Sprite(8, 8)
        assert s.timeline_count == 1
        assert s.active_timeline_index == 0
        assert s.active_timeline.name == "Animation 1"

    def test_add_timeline_appends_and_names_uniquely(self):
        s = Sprite(8, 8)
        s.add_layer("Layer 1")
        tl = s.add_timeline()
        assert s.timeline_count == 2
        assert tl.name == "Animation 2"
        # New timeline starts with one frame.
        s.set_active_timeline(1)
        assert s.frame_count == 1

    def test_frames_are_isolated_between_timelines(self):
        s = _sprite_two_anims()
        assert s.timelines[0].frame_count == 2
        assert s.timelines[1].frame_count == 3

    def test_cel_content_isolated_between_timelines(self):
        s = _sprite_two_anims()
        red = np.zeros((8, 8, 4), dtype=np.uint8)
        red[..., 0] = 255
        red[..., 3] = 255
        s.set_active_timeline(0)
        s.set_cel_pixels(0, 0, red)
        s.set_active_timeline(1)
        # Timeline 1, frame 0 is untouched (transparent).
        assert not np.any(s.get_cel(0, 0).pixels)
        s.set_active_timeline(0)
        assert np.array_equal(s.get_cel(0, 0).pixels, red)

    def test_set_active_timeline_out_of_range(self):
        s = Sprite(8, 8)
        with pytest.raises(IndexError):
            s.set_active_timeline(5)

    def test_remove_timeline_requires_at_least_one(self):
        s = Sprite(8, 8)
        with pytest.raises(ValueError):
            s.remove_timeline(0)

    def test_remove_timeline_clamps_active(self):
        s = _sprite_two_anims()
        s.set_active_timeline(1)
        s.remove_timeline(1)
        assert s.timeline_count == 1
        assert s.active_timeline_index == 0


# ---------------------------------------------------------------------------
# Layer fan-out across timelines
# ---------------------------------------------------------------------------


class TestLayerFanOut:
    def test_add_layer_creates_cels_in_all_timelines(self):
        s = _sprite_two_anims()
        s.add_layer("Layer 2")  # layer index 1
        for ti, tl in enumerate(s.timelines):
            for fi in range(tl.frame_count):
                assert (1, fi) in tl._cels

    def test_remove_layer_reindexes_all_timelines(self):
        s = _sprite_two_anims()
        s.add_layer("Layer 2")
        # Paint layer 1 in both timelines so we can detect misalignment.
        green = np.zeros((8, 8, 4), dtype=np.uint8)
        green[..., 1] = 255
        green[..., 3] = 255
        for ti in range(s.timeline_count):
            s.set_active_timeline(ti)
            s.set_cel_pixels(1, 0, green)
        s.set_active_timeline(0)
        s.remove_layer(0)  # layer 1 becomes layer 0 everywhere
        for ti in range(s.timeline_count):
            s.set_active_timeline(ti)
            assert np.array_equal(s.get_cel(0, 0).pixels, green)

    def test_remove_layer_command_undo_restores_all_timelines(self):
        s = _sprite_two_anims()
        s.add_layer("Layer 2")
        blue = np.zeros((8, 8, 4), dtype=np.uint8)
        blue[..., 2] = 255
        blue[..., 3] = 255
        for ti in range(s.timeline_count):
            s.set_active_timeline(ti)
            s.set_cel_pixels(1, 0, blue)
        s.set_active_timeline(0)
        stack = CommandStack()
        stack.push(RemoveLayerCommand(s, 1))
        stack.undo()
        assert s.layer_count == 2
        for ti in range(s.timeline_count):
            s.set_active_timeline(ti)
            assert np.array_equal(s.get_cel(1, 0).pixels, blue)


# ---------------------------------------------------------------------------
# Canvas transforms stay consistent
# ---------------------------------------------------------------------------


class TestCanvasConsistency:
    def test_resize_canvas_affects_all_timelines(self):
        s = _sprite_two_anims()
        s.resize_canvas(16, 16)
        for tl in s.timelines:
            for cel in tl._cels.values():
                if cel.pixels is not None:
                    assert cel.pixels.shape[:2] == (16, 16)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip_multiple_timelines(self, tmp_path):
        s = _sprite_two_anims()
        s.rename_timeline(0, "idle")
        s.timelines[1].animation.loop_mode = LoopMode.PING_PONG
        s.timelines[1].animation.default_fps = 24
        red = np.zeros((8, 8, 4), dtype=np.uint8)
        red[..., 0] = 255
        red[..., 3] = 255
        s.set_active_timeline(1)
        s.set_cel_pixels(0, 2, red)
        s.set_active_timeline(0)

        dest = tmp_path / "multi.spriter"
        save(s, dest)
        loaded = load(dest)

        assert loaded.timeline_count == 2
        assert loaded.timelines[0].name == "idle"
        assert loaded.timelines[1].name == "run"
        assert loaded.timelines[0].frame_count == 2
        assert loaded.timelines[1].frame_count == 3
        assert loaded.timelines[1].animation.loop_mode == LoopMode.PING_PONG
        assert loaded.timelines[1].animation.default_fps == 24
        loaded.set_active_timeline(1)
        assert np.array_equal(loaded.get_cel(0, 2).pixels, red)

    def test_load_version_1_wraps_single_timeline(self, tmp_path):
        # Hand-craft a version-1 document (top-level frames + cels).
        v1 = {
            "version": 1,
            "width": 4,
            "height": 4,
            "color_mode": "RGBA",
            "layers": [
                {
                    "name": "Layer 1",
                    "visible": True,
                    "locked": False,
                    "opacity": 255,
                    "blend_mode": "normal",
                }
            ],
            "frames": [{"duration_ms": 100}, {"duration_ms": 150}],
            "cels": {},
        }
        dest = tmp_path / "legacy.spriter"
        dest.write_text(json.dumps(v1), encoding="utf-8")
        loaded = load(dest)
        assert loaded.timeline_count == 1
        assert loaded.timelines[0].name == "Animation 1"
        assert loaded.frame_count == 2
        assert loaded.frames[1].duration_ms == 150


# ---------------------------------------------------------------------------
# animation_ops commands
# ---------------------------------------------------------------------------


class TestAnimationCommands:
    def test_add_timeline_command_undo_redo(self):
        s = Sprite(8, 8)
        s.add_layer("Layer 1")
        s.add_frame()
        stack = CommandStack()
        stack.push(AddTimelineCommand(s, "run"))
        assert s.timeline_count == 2
        assert s.active_timeline_index == 1
        stack.undo()
        assert s.timeline_count == 1
        assert s.active_timeline_index == 0
        stack.redo()
        assert s.timeline_count == 2

    def test_remove_timeline_command_undo_restores(self):
        s = _sprite_two_anims()
        stack = CommandStack()
        stack.push(RemoveTimelineCommand(s, 1))
        assert s.timeline_count == 1
        stack.undo()
        assert s.timeline_count == 2
        assert s.timelines[1].name == "run"
        assert s.timelines[1].frame_count == 3

    def test_rename_timeline_command_undo(self):
        s = _sprite_two_anims()
        stack = CommandStack()
        stack.push(RenameTimelineCommand(s, 0, "idle"))
        assert s.timelines[0].name == "idle"
        stack.undo()
        assert s.timelines[0].name == "Animation 1"


# ---------------------------------------------------------------------------
# Sheet export / import
# ---------------------------------------------------------------------------


class TestSheetRows:
    def test_export_multi_timeline_rows(self, tmp_path):
        from PIL import Image

        s = _sprite_two_anims()
        red = np.zeros((8, 8, 4), dtype=np.uint8)
        red[..., 0] = 255
        red[..., 3] = 255
        green = np.zeros((8, 8, 4), dtype=np.uint8)
        green[..., 1] = 255
        green[..., 3] = 255
        # Fill timeline 0 (2 frames) red, timeline 1 (3 frames) green.
        s.set_active_timeline(0)
        for fi in range(2):
            s.set_cel_pixels(0, fi, red)
        s.set_active_timeline(1)
        for fi in range(3):
            s.set_cel_pixels(0, fi, green)
        s.set_active_timeline(0)

        dest = tmp_path / "sheet.png"
        export_sheet(s, dest)
        arr = np.array(Image.open(dest).convert("RGBA"))
        # Grid: 2 rows x 3 cols of 8x8 cells → 24x16.
        assert arr.shape == (16, 24, 4)
        # Row 0: red frames in first two cells, transparent trailing cell.
        assert np.array_equal(arr[0:8, 0:8], red)
        assert np.array_equal(arr[0:8, 8:16], red)
        assert not np.any(arr[0:8, 16:24, 3])  # trailing cell transparent
        # Row 1: three green frames.
        assert np.array_equal(arr[8:16, 0:8], green)
        assert np.array_equal(arr[8:16, 16:24], green)

    def test_import_grid_split_rows(self, tmp_path):
        from PIL import Image

        # Build a 3-wide x 2-tall grid sheet (cells 8x8): row0 red, row1 blue.
        sheet = np.zeros((16, 24, 4), dtype=np.uint8)
        sheet[0:8, :, 0] = 255
        sheet[0:8, :, 3] = 255
        sheet[8:16, :, 2] = 255
        sheet[8:16, :, 3] = 255
        path = tmp_path / "grid.png"
        Image.fromarray(sheet, "RGBA").save(path)

        s = import_sheet(path, 8, 8, split_rows=True)
        assert s.timeline_count == 2
        assert s.timelines[0].frame_count == 3
        assert s.timelines[1].frame_count == 3
        s.set_active_timeline(1)
        cel = s.get_cel(0, 0).pixels
        assert cel[0, 0, 2] == 255  # blue

    def test_import_grid_no_split_single_timeline(self, tmp_path):
        from PIL import Image

        sheet = np.zeros((16, 24, 4), dtype=np.uint8)
        sheet[..., 3] = 255
        path = tmp_path / "grid.png"
        Image.fromarray(sheet, "RGBA").save(path)
        s = import_sheet(path, 8, 8)
        assert s.timeline_count == 1
        assert s.frame_count == 6

    def test_import_auto_split_rows(self, tmp_path):
        from PIL import Image

        # Two rows of two separated blobs each.
        sheet = np.zeros((40, 40, 4), dtype=np.uint8)
        for cy, cx in [(8, 8), (8, 28), (28, 8), (28, 28)]:
            sheet[cy - 3 : cy + 3, cx - 3 : cx + 3, 0] = 255
            sheet[cy - 3 : cy + 3, cx - 3 : cx + 3, 3] = 255
        path = tmp_path / "auto.png"
        Image.fromarray(sheet, "RGBA").save(path)

        s = import_sheet_auto(path, split_rows=True)
        assert s.timeline_count == 2
        assert s.timelines[0].frame_count == 2
        assert s.timelines[1].frame_count == 2
