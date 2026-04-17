# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for layer-level undo/redo commands (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from spriter.commands.base import CommandStack
from spriter.commands.layer_ops import (
    AddLayerCommand,
    DuplicateLayerCommand,
    FlattenCommand,
    MergeLayerDownCommand,
    MoveLayerCommand,
    RemoveLayerCommand,
)
from spriter.core.layer import BlendMode
from spriter.core.sprite import Sprite


def _make_sprite(layers=1, frames=1, width=4, height=4) -> Sprite:
    """Create a sprite with the given number of blank layers and frames."""
    s = Sprite(width, height)
    for i in range(layers):
        s.add_layer(f"Layer {i + 1}")
    for _ in range(frames):
        s.add_frame()
    return s


def _fill_layer(sprite, layer_idx, frame_idx, rgba):
    """Fill the cel at (layer_idx, frame_idx) with a solid colour."""
    px = np.full((sprite.height, sprite.width, 4), rgba, dtype=np.uint8)
    sprite.set_cel_pixels(layer_idx, frame_idx, px)


class TestAddLayerCommand:
    def test_execute_adds_layer(self):
        s = _make_sprite(1, 1)
        stack = CommandStack()
        cmd = AddLayerCommand(s, "NewLayer")
        stack.push(cmd)
        assert s.layer_count == 2

    def test_undo_removes_layer(self):
        s = _make_sprite(1, 1)
        stack = CommandStack()
        cmd = AddLayerCommand(s, "NewLayer")
        stack.push(cmd)
        stack.undo()
        assert s.layer_count == 1

    def test_redo_re_adds_layer(self):
        s = _make_sprite(1, 1)
        stack = CommandStack()
        cmd = AddLayerCommand(s, "NewLayer")
        stack.push(cmd)
        stack.undo()
        stack.redo()
        assert s.layer_count == 2

    def test_description(self):
        s = _make_sprite(1, 1)
        cmd = AddLayerCommand(s, "BG")
        assert "BG" in cmd.description

    def test_insert_at_index(self):
        s = _make_sprite(2, 1)
        stack = CommandStack()
        cmd = AddLayerCommand(s, "Middle", index=1)
        stack.push(cmd)
        assert s.layers[1].name == "Middle"


class TestRemoveLayerCommand:
    def test_execute_removes_layer(self):
        s = _make_sprite(2, 1)
        stack = CommandStack()
        cmd = RemoveLayerCommand(s, 1)
        stack.push(cmd)
        assert s.layer_count == 1

    def test_undo_restores_layer(self):
        s = _make_sprite(2, 1)
        orig_name = s.layers[1].name
        stack = CommandStack()
        cmd = RemoveLayerCommand(s, 1)
        stack.push(cmd)
        stack.undo()
        assert s.layer_count == 2
        assert s.layers[1].name == orig_name

    def test_undo_restores_cel_pixels(self):
        """Removed layer's pixel data must survive undo."""
        s = _make_sprite(2, 2)
        _fill_layer(s, 1, 0, (100, 200, 50, 255))
        _fill_layer(s, 1, 1, (10, 20, 30, 255))
        stack = CommandStack()
        cmd = RemoveLayerCommand(s, 1)
        stack.push(cmd)
        stack.undo()
        cel00 = s.get_cel(1, 0)
        assert cel00.pixels is not None
        assert tuple(cel00.pixels[0, 0]) == (100, 200, 50, 255)

    def test_description_includes_name(self):
        s = _make_sprite(2, 1)
        s._layers[1].name = "FX"  # type: ignore[attr-defined]
        cmd = RemoveLayerCommand(s, 1)
        assert "FX" in cmd.description


class TestDuplicateLayerCommand:
    def test_execute_increases_layer_count(self):
        s = _make_sprite(1, 1)
        _fill_layer(s, 0, 0, (200, 100, 50, 255))
        stack = CommandStack()
        cmd = DuplicateLayerCommand(s, 0)
        stack.push(cmd)
        assert s.layer_count == 2

    def test_duplicate_has_same_pixels(self):
        s = _make_sprite(1, 1)
        _fill_layer(s, 0, 0, (200, 100, 50, 255))
        stack = CommandStack()
        cmd = DuplicateLayerCommand(s, 0)
        stack.push(cmd)
        orig_px = s.get_cel(0, 0).pixels
        dup_px = s.get_cel(1, 0).pixels
        assert orig_px is not None and dup_px is not None
        np.testing.assert_array_equal(orig_px, dup_px)

    def test_duplicate_is_independent_copy(self):
        """Editing the original after duplicate should not affect the copy."""
        s = _make_sprite(1, 1)
        _fill_layer(s, 0, 0, (200, 100, 50, 255))
        stack = CommandStack()
        cmd = DuplicateLayerCommand(s, 0)
        stack.push(cmd)
        _fill_layer(s, 0, 0, (0, 0, 0, 255))
        dup_px = s.get_cel(1, 0).pixels
        assert dup_px is not None
        assert tuple(dup_px[0, 0]) == (200, 100, 50, 255)

    def test_undo_removes_duplicate(self):
        s = _make_sprite(1, 1)
        stack = CommandStack()
        cmd = DuplicateLayerCommand(s, 0)
        stack.push(cmd)
        stack.undo()
        assert s.layer_count == 1


class TestMoveLayerCommand:
    def test_execute_moves_layer(self):
        s = _make_sprite(3, 1)
        original_top = s.layers[2].name
        stack = CommandStack()
        cmd = MoveLayerCommand(s, 2, 0)
        stack.push(cmd)
        assert s.layers[0].name == original_top

    def test_undo_reverts_move(self):
        s = _make_sprite(3, 1)
        names_before = [l.name for l in s.layers]
        stack = CommandStack()
        cmd = MoveLayerCommand(s, 2, 0)
        stack.push(cmd)
        stack.undo()
        assert [l.name for l in s.layers] == names_before


class TestMergeLayerDownCommand:
    def test_execute_reduces_layer_count(self):
        s = _make_sprite(2, 1)
        stack = CommandStack()
        cmd = MergeLayerDownCommand(s, 1)
        stack.push(cmd)
        assert s.layer_count == 1

    def test_raises_when_bottom_layer(self):
        s = _make_sprite(2, 1)
        with pytest.raises(ValueError):
            MergeLayerDownCommand(s, 0)

    def test_merged_pixels_are_composite(self):
        """Merging fully-opaque top over transparent bottom → top colour in bottom."""
        s = _make_sprite(2, 1, width=2, height=2)
        _fill_layer(s, 1, 0, (200, 100, 50, 255))  # top layer: opaque orange
        # bottom layer is blank (transparent)
        stack = CommandStack()
        cmd = MergeLayerDownCommand(s, 1)
        stack.push(cmd)
        px = s.get_cel(0, 0).pixels
        assert px is not None
        np.testing.assert_array_equal(px[0, 0], [200, 100, 50, 255])

    def test_undo_restores_both_layers(self):
        s = _make_sprite(2, 1, width=2, height=2)
        _fill_layer(s, 0, 0, (50, 50, 50, 255))
        _fill_layer(s, 1, 0, (200, 100, 50, 255))
        original_bottom = s.get_cel(0, 0).pixels.copy()  # type: ignore[union-attr]
        original_top = s.get_cel(1, 0).pixels.copy()  # type: ignore[union-attr]

        stack = CommandStack()
        stack.push(MergeLayerDownCommand(s, 1))
        stack.undo()

        assert s.layer_count == 2
        np.testing.assert_array_equal(s.get_cel(0, 0).pixels, original_bottom)
        np.testing.assert_array_equal(s.get_cel(1, 0).pixels, original_top)

    def test_multiple_frames_merged(self):
        """Merge should composite all frames, not just frame 0."""
        s = _make_sprite(2, 2, width=1, height=1)
        _fill_layer(s, 1, 0, (255, 0, 0, 255))
        _fill_layer(s, 1, 1, (0, 255, 0, 255))
        stack = CommandStack()
        stack.push(MergeLayerDownCommand(s, 1))
        px0 = s.get_cel(0, 0).pixels
        px1 = s.get_cel(0, 1).pixels
        assert px0 is not None and px1 is not None
        assert tuple(px0[0, 0]) == (255, 0, 0, 255)
        assert tuple(px1[0, 0]) == (0, 255, 0, 255)


class TestFlattenCommand:
    def test_execute_leaves_single_layer(self):
        s = _make_sprite(3, 1)
        stack = CommandStack()
        stack.push(FlattenCommand(s))
        assert s.layer_count == 1

    def test_flattened_layer_named_merged(self):
        s = _make_sprite(2, 1)
        stack = CommandStack()
        stack.push(FlattenCommand(s))
        assert s.layers[0].name == "Merged"

    def test_undo_restores_all_layers(self):
        s = _make_sprite(3, 2)
        layer_names = [l.name for l in s.layers]
        stack = CommandStack()
        stack.push(FlattenCommand(s))
        stack.undo()
        assert s.layer_count == 3
        assert [l.name for l in s.layers] == layer_names

    def test_undo_restores_cel_data(self):
        s = _make_sprite(2, 1, width=2, height=2)
        _fill_layer(s, 0, 0, (10, 20, 30, 255))
        _fill_layer(s, 1, 0, (200, 100, 50, 255))
        orig_0 = s.get_cel(0, 0).pixels.copy()  # type: ignore[union-attr]
        orig_1 = s.get_cel(1, 0).pixels.copy()  # type: ignore[union-attr]

        stack = CommandStack()
        stack.push(FlattenCommand(s))
        stack.undo()

        np.testing.assert_array_equal(s.get_cel(0, 0).pixels, orig_0)
        np.testing.assert_array_equal(s.get_cel(1, 0).pixels, orig_1)

    def test_flatten_composites_visible_layers(self):
        """Flatten result should match direct composite_frame output."""
        from spriter.core.compositor import composite_frame as cf

        s = _make_sprite(2, 1, width=4, height=4)
        _fill_layer(s, 0, 0, (0, 0, 255, 255))
        _fill_layer(s, 1, 0, (255, 0, 0, 128))
        expected = cf(s, 0)

        stack = CommandStack()
        stack.push(FlattenCommand(s))
        actual = s.get_cel(0, 0).pixels
        assert actual is not None
        np.testing.assert_array_equal(actual, expected)
