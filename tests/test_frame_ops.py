# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for frame-level undo/redo commands (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from spriter.commands.base import CommandStack
from spriter.commands.frame_ops import (
    AddFrameCommand,
    DuplicateFrameCommand,
    MoveFrameCommand,
    RemoveFrameCommand,
)
from spriter.core.sprite import Sprite


def _make_sprite(layers=1, frames=1) -> Sprite:
    s = Sprite(4, 4)
    for i in range(layers):
        s.add_layer(f"L{i + 1}")
    for _ in range(frames):
        s.add_frame()
    return s


def _fill(sprite, layer_idx, frame_idx, rgba):
    px = np.full((4, 4, 4), rgba, dtype=np.uint8)
    sprite.set_cel_pixels(layer_idx, frame_idx, px)


class TestAddFrameCommand:
    def test_execute_adds_frame(self):
        s = _make_sprite(1, 1)
        stack = CommandStack()
        stack.push(AddFrameCommand(s))
        assert s.frame_count == 2

    def test_undo_removes_frame(self):
        s = _make_sprite(1, 1)
        stack = CommandStack()
        stack.push(AddFrameCommand(s))
        stack.undo()
        assert s.frame_count == 1

    def test_redo_re_adds_frame(self):
        s = _make_sprite(1, 1)
        stack = CommandStack()
        stack.push(AddFrameCommand(s))
        stack.undo()
        stack.redo()
        assert s.frame_count == 2

    def test_insert_at_index(self):
        s = _make_sprite(1, 2)
        s.frames[0].duration_ms = 100
        s.frames[1].duration_ms = 200
        stack = CommandStack()
        cmd = AddFrameCommand(s, duration_ms=150, index=1)
        stack.push(cmd)
        assert s.frame_count == 3
        assert s.frames[1].duration_ms == 150

    def test_description(self):
        s = _make_sprite(1, 1)
        cmd = AddFrameCommand(s)
        assert "Frame" in cmd.description


class TestRemoveFrameCommand:
    def test_execute_removes_frame(self):
        s = _make_sprite(1, 2)
        stack = CommandStack()
        stack.push(RemoveFrameCommand(s, 1))
        assert s.frame_count == 1

    def test_undo_restores_frame(self):
        s = _make_sprite(1, 2)
        orig_duration = s.frames[1].duration_ms
        stack = CommandStack()
        stack.push(RemoveFrameCommand(s, 1))
        stack.undo()
        assert s.frame_count == 2
        assert s.frames[1].duration_ms == orig_duration

    def test_undo_restores_cel_pixels(self):
        s = _make_sprite(2, 2)
        _fill(s, 0, 1, (100, 150, 200, 255))
        _fill(s, 1, 1, (50, 75, 100, 255))
        stack = CommandStack()
        stack.push(RemoveFrameCommand(s, 1))
        stack.undo()
        px0 = s.get_cel(0, 1).pixels
        px1 = s.get_cel(1, 1).pixels
        assert px0 is not None and px1 is not None
        assert tuple(px0[0, 0]) == (100, 150, 200, 255)
        assert tuple(px1[0, 0]) == (50, 75, 100, 255)


class TestDuplicateFrameCommand:
    def test_execute_increases_frame_count(self):
        s = _make_sprite(1, 1)
        stack = CommandStack()
        stack.push(DuplicateFrameCommand(s, 0))
        assert s.frame_count == 2

    def test_duplicate_has_same_duration(self):
        s = _make_sprite(1, 1)
        s.frames[0].duration_ms = 250
        stack = CommandStack()
        stack.push(DuplicateFrameCommand(s, 0))
        assert s.frames[1].duration_ms == 250

    def test_duplicate_has_same_pixels(self):
        s = _make_sprite(2, 1)
        _fill(s, 0, 0, (10, 20, 30, 255))
        _fill(s, 1, 0, (40, 50, 60, 255))
        stack = CommandStack()
        stack.push(DuplicateFrameCommand(s, 0))
        # Frame 1 should match frame 0 for both layers.
        for li in range(2):
            orig = s.get_cel(li, 0).pixels
            dup = s.get_cel(li, 1).pixels
            assert orig is not None and dup is not None
            np.testing.assert_array_equal(orig, dup)

    def test_duplicate_is_independent(self):
        """Editing original frame after duplicate should not affect the copy."""
        s = _make_sprite(1, 1)
        _fill(s, 0, 0, (200, 100, 50, 255))
        stack = CommandStack()
        stack.push(DuplicateFrameCommand(s, 0))
        _fill(s, 0, 0, (0, 0, 0, 255))
        dup_px = s.get_cel(0, 1).pixels
        assert dup_px is not None
        assert tuple(dup_px[0, 0]) == (200, 100, 50, 255)

    def test_undo_removes_duplicate(self):
        s = _make_sprite(1, 1)
        stack = CommandStack()
        stack.push(DuplicateFrameCommand(s, 0))
        stack.undo()
        assert s.frame_count == 1


class TestMoveFrameCommand:
    def test_execute_moves_frame(self):
        s = _make_sprite(1, 3)
        s.frames[0].duration_ms = 10
        s.frames[1].duration_ms = 20
        s.frames[2].duration_ms = 30
        stack = CommandStack()
        stack.push(MoveFrameCommand(s, 2, 0))
        # Frame originally at index 2 is now at 0.
        assert s.frames[0].duration_ms == 30

    def test_undo_reverts_move(self):
        s = _make_sprite(1, 3)
        s.frames[0].duration_ms = 10
        s.frames[1].duration_ms = 20
        s.frames[2].duration_ms = 30
        stack = CommandStack()
        stack.push(MoveFrameCommand(s, 2, 0))
        stack.undo()
        assert s.frames[0].duration_ms == 10
        assert s.frames[2].duration_ms == 30

    def test_description(self):
        s = _make_sprite(1, 2)
        cmd = MoveFrameCommand(s, 0, 1)
        assert "Frame" in cmd.description
