# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for commands/draw.py — DrawCelCommand and SetSelectionCommand."""

import numpy as np
import pytest
from spriter.commands.base import CommandStack
from spriter.commands.draw import DrawCelCommand, SetSelectionCommand
from spriter.core.sprite import Sprite


def _sprite() -> Sprite:
    s = Sprite(8, 8)
    s.add_layer()
    s.add_frame()
    return s


class TestDrawCelCommand:
    def test_execute_sets_pixels(self):
        s = _sprite()
        before = np.zeros((8, 8, 4), dtype=np.uint8)
        after = np.full((8, 8, 4), 99, dtype=np.uint8)
        cmd = DrawCelCommand(s, 0, 0, before, after)
        cmd.execute()
        assert (s.get_cel(0, 0).pixels == 99).all()

    def test_undo_restores_pixels(self):
        s = _sprite()
        before = np.full((8, 8, 4), 10, dtype=np.uint8)
        after = np.full((8, 8, 4), 99, dtype=np.uint8)
        cmd = DrawCelCommand(s, 0, 0, before, after)
        cmd.execute()
        cmd.undo()
        assert (s.get_cel(0, 0).pixels == 10).all()

    def test_description(self):
        s = _sprite()
        px = np.zeros((8, 8, 4), dtype=np.uint8)
        cmd = DrawCelCommand(s, 0, 0, px, px, description="Pencil stroke")
        assert cmd.description == "Pencil stroke"

    def test_copies_arrays(self):
        """Mutating originals after construction must not affect the command."""
        s = _sprite()
        before = np.zeros((8, 8, 4), dtype=np.uint8)
        after = np.full((8, 8, 4), 50, dtype=np.uint8)
        cmd = DrawCelCommand(s, 0, 0, before, after)
        before[:] = 77
        after[:] = 77
        cmd.execute()
        assert (s.get_cel(0, 0).pixels == 50).all()

    def test_undo_redo_via_stack(self):
        s = _sprite()
        before = np.zeros((8, 8, 4), dtype=np.uint8)
        after = np.full((8, 8, 4), 42, dtype=np.uint8)
        stack = CommandStack()
        cmd = DrawCelCommand(s, 0, 0, before, after)
        stack.push(cmd)
        assert (s.get_cel(0, 0).pixels == 42).all()
        stack.undo()
        assert (s.get_cel(0, 0).pixels == 0).all()
        stack.redo()
        assert (s.get_cel(0, 0).pixels == 42).all()


class TestSetSelectionCommand:
    def test_execute_sets_mask(self):
        s = _sprite()
        mask = np.ones((8, 8), dtype=bool)
        cmd = SetSelectionCommand(s, None, mask)
        cmd.execute()
        assert s.selection_mask is not None
        assert s.selection_mask.all()

    def test_undo_clears_mask(self):
        s = _sprite()
        mask = np.ones((8, 8), dtype=bool)
        cmd = SetSelectionCommand(s, None, mask)
        cmd.execute()
        cmd.undo()
        assert s.selection_mask is None

    def test_undo_restores_previous_mask(self):
        s = _sprite()
        old = np.zeros((8, 8), dtype=bool)
        old[0, 0] = True
        new = np.ones((8, 8), dtype=bool)
        cmd = SetSelectionCommand(s, old, new)
        cmd.execute()
        cmd.undo()
        assert s.selection_mask is not None
        assert s.selection_mask[0, 0]
        assert not s.selection_mask[1, 1]

    def test_execute_to_none_clears(self):
        s = _sprite()
        s.selection_mask = np.ones((8, 8), dtype=bool)
        cmd = SetSelectionCommand(s, s.selection_mask, None)
        cmd.execute()
        assert s.selection_mask is None

    def test_description(self):
        s = _sprite()
        cmd = SetSelectionCommand(s, None, None)
        assert cmd.description == "Set Selection"

    def test_copies_masks(self):
        s = _sprite()
        before = np.zeros((8, 8), dtype=bool)
        after = np.ones((8, 8), dtype=bool)
        cmd = SetSelectionCommand(s, before, after)
        after[:] = False  # mutate original — must not affect cmd
        cmd.execute()
        assert s.selection_mask is not None
        assert s.selection_mask.all()
