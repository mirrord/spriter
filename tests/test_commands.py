# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for commands/base.py — Command pattern & undo/redo stack."""

import pytest
from spriter.commands.base import Command, CompositeCommand, CommandStack


class _Counter:
    """Minimal shared state mutated by test commands."""

    def __init__(self):
        self.value = 0


class _IncrementCommand(Command):
    def __init__(self, counter: _Counter, amount: int = 1):
        self._counter = counter
        self._amount = amount

    @property
    def description(self) -> str:
        return f"Increment by {self._amount}"

    def execute(self) -> None:
        self._counter.value += self._amount

    def undo(self) -> None:
        self._counter.value -= self._amount


class TestCommandStack:
    def test_push_executes_command(self):
        counter = _Counter()
        stack = CommandStack()
        stack.push(_IncrementCommand(counter, 5))
        assert counter.value == 5

    def test_undo(self):
        counter = _Counter()
        stack = CommandStack()
        stack.push(_IncrementCommand(counter, 3))
        stack.undo()
        assert counter.value == 0

    def test_redo(self):
        counter = _Counter()
        stack = CommandStack()
        stack.push(_IncrementCommand(counter, 3))
        stack.undo()
        stack.redo()
        assert counter.value == 3

    def test_push_after_undo_clears_redo(self):
        counter = _Counter()
        stack = CommandStack()
        stack.push(_IncrementCommand(counter, 1))
        stack.undo()
        stack.push(_IncrementCommand(counter, 2))
        assert not stack.can_redo
        assert counter.value == 2

    def test_can_undo_redo_flags(self):
        counter = _Counter()
        stack = CommandStack()
        assert not stack.can_undo
        assert not stack.can_redo
        stack.push(_IncrementCommand(counter))
        assert stack.can_undo
        assert not stack.can_redo
        stack.undo()
        assert not stack.can_undo
        assert stack.can_redo

    def test_undo_empty_returns_none(self):
        stack = CommandStack()
        assert stack.undo() is None

    def test_redo_empty_returns_none(self):
        stack = CommandStack()
        assert stack.redo() is None

    def test_max_depth_respected(self):
        counter = _Counter()
        stack = CommandStack(max_depth=3)
        for _ in range(5):
            stack.push(_IncrementCommand(counter, 1))
        # Only 3 commands retained.
        for _ in range(3):
            stack.undo()
        # Further undo returns None.
        assert stack.undo() is None
        # After 3 undos, value is 5 - 3 = 2.
        assert counter.value == 2

    def test_max_depth_invalid(self):
        with pytest.raises(ValueError):
            CommandStack(max_depth=0)

    def test_clear(self):
        counter = _Counter()
        stack = CommandStack()
        stack.push(_IncrementCommand(counter))
        stack.clear()
        assert not stack.can_undo
        assert not stack.can_redo

    def test_description_properties(self):
        counter = _Counter()
        stack = CommandStack()
        stack.push(_IncrementCommand(counter, 7))
        assert "7" in stack.undo_description
        stack.undo()
        assert "7" in stack.redo_description

    def test_no_execute_flag(self):
        counter = _Counter()
        stack = CommandStack()
        cmd = _IncrementCommand(counter, 10)
        counter.value = 10  # already executed externally
        stack.push(cmd, execute=False)
        assert counter.value == 10  # no double-execute

    def test_repr(self):
        stack = CommandStack(max_depth=50)
        assert "50" in repr(stack)


class TestCompositeCommand:
    def test_executes_all_in_order(self):
        counter = _Counter()
        results = []

        class _Append(Command):
            def __init__(self, n):
                self._n = n

            def execute(self):
                results.append(self._n)

            def undo(self):
                results.pop()

        composite = CompositeCommand([_Append(1), _Append(2), _Append(3)])
        composite.execute()
        assert results == [1, 2, 3]

    def test_undo_in_reverse(self):
        counter = _Counter()
        commands = [_IncrementCommand(counter, i) for i in [1, 2, 3]]
        composite = CompositeCommand(commands)
        composite.execute()
        assert counter.value == 6
        composite.undo()
        assert counter.value == 0

    def test_add_command(self):
        counter = _Counter()
        composite = CompositeCommand([])
        composite.add(_IncrementCommand(counter, 5))
        composite.execute()
        assert counter.value == 5

    def test_description(self):
        composite = CompositeCommand([], description="Paint stroke")
        assert composite.description == "Paint stroke"

    def test_pushed_on_stack(self):
        counter = _Counter()
        stack = CommandStack()
        composite = CompositeCommand(
            [_IncrementCommand(counter, 10), _IncrementCommand(counter, 5)]
        )
        stack.push(composite)
        assert counter.value == 15
        stack.undo()
        assert counter.value == 0
