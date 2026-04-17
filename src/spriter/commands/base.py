# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Command pattern base classes for undo/redo.

Usage::

    class PaintPixelCommand(Command):
        def __init__(self, pixels, x, y, new_color):
            self._pixels = pixels
            self._x, self._y = x, y
            self._new_color = new_color
            self._old_color = tuple(pixels[y, x])

        def execute(self):
            self._pixels[self._y, self._x] = self._new_color

        def undo(self):
            self._pixels[self._y, self._x] = self._old_color

    stack = CommandStack(max_depth=100)
    cmd = PaintPixelCommand(pixels, 5, 10, (255, 0, 0, 255))
    stack.push(cmd)      # executes and pushes
    stack.undo()         # reverts
    stack.redo()         # re-applies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Optional, Sequence


class Command(ABC):
    """Abstract base for all undoable operations.

    Subclasses must implement :meth:`execute` and :meth:`undo`.
    """

    @abstractmethod
    def execute(self) -> None:
        """Apply the command's effect."""

    @abstractmethod
    def undo(self) -> None:
        """Revert the command's effect."""

    @property
    def description(self) -> str:
        """Human-readable label shown in the Edit menu (override as needed)."""
        return type(self).__name__


class CompositeCommand(Command):
    """Groups multiple commands into a single undoable unit.

    Args:
        commands: Ordered sequence of child commands.
        description: Optional label for the composite.
    """

    def __init__(
        self,
        commands: Sequence[Command],
        description: str = "Compound action",
    ) -> None:
        self._commands: List[Command] = list(commands)
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    def execute(self) -> None:
        """Execute all child commands in order."""
        for cmd in self._commands:
            cmd.execute()

    def undo(self) -> None:
        """Undo all child commands in reverse order."""
        for cmd in reversed(self._commands):
            cmd.undo()

    def add(self, command: Command) -> None:
        """Append a command to this group (before it has been pushed).

        Args:
            command: The command to add.
        """
        self._commands.append(command)


class CommandStack:
    """Manages undo and redo stacks with a configurable depth limit.

    Args:
        max_depth: Maximum number of undoable steps retained.  When the stack
            is full, the oldest command is silently discarded.
    """

    def __init__(self, max_depth: int = 100) -> None:
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self._max_depth = max_depth
        self._undo_stack: Deque[Command] = deque()
        self._redo_stack: Deque[Command] = deque()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def can_undo(self) -> bool:
        """True when there is at least one action to undo."""
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        """True when there is at least one action to redo."""
        return bool(self._redo_stack)

    @property
    def undo_description(self) -> Optional[str]:
        """Description label of the next undo action, or None."""
        if self._undo_stack:
            return self._undo_stack[-1].description
        return None

    @property
    def redo_description(self) -> Optional[str]:
        """Description label of the next redo action, or None."""
        if self._redo_stack:
            return self._redo_stack[-1].description
        return None

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def push(self, command: Command, *, execute: bool = True) -> None:
        """Push a command onto the undo stack, optionally executing it first.

        Pushing a new command always clears the redo stack.

        Args:
            command: The command to record.
            execute: If True (default), call :meth:`~Command.execute` before
                pushing.
        """
        if execute:
            command.execute()
        self._redo_stack.clear()
        self._undo_stack.append(command)
        if len(self._undo_stack) > self._max_depth:
            self._undo_stack.popleft()

    def undo(self) -> Optional[Command]:
        """Undo the most recent command.

        Returns:
            The command that was undone, or None if the stack is empty.
        """
        if not self._undo_stack:
            return None
        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)
        return command

    def redo(self) -> Optional[Command]:
        """Re-apply the most recently undone command.

        Returns:
            The command that was re-applied, or None if the redo stack is empty.
        """
        if not self._redo_stack:
            return None
        command = self._redo_stack.pop()
        command.execute()
        self._undo_stack.append(command)
        return command

    def clear(self) -> None:
        """Discard all undo and redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def __repr__(self) -> str:
        return (
            f"CommandStack(max_depth={self._max_depth}, "
            f"undo={len(self._undo_stack)}, redo={len(self._redo_stack)})"
        )
