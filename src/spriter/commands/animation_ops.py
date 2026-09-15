# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Animation-timeline-level undoable operations.

Each command is pushed through :class:`~spriter.commands.base.CommandStack`
so that adding, removing and renaming animation timelines is fully undoable.

Commands
--------
* :class:`AddTimelineCommand`    — append a new animation timeline
* :class:`RemoveTimelineCommand` — delete an animation timeline
* :class:`RenameTimelineCommand` — rename an animation timeline
"""

from __future__ import annotations

from ..commands.base import Command
from ..core.animation import AnimationTimeline
from ..core.sprite import Sprite


class AddTimelineCommand(Command):
    """Append a new animation timeline and make it active.

    Args:
        sprite: The owning sprite.
        name: Display name; a unique default is generated when omitted.
    """

    def __init__(self, sprite: Sprite, name: str | None = None) -> None:
        self._sprite = sprite
        self._name = name
        self._index: int | None = None
        self._prev_active: int | None = None

    @property
    def description(self) -> str:
        return "Add Animation"

    def execute(self) -> None:
        self._prev_active = self._sprite.active_timeline_index
        timeline = self._sprite.add_timeline(self._name)
        self._index = self._sprite.timelines.index(timeline)
        self._sprite.set_active_timeline(self._index)

    def undo(self) -> None:
        assert self._index is not None and self._prev_active is not None
        self._sprite.remove_timeline(self._index)
        self._sprite.set_active_timeline(
            min(self._prev_active, self._sprite.timeline_count - 1)
        )


class RemoveTimelineCommand(Command):
    """Delete the animation timeline at *index*, saving it for undo.

    Args:
        sprite: The owning sprite.
        index: Timeline index to remove.

    Raises:
        ValueError: If it is the only remaining timeline.
    """

    def __init__(self, sprite: Sprite, index: int) -> None:
        if sprite.timeline_count <= 1:
            raise ValueError("Cannot remove the last remaining animation.")
        self._sprite = sprite
        self._index = index
        self._timeline: AnimationTimeline | None = None
        self._prev_active: int | None = None

    @property
    def description(self) -> str:
        return "Remove Animation"

    def execute(self) -> None:
        self._prev_active = self._sprite.active_timeline_index
        self._timeline = self._sprite.remove_timeline(self._index)

    def undo(self) -> None:
        assert self._timeline is not None and self._prev_active is not None
        self._sprite.insert_timeline(self._index, self._timeline)
        self._sprite.set_active_timeline(self._prev_active)


class RenameTimelineCommand(Command):
    """Rename the animation timeline at *index*.

    Args:
        sprite: The owning sprite.
        index: Timeline index to rename.
        name: New display name.
    """

    def __init__(self, sprite: Sprite, index: int, name: str) -> None:
        self._sprite = sprite
        self._index = index
        self._name = name
        self._old_name: str | None = None

    @property
    def description(self) -> str:
        return "Rename Animation"

    def execute(self) -> None:
        self._old_name = self._sprite.timelines[self._index].name
        self._sprite.rename_timeline(self._index, self._name)

    def undo(self) -> None:
        assert self._old_name is not None
        self._sprite.rename_timeline(self._index, self._old_name)
