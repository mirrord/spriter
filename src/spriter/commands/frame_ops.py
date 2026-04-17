# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Frame-level undoable operations.

Commands
--------
* :class:`AddFrameCommand`      — insert a new blank frame
* :class:`RemoveFrameCommand`   — delete a frame (saves state for undo)
* :class:`DuplicateFrameCommand`— copy a frame after itself
* :class:`MoveFrameCommand`     — reorder frames
"""

from __future__ import annotations

from typing import Dict, Optional

from ..commands.base import Command
from ..core.frame import Cel, Frame
from ..core.sprite import Sprite


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _copy_cel(cel: Cel) -> Cel:
    """Return a deep copy of *cel*."""
    return Cel(
        cel.pixels.copy() if cel.pixels is not None else None,
        linked_frame=cel.linked_frame,
    )


def _shift_cel_frames_up(
    sprite: Sprite,
    from_index: int,
    extra_cels: Optional[Dict[int, Cel]] = None,
    extra_index: Optional[int] = None,
) -> None:
    """Shift all frame indices >= *from_index* up by one in ``_cels``."""
    new_cels = {}
    for (li, fi), cel in sprite._cels.items():  # type: ignore[attr-defined]
        new_fi = fi if fi < from_index else fi + 1
        new_cels[(li, new_fi)] = cel
    if extra_cels is not None and extra_index is not None:
        for li, cel in extra_cels.items():
            new_cels[(li, extra_index)] = cel
    sprite._cels = new_cels  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Add frame
# ---------------------------------------------------------------------------


class AddFrameCommand(Command):
    """Insert a new transparent frame into the sprite.

    Args:
        sprite: The owning sprite.
        duration_ms: Display duration of the new frame in milliseconds.
        index: Position to insert at; appends if ``None``.
    """

    def __init__(
        self,
        sprite: Sprite,
        duration_ms: int = 100,
        *,
        index: Optional[int] = None,
    ) -> None:
        self._sprite = sprite
        self._duration = duration_ms
        self._index = index
        self._actual_index: Optional[int] = None

    @property
    def description(self) -> str:
        return "Add Frame"

    def execute(self) -> None:
        frame = self._sprite.add_frame(self._duration, index=self._index)
        self._actual_index = self._sprite._frames.index(frame)  # type: ignore[attr-defined]

    def undo(self) -> None:
        assert self._actual_index is not None
        self._sprite.remove_frame(self._actual_index)


# ---------------------------------------------------------------------------
# Remove frame
# ---------------------------------------------------------------------------


class RemoveFrameCommand(Command):
    """Delete the frame at *frame_index*, saving state for undo.

    Args:
        sprite: The owning sprite.
        frame_index: Index of the frame to remove.
    """

    def __init__(self, sprite: Sprite, frame_index: int) -> None:
        self._sprite = sprite
        self._index = frame_index
        # Snapshot the frame object and all its cels.
        self._frame: Frame = sprite._frames[frame_index]  # type: ignore[attr-defined]
        self._cels: Dict[int, Cel] = {
            li: _copy_cel(cel)
            for li in range(sprite.layer_count)
            if (cel := sprite._cels.get((li, frame_index))) is not None  # type: ignore[attr-defined]
        }

    @property
    def description(self) -> str:
        return "Remove Frame"

    def execute(self) -> None:
        self._sprite.remove_frame(self._index)

    def undo(self) -> None:
        _shift_cel_frames_up(
            self._sprite,
            self._index,
            extra_cels=self._cels,
            extra_index=self._index,
        )
        self._sprite._frames.insert(self._index, self._frame)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Duplicate frame
# ---------------------------------------------------------------------------


class DuplicateFrameCommand(Command):
    """Copy a frame and insert the copy directly after it.

    Args:
        sprite: The owning sprite.
        frame_index: Index of the frame to duplicate.
    """

    def __init__(self, sprite: Sprite, frame_index: int) -> None:
        self._sprite = sprite
        self._source_index = frame_index
        self._new_index = frame_index + 1

    @property
    def description(self) -> str:
        return "Duplicate Frame"

    def execute(self) -> None:
        src = self._sprite._frames[self._source_index]  # type: ignore[attr-defined]
        new_frame = Frame(src.duration_ms)

        # Shift existing frame indices >= new_index up by 1.
        _shift_cel_frames_up(self._sprite, self._new_index)
        self._sprite._frames.insert(self._new_index, new_frame)  # type: ignore[attr-defined]

        # Copy source cels to the new frame slot.
        for li in range(self._sprite.layer_count):
            src_cel = self._sprite._cels.get((li, self._source_index))  # type: ignore[attr-defined]
            if src_cel is not None:
                self._sprite._cels[(li, self._new_index)] = _copy_cel(src_cel)  # type: ignore[attr-defined]

    def undo(self) -> None:
        # Discard the duplicate frame's cels and shift everything back down.
        new_cels = {}
        for (li, fi), cel in self._sprite._cels.items():  # type: ignore[attr-defined]
            if fi == self._new_index:
                continue
            new_fi = fi if fi < self._new_index else fi - 1
            new_cels[(li, new_fi)] = cel
        self._sprite._cels = new_cels  # type: ignore[attr-defined]
        self._sprite._frames.pop(self._new_index)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Move frame
# ---------------------------------------------------------------------------


class MoveFrameCommand(Command):
    """Reorder frames by moving *from_index* to *to_index*.

    Args:
        sprite: The owning sprite.
        from_index: Current frame position.
        to_index: Target frame position.
    """

    def __init__(self, sprite: Sprite, from_index: int, to_index: int) -> None:
        self._sprite = sprite
        self._from = from_index
        self._to = to_index

    @property
    def description(self) -> str:
        return "Move Frame"

    def execute(self) -> None:
        self._sprite.move_frame(self._from, self._to)

    def undo(self) -> None:
        self._sprite.move_frame(self._to, self._from)
