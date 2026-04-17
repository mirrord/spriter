# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Animation model — sequences, loop modes, tags and per-frame timing.

The :class:`Animation` object is owned by a
:class:`~spriter.core.sprite.Sprite` as ``sprite.animation`` and controls
how the frame sequence is played back.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from .sprite import Sprite


class LoopMode(Enum):
    """Looping behaviour for an animation sequence."""

    LOOP = "loop"  # Repeats indefinitely
    PING_PONG = "pingpong"  # Forward then backward, then forward …
    ONE_SHOT = "oneshot"  # Plays once and stops at the last frame


class AnimationTag:
    """A named sub-sequence of frames with its own loop mode.

    Args:
        name: Display name for the tag.
        from_frame: First frame index (inclusive, 0-based).
        to_frame: Last frame index (inclusive, 0-based).
        color: RGB tuple for colour-coding in the timeline UI.
        loop_mode: How this tag loops when played in isolation.
    """

    def __init__(
        self,
        name: str,
        from_frame: int,
        to_frame: int,
        color: Tuple[int, int, int] = (255, 0, 0),
        loop_mode: LoopMode = LoopMode.LOOP,
    ) -> None:
        if from_frame < 0:
            raise ValueError(f"from_frame must be >= 0, got {from_frame!r}")
        if to_frame < from_frame:
            raise ValueError(
                f"to_frame must be >= from_frame; got {to_frame} < {from_frame}"
            )
        self.name = name
        self.from_frame = from_frame
        self.to_frame = to_frame
        self.color = color
        self.loop_mode = loop_mode

    def __repr__(self) -> str:
        return (
            f"AnimationTag({self.name!r}, "
            f"frames {self.from_frame}..{self.to_frame}, "
            f"{self.loop_mode.name})"
        )


class Animation:
    """Controls animation playback settings for a sprite.

    Attached to a :class:`~spriter.core.sprite.Sprite` as
    ``sprite.animation``.

    Args:
        default_fps: Default frames-per-second used when per-frame duration is
            not considered (e.g. when building a playback timer).
        loop_mode: Global loop mode for the full animation.
    """

    def __init__(
        self,
        default_fps: int = 12,
        loop_mode: LoopMode = LoopMode.LOOP,
    ) -> None:
        if default_fps <= 0:
            raise ValueError(f"default_fps must be positive, got {default_fps!r}")
        self.default_fps = default_fps
        self.loop_mode = loop_mode
        self._tags: List[AnimationTag] = []

    # ------------------------------------------------------------------
    # Tag management
    # ------------------------------------------------------------------

    @property
    def tags(self) -> List[AnimationTag]:
        """Ordered list of animation tags (copy)."""
        return list(self._tags)

    def add_tag(
        self,
        name: str,
        from_frame: int,
        to_frame: int,
        color: Tuple[int, int, int] = (255, 0, 0),
        loop_mode: LoopMode = LoopMode.LOOP,
    ) -> AnimationTag:
        """Create and register a new animation tag.

        Args:
            name: Tag display name.
            from_frame: First frame of the range (inclusive).
            to_frame: Last frame of the range (inclusive).
            color: RGB display colour.
            loop_mode: Loop mode for this sub-sequence.

        Returns:
            The newly created :class:`AnimationTag`.
        """
        tag = AnimationTag(name, from_frame, to_frame, color, loop_mode)
        self._tags.append(tag)
        return tag

    def remove_tag(self, name: str) -> None:
        """Remove the tag with the given *name*.

        Args:
            name: Tag to remove.

        Raises:
            KeyError: If no tag with that name exists.
        """
        for i, tag in enumerate(self._tags):
            if tag.name == name:
                del self._tags[i]
                return
        raise KeyError(f"No animation tag named {name!r}")

    # ------------------------------------------------------------------
    # Playback helpers
    # ------------------------------------------------------------------

    def get_frame_duration_ms(self, sprite: "Sprite", frame_index: int) -> int:
        """Return the display duration in milliseconds for a given frame.

        Uses ``sprite.frames[frame_index].duration_ms`` when the frame exists;
        otherwise falls back to ``1000 // default_fps``.

        Args:
            sprite: The sprite whose frame durations to query.
            frame_index: Index into the frame list.

        Returns:
            Duration in milliseconds (always ≥ 1).
        """
        if sprite.frame_count > 0 and 0 <= frame_index < sprite.frame_count:
            return sprite.frames[frame_index].duration_ms
        return max(1, 1000 // self.default_fps)

    def next_frame(self, current: int, total: int) -> int:
        """Return the index of the frame that follows *current*.

        For :attr:`LoopMode.LOOP` and :attr:`LoopMode.PING_PONG` this wraps
        around to frame 0 after the last frame.  For
        :attr:`LoopMode.ONE_SHOT` playback stops at the last frame.

        Note:
            PING_PONG direction tracking is the caller's responsibility.
            This method always advances by +1 and wraps.

        Args:
            current: Current frame index.
            total: Total number of frames.

        Returns:
            Next frame index.
        """
        if total <= 1:
            return 0
        if self.loop_mode == LoopMode.ONE_SHOT:
            return min(current + 1, total - 1)
        return (current + 1) % total
