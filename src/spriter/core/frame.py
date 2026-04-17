# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Frame and Cel data models.

A *Frame* represents a single animation step with a duration.
A *Cel* is the pixel-data intersection of one Layer and one Frame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Cel:
    """Pixel buffer for a (layer, frame) pair.

    Args:
        pixels: RGBA uint8 array of shape (height, width, 4), or None for an
            empty (transparent) cel.
        linked_frame: Index of another frame whose Cel this one mirrors.
            When set, ``pixels`` should be None and reads should be redirected.
    """

    def __init__(
        self,
        pixels: Optional[np.ndarray] = None,
        *,
        linked_frame: Optional[int] = None,
    ) -> None:
        if pixels is not None:
            if pixels.ndim != 3 or pixels.shape[2] != 4:
                raise ValueError("pixels must be an (H, W, 4) uint8 array")
            if pixels.dtype != np.uint8:
                raise ValueError("pixels dtype must be uint8")
        self._pixels = pixels
        self.linked_frame = linked_frame

    @property
    def is_empty(self) -> bool:
        """True when this cel carries no pixel data of its own."""
        return self._pixels is None

    @property
    def is_linked(self) -> bool:
        """True when this cel mirrors another frame's pixel data."""
        return self.linked_frame is not None

    @property
    def pixels(self) -> Optional[np.ndarray]:
        """The raw pixel buffer (may be None for empty cels)."""
        return self._pixels

    @pixels.setter
    def pixels(self, value: Optional[np.ndarray]) -> None:
        if value is not None:
            if value.ndim != 3 or value.shape[2] != 4:
                raise ValueError("pixels must be an (H, W, 4) uint8 array")
            if value.dtype != np.uint8:
                raise ValueError("pixels dtype must be uint8")
        self._pixels = value

    def clear(self) -> None:
        """Erase all pixel data, making this an empty cel."""
        self._pixels = None

    def __repr__(self) -> str:
        if self.is_linked:
            return f"Cel(linked_frame={self.linked_frame})"
        if self.is_empty:
            return "Cel(empty)"
        assert self._pixels is not None
        h, w = self._pixels.shape[:2]
        return f"Cel(pixels={w}x{h})"


class Frame:
    """A single animation frame with a configurable display duration.

    Args:
        duration_ms: How long this frame is displayed in milliseconds.
    """

    def __init__(self, duration_ms: int = 100) -> None:
        if duration_ms <= 0:
            raise ValueError(f"duration_ms must be positive, got {duration_ms}")
        self.duration_ms = duration_ms

    def __repr__(self) -> str:
        return f"Frame(duration_ms={self.duration_ms})"
