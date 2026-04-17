# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Abstract tool base and shared utilities.

All drawing tools are *stateless command emitters*: they hold transient stroke
state only during an active press-drag-release cycle and commit the result as a
:class:`~spriter.commands.draw.DrawCelCommand` that is pushed onto the
:class:`~spriter.commands.base.CommandStack`.

Usage::

    from spriter.core.sprite import Sprite
    from spriter.commands.base import CommandStack
    from spriter.tools.pencil import PencilTool

    sprite = Sprite(32, 32)
    sprite.add_layer()
    sprite.add_frame()
    stack = CommandStack()

    tool = PencilTool(sprite, stack)
    tool.foreground = (255, 0, 0, 255)
    tool.on_press(5, 5)
    tool.on_drag(6, 6)
    tool.on_release(6, 6)
    # A DrawCelCommand is now on the stack.
    stack.undo()  # revert
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from ..commands.base import CommandStack
from ..commands.draw import DrawCelCommand
from ..core.sprite import Sprite

Color = Tuple[int, int, int, int]


class BrushShape(Enum):
    """Shape used when stamping a brush larger than 1 pixel."""

    SQUARE = "square"
    CIRCLE = "circle"


class Tool(ABC):
    """Abstract base class for all interactive drawing tools.

    Subclasses implement :meth:`on_press`, :meth:`on_drag`, and
    :meth:`on_release`.  Most drawing tools should call :meth:`_begin_stroke`
    at press time and :meth:`_commit_stroke` at release time to leverage the
    built-in undo/redo plumbing.

    Args:
        sprite: The sprite document to operate on.
        stack: The undo/redo command stack.
    """

    def __init__(self, sprite: Sprite, stack: CommandStack) -> None:
        self._sprite = sprite
        self._stack = stack
        self.foreground: Color = (0, 0, 0, 255)
        self.background: Color = (255, 255, 255, 255)
        self.brush_size: int = 1
        self.brush_shape: BrushShape = BrushShape.SQUARE
        self.opacity: int = 255
        self.layer_index: int = 0
        self.frame_index: int = 0
        self._before: Optional[np.ndarray] = None
        self._working: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def on_press(self, x: int, y: int) -> None:
        """Handle a mouse/stylus press at canvas position (x, y)."""

    @abstractmethod
    def on_drag(self, x: int, y: int) -> None:
        """Handle movement while the primary button is held."""

    @abstractmethod
    def on_release(self, x: int, y: int) -> None:
        """Handle button release at canvas position (x, y)."""

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview_overlay(self) -> Optional[np.ndarray]:
        """Return a copy of the in-progress pixel buffer for live preview.

        Returns ``None`` when no stroke is active.
        """
        return self._working.copy() if self._working is not None else None

    # ------------------------------------------------------------------
    # Stroke helpers
    # ------------------------------------------------------------------

    def _begin_stroke(self) -> np.ndarray:
        """Snapshot the active cel and return a mutable working copy.

        Must be called at the start of every stroke (in :meth:`on_press`).

        Returns:
            The working pixel buffer to draw into.
        """
        cel = self._sprite.get_cel(self.layer_index, self.frame_index)
        src = cel.pixels
        if src is not None:
            self._before = src.copy()
        else:
            self._before = np.zeros(
                (self._sprite.height, self._sprite.width, 4), dtype=np.uint8
            )
        self._working = self._before.copy()
        return self._working

    def _commit_stroke(self, description: str = "Draw") -> None:
        """Apply the in-progress working buffer, respecting the active selection.

        Creates and pushes a :class:`~spriter.commands.draw.DrawCelCommand` onto
        the stack **without** re-executing it (the pixels are applied directly).

        If the selection mask is set, only pixels where ``selection_mask == True``
        are updated; all other pixels retain their pre-stroke values.

        Does nothing if no stroke is active or if the pixels are unchanged.
        """
        if self._before is None or self._working is None:
            return

        mask = self._sprite.selection_mask
        if mask is not None:
            # Merge: keep before-pixels outside the selection.
            result = self._before.copy()
            result[mask] = self._working[mask]
            self._working = result

        if np.array_equal(self._before, self._working):
            self._before = self._working = None
            return

        cmd = DrawCelCommand(
            self._sprite,
            self.layer_index,
            self.frame_index,
            self._before,
            self._working,
            description,
        )
        self._sprite.set_cel_pixels(self.layer_index, self.frame_index, self._working)
        self._stack.push(cmd, execute=False)
        self._before = self._working = None

    # ------------------------------------------------------------------
    # Brush helpers
    # ------------------------------------------------------------------

    def _brush_mask(self) -> np.ndarray:
        """Return a boolean (size × size) mask for the current brush.

        Returns:
            Boolean NumPy array of shape ``(brush_size, brush_size)``.
        """
        s = max(1, self.brush_size)
        mask = np.ones((s, s), dtype=bool)
        if self.brush_shape == BrushShape.CIRCLE:
            centre = (s - 1) / 2.0
            for ry in range(s):
                for rx in range(s):
                    if (rx - centre) ** 2 + (ry - centre) ** 2 > (s / 2) ** 2:
                        mask[ry, rx] = False
        return mask

    def _paint_at(
        self,
        pixels: np.ndarray,
        x: int,
        y: int,
        color: Color,
        *,
        erase: bool = False,
    ) -> None:
        """Stamp the brush at (x, y) onto *pixels*.

        When *erase* is True the brush writes fully transparent pixels (ignores
        *color*).  When *erase* is False the brush alpha-composites *color*
        modulated by :attr:`opacity` over the existing pixel.

        Args:
            pixels: RGBA uint8 array to modify in-place.
            x, y: Centre of the brush stamp.
            color: Paint color.
            erase: If True, paint transparent (erase mode).
        """
        bm = self._brush_mask()
        s = bm.shape[0]
        half = s // 2
        if erase:
            erase_color: Color = (0, 0, 0, 0)
            for dy in range(s):
                for dx in range(s):
                    if bm[dy, dx]:
                        _set_raw(pixels, x - half + dx, y - half + dy, erase_color)
        else:
            a = int(color[3] * self.opacity // 255)
            paint: Color = (color[0], color[1], color[2], a)
            for dy in range(s):
                for dx in range(s):
                    if bm[dy, dx]:
                        _alpha_over(pixels, x - half + dx, y - half + dy, paint)

    def _paint_color(self) -> Color:
        """Return :attr:`foreground` (subclasses may override)."""
        return self.foreground


# ---------------------------------------------------------------------------
# Private pixel-level helpers
# ---------------------------------------------------------------------------


def _set_raw(pixels: np.ndarray, x: int, y: int, color: Color) -> None:
    """Write a pixel directly (no blending).  Silently clips out-of-bounds."""
    h, w = pixels.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        pixels[y, x] = color


def _alpha_over(pixels: np.ndarray, x: int, y: int, src: Color) -> None:
    """Porter-Duff 'over' composite *src* onto a single pixel.

    Args:
        pixels: Destination buffer.
        x, y: Target pixel coordinate.
        src: Source color ``(R, G, B, A)``.
    """
    h, w = pixels.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return
    sa = src[3] / 255.0
    if sa == 0.0:
        return
    if sa >= 1.0:
        pixels[y, x] = src
        return
    dst = pixels[y, x]
    da = dst[3] / 255.0
    out_a = sa + da * (1.0 - sa)
    if out_a == 0.0:
        pixels[y, x] = (0, 0, 0, 0)
        return
    inv_sa = 1.0 - sa
    for c in range(3):
        pixels[y, x, c] = int((src[c] * sa + dst[c] * da * inv_sa) / out_a)
    pixels[y, x, 3] = int(out_a * 255)
