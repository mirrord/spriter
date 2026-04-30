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
        # Cached brush mask, invalidated when (size, shape) changes.
        self._brush_mask_cache: Optional[np.ndarray] = None
        self._brush_mask_key: Optional[Tuple[int, BrushShape]] = None

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
        """Return the in-progress pixel buffer for live preview.

        The returned array shares storage with the tool's working buffer; it
        is intended for immediate read-only consumption by the renderer (which
        copies the data into a QImage).  Returns ``None`` when no stroke is
        active.
        """
        return self._working

    def selection_preview_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Return the in-progress selection rectangle as (x0, y0, x1, y1), or ``None``.

        Canvas-space coordinates, normalised so x0 <= x1 and y0 <= y1.
        The default implementation always returns ``None``; selection tools
        override this to expose their live drag rectangle.
        """
        return None

    def cancel(self) -> None:
        """Cancel any in-progress interaction (stroke or selection drag).

        Safe to call at any time; clears transient stroke state.
        """
        self._before = None
        self._working = None

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
        key = (s, self.brush_shape)
        if self._brush_mask_key == key and self._brush_mask_cache is not None:
            return self._brush_mask_cache
        if self.brush_shape == BrushShape.CIRCLE:
            centre = (s - 1) / 2.0
            ys, xs = np.ogrid[:s, :s]
            mask = ((xs - centre) ** 2 + (ys - centre) ** 2) <= (s / 2.0) ** 2
            mask = np.ascontiguousarray(mask)
        else:
            mask = np.ones((s, s), dtype=bool)
        self._brush_mask_cache = mask
        self._brush_mask_key = key
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
        h, w = pixels.shape[:2]

        # Compute clipped destination rectangle.
        dx0 = x - half
        dy0 = y - half
        dx1 = dx0 + s
        dy1 = dy0 + s
        cx0 = max(0, dx0)
        cy0 = max(0, dy0)
        cx1 = min(w, dx1)
        cy1 = min(h, dy1)
        if cx1 <= cx0 or cy1 <= cy0:
            return

        # Corresponding sub-mask of the brush.
        mx0 = cx0 - dx0
        my0 = cy0 - dy0
        mx1 = mx0 + (cx1 - cx0)
        my1 = my0 + (cy1 - cy0)
        sub_mask = bm[my0:my1, mx0:mx1]

        dst = pixels[cy0:cy1, cx0:cx1]

        if erase:
            dst[sub_mask] = (0, 0, 0, 0)
            return

        a = int(color[3]) * int(self.opacity) // 255
        if a <= 0:
            return
        if a >= 255:
            dst[sub_mask] = (color[0], color[1], color[2], 255)
            return

        # Partial alpha: vectorised Porter-Duff "over".
        sa = a / 255.0
        sel = dst[sub_mask].astype(np.float32)  # (N, 4)
        if sel.size == 0:
            return
        da = sel[:, 3] / 255.0
        out_a = sa + da * (1.0 - sa)
        # avoid divide-by-zero
        safe = np.where(out_a > 0.0, out_a, 1.0)
        inv_sa = 1.0 - sa
        for c in range(3):
            sel[:, c] = (color[c] * sa + sel[:, c] * da * inv_sa) / safe
        sel[:, 3] = out_a * 255.0
        dst[sub_mask] = sel.astype(np.uint8)

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
