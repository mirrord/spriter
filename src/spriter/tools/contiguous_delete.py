# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Contiguous delete tool — flood-erases a region to fully transparent.

Behaves like :class:`~spriter.tools.fill.FillTool` but instead of replacing
matching pixels with the foreground color, it sets them to ``(0, 0, 0, 0)``
(fully transparent).
"""

from __future__ import annotations

from ..utils.geometry import flood_fill, flood_fill_tolerance
from .base import Tool

_TRANSPARENT = (0, 0, 0, 0)


class ContiguousDeleteTool(Tool):
    """Flood-deletes the clicked region (writes fully transparent pixels).

    The deletion replaces all contiguous pixels whose color matches (or is
    within :attr:`tolerance` of) the pixel under the cursor with
    ``(0, 0, 0, 0)``.

    Attributes:
        tolerance: Maximum RGBA Euclidean distance from seed color.  ``0``
            requires an exact color match.
        connectivity: ``4`` (cardinal) or ``8`` (diagonal) neighbours.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tolerance: int = 0
        self.connectivity: int = 4

    def on_press(self, x: int, y: int) -> None:
        w = self._begin_stroke()
        if self.tolerance > 0:
            flood_fill_tolerance(
                w, x, y, _TRANSPARENT, self.tolerance, connectivity=self.connectivity
            )
        else:
            flood_fill(w, x, y, _TRANSPARENT, connectivity=self.connectivity)
        self._commit_stroke("Contiguous Delete")

    def on_drag(self, x: int, y: int) -> None:
        pass  # One-shot tool.

    def on_release(self, x: int, y: int) -> None:
        pass
