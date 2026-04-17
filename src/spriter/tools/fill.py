# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Fill (bucket) tool — flood-fills a region with the foreground color."""

from __future__ import annotations

from ..utils.geometry import flood_fill, flood_fill_tolerance
from .base import Tool


class FillTool(Tool):
    """Flood-fills the clicked region with the foreground color.

    The fill replaces all contiguous pixels whose color matches (or is within
    :attr:`tolerance` of) the pixel under the cursor.

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
        color = self._paint_color()
        if self.tolerance > 0:
            flood_fill_tolerance(
                w, x, y, color, self.tolerance, connectivity=self.connectivity
            )
        else:
            flood_fill(w, x, y, color, connectivity=self.connectivity)
        self._commit_stroke("Fill")

    def on_drag(self, x: int, y: int) -> None:
        pass  # Fill is a one-shot tool.

    def on_release(self, x: int, y: int) -> None:
        pass
