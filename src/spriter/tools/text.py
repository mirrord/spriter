# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Text tool — renders bitmap text onto the canvas.

Text is rendered via Pillow's :mod:`PIL.ImageDraw` using the built-in bitmap
font.  The result is alpha-composited over the active layer at the press
position.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import Tool


class TextTool(Tool):
    """Renders a string of text onto the canvas at the clicked position.

    Attributes:
        text: The string to render.  Set this before calling
            :meth:`on_press`.
        font_size: Approximate pixel height of the rendered characters.
            Passed to Pillow's default font loader.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.text: str = ""
        self.font_size: int = 16

    def on_press(self, x: int, y: int) -> None:
        if not self.text:
            return
        w = self._begin_stroke()
        self._render_text(w, x, y)
        self._commit_stroke("Text")

    def on_drag(self, x: int, y: int) -> None:
        pass  # Text is placed on press.

    def on_release(self, x: int, y: int) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render_text(self, pixels: np.ndarray, x: int, y: int) -> None:
        """Render :attr:`text` at (x, y) onto *pixels* using PIL.

        Args:
            pixels: RGBA uint8 buffer to draw into (modified in-place).
            x, y: Top-left anchor of the text.
        """
        img = Image.fromarray(pixels, "RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=self.font_size)
        r, g, b, a = self.foreground
        # Scale alpha by tool opacity.
        a = int(a * self.opacity // 255)
        draw.text((x, y), self.text, fill=(r, g, b, a), font=font)
        pixels[:] = np.array(img, dtype=np.uint8)
