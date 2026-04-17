# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Layer data model: metadata and blend modes."""

from __future__ import annotations

from enum import Enum


class BlendMode(Enum):
    """Supported layer blend modes."""

    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    DARKEN = "darken"
    LIGHTEN = "lighten"


class Layer:
    """Stores per-layer metadata (no pixel data — pixels live in Cels).

    Args:
        name: Display name of the layer.
        visible: Whether the layer is rendered.
        locked: When True, editing is blocked.
        opacity: Alpha multiplier in range 0–255.
        blend_mode: Compositing blend mode.
    """

    def __init__(
        self,
        name: str = "Layer",
        *,
        visible: bool = True,
        locked: bool = False,
        opacity: int = 255,
        blend_mode: BlendMode = BlendMode.NORMAL,
    ) -> None:
        if not (0 <= opacity <= 255):
            raise ValueError(f"opacity must be 0–255, got {opacity}")
        self.name = name
        self.visible = visible
        self.locked = locked
        self.opacity = opacity
        self.blend_mode = blend_mode

    def __repr__(self) -> str:
        return (
            f"Layer(name={self.name!r}, visible={self.visible}, "
            f"locked={self.locked}, opacity={self.opacity}, "
            f"blend_mode={self.blend_mode!r})"
        )
