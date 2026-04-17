# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Layer-level undoable operations.

Each command is pushed through :class:`~spriter.commands.base.CommandStack`
so that all layer manipulations are fully undoable and redoable.

Commands
--------
* :class:`AddLayerCommand`      — insert a new blank layer
* :class:`RemoveLayerCommand`   — delete a layer (saves state for undo)
* :class:`DuplicateLayerCommand`— copy a layer above itself
* :class:`MoveLayerCommand`     — reorder layers
* :class:`MergeLayerDownCommand`— merge a layer into the one below
* :class:`FlattenCommand`       — flatten all visible layers into one
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..commands.base import Command
from ..core.frame import Cel
from ..core.layer import BlendMode, Layer
from ..core.sprite import Sprite


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _copy_cel(cel: Cel) -> Cel:
    """Return a deep copy of *cel*."""
    new_cel = Cel(
        cel.pixels.copy() if cel.pixels is not None else None,
        linked_frame=cel.linked_frame,
    )
    return new_cel


def _shift_cel_layers_up(
    sprite: Sprite,
    from_index: int,
    extra_cels: Optional[Dict[int, Cel]] = None,
    extra_index: Optional[int] = None,
) -> None:
    """Shift all layer indices >= *from_index* up by one in ``_cels``.

    Optionally inserts *extra_cels* (keyed by frame index) at *extra_index*.
    """
    new_cels = {}
    for (li, fi), cel in sprite._cels.items():  # type: ignore[attr-defined]
        new_li = li if li < from_index else li + 1
        new_cels[(new_li, fi)] = cel
    if extra_cels is not None and extra_index is not None:
        for fi, cel in extra_cels.items():
            new_cels[(extra_index, fi)] = cel
    sprite._cels = new_cels  # type: ignore[attr-defined]


def _shift_cel_layers_down(sprite: Sprite, removed_index: int) -> None:
    """Shift all layer indices > *removed_index* down by one in ``_cels``,
    discarding any cels at exactly *removed_index*."""
    new_cels = {}
    for (li, fi), cel in sprite._cels.items():  # type: ignore[attr-defined]
        if li == removed_index:
            continue
        new_li = li if li < removed_index else li - 1
        new_cels[(new_li, fi)] = cel
    sprite._cels = new_cels  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Add layer
# ---------------------------------------------------------------------------


class AddLayerCommand(Command):
    """Insert a new transparent layer into the sprite.

    Args:
        sprite: The owning sprite.
        name: Display name for the new layer.
        index: Position to insert at; appends at the top if ``None``.
        visible: Initial visibility flag.
        locked: Initial lock state.
        opacity: Initial opacity (0–255).
        blend_mode: Initial blend mode.
    """

    def __init__(
        self,
        sprite: Sprite,
        name: str = "Layer",
        *,
        index: Optional[int] = None,
        visible: bool = True,
        locked: bool = False,
        opacity: int = 255,
        blend_mode: BlendMode = BlendMode.NORMAL,
    ) -> None:
        self._sprite = sprite
        self._name = name
        self._index = index
        self._visible = visible
        self._locked = locked
        self._opacity = opacity
        self._blend_mode = blend_mode
        self._actual_index: Optional[int] = None

    @property
    def description(self) -> str:
        return f"Add Layer '{self._name}'"

    def execute(self) -> None:
        layer = self._sprite.add_layer(
            self._name,
            index=self._index,
            visible=self._visible,
            locked=self._locked,
            opacity=self._opacity,
            blend_mode=self._blend_mode,
        )
        self._actual_index = self._sprite._layers.index(layer)  # type: ignore[attr-defined]

    def undo(self) -> None:
        assert self._actual_index is not None
        self._sprite.remove_layer(self._actual_index)


# ---------------------------------------------------------------------------
# Remove layer
# ---------------------------------------------------------------------------


class RemoveLayerCommand(Command):
    """Delete the layer at *layer_index*, saving state for undo.

    Args:
        sprite: The owning sprite.
        layer_index: Index of the layer to remove.
    """

    def __init__(self, sprite: Sprite, layer_index: int) -> None:
        self._sprite = sprite
        self._index = layer_index
        # Snapshot the layer object and its cels before execution.
        self._layer: Layer = sprite._layers[layer_index]  # type: ignore[attr-defined]
        self._cels: Dict[int, Cel] = {
            fi: _copy_cel(cel)
            for fi in range(sprite.frame_count)
            if (cel := sprite._cels.get((layer_index, fi))) is not None  # type: ignore[attr-defined]
        }

    @property
    def description(self) -> str:
        return f"Remove Layer '{self._layer.name}'"

    def execute(self) -> None:
        self._sprite.remove_layer(self._index)

    def undo(self) -> None:
        # Shift existing cel layer-indices >= _index up by 1, then inject
        # the saved cels at _index.
        _shift_cel_layers_up(
            self._sprite,
            self._index,
            extra_cels=self._cels,
            extra_index=self._index,
        )
        self._sprite._layers.insert(self._index, self._layer)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Duplicate layer
# ---------------------------------------------------------------------------


class DuplicateLayerCommand(Command):
    """Copy a layer and insert the copy directly above it.

    Args:
        sprite: The owning sprite.
        layer_index: Index of the layer to duplicate.
    """

    def __init__(self, sprite: Sprite, layer_index: int) -> None:
        self._sprite = sprite
        self._source_index = layer_index
        self._new_index = layer_index + 1

    @property
    def description(self) -> str:
        return "Duplicate Layer"

    def execute(self) -> None:
        src = self._sprite._layers[self._source_index]  # type: ignore[attr-defined]
        new_layer = Layer(
            src.name + " copy",
            visible=src.visible,
            locked=src.locked,
            opacity=src.opacity,
            blend_mode=src.blend_mode,
        )

        # Shift layers above the insertion point up.
        _shift_cel_layers_up(self._sprite, self._new_index)
        self._sprite._layers.insert(self._new_index, new_layer)  # type: ignore[attr-defined]

        # Copy source cels to the new layer slot.
        for fi in range(self._sprite.frame_count):
            src_cel = self._sprite._cels.get((self._source_index, fi))  # type: ignore[attr-defined]
            if src_cel is not None:
                self._sprite._cels[(self._new_index, fi)] = _copy_cel(src_cel)  # type: ignore[attr-defined]

    def undo(self) -> None:
        _shift_cel_layers_down(self._sprite, self._new_index)
        self._sprite._layers.pop(self._new_index)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Move layer
# ---------------------------------------------------------------------------


class MoveLayerCommand(Command):
    """Reorder layers by moving *from_index* to *to_index*.

    Args:
        sprite: The owning sprite.
        from_index: Current layer position.
        to_index: Target layer position.
    """

    def __init__(self, sprite: Sprite, from_index: int, to_index: int) -> None:
        self._sprite = sprite
        self._from = from_index
        self._to = to_index

    @property
    def description(self) -> str:
        return "Move Layer"

    def execute(self) -> None:
        self._sprite.move_layer(self._from, self._to)

    def undo(self) -> None:
        self._sprite.move_layer(self._to, self._from)


# ---------------------------------------------------------------------------
# Merge layer down
# ---------------------------------------------------------------------------


class MergeLayerDownCommand(Command):
    """Merge *layer_index* into the layer immediately below it.

    For each frame the top layer is composited over the bottom layer using the
    top layer's blend mode and opacity.  The merged result replaces the bottom
    layer's pixels.  The top layer is then deleted.

    Args:
        sprite: The owning sprite.
        layer_index: Index of the layer to merge downward (must be > 0).
    """

    def __init__(self, sprite: Sprite, layer_index: int) -> None:
        if layer_index <= 0:
            raise ValueError("Cannot merge the bottom-most layer down")
        self._sprite = sprite
        self._top_index = layer_index
        self._bottom_index = layer_index - 1

        # Snapshot both layers' cels before execution.
        self._top_layer: Layer = sprite._layers[layer_index]  # type: ignore[attr-defined]
        self._top_cels: Dict[int, Cel] = {
            fi: _copy_cel(cel)
            for fi in range(sprite.frame_count)
            if (cel := sprite._cels.get((layer_index, fi))) is not None  # type: ignore[attr-defined]
        }
        self._bottom_cels_before: Dict[int, Cel] = {
            fi: _copy_cel(cel)
            for fi in range(sprite.frame_count)
            if (cel := sprite._cels.get((layer_index - 1, fi))) is not None  # type: ignore[attr-defined]
        }

    @property
    def description(self) -> str:
        return "Merge Layer Down"

    def execute(self) -> None:
        from ..core.compositor import _blend_rgb  # type: ignore[attr-defined]

        top_layer = self._sprite._layers[self._top_index]  # type: ignore[attr-defined]
        alpha_scale = top_layer.opacity / 255.0

        for fi in range(self._sprite.frame_count):
            top_cel = self._sprite._cels.get((self._top_index, fi))  # type: ignore[attr-defined]
            bot_cel = self._sprite._cels.get((self._bottom_index, fi))  # type: ignore[attr-defined]

            # Start with the bottom layer's pixels (or transparent).
            if bot_cel is not None and bot_cel.pixels is not None:
                dst = bot_cel.pixels.astype(np.float32)
            else:
                dst = np.zeros(
                    (self._sprite.height, self._sprite.width, 4), dtype=np.float32
                )

            if top_cel is not None and top_cel.pixels is not None:
                src = top_cel.pixels.astype(np.float32)
                src_rgb_norm = src[..., :3] / 255.0
                src_a = (src[..., 3] / 255.0) * alpha_scale

                dst_rgb_norm = dst[..., :3] / 255.0
                dst_a = dst[..., 3] / 255.0

                blended_rgb = _blend_rgb(
                    src_rgb_norm, dst_rgb_norm, top_layer.blend_mode
                )
                out_a = src_a + dst_a * (1.0 - src_a)
                safe_out_a = np.where(out_a > 0.0, out_a, 1.0)

                merged = np.empty(
                    (self._sprite.height, self._sprite.width, 4), dtype=np.float32
                )
                for ch in range(3):
                    merged[..., ch] = (
                        (
                            blended_rgb[..., ch] * src_a
                            + dst_rgb_norm[..., ch] * dst_a * (1.0 - src_a)
                        )
                        / safe_out_a
                        * 255.0
                    )
                merged[..., 3] = out_a * 255.0
                dst = merged

            result = np.clip(dst, 0, 255).astype(np.uint8)
            self._sprite.set_cel_pixels(self._bottom_index, fi, result)

        # Remove the top layer.
        self._sprite.remove_layer(self._top_index)

    def undo(self) -> None:
        # Step 1: re-shift existing cels to make room for the top layer.
        _shift_cel_layers_up(
            self._sprite,
            self._top_index,
            extra_cels=self._top_cels,
            extra_index=self._top_index,
        )
        # Step 2: re-insert the top layer object.
        self._sprite._layers.insert(self._top_index, self._top_layer)  # type: ignore[attr-defined]
        # Step 3: restore the bottom layer's pre-merge cels.
        for fi, cel in self._bottom_cels_before.items():
            self._sprite._cels[(self._bottom_index, fi)] = cel  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


class FlattenCommand(Command):
    """Flatten all visible layers into a single layer.

    The composited result for each frame is stored in a new layer named
    "Merged".  All original layers (visible and invisible) are removed.

    Args:
        sprite: The owning sprite.
    """

    def __init__(self, sprite: Sprite) -> None:
        self._sprite = sprite
        # Snapshot all layers and cels for undo.
        self._saved_layers: list = list(sprite._layers)  # type: ignore[attr-defined]
        self._saved_cels: Dict[tuple, Cel] = {
            key: _copy_cel(cel)
            for key, cel in sprite._cels.items()  # type: ignore[attr-defined]
        }

    @property
    def description(self) -> str:
        return "Flatten"

    def execute(self) -> None:
        from ..core.compositor import composite_frame

        # Composite each frame into a merged cel.
        merged: Dict[int, Cel] = {}
        for fi in range(self._sprite.frame_count):
            pixels = composite_frame(self._sprite, fi)
            merged[fi] = Cel(pixels)

        # Replace layers with a single "Merged" layer.
        new_layer = Layer(
            "Merged",
            visible=True,
            locked=False,
            opacity=255,
            blend_mode=BlendMode.NORMAL,
        )
        self._sprite._layers = [new_layer]  # type: ignore[attr-defined]
        self._sprite._cels = {(0, fi): cel for fi, cel in merged.items()}  # type: ignore[attr-defined]

    def undo(self) -> None:
        self._sprite._layers = list(self._saved_layers)  # type: ignore[attr-defined]
        self._sprite._cels = {  # type: ignore[attr-defined]
            key: _copy_cel(cel) for key, cel in self._saved_cels.items()
        }
