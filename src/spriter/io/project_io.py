# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Serialization and deserialization of ``.spriter`` project files.

A ``.spriter`` file is a **JSON** document that contains:

* Sprite metadata (canvas size, color mode, format version).
* Layer metadata list (name, visibility, blend mode, opacity).
* Frame metadata list (duration).
* Cel pixel data: one Base64-encoded PNG per (layer, frame) pair.

Autosave files are written next to the project path with an ``~`` suffix
(e.g. ``my_sprite.spriter~``) and are automatically removed on a successful
:func:`save`.

Example::

    from spriter.core.sprite import Sprite
    from spriter.io.project_io import save, load

    sprite = Sprite(32, 32)
    sprite.add_layer("Background")
    sprite.add_frame()
    save(sprite, "my_sprite.spriter")

    loaded = load("my_sprite.spriter")
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from ..core.frame import Cel, Frame
from ..core.layer import BlendMode, Layer
from ..core.sprite import Sprite

_FORMAT_VERSION = 1


def save(sprite: Sprite, path: Union[str, Path]) -> None:
    """Save *sprite* as a ``.spriter`` project file.

    The file is first written to a temporary ``~``-suffixed path so that any
    I/O error leaves the original file intact.

    Args:
        sprite: The sprite document to serialize.
        path: Destination file path (typically ending in ``.spriter``).
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + "~")
    try:
        data = _sprite_to_dict(sprite)
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load(path: Union[str, Path]) -> Sprite:
    """Load a ``.spriter`` project file and return a :class:`~spriter.core.sprite.Sprite`.

    Args:
        path: Path to the ``.spriter`` file.

    Returns:
        The deserialized :class:`~spriter.core.sprite.Sprite`.

    Raises:
        ValueError: If the file format version is not supported.
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return _dict_to_sprite(data)


def autosave(sprite: Sprite, path: Union[str, Path]) -> Path:
    """Write a recovery copy of *sprite* next to *path*.

    The autosave file is named ``<stem>.spriter~``.  Calling :func:`save`
    removes it automatically on success.

    Args:
        sprite: The sprite to back up.
        path: The primary project path (used to derive the autosave location).

    Returns:
        The autosave file path that was written.
    """
    path = Path(path)
    autosave_path = path.with_suffix(path.suffix + "~")
    data = _sprite_to_dict(sprite)
    autosave_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return autosave_path


# ---------------------------------------------------------------------------
# Internal serialization helpers
# ---------------------------------------------------------------------------


def _sprite_to_dict(sprite: Sprite) -> Dict[str, Any]:
    layers_data = [_layer_to_dict(layer) for layer in sprite.layers]
    frames_data = [_frame_to_dict(frame) for frame in sprite.frames]

    cels_data: Dict[str, str] = {}
    for li in range(sprite.layer_count):
        for fi in range(sprite.frame_count):
            cel = sprite._cels.get((li, fi))
            if cel is None:
                continue
            key = f"{li},{fi}"
            if cel.is_linked and cel.linked_frame is not None:
                cels_data[key] = json.dumps({"linked_frame": cel.linked_frame})
            elif cel.pixels is not None:
                cels_data[key] = _pixels_to_b64png(cel.pixels)
            # Empty cels are omitted to keep file size small.

    return {
        "version": _FORMAT_VERSION,
        "width": sprite.width,
        "height": sprite.height,
        "color_mode": sprite.color_mode,
        "layers": layers_data,
        "frames": frames_data,
        "cels": cels_data,
    }


def _dict_to_sprite(data: Dict[str, Any]) -> Sprite:
    version = int(data.get("version", 1))
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported .spriter format version {version}. "
            f"Expected version {_FORMAT_VERSION}."
        )

    sprite = Sprite(
        int(data["width"]),
        int(data["height"]),
        color_mode=str(data.get("color_mode", "RGBA")),
    )

    for layer_data in data.get("layers", []):
        layer = Layer(
            name=str(layer_data.get("name", "Layer")),
            visible=bool(layer_data.get("visible", True)),
            locked=bool(layer_data.get("locked", False)),
            opacity=int(layer_data.get("opacity", 255)),
            blend_mode=BlendMode(layer_data.get("blend_mode", "normal")),
        )
        sprite._layers.append(layer)

    for frame_data in data.get("frames", []):
        frame = Frame(duration_ms=int(frame_data.get("duration_ms", 100)))
        sprite._frames.append(frame)

    for key_str, cel_value in data.get("cels", {}).items():
        li, fi = (int(i) for i in key_str.split(","))
        # Detect linked-cel JSON vs raw base64 PNG.
        if cel_value.startswith("{"):
            cel_meta = json.loads(cel_value)
            sprite._cels[(li, fi)] = Cel(linked_frame=int(cel_meta["linked_frame"]))
        else:
            pixels = _b64png_to_pixels(cel_value)
            sprite._cels[(li, fi)] = Cel(pixels)

    return sprite


def _layer_to_dict(layer: Layer) -> Dict[str, Any]:
    return {
        "name": layer.name,
        "visible": layer.visible,
        "locked": layer.locked,
        "opacity": layer.opacity,
        "blend_mode": layer.blend_mode.value,
    }


def _frame_to_dict(frame: Frame) -> Dict[str, Any]:
    return {"duration_ms": frame.duration_ms}


def _pixels_to_b64png(pixels: np.ndarray) -> str:
    """Encode a NumPy RGBA array as a Base64-encoded PNG string."""
    img = Image.fromarray(pixels, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64png_to_pixels(b64_str: str) -> np.ndarray:
    """Decode a Base64-encoded PNG string back to a NumPy RGBA array."""
    buf = io.BytesIO(base64.b64decode(b64_str))
    img = Image.open(buf).convert("RGBA")
    return np.array(img, dtype=np.uint8)
