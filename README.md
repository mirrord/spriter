# Spriter

A pixel-art sprite editor built with Python and PyQt6. Spriter provides a focused, keyboard-friendly workflow for creating sprites and animations, with full layer support, blend modes, and a clean undo/redo history.

[![PyPI - Version](https://img.shields.io/pypi/v/spriter.svg)](https://pypi.org/project/spriter)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spriter.svg)](https://pypi.org/project/spriter)

---

## Features

- **Pixel canvas** — zoom from 1× to 64×, pan with middle-mouse or Space+drag, toggleable pixel grid
- **Drawing tools** — pencil (with pixel-perfect stroke), eraser, line, rectangle, ellipse, flood fill, eyedropper, rectangular selection, move, and text stamp
- **Layers** — add, delete, duplicate, merge down, flatten; drag-to-reorder; rename layers; assign foreground/background roles; per-layer opacity and blend mode
- **Blend modes** — Normal, Multiply, Screen, Overlay, Darken, Lighten (Porter-Duff alpha compositing)
- **Color picker** — SV-square + hue-strip gradient picker; foreground/background swatches with swap (`X`); HSV sliders, RGB spinboxes, hex input; 16-color palette grid; 16-slot recent-colors row; palette import/export (JASC-PAL, GIMP GPL, hex); right-click palette slots to set or delete entries
- **Animation** — multiple named animation timelines per project (e.g. idle/run/jump), navigable with up/down controls in the timeline panel; per-timeline frame list, duration, loop mode (loop/ping-pong/one-shot), and FPS; animation tags (named ranges within a timeline), real-time preview window, onion skinning (configurable depth and opacity), drag-to-reorder frames, right-click timeline menu
- **Transforms** — flip H/V, rotate 90°/180°, scale canvas, crop to selection, autocrop (shrink canvas to opaque-pixel bbox across all layers/frames), shift/wrap, outline non-transparent pixels, replace color, brightness/contrast/hue-saturation adjustments, scale selection
- **Undo/redo** — configurable history (default 100 levels) with labeled action names, covering all drawing, layer, and transform operations
- **Project files** — `.spriter` format (JSON manifest with embedded PNG cel data)
- **Export** — PNG (single frame or all frames), animated GIF, sprite sheets (horizontal/vertical/grid) with JSON atlas — projects with multiple animations export one row per animation, ICO/cursor, palette files
- **Import** — PNG/any Pillow-supported format as new sprite, sprite-sheet splitting into frames (optionally one animation timeline per row, fixed grid or auto-detected irregular spacing), automatic background-colour detection on sheet import with a choice to remove it, split it onto its own layer, or import as-is, palette files
- **Clipboard** — copy selection as PNG, paste from clipboard
- **Symmetry mode** — horizontal and/or vertical axis mirroring while drawing
- **Reference image overlay** — pin a translucent reference image on the canvas
- **Tiling preview** — 3×3 seamless-texture preview overlay
- **Recent files** — quick-open list in the File menu
- **Drag-and-drop** — open project or image files by dropping onto the window
- **Preferences** — persistent settings for canvas defaults, grid/checker colors, undo depth, autosave interval, theme, and customizable keybindings
- **Diffusion frame prediction** *(optional)* — generate the next animation frame from the current one using a local, user-selectable Hugging Face diffusers image-to-image model; the result has its flat background removed and is uniformly scaled to fit the canvas (see [Diffusion](#diffusion-optional))

## Installation

```console
pip install spriter
```

Requires Python ≥ 3.8 and a working Qt 6 installation (pulled in automatically via `PyQt6`).

The optional diffusion frame-prediction feature needs extra machine-learning
dependencies, installed via the `diffusion` extra:

```console
pip install spriter[diffusion]
```

## Usage

Launch the GUI:

```console
spriter
```

Or from Python:

```python
from spriter.app import main
main()
```

### Keyboard shortcuts

| Action | Shortcut |
|---|---|
| New project | `Ctrl+N` |
| Open project | `Ctrl+O` |
| Save | `Ctrl+S` |
| Save As | `Ctrl+Shift+S` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |
| Zoom in / out | `Ctrl+=` / `Ctrl+-` |
| Fit to window | `Ctrl+Shift+H` |
| Toggle grid | `Ctrl+G` |
| Copy selection | `Ctrl+C` |
| Paste from clipboard | `Ctrl+V` |
| Add layer | `Ctrl+Shift+N` |
| Duplicate layer | `Ctrl+J` |
| Merge down | `Ctrl+E` |
| Pencil | `B` |
| Eraser | `E` |
| Line | `L` |
| Rectangle | `R` |
| Ellipse | `O` |
| Fill | `G` |
| Eyedropper | `I` |
| Select | `S` |
| Move | `M` |
| Text | `T` |
| Swap foreground/background | `X` |

### Diffusion (optional)

With the `diffusion` extra installed, the **Diffusion** menu offers:

- **Select Model…** — point Spriter at a local diffusers model directory or a single `.safetensors` checkpoint file.
- **Download Model…** — fetch a model from the Hugging Face Hub by repo ID
  (e.g. `runwayml/stable-diffusion-v1-5`) into the local model cache.
- **Model Info…** — show details about the currently selected model (path, type, size, and whether the optional dependencies are installed).
- **Generate Next Frame…** — generate a new frame from the current one using
  image-to-image diffusion. You can supply an optional text prompt to guide the
  result. The generated image has its flat background keyed out and is uniformly
  scaled down to fit the canvas, then inserted as a new frame right after the
  current one (undoable).

Generation runs on the GPU when a CUDA device is available, otherwise on the CPU.

## Development

This project uses [Hatch](https://hatch.pypa.io/) for environment and build management.

```console
# Run the test suite (700+ tests across all phases)
hatch test

# Type checking
hatch run types:check
```

See [BUILDING.md](BUILDING.md) for building a standalone Windows `.exe`.

## License

`spriter` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
