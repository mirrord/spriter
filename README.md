# Spriter

A pixel-art sprite editor built with Python and PyQt6. Spriter provides a focused, keyboard-friendly workflow for creating sprites and animations, with full layer support, blend modes, and a clean undo/redo history.

[![PyPI - Version](https://img.shields.io/pypi/v/spriter.svg)](https://pypi.org/project/spriter)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spriter.svg)](https://pypi.org/project/spriter)

---

## Features

- **Pixel canvas** — zoom from 1× to 64×, pan with middle-mouse or Space+drag, toggleable pixel grid
- **Drawing tools** — pencil (with pixel-perfect stroke), eraser, line, rectangle, ellipse, flood fill, eyedropper, rectangular selection, move, and text stamp
- **Layers** — add, delete, duplicate, merge down, flatten; drag-to-reorder; per-layer opacity and blend mode
- **Blend modes** — Normal, Multiply, Screen, Overlay, Darken, Lighten (Porter-Duff alpha compositing)
- **Color picker** — foreground/background swatches, HSV sliders, RGB spinboxes, hex input, 16-color palette grid
- **Undo/redo** — configurable history (default 100 levels) covering all drawing and layer operations
- **Project files** — `.spriter` format (JSON manifest with embedded PNG cel data)
- **Export** — PNG (single frame or all frames), animated GIF, sprite sheets (horizontal/vertical/grid) with JSON atlas, ICO/cursor
- **Import** — PNG/any Pillow-supported format as new sprite, sprite-sheet splitting into frames
- **Clipboard** — copy selection as PNG, paste from clipboard
- **Symmetry mode** — horizontal and/or vertical axis mirroring while drawing
- **Reference image overlay** — pin a translucent reference image on the canvas
- **Tiling preview** — 3×3 seamless-texture preview overlay
- **Recent files** — quick-open list in the File menu
- **Drag-and-drop** — open project or image files by dropping onto the window
- **Preferences** — persistent settings for canvas defaults, grid/checker colors, undo depth, autosave interval, theme, and customizable keybindings

## Installation

```console
pip install spriter
```

Requires Python ≥ 3.8 and a working Qt 6 installation (pulled in automatically via `PyQt6`).

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

## Development

This project uses [Hatch](https://hatch.pypa.io/) for environment and build management.

```console
# Run the test suite (468 tests across all phases)
hatch test

# Type checking
hatch run types:check
```

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the full roadmap.

## License

`spriter` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
