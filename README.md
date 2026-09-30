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
- **Themes** — a bold blue & red-orange "ember" theme (default), plus plain "dark" and "light" options, switchable in Preferences
- **Diffusion frame prediction** *(optional)* — restyle a frame or predict the next animation frame with a local, user-selectable Hugging Face diffusers model; next-frame prediction is motion- and ControlNet-guided and can be sharpened by automatic per-project fine-tuning (see [Diffusion](#diffusion-optional))

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

To also enable automatic per-project fine-tuning of the next-frame predictor,
install the `train` extra (it includes everything in `diffusion`):

```console
pip install spriter[train]
```

### GPU acceleration

Diffusion automatically runs on an NVIDIA GPU (CUDA) or Apple Silicon (MPS)
when one is available, falling back to the CPU otherwise. The device in use is
shown in **Diffusion → Model Info**.

The default `torch` wheel from PyPI is **CPU-only**. To use an NVIDIA GPU,
install a CUDA build of PyTorch from the official index that matches your
driver, for example:

```console
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then verify it with:

```console
python -c "import torch; print(torch.cuda.is_available())"
```

Apple Silicon (MPS) is supported by the standard `torch` wheel — no extra step.

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

- **Select Model…** — point Spriter at a local diffusers model directory or a single `.safetensors` checkpoint file (Stable Diffusion 1.x/2.x and SDXL are auto-detected).
- **Select ControlNet…** — point Spriter at a local ControlNet used to guide next-frame prediction. Both formats are accepted: a diffusers-format directory (`config.json` + weights) or an original single-file checkpoint (`.safetensors`/`.ckpt`) alongside its `config.yaml` (the A1111/ControlNet-v1 layout). A scribble or lineart ControlNet for SD 1.x works well (e.g. `lllyasviel/control_v11p_sd15_scribble`); download it with **Download Model…** or place it on disk.
- **Select IP-Adapter…** — *(optional)* point Spriter at an IP-Adapter weight file (e.g. `ip-adapter_sd15.safetensors` from `h94/IP-Adapter`). When set, next-frame prediction locks the sprite's identity from the current frame so the character stays consistent while the ControlNet drives the new pose.
- **Select Video Model…** — point Spriter at a local Wan 2.1 image-to-video model directory (e.g. `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`) used by **Animate From Frame**.
- **Select Video LoRA…** — *(optional)* apply a LoRA (a `.safetensors` file or directory) on top of the video model.
- **Download Model…** — fetch a model from the Hugging Face Hub by repo ID
  (e.g. `runwayml/stable-diffusion-v1-5`) into the local model cache.
- **Model Info…** — show details about the currently selected model (path, type, size, and whether the optional dependencies are installed).
- **Restyle Frame…** — reinterpret the current frame with image-to-image
  diffusion and an optional text prompt. The result has its flat background
  keyed out and is scaled to fit the canvas, then inserted as a new frame right
  after the current one (undoable). Use this for style variations of a single
  frame.
- **Predict Next Frame…** — predict the *next* animation frame. Motion is
  estimated from the two preceding frames and used to guide a ControlNet so the
  result advances the animation rather than merely restyling it. When an
  IP-Adapter is selected, the sprite's identity is held steady across frames.
  Requires a selected ControlNet and at least two existing frames.
- **Animate From Frame…** — generate a short video from the current frame with
  Wan 2.1 (image-to-video) and a motion prompt, then insert the extracted video
  frames as subsequent animation/tween frames (undoable). On CUDA the model is
  loaded with CPU offload so large Wan checkpoints fit in consumer VRAM.

The tool palette also has a **🎬 Sheet AI** button for **sprite-sheet
generation by prompt**: describe an animation and pick the frame width, height
and count. Spriter uses Wan 2.1 **text-to-video** (with the selected video model
and optional LoRA) to render a clip, scales its frames down to your chosen
dimensions, and drops them into a new animation timeline ready to export as a
sprite sheet (undoable). The generation reuses the same **Select Video Model…**
/ **Select Video LoRA…** settings — point them at a Wan text-to-video model for
this feature.

#### Automatic fine-tuning

With the `train` extra installed, next-frame prediction can learn your
project's own motion. This is designed to stay out of your way:

- **Frame pairs are harvested automatically** from every animation in the
  project — there is nothing to label or configure.
- The first time you predict a frame without a trained predictor, Spriter
  offers to build one; training then runs in the background with a progress
  indicator and **no hyperparameters to set**.
- The resulting adapter is saved per-project and **applied automatically** on
  subsequent predictions — you never have to select or manage it.

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
