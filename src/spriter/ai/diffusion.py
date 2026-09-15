# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Diffusion backend for frame prediction (optional ``spriter[diffusion]`` extra).

Wraps a Hugging Face :mod:`diffusers` image-to-image pipeline.  ``torch`` and
``diffusers`` are imported lazily inside each function so importing this module
never fails when the optional dependencies are absent — callers should gate use
on :func:`is_available`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Default image-to-image parameters.
DEFAULT_STRENGTH = 0.6
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_STEPS = 30


def is_available() -> bool:
    """Return ``True`` when the diffusion optional dependencies are importable."""
    import importlib.util

    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("diffusers") is not None
    )


def _require_available() -> None:
    if not is_available():
        raise RuntimeError(
            "Diffusion features require the optional dependencies. Install them "
            "with:  pip install spriter[diffusion]"
        )


def model_info(model_path: str | Path) -> dict[str, Any]:
    """Return a description of a local model without loading it.

    Inspects the filesystem only, so this carries no machine-learning
    dependencies and is safe to call regardless of :func:`is_available`.

    Args:
        model_path: Path to a diffusers model directory or a single
            ``.safetensors`` checkpoint file.

    Returns:
        A dict with keys:

        * ``path`` — the resolved path as a string.
        * ``exists`` — whether the path exists.
        * ``kind`` — ``"file"``, ``"directory"`` or ``"missing"``.
        * ``type`` — human-readable model type.
        * ``size_bytes`` — total size in bytes (0 when missing).
    """
    path = Path(model_path)
    exists = path.exists()
    if not exists:
        kind = "missing"
        type_label = "Not found"
        size_bytes = 0
    elif path.is_file():
        kind = "file"
        if path.suffix.lower() == ".safetensors":
            type_label = "Single-file checkpoint (.safetensors)"
        else:
            type_label = f"Single file ({path.suffix or 'no extension'})"
        size_bytes = path.stat().st_size
    else:
        kind = "directory"
        type_label = "Diffusers model folder"
        size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return {
        "path": str(path),
        "exists": exists,
        "kind": kind,
        "type": type_label,
        "size_bytes": size_bytes,
    }


def download_model(repo_id: str, dest_dir: str | Path) -> Path:
    """Download a diffusion model from the Hugging Face Hub to *dest_dir*.

    Args:
        repo_id: Hugging Face repository ID (e.g. ``"runwayml/stable-diffusion-v1-5"``).
        dest_dir: Local directory to download the model snapshot into.

    Returns:
        The path to the downloaded model directory.

    Raises:
        RuntimeError: If the optional dependencies are not installed.
    """
    _require_available()
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(repo_id=repo_id, local_dir=str(dest))
    return Path(local_path)


def load_pipeline(model_path: str | Path) -> Any:
    """Load an image-to-image pipeline from a local model.

    Args:
        model_path: Path to either a diffusers model directory or a single
            ``.safetensors`` checkpoint file.

    Returns:
        A ready-to-run diffusers image-to-image pipeline moved to the best
        available device (CUDA when present, otherwise CPU).

    Raises:
        RuntimeError: If the optional dependencies are not installed.
        FileNotFoundError: If *model_path* does not exist.
    """
    _require_available()
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    import torch  # type: ignore[import-not-found]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if path.is_file() and path.suffix.lower() == ".safetensors":
        # Single-file checkpoint (e.g. a downloaded Stable Diffusion model).
        from diffusers import (  # type: ignore[import-not-found]
            StableDiffusionImg2ImgPipeline,
        )

        pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
            str(path), torch_dtype=dtype
        )
    else:
        from diffusers import (  # type: ignore[import-not-found]
            AutoPipelineForImage2Image,
        )

        pipeline = AutoPipelineForImage2Image.from_pretrained(
            str(path), torch_dtype=dtype
        )
    pipeline = pipeline.to(device)
    return pipeline


def generate(
    pipeline: Any,
    init_rgba: np.ndarray,
    prompt: str = "",
    *,
    strength: float = DEFAULT_STRENGTH,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    steps: int = DEFAULT_STEPS,
) -> np.ndarray:
    """Run image-to-image generation and return the result as an RGBA array.

    Args:
        pipeline: A diffusers image-to-image pipeline from :func:`load_pipeline`.
        init_rgba: The initialising ``H×W×4`` ``uint8`` RGBA image (the current
            frame).  Transparent pixels are flattened onto white before use.
        prompt: Optional text prompt guiding the generation.
        strength: How much the init image is transformed (0–1).  Higher values
            deviate further from the source.
        guidance_scale: Classifier-free guidance scale.
        steps: Number of denoising steps.

    Returns:
        The generated image as an ``H×W×4`` ``uint8`` RGBA array.
    """
    init_image = _rgba_to_rgb_image(init_rgba)
    result = pipeline(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    )
    out_image = result.images[0].convert("RGBA")
    return np.array(out_image, dtype=np.uint8)


def _rgba_to_rgb_image(rgba: np.ndarray) -> Image.Image:
    """Flatten an RGBA array onto a white background and return an RGB image."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expected an (H, W, 4) RGBA array")
    rgb = Image.new("RGB", (rgba.shape[1], rgba.shape[0]), (255, 255, 255))
    fg = Image.fromarray(rgba, mode="RGBA")
    rgb.paste(fg, mask=fg.split()[3])
    return rgb
