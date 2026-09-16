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

import json
import struct
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Default image-to-image parameters.
DEFAULT_STRENGTH = 0.6
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_STEPS = 30

# Native training resolutions the models expect; running img2img far below these
# produces incoherent (noisy) output, so the init image is upscaled to match.
SD_NATIVE_SIZE = 512
SDXL_NATIVE_SIZE = 1024


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


def load_pipeline(
    model_path: str | Path, *, disable_safety_checker: bool = True
) -> Any:
    """Load an image-to-image pipeline from a local model.

    Single-file ``.safetensors`` checkpoints are inspected to detect whether
    they are Stable Diffusion XL or Stable Diffusion 1.x/2.x, and the matching
    image-to-image pipeline class is used.  Loading an SDXL checkpoint into the
    SD1.x pipeline (the previous behaviour) fails with
    ``argument of type 'NoneType' is not iterable``.

    Args:
        model_path: Path to either a diffusers model directory or a single
            ``.safetensors`` checkpoint file.
        disable_safety_checker: When ``True`` (the default) the Stable Diffusion
            NSFW safety checker is not loaded.  This avoids frequent false
            positives on pixel-art content.

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
    safety_kwargs: dict[str, Any] = {}
    if disable_safety_checker:
        safety_kwargs["safety_checker"] = None
        safety_kwargs["requires_safety_checker"] = False

    if path.is_file() and path.suffix.lower() == ".safetensors":
        if _is_sdxl_checkpoint(path):
            # SDXL has no safety checker component; don't pass those kwargs.
            from diffusers import (  # type: ignore[import-not-found]
                StableDiffusionXLImg2ImgPipeline,
            )

            pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
                str(path), torch_dtype=dtype
            )
        else:
            from diffusers import (  # type: ignore[import-not-found]
                StableDiffusionImg2ImgPipeline,
            )

            pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                str(path), torch_dtype=dtype, **safety_kwargs
            )
    else:
        from diffusers import (  # type: ignore[import-not-found]
            AutoPipelineForImage2Image,
        )

        pipeline = AutoPipelineForImage2Image.from_pretrained(
            str(path), torch_dtype=dtype, **safety_kwargs
        )
    if disable_safety_checker and hasattr(pipeline, "safety_checker"):
        # Ensure the call-time safety-checker path is fully bypassed.
        pipeline.safety_checker = None
    # dtype is already set at load time; suppress the benign fp32-modules advisory
    # that diffusers emits for the device-placement .to() call (empty module list).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="There are .* modules in .* that should be kept in float32",
        )
        pipeline = pipeline.to(device)
    return pipeline


def _read_safetensors_header_keys(path: str | Path) -> list[str]:
    """Return the tensor names stored in a ``.safetensors`` file header.

    Only the small JSON header is read (8-byte length prefix + JSON), so this
    is cheap and does not load any tensor data or require ``torch``.

    Args:
        path: Path to a ``.safetensors`` file.

    Returns:
        The list of tensor key names (excluding the ``__metadata__`` entry).

    Raises:
        ValueError: If the file does not look like a valid safetensors file.
    """
    with open(path, "rb") as fh:
        size_bytes = fh.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Not a valid safetensors file (truncated header)")
        (header_size,) = struct.unpack("<Q", size_bytes)
        header_json = fh.read(header_size)
    if len(header_json) != header_size:
        raise ValueError("Not a valid safetensors file (truncated header)")
    header = json.loads(header_json.decode("utf-8"))
    return [key for key in header if key != "__metadata__"]


def _is_sdxl_checkpoint(path: str | Path) -> bool:
    """Best-effort detection of a Stable Diffusion XL single-file checkpoint.

    SDXL checkpoints carry a second text encoder under the
    ``conditioner.embedders.`` namespace and an SDXL UNet ``label_emb`` block,
    neither of which exist in SD1.x/2.x checkpoints.  Detection failures fall
    back to ``False`` (treat as SD1.x/2.x), preserving prior behaviour.

    Args:
        path: Path to a ``.safetensors`` checkpoint.

    Returns:
        ``True`` when the checkpoint appears to be SDXL.
    """
    try:
        keys = _read_safetensors_header_keys(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return any(
        key.startswith("conditioner.embedders.") or ".label_emb." in key for key in keys
    )


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

    The init image is upscaled to the model's native resolution (512 for SD,
    1024 for SDXL) before generation, since running far below it produces
    incoherent noise.  The returned image is at that generation resolution;
    callers are expected to scale it back down to the canvas.

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
    # Upscale tiny pixel-art canvases to the model's native resolution; running
    # far below it yields incoherent noise.
    target = _target_generation_size(init_image.size, _native_size_for(pipeline))
    if target != init_image.size:
        init_image = init_image.resize(target, Image.Resampling.LANCZOS)
    result = pipeline(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    )
    out_image = result.images[0].convert("RGBA")
    return np.array(out_image, dtype=np.uint8)


def _native_size_for(pipeline: Any) -> int:
    """Return the native generation resolution for *pipeline* (SDXL vs SD)."""
    return SDXL_NATIVE_SIZE if "XL" in type(pipeline).__name__ else SD_NATIVE_SIZE


def _target_generation_size(size: tuple[int, int], native: int) -> tuple[int, int]:
    """Scale *size* so its longer side equals *native*, snapped to multiples of 8.

    Aspect ratio is preserved and each dimension is at least 8.

    Args:
        size: The ``(width, height)`` of the init image.
        native: The model's native resolution (e.g. 512 or 1024).

    Returns:
        The target ``(width, height)`` for generation.
    """
    w, h = size
    longer = max(w, h)
    if longer <= 0:
        return (native, native)
    scale = native / longer

    def snap(value: float) -> int:
        return max(8, int(round(value * scale / 8)) * 8)

    return (snap(w), snap(h))


def _rgba_to_rgb_image(rgba: np.ndarray) -> Image.Image:
    """Flatten an RGBA array onto a white background and return an RGB image."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expected an (H, W, 4) RGBA array")
    rgb = Image.new("RGB", (rgba.shape[1], rgba.shape[0]), (255, 255, 255))
    fg = Image.fromarray(rgba, mode="RGBA")
    rgb.paste(fg, mask=fg.split()[3])
    return rgb
