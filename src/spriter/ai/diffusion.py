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
DEFAULT_STRENGTH = 0.8
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


def _select_device_and_dtype() -> tuple[str, Any]:
    """Return the best available device and matching torch dtype.

    Prefers CUDA (fp16), then Apple MPS (fp32), then CPU (fp32).  fp16 is only
    used on CUDA where it is both fast and well supported.
    """
    import torch  # type: ignore[import-not-found]

    if torch.cuda.is_available():
        return "cuda", torch.float16
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def active_device() -> str:
    """Return the device diffusion will run on: ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Returns ``"cpu"`` when the optional dependencies are absent, so callers can
    display it without gating on :func:`is_available`.
    """
    if not is_available():
        return "cpu"
    device, _ = _select_device_and_dtype()
    return device


def _dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """Return the dtype keyword accepted by the installed diffusers loaders.

    diffusers renamed ``torch_dtype`` to ``dtype`` (the old name warns from
    0.34 and is removed in 1.0.0), so pick whichever the installed version
    understands to stay quiet on new releases while supporting ``>=0.27``.
    """
    import diffusers  # type: ignore[import-not-found]

    try:
        major, minor = (int(part) for part in diffusers.__version__.split(".")[:2])
    except (ValueError, IndexError, TypeError, AttributeError):
        return {"torch_dtype": dtype}
    if (major, minor) >= (0, 34):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def _finalize_pipeline(pipeline: Any, device: str, disable_safety_checker: bool) -> Any:
    """Null the safety checker (optionally) and move the pipeline to *device*."""
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


def _maybe_load_adapter(pipeline: Any, adapter_path: str | Path | None) -> None:
    """Load LoRA weights from *adapter_path* into *pipeline* when present.

    A missing or ``None`` path is a no-op, so callers can pass an optional
    per-project adapter without first checking whether one has been trained.
    """
    if adapter_path is None:
        return
    path = Path(adapter_path)
    if not path.exists():
        return
    pipeline.load_lora_weights(str(path))


# Default influence of the IP-Adapter identity lock; high enough to hold the
# sprite's look, low enough to let ControlNet drive the new pose.
DEFAULT_IP_ADAPTER_SCALE = 0.6


def _maybe_load_ip_adapter(
    pipeline: Any,
    ip_adapter_path: str | Path | None,
    scale: float = DEFAULT_IP_ADAPTER_SCALE,
) -> bool:
    """Load an IP-Adapter weight file into *pipeline* when present.

    *ip_adapter_path* is the path to a single IP-Adapter weight file (``.bin``
    or ``.safetensors``); its parent folder and filename are handed to
    diffusers' loader.  A missing or ``None`` path is a no-op.

    Returns:
        ``True`` when an IP-Adapter was loaded, else ``False``.
    """
    if ip_adapter_path is None:
        return False
    path = Path(ip_adapter_path)
    if not path.exists():
        return False
    pipeline.load_ip_adapter(str(path.parent), subfolder="", weight_name=path.name)
    pipeline.set_ip_adapter_scale(scale)
    return True


def load_pipeline(
    model_path: str | Path,
    *,
    disable_safety_checker: bool = True,
    adapter_path: str | Path | None = None,
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
        adapter_path: Optional directory holding a trained LoRA adapter to apply.

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

    device, dtype = _select_device_and_dtype()
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
                str(path), **_dtype_kwargs(dtype)
            )
        else:
            from diffusers import (  # type: ignore[import-not-found]
                StableDiffusionImg2ImgPipeline,
            )

            pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                str(path), **_dtype_kwargs(dtype), **safety_kwargs
            )
    else:
        from diffusers import (  # type: ignore[import-not-found]
            AutoPipelineForImage2Image,
        )

        pipeline = AutoPipelineForImage2Image.from_pretrained(
            str(path), **_dtype_kwargs(dtype), **safety_kwargs
        )
    _maybe_load_adapter(pipeline, adapter_path)
    return _finalize_pipeline(pipeline, device, disable_safety_checker)


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


# Single-file ControlNet checkpoint extensions, preferred order (safetensors first).
_CONTROLNET_CKPT_SUFFIXES = (".safetensors", ".ckpt", ".pth", ".pt", ".bin")


def _sibling_controlnet_yaml(folder: Path) -> Path | None:
    """Return the original-format ``config.yaml`` in *folder*, if any."""
    for name in ("config.yaml", "config.yml"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    for candidate in sorted(folder.glob("*.y*ml")):
        return candidate
    return None


def _resolve_controlnet_source(cn_path: Path) -> tuple[Path | None, Path | None]:
    """Classify *cn_path* as a single-file ControlNet or a diffusers directory.

    Returns:
        ``(checkpoint_file, yaml_file)`` for a single-file ControlNet (the YAML
        may be ``None``), or ``(None, None)`` when *cn_path* is a diffusers-format
        directory that :func:`ControlNetModel.from_pretrained` can load directly.
    """
    if cn_path.is_file():
        if cn_path.suffix.lower() in _CONTROLNET_CKPT_SUFFIXES:
            return cn_path, _sibling_controlnet_yaml(cn_path.parent)
        return None, None
    # A directory: diffusers format wins when a config.json is present.
    if (cn_path / "config.json").is_file():
        return None, None
    checkpoints = [
        p
        for p in cn_path.iterdir()
        if p.is_file() and p.suffix.lower() in _CONTROLNET_CKPT_SUFFIXES
    ]
    if not checkpoints:
        return None, None
    # Prefer safetensors, then the configured suffix order.
    checkpoints.sort(key=lambda p: _CONTROLNET_CKPT_SUFFIXES.index(p.suffix.lower()))
    return checkpoints[0], _sibling_controlnet_yaml(cn_path)


def _controlnet_config_from_ldm(original_config: dict, checkpoint: Any) -> dict:
    """Build a diffusers ControlNet config from an original LDM YAML config.

    Replicates diffusers' ``create_controlnet_diffusers_config_from_ldm`` but
    passes the checkpoint through to the UNet config builder, working around a
    diffusers bug where that argument is dropped (raising ``missing 1 required
    positional argument: 'checkpoint'``).
    """
    from diffusers.loaders.single_file_utils import (  # type: ignore[import-not-found]
        create_unet_diffusers_config_from_ldm,
        set_image_size,
    )

    image_size = set_image_size(checkpoint)
    unet = create_unet_diffusers_config_from_ldm(
        original_config, checkpoint, image_size=image_size
    )
    control_params = original_config["model"]["params"]["control_stage_config"][
        "params"
    ]
    return {
        "conditioning_channels": control_params["hint_channels"],
        "in_channels": unet["in_channels"],
        "down_block_types": unet["down_block_types"],
        "block_out_channels": unet["block_out_channels"],
        "layers_per_block": unet["layers_per_block"],
        "cross_attention_dim": unet["cross_attention_dim"],
        "attention_head_dim": unet["attention_head_dim"],
        "use_linear_projection": unet["use_linear_projection"],
        "class_embed_type": unet["class_embed_type"],
        "addition_embed_type": unet["addition_embed_type"],
        "addition_time_embed_dim": unet["addition_time_embed_dim"],
        "projection_class_embeddings_input_dim": unet[
            "projection_class_embeddings_input_dim"
        ],
        "transformer_layers_per_block": unet["transformer_layers_per_block"],
    }


def _load_controlnet_from_original(
    checkpoint_file: Path, yaml_file: Path, dtype: Any
) -> Any:
    """Load a single-file ControlNet using its original ``config.yaml``.

    This is fully offline: the diffusers config is derived from the YAML rather
    than downloaded from the Hub (which the stock single-file loader attempts).
    """
    import yaml as pyyaml
    from diffusers import ControlNetModel  # type: ignore[import-not-found]
    from diffusers.loaders.single_file_utils import (  # type: ignore[import-not-found]
        convert_controlnet_checkpoint,
    )

    checkpoint = _load_checkpoint_state_dict(checkpoint_file)
    with open(yaml_file, encoding="utf-8") as handle:
        original_config = pyyaml.safe_load(handle)
    config = _controlnet_config_from_ldm(original_config, checkpoint)
    converted = convert_controlnet_checkpoint(checkpoint, config)
    model = ControlNetModel.from_config(config)
    model.load_state_dict(converted, strict=False)
    return model.to(dtype)


def _load_checkpoint_state_dict(checkpoint_file: Path) -> Any:
    """Load a ``.safetensors`` or torch checkpoint into a flat state dict."""
    if checkpoint_file.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        return load_file(str(checkpoint_file))
    import torch  # type: ignore[import-not-found]

    state = torch.load(str(checkpoint_file), map_location="cpu", weights_only=True)
    return state.get("state_dict", state) if isinstance(state, dict) else state


def _load_controlnet_model(cn_path: Path, dtype: Any) -> Any:
    """Load a ControlNet from a diffusers directory or an original single file."""
    from diffusers import ControlNetModel  # type: ignore[import-not-found]

    checkpoint_file, yaml_file = _resolve_controlnet_source(cn_path)
    if checkpoint_file is None:
        return ControlNetModel.from_pretrained(str(cn_path), **_dtype_kwargs(dtype))
    if yaml_file is not None:
        return _load_controlnet_from_original(checkpoint_file, yaml_file, dtype)
    # Single-file checkpoint without a YAML: let diffusers infer the config.
    return ControlNetModel.from_single_file(
        str(checkpoint_file), **_dtype_kwargs(dtype)
    )


def load_controlnet_pipeline(
    model_path: str | Path,
    controlnet_path: str | Path,
    *,
    disable_safety_checker: bool = True,
    adapter_path: str | Path | None = None,
    ip_adapter_path: str | Path | None = None,
    ip_adapter_scale: float = DEFAULT_IP_ADAPTER_SCALE,
) -> Any:
    """Load a Stable Diffusion ControlNet image-to-image pipeline.

    Used by :func:`generate_next_frame` to condition generation on a predicted
    next-frame structure map.  Only SD1.x/2.x is supported here (the SDXL
    ControlNet img2img pipeline is a different class); an SDXL base model should
    use the plain :func:`generate` restyle path instead.

    The ControlNet may be either a diffusers-format directory (``config.json`` +
    weights) or an original single-file checkpoint (``.safetensors``/``.ckpt``)
    accompanied by a ``config.yaml`` (the A1111/ControlNet-v1 distribution
    format); both are loaded transparently.

    Args:
        model_path: Path to a diffusers model directory or a single
            ``.safetensors`` SD checkpoint.
        controlnet_path: Path to a local ControlNet: a diffusers-format
            directory, a single-file checkpoint, or a directory holding a
            single-file checkpoint plus its ``config.yaml``.
        disable_safety_checker: When ``True`` (default) the NSFW safety checker
            is not loaded, avoiding false positives on pixel-art content.
        adapter_path: Optional directory holding a trained LoRA adapter that
            teaches the project's own appearance; applied when present.
        ip_adapter_path: Optional IP-Adapter weight file that locks the sprite's
            identity from the current frame; applied when present.
        ip_adapter_scale: Influence of the IP-Adapter identity lock (0–1).

    Returns:
        A ready-to-run ControlNet image-to-image pipeline on the best device.

    Raises:
        RuntimeError: If the optional dependencies are not installed.
        FileNotFoundError: If either path does not exist.
    """
    _require_available()
    path = Path(model_path)
    cn_path = Path(controlnet_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if not cn_path.exists():
        raise FileNotFoundError(f"ControlNet path does not exist: {cn_path}")

    device, dtype = _select_device_and_dtype()
    from diffusers import (  # type: ignore[import-not-found]
        StableDiffusionControlNetImg2ImgPipeline,
    )

    controlnet = _load_controlnet_model(cn_path, dtype)
    safety_kwargs: dict[str, Any] = {}
    if disable_safety_checker:
        safety_kwargs["safety_checker"] = None
        safety_kwargs["requires_safety_checker"] = False

    if path.is_file() and path.suffix.lower() == ".safetensors":
        pipeline = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            str(path), controlnet=controlnet, **_dtype_kwargs(dtype), **safety_kwargs
        )
    else:
        pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            str(path), controlnet=controlnet, **_dtype_kwargs(dtype), **safety_kwargs
        )
    _maybe_load_adapter(pipeline, adapter_path)
    _maybe_load_ip_adapter(pipeline, ip_adapter_path, ip_adapter_scale)
    return _finalize_pipeline(pipeline, device, disable_safety_checker)


def generate_next_frame(
    pipeline: Any,
    context_frames: list[np.ndarray],
    prompt: str = "",
    *,
    strength: float = DEFAULT_STRENGTH,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    steps: int = DEFAULT_STEPS,
    controlnet_conditioning_scale: float = 1.0,
    use_ip_adapter: bool = False,
) -> np.ndarray:
    """Predict the next animation frame using motion-guided ControlNet img2img.

    Motion is estimated from the last two context frames and used to warp the
    current frame's edge map forward; that predicted structure conditions a
    ControlNet pipeline so the result advances the animation rather than merely
    restyling the current frame.

    Args:
        pipeline: A ControlNet image-to-image pipeline from
            :func:`load_controlnet_pipeline`.
        context_frames: Consecutive ``H×W×4`` ``uint8`` RGBA frames, oldest
            first.  The last two are used; at least two are required.
        prompt: Optional text prompt guiding the generation.
        strength: How much the init image is transformed (0–1).
        guidance_scale: Classifier-free guidance scale.
        steps: Number of denoising steps.
        controlnet_conditioning_scale: Weight of the structural guidance.
        use_ip_adapter: When ``True``, feed the current frame as the IP-Adapter
            image so the sprite's identity is preserved (the pipeline must have
            been loaded with an IP-Adapter).

    Returns:
        The generated image as an ``H×W×4`` ``uint8`` RGBA array at generation
        resolution; callers scale it back to the canvas.

    Raises:
        ValueError: If fewer than two context frames are supplied.
    """
    from .motion import edge_map, predict_next_frame

    if len(context_frames) < 2:
        raise ValueError("generate_next_frame requires at least 2 context frames")

    prev, curr = context_frames[-2], context_frames[-1]
    predicted = predict_next_frame(prev, curr)

    init_image = _rgba_to_rgb_image(curr)
    target = _target_generation_size(init_image.size, _native_size_for(pipeline))
    if target != init_image.size:
        init_image = init_image.resize(target, Image.Resampling.LANCZOS)

    control_image = Image.fromarray(edge_map(predicted), mode="L").convert("RGB")
    if control_image.size != target:
        control_image = control_image.resize(target, Image.Resampling.LANCZOS)

    call_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "image": init_image,
        "control_image": control_image,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": steps,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
    }
    if use_ip_adapter:
        # Identity is locked from the current frame; ControlNet drives the pose.
        call_kwargs["ip_adapter_image"] = init_image
    result = pipeline(**call_kwargs)
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
