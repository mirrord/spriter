# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Automatic, zero-configuration fine-tuning of the next-frame predictor.

A small LoRA adapter is trained on the project's own frames so that predicted
frames keep the sprite's appearance.  The design goal is that this stays
invisible to the user: data is harvested automatically (see
:mod:`spriter.ai.dataset`), every hyperparameter has a sensible default, and the
resulting adapter is saved to a per-project location that the predictor loads on
its own.

Heavy machine-learning work lives in :func:`_run_lora_training`, which imports
``torch``/``diffusers``/``peft`` lazily; everything else is import-safe and
unit-testable without those optional dependencies.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .dataset import harvest_frames
from .diffusion import SD_NATIVE_SIZE

# Auto-defaults — deliberately conservative so a usable adapter trains quickly on
# modest hardware with no user tuning.
DEFAULT_TRAIN_STEPS = 400
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_LORA_RANK = 4
DEFAULT_RESOLUTION = SD_NATIVE_SIZE

# Below this many harvested frames there is too little signal to fine-tune.
MIN_TRAIN_FRAMES = 6

# File the LoRA weights are written as (diffusers' default name).
ADAPTER_WEIGHT_NAME = "pytorch_lora_weights.safetensors"

ProgressCallback = Callable[[int, int], None]


def is_train_available() -> bool:
    """Return ``True`` when the fine-tuning optional dependencies are importable."""
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "diffusers", "peft")
    )


def _require_train_available() -> None:
    if not is_train_available():
        raise RuntimeError(
            "Fine-tuning requires the optional training dependencies. Install "
            "them with:  pip install spriter[train]"
        )


def default_adapter_root() -> Path:
    """Return the base directory under which per-project adapters are stored."""
    return Path.home() / ".config" / "spriter" / "adapters"


def adapter_dir_for(project_key: str) -> Path:
    """Return the adapter directory for a project identified by *project_key*.

    *project_key* is typically the project's file path (or a placeholder such as
    ``"untitled"``).  It is hashed so the location is stable and filesystem-safe
    without leaking the full path.

    Args:
        project_key: A stable identifier for the project.

    Returns:
        The per-project adapter directory (not created).
    """
    digest = hashlib.sha1(project_key.encode("utf-8")).hexdigest()[:16]
    return default_adapter_root() / digest


def has_trained_adapter(adapter_dir: str | Path) -> bool:
    """Return ``True`` when *adapter_dir* contains trained LoRA weights."""
    return (Path(adapter_dir) / ADAPTER_WEIGHT_NAME).is_file()


def auto_train_from_sprite(
    sprite: Any,
    model_path: str | Path,
    adapter_dir: str | Path,
    *,
    steps: int = DEFAULT_TRAIN_STEPS,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Harvest *sprite*'s frames and fine-tune a predictor adapter automatically.

    This is the single entry point the UI needs: it requires no configuration
    beyond the base model and where to save the adapter.

    Args:
        sprite: The sprite document to learn from.
        model_path: Base diffusion model to adapt.
        adapter_dir: Directory the trained adapter is written to.
        steps: Number of optimisation steps (defaulted; rarely overridden).
        progress_cb: Optional ``(step, total)`` progress callback.

    Returns:
        The directory the adapter was written to.

    Raises:
        RuntimeError: If the training dependencies are not installed.
        ValueError: If the project has too few frames to train on.
    """
    frames = harvest_frames(sprite)
    return train_next_frame_predictor(
        model_path, frames, adapter_dir, steps=steps, progress_cb=progress_cb
    )


def train_next_frame_predictor(
    model_path: str | Path,
    frames: list[np.ndarray],
    adapter_dir: str | Path,
    *,
    steps: int = DEFAULT_TRAIN_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    rank: int = DEFAULT_LORA_RANK,
    resolution: int = DEFAULT_RESOLUTION,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Fine-tune a LoRA adapter on *frames* and save it to *adapter_dir*.

    Args:
        model_path: Base diffusion model (directory or ``.safetensors``).
        frames: Training images as ``H×W×4`` ``uint8`` RGBA arrays.
        adapter_dir: Directory to write the trained adapter into.
        steps: Number of optimisation steps.
        learning_rate: Optimiser learning rate.
        rank: LoRA rank (adapter capacity).
        resolution: Square resolution frames are trained at.
        progress_cb: Optional ``(step, total)`` progress callback.

    Returns:
        The directory the adapter was written to.

    Raises:
        RuntimeError: If the training dependencies are not installed.
        ValueError: If fewer than :data:`MIN_TRAIN_FRAMES` frames are supplied.
    """
    _require_train_available()
    if len(frames) < MIN_TRAIN_FRAMES:
        raise ValueError(
            f"Need at least {MIN_TRAIN_FRAMES} non-empty frames to fine-tune; "
            f"got {len(frames)}."
        )
    output = Path(adapter_dir)
    output.mkdir(parents=True, exist_ok=True)
    _run_lora_training(
        model_path,
        frames,
        output,
        steps=steps,
        learning_rate=learning_rate,
        rank=rank,
        resolution=resolution,
        progress_cb=progress_cb,
    )
    return output


def _run_lora_training(
    model_path: str | Path,
    frames: list[np.ndarray],
    output: Path,
    *,
    steps: int,
    learning_rate: float,
    rank: int,
    resolution: int,
    progress_cb: ProgressCallback | None,
) -> None:
    """Train a UNet LoRA to reconstruct *frames* and save the weights.

    Standard latent-diffusion objective with an empty prompt: encode each frame
    to latents, add noise, and train only the LoRA parameters to predict it.
    All heavy dependencies are imported here so the module stays import-safe.
    """
    import torch  # type: ignore[import-not-found]
    from diffusers import (  # type: ignore[import-not-found]
        DDPMScheduler,
        StableDiffusionPipeline,
    )
    from peft import LoraConfig  # type: ignore[import-not-found]

    from .diffusion import _select_device_and_dtype

    device, _ = _select_device_and_dtype()
    path = Path(model_path)
    if path.is_file() and path.suffix.lower() == ".safetensors":
        pipe = StableDiffusionPipeline.from_single_file(
            str(path), safety_checker=None, requires_safety_checker=False
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            str(path), safety_checker=None, requires_safety_checker=False
        )

    vae = pipe.vae.to(device)
    unet = pipe.unet.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)

    # Pre-encode a fixed empty-prompt embedding (unconditional training).
    tokens = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        empty_embeds = text_encoder(tokens)[0]

    tensors = [_frame_to_tensor(f, resolution, device) for f in frames]

    unet.train()
    total = max(1, steps)
    for step in range(total):
        image = tensors[step % len(tensors)]
        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (1,), device=device
        ).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        pred = unet(noisy, timesteps, encoder_hidden_states=empty_embeds).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if progress_cb is not None:
            progress_cb(step + 1, total)

    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output),
        unet_lora_layers=_lora_state_dict(unet),
        weight_name=ADAPTER_WEIGHT_NAME,
    )


def _frame_to_tensor(frame: np.ndarray, resolution: int, device: Any) -> Any:
    """Flatten an RGBA frame onto white and return a normalised CHW tensor."""
    import torch  # type: ignore[import-not-found]
    from PIL import Image

    from .diffusion import _rgba_to_rgb_image

    image = _rgba_to_rgb_image(frame).resize(
        (resolution, resolution), Image.Resampling.LANCZOS
    )
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _lora_state_dict(unet: Any) -> Any:
    """Extract the trained LoRA layers from *unet* for saving."""
    from peft.utils import get_peft_model_state_dict  # type: ignore[import-not-found]

    return get_peft_model_state_dict(unet)
