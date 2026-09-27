"""Construction of the frozen Turbo-VAED decoder used for latent features."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def _patch_transformers_cache() -> None:
    try:
        import transformers

        if not hasattr(transformers, "EncoderDecoderCache") and hasattr(
            transformers, "DynamicCache"
        ):
            transformers.EncoderDecoderCache = transformers.DynamicCache
    except Exception:
        pass


def build_turbo_decoder(args: argparse.Namespace, device: torch.device):
    _patch_transformers_cache()
    os.environ.setdefault("_CHECK_PEFT", "0")
    turbo_repo_root = Path(args.turbo_repo_root).expanduser().resolve()
    turbo_src = turbo_repo_root / "diffusers_vae" / "src"
    if str(turbo_src) not in sys.path:
        sys.path.insert(0, str(turbo_src))

    from diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import (
        AutoencoderKLTurboVAED,
    )

    config_path = (
        Path(args.turbo_config).expanduser().resolve()
        if args.turbo_config
        else turbo_repo_root / "configs" / "Turbo-VAED-Cog.json"
    )
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    model = AutoencoderKLTurboVAED.from_config(config=config)
    state = torch.load(
        Path(args.turbo_checkpoint).expanduser().resolve(), map_location="cpu"
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "gen_model" in state:
        state = state["gen_model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported Turbo checkpoint payload: {type(state)!r}")
    state = {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    missing, unexpected = model.decoder.load_state_dict(state, strict=False)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return model, {
        "missing": list(missing),
        "unexpected": list(unexpected),
        "config_path": str(config_path),
    }
