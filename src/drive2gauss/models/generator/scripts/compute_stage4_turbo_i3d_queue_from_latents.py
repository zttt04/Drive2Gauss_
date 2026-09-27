"""Build a Turbo-VAED/I3D Stage4 generated queue from saved RGB latents.

The Stage4 training path decodes current predictions with Turbo-VAED-Cog, so its
warm queue should use the same decoder rather than historical CogVideoX-decoded
JPGs. This torchrun-compatible tool recursively indexes saved ``*_latent.pt``
payloads, keeps one payload per token, decodes six cameras one at a time with
Turbo-VAED-Cog, and stores StyleGAN-V I3D features as ``[N, 6, 400]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from DISTT.utils.frechet_rgb_stage4 import StyleGANVI3DFeatureExtractor


CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
)


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _init_distributed() -> None:
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def _patch_transformers_cache() -> None:
    try:
        import transformers

        if not hasattr(transformers, "EncoderDecoderCache") and hasattr(transformers, "DynamicCache"):
            transformers.EncoderDecoderCache = transformers.DynamicCache
    except Exception:
        pass


def _load_turbo_decoder(args: argparse.Namespace, device: torch.device):
    _patch_transformers_cache()
    os.environ.setdefault("_CHECK_PEFT", "0")
    repo_root = args.turbo_repo_root.expanduser().resolve()
    turbo_src = repo_root / "diffusers_vae" / "src"
    if str(turbo_src) not in sys.path:
        sys.path.insert(0, str(turbo_src))

    from diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import AutoencoderKLTurboVAED

    with args.turbo_config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    decoder = AutoencoderKLTurboVAED.from_config(config=config)
    state = torch.load(args.turbo_checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "gen_model" in state:
        state = state["gen_model"]
    if not isinstance(state, dict):
        raise TypeError(f"unsupported Turbo checkpoint payload: {type(state)!r}")
    state = {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    missing, unexpected = decoder.decoder.load_state_dict(state, strict=False)
    decoder.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
    return decoder, list(missing), list(unexpected)


def _index_latents(roots: list[Path]) -> dict[str, Path]:
    records: dict[str, tuple[int, Path]] = {}
    for root in roots:
        for path in root.expanduser().resolve().rglob("*_latent.pt"):
            token = path.name[: -len("_latent.pt")]
            mtime = path.stat().st_mtime_ns
            previous = records.get(token)
            if previous is None or mtime > previous[0]:
                records[token] = (mtime, path)
    return {token: record[1] for token, record in records.items()}


def _evenly_spaced(tokens: list[str], maximum: int) -> list[str]:
    if maximum <= 0 or len(tokens) <= maximum:
        return tokens
    indices = np.linspace(0, len(tokens) - 1, num=maximum, dtype=np.int64)
    return [tokens[int(index)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--turbo-repo-root", type=Path, required=True)
    parser.add_argument("--turbo-config", type=Path, required=True)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--i3d-checkpoint", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument("--video-length", type=int, default=17)
    args = parser.parse_args()

    _init_distributed()
    rank = _rank()
    world_size = _world_size()
    if not torch.cuda.is_available():
        raise RuntimeError("Turbo/I3D queue extraction requires CUDA")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))

    latent_by_token = _index_latents(args.latent_root)
    tokens = _evenly_spaced(sorted(latent_by_token), int(args.max_tokens))
    if not tokens:
        raise RuntimeError("no *_latent.pt payloads found")
    if rank == 0:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    turbo, missing, unexpected = _load_turbo_decoder(args, device)
    extractor = StyleGANVI3DFeatureExtractor(
        args.i3d_checkpoint,
        clip_length=16,
        resolution=224,
        batch_size=1,
        device=device,
    ).eval()

    local_tokens = tokens[rank::world_size]
    local_features = []
    with torch.no_grad():
        for token in local_tokens:
            payload = torch.load(latent_by_token[token], map_location="cpu")
            rgb_latent = payload.get("rgb_latent") if isinstance(payload, dict) else None
            if not isinstance(rgb_latent, torch.Tensor) or rgb_latent.ndim != 5:
                raise ValueError(f"{latent_by_token[token]} has no RGB latent [NC,C,T,H,W]")
            if rgb_latent.shape[:2] != (len(CAMERAS), 16):
                raise ValueError(f"unexpected RGB latent shape {tuple(rgb_latent.shape)} for {token}")
            view_features = []
            for view_index in range(len(CAMERAS)):
                latent = rgb_latent[view_index : view_index + 1].to(
                    device=device,
                    dtype=torch.float16,
                )
                decoded = turbo.decode(latent * float(args.latent_scale), return_dict=False)[0]
                decoded = decoded[:, :, : int(args.video_length)]
                view_features.append(extractor(decoded).cpu()[0])
            local_features.append(torch.stack(view_features, dim=0))
            print(
                f"rank={rank} completed={len(local_features)}/{len(local_tokens)} token={token}",
                flush=True,
            )

    feature_tensor = (
        torch.stack(local_features, dim=0)
        if local_features
        else torch.empty((0, len(CAMERAS), 400), dtype=torch.float32)
    )
    shard_path = args.output_dir / f"turbo_i3d_queue_rank{rank:02d}.pt"
    torch.save({"tokens": local_tokens, "features": feature_tensor}, shard_path)
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        feature_by_token = {}
        for shard_rank in range(world_size):
            shard = torch.load(
                args.output_dir / f"turbo_i3d_queue_rank{shard_rank:02d}.pt",
                map_location="cpu",
            )
            feature_by_token.update(zip(shard["tokens"], shard["features"]))
        merged = torch.stack([feature_by_token[token] for token in tokens], dim=0)
        metadata = {
            "decoder": "Turbo-VAED-Cog",
            "latent_scale": float(args.latent_scale),
            "feature_extractor": "StyleGAN-V I3D TorchScript",
            "feature_shape": list(merged.shape),
            "video_length": int(args.video_length),
            "cameras": list(CAMERAS),
            "latent_roots": [str(path.expanduser().resolve()) for path in args.latent_root],
            "turbo_checkpoint": str(args.turbo_checkpoint.expanduser().resolve()),
            "i3d_checkpoint": str(args.i3d_checkpoint.expanduser().resolve()),
            "missing_decoder_keys": missing,
            "unexpected_decoder_keys": unexpected,
        }
        output_path = args.output_dir / "stage4_rgb_i3d_step3600_turbo_queue.pt"
        torch.save(
            {"tokens": tokens, "features": merged, "view_names": CAMERAS, "metadata": metadata},
            output_path,
        )
        (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"saved queue {tuple(merged.shape)} to {output_path}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
