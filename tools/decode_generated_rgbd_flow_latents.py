#!/usr/bin/env python3
"""Decode generated six-view RGB/depth/flow latents without rerunning diffusion."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
DISTT_ROOT = REPO_ROOT / "third_party" / "distt"
VIEW_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
)
MODALITIES = ("rgb", "depth", "flow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latent-root",
        type=Path,
        required=True,
        help="Root recursively containing generation/latents/*_latent.pt files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vae-pretrained",
        default=os.environ.get("COGVIDEOX_ROOT", "pretrained/CogVideoX-2b"),
    )
    parser.add_argument("--vae-subfolder", default="vae")
    parser.add_argument("--max-latents", type=int, default=0)
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank, device


def discover_latents(latent_root: Path) -> list[Path]:
    candidates = sorted(latent_root.glob("**/generation/latents/*_latent.pt"))
    if not candidates:
        candidates = sorted(latent_root.glob("**/*_latent.pt"))
    selected: dict[str, Path] = {}
    for path in candidates:
        token = path.name.removesuffix("_latent.pt")
        previous = selected.get(token)
        if previous is None or path.stat().st_mtime_ns > previous.stat().st_mtime_ns:
            selected[token] = path
    return [selected[token] for token in sorted(selected)]


def build_teacher_vae(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, str(DISTT_ROOT))
    try:
        import transformers

        if not hasattr(transformers, "EncoderDecoderCache") and hasattr(
            transformers, "DynamicCache"
        ):
            transformers.EncoderDecoderCache = transformers.DynamicCache
    except ImportError:
        pass
    from DISTT.models.vae.vae_cogvideox import VideoAutoencoderKLCogVideoX

    vae = VideoAutoencoderKLCogVideoX(
        from_pretrained=args.vae_pretrained,
        subfolder=args.vae_subfolder,
        micro_frame_size=None,
        micro_batch_size=1,
    )
    return vae.to(device=device, dtype=torch.float16).eval()


def save_depth_video_npz(depth_video: torch.Tensor, path: Path, depth_max_m: float = 100.0) -> None:
    depth = depth_video.detach().cpu().clone().clamp(-1.0, 1.0)
    depth.add_(1.0).div_(2.0)
    depth = depth.permute(1, 2, 3, 0).float().mean(dim=3, keepdim=True)
    np.savez(path, depth_video=(depth * float(depth_max_m)).numpy())


def validate_payload(payload: object, latent_path: Path) -> tuple[dict[str, torch.Tensor], int, str]:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dictionary payload in {latent_path}, got {type(payload)!r}")
    token = str(payload.get("token", latent_path.name.removesuffix("_latent.pt")))
    latents: dict[str, torch.Tensor] = {}
    for modality in MODALITIES:
        latent = payload.get(f"{modality}_latent")
        if not isinstance(latent, torch.Tensor):
            raise KeyError(f"Missing {modality}_latent tensor in {latent_path}")
        if tuple(latent.shape[:2]) != (len(VIEW_ORDER), 16) or latent.ndim != 5:
            raise ValueError(f"Unexpected {modality} latent shape {tuple(latent.shape)} in {latent_path}")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError(f"Non-finite {modality} latent in {latent_path}")
        latents[modality] = latent
    video_length = int(payload.get("video_length", (latents["rgb"].shape[2] - 1) * 4 + 1))
    if video_length != 17:
        raise ValueError(f"Expected 17 output frames, got {video_length} in {latent_path}")
    return latents, video_length, token


def frame_path(output_dir: Path, modality: str, token: str, view: str, frame: int) -> Path:
    root_name = "gen_video" if modality == "rgb" else f"gen_{modality}"
    return output_dir / root_name / token / view / f"{frame}.jpg"


def depth_path(output_dir: Path, token: str, view: str) -> Path:
    return output_dir / "gen_depth" / f"{token}_gen0" / f"{token}_{view}.npz"


def completion_path(output_dir: Path, token: str) -> Path:
    return output_dir / "completion" / f"{token}.json"


def clip_is_complete(output_dir: Path, token: str, video_length: int = 17) -> bool:
    marker = completion_path(output_dir, token)
    if not marker.is_file():
        return False
    expected = []
    for view in VIEW_ORDER:
        expected.append(depth_path(output_dir, token, view))
        for frame in range(video_length):
            expected.append(frame_path(output_dir, "rgb", token, view, frame))
            expected.append(frame_path(output_dir, "flow", token, view, frame))
    return all(path.is_file() and path.stat().st_size > 0 for path in expected)


def tensor_frame_to_jpeg(frame: torch.Tensor, path: Path) -> None:
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    save_image([frame.detach().cpu()], temporary, normalize=True, value_range=(-1.0, 1.0))
    temporary.replace(path)


def write_clip(
    output_dir: Path,
    token: str,
    source_path: Path,
    decoded: dict[str, torch.Tensor],
    video_length: int,
) -> dict:
    for modality in MODALITIES:
        video = decoded[modality]
        if tuple(video.shape[:3]) != (len(VIEW_ORDER), 3, video_length):
            raise ValueError(f"Unexpected decoded {modality} shape {tuple(video.shape)} for {token}")
    for view_index, view in enumerate(VIEW_ORDER):
        rgb_video = decoded["rgb"][view_index]
        flow_video = decoded["flow"][view_index]
        depth_video = decoded["depth"][view_index]
        for frame in range(video_length):
            tensor_frame_to_jpeg(rgb_video[:, frame], frame_path(output_dir, "rgb", token, view, frame))
            tensor_frame_to_jpeg(flow_video[:, frame], frame_path(output_dir, "flow", token, view, frame))
        npz_path = depth_path(output_dir, token, view)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_npz = npz_path.with_name(f".{npz_path.stem}.tmp.npz")
        save_depth_video_npz(depth_video, temporary_npz)
        temporary_npz.replace(npz_path)
    marker = completion_path(output_dir, token)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary_marker = marker.with_suffix(".tmp")
    record = {
        "token": token,
        "source_latent": str(source_path),
        "video_length": video_length,
        "views": list(VIEW_ORDER),
    }
    temporary_marker.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_marker.replace(marker)
    return record


def git_summary() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        return f"unavailable: {result.stderr.strip()}\n"
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return f"commit: {result.stdout.strip()}\nstatus:\n{status.stdout}"


def write_run_metadata(args: argparse.Namespace, output_dir: Path, world_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["world_size"] = world_size
    (output_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = os.environ.get("D_WORLD_LAUNCH_COMMAND")
    if command is None:
        command = " ".join(shlex.quote(value) for value in sys.argv)
    (output_dir / "command.sh").write_text(command + "\n", encoding="utf-8")
    (output_dir / "git.txt").write_text(git_summary(), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = distributed_context()
    latent_paths = discover_latents(args.latent_root)
    if args.max_latents > 0:
        latent_paths = latent_paths[: args.max_latents]
    if args.fast_dev_run:
        latent_paths = latent_paths[: max(world_size, 1)]
    if not latent_paths:
        raise RuntimeError(f"No generated latent files found under {args.latent_root}")
    if rank == 0:
        write_run_metadata(args, args.output_dir, world_size)
    if world_size > 1:
        dist.barrier()

    vae = build_teacher_vae(args, device)
    assigned = latent_paths[rank::world_size]
    started = time.perf_counter()
    built = skipped = 0
    for latent_path in assigned:
        expected_token = latent_path.name.removesuffix("_latent.pt")
        if clip_is_complete(args.output_dir, expected_token):
            skipped += 1
            continue
        clip_started = time.perf_counter()
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        latents, video_length, token = validate_payload(payload, latent_path)
        if token != expected_token:
            raise ValueError(f"Token mismatch for {latent_path}: payload={token}, filename={expected_token}")
        decoded = {}
        for modality in MODALITIES:
            with torch.inference_mode():
                decoded[modality] = vae.decode(
                    latents[modality].to(device=device, dtype=torch.float16), num_frames=video_length
                )[:, :, :video_length].cpu()
        record = write_clip(
            args.output_dir,
            token,
            latent_path,
            decoded,
            video_length,
        )
        built += 1
        print(
            json.dumps(
                {
                    "rank": rank,
                    "local_rank": local_rank,
                    "clip_seconds": time.perf_counter() - clip_started,
                    **record,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del payload, latents, decoded
    if world_size > 1:
        dist.barrier()
    elapsed = time.perf_counter() - started
    rank_summary = {
        "rank": rank,
        "world_size": world_size,
        "assigned": len(assigned),
        "built": built,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
    }
    run_dir = args.output_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"rank_{rank:02d}.json").write_text(
        json.dumps(rank_summary, indent=2) + "\n", encoding="utf-8"
    )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        complete = sum(
            clip_is_complete(args.output_dir, path.name.removesuffix("_latent.pt")) for path in latent_paths
        )
        summary = {
            "latent_root": str(args.latent_root),
            "output_dir": str(args.output_dir),
            "expected_unique_tokens": len(latent_paths),
            "complete_tokens": complete,
            "world_size": world_size,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
