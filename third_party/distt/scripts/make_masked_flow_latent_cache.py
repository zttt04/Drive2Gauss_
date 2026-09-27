import argparse
import json
import os
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import sys

sys.path.append(".")

import DISTT.utils.module_contrib  # noqa: F401

try:
    import transformers

    if not hasattr(transformers, "EncoderDecoderCache") and hasattr(transformers, "DynamicCache"):
        transformers.EncoderDecoderCache = transformers.DynamicCache
except Exception:
    pass

import DISTT.models  # noqa: F401
from DISTT.registry import MODELS, build_module


VIEW_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build RGB-D + masked optical-flow latent caches for DiST-T."
    )
    parser.add_argument(
        "--cache",
        action="append",
        nargs=2,
        metavar=("MANIFEST_JSONL", "OUTPUT_DIR"),
        required=True,
        help="RGB-D latent manifest and output cache directory. Can be repeated.",
    )
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--flow-root", required=True)
    parser.add_argument("--vae-pretrained", required=True)
    parser.add_argument("--vae-subfolder", default="vae")
    parser.add_argument("--flow-scale-px", type=float, default=64.0)
    parser.add_argument(
        "--flow-representation",
        choices=("masked_optical_rgb_raft_whitebg", "masked_uv_plus_mask"),
        default="masked_optical_rgb_raft_whitebg",
        help=(
            "Flow input encoded by the VAE. The default matches the six-view "
            "preview videos: mask x optical flow rendered with the RAFT/Middlebury "
            "color wheel, where zero flow is white."
        ),
    )
    parser.add_argument(
        "--use-pixels-per-second",
        action="store_true",
        help="Use optical_flow_per_second when present before rendering/normalizing flow.",
    )
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing-flow", action="store_true")
    parser.add_argument(
        "--no-save-flow-loss-mask",
        action="store_true",
        help=(
            "Do not save the downsampled dynamic-object mask used to ignore "
            "white/static flow areas in flow loss."
        ),
    )
    parser.add_argument(
        "--skip-missing-flow-scenes",
        action="store_true",
        help=(
            "Skip manifest rows whose scene is not present in --flow-root. "
            "This is useful when the RGB-D latent manifest covers more scenes "
            "than the available dynamic-flow export."
        ),
    )
    parser.add_argument("--index-cache-dir", default=None)
    parser.add_argument("--scene-token-limit", type=int, default=None)
    return parser.parse_args()


def init_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to encode flow latents with CogVideoX VAE.")
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def atomic_json_dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w") as file:
        json.dump(obj, file)
    os.replace(tmp_path, path)


def atomic_manifest_dump(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    os.replace(tmp_path, path)


def read_manifest(path):
    rows = []
    with open(path) as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def common_parent(paths):
    return Path(os.path.commonpath([os.fspath(path) for path in paths]))


def load_or_build_scene_tokens(args, cache_rows_by_manifest, index_cache_dir, rank):
    max_scene_index = max(
        int(row["scene_index"])
        for rows in cache_rows_by_manifest.values()
        for row in rows
    )
    if args.scene_token_limit is not None:
        max_scene_index = max(max_scene_index, args.scene_token_limit - 1)
    scene_tokens_path = index_cache_dir / "scene_tokens_train_12hz.json"

    if rank == 0 and not scene_tokens_path.exists():
        data = mmcv.load(args.ann_file)
        scene_tokens = data["scene_tokens"][: max_scene_index + 1]
        atomic_json_dump(
            {
                "ann_file": args.ann_file,
                "max_scene_index": max_scene_index,
                "scene_tokens": scene_tokens,
            },
            scene_tokens_path,
        )
    barrier()

    with open(scene_tokens_path) as file:
        payload = json.load(file)
    if payload["ann_file"] != args.ann_file:
        raise RuntimeError(
            f"Scene-token cache was built from {payload['ann_file']}, "
            f"not {args.ann_file}."
        )
    return payload["scene_tokens"]


def flow_index_cache_name(args):
    safe_root = str(args.flow_root).strip("/").replace("/", "_")
    return f"masked_flow_index_{safe_root}_{args.height}x{args.width}.json"


def load_or_build_flow_index(args, index_cache_dir, rank):
    flow_root = Path(args.flow_root)
    index_path = index_cache_dir / flow_index_cache_name(args)

    if rank == 0 and not index_path.exists():
        scenes = {}
        first_token_to_flow_scene = {}
        scene_dirs = sorted(
            flow_root.glob("*_scene"),
            key=lambda path: int(path.name.split("_")[0]),
        )
        for scene_dir in tqdm(scene_dirs, desc="index flow scenes"):
            flow_scene = int(scene_dir.name.split("_")[0])
            scene_payload = {}
            for view in VIEW_ORDER:
                flow_dir = scene_dir / view / "flow" / "flow_arrays"
                view_payload = {}
                for flow_path in sorted(flow_dir.glob("flow_*.npz")):
                    data = np.load(flow_path)
                    source_token = str(data["source_token"].item())
                    target_token = str(data["target_token"].item())
                    flow_idx = int(flow_path.stem.split("_")[-1])
                    view_payload[source_token] = [flow_idx, target_token]
                    if view == "CAM_FRONT" and flow_idx == 0:
                        first_token_to_flow_scene[source_token] = flow_scene
                scene_payload[view] = view_payload
            scenes[str(flow_scene)] = scene_payload

        atomic_json_dump(
            {
                "flow_root": os.fspath(flow_root),
                "height": args.height,
                "width": args.width,
                "views": list(VIEW_ORDER),
                "first_token_to_flow_scene": first_token_to_flow_scene,
                "scenes": scenes,
            },
            index_path,
        )
    barrier()

    with open(index_path) as file:
        payload = json.load(file)
    if payload["flow_root"] != os.fspath(flow_root):
        raise RuntimeError(
            f"Flow index was built from {payload['flow_root']}, not {flow_root}."
        )
    return payload


def build_scene_to_flow_scene(
    scene_tokens, flow_index, required_scene_indices=None, skip_missing=False
):
    first_token_to_flow_scene = flow_index["first_token_to_flow_scene"]
    scene_to_flow_scene = {}
    missing = []
    if required_scene_indices is None:
        scene_indices = range(len(scene_tokens))
    else:
        scene_indices = sorted(required_scene_indices)

    for scene_index in scene_indices:
        if scene_index < 0 or scene_index >= len(scene_tokens):
            missing.append((scene_index, None))
            continue
        tokens = scene_tokens[scene_index]
        flow_scene = first_token_to_flow_scene.get(tokens[0])
        if flow_scene is None:
            missing.append((scene_index, tokens[0]))
        else:
            scene_to_flow_scene[scene_index] = flow_scene
    if missing and not skip_missing:
        raise RuntimeError(f"Missing flow directories for {len(missing)} scenes: {missing[:8]}")
    return scene_to_flow_scene, missing


def clip_tokens_for_row(row, scene_tokens):
    scene_index = int(row["scene_index"])
    start = int(row["scene_frame_start"])
    video_length = int(row["video_length"])
    tokens = scene_tokens[scene_index][start : start + video_length]
    if len(tokens) != video_length:
        raise RuntimeError(
            f"Manifest row {row.get('dataset_index')} asks for {video_length} "
            f"frames at scene {scene_index}:{start}, but only got {len(tokens)}."
        )
    return tokens


def find_missing_edges(row, clip_tokens, scene_to_flow_scene, flow_index):
    flow_scene = scene_to_flow_scene[int(row["scene_index"])]
    scene_index = flow_index["scenes"][str(flow_scene)]
    missing = []
    for view in VIEW_ORDER:
        view_index = scene_index[view]
        for source_token, target_token in zip(clip_tokens[:-1], clip_tokens[1:]):
            item = view_index.get(source_token)
            if item is None or item[1] != target_token:
                missing.append((view, source_token, target_token))
    return missing


def flow_and_mask_paths(args, flow_scene, view, flow_idx):
    scene_dir = Path(args.flow_root) / f"{flow_scene}_scene"
    return (
        scene_dir / view / "flow" / "flow_arrays" / f"flow_{flow_idx:06d}.npz",
        scene_dir / view / "sam" / "masks" / f"dynamic_object_mask_{flow_idx:06d}.png",
    )


def make_colorwheel():
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0
    colorwheel[col : col + ry, 0] = 255
    colorwheel[col : col + ry, 1] = np.floor(255 * np.arange(ry) / ry)
    col += ry
    colorwheel[col : col + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    colorwheel[col : col + yg, 1] = 255
    col += yg
    colorwheel[col : col + gc, 1] = 255
    colorwheel[col : col + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    col += gc
    colorwheel[col : col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    colorwheel[col : col + cb, 2] = 255
    col += cb
    colorwheel[col : col + bm, 2] = 255
    colorwheel[col : col + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    col += bm
    colorwheel[col : col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    colorwheel[col : col + mr, 0] = 255
    return colorwheel


def flow_to_image_fixed_scale(flow, scale):
    scale = max(float(scale), 1.0e-6)
    u = flow[:, :, 0] / scale
    v = flow[:, :, 1] / scale
    rad = np.sqrt(np.square(u) + np.square(v))
    u = np.clip(u, -1.0, 1.0)
    v = np.clip(v, -1.0, 1.0)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    angle = np.arctan2(-v, -u) / np.pi
    fk = (angle + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    image = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    for channel in range(3):
        color0 = colorwheel[k0, channel] / 255.0
        color1 = colorwheel[k1, channel] / 255.0
        color = (1.0 - f) * color0 + f * color1
        inside = rad <= 1.0
        color[inside] = 1.0 - rad[inside] * (1.0 - color[inside])
        color[~inside] *= 0.75
        image[:, :, channel] = np.floor(255.0 * color)
    return image


def flow_from_array(data, args):
    if args.use_pixels_per_second and "optical_flow_per_second" in data.files:
        return data["optical_flow_per_second"].astype(np.float32)
    if args.use_pixels_per_second and "dt_seconds" in data.files:
        dt_seconds = max(float(data["dt_seconds"]), 1.0e-6)
        return data["optical_flow"].astype(np.float32) / dt_seconds
    return data["optical_flow"].astype(np.float32)


def flow_representation_metadata(args):
    if args.flow_representation == "masked_optical_rgb_raft_whitebg":
        return {
            "name": "masked_optical_rgb_raft_whitebg",
            "channels": ["flow_color_r", "flow_color_g", "flow_color_b"],
            "colorwheel": "RAFT/Middlebury fixed-scale",
            "zero_flow": "white",
            "flow_scale_px": args.flow_scale_px,
            "use_pixels_per_second": args.use_pixels_per_second,
            "last_frame": "zero_flow_white",
        }
    return {
        "name": "masked_uv_plus_mask",
        "channels": ["u_masked_over_scale", "v_masked_over_scale", "mask_minus1_to_1"],
        "flow_scale_px": args.flow_scale_px,
        "use_pixels_per_second": args.use_pixels_per_second,
        "last_frame": "zero_flow_empty_mask",
    }


def load_masked_flow_video(args, row, clip_tokens, scene_to_flow_scene, flow_index):
    flow_scene = scene_to_flow_scene[int(row["scene_index"])]
    scene_index = flow_index["scenes"][str(flow_scene)]
    video_length = int(row["video_length"])
    if args.flow_representation == "masked_optical_rgb_raft_whitebg":
        video = np.ones(
            (len(VIEW_ORDER), 3, video_length, args.height, args.width),
            dtype=np.float32,
        )
    else:
        video = np.empty(
            (len(VIEW_ORDER), 3, video_length, args.height, args.width),
            dtype=np.float32,
        )
        video[:, 0:2, -1] = 0.0
        video[:, 2, -1] = -1.0
    flow_loss_mask = np.zeros(
        (len(VIEW_ORDER), 1, video_length, args.height, args.width),
        dtype=np.float32,
    )

    for view_idx, view in enumerate(VIEW_ORDER):
        view_index = scene_index[view]
        for frame_idx, source_token in enumerate(clip_tokens[:-1]):
            item = view_index.get(source_token)
            if item is None or item[1] != clip_tokens[frame_idx + 1]:
                if not args.allow_missing_flow:
                    raise RuntimeError(
                        f"Missing flow edge for scene={row['scene_index']} "
                        f"start={row['scene_frame_start']} view={view} token={source_token}"
                    )
                video[view_idx, 0:2, frame_idx] = 0.0
                video[view_idx, 2, frame_idx] = -1.0
                if args.flow_representation == "masked_optical_rgb_raft_whitebg":
                    video[view_idx, :, frame_idx] = 1.0
                continue

            flow_path, mask_path = flow_and_mask_paths(args, flow_scene, view, item[0])
            data = np.load(flow_path)
            flow = np.nan_to_num(flow_from_array(data, args))
            valid = data["valid"].astype(bool)
            mask = np.asarray(Image.open(mask_path).convert("L")) > 127
            mask = np.logical_and(mask, valid)

            if flow.shape[:2] != (args.height, args.width):
                raise RuntimeError(f"Unexpected flow shape {flow.shape} in {flow_path}")
            if mask.shape != (args.height, args.width):
                raise RuntimeError(f"Unexpected mask shape {mask.shape} in {mask_path}")

            masked_flow = flow * mask[..., None]
            flow_loss_mask[view_idx, 0, frame_idx] = mask.astype(np.float32)
            if args.flow_representation == "masked_optical_rgb_raft_whitebg":
                flow_rgb = flow_to_image_fixed_scale(masked_flow, args.flow_scale_px)
                video[view_idx, :, frame_idx] = (
                    flow_rgb.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
                )
            else:
                video[view_idx, 0, frame_idx] = np.clip(
                    masked_flow[..., 0] / args.flow_scale_px, -1.0, 1.0
                )
                video[view_idx, 1, frame_idx] = np.clip(
                    masked_flow[..., 1] / args.flow_scale_px, -1.0, 1.0
                )
                video[view_idx, 2, frame_idx] = mask.astype(np.float32) * 2.0 - 1.0
    return video, flow_loss_mask


def downsample_flow_loss_mask(flow_loss_mask, flow_latent):
    mask = torch.from_numpy(flow_loss_mask).float()
    mask = F.interpolate(
        mask,
        size=tuple(flow_latent.shape[2:]),
        mode="trilinear",
        align_corners=False,
    )
    return mask.clamp_(0.0, 1.0).half()


def build_vae(args, device):
    vae_cfg = dict(
        type="VideoAutoencoderKLCogVideoX",
        from_pretrained=args.vae_pretrained,
        subfolder=args.vae_subfolder,
        micro_frame_size=None,
        micro_batch_size=1,
    )
    vae = build_module(vae_cfg, MODELS)
    return vae.to(device, torch.float16).eval()


def output_path_for_row(output_dir, row):
    dataset_index = int(row["dataset_index"])
    token = row["token"]
    shard_dir = Path(output_dir) / "clips" / f"{dataset_index // 1000:04d}"
    return shard_dir / f"{dataset_index:08d}_{token}_rgbd_flow_latent.pt"


def attach_flow_loss_mask(payload, flow_loss_mask):
    flow_latent = payload["flow_latent"]
    payload["flow_loss_mask"] = downsample_flow_loss_mask(flow_loss_mask, flow_latent)
    payload["flow_loss_mask_source"] = (
        "dynamic_object_mask_png_and_flow_valid_downsampled_trilinear"
    )
    payload["flow_loss_mask_shape"] = list(payload["flow_loss_mask"].shape)


def save_rgbd_flow_payload(args, vae, device, row, flow_video, flow_loss_mask, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(row["path"], map_location="cpu")
    latent = payload.get("latent")
    rgb_latent = payload.get("rgb_latent")
    depth_latent = payload.get("depth_latent")
    if rgb_latent is None:
        rgb_latent = latent[:, :16]
    if depth_latent is None:
        depth_latent = latent[:, 16:32]

    flow_video = torch.from_numpy(flow_video).to(device=device, dtype=torch.float16)
    with torch.inference_mode():
        flow_latent = vae.encode(flow_video).detach().cpu().half()

    payload["rgb_latent"] = rgb_latent.half()
    payload["depth_latent"] = depth_latent.half()
    payload["flow_latent"] = flow_latent
    if not args.no_save_flow_loss_mask:
        attach_flow_loss_mask(payload, flow_loss_mask)
    payload["latent"] = torch.cat(
        [payload["rgb_latent"], payload["depth_latent"], flow_latent], dim=1
    )
    payload["latent_mode"] = "rgb_depth_masked_flow_sequence"
    payload["source_rgbd_latent_path"] = row["path"]
    payload["flow_representation"] = flow_representation_metadata(args)

    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)
    return payload["latent"].shape, flow_latent.shape


def update_flow_loss_mask_payload(flow_loss_mask, output_path):
    payload = torch.load(output_path, map_location="cpu")
    attach_flow_loss_mask(payload, flow_loss_mask)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)
    return payload["latent"].shape, payload["flow_latent"].shape


def process_cache(args, vae, device, rows, output_dir, scene_tokens, scene_to_flow_scene, flow_index, rank, world_size):
    output_dir = Path(output_dir)
    rank_manifest = output_dir / "manifests" / f"rank{rank:02d}-of-{world_size:02d}.jsonl"

    eligible = []
    missing_examples = []
    missing_scene_examples = []
    missing_scene_rows = 0
    for row in rows:
        if int(row["scene_index"]) not in scene_to_flow_scene:
            if not args.skip_missing_flow_scenes:
                raise RuntimeError(
                    f"Missing flow scene for scene_index={row['scene_index']} "
                    f"dataset_index={row.get('dataset_index')}"
                )
            missing_scene_rows += 1
            if len(missing_scene_examples) < 8:
                missing_scene_examples.append(
                    {
                        "dataset_index": row.get("dataset_index"),
                        "scene_index": row["scene_index"],
                        "scene_frame_start": row.get("scene_frame_start"),
                    }
                )
            continue

        clip_tokens = clip_tokens_for_row(row, scene_tokens)
        missing = find_missing_edges(row, clip_tokens, scene_to_flow_scene, flow_index)
        if missing and not args.allow_missing_flow:
            if len(missing_examples) < 8:
                missing_examples.append(
                    {
                        "dataset_index": row["dataset_index"],
                        "scene_index": row["scene_index"],
                        "scene_frame_start": row["scene_frame_start"],
                        "missing_edges": len(missing),
                        "first_missing": missing[0],
                    }
                )
            continue
        eligible.append((row, clip_tokens, len(missing)))

    if args.limit is not None:
        eligible = eligible[: args.limit]

    assigned = [
        item for ordinal, item in enumerate(eligible) if ordinal % world_size == rank
    ]
    iterator = tqdm(
        assigned,
        desc=f"rank {rank} {output_dir.name}",
        disable=rank != 0,
    )

    manifest_rows = []
    for row, clip_tokens, missing_count in iterator:
        output_path = output_path_for_row(output_dir, row)
        if output_path.exists() and not args.overwrite:
            payload = torch.load(output_path, map_location="cpu")
            latent_shape = payload["latent"].shape
            flow_shape = payload["flow_latent"].shape
            if not args.no_save_flow_loss_mask and "flow_loss_mask" not in payload:
                _, flow_loss_mask = load_masked_flow_video(
                    args, row, clip_tokens, scene_to_flow_scene, flow_index
                )
                latent_shape, flow_shape = update_flow_loss_mask_payload(
                    flow_loss_mask, output_path
                )
        else:
            flow_video, flow_loss_mask = load_masked_flow_video(
                args, row, clip_tokens, scene_to_flow_scene, flow_index
            )
            latent_shape, flow_shape = save_rgbd_flow_payload(
                args, vae, device, row, flow_video, flow_loss_mask, output_path
            )

        new_row = dict(row)
        new_row["source_rgbd_latent_path"] = row["path"]
        new_row["path"] = os.fspath(output_path)
        new_row["shape_latent"] = list(latent_shape)
        new_row["flow_latent_shape"] = list(flow_shape)
        new_row["latent_mode"] = "rgb_depth_masked_flow_sequence"
        new_row["flow_representation"] = args.flow_representation
        new_row["flow_scale_px"] = args.flow_scale_px
        new_row["flow_loss_mask"] = not args.no_save_flow_loss_mask
        new_row["use_pixels_per_second"] = args.use_pixels_per_second
        new_row["flow_missing_edges"] = missing_count
        manifest_rows.append(new_row)

    atomic_manifest_dump(manifest_rows, rank_manifest)
    stats = {
        "input_rows": len(rows),
        "eligible_rows": len(eligible),
        "assigned_rows": len(assigned),
        "missing_flow_scene_rows": missing_scene_rows,
        "missing_flow_scene_examples": missing_scene_examples,
        "missing_examples": missing_examples,
    }
    atomic_json_dump(stats, output_dir / "manifests" / f"rank{rank:02d}-stats.json")
    barrier()

    if rank == 0:
        merged = []
        for manifest_path in sorted((output_dir / "manifests").glob("rank*-of-*.jsonl")):
            merged.extend(read_manifest(manifest_path))
        merged.sort(key=lambda item: (int(item["video_length"]), int(item["dataset_index"])))
        atomic_manifest_dump(merged, output_dir / "manifest_train.jsonl")
        atomic_json_dump(
            {
                "input_rows": len(rows),
                "eligible_rows": len(eligible),
                "output_rows": len(merged),
                "allow_missing_flow": args.allow_missing_flow,
                "skip_missing_flow_scenes": args.skip_missing_flow_scenes,
                "missing_flow_scene_rows": missing_scene_rows,
                "missing_flow_scene_examples": missing_scene_examples,
                "flow_representation": args.flow_representation,
                "flow_scale_px": args.flow_scale_px,
                "flow_loss_mask": not args.no_save_flow_loss_mask,
                "use_pixels_per_second": args.use_pixels_per_second,
                "missing_examples": missing_examples,
            },
            output_dir / "summary.json",
        )
    barrier()


def main():
    args = parse_args()
    rank, world_size, device = init_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True

    cache_specs = [(Path(manifest), Path(output_dir)) for manifest, output_dir in args.cache]
    cache_rows = {manifest: read_manifest(manifest) for manifest, _ in cache_specs}
    if args.index_cache_dir is None:
        index_cache_dir = common_parent([output_dir for _, output_dir in cache_specs]) / "_masked_flow_cache_index"
    else:
        index_cache_dir = Path(args.index_cache_dir)
    index_cache_dir.mkdir(parents=True, exist_ok=True)

    scene_tokens = load_or_build_scene_tokens(args, cache_rows, index_cache_dir, rank)
    flow_index = load_or_build_flow_index(args, index_cache_dir, rank)
    required_scene_indices = {
        int(row["scene_index"])
        for rows in cache_rows.values()
        for row in rows
    }
    scene_to_flow_scene, missing_flow_scenes = build_scene_to_flow_scene(
        scene_tokens,
        flow_index,
        required_scene_indices=required_scene_indices,
        skip_missing=args.skip_missing_flow_scenes,
    )
    if rank == 0 and missing_flow_scenes and args.skip_missing_flow_scenes:
        print(
            "Skipping "
            f"{len(missing_flow_scenes)} manifest scenes absent from flow root; "
            f"examples={missing_flow_scenes[:8]}"
        )
    vae = build_vae(args, device)

    for manifest, output_dir in cache_specs:
        process_cache(
            args,
            vae,
            device,
            cache_rows[manifest],
            output_dir,
            scene_tokens,
            scene_to_flow_scene,
            flow_index,
            rank,
            world_size,
        )


if __name__ == "__main__":
    main()
