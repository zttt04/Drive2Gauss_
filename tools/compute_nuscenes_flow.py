import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


def encode_flow_png(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Encode flow as uint16 PNG channels ``[valid, V, U]`` for OpenCV.

    The logical representation is ``U = round(flow_x * 64 + 32768)`` and
    ``V = round(flow_y * 64 + 32768)``. OpenCV writes channel 0 first in BGR
    order, so the returned array keeps the on-disk channel order explicit.
    """
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow with shape [H, W, 2], got {flow.shape}")
    if valid.shape != flow.shape[:2]:
        raise ValueError(f"Expected valid mask shape {flow.shape[:2]}, got {valid.shape}")
    encoded_u = np.clip(np.rint(flow[..., 0] * 64.0 + 32768.0), 0, 65535).astype(np.uint16)
    encoded_v = np.clip(np.rint(flow[..., 1] * 64.0 + 32768.0), 0, 65535).astype(np.uint16)
    encoded_valid = (valid > 0).astype(np.uint16)
    return np.stack([encoded_valid, encoded_v, encoded_u], axis=-1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SEA-RAFT optical flow and camera-depth rigid flow for nuScenes RDepth frames."
    )
    parser.add_argument("--nuscenes-root", default=os.environ.get("NUSCENES_ROOT", "data/nuscenes"))
    parser.add_argument("--metadata-version", default="advanced_12Hz_trainval")
    parser.add_argument("--scene-source", choices=["rdepth", "nuscenes"], default="rdepth")
    parser.add_argument("--rgb-source", choices=["rdepth", "nuscenes"], default="rdepth")
    parser.add_argument("--rdepth-root", default=os.environ.get("RDEPTH_ROOT", "data/nus_Rdepth"))
    parser.add_argument("--rdepth-scene", default="113_scene")
    parser.add_argument("--scene-token", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--camera", default="CAM_FRONT", choices=CAMERA_ORDER)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-name", default="nuscenes_rgbd_flow_424x800")
    parser.add_argument("--target-height", type=int, default=424)
    parser.add_argument("--target-width", type=int, default=800)
    parser.add_argument("--depth-key", default="refined_depth", choices=["refined_depth", "raw_depth", "mvs_depth"])
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=200.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--skip-same-filename", action="store_true")
    parser.add_argument("--skip-same-rgb", action="store_true")
    parser.add_argument("--save-pixels-per-second", action="store_true")
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--optical-only", action="store_true", help="Skip depth, rigid flow, and residual flow computation.")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--sea-raft-dir", default="tools/SEA-RAFT")
    parser.add_argument("--cfg", default="tools/SEA-RAFT/config/eval/kitti-M.json")
    parser.add_argument(
        "--checkpoint",
        default="tools/SEA-RAFT/models/Tartan-C-T-TSKH-kitti432x960-M.pth",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def resolve_path(path):
    return Path(path).expanduser().resolve()


def make_output_dir(output_root, experiment_name):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = resolve_path(output_root) / f"{timestamp}_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def quote_command_part(value):
    return subprocess.list2cmdline([str(value)])


def write_command_file(output_dir):
    executable = sys.executable or "python"
    command_parts = [executable] + sys.argv
    command = " ".join(quote_command_part(part) for part in command_parts)
    lines = [f"cd {quote_command_part(Path.cwd())}"]
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        lines.append(f"export TORCH_HOME={quote_command_part(torch_home)}")
    lines.append(command)
    (output_dir / "command.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_summary(repo_dir):
    lines = []
    for label, command in [
        ("project_head", ["git", "rev-parse", "HEAD"]),
        ("project_status", ["git", "status", "--short"]),
    ]:
        result = subprocess.run(command, cwd=repo_dir, text=True, capture_output=True, check=False)
        if result.returncode == 0:
            lines.append(f"{label}:\n{result.stdout.strip() or '(empty)'}")
        else:
            lines.append(f"{label}:\n{result.stderr.strip() or '(unavailable)'}")
    return "\n\n".join(lines) + "\n"


def write_config(output_dir, args):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_sea_raft(sea_raft_dir, cfg_path, checkpoint_path, device_name):
    sys.path.insert(0, str(sea_raft_dir))
    sys.path.insert(0, str(sea_raft_dir / "core"))

    from config.parser import json_to_args
    from custom import calc_flow
    from raft import RAFT
    from utils.flow_viz import flow_to_image
    from utils.utils import load_ckpt

    args = json_to_args(str(cfg_path))
    model = RAFT(args)
    load_ckpt(model, str(checkpoint_path))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return args, model, device, calc_flow, flow_to_image


def frame_to_tensor(frame_bgr, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1)[None].to(device)


def quaternion_to_rotation(quaternion):
    w, x, y, z = [float(value) for value in quaternion]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError(f"Invalid quaternion: {quaternion}")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_matrix(rotation_quaternion, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation(rotation_quaternion)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def camera_to_global(sample_data_row, ego_poses, calibrated_sensors):
    ego_pose = ego_poses[sample_data_row["ego_pose_token"]]
    calibrated_sensor = calibrated_sensors[sample_data_row["calibrated_sensor_token"]]
    global_from_ego = transform_matrix(ego_pose["rotation"], ego_pose["translation"])
    ego_from_camera = transform_matrix(calibrated_sensor["rotation"], calibrated_sensor["translation"])
    return global_from_ego @ ego_from_camera


def scaled_intrinsics(intrinsics, source_hw, target_hw):
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scale_x = float(target_w) / float(source_w)
    scale_y = float(target_h) / float(source_h)
    k = np.asarray(intrinsics, dtype=np.float64).copy()
    k[0, 0] *= scale_x
    k[0, 2] *= scale_x
    k[1, 1] *= scale_y
    k[1, 2] *= scale_y
    return k


def load_metadata(nuscenes_root, metadata_version, sample_tokens, camera):
    metadata_root = nuscenes_root / metadata_version
    sample_data_rows = json.loads((metadata_root / "sample_data.json").read_text())
    ego_pose_rows = json.loads((metadata_root / "ego_pose.json").read_text())
    calibrated_sensor_rows = json.loads((metadata_root / "calibrated_sensor.json").read_text())

    wanted = set(sample_tokens)
    sample_data = {
        row["sample_token"]: row
        for row in sample_data_rows
        if row.get("sample_token") in wanted and row.get("channel") == camera
    }
    missing = [token for token in sample_tokens if token not in sample_data]
    if missing:
        raise RuntimeError(f"Missing sample_data rows for {camera}: {missing[:5]}")

    ego_poses = {row["token"]: row for row in ego_pose_rows}
    calibrated_sensors = {row["token"]: row for row in calibrated_sensor_rows}
    return sample_data, ego_poses, calibrated_sensors


def load_scene_tokens(scene_root):
    token_list_path = scene_root / "sample_token_list.json"
    if not token_list_path.exists():
        raise FileNotFoundError(token_list_path)
    return json.loads(token_list_path.read_text())


def load_nuscenes_scene_tokens(nuscenes_root, metadata_version, scene_token=None, scene_name=None):
    metadata_root = nuscenes_root / metadata_version
    scene_rows = json.loads((metadata_root / "scene.json").read_text())
    if scene_token:
        matches = [row for row in scene_rows if row["token"] == scene_token]
    elif scene_name:
        matches = [row for row in scene_rows if row["name"] == scene_name]
    else:
        raise ValueError("--scene-token or --scene-name is required when --scene-source=nuscenes")
    if not matches:
        label = scene_token or scene_name
        raise RuntimeError(f"nuScenes scene not found: {label}")

    selected_scene_token = matches[0]["token"]
    sample_rows = json.loads((metadata_root / "sample.json").read_text())
    scene_samples = [row for row in sample_rows if row.get("scene_token") == selected_scene_token]
    if len(scene_samples) < 2:
        raise RuntimeError(f"nuScenes scene has fewer than two samples: {matches[0]['name']}")
    scene_samples.sort(key=lambda row: row["timestamp"])
    return [row["token"] for row in scene_samples]


def depth_npz_path(scene_root, token, camera, token_index):
    return scene_root / token / camera / f"rgb_depth_{token_index}.npz"


def refined_depth_path(scene_root, token, camera):
    return scene_root / token / camera / "refined_depth.npz"


def depth_path(scene_root, token, camera, token_index, depth_key):
    if depth_key == "refined_depth":
        return refined_depth_path(scene_root, token, camera)
    return depth_npz_path(scene_root, token, camera, token_index)


def available_pair_indices(
    scene_root,
    tokens,
    camera,
    stride,
    start_index,
    depth_key,
    rgb_source,
    require_depth,
    sample_data,
    nuscenes_root,
):
    available = []
    for index, token in enumerate(tokens):
        if index < start_index:
            continue
        if rgb_source == "rdepth":
            rgb_path = depth_npz_path(scene_root, token, camera, index)
            if not rgb_path.exists():
                continue
            if require_depth and not depth_path(scene_root, token, camera, index, depth_key).exists():
                continue
        else:
            row = sample_data.get(token)
            if row is None or not (nuscenes_root / row["filename"]).exists():
                continue
        if index + stride < len(tokens):
            available.append(index)
    available_set = set(available)
    return [index for index in available if index + stride in available_set]


def resize_rgb(rgb_hwc, target_hw):
    target_h, target_w = target_hw
    return cv2.resize(rgb_hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def resize_depth(depth_hw, target_hw):
    target_h, target_w = target_hw
    depth = cv2.resize(depth_hw.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return np.where(np.isfinite(depth), np.maximum(depth, 0.0), 0.0).astype(np.float32)


def read_rdepth_rgb(npz_path, target_hw):
    data = np.load(npz_path)
    return resize_rgb(data["rgb"], target_hw)


def read_nuscenes_rgb(nuscenes_root, sample_data_row, target_hw):
    image_path = nuscenes_root / sample_data_row["filename"]
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return resize_rgb(rgb, target_hw)


def read_rgb_frame(rgb_source, scene_root, token, camera, token_index, sample_data_row, nuscenes_root, target_hw):
    if rgb_source == "rdepth":
        return read_rdepth_rgb(depth_npz_path(scene_root, token, camera, token_index), target_hw)
    return read_nuscenes_rgb(nuscenes_root, sample_data_row, target_hw)


def read_depth(scene_root, token, camera, token_index, depth_key, target_hw):
    npz_path = depth_path(scene_root, token, camera, token_index, depth_key)
    data = np.load(npz_path)
    if depth_key == "refined_depth":
        if "depth_pred" not in data.files:
            raise KeyError(f"depth_pred not found in {npz_path}; keys={data.files}")
        return resize_depth(data["depth_pred"], target_hw)
    if depth_key not in data.files:
        raise KeyError(f"{depth_key} not found in {npz_path}; keys={data.files}")
    return resize_depth(data[depth_key], target_hw)


def compute_rigid_flow(depth, source_k, target_k, source_camera_to_global, target_camera_to_global, min_depth, max_depth):
    height, width = depth.shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
    valid_depth = (depth > float(min_depth)) & (depth < float(max_depth)) & np.isfinite(depth)

    ones = np.ones_like(xs)
    pixels = np.stack([xs, ys, ones], axis=0).reshape(3, -1)
    source_points = np.linalg.inv(source_k) @ pixels
    source_points *= depth.reshape(1, -1).astype(np.float64)

    source_points_h = np.concatenate([source_points, np.ones((1, source_points.shape[1]), dtype=np.float64)], axis=0)
    target_from_source = np.linalg.inv(target_camera_to_global) @ source_camera_to_global
    target_points = target_from_source @ source_points_h
    target_xyz = target_points[:3]
    target_z = target_xyz[2]

    projected = target_k @ target_xyz
    projected_x = projected[0] / np.maximum(projected[2], 1.0e-8)
    projected_y = projected[1] / np.maximum(projected[2], 1.0e-8)

    flow = np.stack(
        [
            projected_x.reshape(height, width) - xs,
            projected_y.reshape(height, width) - ys,
        ],
        axis=-1,
    ).astype(np.float32)
    valid = (
        valid_depth
        & (target_z.reshape(height, width) > float(min_depth))
        & (projected_x.reshape(height, width) >= 0.0)
        & (projected_x.reshape(height, width) <= float(width - 1))
        & (projected_y.reshape(height, width) >= 0.0)
        & (projected_y.reshape(height, width) <= float(height - 1))
    )
    flow[~valid] = 0.0
    return flow, valid


def depth_to_bgr(depth, valid):
    vis = depth.copy()
    if np.any(valid):
        max_value = np.percentile(vis[valid], 95)
        max_value = max(float(max_value), 1.0)
        vis = np.clip(vis / max_value, 0.0, 1.0)
    else:
        vis.fill(0.0)
    vis = (vis * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def flow_stats(flow, valid=None):
    if valid is not None:
        values = flow[valid]
    else:
        values = flow.reshape(-1, 2)
    if values.size == 0:
        return 0.0, 0.0
    magnitude = np.linalg.norm(values, axis=-1)
    return float(magnitude.mean()), float(magnitude.max())


def frame_interval_seconds(source_row, target_row):
    return float(target_row["timestamp"] - source_row["timestamp"]) / 1.0e6


def ensure_video(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size)


def resolution_dir(prefix, target_hw):
    return f"{prefix}_{int(target_hw[0])}x{int(target_hw[1])}"


def write_frame_label(frame_bgr, label):
    out = frame_bgr.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 360), 30), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    if args.scene_source == "nuscenes" and not args.optical_only:
        raise ValueError("--scene-source=nuscenes is currently supported for --optical-only flow generation.")
    if args.rgb_source == "nuscenes" and not args.optical_only:
        raise ValueError("--rgb-source=nuscenes is currently supported for --optical-only flow generation.")

    project_root = Path.cwd().resolve()
    nuscenes_root = resolve_path(args.nuscenes_root)
    rdepth_root = resolve_path(args.rdepth_root)
    scene_root = rdepth_root / args.rdepth_scene
    sea_raft_dir = resolve_path(args.sea_raft_dir)
    cfg_path = resolve_path(args.cfg)
    checkpoint_path = resolve_path(args.checkpoint)
    target_hw = (int(args.target_height), int(args.target_width))

    output_dir = make_output_dir(args.output_root, args.experiment_name)
    write_config(output_dir, args)
    write_command_file(output_dir)
    (output_dir / "git.txt").write_text(git_summary(project_root), encoding="utf-8")

    arrays_dir = output_dir / "flow_arrays"
    rgb_dir = output_dir / resolution_dir("rgb", target_hw)
    depth_dir = output_dir / resolution_dir("depth", target_hw)
    visual_dir = output_dir / "visualizations"
    directories = [rgb_dir]
    if not args.skip_visualizations:
        if not args.optical_only:
            directories.append(depth_dir)
        directories.append(visual_dir)
    for directory in directories:
        directory.mkdir()
    if args.save_arrays:
        arrays_dir.mkdir()

    if args.scene_source == "rdepth":
        tokens = load_scene_tokens(scene_root)
    else:
        tokens = load_nuscenes_scene_tokens(
            nuscenes_root,
            args.metadata_version,
            scene_token=args.scene_token,
            scene_name=args.scene_name,
        )
    sample_data, ego_poses, calibrated_sensors = load_metadata(nuscenes_root, args.metadata_version, tokens, args.camera)
    pair_indices = available_pair_indices(
        scene_root,
        tokens,
        args.camera,
        args.stride,
        args.start_index,
        args.depth_key,
        args.rgb_source,
        require_depth=not args.optical_only,
        sample_data=sample_data,
        nuscenes_root=nuscenes_root,
    )
    if not pair_indices:
        raise RuntimeError(f"No valid adjacent {args.rgb_source} RGB pairs found for {args.rdepth_scene} {args.camera}")

    sea_args, model, device, calc_flow, flow_to_image = load_sea_raft(
        sea_raft_dir, cfg_path, checkpoint_path, args.device
    )

    fps = 12.0
    width, height = target_hw[1], target_hw[0]
    write_videos = not args.skip_videos
    write_visualizations = not args.skip_visualizations
    need_flow_visuals = write_videos or write_visualizations
    optical_video = rigid_video = residual_video = comparison_video = None
    if write_videos:
        optical_video = ensure_video(output_dir / "optical_flow.mp4", fps, (width, height))
        if not args.optical_only:
            rigid_video = ensure_video(output_dir / "rigid_flow.mp4", fps, (width, height))
            residual_video = ensure_video(output_dir / "residual_flow.mp4", fps, (width, height))
            comparison_video = ensure_video(
                output_dir / "comparison_rgb_depth_optical_rigid_residual.mp4",
                fps,
                (width * 2, height * 3),
            )

    metrics_path = output_dir / "metrics.csv"
    processed = 0
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "pair_index",
                "source_token",
                "target_token",
                "source_filename",
                "target_filename",
                "dt_seconds",
                "height",
                "width",
                "valid_ratio",
                "optical_flow_mean",
                "optical_flow_max",
                "rigid_flow_mean",
                "rigid_flow_max",
                "residual_flow_mean",
                "residual_flow_max",
                "cuda_max_memory_mb",
            ],
        )
        writer.writeheader()

        with torch.no_grad():
            for pair_index in pair_indices:
                source_token = tokens[pair_index]
                target_token = tokens[pair_index + args.stride]
                source_row = sample_data[source_token]
                target_row = sample_data[target_token]
                source_rgb = read_rgb_frame(
                    args.rgb_source,
                    scene_root,
                    source_token,
                    args.camera,
                    pair_index,
                    source_row,
                    nuscenes_root,
                    target_hw,
                )
                if args.skip_same_filename and source_row["filename"] == target_row["filename"]:
                    continue
                target_rgb = read_rgb_frame(
                    args.rgb_source,
                    scene_root,
                    target_token,
                    args.camera,
                    pair_index + args.stride,
                    target_row,
                    nuscenes_root,
                    target_hw,
                )
                if args.skip_same_rgb and np.array_equal(source_rgb, target_rgb):
                    continue
                source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
                target_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
                dt_seconds = frame_interval_seconds(source_row, target_row)
                if dt_seconds <= 0.0:
                    continue

                image1 = frame_to_tensor(source_bgr, device)
                image2 = frame_to_tensor(target_bgr, device)
                optical_flow_tensor, _ = calc_flow(sea_args, model, image1, image2)
                optical_flow = optical_flow_tensor[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

                optical_flow_per_second = optical_flow / dt_seconds
                if args.optical_only:
                    valid = np.ones(optical_flow.shape[:2], dtype=bool)
                    rigid_flow = residual_flow = None
                    rigid_flow_per_second = residual_flow_per_second = None
                else:
                    source_depth = read_depth(
                        scene_root, source_token, args.camera, pair_index, args.depth_key, target_hw
                    )
                    source_calibrated = calibrated_sensors[source_row["calibrated_sensor_token"]]
                    target_calibrated = calibrated_sensors[target_row["calibrated_sensor_token"]]
                    source_k = scaled_intrinsics(
                        source_calibrated["camera_intrinsic"],
                        source_hw=(int(source_row["height"]), int(source_row["width"])),
                        target_hw=target_hw,
                    )
                    target_k = scaled_intrinsics(
                        target_calibrated["camera_intrinsic"],
                        source_hw=(int(target_row["height"]), int(target_row["width"])),
                        target_hw=target_hw,
                    )
                    source_camera_to_global = camera_to_global(source_row, ego_poses, calibrated_sensors)
                    target_camera_to_global = camera_to_global(target_row, ego_poses, calibrated_sensors)
                    rigid_flow, valid = compute_rigid_flow(
                        source_depth,
                        source_k,
                        target_k,
                        source_camera_to_global,
                        target_camera_to_global,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                    )
                    residual_flow = optical_flow - rigid_flow
                    rigid_flow_per_second = rigid_flow / dt_seconds
                    residual_flow_per_second = residual_flow / dt_seconds

                if need_flow_visuals:
                    optical_vis = flow_to_image(optical_flow, convert_to_bgr=True)

                    if write_videos:
                        optical_video.write(optical_vis)
                        if not args.optical_only:
                            rigid_vis = flow_to_image(rigid_flow, convert_to_bgr=True)
                            residual_vis = flow_to_image(residual_flow, convert_to_bgr=True)
                            depth_vis = depth_to_bgr(source_depth, valid)
                            rigid_video.write(rigid_vis)
                            residual_video.write(residual_vis)

                            panel = cv2.vconcat(
                                [
                                    cv2.hconcat(
                                        [
                                            write_frame_label(source_bgr, f"RGB {pair_index}"),
                                            write_frame_label(depth_vis, f"Depth {args.depth_key}"),
                                        ]
                                    ),
                                    cv2.hconcat(
                                        [
                                            write_frame_label(optical_vis, "SEA-RAFT optical flow"),
                                            write_frame_label(rigid_vis, "Camera+depth rigid flow"),
                                        ]
                                    ),
                                    cv2.hconcat(
                                        [
                                            write_frame_label(residual_vis, "Residual optical - rigid"),
                                            write_frame_label(target_bgr, f"Target RGB {pair_index + args.stride}"),
                                        ]
                                    ),
                                ]
                            )
                            comparison_video.write(panel)

                cv2.imwrite(str(rgb_dir / f"rgb_{processed:06d}.jpg"), source_bgr)
                if write_visualizations:
                    cv2.imwrite(str(visual_dir / f"optical_{processed:06d}.jpg"), optical_vis)
                    if not args.optical_only:
                        cv2.imwrite(str(depth_dir / f"depth_{processed:06d}.jpg"), depth_vis)
                        cv2.imwrite(str(visual_dir / f"rigid_{processed:06d}.jpg"), rigid_vis)
                        cv2.imwrite(str(visual_dir / f"residual_{processed:06d}.jpg"), residual_vis)
                    if write_videos and not args.optical_only:
                        cv2.imwrite(str(visual_dir / f"comparison_{processed:06d}.jpg"), panel)

                if args.save_arrays:
                    saved_arrays = {
                        "optical_flow": optical_flow,
                        "valid": valid.astype(np.uint8),
                        "dt_seconds": np.float32(dt_seconds),
                        "source_token": source_token,
                        "target_token": target_token,
                        "source_filename": source_row["filename"],
                        "target_filename": target_row["filename"],
                    }
                    if not args.optical_only:
                        saved_arrays.update(
                            {
                                "rigid_flow": rigid_flow,
                                "residual_flow": residual_flow,
                                "depth": source_depth.astype(np.float32),
                            }
                        )
                    if args.save_pixels_per_second:
                        saved_arrays["optical_flow_per_second"] = optical_flow_per_second.astype(np.float32)
                        if not args.optical_only:
                            saved_arrays.update(
                                {
                                    "rigid_flow_per_second": rigid_flow_per_second.astype(np.float32),
                                    "residual_flow_per_second": residual_flow_per_second.astype(np.float32),
                                }
                            )
                    np.savez_compressed(arrays_dir / f"flow_{processed:06d}.npz", **saved_arrays)

                optical_mean, optical_max = flow_stats(optical_flow)
                if args.optical_only:
                    rigid_mean = rigid_max = residual_mean = residual_max = 0.0
                else:
                    rigid_mean, rigid_max = flow_stats(rigid_flow, valid)
                    residual_mean, residual_max = flow_stats(residual_flow, valid)
                writer.writerow(
                    {
                        "pair_index": pair_index,
                        "source_token": source_token,
                        "target_token": target_token,
                        "source_filename": source_row["filename"],
                        "target_filename": target_row["filename"],
                        "dt_seconds": f"{dt_seconds:.9f}",
                        "height": height,
                        "width": width,
                        "valid_ratio": f"{float(valid.mean()):.6f}",
                        "optical_flow_mean": f"{optical_mean:.6f}",
                        "optical_flow_max": f"{optical_max:.6f}",
                        "rigid_flow_mean": f"{rigid_mean:.6f}",
                        "rigid_flow_max": f"{rigid_max:.6f}",
                        "residual_flow_mean": f"{residual_mean:.6f}",
                        "residual_flow_max": f"{residual_max:.6f}",
                        "cuda_max_memory_mb": f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}"
                        if torch.cuda.is_available()
                        else "0.0",
                    }
                )
                metrics_file.flush()
                processed += 1
                if args.max_pairs is not None and processed >= int(args.max_pairs):
                    break

    if write_videos:
        optical_video.release()
        if not args.optical_only:
            rigid_video.release()
            residual_video.release()
            comparison_video.release()

    summary = {
        "output_dir": str(output_dir),
        "processed_pairs": processed,
        "target_hw": list(target_hw),
        "camera": args.camera,
        "scene_source": args.scene_source,
        "rgb_source": args.rgb_source,
        "rdepth_scene": args.rdepth_scene,
        "scene_token": args.scene_token,
        "scene_name": args.scene_name,
        "depth_key": args.depth_key,
        "pair_indices": pair_indices,
        "device": str(device),
        "cfg": str(cfg_path),
        "checkpoint": str(checkpoint_path),
        "optical_only": bool(args.optical_only),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
