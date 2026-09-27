import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


DYNAMIC_CATEGORY_PREFIXES = (
    "vehicle.",
    "human.pedestrian.",
)
DYNAMIC_CATEGORY_NAMES = {
    "animal",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter SAM dynamic-class masks with moving nuScenes 3D box projections."
    )
    parser.add_argument("--flow-result-dir", required=True)
    parser.add_argument("--sam-mask-result-dir", required=True)
    parser.add_argument("--nuscenes-root", default=os.environ.get("NUSCENES_ROOT", "data/nuscenes"))
    parser.add_argument("--metadata-version", default="advanced_12Hz_trainval")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-name", default="nuscenes_velocity_filtered_dynamic_mask")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--target-height", type=int, default=424)
    parser.add_argument("--target-width", type=int, default=800)
    parser.add_argument("--min-speed-mps", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=100)
    return parser.parse_args()


def resolve_path(path):
    return Path(path).expanduser().resolve()


def quote_command_part(value):
    return subprocess.list2cmdline([str(value)])


def make_output_dir(output_root, experiment_name):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = resolve_path(output_root) / f"{timestamp}_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_command_file(output_dir):
    command = " ".join(quote_command_part(part) for part in [sys.executable or "python"] + sys.argv)
    lines = [f"cd {quote_command_part(Path.cwd())}", command]
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


def box_corners_global(annotation):
    width, length, height = [float(value) for value in annotation["size"]]
    x = length / 2.0
    y = width / 2.0
    z = height / 2.0
    local = np.array(
        [
            [x, y, z],
            [x, -y, z],
            [-x, -y, z],
            [-x, y, z],
            [x, y, -z],
            [x, -y, -z],
            [-x, -y, -z],
            [-x, y, -z],
        ],
        dtype=np.float64,
    )
    rotation = quaternion_to_rotation(annotation["rotation"])
    center = np.asarray(annotation["translation"], dtype=np.float64)
    return (rotation @ local.T).T + center


def project_points(points_global, global_from_camera, intrinsics, target_hw):
    height, width = target_hw
    camera_from_global = np.linalg.inv(global_from_camera)
    points_h = np.concatenate([points_global, np.ones((points_global.shape[0], 1), dtype=np.float64)], axis=1)
    points_camera = (camera_from_global @ points_h.T).T[:, :3]
    in_front = points_camera[:, 2] > 1.0e-3
    if np.count_nonzero(in_front) < 3:
        return None
    points_camera = points_camera[in_front]
    projected = (intrinsics @ points_camera.T).T
    xy = projected[:, :2] / np.maximum(projected[:, 2:3], 1.0e-8)
    if np.all((xy[:, 0] < 0) | (xy[:, 0] >= width) | (xy[:, 1] < 0) | (xy[:, 1] >= height)):
        return None
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    return xy.astype(np.int32)


def is_dynamic_category(category_name):
    if category_name in DYNAMIC_CATEGORY_NAMES:
        return True
    return any(category_name.startswith(prefix) for prefix in DYNAMIC_CATEGORY_PREFIXES)


def annotation_speed(annotation, annotations_by_token, samples_by_token):
    candidates = []
    current_center = np.asarray(annotation["translation"], dtype=np.float64)
    current_time = float(samples_by_token[annotation["sample_token"]]["timestamp"]) / 1.0e6
    for neighbor_key in ["prev", "next"]:
        neighbor_token = annotation.get(neighbor_key) or ""
        if not neighbor_token or neighbor_token not in annotations_by_token:
            continue
        neighbor = annotations_by_token[neighbor_token]
        neighbor_center = np.asarray(neighbor["translation"], dtype=np.float64)
        neighbor_time = float(samples_by_token[neighbor["sample_token"]]["timestamp"]) / 1.0e6
        dt = abs(neighbor_time - current_time)
        if dt > 1.0e-6:
            candidates.append(float(np.linalg.norm(neighbor_center[:2] - current_center[:2]) / dt))
    return max(candidates) if candidates else 0.0


def load_metadata(metadata_root, camera):
    samples = json.loads((metadata_root / "sample.json").read_text())
    sample_data_rows = json.loads((metadata_root / "sample_data.json").read_text())
    ego_pose_rows = json.loads((metadata_root / "ego_pose.json").read_text())
    calibrated_sensor_rows = json.loads((metadata_root / "calibrated_sensor.json").read_text())
    annotation_rows = json.loads((metadata_root / "sample_annotation.json").read_text())
    return {
        "samples_by_token": {row["token"]: row for row in samples},
        "sample_data_by_sample_token": {
            row["sample_token"]: row
            for row in sample_data_rows
            if row.get("channel") == camera
        },
        "ego_poses": {row["token"]: row for row in ego_pose_rows},
        "calibrated_sensors": {row["token"]: row for row in calibrated_sensor_rows},
        "annotations_by_token": {row["token"]: row for row in annotation_rows},
        "annotations_by_sample_token": group_annotations_by_sample(annotation_rows),
    }


def group_annotations_by_sample(annotation_rows):
    grouped = {}
    for annotation in annotation_rows:
        grouped.setdefault(annotation["sample_token"], []).append(annotation)
    return grouped


def read_flow_tokens(flow_array_path):
    data = np.load(flow_array_path)
    return str(data["source_token"]), str(data["source_filename"])


def overlay_mask(image_bgr, mask):
    overlay = image_bgr.copy()
    overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([40, 220, 80], dtype=np.float32)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    return overlay


def main():
    args = parse_args()
    project_root = Path.cwd().resolve()
    flow_result_dir = resolve_path(args.flow_result_dir)
    sam_mask_result_dir = resolve_path(args.sam_mask_result_dir)
    metadata_root = resolve_path(args.nuscenes_root) / args.metadata_version

    flow_summary_path = flow_result_dir / "summary.json"
    flow_summary = json.loads(flow_summary_path.read_text()) if flow_summary_path.exists() else {}
    camera = args.camera or flow_summary.get("camera")
    if not camera:
        raise ValueError("--camera is required when flow summary does not contain a camera")

    output_dir = make_output_dir(args.output_root, args.experiment_name)
    mask_dir = output_dir / "masks"
    vis_dir = output_dir / "visualizations"
    box_mask_dir = output_dir / "box_velocity_masks"
    for directory in [mask_dir, vis_dir, box_mask_dir]:
        directory.mkdir()

    (output_dir / "config.yaml").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    write_command_file(output_dir)
    (output_dir / "git.txt").write_text(git_summary(project_root), encoding="utf-8")

    metadata = load_metadata(metadata_root, camera)
    target_hw = (int(args.target_height), int(args.target_width))

    rgb_paths = sorted((flow_result_dir / "rgb_424x800").glob("rgb_*.jpg"))
    flow_paths = sorted((flow_result_dir / "flow_arrays").glob("flow_*.npz"))
    sam_paths = sorted((sam_mask_result_dir / "masks").glob("dynamic_object_mask_*.png"))
    frame_count = min(len(rgb_paths), len(flow_paths), len(sam_paths))
    if args.max_frames > 0:
        frame_count = min(frame_count, int(args.max_frames))
    if frame_count <= 0:
        raise RuntimeError("No complete RGB, flow, and SAM mask frames found")

    frame_summaries = []
    for index in range(frame_count):
        rgb = cv2.imread(str(rgb_paths[index]))
        sam_mask = cv2.imread(str(sam_paths[index]), cv2.IMREAD_GRAYSCALE)
        if rgb is None or sam_mask is None:
            raise RuntimeError(f"Failed to read frame {index}")
        if sam_mask.shape != rgb.shape[:2]:
            sam_mask = cv2.resize(sam_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        sam_mask = sam_mask > 0

        source_token, _ = read_flow_tokens(flow_paths[index])
        sample_data_row = metadata["sample_data_by_sample_token"][source_token]
        calibrated_sensor = metadata["calibrated_sensors"][sample_data_row["calibrated_sensor_token"]]
        intrinsics = scaled_intrinsics(
            calibrated_sensor["camera_intrinsic"],
            source_hw=(int(sample_data_row["height"]), int(sample_data_row["width"])),
            target_hw=target_hw,
        )
        global_from_camera = camera_to_global(sample_data_row, metadata["ego_poses"], metadata["calibrated_sensors"])

        box_mask = np.zeros(target_hw, dtype=np.uint8)
        moving_annotations = []
        for annotation in metadata["annotations_by_sample_token"].get(source_token, []):
            category_name = annotation.get("category_name", "")
            if not is_dynamic_category(category_name):
                continue
            speed = annotation_speed(annotation, metadata["annotations_by_token"], metadata["samples_by_token"])
            if speed < float(args.min_speed_mps):
                continue
            corners = box_corners_global(annotation)
            projected = project_points(corners, global_from_camera, intrinsics, target_hw)
            if projected is None:
                continue
            hull = cv2.convexHull(projected.reshape(-1, 1, 2))
            cv2.fillConvexPoly(box_mask, hull, 255)
            moving_annotations.append(
                {
                    "token": annotation["token"],
                    "category_name": category_name,
                    "speed_mps": speed,
                }
            )

        if box_mask.shape != rgb.shape[:2]:
            box_mask = cv2.resize(box_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        box_mask_bool = box_mask > 0
        filtered_mask = sam_mask & box_mask_bool

        cv2.imwrite(str(box_mask_dir / f"box_velocity_mask_{index:06d}.png"), box_mask_bool.astype(np.uint8) * 255)
        cv2.imwrite(str(mask_dir / f"dynamic_object_mask_{index:06d}.png"), filtered_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(vis_dir / f"velocity_filtered_mask_overlay_{index:06d}.jpg"), overlay_mask(rgb, filtered_mask))

        frame_summaries.append(
            {
                "frame": index,
                "moving_annotations": len(moving_annotations),
                "sam_mask_ratio": float(sam_mask.mean()),
                "box_velocity_mask_ratio": float(box_mask_bool.mean()),
                "filtered_mask_ratio": float(filtered_mask.mean()),
                "categories": sorted({row["category_name"] for row in moving_annotations}),
            }
        )

    summary = {
        "output_dir": str(output_dir),
        "flow_result_dir": str(flow_result_dir),
        "sam_mask_result_dir": str(sam_mask_result_dir),
        "camera": camera,
        "frame_count": frame_count,
        "min_speed_mps": args.min_speed_mps,
        "mean_sam_mask_ratio": float(np.mean([row["sam_mask_ratio"] for row in frame_summaries])),
        "mean_box_velocity_mask_ratio": float(np.mean([row["box_velocity_mask_ratio"] for row in frame_summaries])),
        "mean_filtered_mask_ratio": float(np.mean([row["filtered_mask_ratio"] for row in frame_summaries])),
        "frame_summaries": frame_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "frame_summaries"}, indent=2))


if __name__ == "__main__":
    main()
