import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    nuscenes_root = Path(os.environ.get("NUSCENES_ROOT", "data/nuscenes"))
    parser = argparse.ArgumentParser(description="Run nuScenes flow and SAM masks for a shard of RDepth scenes.")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--scene-limit", type=int, default=300)
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--scene-list-path", default=None)
    parser.add_argument(
        "--train-info-path",
        default=str(nuscenes_root / "nuscenes_mmdet3d-12Hz/nuscenes_advanced_12Hz_infos_train.pkl"),
    )
    parser.add_argument(
        "--val-info-path",
        default=str(nuscenes_root / "nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl"),
    )
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--target-height", type=int, default=424)
    parser.add_argument("--target-width", type=int, default=800)
    parser.add_argument("--scene-source", choices=["rdepth", "nuscenes"], default="rdepth")
    parser.add_argument("--rgb-source", choices=["rdepth", "nuscenes"], default="rdepth")
    parser.add_argument("--metadata-version", default="advanced_12Hz_trainval")
    parser.add_argument("--optical-only", action="store_true")
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--skip-flow", action="store_true")
    parser.add_argument("--skip-sam", action="store_true")
    parser.add_argument("--skip-arrays", action="store_true")
    parser.add_argument("--wait-for-flow", action="store_true")
    parser.add_argument("--wait-poll-seconds", type=int, default=60)
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--nuscenes-root", default=str(nuscenes_root))
    parser.add_argument("--rdepth-root", default=os.environ.get("RDEPTH_ROOT", "data/nus_Rdepth"))
    parser.add_argument("--output-root", default="outputs/nuscenes_dynamic_flow")
    parser.add_argument("--torch-home", default=os.environ.get("TORCH_HOME", str(Path.home() / ".cache/torch")))
    parser.add_argument("--flow-python", default=sys.executable)
    parser.add_argument("--sam-python", default=sys.executable)
    parser.add_argument("--status", choices=["ACTIVE", "DRY_RUN"], default="ACTIVE")
    return parser.parse_args()


def scene_sort_key(path):
    match = re.match(r"(\d+)_scene$", path.name)
    if match:
        return int(match.group(1))
    return path.name


def complete_scene(scene_root):
    token_path = scene_root / "sample_token_list.json"
    if not token_path.exists():
        return False
    tokens = json.loads(token_path.read_text())
    if len(tokens) < 2:
        return False
    for camera in CAMERA_ORDER:
        found_pair = False
        for index, token in enumerate(tokens[:-1]):
            next_token = tokens[index + 1]
            source_rgb = scene_root / token / camera / f"rgb_depth_{index}.npz"
            source_depth = scene_root / token / camera / "refined_depth.npz"
            target_rgb = scene_root / next_token / camera / f"rgb_depth_{index + 1}.npz"
            if source_rgb.exists() and source_depth.exists() and target_rgb.exists():
                found_pair = True
                break
        if not found_pair:
            return False
    return True


def train_sample_tokens(info_path):
    data = pickle.load(open(info_path, "rb"))
    return {info["token"] for info in data["infos"]}


def train_scene_keys(info_path):
    data = pickle.load(open(info_path, "rb"))
    if "scene_tokens" in data:
        scene_tokens = set()
        first_sample_tokens = set()
        for item in data["scene_tokens"]:
            if isinstance(item, str):
                scene_tokens.add(item)
            elif item:
                first_sample_tokens.add(item[0])
        return scene_tokens, first_sample_tokens
    return None, None


def selected_rdepth_scenes(rdepth_root, scene_limit, split, train_info_path, val_info_path, scene_list_path):
    if scene_list_path:
        scene_names = json.loads(Path(scene_list_path).read_text())
        return [{"output_name": name, "rdepth_scene": name} for name in scene_names[:scene_limit]]
    scenes = [path for path in sorted(rdepth_root.glob("*_scene"), key=scene_sort_key) if complete_scene(path)]
    if split in ("train", "val"):
        info_path = train_info_path if split == "train" else val_info_path
        tokens = train_sample_tokens(info_path)
        train_scenes = []
        for scene in scenes:
            scene_tokens = json.loads((scene / "sample_token_list.json").read_text())
            if scene_tokens and scene_tokens[0] in tokens:
                train_scenes.append(scene)
        scenes = train_scenes
    return [{"output_name": scene.name, "rdepth_scene": scene.name} for scene in scenes[:scene_limit]]


def selected_nuscenes_scenes(nuscenes_root, metadata_version, scene_limit, split, train_info_path, val_info_path, scene_list_path):
    metadata_root = Path(nuscenes_root) / metadata_version
    scene_rows = json.loads((metadata_root / "scene.json").read_text())
    indexed = [
        {
            "output_name": f"{index}_scene",
            "rdepth_scene": f"{index}_scene",
            "scene_token": row["token"],
            "scene_name": row["name"],
        }
        for index, row in enumerate(scene_rows)
    ]
    if scene_list_path:
        wanted = set(json.loads(Path(scene_list_path).read_text()))
        indexed = [
            scene
            for scene in indexed
            if scene["output_name"] in wanted
            or scene["scene_name"] in wanted
            or scene["scene_token"] in wanted
        ]
        return indexed[:scene_limit]
    if split in ("train", "val"):
        info_path = train_info_path if split == "train" else val_info_path
        scene_tokens, first_sample_tokens = train_scene_keys(info_path)
        if scene_tokens is None:
            raise RuntimeError(f"scene_tokens not found in {info_path}; cannot select {split} scenes")
        indexed = [
            scene
            for scene, row in zip(indexed, scene_rows)
            if scene["scene_token"] in scene_tokens or row["first_sample_token"] in first_sample_tokens
        ]
    return indexed[:scene_limit]


def selected_scenes(args, rdepth_root):
    if args.scene_source == "rdepth":
        return selected_rdepth_scenes(
            rdepth_root,
            args.scene_limit,
            args.split,
            args.train_info_path,
            args.val_info_path,
            args.scene_list_path,
        )
    return selected_nuscenes_scenes(
        args.nuscenes_root,
        args.metadata_version,
        args.scene_limit,
        args.split,
        args.train_info_path,
        args.val_info_path,
        args.scene_list_path,
    )


def run_command(command, env, log_path, cwd, dry_run):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + subprocess.list2cmdline([str(part) for part in command]) + "\n")
        log_file.flush()
        if dry_run:
            return 0
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        log_file.write(f"\nexit_code={result.returncode}\n")
        return result.returncode


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    rdepth_root = Path(args.rdepth_root).resolve()
    output_root = Path(args.output_root).resolve()
    worker_root = output_root / f"worker_{args.worker_id:02d}_gpu_{args.gpu_id}"
    log_root = worker_root / "logs"
    worker_root.mkdir(parents=True, exist_ok=True)

    scenes = selected_scenes(args, rdepth_root)
    shard = scenes[args.worker_id :: args.num_workers]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["TORCH_HOME"] = args.torch_home

    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "gpu_id": args.gpu_id,
        "scene_limit": args.scene_limit,
        "split": args.split,
        "info_path": args.train_info_path if args.split == "train" else args.val_info_path if args.split == "val" else None,
        "scene_list_path": args.scene_list_path,
        "max_pairs": args.max_pairs,
        "target_height": args.target_height,
        "target_width": args.target_width,
        "scene_source": args.scene_source,
        "rgb_source": args.rgb_source,
        "metadata_version": args.metadata_version,
        "optical_only": args.optical_only,
        "mask_only": args.mask_only,
        "skip_flow": args.skip_flow,
        "skip_sam": args.skip_sam,
        "skip_arrays": args.skip_arrays,
        "wait_for_flow": args.wait_for_flow,
        "scene_count": len(shard),
        "scenes": shard,
    }
    (worker_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    dry_run = args.status == "DRY_RUN"
    for scene in shard:
        scene_output_root = output_root / scene["output_name"]
        scene_log_root = log_root / scene["output_name"]
        for camera in CAMERA_ORDER:
            camera_root = scene_output_root / camera
            flow_dir = camera_root / "flow"
            sam_dir = camera_root / "sam"
            flow_done = flow_dir / "summary.json"
            sam_done = sam_dir / "summary.json"

            if not args.skip_flow and not flow_done.exists():
                flow_command = [
                    args.flow_python,
                    "tools/compute_nuscenes_flow.py",
                    "--nuscenes-root",
                    args.nuscenes_root,
                    "--metadata-version",
                    args.metadata_version,
                    "--scene-source",
                    args.scene_source,
                    "--rgb-source",
                    args.rgb_source,
                    "--rdepth-root",
                    str(rdepth_root),
                    "--rdepth-scene",
                    scene["rdepth_scene"],
                    "--camera",
                    camera,
                    "--output-root",
                    str(camera_root),
                    "--experiment-name",
                    "flow",
                    "--target-height",
                    str(args.target_height),
                    "--target-width",
                    str(args.target_width),
                    "--skip-same-filename",
                    "--skip-same-rgb",
                    "--save-pixels-per-second",
                    "--depth-key",
                    "refined_depth",
                    "--skip-videos",
                ]
                if not args.skip_arrays:
                    flow_command.append("--save-arrays")
                    flow_command.append("--skip-visualizations")
                if args.scene_source == "nuscenes":
                    flow_command.extend(["--scene-token", scene["scene_token"], "--scene-name", scene["scene_name"]])
                if args.max_pairs > 0:
                    flow_command.extend(["--max-pairs", str(args.max_pairs)])
                if args.optical_only:
                    flow_command.append("--optical-only")
                code = run_command(flow_command, env, scene_log_root / f"{camera}_flow.log", project_root, dry_run)
                if code != 0:
                    continue
                generated = sorted(camera_root.glob("*_flow"))
                if generated and generated[-1] != flow_dir and not flow_dir.exists() and not dry_run:
                    generated[-1].rename(flow_dir)
                flow_done = flow_dir / "summary.json"

            while args.wait_for_flow and not flow_done.exists() and not args.skip_sam:
                time.sleep(max(1, int(args.wait_poll_seconds)))

            if not args.skip_sam and not sam_done.exists() and flow_done.exists():
                sam_command = [
                    args.sam_python,
                    "tools/segment_dynamic_objects_with_sam.py",
                    "--flow-result-dir",
                    str(flow_dir),
                    "--output-root",
                    str(camera_root),
                    "--experiment-name",
                    "sam",
                    "--device",
                    "cuda",
                    "--skip-video",
                    "--skip-visualizations",
                ]
                sam_command.extend(["--max-frames", str(args.max_pairs)])
                if args.mask_only:
                    sam_command.append("--mask-only")
                code = run_command(sam_command, env, scene_log_root / f"{camera}_sam.log", project_root, dry_run)
                if code == 0:
                    generated = sorted(camera_root.glob("*_sam"))
                    if generated and generated[-1] != sam_dir and not sam_dir.exists() and not dry_run:
                        generated[-1].rename(sam_dir)
                    sam_done = sam_dir / "summary.json"


if __name__ == "__main__":
    main()
