import argparse
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert


def parse_args():
    parser = argparse.ArgumentParser(description="Segment dynamic object classes and mask corrected residual flow.")
    parser.add_argument("--flow-result-dir")
    parser.add_argument(
        "--job-manifest",
        help="Optional JSON manifest with flow_result_dir/output_root jobs. Models are loaded once for all jobs.",
    )
    parser.add_argument(
        "--grounded-sam2-root",
        default=os.environ.get("GROUNDED_SAM2_ROOT", "third_party/Grounded-SAM-2"),
    )
    parser.add_argument(
        "--extra-site-packages",
        action="append",
        default=[],
        help="Append extra Python site-packages paths after the current environment's packages.",
    )
    parser.add_argument("--sea-raft-dir", default="tools/SEA-RAFT")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-name", default="sam_dynamic_object_residual_flow")
    parser.add_argument(
        "--output-mode",
        default="timestamped",
        choices=["timestamped", "fixed"],
        help="Use timestamped experiment dirs or write directly to output_root/experiment_name.",
    )
    parser.add_argument("--text-prompt", default="car. truck. bus. motorcycle. bicycle. person. pedestrian. cyclist.")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--grounding-dino-config", default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounding-dino-checkpoint", default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--amp-dtype",
        default="float16",
        choices=["none", "bfloat16", "float16"],
        help="Use CUDA autocast for GroundingDINO/SAM2 inference. Flash attention requires fp16 or bf16.",
    )
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--tile-width", type=int, default=400)
    parser.add_argument("--tile-height", type=int, default=212)
    parser.add_argument("--residual-magnitude-threshold", type=float, default=1.0)
    parser.add_argument("--mask-only", action="store_true", help="Only save dynamic object masks; do not require residual flow.")
    parser.add_argument("--save-instances", action="store_true", help="Save per-detection SAM masks and metadata.")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def resolve_path(path):
    return Path(path).expanduser().resolve()


def make_output_dir(output_root, experiment_name, output_mode):
    if output_mode == "fixed":
        output_dir = resolve_path(output_root) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = resolve_path(output_root) / f"{timestamp}_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def quote_command_part(value):
    return subprocess.list2cmdline([str(value)])


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


def add_label(frame, label):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_tile(frame, tile_size):
    return cv2.resize(frame, tile_size, interpolation=cv2.INTER_AREA)


def overlay_mask(image_bgr, mask):
    overlay = image_bgr.copy()
    overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([40, 220, 80], dtype=np.float32)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def magnitude_heatmap(flow, mask):
    magnitude = np.linalg.norm(flow, axis=-1)
    if np.any(mask):
        scale = np.percentile(magnitude[mask], 95)
        scale = max(float(scale), 1.0)
    else:
        scale = max(float(np.percentile(magnitude, 95)), 1.0)
    vis = np.clip(magnitude / scale, 0.0, 1.0)
    heatmap = cv2.applyColorMap((vis * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap[~mask] = 0
    return heatmap


def build_models(args, grounded_sam2_root):
    for extra_path in args.extra_site_packages:
        if extra_path and extra_path not in sys.path:
            sys.path.append(extra_path)
    sys.path.insert(0, str(grounded_sam2_root))
    sys.path.insert(0, str(grounded_sam2_root / "grounding_dino"))

    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    original_cwd = Path.cwd()
    os.chdir(grounded_sam2_root)
    try:
        sam2_model = build_sam2(
            args.sam2_model_config,
            str(grounded_sam2_root / args.sam2_checkpoint),
            device=device,
        )
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        grounding_model = load_model(
            model_config_path=str(grounded_sam2_root / args.grounding_dino_config),
            model_checkpoint_path=str(grounded_sam2_root / args.grounding_dino_checkpoint),
            device=device,
        )
    finally:
        os.chdir(original_cwd)
    grounding_model.eval()
    return device, sam2_predictor, grounding_model, load_image, predict


def build_flow_visualizer(sea_raft_dir):
    sys.path.insert(0, str(sea_raft_dir / "core"))
    from utils.flow_viz import flow_to_image

    return flow_to_image


def find_rgb_dir(flow_result_dir):
    candidates = sorted(flow_result_dir.glob("rgb_*x*"))
    if candidates:
        return candidates[0]
    return flow_result_dir / "rgb_424x800"


def to_binary_masks(masks, scores):
    if masks.size == 0:
        return np.empty((0, 0, 0), dtype=bool)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    return np.asarray(masks) > 0.5


def autocast_context(device, amp_dtype):
    if device != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def segment_image(image_path, args, sam2_predictor, grounding_model, load_image, predict, device):
    image_source, image = load_image(str(image_path))
    if torch.is_tensor(image):
        image = image.clone()
    else:
        image = np.array(image, copy=True)

    with torch.inference_mode(), autocast_context(device, args.amp_dtype):
        sam2_predictor.set_image(image_source)
        boxes, _, labels = predict(
            model=grounding_model,
            image=image,
            caption=args.text_prompt.lower().strip(),
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=device,
        )

    height, width = image_source.shape[:2]
    if len(labels) == 0:
        return np.zeros((height, width), dtype=bool), [], np.empty((0, height, width), dtype=bool)

    boxes = boxes * torch.Tensor([width, height, width, height])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    with torch.inference_mode(), autocast_context(device, args.amp_dtype):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    binary_masks = to_binary_masks(masks, scores)
    if binary_masks.size == 0:
        return np.zeros((height, width), dtype=bool), [], np.empty((0, height, width), dtype=bool)
    combined = np.any(binary_masks, axis=0)
    annotations = [
        {"label": str(label), "box": box.tolist(), "score": float(score)}
        for label, box, score in zip(labels, input_boxes, np.asarray(scores).reshape(-1))
    ]
    return combined, annotations, binary_masks


def load_jobs(args):
    if args.job_manifest:
        jobs = json.loads(resolve_path(args.job_manifest).read_text(encoding="utf-8"))
        if not isinstance(jobs, list):
            raise RuntimeError("--job-manifest must contain a JSON list")
        return jobs
    if not args.flow_result_dir:
        raise RuntimeError("--flow-result-dir is required when --job-manifest is not provided")
    return [
        {
            "flow_result_dir": args.flow_result_dir,
            "output_root": args.output_root,
            "experiment_name": args.experiment_name,
        }
    ]


def process_job(args, job, project_root, device, sam2_predictor, grounding_model, load_image, predict):
    flow_result_dir = resolve_path(job["flow_result_dir"])
    output_root = job.get("output_root", args.output_root)
    experiment_name = job.get("experiment_name", args.experiment_name)
    sea_raft_dir = resolve_path(args.sea_raft_dir)
    output_dir = make_output_dir(output_root, experiment_name, args.output_mode)
    if (output_dir / "summary.json").exists() and not args.overwrite_existing:
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        summary["skipped_existing"] = True
        print(json.dumps({key: value for key, value in summary.items() if key != "frame_summaries"}, indent=2))
        return summary

    mask_dir = output_dir / "masks"
    instance_dir = output_dir / "instances"
    vis_dir = output_dir / "visualizations"
    mask_dir.mkdir(exist_ok=True)
    if args.save_instances:
        instance_dir.mkdir(exist_ok=True)
    if not args.skip_visualizations:
        vis_dir.mkdir(exist_ok=True)

    (output_dir / "config.yaml").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    write_command_file(output_dir)
    (output_dir / "git.txt").write_text(git_summary(project_root), encoding="utf-8")

    write_video = not args.skip_video
    write_visualizations = not args.skip_visualizations
    need_flow_visuals = (write_video or write_visualizations) and not args.mask_only
    flow_to_image = build_flow_visualizer(sea_raft_dir) if need_flow_visuals else None

    rgb_paths = sorted(find_rgb_dir(flow_result_dir).glob("rgb_*.jpg"))
    flow_paths = sorted((flow_result_dir / "flow_arrays").glob("flow_*.npz"))
    frame_count = len(rgb_paths) if args.mask_only else min(len(rgb_paths), len(flow_paths))
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    if frame_count <= 0:
        raise RuntimeError("No RGB frames and flow arrays found")

    tile_size = (args.tile_width, args.tile_height)
    video_size = (args.tile_width * (2 if args.mask_only else 4), args.tile_height)
    video_path = output_dir / ("rgb_mask.mp4" if args.mask_only else "rgb_mask_residual_dynamic_flow.mp4")
    writer = None
    if write_video:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, video_size)
    preview = None
    frame_summaries = []

    for index in range(frame_count):
        rgb = cv2.imread(str(rgb_paths[index]))
        if rgb is None:
            raise RuntimeError(f"Failed to read {rgb_paths[index]}")
        if args.mask_only:
            residual_magnitude = np.ones(rgb.shape[:2], dtype=np.float32)
            dynamic_flow_mask = None
        else:
            data = np.load(flow_paths[index])
            corrected_residual_flow = data["optical_flow"].astype(np.float32) - data["rigid_flow"].astype(np.float32)
            residual_magnitude = np.linalg.norm(corrected_residual_flow, axis=-1)

        dynamic_class_mask, annotations, instance_masks = segment_image(
            rgb_paths[index], args, sam2_predictor, grounding_model, load_image, predict, device
        )
        if dynamic_class_mask.shape != rgb.shape[:2]:
            dynamic_class_mask = cv2.resize(
                dynamic_class_mask.astype(np.uint8),
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if not args.mask_only:
            dynamic_flow_mask = dynamic_class_mask & (residual_magnitude >= float(args.residual_magnitude_threshold))
            dynamic_flow = corrected_residual_flow.copy()
            dynamic_flow[~dynamic_flow_mask] = 0.0

        cv2.imwrite(str(mask_dir / f"dynamic_object_mask_{index:06d}.png"), dynamic_class_mask.astype(np.uint8) * 255)
        if args.save_instances:
            np.savez_compressed(
                instance_dir / f"instances_{index:06d}.npz",
                masks=instance_masks.astype(np.uint8),
                boxes=np.asarray([annotation["box"] for annotation in annotations], dtype=np.float32),
                scores=np.asarray([annotation["score"] for annotation in annotations], dtype=np.float32),
                labels=np.asarray([annotation["label"] for annotation in annotations]),
            )
        if args.mask_only and (write_video or write_visualizations):
            mask_overlay = overlay_mask(rgb, dynamic_class_mask)
            if write_visualizations:
                cv2.imwrite(str(vis_dir / f"mask_overlay_{index:06d}.jpg"), mask_overlay)
            if write_video:
                frame = np.hstack(
                    [
                        add_label(resize_tile(rgb, tile_size), f"RGB {index:03d}"),
                        add_label(resize_tile(mask_overlay, tile_size), "SAM dynamic classes"),
                    ]
                )
                writer.write(frame)
                if preview is None:
                    preview = frame
        elif need_flow_visuals:
            residual_vis = flow_to_image(corrected_residual_flow, convert_to_bgr=True)
            dynamic_flow_vis = flow_to_image(dynamic_flow, convert_to_bgr=True)
            dynamic_heatmap = magnitude_heatmap(corrected_residual_flow, dynamic_flow_mask)
            dynamic_flow_vis = cv2.addWeighted(dynamic_flow_vis, 0.75, dynamic_heatmap, 0.25, 0.0)
            mask_overlay = overlay_mask(rgb, dynamic_class_mask)

            if write_visualizations:
                cv2.imwrite(str(vis_dir / f"mask_overlay_{index:06d}.jpg"), mask_overlay)
                cv2.imwrite(str(vis_dir / f"dynamic_flow_{index:06d}.jpg"), dynamic_flow_vis)

            if write_video:
                frame = np.hstack(
                    [
                        add_label(resize_tile(rgb, tile_size), f"RGB {index:03d}"),
                        add_label(resize_tile(mask_overlay, tile_size), "SAM dynamic classes"),
                        add_label(resize_tile(residual_vis, tile_size), "corrected residual"),
                        add_label(resize_tile(dynamic_flow_vis, tile_size), "mask x residual"),
                    ]
                )
                writer.write(frame)
                if preview is None:
                    preview = frame

        frame_summaries.append(
            {
                "frame": index,
                "detections": len(annotations),
                "dynamic_class_mask_ratio": float(dynamic_class_mask.mean()),
                "dynamic_flow_mask_ratio": float(dynamic_flow_mask.mean()) if dynamic_flow_mask is not None else 0.0,
                "mean_dynamic_residual_magnitude": float(residual_magnitude[dynamic_flow_mask].mean())
                if dynamic_flow_mask is not None and np.any(dynamic_flow_mask)
                else 0.0,
            }
        )

    if write_video:
        writer.release()
    if preview is not None:
        cv2.imwrite(str(output_dir / "preview_frame_000.jpg"), preview)

    summary = {
        "output_dir": str(output_dir),
        "source_dir": str(flow_result_dir),
        "video_path": str(video_path),
        "frame_count": frame_count,
        "device": device,
        "amp_dtype": args.amp_dtype,
        "text_prompt": args.text_prompt,
        "residual_magnitude_threshold": args.residual_magnitude_threshold,
        "mask_only": bool(args.mask_only),
        "mean_dynamic_class_mask_ratio": float(np.mean([row["dynamic_class_mask_ratio"] for row in frame_summaries])),
        "mean_dynamic_flow_mask_ratio": float(np.mean([row["dynamic_flow_mask_ratio"] for row in frame_summaries])),
        "frame_summaries": frame_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "frame_summaries"}, indent=2))
    return summary


def main():
    args = parse_args()
    project_root = Path.cwd().resolve()
    grounded_sam2_root = resolve_path(args.grounded_sam2_root)
    jobs = load_jobs(args)

    device, sam2_predictor, grounding_model, load_image, predict = build_models(args, grounded_sam2_root)
    summaries = []
    errors = []
    for index, job in enumerate(jobs):
        try:
            summary = process_job(
                args,
                job,
                project_root,
                device,
                sam2_predictor,
                grounding_model,
                load_image,
                predict,
            )
            summaries.append(summary)
        except Exception as exc:
            error = {"job_index": index, "job": job, "error": repr(exc)}
            errors.append(error)
            print(json.dumps(error, indent=2), file=sys.stderr)
    batch_summary = {
        "job_count": len(jobs),
        "completed": len(summaries),
        "errors": len(errors),
        "device": device,
        "amp_dtype": args.amp_dtype,
    }
    if args.job_manifest:
        print(json.dumps(batch_summary, indent=2))
    if errors:
        raise RuntimeError(json.dumps({"batch_summary": batch_summary, "errors": errors[:5]}, indent=2))


if __name__ == "__main__":
    main()
