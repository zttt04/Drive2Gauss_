import os
import math
import copy
import random
import logging
from collections import OrderedDict
from functools import partial

import torch
import torch.distributed as dist
from einops import rearrange, repeat
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from colossalai.booster.plugin import LowLevelZeroPlugin

from DISTT.acceleration.parallel_states import set_data_parallel_group, set_sequence_parallel_group, get_data_parallel_group
from DISTT.acceleration.plugin import ZeroSeqParallelPlugin
from DISTT.registry import SCHEDULERS, build_module
from DISTT.datasets import save_sample
from DISTT.acceleration.communications import gather_tensors

from .misc import get_logger, collate_bboxes_to_maxlen, move_to, add_box_latent, warn_once
from .inference_utils import add_null_condition, concat_6_views_pt, enable_offload


def _load_cached_validation_latents(cached_latent_path, vae_out_channels, latent_modalities, device, dtype):
    if cached_latent_path is None:
        return None
    if isinstance(cached_latent_path, str):
        paths = [cached_latent_path]
    else:
        paths = list(cached_latent_path)

    fallback_slices = {
        "rgb": slice(0, vae_out_channels),
        "depth": slice(vae_out_channels, vae_out_channels * 2),
        "flow": slice(vae_out_channels * 2, vae_out_channels * 3),
    }
    latents = {modality: [] for modality in latent_modalities}
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        latent = payload.get("latent")
        for modality in latent_modalities:
            modality_latent = payload.get(f"{modality}_latent")
            if modality_latent is None:
                if latent is None or modality not in fallback_slices:
                    raise KeyError(f"Missing {modality}_latent in cached payload: {path}")
                modality_latent = latent[:, fallback_slices[modality]]
            latents[modality].append(modality_latent)
    return {
        modality: torch.stack(items, dim=0).to(device=device, dtype=dtype)
        for modality, items in latents.items()
    }


def _decode_modalities(vae, latent, latent_modalities, vae_out_channels, num_views, num_frames, dtype):
    latent = rearrange(latent, "B (C NC) T H W -> B NC C T H W", NC=num_views)
    decoded = {}
    for idx, modality in enumerate(latent_modalities):
        start = idx * vae_out_channels
        end = start + vae_out_channels
        modality_latent = latent[:, :, start:end]
        modality_latent = rearrange(modality_latent, "B NC C T H W -> (B NC) C T H W")
        modality_video = vae.decode(modality_latent.to(dtype), num_frames=num_frames)
        decoded[modality] = rearrange(modality_video, "(B NC) C T H W -> B NC C T H W", NC=num_views)
    return decoded


def _decode_cached_modalities(vae, latent_dict, latent_modalities, num_views, num_frames, dtype):
    if latent_dict is None:
        return None
    decoded = {}
    for modality in latent_modalities:
        if modality not in latent_dict:
            continue
        modality_latent = rearrange(latent_dict[modality], "B NC C T H W -> (B NC) C T H W")
        modality_video = vae.decode(modality_latent.to(dtype), num_frames=num_frames)
        decoded[modality] = rearrange(modality_video, "(B NC) C T H W -> B NC C T H W", NC=num_views)
    return decoded


def _build_validation_x_ref(
    latent_dict,
    latent_modalities,
    vae_out_channels,
    num_views,
    latent_size,
    batch_size,
    device,
    dtype,
    rgb_ref_latent=None,
):
    ref_parts = []
    if latent_dict is not None:
        for modality in latent_modalities:
            modality_ref = latent_dict[modality][:, :, :, :1]
            if modality != "rgb":
                modality_ref = torch.zeros_like(modality_ref)
            ref_parts.append(modality_ref)
    elif rgb_ref_latent is not None:
        rgb_ref_latent = rearrange(rgb_ref_latent, "(B NC) C T H W -> B NC C T H W", NC=num_views)
        _, _, _, _, latent_height, latent_width = rgb_ref_latent.shape
        for modality in latent_modalities:
            if modality == "rgb":
                ref_parts.append(rgb_ref_latent)
            else:
                ref_parts.append(
                    torch.zeros(
                        batch_size,
                        num_views,
                        vae_out_channels,
                        1,
                        latent_height,
                        latent_width,
                        device=device,
                        dtype=dtype,
                    )
                )
    else:
        _, latent_height, latent_width = latent_size
        for _ in latent_modalities:
            ref_parts.append(
                torch.zeros(
                    batch_size,
                    num_views,
                    vae_out_channels,
                    1,
                    latent_height,
                    latent_width,
                    device=device,
                    dtype=dtype,
                )
            )
    x_ref = torch.cat(ref_parts, dim=2)
    return rearrange(x_ref, "B NC C T H W -> B (C NC) T H W")


def _clip_at(clips, idx):
    return clips[idx] if isinstance(clips, (list, tuple)) else clips[idx]


def _num_clips(clips):
    return len(clips) if isinstance(clips, (list, tuple)) else clips.shape[0]


def _save_validation_pairs(decoded_samples, decoded_gt, video_save_dir, total_num, fpss, save_fps, verbose):
    rgb_samples = decoded_samples.get("rgb")
    if rgb_samples is None:
        return 0

    num_clips = _num_clips(rgb_samples)
    for idx in range(num_clips):
        fps = save_fps if save_fps else fpss[idx]
        sample_rgb = concat_6_views_pt(_clip_at(rgb_samples, idx), oneline=False)
        if decoded_gt is not None and "rgb" in decoded_gt:
            gt_rgb = concat_6_views_pt(_clip_at(decoded_gt["rgb"], idx), oneline=False)
            rgb_pair = torch.cat([gt_rgb, sample_rgb], dim=2)
        else:
            rgb_pair = sample_rgb
        save_sample(
            rgb_pair.cpu(),
            fps=fps,
            save_path=os.path.join(video_save_dir, f"rgb_only_{total_num + idx:04d}"),
            high_quality=True,
            verbose=verbose >= 2,
        )

        panel_rows = []
        for modality in ("rgb", "depth", "flow"):
            if modality not in decoded_samples:
                continue
            sample_grid = concat_6_views_pt(_clip_at(decoded_samples[modality], idx), oneline=False)
            if decoded_gt is not None and modality in decoded_gt:
                gt_grid = concat_6_views_pt(_clip_at(decoded_gt[modality], idx), oneline=False)
                panel_rows.append(torch.cat([gt_grid, sample_grid], dim=3))
            else:
                panel_rows.append(sample_grid)
        if panel_rows:
            rgbd_flow_panel = torch.cat(panel_rows, dim=2)
            save_sample(
                rgbd_flow_panel.cpu(),
                fps=fps,
                save_path=os.path.join(video_save_dir, f"rgbd_flow_{total_num + idx:04d}"),
                high_quality=True,
                verbose=verbose >= 2,
            )
    return num_clips


@torch.no_grad()
def run_validation(val_cfg, text_encoder, vae, model, device, dtype,
                   val_loader: torch.utils.data.DataLoader,
                   coordinator: DistCoordinator, global_step: int,
                   exp_dir: str, mv_order_map, t_order_map):
    video_save_dir = os.path.join(exp_dir, f"validation-global_step{global_step}")
    if coordinator.is_master():
        os.makedirs(video_save_dir, exist_ok=True)
    verbose = val_cfg.get("verbose", 1)
    num_sample = val_cfg.get("num_sample", 2)
    save_fps = val_cfg.save_fps
    compare_gt = val_cfg.get("compare_gt", True)
    latent_modalities = tuple(val_cfg.get("latent_modalities", ("rgb",)))
    vae_out_channels = val_cfg.get("vae_out_channels", getattr(vae, "out_channels", 4))
    latent_channels = vae_out_channels * len(latent_modalities)

    val_cfg.cpu_offload = val_cfg.get("cpu_offload", False)
    if val_cfg.cpu_offload:
        raise NotImplementedError()
        text_encoder.t5.model.to("cpu")
        model.to("cpu")
        vae.to("cpu")
        text_encoder.t5.model, model, vae, last_hook = enable_offload(
            text_encoder.t5.model, model, vae, device)

    validation_scheduler = build_module(val_cfg.scheduler, SCHEDULERS)
    text_encoder.y_embedder = model.module.y_embedder  # hack for classifier-free guidance
    model.eval()

    total_num = 0
    for i, batch in enumerate(val_loader):
        cached_latent_path = batch.pop("cached_latent_path", None)
        generator = torch.Generator("cpu").manual_seed(val_cfg.seed)
        bl_generator = torch.Generator("cpu").manual_seed(val_cfg.seed)
        B, T, NC = batch["pixel_values"].shape[:3]
        latent_size = vae.get_latent_size((T, *batch["pixel_values"].shape[-2:]))
        gt_latents = _load_cached_validation_latents(
            cached_latent_path,
            vae_out_channels,
            latent_modalities,
            device,
            dtype,
        )
        if gt_latents is not None:
            first_modality = latent_modalities[0]
            latent_size = gt_latents[first_modality].shape[-3:]
        rgb_ref_latent = None
        if gt_latents is None and T > 1:
            x_ref = batch["pixel_values"][:, :1].to(device, dtype)
            x_ref = rearrange(x_ref, "B T NC C H W -> (B NC) C T H W")
            rgb_ref_latent = vae.encode(x_ref)

        # == prepare batch prompts ==
        y = batch.pop("captions")[0]  # B, just take first frame
        maps = batch.pop("bev_map_with_aux").to(device, dtype)  # B, T, C, H, W
        bbox = batch.pop("bboxes_3d_data")
        # B len list (T, NC, len, 8, 3)
        bbox = [bbox_i.data for bbox_i in bbox]
        # B, T, NC, len, 8, 3
        # TODO: `bbox` may have some redundancy on `NC` dim.
        # NOTE: we reshape the data later!
        bbox = collate_bboxes_to_maxlen(bbox, device, dtype, NC, T)
        # B, T, NC, 3, 7
        cams = batch.pop("camera_param").to(device, dtype)
        cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
        rel_pos = batch.pop("frame_emb").to(device, dtype)
        rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)  # BxNC, T, 1, 4, 4

        # == model input format ==
        model_args = {}
        model_args["maps"] = maps
        model_args["bbox"] = bbox
        model_args["cams"] = cams
        model_args["rel_pos"] = rel_pos
        model_args["fps"] = batch.pop('fps')
        model_args["height"] = batch.pop("height")
        model_args["width"] = batch.pop("width")
        model_args["num_frames"] = batch.pop("num_frames")
        if T > 1:
            model_args["x_ref"] = _build_validation_x_ref(
                gt_latents,
                latent_modalities,
                vae_out_channels,
                NC,
                latent_size,
                B,
                device,
                dtype,
                rgb_ref_latent=rgb_ref_latent,
            )
        model_args = move_to(model_args, device=device, dtype=dtype)
        # no need to move these
        model_args["mv_order_map"] = mv_order_map
        model_args["t_order_map"] = t_order_map

        logging.info('start gather fps ...')
        _fpss = gather_tensors(model_args['fps'], pg=get_data_parallel_group())
        logging.info('end gather fps ...')
        for ns in range(num_sample):
            z = torch.randn(
                len(y), latent_channels * NC, *latent_size, generator=generator,
            ).to(device=device, dtype=dtype)
            # == sample box ==
            if bbox is not None:
                # null set values to all zeros, this should be safe
                bbox = add_box_latent(bbox, B, NC, T, 
                    partial(model.module.sample_box_latent, generator=bl_generator))
                # overwrite!
                new_bbox = {}
                for k, v in bbox.items():
                    new_bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")  # BxNC, T, len, 3, 7
                model_args["bbox"] = move_to(new_bbox, device=device, dtype=dtype)
            # == add null condition ==
            # y is handled by scheduler.sample
            if val_cfg.scheduler.type == "dpm-solver" and val_cfg.scheduler.cfg_scale == 1.0:
                _model_args = copy.deepcopy(model_args)
            else:
                _model_args = add_null_condition(
                    copy.deepcopy(model_args),
                    model.module.camera_embedder.uncond_cam.to(device),
                    model.module.frame_embedder.uncond_cam.to(device),
                    prepend=(val_cfg.scheduler.type == "dpm-solver"),
                )
            # == inference ==
            samples = validation_scheduler.sample(
                model,
                text_encoder,
                z=z,
                prompts=y,
                device=device,
                additional_args=_model_args,
                progress=verbose >= 2 and coordinator.is_master(),
                mask=None,
            )
            decoded_samples = _decode_modalities(
                vae,
                samples,
                latent_modalities,
                vae_out_channels,
                NC,
                T,
                dtype,
            )
            if val_cfg.cpu_offload:
                last_hook.offload()
            del z, samples
            torch.cuda.empty_cache()

            # gather sample from all processes
            coordinator.block_all()
            logging.info("start gather sample ...")
            gathered_samples = {
                modality: gather_tensors(video, pg=get_data_parallel_group())
                for modality, video in decoded_samples.items()
            }
            logging.info("end gather sample ...")

            # == save samples ==
            if coordinator.is_master():
                sample_clips = {modality: [] for modality in gathered_samples}
                fpss = []
                for rank_idx, fps in enumerate(_fpss):
                    for modality, modality_samples in gathered_samples.items():
                        sample_clips[modality] += [s.cpu() for s in modality_samples[rank_idx]]
                    fpss += [int(_fps) for _fps in fps]
            del decoded_samples, gathered_samples
            coordinator.block_all()

        if compare_gt:
            decoded_gt = _decode_cached_modalities(
                vae,
                gt_latents,
                latent_modalities,
                NC,
                T,
                dtype,
            )
            if decoded_gt is None:
                x = batch.pop("pixel_values").to(device, dtype)
                decoded_gt = {"rgb": rearrange(x, "B T NC C H W -> B NC C T H W")}
            torch.cuda.empty_cache()
            logging.info("start gather gt ...")
            gathered_gt = {
                modality: gather_tensors(video, pg=get_data_parallel_group())
                for modality, video in decoded_gt.items()
            }
            logging.info("end gather gt ...")
            if coordinator.is_master():
                gt_clips = {modality: [] for modality in gathered_gt}
                fpss = []
                for rank_idx, fps in enumerate(_fpss):
                    for modality, modality_gt in gathered_gt.items():
                        gt_clips[modality] += [s.cpu() for s in modality_gt[rank_idx]]
                    fpss += [int(_fps) for _fps in fps]
                total_num += _save_validation_pairs(
                    sample_clips,
                    gt_clips,
                    video_save_dir,
                    total_num,
                    fpss,
                    save_fps,
                    verbose,
                )
            del gathered_gt
        elif coordinator.is_master():
            fpss = []
            for fps in _fpss:
                fpss += [int(_fps) for _fps in fps]
            total_num += _save_validation_pairs(
                sample_clips,
                None,
                video_save_dir,
                total_num,
                fpss,
                save_fps,
                verbose,
            )
        del _fpss
        torch.cuda.synchronize()
        coordinator.block_all()

    if val_cfg.cpu_offload:
        # TODO: need to remove hooks
        raise NotImplementedError()
    model.train()
    return video_save_dir


def create_colossalai_plugin(plugin, dtype, grad_clip, sp_size, reduce_bucket_size_in_m: int = 20, overlap_allgather=False, verbose=False):
    if plugin == "zero2":
        assert sp_size == 1, "Zero2 plugin does not support sequence parallelism"
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=dtype,
            initial_scale=2**16,
            max_norm=grad_clip,
            reduce_bucket_size_in_m=reduce_bucket_size_in_m,
            overlap_allgather=overlap_allgather,
            verbose=verbose,
        )
        dp_size = dist.get_world_size()
        DP_AXIS, SP_AXIS = 0, 1
        pg_mesh = ProcessGroupMesh(dp_size, sp_size)
        dp_group = pg_mesh.get_group_along_axis(DP_AXIS)
        sp_group = pg_mesh.get_group_along_axis(SP_AXIS)
        set_data_parallel_group(dp_group)
        set_sequence_parallel_group(sp_group)
    elif plugin == "zero2-seq":
        assert sp_size > 1, "Zero2-seq plugin requires sequence parallelism"
        plugin = ZeroSeqParallelPlugin(
            sp_size=sp_size,
            stage=2,
            precision=dtype,
            initial_scale=2**16,
            max_norm=grad_clip,
            reduce_bucket_size_in_m=reduce_bucket_size_in_m,
            overlap_allgather=overlap_allgather,
            verbose=verbose,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {plugin}")
    return plugin


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if not param.requires_grad:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                if hasattr(optimizer, "_param_store"):
                    master_param = optimizer._param_store.working_to_master_param[param_id]
                else:
                    master_param = optimizer.working_to_master_param[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "identity",
            "quarter_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "image_random",
            "image_head",
            "image_tail",
            "image_head_tail",
            "random",
            "intepolate",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        get_logger().info("mask ratios: %s", mask_ratios)
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[2]
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1 or condition_frames_max <= 1:
            return mask

        if mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "intepolate":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames, device=x.device) > mask_ratio
        # if mask is all False, set the last frame to True
        if not mask.any():
            mask[-1] = 1

        return mask

    def get_masks(self, x):
        masks = []
        for _ in range(len(x)):
            mask = self.get_mask(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks


def sp_vae(x, vae_func, sp_group: dist.ProcessGroup):
    """use sp_group to scatter vae encode

    Args:
        x (torch.Tensor): (B NC) C T ... or B C T ...
        vae (nn.Module): vae model
        dp_group (dist.ProcessGroup): _description_
    """
    group_size = dist.get_world_size(sp_group)
    local_rank = dist.get_rank(sp_group)
    B = x.shape[0]

    copy_size = group_size
    while copy_size < B:
        copy_size += group_size
    per_rank_bs = copy_size // group_size

    if per_rank_bs >= B:
        warn_once(
            f"x shape {x.shape} with {group_size} ranks does not fit dp_encode "
            f"fallback to the normal one."
        )
        return vae_func(x)

    if copy_size > B:
        x_copy_num = math.ceil(copy_size / B)
        x_temp = torch.cat([x for _ in range(x_copy_num)])[:copy_size]
        warn_once(f"Pad B={B} to {x_temp.shape}")
    elif copy_size < B:
        raise RuntimeError(f"{x.shape} got copy_size={copy_size}")
    else:
        x_temp = x

    local_x = x_temp[local_rank * per_rank_bs:(local_rank + 1) * per_rank_bs]
    assert local_x.shape[0] == per_rank_bs
    del x_temp
    local_latent = vae_func(local_x)

    global_latent = [torch.empty_like(local_latent) for _ in range(group_size)]
    dist.all_gather(global_latent, local_latent, group=sp_group)
    dist.barrier(sp_group)
    del local_latent
    global_latent = torch.cat(global_latent, dim=0)[:B]
    return global_latent
