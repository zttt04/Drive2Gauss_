import os
import csv
import json
import pickle
import types
from contextlib import nullcontext
import sys
import random
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from pprint import pformat

sys.path.append(".")
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import numpy as np
import cv2
import torch
import torch.nn.functional as F
if not torch.cuda.is_available() or DEVICE_TYPE == 'npu':
    USE_NPU = True
    os.environ['DEVICE_TYPE'] = "npu"
    DEVICE_TYPE = "npu"
    print("Enable NPU!")
    try:
        # just before torch_npu, let xformers know there is no gpu
        import xformers
        import xformers.ops
    except Exception as e:
        print(f"Got {e} during import xformers!")
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    USE_NPU = False
import DISTT.utils.module_contrib

import torch.distributed as dist
from einops import rearrange, repeat
import colossalai
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm
from mmcv.parallel import DataContainer

import logging
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('shapely.geos').setLevel(logging.WARNING)
logging.getLogger('numba.core').setLevel(logging.INFO)
logging.getLogger('DISTT.models.vae.vae_cogvideox').setLevel(logging.WARNING)

from DISTT.acceleration.checkpoint import set_grad_checkpoint
from DISTT.acceleration.parallel_states import get_data_parallel_group, get_sequence_parallel_group
from DISTT.datasets.dataloader import prepare_dataloader
from DISTT.registry import DATASETS, MODELS, SCHEDULERS, build_module
from DISTT.utils.ckpt_utils import (
    RandomStateManager,
    adapt_state_dict_shapes,
    load,
    model_gathering,
    model_sharding,
    prepare_ckpt,
    record_model_param_shape,
    save,
)
from DISTT.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config, merge_dataset_cfg, mmengine_conf_get, mmengine_conf_set
from DISTT.utils.lr_scheduler import LinearWarmupLR, MultiStepWithLinearWarmupLR
from DISTT.utils.misc import (
    Timer,
    all_reduce_mean,
    reset_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
    collate_bboxes_to_maxlen,
    move_to,
    add_box_latent,
)
from DISTT.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema, run_validation, sp_vae
from DISTT.utils.frechet_rgb_loss import OnlineFrechetRGBLoss
from DISTT.utils.frechet_rgb_stage4 import (
    Stage4PerViewFrechetRGBEMALoss,
    Stage4PerViewFrechetRGBLoss,
    StyleGANVI3DFeatureExtractor,
    build_per_view_rgb_features,
    decode_rgb_stage4,
    decode_rgb_stage4_turbo,
    load_stage4_reference_stats,
    prediction_gradient_surrogate,
    select_single_camera_latent,
)


def append_loss_history(
    exp_dir,
    global_step,
    loss_value,
    avg_loss,
    lr,
    diffusion_loss=None,
    fd_rgb_loss=None,
    fd_rgb_raw=None,
    fd_rgb_raw_per_view=None,
    fd_rgb_normalized_per_view=None,
    static_geo_loss=None,
    static_geo_mask_ratio=None,
    static_geo_stage1_rdepth_loss=None,
    static_geo_stage1_lidar_loss=None,
    static_geo_stage1_rdepth_ratio=None,
    static_geo_stage1_lidar_ratio=None,
):
    history_path = os.path.join(exp_dir, "loss_history.csv")
    file_exists = os.path.exists(history_path)
    with open(history_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "global_step",
                    "loss",
                    "avg_loss",
                    "lr",
                    "diffusion_loss",
                    "static_geo_loss",
                    "static_geo_mask_ratio",
                    "static_geo_stage1_rdepth_loss",
                    "static_geo_stage1_lidar_loss",
                    "static_geo_stage1_rdepth_ratio",
                    "static_geo_stage1_lidar_ratio",
                    "fd_rgb_loss",
                    "fd_rgb_raw",
                    *[f"fd_rgb_raw_view_{index}" for index in range(6)],
                    *[f"fd_rgb_normalized_view_{index}" for index in range(6)],
                ]
            )
            history_columns = 26
        else:
            with open(history_path, newline="") as history_file:
                history_columns = len(next(csv.reader(history_file), []))
        row = [
            global_step,
            loss_value,
            avg_loss,
            lr,
            diffusion_loss,
            static_geo_loss,
            static_geo_mask_ratio,
            static_geo_stage1_rdepth_loss,
            static_geo_stage1_lidar_loss,
            static_geo_stage1_rdepth_ratio,
            static_geo_stage1_lidar_ratio,
            fd_rgb_loss,
            fd_rgb_raw,
            *(fd_rgb_raw_per_view or [None] * 6),
            *(fd_rgb_normalized_per_view or [None] * 6),
        ]
        writer.writerow(row[:history_columns])
    return history_path


def save_loss_curve(exp_dir):
    history_path = os.path.join(exp_dir, "loss_history.csv")
    curve_path = os.path.join(exp_dir, "loss_curve.png")
    if not os.path.exists(history_path):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("matplotlib unavailable; skip loss curve: %s", exc)
        return None

    steps, losses, avg_losses = [], [], []
    diffusion_losses, static_geo_losses = [], []
    with open(history_path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["global_step"]))
            losses.append(float(row["loss"]))
            avg_losses.append(float(row["avg_loss"]))
            if row.get("diffusion_loss") not in (None, ""):
                diffusion_losses.append(float(row["diffusion_loss"]))
            if row.get("static_geo_loss") not in (None, ""):
                static_geo_losses.append(float(row["static_geo_loss"]))
    if not steps:
        return None

    plt.figure(figsize=(9, 5))
    plt.plot(steps, losses, linewidth=0.8, alpha=0.35, label="loss")
    plt.plot(steps, avg_losses, linewidth=1.5, label="avg_loss")
    if len(diffusion_losses) == len(steps):
        plt.plot(steps, diffusion_losses, linewidth=1.0, label="diffusion_loss")
    if len(static_geo_losses) == len(steps):
        plt.plot(steps, static_geo_losses, linewidth=1.0, label="static_geo_loss")
    plt.xlabel("global_step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    return curve_path


def load_cached_latents(
    cached_latent_path,
    vae_out_channels,
    latent_modalities,
    device,
    dtype,
    load_flow_loss_mask=False,
):
    if isinstance(cached_latent_path, str):
        paths = [cached_latent_path]
    else:
        paths = list(cached_latent_path)

    latent_by_modality = {modality: [] for modality in latent_modalities}
    flow_loss_masks = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        latent = payload.get("latent")
        fallback_slices = {
            "rgb": slice(0, vae_out_channels),
            "depth": slice(vae_out_channels, vae_out_channels * 2),
            "flow": slice(vae_out_channels * 2, vae_out_channels * 3),
        }
        for modality in latent_modalities:
            modality_latent = payload.get(f"{modality}_latent")
            if modality_latent is None:
                if latent is None or modality not in fallback_slices:
                    raise KeyError(f"Missing {modality}_latent in cached payload: {path}")
                modality_latent = latent[:, fallback_slices[modality]]
            latent_by_modality[modality].append(modality_latent)
        if load_flow_loss_mask:
            flow_loss_mask = payload.get("flow_loss_mask")
            if flow_loss_mask is None:
                raise KeyError(f"Missing flow_loss_mask in cached payload: {path}")
            flow_loss_masks.append(flow_loss_mask)

    outputs = []
    for modality in latent_modalities:
        modality_latents = torch.stack(latent_by_modality[modality], dim=0).to(device=device, dtype=dtype)
        modality_latents = rearrange(modality_latents, "B NC C T H W -> (B NC) C T H W")
        outputs.append(modality_latents)

    flow_loss_mask = None
    if load_flow_loss_mask:
        flow_loss_mask = torch.stack(flow_loss_masks, dim=0).to(device=device, dtype=dtype)
        flow_loss_mask = rearrange(flow_loss_mask, "B NC C T H W -> (B NC) C T H W")
    return outputs, flow_loss_mask


def build_modality_loss_channel_weights(
    modality_loss_weights,
    latent_modalities,
    vae_out_channels,
    num_cameras,
    device,
    dtype,
):
    if not modality_loss_weights:
        return None

    modality_weights = []
    for modality in latent_modalities:
        modality_weight = float(modality_loss_weights.get(modality, 1.0))
        modality_weights.extend([modality_weight] * vae_out_channels)

    channel_weights = torch.tensor(modality_weights, device=device, dtype=dtype)
    # x is arranged as B, (C NC), T, H, W, so every latent channel is expanded
    # across the camera/view dimension.
    channel_weights = channel_weights.repeat_interleave(num_cameras)
    return channel_weights.view(1, -1, 1, 1, 1)


def build_frame_loss_weights(loss_frame_weights, num_frames, latent_num_frames, device, dtype):
    if not loss_frame_weights:
        return None

    if hasattr(loss_frame_weights, "to_dict"):
        loss_frame_weights = loss_frame_weights.to_dict()

    if isinstance(loss_frame_weights, dict):
        weights = loss_frame_weights.get(str(num_frames), None)
        if weights is None:
            weights = loss_frame_weights.get(num_frames, None)
        if weights is None:
            weights = loss_frame_weights.get("default", None)
    else:
        weights = loss_frame_weights

    if weights is None:
        return None

    weights = list(weights)
    if len(weights) == num_frames and latent_num_frames != num_frames:
        raw_weights = torch.tensor(weights, device=device, dtype=dtype)
        latent_indices = torch.linspace(
            0,
            num_frames - 1,
            steps=latent_num_frames,
            device=device,
        ).round().long()
        return raw_weights.index_select(0, latent_indices)

    if len(weights) != latent_num_frames:
        raise ValueError(
            "loss_frame_weights for "
            f"{num_frames} raw frames / {latent_num_frames} latent frames "
            f"has {len(weights)} values: {weights}"
        )
    return torch.tensor(weights, device=device, dtype=dtype)


COGVIDEOX_SCALING_FACTOR = 1.15258426
STATIC_GEO_CAMERA_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
)


def patch_transformers_cache():
    try:
        import transformers

        if not hasattr(transformers, "EncoderDecoderCache") and hasattr(transformers, "DynamicCache"):
            transformers.EncoderDecoderCache = transformers.DynamicCache
    except Exception:
        pass


def build_static_geo_turbo_decoder(cfg, device, dtype, logger):
    patch_transformers_cache()
    os.environ.setdefault("_CHECK_PEFT", "0")
    try:
        import diffusers.utils.import_utils as diffusers_import_utils

        if not hasattr(diffusers_import_utils, "is_optimum_quanto_version"):
            diffusers_import_utils.is_optimum_quanto_version = lambda *args, **kwargs: False
    except Exception:
        pass
    default_repo_root = Path(__file__).resolve().parents[3] / "Turbo-VAED-main"
    repo_root = Path(cfg.get("static_geo_loss_turbo_repo_root", default_repo_root)).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    turbo_src = repo_root / "diffusers_vae" / "src"
    if str(turbo_src) not in sys.path:
        sys.path.insert(0, str(turbo_src))

    try:
        from diffusers_vae.src.diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import AutoencoderKLTurboVAED
    except ModuleNotFoundError:
        from diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import AutoencoderKLTurboVAED

    config_path = cfg.get("static_geo_loss_turbo_config", None)
    if config_path is None:
        config_path = repo_root / "configs" / "Turbo-VAED-Cog.json"
    else:
        config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(
        cfg.get(
            "static_geo_loss_turbo_checkpoint",
            os.environ.get(
                "TURBO_VAED_CHECKPOINT",
                "pretrained/Turbo-VAED-Cog.pth",
            ),
        )
    ).expanduser().resolve()

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    turbo_decoder = AutoencoderKLTurboVAED.from_config(config=config)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "gen_model" in state:
        state = state["gen_model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported Turbo-VAED checkpoint payload: {type(state)!r}")
    state = {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    missing, unexpected = turbo_decoder.decoder.load_state_dict(state, strict=False)
    if bool(cfg.get("static_geo_loss_turbo_enable_slicing", True)):
        turbo_decoder.enable_slicing()
    if bool(cfg.get("static_geo_loss_turbo_enable_tiling", True)):
        turbo_decoder.enable_tiling()
    if bool(cfg.get("static_geo_loss_turbo_enable_framewise_decoding", True)):
        turbo_decoder.use_framewise_decoding = True
    configure_static_geo_turbo_checkpointing(turbo_decoder, cfg, logger)
    turbo_decoder.to(device=device, dtype=dtype).eval()
    turbo_decoder.requires_grad_(False)
    logger.info(
        "Loaded frozen Turbo-VAED decoder for static_geo_loss: checkpoint=%s config=%s missing=%d unexpected=%d gradient_checkpointing=%s policy=%s",
        checkpoint_path,
        config_path,
        len(missing),
        len(unexpected),
        getattr(turbo_decoder, "is_gradient_checkpointing", False),
        getattr(turbo_decoder, "_static_geo_gradient_checkpoint_policy", "none"),
    )
    return turbo_decoder


def build_fd_rgb_turbo_decoder(cfg, device, dtype, logger):
    """Build Turbo-VAED-Cog using FD-specific config keys and the shared loader."""
    turbo_cfg = dict(cfg)
    key_map = {
        "fd_rgb_stage4_turbo_repo_root": "static_geo_loss_turbo_repo_root",
        "fd_rgb_stage4_turbo_config": "static_geo_loss_turbo_config",
        "fd_rgb_stage4_turbo_checkpoint": "static_geo_loss_turbo_checkpoint",
        "fd_rgb_stage4_turbo_enable_slicing": "static_geo_loss_turbo_enable_slicing",
        "fd_rgb_stage4_turbo_enable_tiling": "static_geo_loss_turbo_enable_tiling",
        "fd_rgb_stage4_turbo_enable_framewise_decoding": "static_geo_loss_turbo_enable_framewise_decoding",
        "fd_rgb_stage4_turbo_enable_gradient_checkpointing": "static_geo_loss_turbo_enable_gradient_checkpointing",
        "fd_rgb_stage4_turbo_gradient_checkpoint_policy": "static_geo_loss_turbo_gradient_checkpoint_policy",
        "fd_rgb_stage4_turbo_checkpoint_up_block_start": "static_geo_loss_turbo_checkpoint_up_block_start",
        "fd_rgb_stage4_turbo_checkpoint_mid_block": "static_geo_loss_turbo_checkpoint_mid_block",
    }
    for source_key, target_key in key_map.items():
        if source_key in cfg:
            turbo_cfg[target_key] = cfg[source_key]
    decoder = build_static_geo_turbo_decoder(turbo_cfg, device, dtype, logger)
    logger.info("Using frozen Turbo-VAED-Cog as the Stage4 RGB FD decoder.")
    return decoder


def configure_static_geo_turbo_checkpointing(turbo_decoder, cfg, logger):
    policy = cfg.get("static_geo_loss_turbo_gradient_checkpoint_policy", None)
    if policy is None:
        policy = "all" if bool(cfg.get("static_geo_loss_turbo_enable_gradient_checkpointing", True)) else "none"
    policy = str(policy).lower()
    aliases = {
        "true": "all",
        "1": "all",
        "yes": "all",
        "false": "none",
        "0": "none",
        "no": "none",
        "last_half": "late",
        "highres": "late",
    }
    policy = aliases.get(policy, policy)

    turbo_decoder._static_geo_gradient_checkpoint_policy = policy
    if policy == "all":
        turbo_decoder.enable_gradient_checkpointing()
        return
    if policy == "none":
        return
    if policy != "late":
        raise ValueError(
            "Unsupported static_geo_loss_turbo_gradient_checkpoint_policy="
            f"{policy!r}; expected one of all, none, late."
        )

    decoder = getattr(turbo_decoder, "decoder", None)
    up_blocks = getattr(decoder, "up_blocks", None)
    if decoder is None or up_blocks is None:
        raise AttributeError("Turbo-VAED decoder does not expose decoder.up_blocks for partial checkpointing.")

    num_up_blocks = len(up_blocks)
    default_start = max(0, num_up_blocks // 2)
    checkpoint_up_block_start = int(
        cfg.get("static_geo_loss_turbo_checkpoint_up_block_start", default_start)
    )
    checkpoint_up_block_start = max(0, min(checkpoint_up_block_start, num_up_blocks))
    checkpoint_up_block_indices = set(range(checkpoint_up_block_start, num_up_blocks))
    checkpoint_mid_block = bool(cfg.get("static_geo_loss_turbo_checkpoint_mid_block", False))
    decoder.gradient_checkpointing = True
    decoder._static_geo_checkpoint_mid_block = checkpoint_mid_block
    decoder._static_geo_checkpoint_up_block_indices = tuple(sorted(checkpoint_up_block_indices))

    def partial_checkpoint_forward(self, hidden_states, temb=None, feature_enabled=False):
        hidden_states = self.conv_in(hidden_states)
        feature_backup = {}
        checkpoint_enabled = torch.is_grad_enabled()

        def checkpoint_module(module, *inputs):
            def custom_forward(*custom_inputs):
                return module(*custom_inputs)

            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                *inputs,
                use_reentrant=False,
            )

        if checkpoint_enabled and checkpoint_mid_block:
            hidden_states = checkpoint_module(self.mid_block, hidden_states, temb)
        else:
            hidden_states = self.mid_block(hidden_states, temb)
            if feature_enabled:
                feature_backup["mid_block"] = hidden_states

        for index, up_block in enumerate(self.up_blocks):
            if checkpoint_enabled and index in checkpoint_up_block_indices:
                hidden_states = checkpoint_module(up_block, hidden_states, temb)
            else:
                hidden_states = up_block(hidden_states, temb)
                if feature_enabled:
                    feature_backup[f"up_block_{index}"] = hidden_states

        if self.patch_size >= 2:
            hidden_states = self.norm_up_1(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            hidden_states = self.conv_act(hidden_states)
            hidden_states_array = []
            for t in range(hidden_states.shape[2]):
                h = self.upsampler2d_1(hidden_states[:, :, t, :, :])
                hidden_states_array.append(h)
            hidden_states = torch.stack(hidden_states_array, dim=2)

        if self.patch_size >= 4:
            hidden_states = self.norm_up_2(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            hidden_states = self.conv_act(hidden_states)
            hidden_states = self.upsampler2d_2(hidden_states)

        if self.patch_size == 1:
            hidden_states = self.norm_out(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        else:
            variance = hidden_states.pow(2).mean(1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + 1e-8)

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        if feature_enabled:
            return hidden_states, feature_backup
        return hidden_states

    decoder.forward = types.MethodType(partial_checkpoint_forward, decoder)
    logger.info(
        "Using partial Turbo-VAED checkpointing for static_geo_loss: policy=%s checkpoint_mid_block=%s checkpoint_up_blocks=%s/%s",
        policy,
        checkpoint_mid_block,
        sorted(checkpoint_up_block_indices),
        num_up_blocks,
    )


def decoded_to_metric_depth(decoded, depth_max):
    depth_norm = decoded.float().mean(dim=1, keepdim=True).clamp(-1.0, 1.0)
    return ((depth_norm + 1.0) * 0.5 * depth_max).clamp(0.0, depth_max)


def extract_modality_latent(latent, modality, latent_modalities, vae_out_channels, num_cameras):
    modality_index = latent_modalities.index(modality)
    start = modality_index * vae_out_channels * num_cameras
    end = start + vae_out_channels * num_cameras
    return rearrange(
        latent[:, start:end],
        "B (C NC) T H W -> (B NC) C T H W",
        C=vae_out_channels,
        NC=num_cameras,
    ).contiguous()


def decode_rgb_for_frechet(vae, rgb_latent, decode_batch_size, use_checkpoint):
    """Decode RGB latent chunks while retaining gradients only for predictions."""
    if use_checkpoint:
        from torch.utils.checkpoint import checkpoint as activation_checkpoint

    vae_dtype = next(vae.parameters()).dtype
    if rgb_latent.dtype != vae_dtype:
        rgb_latent = rgb_latent.to(dtype=vae_dtype)
    decoded_chunks = []
    for latent_chunk in rgb_latent.split(max(1, int(decode_batch_size)), dim=0):
        if use_checkpoint and torch.is_grad_enabled() and latent_chunk.requires_grad:
            decoded = activation_checkpoint(
                lambda z: vae.decode(z),
                latent_chunk,
                use_reentrant=False,
            )
        else:
            decoded = vae.decode(latent_chunk)
        decoded_chunks.append(decoded)
    return torch.cat(decoded_chunks, dim=0)


def build_rgb_frechet_features(decoded, batch_size, num_cameras, pool_size):
    """Create compact, camera-balanced RGB representation features."""
    decoded = decoded.float().clamp(-1.0, 1.0).add(1.0).mul(0.5)
    pooled = F.adaptive_avg_pool3d(decoded, (1, int(pool_size), int(pool_size)))
    features = pooled.flatten(1)
    return features.view(batch_size, num_cameras, -1).mean(dim=1)


def camera_param_to_camera2lidar(camera_param):
    camera2lidar = torch.eye(
        4,
        device=camera_param.device,
        dtype=torch.float32,
    ).view(1, 1, 1, 4, 4).repeat(
        camera_param.shape[0],
        camera_param.shape[1],
        camera_param.shape[2],
        1,
        1,
    )
    camera2lidar[..., :3, :4] = camera_param[..., :, 3:7].float()
    return camera2lidar


def resolve_static_geo_lidar_path(path, data_root):
    prefixes = ("../data/nuscenes/", "data/nuscenes/", "nuscenes/")
    for prefix in prefixes:
        if path.startswith(prefix):
            return os.path.join(data_root, path[len(prefix):])
    if os.path.isabs(path):
        return path
    return os.path.join(data_root, path)


def load_static_geo_lidar_infos(ann_file, data_root, logger):
    if not ann_file:
        raise ValueError("static_geo_loss_lidar_ann_file is required when LiDAR loss is enabled.")
    with open(ann_file, "rb") as file:
        annotations = pickle.load(file)
    infos = {}
    for info in annotations["infos"]:
        info = dict(info)
        info["lidar_path"] = resolve_static_geo_lidar_path(info["lidar_path"], data_root)
        infos[info["token"]] = info
    logger.info("Loaded %d LiDAR infos for static_geo stage1 loss from %s", len(infos), ann_file)
    return infos


def load_static_geo_depth_root_map(depth_root_json, logger):
    if not depth_root_json:
        raise ValueError("static_geo_loss_depth_root_json is required for rdepth_lidar_stage1.")
    with open(depth_root_json, "r") as file:
        sample_token_to_depth = json.load(file)
    logger.info(
        "Loaded %d RDepth token mappings for static_geo stage1 loss from %s",
        len(sample_token_to_depth),
        depth_root_json,
    )
    return sample_token_to_depth


def extract_meta_token(meta_data, frame_index, batch_index):
    item = meta_data["metas"][frame_index][batch_index]
    if isinstance(item, DataContainer):
        item = item.data
    return item["token"]


def resize_static_geo_array(array, width, height, interpolation):
    if array.shape == (height, width):
        return array
    return cv2.resize(array, (width, height), interpolation=interpolation)


def load_static_geo_rdepth_target(
    token,
    view_index,
    sample_token_to_depth,
    rdepth_root,
    width,
    height,
    sky_class_id,
):
    depth_relative_root = sample_token_to_depth.get(token)
    if depth_relative_root is None:
        return None, None
    frame_root = Path(rdepth_root) / depth_relative_root.lstrip("/")
    camera = STATIC_GEO_CAMERA_ORDER[int(view_index)]
    depth_path = frame_root / camera / "refined_depth.npz"
    semantic_path = frame_root / camera / "semantic_oneformer.npz"
    if not depth_path.exists() or not semantic_path.exists():
        return None, None
    depth = np.load(depth_path)["depth_pred"].astype(np.float32)
    depth = resize_static_geo_array(depth, width, height, cv2.INTER_CUBIC)
    semantic = np.load(semantic_path)["sem"]
    semantic = resize_static_geo_array(semantic, width, height, cv2.INTER_NEAREST)
    valid = semantic != int(sky_class_id)
    return np.ascontiguousarray(depth), np.ascontiguousarray(valid)


def project_lidar_depth_with_camera_param(
    lidar_path,
    camera_param,
    width,
    height,
    min_depth,
    max_depth,
    min_lidar_radius,
    max_points,
):
    points = np.fromfile(lidar_path, dtype=np.float32)
    if points.size % 5 != 0:
        raise ValueError(f"Unexpected NuScenes LiDAR shape in {lidar_path}: {points.size}")
    points = points.reshape(-1, 5)[:, :3]
    radius = np.linalg.norm(points[:, :2], axis=1)
    points = points[radius >= min_lidar_radius]
    if max_points > 0 and points.shape[0] > max_points:
        step = int(np.ceil(points.shape[0] / max_points))
        points = points[::step]

    intrinsics = camera_param[:, :3].astype(np.float64)
    camera_to_lidar = np.eye(4, dtype=np.float64)
    camera_to_lidar[:3, :4] = camera_param[:, 3:7].astype(np.float64)
    lidar_to_camera = np.linalg.inv(camera_to_lidar)
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (lidar_to_camera @ homogeneous.T).T
    depth = camera_points[:, 2]
    pixel_x = camera_points[:, 0] / np.maximum(depth, 1.0e-6) * intrinsics[0, 0]
    pixel_x += intrinsics[0, 2]
    pixel_y = camera_points[:, 1] / np.maximum(depth, 1.0e-6) * intrinsics[1, 1]
    pixel_y += intrinsics[1, 2]
    valid = (
        (depth > min_depth)
        & (depth < max_depth)
        & (pixel_x >= 0.0)
        & (pixel_x <= width - 1)
        & (pixel_y >= 0.0)
        & (pixel_y <= height - 1)
    )
    if not np.any(valid):
        return None, None
    pixel_x = np.rint(pixel_x[valid]).astype(np.int64).clip(0, width - 1)
    pixel_y = np.rint(pixel_y[valid]).astype(np.int64).clip(0, height - 1)
    depth = depth[valid].astype(np.float32)
    pixel_index = pixel_y * width + pixel_x
    z_buffer = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(z_buffer, pixel_index, depth)
    valid_pixel_index = np.flatnonzero(np.isfinite(z_buffer))
    return valid_pixel_index, z_buffer[valid_pixel_index]


def decode_static_geo_metric_depth(
    depth_latent,
    turbo_decoder,
    latent_scale,
    depth_max,
    checkpoint_decode,
):
    decode_dtype = next(turbo_decoder.parameters()).dtype
    scaled_depth_latent = (depth_latent * latent_scale).to(dtype=decode_dtype)

    def decode_depth_latent(latent):
        return turbo_decoder.decode(latent, return_dict=False)[0]

    if checkpoint_decode and torch.is_grad_enabled() and scaled_depth_latent.requires_grad:
        from torch.utils.checkpoint import checkpoint as activation_checkpoint

        decoded = activation_checkpoint(
            decode_depth_latent,
            scaled_depth_latent,
            use_reentrant=False,
        )
    else:
        decoded = decode_depth_latent(scaled_depth_latent)
    return decoded_to_metric_depth(decoded, depth_max)


def stage1_rdepth_lidar_geo_loss(
    pred_x0,
    camera_param,
    meta_data,
    turbo_decoder,
    latent_modalities,
    vae_out_channels,
    num_cameras,
    video_length,
    depth_max,
    latent_scale,
    min_valid_depth,
    sky_depth_threshold,
    checkpoint_decode,
    sample_frames,
    sample_views,
    contiguous_sample_views,
    rdepth_weight,
    lidar_weight,
    huber_beta,
    sample_token_to_depth,
    rdepth_root,
    sky_class_id,
    lidar_infos,
    lidar_min_radius,
    lidar_max_points,
    record_time=False,
    debug_ratio=False,
):
    profile_times = {}

    def profile_section(name):
        class ProfileContext:
            def __enter__(self_inner):
                if record_time:
                    torch.cuda.synchronize()
                    self_inner.start_time = time.time()
                return self_inner

            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                if record_time:
                    torch.cuda.synchronize()
                    profile_times[name] = profile_times.get(name, 0.0) + time.time() - self_inner.start_time

        return ProfileContext()

    if "depth" not in latent_modalities:
        raise ValueError("static_geo_loss requires 'depth' in latent_modalities.")
    with profile_section("select_depth_latent"):
        batch_size = pred_x0.shape[0]
        selected_views = sample_static_geo_view_indices(
            num_cameras,
            int(sample_views),
            bool(contiguous_sample_views),
        )
        view_index_tensor = torch.tensor(selected_views, device=pred_x0.device, dtype=torch.long)

        pred_depth_latent = extract_modality_latent(
            pred_x0,
            "depth",
            latent_modalities,
            vae_out_channels,
            num_cameras,
        )
        pred_depth_latent = rearrange(
            pred_depth_latent,
            "(B NC) C T H W -> B NC C T H W",
            B=batch_size,
            NC=num_cameras,
        )
        pred_depth_latent = pred_depth_latent.index_select(1, view_index_tensor)
        selected_view_count = len(selected_views)
        pred_depth_latent = rearrange(pred_depth_latent, "B V C T H W -> (B V) C T H W")

    with profile_section("decode_pred_depth"):
        pred_metric = decode_static_geo_metric_depth(
            pred_depth_latent,
            turbo_decoder,
            latent_scale,
            depth_max,
            checkpoint_decode,
        )
        pred_metric = rearrange(
            pred_metric[:, 0],
            "(B V) T H W -> B V T H W",
            B=batch_size,
            V=selected_view_count,
        )
        frames = min(int(video_length), pred_metric.shape[2])
        pred_metric = pred_metric[:, :, :frames]
        selected_frames = sample_static_geo_stratified_indices(frames, int(sample_frames))
        frame_index_tensor = torch.tensor(selected_frames, device=pred_metric.device, dtype=torch.long)
        selected_frame_count = len(selected_frames)
        pred_metric = pred_metric.index_select(2, frame_index_tensor)
    if sample_token_to_depth is None or not rdepth_root:
        raise RuntimeError("RDepth stage1 loss requires original RDepth root and token map.")
    if meta_data is None:
        raise RuntimeError("RDepth stage1 loss requires batch meta_data.")

    with profile_section("load_rdepth_targets"):
        _, _, _, height, width = pred_metric.shape
        target_metric = pred_metric.new_zeros(pred_metric.shape)
        target_non_sky = torch.zeros(pred_metric.shape, device=pred_metric.device, dtype=torch.bool)
        target_available = torch.zeros(pred_metric.shape, device=pred_metric.device, dtype=torch.bool)
        debug_rdepth_items = []
        for batch_index in range(batch_size):
            for local_frame, frame_index in enumerate(selected_frames):
                token = extract_meta_token(meta_data, frame_index, batch_index)
                for local_view, view_index in enumerate(selected_views):
                    depth, non_sky = load_static_geo_rdepth_target(
                        token,
                        view_index,
                        sample_token_to_depth,
                        rdepth_root,
                        width,
                        height,
                        sky_class_id,
                    )
                    if depth is None or non_sky is None:
                        continue
                    target_metric[batch_index, local_view, local_frame] = torch.as_tensor(
                        depth,
                        device=pred_metric.device,
                        dtype=pred_metric.dtype,
                    )
                    target_non_sky[batch_index, local_view, local_frame] = torch.as_tensor(
                        non_sky,
                        device=pred_metric.device,
                        dtype=torch.bool,
                    )
                    target_available[batch_index, local_view, local_frame] = True
                    if debug_ratio and (
                        (not dist.is_available())
                        or (not dist.is_initialized())
                        or dist.get_rank() == 0
                    ) and len(debug_rdepth_items) < 24:
                        item_valid = (
                            (depth > min_valid_depth)
                            & (depth < sky_depth_threshold)
                            & np.isfinite(depth)
                            & non_sky
                        )
                        debug_rdepth_items.append(
                            (
                                token,
                                STATIC_GEO_CAMERA_ORDER[int(view_index)],
                                float((depth > min_valid_depth).mean()),
                                float((depth < sky_depth_threshold).mean()),
                                float(np.isfinite(depth).mean()),
                                float(non_sky.mean()),
                                float(item_valid.mean()),
                                float(np.nanmin(depth)),
                                float(np.nanmax(depth)),
                            )
                        )

    with profile_section("rdepth_dense_loss"):
        finite_target = torch.isfinite(target_metric)
        depth_gt_min = target_metric > min_valid_depth
        depth_lt_max = target_metric < sky_depth_threshold
        valid = (
            depth_gt_min
            & depth_lt_max
            & finite_target
            & target_non_sky
            & target_available
        ).detach()
        log_pred = torch.log(pred_metric.clamp_min(min_valid_depth))
        log_target = torch.log(target_metric.clamp_min(min_valid_depth))
        dense_error = F.smooth_l1_loss(
            log_pred,
            log_target,
            beta=float(huber_beta),
            reduction="none",
        )
        valid_count = valid.sum()
        if valid_count.item() == 0:
            dense_loss = pred_metric.sum() * 0.0
            dense_ratio = pred_metric.new_tensor(0.0)
        else:
            dense_loss = dense_error[valid].sum() / valid_count.to(pred_metric.dtype)
            available_count = target_available.sum().clamp_min(1)
            dense_ratio = valid_count.to(pred_metric.dtype) / available_count.to(pred_metric.dtype)
        if debug_ratio:
            with torch.no_grad():
                stats = torch.stack(
                    [
                        valid.float().mean(),
                        depth_gt_min.float().mean(),
                        depth_lt_max.float().mean(),
                        finite_target.float().mean(),
                        target_non_sky.float().mean(),
                    ]
                ).detach().float()
                if dist.is_available() and dist.is_initialized():
                    gathered = [torch.zeros_like(stats) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered, stats)
                    gathered = torch.stack(gathered)
                    if dist.get_rank() == 0:
                        logging.getLogger().info(
                            "Stage1 RDepth ratio debug all-rank | "
                            "valid mean/min/max %.4f/%.4f/%.4f | "
                            "depth>min %.4f | depth<thresh %.4f | finite %.4f | non_sky %.4f",
                            gathered[:, 0].mean().item(),
                            gathered[:, 0].min().item(),
                            gathered[:, 0].max().item(),
                            gathered[:, 1].mean().item(),
                            gathered[:, 2].mean().item(),
                            gathered[:, 3].mean().item(),
                            gathered[:, 4].mean().item(),
                        )
                elif not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                    logging.getLogger().info(
                        "Stage1 RDepth ratio debug | valid %.4f | depth>min %.4f | "
                        "depth<thresh %.4f | finite %.4f | non_sky %.4f",
                        stats[0].item(),
                        stats[1].item(),
                        stats[2].item(),
                        stats[3].item(),
                        stats[4].item(),
                    )
                per_view_ratio = valid.float().mean(dim=(0, 2, 3, 4)).detach().float()
                if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                    finite_values = target_metric[finite_target].detach().float()
                    if finite_values.numel() > 1000000:
                        step = int(np.ceil(finite_values.numel() / 1000000))
                        finite_values = finite_values[::step]
                    quantiles = torch.quantile(
                        finite_values,
                        torch.tensor([0.5, 0.9, 0.99], device=finite_values.device),
                    )
                    logging.getLogger().info(
                        "Stage1 RDepth ratio debug rank0 | selected_views=%s selected_frames=%s "
                        "per_view_valid=%s depth_q50/q90/q99=%.2f/%.2f/%.2f thresh=%.1f",
                        selected_views,
                        selected_frames,
                        [round(value, 4) for value in per_view_ratio.cpu().tolist()],
                        quantiles[0].item(),
                        quantiles[1].item(),
                        quantiles[2].item(),
                        float(sky_depth_threshold),
                    )
                    for debug_item in debug_rdepth_items:
                        logging.getLogger().info(
                            "Stage1 RDepth file debug rank0 | token=%s camera=%s "
                            "depth>min=%.4f depth<thresh=%.4f finite=%.4f non_sky=%.4f "
                            "valid=%.4f depth_min/max=%.2f/%.2f",
                            debug_item[0],
                            debug_item[1],
                            debug_item[2],
                            debug_item[3],
                            debug_item[4],
                            debug_item[5],
                            debug_item[6],
                            debug_item[7],
                            debug_item[8],
                        )

    lidar_loss = pred_metric.sum() * 0.0
    lidar_ratio = pred_metric.new_tensor(0.0)
    with profile_section("lidar_sparse_loss"):
        if float(lidar_weight) > 0.0:
            if lidar_infos is None:
                raise RuntimeError("LiDAR stage1 loss is enabled but no lidar_infos were loaded.")
            if meta_data is None:
                raise RuntimeError("LiDAR stage1 loss requires batch meta_data.")
            camera_param_cpu = camera_param[:, :frames].detach().float().cpu().numpy()
            lidar_loss_sum = pred_metric.new_tensor(0.0)
            lidar_count = pred_metric.new_tensor(0.0)
            for batch_index in range(batch_size):
                for local_frame, frame_index in enumerate(selected_frames):
                    token = extract_meta_token(meta_data, frame_index, batch_index)
                    lidar_info = lidar_infos.get(token)
                    if lidar_info is None:
                        continue
                    lidar_path = lidar_info["lidar_path"]
                    for local_view, view_index in enumerate(selected_views):
                        pixel_index, lidar_depth = project_lidar_depth_with_camera_param(
                            lidar_path,
                            camera_param_cpu[batch_index, frame_index, view_index],
                            width,
                            height,
                            min_valid_depth,
                            sky_depth_threshold,
                            lidar_min_radius,
                            int(lidar_max_points),
                        )
                        if pixel_index is None:
                            continue
                        pixel_index = torch.as_tensor(
                            pixel_index,
                            device=pred_metric.device,
                            dtype=torch.long,
                        )
                        lidar_depth = torch.as_tensor(
                            lidar_depth,
                            device=pred_metric.device,
                            dtype=pred_metric.dtype,
                        )
                        prediction = pred_metric[batch_index, local_view, local_frame].reshape(-1)[pixel_index]
                        prediction_valid = (
                            torch.isfinite(prediction)
                            & (prediction > min_valid_depth)
                            & (prediction < sky_depth_threshold)
                        )
                        if not prediction_valid.any():
                            continue
                        lidar_error = F.smooth_l1_loss(
                            torch.log(prediction[prediction_valid].clamp_min(min_valid_depth)),
                            torch.log(lidar_depth[prediction_valid].clamp_min(min_valid_depth)),
                            beta=float(huber_beta),
                            reduction="sum",
                        )
                        lidar_loss_sum = lidar_loss_sum + lidar_error
                        lidar_count = lidar_count + prediction_valid.sum().to(pred_metric.dtype)
            if lidar_count.item() > 0:
                lidar_loss = lidar_loss_sum / lidar_count
                lidar_ratio = lidar_count / max(
                    batch_size * selected_view_count * selected_frame_count * height * width,
                    1,
                )

    total_loss = float(rdepth_weight) * dense_loss + float(lidar_weight) * lidar_loss
    if record_time:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        timing = " | ".join(f"{name}: {elapsed:.3f}s" for name, elapsed in profile_times.items())
        logging.getLogger().info("Stage1 profile rank %d | %s", rank, timing)
    return total_loss, dense_loss.detach(), lidar_loss.detach(), dense_ratio.detach(), lidar_ratio.detach()


def sample_static_geo_indices(length, sample_count):
    if sample_count <= 0 or sample_count >= length:
        return list(range(length))
    return sorted(random.sample(range(length), sample_count))


def sample_static_geo_stratified_frames(length):
    if length < 3:
        raise ValueError(
            f"Stratified static-geo sampling requires at least 3 frames, got {length}."
        )
    if length >= 17:
        frame_ranges = ((0, 2), (3, 8), (9, length - 1))
    else:
        first_end = max(0, length // 3 - 1)
        middle_end = max(first_end + 1, 2 * length // 3 - 1)
        frame_ranges = (
            (0, first_end),
            (first_end + 1, middle_end),
            (middle_end + 1, length - 1),
        )
    return [random.randint(start, end) for start, end in frame_ranges]


def sample_static_geo_stratified_indices(length, sample_count):
    if sample_count <= 0 or sample_count >= length:
        return list(range(length))
    indices = []
    for bin_index in range(sample_count):
        start = bin_index * length // sample_count
        end = (bin_index + 1) * length // sample_count - 1
        indices.append(random.randint(start, max(start, end)))
    return indices


def sample_static_geo_view_indices(num_cameras, sample_count, contiguous):
    if sample_count <= 0 or sample_count >= num_cameras:
        return list(range(num_cameras))
    if not contiguous:
        return sample_static_geo_indices(num_cameras, sample_count)
    start = random.randrange(num_cameras)
    return [(start + offset) % num_cameras for offset in range(sample_count)]


def geovideo_voxel_simplify_and_denoise(
    points,
    frame_count,
    nearest_neighbor_quantile,
    outlier_neighbor_count,
    outlier_std_ratio,
    knn_workers,
    fixed_voxel_size=None,
    outlier_backend="scipy",
    gpu_knn_query_chunk_size=262144,
    gpu_sor_search_radius=2.0,
    return_stats=False,
    return_debug=False,
):
    """Apply the adaptive voxel and training-time SOR described by GeoVideo."""
    if points.shape[0] < 2:
        result = points
        stats = {
            "input_points": int(points.shape[0]),
            "voxel_points": int(points.shape[0]),
            "output_points": int(points.shape[0]),
            "nearest_seconds": 0.0,
            "voxel_seconds": 0.0,
            "outlier_seconds": 0.0,
            "fallback_points": 0,
        }
        return (result, stats) if return_stats else result

    cKDTree = None
    if fixed_voxel_size is None or outlier_backend == "scipy":
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise RuntimeError(
                "GeoVideo adaptive voxel or CPU SOR requires scipy.spatial.cKDTree."
            ) from exc

    if fixed_voxel_size is None:
        nearest_start = time.perf_counter()
        detached_points = points.detach().float().cpu().numpy()
        nearest_distances, _ = cKDTree(detached_points).query(
            detached_points,
            k=2,
            workers=int(knn_workers),
        )
        representative_distance = float(
            torch.quantile(
                torch.from_numpy(nearest_distances[:, 1]),
                float(nearest_neighbor_quantile),
            ).item()
        )
        voxel_size = max(
            float(frame_count) ** (1.0 / 3.0) * representative_distance,
            1.0e-6,
        )
        nearest_seconds = time.perf_counter() - nearest_start
    else:
        voxel_size = max(float(fixed_voxel_size), 1.0e-6)
        nearest_seconds = 0.0

    voxel_start = time.perf_counter()
    voxel_indices = torch.floor(points.detach().float() / voxel_size).to(torch.int64)
    _, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
    voxel_count = int(inverse_indices.max().item()) + 1
    centroid_sum = points.new_zeros((voxel_count, 3))
    centroid_sum.index_add_(0, inverse_indices, points)
    centroid_count = points.new_zeros((voxel_count, 1))
    centroid_count.index_add_(
        0,
        inverse_indices,
        points.new_ones((points.shape[0], 1)),
    )
    centroids = centroid_sum / centroid_count.clamp_min(1.0)
    if centroids.is_cuda:
        torch.cuda.synchronize(centroids.device)
    voxel_seconds = time.perf_counter() - voxel_start

    neighbor_count = min(int(outlier_neighbor_count), centroids.shape[0] - 1)
    if neighbor_count <= 0:
        result = centroids
        stats = {
            "input_points": int(points.shape[0]),
            "voxel_points": int(centroids.shape[0]),
            "output_points": int(result.shape[0]),
            "nearest_seconds": nearest_seconds,
            "voxel_seconds": voxel_seconds,
            "outlier_seconds": 0.0,
            "fallback_points": 0,
        }
        return (result, stats) if return_stats else result
    outlier_start = time.perf_counter()
    fallback_point_count = 0
    if outlier_backend == "scipy":
        detached_centroids = centroids.detach().float().cpu().numpy()
        neighbor_distances, _ = cKDTree(detached_centroids).query(
            detached_centroids,
            k=neighbor_count + 1,
            workers=int(knn_workers),
        )
        mean_neighbor_distance = torch.from_numpy(
            neighbor_distances[:, 1:].mean(axis=1)
        )
    elif outlier_backend in ("open3d_cuda", "open3d_cuda_hybrid"):
        if not centroids.is_cuda:
            raise RuntimeError("open3d_cuda SOR requires CUDA centroids.")
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("open3d_cuda SOR requires Open3D.") from exc
        if not o3d.core.cuda.is_available():
            raise RuntimeError("The installed Open3D build has no CUDA support.")
        detached_centroids = centroids.detach().float().contiguous()
        open3d_centroids = o3d.core.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(detached_centroids)
        )
        mean_neighbor_distance = torch.empty(
            detached_centroids.shape[0],
            device=detached_centroids.device,
            dtype=torch.float32,
        )
        if outlier_backend == "open3d_cuda":
            neighbor_search = o3d.core.nns.NearestNeighborSearch(open3d_centroids)
            if not neighbor_search.knn_index():
                raise RuntimeError("Open3D failed to build the CUDA KNN index.")
            query_chunk_size = max(int(gpu_knn_query_chunk_size), 1)
            for query_start in range(0, detached_centroids.shape[0], query_chunk_size):
                query_end = min(
                    query_start + query_chunk_size,
                    detached_centroids.shape[0],
                )
                _, squared_distances = neighbor_search.knn_search(
                    open3d_centroids[query_start:query_end],
                    neighbor_count + 1,
                )
                torch_squared_distances = torch.utils.dlpack.from_dlpack(
                    squared_distances.to_dlpack()
                )
                mean_neighbor_distance[query_start:query_end] = torch.sqrt(
                    torch_squared_distances[:, 1:].clamp_min_(0.0)
                ).mean(dim=1)
        else:
            search_radius = max(float(gpu_sor_search_radius), voxel_size)
            radius_search = o3d.core.nns.NearestNeighborSearch(open3d_centroids)
            if not radius_search.fixed_radius_index(search_radius):
                raise RuntimeError("Open3D failed to build the CUDA radius index.")
            _, squared_distances, neighbor_counts = radius_search.hybrid_search(
                open3d_centroids,
                search_radius,
                neighbor_count + 1,
            )
            torch_squared_distances = torch.utils.dlpack.from_dlpack(
                squared_distances.to_dlpack()
            )
            torch_neighbor_counts = torch.utils.dlpack.from_dlpack(
                neighbor_counts.to_dlpack()
            ).to(torch.int64)
            enough_neighbors = torch_neighbor_counts >= neighbor_count + 1
            mean_neighbor_distance[enough_neighbors] = torch.sqrt(
                torch_squared_distances[enough_neighbors, 1:].clamp_min_(0.0)
            ).mean(dim=1)
            unresolved_indices = torch.nonzero(
                ~enough_neighbors,
                as_tuple=False,
            ).flatten()
            fallback_point_count = int(unresolved_indices.numel())
            if unresolved_indices.numel() > 0:
                exact_search = o3d.core.nns.NearestNeighborSearch(open3d_centroids)
                if not exact_search.knn_index():
                    raise RuntimeError("Open3D failed to build the fallback KNN index.")
                unresolved_centroids = detached_centroids[
                    unresolved_indices
                ].contiguous()
                open3d_unresolved = o3d.core.Tensor.from_dlpack(
                    torch.utils.dlpack.to_dlpack(unresolved_centroids)
                )
                _, unresolved_squared_distances = exact_search.knn_search(
                    open3d_unresolved,
                    neighbor_count + 1,
                )
                torch_unresolved_squared_distances = torch.utils.dlpack.from_dlpack(
                    unresolved_squared_distances.to_dlpack()
                )
                mean_neighbor_distance[unresolved_indices] = torch.sqrt(
                    torch_unresolved_squared_distances[:, 1:].clamp_min_(0.0)
                ).mean(dim=1)
    else:
        raise ValueError(
            f"Unsupported static-geo outlier backend: {outlier_backend!r}."
        )
    distance_threshold = (
        mean_neighbor_distance.mean()
        + float(outlier_std_ratio) * mean_neighbor_distance.std(unbiased=False)
    )
    keep = (mean_neighbor_distance <= distance_threshold).to(
        device=centroids.device,
        dtype=torch.bool,
    )
    result = centroids[keep]
    if result.is_cuda:
        torch.cuda.synchronize(result.device)
    stats = {
        "input_points": int(points.shape[0]),
        "voxel_points": int(centroids.shape[0]),
        "output_points": int(result.shape[0]),
        "nearest_seconds": nearest_seconds,
        "voxel_seconds": voxel_seconds,
        "outlier_seconds": time.perf_counter() - outlier_start,
        "fallback_points": fallback_point_count,
    }
    if return_debug:
        stats["voxel_centroids"] = centroids.detach().float().cpu()
        stats["outlier_keep_mask"] = keep.detach().cpu()
        stats["mean_neighbor_distance"] = mean_neighbor_distance.float().cpu()
        stats["outlier_distance_threshold"] = float(distance_threshold.item())
        stats["voxel_size"] = float(voxel_size)
    return (result, stats) if return_stats else result


def geovideo_single_view_static_pointcloud_loss(
    pred_x0,
    flow_loss_mask,
    camera_param,
    frame_transform,
    turbo_decoder,
    latent_modalities,
    vae_out_channels,
    num_cameras,
    video_length,
    depth_max,
    latent_scale,
    min_valid_depth,
    sky_depth_threshold,
    static_mask_threshold,
    require_flow_mask,
    sample_frames,
    reprojection_tolerance,
    nearest_neighbor_quantile,
    outlier_neighbor_count,
    outlier_std_ratio,
    knn_workers,
    checkpoint_decode,
    fixed_voxel_size=None,
    outlier_backend="scipy",
    gpu_knn_query_chunk_size=262144,
    gpu_sor_search_radius=2.0,
    record_time=False,
    debug_export_path=None,
    raw_debug_export_path=None,
):
    total_start = time.perf_counter()
    if "depth" not in latent_modalities:
        raise ValueError("static_geo_loss requires 'depth' in latent_modalities.")
    if flow_loss_mask is None and require_flow_mask:
        raise RuntimeError("static_geo_loss requires cached flow_loss_mask for static-pixel selection.")
    if camera_param is None:
        raise RuntimeError("static_geo_loss requires camera_param for point-cloud reprojection.")

    depth_latent = extract_modality_latent(
        pred_x0,
        "depth",
        latent_modalities,
        vae_out_channels,
        num_cameras,
    )
    batch_size = depth_latent.shape[0] // num_cameras
    selected_view = random.randrange(num_cameras)
    depth_latent = rearrange(
        depth_latent,
        "(B NC) C T H W -> B NC C T H W",
        B=batch_size,
        NC=num_cameras,
    )[:, selected_view]
    decode_dtype = next(turbo_decoder.parameters()).dtype
    scaled_depth_latent = (depth_latent * latent_scale).to(dtype=decode_dtype)

    def decode_depth_latent(latent):
        return turbo_decoder.decode(latent, return_dict=False)[0]

    decode_start = time.perf_counter()
    if checkpoint_decode and torch.is_grad_enabled() and scaled_depth_latent.requires_grad:
        from torch.utils.checkpoint import checkpoint as activation_checkpoint

        decoded = activation_checkpoint(
            decode_depth_latent,
            scaled_depth_latent,
            use_reentrant=False,
        )
    else:
        decoded = decode_depth_latent(scaled_depth_latent)
    if decoded.is_cuda:
        torch.cuda.synchronize(decoded.device)
    decode_seconds = time.perf_counter() - decode_start

    pred_metric = decoded_to_metric_depth(decoded, depth_max)
    frames = min(int(video_length), pred_metric.shape[2])
    selected_frames = sample_static_geo_stratified_indices(frames, int(sample_frames))
    selected_frame_tensor = torch.tensor(
        selected_frames,
        device=pred_metric.device,
        dtype=torch.long,
    )
    pred_depth = pred_metric[:, 0, :frames].index_select(1, selected_frame_tensor)
    _, selected_frame_count, height, width = pred_depth.shape

    static_mask = (
        (pred_depth > min_valid_depth) & (pred_depth < sky_depth_threshold)
    ).detach()
    if flow_loss_mask is not None:
        dynamic_mask = rearrange(
            flow_loss_mask.float(),
            "(B NC) 1 T H W -> B NC 1 T H W",
            B=batch_size,
            NC=num_cameras,
        )[:, selected_view]
        dynamic_mask = F.interpolate(
            dynamic_mask,
            size=(frames, height, width),
            mode="trilinear",
            align_corners=False,
        )[:, 0].index_select(1, selected_frame_tensor)
        static_mask = static_mask & (dynamic_mask.detach() < static_mask_threshold)

    camera_param = camera_param[:, :frames].float()
    intrinsics = camera_param[..., :3]
    camera2lidar = camera_param_to_camera2lidar(camera_param)
    lidar2camera = torch.linalg.inv(camera2lidar)
    if frame_transform is None:
        frame_transform = torch.eye(
            4,
            device=camera_param.device,
            dtype=torch.float32,
        ).view(1, 1, 4, 4).repeat(batch_size, frames, 1, 1)
    else:
        frame_transform = frame_transform[:, :frames].float()
    top2frame = frame_transform
    frame2top = torch.linalg.inv(top2frame)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=pred_depth.device, dtype=torch.float32),
        torch.arange(width, device=pred_depth.device, dtype=torch.float32),
        indexing="ij",
    )
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    flat_ones = torch.ones_like(flat_x)
    pixel_count = height * width

    loss_numerator = pred_depth.new_tensor(0.0)
    valid_denominator = pred_depth.new_tensor(0.0)
    reliable_pixel_count = pred_depth.new_tensor(0.0)
    possible_pixel_count = float(batch_size * selected_frame_count * pixel_count)

    point_build_seconds = 0.0
    nearest_seconds = 0.0
    voxel_seconds = 0.0
    outlier_seconds = 0.0
    reprojection_seconds = 0.0
    input_point_count = 0
    voxel_point_count = 0
    output_point_count = 0
    fallback_point_count = 0

    for batch_index in range(batch_size):
        point_build_start = time.perf_counter()
        distributed_rank = dist.get_rank() if dist.is_initialized() else 0
        raw_debug_pattern = (
            str(raw_debug_export_path) if raw_debug_export_path else ""
        )
        raw_export_all_ranks = "{rank}" in raw_debug_pattern
        resolved_raw_debug_path = (
            raw_debug_pattern.format(
                rank=distributed_rank,
                batch_index=batch_index,
            )
            if raw_debug_pattern
            else ""
        )
        should_export_raw = (
            bool(resolved_raw_debug_path)
            and batch_index == 0
            and (raw_export_all_ranks or distributed_rank == 0)
            and not Path(resolved_raw_debug_path).exists()
        )
        frame_pointclouds = []
        point_frame_indices = []
        for local_frame, frame_index in enumerate(selected_frames):
            flat_depth = pred_depth[batch_index, local_frame].reshape(-1)
            flat_static = static_mask[batch_index, local_frame].reshape(-1)
            if not flat_static.any():
                continue
            depth = flat_depth[flat_static]
            pixel_x = flat_x[flat_static]
            pixel_y = flat_y[flat_static]
            intrinsic = intrinsics[batch_index, frame_index, selected_view]
            point_camera = torch.stack(
                [
                    (pixel_x - intrinsic[0, 2]) / intrinsic[0, 0].clamp_min(1.0e-6) * depth,
                    (pixel_y - intrinsic[1, 2]) / intrinsic[1, 1].clamp_min(1.0e-6) * depth,
                    depth,
                    flat_ones[flat_static],
                ],
                dim=-1,
            )
            point_lidar = torch.matmul(
                camera2lidar[batch_index, frame_index, selected_view],
                point_camera.unsqueeze(-1),
            ).squeeze(-1)
            point_top = torch.matmul(
                frame2top[batch_index, frame_index],
                point_lidar.unsqueeze(-1),
            ).squeeze(-1)
            frame_pointclouds.append(point_top[:, :3])
            if should_export_raw:
                point_frame_indices.append(
                    torch.full(
                        (point_top.shape[0],),
                        frame_index,
                        dtype=torch.int16,
                    )
                )
        if not frame_pointclouds:
            continue

        global_pointcloud = torch.cat(frame_pointclouds, dim=0)
        if should_export_raw:
            Path(resolved_raw_debug_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "rank": distributed_rank,
                    "selected_view": selected_view,
                    "selected_frames": selected_frames,
                    "points": global_pointcloud.detach().float().cpu(),
                    "frame_indices": torch.cat(point_frame_indices, dim=0),
                },
                resolved_raw_debug_path,
            )
        if global_pointcloud.is_cuda:
            torch.cuda.synchronize(global_pointcloud.device)
        point_build_seconds += time.perf_counter() - point_build_start
        debug_export_pattern = str(debug_export_path) if debug_export_path else ""
        export_all_ranks = "{rank}" in debug_export_pattern
        resolved_debug_export_path = (
            debug_export_pattern.format(
                rank=distributed_rank,
                batch_index=batch_index,
            )
            if debug_export_pattern
            else ""
        )
        should_export_debug = (
            bool(resolved_debug_export_path)
            and batch_index == 0
            and (export_all_ranks or distributed_rank == 0)
            and not Path(resolved_debug_export_path).exists()
        )
        global_pointcloud, pointcloud_stats = geovideo_voxel_simplify_and_denoise(
            global_pointcloud,
            selected_frame_count,
            nearest_neighbor_quantile,
            outlier_neighbor_count,
            outlier_std_ratio,
            knn_workers,
            fixed_voxel_size=fixed_voxel_size,
            outlier_backend=outlier_backend,
            gpu_knn_query_chunk_size=gpu_knn_query_chunk_size,
            gpu_sor_search_radius=gpu_sor_search_radius,
            return_stats=True,
            return_debug=should_export_debug,
        )
        input_point_count += pointcloud_stats["input_points"]
        voxel_point_count += pointcloud_stats["voxel_points"]
        output_point_count += pointcloud_stats["output_points"]
        fallback_point_count += pointcloud_stats["fallback_points"]
        nearest_seconds += pointcloud_stats["nearest_seconds"]
        voxel_seconds += pointcloud_stats["voxel_seconds"]
        outlier_seconds += pointcloud_stats["outlier_seconds"]
        if should_export_debug:
            Path(resolved_debug_export_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "rank": distributed_rank,
                    "selected_view": selected_view,
                    "selected_frames": selected_frames,
                    "input_point_count": pointcloud_stats["input_points"],
                    "voxel_point_count": pointcloud_stats["voxel_points"],
                    "output_point_count": pointcloud_stats["output_points"],
                    "voxel_centroids": pointcloud_stats["voxel_centroids"],
                    "outlier_keep_mask": pointcloud_stats["outlier_keep_mask"],
                    "mean_neighbor_distance": pointcloud_stats["mean_neighbor_distance"],
                    "outlier_distance_threshold": pointcloud_stats[
                        "outlier_distance_threshold"
                    ],
                    "voxel_size": pointcloud_stats["voxel_size"],
                    "predicted_depth": pred_depth[batch_index].detach().float().cpu(),
                    "static_mask": static_mask[batch_index].detach().cpu(),
                },
                resolved_debug_export_path,
            )
        if global_pointcloud.shape[0] == 0:
            continue
        homogeneous_global = torch.cat(
            [global_pointcloud, global_pointcloud.new_ones((global_pointcloud.shape[0], 1))],
            dim=-1,
        )

        reprojection_start = time.perf_counter()
        for local_frame, frame_index in enumerate(selected_frames):
            point_lidar = torch.matmul(
                top2frame[batch_index, frame_index],
                homogeneous_global.unsqueeze(-1),
            ).squeeze(-1)
            point_camera = torch.matmul(
                lidar2camera[batch_index, frame_index, selected_view],
                point_lidar.unsqueeze(-1),
            ).squeeze(-1)
            projected_depth = point_camera[:, 2]
            intrinsic = intrinsics[batch_index, frame_index, selected_view]
            projected_x = (
                point_camera[:, 0] / projected_depth.clamp_min(1.0e-6) * intrinsic[0, 0]
                + intrinsic[0, 2]
            )
            projected_y = (
                point_camera[:, 1] / projected_depth.clamp_min(1.0e-6) * intrinsic[1, 1]
                + intrinsic[1, 2]
            )
            point_valid = (
                (projected_depth > min_valid_depth)
                & (projected_x >= 0.0)
                & (projected_x <= width - 1)
                & (projected_y >= 0.0)
                & (projected_y <= height - 1)
            ).detach()
            pixel_index = (
                projected_y.round().long().clamp(0, height - 1) * width
                + projected_x.round().long().clamp(0, width - 1)
            )
            depth_sum = pred_depth.new_zeros(pixel_count)
            point_count = pred_depth.new_zeros(pixel_count)
            depth_sum.scatter_add_(
                0,
                pixel_index,
                projected_depth * point_valid.to(dtype=pred_depth.dtype),
            )
            point_count.scatter_add_(
                0,
                pixel_index,
                point_valid.to(dtype=pred_depth.dtype),
            )
            target_depth = pred_depth[batch_index, local_frame].reshape(-1)
            target_static = static_mask[batch_index, local_frame].reshape(-1)
            pixel_valid = (point_count > 0) & target_static
            valid_count = pixel_valid.sum()
            if valid_count.item() == 0:
                continue
            reprojected_depth = depth_sum / point_count.clamp_min(1.0)
            normalized_error = (reprojected_depth - target_depth).abs() / depth_max
            reliable = (
                normalized_error.detach() < reprojection_tolerance
            ) & pixel_valid
            loss_numerator = loss_numerator + normalized_error[reliable].sum()
            valid_denominator = valid_denominator + valid_count.to(pred_depth.dtype)
            reliable_pixel_count = reliable_pixel_count + reliable.sum().to(pred_depth.dtype)
        if pred_depth.is_cuda:
            torch.cuda.synchronize(pred_depth.device)
        reprojection_seconds += time.perf_counter() - reprojection_start

    if valid_denominator.item() == 0:
        loss = pred_depth.sum() * 0.0
        reliable_ratio = pred_depth.new_tensor(0.0)
    else:
        loss = loss_numerator / valid_denominator
        reliable_ratio = reliable_pixel_count / max(possible_pixel_count, 1.0)
    if record_time:
        distributed_rank = dist.get_rank() if dist.is_initialized() else 0
        logging.info(
            "GeoVideo static-geo timing: rank=%d backend=%s view=%d frames=%s "
            "points=%d->%d->%d fallback=%d "
            "decode=%.3fs point_build=%.3fs nearest=%.3fs voxel=%.3fs "
            "outlier=%.3fs reproject=%.3fs total=%.3fs",
            distributed_rank,
            outlier_backend,
            selected_view,
            selected_frames,
            input_point_count,
            voxel_point_count,
            output_point_count,
            fallback_point_count,
            decode_seconds,
            point_build_seconds,
            nearest_seconds,
            voxel_seconds,
            outlier_seconds,
            reprojection_seconds,
            time.perf_counter() - total_start,
        )
        if distributed_rank != 0:
            print(
                "GeoVideo static-geo timing: "
                f"rank={distributed_rank} backend={outlier_backend} "
                f"view={selected_view} frames={selected_frames} "
                f"points={input_point_count}->{voxel_point_count}->{output_point_count} "
                f"fallback={fallback_point_count} decode={decode_seconds:.3f}s "
                f"point_build={point_build_seconds:.3f}s nearest={nearest_seconds:.3f}s "
                f"voxel={voxel_seconds:.3f}s outlier={outlier_seconds:.3f}s "
                f"reproject={reprojection_seconds:.3f}s "
                f"total={time.perf_counter() - total_start:.3f}s",
                flush=True,
            )
    return loss, reliable_ratio


def static_geo_warp_consistency_loss(
    pred_x0,
    flow_loss_mask,
    camera_param,
    frame_transform,
    turbo_decoder,
    latent_modalities,
    vae_out_channels,
    num_cameras,
    video_length,
    depth_max,
    latent_scale,
    min_valid_depth,
    sky_depth_threshold,
    static_mask_threshold,
    require_flow_mask,
    grid_stride,
    reprojection_tolerance,
    sample_views,
    contiguous_sample_views,
    middle_frame_weight,
    late_frame_weight,
    checkpoint_decode,
):
    if "depth" not in latent_modalities:
        raise ValueError("static_geo_loss requires 'depth' in latent_modalities.")
    if flow_loss_mask is None and require_flow_mask:
        raise RuntimeError("static_geo_loss requires cached flow_loss_mask for static-pixel selection.")
    if camera_param is None:
        raise RuntimeError("static_geo_loss requires camera_param for geometric warping.")

    depth_latent = extract_modality_latent(
        pred_x0,
        "depth",
        latent_modalities,
        vae_out_channels,
        num_cameras,
    )
    if depth_latent.shape[0] % num_cameras != 0:
        raise ValueError(
            f"Expected depth latent batch to be divisible by num_cameras={num_cameras}, "
            f"got {depth_latent.shape[0]}."
        )
    batch_size = depth_latent.shape[0] // num_cameras
    selected_views = sample_static_geo_view_indices(
        num_cameras,
        int(sample_views),
        bool(contiguous_sample_views),
    )
    selected_view_count = len(selected_views)
    selected_view_tensor = torch.tensor(
        selected_views,
        device=depth_latent.device,
        dtype=torch.long,
    )
    depth_latent = rearrange(
        depth_latent,
        "(B NC) C T H W -> B NC C T H W",
        B=batch_size,
        NC=num_cameras,
    ).index_select(1, selected_view_tensor)
    depth_latent = rearrange(depth_latent, "B NV C T H W -> (B NV) C T H W").contiguous()
    decode_dtype = next(turbo_decoder.parameters()).dtype
    scaled_depth_latent = (depth_latent * latent_scale).to(dtype=decode_dtype)

    def decode_depth_latent(latent):
        return turbo_decoder.decode(latent, return_dict=False)[0]

    if checkpoint_decode and torch.is_grad_enabled() and scaled_depth_latent.requires_grad:
        from torch.utils.checkpoint import checkpoint as activation_checkpoint

        decoded = activation_checkpoint(
            decode_depth_latent,
            scaled_depth_latent,
            use_reentrant=False,
        )
    else:
        decoded = decode_depth_latent(scaled_depth_latent)
    pred_metric = decoded_to_metric_depth(decoded, depth_max)
    frames = min(int(video_length), pred_metric.shape[2])
    pred_metric = pred_metric[:, :, :frames]
    _, _, _, height, width = pred_metric.shape
    pred_depth = rearrange(pred_metric, "(B NV) 1 T H W -> B NV T H W", NV=selected_view_count)

    static_mask = torch.ones_like(pred_depth, dtype=torch.bool)
    valid_depth = (pred_depth > min_valid_depth) & (pred_depth < sky_depth_threshold)
    static_mask = static_mask & valid_depth.detach()
    if flow_loss_mask is not None:
        dynamic_mask = rearrange(
            flow_loss_mask.float(),
            "(B NC) 1 T H W -> B NC 1 T H W",
            B=batch_size,
            NC=num_cameras,
        ).index_select(1, selected_view_tensor)
        dynamic_mask = rearrange(dynamic_mask, "B NV 1 T H W -> (B NV) 1 T H W").contiguous()
        dynamic_mask = F.interpolate(
            dynamic_mask,
            size=(frames, height, width),
            mode="trilinear",
            align_corners=False,
        )
        dynamic_mask = rearrange(dynamic_mask, "(B NV) 1 T H W -> B NV T H W", NV=selected_view_count)
        static_mask = static_mask & (dynamic_mask.detach() < static_mask_threshold)

    camera_param = camera_param[:, :frames].float()
    intrinsics = camera_param[..., :3]
    camera2lidar = camera_param_to_camera2lidar(camera_param)
    lidar2camera = torch.linalg.inv(camera2lidar)
    if frame_transform is None:
        frame_transform = torch.eye(
            4,
            device=camera_param.device,
            dtype=torch.float32,
        ).view(1, 1, 4, 4).repeat(camera_param.shape[0], frames, 1, 1)
    else:
        frame_transform = frame_transform[:, :frames].float()
    top2frame = frame_transform
    frame2top = torch.linalg.inv(top2frame)

    anchor_frame, middle_frame, late_frame = sample_static_geo_stratified_frames(frames)
    source_frames = (
        (middle_frame, float(middle_frame_weight)),
        (late_frame, float(late_frame_weight)),
    )

    stride = max(1, int(grid_stride))
    y_coords = torch.arange(0, height, stride, device=pred_metric.device, dtype=torch.float32)
    x_coords = torch.arange(0, width, stride, device=pred_metric.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    xx = xx.flatten()
    yy = yy.flatten()
    ones = torch.ones_like(xx)
    pixel_count = height * width

    loss_sum = pred_metric.new_tensor(0.0)
    weight_sum = pred_metric.new_tensor(0.0)
    valid_ratio_sum = pred_metric.new_tensor(0.0)
    pair_weight_sum = 0.0

    for src_t, pair_weight in source_frames:
        if pair_weight <= 0:
            continue
        for view_position, view_index in enumerate(selected_views):
            sampled_src_depth = pred_depth[
                :, view_position, src_t, yy.long(), xx.long()
            ]
            src_static = static_mask[
                :, view_position, src_t, yy.long(), xx.long()
            ]
            src_k = intrinsics[:, src_t, view_index]
            src_x = (
                (xx[None] - src_k[:, 0, 2:3])
                / src_k[:, 0, 0:1].clamp_min(1.0e-6)
            )
            src_y = (
                (yy[None] - src_k[:, 1, 2:3])
                / src_k[:, 1, 1:2].clamp_min(1.0e-6)
            )
            src_points_cam = torch.stack(
                [
                    src_x * sampled_src_depth,
                    src_y * sampled_src_depth,
                    sampled_src_depth,
                    ones[None].expand_as(sampled_src_depth),
                ],
                dim=-1,
            )
            src_points_lidar = torch.matmul(
                camera2lidar[:, src_t, view_index].unsqueeze(1),
                src_points_cam.unsqueeze(-1),
            ).squeeze(-1)
            src_points_top = torch.matmul(
                frame2top[:, src_t].unsqueeze(1),
                src_points_lidar.unsqueeze(-1),
            ).squeeze(-1)
            anchor_points_lidar = torch.matmul(
                top2frame[:, anchor_frame].unsqueeze(1),
                src_points_top.unsqueeze(-1),
            ).squeeze(-1)
            anchor_points_cam = torch.matmul(
                lidar2camera[:, anchor_frame, view_index].unsqueeze(1),
                anchor_points_lidar.unsqueeze(-1),
            ).squeeze(-1)

            anchor_z = anchor_points_cam[..., 2]
            anchor_k = intrinsics[:, anchor_frame, view_index]
            anchor_u = (
                anchor_points_cam[..., 0]
                / anchor_z.clamp_min(1.0e-6)
                * anchor_k[:, 0, 0:1]
                + anchor_k[:, 0, 2:3]
            )
            anchor_v = (
                anchor_points_cam[..., 1]
                / anchor_z.clamp_min(1.0e-6)
                * anchor_k[:, 1, 1:2]
                + anchor_k[:, 1, 2:3]
            )
            in_view = (
                (anchor_z > min_valid_depth)
                & (anchor_u >= 0.0)
                & (anchor_u <= width - 1)
                & (anchor_v >= 0.0)
                & (anchor_v <= height - 1)
            )
            anchor_u_idx = anchor_u.round().long().clamp(0, width - 1)
            anchor_v_idx = anchor_v.round().long().clamp(0, height - 1)
            anchor_pixel = anchor_v_idx * width + anchor_u_idx
            point_valid = src_static & in_view.detach()

            reprojected_depth = pred_metric.new_zeros(batch_size, pixel_count)
            reprojected_count = pred_metric.new_zeros(batch_size, pixel_count)
            reprojected_depth.scatter_add_(
                1,
                anchor_pixel,
                anchor_z * point_valid.to(dtype=pred_metric.dtype),
            )
            reprojected_count.scatter_add_(
                1,
                anchor_pixel,
                point_valid.to(dtype=pred_metric.dtype),
            )

            anchor_depth = pred_depth[
                :, view_position, anchor_frame
            ].reshape(batch_size, pixel_count)
            anchor_static = static_mask[
                :, view_position, anchor_frame
            ].reshape(batch_size, pixel_count)
            pixel_valid = (reprojected_count > 0) & anchor_static
            if not pixel_valid.any():
                continue

            d_hat = reprojected_depth / reprojected_count.clamp_min(1.0)
            depth_error = (d_hat - anchor_depth).abs() / depth_max
            reliable_weight = (
                depth_error.detach() < reprojection_tolerance
            ).to(dtype=pred_metric.dtype)
            valid_weight = (
                pixel_valid.to(dtype=pred_metric.dtype)
                * reliable_weight
                * pair_weight
            )
            valid_count = valid_weight.sum()
            if valid_count.item() == 0:
                continue
            loss_sum = loss_sum + (depth_error * valid_weight).sum()
            weight_sum = weight_sum + valid_count
            valid_ratio_sum = valid_ratio_sum + valid_weight.mean()
            pair_weight_sum += pair_weight

    if weight_sum.item() == 0:
        return pred_metric.sum() * 0.0, pred_metric.new_tensor(0.0)
    return (
        loss_sum / weight_sum.clamp_min(1.0),
        valid_ratio_sum / max(pair_weight_sum, 1.0e-6),
    )


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)
    if cfg.get("vsdebug", False):
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')
        cfg.record_time = True
    enable_debug = cfg.get("debug", False)
    if enable_debug:
        cfg.outputs = os.path.join(cfg.get("outputs", "outputs"), "debug")
        cfg.ckpt_every = 50
        cfg.record_time = True
    verbose_mode = cfg.get("verbose_mode", False)
    if verbose_mode:
        cfg.record_time = True
    record_time = cfg.get("record_time", False)

    # data config
    if cfg.num_frames is None:  # variable length dataset!
        num_data_cfgs = len(cfg.data_cfg_names)
        # print(num_data_cfgs)
        datasets = []
        val_datasets = []
        for idx, (res, data_cfg_name) in enumerate(cfg.data_cfg_names):
            overrides = cfg.get("dataset_cfg_overrides", [[]] * num_data_cfgs)[idx]
            dataset, val_dataset = merge_dataset_cfg(cfg, data_cfg_name, overrides)
            datasets.append((res, dataset))
            val_datasets.append((res, val_dataset))
        cfg.dataset = {"type": "NuScenesMultiResDataset", "cfg": datasets}
        cfg.val_dataset = {"type": "NuScenesMultiResDataset", "cfg": val_datasets}
    else:  # single dataset!
        cfg.dataset, cfg.val_dataset = merge_dataset_cfg(
            cfg, cfg.data_cfg_name, cfg.get("dataset_cfg_overrides", []),
            cfg.num_frames)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    if USE_NPU:  # disable some kernels
        if mmengine_conf_get(cfg, "text_encoder.shardformer", None):
            mmengine_conf_set(cfg, "text_encoder.shardformer", False)
        if mmengine_conf_get(cfg, "model.bbox_embedder_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.bbox_embedder_param.enable_xformers", False)
        if mmengine_conf_get(cfg, "model.frame_emb_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.frame_emb_param.enable_xformers", False)

    # == colossalai init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))
    torch.cuda.manual_seed_all(cfg.get("seed", 1024))
    coordinator = DistCoordinator()
    # a bug with DistCoordinator
    coordinator._local_rank = int(coordinator._local_rank)
    device = get_current_device()

    # == init exp_dir ==
    if cfg.get("overfit", None) is not None:
        cfg.tag = f"{cfg.tag}_" if cfg.get("tag", "") != "" else ""
        cfg.tag += "overfit-" + str(cfg.get("overfit", None))
    exp_name, exp_dir = define_experiment_workspace(cfg, use_date=True)
    coordinator.block_all()
    if coordinator.is_node_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # == init logger, tensorboard & wandb ==
    logger = reset_logger(exp_dir, enable_debug)
    logger.info("Experiment directory created at %s", exp_dir)
    # logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    logger.info(f"ColossalAI version: {colossalai.__version__}")
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)

    # == init ColossalAI booster ==
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
        reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
        # NOTE: do not enable this, precision do not match.
        overlap_allgather=cfg.get("overlap_allgather", False),
        verbose=verbose_mode,
    )
    booster = Booster(plugin=plugin)
    torch.set_num_threads(1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    if cfg.get("overfit", None) is not None:
        _overfit_idxs = random.sample(range(len(dataset)), cfg.overfit)
        logger.info(f"Overfit on: {_overfit_idxs}")
        overfit_idxs = []
        for _ in range(cfg.epochs):
            overfit_idxs += _overfit_idxs
            random.shuffle(_overfit_idxs)
        cfg.epochs = 1
        dataset = torch.utils.data.Subset(dataset, overfit_idxs)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=cfg.get("shuffle", True) if cfg.get("overfit", None) is None else False,
        drop_last=True,
        pin_memory=cfg.get("pin_memory", True), #mod
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
        bucket_sample_ratios=cfg.get("bucket_sample_ratios", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    # val
    if cfg.get("overfit", None) is not None:
        # first n samples, actually this is all unique samples.
        val_dataset = torch.utils.data.Subset(dataset, list(range(cfg.overfit)))
    else:
        if cfg.get("validation_on_train", False):
            logger.info("Using training dataset config for validation samples.")
            val_dataset = build_module(cfg.dataset, DATASETS)
        else:
            if cfg.get("disable_val_latent_cache", False):
                val_dataset_cfg = deepcopy(cfg.val_dataset)
                for dataset_cfg in val_dataset_cfg["cfg"]:
                    dataset_cfg[1].pop("latent_manifest_path", None)
                    dataset_cfg[1]["skip_refine_depth"] = True
            else:
                val_dataset_cfg = cfg.val_dataset
            val_dataset = build_module(val_dataset_cfg, DATASETS)
        if cfg.val.validation_index != "all":
            if len(cfg.val.validation_index) < get_data_parallel_group().size():
                if isinstance(cfg.val.validation_index[0], int):
                    # we use max world size 32 before, keep the same.
                    cfg.val.validation_index += random.sample(
                        list(set(range(len(val_dataset))) - set(cfg.val.validation_index)),
                        min(get_data_parallel_group().size(), 32) - len(cfg.val.validation_index),
                    )
                    # for larger than 32, add them one-by-one.
                    if get_data_parallel_group().size() > 32:
                        while len(cfg.val.validation_index) < get_data_parallel_group().size():
                            cfg.val.validation_index += random.sample(
                                list(set(range(len(val_dataset)))
                                     - set(cfg.val.validation_index)), 1,
                            )
                else:
                    while len(cfg.val.validation_index) < get_data_parallel_group().size():
                        new_key = val_dataset.rand_another_key()
                        if new_key not in cfg.val.validation_index:
                            cfg.val.validation_index.append(new_key)
                logging.info(f"validation_index rewrite as: {cfg.val.validation_index}")
            val_dataset = torch.utils.data.Subset(
                val_dataset, cfg.val.validation_index)
        else:
            raise NotImplementedError()
    logger.info("Val Dataset contains %s samples.", len(val_dataset))
    dataloader_args['shuffle'] = False
    dataloader_args['dataset'] = val_dataset
    dataloader_args['batch_size'] = cfg.val.get("batch_size", 1)
    dataloader_args['num_workers'] = cfg.val.get("num_workers", 2)
    val_dataloader, val_sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )

    def collate_data_container_fn(batch, *, collate_fn_map=None):
        return batch
    # add datacontainer handler
    torch.utils.data._utils.collate.default_collate_fn_map.update({
        DataContainer: collate_data_container_fn
    })

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    # NOTE: set to true/false,
    # https://github.com/huggingface/transformers/issues/5486
    # if the program gets stuck, try set it to false
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype)
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length
    else:
        text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
        text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)

    # == build vae ==
    vae = build_module(cfg.get("vae", None), MODELS)
    if vae is not None:
        vae = vae.to(device, dtype).eval()
    # if vae is not None:
    #     input_size = (dataset.num_frames, *dataset.image_size)
    #     latent_size = vae.get_latent_size(input_size)
    #     vae_out_channels = vae.out_channels
    # else:
    latent_size = (None, None, None)
    vae_out_channels = cfg.get("vae_out_channels", 4)
    latent_modalities = tuple(cfg.get("latent_modalities", ("rgb", "depth")))
    modality_loss_weights = cfg.get("modality_loss_weights", None)
    if modality_loss_weights:
        logger.info("Using modality loss weights: %s", dict(modality_loss_weights))
    loss_frame_weights = cfg.get("loss_frame_weights", None)
    if loss_frame_weights:
        logger.info("Using frame loss weights: %s", pformat(loss_frame_weights))
    use_flow_loss_mask = bool(cfg.get("use_flow_loss_mask", False))
    if use_flow_loss_mask:
        if "flow" not in latent_modalities:
            raise ValueError("use_flow_loss_mask=True requires 'flow' in latent_modalities.")
        logger.info("Using cached flow_loss_mask to ignore static/white flow areas.")
    use_fd_rgb = bool(cfg.get("fd_rgb_enabled", False))
    use_fd_rgb_stage4 = bool(cfg.get("fd_rgb_stage4_enabled", False))
    if use_fd_rgb and use_fd_rgb_stage4:
        raise ValueError("Choose either fd_rgb_enabled or fd_rgb_stage4_enabled, not both.")
    fd_rgb_every = int(cfg.get("fd_rgb_every", 4))
    fd_rgb_weight = float(cfg.get("fd_rgb_weight", 0.005))
    fd_rgb_pool_size = int(cfg.get("fd_rgb_pool_size", 4))
    fd_rgb_decode_batch_size = int(cfg.get("fd_rgb_decode_batch_size", 1))
    fd_rgb_decoder_checkpoint = bool(cfg.get("fd_rgb_decoder_checkpoint", True))
    fd_rgb_record_time = bool(cfg.get("fd_rgb_record_time", False))
    fd_rgb_profile_only = bool(cfg.get("fd_rgb_profile_only", False))
    fd_rgb_loss_state = None
    fd_rgb_stage4_loss_state = None
    fd_rgb_stage4_loss_states = None
    fd_rgb_stage4_feature_extractor = None
    fd_rgb_stage4_turbo_decoder = None
    fd_rgb_stage4_decoder_type = "cogvideox"
    fd_rgb_stage4_single_view = False
    fd_rgb_stage4_view_count = 1
    fd_rgb_stage4_frame_lengths = None
    fd_rgb_stage4_event_count = 0
    fd_rgb_stage4_statistics = "queue"
    if use_fd_rgb:
        if "rgb" not in latent_modalities:
            raise ValueError("fd_rgb_enabled=True requires 'rgb' in latent_modalities.")
        if vae is None:
            raise RuntimeError("fd_rgb_enabled=True requires a VAE decoder.")
        if fd_rgb_every <= 0:
            raise ValueError("fd_rgb_every must be positive.")
        vae.requires_grad_(False)
        fd_rgb_feature_dim = 3 * fd_rgb_pool_size * fd_rgb_pool_size
        fd_rgb_loss_state = OnlineFrechetRGBLoss(
            feature_dim=fd_rgb_feature_dim,
            queue_size=int(cfg.get("fd_rgb_queue_size", 128)),
            min_population=int(cfg.get("fd_rgb_min_population", 8)),
            eps=float(cfg.get("fd_rgb_covariance_eps", 1e-4)),
        )
        logger.info(
            "Enabled RGB FD: every=%s weight=%s feature_dim=%s queue=%s min_population=%s decoder_checkpoint=%s",
            fd_rgb_every,
            fd_rgb_weight,
            fd_rgb_feature_dim,
            cfg.get("fd_rgb_queue_size", 128),
            cfg.get("fd_rgb_min_population", 8),
            fd_rgb_decoder_checkpoint,
        )
    if use_fd_rgb_stage4:
        if "rgb" not in latent_modalities:
            raise ValueError("fd_rgb_stage4_enabled=True requires 'rgb' in latent_modalities.")
        if vae is None:
            raise RuntimeError("fd_rgb_stage4_enabled=True requires a VAE decoder.")
        if fd_rgb_every <= 0:
            raise ValueError("fd_rgb_every must be positive.")
        reference_path = cfg.get("fd_rgb_stage4_reference_stats_path", None)
        if not reference_path:
            raise ValueError("fd_rgb_stage4_reference_stats_path is required for Stage4 RGB FD.")
        reference = load_stage4_reference_stats(reference_path, device=device)
        expected_stage4_views = int(cfg.get("fd_rgb_stage4_num_cameras", reference["mu"].shape[0]))
        if int(reference["mu"].shape[0]) != expected_stage4_views:
            raise ValueError(
                f"Stage4 reference stats have {reference['mu'].shape[0]} views, expected {expected_stage4_views}."
            )
        fd_rgb_stage4_single_view = bool(cfg.get("fd_rgb_stage4_single_view", False))
        fd_rgb_stage4_view_count = int(cfg.get("fd_rgb_stage4_view_count", 1 if fd_rgb_stage4_single_view else expected_stage4_views))
        if not 1 <= fd_rgb_stage4_view_count <= expected_stage4_views:
            raise ValueError(f"fd_rgb_stage4_view_count must be in [1,{expected_stage4_views}]")
        if fd_rgb_stage4_single_view and fd_rgb_stage4_view_count != 1:
            raise ValueError("fd_rgb_stage4_single_view and fd_rgb_stage4_view_count>1 are incompatible")
        configured_frame_lengths = cfg.get("fd_rgb_stage4_frame_lengths", None)
        if configured_frame_lengths is not None:
            fd_rgb_stage4_frame_lengths = tuple(int(value) for value in configured_frame_lengths)
            if not fd_rgb_stage4_frame_lengths:
                raise ValueError("fd_rgb_stage4_frame_lengths must not be empty when configured")
        fd_rgb_stage4_decoder_type = str(cfg.get("fd_rgb_stage4_decoder", "cogvideox")).lower()
        if fd_rgb_stage4_decoder_type == "cogvideox":
            vae.requires_grad_(False)
            if not bool(getattr(vae.module, "use_tiling", False)):
                vae.module.enable_tiling()
        elif fd_rgb_stage4_decoder_type == "turbo_vaed_cog":
            fd_rgb_stage4_turbo_decoder = build_fd_rgb_turbo_decoder(cfg, device, dtype, logger)
        else:
            raise ValueError(
                f"unknown fd_rgb_stage4_decoder={fd_rgb_stage4_decoder_type!r}; "
                "expected 'cogvideox' or 'turbo_vaed_cog'"
            )
        feature_mode = str(cfg.get("fd_rgb_stage4_feature_mode", "pooled_rgb"))
        if feature_mode == "styleganv_i3d":
            i3d_checkpoint = cfg.get("fd_rgb_stage4_i3d_checkpoint", None)
            if not i3d_checkpoint:
                raise ValueError("fd_rgb_stage4_i3d_checkpoint is required for styleganv_i3d features")
            fd_rgb_stage4_feature_extractor = StyleGANVI3DFeatureExtractor(
                i3d_checkpoint,
                clip_length=int(cfg.get("fd_rgb_stage4_i3d_clip_length", 16)),
                resolution=int(cfg.get("fd_rgb_stage4_i3d_resolution", 224)),
                batch_size=int(cfg.get("fd_rgb_stage4_feature_batch_size", 1)),
                device=device,
            )
            fd_rgb_stage4_feature_extractor.eval()
        elif feature_mode != "pooled_rgb":
            raise ValueError(f"unknown fd_rgb_stage4_feature_mode={feature_mode!r}")
        fd_rgb_stage4_statistics = str(cfg.get("fd_rgb_stage4_statistics", "queue")).lower()
        if fd_rgb_stage4_statistics not in {"queue", "ema"}:
            raise ValueError("fd_rgb_stage4_statistics must be 'queue' or 'ema'")
        if fd_rgb_stage4_statistics == "ema" and fd_rgb_stage4_view_count != expected_stage4_views:
            raise ValueError("Stage4 EMA statistics currently require all configured camera views")
        queue_size = int(cfg.get("fd_rgb_stage4_queue_size", 128))
        min_population = int(cfg.get("fd_rgb_stage4_min_population", 8))
        covariance_eps = float(cfg.get("fd_rgb_stage4_covariance_eps", 1e-4))
        if fd_rgb_stage4_statistics == "ema":
            fd_rgb_stage4_loss_state = Stage4PerViewFrechetRGBEMALoss(
                reference["mu"],
                reference["cov"],
                decay=float(cfg.get("fd_rgb_stage4_ema_decay", 0.999)),
                eps=covariance_eps,
                normalization_eps=float(cfg.get("fd_rgb_stage4_normalization_eps", 0.01)),
            )
        elif fd_rgb_stage4_single_view or fd_rgb_stage4_view_count < expected_stage4_views:
            fd_rgb_stage4_loss_states = [
                Stage4PerViewFrechetRGBLoss(
                    reference["mu"][view_index : view_index + 1],
                    reference["cov"][view_index : view_index + 1],
                    queue_size=queue_size,
                    min_population=min_population,
                    eps=covariance_eps,
                )
                for view_index in range(expected_stage4_views)
            ]
        else:
            fd_rgb_stage4_loss_state = Stage4PerViewFrechetRGBLoss(
                reference["mu"],
                reference["cov"],
                queue_size=queue_size,
                min_population=min_population,
                eps=covariance_eps,
            )
        initialization_path = cfg.get(
            "fd_rgb_stage4_ema_init_path" if fd_rgb_stage4_statistics == "ema" else "fd_rgb_stage4_queue_path",
            None,
        )
        if initialization_path:
            queue_payload = torch.load(initialization_path, map_location=device)
            queue_features = queue_payload.get("features", queue_payload) if isinstance(queue_payload, dict) else queue_payload
            queue_features = torch.as_tensor(queue_features, device=device, dtype=torch.float32)
            if queue_features.ndim != 3 or queue_features.shape[1:] != reference["mu"].shape:
                raise ValueError(
                    f"Stage4 queue must be [N,{expected_stage4_views},{reference['mu'].shape[1]}], "
                    f"got {tuple(queue_features.shape)}"
                )
            if fd_rgb_stage4_statistics == "queue" and (
                fd_rgb_stage4_single_view or fd_rgb_stage4_view_count < expected_stage4_views
            ):
                for view_index, loss_state in enumerate(fd_rgb_stage4_loss_states):
                    loss_state.prefill(queue_features[:, view_index : view_index + 1])
            else:
                fd_rgb_stage4_loss_state.prefill(queue_features)
            logger.info(
                "Initialized Stage4 generated %s statistics: path=%s shape=%s",
                fd_rgb_stage4_statistics,
                initialization_path,
                tuple(queue_features.shape),
            )
        logger.info(
            "Enabled Stage4 RGB FD: every=%s weight=%s views=%s single_view=%s frames=%s "
            "decoder=%s feature_dim=%s mode=%s statistics=%s queue=%s ema_decay=%s reference=%s",
            fd_rgb_every,
            fd_rgb_weight,
            expected_stage4_views,
            fd_rgb_stage4_single_view,
            fd_rgb_stage4_frame_lengths,
            fd_rgb_stage4_decoder_type,
            reference["mu"].shape[1],
            feature_mode,
            fd_rgb_stage4_statistics,
            queue_size,
            cfg.get("fd_rgb_stage4_ema_decay", None),
            reference_path,
        )
    static_geo_loss_weight = float(cfg.get("static_geo_loss_weight", 0.0))
    use_static_geo_loss = static_geo_loss_weight > 0.0
    static_geo_loss_mode = str(cfg.get("static_geo_loss_mode", "early_anchor"))
    static_geo_loss_require_flow_mask = bool(cfg.get("static_geo_loss_require_flow_mask", True))
    static_geo_lidar_infos = None
    static_geo_sample_token_to_depth = None
    if use_static_geo_loss:
        if "depth" not in latent_modalities:
            raise ValueError("static_geo_loss_weight > 0 requires 'depth' in latent_modalities.")
        logger.info(
            "Using static_geo_loss with frozen Turbo-VAED decoder: mode=%s weight=%s latent_scale=%s require_flow_mask=%s",
            static_geo_loss_mode,
            static_geo_loss_weight,
            cfg.get("static_geo_loss_turbo_latent_scale", 1.0 / COGVIDEOX_SCALING_FACTOR),
            static_geo_loss_require_flow_mask,
        )
        if static_geo_loss_mode == "rdepth_lidar_stage1":
            static_geo_sample_token_to_depth = load_static_geo_depth_root_map(
                cfg.get("static_geo_loss_depth_root_json", cfg.get("depth_root_json", None)),
                logger,
            )
            if float(cfg.get("static_geo_loss_lidar_weight", 0.0)) > 0.0:
                static_geo_lidar_infos = load_static_geo_lidar_infos(
                    cfg.get("static_geo_loss_lidar_ann_file", None),
                    cfg.get("static_geo_loss_lidar_data_root", cfg.get("nus_root", "")),
                    logger,
                )
    static_geo_turbo_decoder = None

    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae_out_channels*len(latent_modalities),
            caption_channels=text_encoder_output_dim,
            model_max_length=text_encoder_model_max_length,
            enable_sequence_parallelism=cfg.get("sp_size", 1) > 1,
        )
        .to(device, dtype)
        .train()
    )
    model.prepare_text_embedding(text_encoder)
    # partial load pretrain (e.g., image pretrain)
    if cfg.get("partial_load", None) and not cfg.get("load", None):
        load_dir = cfg.partial_load
        if os.path.isdir(load_dir):
            from glob import glob
            weight = {}
            for path in glob(os.path.join(load_dir, "model/pytorch_model-*")):
                weight.update(torch.load(path, map_location="cpu"))
        else:
            weight = torch.load(load_dir, map_location="cpu")
        if cfg.get("partial_load_adapt_shapes", False):
            weight = adapt_state_dict_shapes(weight, model)
        missing_keys, unexpected_keys = model.load_state_dict(weight, strict=False)
        logger.info(f"[partial load] Missing keys: {missing_keys}")
        logger.info(f"[partial load] Unexpected keys: {unexpected_keys}")
        del weight, missing_keys, unexpected_keys
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Fix: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel - model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build ema for diffusion model ==
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    ema.eval()
    update_ema(ema, model, decay=0, sharded=False)

    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    if use_static_geo_loss:
        static_geo_turbo_decoder = build_static_geo_turbo_decoder(cfg, device, dtype, logger)

    # == setup optimizer ==
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )

    warmup_steps = cfg.get("warmup_steps", None)
    milestones_lr = cfg.get("milestones_lr", None)

    if warmup_steps is None:
        lr_scheduler = None
    else:
        if milestones_lr is None:
            lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)
        else:
            lr_scheduler = MultiStepWithLinearWarmupLR(
                optimizer, milestones_lr=milestones_lr, warmup_steps=warmup_steps)

    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    resume_global_step = 0
    drop_cond_ratio = cfg.get("drop_cond_ratio", 0.0)
    drop_cond_ratio_t = cfg.get("drop_cond_ratio_t", 0.4)
    running_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=None if cfg.get("start_from_scratch", False) or cfg.get("reset_optimizer", False) else optimizer,
            lr_scheduler=None if cfg.get("reset_lr", False) or cfg.get("start_from_scratch", False) else lr_scheduler,
            sampler=None if cfg.get("start_from_scratch", False) or cfg.get("reset_sampler", False) else sampler,
            local_master=coordinator.is_node_master(),
        )
        if not cfg.get("start_from_scratch", False):
            start_epoch, start_step = ret
            running_states_path = os.path.join(cfg.load, "running_states.json")
            with open(running_states_path, "r") as running_states_file:
                resume_global_step = int(json.load(running_states_file).get("global_step", 0))
            if cfg.get("reset_sampler", False):
                start_epoch = 0
                start_step = 0
            if cfg.get("reset_lr", False) and lr_scheduler:
                lr_scheduler.last_epoch = resume_global_step
        logger.info(
            "Loaded checkpoint %s at epoch %s step %s global_step %s",
            cfg.load,
            start_epoch,
            start_step,
            resume_global_step,
        )
        if use_fd_rgb_stage4 and fd_rgb_stage4_statistics == "ema":
            fd_ema_path = os.path.join(cfg.load, "fd_rgb_stage4_ema.pt")
            if os.path.exists(fd_ema_path):
                fd_rgb_stage4_loss_state.load_state_dict(torch.load(fd_ema_path, map_location=device))
                logger.info("Restored Stage4 RGB FD EMA statistics from %s", fd_ema_path)
            elif resume_global_step > 3600:
                logger.warning(
                    "Checkpoint has no Stage4 RGB FD EMA state; using configured initialization statistics"
                )

    global_step_offset = resume_global_step - (start_epoch * num_steps_per_epoch + start_step)

    if enable_debug:
        save_dir = save(
            booster,
            exp_dir,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=sampler,
            epoch=start_epoch,
            step=start_step,
            global_step=resume_global_step,
            batch_size=cfg.get("batch_size", None),
        )
        logger.info(f"Save your model to {save_dir} before training.")

    model_sharding(ema)

    if cfg.get("validation_before_run", False):
        with RandomStateManager(verbose=True):
            coordinator.block_all()
            run_validation(
                cfg.val,
                text_encoder,
                vae,
                model,
                device,
                dtype,
                val_dataloader,
                coordinator,
                resume_global_step,
                exp_dir,
                cfg.mv_order_map,
                cfg.t_order_map,
            )
            val_sampler.reset()

    with RandomStateManager(verbose=True):
        print(f"{torch.randn(3)} {torch.randn(3, device=get_current_device())} "
              f"on rank {dist.get_rank()} "
              f"dp_rank {dist.get_rank(get_data_parallel_group())}")

    # =======================================================
    # 5. training loop
    # =======================================================
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    coordinator.block_all()
    timers = {}
    timer_keys = [
        "move_data",
        "encode",
        "move_data2",
        "mask",
        "diffusion",
        "backward",
        "update_ema",
        "reduce_loss",
        "misc",
    ]
    for key in timer_keys:
        if record_time:
            timers[key] = Timer(key, coordinator=None)
        else:
            timers[key] = nullcontext()
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                if verbose_mode:
                    logger.info(f"Dataloader returns data! step={step}")
                cached_latent_path = batch.pop("cached_latent_path", None)
                flow_loss_mask = None
                static_geo_flow_loss_mask = None
                static_geo_camera_param = None
                static_geo_frame_transform = None
                if use_static_geo_loss:
                    static_geo_camera_param = batch.get("camera_param", None)
                    if static_geo_camera_param is not None:
                        static_geo_camera_param = static_geo_camera_param.to(device)
                    static_geo_frame_transform = batch.get("frame_emb", None)
                    if isinstance(static_geo_frame_transform, torch.Tensor):
                        static_geo_frame_transform = static_geo_frame_transform.to(device)
                B, T, NC = batch["pixel_values"].shape[:3]
                logging.debug(f"bs = {B}; t = {T}; shape = {batch['pixel_values'].shape}")
                timer_list = []
                with timers["move_data"] as move_data_t:
                    if cached_latent_path is None:
                        x = batch.pop("pixel_values").to(device, dtype)
                        Rdepth = batch["refine_depths"].to(device,dtype)
                        Rdepth = Rdepth.unsqueeze(3)
                        Rdepth = Rdepth.repeat(1, 1, 1, 3, 1, 1)

                        if T>1:
                            x_ref = x[:,0:1]
                            # x = x[:,1:] #T-1
                            x_ref = rearrange(x_ref, "B T NC C ... -> (B NC) C T ...")

                            Rdepth_ref=Rdepth[:,0:1] #B T NC ...
                            Rdepth_ref_zero = torch.zeros(Rdepth_ref.shape).to(device,dtype)
                            Rdepth_ref = Rdepth_ref_zero
                            Rdepth_ref = rearrange(Rdepth_ref, "B T NC C ... -> (B NC) C T ...")

                        x = rearrange(x, "B T NC C ... -> (B NC) C T ...")  # BxNC, C, T, H, W
                        Rdepth = rearrange(Rdepth, "B T NC C ... -> (B NC) C T ...")  # BxNC, C, T, H, W
                        flow_video = None
                        if "flow" in latent_modalities:
                            flow_video = batch.pop("flow_rgb_values").to(device, dtype)
                            flow_video = rearrange(
                                flow_video, "B T NC C ... -> (B NC) C T ..."
                            )
                    else:
                        batch.pop("pixel_values")
                        batch.pop("refine_depths", None)
                        cached_latents, flow_loss_mask = load_cached_latents(
                            cached_latent_path,
                            cfg.get("vae_out_channels", 4),
                            latent_modalities,
                            device,
                            dtype,
                            load_flow_loss_mask=use_flow_loss_mask
                            or (use_static_geo_loss and static_geo_loss_require_flow_mask),
                        )
                        static_geo_flow_loss_mask = flow_loss_mask
                        x, Rdepth = cached_latents[:2]
                        flow_latent = cached_latents[2] if len(cached_latents) > 2 else None
                        if T>1:
                            x_ref = x[:, :, :1]
                            Rdepth_ref = torch.zeros_like(Rdepth[:, :, :1])
                            flow_ref = torch.zeros_like(flow_latent[:, :, :1]) if flow_latent is not None else None

                    y = batch.pop("captions")[0]  # B, just take first frame
                    maps = batch.pop("bev_map_with_aux").to(device, dtype)  # B, T, C, H, W
                    bbox = batch.pop("bboxes_3d_data")
                    # B len list (T, NC=1, len, 8, 3)
                    bbox = [bbox_i.data for bbox_i in bbox]
                    # B, T, NC, len, 8, 3
                    # TODO: `bbox` has redundancy on `NC` dim. They are direct
                    # copies and should be differentiate through mask.
                    bbox = collate_bboxes_to_maxlen(bbox, device, dtype, NC, T)
                    if bbox is not None:
                        bbox = add_box_latent(bbox, B, NC, T, model.module.sample_box_latent)

                        for k, v in bbox.items():
                            bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")  # BxNC, T, len, 3, 7
                    # B, T, NC, 3, 7
                    cams = batch.pop("camera_param").to(device, dtype)
                    cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
                    rel_pos = batch.pop("frame_emb").to(device, dtype)
                    rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)  # BxNC, T, 1, 4, 4
                    # meta_data: T, B
                if record_time:
                    timer_list.append(move_data_t)

                # == visual and text encoding ==
                with timers["encode"] as encode_t:
                    with torch.no_grad():
                        # Prepare visual inputs
                        if cached_latent_path is not None:
                            pass
                        elif cfg.get("load_video_features", False):
                            x = x.to(device, dtype)
                        else:
                            # if USE_NPU:
                            if False:
                                x = vae.encode(x)  # [B, C, T, H/P, W/P]
                            else:
                                with RandomStateManager(verbose=verbose_mode):
                                    # NOTE: due to randomness, they may not match!
                                    x = sp_vae(x, vae.encode,
                                               get_sequence_parallel_group())
                                    Rdepth = sp_vae(Rdepth, vae.encode,
                                               get_sequence_parallel_group())
                                    if flow_video is not None:
                                        flow_latent = sp_vae(
                                            flow_video, vae.encode,
                                            get_sequence_parallel_group())
                                    if T>1:
                                        x_ref = sp_vae(x_ref, vae.encode,
                                                get_sequence_parallel_group())
                                        Rdepth_ref = sp_vae(Rdepth_ref, vae.encode,
                                                get_sequence_parallel_group())
                                        if flow_video is not None:
                                            flow_ref = torch.zeros_like(flow_latent[:, :, :1])
                                        
                            # assert torch.allclose(x_old, x)
                        # Prepare text inputs
                        if cfg.get("load_text_features", False):
                            model_args = {"y": y.to(device, dtype)}
                            mask = batch.pop("mask")
                            if isinstance(mask, torch.Tensor):
                                mask = mask.to(device, dtype)
                            model_args["mask"] = mask
                        else:
                            ret = text_encoder.encode(y)
                            model_args = {k: v for k, v in ret.items()}
                if record_time:
                    timer_list.append(encode_t)
                if verbose_mode:
                    logger.info(f"encoder done! step={step}")

                with timers["move_data2"] as move_data_t:
                    # == unconditionsl mask ==
                    # y -> replace
                    # map -> disable
                    # box -> need mask, on temporal dim
                    # cam/rel_pos -> need mask, on BxNC dim
                    drop_cond_mask = torch.ones((B))  # camera
                    drop_frame_mask = torch.ones((B, T))  # box & rel_pos
                    if drop_cond_ratio > 0:
                        for bs in range(B):
                            # 1. at `drop_cond_ratio`, we drop all conditions
                            # this aligns with `class_dropout_prob` in `CaptionEmbedder`
                            if random.random() < drop_cond_ratio:  # we need drop
                                drop_cond_mask[bs] = 0
                                drop_frame_mask[bs, :] = 0
                                model_args["mask"][bs] = 1  # need to keep all tokens if uncond
                                continue
                            # 2. otherwise, we randomly pick some frames to drop
                            # make sure we do not drop the first and the last frame
                            t_ids = random.sample(
                                range(1, T - 1), int(drop_cond_ratio_t * (T - 2)))
                            drop_frame_mask[bs, t_ids] = 0

                    # == video meta info ==
                    # for k, v in batch.items():
                    #     if isinstance(v, torch.Tensor):
                    #         model_args[k] = v.to(device, dtype)
                    model_args["maps"] = maps
                    model_args["bbox"] = bbox
                    model_args["cams"] = cams
                    model_args["rel_pos"] = rel_pos
                    model_args["drop_cond_mask"] = drop_cond_mask
                    model_args["drop_frame_mask"] = drop_frame_mask
                    model_args["fps"] = batch.pop('fps')
                    model_args["height"] = batch.pop("height")
                    model_args["width"] = batch.pop("width")
                    model_args["num_frames"] = batch.pop("num_frames")

                    if T>1:
                        x_ref = rearrange(x_ref, "(B NC) C T ... -> B T NC C ...", NC=NC)
                        Rdepth_ref = rearrange(Rdepth_ref,"(B NC) C T ... -> B T NC C ...", NC=NC)
                        ref_latents = [x_ref, Rdepth_ref]
                        if "flow" in latent_modalities:
                            flow_ref = rearrange(flow_ref, "(B NC) C T ... -> B T NC C ...", NC=NC)
                            ref_latents.append(flow_ref)
                        x_ref = torch.cat(ref_latents, 3)
                        x_ref = rearrange(x_ref, "B T NC C ... -> B (C NC) T ...", NC=NC)
                        # x_ref = rearrange(x_ref, "(B NC) C T ... -> B (C NC) T ...", NC=NC)
                        model_args["x_ref"]=x_ref

                    model_args = move_to(model_args, device=device, dtype=dtype)
                    # no need to move these
                    model_args["mv_order_map"] = cfg.get("mv_order_map")
                    model_args["t_order_map"] = cfg.get("t_order_map")
  
                if record_time:
                    timer_list.append(move_data_t)

                # == mask ==
                with timers["mask"] as mask_t:
                    # x_mask & scheduler assumes B, C, T dims. we should keep
                    # them as it is. Scheduler further assumes C is the second
                    # (data) dim, T is the third (view) dim.
                    x = rearrange(x, "(B NC) C T ... -> B T NC C ...", NC=NC)
                    Rdepth = rearrange(Rdepth,"(B NC) C T ... -> B T NC C ...", NC=NC)
                    latent_parts = [x, Rdepth]
                    loss_element_mask = None
                    if "flow" in latent_modalities:
                        flow_latent = rearrange(flow_latent, "(B NC) C T ... -> B T NC C ...", NC=NC)
                        latent_parts.append(flow_latent)
                        if use_flow_loss_mask:
                            if flow_loss_mask is None:
                                raise RuntimeError(
                                    "use_flow_loss_mask=True requires cached flow_loss_mask."
                                )
                            flow_loss_mask = rearrange(
                                flow_loss_mask,
                                "(B NC) C T ... -> B T NC C ...",
                                NC=NC,
                            )
                            loss_masks = []
                            for modality in latent_modalities:
                                if modality == "flow":
                                    loss_masks.append(
                                        flow_loss_mask.expand(
                                            -1,
                                            -1,
                                            -1,
                                            vae_out_channels,
                                            -1,
                                            -1,
                                        )
                                    )
                                else:
                                    loss_masks.append(
                                        torch.ones_like(flow_loss_mask).expand(
                                            -1,
                                            -1,
                                            -1,
                                            vae_out_channels,
                                            -1,
                                            -1,
                                        )
                                    )
                            loss_element_mask = torch.cat(loss_masks, 3)
                            loss_element_mask = rearrange(
                                loss_element_mask,
                                "B T NC C ... -> B (C NC) T ...",
                                NC=NC,
                            )
                    x = torch.cat(latent_parts, 3)
                    x = rearrange(x, "B T NC C ... -> B (C NC) T ...", NC=NC)
                    frame_loss_weights = build_frame_loss_weights(
                        loss_frame_weights,
                        T,
                        x.shape[2],
                        device,
                        dtype,
                    )
                    if frame_loss_weights is not None:
                        frame_loss_mask = frame_loss_weights.view(1, 1, x.shape[2], 1, 1).expand(
                            x.shape[0],
                            -1,
                            -1,
                            -1,
                            -1,
                        )
                        if loss_element_mask is None:
                            loss_element_mask = frame_loss_mask
                        else:
                            loss_element_mask = loss_element_mask * frame_loss_mask

                    # x = rearrange(x, "(B NC) C T ... -> B (C NC) T ...", NC=NC)  # B, (C, NC), T, H, W
                    x_fm=x
                    # if T>1:
                    #     x_fm = torch.cat((x_ref,x),axis=2)
                    # else:
                    #     x_fm=x
                    mask = None
                    if cfg.get("mask_ratios", None) is not None:
                        mask_fm = mask_generator.get_masks(x_fm)
                        # if T>1:
                        #     mask = mask_fm[:,1:]
                        # else:
                        #     mask=mask_fm
                        # print(mask.shape)
                        mask=mask_fm
                        model_args["x_mask"] = mask_fm
                if record_time:
                    timer_list.append(mask_t)

                if verbose_mode:
                    logger.info(f"Start model forward step! step={step}")
                # == diffusion loss computation ==
                global_step = epoch * num_steps_per_epoch + step + global_step_offset
                fd_rgb_schedule_due = global_step % fd_rgb_every == 0
                fd_rgb_stage4_length_due = (
                    fd_rgb_stage4_frame_lengths is None or int(T) in fd_rgb_stage4_frame_lengths
                )
                fd_rgb_due = (
                    (use_fd_rgb and fd_rgb_schedule_due)
                    or (use_fd_rgb_stage4 and fd_rgb_schedule_due and fd_rgb_stage4_length_due)
                )
                fd_rgb_real_features = None
                fd_rgb_fake_features = None
                fd_rgb_surrogate = None
                fd_rgb_stage4_selected_view = None
                fd_rgb_stage4_selected_views = None
                fd_rgb_stage4_active_loss_state = None
                fd_rgb_stage4_active_loss_states = None
                fd_rgb_raw_per_view = None
                fd_rgb_normalized_per_view = None
                fd_rgb_timing = {} if fd_rgb_due and fd_rgb_record_time else None
                with timers["diffusion"] as loss_t:
                    if fd_rgb_timing is not None:
                        torch.cuda.synchronize()
                        fd_rgb_timing["model_forward_start"] = time.perf_counter()
                    loss_channel_weights = build_modality_loss_channel_weights(
                        modality_loss_weights,
                        latent_modalities,
                        vae_out_channels,
                        NC,
                        device,
                        dtype,
                    )
                    loss_dict = scheduler.training_losses(
                        model,
                        x,
                        model_args,
                        mask=mask,
                        loss_channel_weights=loss_channel_weights,
                        loss_element_mask=loss_element_mask,
                        return_pred_x0=use_static_geo_loss or fd_rgb_due,
                    )
                    if fd_rgb_timing is not None:
                        torch.cuda.synchronize()
                        fd_rgb_timing["model_forward_and_diffusion"] = (
                            time.perf_counter() - fd_rgb_timing.pop("model_forward_start")
                        )
                    if fd_rgb_due and use_fd_rgb:
                        if fd_rgb_timing is not None:
                            fd_rgb_timing["latent_extract_start"] = time.perf_counter()
                        rgb_target_latent = extract_modality_latent(
                            x,
                            "rgb",
                            latent_modalities,
                            vae_out_channels,
                            NC,
                        )
                        rgb_pred_latent = extract_modality_latent(
                            loss_dict["pred_x0"],
                            "rgb",
                            latent_modalities,
                            vae_out_channels,
                            NC,
                        )
                        if fd_rgb_timing is not None:
                            fd_rgb_timing["latent_extract"] = time.perf_counter() - fd_rgb_timing.pop(
                                "latent_extract_start"
                            )
                            torch.cuda.synchronize()
                            fd_rgb_timing["pred_decode_start"] = time.perf_counter()
                        rgb_pred_decoded = decode_rgb_for_frechet(
                            vae,
                            rgb_pred_latent,
                            fd_rgb_decode_batch_size,
                            fd_rgb_decoder_checkpoint,
                        )
                        if fd_rgb_timing is not None:
                            torch.cuda.synchronize()
                            fd_rgb_timing["pred_decode"] = time.perf_counter() - fd_rgb_timing.pop(
                                "pred_decode_start"
                            )
                            fd_rgb_timing["target_decode_start"] = time.perf_counter()
                        with torch.no_grad():
                            rgb_target_decoded = decode_rgb_for_frechet(
                                vae,
                                rgb_target_latent,
                                fd_rgb_decode_batch_size,
                                False,
                            )
                        if fd_rgb_timing is not None:
                            torch.cuda.synchronize()
                            fd_rgb_timing["target_decode"] = time.perf_counter() - fd_rgb_timing.pop(
                                "target_decode_start"
                            )
                            fd_rgb_timing["feature_stats_start"] = time.perf_counter()
                        fd_rgb_fake_features = build_rgb_frechet_features(
                            rgb_pred_decoded,
                            B,
                            NC,
                            fd_rgb_pool_size,
                        )
                        fd_rgb_real_features = build_rgb_frechet_features(
                            rgb_target_decoded,
                            B,
                            NC,
                            fd_rgb_pool_size,
                        )
                        if fd_rgb_timing is not None:
                            torch.cuda.synchronize()
                            fd_rgb_timing["feature_build"] = time.perf_counter() - fd_rgb_timing.pop(
                                "feature_stats_start"
                            )
                            fd_rgb_timing["fd_stats_start"] = time.perf_counter()
                        fd_rgb_value, fd_rgb_valid = fd_rgb_loss_state(
                            fd_rgb_fake_features,
                            fd_rgb_real_features,
                        )
                        if fd_rgb_timing is not None:
                            torch.cuda.synchronize()
                            fd_rgb_timing["fd_stats"] = time.perf_counter() - fd_rgb_timing.pop(
                                "fd_stats_start"
                            )
                        loss_dict["fd_rgb_loss"] = fd_rgb_value
                        loss_dict["fd_rgb_population_ready"] = torch.tensor(
                            float(fd_rgb_valid), device=device, dtype=torch.float32
                        )
                        if verbose_mode:
                            logger.info(
                                "RGB FD step=%s value=%s population_ready=%s",
                                global_step,
                                fd_rgb_value.detach().float().item(),
                                fd_rgb_valid,
                            )
                        if fd_rgb_timing is not None and dist.get_rank() == 0:
                            logger.info(
                                "[FD-RGB timing] global_step=%s model_forward_and_diffusion=%.3fs "
                                "latent_extract=%.3fs pred_decode=%.3fs target_decode=%.3fs "
                                "feature_build=%.3fs fd_stats=%.3fs",
                                global_step,
                                fd_rgb_timing["model_forward_and_diffusion"],
                                fd_rgb_timing["latent_extract"],
                                fd_rgb_timing["pred_decode"],
                                fd_rgb_timing["target_decode"],
                                fd_rgb_timing["feature_build"],
                                fd_rgb_timing["fd_stats"],
                            )
                        if fd_rgb_profile_only:
                            coordinator.block_all()
                            if dist.get_rank() == 0:
                                logger.info(
                                    "[FD-RGB profile-only] completed forward/statistics at global_step=%s; "
                                    "exiting before backward",
                                    global_step,
                                )
                            raise SystemExit(0)
                    if fd_rgb_due and use_fd_rgb_stage4:
                        if fd_rgb_timing is not None:
                            fd_rgb_timing["latent_extract_start"] = time.perf_counter()
                        rgb_pred_latent = extract_modality_latent(
                            loss_dict["pred_x0"],
                            "rgb",
                            latent_modalities,
                            vae_out_channels,
                            NC,
                        )
                        active_stage4_views = NC
                        if fd_rgb_stage4_single_view or fd_rgb_stage4_view_count < NC:
                            # Count only eligible events so skipped buckets cannot
                            # bias the rotating contiguous camera window.
                            fd_rgb_stage4_selected_view = fd_rgb_stage4_event_count % NC
                            fd_rgb_stage4_event_count += 1
                            fd_rgb_stage4_selected_views = [
                                (fd_rgb_stage4_selected_view + offset) % NC
                                for offset in range(fd_rgb_stage4_view_count)
                            ]
                            indices = torch.tensor(
                                [batch_index * NC + view_index for batch_index in range(B) for view_index in fd_rgb_stage4_selected_views],
                                device=rgb_pred_latent.device,
                            )
                            rgb_pred_latent = rgb_pred_latent.index_select(0, indices)
                            active_stage4_views = fd_rgb_stage4_view_count
                            fd_rgb_stage4_active_loss_states = [
                                fd_rgb_stage4_loss_states[view_index]
                                for view_index in fd_rgb_stage4_selected_views
                            ]
                        else:
                            fd_rgb_stage4_active_loss_state = fd_rgb_stage4_loss_state
                        if fd_rgb_timing is not None:
                            fd_rgb_timing["latent_extract"] = time.perf_counter() - fd_rgb_timing.pop(
                                "latent_extract_start"
                            )
                            torch.cuda.synchronize()
                            fd_rgb_timing["pred_decode_start"] = time.perf_counter()
                        # The detached leaf closes the frozen decoder/feature graph
                        # before the diffusion model's normal backward pass.
                        rgb_fd_boundary = rgb_pred_latent.detach().requires_grad_(True)
                        if fd_rgb_stage4_decoder_type == "turbo_vaed_cog":
                            rgb_pred_decoded = decode_rgb_stage4_turbo(
                                fd_rgb_stage4_turbo_decoder,
                                rgb_fd_boundary,
                                latent_scale=float(
                                    cfg.get(
                                        "fd_rgb_stage4_turbo_latent_scale",
                                        1.0 / COGVIDEOX_SCALING_FACTOR,
                                    )
                                ),
                                checkpoint_decode=bool(
                                    cfg.get("fd_rgb_stage4_turbo_checkpoint_decode", False)
                                ),
                            )[:, :, : int(T)]
                        else:
                            rgb_pred_decoded = decode_rgb_stage4(
                                vae,
                                rgb_fd_boundary,
                                use_bounded_vjp=bool(cfg.get("fd_rgb_stage4_bounded_vjp", True)),
                            )
                        if fd_rgb_timing is not None:
                            torch.cuda.synchronize()
                            fd_rgb_timing["pred_decode"] = time.perf_counter() - fd_rgb_timing.pop(
                                "pred_decode_start"
                            )
                            fd_rgb_timing["feature_stats_start"] = time.perf_counter()
                        if fd_rgb_stage4_feature_extractor is None:
                            fd_rgb_fake_features = build_per_view_rgb_features(
                                rgb_pred_decoded,
                                B,
                                active_stage4_views,
                                fd_rgb_pool_size,
                            )
                        else:
                            fd_rgb_fake_features = fd_rgb_stage4_feature_extractor(rgb_pred_decoded).view(
                                B, active_stage4_views, -1
                            )
                        if fd_rgb_timing is not None:
                            torch.cuda.synchronize()
                            fd_rgb_timing["feature_build"] = time.perf_counter() - fd_rgb_timing.pop(
                                "feature_stats_start"
                            )
                            fd_rgb_timing["fd_stats_start"] = time.perf_counter()
                        if fd_rgb_stage4_statistics == "ema":
                            (
                                fd_rgb_value,
                                fd_rgb_raw_per_view,
                                fd_rgb_normalized_per_view,
                            ) = fd_rgb_stage4_active_loss_state(fd_rgb_fake_features)
                            fd_rgb_valid = True
                        elif fd_rgb_stage4_active_loss_states is not None:
                            per_view_results = [
                                state(fd_rgb_fake_features[:, local_index : local_index + 1])
                                for local_index, state in enumerate(fd_rgb_stage4_active_loss_states)
                            ]
                            fd_rgb_value = torch.stack([result[0] for result in per_view_results]).mean()
                            fd_rgb_valid = all(result[1] for result in per_view_results)
                        else:
                            fd_rgb_value, fd_rgb_valid = fd_rgb_stage4_active_loss_state(fd_rgb_fake_features)
                        if fd_rgb_timing is not None:
                            torch.cuda.synchronize()
                            fd_rgb_timing["fd_loss_forward"] = time.perf_counter() - fd_rgb_timing.pop(
                                "fd_stats_start"
                            )
                            fd_rgb_timing["feature_backward_start"] = time.perf_counter()
                        fd_rgb_queue_features = fd_rgb_fake_features.detach()
                        fd_rgb_gradient = None
                        if fd_rgb_valid:
                            if bool(cfg.get("fd_rgb_stage4_boundary_backward", False)):
                                # The decoder/I3D graph starts at this detached leaf, so
                                # backward cannot enter the diffusion model. This supports
                                # Turbo's memory-efficient reentrant block checkpoints,
                                # whereas torch.autograd.grad does not.
                                fd_rgb_value.backward()
                                fd_rgb_gradient = rgb_fd_boundary.grad
                                if fd_rgb_gradient is None:
                                    raise RuntimeError("Stage4 RGB FD produced no latent-boundary gradient")
                            else:
                                fd_rgb_gradient = torch.autograd.grad(
                                    fd_rgb_value,
                                    rgb_fd_boundary,
                                    retain_graph=False,
                                    create_graph=False,
                                )[0]
                            fd_rgb_surrogate = prediction_gradient_surrogate(
                                rgb_pred_latent,
                                fd_rgb_gradient,
                            )
                            if bool(cfg.get("fd_rgb_stage4_correct_ddp_gradient_scale", False)):
                                # The differentiable gather returns only this rank's
                                # feature gradient. DDP later averages model gradients,
                                # so compensate to recover the global FD derivative.
                                fd_rgb_surrogate = fd_rgb_surrogate * dist.get_world_size()
                        else:
                            fd_rgb_surrogate = rgb_pred_latent.sum() * 0.0
                        if fd_rgb_timing is not None:
                            torch.cuda.synchronize()
                            fd_rgb_timing["feature_backward"] = time.perf_counter() - fd_rgb_timing.pop(
                                "feature_backward_start"
                            )
                            if dist.get_rank() == 0:
                                logger.info(
                                    "[Stage4 FD-RGB timing] global_step=%s view=%s decoder=%s "
                                    "model_forward_and_diffusion=%.3fs latent_extract=%.3fs "
                                    "pred_decode=%.3fs feature_build=%.3fs fd_loss_forward=%.3fs "
                                    "feature_decoder_backward=%.3fs",
                                    global_step,
                                    fd_rgb_stage4_selected_view,
                                    fd_rgb_stage4_decoder_type,
                                    fd_rgb_timing["model_forward_and_diffusion"],
                                    fd_rgb_timing["latent_extract"],
                                    fd_rgb_timing["pred_decode"],
                                    fd_rgb_timing["feature_build"],
                                    fd_rgb_timing["fd_loss_forward"],
                                    fd_rgb_timing["feature_backward"],
                                )
                        fd_rgb_value = fd_rgb_value.detach()
                        if fd_rgb_raw_per_view is not None:
                            loss_dict["fd_rgb_raw"] = fd_rgb_raw_per_view.detach().mean()
                            loss_dict["fd_rgb_raw_per_view"] = fd_rgb_raw_per_view.detach()
                            loss_dict["fd_rgb_normalized_per_view"] = fd_rgb_normalized_per_view.detach()
                        fd_rgb_fake_features = fd_rgb_queue_features
                        loss_dict["fd_rgb_loss"] = fd_rgb_value
                        loss_dict["fd_rgb_population_ready"] = torch.tensor(
                            float(fd_rgb_valid), device=device, dtype=torch.float32
                        )
                        del rgb_fd_boundary, rgb_pred_decoded, fd_rgb_queue_features
                        if fd_rgb_gradient is not None:
                            del fd_rgb_gradient
                        if verbose_mode:
                            logger.info(
                                "Stage4 RGB FD step=%s normalized=%s raw=%s population_ready=%s view=%s decoder=%s",
                                global_step,
                                fd_rgb_value.detach().float().item(),
                                (
                                    loss_dict["fd_rgb_raw"].detach().float().item()
                                    if "fd_rgb_raw" in loss_dict
                                    else fd_rgb_value.detach().float().item()
                                ),
                                fd_rgb_valid,
                                fd_rgb_stage4_selected_view,
                                fd_rgb_stage4_decoder_type,
                            )
                        if fd_rgb_profile_only:
                            coordinator.block_all()
                            if dist.get_rank() == 0:
                                logger.info(
                                    "[Stage4 FD-RGB profile-only] completed decoder/I3D/FD boundary backward "
                                    "at global_step=%s; exiting before diffusion backward",
                                    global_step,
                                )
                            raise SystemExit(0)
                    if use_static_geo_loss:
                        common_static_geo_args = (
                            loss_dict["pred_x0"],
                            static_geo_flow_loss_mask,
                            static_geo_camera_param,
                            static_geo_frame_transform,
                            static_geo_turbo_decoder,
                            latent_modalities,
                            vae_out_channels,
                            NC,
                            T,
                            float(cfg.get("static_geo_loss_depth_max", 100.0)),
                            float(
                                cfg.get(
                                    "static_geo_loss_turbo_latent_scale",
                                    1.0 / COGVIDEOX_SCALING_FACTOR,
                                )
                            ),
                            float(cfg.get("static_geo_loss_min_valid_depth", 0.1)),
                            float(cfg.get("static_geo_loss_sky_depth_threshold", 99.5)),
                            float(cfg.get("static_geo_loss_static_mask_threshold", 0.5)),
                            static_geo_loss_require_flow_mask,
                        )
                        if static_geo_loss_mode == "geovideo_single_view":
                            static_geo_loss, static_geo_mask_ratio = (
                                geovideo_single_view_static_pointcloud_loss(
                                    *common_static_geo_args,
                                    int(cfg.get("static_geo_loss_sample_frames", 6)),
                                    float(cfg.get("static_geo_loss_reprojection_tolerance", 0.05)),
                                    float(cfg.get("static_geo_loss_voxel_nn_quantile", 0.05)),
                                    int(cfg.get("static_geo_loss_outlier_neighbor_count", 20)),
                                    float(cfg.get("static_geo_loss_outlier_std_ratio", 1.0)),
                                    int(cfg.get("static_geo_loss_knn_workers", 1)),
                                    bool(cfg.get("static_geo_loss_checkpoint_decode", True)),
                                    cfg.get("static_geo_loss_fixed_voxel_size", None),
                                    cfg.get("static_geo_loss_outlier_backend", "scipy"),
                                    int(cfg.get("static_geo_loss_gpu_knn_query_chunk_size", 262144)),
                                    float(cfg.get("static_geo_loss_gpu_sor_search_radius", 2.0)),
                                    bool(cfg.get("static_geo_loss_record_time", False)),
                                    cfg.get("static_geo_loss_debug_export_path", None),
                                    cfg.get("static_geo_loss_raw_debug_export_path", None),
                                )
                            )
                        elif static_geo_loss_mode == "early_anchor":
                            static_geo_loss, static_geo_mask_ratio = static_geo_warp_consistency_loss(
                                *common_static_geo_args,
                                int(cfg.get("static_geo_loss_grid_stride", 16)),
                                float(cfg.get("static_geo_loss_reprojection_tolerance", 0.05)),
                                int(cfg.get("static_geo_loss_sample_views", 3)),
                                bool(cfg.get("static_geo_loss_contiguous_sample_views", True)),
                                float(cfg.get("static_geo_loss_middle_frame_weight", 1.0)),
                                float(cfg.get("static_geo_loss_late_frame_weight", 0.5)),
                                bool(cfg.get("static_geo_loss_checkpoint_decode", True)),
                            )
                        elif static_geo_loss_mode == "rdepth_lidar_stage1":
                            (
                                static_geo_loss,
                                static_geo_stage1_rdepth_loss,
                                static_geo_stage1_lidar_loss,
                                static_geo_stage1_rdepth_ratio,
                                static_geo_stage1_lidar_ratio,
                            ) = stage1_rdepth_lidar_geo_loss(
                                loss_dict["pred_x0"],
                                static_geo_camera_param,
                                batch.get("meta_data", None),
                                static_geo_turbo_decoder,
                                latent_modalities,
                                vae_out_channels,
                                NC,
                                T,
                                float(cfg.get("static_geo_loss_depth_max", 100.0)),
                                float(
                                    cfg.get(
                                        "static_geo_loss_turbo_latent_scale",
                                        1.0 / COGVIDEOX_SCALING_FACTOR,
                                    )
                                ),
                                float(cfg.get("static_geo_loss_min_valid_depth", 0.1)),
                                float(cfg.get("static_geo_loss_sky_depth_threshold", 99.5)),
                                bool(cfg.get("static_geo_loss_checkpoint_decode", True)),
                                int(cfg.get("static_geo_loss_sample_frames", 6)),
                                int(cfg.get("static_geo_loss_sample_views", 3)),
                                bool(cfg.get("static_geo_loss_contiguous_sample_views", True)),
                                float(cfg.get("static_geo_loss_rdepth_weight", 1.0)),
                                float(cfg.get("static_geo_loss_lidar_weight", 0.0)),
                                float(cfg.get("static_geo_loss_log_huber_beta", 0.05)),
                                static_geo_sample_token_to_depth,
                                cfg.get(
                                    "static_geo_loss_rdepth_root",
                                    os.environ.get("RDEPTH_ROOT", "data/nus_Rdepth"),
                                ),
                                int(cfg.get("static_geo_loss_sky_class_id", 27)),
                                static_geo_lidar_infos,
                                float(cfg.get("static_geo_loss_lidar_min_radius", 1.0)),
                                int(cfg.get("static_geo_loss_lidar_max_points", 0)),
                                bool(cfg.get("static_geo_loss_record_time", False)),
                                bool(cfg.get("static_geo_loss_debug_ratio", False)),
                            )
                            static_geo_mask_ratio = static_geo_stage1_rdepth_ratio
                            loss_dict["static_geo_stage1_rdepth_loss"] = static_geo_stage1_rdepth_loss
                            loss_dict["static_geo_stage1_lidar_loss"] = static_geo_stage1_lidar_loss
                            loss_dict["static_geo_stage1_rdepth_ratio"] = static_geo_stage1_rdepth_ratio
                            loss_dict["static_geo_stage1_lidar_ratio"] = static_geo_stage1_lidar_ratio
                        else:
                            raise ValueError(
                                f"Unsupported static_geo_loss_mode={static_geo_loss_mode!r}."
                            )
                        loss_dict["static_geo_loss"] = static_geo_loss
                        loss_dict["static_geo_mask_ratio"] = static_geo_mask_ratio
                if record_time:
                    timer_list.append(loss_t)
                # NOTE: backward needs all_reduce, we sychronize here!
                coordinator.block_all()

                if verbose_mode:
                    logger.info(f"Start model backward step! step={step}, loss={loss_dict['loss']}")
                # == backward & update ==
                with timers["backward"] as backward_t:
                    loss = loss_dict["loss"].mean()
                    if use_static_geo_loss:
                        loss = loss + static_geo_loss_weight * loss_dict["static_geo_loss"]
                    if fd_rgb_due and use_fd_rgb:
                        loss = loss + fd_rgb_weight * loss_dict["fd_rgb_loss"]
                    if fd_rgb_due and use_fd_rgb_stage4:
                        loss = loss + fd_rgb_weight * fd_rgb_surrogate
                    local_loss_finite = torch.isfinite(loss.detach()).all()
                    finite_flag = local_loss_finite.to(device=device, dtype=torch.int32)
                    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                    if finite_flag.item() == 0:
                        bad_rank = -1 if local_loss_finite.item() else dist.get_rank()
                        bad_rank_tensor = torch.tensor(bad_rank, device=device, dtype=torch.int32)
                        dist.all_reduce(bad_rank_tensor, op=dist.ReduceOp.MAX)
                        logger.error(
                            "Non-finite loss detected; aborting before backward: "
                            "epoch=%s step=%s global_step=%s rank=%s frames=%s cached_latent_path=%s loss=%s",
                            epoch,
                            step,
                            global_step,
                            bad_rank_tensor.item(),
                            T,
                            cached_latent_path,
                            loss.detach().float().item(),
                        )
                        raise FloatingPointError(
                            f"Non-finite loss at global_step={global_step} rank={bad_rank_tensor.item()}"
                        )
                    booster.backward(loss=loss, optimizer=optimizer)
                    if verbose_mode:
                        logger.info(f"Start model update step! step={step}")
                    optimizer.step()
                    if fd_rgb_due and use_fd_rgb:
                        fd_rgb_loss_state.update(fd_rgb_real_features, fd_rgb_fake_features)
                    if fd_rgb_due and use_fd_rgb_stage4:
                        if fd_rgb_stage4_active_loss_states is not None:
                            for local_index, state in enumerate(fd_rgb_stage4_active_loss_states):
                                state.update(fd_rgb_fake_features[:, local_index : local_index + 1])
                        else:
                            fd_rgb_stage4_active_loss_state.update(fd_rgb_fake_features)
                    if enable_debug:
                        for n, p in model.named_parameters():
                            if not (p == p).all():
                                logger.info(f"Got nan on {n}")
                    optimizer.zero_grad()

                    # update learning rate
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                if record_time:
                    timer_list.append(backward_t)

                if verbose_mode:
                    logger.info(f"Start after step ops! step={step}")
                # == update EMA ==
                with timers["update_ema"] as ema_t:
                    update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))
                if record_time:
                    timer_list.append(ema_t)

                # == update log info ==
                with timers["reduce_loss"] as reduce_loss_t:
                    all_reduce_mean(loss)
                    running_loss += loss.item()
                    global_step = epoch * num_steps_per_epoch + step + global_step_offset
                    log_step += 1
                    acc_step += 1
                if record_time:
                    timer_list.append(reduce_loss_t)

                if record_time:
                    misc_t = timers['misc'].__enter__()
                    timer_list.append(misc_t)
                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    lr = optimizer.param_groups[0]["lr"]
                    # progress bar, use str to avoid conversion
                    pbar.set_postfix({"loss": avg_loss, "step": str(step), "global_step": str(global_step), "lr": lr})
                    # tensorboard
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    tb_writer.add_scalar("avg_loss", avg_loss, global_step)
                    tb_writer.add_scalar("lr", lr, global_step)
                    if use_static_geo_loss:
                        tb_writer.add_scalar(
                            "loss/diffusion",
                            loss_dict["loss"].mean().detach().float().item(),
                            global_step,
                        )
                        tb_writer.add_scalar(
                            "loss/static_geo",
                            loss_dict["static_geo_loss"].detach().float().item(),
                            global_step,
                        )
                        tb_writer.add_scalar(
                            "static_geo/mask_ratio",
                            loss_dict["static_geo_mask_ratio"].detach().float().item(),
                            global_step,
                        )
                        if static_geo_loss_mode == "rdepth_lidar_stage1":
                            tb_writer.add_scalar(
                                "static_geo_stage1/rdepth_loss",
                                loss_dict["static_geo_stage1_rdepth_loss"].detach().float().item(),
                                global_step,
                            )
                            tb_writer.add_scalar(
                                "static_geo_stage1/lidar_loss",
                                loss_dict["static_geo_stage1_lidar_loss"].detach().float().item(),
                                global_step,
                            )
                            tb_writer.add_scalar(
                                "static_geo_stage1/rdepth_ratio",
                                loss_dict["static_geo_stage1_rdepth_ratio"].detach().float().item(),
                                global_step,
                            )
                            tb_writer.add_scalar(
                                "static_geo_stage1/lidar_ratio",
                                loss_dict["static_geo_stage1_lidar_ratio"].detach().float().item(),
                                global_step,
                            )
                    if fd_rgb_due:
                        tb_writer.add_scalar(
                            "loss/fd_rgb",
                            loss_dict["fd_rgb_loss"].detach().float().item(),
                            global_step,
                        )
                        tb_writer.add_scalar(
                            "fd_rgb/population_ready",
                            loss_dict["fd_rgb_population_ready"].detach().float().item(),
                            global_step,
                        )
                        if "fd_rgb_raw" in loss_dict:
                            tb_writer.add_scalar(
                                "fd_rgb/raw_mean",
                                loss_dict["fd_rgb_raw"].detach().float().item(),
                                global_step,
                            )
                            for view_index, (raw_value, normalized_value) in enumerate(zip(
                                loss_dict["fd_rgb_raw_per_view"],
                                loss_dict["fd_rgb_normalized_per_view"],
                            )):
                                tb_writer.add_scalar(
                                    f"fd_rgb/raw_view_{view_index}",
                                    raw_value.detach().float().item(),
                                    global_step,
                                )
                                tb_writer.add_scalar(
                                    f"fd_rgb/normalized_view_{view_index}",
                                    normalized_value.detach().float().item(),
                                    global_step,
                                )
                    append_loss_history(
                        exp_dir,
                        global_step,
                        loss.item(),
                        avg_loss,
                        lr,
                        diffusion_loss=(
                            loss_dict["loss"].mean().detach().float().item()
                            if use_static_geo_loss
                            else None
                        ),
                        fd_rgb_loss=(
                            loss_dict["fd_rgb_loss"].detach().float().item()
                            if fd_rgb_due
                            else None
                        ),
                        fd_rgb_raw=(
                            loss_dict["fd_rgb_raw"].detach().float().item()
                            if fd_rgb_due and "fd_rgb_raw" in loss_dict
                            else None
                        ),
                        fd_rgb_raw_per_view=(
                            loss_dict["fd_rgb_raw_per_view"].detach().float().cpu().tolist()
                            if fd_rgb_due and "fd_rgb_raw_per_view" in loss_dict
                            else None
                        ),
                        fd_rgb_normalized_per_view=(
                            loss_dict["fd_rgb_normalized_per_view"].detach().float().cpu().tolist()
                            if fd_rgb_due and "fd_rgb_normalized_per_view" in loss_dict
                            else None
                        ),
                        static_geo_loss=(
                            loss_dict["static_geo_loss"].detach().float().item()
                            if use_static_geo_loss
                            else None
                        ),
                        static_geo_mask_ratio=(
                            loss_dict["static_geo_mask_ratio"].detach().float().item()
                            if use_static_geo_loss
                            else None
                        ),
                        static_geo_stage1_rdepth_loss=(
                            loss_dict["static_geo_stage1_rdepth_loss"].detach().float().item()
                            if use_static_geo_loss and static_geo_loss_mode == "rdepth_lidar_stage1"
                            else None
                        ),
                        static_geo_stage1_lidar_loss=(
                            loss_dict["static_geo_stage1_lidar_loss"].detach().float().item()
                            if use_static_geo_loss and static_geo_loss_mode == "rdepth_lidar_stage1"
                            else None
                        ),
                        static_geo_stage1_rdepth_ratio=(
                            loss_dict["static_geo_stage1_rdepth_ratio"].detach().float().item()
                            if use_static_geo_loss and static_geo_loss_mode == "rdepth_lidar_stage1"
                            else None
                        ),
                        static_geo_stage1_lidar_ratio=(
                            loss_dict["static_geo_stage1_lidar_ratio"].detach().float().item()
                            if use_static_geo_loss and static_geo_loss_mode == "rdepth_lidar_stage1"
                            else None
                        ),
                    )
                    loss_curve_every = cfg.get("loss_curve_every", cfg.get("report_every", 0))
                    if loss_curve_every > 0 and (global_step + 1) % loss_curve_every == 0:
                        save_loss_curve(exp_dir)

                    running_loss = 0.0
                    log_step = 0

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                max_train_steps = cfg.get("max_train_steps", None)
                reached_max_train_steps = (
                    max_train_steps is not None
                    and global_step + 1 >= max_train_steps
                )
                periodic_checkpoint = (
                    ckpt_every > 0 and (global_step + 1) % ckpt_every == 0
                )
                final_checkpoint = ckpt_every > 0 and reached_max_train_steps
                if periodic_checkpoint or final_checkpoint:
                    if verbose_mode:
                        logger.info(f"Start to save ckpt! step={step}")
                    model_gathering(ema, ema_shape_dict)
                    save_dir = save(
                        booster,
                        exp_dir,
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        sampler=sampler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    if coordinator.is_master() and use_fd_rgb_stage4 and fd_rgb_stage4_statistics == "ema":
                        torch.save(
                            fd_rgb_stage4_loss_state.state_dict(),
                            os.path.join(save_dir, "fd_rgb_stage4_ema.pt"),
                        )
                    if dist.get_rank() == 0:
                        model_sharding(ema)
                    logger.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        save_dir,
                    )
                    if coordinator.is_master():
                        save_loss_curve(exp_dir)
                    sub_dir_name = os.path.basename(save_dir)

                if reached_max_train_steps:
                    logger.info("Reached max_train_steps=%s, stop training.", max_train_steps)
                    coordinator.block_all()
                    return

                report_every = cfg.get("report_every", 0)
                if report_every > 0 and (global_step + 1) % report_every == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    val_dir = run_validation(
                        cfg.val,
                        text_encoder,
                        vae,
                        model,
                        device,
                        dtype,
                        val_dataloader,
                        coordinator,
                        global_step + 1,
                        exp_dir,
                        cfg.mv_order_map,
                        cfg.t_order_map,
                    )
                    val_sampler.reset()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    if coordinator.is_master():
                        save_loss_curve(exp_dir)
                    sub_dir_name = os.path.basename(val_dir)

                if record_time:
                    misc_t.__exit__(*sys.exc_info())
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for timer in timer_list:
                        log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                    log_str += f"Total: {sum([t.elapsed_time for t in timer_list]):.3f}s"
                    logger.info(log_str)

                if enable_debug and step > 50:
                    break
        if enable_debug:
            break
        sampler.reset()
        start_step = 0


if __name__ == "__main__":
    main()
