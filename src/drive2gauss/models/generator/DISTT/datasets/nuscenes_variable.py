from collections import OrderedDict, defaultdict
from pprint import pformat
from pathlib import Path
from typing import Iterator, List, Optional
import logging

import numpy as np
import torch
import torch.distributed as dist
import mmcv
from mmengine.config import ConfigDict

from DISTT.utils.misc import format_numel_str
from DISTT.registry import DATASETS, build_module
from ..mmdet_plugin.datasets import NuScenesDataset
from .nuscenes_t_dataset import NuScenesTDataset, collate_fn_single_clip, transform_bbox
from .utils import IMG_FPS

import json
import os
from drive2gauss.data.motion_release import MotionRelease
@DATASETS.register_module()
class NuScenesVariableDataset(NuScenesTDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length: list[int] = None,
        start_on_keyframe=True,
        next2topv2=True,
        trans_box2top=False,
        base_fps=12,
        fps: list[list[int]] = None,
        repeat_times: list[int] = None,
        img_collate_param={},
        micro_frame_size=None,
        balance_keywords=None,
        drop_ori_imgs=False,
        latent_manifest_path=None,
        depth_root_json=None,
        skip_refine_depth=False,
        depth_release_root=None,
        motion_release_manifest=None,
    ) -> None:
        self.video_lengths = video_length
        self.start_on_keyframe = start_on_keyframe
        self.fps = fps
        self.micro_frame_size = micro_frame_size
        self.repeat_times = repeat_times
        self.balance_keywords = balance_keywords
        self.latent_manifest_path = latent_manifest_path
        self.latent_cache_by_source_key = self.load_latent_manifest(latent_manifest_path)
        self.latent_cache_by_key = {}
        NuScenesDataset.__init__(
            self, ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag,
            force_all_boxes)
        if "12Hz" in ann_file and start_on_keyframe:
            logging.warning("12Hz should use all starting frame to train, please "
                         "double-check!")
        self.next2topv2 = next2topv2
        self.trans_box2top = trans_box2top
        self.allow_class = None
        self.del_box_ratio = 0.0
        self.drop_nearest_car = 0
        self.img_collate_param = img_collate_param
        if isinstance(self.img_collate_param, ConfigDict):
            self.img_collate_param = img_collate_param.to_dict()
        self.base_fps = base_fps
        self.drop_ori_imgs = drop_ori_imgs
        self.skip_refine_depth = skip_refine_depth
        self.depth_release_root = depth_release_root or os.environ.get("DRIVE2GAUSS_DATASET_ROOT")
        if self.depth_release_root:
            self.depth_release_root = os.path.abspath(self.depth_release_root)
            manifest_path = motion_release_manifest or os.path.join(
                self.depth_release_root, "manifest.jsonl")
            self.depth_release_token_index = self._load_depth_release_index(manifest_path)
            self.motion_release = MotionRelease(
                self.depth_release_root, Path(manifest_path))
            self.sample_token2depth = {}
            return
        self.motion_release = None
        depth_root_json = depth_root_json or self.find_depth_root_json()
        with open(depth_root_json, 'r') as file:
            self.sample_token2depth = json.load(file) 

    @staticmethod
    def find_depth_root_json():
        candidates = [
            os.environ.get("DEPTH_ROOT_JSON"),
            "data/nus_sampletoken2depthroot.json",
            "misc/nus_sampletoken2depthroot.json",
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        raise FileNotFoundError(
            "Cannot find nus_sampletoken2depthroot.json. Tried: "
            + ", ".join(candidates)
        )

    @staticmethod
    def iter_manifest_paths(latent_manifest_path):
        if latent_manifest_path is None:
            return []
        if isinstance(latent_manifest_path, (str, os.PathLike)):
            return [os.fspath(latent_manifest_path)]
        if isinstance(latent_manifest_path, dict):
            return [os.fspath(path) for path in latent_manifest_path.values()]
        return [os.fspath(path) for path in latent_manifest_path]

    def load_latent_manifest(self, latent_manifest_path):
        cache_by_key = {}
        manifest_paths = self.iter_manifest_paths(latent_manifest_path)
        for manifest_path in manifest_paths:
            manifest_root = os.path.dirname(os.path.abspath(manifest_path))
            with open(manifest_path, "r") as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    video_length = int(row["video_length"])
                    dataset_index = int(row["dataset_index"])
                    latent_path = os.path.expanduser(row["path"])
                    if not os.path.isabs(latent_path):
                        path_base = os.path.expanduser(row.get("path_base", "."))
                        if not os.path.isabs(path_base):
                            path_base = os.path.join(manifest_root, path_base)
                        latent_path = os.path.abspath(os.path.join(path_base, latent_path))
                    cache_by_key[(video_length, dataset_index)] = latent_path
        if manifest_paths:
            logging.info(
                "[%s] Loaded %s cached latents from %s",
                self.__class__.__name__,
                len(cache_by_key),
                manifest_paths,
            )
        return cache_by_key

    @property
    def num_frames(self):
        raise NotImplementedError()

    def build_clips(self, data_infos, scene_tokens, video_length, repeat_times=1):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        if self.balance_keywords is not None:
            data_infos, scene_tokens = self.balance_annotations(
                data_infos, scene_tokens)
        all_clips = []
        skip1, skip2 = 0, 0
        for scene in scene_tokens:
            if video_length == "full":
                clip = [self.token_data_dict[token] for token in scene]
                if self.micro_frame_size is not None:
                    # trim to micro_frame_size
                    res = len(clip) % self.micro_frame_size - 1
                    if res > 0:
                        clip = clip[:-res]
                all_clips.append(clip)
            else:
                for start in range(len(scene) - video_length + 1):
                    if self.start_on_keyframe and ";" in scene[start]:
                        skip1 += 1
                        continue  # this is not a keyframe
                    if self.start_on_keyframe and len(scene[start]) >= 33:
                        skip2 += 1
                        continue  # this is not a keyframe
                    clip = [self.token_data_dict[token]
                            for token in scene[start: start + video_length]]
                    if self.micro_frame_size is not None:
                        assert len(clip) % self.micro_frame_size <= 1
                    all_clips.append(clip)
        if repeat_times > 1:
            assert isinstance(repeat_times, int)
            all_clips = all_clips * repeat_times
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {video_length}-clip, "
                     f"which has {len(all_clips)} in total. We skip {skip1} + "
                     f"{skip2} = {skip1 + skip2} possible starting frames.")
        return all_clips

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = OrderedDict()
        for idx, video_length in enumerate(self.video_lengths):
            if self.repeat_times is not None:
                repeat_times = self.repeat_times[idx]
            else:
                repeat_times = 1
            clips = self.build_clips(
                data_infos, data['scene_tokens'], video_length, repeat_times)
            if self.latent_cache_by_source_key:
                filtered_clips = []
                for source_idx, clip in enumerate(clips):
                    latent_path = self.latent_cache_by_source_key.get(
                        (video_length, source_idx))
                    if latent_path is None:
                        continue
                    self.latent_cache_by_key[(video_length, len(filtered_clips))] = latent_path
                    filtered_clips.append(clip)
                logging.info(
                    "[%s] Keep %s/%s %s-frame clips with cached latents.",
                    self.__class__.__name__,
                    len(filtered_clips),
                    len(clips),
                    video_length,
                )
                clips = filtered_clips
            self.clip_infos[video_length] = clips
        return data_infos

    def __len__(self):
        return sum(self.key_len(key) for key in self.possible_keys)

    def key_len(self, key):
        if isinstance(key, str):
            fps, t = key.split("-")
            fps = int(fps)
            t = t if t == "full" else int(t)
        elif isinstance(key, tuple):
            fps, t = key
        else:
            raise TypeError(key)
        return len(self.clip_infos[t])

    @property
    def possible_keys(self):
        keys = []
        for f, t in zip(self.fps, self.clip_infos.keys()):
            for fps in f:
                keys.append((fps, t))
        return keys

    def parse_index(self, index):
        idx, real_t, fps = index.split("-")
        idx, fps = map(int, [idx, fps])
        real_t = real_t if real_t == "full" else int(real_t)
        return idx, real_t, fps

    def _rand_another(self, index):
        idx, real_t, fps = self.parse_index(index)
        pool = list(range(len(self.clip_infos[real_t])))
        idx = np.random.choice(pool)
        return f"{idx}-{real_t}-{fps}"

    def get_data_info(self, idx, num_frames, interval):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[num_frames][idx][0::interval]
        frames = self.load_clip(clip)
        return frames

    def load_frames(self, frames):
        if not self.latent_cache_by_source_key and not self.skip_refine_depth:
            ret_dicts = super().load_frames(frames)
            if ret_dicts is not None and self.motion_release is not None:
                ret_dicts["flow_rgb_values"] = self.load_release_flow_rgb(frames)
            return ret_dicts
        if None in frames:
            return None
        examples = []
        first_frame_boxes = None
        for frame in frames:
            self.pre_pipeline(frame)
            example = self.pipeline(frame)
            image_shape = example["img"].data.shape
            example["Refine_depth"] = torch.zeros(
                image_shape[0],
                image_shape[-2],
                image_shape[-1],
                dtype=torch.float32,
            )
            if self.filter_empty_gt and frame['is_key_frame'] and (
                example is None or ~(example["gt_labels_3d"]._data != -1).any()
            ):
                return None
            if self.trans_box2top:
                if first_frame_boxes is None:
                    first_frame_boxes = {
                        'gt_bboxes_3d': example['gt_bboxes_3d'],
                        'gt_labels_3d': example['gt_labels_3d'],
                    }
                else:
                    this_frame_boxes = transform_bbox(
                        first_frame_boxes, frame['next2top'])
                    example['gt_bboxes_3d'] = this_frame_boxes['gt_bboxes_3d']
                    example['gt_labels_3d'] = this_frame_boxes['gt_labels_3d']
            examples.append(example)
        if self.del_box_ratio > 0 or self.allow_class is not None or self.drop_nearest_car > 0:
            self.rand_del_box(
                examples, self.del_box_ratio, self.allow_class, self.drop_nearest_car)
        ret_dicts = collate_fn_single_clip(examples, **self.img_collate_param)
        if self.img_collate_param.get("return_raw_data", False):
            return ret_dicts
        ret_dicts['height'] = ret_dicts['pixel_values'].shape[-2]
        ret_dicts['width'] = ret_dicts['pixel_values'].shape[-1]
        if self.drop_ori_imgs:
            ret_dicts["pixel_values_shape"] = torch.IntTensor(
                list(ret_dicts['pixel_values'].shape))
            ret_dicts.pop("pixel_values")
        return ret_dicts

    def load_release_flow_rgb(self, frames):
        view_order = (
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
        )
        height, width = 424, 800
        values = np.full(
            (len(frames), len(view_order), height, width, 3), 255, dtype=np.uint8)
        for frame_index, (source, target) in enumerate(zip(frames[:-1], frames[1:])):
            for view_index, camera in enumerate(view_order):
                rgb, _ = self.motion_release.load_masked_flow_rgb(
                    str(source["token"]), str(target["token"]), camera)
                values[frame_index, view_index] = rgb
        tensor = torch.from_numpy(values).permute(0, 1, 4, 2, 3).float()
        return tensor.div_(127.5).sub_(1.0)

    def resolve_data_path(self, path):
        if os.path.isabs(path):
            return path
        path = path.replace("../", "").replace("./", "")
        dataset_root = getattr(self, "data_root", None) or self.dataset_root
        dataset_root = dataset_root.rstrip("/")
        if path.startswith("data/nuscenes/"):
            return os.path.join(dataset_root, path[len("data/nuscenes/"):])
        if path.startswith("nuscenes/"):
            return os.path.join(dataset_root, path[len("nuscenes/"):])
        return os.path.join(dataset_root, path)

    def prepare_train_data(self, index):
        idx, requested_t, fps = self.parse_index(index)
        if isinstance(requested_t, str) or requested_t > 1:
            assert fps <= self.base_fps
            interval = self.base_fps // fps
        else:
            interval = 1
        frames = self.get_data_info(idx, requested_t, interval=interval)

        # My modi
        for frame in frames:
            frame['lidar_path'] = self.resolve_data_path(frame['lidar_path'])
            frame['image_paths'] = [
                self.resolve_data_path(s) for s in frame['image_paths']]

        real_t = len(frames)  # NOTE: we have load interval, real_t may change
        ret_dicts = self.load_frames(frames)
        if ret_dicts is None:
            return None
        ret_dicts['fps'] = IMG_FPS if real_t == 1 else fps
        ret_dicts['num_frames'] = real_t
        latent_path = self.latent_cache_by_key.get((requested_t, idx))
        if latent_path is not None:
            ret_dicts["cached_latent_path"] = latent_path
        return ret_dicts


@DATASETS.register_module()
class NuScenesMultiResDataset(torch.utils.data.Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.datasets = OrderedDict()
        for (res, d_cfg) in cfg:
            dataset: NuScenesVariableDataset = build_module(d_cfg, DATASETS)
            self.datasets[res] = dataset

    def as_buckets(self):
        buckets = OrderedDict()  # str: list of indexes
        for res, v in self.datasets.items():
            for key in v.possible_keys:
                buckets["-".join(map(str, [*res, *key]))] = list(
                    range(v.key_len("-".join(map(str, key)))))
        return buckets

    def rand_another_key(self):
        buckets = self.as_buckets()
        key = np.random.choice(list(buckets.keys()))
        idx = np.random.choice(buckets[key])
        return f"{idx}-{key}"

    def parse_index(self, index: str):
        idx, real_h, real_w, fps = map(int, index.split("-")[:-1])
        real_t = index.split("-")[-1]
        real_t = real_t if real_t == "full" else int(real_t)
        return idx, real_h, real_w, fps, real_t

    def __len__(self):
        return sum(len(v) for v in self.datasets.values())

    def __getitem__(self, index):
        idx, real_h, real_w, fps, real_t = self.parse_index(index)
        sub_index = f"{idx}-{real_t}-{fps}"
        return self.datasets[(real_h, real_w)][sub_index]


class NuScenesVariableBatchSampler(torch.utils.data.DistributedSampler):
    def __init__(
        self,
        dataset: NuScenesMultiResDataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        bucket_sample_ratios: Optional[dict] = None,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.bs_config = bucket_config
        self.verbose = verbose
        self.bucket_sample_ratios = bucket_sample_ratios
        self.last_micro_batch_access_index = 0

        self._source_bucket_sample_dict = self.dataset.as_buckets()
        self._default_bucket_sample_dict = OrderedDict(
            {k: list(v) for k, v in self._source_bucket_sample_dict.items()}
        )
        self._bucket_sample_target_counts = self._compute_bucket_sample_target_counts()
        self._default_bucket_micro_batch_count = OrderedDict()
        self.approximate_num_batch = 0
        # process the samples
        for bucket_id, data_list in self._default_bucket_sample_dict.items():
            target_count = self._bucket_sample_target_counts.get(bucket_id)
            if target_count is not None:
                data_list = self._resize_bucket_samples(data_list, target_count)
            # handle droplast
            bs_per_gpu = self.bs_config[bucket_id]
            if bs_per_gpu == -1:
                logging.warning(f"Got bs=-1, we drop {bucket_id}.")
                continue
            data_list = self._adjust_bucket_to_batch_size(data_list, bs_per_gpu)
            self._default_bucket_sample_dict[bucket_id] = data_list
            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bs_per_gpu
            self._default_bucket_micro_batch_count[bucket_id] = num_micro_batches
            self.approximate_num_batch += num_micro_batches
        self._print_bucket_info(self._default_bucket_sample_dict)

    @staticmethod
    def _bucket_length_key(bucket_id):
        return bucket_id.split("-")[-1]

    def _ratio_for_bucket(self, bucket_id):
        if self.bucket_sample_ratios is None:
            return None
        length_key = self._bucket_length_key(bucket_id)
        return self.bucket_sample_ratios.get(length_key, self.bucket_sample_ratios.get(bucket_id))

    @staticmethod
    def _resize_bucket_samples(data_list, target_count):
        data_list = list(data_list)
        if target_count <= len(data_list):
            return data_list[:target_count]
        repeats, remainder = divmod(target_count, len(data_list))
        return data_list * repeats + data_list[:remainder]

    def _adjust_bucket_to_batch_size(self, data_list, bs_per_gpu):
        data_list = list(data_list)
        remainder = len(data_list) % bs_per_gpu
        if remainder <= 0:
            return data_list
        if not self.drop_last:
            return data_list + data_list[: bs_per_gpu - remainder]
        return data_list[:-remainder]

    def _compute_bucket_sample_target_counts(self):
        if not self.bucket_sample_ratios:
            return {}

        ratio_by_bucket = OrderedDict()
        for bucket_id in self._default_bucket_sample_dict:
            if self.bs_config.get(bucket_id, -1) == -1:
                continue
            ratio = self._ratio_for_bucket(bucket_id)
            if ratio is None:
                continue
            ratio = float(ratio)
            if ratio <= 0:
                raise ValueError(f"bucket_sample_ratios for {bucket_id} must be positive, got {ratio}")
            ratio_by_bucket[bucket_id] = ratio

        if not ratio_by_bucket:
            return {}

        total_available = sum(len(self._default_bucket_sample_dict[bucket_id]) for bucket_id in ratio_by_bucket)
        total_ratio = sum(ratio_by_bucket.values())
        target_counts = {}
        for bucket_id, ratio in ratio_by_bucket.items():
            target_counts[bucket_id] = max(1, round(total_available * ratio / total_ratio))
        return target_counts

    def __iter__(self) -> Iterator[List[str]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_last_consumed = OrderedDict()

        if self._bucket_sample_target_counts:
            bucket_sample_dict = OrderedDict(
                {k: list(v) for k, v in self._source_bucket_sample_dict.items()}
            )
        else:
            bucket_sample_dict = OrderedDict(
                {k: list(v) for k, v in self._default_bucket_sample_dict.items()}
            )
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
            target_count = self._bucket_sample_target_counts.get(bucket_id)
            if target_count is not None:
                data_list = self._resize_bucket_samples(data_list, target_count)
                data_list = self._adjust_bucket_to_batch_size(data_list, self.bs_config[bucket_id])
            bucket_sample_dict[bucket_id] = data_list

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in self._default_bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # make the number of bucket accesses divisible by dp size
        remainder = len(bucket_id_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_id_access_order = bucket_id_access_order[: len(bucket_id_access_order) - remainder]
            else:
                bucket_id_access_order += bucket_id_access_order[: self.num_replicas - remainder]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order) // self.num_replicas
        # NOTE: all the dict/indexes should be the same across all devices.
        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bs_config[bucket_id]
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self.num_replicas: (i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.bs_config[bucket_id]
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each GPU
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0]: boundary[1]]

            # encode t, h, w into the sample index
            cur_micro_batch = [f"{idx}-{bucket_id}" for idx in cur_micro_batch]
            yield cur_micro_batch

        self.reset()

    def __len__(self) -> int:
        return self.get_num_batch() // self.num_replicas

    def reset(self):
        self.last_micro_batch_access_index = 0

    def get_num_batch(self) -> int:
        # calculate the number of batches
        if self.verbose:
            self._print_bucket_info(self._default_bucket_sample_dict)
        return self.approximate_num_batch

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        full_dict = defaultdict(lambda: [0, 0])
        num_img_dict = defaultdict(lambda: [0, 0])
        num_vid_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            real_h, real_w, fps = map(int, k.split("-")[:-1])
            real_t = k.split("-")[-1]
            real_t = real_t if real_t == "full" else int(real_t)
            num_batch = size // self.bs_config[k]

            total_samples += size
            total_batch += num_batch

            full_dict[k][0] += size
            full_dict[k][1] += num_batch

            if real_t == 1:
                num_img_dict[k][0] += size
                num_img_dict[k][1] += num_batch
            else:
                num_vid_dict[k][0] += size
                num_vid_dict[k][1] += num_batch

        # log
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            logging.info("Bucket Info:")
            logging.info(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(full_dict, sort_dicts=False)
            )
            logging.info(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_img_dict, sort_dicts=False)
            )
            logging.info(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_vid_dict, sort_dicts=False)
            )
            logging.info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "last_micro_batch_access_index": num_steps * self.num_replicas,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)
