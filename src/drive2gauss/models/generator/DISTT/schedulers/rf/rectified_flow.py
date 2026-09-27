from typing import List
import logging

import torch
from torch.distributions import LogisticNormal
from einops import rearrange

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def mean_flat(tensor: torch.Tensor, mask=None, channel_weights=None, element_mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    weight = None
    if channel_weights is not None:
        assert tensor.dim() == 5
        channel_weights = channel_weights.to(device=tensor.device, dtype=tensor.dtype)
        if channel_weights.dim() == 1:
            channel_weights = channel_weights.view(1, -1, 1, 1, 1)
        assert channel_weights.shape[1] == tensor.shape[1]
        weight = channel_weights

    if element_mask is not None:
        assert tensor.dim() == 5
        element_mask = element_mask.to(device=tensor.device, dtype=tensor.dtype)
        assert element_mask.shape == tensor.shape or element_mask.shape[0] == tensor.shape[0]
        weight = element_mask if weight is None else weight * element_mask

    if mask is not None:
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        temporal_mask = mask[:, None, :, None, None].to(
            device=tensor.device, dtype=tensor.dtype
        )
        weight = temporal_mask if weight is None else weight * temporal_mask

    if weight is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    weighted_tensor = tensor * weight
    denom = weight.expand_as(tensor).sum(dim=list(range(1, len(tensor.shape))))
    denom = denom.clamp_min(torch.finfo(tensor.dtype).eps)
    return weighted_tensor.sum(dim=list(range(1, len(tensor.shape)))) / denom


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: List[int]):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
    cog_style=False,
):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()

    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    # TODO: hard-coded, may change later!
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        if cog_style:
            num_frames = model_kwargs["num_frames"] // 4 + model_kwargs["num_frames"] % 2
        else:
            num_frames = model_kwargs["num_frames"] // 17 * 5
    assert (num_frames >= 1).all(), "num_frames cannot be less than 1"
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    assert (ratio > 0).all(), "ratio cannot be 0"
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
        cog_style_trans=False,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        if cog_style_trans:
            logging.warning("Use `cog_style_trans`. Please make sure train&inference is consistent!")
        self.cog_style_trans = cog_style_trans

    def training_losses(
        self,
        model,
        x_start,
        model_kwargs=None,
        noise=None,
        mask=None,
        weights=None,
        t=None,
        loss_channel_weights=None,
        loss_element_mask=None,
        return_pred_x0=False,
    ):
        """
        Compute training losses for a single timestep.
        Arguments format copied from DISTT/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps, cog_style=self.cog_style_trans)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        model_output = model(x_t, t, **model_kwargs)
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]
        velocity_pred = model_output
        if weights is None:
            loss = mean_flat(
                (velocity_pred - (x_start - noise)).pow(2),
                mask=mask,
                channel_weights=loss_channel_weights,
                element_mask=loss_element_mask,
            )
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(
                weight * (velocity_pred - (x_start - noise)).pow(2),
                mask=mask,
                channel_weights=loss_channel_weights,
                element_mask=loss_element_mask,
            )
        terms["loss"] = loss
        if return_pred_x0:
            noise_ratio = (t.float() / self.num_timesteps).view(-1, 1, 1, 1, 1)
            terms["pred_x0"] = x_t + noise_ratio.to(dtype=velocity_pred.dtype) * velocity_pred
            terms["t"] = t

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise
