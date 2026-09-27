"""Small-batch, differentiable Frechet loss utilities for RGB training."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist


logger = logging.getLogger("DISTT.frechet_rgb_loss")


class _DifferentiableAllGather(torch.autograd.Function):
    """Gather equal-sized local feature chunks while preserving local grads."""

    @staticmethod
    def forward(ctx, features):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        gathered = [torch.zeros_like(features) for _ in range(world_size)]
        dist.all_gather(gathered, features.contiguous())
        ctx.rank = rank
        ctx.local_size = features.shape[0]
        gathered[rank] = features
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.rank * ctx.local_size
        end = start + ctx.local_size
        return grad_output[start:end].contiguous()


def differentiable_all_gather(features: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return features
    return _DifferentiableAllGather.apply(features)


def _gather_detached(features: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return features.detach()
    gathered = [torch.zeros_like(features) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, features.detach().contiguous())
    return torch.cat(gathered, dim=0)


def _statistics(features: torch.Tensor):
    features = features.to(dtype=torch.float32)
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.transpose(0, 1).matmul(centered) / max(features.shape[0] - 1, 1)
    return mean, 0.5 * (covariance + covariance.transpose(0, 1))


def _sqrt_psd(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(eps).sqrt()
    return (eigenvectors * eigenvalues.unsqueeze(0)).matmul(eigenvectors.transpose(0, 1))


class OnlineFrechetRGBLoss:
    """FD over a bounded online population, with gradients only for current fakes.

    The queue is intentionally process-local and rebuilt after resume. It is a
    warmup population, not optimizer state, so it never changes checkpoints.
    """

    def __init__(self, feature_dim: int, queue_size: int = 128, min_population: int = 8, eps: float = 1e-4):
        self.feature_dim = int(feature_dim)
        self.queue_size = int(queue_size)
        self.min_population = int(min_population)
        self.eps = float(eps)
        self.real_queue = None
        self.fake_queue = None
        self.warned_warmup = False

    def _append(self, queue, features):
        features = features.detach().to(dtype=torch.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected RGB FD features with shape (N, {self.feature_dim}), got {tuple(features.shape)}"
            )
        if queue is None:
            queue = features
        else:
            queue = torch.cat([queue, features], dim=0)
        return queue[-self.queue_size :]

    def update(self, real_features: torch.Tensor, fake_features: torch.Tensor):
        self.real_queue = self._append(self.real_queue, _gather_detached(real_features))
        self.fake_queue = self._append(self.fake_queue, _gather_detached(fake_features))

    def __call__(self, fake_features: torch.Tensor, real_features: torch.Tensor):
        fake_current = differentiable_all_gather(fake_features)
        real_current = _gather_detached(real_features)
        fake_population = fake_current if self.fake_queue is None else torch.cat([self.fake_queue, fake_current], dim=0)
        real_population = real_current if self.real_queue is None else torch.cat([self.real_queue, real_current], dim=0)

        if min(fake_population.shape[0], real_population.shape[0]) < self.min_population:
            if not self.warned_warmup:
                logger.info(
                    "RGB FD warmup: population=%s/%s min_population=%s",
                    fake_population.shape[0],
                    real_population.shape[0],
                    self.min_population,
                )
                self.warned_warmup = True
            return fake_features.sum() * 0.0, False

        fake_mean, fake_cov = _statistics(fake_population)
        real_mean, real_cov = _statistics(real_population)
        eye = torch.eye(self.feature_dim, device=fake_cov.device, dtype=fake_cov.dtype)
        fake_cov = fake_cov + self.eps * eye
        real_cov = real_cov + self.eps * eye
        real_cov_sqrt = _sqrt_psd(real_cov, self.eps)
        middle = real_cov_sqrt.matmul(fake_cov).matmul(real_cov_sqrt)
        covmean_trace = torch.linalg.eigvalsh(0.5 * (middle + middle.transpose(0, 1))).clamp_min(0).sqrt().sum()
        distance = (fake_mean - real_mean).dot(fake_mean - real_mean)
        distance = distance + torch.trace(fake_cov) + torch.trace(real_cov) - 2.0 * covmean_trace
        distance = distance.clamp_min(0.0)
        if not torch.isfinite(distance).all():
            logger.warning("RGB FD became non-finite; skipping this FD event")
            return fake_features.sum() * 0.0, False
        return distance, True
