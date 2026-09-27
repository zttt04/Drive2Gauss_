from __future__ import annotations

import torch
from torch import nn

from DISTT.utils.frechet_rgb_stage4 import (
    Stage4PerViewFrechetRGBEMALoss,
    Stage4PerViewFrechetRGBLoss,
    _frechet_from_stats,
    build_per_view_rgb_features,
    memory_bounded_cogvideox_decode,
    select_single_camera_latent,
)


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv3d(2, 2, 1, bias=False)
        self.decoder = nn.Conv3d(2, 3, 1, bias=False)
        self.use_tiling = True
        self.tile_latent_min_height = 2
        self.tile_latent_min_width = 2
        self.tile_sample_min_height = 2
        self.tile_sample_min_width = 2
        self.tile_overlap_factor_height = 0.5
        self.tile_overlap_factor_width = 0.5
        self.num_latent_frames_batch_size = 2

    def _clear_fake_context_parallel_cache(self):
        return None

    def blend_v(self, a, b, extent):
        result = b.clone()
        for offset in range(extent):
            result[..., offset, :] = a[..., -extent + offset, :] * (1 - offset / extent) + b[..., offset, :] * (offset / extent)
        return result

    def blend_h(self, a, b, extent):
        result = b.clone()
        for offset in range(extent):
            result[..., offset] = a[..., -extent + offset] * (1 - offset / extent) + b[..., offset] * (offset / extent)
        return result

    def tiled_decode(self, z, return_dict=True):
        rows, columns = [], []
        overlap_h = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
        overlap_w = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
        blend_h = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_w = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        limit_h = self.tile_sample_min_height - blend_h
        limit_w = self.tile_sample_min_width - blend_w
        for row in range(0, z.shape[-2], overlap_h):
            row_tiles = []
            for column in range(0, z.shape[-1], overlap_w):
                chunks = []
                for start, end in ((0, 3),):
                    tile = self.decoder(self.post_quant_conv(z[:, :, start:end, row:row + 2, column:column + 2]))
                    chunks.append(tile)
                row_tiles.append(torch.cat(chunks, dim=2))
            rows.append(row_tiles)
        result_rows = []
        for row_index, row in enumerate(rows):
            result_row = []
            for column_index, tile in enumerate(row):
                if row_index:
                    tile = self.blend_v(rows[row_index - 1][column_index], tile, blend_h)
                if column_index:
                    tile = self.blend_h(row[column_index - 1], tile, blend_w)
                result_row.append(tile[:, :, :, :limit_h, :limit_w])
            result_rows.append(torch.cat(result_row, dim=4))
        output = torch.cat(result_rows, dim=3)
        return type("DecodeOutput", (), {"sample": output})()


class _TinyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = _TinyModule()
        self.scaling_factor = 1.0


def test_bounded_decode_matches_tiled_vjp():
    torch.manual_seed(7)
    wrapper = _TinyWrapper().eval()
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    latent = torch.randn(1, 2, 3, 3, 3, requires_grad=True)
    bounded = memory_bounded_cogvideox_decode(wrapper, latent)
    reference_latent = latent.detach().clone().requires_grad_(True)
    module = wrapper.module
    reference = module.tiled_decode(reference_latent).sample
    assert bounded.shape == reference.shape
    assert torch.allclose(bounded, reference, atol=1e-6, rtol=1e-5)
    output_gradient = torch.randn_like(bounded)
    bounded.backward(output_gradient)
    reference.backward(output_gradient)
    assert torch.allclose(latent.grad, reference_latent.grad, atol=1e-6, rtol=1e-5)


def test_stage4_fd_is_per_view_and_differentiable():
    torch.manual_seed(3)
    reference = torch.randn(12, 2, 4)
    mu = reference.mean(0)
    centered = reference - mu
    cov = torch.einsum("nvd,nve->vde", centered, centered) / (reference.shape[0] - 1)
    loss_state = Stage4PerViewFrechetRGBLoss(mu, cov, queue_size=0, min_population=2)
    fake = (reference[:4] + 0.05 * torch.randn(4, 2, 4)).requires_grad_(True)
    value, valid = loss_state(fake)
    assert valid
    value.backward()
    assert fake.grad is not None
    assert fake.grad.shape == fake.shape


def test_rgb_features_keep_camera_axis():
    decoded = torch.zeros(4 * 2, 3, 3, 8, 8)
    decoded[1::2] = 1
    features = build_per_view_rgb_features(decoded, batch_size=4, num_cameras=2, pool_size=2)
    assert features.shape == (4, 2, 12)
    assert not torch.equal(features[:, 0], features[:, 1])


def test_select_single_camera_latent_keeps_batch_order():
    latent = torch.arange(2 * 3, dtype=torch.float32).view(6, 1, 1, 1, 1)
    selected = select_single_camera_latent(latent, batch_size=2, num_cameras=3, camera_index=1)
    assert selected.flatten().tolist() == [1.0, 4.0]


def test_stage4_queue_prefill_truncates_and_update_appends():
    reference = torch.randn(8, 1, 4)
    mu = reference.mean(0)
    centered = reference - mu
    cov = torch.einsum("nvd,nve->vde", centered, centered) / (reference.shape[0] - 1)
    loss_state = Stage4PerViewFrechetRGBLoss(mu, cov, queue_size=4, min_population=2)
    loss_state.prefill(reference[:6])
    assert torch.equal(loss_state.fake_queue, reference[2:6])
    update = torch.full((1, 1, 4), 9.0)
    loss_state.update(update)
    assert torch.equal(loss_state.fake_queue[:-1], reference[3:6])
    assert torch.equal(loss_state.fake_queue[-1:], update)


def test_cached_reference_sqrt_matches_direct_frechet():
    torch.manual_seed(11)
    reference = torch.randn(12, 1, 4)
    fake = torch.randn(9, 1, 4)
    ref_mu = reference.mean(0)
    ref_centered = reference - ref_mu
    ref_cov = torch.einsum("nvd,nve->vde", ref_centered, ref_centered) / (reference.shape[0] - 1)
    fake_mu = fake.mean(0)
    fake_centered = fake - fake_mu
    fake_cov = torch.einsum("nvd,nve->vde", fake_centered, fake_centered) / (fake.shape[0] - 1)
    loss_state = Stage4PerViewFrechetRGBLoss(ref_mu, ref_cov, queue_size=0, min_population=2)
    direct = _frechet_from_stats(fake_mu, fake_cov, ref_mu, ref_cov, loss_state.eps)
    cached = _frechet_from_stats(
        fake_mu,
        fake_cov,
        ref_mu,
        ref_cov,
        loss_state.eps,
        ref_sqrt=loss_state.reference_cov_sqrt,
    )
    assert torch.allclose(cached, direct, atol=1e-6, rtol=1e-5)


def test_stage4_ema_normalizes_each_view_and_updates_detached_history():
    torch.manual_seed(19)
    reference = torch.randn(32, 2, 4)
    ref_mu = reference.mean(0)
    centered = reference - ref_mu
    ref_cov = torch.einsum("nvd,nve->vde", centered, centered) / (reference.shape[0] - 1)
    state = Stage4PerViewFrechetRGBEMALoss(ref_mu, ref_cov, decay=0.9)
    state.prefill(reference[:16])
    fake = (reference[16:20] + torch.tensor([0.2, 0.8]).view(1, 2, 1)).requires_grad_(True)
    normalized, raw_per_view, normalized_per_view = state(fake)
    assert raw_per_view.shape == normalized_per_view.shape == (2,)
    assert torch.allclose(normalized, normalized_per_view.mean())
    assert torch.allclose(
        normalized_per_view,
        raw_per_view / (raw_per_view.detach() + 0.01),
    )
    normalized.backward()
    assert fake.grad is not None
    state.update(fake.detach())
    assert state.fake_mu.grad_fn is None
    assert state.fake_second_moment.grad_fn is None


def test_stage4_ema_state_round_trip():
    reference = torch.randn(12, 1, 4)
    ref_mu = reference.mean(0)
    centered = reference - ref_mu
    ref_cov = torch.einsum("nvd,nve->vde", centered, centered) / (reference.shape[0] - 1)
    source = Stage4PerViewFrechetRGBEMALoss(ref_mu, ref_cov)
    source.prefill(reference)
    restored = Stage4PerViewFrechetRGBEMALoss(ref_mu, ref_cov)
    restored.load_state_dict(source.state_dict())
    assert torch.equal(restored.fake_mu, source.fake_mu)
    assert torch.equal(restored.fake_second_moment, source.fake_second_moment)
