# Released checkpoints

Drive2Gauss uses two checkpoints in the normal generated-latent pipeline. The
files are not stored in Git.

| Stage | Published filename | Purpose |
| --- | --- | --- |
| Multimodal diffusion | `drive2gauss_distt_step3600` | Generate 17-frame, six-view RGB-D-flow latents |
| Gaussian decoder | `drive2gauss_feature_unet_step3744.pt` | Decode generated RGB-D-flow latents into dynamic Gaussians and render RGB |

The Gaussian decoder is the feature-UNet checkpoint trained for generated
latents. It uses four contiguous windows starting at frames 0, 4, 8, and 12,
three front cameras, 80,000 queries per frame/view, and at most 960,000 queries
per window. It uses the generated flow normally; `zero_flow_input` is false.

## Reference evaluation

The full evaluation contains 1,976 clips and 94,848 rendered images:

| PSNR-infer | SSIM-infer | LPIPS-infer (AlexNet) |
| ---: | ---: | ---: |
| 29.90838 | 0.883813 | 0.101202 |

These metrics compare the Gaussian render with RGB decoded from the same
generated latent. They are not metrics against the original camera RGB. The
same evaluation reports LPIPS-GT 0.402198.

## Publication manifest

Before attaching files to a GitHub release or model hub, replace the two
`PENDING_UPLOAD` values below with immutable URLs and fill size/SHA256 from the
uploaded bytes. Do not publish a direct-RGB or zero-flow experiment under these
names.

```text
drive2gauss_distt_step3600               PENDING_UPLOAD
drive2gauss_feature_unet_step3744.pt     PENDING_UPLOAD
```
