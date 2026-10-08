# Released checkpoints

Drive2Gauss uses two checkpoints in the normal generated-latent pipeline. The
files are not stored in Git. Download them from the
[Google Drive checkpoint folder](https://drive.google.com/drive/folders/1qGS1ix6Krm3LLeDMmzKqd4sgKRCewm-c).
The folder currently redirects unauthenticated visitors to Google sign-in;
access depends on the folder's sharing settings.

| Stage | Published filename | Purpose |
| --- | --- | --- |
| Multimodal diffusion EMA | `drive2gauss_step3600_ema.pt` (8,154,303,318 bytes; SHA256 `255bbd0a9dba345291de61db59cb45c5dfedc67a049e1a2e990c60fe7c62b6b8`) | Generate 17-frame, six-view RGB-D-flow latents |
| Gaussian decoder | `drive2gauss_feature_unet_step3744.pt` (4,471,940 bytes; SHA256 `52b788deaf2c7dbb62e297faa4f9e99dce3de321cb5c453cfc767605d5416502`) | Decode generated RGB-D-flow latents into dynamic Gaussians and render RGB |

Only the generator EMA weights are in the released file; the complete training
state and optimizer state are not part of this download. CogVideoX, T5, and
other separately licensed pretrained assets must be obtained from their
upstream sources. For the directory layout in `configs/paths.env.example`,
place the downloaded generator file at
`checkpoints/drive2gauss_step3600/ema.pt`.

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

The SHA256 values above were verified against the uploaded checkpoint bytes.
Do not substitute a direct-RGB or zero-flow experiment under these filenames.
