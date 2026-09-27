# Third-party notices

The BSD-3-Clause license in this repository applies only to original
Drive2Gauss code. It does not relicense datasets, model weights, or third-party
software.

| Component | Use in Drive2Gauss | Upstream and license source |
| --- | --- | --- |
| MagicDrive-V2 | Architecture and data-pipeline ancestry of DiST-T | [flymin/MagicDrive-V2](https://github.com/flymin/MagicDrive-V2); retain its upstream license and notices |
| CogVideoX | RGB/depth/flow VAE initialization | [THUDM/CogVideoX](https://github.com/THUDM/CogVideo) |
| gsplat | Gaussian rasterization | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) |
| SEA-RAFT | Optical-flow preprocessing | [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) |
| Grounded SAM 2 | Optional dynamic-object masks | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |
| Turbo-VAED | Latent decoding and feature extraction | Obtain from its upstream distribution and follow its model/code terms |
| nuScenes | Training and evaluation data | [nuScenes terms](https://www.nuscenes.org/terms-of-use) |
| LPIPS | Perceptual metric | [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) |

Users must download external datasets and pretrained weights themselves and
accept the corresponding terms. The development origin of the integrated
generator is documented separately in [generator_origin.md](generator_origin.md).
