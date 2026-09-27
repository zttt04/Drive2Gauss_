# Third-party notices

The BSD-3-Clause license in this repository applies only to original
Drive2Gauss code. It does not relicense datasets, model weights, or third-party
software.

| Component | Use in Drive2Gauss | Upstream and license source |
| --- | --- | --- |
| DiST-4D / DiST-T | Base multimodal diffusion implementation in `third_party/distt` | [royalmelon0505/dist4d](https://github.com/royalmelon0505/dist4d); the upstream repository currently exposes no LICENSE file, so redistribution permission must be obtained before publishing this vendored tree |
| MagicDrive-V2 | Architecture and data-pipeline ancestry of DiST-T | [flymin/MagicDrive-V2](https://github.com/flymin/MagicDrive-V2); retain its upstream license and notices |
| CogVideoX | RGB/depth/flow VAE initialization | [THUDM/CogVideoX](https://github.com/THUDM/CogVideo) |
| gsplat | Gaussian rasterization | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) |
| SEA-RAFT | Optical-flow preprocessing | [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) |
| Grounded SAM 2 | Optional dynamic-object masks | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |
| Turbo-VAED | Latent decoding and feature extraction | Obtain from its upstream distribution and follow its model/code terms |
| nuScenes | Training and evaluation data | [nuScenes terms](https://www.nuscenes.org/terms-of-use) |
| LPIPS | Perceptual metric | [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) |

Users must download external datasets and pretrained weights themselves and
accept the corresponding terms. The current `third_party/distt` directory must
not be published under the Drive2Gauss BSD license unless the DiST-4D authors
provide redistribution permission. Without that permission, replace it with a
pinned external dependency and publish only the Drive2Gauss patch/adaptation.
