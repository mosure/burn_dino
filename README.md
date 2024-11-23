# burn_dino 🔥🦖

[![test](https://github.com/mosure/burn_dino/workflows/test/badge.svg)](https://github.com/Mosure/burn_dino/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/burn_dino)](https://raw.githubusercontent.com/mosure/burn_dino/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/mosure/burn_dino)](https://github.com/mosure/burn_dino)
[![GitHub Releases](https://img.shields.io/github/v/release/mosure/burn_dino?include_prereleases&sort=semver)](https://github.com/mosure/burn_dino/releases)
[![GitHub Issues](https://img.shields.io/github/issues/mosure/burn_dino)](https://github.com/mosure/burn_dino/issues)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/mosure/burn_dino.svg)](http://isitmaintained.com/project/mosure/burn_dino)
[![crates.io](https://img.shields.io/crates/v/burn_dino.svg)](https://crates.io/crates/burn_dino)


burn [dinov2](https://arxiv.org/abs/2304.07193) model, view the [live demo]()

| input               | fg pca               |
|-----------------------|-----------------------|
| ![Alt text](./assets/images/dino_0.png)    | ![Alt text](./docs/images/dino_0_pca.png)    |
| ![Alt text](./assets/images/dino_1.png)    | ![Alt text](./docs/images/dino_1_pca.png)    |
| ![Alt text](./assets/images/dino_2.png)    | ![Alt text](./docs/images/dino_2_pca.png)    |
| ![Alt text](./assets/images/dino_3.png)    | ![Alt text](./docs/images/dino_3_pca.png)    |

`cargo run --example pca`
`cargo run -p bevy_burn_dino`


## features

- [x] inference
- [x] pca transform layer
- [x] ViT configurations
- [x] real-time camera demo
- [ ] training (loss + dropout)
- [ ] optimized attention
- [ ] automatic weights cache/download
- [ ] quantization
- [ ] feature upsampling


## setup
- download pre-trained model (ViT-S|B, /wo registers) from [here](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models)
- place in `./assets/models`
- run import tool `cargo run --bin import`

<!---
TODO: release converted/quantized mpk models /w net loader
-->


## benchmarks

- `cargo bench`
- open `target/criterion/report/index.html`
