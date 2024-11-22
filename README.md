# burn_dino
burn dinov2 model 🔥

| input               | fg pca               |
|-----------------------|-----------------------|
| ![Alt text](./assets/images/dino_0.png)    | ![Alt text](./docs/images/dino_0_pca.png)    |
| ![Alt text](./assets/images/dino_1.png)    | ![Alt text](./docs/images/dino_1_pca.png)    |
| ![Alt text](./assets/images/dino_2.png)    | ![Alt text](./docs/images/dino_2_pca.png)    |
| ![Alt text](./assets/images/dino_3.png)    | ![Alt text](./docs/images/dino_3_pca.png)    |

`cargo run --example pca`


## features

- [x] inference
- [x] pca transform layer
- [x] all ViT configurations
- [ ] training (loss + dropout)
- [ ] optimized attention
- [ ] automatic weights cache/download


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
