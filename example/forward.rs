use burn::{
    prelude::*,
    backend::Wgpu,
};

use burn_dinov2::model::dinov2::DinoVisionTransformerConfig;


fn main() {
    let device = Default::default();

    let config = DinoVisionTransformerConfig::vits();
    let dino = config.init(&device);

    let input: Tensor<Wgpu, 4> = Tensor::zeros([1, config.input_channels, config.image_size, config.image_size], &device);
    dino.forward(input.clone(), None);
}
