use burn::prelude::*;

#[derive(Config, Debug)]
pub struct PatchEmbedConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub input_channels: usize,
    pub embedding_dimension: usize,
}

impl Default for PatchEmbedConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            input_channels: 3,
            embedding_dimension: 768,
        }
    }
}

impl PatchEmbedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbed<B> {
        PatchEmbed::new(device, self.clone())
    }
}

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    proj: nn::conv::Conv2d<B>,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(device: &B::Device, config: PatchEmbedConfig) -> Self {
        let kernel_size = [config.patch_size, config.patch_size];
        let proj = nn::conv::Conv2dConfig::new(
            [config.input_channels, config.embedding_dimension],
            kernel_size,
        )
        .with_stride(kernel_size)
        .init(device);

        Self { proj }
    }

    #[allow(non_snake_case)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        self.proj.forward(x).flatten(2, 3).swap_dims(1, 2)
    }
}
