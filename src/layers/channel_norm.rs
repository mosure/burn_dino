use burn::prelude::*;

use crate::layers::layer_norm::{
    LayerNorm,
    LayerNormConfig,
};


#[derive(Config)]
pub struct ChannelNormConfig {
    pub dim: usize,
}

impl Default for ChannelNormConfig {
    fn default() -> Self {
        Self::new(0)
    }
}

impl ChannelNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ChannelNorm<B> {
        ChannelNorm::new(device, &self)
    }
}


#[derive(Module, Debug)]
pub struct ChannelNorm<B: Backend> {
    pub layer_norm: LayerNorm<B>,
}

impl<B: Backend> ChannelNorm<B> {
    pub fn new(
        device: &B::Device,
        config: &ChannelNormConfig,
    ) -> Self {
        let layer_norm = LayerNormConfig {
            dim: config.dim,
        }.init(device);

        Self {
            layer_norm
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.layer_norm.forward(x.permute([0, 2, 3, 1]))
            .permute([0, 3, 1, 2])
    }
}
