use burn::prelude::*;


#[derive(Config)]
pub struct LayerScaleConfig {
    pub dim: usize,
    pub init_values: f32,
}

impl Default for LayerScaleConfig {
    fn default() -> Self {
        Self {
            dim: 0,
            init_values: 1e-5,
        }
    }
}


#[derive(Module, Debug)]
pub struct LayerScale<B: Backend, const D: usize> {
    pub gamma: Tensor<B, 1>,
}

impl<B: Backend, const D: usize> LayerScale<B, D> {
    pub fn new(
        device: &B::Device,
        config: LayerScaleConfig,
    ) -> Self {
        let gamma = Tensor::ones(
            [config.dim],
            device,
        )
        .mul_scalar(config.init_values);

        Self {
            gamma,
        }
    }

    pub fn forward(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let shape = x.shape();
        x.mul(self.gamma.clone().expand(shape))
    }
}
