use burn::{module::Param, nn::Initializer, prelude::*};

#[derive(Config, Debug)]
pub struct LayerScaleConfig {
    pub dim: usize,
}

impl Default for LayerScaleConfig {
    fn default() -> Self {
        Self::new(0)
    }
}

impl LayerScaleConfig {
    pub fn init<B: Backend, const D: usize>(&self, device: &B::Device) -> LayerScale<B, D> {
        LayerScale::new(device, self)
    }
}

#[derive(Module, Debug)]
pub struct LayerScale<B: Backend, const D: usize> {
    pub gamma: Param<Tensor<B, 1>>,
}

impl<B: Backend, const D: usize> LayerScale<B, D> {
    pub fn new(device: &B::Device, config: &LayerScaleConfig) -> Self {
        let gamma = Initializer::Constant { value: 1e-5 }.init([config.dim], device);

        Self { gamma }
    }

    pub fn forward(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let shape = x.shape();
        x.mul(self.gamma.val().expand(shape))
    }
}
