use burn::{
    prelude::*,
    module::Param,
    nn::Initializer,
};


#[derive(Config)]
pub struct LayerScaleConfig {
    pub dim: usize,

    #[config(default = "Initializer::Constant{value:1e-5}")]
    pub initializer: Initializer,
}

impl Default for LayerScaleConfig {
    fn default() -> Self {
        Self::new(0)
    }
}

impl LayerScaleConfig {
    pub fn init<B: Backend, const D: usize>(&self, device: &B::Device) -> LayerScale<B, D> {
        LayerScale::new(device, &self)
    }
}


#[derive(Module, Debug)]
pub struct LayerScale<B: Backend, const D: usize> {
    pub gamma: Param<Tensor<B, 1>>,
}

impl<B: Backend, const D: usize> LayerScale<B, D> {
    pub fn new(
        device: &B::Device,
        config: &LayerScaleConfig,
    ) -> Self {
        let gamma = config.initializer.init([config.dim], device);

        Self {
            gamma,
        }
    }

    pub fn forward(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let shape = x.shape();
        x.mul(self.gamma.val().expand(shape))
    }
}
