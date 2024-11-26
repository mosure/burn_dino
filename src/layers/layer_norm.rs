use burn::{
    prelude::*,
    module::Param,
    nn::Initializer,
};


#[derive(Config)]
pub struct LayerNormConfig {
    pub dim: usize,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self::new(0)
    }
}

impl LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        LayerNorm::new(device, self)
    }
}


#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(
        device: &B::Device,
        config: &LayerNormConfig,
    ) -> Self {
        let gamma = Initializer::Ones.init([config.dim], device);
        let beta = Initializer::Zeros.init([config.dim], device);

        Self {
            gamma,
            beta,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let n = x.shape().dims[D - 1] as f32;

        let mean = x.clone().mean_dim(D - 1);
        let diff = x.clone().sub(mean);
        let var = diff.clone().powi_scalar(2).sum_dim(D - 1).div_scalar(n);

        let input_normalized = diff.div(var.add_scalar(1e-5).sqrt());

        // TODO: numerically different than torch layernorm, write test
        input_normalized
            .mul(self.gamma.val().unsqueeze())
            .add(self.beta.val().unsqueeze())
    }
}
