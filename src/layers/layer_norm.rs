use burn::{module::Param, nn::Initializer, prelude::*};

#[derive(Config, Debug)]
pub struct LayerNormConfig {
    pub dim: usize,
    #[config(default = 1e-6)]
    pub epsilon: f64,
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
    epsilon: f64,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(device: &B::Device, config: &LayerNormConfig) -> Self {
        let gamma = Initializer::Ones.init([config.dim], device);
        let beta = Initializer::Zeros.init([config.dim], device);

        Self {
            gamma,
            beta,
            epsilon: config.epsilon,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let (var, mean) = x.clone().var_mean_bias(D - 1);
        let input_normalized = x.sub(mean).div(var.add_scalar(self.epsilon).sqrt());

        input_normalized
            .mul(self.gamma.val().unsqueeze())
            .add(self.beta.val().unsqueeze())
    }
}
