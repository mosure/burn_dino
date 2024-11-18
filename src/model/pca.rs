use burn::{
    prelude::*,
    module::Param,
    nn::Initializer,
};


#[derive(Config)]
pub struct PcaTransformConfig {
    pub input_dim: usize,
    pub output_dim: usize,

    #[config(default = "Initializer::Constant{value:1e-5}")]
    pub initializer: Initializer,
}

impl Default for PcaTransformConfig {
    fn default() -> Self {
        Self::new(384, 3)
    }
}

impl PcaTransformConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PcaTransform<B> {
        PcaTransform::new(device, &self)
    }
}


#[derive(Module, Debug)]
pub struct PcaTransform<B: Backend> {
    // TODO: add auxillary features
    pub components: Param<Tensor<B, 2>>,
    pub mean: Param<Tensor<B, 2>>,
}

impl<B: Backend> PcaTransform<B> {
    pub fn new(
        device: &B::Device,
        config: &PcaTransformConfig,
    ) -> Self {
        let components = config.initializer.init([config.output_dim, config.input_dim], device);
        let mean = config.initializer.init([1, config.input_dim], device);

        Self {
            components,
            mean,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let transformed = x.matmul(self.components.val().transpose());
        transformed - self.mean.val().matmul(self.components.val().transpose())
    }
}
