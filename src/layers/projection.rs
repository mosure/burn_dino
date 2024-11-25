use burn::{
    prelude::*,
    nn::{
        conv::{
            Conv2d,
            Conv2dConfig,
        },
        Gelu,
        Dropout,
        DropoutConfig,
    },
};


#[derive(Config)]
pub struct ProjectionConfig {
    pub input_dim: usize,
    pub output_dim: usize,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl ProjectionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Projection<B> {
        Projection::new(device, &self)
    }
}


#[derive(Module, Debug)]
pub struct Projection<B: Backend> {
    pub conv1: Conv2d<B>,
    pub gelu: Gelu,
    pub dropout: Dropout,
    pub conv2: Conv2d<B>,
}

impl<B: Backend> Projection<B> {
    pub fn new(
        device: &B::Device,
        config: &ProjectionConfig,
    ) -> Self {
        let conv1 = Conv2dConfig::new(
                [config.input_dim, config.output_dim],
                [1, 1],
            )
            .with_bias(true)
            .with_stride([1, 1])
            .init(device);

        let gelu = Gelu::new();
        let dropout = DropoutConfig::new(0.1).init();

        let conv2 = Conv2dConfig::new(
                [config.output_dim, config.output_dim],
                [1, 1],
            )
            .with_bias(true)
            .with_stride([1, 1])
            .init(device);

        Self {
            conv1,
            gelu,
            dropout,
            conv2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x);

        x
    }
}
