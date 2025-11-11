use burn::prelude::*;

#[derive(Config, Debug)]
pub struct MlpConfig {
    pub in_features: usize,
    pub hidden_features: Option<usize>,
    pub out_features: Option<usize>,
    pub dropout: Option<nn::DropoutConfig>,
    pub bias: Option<bool>,
}

impl MlpConfig {
    pub fn init<B: Backend, const D: usize>(&self, device: &B::Device) -> Mlp<B, D> {
        Mlp::new(device, self.clone())
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend, const D: usize> {
    pub act: nn::Gelu,
    pub dropout: nn::Dropout,
    pub fc1: nn::Linear<B>,
    pub fc2: nn::Linear<B>,
}

impl<B: Backend, const D: usize> Mlp<B, D> {
    fn new(device: &B::Device, config: MlpConfig) -> Self {
        let hidden_features = config.hidden_features.unwrap_or(config.in_features);
        let fc1 = nn::LinearConfig::new(config.in_features, hidden_features)
            .with_bias(config.bias.unwrap_or(false))
            .init(device);

        let out_features = config.out_features.unwrap_or(config.in_features);
        let fc2 = nn::LinearConfig::new(hidden_features, out_features)
            .with_bias(config.bias.unwrap_or(false))
            .init(device);

        let act = nn::Gelu::new();
        let dropout = config.dropout.unwrap_or(nn::DropoutConfig::new(0.0)).init();

        Self {
            act,
            dropout,
            fc1,
            fc2,
        }
    }

    pub fn forward(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        let x = self.dropout.forward(x);
        let x = self.fc2.forward(x);
        self.dropout.forward(x)
    }
}
