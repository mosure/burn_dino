use burn::prelude::*;

use crate::layers::{
    attention::{
        Attention,
        AttentionConfig,
    },
    layer_scale::{
        LayerScale,
        LayerScaleConfig,
    },
    mlp::{
        Mlp,
        MlpConfig,
    }
};


#[derive(Config)]
struct TransformerBlockConfig {
    attn: AttentionConfig,
    layer_scale: Option<LayerScaleConfig>,
    mlp_ratio: f32,
}


#[derive(Module, Debug)]
struct TransformerBlock<B: Backend> {
    norm1: nn::LayerNorm<B>,
    attention: Attention<B>,
    ls1: Option<LayerScale<B, 3>>,
    // TODO: drop_path_1

    norm2: nn::LayerNorm<B>,
    mlp: Mlp<B, 3>,
    ls2: Option<LayerScale<B, 3>>,
    // TODO: drop_path_2
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        device: &B::Device,
        config: TransformerBlockConfig,
    ) -> Self {
        let norm1 = nn::LayerNormConfig::new(config.attn.dim).init(device);
        let attention = config.attn.init(device);

        // self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        // self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        let norm2 = nn::LayerNormConfig::new(config.attn.dim).init(device);

        let mlp_hidden_dim = (config.attn.dim as f32 * config.mlp_ratio) as usize;
        let mlp = MlpConfig::new(config.attn.dim)
            .with_hidden_features(mlp_hidden_dim.into())
            .with_bias(true.into())
            .init::<B, 3>(device);

        // self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        // self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        Self {
            norm1,
            attention,
            ls1: None,
            norm2,
            mlp,
            ls2: None,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // TODO: implement train mode drop_path and `drop_add_residual_stochastic_depth` for sample_drop_ratio > 0.1

        let residual = self.attention.forward(self.norm1.forward(x.clone()));
        let residual = if let Some(ls1) = &self.ls1 {
            ls1.forward(residual)
        } else {
            residual
        };
        let x = x + residual;

        let residual = self.mlp.forward(self.norm2.forward(x.clone()));
        let residual = if let Some(ls2) = &self.ls2 {
            ls2.forward(residual)
        } else {
            residual
        };
        x + residual
    }
}
