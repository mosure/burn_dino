use burn::prelude::*;

use crate::layers::{
    attention::{
        Attention,
        AttentionConfig,
    },
    layer_norm::{
        LayerNorm,
        LayerNormConfig,
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
pub struct BlockConfig {
    pub attn: AttentionConfig,
    pub layer_scale: Option<LayerScaleConfig>,
    pub mlp_ratio: f32,
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            attn: AttentionConfig::default(),
            layer_scale: None,
            mlp_ratio: 4.0,
        }
    }
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block::new(device, self.clone())
    }
}


#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    ls1: Option<LayerScale<B, 3>>,
    // TODO: drop_path_1

    norm2: LayerNorm<B>,
    mlp: Mlp<B, 3>,
    ls2: Option<LayerScale<B, 3>>,
    // TODO: drop_path_2
}

impl<B: Backend> Block<B> {
    pub fn new(
        device: &B::Device,
        config: BlockConfig,
    ) -> Self {
        let norm1 = LayerNormConfig::new(config.attn.dim).init(device);
        let attn = config.attn.init(device);

        // self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        // self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        let ls1 = if let Some(layer_scale_config) = &config.layer_scale {
            layer_scale_config.init::<B, 3>(device).into()
        } else {
            None
        };

        let norm2 = LayerNormConfig::new(config.attn.dim).init(device);

        let mlp_hidden_dim = (config.attn.dim as f32 * config.mlp_ratio) as usize;
        let mlp = MlpConfig::new(config.attn.dim)
            .with_hidden_features(mlp_hidden_dim.into())
            .with_bias(true.into())
            .init::<B, 3>(device);

        let ls2 = if let Some(layer_scale_config) = &config.layer_scale {
            layer_scale_config.init::<B, 3>(device).into()
        } else {
            None
        };
        // self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3>
    {
        // TODO: implement train mode drop_path and `drop_add_residual_stochastic_depth` for sample_drop_ratio > 0.1

        let norm = self.norm1.forward(x.clone());
        let residual = self.attn.forward(norm.clone());
        let residual = if let Some(ls1) = &self.ls1 {
            ls1.forward(residual)
        } else {
            residual
        };

        let x = x + residual;

        let norm = self.norm2.forward(x.clone());
        let mlp = self.mlp.forward(norm.clone());
        let residual = if let Some(ls2) = &self.ls2 {
            ls2.forward(mlp.clone())
        } else {
            mlp.clone()
        };
        x + residual
    }
}
