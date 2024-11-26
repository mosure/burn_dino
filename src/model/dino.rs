use burn::{
    prelude::*,
    module::Param,
    nn::{
        Gelu,
        Initializer,
    },
};

use crate::layers::{
    attention::AttentionConfig,
    block::{
        Block,
        BlockConfig,
    },
    layer_norm::{
        LayerNorm,
        LayerNormConfig,
    },
    layer_scale::LayerScaleConfig,
    patch_embed::{
        PatchEmbed,
        PatchEmbedConfig,
    },
};


#[derive(Config)]
pub struct DinoVisionTransformerConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub input_channels: usize,
    pub embedding_dimension: usize,
    pub depth: usize,
    pub block_config: BlockConfig,
    pub positional_encoding_interpolate: nn::interpolate::Interpolate2dConfig,
    pub num_patches: usize,

    #[config(default = "Initializer::Normal{mean:0.02, std:1.0}")]
    pub initializer: Initializer,
}

impl DinoVisionTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DinoVisionTransformer<B> {
        DinoVisionTransformer::new(device, self.clone())
    }

    #[allow(non_snake_case)]
    pub fn from_size(image_size: Option<usize>, patch_size: Option<usize>) -> Self {
        let image_size = image_size.unwrap_or(518);
        let patch_size = patch_size.unwrap_or(14);

        let interpolate_size = [image_size / patch_size, image_size / patch_size];

        let dim = 768;

        let w0 = image_size / patch_size;
        let h0 = image_size / patch_size;
        let num_patches = w0 * h0;

        // let M = num_patches.isqrt();
        // let sx = (w0 as f32 * 0.1) / M as f32;
        // let sy = (h0 as f32 * 0.1) / M as f32;

        Self::new(
            image_size,
            patch_size,
            3,
            dim,
            12,
            BlockConfig {
                attn: AttentionConfig {
                    dim,
                    ..Default::default()
                },
                layer_scale: LayerScaleConfig {
                    dim,
                }.into(),
                ..Default::default()
            },
            nn::interpolate::Interpolate2dConfig {
                mode: nn::interpolate::InterpolateMode::Cubic,
                output_size: interpolate_size.into(),
                scale_factor: None, //[sx, sy].into(),
            },
            num_patches,
        )
    }

    pub fn vits(image_size: Option<usize>, patch_size: Option<usize>) -> Self {
        let embedding_dimension = 384;
        Self {
            embedding_dimension,
            block_config: BlockConfig {
                attn: AttentionConfig {
                    dim: embedding_dimension,
                    num_heads: 6,
                    ..Default::default()
                },
                layer_scale: LayerScaleConfig {
                    dim: embedding_dimension,
                }.into(),
                ..Default::default()
            },
            ..Self::from_size(image_size, patch_size)
        }
    }

    pub fn vitb(image_size: Option<usize>, patch_size: Option<usize>) -> Self {
        Self::from_size(image_size, patch_size)
    }

    pub fn vitl(image_size: Option<usize>, patch_size: Option<usize>) -> Self {
        let embedding_dimension = 1024;
        Self {
            embedding_dimension,
            depth: 24,
            block_config: BlockConfig {
                attn: AttentionConfig {
                    dim: embedding_dimension,
                    num_heads: 16,
                    ..Default::default()
                },
                layer_scale: LayerScaleConfig {
                    dim: embedding_dimension,
                }.into(),
                ..Default::default()
            },
            ..Self::from_size(image_size, patch_size)
        }
    }

    pub fn vitg(image_size: Option<usize>, patch_size: Option<usize>) -> Self {
        let embedding_dimension = 1536;
        Self {
            embedding_dimension,
            depth: 40,
            block_config: BlockConfig {
                attn: AttentionConfig {
                    dim: embedding_dimension,
                    num_heads: 24,
                    ..Default::default()
                },
                layer_scale: LayerScaleConfig {
                    dim: embedding_dimension,
                }.into(),
                ..Default::default()
            },
            ..Self::from_size(image_size, patch_size)
        }
    }
}


#[derive(Debug, Clone)]
pub struct DinoOutput<B: Backend> {
    pub x_norm_clstoken: Tensor<B, 2>,
    pub x_norm_patchtokens: Tensor<B, 3>,
    pub x_prenorm: Tensor<B, 3>,
    pub masks: Option<Tensor<B, 3, Bool>>,
}


#[derive(Module, Debug)]
pub struct DinoVisionTransformer<B: Backend> {
    activation: Gelu,
    cls_token: Param<Tensor<B, 3>>,
    pub pos_embed: Param<Tensor<B, 3>>,
    mask_token: Param<Tensor<B, 2>>,
    interpolate: nn::interpolate::Interpolate2d,
    patch_embed: PatchEmbed<B>,
    norm: LayerNorm<B>,
    blocks: Vec<Block<B>>,
    patch_size: usize,
}

impl<B: Backend> DinoVisionTransformer<B> {
    pub fn new(
        device: &B::Device,
        config: DinoVisionTransformerConfig,
    ) -> Self {
        // TODO: initialize cls_token and pos_embed with trainable weights
        // trunc_normal_(self.pos_embed, std=0.02)
        // nn.init.normal_(self.cls_token, std=1e-6)
        // if self.register_tokens is not None:
        //     nn.init.normal_(self.register_tokens, std=1e-6)
        // named_apply(init_weights_vit_timm, self)
        // if isinstance(module, nn.Linear):
        // trunc_normal_(module.weight, std=0.02)
        // if module.bias is not None:
        //     nn.init.zeros_(module.bias)

        let cls_token = config.initializer.init(
            [1, 1, config.embedding_dimension],
            device,
        );

        let num_tokens = 1;
        let pos_embed = config.initializer.init(
            [1, config.num_patches + num_tokens, config.embedding_dimension],
            device,
        );

        let mask_token = config.initializer.init(
            [1, config.embedding_dimension],
            device,
        );

        let interpolate = config.positional_encoding_interpolate.init();

        let patch_embed = PatchEmbedConfig::new(
            config.image_size,
            config.patch_size,
            config.input_channels,
            config.embedding_dimension,
        ).init(device);

        let norm: LayerNorm<B> = LayerNormConfig::new(config.embedding_dimension)
            .init(device);

        let mut blocks = Vec::with_capacity(config.depth);
        for _ in 0..config.depth {
            let block = config.block_config.init(device);
            blocks.push(block);
        }

        Self {
            activation: Gelu::new(),
            cls_token,
            pos_embed,
            mask_token,
            interpolate,
            patch_embed,
            norm,
            blocks,
            patch_size: config.patch_size,
        }
    }

    #[allow(non_snake_case)]
    pub fn interpolate_pos_encoding(
        &self,
        x: Tensor<B, 3>,
        W: usize,
        H: usize,
    ) -> Tensor<B, 3> {
        let npatch = x.shape().dims[1] - 1;
        let N = self.pos_embed.shape().dims[1] - 1;

        if npatch == N && W == H {
            return self.pos_embed.val().clone();
        }

        let b_dim = self.pos_embed.shape().dims[0];
        let n_dim = self.pos_embed.shape().dims[1];
        // let c_dim: usize = self.pos_embed.shape().dims[2];

        let class_pos_embed: Tensor<B, 2> = self.pos_embed.val().clone().slice([0..b_dim, 0..1]).squeeze(1);
        let patch_pos_embed = self.pos_embed.val().clone().slice([0..b_dim, 1..n_dim]);
        let dim = x.shape().dims[2];
        let M = N.isqrt();

        assert!(
            N == M * M,
            "number of patches should be a square number",
        );

        let patch_pos_embed = self.interpolate.forward(
            patch_pos_embed.reshape([1, M, M, dim]).permute([0, 3, 1, 2]),
        ).permute([0, 2, 3, 1]).reshape([1_i32, -1, dim as i32]);

        Tensor::cat(
            vec![
                class_pos_embed.unsqueeze_dim(0),
                patch_pos_embed,
            ],
            1,
        )
    }

    #[allow(non_snake_case)]
    pub fn prepare_tokens_with_masks(
        &self,
        x: Tensor<B, 4>,
        mask: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 3> {
        // TODO: H, W?
        let [_B, _C, W, H] = x.shape().dims();

        let x = self.patch_embed.forward(x);
        let x = if let Some(mask) = mask {
            x.mask_where(mask, self.mask_token.val().unsqueeze_dim(0))
        } else {
            x
        };

        let x = Tensor::cat(
            vec![self.cls_token.val().expand([x.shape().dims[0] as i64, -1, -1]), x],
            1
        );

        let residual = self.interpolate_pos_encoding(x.clone(), W, H);
        x + residual.clone()
    }

    #[allow(non_snake_case)]
    pub fn intermediate_layers(
        &self,
        x: Tensor<B, 4>,
        layers: &[usize],
    ) -> (
        Vec<Tensor<B, 2>>,
        Vec<Tensor<B, 4>>,
    ) {
        let mut x = self.prepare_tokens_with_masks(x, None);

        let mut class_tokens = Vec::with_capacity(layers.len());
        let mut output = Vec::with_capacity(layers.len());

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(x);

            if layers.contains(&i) {
                let x = self.norm.forward(x.clone());

                let class_token: Tensor<B, 2> = x.clone().slice([0..x.shape().dims[0], 0..1]).squeeze(1);
                let out = x.clone().slice([0..x.shape().dims[0], 1..x.shape().dims[1]]);

                let [B, _, W, H] = x.shape().dims();
                let reshaped = out.reshape([B as i32, (W / self.patch_size) as i32, (H / self.patch_size) as i32, -1])
                    .permute([0, 3, 1, 2]);

                class_tokens.push(class_token);
                output.push(reshaped);
            }
        }

        (class_tokens, output)
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        masks: Option<Tensor<B, 3, Bool>>,
    ) -> DinoOutput<B> {
        let mut x = self.prepare_tokens_with_masks(x, None);

        for block in &self.blocks {
            x = block.forward(x);
        }

        let x_norm = self.norm.forward(x.clone());

        let b_dim = x.shape().dims[0];
        let n_dim = x.shape().dims[1];

        let x_norm_clstoken = x_norm.clone().slice([0..b_dim, 0..1]).squeeze(1);
        let x_norm_patchtokens = x_norm.clone().slice([0..b_dim, 1..n_dim]);

        DinoOutput {
            x_norm_clstoken,
            x_norm_patchtokens,
            x_prenorm: x,
            masks,
        }
    }
}
