use std::ops::RangeFull;

use burn::{
    prelude::*,
    nn::Gelu,
};

use crate::layers::{
    attention::AttentionConfig,
    block::{
        Block,
        BlockConfig,
    },
    patch_embed::{
        PatchEmbed,
        PatchEmbedConfig,
    },
};


#[derive(Config)]
struct DinoVisionTransformerConfig {
    image_size: usize,
    patch_size: usize,
    input_channels: usize,
    embedding_dimension: usize,
    depth: usize,
    block_config: BlockConfig,
    positional_encoding_interpolate: nn::interpolate::Interpolate2dConfig,
}

impl Default for DinoVisionTransformerConfig {
    // TODO: no default, only provide small, base, large, giant configurations
    fn default() -> Self {
        let image_size = 224;
        let patch_size = 16;

        let interpolate_size = [image_size / patch_size, image_size / patch_size];

        Self {
            image_size,
            patch_size,
            input_channels: 3,
            embedding_dimension: 768,
            depth: 12,
            block_config: BlockConfig::default(),
            positional_encoding_interpolate: nn::interpolate::Interpolate2dConfig {
                mode: nn::interpolate::InterpolateMode::Cubic,
                output_size: interpolate_size.into(),
                scale_factor: None,
            },
        }
    }
}

impl DinoVisionTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DinoVisionTransformer<B> {
        DinoVisionTransformer::new(device, self.clone())
    }

    // TODO: small, base, large, giant configurations
}


#[derive(Debug, Clone)]
struct DinoOutput<B: Backend> {
    pub x_norm_clstoken: Tensor<B, 2>,
    pub x_norm_regtokens: Tensor<B, 3>,
    pub x_norm_patchtokens: Tensor<B, 3>,
    pub x_prenorm: Tensor<B, 3>,
    pub masks: Option<Tensor<B, 3>>,
}


#[derive(Module, Debug)]
struct DinoVisionTransformer<B: Backend> {
    activation: Gelu,
    cls_token: Tensor<B, 3>,
    pos_embed: Tensor<B, 3>,
    mask_token: Tensor<B, 2>,
    interpolate: nn::interpolate::Interpolate2d,
    patch_embed: PatchEmbed<B>,
    layer_norm: nn::LayerNorm<B>,
    blocks: Vec<Block<B>>,
}

impl<B: Backend> DinoVisionTransformer<B> {
    pub fn new(
        device: &B::Device,
        config: DinoVisionTransformerConfig,
    ) -> Self {
        // TODO: initialize cls_token and pos_embed
        let cls_token: Tensor<B, 3> = Tensor::zeros(
            [1, 1, config.embedding_dimension],
            device,
        );

        let num_patches = 1;
        let num_tokens = 1;

        let pos_embed: Tensor<B, 3> = Tensor::zeros(
            [1, num_patches + num_tokens, config.embedding_dimension],
            device,
        );

        let mask_token: Tensor<B, 2> = Tensor::zeros(
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

        let layer_norm = nn::LayerNormConfig::new(config.embedding_dimension)
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
            layer_norm,
            blocks,
        }
    }

    #[allow(non_snake_case)]
    pub fn interpolate_pos_encoding(
        &self,
        x: Tensor<B, 3>,
        W: usize,
        H: usize,
    ) -> Tensor<B, 3> {
        let npatch = x.shape().dims[1];
        let N = self.pos_embed.shape().dims[1] - 1;

        if npatch == N && W == H {
            return self.pos_embed.clone();
        }

        let class_pos_embed = self.pos_embed.clone().slice([RangeFull, 0..1]);
        let patch_pos_embed = self.pos_embed.clone().slice([RangeFull, 1..]);
        let dim = x.shape().dims[2];
        let w0 = W / self.patch_size;
        let h0 = H / self.patch_size;
        let M = N.isqrt();

        assert!(
            N == M * M,
            "number of patches should be a square number",
        );

        let patch_pos_embed = self.interpolate.forward(
            patch_pos_embed.reshape([1, M, M, dim]).permute([0, 3, 1, 2]),
        ).permute([0, 2, 3, 1]).reshape([1, -1, dim]);

        return Tensor::cat(
            vec![
                class_pos_embed.unsqueeze_dim(0),
                patch_pos_embed,
            ],
            1,
        );
    }

    #[allow(non_snake_case)]
    pub fn prepare_tokens_with_masks(
        &self,
        x: Tensor<B, 4>,
        mask: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 3> {
        // TODO: H, W?
        let [B, C, W, H] = x.shape().dims();

        // let x = self.patch_embed.forward(x);

        let x = if let Some(mask) = mask {
            x.mask_where(mask, self.mask_token)
        } else {
            x
        };

        let x = Tensor::cat(
            &[self.cls_token.expand([x.shape().dims[0], -1, -1]), x],
            1
        );

        let x = x + self.interpolate_pos_encoding(x, W, H);

        x
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        masks: Option<Tensor<B, 3, Bool>>,
    ) -> DinoOutput<B> {
        let mut x = self.prepare_tokens_with_masks(x, masks);

        for block in &self.blocks {
            x = block.forward(x);
        }

        let x = self.layer_norm.forward(x);

        DinoOutput {
            x_norm_clstoken: Tensor::zeros([1, 1, 1], x.device()),
            x_norm_regtokens: Tensor::zeros([1, 1, 1], x.device()),
            x_norm_patchtokens: Tensor::zeros([1, 1, 1], x.device()),
            x_prenorm: Tensor::zeros([1, 1, 1], x.device()),
            masks: None,
        }
    }
}
