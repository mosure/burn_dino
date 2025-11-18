use burn::{
    module::Param,
    nn::{Gelu, Initializer},
    prelude::*,
};

use crate::layers::{
    attention::AttentionConfig,
    block::{Block, BlockConfig},
    layer_norm::{LayerNorm, LayerNormConfig},
    layer_scale::LayerScaleConfig,
    patch_embed::{PatchEmbed, PatchEmbedConfig},
    rope::RopeConfig,
};

#[derive(Config, Debug)]
pub struct DinoVisionTransformerConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub input_channels: usize,
    pub embedding_dimension: usize,
    pub depth: usize,
    pub block_config: BlockConfig,
    pub positional_encoding_interpolate: nn::interpolate::Interpolate2dConfig,
    pub num_patches: usize,
    #[config(default = "0")]
    pub register_token_count: usize,

    #[config(default = "true")]
    pub use_register_tokens: bool,

    #[config(default = "true")]
    pub normalize_intermediate_tokens: bool,

    #[config(default = "Initializer::Normal{mean:0.02, std:1.0}")]
    pub initializer: Initializer,
    #[config(default = "None")]
    pub alt_block_start: Option<usize>,
    #[config(default = "None")]
    pub rope_block_start: Option<usize>,
    #[config(default = "100.0")]
    pub rope_frequency: f32,
    #[config(default = "None")]
    pub qk_norm_block_start: Option<usize>,
    #[config(default = "false")]
    pub cat_token: bool,
    #[config(default = "false")]
    pub use_camera_tokens: bool,
    #[config(default = "true")]
    pub use_mask_token: bool,
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
                    quiet_softmax: false,
                    ..Default::default()
                },
                layer_scale: LayerScaleConfig { dim }.into(),
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
                }
                .into(),
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
                }
                .into(),
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
                }
                .into(),
                ..Default::default()
            },
            ..Self::from_size(image_size, patch_size)
        }
    }

    pub fn with_register_tokens(mut self, count: usize) -> Self {
        self.register_token_count = count;
        self.use_register_tokens = count > 0;
        self
    }

    pub fn without_register_tokens(mut self) -> Self {
        self.register_token_count = 0;
        self.use_register_tokens = false;
        self
    }
}

#[derive(Debug, Clone)]
pub struct DinoOutput<B: Backend> {
    pub x_norm_clstoken: Tensor<B, 2>,
    pub x_norm_patchtokens: Tensor<B, 3>,
    pub x_norm_regtokens: Option<Tensor<B, 3>>,
    pub x_prenorm: Tensor<B, 3>,
    pub masks: Option<Tensor<B, 3, Bool>>,
}

pub struct DinoIntermediate<B: Backend> {
    pub patches: Tensor<B, 3>,
    pub camera: Option<Tensor<B, 2>>,
}

struct BlockSnapshot<B: Backend> {
    tensor: Tensor<B, 3>,
    camera: Option<Tensor<B, 2>>,
}

#[derive(Module, Debug)]
pub struct DinoVisionTransformer<B: Backend> {
    activation: Gelu,
    cls_token: Param<Tensor<B, 3>>,
    pub pos_embed: Param<Tensor<B, 3>>,
    mask_token: Option<Param<Tensor<B, 2>>>,
    register_tokens: Option<Param<Tensor<B, 3>>>,
    camera_token: Option<Param<Tensor<B, 3>>>,
    interpolate: nn::interpolate::Interpolate2d,
    patch_embed: PatchEmbed<B>,
    norm: LayerNorm<B>,
    blocks: Vec<Block<B>>,
    patch_size: usize,
    register_token_count: usize,
    normalize_intermediate_tokens: bool,
    embedding_dim: usize,
    patch_token_start: usize,
    alt_block_start: Option<usize>,
    rope_block_start: Option<usize>,
    rope_frequency: f32,
    cat_tokens: bool,
    use_camera_tokens: bool,
}

impl<B: Backend> DinoVisionTransformer<B> {
    pub fn new(device: &B::Device, config: DinoVisionTransformerConfig) -> Self {
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

        let cls_token = config
            .initializer
            .init([1, 1, config.embedding_dimension], device);

        let num_tokens = 1 + if config.use_register_tokens {
            config.register_token_count
        } else {
            0
        };
        let pos_embed = config.initializer.init(
            [
                1,
                config.num_patches + num_tokens,
                config.embedding_dimension,
            ],
            device,
        );

        let mask_token = if config.use_mask_token {
            Some(
                config
                    .initializer
                    .init([1, config.embedding_dimension], device),
            )
        } else {
            None
        };

        let register_tokens = if config.use_register_tokens && config.register_token_count > 0 {
            Some(
                Initializer::Normal {
                    mean: 0.0,
                    std: 1e-6,
                }
                .init(
                    [1, config.register_token_count, config.embedding_dimension],
                    device,
                ),
            )
        } else {
            None
        };

        let camera_token = if config.use_camera_tokens {
            Some(
                Initializer::Normal {
                    mean: 0.0,
                    std: 1e-6,
                }
                .init([1, 2, config.embedding_dimension], device),
            )
        } else {
            None
        };

        let interpolate = config.positional_encoding_interpolate.init();

        let patch_embed = PatchEmbedConfig::new(
            config.image_size,
            config.patch_size,
            config.input_channels,
            config.embedding_dimension,
        )
        .init(device);

        let norm: LayerNorm<B> = LayerNormConfig::new(config.embedding_dimension).init(device);

        let mut blocks = Vec::with_capacity(config.depth);
        for index in 0..config.depth {
            let mut block_config = config.block_config.clone();
            if let Some(start) = config.qk_norm_block_start
                && index >= start
            {
                block_config.attn.qk_norm = true;
            }
            if let Some(start) = config.rope_block_start
                && index >= start {
                block_config.attn.rope = Some(RopeConfig {
                    base_frequency: config.rope_frequency,
                });
            }
            let block = block_config.init(device);
            blocks.push(block);
        }

        let register_token_count = if config.use_register_tokens {
            config.register_token_count
        } else {
            0
        };

        let patch_token_start = 1 + register_token_count;

        Self {
            activation: Gelu::new(),
            cls_token,
            pos_embed,
            mask_token,
            register_tokens,
            camera_token,
            interpolate,
            patch_embed,
            norm,
            blocks,
            patch_size: config.patch_size,
            register_token_count,
            normalize_intermediate_tokens: config.normalize_intermediate_tokens,
            embedding_dim: config.embedding_dimension,
            patch_token_start,
            alt_block_start: config.alt_block_start,
            rope_block_start: config.rope_block_start,
            rope_frequency: config.rope_frequency,
            cat_tokens: config.cat_token,
            use_camera_tokens: config.use_camera_tokens,
        }
    }

    fn finalize_output(
        &self,
        tokens: Tensor<B, 3>,
        masks: Option<Tensor<B, 3, Bool>>,
    ) -> DinoOutput<B> {
        let x_norm = self.norm.forward(tokens.clone());

        let b_dim = tokens.shape().dims[0];
        let n_dim = tokens.shape().dims[1];
        let reg_count = self.register_token_count;
        let x_norm_clstoken = x_norm.clone().slice([0..b_dim, 0..1]).squeeze_dim(1);
        let x_norm_regtokens = if reg_count > 0 {
            Some(x_norm.clone().slice([0..b_dim, 1..(1 + reg_count)]))
        } else {
            None
        };
        let patch_start = self.patch_token_start;
        let x_norm_patchtokens = x_norm.clone().slice([0..b_dim, patch_start..n_dim]);

        DinoOutput {
            x_norm_clstoken,
            x_norm_patchtokens,
            x_norm_regtokens,
            x_prenorm: tokens,
            masks,
        }
    }

    #[allow(non_snake_case)]
    pub fn interpolate_pos_encoding(&self, x: Tensor<B, 3>, W: usize, H: usize) -> Tensor<B, 3> {
        let npatch = x.shape().dims[1] - 1;
        let register_offset = self.register_token_count;
        let tokens_start = 1 + register_offset;
        let N = self.pos_embed.shape().dims[1] - tokens_start;

        let b_dim = self.pos_embed.shape().dims[0];
        let n_dim = self.pos_embed.shape().dims[1];
        // let c_dim: usize = self.pos_embed.shape().dims[2];

        let class_pos_embed: Tensor<B, 2> = self
            .pos_embed
            .val()
            .clone()
            .slice([0..b_dim, 0..1])
            .squeeze_dim(1);
        let patch_pos_embed = self
            .pos_embed
            .val()
            .clone()
            .slice([0..b_dim, tokens_start..n_dim]);

        if npatch == N && W == H {
            return Tensor::cat(vec![class_pos_embed.unsqueeze_dim(0), patch_pos_embed], 1);
        }

        let dim = x.shape().dims[2];
        let M = N.isqrt();

        assert!(N == M * M, "number of patches should be a square number",);

        let patch_pos_embed = self
            .interpolate
            .forward(
                patch_pos_embed
                    .reshape([1, M, M, dim])
                    .permute([0, 3, 1, 2]),
            )
            .permute([0, 2, 3, 1])
            .reshape([1_i32, -1, dim as i32]);

        Tensor::cat(vec![class_pos_embed.unsqueeze_dim(0), patch_pos_embed], 1)
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
            if let Some(mask_token) = &self.mask_token {
                x.mask_where(mask, mask_token.val().unsqueeze_dim(0))
            } else {
                x
            }
        } else {
            x
        };

        let x = Tensor::cat(
            vec![
                self.cls_token
                    .val()
                    .expand([x.shape().dims[0] as i64, -1, -1]),
                x,
            ],
            1,
        );

        let residual = self.interpolate_pos_encoding(x.clone(), W, H);
        let x = x + residual;

        if let Some(register_tokens) = &self.register_tokens {
            let cls = x.clone().slice([0..x.shape().dims[0], 0..1]);
            let patches = x
                .clone()
                .slice([0..x.shape().dims[0], 1..x.shape().dims[1]]);
            let registers = register_tokens
                .val()
                .expand([x.shape().dims[0] as i64, -1, -1]);
            Tensor::cat(vec![cls, registers, patches], 1)
        } else {
            x
        }
    }

    #[allow(non_snake_case)]
    pub fn forward_with_intermediate_tokens(
        &self,
        x: Tensor<B, 4>,
        layers: &[usize],
    ) -> (DinoOutput<B>, Vec<Tensor<B, 3>>) {
        let (output, hooks, _) = self.forward_with_intermediate_tokens_ext(x, layers, &[], None);
        let tensors = hooks.into_iter().map(|hook| hook.patches).collect();
        (output, tensors)
    }

    pub fn forward_with_intermediate_tokens_ext(
        &self,
        x: Tensor<B, 4>,
        layers: &[usize],
        export_layers: &[usize],
        camera_token: Option<Tensor<B, 2>>,
    ) -> (DinoOutput<B>, Vec<DinoIntermediate<B>>, Vec<Tensor<B, 3>>) {
        let dims = x.shape().dims::<4>();
        let batch = dims[0];
        let height = dims[2];
        let width = dims[3];

        let mut tokens = self.prepare_tokens_with_masks(x, None);
        let device = tokens.device();

        let rope_positions = self.prepare_rope_positions(batch, width, height, &device);

        let mut snapshots = Vec::with_capacity(layers.len());
        let mut aux_snapshots = Vec::with_capacity(export_layers.len());
        let mut local_snapshot = tokens.clone();

        for (index, block) in self.blocks.iter().enumerate() {
            if self
                .alt_block_start
                .map(|start| index == start)
                .unwrap_or(false)
            {
                tokens = self.apply_camera_token(tokens, camera_token.clone());
            }
            let rope_active = self
                .rope_block_start
                .map(|start| index >= start)
                .unwrap_or(false);
            let local_pos = if rope_active {
                rope_positions.as_ref().map(|(local, _)| local)
            } else {
                None
            };
            let global_pos = if rope_active {
                rope_positions.as_ref().map(|(_, global)| global)
            } else {
                None
            };

            let use_alt = self
                .alt_block_start
                .map(|start| index >= start)
                .unwrap_or(false);

            if use_alt && index % 2 == 1 {
                tokens = block.forward(tokens, global_pos, None);
            } else {
                tokens = block.forward(tokens, local_pos, None);
                local_snapshot = tokens.clone();
            }

            if export_layers.contains(&index) {
                aux_snapshots.push(tokens.clone());
            }

            if layers.contains(&index) {
                snapshots.push(self.capture_snapshot(tokens.clone(), local_snapshot.clone()));
            }
        }

        let intermediates = snapshots
            .into_iter()
            .map(|snapshot| self.finalize_snapshot(snapshot))
            .collect();

        let aux = aux_snapshots
            .into_iter()
            .map(|tensor| self.normalize_aux_snapshot(tensor))
            .collect();

        let output = self.finalize_output(tokens, None);
        (output, intermediates, aux)
    }

    pub fn forward(&self, x: Tensor<B, 4>, masks: Option<Tensor<B, 3, Bool>>) -> DinoOutput<B> {
        let mut tokens = self.prepare_tokens_with_masks(x, None);

        for block in &self.blocks {
            tokens = block.forward(tokens, None, None);
        }

        self.finalize_output(tokens, masks)
    }

    fn apply_camera_token(
        &self,
        tokens: Tensor<B, 3>,
        provided: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        if self.alt_block_start.is_none() {
            return tokens;
        }

        let batch = tokens.shape().dims[0];
        let embed_dim = tokens.shape().dims[2];
        let replacement = if let Some(token) = provided {
            token
        } else if let Some(param) = &self.camera_token {
            param
                .val()
                .clone()
                .slice([0..1, 0..1, 0..embed_dim as i32])
                .reshape([1, embed_dim as i32])
                .repeat_dim(0, batch)
        } else {
            return tokens;
        };

        let head = replacement.reshape([batch as i32, 1, embed_dim as i32]);
        let tail = tokens.clone().slice([
            0..batch as i32,
            1..tokens.shape().dims[1] as i32,
            0..embed_dim as i32,
        ]);
        Tensor::cat(vec![head, tail], 1)
    }

    fn prepare_rope_positions(
        &self,
        batch: usize,
        width: usize,
        height: usize,
        device: &B::Device,
    ) -> Option<(Tensor<B, 3>, Tensor<B, 3>)> {
        self.rope_block_start?;

        let patches_w = width / self.patch_size;
        let patches_h = height / self.patch_size;
        let patch_tokens = patches_w * patches_h;
        let total_tokens = self.patch_token_start + patch_tokens;

        let mut local = Vec::with_capacity(total_tokens * 2);
        for _ in 0..self.patch_token_start {
            local.extend_from_slice(&[0.0, 0.0]);
        }
        for y in 0..patches_h {
            for x in 0..patches_w {
                local.push((y + 1) as f32);
                local.push((x + 1) as f32);
            }
        }

        let mut global = Vec::with_capacity(total_tokens * 2);
        for _ in 0..self.patch_token_start {
            global.extend_from_slice(&[0.0, 0.0]);
        }
        for _ in 0..patch_tokens {
            global.extend_from_slice(&[1.0, 1.0]);
        }

        let mut local_buf = Vec::with_capacity(batch * total_tokens * 2);
        let mut global_buf = Vec::with_capacity(batch * total_tokens * 2);
        for _ in 0..batch {
            local_buf.extend_from_slice(&local);
            global_buf.extend_from_slice(&global);
        }

        let local_tensor = Tensor::<B, 1>::from_floats(local_buf.as_slice(), device).reshape([
            batch as i32,
            total_tokens as i32,
            2,
        ]);
        let global_tensor = Tensor::<B, 1>::from_floats(global_buf.as_slice(), device).reshape([
            batch as i32,
            total_tokens as i32,
            2,
        ]);
        Some((local_tensor, global_tensor))
    }

    fn capture_snapshot(&self, tokens: Tensor<B, 3>, local: Tensor<B, 3>) -> BlockSnapshot<B> {
        let combined = if self.cat_tokens {
            Tensor::cat(vec![local, tokens.clone()], 2)
        } else {
            tokens.clone()
        };
        let camera = if self.use_camera_tokens {
            Some(
                combined
                    .clone()
                    .slice([
                        0..combined.shape().dims[0] as i32,
                        0..1,
                        0..combined.shape().dims[2] as i32,
                    ])
                    .squeeze_dim(1),
            )
        } else {
            None
        };
        BlockSnapshot {
            tensor: combined,
            camera,
        }
    }

    fn finalize_snapshot(&self, snapshot: BlockSnapshot<B>) -> DinoIntermediate<B> {
        let normalized = if self.cat_tokens {
            self.normalize_cat_tokens(snapshot.tensor)
        } else if self.normalize_intermediate_tokens {
            self.norm.forward(snapshot.tensor)
        } else {
            snapshot.tensor
        };
        let dims = normalized.shape().dims::<3>();
        let patches = normalized.slice([
            0..dims[0] as i32,
            self.patch_token_start as i32..dims[1] as i32,
            0..dims[2] as i32,
        ]);
        DinoIntermediate {
            patches,
            camera: snapshot.camera,
        }
    }

    fn normalize_cat_tokens(&self, tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = tensor.shape().dims::<3>();
        let total = dims[2];
        if total == self.embedding_dim {
            if self.normalize_intermediate_tokens {
                self.norm.forward(tensor)
            } else {
                tensor
            }
        } else if total == self.embedding_dim * 2 {
            let local = tensor.clone().slice([
                0..dims[0] as i32,
                0..dims[1] as i32,
                0..self.embedding_dim as i32,
            ]);
            let global = tensor.slice([
                0..dims[0] as i32,
                0..dims[1] as i32,
                self.embedding_dim as i32..total as i32,
            ]);
            let norm_global = if self.normalize_intermediate_tokens {
                self.norm.forward(global)
            } else {
                global
            };
            Tensor::cat(vec![local, norm_global], 2)
        } else if self.normalize_intermediate_tokens {
            self.norm.forward(tensor)
        } else {
            tensor
        }
    }

    fn normalize_aux_snapshot(&self, tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        let tensor = if self.normalize_intermediate_tokens {
            self.norm.forward(tensor)
        } else {
            tensor
        };
        let dims = tensor.shape().dims::<3>();
        tensor.slice([
            0..dims[0] as i32,
            self.patch_token_start as i32..dims[1] as i32,
            0..dims[2] as i32,
        ])
    }
}
