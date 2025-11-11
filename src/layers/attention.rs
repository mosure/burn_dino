use burn::{
    prelude::*,
    tensor::activation::{quiet_softmax, softmax},
};

#[derive(Config, Debug)]
pub struct AttentionConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub qkv_bias: bool,
    pub proj_bias: bool,
    pub attn_drop: f64,
    pub proj_drop: f64,
    pub quiet_softmax: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            dim: 768,
            num_heads: 12,
            qkv_bias: true,
            proj_bias: true,
            attn_drop: 0.0,
            proj_drop: 0.0,
            quiet_softmax: false,
        }
    }
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        Attention::new(device, self.clone())
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub qkv: nn::Linear<B>,
    pub attn_drop: nn::Dropout,
    pub proj: nn::Linear<B>,
    pub proj_drop: nn::Dropout,
    pub num_heads: usize,
    pub scale: f32,
    pub quiet_softmax: bool,
}

impl<B: Backend> Attention<B> {
    pub fn new(device: &B::Device, config: AttentionConfig) -> Self {
        let head_dim = config.dim / config.num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let qkv = nn::LinearConfig::new(config.dim, config.dim * 3)
            .with_bias(config.qkv_bias)
            .init::<B>(device);

        let attn_drop = nn::DropoutConfig::new(config.attn_drop).init();

        let proj = nn::LinearConfig::new(config.dim, config.dim)
            .with_bias(config.proj_bias)
            .init::<B>(device);

        let proj_drop = nn::DropoutConfig::new(config.proj_drop).init();

        Self {
            qkv,
            attn_drop,
            proj,
            proj_drop,
            num_heads: config.num_heads,
            scale,
            quiet_softmax: config.quiet_softmax,
        }
    }

    #[allow(non_snake_case, clippy::single_range_in_vec_init)]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [B, N, C] = x.shape().dims();

        let qkv = self
            .qkv
            .forward(x)
            .reshape([B, N, 3, self.num_heads, C / self.num_heads])
            .permute([2, 0, 3, 1, 4]);

        let q: Tensor<B, 4> = qkv.clone().slice([0..1]).squeeze_dim(0) * self.scale;
        let k = qkv.clone().slice([1..2]).squeeze_dim(0);
        let v = qkv.slice([2..3]).squeeze_dim(0);

        let attn = q.matmul(k.swap_dims(2, 3));

        let attn = if self.quiet_softmax {
            quiet_softmax(attn, 3)
        } else {
            softmax(attn, 3)
        };

        let attn = self.attn_drop.forward(attn);

        let x = attn.matmul(v).swap_dims(1, 2).reshape([B, N, C]);

        let x = self.proj.forward(x);
        self.proj_drop.forward(x)
    }
}
