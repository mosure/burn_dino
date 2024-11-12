// def make_2tuple(x):
//     if isinstance(x, tuple):
//         assert len(x) == 2
//         return x

//     assert isinstance(x, int)
//     return (x, x)


// class PatchEmbed(nn.Module):
//     """
//     2D image to patch embedding: (B,C,H,W) -> (B,N,D)

//     Args:
//         img_size: Image size.
//         patch_size: Patch token size.
//         in_chans: Number of input image channels.
//         embed_dim: Number of linear projection output channels.
//         norm_layer: Normalization layer.
//     """

//     def __init__(
//         self,
//         img_size: Union[int, Tuple[int, int]] = 224,
//         patch_size: Union[int, Tuple[int, int]] = 16,
//         in_chans: int = 3,
//         embed_dim: int = 768,
//         norm_layer: Optional[Callable] = None,
//         flatten_embedding: bool = True,
//     ) -> None:
//         super().__init__()

//         image_HW = make_2tuple(img_size)
//         patch_HW = make_2tuple(patch_size)
//         patch_grid_size = (
//             image_HW[0] // patch_HW[0],
//             image_HW[1] // patch_HW[1],
//         )

//         self.img_size = image_HW
//         self.patch_size = patch_HW
//         self.patches_resolution = patch_grid_size
//         self.num_patches = patch_grid_size[0] * patch_grid_size[1]

//         self.in_chans = in_chans
//         self.embed_dim = embed_dim

//         self.flatten_embedding = flatten_embedding

//         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
//         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

//     def forward(self, x: Tensor) -> Tensor:
//         _, _, H, W = x.shape
//         patch_H, patch_W = self.patch_size

//         assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
//         assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

//         x = self.proj(x)  # B C H W
//         H, W = x.size(2), x.size(3)
//         x = x.flatten(2).transpose(1, 2)  # B HW C
//         x = self.norm(x)
//         if not self.flatten_embedding:
//             x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
//         return x

//     def flops(self) -> float:
//         Ho, Wo = self.patches_resolution
//         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
//         if self.norm is not None:
//             flops += Ho * Wo * self.embed_dim
//         return flops


use burn::{
    prelude::*,
    tensor::activation::{
        quiet_softmax,
        softmax,
    },
};


#[derive(Config)]
pub struct PatchEmbedConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub qkv_bias: bool,
    pub proj_bias: bool,
    pub attn_drop: f64,
    pub proj_drop: f64,
    pub quiet_softmax: bool,
}

impl Default for PatchEmbedConfig {
    fn default() -> Self {
        Self {
            dim: 0,
            num_heads: 8,
            qkv_bias: false,
            proj_bias: true,
            attn_drop: 0.0,
            proj_drop: 0.0,
            quiet_softmax: false,
        }
    }
}

impl PatchEmbedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbed<B> {
        PatchEmbed::new(device, self.clone())
    }
}


#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    pub qkv: nn::Linear<B>,
    pub attn_drop: nn::Dropout,
    pub proj: nn::Linear<B>,
    pub proj_drop: nn::Dropout,
    pub num_heads: usize,
    pub scale: f32,
    pub quiet_softmax: bool,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(
        device: &B::Device,
        config: PatchEmbedConfig,
    ) -> Self {
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

    #[allow(non_snake_case)]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [B, N, C] = x.shape().dims();

        let qkv = self.qkv.forward(x)
            .reshape([B, N, 3, self.num_heads, C / self.num_heads])
            .permute([2, 0, 3, 1, 4]);

        let q: Tensor<B, 4> = qkv.clone().slice([0..1]).squeeze(0) * self.scale;
        let k = qkv.clone().slice([1..2]).squeeze(0);
        let v = qkv.slice([2..3]).squeeze(0);

        let attn = q.matmul(k.swap_dims(2, 3));

        let attn = if self.quiet_softmax {
            quiet_softmax(attn, 3)
        } else {
            softmax(attn, 3)
        };

        let attn = self.attn_drop.forward(attn);

        let x = attn.matmul(v)
            .swap_dims(1, 2)
            .reshape([B, N, C]);

        let x = self.proj.forward(x);
        let x = self.proj_drop.forward(x);

        x
    }
}

