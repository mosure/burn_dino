use burn::{
    prelude::*,
    nn::Gelu,
};


#[derive(Config)]
struct DinoVisionTransformerConfig {
    image_size: usize,
    patch_size: usize,
    input_channels: usize,
    embedding_dimension: usize,
    depth: usize,
    num_heads: usize,
    mlp_ratio: f32,
    qkv_bias: bool,
    ffn_bias: bool,
    proj_bias: bool,
    attention_config: nn::attention::MultiHeadAttentionConfig,
}

impl Default for DinoVisionTransformerConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            input_channels: 3,
            embedding_dimension: 768,
            depth: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            qkv_bias: true,
            ffn_bias: true,
            proj_bias: true,
        }
    }
}

#[derive(Module, Debug)]
struct DinoVisionTransformer<B: Backend> {
    activation: Gelu,
    cls_token: Tensor<B, 3>,
    pos_embed: Tensor<B, 3>,
}

impl<B: Backend> DinoVisionTransformer<B> {
    pub fn new(
        device: &B::Device,
        config: DinoVisionTransformerConfig,
    ) -> Self {
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


        // let conv1 = ConvBlock::new([1, 8], [3, 3], device); // out: [Batch,8,26,26]
        // let conv2 = ConvBlock::new([8, 16], [3, 3], device); // out: [Batch,16,24x24]
        // let conv3 = ConvBlock::new([16, 24], [3, 3], device); // out: [Batch,24,22x22]
        // let hidden_size = 24 * 22 * 22;
        // let fc1 = nn::LinearConfig::new(hidden_size, 32)
        //     .with_bias(false)
        //     .init(device);
        // let fc2 = nn::LinearConfig::new(32, NUM_CLASSES)
        //     .with_bias(false)
        //     .init(device);

        // let dropout = nn::DropoutConfig::new(0.5).init();

        // Self {
        //     conv1,
        //     conv2,
        //     conv3,
        //     dropout,
        //     fc1,
        //     fc2,
        //     activation: nn::Gelu::new(),
        // }

        Self {
            activation: Gelu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        // let [batch_size, height, width] = input.dims();

        // let x = input.reshape([batch_size, 1, height, width]).detach();
        // let x = self.conv1.forward(x);
        // let x = self.conv2.forward(x);
        // let x = self.conv3.forward(x);

        // let [batch_size, channels, height, width] = x.dims();
        // let x = x.reshape([batch_size, channels * height * width]);

        // let x = self.dropout.forward(x);
        // let x = self.fc1.forward(x);
        // let x = self.activation.forward(x);

        // self.fc2.forward(x)
    }
}
