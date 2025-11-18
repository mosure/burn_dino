use burn::{
    prelude::*,
    serde::{Deserialize, Serialize},
};

#[derive(Clone, Copy, Debug, Module, Serialize, Deserialize)]
pub struct RopeConfig {
    pub base_frequency: f32,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            base_frequency: 100.0,
        }
    }
}

pub struct RotaryEmbedding;

impl RotaryEmbedding {
    pub fn apply<B: Backend>(
        tokens: Tensor<B, 4>,
        positions: &Tensor<B, 3>,
        config: RopeConfig,
    ) -> Tensor<B, 4> {
        let dims = tokens.shape().dims::<4>();
        let chunk = dims[3] / 2;
        assert!(
            chunk.is_multiple_of(2),
            "RoPE expects the token dimension to be divisible by 4"
        );

        let vert = tokens.clone().slice([
            0..dims[0] as i32,
            0..dims[1] as i32,
            0..dims[2] as i32,
            0..chunk as i32,
        ]);
        let horiz = tokens.slice([
            0..dims[0] as i32,
            0..dims[1] as i32,
            0..dims[2] as i32,
            chunk as i32..(chunk * 2) as i32,
        ]);

        let vert = Self::apply_axis(vert, positions, 0, config);
        let horiz = Self::apply_axis(horiz, positions, 1, config);
        Tensor::cat(vec![vert, horiz], 3)
    }

    fn apply_axis<B: Backend>(
        tokens: Tensor<B, 4>,
        positions: &Tensor<B, 3>,
        axis: usize,
        config: RopeConfig,
    ) -> Tensor<B, 4> {
        let dims = tokens.shape().dims::<4>();
        let batch = dims[0];
        let heads = dims[1];
        let token_count = dims[2];
        let feature_dim = dims[3];
        assert!(
            feature_dim.is_multiple_of(2),
            "RoPE axis expects an even feature dimension"
        );

        let device = tokens.device();
        let steps = feature_dim / 2;
        let inv_freq = Self::inv_frequencies(feature_dim, config.base_frequency);
        let inv_freq =
            Tensor::<B, 1>::from_floats(inv_freq.as_slice(), &device).reshape([1, steps as i32]);

        let coords = positions
            .clone()
            .slice([
                0..batch as i32,
                0..token_count as i32,
                axis as i32..axis as i32 + 1,
            ])
            .reshape([(batch * token_count) as i32, 1]);

        let angles = coords.matmul(inv_freq);
        let angles: Tensor<B, 3> = angles.reshape([batch as i32, token_count as i32, steps as i32]);
        let angles: Tensor<B, 3> = Tensor::cat(vec![angles.clone(), angles], 2);
        let angles: Tensor<B, 4> = angles.unsqueeze_dim(1).repeat_dim(1, heads);
        let cos = angles.clone().cos();
        let sin = angles.sin();
        let rotated = Self::rotate(tokens.clone());
        tokens * cos + rotated * sin
    }

    fn rotate<B: Backend>(tokens: Tensor<B, 4>) -> Tensor<B, 4> {
        let dims = tokens.shape().dims::<4>();
        let half = dims[3] / 2;
        let first = tokens.clone().slice([
            0..dims[0] as i32,
            0..dims[1] as i32,
            0..dims[2] as i32,
            0..half as i32,
        ]);
        let second = tokens.slice([
            0..dims[0] as i32,
            0..dims[1] as i32,
            0..dims[2] as i32,
            half as i32..dims[3] as i32,
        ]);
        let neg_second = second.mul_scalar(-1.0);
        Tensor::cat(vec![neg_second, first], 3)
    }

    fn inv_frequencies(feature_dim: usize, base: f32) -> Vec<f32> {
        let mut inv = Vec::with_capacity(feature_dim / 2);
        let denom = feature_dim as f32;
        for idx in (0..feature_dim).step_by(2) {
            let exponent = idx as f32 / denom;
            let freq = base.powf(exponent);
            inv.push(1.0 / freq);
        }
        inv
    }
}
