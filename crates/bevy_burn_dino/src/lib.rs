use std::sync::{Arc, Mutex};

use burn::nn::interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode};
use burn::prelude::*;
use image::{DynamicImage, RgbImage};

use burn_dino::model::{
    dino::{DinoVisionTransformer, DinoVisionTransformerConfig},
    pca::PcaTransform,
};

pub mod platform;

/// Heuristic per-channel ranges captured from earlier dataset sweeps.
/// This avoids per-frame GPU read-backs on platforms (like WASM) that cannot block.
const PCA_STATIC_MIN_MAX: [(f32, f32); 3] = [
    (-22.9175, 25.0372),
    (-22.2775, 20.7716),
    (-22.1609, 17.3058),
];
const PCA_NORMALIZATION_EPS: f32 = 1e-6;

fn normalize<B: Backend>(input: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 4> {
    let mean: Tensor<B, 1> = Tensor::from_floats([0.485, 0.456, 0.406], device);
    let std: Tensor<B, 1> = Tensor::from_floats([0.229, 0.224, 0.225], device);

    input
        .permute([0, 2, 3, 1])
        .sub(mean.unsqueeze())
        .div(std.unsqueeze())
        .permute([0, 3, 1, 2])
}

fn preprocess_image<B: Backend>(
    image: RgbImage,
    config: &DinoVisionTransformerConfig,
    device: &B::Device,
) -> Tensor<B, 4> {
    let img = DynamicImage::ImageRgb8(image)
        .resize_exact(
            config.image_size as u32,
            config.image_size as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb32f();

    let samples = img.as_flat_samples();
    let floats: &[f32] = samples.as_slice();

    let input: Tensor<B, 1> = Tensor::from_floats(floats, device);

    let input = input
        .reshape([
            1,
            config.image_size,
            config.image_size,
            config.input_channels,
        ])
        .permute([0, 3, 1, 2]);

    normalize(input, device)
}

// TODO: benchmark process_frame
pub async fn process_frame<B: Backend>(
    input: RgbImage,
    dino_config: DinoVisionTransformerConfig,
    dino_model: Arc<Mutex<DinoVisionTransformer<B>>>,
    pca_model: Arc<Mutex<PcaTransform<B>>>,
    device: B::Device,
) -> Tensor<B, 3> {
    let input_tensor: Tensor<B, 4> = preprocess_image(input, &dino_config, &device);

    let dino_features = {
        let model = dino_model.lock().unwrap();
        model.forward(input_tensor.clone(), None).x_norm_patchtokens
    };

    let batch = dino_features.shape().dims[0];
    let elements = dino_features.shape().dims[1];
    let embedding_dim = dino_features.shape().dims[2];
    let n_samples = batch * elements;
    let spatial_size = elements.isqrt();

    let x = dino_features.reshape([n_samples, embedding_dim]);

    let mut pca_features = {
        let pca_transform = pca_model.lock().unwrap();
        pca_transform.forward(x.clone())
    };

    // Dynamic min-max scaling kept for future reference once async reductions are viable.
    /*
    for i in 0..3 {
        let slice = pca_features.clone().slice([0..n_samples, i..i + 1]);
        let slice_min = slice.clone().min();
        let slice_max = slice.clone().max();
        let scaled = slice
            .sub(slice_min.clone().unsqueeze())
            .div((slice_max - slice_min).unsqueeze());

        pca_features = pca_features.slice_assign([0..n_samples, i..i + 1], scaled);
    }
    */

    for (channel_idx, (min_val, max_val)) in PCA_STATIC_MIN_MAX.iter().enumerate() {
        let slice = pca_features
            .clone()
            .slice([0..n_samples, channel_idx..channel_idx + 1]);

        let denom = (*max_val - *min_val).max(PCA_NORMALIZATION_EPS);
        let scaled = slice.sub_scalar(*min_val).div_scalar(denom).clamp(0.0, 1.0);

        pca_features =
            pca_features.slice_assign([0..n_samples, channel_idx..channel_idx + 1], scaled);
    }

    let mut pca_features = pca_features.reshape([batch, spatial_size, spatial_size, 3]);
    pca_features = pca_features.permute([0, 3, 1, 2]);

    let upsample: Interpolate2d = Interpolate2dConfig {
        output_size: Some([dino_config.image_size, dino_config.image_size]),
        scale_factor: None,
        mode: InterpolateMode::Cubic,
    }
    .init();
    let pca_features = upsample.forward(pca_features).permute([0, 2, 3, 1]);

    let rgb = pca_features.squeeze_dim(0);
    let alpha = Tensor::<B, 3>::ones([dino_config.image_size, dino_config.image_size, 1], &device);

    Tensor::<B, 3>::cat(vec![rgb, alpha], 2)
}
