use std::sync::{Arc, Mutex};

use burn::prelude::*;
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage, RgbaImage};

use burn_dino::model::{
    dino::{DinoVisionTransformer, DinoVisionTransformerConfig},
    pca::PcaTransform,
};

pub mod platform;

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

async fn to_image<B: Backend>(
    image: Tensor<B, 4>,
    upsample_height: u32,
    upsample_width: u32,
) -> RgbaImage {
    let height = image.shape().dims[1];
    let width = image.shape().dims[2];

    let image = image.to_data_async().await.to_vec::<f32>().unwrap();
    let image =
        ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(width as u32, height as u32, image).unwrap();

    DynamicImage::ImageRgb32F(image)
        .resize_exact(
            upsample_width,
            upsample_height,
            image::imageops::FilterType::Triangle,
        )
        .to_rgba8()
}

// TODO: benchmark process_frame
pub async fn process_frame<B: Backend>(
    input: RgbImage,
    dino_config: DinoVisionTransformerConfig,
    dino_model: Arc<Mutex<DinoVisionTransformer<B>>>,
    pca_model: Arc<Mutex<PcaTransform<B>>>,
    device: B::Device,
) -> Vec<u8> {
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

    // pca min-max scaling
    for i in 0..3 {
        let slice = pca_features.clone().slice([0..n_samples, i..i + 1]);
        let slice_min = slice.clone().min();
        let slice_max = slice.clone().max();
        let scaled = slice
            .sub(slice_min.clone().unsqueeze())
            .div((slice_max - slice_min).unsqueeze());

        pca_features = pca_features.slice_assign([0..n_samples, i..i + 1], scaled);
    }

    let pca_features = pca_features.reshape([batch, spatial_size, spatial_size, 3]);

    let pca_features = to_image(
        pca_features,
        dino_config.image_size as u32,
        dino_config.image_size as u32,
    )
    .await;

    pca_features.into_raw()
}
