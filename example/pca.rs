use std::path::Path;

use burn::{
    backend::wgpu::Wgpu,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};
use image::{load_from_memory_with_format, DynamicImage, ImageFormat, RgbImage};

use burn_dino::model::{
    dino::{DinoVisionTransformer, DinoVisionTransformerConfig},
    pca::{PcaTransform, PcaTransformConfig},
};

static DINO_STATE_ENCODED: &[u8] = include_bytes!("../assets/models/dinov2.mpk");
static PCA_STATE_ENCODED: &[u8] = include_bytes!("../assets/models/face_pca.mpk");

static INPUT_IMAGE_0: &[u8] = include_bytes!("../assets/images/dino_0.png");
static INPUT_IMAGE_1: &[u8] = include_bytes!("../assets/images/dino_1.png");
static INPUT_IMAGE_2: &[u8] = include_bytes!("../assets/images/dino_2.png");
static INPUT_IMAGE_3: &[u8] = include_bytes!("../assets/images/dino_3.png");

fn load_model<B: Backend>(
    config: &DinoVisionTransformerConfig,
    device: &B::Device,
) -> DinoVisionTransformer<B> {
    let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
        .load(DINO_STATE_ENCODED.to_vec(), &Default::default())
        .expect("failed to decode state");

    let model = config.init(device);
    model.load_record(record)
}

fn load_pca_model<B: Backend>(config: &PcaTransformConfig, device: &B::Device) -> PcaTransform<B> {
    let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
        .load(PCA_STATE_ENCODED.to_vec(), &Default::default())
        .expect("failed to decode state");

    let model = config.init(device);
    model.load_record(record)
}

fn normalize<B: Backend>(input: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 4> {
    let mean: Tensor<B, 1> = Tensor::from_floats([0.485, 0.456, 0.406], device);
    let std: Tensor<B, 1> = Tensor::from_floats([0.229, 0.224, 0.225], device);

    input
        .permute([0, 2, 3, 1])
        .sub(mean.unsqueeze())
        .div(std.unsqueeze())
        .permute([0, 3, 1, 2])
}

fn load_image<B: Backend>(
    bytes: &[u8],
    config: &DinoVisionTransformerConfig,
    device: &B::Device,
) -> Tensor<B, 4> {
    let img = load_from_memory_with_format(bytes, ImageFormat::Png)
        .unwrap()
        .resize_exact(
            config.image_size as u32,
            config.image_size as u32,
            image::imageops::FilterType::Lanczos3,
        );

    let img = match img {
        DynamicImage::ImageRgb8(img) => img,
        _ => img.to_rgb8(),
    };

    let img_data: Vec<f32> = img
        .pixels()
        .flat_map(|p| p.0.iter().map(|&c| c as f32 / 255.0))
        .collect();

    let input: Tensor<B, 1> = Tensor::from_floats(img_data.as_slice(), device);

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

fn write_images<B: Backend>(
    images: Tensor<B, 4>,
    output_directory: &Path,
    upsample_height: u32,
    upsample_width: u32,
) {
    let batch = images.shape().dims[0];
    let height = images.shape().dims[1];
    let width = images.shape().dims[2];
    let channels = images.shape().dims[3];

    let image_size = height * width * channels;

    let images = images.clamp(0.0, 1.0).mul_scalar(255.0);
    let images = images.to_data().to_vec::<f32>().unwrap();

    for i in 0..batch {
        let offset = i * image_size;

        let image_slice = &images[offset..offset + image_size];
        let image_slice_u8: Vec<u8> = image_slice.iter().map(|&v| v as u8).collect();

        let img = RgbImage::from_raw(width as u32, height as u32, image_slice_u8).unwrap();

        let img = DynamicImage::ImageRgb8(img).resize_exact(
            upsample_width,
            upsample_height,
            image::imageops::FilterType::Lanczos3,
        );

        std::fs::create_dir_all(output_directory).unwrap();

        let output_path = output_directory.join(format!("{}.png", i));
        img.save(output_path).unwrap();
    }
}

fn main() {
    let device = Default::default();
    let config = DinoVisionTransformerConfig {
        ..DinoVisionTransformerConfig::vits(None, None)
    };
    let dino = load_model(&config, &device);

    let input_pngs = vec![INPUT_IMAGE_0, INPUT_IMAGE_1, INPUT_IMAGE_2, INPUT_IMAGE_3];

    let mut input_tensors = Vec::new();
    for input in input_pngs {
        let input_tensor: Tensor<Wgpu, 4> = load_image(input, &config, &device);
        input_tensors.push(input_tensor);
    }

    let batched_input = Tensor::cat(input_tensors, 0);
    let dino_features = dino.forward(batched_input.clone(), None).x_norm_patchtokens;

    let batch = dino_features.shape().dims[0];
    let elements = dino_features.shape().dims[1];
    let embedding_dim = dino_features.shape().dims[2];
    let n_samples = batch * elements;
    let spatial_size = elements.isqrt();

    let x = dino_features.reshape([n_samples, embedding_dim]);

    let pca_config = PcaTransformConfig::new(embedding_dim, 3);
    let pca_transform = load_pca_model(&pca_config, &device);
    let mut pca_features = pca_transform.forward(x.clone());

    // pca min-max scaling
    for i in 0..3 {
        let slice = pca_features.clone().slice([0..n_samples, i..i + 1]);
        let slice_min = slice.clone().min().into_scalar();
        let slice_max = slice.clone().max().into_scalar();
        let scaled = slice
            .sub_scalar(slice_min)
            .div_scalar(slice_max - slice_min);

        pca_features = pca_features.slice_assign([0..n_samples, i..i + 1], scaled);
    }

    let pca_features = pca_features.reshape([batch, spatial_size, spatial_size, 3]);
    write_images(
        pca_features,
        Path::new("output/pca"),
        config.image_size as u32,
        config.image_size as u32,
    );
}
