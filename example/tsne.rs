use burn::{
    prelude::*,
    backend::wgpu::Wgpu,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};
use image::{
    load_from_memory_with_format,
    DynamicImage,
    GenericImage,
    ImageFormat,
    Luma,
    Pixel,
    RgbImage,
};
use bhtsne::tSNE;
use ndarray::{Array2, Array4, ArrayBase, ArrayView2, Ix4, s, ViewRepr};

use burn_dino::model::dino::{
    DinoVisionTransformer,
    DinoVisionTransformerConfig,
};


static STATE_ENCODED: &[u8] = include_bytes!("../assets/models/dinov2.mpk");

static INPUT_IMAGE_0: &[u8] = include_bytes!("../assets/images/dino_0.png");
static INPUT_IMAGE_1: &[u8] = include_bytes!("../assets/images/dino_1.png");
static INPUT_IMAGE_2: &[u8] = include_bytes!("../assets/images/dino_2.png");
static INPUT_IMAGE_3: &[u8] = include_bytes!("../assets/images/dino_3.png");


pub fn load_model<B: Backend>(
    config: &DinoVisionTransformerConfig,
    device: &B::Device,
) -> DinoVisionTransformer<B> {
    let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
        .load(STATE_ENCODED.to_vec(), &Default::default())
        .expect("failed to decode state");

    let model= config.init(device);
    model.load_record(record)
}


fn normalize<B: Backend>(
    input: Tensor<B, 4>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mean: Tensor<B, 1> = Tensor::from_floats([0.485, 0.456, 0.406], device);
    let std: Tensor<B, 1> = Tensor::from_floats([0.229, 0.224, 0.225], device);

    input
        .permute([0, 2, 3, 1])
        .sub(mean.unsqueeze())
        .div(std.unsqueeze())
        .permute([0, 3, 1, 2])
}

pub fn load_image<B: Backend>(
    bytes: &[u8],
    config: &DinoVisionTransformerConfig,
    device: &B::Device,
) -> Tensor<B, 4> {
    let img = load_from_memory_with_format(bytes, ImageFormat::Png)
        .unwrap()
        .resize_exact(config.image_size as u32, config.image_size as u32, image::imageops::FilterType::Lanczos3);

    let img = match img {
        DynamicImage::ImageRgb8(img) => img,
        _ => img.to_rgb8(),
    };

    let img_data: Vec<f32> = img
        .pixels()
        .flat_map(|p| p.0.iter().map(|&c| c as f32 / 255.0))
        .collect();

    let input: Tensor<B, 1> = Tensor::from_floats(
        img_data.as_slice(),
        device,
    );

    let input = input.reshape([1, config.input_channels, config.image_size, config.image_size]);

    normalize(input, device)
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
    let output = dino.forward(batched_input.clone(), None).x_norm_patchtokens;

    let batch = output.shape().dims[0];
    let elements = output.shape().dims[1];
    let features = output.shape().dims[2];
    let n_samples = batch * elements;

    let spatial_size = elements.isqrt();

    let x = output.reshape([n_samples, features]);
    let binding = x.to_data()
        .to_vec::<f32>()
        .unwrap();
    let data: Vec<&[f32]> = binding
        .chunks(config.embedding_dimension)
        .collect();

    let tsne_features = tSNE::new(&data)
        .embedding_dim(3)
        .perplexity(10.0)
        .epochs(1000)
        .barnes_hut(0.5, |sample_a, sample_b| {
            sample_a.iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .embedding();
    let mut tsne_features = Array2::from_shape_vec((n_samples, 3), tsne_features).unwrap();

    for mut col in tsne_features.columns_mut() {
        let min = col.fold(f32::INFINITY, |a, &b| a.min(b));
        let max = col.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;
        col.mapv_inplace(|x| (x - min) / range);
    }

    let tsne_features = tsne_features.to_shape([batch, spatial_size, spatial_size, 3]).unwrap();

    for (i, img) in tsne_features.outer_iter().enumerate() {
        let collected: Vec<u8> = img.iter()
            .map(|&x| (x * 255.0)
                .max(0.0)
                .min(255.0) as u8
            ).collect();
        let img = RgbImage::from_raw(
                spatial_size as u32,
                spatial_size as u32,
                collected,
            )
            .unwrap();

        img.save(format!("output_{}.png", i)).unwrap();
    }
}
