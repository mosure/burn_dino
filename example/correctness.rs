use burn::{
    prelude::*,
    backend::wgpu::Wgpu,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};
use image::{
    load_from_memory_with_format,
    DynamicImage,
    ImageFormat,
};
use safetensors::tensor::SafeTensors;

use burn_dinov2::model::dinov2::{
    DinoVisionTransformer,
    DinoVisionTransformerConfig,
};


static STATE_ENCODED: &[u8] = include_bytes!("../assets/models/dinov2.mpk");
static INPUT_IMAGE_0: &[u8] = include_bytes!("../assets/images/dino_0.png");
static STANDARD_OUTPUT: &[u8] = include_bytes!("../assets/tensors/dino_0_small.st");


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

    let input_tensor: Tensor<Wgpu, 4> = load_image(INPUT_IMAGE_0, &config, &device);
    let output = dino.forward(input_tensor, None).x_norm_patchtokens;

    let min = output.clone().min();
    let max = output.clone().max();
    let mean = output.clone().mean();
    println!("Min: {}, Max: {}, Mean: {}", min, max, mean);

    let output_flat: Vec<f32> = output.to_data().to_vec().unwrap();

    let standard_output = SafeTensors::deserialize(STANDARD_OUTPUT)
        .unwrap()
        .tensor("output")
        .unwrap();

    assert_eq!(output.shape().dims.as_slice(), standard_output.shape());

    let standard_output_tensor: Vec<f32> = standard_output
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let mae: f32 = output_flat
        .iter()
        .zip(standard_output_tensor.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / output_flat.len() as f32;

    println!("Mean Absolute Error (MAE): {}", mae);

    assert!(mae < 1e-4, "Output does not match Torch reference");
}
