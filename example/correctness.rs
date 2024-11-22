use burn::{
    prelude::*,
    backend::wgpu::Wgpu,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};
use image::{
    load_from_memory_with_format, DynamicImage, GenericImageView, ImageFormat,
};
use safetensors::tensor::SafeTensors;

use burn_dino::model::dino::{
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


fn center_crop(image: &DynamicImage, crop_width: u32, crop_height: u32) -> DynamicImage {
    let (img_width, img_height) = image.dimensions();

    let crop_width = crop_width.min(img_width);
    let crop_height = crop_height.min(img_height);

    let x = (img_width - crop_width) / 2;
    let y = (img_height - crop_height) / 2;

    image.crop_imm(x, y, crop_width, crop_height)
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

fn load_image<B: Backend>(
    bytes: &[u8],
    config: &DinoVisionTransformerConfig,
    device: &B::Device,
) -> Tensor<B, 4> {
    let img = load_from_memory_with_format(bytes, ImageFormat::Png)
        .unwrap()
        .resize_exact(config.image_size as u32 + 2, config.image_size as u32 + 2, image::imageops::FilterType::Lanczos3);
    let img = center_crop(&img, config.image_size as u32, config.image_size as u32);

    let img_data: Vec<f32> = img.to_rgb32f()
        .pixels()
        .flat_map(|p| p.0)
        .collect();

    let input: Tensor<B, 1> = Tensor::from_floats(
        img_data.as_slice(),
        device,
    );

    let input = input.reshape([1, config.image_size, config.image_size, config.input_channels])
        .permute([0, 3, 1, 2]);

    normalize(input, device)
}


fn main() {
    let device = Default::default();
    let config = DinoVisionTransformerConfig {
        ..DinoVisionTransformerConfig::vits(None, None)
    };
    let dino = load_model::<Wgpu>(&config, &device);

    // let debug_output = SafeTensors::deserialize(STANDARD_OUTPUT).unwrap();
    // let input = debug_output.tensor("input").unwrap();
    // let standard_input: Vec<f32> = input
    //     .data()
    //     .chunks_exact(4)
    //     .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
    //     .collect();
    // let standard_input_tensor = Tensor::<Wgpu, 1>::from_floats(standard_input.as_slice(), &device)
    //     .reshape([1, config.input_channels, config.image_size, config.image_size]);
    // let input_tensor = standard_input_tensor.clone();

    let input_tensor: Tensor<Wgpu, 4> = load_image(INPUT_IMAGE_0, &config, &device);

    // let standard_diff = input_tensor.clone().sub(standard_input_tensor.clone()).abs();
    // let min_diff = standard_diff.clone().min();
    // let max_diff = standard_diff.clone().max();
    // let range = max_diff - min_diff;
    // let standard_diff_min_max_norm = standard_diff.clone().div_scalar(range.into_scalar());

    // let standard_diff_min_max_norm_flat: Vec<f32> = standard_diff_min_max_norm.to_data().to_vec().unwrap();
    // let standard_diff_min_max_norm_flat: Vec<u8> = standard_diff_min_max_norm_flat
    //     .iter()
    //     .map(|&v| (v * 255.0).round() as u8)
    //     .collect();
    // let img = RgbImage::from_raw(
    //     config.image_size as u32,
    //     config.image_size as u32,
    //     standard_diff_min_max_norm_flat,
    // ).unwrap();
    // img.save("diff.png").unwrap();

    // let diff_sum = standard_diff.sum().into_scalar();
    // println!("diff sum: {}", diff_sum);

    // assert!(
    //     input_tensor.clone().all_close(standard_input_tensor, 2e-1.into(), None),
    //     "input does not match torch reference",
    // );

    let output = dino.forward(input_tensor, None);
    let x_norm_patchtokens = output.x_norm_patchtokens;
    let x_norm_patchtokens_flat: Vec<f32> = x_norm_patchtokens.to_data().to_vec().unwrap();

    let standard_output = SafeTensors::deserialize(STANDARD_OUTPUT)
        .unwrap()
        .tensor("output")
        .unwrap();
    let standard_output_tensor: Vec<f32> = standard_output
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let mse: f32 = x_norm_patchtokens_flat
        .iter()
        .zip(standard_output_tensor.iter())
        .map(|(a, b)| (a - b).powf(2.0))
        .sum::<f32>() / x_norm_patchtokens_flat.len() as f32;

    println!("(MSE): {}", mse);

    assert!(mse < 2e-2, "output does not match torch reference");
}
