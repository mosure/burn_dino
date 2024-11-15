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
};

use burn_dinov2::model::dinov2::{
    DinoVisionTransformer,
    DinoVisionTransformerConfig,
};


static STATE_ENCODED: &[u8] = include_bytes!("../assets/models/dinov2.mpk");
static INPUT_IMAGE_BYTES: &[u8] = include_bytes!("../assets/images/dino.png");


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
    config: &DinoVisionTransformerConfig,
    device: &B::Device,
) -> Tensor<B, 4> {
    let img = load_from_memory_with_format(INPUT_IMAGE_BYTES, ImageFormat::Png)
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
    let config = DinoVisionTransformerConfig::vits();
    let dino = load_model(&config, &device);

    let input: Tensor<Wgpu, 4> = load_image(&config, &device);
    let output = dino.forward(input.clone(), None);
    let output = output.x_norm_patchtokens;

    let batch = output.shape().dims[0];
    let spatial_size = output.shape().dims[1].isqrt();
    let features = output.shape().dims[2];

    let output = output
        .reshape([
            batch,
            spatial_size,
            spatial_size,
            features,
        ])
        .mean_dim(3);

    println!("normalized patchtokens {:?}", output.shape());

    let data: Vec<f32> = output.to_data().to_vec().unwrap();
    let mut img = DynamicImage::new_luma8(spatial_size as u32, spatial_size as u32);

    for (i, pixel) in data.iter().enumerate() {
        let x = (i % spatial_size) as u32;
        let y = (i / spatial_size) as u32;
        let pixel_value = (pixel * 255.0).clamp(0.0, 255.0) as u8;
        img.put_pixel(x, y, Luma([pixel_value]).to_rgba());
    }

    img.save("output.png").unwrap();

    println!("output saved to: {}", std::fs::canonicalize("output.png").unwrap().to_str().unwrap());
}
