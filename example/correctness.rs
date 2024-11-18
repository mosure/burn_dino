use burn::{
    prelude::*,
    backend::wgpu::Wgpu,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};
use image::{
    load_from_memory_with_format,
    ImageFormat,
};
use safetensors::tensor::SafeTensors;

use burn_dinov2::model::dinov2::{
    DinoVisionTransformer,
    DinoVisionTransformerConfig,
};


static STATE_ENCODED: &[u8] = include_bytes!("../assets/models/dinov2.mpk");
static INPUT_IMAGE_0: &[u8] = include_bytes!("../assets/images/dino_0.png");
static STANDARD_OUTPUT: &[u8] = include_bytes!("../assets/tensors/dino_0_small_debug.st");


pub fn load_model<B: Backend>(
    config: &DinoVisionTransformerConfig,
    device: &B::Device,
) -> DinoVisionTransformer<B> {
    let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
        .load(STATE_ENCODED.to_vec(), &Default::default())
        .expect("failed to decode state");

    let model= config.init(device);

    let min_pos_embed = model.pos_embed.val().min();
    let max_pos_embed = model.pos_embed.val().max();
    let mean_pos_embed = model.pos_embed.val().mean();
    println!("INIT ------ Pos Embed Min: {}, Max: {}, Mean: {}", min_pos_embed, max_pos_embed, mean_pos_embed);

    let model = model.load_record(record);

    let min_pos_embed = model.pos_embed.val().min();
    let max_pos_embed = model.pos_embed.val().max();
    let mean_pos_embed = model.pos_embed.val().mean();
    println!("LOAD ------ Pos Embed Min: {}, Max: {}, Mean: {}", min_pos_embed, max_pos_embed, mean_pos_embed);

    model
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

    let img_data: Vec<f32> = img.to_rgb32f()
        .pixels()
        .flat_map(|p| p.0)
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
    let dino = load_model::<Wgpu>(&config, &device);

    let debug_output = SafeTensors::deserialize(STANDARD_OUTPUT).unwrap();
    let input = debug_output.tensor("input").unwrap();
    let standard_input: Vec<f32> = input
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // let input_tensor: Tensor<Wgpu, 4> = load_image(INPUT_IMAGE_0, &config, &device);
    // let input_tensor = Tensor::zeros([1, config.input_channels, config.image_size, config.image_size], &device);
    let input_tensor = Tensor::<Wgpu, 1>::from_floats(standard_input.as_slice(), &device)
        .reshape([1, config.input_channels, config.image_size, config.image_size]);

    let patch_embed = debug_output.tensor("patch_embed").unwrap();
    let interpolated_pos_encoding = debug_output.tensor("interpolated_pos_encoding").unwrap();
    let pos_embed_raw = debug_output.tensor("pos_embed_raw").unwrap();
    let patch_pos_embed_pre = debug_output.tensor("patch_pos_embed").unwrap();
    let patch_pos_embed_post = debug_output.tensor("patch_pos_embed_interpolated").unwrap();
    let prepared_tokens = debug_output.tensor("prepared_tokens").unwrap();
    let block_0 = debug_output.tensor("block_0").unwrap();
    let block_0_attn_residual = debug_output.tensor("block_0_attn_residual").unwrap();
    let block_0_attn = debug_output.tensor("block_0_attn").unwrap();
    let block_0_attn_norm = debug_output.tensor("block_0_attn_norm").unwrap();
    let block_0_attn_norm_weight = debug_output.tensor("block_0_attn_norm_weight").unwrap();
    let block_0_attn_norm_bias = debug_output.tensor("block_0_attn_norm_bias").unwrap();
    let block_0_mlp_residual = debug_output.tensor("block_0_mlp_residual").unwrap();
    let block_0_mlp = debug_output.tensor("block_0_mlp").unwrap();
    let block_0_mlp_norm = debug_output.tensor("block_0_mlp_norm").unwrap();

    let output = dino.forward(input_tensor, None);

    let x_norm_patchtokens = output.x_norm_patchtokens;
    let x_norm_patchtokens_flat: Vec<f32> = x_norm_patchtokens.to_data().to_vec().unwrap();
    let min = x_norm_patchtokens.clone().min();
    let max = x_norm_patchtokens.clone().max();
    let mean = x_norm_patchtokens.clone().mean();
    println!("Min: {}, Max: {}, Mean: {}", min, max, mean);

    let actual_input = output.input;
    let actual_patch_embed = output.patch_embed;
    let actual_pos_embed_raw = dino.pos_embed.clone();
    let actual_interpolated_pos_encoding = output.interpolated_pos_encoding;
    let actual_patch_pos_embed_pre = output.patch_pos_embed_pre;
    let actual_patch_pos_embed_post = output.patch_pos_embed_post;
    let actual_prepared_tokens = output.prepared_tokens;
    let actual_block_0 = output.block_0;
    let actual_block_0_attn_residual = output.block_0_attn_residual;
    let actual_block_0_attn = output.block_0_attn;
    let actual_block_0_attn_norm = output.block_0_attn_norm;
    let actual_block_0_attn_norm_weight = output.block_0_attn_norm_weight;
    let actual_block_0_attn_norm_bias = output.block_0_attn_norm_bias;
    let actual_block_0_mlp_residual = output.block_0_mlp_residual;
    let actual_block_0_mlp = output.block_0_mlp;
    let actual_block_0_mlp_norm = output.block_0_mlp_norm;

    let actual_input_min = actual_input.clone().min();
    let actual_input_max = actual_input.clone().max();
    let actual_input_mean = actual_input.clone().mean();
    println!("Actual Input Min: {}, Max: {}, Mean: {}", actual_input_min, actual_input_max, actual_input_mean);

    let actual_pos_embed_min = actual_pos_embed_raw.val().min();
    let actual_pos_embed_max = actual_pos_embed_raw.val().max();
    let actual_pos_embed_mean = actual_pos_embed_raw.val().mean();
    println!("Actual Pos Embed Min: {}, Max: {}, Mean: {}", actual_pos_embed_min, actual_pos_embed_max, actual_pos_embed_mean);

    let standard_patch_embed: Vec<f32> = patch_embed
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_pos_embed_raw: Vec<f32> = pos_embed_raw
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_interpolated_pos_encoding: Vec<f32> = interpolated_pos_encoding
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_patch_pos_embed_pre: Vec<f32> = patch_pos_embed_pre
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_patch_pos_embed_post: Vec<f32> = patch_pos_embed_post
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_prepared_tokens: Vec<f32> = prepared_tokens
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_block_0: Vec<f32> = block_0
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_block_0_attn_residual: Vec<f32> = block_0_attn_residual
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_block_0_attn: Vec<f32> = block_0_attn
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_block_0_attn_norm: Vec<f32> = block_0_attn_norm
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_block_0_attn_norm_weight: Vec<f32> = block_0_attn_norm_weight
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_block_0_attn_norm_bias: Vec<f32> = block_0_attn_norm_bias
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_block_0_mlp_residual: Vec<f32> = block_0_mlp_residual
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_block_0_mlp: Vec<f32> = block_0_mlp
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let standard_block_0_mlp_norm: Vec<f32> = block_0_mlp_norm
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let data = actual_input
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mse_input: f32 = data.iter()
        .zip(standard_input.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / data.len() as f32;

    let data = actual_patch_embed
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_patch_embed: f32 = data.iter()
        .zip(standard_patch_embed.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_patch_embed.len());

    let data = actual_pos_embed_raw
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_pos_embed_raw: f32 = data.iter()
        .zip(standard_pos_embed_raw.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_pos_embed_raw.len());

    let data = actual_interpolated_pos_encoding
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_interpolated_pos_encoding: f32 = data.iter()
        .zip(standard_interpolated_pos_encoding.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_interpolated_pos_encoding.len());

    let data = actual_patch_pos_embed_pre
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_patch_pos_embed_pre: f32 = data.iter()
        .zip(standard_patch_pos_embed_pre.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_patch_pos_embed_pre.len());

    let data = actual_patch_pos_embed_post
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_patch_pos_embed_post: f32 = data.iter()
        .zip(standard_patch_pos_embed_post.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_patch_pos_embed_post.len());

    let data = actual_prepared_tokens
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_prepared_tokens: f32 = data.iter()
        .zip(standard_prepared_tokens.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_prepared_tokens.len());

    let data = actual_block_0
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_block_0: f32 = data.iter()
        .zip(standard_block_0.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_block_0.len());

    let data = actual_block_0_attn_residual
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_block_0_attn_residual: f32 = data.iter()
        .zip(standard_block_0_attn_residual.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_block_0_attn_residual.len());

    let data = actual_block_0_attn
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_block_0_attn: f32 = data.iter()
        .zip(standard_block_0_attn.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_block_0_attn.len());

    let data = actual_block_0_attn_norm
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_block_0_attn_norm: f32 = data.iter()
        .zip(standard_block_0_attn_norm.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_block_0_attn_norm.len());

    let data = actual_block_0_attn_norm_weight
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_block_0_attn_norm_weight: f32 = data.iter()
        .zip(standard_block_0_attn_norm_weight.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;

    let data = actual_block_0_attn_norm_bias
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_block_0_attn_norm_bias: f32 = data.iter()
        .zip(standard_block_0_attn_norm_bias.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;

    let data = actual_block_0_mlp_residual
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_block_0_mlp_residual: f32 = data.iter()
        .zip(standard_block_0_mlp_residual.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_block_0_mlp_residual.len());

    let data = actual_block_0_mlp
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_block_0_mlp: f32 = data.iter()
        .zip(standard_block_0_mlp.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_block_0_mlp.len());

    let data = actual_block_0_mlp_norm
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_block_0_mlp_norm: f32 = data.iter()
        .zip(standard_block_0_mlp_norm.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / data.len() as f32;
    assert!(data.len() == standard_block_0_mlp_norm.len());

    println!("(MSE) for input: {}", mse_input);
    println!("(MAE) for patch_embed: {}", mae_patch_embed);
    println!("(MAE) for interpolated_pos_encoding: {}", mae_interpolated_pos_encoding);
    println!("(MAE) for pos_embed_raw: {}", mae_pos_embed_raw);
    println!("(MAE) for patch_pos_embed_pre: {}", mae_patch_pos_embed_pre);
    println!("(MAE) for patch_pos_embed_post: {}", mae_patch_pos_embed_post);
    println!("(MAE) for prepared_tokens: {}", mae_prepared_tokens);
    println!("(MAE) for block_0: {}", mae_block_0);
    println!("(MAE) for block_0_attn_residual: {}", mae_block_0_attn_residual);
    println!("(MAE) for block_0_attn: {}", mae_block_0_attn);
    println!("(MAE) for block_0_attn_norm: {}", mae_block_0_attn_norm);
    println!("(MAE) for block_0_attn_norm_weight: {}", mae_block_0_attn_norm_weight);
    println!("(MAE) for block_0_attn_norm_bias: {}", mae_block_0_attn_norm_bias);
    println!("(MAE) for block_0_mlp_residual: {}", mae_block_0_mlp_residual);
    println!("(MAE) for block_0_mlp: {}", mae_block_0_mlp);
    println!("(MAE) for block_0_mlp_norm: {}", mae_block_0_mlp_norm);

    let standard_output = SafeTensors::deserialize(STANDARD_OUTPUT)
        .unwrap()
        .tensor("output")
        .unwrap();

    assert_eq!(x_norm_patchtokens.shape().dims.as_slice(), standard_output.shape());

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

    assert!(mse < 1e-4, "output does not match torch reference");
}
