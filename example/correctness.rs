use std::{env, error::Error, path::PathBuf};

use burn::{backend::ndarray::NdArray, tensor::backend::Backend};
use burn_dino::{
    correctness::{self, CorrectnessReference},
    model::dino::DinoVisionTransformerConfig,
};

type InferenceBackend = NdArray<f32>;

fn main() -> Result<(), Box<dyn Error>> {
    let checkpoint_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/models/dinov2.mpk"));
    let reference_path = env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/correctness/dinov2_vits14_reference.safetensors"));

    let device = <InferenceBackend as Backend>::Device::default();
    let config = DinoVisionTransformerConfig::vits(None, None);

    println!("Loading Burn checkpoint from {}", checkpoint_path.display());
    println!("Using Torch reference from {}", reference_path.display());

    let reference = CorrectnessReference::load(&reference_path)?;
    let model = correctness::load_model_from_checkpoint::<InferenceBackend>(
        &config,
        &checkpoint_path,
        &device,
    )?;
    let stats = correctness::run_correctness(&model, &reference, &device)?;

    println!(
        "Patch tokens -> mean_abs: {:.6}, max_abs: {:.6}, mse: {:.6}",
        stats.patch_tokens.mean_abs, stats.patch_tokens.max_abs, stats.patch_tokens.mse
    );
    println!(
        "Cls token    -> mean_abs: {:.6}, max_abs: {:.6}, mse: {:.6}",
        stats.cls_token.mean_abs, stats.cls_token.max_abs, stats.cls_token.mse
    );
    if let Some(reg) = stats.register_tokens.as_ref() {
        println!(
            "Register tok -> mean_abs: {:.6}, max_abs: {:.6}, mse: {:.6}",
            reg.mean_abs, reg.max_abs, reg.mse
        );
    }

    if stats.within_defaults() {
        println!("Burn output matches Torch reference within tolerance.");
        return Ok(());
    }

    if let Ok(outputs) = correctness::collect_outputs(&model, &reference, &device) {
        for idx in 0..5 {
            println!(
                "patch[{idx}]: burn={:.6}, torch={:.6}, diff={:.6}",
                outputs.burn_patchtokens[idx],
                outputs.torch_patchtokens[idx],
                outputs.burn_patchtokens[idx] - outputs.torch_patchtokens[idx]
            );
        }

        if let Some((max_idx, max_value, burn, torch)) = outputs
            .burn_patchtokens
            .iter()
            .zip(outputs.torch_patchtokens.iter())
            .enumerate()
            .map(|(idx, (b, t))| (idx, (b - t).abs(), *b, *t))
            .max_by(|a, b| a.1.total_cmp(&b.1))
        {
            println!(
                "max diff patch[{max_idx}]: burn={burn:.6}, torch={torch:.6}, diff={max_value:.6}"
            );
        }
    }

    Err("Burn output deviates from Torch reference beyond tolerance.".into())
}
