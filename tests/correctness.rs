#![cfg(feature = "backend_ndarray")]

use std::path::PathBuf;

use burn::{backend::ndarray::NdArray, tensor::backend::Backend};
use burn_dino::{
    correctness::{self, CorrectnessReference},
    model::dino::DinoVisionTransformerConfig,
};

type TestBackend = NdArray<f32>;

#[test]
fn dinov2_vits14_matches_torch_reference() {
    let checkpoint = PathBuf::from("assets/models/dinov2.mpk");
    let reference = PathBuf::from("assets/correctness/dinov2_vits14_reference.safetensors");

    assert!(
        checkpoint.exists(),
        "Burn checkpoint {} missing. Run `cargo run --bin import --features import` first.",
        checkpoint.display()
    );
    assert!(
        reference.exists(),
        "Reference file {} missing. Run `python tool/correctness.py` first.",
        reference.display()
    );

    let device = <TestBackend as Backend>::Device::default();
    let config = DinoVisionTransformerConfig::vits(None, None);
    let reference_data =
        CorrectnessReference::load(&reference).expect("Failed to load reference tensors");
    let model =
        correctness::load_model_from_checkpoint::<TestBackend>(&config, &checkpoint, &device)
            .expect("Failed to load Burn checkpoint");
    let stats =
        correctness::run_correctness(&model, &reference_data, &device).expect("Correctness run");

    assert!(
        stats.within_defaults(),
        "Burn outputs diverge from Torch reference. patch max_abs={:.6}, mse={:.6}",
        stats.patch_tokens.max_abs,
        stats.patch_tokens.mse
    );
}
