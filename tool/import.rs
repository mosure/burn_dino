use std::{
    fs,
    path::{Path, PathBuf},
};

use bevy_args::{Deserialize, Parser, Serialize, ValueEnum, parse_args};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use burn_store::{
    ApplyResult, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};

use burn_dino::{
    correctness::{self, CorrectnessReference},
    model::{
        dino::{DinoVisionTransformer, DinoVisionTransformerConfig},
        pca::{PcaTransform, PcaTransformConfig},
    },
};

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize, ValueEnum)]
pub enum VitType {
    #[default]
    Small,
    Base,
    Large,
    Giant,
}

impl VitType {
    fn weights_file(&self) -> &'static str {
        match self {
            Self::Small => "dinov2_vits14_pretrain.safetensors",
            Self::Base => "dinov2_vitb14_pretrain.safetensors",
            Self::Large => "dinov2_vitl14_pretrain.safetensors",
            Self::Giant => "dinov2_vitg14_pretrain.safetensors",
        }
    }

    fn reference_file(&self) -> &'static str {
        match self {
            Self::Small => "dinov2_vits14_reference.safetensors",
            Self::Base => "dinov2_vitb14_reference.safetensors",
            Self::Large => "dinov2_vitl14_reference.safetensors",
            Self::Giant => "dinov2_vitg14_reference.safetensors",
        }
    }

    fn config(&self) -> DinoVisionTransformerConfig {
        match self {
            Self::Small => DinoVisionTransformerConfig::vits(None, None),
            Self::Base => DinoVisionTransformerConfig::vitb(None, None),
            Self::Large => DinoVisionTransformerConfig::vitl(None, None),
            Self::Giant => DinoVisionTransformerConfig::vitg(None, None),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, Parser)]
#[command(about = "burn_dino import", version, long_about = None)]
pub struct DinoImportConfig {
    #[arg(long, value_enum, default_value_t = VitType::Small)]
    pub vit_type: VitType,

    #[arg(long)]
    pub weights_path: Option<PathBuf>,

    #[arg(long, default_value = "assets/models/dinov2")]
    pub output: PathBuf,

    #[arg(long)]
    pub reference: Option<PathBuf>,

    #[arg(long)]
    pub validate: bool,

    #[arg(long)]
    pub register_tokens: Option<usize>,

    #[arg(long, default_value = "assets/models/face_pca.safetensors")]
    pub pca_weights: PathBuf,

    #[arg(long, default_value = "assets/models/pca")]
    pub pca_output: PathBuf,

    #[arg(long)]
    pub skip_pca: bool,
}

type Backend = burn::backend::NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args::<DinoImportConfig>();
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let mut config = args.vit_type.config();
    if let Some(register_tokens) = args.register_tokens {
        config = config.with_register_tokens(register_tokens);
    }

    let checkpoint_path = import_dino_weights(&device, &args, &config)?;

    if !args.skip_pca {
        import_pca_weights(&device, &args)?;
    }

    if args.validate {
        let reference_path = args.reference.clone().unwrap_or_else(|| {
            PathBuf::from("assets/correctness").join(args.vit_type.reference_file())
        });
        run_validation(&device, &checkpoint_path, &reference_path, &config)?;
    }

    Ok(())
}

fn import_dino_weights(
    device: &<Backend as burn::tensor::backend::Backend>::Device,
    args: &DinoImportConfig,
    config: &DinoVisionTransformerConfig,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let weights_path = args
        .weights_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("assets/models").join(args.vit_type.weights_file()));

    if !weights_path.exists() {
        if weights_path
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("safetensors"))
            .unwrap_or(false)
        {
            let legacy = weights_path.with_extension("pth");
            if legacy.exists() {
                return Err(format!(
                    "Safetensors checkpoint '{}' not found. Convert the PyTorch file `{}` using `python tool/export_weights.py --input {}`.",
                    weights_path.display(),
                    legacy.display(),
                    legacy.display()
                )
                .into());
            }
        }
        return Err(format!(
            "Checkpoint '{}' not found. Run `python tool/export_weights.py` first.",
            weights_path.display()
        )
        .into());
    }

    let (output_base, checkpoint_file) = normalize_output_paths(&args.output);
    if let Some(parent) = output_base.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Loading PyTorch weights from {}", weights_path.display());

    let mut model = DinoVisionTransformer::<Backend>::new(device, config.clone());
    let mut store = build_dino_store(&weights_path)?;
    let result = model
        .load_from(&mut store)
        .map_err(|err| format!("failed to apply PyTorch checkpoint: {err}"))?;
    report_apply_result("dino", &result);

    model
        .clone()
        .save_file(
            output_base.clone(),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .map_err(|err| format!("failed to save model record: {err}"))?;

    println!("Saved Burn checkpoint to {}", checkpoint_file.display());
    Ok(checkpoint_file)
}

fn import_pca_weights(
    device: &<Backend as burn::tensor::backend::Backend>::Device,
    args: &DinoImportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if !args.pca_weights.exists() {
        println!(
            "Skipping PCA import. weights file '{}' not found. Run `python tool/export_weights.py --input assets/models/face_pca.pth` first.",
            args.pca_weights.display()
        );
        return Ok(());
    }

    let (output_base, output_file) = normalize_output_paths(&args.pca_output);
    if let Some(parent) = output_base.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut model = PcaTransform::<Backend>::new(device, &PcaTransformConfig::default());
    let mut store = build_pca_store(&args.pca_weights)?;
    let result = model
        .load_from(&mut store)
        .map_err(|err| format!("failed to apply PCA weights: {err}"))?;
    report_apply_result("pca", &result);

    model
        .clone()
        .save_file(
            output_base.clone(),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .map_err(|err| format!("failed to save PCA record: {err}"))?;

    println!("Saved PCA checkpoint to {}", output_file.display());
    Ok(())
}

fn run_validation(
    device: &<Backend as burn::tensor::backend::Backend>::Device,
    checkpoint_path: &Path,
    reference_path: &Path,
    config: &DinoVisionTransformerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if !reference_path.exists() {
        println!(
            "Reference file '{}' not found. Skipping correctness validation.",
            reference_path.display()
        );
        return Ok(());
    }

    println!(
        "Validating Burn checkpoint against reference {}",
        reference_path.display()
    );

    let reference = CorrectnessReference::load(reference_path)?;
    let model =
        correctness::load_model_from_checkpoint::<Backend>(config, checkpoint_path, device)?;

    let stats = correctness::run_correctness(&model, &reference, device)?;
    println!(
        "patch_tokens: mean_abs={:.6}, max_abs={:.6}, mse={:.6}",
        stats.patch_tokens.mean_abs, stats.patch_tokens.max_abs, stats.patch_tokens.mse
    );
    println!(
        "cls_token: mean_abs={:.6}, max_abs={:.6}, mse={:.6}",
        stats.cls_token.mean_abs, stats.cls_token.max_abs, stats.cls_token.mse
    );
    if let Some(reg) = stats.register_tokens.as_ref() {
        println!(
            "reg_tokens: mean_abs={:.6}, max_abs={:.6}, mse={:.6}",
            reg.mean_abs, reg.max_abs, reg.mse
        );
    }

    if stats.within_defaults() {
        println!("Correctness validation passed.");
        Ok(())
    } else {
        Err(
            "Burn output deviates from Torch reference beyond tolerance."
                .to_string()
                .into(),
        )
    }
}

fn normalize_output_paths(path: &Path) -> (PathBuf, PathBuf) {
    let base = if path
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("mpk"))
        .unwrap_or(false)
    {
        path.with_extension("")
    } else {
        path.to_path_buf()
    };

    let file = base.with_extension("mpk");
    (base, file)
}

fn build_dino_store(path: &Path) -> Result<SafetensorsStore, Box<dyn std::error::Error>> {
    let mut remapper = KeyRemapper::new();
    for &(from, to) in key_remap_rules() {
        remapper = remapper
            .add_pattern(from, to)
            .map_err(|err| format!("invalid remap rule {from}->{to}: {err}"))?;
    }

    let store = SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .remap(remapper)
        .validate(true);

    Ok(store)
}

fn build_pca_store(path: &Path) -> Result<SafetensorsStore, Box<dyn std::error::Error>> {
    Ok(SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .validate(true))
}

fn key_remap_rules() -> &'static [(&'static str, &'static str)] {
    &[
        (r"^(blocks\.\d+\.norm\d?)\.weight$", "$1.gamma"),
        (r"^(blocks\.\d+\.norm\d?)\.bias$", "$1.beta"),
        (r"^(norm)\.weight$", "$1.gamma"),
        (r"^(norm)\.bias$", "$1.beta"),
    ]
}

fn report_apply_result(prefix: &str, result: &ApplyResult) {
    println!(
        "[IMPORT] {} tensors applied: {} (missing {}, unused {}, skipped {})",
        prefix,
        result.applied.len(),
        result.missing.len(),
        result.unused.len(),
        result.skipped.len()
    );

    if !result.missing.is_empty() {
        println!("[IMPORT] Missing {} tensor(s):", result.missing.len());
        for key in &result.missing {
            println!("  - {key}");
        }
    }

    if !result.unused.is_empty() {
        println!("[IMPORT] Unused {} tensor(s):", result.unused.len());
        for key in &result.unused {
            println!("  - {key}");
        }
    }

    if !result.skipped.is_empty() {
        println!("[IMPORT] Skipped {} tensor(s):", result.skipped.len());
        for key in &result.skipped {
            println!("  - {key}");
        }
    }
}
