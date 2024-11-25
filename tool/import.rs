use bevy_args::{
    parse_args,
    Deserialize,
    Parser,
    Serialize,
    ValueEnum,
};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

use burn_dino::{
    layers::jbu::JbuStackRecord,
    model::{
        dino::DinoVisionTransformerRecord,
        pca::PcaTransformRecord,
    },
};
use cubecl::wgpu::WgpuRuntime;


#[derive(
    Debug,
    Default,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    ValueEnum,
)]
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
            Self::Small => "dinov2_vits14_pretrain.pth",
            Self::Base => "dinov2_vitb14_pretrain.pth",
            Self::Large => "dinov2_vitl14_pretrain.pth",
            Self::Giant => "dinov2_vitg14_pretrain.pth",
        }
    }
}

#[derive(
    Clone,
    Debug,
    Default,
    Serialize,
    Deserialize,
    Parser,
)]
#[command(about = "burn_dino import", version, long_about = None)]
pub struct DinoImportConfig {
    #[arg(long, value_enum, default_value_t = VitType::Small)]
    pub vit_type: VitType,
}


// type ImportBackend = burn::backend::Wgpu;
type ImportBackend = burn::backend::wgpu::JitBackend<WgpuRuntime, f32, i32>;

fn main() {
    let args = parse_args::<DinoImportConfig>();

    let device = Default::default();

    let weights_path = format!("./assets/models/{}", args.vit_type.weights_file());
    println!("loading weights from: {}", weights_path);

    let load_args = LoadArgs::new(weights_path.into())
        .with_debug_print();
    let record: DinoVisionTransformerRecord<ImportBackend> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .expect("failed to decode state");

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(record, "./assets/models/dinov2".into())
        .expect("failed to save model record");


    // pca layer
    let pca_weights = "./assets/models/dino_pca.pth";
    let load_args = LoadArgs::new(pca_weights.into())
        .with_debug_print();

    let record: PcaTransformRecord<ImportBackend> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .expect("failed to decode state");

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(record, "./assets/models/pca".into())
        .expect("failed to save model record");


    // upsampler
    let upsampler_weights = "./assets/models/upsampler.pth";
    let load_args = LoadArgs::new(upsampler_weights.into())
        .with_key_remap("range_proj\\.0\\.(.*)", "range_proj.conv1.$1")
        .with_key_remap("range_proj\\.3\\.(.*)", "range_proj.conv2.$1")
        .with_key_remap("fixup_proj\\.0\\.(.*)", "fixup_proj.conv1.$1")
        .with_key_remap("fixup_proj\\.3\\.(.*)", "fixup_proj.conv2.$1")
        .with_key_remap("fixup_proj\\.1\\.(.*)", "fixup_proj.conv.$1")
        .with_debug_print();

    let record: JbuStackRecord<ImportBackend> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .expect("failed to decode state");

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(record, "./assets/models/upsampler".into())
        .expect("failed to save model record");
}
