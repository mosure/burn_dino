use burn::{module::Module, record::{FullPrecisionSettings, NamedMpkFileRecorder, Record, Recorder}};
use burn_import::pytorch::PyTorchFileRecorder;

use burn_dinov2::model::dinov2::{
    DinoVisionTransformer,
    DinoVisionTransformerConfig,
};

type Backend = burn::backend::NdArray<f32>;


fn main() {
    let device: Backend = Default::default();

    let model = DinoVisionTransformerConfig::vits().init(&device);

    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    model.load_file(
        "./assets/models/dinov2_vits14_pretrain.pth".into(),
        &recorder,
        &device,
    );

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(record, "./assets/models/dinov2".into())
        .expect("failed to save model record");
}
