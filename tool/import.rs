use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;

use burn_dinov2::model::dinov2::DinoVisionTransformerRecord;

type Backend = burn::backend::NdArray<f32>;


fn main() {
    let device = Default::default();

    let record: DinoVisionTransformerRecord<Backend> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(
            "./assets/models/dinov2_vits14_pretrain.pth".into(),
            &device,
        )
        .expect("failed to decode state");

    // let model = DinoVisionTransformerConfig::vits().init(&device);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(record, "./assets/models/dinov2".into())
        .expect("failed to save model record");
}
