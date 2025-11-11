pub mod correctness;
pub mod layers;
pub mod model;

#[cfg(test)]
mod tests {
    use super::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
    use burn::prelude::*;

    #[cfg(feature = "backend_ndarray")]
    type NdArrayBackend = burn::backend::NdArray<f32>;

    fn test_config() -> DinoVisionTransformerConfig {
        DinoVisionTransformerConfig::vits(None, None)
    }

    fn build_model<B: Backend>(device: &B::Device) -> DinoVisionTransformer<B> {
        DinoVisionTransformer::new(device, test_config())
    }

    #[test]
    #[cfg(feature = "backend_ndarray")]
    fn dino_initializes_ndarray() {
        let device = <NdArrayBackend as Backend>::Device::default();
        let _ = build_model::<NdArrayBackend>(&device);
    }

    #[test]
    #[cfg(feature = "backend_ndarray")]
    fn dino_roundtrip_record_ndarray() {
        let device = <NdArrayBackend as Backend>::Device::default();
        let model = build_model::<NdArrayBackend>(&device);
        let record = model.clone().into_record();
        let loaded = build_model::<NdArrayBackend>(&device).load_record(record);
        let size = loaded.pos_embed.shape().dims[2];
        assert_eq!(size, model.pos_embed.shape().dims[2]);
    }

    #[test]
    #[cfg(feature = "backend_ndarray")]
    fn dino_runs_inference_ndarray() {
        let device = <NdArrayBackend as Backend>::Device::default();
        let config = test_config();
        let embed_dim = config.embedding_dimension;
        let image_size = config.image_size;
        let model = DinoVisionTransformer::new(&device, config);
        let input = Tensor::<NdArrayBackend, 4>::zeros([1, 3, image_size, image_size], &device);
        let output = model.forward(input, None);
        assert_eq!(output.x_norm_patchtokens.shape().dims[2], embed_dim);
    }
}
