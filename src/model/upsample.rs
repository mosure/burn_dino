use burn::prelude::*;

use crate::{
    kernels::adaptive_conv::Backend,
    layers::{
        channel_norm::{
            ChannelNorm,
            ChannelNormConfig,
        },
        jbu::{
            JbuStack,
            JbuStackConfig,
        },
    },
};


// TODO: move to dedicated repo (or separate crate)
#[derive(Config)]
pub struct FeatureUpsampleConfig {
    pub jbu: JbuStackConfig,
    pub use_norm: bool,
}

impl Default for FeatureUpsampleConfig {
    fn default() -> Self {
        Self::new(JbuStackConfig::default(), false)
    }
}

impl FeatureUpsampleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeatureUpsample<B> {
        FeatureUpsample::new(device, &self)
    }
}


#[derive(Module, Debug)]
pub struct FeatureUpsample<B: Backend> {
    channel_norm: Option<ChannelNorm<B>>,
    upsampler: JbuStack<B>,
}

impl<B: Backend> FeatureUpsample<B> {
    pub fn new(
        device: &B::Device,
        config: &FeatureUpsampleConfig,
    ) -> Self {
        let channel_norm = if config.use_norm {
            Some(ChannelNorm::new(
                device,
                &ChannelNormConfig::new(config.jbu.feat_dim),
            ))
        } else {
            None
        };

        let upsampler = config.jbu.init(device);

        Self {
            channel_norm,
            upsampler,
        }
    }

    pub fn forward(
        &self,
        features: Tensor<B, 4>,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let features = if let Some(channel_norm) = &self.channel_norm {
            channel_norm.forward(features)
        } else {
            features
        };
        self.upsampler.forward(features, input)
    }
}
