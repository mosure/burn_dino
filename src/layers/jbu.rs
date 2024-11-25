use burn::{
    prelude::*,
    module::Param,
    nn::{
        conv::{
            Conv2d,
            Conv2dConfig
        },
        Dropout,
        DropoutConfig,
        interpolate::{
            Interpolate2d,
            Interpolate2dConfig,
            InterpolateMode,
        },
        pool::{
            AdaptiveAvgPool2d,
            AdaptiveAvgPool2dConfig,
        },
        Initializer,
        Unfold4d,
        Unfold4dConfig,
    },
    tensor::activation::softmax,
};

use crate::{
    kernels::adaptive_conv::{
        adaptive_conv,
        Backend,
    },
    layers::projection::{
        Projection,
        ProjectionConfig,
    },
};


#[derive(Config)]
pub struct JbuLearnedRangeConfig {
    pub guidance_dim: usize,
    pub feat_dim: usize,
    pub key_dim: usize,
    pub radius: usize,
    pub width: usize,
    pub height: usize,
}

impl Default for JbuLearnedRangeConfig {
    fn default() -> Self {
        Self::new(3, 384, 32, 3, 518, 518)
    }
}

impl JbuLearnedRangeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> JbuLearnedRange<B> {
        JbuLearnedRange::new(device, &self)
    }
}


fn einsum_bchwp_bchw_to_bphw<B: Backend>(
    queries: Tensor<B, 5>,
    proj_x: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let proj_x_unsqueezed: Tensor<B, 5> = proj_x.unsqueeze_dim(4);
    let product = queries * proj_x_unsqueezed;
    let summed = product.sum_dim(1).squeeze(1);
    let out = summed.permute([0, 3, 1, 2]);

    out
}

fn linspace<B: Backend>(
    start: f32,
    end: f32,
    steps: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    if steps == 1 {
        return Tensor::<B, 1>::from_data([start], device);
    }

    let indices = Tensor::<B, 1>::from_data(
        (0..steps)
            .map(|i| i as f32)
            .collect::<Vec<f32>>()
            .as_slice(),
        device,
    );

    let step = (end - start) / ((steps - 1) as f32);
    let linspace = indices
        .mul_scalar(step)
        .add_scalar(start);

    linspace
}

fn meshgrid<B: Backend, const I: usize>(
    inputs: [Tensor<B, 1>; I],
) -> Vec<Tensor<B, I>> {
    assert!(I >= 1, "at least one input tensor is required.");

    let sizes: Vec<usize> = inputs.iter().map(|t| t.shape().dims[0]).collect();
    let mut grids = Vec::with_capacity(I);

    for (i, input) in inputs.iter().enumerate() {
        let mut shape = [1; I];
        shape[i] = sizes[i];

        let reshaped = input.clone().reshape(shape);

        let repeats: Vec<usize> = sizes.iter().enumerate().map(|(j, &size)| {
            if i == j {
                1
            } else {
                size
            }
        }).collect();

        let repeated = reshaped.repeat(&repeats);

        grids.push(repeated);
    }

    grids
}


#[derive(Module, Debug)]
pub struct JbuLearnedRange<B: Backend> {
    range_proj: Projection<B>,
    fixup_proj: Projection<B>,
    range_temp: Param<Tensor<B, 1>>,
    sigma_spatial: Param<Tensor<B, 1>>,
    radius: usize,
    diameter: usize,
    key_dim: usize,
    upsample: Interpolate2d,
    patch: Tensor<B, 3>,
    unfold: Unfold4d,
}

impl<B: Backend> JbuLearnedRange<B> {
    pub fn new(
        device: &B::Device,
        config: &JbuLearnedRangeConfig,
    ) -> Self {
        let range_proj = ProjectionConfig::new(config.guidance_dim, config.key_dim)
            .init(device);

        let diameter = config.radius * 2 + 1;
        let fixup_output_dim = diameter * diameter;
        let fixup_proj = ProjectionConfig::new(config.guidance_dim + fixup_output_dim, fixup_output_dim)
            .init(device);

        let range_temp = Initializer::Zeros.init([1], device);
        let sigma_spatial = Initializer::Ones.init([1], device);

        let upsample = Interpolate2dConfig::new()
            .with_mode(InterpolateMode::Cubic)
            .with_output_size([config.height, config.width].into())
            .init();

        let dist_range = linspace(-1.0, 1.0, diameter, device);
        let xy = meshgrid([dist_range.clone(), dist_range.clone()]);
        let x = xy[0].clone();
        let y = xy[1].clone();
        let patch = Tensor::cat(
            vec![x.unsqueeze_dim(0), y.unsqueeze_dim(0)],
            0,
        );

        let unfold = Unfold4dConfig::new([diameter, diameter])
            .with_dilation([1, 1])
            .with_stride([1, 1])
            .with_padding([0, 0])
            .init();

        Self {
            range_proj,
            fixup_proj,
            range_temp,
            sigma_spatial,
            radius: config.radius,
            diameter,
            key_dim: config.key_dim,
            upsample,
            patch,
            unfold,
        }
    }

    fn get_range_kernel(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [B, _C, H, W] = x.shape().dims();

        let proj_x = self.range_proj.forward(x);
        let proj_x_padded = proj_x.clone().pad(
            (self.radius, self.radius, self.radius, self.radius),
            0.0.elem(),  // TODO: reflect pad
        );

        // TODO: unfold operator hits a size limit 2^16 when processing [4, 32, 302, 302] inputs
        println!("proj_x_padded: {:?}", proj_x_padded.shape());

        let queries = self.unfold.forward(proj_x_padded)
            .reshape([B, self.key_dim, self.diameter * self.diameter, H, W])
            .permute([0, 1, 3, 4, 2]);

        let pos_temp = self.range_temp.val().exp().clamp(1e-4, 1e4);

        let range_kernel_logits = einsum_bchwp_bchw_to_bphw(queries, proj_x)
            .mul_scalar(pos_temp.into_scalar());
        softmax(range_kernel_logits, 1)
    }

    fn get_spatial_kernel(&self) -> Tensor<B, 4> {
        let sigma_div = self.sigma_spatial
            .val()
            .powf_scalar(2.0)
            .mul_scalar(2.0);

        let spatial_kernel = (
            self.patch
                .clone()
                .neg()
                .sum_dim(0)
                .div_scalar(sigma_div.into_scalar())
            )
            .exp()
            .reshape([1, self.diameter * self.diameter, 1, 1]);

        spatial_kernel
    }

    pub fn forward(
        &self,
        source: Tensor<B, 4>,
        guidance: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [GB, _GC, GH, GW] = guidance.shape().dims();
        let [SB, _SC, _SH, _SW] = source.shape().dims();

        assert_eq!(GB, SB);

        let spatial_kernel = self.get_spatial_kernel();
        let range_kernel = self.get_range_kernel(guidance.clone());

        let combined_kernel = range_kernel * spatial_kernel;
        let kernel_sum = combined_kernel.clone()
            .sum_dim(1)
            .clamp_min(1e-7);

        let combined_kernel = combined_kernel.div(kernel_sum);

        let fixup_cat = Tensor::cat(
            vec![combined_kernel.clone(), guidance],
            1,
        );
        let fixup_term = self.fixup_proj.forward(fixup_cat).mul_scalar(0.1);

        let combined_kernel = combined_kernel + fixup_term;

        let combined_kernel = combined_kernel
            .permute([0, 2, 3, 1])
            .reshape([GB, GH, GW, self.diameter, self.diameter]);

        let hr_source = self.upsample.forward(source);
        let hr_source_padded = hr_source.pad(
            (self.radius, self.radius, self.radius, self.radius),
            0.0.elem(),
        );

        adaptive_conv(hr_source_padded, combined_kernel)
    }
}


#[derive(Config)]
pub struct JbuStackConfig {
    pub feat_dim: usize,
    pub width: usize,
    pub height: usize,
    pub feature_width: usize,
    pub feature_height: usize,
}

impl Default for JbuStackConfig {
    fn default() -> Self {
        let patch_size = 14;
        let input_size = 518;
        let feature_size = input_size / patch_size;

        Self::new(
            384,
            input_size,
            input_size,
            feature_size,
            feature_size,
        )
    }
}

impl JbuStackConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> JbuStack<B> {
        JbuStack::new(device, &self)
    }
}

#[derive(Module, Debug)]
pub struct JbuStackFixup<B: Backend> {
    conv: Conv2d<B>,
    dropout: Dropout,
}

impl<B: Backend> JbuStackFixup<B> {
    pub fn new(
        device: &B::Device,
        feat_dim: usize,
    ) -> Self {
        let conv = Conv2dConfig::new(
            [feat_dim, feat_dim],
            [1, 1],
        ).init(device);

        let dropout = DropoutConfig::new(0.2)
            .init();

        Self {
            conv,
            dropout,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let x = self.dropout.forward(x);
        self.conv.forward(x)
    }
}


#[derive(Module, Debug)]
pub struct JbuStack<B: Backend> {
    up1: JbuLearnedRange<B>,
    up2: JbuLearnedRange<B>,
    up3: JbuLearnedRange<B>,
    up4: JbuLearnedRange<B>,
    pool1: AdaptiveAvgPool2d,
    pool2: AdaptiveAvgPool2d,
    pool3: AdaptiveAvgPool2d,
    pool4: AdaptiveAvgPool2d,
    fixup_proj: JbuStackFixup<B>,
}

impl<B: Backend> JbuStack<B> {
    pub fn new(
        device: &B::Device,
        config: &JbuStackConfig,
    ) -> Self {
        let out1 = config.feature_height * 2;
        let out2 = config.feature_height * 4;
        let out3 = config.feature_height * 8;
        let out4 = config.feature_height * 16;

        let up1 = JbuLearnedRangeConfig::new(
            3,
            config.feat_dim,
            32,
            3,
            out1,
            out1,
        ).init(device);

        let up2 = JbuLearnedRangeConfig::new(
            3,
            config.feat_dim,
            32,
            3,
            out2,
            out2,
        ).init(device);

        let up3 = JbuLearnedRangeConfig::new(
            3,
            config.feat_dim,
            32,
            3,
            out3,
            out3,
        ).init(device);

        let up4 = JbuLearnedRangeConfig::new(
            3,
            config.feat_dim,
            32,
            3,
            out4,
            out4,
        ).init(device);

        let pool1 = AdaptiveAvgPool2dConfig::new(
            [out1, out1],
        ).init();

        let pool2 = AdaptiveAvgPool2dConfig::new(
            [out2, out2],
        ).init();

        let pool3 = AdaptiveAvgPool2dConfig::new(
            [out3, out3],
        ).init();

        let pool4 = AdaptiveAvgPool2dConfig::new(
            [out4, out4],
        ).init();

        let fixup_proj = JbuStackFixup::new(device, config.feat_dim);

        Self {
            up1,
            up2,
            up3,
            up4,
            pool1,
            pool2,
            pool3,
            pool4,
            fixup_proj,
        }
    }

    fn upsample(
        &self,
        source: Tensor<B, 4>,
        guidance: Tensor<B, 4>,
        up: &JbuLearnedRange<B>,
        pool: &AdaptiveAvgPool2d,
    ) -> Tensor<B, 4> {
        let small_guidance = pool.forward(guidance);
        up.forward(source, small_guidance)
    }

    pub fn forward(
        &self,
        source: Tensor<B, 4>,
        guidance: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let source = self.upsample(source, guidance.clone(), &self.up1, &self.pool1);
        let source = self.upsample(source, guidance.clone(), &self.up2, &self.pool2);
        let source = self.upsample(source, guidance.clone(), &self.up3, &self.pool3);
        let source = self.upsample(source, guidance, &self.up4, &self.pool4);

        self.fixup_proj.forward(source.clone()).mul_scalar(0.1) + source
    }
}
