#![cfg_attr(not(feature = "backend_wgpu"), allow(dead_code))]

#[cfg(feature = "backend_wgpu")]
use burn::{backend::wgpu::Wgpu, prelude::*};
#[cfg(feature = "backend_wgpu")]
use burn_dino::model::dino::DinoVisionTransformerConfig;
#[cfg(feature = "backend_wgpu")]
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

#[cfg(feature = "backend_wgpu")]
criterion_group! {
    name = ladon_burn_benchmarks;
    config = Criterion::default().sample_size(500);
    targets = inference_benchmark,
}
#[cfg(feature = "backend_wgpu")]
criterion_main!(ladon_burn_benchmarks);

#[cfg(feature = "backend_wgpu")]
fn inference_benchmark(c: &mut Criterion) {
    let configs = [
        (DinoVisionTransformerConfig::vits(None, None), "vits"),
        (DinoVisionTransformerConfig::vitb(None, None), "vitb"),
        (DinoVisionTransformerConfig::vitl(None, None), "vitl"),
        // (DinoVisionTransformerConfig::vitg(None, None), "vitg"),
    ];

    let mut group = c.benchmark_group("burn_dinov2_inference");
    for (config, name) in configs.iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("vit", name), &config, |b, &config| {
            let device = Default::default();
            let model = config.init(&device);
            let input: Tensor<Wgpu, 4> = Tensor::zeros(
                [
                    1,
                    config.input_channels,
                    config.image_size,
                    config.image_size,
                ],
                &device,
            );

            b.iter(|| {
                model
                    .forward(input.clone(), None)
                    .x_norm_patchtokens
                    .to_data()
            });
        });
    }
}

#[cfg(not(feature = "backend_wgpu"))]
fn main() {
    eprintln!(
        "dinov2 benchmark requires `--features backend_wgpu` (fusion path). \
         Re-run with `cargo bench --features backend_wgpu`."
    );
}
