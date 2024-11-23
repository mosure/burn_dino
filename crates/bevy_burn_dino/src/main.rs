// TODO: mpsc channel to trigger inference
// TODO: javascript bindings for image input
use std::sync::{Arc, Mutex};

use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{
            Extent3d,
            TextureDescriptor,
            TextureDimension,
            TextureFormat,
            TextureUsages,
        },
    },
};
use bevy_args::{
    Deserialize,
    Parser,
    Serialize,
};
use burn::{
    prelude::*,
    backend::wgpu::Wgpu,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};
use image::{
    DynamicImage,
    RgbImage,
    RgbaImage,
};

use burn_dino::model::{
    dino::{
        DinoVisionTransformer,
        DinoVisionTransformerConfig,
    },
    pca::{
        PcaTransform, PcaTransformConfig
    },
};


// TODO: support multiple PCA heads with args/inspector switching


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
    #[arg(long, default_value = "true")]
    pub pca_only: bool,
}



static DINO_STATE_ENCODED: &[u8] = include_bytes!("../../../assets/models/dinov2.mpk");
static PCA_STATE_ENCODED: &[u8] = include_bytes!("../../../assets/models/pca.mpk");


fn load_model<B: Backend>(
    config: &DinoVisionTransformerConfig,
    device: &B::Device,
) -> DinoVisionTransformer<B> {
    let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
        .load(DINO_STATE_ENCODED.to_vec(), &Default::default())
        .expect("failed to decode state");

    let model= config.init(device);
    model.load_record(record)
}

fn load_pca_model<B: Backend>(
    config: &PcaTransformConfig,
    device: &B::Device,
) -> PcaTransform<B> {
    let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
        .load(PCA_STATE_ENCODED.to_vec(), &Default::default())
        .expect("failed to decode state");

    let model= config.init(device);
    model.load_record(record)
}


fn normalize<B: Backend>(
    input: Tensor<B, 4>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mean: Tensor<B, 1> = Tensor::from_floats([0.485, 0.456, 0.406], device);
    let std: Tensor<B, 1> = Tensor::from_floats([0.229, 0.224, 0.225], device);

    input
        .permute([0, 2, 3, 1])
        .sub(mean.unsqueeze())
        .div(std.unsqueeze())
        .permute([0, 3, 1, 2])
}

fn preprocess_image<B: Backend>(
    image: RgbImage,
    config: &DinoVisionTransformerConfig,
    device: &B::Device,
) -> Tensor<B, 4> {
    let img = DynamicImage::ImageRgb8(image)
        .resize_exact(config.image_size as u32, config.image_size as u32, image::imageops::FilterType::Lanczos3);

    let img = match img {
        DynamicImage::ImageRgb8(img) => img,
        _ => img.to_rgb8(),
    };

    let img_data: Vec<f32> = img
        .pixels()
        .flat_map(|p| p.0.iter().map(|&c| c as f32 / 255.0))
        .collect();

    let input: Tensor<B, 1> = Tensor::from_floats(
        img_data.as_slice(),
        device,
    );

    let input = input.reshape([1, config.image_size, config.image_size, config.input_channels])
        .permute([0, 3, 1, 2]);

    normalize(input, device)
}


fn to_image<B: Backend>(
    images: Tensor<B, 4>,
    upsample_height: u32,
    upsample_width: u32,
) -> RgbaImage {
    let height = images.shape().dims[1];
    let width = images.shape().dims[2];
    let channels = images.shape().dims[3];

    let image_size = height * width * channels;

    let images = images.clamp(0.0, 1.0).mul_scalar(255.0);
    let images = images.to_data().to_vec::<f32>().unwrap();

    let image_slice = &images[0..image_size];
    let image_slice_u8: Vec<u8> = image_slice.iter().map(|&v| v as u8).collect();

    let img = RgbImage::from_raw(
        width as u32,
        height as u32,
        image_slice_u8,
    ).unwrap();

    let img = DynamicImage::ImageRgb8(img)
        .resize_exact(upsample_width, upsample_height, image::imageops::FilterType::Lanczos3);

    img.to_rgba8()
}


fn process_frame(
    input: RgbImage,
    dino: Res<DinoModel<Wgpu>>,
) -> Image {
    let input_tensor: Tensor<Wgpu, 4> = preprocess_image(input, &dino.config, &dino.device);
    let model = dino.model.lock().unwrap();
    let dino_features = model.forward(input_tensor.clone(), None).x_norm_patchtokens;

    let batch = dino_features.shape().dims[0];
    let elements = dino_features.shape().dims[1];
    let embedding_dim = dino_features.shape().dims[2];
    let n_samples = batch * elements;
    let spatial_size = elements.isqrt();

    // TODO: pad x to expected pca input size
    let x = dino_features.reshape([n_samples, embedding_dim]);

    let pca_config = PcaTransformConfig::new(
        batch,
        embedding_dim,
        3,
    );
    let pca_transform = load_pca_model(&pca_config, &dino.device);
    let mut pca_features = pca_transform.forward(x.clone());

    // pca min-max scaling
    for i in 0..3 {
        let slice = pca_features.clone().slice([0..n_samples, i..i+1]);
        let slice_min = slice.clone().min().into_scalar();
        let slice_max = slice.clone().max().into_scalar();
        let scaled = slice.sub_scalar(slice_min).div_scalar(slice_max - slice_min);

        pca_features = pca_features.slice_assign(
            [0..n_samples, i..i+1],
            scaled,
        );
    }

    let pca_features = pca_features.reshape([batch, spatial_size, spatial_size, 3]);
    let pca_features = to_image(
        pca_features,
        dino.config.image_size as u32,
        dino.config.image_size as u32,
    );

    let size = Extent3d {
        width: dino.config.image_size as u32,
        height: dino.config.image_size as u32,
        depth_or_array_layers: 1,
    };

    Image {
        data: pca_features.to_vec(),
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::COPY_SRC
                 | TextureUsages::COPY_DST
                 | TextureUsages::TEXTURE_BINDING
                 | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        },
        ..Default::default()
    }
}


#[cfg(feature = "native")]
mod native {
    use std::sync::{
        Arc,
        Mutex,
        mpsc::{
            self,
            Sender,
            Receiver,
            TryRecvError,
        },
    };

    use image::RgbImage;
    use nokhwa::{
        nokhwa_initialize,
        query,
        CallbackCamera,
        pixel_format::RgbFormat,
        utils::{
            ApiBackend,
            RequestedFormat,
            RequestedFormatType,
        },
    };
    use once_cell::sync::OnceCell;

    pub static SAMPLE_RECEIVER: OnceCell<Arc<Mutex<Receiver<RgbImage>>>> = OnceCell::new();
    pub static SAMPLE_SENDER: OnceCell<Sender<RgbImage>> = OnceCell::new();

    pub static APP_RUN_RECEIVER: OnceCell<Arc<Mutex<Receiver<()>>>> = OnceCell::new();
    pub static APP_RUN_SENDER: OnceCell<Sender<()>> = OnceCell::new();

    pub fn native_camera_thread() {
        let (
            sample_sender,
            sample_receiver,
        ) = mpsc::channel();
        SAMPLE_RECEIVER.set(Arc::new(Mutex::new(sample_receiver))).unwrap();
        SAMPLE_SENDER.set(sample_sender).unwrap();

        let (
            app_run_sender,
            app_run_receiver,
        ) = mpsc::channel();
        APP_RUN_RECEIVER.set(Arc::new(Mutex::new(app_run_receiver))).unwrap();
        APP_RUN_SENDER.set(app_run_sender).unwrap();

        nokhwa_initialize(|granted| {
            if !granted {
                panic!("failed to initialize camera");
            }
        });

        let devices = query(ApiBackend::Auto).unwrap();
        let index = devices.first().unwrap().index();

        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
        let mut camera = CallbackCamera::new(index.clone(), format, |buffer| {
            let image = buffer.decode_image::<RgbFormat>().unwrap();
            let sender = SAMPLE_SENDER.get().unwrap();
            sender.send(image).unwrap();
        }).unwrap();

        camera.open_stream().unwrap();

        loop {
            camera.poll_frame().unwrap();

            let receiver = APP_RUN_RECEIVER.get().unwrap();
            match receiver.lock().unwrap().try_recv() {
                Ok(_) => break,
                Err(TryRecvError::Empty) => continue,
                Err(TryRecvError::Disconnected) => break,
            };
        }

        camera.stop_stream().unwrap();
    }
}


#[derive(Resource)]
struct DinoModel<B: Backend> {
    config: DinoVisionTransformerConfig,
    device: B::Device,
    model: Arc::<Mutex::<DinoVisionTransformer<B>>>,
}

#[derive(Resource, Default)]
struct PcaFeatures {
    image: Handle<Image>,
}

#[cfg(feature = "native")]
fn process_frames(
    dino_model: Res<DinoModel<Wgpu>>,
    pca_features_handle: Res<PcaFeatures>,
    mut images: ResMut<Assets<Image>>,
) {
    let receiver = native::SAMPLE_RECEIVER.get().unwrap();
    let receiver = receiver.lock().unwrap();

    if receiver.try_recv().is_ok() {
        // TODO: share wgpu io between bevy/burn
        let image = receiver.recv().unwrap();
        let image = process_frame(image, dino_model);

        images.insert(&pca_features_handle.image, image);
    }
}

// TODO: web-sys ffi
#[cfg(feature = "web")]
fn frame_input() {

}

fn setup_ui(
    mut commands: Commands,
    dino: Res<DinoModel<Wgpu>>,
    mut pca_image: ResMut<PcaFeatures>,
    mut images: ResMut<Assets<Image>>,
) {
    pca_image.image = images.add(Image::new_fill(
        Extent3d {
            width: dino.config.image_size as u32,
            height: dino.config.image_size as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    ));

    commands.spawn(NodeBundle {
        style: Style {
            display: Display::Grid,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            grid_template_columns: RepeatedGridTrack::flex(1, 1.0),
            grid_template_rows: RepeatedGridTrack::flex(1, 1.0),
            ..default()
        },
        background_color: BackgroundColor(Color::BLACK),
        ..default()
    })
        .with_children(|builder| {
            // TODO: view input
            // builder.spawn(ImageBundle {
            //     style: Style {
            //         width: Val::Percent(100.0),
            //         height: Val::Percent(100.0),
            //         ..default()
            //     },
            //     image: UiImage {
            //         texture: foreground,
            //         ..default()
            //     },
            //     ..default()
            // });

            builder.spawn(ImageBundle {
                style: Style {
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    ..default()
                },
                image: UiImage {
                    texture: pca_image.image.clone(),
                    ..default()
                },
                ..default()
            });
        });
}

fn run_app() {
    // TODO: move model load to startup/async task
    let device = Default::default();
    let config = DinoVisionTransformerConfig {
        ..DinoVisionTransformerConfig::vits(None, None)
    };
    let dino = load_model::<Wgpu>(&config, &device);

    let mut app = App::new();

    app.init_resource::<PcaFeatures>();
    app.insert_resource(DinoModel {
        config,
        device,
        model: Arc::new(Mutex::new(dino)),
    });

    app.add_systems(Startup, setup_ui);
    app.add_systems(Update, process_frames);

    app.run();
}


fn main() {
    #[cfg(feature = "native")]
    {
        std::thread::spawn(native::native_camera_thread);
    }

    run_app();
}
