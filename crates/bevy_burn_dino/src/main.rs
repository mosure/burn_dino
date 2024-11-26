use std::sync::{Arc, Mutex};

use bevy::{
    prelude::*,
    color::palettes::css::GOLD,
    diagnostic::{
        DiagnosticsStore,
        FrameTimeDiagnosticsPlugin,
    },
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{
            Extent3d,
            TextureDimension,
            TextureFormat,
        },
        settings::{
            RenderCreation,
            WgpuFeatures,
            WgpuSettings,
        },
        RenderPlugin,
    },
};
use bevy_args::{
    parse_args,
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
    ImageBuffer,
    Rgb,
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
    Resource,
    Clone,
    Debug,
    Serialize,
    Deserialize,
    Parser,
    Reflect,
)]
#[reflect(Resource)]
#[command(about = "burn_dino import", version, long_about = None)]
pub struct DinoImportConfig {
    #[arg(long, default_value = "true")]
    pub pca_only: bool,

    #[arg(long, default_value = "true")]
    pub press_esc_to_close: bool,

    #[arg(long, default_value = "true")]
    pub show_fps: bool,

    #[arg(long, default_value = "518")]
    pub inference_height: usize,

    #[arg(long, default_value = "518")]
    pub inference_width: usize,
}

impl Default for DinoImportConfig {
    fn default() -> Self {
        Self {
            pca_only: true,
            press_esc_to_close: true,
            show_fps: true,
            inference_height: 518,
            inference_width: 518,
        }
    }
}


static DINO_STATE_ENCODED: &[u8] = include_bytes!("../../../assets/models/dinov2.mpk");
static PCA_STATE_ENCODED: &[u8] = include_bytes!("../../../assets/models/person_pca.mpk");


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
        .resize_exact(config.image_size as u32, config.image_size as u32, image::imageops::FilterType::Triangle)
        .to_rgb32f();

    let samples = img.as_flat_samples();
    let floats: &[f32] = samples.as_slice();

    let input: Tensor<B, 1> = Tensor::from_floats(
        floats,
        device,
    );

    let input = input.reshape([1, config.image_size, config.image_size, config.input_channels])
        .permute([0, 3, 1, 2]);

    normalize(input, device)
}


fn to_image<B: Backend>(
    image: Tensor<B, 4>,
    upsample_height: u32,
    upsample_width: u32,
) -> RgbaImage {
    let height = image.shape().dims[1];
    let width = image.shape().dims[2];

    let image = image.to_data().to_vec::<f32>().unwrap();
    let image = ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(
        width as u32,
        height as u32,
        image,
    ).unwrap();

    DynamicImage::ImageRgb32F(image)
        .resize_exact(upsample_width, upsample_height, image::imageops::FilterType::Triangle)
        .to_rgba8()
}


fn process_frame(
    input: RgbImage,
    dino: Res<DinoModel<Wgpu>>,
    pca_transform: Res<PcaTransformModel<Wgpu>>,
    image_handle: &Handle<Image>,
    mut images: ResMut<Assets<Image>>,
) {
    let input_tensor: Tensor<Wgpu, 4> = preprocess_image(input, &dino.config, &dino.device);

    let model = dino.model.lock().unwrap();
    let dino_features = model.forward(input_tensor.clone(), None).x_norm_patchtokens;

    let batch = dino_features.shape().dims[0];
    let elements = dino_features.shape().dims[1];
    let embedding_dim = dino_features.shape().dims[2];
    let n_samples = batch * elements;
    let spatial_size = elements.isqrt();

    let x = dino_features.reshape([n_samples, embedding_dim]);

    let pca_transform = pca_transform.model.lock().unwrap();
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

    // TODO: share wgpu io between bevy/burn
    let existing_image = images.get_mut(image_handle).unwrap();
    existing_image.data = pca_features.into_raw();
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

#[derive(Resource)]
struct PcaTransformModel<B: Backend> {
    model: Arc::<Mutex::<PcaTransform<B>>>,
}


#[derive(Resource, Default)]
struct PcaFeatures {
    image: Handle<Image>,
}

#[cfg(feature = "native")]
fn process_frames(
    dino_model: Res<DinoModel<Wgpu>>,
    pca_transform: Res<PcaTransformModel<Wgpu>>,
    pca_features_handle: Res<PcaFeatures>,
    images: ResMut<Assets<Image>>,
) {
    let receiver = native::SAMPLE_RECEIVER.get().unwrap();
    let mut last_image = None;

    {
        let receiver = receiver.lock().unwrap();
        while let Ok(image) = receiver.try_recv() {
            last_image = Some(image);
        }
    }

    if let Some(image) = last_image {
        process_frame(
            image,
            dino_model,
            pca_transform,
            &pca_features_handle.image,
            images,
        );
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
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    ));

    commands.spawn(Node {
        display: Display::Grid,
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        grid_template_columns: RepeatedGridTrack::flex(1, 1.0),
        grid_template_rows: RepeatedGridTrack::flex(1, 1.0),
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

            builder.spawn(UiImage {
                image: pca_image.image.clone(),
                image_mode: NodeImageMode::Stretch,
                ..default()
            });
        });

    commands.spawn(Camera2d::default());
}

pub fn viewer_app() -> App {
    let args = parse_args::<DinoImportConfig>();

    let mut app = App::new();
    app.insert_resource(args.clone());

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        // fit_canvas_to_parent: true,
        canvas: Some("#bevy".to_string()),
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: true,
        title: args.name.clone(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    #[cfg(not(target_arch = "wasm32"))]
    let primary_window = Some(Window {
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: false,
        resolution: (1024.0, 1024.0).into(),
        title: "bevy_burn_dino".to_string(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    app.insert_resource(ClearColor(Color::srgba(0.0, 0.0, 0.0, 0.0)));


    let default_plugins = DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(RenderPlugin {
            render_creation: RenderCreation::Automatic(WgpuSettings {
                features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | WgpuFeatures::SHADER_F16,
                ..Default::default()
            }),
            ..Default::default()
        })
        .set(WindowPlugin {
            primary_window,
            ..default()
        });

    app.add_plugins(default_plugins);

    #[cfg(feature = "editor")]
    if args.editor {
        app.register_type::<BevyZeroverseConfig>();
        app.add_plugins(WorldInspectorPlugin::new());
    }

    // TODO: add viewer configs
    if args.press_esc_to_close {
        app.add_systems(Update, press_esc_close);
    }

    if args.show_fps {
        app.add_plugins(FrameTimeDiagnosticsPlugin);
        app.add_systems(Startup, fps_display_setup);
        app.add_systems(Update, fps_update_system);
    }

    app
}

fn press_esc_close(
    keys: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}

fn fps_display_setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    commands.spawn((
        Text("fps: ".to_string()),
        TextFont {
            font: asset_server.load("fonts/Caveat-Bold.ttf"),
            font_size: 60.0,
            ..Default::default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(5.0),
            left: Val::Px(15.0),
            ..default()
        },
        ZIndex(2),
    )).with_child((
        FpsText,
        TextColor(Color::Srgba(GOLD)),
        TextFont {
            font: asset_server.load("fonts/Caveat-Bold.ttf"),
            font_size: 60.0,
            ..Default::default()
        },
        TextSpan::default(),
    ));
}

#[derive(Component)]
struct FpsText;

fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut TextSpan, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                **text = format!("{value:.2}");
            }
        }
    }
}

fn run_app() {
    // TODO: move model load to startup/async task
    let device = Default::default();
    let config = DinoVisionTransformerConfig {
        ..DinoVisionTransformerConfig::vits(None, None)  // TODO: supply image size fron config
    };
    let dino = load_model::<Wgpu>(&config, &device);

    let pca_config = PcaTransformConfig::new(
        config.embedding_dimension,
        3,
    );
    let pca_transform = load_pca_model::<Wgpu>(&pca_config, &device);

    let mut app = viewer_app();

    app.init_resource::<PcaFeatures>();
    app.insert_resource(DinoModel {
        config,
        device: device,
        model: Arc::new(Mutex::new(dino)),
    });
    app.insert_resource(PcaTransformModel {
        model: Arc::new(Mutex::new(pca_transform)),
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
