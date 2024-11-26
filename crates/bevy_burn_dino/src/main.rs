use std::sync::{Arc, Mutex};

use bevy::{
    prelude::*,
    color::palettes::css::GOLD,
    diagnostic::{
        DiagnosticsStore,
        FrameTimeDiagnosticsPlugin,
    },
    ecs::{system::SystemState, world::CommandQueue},
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
    tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task},
};
use bevy_args::{
    parse_args,
    Deserialize,
    Parser,
    Serialize,
};
use burn::{
    prelude::*,
    backend::wgpu::{init_async, AutoGraphicsApi, Wgpu},
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

    #[arg(long, default_value = "false")]
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
            show_fps: false,  // TODO: display inference fps (UI fps is decoupled via async compute pool)
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


async fn to_image<B: Backend>(
    image: Tensor<B, 4>,
    upsample_height: u32,
    upsample_width: u32,
) -> RgbaImage {
    let height = image.shape().dims[1];
    let width = image.shape().dims[2];

    let image = image.to_data_async().await.to_vec::<f32>().unwrap();
    let image = ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(
        width as u32,
        height as u32,
        image,
    ).unwrap();

    DynamicImage::ImageRgb32F(image)
        .resize_exact(upsample_width, upsample_height, image::imageops::FilterType::Triangle)
        .to_rgba8()
}


// TODO: benchmark process_frame
async fn process_frame<B: Backend>(
    input: RgbImage,
    dino_config: DinoVisionTransformerConfig,
    dino_model: Arc<Mutex<DinoVisionTransformer<B>>>,
    pca_model: Arc<Mutex<PcaTransform<B>>>,
    device: B::Device,
) -> Vec<u8> {
    let input_tensor: Tensor<B, 4> = preprocess_image(input, &dino_config, &device);

    let dino_features = {
        let model = dino_model.lock().unwrap();
        model.forward(input_tensor.clone(), None).x_norm_patchtokens
    };

    let batch = dino_features.shape().dims[0];
    let elements = dino_features.shape().dims[1];
    let embedding_dim = dino_features.shape().dims[2];
    let n_samples = batch * elements;
    let spatial_size = elements.isqrt();

    let x = dino_features.reshape([n_samples, embedding_dim]);

    let mut pca_features = {
        let pca_transform = pca_model.lock().unwrap();
        pca_transform.forward(x.clone())
    };

    // pca min-max scaling
    for i in 0..3 {
        let slice = pca_features.clone().slice([0..n_samples, i..i+1]);
        let slice_min = slice.clone().min();
        let slice_max = slice.clone().max();
        let scaled = slice
            .sub(slice_min.clone().unsqueeze())
            .div((slice_max - slice_min).unsqueeze());

        pca_features = pca_features.slice_assign(
            [0..n_samples, i..i+1],
            scaled,
        );
    }

    let pca_features = pca_features.reshape([batch, spatial_size, spatial_size, 3]);

    let pca_features = to_image(
        pca_features,
        dino_config.image_size as u32,
        dino_config.image_size as u32,
    ).await;

    pca_features.into_raw()
}


#[cfg(feature = "native")]
mod native {
    use std::sync::{
        Arc,
        Mutex,
        mpsc::{
            self,
            Sender,
            SyncSender,
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
    pub static SAMPLE_SENDER: OnceCell<SyncSender<RgbImage>> = OnceCell::new();

    pub static APP_RUN_RECEIVER: OnceCell<Arc<Mutex<Receiver<()>>>> = OnceCell::new();
    pub static APP_RUN_SENDER: OnceCell<Sender<()>> = OnceCell::new();

    pub fn native_camera_thread() {
        let (
            sample_sender,
            sample_receiver,
        ) = mpsc::sync_channel(1);
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


#[cfg(feature = "web")]
mod web {
    use std::cell::RefCell;

    use image::{
        DynamicImage,
        RgbImage,
        RgbaImage,
    };
    use wasm_bindgen::prelude::*;

    thread_local! {
        pub static SAMPLE_RECEIVER: RefCell<Option<RgbImage>> = RefCell::new(None);
    }

    #[wasm_bindgen]
    pub fn frame_input(pixel_data: &[u8], width: u32, height: u32) {
        let rgba_image = RgbaImage::from_raw(width, height, pixel_data.to_vec())
            .expect("failed to create RgbImage");

        // TODO: perform video element -> burn's webgpu texture conversion directly
        // TODO: perform this conversion from tensors
        let dynamic_image = DynamicImage::ImageRgba8(rgba_image);
        let rgb_image: RgbImage = dynamic_image.to_rgb8();

        SAMPLE_RECEIVER.with(|receiver| {
            *receiver.borrow_mut() = Some(rgb_image);
        });
    }
}


#[derive(Resource)]
struct DinoModel<B: Backend> {
    config: DinoVisionTransformerConfig,
    device: B::Device,
    model: Arc<Mutex<DinoVisionTransformer<B>>>,
}

#[derive(Resource)]
struct PcaTransformModel<B: Backend> {
    model: Arc::<Mutex::<PcaTransform<B>>>,
}


#[derive(Resource, Default)]
struct PcaFeatures {
    image: Handle<Image>,
}


#[derive(Component)]
struct ProcessImage(Task<CommandQueue>);

#[cfg(feature = "native")]
fn receive_image() -> Option<RgbImage> {
    let receiver = native::SAMPLE_RECEIVER.get().unwrap();
    let mut last_image = None;

    {
        let receiver = receiver.lock().unwrap();
        while let Ok(image) = receiver.try_recv() {
            last_image = Some(image);
        }
    }

    last_image
}

#[cfg(feature = "web")]
fn receive_image() -> Option<RgbImage> {
    web::SAMPLE_RECEIVER.with(|receiver| {
        receiver.borrow_mut().take()
    })
}

fn process_frames(
    mut commands: Commands,
    dino_model: Res<DinoModel<Wgpu>>,
    pca_transform: Res<PcaTransformModel<Wgpu>>,
    pca_features_handle: Res<PcaFeatures>,
    active_tasks: Query<&ProcessImage>,
) {
    // TODO: move to config
    // TODO: fix multiple in flight deadlock
    let inference_max_in_flight = 1;
    if active_tasks.iter().count() >= inference_max_in_flight {
        return;
    }

    if let Some(image) = receive_image() {
        let thread_pool = AsyncComputeTaskPool::get();
        let entity = commands.spawn_empty().id();

        let device = dino_model.device.clone();
        let dino_config = dino_model.config.clone();
        let dino_model = dino_model.model.clone();
        let image_handle = pca_features_handle.image.clone();
        let pca_model = pca_transform.model.clone();

        let task = thread_pool.spawn(async move {
            let img_data = process_frame(
                image,
                dino_config,
                dino_model,
                pca_model,
                device,
            ).await;

            let mut command_queue = CommandQueue::default();
            command_queue.push(move |world: &mut World| {
                let mut system_state = SystemState::<
                    ResMut<Assets<Image>>,
                >::new(world);
                let mut images = system_state.get_mut(world);

                // TODO: share wgpu io between bevy/burn
                let existing_image = images.get_mut(&image_handle).unwrap();
                existing_image.data = img_data;

                world
                    .entity_mut(entity)
                    .remove::<ProcessImage>();
            });

            command_queue
        });

        commands.entity(entity).insert(ProcessImage(task));
    }
}

fn handle_tasks(mut commands: Commands, mut transform_tasks: Query<&mut ProcessImage>) {
    for mut task in &mut transform_tasks {
        if let Some(mut commands_queue) = block_on(future::poll_once(&mut task.0)) {
            commands.append(&mut commands_queue);
        }
    }
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
            builder.spawn(UiImage {
                image: pca_image.image.clone(),
                image_mode: NodeImageMode::Stretch,
                ..default()
            });
        });

    commands.spawn(Camera2d);
}

pub fn viewer_app() -> App {
    let args = parse_args::<DinoImportConfig>();

    let mut app = App::new();
    app.insert_resource(args.clone());

    let title = "bevy_burn_dino".to_string();

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        // fit_canvas_to_parent: true,
        canvas: Some("#bevy".to_string()),
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: true,
        title,

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
        title,

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

async fn run_app() {
    log("running app...");

    let device = Default::default();
    init_async::<AutoGraphicsApi>(&device, Default::default()).await;

    log("device created");

    let config = DinoVisionTransformerConfig {
        ..DinoVisionTransformerConfig::vits(None, None)  // TODO: supply image size fron config
    };
    let dino = load_model::<Wgpu>(&config, &device);

    log("dino model loaded");

    let pca_config = PcaTransformConfig::new(
        config.embedding_dimension,
        3,
    );
    let pca_transform = load_pca_model::<Wgpu>(&pca_config, &device);

    log("pca model loaded");

    let mut app = viewer_app();

    app.init_resource::<PcaFeatures>();
    app.insert_resource(DinoModel {
        config,
        device,
        model: Arc::new(Mutex::new(dino)),
    });
    app.insert_resource(PcaTransformModel {
        model: Arc::new(Mutex::new(pca_transform)),
    });

    app.add_systems(Startup, setup_ui);
    app.add_systems(
        Update,
        (
            handle_tasks,
            process_frames,
        ),
    );

    log("running bevy app...");

    app.run();
}


pub fn log(_msg: &str) {
    #[cfg(debug_assertions)]
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::log_1(&_msg.into());
    }
    #[cfg(debug_assertions)]
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("{}", _msg);
    }
}


fn main() {
    #[cfg(feature = "native")]
    {
        std::thread::spawn(native::native_camera_thread);
        futures::executor::block_on(run_app());
    }

    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(debug_assertions)]
        console_error_panic_hook::set_once();

        wasm_bindgen_futures::spawn_local(run_app());
    }
}
