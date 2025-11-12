#![recursion_limit = "256"]

use std::sync::{Arc, Mutex};

use bevy::asset::{AssetEvent, RenderAssetUsages};
use bevy::{
    color::palettes::css::GOLD,
    diagnostic::{
        Diagnostic, DiagnosticPath, Diagnostics, DiagnosticsStore, FrameTimeDiagnosticsPlugin,
        RegisterDiagnostic,
    },
    ecs::world::CommandQueue,
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
        RenderPlugin,
    },
    tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task},
    ui::widget::ImageNode,
    window::WindowResolution,
};
use bevy_args::{parse_args, Deserialize, Parser, Serialize, ValueEnum};
use bevy_burn::{BevyBurnBridgePlugin, BevyBurnHandle, BindingDirection, TransferKind};
use bevy_webcam::{BevyWebcamPlugin, WebcamStream};
use burn::{
    backend::wgpu::graphics::AutoGraphicsApi,
    backend::wgpu::{init_setup_async, Wgpu},
    prelude::*,
};
use image::RgbImage;

use burn_dino::model::{
    dino::{DinoVisionTransformer, DinoVisionTransformerConfig},
    pca::{PcaTransform, PcaTransformConfig},
};

use bevy_burn_dino::process_frame;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize, Reflect, ValueEnum)]
pub enum PcaType {
    Adaptive, // TODO: window adaptive pca
    #[default]
    Face,
    Person,
}

impl PcaType {
    #[allow(dead_code)]
    const fn pca_weights_mpk(&self) -> &'static str {
        match self {
            PcaType::Adaptive => "adaptive_pca.mpk",
            PcaType::Face => "face_pca.mpk",
            PcaType::Person => "person_pca.mpk",
        }
    }
}

#[derive(Resource, Clone, Debug, Serialize, Deserialize, Parser, Reflect)]
#[reflect(Resource)]
#[command(about = "bevy_burn_dino", version, long_about = None)]
pub struct BevyBurnDinoConfig {
    #[arg(long, default_value = "true")]
    pub press_esc_to_close: bool,

    #[arg(long, default_value = "true")]
    pub show_fps: bool,

    #[arg(long, default_value = "518")]
    pub inference_height: usize,

    #[arg(long, default_value = "518")]
    pub inference_width: usize,

    #[arg(long, value_enum, default_value_t = PcaType::Face)]
    pub pca_type: PcaType,
}

impl Default for BevyBurnDinoConfig {
    fn default() -> Self {
        Self {
            press_esc_to_close: true,
            show_fps: true, // TODO: display inference fps (UI fps is decoupled via async compute pool)
            inference_height: 512,
            inference_width: 512,
            pca_type: PcaType::default(),
        }
    }
}

fn sanitize_inference_dim(value: usize) -> usize {
    let clamped = value.clamp(64, 2048);
    let remainder = clamped % 32;
    if remainder == 0 {
        clamped
    } else {
        clamped + (32 - remainder)
    }
}

#[cfg(feature = "native")]
mod io {
    use super::PcaType;
    use burn::{
        prelude::*,
        record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    };
    use burn_dino::model::{
        dino::{DinoVisionTransformer, DinoVisionTransformerConfig},
        pca::{PcaTransform, PcaTransformConfig},
    };

    static DINO_STATE_ENCODED: &[u8] = include_bytes!("../../../assets/models/dinov2.mpk");
    static FACE_PCA_STATE_ENCODED: &[u8] = include_bytes!("../../../assets/models/face_pca.mpk");
    static PERSON_PCA_STATE_ENCODED: &[u8] =
        include_bytes!("../../../assets/models/person_pca.mpk");

    pub async fn load_model<B: Backend>(
        config: &DinoVisionTransformerConfig,
        device: &B::Device,
    ) -> DinoVisionTransformer<B> {
        let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
            .load(DINO_STATE_ENCODED.to_vec(), &Default::default())
            .expect("failed to decode state");

        let model = config.init(device);
        model.load_record(record)
    }

    pub async fn load_pca_model<B: Backend>(
        config: &PcaTransformConfig,
        pca_type: PcaType,
        device: &B::Device,
    ) -> PcaTransform<B> {
        let data = match pca_type {
            PcaType::Adaptive => unimplemented!(),
            PcaType::Face => FACE_PCA_STATE_ENCODED,
            PcaType::Person => PERSON_PCA_STATE_ENCODED,
        };

        let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
            .load(data.to_vec(), &Default::default())
            .expect("failed to decode state");

        let model = config.init(device);
        model.load_record(record)
    }
}

#[cfg(feature = "web")]
mod io {
    use burn::{
        prelude::*,
        record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    };
    use burn_dino::model::{
        dino::{DinoVisionTransformer, DinoVisionTransformerConfig},
        pca::{PcaTransform, PcaTransformConfig},
    };
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{window, Request, RequestInit, RequestMode, Response};

    pub async fn load_model<B: Backend>(
        config: &DinoVisionTransformerConfig,
        device: &B::Device,
    ) -> DinoVisionTransformer<B> {
        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init("./assets/models/dinov2.mpk", &opts).unwrap();

        let window = window().unwrap();
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await;
        let resp: Response = resp_value.unwrap().dyn_into().unwrap();

        let array_buffer = JsFuture::from(
            resp.array_buffer()
                .expect("failed to download model weights"),
        )
        .await
        .unwrap();
        let uint8_array = Uint8Array::new(&array_buffer);

        let mut data = vec![0; uint8_array.length() as usize];
        uint8_array.copy_to(&mut data[..]);

        let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
            .load(data, &Default::default())
            .expect("failed to decode state");

        let model = config.init(device);
        model.load_record(record)
    }

    pub async fn load_pca_model<B: Backend>(
        config: &PcaTransformConfig,
        pca_type: super::PcaType,
        device: &B::Device,
    ) -> PcaTransform<B> {
        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(
            &format!("./assets/models/{}", pca_type.pca_weights_mpk()),
            &opts,
        )
        .unwrap();

        let window = window().unwrap();
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await;
        let resp: Response = resp_value.unwrap().dyn_into().unwrap();

        let array_buffer =
            JsFuture::from(resp.array_buffer().expect("failed to download pca weights"))
                .await
                .unwrap();
        let uint8_array = Uint8Array::new(&array_buffer);

        let mut data = vec![0; uint8_array.length() as usize];
        uint8_array.copy_to(&mut data[..]);

        let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
            .load(data, &Default::default())
            .expect("failed to decode state");

        let model = config.init(device);
        model.load_record(record)
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
    model: Arc<Mutex<PcaTransform<B>>>,
}

#[derive(Resource, Default)]
struct PcaFeatures {
    image: Handle<Image>,
    entity: Option<Entity>,
}

#[derive(Component)]
struct ProcessImage(Task<CommandQueue>);

fn process_frames(
    mut commands: Commands,
    dino_model: Res<DinoModel<Wgpu>>,
    pca_transform: Res<PcaTransformModel<Wgpu>>,
    pca_features_handle: Res<PcaFeatures>,
    active_tasks: Query<&ProcessImage>,
    stream: Res<WebcamStream>,
    images: Res<Assets<Image>>,
    mut webcam_events: MessageReader<AssetEvent<Image>>,
) {
    let Some(image_entity) = pca_features_handle.entity else {
        return;
    };

    if stream.frame == Handle::default() {
        return;
    }

    let frame_id = stream.frame.id();
    let mut frame_changed = false;
    for event in webcam_events.read() {
        if event.is_added(frame_id) || event.is_modified(frame_id) {
            frame_changed = true;
        } else if event.is_removed(frame_id) {
            warn!("webcam frame handle removed, waiting for it to be re-created");
        }
    }

    if !frame_changed {
        return;
    }

    // TODO: move to config
    // TODO: fix multiple in flight deadlock
    let inference_max_in_flight = 1;
    if active_tasks.iter().count() >= inference_max_in_flight {
        return;
    }

    let Some(rgb_image) = copy_webcam_frame(&stream.frame, &images) else {
        return;
    };

    let thread_pool = AsyncComputeTaskPool::get();
    let task_entity = commands.spawn_empty().id();

    let device = dino_model.device.clone();
    let dino_config = dino_model.config.clone();
    let dino_model = dino_model.model.clone();
    let pca_model = pca_transform.model.clone();
    let target_entity = image_entity;

    let task = thread_pool.spawn({
        let device_clone = device.clone();
        async move {
            let tensor =
                process_frame(rgb_image, dino_config, dino_model, pca_model, device_clone).await;

            let mut command_queue = CommandQueue::default();
            command_queue.push(move |world: &mut World| {
                if let Ok(mut image_entity_mut) = world.get_entity_mut(target_entity) {
                    if let Some(mut handle) = image_entity_mut.get_mut::<BevyBurnHandle<Wgpu>>() {
                        handle.tensor = tensor.clone();
                        handle.upload = true;
                    }
                }

                if let Ok(mut tracker) = world.get_entity_mut(task_entity) {
                    tracker.remove::<ProcessImage>();
                    tracker.despawn();
                }
            });

            command_queue
        }
    });

    commands.entity(task_entity).insert(ProcessImage(task));
}

fn copy_webcam_frame(handle: &Handle<Image>, images: &Assets<Image>) -> Option<RgbImage> {
    let Some(image) = images.get(handle) else {
        warn!("webcam texture handle is missing from the asset storage");
        return None;
    };

    let Some(pixels) = image.data.as_ref() else {
        warn!("webcam frame texture is missing CPU-side pixel data");
        return None;
    };

    if pixels.len() % 4 != 0 {
        warn!("webcam frame buffer is not aligned to 4 bytes per pixel");
        return None;
    }

    let format = image.texture_descriptor.format;
    if format != TextureFormat::Rgba8Unorm && format != TextureFormat::Rgba8UnormSrgb {
        warn!(
            "unsupported webcam texture format {:?}, expected RGBA8",
            format
        );
        return None;
    }

    let width = image.texture_descriptor.size.width;
    let height = image.texture_descriptor.size.height;
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for chunk in pixels.chunks_exact(4) {
        rgb.extend_from_slice(&chunk[..3]);
    }

    RgbImage::from_vec(width, height, rgb)
}

fn handle_tasks(
    mut commands: Commands,
    mut diagnostics: Diagnostics,
    mut last_frame: Local<Time<Real>>,
    mut transform_tasks: Query<&mut ProcessImage>,
) {
    for mut task in &mut transform_tasks {
        if let Some(mut commands_queue) = block_on(future::poll_once(&mut task.0)) {
            if let Some(last_instant) = last_frame.last_update() {
                let delta_seconds = last_instant.elapsed().as_secs_f64();
                if delta_seconds > 0.0 {
                    diagnostics.add_measurement(&INFERENCE_FPS, || 1.0 / delta_seconds);
                }
            }
            last_frame.update();

            commands.append(&mut commands_queue);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn pca_image_setup() -> (&'static [u8], TextureFormat, TransferKind, TextureUsages) {
    (
        &[0u8; 4],
        TextureFormat::Rgba8UnormSrgb,
        TransferKind::Cpu,
        TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn pca_image_setup() -> (&'static [u8], TextureFormat, TransferKind, TextureUsages) {
    (
        &[0u8; 16],
        TextureFormat::Rgba32Float,
        TransferKind::Gpu,
        TextureUsages::COPY_SRC
            | TextureUsages::COPY_DST
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING,
    )
}

fn setup_ui(
    mut commands: Commands,
    dino: Res<DinoModel<Wgpu>>,
    mut pca_image: ResMut<PcaFeatures>,
    mut images: ResMut<Assets<Image>>,
) {
    let size = Extent3d {
        width: dino.config.image_size as u32,
        height: dino.config.image_size as u32,
        depth_or_array_layers: 1,
    };
    let (fill_bytes, texture_format, transfer_kind, texture_usage) = pca_image_setup();
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        fill_bytes,
        texture_format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= texture_usage;
    pca_image.image = images.add(image);

    let mut image_entity = None;
    commands
        .spawn(Node {
            display: Display::Grid,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            grid_template_columns: RepeatedGridTrack::flex(1, 1.0),
            grid_template_rows: RepeatedGridTrack::flex(1, 1.0),
            ..default()
        })
        .with_children(|builder| {
            let entity = builder
                .spawn((
                    ImageNode::new(pca_image.image.clone()).with_mode(NodeImageMode::Stretch),
                    BevyBurnHandle::<Wgpu> {
                        bevy_image: pca_image.image.clone(),
                        tensor: Tensor::<Wgpu, 3>::zeros(
                            [dino.config.image_size, dino.config.image_size, 4],
                            &dino.device,
                        ),
                        upload: true,
                        direction: BindingDirection::BurnToBevy,
                        xfer: transfer_kind,
                    },
                ))
                .id();
            image_entity = Some(entity);
        });
    pca_image.entity = image_entity;

    commands.spawn(Camera2d);
}

pub fn viewer_app(args: BevyBurnDinoConfig) -> App {
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
        resolution: WindowResolution::new(1024, 1024),
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
                // Request only cross-adapter-safe features. Some drivers do not expose SHADER_F16.
                features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            }),
            ..Default::default()
        })
        .set(WindowPlugin {
            primary_window,
            ..default()
        });

    app.add_plugins(default_plugins);
    app.add_plugins(BevyWebcamPlugin::default());
    app.add_plugins(BevyBurnBridgePlugin::<Wgpu>::default());

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
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        app.register_diagnostic(Diagnostic::new(INFERENCE_FPS));
        app.add_systems(Startup, fps_display_setup);
        app.add_systems(Update, fps_update_system);
    }

    app
}

fn press_esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

const INFERENCE_FPS: DiagnosticPath = DiagnosticPath::const_new("inference_fps");

fn fps_display_setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn((
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
        ))
        .with_child((
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
        if let Some(fps) = diagnostics.get(&INFERENCE_FPS) {
            if let Some(value) = fps.smoothed() {
                **text = format!("{value:.2}");
            }
        }
    }
}

async fn run_app() {
    log("running app...");

    let mut args = parse_args::<BevyBurnDinoConfig>();
    let sanitized_height = sanitize_inference_dim(args.inference_height);
    let sanitized_width = sanitize_inference_dim(args.inference_width);
    let image_size = sanitized_height.min(sanitized_width);
    args.inference_height = image_size;
    args.inference_width = image_size;
    log(&format!("{:?}", args));

    let device = Default::default();
    init_setup_async::<AutoGraphicsApi>(&device, Default::default()).await;

    log("device created");

    let config = DinoVisionTransformerConfig {
        ..DinoVisionTransformerConfig::vits(Some(image_size), None)
    };
    let dino = io::load_model::<Wgpu>(&config, &device).await;

    log("dino model loaded");

    // TODO: support adaptive PCA
    let pca_config = PcaTransformConfig::new(config.embedding_dimension, 3);
    let pca_transform =
        io::load_pca_model::<Wgpu>(&pca_config, args.pca_type.clone(), &device).await;

    log("pca model loaded");

    let mut app = viewer_app(args);

    app.init_resource::<PcaFeatures>();
    app.insert_resource(DinoModel {
        config,
        device: device.clone(),
        model: Arc::new(Mutex::new(dino)),
    });
    app.insert_resource(PcaTransformModel {
        model: Arc::new(Mutex::new(pca_transform)),
    });

    app.add_systems(Startup, setup_ui);
    app.add_systems(Update, (handle_tasks, process_frames));

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
        futures::executor::block_on(run_app());
    }

    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(debug_assertions)]
        console_error_panic_hook::set_once();

        wasm_bindgen_futures::spawn_local(run_app());
    }
}
