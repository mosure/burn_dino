#[cfg(feature = "native")]
pub mod camera {
    use std::sync::{
        mpsc::{self, Receiver, Sender, SyncSender, TryRecvError},
        Arc, Mutex,
    };

    use image::RgbImage;
    use nokhwa::{
        nokhwa_initialize,
        pixel_format::RgbFormat,
        query,
        utils::{ApiBackend, RequestedFormat, RequestedFormatType},
        CallbackCamera,
    };
    use once_cell::sync::OnceCell;

    pub static SAMPLE_RECEIVER: OnceCell<Arc<Mutex<Receiver<RgbImage>>>> = OnceCell::new();
    pub static SAMPLE_SENDER: OnceCell<SyncSender<RgbImage>> = OnceCell::new();

    pub static APP_RUN_RECEIVER: OnceCell<Arc<Mutex<Receiver<()>>>> = OnceCell::new();
    pub static APP_RUN_SENDER: OnceCell<Sender<()>> = OnceCell::new();

    pub fn native_camera_thread() {
        let (sample_sender, sample_receiver) = mpsc::sync_channel(1);
        SAMPLE_RECEIVER
            .set(Arc::new(Mutex::new(sample_receiver)))
            .unwrap();
        SAMPLE_SENDER.set(sample_sender).unwrap();

        let (app_run_sender, app_run_receiver) = mpsc::channel();
        APP_RUN_RECEIVER
            .set(Arc::new(Mutex::new(app_run_receiver)))
            .unwrap();
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
        })
        .unwrap();

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

    pub fn receive_image() -> Option<RgbImage> {
        let receiver = SAMPLE_RECEIVER.get().unwrap();
        let mut last_image = None;

        {
            let receiver = receiver.lock().unwrap();
            while let Ok(image) = receiver.try_recv() {
                last_image = Some(image);
            }
        }

        last_image
    }
}

#[cfg(feature = "web")]
pub mod camera {
    use std::cell::RefCell;

    use image::{DynamicImage, RgbImage, RgbaImage};
    use wasm_bindgen::prelude::*;

    thread_local! {
        pub static SAMPLE_RECEIVER: RefCell<Option<RgbImage>> = RefCell::new(None);
    }

    #[wasm_bindgen]
    pub fn frame_input(pixel_data: &[u8], width: u32, height: u32) {
        let rgba_image = RgbaImage::from_raw(width, height, pixel_data.to_vec())
            .expect("failed to create RgbImage");

        // TODO: perform video element -> burn's webgpu texture conversion directly
        let dynamic_image = DynamicImage::ImageRgba8(rgba_image);
        let rgb_image: RgbImage = dynamic_image.to_rgb8();

        SAMPLE_RECEIVER.with(|receiver| {
            *receiver.borrow_mut() = Some(rgb_image);
        });
    }

    pub fn receive_image() -> Option<RgbImage> {
        SAMPLE_RECEIVER.with(|receiver| receiver.borrow_mut().take())
    }
}
