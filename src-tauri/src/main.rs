// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;

use lazy_static::lazy_static;

use tauri::Manager;

use image::imageops::FilterType;
use ordered_float::OrderedFloat;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

#[derive(Clone, serde::Serialize)]
struct PredictionPayload {
    prediction: char,
    confidence: f32,
}

const CHARS: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const MODEL_DIR: &str = "model/";

lazy_static! {
    pub static ref GRAPH_SESH: (Graph, SavedModelBundle) = {
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, MODEL_DIR)
                .expect("cannot load model");

        (graph, bundle)
    };
}

/// original can be found at:
///
/// https://github.com/tensorflow/rust/blob/756920c8f18ea121d49fa1452c4604494c20ca49/examples/keras_single_input_saved_model.rs
fn run_prediction(inputs: Vec<u8>) -> Result<Vec<f32>, String> {
    let graph = &GRAPH_SESH.0;
    let bundle = &GRAPH_SESH.1;

    let inputs: Vec<f32> = inputs.iter().map(|b| *b as f32).collect();

    let input_tensor = Tensor::new(&[1, 32, 32, 1])
        .with_values(&inputs)
        .map_err(|e| e.to_string())?;

    let signature = bundle
        .meta_graph_def()
        .get_signature("serving_default")
        .map_err(|e| e.to_string())?;

    let input_info = signature
        .get_input("input_layer")
        .map_err(|e| e.to_string())?;
    let output_info = signature
        .get_output("output_0")
        .map_err(|e| e.to_string())?;

    let input_op = graph
        .operation_by_name_required(&input_info.name().name)
        .map_err(|e| e.to_string())?;
    let output_op = graph
        .operation_by_name_required(&output_info.name().name)
        .map_err(|e| e.to_string())?;

    let mut args = SessionRunArgs::new();
    args.add_feed(&input_op, 0, &input_tensor);
    let output_token = args.request_fetch(&output_op, 0);

    bundle.session.run(&mut args).map_err(|e| e.to_string())?;

    let output = args.fetch(output_token).map_err(|e| e.to_string())?;
    Ok(output.to_vec())
}

#[tauri::command]
async fn predict(
    handle: tauri::AppHandle,
    filename: &str,
    show_image: bool,
) -> Result<(char, f32), String> {
    let img = image::open(filename).map_err(|e| e.to_string())?;
    if show_image {
        const MSG: &str = "literally no reason to fail";
        _ = tauri::WindowBuilder::new(
            &handle,
            "input",
            tauri::WindowUrl::External(format!("file://{filename}").parse().expect(MSG)),
        )
        .title("Input Image")
        .center()
        .inner_size(img.width() as f64, img.height() as f64)
        .resizable(false)
        .build();
    }

    let img = img
        .resize_exact(32, 32, FilterType::Nearest)
        .into_luma8()
        .into_raw();

    let outputs = run_prediction(img)?;
    let (highest_idx, highest) = outputs
        .iter()
        .enumerate()
        .max_by_key(|(_, &n)| OrderedFloat::from(n))
        .expect("output tensor is never empty");

    const MSG: &str = "index is always within bounds";
    Ok((CHARS.chars().nth(highest_idx).expect(MSG), *highest))
}

#[tauri::command]
async fn verdict(
    handle: tauri::AppHandle,
    prediction: char,
    confidence: f32,
) -> Result<(), String> {
    let window = Arc::new(
        tauri::WindowBuilder::new(&handle, "verdict", tauri::WindowUrl::App("/verdict".into()))
            .title("Verdict")
            .focused(true)
            .inner_size(400_f64, 420_f64)
            .resizable(false)
            .build()
            .map_err(|e| e.to_string())?,
    );

    let msg = PredictionPayload {
        prediction,
        confidence,
    };

    Arc::clone(&window).once("ready", move |_| {
        _ = window.emit("prediction", msg);
    });

    Ok(())
}

#[tauri::command]
async fn close_pred_windows(handle: tauri::AppHandle) {
    let windows = handle.windows();

    if let Some(w) = windows.get("verdict") {
        _ = w.close();
    }

    if let Some(w) = windows.get("input") {
        _ = w.close();
    }
}

#[tauri::command]
fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            predict,
            verdict,
            close_pred_windows,
            get_version
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
