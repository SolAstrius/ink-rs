use ink_inference::{npy, spec, Backend, Model};
use std::path::PathBuf;
use std::time::Instant;

fn quoted(s: &str) -> String {
    let mut output = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch if ch < ' ' => output.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => output.push(ch),
        }
    }
    output.push('"');
    output
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut model_path: Option<PathBuf> = None;
    let mut feature_path: Option<PathBuf> = None;
    let mut spec_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut backend = Backend::Auto;
    let mut iterations = 3;
    let mut threads = 1;
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("ink-infer --model MODEL.tflite --features FEATURES.npy [--spec SPEC.recospec] [--out LOGITS.npy] [--backend auto|scalar|neon] [--threads 1|2] [--iterations N]");
            return Ok(());
        }
        let value = args.next().ok_or("Missing option value")?;
        match arg.as_str() {
            "--model" => model_path = Some(value.into()),
            "--features" => feature_path = Some(value.into()),
            "--spec" => spec_path = Some(value.into()),
            "--out" => output_path = Some(value.into()),
            "--iterations" => iterations = value.parse::<usize>()?,
            "--threads" => threads = value.parse::<usize>()?,
            "--backend" => {
                backend = match value.as_str() {
                    "auto" => Backend::Auto,
                    "scalar" => Backend::Scalar,
                    "neon" => Backend::Neon,
                    _ => return Err("Unknown backend".into()),
                }
            }
            _ => return Err(format!("Unknown option {arg}").into()),
        }
    }
    if iterations == 0 {
        return Err("Iterations must be positive".into());
    }
    let start = Instant::now();
    let model = Model::from_file(model_path.ok_or("--model is required")?)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (rows, columns, features) = npy::read(feature_path.ok_or("--features is required")?)?;
    if columns != model.input_width() {
        return Err("Feature matrix must have 10 columns".into());
    }
    let backend = backend.resolve()?;
    let mut session = model.into_session(backend, threads)?;
    std::hint::black_box(session.infer(&features, rows)?); // warmup and buffer sizing
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        std::hint::black_box(session.infer(std::hint::black_box(&features), rows)?);
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    timings.sort_by(f64::total_cmp);
    let logits = session.logits();
    let greedy = if let Some(path) = spec_path {
        let alphabet = spec::alphabet(&std::fs::read(path)?)?;
        if alphabet.len() + 1 != session.output_width() {
            return Err("Alphabet/model dimensions differ".into());
        }
        quoted(&spec::greedy(logits, &alphabet)?)
    } else {
        "null".into()
    };
    if let Some(path) = output_path {
        npy::write(path, rows, session.output_width(), logits)?;
    }
    println!("{{\"backend\":{},\"threads\":{},\"layers\":{},\"timesteps\":{},\"classes\":{},\"model_load_ms\":{:.3},\"best_ms\":{:.3},\"median_ms\":{:.3},\"greedy\":{}}}",
        quoted(match backend { Backend::Neon => "neon", _ => "scalar" }), session.threads(), session.layer_count(), rows, session.output_width(), load_ms, timings[0], timings[timings.len() / 2], greedy);
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
