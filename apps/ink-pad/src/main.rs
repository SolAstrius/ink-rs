use ink_pad::engine::{default_packs, strokes_from_json, Language, Recognizer};
use std::path::PathBuf;

#[cfg(target_os = "linux")]
mod gui;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut packs = default_packs();
    let mut language = Language::English;
    let mut replay = None::<PathBuf>;
    let mut preview = None::<PathBuf>;
    let mut dump = None::<PathBuf>;
    let mut device = None::<PathBuf>;
    let mut json = false;
    let mut beam = 10.0;
    let mut auto_delay_ms = 900;
    let mut typing = true;
    let mut type_result = false;
    let mut type_text = None::<String>;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!("ink-pad [--packs DIR] [--lang en|ru] [--device /dev/input/eventN] [--auto-delay-ms N] [--beam N] [--stdout-only]\n  --replay strokes.json [--json] [--dump-features features.npy] [--type-result]\n  --type-text TEXT\n  --preview panel.png");
            return Ok(());
        }
        if arg == "--json" {
            json = true;
            continue;
        }
        if arg == "--stdout-only" {
            typing = false;
            continue;
        }
        if arg == "--type-result" {
            type_result = true;
            continue;
        }
        let value = args.next().ok_or("Missing argument value")?;
        match arg.as_str() {
            "--packs" => packs = value.into(),
            "--lang" => language = Language::parse(&value)?,
            "--replay" => replay = Some(value.into()),
            "--preview" => preview = Some(value.into()),
            "--dump-features" => dump = Some(value.into()),
            "--device" => device = Some(value.into()),
            "--beam" => beam = value.parse()?,
            "--idle-ms" | "--auto-delay-ms" => auto_delay_ms = value.parse::<u64>()?,
            "--type-text" => type_text = Some(value),
            _ => return Err(format!("Unknown option: {arg}").into()),
        }
    }
    if let Some(text) = type_text {
        #[cfg(target_os = "linux")]
        return ink_pad::keyboard::type_once(&text);
        #[cfg(not(target_os = "linux"))]
        {
            let _ = text;
            return Err("Wayland text input runs on Linux".into());
        }
    }
    if let Some(path) = preview {
        let mut pad = ink_pad::pad::Pad::new(930.0, 1240.0, language);
        pad.typing = typing;
        if !typing {
            pad.auto_delay_ms = 0;
            pad.result = "Write, then tap Recognize.".into();
        }
        if language == Language::Russian {
            pad.result = "а это я пишу на русском".into();
        }
        pad.strokes = vec![(0..120)
            .map(|i| ink_inference::features::Point {
                x: 40.0 + i as f64 * 3.0,
                y: 150.0 + (i as f64 / 5.0).sin() * 24.0,
                time_ms: i as f64 * 20.0,
            })
            .collect()];
        ink_pad::draw::Renderer::default()
            .render(&pad, 2)
            .save_png(path)?;
        return Ok(());
    }
    if let Some(path) = replay {
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let strokes = strokes_from_json(&data)?;
        let mut engine = Recognizer::load(&packs, language, beam)?;
        let result = engine.recognize(&strokes)?;
        if type_result {
            #[cfg(target_os = "linux")]
            ink_pad::keyboard::type_once(&result.hypotheses[0].text)?;
            #[cfg(not(target_os = "linux"))]
            return Err("Wayland text input runs on Linux".into());
        }
        if let Some(path) = dump {
            ink_inference::npy::write(path, result.features.len() / 10, 10, &result.features)?;
        }
        if json {
            let hypotheses: Vec<_> = result
                .hypotheses
                .iter()
                .map(|h| serde_json::json!({"text":h.text,"cost":h.cost}))
                .collect();
            println!(
                "{}",
                serde_json::json!({"greedy":result.greedy,"hypotheses":hypotheses,"timesteps":result.features.len()/10})
            );
        } else {
            println!("{}", result.hypotheses[0].text);
        }
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    return gui::run(packs, language, beam, device, auto_delay_ms, typing);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (device, auto_delay_ms, typing);
        Err("The evdev/Wayland GUI runs on Linux; use --replay or --preview on this host".into())
    }
}
fn main() {
    if let Err(error) = run() {
        eprintln!("ink-pad: {error}");
        std::process::exit(1);
    }
}
