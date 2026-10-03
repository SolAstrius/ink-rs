//! Exercise changing input lengths and language reloads on one retained worker.
use ink_pad::engine::{strokes_from_json, Language, Recognizer};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let english = args.next().ok_or("English replay JSON path required")?;
    let russian = args.next().ok_or("Russian replay JSON path required")?;
    let packs = PathBuf::from(args.next().ok_or("model-pack path required")?);
    // Keep the captured value dynamic to expose register-preservation failures.
    let beam: f64 = args.next().ok_or("beam width required")?.parse()?;
    let en = strokes_from_json(&serde_json::from_slice(&std::fs::read(english)?)?)?;
    let ru = strokes_from_json(&serde_json::from_slice(&std::fs::read(russian)?)?)?;
    let inputs = vec![
        (
            Language::English,
            en.clone(),
            Some("that's a fairly long sentence"),
        ),
        (Language::English, en[..5.min(en.len())].to_vec(), None),
        (
            Language::English,
            en.clone(),
            Some("that's a fairly long sentence"),
        ),
        (
            Language::Russian,
            ru.clone(),
            Some("а это я пишу на русском"),
        ),
        (Language::Russian, ru[..5.min(ru.len())].to_vec(), None),
        (
            Language::Russian,
            ru.clone(),
            Some("а это я пишу на русском"),
        ),
        (Language::English, en, Some("that's a fairly long sentence")),
        (Language::Russian, ru, Some("а это я пишу на русском")),
    ];
    let results = std::thread::Builder::new().name("ink-recognizer".into()).spawn(move || {
        let mut loaded = None::<(Language, Recognizer)>;
        let mut results = Vec::new();
        for (language, strokes, expected) in inputs {
            if loaded.as_ref().is_none_or(|(lang, _)| *lang != language) {
                drop(loaded.take());
                loaded = Some((language, Recognizer::load(&packs, language, beam)?));
            }
            let result = loaded.as_mut().unwrap().1.recognize(&strokes)?;
            let text = result.hypotheses[0].text.clone();
            if let Some(expected) = expected { assert_eq!(text, expected); }
            results.push(serde_json::json!({"language":language.tag(),"timesteps":result.features.len()/10,"text":text}));
        }
        Ok::<_, ink_inference::Error>(results)
    })?.join().map_err(|_| "recognition worker panicked")??;
    println!(
        "{}",
        serde_json::json!({"worker_threads":1, "beam":beam, "calls":results})
    );
    Ok(())
}
