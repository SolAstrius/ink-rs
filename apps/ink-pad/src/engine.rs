use ink_inference::{
    decoder::{BeamDecoder, CompactLm, Hypothesis, SearchOptions},
    features::{self, Stroke},
    spec, Backend, Model, Result, Session,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Language {
    English,
    Russian,
}

impl Language {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "en" => Ok(Self::English),
            "ru" => Ok(Self::Russian),
            _ => Err(ink_inference::Error("Language must be en or ru".into())),
        }
    }
    pub fn tag(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::Russian => "ru",
        }
    }
    fn paths(self, packs: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let (model, spec, lm) = match self {
            Self::English => ("x_indy_lstm.latin.6x216.tflite.20191208/latin_indy_lstm_6x216_20191208.tflite",
                "x_qrnn.en.reco_20200318.fst_20191208.recospec/qrnn.en.reco_20200318.fst_20191208.recospec.local",
                "x_en.20191208.compact.fst/en.compact.fst.local"),
            Self::Russian => ("x_indy_lstm.cyrillic.4x280.tflite.20191206/cyrillic_indy_lstm_4x280_20191206.tflite",
                "x_qrnn.ru.reco_20200717.fst_20191211.recospec/qrnn.ru.reco_20200717.fst_20191211.recospec.local",
                "x_ru.20191211.compact.fst/ru_ru.compact.fst.local"),
        };
        (packs.join(model), packs.join(spec), packs.join(lm))
    }
}

pub struct Recognition {
    pub greedy: String,
    pub hypotheses: Vec<Hypothesis>,
    pub features: Vec<f32>,
}

pub struct Recognizer {
    session: Session,
    alphabet: Vec<String>,
    decoder: BeamDecoder,
    options: SearchOptions,
}

impl Recognizer {
    pub fn load(packs: &Path, language: Language, beam: f64) -> Result<Self> {
        let (model, specification, fst) = language.paths(packs);
        let specification = std::fs::read(specification)?;
        let alphabet = spec::alphabet(&specification)?;
        let (lm_weight, char_bonus) = spec::decoder_weights(&specification)?;
        // The UI is the other compute thread; inference gets one core here.
        let session = Model::from_file(model)?.into_session(Backend::Auto, 1)?;
        let decoder = BeamDecoder::new(CompactLm::from_file(fst)?, alphabet.clone())?;
        Ok(Self {
            session,
            alphabet,
            decoder,
            options: SearchOptions {
                beam,
                max_active: 1000,
                nbest: 10,
                lm_weight,
                char_bonus,
            },
        })
    }
    pub fn recognize(&mut self, strokes: &[Stroke]) -> Result<Recognition> {
        let features = features::featurize(strokes)?;
        let logits = self.session.infer(&features, features.len() / 10)?;
        let greedy = spec::greedy(logits, &self.alphabet)?;
        let hypotheses = self.decoder.decode(logits, self.options)?;
        Ok(Recognition {
            greedy,
            hypotheses,
            features,
        })
    }
}

pub fn default_packs() -> PathBuf {
    if let Some(path) = std::env::var_os("INK_PACKS") {
        return path.into();
    }
    let local = PathBuf::from("models/packs");
    if local.is_dir() {
        local
    } else {
        PathBuf::from("/usr/local/share/ink-rs/packs")
    }
}

pub fn strokes_from_json(value: &serde_json::Value) -> Result<Vec<Stroke>> {
    let values = value
        .get("strokes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            ink_inference::Error("Expected {strokes: [[[x,y,t_ms], ...], ...]}".into())
        })?;
    values
        .iter()
        .map(|stroke| {
            stroke
                .as_array()
                .ok_or_else(|| ink_inference::Error("Stroke must be an array".into()))?
                .iter()
                .map(|point| {
                    let p = point
                        .as_array()
                        .ok_or_else(|| ink_inference::Error("Point must be [x,y,t_ms]".into()))?;
                    if p.len() != 3 {
                        return Err(ink_inference::Error("Point must have three values".into()));
                    }
                    let number = |i: usize| {
                        p[i].as_f64().ok_or_else(|| {
                            ink_inference::Error("Point values must be numbers".into())
                        })
                    };
                    Ok(features::Point {
                        x: number(0)?,
                        y: number(1)?,
                        time_ms: number(2)?,
                    })
                })
                .collect()
        })
        .collect()
}
