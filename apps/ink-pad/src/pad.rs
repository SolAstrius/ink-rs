use crate::{
    engine::Language,
    input::{Edge, Frame},
};
use ink_inference::features::{Point, Stroke};
use std::time::{Duration, Instant};

pub const TOOLBAR: f64 = 44.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Recognize,
    Clear,
    English,
    Russian,
    Close,
    Space,
    Backspace,
    Enter,
    DeleteWord,
    Left,
    Right,
    Up,
    Down,
    ToggleAuto,
}

#[derive(Clone, Copy)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}
impl Rect {
    pub fn contains(self, x: f64, y: f64) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height
    }
}

pub fn buttons(width: f64, typing: bool) -> Vec<(Action, Rect, &'static str)> {
    let mut buttons = vec![
        (
            Action::English,
            Rect {
                x: 64.0,
                y: 7.0,
                width: 40.0,
                height: 30.0,
            },
            "EN",
        ),
        (
            Action::Russian,
            Rect {
                x: 110.0,
                y: 7.0,
                width: 40.0,
                height: 30.0,
            },
            "RU",
        ),
        (
            Action::Recognize,
            Rect {
                x: width - 240.0,
                y: 7.0,
                width: 112.0,
                height: 30.0,
            },
            if typing { "Insert" } else { "Recognize" },
        ),
        (
            Action::Clear,
            Rect {
                x: width - 120.0,
                y: 7.0,
                width: 66.0,
                height: 30.0,
            },
            "Clear",
        ),
        (
            Action::Close,
            Rect {
                x: width - 46.0,
                y: 7.0,
                width: 34.0,
                height: 30.0,
            },
            "X",
        ),
    ];
    if typing {
        buttons.extend([
            (
                Action::Space,
                Rect {
                    x: 160.0,
                    y: 7.0,
                    width: 64.0,
                    height: 30.0,
                },
                "Space",
            ),
            (
                Action::Backspace,
                Rect {
                    x: 236.0,
                    y: 7.0,
                    width: 62.0,
                    height: 30.0,
                },
                "Bksp",
            ),
        ]);
        let control = |action, x, width, label| {
            (
                action,
                Rect {
                    x,
                    y: 7.0,
                    width,
                    height: 30.0,
                },
                label,
            )
        };
        buttons.extend([
            control(Action::DeleteWord, 310.0, 90.0, "Del word"),
            control(Action::Enter, 412.0, 68.0, "Enter"),
            control(Action::Left, 492.0, 38.0, "←"),
            control(Action::Right, 538.0, 38.0, "→"),
            control(Action::Up, 584.0, 38.0, "↑"),
            control(Action::Down, 630.0, 38.0, "↓"),
        ]);
    }
    buttons
}

pub fn auto_button(width: f64, height: f64) -> Rect {
    Rect {
        x: width - 110.0,
        y: height - 25.0,
        width: 98.0,
        height: 22.0,
    }
}

pub fn auto_button_label(delay_ms: u64) -> String {
    if delay_ms == 0 {
        "Auto off".into()
    } else {
        format!("Auto {:.1}s", delay_ms as f32 / 1000.0)
    }
}

pub struct Pad {
    pub width: f64,
    pub height: f64,
    pub screen_height: f64,
    pub strokes: Vec<Stroke>,
    pub current: Stroke,
    pub language: Language,
    pub generation: u64,
    pub result: String,
    pub busy: bool,
    pub typing: bool,
    pub auto_delay_ms: u64,
    capturing: bool,
    pen_contact: bool,
    last_pen_up: Option<Instant>,
    origin_time: Option<f64>,
}

impl Pad {
    pub fn new(width: f64, screen_height: f64, language: Language) -> Self {
        Self {
            width,
            height: screen_height / 4.0,
            screen_height,
            language,
            strokes: Vec::new(),
            current: Vec::new(),
            generation: 0,
            result: "Focus a text field; Auto inserts after a pause.".into(),
            busy: false,
            typing: true,
            auto_delay_ms: 900,
            capturing: false,
            pen_contact: false,
            last_pen_up: None,
            origin_time: None,
        }
    }
    pub fn clear(&mut self) {
        self.strokes.clear();
        self.current.clear();
        self.capturing = false;
        self.origin_time = None;
        self.last_pen_up = None;
        self.generation += 1;
        self.result.clear();
    }
    pub fn is_down(&self) -> bool {
        self.capturing
    }
    pub fn handle(&mut self, frame: Frame) -> Option<Action> {
        self.handle_at(frame, Instant::now())
    }
    pub fn handle_at(&mut self, frame: Frame, now: Instant) -> Option<Action> {
        if frame.edge == Edge::Down {
            self.pen_contact = true;
            self.last_pen_up = None;
        } else if frame.edge == Edge::Up {
            self.pen_contact = false;
        }
        let (x, y) = (frame.x, frame.y - (self.screen_height - self.height));
        let canvas = Rect {
            x: 8.0,
            y: TOOLBAR + 4.0,
            width: self.width - 16.0,
            height: self.height - TOOLBAR - 32.0,
        };
        if frame.edge == Edge::Down {
            if self.typing && auto_button(self.width, self.height).contains(x, y) {
                return Some(Action::ToggleAuto);
            }
            if let Some((action, _, _)) = buttons(self.width, self.typing)
                .iter()
                .find(|(_, rect, _)| rect.contains(x, y))
            {
                return Some(*action);
            }
            if canvas.contains(x, y) {
                if frame.eraser {
                    self.strokes.pop();
                    self.result.clear();
                    self.generation += 1;
                    return None;
                }
                self.current.clear();
                self.capturing = true;
                self.result.clear();
                self.origin_time.get_or_insert(frame.time_ms);
            }
        }
        if self.capturing && (frame.edge == Edge::Down || frame.moved || frame.edge == Edge::Up) {
            if canvas.contains(x, y) {
                let point = Point {
                    x,
                    y,
                    time_ms: (frame.time_ms - self.origin_time.unwrap_or(frame.time_ms)).max(0.0),
                };
                if self
                    .current
                    .last()
                    .is_none_or(|last| last.x != point.x || last.y != point.y)
                {
                    self.current.push(point);
                    self.generation += 1;
                }
            } else {
                self.finish();
            }
        }
        if frame.edge == Edge::Up {
            self.finish();
            if !self.strokes.is_empty() {
                self.last_pen_up = Some(now);
            }
        }
        None
    }
    pub fn inserted(&mut self, text: String) {
        self.clear();
        self.result = text;
    }
    pub fn auto_ready(&self, now: Instant, last_submitted: u64) -> bool {
        self.auto_delay_ms != 0
            && !self.busy
            && !self.pen_contact
            && !self.capturing
            && !self.strokes.is_empty()
            && self.generation != last_submitted
            && self.last_pen_up.is_some_and(|up| {
                now.saturating_duration_since(up) >= Duration::from_millis(self.auto_delay_ms)
            })
    }
    pub fn toggle_auto(&mut self, now: Instant) {
        self.auto_delay_ms = if self.auto_delay_ms == 0 { 900 } else { 0 };
        self.last_pen_up = if !self.strokes.is_empty() && !self.pen_contact {
            Some(now)
        } else {
            None
        };
    }
    fn finish(&mut self) {
        self.capturing = false;
        if !self.current.is_empty() {
            self.strokes.push(std::mem::take(&mut self.current));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(edge: Edge, x: f64, y: f64) -> Frame {
        Frame {
            edge,
            x,
            y,
            time_ms: 100.0,
            moved: true,
            eraser: false,
        }
    }
    #[test]
    fn bottom_quarter_mapping_and_stroke_boundaries() {
        let mut pad = Pad::new(930.0, 1240.0, Language::English);
        assert_eq!(pad.height, 310.0);
        pad.handle(frame(Edge::Down, 100.0, 100.0));
        assert!(pad.current.is_empty());
        pad.handle(frame(Edge::Down, 100.0, 1030.0));
        pad.handle(frame(Edge::None, 120.0, 1050.0));
        pad.handle(frame(Edge::Up, 120.0, 1050.0));
        assert_eq!(pad.strokes.len(), 1);
        assert_eq!(pad.strokes[0][0].y, 100.0);
        assert!(!pad.is_down());
    }
    #[test]
    fn inserted_text_clears_ink_and_starts_the_next_capture_fresh() {
        let mut pad = Pad::new(930.0, 1240.0, Language::Russian);
        pad.handle(frame(Edge::Down, 100.0, 1030.0));
        pad.handle(frame(Edge::Up, 110.0, 1040.0));
        let generation = pad.generation;
        pad.inserted("привет".into());
        assert!(pad.strokes.is_empty() && pad.current.is_empty());
        assert_eq!(pad.result, "привет");
        assert_eq!(pad.language, Language::Russian);
        assert!(pad.generation > generation);
        let mut next = frame(Edge::Down, 150.0, 1030.0);
        next.time_ms = 9000.0;
        pad.handle(next);
        assert_eq!(pad.current[0].time_ms, 0.0);
        assert_eq!(pad.current.len(), 1);
    }
}
