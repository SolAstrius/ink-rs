use crate::engine::Language;
use crate::pad::{auto_button, auto_button_label, buttons, Action, Pad, TOOLBAR};
use fontdue::{Font, FontSettings};
use ink_inference::features::Point;
use std::collections::HashMap;
use tiny_skia::{Color, Paint, PathBuilder, Pixmap, Rect, Stroke, Transform};

pub struct Renderer {
    font: Font,
    fallback: Font,
    glyphs: HashMap<(char, u32), (fontdue::Metrics, Vec<u8>)>,
}
impl Default for Renderer {
    fn default() -> Self {
        Self {
            font: Font::from_bytes(
                include_bytes!("../assets/fonts/AtkinsonHyperlegibleNext-Regular.ttf") as &[u8],
                FontSettings::default(),
            )
            .unwrap(),
            fallback: Font::from_bytes(
                include_bytes!("../assets/fonts/DejaVuSans.ttf") as &[u8],
                FontSettings::default(),
            )
            .unwrap(),
            glyphs: HashMap::new(),
        }
    }
}
fn paint(gray: u8) -> Paint<'static> {
    let mut p = Paint::default();
    p.set_color_rgba8(gray, gray, gray, 255);
    p
}

impl Renderer {
    /// Paint only the new ink segment into the retained surface. Integer black
    /// pixels avoid allocating/rasterizing the complete page on every pen frame.
    pub fn segment(pm: &mut Pixmap, from: Point, to: Point, scale: u32) {
        let scale = scale.max(1) as i32;
        let (mut x, mut y) = (
            (from.x * scale as f64).round() as i32,
            (from.y * scale as f64).round() as i32,
        );
        let (end_x, end_y) = (
            (to.x * scale as f64).round() as i32,
            (to.y * scale as f64).round() as i32,
        );
        let dx = (end_x - x).abs();
        let dy = -(end_y - y).abs();
        let sx = if x < end_x { 1 } else { -1 };
        let sy = if y < end_y { 1 } else { -1 };
        let mut error = dx + dy;
        let width = pm.width() as i32;
        let height = pm.height() as i32;
        loop {
            for row in -scale..=scale {
                let py = y + row;
                if py < 0 || py >= height {
                    continue;
                }
                for column in -scale..=scale {
                    let px = x + column;
                    if px < 0 || px >= width || row * row + column * column > scale * scale {
                        continue;
                    }
                    let offset = (py as usize * width as usize + px as usize) * 4;
                    pm.data_mut()[offset..offset + 4].copy_from_slice(&[0, 0, 0, 255]);
                }
            }
            if x == end_x && y == end_y {
                break;
            }
            let twice = 2 * error;
            if twice >= dy {
                error += dy;
                x += sx;
            }
            if twice <= dx {
                error += dx;
                y += sy;
            }
        }
    }

    fn monochrome(pm: &mut Pixmap, first_row: usize) {
        const BAYER: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
        let width = pm.width() as usize;
        for (row, pixels) in pm
            .data_mut()
            .chunks_exact_mut(width * 4)
            .enumerate()
            .skip(first_row)
        {
            for (column, pixel) in pixels.chunks_exact_mut(4).enumerate() {
                let gray = if pixel[0] > BAYER[row & 3][column & 3] * 16 + 8 {
                    255
                } else {
                    0
                };
                pixel.copy_from_slice(&[gray, gray, gray, 255]);
            }
        }
    }

    fn text(&mut self, pm: &mut Pixmap, text: &str, x: f32, baseline: f32, size: f32, gray: u8) {
        let mut cursor = x;
        for ch in text.chars() {
            let (metrics, mask) = self.glyphs.entry((ch, size.to_bits())).or_insert_with(|| {
                if self.font.has_glyph(ch) {
                    self.font.rasterize(ch, size)
                } else {
                    self.fallback.rasterize(ch, size)
                }
            });
            if cursor + metrics.advance_width >= pm.width() as f32 - 10.0 {
                break;
            }
            let top = baseline - metrics.height as f32 - metrics.ymin as f32;
            for row in 0..metrics.height {
                for column in 0..metrics.width {
                    let px = cursor as i32 + metrics.xmin + column as i32;
                    let py = top as i32 + row as i32;
                    if px < 0 || py < 0 || px >= pm.width() as i32 || py >= pm.height() as i32 {
                        continue;
                    }
                    let alpha = mask[row * metrics.width + column] as u32;
                    let index = (py as usize * pm.width() as usize + px as usize) * 4;
                    for channel in 0..3 {
                        let old = pm.data()[index + channel] as u32;
                        pm.data_mut()[index + channel] =
                            ((gray as u32 * alpha + old * (255 - alpha)) / 255) as u8;
                    }
                }
            }
            cursor += metrics.advance_width;
        }
    }
    pub fn render(&mut self, pad: &Pad, scale: u32) -> Pixmap {
        let scale = scale.max(1) as f32;
        let mut pm = Pixmap::new(
            (pad.width * scale as f64).round() as u32,
            (pad.height * scale as f64).round() as u32,
        )
        .unwrap();
        pm.fill(Color::WHITE);
        let mut line = PathBuilder::new();
        line.move_to(0.0, (TOOLBAR as f32) * scale);
        line.line_to(pm.width() as f32, TOOLBAR as f32 * scale);
        pm.stroke_path(
            &line.finish().unwrap(),
            &paint(0),
            &Stroke {
                width: scale,
                ..Stroke::default()
            },
            Transform::identity(),
            None,
        );
        self.text(
            &mut pm,
            "INK PAD",
            12.0 * scale,
            28.0 * scale,
            16.0 * scale,
            0,
        );
        for (action, rect, label) in buttons(pad.width, pad.typing) {
            let selected = (action == Action::English && pad.language == Language::English)
                || (action == Action::Russian && pad.language == Language::Russian);
            let r = Rect::from_xywh(
                rect.x as f32 * scale,
                rect.y as f32 * scale,
                rect.width as f32 * scale,
                rect.height as f32 * scale,
            )
            .unwrap();
            pm.fill_rect(
                r,
                &paint(if selected { 0 } else { 255 }),
                Transform::identity(),
                None,
            );
            let path = PathBuilder::from_rect(r);
            pm.stroke_path(
                &path,
                &paint(0),
                &Stroke {
                    width: scale,
                    ..Stroke::default()
                },
                Transform::identity(),
                None,
            );
            self.text(
                &mut pm,
                label,
                (rect.x as f32 + 8.0) * scale,
                28.0 * scale,
                15.0 * scale,
                if selected { 255 } else { 0 },
            );
        }
        for stroke in pad.strokes.iter().chain(std::iter::once(&pad.current)) {
            if let Some(first) = stroke.first() {
                Self::segment(&mut pm, *first, *first, scale as u32);
            }
            for points in stroke.windows(2) {
                Self::segment(&mut pm, points[0], points[1], scale as u32);
            }
        }
        self.status(&mut pm, pad, scale as u32);
        Self::monochrome(&mut pm, 0);
        pm
    }

    pub fn status(&mut self, pm: &mut Pixmap, pad: &Pad, scale: u32) {
        let scale = scale.max(1) as f32;
        let top = ((pad.height - 28.0) * scale as f64).floor().max(0.0) as usize;
        let stride = pm.width() as usize * 4;
        pm.data_mut()[top * stride..].fill(255);
        let status = if pad.busy {
            "Recognizing..."
        } else {
            &pad.result
        };
        self.text(
            pm,
            status,
            12.0 * scale,
            (pad.height as f32 - 9.0) * scale,
            14.0 * scale,
            0,
        );
        if pad.typing {
            let button = auto_button(pad.width, pad.height);
            let r = Rect::from_xywh(
                (button.x * scale as f64) as f32,
                (button.y * scale as f64) as f32,
                (button.width * scale as f64) as f32,
                (button.height * scale as f64) as f32,
            )
            .unwrap();
            let active = pad.auto_delay_ms != 0;
            pm.fill_rect(
                r,
                &paint(if active { 0 } else { 255 }),
                Transform::identity(),
                None,
            );
            pm.stroke_path(
                &PathBuilder::from_rect(r),
                &paint(0),
                &Stroke {
                    width: scale,
                    ..Stroke::default()
                },
                Transform::identity(),
                None,
            );
            self.text(
                pm,
                &auto_button_label(pad.auto_delay_ms),
                (button.x as f32 + 6.0) * scale,
                (button.y as f32 + 16.0) * scale,
                11.0 * scale,
                if active { 255 } else { 0 },
            );
        }
        Self::monochrome(pm, top);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incremental_ink_and_status_match_full_black_white_render() {
        let mut pad = Pad::new(930.0, 1240.0, Language::English);
        pad.result.clear();
        let mut renderer = Renderer::default();
        let mut actual = renderer.render(&pad, 2);
        let points: Vec<_> = (0..160)
            .map(|i| Point {
                x: 50.0 + i as f64 * 2.0,
                y: 150.0 + (i as f64 / 7.0).sin() * 30.0,
                time_ms: i as f64,
            })
            .collect();
        Renderer::segment(&mut actual, points[0], points[0], 2);
        for pair in points.windows(2) {
            Renderer::segment(&mut actual, pair[0], pair[1], 2);
        }
        pad.strokes.push(points);
        pad.result = "проверка".into();
        renderer.status(&mut actual, &pad, 2);
        let expected = renderer.render(&pad, 2);
        assert_eq!(actual.data(), expected.data());
        assert!(actual
            .data()
            .chunks_exact(4)
            .all(|pixel| (pixel[0] == 0 || pixel[0] == 255)
                && pixel[0] == pixel[1]
                && pixel[1] == pixel[2]
                && pixel[3] == 255));
    }

    #[test]
    fn russian_results_have_display_glyphs() {
        let renderer = Renderer::default();
        for ch in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя".chars()
        {
            assert!(
                renderer.font.has_glyph(ch) || renderer.fallback.has_glyph(ch),
                "Missing glyph: {ch}"
            );
        }
    }
}
