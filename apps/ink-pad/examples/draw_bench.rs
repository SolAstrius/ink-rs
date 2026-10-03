use ink_inference::features::Point;
use ink_pad::{draw::Renderer, engine::Language, pad::Pad};
use std::{hint::black_box, time::Instant};

fn main() {
    let mut pad = Pad::new(930.0, 1240.0, Language::English);
    pad.result.clear();
    let points: Vec<_> = (0..1500)
        .map(|i| Point {
            x: 30.0 + (i % 750) as f64 * 1.1,
            y: 100.0 + (i as f64 / 12.0).sin() * 25.0 + (i / 750) as f64 * 90.0,
            time_ms: i as f64,
        })
        .collect();
    pad.strokes.push(points.clone());
    let mut renderer = Renderer::default();
    let mut pixels = renderer.render(&pad, 2);
    let mut full = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        black_box(renderer.render(black_box(&pad), 2));
        full.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    full.sort_by(f64::total_cmp);
    let start = Instant::now();
    let repetitions = 20;
    for _ in 0..repetitions {
        for points in points.windows(2) {
            Renderer::segment(black_box(&mut pixels), points[0], points[1], 2);
        }
    }
    let segments = (points.len() - 1) * repetitions;
    let microseconds = start.elapsed().as_secs_f64() * 1e6;
    println!(
        "{}",
        serde_json::json!({
            "scope":"CPU rendering only; excludes compositor and physical e-ink latency",
            "full_render_median_ms":full[3], "segments":segments,
            "incremental_segment_us":microseconds / segments as f64,
            "black_white":pixels.data().chunks_exact(4).all(|p| p[0] == 0 || p[0] == 255),
        })
    );
}
