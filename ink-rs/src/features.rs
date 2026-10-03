//! Stroke normalization and the validated ten-feature cubic-curve encoding.
//! Uses float64 for fitting and float32 for the model input.

use crate::{error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub time_ms: f64,
}

pub type Stroke = Vec<Point>;
type V = [f64; 3];
type Curve = [V; 4];

fn distance(a: V, b: V) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn norm(a: V) -> f64 {
    a[0].hypot(a[1])
}
fn cross(a: V, b: V) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn evaluate(c: &Curve, s: f64) -> V {
    std::array::from_fn(|d| c[0][d] + s * c[1][d] + s * s * c[2][d] + s * s * s * c[3][d])
}
fn derivatives(c: &Curve, s: f64) -> (V, V) {
    (
        std::array::from_fn(|d| c[1][d] + 2.0 * s * c[2][d] + 3.0 * s * s * c[3][d]),
        std::array::from_fn(|d| 2.0 * c[2][d] + 6.0 * s * c[3][d]),
    )
}

/// Householder QR for a Vandermonde matrix with at most four columns.
fn least_squares(s: &[f64], points: &[V], degree: usize) -> Curve {
    let columns = degree + 1;
    let mut a: Vec<[f64; 4]> = s.iter().map(|x| [1.0, *x, x * x, x * x * x]).collect();
    let mut b = points.to_vec();
    for k in 0..columns {
        let length = a[k..].iter().fold(0.0_f64, |sum, row| sum.hypot(row[k]));
        if length < 1e-15 {
            continue;
        }
        let alpha = if a[k][k] >= 0.0 { -length } else { length };
        let mut v: Vec<_> = a[k..].iter().map(|row| row[k]).collect();
        v[0] -= alpha;
        let square: f64 = v.iter().map(|x| x * x).sum();
        if square == 0.0 {
            continue;
        }
        for column in k..columns {
            let coefficient = 2.0
                * a[k..]
                    .iter()
                    .zip(&v)
                    .map(|(row, v)| row[column] * v)
                    .sum::<f64>()
                / square;
            for (row, v) in a[k..].iter_mut().zip(&v) {
                row[column] -= coefficient * v;
            }
        }
        for dimension in 0..3 {
            let coefficient = 2.0
                * b[k..]
                    .iter()
                    .zip(&v)
                    .map(|(row, v)| row[dimension] * v)
                    .sum::<f64>()
                / square;
            for (row, v) in b[k..].iter_mut().zip(&v) {
                row[dimension] -= coefficient * v;
            }
        }
    }
    let mut curve = [[0.0; 3]; 4];
    for dimension in 0..3 {
        for k in (0..columns).rev() {
            let remaining: f64 = (k + 1..columns)
                .map(|j| a[k][j] * curve[j][dimension])
                .sum();
            if a[k][k].abs() > 1e-15 {
                curve[k][dimension] = (b[k][dimension] - remaining) / a[k][k];
            }
        }
    }
    curve
}

fn fit_segment(points: &[V], convergence: f64) -> (Curve, f64, f64, Vec<f64>) {
    let n = points.len();
    let mut arc = vec![0.0; n];
    for i in 1..n {
        arc[i] = arc[i - 1] + distance(points[i], points[i - 1]);
    }
    let mut s: Vec<_> = if arc[n - 1] > 0.0 {
        arc.iter().map(|d| d / arc[n - 1]).collect()
    } else {
        (0..n)
            .map(|i| {
                if n > 1 {
                    i as f64 / (n - 1) as f64
                } else {
                    0.0
                }
            })
            .collect()
    };
    let mut previous = None;
    let mut result = ([[0.0; 3]; 4], 0.0, 0.0);
    for _ in 0..30 {
        let curve = least_squares(&s, points, (n - 1).min(3));
        let distances: Vec<_> = s
            .iter()
            .zip(points)
            .map(|(s, p)| distance(evaluate(&curve, *s), *p))
            .collect();
        let rms = (distances.iter().map(|d| d * d).sum::<f64>() / n as f64).sqrt();
        let maximum = distances.iter().copied().fold(0.0, f64::max);
        for (s, p) in s.iter_mut().zip(points) {
            let fitted = evaluate(&curve, *s);
            let residual = std::array::from_fn(|d| p[d] - fitted[d]);
            let (first, second) = derivatives(&curve, *s);
            let numerator = dot(first, residual);
            let denominator = dot(second, residual) - dot(first, first);
            *s = (*s
                - numerator
                    / if denominator == 0.0 {
                        -1e-30
                    } else {
                        denominator
                    })
            .clamp(0.0, 1.0);
        }
        s[0] = 0.0;
        s[n - 1] = 1.0;
        result = (curve, rms, maximum);
        if rms < convergence || previous == Some(rms) {
            break;
        }
        previous = Some(rms);
    }
    (result.0, result.1, result.2, s)
}

fn controls(c: &Curve) -> [V; 4] {
    [
        std::array::from_fn(|d| c[1][d] + c[2][d] + c[3][d]),
        std::array::from_fn(|d| c[1][d] / 3.0),
        std::array::from_fn(|d| (c[1][d] + c[2][d]) / 3.0),
        std::array::from_fn(|d| -c[1][d] / 3.0 - 2.0 * c[2][d] / 3.0 - c[3][d]),
    ]
}

fn bend_ok(c: &Curve) -> bool {
    let [e, v1, v2, v3] = controls(c);
    let (n1, n2, n3, chord) = (norm(v1), norm(v2), norm(v3), norm(e));
    if n1 == 0.0 || n2 == 0.0 || chord == 0.0 {
        return false;
    }
    if dot(v1, v2) / (n1 * n2) < -0.8 {
        return false;
    }
    if n3 > 0.0 && -dot(v2, v3) / (n2 * n3) < -0.8 {
        return false;
    }
    n1 + n2 + n3 < 3.0 * chord
}

fn dedup(points: &[V], diagonal: f64) -> Vec<V> {
    let threshold = (diagonal as f32 * 0.000_345_267_f32) as f64;
    let n = points.len();
    let mut arc = vec![0.0; n];
    for i in 1..n {
        arc[i] = arc[i - 1] + distance(points[i], points[i - 1]);
    }
    let mut counts = vec![1; n];
    let mut previous = vec![0; n];
    let mut best_counts = vec![1; n];
    let mut best = 0;
    for i in 1..n {
        previous[i] = i;
        for j in (0..i).rev() {
            if counts[i] > best_counts[j] + 1 {
                break;
            }
            if counts[j] + 1 < counts[i] || arc[i] - arc[j] <= threshold {
                continue;
            }
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            if dx * dx + dy * dy <= threshold * threshold {
                continue;
            }
            counts[i] = counts[j] + 1;
            previous[i] = j;
            if counts[i] >= counts[best] {
                best = i;
            }
        }
        best_counts[i] = counts[best];
    }
    let mut path = Vec::with_capacity(counts[best]);
    let mut k = best;
    for _ in 0..counts[best] {
        path.push(points[k]);
        k = previous[k];
    }
    path.reverse();
    path
}

fn split_angle(points: &[V], threshold: f64) -> Option<usize> {
    let mut best = None;
    let mut smallest = 1.0;
    for i in 1..points.len().saturating_sub(1) {
        let mut j = i as isize - 1;
        while j >= 0 && distance(points[i], points[j as usize]) < threshold {
            j -= 1;
        }
        let mut k = i + 1;
        while k < points.len() && distance(points[i], points[k]) < threshold {
            k += 1;
        }
        if j < 0 || k == points.len() {
            continue;
        }
        let a = std::array::from_fn(|d| points[i][d] - points[j as usize][d]);
        let b = std::array::from_fn(|d| points[k][d] - points[i][d]);
        let cosine = dot(a, b) / (norm(a) * norm(b));
        if cosine < smallest {
            smallest = cosine;
            best = Some(i);
        }
    }
    best
}

fn split_extreme(curve: &Curve, s: &[f64]) -> usize {
    let mut maximum = -1.0;
    let mut selected = s[1];
    for i in 0..100 {
        let value = s[1] + (s[s.len() - 2] - s[1]) * i as f64 / 99.0;
        let (first, second) = derivatives(curve, value);
        let curvature = cross(first, second).abs() / dot(first, first).max(1e-30).powf(1.5);
        if curvature > maximum {
            maximum = curvature;
            selected = value;
        }
    }
    (1..s.len() - 1)
        .min_by(|a, b| {
            (s[*a] - selected)
                .abs()
                .total_cmp(&(s[*b] - selected).abs())
        })
        .unwrap()
}

fn segment(points: &[V], start: usize, n: usize, tolerance: f64) -> (Curve, f64, f64, Vec<f64>) {
    let mut local = points[start..start + n].to_vec();
    let origin = local[0][2];
    for point in &mut local {
        point[2] -= origin;
    }
    fit_segment(&local, tolerance)
}

fn fit_stroke(points: &[V]) -> Vec<Curve> {
    let (mut xmin, mut xmax, mut ymin, mut ymax) =
        (points[0][0], points[0][0], points[0][1], points[0][1]);
    for p in points {
        xmin = xmin.min(p[0]);
        xmax = xmax.max(p[0]);
        ymin = ymin.min(p[1]);
        ymax = ymax.max(p[1]);
    }
    let diagonal = (xmax - xmin).hypot(ymax - ymin);
    let mut points = dedup(points, diagonal);
    let length: f64 = points.windows(2).map(|p| distance(p[0], p[1])).sum();
    let duration = points.last().unwrap()[2] - points[0][2];
    let time_scale = if duration > 0.0 {
        length / duration
    } else {
        0.0
    };
    for point in &mut points {
        point[2] *= time_scale;
    }
    let (angle, maximum, rms) = (0.05 * diagonal, 0.02 * diagonal, 0.01 * diagonal);
    let mut stack = vec![(0, points.len())];
    let mut out = Vec::new();
    while let Some((start, n)) = stack.pop() {
        if n > 80 {
            if let Some(i) = split_angle(&points[start..start + n], angle) {
                stack.push((start + i, n - i));
                stack.push((start, i + 1));
                continue;
            }
        }
        let (curve, fit_rms, fit_max, s) = segment(&points, start, n, 0.1 * rms);
        if fit_rms > rms || fit_max > maximum {
            if let Some(i) = split_angle(&points[start..start + n], angle) {
                stack.push((start + i, n - i));
                stack.push((start, i + 1));
                continue;
            }
            out.push((start, n, curve));
            continue;
        }
        if n >= 4 && !bend_ok(&curve) {
            let i = split_extreme(&curve, &s);
            stack.push((start + i, n - i));
            stack.push((start, i + 1));
            continue;
        }
        out.push((start, n, curve));
    }
    let mut k = 0;
    while k + 1 < out.len() {
        let (start, first, _) = out[k];
        let (_, second, _) = out[k + 1];
        let n = first + second - 1;
        let (curve, fit_rms, fit_max, _) = segment(&points, start, n, 0.1 * rms);
        if fit_rms <= rms && fit_max <= maximum && (first + second < 5 || bend_ok(&curve)) {
            out[k] = (start, n, curve);
            out.remove(k + 1);
        } else {
            k += 1;
        }
    }
    out.into_iter().map(|(_, _, curve)| curve).collect()
}

fn append_features(output: &mut Vec<f32>, points: &[V], pen: bool) {
    for curve in fit_stroke(points) {
        let [e, v1, _, v3] = controls(&curve);
        let length = norm(e);
        output.extend([
            if pen { 1.0 } else { 0.0 },
            e[0] as f32,
            e[1] as f32,
            cross(e, v1).atan2(dot(e, v1)) as f32,
            if length > 0.0 {
                (norm(v1) / length) as f32
            } else {
                0.0
            },
            cross(v3, e).atan2(-dot(v3, e)) as f32,
            if length > 0.0 {
                (norm(v3) / length) as f32
            } else {
                0.0
            },
            e[2] as f32,
            v1[2] as f32,
            v3[2] as f32,
        ]);
    }
}

pub fn featurize(strokes: &[Stroke]) -> Result<Vec<f32>> {
    let strokes: Vec<_> = strokes.iter().filter(|s| !s.is_empty()).collect();
    if strokes.is_empty() {
        return Err(error("No ink to recognize"));
    }
    if strokes
        .iter()
        .flat_map(|s| s.iter())
        .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.time_ms.is_finite())
    {
        return Err(error("Non-finite ink coordinate or timestamp"));
    }
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for point in strokes.iter().flat_map(|s| s.iter()) {
        xmin = xmin.min(point.x);
        xmax = xmax.max(point.x);
        ymin = ymin.min(point.y);
        ymax = ymax.max(point.y);
    }
    let mut height = (ymax - ymin).max((xmax - xmin) / 100.0);
    if height < 2.0_f64.powi(-23) {
        height = 1.0;
    }
    let x0 = strokes[0][0].x;
    let normalized: Vec<Vec<V>> = strokes
        .iter()
        .map(|s| {
            s.iter()
                .map(|p| [(p.x - x0) / height, (p.y - ymin) / height, p.time_ms])
                .collect()
        })
        .collect();
    let mut output = Vec::new();
    for (i, stroke) in normalized.iter().enumerate() {
        if i > 0 {
            append_features(
                &mut output,
                &[*normalized[i - 1].last().unwrap(), stroke[0]],
                false,
            );
        }
        append_features(&mut output, stroke, true);
    }
    if output.iter().any(|v| !v.is_finite()) {
        return Err(error("Curve fitting produced non-finite features"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translation_and_uniform_scale_preserve_features() {
        let make = |scale: f64, offset: f64| {
            vec![(0..40)
                .map(|i| Point {
                    x: i as f64 * scale + offset,
                    y: (i as f64 / 8.0).sin() * scale + offset,
                    time_ms: i as f64 * 20.0,
                })
                .collect()]
        };
        let first = featurize(&make(1.0, 0.0)).unwrap();
        let second = featurize(&make(3.0, 100.0)).unwrap();
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn pen_lifts_and_degenerate_dots_are_finite() {
        let strokes = vec![
            vec![Point {
                x: 1.0,
                y: 2.0,
                time_ms: 0.0,
            }],
            vec![Point {
                x: 4.0,
                y: 6.0,
                time_ms: 20.0,
            }],
        ];
        let f = featurize(&strokes).unwrap();
        assert_eq!(f.len(), 30);
        assert_eq!(f[0], 1.0);
        assert_eq!(f[10], 0.0);
        assert_eq!(f[20], 1.0);
        assert!(f.iter().all(|v| v.is_finite()));
    }
}
