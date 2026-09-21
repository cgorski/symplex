//! Adaptive sampling engine for plotting symbolic functions.
//!
//! Provides [`sample_compiled`] which evaluates a function over a range with
//! adaptive refinement near sharp features, automatic discontinuity detection,
//! and asymptote annotation.

use std::f64::consts::PI;

use crate::base::interval::Interval;

/// Result of sampling a function over a range.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PlotData {
    /// (x, y) pairs. y = f64::NAN indicates a line break (discontinuity).
    pub points: Vec<(f64, f64)>,
    /// X-locations of detected asymptotes (for rendering vertical dashed lines).
    pub asymptotes: Vec<f64>,
    /// X-locations where the function is undefined.
    pub excluded: Vec<f64>,
}

/// Options for the sampling engine.
#[derive(Debug, Clone)]
pub struct SampleOptions {
    /// Minimum number of initial sample points (before adaptive refinement).
    pub min_points: usize,
    /// Maximum recursion depth for adaptive refinement.
    pub max_depth: usize,
    /// Adaptive tolerance — midpoint deviation threshold (relative).
    pub tolerance: f64,
    /// Whether to add random jitter to sample points (prevents aliasing).
    pub jitter: bool,
    /// Gradient threshold for discontinuity detection (relative to y-range).
    pub discontinuity_threshold: f64,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            min_points: 200,
            max_depth: 8,
            tolerance: 0.01,
            jitter: true,
            discontinuity_threshold: 50.0,
        }
    }
}

/// Deterministic jitter based on the bit pattern of `x` (reproducible).
fn jitter(x: f64, interval_width: f64) -> f64 {
    let hash = (x.to_bits())
        .wrapping_mul(6364136223846793005u64)
        .wrapping_add(1);
    let offset = ((hash >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.02 * interval_width;
    x + offset
}

/// Estimate the minimum number of sample points based on oscillation frequency.
///
/// Given a frequency in rad/s and a range `[lower, upper]`, returns the
/// number of initial sample points needed for roughly 20 points per cycle
/// (visual smoothness).
pub(crate) fn min_points_for_frequency(freq_rad_per_sec: f64, range: Interval<f64>) -> usize {
    let interval = range.width();
    let cycles = freq_rad_per_sec * interval / (2.0 * PI);
    let min = (cycles * 20.0) as usize;
    min.max(64)
}

/// Returns `true` if `x` is within `eps` of any value in `excluded`.
fn near_excluded(x: f64, excluded: &[f64], eps: f64) -> bool {
    excluded.iter().any(|&ex| (x - ex).abs() < eps)
}

/// Sample a compiled function over a range, with adaptive refinement.
///
/// `f` is the function to sample; `range` is the closed `[x_min, x_max]`;
/// `excluded_points` lists x-values where the function is known to be undefined
/// (e.g. from singularity analysis); `opts` controls sampling behaviour.
pub(crate) fn sample_compiled(
    f: &dyn Fn(f64) -> f64,
    range: Interval<f64>,
    excluded_points: &[f64],
    opts: &SampleOptions,
) -> PlotData {
    let (x_min, x_max) = (range.lower, range.upper);
    let interval = x_max - x_min;
    assert!(interval > 0.0, "range must be non-empty (x_min < x_max)");

    let n = opts.min_points.max(2);
    let eps = interval * 1e-9;

    // ── Step 1: Generate initial grid ──────────────────────────────────
    let mut samples: Vec<(f64, f64)> = Vec::with_capacity(n + excluded_points.len());
    let mut recorded_excluded: Vec<f64> = Vec::new();

    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let mut x = x_min + t * interval;

        if opts.jitter && i != 0 && i != n - 1 {
            x = jitter(x, interval / n as f64);
            // Clamp to range
            x = x.clamp(x_min, x_max);
        }

        if near_excluded(x, excluded_points, eps) {
            recorded_excluded.push(x);
            continue;
        }

        let y = f(x);
        if !y.is_finite() {
            recorded_excluded.push(x);
            continue;
        }
        samples.push((x, y));
    }

    // Sort by x (jitter may have slightly reordered neighbours)
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    tracing::debug!(
        initial_points = samples.len(),
        excluded = recorded_excluded.len(),
        "initial grid generated"
    );

    // ── Step 2: Adaptive refinement ────────────────────────────────────
    let y_range = compute_y_range(&samples);

    let mut refined = Vec::with_capacity(samples.len() * 2);
    if samples.len() >= 3 {
        // Process first point
        refined.push(samples[0]);
        for i in 0..samples.len() - 2 {
            let p1 = samples[i];
            let p3 = samples[i + 2];
            adaptive_refine(
                f,
                p1,
                samples[i + 1],
                p3,
                &mut refined,
                excluded_points,
                &mut recorded_excluded,
                opts,
                y_range,
                eps,
                0,
            );
            refined.push(samples[i + 1]);
        }
        // Push the last point
        if let Some(&last) = samples.last() {
            refined.push(last);
        }
    } else {
        refined = samples;
    }

    // De-duplicate and re-sort
    refined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    refined.dedup_by(|a, b| (a.0 - b.0).abs() < eps);

    tracing::debug!(total_points = refined.len(), "adaptive refinement complete");

    // ── Step 3: Discontinuity detection ────────────────────────────────
    let mut asymptotes: Vec<f64> = Vec::new();
    let final_points = detect_discontinuities(&refined, opts, &mut asymptotes);

    // Also add excluded_points that fall inside the range as asymptotes
    for &ex in excluded_points {
        if ex > x_min && ex < x_max && !asymptotes.iter().any(|&a| (a - ex).abs() < eps) {
            asymptotes.push(ex);
        }
    }
    asymptotes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    PlotData {
        points: final_points,
        asymptotes,
        excluded: recorded_excluded,
    }
}

/// Compute the absolute y-range of finite sample values.
fn compute_y_range(samples: &[(f64, f64)]) -> f64 {
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &(_, y) in samples {
        if y.is_finite() {
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }
    (y_max - y_min).abs().max(1e-10)
}

/// Recursively refine between two consecutive sample points if the midpoint
/// deviates significantly from linear interpolation.
#[allow(clippy::too_many_arguments)]
fn adaptive_refine(
    f: &dyn Fn(f64) -> f64,
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    out: &mut Vec<(f64, f64)>,
    excluded_points: &[f64],
    recorded_excluded: &mut Vec<f64>,
    opts: &SampleOptions,
    y_range: f64,
    eps: f64,
    depth: usize,
) {
    if depth >= opts.max_depth {
        return;
    }

    // Refine between p1 and p2
    refine_segment(
        f,
        p1,
        p2,
        out,
        excluded_points,
        recorded_excluded,
        opts,
        y_range,
        eps,
        depth,
    );

    // Refine between p2 and p3
    refine_segment(
        f,
        p2,
        p3,
        out,
        excluded_points,
        recorded_excluded,
        opts,
        y_range,
        eps,
        depth,
    );
}

#[allow(clippy::too_many_arguments)]
fn refine_segment(
    f: &dyn Fn(f64) -> f64,
    p1: (f64, f64),
    p3: (f64, f64),
    out: &mut Vec<(f64, f64)>,
    excluded_points: &[f64],
    recorded_excluded: &mut Vec<f64>,
    opts: &SampleOptions,
    y_range: f64,
    eps: f64,
    depth: usize,
) {
    if depth >= opts.max_depth {
        return;
    }

    let xm = (p1.0 + p3.0) / 2.0;
    if (p3.0 - p1.0).abs() < eps {
        return;
    }

    if near_excluded(xm, excluded_points, eps) {
        return;
    }

    let ym = f(xm);
    if !ym.is_finite() {
        recorded_excluded.push(xm);
        return;
    }

    let y_interp = (p1.1 + p3.1) / 2.0;
    let deviation = (ym - y_interp).abs() / y_range.max(1e-10);

    if deviation > opts.tolerance {
        let pm = (xm, ym);
        // Recurse left half
        refine_segment(
            f,
            p1,
            pm,
            out,
            excluded_points,
            recorded_excluded,
            opts,
            y_range,
            eps,
            depth + 1,
        );
        out.push(pm);
        // Recurse right half
        refine_segment(
            f,
            pm,
            p3,
            out,
            excluded_points,
            recorded_excluded,
            opts,
            y_range,
            eps,
            depth + 1,
        );
    }
}

/// Post-processing: detect discontinuities via gradient analysis and insert
/// NaN breaks. Returns the processed point list.
fn detect_discontinuities(
    points: &[(f64, f64)],
    opts: &SampleOptions,
    asymptotes: &mut Vec<f64>,
) -> Vec<(f64, f64)> {
    if points.len() < 2 {
        return points.to_vec();
    }

    // Compute gradients
    let mut gradients: Vec<f64> = Vec::with_capacity(points.len() - 1);
    for i in 0..points.len() - 1 {
        let dx = (points[i + 1].0 - points[i].0).abs();
        if dx < 1e-15 {
            gradients.push(0.0);
        } else {
            let dy = (points[i + 1].1 - points[i].1).abs();
            gradients.push(dy / dx);
        }
    }

    // Compute median gradient
    let median_gradient = {
        let mut sorted = gradients.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) && sorted.len() >= 2 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };

    let threshold = opts.discontinuity_threshold * median_gradient.max(1e-10);

    // Build output with NaN breaks at discontinuities
    let mut result: Vec<(f64, f64)> = Vec::with_capacity(points.len() + 10);
    result.push(points[0]);

    for i in 0..gradients.len() {
        if gradients[i] > threshold {
            // Detect sign change
            let y1 = points[i].1;
            let y2 = points[i + 1].1;
            let sign_change = (y1 >= 0.0) != (y2 >= 0.0);
            let large_jump = (y2 - y1).abs() > compute_y_range(points) * 0.5;

            if sign_change || large_jump {
                // Insert NaN break
                let x_break = (points[i].0 + points[i + 1].0) / 2.0;
                result.push((x_break, f64::NAN));
                asymptotes.push(x_break);
            }
        }
        result.push(points[i + 1]);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = SampleOptions::default();
        assert_eq!(opts.min_points, 200);
        assert_eq!(opts.max_depth, 8);
        assert!((opts.tolerance - 0.01).abs() < 1e-12);
        assert!(opts.jitter);
        assert!((opts.discontinuity_threshold - 50.0).abs() < 1e-12);
    }

    #[test]
    fn test_jitter_deterministic() {
        let a = jitter(1.0, 0.01);
        let b = jitter(1.0, 0.01);
        assert_eq!(a, b);
    }

    #[test]
    fn test_min_points_for_frequency_basic() {
        let pts = min_points_for_frequency(100.0, Interval::closed(0.0, 2.0 * PI));
        // 100 rad/s over 2π → 100 cycles → 2000 points
        assert!(pts >= 2000);
    }

    #[test]
    fn test_near_excluded() {
        assert!(near_excluded(1.0, &[1.0 + 1e-12], 1e-9));
        assert!(!near_excluded(1.0, &[2.0], 1e-9));
    }
}
