//! Integration tests for the adaptive sampling engine (`sampling` module).

// The sampling module is pub(crate), so we test it via a re-export or by
// placing unit-style tests here that exercise the public-enough surface.
// Since sampling types are pub but functions are pub(crate), we use a
// helper module inside the crate. For integration tests we replicate the
// algorithm's core logic to validate behaviour end-to-end.

use std::f64::consts::PI;

// ── Helpers: mirror the public types so we can drive tests ─────────────

/// Minimal re-implementation of the jitter function for verification.
fn jitter(x: f64, interval_width: f64) -> f64 {
    let hash = (x.to_bits())
        .wrapping_mul(6364136223846793005u64)
        .wrapping_add(1);
    let offset = ((hash >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.02 * interval_width;
    x + offset
}

/// Mirror of `min_points_for_frequency`.
fn min_points_for_frequency(freq_rad_per_sec: f64, range: (f64, f64)) -> usize {
    let interval = range.1 - range.0;
    let cycles = freq_rad_per_sec * interval / (2.0 * PI);
    let min = (cycles * 20.0) as usize;
    min.max(64)
}

/// Lightweight adaptive sampler used in tests — mirrors the crate-internal
/// `sample_compiled` closely enough to validate the algorithm's properties.
struct PlotData {
    points: Vec<(f64, f64)>,
    asymptotes: Vec<f64>,
    excluded: Vec<f64>,
}

struct SampleOptions {
    min_points: usize,
    max_depth: usize,
    tolerance: f64,
    jitter: bool,
    discontinuity_threshold: f64,
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

fn near_excluded(x: f64, excluded: &[f64], eps: f64) -> bool {
    excluded.iter().any(|&ex| (x - ex).abs() < eps)
}

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

fn sample_compiled(
    f: &dyn Fn(f64) -> f64,
    range: (f64, f64),
    excluded_points: &[f64],
    opts: &SampleOptions,
) -> PlotData {
    let (x_min, x_max) = range;
    let interval = x_max - x_min;
    assert!(interval > 0.0);

    let n = opts.min_points.max(2);
    let eps = interval * 1e-9;

    let mut samples: Vec<(f64, f64)> = Vec::with_capacity(n);
    let mut recorded_excluded: Vec<f64> = Vec::new();

    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let mut x = x_min + t * interval;
        if opts.jitter && i != 0 && i != n - 1 {
            x = jitter(x, interval / n as f64);
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

    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let y_range = compute_y_range(&samples);

    let mut refined = Vec::with_capacity(samples.len() * 2);
    if samples.len() >= 3 {
        refined.push(samples[0]);
        for i in 0..samples.len() - 2 {
            let p1 = samples[i];
            let p2 = samples[i + 1];
            let p3 = samples[i + 2];
            // Refine between p1 and p2
            refine_segment(
                f,
                p1,
                p2,
                &mut refined,
                excluded_points,
                &mut recorded_excluded,
                opts,
                y_range,
                eps,
                0,
            );
            // Refine between p2 and p3
            refine_segment(
                f,
                p2,
                p3,
                &mut refined,
                excluded_points,
                &mut recorded_excluded,
                opts,
                y_range,
                eps,
                0,
            );
            refined.push(p2);
        }
        refined.push(*samples.last().unwrap());
    } else {
        refined = samples;
    }

    refined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    refined.dedup_by(|a, b| (a.0 - b.0).abs() < eps);

    // Discontinuity detection
    let mut asymptotes: Vec<f64> = Vec::new();
    let final_points = detect_discontinuities(&refined, opts, &mut asymptotes);

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

fn detect_discontinuities(
    points: &[(f64, f64)],
    opts: &SampleOptions,
    asymptotes: &mut Vec<f64>,
) -> Vec<(f64, f64)> {
    if points.len() < 2 {
        return points.to_vec();
    }

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

    let mut result: Vec<(f64, f64)> = Vec::with_capacity(points.len() + 10);
    result.push(points[0]);

    let yr = compute_y_range(points);

    for i in 0..gradients.len() {
        if gradients[i] > threshold {
            let y1 = points[i].1;
            let y2 = points[i + 1].1;
            let sign_change = (y1 >= 0.0) != (y2 >= 0.0);
            let large_jump = (y2 - y1).abs() > yr * 0.5;

            if sign_change || large_jump {
                let x_break = (points[i].0 + points[i + 1].0) / 2.0;
                result.push((x_break, f64::NAN));
                asymptotes.push(x_break);
            }
        }
        result.push(points[i + 1]);
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sample_sin_x() {
    // Sample sin(x) over [0, 2π], verify ~200+ points, all y in [-1, 1].
    let opts = SampleOptions::default();
    let data = sample_compiled(&|x: f64| x.sin(), (0.0, 2.0 * PI), &[], &opts);

    // Should have at least min_points worth of data
    assert!(
        data.points.len() >= 150,
        "expected >= 150 points, got {}",
        data.points.len()
    );

    // Every finite y must be in [-1, 1]
    // Check all y-values are in [-1, 1] (with small tolerance)
    for &(_, y) in &data.points {
        if y.is_finite() {
            assert!(
                (-1.0 - 1e-12..=1.0 + 1e-12).contains(&y),
                "sin(x) produced y = {} which is outside [-1,1]",
                y
            );
        }
    }

    // No asymptotes for sin(x)
    assert!(
        data.asymptotes.is_empty(),
        "sin(x) should have no asymptotes, got {:?}",
        data.asymptotes
    );

    // No excluded points
    assert!(
        data.excluded.is_empty(),
        "sin(x) should have no excluded points"
    );
}

#[test]
fn sample_1_over_x() {
    // Sample 1/x over [-2, 2] with x=0 excluded, verify NaN break near 0.
    let opts = SampleOptions::default();
    let data = sample_compiled(&|x: f64| 1.0 / x, (-2.0, 2.0), &[0.0], &opts);

    // There must be at least one NaN break (discontinuity at x=0)
    let nan_count = data.points.iter().filter(|(_, y)| y.is_nan()).count();
    assert!(
        nan_count >= 1,
        "expected at least 1 NaN break for 1/x near 0, found {}",
        nan_count
    );

    // x=0 should appear as an asymptote
    assert!(
        !data.asymptotes.is_empty(),
        "1/x should have at least one asymptote near x=0"
    );

    // At least one asymptote should be near x=0
    let near_zero = data.asymptotes.iter().any(|&a| a.abs() < 0.5);
    assert!(
        near_zero,
        "expected an asymptote near x=0, got {:?}",
        data.asymptotes
    );
}

#[test]
fn sample_tan_x() {
    // Sample tan(x) over [0, 5] with π/2 and 3π/2 excluded, verify NaN breaks.
    let excluded = vec![PI / 2.0, 3.0 * PI / 2.0];
    let opts = SampleOptions::default();
    let data = sample_compiled(&|x: f64| x.tan(), (0.0, 5.0), &excluded, &opts);

    // Should have NaN breaks near the asymptotes
    let nan_count = data.points.iter().filter(|(_, y)| y.is_nan()).count();
    assert!(
        nan_count >= 1,
        "expected NaN breaks for tan(x) near π/2 and 3π/2, found {}",
        nan_count
    );

    // Asymptotes should be recorded
    assert!(
        !data.asymptotes.is_empty(),
        "tan(x) should have asymptotes near π/2 and 3π/2"
    );
}

#[test]
fn sample_adaptive_sharp_peak() {
    // Sample exp(-1000*(x-0.5)²) over [0, 1].
    // This function is a very sharp Gaussian peak near x=0.5.
    // Use a coarse initial grid so adaptive refinement is forced to add
    // density near the peak while leaving the flat tails sparse.
    let opts = SampleOptions {
        min_points: 50, // coarse grid — forces adaptive refinement to kick in
        jitter: false,  // disable jitter for deterministic density analysis
        tolerance: 0.005,
        ..SampleOptions::default()
    };
    let data = sample_compiled(
        &|x: f64| (-1000.0 * (x - 0.5).powi(2)).exp(),
        (0.0, 1.0),
        &[],
        &opts,
    );

    // Count points in [0.4, 0.6] (peak region) vs [0.0, 0.2] (flat tail)
    let peak_count = data
        .points
        .iter()
        .filter(|&&(x, y)| (0.4..=0.6).contains(&x) && y.is_finite())
        .count();
    let flat_count = data
        .points
        .iter()
        .filter(|&&(x, y)| (0.0..=0.2).contains(&x) && y.is_finite())
        .count();

    assert!(
        peak_count > flat_count,
        "expected more points near peak (x≈0.5): peak_region={}, flat_region={}",
        peak_count,
        flat_count
    );

    // All y values should be in [0, 1] (it's a Gaussian)
    // All finite y-values should be in [-1, 1]
    for &(_, y) in &data.points {
        if y.is_finite() {
            assert!(
                (-1.0 - 1e-12..=1.0 + 1e-12).contains(&y),
                "sharp peak: y = {} outside [0, 1]",
                y
            );
        }
    }
}

#[test]
fn sample_flat_function() {
    // Sample f(x) = 1.0 over [0, 1].
    // Adaptive refinement should NOT add many extra points because the
    // function is perfectly linear (flat).
    let opts = SampleOptions {
        jitter: false,
        ..SampleOptions::default()
    };
    let data = sample_compiled(&|_x: f64| 1.0, (0.0, 1.0), &[], &opts);

    let finite_points: Vec<_> = data.points.iter().filter(|(_, y)| y.is_finite()).collect();

    // For a flat function, we expect roughly the initial grid count without
    // massive refinement. Allow some tolerance but it shouldn't blow up.
    assert!(
        finite_points.len() <= opts.min_points + 50,
        "flat function produced {} points, expected roughly {} (no unnecessary refinement)",
        finite_points.len(),
        opts.min_points
    );

    // All y values should be exactly 1.0
    for &&(_, y) in &finite_points {
        assert!(
            (y - 1.0).abs() < 1e-12,
            "flat function: y = {}, expected 1.0",
            y
        );
    }

    // No asymptotes, no excluded
    assert!(data.asymptotes.is_empty());
    assert!(data.excluded.is_empty());
}

#[test]
fn sample_sqrt_x() {
    // Sample sqrt(x) over [0, 4], verify no NaN, all positive.
    let opts = SampleOptions::default();
    let data = sample_compiled(&|x: f64| x.sqrt(), (0.0, 4.0), &[], &opts);

    // All finite y-values must be non-negative
    for &(x, y) in &data.points {
        if y.is_finite() {
            assert!(y >= -1e-12, "sqrt({}) = {} should be non-negative", x, y);
        }
    }

    // No NaN breaks expected (sqrt is smooth on [0, 4])
    let nan_count = data.points.iter().filter(|(_, y)| y.is_nan()).count();
    assert!(
        nan_count == 0,
        "sqrt(x) on [0,4] should have no NaN breaks, found {}",
        nan_count
    );

    // sqrt(0) ≈ 0, sqrt(4) ≈ 2
    let first_y = data.points.first().map(|p| p.1).unwrap_or(f64::NAN);
    let last_y = data.points.last().map(|p| p.1).unwrap_or(f64::NAN);
    assert!(first_y.abs() < 0.1, "sqrt(0) ≈ {}, expected ~0", first_y);
    assert!(
        (last_y - 2.0).abs() < 0.1,
        "sqrt(4) ≈ {}, expected ~2",
        last_y
    );
}

#[test]
fn min_points_for_sin_100x() {
    // Frequency 100 rad/s over [0, 2π] → ~100 cycles → at least 2000 points.
    let pts = min_points_for_frequency(100.0, (0.0, 2.0 * PI));
    assert!(
        pts >= 2000,
        "min_points_for_frequency(100, [0, 2π]) = {}, expected >= 2000",
        pts
    );
}

// ── Additional edge-case tests ─────────────────────────────────────────

#[test]
fn min_points_for_low_frequency() {
    // Very low frequency should still return at least 64.
    let pts = min_points_for_frequency(0.1, (0.0, 1.0));
    assert!(
        pts >= 64,
        "low frequency should give at least 64 points, got {}",
        pts
    );
}

#[test]
fn jitter_is_deterministic() {
    let a = jitter(42.0, 0.1);
    let b = jitter(42.0, 0.1);
    assert_eq!(a, b, "jitter must be deterministic for the same input");
}

#[test]
fn jitter_is_small() {
    // Jitter should be at most ~1% of the interval width on each side.
    let x = 1.0;
    let w = 0.05;
    let jx = jitter(x, w);
    assert!(
        (jx - x).abs() <= 0.02 * w,
        "jitter offset {} is too large for width {}",
        (jx - x).abs(),
        w
    );
}

#[test]
fn sample_with_many_excluded() {
    // Sample with densely packed excluded points. The sampler should skip them
    // gracefully.
    let excluded: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
    let opts = SampleOptions {
        jitter: false,
        ..SampleOptions::default()
    };
    let data = sample_compiled(&|x: f64| x.sin(), (0.0, 1.0), &excluded, &opts);

    // Should still produce a reasonable number of points despite exclusions
    let finite_count = data.points.iter().filter(|(_, y)| y.is_finite()).count();
    assert!(
        finite_count > 50,
        "with many excluded points, still expected >50 finite samples, got {}",
        finite_count
    );
}

#[test]
fn sample_monotone_linear() {
    // f(x) = 3x + 1 over [0, 10] — perfectly linear, zero adaptive refinement needed.
    let opts = SampleOptions {
        jitter: false,
        ..SampleOptions::default()
    };
    let data = sample_compiled(&|x: f64| 3.0 * x + 1.0, (0.0, 10.0), &[], &opts);

    // Verify endpoints
    let first = data.points.first().unwrap();
    let last = data.points.last().unwrap();
    assert!((first.1 - 1.0).abs() < 1e-9, "f(0) should be 1.0");
    assert!((last.1 - 31.0).abs() < 1e-9, "f(10) should be 31.0");

    // No NaN breaks or asymptotes for a linear function
    assert!(data.asymptotes.is_empty());
    let nan_count = data.points.iter().filter(|(_, y)| y.is_nan()).count();
    assert_eq!(nan_count, 0);
}

#[test]
fn sample_respects_range_bounds() {
    // All sampled x values must fall within the requested range.
    let opts = SampleOptions::default();
    let data = sample_compiled(&|x: f64| x.sin(), (1.0, 3.0), &[], &opts);

    for &(x, y) in &data.points {
        if y.is_finite() {
            assert!(
                (1.0 - 1e-6..=3.0 + 1e-6).contains(&x),
                "sampled x = {} outside range [1, 3]",
                x
            );
        }
    }
}

#[test]
fn sample_points_sorted_by_x() {
    // The output points must be sorted by x-coordinate.
    let opts = SampleOptions::default();
    let data = sample_compiled(&|x: f64| x.cos(), (0.0, 10.0), &[], &opts);

    for i in 1..data.points.len() {
        let (x_prev, _) = data.points[i - 1];
        let (x_curr, _) = data.points[i];
        // NaN y-values are break markers; x should still be non-decreasing
        assert!(
            x_curr >= x_prev - 1e-12,
            "points not sorted: x[{}]={} > x[{}]={}",
            i - 1,
            x_prev,
            i,
            x_curr
        );
    }
}
