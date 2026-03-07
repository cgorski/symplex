//! Self-contained SVG string generation for plotting symbolic expressions.
//!
//! Produces clean, readable SVG markup with axes, grid lines, tick labels,
//! titles, and multiple series support. NaN gaps in data are handled by
//! splitting polylines.

use std::fmt::Write;

/// Options controlling the appearance of an SVG plot.
pub struct SvgPlotOptions {
    /// SVG width in pixels (default 600).
    pub width: f64,
    /// SVG height in pixels (default 400).
    pub height: f64,
    /// Margin for axes and labels (default 60).
    pub margin: f64,
    /// Optional plot title.
    pub title: Option<String>,
    /// Optional x-axis label.
    pub xlabel: Option<String>,
    /// Optional y-axis label.
    pub ylabel: Option<String>,
    /// Whether to draw grid lines (default true).
    pub grid: bool,
    /// Colors for each series (cycles if fewer than series count).
    pub colors: Vec<String>,
}

impl Default for SvgPlotOptions {
    fn default() -> Self {
        Self {
            width: 600.0,
            height: 400.0,
            margin: 60.0,
            title: None,
            xlabel: None,
            ylabel: None,
            grid: true,
            colors: vec![
                "#2266cc".to_string(),
                "#cc4422".to_string(),
                "#22aa44".to_string(),
                "#aa44cc".to_string(),
                "#ccaa22".to_string(),
                "#44aacc".to_string(),
            ],
        }
    }
}

// ── Heckbert "nice numbers" tick algorithm ─────────────────────────────

/// Compute a "nice" number close to `x`.
///
/// If `round` is true, round to the nearest nice number; otherwise, ceiling.
fn nice_num(x: f64, round: bool) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let exp = x.abs().log10().floor();
    let frac = x.abs() / 10f64.powf(exp);
    let nice = if round {
        if frac < 1.5 {
            1.0
        } else if frac < 3.0 {
            2.0
        } else if frac < 7.0 {
            5.0
        } else {
            10.0
        }
    } else if frac <= 1.0 {
        1.0
    } else if frac <= 2.0 {
        2.0
    } else if frac <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * 10f64.powf(exp)
}

/// Compute nice axis bounds and tick spacing for a data range.
///
/// Returns `(graph_min, graph_max, tick_step)`.
fn nice_ticks(data_min: f64, data_max: f64, n_ticks: usize) -> (f64, f64, f64) {
    let range = data_max - data_min;
    if range.abs() < 1e-15 {
        // Degenerate: constant data — fabricate a range around the value
        let center = data_min;
        let half = if center.abs() < 1e-15 {
            1.0
        } else {
            center.abs() * 0.1
        };
        return (center - half, center + half, half);
    }
    let nice_range = nice_num(range, false);
    let d = nice_num(nice_range / (n_ticks as f64 - 1.0), true);
    if d.abs() < 1e-18 {
        return (data_min, data_max, range);
    }
    let graph_min = (data_min / d).floor() * d;
    let graph_max = (data_max / d).ceil() * d;
    (graph_min, graph_max, d)
}

// ── Formatting helpers ─────────────────────────────────────────────────

/// Format a coordinate value to a compact string (1 decimal or integer).
fn fmt_coord(v: f64) -> String {
    if (v - v.round()).abs() < 0.05 {
        format!("{:.0}", v)
    } else {
        format!("{:.1}", v)
    }
}

/// Format a tick label value compactly.
fn fmt_tick(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if v.abs() >= 1e5 || v.abs() < 1e-3 {
        return format!("{:.1e}", v);
    }
    // Try integer first
    if (v - v.round()).abs() < 1e-9 {
        return format!("{:.0}", v);
    }
    // Up to 4 decimal places, strip trailing zeros
    let s = format!("{:.4}", v);
    let s = s.trim_end_matches('0');
    let s = s.trim_end_matches('.');
    s.to_string()
}

/// Escape text for safe embedding in SVG/XML.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── Main renderer ──────────────────────────────────────────────────────

/// Render plot data as an SVG string.
///
/// `series` is a slice of `(points, label)` pairs. Each point set is a
/// slice of `(x, y)` values — `NaN` y-values cause line breaks.
pub(crate) fn svg_plot(series: &[(&[(f64, f64)], &str)], opts: &SvgPlotOptions) -> String {
    let mut svg = String::with_capacity(4096);

    let margin = opts.margin;
    let plot_w = opts.width - 2.0 * margin;
    let plot_h = opts.height - 2.0 * margin;

    // ── 1. Compute data bounds across all series ───────────────────
    let (x_min, x_max, y_min, y_max) = data_bounds(series);

    // ── 2. Compute nice ticks ──────────────────────────────────────
    let (gx_min, gx_max, dx) = nice_ticks(x_min, x_max, 8);
    let (gy_min, gy_max, dy) = nice_ticks(y_min, y_max, 6);

    // ── 3. Affine transform helpers ────────────────────────────────
    let x_range = gx_max - gx_min;
    let y_range = gy_max - gy_min;
    let sx = if x_range.abs() > 1e-18 {
        plot_w / x_range
    } else {
        1.0
    };
    let sy = if y_range.abs() > 1e-18 {
        plot_h / y_range
    } else {
        1.0
    };

    let to_svg_x = |x: f64| -> f64 { margin + (x - gx_min) * sx };
    let to_svg_y = |y: f64| -> f64 { margin + (gy_max - y) * sy };

    // ── 4. SVG header ──────────────────────────────────────────────
    let _ = writeln!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"##,
        fmt_coord(opts.width),
        fmt_coord(opts.height),
        fmt_coord(opts.width),
        fmt_coord(opts.height),
    );

    // Background
    let _ = writeln!(
        svg,
        r##"  <rect width="{}" height="{}" fill="white"/>"##,
        fmt_coord(opts.width),
        fmt_coord(opts.height),
    );

    // ── 5. Grid lines ──────────────────────────────────────────────
    if opts.grid {
        emit_grid(
            &mut svg,
            &to_svg_x,
            &to_svg_y,
            &GridParams {
                margin,
                plot_w,
                plot_h,
                gx_min,
                gx_max,
                dx,
                gy_min,
                gy_max,
                dy,
            },
        );
    }

    // ── 6. Axes lines ──────────────────────────────────────────────
    // Plot area border
    let _ = writeln!(
        svg,
        r##"  <rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#333" stroke-width="1"/>"##,
        fmt_coord(margin),
        fmt_coord(margin),
        fmt_coord(plot_w),
        fmt_coord(plot_h),
    );

    // Zero lines (if visible)
    if gx_min <= 0.0 && gx_max >= 0.0 {
        let zx = to_svg_x(0.0);
        let _ = writeln!(
            svg,
            r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#666" stroke-width="0.8" stroke-dasharray="4,2"/>"##,
            fmt_coord(zx),
            fmt_coord(margin),
            fmt_coord(zx),
            fmt_coord(margin + plot_h),
        );
    }
    if gy_min <= 0.0 && gy_max >= 0.0 {
        let zy = to_svg_y(0.0);
        let _ = writeln!(
            svg,
            r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#666" stroke-width="0.8" stroke-dasharray="4,2"/>"##,
            fmt_coord(margin),
            fmt_coord(zy),
            fmt_coord(margin + plot_w),
            fmt_coord(zy),
        );
    }

    // ── 7. Tick marks and labels ───────────────────────────────────
    emit_x_ticks(&mut svg, &to_svg_x, margin, plot_h, gx_min, gx_max, dx);
    emit_y_ticks(&mut svg, &to_svg_y, margin, gy_min, gy_max, dy);

    // ── 8. Series polylines ────────────────────────────────────────
    for (i, (points, label)) in series.iter().enumerate() {
        let color = if opts.colors.is_empty() {
            "#2266cc"
        } else {
            &opts.colors[i % opts.colors.len()]
        };
        emit_series(&mut svg, points, color, label, &to_svg_x, &to_svg_y);
    }

    // ── 9. Legend (if multiple series) ──────────────────────────────
    if series.len() > 1 {
        emit_legend(&mut svg, series, opts, margin, plot_w);
    }

    // ── 10. Title, xlabel, ylabel ──────────────────────────────────
    if let Some(ref title) = opts.title {
        let _ = writeln!(
            svg,
            r##"  <text x="{}" y="{}" text-anchor="middle" font-size="16" font-family="sans-serif" font-weight="bold">{}</text>"##,
            fmt_coord(opts.width / 2.0),
            fmt_coord(margin * 0.5),
            xml_escape(title),
        );
    }
    if let Some(ref xlabel) = opts.xlabel {
        let _ = writeln!(
            svg,
            r##"  <text x="{}" y="{}" text-anchor="middle" font-size="12" font-family="sans-serif">{}</text>"##,
            fmt_coord(opts.width / 2.0),
            fmt_coord(opts.height - 5.0),
            xml_escape(xlabel),
        );
    }
    if let Some(ref ylabel) = opts.ylabel {
        let _ = writeln!(
            svg,
            r##"  <text x="{}" y="{}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90,{},{})">{}</text>"##,
            fmt_coord(15.0),
            fmt_coord(opts.height / 2.0),
            fmt_coord(15.0),
            fmt_coord(opts.height / 2.0),
            xml_escape(ylabel),
        );
    }

    // ── 11. Close SVG ──────────────────────────────────────────────
    svg.push_str("</svg>\n");
    svg
}

// ── Internal helpers ───────────────────────────────────────────────────

/// Compute data bounds across all series, skipping NaN values.
fn data_bounds(series: &[(&[(f64, f64)], &str)]) -> (f64, f64, f64, f64) {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for (points, _) in series {
        for &(x, y) in *points {
            if x.is_finite() {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
            }
            if y.is_finite() {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
    }

    // Fallback for empty data
    if !x_min.is_finite() || !x_max.is_finite() {
        x_min = 0.0;
        x_max = 1.0;
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    // Ensure non-degenerate
    if (x_max - x_min).abs() < 1e-15 {
        x_min -= 0.5;
        x_max += 0.5;
    }
    if (y_max - y_min).abs() < 1e-15 {
        y_min -= 0.5;
        y_max += 0.5;
    }
    (x_min, x_max, y_min, y_max)
}

/// Parameters for grid rendering, bundled to avoid too many function arguments.
struct GridParams {
    margin: f64,
    plot_w: f64,
    plot_h: f64,
    gx_min: f64,
    gx_max: f64,
    dx: f64,
    gy_min: f64,
    gy_max: f64,
    dy: f64,
}

/// Emit grid lines.
fn emit_grid(
    svg: &mut String,
    to_svg_x: &dyn Fn(f64) -> f64,
    to_svg_y: &dyn Fn(f64) -> f64,
    g: &GridParams,
) {
    svg.push_str("  <!-- grid -->\n");

    // Vertical grid lines
    let mut x = g.gx_min;
    let max_iter = 200;
    let mut count = 0;
    while x <= g.gx_max + g.dx * 0.001 && count < max_iter {
        let sx = to_svg_x(x);
        if sx >= g.margin - 0.5 && sx <= g.margin + g.plot_w + 0.5 {
            let _ = writeln!(
                svg,
                r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#ddd" stroke-width="0.5"/>"##,
                fmt_coord(sx),
                fmt_coord(g.margin),
                fmt_coord(sx),
                fmt_coord(g.margin + g.plot_h),
            );
        }
        x += g.dx;
        count += 1;
    }

    // Horizontal grid lines
    let mut y = g.gy_min;
    count = 0;
    while y <= g.gy_max + g.dy * 0.001 && count < max_iter {
        let sy = to_svg_y(y);
        if sy >= g.margin - 0.5 && sy <= g.margin + g.plot_h + 0.5 {
            let _ = writeln!(
                svg,
                r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#ddd" stroke-width="0.5"/>"##,
                fmt_coord(g.margin),
                fmt_coord(sy),
                fmt_coord(g.margin + g.plot_w),
                fmt_coord(sy),
            );
        }
        y += g.dy;
        count += 1;
    }
}

/// Emit x-axis tick marks and labels.
fn emit_x_ticks(
    svg: &mut String,
    to_svg_x: &dyn Fn(f64) -> f64,
    margin: f64,
    plot_h: f64,
    gx_min: f64,
    gx_max: f64,
    dx: f64,
) {
    svg.push_str("  <!-- x ticks -->\n");
    let mut x = gx_min;
    let bottom = margin + plot_h;
    let max_iter = 200;
    let mut count = 0;
    while x <= gx_max + dx * 0.001 && count < max_iter {
        let sx = to_svg_x(x);
        // Tick mark
        let _ = writeln!(
            svg,
            r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#333" stroke-width="1"/>"##,
            fmt_coord(sx),
            fmt_coord(bottom),
            fmt_coord(sx),
            fmt_coord(bottom + 5.0),
        );
        // Label
        let _ = writeln!(
            svg,
            r##"  <text x="{}" y="{}" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#333">{}</text>"##,
            fmt_coord(sx),
            fmt_coord(bottom + 16.0),
            xml_escape(&fmt_tick(x)),
        );
        x += dx;
        count += 1;
    }
}

/// Emit y-axis tick marks and labels.
fn emit_y_ticks(
    svg: &mut String,
    to_svg_y: &dyn Fn(f64) -> f64,
    margin: f64,
    gy_min: f64,
    gy_max: f64,
    dy: f64,
) {
    svg.push_str("  <!-- y ticks -->\n");
    let mut y = gy_min;
    let max_iter = 200;
    let mut count = 0;
    while y <= gy_max + dy * 0.001 && count < max_iter {
        let sy = to_svg_y(y);
        // Tick mark
        let _ = writeln!(
            svg,
            r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#333" stroke-width="1"/>"##,
            fmt_coord(margin - 5.0),
            fmt_coord(sy),
            fmt_coord(margin),
            fmt_coord(sy),
        );
        // Label
        let _ = writeln!(
            svg,
            r##"  <text x="{}" y="{}" text-anchor="end" font-size="10" font-family="sans-serif" fill="#333">{}</text>"##,
            fmt_coord(margin - 8.0),
            fmt_coord(sy + 3.5),
            xml_escape(&fmt_tick(y)),
        );
        y += dy;
        count += 1;
    }
}

/// Emit a single series as one or more `<polyline>` elements, breaking at NaN gaps.
fn emit_series(
    svg: &mut String,
    points: &[(f64, f64)],
    color: &str,
    label: &str,
    to_svg_x: &dyn Fn(f64) -> f64,
    to_svg_y: &dyn Fn(f64) -> f64,
) {
    // Collect segments (runs of finite points)
    let segments = split_nan_segments(points);
    if segments.is_empty() {
        return;
    }

    let _ = writeln!(svg, "  <!-- series: {} -->", xml_escape(label));

    for seg in &segments {
        if seg.is_empty() {
            continue;
        }
        let mut pts_str = String::with_capacity(seg.len() * 16);
        for (i, &(x, y)) in seg.iter().enumerate() {
            if i > 0 {
                pts_str.push(' ');
            }
            let _ = write!(
                pts_str,
                "{},{}",
                fmt_coord(to_svg_x(x)),
                fmt_coord(to_svg_y(y))
            );
        }
        let _ = writeln!(
            svg,
            r##"  <polyline points="{}" fill="none" stroke="{}" stroke-width="2" stroke-linejoin="round"/>"##,
            pts_str,
            xml_escape(color),
        );
    }
}

/// Split a point list into segments separated by NaN y-values.
fn split_nan_segments(points: &[(f64, f64)]) -> Vec<Vec<(f64, f64)>> {
    let mut segments: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut current: Vec<(f64, f64)> = Vec::new();

    for &(x, y) in points {
        if y.is_finite() && x.is_finite() {
            current.push((x, y));
        } else if !current.is_empty() {
            segments.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Emit a legend box in the upper right corner.
fn emit_legend(
    svg: &mut String,
    series: &[(&[(f64, f64)], &str)],
    opts: &SvgPlotOptions,
    margin: f64,
    plot_w: f64,
) {
    let line_h = 18.0;
    let box_w = 120.0;
    let box_h = series.len() as f64 * line_h + 10.0;
    let bx = margin + plot_w - box_w - 10.0;
    let by = margin + 10.0;

    let _ = writeln!(
        svg,
        r##"  <rect x="{}" y="{}" width="{}" height="{}" fill="white" fill-opacity="0.85" stroke="#999" stroke-width="0.5" rx="3"/>"##,
        fmt_coord(bx),
        fmt_coord(by),
        fmt_coord(box_w),
        fmt_coord(box_h),
    );

    for (i, (_, label)) in series.iter().enumerate() {
        let color = if opts.colors.is_empty() {
            "#2266cc"
        } else {
            &opts.colors[i % opts.colors.len()]
        };
        let ly = by + 8.0 + (i as f64) * line_h + line_h * 0.5;
        // Color swatch line
        let _ = writeln!(
            svg,
            r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="3"/>"##,
            fmt_coord(bx + 8.0),
            fmt_coord(ly),
            fmt_coord(bx + 28.0),
            fmt_coord(ly),
            xml_escape(color),
        );
        // Label
        let _ = writeln!(
            svg,
            r##"  <text x="{}" y="{}" font-size="11" font-family="sans-serif" fill="#333">{}</text>"##,
            fmt_coord(bx + 34.0),
            fmt_coord(ly + 4.0),
            xml_escape(label),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_num_rounding() {
        assert!((nice_num(12.0, true) - 10.0).abs() < 1e-10);
        assert!((nice_num(35.0, true) - 50.0).abs() < 1e-10);
        assert!((nice_num(75.0, true) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn nice_ticks_basic() {
        let (gmin, gmax, d) = nice_ticks(0.3, 9.7, 5);
        assert!(gmin <= 0.3);
        assert!(gmax >= 9.7);
        assert!(d > 0.0);
    }

    #[test]
    fn nice_ticks_degenerate() {
        let (gmin, gmax, d) = nice_ticks(5.0, 5.0, 5);
        assert!(gmin < gmax);
        assert!(d > 0.0);
    }

    #[test]
    fn split_nan_segments_basic() {
        let pts = vec![(0.0, 1.0), (1.0, 2.0), (2.0, f64::NAN), (3.0, 4.0)];
        let segs = split_nan_segments(&pts);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].len(), 2);
        assert_eq!(segs[1].len(), 1);
    }

    #[test]
    fn svg_basic_structure() {
        let pts: Vec<(f64, f64)> = (0..10)
            .map(|i| {
                let x = i as f64 * 0.1;
                (x, x.sin())
            })
            .collect();
        let series = vec![(pts.as_slice(), "sin(x)")];
        let opts = SvgPlotOptions::default();
        let svg = svg_plot(&series, &opts);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<polyline"));
    }

    #[test]
    fn svg_multi_series() {
        let pts1: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, (i as f64).sin())).collect();
        let pts2: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, (i as f64).cos())).collect();
        let series = vec![(pts1.as_slice(), "sin(x)"), (pts2.as_slice(), "cos(x)")];
        let opts = SvgPlotOptions::default();
        let svg = svg_plot(&series, &opts);
        let polyline_count = svg.matches("<polyline").count();
        assert!(
            polyline_count >= 2,
            "expected at least 2 polylines, got {polyline_count}"
        );
    }

    #[test]
    fn svg_with_title() {
        let pts: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 1.0)];
        let series = vec![(pts.as_slice(), "line")];
        let opts = SvgPlotOptions {
            title: Some("Test Title".to_string()),
            xlabel: Some("X".to_string()),
            ylabel: Some("Y".to_string()),
            ..Default::default()
        };
        let svg = svg_plot(&series, &opts);
        assert!(svg.contains("Test Title"));
    }

    #[test]
    fn fmt_tick_values() {
        assert_eq!(fmt_tick(0.0), "0");
        assert_eq!(fmt_tick(3.0), "3");
        assert_eq!(fmt_tick(1.5), "1.5");
    }

    #[test]
    fn xml_escape_works() {
        assert_eq!(xml_escape("a<b>c"), "a&lt;b&gt;c");
        assert_eq!(xml_escape("a&b"), "a&amp;b");
    }

    #[test]
    fn data_bounds_skips_nan() {
        let pts: Vec<(f64, f64)> = vec![(0.0, 1.0), (1.0, f64::NAN), (2.0, 3.0)];
        let series = vec![(pts.as_slice(), "test")];
        let (x_min, x_max, y_min, y_max) = data_bounds(&series);
        assert!((x_min - 0.0).abs() < 1e-10);
        assert!((x_max - 2.0).abs() < 1e-10);
        assert!((y_min - 1.0).abs() < 1e-10);
        assert!((y_max - 3.0).abs() < 1e-10);
    }
}
