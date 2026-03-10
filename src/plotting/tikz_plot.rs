//! TikZ/PGFplots rendering backend.
//!
//! Emits self-contained PGFplots code suitable for inclusion in LaTeX documents.
//! PGFplots handles axes, ticks, and legends, so this module focuses on
//! correctly formatting coordinate data and axis options.

/// Render plot data as a TikZ/PGFplots code string.
///
/// Each entry in `series` is a `(points, label)` pair. NaN y-values trigger
/// a break: the current `\addplot` is closed and a new one begins (unlabeled)
/// so that discontinuities appear as gaps rather than wild lines.
///
/// `log_x` / `log_y` switch the corresponding axis to logarithmic mode.
pub(crate) fn tikz_plot(
    series: &[(&[(f64, f64)], &str)],
    title: Option<&str>,
    xlabel: Option<&str>,
    ylabel: Option<&str>,
    log_x: bool,
    log_y: bool,
) -> String {
    let colors = [
        "blue",
        "red",
        "green!60!black",
        "orange",
        "purple",
        "cyan",
        "brown",
        "magenta",
    ];

    let mut out = String::with_capacity(4096);
    out.push_str("\\begin{tikzpicture}\n");
    out.push_str("\\begin{axis}[\n");
    out.push_str("  width=10cm, height=6cm,\n");

    if let Some(t) = title {
        out.push_str(&format!("  title={{{}}},\n", latex_escape(t)));
    }
    if let Some(xl) = xlabel {
        out.push_str(&format!("  xlabel={{{}}},\n", latex_escape(xl)));
    }
    if let Some(yl) = ylabel {
        out.push_str(&format!("  ylabel={{{}}},\n", latex_escape(yl)));
    }

    out.push_str("  grid=major,\n");

    if log_x {
        out.push_str("  xmode=log,\n");
    } else {
        out.push_str("  xmode=normal,\n");
    }
    if log_y {
        out.push_str("  ymode=log,\n");
    } else {
        out.push_str("  ymode=normal,\n");
    }

    if series.len() > 1 {
        out.push_str("  legend pos=north east,\n");
    }

    out.push_str("]\n");

    for (si, &(points, label)) in series.iter().enumerate() {
        let color = colors[si % colors.len()];
        emit_series(&mut out, points, label, color, si == 0 || series.len() > 1);
    }

    out.push_str("\\end{axis}\n");
    out.push_str("\\end{tikzpicture}\n");
    out
}

/// Emit one logical series, splitting on NaN gaps.
fn emit_series(
    out: &mut String,
    points: &[(f64, f64)],
    label: &str,
    color: &str,
    show_legend: bool,
) {
    // Split points into contiguous segments (no NaN).
    let segments = split_on_nan(points);

    for (seg_idx, seg) in segments.iter().enumerate() {
        if seg.is_empty() {
            continue;
        }

        out.push_str(&format!(
            "\\addplot[{}, thick, no markers] coordinates {{\n",
            color
        ));

        for &(x, y) in seg {
            out.push_str(&format!("  ({}, {})\n", fmt_coord(x), fmt_coord(y)));
        }

        out.push_str("};\n");

        // Only add legend entry for the first segment of this series.
        if seg_idx == 0 && show_legend && !label.is_empty() {
            out.push_str(&format!("\\addlegendentry{{{}}}\n", latex_escape(label)));
        }
    }
}

/// Split a point slice into contiguous segments, breaking at NaN y-values.
fn split_on_nan(points: &[(f64, f64)]) -> Vec<Vec<(f64, f64)>> {
    let mut segments: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut current: Vec<(f64, f64)> = Vec::new();

    for &(x, y) in points {
        if y.is_nan() || x.is_nan() {
            if !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
        } else {
            current.push((x, y));
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Format a coordinate value with reasonable precision.
/// Integers get no decimal point; others get up to 6 significant decimals.
fn fmt_coord(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if (v - v.round()).abs() < 1e-10 && v.abs() < 1e12 {
        return format!("{:.0}", v);
    }
    // Strip trailing zeros.
    let s = format!("{:.6}", v);
    let s = s.trim_end_matches('0');
    let s = s.trim_end_matches('.');
    s.to_string()
}

/// Minimal LaTeX escaping for text content in titles/labels.
fn latex_escape(s: &str) -> String {
    s.replace('\\', "\\textbackslash{}")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('$', "\\$")
        .replace('#', "\\#")
        .replace('_', "\\_")
        .replace('^', "\\^{}")
        .replace('~', "\\~{}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tikz_output() {
        let pts: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, (i * i) as f64)).collect();
        let output = tikz_plot(
            &[(&pts, "x^2")],
            Some("Quadratic"),
            Some("x"),
            Some("y"),
            false,
            false,
        );
        assert!(output.contains("\\begin{tikzpicture}"));
        assert!(output.contains("\\end{tikzpicture}"));
        assert!(output.contains("\\begin{axis}"));
        assert!(output.contains("\\end{axis}"));
        assert!(output.contains("\\addplot"));
        assert!(output.contains("title={Quadratic}"));
        assert!(output.contains("xlabel={x}"));
        assert!(output.contains("ylabel={y}"));
        assert!(output.contains("\\addlegendentry{x\\^{}2}"));
    }

    #[test]
    fn nan_breaks_segments() {
        let pts = vec![
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, f64::NAN),
            (3.0, 4.0),
            (4.0, 5.0),
        ];
        let output = tikz_plot(&[(&pts, "f")], None, None, None, false, false);
        // Should have two \addplot commands (one per segment).
        let count = output.matches("\\addplot[").count();
        assert_eq!(
            count, 2,
            "expected 2 addplot blocks for NaN-split data, got {count}"
        );
    }

    #[test]
    fn log_axes() {
        let pts = vec![(1.0, 10.0), (10.0, 100.0), (100.0, 1000.0)];
        let output = tikz_plot(&[(&pts, "log")], None, None, None, true, true);
        assert!(output.contains("xmode=log"));
        assert!(output.contains("ymode=log"));
    }

    #[test]
    fn multiple_series() {
        let pts1: Vec<(f64, f64)> = (0..3).map(|i| (i as f64, i as f64)).collect();
        let pts2: Vec<(f64, f64)> = (0..3).map(|i| (i as f64, (i * 2) as f64)).collect();
        let output = tikz_plot(
            &[(&pts1, "linear"), (&pts2, "double")],
            None,
            None,
            None,
            false,
            false,
        );
        assert!(output.contains("\\addlegendentry{linear}"));
        assert!(output.contains("\\addlegendentry{double}"));
        // Two series ⇒ at least 2 addplot blocks.
        let count = output.matches("\\addplot[").count();
        assert!(count >= 2);
    }

    #[test]
    fn split_on_nan_basic() {
        let pts = vec![(0.0, 1.0), (1.0, f64::NAN), (2.0, 3.0)];
        let segs = split_on_nan(&pts);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].len(), 1);
        assert_eq!(segs[1].len(), 1);
    }

    #[test]
    fn fmt_coord_integers() {
        assert_eq!(fmt_coord(0.0), "0");
        assert_eq!(fmt_coord(3.0), "3");
        assert_eq!(fmt_coord(-7.0), "-7");
    }

    #[test]
    fn fmt_coord_decimals() {
        let s = fmt_coord(1.5);
        assert_eq!(s, "1.5");
    }

    #[test]
    fn empty_series() {
        let pts: Vec<(f64, f64)> = vec![];
        let output = tikz_plot(&[(&pts, "empty")], None, None, None, false, false);
        assert!(output.contains("\\begin{axis}"));
        assert!(output.contains("\\end{axis}"));
    }
}
