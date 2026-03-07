//! ASCII art terminal plot, modeled after SymPy's `textplot.py`.
//!
//! Renders sampled `(x, y)` data as a character grid suitable for
//! terminal output.

use std::fmt::Write;

/// Render an ASCII art plot of sampled data to a string.
///
/// `points` is a slice of `(x, y)` pairs.  NaN y-values are filtered out.
/// `width` is the character width of the plot area (default 60).
/// `height` is the character height of the plot area (default 21).
/// `title` is an optional title printed above the plot.
pub(crate) fn textplot(
    points: &[(f64, f64)],
    width: usize,
    height: usize,
    title: Option<&str>,
) -> String {
    let width = if width < 10 { 10 } else { width };
    let height = if height < 5 { 5 } else { height };

    // ── Step 1: Filter out NaN / infinite points ───────────────────
    let valid: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();

    if valid.is_empty() {
        return "[no valid data to plot]".to_string();
    }

    // ── Step 2: Compute data bounds ────────────────────────────────
    let x_min = valid.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = valid.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = valid.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = valid.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

    // Handle degenerate ranges
    let (x_min, x_max) = if (x_max - x_min).abs() < 1e-15 {
        (x_min - 1.0, x_max + 1.0)
    } else {
        (x_min, x_max)
    };
    let (y_min, y_max) = if (y_max - y_min).abs() < 1e-15 {
        (y_min - 1.0, y_max + 1.0)
    } else {
        (y_min, y_max)
    };

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    // ── Step 3: Create character grid ──────────────────────────────
    // grid[row][col] — row 0 is the top (y_max), row height-1 is bottom (y_min)
    let mut grid = vec![vec![' '; width]; height];

    // ── Step 4: Map points to grid coordinates and place markers ───
    // Collect grid-mapped points in order for segment slope analysis
    let mut mapped: Vec<(usize, usize)> = Vec::with_capacity(valid.len());

    for &(x, y) in &valid {
        let col_f = (x - x_min) / x_range * (width as f64 - 1.0);
        let row_f = (y_max - y) / y_range * (height as f64 - 1.0);

        let col = (col_f.round() as isize).clamp(0, width as isize - 1) as usize;
        let row = (row_f.round() as isize).clamp(0, height as isize - 1) as usize;

        mapped.push((row, col));
    }

    // ── Step 5: Place characters: '.' for points, '/' or '\' for steep segments
    for i in 0..mapped.len() {
        let (row, col) = mapped[i];
        if i > 0 {
            let (prev_row, prev_col) = mapped[i - 1];
            // Only draw slope chars for adjacent columns
            let col_diff = (col as isize - prev_col as isize).unsigned_abs();
            if col_diff <= 2 && col_diff > 0 {
                let row_diff = row as isize - prev_row as isize;
                if row_diff < -1 {
                    // y increasing (row decreasing) → '/'
                    grid[row][col] = '/';
                    continue;
                } else if row_diff > 1 {
                    // y decreasing (row increasing) → '\'
                    grid[row][col] = '\\';
                    continue;
                }
            }
        }
        grid[row][col] = '.';
    }

    // ── Step 6: Draw x-axis (underscore row at y=0 if visible) ─────
    if y_min <= 0.0 && y_max >= 0.0 {
        let zero_row_f = (y_max - 0.0) / y_range * (height as f64 - 1.0);
        let zero_row = (zero_row_f.round() as isize).clamp(0, height as isize - 1) as usize;
        for cell in grid[zero_row].iter_mut() {
            if *cell == ' ' {
                *cell = '_';
            }
        }
    }

    // ── Step 7: Build output with y-axis labels on left margin ─────
    let label_width = 10;
    let mut output = String::with_capacity((width + label_width + 4) * (height + 4));

    // Title
    if let Some(t) = title {
        let pad = (width + label_width + 3).saturating_sub(t.len()) / 2;
        let _ = writeln!(output, "{:>pad$}{}", "", t, pad = pad);
        output.push('\n');
    }

    for (row_idx, row_data) in grid.iter().enumerate() {
        // Y-axis label: show at top, middle, and bottom rows
        let label = if row_idx == 0 {
            format_num(y_max)
        } else if row_idx == height - 1 {
            format_num(y_min)
        } else if row_idx == height / 2 {
            format_num((y_min + y_max) / 2.0)
        } else {
            String::new()
        };

        let _ = write!(output, "{:>width$} | ", label, width = label_width);
        let line: String = row_data.iter().collect();
        output.push_str(&line);
        output.push('\n');
    }

    // ── Step 8: X-axis line and labels at bottom ───────────────────
    // Separator line
    let _ = write!(output, "{:>width$} +-", "", width = label_width);
    for _ in 0..width {
        output.push('-');
    }
    output.push('\n');

    // X-axis labels: left, center, right
    let x_left = format_num(x_min);
    let x_mid = format_num((x_min + x_max) / 2.0);
    let x_right = format_num(x_max);

    let _ = write!(output, "{:>width$}   ", "", width = label_width);
    let _ = write!(output, "{}", x_left);

    let mid_pos = width / 2;
    let left_used = x_left.len();
    if mid_pos > left_used + 1 {
        let pad = mid_pos - left_used;
        let _ = write!(output, "{:>pad$}{}", "", x_mid, pad = pad);
        let right_start = mid_pos + x_mid.len();
        if width > right_start + 1 {
            let rpad = width - right_start;
            let _ = write!(output, "{:>rpad$}", x_right, rpad = rpad);
        }
    } else {
        // Tight layout — just put right label
        let rpad = width.saturating_sub(left_used);
        let _ = write!(output, "{:>rpad$}", x_right, rpad = rpad);
    }
    output.push('\n');

    output
}

/// Format a number compactly for axis labels.
fn format_num(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if v.abs() >= 1e4 || v.abs() < 0.01 {
        format!("{:.2e}", v)
    } else if (v - v.round()).abs() < 1e-9 {
        format!("{:.0}", v)
    } else {
        let s = format!("{:.4}", v);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn textplot_basic_sine() {
        let points: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let x = i as f64 * std::f64::consts::TAU / 99.0;
                (x, x.sin())
            })
            .collect();
        let plot = textplot(&points, 60, 21, Some("sin(x)"));
        assert!(!plot.is_empty());
        assert!(plot.contains("sin(x)"));
        // Should have some plot characters
        assert!(plot.contains('.') || plot.contains('/') || plot.contains('\\'));
        // Should have y-axis labels
        assert!(plot.contains('|'));
    }

    #[test]
    fn textplot_empty_data() {
        let plot = textplot(&[], 60, 21, None);
        assert!(plot.contains("no valid data"));
    }

    #[test]
    fn textplot_nan_filtered() {
        let points = vec![(0.0, 1.0), (1.0, f64::NAN), (2.0, 3.0)];
        let plot = textplot(&points, 40, 11, None);
        assert!(!plot.is_empty());
        // Should still produce a plot with the 2 valid points
        assert!(plot.contains('|'));
    }

    #[test]
    fn textplot_single_point() {
        let points = vec![(1.0, 1.0)];
        let plot = textplot(&points, 30, 10, None);
        assert!(plot.contains('.'));
    }

    #[test]
    fn textplot_constant_function() {
        let points: Vec<(f64, f64)> = (0..50)
            .map(|i| (i as f64, 5.0))
            .collect();
        let plot = textplot(&points, 50, 15, Some("y=5"));
        assert!(plot.contains("y=5"));
        assert!(!plot.is_empty());
    }

    #[test]
    fn format_num_zero() {
        assert_eq!(format_num(0.0), "0");
    }

    #[test]
    fn format_num_integer() {
        assert_eq!(format_num(42.0), "42");
    }

    #[test]
    fn format_num_decimal() {
        let s = format_num(3.14158);
        assert!(s.starts_with("3.14"));
    }

    #[test]
    fn format_num_scientific() {
        let s = format_num(1e10);
        assert!(s.contains('e'));
    }
}
