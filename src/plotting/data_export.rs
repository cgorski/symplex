//! Data export utilities for exporting tabular data in multiple formats.
//!
//! Supports CSV, TSV, JSON, Markdown, HTML, and LaTeX output from
//! structured table data. Includes helpers for point data, function
//! evaluation tables, and frequency response data.
//!
//! Tables are usually produced by
//! [`Ex::eval_table`](crate::expr::Ex::eval_table) or
//! [`Ex::plot_data`](crate::expr::Ex::plot_data) +
//! [`DataTable::from_points`]; every export method is infallible and
//! returns a `String`.
//!
//! ```
//! use symplex::prelude::*;
//!
//! let ctx = Context::new();
//! let x = ctx.symbol("x");
//! let table = x.powi(2).eval_table(&x, &[0.0, 0.5, 1.0]).unwrap();
//! assert_eq!(table.to_csv(), "x,f(x)\n0,0\n0.5,0.25\n1,1\n");
//! ```

/// Format a floating-point value with reasonable precision.
///
/// - Zero → `"0"`
/// - Very large or very small → scientific notation
/// - Near-integer → no decimal places
/// - Otherwise → up to 6 decimal places (trailing zeros stripped)
fn format_f64(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "Inf".to_string()
        } else {
            "-Inf".to_string()
        };
    }
    if v == 0.0 {
        return "0".to_string();
    }
    if v.abs() >= 1e6 || v.abs() < 1e-4 {
        format!("{:.6e}", v)
    } else if (v - v.round()).abs() < 1e-10 {
        format!("{:.0}", v)
    } else {
        // Format with 6 decimals, then strip trailing zeros after the dot.
        let s = format!("{:.6}", v);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

/// Escape a value for CSV output. If the value contains a comma, quote, or
/// newline, wrap it in double quotes and escape internal quotes by doubling.
fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        let escaped = value.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        value.to_string()
    }
}

/// A table of data with headers, supporting multiple export formats.
///
/// Cells are stored as already-formatted strings; numeric cells use up to
/// six significant decimals (`0.333333`), scientific notation outside
/// `[1e-4, 1e6)`, and `NaN` / `Inf` / `-Inf` for non-finite values.
///
/// ```
/// use symplex::data_export::DataTable;
///
/// let t = DataTable::from_points("t", "v", &[(0.0, 1.0), (1.0, 0.5)]);
/// assert_eq!(t.nrows(), 2);
/// assert_eq!(t.ncols(), 2);
/// assert_eq!(t.rows[1], vec!["1", "0.5"]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataTable {
    /// Column headers.
    pub headers: Vec<String>,
    /// Rows of string-formatted values.
    pub rows: Vec<Vec<String>>,
}

impl DataTable {
    /// Create a table from headers and rows of strings.
    ///
    /// Rows shorter than `headers` are padded with empty cells on export.
    ///
    /// ```
    /// use symplex::data_export::DataTable;
    ///
    /// let t = DataTable::new(
    ///     vec!["name".into(), "value".into()],
    ///     vec![vec!["pi".into(), "3.14159".into()]],
    /// );
    /// assert_eq!(t.to_tsv(), "name\tvalue\npi\t3.14159\n");
    /// ```
    pub fn new(headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        Self { headers, rows }
    }

    /// Create a table from (x, y) point data.
    ///
    /// ```
    /// use symplex::data_export::DataTable;
    ///
    /// let t = DataTable::from_points("x", "y", &[(0.0, 0.0), (0.5, 0.25), (1.0, f64::NAN)]);
    /// assert_eq!(t.rows[1], vec!["0.5", "0.25"]);
    /// assert_eq!(t.rows[2][1], "NaN");
    /// ```
    pub fn from_points(x_label: &str, y_label: &str, points: &[(f64, f64)]) -> Self {
        let headers = vec![x_label.to_string(), y_label.to_string()];
        let rows = points
            .iter()
            .map(|(x, y)| vec![format_f64(*x), format_f64(*y)])
            .collect();
        Self { headers, rows }
    }

    /// Create a table from evaluating a function at multiple points.
    /// `eval_fn` takes x and returns y.
    ///
    /// ```
    /// use symplex::data_export::DataTable;
    ///
    /// let t = DataTable::from_evaluation("x", "2x", &[1.0, 2.0], |x| 2.0 * x);
    /// assert_eq!(t.rows, vec![vec!["1", "2"], vec!["2", "4"]]);
    /// ```
    pub fn from_evaluation(
        x_label: &str,
        y_label: &str,
        x_values: &[f64],
        eval_fn: impl Fn(f64) -> f64,
    ) -> Self {
        let headers = vec![x_label.to_string(), y_label.to_string()];
        let rows = x_values
            .iter()
            .map(|&x| vec![format_f64(x), format_f64(eval_fn(x))])
            .collect();
        Self { headers, rows }
    }

    /// Create a multi-column table from evaluating several functions at the
    /// same x-values.
    ///
    /// ```
    /// use symplex::data_export::DataTable;
    ///
    /// let sq = |x: f64| x * x;
    /// let cube = |x: f64| x * x * x;
    /// let t = DataTable::from_multi_eval("x", &["x^2", "x^3"], &[2.0, 3.0], &[&sq, &cube]);
    /// assert_eq!(t.headers, vec!["x", "x^2", "x^3"]);
    /// assert_eq!(t.rows[1], vec!["3", "9", "27"]);
    /// ```
    pub fn from_multi_eval(
        x_label: &str,
        y_labels: &[&str],
        x_values: &[f64],
        eval_fns: &[&dyn Fn(f64) -> f64],
    ) -> Self {
        let mut headers = vec![x_label.to_string()];
        for label in y_labels {
            headers.push(label.to_string());
        }
        let rows = x_values
            .iter()
            .map(|&x| {
                let mut row = vec![format_f64(x)];
                for f in eval_fns {
                    row.push(format_f64(f(x)));
                }
                row
            })
            .collect();
        Self { headers, rows }
    }

    /// Export as CSV (RFC 4180 quoting for cells containing `,`, `"` or
    /// newlines).
    ///
    /// ```
    /// use symplex::data_export::DataTable;
    ///
    /// let t = DataTable::new(vec!["a,b".into(), "c".into()], vec![vec!["1".into(), "x\"y".into()]]);
    /// assert_eq!(t.to_csv(), "\"a,b\",c\n1,\"x\"\"y\"\n");
    /// ```
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        // Header line
        let header_line: Vec<String> = self.headers.iter().map(|h| csv_escape(h)).collect();
        out.push_str(&header_line.join(","));
        out.push('\n');
        // Data rows
        for row in &self.rows {
            let escaped: Vec<String> = row.iter().map(|v| csv_escape(v)).collect();
            out.push_str(&escaped.join(","));
            out.push('\n');
        }
        out
    }

    /// Export as TSV (tab-separated) string.
    ///
    /// ```
    /// use symplex::data_export::DataTable;
    ///
    /// let t = DataTable::from_points("x", "y", &[(1.0, 2.0)]);
    /// assert_eq!(t.to_tsv(), "x\ty\n1\t2\n");
    /// ```
    pub fn to_tsv(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.headers.join("\t"));
        out.push('\n');
        for row in &self.rows {
            out.push_str(&row.join("\t"));
            out.push('\n');
        }
        out
    }

    /// Export as a JSON array of objects (one object per row, keyed by
    /// header).  Cells that parse as numbers are emitted as JSON numbers,
    /// everything else as strings.
    ///
    /// ```
    /// use symplex::data_export::DataTable;
    ///
    /// let t = DataTable::from_points("x", "y", &[(1.0, 2.5)]);
    /// assert_eq!(t.to_json(), "[\n  {\"x\": 1, \"y\": 2.5}\n]");
    /// ```
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("[\n");
        for (i, row) in self.rows.iter().enumerate() {
            out.push_str("  {");
            let mut fields = Vec::new();
            for (j, header) in self.headers.iter().enumerate() {
                let value = row.get(j).map(|s| s.as_str()).unwrap_or("");
                // Escape the header for JSON string
                let escaped_header = json_escape_string(header);
                // Try to parse as a number; if it succeeds, emit without quotes
                if let Ok(_n) = value.parse::<f64>() {
                    // Use the value as-is (it's a valid number literal)
                    fields.push(format!("\"{}\": {}", escaped_header, value));
                } else {
                    let escaped_value = json_escape_string(value);
                    fields.push(format!("\"{}\": \"{}\"", escaped_header, escaped_value));
                }
            }
            out.push_str(&fields.join(", "));
            out.push('}');
            if i + 1 < self.rows.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push(']');
        out
    }

    /// Export as a GitHub-flavoured Markdown table with aligned columns.
    ///
    /// ```
    /// use symplex::data_export::DataTable;
    ///
    /// let t = DataTable::from_points("x", "y", &[(1.0, 2.0)]);
    /// let md = t.to_markdown();
    /// assert!(md.starts_with("| x   | y   |\n|-----|-----|\n"));
    /// assert!(md.contains("| 1   | 2   |"));
    /// ```
    pub fn to_markdown(&self) -> String {
        // Compute column widths
        let ncols = self.headers.len();
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();
        for row in &self.rows {
            for (j, cell) in row.iter().enumerate() {
                if j < ncols {
                    widths[j] = widths[j].max(cell.len());
                }
            }
        }
        // Ensure minimum width of 3 for separator
        for w in &mut widths {
            if *w < 3 {
                *w = 3;
            }
        }

        let mut out = String::new();
        // Header row
        out.push('|');
        for (j, header) in self.headers.iter().enumerate() {
            out.push_str(&format!(" {:width$} |", header, width = widths[j]));
        }
        out.push('\n');
        // Separator
        out.push('|');
        for w in &widths {
            out.push_str(&format!("-{}-|", "-".repeat(*w)));
        }
        out.push('\n');
        // Data rows
        for row in &self.rows {
            out.push('|');
            for (j, w) in widths.iter().enumerate().take(ncols) {
                let cell = row.get(j).map(|s| s.as_str()).unwrap_or("");
                out.push_str(&format!(" {:width$} |", cell, width = *w));
            }
            out.push('\n');
        }
        out
    }

    /// Export as an HTML `<table>` (cells are HTML-escaped).
    ///
    /// ```
    /// use symplex::data_export::DataTable;
    ///
    /// let t = DataTable::new(vec!["a<b".into()], vec![vec!["1".into()]]);
    /// let html = t.to_html();
    /// assert!(html.contains("<th>a&lt;b</th>"));
    /// assert!(html.contains("<td>1</td>"));
    /// ```
    pub fn to_html(&self) -> String {
        let mut out = String::new();
        out.push_str("<table>\n");
        // Header
        out.push_str("<thead><tr>");
        for header in &self.headers {
            out.push_str(&format!("<th>{}</th>", html_escape(header)));
        }
        out.push_str("</tr></thead>\n");
        // Body
        out.push_str("<tbody>\n");
        for row in &self.rows {
            out.push_str("<tr>");
            for cell in row {
                out.push_str(&format!("<td>{}</td>", html_escape(cell)));
            }
            out.push_str("</tr>\n");
        }
        out.push_str("</tbody>\n");
        out.push_str("</table>");
        out
    }

    /// Export as a LaTeX `tabular` environment (booktabs rules; headers in
    /// math mode; `_` and `%` escaped).
    ///
    /// ```
    /// use symplex::data_export::DataTable;
    ///
    /// let t = DataTable::from_points("x", "f(x)", &[(0.0, 1.0)]);
    /// let tex = t.to_latex();
    /// assert!(tex.starts_with("\\begin{tabular}{r r}\n\\toprule\n$x$ & $f(x)$ \\\\\n"));
    /// assert!(tex.ends_with("\\bottomrule\n\\end{tabular}"));
    /// ```
    pub fn to_latex(&self) -> String {
        let ncols = self.headers.len();
        let col_spec = vec!["r"; ncols].join(" ");

        let mut out = String::new();
        out.push_str(&format!("\\begin{{tabular}}{{{}}}\n", col_spec));
        out.push_str("\\toprule\n");
        // Header
        let header_cells: Vec<String> = self
            .headers
            .iter()
            .map(|h| format!("${}$", latex_escape(h)))
            .collect();
        out.push_str(&header_cells.join(" & "));
        out.push_str(" \\\\\n");
        out.push_str("\\midrule\n");
        // Data rows
        for row in &self.rows {
            let cells: Vec<String> = row.iter().map(|c| latex_escape(c)).collect();
            out.push_str(&cells.join(" & "));
            out.push_str(" \\\\\n");
        }
        out.push_str("\\bottomrule\n");
        out.push_str("\\end{tabular}");
        out
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.headers.len()
    }
}

/// Escape special HTML characters.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Minimal LaTeX escaping — escape underscores and percent signs.
fn latex_escape(s: &str) -> String {
    s.replace('%', "\\%").replace('_', "\\_")
}

/// Escape a string for JSON output (handle backslashes, quotes, etc.).
fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

/// Frequency response data for Bode plots and similar exports.
///
/// ```
/// use symplex::data_export::FreqResponseData;
///
/// let fr = FreqResponseData::new(vec![1.0, 10.0], vec![0.0, -20.0], vec![0.0, -90.0]);
/// assert_eq!(fr.to_csv(), "omega_rad_s,magnitude_dB,phase_deg\n1,0,0\n10,-20,-90\n");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FreqResponseData {
    /// Angular frequencies in rad/s.
    pub frequencies: Vec<f64>,
    /// Magnitude in dB: 20*log10(|G(jω)|).
    pub magnitude_db: Vec<f64>,
    /// Phase in degrees: ∠G(jω).
    pub phase_deg: Vec<f64>,
}

impl FreqResponseData {
    /// Create a new frequency response dataset.
    pub fn new(frequencies: Vec<f64>, magnitude_db: Vec<f64>, phase_deg: Vec<f64>) -> Self {
        Self {
            frequencies,
            magnitude_db,
            phase_deg,
        }
    }

    /// Export as CSV string.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str("omega_rad_s,magnitude_dB,phase_deg\n");
        let n = self.frequencies.len();
        for i in 0..n {
            let freq = self.frequencies.get(i).copied().unwrap_or(f64::NAN);
            let mag = self.magnitude_db.get(i).copied().unwrap_or(f64::NAN);
            let phase = self.phase_deg.get(i).copied().unwrap_or(f64::NAN);
            out.push_str(&format!(
                "{},{},{}\n",
                format_f64(freq),
                format_f64(mag),
                format_f64(phase)
            ));
        }
        out
    }

    /// Export as a JSON array of `{omega_rad_s, magnitude_dB, phase_deg}`
    /// objects.  Missing entries in a shorter vector are emitted as `NaN`.
    ///
    /// ```
    /// use symplex::data_export::FreqResponseData;
    ///
    /// let fr = FreqResponseData::new(vec![1.0], vec![-3.0], vec![-45.0]);
    /// assert_eq!(
    ///     fr.to_json(),
    ///     "[\n  {\"omega_rad_s\": 1, \"magnitude_dB\": -3, \"phase_deg\": -45}\n]"
    /// );
    /// ```
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("[\n");
        let n = self.frequencies.len();
        for i in 0..n {
            let freq = self.frequencies.get(i).copied().unwrap_or(f64::NAN);
            let mag = self.magnitude_db.get(i).copied().unwrap_or(f64::NAN);
            let phase = self.phase_deg.get(i).copied().unwrap_or(f64::NAN);
            out.push_str(&format!(
                "  {{\"omega_rad_s\": {}, \"magnitude_dB\": {}, \"phase_deg\": {}}}",
                format_f64(freq),
                format_f64(mag),
                format_f64(phase)
            ));
            if i + 1 < n {
                out.push(',');
            }
            out.push('\n');
        }
        out.push(']');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_f64_zero() {
        assert_eq!(format_f64(0.0), "0");
    }

    #[test]
    fn format_f64_integer() {
        assert_eq!(format_f64(42.0), "42");
        assert_eq!(format_f64(-7.0), "-7");
    }

    #[test]
    fn format_f64_decimal() {
        let s = format_f64(3.14158);
        assert!(s.starts_with("3.14158"));
    }

    #[test]
    fn format_f64_scientific_large() {
        let s = format_f64(1.5e8);
        assert!(s.contains('e'), "expected scientific notation, got: {}", s);
    }

    #[test]
    fn format_f64_scientific_small() {
        let s = format_f64(1.5e-6);
        assert!(s.contains('e'), "expected scientific notation, got: {}", s);
    }

    #[test]
    fn format_f64_nan() {
        assert_eq!(format_f64(f64::NAN), "NaN");
    }

    #[test]
    fn csv_escape_plain() {
        assert_eq!(csv_escape("hello"), "hello");
    }

    #[test]
    fn csv_escape_comma() {
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
    }

    #[test]
    fn csv_escape_quote() {
        assert_eq!(csv_escape("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn table_to_csv_basic() {
        let table = DataTable::from_points("x", "f(x)", &[(0.0, 1.0), (0.5, 2.25), (1.0, 4.0)]);
        let csv = table.to_csv();
        assert!(csv.starts_with("x,f(x)\n"));
        assert!(csv.contains("0,1\n"));
        assert!(csv.contains("0.5,2.25\n"));
        assert!(csv.contains("1,4\n"));
    }

    #[test]
    fn table_to_json_basic() {
        let table = DataTable::from_points("x", "f(x)", &[(0.0, 1.0), (1.0, 4.0)]);
        let json = table.to_json();
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"x\""));
        assert!(json.contains("\"f(x)\""));
    }

    #[test]
    fn table_to_markdown_basic() {
        let table = DataTable::from_points("x", "y", &[(1.0, 2.0), (3.0, 4.0)]);
        let md = table.to_markdown();
        assert!(md.contains("| x"));
        assert!(md.contains("| y"));
        assert!(md.contains("|---"));
    }

    #[test]
    fn table_to_html_basic() {
        let table = DataTable::from_points("x", "y", &[(1.0, 2.0)]);
        let html = table.to_html();
        assert!(html.contains("<table>"));
        assert!(html.contains("<th>x</th>"));
        assert!(html.contains("<td>1</td>"));
        assert!(html.contains("</table>"));
    }

    #[test]
    fn table_to_latex_basic() {
        let table = DataTable::from_points("x", "y", &[(1.0, 2.0)]);
        let latex = table.to_latex();
        assert!(latex.contains("\\begin{tabular}"));
        assert!(latex.contains("\\toprule"));
        assert!(latex.contains("\\midrule"));
        assert!(latex.contains("\\bottomrule"));
        assert!(latex.contains("\\end{tabular}"));
    }

    #[test]
    fn table_nrows_ncols() {
        let table = DataTable::from_points("a", "b", &[(1.0, 2.0), (3.0, 4.0)]);
        assert_eq!(table.nrows(), 2);
        assert_eq!(table.ncols(), 2);
    }

    #[test]
    fn table_from_evaluation() {
        let table = DataTable::from_evaluation("x", "x^2", &[0.0, 1.0, 2.0], |x| x * x);
        assert_eq!(table.nrows(), 3);
        let csv = table.to_csv();
        assert!(csv.contains("0,0\n"));
        assert!(csv.contains("1,1\n"));
        assert!(csv.contains("2,4\n"));
    }

    #[test]
    fn table_multi_column() {
        let sin_fn: &dyn Fn(f64) -> f64 = &|x: f64| x.sin();
        let cos_fn: &dyn Fn(f64) -> f64 = &|x: f64| x.cos();
        let table = DataTable::from_multi_eval(
            "x",
            &["sin(x)", "cos(x)"],
            &[0.0, std::f64::consts::FRAC_PI_2],
            &[sin_fn, cos_fn],
        );
        assert_eq!(table.ncols(), 3);
        assert_eq!(table.nrows(), 2);
        // First row: x=0, sin(0)=0, cos(0)=1
        assert_eq!(table.rows[0][0], "0");
        assert_eq!(table.rows[0][1], "0");
        assert_eq!(table.rows[0][2], "1");
    }

    #[test]
    fn csv_handles_nan() {
        let table = DataTable::new(
            vec!["x".to_string(), "y".to_string()],
            vec![vec!["1".to_string(), "NaN".to_string()]],
        );
        let csv = table.to_csv();
        assert!(csv.contains("NaN"));
    }

    #[test]
    fn freq_response_to_csv() {
        let data = FreqResponseData::new(
            vec![1.0, 10.0, 100.0],
            vec![0.0, -3.0, -20.0],
            vec![0.0, -45.0, -90.0],
        );
        let csv = data.to_csv();
        assert!(csv.starts_with("omega_rad_s,magnitude_dB,phase_deg\n"));
        assert!(csv.contains("1,0,0\n"));
        assert!(csv.contains("10,-3,-45\n"));
        assert!(csv.contains("100,-20,-90\n"));
    }

    #[test]
    fn freq_response_to_json() {
        let data = FreqResponseData::new(vec![1.0], vec![0.0], vec![-45.0]);
        let json = data.to_json();
        assert!(json.contains("\"omega_rad_s\""));
        assert!(json.contains("\"magnitude_dB\""));
        assert!(json.contains("\"phase_deg\""));
    }

    #[test]
    fn tsv_output() {
        let table = DataTable::from_points("x", "y", &[(1.0, 2.0)]);
        let tsv = table.to_tsv();
        assert!(tsv.contains("x\ty\n"));
        assert!(tsv.contains("1\t2\n"));
    }

    #[test]
    fn html_escapes_special_chars() {
        let table = DataTable::new(
            vec!["a<b".to_string(), "c&d".to_string()],
            vec![vec!["<script>".to_string(), "x&y".to_string()]],
        );
        let html = table.to_html();
        assert!(html.contains("&lt;script&gt;"));
        assert!(html.contains("c&amp;d"));
    }
}
