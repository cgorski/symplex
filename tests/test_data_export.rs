//! Integration tests for the `data_export` module.

use symplex::data_export::{DataTable, FreqResponseData};

#[test]
fn table_to_csv_basic() {
    let table = DataTable::from_points("x", "f(x)", &[(0.0, 1.0), (0.5, 2.25), (1.0, 4.0)]);
    let csv = table.to_csv();
    assert_eq!(table.nrows(), 3);
    // Header line
    assert!(csv.starts_with("x,f(x)\n"), "CSV should start with header line, got:\n{csv}");
    // Data rows
    assert!(csv.contains("0,1\n"), "expected row '0,1', got:\n{csv}");
    assert!(csv.contains("0.5,2.25\n"), "expected row '0.5,2.25', got:\n{csv}");
    assert!(csv.contains("1,4\n"), "expected row '1,4', got:\n{csv}");
    // Exactly 4 lines (header + 3 data rows)
    let line_count = csv.lines().count();
    assert_eq!(line_count, 4, "expected 4 lines, got {line_count}");
}

#[test]
fn table_to_json_basic() {
    let table = DataTable::from_points("x", "f(x)", &[(0.0, 1.0), (0.5, 2.25), (1.0, 4.0)]);
    let json = table.to_json();
    assert!(json.starts_with('['), "JSON should start with '[', got:\n{json}");
    assert!(json.trim_end().ends_with(']'), "JSON should end with ']', got:\n{json}");
    // Should contain header keys
    assert!(json.contains("\"x\""), "JSON should contain key 'x', got:\n{json}");
    assert!(json.contains("\"f(x)\""), "JSON should contain key 'f(x)', got:\n{json}");
    // Should contain numeric values (not quoted)
    assert!(json.contains(": 0"), "JSON should contain value 0");
    assert!(json.contains(": 2.25"), "JSON should contain value 2.25");
    // Should have 3 objects
    let brace_count = json.matches('{').count();
    assert_eq!(brace_count, 3, "expected 3 JSON objects, got {brace_count}");
}

#[test]
fn table_to_markdown_basic() {
    let table = DataTable::from_points("x", "f(x)", &[(0.0, 1.0), (0.5, 2.25), (1.0, 4.0)]);
    let md = table.to_markdown();
    let lines: Vec<&str> = md.lines().collect();
    // At least header + separator + 3 data rows = 5 lines
    assert!(lines.len() >= 5, "expected at least 5 lines, got {}:\n{md}", lines.len());
    // Header line has pipes
    assert!(lines[0].contains("| x"), "header should contain '| x', got: {}", lines[0]);
    assert!(lines[0].contains("| f(x)"), "header should contain '| f(x)', got: {}", lines[0]);
    // Separator line has dashes
    assert!(lines[1].contains("|---") || lines[1].contains("|-"), "separator should have dashes, got: {}", lines[1]);
    // Data rows have pipes
    for line in &lines[2..] {
        assert!(line.starts_with('|'), "data line should start with '|', got: {line}");
        assert!(line.ends_with('|'), "data line should end with '|', got: {line}");
    }
}

#[test]
fn table_to_html_basic() {
    let table = DataTable::from_points("x", "f(x)", &[(0.0, 1.0), (0.5, 2.25), (1.0, 4.0)]);
    let html = table.to_html();
    // Structure checks
    assert!(html.contains("<table>"), "should contain <table>");
    assert!(html.contains("</table>"), "should contain </table>");
    assert!(html.contains("<thead>"), "should contain <thead>");
    assert!(html.contains("<tbody>"), "should contain <tbody>");
    // Headers
    assert!(html.contains("<th>x</th>"), "should contain <th>x</th>");
    assert!(html.contains("<th>f(x)</th>"), "should contain <th>f(x)</th>");
    // Data cells
    assert!(html.contains("<td>0</td>"), "should contain <td>0</td>, got:\n{html}");
    assert!(html.contains("<td>1</td>"), "should contain <td>1</td>, got:\n{html}");
    assert!(html.contains("<td>0.5</td>"), "should contain <td>0.5</td>, got:\n{html}");
    assert!(html.contains("<td>2.25</td>"), "should contain <td>2.25</td>, got:\n{html}");
    // 3 data rows
    let tr_count = html.matches("<tr>").count();
    // 1 header <tr> + 3 data <tr>
    assert_eq!(tr_count, 4, "expected 4 <tr> tags (1 header + 3 data), got {tr_count}");
}

#[test]
fn table_to_latex_basic() {
    let table = DataTable::from_points("x", "f(x)", &[(0.0, 1.0), (0.5, 2.25), (1.0, 4.0)]);
    let latex = table.to_latex();
    // Booktabs commands
    assert!(latex.contains("\\begin{tabular}"), "should contain \\begin{{tabular}}");
    assert!(latex.contains("\\end{tabular}"), "should contain \\end{{tabular}}");
    assert!(latex.contains("\\toprule"), "should contain \\toprule");
    assert!(latex.contains("\\midrule"), "should contain \\midrule");
    assert!(latex.contains("\\bottomrule"), "should contain \\bottomrule");
    // Headers wrapped in $...$
    assert!(latex.contains("$x$"), "should contain $x$ header");
    assert!(latex.contains("$f(x)$"), "should contain $f(x)$ header");
    // Column separator
    assert!(latex.contains(" & "), "should use '&' column separator");
    // Row terminator
    assert!(latex.contains("\\\\"), "should contain '\\\\' row terminator");
}

#[test]
fn table_from_points() {
    let points = vec![(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)];
    let table = DataTable::from_points("t", "v(t)", &points);
    assert_eq!(table.nrows(), 3);
    assert_eq!(table.ncols(), 2);
    assert_eq!(table.headers[0], "t");
    assert_eq!(table.headers[1], "v(t)");
    // Check that rows have the correct formatted values
    assert_eq!(table.rows[0][0], "1");
    assert_eq!(table.rows[0][1], "10");
    assert_eq!(table.rows[1][0], "2");
    assert_eq!(table.rows[1][1], "20");
    assert_eq!(table.rows[2][0], "3");
    assert_eq!(table.rows[2][1], "30");
}

#[test]
fn table_from_evaluation() {
    let table = DataTable::from_evaluation("x", "x^2", &[0.0, 1.0, 2.0, 3.0], |x| x * x);
    assert_eq!(table.nrows(), 4);
    assert_eq!(table.ncols(), 2);
    let csv = table.to_csv();
    assert!(csv.contains("x,x^2\n"), "header should be 'x,x^2'");
    assert!(csv.contains("0,0\n"), "0^2 = 0");
    assert!(csv.contains("1,1\n"), "1^2 = 1");
    assert!(csv.contains("2,4\n"), "2^2 = 4");
    assert!(csv.contains("3,9\n"), "3^2 = 9");
}

#[test]
fn table_multi_column() {
    let sin_fn: &dyn Fn(f64) -> f64 = &|x: f64| x.sin();
    let cos_fn: &dyn Fn(f64) -> f64 = &|x: f64| x.cos();
    let x_vals = [0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI];
    let table = DataTable::from_multi_eval("x", &["sin(x)", "cos(x)"], &x_vals, &[sin_fn, cos_fn]);
    assert_eq!(table.ncols(), 3, "should have 3 columns (x, sin, cos)");
    assert_eq!(table.nrows(), 3, "should have 3 rows");
    assert_eq!(table.headers, vec!["x", "sin(x)", "cos(x)"]);
    // Row 0: x=0 → sin(0)=0, cos(0)=1
    assert_eq!(table.rows[0][0], "0");
    assert_eq!(table.rows[0][1], "0");
    assert_eq!(table.rows[0][2], "1");
    // Row 1: x=π/2 → sin=1, cos≈0
    assert_eq!(table.rows[1][1], "1");
    // cos(π/2) should be very close to 0
    let cos_pi2: f64 = table.rows[1][2].parse().unwrap_or(999.0);
    assert!(cos_pi2.abs() < 1e-10, "cos(π/2) should be ~0, got {cos_pi2}");
}

#[test]
fn csv_handles_nan() {
    // Build a table where NaN appears via evaluation
    let table = DataTable::from_evaluation("x", "y", &[0.0, f64::NAN, 1.0], |x| x);
    let csv = table.to_csv();
    // NaN should appear in the output (not crash, not produce empty string)
    assert!(csv.contains("NaN"), "NaN values should be rendered as 'NaN', got:\n{csv}");

    // Also test with a direct NaN value in rows
    let table2 = DataTable::new(
        vec!["a".to_string(), "b".to_string()],
        vec![
            vec!["1".to_string(), "NaN".to_string()],
            vec!["NaN".to_string(), "2".to_string()],
        ],
    );
    let csv2 = table2.to_csv();
    let nan_count = csv2.matches("NaN").count();
    assert_eq!(nan_count, 2, "should have 2 NaN values in CSV, got:\n{csv2}");
}

#[test]
fn freq_response_to_csv() {
    let data = FreqResponseData::new(
        vec![0.1, 1.0, 10.0, 100.0, 1000.0],
        vec![0.0, 0.0, -3.0, -20.0, -40.0],
        vec![0.0, -5.0, -45.0, -85.0, -90.0],
    );
    let csv = data.to_csv();
    // Check header
    assert!(
        csv.starts_with("omega_rad_s,magnitude_dB,phase_deg\n"),
        "CSV should start with correct header, got:\n{csv}"
    );
    // Check that all 5 data rows are present
    let line_count = csv.lines().count();
    assert_eq!(line_count, 6, "expected 6 lines (1 header + 5 data), got {line_count}");
    // Check specific values
    assert!(csv.contains("1,0,-5\n"), "should contain row for ω=1, got:\n{csv}");
    assert!(csv.contains("10,-3,-45\n"), "should contain row for ω=10, got:\n{csv}");
    assert!(csv.contains("1000,-40,-90\n"), "should contain row for ω=1000, got:\n{csv}");

    // Also check JSON export works
    let json = data.to_json();
    assert!(json.starts_with('['));
    assert!(json.trim_end().ends_with(']'));
    assert!(json.contains("\"omega_rad_s\""));
    assert!(json.contains("\"magnitude_dB\""));
    assert!(json.contains("\"phase_deg\""));
    let obj_count = json.matches('{').count();
    assert_eq!(obj_count, 5, "expected 5 JSON objects, got {obj_count}");
}
