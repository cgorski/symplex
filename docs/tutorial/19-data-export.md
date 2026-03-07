# Chapter 19: Data Export

Symplex can export evaluation data to CSV, TSV, JSON, Markdown, HTML, and LaTeX — making it easy to move results into spreadsheets, papers, or web pages. The core type is `DataTable` from `symplex::data_export`.

## Building a DataTable

### From Raw Data

`DataTable::new` takes headers and rows of strings:

```rust
use symplex::data_export::DataTable;

let table = DataTable::new(
    vec!["x".into(), "f(x)".into()],
    vec![
        vec!["0".into(), "0".into()],
        vec!["1".into(), "1".into()],
        vec!["2".into(), "4".into()],
    ],
);
assert_eq!(table.nrows(), 3);
assert_eq!(table.ncols(), 2);
```

### From Point Data

`DataTable::from_points` builds a two-column table from `(x, y)` pairs:

```rust
use symplex::data_export::DataTable;

let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0)];
let table = DataTable::from_points("x", "x²", &points);
assert_eq!(table.nrows(), 4);
```

### From a Function

`DataTable::from_evaluation` evaluates a closure at specified x-values:

```rust
use symplex::data_export::DataTable;

let x_vals = vec![0.0, 0.5, 1.0, 1.5, 2.0];
let table = DataTable::from_evaluation("x", "sin(x)", &x_vals, |x| x.sin());
assert_eq!(table.nrows(), 5);
```

### Multi-Column Tables

`DataTable::from_multi_eval` evaluates multiple functions at the same x-values — perfect for comparing expressions side by side:

```rust
use symplex::data_export::DataTable;

let x_vals = vec![0.0, 1.0, 2.0, 3.0];
let sin_fn: &dyn Fn(f64) -> f64 = &|x: f64| x.sin();
let cos_fn: &dyn Fn(f64) -> f64 = &|x: f64| x.cos();

let table = DataTable::from_multi_eval(
    "x",
    &["sin(x)", "cos(x)"],
    &x_vals,
    &[sin_fn, cos_fn],
);
assert_eq!(table.ncols(), 3); // x, sin(x), cos(x)
assert_eq!(table.nrows(), 4);
```

## Export Formats

Every `DataTable` can be exported to six formats. Here's the same table in each:

```rust
use symplex::data_export::DataTable;

let points = vec![(1.0, 1.0), (2.0, 4.0), (3.0, 9.0)];
let table = DataTable::from_points("x", "x²", &points);

// Comma-separated values
let csv = table.to_csv();
println!("{csv}");
// x,x²
// 1,1
// 2,4
// 3,9

// Tab-separated values
let tsv = table.to_tsv();

// JSON array of objects
let json = table.to_json();
println!("{json}");
// [
//   {"x": 1, "x²": 1},
//   {"x": 2, "x²": 4},
//   {"x": 3, "x²": 9}
// ]

// Markdown table (great for README files)
let md = table.to_markdown();
println!("{md}");
// | x   | x²  |
// |-----|-----|
// | 1   | 1   |
// | 2   | 4   |
// | 3   | 9   |

// HTML <table> element
let html = table.to_html();

// LaTeX tabular with booktabs
let latex = table.to_latex();
println!("{latex}");
// \begin{tabular}{r r}
// \toprule
// $x$ & $x²$ \\
// \midrule
// 1 & 1 \\
// 2 & 4 \\
// 3 & 9 \\
// \bottomrule
// \end{tabular}
```

## Expression-Level Export

Symplex expressions have two convenience methods that tie directly into data export.

### `Ex::plot_data` — Raw Sample Points

`.plot_data(&var, a, b, n)` compiles the expression and evaluates it at `n` uniformly-spaced points over `[a, b]`, returning `Vec<(f64, f64)>`:

```rust
use symplex::prelude::*;

let x = symplex::var("x");
let f = x.powi(2);

let data = f.plot_data(&x, 0.0, 3.0, 4);
// data ≈ [(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0)]
assert_eq!(data.len(), 4);
```

### `Ex::eval_table` — Direct to DataTable

`.eval_table(&var, &points)` returns a `DataTable` ready for export:

```rust
use symplex::prelude::*;

let x = symplex::var("x");
let f = x.powi(2);

let table = f.eval_table(&x, &[0.0, 1.0, 2.0, 3.0]);
assert_eq!(table.nrows(), 4);
assert_eq!(table.ncols(), 2); // "x" and "f(x)"
```

## Complete Workflow

Here's the typical pipeline: define an expression, evaluate it, and export to multiple formats.

```rust
use symplex::prelude::*;
use symplex::data_export::DataTable;

fn main() {
    let x = symplex::var("x");

    // Define a function
    let f = expr!(x^3 - 3*x^2 + 2*x);
    println!("f(x) = {f}");

    // Generate evaluation points
    let points: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1).collect();

    // Build the table directly from the expression
    let table = f.eval_table(&x, &points);

    // Export to CSV (e.g., for a spreadsheet)
    let csv = table.to_csv();
    println!("=== CSV ===\n{csv}");

    // Export to Markdown (e.g., for documentation)
    let md = table.to_markdown();
    println!("=== Markdown ===\n{md}");

    // Export to LaTeX (e.g., for a paper)
    let latex = table.to_latex();
    println!("=== LaTeX ===\n{latex}");

    // Multi-function comparison: f(x) and f'(x)
    let df = f.diff(&x);
    println!("f'(x) = {df}");

    let x_vals: Vec<f64> = (0..=10).map(|i| i as f64 * 0.2).collect();

    // Compile both expressions for fast evaluation
    let f_compiled = f.compile(&["x"]).unwrap();
    let df_compiled = df.compile(&["x"]).unwrap();

    let f_fn: &dyn Fn(f64) -> f64 = &|x| f_compiled(&[x]);
    let df_fn: &dyn Fn(f64) -> f64 = &|x| df_compiled(&[x]);

    let comparison = DataTable::from_multi_eval(
        "x",
        &["f(x)", "f'(x)"],
        &x_vals,
        &[f_fn, df_fn],
    );

    println!("=== Function vs Derivative ===");
    println!("{}", comparison.to_markdown());
}
```

## What's Next

Several data export features are planned for future releases:

```rust
// FreqResponseData export for Bode plot data
// let bode = FreqResponseData::new(freqs, mag_db, phase_deg);
// let csv = bode.to_csv();

// DataFrame interop (e.g., polars or arrow)
// let df = table.to_dataframe();

// Parquet columnar format for large datasets
// let bytes = table.to_parquet();
```

---

*[← Chapter 16: Complex Numbers](16-complex-numbers.md) | [Back to Table of Contents](index.md) | [Chapter 20: Finite Differences →](20-finite-differences.md)*