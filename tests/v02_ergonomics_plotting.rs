//! v0.2 ergonomics — `Result`-returning plotting / tabulation API.
//!
//! `plot_data`, `textplot`, `to_svg`, `to_tikz`, `eval_table`.

use symplex::prelude::*;

fn setup() -> (Context, Ex) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    (ctx, x)
}

// ── plot_data ──────────────────────────────────────────────────────────

#[test]
fn plot_data_exact_grid_and_values() {
    let (_ctx, x) = setup();
    let data = x.powi(2).plot_data(&x, 0.0, 1.0, 5).unwrap();
    assert_eq!(data.len(), 5);
    let xs: Vec<f64> = data.iter().map(|p| p.0).collect();
    assert_eq!(xs, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    for (xv, yv) in &data {
        assert!((yv - xv * xv).abs() < 1e-12);
    }
}

#[test]
fn plot_data_hits_right_endpoint_exactly() {
    let (_ctx, x) = setup();
    let data = x.plot_data(&x, 0.0, 0.3, 4).unwrap();
    assert_eq!(data.last().unwrap().0, 0.3);
}

#[test]
fn plot_data_constant_expression() {
    let (ctx, x) = setup();
    let data = ctx.int(5).plot_data(&x, 0.0, 10.0, 3).unwrap();
    assert!(data.iter().all(|(_, y)| *y == 5.0));
}

#[test]
fn plot_data_partial_domain_gives_nan_points() {
    let (_ctx, x) = setup();
    // ln(x) is undefined for x ≤ 0 but fine for x > 0.
    let data = x.ln().plot_data(&x, -1.0, 1.0, 5).unwrap();
    assert_eq!(data.len(), 5);
    assert!(!data[0].1.is_finite());
    assert!(data[4].1.abs() < 1e-12); // ln(1) = 0
}

#[test]
fn plot_data_all_undefined_is_computation_failed() {
    let (_ctx, x) = setup();
    assert!(matches!(
        x.ln().plot_data(&x, -2.0, -1.0, 5),
        Err(SymplexError::ComputationFailed {
            operation: "plot_data",
            ..
        })
    ));
}

#[test]
fn plot_data_rejects_too_few_points() {
    let (_ctx, x) = setup();
    for n in [0, 1] {
        assert!(matches!(
            x.plot_data(&x, 0.0, 1.0, n),
            Err(SymplexError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn plot_data_rejects_bad_range() {
    let (_ctx, x) = setup();
    assert!(x.plot_data(&x, 1.0, 0.0, 5).is_err());
    assert!(x.plot_data(&x, 1.0, 1.0, 5).is_err());
    assert!(x.plot_data(&x, f64::NAN, 1.0, 5).is_err());
    assert!(x.plot_data(&x, 0.0, f64::INFINITY, 5).is_err());
}

#[test]
fn plot_data_rejects_extra_free_symbol() {
    let (ctx, x) = setup();
    let y = ctx.symbol("y");
    match (&x + &y).plot_data(&x, 0.0, 1.0, 5) {
        Err(SymplexError::FreeSymbol { name }) => assert_eq!(name, "y"),
        other => panic!("expected FreeSymbol, got {other:?}"),
    }
}

#[test]
fn plot_data_rejects_non_symbol_variable() {
    let (_ctx, x) = setup();
    let e = x.powi(2);
    assert!(matches!(
        e.plot_data(&e, 0.0, 1.0, 5),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn plot_data_falls_back_for_uncompilable_nodes() {
    // An unevaluated definite integral with a symbolic bound can't be
    // compiled; the fallback substitutes exactly and evaluates.
    let (ctx, x) = setup();
    let t = ctx.symbol("t");
    let f = t.integrate_definite(&t, &ctx.zero(), &x); // = x^2/2 if evaluated
    let data = f.plot_data(&x, 0.0, 2.0, 3).unwrap();
    assert!((data[2].1 - 2.0).abs() < 1e-9, "{data:?}");
}

// ── textplot / to_svg / to_tikz ────────────────────────────────────────

#[test]
fn textplot_ok_has_grid_and_markers() {
    let (_ctx, x) = setup();
    let plot = x.sin().textplot(&x, 0.0, 6.28).unwrap();
    assert!(plot.lines().count() >= 21);
    assert!(plot.contains('.') || plot.contains('/') || plot.contains('\\'));
}

#[test]
fn textplot_skips_singularities() {
    let (ctx, x) = setup();
    let plot = (&ctx.one() / &x).textplot(&x, -1.0, 1.0).unwrap();
    assert!(!plot.contains("no valid data"));
}

#[test]
fn textplot_errors_propagate() {
    let (ctx, x) = setup();
    let y = ctx.symbol("y");
    assert!(matches!(
        (&x * &y).textplot(&x, 0.0, 1.0),
        Err(SymplexError::FreeSymbol { .. })
    ));
    assert!(matches!(
        x.textplot(&x, 2.0, 1.0),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        (-x.powi(2) - 1).sqrt().textplot(&x, 0.0, 1.0),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn to_svg_ok_is_well_formed() {
    let (_ctx, x) = setup();
    let svg = x.cos().to_svg(&x, 0.0, 3.0).unwrap();
    assert_eq!(svg.matches("<svg").count(), 1);
    assert_eq!(svg.matches("</svg>").count(), 1);
    assert!(svg.contains("<polyline"));
}

#[test]
fn to_svg_errors_propagate() {
    let (ctx, x) = setup();
    assert!((&x + &ctx.symbol("z")).to_svg(&x, 0.0, 1.0).is_err());
    assert!(x.to_svg(&x, 0.0, 0.0).is_err());
}

#[test]
fn to_tikz_ok_has_axis_environment() {
    let (_ctx, x) = setup();
    let tikz = x.powi(3).to_tikz(&x, -1.0, 1.0).unwrap();
    assert!(tikz.contains("\\begin{tikzpicture}"));
    assert!(tikz.contains("\\begin{axis}"));
    assert!(tikz.contains("\\addplot"));
}

#[test]
fn to_tikz_errors_propagate() {
    let (ctx, x) = setup();
    assert!((&x + &ctx.symbol("z")).to_tikz(&x, 0.0, 1.0).is_err());
    assert!(x.to_tikz(&x, f64::NEG_INFINITY, 1.0).is_err());
}

// ── eval_table ─────────────────────────────────────────────────────────

#[test]
fn eval_table_values_and_export() {
    let (_ctx, x) = setup();
    let table = x.powi(2).eval_table(&x, &[0.0, 1.0, 2.0, 3.0]).unwrap();
    assert_eq!(table.nrows(), 4);
    assert_eq!(table.ncols(), 2);
    assert_eq!(table.headers, vec!["x", "f(x)"]);
    assert_eq!(table.rows[3], vec!["3", "9"]);
    let csv = table.to_csv();
    assert!(csv.starts_with("x,f(x)\n0,0\n1,1\n"));
    assert!(table.to_json().contains("\"f(x)\": 9"));
    assert!(table.to_markdown().contains("| 9"));
}

#[test]
fn eval_table_non_finite_results_are_cells_not_errors() {
    let (ctx, x) = setup();
    let table = (&ctx.one() / &x).eval_table(&x, &[0.0, 1.0]).unwrap();
    assert_eq!(table.rows[0][1], "Inf");
    assert_eq!(table.rows[1][1], "1");
}

#[test]
fn eval_table_empty_points_is_empty_table() {
    let (_ctx, x) = setup();
    let table = x.eval_table(&x, &[]).unwrap();
    assert_eq!(table.nrows(), 0);
}

#[test]
fn eval_table_rejects_non_finite_inputs_and_free_symbols() {
    let (ctx, x) = setup();
    assert!(matches!(
        x.eval_table(&x, &[0.0, f64::NAN]),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        (&x + &ctx.symbol("y")).eval_table(&x, &[0.0]),
        Err(SymplexError::FreeSymbol { .. })
    ));
    let e = x.powi(2);
    assert!(e.eval_table(&e, &[0.0]).is_err());
}

#[test]
fn eval_table_with_trig_uses_compiled_path() {
    let (_ctx, x) = setup();
    let table = x
        .sin()
        .eval_table(&x, &[0.0, std::f64::consts::FRAC_PI_2])
        .unwrap();
    assert_eq!(table.rows[0][1], "0");
    assert_eq!(table.rows[1][1], "1");
}
