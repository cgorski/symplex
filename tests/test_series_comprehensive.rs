//! Comprehensive Taylor/Maclaurin series tests with numerical accuracy verification.

mod common;

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate a series expression at a rational point x = p/q and return the f64 value.
fn eval_series_at(series: &Ex, var: &Ex, p: i64, q: i64) -> f64 {
    let pt = symplex::rational(p, q);
    series
        .subs(var, &pt)
        .eval()
        .eval_f64()
        .expect("series numerical evaluation should succeed")
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. exp(x) Maclaurin series (around x=0), order 6
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn taylor_exp_at_0() {
    let x = symplex::var("x");
    let series = x.exp().maclaurin(&x, 6);
    assert!(series.is_ok(), "exp(x) maclaurin failed: {:?}", series.err());
    let expanded = series.unwrap().expand().eval();

    // Evaluate at x = 1/2
    let val = eval_series_at(&expanded, &x, 1, 2);
    let exact = 0.5_f64.exp();
    assert!(
        (val - exact).abs() < 1e-3,
        "exp series at x=0.5: got {val}, expected {exact}, err={}",
        (val - exact).abs()
    );

    // Also verify at x = 1
    let val_1 = eval_series_at(&expanded, &x, 1, 1);
    let exact_1 = 1.0_f64.exp();
    assert!(
        (val_1 - exact_1).abs() < 0.01,
        "exp series at x=1: got {val_1}, expected {exact_1}, err={}",
        (val_1 - exact_1).abs()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. sin(x) Maclaurin series (around x=0), order 8
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn taylor_sin_at_0() {
    let x = symplex::var("x");
    let series = x.sin().maclaurin(&x, 8);
    assert!(series.is_ok(), "sin(x) maclaurin failed: {:?}", series.err());
    let expanded = series.unwrap().expand().eval();

    // Evaluate at x = 1/2
    let val = eval_series_at(&expanded, &x, 1, 2);
    let exact = 0.5_f64.sin();
    assert!(
        (val - exact).abs() < 1e-6,
        "sin series at x=0.5: got {val}, expected {exact}, err={}",
        (val - exact).abs()
    );

    // Evaluate at x = 1
    let val_1 = eval_series_at(&expanded, &x, 1, 1);
    let exact_1 = 1.0_f64.sin();
    assert!(
        (val_1 - exact_1).abs() < 1e-4,
        "sin series at x=1: got {val_1}, expected {exact_1}, err={}",
        (val_1 - exact_1).abs()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. cos(x) Maclaurin series (around x=0), order 8
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn taylor_cos_at_0() {
    let x = symplex::var("x");
    let series = x.cos().maclaurin(&x, 8);
    assert!(series.is_ok(), "cos(x) maclaurin failed: {:?}", series.err());
    let expanded = series.unwrap().expand().eval();

    // Evaluate at x = 1/2
    let val = eval_series_at(&expanded, &x, 1, 2);
    let exact = 0.5_f64.cos();
    assert!(
        (val - exact).abs() < 1e-6,
        "cos series at x=0.5: got {val}, expected {exact}, err={}",
        (val - exact).abs()
    );

    // Evaluate at x = 1
    let val_1 = eval_series_at(&expanded, &x, 1, 1);
    let exact_1 = 1.0_f64.cos();
    assert!(
        (val_1 - exact_1).abs() < 1e-4,
        "cos series at x=1: got {val_1}, expected {exact_1}, err={}",
        (val_1 - exact_1).abs()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. ln(x) Taylor series around x=1, order 8
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn taylor_ln_at_1() {
    let x = symplex::var("x");
    // ln(x) expanded around x = 1
    let series = x.ln().series(&x, &symplex::int(1), 8);
    assert!(series.is_ok(), "ln(x) series at 1 failed: {:?}", series.err());
    let expanded = series.unwrap().expand().eval();

    // Evaluate at x = 11/10 (i.e. x = 1.1, close to the expansion point)
    let val = eval_series_at(&expanded, &x, 11, 10);
    let exact = 1.1_f64.ln();
    assert!(
        (val - exact).abs() < 1e-6,
        "ln series at x=1.1: got {val}, expected {exact}, err={}",
        (val - exact).abs()
    );

    // Also verify at x = 1 itself: ln(1) = 0
    let val_center = eval_series_at(&expanded, &x, 1, 1);
    assert!(
        val_center.abs() < 1e-12,
        "ln series at x=1 (expansion point) should be 0: got {val_center}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. exp(x) Taylor series around x=1, order 6
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn taylor_exp_at_1() {
    let x = symplex::var("x");
    // exp(x) expanded around x = 1
    let series = x.exp().series(&x, &symplex::int(1), 6);
    assert!(series.is_ok(), "exp(x) series at 1 failed: {:?}", series.err());
    let expanded = series.unwrap().expand().eval();

    // Evaluate at x = 11/10 (x = 1.1)
    let val = eval_series_at(&expanded, &x, 11, 10);
    let exact = 1.1_f64.exp();
    assert!(
        (val - exact).abs() < 1e-4,
        "exp series(at 1) at x=1.1: got {val}, expected {exact}, err={}",
        (val - exact).abs()
    );

    // Verify at expansion point x=1: exp(1) = e
    let val_center = eval_series_at(&expanded, &x, 1, 1);
    let exact_center = 1.0_f64.exp();
    assert!(
        (val_center - exact_center).abs() < 1e-10,
        "exp series at x=1 (expansion point) should be e: got {val_center}, expected {exact_center}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. sinh(x) Maclaurin series, order 8
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn taylor_sinh_at_0() {
    let x = symplex::var("x");
    let series = x.sinh().maclaurin(&x, 8);
    assert!(series.is_ok(), "sinh(x) maclaurin failed: {:?}", series.err());
    let expanded = series.unwrap().expand().eval();

    // sinh(x) = x + x^3/6 + x^5/120 + x^7/5040 + ...
    // Evaluate at x = 1/2
    let val = eval_series_at(&expanded, &x, 1, 2);
    let exact = 0.5_f64.sinh();
    assert!(
        (val - exact).abs() < 1e-6,
        "sinh series at x=0.5: got {val}, expected {exact}, err={}",
        (val - exact).abs()
    );

    // Evaluate at x = 1
    let val_1 = eval_series_at(&expanded, &x, 1, 1);
    let exact_1 = 1.0_f64.sinh();
    assert!(
        (val_1 - exact_1).abs() < 1e-4,
        "sinh series at x=1: got {val_1}, expected {exact_1}, err={}",
        (val_1 - exact_1).abs()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. cosh(x) Maclaurin series, order 8
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn taylor_cosh_at_0() {
    let x = symplex::var("x");
    let series = x.cosh().maclaurin(&x, 8);
    assert!(series.is_ok(), "cosh(x) maclaurin failed: {:?}", series.err());
    let expanded = series.unwrap().expand().eval();

    // cosh(x) = 1 + x^2/2 + x^4/24 + x^6/720 + ...
    // Evaluate at x = 1/2
    let val = eval_series_at(&expanded, &x, 1, 2);
    let exact = 0.5_f64.cosh();
    assert!(
        (val - exact).abs() < 1e-6,
        "cosh series at x=0.5: got {val}, expected {exact}, err={}",
        (val - exact).abs()
    );

    // Evaluate at x = 1
    let val_1 = eval_series_at(&expanded, &x, 1, 1);
    let exact_1 = 1.0_f64.cosh();
    assert!(
        (val_1 - exact_1).abs() < 1e-4,
        "cosh series at x=1: got {val_1}, expected {exact_1}, err={}",
        (val_1 - exact_1).abs()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Composition: exp(sin(x)) — no fast path, order 6
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn taylor_composition_exp_sin() {
    let x = symplex::var("x");
    // exp(sin(x)) — a composition that doesn't match a single known pattern,
    // requiring the general Taylor expansion machinery.
    let expr = x.sin().exp();
    let series = expr.maclaurin(&x, 6);
    assert!(
        series.is_ok(),
        "exp(sin(x)) maclaurin failed: {:?}",
        series.err()
    );
    let expanded = series.unwrap().expand().eval();

    // exp(sin(x)) at x=0: exp(sin(0)) = exp(0) = 1
    let val_0 = eval_series_at(&expanded, &x, 0, 1);
    assert!(
        (val_0 - 1.0).abs() < 1e-10,
        "exp(sin(x)) series at x=0 should be 1: got {val_0}"
    );

    // exp(sin(x)) at x = 1/2
    let val = eval_series_at(&expanded, &x, 1, 2);
    let exact = 0.5_f64.sin().exp();
    assert!(
        (val - exact).abs() < 1e-3,
        "exp(sin(x)) series at x=0.5: got {val}, expected {exact}, err={}",
        (val - exact).abs()
    );

    // exp(sin(x)) at x = 1/5 (smaller value, should be very accurate)
    let val_small = eval_series_at(&expanded, &x, 1, 5);
    let exact_small = 0.2_f64.sin().exp();
    assert!(
        (val_small - exact_small).abs() < 1e-5,
        "exp(sin(x)) series at x=0.2: got {val_small}, expected {exact_small}, err={}",
        (val_small - exact_small).abs()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Verify sin(x) series error decreases with order
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_numerical_accuracy_sin() {
    // At x = 1/2, increasing the order of the Maclaurin series for sin(x)
    // should produce strictly decreasing approximation error.
    let x = symplex::var("x");
    let exact = 0.5_f64.sin();
    let orders = [3u32, 5, 7, 9, 11];
    let mut prev_err = f64::MAX;

    for &order in &orders {
        let series = x.sin().maclaurin(&x, order);
        if let Ok(s) = series {
            let expanded = s.expand().eval();
            if let Ok(val) = expanded
                .subs(&x, &symplex::rational(1, 2))
                .eval()
                .eval_f64()
            {
                let err = (val - exact).abs();
                assert!(
                    err < prev_err,
                    "sin series error should decrease: order {order} err={err} >= prev_err={prev_err}"
                );
                prev_err = err;
            }
        }
    }
    // After all orders, the error should be very small
    assert!(
        prev_err < 1e-6,
        "sin series at order 11 should be very accurate, err={prev_err}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Verify exp(x) series error decreases with order
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_numerical_accuracy_exp() {
    // At x = 1/2, increasing the order of the Maclaurin series for exp(x)
    // should produce strictly decreasing approximation error.
    let x = symplex::var("x");
    let exact = 0.5_f64.exp();
    let orders = [2u32, 4, 6, 8, 10];
    let mut prev_err = f64::MAX;

    for &order in &orders {
        let series = x.exp().maclaurin(&x, order);
        if let Ok(s) = series {
            let expanded = s.expand().eval();
            if let Ok(val) = expanded
                .subs(&x, &symplex::rational(1, 2))
                .eval()
                .eval_f64()
            {
                let err = (val - exact).abs();
                assert!(
                    err < prev_err,
                    "exp series error should decrease: order {order} err={err} >= prev_err={prev_err}"
                );
                prev_err = err;
            }
        }
    }
    // After all orders, the error should be very small
    assert!(
        prev_err < 1e-8,
        "exp series at order 10 should be very accurate, err={prev_err}"
    );
}
