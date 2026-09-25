//! symplex 0.2 — limits: the standard battery (two-sided, one-sided, at
//! ±∞), `try_limit` semantics, non-existent limits, and the work budget.
//!
//! Every limit is verified numerically by approaching the point along the
//! requested direction with a *rate-aware* check: fast-converging limits
//! must match closely, slowly-converging ones (`√x·ln x` at `0⁺`,
//! `ln x / x^{1/3}` at `∞`) must at least approach the claimed value
//! monotonically. Sampling uses compiled `f64` evaluation so that huge
//! arguments (`Γ(10⁵)`) never trigger exact big-integer arithmetic.
//!
//! Tests are deliberately small (≤ 5 limits each) so that a failure or a
//! hang is immediately attributable.

use symplex::prelude::*;

/// Wall-clock hang guard.  Two seconds on a developer machine; scaled up on
/// shared CI runners (`CI` is set), which are several times slower and noisy.
fn time_budget(secs: u64) -> std::time::Duration {
    let mult = if std::env::var_os("CI").is_some() {
        5
    } else {
        1
    };
    std::time::Duration::from_secs(secs * mult)
}

// ═══════════════════════════════════════════════════════════════════════════
// Harness
// ═══════════════════════════════════════════════════════════════════════════

/// How a limit point is approached numerically.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
    Both,
}

/// Numeric samples approaching `point` from `side` (closest last).
///
/// At `±∞` the samples start small (5, 10, 20, …) because `f64` overflows
/// or cancels catastrophically for many exponential expressions well before
/// `10⁸`; the convergence checks below ignore non-finite samples. At a
/// finite point the offsets stop at `10⁻⁶`, beyond which cancellation in
/// `x − sin x`-type numerators destroys all significant digits.
fn sample_points(point: &Ex, side: Side) -> Vec<Vec<f64>> {
    let p = format!("{point}");
    let big = [5.0, 10.0, 20.0, 50.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8];
    if p == "oo" {
        return vec![big.to_vec()];
    }
    if p == "-oo" {
        return vec![big.iter().map(|v| -v).collect()];
    }
    let p = point
        .eval_f64()
        .unwrap_or_else(|e| panic!("finite limit point {point} must be numeric: {e}"));
    let hs: Vec<f64> = (1..=6).map(|k| 10f64.powi(-k)).collect();
    let right: Vec<f64> = hs.iter().map(|h| p + h).collect();
    let left: Vec<f64> = hs.iter().map(|h| p - h).collect();
    match side {
        Side::Right => vec![right],
        Side::Left => vec![left],
        Side::Both => vec![right, left],
    }
}

/// Rate-aware convergence check of the samples `vals` (closest last)
/// against a finite target.
///
/// Non-finite samples (overflow) are ignored. The check passes when
///
/// 1. some sample is within `10⁻⁶·scale` of the target (fast convergence;
///    catastrophic cancellation may destroy the *later* samples, as in
///    `(sin x − x + x³/6 − x⁵/120)/x⁷`), or
/// 2. the best of the last three finite samples is within `10⁻⁴·scale`, or
/// 3. (slow convergence, e.g. `√x·ln x` at `0⁺` or `W(x)/ln x` at `∞`) the
///    last four finite errors decrease monotonically, the last one is below
///    `0.3·scale`, and it is at most 85 % of the fourth-from-last.
fn check_finite_convergence(vals: &[f64], target: f64, label: &str) {
    let errs: Vec<f64> = vals
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - target).abs())
        .collect();
    assert!(!errs.is_empty(), "{label}: no finite samples ({vals:?})");
    let scale = target.abs().max(1.0);
    if errs.iter().any(|e| *e <= 1e-6 * scale) {
        return;
    }
    if errs.len() >= 3 {
        let tail_min = errs[errs.len() - 3..]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if tail_min <= 1e-4 * scale {
            return;
        }
    }
    if errs.len() >= 4 {
        let tail = &errs[errs.len() - 4..];
        let decreasing = tail.windows(2).all(|w| w[1] < w[0]);
        let last = tail[3];
        if decreasing && last <= 0.3 * scale && last <= 0.85 * tail[0] {
            return;
        }
    }
    panic!("{label}: samples {vals:?} do not converge to {target} (errors {errs:?})");
}

/// The samples must have the right sign and grow without bound.
/// `±inf` samples (overflow) of the right sign count as diverging.
fn check_divergence(vals: &[f64], positive: bool, label: &str) {
    let samples: Vec<f64> = vals.iter().cloned().filter(|v| !v.is_nan()).collect();
    assert!(samples.len() >= 2, "{label}: too few samples ({vals:?})");
    let n = samples.len();
    for v in &samples[n.saturating_sub(3)..] {
        assert!(
            if positive { *v > 0.0 } else { *v < 0.0 },
            "{label}: expected → {}∞ but sampled {v} ({vals:?})",
            if positive { "+" } else { "−" }
        );
    }
    let first = samples[0].abs();
    let last = samples[n - 1].abs();
    assert!(
        last.is_infinite() || (last > first && last >= 10.0),
        "{label}: |values| not diverging ({vals:?})"
    );
}

/// Verify a computed limit `lim` of `expr` numerically along `side`.
///
/// `params` are substituted (exactly) into both `expr` and `lim` first so
/// that symbolic parameters with assumptions can be checked too.
fn verify_numerically(expr: &Ex, var: &Ex, point: &Ex, side: Side, lim: &Ex, label: &str) {
    let var_name = format!("{var}");
    let f = match expr.compile(&[&var_name]) {
        Ok(f) => f,
        Err(e) => panic!("{label}: cannot compile {expr} for numeric verification: {e}"),
    };
    let lim_s = format!("{lim}");
    let target = if lim_s == "oo" || lim_s == "-oo" {
        None
    } else {
        Some(
            lim.eval_f64()
                .unwrap_or_else(|e| panic!("{label}: limit {lim} not numeric: {e}")),
        )
    };
    for pts in sample_points(point, side) {
        let vals: Vec<f64> = pts.iter().map(|p| f.call(&[*p])).collect();
        match target {
            Some(t) => check_finite_convergence(&vals, t, label),
            None => check_divergence(&vals, lim_s == "oo", label),
        }
    }
}

/// Compute the limit of `expr` along `side`, compare with `expected`
/// (as a string, or numerically when the display differs), check that no
/// internal symbol leaked, and verify numerically after substituting
/// `params`.
fn check_limit(expr: &Ex, var: &Ex, point: &Ex, side: Side, expected: &str, params: &[(&Ex, i64)]) {
    let ctx = expr.context();
    let label = format!("lim_{{{var}→{point}{}}} {expr}", side_suffix(side));
    let r = match side {
        Side::Both => expr.limit(var, point),
        Side::Left => expr.limit_left(var, point),
        Side::Right => expr.limit_right(var, point),
    };
    let got = format!("{r}");
    assert!(!r.has_unevaluated(), "{label}: unevaluated ({got})");
    assert_no_internal_symbols(&r, &label);
    assert!(
        !got.contains("zoo") && !got.contains("nan"),
        "{label}: indeterminate result {got}"
    );

    let expected_ex = ctx
        .parse(expected)
        .unwrap_or_else(|e| panic!("{label}: bad expected string {expected:?}: {e}"));
    let same_string = got == expected;
    let same_numeric = matches!(
        (r.eval_f64(), expected_ex.eval_f64()),
        (Ok(g), Ok(e)) if (g - e).abs() <= 1e-9 * e.abs().max(1.0)
    );
    let same_symbolic = (&r - &expected_ex).simplify().is_zero_structural();
    assert!(
        same_string || same_numeric || same_symbolic,
        "{label}: got {got}, expected {expected}"
    );

    // Numeric verification with parameters bound to concrete values.
    let mut e_num = expr.clone();
    let mut r_num = r.clone();
    let mut p_num = point.clone();
    for (p, v) in params {
        let val = ctx.int(*v);
        e_num = e_num.subs(p, &val);
        r_num = r_num.subs(p, &val);
        p_num = p_num.subs(p, &val);
    }
    verify_numerically(&e_num, var, &p_num, side, &r_num, &label);
}

fn side_suffix(side: Side) -> &'static str {
    match side {
        Side::Left => "⁻",
        Side::Right => "⁺",
        Side::Both => "",
    }
}

/// No internal dummy (`__gw…`, `__lim…`, `__limit_t`) may ever appear in a
/// result handed to the user.
fn assert_no_internal_symbols(r: &Ex, label: &str) {
    let s = format!("{r}");
    assert!(!s.contains("__"), "{label}: internal symbol leaked: {s}");
    for sym in r.free_symbols() {
        let name = format!("{sym}");
        assert!(
            !name.starts_with("__"),
            "{label}: internal symbol {name} in result {s}"
        );
    }
}

/// Two-sided limit check with numeric verification from both sides.
fn both(expr: &Ex, var: &Ex, point: &Ex, expected: &str) {
    check_limit(expr, var, point, Side::Both, expected, &[]);
}

/// Right-hand limit check.
fn right(expr: &Ex, var: &Ex, point: &Ex, expected: &str) {
    check_limit(expr, var, point, Side::Right, expected, &[]);
}

/// Left-hand limit check.
fn left(expr: &Ex, var: &Ex, point: &Ex, expected: &str) {
    check_limit(expr, var, point, Side::Left, expected, &[]);
}

/// Two-sided limit that is *computed two-sided* but only verified
/// numerically from the right (the left side leaves the real domain, e.g.
/// `x ln x`, `xˣ`).
fn both_verify_right(expr: &Ex, var: &Ex, point: &Ex, expected: &str) {
    let r = expr.limit(var, point);
    assert!(!r.has_unevaluated(), "{expr}: unevaluated ({r})");
    assert_eq!(format!("{r}"), expected, "{expr}");
    check_limit(expr, var, point, Side::Right, expected, &[]);
}

/// The limit must not exist / not be computable: unevaluated `Limit` node
/// and `Err` from `try_limit`.
fn no_limit(expr: &Ex, var: &Ex, point: &Ex) {
    let r = expr.limit(var, point);
    assert!(
        r.has_unevaluated(),
        "{expr} at {point}: expected unevaluated, got {r}"
    );
    assert!(
        format!("{r}").starts_with("Limit("),
        "{expr} at {point}: unevaluated form should be the formal Limit node, got {r}"
    );
    assert!(
        expr.try_limit(var, point).is_err(),
        "{expr} at {point}: try_limit should be Err"
    );
    assert_no_internal_symbols(&r, &format!("{expr}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Two-sided limits at 0 — L'Hôpital / series classics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn classics_trig_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    both(&(&x.sin() / &x), &x, &z, "1");
    both(&((1 - x.cos()) / x.powi(2)), &x, &z, "1/2");
    both(&((&x - x.sin()) / x.powi(3)), &x, &z, "1/6");
    both(&((x.tan() - x.sin()) / x.powi(3)), &x, &z, "1/2");
    both(&((1 - (&x * 2).cos()) / x.powi(2)), &x, &z, "2");
}

#[test]
fn classics_exp_log_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    both(&((x.exp() - 1 - &x) / x.powi(2)), &x, &z, "1/2");
    both(&((x.exp() - 1) / &x), &x, &z, "1");
    both(&((1 + &x).ln() / &x), &x, &z, "1");
    both(&((x.exp() - 1) / x.sin()), &x, &z, "1");
    both(
        &((ctx.int(2).pow(&x) - ctx.int(3).pow(&x)) / &x),
        &x,
        &z,
        "ln(2) - ln(3)",
    );
}

#[test]
fn classics_algebraic_and_inverse_trig_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    both(&(((1 + &x).sqrt() - 1) / &x), &x, &z, "1/2");
    both(&((x.cos() - 1) / (&x * x.sin())), &x, &z, "-1/2");
    both(&(x.asin() / &x), &x, &z, "1");
    both(&(x.tan() / &x), &x, &z, "1");
    both(&(x.sec()), &x, &z, "1");
}

#[test]
fn classics_hyperbolic_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    both(&(x.sinh() / &x), &x, &z, "1");
    both(&((x.cosh() - 1) / x.powi(2)), &x, &z, "1/2");
    both(&(x.tanh() / &x), &x, &z, "1");
    both(&(x.asinh() / &x), &x, &z, "1");
}

#[test]
fn deep_cancellations_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    both(&(x.cot() - 1 / &x), &x, &z, "0");
    both(&(1 / x.sin() - 1 / &x), &x, &z, "0");
    both(&(1 / x.powi(2) - 1 / x.sin().powi(2)), &x, &z, "-1/3");
    both(
        &((x.exp() - 1 - &x - x.powi(2) / 2 - x.powi(3) / 6) / x.powi(4)),
        &x,
        &z,
        "1/24",
    );
    both(
        &((x.sin() - &x + x.powi(3) / 6 - x.powi(5) / 120) / x.powi(7)),
        &x,
        &z,
        "-1/5040",
    );
}

#[test]
fn nested_function_cancellations_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    both(&(x.sin().sin().sin().sin() / &x), &x, &z, "1");
    both(
        &((x.sin().sin().sin() - x.tan().tan().tan()) / x.powi(3)),
        &x,
        &z,
        "-3/2",
    );
    both(
        &(((x.sin().sin() - x.tan().tan()) / x.powi(3)).exp() - 1),
        &x,
        &z,
        "exp(-1) - 1",
    );
}

#[test]
fn one_to_the_infinity_forms_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    both(&((1 + &x).pow(&(1 / &x))), &x, &z, "E");
    both(&((1 + x.sin()).pow(&(1 / &x))), &x, &z, "E");
    both(&(x.cos().pow(&(1 / x.powi(2)))), &x, &z, "exp(-1/2)");
    both(
        &(((1 + &x).pow(&(1 / &x)) - ctx.e()) / &x),
        &x,
        &z,
        "-1/2*E",
    );
}

#[test]
fn bounded_oscillation_times_vanishing_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    both(&(&x * (1 / &x).sin()), &x, &z, "0");
    both(&(x.powi(2) * (1 / &x).sin()), &x, &z, "0");
    both(&(&x * (1 / &x).cos()), &x, &z, "0");
    both(&(x.sin().powi(2) / &x), &x, &ctx.infinity(), "0");
}

#[test]
fn logarithmic_forms_at_zero_plus() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    both_verify_right(&(&x * x.ln()), &x, &z, "0");
    both_verify_right(&(x.powi(2) * x.ln()), &x, &z, "0");
    both_verify_right(&(x.pow(&x)), &x, &z, "1");
    right(&(x.sqrt() * x.ln()), &x, &z, "0");
    right(&((x.pow(&x) - 1) / (&x * x.ln())), &x, &z, "1");
}

#[test]
fn continuous_and_directional_nodes_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    both(&x.abs(), &x, &z, "0");
    right(&x.sqrt(), &x, &z, "0");
    both(&x.max_with(&ctx.int(1)), &x, &z, "1");
    both(&x.floor(), &x, &ctx.rational(1, 2), "0");
    both(&(x.abs() / x.powi(2)), &x, &z, "oo");
}

// ═══════════════════════════════════════════════════════════════════════════
// Two-sided limits at other finite points
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rational_and_log_forms_at_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    both(&((x.powi(3) - 1) / (&x - 1)), &x, &one, "3");
    both(&((x.powi(2) - 1) / (&x - 1)), &x, &one, "2");
    both(&(x.ln() / (&x - 1)), &x, &one, "1");
    both(&((&x - 1) / x.ln()), &x, &one, "1");
    both(&(1 / (&x - 1).powi(2)), &x, &one, "oo");
}

#[test]
fn direct_substitution_points() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    both(&x.sin(), &x, &ctx.int(1), "sin(1)");
    both(&(x.exp() + ctx.pi()), &x, &ctx.int(0), "1 + pi");
    both(&x.gamma(), &x, &ctx.int(3), "2");
    both(&(x.powi(2) + 1), &x, &ctx.int(2), "5");
    both(&x.atan(), &x, &ctx.int(1), "1/4*pi");
}

// ═══════════════════════════════════════════════════════════════════════════
// Limits at +∞
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn one_to_the_infinity_forms_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&(x.pow(&(1 / &x))), &x, &inf, "1");
    both(&((1 + 1 / &x).pow(&x)), &x, &inf, "E");
    both(&((1 + 2 / &x).pow(&(&x * 3))), &x, &inf, "exp(6)");
    both(&((1 + 1 / &x).pow(&x.powi(2))), &x, &inf, "oo");
    both(&(x.pow(&(1 / x.ln()))), &x, &inf, "E");
}

#[test]
fn radical_differences_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&((x.powi(2) + &x).sqrt() - &x), &x, &inf, "1/2");
    both(&(&x - (x.powi(2) - 1).sqrt()), &x, &inf, "0");
    both(&(x.sqrt() * ((&x + 1).sqrt() - x.sqrt())), &x, &inf, "1/2");
    both(
        &((x.powi(3) + x.powi(2)).pow(&ctx.rational(1, 3)) - &x),
        &x,
        &inf,
        "1/3",
    );
    both(
        &(x.pow(&ctx.rational(3, 2)) * ((&x + 1).sqrt() + (&x - 1).sqrt() - 2 * x.sqrt())),
        &x,
        &inf,
        "-1/4",
    );
}

#[test]
fn logarithms_vs_powers_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&(x.ln() / x.sqrt()), &x, &inf, "0");
    both(&(x.ln() / x.pow(&ctx.rational(1, 3))), &x, &inf, "0");
    both(&(x.ln() / &x), &x, &inf, "0");
    both(&(x.ln().powi(2) / &x), &x, &inf, "0");
    both(&(x.ln().ln() / x.ln()), &x, &inf, "0");
}

#[test]
fn powers_vs_exponentials_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&(&x * (-&x).exp()), &x, &inf, "0");
    both(&(x.powi(2) * (-&x).exp()), &x, &inf, "0");
    both(&(x.powi(5) / x.exp()), &x, &inf, "0");
    both(&((-x.powi(2)).exp()), &x, &inf, "0");
    both(&(x.exp() / x.pow(&x)), &x, &inf, "0");
}

#[test]
fn sums_of_exponentials_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&(x.exp() - (&x * 2).exp()), &x, &inf, "-oo");
    both(&((x.exp() + (-&x).exp()) / x.exp()), &x, &inf, "1");
    both(&(x.exp() / (x.exp() + 1)), &x, &inf, "1");
    both(&(x.sinh() / x.exp()), &x, &inf, "1/2");
    both(
        &((x.exp() + (-&x).exp()) / (x.exp() - (-&x).exp())),
        &x,
        &inf,
        "1",
    );
}

#[test]
fn nested_exponentials_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&((&x - (-&x).exp()).exp() - x.exp()), &x, &inf, "-1");
    both(
        &(x.powi(2).exp() - (x.powi(2) + 1 / &x).exp()),
        &x,
        &inf,
        "-oo",
    );
    both(&(x.exp().exp() / x.powi(2).exp()), &x, &inf, "oo");
    both(&(x.exp().lambertw() / &x), &x, &inf, "1");
}

#[test]
fn rational_functions_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&((x.powi(2) + 1) / (&x + 1)), &x, &inf, "oo");
    both(&((&x.powi(2) * 3 + 1) / (x.powi(2) - &x)), &x, &inf, "3");
    both(&(x.powi(2) - x.powi(3)), &x, &inf, "-oo");
    both(&(&x / (x.powi(2) + 1)), &x, &inf, "0");
    both(&(x.powi(2) / (x.exp() - 1)), &x, &inf, "0");
}

#[test]
fn asymptotic_constants_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&x.atan(), &x, &inf, "1/2*pi");
    both(&x.erf(), &x, &inf, "1");
    both(&x.tanh(), &x, &inf, "1");
    both(&x.erfc(), &x, &inf, "0");
    both(&(&x * x.tanh()), &x, &inf, "oo");
}

#[test]
fn special_functions_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&(x.lambertw() / x.ln()), &x, &inf, "1");
    both(&x.gamma(), &x, &inf, "oo");
    both(&(1 / x.gamma()), &x, &inf, "0");
    both(&x.min_with(&ctx.int(1)), &x, &inf, "1");
}

#[test]
fn logarithmic_differences_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&((&x + 1).ln() - x.ln()), &x, &inf, "0");
    both(&(&x * ((&x + 1).ln() - x.ln())), &x, &inf, "1");
    both(&(x.asinh() - x.ln()), &x, &inf, "ln(2)");
    both(&(&x - x.powi(2) * (1 + 1 / &x).ln()), &x, &inf, "1/2");
    both(&((x.powi(2) + 1).ln() / x.ln()), &x, &inf, "2");
}

#[test]
fn oscillation_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    both(&(&x * (1 / &x).sin()), &x, &inf, "1");
    both(&(x.cos() / &x), &x, &inf, "0");
    both(&(x.sin() / &x), &x, &inf, "0");
    both(&((-&x).exp() * x.cos()), &x, &inf, "0");
    both(&((&x * (1 / &x).sin() - 1) * x.powi(2)), &x, &inf, "-1/6");
}

// ═══════════════════════════════════════════════════════════════════════════
// Limits at −∞
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn asymptotic_constants_at_negative_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let ninf = ctx.neg_infinity();
    both(&x.atan(), &x, &ninf, "-1/2*pi");
    both(&x.erf(), &x, &ninf, "-1");
    both(&x.tanh(), &x, &ninf, "-1");
    both(&x.erfc(), &x, &ninf, "2");
    both(&x.exp(), &x, &ninf, "0");
}

#[test]
fn growth_at_negative_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let ninf = ctx.neg_infinity();
    both(&x.powi(3), &x, &ninf, "-oo");
    both(&(&x * x.exp()), &x, &ninf, "0");
    both(&(x.powi(3) * x.exp()), &x, &ninf, "0");
    both(&(x.exp() / (x.exp() + 1)), &x, &ninf, "0");
    both(&(x.exp() * x.sin()), &x, &ninf, "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic parameters with assumptions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn symbolic_parameters_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let b = ctx.symbol_with("b", &[Assumption::Positive]).unwrap();
    let z = ctx.int(0);
    let p = [(&a, 2), (&b, 5)];
    check_limit(&a, &x, &z, Side::Both, "a", &p);
    check_limit(&((a.pow(&x) - 1) / &x), &x, &z, Side::Both, "ln(a)", &p);
    check_limit(&((&a * &x).sin() / &x), &x, &z, Side::Both, "a", &p);
    check_limit(
        &((a.pow(&x) - b.pow(&x)) / &x),
        &x,
        &z,
        Side::Both,
        "ln(a) - ln(b)",
        &p,
    );
}

#[test]
fn symbolic_parameters_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let b = ctx.symbol_with("b", &[Assumption::Positive]).unwrap();
    let inf = ctx.infinity();
    let p = [(&a, 2), (&b, 3)];
    check_limit(
        &((1 + &a / &x).pow(&(&b * &x))),
        &x,
        &inf,
        Side::Both,
        "exp(a*b)",
        &p,
    );
    check_limit(&((&x + &a) / (&x - &a)), &x, &inf, Side::Both, "1", &p);
    check_limit(&(&a * &x), &x, &inf, Side::Both, "oo", &p);
    check_limit(&(&a / &x), &x, &inf, Side::Both, "0", &p);
    check_limit(&((1 + &a / &x).pow(&x)), &x, &inf, Side::Both, "exp(a)", &p);
}

/// Reported wrong on `main`: `e^{−ax}` with `a > 0` at `∞` returned `oo`.
#[test]
fn decaying_exponential_with_positive_parameter() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let c = ctx.symbol_with("c", &[Assumption::Negative]).unwrap();
    let n = ctx.symbol_with("n", &[Assumption::Positive]).unwrap();
    let eps = ctx.symbol_with("eps", &[Assumption::Positive]).unwrap();
    let inf = ctx.infinity();
    check_limit(&((-&a * &x).exp()), &x, &inf, Side::Both, "0", &[(&a, 3)]);
    check_limit(&((&c * &x).exp()), &x, &inf, Side::Both, "0", &[(&c, -2)]);
    check_limit(&((&a * &x).exp()), &x, &inf, Side::Both, "oo", &[(&a, 3)]);
    check_limit(
        &(x.ln() / x.pow(&eps)),
        &x,
        &inf,
        Side::Both,
        "0",
        &[(&eps, 1)],
    );
    check_limit(
        &(x.pow(&n) * (-&x).exp()),
        &x,
        &inf,
        Side::Both,
        "0",
        &[(&n, 4)],
    );
}

#[test]
fn unknown_sign_parameters_are_not_guessed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let u = ctx.symbol("u");
    let inf = ctx.infinity();
    assert!((&u * &x).limit(&x, &inf).has_unevaluated());
    assert!((&u * &x).exp().limit(&x, &inf).has_unevaluated());
    assert!((&u / &x).limit(&x, &ctx.int(0)).has_unevaluated());
}

// ═══════════════════════════════════════════════════════════════════════════
// One-sided limits
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn one_sided_reciprocals_and_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    right(&(1 / &x), &x, &z, "oo");
    left(&(1 / &x), &x, &z, "-oo");
    right(&(x.abs() / &x), &x, &z, "1");
    left(&(x.abs() / &x), &x, &z, "-1");
    right(&(&x / x.abs()), &x, &z, "1");
    left(&(&x / x.abs()), &x, &z, "-1");
}

#[test]
fn one_sided_exponentials_of_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    right(&((1 / &x).exp()), &x, &z, "oo");
    left(&((1 / &x).exp()), &x, &z, "0");
    right(&(1 / (1 + (1 / &x).exp())), &x, &z, "0");
    left(&(1 / (1 + (1 / &x).exp())), &x, &z, "1");
    right(&((-1 / &x).exp() / &x), &x, &z, "0");
    left(&((-1 / &x).exp() / &x), &x, &z, "-oo");
}

#[test]
fn one_sided_floor_and_ceiling_at_integers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    right(&x.floor(), &x, &two, "2");
    left(&x.floor(), &x, &two, "1");
    right(&x.ceiling(), &x, &two, "3");
    left(&x.ceiling(), &x, &two, "2");
    right(&(x.floor() / &x), &x, &two, "1");
    left(&(x.floor() / &x), &x, &two, "1/2");
}

#[test]
fn one_sided_tan_at_half_pi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half_pi = ctx.pi() / 2;
    right(&x.tan(), &x, &half_pi, "-oo");
    left(&x.tan(), &x, &half_pi, "oo");
    right(&x.sec(), &x, &half_pi, "-oo");
    left(&x.sec(), &x, &half_pi, "oo");
}

#[test]
fn one_sided_heaviside_and_sign() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    let one = ctx.int(1);
    right(&x.heaviside(), &x, &z, "1");
    left(&x.heaviside(), &x, &z, "0");
    right(&x.sign(), &x, &z, "1");
    left(&x.sign(), &x, &z, "-1");
    right(&(x.powi(2) - 1).sign(), &x, &one, "1");
    left(&(x.powi(2) - 1).sign(), &x, &one, "-1");
}

#[test]
fn one_sided_poles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let z = ctx.int(0);
    right(&(1 / (&x - 1).powi(3)), &x, &one, "oo");
    left(&(1 / (&x - 1).powi(3)), &x, &one, "-oo");
    right(&(1 / (&x - 1).powi(2)), &x, &one, "oo");
    left(&(1 / (&x - 1).powi(2)), &x, &one, "oo");
    right(&x.gamma(), &x, &z, "oo");
    left(&x.gamma(), &x, &z, "-oo");
}

#[test]
fn one_sided_atan_of_reciprocal_and_abs_shift() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    let one = ctx.int(1);
    right(&((1 / &x).atan()), &x, &z, "1/2*pi");
    left(&((1 / &x).atan()), &x, &z, "-1/2*pi");
    right(&((&x - 1).abs() / (&x - 1)), &x, &one, "1");
    left(&((&x - 1).abs() / (&x - 1)), &x, &one, "-1");
}

#[test]
fn right_only_logarithmic_forms_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    right(&x.ln(), &x, &z, "-oo");
    right(&(x.ln() / &x), &x, &z, "-oo");
    right(&(x.pow(&(1 / &x))), &x, &z, "0");
    right(&(x.pow(&(-&x))), &x, &z, "1");
    check_limit(&((&x - &a).ln()), &x, &a, Side::Right, "-oo", &[(&a, 2)]);
}

#[test]
fn one_sided_limits_equal_two_sided_for_continuous_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    right(&(&x.sin() / &x), &x, &z, "1");
    left(&(&x.sin() / &x), &x, &z, "1");
    right(&((x.exp() - 1) / &x), &x, &z, "1");
    left(&((x.exp() - 1) / &x), &x, &z, "1");
    // Direction is irrelevant at ±∞.
    right(&x.atan(), &x, &ctx.infinity(), "1/2*pi");
    left(&x.atan(), &x, &ctx.neg_infinity(), "-1/2*pi");
}

// ═══════════════════════════════════════════════════════════════════════════
// Piecewise
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn piecewise_limits_use_the_approaching_branch() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    // f(x) = x²  (x < 1),  2x − 1 (x ≥ 1)  — continuous at 1
    let f = Ex::piecewise(&[(&x.powi(2), &x.lt(&one)), (&(&x * 2 - 1), &one.le(&x))]);
    left(&f, &x, &one, "1");
    right(&f, &x, &one, "1");
    both(&f, &x, &one, "1");

    // g(x) = x²  (x < 1),  2x (x ≥ 1) — jump at 1
    let g = Ex::piecewise(&[(&x.powi(2), &x.lt(&one)), (&(&x * 2), &one.le(&x))]);
    right(&g, &x, &one, "2");
    left(&g, &x, &one, "1");
    no_limit(&g, &x, &one);
}

#[test]
fn piecewise_jump_and_far_from_boundary() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let one = ctx.int(1);
    // h(x) = −1 (x < 0), 1 (x ≥ 0)
    let h = Ex::piecewise(&[(&ctx.int(-1), &x.lt(&zero)), (&one, &zero.le(&x))]);
    left(&h, &x, &zero, "-1");
    right(&h, &x, &zero, "1");
    no_limit(&h, &x, &zero);
    // Away from the boundary the limit is plain substitution.
    both(&h, &x, &ctx.int(5), "1");
    both(&h, &x, &ctx.int(-5), "-1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Limits that do not exist
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn no_limit_jump_discontinuities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    no_limit(&(1 / &x), &x, &z);
    no_limit(&(x.abs() / &x), &x, &z);
    no_limit(&(&x / x.abs()), &x, &z);
    no_limit(&x.heaviside(), &x, &z);
    no_limit(&x.sign(), &x, &z);
}

#[test]
fn no_limit_essential_singularities_and_poles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = ctx.int(0);
    no_limit(&((1 / &x).exp()), &x, &z);
    no_limit(&(1 / (1 + (1 / &x).exp())), &x, &z);
    no_limit(&x.tan(), &x, &(ctx.pi() / 2));
    no_limit(&(1 / (&x - 1).powi(3)), &x, &ctx.int(1));
    no_limit(&((1 / &x).atan()), &x, &z);
}

#[test]
fn no_limit_oscillation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    no_limit(&((1 / &x).sin()), &x, &ctx.int(0));
    no_limit(&x.sin(), &x, &inf);
    no_limit(&(&x * x.sin()), &x, &inf);
    no_limit(&(&x * x.powi(2).sin()), &x, &inf);
    no_limit(&x.floor(), &x, &ctx.int(2));
}

/// Functions of a bounded oscillation that are themselves unbounded must
/// not be treated as bounded (`sin(1/x)^x → 1` was a wrong answer).
#[test]
fn no_limit_unbounded_functions_of_oscillation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    assert!(
        (1 / &x)
            .sin()
            .pow(&x)
            .limit_right(&x, &ctx.int(0))
            .has_unevaluated()
    );
    assert!(
        (1 / &x)
            .sin()
            .pow(&x)
            .try_limit_right(&x, &ctx.int(0))
            .is_err()
    );
    no_limit(&(1 / (&x * x.sin())), &x, &inf);
    no_limit(&((2 + x.sin()).ln() / &x), &x, &inf);
    no_limit(&x.erf().pow(&x), &x, &inf);
}

// ═══════════════════════════════════════════════════════════════════════════
// Not wrong: hard limits the engine cannot do must stay unevaluated
// ═══════════════════════════════════════════════════════════════════════════

/// `Γ(x+1)/(x·Γ(x)) → 0` and `lnΓ(x)/(x ln x) → 0` were wrong answers
/// produced by a raw `subs(t = 0)` folding `0·Γ(zoo) → 0`.
#[test]
fn gamma_ratios_are_not_answered_wrongly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    for e in [
        (&x + 1).gamma() / x.gamma() / &x,
        x.log_gamma() / (&x * x.ln()),
        x.pow(&x) / x.factorial(),
        x.gamma() / (&x + ctx.rational(1, 2)).gamma() * x.sqrt(),
    ] {
        let r = e.limit(&x, &inf);
        if !r.has_unevaluated() {
            // If the engine ever learns these, the answer must be 1 / 1 / ∞ / 1.
            let s = format!("{r}");
            assert!(s == "1" || s == "oo", "{e}: unexpected {s}");
        }
        assert_no_internal_symbols(&r, &format!("{e}"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// API semantics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn direction_default_is_both() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin() / &x;
    let both = f.limit_dir(&x, &ctx.int(0), Default::default());
    assert_eq!(format!("{both}"), "1");
    assert_eq!(format!("{}", f.limit(&x, &ctx.int(0))), "1");
    assert!(
        (1 / &x)
            .try_limit_dir(&x, &ctx.int(0), Default::default())
            .is_err()
    );
}

#[test]
fn try_limit_semantics() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    // ±∞ are legitimate limit values → Ok.
    assert_eq!(
        format!("{}", x.exp().try_limit(&x, &ctx.infinity()).unwrap()),
        "oo"
    );
    assert_eq!(
        format!("{}", (-&x).try_limit(&x, &ctx.infinity()).unwrap()),
        "-oo"
    );
    assert_eq!(
        format!("{}", (1 / x.powi(2)).try_limit(&x, &zero).unwrap()),
        "oo"
    );

    // Differing one-sided limits → ComputationFailed with an explanatory reason.
    let err = (1 / &x).try_limit(&x, &zero).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("differ"), "{msg}");
    assert!(msg.contains("oo"), "{msg}");
    assert!(
        matches!(
            err,
            SymplexError::ComputationFailed {
                operation: "limit",
                ..
            }
        ),
        "{err:?}"
    );

    // One-sided try variants.
    assert_eq!(
        format!("{}", (1 / &x).try_limit_right(&x, &zero).unwrap()),
        "oo"
    );
    assert_eq!(
        format!("{}", (1 / &x).try_limit_left(&x, &zero).unwrap()),
        "-oo"
    );

    // No limit at all (oscillation) → Err.
    assert!((1 / &x).sin().try_limit_right(&x, &zero).is_err());
    assert!(x.sin().try_limit(&x, &ctx.infinity()).is_err());

    // Non-symbol variable → InvalidArgument.
    let err = x.try_limit(&ctx.int(2), &zero).unwrap_err();
    assert!(
        matches!(err, SymplexError::InvalidArgument { .. }),
        "{err:?}"
    );
}

#[test]
fn unevaluated_one_sided_limit_is_the_two_sided_limit_node() {
    // `Limit(body, var, point)` has no direction slot; a one-sided limit
    // that cannot be computed falls back to the plain formal node.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (1 / &x).sin().limit_right(&x, &ctx.int(0));
    assert_eq!(format!("{r}"), "Limit(sin(1/x), x, 0)");
    assert_eq!(r, (1 / &x).sin().limit(&x, &ctx.int(0)));
}

#[test]
fn limit_results_never_leak_internal_symbols() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exprs: Vec<Ex> = vec![
        x.erf(),
        x.atan(),
        x.tanh(),
        x.gamma(),
        x.sin(),
        &x * x.sin(),
        x.lambertw() / x.ln(),
        (1 / &x).sin(),
        x.bessel_j(&ctx.int(0)),
        x.floor() / &x,
        x.digamma() - x.ln(),
        x.erfc() * x.powi(2).exp() * &x,
    ];
    for e in &exprs {
        for p in [ctx.infinity(), ctx.neg_infinity(), ctx.int(0)] {
            for (name, r) in [
                ("two-sided", e.limit(&x, &p)),
                ("left", e.limit_left(&x, &p)),
                ("right", e.limit_right(&x, &p)),
            ] {
                assert_no_internal_symbols(&r, &format!("{e} at {p} ({name})"));
                let s = format!("{r}");
                assert!(
                    !s.contains("zoo") && !s.contains("nan"),
                    "indeterminate leaked: {s}"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Work budget
// ═══════════════════════════════════════════════════════════════════════════

/// Pathological inputs must come back (unevaluated or not) quickly instead
/// of spinning: the Gruntz engine carries an internal work budget.
#[test]
fn pathological_inputs_return_within_budget() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inf = ctx.infinity();
    let zero = ctx.int(0);

    // Deeply nested trig cancellation whose series explode combinatorially.
    let deep4 = (x.sin().sin().sin().sin() - x.tan().tan().tan().tan()) / x.powi(5);
    // Six-level exponential towers in different comparability classes.
    let mut tower = x.clone();
    let mut tower2 = &x - 1 / &x;
    for _ in 0..6 {
        tower = tower.exp();
        tower2 = tower2.exp();
    }
    // Gruntz's thesis example 8.1-style monster.
    let monster = (x.exp() * (x.exp() * (-x.powi(2).exp().exp().exp())).exp()).exp();
    // Large sum of interacting exponential classes.
    let mut big = ctx.int(0);
    for k in 1..=8 {
        big = &big + (&x * k).exp() / ((&x * k).exp() + 1) * (&x * (k + 1)).exp();
    }
    let prod = (1..=6).fold(ctx.int(1), |acc, k| {
        acc * (1 / x.powi(k)).sin().pow(&ctx.rational(1, k))
    });

    let cases: Vec<(Ex, Ex)> = vec![
        (deep4, zero.clone()),
        (&tower / &tower2, inf.clone()),
        (&tower - &tower2, inf.clone()),
        (monster, inf.clone()),
        (big, inf.clone()),
        (prod, zero.clone()),
        (x.pow(&x.pow(&x.pow(&x))) / x.exp().exp().exp(), inf.clone()),
    ];
    for (e, p) in &cases {
        let t0 = std::time::Instant::now();
        let r = e.limit_right(&x, p);
        let elapsed = t0.elapsed();
        assert!(elapsed < time_budget(2), "{e}: took {elapsed:?}");
        assert_no_internal_symbols(&r, &format!("{e}"));
        let s = format!("{r}");
        assert!(!s.contains("zoo") && !s.contains("nan"), "{s}");
    }
}
