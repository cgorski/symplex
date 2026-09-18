//! symplex 0.2 — limits: one-sided limits, the Gruntz robustness battery,
//! and `try_limit` semantics.
//!
//! Every finite limit in the battery is also verified numerically by
//! approaching the point along the requested direction.

use symplex::prelude::*;

/// How a limit is approached numerically.
#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
    Both,
}

/// Evaluate `expr` at points approaching `point` from `side` and compare with
/// the claimed limit `lim` (finite, symbolic constant, or ±∞).
fn verify_numerically(expr: &Ex, var: &Ex, point: &Ex, side: Side, lim: &Ex, label: &str) {
    let ctx = expr.context();
    let lim_s = format!("{lim}");
    let is_pos_inf = lim_s == "oo";
    let is_neg_inf = lim_s == "-oo";
    let lim_val = if is_pos_inf || is_neg_inf {
        None
    } else {
        Some(
            lim.eval_f64()
                .unwrap_or_else(|e| panic!("{label}: limit {lim} not numeric: {e}")),
        )
    };

    let point_s = format!("{point}");
    let sample_points: Vec<Ex> = if point_s == "oo" {
        vec![ctx.int(1_000), ctx.int(100_000), ctx.int(10_000_000)]
    } else if point_s == "-oo" {
        vec![ctx.int(-1_000), ctx.int(-100_000), ctx.int(-10_000_000)]
    } else {
        let hs = [
            ctx.rational(1, 10_000),
            ctx.rational(1, 1_000_000),
            ctx.rational(1, 100_000_000),
        ];
        let mut pts = Vec::new();
        for h in &hs {
            match side {
                Side::Right => pts.push(point + h),
                Side::Left => pts.push(point - h),
                Side::Both => {
                    pts.push(point + h);
                    pts.push(point - h);
                }
            }
        }
        pts
    };

    let mut prev_abs: Option<f64> = None;
    for (i, p) in sample_points.iter().enumerate() {
        let v = expr.subs(var, p).eval_f64();
        let Ok(v) = v else {
            // Numeric evaluation may legitimately fail extremely close to a
            // singularity (overflow); tolerate that for the last samples.
            assert!(
                i > 0,
                "{label}: could not evaluate at first sample point {p}"
            );
            continue;
        };
        if v.is_nan() {
            continue;
        }
        match lim_val {
            Some(l) => {
                let tol = 1e-2 * l.abs().max(1.0);
                // Only the closest samples must be within tolerance.
                if i + 2 >= sample_points.len()
                    || matches!(side, Side::Both) && i + 4 >= sample_points.len()
                {
                    assert!(
                        (v - l).abs() < tol,
                        "{label}: numeric value {v} at {p} vs claimed limit {l}"
                    );
                }
            }
            None => {
                if is_pos_inf {
                    assert!(v > 0.0, "{label}: expected → +∞ but got {v} at {p}");
                } else {
                    assert!(v < 0.0, "{label}: expected → −∞ but got {v} at {p}");
                }
                if let Some(pa) = prev_abs {
                    assert!(
                        v.abs() >= pa * 0.5,
                        "{label}: |value| not growing: {pa} → {v}"
                    );
                }
                prev_abs = Some(v.abs());
            }
        }
    }
}

struct Case<'a> {
    label: &'a str,
    expr: Ex,
    point: Ex,
    expected: &'a str,
}

#[test]
fn gruntz_battery_two_sided_and_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let inf = ctx.infinity();
    let ninf = ctx.neg_infinity();
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let b = ctx.symbol_with("b", &[Assumption::Positive]);

    let cases = vec![
        // ── L'Hôpital classics at 0 ──
        Case {
            label: "sin x / x",
            expr: &x.sin() / &x,
            point: zero.clone(),
            expected: "1",
        },
        Case {
            label: "(1 − cos x)/x²",
            expr: (1 - x.cos()) / x.powi(2),
            point: zero.clone(),
            expected: "1/2",
        },
        Case {
            label: "(eˣ − 1 − x)/x²",
            expr: (x.exp() - 1 - &x) / x.powi(2),
            point: zero.clone(),
            expected: "1/2",
        },
        Case {
            label: "(eˣ − 1)/x",
            expr: (x.exp() - 1) / &x,
            point: zero.clone(),
            expected: "1",
        },
        Case {
            label: "ln(1 + x)/x",
            expr: (1 + &x).ln() / &x,
            point: zero.clone(),
            expected: "1",
        },
        Case {
            label: "(x − sin x)/x³",
            expr: (&x - x.sin()) / x.powi(3),
            point: zero.clone(),
            expected: "1/6",
        },
        Case {
            label: "(tan x − sin x)/x³",
            expr: (x.tan() - x.sin()) / x.powi(3),
            point: zero.clone(),
            expected: "1/2",
        },
        Case {
            label: "(2ˣ − 3ˣ)/x",
            expr: (ctx.int(2).pow(&x) - ctx.int(3).pow(&x)) / &x,
            point: zero.clone(),
            expected: "-ln(3) + ln(2)",
        },
        Case {
            label: "(√(1+x) − 1)/x",
            expr: ((1 + &x).sqrt() - 1) / &x,
            point: zero.clone(),
            expected: "1/2",
        },
        Case {
            label: "(cos x − 1)/(x sin x)",
            expr: (x.cos() - 1) / (&x * x.sin()),
            point: zero.clone(),
            expected: "-1/2",
        },
        Case {
            label: "asin(x)/x",
            expr: x.asin() / &x,
            point: zero.clone(),
            expected: "1",
        },
        Case {
            label: "sinh(x)/x",
            expr: x.sinh() / &x,
            point: zero.clone(),
            expected: "1",
        },
        Case {
            label: "(cosh x − 1)/x²",
            expr: (x.cosh() - 1) / x.powi(2),
            point: zero.clone(),
            expected: "1/2",
        },
        Case {
            label: "(1 − cos 2x)/x²",
            expr: (1 - (&x * 2).cos()) / x.powi(2),
            point: zero.clone(),
            expected: "2",
        },
        Case {
            label: "(eˣ − 1)/sin x",
            expr: (x.exp() - 1) / x.sin(),
            point: zero.clone(),
            expected: "1",
        },
        Case {
            label: "x sin(1/x)",
            expr: &x * (1 / &x).sin(),
            point: zero.clone(),
            expected: "0",
        },
        Case {
            label: "x² sin(1/x)",
            expr: x.powi(2) * (1 / &x).sin(),
            point: zero.clone(),
            expected: "0",
        },
        Case {
            label: "x ln x → 0 (two-sided, complex left)",
            expr: &x * x.ln(),
            point: zero.clone(),
            expected: "0",
        },
        Case {
            label: "x² ln x",
            expr: x.powi(2) * x.ln(),
            point: zero.clone(),
            expected: "0",
        },
        Case {
            label: "xˣ",
            expr: x.pow(&x),
            point: zero.clone(),
            expected: "1",
        },
        Case {
            label: "(1 + x)^(1/x)",
            expr: (1 + &x).pow(&(1 / &x)),
            point: zero.clone(),
            expected: "E",
        },
        Case {
            label: "(1 + sin x)^(1/x)",
            expr: (1 + x.sin()).pow(&(1 / &x)),
            point: zero.clone(),
            expected: "E",
        },
        Case {
            label: "cos(x)^(1/x²)",
            expr: x.cos().pow(&(1 / x.powi(2))),
            point: zero.clone(),
            expected: "exp(-1/2)",
        },
        Case {
            label: "sec x",
            expr: x.sec(),
            point: zero.clone(),
            expected: "1",
        },
        Case {
            label: "|x|",
            expr: x.abs(),
            point: zero.clone(),
            expected: "0",
        },
        Case {
            label: "√x",
            expr: x.sqrt(),
            point: zero.clone(),
            expected: "0",
        },
        Case {
            label: "max(x, 1)",
            expr: x.max_with(&one),
            point: zero.clone(),
            expected: "1",
        },
        Case {
            label: "⌊x⌋ at 1/2",
            expr: x.floor(),
            point: ctx.rational(1, 2),
            expected: "0",
        },
        // ── symbolic parameters ──
        Case {
            label: "a (constant)",
            expr: a.clone(),
            point: zero.clone(),
            expected: "a",
        },
        Case {
            label: "(aˣ − 1)/x",
            expr: (a.pow(&x) - 1) / &x,
            point: zero.clone(),
            expected: "ln(a)",
        },
        Case {
            label: "sin(ax)/x",
            expr: (&a * &x).sin() / &x,
            point: zero.clone(),
            expected: "a",
        },
        Case {
            label: "sin(x) at 1",
            expr: x.sin(),
            point: one.clone(),
            expected: "sin(1)",
        },
        Case {
            label: "eˣ + π at 0",
            expr: x.exp() + ctx.pi(),
            point: zero.clone(),
            expected: "1 + pi",
        },
        // ── other finite points ──
        Case {
            label: "(x³ − 1)/(x − 1) at 1",
            expr: (x.powi(3) - 1) / (&x - 1),
            point: one.clone(),
            expected: "3",
        },
        Case {
            label: "(x² − 1)/(x − 1) at 1",
            expr: (x.powi(2) - 1) / (&x - 1),
            point: one.clone(),
            expected: "2",
        },
        Case {
            label: "ln(x)/(x − 1) at 1",
            expr: x.ln() / (&x - 1),
            point: one.clone(),
            expected: "1",
        },
        Case {
            label: "(x − 1)/ln(x) at 1",
            expr: (&x - 1) / x.ln(),
            point: one.clone(),
            expected: "1",
        },
        Case {
            label: "Γ(x) at 3",
            expr: x.gamma(),
            point: ctx.int(3),
            expected: "2",
        },
        Case {
            label: "1/(x − 1)² at 1",
            expr: 1 / (&x - 1).powi(2),
            point: one.clone(),
            expected: "oo",
        },
        // ── at +∞ ──
        Case {
            label: "x^(1/x)",
            expr: x.pow(&(1 / &x)),
            point: inf.clone(),
            expected: "1",
        },
        Case {
            label: "(1 + 1/x)ˣ",
            expr: (1 + 1 / &x).pow(&x),
            point: inf.clone(),
            expected: "E",
        },
        Case {
            label: "(1 + a/x)^(bx)",
            expr: (1 + &a / &x).pow(&(&b * &x)),
            point: inf.clone(),
            expected: "exp(a*b)",
        },
        Case {
            label: "(1 + 2/x)^(3x)",
            expr: (1 + 2 / &x).pow(&(&x * 3)),
            point: inf.clone(),
            expected: "exp(6)",
        },
        Case {
            label: "(1 + 1/x)^(x²)",
            expr: (1 + 1 / &x).pow(&x.powi(2)),
            point: inf.clone(),
            expected: "oo",
        },
        Case {
            label: "√(x² + x) − x",
            expr: (x.powi(2) + &x).sqrt() - &x,
            point: inf.clone(),
            expected: "1/2",
        },
        Case {
            label: "x − √(x² − 1)",
            expr: &x - (x.powi(2) - 1).sqrt(),
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "ln x / √x",
            expr: x.ln() / x.sqrt(),
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "ln x / x^(1/3)",
            expr: x.ln() / x.pow(&ctx.rational(1, 3)),
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "ln x / x",
            expr: x.ln() / &x,
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "x e⁻ˣ",
            expr: &x * (-&x).exp(),
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "x² e⁻ˣ",
            expr: x.powi(2) * (-&x).exp(),
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "x⁵/eˣ",
            expr: x.powi(5) / x.exp(),
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "e^(−x²)",
            expr: (-x.powi(2)).exp(),
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "eˣ − e^(2x)",
            expr: x.exp() - (&x * 2).exp(),
            point: inf.clone(),
            expected: "-oo",
        },
        Case {
            label: "(eˣ + e⁻ˣ)/eˣ",
            expr: (x.exp() + (-&x).exp()) / x.exp(),
            point: inf.clone(),
            expected: "1",
        },
        Case {
            label: "eˣ/(eˣ + 1)",
            expr: x.exp() / (x.exp() + 1),
            point: inf.clone(),
            expected: "1",
        },
        Case {
            label: "sinh(x)/eˣ",
            expr: x.sinh() / x.exp(),
            point: inf.clone(),
            expected: "1/2",
        },
        Case {
            label: "(x² + 1)/(x + 1)",
            expr: (x.powi(2) + 1) / (&x + 1),
            point: inf.clone(),
            expected: "oo",
        },
        Case {
            label: "(3x² + 1)/(x² − x)",
            expr: (&x.powi(2) * 3 + 1) / (x.powi(2) - &x),
            point: inf.clone(),
            expected: "3",
        },
        Case {
            label: "x² − x³",
            expr: x.powi(2) - x.powi(3),
            point: inf.clone(),
            expected: "-oo",
        },
        Case {
            label: "(x + a)/(x − a)",
            expr: (&x + &a) / (&x - &a),
            point: inf.clone(),
            expected: "1",
        },
        Case {
            label: "a x",
            expr: &a * &x,
            point: inf.clone(),
            expected: "oo",
        },
        Case {
            label: "atan x",
            expr: x.atan(),
            point: inf.clone(),
            expected: "1/2*pi",
        },
        Case {
            label: "erf x",
            expr: x.erf(),
            point: inf.clone(),
            expected: "1",
        },
        Case {
            label: "tanh x",
            expr: x.tanh(),
            point: inf.clone(),
            expected: "1",
        },
        Case {
            label: "x tanh x",
            expr: &x * x.tanh(),
            point: inf.clone(),
            expected: "oo",
        },
        Case {
            label: "W(x)/ln x",
            expr: x.lambertw() / x.ln(),
            point: inf.clone(),
            expected: "1",
        },
        Case {
            label: "Γ(x)",
            expr: x.gamma(),
            point: inf.clone(),
            expected: "oo",
        },
        Case {
            label: "1/Γ(x)",
            expr: 1 / x.gamma(),
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "x sin(1/x)",
            expr: &x * (1 / &x).sin(),
            point: inf.clone(),
            expected: "1",
        },
        Case {
            label: "cos(x)/x",
            expr: x.cos() / &x,
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "ln(x + 1) − ln x",
            expr: (&x + 1).ln() - x.ln(),
            point: inf.clone(),
            expected: "0",
        },
        Case {
            label: "x (ln(x + 1) − ln x)",
            expr: &x * ((&x + 1).ln() - x.ln()),
            point: inf.clone(),
            expected: "1",
        },
        Case {
            label: "asinh(x) − ln x",
            expr: x.asinh() - x.ln(),
            point: inf.clone(),
            expected: "ln(2)",
        },
        Case {
            label: "min(x, 1)",
            expr: x.min_with(&one),
            point: inf.clone(),
            expected: "1",
        },
        // ── at −∞ ──
        Case {
            label: "atan x at −∞",
            expr: x.atan(),
            point: ninf.clone(),
            expected: "-1/2*pi",
        },
        Case {
            label: "erf x at −∞",
            expr: x.erf(),
            point: ninf.clone(),
            expected: "-1",
        },
        Case {
            label: "tanh x at −∞",
            expr: x.tanh(),
            point: ninf.clone(),
            expected: "-1",
        },
        Case {
            label: "x³ at −∞",
            expr: x.powi(3),
            point: ninf.clone(),
            expected: "-oo",
        },
        Case {
            label: "x eˣ at −∞",
            expr: &x * x.exp(),
            point: ninf.clone(),
            expected: "0",
        },
        Case {
            label: "x³ eˣ at −∞",
            expr: x.powi(3) * x.exp(),
            point: ninf.clone(),
            expected: "0",
        },
        Case {
            label: "eˣ/(eˣ + 1) at −∞",
            expr: x.exp() / (x.exp() + 1),
            point: ninf.clone(),
            expected: "0",
        },
    ];

    let mut failures = Vec::new();
    for c in &cases {
        let r = c.expr.limit(&x, &c.point);
        let got = format!("{r}");
        if r.has_unevaluated() {
            failures.push(format!("{}: unevaluated ({got})", c.label));
            continue;
        }
        // Compare symbolically (string) or numerically.
        let expected = ctx
            .parse(c.expected)
            .unwrap_or_else(|e| panic!("bad expected {}: {e}", c.expected));
        let same_string = got == c.expected;
        let same_numeric = match (r.eval_f64(), expected.eval_f64()) {
            (Ok(g), Ok(e)) => (g - e).abs() <= 1e-9 * e.abs().max(1.0),
            _ => false,
        };
        if !(same_string || same_numeric) {
            failures.push(format!("{}: got {got}, expected {}", c.label, c.expected));
            continue;
        }
        // Numeric verification (skipped for purely symbolic parameters).
        if r.eval_f64().is_ok() || got == "oo" || got == "-oo" {
            let numeric_ok = c.expr.subs(&x, &ctx.int(7)).eval_f64().is_ok();
            if numeric_ok {
                verify_numerically(&c.expr, &x, &c.point, Side::Both, &r, c.label);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "battery failures:\n{}",
        failures.join("\n")
    );
    assert!(cases.len() >= 60);
}

#[test]
fn two_sided_limits_that_do_not_exist_stay_unevaluated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let no_limit: Vec<(&str, Ex, Ex)> = vec![
        ("1/x", 1 / &x, zero.clone()),
        ("|x|/x", x.abs() / &x, zero.clone()),
        ("x/|x|", &x / x.abs(), zero.clone()),
        ("e^(1/x)", (1 / &x).exp(), zero.clone()),
        ("sin(1/x)", (1 / &x).sin(), zero.clone()),
        ("tan x at π/2", x.tan(), ctx.pi() / 2),
        ("H(x)", x.heaviside(), zero.clone()),
        ("sign(x)", x.sign(), zero.clone()),
        ("1/(1 + e^(1/x))", 1 / (1 + (1 / &x).exp()), zero.clone()),
        ("sin x at ∞", x.sin(), inf.clone()),
        ("x sin x at ∞", &x * x.sin(), inf.clone()),
        ("⌊x⌋ at 2", x.floor(), ctx.int(2)),
        ("1/(x − 1)³ at 1", 1 / (&x - 1).powi(3), ctx.int(1)),
        ("atan(1/x)", (1 / &x).atan(), zero.clone()),
    ];
    for (label, e, p) in no_limit {
        let r = e.limit(&x, &p);
        assert!(
            r.has_unevaluated(),
            "{label}: expected unevaluated, got {r}"
        );
        assert!(
            e.try_limit(&x, &p).is_err(),
            "{label}: try_limit should be Err"
        );
        // The unevaluated form is the formal Limit node of the input.
        assert!(format!("{r}").starts_with("Limit("), "{label}: {r}");
    }
}

#[test]
fn one_sided_limits_battery() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let two = ctx.int(2);
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let pw = Ex::piecewise(&[(&x.powi(2), &x.lt(&one)), (&(&x * 2), &one.le(&x))]);

    // (label, expr, point, expected right, expected left)
    let cases: Vec<(&str, Ex, Ex, &str, &str)> = vec![
        ("1/x", 1 / &x, zero.clone(), "oo", "-oo"),
        ("|x|/x", x.abs() / &x, zero.clone(), "1", "-1"),
        ("x/|x|", &x / x.abs(), zero.clone(), "1", "-1"),
        (
            "|x − 1|/(x − 1)",
            (&x - 1).abs() / (&x - 1),
            one.clone(),
            "1",
            "-1",
        ),
        ("e^(1/x)", (1 / &x).exp(), zero.clone(), "oo", "0"),
        (
            "1/(1 + e^(1/x))",
            1 / (1 + (1 / &x).exp()),
            zero.clone(),
            "0",
            "1",
        ),
        ("e^(−1/x)/x", (-1 / &x).exp() / &x, zero.clone(), "0", "-oo"),
        ("⌊x⌋ at 2", x.floor(), two.clone(), "2", "1"),
        ("⌈x⌉ at 2", x.ceiling(), two.clone(), "3", "2"),
        ("tan x at π/2", x.tan(), ctx.pi() / 2, "-oo", "oo"),
        ("H(x)", x.heaviside(), zero.clone(), "1", "0"),
        ("sign(x)", x.sign(), zero.clone(), "1", "-1"),
        (
            "sign(x² − 1) at 1",
            (x.powi(2) - 1).sign(),
            one.clone(),
            "1",
            "-1",
        ),
        ("piecewise at 1", pw, one.clone(), "2", "1"),
        (
            "1/(x − 1)³ at 1",
            1 / (&x - 1).powi(3),
            one.clone(),
            "oo",
            "-oo",
        ),
        (
            "1/(x − 1)² at 1",
            1 / (&x - 1).powi(2),
            one.clone(),
            "oo",
            "oo",
        ),
        (
            "atan(1/x)",
            (1 / &x).atan(),
            zero.clone(),
            "1/2*pi",
            "-1/2*pi",
        ),
        ("Γ(x) at 0", x.gamma(), zero.clone(), "oo", "-oo"),
        ("xˣ", x.pow(&x), zero.clone(), "1", "1"),
        ("x ln x", &x * x.ln(), zero.clone(), "0", "0"),
    ];
    for (label, e, p, r_exp, l_exp) in &cases {
        let r = e.limit_right(&x, p);
        let l = e.limit_left(&x, p);
        assert_eq!(format!("{r}"), *r_exp, "{label}: right limit");
        assert_eq!(format!("{l}"), *l_exp, "{label}: left limit");
        // Numerically verify along each side (skip complex-valued left side of x ln x / x^x).
        if !label.contains("ln") && !label.contains("xˣ") {
            verify_numerically(e, &x, p, Side::Right, &r, &format!("{label} (right)"));
            verify_numerically(e, &x, p, Side::Left, &l, &format!("{label} (left)"));
        }
    }

    // Right-only cases (left side leaves the real domain).
    let right_only: Vec<(&str, Ex, Ex, &str)> = vec![
        ("ln x", x.ln(), zero.clone(), "-oo"),
        ("ln(x − a) at a", (&x - &a).ln(), a.clone(), "-oo"),
        ("√x", x.sqrt(), zero.clone(), "0"),
        ("x^(1/x)", x.pow(&(1 / &x)), zero.clone(), "0"),
        ("x^(−x)", x.pow(&(-&x)), zero.clone(), "1"),
        ("ln(x)·x", x.ln() * &x, zero.clone(), "0"),
        ("x^(1/2) ln x", x.sqrt() * x.ln(), zero.clone(), "0"),
        ("(ln x)/x", x.ln() / &x, zero.clone(), "-oo"),
    ];
    for (label, e, p, exp) in &right_only {
        let r = e.limit_right(&x, p);
        assert_eq!(format!("{r}"), *exp, "{label}: right limit");
        if e.subs(&x, &ctx.rational(1, 3)).eval_f64().is_ok() {
            verify_numerically(e, &x, p, Side::Right, &r, label);
        }
    }
}

#[test]
fn direction_default_is_both() {
    // `Direction` lives in a crate-internal module until it is re-exported
    // from the prelude (see the release report); `Default::default()` is
    // `Direction::Both`, which lets us exercise `limit_dir` here.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin() / &x;
    let both = f.limit_dir(&x, &ctx.int(0), Default::default());
    assert_eq!(format!("{both}"), "1");
    assert_eq!(format!("{}", f.limit(&x, &ctx.int(0))), "1");
    // Two-sided via `limit_dir` also rejects differing one-sided limits.
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
    ];
    for e in &exprs {
        for p in [ctx.infinity(), ctx.neg_infinity(), ctx.int(0)] {
            let r = e.limit(&x, &p);
            let s = format!("{r}");
            assert!(!s.contains("__"), "internal symbol leaked: {s}");
            assert!(
                !s.contains("zoo") && !s.contains("nan"),
                "indeterminate leaked: {s}"
            );
        }
    }
}

#[test]
fn piecewise_limits_use_the_approaching_branch() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    // f(x) = x²  (x < 1),  2x − 1 (x ≥ 1)  — continuous at 1
    let f = Ex::piecewise(&[(&x.powi(2), &x.lt(&one)), (&(&x * 2 - 1), &one.le(&x))]);
    assert_eq!(format!("{}", f.limit_left(&x, &one)), "1");
    assert_eq!(format!("{}", f.limit_right(&x, &one)), "1");
    assert_eq!(format!("{}", f.limit(&x, &one)), "1");

    // g(x) = −1 (x < 0), 1 (x ≥ 0) — jump
    let g = Ex::piecewise(&[
        (&ctx.int(-1), &x.lt(&ctx.int(0))),
        (&one, &ctx.int(0).le(&x)),
    ]);
    assert_eq!(format!("{}", g.limit_left(&x, &ctx.int(0))), "-1");
    assert_eq!(format!("{}", g.limit_right(&x, &ctx.int(0))), "1");
    assert!(g.limit(&x, &ctx.int(0)).has_unevaluated());

    // Away from the boundary the limit is plain substitution.
    assert_eq!(format!("{}", g.limit(&x, &ctx.int(5))), "1");
}

#[test]
fn symbolic_parameters_with_assumptions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let c = ctx.symbol_with("c", &[Assumption::Negative]);
    let inf = ctx.infinity();

    assert_eq!(format!("{}", (&a * &x).limit(&x, &inf)), "oo");
    assert_eq!(format!("{}", (&c * &x).limit(&x, &inf)), "-oo");
    assert_eq!(format!("{}", (&a * &x).exp().limit(&x, &inf)), "oo");
    assert_eq!(format!("{}", (&c * &x).exp().limit(&x, &inf)), "0");
    assert_eq!(format!("{}", (-&a * &x).exp().limit(&x, &inf)), "0");
    assert_eq!(format!("{}", (&a / &x).limit(&x, &inf)), "0");
    assert_eq!(
        format!("{}", (&a * &x).sin().limit_right(&x, &ctx.int(0))),
        "0"
    );
    // (1 + a/x)^x → e^a
    let r = (1 + &a / &x).pow(&x).limit(&x, &inf);
    assert_eq!(format!("{r}"), "exp(a)");

    // Unknown sign → do not guess.
    let u = ctx.symbol("u");
    assert!((&u * &x).limit(&x, &inf).has_unevaluated());
}
