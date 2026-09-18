//! v0.2 ergonomics — numeric ingestion on `Context`.
//!
//! Covers `from_f64` (exact dyadic), `from_f64_approx` / `from_f64_nice`,
//! `from_bigint` / `from_ratio` / `from_i128` / `from_u64`,
//! `rational_str` / `decimal_str`, `complex`, `symbols` /
//! `symbols_indexed`, `apply`, and `sum` / `product`.

use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::numeric::{f64_to_ratio_exact, ratio_to_f64};
use symplex::prelude::*;

fn r(p: i64, q: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(q))
}

// ── from_f64 (exact) ───────────────────────────────────────────────────

#[test]
fn from_f64_half_is_one_half() {
    let ctx = Context::new();
    assert_eq!(ctx.from_f64(0.5).unwrap(), ctx.rational(1, 2));
}

#[test]
fn from_f64_tenth_is_the_dyadic_not_one_tenth() {
    let ctx = Context::new();
    let e = ctx.from_f64(0.1).unwrap();
    assert_ne!(e, ctx.rational(1, 10));
    assert_eq!(
        e.as_rational().unwrap(),
        r(3602879701896397, 36028797018963968)
    );
}

#[test]
fn from_f64_integers_and_negative_zero() {
    let ctx = Context::new();
    assert_eq!(ctx.from_f64(42.0).unwrap(), ctx.int(42));
    assert_eq!(ctx.from_f64(-7.0).unwrap(), ctx.int(-7));
    assert_eq!(ctx.from_f64(-0.0).unwrap(), ctx.zero());
}

#[test]
fn from_f64_infinities_map_to_symbolic_nodes() {
    let ctx = Context::new();
    assert_eq!(ctx.from_f64(f64::INFINITY).unwrap(), ctx.infinity());
    assert_eq!(ctx.from_f64(f64::NEG_INFINITY).unwrap(), ctx.neg_infinity());
}

#[test]
fn from_f64_nan_is_invalid_argument() {
    let ctx = Context::new();
    assert!(matches!(
        ctx.from_f64(f64::NAN),
        Err(SymplexError::InvalidArgument {
            operation: "from_f64",
            ..
        })
    ));
}

#[test]
fn from_f64_round_trips_bit_for_bit() {
    let ctx = Context::new();
    for v in [
        0.1,
        0.2,
        0.3,
        1.0 / 3.0,
        std::f64::consts::PI,
        1e-300,
        1e300,
        f64::MAX,
        f64::MIN_POSITIVE,
        5e-324,
    ] {
        let back = ratio_to_f64(&ctx.from_f64(v).unwrap().as_rational().unwrap()).unwrap();
        assert_eq!(back.to_bits(), v.to_bits(), "v = {v:e}");
    }
}

#[test]
fn from_f64_agrees_with_numeric_module() {
    let ctx = Context::new();
    let v = 123.456_789;
    assert_eq!(
        ctx.from_f64(v).unwrap().as_rational(),
        f64_to_ratio_exact(v)
    );
}

#[test]
fn from_f64_arithmetic_is_exact() {
    // 0.1 + 0.2 != 0.3 in f64; the exact rationals reflect that.
    let ctx = Context::new();
    let sum = &ctx.from_f64(0.1).unwrap() + &ctx.from_f64(0.2).unwrap();
    let three_tenths = ctx.from_f64(0.3).unwrap();
    assert_ne!(sum, three_tenths);
    assert_eq!(
        sum.compare_numeric(&three_tenths),
        Some(std::cmp::Ordering::Greater)
    );
}

// ── from_f64_approx / from_f64_nice ────────────────────────────────────

#[test]
fn from_f64_approx_recovers_human_decimals() {
    let ctx = Context::new();
    for (v, p, q) in [
        (0.1, 1, 10),
        (0.2, 1, 5),
        (0.3, 3, 10),
        (0.75, 3, 4),
        (2.0 / 3.0, 2, 3),
    ] {
        assert_eq!(
            ctx.from_f64_approx(v, 1_000_000).unwrap(),
            ctx.rational(p, q),
            "v = {v}"
        );
    }
}

#[test]
fn from_f64_approx_respects_bound() {
    let ctx = Context::new();
    assert_eq!(
        ctx.from_f64_approx(std::f64::consts::PI, 1000).unwrap(),
        ctx.rational(355, 113)
    );
    assert_eq!(
        ctx.from_f64_approx(std::f64::consts::PI, 10).unwrap(),
        ctx.rational(22, 7)
    );
    assert_eq!(
        ctx.from_f64_approx(std::f64::consts::PI, 1).unwrap(),
        ctx.int(3)
    );
}

#[test]
fn from_f64_approx_rejects_bad_input() {
    let ctx = Context::new();
    assert!(ctx.from_f64_approx(0.5, 0).is_err());
    assert!(ctx.from_f64_approx(f64::NAN, 10).is_err());
    assert_eq!(
        ctx.from_f64_approx(f64::INFINITY, 10).unwrap(),
        ctx.infinity()
    );
}

#[test]
fn from_f64_nice_examples() {
    let ctx = Context::new();
    assert_eq!(ctx.from_f64_nice(0.1).unwrap(), ctx.rational(1, 10));
    assert_eq!(ctx.from_f64_nice(0.1 + 0.2).unwrap(), ctx.rational(3, 10));
    assert_eq!(ctx.from_f64_nice(1.0 / 7.0).unwrap(), ctx.rational(1, 7));
    assert_eq!(ctx.from_f64_nice(-12.5).unwrap(), ctx.rational(-25, 2));
    assert_eq!(ctx.from_f64_nice(0.0).unwrap(), ctx.zero());
    assert_eq!(
        ctx.from_f64_nice(0.123456789).unwrap(),
        ctx.rational(123456789, 1_000_000_000)
    );
}

// ── BigInt / Ratio / i128 / u64 ────────────────────────────────────────

#[test]
fn from_bigint_large_value() {
    let ctx = Context::new();
    let n = BigInt::from(2).pow(200);
    let e = ctx.from_bigint(n.clone());
    assert_eq!(e.as_bigint(), Some(n));
    assert_eq!(format!("{}", e), format!("{}", BigInt::from(2).pow(200)));
}

#[test]
fn from_ratio_normalises() {
    let ctx = Context::new();
    assert_eq!(ctx.from_ratio(r(6, -4)), ctx.rational(-3, 2));
    assert_eq!(ctx.from_ratio(r(0, 5)), ctx.zero());
    assert_eq!(ctx.from_ratio(r(8, 4)), ctx.int(2));
}

#[test]
fn from_ratio_raw_zero_denominator_is_not_a_panic() {
    let ctx = Context::new();
    assert_eq!(
        ctx.from_ratio(Ratio::new_raw(BigInt::from(3), BigInt::from(0))),
        ctx.complex_infinity()
    );
    assert_eq!(
        ctx.from_ratio(Ratio::new_raw(BigInt::from(0), BigInt::from(0))),
        ctx.nan()
    );
}

#[test]
fn from_i128_and_u64_extremes() {
    let ctx = Context::new();
    assert_eq!(ctx.from_i128(0), ctx.zero());
    assert_eq!(ctx.from_i128(-1), ctx.int(-1));
    assert_eq!(
        ctx.from_i128(i128::MAX).as_bigint(),
        Some(BigInt::from(i128::MAX))
    );
    assert_eq!(
        ctx.from_u64(u64::MAX).as_bigint(),
        Some(BigInt::from(u64::MAX))
    );
    assert_eq!(ctx.from_u64(7), ctx.int(7));
}

#[test]
fn ingestion_is_hash_consed() {
    let ctx = Context::new();
    let a = ctx.from_f64(0.25).unwrap();
    let b = ctx.rational(1, 4);
    let c = ctx.from_ratio(r(2, 8));
    let d = ctx.decimal_str("0.25").unwrap();
    assert!(a == b && b == c && c == d);
}

// ── rational_str / decimal_str ─────────────────────────────────────────

#[test]
fn rational_str_forms() {
    let ctx = Context::new();
    assert_eq!(ctx.rational_str("22/7").unwrap(), ctx.rational(22, 7));
    assert_eq!(ctx.rational_str("-1/3").unwrap(), ctx.rational(-1, 3));
    assert_eq!(ctx.rational_str("4/-6").unwrap(), ctx.rational(-2, 3));
    assert_eq!(ctx.rational_str("12").unwrap(), ctx.int(12));
    assert_eq!(ctx.rational_str("  3 / 9  ").unwrap(), ctx.rational(1, 3));
}

#[test]
fn rational_str_arbitrary_precision() {
    let ctx = Context::new();
    let e = ctx
        .rational_str("340282366920938463463374607431768211456/2")
        .unwrap();
    assert_eq!(e.as_bigint(), Some(BigInt::from(2).pow(127)));
}

#[test]
fn rational_str_rejects_garbage() {
    let ctx = Context::new();
    for bad in ["", "1/0", "x/2", "1.5", "1/2/3", "/", "1/"] {
        assert!(
            matches!(
                ctx.rational_str(bad),
                Err(SymplexError::InvalidArgument { .. })
            ),
            "{bad:?}"
        );
    }
}

#[test]
fn decimal_str_basic() {
    let ctx = Context::new();
    assert_eq!(ctx.decimal_str("0.1").unwrap(), ctx.rational(1, 10));
    assert_eq!(ctx.decimal_str("-2.75").unwrap(), ctx.rational(-11, 4));
    assert_eq!(ctx.decimal_str("100").unwrap(), ctx.int(100));
    assert_eq!(ctx.decimal_str(".5").unwrap(), ctx.rational(1, 2));
}

#[test]
fn decimal_str_exponents() {
    let ctx = Context::new();
    assert_eq!(ctx.decimal_str("1e-5").unwrap(), ctx.rational(1, 100_000));
    assert_eq!(ctx.decimal_str("2.5E3").unwrap(), ctx.int(2500));
    assert_eq!(ctx.decimal_str("-1.5e+1").unwrap(), ctx.int(-15));
    assert_eq!(
        ctx.decimal_str("6.02e23").unwrap(),
        ctx.from_bigint(BigInt::from(602) * BigInt::from(10).pow(21))
    );
}

#[test]
fn decimal_str_is_exact_for_long_inputs() {
    let ctx = Context::new();
    let s = "3.14159265358979323846264338327950288419716939937510";
    let e = ctx.decimal_str(s).unwrap();
    let scaled = &e * &ctx.from_bigint(BigInt::from(10).pow(50));
    assert_eq!(
        scaled.as_bigint().unwrap().to_string(),
        "314159265358979323846264338327950288419716939937510"
    );
}

#[test]
fn decimal_str_rejects_non_decimals() {
    let ctx = Context::new();
    for bad in ["", "1/2", "x", "1e", "e3", ".", "1.2.3", "0x1F", "1_000"] {
        assert!(ctx.decimal_str(bad).is_err(), "{bad:?}");
    }
}

// ── complex / symbols / apply ──────────────────────────────────────────

#[test]
fn complex_builds_re_plus_im_i() {
    let ctx = Context::new();
    let z = ctx.complex(&ctx.int(1), &ctx.int(2));
    assert_eq!(z, &ctx.int(1) + &(&ctx.i_unit() * 2));
    let (re, im) = z.as_real_imag();
    assert_eq!(re, ctx.int(1));
    assert_eq!(im, ctx.int(2));
}

#[test]
fn complex_with_symbolic_parts() {
    let ctx = Context::new();
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let z = ctx.complex(&a, &b);
    assert_eq!(z, &a + &(&b * &ctx.i_unit()));
    assert_eq!(ctx.complex(&a, &ctx.zero()), a);
}

#[test]
fn symbols_and_symbols_indexed() {
    let ctx = Context::new();
    let v = ctx.symbols(&["alpha", "beta"]);
    assert_eq!(v, vec![ctx.symbol("alpha"), ctx.symbol("beta")]);
    let idx = ctx.symbols_indexed("q", 4);
    let names: Vec<String> = idx.iter().map(|s| format!("{s}")).collect();
    assert_eq!(names, vec!["q0", "q1", "q2", "q3"]);
    assert!(ctx.symbols(&[]).is_empty());
}

#[test]
fn apply_creates_generic_function_application() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.apply("f", &[&x]);
    assert_eq!(format!("{f}"), "f(x)");
    assert_eq!(f.expr_type(), ExprType::Apply);
    assert_eq!(f.free_symbols(), vec![x.clone()]);
}

#[test]
fn apply_is_left_alone_by_eval_and_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.apply("f", &[&x]);
    assert_eq!(f.eval(), f);
    assert_eq!(f.simplify(), f);
    // Arguments are still evaluated.
    let g = ctx.apply("g", &[&(&ctx.int(2) + 3)]);
    assert_eq!(g, ctx.apply("g", &[ctx.int(5)]));
}

#[test]
fn apply_diff_yields_formal_derivative_with_chain_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.apply("f", &[&x]);
    let d = f.diff(&x);
    assert_eq!(format!("{d}"), "Derivative(f(x), x)");
    assert!(d.has_unevaluated());
    assert!(f.try_diff(&x).is_err());
    let chain = ctx.apply("f", &[x.powi(2)]).diff(&x);
    assert_eq!(format!("{chain}"), "2*x*Derivative(f(x^2), x^2)");
}

#[test]
fn apply_cannot_be_evaluated_numerically() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.apply("f", &[&x]);
    assert!(matches!(
        f.compile(&["x"]),
        Err(SymplexError::NotImplemented(_))
    ));
    assert!(f.subs_i64(&x, 1).eval_f64().is_err());
}

#[test]
fn apply_multiple_args_and_zero_args() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let h = ctx.apply("h", &[&x, &y, &ctx.int(3)]);
    assert_eq!(format!("{h}"), "h(x, y, 3)");
    assert_eq!(h.args().len(), 3);
    let k = ctx.apply::<Ex>("k", &[]);
    assert_eq!(format!("{k}"), "k()");
}

#[test]
fn apply_survives_json_round_trip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.apply("f", &[&x, &ctx.int(2)]);
    let back = ctx.from_json(&f.to_json().unwrap()).unwrap();
    assert_eq!(back, f);
}

// ── sum / product ──────────────────────────────────────────────────────

#[test]
fn context_sum_product_on_empty() {
    let ctx = Context::new();
    assert_eq!(ctx.sum(Vec::<Ex>::new()), ctx.zero());
    assert_eq!(ctx.product(Vec::<Ex>::new()), ctx.one());
}

#[test]
fn context_sum_product_on_values() {
    let ctx = Context::new();
    let xs = ctx.symbols_indexed("x", 3);
    assert_eq!(ctx.sum(&xs), &(&xs[0] + &xs[1]) + &xs[2]);
    assert_eq!(ctx.product(xs.iter()), &(&xs[0] * &xs[1]) * &xs[2]);
    assert_eq!(ctx.sum((1..=100).map(|n| ctx.int(n))), ctx.int(5050));
}
