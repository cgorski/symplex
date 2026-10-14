//! Tests of the 0.50 refactor (w6d): the rest of the function registry and compile of special functions.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `Context::from_tree` of an `Apply` named like an internal function of
/// the limit engine builds a user function, about which nothing is known.
///
/// Before (0.49): the internal functions were `Apply` nodes resolved by the
/// text of their head, so `from_tree(Apply{"__ei_asym", [x]})` (an
/// infallible public API; `from_json` and `apply` already refused the name)
/// built the internal `e^{−x}·Ei(x)` with its asymptotic semantics:
/// `limit(__ei_asym(x), x, ∞)` was `0`.  Oracle: an undefined function has
/// no limit to compute (the result stays a formal `Limit`, or is an error).
#[test]
fn from_tree_never_builds_an_internal_function() {
    use symplex::tree::ExprTree;
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let tree = ExprTree::Apply {
        name: "__ei_asym".to_string(),
        args: vec![ExprTree::Symbol {
            name: "x".to_string(),
        }],
    };
    let f = ctx.from_tree(&tree);
    assert_eq!(f.to_string(), "__ei_asym(x)");
    let lim = f.limit(&x, &ctx.infinity());
    assert_ne!(
        lim.to_string(),
        "0",
        "an unknown function took on internal semantics"
    );
    assert!(f.try_limit(&x, &ctx.infinity()).is_err() || lim.to_string().contains("Limit"));
    // The JSON form and the other entry points still refuse the name.
    let json = f.to_json().unwrap();
    assert!(ctx.from_json(&json).is_err());
    assert!(ctx.apply("__ei_asym", &[&x]).is_err());
    assert!(ctx.parse("__ei_asym(x)").is_err());
}

/// The limits that go through the internal functions (now nodes of their
/// own, built only by the limit engine) are unchanged.
///
/// Oracle: DLMF 6.12.2 `Ei(x) ~ eˣ/x`, 7.12.1 `erfc(x) ~ e^{−x²}/(x√π)`,
/// 6.2.16 with 6.12.2 `Chi(x) ~ eˣ/(2x)`, 10.40.1 `I₀(x) ~ eˣ/√(2πx)`:
/// the limits are `1`, `1/√π`, `1/2`, `1/√(2π)`.
#[test]
fn limits_through_the_internal_functions_are_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();
    let e_neg_x = (-&x).exp();
    let cases = [
        (&x.ei() * &x * &e_neg_x, "1"),
        (&x.erfc() * &x * &x.powi(2).exp(), "1/sqrt(pi)"),
        (&x.chi() * &x * &e_neg_x, "1/2"),
    ];
    for (e, want) in cases {
        let got = e.try_limit(&x, &oo).unwrap();
        assert_eq!(got, ctx.parse(want).unwrap(), "{e}: {got}");
    }
    let i0 = ctx.parse("besseli(0, x)*sqrt(x)*exp(-x)").unwrap();
    let got = i0.try_limit(&x, &oo).unwrap();
    let v = got.eval_f64().unwrap();
    assert!((v - 0.398_942_280_401_432_7).abs() < 1e-15, "{got}");
}

/// The Rust, C and Python emitters refuse the `DiracDelta` of a constant
/// without a value, as `compile()` does.
///
/// Before (0.49): they wrote any `DiracDelta` as `0.0`:
/// `ln(√2 + DiracDelta(0)).to_rust_fn("f", &[])` was
/// `(0.5_f64.exp2() + 0.0_f64).ln()` = 0.3466, while `compile()` refused.
/// Oracle: `δ(0)` has no value (the distribution is not a function at its
/// support; SymPy leaves `DiracDelta(0)` unevaluated and `N(DiracDelta(0))`
/// stays `DiracDelta(0)`).
#[test]
fn every_back_end_refuses_dirac_delta_of_zero() {
    let ctx = Context::new();
    let e = (ctx.int(2).sqrt() + ctx.int(0).dirac_delta()).ln();
    assert!(matches!(
        e.compile(&[]),
        Err(SymplexError::Unevaluable { .. })
    ));
    for r in [
        e.to_rust_fn("f", &[]),
        e.to_c_fn("f", &[]),
        e.to_python(),
        e.to_python_fn("f", &[]),
        e.to_numpy(),
        e.to_julia(),
    ] {
        match r {
            Err(SymplexError::Unevaluable { reason }) => {
                assert!(reason.contains("DiracDelta(0)"), "{reason}")
            }
            other => panic!("expected Unevaluable, got {other:?}"),
        }
    }
    // A constant with a value: `δ(1) = 0`, a literal in every back end.
    let one = (ctx.int(2).sqrt() + ctx.int(1).dirac_delta()).ln();
    let v = one.compile(&[]).unwrap().call(&[]);
    assert!((v - std::f64::consts::LN_2 / 2.0).abs() < 1e-16, "{v}");
    assert!(one.to_rust_fn("f", &[]).is_ok());
    assert!(one.to_c_fn("f", &[]).is_ok());
    assert!(one.to_python().is_ok());
}

/// `DiracDelta(u)` of a variable `u` has the same pointwise value in every
/// back end: 0 where `u ≠ 0`, NaN at `u = 0` (where it has no value) and for
/// a NaN argument.
///
/// Before (0.49): the VM gave `0.0` at `u = 0` (a value the distribution
/// does not have), and the Rust, C and Python emitters wrote `0.0` for any
/// argument (also NaN).  Oracle: the definition of `δ` (0 off the origin,
/// no value at it), as `evalf` (`DiracDelta(0)` is `Unevaluable`).
#[test]
fn dirac_delta_of_a_variable_is_its_pointwise_value_everywhere() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = (&x - 1).dirac_delta();
    let f = d.compile(&["x"]).unwrap();
    assert_eq!(f.call(&[0.5]), 0.0);
    assert_eq!(f.call(&[f64::INFINITY]), 0.0);
    assert!(f.call(&[1.0]).is_nan());
    assert!(f.call(&[f64::NAN]).is_nan());
    let rust = d.to_rust_fn("f", &["x"]).unwrap();
    assert!(rust.contains("abs()") && rust.contains("NAN"), "{rust}");
    let c = d.to_c_fn("f", &["x"]).unwrap();
    assert!(
        c.contains("symplex_dirac_delta(") && c.contains("fabs(x) > 0.0 ? 0.0 : NAN"),
        "{c}"
    );
    assert_eq!(
        d.to_python().unwrap(),
        "(0.0 if abs(x - 1) > 0 else math.nan)"
    );
    assert_eq!(
        d.to_numpy().unwrap(),
        "numpy.where(numpy.abs(x - 1) > 0, 0.0, numpy.nan)"
    );
    assert_eq!(d.to_julia().unwrap(), "(abs(x - 1) > 0 ? 0.0 : NaN)");
}

/// `compile()`, `to_rust_fn` and `to_c_fn` evaluate the complete elliptic
/// integrals `K(m)`, `E(m)` (parameter `m`), with the same runtime routine
/// in all three (AGM, DLMF 19.8.5–6; series about `m = 1`, DLMF 19.12.2;
/// `m < 0` by DLMF 19.7.5): at most 5 and 6 ulps from mpmath on 69,758
/// points, bit-identical across the back ends.
///
/// Before (0.49): all three refused them ("cannot compile
/// `Apply(elliptic_k, …)` to a numerical function").  Oracle: mpmath 1.3
/// `ellipk`/`ellipe` at 30 digits: K(1/2) = 1.8540746773013719184,
/// E(1/2) = 1.3506438810476755025, K(−1) = 1.3110287771460599052,
/// E(0.99) = 1.0159935450252239477, E(−10¹⁰) = 100000.00006699609913;
/// `K(1) = +∞`, `E(1) = 1`; for `m > 1` the values are not real (NaN).
#[test]
fn compile_evaluates_the_complete_elliptic_integrals() {
    let ctx = Context::new();
    let k = ctx.parse("elliptic_k(x)").unwrap();
    let e = ctx.parse("elliptic_e(x)").unwrap();
    let fk = k.compile(&["x"]).unwrap();
    let fe = e.compile(&["x"]).unwrap();
    let close = |a: f64, b: f64| ((a - b) / b).abs() <= 2e-15;
    assert!(close(fk.call(&[0.5]), 1.854_074_677_301_371_9));
    assert!(close(fe.call(&[0.5]), 1.350_643_881_047_675_5));
    assert!(close(fk.call(&[-1.0]), 1.311_028_777_146_059_9));
    assert!(close(fe.call(&[0.99]), 1.015_993_545_025_224));
    assert!(close(fe.call(&[-1e10]), 100_000.000_066_996_1));
    assert_eq!(fk.call(&[1.0]), f64::INFINITY);
    assert_eq!(fe.call(&[1.0]), 1.0);
    assert!(fk.call(&[2.0]).is_nan() && fe.call(&[2.0]).is_nan());
    let rust = k.to_rust_fn("f", &["x"]).unwrap();
    assert!(rust.contains("symplex_rt::elliptic_k(x)"), "{rust}");
    let c = e.to_c_fn("f", &["x"]).unwrap();
    assert!(
        c.contains("symplex_elliptic_e(x)") && c.contains("symplex_elliptic_ke("),
        "{c}"
    );
    // Python's `math` has neither: the refusal names SciPy's routine.
    match k.to_python() {
        Err(SymplexError::NotImplemented(msg)) => {
            assert!(msg.contains("scipy.special.ellipk"), "{msg}")
        }
        other => panic!("expected NotImplemented, got {other:?}"),
    }
}
