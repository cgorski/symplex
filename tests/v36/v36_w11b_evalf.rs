//! Tests of the 0.50 refactor (w11b): evalf's ball type completed.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn decimal(src: &str, digits: u32) -> Result<String, SymplexError> {
    let ctx = Context::new();
    ctx.parse(src).unwrap().eval_decimal(digits)
}

/// The bound rules moved into `Ball` methods (`b^x`, the inverse functions
/// next to a branch point, `atan2` on its cut, `floor`, `max`,
/// `KroneckerDelta`, a special function) and the values and bounds held in
/// one map of balls instead of two certify the same digits.  Nothing was
/// wrong before: this pins the values the rules certify.
///
/// Oracle: mpmath 1.3 at 40 digits (100 for `asin`) — `atan2(1, -1)` =
/// 2.356194490192344928846982537459627163148, `asin(1 - 10**-40)` =
/// 1.570796326794896619217179556016020491611, `besselj(0, 17/7)` =
/// -0.01226599133046086110248385259170676701889, `(2/3)**sqrt(2)` =
/// 0.5635978831234872766838163906572957236125, `sqrt(2)` =
/// 1.414213562373095048801688724209698078570, and
/// `floor(10**30*(sqrt(2) - 1))` = 414213562373095048801688724209.
#[test]
fn moved_bound_rules_keep_their_digits() {
    for (src, want) in [
        ("atan2(1, -1)", "2.35619449019234492884698253746"),
        ("asin(1 - 10^-40)", "1.57079632679489661921717955602"),
        ("besselj(0, 17/7)", "-0.0122659913304608611024838525917"),
        ("(2/3)^sqrt(2)", "0.563597883123487276683816390657"),
        ("max(sqrt(2), 7/5)", "1.41421356237309504880168872421"),
        (
            "floor(10^30*(sqrt(2) - 1))",
            "414213562373095048801688724209",
        ),
        ("KroneckerDelta(sqrt(2), 7/5)", "0"),
    ] {
        assert_eq!(decimal(src, 30).unwrap(), want, "{src}");
    }
}

/// `compile` decides a symbol-free `DiracDelta(c)` as `evalf` does: 0 for a
/// `c` certainly not 0, no value at 0.  Up to 0.49 the VM's pointwise 0
/// compiled `ln(sqrt(2) + DiracDelta(0))` to `0.34657359027997264` (`ln √2`)
/// while `evalf` refused it.
///
/// Oracle: the definition (`δ(x) = 0` for `x ≠ 0`, no value at 0; SymPy
/// 1.14 leaves `DiracDelta(0)` unevaluated and `.evalf()` gives no number)
/// and `eval_decimal`, which refuses `DiracDelta(0)`.
#[test]
fn compile_refuses_a_constant_dirac_delta_at_zero() {
    let ctx = Context::new();
    let e = ctx.parse("ln(sqrt(2) + DiracDelta(0))").unwrap();
    assert!(e.eval_decimal(15).is_err());
    let err = e
        .compile(&[])
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("DiracDelta(0)"), "compiled, or: {err}");
    let e = ctx.parse("x + DiracDelta(0)").unwrap();
    assert!(e.compile(&["x"]).is_err());
    // A nonzero constant argument folds to 0, a symbolic one is unchanged.
    let f = ctx
        .parse("x + DiracDelta(sqrt(2) - 1)")
        .unwrap()
        .compile(&["x"])
        .unwrap();
    assert_eq!(f(&[2.0]), 2.0);
    let f = ctx.parse("DiracDelta(x)").unwrap().compile(&["x"]).unwrap();
    assert_eq!(f(&[1.0]), 0.0);
}

/// `E₁` has a logarithmic pole at 0: `expint(1, 0)` is `+∞`, refused as
/// divergent like `expint(n, 0)` for every `n ≤ 1`.  Up to 0.49 `n − 1 = +0`
/// counted as positive, the value came out `1/0`, and the refusal was
/// "precision exhausted" (requested 30 digits, achieved 0).
///
/// Oracle: mpmath 1.3 `expint(1, 0)` = `+inf`, `expint(2, 0)` = 1, `e1(1e-30)`
/// = 68.50033712491983765993323155044952379699.
#[test]
fn expint_one_at_zero_is_a_pole() {
    let err = decimal("expint(1, 0)", 30).unwrap_err();
    assert!(
        matches!(err, SymplexError::Unevaluable { .. }) && err.to_string().contains("diverges"),
        "{err:?}"
    );
    assert_eq!(decimal("expint(2, 0)", 30).unwrap(), "1");
    assert_eq!(
        decimal("expint(1, 10^-30)", 30).unwrap(),
        "68.5003371249198376599332315504"
    );
}

/// One convention for the Bernoulli numbers, `B₁ = +1/2` (`Bₙ = Bₙ(1) =
/// −n·ζ(1 − n)`): the exact `bernoulli(1)`, `ntheory::bernoulli(1)` and
/// the Bernoulli function `−s·ζ(1 − s)` that `evalf` computes off the
/// integers now agree.  Up to 0.49 `bernoulli(1)` was `−1/2` while
/// `bernoulli(1 + 10⁻³⁰)` and `bernoulli(1 − 10⁻³⁰)` were `+0.5`: a jump
/// of 1 at the one integer where the conventions differ.
///
/// Oracle: SymPy 1.14 `bernoulli(1)` = 1/2, `bernoulli(1 ± Rational(1,
/// 10**30)).evalf(30)` = 0.500000000000000000000000000000; mpmath 1.3 at 100
/// digits `-s*zeta(1 - s)` = 0.4999999999999999999999999999995810614668
/// (`s = 1 + 10**-30`), 0.5000000000000000000000000000004189385332
/// (`s = 1 - 10**-30`); the identity `ζ(0) = −B₁`.
#[test]
fn bernoulli_one_is_the_value_of_the_bernoulli_function() {
    let ctx = Context::new();
    let half = ctx.parse("1/2").unwrap();
    assert_eq!(ctx.parse("bernoulli(1)").unwrap().eval(), half);
    assert_eq!(ctx.parse("-zeta(0)").unwrap().eval(), half);
    assert_eq!(decimal("bernoulli(1)", 30).unwrap(), "0.5");
    for (s, want) in [
        (
            "bernoulli(1 + 10^-30)",
            "0.4999999999999999999999999999995810614668",
        ),
        (
            "bernoulli(1 - 10^-30)",
            "0.5000000000000000000000000000004189385332",
        ),
    ] {
        assert_eq!(decimal(s, 40).unwrap(), want, "{s}");
    }
    let r = |p: i64, q: i64| num_rational::Ratio::new(p.into(), q.into());
    assert_eq!(symplex::ntheory::bernoulli(1), Some(r(1, 2)));
    // The other numbers are those of both conventions.
    assert_eq!(symplex::ntheory::bernoulli(2), Some(r(1, 6)));
    assert_eq!(symplex::ntheory::bernoulli(3), Some(r(0, 1)));
}
