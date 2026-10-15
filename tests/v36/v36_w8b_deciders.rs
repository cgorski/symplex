//! Tests of the 0.50 refactor (w8b): the remaining deciders and budgets.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `Γ(1/3)·Γ(2/3) − 2π/√3` is 0, but up to 0.49 `is_zero` answered `None`
/// (and `is_positive`, `is_nonnegative` too): the value is 0 to every
/// digit, and the exact zero tests of `eval` do not know the reflection
/// formula.  The query now asks the zero test of the limit engine, which
/// rewrites `Γ(1/3)·Γ(2/3)` as `π/sin(π/3)`.
///
/// Oracle (exact identity): `Γ(z)·Γ(1 − z) = π/sin(πz)`, and `sin(π/3) =
/// √3/2`, so the constant is `2π/√3 − 2π/√3 = 0`.
#[test]
fn a_zero_by_the_reflection_formula_is_zero() {
    let ctx = Context::new();
    let k = ctx.parse("gamma(1/3)*gamma(2/3) - 2*pi/sqrt(3)").unwrap();
    assert_eq!(k.is_zero(), Some(true));
    assert_eq!(k.is_positive(), Some(false));
    assert_eq!(k.is_nonnegative(), Some(true));
    // Asked again in the same context (the verdict is remembered).
    assert_eq!(k.is_zero(), Some(true));
}

/// `sin²1 + cos²1 − 1` is 0: up to 0.49 `is_zero` said `None`; the
/// identities of the limit engine's zero test (a rational function of
/// `e^i` that vanishes) prove it.  A tiny nonzero constant is shown
/// nonzero by the deeper certified evaluation: `e^{10⁻¹⁵⁰} − 1` (`None`
/// up to 0.49).  A constant `evalf` cannot evaluate stays undecided.
///
/// Oracles: `sin²z + cos²z = 1` (exact identity); mpmath
/// `mp.dps = 200; exp(mpf(10)**-150) - 1` = `1.0000…e-150` (nonzero).
#[test]
fn constants_zero_by_an_identity_or_nonzero_below_the_digits() {
    let ctx = Context::new();
    let c = ctx.parse("sin(1)^2 + cos(1)^2 - 1").unwrap();
    assert_eq!(c.is_zero(), Some(true));
    let tiny = ctx.parse("exp(10^-150) - 1").unwrap();
    assert_eq!(tiny.is_zero(), Some(false));
    // (`legendre(3/2, E)` has a value since 0.54: 5.2479; a Legendre
    // function on its cut `x ≤ −1` has none here.)
    let opaque = ctx.parse("legendre(3/2, -E) - 1").unwrap();
    assert_eq!(opaque.is_zero(), None);
    assert_eq!(
        ctx.parse("legendre(3/2, E) - 1").unwrap().is_zero(),
        Some(false)
    );
}

/// The zero test behind `is_zero` makes no claim for a value that is not a
/// finite number, nor for `δ` at a non-real argument: the limit engine's
/// test reads `δ(a)` as 0 for every `a ≠ 0`, but `δ` is defined on the
/// reals.  (A first version of this change answered `Some(false)` for
/// NaN and `zoo` — 902 golden cases — and `Some(true)` for `δ(i)`.)  `δ`
/// at a real nonzero argument is 0.
///
/// Oracle: SymPy 1.14 `nan.is_zero` is `None`; `DiracDelta(I)` raises
/// "Function defined only for Real Values"; `DiracDelta(atan(pi))` is `0`.
#[test]
fn no_zero_claim_for_nan_infinities_or_delta_off_the_real_line() {
    let ctx = Context::new();
    for s in [
        "harmonic(-1)*E",
        "re(besselk(-1/2, 0))",
        "DiracDelta(I)",
        "DiracDelta(sqrt(2) + I)",
    ] {
        let e = ctx.parse(s).unwrap();
        assert_eq!(e.is_zero(), None, "{s} = {e}");
    }
    let d = ctx.parse("DiracDelta(atan(pi))").unwrap();
    assert_eq!(d.is_zero(), Some(true));
}

/// `Σ_{k≥1} 2ᵏ·e^{−c·k}` with `c = 0.6931471805599453` diverges: the ratio
/// is `2·e^{−c} = e^{ln 2 − c}` and `ln 2 − c ≈ 9.4·10⁻¹⁸ > 0`.  Up to 0.49
/// the sign of `ln 2 − c` was that of an `f64` sum, which is exactly `0.0`
/// here, and `0.0` counted as negative: `is_convergent` said `Some(true)`.
/// Now the sign is certified.  The neighbouring rational (one unit more in
/// the last digit) gives a ratio below 1, which converges.
///
/// Oracle: mpmath `mp.dps = 40; log(2) - mpf(6931471805599453)/10**16` =
/// `9.4171…e-18`, and `log(2) - mpf(6931471805599454)/10**16` =
/// `-9.0582…e-18`.
#[test]
fn the_sign_of_a_growth_rate_next_to_ln_2_is_certified() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let above = ctx
        .parse("2^k*exp(-6931471805599453/10000000000000000*k)")
        .unwrap();
    assert_eq!(above.is_convergent(&k), Some(false));
    assert_eq!(above.is_absolutely_convergent(&k), Some(false));
    let below = ctx
        .parse("2^k*exp(-6931471805599454/10000000000000000*k)")
        .unwrap();
    assert_eq!(below.is_convergent(&k), Some(true));
}

/// `Σ_{k≥1} c` for the non-real constant `c = −1 + i·(e^{10⁻³⁰} − 1 −
/// 10⁻³⁰)` diverges, but not to `−∞`: its terms have the imaginary part
/// `≈ 5·10⁻⁶¹`.  Up to 0.49 the sign of `c` was read off a 16-digit value,
/// `−1`, without a proof that `c` is real, and the sum was `−∞`.  Now the
/// sign of a constant is the one its assumption queries decide (a
/// certified value of a constant proved real), and the sum stays
/// unevaluated; the real constant `e^{10⁻³⁰} − 1 − 10⁻³⁰ > 0` still sums
/// to `∞`.
///
/// Oracle: mpmath `mp.dps = 100; exp(mpf(10)**-30) - 1 - mpf(10)**-30` =
/// `5.0000…e-61`, so `Im c ≠ 0` and `Σ c = N·c` leaves the real axis.
#[test]
fn a_divergent_sum_of_a_non_real_constant_has_no_real_direction() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let c = ctx.parse("-1 + I*(exp(10^-30) - 1 - 10^-30)").unwrap();
    let s = c.summation(&k, &ctx.int(1), &ctx.infinity());
    assert_ne!(s.to_string(), "-oo");
    assert!(s.to_string().starts_with("Sum("), "Σ c = {s}");
    let real = ctx.parse("exp(10^-30) - 1 - 10^-30").unwrap();
    let s = real.summation(&k, &ctx.int(1), &ctx.infinity());
    assert_eq!(s.to_string(), "oo");
}

/// The step budget of the Gruntz algorithm is a deterministic `Work`
/// budget now (`BudgetKind::LimitSteps`), and the integrator's step budget
/// and the parametric rational integrator's are too: their kinds are
/// named in the public enum.  Up to 0.49 they were a counter in
/// `gruntz::Budget` and thread-local counters.
///
/// Oracle: the names are the documented `Display` strings.
#[test]
fn the_new_budget_kinds_have_names() {
    use symplex::BudgetKind;
    assert_eq!(BudgetKind::LimitSteps.to_string(), "limit steps");
    assert_eq!(
        BudgetKind::IntegrationSteps.to_string(),
        "integration steps"
    );
    assert_eq!(
        BudgetKind::ParametricRational.to_string(),
        "parametric rational integration"
    );
}
