//! 0.22 — coverage regressions from the 0.21 differential audit: public
//! entry points that the `cargo llvm-cov` run behind
//! `target/scratch/coverage.md` found no test binary ever entered.
//!
//! Sections follow the report's priority list: the antiderivative-based
//! Fourier fallback (`src/calculus/fourier.rs`, 0 %), `Matrix::lll`,
//! `QMatrix::{get_mut, set, as_slice}`, the discontinuity checks of
//! `calculus/limit.rs`, every method of `certificates::Outcome`, the LaTeX
//! / MathML paths of node kinds no test had printed, and the arena
//! compaction / `ExprTree` round trip over the same node kinds.
//!
//! Reference values cite SymPy 1.14 (`symplex/.venv/bin/python`) by the
//! call that produced them, or the textbook identity when SymPy has no
//! closed form.  Exact results are compared with `Ex::equals` / `==`
//! (structural identity) or `Ratio<BigInt>` equality; `f64` at `1e-12`.
//! Printer strings are pinned after reading the printer code and checking
//! them against the LaTeX macros / Presentation MathML elements they emit.
//! Nothing below was computed by hand.

use symplex::certificates::{
    BoxBound, Outcome, ParamBound, PolyhedronOpts, Ray, SosOpts, prove_nonnegative_on_box,
    prove_nonnegative_on_halfline, prove_nonnegative_on_polyhedron, prove_sos,
};
use symplex::linprog::{q, qi};
use symplex::num_rational::Ratio;
use symplex::prelude::*;

/// Absolute closeness at the tolerance the math allows (`1e-12` for an
/// exact expression evaluated in `f64`).
fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "{label}: got {actual:.17e}, expected {expected:.17e}"
    );
}

/// `e` at `x = 1/3` in `f64` (the evaluation point used for every numeric
/// Fourier check below; SymPy: `N(expr.subs(x, Rational(1, 3)), 20)`).
fn at_one_third(e: &Ex, x: &Ex) -> f64 {
    let ctx = x.context();
    e.subs(x, &ctx.rational(1, 3))
        .eval()
        .eval_f64()
        .unwrap_or_else(|err| panic!("{e} at x = 1/3 did not evaluate: {err:?}"))
}

fn err_is_computation_failed<T: std::fmt::Debug>(r: &Result<T, SymplexError>, label: &str) {
    assert!(
        matches!(r, Err(SymplexError::ComputationFailed { .. })),
        "{label}: expected a ComputationFailed error, got {r:?}"
    );
}

fn err_is_invalid<T: std::fmt::Debug>(r: &Result<T, SymplexError>, label: &str) {
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "{label}: expected an InvalidArgument error, got {r:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Fourier series — `Ex::fourier_series` (fixed interval) and the general
// `fourier_series_on`
// ═══════════════════════════════════════════════════════════════════════════

/// SymPy: `fourier_series(x, (x, -pi, pi)).truncate(3)` =
/// `2*sin(x) - sin(2*x) + 2*sin(3*x)/3`.
#[test]
fn fourier_series_of_x_is_the_sawtooth_sine_series() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.fourier_series(&x, 3);
    let expected = 2 * x.sin() - (2 * &x).sin() + ctx.rational(2, 3) * (3 * &x).sin();
    assert_eq!(series.equals(&expected), Some(true), "got {series}");
}

/// SymPy: `fourier_series(x**2, (x, -pi, pi)).truncate(4)` =
/// `-4*cos(x) + cos(2*x) - 4*cos(3*x)/9 + pi**2/3`; at `x = 1/3` that
/// partial sum is `0.055793251050832563938` (`N(…, 20)`).
#[test]
fn fourier_series_of_x_squared_is_pi_squared_over_3_minus_alternating_cosines() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.powi(2).fourier_series(&x, 3);
    let expected =
        ctx.pi().powi(2) / 3 - 4 * x.cos() + (2 * &x).cos() - ctx.rational(4, 9) * (3 * &x).cos();
    assert_eq!(series.equals(&expected), Some(true), "got {series}");
    close(
        at_one_third(&series, &x),
        0.05579325105083256,
        "x^2 partial sum at 1/3",
    );
}

/// SymPy: `fourier_series(cos(x), (x, -pi, pi)).truncate(3)` = `cos(x)`
/// (a₁ = 1, every other coefficient 0).
#[test]
fn fourier_series_of_cos_x_reproduces_cos_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.cos().fourier_series(&x, 3);
    assert_eq!(series.equals(&x.cos()), Some(true), "got {series}");
}

/// SymPy: `fourier_series(sign(x), (x, -pi, pi)).truncate(3)` =
/// `4*sin(x)/pi + 4*sin(3*x)/(3*pi) + 4*sin(5*x)/(5*pi)` — SymPy's
/// `truncate(n)` counts non-zero terms, so its first three are the odd
/// harmonics 1, 3, 5; `fourier_series(&x, 3)` stops at harmonic 3.
#[test]
fn fourier_series_of_sign_x_is_the_square_wave_odd_sine_series() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.sign().fourier_series(&x, 3);
    let expected = 4 / ctx.pi() * (x.sin() + (3 * &x).sin() / 3);
    assert_eq!(series.equals(&expected), Some(true), "got {series}");
}

/// The fixed-interval `fourier_series` and the general
/// `fourier_series_on(&x, -π, π, n).truncate(n)` are different
/// implementations; on `x²` they must produce the same partial sum.
#[test]
fn fourier_series_on_symmetric_interval_agrees_with_fourier_series() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let fixed = x.powi(2).fourier_series(&x, 3);
    let general = x
        .powi(2)
        .fourier_series_on(&x, &(-&pi), &pi, 3)
        .expect("x^2 has closed-form coefficients")
        .truncate(3);
    assert_eq!(fixed.equals(&general), Some(true), "{fixed} vs {general}");
    close(
        at_one_third(&fixed, &x) - at_one_third(&general, &x),
        0.0,
        "difference at 1/3",
    );
}
/// `exp(a·x)` with a free parameter, one harmonic: every coefficient
/// integral has a closed form (a `Piecewise` on `a ≠ 0`), so this still
/// goes through `fourier_series_on`.  Specialised to `a = 1/2` and
/// evaluated at `x = 1/3`: `1.2947737090821172637` (SymPy: `a0/2 +
/// a1*cos(x) + b1*sin(x)` with `a0 = 4*sinh(pi/2)/pi`, `a1 =
/// -4*sinh(pi/2)/(5*pi)`, `b1 = 8*sinh(pi/2)/(5*pi)`; `N(…, 20)`).
#[test]
fn fourier_series_of_exp_ax_specialises_to_the_exp_half_x_series() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let series = (&a * &x).exp().fourier_series(&x, 1);
    assert!(!series.has_unevaluated(), "{series}");
    let at_half = series.subs(&a, &ctx.rational(1, 2)).eval();
    close(
        at_one_third(&at_half, &x),
        1.294773709082117,
        "exp(x/2) one-harmonic partial sum at 1/3",
    );
}

/// `cosh(a·x)`, one harmonic: an even function, so the series has no sine
/// terms.  At `a = 1/2` the partial sum is `2·sinh(π/2)/π · (1 − (2/5)
/// cos x)` (SymPy: `integrate(cosh(x/2), (x, -pi, pi))/(2*pi)` =
/// `2*sinh(pi/2)/pi`, `integrate(cosh(x/2)*cos(x), (x, -pi, pi))/pi` =
/// `-4*sinh(pi/2)/(5*pi)`), which at `x = 1/3` is
/// `0.91128781279706889793`.
#[test]
fn fourier_series_of_cosh_ax_has_only_cosine_terms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let series = (&a * &x).cosh().fourier_series(&x, 1);
    let at_half = series.subs(&a, &ctx.rational(1, 2)).eval();
    let expected = 2 * (ctx.pi() / 2).sinh() / ctx.pi() * (1 - ctx.rational(2, 5) * x.cos());
    assert_eq!(at_half.equals(&expected), Some(true), "got {at_half}");
    close(
        at_one_third(&at_half, &x),
        0.9112878127970689,
        "cosh(x/2) one-harmonic partial sum at 1/3",
    );
}

/// `x·exp(a·x)` (odd × neither), one harmonic: both a₁ and b₁ are
/// non-zero.  SymPy with `a = 1/2`: `a0 = integrate(x*exp(x/2),
/// (x,-pi,pi))/pi`, `a1`, `b1` likewise; `N(a0/2 + a1*cos(1/3) +
/// b1*sin(1/3), 20)` = `0.53367730905735027508`.
#[test]
fn fourier_series_of_x_exp_ax_matches_sympy_numerically() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let series = (&x * (&a * &x).exp()).fourier_series(&x, 1);
    assert!(!series.has_unevaluated(), "{series}");
    let at_half = series.subs(&a, &ctx.rational(1, 2)).eval();
    close(
        at_one_third(&at_half, &x),
        0.5336773090573503,
        "x exp(x/2) one-harmonic partial sum at 1/3",
    );
}

// ── The antiderivative fallback (`calculus/fourier.rs`) ─────────────────
//
// `Ex::fourier_series` tries `fourier_series_on` first and only falls back
// to the antiderivative expansion when a coefficient's definite integral
// has no closed form.  With a free parameter that happened (until 0.24) as
// soon as the integrand was `exp(k·a·x)·sin(m·x)` with `k·m ≥ 2` —
// `exp(2ax)` at the first harmonic, `exp(ax)` at the second — where the
// antiderivative carries a `Piecewise` on the complex zero of `k²a² + m²`
// (`a ≠ (−4)^(−1/2)`, i.e. `a ≠ −i/2`).  Since 0.25 those coefficients
// have closed forms and the direct route succeeds; the tests below check
// its value against SymPy and the mean-value property of the series.

/// The mean of a truncated Fourier series over the period is `a₀/2 =
/// (1/2π)∫f`, and the harmonics cancel on the equispaced sample `x = 0,
/// π`.  For `exp(2·a·x)` (the fallback path until 0.24: `fourier_series_on` failed at
/// `b₁`) with `a = 1/2` that is `sinh(π)/π` (mpmath 1.3, 30 digits:
/// `3.67607791037497772069569749203`).
#[test]
fn fourier_series_fallback_for_exp_2ax_has_the_mean_value_of_f() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let pi = ctx.pi();
    let f = (2 * &a * &x).exp();
    // Since 0.25 the coefficient integrals have closed forms and the direct
    // route succeeds: at a = 1/2, x = 1/3 the one-harmonic partial sum of eˣ
    // is 1.405135751056150642739163 (SymPy 1.14: a0/2 + a1*cos(1/3) +
    // b1*sin(1/3) with the coefficients by `integrate(exp(x)*cos(k*x),
    // (x, -pi, pi))/pi` etc., `N(…, 25)`).
    let direct = f
        .fourier_series_on(&x, &(-&pi), &pi, 1)
        .expect("closed-form coefficients")
        .truncate(1);
    let v = direct
        .subs(&a, &ctx.rational(1, 2))
        .subs(&x, &ctx.rational(1, 3))
        .eval();
    assert!(
        v.eval_decimal(25)
            .unwrap()
            .starts_with("1.40513575105615064273916"),
        "{direct}"
    );
    let series = f.fourier_series(&x, 1);
    assert!(
        !series.has_unevaluated(),
        "fallback left an unevaluated node: {series}"
    );
    let at_half = series.subs(&a, &ctx.rational(1, 2)).eval();
    let mean = ((at_half.subs(&x, &ctx.int(0)) + at_half.subs(&x, &pi)) / 2).eval();
    // sinh(π)/π written as the integrator leaves it: (e^π − e^{−π})/(2π).
    let expected = (pi.exp() - (-&pi).exp()) / (2 * &pi);
    assert_eq!(mean.equals(&expected), Some(true), "mean = {mean}");
    close(mean.eval_f64().unwrap(), 3.676077910374978, "sinh(pi)/pi");
}

/// Same property for the two-harmonic fallback series of `exp(a·x)`
/// (`fourier_series_on` failed at `b₂` until 0.24): the average over `x = 0, π/2, π,
/// 3π/2` kills harmonics 1 and 2 and leaves `a₀/2 = 2·sinh(π/2)/π` at
/// `a = 1/2` (mpmath: `1.46505238333663487760917937411`).
#[test]
fn fourier_series_fallback_for_exp_ax_two_harmonics_has_the_mean_value_of_f() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let pi = ctx.pi();
    let f = (&a * &x).exp();
    // The direct route succeeds since 0.25: at a = 1/2, x = 1/3 the
    // two-harmonic partial sum of e^(x/2) is 1.003901872535325748017044
    // (SymPy 1.14, as above).
    let direct = f
        .fourier_series_on(&x, &(-&pi), &pi, 2)
        .expect("closed-form coefficients")
        .truncate(2);
    let v = direct
        .subs(&a, &ctx.rational(1, 2))
        .subs(&x, &ctx.rational(1, 3))
        .eval();
    assert!(
        v.eval_decimal(25)
            .unwrap()
            .starts_with("1.00390187253532574801704"),
        "{direct}"
    );
    let series = f.fourier_series(&x, 2);
    assert!(!series.has_unevaluated(), "{series}");
    let at_half = series.subs(&a, &ctx.rational(1, 2)).eval();
    let samples = [ctx.int(0), &pi / 2, pi.clone(), 3 * &pi / 2];
    let mut total = ctx.int(0);
    for s in &samples {
        total = (&total + at_half.subs(&x, s)).eval();
    }
    let mean = (&total / 4).eval();
    // 2 sinh(π/2)/π as the integrator leaves it: (e^{π/2} − e^{−π/2})/π.
    let expected = ((&pi / 2).exp() - (-&pi / 2).exp()) / &pi;
    assert_eq!(mean.equals(&expected), Some(true), "mean = {mean}");
    close(
        mean.eval_f64().unwrap(),
        1.465052383336635,
        "2 sinh(pi/2)/pi",
    );
}

/// The whole fallback partial sum for `exp(2·a·x)` at `a = 1/2` is the
/// one-harmonic series of `eˣ`: at `x = 1/3`, `1.4051357510561506427`
/// (SymPy: `a0 = 2*sinh(pi)/pi`, `a1 = -sinh(pi)/pi`, `b1 = sinh(pi)/pi`;
/// `N(a0/2 + a1*cos(1/3) + b1*sin(1/3), 20)`).  The first-harmonic
/// coefficients carry `Piecewise(… if 1/2 != (-4)^(-1/2), …)` after the
/// substitution; until 0.22 evalf could not decide an (in)equality between
/// a rational and a complex literal, so the sum had no numeric value.
/// Equality needs no order, so since 0.22.1 it is decided for complex
/// operands too.
#[test]
fn fourier_series_fallback_for_exp_2ax_is_evaluable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let series = (2 * &a * &x).exp().fourier_series(&x, 1);
    let at_half = series.subs(&a, &ctx.rational(1, 2)).eval();
    close(
        at_one_third(&at_half, &x),
        1.405135751056151,
        "exp(x) one-harmonic partial sum at 1/3",
    );
}

/// The second harmonic of the `exp(a·x)` fallback has the same
/// undecidable `Piecewise` (`a ≠ (−1/4)^(−1/2)`).  SymPy (`a0/2 + Σ_{k≤2}
/// a_k cos(kx) + b_k sin(kx)` at `a = 1/2`, `x = 1/3`):
/// `1.0039018725353257480`.
#[test]
fn fourier_series_fallback_second_harmonic_of_exp_ax_is_evaluable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let series = (&a * &x).exp().fourier_series(&x, 2);
    let at_half = series.subs(&a, &ctx.rational(1, 2)).eval();
    close(
        at_one_third(&at_half, &x),
        1.003901872535326,
        "exp(x/2) two-harmonic partial sum at 1/3",
    );
}

/// `ln(2 + cos x)` has no elementary antiderivative.  0.22's fallback
/// substituted `±π` into the unevaluated antiderivative (and into
/// tan-half-angle antiderivatives, `zoo` at `±π`) and returned the *number*
/// `nan` with `has_unevaluated() == false`.  Since 0.22.1 each coefficient
/// goes through the definite integrator and, without a closed form, stays a
/// `DefiniteIntegral` that evaluates by quadrature.  The true
/// coefficients (Gradshteyn–Ryzhik 4.224.9, `r = 2 − √3`):
/// `a₀/2 = ln((2 + √3)/2)`, `aₙ = 2(−1)^{n+1} rⁿ/n`; SymPy confirms them
/// numerically (`N(Integral(log(2+cos(x))*cos(x), (x,-pi,pi))/pi, 20)` =
/// `0.53589838486224541294` = `4 − 2√3`) and the two-harmonic partial sum
/// at `x = 1/3` is `1.0737874509678854546`.
#[test]
fn fourier_series_of_ln_2_plus_cos_x_is_not_nan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = (2 + x.cos()).ln().fourier_series(&x, 2);
    assert!(series.has_unevaluated(), "{series}");
    close(
        at_one_third(&series, &x),
        1.073787450967885,
        "ln(2 + cos x) two-harmonic partial sum at 1/3",
    );
}
// ═══════════════════════════════════════════════════════════════════════════
// `Matrix::lll`
// ═══════════════════════════════════════════════════════════════════════════

/// SymPy: `DomainMatrix([[1,1,1],[-1,0,2],[3,5,6]], (3,3), ZZ).lll()`
/// (and `Matrix(...).lll(delta=Rational(3,4))`) =
/// `[[0, 1, 0], [1, 0, 1], [-1, 0, 2]]`.
#[test]
fn matrix_lll_reduces_the_textbook_basis_like_sympy() {
    let ctx = Context::new();
    let b = matrix![ctx, [1, 1, 1], [-1, 0, 2], [3, 5, 6]];
    let reduced = b.lll(Ratio::new(3, 4)).unwrap();
    assert_eq!(reduced, matrix![ctx, [0, 1, 0], [1, 0, 1], [-1, 0, 2]]);
}

/// The reduced basis spans the same lattice: `T = R·B⁻¹` is an integer
/// matrix with `det T = 1` (SymPy: `T = [[-4, -1, 1], [5, 1, -1], [0, 1,
/// 0]]`), and `|det R| = |det B| = 3`.
#[test]
fn matrix_lll_result_is_a_unimodular_transform_of_the_input() {
    let ctx = Context::new();
    let b = matrix![ctx, [1, 1, 1], [-1, 0, 2], [3, 5, 6]];
    let r = b.lll(Ratio::new(3, 4)).unwrap();
    let t = (&r * &b.inv().unwrap()).eval();
    assert_eq!(t, matrix![ctx, [-4, -1, 1], [5, 1, -1], [0, 1, 0]]);
    assert_eq!(t.det().unwrap(), ctx.int(1));
    assert_eq!(b.det().unwrap(), ctx.int(-3));
    assert_eq!(r.det().unwrap(), ctx.int(-3));
}

/// SymPy: `Matrix.eye(3).lll()` = `eye(3)` — an already-reduced basis is a
/// fixed point.
#[test]
fn matrix_lll_of_the_identity_is_the_identity() {
    let ctx = Context::new();
    let id = Matrix::identity(&ctx, 3).unwrap();
    assert_eq!(id.lll(Ratio::new(3, 4)).unwrap(), id);
}

/// SymPy: `Matrix([[201, 37], [1648, 297]]).lll()` = `[[1, 32], [40, 1]]`
/// (det −1279 preserved up to sign).
#[test]
fn matrix_lll_reduces_a_skewed_2d_basis_to_short_vectors() {
    let ctx = Context::new();
    let b = matrix![ctx, [201, 37], [1648, 297]];
    let r = b.lll(Ratio::new(3, 4)).unwrap();
    assert_eq!(r, matrix![ctx, [1, 32], [40, 1]]);
    assert_eq!(r.det().unwrap(), ctx.int(-1279));
}

/// The Lovász parameter must lie in the open interval `(1/4, 1)` (Lenstra,
/// Lenstra & Lovász 1982); both endpoints are rejected.
#[test]
fn matrix_lll_rejects_delta_outside_the_open_interval() {
    let ctx = Context::new();
    let b = matrix![ctx, [1, 1, 1], [-1, 0, 2], [3, 5, 6]];
    err_is_invalid(&b.lll(Ratio::new(1, 4)), "delta = 1/4");
    err_is_invalid(&b.lll(Ratio::new(1, 1)), "delta = 1");
    assert!(
        b.lll(Ratio::new(99, 100)).is_ok(),
        "delta = 99/100 is valid"
    );
}

/// Dependent rows are not a lattice basis and a symbolic entry is not an
/// integer: both are `InvalidArgument`, not a wrong reduction.
#[test]
fn matrix_lll_rejects_dependent_rows_and_non_integer_entries() {
    let ctx = Context::new();
    err_is_invalid(
        &matrix![ctx, [1, 2], [2, 4]].lll(Ratio::new(3, 4)),
        "dependent rows",
    );
    let t = ctx.symbol("t");
    err_is_invalid(
        &(matrix![ctx, [1, 2], [3, 4]] * &t)
            .eval()
            .lll(Ratio::new(3, 4)),
        "symbolic entries",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// `QMatrix::{set, get_mut, as_slice}`
// ═══════════════════════════════════════════════════════════════════════════

/// SymPy: `Matrix([[2,1,0],[1,3,1],[0,1,4]]).det()` = 18; with `A[1,1] =
/// 7/2` the determinant is 22.
#[test]
fn qmatrix_set_changes_the_determinant_exactly() {
    let mut a = QMatrix::from_i64(&[&[2, 1, 0], &[1, 3, 1], &[0, 1, 4]]).unwrap();
    assert_eq!(a.det().unwrap(), qi(18));
    a.set(1, 1, q(7, 2));
    assert_eq!(*a.get(1, 1), q(7, 2));
    assert_eq!(a.det().unwrap(), qi(22));
}

/// `get_mut` writes through to the row-major buffer exposed by `as_slice`;
/// SymPy: with `A[0,2] = -5/3` the determinant is `49/3`.
#[test]
fn qmatrix_get_mut_writes_through_to_as_slice() {
    let mut a = QMatrix::from_i64(&[&[2, 1, 0], &[1, 3, 1], &[0, 1, 4]]).unwrap();
    *a.get_mut(0, 2) = q(-5, 3);
    assert_eq!(a.det().unwrap(), q(49, 3));
    let flat = a.as_slice();
    assert_eq!(flat.len(), 9);
    assert_eq!(flat[2], q(-5, 3));
    assert_eq!(
        flat,
        &[
            qi(2),
            qi(1),
            q(-5, 3),
            qi(1),
            qi(3),
            qi(1),
            qi(0),
            qi(1),
            qi(4)
        ]
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// `certificates::Outcome`
// ═══════════════════════════════════════════════════════════════════════════

/// `x² + 1 ≥ 0` for `x ≥ 0`: the shifted coefficients `[1, 0, 1]` are all
/// non-negative, so the Pólya power is 0.  Every accessor of a `Proved`
/// outcome answers accordingly and `Display` is `proved: <certificate>`.
#[test]
fn outcome_proved_exposes_the_certificate_and_nothing_else() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let out =
        prove_nonnegative_on_halfline(&(x.powi(2) + 1), &x, &ctx.int(0), Ray::AtLeast, 10).unwrap();
    assert!(out.is_proved());
    assert!(!out.is_refuted());
    assert!(!out.is_unknown());
    let cert = out.certificate().expect("proved");
    assert!(cert.verify());
    assert_eq!(cert.polya_power(), 0);
    assert!(out.refutation().is_none());
    assert!(out.unknown().is_none());
    assert_eq!(format!("{out}"), format!("proved: {cert}"));
    assert_eq!(format!("{out}"), "proved: x^2 + 1 = x^2 + 1, x ≥ 0");
}

/// `x − 1 ≥ 0` for `x ≥ 0` is false at the endpoint: refuted at `x = 0`
/// with value `−1`, and `Display` names the point.
#[test]
fn outcome_refuted_exposes_the_counterexample() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let out = prove_nonnegative_on_halfline(&(&x - 1), &x, &ctx.int(0), Ray::AtLeast, 10).unwrap();
    assert!(out.is_refuted());
    assert!(!out.is_proved());
    assert!(!out.is_unknown());
    let (point, value) = out.refutation().expect("refuted");
    assert_eq!(point.len(), 1);
    assert_eq!(point[0].0, x);
    assert_eq!(point[0].1, qi(0));
    assert_eq!(*value, qi(-1));
    assert!(out.certificate().is_none());
    assert!(out.unknown().is_none());
    assert_eq!(format!("{out}"), "refuted at (x = 0): value -1");
}

/// `x² − x + 1 > 0` everywhere (discriminant −3), but its shifted
/// coefficients `[1, −1, 1]` need a Pólya multiplier; with the budget
/// `max_polya_power = 0` the search stops and reports `Unknown` carrying
/// the budget it used.
#[test]
fn outcome_unknown_exposes_the_search_account() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let out =
        prove_nonnegative_on_halfline(&(x.powi(2) - &x + 1), &x, &ctx.int(0), Ray::AtLeast, 0)
            .unwrap();
    assert!(out.is_unknown());
    assert!(!out.is_proved());
    assert!(!out.is_refuted());
    assert_eq!(out.unknown().expect("unknown").max_polya_power, 0);
    assert!(out.certificate().is_none());
    assert!(out.refutation().is_none());
    assert_eq!(
        format!("{out}"),
        "unknown: non-negative by Sturm's theorem, but no certificate up to Pólya power 0"
    );
    // The same goal with a real budget is proved (Pólya's theorem applies
    // to a strictly positive polynomial).
    let proved =
        prove_nonnegative_on_halfline(&(x.powi(2) - &x + 1), &x, &ctx.int(0), Ray::AtLeast, 10)
            .unwrap();
    assert!(proved.is_proved(), "{proved}");
}

/// `map_certificate` rewrites the `Proved` payload and passes `Refuted`
/// (point, value, parameter) and `Unknown` through unchanged.
#[test]
fn outcome_map_certificate_touches_only_the_proved_variant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let goal = x.powi(2) + 1;
    let proved = prove_nonnegative_on_halfline(&goal, &x, &ctx.int(0), Ray::AtLeast, 10).unwrap();
    assert_eq!(
        proved.map_certificate(|c| c.polya_power()),
        Outcome::Proved(0u32)
    );

    let refuted =
        prove_nonnegative_on_halfline(&(&x - 1), &x, &ctx.int(0), Ray::AtLeast, 10).unwrap();
    match refuted.map_certificate(|c| c.polya_power()) {
        Outcome::Refuted { point, value, .. } => {
            assert_eq!(point, vec![(x.clone(), qi(0))]);
            assert_eq!(value, qi(-1));
        }
        other => panic!("refuted must stay refuted: {other}"),
    }

    let unknown =
        prove_nonnegative_on_halfline(&(x.powi(2) - &x + 1), &x, &ctx.int(0), Ray::AtLeast, 0)
            .unwrap();
    let account = unknown.unknown().expect("unknown").clone();
    assert_eq!(
        unknown.map_certificate(|c| c.polya_power()),
        Outcome::Unknown(account)
    );
}

/// `into_certificate` moves the certificate out of `Proved` and is `None`
/// for the other two variants.
#[test]
fn outcome_into_certificate_moves_the_certificate_out() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let proved =
        prove_nonnegative_on_halfline(&(x.powi(2) + 1), &x, &ctx.int(0), Ray::AtLeast, 10).unwrap();
    let cert = proved.into_certificate().expect("proved");
    assert!(cert.verify());
    assert_eq!(cert.polya_power(), 0);

    let refuted =
        prove_nonnegative_on_halfline(&(&x - 1), &x, &ctx.int(0), Ray::AtLeast, 10).unwrap();
    assert!(refuted.into_certificate().is_none());

    let unknown =
        prove_nonnegative_on_halfline(&(x.powi(2) - &x + 1), &x, &ctx.int(0), Ray::AtLeast, 0)
            .unwrap();
    assert!(unknown.into_certificate().is_none());
}

/// On `{0 ≤ r ≤ 1/2}` with parameter `j ≥ 1`, `j − 2jr − 1 ≥ 0` fails at
/// `r = 1/2, j = 1` with value `−1` (the `prove_nonnegative_on_polyhedron`
/// doctest); the `Display` of a parametric refutation appends the sampled
/// parameter.
#[test]
fn outcome_display_of_a_parametric_refutation_names_the_parameter() {
    let ctx = Context::new();
    let (j, r) = (ctx.symbol("j"), ctx.symbol("r"));
    let hyps = [r.clone(), ctx.rational(1, 2) - &r];
    let param = ParamBound {
        var: j.clone(),
        lower: ctx.int(1),
    };
    let out = prove_nonnegative_on_polyhedron(
        &(&j - &j * &r * 2 - 1),
        &hyps,
        Some(&param),
        &PolyhedronOpts::default(),
    )
    .unwrap();
    match &out {
        Outcome::Refuted {
            point,
            value,
            param_value,
            ..
        } => {
            assert_eq!(point.len(), 2);
            assert_eq!(point[0], (r.clone(), q(1, 2)));
            assert_eq!(point[1], (j.clone(), qi(1)));
            assert_eq!(*value, qi(-1));
            assert_eq!(*param_value, Some(qi(1)));
        }
        other => panic!("{other}"),
    }
    assert_eq!(
        format!("{out}"),
        "refuted at (r = 1/2, j = 1): value -1 (parameter 1)"
    );
}

/// The box prover's `Outcome`: `x − 2 ≥ 0` on `[0, 1]` is refuted at
/// `x = 0` with value `−2`; `x(1 − x) ≥ 0` there is proved.
#[test]
fn box_prover_refutes_x_minus_2_at_the_origin_and_proves_x_times_1_minus_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let bounds = [BoxBound {
        var: x.clone(),
        lo: ctx.int(0),
        hi: ctx.int(1),
    }];
    let refuted = prove_nonnegative_on_box(&(&x - 2), &bounds, 1).unwrap();
    let (point, value) = refuted.refutation().expect("refuted");
    assert_eq!(point, &[(x.clone(), qi(0))]);
    assert_eq!(*value, qi(-2));
    assert_eq!(format!("{refuted}"), "refuted at (x = 0): value -2");

    let proved = prove_nonnegative_on_box(&(&x * (1 - &x)), &bounds, 2).unwrap();
    let cert = proved.certificate().expect("proved");
    assert!(cert.verify());
    assert!(format!("{proved}").starts_with("proved: "));
    assert!(format!("{proved}").ends_with(", 0 ≤ x ≤ 1"));
}

/// Motzkin's `x⁴y² + x²y⁴ − 3x²y² + 1` is non-negative on ℝ² but not a sum
/// of squares (Motzkin 1967), so `prove_sos` must answer `Unknown` — never
/// `Proved` (unsound) nor `Refuted` (there is no negative point).
#[test]
fn sos_prover_reports_motzkin_as_unknown_with_a_reason() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let motzkin = x.powi(4) * y.powi(2) + x.powi(2) * y.powi(4) - 3 * x.powi(2) * y.powi(2) + 1;
    let out = prove_sos(&motzkin, &[x, y], &SosOpts::default()).unwrap();
    assert!(out.is_unknown(), "{out}");
    let account = out.unknown().expect("unknown");
    assert!(!account.reason.is_empty());
    assert_eq!(account.budget_exhausted, None);
    assert_eq!(format!("{out}"), format!("unknown: {}", account.reason));
    assert!(out.certificate().is_none());
    assert!(out.refutation().is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// `calculus/limit.rs` — validation and the discontinuity checks of the
// safe-substitution step
// ═══════════════════════════════════════════════════════════════════════════

/// A constant that is not a legitimate limit value (`NaN`, `zoo`) is an
/// error, not a value (the module contract: a limit is `±∞` or a finite
/// constant; SymPy returns `nan` for `limit(nan, x, 0)`).
#[test]
fn limit_of_a_nan_or_zoo_constant_is_an_error() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    err_is_computation_failed(&ctx.nan().try_limit(&x, &ctx.int(0)), "limit of nan");
    err_is_computation_failed(
        &ctx.complex_infinity().try_limit(&x, &ctx.int(0)),
        "limit of zoo",
    );
    assert!(ctx.nan().limit(&x, &ctx.int(0)).has_unevaluated());
}

/// The limit point must be finite or `±∞`.
#[test]
fn limit_at_a_nan_or_zoo_point_is_an_error() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    err_is_computation_failed(&x.try_limit(&x, &ctx.nan()), "point nan");
    err_is_computation_failed(&x.try_limit(&x, &ctx.complex_infinity()), "point zoo");
    let e = x.try_limit(&x, &ctx.nan()).unwrap_err().to_string();
    assert!(e.contains("finite or ±∞"), "{e}");
}

/// SymPy: `limit(atanh(x), x, 1, '-')` = `oo`, `limit(atanh(x), x, -1,
/// '+')` = `-oo`.
#[test]
fn one_sided_limits_of_atanh_at_its_poles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.atanh().limit_left(&x, &ctx.int(1)), ctx.infinity());
    assert_eq!(x.atanh().limit_right(&x, &ctx.int(-1)), ctx.neg_infinity());
}

/// SymPy: `limit(tan(x), x, pi/2, '-')` = `oo`, `'+'` = `-oo`.
#[test]
fn one_sided_limits_of_tan_at_pi_over_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half_pi = ctx.pi() / 2;
    assert_eq!(x.tan().limit_left(&x, &half_pi), ctx.infinity());
    assert_eq!(
        x.tan().limit_dir(&x, &half_pi, Direction::Right),
        ctx.neg_infinity()
    );
}

/// SymPy: `limit(floor(x), x, 2, '+')` = 2, `'-'` = 1;
/// `limit(ceiling(x), x, 2, '+')` = 3.
#[test]
fn one_sided_limits_of_floor_and_ceiling_at_an_integer() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.floor().limit_right(&x, &ctx.int(2)), ctx.int(2));
    assert_eq!(x.floor().limit_left(&x, &ctx.int(2)), ctx.int(1));
    assert_eq!(x.ceiling().limit_right(&x, &ctx.int(2)), ctx.int(3));
    // The two-sided limit does not exist.
    assert!(x.floor().try_limit(&x, &ctx.int(2)).is_err());
}

/// SymPy: `limit(gamma(x), x, 0, '+')` = `oo`.  `ψ(x) = −1/x − γ + O(x)`
/// (DLMF 5.7.6), so `ψ(x) → −∞` as `x → 0⁺` (SymPy's
/// `limit(polygamma(0, x), x, 0, '+')` answers the undirected `zoo`).
#[test]
fn one_sided_limits_of_gamma_and_digamma_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.gamma().limit_right(&x, &ctx.int(0)), ctx.infinity());
    assert_eq!(x.digamma().limit_right(&x, &ctx.int(0)), ctx.neg_infinity());
    // Γ(x + 1) → +∞ as x → −1⁺ (the pole of Γ at 0 from the right).
    assert_eq!(
        (&x + 1).gamma().limit_right(&x, &ctx.int(-1)),
        ctx.infinity()
    );
}

/// SymPy: `limit((1/x)**(1/2), x, 0, '+')` = `oo`, `limit((-1/x)**3, …)`
/// = `-oo`, `limit((-1/x)**2, …)` = `oo`, `limit((1/x)**(-1/2), …)` = 0.
#[test]
fn right_limits_of_powers_of_one_over_x_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    assert_eq!(
        (1 / &x).pow(&ctx.rational(1, 2)).limit_right(&x, &zero),
        ctx.infinity()
    );
    assert_eq!((-1 / &x).powi(3).limit_right(&x, &zero), ctx.neg_infinity());
    assert_eq!((-1 / &x).powi(2).limit_right(&x, &zero), ctx.infinity());
    assert_eq!(
        (1 / &x).pow(&ctx.rational(-1, 2)).limit_right(&x, &zero),
        ctx.int(0)
    );
}

/// SymPy: `limit(2**(1/x), x, 0, '+')` = `oo`, `limit(2**(-1/x), x, 0,
/// '+')` = 0, `limit(3 - 1/x, x, 0, '+')` = `-oo`, `limit(log(x), x, 0,
/// '+')` = `-oo`.
#[test]
fn right_limits_of_exponentials_and_shifts_of_one_over_x_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    assert_eq!(
        ctx.int(2).pow(&(1 / &x)).limit_right(&x, &zero),
        ctx.infinity()
    );
    assert_eq!(
        ctx.int(2).pow(&(-1 / &x)).limit_right(&x, &zero),
        ctx.int(0)
    );
    assert_eq!((3 - 1 / &x).limit_right(&x, &zero), ctx.neg_infinity());
    assert_eq!(x.ln().limit_right(&x, &zero), ctx.neg_infinity());
}

/// SymPy: `limit(x**x, x, 0, '+')` = 1.
#[test]
fn right_limit_of_x_to_the_x_at_zero_is_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.pow(&x).limit_right(&x, &ctx.int(0)), ctx.int(1));
}

/// SymPy: `limit(sign(x), x, 0, '+')` = 1 and `'-'` = −1, so the
/// two-sided limit does not exist; the error names both sides.
/// `limit(Heaviside(x), x, 0, '+')` = 1.
#[test]
fn two_sided_limit_of_sign_at_zero_reports_the_differing_sides() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    assert_eq!(x.sign().limit_right(&x, &zero), ctx.int(1));
    assert_eq!(x.sign().limit_left(&x, &zero), ctx.int(-1));
    let e = x.sign().try_limit(&x, &zero).unwrap_err().to_string();
    assert!(e.contains("left = -1, right = 1"), "{e}");
    assert_eq!(x.heaviside().limit_right(&x, &zero), ctx.int(1));
}

/// `x!` is continuous at `−1/2` (no pole: the `Factorial` continuity check
/// only rejects negative integers), so the limit is the value `(−1/2)!`
/// by direct substitution; SymPy: `limit(factorial(x), x, -1/2)` =
/// `factorial(-1/2)`.
#[test]
fn limit_of_factorial_at_minus_one_half_is_its_value_there() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let l = x.factorial().limit(&x, &ctx.rational(-1, 2));
    assert!(!l.has_unevaluated(), "{l}");
    assert_eq!(l, ctx.rational(-1, 2).factorial());
}

/// `(−1/2)! = Γ(1/2) = √π`; SymPy: `N(factorial(-1/2), 20)` =
/// `1.7724538509055160273`.  (0.22 refused: "cannot numerically evaluate
/// symbolic factorial"; since 0.22.1 evalf computes `Γ(x + 1)`.)
#[test]
fn factorial_of_minus_one_half_evaluates_to_sqrt_pi() {
    let ctx = Context::new();
    let f = ctx.rational(-1, 2).factorial();
    close(
        f.eval_f64().unwrap(),
        1.772453850905516,
        "(-1/2)! = sqrt(pi)",
    );
}

/// `x! = Γ(x + 1)` at its poles, one-sided.  SymPy 1.14 leaves
/// `limit(factorial(x), x, -1, '+')` unevaluated, so the references are the
/// Gamma limits it does compute: `limit(gamma(t), t, 0, '+') = oo`,
/// `(t, 0, '-') = -oo`, `(t, -1, '+') = -oo`, `(t, -1, '-') = oo`.  Until
/// 0.22 the `Factorial` node came back as an unevaluated `Limit`; since
/// 0.22.1 gruntz rewrites it to `Γ(x + 1)`.
#[test]
fn right_limit_of_factorial_at_minus_one_is_infinite() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.factorial();
    assert_eq!(f.limit_right(&x, &ctx.int(-1)), ctx.infinity());
    assert_eq!(f.limit_left(&x, &ctx.int(-1)), ctx.neg_infinity());
    assert_eq!(f.limit_right(&x, &ctx.int(-2)), ctx.neg_infinity());
    assert_eq!(f.limit_left(&x, &ctx.int(-2)), ctx.infinity());
}

// ═══════════════════════════════════════════════════════════════════════════
// LaTeX — node kinds no test had printed (`output/latex.rs`)
// ═══════════════════════════════════════════════════════════════════════════

/// `\tanh` is a standard LaTeX macro; there is no `\asinh` etc., so the
/// inverse hyperbolics go through `\operatorname{}`.
#[test]
fn latex_of_tanh_and_the_inverse_hyperbolics() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.tanh().to_latex(), r"\tanh\left(x\right)");
    assert_eq!(x.asinh().to_latex(), r"\operatorname{asinh}\left(x\right)");
    assert_eq!(x.acosh().to_latex(), r"\operatorname{acosh}\left(x\right)");
    assert_eq!(x.atanh().to_latex(), r"\operatorname{atanh}\left(x\right)");
}

/// `sgn`, `H`, `erf`, `erfc`, `W` are `\operatorname{}`s; `δ`, `Γ`,
/// `ln Γ`, `ψ` are the Greek macros.
#[test]
fn latex_of_sign_heaviside_dirac_gamma_family_and_error_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.sign().to_latex(), r"\operatorname{sgn}\left(x\right)");
    assert_eq!(x.heaviside().to_latex(), r"\operatorname{H}\left(x\right)");
    assert_eq!(x.dirac_delta().to_latex(), r"\delta\left(x\right)");
    assert_eq!(x.gamma().to_latex(), r"\Gamma\left(x\right)");
    assert_eq!(x.log_gamma().to_latex(), r"\ln \Gamma\left(x\right)");
    assert_eq!(x.digamma().to_latex(), r"\psi\left(x\right)");
    assert_eq!(x.erf().to_latex(), r"\operatorname{erf}\left(x\right)");
    assert_eq!(x.erfc().to_latex(), r"\operatorname{erfc}\left(x\right)");
    assert_eq!(x.lambertw().to_latex(), r"\operatorname{W}\left(x\right)");
}

/// `\lfloor x\rfloor` / `\lceil x\rceil` (amsmath delimiters).
#[test]
fn latex_of_floor_and_ceiling() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.floor().to_latex(), r"\lfloor x\rfloor");
    assert_eq!(x.ceiling().to_latex(), r"\lceil x\rceil");
}

/// An atom gets a bare `!`; a compound argument is parenthesised so that
/// `(x + 1)!` is not read as `x + 1!`.
#[test]
fn latex_of_factorial_parenthesises_a_compound_argument() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    assert_eq!(n.factorial().to_latex(), "n!");
    assert_eq!((&n + 1).factorial().to_latex(), r"\left(n + 1\right)!");
}

/// `\binom{n}{k}` (amsmath), `\mathrm{B}(a, b)` for the beta function,
/// `\operatorname{atan2}(y, x)`.
#[test]
fn latex_of_binomial_beta_and_atan2() {
    let ctx = Context::new();
    let (x, y, n) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("n"));
    assert_eq!(n.binomial(&ctx.int(2)).to_latex(), r"\binom{n}{2}");
    assert_eq!(x.beta(&y).to_latex(), r"\mathrm{B}\left(x, y\right)");
    assert_eq!(
        y.atan2(&x).to_latex(),
        r"\operatorname{atan2}\left(y, x\right)"
    );
}

/// `\min` / `\max` are LaTeX operators; the n-ary constructors keep the
/// argument order.
#[test]
fn latex_of_min_and_max_keep_argument_order() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    assert_eq!(
        Ex::min_of(&ctx, [x.clone(), y.clone()]).to_latex(),
        r"\min\left(x, y\right)"
    );
    assert_eq!(
        Ex::max_of(&ctx, [x.clone(), y.clone(), ctx.int(1)]).to_latex(),
        r"\max\left(x, y, 1\right)"
    );
    assert_eq!(x.min_with(&y).to_latex(), r"\min\left(x, y\right)");
}

/// `\lim_{x \to 0} \frac{1}{x}` for a limit that does not exist (kept as a
/// formal node) and `\operatorname{Res}_{x=0} …` for a residue the engine
/// leaves formal.
#[test]
fn latex_of_formal_limit_and_residue_nodes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lim = (1 / &x).limit(&x, &ctx.int(0));
    assert!(lim.has_unevaluated());
    assert_eq!(lim.to_latex(), r"\lim_{x \to 0} \frac{1}{x}");
    let res = (1 / &x).exp().residue(&x, &ctx.int(0));
    assert!(res.has_unevaluated());
    assert_eq!(
        res.to_latex(),
        r"\operatorname{Res}_{x=0} \exp\left(\frac{1}{x}\right)"
    );
}

/// A Laplace transform with no table entry is kept as
/// `\mathcal{L}\left\{f\right\}`; an integral with no closed form as
/// `\int f \, dx`.
#[test]
fn latex_of_formal_laplace_transform_and_integral_nodes() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let lap = t.tan().exp().laplace(&t, &s);
    assert!(lap.has_unevaluated());
    assert_eq!(
        lap.to_latex(),
        r"\mathcal{L}\left\{\exp\left(\tan\left(t\right)\right)\right\}"
    );
    let int = (t.exp() * t.sin() * t.ln()).integrate(&t);
    assert!(int.has_unevaluated());
    assert_eq!(
        int.to_latex(),
        r"\int \sin\left(t\right) \exp\left(t\right) \ln\left(t\right)\, dt"
    );
}

/// `zoo` is `\tilde{\infty}` (SymPy's LaTeX for `zoo`), `nan` is
/// `\text{NaN}`.
#[test]
fn latex_of_complex_infinity_and_nan() {
    let ctx = Context::new();
    assert_eq!(ctx.complex_infinity().to_latex(), r"\tilde{\infty}");
    assert_eq!(ctx.nan().to_latex(), r"\text{NaN}");
}

/// `\sum_{n=1}^{\infty} \frac{1}{n^{2}}` (0.21 emitted `\sum_{n=1^{\infty} …`,
/// an unbalanced brace; fixed in 0.22).
#[test]
fn latex_of_sum_closes_the_subscript_brace() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let sum = Ex::symbolic_sum(&(1 / n.powi(2)), &n, &ctx.int(1), &ctx.infinity());
    assert_eq!(sum.to_latex(), r"\sum_{n=1}^{\infty} \frac{1}{n^{2}}");
}

/// Same construction for `\prod`.
#[test]
fn latex_of_product_closes_the_subscript_brace() {
    let ctx = Context::new();
    let (k, n) = (ctx.symbol("k"), ctx.symbol("n"));
    let prod = Ex::symbolic_product(&k, &k, &ctx.int(1), &n);
    assert_eq!(prod.to_latex(), r"\prod_{k=1}^{n} k");
}

// ═══════════════════════════════════════════════════════════════════════════
// Presentation MathML — the same node kinds (`output/mathml.rs`)
// ═══════════════════════════════════════════════════════════════════════════

const MATHML_OPEN: &str = "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">";

/// The `<math>` body of `e`, asserting the wrapper.
fn mathml_body(e: &Ex) -> String {
    let xml = e.to_mathml().unwrap_or_else(|err| panic!("{e}: {err:?}"));
    assert!(
        xml.starts_with(MATHML_OPEN) && xml.ends_with("</math>"),
        "{xml}"
    );
    xml[MATHML_OPEN.len()..xml.len() - "</math>".len()].to_string()
}

/// `<mi>f</mi><mo>&#x2061;</mo><mrow><mo>(</mo>…<mo>)</mo></mrow>` — a
/// function application with the invisible `ApplyFunction` (U+2061).
fn mathml_apply(head: &str, arg: &str) -> String {
    format!("<mrow>{head}<mo>&#x2061;</mo><mrow><mo>(</mo>{arg}<mo>)</mo></mrow></mrow>")
}

/// Named functions use `<mi>name</mi>` plus `ApplyFunction`.
#[test]
fn mathml_of_tanh_the_inverse_hyperbolics_and_the_named_special_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (e, name) in [
        (x.tanh(), "tanh"),
        (x.asinh(), "asinh"),
        (x.acosh(), "acosh"),
        (x.atanh(), "atanh"),
        (x.sign(), "sgn"),
        (x.heaviside(), "H"),
        (x.erf(), "erf"),
        (x.erfc(), "erfc"),
        (x.lambertw(), "W"),
        (x.arg(), "arg"),
        (x.si(), "Si"),
        (x.ci(), "Ci"),
        (x.ei(), "Ei"),
        (x.li(), "li"),
    ] {
        assert_eq!(
            mathml_body(&e),
            mathml_apply(&format!("<mi>{name}</mi>"), "<mi>x</mi>"),
            "{e}"
        );
    }
}

/// Greek-letter heads are numeric character references: `δ` U+03B4, `Γ`
/// U+0393, `ψ` U+03C8, `ζ` U+03B6; `ln Γ` is an `<mrow>` head.
#[test]
fn mathml_of_dirac_gamma_loggamma_digamma_and_zeta_use_greek_character_references() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        mathml_body(&x.dirac_delta()),
        mathml_apply("<mi>&#x3B4;</mi>", "<mi>x</mi>")
    );
    assert_eq!(
        mathml_body(&x.gamma()),
        mathml_apply("<mi>&#x393;</mi>", "<mi>x</mi>")
    );
    assert_eq!(
        mathml_body(&x.log_gamma()),
        mathml_apply("<mrow><mi>ln</mi><mi>&#x393;</mi></mrow>", "<mi>x</mi>")
    );
    assert_eq!(
        mathml_body(&x.digamma()),
        mathml_apply("<mi>&#x3C8;</mi>", "<mi>x</mi>")
    );
    assert_eq!(
        mathml_body(&x.zeta()),
        mathml_apply("<mi>&#x3B6;</mi>", "<mi>x</mi>")
    );
}

/// `ℜ` U+211C and `ℑ` U+2111 as function heads; the conjugate is an
/// `<mover>` with the macron U+00AF.
#[test]
fn mathml_of_re_im_and_conjugate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        mathml_body(&x.re()),
        mathml_apply("<mi>&#x211C;</mi>", "<mi>x</mi>")
    );
    assert_eq!(
        mathml_body(&x.im()),
        mathml_apply("<mi>&#x2111;</mi>", "<mi>x</mi>")
    );
    assert_eq!(
        mathml_body(&x.conjugate()),
        "<mover><mi>x</mi><mo>&#xAF;</mo></mover>"
    );
}

/// `⌊ ⌋` U+230A/U+230B and `⌈ ⌉` U+2308/U+2309 as `<mo>` fences.
#[test]
fn mathml_of_floor_and_ceiling_use_the_bracket_characters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        mathml_body(&x.floor()),
        "<mrow><mo>&#x230A;</mo><mi>x</mi><mo>&#x230B;</mo></mrow>"
    );
    assert_eq!(
        mathml_body(&x.ceiling()),
        "<mrow><mo>&#x2308;</mo><mi>x</mi><mo>&#x2309;</mo></mrow>"
    );
}

/// Postfix `<mo>!</mo>`, with a compound argument parenthesised.
#[test]
fn mathml_of_factorial_parenthesises_a_compound_argument() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    assert_eq!(
        mathml_body(&n.factorial()),
        "<mrow><mi>n</mi><mo>!</mo></mrow>"
    );
    assert_eq!(
        mathml_body(&(&n + 1).factorial()),
        "<mrow><mrow><mo>(</mo><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow><mo>!</mo></mrow>"
    );
}

/// A binomial coefficient is a zero-thickness `<mfrac>` in parentheses
/// (the MathML idiom for `\binom`); `B` and `atan2` are two-argument
/// applications.
#[test]
fn mathml_of_binomial_beta_and_atan2() {
    let ctx = Context::new();
    let (x, y, n) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("n"));
    assert_eq!(
        mathml_body(&n.binomial(&ctx.int(2))),
        "<mrow><mo>(</mo><mfrac linethickness=\"0\"><mi>n</mi><mn>2</mn></mfrac><mo>)</mo></mrow>"
    );
    assert_eq!(
        mathml_body(&x.beta(&y)),
        mathml_apply("<mi>B</mi>", "<mi>x</mi><mo>,</mo><mi>y</mi>")
    );
    assert_eq!(
        mathml_body(&y.atan2(&x)),
        mathml_apply("<mi>atan2</mi>", "<mi>y</mi><mo>,</mo><mi>x</mi>")
    );
}

/// `min` / `max` as n-ary applications; `ψ^{(2)}` as an `<msup>` head;
/// `δ_{xy}` as an `<msub>`.
#[test]
fn mathml_of_min_max_polygamma_and_kronecker_delta() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    assert_eq!(
        mathml_body(&Ex::min_of(&ctx, [x.clone(), y.clone()])),
        mathml_apply("<mi>min</mi>", "<mi>x</mi><mo>,</mo><mi>y</mi>")
    );
    assert_eq!(
        mathml_body(&Ex::max_of(&ctx, [x.clone(), y.clone(), ctx.int(1)])),
        mathml_apply(
            "<mi>max</mi>",
            "<mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mn>1</mn>"
        )
    );
    assert_eq!(
        mathml_body(&x.polygamma(&ctx.int(2))),
        mathml_apply(
            "<msup><mi>&#x3C8;</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup>",
            "<mi>x</mi>"
        )
    );
    assert_eq!(
        mathml_body(&x.kronecker_delta(&y)),
        "<msub><mi>&#x3B4;</mi><mrow><mi>x</mi><mi>y</mi></mrow></msub>"
    );
}

/// `∑` U+2211 / `∏` U+220F with `<munderover>` limits.
#[test]
fn mathml_of_sum_and_product_use_munderover() {
    let ctx = Context::new();
    let (k, n) = (ctx.symbol("k"), ctx.symbol("n"));
    let sum = Ex::symbolic_sum(&(1 / n.powi(2)), &n, &ctx.int(1), &ctx.infinity());
    assert_eq!(
        mathml_body(&sum),
        "<mrow><munderover><mo>&#x2211;</mo><mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow><mi>&#x221E;</mi></munderover><mfrac><mn>1</mn><msup><mi>n</mi><mn>2</mn></msup></mfrac></mrow>"
    );
    let prod = Ex::symbolic_product(&k, &k, &ctx.int(1), &n);
    assert_eq!(
        mathml_body(&prod),
        "<mrow><munderover><mo>&#x220F;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>k</mi></mrow>"
    );
}

/// `lim` with `<munder>` and `→` U+2192; `Res` with `<munder>` and `=`.
#[test]
fn mathml_of_formal_limit_and_residue_nodes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lim = (1 / &x).limit(&x, &ctx.int(0));
    assert_eq!(
        mathml_body(&lim),
        "<mrow><munder><mo>lim</mo><mrow><mi>x</mi><mo>&#x2192;</mo><mn>0</mn></mrow></munder><mfrac><mn>1</mn><mi>x</mi></mfrac></mrow>"
    );
    let res = (1 / &x).exp().residue(&x, &ctx.int(0));
    assert_eq!(
        mathml_body(&res),
        format!(
            "<mrow><munder><mi>Res</mi><mrow><mi>x</mi><mo>=</mo><mn>0</mn></mrow></munder>{}</mrow>",
            mathml_apply("<mi>exp</mi>", "<mfrac><mn>1</mn><mi>x</mi></mfrac>")
        )
    );
}

/// A formal Laplace transform is a script `L` applied to `{f}`; a formal
/// integral is `∫` U+222B followed by the body and `d x`.
#[test]
fn mathml_of_formal_laplace_transform_and_integral_nodes() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let lap = t.tan().exp().laplace(&t, &s);
    let body = mathml_apply("<mi>exp</mi>", &mathml_apply("<mi>tan</mi>", "<mi>t</mi>"));
    assert_eq!(
        mathml_body(&lap),
        format!(
            "<mrow><mi mathvariant=\"script\">L</mi><mo>&#x2061;</mo><mrow><mo>{{</mo>{body}<mo>}}</mo></mrow></mrow>"
        )
    );
    let int = t.tan().exp().integrate(&t);
    assert!(int.has_unevaluated());
    assert_eq!(
        mathml_body(&int),
        format!("<mrow><mo>&#x222B;</mo>{body}<mi>d</mi><mi>t</mi></mrow>")
    );
}

/// `zoo` is `∞` with a tilde over it; `nan` is `<mtext>`.
#[test]
fn mathml_of_complex_infinity_and_nan() {
    let ctx = Context::new();
    assert_eq!(
        mathml_body(&ctx.complex_infinity()),
        "<mover><mi>&#x221E;</mi><mo>~</mo></mover>"
    );
    assert_eq!(mathml_body(&ctx.nan()), "<mtext>NaN</mtext>");
}

// ═══════════════════════════════════════════════════════════════════════════
// Arena compaction (`base/compact.rs`) and the `ExprTree` round trip
// (`output/tree.rs`) over the same node kinds
// ═══════════════════════════════════════════════════════════════════════════

/// One expression per node kind that the coverage run found untransferred
/// / unserialised, all in one context.
fn one_of_each_node_kind(ctx: &Context) -> Vec<Ex> {
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let n = ctx.symbol("n");
    let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    vec![
        x.tanh(),
        x.asinh(),
        x.acosh(),
        x.atanh(),
        x.sign(),
        x.heaviside(),
        x.dirac_delta(),
        x.gamma(),
        x.log_gamma(),
        x.digamma(),
        x.erf(),
        x.erfc(),
        x.lambertw(),
        x.floor(),
        x.ceiling(),
        x.beta(&y),
        y.atan2(&x),
        x.polygamma(&ctx.int(2)),
        x.kronecker_delta(&y),
        x.zeta(),
        x.conjugate(),
        x.re(),
        x.im(),
        x.arg(),
        x.si(),
        x.ci(),
        x.ei(),
        x.li(),
        Ex::min_of(ctx, [x.clone(), y.clone()]),
        Ex::max_of(ctx, [x.clone(), y.clone(), ctx.int(1)]),
        Ex::symbolic_sum(&(1 / n.powi(2)), &n, &ctx.int(1), &ctx.infinity()),
        Ex::symbolic_product(&n, &n, &ctx.int(1), &ctx.int(5)),
        (1 / &x).limit(&x, &ctx.int(0)),
        (1 / &x).exp().residue(&x, &ctx.int(0)),
        x.tan().exp().laplace(&x, &y),
        x.tan().exp().integrate(&x),
        Ex::piecewise(&[(&x, &x.gt(&ctx.int(0))), (&(-&x), &ctx.bool_true())]),
        ctx.physical_constant("c", ctx.int(299_792_458)),
        ctx.complex_infinity(),
        ctx.nan(),
        ctx.e(),
        ctx.euler_gamma(),
        ctx.catalan(),
        ctx.golden_ratio(),
        &p * &x,
    ]
}

/// `Context::compact` copies every node kind into a fresh arena: the copies
/// print identically, a numeric composite still evaluates, and a symbol's
/// assumptions survive the transfer.
#[test]
fn compact_transfers_every_node_kind_and_keeps_assumptions() {
    let ctx = Context::new();
    let mut roots = one_of_each_node_kind(&ctx);
    let x = ctx.symbol("x");
    roots.push((&x + 1).factorial());
    roots.push(x.binomial(&ctx.int(2)));
    // A numeric composite through the same node kinds:
    // tanh(1/2) + floor(7/3) + atan2(1, 2) + Γ(1/2) + max(1, 3).
    let numeric = ctx.rational(1, 2).tanh()
        + ctx.rational(7, 3).floor()
        + ctx.int(1).atan2(&ctx.int(2))
        + ctx.rational(1, 2).gamma()
        + Ex::max_of(&ctx, [ctx.int(1), ctx.int(3)]);
    let before = numeric.eval_f64().unwrap();
    roots.push(numeric);
    let p_index = roots.len() - 4;
    assert_eq!(roots[p_index].to_string(), "p*x");

    let (new_ctx, copies) = ctx.compact(&roots);
    assert_eq!(copies.len(), roots.len());
    for (old, new) in roots.iter().zip(&copies) {
        assert_eq!(old.to_string(), new.to_string());
        assert_eq!(old.to_latex(), new.to_latex());
        assert_eq!(old.to_tree(), new.to_tree(), "{old}");
    }
    // tanh(1/2) + 2 + atan(1/2) + sqrt(pi) + 3: mpmath 1.3 at 30 digits
    // `tanh(1/2) + 2 + atan2(1, 2) + gamma(1/2) + 3` =
    // 7.69821861716633190201474219845 (scipy 1.18: 7.698218617166331);
    // the compacted copy evaluates to the same f64.
    let after = copies.last().unwrap().eval_f64().unwrap();
    assert_eq!(after, before);
    close(after, 7.698218617166332, "numeric composite");
    // The positivity assumption on `p` travelled with the symbol.
    let new_p = new_ctx.symbol("p");
    assert_eq!(new_p.is_positive(), Some(true));
    assert_eq!(copies[p_index].is_positive(), None);
    // The new arena holds only what is reachable.
    assert!(new_ctx.node_count() < ctx.node_count());
}

/// `Context::from_tree(e.to_tree())` is the identity (same arena node) for
/// every node kind, including the formal calculus nodes and `Piecewise`.
#[test]
fn tree_round_trip_is_the_identity_for_every_node_kind() {
    let ctx = Context::new();
    for e in one_of_each_node_kind(&ctx) {
        let tree = e.to_tree();
        let back = ctx.from_tree(&tree);
        assert_eq!(back, e, "{e}: tree {tree:?}");
        // And through JSON.
        let json = e.to_json().unwrap();
        assert_eq!(ctx.from_json(&json).unwrap(), e, "{e}");
    }
}

/// `to_tree` serialises `Factorial` / `Binomial` as `Apply { name:
/// "factorial" | "binomial" }`; until 0.22 `from_tree` turned every `Apply`
/// into an opaque user function, so the round trip lost the built-in (it
/// printed as `factorial(n + 1)` and no longer evaluated).  Fixed in 0.22.1.
#[test]
fn tree_round_trip_preserves_factorial_and_binomial() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let fact = (&n + 1).factorial();
    let back = ctx.from_tree(&fact.to_tree());
    assert_eq!(back, fact, "{back}");
    assert_eq!(back.subs(&n, &ctx.int(4)).eval(), ctx.int(120));
    let binom = n.binomial(&ctx.int(2));
    let back = ctx.from_tree(&binom.to_tree());
    assert_eq!(back, binom, "{back}");
    assert_eq!(back.subs(&n, &ctx.int(5)).eval(), ctx.int(10));
}

/// `liveness_ratio` is the reachable fraction of the arena (a proper
/// fraction once dead intermediates exist) and `should_compact` is `false`
/// for an arena far below its 100 K-node threshold.
#[test]
fn liveness_ratio_and_should_compact_on_a_small_arena() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Dead intermediates: the expansion is dropped, only the factor is kept.
    let _ = (&x + 1).powi(6).expand();
    let root = &x + 1;
    let ratio = ctx.liveness_ratio(std::slice::from_ref(&root));
    assert!(ratio > 0.0 && ratio < 1.0, "liveness = {ratio}");
    assert!(!ctx.should_compact(std::slice::from_ref(&root)));
    assert!(ctx.node_count() < 100_000);
    let (new_ctx, new_roots) = ctx.compact(std::slice::from_ref(&root));
    let new_ratio = new_ctx.liveness_ratio(&new_roots);
    assert!(new_ratio >= ratio, "{new_ratio} < {ratio}");
}

/// Found by `fuzz_parser` (0.22.3): `7.4**77.4**74` overflowed the stack.
/// `(37/5)^((387/5)^74)` splits into `37^(a/b)·5^(−a/b)`, and the radical
/// normal form rewrote `5^(−a/b)` as `5^(−(k+1))·5^((b−s)/b)` — but
/// `5^(k+1)` (k ≈ 10¹⁴⁰) cannot fold to a number, so `mul` added the
/// exponents back to `−a/b` and the rewrite recursed without end.  It now
/// applies only when the integer power folds.
#[test]
fn parsing_a_large_exact_power_does_not_overflow_the_stack() {
    let handle = std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let ctx = Context::new();
            let e = symplex::parse::parse(&ctx, "7.4**77.4**74").expect("parses");
            let shown = format!("{e}");
            assert!(!shown.is_empty());
            // The same power built directly, and its two halves.
            let big = ctx.rational(387, 5).powi(74);
            let p = ctx.rational(37, 5).pow(&big);
            assert!(!format!("{p}").is_empty());
            assert!(!format!("{}", ctx.int(5).pow(&-&big)).is_empty());
        })
        .expect("spawn");
    handle.join().expect("no stack overflow");
}
