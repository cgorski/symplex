//! Tests of session 17, package WP-04b (s17c_review_findings): the behaviours that the seven
//! reviews of the 1.0 API design (`A` surface … `G` features) read in the source and did not
//! run.  Each was run; this file keeps one test per behaviour.
//!
//! * A **confirmed** defect is `#[ignore = "known bug: …"]` (or "known inconsistency", "known
//!   gap"): the test states what is right, fails today, and its doc comment says what is
//!   returned today.  The fix removes the `ignore`.  Where the 1.0 design fixes a defect by
//!   changing a signature (a parameter that becomes a `Symbol`, a function that becomes
//!   fallible), the test is written against today's API and says which rule replaces it.
//! * A behaviour that turned out **right** is pinned by a test that runs.
//!
//! Every expected value names its oracle: mpmath 1.3.0 at two precisions, SymPy 1.14, exact
//! arithmetic, or the definition.  The calls are in `oracle_tests.py`, `check_gamma.py` and
//! `check_integrals.py` of the package's scratch; their output is quoted here.  Ids (`W01` …)
//! are the rows of the package's `NOTES.md`; "review D §1.3" is the section of the review
//! report that predicted the behaviour.
//!
//! Tier: "stable" is the part of the library that 1.0 promises (`API_1.0.md`, Part 1);
//! units and control are experimental there.

#![allow(unused_imports)]
use num_bigint::BigInt;
use symplex::linprog::qi;
use symplex::ntheory;
use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::regression::{LogitOpts, logit};
use symplex::units::{AngularVelocity, Frequency};

/// A matrix from the sources of its entries.
fn mat(ctx: &Context, rows: &[&[&str]]) -> Matrix {
    Matrix::new(
        rows.iter()
            .map(|r| r.iter().map(|s| ctx.parse(s).unwrap()).collect())
            .collect(),
    )
    .unwrap()
}

/// `src` to `digits` digits.
fn dec(ctx: &Context, src: &str, digits: u32) -> String {
    ctx.parse(src).unwrap().eval_decimal(digits).unwrap()
}

// ── Review D: numbers ───────────────────────────────────────────────────────

/// W01 (review D §1.3, §7-1; stable).  `Γ(x)` is `exp(ln Γ(x))` with `ln Γ` computed at
/// `prec + 20` bits (`evalf/gamma.rs: stirling_log_gamma`).  The absolute error of `ln Γ` is
/// the relative error of `Γ`, so `Γ(x)` carries `prec + 20 − log₂ ln Γ(x)` bits, while its
/// bound claims `prec − 1` (`accuracy::finish`, two units in the last place): too small by a
/// factor `2^(log₂ ln Γ(x) − 21)`, which is 6 at `x = 10⁶`, 70 at `10⁷` and 600 at `7·10⁷`.
/// The digits of `Γ(x)` itself are right (the test below); a difference that cancels 50 to
/// 62 bits of it is printed with wrong last digits.
///
/// Today, at 32 digits (`M` is the leading bits of `Γ(x)`):
///
/// * `gamma(280000001/4) - 17854627671268857*2^1723270899` prints
///   `-8.6849403030167786554183215264717e518756229` (the last two digits; 12.3 units);
/// * `gamma(92485365/4) - 44760030115325899*2^532253827` prints `…4423732e160224366`
///   (1.04 units; the smallest argument found with more than one unit, `x ≈ 2.3·10⁷`);
/// * `gamma(21196207/4) - 261593990654089*2^110721550` prints `…699555306e33330504`
///   (0.61 units, `x ≈ 5.3·10⁶`: the wrong neighbour of the correctly rounded value).
///
/// Hunt: 396,500 such differences over 100 arguments in `[10⁵, 7.7·10⁷]`, 61 depths and the
/// digit counts 16 to 80: no error above 0.5 units below `10⁶`, 9 above 0.5625 in `[10⁶,
/// 10⁷)`, 46 above one unit from `2.3·10⁷` on.
///
/// Oracle: `gamma(mpf(280000001)/4) - 17854627671268857*mpf(2)**1723270899` with `mp.prec =
/// 700` and `1100` = `-8.68494030301677865541832152647046633477e+518756229`;
/// `gamma(mpf(92485365)/4) - 44760030115325899*mpf(2)**532253827` =
/// `-5.487362057415743048190185442373304013882e+160224366`;
/// `gamma(mpf(21196207)/4) - 261593990654089*mpf(2)**110721550` =
/// `8.792592170743907867485869955530538701435e+33330504`.
#[test]
#[ignore = "known bug: the bound of gamma(x) ignores the factor ln Γ(x) (wrong last digits under a cancellation, x ≳ 10⁶)"]
fn gamma_of_a_large_argument_keeps_its_digits_under_a_cancellation() {
    let ctx = Context::new();
    assert_eq!(
        dec(
            &ctx,
            "gamma(280000001/4) - 17854627671268857*2^1723270899",
            32
        ),
        "-8.6849403030167786554183215264705e518756229"
    );
    assert_eq!(
        dec(
            &ctx,
            "gamma(92485365/4) - 44760030115325899*2^532253827",
            32
        ),
        "-5.4873620574157430481901854423733e160224366"
    );
    assert_eq!(
        dec(&ctx, "gamma(21196207/4) - 261593990654089*2^110721550", 32),
        "8.7925921707439078674858699555305e33330504"
    );
}

/// W01, the part that is right: the digits of `Γ(x)` itself.  732 values (244 arguments in
/// `[10⁵, 7·10⁷]`, exactly representable ones included, at 16, 30 and 50 digits) have no
/// error above 0.4999 units of the last digit.
///
/// Oracle: `mpmath.gamma(mpf(40000001)/4)` and `mpmath.gamma(mpf(400001)/4)` with `mp.dps =
/// 80` and `120` (`check_gamma.py direct`).
#[test]
fn gamma_of_a_large_argument_has_its_digits() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "gamma(40000001/4)", 30),
        "6.76172362146883043963906723222e65657053"
    );
    assert_eq!(
        dec(&ctx, "gamma(400001/4)", 30),
        "5.0222642970233944983456122485e456569"
    );
}

/// W20 (review D §1.6; stable) — not a bug, pinned: an `f64` operand is converted exactly.
/// `&x * 0.1` is the double nearest to 0.1 times `x`, not `x/10`, which is what the parser
/// reads in the text `0.1`.  (The 1.0 design makes the operator a compile error, rule F.)
///
/// Oracle: `fractions.Fraction(0.1)` = `3602879701896397/36028797018963968`.
#[test]
fn an_f64_operand_is_converted_exactly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        (&x * 0.1).to_string(),
        "3602879701896397/36028797018963968*x"
    );
    assert_eq!(ctx.parse("0.1*x").unwrap().to_string(), "1/10*x");
}

/// W21 (review D §1.6; stable).  Today `&x * f64::NAN` is the `nan` node and `&x +
/// f64::INFINITY` is `x + oo`, without a word, while `Context::from_f64(NaN)` is an error
/// ("NaN has no exact rational value").  Right: the two doors agree.  (Under rule F of the
/// 1.0 design the operator does not compile and this test becomes a UI snapshot.)
///
/// Oracle: the library's own named constructor.
#[test]
#[ignore = "known inconsistency: `&x * f64::NAN` is silently the nan node; Context::from_f64(NaN) is an error"]
fn a_nan_operand_is_not_accepted_where_from_f64_refuses_it() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(ctx.from_f64(f64::NAN).is_err());
    assert_ne!(&x * f64::NAN, ctx.nan());
}

/// W22 (review D §1.2; stable) — not a bug, pinned: a printed `0` was a zero in every case
/// tried.  Of 22 expressions with a tiny or zero value, the 2 that print `0` are zero, 10
/// nonzero ones print their value and 10 are refused; none printed `0` for a nonzero value.
/// (The review's point stands for the type: the string cannot say "proved 0" or "0 within
/// the error".)
///
/// Oracle: `e^h − 1 − h = h²/2 + h³/6 + …`, so with `h = 10⁻⁴⁰⁰` the value is `5·10⁻⁸⁰¹·(1 +
/// h/3 + …)`; `sin² 1 + cos² 1 − 1 = 0` (mpmath, 50 and 100 digits: `0.0`).
#[test]
fn a_tiny_nonzero_value_is_printed_and_a_zero_prints_zero() {
    let ctx = Context::new();
    assert_eq!(dec(&ctx, "exp(10^(-400)) - 1 - 10^(-400)", 16), "5e-801");
    assert_eq!(dec(&ctx, "sin(1)^2 + cos(1)^2 - 1", 16), "0");
}

/// W23 (review D §1.8 F1, §7-2; stable).  Today `eval_decimal(0)` prints every digit of the
/// 128-bit working value while 4 bits are certified: `pi` gives
/// `3.1415926535897932384626433832795028842` and `1/3` gives
/// `0.333333333333333333333333333333333333334`, whose last digit is not a digit of 1/3.
/// Right: a refusal (`stats::rational_decimal` refuses `digits == 0`).
///
/// Oracle: exact arithmetic, `10**39 // 3` = `333…333` (39 threes).
#[test]
#[ignore = "known bug: eval_decimal(0) prints 38 uncertified digits (1/3 ends in 4)"]
fn eval_decimal_of_zero_digits_is_refused() {
    let ctx = Context::new();
    assert!(ctx.rational(1, 3).eval_decimal(0).is_err());
    assert!(ctx.pi().eval_decimal(0).is_err());
}

/// W24 (review D §1.2; stable) — not a bug, pinned: the layout depends on the number of
/// digits the value has, not only on its size.  The exact integer `20000000` has one digit
/// and prints in exponent form; its neighbours print plainly.  ("Prints exactly as today" is
/// the proof of the stage that moves `Display` to the new number type.)
///
/// Oracle: the values are the inputs.
#[test]
fn an_exact_integer_is_laid_out_by_the_digits_it_has() {
    let ctx = Context::new();
    assert_eq!(dec(&ctx, "20000000", 10), "2e7");
    assert_eq!(dec(&ctx, "20000001", 10), "20000001");
    assert_eq!(dec(&ctx, "20000000 + 10^(-20)", 10), "20000000");
    assert_eq!(dec(&ctx, "123456", 3), "123000");
}

/// W25 (review D §1.8 F2; review G N-G3, B-C07-09; stable).  The doc of `eval_f64`: it
/// "evaluates at the precision of `eval_decimal` with 16 digits … and rounds the result once,
/// to the nearest `f64`".  Through an unevaluated definite integral 8 digits are certified
/// (`evalf/integral.rs: QUADRATURE_F64_DIGITS`).  Today 12 of 16 such integrals are not the
/// nearest double: `Integral(x^x, x, 0, 1)` gives `0.7834305107121342` (1.8 ulp off),
/// `Integral(abs(sin(7*x))^(1/2)*exp(x), x, 0, 3)` is 14.6 ulp off (14.8 digits).
/// `eval_decimal(30)` is right for all 16 (asserted first).  Right: the nearest double, or
/// a doc that says 8 digits.
///
/// Oracle: mpmath `quad(lambda x: x**x, linspace(0, 1, 1201))` at 40 and 60 digits =
/// `0.78343051071213440705926438652698`, whose nearest double is `0.7834305107121344`
/// (`check_integrals.py`).
#[test]
#[ignore = "known bug: eval_f64 of an unevaluated definite integral is not the nearest f64 (8 digits are certified, the doc says 16)"]
fn eval_f64_of_a_definite_integral_is_the_nearest_double() {
    let ctx = Context::new();
    let sophomore = ctx.parse("Integral(x^x, x, 0, 1)").unwrap();
    assert_eq!(
        sophomore.eval_decimal(30).unwrap(),
        "0.783430510712134407059264386527"
    );
    assert_eq!(sophomore.eval_f64().unwrap(), 0.783_430_510_712_134_4);
}

// ── Review F: matrices, statistics, units, polynomials ──────────────────────

/// W02 (review F rule R1, §3.1; stable).  The pivot test of `rref` (`matrix/eigen.rs:
/// pivot_is_zero`) ends in `.is_true()` on a three-valued answer: an entry whose zero test
/// is undecided is a pivot.  Today the 1×1 matrix `[[sin(1)^2 + cos(1)^2 - 1]]` has `rank()
/// == 1` and an empty `nullspace()`, and `[[1, 1], [1, 1 + e]]` has rank 2, for each `e`
/// below — although `Ex::is_zero()` of the two constants is `Some(true)` (the next test).
/// 15 of 24 zero forms tried are counted as pivots.  Right: rank 0 and 1.
///
/// Oracle: mpmath `sin(1)**2 + cos(1)**2 - 1` = `0.0` and `log(6) - log(2) - log(3)` =
/// `1.4e-101` at 100 digits (0.0 at 50); SymPy `Matrix([[sin(1)**2 + cos(1)**2 -
/// 1]]).rank(simplify=True)` = 0 and `Matrix([[1, 1], [1, 1 + log(6) - log(2) -
/// log(3)]]).rank(simplify=True)` = 1; the identity `sin² + cos² = 1`.
#[test]
#[ignore = "known bug: rank, rref and nullspace take an entry that is zero but not proved so by the pivot test for a pivot"]
fn rank_does_not_count_a_zero_entry_as_a_pivot() {
    for zero in [
        "sin(1)^2 + cos(1)^2 - 1",
        "log(6) - log(2) - log(3)",
        "sin(x)^2 + cos(x)^2 - 1",
    ] {
        let ctx = Context::new();
        let e = ctx.parse(zero).unwrap();
        let one = ctx.int(1);
        let single = Matrix::new(vec![vec![e.clone()]]).unwrap();
        assert_eq!(single.rank(), 0, "[[{zero}]]");
        assert_eq!(single.nullspace().len(), 1, "[[{zero}]]");
        let two = Matrix::new(vec![
            vec![one.clone(), one.clone()],
            vec![one.clone(), &one + &e],
        ])
        .unwrap();
        assert_eq!(two.rank(), 1, "[[1, 1], [1, 1 + {zero}]]");
        assert_eq!(two.nullspace().len(), 1, "[[1, 1], [1, 1 + {zero}]]");
    }
}

/// W02, what is right today: the zero test of expressions decides the two constants that
/// `rank` takes for pivots, and the rank of a matrix of symbols is the generic one, as
/// `rref` documents (`[[x, 1], [1, x]]` has rank 2; at `x = 1` it has rank 1).
///
/// Oracle: as above; `det [[x, 1], [1, x]] = x² − 1`, zero at `x = ±1` only.
#[test]
fn the_zero_test_decides_the_constants_and_the_rank_of_symbols_is_generic() {
    let ctx = Context::new();
    for zero in ["sin(1)^2 + cos(1)^2 - 1", "log(6) - log(2) - log(3)"] {
        assert_eq!(ctx.parse(zero).unwrap().is_zero(), Some(true), "{zero}");
    }
    let x = ctx.symbol("x");
    let m = mat(&ctx, &[&["x", "1"], &["1", "x"]]);
    assert_eq!(m.rank(), 2);
    assert_eq!(m.subs(&x, &ctx.int(1)).rank(), 1);
}

/// W08 (review B §1.3-5, §7-1a; stable).  Today `[[1, 1], [0, x]].is_diagonalizable()` is
/// `Some(true)`: the eigenvalues `1` and `x` each have one eigenvector for a generic `x`.
/// At `x = 1` the matrix is the Jordan block `[[1, 1], [0, 1]]`.  Right: not `Some(true)` —
/// the doc of `Some(true)` has no "generically".
///
/// Oracle: SymPy `Matrix([[1, 1], [0, 1]]).is_diagonalizable()` = `False`; the library after
/// substitution (asserted first).
#[test]
#[ignore = "known bug: is_diagonalizable is Some(true) for [[1, 1], [0, x]], a Jordan block at x = 1"]
fn is_diagonalizable_true_holds_for_every_value_of_the_symbols() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = mat(&ctx, &[&["1", "1"], &["0", "x"]]);
    assert_eq!(m.subs(&x, &ctx.int(1)).is_diagonalizable(), Some(false));
    assert_ne!(m.is_diagonalizable(), Some(true));
}

/// W18 (review C §1.2, §7-3; stable).  `diagonalize` reports `NotDiagonalizable` whenever an
/// eigenvalue has fewer eigenvectors than its multiplicity, without the guard that makes
/// `is_diagonalizable` answer `None` when a zero test was undecided.  Today, for the
/// identity matrix written as `[[1, gamma(x + 1) - x*gamma(x)], [0, 1]]`: `eigenvects()`
/// returns one eigenvector for the eigenvalue 1 of multiplicity 2, `is_diagonalizable()` is
/// `None` and `diagonalize()` is `Err(NotDiagonalizable)`.  Right: the matrix is diagonal;
/// at least the error must not state the opposite.
///
/// Oracle: SymPy `simplify(gamma(x + 1) - x*gamma(x))` = 0; mpmath `gamma(x + 1) -
/// x*gamma(x)` at `x = 1/3, 5/7, 11/4`, 50 and 100 digits: `7e-102`, `0.0`, `0.0`;
/// `eye(2).is_diagonalizable()` = `True`.
#[test]
#[ignore = "known bug: diagonalize answers NotDiagonalizable for the identity matrix with the entry gamma(x + 1) - x*gamma(x)"]
fn diagonalize_does_not_call_a_diagonal_matrix_not_diagonalizable() {
    let ctx = Context::new();
    let m = mat(&ctx, &[&["1", "gamma(x + 1) - x*gamma(x)"], &["0", "1"]]);
    assert!(!matches!(
        m.diagonalize(),
        Err(SymplexError::NotDiagonalizable { .. })
    ));
    if let Ok(spaces) = m.eigenvects() {
        for (value, multiplicity, vectors) in &spaces {
            assert_eq!(vectors.len(), *multiplicity, "eigenvalue {value}");
        }
    }
}

/// W18, the other direction (stable): the query refuses where the answer is known.  Today
/// `[[sin(x), 1], [0, sin(x)]].is_diagonalizable()` is `None` while `diagonalize()` says
/// `NotDiagonalizable`, rightly: it is a Jordan block for every `x`.
///
/// Oracle: `A − sin(x)·I = [[0, 1], [0, 0]]` has rank 1 whatever `x` is (exact).
#[test]
#[ignore = "known inconsistency: is_diagonalizable is None for [[sin(x), 1], [0, sin(x)]]; diagonalize says NotDiagonalizable"]
fn is_diagonalizable_and_diagonalize_agree_on_a_jordan_block() {
    let ctx = Context::new();
    let m = mat(&ctx, &[&["sin(x)", "1"], &["0", "sin(x)"]]);
    assert!(matches!(
        m.diagonalize(),
        Err(SymplexError::NotDiagonalizable { .. })
    ));
    assert_eq!(m.is_diagonalizable(), Some(false));
}

/// W05 (review F §4.1; stable).  One name, two quantities: `Distribution::kurtosis` is
/// `E[(X − μ)⁴]/σ⁴`, `stats::data::kurtosis` is that minus 3.  Today the law Bernoulli(1/2)
/// has kurtosis `1` and its population `[0, 1]` has kurtosis `-2`.  Each is what its own doc
/// says; the 1.0 design renames both to `excess_kurtosis`.
///
/// Oracle: exact arithmetic, `μ = 1/2`, `σ² = 1/4`, `E(X − μ)⁴ = 1/16`: 1, and 1 − 3 = −2;
/// the excess kurtosis of a normal law is 0 by definition (`Distribution::normal(0,
/// 1).kurtosis()` is 3).
#[test]
#[ignore = "known inconsistency: Distribution::kurtosis and stats::data::kurtosis differ by 3 under one name"]
fn kurtosis_is_one_quantity_under_one_name() {
    let ctx = Context::new();
    let law = Distribution::bernoulli(ctx.rational(1, 2)).kurtosis();
    let population = symplex::stats::data::kurtosis(&[qi(0), qi(1)]).unwrap();
    assert_eq!(law.to_string(), population.to_string());
}

/// W07 (review F §4.1; stable).  The plain constructors do not check ("a non-positive
/// numeric `std` is a programming error and yields the distribution anyway").  Today
/// `Distribution::normal(0, -1)` has the density `-exp(-x²/2)/sqrt(2π)`, total mass −1 and
/// `std()` 1; `exponential(-2)` has density `-2e²` at 1.  Right: no such object (the 1.0
/// design has one validated constructor; then this test asks for its error).
///
/// Oracle: the definition of a density (non-negative, total mass 1).
#[test]
#[ignore = "known bug: Distribution::normal(0, -1) is accepted and has a negative density (total mass -1)"]
fn a_distribution_has_a_density_of_total_mass_one() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let d = Distribution::normal(ctx.int(0), ctx.int(-1));
    let total = d
        .density(&x)
        .integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity());
    assert_eq!(total, ctx.int(1));
    let e = Distribution::exponential(ctx.int(-2));
    assert_ne!(e.density(&ctx.int(1)).query(Props::NEGATIVE), Some(true));
}

/// W07, what is right today: the checked constructors refuse the same parameters.
///
/// Oracle: the parameter domains (`σ > 0`, rate `> 0`, `a < b`).
#[test]
fn the_checked_distribution_constructors_refuse_invalid_parameters() {
    let ctx = Context::new();
    assert!(Distribution::try_normal(ctx.int(0), ctx.int(-1)).is_err());
    assert!(Distribution::try_exponential(ctx.int(-2)).is_err());
    assert!(Distribution::try_uniform(ctx.int(3), ctx.int(1)).is_err());
}

/// W32 (review F §4.1; stable).  A fit that did not converge is returned.  Today `logit`
/// with `max_iter = 1` is `Ok` with `converged == false`, coefficients `[0.0495, 0.396]` and
/// the p-value 0.0050, where the converged fit (6 steps) has `[0.0639, 0.5115]` and 0.0015.
/// The flag is documented; the review asks for an error (`cox_ph` already gives one).
///
/// Oracle: the library's own converged fit of the same data.
#[test]
#[ignore = "known gap: logit returns Ok(converged = false) with the last iterate instead of an error"]
fn a_fit_that_did_not_converge_is_an_error() {
    let xs: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i) / 4.0 - 5.0]).collect();
    let ys: Vec<bool> = (0..40).map(|i| (i * 7 + 3) % 11 < i / 4 + 1).collect();
    let full = logit(&ys, &xs, true, &LogitOpts::default()).unwrap();
    assert!(full.converged);
    let one_step = LogitOpts {
        max_iter: 1,
        ..LogitOpts::default()
    };
    assert!(logit(&ys, &xs, true, &one_step).is_err());
}

/// W03 (review F §5.1-1; experimental).  `Frequency::from_angular_velocity` re-tags the
/// expression.  Today `AngularVelocity::rpm(60)` holds `2*pi` (rad/s) and the "frequency"
/// made from it holds `2*pi`, while `Frequency::rpm(60)` holds `1` (Hz).  Right: `f = ω/2π`,
/// 1 Hz.
///
/// Oracle: the definition, 60 revolutions per minute = 1 per second.
#[test]
#[ignore = "known bug: Frequency::from_angular_velocity does not divide by 2π (60 rpm becomes 2π Hz)"]
fn frequency_from_angular_velocity_divides_by_two_pi() {
    let ctx = Context::new();
    let sixty = ctx.int(60);
    let f = Frequency::from_angular_velocity(AngularVelocity::rpm(&sixty));
    assert_eq!(f.inner(), Frequency::rpm(&sixty).inner());
    assert_eq!(*f.inner(), ctx.int(1));
}

/// W31 (review F §2.1; stable).  `Poly::is_zero` is structural after `expand` and `eval`.
/// Today `Poly((sin(a)^2 + cos(a)^2 - 1)*x, x)` has `is_zero() == false` and
/// `total_degree() == Some(1)`; the same for the constant coefficients below, whose
/// `Ex::is_zero()` is `Some(true)`.  Right: the zero polynomial (or a three-valued answer).
///
/// Oracle: `sin² + cos² = 1`; mpmath as for W02.
#[test]
#[ignore = "known bug: Poly::is_zero is false and total_degree is Some for a polynomial whose coefficients are zero by an identity"]
fn poly_is_zero_of_a_polynomial_with_zero_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "(sin(a)^2 + cos(a)^2 - 1)*x",
        "(sin(1)^2 + cos(1)^2 - 1)*x",
        "(log(6) - log(2) - log(3))*x^3",
    ] {
        let p = Poly::try_new(&ctx.parse(src).unwrap(), &[&x]).unwrap();
        assert!(p.is_zero(), "{src}");
        assert_eq!(p.total_degree(), None, "{src}");
    }
}

/// H02 (review F §2.1, §7 rank 1; `Ex::groebner` and the stable `solve_polynomial_system`).
/// Buchberger's loop has no budget and the entry points return no `Result` of a refusal:
/// today `Ex::groebner` of cyclic-6 does not return within 20 s (release build; cyclic-5
/// takes 0.03 s, katsura-5 1.4 s, katsura-6 and cyclic-7 also run out), and nothing can stop
/// it.  Right: a basis or a refusal.  The call runs on a thread of its own, which is left
/// running when the wait ends: run this test in a process of its own (nextest does).
///
/// Oracle: none needed (the test asks only that the call returns).
#[test]
#[ignore = "known bug: Ex::groebner has no budget: cyclic-6 does not return in 20 s and cannot be stopped"]
fn a_groebner_basis_is_returned_or_refused() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let ctx = Context::new();
        let n = 6;
        let vars: Vec<Ex> = (0..n).map(|i| ctx.symbol(&format!("x{i}"))).collect();
        let mut polys = Vec::new();
        for k in 1..n {
            let mut sum = ctx.int(0);
            for i in 0..n {
                let mut prod = ctx.int(1);
                for j in 0..k {
                    prod = &prod * &vars[(i + j) % n];
                }
                sum = &sum + &prod;
            }
            polys.push(sum);
        }
        let mut prod = ctx.int(1);
        for v in &vars {
            prod = &prod * v;
        }
        polys.push(&prod - 1);
        let done = Ex::groebner(&polys, &vars, MonomialOrder::GrevLex).is_ok();
        let _ = tx.send(done);
    });
    let answered = rx.recv_timeout(std::time::Duration::from_secs(30));
    assert!(answered.is_ok(), "no answer in 30 s");
}

// ── Review B: yes/no questions and lists ────────────────────────────────────

/// W09 (review B §1.3-7, §7-1b; stable).  `probably_equal` draws rationals `p/q` with `p` in
/// `[−64, 64]`, `q` in `[1, 8]` whatever the symbols' assumptions, and answers `Some(false)`
/// ("a concrete point was found where the two sides differ") at a point outside the declared
/// domain.  Today, for an integer `n`, `(-1)^(n*(n + 1))` against `1` is `Some(false)`; for a
/// positive `p`, `atan(p) + atan(1/p)` against `pi/2` is `Some(false)`.  (Where `equals`
/// proves the identity first — `cos(pi*n)` against `(-1)^n` — nothing is seen.)  14 of 29
/// such identities are refuted.  Right: not `Some(false)`.
///
/// Oracle: exact, `{(-1)**(n*(n+1)) for n in range(-6, 7)}` = `{1}`; mpmath `atan(x) +
/// atan(1/x) - pi/2` at `x = 1/3, 5/7, 11/4`, 50 and 100 digits: `1e-101`, `0.0`, `0.0`.
#[test]
#[ignore = "known bug: probably_equal refutes an identity at a sample outside the symbol's declared domain"]
fn probably_equal_does_not_refute_an_identity_of_the_declared_domain() {
    let ctx = Context::new();
    let _n = ctx.symbol_with("n", &[Assumption::Integer]).unwrap();
    let parity = ctx.parse("(-1)^(n*(n + 1))").unwrap();
    assert_ne!(parity.probably_equal(&ctx.int(1), 8), Some(false));
    let _p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    let sum = ctx.parse("atan(p) + atan(1/p)").unwrap();
    let half_pi = ctx.parse("pi/2").unwrap();
    assert_ne!(sum.probably_equal(&half_pi, 8), Some(false));
}

/// W10 (review B §1.3-6; stable).  One fact on two channels.  Today "no root in the
/// declared domain" is `Ok([])` for a polynomial (`x + 3` for a positive `x`) and
/// `Err(NoSolution)` for anything else (`exp(x) - 1` for a positive `x`).  The doc's
/// sentence "`Ok(vec![])` is reserved for genuine equations whose roots could not be found"
/// describes neither: in 38 equations tried, "not found" was always `Err(ComputationFailed)`
/// and every `Ok([])` was a proof (so `stationary_points` and `singularities`, which read
/// it as "none", were right on 11 inputs).
///
/// Oracle: `x + 3 = 0` only at −3; `eˣ = 1` on the real line only at 0.
#[test]
#[ignore = "known inconsistency: no root in the declared domain is Ok([]) for a polynomial and Err(NoSolution) for exp(x) - 1"]
fn no_root_in_the_declared_domain_is_reported_one_way() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let polynomial = (&x + 3).solve(&x);
    let exponential = (&x.exp() - 1).solve(&x);
    assert_eq!(
        matches!(polynomial, Err(SymplexError::NoSolution { .. })),
        matches!(exponential, Err(SymplexError::NoSolution { .. }))
    );
}

/// W11 (review B §1.3-1; stable).  `check_ode_solution` returns a `bool` whose `false` means
/// "no route found zero".  Today it is `false` for `y = 2*atan(exp(x))` and `y =
/// 2*atan(tanh(x/2))` as solutions of `y' = 1/cosh(x)` (and for 4 more correct solutions of
/// 24 tried).  Right: `true`, or a third value.
///
/// Oracle: SymPy `diff(2*atan(exp(x)), x) - 1/cosh(x)` evaluated to 30 digits at `x = 1/3,
/// -2/5, 7/10`: 0, 0, 0; `simplify(diff(2*atan(tanh(x/2)), x) - 1/cosh(x))` = 0.
#[test]
#[ignore = "known bug: check_ode_solution is false for a correct solution it cannot reduce (false means not proved)"]
fn check_ode_solution_accepts_a_correct_solution() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let ode = &y.formal_diff(&x) - &ctx.parse("1/cosh(x)").unwrap();
    for solution in ["2*atan(exp(x))", "2*atan(tanh(x/2))"] {
        let s = ctx.parse(solution).unwrap();
        assert!(ode.check_ode_solution(&s, &y, &x), "y = {solution}");
    }
}

/// W12 (review B §1.3-3; experimental).  `is_controllable` is `rank(controllability matrix)
/// == n` with the rank of W02.  Today the system `A = [[0]]`, `B = [[sin(1)^2 + cos(1)^2 -
/// 1]]` — `B` is zero — is "controllable".  Right: `false`.
///
/// Oracle: `rank [B] = rank [[0]] = 0 < 1` (SymPy `Matrix([[0]]).rank()` = 0).
#[test]
#[ignore = "known bug: is_controllable is true for an input matrix that is zero but not proved so by the pivot test"]
fn a_system_with_a_zero_input_matrix_is_not_controllable() {
    let ctx = Context::new();
    let system = StateSpace::new(
        mat(&ctx, &[&["0"]]),
        mat(&ctx, &[&["sin(1)^2 + cos(1)^2 - 1"]]),
        mat(&ctx, &[&["1"]]),
        mat(&ctx, &[&["0"]]),
    )
    .unwrap();
    assert!(!system.is_controllable());
}

/// W13 (review B §1.3-4; stable).  `Equation::is_identity` ("does the equation hold for all
/// values of its symbols?") is `lhs.equals(rhs)`.  Today `x = 3` is `None` (a doctest pins
/// it), although `x = x + 1` is `Some(false)` and `(x > 0).is_tautology()` is `Some(false)`
/// (asserted first).  Right: `Some(false)`, the equation fails at `x = 0`.
///
/// Oracle: substitution, `0 ≠ 3`.
#[test]
#[ignore = "known bug: Equation::new(x, 3).is_identity() is None; the equation does not hold for all x"]
fn is_identity_of_an_equation_that_fails_somewhere_is_false() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(Equation::new(x.clone(), &x + 1).is_identity(), Some(false));
    assert_eq!(x.gt(&ctx.int(0)).is_tautology(), Some(false));
    assert_eq!(
        Equation::new(x.clone(), ctx.int(3)).is_identity(),
        Some(false)
    );
}

/// W14 (review B §1.3-7; stable).  Today `(1/2).is_prime_value()` is `None` (its doctest
/// says so) while `is_prime()` of the same value is `Some(false)`.  Right: `Some(false)`.
///
/// Oracle: a prime is an integer.
#[test]
#[ignore = "known bug: is_prime_value is None for 1/2, which is not a prime"]
fn is_prime_value_of_a_fraction_is_false() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    assert_eq!(half.is_prime(), Some(false));
    assert_eq!(half.is_prime_value(), Some(false));
}

/// W15 (review B §1.3, §7-1d; stable).  A query of several properties is `Some(false)` only
/// when all of them are known false.  Today a positive `x` has `query(NEGATIVE) ==
/// Some(false)` and `query(POSITIVE | NEGATIVE) == None`; `1/2` has `query(POSITIVE |
/// INTEGER) == None`.  Right: a conjunction with a false part is false.
///
/// Oracle: logic.
#[test]
#[ignore = "known bug: a query of two properties, one of them false, is None"]
fn a_query_of_two_properties_with_a_false_one_is_false() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert_eq!(x.query(Props::NEGATIVE), Some(false));
    assert_eq!(x.query(Props::POSITIVE | Props::NEGATIVE), Some(false));
    let half = ctx.rational(1, 2);
    assert_eq!(half.query(Props::POSITIVE | Props::INTEGER), Some(false));
}

/// W16 (review B §1.3, §7-1e; stable).  Today `I.is_less_than(1)` and `(1 + I).is_less_than(2)`
/// are `None`.  That is sound; convention D-3 of the 1.0 design ("an order relation holds
/// only where both sides are real") makes them `Some(false)`.
///
/// Oracle: the convention.
#[test]
#[ignore = "convention D-3 is not implemented: an order relation with a non-real side is None, not Some(false)"]
fn is_less_than_with_a_non_real_side_is_false() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(i.is_less_than(&ctx.int(1)), Some(false));
    assert_eq!((&i + 1).is_less_than(&ctx.int(2)), Some(false));
}

/// W17 (review B §4-3, §7-1c; stable).  Two entry points for one question.  Today
/// `cos(pi/16)` is positive by `Ex::query` and unknown by `Context::query` ("the structural
/// answer … without its value-based decisions for constants"); so are `cos(1)`, `sin(3)`,
/// `pi - 3`, `sqrt(2) - 1`, `log(2)`.  Right: one answer.
///
/// Oracle: mpmath `cos(pi/16)` = `0.98078528…` > 0.
#[test]
#[ignore = "known inconsistency: Context::query is None where Ex::query is Some(true) (cos(pi/16) > 0)"]
fn context_query_and_ex_query_give_one_answer() {
    let ctx = Context::new();
    let e = ctx.parse("cos(pi/16)").unwrap();
    assert_eq!(e.query(Props::POSITIVE), Some(true));
    assert_eq!(ctx.query(&e, Props::POSITIVE), e.query(Props::POSITIVE));
}

// ── Review C: refusals and stand-in values ──────────────────────────────────

/// W19 (review C §1.4; stable).  A value stands in for a refusal.  Today `mod_pow(2, -1, 7)`
/// is `0` ("returns 0 if `modulus ≤ 0` or `exp < 0`"), and 0 is also a residue; likewise
/// `mod_pow(2, 10, -7)`, `totient(0)`, `mobius(0)`, `divisors(0) == []`, `multiplicity(2,
/// 0)` (infinite), `is_quad_residue(1, 0) == false`.  All are documented.  Right for this
/// one: 4, the inverse of 2 modulo 7 (or an error, once the function can return one).
///
/// Oracle: Python `pow(2, -1, 7)` = 4.
#[test]
#[ignore = "known bug: mod_pow(2, -1, 7) is the stand-in 0; the inverse of 2 modulo 7 is 4"]
fn mod_pow_with_a_negative_exponent_is_not_the_residue_zero() {
    assert_eq!(ntheory::mod_pow(2, -1, 7), BigInt::from(4));
}

// ── Review E: core types ────────────────────────────────────────────────────

/// H01 (review E I-F4, §7-12c; review C §7-6; stable).  `Ex::replace` runs the caller's
/// closure with the context write-locked.  Today a closure that builds (`&x + 1`), that
/// calls `ctx.int(2)`, that prints a handle or that asks `x.is_positive()` never returns
/// (each killed after 20 s; `try_replace` the same); only `==` and `clone` are safe, which
/// the doc states.  Right: the result, `(x + 1)^2 + sin(x + 1)`.  The call runs on a thread
/// of its own, which stays blocked when the wait ends.
///
/// Oracle: the same replacement with the value built before the call (the next test).
#[test]
#[ignore = "known bug: a closure that builds an expression inside Ex::replace deadlocks on the context's lock"]
fn replace_with_a_closure_that_builds_returns() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let e = x.powi(2) + x.sin();
        let replaced = e.replace(|v| if v == x { Some(&x + 1) } else { None });
        let _ = tx.send(replaced.to_string());
    });
    let answered = rx.recv_timeout(std::time::Duration::from_secs(10));
    assert_eq!(answered.ok().as_deref(), Some("(x + 1)^2 + sin(x + 1)"));
}

/// H01, what is right today: the documented use, a replacement built before the call.
///
/// Oracle: substitution by hand, `x² + sin x` at `x + 1`.
#[test]
fn replace_with_a_value_built_before_the_call() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(2) + x.sin();
    let shifted = &x + 1;
    let replaced = e.replace(|v| if v == x { Some(shifted.clone()) } else { None });
    assert_eq!(replaced.to_string(), "(x + 1)^2 + sin(x + 1)");
}

/// W04 (review E F5, §7-12d; stable).  `symbol_with` on an existing name replaces the
/// assumptions that canonical forms have already used.  Today, with `n` an integer,
/// `cos(pi*n)` is built as `(-1)^n`; after `ctx.symbol_with("n", &[Real])` the handle is
/// still `(-1)^n`, the same text builds `cos(n*pi)`, the two are unequal (`equals` is
/// `None`), and at the now admissible `n = 1/2` the old handle is `I` where `cos(π/2) = 0`.
/// (21 constructor folds depend on an assumption: `floor(n)`, `sqrt(n^2)`, `abs(n)`, `n/n`
/// …)  Right: the second declaration is refused (the 1.0 design, question Q-S1), or nothing
/// built before it is wrong after it.
///
/// Oracle: `cos(π/2) = 0`.
#[test]
#[ignore = "known bug: symbol_with on an existing name leaves canonical forms built under the old assumptions"]
fn a_redeclared_symbol_leaves_no_stale_canonical_form() {
    let ctx = Context::new();
    let n = ctx.symbol_with("n", &[Assumption::Integer]).unwrap();
    let before = ctx.parse("cos(pi*n)").unwrap();
    if ctx.symbol_with("n", &[Assumption::Real]).is_ok() {
        let half = ctx.rational(1, 2);
        assert_eq!(before.subs(&n, &half).eval(), ctx.int(0));
        assert_eq!(before, ctx.parse("cos(pi*n)").unwrap());
    }
}

/// W26 (review E F3, §7-12a; stable).  Today `diff_with_dependent` with respect to a number
/// or a product is `0` — a value — where `diff` gives the formal `Derivative(…, 2)` and
/// `try_diff` an error.  Right: not a value (the 1.0 design takes a `Symbol`).
///
/// Oracle: `d(x·y + sin x)/dx = y + cos x ≠ 0`; a derivative by `2` is not defined.
#[test]
#[ignore = "known bug: diff_with_dependent with respect to a number or to 2*x is 0"]
fn diff_with_dependent_by_a_non_symbol_is_not_zero() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &(&x * &y) + &x.sin();
    assert!(e.try_diff(&ctx.int(2)).is_err());
    assert!(
        !e.diff_with_dependent(&ctx.int(2), &[&y])
            .is_zero_structural()
    );
    assert!(!e.diff_with_dependent(&(&x * 2), &[&y]).is_zero_structural());
}

/// W27 (review E §1 I-F3, §7-12b; stable).  `args()` hands out the children in the parent's
/// sort.  Today the sides of `x > 1` come out as `BoolEx`, and `(x > 1).args()[0].not()`
/// builds `!x`, `args[0].and(&args[1])` builds `x & 1`: Boolean operations on numbers, which
/// the sorts exist to prevent.  Right: no such expression (the 1.0 design returns the
/// children in their own sorts).
///
/// Oracle: the type rule ("`Ex`, `BoolEx` and `SetEx` are distinct types").
#[test]
#[ignore = "known bug: args() of a relation returns its numeric sides as BoolEx ((x > 1).args()[0].not() is !x)"]
fn the_sides_of_a_relation_are_not_boolean_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sides = x.gt(&ctx.int(1)).args();
    assert_ne!(sides[0].not().to_string(), "!x");
    assert_ne!(sides[0].and(&sides[1]).to_string(), "x & 1");
}

/// W28 (review E §4-5, §7-12e; stable).  The limit engine's dummies live in the user's name
/// space (`base/arena.rs: positive_symbol`).  Today, after `(sin(x)/x).limit(x, 0)`, the
/// user's symbol `__lim_r` is positive: `is_positive()` is `Some(true)` and
/// `sqrt(__lim_r^2)` is built as `__lim_r`, which is −1 at −1.  Right: untouched (or the
/// name refused when the symbol is made).
///
/// Oracle: `√((−1)²) = 1`.
#[test]
#[ignore = "known bug: a limit at a finite point declares the user's symbol `__lim_r` positive"]
fn a_limit_leaves_the_users_symbols_alone() {
    let ctx = Context::new();
    let r = ctx.symbol("__lim_r");
    let x = ctx.symbol("x");
    assert_eq!((&x.sin() / &x).limit(&x, &ctx.int(0)), ctx.int(1));
    assert_eq!(r.is_positive(), None);
    let root = ctx.parse("sqrt(__lim_r^2)").unwrap();
    assert_eq!(root.subs(&r, &ctx.int(-1)).eval(), ctx.int(1));
}

/// W29 (review E F3; stable).  The unevaluated `Limit` node has no slot for the direction.
/// Today, where the engine cannot evaluate — `abs(x)/x*Sum(1/k^(x + 2), k, 1, oo)` at 0 —
/// `limit_left` and `limit_right` return the same node as `limit`, so they compare equal
/// and their difference is `0`.  Right: `−ζ(2)` and `ζ(2)`, or two different nodes.
///
/// Oracle: SymPy `limit(Abs(x)/x*zeta(x + 2), x, 0, '-')` = `-pi**2/6`, `'+'`: `pi**2/6`;
/// mpmath `sign(x)*zeta(2 + x)` at `x = ∓1e-30`: `∓1.644934066848226436472415`.
#[test]
#[ignore = "known bug: an unevaluated one-sided limit is the two-sided Limit node (limit_left == limit_right)"]
fn an_unevaluated_one_sided_limit_keeps_its_side() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let e = ctx.parse("abs(x)/x*Sum(1/k^(x + 2), k, 1, oo)").unwrap();
    let (left, right) = (e.limit_left(&x, &zero), e.limit_right(&x, &zero));
    assert_ne!(left, right);
    assert!(!(&left - &right).is_zero_structural());
}

/// W30 (review E F3; stable).  A non-symbol where a variable belongs meets a different fate
/// in every method.  Today `sin(x).limit(&zero, &x)` (arguments swapped) builds
/// `Limit(sin(x), 0, x)`, `3.assume(Negative)` and `(x + 1).assume(Positive)` are `Ok` and
/// do nothing (`(x + 1).is_positive()` stays `None`).  `try_limit` refuses (asserted first).
/// Right: a refusal (the 1.0 design: a `Symbol` parameter).
///
/// Oracle: 3 is not negative.
#[test]
#[ignore = "known bug: a number where a variable belongs: limit builds Limit(sin(x), 0, x), assume returns Ok and does nothing"]
fn a_non_symbol_where_a_variable_belongs_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    assert!(x.sin().try_limit(&zero, &x).is_err());
    assert_ne!(x.sin().limit(&zero, &x).to_string(), "Limit(sin(x), 0, x)");
    assert!(ctx.int(3).assume(Assumption::Negative).is_err());
    assert!((&x + 1).assume(Assumption::Positive).is_err());
}

// ── Review A: the surface (names and macros) ────────────────────────────────

/// W06 (review A §4-2, finding M1; stable).  The printer writes a symbol by its name and the
/// parser looks an identifier up among the constants first.  Today `ctx.symbol("i")` prints
/// `i`, which `ctx.parse` reads as the imaginary unit: `i^2 + 1` (5 at `i = 2`) reads back
/// as `0`.  The same for `e`, `I`, `E`, `pi`, `Pi`, `PI`, `oo`, `inf`, `Inf`, `nan`, `zoo`,
/// `EulerGamma`, `euler_gamma`, `Catalan`, `GoldenRatio`, `golden_ratio`.  Right:
/// `parse(print(e)) == e`, or the name refused.  (`to_json`/`from_json` keeps the symbol:
/// asserted first.)
///
/// Oracle: `2² + 1 = 5`.
#[test]
#[ignore = "known bug: a symbol named i, e, pi, … prints as the constant of that name and reads back as it"]
fn a_symbol_named_like_a_constant_survives_print_and_parse() {
    for name in ["i", "e", "I", "E", "pi", "oo", "nan", "zoo"] {
        let ctx = Context::new();
        let s = ctx.symbol(name);
        let e = &s.powi(2) + 1;
        assert_eq!(ctx.from_json(&e.to_json().unwrap()).unwrap(), e);
        let back = ctx.parse(&e.to_string()).unwrap();
        assert_eq!(back, e, "symbol `{name}`: `{e}` reads back as `{back}`");
    }
}

/// W34 (review A findings M2, M3; stable).  In `expr!` an identifier that is a constant's
/// name is the constant, whatever is in scope.  Today `expr!(ctx, E - I*R)` with variables
/// `E`, `I`, `R` is `e − i·R` (at `E = 10, I = 2, R = 3` it is `E − 3i`, not 4).  (Also
/// seen: `expr!(ctx, n*m + 1)` with `n, m: i64` is an `i64`, which overflows.)  Right: the
/// variables, or a compile error.
///
/// Oracle: `10 − 2·3 = 4`.
#[test]
#[ignore = "known bug: expr! reads E and I as the constants e and i although variables of those names are in scope"]
#[allow(non_snake_case)]
fn expr_macro_uses_the_variables_in_scope() {
    let ctx = Context::new();
    let ctx = &ctx;
    let (E, I, R) = (ctx.symbol("E"), ctx.symbol("I"), ctx.symbol("R"));
    let voltage = expr!(ctx, E - I * R);
    let value = voltage
        .subs(&E, &ctx.int(10))
        .subs(&I, &ctx.int(2))
        .subs(&R, &ctx.int(3))
        .eval();
    assert_eq!(value, ctx.int(4));
}

// ── Review G: features and documents ────────────────────────────────────────

/// W35 (review G B-C17-01; stable).  The book: compiled code computes "the value `eval_f64`
/// gives, or NaN where that value is not real", "under one rule without exceptions".  Today
/// the compiled `fibonacci(x)` is NaN at `x = 0.5` (and `lucas(x)` at 2.5), where `eval_f64`
/// gives the real value `0.5688644810057831` (asserted first).  Right: that value.
///
/// Oracle: `mpmath.fibonacci(mpf(1)/2)` at 50 and 100 digits = `0.5688644810057831072783079`
/// (= `(φ^x − cos(πx)·φ^(−x))/√5` at `x = 1/2`).
#[test]
#[ignore = "known bug: compiled fibonacci(x) is NaN at a non-integer x where eval_f64 gives the real value"]
fn compiled_fibonacci_agrees_with_eval_f64() {
    let ctx = Context::new();
    let exact = ctx.parse("fibonacci(1/2)").unwrap().eval_f64().unwrap();
    assert_eq!(exact, 0.568_864_481_005_783_1);
    let compiled = ctx.parse("fibonacci(x)").unwrap().compile(&["x"]).unwrap();
    let value = compiled.call(&[0.5]);
    assert!((value - exact).abs() < 1e-14, "compiled: {value}");
}

/// W36 (review G N-G2, §7.4; stable).  Six assumption flags are accepted and read by no
/// engine.  Today, with `A` and `B` declared `NotCommutative`, `query(COMMUTATIVE)` is
/// `Some(false)` and yet `A*B == B*A`, `A*B - B*A` is `0` and `(A*B)^2` is `A^2*B^2`.
/// Right: products that do not commute, or the declaration refused (the review removes the
/// six variants).
///
/// Oracle: the meaning of the flag.
#[test]
#[ignore = "known bug: Assumption::NotCommutative is accepted and ignored (A*B - B*A is 0)"]
fn a_noncommutative_symbol_does_not_commute() {
    let ctx = Context::new();
    let declared = (
        ctx.symbol_with("A", &[Assumption::NotCommutative]),
        ctx.symbol_with("B", &[Assumption::NotCommutative]),
    );
    if let (Ok(a), Ok(b)) = declared {
        assert_ne!(&a * &b, &b * &a);
    }
}

/// W37 (review G R10, §7.4; stable).  `mellin_transform` returns the transform and its
/// strip; `inverse_mellin_transform` cannot take the strip and chooses the one right of the
/// pole (its doc says so).  Today the transform of `H(x − 1)` is `-1/s` on `re(s) < 0`
/// (right, asserted first), and its inverse is `-H(1 − x)`: 0 at `x = 3` and −1 at `x = 1/3`
/// instead of 1 and 0.  The same for `H(x − 1)/x²`, `H(x − 1)/x³`, `H(x − 2)/x`.
///
/// Oracle: SymPy `mellin_transform(Heaviside(x - 1), x, s)` = `(-1/s, (-oo, 0), True)`;
/// `inverse_mellin_transform(-1/s, s, x, (None, 0))` = `Heaviside(x - 1)`, and with the
/// strip `(0, None)`: `-Heaviside(1 - x)`.
#[test]
#[ignore = "known bug: inverse_mellin_transform cannot take the strip: the transform of H(x - 1) inverts to -H(1 - x)"]
fn a_mellin_transform_on_a_left_strip_inverts_to_its_function() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let s = ctx.symbol("s");
    let step = ctx.parse("heaviside(x - 1)").unwrap();
    let (transform, strip) = step.mellin_transform(&x, &s).unwrap();
    assert_eq!(transform, ctx.parse("-1/s").unwrap());
    assert_eq!(strip.to_string(), "0 > re(s)");
    let back = transform.inverse_mellin_transform(&s, &x).unwrap();
    assert_eq!(back.subs(&x, &ctx.int(3)).eval(), ctx.int(1));
    assert_eq!(back.subs(&x, &ctx.rational(1, 3)).eval(), ctx.int(0));
}

/// W38 (review G F-C01-10, §7.4; stable) — as the review expected, decided since 0.59:
/// `(erf(1) + erfc(1) - 1).is_zero()` is `Some(true)`, also at `1/3`, `2`, `π`, `√2`, `1 + i`
/// and with a common factor.  (Left open, the roadmap's B-C02-29: `erf(1)^2 - (1 -
/// erfc(1))^2` is `None`.)
///
/// Oracle: `erf + erfc = 1` (the definition of `erfc`).
#[test]
fn erf_plus_erfc_minus_one_is_zero() {
    let ctx = Context::new();
    for src in [
        "erf(1) + erfc(1) - 1",
        "erf(1/3) + erfc(1/3) - 1",
        "2*erf(1) + 2*erfc(1) - 2",
        "erf(pi) + erfc(pi) - 1",
    ] {
        assert_eq!(ctx.parse(src).unwrap().is_zero(), Some(true), "{src}");
    }
}

/// W39 (review G N-G4; stable).  A result that prints and does not read back.  Today
/// `integrate(1/(x^5 - x + 1))` prints `RootSum(2869*__rs_t^5 + …, __rs_t -> __rs_t*ln(…))`,
/// which `parse` refuses ("expected expression, got Gt"; the parser has `rootof` and no
/// `rootsum`).  Right: `parse(print(e)) == e` for what a stable function returns.
///
/// Oracle: the round-trip property.
#[test]
#[ignore = "known gap: integrate returns a RootSum whose printed form parse refuses"]
fn a_rootsum_result_reads_back() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integral = ctx.parse("1/(x^5 - x + 1)").unwrap().integrate(&x);
    let back = ctx.parse(&integral.to_string());
    assert_eq!(back.ok(), Some(integral));
}

/// W39, second half (review G N-G4a; stable).  Today `to_mathml` refuses a `RootOf` ("no
/// presentation for `RootOf`"), which `solve` returns routinely and `to_latex` prints
/// (`\operatorname{RootOf}\left(x^{5} - x + 1, 0\right)`, asserted first).
///
/// Oracle: the other printer.
#[test]
#[ignore = "known gap: to_mathml refuses RootOf, which solve returns and to_latex prints"]
fn a_rootof_has_a_mathml_form() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let roots = ctx.parse("x^5 - x + 1").unwrap().solve(&x).unwrap();
    assert_eq!(roots.len(), 5);
    assert!(roots[0].to_latex().contains("RootOf"));
    assert!(roots[0].to_mathml().is_ok());
}

/// R03 (review B §7-2; stable) — pinned: a principal solve and the general solve of the same
/// equation can be compared.  `solve(sin(x))` returns the principal roots `[0, pi]`;
/// `solve_general` returns two families with one integer parameter, which is how a caller
/// (or the completeness type of the 1.0 design) learns that the first list is not all.
///
/// Oracle: `sin x = 0` exactly at `x = nπ`.
#[test]
fn a_principal_solve_can_be_compared_with_the_general_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let principal = x.sin().solve(&x).unwrap();
    assert_eq!(principal, vec![ctx.int(0), ctx.pi()]);
    let general = x.sin().solve_general(&x).unwrap();
    assert_eq!(general.solutions.len(), 2);
    assert_eq!(general.parameters.len(), 1);
}
