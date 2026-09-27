//! evalf after 0.30: refusals and false `PrecisionExhausted` where a
//! certified value exists, and the minimal polynomial of a root of a
//! non-real number.  Reference values cite the mpmath call (exact rational
//! inputs, two `mp.dps` settings agreeing) or the SymPy call.

use symplex::prelude::*;

fn dec(ctx: &Context, src: &str, digits: u32) -> Result<String, SymplexError> {
    ctx.parse(src).unwrap().eval_decimal(digits)
}

/// A quotient whose numerator and denominator are both 0 at the first
/// working precision (`e^ε − 1` with `ε = 10⁻⁴⁵` is 0 at 128 bits) gave NaN,
/// and before 0.31 that was a `PrecisionExhausted` at once: `eval_f64`,
/// `eval_complex64` and 16 digits were refused although 30 and 60 digits
/// worked.  Now the precision is raised as for a value that is zero to the
/// working precision.
///
/// mpmath: `mp.dps=120; e=mpf(10)**-45; (exp(e)-1)/(exp(2*e)-1)` →
/// `0.4999999999999999999999999999999999999999999997500000…`;
/// `mp.dps=1500; e=mpf(10)**-300` → `0.5` to 30 digits.
#[test]
fn quotient_of_two_differences_zero_at_the_first_precision() {
    let ctx = Context::new();
    let src = "(exp(10^(-45)) - 1)/(exp(2*10^(-45)) - 1)";
    let f = ctx.parse(src).unwrap();
    assert_eq!(f.eval_f64().unwrap(), 0.5);
    let z = f.eval_complex64().unwrap();
    assert_eq!((z.re, z.im), (0.5, 0.0));
    assert_eq!(f.eval_decimal(16).unwrap(), "0.5");
    assert_eq!(f.eval_decimal(30).unwrap(), "0.5");
    assert_eq!(
        f.eval_decimal(60).unwrap(),
        "0.49999999999999999999999999999999999999999999975"
    );
    // 10⁻³⁰⁰: 0/0 up to about 1,000 bits, within the zero search at 16 digits.
    assert_eq!(
        dec(&ctx, "(exp(10^(-300)) - 1)/(exp(2*10^(-300)) - 1)", 16).unwrap(),
        "0.5"
    );
}

/// The same with the value itself small: the quotient minus `1/2`.
///
/// mpmath: `mp.dps=1500; e=mpf(10)**-150; (exp(e)-1)/(exp(2*e)-1) - mpf(1)/2`
/// → `-2.5e-151`.
#[test]
fn quotient_minus_its_limit_is_found_by_the_zero_search() {
    let ctx = Context::new();
    let s = dec(
        &ctx,
        "(exp(10^(-150)) - 1)/(exp(2*10^(-150)) - 1) - 1/2",
        16,
    )
    .unwrap();
    assert_eq!(s, "-2.5e-151");
}

/// A radicand that is exactly `−1` built by complex arithmetic
/// (`8/(1 + i) = 4 − 4i`) had an imaginary part that cancelled to a
/// rounding residue within its error of 0, so the side of the cut of `√`
/// was undecidable at every precision: refused at 16, 30 and 60 digits
/// before 0.31.  A complex-rational sub-expression now takes its exact
/// value, and an exact `−1` is on the cut: the principal value.
///
/// mpmath: `sqrt(mpc(-1, 0))` → `1j`; `log(-1)` → `3.14159…j`;
/// `acos(-3)` → `(3.14159265358979 - 1.76274717403909j)`;
/// `asinh(mpc(0, -5))` → `(-2.29243166956118 - 1.5707963267949j)`
/// (SymPy's `N(asinh(-5*I))` takes the other side of the cut, `+2.29…`;
/// symplex follows mpmath's counter-clockwise convention, see `accuracy.rs`).
#[test]
fn exact_complex_rationals_on_a_branch_cut() {
    let ctx = Context::new();
    for d in [16, 30, 60] {
        assert_eq!(dec(&ctx, "sqrt(8/(1 + I) + 4*I - 5)", d).unwrap(), "i");
    }
    let z = ctx
        .parse("sqrt(8/(1 + I) + 4*I - 5)")
        .unwrap()
        .eval_complex64()
        .unwrap();
    assert_eq!((z.re, z.im), (0.0, 1.0));
    assert_eq!(
        dec(&ctx, "ln(8/(1 + I) + 4*I - 5)", 30).unwrap(),
        "3.14159265358979323846264338328*i"
    );
    assert_eq!(
        dec(&ctx, "acos(8/(1 + I) + 4*I - 7)", 16).unwrap(),
        "3.141592653589793 - 1.762747174039086*i"
    );
    assert_eq!(
        dec(&ctx, "asinh(116/(2 + 5*I) + 15*I - 8)", 16).unwrap(),
        "-2.292431669561178 - 1.570796326794897*i"
    );
}

/// An exactly represented tiny imaginary part decides the side of the cut
/// for a non-integer power (before 0.31 refused at 16 and 30 digits: the
/// hunter's `(29/(2 − 5i) + (−10⁻¹⁵⁰ − 5)i − 16/3)^(1/3)`, radicand
/// `−10/3 − 10⁻¹⁵⁰·i`, just below the cut: the root is near `e^{−iπ/3}`).
///
/// mpmath: `mp.dps=200; mpc(-mpf(10)/3, -mpf(10)**-150)**(mpf(1)/3)` (the same
/// at `mp.dps=300`) →
/// `(0.74690079109286078479124700233976 - 1.29367011838622283480674794154659j)`.
#[test]
fn exact_tiny_imaginary_part_decides_the_cut_of_a_cube_root() {
    let ctx = Context::new();
    let s = dec(&ctx, "(29/(2 - 5*I) + (-10^(-150) - 5)*I - 16/3)^(1/3)", 30).unwrap();
    assert_eq!(
        s,
        "0.74690079109286078479124700234 - 1.29367011838622283480674794155*i"
    );
}

/// The argument of `erfc⁻¹` cancels about 500 bits (`sin(1 + 10⁻¹⁵⁰) −
/// sin 1 ≈ 5.4·10⁻¹⁵¹`).  At 16 and 30 digits the adaptive loop stopped at
/// its cap (384, 412 bits) because only a *root* that was noise went
/// further; the argument's ball, not the root's, is what lacked precision.
/// Before 0.31 refused at 16 and 30 digits, while 60 digits worked.
///
/// mpmath: `mp.dps=400; y=sin(1+mpf(10)**-150)-sin(1); erfinv(1-y)` (the same
/// at `mp.dps=500`) → `18.5070640152068969795248015234399368804415276610137…`.
#[test]
fn cancellation_inside_an_argument_is_pursued_like_a_root() {
    let ctx = Context::new();
    let src = "erfcinv(sin(1 + 10^(-150)) - sin(1))";
    let f = ctx.parse(src).unwrap();
    assert_eq!(f.eval_f64().unwrap(), 18.507064015206897);
    assert_eq!(f.eval_decimal(16).unwrap(), "18.5070640152069");
    assert_eq!(
        f.eval_decimal(30).unwrap(),
        "18.5070640152068969795248015234"
    );
    // A pole of Γ reached by a cancellation: `1 − cos(10⁻¹⁰⁰) = 5·10⁻²⁰¹`.
    // mpmath: `mp.dps=1500; gamma(1 - cos(mpf(10)**-100))` → `2.0e+200`.
    assert_eq!(dec(&ctx, "gamma(1 - cos(10^(-100)))", 16).unwrap(), "2e200");
    // The side of a cut decided by a part that cancels 660 bits.
    // mpmath: `mp.dps=1500; sqrt(mpc(-64, sin(1+8*mpf(10)**-200)-sin(1)))` →
    // `(3.3e-201 + 8.0j)`: the real part is negligible at 16 digits.
    assert_eq!(
        dec(&ctx, "sqrt(-64 + I*(sin(1 + 8*10^(-200)) - sin(1)))", 16).unwrap(),
        "8*i"
    );
}

/// A value beyond the exponent range of the arbitrary-precision floats is
/// reported as such (as `uppergamma` did), not as `PrecisionExhausted` —
/// which it was before 0.31 for `Γ(10²⁰/3)` and `exp(10¹⁶)`.
///
/// mpmath: `mp.dps=30; loggamma(mpf(10)**20/3)/log(2)` → `2.114…e21`.
#[test]
fn overflow_of_the_exponent_range_is_reported() {
    let ctx = Context::new();
    for src in ["gamma(10^20/3)", "exp(10^16)"] {
        let f = ctx.parse(src).unwrap();
        for r in [
            f.eval_decimal(16),
            f.eval_decimal(30),
            f.eval_f64().map(|x| x.to_string()),
        ] {
            match r {
                Err(SymplexError::Unevaluable { reason }) => {
                    assert!(reason.contains("overflows"), "{src}: {reason}");
                }
                other => panic!("{src}: {other:?}"),
            }
        }
    }
    match ctx.parse("gamma(10^20/3)").unwrap().eval_decimal(16) {
        Err(SymplexError::Unevaluable { reason }) => {
            assert!(reason.contains("2.114e21"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
}

/// Huge but well-conditioned quotients.  `Γ(10⁸) ≈ 2^(2.5·10⁹)` and
/// `exp(10¹⁶)` are beyond the exponent range (about `2^(±2.1·10⁹)`), so
/// `Γ(10⁸ + 1)/Γ(10⁸)/10⁸` and `erfc(10⁸)·10⁸·e^(10¹⁶)` are refused with
/// that reason (before 0.31: `PrecisionExhausted`, achieved 0 digits); in
/// range they evaluate.
///
/// mpmath: `mp.dps=60; erfc(mpf(10)**4)*mpf(10)**4*exp(mpf(10)**8)` →
/// `0.564189580726808411523515725046664722042862738751771196042314`
/// (the same at `mp.dps=120`); `gamma(mpf(10)**7+1)/gamma(mpf(10)**7)/mpf(10)**7`
/// → `1.0`.
#[test]
fn huge_well_conditioned_quotients() {
    let ctx = Context::new();
    for src in [
        "gamma(10^8+1)/gamma(10^8)/10^8",
        "erfc(10^8)*10^8*exp(10^16)",
    ] {
        match dec(&ctx, src, 16) {
            Err(SymplexError::Unevaluable { reason }) => {
                assert!(reason.contains("overflows"), "{src}: {reason}");
            }
            other => panic!("{src}: {other:?}"),
        }
    }
    assert_eq!(
        dec(&ctx, "gamma(10^7+1)/gamma(10^7)/10^7", 16).unwrap(),
        "1"
    );
    assert_eq!(
        dec(&ctx, "erfc(10^4)*10^4*exp(10^8)", 30).unwrap(),
        "0.564189580726808411523515725047"
    );
}

/// `loggamma` far above the real axis left of `Re z = ½`: the bound on
/// `|cot πu|` was `cosh(πb)/sinh(πb)` in `f64`, `∞/∞ = NaN` for `b ≥ 226`,
/// so the value had no bound at any precision (refused before 0.31).
///
/// mpmath: `mp.dps=60; loggamma(mpc(mpf(-6)/17, mpf(10)**8))` →
/// `(-157079647.472308233854300572176876738322992214469645922540 +
/// 1742068073.05543967702139887318052207322980727872463040127j)`.
#[test]
fn loggamma_far_above_the_axis_left_of_one_half() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "loggamma(-6/17 + 10^8*I)", 16).unwrap(),
        "-157079647.4723082 + 1742068073.05544*i"
    );
}

/// `minimal_polynomial` of a square root (or `n`-th root) of a number that
/// is not a positive real: `m_α(xⁿ)`, factored, and the factor vanishing at
/// the principal root (`arg ∈ (−π, π]`).  Before 0.31 `None`, so the exact
/// zero test of `√(−4i − 3) − 1 + 2i` was undecided.
///
/// SymPy: `minimal_polynomial(sqrt(-4*I-3), x)` → `x**2 - 2*x + 5`;
/// `sqrt(1+I)` → `x**4 - 2*x**2 + 2`; `(-2)**Rational(1,3)` → `x**3 + 2`;
/// `sqrt(sqrt(2)-3)` → `x**4 + 6*x**2 + 7`; `(3+4*I)**Rational(1,3)` →
/// `x**6 - 6*x**3 + 25`; `sqrt(-I)` → `x**4 + 1`;
/// `(-4*I-3)**Rational(3,2)` → `x**2 + 22*x + 125`.
#[test]
fn minimal_polynomial_of_roots_of_non_real_numbers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        ("sqrt(-4*I - 3)", "x^2 - 2*x + 5"),
        ("sqrt(1 + I)", "x^4 - 2*x^2 + 2"),
        ("(-2)^(1/3)", "x^3 + 2"),
        ("sqrt(sqrt(2) - 3)", "x^4 + 6*x^2 + 7"),
        ("(3 + 4*I)^(1/3)", "x^6 - 6*x^3 + 25"),
        ("sqrt(-I)", "x^4 + 1"),
        ("(-4*I - 3)^(3/2)", "x^2 + 22*x + 125"),
        ("sqrt(8/(1 + I) + 4*I - 5)", "x^2 + 1"),
        ("sqrt(-4*I - 3) - 1 + 2*I", "x"),
        ("(-1)^(1/3) - 1/2 - sqrt(3)*I/2", "x"),
    ];
    for (src, want) in cases {
        let got = ctx.parse(src).unwrap().minimal_polynomial(&x);
        assert_eq!(got, Some(ctx.parse(want).unwrap()), "{src}");
    }
}

/// The certified zero test behind linear algebra (`Matrix::is_zero`, which
/// falls back to the exact test for algebraic numbers when the value is 0
/// only to the precision reached) decides zeros built from principal roots
/// of negative and non-real numbers.  Before 0.31 the cube-root identity was
/// undecided (`None`): `(−2)^(1/3)` was not recognised as algebraic.
///
/// SymPy: `minimal_polynomial((-2)**Rational(1,3) - 2**Rational(1,3)*(S(1)/2
/// + sqrt(3)*I/2), x)` → `x`; `minimal_polynomial(sqrt(-4*I-3) - 1 + 2*I, x)`
/// → `x`; `minimal_polynomial(sqrt(-4*I-3) - 1 - 2*I, x)` → `x**2 + 16`.
#[test]
fn zero_test_of_principal_roots() {
    let ctx = Context::new();
    let is_zero = |src: &str| {
        Matrix::new(vec![vec![ctx.parse(src).unwrap()]])
            .unwrap()
            .is_zero()
    };
    assert_eq!(
        is_zero("(-2)^(1/3) - 2^(1/3)*(1/2 + sqrt(3)*I/2)"),
        Some(true)
    );
    assert_eq!(is_zero("sqrt(-4*I - 3) - 1 + 2*I"), Some(true));
    assert_eq!(is_zero("sqrt(-4*I - 3) - 1 - 2*I"), Some(false));
}

/// `ζ(s)` at an `s` so large that its parity is unknown at the working
/// precision is refused at once: `zeta(erfi(−1000))` (`s ≈ −1.7·10^434291`)
/// ran the argument reduction of `sin(πs/2)` for minutes (fuzz_evalf,
/// 0.31).  Moderate negative arguments keep their values.
///
/// mpmath (dps 30): `zeta(mpf(-101)/2)` → `2399094238135732095670783.28579`;
/// `zeta(-mpf(10)**6 - mpf(1)/2)` → `-1.00478689391928874432646062416e+4767531`.
#[test]
fn zeta_of_a_huge_negative_argument_is_refused_at_once() {
    let ctx = Context::new();
    let t = std::time::Instant::now();
    assert!(
        ctx.parse("zeta(erfi(-1000))")
            .unwrap()
            .eval_decimal(16)
            .is_err()
    );
    assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
    assert_eq!(
        ctx.parse("zeta(-101/2)").unwrap().eval_decimal(16).unwrap(),
        "2.399094238135732e24"
    );
    assert_eq!(
        ctx.parse("zeta(-10^6 - 1/2)")
            .unwrap()
            .eval_decimal(16)
            .unwrap(),
        "-1.004786893919289e4767531"
    );
}
