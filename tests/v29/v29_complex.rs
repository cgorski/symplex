//! After 0.29 — singular atoms and complex parts: function applications at
//! exact special arguments are folded when they are built (so the
//! canonical arithmetic never cancels a zero or an infinite atom), `evalf`
//! no longer certifies a zero ball wider than the requested digits, and the
//! complex-parts hunter (`re`, `im`, `abs`, `arg`, `conjugate`,
//! `as_real_imag`, `expand_complex`, `polar`, `abs_squared` and the
//! assumption queries against certified values at real and complex points).
//!
//! Each test says what was wrong before; every reference value cites the
//! SymPy 1.14 / mpmath 1.3.0 call that produced it.

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

fn show(e: &Ex) -> String {
    format!("{e}")
}

// ═══════════════════════════════════════════════════════════════════════════
// Substitution at a removable singularity is the value (undefined), not the limit
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn substitution_at_a_removable_singularity_is_undefined_not_the_limit() {
    // Construction kept `sin(0)`, `cos(0)`, `ln(1)`, … as atoms, and the
    // canonical `x·x⁻¹ → 1`, `0·x → 0` then cancelled them: the value at
    // the point came out as the limit (or as `zoo` for `(1 − cos 0)·0⁻²`,
    // the hidden zero `1 − cos 0` not being seen).
    // SymPy: `(sin(8*n)/(8*sin(n))).subs(n, 0)`, `((1-cos(x))/(1-cos(2*x)))
    //   .subs(x, 0)`, `((exp(x)-1)/(exp(2*x)-1)).subs(x, 0)`,
    //   `((1-cos(x))/x**2).subs(x, 0)`, `(log(cos(x))/log(cos(2*x))).subs(x,
    //   0)`, `(sinh(2*x)/sinh(x)).subs(x, 0)`, `(x*log(x)).subs(x, 0)`,
    //   `(tan(x)*cos(x)).subs(x, pi/2)`, `(atanh(x)-atanh(x**2)).subs(x, 1)`,
    //   `(gamma(x)*sin(pi*x)).subs(x, 0)` → nan, every one.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let cases: [(&str, Ex, &str); 10] = [
        ("sin(8*x)/(8*sin(x))", zero.clone(), "1/8"),
        ("(1-cos(x))/(1-cos(2*x))", zero.clone(), "1"),
        ("(exp(x)-1)/(exp(2*x)-1)", zero.clone(), "1"),
        ("(1-cos(x))/x^2", zero.clone(), "zoo"),
        ("ln(cos(x))/ln(cos(2*x))", zero.clone(), "1"),
        ("sinh(2*x)/sinh(x)", zero.clone(), "1"),
        ("x*ln(x)", zero.clone(), "0"),
        ("tan(x)*cos(x)", ctx.pi() / 2, "cos(1/2*pi)*tan(1/2*pi)"),
        ("atanh(x)-atanh(x^2)", ctx.int(1), "0"),
        ("gamma(x)*sin(pi*x)", zero.clone(), "sin(0)*Gamma(0)"),
    ];
    for (s, at, before) in cases {
        let v = p(&ctx, s).subs(&x, &at);
        assert_eq!(show(&v), "nan", "{s} at x = {at} (before 0.30: {before})");
    }
    // A symbol stays generic: sin(x)/sin(x) is 1 before the substitution.
    // SymPy: `(sin(x)/sin(x)).subs(x, 0)` → 1.
    assert_eq!(show(&p(&ctx, "sin(x)/sin(x)").subs(&x, &zero)), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Constant applications fold to their rational / singular values when built
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn constant_applications_fold_at_construction() {
    // Before 0.30 each of these stayed an unevaluated atom until `eval()`
    // (some — `ln(0)`, `Γ(0)`, `atanh(1)`, `(−1)!` — even after it), and
    // `ln(0)/ln(0)` was 1, `ln(0) − ln(0)` was 0.
    // SymPy (each call as written, `log` for `ln`): log(0) → zoo, sin(0) → 0,
    //   log(0)/log(0) → nan, log(0)-log(0) → nan, cos(pi/2) → 0,
    //   tan(pi/2) → zoo, gamma(0) → zoo, gamma(-3) → zoo, log(1) → 0,
    //   exp(0) → 1, exp(-oo) → 0, exp(oo) → oo, exp(zoo) → nan,
    //   atanh(1) → oo, atanh(-1) → -oo, sinh(I*pi) → 0, cosh(I*pi/2) → 0,
    //   tanh(I*pi/2) → zoo, acosh(1) → 0, asin(0) → 0, loggamma(0) → oo,
    //   loggamma(1) → 0, digamma(0) → zoo, erfc(oo) → 0, LambertW(0) → 0,
    //   LambertW(-exp(-1)) → -1, floor(Rational(1,2)) → 0, factorial(-1) →
    //   zoo, binomial(2,3) → 0, binomial(2,-1) → 0, beta(0,1) → zoo,
    //   Heaviside(-1) → 0, DiracDelta(1) → 0, besselj(1,0) → 0,
    //   bessely(0,0) → -oo, besselk(0,0) → oo, erfinv(1) → oo,
    //   elliptic_k(1) → zoo, polylog(2,0) → 0, cos(pi) → -1,
    //   sin(pi/6) → 1/2, gamma(5) → 24.
    let ctx = Context::new();
    let cases = [
        ("ln(0)", "zoo"),
        ("sin(0)", "0"),
        ("ln(0)/ln(0)", "nan"),
        ("ln(0)-ln(0)", "nan"),
        ("cos(pi/2)", "0"),
        ("tan(pi/2)", "zoo"),
        ("gamma(0)", "zoo"),
        ("gamma(-3)", "zoo"),
        ("ln(1)", "0"),
        ("exp(0)", "1"),
        ("exp(-oo)", "0"),
        ("exp(oo)", "oo"),
        ("exp(zoo)", "nan"),
        ("atanh(1)", "oo"),
        ("atanh(-1)", "-oo"),
        ("sinh(I*pi)", "0"),
        ("cosh(I*pi/2)", "0"),
        ("tanh(I*pi/2)", "zoo"),
        ("acosh(1)", "0"),
        ("asin(0)", "0"),
        ("loggamma(0)", "oo"),
        ("loggamma(1)", "0"),
        ("digamma(0)", "zoo"),
        ("erfc(oo)", "0"),
        ("LambertW(0)", "0"),
        ("LambertW(-exp(-1))", "-1"),
        ("floor(1/2)", "0"),
        ("factorial(-1)", "zoo"),
        ("binomial(2,3)", "0"),
        ("binomial(2,-1)", "0"),
        ("beta(0,1)", "zoo"),
        ("Heaviside(-1)", "0"),
        ("DiracDelta(1)", "0"),
        ("besselj(1,0)", "0"),
        ("bessely(0,0)", "-oo"),
        ("besselk(0,0)", "oo"),
        ("erfinv(1)", "oo"),
        ("elliptic_k(1)", "zoo"),
        ("polylog(2,0)", "0"),
        ("cos(pi)", "-1"),
        ("sin(pi/6)", "1/2"),
    ];
    let mut bad = Vec::new();
    for (s, want) in cases {
        let got = show(&p(&ctx, s));
        if got != want {
            bad.push(format!("{s}: got {got}, want {want}"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    // Values that are rational multiples of `i` fold too; the last one was
    // refused by `evalf` before (found by the hunter: `sqrt(−i + sinh z)` at
    // `z = iπ/2`).  SymPy: exp(I*pi/2) → I, exp(3*I*pi/2) → -I,
    // sinh(I*pi/6) → I/2, tanh(I*pi/4) → I, sqrt(-I + sinh(I*pi/2)) → 0;
    // exp(I*pi/3) stays exp(I*pi/3).
    for (s, want) in [
        ("exp(I*pi/2)", "I"),
        ("exp(3*I*pi/2)", "-I"),
        ("sinh(I*pi/6)", "1/2*I"),
        ("tanh(I*pi/4)", "I"),
        ("sqrt(-I + sinh(I*pi/2))", "0"),
        ("exp(I*pi/3)", "exp(1/3*pi*I)"),
    ] {
        assert_eq!(show(&p(&ctx, s)), want, "{s}");
    }
    // `atan(±i)` is infinite (SymPy: `atan(I)` → `oo*I`); since the 0.30
    // leftovers it keeps its direction (it was `zoo`; see
    // `v29_leftovers2::directed_infinities_*`).
    assert_eq!(show(&p(&ctx, "atan(I)")), "I*oo");
    // Irrational special values are left to `eval` (sin(π/4) = √2/2), and so
    // are the positive-integer values of Γ (numbers that can be huge).
    let s = p(&ctx, "sin(pi/4)");
    assert_eq!(show(&s), "sin(1/4*pi)");
    assert_eq!(show(&s.eval()), "1/2*sqrt(2)");
    let g = p(&ctx, "gamma(5)");
    assert_eq!(show(&g), "Gamma(5)");
    assert_eq!(show(&g.eval()), "24");
}

#[test]
fn a_function_of_an_exact_inverse_trig_value_folds() {
    // `asin(1)`, `acos(0)`, `atan(1)` and `ln(−1)` keep their form (their
    // values are irrational), but a function of them is at a special
    // point: before 0.30 `tan(asin(1))/tan(asin(1))` was 1 and
    // `abs(tan(asin(1)))` was refused by `evalf` (found by the hunter).
    // SymPy: tan(2*atan(1)) → zoo, sin(2*asin(1)) → 0, cos(acos(0)) → 0,
    //   tan(asin(1))/tan(asin(1)) → nan, exp(log(-1)) → -1,
    //   exp(2*log(-1)) → 1, sinh(log(-1)) → 0, sin(3*acos(Rational(1,2))) → 0,
    //   tan(atan(oo)) → zoo, asin(1) → pi/2.
    let ctx = Context::new();
    for (s, want) in [
        ("tan(2*atan(1))", "zoo"),
        ("sin(2*asin(1))", "0"),
        ("cos(acos(0))", "0"),
        ("tan(asin(1))/tan(asin(1))", "nan"),
        ("exp(ln(-1))", "-1"),
        ("exp(2*ln(-1))", "1"),
        ("sinh(ln(-1))", "0"),
        ("sin(3*acos(1/2))", "0"),
        ("tan(atan(oo))", "zoo"),
        ("asin(1)", "asin(1)"),
    ] {
        assert_eq!(show(&p(&ctx, s)), want, "{s}");
    }
    // SymPy: atan(oo) → pi/2, atan(-oo) → -pi/2, atanh(oo) → -I*pi/2,
    //   atanh(-oo) → I*pi/2 (before 0.30 `eval` kept them and `evalf`
    //   refused them as infinite arguments).
    for (s, want) in [
        ("atan(oo)", "1/2*pi"),
        ("atan(-oo)", "-1/2*pi"),
        ("atanh(oo)", "-1/2*pi*I"),
        ("atanh(-oo)", "1/2*pi*I"),
    ] {
        assert_eq!(show(&p(&ctx, s).eval()), want, "{s}");
    }
}

#[test]
fn modulus_and_sign_of_exact_constants_fold() {
    // `abs(π)` and `abs(1/√i)` stayed atoms and hid a special point from
    // the function around them: `acosh(sin(|π|/2))` (acosh(1) = 0) and
    // `acos(asin(|1/√i|))` were refused by `evalf` (found by the hunter).
    // SymPy: Abs(pi) → pi, sign(-2*E) → -1, Abs(1/sqrt(I)) → 1,
    //   Abs((-1)**Rational(1,3)) → 1, Abs(exp(I*pi/3)) → 1,
    //   acosh(sin(Abs(pi)/2)) → 0, Abs(pi*I) → pi, Abs(-2*pi*I) → 2*pi,
    //   Abs(sqrt(2)*I) → sqrt(2), acosh(sin(Abs(pi*I)/2)) → 0.
    let ctx = Context::new();
    for (s, want) in [
        ("abs(pi)", "pi"),
        ("sign(-2*E)", "-1"),
        ("abs(1/sqrt(I))", "1"),
        ("abs((-1)^(1/3))", "1"),
        ("abs(exp(I*pi/3))", "1"),
        ("acosh(sin(abs(pi)/2))", "0"),
        ("abs(pi*I)", "pi"),
        ("abs(-2*pi*I)", "2*pi"),
        ("abs(sqrt(2)*I)", "sqrt(2)"),
        ("acosh(sin(abs(pi*I)/2))", "0"),
    ] {
        assert_eq!(show(&p(&ctx, s)), want, "{s}");
    }
    assert!(
        p(&ctx, "acos(asin(abs(1/sqrt(I))))")
            .eval_complex64()
            .is_ok()
    );
}

#[test]
fn every_path_that_builds_a_node_folds() {
    // Substitution rebuilds `sinh`, `atanh`, `Γ`, … by interning them
    // directly (not through the `Ex` constructors); the fold sits in the
    // interner, so they fold too.  SymPy: `(sinh(x)*gamma(x)).subs(x, 0)`
    // → nan (0·zoo), `atanh(x).subs(x, 1)` → oo.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.sinh() * x.gamma();
    assert_eq!(show(&e.subs(&x, &ctx.int(0))), "nan");
    assert_eq!(show(&x.atanh().subs(&x, &ctx.int(1))), "oo");
}

// ═══════════════════════════════════════════════════════════════════════════
// exp(ln 0), atan2(0, 0), 0^z
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exp_of_the_logarithm_of_zero_has_no_value() {
    // Before 0.30 `exp(ln(0))` evaluated to 0 (`eval` folded `exp(ln w) = w`
    // for every `w`, and `evalf` took the value of `w`).
    // SymPy: `exp(log(0))` → nan; `simplify((1+sqrt(2))**2-3-2*sqrt(2))` → 0.
    let ctx = Context::new();
    let e = p(&ctx, "exp(ln(0))");
    assert_eq!(show(&e), "nan");
    // A zero the canonical form does not see: `exp(ln w)` is not `w` when
    // `w` is not known non-zero, and `evalf` refuses (it printed 0).
    let hidden = p(&ctx, "(1+sqrt(2))^2-3-2*sqrt(2)");
    assert_eq!(hidden.eval_decimal(20).as_deref().ok(), Some("0"));
    let e = hidden.ln().exp();
    assert!(e.eval_decimal(20).is_err(), "{:?}", e.eval_decimal(20));
    assert!(e.eval_f64().is_err());
    // A constant known to be non-zero still folds, and a symbol is generic.
    // SymPy: `exp(log(pi))` → pi, `exp(log(x))` → x.
    assert_eq!(show(&p(&ctx, "exp(ln(pi))").eval()), "pi");
    assert_eq!(show(&p(&ctx, "exp(ln(x))").eval()), "x");
}

#[test]
fn atan2_of_the_origin_is_undefined() {
    // `eval` folded `atan2(0, 0)` to 0 "by convention"; the argument of 0
    // is undefined (`arg(0)` is nan already).  SymPy: `atan2(0, 0)` → nan.
    let ctx = Context::new();
    assert_eq!(show(&p(&ctx, "atan2(0, 0)")), "nan");
    let e = p(&ctx, "atan2(sin(pi/4) - cos(pi/4), 0)");
    assert_eq!(show(&e.eval()), "nan", "before 0.30: 0");
    // SymPy: `atan2(0, 1)` → 0, `atan2(0, -1)` → pi.
    assert_eq!(show(&p(&ctx, "atan2(0, 1)")), "0");
    assert_eq!(show(&p(&ctx, "atan2(0, -1)").eval()), "pi");
}

#[test]
fn zero_to_a_non_numeric_power() {
    // Before 0.30 these stayed as `0^π`, `0^I`, …, and `evalf` printed `0`
    // for `0^I` and `0^(−1 + I)` (and refused `0^π`).
    // SymPy: `0**pi` → 0, `0**(-pi)` → zoo, `0**I` → nan, `0**(1+I)` → nan,
    //   `0**(-1+I)` → nan; mpmath: `mpc(0)**mpc(1, 1)`, `mpc(0)**mpc(-1, 1)`,
    //   `mpc(0)**mpc(0, 1)` → (nan + nanj).
    let ctx = Context::new();
    for (s, want) in [
        ("0^pi", "0"),
        ("0^sqrt(2)", "0"),
        ("0^(-pi)", "zoo"),
        ("0^I", "nan"),
        ("0^(2*I)", "nan"),
        ("0^(1+I)", "nan"),
        ("0^(-1+I)", "nan"),
    ] {
        assert_eq!(show(&p(&ctx, s)), want, "{s}");
    }
    // An exponent of unknown sign keeps the power.
    assert_eq!(show(&p(&ctx, "0^x")), "0^x");
    let pos = ctx.symbol_with("q", &[Assumption::Positive]).unwrap();
    // SymPy: `0**Symbol('q', positive=True)` → 0.
    assert_eq!(show(&ctx.int(0).pow(&pos)), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// ±∞ times a factor of unknown sign
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infinity_times_a_factor_of_unknown_sign_is_complex_infinity() {
    // Canonical multiplication absorbed every factor into `±∞`: `x·∞` was
    // `∞`, so at `x = −1` it was `∞` (SymPy: `(x*oo).subs(x, -1)` → -oo),
    // and `i·∞` was `∞` (SymPy: `I*oo` → `oo*I`).  A factor of unknown sign
    // or not real now carries the direction: the product stays `x·∞`, `i·∞`
    // (it was `zoo` in between; see `v29_leftovers2::directed_infinities_*`).
    // Factors of known sign still orient it.
    // SymPy: `log(2)*oo` → oo, `(3-pi)*oo` → -oo, `pi*oo` → oo.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();
    assert_eq!(show(&(&x * &oo)), "x*oo");
    assert_eq!(show(&(&x * &oo).subs(&x, &ctx.int(-1))), "-oo");
    assert_eq!(show(&(ctx.i_unit() * &oo)), "I*oo");
    assert_eq!(show(&p(&ctx, "ln(2)*oo")), "oo");
    assert_eq!(show(&p(&ctx, "(3-pi)*oo")), "-oo");
    assert_eq!(show(&p(&ctx, "pi*oo")), "oo");
    let q = ctx.symbol_with("q", &[Assumption::Negative]).unwrap();
    assert_eq!(show(&(&q * &oo)), "-oo");
}

// ═══════════════════════════════════════════════════════════════════════════
// evalf: a zero ball is 0 only when it is below the requested digits
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_resolves_a_deep_cancellation_instead_of_printing_zero() {
    // `exp(exp(2ix)).rewrite_as_trig()` is `(i·sin s + cos s)·e^(cos 2x)` with
    // `s = sin 2x`; at `x = 1/3 + 4i`, `|cos s| ≈ 2.6·10⁵⁰⁸` cancels to
    // `e^(is) ≈ 10⁻⁵⁰⁹` (about 3,400 bits).  The search for a zero stopped
    // 1,024 bits past the working precision and printed the zero ball (of
    // radius 2²²²⁷) as `0`.  The rewrite is right: both forms agree.
    // mpmath: mp.dps = 1200; z = mpf(1)/3 + 4j; s = sin(2*z);
    //   (sin(s)*1j + cos(s))*exp(cos(2*z)) and exp(exp(2j*z))
    //   = 1.0002636490393672278816316222253 + 0.00020749465347554729972795772148078j
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = p(&ctx, "exp(exp(2*I*x))").rewrite_as_trig();
    let at = ctx.rational(1, 3) + ctx.i_unit() * 4;
    let v = r.subs(&x, &at).eval_complex64().expect("evaluates");
    let want = Complex64::new(1.000_263_649_039_367_2, 0.000_207_494_653_475_547_3);
    assert!((v - want).norm() < 1e-14, "{v}");
    let s = r.subs(&x, &at).eval_decimal(30).expect("evaluates");
    assert_eq!(
        s,
        "1.00026364903936722788163162223 + 0.000207494653475547299727957721481*i"
    );
}

/// Before: `sqrt(−oo)·(x·sin(f(x))·sqrt(−oo))` stayed `x·sin(f(x))·(−oo)`:
/// the two square roots combined to the factor `−∞`, which the product kept
/// as an ordinary factor, while its display parsed back as `zoo` (`±∞`
/// times factors of unknown sign; found by `fuzz_roundtrip`).  A combined
/// factor that is an infinity or `nan` now goes through the same rules as
/// one written directly.
#[test]
fn a_combined_infinite_factor_is_multiplied_like_a_written_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.apply("f", &[&x]).unwrap();
    let s = ctx.neg_infinity().sqrt();
    let e = &s * &(&(&x * &f.sin()) * &s);
    // `√(−∞) = i·∞` and the product is the directed infinity `−x·sin(f(x))·∞`
    // since the 0.30 leftovers (it was `zoo`).
    assert_eq!(e, -(&x * &f.sin() * ctx.infinity()));
    assert_eq!(ctx.parse(&e.to_string()).unwrap(), e);
    // sqrt(−oo)² = −oo, with a positive coefficient kept.
    assert_eq!(&(&ctx.int(2) * &s) * &s, ctx.neg_infinity());
}
