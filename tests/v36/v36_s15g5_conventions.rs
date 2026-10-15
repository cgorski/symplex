//! Tests of session 15 (s15g5): printing complex values, binomial, query semantics, sampling seeds.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

// ── D3: printing values whose imaginary part is small next to the real part ──

/// `Chi(−60) = Chi(60) + iπ` (`Chi z = γ + ln z + ∫₀^z (cosh t − 1)/t dt`,
/// DLMF §6.2(ii), with the principal logarithm).  Up to 0.52 `eval_decimal(10)` printed only the
/// real part, `9.68091107e23`, `eval_f64` returned `9.680911069646383e23`
/// and `eval_complex64` had `im == 0.0`: the imaginary part `π` lies below
/// the 10th (and 16th) digit of the real part, but it is certified nonzero.
/// Oracle: mpmath `chi(-60)` = `9.68091106964638269410362983438e23 +
/// 3.14159265358979323846264338328j` (SymPy `N(Chi(-60))` agrees).
#[test]
fn chi_at_minus_60_shows_its_imaginary_part() {
    let ctx = Context::new();
    let chi = parse(&ctx, "Chi(-60)");
    assert_eq!(
        chi.eval_decimal(10).unwrap(),
        "9.68091107e23 + 3.141592654*i"
    );
    assert_eq!(
        chi.eval_decimal(30).unwrap(),
        "968091106964638269410362.983438 + 3.14159265358979323846264338328*i"
    );
    let z = chi.eval_complex64().unwrap();
    assert_eq!(z.re, 9.680911069646383e23);
    assert_eq!(z.im, std::f64::consts::PI);
    assert!(matches!(
        chi.eval_f64(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    // `compile` refuses a constant that is not real, as for `asin(2)`; up to
    // 0.52 it folded `Chi(−60)` to its real part (`f(1) = 9.68·10²³ + 1`).
    let f = parse(&ctx, "Chi(-60) + x").compile(&["x"]);
    assert!(matches!(f, Err(SymplexError::NotImplemented(_))));
}

/// `E₁(−60) = −Ei(60) − iπ` (`E₁(z) = −γ − ln z − Σ (−z)ⁿ/(n·n!)`, DLMF
/// §6.6, principal logarithm) and `Γ(0, z) = E₁(z)` (DLMF §8.4); up to
/// 0.52 both printed `-1.936182214e24` at 10 digits.
/// Oracle: mpmath `expint(1, -60)` = `gammainc(0, -60)` =
/// `-1.93618221392927653882072596688e24 - 3.14159265358979323846264338328j`.
#[test]
fn exponential_integral_at_minus_60_shows_its_imaginary_part() {
    let ctx = Context::new();
    for s in ["expint(1, -60)", "uppergamma(0, -60)"] {
        let e = parse(&ctx, s);
        assert_eq!(
            e.eval_decimal(10).unwrap(),
            "-1.936182214e24 - 3.141592654*i",
            "{s}"
        );
        assert_eq!(e.eval_complex64().unwrap().im, -std::f64::consts::PI, "{s}");
        assert!(e.eval_f64().is_err(), "{s}");
    }
}

/// An exact imaginary unit next to a huge real part, and an exact small
/// real part next to a huge imaginary one.  Up to 0.52: `10³⁰ + i` printed
/// `1e30` at 10 digits and was `1e30 + 0i` as a `Complex64`; `1 + 10³⁰·i`
/// printed `1e30*i` and had `re == 0.0`; `iπ + 10⁻³⁰` printed
/// `3.141592654*i`.  Oracle: the exact values.
#[test]
fn exact_parts_are_printed_however_small() {
    let ctx = Context::new();
    let a = parse(&ctx, "10^30 + I");
    assert_eq!(a.eval_decimal(10).unwrap(), "1e30 + i");
    assert_eq!(a.eval_complex64().unwrap(), Complex64::new(1e30, 1.0));
    let b = parse(&ctx, "I*10^30 + 1");
    assert_eq!(b.eval_decimal(10).unwrap(), "1 + 1e30*i");
    assert_eq!(b.eval_complex64().unwrap(), Complex64::new(1.0, 1e30));
    let c = parse(&ctx, "log(-1) + 10^(-30)");
    assert_eq!(c.eval_decimal(10).unwrap(), "1e-30 + 3.141592654*i");
}

/// A certified imaginary part of `2·10⁻²⁰`: `(1 + i·10⁻²⁰)² = 1 − 10⁻⁴⁰ +
/// 2·10⁻²⁰·i` (exact).  Up to 0.52 `eval_f64` returned `1.0` and
/// `eval_complex64` had `im == 0.0`; `eval_decimal(10)` printed `1`.  So
/// for `e⁴⁰ + i·sin(10⁻²⁰)` (`im = 10⁻²⁰ − 10⁻⁶⁰/6`, mpmath).
#[test]
fn eval_f64_refuses_a_tiny_certified_imaginary_part() {
    let ctx = Context::new();
    let e = parse(&ctx, "(1 + I*10^(-20))^2");
    assert!(matches!(
        e.eval_f64(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert_eq!(e.eval_complex64().unwrap(), Complex64::new(1.0, 2e-20));
    assert_eq!(e.eval_decimal(10).unwrap(), "1 + 2e-20*i");
    let f = parse(&ctx, "exp(40) + I*sin(10^(-20))");
    assert_eq!(f.eval_decimal(10).unwrap(), "2.353852668e17 + 1e-20*i");
    assert!(f.eval_f64().is_err());
}

/// An imaginary part that is 0 within its error stays absent: these values
/// are real (exactly: `(e^{iπ/7})⁷ = −1`, `Chi(−60) − iπ = Chi(60)`,
/// `√−2² = −2`, `cos(i) = cosh 1`), and must not print or return a
/// spurious imaginary part.  Oracle: the exact identities; mpmath `chi(60)`
/// = `9.68091106964638269410362983438e23`, `cosh(1)` = `1.54308063481524`.
#[test]
fn imaginary_parts_consistent_with_zero_stay_absent() {
    let ctx = Context::new();
    let cases = [
        ("exp(I*pi/7)^7", "-1", -1.0),
        ("Chi(-60) - I*pi", "9.68091107e23", 9.680911069646383e23),
        ("sqrt(-2)^2", "-2", -2.0),
        ("cos(I)", "1.543080635", 1.5430806348152437),
    ];
    for (s, d10, f) in cases {
        let e = parse(&ctx, s);
        assert_eq!(e.eval_decimal(10).unwrap(), d10, "{s}");
        assert_eq!(e.eval_f64().unwrap(), f, "{s}");
        assert_eq!(e.eval_complex64().unwrap().im, 0.0, "{s}");
    }
}

/// `(−r)ⁿ` for an exact integer `n` beyond `i32` is real, as the bound of the
/// power says (an exact imaginary part); it was formed as `exp(n·Log b)`, whose
/// imaginary part is the rounding residue of `n·π` (`2.8·10⁻²²` at 10
/// digits, `1.6·10⁻⁶⁰` at 15).  With that bound the residue counted as a
/// certified part: hidden up to 0.52 by the rule that dropped negligible
/// parts, it would print under D3 (`-7.389056099 + 2.814264205e-22*i`, and
/// `PrecisionExhausted` at 16 digits).  Now `(−1)ⁿ·rⁿ`.  Oracle: mpmath
/// `-exp((2**53 + 1)*log(1 + mpf(2)**-52))` = `-7.3890560989306502272`
/// (dps 60 and 90).
#[test]
fn odd_powers_of_a_negative_base_beyond_i32_are_real() {
    let ctx = Context::new();
    let e = parse(&ctx, "(-(1 + 2^-52))^9007199254740993");
    assert_eq!(e.eval_decimal(10).unwrap(), "-7.389056099");
    assert_eq!(e.eval_decimal(20).unwrap(), "-7.3890560989306502272");
    assert_eq!(e.eval_f64().unwrap(), -7.38905609893065);
    assert_eq!(e.eval_complex64().unwrap().im, 0.0);
    assert_eq!(
        parse(&ctx, "(-2)^(10^17 + 1)/2^(10^17 + 1)")
            .eval_decimal(10)
            .unwrap(),
        "-1"
    );
}

// ── D4: binomial(n, k) for a negative integer k is 0 ──

/// `C(n, k) = 0` for a negative integer `k` and every `n` (Graham, Knuth,
/// Patashnik, *Concrete Mathematics* (5.1); SymPy 1.14 `binomial(-2, -4)`
/// = 0, `binomial(n, -1)` = 0).  Up to 0.52 `eval` kept `C(−2, −4)`,
/// `C(−7, −9)`, `C(−1, −1)`, `C(n, −1)` and `C(i, −2)`, while `eval_f64`
/// gave mpmath's limit along `n − k` fixed (`3`, `28`, `1`) and `compile`
/// gave 0.
#[test]
fn binomial_at_a_negative_integer_k_is_zero_everywhere() {
    let ctx = Context::new();
    for (n, k) in [
        ("-2", "-4"),
        ("-7", "-9"),
        ("-1", "-1"),
        ("-3", "-3"),
        ("-4", "-2"),
        ("I", "-2"),
        ("pi", "-1"),
    ] {
        let e = parse(&ctx, &format!("binomial({n}, {k})"));
        assert_eq!(e.eval(), ctx.int(0), "C({n}, {k})");
        assert_eq!(e.eval_f64().unwrap(), 0.0, "C({n}, {k})");
        assert_eq!(e.eval_decimal(10).unwrap(), "0", "C({n}, {k})");
    }
    assert_eq!(parse(&ctx, "binomial(n, -1)").eval(), ctx.int(0));
    let m = ctx
        .symbol_with("m", &[Assumption::Integer, Assumption::Negative])
        .unwrap();
    assert_eq!(ctx.symbol("n").binomial(&m).eval(), ctx.int(0));
    // A constant `n` without a value (`min` of a non-real `E₂(−3)`) leaves
    // the coefficient without one: not folded to 0.
    let undefined = parse(&ctx, "binomial(Min(sqrt(2), 4, expint(2, -3)), -1)");
    assert_ne!(undefined.eval(), ctx.int(0));
    assert!(undefined.eval_decimal(10).is_err());
    // `compile` agreed already; it still does.
    let f = parse(&ctx, "binomial(x, -4)").compile(&["x"]).unwrap();
    assert_eq!(f.call(&[-2.0]), 0.0);
    let g = parse(&ctx, "binomial(-2, x)").compile(&["x"]).unwrap();
    assert_eq!(g.call(&[-4.0]), 0.0);
}

/// Pascal's rule `C(r, k) = C(r − 1, k) + C(r − 1, k − 1)` holds for every
/// pair of integers under the convention (GKP (5.8)); the symmetry `C(n, k)
/// = C(n, n − k)` only for `n ≥ 0` (`C(−2, −4) = 0`, `C(−2, 2) = 3`).  Up to
/// 0.52 the rule failed where `eval` kept a coefficient at two negative
/// integers (`C(−1, −1)` against `C(−2, −1) + C(−2, −2)`).  Oracle: the
/// integer identities.
#[test]
fn binomial_keeps_pascals_rule_for_all_integers() {
    let ctx = Context::new();
    let c = |n: i64, k: i64| ctx.int(n).binomial(&ctx.int(k)).eval();
    for r in -6..=6 {
        for k in -6..=8 {
            let lhs = c(r, k);
            let rhs = (&c(r - 1, k) + &c(r - 1, k - 1)).eval();
            assert_eq!(lhs, rhs, "Pascal at ({r}, {k})");
            assert!(lhs.as_rational().is_some(), "C({r}, {k}) = {lhs}");
        }
    }
    for n in 0..=6 {
        for k in -3..=9 {
            assert_eq!(c(n, k), c(n, n - k), "symmetry at ({n}, {k})");
        }
    }
    assert_eq!(c(-2, -4), ctx.int(0));
    assert_eq!(c(-2, 2), ctx.int(3));
}

/// `Σ_{k≥0} C(k − 3, −3)/2^k`: every term is 0 under the convention, so the
/// sum is 0.  Up to 0.52 the terms `C(−3, −3)`, `C(−2, −3)`, `C(−1, −3)`
/// were kept by `eval` (`evalf`: 1, −2, 1) and the infinite sum was `1/4`,
/// from the closed form of `Σ C(k + c, k)·x^k` applied through the symmetry
/// `C(k + c, c) = C(k + c, k)`, which fails at `c = −3`.  Oracle: SymPy 1.14
/// `summation(binomial(k - 3, -3)/2**k, (k, 0, oo))` = 0.
#[test]
fn sums_of_binomials_at_a_negative_lower_index() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = parse(&ctx, "binomial(k - 3, -3)*(1/2)^k");
    assert_eq!(body.summation(&k, &ctx.int(0), &ctx.infinity()), ctx.int(0));
    assert_eq!(body.summation(&k, &ctx.int(0), &ctx.int(5)), ctx.int(0));
    // The shape `C(k + c, k)` keeps its closed form: Σ C(k − 3, k)/2^k =
    // (1 − 1/2)² = 1/4 (terms 1, −1, 1/4, 0, …).
    let other = parse(&ctx, "binomial(k - 3, k)*(1/2)^k");
    assert_eq!(
        other.summation(&k, &ctx.int(0), &ctx.infinity()),
        ctx.rational(1, 4)
    );
}

// ── D7: is_finite is the pole detector; realness reads the natural domain ──

/// `is_finite` of an expression with symbols is about every point of the
/// symbols' domain: a pole there makes it undecided.  Up to 0.52 the
/// natural-domain reading (`real` implies `finite`) called `1/x`, `x⁻²`,
/// `tan x`, `ln x`, `1/(x − 1)`, `Γ(x)`, `x/(x + 2)`, `e^{1/x}`, `sin(1/x)`
/// and `1/sin x` finite for a real `x`, `1/n` and `Γ(n)` for an integer
/// `n`, and `atan z` for a complex `z` (poles at `±i`).  Realness keeps the
/// natural-domain reading (`(1/x).is_real()` is `Some(true)`; SymPy 1.14:
/// `None`, `is_extended_real` `True`).  Oracle: the poles of the functions
/// (`tan`, `Γ`, `atan`: DLMF §4.14, §5.2, §4.23).
#[test]
fn is_finite_does_not_call_a_pole_finite() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let one = ctx.one();
    let poles = [
        &one / &x,
        x.powi(-2),
        x.tan(),
        x.ln(),
        &one / (&x - 1),
        x.gamma(),
        &x / (&x + 2),
        (&one / &x).exp(),
        (&one / &x).sin(),
        &one / &x.sin(),
    ];
    for e in &poles {
        assert_eq!(e.is_finite(), None, "{e}");
        assert_eq!(ctx.query(e, Props::FINITE), None, "{e}");
        assert_eq!(e.query(Props::INFINITE), None, "{e}");
    }
    assert_eq!((&one / &x).is_real(), Some(true));
    // No pole on the domain.
    for e in [
        &one / (&x.powi(2) + 1),
        x.sqrt(),
        x.atan(),
        x.exp(),
        (&x.powi(2) + 1).ln(),
    ] {
        assert_eq!(e.is_finite(), Some(true), "{e}");
    }
    let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    for e in [
        &one / &p,
        p.ln(),
        p.gamma(),
        &p / (&p + 2),
        (&one / &p).exp(),
    ] {
        assert_eq!(e.is_finite(), Some(true), "{e}");
        assert_eq!(e.query(Props::INFINITE), Some(false), "{e}");
    }
    assert_eq!(p.tan().is_finite(), None);
    let n = ctx.symbol_with("n", &[Assumption::Integer]).unwrap();
    assert_eq!((&one / &n).is_finite(), None);
    assert_eq!(n.gamma().is_finite(), None);
    let z = ctx.symbol_with("z", &[Assumption::Complex]).unwrap();
    assert_eq!(z.atan().is_finite(), None);
    // Constants: no "elsewhere".
    assert_eq!(parse(&ctx, "1/0").is_finite(), Some(false));
    assert_eq!(parse(&ctx, "tan(1)").is_finite(), Some(true));
}

// ── S8: sampling seeds from structure, not from ids ──

/// `probably_equal` samples rational points from a generator seeded by the
/// two expressions.  Up to 0.52 the seed mixed their ids, so a verdict that
/// depends on the sample (`√(x²) = x` holds at positive points only) changed
/// with the order in which `x` and `y` had been created and with what the
/// context had interned before: `TTFTTFFFFTFFTFF` against `TTFFFFFFFFFTFFT`
/// for the fifteen verdicts below.  Oracle: the verdicts are equal in every
/// context (and `Some(false)` is right whenever returned: each identity
/// fails somewhere).
#[test]
fn probably_equal_does_not_depend_on_creation_order() {
    let verdicts = |x_first: bool, noise: bool| -> String {
        let ctx = Context::new();
        if noise {
            for i in 0..37 {
                let s = ctx.symbol(&format!("zz{i}"));
                let _ = (&s + i).sin() * &s;
            }
        }
        let (x, y) = if x_first {
            let x = ctx.symbol("x");
            (x, ctx.symbol("y"))
        } else {
            let y = ctx.symbol("y");
            (ctx.symbol("x"), y)
        };
        let o = |v: Option<bool>| match v {
            Some(true) => 'T',
            Some(false) => 'F',
            None => '-',
        };
        let mut out = String::new();
        for samples in [1usize, 2, 3] {
            out.push(o(x.powi(2).sqrt().probably_equal(&x, samples)));
            out.push(o((&x * &y)
                .powi(2)
                .sqrt()
                .probably_equal(&(&x * &y), samples)));
            out.push(o(x.abs().probably_equal(&x, samples)));
            out.push(o((&x - &y).abs().probably_equal(&(&y - &x), samples)));
            out.push(o((&x / &y)
                .ln()
                .probably_equal(&(x.ln() - y.ln()), samples)));
        }
        out
    };
    let reference = verdicts(true, false);
    for (x_first, noise) in [(false, false), (true, true), (false, true)] {
        assert_eq!(
            verdicts(x_first, noise),
            reference,
            "x first {x_first}, noise {noise}"
        );
    }
}
