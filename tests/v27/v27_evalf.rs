//! Known bugs in the arbitrary-precision evaluator (`evalf`, 0.29): each
//! pinned by the value it now returns, against an mpmath 1.3 / SymPy 1.14
//! oracle quoted next to it (inputs exact: `mpf(1)/mpf(10)**30`, never a
//! decimal literal).

use std::time::{Duration, Instant};

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use symplex::linprog::Q;
use symplex::prelude::*;

fn rel_close(got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol * want.abs(),
        "got {got:e}, want {want:e} (rel tol {tol:e})"
    );
}

// ── Two zeros are not an agreement ──────────────────────────────────────────

/// `1 − e⁻¹·S` with `S = Σ_{k ≤ 60} 1/k!` as an exact rational (the
/// Poisson(1) tail beyond 60).  Before: `eval_f64` gave `0.0` — the value
/// cancelled to exactly 0 at 128 and 256 bits, and two zeros counted as
/// two evaluations that agree, so the 384-bit evaluation that certifies it
/// was never tried (`eval_decimal(20)` was right: it starts higher).
#[test]
fn a_value_that_cancels_to_zero_twice_is_resolved_at_the_cap() {
    let ctx = Context::new();
    let mut s = ctx.zero();
    let mut f = ctx.one();
    for k in 0..=60 {
        if k > 0 {
            f *= ctx.int(k);
        }
        s += ctx.one() / &f;
    }
    let s = s.eval();
    assert!(s.as_rational().is_some(), "S is an exact rational");
    let e = ctx.one() - (-ctx.one()).exp() * s;
    // mpmath: gammainc(61, 0, 1, regularized=True) = 7.366493966619451715322743729314e-85
    rel_close(e.eval_f64().unwrap(), 7.366_493_966_619_452e-85, 1e-15);
    assert_eq!(e.eval_decimal(20).unwrap(), "7.3664939666194517153e-85");
}

/// A true zero still comes out `0`: it cancels at every precision, reaches
/// the cap, and its error ball contains 0.
#[test]
fn true_zeros_are_still_zero() {
    let ctx = Context::new();
    for s in [
        "cos(pi/5) - (1 + sqrt(5))/4",
        "sin(pi/12) - (sqrt(6) - sqrt(2))/4",
        "atan(1/2) + atan(1/3) - pi/4",
        "ln(8) - 3*ln(2)",
    ] {
        let e = ctx.parse(s).unwrap();
        assert_eq!(e.eval_decimal(30).unwrap(), "0", "{s}");
        assert_eq!(e.eval_f64().unwrap(), 0.0, "{s}");
    }
}

// ── ln(exp(x)) for very negative x ──────────────────────────────────────────

/// Before: `ln(exp(−2601/10))` was `PrecisionExhausted { achieved: 0 }` (and
/// `−2401/10`, `−2501/10` lost digits): the bound of `exp z` was
/// `err(z)·max(1, |exp z|)`, the absolute error of the argument, so the
/// ball around `exp(−260.1) ≈ 2⁻³⁷⁵` contained 0 at the 384-bit cap and
/// `ln` of it had no bound.  `err(exp z) = |exp z|·err(z)`.
#[test]
fn ln_of_exp_of_a_very_negative_argument() {
    let ctx = Context::new();
    for (num, want) in [(-2401, -240.1), (-2601, -260.1), (-10001, -1000.1)] {
        let e = ctx.rational(num, 10).exp().ln();
        assert_eq!(e.eval_f64().unwrap(), want, "ln(exp({num}/10))");
    }
    // mpmath: exp(mpf(-2601)/10) = 1.096491468661601648553787296731717704181e-113
    assert_eq!(
        ctx.rational(-2601, 10).exp().eval_decimal(20).unwrap(),
        "1.0964914686616016486e-113"
    );
}

// ── Huge exact rationals ────────────────────────────────────────────────────

/// `7^28000 / 3^49500`, two coprime integers of ~78,500 bits as one exact
/// rational.
fn huge_ratio() -> Q {
    let n = num_traits::pow(BigInt::from(7), 28_000);
    let d = num_traits::pow(BigInt::from(3), 49_500);
    Q::new_raw(n, d)
}

/// Before: `bigint_to_bigfloat` accumulated the limbs one at a time at the
/// full width of the integer (a full-width multiplication per limb,
/// quadratic): `eval_f64` of the 86,000-bit p-value of
/// `binomial_test(2600, 20000, 3/20)` took 12 s (debug) where
/// `Ratio::to_f64` takes 70 µs.  The integer is now handed over as its
/// limbs and rounded once.  Timed against `Ratio::to_f64` of the same
/// rational over interleaved rounds (see `v27_perf.rs` for why): about 5×
/// now, four orders of magnitude before; bound 100×.
#[test]
fn a_huge_rational_converts_in_linear_time() {
    let ctx = Context::new();
    let q = huge_ratio();
    let x = ctx.from_ratio(q.clone());
    // mpmath (dps 50): mpf(7)**28000/mpf(3)**49500 = 1.7498941363840049270223501661484534e+45
    assert_eq!(
        x.eval_decimal(30).unwrap(),
        "1.74989413638400492702235016615e45"
    );
    let reference = q.to_f64().unwrap();
    rel_close(x.eval_f64().unwrap(), reference, 1e-15);
    let (mut t_eval, mut t_ref) = (Duration::ZERO, Duration::ZERO);
    for _ in 0..5 {
        let t = Instant::now();
        let v = x.eval_f64().unwrap();
        t_eval += t.elapsed();
        let t = Instant::now();
        let r = q.to_f64().unwrap();
        t_ref += t.elapsed();
        assert_eq!(v.to_bits(), r.to_bits());
    }
    assert!(
        t_eval < t_ref * 100,
        "eval_f64 {t_eval:?} vs Ratio::to_f64 {t_ref:?} (0.28: 10⁴×)"
    );
}

// ── Infinite sums ───────────────────────────────────────────────────────────

/// Before: `Unevaluable { "sub-expression … not in cache" }` for every
/// infinite `Sum` the summer could not close (the `∞` bound has no value).
/// A hypergeometric term is now summed with a rigorous tail bound.
#[test]
fn infinite_hypergeometric_sums_evaluate() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let oo = ctx.infinity();
    // A negative-binomial tail beyond 2000.
    // sympy: summation(binomial(k+2, k)*Rational(1,2)**k/8, (k, 2001, oo)).evalf(40)
    //   = 2.185081158727083423549769173923707518737e-597
    // (mpmath, dps 60, Σ_{k=2001}^{4000} directly: 2.1850811587270834235497691739237…e-597;
    //  mpmath's nsum returns 2.18292…e-597 here — its extrapolation is wrong on this tail)
    let tail = ((&k + ctx.int(2)).binomial(&k) * ctx.rational(1, 2).pow(&k) / ctx.int(8))
        .summation(&k, &ctx.int(2001), &oo);
    assert!(tail.to_string().starts_with("Sum("), "{tail}");
    assert_eq!(tail.eval_decimal(20).unwrap(), "2.1850811587270834235e-597");

    // mpmath: nsum(lambda k: (-1)**k/(fac(k)*(k+1)), [0, inf]) = 0.632120558828557678404476229839
    //   (= 1 − 1/e)
    let s =
        (ctx.int(-1).pow(&k) / (k.factorial() * (&k + ctx.one()))).summation(&k, &ctx.zero(), &oo);
    assert!(s.to_string().starts_with("Sum("), "{s}");
    assert_eq!(s.eval_decimal(25).unwrap(), "0.6321205588285576784044762");

    // mpmath: nsum(lambda k: binomial(2*k, k)/mpf(5)**k, [0, inf]) = 2.23606797749978969640917366873
    //   (= √5)
    let s = ((ctx.int(2) * &k).binomial(&k) / ctx.int(5).pow(&k)).summation(&k, &ctx.zero(), &oo);
    assert_eq!(s.eval_decimal(25).unwrap(), "2.236067977499789696409174");

    // mpmath: nsum(lambda k: 1/((k+1)*(k+2)*mpf(3)**k), [0, inf]) = 0.567209351351013708131921307214
    let s = (ctx.one() / ((&k + ctx.one()) * (&k + ctx.int(2)) * ctx.int(3).pow(&k))).summation(
        &k,
        &ctx.zero(),
        &oo,
    );
    assert_eq!(s.eval_decimal(25).unwrap(), "0.5672093513510137081319213");
    assert_eq!(s.eval_f64().unwrap(), 0.567_209_351_351_013_7);
}

/// A divergent sum is an error, not a number; a sum that converges only
/// polynomially, or whose term is not hypergeometric, has no rigorous
/// bound here and is refused.
#[test]
fn divergent_and_slowly_converging_sums_are_refused() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let oo = ctx.infinity();
    let sum = |body: Ex| body.summation(&k, &ctx.zero(), &oo);
    for body in [
        ctx.int(3).pow(&k) / (k.powi(2) + ctx.one()),
        k.factorial() / (k.powi(2) + ctx.one()),
        ctx.int(-1).pow(&k) * &k / (&k + ctx.one()),
    ] {
        let s = sum(body);
        assert!(
            matches!(s.eval_f64(), Err(SymplexError::Divergent { .. })),
            "{s}: {:?}",
            s.eval_f64()
        );
    }
    for body in [
        ctx.one() / (k.powi(2) + ctx.one()),
        ctx.int(-1).pow(&k) / (k.powi(2) + ctx.one()),
        k.sin() / ctx.int(2).pow(&k),
    ] {
        let s = sum(body);
        assert!(
            matches!(s.eval_f64(), Err(SymplexError::Unevaluable { .. })),
            "{s}: {:?}",
            s.eval_f64()
        );
    }
}

// ── RootOf: certified radius, certified realness ────────────────────────────

/// Before: `im(RootOf(y⁵ − y + 1, 0))` was `−4.04e-161` and `im(RootOf(y³ −
/// 3y + 1, 1))` `7.36e-152`, both "certified" to 30 digits: Aberth's noise
/// on a real root, trusted to the working precision.  Each root is now
/// bounded by a Newton inclusion disk, and a disk isolated from the others
/// together with its mirror image holds a real root.
#[test]
fn a_real_rootof_has_no_imaginary_part() {
    let ctx = Context::new();
    assert_eq!(
        ctx.parse("im(RootOf(y^5 - y + 1, 0))")
            .unwrap()
            .eval_decimal(30)
            .unwrap(),
        "0"
    );
    assert_eq!(
        ctx.parse("im(RootOf(y^3 - 3*y + 1, 1))")
            .unwrap()
            .eval_decimal(30)
            .unwrap(),
        "0"
    );
    // mpmath (dps 130): polyroots([1,0,0,0,-1,1], maxsteps=500, extraprec=500)
    //   [0] = -1.167303978261418684256045899854842180720560371525489039140082449275651903429527053180685205049728673
    //   [3] = -0.181232444469875383901800237781 - 1.08395410131771066843034449298j
    assert_eq!(
        ctx.parse("RootOf(y^5 - y + 1, 0)")
            .unwrap()
            .eval_decimal(30)
            .unwrap(),
        "-1.16730397826141868425604589985"
    );
    assert_eq!(
        ctx.parse("RootOf(y^5 - y + 1, 1)")
            .unwrap()
            .eval_decimal(30)
            .unwrap(),
        "-0.181232444469875383901800237781 - 1.08395410131771066843034449298*i"
    );
    // Beyond the Aberth tolerance (10⁻³⁰): the root is polished by Newton.
    assert_eq!(
        ctx.parse("RootOf(y^5 - y + 1, 0)")
            .unwrap()
            .eval_decimal(100)
            .unwrap(),
        "-1.167303978261418684256045899854842180720560371525489039140082449275651903429527053180685205049728673"
    );
    // mpmath: polyroots([1,0,-3,1]) → [-1.87938524157181676810821855465, 0.347296355333860697703433253539,
    //   1.53208888623795607040478530111]
    assert_eq!(
        ctx.parse("RootOf(y^3 - 3*y + 1, 1)")
            .unwrap()
            .eval_decimal(30)
            .unwrap(),
        "0.347296355333860697703433253539"
    );
}

// ── Piecewise decisions ─────────────────────────────────────────────────────

/// Before: at `x = √2 + 10⁻¹⁰⁰`, `Piecewise((0, x < √2), (1, True))` came
/// out `0` although the condition's `eval()` is False: the decision was
/// read off the rounded values.  A decision is now certain only when the
/// difference is outside its error ball (or both sides are exact); an
/// uncertain one sends the evaluation to a higher precision, and past the
/// budget it is refused.
#[test]
fn piecewise_decides_only_certified_conditions() {
    let ctx = Context::new();
    let r2 = ctx.parse("sqrt(2)").unwrap();
    let (zero, one, t) = (ctx.zero(), ctx.one(), ctx.bool_true());
    let near = |gap: &str| {
        let x = &r2 + &ctx.parse(gap).unwrap();
        let cond = x.lt(&r2);
        assert_eq!(cond.eval().to_string(), "False");
        Ex::piecewise(&[(&zero, &cond), (&one, &t)])
    };
    assert_eq!(near("10^(-100)").eval_decimal(20).unwrap(), "1");
    assert_eq!(near("10^(-100)").eval_f64().unwrap(), 1.0);
    // Beyond what twice the working precision plus 256 bits resolves.
    assert!(matches!(
        near("10^(-200)").eval_decimal(20),
        Err(SymplexError::PrecisionExhausted { .. })
    ));
    // Clear decisions are unaffected.
    let e = ctx.parse("exp(1)").unwrap();
    let pw = Ex::piecewise(&[(&e, &e.gt(&ctx.int(2))), (&one, &t)]);
    assert_eq!(pw.eval_decimal(20).unwrap(), "2.7182818284590452354");
}

/// Before: a value below astro-float's exponent range came out as an
/// *exact* 0 (its `exp` flags no inexactness), so a decision on it was
/// certain and wrong: `Piecewise((1, exp(−4·10⁹) > 0), (0, True))` was `0`
/// and `sign(exp(−10²⁰) − exp(−3·10²⁰))` was `0`.  An underflow now has
/// a bound (`2^EXPONENT_MIN`): the value alone is still `0` to the
/// precision reached, the decision is refused.
#[test]
fn an_underflow_is_not_an_exact_zero() {
    let ctx = Context::new();
    let tiny = ctx.parse("exp(-4*10^9)").unwrap();
    assert_eq!(tiny.eval_decimal(20).unwrap(), "0");
    let (zero, one, t) = (ctx.zero(), ctx.one(), ctx.bool_true());
    let pw = Ex::piecewise(&[(&one, &tiny.gt(&zero)), (&zero, &t)]);
    assert!(matches!(
        pw.eval_decimal(20),
        Err(SymplexError::PrecisionExhausted { .. })
    ));
    let s = ctx.parse("sign(exp(-10^20) - exp(-3*10^20))").unwrap();
    assert!(matches!(
        s.eval_decimal(20),
        Err(SymplexError::PrecisionExhausted { .. })
    ));
    assert_eq!(
        ctx.parse("1 + 2^(-10^20)")
            .unwrap()
            .eval_decimal(20)
            .unwrap(),
        "1"
    );
}

// ── lowergamma at irrational arguments ──────────────────────────────────────

/// Before: `γ(s, x)` was folded into `Γ(s) − Γ(s, x)` at an irrational `x`
/// even where that cancels catastrophically (the guard only read rational
/// `x`): `γ(5, √2/10³⁰)` came out `0` and `γ(11, π/10¹⁰)`
/// `PrecisionExhausted`.  The guard now sizes any real constant `x`.
#[test]
fn lowergamma_guard_covers_irrational_arguments() {
    let ctx = Context::new();
    let a = ctx.parse("sqrt(2)/10^30").unwrap().lowergamma(&ctx.int(5));
    assert!(
        a.eval().to_string().starts_with("lowergamma("),
        "{}",
        a.eval()
    );
    // mpmath: gammainc(5, 0, sqrt(2)/mpf(10)**30) = 1.131370849898476039041350979366425129522e-150
    rel_close(a.eval_f64().unwrap(), 1.131_370_849_898_476e-150, 1e-15);
    let b = ctx.parse("pi/10^10").unwrap().lowergamma(&ctx.int(11));
    // mpmath: gammainc(11, 0, pi/mpf(10)**10) = 2.674581980810599292227313303528924244866e-106
    assert_eq!(b.eval_decimal(20).unwrap(), "2.6745819808105992922e-106");
    // Where the closed form does not cancel it is still used.
    let c = ctx.parse("sqrt(2)").unwrap().lowergamma(&ctx.int(2));
    assert!(!c.eval().to_string().contains("lowergamma"), "{}", c.eval());
}
