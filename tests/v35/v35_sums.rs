//! Symbolic sums, products and convergence decisions (the summation hunt
//! after 0.43).  Every reference value cites its oracle: the finite sum or
//! product computed term by term, mpmath 1.3 `nsum` at 40 digits for an
//! infinite sum, SymPy 1.14 `summation` / `product` / `Sum.is_convergent`
//! as a second opinion (it is wrong in two places noted below).

use symplex::prelude::*;

fn pos(ctx: &Context, name: &str) -> Ex {
    ctx.symbol_with(name, &[Assumption::Positive]).unwrap()
}

fn int_sym(ctx: &Context, name: &str, extra: Assumption) -> Ex {
    ctx.symbol_with(name, &[Assumption::Integer, extra])
        .unwrap()
}

/// `got` equals the value `want` (parsed), to 1e-12 relative.
fn assert_close(got: &Ex, want: &str) {
    let ctx = got.context();
    let w = ctx.parse(want).unwrap().eval_complex64().unwrap();
    let g = got
        .eval_complex64()
        .unwrap_or_else(|e| panic!("{got} does not evaluate: {e}"));
    let scale = 1f64.max(w.norm());
    assert!(
        (g - w).norm() <= 1e-12 * scale,
        "{got} = {g} (want {want} = {w})"
    );
}

/// `Σ_{k=lo}^{hi} body` by terms.
fn direct_sum(body: &Ex, k: &Ex, lo: i64, hi: i64) -> Ex {
    let ctx = body.context();
    let mut acc = ctx.int(0);
    for j in lo..=hi {
        acc = (&acc + &body.subs_i64(k, j)).eval();
    }
    acc
}

/// Before: a geometric sum with a symbolic ratio `y` always came out as
/// `Piecewise((…, y = 1), (…, y ≠ 1))`, also when the declared assumptions
/// rule out `y = 1`: `Σ_{k=0}^{n} k·(−a)^k` for a positive `a` carried the
/// branch `−a = 1`, `Σ (a + 1)^k` the branch `a + 1 = 1`, `Σ (a/(a + 1))^k`
/// a branch that no `a` reaches, and `(2^a)^k`, `exp(a)^k` the branches
/// `2^a = 1`, `exp(a) = 1` (only `a = 2πin/ln 2`, `2πin`, not positive).
/// For a positive integer `p`, `2p = 1` is impossible too.  Ratios that can
/// be 1 (`2a`, `a`, an unassumed `x`) keep the case split.
///
/// Oracle: the sums term by term at `n = 0..6`, `a = 2`, `p = 3`; SymPy
/// 1.14 `summation(k*(-a)**k, (k, 0, n))` (a positive) has no case split.
#[test]
fn geometric_ratio_that_cannot_be_one_has_no_case_split() {
    let ctx = Context::new();
    let k = ctx.symbol_with("k", &[Assumption::Integer]).unwrap();
    let n = int_sym(&ctx, "n", Assumption::NonNegative);
    let p = int_sym(&ctx, "p", Assumption::Positive);
    let a = pos(&ctx, "a");
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let zero = ctx.int(0);
    let unreachable = [
        &k * &(-&a).pow(&k),
        (&a + &one).pow(&k),
        (&a / &(&a + &one)).pow(&k),
        ctx.int(2).pow(&a).pow(&k),
        a.exp().pow(&k),
        (&a * &ctx.int(-1) - &one).pow(&k) * &k,
        (&ctx.int(2) * &p).pow(&k),
        (&p + &one).pow(&k),
    ];
    for body in &unreachable {
        let s = body.summation(&k, &zero, &n);
        assert!(!s.to_string().contains("Piecewise"), "Σ {body} = {s}");
        for nv in 0..=6 {
            let at = |e: &Ex| e.subs_i64(&n, nv).subs_i64(&a, 2).subs_i64(&p, 3).eval();
            let want = direct_sum(&body.subs_i64(&a, 2).subs_i64(&p, 3), &k, 0, nv);
            assert_close(&at(&s), &want.to_string());
        }
    }
    for body in [(&ctx.int(2) * &a).pow(&k), a.pow(&k), x.pow(&k)] {
        let s = body.summation(&k, &zero, &n);
        assert!(s.to_string().contains("Piecewise"), "Σ {body} = {s}");
    }
    // the reachable branch is right where it applies (2a = 1)
    let s = (&ctx.int(2) * &a).pow(&k).summation(&k, &zero, &n);
    let half = ctx.rational(1, 2);
    assert_eq!(s.subs(&a, &half).subs_i64(&n, 4).eval().to_string(), "5");
}

/// Before: `Σ_{k=0}^{n} C(n, k)·(−1)^k` for a positive `n` was
/// `Piecewise((1, n = 0), (0, n ≠ 0))`, with the unreachable branch
/// `n = 0`; `Σ (−1)^k·P(k)·C(n, k)` with a polynomial `P` of degree ≥ 1
/// (and a factor besides `k`) was refused.  It is `(−1)^j·j!·s_j` at
/// `n = j ≤ deg P` (`s_j` the falling-factorial coefficients) and 0 beyond.
///
/// Oracle: the sums term by term at `n = 0..6`.  SymPy 1.14 gives `0` for
/// `summation((-1)**k*(k + 2)*binomial(n, k), (k, 0, n))`, wrong at `n = 0`
/// (2) and `n = 1` (−1).
#[test]
fn alternating_binomial_sums_by_cases() {
    let ctx = Context::new();
    let k = ctx.symbol_with("k", &[Assumption::Integer]).unwrap();
    let n = int_sym(&ctx, "n", Assumption::NonNegative);
    let p = int_sym(&ctx, "p", Assumption::Positive);
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let m1 = ctx.int(-1);
    let s = (p.binomial(&k) * m1.pow(&k)).summation(&k, &zero, &p);
    assert_eq!(s.to_string(), "0");
    let s = (n.binomial(&k) * m1.pow(&k)).summation(&k, &zero, &n);
    assert!(s.to_string().contains("Piecewise"), "{s}");
    let bodies = [
        m1.pow(&k) * (&k + &ctx.int(2)) * n.binomial(&k),
        m1.pow(&k) * k.powi(2) * n.binomial(&k),
        m1.pow(&k) * (k.powi(3) - &k + &one) * n.binomial(&k),
    ];
    for body in &bodies {
        for lo in [0, 1] {
            let s = body.summation(&k, &ctx.int(lo), &n);
            assert!(!s.has_unevaluated(), "Σ {body} = {s}");
            for nv in 0..=6 {
                let want = direct_sum(&body.subs_i64(&n, nv), &k, lo, nv);
                assert_eq!(
                    s.subs_i64(&n, nv).eval(),
                    want,
                    "Σ_{{k={lo}}}^{nv} {body}: {s}"
                );
            }
        }
    }
    // n positive: the case n = 0 is dropped
    let body = m1.pow(&k) * (&k + &ctx.int(2)) * p.binomial(&k);
    let s = body.summation(&k, &zero, &p);
    assert!(!s.to_string().contains("p == 0"), "{s}");
}

/// Before: `Π f(k)^e = (Π f(k))^e` was applied for every exponent, which
/// holds for principal powers only when the factors are positive:
/// `Π_{k=1}^{n} √(k − 3)` came out `√(rising_factorial(−2, n))`, `√2` at
/// `n = 2` where the product is `√(−2)·√(−1) = −√2` (SymPy 1.14 has the
/// same flaw), and `Π_{k=1}^{n} √(−k)` was `√((−1)^n·n!)`.  A factor
/// negative over the whole range takes the phase `((−1)^e)^count`; mixed
/// signs stay unevaluated.
///
/// Oracle: the products factor by factor.  (Up to 0.46 this cited SymPy
/// 1.14 for `product(sqrt(-k), (k, 1, n))` → `I**n*sqrt(factorial(n))`;
/// SymPy 1.14 returns `sqrt((-1)**n*factorial(n))`, the form symplex had,
/// which is `√2` at `n = 2` where the product is `−√2`.)
#[test]
fn products_of_roots_respect_principal_branches() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = int_sym(&ctx, "n", Assumption::NonNegative);
    let one = ctx.int(1);
    let p = (&k - &ctx.int(3)).sqrt().product_over(&k, &one, &n);
    assert!(p.has_unevaluated(), "{p}");
    let p = (&k - &ctx.int(3))
        .pow(&ctx.rational(1, 3))
        .product_over(&k, &one, &n);
    assert!(p.has_unevaluated(), "{p}");
    let body = (-&k).sqrt();
    let p = body.product_over(&k, &one, &n);
    assert!(!p.has_unevaluated(), "{p}");
    for nv in 0..=5 {
        let want = body.product_over(&k, &one, &ctx.int(nv));
        let got = p.subs_i64(&n, nv).eval();
        assert_close(&got, &want.to_string());
    }
    assert_eq!(k.sqrt().product_over(&k, &one, &n).to_string(), "sqrt(n!)");
    // positive factors: the Gamma ratio cancels inside the root
    let body = (&one - &(&one / &(&k + &one)).powi(2)).sqrt();
    let p = body.product_over(&k, &one, &n);
    assert_eq!(p.to_string(), "sqrt((n + 2)/(2*(n + 1)))");
}

/// Before: `is_convergent` stripped `sin`/`cos` factors as bounded by 1
/// whatever their argument, so `Σ sin((1 + i)k)/k²` (terms growing like
/// `e^k/(2k²)`) and `Σ sin(xk)/k²` for an unassumed (complex) `x` were
/// reported convergent.  A symbolic factor that may vanish no longer
/// makes a divergence claim: `Σ x^k·k!` and `Σ x` converge at `x = 0`.
///
/// Oracle: `|sin((1 + i)k)| ≥ sinh(k)`; SymPy 1.14 raises
/// `NotImplementedError` for both trig cases.
#[test]
fn convergence_claims_hold_for_every_parameter_value() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let x = ctx.symbol("x");
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let a = pos(&ctx, "a");
    let i = ctx.i_unit();
    let one = ctx.int(1);
    let k2 = k.powi(2);
    assert_eq!((((&one + &i) * &k).sin() / &k2).is_convergent(&k), None);
    assert_eq!(((&x * &k).sin() / &k2).is_convergent(&k), None);
    assert_eq!(((&r * &k).sin() / &k2).is_convergent(&k), Some(true));
    assert_eq!(
        ((&ctx.int(2) * &k).cos() / &k2).is_convergent(&k),
        Some(true)
    );
    assert_eq!((x.pow(&k) * k.factorial()).is_convergent(&k), None);
    assert_eq!((a.pow(&k) * k.factorial()).is_convergent(&k), Some(false));
    assert_eq!(x.is_convergent(&k), None);
    assert_eq!(a.is_convergent(&k), Some(false));
    let inf = ctx.infinity();
    assert!(!matches!(
        (&x / &k).try_summation(&k, &one, &inf),
        Err(SymplexError::Divergent { .. })
    ));
    assert!(matches!(
        (&a / &k).try_summation(&k, &one, &inf),
        Err(SymplexError::Divergent { .. })
    ));
}

/// Before: undecided.  The alternating-series test for `(−1)^k·k^a·ln(k)^b`
/// and the limit comparison with `1/k` and `1/k^p` for terms in a Hardy
/// field (`sin`, `cos`, `exp`, … of arguments tending to 0).
///
/// Oracle: SymPy 1.14 `Sum(…, (k, 2, oo)).is_convergent()`: `(-1)**k/log(k)`
/// → `True`, `sin(1/k)` → `False`, `sin(k**-2)` → `True`,
/// `sin(1/sqrt(k))` → `False`; `1 − cos(1/k) ~ 1/(2k²)`, `e^{1/k} − 1 ~ 1/k`.
#[test]
fn convergence_by_alternation_and_limit_comparison() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let one = ctx.int(1);
    let m1 = ctx.int(-1);
    let inv_k = &one / &k;
    assert_eq!((m1.pow(&k) / k.ln()).is_convergent(&k), Some(true));
    assert_eq!(
        (m1.pow(&k) / k.ln()).is_absolutely_convergent(&k),
        Some(false)
    );
    assert_eq!((m1.pow(&k) * k.ln() / &k).is_convergent(&k), Some(true));
    assert_eq!((m1.pow(&k) * k.ln()).is_convergent(&k), Some(false));
    assert_eq!(inv_k.sin().is_convergent(&k), Some(false));
    assert_eq!(k.powi(-2).sin().is_convergent(&k), Some(true));
    assert_eq!(k.sqrt().pow(&m1).sin().is_convergent(&k), Some(false));
    assert_eq!((&one - &inv_k.cos()).is_convergent(&k), Some(true));
    assert_eq!((inv_k.exp() - &one).is_convergent(&k), Some(false));
}

/// Before: refused.  Alternating rational functions sum pole by pole
/// (each `(−1)^k/(k + β)` an alternating p-series); a geometric ratio
/// whose modulus the declared assumptions bound (`1/(a + 1)` for a positive
/// `a`) or a non-real constant ratio (`i/2`) decides convergence.
///
/// Oracle: mpmath `nsum` at 40 digits; SymPy 1.14: `-1 + 2*log(2)`,
/// `-19/18 + 4*log(2)/3`, `1/((1 - 1/(a + 1))*(a + 1))` (= `1/a`).
#[test]
fn alternating_rational_and_bounded_ratio_series() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let a = pos(&ctx, "a");
    let one = ctx.int(1);
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let m1 = ctx.int(-1);
    let s = (m1.pow(&k) / (&(&k + &one) * &(&k + &ctx.int(2)))).summation(&k, &zero, &inf);
    assert_eq!(s.to_string(), "2*ln(2) - 1");
    let s =
        (ctx.int(2) * m1.pow(&k) / (&(&k + &one) * &(&k + &ctx.int(4)))).summation(&k, &one, &inf);
    assert_close(&s, "4*log(2)/3 - 19/18");
    let s =
        (m1.pow(&k) * (&k + &ctx.int(3)) / (k.powi(2) * &(&k + &one))).summation(&k, &one, &inf);
    // 3/k² − 2/k + 2/(k + 1): −π²/4 + 2·ln 2 + 2·(ln 2 − 1); nsum −1.6948123780325584
    assert_close(&s, "-pi^2/4 + 4*log(2) - 2");
    let s = (&one / &(&a + &one)).pow(&k).summation(&k, &one, &inf);
    assert_eq!(s.to_string(), "1/a");
    let s = (&a + &one).pow(&k).summation(&k, &zero, &inf);
    assert_eq!(s.to_string(), "oo");
    let i = ctx.i_unit();
    let s = (&i / &ctx.int(2)).pow(&k).summation(&k, &zero, &inf);
    assert_close(&s, "4/5 + 2*I/5");
}

/// Before: refused.  `Σ x^k/(k + β)` (integer `β`) is the logarithm series
/// after an index shift; a negative numeric base is an alternation also
/// for the series in `x²` (`cos`, `sin`); `Σ C(2k, k)·x^k = 1/√(1 − 4x)`.
///
/// Oracle: mpmath `nsum`; SymPy 1.14: `2*log(2)`, `-10/3 + 8*log(3/2)`,
/// `cos(sqrt(2))`, `sqrt(2)*sin(sqrt(2))/2`, `sqrt(2)`.
#[test]
fn power_series_after_shifts_and_sign_changes() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let one = ctx.int(1);
    let zero = ctx.int(0);
    let two = ctx.int(2);
    let inf = ctx.infinity();
    let s = (ctx.rational(1, 2).pow(&k) / (&k + &one)).summation(&k, &zero, &inf);
    assert_eq!(s.to_string(), "2*ln(2)");
    let s = (ctx.rational(-1, 2).pow(&k) / (&k + &ctx.int(3))).summation(&k, &one, &inf);
    assert_close(&s, "8*log(3/2) - 10/3");
    let s = (ctx.int(-2).pow(&k) / (&two * &k).factorial()).summation(&k, &zero, &inf);
    assert_eq!(s.to_string(), "cos(sqrt(2))");
    let s = (ctx.int(-2).pow(&k) / (&(&two * &k) + &one).factorial()).summation(&k, &zero, &inf);
    assert_close(&s, "sqrt(2)*sin(sqrt(2))/2");
    let s = ((&two * &k).binomial(&k) / ctx.int(8).pow(&k)).summation(&k, &zero, &inf);
    assert_eq!(s.to_string(), "sqrt(2)");
    let s = ((&two * &k).binomial(&k) * ctx.rational(-1, 4).pow(&k)).summation(&k, &zero, &inf);
    assert_close(&s, "1/sqrt(2)");
}

/// Before: refused.  Poles at a symbolic offset (`1/(k + n)`) sum to
/// harmonic numbers; the infinite telescoping `1/(k + a) − 1/(k + a + 1)`
/// needs `lim 1/(k + a) = 0`; an infinite product `Π b^{f(k)}` is
/// `b^{Σ f(k)}`.
///
/// Oracle: SymPy 1.14 `summation(1/(n + k), (k, 1, n))` →
/// `-harmonic(n) + harmonic(2*n)` (checked term by term at `n = 1..6`),
/// `summation(1/(k + a) - 1/(k + a + 1), (k, 0, oo))` → `1/a`,
/// `product((S(1)/4)**(S(1)/2)**k, (k, 1, oo))` → `1/4`.
#[test]
fn symbolic_offsets_telescoping_and_power_products() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = int_sym(&ctx, "n", Assumption::NonNegative);
    let a = pos(&ctx, "a");
    let one = ctx.int(1);
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let body = &one / &(&n + &k);
    let s = body.summation(&k, &one, &n);
    assert!(!s.has_unevaluated(), "{s}");
    for nv in 1..=6 {
        let want = direct_sum(&body.subs_i64(&n, nv), &k, 1, nv);
        assert_eq!(s.subs_i64(&n, nv).eval(), want, "n = {nv}: {s}");
    }
    let body = &one / &(&k + &a) - &one / &(&(&k + &a) + &one);
    assert_eq!(body.summation(&k, &zero, &inf).to_string(), "1/a");
    let p = ctx
        .rational(1, 4)
        .pow(&ctx.rational(1, 2).pow(&k))
        .product_over(&k, &one, &inf);
    assert_eq!(p.to_string(), "1/4");
    let p = k.powi(-2).exp().product_over(&k, &one, &inf);
    assert_eq!(p.to_string(), "exp(1/6*pi^2)");
}
