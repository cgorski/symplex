//! Polylogarithm sums `Σ_{k≥lo} R(k)·Y^k` (batch 7 after 0.47): a rational
//! `R` times a geometric term.  Up to 0.47 only the table's `Σ x^k/k =
//! −ln(1 − x)` and the polynomial weights `Σ P(k)·Y^k` closed; `Σ (1/2)^k/k²`
//! stayed a `Sum`.  Every reference value is mpmath 1.3 `nsum` at 40 digits
//! (agreeing with 60 digits to below 1e-40); SymPy 1.14 `summation` is cited
//! as a second opinion.

use symplex::prelude::*;

fn pos(ctx: &Context, name: &str) -> Ex {
    ctx.symbol_with(name, &[Assumption::Positive]).unwrap()
}

/// `Σ_{k=lo}^{∞} body` with `body` parsed.
fn sum_from(ctx: &Context, body: &str, lo: i64) -> Ex {
    let k = ctx.symbol("k");
    ctx.parse(body)
        .unwrap()
        .summation(&k, &ctx.int(lo), &ctx.infinity())
}

/// `got` is closed and equals `want_re + want_im·i` to 1e-12 relative.
fn assert_value(got: &Ex, want_re: f64, want_im: f64) {
    assert!(!got.has_unevaluated(), "not closed: {got}");
    let g = got
        .eval_complex64()
        .unwrap_or_else(|e| panic!("{got} does not evaluate: {e}"));
    let scale = 1f64.max(want_re.hypot(want_im));
    assert!(
        (g.re - want_re).hypot(g.im - want_im) <= 1e-12 * scale,
        "{got} = {g} (want {want_re} + {want_im}i)"
    );
}

/// Before: `Σ_{k≥1} (1/2)^k/k²`, `(1/3)^k/k²`, `(1/2)^k/k³`, `(−1/2)^k/k²`
/// stayed unevaluated (only `Σ (1/2)^k/k = ln 2` closed, by the table).
/// Now `Li_s(Y)`, and `eval` folds `Li₂(1/2) = π²/12 − ln²2/2`.
///
/// Oracle: mpmath `nsum(lambda k: mpf(1)/2**k/k**2, [1, inf])` =
/// 0.58224052646501250590…, `(1/3)^k/k²` 0.36621322997706348761…,
/// `(1/2)^k/k³` 0.53721319360804020094…, `(−1/2)^k/k²`
/// −0.44841420692364620244…; SymPy 1.14 gives the same closed forms.
#[test]
fn numeric_ratio_is_a_polylogarithm() {
    let ctx = Context::new();
    let s = sum_from(&ctx, "(1/2)^k/k^2", 1);
    assert_eq!(s.to_string(), "-1/2*ln(2)^2 + 1/12*pi^2");
    assert_value(&s, 0.582_240_526_465_012_5, 0.0);
    let s = sum_from(&ctx, "(1/3)^k/k^2", 1);
    assert_eq!(s.to_string(), "polylog(2, 1/3)");
    assert_value(&s, 0.366_213_229_977_063_5, 0.0);
    // `polylog(3, 1/2)` folds since 0.50 (`7ζ(3)/8 − π² ln 2/12 + ln³2/6`).
    let s = sum_from(&ctx, "(1/2)^k/k^3", 1);
    assert_eq!(s.to_string(), "1/6*ln(2)^3 - 1/12*pi^2*ln(2) + 7/8*zeta(3)");
    assert_value(&s, 0.537_213_193_608_040_2, 0.0);
    for body in ["(-1/2)^k/k^2", "(-1)^k*(1/2)^k/k^2"] {
        let s = sum_from(&ctx, body, 1);
        assert_eq!(s.to_string(), "polylog(2, -1/2)", "{body}");
        assert_value(&s, -0.448_414_206_923_646_2, 0.0);
    }
    // k·k⁻³ reduces the order; exp(−k) is the ratio 1/e.
    assert_value(
        &sum_from(&ctx, "k*(1/2)^k/k^3", 1),
        0.582_240_526_465_012_5,
        0.0,
    );
    let s = sum_from(&ctx, "exp(-k)/k^2", 1);
    assert_eq!(s.to_string(), "polylog(2, exp(-1))");
}

/// Before: shifted poles and other lower limits stayed unevaluated.
/// `Σ_{k≥lo} Y^k/(k + β)^m = Y^(−β)·(Li_m(Y) − Σ_{j<lo+β} Y^j/j^m)`.
///
/// Oracle (mpmath `nsum`, 40 digits): `Σ_{k≥1} (1/2)^k/(k + 1)²` =
/// 0.16448105293002501180…, from `k = 0` 1.16448105293002501180…,
/// `Σ_{k≥3} (1/2)^k/k²` 0.01974052646501250590…, `Σ_{k≥4} (1/2)^k/(k − 3)²`
/// 0.07278006580812656323…, `Σ_{k≥1} (1/2)^(k+1)/k²`
/// 0.29112026323250625295…; SymPy 1.14 agrees (`−1 − log(2)²+ π²/6`, …).
#[test]
fn shifted_poles_and_lower_limits() {
    let ctx = Context::new();
    let s = sum_from(&ctx, "(1/2)^k/(k+1)^2", 1);
    assert_eq!(s.to_string(), "-ln(2)^2 + 1/6*pi^2 - 1");
    assert_value(&s, 0.164_481_052_930_025, 0.0);
    let s = sum_from(&ctx, "(1/2)^k/(k+1)^2", 0);
    assert_eq!(s.to_string(), "-ln(2)^2 + 1/6*pi^2");
    assert_value(&s, 1.164_481_052_930_025, 0.0);
    assert_value(
        &sum_from(&ctx, "(1/2)^k/k^2", 3),
        0.019_740_526_465_012_51,
        0.0,
    );
    assert_value(
        &sum_from(&ctx, "(1/2)^k/(k-3)^2", 4),
        0.072_780_065_808_126_56,
        0.0,
    );
    assert_value(
        &sum_from(&ctx, "(1/2)^(k+1)/k^2", 1),
        0.291_120_263_232_506_3,
        0.0,
    );
}

/// Before: a rational weight with several poles stayed unevaluated; now
/// by partial fractions (the polynomial part by the arithmetico-geometric
/// sum).
///
/// Oracle (mpmath `nsum`, 40 digits): `Σ_{k≥1} (1/2)^k/(k²(k + 1))` =
/// 0.27538770702495781531…, `Σ (1/2)^k (k² + 1)/k` 2.69314718055994530941…
/// (`ln 2 + 2`), `Σ (1/3)^k/(k(k + 2))` 0.12813956756734247208…
/// (`7/4 − 4 ln(3/2)`), `Σ (2/3)^k (k³ + 1)/(k²(k + 2)²)`
/// 0.30500390832665561613…; SymPy 1.14 gives the same values.
#[test]
fn rational_weights_by_partial_fractions() {
    let ctx = Context::new();
    let s = sum_from(&ctx, "(1/2)^k/(k^2*(k+1))", 1);
    assert_eq!(s.to_string(), "-1/2*ln(2)^2 + 1/12*pi^2 + ln(2) - 1");
    assert_value(&s, 0.275_387_707_024_957_8, 0.0);
    let s = sum_from(&ctx, "(1/2)^k*(k^2+1)/k", 1);
    assert_eq!(s.to_string(), "ln(2) + 2");
    let s = sum_from(&ctx, "(1/3)^k/(k*(k+2))", 1);
    assert_eq!(s.to_string(), "-4*ln(3/2) + 7/4");
    assert_value(&s, 0.128_139_567_567_342_5, 0.0);
    assert_value(
        &sum_from(&ctx, "(2/3)^k*(k^3+1)/(k^2*(k+2)^2)", 1),
        0.305_003_908_326_655_6,
        0.0,
    );
    // Telescoping weights still go through Gosper (unchanged form).
    assert_eq!(
        sum_from(&ctx, "(1/2)^k*(k+2)/(k*(k+1))", 1).to_string(),
        "1"
    );
}

/// Before: poles at half-integers `(2k + h)^(−m)` with `m ≥ 2` stayed
/// unevaluated.  With `w² = Y` the sum is the odd part
/// `(Li_m(w) − Li_m(−w))/2` of the polylogarithm.
///
/// Oracle (mpmath `nsum`, 40 digits): `Σ_{k≥0} (1/2)^k/(2k + 1)²` =
/// 1.06928637514523744528…, `Σ_{k≥1} (1/4)^k/(2k − 1)³`
/// 0.25245275956673426888…, `Σ_{k≥0} (−1/2)^k/(2k + 1)²`
/// 0.95247416912953901172… (real, though written with `Li₂(±i/√2)`);
/// SymPy 1.14 gives the same closed forms.
#[test]
fn half_integer_poles_use_the_odd_part() {
    let ctx = Context::new();
    assert_value(
        &sum_from(&ctx, "(1/2)^k/(2*k+1)^2", 0),
        1.069_286_375_145_237_4,
        0.0,
    );
    // `polylog(3, 1/2)` folds since 0.50 (`7ζ(3)/8 − π² ln 2/12 + ln³2/6`).
    let s = sum_from(&ctx, "(1/4)^k/(2*k-1)^3", 1);
    assert_eq!(
        s.to_string(),
        "1/24*ln(2)^3 - 1/48*pi^2*ln(2) - 1/4*polylog(3, -1/2) + 7/32*zeta(3)"
    );
    assert_value(&s, 0.252_452_759_566_734_3, 0.0);
    assert_value(
        &sum_from(&ctx, "(-1/2)^k/(2*k+1)^2", 0),
        0.952_474_169_129_539,
        0.0,
    );
    // Terms with 2k + h < 0 lead the odd series (a positive `a`; mpmath
    // `nsum(lambda k: mpf(1)/9**k*(-3*k)/(2*k-3), [1, inf])` at `a = 2` =
    // 0.25473773495333789088…; SymPy 1.14 leaves a `Piecewise` with a `Sum`).
    let a = pos(&ctx, "a");
    let s = sum_from(&ctx, "(1/(a+1)^2)^k*(-3*k)/(2*k-3)", 1);
    assert_value(&s.subs(&a, &ctx.int(2)), 0.254_737_734_953_337_9, 0.0);
}

/// Before: with the trivial ratio `(−1)^(2k)` (not folded to 1 for an
/// unassumed `k`) a convergent rational sum stayed unevaluated, and so did
/// a rational sum with poles at integers and half-integers both.  The
/// divergent `Li_1(1)` parts cancel (their coefficients add up to 0 when
/// `deg R ≤ −2`) and are dropped.
///
/// Oracle: mpmath `nsum(lambda k: -1/((k+2)**2*(k-2)**3), [3, inf])` =
/// −0.04468025817507513973… (SymPy 1.14: `−zeta(3)/16 − 215/4608 +
/// pi²/128`), `Σ_{k≥0} 1/((k + 1)(2k + 1))` = 2 ln 2 =
/// 1.38629436111989061883…, `Σ_{k≥1} (1 − 3k)/((2k − 3)²(k + 1)(k + 3))` =
/// −0.65054694878052161850… (SymPy 1.14: `nan`, which is wrong).
#[test]
fn trivial_ratio_regularises_the_logarithms() {
    let ctx = Context::new();
    assert_value(
        &sum_from(&ctx, "(-1)^(2*k)*(-1)/((k+2)^2*(k-2)^3)", 3),
        -0.044_680_258_175_075_14,
        0.0,
    );
    assert_value(
        &sum_from(&ctx, "(-1)^(2*k)/((k+1)*(2*k+1))", 0),
        std::f64::consts::LN_2 * 2.0,
        0.0,
    );
    let s = sum_from(&ctx, "(1-3*k)/((2*k-3)^2*(k+1)*(k+3))", 1);
    assert_eq!(s.to_string(), "-7/180*pi^2 - 74/2025*ln(2) - 2933/12150");
    assert_value(&s, -0.650_546_948_780_521_6, 0.0);
}

/// Before: a ratio on the unit circle other than `±1` was never decided
/// (`Σ i^k/k` stayed unevaluated although it converges by Dirichlet's
/// test); now an exactly unimodular, non-rational ratio closes when
/// `deg R < 0` (Abel's theorem gives the value).
///
/// Oracle (mpmath `nsum`, 40 digits): `Σ_{k≥1} i^k/k` =
/// −0.34657359027997265470… + 0.78539816339744830961…i, `Σ i^k/k²`
/// −0.20561675835602830455… + 0.91596559417721901505…i (`−π²/48 + iG`),
/// `Σ ((3 + 4i)/5)^k/k²` 0.40331124988898543458… + 1.00841303731693537477…i,
/// `Σ e^(iπk/3)/k` = iπ/3.  SymPy 1.14 leaves `Σ exp(iπk/3)/k`
/// unevaluated.
#[test]
fn unimodular_ratio_converges_by_dirichlet() {
    let ctx = Context::new();
    assert_value(
        &sum_from(&ctx, "I^k/k", 1),
        -0.346_573_590_279_972_65,
        std::f64::consts::FRAC_PI_4,
    );
    // `polylog(2, I)` folds to `−π²/48 + iG` since 0.50.
    let s = sum_from(&ctx, "I^k/k^2", 1);
    assert_eq!(s.to_string(), "-1/48*pi^2 + I*Catalan");
    assert_value(&s, -0.205_616_758_356_028_3, 0.915_965_594_177_219);
    assert_value(
        &sum_from(&ctx, "((3+4*I)/5)^k/k^2", 1),
        0.403_311_249_888_985_4,
        1.008_413_037_316_935_4,
    );
    assert_value(
        &sum_from(&ctx, "exp(I*pi/3)^k/k", 1),
        0.0,
        std::f64::consts::FRAC_PI_3,
    );
    // Terms that do not tend to 0 on the circle: no closed form.
    assert!(sum_from(&ctx, "I^k*k/(k+1)", 1).has_unevaluated());
}

/// Before: `Σ (−1/(a + 1))^k/k²` and `Σ_{k≥0} (1/(a + 1))^k/(k + 1)²` for a
/// positive `a` stayed unevaluated (the ratio's modulus is below 1 by the
/// assumptions).  The unassumed `Σ x^k/k²`, `Σ x^k/k`, `Σ k·x^k` stay
/// formal, as `Σ x^k` does (module convention: no `|x| < 1` to go by);
/// SymPy 1.14 answers `Piecewise((polylog(2, x), Abs(x) <= 1), (Sum(…),
/// True))` there.
///
/// Oracle (mpmath `nsum` at `a = 2`, 40 digits): `Σ (1/3)^k/k²` =
/// 0.36621322997706348761…, `Σ (−1/3)^k/k²` −0.30903312648780847231…,
/// `Σ_{k≥0} (1/3)^k/(k + 1)²` 1.09863968993119046285…; SymPy 1.14:
/// `polylog(2, 1/(a + 1))`, `polylog(2, -1/(a + 1))`,
/// `(a + 1)*polylog(2, 1/(a + 1))`.
#[test]
fn assumption_bounded_symbolic_ratio() {
    let ctx = Context::new();
    let a = pos(&ctx, "a");
    let two = ctx.int(2);
    let s = sum_from(&ctx, "(1/(a+1))^k/k^2", 1);
    assert_eq!(s.to_string(), "polylog(2, 1/(a + 1))");
    assert_value(&s.subs(&a, &two), 0.366_213_229_977_063_5, 0.0);
    let s = sum_from(&ctx, "(-1/(a+1))^k/k^2", 1);
    assert_eq!(s.to_string(), "polylog(2, -1/(a + 1))");
    assert_value(&s.subs(&a, &two), -0.309_033_126_487_808_47, 0.0);
    let s = sum_from(&ctx, "(1/(a+1))^k/(k+1)^2", 0);
    assert_eq!(s.to_string(), "(a + 1)*polylog(2, 1/(a + 1))");
    assert_value(&s.subs(&a, &two), 1.098_639_689_931_190_5, 0.0);
    for body in ["x^k/k^2", "x^k/k", "k*x^k", "x^k/(k+1)^2"] {
        assert!(sum_from(&ctx, body, 1).has_unevaluated(), "{body}");
    }
}

/// Divergent or undecided cases are unchanged: `Σ 2^k/k²` → ∞, and at the
/// ratio `−1` a polynomial part (terms not tending to 0) gives no closed
/// form.
#[test]
fn divergent_ratios_stay_divergent() {
    let ctx = Context::new();
    assert_eq!(sum_from(&ctx, "2^k/k^2", 1).to_string(), "oo");
    let k = ctx.symbol("k");
    let body = ctx.parse("(-1)^k*(k^2+1)/k").unwrap();
    assert!(
        body.try_summation(&k, &ctx.int(1), &ctx.infinity())
            .is_err()
    );
}
