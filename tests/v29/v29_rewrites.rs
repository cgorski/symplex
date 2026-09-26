//! After 0.29 — the rewrite and integration hunter over ℂ: every rewriting
//! transform (simplify, expand*, factor, together, cancel, apart, ratsimp,
//! radsimp, powsimp, powdenest, logcombine, trigsimp, fu, rewrite, collect,
//! refine, …) compared with its input at real and complex points where both
//! are continuous, the same under declared assumptions at points that
//! satisfy them, the assumption queries against the values, and
//! antiderivatives / definite integrals against a central difference and
//! quadrature.
//!
//! Each test says what was wrong before (with the point where the value
//! differed); every reference value cites the mpmath 1.3.0 / SymPy 1.14 call
//! that produced it.

use symplex::prelude::*;

/// `e` at `var = v` as a complex number, through a 30-digit evaluation.
fn value_at(e: &Ex, var: &Ex, v: &Ex) -> Complex64 {
    e.subs(var, v).eval_complex64().expect("evaluates")
}

fn assert_close(z: Complex64, re: f64, im: f64, what: &str) {
    let w = Complex64::new(re, im);
    assert!(
        (z - w).norm() < 1e-12 * w.norm().max(1.0),
        "{what}: {z} vs {w}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// powdenest: a positive rational to a non-real power is not non-negative
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn powdenest_keeps_the_root_of_a_product_of_complex_powers() {
    // Before, `powdenest(false)` counted `2^x` as non-negative (a positive
    // rational base, whatever the exponent) and split `√(2^x·3^x)` into
    // `√(2^x)·√(3^x)`: at `x = −1/100 + 2i` that is `−0.2172… + 0.9670…i`,
    // the negative of the value.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (ctx.int(2).pow(&x) * ctx.int(3).pow(&x)).sqrt();
    let d = e.powdenest(false);
    let at = ctx.rational(-1, 100) + ctx.i_unit() * 2;
    // mpmath: mp.dps = 30; x = mpc(-0.01, 2); sqrt(power(2, x)*power(3, x))
    //   = 0.217_214_724_342_319_3177819171947 - 0.966984867774120083190645172447j
    assert_close(
        value_at(&d, &x, &at),
        0.217_214_724_342_319_3,
        -0.966_984_867_774_120_1,
        "powdenest",
    );
    // A rational exponent keeps the old behaviour: √(2^(1/3)·3^(1/3)) splits.
    let r = (ctx.int(2).pow(&ctx.rational(1, 3)) * ctx.int(3).pow(&ctx.rational(1, 3)) * &x).sqrt();
    assert_ne!(r.powdenest(false), r);
}

// ═══════════════════════════════════════════════════════════════════════════
// simplify's pow_pow: a non-negative base needs a real inner exponent
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_keeps_a_power_of_a_complex_power_of_a_positive_base() {
    // Before, `pow_pow` accepted `(256^(3i/2))^x → 256^(3ix/2)` because the
    // base is positive; but `Im(3i/2·ln 256) ≈ 8.3 > π`, so the inner power's
    // logarithm is not `3i/2·ln 256`.  At `x = 1/3` the simplified form was
    // `−0.9327 + 0.3607i` (mpmath: power(256, 1.5j/3)).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (ctx.rational(3, 2) * ctx.int(256).ln() * ctx.i_unit())
        .exp()
        .pow(&x);
    let s = e.simplify();
    // mpmath: mp.dps = 30; b = exp(mpf(3)/2*log(256)*1j); power(b, mpf(1)/3)
    //   = 0.778_707_288_759_266467429545007 + 0.627_387_406_976_895_8093325015771j
    assert_close(
        value_at(&s, &x, &ctx.rational(1, 3)),
        0.778_707_288_759_266,
        0.627_387_406_976_895_8,
        "simplify",
    );
    // A real inner exponent over a positive base still flattens.
    let t = ctx.int(2).pow(&ctx.rational(3, 2)).pow(&x).simplify();
    assert_eq!(t, ctx.int(2).pow(&(ctx.rational(3, 2) * &x)));
}

// ═══════════════════════════════════════════════════════════════════════════
// expand_log: a negative factor that is not a number
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_log_of_a_negated_negative_symbol_terminates() {
    // Before, `expand_log(ln(−w))` for `w < 0` looped forever: the negative
    // factor `w` was replaced by `−w = (−1)·w` and pushed back onto the work
    // list, which split it the same way again (the asm hunter's
    // `expand_log` hangs, e.g. `ln(−8w)`, `ln(w·e)`, `ln(−w(w − 3))`).
    let ctx = Context::new();
    let w = ctx.symbol_with("w", &[Assumption::Negative]).unwrap();
    let e = (-&w).ln();
    assert_eq!(e.expand_log(), e);
    let at = ctx.rational(-1, 3);
    // mpmath: mp.dps = 30; log(-8*mpf(-1)/3) = 0.980829253011726236856451127452
    let t = (ctx.int(-8) * &w).ln().expand_log();
    assert_eq!(t, ctx.int(8).ln() + (-&w).ln());
    assert_close(
        value_at(&t, &w, &at),
        0.980_829_253_011_726_2,
        0.0,
        "ln(-8w)",
    );
    // mpmath: log(mpf(-1)/3*e) = -0.0986122886681096913952452369225 + 3.14159265358979323846264338328j
    let u = (&w * ctx.e()).ln().expand_log();
    assert_close(
        value_at(&u, &w, &at),
        -0.098_612_288_668_109_69,
        std::f64::consts::PI,
        "ln(w e)",
    );
    // ln(−w(w − 3)): both factors negative, the product negative.
    let v = (-&w * (&w - 3)).ln();
    let ve = v.expand_log();
    assert_close(
        value_at(&ve, &w, &at),
        value_at(&v, &w, &at).re,
        value_at(&v, &w, &at).im,
        "ln(-w(w-3))",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// expand: terms that fold into sums are expanded too (idempotence)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_is_a_sum_of_products_when_a_radical_squares_out() {
    // Before, `expand((x + √(4 − √5))⁴)` left `6x²·(4 − √5)` and
    // `(4 − √5)²` (the canonical folding of `(√(4 − √5))²` and `⁴` in the
    // new terms, which the bottom-up pass did not revisit): not the sum of
    // products `expand` promises, and a second `expand` changed it.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x + (ctx.int(4) - ctx.int(5).sqrt()).sqrt()).powi(4);
    let once = e.expand();
    assert_eq!(once.expand(), once);
    // SymPy: Poly(expand((x + sqrt(4 - sqrt(5)))**4), x).all_coeffs()
    //   = [1, 4*sqrt(4 - sqrt(5)), 24 - 6*sqrt(5), …, 21 - 8*sqrt(5)]
    let five = ctx.int(5).sqrt();
    assert_eq!(once.subs(&x, &ctx.int(0)), ctx.int(21) - ctx.int(8) * &five);
    assert_eq!(
        once.diff(&x).diff(&x).subs(&x, &ctx.int(0)),
        (ctx.int(24) - ctx.int(6) * &five) * 2
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumptions: real · i is imaginary only for a non-zero real factor
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a_real_times_i_that_may_vanish_is_not_known_imaginary() {
    // Before, `real · imaginary` was `imaginary` — which implies non-zero
    // and non-real — also when the real factor may be 0: `(r·i).is_real()`
    // was `Some(false)`, `(r·i).is_zero()` `Some(false)`, and `|m·i|²`
    // positive for `m ≥ 0`, all wrong at `r = m = 0` (value 0).
    let ctx = Context::new();
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let m = ctx.symbol_with("m", &[Assumption::NonNegative]).unwrap();
    let q = ctx
        .symbol_with("q", &[Assumption::Real, Assumption::NonZero])
        .unwrap();
    let i = ctx.i_unit();
    // SymPy: r = Symbol('r', real=True); (I*r).is_real, (I*r).is_imaginary,
    //   (I*r).is_zero  ->  None, None, None
    let ri = &r * &i;
    assert_eq!(ri.is_real(), None);
    assert_eq!(ri.is_imaginary(), None);
    assert_eq!(ri.is_zero(), None);
    // SymPy: m = Symbol('m', nonnegative=True); (Abs(I*m)**2).is_positive -> None
    let am = (&m * &i).abs().powi(2);
    assert_eq!(am.is_positive(), None);
    assert_eq!(am.is_zero(), None);
    // SymPy: q = Symbol('q', real=True, nonzero=True); (I*q).is_imaginary,
    //   (I*q).is_real  ->  True, False
    let qi = &q * &i;
    assert_eq!(qi.is_imaginary(), Some(true));
    assert_eq!(qi.is_real(), Some(false));
}

#[test]
fn refine_keeps_a_rewritten_argument() {
    // Before, `refine` rebuilt `abs`/`sign`/powers only when the *new*
    // child was a key of its rewrite cache (it never is), so a rewrite
    // below them was dropped: `refine(sign(√(r²)))` for real `r` returned
    // its input instead of `sign(|r|)`, and `refine((√(r²) + 1)³)` its
    // input instead of `(|r| + 1)³`.
    let ctx = Context::new();
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    // SymPy: refine(Abs(sqrt(r**2))) with r real -> Abs(r);
    //        refine(sign(sqrt(r**2))) -> sign(Abs(r))
    assert_eq!(r.powi(2).sqrt().abs().refine(), r.abs());
    assert_eq!(r.powi(2).sqrt().sign().refine(), r.abs().sign());
    assert_eq!(
        (r.powi(2).sqrt() + 1).powi(3).refine(),
        (r.abs() + 1).powi(3)
    );
}

#[test]
fn a_query_after_assume_sees_the_new_assumption() {
    // Before, `assume` updated the symbol's own entry in the context's
    // assumption cache but kept every entry derived from it: `z + 1`, asked
    // once before `z.assume(Positive)`, stayed "sign unknown" (a cached
    // entry is returned without recomputation), and so did `w²` asked for
    // through `√(w²)` before `w.assume(Real)`.
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let g = &z + 1;
    assert_eq!(g.is_positive(), None);
    let z = z.assume(Assumption::Positive).unwrap();
    // SymPy: (Symbol('z', positive=True) + 1).is_positive -> True
    assert_eq!(g.is_positive(), Some(true));
    assert_eq!(z.is_positive(), Some(true));
    let w = ctx.symbol("w");
    assert_eq!(w.powi(2).sqrt().is_real(), None);
    let _w = w.clone().assume(Assumption::Real).unwrap();
    // SymPy: (Symbol('w', real=True)**2).is_nonnegative -> True
    assert_eq!(w.powi(2).is_nonnegative(), Some(true));
}
