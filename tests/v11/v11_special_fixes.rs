//! 0.11.1 review fixes: special_fixes.  Reference values cite SymPy 1.14 / mpmath
//! 1.3 (`symplex/.venv/bin/python`).
//!
//! Covers: `Γ(s, 0)` / `γ(s, 0)` for non-positive `s`, symbolic-parameter
//! orthogonal polynomials (degree cap + polynomial-time Jacobi expansion),
//! `polylog` near `z = 1` and for `s < 0`, `erf`/`erfc` at large arguments,
//! the Dirichlet eta function near `s = 1`, `Li_s(−1)` for `s ≤ 1`, Airy
//! derivatives at `0`, `η(−n)` for huge `n`, and series that used to return
//! partial sums.

// Reference values are quoted at the precision mpmath printed them.
#![allow(clippy::excessive_precision)]

use std::time::{Duration, Instant};
use symplex::prelude::*;

fn decimal(e: &Ex, digits: u32) -> String {
    e.eval_decimal(digits)
        .unwrap_or_else(|err| panic!("eval_decimal({e}, {digits}) failed: {err}"))
}

fn assert_prefix(e: &Ex, digits: u32, prefix: &str) {
    let s = decimal(e, digits);
    assert!(
        s.starts_with(prefix),
        "{e} at {digits} digits: {s} !~ {prefix}"
    );
}

/// Relative closeness `|a − b| ≤ tol·max(1, |b|)`.
fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Γ(s, 0) and γ(s, 0) for s ≤ 0
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn incomplete_gamma_at_zero_refuses_nonpositive_s() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    // ∫₀^∞ t^{s−1} e^{−t} dt diverges for s ≤ 0: SymPy gives `zoo` for
    // uppergamma(-1/2, 0) / uppergamma(-1, 0); we leave the node alone.
    for s in [ctx.rational(-1, 2), ctx.int(-1), ctx.int(0)] {
        let g = zero.uppergamma(&s).eval();
        assert_eq!(format!("{g}"), format!("uppergamma({s}, 0)"), "{g}");
        let l = zero.lowergamma(&s).eval();
        assert_eq!(format!("{l}"), format!("lowergamma({s}, 0)"), "{l}");
    }
    // s > 0 keeps folding: Γ(1/2, 0) = Γ(1/2) (SymPy: sqrt(pi)), γ(3, 0) = 0.
    let g = zero.uppergamma(&ctx.rational(1, 2)).eval();
    assert_eq!(g, ctx.rational(1, 2).gamma(), "{g}");
    assert_eq!(g.eval(), ctx.pi().sqrt(), "{g}");
    assert_eq!(zero.uppergamma(&ctx.int(3)).eval().eval(), ctx.int(2));
    assert_eq!(zero.lowergamma(&ctx.int(3)).eval(), zero);
    // Symbolic s (assumptions are not visible to `eval`) keeps the fold that
    // the Gamma-distribution CDF `γ(k, 0)/Γ(k) = 0` relies on (SymPy:
    // lowergamma(k, 0) = 0; uppergamma(k, 0) → gamma(k) for positive k).
    let k = ctx.symbol("k");
    assert_eq!(zero.lowergamma(&k).eval(), zero);
    assert_eq!(zero.uppergamma(&k).eval(), k.gamma());
    // Γ(s, ∞) = 0 and γ(s, ∞) = Γ(s) are untouched (SymPy: lowergamma(-1/2, oo) = -2*sqrt(pi)).
    let inf = ctx.infinity();
    assert_eq!(inf.uppergamma(&ctx.rational(-1, 2)).eval(), zero);
    assert_eq!(
        inf.lowergamma(&ctx.rational(-1, 2)).eval(),
        ctx.rational(-1, 2).gamma()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Symbolic-parameter orthogonal polynomials
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn jacobi_symbolic_parameters_expand_in_polynomial_time() {
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    let t = Instant::now();
    let p14 = x.jacobi(&ctx.int(14), &a, &b).eval();
    let elapsed = t.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "jacobi(14, a, b, x).eval() took {elapsed:?}"
    );
    assert!(
        !format!("{p14}").starts_with("jacobi("),
        "not expanded: {p14}"
    );
    // mpmath 1.3: jacobi(14, 1/3, 1/2, 2/5) = -0.276151653913946586721468290451
    let v = p14
        .subs_map(&[
            (&a, &ctx.rational(1, 3)),
            (&b, &ctx.rational(1, 2)),
            (&x, &ctx.rational(2, 5)),
        ])
        .eval_f64()
        .unwrap();
    assert!(close(v, -0.276151653913946586721468290451, 1e-12), "{v}");
    // The same polynomial as the closed-form sum for small n.
    let p1 = x.jacobi(&ctx.int(1), &a, &b).eval();
    let expected = ((&a - &b) / 2 + (&a + &b + 2) * &x / 2).expand();
    assert_eq!(p1, expected, "{p1}");
}

#[test]
fn symbolic_parameter_degree_is_capped_at_16() {
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    // Above the cap the node is returned unevaluated — quickly.
    let t = Instant::now();
    let j = x.jacobi(&ctx.int(17), &a, &b).eval();
    assert_eq!(format!("{j}"), "jacobi(17, a, b, x)");
    let g = x.gegenbauer(&ctx.int(17), &a).eval();
    assert_eq!(format!("{g}"), "gegenbauer(17, a, x)");
    let l = x.assoc_laguerre(&ctx.int(17), &a).eval();
    assert_eq!(format!("{l}"), "assoc_laguerre(17, a, x)");
    let big = x.jacobi(&ctx.int(1000), &a, &b).eval();
    assert_eq!(format!("{big}"), "jacobi(1000, a, b, x)");
    assert!(t.elapsed() < Duration::from_secs(1));
    // At the cap they still expand.
    // mpmath 1.3: gegenbauer(16, 1/3, 2/5) = 0.0856623638834248792109214848533
    let g16 = x.gegenbauer(&ctx.int(16), &a).eval();
    assert!(!format!("{g16}").starts_with("gegenbauer("), "{g16}");
    let v = g16
        .subs_map(&[(&a, &ctx.rational(1, 3)), (&x, &ctx.rational(2, 5))])
        .eval_f64()
        .unwrap();
    assert!(close(v, 0.0856623638834248792109214848533, 1e-12), "{v}");
    // mpmath 1.3: laguerre(16, 1/3, 2/5) = -0.613267931309906753231665300159
    let l16 = x.assoc_laguerre(&ctx.int(16), &a).eval();
    assert!(!format!("{l16}").starts_with("assoc_laguerre("), "{l16}");
    let v = l16
        .subs_map(&[(&a, &ctx.rational(1, 3)), (&x, &ctx.rational(2, 5))])
        .eval_f64()
        .unwrap();
    assert!(close(v, -0.613267931309906753231665300159, 1e-12), "{v}");
    // Numeric parameters keep the old (max_pow_exponent) bound.
    // mpmath 1.3: jacobi(20, 1/3, 1/2, 2/5) = -0.141936232905513088097370878111
    let j20 = x
        .jacobi(&ctx.int(20), &ctx.rational(1, 3), &ctx.rational(1, 2))
        .eval();
    assert!(!format!("{j20}").starts_with("jacobi("), "{j20}");
    let v = j20
        .subs_map(&[(&x, &ctx.rational(2, 5))])
        .eval_f64()
        .unwrap();
    assert!(close(v, -0.141936232905513088097370878111, 1e-12), "{v}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3./4. polylog: convergence near z = 1 and for s < 0
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polylog_near_one_converges_on_the_actual_terms() {
    let ctx = Context::new();
    // mpmath 1.3 (mp.dps = 210): polylog(2, 3/4) =
    // 0.97846939293030610374306666652456149776148427461948725210482919950064176379261480718914647010170431695060306596024516533142767692955674892176667212796640699242326921745416905430377193223791871972416849
    // The old termination test stopped ~50 terms early (wrong from digit ~150).
    assert_prefix(
        &ctx.rational(3, 4).polylog(&ctx.int(2)),
        200,
        "0.9784693929303061037430666665245614977614842746194872521048291995006417637926148071891464701017043169506030659602451653314276769295567489217666721279664069924232692174541690543037719322379187197241684",
    );
    // mpmath 1.3: polylog(3/2, 9/10) = 1.61443852856633962632782266165666 (non-integer s, |μ| small)
    assert_prefix(
        &ctx.rational(9, 10).polylog(&ctx.rational(3, 2)),
        30,
        "1.6144385285663396263278226616",
    );
    // mpmath 1.3: polylog(1/2, 99/100) = 16.221830753428111347296005044475464 (s < 1 near z = 1)
    assert_prefix(
        &ctx.rational(99, 100).polylog(&ctx.rational(1, 2)),
        30,
        "16.22183075342811134729600504",
    );
    // mpmath 1.3: polylog(3, -9/10) = -0.81863820154436384207037512030772151 (z < 0 via the square identity)
    assert_prefix(
        &ctx.rational(-9, 10).polylog(&ctx.int(3)),
        30,
        "-0.8186382015443638420703751203",
    );
}

#[test]
fn polylog_negative_order_series_is_not_a_partial_sum() {
    let ctx = Context::new();
    // mpmath 1.3 (mp.dps = 200, direct Σ z^k k^{81/2} cross-check):
    // polylog(-81/2, 1/2) = 2.101146467457202013397669382627343033481e+55
    // The old term budget ignored the k^{|s|} growth and returned a partial
    // sum wrong from digit 24.
    assert_prefix(
        &ctx.rational(1, 2).polylog(&ctx.rational(-81, 2)),
        30,
        "2.1011464674572020133976693826",
    );
    // z < 0: the alternating terms peak near 2^{179} while the sum is ≈ 2^{93};
    // the lost bits are measured and the sum recomputed.
    // mpmath 1.3 (mp.dps = 200 direct sum; mpmath's own polylog is only good
    // to ~17 digits here at dps = 40): polylog(-81/2, -1/2) =
    // 8449034274208159355099097524.89041687475
    assert_prefix(
        &ctx.rational(-1, 2).polylog(&ctx.rational(-81, 2)),
        30,
        "8449034274208159355099097524.8",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. erf / erfc at large arguments
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn erfc_is_computed_to_relative_precision() {
    let ctx = Context::new();
    // mpmath 1.3: erfc(7) = 4.18382560777941439861401e-23 (was −3.749e−15)
    assert_prefix(&ctx.int(7).erfc(), 16, "4.183825607779414e-23");
    // mpmath 1.3: erfc(10) = 2.088487583762544757000786294957788611560818119321163727e-45
    assert_prefix(
        &ctx.int(10).erfc(),
        50,
        "2.0884875837625447570007862949577886115608181193212e-45",
    );
    // mpmath 1.3: erfc(30) = 2.564656203756111600033397e-393 (asymptotic branch)
    assert_prefix(&ctx.int(30).erfc(), 16, "2.564656203756112e-393");
    // mpmath 1.3: erfc(3) = 0.00002209049699858544137277613
    assert_prefix(&ctx.int(3).erfc(), 16, "2.209049699858544e-5");
    // mpmath 1.3: erfc(-3) = 1.999977909503001414558627
    assert_prefix(&ctx.int(-3).erfc(), 16, "1.999977909503001");
}

#[test]
fn erf_taylor_branch_survives_the_cancellation() {
    let ctx = Context::new();
    // mpmath 1.3: erf(7.4) = 0.99999999999999999999999987516143536 (was 1.000000145931148)
    let x = ctx.rational(74, 10).erf();
    assert_eq!(decimal(&x, 16), "1");
    assert_prefix(&x, 30, "0.99999999999999999999999987516");
    // mpmath 1.3: erf(10) = 0.9999999999999999999999999999999999999999999979115124162
    // (was −7.7e16 at 50 digits: the Taylor loop ran out of terms)
    assert_prefix(
        &ctx.int(10).erf(),
        50,
        "0.9999999999999999999999999999999999999999999979115",
    );
    // mpmath 1.3: erf(1) = 0.842700792949714869341220635082609259296066998
    assert_prefix(
        &ctx.int(1).erf(),
        40,
        "0.842700792949714869341220635082609259296",
    );
    // mpmath 1.3: erf(1/2) = 0.520499877813046537682746653891964528736451576
    assert_prefix(
        &ctx.rational(1, 2).erf(),
        40,
        "0.520499877813046537682746653891964528736",
    );
}

#[test]
fn uppergamma_half_integer_through_erfc_is_accurate() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    // eval folds Γ(1/2, 49) → √π erfc(7).
    // mpmath 1.3: gammainc(1/2, 49) = 7.415637810025734144833593e-23 (was −6.645e−15)
    let v = ctx.int(49).uppergamma(&half).eval_f64().unwrap();
    assert!(close(v, 7.415637810025734144833593e-23, 1e-14), "{v}");
    assert_prefix(&ctx.int(49).uppergamma(&half), 16, "7.415637810025734e-23");
    // mpmath 1.3: gammainc(1/2, 36) = 3.814274020654150817072593e-17
    assert_prefix(&ctx.int(36).uppergamma(&half), 16, "3.814274020654151e-17");
    // mpmath 1.3: gammainc(3/2, 50) = 1.377337934543816183506917e-21
    assert_prefix(
        &ctx.int(50).uppergamma(&ctx.rational(3, 2)),
        16,
        "1.377337934543816e-21",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Dirichlet eta near s = 1; Li_s(−1) for s ≤ 1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dirichlet_eta_is_smooth_through_s_equals_one() {
    let ctx = Context::new();
    let tiny = ctx.rational(1, 10).powi(20);
    // mpmath 1.3: altzeta(1 + 1e-20) = 0.69314718055994530941883081049560088
    // (was Err("zeta(1) is a pole"))
    let s = ctx.int(1) + &tiny;
    assert_prefix(&s.dirichlet_eta(), 16, "0.6931471805599453");
    assert_prefix(&s.dirichlet_eta(), 30, "0.69314718055994530941883081049");
    // mpmath 1.3: altzeta(1 - 1e-20) = 0.69314718055994530941563343242075226
    let s = ctx.int(1) - &tiny;
    assert_prefix(&s.dirichlet_eta(), 30, "0.69314718055994530941563343242");
    // mpmath 1.3: altzeta(1/2) = 0.6048986434216303702472659142359555
    assert_prefix(
        &ctx.rational(1, 2).dirichlet_eta(),
        30,
        "0.60489864342163037024726591423",
    );
    // mpmath 1.3: altzeta(3/2) = 0.76514702462540794536726875860347818
    assert_prefix(
        &ctx.rational(3, 2).dirichlet_eta(),
        30,
        "0.76514702462540794536726875860",
    );
    // s ≤ 0 goes through ζ: mpmath altzeta(-1/2) = 0.38010481260968401677754215655180836
    assert_prefix(
        &ctx.rational(-1, 2).dirichlet_eta(),
        30,
        "0.38010481260968401677754215655",
    );
}

#[test]
fn polylog_at_minus_one_for_s_at_most_one() {
    let ctx = Context::new();
    let m1 = ctx.int(-1);
    // mpmath 1.3: polylog(1/2, -1) = -0.6048986434216303702472659142359555
    assert_prefix(
        &m1.polylog(&ctx.rational(1, 2)),
        30,
        "-0.60489864342163037024726591423",
    );
    // mpmath 1.3: polylog(1, -1) = -ln 2 = -0.69314718055994530941723212145817657
    assert_prefix(
        &m1.polylog(&ctx.int(1)),
        30,
        "-0.69314718055994530941723212145",
    );
    // mpmath 1.3: polylog(-1/2, -1) = -0.38010481260968401677754215655180836
    assert_prefix(
        &m1.polylog(&ctx.rational(-1, 2)),
        30,
        "-0.38010481260968401677754215655",
    );
    // z = +1 folds to ζ(s) symbolically (SymPy: polylog(1/2, 1) = zeta(1/2));
    // the numeric path still refuses s ≤ 1 there (covered by the evalf unit tests).
    assert_eq!(
        ctx.int(1).polylog(&ctx.rational(1, 2)).eval(),
        ctx.rational(1, 2).zeta()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Airy derivatives at 0 (numeric path; eval folds these exactly)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn airy_derivatives_at_zero_are_finite() {
    let ctx = Context::new();
    // mpmath 1.3: airyai(0, derivative=1) = -0.25881940379280679840518356018920396
    assert_prefix(
        &ctx.int(0).airyaiprime(),
        30,
        "-0.25881940379280679840518356018",
    );
    // mpmath 1.3: airybi(0, derivative=1) = 0.44828835735382635791482371039882839
    assert_prefix(
        &ctx.int(0).airybiprime(),
        30,
        "0.44828835735382635791482371039",
    );
    // A tiny non-zero argument through the series (the previously unguarded
    // branch): mpmath airyai(1e-30, derivative=1) = -0.25881940379280679840518356018920396
    let tiny = ctx.rational(1, 10).powi(30);
    assert_prefix(&tiny.airyaiprime(), 20, "-0.2588194037928067984");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. η(−n) for huge n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dirichlet_eta_huge_negative_integer_is_cheap() {
    let ctx = Context::new();
    let t = Instant::now();
    // η(−2k) = 0 exactly (trivial zeros of ζ); SymPy: dirichlet_eta(-1000000) = 0.
    assert_eq!(ctx.int(-1_000_000).dirichlet_eta().eval(), ctx.int(0));
    // Odd: the Bernoulli number B_{1000000} is not computed; left alone.
    let e = ctx.int(-999_999).dirichlet_eta().eval();
    assert_eq!(format!("{e}"), "dirichlet_eta(-999999)");
    assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
    // Within the cap the exact values are unchanged: η(−3) = −1/8 (SymPy: -1/8),
    // η(−63) = (1 − 2^64) ζ(−63) with ζ(−63) = −B₆₄/64.
    assert_eq!(ctx.int(-3).dirichlet_eta().eval(), ctx.rational(-1, 8));
    let e63 = ctx.int(-63).dirichlet_eta().eval();
    let two64 = ctx.int(2).powi(64);
    let expected = ((ctx.int(1) - two64) * ctx.int(-63).zeta()).eval();
    assert_eq!(e63, expected, "{e63}");
    assert_eq!(ctx.int(-62).dirichlet_eta().eval(), ctx.int(0));
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. diff.rs reuses eval::apply_named (behavioural smoke test)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn special_function_derivatives_still_build_apply_nodes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(format!("{}", x.airyai().diff(&x)), "airyaiprime(x)");
    assert_eq!(format!("{}", x.airyaiprime().diff(&x)), "x*airyai(x)");
    let (n, a) = (ctx.symbol("n"), ctx.symbol("a"));
    assert_eq!(
        format!("{}", x.gegenbauer(&n, &a).diff(&x)),
        "2*a*gegenbauer(n - 1, a + 1, x)"
    );
    // The same interned node as the constructor builds.
    let d = x.airyai().diff(&x);
    assert_eq!(d, x.airyaiprime());
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Series that converge still do (the exhaustion exits are now errors)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergent_series_still_converge() {
    let ctx = Context::new();
    // mpmath 1.3: shi(40) = 3019859131805620.789179616, chi(40) = 3019859131805620.789179616
    assert_prefix(&ctx.int(40).shi(), 20, "3019859131805620.789");
    assert_prefix(&ctx.int(40).chi(), 20, "3019859131805620.789");
    // mpmath 1.3: fresnels(3) = 0.4963129989673750360976123, fresnelc(3) = 0.6057207892976856295561611
    assert_prefix(&ctx.int(3).fresnels(), 20, "0.496312998967375036");
    assert_prefix(&ctx.int(3).fresnelc(), 20, "0.605720789297685629");
    // mpmath 1.3: erfi(5) = 8298273880.676803516146223
    assert_prefix(&ctx.int(5).erfi(), 20, "8298273880.6768035161");
    // mpmath 1.3: gammainc(3, 0, 2) = 0.6466471676338730810600051
    assert_prefix(
        &ctx.int(2).lowergamma(&ctx.int(3)),
        20,
        "0.64664716763387308106",
    );
    // mpmath 1.3: polylog(2, 3/4) = 0.97846939293030610374306666652456149776148427461949
    assert_prefix(
        &ctx.rational(3, 4).polylog(&ctx.int(2)),
        50,
        "0.9784693929303061037430666665245614977614842746194",
    );
}
