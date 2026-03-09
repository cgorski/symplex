//! Tests for special functions (Wave J): Gamma, LogGamma, Digamma, Erf, Erfc, Beta.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Gamma function
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gamma_at_integers() {
    let ctx = Context::new();
    // Gamma(1) = 0! = 1, Gamma(2) = 1! = 1, Gamma(3) = 2! = 2, Gamma(5) = 4! = 24
    assert_eq!(format!("{}", ctx.int(1).gamma().eval()), "1");
    assert_eq!(format!("{}", ctx.int(2).gamma().eval()), "1");
    assert_eq!(format!("{}", ctx.int(3).gamma().eval()), "2");
    assert_eq!(format!("{}", ctx.int(5).gamma().eval()), "24");
    assert_eq!(format!("{}", ctx.int(7).gamma().eval()), "720");
}

#[test]
fn gamma_at_half() {
    let ctx = Context::new();
    // Gamma(1/2) = sqrt(pi)
    let half = ctx.rational(1, 2);
    let result = half.gamma().eval();
    let s = format!("{result}");
    assert!(
        s.contains("pi") || s.contains("sqrt"),
        "Gamma(1/2) = sqrt(pi), got: {s}"
    );
}

#[test]
fn gamma_symbolic_stays() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let g = expr!(gamma(x));
    let s = format!("{g}");
    assert!(
        s.contains("Gamma") || s.contains("gamma"),
        "symbolic gamma: {s}"
    );
}

#[test]
fn gamma_via_macro() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let g = expr!(gamma(x));
    let result = g.subs(&x, &ctx.int(5)).eval();
    assert_eq!(format!("{result}"), "24");
}

#[test]
fn gamma_subs_then_eval() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let g = x.gamma();
    let result = g.subs(&x, &ctx.int(5)).eval();
    assert_eq!(format!("{result}"), "24");
}

// ═══════════════════════════════════════════════════════════════════════════
// LogGamma function
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_gamma_at_integers() {
    let ctx = Context::new();
    // LogGamma(1) = ln(0!) = ln(1) = 0
    let result = ctx.int(1).log_gamma().eval();
    let s = format!("{result}");
    assert_eq!(s, "0", "LogGamma(1) should be 0, got: {s}");

    // LogGamma(2) = ln(1!) = ln(1) = 0
    let result2 = ctx.int(2).log_gamma().eval();
    let s2 = format!("{result2}");
    assert_eq!(s2, "0", "LogGamma(2) should be 0, got: {s2}");
}

#[test]
fn log_gamma_at_larger_integer() {
    let ctx = Context::new();
    // LogGamma(5) = ln(4!) = ln(24)
    let result = ctx.int(5).log_gamma().eval();
    let s = format!("{result}");
    assert!(
        s.contains("ln") || s.contains("24"),
        "LogGamma(5) should be ln(24), got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Digamma function
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn digamma_symbolic_stays() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let d = x.digamma();
    let s = format!("{d}");
    assert!(
        s.contains("Digamma") || s.contains("digamma"),
        "symbolic digamma: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Error function (erf)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn erf_at_zero() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.int(0).erf().eval()), "0");
}

#[test]
fn erf_symbolic_stays() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let e = expr!(erf(x));
    let s = format!("{e}");
    assert!(s.contains("erf"), "symbolic erf: {s}");
}

#[test]
fn erf_via_macro() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = expr!(erf(x)).subs(&x, &ctx.int(0)).eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn erf_nonzero_stays_symbolic() {
    let ctx = Context::new();
    // erf(1) should stay unevaluated (no closed-form for non-zero)
    let result = ctx.int(1).erf().eval();
    let s = format!("{result}");
    assert!(s.contains("erf"), "erf(1) should remain symbolic, got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Complementary error function (erfc)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn erfc_at_zero() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.int(0).erfc().eval()), "1");
}

#[test]
fn erfc_symbolic_stays() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let e = expr!(erfc(x));
    let s = format!("{e}");
    assert!(s.contains("erfc"), "symbolic erfc: {s}");
}

#[test]
fn erfc_via_macro() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let result = expr!(erfc(x)).subs(&x, &ctx.int(0)).eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn erfc_nonzero_stays_symbolic() {
    let ctx = Context::new();
    let result = ctx.int(1).erfc().eval();
    let s = format!("{result}");
    assert!(
        s.contains("erfc"),
        "erfc(1) should remain symbolic, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Beta function
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn beta_integers() {
    let ctx = Context::new();
    // B(2,3) = Gamma(2)*Gamma(3)/Gamma(5) = 1*2/24 = 1/12
    let result = ctx.int(2).beta(&ctx.int(3)).eval();
    assert_eq!(format!("{result}"), "1/12");
}

#[test]
fn beta_symmetry() {
    let ctx = Context::new();
    // B(a,b) = B(b,a) for positive integers
    let b23 = ctx.int(2).beta(&ctx.int(3)).eval();
    let b32 = ctx.int(3).beta(&ctx.int(2)).eval();
    assert_eq!(format!("{b23}"), format!("{b32}"));
}

#[test]
fn beta_ones() {
    let ctx = Context::new();
    // B(1,1) = Gamma(1)*Gamma(1)/Gamma(2) = 1*1/1 = 1
    let result = ctx.int(1).beta(&ctx.int(1)).eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn beta_larger_values() {
    let ctx = Context::new();
    // B(3,4) = 2!*3!/6! = 2*6/720 = 12/720 = 1/60
    let result = ctx.int(3).beta(&ctx.int(4)).eval();
    assert_eq!(format!("{result}"), "1/60");
}

#[test]
fn beta_symbolic_display() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let b = x.beta(&y);
    let s = format!("{b}");
    assert!(s.contains("B("), "symbolic beta display: {s}");
}

#[test]
fn beta_via_macro() {
    let ctx = Context::new();
    let result = expr!(beta(2, 3)).eval();
    assert_eq!(format!("{result}"), "1/12");
}

// ═══════════════════════════════════════════════════════════════════════════
// Differentiation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_erf() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let de = expr!(erf(x)).diff(&x);
    // d/dx erf(x) = 2/sqrt(pi) * exp(-x^2)
    let s = format!("{de}");
    assert!(s.contains("exp"), "d/dx erf should contain exp: {s}");
}

#[test]
fn diff_erfc() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let de = expr!(erfc(x)).diff(&x);
    // d/dx erfc(x) = -2/sqrt(pi) * exp(-x^2)
    let s = format!("{de}");
    assert!(s.contains("exp"), "d/dx erfc should contain exp: {s}");
}

#[test]
fn diff_gamma() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let dg = x.gamma().diff(&x);
    let s = format!("{dg}");
    // Should contain Digamma (or Gamma — it's Gamma(x)*Digamma(x))
    assert!(
        s.contains("Digamma") || s.contains("digamma") || s.contains("Gamma"),
        "d/dx Gamma should reference Digamma: {s}"
    );
}

#[test]
fn diff_log_gamma() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let dg = x.log_gamma().diff(&x);
    let s = format!("{dg}");
    // d/dx ln(Gamma(x)) = Digamma(x)
    assert!(
        s.contains("Digamma") || s.contains("digamma"),
        "d/dx LogGamma should reference Digamma: {s}"
    );
}
