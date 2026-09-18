//! 0.2 complex-analysis nodes: `Re`, `Im`, `Conjugate`, `Arg`.
//!
//! These tests exercise the public `Ex` API: construction-time folding,
//! assumption awareness (bare symbols are *not* assumed real), display /
//! LaTeX / parse / JSON round-trips, differentiation, numerical evaluation
//! and the derived helpers (`as_real_imag`, `expand_complex`, `polar`,
//! `abs_squared`, `is_real_valued`).

use symplex::prelude::*;

fn real(ctx: &Context, name: &str) -> Ex {
    ctx.symbol_with(name, &[Assumption::Real])
}

fn positive(ctx: &Context, name: &str) -> Ex {
    ctx.symbol_with(name, &[Assumption::Positive])
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction-time folding
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn numbers_and_constants_are_real() {
    let ctx = Context::new();
    for c in [
        ctx.int(7),
        ctx.rational(3, 5),
        ctx.pi(),
        ctx.e(),
        ctx.euler_gamma(),
        ctx.catalan(),
        ctx.golden_ratio(),
    ] {
        assert_eq!(c.re(), c, "re({c}) should be itself");
        assert!(c.im().is_zero_structural(), "im({c}) should be 0");
        assert_eq!(c.conjugate(), c, "conjugate({c}) should be itself");
        assert!(c.arg().is_zero_structural(), "arg({c}) should be 0");
    }
    let neg = ctx.rational(-3, 5);
    assert_eq!(neg.re(), neg);
    assert_eq!(neg.conjugate(), neg);
    assert_eq!(format!("{}", ctx.int(-4).arg()), "pi");
    assert_eq!(format!("{}", ctx.rational(-1, 3).arg()), "pi");
}

#[test]
fn imaginary_unit() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert!(i.re().is_zero_structural());
    assert!(i.im().is_one_structural());
    assert_eq!(format!("{}", i.conjugate()), "-I");
    assert_eq!(format!("{}", i.arg()), "1/2*pi");
    assert_eq!(format!("{}", (-&i).arg()), "-1/2*pi");
}

#[test]
fn gaussian_integer_decomposition() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let z = &ctx.int(3) + &(&ctx.int(4) * &i);
    assert_eq!(format!("{}", z.re()), "3");
    assert_eq!(format!("{}", z.im()), "4");
    assert_eq!(format!("{}", z.conjugate()), "-4*I + 3");
    let (re, im) = z.as_real_imag();
    assert_eq!(format!("{re}"), "3");
    assert_eq!(format!("{im}"), "4");
    // (3+4i)(1-2i) = 3 - 6i + 4i + 8 = 11 - 2i
    let w = &ctx.int(1) - &(&ctx.int(2) * &i);
    let p = (&z * &w).expand();
    assert_eq!(format!("{}", p.re().eval()), "11");
    assert_eq!(format!("{}", p.im().eval()), "-2");
}

#[test]
fn bare_symbol_is_not_assumed_real() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    assert_eq!(format!("{}", z.re()), "re(z)");
    assert_eq!(format!("{}", z.im()), "im(z)");
    assert_eq!(format!("{}", z.conjugate()), "conjugate(z)");
    assert_eq!(format!("{}", z.arg()), "arg(z)");
    assert_eq!(z.re().expr_type(), symplex::expr::ExprType::Function);
    // None of these count as "unevaluated" — they are functions.
    assert!(!z.re().has_unevaluated());
    assert!(!z.conjugate().has_unevaluated());
}

#[test]
fn real_symbol_folds() {
    let ctx = Context::new();
    let x = real(&ctx, "x");
    assert_eq!(x.re(), x);
    assert!(x.im().is_zero_structural());
    assert_eq!(x.conjugate(), x);
    // sign unknown → arg stays symbolic
    assert_eq!(format!("{}", x.arg()), "arg(x)");
    let p = positive(&ctx, "p");
    assert!(p.arg().is_zero_structural());
    assert_eq!(format!("{}", (-&p).arg()), "pi");
    // Integer/positive imply real.
    let n = ctx.symbol_with("n", &[Assumption::Integer]);
    assert_eq!(n.re(), n);
    assert_eq!(n.conjugate(), n);
}

#[test]
fn imaginary_symbol_folds() {
    let ctx = Context::new();
    let y = ctx.symbol_with("y", &[Assumption::Imaginary]);
    assert!(y.re().is_zero_structural());
    assert_eq!(format!("{}", y.im()), "-y*I");
    assert_eq!(format!("{}", y.conjugate()), "-y");
}

#[test]
fn linearity_over_add_and_real_scaling() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let x = real(&ctx, "x");
    let i = ctx.i_unit();
    let e = &(&z + &x) + &(&ctx.int(2) * &i);
    assert_eq!(format!("{}", e.re()), "x + re(z)");
    assert_eq!(format!("{}", e.im()), "im(z) + 2");
    assert_eq!(format!("{}", (&ctx.int(3) * &z).re()), "3*re(z)");
    assert_eq!(format!("{}", (&x * &z).re()), "x*re(z)");
    assert_eq!(format!("{}", (&i * &z).re()), "-im(z)");
    assert_eq!(format!("{}", (&i * &z).im()), "re(z)");
}

#[test]
fn product_of_unknowns_stays_compact() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let w = ctx.symbol("w");
    assert_eq!(format!("{}", (&z * &w).re()), "re(w*z)");
    assert_eq!(format!("{}", (&z * &w).im()), "im(w*z)");
    // …but a fully known product is expanded.
    let a = real(&ctx, "a");
    let b = real(&ctx, "b");
    let i = ctx.i_unit();
    let z1 = &a + &(&b * &i);
    let sq = z1.powi(2);
    assert_eq!(format!("{}", sq.re()), "a^2 - b^2");
    assert_eq!(format!("{}", sq.im()), "2*a*b");
}

#[test]
fn interplay_identities() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let r = z.re();
    let m = z.im();
    let c = z.conjugate();
    assert_eq!(r.re(), r);
    assert!(r.im().is_zero_structural());
    assert_eq!(m.re(), m);
    assert!(m.im().is_zero_structural());
    assert_eq!(c.re(), r);
    assert_eq!(c.im(), -&m);
    assert_eq!(c.conjugate(), z);
    assert_eq!(z.arg().re(), z.arg());
    assert_eq!(z.abs().re(), z.abs());
    assert_eq!(z.abs().conjugate(), z.abs());
}

#[test]
fn conjugate_commutes_with_analytic_functions_only() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let cz = z.conjugate();
    assert_eq!(z.sin().conjugate(), cz.sin());
    assert_eq!(z.cos().conjugate(), cz.cos());
    assert_eq!(z.exp().conjugate(), cz.exp());
    assert_eq!(z.sinh().conjugate(), cz.sinh());
    assert_eq!(z.cosh().conjugate(), cz.cosh());
    assert_eq!(z.gamma().conjugate(), cz.gamma());
    assert_eq!(z.erf().conjugate(), cz.erf());
    assert_eq!(z.zeta().conjugate(), cz.zeta());
    assert_eq!(z.powi(3).conjugate(), cz.powi(3));
    assert_eq!((&z + 1).conjugate(), &cz + 1);
    // Branch cuts: stay unevaluated.
    assert_eq!(format!("{}", z.ln().conjugate()), "conjugate(ln(z))");
    assert_eq!(format!("{}", z.sqrt().conjugate()), "conjugate(sqrt(z))");
    assert_eq!(format!("{}", z.asin().conjugate()), "conjugate(asin(z))");
    // Positive base with complex exponent: conj(2^z) = 2^conj(z).
    let two = ctx.int(2);
    assert_eq!(two.pow(&z).conjugate(), two.pow(&cz));
}

#[test]
fn exp_and_trig_of_complex_arguments() {
    let ctx = Context::new();
    let x = real(&ctx, "x");
    let y = real(&ctx, "y");
    let i = ctx.i_unit();
    let z = &x + &(&y * &i);
    // exp(x + iy) = e^x cos y + i e^x sin y
    let e = z.exp();
    assert_eq!(format!("{}", e.re()), "cos(y)*exp(x)");
    assert_eq!(format!("{}", e.im()), "sin(y)*exp(x)");
    // sin(x + iy) = sin x cosh y + i cos x sinh y
    let s = z.sin();
    assert_eq!(format!("{}", s.re()), "sin(x)*cosh(y)");
    assert_eq!(format!("{}", s.im()), "cos(x)*sinh(y)");
    // cos(x + iy) = cos x cosh y − i sin x sinh y
    let c = z.cos();
    assert_eq!(format!("{}", c.re()), "cos(x)*cosh(y)");
    assert_eq!(format!("{}", c.im()), "-sin(x)*sinh(y)");
    // Euler: e^{ix}
    let eix = (&i * &x).exp();
    assert_eq!(format!("{}", eix.re()), "cos(x)");
    assert_eq!(format!("{}", eix.im()), "sin(x)");
    // ln(x + iy) = ½ ln(x² + y²) + i atan2(y, x)
    let l = z.ln();
    assert_eq!(format!("{}", l.re()), "1/2*ln(x^2 + y^2)");
    assert_eq!(format!("{}", l.im()), "atan2(y, x)");
}

#[test]
fn branch_cuts_stay_unevaluated() {
    let ctx = Context::new();
    let x = real(&ctx, "x");
    let z = ctx.symbol("z");
    // sqrt of a real with unknown sign, or of a complex unknown
    assert_eq!(format!("{}", x.sqrt().re()), "re(sqrt(x))");
    assert_eq!(format!("{}", z.sqrt().im()), "im(sqrt(z))");
    // ln of unknown-sign real: re = ln|x|, im = arg(x)
    assert_eq!(format!("{}", x.ln().re()), "ln(abs(x))");
    assert_eq!(format!("{}", x.ln().im()), "arg(x)");
    // but positive base is fine
    let p = positive(&ctx, "p");
    assert_eq!(p.sqrt().re(), p.sqrt());
    assert_eq!(p.ln().re(), p.ln());
    assert!(p.ln().im().is_zero_structural());
}

#[test]
fn arg_rules() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let one = ctx.int(1);
    assert_eq!(format!("{}", (&one + &i).arg()), "1/4*pi");
    assert_eq!(format!("{}", (&ctx.int(-1) + &i).arg()), "3/4*pi");
    assert_eq!(format!("{}", (&ctx.int(-1) - &i).arg()), "-3/4*pi");
    assert_eq!(format!("{}", (&ctx.int(3) * &i).arg()), "1/2*pi");
    // arg(0) is undefined
    assert_eq!(ctx.zero().arg(), ctx.nan());
    // positive factors drop out
    let z = ctx.symbol("z");
    assert_eq!((&ctx.int(5) * &z).arg(), z.arg());
    let p = positive(&ctx, "p");
    assert_eq!((&p * &z).arg(), z.arg());
    // arg(exp(i·k·π)) reduces into (−π, π]
    let k = ctx.rational(7, 2);
    let e = (&(&i * &k) * &ctx.pi()).exp();
    assert_eq!(format!("{}", e.arg()), "-1/2*pi");
    // symbolic real x, y: atan2
    let x = real(&ctx, "x");
    let y = real(&ctx, "y");
    assert_eq!(format!("{}", (&x + &(&i * &y)).arg()), "atan2(y, x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumption propagation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn assumptions_of_complex_nodes() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    assert_eq!(z.re().is_real(), Some(true));
    assert_eq!(z.im().is_real(), Some(true));
    assert_eq!(z.arg().is_real(), Some(true));
    assert_eq!(z.conjugate().is_real(), None);
    assert_eq!(z.is_real(), None);
    let x = real(&ctx, "x");
    // conjugate(x) folds to x, so build the node through a non-folding path
    let n = ctx.symbol_with("n", &[Assumption::Integer]);
    assert_eq!((&n + &z).conjugate().is_real(), None);
    assert_eq!(x.conjugate().is_real(), Some(true));
    assert_eq!(z.abs().is_real(), Some(true));
    assert_eq!(z.abs().is_negative(), Some(false));
}

// ═══════════════════════════════════════════════════════════════════════════
// Output formats & parsing
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_latex_pretty() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    assert_eq!(format!("{}", z.re()), "re(z)");
    assert_eq!(format!("{}", z.im()), "im(z)");
    assert_eq!(format!("{}", z.conjugate()), "conjugate(z)");
    assert_eq!(format!("{}", z.arg()), "arg(z)");
    assert_eq!(z.re().to_latex(), r"\Re\left(z\right)");
    assert_eq!(z.im().to_latex(), r"\Im\left(z\right)");
    assert_eq!(z.conjugate().to_latex(), r"\overline{z}");
    assert_eq!(z.arg().to_latex(), r"\arg\left(z\right)");
    assert_eq!(z.re().pretty_ascii().trim(), "re(z)");
    assert!(z.conjugate().pretty().contains('‾'));
    assert_eq!(z.conjugate().pretty_ascii().trim(), "conjugate(z)");
}

#[test]
fn parse_round_trip() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    for e in [z.re(), z.im(), z.conjugate(), z.arg()] {
        let s = format!("{e}");
        let back = ctx.parse(&s).unwrap();
        assert_eq!(back, e, "round trip of {s}");
    }
    // `conj` alias
    assert_eq!(ctx.parse("conj(z)").unwrap(), z.conjugate());
    // Parsing folds numeric arguments.
    assert_eq!(format!("{}", ctx.parse("re(3 + 4*I)").unwrap()), "3");
    assert_eq!(format!("{}", ctx.parse("arg(-2)").unwrap()), "pi");
}

#[test]
fn tree_json_round_trip() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let e = &(&z.re() + &z.im()) * &(&z.conjugate() + &z.arg());
    let tree = e.to_tree();
    let json = serde_json::to_string(&tree).unwrap();
    assert!(json.contains("\"Conjugate\""));
    let tree2: symplex::tree::ExprTree = serde_json::from_str(&json).unwrap();
    let back = ctx.from_tree(&tree2);
    assert_eq!(back, e);
}

// ═══════════════════════════════════════════════════════════════════════════
// Substitution and evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn substitution_refolds() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let i = ctx.i_unit();
    let w = &ctx.int(3) + &(&ctx.int(4) * &i);
    assert_eq!(format!("{}", z.re().subs(&z, &w)), "3");
    assert_eq!(format!("{}", z.im().subs(&z, &w)), "4");
    assert_eq!(format!("{}", z.conjugate().subs(&z, &w)), "-4*I + 3");
    let a = (&ctx.int(1) + &i).arg();
    assert_eq!(z.arg().subs(&z, &(&ctx.int(1) + &i)), a);
    // eval() also refolds once the argument is concrete
    let e = z.re().subs(&z, &(&ctx.int(2) * &i)).eval();
    assert!(e.is_zero_structural());
}

#[test]
fn numerical_evaluation() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let i = ctx.i_unit();
    // Build unevaluated nodes, then substitute a complex number and evalf.
    let w = &ctx.int(3) + &(&ctx.int(-4) * &i);
    let re_v = z.re().subs(&z, &w).eval_f64().unwrap();
    let im_v = z.im().subs(&z, &w).eval_f64().unwrap();
    let arg_v = z.arg().subs(&z, &w).eval_f64().unwrap();
    assert!(approx(re_v, 3.0, 1e-12));
    assert!(approx(im_v, -4.0, 1e-12));
    assert!(approx(arg_v, (-4.0f64).atan2(3.0), 1e-12));
    let (cr, ci) = z.conjugate().subs(&z, &w).eval_complex64().unwrap();
    assert!(approx(cr, 3.0, 1e-12) && approx(ci, 4.0, 1e-12));
    // arg of exp(i·t) for numeric t through the general path
    let t = ctx.rational(1, 3);
    let v = (&i * &t).exp().arg().eval_f64().unwrap();
    assert!(approx(v, 1.0 / 3.0, 1e-12));
    // Arg via evalf of a genuinely complex sub-expression (2+i)^3
    let cube = (&ctx.int(2) + &i).powi(3);
    let expected = (2.0 + 1.0 * 0.0f64).mul_add(0.0, 0.0) + (11.0f64).atan2(2.0);
    assert!(approx(cube.arg().eval_f64().unwrap(), expected, 1e-12));
}

#[test]
fn eval_decimal_of_complex_parts() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let z = (&ctx.int(1) + &i).ln(); // ln(1+i) = ½ ln 2 + iπ/4
    let re_s = z.re().eval_decimal(25).unwrap();
    assert!(re_s.starts_with("0.34657359027997265470861"), "{re_s}");
    let im_s = z.im().eval_decimal(25).unwrap();
    assert!(im_s.starts_with("0.78539816339744830961566"), "{im_s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Differentiation (derivative along the real axis only)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_with_real_variable() {
    let ctx = Context::new();
    let t = real(&ctx, "t");
    let z = ctx.symbol("z");
    let f = &z * &t.powi(2); // re(z t²)
    let d = f.re().diff(&t);
    assert_eq!(format!("{d}"), "2*t*re(z)");
    let d = f.im().diff(&t);
    assert_eq!(format!("{d}"), "2*t*im(z)");
    let d = f.conjugate().diff(&t);
    assert_eq!(format!("{d}"), "2*t*conjugate(z)");
    // d/dt arg(f) = im(f'/f)
    let g = &z * &t.exp();
    let d = g.arg().diff(&t);
    assert!(d.is_zero_structural(), "arg(z e^t)' = im(1) = 0, got {d}");
    let h = &z + &t;
    let d = h.arg().diff(&t);
    assert_eq!(format!("{d}"), "im(1/(t + z))");
    // constants differentiate to zero
    assert!(z.re().diff(&t).is_zero_structural());
}

#[test]
fn diff_with_unassumed_variable_is_formal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = x.re().diff(&x);
    assert_eq!(format!("{d}"), "Derivative(re(x), x)");
    assert!(d.has_unevaluated());
    assert!(x.re().try_diff(&x).is_err());
    let d = x.conjugate().diff(&x);
    assert_eq!(format!("{d}"), "Derivative(conjugate(x), x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Derived helpers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_complex_polar_abs_squared() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let x = real(&ctx, "x");
    let i = ctx.i_unit();
    assert_eq!(format!("{}", z.expand_complex()), "im(z)*I + re(z)");
    assert_eq!(x.expand_complex(), x);
    let e = (&i * &x).exp().expand_complex();
    assert_eq!(format!("{e}"), "sin(x)*I + cos(x)");

    let w = &ctx.int(3) + &(&ctx.int(4) * &i);
    let (r, theta) = w.polar();
    assert!(approx(r.eval_f64().unwrap(), 5.0, 1e-12));
    assert!(approx(
        theta.eval_f64().unwrap(),
        (4.0f64).atan2(3.0),
        1e-12
    ));
    assert_eq!(format!("{}", w.abs_squared().eval()), "25");
    assert_eq!(format!("{}", z.abs_squared()), "z*conjugate(z)");
    let y = real(&ctx, "y");
    let u = &x + &(&i * &y);
    assert_eq!(format!("{}", u.abs_squared()), "x^2 + y^2");
}

#[test]
fn is_real_valued_three_valued() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let z = ctx.symbol("z");
    let x = real(&ctx, "x");
    assert_eq!(ctx.int(5).is_real_valued(), Some(true));
    assert_eq!(ctx.pi().is_real_valued(), Some(true));
    assert_eq!(x.is_real_valued(), Some(true));
    assert_eq!(x.sin().is_real_valued(), Some(true));
    assert_eq!(x.exp().is_real_valued(), Some(true));
    assert_eq!(i.is_real_valued(), Some(false));
    assert_eq!((&x + &i).is_real_valued(), Some(false));
    assert_eq!(
        (&ctx.int(3) + &(&ctx.int(4) * &i)).is_real_valued(),
        Some(false)
    );
    assert_eq!(z.is_real_valued(), None);
    assert_eq!(z.sin().is_real_valued(), None);
    assert_eq!(z.abs().is_real_valued(), Some(true));
    assert_eq!(z.re().is_real_valued(), Some(true));
    assert_eq!((&z * &z.conjugate()).is_real_valued(), None);
    // i·y with y real and non-zero is not real.
    let p = positive(&ctx, "p");
    assert_eq!((&i * &p).is_real_valued(), Some(false));
    // (The assumption system classifies I*y as IMAGINARY for real y, so
    // this is never reported as real.)
    let y = real(&ctx, "y");
    assert_ne!((&i * &y).is_real_valued(), Some(true));
}

#[test]
fn compact_preserves_complex_nodes() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let e = &z.re() + &(&z.conjugate() * &z.arg());
    let (ctx2, roots) = ctx.compact(std::slice::from_ref(&e));
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), format!("{e}"));
    // and it still behaves like the original
    let z2 = ctx2.symbol("z");
    assert_eq!(format!("{}", z2.re()), "re(z)");
}
