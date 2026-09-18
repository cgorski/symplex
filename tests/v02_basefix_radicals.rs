//! symplex 0.2 base-layer fixes — products of numeric radicals.
//!
//! `canon_mul`'s exponent-grouping pass (`2^(1/2)·3^(1/2) → 6^(1/2)`)
//! must never hand back a `Mul` nested inside a `Mul`; in debug builds
//! that fires the canonical-form `debug_assert`.

use symplex::expr::ExprType;
use symplex::prelude::*;

fn s<T: std::fmt::Display>(e: &T) -> String {
    format!("{e}")
}

fn assert_flat_mul(e: &Ex) {
    if e.expr_type() == ExprType::Mul {
        for a in e.args() {
            assert_ne!(a.expr_type(), ExprType::Mul, "nested Mul inside Mul: {e}");
        }
    }
}

#[test]
fn sqrt6_over_3_times_sqrt3_over_3() {
    let ctx = Context::new();
    let a = ctx.int(6).sqrt() / 3;
    let b = ctx.int(3).sqrt() / 3;
    let p = &a * &b;
    assert_flat_mul(&p);
    // √6·√3/9 = √18/9 = 3√2/9 = √2/3
    let v = p.eval_f64().unwrap();
    assert!((v - 2f64.sqrt() / 3.0).abs() < 1e-12, "{p} = {v}");
}

#[test]
fn sqrt2_sqrt2_sqrt3_chain() {
    let ctx = Context::new();
    let r2 = ctx.int(2).sqrt();
    let r3 = ctx.int(3).sqrt();
    let p = &(&r2 * &r2) * &r3;
    assert_flat_mul(&p);
    assert_eq!(s(&p), "2*sqrt(3)", "{p}");
    let q = &r2 * &(&r2 * &r3);
    assert_eq!(p, q);
}

#[test]
fn sqrt2_sqrt3_sqrt6_is_six() {
    let ctx = Context::new();
    let r2 = ctx.int(2).sqrt();
    let r3 = ctx.int(3).sqrt();
    let r6 = ctx.int(6).sqrt();
    assert_eq!(s(&(&(&r2 * &r3) * &r6)), "6");
    assert_eq!(s(&(&r2 * &(&r3 * &r6))), "6");
    assert_eq!(s(&(&(&r6 * &r2) * &r3)), "6");
}

#[test]
fn radical_products_with_rational_coefficients() {
    let ctx = Context::new();
    let r2 = ctx.int(2).sqrt();
    let r3 = ctx.int(3).sqrt();
    let r6 = ctx.int(6).sqrt();
    let cases = [
        (&r2 / 2) * (&r3 / 3),
        (&r6 / 3) * (&r6 / 2),
        (&r2 * &r3) * (&r2 / 5),
        ctx.rational(1, 2) * &r2 * &r3 * &r6,
        (&r2 / 3) * (&r3 / 3) * (&r6 / 3),
    ];
    let expect = [
        6f64.sqrt() / 6.0,
        1.0,
        2.0 * 3f64.sqrt() / 5.0,
        3.0,
        6.0 / 27.0,
    ];
    for (p, e) in cases.iter().zip(expect) {
        assert_flat_mul(p);
        let v = p.eval_f64().unwrap();
        assert!((v - e).abs() < 1e-12, "{p} = {v}, expected {e}");
    }
}

#[test]
fn cube_roots_combine_too() {
    let ctx = Context::new();
    let third = ctx.rational(1, 3);
    let c2 = ctx.int(2).pow(&third);
    let c4 = ctx.int(4).pow(&third);
    let c3 = ctx.int(3).pow(&third);
    let p = &(&c2 * &c3) * &c4;
    assert_flat_mul(&p);
    // ∛2·∛3·∛4 = ∛24 = 2·∛3
    let v = p.eval_f64().unwrap();
    assert!((v - 2.0 * 3f64.cbrt()).abs() < 1e-12, "{p} = {v}");
}

#[test]
fn integer_power_of_mul_base_after_exponents_combine() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let xy = &x * &y;
    let a = xy.pow(&ctx.rational(1, 2));
    let b = xy.pow(&ctx.rational(3, 2));
    let p = &a * &b;
    assert_flat_mul(&p);
    assert_eq!(p, &x.powi(2) * &y.powi(2), "{p}");
    let q = &(&a * &b) * &x;
    assert_flat_mul(&q);
    assert_eq!(q, &x.powi(3) * &y.powi(2), "{q}");
}

#[test]
fn negative_base_radicals_combine_flat() {
    let ctx = Context::new();
    let quarter = ctx.rational(1, 4);
    let r = ctx.int(-2).pow(&quarter);
    let r6 = ctx.int(6).sqrt();
    // (-2)^(1/4)·(-2)^(1/4) = (-2)^(1/2) = i·√2 ; times √6 → i·√12 = 2·i·√3
    let p = &(&r * &r) * &r6;
    assert_flat_mul(&p);
    let (re, im) = p.eval_complex64().unwrap();
    assert!(
        re.abs() < 1e-12 && (im - 2.0 * 3f64.sqrt()).abs() < 1e-12,
        "{p}"
    );
    let p2 = &(&r * &r6) * &r;
    assert_eq!(p, p2);
}

#[test]
fn together_with_symbolic_radical_denominator() {
    let ctx = Context::new();
    let w = ctx.symbol("omega");
    let neg_w2 = -w.powi(2);
    let root = neg_w2.sqrt();
    let expr = &neg_w2 / &root - &root;
    let t = expr.together();
    assert_flat_mul(&t);
    // Must not panic; the value is preserved at a sample point.
    let (re0, im0) = expr.subs_i64(&w, 2).eval_complex64().unwrap();
    let (re1, im1) = t.subs_i64(&w, 2).eval_complex64().unwrap();
    assert!(
        (re0 - re1).abs() < 1e-9 && (im0 - im1).abs() < 1e-9,
        "{expr} vs {t}"
    );
}
