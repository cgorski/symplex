//! symplex 0.2 — algebraic substitution (`Ex::subs_algebraic`).
//!
//! Every result is verified by substituting `old` back for `new` and
//! comparing numerically with the original at several rational points.

use symplex::prelude::*;

const POINTS: &[(i64, i64)] = &[(3, 7), (-5, 3), (11, 4), (1, 9), (-13, 6), (7, 2)];

/// Check `result[new := old] == original` numerically.
fn assert_back_substitutes(original: &Ex, result: &Ex, old: &Ex, new: &Ex, label: &str) {
    let ctx = original.context();
    let back = result.subs(new, old);
    let syms = original.free_symbols();
    let mut checked = 0;
    for (i, &(p, q)) in POINTS.iter().enumerate() {
        let mut a = original.clone();
        let mut b = back.clone();
        for (j, s) in syms.iter().enumerate() {
            let (pp, qq) = POINTS[(i + j) % POINTS.len()];
            let v = ctx.rational(pp + p, qq + q);
            a = a.subs(s, &v);
            b = b.subs(s, &v);
        }
        match (a.eval_f64(), b.eval_f64()) {
            (Ok(va), Ok(vb)) if !va.is_finite() && !vb.is_finite() => {} // overflow on both sides
            (Ok(va), Ok(vb)) => {
                assert!(
                    (va - vb).abs() <= 1e-9 * (1.0 + va.abs().max(vb.abs())),
                    "{label}: {va} vs {vb}\n  original: {original}\n  result:   {result}\n  back:     {back}"
                );
                checked += 1;
            }
            (Err(_), Err(_)) => {}
            (ra, rb) => panic!(
                "{label}: evaluation mismatch {ra:?} vs {rb:?}\n  original: {original}\n  result: {result}"
            ),
        }
    }
    assert!(checked >= 2, "{label}: too few evaluable points");
}

fn s(e: &Ex) -> String {
    format!("{e}")
}

// ═══════════════════════════════════════════════════════════════════════════
// Powers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn power_exact_multiple() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let (old, new) = (x.powi(2), y.clone());
    let e = x.powi(4);
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "y^2");
    assert_back_substitutes(&e, &r, &old, &new, "x^4");
}

#[test]
fn power_with_remainder() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let (old, new) = (x.powi(2), y.clone());
    let e = x.powi(3);
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "x*y");
    assert_back_substitutes(&e, &r, &old, &new, "x^3");
    let e5 = x.powi(5);
    assert_eq!(s(&e5.subs_algebraic(&old, &new)), "x*y^2");
}

#[test]
fn power_negative_exponent() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let (old, new) = (x.powi(2), y.clone());
    let e = x.powi(-2);
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "1/y");
    assert_back_substitutes(&e, &r, &old, &new, "1/x^2");
    let e3 = x.powi(-3);
    let r3 = e3.subs_algebraic(&old, &new);
    assert_eq!(s(&r3), "1/(x*y)");
    assert_back_substitutes(&e3, &r3, &old, &new, "1/x^3");
}

#[test]
fn power_too_small_is_untouched() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &x + 1;
    assert_eq!(e.subs_algebraic(&x.powi(2), &y), e);
    // x^(1/2) is not a multiple of x^2.
    assert_eq!(x.sqrt().subs_algebraic(&x.powi(2), &y), x.sqrt());
}

#[test]
fn power_sqrt_old() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let (old, new) = (x.sqrt(), y.clone());
    let e = &x * 3 + &x.sqrt();
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "3*y^2 + y");
    assert_back_substitutes(&e, &r, &old, &new, "sqrt");
}

#[test]
fn power_inside_functions_and_sums() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let (old, new) = (x.powi(2), y.clone());
    let e = &x.powi(4).sin() + &x.powi(6) * 2 + &x;
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "2*y^3 + x + sin(y^2)");
    assert_back_substitutes(&e, &r, &old, &new, "nested");
}

#[test]
fn exp_as_power_of_e() {
    let ctx = Context::new();
    let (x, t) = (ctx.symbol("x"), ctx.symbol("t"));
    let (old, new) = (x.exp(), t.clone());
    let e = (&x * 2).exp();
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "t^2");
    assert_back_substitutes(&e, &r, &old, &new, "exp(2x)");
    let f = (-&x).exp() + (&x * 3).exp();
    let rf = f.subs_algebraic(&old, &new);
    assert_eq!(s(&rf), "t^3 + 1/t");
    assert_back_substitutes(&f, &rf, &old, &new, "exp(-x) + exp(3x)");
}

#[test]
fn exp_with_symbolic_remainder() {
    let ctx = Context::new();
    let (x, y, t) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("t"));
    let (old, new) = (x.exp(), t.clone());
    let e = (&x * 2 + &y).exp();
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "t^2*exp(y)");
    assert_back_substitutes(&e, &r, &old, &new, "exp(2x + y)");
}

#[test]
fn symbol_old_behaves_like_structural_subs() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &x.powi(3).sin() + &x * 2;
    assert_eq!(e.subs_algebraic(&x, &y), e.subs(&x, &y));
}

// ═══════════════════════════════════════════════════════════════════════════
// Products
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn product_inside_larger_product() {
    let ctx = Context::new();
    let (x, y, z, w) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("z"),
        ctx.symbol("w"),
    );
    let (old, new) = (&x * &y, w.clone());
    let e = &x * &y * &z * 2;
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "2*w*z");
    assert_back_substitutes(&e, &r, &old, &new, "2xyz");
}

#[test]
fn product_with_powers() {
    let ctx = Context::new();
    let (x, y, w) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("w"));
    let (old, new) = (&x * &y, w.clone());
    let e = &x.powi(2) * &y.powi(3);
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "y*w^2");
    assert_back_substitutes(&e, &r, &old, &new, "x^2 y^3");
}

#[test]
fn product_reciprocal() {
    let ctx = Context::new();
    let (x, y, w) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("w"));
    let (old, new) = (&x * &y, w.clone());
    let e = 1 / (&x * &y);
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "1/w");
    assert_back_substitutes(&e, &r, &old, &new, "1/(xy)");
}

#[test]
fn product_with_coefficient_in_old() {
    let ctx = Context::new();
    let (x, y, w) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("w"));
    let (old, new) = (&x * &y * 2, w.clone());
    let e = &x * &y * 6;
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "3*w");
    assert_back_substitutes(&e, &r, &old, &new, "6xy / 2xy");
    let f = &x.powi(2) * &y.powi(2) * 8;
    let rf = f.subs_algebraic(&old, &new);
    assert_eq!(s(&rf), "2*w^2");
    assert_back_substitutes(&f, &rf, &old, &new, "8x²y² / (2xy)²");
}

#[test]
fn product_missing_factor_is_untouched() {
    let ctx = Context::new();
    let (x, y, z, w) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("z"),
        ctx.symbol("w"),
    );
    let e = &x * &z * 2;
    assert_eq!(e.subs_algebraic(&(&x * &y), &w), e);
    // Mixed exponent signs: x^2 / y is not a power of x*y.
    let f = &x.powi(2) / &y;
    assert_eq!(f.subs_algebraic(&(&x * &y), &w), f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Sums
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sum_subset_of_terms() {
    let ctx = Context::new();
    let (a, b, c, d) = (
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
        ctx.symbol("d"),
    );
    let (old, new) = (&a + &b, d.clone());
    let e = &a + &b + &c;
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "c + d");
    assert_back_substitutes(&e, &r, &old, &new, "a+b+c");
}

#[test]
fn sum_with_common_multiple() {
    let ctx = Context::new();
    let (a, b, c, d) = (
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
        ctx.symbol("d"),
    );
    let (old, new) = (&a + &b, d.clone());
    let e = &a * 2 + &b * 2 + &c;
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "c + 2*d");
    assert_back_substitutes(&e, &r, &old, &new, "2a+2b+c");
    let f = -&a - &b + &c;
    let rf = f.subs_algebraic(&old, &new);
    assert_eq!(s(&rf), "c - d");
    assert_back_substitutes(&f, &rf, &old, &new, "-a-b+c");
}

#[test]
fn sum_with_constant_remainder() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let (old, new) = (&x + 1, y.clone());
    let e = &x + 3;
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "y + 2");
    assert_back_substitutes(&e, &r, &old, &new, "x+3");
}

#[test]
fn sum_mismatched_coefficients_untouched() {
    let ctx = Context::new();
    let (a, b, c, d) = (
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
        ctx.symbol("d"),
    );
    let e = &a * 2 + &b + &c;
    assert_eq!(e.subs_algebraic(&(&a + &b), &d), e);
}

#[test]
fn sum_inside_function() {
    let ctx = Context::new();
    let (a, b, c, d) = (
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
        ctx.symbol("d"),
    );
    let (old, new) = (&a + &b, d.clone());
    let e = (&a + &b + &c).sin() * &(&a + &b);
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "d*sin(c + d)");
    assert_back_substitutes(&e, &r, &old, &new, "sin(a+b+c)(a+b)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Misc
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn identical_old_and_new_is_noop() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(4);
    assert_eq!(e.subs_algebraic(&x.powi(2), &x.powi(2)), e);
}

#[test]
fn new_may_be_compound() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let (old, new) = (x.powi(2), &y + 1);
    let e = &x.powi(4) + &x.powi(2);
    let r = e.subs_algebraic(&old, &new);
    assert_eq!(s(&r), "y + (y + 1)^2 + 1");
    // Back-substitute by solving new = old for y: y = x^2 - 1.
    let back = r.subs(&y, &(&x.powi(2) - 1));
    for p in [2i64, 3, -5, 7] {
        let a = e.subs_i64(&x, p).eval_f64().unwrap();
        let b = back.subs_i64(&x, p).eval_f64().unwrap();
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}

#[test]
fn value_preservation_sweep() {
    let ctx = Context::new();
    let (x, y, z, u) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("z"),
        ctx.symbol("u"),
    );
    let cases: Vec<(Ex, Ex)> = vec![
        (
            x.powi(2),
            (&x.powi(6) + &x.powi(3) * &y + 1 / &x.powi(4)).exp(),
        ),
        (&x * &y, &x.powi(3) * &y.powi(3) * &z + &x * &y / &z),
        (&x + &y, (&x + &y + &z).powi(2) * &(&x * 3 + &y * 3)),
        (x.exp(), &(&x * 4).exp() + &(-&x * 2).exp() * &z),
    ];
    for (old, e) in cases {
        let r = e.subs_algebraic(&old, &u);
        assert_ne!(r, e, "{e} should change under {old} → u");
        assert_back_substitutes(&e, &r, &old, &u, &format!("{e}"));
    }
}
