//! symplex 0.9 — algebraic numbers and polynomial algebra on `Ex`:
//! `minimal_polynomial`, `gcd_all` / `lcm_all`, `Ex::groebner` /
//! `reduce_modulo`, `real_roots` / `root_of`, `factor_mod`,
//! `resultant_symbolic` / `discriminant_symbolic`.
//!
//! Reference values are from SymPy 1.14 (`symplex/.venv`), quoted in the
//! comments of each test; the functions that should be exact are asserted
//! structurally, `RootOf` values numerically.

use symplex::multipoly::MonomialOrder;
use symplex::prelude::*;

fn s(e: &Ex) -> String {
    format!("{e}")
}

fn strings(es: &[Ex]) -> Vec<String> {
    es.iter().map(s).collect()
}

/// `p` is polynomial in the free symbols and `q` divides it exactly
/// (`Ex::gcd_all` normalises, so "up to a constant" is checked this way).
fn assert_divides(q: &Ex, p: &Ex, label: &str) {
    let quotient = (p / q).ratsimp();
    let vars = p.free_symbols();
    assert!(
        Poly::new(&quotient, &vars.iter().collect::<Vec<_>>()).is_some(),
        "{label}: {q} does not divide {p} (quotient {quotient})"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. minimal_polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn minimal_polynomial_sqrt2_plus_sqrt3() {
    // SymPy: minimal_polynomial(sqrt(2) + sqrt(3), x) == x**4 - 10*x**2 + 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.int(2).sqrt() + ctx.int(3).sqrt();
    let m = a.minimal_polynomial(&x).expect("√2 + √3 is algebraic");
    assert_eq!(m, &x.powi(4) - &x.powi(2) * 10 + 1, "got {m}");
}

#[test]
fn minimal_polynomial_cube_root_of_two() {
    // SymPy: minimal_polynomial(cbrt(2), x) == x**3 - 2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cbrt2 = ctx.int(2).pow(&ctx.rational(1, 3));
    let m = cbrt2.minimal_polynomial(&x).expect("∛2 is algebraic");
    assert_eq!(m, &x.powi(3) - 2, "got {m}");
}

#[test]
fn minimal_polynomial_rational_is_integer_primitive() {
    // SymPy: minimal_polynomial(Rational(3, 4), x) == 4*x - 3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = ctx
        .rational(3, 4)
        .minimal_polynomial(&x)
        .expect("3/4 is algebraic");
    assert_eq!(m, &x * 4 - 3, "got {m}");
}

#[test]
fn minimal_polynomial_golden_ratio_and_imaginary_unit() {
    // SymPy: minimal_polynomial((1 + sqrt(5))/2, x) == x**2 - x - 1;
    //        minimal_polynomial(I, x) == x**2 + 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let phi = (ctx.int(1) + ctx.int(5).sqrt()) / 2;
    assert_eq!(phi.minimal_polynomial(&x).unwrap(), &x.powi(2) - &x - 1);
    assert_eq!(
        ctx.golden_ratio().minimal_polynomial(&x).unwrap(),
        &x.powi(2) - &x - 1
    );
    assert_eq!(ctx.i_unit().minimal_polynomial(&x).unwrap(), &x.powi(2) + 1);
}

#[test]
fn minimal_polynomial_powers_and_reciprocals_of_algebraic_numbers() {
    // SymPy: minimal_polynomial((1 + sqrt(2))**2, x)  == x**2 - 6*x + 1
    //        minimal_polynomial(1/(1 + sqrt(2)), x)   == x**2 + 2*x - 1
    //        minimal_polynomial((1 + sqrt(2))**-3, x) == x**2 + 14*x - 1
    //        minimal_polynomial(sqrt(3 + 2*sqrt(2)), x) == x**2 - 2*x - 1   (= 1 + √2)
    //        minimal_polynomial(1/sqrt(2), x)         == 2*x**2 - 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.int(1) + ctx.int(2).sqrt();
    assert_eq!(
        a.powi(2).minimal_polynomial(&x).unwrap(),
        &x.powi(2) - &x * 6 + 1
    );
    assert_eq!(
        a.powi(-1).minimal_polynomial(&x).unwrap(),
        &x.powi(2) + &x * 2 - 1
    );
    assert_eq!(
        a.powi(-3).minimal_polynomial(&x).unwrap(),
        &x.powi(2) + &x * 14 - 1
    );
    let nested = (ctx.int(3) + ctx.int(2).sqrt() * 2).sqrt();
    assert_eq!(
        nested.minimal_polynomial(&x).unwrap(),
        &x.powi(2) - &x * 2 - 1
    );
    let inv_sqrt2 = ctx.int(1) / ctx.int(2).sqrt();
    assert_eq!(
        inv_sqrt2.minimal_polynomial(&x).unwrap(),
        &x.powi(2) * 2 - 1
    );
}

#[test]
fn minimal_polynomial_vanishes_at_the_number() {
    // m(α) must be 0 numerically for a few algebraic α.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let alphas = [
        ctx.int(2).sqrt() + ctx.int(3).sqrt(),
        ctx.int(2).pow(&ctx.rational(1, 3)) - ctx.int(5).sqrt(),
        (ctx.int(1) + ctx.int(2).sqrt()) * ctx.int(3).pow(&ctx.rational(1, 4)),
    ];
    for alpha in &alphas {
        let m = alpha
            .minimal_polynomial(&x)
            .unwrap_or_else(|| panic!("{alpha} is algebraic"));
        let v = m.subs(&x, alpha).eval_f64().unwrap();
        assert!(v.abs() < 1e-9, "m(α) = {v} for α = {alpha}, m = {m}");
    }
}

#[test]
fn minimal_polynomial_none_for_non_algebraic_input() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(
        ctx.pi().minimal_polynomial(&x).is_none(),
        "π is transcendental"
    );
    assert!(
        ctx.e().minimal_polynomial(&x).is_none(),
        "e is transcendental"
    );
    assert!(
        x.minimal_polynomial(&x).is_none(),
        "a free symbol is not a number"
    );
    assert!((&x + ctx.int(2).sqrt()).minimal_polynomial(&x).is_none());
    assert!(ctx.int(2).sin().minimal_polynomial(&x).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. gcd_all / lcm_all
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gcd_all_of_difference_of_squares_and_linear_factor() {
    // SymPy: gcd(x**2 - y**2, x - y) == x - y
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = &x.powi(2) - &y.powi(2);
    let g = &x - &y;
    let d = f.gcd_all(&g).expect("both are polynomials");
    assert_divides(&d, &f, "gcd | f");
    assert_divides(&d, &g, "gcd | g");
    assert_divides(&g, &d, "g | gcd (gcd has degree 1)");
    assert_eq!(
        d, g,
        "normalised: integer, primitive, positive leading coefficient"
    );
}

#[test]
fn gcd_all_keeps_integer_content_like_sympy_over_zz() {
    // SymPy: gcd(2*x, 4*x) == 2*x; gcd(2*x**2 - 2*y**2, 4*x - 4*y) == 2*x - 2*y;
    //        gcd(x/2, x) == x; gcd(4, 6) == 2
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    assert_eq!((&x * 2).gcd_all(&(&x * 4)).unwrap(), &x * 2);
    let d = (&x.powi(2) * 2 - &y.powi(2) * 2)
        .gcd_all(&(&x * 4 - &y * 4))
        .unwrap();
    assert_eq!(d, &x * 2 - &y * 2, "got {d}");
    assert_eq!((&x / 2).gcd_all(&x).unwrap(), x);
    assert_eq!(ctx.int(4).gcd_all(&ctx.int(6)).unwrap(), ctx.int(2));
}

#[test]
fn gcd_all_three_variables_and_repeated_factor() {
    // SymPy: gcd(x*y*z + x*y, x*z + x) == x*z + x
    //        gcd((x+y)**3*(x-y), (x+y)**2*(x+2*y)) == x**2 + 2*x*y + y**2
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let d = (&x * &y * &z + &x * &y).gcd_all(&(&x * &z + &x)).unwrap();
    assert_eq!(d, &x * &z + &x, "got {d}");
    let f = (&x + &y).powi(3) * (&x - &y);
    let g = (&x + &y).powi(2) * (&x + &y * 2);
    let d = f.gcd_all(&g).unwrap();
    assert_eq!(d, (&x + &y).powi(2).expand(), "got {d}");
}

#[test]
fn lcm_all_matches_sympy() {
    // SymPy: lcm(x**2 - y**2, x - y) == x**2 - y**2; lcm(2*x, 4*x) == 4*x;
    //        lcm(x*y, y**2) == x*y**2; lcm(x*y*z + x*y, x*z + x) == x*y*z + x*y
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let f = &x.powi(2) - &y.powi(2);
    assert_eq!(f.lcm_all(&(&x - &y)).unwrap(), f);
    assert_eq!((&x * 2).lcm_all(&(&x * 4)).unwrap(), &x * 4);
    assert_eq!((&x * &y).lcm_all(&y.powi(2)).unwrap(), &x * &y.powi(2));
    let l = (&x * &y * &z + &x * &y).lcm_all(&(&x * &z + &x)).unwrap();
    assert_eq!(l, &x * &y * &z + &x * &y, "got {l}");
    // gcd · lcm = f · g up to sign for coprime-content inputs
    let (a, b) = (&x.powi(2) + &x * &y, &x * &y + &y.powi(2));
    let prod = (a.gcd_all(&b).unwrap() * a.lcm_all(&b).unwrap()).expand();
    assert_eq!(prod, (&a * &b).expand(), "gcd·lcm = f·g");
}

#[test]
fn gcd_all_none_for_non_polynomial_input() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(x.sin().gcd_all(&x).is_none(), "sin(x) is not a polynomial");
    assert!(
        (ctx.int(1) / &x).gcd_all(&x).is_none(),
        "1/x is a rational function"
    );
    assert!(
        (&x * ctx.pi()).gcd_all(&x).is_none(),
        "π is not a rational coefficient"
    );
    assert!(x.sqrt().lcm_all(&x).is_none(), "√x is not a polynomial");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. groebner / reduce_modulo
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn groebner_lex_circle_and_line() {
    // SymPy: groebner([x**2 + y**2 - 1, x - y], x, y, order='lex')
    //        == [x - y, 2*y**2 - 1]      (symplex returns monic: y² − 1/2)
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let vars = [x.clone(), y.clone()];
    let basis = Ex::groebner(
        &[&x.powi(2) + &y.powi(2) - 1, &x - &y],
        &vars,
        MonomialOrder::Lex,
    )
    .unwrap();
    assert_eq!(
        basis,
        vec![&x - &y, &y.powi(2) - ctx.rational(1, 2)],
        "got {:?}",
        strings(&basis)
    );
}

#[test]
fn groebner_grevlex_circle_and_line() {
    // SymPy: groebner([x**2 + y**2 - 1, x - y], x, y, order='grevlex')
    //        == [2*y**2 - 1, x - y]      (monic here; leading monomials descending: y² > x)
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let vars = [x.clone(), y.clone()];
    let basis = Ex::groebner(
        &[&x.powi(2) + &y.powi(2) - 1, &x - &y],
        &vars,
        MonomialOrder::GrevLex,
    )
    .unwrap();
    assert_eq!(
        basis,
        vec![&y.powi(2) - ctx.rational(1, 2), &x - &y],
        "got {:?}",
        strings(&basis)
    );
}

#[test]
fn groebner_lex_three_variables_is_triangular() {
    // SymPy: groebner([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1], x, y, z, order='lex')
    //   == [x + y + z**2 - 1, y**2 - y - z**2 + z, 2*y*z**2 + z**4 - z**2, z**6 - 4*z**4 + 4*z**3 - z**2]
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let vars = [x.clone(), y.clone(), z.clone()];
    let system = [
        &x.powi(2) + &y + &z - 1,
        &x + &y.powi(2) + &z - 1,
        &x + &y + &z.powi(2) - 1,
    ];
    let basis = Ex::groebner(&system, &vars, MonomialOrder::Lex).unwrap();
    let expected = vec![
        &x + &y + &z.powi(2) - 1,
        &y.powi(2) - &y - &z.powi(2) + &z,
        &y * &z.powi(2) + &z.powi(4) * ctx.rational(1, 2) - &z.powi(2) * ctx.rational(1, 2),
        &z.powi(6) - &z.powi(4) * 4 + &z.powi(3) * 4 - &z.powi(2),
    ];
    assert_eq!(basis, expected, "got {:?}", strings(&basis));
    // Every generator reduces to zero modulo the basis.
    for f in &system {
        let r = f.reduce_modulo(&basis, &vars, MonomialOrder::Lex).unwrap();
        assert!(r.is_zero_structural(), "{f} reduced to {r}");
    }
}

#[test]
fn groebner_grevlex_three_variables_input_already_a_basis() {
    // SymPy: groebner([...], x, y, z, order='grevlex') == the (monic) inputs
    //        [x**2 + y + z - 1, y**2 + x + z - 1, z**2 + x + y - 1]
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let vars = [x.clone(), y.clone(), z.clone()];
    let f1 = &x.powi(2) + &y + &z - 1;
    let f2 = &x + &y.powi(2) + &z - 1;
    let f3 = &x + &y + &z.powi(2) - 1;
    let basis = Ex::groebner(
        &[f1.clone(), f2.clone(), f3.clone()],
        &vars,
        MonomialOrder::GrevLex,
    )
    .unwrap();
    assert_eq!(basis, vec![f1, f2, f3], "got {:?}", strings(&basis));
}

#[test]
fn reduce_modulo_gives_the_normal_form() {
    // SymPy: G = groebner([x**2 + y**2 - 1, x - y], x, y, order='lex')
    //        G.reduce(x**2) == ([x + y, 1/2], 1/2); G.reduce(x**3) == (…, y/2);
    //        G.reduce(x**2 - y**2) == (…, 0)
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let vars = [x.clone(), y.clone()];
    let gens = [&x.powi(2) + &y.powi(2) - 1, &x - &y];
    for order in [MonomialOrder::Lex, MonomialOrder::GrevLex] {
        let basis = Ex::groebner(&gens, &vars, order).unwrap();
        let r = x.powi(2).reduce_modulo(&basis, &vars, order).unwrap();
        assert_eq!(r, ctx.rational(1, 2), "{order:?}: x² → {r}");
        let r = x.powi(3).reduce_modulo(&basis, &vars, order).unwrap();
        assert_eq!(r, &y * ctx.rational(1, 2), "{order:?}: x³ → {r}");
        let r = (&x.powi(2) - &y.powi(2))
            .reduce_modulo(&basis, &vars, order)
            .unwrap();
        assert!(
            r.is_zero_structural(),
            "{order:?}: x² − y² is in the ideal, got {r}"
        );
    }
}

#[test]
fn groebner_trivial_and_unit_ideals() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let vars = [x.clone(), y.clone()];
    assert!(
        Ex::groebner(&[], &vars, MonomialOrder::Lex)
            .unwrap()
            .is_empty()
    );
    assert!(
        Ex::groebner(&[ctx.zero()], &vars, MonomialOrder::GrevLex)
            .unwrap()
            .is_empty()
    );
    // ⟨x, x − 1⟩ = ⟨1⟩
    let basis = Ex::groebner(&[x.clone(), &x - 1], &vars, MonomialOrder::Lex).unwrap();
    assert_eq!(basis, vec![ctx.one()]);
}

#[test]
fn groebner_rejects_bad_variables_and_non_polynomials() {
    let ctx = Context::new();
    let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
    let vars = [x.clone(), y.clone()];
    let just_x = std::slice::from_ref(&x);
    assert!(matches!(
        Ex::groebner(just_x, &[], MonomialOrder::Lex),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        Ex::groebner(just_x, &[x.clone(), x.clone()], MonomialOrder::Lex),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        Ex::groebner(just_x, &[x.powi(2)], MonomialOrder::Lex),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let e = Ex::groebner(&[x.sin()], &vars, MonomialOrder::Lex)
        .unwrap_err()
        .to_string();
    assert!(e.contains("sin(x)"), "{e}");
    // A symbolic coefficient is not a rational coefficient.
    let e = Ex::groebner(&[&a * &x], &vars, MonomialOrder::GrevLex)
        .unwrap_err()
        .to_string();
    assert!(e.contains("a*x"), "{e}");
    assert!(matches!(
        x.reduce_modulo(&[ctx.int(1) / &x], &vars, MonomialOrder::Lex),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. real_roots / root_of
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn real_roots_of_x_cubed_minus_2x_are_ascending() {
    // SymPy: real_roots(x**3 - 2*x) == [-sqrt(2), 0, sqrt(2)]
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &x * 2;
    let roots = f
        .real_roots(&x)
        .expect("polynomial with rational coefficients");
    assert_eq!(roots.len(), 3, "got {:?}", strings(&roots));
    let vals: Vec<f64> = roots.iter().map(|r| r.eval_f64().unwrap()).collect();
    let expected = [-std::f64::consts::SQRT_2, 0.0, std::f64::consts::SQRT_2];
    for (v, e) in vals.iter().zip(expected) {
        assert!((v - e).abs() < 1e-9, "roots {vals:?} vs {expected:?}");
    }
    assert_eq!(roots[1], ctx.int(0), "the rational root is exact");
    assert_eq!(s(&roots[0]), "RootOf(x^2 - 2, 0)");
    assert_eq!(s(&roots[2]), "RootOf(x^2 - 2, 1)");
    // Substituting each root back gives 0 (within f64).
    for r in &roots {
        assert!(f.subs(&x, r).eval_f64().unwrap().abs() < 1e-9);
    }
}

#[test]
fn root_of_indexes_the_ascending_real_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &x * 2;
    let largest = f.root_of(&x, 2).expect("three real roots");
    assert!((largest.eval_f64().unwrap() - std::f64::consts::SQRT_2).abs() < 1e-9);
    assert_eq!(largest, f.real_roots(&x).unwrap()[2]);
    assert_eq!(f.root_of(&x, 1).unwrap(), ctx.int(0));
    assert!(f.root_of(&x, 3).is_none(), "only three real roots");
}

#[test]
fn real_roots_interleave_factors_and_skip_complex_pairs() {
    // SymPy: real_roots((x**3 - x - 1)*(x**2 - 2)) ≈ [-1.41421356, 1.32471796, 1.41421356]
    //        (x³ − x − 1 has one real root and a complex pair; index 2 in its (re, im)-sorted roots)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x.powi(3) - &x - 1) * (&x.powi(2) - 2);
    let roots = f.real_roots(&x).unwrap();
    assert_eq!(
        strings(&roots),
        [
            "RootOf(x^2 - 2, 0)",
            "RootOf(x^3 - x - 1, 2)",
            "RootOf(x^2 - 2, 1)"
        ]
    );
    let vals: Vec<f64> = roots.iter().map(|r| r.eval_f64().unwrap()).collect();
    let expected = [
        -std::f64::consts::SQRT_2,
        1.324_717_957_244_746,
        std::f64::consts::SQRT_2,
    ];
    for (v, e) in vals.iter().zip(expected) {
        assert!((v - e).abs() < 1e-9, "roots {vals:?} vs {expected:?}");
    }
}

#[test]
fn real_roots_quintic_without_radicals() {
    // SymPy: real_roots(x**5 - x - 1) == [CRootOf(x**5 - x - 1, 0)] ≈ 1.1673039782614187
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(5) - &x - 1;
    let roots = f.real_roots(&x).unwrap();
    assert_eq!(roots.len(), 1, "got {:?}", strings(&roots));
    assert!((roots[0].eval_f64().unwrap() - 1.1673039782614187).abs() < 1e-9);
    assert_eq!(f.root_of(&x, 0).unwrap(), roots[0]);
}

#[test]
fn real_roots_lists_a_repeated_root_once_and_rational_roots_exactly() {
    // SymPy real_roots((x - 1)**2*(x - 2)) == [1, 1, 2]; symplex reports distinct roots.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x - 1).powi(2) * (&x - 2);
    assert_eq!(f.real_roots(&x).unwrap(), vec![ctx.int(1), ctx.int(2)]);
    // 6x² − 5x + 1 = (2x − 1)(3x − 1): rational roots 1/3 < 1/2
    let g = &x.powi(2) * 6 - &x * 5 + 1;
    assert_eq!(
        g.real_roots(&x).unwrap(),
        vec![ctx.rational(1, 3), ctx.rational(1, 2)]
    );
}

#[test]
fn real_roots_empty_for_no_real_roots_and_none_for_non_polynomials() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!((&x.powi(2) + 1).real_roots(&x), Some(vec![]));
    assert!((&x.powi(2) + 1).root_of(&x, 0).is_none());
    assert!(
        ctx.int(3).real_roots(&x).is_none(),
        "a constant has no roots to list"
    );
    assert!(x.sin().real_roots(&x).is_none());
    assert!(
        (&x + ctx.pi()).real_roots(&x).is_none(),
        "irrational coefficient"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. factor_mod
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_mod_x_squared_plus_one_mod_5() {
    // SymPy: factor_list(x**2 + 1, modulus=5) == (1, [(x + 2, 1), (x - 2, 1)])
    //        (x − 2 ≡ x + 3 mod 5; symplex uses representatives in [0, p))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (lc, factors) = (&x.powi(2) + 1).factor_mod(&x, 5).unwrap();
    assert_eq!(lc, ctx.int(1));
    let mut got: Vec<(String, u32)> = factors.iter().map(|(f, m)| (s(f), *m)).collect();
    got.sort();
    assert_eq!(got, [("x + 2".to_string(), 1), ("x + 3".to_string(), 1)]);
    // The product reproduces x² + 1 modulo 5.
    let prod = (&factors[0].0 * &factors[1].0).expand();
    assert_eq!(prod, &x.powi(2) + &x * 5 + 6);
}

#[test]
fn factor_mod_irreducible_and_split_cases() {
    // SymPy: factor_list(x**2 + 1, modulus=3) == (1, [(x**2 + 1, 1)])
    //        factor_list(x**4 - 1, modulus=5) == (1, [(x + 1, 1), (x + 2, 1), (x - 2, 1), (x - 1, 1)])
    //        factor_list(2*x**2 + 2, modulus=5) == (2, [(x + 2, 1), (x - 2, 1)])
    //        factor_list((x**2 + 1)**2, modulus=5) == (1, [(x + 2, 2), (x - 2, 2)])
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (lc, factors) = (&x.powi(2) + 1).factor_mod(&x, 3).unwrap();
    assert_eq!(lc, ctx.int(1));
    assert_eq!(factors, vec![(&x.powi(2) + 1, 1)]);

    let (lc, factors) = (&x.powi(4) - 1).factor_mod(&x, 5).unwrap();
    assert_eq!(lc, ctx.int(1));
    assert_eq!(
        factors,
        vec![(&x + 1, 1), (&x + 2, 1), (&x + 3, 1), (&x + 4, 1)]
    );

    let (lc, factors) = (&x.powi(2) * 2 + 2).factor_mod(&x, 5).unwrap();
    assert_eq!(lc, ctx.int(2));
    assert_eq!(factors, vec![(&x + 2, 1), (&x + 3, 1)]);

    let (_, factors) = (&x.powi(2) + 1).powi(2).factor_mod(&x, 5).unwrap();
    assert_eq!(factors, vec![(&x + 2, 2), (&x + 3, 2)]);
}

#[test]
fn factor_mod_reduces_rational_coefficients_through_inverses() {
    // x/2 + 1 ≡ 3x + 1 ≡ 3(x + 2) (mod 5), since 2⁻¹ = 3 and 3⁻¹ = 2: 2·1 = 2.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (lc, factors) = (&x / 2 + 1).factor_mod(&x, 5).unwrap();
    assert_eq!(lc, ctx.int(3));
    assert_eq!(factors, vec![(&x + 2, 1)]);
    // Vanishing mod p: 5x ≡ 0.
    let (lc, factors) = (&x * 5).factor_mod(&x, 5).unwrap();
    assert_eq!(lc, ctx.int(0));
    assert!(factors.is_empty());
}

#[test]
fn factor_mod_rejects_composite_unsupported_moduli_and_bad_input() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(2) + 1;
    let e = f.factor_mod(&x, 6).unwrap_err();
    assert!(matches!(e, SymplexError::InvalidArgument { .. }), "{e}");
    assert!(e.to_string().contains("6 is not prime"), "{e}");
    assert!(matches!(
        f.factor_mod(&x, 1),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        f.factor_mod(&x, 0),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // 2 is prime but the GF(p) arithmetic needs an odd prime.
    let e = f.factor_mod(&x, 2).unwrap_err().to_string();
    assert!(e.contains("odd prime"), "{e}");
    assert!(matches!(
        x.sin().factor_mod(&x, 5),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Denominator divisible by p has no inverse.
    let e = (&x / 5 + 1).factor_mod(&x, 5).unwrap_err().to_string();
    assert!(e.contains("denominator divisible by 5"), "{e}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. resultant_symbolic / discriminant_symbolic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn resultant_symbolic_linear_and_quadratic() {
    // SymPy: resultant(x - a, x - b, x) == a - b; resultant(x**2 + a, x + b, x) == a + b**2
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    let r = (&x - &a).resultant_symbolic(&(&x - &b), &x).unwrap();
    assert_eq!(r, &a - &b, "got {r}");
    let r = (&x.powi(2) + &a)
        .resultant_symbolic(&(&x + &b), &x)
        .unwrap();
    assert_eq!(r, &a + &b.powi(2), "got {r}");
}

#[test]
fn resultant_symbolic_two_quadratics_uses_4x4_sylvester() {
    // SymPy: expand(resultant(x**2 + a*x + b, x**2 + c*x + d, x))
    //        == a**2*d - a*b*c - a*c*d + b**2 + b*c**2 - 2*b*d + d**2
    let ctx = Context::new();
    let (x, a, b, c, d) = (
        ctx.symbol("x"),
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
        ctx.symbol("d"),
    );
    let f = &x.powi(2) + &a * &x + &b;
    let g = &x.powi(2) + &c * &x + &d;
    let r = f.resultant_symbolic(&g, &x).unwrap();
    let expected = &a.powi(2) * &d - &a * &b * &c - &a * &c * &d + &b.powi(2) + &b * &c.powi(2)
        - &b * &d * 2
        + &d.powi(2);
    assert_eq!(r, expected.expand(), "got {r}");
}

#[test]
fn resultant_symbolic_agrees_with_rational_resultant_and_handles_constants() {
    // SymPy: resultant(x**2 - 1, x - 3, x) == 8; resultant(a, x**2 + 1, x) == a**2; resultant(2, 3, x) == 1
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let f = &x.powi(2) - 1;
    let g = &x - 3;
    assert_eq!(f.resultant_symbolic(&g, &x).unwrap(), ctx.int(8));
    assert_eq!(f.resultant_symbolic(&g, &x), f.resultant(&g, &x));
    assert_eq!(
        a.resultant_symbolic(&(&x.powi(2) + 1), &x).unwrap(),
        a.powi(2)
    );
    assert_eq!(
        ctx.int(2).resultant_symbolic(&ctx.int(3), &x).unwrap(),
        ctx.int(1)
    );
    assert!(x.sin().resultant_symbolic(&x, &x).is_none());
    // Shared root ⇒ zero: res(x − a, x² − a²) = 0
    let r = (&x - &a)
        .resultant_symbolic(&(&x.powi(2) - &a.powi(2)), &x)
        .unwrap();
    assert!(r.is_zero_structural(), "got {r}");
}

#[test]
fn discriminant_symbolic_quadratic_cubic_quartic() {
    // SymPy: discriminant(a*x**2 + b*x + c, x) == -4*a*c + b**2
    //        discriminant(x**3 + p*x + q, x) == -4*p**3 - 27*q**2
    //        discriminant(x**4 + p*x + q, x) == -27*p**4 + 256*q**3
    let ctx = Context::new();
    let (x, a, b, c, p, q) = (
        ctx.symbol("x"),
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
        ctx.symbol("p"),
        ctx.symbol("q"),
    );
    let quad = &a * &x.powi(2) + &b * &x + &c;
    let d = quad.discriminant_symbolic(&x).unwrap();
    assert_eq!(d, &b.powi(2) - &a * &c * 4, "got {d}");
    let cubic = &x.powi(3) + &p * &x + &q;
    let d = cubic.discriminant_symbolic(&x).unwrap();
    assert_eq!(d, -(&p.powi(3) * 4) - &q.powi(2) * 27, "got {d}");
    let quartic = &x.powi(4) + &p * &x + &q;
    let d = quartic.discriminant_symbolic(&x).unwrap();
    assert_eq!(d, -(&p.powi(4) * 27) + &q.powi(3) * 256, "got {d}");
}

#[test]
fn discriminant_symbolic_edge_cases_agree_with_rational_discriminant() {
    // SymPy: discriminant(a*x + b, x) == 1; discriminant(x**2 + 3*x + 1, x) == 5
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    assert_eq!(
        (&a * &x + &b).discriminant_symbolic(&x).unwrap(),
        ctx.int(1)
    );
    let f = &x.powi(2) + &x * 3 + 1;
    assert_eq!(f.discriminant_symbolic(&x).unwrap(), ctx.int(5));
    assert_eq!(f.discriminant_symbolic(&x), f.discriminant(&x));
    assert!(a.discriminant_symbolic(&x).is_none(), "constant: undefined");
    assert!(x.exp().discriminant_symbolic(&x).is_none());
}
