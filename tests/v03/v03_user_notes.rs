//! symplex 0.3.1 — fixes driven by field notes from downstream tools:
//! integer access to rational literals and re-exported `num` crates,
//! SymPy-semantics `as_numer_denom`, deep `together`, root/sign primitives on
//! `Poly` under discoverable names, `Poly::shift`, and `to_lean`.

use symplex::lean::LeanOpts;
use symplex::num_bigint::BigInt;
use symplex::num_rational::Ratio;
use symplex::num_traits::{Signed, Zero};
use symplex::poly_ex::Poly;
use symplex::prelude::*;

fn s(e: &Ex) -> String {
    e.to_string()
}

fn pair(p: (Ex, Ex)) -> (String, String) {
    (p.0.to_string(), p.1.to_string())
}

// ── Note 1: naming and extracting rational literals ─────────────────────

#[test]
fn num_crates_are_reexported_and_usable_downstream() {
    let ctx = Context::new();
    let r: Ratio<BigInt> = ctx.rational(3, 31).as_rational().unwrap();
    assert_eq!(r, Ratio::new(BigInt::from(3), BigInt::from(31)));
    assert!(!r.is_zero());
    // Round trip through the re-exported type.
    assert_eq!(ctx.from_ratio(r), ctx.rational(3, 31));
}

#[test]
fn as_ratio_parts_and_i128_expose_p_and_q() {
    let ctx = Context::new();
    assert_eq!(ctx.rational(3, 31).as_ratio_i128(), Some((3, 31)));
    assert_eq!(ctx.rational(-6, 4).as_ratio_i128(), Some((-3, 2)));
    assert_eq!(ctx.int(0).as_ratio_i128(), Some((0, 1)));
    assert_eq!(
        ctx.rational(-6, 4).as_ratio_parts(),
        Some((BigInt::from(-3), BigInt::from(2)))
    );
    // Not a literal until evaluated.
    let e = ctx.rational(1, 3) + ctx.rational(1, 6);
    assert_eq!(e.as_ratio_i128(), Some((1, 2)));
    let x = ctx.symbol("x");
    assert_eq!((&x + 1).as_ratio_i128(), None);
    assert_eq!(ctx.pi().as_ratio_parts(), None);
    // Overflow of i128 is None, not a wrong value.
    let big = ctx.from_bigint(BigInt::from(2).pow(130));
    assert_eq!(big.as_ratio_i128(), None);
    assert!(big.as_ratio_parts().is_some());
}

// ── Note 2: as_numer_denom on numbers and coefficients ──────────────────

#[test]
fn as_numer_denom_splits_rational_literals() {
    let ctx = Context::new();
    assert_eq!(
        pair(ctx.rational(3, 31).as_numer_denom()),
        ("3".into(), "31".into())
    );
    assert_eq!(
        pair(ctx.rational(-3, 31).as_numer_denom()),
        ("-3".into(), "31".into())
    );
    assert_eq!(pair(ctx.int(7).as_numer_denom()), ("7".into(), "1".into()));
    let x = ctx.symbol("x");
    assert_eq!(
        pair((&x * 2 / 3).as_numer_denom()),
        ("2*x".into(), "3".into())
    );
    assert_eq!(pair((-&x / 2).as_numer_denom()), ("-x".into(), "2".into()));
}

#[test]
fn as_numer_denom_combines_sums_like_sympy() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    assert_eq!(
        pair((&x / 2 + ctx.rational(1, 3)).as_numer_denom()),
        ("3*x + 2".into(), "6".into())
    );
    assert_eq!(
        pair((1 / &x + 1 / &y).as_numer_denom()),
        ("x + y".into(), "x*y".into())
    );
    // No cancellation: structural, like SymPy.
    let (n, d) = ((&x.powi(2) - 1) / (&x - 1)).as_numer_denom();
    assert_eq!(s(&n), "x^2 - 1");
    assert_eq!(s(&d), "x - 1");
    // Denominator-free input is (self, 1).
    let p = &x.powi(2) + &y;
    let (n, d) = p.as_numer_denom();
    assert_eq!(n, p);
    assert!(d.is_one_structural());
}

// ── Note 3: together is deep ────────────────────────────────────────────

#[test]
fn together_flattens_nested_fractions() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    // A numerator that itself contains a fraction, over a polynomial.
    let inner = -96 * &j.powi(3) / (8 * &j + 7) + 12 * &j.powi(2) + 1;
    let e = &inner / (16 * &j.powi(2) + 22 * &j + 7);
    let t = e.together();
    let (n, d) = t.as_numer_denom();
    // Both parts are genuine polynomials in j now.
    let pn = Poly::new(&n, &[&j]).expect("numerator is polynomial");
    let pd = Poly::new(&d, &[&j]).expect("denominator is polynomial");
    assert!(pn.has_rational_coeffs() && pd.has_rational_coeffs());
    // −96j³ + (12j² + 1)(8j + 7): the cubic terms cancel, leaving degree 2
    // over the degree-3 product of the two denominators.
    assert_eq!(pn.total_degree(), Some(2));
    assert_eq!(pd.total_degree(), Some(3));
    // Value preserved at a sample point.
    let at = |f: &Ex| f.subs(&j, &ctx.rational(5, 3)).eval();
    assert_eq!(at(&t).as_rational(), at(&e).as_rational());
    assert_eq!(
        at(&(&n / &d)).as_rational(),
        at(&e).as_rational(),
        "n/d must equal the input"
    );
}

#[test]
fn together_handles_powers_of_fractions_and_products() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (&x + 1 / &y).powi(2) * (1 / &x + &y);
    let (n, d) = e.together().as_numer_denom();
    // Everything polynomial, denominator y^2 * x.
    assert!(Poly::new(&n, &[&x, &y]).is_some());
    let pd = Poly::new(&d, &[&x, &y]).unwrap();
    assert_eq!(pd.degree_in(&y), Some(2));
    assert_eq!(pd.degree_in(&x), Some(1));
    let at = |f: &Ex| {
        f.subs_map(&[(&x, &ctx.rational(2, 7)), (&y, &ctx.rational(-3, 5))])
            .eval()
            .as_rational()
    };
    assert_eq!(at(&(&n / &d)), at(&e));
}

#[test]
fn together_leaves_function_arguments_alone() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (1 / &x + 1).sin() / &x;
    let (n, d) = e.together().as_numer_denom();
    assert_eq!(s(&n), "sin(1/x + 1)");
    assert_eq!(s(&d), "x");
}

#[test]
fn together_and_ratsimp_agree_in_value_but_only_ratsimp_cancels() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x.powi(2) - 1) / (&x - 1) + 1 / (&x + 1);
    let t = e.together();
    let r = e.ratsimp();
    let at = |f: &Ex| f.subs(&x, &ctx.rational(7, 2)).eval().as_rational();
    assert_eq!(at(&t), at(&e));
    assert_eq!(at(&r), at(&e));
    let (_, dt) = t.as_numer_denom();
    let (_, dr) = r.as_numer_denom();
    assert_eq!(Poly::new(&dt, &[&x]).unwrap().total_degree(), Some(2));
    assert_eq!(Poly::new(&dr, &[&x]).unwrap().total_degree(), Some(1));
}

// ── Note 4: root counting / sign on Poly ────────────────────────────────

#[test]
fn poly_root_counting_and_sign_primitives() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    let p = Poly::new(&((&j - 1) * (&j - 2) * (&j - 3)), &[&j]).unwrap();
    assert_eq!(p.count_real_roots(), Some(3));
    assert_eq!(p.count_real_roots_in(&ctx.int(2), &ctx.infinity()), Some(2));
    assert_eq!(
        p.count_real_roots_in(&ctx.rational(5, 2), &ctx.infinity()),
        Some(1)
    );
    assert_eq!(p.count_real_roots_in(&ctx.int(4), &ctx.infinity()), Some(0));
    assert_eq!(p.real_roots_isolate().len(), 3);
    assert_eq!(
        p.is_nonnegative_on(&ctx.int(3), &ctx.infinity()),
        Some(true)
    );
    assert_eq!(
        p.is_nonnegative_on(&ctx.int(2), &ctx.infinity()),
        Some(false)
    );
    assert_eq!(p.is_positive_on(&ctx.int(3), &ctx.infinity()), Some(false));
    assert_eq!(
        p.is_positive_on(&ctx.rational(7, 2), &ctx.infinity()),
        Some(true)
    );
    // Ex-level name matches.
    let e = p.to_ex();
    assert_eq!(
        e.count_real_roots_in(&j, &ctx.int(2), &ctx.infinity()),
        Some(2)
    );
    // Multivariate / symbolic coefficients → None, never a guess.
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let q = Poly::new(&(&a * &x + 1), &[&x]).unwrap();
    assert_eq!(q.count_real_roots(), None);
    assert_eq!(q.is_nonnegative_on(&ctx.int(0), &ctx.infinity()), None);
    let m = Poly::new(&(&x + &j), &[&x, &j]).unwrap();
    assert_eq!(m.count_real_roots_in(&ctx.int(0), &ctx.int(1)), None);
    assert!(m.real_roots_isolate().is_empty());
}

#[test]
fn poly_shift_gives_the_descartes_style_certificate() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    // p = (j - 1)(j - 3) is ≥ 0 on [3, ∞): after j ↦ k + 3 every coefficient is ≥ 0.
    let p = Poly::new(&((&j - 1) * (&j - 3)), &[&j]).unwrap();
    let shifted = p.shift(&j, &ctx.int(3)).unwrap();
    assert_eq!(shifted.to_string(), "Poly(j^2 + 2*j, j)");
    assert!(
        shifted
            .coeffs()
            .iter()
            .all(|c| c.as_rational().is_some_and(|r| !r.is_negative()))
    );
    // Shifting by 2 exposes the sign change (the root at 3 is inside).
    let shifted2 = p.shift(&j, &ctx.int(2)).unwrap();
    assert!(
        shifted2
            .coeffs()
            .iter()
            .any(|c| c.as_rational().is_some_and(|r| r.is_negative()))
    );
    // Shift composes with evaluation: p(k + 3) at k = 1 is p(4) = 3.
    assert_eq!(shifted.eval(&[&ctx.int(1)]).unwrap(), ctx.int(3));
    // Multivariate shift of one generator, symbolic shift amount.
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let m = Poly::new(&(&x * &j + &j.powi(2)), &[&x, &j]).unwrap();
    let ms = m.shift(&j, &a).unwrap();
    assert_eq!(ms.gens().len(), 2);
    assert_eq!(ms.degree_in(&j), Some(2));
    // Errors: not a generator / shift mentions a generator.
    assert!(m.shift(&a, &ctx.int(1)).is_err());
    assert!(m.shift(&j, &x).is_err());
}

#[test]
fn count_real_roots_in_replaces_the_removed_alias() {
    // 0.4 removed `roots_count_real` (the 0.3 alias); the name is
    // `count_real_roots_in` on `Ex` and on `Poly`.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &x;
    assert_eq!(f.count_real_roots_in(&x, &ctx.int(0), &ctx.int(5)), Some(2));
    let p = Poly::new(&f, &[&x]).unwrap();
    assert_eq!(p.count_real_roots_in(&ctx.int(0), &ctx.int(5)), Some(2));
}

// ── Note 5: Lean 4 output ────────────────────────────────────────────────

#[test]
fn to_lean_matches_mathlib_spacing_and_typing() {
    let ctx = Context::new();
    let (j, k) = (ctx.symbol("j"), ctx.symbol("k"));
    let p = (2 * &j + 1) * (&j.powi(2) - 3 * &j + 1);
    assert_eq!(
        p.expand().to_lean().unwrap(),
        "2 * j ^ 3 - 5 * j ^ 2 - j + 1"
    );
    assert_eq!(
        ((&j - 1) / (2 * &j)).to_lean().unwrap(),
        "(j - 1) / (2 * j)"
    );
    assert_eq!((&j / 2 - &k / 3).to_lean().unwrap(), "j / 2 - k / 3");
    assert_eq!(ctx.rational(3, 31).to_lean().unwrap(), "(3 / 31 : ℝ)");
    assert_eq!((&j * ctx.rational(3, 31)).to_lean().unwrap(), "3 * j / 31");
    assert_eq!(
        (&j.powi(2) + 1).sqrt().to_lean().unwrap(),
        "Real.sqrt (j ^ 2 + 1)"
    );
    assert_eq!((-&j).exp().to_lean().unwrap(), "Real.exp (-j)");
    assert_eq!(j.powi(-2).to_lean().unwrap(), "(j ^ 2)⁻¹");
    // Propositions.
    assert_eq!(j.ge(&ctx.int(2)).to_lean().unwrap(), "2 ≤ j");
    assert_eq!(
        (&j.powi(2) - 1).gt(&ctx.int(0)).to_lean().unwrap(),
        "0 < j ^ 2 - 1"
    );
    // Options.
    // 0.4: `LeanOpts` gained a field, so literals need `..Default::default()`.
    let opts = LeanOpts {
        real_type: "ℚ".into(),
        ..Default::default()
    };
    assert_eq!(
        ctx.rational(1, 2).to_lean_with(&opts).unwrap(),
        "(1 / 2 : ℚ)"
    );
    // Identifiers that are not valid Lean names are quoted.
    let odd = ctx.symbol("x-1");
    assert_eq!((&odd + 1).to_lean().unwrap(), "«x-1» + 1");
    // Unsupported nodes are errors.
    assert!(matches!(
        j.bessel_j(&ctx.int(0)).to_lean(),
        Err(SymplexError::NotImplemented(_))
    ));
}

#[test]
fn to_lean_of_a_certificate_is_usable_verbatim() {
    // The shifted-coefficient certificate rendered as a Lean statement.
    let ctx = Context::new();
    let j = ctx.symbol("j");
    let p = (&j - 1) * (&j - 3);
    let poly = Poly::new(&p, &[&j]).unwrap();
    let shifted = poly.shift(&j, &ctx.int(3)).unwrap().to_ex();
    let stmt = format!(
        "theorem nonneg (j : ℝ) (hj : 3 ≤ j) : 0 ≤ {} := by nlinarith [hj]",
        p.expand().to_lean().unwrap()
    );
    assert_eq!(
        stmt,
        "theorem nonneg (j : ℝ) (hj : 3 ≤ j) : 0 ≤ j ^ 2 - 4 * j + 3 := by nlinarith [hj]"
    );
    assert_eq!(shifted.to_lean().unwrap(), "j ^ 2 + 2 * j");
}

// ── Powers of products beyond the canonicalisation threshold ─────────────

#[test]
fn numeric_coefficient_is_pulled_out_of_large_powers() {
    // Canonicalisation distributes `(a·b)^n` only for |n| ≤ 10; a numeric
    // coefficient must be pulled out regardless, or `(-x)^11` is stuck as an
    // opaque power and `expand`/`Poly::new` miss the leading term.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    assert_eq!(s(&(-&x).powi(11)), "-x^11");
    assert_eq!(s(&(2 * &x).powi(13)), "8192*x^13");
    assert_eq!(s(&(-&x).powi(12)), "x^12");
    let e = (2 - &x).powi(11).expand();
    assert!(s(&e).starts_with("-x^11 + 22*x^10"), "{e}");
    assert_eq!(e.degree(&x), Some(11));
    // Products of symbols are left alone (swell guard) but are still
    // recognised as monomials by the polynomial view.
    assert_eq!(s(&(&x * &y).powi(12)), "(x*y)^12");
    let p = Poly::new(&((&x * &y).powi(12) + 1), &[&x, &y]).unwrap();
    assert_eq!(p.to_string(), "Poly(x^12*y^12 + 1, x, y)");
    let a = ctx.symbol("a");
    let q = Poly::new(&(&a * &x * &y).powi(11), &[&x, &y]).unwrap();
    assert_eq!(q.to_string(), "Poly(a^11*x^11*y^11, x, y)");
    // Value preserved.
    let at = |f: &Ex| {
        f.subs_map(&[(&x, &ctx.rational(3, 2)), (&y, &ctx.int(-2))])
            .eval()
    };
    let raw = (2 - &x).powi(11);
    assert_eq!(at(&raw).as_rational(), at(&e).as_rational());
}
