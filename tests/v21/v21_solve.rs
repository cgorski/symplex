//! 0.22 — `Ex::solve` regressions from the 0.21 differential audit, and the
//! `𝔽₂` root finding of `poly::modpoly`.
//!
//! Root sets cite SymPy 1.14 (`symplex/.venv/bin/python`): `sympy.roots`
//! on the expanded polynomial, `Poly(..., modulus=2).ground_roots()` for
//! the finite field.  Polynomial roots are compared as exact values through
//! `Ex::equals`, and the output is checked to contain no `RootOf`
//! placeholder where SymPy has none.

use symplex::factor_zassenhaus::modpoly::{Fp64, PolyIn};
use symplex::prelude::*;

fn strings(es: &[Ex]) -> Vec<String> {
    es.iter().map(|e| format!("{e}")).collect()
}

/// `roots` is exactly the set `expected` (order-free, no duplicates, no
/// `RootOf`), each membership decided by `Ex::equals`.
fn assert_root_set(roots: &[Ex], expected: &[Ex], label: &str) {
    let got = strings(roots);
    assert!(
        got.iter().all(|s| !s.contains("RootOf")),
        "{label}: a RootOf placeholder survived: {got:?}"
    );
    assert_eq!(
        roots.len(),
        expected.len(),
        "{label}: expected {} distinct roots, got {got:?}",
        expected.len()
    );
    for e in expected {
        assert!(
            roots.iter().any(|r| r.equals(e) == Some(true)),
            "{label}: root {e} missing from {got:?}"
        );
    }
    for (i, r) in roots.iter().enumerate() {
        for other in &roots[i + 1..] {
            assert_ne!(
                r.equals(other),
                Some(true),
                "{label}: duplicate root {r} in {got:?}"
            );
        }
    }
}

/// `p = (3x − 7)(3x + 7)(2x − 5)²(x − 9)³`, expanded:
/// `36x⁷ − 1152x⁶ + 13637x⁵ − 69787x⁴ + 110582x³ + 250074x² − 1012095x + 893025`.
fn poly_p(ctx: &Context, x: &Ex) -> Ex {
    let f = (&(x * 3) - 7) * (&(x * 3) + 7) * (&(x * 2) - 5).powi(2) * (x - 9).powi(3);
    let _ = ctx;
    f.expand()
}

/// SymPy: `roots(p) == {7/3: 1, -7/3: 1, 5/2: 2, 9: 3}`.
fn roots_p(ctx: &Context) -> Vec<Ex> {
    vec![
        ctx.rational(7, 3),
        ctx.rational(-7, 3),
        ctx.rational(5, 2),
        ctx.int(9),
    ]
}

/// `q = (2x + 3)(4x − 9)²(x + 2)³(x² − 2)²`, expanded:
/// `32x¹⁰ + 96x⁹ − 374x⁸ − 1361x⁷ + 1154x⁶ + 6776x⁵ + 792x⁴ − 13844x³ − 7608x² + 9936x + 7776`.
fn poly_q(x: &Ex) -> Ex {
    ((&(x * 2) + 3) * (&(x * 4) - 9).powi(2) * (x + 2).powi(3) * (&x.powi(2) - 2).powi(2)).expand()
}

/// SymPy: `roots(q) == {-3/2: 1, 9/4: 2, -2: 3, -sqrt(2): 2, sqrt(2): 2}`.
fn roots_q(ctx: &Context) -> Vec<Ex> {
    let sqrt2 = ctx.int(2).sqrt();
    vec![
        ctx.rational(-3, 2),
        ctx.rational(9, 4),
        ctx.int(-2),
        sqrt2.clone(),
        -&sqrt2,
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// B — `c·p(x)` used to degrade to RootOf
// ═══════════════════════════════════════════════════════════════════════════

/// `p.solve(x)` already worked; `(4·p).expand().solve(x)` returned seven
/// `RootOf(144x⁷ − …, k)` because the divisor enumeration gave up above
/// `10⁶` (`4·893025 = 3572100`).  SymPy: `roots(expand(4*p)) == roots(p)`.
#[test]
fn solve_four_times_p_gives_the_rational_roots() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = poly_p(&ctx, &x);
    assert_root_set(&p.solve(&x)?, &roots_p(&ctx), "p");
    let four_p = (&p * 4).expand();
    assert_root_set(&four_p.solve(&x)?, &roots_p(&ctx), "4·p");
    Ok(())
}

/// `−384·q`: SymPy `roots(expand(-384*q)) == {-3/2: 1, 9/4: 2, -2: 3, -sqrt(2): 2, sqrt(2): 2}`;
/// the solver returned ten `RootOf`s.
#[test]
fn solve_minus_384_times_q_gives_the_five_distinct_roots() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let q = poly_q(&x);
    assert_root_set(&q.solve(&x)?, &roots_q(&ctx), "q");
    let scaled = (&q * -384).expand();
    assert_root_set(&scaled.solve(&x)?, &roots_q(&ctx), "−384·q");
    Ok(())
}

/// `r = −10368(x + 4)²(x − 3)³(x − 1)³(4x + 1)³(x² + 1)`: SymPy
/// `roots(expand(r)) == {-4: 2, 3: 3, 1: 3, -1/4: 3, -I: 1, I: 1}`.  The
/// solver returned `1`, `−1/4` and eleven `RootOf`s over a degree-11
/// cofactor that still contained `(x − 1)²(4x + 1)²`.
#[test]
fn solve_degree_13_with_repeated_factors_and_a_complex_pair() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = ((&x + 4).powi(2)
        * (&x - 3).powi(3)
        * (&x - 1).powi(3)
        * (&(&x * 4) + 1).powi(3)
        * (&x.powi(2) + 1)
        * -10368)
        .expand();
    let i = ctx.i_unit();
    let expected = vec![
        ctx.int(-4),
        ctx.int(3),
        ctx.int(1),
        ctx.rational(-1, 4),
        i.clone(),
        -&i,
    ];
    assert_root_set(
        &r.solve(&x)?,
        &expected,
        "−10368·(x+4)²(x−3)³(x−1)³(4x+1)³(x²+1)",
    );
    Ok(())
}

/// The root set is invariant under a constant factor: `c·p` for
/// `c ∈ {1, 4, −384, 10⁹}` all give SymPy's `{7/3, −7/3, 5/2, 9}`.
#[test]
fn solve_root_set_is_invariant_under_a_constant_factor() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = poly_p(&ctx, &x);
    for c in [1i64, 4, -384, 1_000_000_000] {
        let scaled = (&p * c).expand();
        assert_root_set(&scaled.solve(&x)?, &roots_p(&ctx), &format!("{c}·p"));
    }
    // and the same for q, whose roots include ±√2
    let q = poly_q(&x);
    for c in [1i64, 4, -384, 1_000_000_000] {
        let scaled = (&q * c).expand();
        assert_root_set(&scaled.solve(&x)?, &roots_q(&ctx), &format!("{c}·q"));
    }
    Ok(())
}

/// `x⁵ − x − 1` is irreducible (SymPy `factor_list` = `(1, [(x**5 - x - 1, 1)])`):
/// five `RootOf(x^5 - x - 1, k)` placeholders, `k = 0..4`, and nothing else.
#[test]
fn solve_irreducible_quintic_still_yields_five_rootofs() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(5) - &x - 1;
    let roots = f.solve(&x)?;
    let got = strings(&roots);
    assert_eq!(roots.len(), 5, "{got:?}");
    for k in 0..5 {
        let expected = format!("RootOf(x^5 - x - 1, {k})");
        assert!(got.contains(&expected), "{expected} missing from {got:?}");
    }
    Ok(())
}

/// `(x⁵ − x − 1)(x − 2)`: SymPy `factor_list` = `(1, [(x - 2, 1), (x**5 - x - 1, 1)])`,
/// `roots` = `{2: 1}` plus the quintic's `CRootOf`s.  The placeholders are
/// over the quintic factor only, never over the sextic.
#[test]
fn solve_quintic_times_linear_places_rootofs_over_the_quintic_only() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ((&x.powi(5) - &x - 1) * (&x - 2)).expand();
    let roots = f.solve(&x)?;
    let got = strings(&roots);
    assert_eq!(roots.len(), 6, "{got:?}");
    assert!(
        roots.iter().any(|r| r.equals(&ctx.int(2)) == Some(true)),
        "2 missing from {got:?}"
    );
    for k in 0..5 {
        let expected = format!("RootOf(x^5 - x - 1, {k})");
        assert!(got.contains(&expected), "{expected} missing from {got:?}");
    }
    // Scaling and a repeated linear factor change nothing: 6·(x⁵ − x − 1)(x − 2)²
    let g = (&f * (&x - 2) * 6).expand();
    let roots = g.solve(&x)?;
    let got = strings(&roots);
    assert_eq!(roots.len(), 6, "{got:?}");
    assert!(
        got.iter().filter(|s| s.contains("RootOf")).count() == 5,
        "{got:?}"
    );
    assert!(
        got.iter()
            .all(|s| !s.contains("RootOf") || s.contains("x^5 - x - 1")),
        "a RootOf over something other than the quintic: {got:?}"
    );
    Ok(())
}

/// `solve` returns distinct roots: `(x − 1)²` gives `[1]`, and so does
/// `7·(x − 1)²·(x² + 2x + 1)` (= `7(x − 1)²(x + 1)²`, roots `{1, −1}`).
#[test]
fn solve_returns_distinct_roots() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let roots = (&x - 1).powi(2).expand().solve(&x)?;
    assert_root_set(&roots, &[ctx.int(1)], "(x − 1)²");
    let f = ((&x - 1).powi(2) * (&x.powi(2) + &x * 2 + 1) * 7).expand();
    assert_root_set(
        &f.solve(&x)?,
        &[ctx.int(1), ctx.int(-1)],
        "7(x − 1)²(x + 1)²",
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// C — F_p roots for p = 2
// ═══════════════════════════════════════════════════════════════════════════

/// `x² + x` over `𝔽₂`: SymPy `Poly(x**2 + x, x, modulus=2).ground_roots() == {0: 1, -1: 1}`,
/// i.e. the roots `{0, 1}`.  `PolyIn::roots` returned `[]` (the
/// Cantor–Zassenhaus exponent `(p − 1)/2 = 0` never split anything).
#[test]
fn fp2_roots_of_x_squared_plus_x() {
    let f = PolyIn::from_coeffs(Fp64::new(2), vec![0, 1, 1]);
    assert_eq!(f.roots(), vec![0, 1]);
}

/// `x³ + x + 1` over `𝔽₂` is irreducible: SymPy `ground_roots() == {}`.
#[test]
fn fp2_roots_of_irreducible_cubic_are_none() {
    let f = PolyIn::from_coeffs(Fp64::new(2), vec![1, 1, 0, 1]);
    assert!(f.roots().is_empty());
}

/// `x² + 1 = (x + 1)²` over `𝔽₂`: SymPy `ground_roots() == {-1: 2}`; the
/// distinct root set is `{1}`.
#[test]
fn fp2_roots_of_x_squared_plus_one_is_the_double_root_once() {
    let f = PolyIn::from_coeffs(Fp64::new(2), vec![1, 0, 1]);
    assert_eq!(f.roots(), vec![1]);
}

/// More `𝔽₂` cases: `x⁴ + x` (SymPy `{0: 1, -1: 1}`), `x` (`{0: 1}`), `x + 1`
/// (`{-1: 1}`), a constant (no roots), and `x² + x` over `𝔽₃` (`{0: 1, -1: 1}`
/// = `{0, 2}`), which takes the odd-prime route.
#[test]
fn fp2_roots_more_cases_and_odd_prime_unchanged() {
    let two = Fp64::new(2);
    assert_eq!(
        PolyIn::from_coeffs(two, vec![0, 1, 0, 0, 1]).roots(),
        vec![0, 1]
    );
    assert_eq!(PolyIn::from_coeffs(two, vec![0, 1]).roots(), vec![0]);
    assert_eq!(PolyIn::from_coeffs(two, vec![1, 1]).roots(), vec![1]);
    assert!(PolyIn::from_coeffs(two, vec![1]).roots().is_empty());
    assert!(PolyIn::zero(two).roots().is_empty());
    assert_eq!(
        PolyIn::from_coeffs(Fp64::new(3), vec![0, 1, 1]).roots(),
        vec![0, 2]
    );
}
