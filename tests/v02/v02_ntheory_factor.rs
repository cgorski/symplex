//! Integration tests for 0.2 polynomial factorization over ℤ
//! (Berlekamp–Zassenhaus + Kronecker multivariate) through the public API.

use num_bigint::BigInt;
use num_rational::Ratio;
use proptest::prelude::*;
use symplex::factor_zassenhaus::{
    Poly, factor_mod_p, factor_multivariate, factor_squarefree_z, factor_zassenhaus,
    factor_zassenhaus_with_content, is_irreducible_z,
};
use symplex::multipoly::MultiPoly;
use symplex::prelude::*;

fn poly(coeffs: &[i64]) -> Poly {
    Poly::from_coeffs(
        coeffs
            .iter()
            .map(|&c| Ratio::from_integer(BigInt::from(c)))
            .collect(),
    )
}

/// Multiply the factorization back and compare with the input.
fn assert_reconstructs(f: &Poly) -> Vec<(Poly, u32)> {
    let (content, factors) = factor_zassenhaus_with_content(f);
    let mut back = Poly::constant(content);
    for (g, m) in &factors {
        for _ in 0..*m {
            back = &back * g;
        }
        assert!(
            g.leading_coeff()
                .is_some_and(|c| *c.numer() > BigInt::from(0)),
            "factor must have positive leading coefficient: {g}"
        );
        assert_eq!(
            is_irreducible_z(g),
            Some(true),
            "reported factor must be irreducible: {g}"
        );
    }
    assert_eq!(back, *f, "product of factors ≠ input");
    factors
}

fn x_pow_n_minus_1(n: usize) -> Poly {
    let mut c = vec![0i64; n + 1];
    c[0] = -1;
    c[n] = 1;
    poly(&c)
}

fn tau(n: usize) -> usize {
    (1..=n).filter(|d| n.is_multiple_of(*d)).count()
}

// ═══════════════════════════════════════════════════════════════════════════
// Ex-level factoring
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ex_factor_x12_minus_1_fully() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(12) - 1;
    let (content, factors) = f.factor_list(&x);
    assert_eq!(format!("{content}"), "1");
    assert_eq!(factors.len(), 6);
    let s = format!("{}", f.factor(&x));
    for cyclo in [
        "x - 1",
        "x + 1",
        "x^2 + x + 1",
        "x^2 + 1",
        "x^2 - x + 1",
        "x^4 - x^2 + 1",
    ] {
        assert!(s.contains(cyclo), "missing {cyclo} in {s}");
    }
}

#[test]
fn ex_factor_x8_plus_x4_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(8) + &x.powi(4) + 1;
    let (_, factors) = f.factor_list(&x);
    let mut degs: Vec<usize> = factors.iter().map(|(g, _)| g.degree(&x).unwrap()).collect();
    degs.sort_unstable();
    assert_eq!(degs, vec![2, 2, 4]);
}

#[test]
fn ex_factor_sophie_germain_and_x4_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = format!("{}", (&x.powi(4) + 4).factor(&x));
    assert!(
        s.contains("x^2 - 2*x + 2") && s.contains("x^2 + 2*x + 2"),
        "{s}"
    );
    // x^4 + 1 is irreducible: unchanged.
    let f = &x.powi(4) + 1;
    assert_eq!(format!("{}", f.factor(&x)), "x^4 + 1");
    assert_eq!(f.is_irreducible(&x), Some(true));
}

#[test]
fn ex_factor_non_monic_quartic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 6x⁴ − 7x³ − 8x² + 7x + 2 = (x − 1)(x + 1)(6x² − 7x − 2)
    let f = &x.powi(4) * 6 - &x.powi(3) * 7 - &x.powi(2) * 8 + &x * 7 + 2;
    let (content, factors) = f.factor_list(&x);
    assert_eq!(format!("{content}"), "1");
    assert_eq!(factors.len(), 3);
    let s = format!("{}", f.factor(&x));
    assert!(s.contains("6*x^2 - 7*x - 2"), "{s}");
    // Value preserved at several points.
    let g = f.factor(&x);
    for v in [-3i64, -1, 0, 2, 5] {
        assert_eq!(
            f.subs_i64(&x, v).eval_f64().unwrap(),
            g.subs_i64(&x, v).eval_f64().unwrap()
        );
    }
}

#[test]
fn ex_factor_swinnerton_dyer_irreducible() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∏(x ± √2 ± √3 ± √5)
    let f = &x.powi(8) - &x.powi(6) * 40 + &x.powi(4) * 352 - &x.powi(2) * 960 + 576;
    assert_eq!(f.is_irreducible(&x), Some(true));
    let (_, factors) = f.factor_list(&x);
    assert_eq!(factors.len(), 1);
    assert_eq!(factors[0].1, 1);
}

#[test]
fn ex_factor_repeated_and_content() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 12 (x − 1)³ (x² + 1)
    let f = ((&x - 1).powi(3) * (&x.powi(2) + 1) * 12).expand();
    let (content, factors) = f.factor_list(&x);
    assert_eq!(format!("{content}"), "12");
    let mut described: Vec<(String, u32)> =
        factors.iter().map(|(g, m)| (format!("{g}"), *m)).collect();
    described.sort();
    assert_eq!(
        described,
        vec![("x - 1".to_string(), 3), ("x^2 + 1".to_string(), 1)]
    );
}

#[test]
fn ex_factor_multivariate_examples() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));

    let s = format!("{}", (&x.powi(2) - &y.powi(2)).factor(&x));
    assert!(s.contains("x - y") && s.contains("x + y"), "{s}");

    let s = format!("{}", (&x.powi(3) - &y.powi(3)).factor_all());
    assert!(s.contains("x - y"), "{s}");
    assert!(
        s.contains("x^2 + x*y + y^2") || s.contains("x^2 + y^2 + x*y"),
        "{s}"
    );

    let s = format!("{}", (&x.powi(2) + &x * &y * 2 + &y.powi(2)).factor_all());
    assert!(s.contains("(x + y)^2"), "{s}");

    let s = format!("{}", (&x.powi(2) * &y - &y).factor_all());
    assert!(
        s.contains("y") && s.contains("x - 1") && s.contains("x + 1"),
        "{s}"
    );

    // (y + 1) x² − (y + 1)
    let f = (&y + 1) * &x.powi(2) - (&y + 1);
    let f = f.expand();
    let (content, factors) = f.factor_list_all();
    assert_eq!(format!("{content}"), "1");
    let mut names: Vec<String> = factors.iter().map(|(g, _)| format!("{g}")).collect();
    names.sort();
    let mut expected = vec![
        "x - 1".to_string(),
        "x + 1".to_string(),
        "y + 1".to_string(),
    ];
    expected.sort();
    assert_eq!(names, expected);
}

#[test]
fn ex_factor_three_variables() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let f = ((&x + &y + &z) * (&x - &y) * (&y + &z)).expand();
    let (_, factors) = f.factor_list_all();
    assert_eq!(factors.len(), 3);
    let back = factors
        .iter()
        .fold(ctx.int(1), |acc, (g, m)| acc * g.powi(*m as i64))
        .expand();
    assert_eq!(format!("{back}"), format!("{f}"));
}

#[test]
fn ex_factor_all_single_symbol_and_non_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(2) - 5 * &x + 6;
    let s = format!("{}", f.factor_all());
    assert!(s.contains("x - 2") && s.contains("x - 3"), "{s}");
    // sin(x) is left alone.
    let g = x.sin() + 1;
    assert_eq!(format!("{}", g.factor_all()), format!("{g}"));
    let (c, fs) = g.factor_list(&x);
    assert_eq!(format!("{c}"), "1");
    assert_eq!(fs.len(), 1);
}

#[test]
fn ex_factor_preserves_value_on_random_products() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let parts = [
        &x.powi(2) + 1,
        &x.powi(2) * 2 + &x + 3,
        &x.powi(3) + &x + 1,
        &x.powi(3) * 2 - 3,
        &x.powi(4) + 1,
        &x.powi(5) - &x - 1,
    ];
    for i in 0..parts.len() {
        for j in i + 1..parts.len() {
            for k in j + 1..parts.len() {
                let f = (&parts[i] * &parts[j] * &parts[k]).expand();
                let (_, factors) = f.factor_list(&x);
                assert_eq!(factors.len(), 3, "{f}");
                for v in [-2i64, -1, 0, 1, 2, 3] {
                    let lhs = f.subs_i64(&x, v).eval_f64().unwrap();
                    let rhs = f.factor(&x).subs_i64(&x, v).eval_f64().unwrap();
                    assert!((lhs - rhs).abs() <= 1e-9 * lhs.abs().max(1.0));
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Poly-level API
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn poly_cyclotomic_products_up_to_60() {
    for n in 1..=60usize {
        let factors = assert_reconstructs(&x_pow_n_minus_1(n));
        assert_eq!(factors.len(), tau(n), "x^{n} − 1 must have τ(n) factors");
    }
}

#[test]
fn poly_x105_minus_1_in_reasonable_time() {
    let start = std::time::Instant::now();
    let factors = assert_reconstructs(&x_pow_n_minus_1(105));
    assert_eq!(factors.len(), 8);
    assert!(
        start.elapsed().as_secs() < 30,
        "x^105 − 1 took {:?}",
        start.elapsed()
    );
}

#[test]
fn poly_factor_squarefree_z_examples() {
    let b = |v: i64| BigInt::from(v);
    let fs = factor_squarefree_z(&[b(4), b(0), b(0), b(0), b(1)]);
    assert_eq!(fs, vec![vec![b(2), b(-2), b(1)], vec![b(2), b(2), b(1)]]);
    let fs = factor_squarefree_z(&[b(1), b(0), b(0), b(0), b(1)]);
    assert_eq!(fs.len(), 1);
}

#[test]
fn poly_factor_mod_p_reconstructs() {
    for p in [3u64, 5, 7, 11, 13, 101] {
        let f = poly(&[7, -3, 0, 5, 1, 1]);
        let (lc, factors) = factor_mod_p(&f, p).unwrap();
        // Reconstruct mod p.
        let mut prod: Vec<u64> = vec![lc % p];
        for (g, m) in &factors {
            for _ in 0..*m {
                let mut next = vec![0u64; prod.len() + g.len() - 1];
                for (i, a) in prod.iter().enumerate() {
                    for (j, c) in g.iter().enumerate() {
                        next[i + j] = (next[i + j] + a * c) % p;
                    }
                }
                prod = next;
            }
        }
        let expected: Vec<u64> = f
            .coeffs()
            .iter()
            .map(|c| {
                let v = c.numer() % BigInt::from(p);
                let v: i64 = ((v + BigInt::from(p)) % BigInt::from(p))
                    .try_into()
                    .unwrap();
                v as u64
            })
            .collect();
        assert_eq!(prod, expected, "p = {p}");
    }
}

#[test]
fn poly_multivariate_kronecker_direct() {
    let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    // (x² + y)(x − y²)(x + y + 1)
    let one = MultiPoly::from_int(2, 1);
    let a = x.mul(&x).add(&y);
    let b = x.sub(&y.mul(&y));
    let c = x.add(&y).add(&one);
    let f = a.mul(&b).mul(&c);
    let (content, factors) = factor_multivariate(&f).unwrap();
    assert_eq!(factors.len(), 3);
    let mut back = MultiPoly::constant(2, content);
    for (g, m) in &factors {
        for _ in 0..*m {
            back = back.mul(g);
        }
    }
    assert_eq!(back, f);
    // Factors are normalised to a positive leading coefficient in the
    // monomial order, so each expected factor appears up to sign.
    for expected in [&a, &b, &c] {
        assert!(
            factors
                .iter()
                .any(|(g, _)| *g == *expected || *g == expected.neg()),
            "missing {expected:?}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Property tests
// ═══════════════════════════════════════════════════════════════════════════

fn arb_small_poly(max_deg: usize) -> impl Strategy<Value = Poly> {
    prop::collection::vec(-6i64..7, 1..=max_deg + 1).prop_map(|c| poly(&c))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// factor_zassenhaus reconstructs the input and reports only irreducibles.
    #[test]
    fn prop_factor_reconstructs(a in arb_small_poly(3), b in arb_small_poly(3), c in arb_small_poly(2)) {
        let f = &(&a * &b) * &c;
        prop_assume!(!f.is_zero());
        let (content, factors) = factor_zassenhaus_with_content(&f);
        let mut back = Poly::constant(content);
        for (g, m) in &factors {
            for _ in 0..*m {
                back = &back * g;
            }
            prop_assert_eq!(is_irreducible_z(g), Some(true));
        }
        prop_assert_eq!(back, f);
    }

    /// Products of two polynomials never come out irreducible.
    #[test]
    fn prop_product_is_reducible(a in arb_small_poly(3), b in arb_small_poly(3)) {
        prop_assume!(a.degree().unwrap_or(0) >= 1 && b.degree().unwrap_or(0) >= 1);
        let f = &a * &b;
        prop_assert_eq!(is_irreducible_z(&f), Some(false));
        let factors = factor_zassenhaus(&f);
        let total: usize = factors.iter().map(|(g, m)| g.degree().unwrap_or(0) * *m as usize).sum();
        prop_assert_eq!(total, f.degree().unwrap());
    }

    /// factor_mod_p reconstructs modulo p for random polynomials.
    #[test]
    fn prop_factor_mod_p(coeffs in prop::collection::vec(0i64..50, 1..=7), p in prop::sample::select(vec![3u64, 5, 7, 11, 13, 17])) {
        let f = poly(&coeffs);
        prop_assume!(!f.is_zero());
        let (lc, factors) = factor_mod_p(&f, p).unwrap();
        let mut prod: Vec<u64> = vec![lc % p];
        for (g, m) in &factors {
            for _ in 0..*m {
                let mut next = vec![0u64; prod.len() + g.len() - 1];
                for (i, a) in prod.iter().enumerate() {
                    for (j, c) in g.iter().enumerate() {
                        next[i + j] = (next[i + j] + a * c) % p;
                    }
                }
                prod = next;
            }
        }
        while prod.last() == Some(&0) { prod.pop(); }
        let mut expected: Vec<u64> = coeffs.iter().map(|&c| (c as u64) % p).collect();
        while expected.last() == Some(&0) { expected.pop(); }
        prop_assert_eq!(prod, expected);
    }
}
