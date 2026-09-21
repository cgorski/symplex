//! symplex 0.2 base-layer fixes — `nroots` returns exactly-real roots with
//! `im == 0.0` for polynomials with rational coefficients, so that the
//! numeric real-root count always agrees with the exact Sturm count.

use symplex::prelude::*;

fn real_count(roots: &[Complex64]) -> usize {
    roots.iter().filter(|z| z.im == 0.0).count()
}

#[test]
fn x_cubed_plus_x_has_one_real_root_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3) + &x;
    assert_eq!(f.count_real_roots(&x), Some(1));
    let roots = f.nroots(&x, 20).unwrap();
    assert_eq!(roots.len(), 3, "{roots:?}");
    assert_eq!(real_count(&roots), 1, "{roots:?}");
    let zero = roots.iter().find(|z| z.im == 0.0).unwrap();
    assert_eq!(zero.re, 0.0, "exact zero root: {roots:?}");
    // The complex pair is ±i (to f64 accuracy).
    let cplx: Vec<_> = roots.iter().filter(|z| z.im != 0.0).collect();
    assert_eq!(cplx.len(), 2);
    for z in &cplx {
        assert!(
            z.re.abs() < 1e-30 && (z.im.abs() - 1.0).abs() < 1e-14,
            "{roots:?}"
        );
    }
}

#[test]
fn real_roots_have_exactly_zero_imaginary_part() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<(Ex, usize)> = vec![
        (x.powi(2) - 2, 2),
        (x.powi(3) - &x, 3),
        (x.powi(3) - &x - 1, 1),
        (x.powi(4) - 1, 2),
        (x.powi(5) - &x - 1, 1),
        (x.powi(4) + 1, 0),
        (x.powi(2) + 1, 0),
        (x.powi(3) + &x * 2, 1),
        (x.powi(6) - &x.powi(3) * 3 + 2, 2),
        ((&x.powi(2) - 2) * (&x.powi(2) + 3) * (&x - 5), 3),
    ];
    for (f, expected) in cases {
        let sturm = f.count_real_roots(&x).unwrap();
        assert_eq!(sturm, expected, "Sturm count for {f}");
        let roots = f.nroots(&x, 20).unwrap();
        assert_eq!(real_count(&roots), sturm, "{f}: {roots:?}");
        // No lingering noise on the real roots' imaginary parts.
        for z in &roots {
            assert!(
                z.im == 0.0 || z.im.abs() > 1e-6,
                "{f}: root {z} neither real nor clearly complex"
            );
        }
    }
}

#[test]
fn zero_root_with_multiplicity_is_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x²(x² + 1): double root at 0, two imaginary roots.
    let f = x.powi(4) + x.powi(2);
    let roots = f.nroots(&x, 20).unwrap();
    assert_eq!(roots.len(), 4);
    let zeros = roots
        .iter()
        .filter(|r| **r == Complex64::new(0.0, 0.0))
        .count();
    assert_eq!(zeros, 2, "{roots:?}");
    assert_eq!(real_count(&roots), 2);
    assert_eq!(
        f.count_real_roots(&x),
        Some(1),
        "Sturm counts distinct roots"
    );
    let sf = f.square_free_part(&x).unwrap();
    assert_eq!(real_count(&sf.nroots(&x, 20).unwrap()), 1);
}

#[test]
fn complex_roots_keep_their_imaginary_parts() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Roots ±i and ±2i.
    let f = (x.powi(2) + 1) * (x.powi(2) + 4);
    let roots = f.nroots(&x, 20).unwrap();
    assert_eq!(real_count(&roots), 0, "{roots:?}");
    for z in &roots {
        assert!(z.re.abs() < 1e-30, "{roots:?}");
    }
    let mut ims: Vec<f64> = roots.iter().map(|r| r.im).collect();
    ims.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (got, want) in ims.iter().zip([-2.0, -1.0, 1.0, 2.0]) {
        assert!((got - want).abs() < 1e-12, "{roots:?}");
    }
}

#[test]
fn nearly_real_complex_pair_is_not_flattened() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x - 1)² + 10⁻¹⁰: complex pair 1 ± 10⁻⁵ i — well above the noise
    // floor, must stay complex.
    let f = (&x - 1).powi(2) + ctx.rational(1, 10_000_000_000);
    assert_eq!(f.count_real_roots(&x), Some(0));
    let roots = f.nroots(&x, 20).unwrap();
    assert_eq!(real_count(&roots), 0, "{roots:?}");
    for z in &roots {
        assert!((z.im.abs() - 1e-5).abs() < 1e-12, "{roots:?}");
    }
}

#[test]
fn sturm_matches_nroots_on_sampled_polynomials() {
    // Deterministic sweep over small integer-coefficient polynomials.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let coeffs = [-3i64, -1, 0, 1, 2];
    for &a in &coeffs {
        for &b in &coeffs {
            for &c in &coeffs {
                for &d in &[-2i64, 1, 3] {
                    // d x³ + c x² + b x + a  (d ≠ 0)
                    let f = &x.powi(3) * d + &x.powi(2) * c + &x * b + a;
                    let sturm = f.count_real_roots(&x).unwrap();
                    let sf = f.square_free_part(&x).unwrap();
                    let roots = sf.nroots(&x, 20).unwrap();
                    assert_eq!(real_count(&roots), sturm, "{f}: {roots:?}");
                }
            }
        }
    }
}
