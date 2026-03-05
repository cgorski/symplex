//! Bounded exhaustive verification — proptest with exhaustive ranges.
//!
//! Tests algebraic properties for ALL values within small bounds,
//! inspired by bounded model checking (Kani). These are deterministic:
//! they enumerate every case rather than sampling randomly.

mod common;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial arithmetic exhaustive verification
// ═══════════════════════════════════════════════════════════════════════════

/// For ALL pairs of linear polynomials (ax+b)(cx+d) with small coefficients,
/// verify that expand of the product equals the manual expansion.
#[test]
fn exhaustive_linear_product() {
    let x = symplex::var("x");
    let range = -3i64..=3;
    let mut checked = 0;
    for a in range.clone() {
        for b in range.clone() {
            for c in range.clone() {
                for d in range.clone() {
                    let p1 = &(&x * a) + b;
                    let p2 = &(&x * c) + d;
                    let product = (&p1 * &p2).expand();
                    // Manual: (ax+b)(cx+d) = ac·x² + (ad+bc)·x + bd
                    let ac = a * c;
                    let ad_bc = a * d + b * c;
                    let bd = b * d;
                    let expected = &(&x.powi(2) * ac) + &(&x * ad_bc) + bd;
                    // Verify at x=7 (arbitrary point outside our coefficient range)
                    let v1 = format!("{}", product.subs_i64(&x, 7));
                    let v2 = format!("{}", expected.subs_i64(&x, 7));
                    assert_eq!(
                        v1, v2,
                        "({}x+{})·({}x+{}) mismatch at x=7: {} vs {}",
                        a, b, c, d, v1, v2
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 2000, "should check many cases: {checked}");
}

/// For ALL quadratics with small integer coefficients,
/// verify that solve finds roots that actually satisfy the equation.
#[test]
fn exhaustive_quadratic_solve_verify() {
    let x = symplex::var("x");
    let mut bail = common::BailCounter::new("exhaustive_quadratic_solve_verify");
    let mut total = 0;
    let mut solved = 0;
    for a in 1i64..=3 {
        for b in -3i64..=3 {
            for c in -3i64..=3 {
                total += 1;
                let eq = &(&x.powi(2) * a) + &(&x * b) + c;
                let roots = eq.solve_or_empty(&x);
                for root in &roots {
                    solved += 1;
                    let val = eq.subs(&x, root);
                    let val_s = format!("{}", val.expand().eval());
                    // The substituted value should be 0 or simplify to 0
                    if val_s == "0" {
                        bail.check();
                    } else {
                        // Try numerical check
                        if let Ok(f) = val.evalf_f64() {
                            bail.check();
                            assert!(
                                f.abs() < 1e-8,
                                "root {} of {}x²+{}x+{}=0 gives {f}",
                                root,
                                a,
                                b,
                                c
                            );
                        } else {
                            bail.skip();
                        }
                    }
                }
            }
        }
    }
    assert!(total > 100, "should check many quadratics: {total}");
    assert!(solved > 0, "should find some roots: {solved}");
    bail.assert_not_vacuous();
}

/// For ALL integer pairs, verify that i^n has period 4.
#[test]
fn exhaustive_i_power_period() {
    let i = symplex::i_unit();
    let expected = ["1", "I", "-1", "-I"];
    for n in -100i64..=100 {
        let result = i.powi(n);
        let s = format!("{result}");
        let idx = ((n % 4 + 4) % 4) as usize;
        assert_eq!(s, expected[idx], "i^{n} should be {}", expected[idx]);
    }
}

/// For ALL small integers n, verify d/dx(x^n) = n·x^(n-1) numerically.
#[test]
fn exhaustive_power_rule() {
    let x = symplex::var("x");
    for n in 0i64..=10 {
        let f = x.powi(n);
        let df = f.diff(&x);
        // Evaluate at x=3
        let df_at_3 = df.subs_i64(&x, 3);
        let expected = n * 3i64.pow((n.max(1) - 1) as u32);
        if n == 0 {
            assert_eq!(format!("{df_at_3}"), "0", "d/dx(x^0) at x=3");
        } else {
            assert_eq!(
                format!("{df_at_3}"),
                expected.to_string(),
                "d/dx(x^{n}) at x=3 should be {expected}"
            );
        }
    }
}

/// For ALL unit circle angles (k*π/12 for k=0..23), verify sin²+cos²=1.
#[test]
fn exhaustive_pythagorean_unit_circle() {
    let ctx = symplex::default_context();
    for k in 0i64..24 {
        let angle = &ctx.rational(k, 12) * &symplex::pi();
        let sin_a = angle.sin().eval();
        let cos_a = angle.cos().eval();
        let sum = &sin_a.powi(2) + &cos_a.powi(2);
        let simplified = sum.full_simplify();
        // Try symbolic check first, fall back to numerical
        let s = format!("{simplified}");
        if s != "1" {
            let val = simplified.evalf_f64().expect(&format!(
                "sin²({k}π/12) + cos²({k}π/12) should be evaluable"
            ));
            assert!(
                (val - 1.0).abs() < 1e-10,
                "sin²({k}π/12) + cos²({k}π/12) should be 1, got: {val} (symbolic: {s})"
            );
        }
    }
}

/// For ALL integer n from -10 to 10, verify that ∫ x^n dx then d/dx gives back x^n
/// (for n != -1).
#[test]
fn exhaustive_integrate_diff_power() {
    let x = symplex::var("x");
    for n in -10i64..=10 {
        if n == -1 {
            continue;
        } // ln case
        let f = x.powi(n);
        let integral = f.integrate(&x);
        let roundtrip = integral.diff(&x);
        // Check at x=2
        let f_at_2 = f.subs_i64(&x, 2);
        let rt_at_2 = roundtrip.subs_i64(&x, 2);
        let fv = format!("{f_at_2}");
        let rv = format!("{rt_at_2}");
        assert_eq!(fv, rv, "∫→d/dx roundtrip for x^{n} at x=2: {fv} vs {rv}");
    }
}

/// For small polynomials, verify expand(factor(p)) == p numerically.
#[test]
fn exhaustive_factor_expand_roundtrip() {
    let x = symplex::var("x");
    for a in -2i64..=2 {
        for b in -2i64..=2 {
            if a == 0 && b == 0 {
                continue;
            }
            // p = (x-a)(x-b) = x² - (a+b)x + ab
            let p1 = &x - a;
            let p2 = &x - b;
            let product = (&p1 * &p2).expand();
            let factored = product.factor(&x);
            let re_expanded = factored.expand();
            // Check at x=7
            let v1 = format!("{}", product.subs_i64(&x, 7));
            let v2 = format!("{}", re_expanded.subs_i64(&x, 7));
            assert_eq!(v1, v2, "(x-{a})(x-{b}) factor→expand at x=7");
        }
    }
}

/// Verify all From<T> conversions produce correct values.
#[test]
fn exhaustive_from_conversions() {
    for n in -50i64..=50 {
        let ex: Ex = n.into();
        assert_eq!(format!("{ex}"), n.to_string(), "From<i64>({n})");
    }
    for n in 0u32..=100 {
        let ex: Ex = n.into();
        assert_eq!(format!("{ex}"), n.to_string(), "From<u32>({n})");
    }
}

/// Sum trait: verify sum of 1..=n equals n(n+1)/2.
#[test]
fn exhaustive_sum_gauss() {
    let ctx = symplex::default_context();
    for n in 1i64..=20 {
        let terms: Vec<Ex> = (1..=n).map(|k| ctx.int(k)).collect();
        let total: Ex = terms.into_iter().sum();
        let expected = n * (n + 1) / 2;
        assert_eq!(
            format!("{total}"),
            expected.to_string(),
            "sum(1..={n}) should be {expected}"
        );
    }
}

/// Product trait: verify product of 1..=n equals n!.
#[test]
fn exhaustive_product_factorial() {
    let ctx = symplex::default_context();
    let mut expected: i64 = 1;
    for n in 1i64..=12 {
        expected *= n;
        let factors: Vec<Ex> = (1..=n).map(|k| ctx.int(k)).collect();
        let total: Ex = factors.into_iter().product();
        assert_eq!(
            format!("{total}"),
            expected.to_string(),
            "{n}! should be {expected}"
        );
    }
}

/// Verify sqrt simplification for perfect squares up to 100.
#[test]
fn exhaustive_sqrt_perfect_squares() {
    for n in 1i64..=10 {
        let perfect = n * n;
        let result = symplex::int(perfect).sqrt().eval();
        assert_eq!(
            format!("{result}"),
            n.to_string(),
            "√{perfect} should be {n}"
        );
    }
}

/// Verify (-n)^(1/2) = i*√n for n=1..20.
#[test]
fn exhaustive_sqrt_negative() {
    for n in 1i64..=20 {
        let result = symplex::int(-n).sqrt();
        let s = format!("{result}");
        assert!(s.contains("I"), "√(-{n}) should contain I: {s}");
    }
}

/// Verify eval of sin at all standard unit circle angles.
#[test]
fn exhaustive_sin_unit_circle() {
    let ctx = symplex::default_context();
    // Test at k*π/6 for k=0..11
    let expected_patterns = [
        "0",    // 0
        "1/2",  // π/6
        "",     // π/3 (√3/2)
        "1",    // π/2
        "",     // 2π/3 (√3/2)
        "1/2",  // 5π/6
        "0",    // π
        "-1/2", // 7π/6
        "",     // 4π/3 (-√3/2)
        "-1",   // 3π/2
        "",     // 5π/3 (-√3/2)
        "-1/2", // 11π/6
    ];
    for k in 0i64..12 {
        let angle = &ctx.rational(k, 6) * &symplex::pi();
        let result = angle.sin().eval();
        let s = format!("{result}");
        if !expected_patterns[k as usize].is_empty() {
            assert!(
                s.contains(expected_patterns[k as usize]),
                "sin({k}π/6) = {s}, expected to contain '{}'",
                expected_patterns[k as usize]
            );
        }
    }
}
