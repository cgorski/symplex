//! Integration tests for the z-transform and inverse z-transform.

mod common;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: numerically verify a z-transform result at a given z value
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate the z-transform result expression at a specific numeric z value
/// and compare against the expected numeric value.
fn verify_z_numerically(
    result: &Ex,
    z: &Ex,
    z_num: i64,
    z_den: i64,
    expected: f64,
    label: &str,
) {
    let z_val = symplex::rational(z_num, z_den);
    let at_z = result.subs(z, &z_val).eval();
    let val = at_z.eval_f64().expect(&format!(
        "{label}: should evaluate numerically at z={z_num}/{z_den}"
    ));
    assert!(
        (val - expected).abs() < 1e-3,
        "{label}: at z={z_num}/{z_den}, expected {expected}, got {val}"
    );
}

/// Verify a forward z-transform by comparing X(z) evaluated at z=r against
/// the partial sum Σₖ₌₀^N x(k)·r⁻ᵏ for large N.
fn verify_z_transform_partial_sum(
    x_of_n: &Ex,
    x_of_z: &Ex,
    n: &Ex,
    z: &Ex,
    r_num: i64,
    r_den: i64,
    num_terms: u32,
    tol: f64,
    label: &str,
) {
    // Evaluate X(z) at z = r
    let r_val = symplex::rational(r_num, r_den);
    let xz_at_r = x_of_z.subs(z, &r_val).eval();
    let xz_f64 = xz_at_r.eval_f64().expect(&format!(
        "{label}: X(z) should evaluate at z={r_num}/{r_den}"
    ));

    // Compute partial sum Σₖ₌₀^N x(k) · r⁻ᵏ
    let mut partial_sum: f64 = 0.0;
    for k in 0..num_terms {
        let k_val = symplex::int(k as i64);
        let x_at_k = x_of_n.subs(n, &k_val).eval();
        if let Ok(xk) = x_at_k.eval_f64() {
            let r_f64 = r_num as f64 / r_den as f64;
            let r_neg_k = r_f64.powi(-(k as i32));
            partial_sum += xk * r_neg_k;
        }
    }

    assert!(
        (xz_f64 - partial_sum).abs() < tol,
        "{label}: X(z) at z={r_num}/{r_den} = {xz_f64}, partial sum ({num_terms} terms) = {partial_sum}, diff = {}",
        (xz_f64 - partial_sum).abs()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward Z-transforms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn z_transform_constant() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let f = symplex::int(5);
    // Z{5} = 5z/(z-1)
    let result = f.z_transform(&n, &z).unwrap();
    let d = format!("{result}");
    assert!(
        d.contains("5") && d.contains("z"),
        "Z{{5}} should be 5z/(z-1), got: {d}"
    );
    // Numerical: at z=3, 5*3/(3-1) = 15/2 = 7.5
    verify_z_numerically(&result, &z, 3, 1, 7.5, "Z{5}");
}

#[test]
fn z_transform_unit_step() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let f = symplex::int(1);
    // Z{1} = z/(z-1)
    let result = f.z_transform(&n, &z).unwrap();
    let d = format!("{result}");
    assert!(d.contains("z"), "Z{{1}} should involve z, got: {d}");
    // Numerical: at z=4, 4/(4-1) = 4/3 ≈ 1.333
    verify_z_numerically(&result, &z, 4, 1, 4.0 / 3.0, "Z{1}");
}

#[test]
fn z_transform_exponential() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let half = symplex::rational(1, 2);
    let f = half.pow(&n); // (1/2)^n
    // Z{(1/2)^n} = z/(z - 1/2) = 2z/(2z - 1)
    let result = f.z_transform(&n, &z).unwrap();
    let d = format!("{result}");
    assert!(
        d.contains("z"),
        "Z{{(1/2)^n}} should involve z, got: {d}"
    );
    // Numerical: at z=3, 3/(3-0.5) = 3/2.5 = 1.2
    verify_z_numerically(&result, &z, 3, 1, 3.0 / 2.5, "Z{(1/2)^n}");
}

#[test]
fn z_transform_exponential_integer_base() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let two = symplex::int(2);
    let f = two.pow(&n); // 2^n
    // Z{2^n} = z/(z - 2)
    let result = f.z_transform(&n, &z).unwrap();
    let d = format!("{result}");
    assert!(
        d.contains("z") && d.contains("2"),
        "Z{{2^n}} should involve z and 2, got: {d}"
    );
    // Numerical: at z=5, 5/(5-2) = 5/3 ≈ 1.667
    verify_z_numerically(&result, &z, 5, 1, 5.0 / 3.0, "Z{2^n}");
}

#[test]
fn z_transform_sin() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    // Z{sin(3n)} = z·sin(3) / (z² - 2z·cos(3) + 1)
    let f = (&n * 3).sin();
    let result = f.z_transform(&n, &z).unwrap();
    let d = format!("{result}");
    assert!(
        d.contains("sin") && d.contains("z"),
        "Z{{sin(3n)}} should involve sin and z, got: {d}"
    );
}

#[test]
fn z_transform_cos() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    // Z{cos(n)} = z·(z - cos(1)) / (z² - 2z·cos(1) + 1)
    let f = n.cos();
    let result = f.z_transform(&n, &z).unwrap();
    let d = format!("{result}");
    assert!(
        d.contains("cos") && d.contains("z"),
        "Z{{cos(n)}} should involve cos and z, got: {d}"
    );
}

#[test]
fn z_transform_linearity() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    // Z{3·(1/2)^n + 2·(1/3)^n} should succeed via linearity
    let half = symplex::rational(1, 2);
    let third = symplex::rational(1, 3);
    let term1 = &half.pow(&n) * 3;
    let term2 = &third.pow(&n) * 2;
    let f = &term1 + &term2;
    let result = f.z_transform(&n, &z);
    assert!(
        result.is_ok(),
        "linearity should work: {:?}",
        result.err()
    );
    let r = result.unwrap();
    let d = format!("{r}");
    assert!(d.contains("z"), "result should contain z, got: {d}");
    // Numerical: at z=4, 3·4/(4-0.5) + 2·4/(4-1/3) = 12/3.5 + 8/3.667
    //          = 3.4286 + 2.1818 ≈ 5.6104
    let expected = 3.0 * 4.0 / 3.5 + 2.0 * 4.0 / (4.0 - 1.0 / 3.0);
    verify_z_numerically(&r, &z, 4, 1, expected, "Z{3·(1/2)^n + 2·(1/3)^n}");
}

#[test]
fn z_transform_n_var() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    // Z{n} = z/(z-1)²
    let result = n.z_transform(&n, &z).unwrap();
    let d = format!("{result}");
    assert!(
        d.contains("z"),
        "Z{{n}} should involve z, got: {d}"
    );
    // Numerical: at z=3, 3/(3-1)^2 = 3/4 = 0.75
    verify_z_numerically(&result, &z, 3, 1, 0.75, "Z{n}");
}

#[test]
fn z_transform_scaled_constant() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let f = symplex::int(7);
    // Z{7} = 7z/(z-1)
    let result = f.z_transform(&n, &z).unwrap();
    // at z=2, 7*2/(2-1) = 14
    verify_z_numerically(&result, &z, 2, 1, 14.0, "Z{7}");
}

#[test]
fn z_transform_n_times_a_n() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let two = symplex::int(2);
    let f = &n * &two.pow(&n); // n · 2^n
    // Z{n·2^n} = 2z/(z-2)²
    let result = f.z_transform(&n, &z).unwrap();
    let d = format!("{result}");
    assert!(
        d.contains("z") && d.contains("2"),
        "Z{{n·2^n}} should involve z and 2, got: {d}"
    );
    // Numerical: at z=5, 2·5/(5-2)² = 10/9 ≈ 1.111
    verify_z_numerically(&result, &z, 5, 1, 10.0 / 9.0, "Z{n·2^n}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse Z-transforms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_z_transform_simple() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    // Z⁻¹{z/(z-2)} = 2^n
    let f = &z / &(&z - 2);
    let result = f.inverse_z_transform(&z, &n);
    assert!(
        result.is_ok(),
        "Z⁻¹{{z/(z-2)}} should succeed: {:?}",
        result.err()
    );
    let r = result.unwrap();
    let d = format!("{r}");
    // Should contain "2" and "n" (representing 2^n)
    assert!(
        d.contains("2") && d.contains("n"),
        "Z⁻¹{{z/(z-2)}} should be 2^n, got: {d}"
    );
    // Verify: at n=3, result should be 8
    let at_3 = r.subs(&n, &symplex::int(3)).eval();
    let val = at_3.eval_f64().expect("should evaluate at n=3");
    assert!(
        (val - 8.0).abs() < 1e-6,
        "Z⁻¹{{z/(z-2)}} at n=3 should be 8, got: {val}"
    );
}

#[test]
fn inverse_z_transform_unit_step() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    // Z⁻¹{z/(z-1)} = 1^n = 1 (unit step)
    let f = &z / &(&z - 1);
    let result = f.inverse_z_transform(&z, &n);
    assert!(
        result.is_ok(),
        "Z⁻¹{{z/(z-1)}} should succeed: {:?}",
        result.err()
    );
    let r = result.unwrap();
    // Verify: at n=5, result should be 1
    let at_5 = r.subs(&n, &symplex::int(5)).eval();
    let val = at_5.eval_f64().expect("should evaluate at n=5");
    assert!(
        (val - 1.0).abs() < 1e-6,
        "Z⁻¹{{z/(z-1)}} at n=5 should be 1, got: {val}"
    );
}

#[test]
fn inverse_z_transform_scaled() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    // Z⁻¹{3z/(z-2)} = 3·2^n
    let three = symplex::int(3);
    let f = &(&three * &z) / &(&z - 2);
    let result = f.inverse_z_transform(&z, &n);
    assert!(
        result.is_ok(),
        "Z⁻¹{{3z/(z-2)}} should succeed: {:?}",
        result.err()
    );
    let r = result.unwrap();
    // Verify: at n=2, 3·2² = 12
    let at_2 = r.subs(&n, &symplex::int(2)).eval();
    let val = at_2.eval_f64().expect("should evaluate at n=2");
    assert!(
        (val - 12.0).abs() < 1e-6,
        "Z⁻¹{{3z/(z-2)}} at n=2 should be 12, got: {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Validation: rejects non-symbol arguments
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn z_transform_rejects_non_symbol_n() {
    let z = symplex::var("z");
    let f = symplex::int(1);
    let bad_n = symplex::int(42);
    let result = f.z_transform(&bad_n, &z);
    assert!(result.is_err(), "should reject non-symbol n");
}

#[test]
fn z_transform_rejects_non_symbol_z() {
    let n = symplex::var("n");
    let f = symplex::int(1);
    let bad_z = symplex::int(42);
    let result = f.z_transform(&n, &bad_z);
    assert!(result.is_err(), "should reject non-symbol z");
}

#[test]
fn inverse_z_transform_rejects_non_symbol_z() {
    let n = symplex::var("n");
    let bad_z = symplex::int(7);
    let f = symplex::int(1);
    let result = f.inverse_z_transform(&bad_z, &n);
    assert!(result.is_err(), "should reject non-symbol z");
}

#[test]
fn inverse_z_transform_rejects_non_symbol_n() {
    let z = symplex::var("z");
    let bad_n = symplex::int(7);
    let f = &z / &(&z - 1);
    let result = f.inverse_z_transform(&z, &bad_n);
    assert!(result.is_err(), "should reject non-symbol n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical verification: partial sum check
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn z_transform_numerical_verify_exponential() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let half = symplex::rational(1, 2);
    let x_of_n = half.pow(&n); // (1/2)^n
    let x_of_z = x_of_n.z_transform(&n, &z).unwrap();

    // Verify: X(z) at z=3 ≈ Σₖ₌₀^50 (1/2)^k · 3^(-k)
    verify_z_transform_partial_sum(
        &x_of_n,
        &x_of_z,
        &n,
        &z,
        3,
        1,
        50,
        1e-6,
        "Z{(1/2)^n} partial sum check",
    );
}

#[test]
fn z_transform_numerical_verify_constant() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let x_of_n = symplex::int(5);
    let x_of_z = x_of_n.z_transform(&n, &z).unwrap();

    // Verify: X(z) at z=4 ≈ Σₖ₌₀^100 5 · 4^(-k)
    verify_z_transform_partial_sum(
        &x_of_n,
        &x_of_z,
        &n,
        &z,
        4,
        1,
        100,
        1e-3,
        "Z{5} partial sum check",
    );
}

#[test]
fn z_transform_numerical_verify_2_to_n() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let two = symplex::int(2);
    let x_of_n = two.pow(&n); // 2^n
    let x_of_z = x_of_n.z_transform(&n, &z).unwrap();

    // Need |z| > |a| = 2, so use z = 5
    verify_z_transform_partial_sum(
        &x_of_n,
        &x_of_z,
        &n,
        &z,
        5,
        1,
        50,
        1e-3,
        "Z{2^n} partial sum check",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Roundtrip: forward then inverse
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_z_transform_exponential() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let half = symplex::rational(1, 2);
    let original = half.pow(&n); // (1/2)^n

    // Forward: Z{(1/2)^n} = z/(z - 1/2)
    let z_domain = original.z_transform(&n, &z).unwrap();

    // Inverse: Z⁻¹{z/(z - 1/2)} should give back (1/2)^n
    let recovered = z_domain.inverse_z_transform(&z, &n);
    assert!(
        recovered.is_ok(),
        "roundtrip inverse should succeed: {:?}",
        recovered.err()
    );
    let recovered = recovered.unwrap();

    // Verify numerically: evaluate both at n = 0, 1, 2, 3, 4
    for k in 0..=4 {
        let k_val = symplex::int(k);
        let orig_val = original.subs(&n, &k_val).eval();
        let rec_val = recovered.subs(&n, &k_val).eval();
        let o = orig_val.eval_f64().expect("original should evaluate");
        let r = rec_val.eval_f64().expect("recovered should evaluate");
        assert!(
            (o - r).abs() < 1e-6,
            "roundtrip at n={k}: original={o}, recovered={r}"
        );
    }
}

#[test]
fn roundtrip_z_transform_integer_base() {
    let n = symplex::var("n");
    let z = symplex::var("z");
    let three = symplex::int(3);
    let original = three.pow(&n); // 3^n

    let z_domain = original.z_transform(&n, &z).unwrap();
    let recovered = z_domain.inverse_z_transform(&z, &n);
    assert!(
        recovered.is_ok(),
        "roundtrip inverse should succeed: {:?}",
        recovered.err()
    );
    let recovered = recovered.unwrap();

    for k in 0..=3 {
        let k_val = symplex::int(k);
        let orig_val = original.subs(&n, &k_val).eval();
        let rec_val = recovered.subs(&n, &k_val).eval();
        let o = orig_val.eval_f64().expect("original should evaluate");
        let r = rec_val.eval_f64().expect("recovered should evaluate");
        assert!(
            (o - r).abs() < 1e-6,
            "roundtrip at n={k}: original={o}, recovered={r}"
        );
    }
}
