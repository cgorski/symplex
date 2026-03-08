//! Tests for numerical evaluation of special functions (Wave β).
//!
//! Covers: Gamma, LogGamma, Digamma, erf, erfc, Beta.

#[test]
fn evalf_gamma_at_5() {
    // Gamma(5) = 4! = 24
    let result = symplex::default_context().int(5).gamma().eval_f64().unwrap();
    assert!(
        (result - 24.0).abs() < 1e-10,
        "Gamma(5) should be 24, got {result}"
    );
}

#[test]
fn evalf_gamma_at_half() {
    // Gamma(0.5) = sqrt(pi) ≈ 1.7724538509
    let half = symplex::default_context().rational(1, 2);
    let result = half.gamma().eval_f64().unwrap();
    assert!(
        (result - std::f64::consts::PI.sqrt()).abs() < 1e-10,
        "Gamma(1/2) should be sqrt(pi), got {result}"
    );
}

#[test]
fn evalf_gamma_at_1() {
    // Gamma(1) = 0! = 1
    let result = symplex::default_context().int(1).gamma().eval_f64().unwrap();
    assert!(
        (result - 1.0).abs() < 1e-10,
        "Gamma(1) should be 1, got {result}"
    );
}

#[test]
fn evalf_gamma_at_3_5() {
    // Gamma(3.5) = 2.5 * 1.5 * 0.5 * sqrt(pi) ≈ 3.32335097
    let val = symplex::default_context().rational(7, 2);
    let result = val.gamma().eval_f64().unwrap();
    let expected = 2.5 * 1.5 * 0.5 * std::f64::consts::PI.sqrt();
    assert!(
        (result - expected).abs() < 1e-8,
        "Gamma(3.5) should be {expected}, got {result}"
    );
}

#[test]
fn evalf_gamma_at_negative_half() {
    // Gamma(-0.5) = -2*sqrt(pi) ≈ -3.5449077018
    let val = symplex::default_context().rational(-1, 2);
    let result = val.gamma().eval_f64().unwrap();
    let expected = -2.0 * std::f64::consts::PI.sqrt();
    assert!(
        (result - expected).abs() < 1e-8,
        "Gamma(-0.5) should be {expected}, got {result}"
    );
}

#[test]
fn evalf_erf_at_zero() {
    // erf(0) = 0
    let result = symplex::default_context().int(0).erf().eval_f64().unwrap();
    assert!(result.abs() < 1e-15, "erf(0) should be 0, got {result}");
}

#[test]
fn evalf_erf_at_one() {
    // erf(1) ≈ 0.8427007929
    let result = symplex::default_context().int(1).erf().eval_f64().unwrap();
    assert!(
        (result - 0.8427007929497148).abs() < 1e-10,
        "erf(1) should be ~0.8427, got {result}"
    );
}

#[test]
fn evalf_erf_at_large() {
    // erf(5) ≈ 1.0 (very close)
    let result = symplex::default_context().int(5).erf().eval_f64().unwrap();
    assert!(
        (result - 1.0).abs() < 1e-10,
        "erf(5) should be ~1.0, got {result}"
    );
}

#[test]
fn evalf_erf_negative_symmetry() {
    // erf is odd: erf(-x) = -erf(x)
    let pos = symplex::default_context().rational(3, 4).erf().eval_f64().unwrap();
    let neg = symplex::default_context().rational(-3, 4).erf().eval_f64().unwrap();
    assert!(
        (pos + neg).abs() < 1e-12,
        "erf(-x) should be -erf(x): erf(0.75)={pos}, erf(-0.75)={neg}"
    );
}

#[test]
fn evalf_erfc_at_zero() {
    // erfc(0) = 1 - erf(0) = 1
    let result = symplex::default_context().int(0).erfc().eval_f64().unwrap();
    assert!(
        (result - 1.0).abs() < 1e-12,
        "erfc(0) should be 1, got {result}"
    );
}

#[test]
fn evalf_erfc_plus_erf_is_one() {
    // erf(x) + erfc(x) = 1
    let x = symplex::default_context().rational(7, 5);
    let erf_val = x.erf().eval_f64().unwrap();
    let erfc_val = x.erfc().eval_f64().unwrap();
    assert!(
        (erf_val + erfc_val - 1.0).abs() < 1e-12,
        "erf(x) + erfc(x) should be 1: {erf_val} + {erfc_val}"
    );
}

#[test]
fn evalf_beta_2_3() {
    // Beta(2, 3) = Gamma(2)*Gamma(3)/Gamma(5) = 1*2/24 = 1/12
    let result = symplex::default_context().int(2).beta(&symplex::default_context().int(3)).eval_f64().unwrap();
    assert!(
        (result - 1.0 / 12.0).abs() < 1e-10,
        "Beta(2,3) should be 1/12, got {result}"
    );
}

#[test]
fn evalf_beta_symmetry() {
    // Beta(a,b) = Beta(b,a)
    let ab = symplex::default_context().rational(3, 2)
        .beta(&symplex::default_context().rational(5, 2))
        .eval_f64()
        .unwrap();
    let ba = symplex::default_context().rational(5, 2)
        .beta(&symplex::default_context().rational(3, 2))
        .eval_f64()
        .unwrap();
    assert!(
        (ab - ba).abs() < 1e-10,
        "Beta(3/2,5/2)={ab} should equal Beta(5/2,3/2)={ba}"
    );
}

#[test]
fn evalf_beta_half_half() {
    // Beta(1/2, 1/2) = Gamma(1/2)^2 / Gamma(1) = pi / 1 = pi
    let result = symplex::default_context().rational(1, 2)
        .beta(&symplex::default_context().rational(1, 2))
        .eval_f64()
        .unwrap();
    assert!(
        (result - std::f64::consts::PI).abs() < 1e-8,
        "Beta(1/2,1/2) should be pi, got {result}"
    );
}

#[test]
fn evalf_log_gamma_at_5() {
    // LogGamma(5) = ln(Gamma(5)) = ln(24) ≈ 3.17805383
    let result = symplex::default_context().int(5).log_gamma().eval_f64().unwrap();
    let expected = 24.0_f64.ln();
    assert!(
        (result - expected).abs() < 1e-10,
        "LogGamma(5) should be ln(24)={expected}, got {result}"
    );
}

#[test]
fn evalf_digamma_at_1() {
    // psi(1) = -gamma (Euler-Mascheroni) ≈ -0.5772156649
    let result = symplex::default_context().int(1).digamma().eval_f64().unwrap();
    let euler_mascheroni = 0.5772156649015329;
    assert!(
        (result + euler_mascheroni).abs() < 1e-10,
        "psi(1) should be -{euler_mascheroni}, got {result}"
    );
}

#[test]
fn evalf_digamma_at_2() {
    // psi(2) = psi(1) + 1/1 = 1 - gamma ≈ 0.42278433
    let result = symplex::default_context().int(2).digamma().eval_f64().unwrap();
    let expected = 1.0 - 0.5772156649015329;
    assert!(
        (result - expected).abs() < 1e-10,
        "psi(2) should be {expected}, got {result}"
    );
}
