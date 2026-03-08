//! Integration tests for arbitrary-precision numerical evaluation,
//! focusing on the Stirling-series Gamma function path.

// ── Gamma at half-integer arguments ────────────────────────────────────

#[test]
fn evalf_gamma_half_50_digits() {
    // Γ(1/2) = √π ≈ 1.7724538509055159…
    let half = symplex::default_context().rational(1, 2);
    let result = half.gamma().eval_decimal(50).unwrap();
    assert!(
        result.starts_with("1.77245385090551"),
        "Gamma(1/2) at 50 digits: {result}"
    );
}

#[test]
fn evalf_gamma_half_30_digits() {
    let half = symplex::default_context().rational(1, 2);
    let result = half.gamma().eval_decimal(30).unwrap();
    assert!(
        result.starts_with("1.7724538509055"),
        "Gamma(1/2) at 30 digits: {result}"
    );
}

#[test]
fn evalf_gamma_3_5_high_prec() {
    // Γ(7/2) = (5/2)(3/2)(1/2)√π = 15√π/8 ≈ 3.32335097…
    let val = symplex::default_context().rational(7, 2);
    let result = val.gamma().eval_decimal(30).unwrap();
    assert!(
        result.starts_with("3.3233509"),
        "Gamma(7/2) at 30 digits: {result}"
    );
}

// ── Gamma at positive integers ─────────────────────────────────────────

#[test]
fn evalf_gamma_1_is_one() {
    // Γ(1) = 0! = 1
    let result = symplex::default_context().int(1).gamma().eval_decimal(30).unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
        (val - 1.0).abs() < 1e-12,
        "Gamma(1) should be 1, got {result}"
    );
}

#[test]
fn evalf_gamma_5_is_24() {
    // Γ(5) = 4! = 24
    // With eval-before-evalf, this should reduce to exact 24 via
    // eval (Gamma(5) → factorial(4) → 24) before Stirling ever runs.
    // If this produces "23.999..." it means eval-before-evalf is broken.
    let result = symplex::default_context().int(5).gamma().eval_decimal(50).unwrap();
    assert!(
        result.starts_with("24"),
        "Gamma(5) should be exactly 24 (eval reduces before evalf), got: {result}"
    );
}

#[test]
fn evalf_gamma_10_is_362880() {
    // Γ(10) = 9! = 362880
    let result = symplex::default_context().int(10).gamma().eval_decimal(30).unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
        (val - 362_880.0).abs() < 1e-4,
        "Gamma(10) should be 362880, got {val}"
    );
}

// ── Gamma at negative half-integers ────────────────────────────────────

#[test]
fn evalf_gamma_neg_half_30_digits() {
    // Γ(-1/2) = -2√π ≈ -3.5449077018110320…
    let val = symplex::default_context().rational(-1, 2);
    let result = val.gamma().eval_decimal(30).unwrap();
    assert!(
        result.starts_with("-3.54490770181103"),
        "Gamma(-1/2) at 30 digits: {result}"
    );
}

#[test]
fn evalf_gamma_neg_3_halves() {
    // Γ(-3/2) = (4/3)√π ≈ 2.36327180120735…
    // Actually Γ(-3/2) = 4√π/3 ≈ 2.36327180120735…
    // Using the recurrence: Γ(-1/2) = -2√π, Γ(-3/2) = Γ(-1/2)/(-3/2) = -2√π / (-3/2) = 4√π/3
    let val = symplex::default_context().rational(-3, 2);
    let result = val.gamma().eval_decimal(20).unwrap();
    let fval: f64 = result.parse().unwrap();
    let expected = 4.0 * std::f64::consts::PI.sqrt() / 3.0;
    assert!(
        (fval - expected).abs() < 1e-10,
        "Gamma(-3/2) ≈ {expected}, got {fval}"
    );
}

// ── Precision consistency ──────────────────────────────────────────────

#[test]
fn gamma_half_digits_increase_monotonically() {
    // Requesting more digits should give a longer, consistent prefix.
    let half = symplex::default_context().rational(1, 2);
    let r15 = half.gamma().eval_decimal(15).unwrap();
    let r30 = half.gamma().eval_decimal(30).unwrap();
    let r50 = half.gamma().eval_decimal(50).unwrap();

    // The 15-digit result should be a prefix of (or consistent with) the 30/50 digit results.
    let prefix = &r15[..10.min(r15.len())];
    assert!(
        r30.starts_with(prefix),
        "30-digit result should be consistent with 15-digit: r15={r15}, r30={r30}"
    );
    assert!(
        r50.starts_with(prefix),
        "50-digit result should be consistent with 15-digit: r15={r15}, r50={r50}"
    );
}

#[test]
fn gamma_integer_same_at_multiple_precisions() {
    // Γ(5) = 24 at any precision.
    let five = symplex::default_context().int(5);
    let r10 = five.gamma().eval_decimal(10).unwrap();
    let r30 = five.gamma().eval_decimal(30).unwrap();
    let r50 = five.gamma().eval_decimal(50).unwrap();

    for (digits, r) in [(10, &r10), (30, &r30), (50, &r50)] {
        let val: f64 = r.parse().unwrap();
        assert!(
            (val - 24.0).abs() < 1e-6,
            "Gamma(5) at {digits} digits should be ~24, got {r}"
        );
    }
}

// ── Gamma at non-trivial real arguments ────────────────────────────────

#[test]
fn evalf_gamma_one_quarter() {
    // Γ(1/4) ≈ 3.62560990272050…
    let val = symplex::default_context().rational(1, 4);
    let result = val.gamma().eval_decimal(20).unwrap();
    assert!(
        result.starts_with("3.6256099"),
        "Gamma(1/4) at 20 digits: {result}"
    );
}

#[test]
fn evalf_gamma_three_quarters() {
    // Γ(3/4) ≈ 1.22541670247517…
    let val = symplex::default_context().rational(3, 4);
    let result = val.gamma().eval_decimal(20).unwrap();
    assert!(
        result.starts_with("1.2254167"),
        "Gamma(3/4) at 20 digits: {result}"
    );
}

// ── Pole detection ─────────────────────────────────────────────────────

#[test]
fn evalf_gamma_zero_is_pole() {
    let result = symplex::default_context().int(0).gamma().eval_decimal(15);
    assert!(result.is_err(), "Gamma(0) should error at a pole");
}

#[test]
fn evalf_gamma_neg_integer_is_pole() {
    let result = symplex::default_context().int(-1).gamma().eval_decimal(15);
    assert!(result.is_err(), "Gamma(-1) should error at a pole");

    let result2 = symplex::default_context().int(-5).gamma().eval_decimal(15);
    assert!(result2.is_err(), "Gamma(-5) should error at a pole");
}

// ── Reflection formula identity ────────────────────────────────────────

#[test]
fn gamma_reflection_identity() {
    // Γ(z)·Γ(1−z) = π/sin(πz) for non-integer z.
    // Test with z = 1/3: Γ(1/3)·Γ(2/3) should equal 2π/√3 ≈ 3.6275987…
    // Actually: π/sin(π/3) = π/(√3/2) = 2π/√3
    let z = symplex::default_context().rational(1, 3);
    let gz = z.gamma().eval_f64().unwrap();
    let one_minus_z = symplex::default_context().rational(2, 3);
    let g1mz = one_minus_z.gamma().eval_f64().unwrap();
    let product = gz * g1mz;
    let expected = 2.0 * std::f64::consts::PI / 3.0_f64.sqrt();
    assert!(
        (product - expected).abs() < 1e-8,
        "Γ(1/3)·Γ(2/3) should be 2π/√3 ≈ {expected}, got {product}"
    );
}
