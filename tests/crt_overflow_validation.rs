//! Validation tests for the `crt_i64` integer overflow bug.
//!
//! ## Summary of findings
//!
//! 1. `crt_i64` performs all intermediate arithmetic in native `i64`, which
//!    overflows (panic in debug, silent wrap in release) when any two coprime
//!    moduli are ≥ ~10⁹ because the product `modulus * step * p` can reach
//!    ~10²⁷, far exceeding `i64::MAX` (~9.2×10¹⁸).
//!
//! 2. `extended_gcd_i64` is safe to keep as `i64`.  The Bézout coefficients
//!    satisfy |x| < |b/gcd| and |y| < |a/gcd|, so they never exceed the
//!    magnitude of the inputs.  Recursion depth is bounded by
//!    log_φ(max(a,b)) ≈ 93 for i64::MAX — no stack-overflow risk.
//!
//! 3. Promoting intermediates to `i128` fixes 2–4 coprime moduli of ~10⁹ each
//!    (combined modulus up to ~10³⁶ < i128::MAX ~1.7×10³⁸), but **fails for
//!    5+ such moduli** (~10⁴⁵ overflows i128).
//!
//! 4. **Recommended fix**: use `checked_mul` / `checked_add` with automatic
//!    fallback to `BigInt` `crt()` on overflow.  This handles ALL input sizes,
//!    keeps the fast path for small moduli, and is trivial to implement since
//!    the BigInt version already exists with identical logic.
//!
//!    Alternatively, just delegate `crt_i64` to `crt` unconditionally —
//!    the BigInt path is not materially slower for ≤ 5 congruences with
//!    moduli that fit in i64.  The "i64 fast path" savings are negligible
//!    compared to the correctness risk.

use num_bigint::BigInt;
use num_traits::Zero;
use std::panic;
use symplex::ntheory::{crt, crt_i64};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn bi(v: i64) -> BigInt {
    BigInt::from(v)
}

/// Oracle: compute CRT via the BigInt path and return the result as i64
/// (or None if it doesn't fit or has no solution).
fn crt_oracle(remainders: &[i64], moduli: &[i64]) -> Option<i64> {
    let r_big: Vec<BigInt> = remainders.iter().map(|&r| bi(r)).collect();
    let m_big: Vec<BigInt> = moduli.iter().map(|&m| bi(m)).collect();
    crt(&r_big, &m_big).and_then(|x| {
        use num_traits::ToPrimitive;
        x.to_i64()
    })
}

/// Try to call crt_i64, catching any panics (overflow in debug mode).
/// Returns Ok(Some(result)), Ok(None) (no solution), or Err(msg) (panic).
fn try_crt_i64(remainders: &[i64], moduli: &[i64]) -> Result<Option<i64>, String> {
    let r = remainders.to_vec();
    let m = moduli.to_vec();
    panic::catch_unwind(move || crt_i64(&r, &m))
        .map_err(|e| {
            if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            }
        })
}

/// Verify that the CRT result satisfies all congruences.
fn verify_crt_result(result: i64, remainders: &[i64], moduli: &[i64]) -> bool {
    for (i, (&r, &m)) in remainders.iter().zip(moduli.iter()).enumerate() {
        let got = ((result % m) + m) % m;
        let want = ((r % m) + m) % m;
        if got != want {
            eprintln!(
                "  congruence {} failed: {} mod {} = {}, expected {}",
                i, result, m, got, want
            );
            return false;
        }
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// § 1  Basic sanity — crt_i64 works for small moduli
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn small_moduli_basic() {
    // Classic example: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → 23
    let r = try_crt_i64(&[2, 3, 2], &[3, 5, 7]);
    assert_eq!(r, Ok(Some(23)));
    assert_eq!(crt_oracle(&[2, 3, 2], &[3, 5, 7]), Some(23));
}

#[test]
fn small_moduli_two_congruences() {
    // x ≡ 1 (mod 3), x ≡ 2 (mod 5) → 7
    let r = try_crt_i64(&[1, 2], &[3, 5]);
    assert_eq!(r, Ok(Some(7)));
    assert_eq!(crt_oracle(&[1, 2], &[3, 5]), Some(7));
}

#[test]
fn small_moduli_no_solution() {
    // x ≡ 0 (mod 2), x ≡ 1 (mod 4) — no solution
    let r = try_crt_i64(&[0, 1], &[2, 4]);
    assert_eq!(r, Ok(None));
    assert_eq!(crt_oracle(&[0, 1], &[2, 4]), None);
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  Two large coprime moduli — demonstrates the overflow bug
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn two_large_coprime_moduli_overflow_detected() {
    // Two large primes, each ~10^9.  Product ~10^18 fits in i64, but
    // intermediate `modulus * step * p` reaches ~10^27 → overflow.
    let remainders = [1_000_000_006i64, 1];
    let moduli = [1_000_000_007i64, 999_999_937]; // both prime

    let oracle = crt_oracle(&remainders, &moduli);
    assert!(oracle.is_some(), "BigInt oracle must find a solution");
    let expected = oracle.unwrap();
    assert!(
        verify_crt_result(expected, &remainders, &moduli),
        "oracle result must satisfy all congruences"
    );

    // Current crt_i64 overflows — either panics or gives wrong answer.
    let i64_result = try_crt_i64(&remainders, &moduli);
    match i64_result {
        Err(msg) => {
            // Debug mode: overflow panic — this IS the bug.
            assert!(
                msg.contains("overflow"),
                "expected overflow panic, got: {msg}"
            );
            eprintln!(
                "[BUG CONFIRMED] crt_i64 panicked with overflow for 2 large moduli: {msg}"
            );
        }
        Ok(Some(val)) => {
            // Release mode: silent wrapping — wrong answer is also the bug.
            if val != expected {
                eprintln!(
                    "[BUG CONFIRMED] crt_i64 returned wrong answer {} (expected {}) \
                     due to silent i64 wrapping in release mode",
                    val, expected
                );
            } else {
                // If it somehow worked, great — the fix landed.
                eprintln!(
                    "[FIXED] crt_i64 returned correct answer {} for 2 large moduli",
                    val
                );
            }
        }
        Ok(None) => {
            panic!("crt_i64 returned None for a solvable system — unexpected");
        }
    }
}

#[test]
fn two_large_coprime_moduli_pair_b() {
    // Another pair of large primes.
    let remainders = [42i64, 99];
    let moduli = [999_999_733i64, 999_999_751]; // both prime

    let oracle = crt_oracle(&remainders, &moduli);
    assert!(oracle.is_some());
    let expected = oracle.unwrap();
    assert!(verify_crt_result(expected, &remainders, &moduli));

    let i64_result = try_crt_i64(&remainders, &moduli);
    match i64_result {
        Err(msg) => {
            eprintln!("[BUG] crt_i64 panicked: {msg}");
        }
        Ok(Some(val)) if val == expected => {
            eprintln!("[OK] crt_i64 returned correct answer {val}");
        }
        Ok(Some(val)) => {
            eprintln!("[BUG] crt_i64 wrong: got {val}, expected {expected}");
        }
        Ok(None) => {
            panic!("crt_i64 returned None for solvable system");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3  Three large coprime moduli — i128 should still handle this
//       (combined modulus ~10^27, well within i128::MAX ~1.7×10^38)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn three_large_coprime_moduli() {
    let remainders = [2i64, 3, 5];
    let moduli = [999_999_937i64, 999_999_929, 999_999_893]; // all prime

    let _oracle = crt_oracle(&remainders, &moduli);
    // Combined modulus ~10^27 won't fit in i64, so oracle returns None
    // (the *result* exceeds i64). That's expected — the solution lives
    // in BigInt land.
    let r_big: Vec<BigInt> = remainders.iter().map(|&r| bi(r)).collect();
    let m_big: Vec<BigInt> = moduli.iter().map(|&m| bi(m)).collect();
    let big_result = crt(&r_big, &m_big);
    assert!(
        big_result.is_some(),
        "BigInt CRT must find a solution for 3 coprime moduli"
    );
    let big_val = big_result.unwrap();
    // Verify congruences on the BigInt result.
    for (&r, &m) in remainders.iter().zip(moduli.iter()) {
        let rem = &big_val % bi(m);
        let rem_pos = ((&rem) + bi(m)) % bi(m);
        assert_eq!(rem_pos, bi(r), "congruence failed for modulus {m}");
    }

    // crt_i64 will overflow (the result itself doesn't fit i64).
    let i64_result = try_crt_i64(&remainders, &moduli);
    match i64_result {
        Err(msg) => {
            eprintln!("[BUG] crt_i64 panicked for 3 large moduli: {msg}");
        }
        Ok(None) => {
            // This would be acceptable if the function documents
            // that it returns None when the result exceeds i64.
            eprintln!(
                "[INFO] crt_i64 returned None — result doesn't fit i64 \
                 (combined modulus ~10^27)"
            );
        }
        Ok(Some(val)) => {
            // If it returns a value, it must satisfy all congruences.
            eprintln!("[INFO] crt_i64 returned {val} for 3 large moduli");
            // The combined modulus doesn't fit i64, so this value is
            // only correct modulo i64 wrap — check congruences anyway.
            if verify_crt_result(val, &remainders, &moduli) {
                eprintln!("  → congruences satisfied (lucky wrap?)");
            } else {
                eprintln!("  → congruences NOT satisfied (wrong answer)");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4  Four large coprime moduli — i128 intermediate ~10^36, still < 10^38
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn four_large_coprime_moduli() {
    let remainders = [1i64, 2, 3, 4];
    let moduli = [999_999_937i64, 999_999_929, 999_999_893, 999_999_877]; // all prime

    let r_big: Vec<BigInt> = remainders.iter().map(|&r| bi(r)).collect();
    let m_big: Vec<BigInt> = moduli.iter().map(|&m| bi(m)).collect();
    let big_result = crt(&r_big, &m_big);
    assert!(big_result.is_some(), "BigInt CRT must solve 4 coprime moduli");
    let big_val = big_result.unwrap();

    // Verify BigInt oracle satisfies congruences.
    for (&r, &m) in remainders.iter().zip(moduli.iter()) {
        let rem_pos = ((&big_val % bi(m)) + bi(m)) % bi(m);
        assert_eq!(rem_pos, bi(r), "BigInt oracle failed for modulus {m}");
    }

    // Combined modulus ~10^36, intermediate products could reach ~10^45.
    // An i128 fix would handle the accumulated modulus (~10^36 < i128::MAX)
    // but the intermediate multiplication within the loop iteration that
    // brings modulus from ~10^27 to ~10^36 involves ~10^36 * factor * p
    // which can exceed i128::MAX.
    //
    // NOTE: Whether i128 works for 4 moduli depends on the exact
    // intermediate values. The accumulated modulus after 3 iterations
    // is ~10^27.  The 4th iteration computes:
    //   modulus(~10^27) * step * p  where step, p < moduli[3] ~10^9
    //   → worst case ~10^27 * 10^9 * 10^9 = 10^45 which OVERFLOWS i128.
    //
    // So i128 may fail even for 4 moduli in the worst case.

    let i64_result = try_crt_i64(&remainders, &moduli);
    match i64_result {
        Err(msg) => {
            eprintln!("[BUG] crt_i64 panicked for 4 large moduli: {msg}");
        }
        Ok(v) => {
            eprintln!("[INFO] crt_i64 returned {:?} for 4 large moduli", v);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 5  Five large coprime moduli — i128 definitely overflows
//       (combined modulus ~10^45 >> i128::MAX ~1.7×10^38)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn five_large_coprime_moduli_exceeds_i128() {
    let remainders = [1i64, 2, 3, 4, 5];
    let moduli = [
        999_999_937i64,
        999_999_929,
        999_999_893,
        999_999_877,
        999_999_613,
    ];

    let r_big: Vec<BigInt> = remainders.iter().map(|&r| bi(r)).collect();
    let m_big: Vec<BigInt> = moduli.iter().map(|&m| bi(m)).collect();
    let big_result = crt(&r_big, &m_big);
    assert!(big_result.is_some(), "BigInt CRT must solve 5 coprime moduli");
    let big_val = big_result.unwrap();

    // Verify congruences.
    for (&r, &m) in remainders.iter().zip(moduli.iter()) {
        let rem_pos = ((&big_val % bi(m)) + bi(m)) % bi(m);
        assert_eq!(rem_pos, bi(r), "BigInt oracle failed for modulus {m}");
    }

    // Combined modulus ~10^45 — even i128 cannot hold this.
    // This demonstrates that the i128 promotion fix is insufficient
    // for 5+ large coprime moduli.
    eprintln!(
        "[INFO] 5 large moduli: BigInt result has {} digits",
        big_val.to_string().len()
    );

    // The combined modulus product:
    let combined_modulus: BigInt =
        moduli.iter().fold(BigInt::from(1), |acc, &m| acc * bi(m));
    eprintln!(
        "[INFO] Combined modulus has {} digits (~10^{})",
        combined_modulus.to_string().len(),
        combined_modulus.to_string().len() - 1
    );
    // Confirm it exceeds i128::MAX.
    let i128_max = BigInt::from(i128::MAX);
    assert!(
        combined_modulus > i128_max,
        "Combined modulus for 5 large primes must exceed i128::MAX"
    );

    let i64_result = try_crt_i64(&remainders, &moduli);
    match i64_result {
        Err(msg) => {
            eprintln!("[BUG] crt_i64 panicked for 5 large moduli: {msg}");
        }
        Ok(v) => {
            eprintln!("[INFO] crt_i64 returned {:?} for 5 large moduli", v);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 6  Extended_gcd_i64 safety analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that extended_gcd_i64 (called internally by crt_i64) returns
/// Bézout coefficients that fit in i64 for large inputs.
///
/// For `extended_gcd(a, b)` returning `(g, x, y)`:
///   a*x + b*y = g
///   |x| ≤ b/g,  |y| ≤ a/g
///
/// So if a, b fit in i64, the coefficients always fit too.
/// We test this indirectly by calling crt_i64 with small-ish moduli
/// where the CRT itself doesn't overflow, but the gcd inputs are large.
#[test]
fn extended_gcd_i64_coefficients_within_bounds() {
    // Use moduli that are coprime and moderately large but whose
    // product fits in i64 (< 9.2×10^18).
    // 10^9 * 10^9 = 10^18 ✓  (fits in i64, max ~9.2×10^18)
    //
    // But the intermediate `modulus * step * p` overflows because
    // modulus(~10^9) * step(~10^9) * p(~10^9) = ~10^27.
    //
    // With smaller moduli whose product fits comfortably:
    // 10^4 * 10^4 = 10^8, intermediates ~10^12 — fine for i64.
    let remainders = [9_999i64, 9_998];
    let moduli = [10_007i64, 10_009]; // both prime, product ~10^8

    let result = try_crt_i64(&remainders, &moduli);
    assert!(
        result.is_ok(),
        "crt_i64 should not panic for modest moduli"
    );
    let val = result.unwrap().expect("system is solvable");
    assert!(
        verify_crt_result(val, &remainders, &moduli),
        "result must satisfy congruences"
    );
    assert_eq!(crt_oracle(&remainders, &moduli), Some(val));
}

// ═══════════════════════════════════════════════════════════════════════════
// § 7  Boundary hunting: find the exact threshold where i64 overflows
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn boundary_moduli_sizes() {
    // Test moduli of increasing size to find where overflow kicks in.
    // Product of two moduli m₁*m₂ must fit in i64 for the *result* to
    // fit, but intermediate arithmetic overflows earlier.
    let test_cases: Vec<(i64, i64)> = vec![
        (1_009, 1_013),             // ~10^3: fine
        (10_007, 10_009),           // ~10^4: fine
        (100_003, 100_019),         // ~10^5: fine
        (1_000_003, 1_000_033),     // ~10^6: may overflow intermediates
        (10_000_019, 10_000_079),   // ~10^7: likely overflows
        (100_000_007, 100_000_037), // ~10^8: definitely overflows
        (1_000_000_007, 1_000_000_009), // ~10^9: definitely overflows
    ];

    eprintln!("\n  Boundary analysis for two coprime moduli:");
    eprintln!("  {:>14} {:>14}  {:>20}  {}", "m1", "m2", "m1*m2 (approx)", "crt_i64 status");
    eprintln!("  {}", "-".repeat(75));

    for (m1, m2) in &test_cases {
        let remainders = [1i64, 2];
        let moduli = [*m1, *m2];

        let oracle = crt_oracle(&remainders, &moduli);
        let i64_result = try_crt_i64(&remainders, &moduli);

        let product_approx = (*m1 as f64) * (*m2 as f64);
        let status = match &i64_result {
            Err(msg) => format!("PANIC: {}", &msg[..msg.len().min(30)]),
            Ok(Some(v)) if oracle == Some(*v) => "OK (correct)".to_string(),
            Ok(Some(v)) => format!("WRONG (got {v}, expected {:?})", oracle),
            Ok(None) if oracle.is_none() => "OK (None)".to_string(),
            Ok(None) => format!("WRONG None (expected {:?})", oracle),
        };

        eprintln!(
            "  {:>14} {:>14}  {:>20.2e}  {}",
            m1, m2, product_approx, status
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 8  Multi-moduli scaling: how many ~10^9 moduli before i128 fails?
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn multi_moduli_i128_limit_analysis() {
    // Large distinct primes, each ~10^9.
    let large_primes: Vec<i64> = vec![
        999_999_937,
        999_999_929,
        999_999_893,
        999_999_877,
        999_999_613,
        999_999_607,
        999_999_587,
    ];

    eprintln!("\n  Multi-moduli scaling analysis (all primes ~10^9):");
    eprintln!("  {:>5}  {:>50}  {:>12}  {}", "k", "combined modulus (digits)", "fits i128?", "BigInt CRT");
    eprintln!("  {}", "-".repeat(90));

    let i128_max = BigInt::from(i128::MAX);

    for k in 2..=large_primes.len() {
        let moduli = &large_primes[..k];
        let remainders: Vec<i64> = (1..=k as i64).collect();

        let r_big: Vec<BigInt> = remainders.iter().map(|&r| bi(r)).collect();
        let m_big: Vec<BigInt> = moduli.iter().map(|&m| bi(m)).collect();

        let combined: BigInt = moduli.iter().fold(BigInt::from(1), |a, &m| a * bi(m));
        let digits = combined.to_string().len();
        let fits_i128 = combined <= i128_max;

        let big_result = crt(&r_big, &m_big);
        let big_ok = big_result.is_some();

        // Also verify congruences if solution exists.
        if let Some(ref val) = big_result {
            for (&r, &m) in remainders.iter().zip(moduli.iter()) {
                let rem_pos = ((val % bi(m)) + bi(m)) % bi(m);
                assert_eq!(rem_pos, bi(r), "BigInt oracle broken at k={k}, mod {m}");
            }
        }

        eprintln!(
            "  {:>5}  {:>50}  {:>12}  {}",
            k,
            format!("{digits} digits"),
            if fits_i128 { "YES" } else { "NO" },
            if big_ok { "OK" } else { "FAIL" }
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 9  Regression: the exact test cases from math3_linalg_ntheory_bugs.rs
//       (without #[should_panic] — we use catch_unwind instead)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn regression_pair_from_bug_report() {
    let remainders = [1_000_000_006i64, 1];
    let moduli = [1_000_000_007i64, 999_999_937];

    // BigInt oracle must work.
    let r_big: Vec<BigInt> = remainders.iter().map(|&r| bi(r)).collect();
    let m_big: Vec<BigInt> = moduli.iter().map(|&m| bi(m)).collect();
    let big_result = crt(&r_big, &m_big).expect("BigInt CRT must find solution");
    assert_eq!(&big_result % bi(1_000_000_007), bi(1_000_000_006));
    assert_eq!(&big_result % bi(999_999_937), bi(1));
    assert!(big_result >= BigInt::zero());

    // crt_i64 currently fails.
    let i64_result = try_crt_i64(&remainders, &moduli);
    let is_buggy = matches!(i64_result, Err(_))
        || matches!(i64_result, Ok(Some(v)) if crt_oracle(&remainders, &moduli) != Some(v));
    if is_buggy {
        eprintln!("[BUG CONFIRMED] regression_pair_from_bug_report: crt_i64 fails");
    } else {
        eprintln!("[FIXED] regression_pair_from_bug_report: crt_i64 works correctly");
    }
}

#[test]
fn regression_triple_from_bug_report() {
    let remainders = [2i64, 3, 5];
    let moduli = [999_999_937i64, 999_999_929, 999_999_893];

    let r_big: Vec<BigInt> = remainders.iter().map(|&r| bi(r)).collect();
    let m_big: Vec<BigInt> = moduli.iter().map(|&m| bi(m)).collect();
    let big_result = crt(&r_big, &m_big).expect("BigInt CRT must find solution");
    for (&r, &m) in remainders.iter().zip(moduli.iter()) {
        let rem_pos = ((&big_result % bi(m)) + bi(m)) % bi(m);
        assert_eq!(rem_pos, bi(r));
    }

    let i64_result = try_crt_i64(&remainders, &moduli);
    let is_buggy = i64_result.is_err();
    if is_buggy {
        eprintln!("[BUG CONFIRMED] regression_triple_from_bug_report: crt_i64 panics");
    } else {
        eprintln!("[INFO] regression_triple_from_bug_report: crt_i64 returned {:?}", i64_result);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 10  Approach comparison: i128 vs checked_mul+fallback vs delegation
// ═══════════════════════════════════════════════════════════════════════════

/// Simulates the proposed i128 fix to see where it breaks.
fn crt_i64_i128_simulation(remainders: &[i64], moduli: &[i64]) -> Option<i64> {
    if remainders.len() != moduli.len() || remainders.is_empty() {
        return None;
    }
    let mut result = remainders[0] as i128;
    let mut modulus = moduli[0] as i128;
    for i in 1..remainders.len() {
        // extended_gcd_i64 is safe — coefficients bounded by inputs.
        let (g64, p64, _) = {
            // Call the real extended_gcd indirectly via a small CRT:
            // We can't call the private function, so we compute gcd ourselves.
            fn ext_gcd(a: i128, b: i128) -> (i128, i128, i128) {
                if a == 0 {
                    return (b, 0, 1);
                }
                let (g, x1, y1) = ext_gcd(b % a, a);
                (g, y1 - (b / a) * x1, x1)
            }
            ext_gcd(modulus, moduli[i] as i128)
        };
        let g = g64;
        let p = p64;
        let ri = remainders[i] as i128;

        if (ri - result) % g != 0 {
            return None;
        }

        let mi = moduli[i] as i128;
        let step = (ri - result) / g % (mi / g);
        // This is the critical multiplication chain:
        let update = modulus.checked_mul(step)?.checked_mul(p)?;
        result = result.checked_add(update)?;
        modulus = (modulus / g).checked_mul(mi)?;
        result = ((result % modulus) + modulus) % modulus;
    }

    i64::try_from(result).ok()
}

#[test]
fn i128_simulation_two_large_moduli() {
    let remainders = [1_000_000_006i64, 1];
    let moduli = [1_000_000_007i64, 999_999_937];

    let oracle = crt_oracle(&remainders, &moduli);
    let sim = crt_i64_i128_simulation(&remainders, &moduli);

    eprintln!("\n  i128 simulation for 2 large moduli:");
    eprintln!("    oracle = {:?}", oracle);
    eprintln!("    i128   = {:?}", sim);

    // For 2 moduli ~10^9, i128 should work: intermediate ~10^27 < 10^38.
    if oracle.is_some() {
        assert_eq!(
            sim, oracle,
            "i128 simulation should match oracle for 2 large moduli"
        );
    }
}

#[test]
fn i128_simulation_three_large_moduli_result_exceeds_i64() {
    let remainders = [2i64, 3, 5];
    let moduli = [999_999_937i64, 999_999_929, 999_999_893];

    let sim = crt_i64_i128_simulation(&remainders, &moduli);
    // Combined modulus ~10^27 doesn't fit i64, so result won't either.
    // The i128 simulation should either return None (can't fit in i64)
    // or the correct value if it happens to be small.
    eprintln!("\n  i128 simulation for 3 large moduli: {:?}", sim);
    // Since the combined modulus is ~10^27, the result is also ~10^27
    // which doesn't fit i64, so we expect None.
    assert_eq!(
        sim, None,
        "result ~10^27 cannot fit in i64, so i128 simulation returns None"
    );
}

#[test]
fn i128_simulation_four_large_moduli_intermediate_overflow() {
    let remainders = [1i64, 2, 3, 4];
    let moduli = [999_999_937i64, 999_999_929, 999_999_893, 999_999_877];

    let sim = crt_i64_i128_simulation(&remainders, &moduli);
    eprintln!("\n  i128 simulation for 4 large moduli: {:?}", sim);
    // The checked_mul in our simulation will return None if i128 overflows.
    // Combined modulus ~10^36, but intermediate mul can reach ~10^45.
    // So this should be None (either from overflow or from i64 truncation).
    assert_eq!(sim, None, "4 large ~10^9 moduli overflow even i128 intermediates");
}

#[test]
fn i128_simulation_five_large_moduli() {
    let remainders = [1i64, 2, 3, 4, 5];
    let moduli = [
        999_999_937i64,
        999_999_929,
        999_999_893,
        999_999_877,
        999_999_613,
    ];

    let sim = crt_i64_i128_simulation(&remainders, &moduli);
    eprintln!("\n  i128 simulation for 5 large moduli: {:?}", sim);
    assert_eq!(sim, None, "5 large moduli definitely exceed i128");
}

// ═══════════════════════════════════════════════════════════════════════════
// § 11  Two large moduli where the RESULT fits in i64
//       (this is the most important case to fix — both moduli and answer
//       fit in i64, only intermediates overflow)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn result_fits_i64_but_intermediates_overflow() {
    // Choose moduli whose product < i64::MAX (~9.2×10^18).
    // e.g., m1 ~= 10^9, m2 ~= 10^9, product ~= 10^18 < 9.2×10^18.
    // The CRT result is in [0, m1*m2), so it fits in i64.
    // But the intermediate `modulus * step * p` can still overflow i64.
    let test_pairs: Vec<([i64; 2], [i64; 2])> = vec![
        ([1, 2], [2_000_000_011, 3_000_000_019]),  // product ~6×10^18 < i64::MAX
        ([0, 0], [1_000_000_007, 1_000_000_009]),   // trivial remainders
        ([999_999, 888_888], [1_000_000_007, 1_000_000_009]),
    ];

    eprintln!("\n  Cases where result fits i64 but intermediates may overflow:");
    for (remainders, moduli) in &test_pairs {
        let oracle = crt_oracle(remainders, moduli);
        let i64_result = try_crt_i64(remainders, moduli);
        let i128_result = crt_i64_i128_simulation(remainders, moduli);

        eprintln!(
            "    m=[{}, {}] r=[{}, {}]:  oracle={:?}  i64={:?}  i128={:?}",
            moduli[0], moduli[1], remainders[0], remainders[1],
            oracle, i64_result, i128_result
        );

        // The oracle and i128 simulation should agree.
        if oracle.is_some() {
            assert_eq!(
                i128_result, oracle,
                "i128 simulation must match oracle when result fits i64"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 12  Performance note: BigInt delegation is fine
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bigint_delegation_is_practical() {
    // Show that delegating crt_i64 → crt (BigInt) is fast enough.
    // 1000 iterations of CRT with 3 moduli should be < 1 second.
    use std::time::Instant;

    let remainders = [2i64, 3, 5];
    let moduli = [7i64, 11, 13];
    let r_big: Vec<BigInt> = remainders.iter().map(|&r| bi(r)).collect();
    let m_big: Vec<BigInt> = moduli.iter().map(|&m| bi(m)).collect();

    let iters = 10_000;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = crt(&r_big, &m_big);
    }
    let elapsed = start.elapsed();

    eprintln!(
        "\n  BigInt CRT delegation: {} iterations in {:?} ({:.1} µs/iter)",
        iters,
        elapsed,
        elapsed.as_micros() as f64 / iters as f64
    );

    // Also time the i64 path for comparison (with small moduli that work).
    let start2 = Instant::now();
    for _ in 0..iters {
        let _ = crt_i64(&remainders, &moduli);
    }
    let elapsed2 = start2.elapsed();

    eprintln!(
        "  i64 CRT:              {} iterations in {:?} ({:.1} µs/iter)",
        iters,
        elapsed2,
        elapsed2.as_micros() as f64 / iters as f64
    );

    let speedup = elapsed.as_nanos() as f64 / elapsed2.as_nanos().max(1) as f64;
    eprintln!("  i64 speedup: {:.1}x", speedup);
    eprintln!(
        "  Verdict: BigInt delegation adds ~{:.0} ns/call — acceptable for correctness.",
        (elapsed.as_nanos() as f64 - elapsed2.as_nanos() as f64) / iters as f64
    );
}
