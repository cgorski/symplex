//! symplex 0.2 base-layer fixes — numeric evaluation boundary.
//!
//! * `eval_decimal(n)` rounds the last digit (it used to truncate).
//! * `eval_f64` on an expression with free symbols reports
//!   `SymplexError::FreeSymbol { name }` rather than an internal cache miss.

use symplex::prelude::*;

fn dec(e: &Ex, digits: u32) -> String {
    e.eval_decimal(digits).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// eval_decimal rounding
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pi_rounds_last_digit() {
    let ctx = Context::new();
    let pi = ctx.pi();
    // 3.14159265358979… → the 10th digit is 3 followed by 5897…, rounds up.
    assert_eq!(dec(&pi, 10), "3.141592654");
    assert_eq!(dec(&pi, 3), "3.14");
    assert_eq!(dec(&pi, 4), "3.142");
    assert_eq!(dec(&pi, 15), "3.14159265358979");
    assert_eq!(dec(&pi, 16), "3.141592653589793");
    assert_eq!(dec(&pi, 20), "3.1415926535897932385");
}

#[test]
fn e_rounds_last_digit() {
    let ctx = Context::new();
    let e = ctx.e();
    // 2.718281828459045… → 10 digits: 2.718281828 (next digit 4, stays).
    assert_eq!(dec(&e, 10), "2.718281828");
    // 11 digits: 2.7182818284|59 → 2.7182818285
    assert_eq!(dec(&e, 11), "2.7182818285");
    assert_eq!(dec(&e, 2), "2.7");
    assert_eq!(dec(&e, 1), "3");
}

#[test]
fn two_thirds_rounds_up() {
    let ctx = Context::new();
    let r = ctx.rational(2, 3);
    assert_eq!(dec(&r, 10), "0.6666666667");
    assert_eq!(dec(&r, 3), "0.667");
    assert_eq!(dec(&r, 1), "0.7");
    let r = ctx.rational(1, 3);
    assert_eq!(dec(&r, 5), "0.33333");
}

#[test]
fn exact_values_are_not_padded_and_ties_go_to_even() {
    let ctx = Context::new();
    assert_eq!(dec(&ctx.rational(1, 8), 10), "0.125");
    // Exact ties (the binary value is exactly …5) round half to even.
    assert_eq!(dec(&ctx.rational(1, 8), 2), "0.12");
    assert_eq!(dec(&ctx.rational(3, 8), 2), "0.38");
    assert_eq!(dec(&ctx.rational(1, 8), 1), "0.1");
    assert_eq!(dec(&ctx.int(42), 3), "42");
    assert_eq!(dec(&ctx.rational(5, 2), 1), "2");
    assert_eq!(dec(&ctx.rational(7, 2), 1), "4");
    // Not a tie: anything beyond the 5 rounds up.
    assert_eq!(dec(&ctx.rational(1_251, 10_000), 2), "0.13");
}

#[test]
fn nines_carry_all_the_way() {
    let ctx = Context::new();
    // 0.999999999999 at 5 digits → 1
    let v = ctx.rational(999_999_999_999, 1_000_000_000_000);
    assert_eq!(dec(&v, 5), "1");
    assert_eq!(dec(&v, 12), "0.999999999999");
    // 9.9996 at 4 digits → 10 (carry out of the leading digit)
    let v = ctx.rational(99_996, 10_000);
    assert_eq!(dec(&v, 4), "10");
    assert_eq!(dec(&v, 5), "9.9996");
    // 199.96 at 4 digits → 200
    let v = ctx.rational(19_996, 100);
    assert_eq!(dec(&v, 4), "200");
    // 0.09996 at 3 digits → 0.1
    let v = ctx.rational(9_996, 100_000);
    assert_eq!(dec(&v, 3), "0.1");
}

#[test]
fn negative_values_round_by_magnitude() {
    let ctx = Context::new();
    let mpi = -ctx.pi();
    assert_eq!(dec(&mpi, 10), "-3.141592654");
    assert_eq!(dec(&(-ctx.rational(2, 3)), 3), "-0.667");
    let v = -ctx.rational(99_996, 10_000);
    assert_eq!(dec(&v, 4), "-10");
}

#[test]
fn tiny_values_round_in_exponent_form() {
    let ctx = Context::new();
    // 1/(3·10¹⁰) = 3.333…e-11
    let v = ctx.rational(1, 30_000_000_000);
    assert_eq!(dec(&v, 10), "3.333333333e-11");
    // 2/(3·10¹⁰) = 6.666…e-11 → rounds up
    let v = ctx.rational(2, 30_000_000_000);
    assert_eq!(dec(&v, 10), "6.666666667e-11");
    // 9.9996e-11 at 4 digits → 1e-10
    let v = ctx.rational(99_996, 1_000_000_000_000_000);
    assert_eq!(dec(&v, 4), "1e-10");
    // Large: 2/3 · 10^20
    let v = ctx.rational(2, 3) * ctx.int(10).powi(20);
    assert_eq!(dec(&v, 5), "6.6667e19");
}

#[test]
fn sqrt2_and_complex_parts_round() {
    let ctx = Context::new();
    let r2 = ctx.int(2).sqrt();
    // 1.41421356237309504880…
    assert_eq!(dec(&r2, 10), "1.414213562");
    assert_eq!(dec(&r2, 11), "1.4142135624");
    // (1 + 2/3 i): both parts rounded.
    let z = ctx.rational(1, 3) + ctx.rational(2, 3) * ctx.i_unit();
    assert_eq!(dec(&z, 4), "0.3333 + 0.6667*i");
}

#[test]
fn eval_f64_agrees_with_rounding() {
    let ctx = Context::new();
    let v = ctx.pi().eval_f64().unwrap();
    assert_eq!(v, std::f64::consts::PI);
    let v = ctx.rational(2, 3).eval_f64().unwrap();
    assert!((v - 2.0 / 3.0).abs() <= f64::EPSILON, "{v}");
}

// ═══════════════════════════════════════════════════════════════════════════
// eval_f64 with free symbols
// ═══════════════════════════════════════════════════════════════════════════

fn expect_free_symbol(res: Result<f64, SymplexError>, expected: &str) {
    match res {
        Err(SymplexError::FreeSymbol { name }) => assert_eq!(name, expected),
        other => panic!("expected FreeSymbol {{ {expected} }}, got {other:?}"),
    }
}

#[test]
fn eval_f64_bare_symbol_reports_free_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    expect_free_symbol(x.eval_f64(), "x");
}

#[test]
fn eval_f64_compound_expression_reports_free_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (x.sin() + 1) * ctx.pi() + x.powi(2).exp();
    expect_free_symbol(e.eval_f64(), "x");
    expect_free_symbol((&x + 1).eval_f64(), "x");
    expect_free_symbol(x.sin().eval_f64(), "x");
    let y = ctx.symbol("y");
    expect_free_symbol((&y * 2).eval_f64(), "y");
}

#[test]
fn eval_f64_after_partial_substitution_names_remaining_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let e = &x * &y + 1;
    expect_free_symbol(e.subs_i64(&x, 2).eval_f64(), "y");
    expect_free_symbol(e.subs_i64(&y, 2).eval_f64(), "x");
    assert_eq!(e.subs_i64(&x, 2).subs_i64(&y, 3).eval_f64().unwrap(), 7.0);
}

#[test]
fn eval_decimal_and_complex64_report_free_symbol_too() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let e = z.cos() * ctx.i_unit();
    assert!(matches!(
        e.eval_decimal(10),
        Err(SymplexError::FreeSymbol { ref name }) if name == "z"
    ));
    assert!(matches!(
        e.eval_complex64(),
        Err(SymplexError::FreeSymbol { ref name }) if name == "z"
    ));
}

#[test]
fn bound_variables_do_not_count_as_free_for_evalf() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let s = k.sin().summation(&k, &ctx.int(1), &ctx.int(5));
    let v = s.eval_f64().unwrap();
    let expected: f64 = (1..=5).map(|i| (i as f64).sin()).sum();
    assert!((v - expected).abs() < 1e-9);
}
