//! Regression tests for small API-level numeric bugs fixed in 0.2 numfix:
//!
//! * `Context::rational(p, 0)` panicked inside `num-rational`.

use symplex::prelude::*;

#[test]
fn rational_with_zero_denominator_is_complex_infinity() {
    let ctx = Context::new();
    assert_eq!(ctx.rational(1, 0), ctx.complex_infinity());
    assert_eq!(ctx.rational(-7, 0), ctx.complex_infinity());
    assert_eq!(ctx.rational(i64::MAX, 0), ctx.complex_infinity());
    // Same value as division builds.
    assert_eq!(ctx.rational(5, 0), ctx.int(5) / ctx.int(0));
    assert_eq!(
        format!("{}", ctx.rational(1, 0)),
        format!("{}", ctx.complex_infinity())
    );
}

#[test]
fn rational_zero_over_zero_is_nan() {
    let ctx = Context::new();
    assert_eq!(ctx.rational(0, 0), ctx.nan());
    assert_eq!(ctx.rational(0, 0), ctx.int(0) / ctx.int(0));
}

#[test]
fn rational_nonzero_denominator_unchanged() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.rational(6, -4)), "-3/2");
    assert_eq!(ctx.rational(4, 2), ctx.int(2));
    assert_eq!(ctx.rational(0, 5), ctx.int(0));
}
