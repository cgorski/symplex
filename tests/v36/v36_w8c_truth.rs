//! Tests of the 0.50 refactor (w8c): Truth inside the crate and the truth-value ratchet.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// `[[e^{−10³⁰} − e^{−2·10³⁰}]]` is positive definite: its entry is
/// `e^{−10³⁰}(1 − e^{−10³⁰}) > 0`.  Up to 0.49 the matrix sign tests fell
/// back on the sign of the entry's `f64` value, which underflows to `0.0`:
/// `is_positive_definite` was `Some(false)` and `cholesky` failed with "matrix
/// is not positive definite".  The value is below the exponent range of the
/// certified evaluation, so the answer is now `None` and `cholesky` reports
/// that it cannot decide the sign of the pivot.
///
/// Oracle (exact): `e^{−a} − e^{−2a} = e^{−a}(1 − e^{−a}) > 0` for `a > 0`.
#[test]
fn a_positive_entry_below_the_f64_range_is_not_called_indefinite() {
    let ctx = Context::new();
    let c = ctx.parse("exp(-10^30) - exp(-2*10^30)").unwrap();
    let m = Matrix::new(vec![vec![c]]).unwrap();
    assert_eq!(m.is_positive_definite(), None);
    let err = m.cholesky().unwrap_err().to_string();
    assert!(err.contains("cannot decide the sign"), "{err}");
    assert!(!err.contains("not positive definite"), "{err}");
}

/// `[[e^{−10⁻⁵⁰⁰⁰} − 1]]` (`≈ −10⁻⁵⁰⁰⁰`) and `[[e^{−2·10³⁰} − e^{−10³⁰}]]`
/// are negative, so not positive semidefinite.  Up to 0.49
/// `is_positive_semidefinite` was `Some(true)` for both: the `f64` value of
/// the entry, `0.0`, passed `v ≥ 0`.  Now `None` (no certified sign).
///
/// Oracles: `e^t < 1` for `t < 0` (exact; mpmath `mp.dps = 5100;
/// exp(-mpf(10)**-5000) - 1` = `-1.0e-5000`); `e^{−2a} − e^{−a} =
/// −e^{−a}(1 − e^{−a}) < 0`.
#[test]
fn a_negative_entry_that_rounds_to_zero_is_not_semidefinite() {
    let ctx = Context::new();
    for s in ["exp(-10^-5000) - 1", "exp(-2*10^30) - exp(-10^30)"] {
        let c = ctx.parse(s).unwrap();
        let m = Matrix::new(vec![vec![c]]).unwrap();
        assert_eq!(m.is_positive_semidefinite(), None, "{s}");
    }
    // Decided entries are unaffected.
    let m = Matrix::new(vec![vec![ctx.parse("sqrt(2) - 1").unwrap()]]).unwrap();
    assert_eq!(m.is_positive_definite(), Some(true));
    let m = Matrix::new(vec![vec![ctx.parse("1 - sqrt(2)").unwrap()]]).unwrap();
    assert_eq!(m.is_positive_semidefinite(), Some(false));
}

/// `y′ = y`, `y(0) = 0`, `y′(0) = e^{−10⁻⁵⁰⁰⁰} − 1` has no solution: `y(0)
/// = 0` forces `y ≡ 0`, whose derivative at 0 is `0 ≠ e^{−10⁻⁵⁰⁰⁰} − 1`.
/// Up to 0.49 `solve_ode_ivp` returned `y = 0`: the leftover condition
/// `0 = e^{−10⁻⁵⁰⁰⁰} − 1` was "satisfied" because its `f64` value was below
/// `10⁻¹⁰`.  The condition is now decided by `is_zero` (exact tests,
/// certified values), which cannot decide it, so the call refuses; a
/// condition proved false still reports contradictory conditions.
///
/// Oracle: `e^t − 1 < 0` for `t < 0` (exact; mpmath at 5100 digits gives
/// `-1.0e-5000`).
#[test]
fn an_initial_condition_below_1e_10_is_not_taken_as_satisfied() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let ode = &y.formal_diff(&x) - &y;
    let ics = |v: &str| {
        [
            InitialCondition {
                order: 0,
                x: ctx.int(0),
                value: ctx.int(0),
            },
            InitialCondition {
                order: 1,
                x: ctx.int(0),
                value: ctx.parse(v).unwrap(),
            },
        ]
    };
    let r = ode.solve_ode_ivp(&y, &x, &ics("exp(-10^-5000) - 1"));
    assert!(r.is_err(), "{r:?}");
    let r = ode.solve_ode_ivp(&y, &x, &ics("10^-11"));
    assert!(
        r.unwrap_err().to_string().contains("contradictory"),
        "a decided nonzero residual is a contradiction"
    );
    let r = ode.solve_ode_ivp(&y, &x, &ics("0")).unwrap();
    assert!(r.is_zero_structural(), "{r}");
}
