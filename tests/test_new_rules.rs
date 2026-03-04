//! Tests for new simplification rules:
//! asin(sin(x))→x, acos(cos(x))→x, atan(tan(x))→x, cosh²-sinh²→1

#[test]
fn simplify_asin_sin() {
    let x = symplex::var("x");
    let expr = x.sin().asin();
    assert_eq!(format!("{}", expr.simplify()), "x");
}

#[test]
fn simplify_acos_cos() {
    let x = symplex::var("x");
    let expr = x.cos().acos();
    assert_eq!(format!("{}", expr.simplify()), "x");
}

#[test]
fn simplify_atan_tan() {
    let x = symplex::var("x");
    let expr = x.tan().atan();
    assert_eq!(format!("{}", expr.simplify()), "x");
}

#[test]
fn simplify_cosh_sinh_identity() {
    let x = symplex::var("x");
    // cosh²(x) - sinh²(x) = 1
    let expr = &x.cosh().powi(2) - &x.sinh().powi(2);
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "1");
}

#[test]
fn simplify_cosh_sinh_in_larger_sum() {
    let x = symplex::var("x");
    // 5 + cosh²(x) - sinh²(x) = 6
    let expr = &x.cosh().powi(2) - &x.sinh().powi(2) + 5;
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "6");
}

#[test]
fn simplify_inverse_trig_nested() {
    let x = symplex::var("x");
    // asin(sin(x)) + 1 should simplify to x + 1
    let expr = &x.sin().asin() + 1;
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "x + 1");
}

#[test]
fn full_simplify_inverse_trig() {
    let x = symplex::var("x");
    let expr = x.sin().asin();
    assert_eq!(format!("{}", expr.full_simplify()), "x");
}
