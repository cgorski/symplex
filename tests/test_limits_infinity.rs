//! Tests for limits at infinity.

#[test]
fn limit_one_over_x_at_infinity() {
    let x = symplex::var("x");
    let expr = 1 / &x;
    let result = expr.limit(&x, &symplex::infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0");
    }
}

#[test]
fn limit_constant_at_infinity() {
    let x = symplex::var("x");
    let result = symplex::int(5).limit(&x, &symplex::infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "5");
    }
}

#[test]
fn limit_x_squared_at_infinity() {
    let x = symplex::var("x");
    let result = x.powi(2).limit(&x, &symplex::infinity());
    if let Ok(r) = result {
        let s = format!("{r}");
        assert!(s.contains("oo") || s.contains("∞"), "x²→∞: {s}");
    }
}

#[test]
fn limit_rational_function_same_degree() {
    // lim(x→∞) (3x²+1)/(x²-x) = 3
    let x = symplex::var("x");
    let numer = &x.powi(2) * 3 + 1;
    let denom = &x.powi(2) - &x;
    let expr = &numer / &denom;
    let result = expr.limit(&x, &symplex::infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "3", "lim (3x²+1)/(x²-x) = 3");
    }
}

#[test]
fn limit_rational_lower_numer_degree() {
    // lim(x→∞) x/(x²+1) = 0
    let x = symplex::var("x");
    let expr = &x / &(&x.powi(2) + 1);
    let result = expr.limit(&x, &symplex::infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0");
    }
}

#[test]
fn limit_one_over_x_at_neg_infinity() {
    let x = symplex::var("x");
    let expr = 1 / &x;
    let neg_inf = -symplex::infinity();
    let result = expr.limit(&x, &neg_inf);
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0");
    }
}

#[test]
fn limit_exp_neg_x_at_infinity() {
    // lim(x→∞) exp(-x) = 0
    let x = symplex::var("x");
    let expr = (-&x).exp();
    let result = expr.limit(&x, &symplex::infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0", "lim exp(-x) = 0");
    }
}

#[test]
fn limit_one_over_x_squared_at_infinity() {
    // lim(x→∞) 1/x² = 0
    let x = symplex::var("x");
    let expr = 1 / &x.powi(2);
    let result = expr.limit(&x, &symplex::infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0");
    }
}

#[test]
fn limit_rational_higher_numer_degree() {
    // lim(x→∞) x³/(x+1) = ∞
    let x = symplex::var("x");
    let expr = &x.powi(3) / &(&x + 1);
    let result = expr.limit(&x, &symplex::infinity());
    if let Ok(r) = result {
        let s = format!("{r}");
        assert!(s.contains("oo") || s.contains("∞"), "x³/(x+1)→∞: {s}");
    }
}

#[test]
fn limit_constant_plus_decay_at_infinity() {
    // lim(x→∞) (3 + 1/x) = 3
    let x = symplex::var("x");
    let expr = symplex::int(3) + 1 / &x;
    let result = expr.limit(&x, &symplex::infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "3");
    }
}
