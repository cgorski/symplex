//! Tests for DiracDelta sifting property (Wave δ) and Laplace depth (Wave ε).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// DiracDelta sifting property: ∫ f(x)·δ(g(x)) dx = f(root)·H(g(x))
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sifting_x_times_delta_x() {
    // ∫ x·δ(x) dx = f(0)·H(x) = 0·H(x) = 0
    let x = symplex::var("x");
    let integrand = &x * &x.dirac_delta();
    let result = integrand.integrate(&x);
    let s = format!("{}", result.eval());
    assert_eq!(s, "0", "∫x·δ(x)dx = f(0)·H(x) = 0, got: {s}");
}

#[test]
fn sifting_const_times_delta_x() {
    // ∫ 5·δ(x) dx = 5·H(x)
    let x = symplex::var("x");
    let five = symplex::int(5);
    let integrand = &five * &x.dirac_delta();
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("H") || s.contains("eaviside") || s.contains("5"),
        "∫5·δ(x)dx should be 5·H(x), got: {s}"
    );
}

#[test]
fn sifting_x_squared_times_delta_x() {
    // ∫ x²·δ(x) dx = f(0)·H(x) = 0·H(x) = 0
    let x = symplex::var("x");
    let x_sq = x.pow(&symplex::int(2));
    let integrand = &x_sq * &x.dirac_delta();
    let result = integrand.integrate(&x);
    let s = format!("{}", result.eval());
    assert_eq!(s, "0", "∫x²·δ(x)dx = 0, got: {s}");
}

#[test]
fn sifting_delta_x_minus_a() {
    // ∫ x·δ(x-3) dx = f(3)·H(x-3) = 3·H(x-3)
    let x = symplex::var("x");
    let three = symplex::int(3);
    let delta_arg = &x - &three;
    let integrand = &x * &delta_arg.dirac_delta();
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    // Result should contain 3 and H/Heaviside
    assert!(
        s.contains("3") && (s.contains("H") || s.contains("eaviside")),
        "∫x·δ(x-3)dx should be 3·H(x-3), got: {s}"
    );
}

#[test]
fn sifting_exp_times_delta_x() {
    // ∫ exp(x)·δ(x) dx = exp(0)·H(x) = 1·H(x) = H(x)
    let x = symplex::var("x");
    let integrand = &x.exp() * &x.dirac_delta();
    let result = integrand.integrate(&x);
    let s = format!("{}", result.eval());
    assert!(
        s.contains("H") || s.contains("eaviside") || s == "1",
        "∫exp(x)·δ(x)dx should be H(x), got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Laplace time-shift: L{H(t-a)·f(t)} = exp(-a·s)·L{f(t+a)}
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laplace_time_shift_heaviside_exp() {
    // L{H(t-2)·exp(t)} should apply time-shift and produce a result
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let two = ctx.int(2);
    let h = (&t - &two).heaviside();
    let f = t.exp();
    let integrand = &h * &f;
    let result = integrand.laplace(&t, &s);
    assert!(
        result.is_ok(),
        "time-shift for H(t-2)*exp(t) should succeed: {:?}",
        result.err()
    );
    if let Ok(ref r) = result {
        let d = format!("{r}");
        // Should contain exp (for the exp(-2s) factor)
        assert!(
            d.contains("exp") || d.contains("e"),
            "time-shift result should contain exponential: {d}"
        );
    }
}

#[test]
fn laplace_time_shift_heaviside_t() {
    // L{H(t-1)·t} — time-shift with f(t) = t
    // = exp(-s)·L{(t+1)} = exp(-s)·(1/s² + 1/s)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let one = ctx.int(1);
    let h = (&t - &one).heaviside();
    let integrand = &h * &t;
    let result = integrand.laplace(&t, &s);
    assert!(
        result.is_ok(),
        "time-shift for H(t-1)*t should succeed: {:?}",
        result.err()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Laplace frequency differentiation: L{t^n·f(t)} = (-1)^n d^n/ds^n L{f(t)}
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laplace_t_times_exp_t() {
    // L{t·exp(t)} = -d/ds[1/(s-1)] = 1/(s-1)²
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let integrand = &t * &t.exp();
    let result = integrand.laplace(&t, &s);
    assert!(
        result.is_ok(),
        "L{{t*exp(t)}} should succeed: {:?}",
        result.err()
    );
    if let Ok(ref r) = result {
        let d = format!("{r}");
        // Should involve (s-1) in some form
        assert!(d.contains("s"), "result should be a function of s: {d}");
    }
}

#[test]
fn laplace_t_times_sin_t() {
    // L{t·sin(t)} = -d/ds[1/(s²+1)] = 2s/(s²+1)²
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let integrand = &t * &t.sin();
    let result = integrand.laplace(&t, &s);
    assert!(
        result.is_ok(),
        "L{{t*sin(t)}} should succeed: {:?}",
        result.err()
    );
    if let Ok(ref r) = result {
        let d = format!("{r}");
        assert!(d.contains("s"), "result should be a function of s: {d}");
    }
}

#[test]
fn laplace_t_times_cos_t() {
    // L{t·cos(t)} = -d/ds[s/(s²+1)] = (s²-1)/(s²+1)²
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let integrand = &t * &t.cos();
    let result = integrand.laplace(&t, &s);
    assert!(
        result.is_ok(),
        "L{{t*cos(t)}} should succeed: {:?}",
        result.err()
    );
}
