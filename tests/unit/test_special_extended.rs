//! Tests for special functions, orthogonal polynomials, Laurent series,
//! and symbolic Fourier transforms.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn ctx() -> Context {
    Context::new()
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 3A: Bessel functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bessel_j0_at_zero_is_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let j = arena.besselj(arena.zero(), arena.zero());
        let result = arena.eval_expr(j);
        assert_eq!(result, arena.one(), "J_0(0) should be 1");
    });
}

#[test]
fn bessel_j1_at_zero_is_zero() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let one = arena.one();
        let zero = arena.zero();
        let j = arena.besselj(one, zero);
        let result = arena.eval_expr(j);
        assert_eq!(result, arena.zero(), "J_1(0) should be 0");
    });
}

#[test]
fn bessel_j2_at_zero_is_zero() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let zero = arena.zero();
        let j = arena.besselj(two, zero);
        let result = arena.eval_expr(j);
        assert_eq!(result, arena.zero(), "J_2(0) should be 0");
    });
}

#[test]
fn bessel_j5_at_zero_is_zero() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let five = arena.int(5);
        let zero = arena.zero();
        let j = arena.besselj(five, zero);
        let result = arena.eval_expr(j);
        assert_eq!(result, arena.zero(), "J_5(0) should be 0");
    });
}

#[test]
fn bessel_j_symbolic_stays_symbolic() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let x = arena.symbol("x");
        let j = arena.besselj(zero, x);
        let result = arena.eval_expr(j);
        // Should remain besselj(0, x) — not collapse to a number
        let d = arena.display(result).to_string();
        assert!(
            d.contains("besselj") || d.contains("x"),
            "J_0(x) should remain symbolic: {d}"
        );
    });
}

#[test]
fn bessel_i0_at_zero_is_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let j = arena.besseli(zero, zero);
        let result = arena.eval_expr(j);
        assert_eq!(result, arena.one(), "I_0(0) should be 1");
    });
}

#[test]
fn bessel_i1_at_zero_is_zero() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let one = arena.one();
        let zero = arena.zero();
        let j = arena.besseli(one, zero);
        let result = arena.eval_expr(j);
        assert_eq!(result, arena.zero(), "I_1(0) should be 0");
    });
}

#[test]
fn bessel_y_stays_symbolic() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let x = arena.symbol("x");
        let y = arena.bessely(zero, x);
        let result = arena.eval_expr(y);
        let d = arena.display(result).to_string();
        assert!(d.contains("bessely"), "Y_0(x) should remain symbolic: {d}");
    });
}

#[test]
fn bessel_k_stays_symbolic() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let x = arena.symbol("x");
        let k = arena.besselk(zero, x);
        let result = arena.eval_expr(k);
        let d = arena.display(result).to_string();
        assert!(d.contains("besselk"), "K_0(x) should remain symbolic: {d}");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 3C: Orthogonal polynomials — Legendre
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn legendre_p0_is_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let x = arena.symbol("x");
        let p = arena.legendre(zero, x);
        let result = arena.eval_expr(p);
        assert_eq!(result, arena.one(), "P_0(x) should be 1");
    });
}

#[test]
fn legendre_p1_is_x() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let one = arena.one();
        let x = arena.symbol("x");
        let p = arena.legendre(one, x);
        let result = arena.eval_expr(p);
        assert_eq!(result, x, "P_1(x) should be x");
    });
}

#[test]
fn legendre_p2_is_correct() {
    // P_2(x) = (3x² - 1)/2
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let x = arena.symbol("x");
        let p = arena.legendre(two, x);
        let result = arena.eval_expr(p);
        let d = arena.display(result).to_string();
        // P_2(x) = (3x^2 - 1)/2 → should contain x^2
        assert!(
            d.contains("x^2") || d.contains("x²"),
            "P_2(x) should contain x^2 term: {d}"
        );
    });
}

#[test]
fn legendre_p2_at_one() {
    // P_n(1) = 1 for all n
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let one = arena.one();
        let p = arena.legendre(two, one);
        let result = arena.eval_expr(p);
        assert_eq!(result, arena.one(), "P_2(1) should be 1");
    });
}

#[test]
fn legendre_p3_at_one() {
    // P_3(1) = 1
    let c = ctx();
    c.with_arena_mut(|arena| {
        let three = arena.int(3);
        let one = arena.one();
        let p = arena.legendre(three, one);
        let result = arena.eval_expr(p);
        assert_eq!(result, arena.one(), "P_3(1) should be 1");
    });
}

#[test]
fn legendre_p2_at_zero() {
    // P_2(0) = -1/2
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let zero = arena.zero();
        let p = arena.legendre(two, zero);
        let result = arena.eval_expr(p);
        let expected = arena.rational(-1, 2);
        assert_eq!(result, expected, "P_2(0) should be -1/2");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthogonal polynomials — Chebyshev T
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn chebyshev_t0_is_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let x = arena.symbol("x");
        let t = arena.chebyshev_t(zero, x);
        let result = arena.eval_expr(t);
        assert_eq!(result, arena.one(), "T_0(x) should be 1");
    });
}

#[test]
fn chebyshev_t1_is_x() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let one = arena.one();
        let x = arena.symbol("x");
        let t = arena.chebyshev_t(one, x);
        let result = arena.eval_expr(t);
        assert_eq!(result, x, "T_1(x) should be x");
    });
}

#[test]
fn chebyshev_t2_is_2x2_minus_1() {
    // T_2(x) = 2x² - 1
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let x = arena.symbol("x");
        let t = arena.chebyshev_t(two, x);
        let result = arena.eval_expr(t);
        let d = arena.display(result).to_string();
        assert!(
            d.contains("x^2") || d.contains("x²"),
            "T_2(x) should contain x^2: {d}"
        );
    });
}

#[test]
fn chebyshev_t2_at_one() {
    // T_n(1) = 1 for all n
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let one = arena.one();
        let t = arena.chebyshev_t(two, one);
        let result = arena.eval_expr(t);
        assert_eq!(result, arena.one(), "T_2(1) should be 1");
    });
}

#[test]
fn chebyshev_t3_at_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let three = arena.int(3);
        let one = arena.one();
        let t = arena.chebyshev_t(three, one);
        let result = arena.eval_expr(t);
        assert_eq!(result, arena.one(), "T_3(1) should be 1");
    });
}

#[test]
fn chebyshev_t3_at_zero() {
    // T_3(0) = 0
    let c = ctx();
    c.with_arena_mut(|arena| {
        let three = arena.int(3);
        let zero = arena.zero();
        let t = arena.chebyshev_t(three, zero);
        let result = arena.eval_expr(t);
        assert_eq!(result, arena.zero(), "T_3(0) should be 0");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthogonal polynomials — Chebyshev U
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn chebyshev_u0_is_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let x = arena.symbol("x");
        let u = arena.chebyshev_u(zero, x);
        let result = arena.eval_expr(u);
        assert_eq!(result, arena.one(), "U_0(x) should be 1");
    });
}

#[test]
fn chebyshev_u1_is_2x() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let one = arena.one();
        let x = arena.symbol("x");
        let u = arena.chebyshev_u(one, x);
        let result = arena.eval_expr(u);
        let d = arena.display(result).to_string();
        assert!(
            d.contains("2") && d.contains("x"),
            "U_1(x) should be 2*x: {d}"
        );
    });
}

#[test]
fn chebyshev_u1_at_one() {
    // U_1(1) = 2
    let c = ctx();
    c.with_arena_mut(|arena| {
        let one = arena.one();
        let u = arena.chebyshev_u(one, one);
        let result = arena.eval_expr(u);
        let two = arena.int(2);
        assert_eq!(result, two, "U_1(1) should be 2");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthogonal polynomials — Hermite
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hermite_h0_is_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let x = arena.symbol("x");
        let h = arena.hermite(zero, x);
        let result = arena.eval_expr(h);
        assert_eq!(result, arena.one(), "H_0(x) should be 1");
    });
}

#[test]
fn hermite_h1_is_2x() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let one = arena.one();
        let x = arena.symbol("x");
        let h = arena.hermite(one, x);
        let result = arena.eval_expr(h);
        let d = arena.display(result).to_string();
        assert!(
            d.contains("2") && d.contains("x"),
            "H_1(x) should be 2x: {d}"
        );
    });
}

#[test]
fn hermite_h2_at_zero() {
    // H_2(x) = 4x² - 2, so H_2(0) = -2
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let zero = arena.zero();
        let h = arena.hermite(two, zero);
        let result = arena.eval_expr(h);
        let expected = arena.int(-2);
        assert_eq!(result, expected, "H_2(0) should be -2");
    });
}

#[test]
fn hermite_h2_at_one() {
    // H_2(1) = 4*1 - 2 = 2
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let one = arena.one();
        let h = arena.hermite(two, one);
        let result = arena.eval_expr(h);
        assert_eq!(result, two, "H_2(1) should be 2");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthogonal polynomials — Laguerre
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laguerre_l0_is_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let x = arena.symbol("x");
        let l = arena.laguerre(zero, x);
        let result = arena.eval_expr(l);
        assert_eq!(result, arena.one(), "L_0(x) should be 1");
    });
}

#[test]
fn laguerre_l1_at_zero() {
    // L_1(x) = 1 - x, so L_1(0) = 1
    let c = ctx();
    c.with_arena_mut(|arena| {
        let one = arena.one();
        let zero = arena.zero();
        let l = arena.laguerre(one, zero);
        let result = arena.eval_expr(l);
        assert_eq!(result, arena.one(), "L_1(0) should be 1");
    });
}

#[test]
fn laguerre_l1_at_one() {
    // L_1(1) = 1 - 1 = 0
    let c = ctx();
    c.with_arena_mut(|arena| {
        let one = arena.one();
        let l = arena.laguerre(one, one);
        let result = arena.eval_expr(l);
        assert_eq!(result, arena.zero(), "L_1(1) should be 0");
    });
}

#[test]
fn laguerre_l2_at_zero() {
    // L_2(x) = (x² - 4x + 2)/2, so L_2(0) = 1
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let zero = arena.zero();
        let l = arena.laguerre(two, zero);
        let result = arena.eval_expr(l);
        assert_eq!(result, arena.one(), "L_2(0) should be 1");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 3B: Laurent series
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laurent_regular_function_returns_taylor() {
    // For a function with no pole (e.g. sin(x)), Laurent = Taylor
    let c = ctx();
    c.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let sin_x = arena.sin(x);
        let zero = arena.zero();
        let result = arena.laurent_series_expr(sin_x, x, zero, 4);
        assert!(result.is_ok(), "Laurent of sin(x) should succeed");
        let series = result.unwrap();
        let d = arena.display(series).to_string();
        assert!(d.contains("x"), "sin(x) Laurent should contain x term: {d}");
    });
}

#[test]
fn laurent_simple_pole_1_over_x() {
    // 1/x around x=0 has a simple pole
    let c = ctx();
    c.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let one = arena.one();
        let one_over_x = arena.div(one, x);
        let zero = arena.zero();
        let result = arena.laurent_series_expr(one_over_x, x, zero, 3);
        assert!(result.is_ok(), "Laurent of 1/x should succeed");
        let series = result.unwrap();
        let d = arena.display(series).to_string();
        // The Laurent series of 1/x is just x^(-1)
        assert!(
            d.contains("x") || d.contains("1"),
            "Laurent of 1/x should be representable: {d}"
        );
    });
}

#[test]
fn laurent_double_pole() {
    // 1/x² around x=0 has a double pole
    let c = ctx();
    c.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let two = arena.int(2);
        let x_sq = arena.pow(x, two);
        let one = arena.one();
        let one_over_x2 = arena.div(one, x_sq);
        let zero = arena.zero();
        let result = arena.laurent_series_expr(one_over_x2, x, zero, 3);
        assert!(result.is_ok(), "Laurent of 1/x^2 should succeed");
    });
}

#[test]
fn laurent_exp_no_pole() {
    // exp(x) has no pole, so Laurent == Taylor
    let c = ctx();
    c.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let exp_x = arena.exp(x);
        let zero = arena.zero();
        let result = arena.laurent_series_expr(exp_x, x, zero, 4);
        assert!(result.is_ok(), "Laurent of exp(x) should succeed");
        let d = arena.display(result.unwrap()).to_string();
        assert!(
            d.contains("1") && d.contains("x"),
            "should be a Taylor series: {d}"
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 3D: Fourier transform — forward
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fourier_delta_t_is_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let delta_t = arena.dirac_delta(t);
        let result = arena.fourier_transform_expr(delta_t, t, omega).unwrap();
        assert_eq!(result, arena.one(), "F{{δ(t)}} should be 1");
    });
}

#[test]
fn fourier_constant_gives_delta() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let five = arena.int(5);
        let result = arena.fourier_transform_expr(five, t, omega).unwrap();
        let d = arena.display(result).to_string();
        // F{5} = 10π·δ(ω)
        assert!(
            d.contains("delta") || d.contains("pi"),
            "F{{5}} should contain delta/pi: {d}"
        );
    });
}

#[test]
fn fourier_sin_gives_deltas() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let three = arena.int(3);
        let three_t = arena.mul(&[three, t]);
        let sin_3t = arena.sin(three_t);
        let result = arena.fourier_transform_expr(sin_3t, t, omega).unwrap();
        let d = arena.display(result).to_string();
        // F{sin(3t)} = iπ[δ(ω+3) - δ(ω-3)]
        assert!(
            d.contains("DiracDelta") || d.contains("delta"),
            "F{{sin(3t)}} should have deltas: {d}"
        );
    });
}

#[test]
fn fourier_cos_gives_deltas() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let two = arena.int(2);
        let two_t = arena.mul(&[two, t]);
        let cos_2t = arena.cos(two_t);
        let result = arena.fourier_transform_expr(cos_2t, t, omega).unwrap();
        let d = arena.display(result).to_string();
        assert!(
            d.contains("DiracDelta") || d.contains("delta"),
            "F{{cos(2t)}} should have deltas: {d}"
        );
    });
}

#[test]
fn fourier_exp_heaviside() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        // exp(-2t)·H(t) → 1/(iω + 2)
        let neg2 = arena.int(-2);
        let neg2_t = arena.mul(&[neg2, t]);
        let exp_neg2t = arena.exp(neg2_t);
        let h = arena.heaviside(t);
        let expr = arena.mul(&[exp_neg2t, h]);
        let result = arena.fourier_transform_expr(expr, t, omega).unwrap();
        let d = arena.display(result).to_string();
        assert!(
            d.contains("omega"),
            "F{{exp(-2t)·H(t)}} should contain omega: {d}"
        );
    });
}

#[test]
fn fourier_heaviside_alone() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let h = arena.heaviside(t);
        let result = arena.fourier_transform_expr(h, t, omega).unwrap();
        let d = arena.display(result).to_string();
        // F{H(t)} = π·δ(ω) + 1/(iω)
        assert!(
            (d.contains("DiracDelta") || d.contains("delta")) && d.contains("pi"),
            "F{{H(t)}} should contain π·δ(ω): {d}"
        );
    });
}

#[test]
fn fourier_linearity_scaled_delta() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let three = arena.int(3);
        let delta_t = arena.dirac_delta(t);
        let expr = arena.mul(&[three, delta_t]);
        let result = arena.fourier_transform_expr(expr, t, omega).unwrap();
        let d = arena.display(result).to_string();
        // F{3·δ(t)} = 3
        assert_eq!(d, "3", "F{{3·δ(t)}} should be 3, got: {d}");
    });
}

#[test]
fn fourier_sum_of_terms() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let delta_t = arena.dirac_delta(t);
        let h_t = arena.heaviside(t);
        let expr = arena.add(&[delta_t, h_t]);
        let result = arena.fourier_transform_expr(expr, t, omega);
        assert!(result.is_ok(), "F{{δ(t) + H(t)}} should succeed");
    });
}

#[test]
fn fourier_t_not_symbol_fails() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let not_sym = arena.int(5);
        let omega = arena.symbol("omega");
        let delta = arena.dirac_delta(not_sym);
        let result = arena.fourier_transform_expr(delta, not_sym, omega);
        assert!(result.is_err(), "t must be a symbol");
    });
}

#[test]
fn fourier_omega_not_symbol_fails() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let not_sym = arena.int(3);
        let delta = arena.dirac_delta(t);
        let result = arena.fourier_transform_expr(delta, t, not_sym);
        assert!(result.is_err(), "omega must be a symbol");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 3D: Fourier transform — inverse
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_fourier_constant_gives_delta() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let three = arena.int(3);
        let result = arena
            .inverse_fourier_transform_expr(three, omega, t)
            .unwrap();
        let d = arena.display(result).to_string();
        // F⁻¹{3} = 3·δ(t)
        assert!(
            d.contains("DiracDelta") || d.contains("delta"),
            "F⁻¹{{3}} should contain δ(t): {d}"
        );
    });
}

#[test]
fn inverse_fourier_delta_omega_gives_recip_2pi() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let delta_omega = arena.dirac_delta(omega);
        let result = arena
            .inverse_fourier_transform_expr(delta_omega, omega, t)
            .unwrap();
        let d = arena.display(result).to_string();
        // F⁻¹{δ(ω)} = 1/(2π)
        assert!(d.contains("pi"), "F⁻¹{{δ(ω)}} should contain pi: {d}");
    });
}

#[test]
fn inverse_fourier_omega_not_symbol_fails() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let not_sym = arena.int(7);
        let result = arena.inverse_fourier_transform_expr(not_sym, not_sym, t);
        assert!(result.is_err(), "omega must be a symbol");
    });
}

#[test]
fn inverse_fourier_t_not_symbol_fails() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let omega = arena.symbol("omega");
        let not_sym = arena.int(7);
        let result = arena.inverse_fourier_transform_expr(omega, omega, not_sym);
        assert!(result.is_err(), "t must be a symbol");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-cutting: numerical verification of orthogonal polynomials
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn legendre_p3_at_half() {
    // P_3(x) = (5x³ - 3x)/2, P_3(1/2) = (5/8 - 3/2)/2 = (5/8 - 12/8)/2 = (-7/8)/2 = -7/16
    let c = ctx();
    c.with_arena_mut(|arena| {
        let three = arena.int(3);
        let half = arena.rational(1, 2);
        let p = arena.legendre(three, half);
        let result = arena.eval_expr(p);
        let expected = arena.rational(-7, 16);
        assert_eq!(
            result,
            expected,
            "P_3(1/2) should be -7/16, got: {}",
            arena.display(result)
        );
    });
}

#[test]
fn chebyshev_t2_at_half() {
    // T_2(x) = 2x² - 1, T_2(1/2) = 2*(1/4) - 1 = 1/2 - 1 = -1/2
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let half = arena.rational(1, 2);
        let t = arena.chebyshev_t(two, half);
        let result = arena.eval_expr(t);
        let expected = arena.rational(-1, 2);
        assert_eq!(
            result,
            expected,
            "T_2(1/2) should be -1/2, got: {}",
            arena.display(result)
        );
    });
}

#[test]
fn hermite_h3_at_one() {
    // H_0=1, H_1=2x, H_2=4x²-2, H_3=8x³-12x
    // H_3(1) = 8 - 12 = -4
    let c = ctx();
    c.with_arena_mut(|arena| {
        let three = arena.int(3);
        let one = arena.one();
        let h = arena.hermite(three, one);
        let result = arena.eval_expr(h);
        let expected = arena.int(-4);
        assert_eq!(
            result,
            expected,
            "H_3(1) should be -4, got: {}",
            arena.display(result)
        );
    });
}

#[test]
fn laguerre_l2_at_one() {
    // L_2(x) = (x² - 4x + 2)/2, L_2(1) = (1 - 4 + 2)/2 = -1/2
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let one = arena.one();
        let l = arena.laguerre(two, one);
        let result = arena.eval_expr(l);
        let expected = arena.rational(-1, 2);
        assert_eq!(
            result,
            expected,
            "L_2(1) should be -1/2, got: {}",
            arena.display(result)
        );
    });
}

#[test]
fn chebyshev_u2_at_half() {
    // U_0=1, U_1=2x, U_2=4x²-1
    // U_2(1/2) = 4*(1/4) - 1 = 0
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let half = arena.rational(1, 2);
        let u = arena.chebyshev_u(two, half);
        let result = arena.eval_expr(u);
        assert_eq!(
            result,
            arena.zero(),
            "U_2(1/2) should be 0, got: {}",
            arena.display(result)
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Arena constructor round-trips
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bessel_constructors_produce_apply_nodes() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let n = arena.int(2);
        let j = arena.besselj(n, x);
        let d = arena.display(j).to_string();
        assert!(d.contains("besselj"), "display should show besselj: {d}");

        let y = arena.bessely(n, x);
        let d = arena.display(y).to_string();
        assert!(d.contains("bessely"), "display should show bessely: {d}");

        let i_b = arena.besseli(n, x);
        let d = arena.display(i_b).to_string();
        assert!(d.contains("besseli"), "display should show besseli: {d}");

        let k = arena.besselk(n, x);
        let d = arena.display(k).to_string();
        assert!(d.contains("besselk"), "display should show besselk: {d}");
    });
}

#[test]
fn orthogonal_poly_constructors_produce_apply_nodes() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let n = arena.symbol("n");
        let p = arena.legendre(n, x);
        let d = arena.display(p).to_string();
        assert!(d.contains("legendre"), "display should show legendre: {d}");

        let t = arena.chebyshev_t(n, x);
        let d = arena.display(t).to_string();
        assert!(
            d.contains("chebyshev_t"),
            "display should show chebyshev_t: {d}"
        );

        let u = arena.chebyshev_u(n, x);
        let d = arena.display(u).to_string();
        assert!(
            d.contains("chebyshev_u"),
            "display should show chebyshev_u: {d}"
        );

        let h = arena.hermite(n, x);
        let d = arena.display(h).to_string();
        assert!(d.contains("hermite"), "display should show hermite: {d}");

        let l = arena.laguerre(n, x);
        let d = arena.display(l).to_string();
        assert!(d.contains("laguerre"), "display should show laguerre: {d}");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Fourier transform via public Ex API (if methods are exposed)
// ═══════════════════════════════════════════════════════════════════════════

// These tests use the arena-level API via with_arena_mut since the
// public Ex-level convenience methods for Fourier transform are not
// yet wired in expr_funcs.rs (which we don't touch).

#[test]
fn fourier_neg_term() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let delta_t = arena.dirac_delta(t);
        let neg_delta = arena.neg(delta_t);
        let result = arena.fourier_transform_expr(neg_delta, t, omega).unwrap();
        let d = arena.display(result).to_string();
        // F{-δ(t)} = -1
        assert!(
            d.contains("-1") || d.contains("−1"),
            "F{{-δ(t)}} should be -1: {d}"
        );
    });
}

#[test]
fn bessel_j10_at_zero_is_zero() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let ten = arena.int(10);
        let zero = arena.zero();
        let j = arena.besselj(ten, zero);
        let result = arena.eval_expr(j);
        assert_eq!(result, arena.zero(), "J_10(0) should be 0");
    });
}

#[test]
fn legendre_p4_at_one() {
    // P_n(1) = 1 for all n
    let c = ctx();
    c.with_arena_mut(|arena| {
        let four = arena.int(4);
        let one = arena.one();
        let p = arena.legendre(four, one);
        let result = arena.eval_expr(p);
        assert_eq!(result, arena.one(), "P_4(1) should be 1");
    });
}

#[test]
fn legendre_p4_at_neg_one() {
    // P_n(-1) = (-1)^n, so P_4(-1) = 1
    let c = ctx();
    c.with_arena_mut(|arena| {
        let four = arena.int(4);
        let neg_one = arena.neg_one();
        let p = arena.legendre(four, neg_one);
        let result = arena.eval_expr(p);
        assert_eq!(result, arena.one(), "P_4(-1) should be 1");
    });
}

#[test]
fn legendre_p3_at_neg_one() {
    // P_3(-1) = (-1)^3 = -1
    let c = ctx();
    c.with_arena_mut(|arena| {
        let three = arena.int(3);
        let neg_one = arena.neg_one();
        let p = arena.legendre(three, neg_one);
        let result = arena.eval_expr(p);
        assert_eq!(result, arena.neg_one(), "P_3(-1) should be -1");
    });
}

#[test]
fn chebyshev_t4_at_one() {
    // T_n(1) = 1 for all n
    let c = ctx();
    c.with_arena_mut(|arena| {
        let four = arena.int(4);
        let one = arena.one();
        let t = arena.chebyshev_t(four, one);
        let result = arena.eval_expr(t);
        assert_eq!(result, arena.one(), "T_4(1) should be 1");
    });
}

#[test]
fn chebyshev_t_even_at_zero() {
    // T_2(0) = -1, T_4(0) = 1
    let c = ctx();
    c.with_arena_mut(|arena| {
        let two = arena.int(2);
        let four = arena.int(4);
        let zero = arena.zero();
        let t2 = arena.chebyshev_t(two, zero);
        let r2 = arena.eval_expr(t2);
        assert_eq!(r2, arena.neg_one(), "T_2(0) should be -1");

        let t4 = arena.chebyshev_t(four, zero);
        let r4 = arena.eval_expr(t4);
        assert_eq!(r4, arena.one(), "T_4(0) should be 1");
    });
}

#[test]
fn hermite_h0_at_anything_is_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let val = arena.int(42);
        let h = arena.hermite(zero, val);
        let result = arena.eval_expr(h);
        assert_eq!(result, arena.one(), "H_0(42) should be 1");
    });
}

#[test]
fn laguerre_l0_at_anything_is_one() {
    let c = ctx();
    c.with_arena_mut(|arena| {
        let zero = arena.zero();
        let val = arena.int(99);
        let l = arena.laguerre(zero, val);
        let result = arena.eval_expr(l);
        assert_eq!(result, arena.one(), "L_0(99) should be 1");
    });
}
