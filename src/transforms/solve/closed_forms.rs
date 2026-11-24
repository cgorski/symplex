//! Linear solver: a*x + b = 0 → x = -b/a (part of [`solve`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Linear solver: a*x + b = 0 → x = -b/a
// ═══════════════════════════════════════════════════════════════════════════

pub(super) fn solve_linear(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(1); // coefficient of x
    let b = poly.coeff(0); // constant term

    if a.is_zero() {
        return Vec::new(); // Degenerate: 0*x + b = 0.
    }

    // x = -b / a
    let value = -b / a;
    let value_id = arena.num_ratio(value.clone());

    vec![Solution { value: value_id }]
}

// ═══════════════════════════════════════════════════════════════════════════
// Quadratic solver: a*x² + b*x + c = 0
// ═══════════════════════════════════════════════════════════════════════════

pub(super) fn solve_quadratic(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(2);
    let b = poly.coeff(1);
    let c = poly.coeff(0);

    if a.is_zero() {
        // Actually linear.
        let linear = Poly::from_coeffs(vec![c, b]);
        return solve_linear(arena, &linear);
    }

    // Discriminant: b² - 4ac
    let discriminant = &b * &b - Ratio::from_integer(BigInt::from(4)) * &a * &c;

    if discriminant.is_zero() {
        // Double root: x = -b / (2a)
        let two_a = Ratio::from_integer(BigInt::from(2)) * &a;
        let value = -b / two_a;
        let value_id = arena.num_ratio(value.clone());
        return vec![Solution { value: value_id }];
    }

    if discriminant.is_negative() {
        // Complex roots: x = (-b ± i√|Δ|) / (2a)
        let abs_disc = -discriminant;
        let neg_b = arena.num_ratio((-&b).clone());
        let abs_disc_id = arena.num_ratio(abs_disc.clone());

        // Check if |Δ| is a perfect square
        let sqrt_abs_disc = if let Some(s) = rational_sqrt(&abs_disc) {
            arena.num_ratio(s.clone())
        } else {
            arena.sqrt(abs_disc_id)
        };

        let i_sqrt = arena.mul(&[arena.i_unit, sqrt_abs_disc]);
        let two_a_val = Ratio::from_integer(BigInt::from(2)) * &a;
        let two_a_id = arena.num_ratio(two_a_val.clone());
        let two_a_inv = {
            let neg_one = arena.neg_one;
            arena.pow(two_a_id, neg_one)
        };

        // x1 = (-b + i√|Δ|) / (2a)
        let sum1 = arena.add(&[neg_b, i_sqrt]);
        let x1 = arena.mul(&[sum1, two_a_inv]);

        // x2 = (-b - i√|Δ|) / (2a)
        let neg_i_sqrt = arena.neg(i_sqrt);
        let sum2 = arena.add(&[neg_b, neg_i_sqrt]);
        let x2 = arena.mul(&[sum2, two_a_inv]);

        return vec![Solution { value: x1 }, Solution { value: x2 }];
    }

    // Check if the discriminant is a perfect square (rational root).
    if let Some(sqrt_disc) = rational_sqrt(&discriminant) {
        let two_a = Ratio::from_integer(BigInt::from(2)) * &a;

        // x = (-b ± √Δ) / (2a)
        let x1 = (-&b + &sqrt_disc) / &two_a;
        let x2 = (-&b - &sqrt_disc) / &two_a;

        let x1_id = arena.num_ratio(x1.clone());
        let x2_id = arena.num_ratio(x2.clone());

        if x1 == x2 {
            vec![Solution { value: x1_id }]
        } else {
            vec![Solution { value: x1_id }, Solution { value: x2_id }]
        }
    } else {
        // Discriminant is not a perfect square — roots are irrational.
        // Express as (-b ± sqrt(discriminant)) / (2*a) symbolically.
        let neg_b = {
            let neg_b_val = -&b;
            arena.num_ratio(neg_b_val.clone())
        };
        let sqrt_disc = {
            let disc_id = arena.num_ratio(discriminant.clone());
            arena.sqrt(disc_id)
        };
        let two_a_val = Ratio::from_integer(BigInt::from(2)) * &a;
        let two_a_id = arena.num_ratio(two_a_val.clone());
        let two_a_inv = {
            let neg_one = arena.neg_one;
            arena.pow(two_a_id, neg_one)
        };

        // x1 = (-b + sqrt(disc)) / (2a)
        let sum1 = arena.add(&[neg_b, sqrt_disc]);
        let x1 = arena.mul(&[sum1, two_a_inv]);

        // x2 = (-b - sqrt(disc)) / (2a)
        let neg_sqrt = arena.neg(sqrt_disc);
        let sum2 = arena.add(&[neg_b, neg_sqrt]);
        let x2 = arena.mul(&[sum2, two_a_inv]);

        vec![Solution { value: x1 }, Solution { value: x2 }]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cubic solver: a*x³ + b*x² + c*x + d = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a cubic polynomial. Tries rational roots first, then falls back
/// to Cardano's formula for irrational / complex roots.
pub(super) fn solve_cubic(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(3);
    if a.is_zero() {
        let quadratic = Poly::from_coeffs(vec![poly.coeff(0), poly.coeff(1), poly.coeff(2)]);
        return solve_quadratic(arena, &quadratic);
    }

    // Try rational roots first — exact answers are preferable.
    let rational_attempt = solve_rational_roots(arena, var, poly);
    if !rational_attempt.is_empty() {
        return rational_attempt;
    }

    // No rational roots — use Cardano's formula.
    solve_cubic_cardano(arena, poly)
}

/// Pure Cardano's formula (no rational-root attempt) to avoid re-entry loops.
fn solve_cubic_cardano(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(3);
    let b = poly.coeff(2);
    let c = poly.coeff(1);
    let d = poly.coeff(0);

    if a.is_zero() {
        let quadratic = Poly::from_coeffs(vec![d, c, b]);
        return solve_quadratic(arena, &quadratic);
    }

    // Binomial a·x³ + d = 0: cleaner roots-of-unity form than Cardano.
    if let Some(roots) = try_solve_binomial_rational(arena, poly) {
        return roots;
    }

    let a_id = arena.num_ratio(a.clone());
    let b_id = arena.num_ratio(b.clone());
    let c_id = arena.num_ratio(c.clone());
    let d_id = arena.num_ratio(d.clone());

    let three = arena.int(3);
    let nine = arena.int(9);
    let twenty_seven = arena.int(27);
    let two = arena.int(2);
    let four = arena.int(4);

    // Depress to t³ + pt + q = 0  via  x = t - b/(3a)
    //   p = (3ac - b²) / (3a²)
    //   q = (2b³ - 9abc + 27a²d) / (27a³)

    // p_num = 3ac - b²
    let tmp1 = arena.mul(&[three, a_id, c_id]);
    let tmp2 = arena.mul(&[b_id, b_id]);
    let p_num = arena.sub(tmp1, tmp2);
    let p_den = arena.mul(&[three, a_id, a_id]);
    let p = arena.div(p_num, p_den);

    // q_num = 2b³ - 9abc + 27a²d
    let tmp3 = arena.mul(&[two, b_id, b_id, b_id]);
    let tmp4 = arena.mul(&[nine, a_id, b_id, c_id]);
    let tmp4n = arena.neg(tmp4);
    let tmp5 = arena.mul(&[twenty_seven, a_id, a_id, d_id]);
    let q_num = arena.add(&[tmp3, tmp4n, tmp5]);
    let q_den = arena.mul(&[twenty_seven, a_id, a_id, a_id]);
    let q = arena.div(q_num, q_den);

    // Discriminant: Δ = q²/4 + p³/27
    let qq = arena.mul(&[q, q]);
    let qq_over4 = arena.div(qq, four);
    let ppp = arena.mul(&[p, p, p]);
    let ppp_over27 = arena.div(ppp, twenty_seven);
    let disc = arena.add(&[qq_over4, ppp_over27]);

    // √Δ
    let half = arena.rational(1, 2);
    let sqrt_disc = arena.pow(disc, half);

    // Cardano: t = cbrt(-q/2 + √Δ) + cbrt(-q/2 - √Δ)
    let neg_q = arena.neg(q);
    let neg_q_half = arena.div(neg_q, two);

    let u_arg = arena.add(&[neg_q_half, sqrt_disc]);
    let v_arg = arena.sub(neg_q_half, sqrt_disc);

    // Cardano's formula needs the *real* cube roots of the two radicands
    // (their product must be −p/3).  A `Pow(negative, 1/3)` node is
    // evaluated on the principal complex branch by `evalf`, which would
    // silently produce wrong roots, so a radicand that is provably
    // negative is written as −cbrt(|radicand|).  The signs follow from the
    // exact rational data: for Δ ≥ 0, √Δ ≥ |q|/2 exactly when p ≥ 0.
    let (p_rat, q_rat, disc_rat) = {
        let three_r = Ratio::from_integer(BigInt::from(3));
        let nine_r = Ratio::from_integer(BigInt::from(9));
        let two_r = Ratio::from_integer(BigInt::from(2));
        let four_r = Ratio::from_integer(BigInt::from(4));
        let twenty_seven_r = Ratio::from_integer(BigInt::from(27));
        let p_r = (&three_r * &a * &c - &b * &b) / (&three_r * &a * &a);
        let q_r = (&two_r * &b * &b * &b - &nine_r * &a * &b * &c + &twenty_seven_r * &a * &a * &d)
            / (&twenty_seven_r * &a * &a * &a);
        let disc_r = &q_r * &q_r / &four_r + &p_r * &p_r * &p_r / &twenty_seven_r;
        (p_r, q_r, disc_r)
    };
    let (u_sign, v_sign) = cardano_radicand_signs(&p_rat, &q_rat, &disc_rat);

    let u = real_cbrt_with_sign(arena, u_arg, u_sign);
    let v = real_cbrt_with_sign(arena, v_arg, v_sign);

    let t1 = arena.add(&[u, v]);

    // Shift back: x = t - b/(3a)
    let three_a = arena.mul(&[three, a_id]);
    let shift = arena.div(b_id, three_a);
    let x1 = arena.sub(t1, shift);

    // Other two roots via cube roots of unity:
    //   ω  = (-1 + i√3)/2
    //   ω² = (-1 - i√3)/2
    let omega_re = arena.rational(-1, 2);
    let sqrt3 = arena.pow(three, half);
    let sqrt3_half = arena.div(sqrt3, two);
    let i_unit = arena.i_unit;
    let omega_im = arena.mul(&[sqrt3_half, i_unit]);
    let omega = arena.add(&[omega_re, omega_im]);
    let omega2 = arena.sub(omega_re, omega_im);

    let ou = arena.mul(&[omega, u]);
    let o2v = arena.mul(&[omega2, v]);
    let t2 = arena.add(&[ou, o2v]);
    let o2u = arena.mul(&[omega2, u]);
    let ov = arena.mul(&[omega, v]);
    let t3 = arena.add(&[o2u, ov]);

    let x2 = arena.sub(t2, shift);
    let x3 = arena.sub(t3, shift);

    // Simplify all roots through eval
    let x1s = crate::transforms::eval::eval(arena, x1);
    let x2s = crate::transforms::eval::eval(arena, x2);
    let x3s = crate::transforms::eval::eval(arena, x3);

    vec![
        Solution { value: x1s },
        Solution { value: x2s },
        Solution { value: x3s },
    ]
}

/// Signs of the Cardano radicands `−q/2 + √Δ` and `−q/2 − √Δ` for the
/// depressed cubic `t³ + pt + q` with `Δ = q²/4 + p³/27`.
///
/// Returns `(sign_u, sign_v)` with values in `{-1, 0, 1}`; `0` is also used
/// when `Δ < 0` (complex radicands, principal branch is correct there).
fn cardano_radicand_signs(p: &Q, q: &Q, disc: &Q) -> (i8, i8) {
    use std::cmp::Ordering;
    if disc.is_negative() {
        return (0, 0);
    }
    let neg_q_sign: i8 = match q.cmp(&Ratio::zero()) {
        Ordering::Less => 1,
        Ordering::Equal => 0,
        Ordering::Greater => -1,
    };
    if disc.is_zero() {
        // Both radicands equal −q/2.
        return (neg_q_sign, neg_q_sign);
    }
    // Δ > 0: √Δ > |q|/2 ⇔ p > 0; √Δ = |q|/2 ⇔ p = 0; √Δ < |q|/2 ⇔ p < 0.
    let p_sign = match p.cmp(&Ratio::zero()) {
        Ordering::Less => -1i8,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    };
    let u_sign = if neg_q_sign >= 0 {
        // −q/2 ≥ 0 and √Δ > 0 ⇒ positive.
        1
    } else {
        // −q/2 < 0: sign decided by whether √Δ exceeds |q|/2.
        p_sign
    };
    let v_sign = if neg_q_sign <= 0 {
        -1
    } else {
        // −q/2 > 0: −q/2 − √Δ is positive iff √Δ < q/2 ⇔ p < 0.
        -p_sign
    };
    (u_sign, v_sign)
}

/// Build the real cube root of `arg` given its known sign: `cbrt(arg)` for
/// a positive radicand, `−cbrt(−arg)` for a negative one (so that no cube
/// root of a negative real is ever emitted), `0` for a zero radicand.  An
/// unknown sign (`0` for a complex radicand) falls back to the principal
/// branch `arg^(1/3)`.
fn real_cbrt_with_sign(arena: &mut Arena, arg: ExprId, sign: i8) -> ExprId {
    let third = arena.rational(1, 3);
    match sign {
        1 => arena.pow(arg, third),
        -1 => {
            let neg_arg = arena.neg(arg);
            let neg_arg = crate::transforms::eval::eval(arena, neg_arg);
            let root = arena.pow(neg_arg, third);
            arena.neg(root)
        }
        _ => {
            // Zero radicand (exactly), or complex radicand (principal branch).
            let ev = crate::transforms::eval::eval(arena, arg);
            if arena.is_zero_structural(ev) {
                return arena.zero;
            }
            arena.pow(arg, third)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Quartic solver: Ferrari's method  a*x⁴ + b*x³ + c*x² + d*x + e = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a quartic polynomial. Tries rational roots first, then falls back
/// to Ferrari's method.
pub(super) fn solve_quartic(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(4);
    if a.is_zero() {
        let cubic = Poly::from_coeffs(vec![
            poly.coeff(0),
            poly.coeff(1),
            poly.coeff(2),
            poly.coeff(3),
        ]);
        return solve_cubic(arena, var, &cubic);
    }

    // Try rational roots first.
    let rational_attempt = solve_rational_roots(arena, var, poly);
    if !rational_attempt.is_empty() {
        return rational_attempt;
    }

    // No rational roots — use Ferrari's method.
    solve_quartic_ferrari(arena, poly)
}

/// Pure Ferrari's method (no rational-root attempt) to avoid re-entry loops.
fn solve_quartic_ferrari(arena: &mut Arena, poly: &Poly) -> Vec<Solution> {
    let a = poly.coeff(4);
    let b = poly.coeff(3);
    let c = poly.coeff(2);
    let d = poly.coeff(1);
    let e = poly.coeff(0);

    if a.is_zero() {
        let cubic = Poly::from_coeffs(vec![e.clone(), d, c, b]);
        return solve_cubic_cardano(arena, &cubic);
    }

    // Binomial a·x⁴ + e = 0: cleaner roots-of-unity form than Ferrari.
    if let Some(roots) = try_solve_binomial_rational(arena, poly) {
        return roots;
    }

    // Depress to t⁴ + pt² + qt + r = 0  via  x = t - b/(4a)
    //   p = (8ac - 3b²) / (8a²)
    //   q = (b³ - 4abc + 8a²d) / (8a³)
    //   r = (-3b⁴ + 256a³e - 64a²bd + 16ab²c) / (256a⁴)

    // Exact rational depressed coefficients (also used for the biquadratic
    // shortcut below and for the resolvent-root filter).
    let (p_rat, q_rat, r_rat) = {
        let r3 = Ratio::from_integer(BigInt::from(3));
        let r4 = Ratio::from_integer(BigInt::from(4));
        let r8 = Ratio::from_integer(BigInt::from(8));
        let r16 = Ratio::from_integer(BigInt::from(16));
        let r64 = Ratio::from_integer(BigInt::from(64));
        let r256 = Ratio::from_integer(BigInt::from(256));
        let a2 = &a * &a;
        let a3 = &a2 * &a;
        let a4 = &a3 * &a;
        let b2 = &b * &b;
        let p_r = (&r8 * &a * &c - &r3 * &b2) / (&r8 * &a2);
        let q_r = (&b2 * &b - &r4 * &a * &b * &c + &r8 * &a2 * &d) / (&r8 * &a3);
        let r_r = (-&r3 * &b2 * &b2 + &r256 * &a3 * &e - &r64 * &a2 * &b * &d
            + &r16 * &a * &b2 * &c)
            / (&r256 * &a4);
        (p_r, q_r, r_r)
    };

    // Biquadratic t⁴ + pt² + r = 0 (q = 0): Ferrari's factorisation
    // degenerates (k = √(2m − p) = 0 for the rational resolvent root
    // m = p/2, giving 0/0), so solve the quadratic in s = t² and take
    // t = ±√s instead.
    if q_rat.is_zero() {
        let quad = Poly::from_coeffs(vec![r_rat.clone(), p_rat.clone(), Ratio::one()]);
        let s_roots = solve_quadratic(arena, &quad);
        let four_a = Ratio::from_integer(BigInt::from(4)) * &a;
        let shift_rat = &b / &four_a;
        let shift = arena.num_ratio(shift_rat.clone());
        let half = arena.rational(1, 2);
        let mut out: Vec<Solution> = Vec::new();
        for s in s_roots {
            let s_ev = crate::transforms::eval::eval(arena, s.value);
            let t_pos = arena.pow(s_ev, half);
            let t_neg = arena.neg(t_pos);
            for t in [t_pos, t_neg] {
                let x = arena.sub(t, shift);
                let x = crate::transforms::eval::eval(arena, x);
                if !out.iter().any(|o| o.value == x) {
                    out.push(Solution { value: x });
                }
            }
        }
        return out;
    }

    let a_id = arena.num_ratio(a.clone());
    let b_id = arena.num_ratio(b.clone());
    let c_id = arena.num_ratio(c.clone());
    let d_id = arena.num_ratio(d.clone());
    let e_id = arena.num_ratio(e.clone());

    let two = arena.int(2);
    let three = arena.int(3);
    let four = arena.int(4);
    let eight = arena.int(8);
    let sixteen = arena.int(16);
    let sixty_four = arena.int(64);
    let two_fifty_six = arena.int(256);

    // p = (8ac - 3b²) / (8a²)
    let tmp_8ac = arena.mul(&[eight, a_id, c_id]);
    let tmp_3bb = arena.mul(&[three, b_id, b_id]);
    let p_num = arena.sub(tmp_8ac, tmp_3bb);
    let p_den = arena.mul(&[eight, a_id, a_id]);
    let p = arena.div(p_num, p_den);

    // q = (b³ - 4abc + 8a²d) / (8a³)
    let tmp_bbb = arena.mul(&[b_id, b_id, b_id]);
    let tmp_4abc = arena.mul(&[four, a_id, b_id, c_id]);
    let tmp_4abc_n = arena.neg(tmp_4abc);
    let tmp_8aad = arena.mul(&[eight, a_id, a_id, d_id]);
    let q_num = arena.add(&[tmp_bbb, tmp_4abc_n, tmp_8aad]);
    let q_den = arena.mul(&[eight, a_id, a_id, a_id]);
    let q = arena.div(q_num, q_den);

    // r = (-3b⁴ + 256a³e - 64a²bd + 16ab²c) / (256a⁴)
    let tmp_3b4 = arena.mul(&[three, b_id, b_id, b_id, b_id]);
    let tmp_3b4_n = arena.neg(tmp_3b4);
    let tmp_256a3e = arena.mul(&[two_fifty_six, a_id, a_id, a_id, e_id]);
    let tmp_64a2bd = arena.mul(&[sixty_four, a_id, a_id, b_id, d_id]);
    let tmp_64a2bd_n = arena.neg(tmp_64a2bd);
    let tmp_16ab2c = arena.mul(&[sixteen, a_id, b_id, b_id, c_id]);
    let r_num = arena.add(&[tmp_3b4_n, tmp_256a3e, tmp_64a2bd_n, tmp_16ab2c]);
    let r_den = arena.mul(&[two_fifty_six, a_id, a_id, a_id, a_id]);
    let r = arena.div(r_num, r_den);

    // Resolvent cubic:  8m³ - 4pm² - 8rm + (4pr - q²) = 0
    // We need the coefficients as rationals to build a Poly.
    let resolvent_c3 = eight;
    let neg_four = arena.neg(four);
    let resolvent_c2 = arena.mul(&[neg_four, p]);
    let neg_eight = arena.neg(eight);
    let resolvent_c1 = arena.mul(&[neg_eight, r]);
    let tmp_4pr = arena.mul(&[four, p, r]);
    let tmp_qq = arena.mul(&[q, q]);
    let tmp_qq_n = arena.neg(tmp_qq);
    let resolvent_c0 = arena.add(&[tmp_4pr, tmp_qq_n]);

    let resolvent_c3_e = crate::transforms::eval::eval(arena, resolvent_c3);
    let resolvent_c2_e = crate::transforms::eval::eval(arena, resolvent_c2);
    let resolvent_c1_e = crate::transforms::eval::eval(arena, resolvent_c1);
    let resolvent_c0_e = crate::transforms::eval::eval(arena, resolvent_c0);

    let rc3 = match arena.as_num(resolvent_c3_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };
    let rc2 = match arena.as_num(resolvent_c2_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };
    let rc1 = match arena.as_num(resolvent_c1_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };
    let rc0 = match arena.as_num(resolvent_c0_e) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };

    let resolvent_poly = Poly::from_coeffs(vec![rc0, rc1, rc2, rc3]);

    // We need 2m - p ≠ 0 for a non-degenerate Ferrari factorization; with
    // q ≠ 0 (guaranteed above) m = p/2 is never a resolvent root.
    debug_assert!(!q_rat.is_zero());

    // Try rational roots of the resolvent cubic first.
    // This avoids the *casus irreducibilis* problem where Cardano's formula
    // produces complex cube roots for real rational roots, yielding
    // unsimplifiable expressions like cbrt(±i·√(1/27)).
    let m = if let Some(m_rat) = find_preferred_resolvent_root(&resolvent_poly, &p_rat) {
        arena.num_ratio(m_rat.clone())
    } else {
        let m_solutions = solve_cubic_cardano(arena, &resolvent_poly);
        if m_solutions.is_empty() {
            return Vec::new();
        }
        m_solutions[0].value
    };

    // Factor into two quadratics via √(2m − p):
    //   t² + k·t + (m − q/(2k)) = 0
    //   t² − k·t + (m + q/(2k)) = 0
    // where k = √(2m − p)
    let half = arena.rational(1, 2);
    let two_m = arena.mul(&[two, m]);
    let two_m_minus_p = arena.sub(two_m, p);
    let k = arena.pow(two_m_minus_p, half);

    let two_k = arena.mul(&[two, k]);
    let q_over_2k = arena.div(q, two_k);

    // shift = b/(4a)
    let four_a = arena.mul(&[four, a_id]);
    let shift = arena.div(b_id, four_a);

    // Quadratic 1:  t² + kt + (m - q/(2k)) = 0
    let s1 = arena.sub(m, q_over_2k);
    let kk = arena.mul(&[k, k]);
    let four_s1 = arena.mul(&[four, s1]);
    let disc1 = arena.sub(kk, four_s1);
    let sqrt_disc1 = arena.pow(disc1, half);
    let neg_k = arena.neg(k);

    let sum1a = arena.add(&[neg_k, sqrt_disc1]);
    let t1a = arena.div(sum1a, two);
    let diff1b = arena.sub(neg_k, sqrt_disc1);
    let t1b = arena.div(diff1b, two);

    let x1 = arena.sub(t1a, shift);
    let x2 = arena.sub(t1b, shift);

    // Quadratic 2:  t² - kt + (m + q/(2k)) = 0
    let s2 = arena.add(&[m, q_over_2k]);
    let kk2 = arena.mul(&[k, k]);
    let four_s2 = arena.mul(&[four, s2]);
    let disc2 = arena.sub(kk2, four_s2);
    let sqrt_disc2 = arena.pow(disc2, half);

    let sum2a = arena.add(&[k, sqrt_disc2]);
    let t2a = arena.div(sum2a, two);
    let diff2b = arena.sub(k, sqrt_disc2);
    let t2b = arena.div(diff2b, two);

    let x3 = arena.sub(t2a, shift);
    let x4 = arena.sub(t2b, shift);

    // Simplify all roots
    let x1s = crate::transforms::eval::eval(arena, x1);
    let x2s = crate::transforms::eval::eval(arena, x2);
    let x3s = crate::transforms::eval::eval(arena, x3);
    let x4s = crate::transforms::eval::eval(arena, x4);

    vec![
        Solution { value: x1s },
        Solution { value: x2s },
        Solution { value: x3s },
        Solution { value: x4s },
    ]
}

/// Find a rational root of the resolvent cubic that yields a non-degenerate
/// Ferrari factorization (i.e. `2m − p ≠ 0`, so that `k = √(2m−p) ≠ 0`).
///
/// Uses the Rational Root Theorem: for a polynomial with integer coefficients,
/// every rational root `p/q` satisfies `p | a₀` and `q | aₙ`; when that
/// search finds no non-degenerate root (it enumerates no divisors of a
/// coefficient above 10⁶), the linear factors of the resolvent over ℤ.
///
/// Returns `None` if no rational root is found (Cardano fallback will be used).
fn find_preferred_resolvent_root(resolvent: &Poly, p_rat: &Q) -> Option<Q> {
    let found = rational_resolvent_root_by_divisors(resolvent, p_rat);
    if let Some((m, true)) = found {
        return Some(m);
    }
    // `divisors` gives up above 10⁶ (it returns `{1, n}`), so a resolvent
    // with large coefficients kept its rational root hidden and Cardano's
    // formula took over: the Rothstein–Trager polynomial `7168t⁴ + 7168t³ +
    // 96t² + 16t + 1` of `∫ x³/(x⁴ + 4x³ − 1) dx` (resolvent root `−9/224`)
    // had its roots in cube roots of `81/44957696·√21 ± …` instead of in
    // `ℚ(√2)` (up to 0.46).  Its linear factors over ℤ are its rational
    // roots.
    let two = Ratio::from_integer(BigInt::from(2));
    let (_, factors) = resolvent.factor_over_z();
    let mut fallback = found.map(|(m, _)| m);
    for (f, _) in &factors {
        if f.degree() == Some(1) {
            let m = -f.coeff(0) / f.coeff(1);
            if &two * &m != *p_rat {
                return Some(m);
            }
            fallback.get_or_insert(m);
        }
    }
    fallback
}

/// The rational-root search of [`find_preferred_resolvent_root`] over the
/// divisors of the end coefficients: a root and whether it is
/// non-degenerate (`2m ≠ p`).
fn rational_resolvent_root_by_divisors(resolvent: &Poly, p_rat: &Q) -> Option<(Q, bool)> {
    let (int_poly, _scale) = clear_denominators(resolvent);
    let a0 = int_poly.coeff(0).to_integer();
    let an = {
        let v = int_poly.leading_coeff()?;
        v.to_integer()
    };

    let two_r = Ratio::from_integer(BigInt::from(2));
    let mut fallback: Option<Q> = None;

    if a0.is_zero() {
        // m = 0 is a root.  Record it but keep looking for a non-degenerate one.
        let zero = Ratio::zero();
        if &two_r * &zero - p_rat != Ratio::zero() {
            return Some((zero, true));
        }
        fallback = Some(zero);

        // Divide out m and check the remaining quadratic for rational roots.
        let reduced_coeffs: Vec<Q> = resolvent.coeffs().iter().skip(1).cloned().collect();
        let reduced = Poly::from_coeffs(reduced_coeffs);
        let aq = reduced.coeff(2);
        let bq = reduced.coeff(1);
        let cq = reduced.coeff(0);
        if !aq.is_zero() {
            let disc = &bq * &bq - Ratio::from_integer(BigInt::from(4)) * &aq * &cq;
            if let Some(sqrt_d) = rational_sqrt(&disc) {
                let two_aq = Ratio::from_integer(BigInt::from(2)) * &aq;
                for candidate in [(-&bq + &sqrt_d) / &two_aq, (-&bq - &sqrt_d) / &two_aq] {
                    if &two_r * &candidate - p_rat != Ratio::zero() {
                        return Some((candidate, true));
                    }
                    if fallback.is_none() {
                        fallback = Some(candidate);
                    }
                }
            }
        }
    } else {
        let divs_a0 = divisors(&a0.abs());
        let divs_an = divisors(&an.abs());

        for p_div in &divs_a0 {
            for q_div in &divs_an {
                for &sign in &[1i64, -1i64] {
                    let candidate = Ratio::new(p_div * BigInt::from(sign), q_div.clone());
                    if resolvent.eval(&candidate).is_zero() {
                        // Prefer a root where 2m - p ≠ 0.
                        if &two_r * &candidate - p_rat != Ratio::zero() {
                            return Some((candidate, true));
                        }
                        if fallback.is_none() {
                            fallback = Some(candidate);
                        }
                    }
                }
            }
        }
    }

    fallback.map(|m| (m, false))
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact factorisation over ℤ for higher-degree polynomials
// ═══════════════════════════════════════════════════════════════════════════

/// The roots of a polynomial over ℚ through its exact factorisation over
/// ℤ ([`Poly::factor_over_z`]: content, Yun's square-free decomposition,
/// Berlekamp–Zassenhaus).
///
/// The content is dropped first, so `c·p` has exactly the roots of `p` for
/// every rational `c ≠ 0`, and every irreducible factor is solved once,
/// whatever its multiplicity: a linear factor *is* a rational root, a
/// quadratic, cubic or quartic factor goes to the radical formulas, and a
/// `RootOf(f, k)` placeholder is emitted only for an irreducible `f` of
/// degree ≥ 5, with `k` running over the roots of *that factor* — never
/// over a reducible or non-square-free polynomial.  The result lists the
/// **distinct** roots.
///
/// This replaced a Rational Root Theorem search whose divisor enumeration
/// gave up (silently, returning `{1, |n|}`) for a coefficient above `10⁶`,
/// so that `4·p(x)` lost the rational roots `p(x)` had, and which divided a
/// found root out once only, leaving its multiplicity in the cofactor
/// handed to `RootOf`.
///
/// Used for degree ≥ 5 and as the first pass of the cubic and quartic
/// solvers.  A binomial `a·xⁿ + b` of degree ≥ 5 keeps its roots-of-unity
/// form (before factoring, so `x⁵ − 32` is not split into a linear and a
/// quartic factor).
pub(super) fn solve_rational_roots(arena: &mut Arena, var: ExprId, poly: &Poly) -> Vec<Solution> {
    let degree = match poly.degree() {
        Some(d) if d >= 1 => d,
        _ => return Vec::new(),
    };

    // Binomial a·xⁿ + b = 0 (n ≥ 5): all n roots explicitly via roots of
    // unity.  Cubics/quartics get here via their own solvers, which fall
    // back to the binomial form only after rational-root extraction.
    if degree > 4
        && let Some(roots) = try_solve_binomial_rational(arena, poly)
    {
        return roots;
    }

    let (_content, factors) = poly.factor_over_z();
    let mut roots: Vec<Solution> = Vec::new();
    for (factor, _multiplicity) in &factors {
        let found = match factor.degree() {
            None | Some(0) => Vec::new(),
            Some(1) => solve_linear(arena, factor),
            Some(2) => solve_quadratic(arena, factor),
            // Irreducible over ℚ: no rational root to extract, straight to
            // the radical formulas (which are invariant under the scaling
            // that made the factor primitive).
            Some(3) => solve_cubic_cardano(arena, factor),
            Some(4) => solve_quartic_ferrari(arena, factor),
            Some(d) => {
                if let Some(explicit) = try_solve_binomial_rational(arena, factor) {
                    explicit
                } else {
                    let factor_expr = polybridge::poly_to_expr(arena, factor, var);
                    (0..d)
                        .map(|i| {
                            let idx = arena.int(i as i64);
                            Solution {
                                value: arena.intern(ExprNode::RootOf(factor_expr, var, idx)),
                            }
                        })
                        .collect()
                }
            }
        };
        for sol in found {
            if !roots.iter().any(|s| s.value == sol.value) {
                roots.push(sol);
            }
        }
    }

    tracing::debug!(
        factors = factors.len(),
        roots = roots.len(),
        "roots via factorisation over ℤ"
    );
    roots
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Try to compute the exact square root of a non-negative rational.
///
/// Returns `Some(√r)` if `r` is a perfect square (both numerator and
/// denominator are perfect squares), or `None` otherwise.
pub(super) fn rational_sqrt(r: &Q) -> Option<Q> {
    if r.is_negative() {
        return None;
    }
    if r.is_zero() {
        return Some(Ratio::zero());
    }

    let n = r.numer().abs();
    let d = r.denom().abs();

    let sqrt_n = n.sqrt();
    let sqrt_d = d.sqrt();

    if &sqrt_n * &sqrt_n == n && &sqrt_d * &sqrt_d == d {
        Some(Ratio::new(sqrt_n, sqrt_d))
    } else {
        None
    }
}

/// Clear denominators of a polynomial's coefficients.
///
/// Returns the integer-coefficient polynomial and the scale factor.
pub(super) fn clear_denominators(poly: &Poly) -> (Poly, BigInt) {
    if poly.is_zero() {
        return (Poly::zero(), BigInt::one());
    }

    // Compute LCM of all denominators.
    let mut lcm = BigInt::one();
    for c in poly.coeffs() {
        let d = c.denom().abs();
        lcm = num_integer::lcm(lcm, d);
    }

    // Multiply all coefficients by the LCM.
    let scale = Ratio::from_integer(lcm.clone());
    let scaled = poly.scale(&scale);

    (scaled, lcm)
}

/// Compute all positive divisors of `n` (a positive BigInt).
///
/// Returns them in ascending order.  For n=0, returns `[1]` as a
/// fallback.
pub(super) fn divisors(n: &BigInt) -> Vec<BigInt> {
    if n.is_zero() {
        return vec![BigInt::one()];
    }

    let n_abs = n.abs();

    // For small numbers, brute-force trial division.
    // For large numbers, this would be slow — but our polynomials
    // typically have small coefficients.
    let limit: u64 = match (&n_abs).try_into() {
        Ok(v) if v <= 1_000_000u64 => v,
        _ => {
            // Very large coefficient — just return 1 and n.
            return vec![BigInt::one(), n_abs];
        }
    };

    let mut result = Vec::new();
    let mut i = 1u64;
    while i * i <= limit {
        if limit.is_multiple_of(i) {
            result.push(BigInt::from(i));
            if i * i != limit {
                result.push(BigInt::from(limit / i));
            }
        }
        i += 1;
    }

    result.sort();
    result
}
