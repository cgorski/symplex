//! Known-value table (part of [`definite`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Known-value table
// ═══════════════════════════════════════════════════════════════════════════

/// Canonical interval shapes recognised by the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IntervalShape {
    ZeroInf,
    FullLine,
    ZeroOne,
    ZeroHalfPi,
    ZeroPi,
    ZeroTwoPi,
    NegPiPi,
    Other,
}

pub(super) fn classify_interval(arena: &mut Arena, a: ExprId, b: ExprId) -> IntervalShape {
    let zero = arena.zero();
    let one = arena.one();
    let pi = arena.pi();
    let half = arena.rational(1, 2);
    let half_pi = arena.mul(&[half, pi]);
    let two = arena.int(2);
    let two_pi = arena.mul(&[two, pi]);
    let neg_pi = arena.neg(pi);
    let same =
        |arena: &mut Arena, p: ExprId, q: ExprId| cmp_points(arena, p, q) == Some(Ordering::Equal);

    if real_infinity(arena, a) == Some(Sign::Negative)
        && real_infinity(arena, b) == Some(Sign::Positive)
    {
        return IntervalShape::FullLine;
    }
    if same(arena, a, zero) {
        if real_infinity(arena, b) == Some(Sign::Positive) {
            return IntervalShape::ZeroInf;
        }
        if same(arena, b, one) {
            return IntervalShape::ZeroOne;
        }
        if same(arena, b, half_pi) {
            return IntervalShape::ZeroHalfPi;
        }
        if same(arena, b, pi) {
            return IntervalShape::ZeroPi;
        }
        if same(arena, b, two_pi) {
            return IntervalShape::ZeroTwoPi;
        }
    }
    if same(arena, a, neg_pi) && same(arena, b, pi) {
        return IntervalShape::NegPiPi;
    }
    IntervalShape::Other
}

/// Split `f` into an `x`-free coefficient and `x`-dependent factors.
/// Fractional/negative powers of products are distributed over the
/// factors so that e.g. `(x(1−x))^{−1/2}` becomes `x^{−1/2}(1−x)^{−1/2}`.
pub(super) fn factorize(arena: &mut Arena, f: ExprId, x: ExprId) -> (Vec<ExprId>, Vec<ExprId>) {
    let mut coeff = Vec::new();
    let mut dep = Vec::new();
    let mut stack: Vec<ExprId> = vec![f];
    while let Some(id) = stack.pop() {
        if !walk::contains(arena, id, x) {
            coeff.push(id);
            continue;
        }
        match arena.node(id).clone() {
            ExprNode::Mul(children) => stack.extend(children.iter().copied()),
            ExprNode::Neg(inner) => {
                coeff.push(arena.neg_one());
                stack.push(inner);
            }
            ExprNode::Pow(base, exp) if arena.as_num(exp).is_some() => {
                match arena.node(base).clone() {
                    ExprNode::Mul(children) => {
                        for c in children.iter() {
                            let p = arena.pow(*c, exp);
                            stack.push(p);
                        }
                    }
                    // (b^m)^n with numeric m, n → b^{mn} (real-valued
                    // context; the canon layer flattens only an integer n).
                    ExprNode::Pow(inner_base, inner_exp) if arena.as_num(inner_exp).is_some() => {
                        let m = arena.as_num(inner_exp).cloned().unwrap_or_else(Ratio::one);
                        let n = arena.as_num(exp).cloned().unwrap_or_else(Ratio::one);
                        let mn = ratio_expr(arena, &(m * n));
                        let flat = arena.pow(inner_base, mn);
                        stack.push(flat);
                    }
                    _ => dep.push(id),
                }
            }
            _ => dep.push(id),
        }
    }
    (coeff, dep)
}

/// Multiply a table value by the constant coefficient.
pub(super) fn with_coeff(arena: &mut Arena, coeff: &[ExprId], v: ExprId) -> ExprId {
    let mut all: Vec<ExprId> = coeff.to_vec();
    all.push(v);
    let r = arena.mul(&all);
    safe_eval(arena, r)
}

/// `x^p` → `p` (as an expression); `x` → `1`.
fn as_x_power(arena: &Arena, e: ExprId, x: ExprId) -> Option<ExprId> {
    if e == x {
        return Some(arena.one());
    }
    if let ExprNode::Pow(base, exp) = arena.node(e)
        && *base == x
        && !walk::contains(arena, *exp, x)
    {
        return Some(*exp);
    }
    None
}

/// Rational value of a numeric expression.
fn as_ratio(arena: &Arena, e: ExprId) -> Option<Q> {
    arena.as_num(e).cloned()
}

/// `Pow(base, −m)` → `(base, m)` with `m` a positive number; a bare
/// non-power expression is not a reciprocal.
fn as_reciprocal(arena: &Arena, e: ExprId) -> Option<(ExprId, Q)> {
    if let ExprNode::Pow(base, exp) = arena.node(e)
        && let Some(r) = arena.as_num(*exp)
        && r.is_negative()
    {
        return Some((*base, -r.clone()));
    }
    None
}

/// Trig kind for table matching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Trig {
    Sin,
    Cos,
}

/// `sin(arg)^n` / `cos(arg)^n` → `(kind, arg, n)`; bare `sin(arg)` has `n = 1`.
fn as_trig_power(arena: &Arena, e: ExprId) -> Option<(Trig, ExprId, Q)> {
    let (inner, n) = if let ExprNode::Pow(base, exp) = arena.node(e) {
        (*base, arena.as_num(*exp)?.clone())
    } else {
        (e, Ratio::one())
    };
    match arena.node(inner) {
        ExprNode::Sin(arg) => Some((Trig::Sin, *arg, n)),
        ExprNode::Cos(arg) => Some((Trig::Cos, *arg, n)),
        _ => None,
    }
}

/// `ln(arg)^n` → `(arg, n)`.
fn as_ln_power(arena: &Arena, e: ExprId) -> Option<(ExprId, Q)> {
    let (inner, n) = if let ExprNode::Pow(base, exp) = arena.node(e) {
        (*base, arena.as_num(*exp)?.clone())
    } else {
        (e, Ratio::one())
    };
    match arena.node(inner) {
        ExprNode::Ln(arg) => Some((*arg, n)),
        _ => None,
    }
}

/// Exact `Γ(r)` for integer and half-integer `r`; a `Gamma` node otherwise.
pub(super) fn gamma_exact(arena: &mut Arena, r: &Q) -> Option<ExprId> {
    if r.is_integer() {
        if !r.is_positive() {
            return None;
        }
        let n = r.to_integer().to_u64()?;
        if n > 170 {
            return None;
        }
        let mut acc = BigInt::one();
        for k in 2..n {
            acc *= BigInt::from(k);
        }
        return Some(arena.big_int(acc));
    }
    let two = BigInt::from(2);
    if *r.denom() == two {
        // Γ(k + ½) = (2k)!/(4^k k!) √π  for k ≥ 0; use recurrence downwards.
        let twice = r.numer().clone(); // odd
        let k2 = (&twice - BigInt::one()) / &two; // r = k2 + 1/2
        let k = k2.to_i64()?;
        if k.abs() > 200 {
            return None;
        }
        // Compute rational factor c with Γ(k + ½) = c √π.
        let mut c = Ratio::one();
        if k >= 0 {
            for j in 0..k {
                // Γ(j + 3/2) = (j + 1/2) Γ(j + 1/2)
                c *= Ratio::new(BigInt::from(2 * j + 1), two.clone());
            }
        } else {
            for j in 0..(-k) {
                // Γ(−j − 1/2) = Γ(−j + 1/2) / (−j − 1/2)
                c /= Ratio::new(BigInt::from(-2 * j - 1), two.clone());
            }
        }
        let c_id = {
            let nid = arena.intern_num(c);
            arena.intern(ExprNode::Num(nid))
        };
        let pi = arena.pi();
        let sqrt_pi = arena.sqrt(pi);
        return Some(arena.mul(&[c_id, sqrt_pi]));
    }
    let r_id = {
        let nid = arena.intern_num(r.clone());
        arena.intern(ExprNode::Num(nid))
    };
    Some(arena.gamma(r_id))
}

/// `Γ(e)` for a general expression: exact when numeric, symbolic otherwise.
fn gamma_of(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    if let Some(r) = as_ratio(arena, e) {
        return gamma_exact(arena, &r);
    }
    Some(arena.gamma(e))
}

/// `B(p, q)`: `π / sin(πp)` when `p + q = 1`, exact gammas otherwise.
fn beta_of(arena: &mut Arena, p: ExprId, q: ExprId) -> Option<ExprId> {
    if let (Some(rp), Some(rq)) = (as_ratio(arena, p), as_ratio(arena, q)) {
        if !rp.is_positive() || !rq.is_positive() {
            return None;
        }
        if &rp + &rq == Ratio::one() {
            let pi = arena.pi();
            let pp = arena.mul(&[p, pi]);
            let s = arena.sin(pp);
            let r = arena.div(pi, s);
            return Some(safe_eval(arena, r));
        }
    }
    let gp = gamma_of(arena, p)?;
    let gq = gamma_of(arena, q)?;
    let pq = arena.add(&[p, q]);
    let pq = safe_eval(arena, pq);
    let gpq = gamma_of(arena, pq)?;
    let num = arena.mul(&[gp, gq]);
    let r = arena.div(num, gpq);
    Some(safe_eval(arena, r))
}

/// `ζ(2k)` exactly via Bernoulli numbers.
pub(super) fn zeta_even(arena: &mut Arena, two_k: u32) -> Option<ExprId> {
    if two_k == 0 || !two_k.is_multiple_of(2) || two_k > 60 {
        return None;
    }
    let k = two_k / 2;
    let b = crate::base::bernoulli::bernoulli(two_k as usize);
    // ζ(2k) = (−1)^{k+1} B_{2k} (2π)^{2k} / (2 (2k)!)
    let mut fact = BigInt::one();
    for j in 2..=two_k {
        fact *= BigInt::from(j);
    }
    let sign = if k % 2 == 1 { 1 } else { -1 };
    let two_pow = BigInt::from(2).pow(two_k);
    let coeff = b * Ratio::from_integer(two_pow) * Ratio::from_integer(BigInt::from(sign))
        / Ratio::from_integer(fact * BigInt::from(2));
    let c_id = {
        let nid = arena.intern_num(coeff);
        arena.intern(ExprNode::Num(nid))
    };
    let pi = arena.pi();
    let n = arena.int(two_k as i64);
    let pi_pow = arena.pow(pi, n);
    Some(arena.mul(&[c_id, pi_pow]))
}

/// Coefficients `[c₀, c₁, c₂]` of a quadratic in `x` (zero-padded).
fn quadratic_coeffs(arena: &mut Arena, e: ExprId, x: ExprId) -> Option<[ExprId; 3]> {
    let c = calculus_util::poly_coeffs_symbolic(arena, e, x)?;
    if c.len() > 3 {
        return None;
    }
    let z = arena.zero();
    Some([
        c.first().copied().unwrap_or(z),
        c.get(1).copied().unwrap_or(z),
        c.get(2).copied().unwrap_or(z),
    ])
}

/// Decompose `k + c·x^n` (exactly two terms, `k`, `c` free of `x`).
fn as_binomial_in_x(arena: &mut Arena, e: ExprId, x: ExprId) -> Option<(ExprId, ExprId, Q)> {
    let terms: Vec<ExprId> = match arena.node(e).clone() {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => return None,
    };
    if terms.len() != 2 {
        return None;
    }
    let mut k: Option<ExprId> = None;
    let mut cx: Option<(ExprId, Q)> = None;
    for t in terms {
        if !walk::contains(arena, t, x) {
            k = Some(t);
            continue;
        }
        // c · x^n
        let (coeff, dep) = factorize(arena, t, x);
        if dep.len() != 1 {
            return None;
        }
        let n = as_x_power(arena, dep[0], x)?;
        let n = as_ratio(arena, n)?;
        let c = if coeff.is_empty() {
            arena.one()
        } else {
            let m = arena.mul(&coeff);
            safe_eval(arena, m)
        };
        cx = Some((c, n));
    }
    let k = k?;
    let (c, n) = cx?;
    Some((k, c, n))
}

/// Match `f` against the known-value table for `∫ₐᵇ`.
pub(super) fn table_lookup(
    arena: &mut Arena,
    f: ExprId,
    x: ExprId,
    a: ExprId,
    b: ExprId,
) -> Option<ExprId> {
    let iv = classify_interval(arena, a, b);
    if iv == IntervalShape::Other {
        return None;
    }
    let (coeff, dep) = factorize(arena, f, x);
    if dep.is_empty() || dep.len() > 4 {
        return None;
    }
    let v = match iv {
        IntervalShape::ZeroInf => table_zero_inf(arena, &dep, x),
        IntervalShape::FullLine => table_full_line(arena, &dep, x),
        IntervalShape::ZeroOne => table_zero_one(arena, &dep, x),
        IntervalShape::ZeroHalfPi
        | IntervalShape::ZeroPi
        | IntervalShape::ZeroTwoPi
        | IntervalShape::NegPiPi => table_trig(arena, &dep, x, iv),
        IntervalShape::Other => None,
    }?;
    Some(with_coeff(arena, &coeff, v))
}

/// Shape of a parsed `[0, ∞)`-type integrand.
#[derive(Default)]
struct Shape {
    /// exponent of the bare `x^p` factor (None if absent)
    x_pow: Option<ExprId>,
    /// argument of an `exp(...)` factor
    exp_arg: Option<ExprId>,
    /// `(kind, arg, n)` of trig factors (at most two)
    trig: Vec<(Trig, ExprId, Q)>,
    /// `(base, m)` of reciprocal factors
    recip: Vec<(ExprId, Q)>,
    /// `(arg, n)` of a logarithm factor
    ln: Option<(ExprId, Q)>,
    /// `(arg, m)` of `cosh(arg)^{-m}` / `sinh(arg)^{-m}` (kind: true = cosh)
    hyp_recip: Option<(bool, ExprId, Q)>,
    /// any factor that could not be classified
    other: bool,
}

/// Classify the dependent factors.
fn parse_shape(arena: &Arena, dep: &[ExprId], x: ExprId) -> Shape {
    let mut s = Shape::default();
    for &d in dep {
        if let Some(p) = as_x_power(arena, d, x) {
            if s.x_pow.is_some() {
                s.other = true;
            }
            s.x_pow = Some(p);
        } else if let ExprNode::Exp(arg) = arena.node(d) {
            if s.exp_arg.is_some() {
                s.other = true;
            }
            s.exp_arg = Some(*arg);
        } else if let Some(t) = as_trig_power(arena, d) {
            s.trig.push(t);
        } else if let Some((base, m)) = as_reciprocal(arena, d) {
            match arena.node(base) {
                ExprNode::Cosh(arg) => s.hyp_recip = Some((true, *arg, m)),
                ExprNode::Sinh(arg) => s.hyp_recip = Some((false, *arg, m)),
                _ => s.recip.push((base, m)),
            }
        } else if let Some(l) = as_ln_power(arena, d) {
            if s.ln.is_some() {
                s.other = true;
            }
            s.ln = Some(l);
        } else {
            s.other = true;
        }
    }
    s
}

/// Extract `α` from a purely linear argument `α·x` (no constant term).
pub(super) fn pure_linear_coeff(arena: &mut Arena, arg: ExprId, x: ExprId) -> Option<ExprId> {
    let lin = calculus_util::linear_coeffs_exact(arena, arg, x)?;
    if !arena.is_zero_structural(lin.intercept) {
        return None;
    }
    Some(lin.slope)
}

/// Build `Num` from a rational.
fn ratio_expr(arena: &mut Arena, r: &Q) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

/// A single `[0, ∞)` table entry.
type HalfLineEntry = fn(&mut Arena, &HalfLine, ExprId) -> Option<ExprId>;

/// Shared pre-processing for the `[0, ∞)` entries.
struct HalfLine {
    s: Shape,
    /// exponent `p` of the `x^p` factor (0 if absent)
    p: ExprId,
    /// `p + 1`, evaluated
    p1: ExprId,
}

/// Table for `∫₀^∞`: each entry is independent, so a failed structural
/// match inside one entry never prevents the next from being tried.
fn table_zero_inf(arena: &mut Arena, dep: &[ExprId], x: ExprId) -> Option<ExprId> {
    let s = parse_shape(arena, dep, x);
    if s.other {
        return None;
    }
    let one = arena.one();
    let p = s.x_pow.unwrap_or(arena.zero());
    let p1 = arena.add(&[p, one]);
    let p1 = safe_eval(arena, p1);
    let h = HalfLine { s, p, p1 };

    let entries: [HalfLineEntry; 10] = [
        entry_exp_linear,
        entry_exp_gaussian,
        entry_fresnel,
        entry_trig_over_power,
        entry_algebraic_beta,
        entry_trig_over_quadratic,
        entry_ln_over_quadratic,
        entry_bose_fermi,
        entry_hyperbolic_recip,
        entry_none,
    ];
    for entry in entries {
        if let Some(v) = entry(arena, &h, x) {
            return Some(v);
        }
    }
    None
}

/// Placeholder to keep the entry table length fixed.
fn entry_none(_arena: &mut Arena, _h: &HalfLine, _x: ExprId) -> Option<ExprId> {
    None
}

/// `x^p e^{−a x + c}` → `e^c Γ(p+1)/a^{p+1}`;
/// `e^{−a x} sin(bx)` → `b/(a²+b²)`;  `e^{−a x} cos(bx)` → `a/(a²+b²)`;
/// `e^{−a x} sin(bx)/x` → `atan(b/a)`   (all with `a > 0`).
fn entry_exp_linear(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    let arg = s.exp_arg?;
    if !s.recip.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    let [c0, c1, c2] = quadratic_coeffs(arena, arg, x)?;
    if !arena.is_zero_structural(c2) {
        return None;
    }
    let e_c0 = arena.exp(c0);
    let a = arena.neg(c1);
    let a = safe_eval(arena, a);
    if !is_positive(arena, a) {
        return None;
    }
    if s.trig.is_empty() {
        if !is_positive(arena, h.p1) {
            return None;
        }
        let g = gamma_of(arena, h.p1)?;
        let neg_p1 = arena.neg(h.p1);
        let a_pow = arena.pow(a, neg_p1);
        return Some(arena.mul(&[e_c0, g, a_pow]));
    }
    if s.trig.len() != 1 {
        return None;
    }
    let (kind, targ, n) = s.trig[0].clone();
    if n != Ratio::one() {
        return None;
    }
    let bcoef = pure_linear_coeff(arena, targ, x)?;
    // e^{−ax} sin(bx) converges only for a > |Im b|: ask for a real b.
    if !is_real(arena, bcoef) {
        return None;
    }
    if arena.is_zero_structural(h.p) {
        let a2 = arena.mul(&[a, a]);
        let b2 = arena.mul(&[bcoef, bcoef]);
        let denom = arena.add(&[a2, b2]);
        let num = match kind {
            Trig::Sin => bcoef,
            Trig::Cos => a,
        };
        let r = arena.div(num, denom);
        return Some(arena.mul(&[e_c0, r]));
    }
    if kind == Trig::Sin && h.p == arena.neg_one() {
        let r = arena.div(bcoef, a);
        let at = arena.atan(r);
        return Some(arena.mul(&[e_c0, at]));
    }
    None
}

/// `x^p e^{−a x² + c}` → `e^c Γ((p+1)/2) / (2 a^{(p+1)/2})`;
/// `e^{−a x²} cos(bx)` → `½ √(π/a) e^{−b²/(4a)}`   (`a > 0`).
fn entry_exp_gaussian(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    let arg = s.exp_arg?;
    if !s.recip.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    let [c0, c1, c2] = quadratic_coeffs(arena, arg, x)?;
    if arena.is_zero_structural(c2) || !arena.is_zero_structural(c1) {
        return None;
    }
    let e_c0 = arena.exp(c0);
    let a = arena.neg(c2);
    let a = safe_eval(arena, a);
    if !is_positive(arena, a) {
        return None;
    }
    let half = arena.rational(1, 2);
    if s.trig.is_empty() {
        if !is_positive(arena, h.p1) {
            return None;
        }
        let s_arg = arena.mul(&[half, h.p1]);
        let s_arg = safe_eval(arena, s_arg);
        let g = gamma_of(arena, s_arg)?;
        let neg_s = arena.neg(s_arg);
        let a_pow = arena.pow(a, neg_s);
        return Some(arena.mul(&[e_c0, half, g, a_pow]));
    }
    if s.trig.len() != 1 || !arena.is_zero_structural(h.p) {
        return None;
    }
    let (kind, targ, n) = s.trig[0].clone();
    if kind != Trig::Cos || n != Ratio::one() {
        return None;
    }
    let bcoef = pure_linear_coeff(arena, targ, x)?;
    let pi = arena.pi();
    let pi_over_a = arena.div(pi, a);
    let root = arena.sqrt(pi_over_a);
    let b2 = arena.mul(&[bcoef, bcoef]);
    let four = arena.int(4);
    let four_a = arena.mul(&[four, a]);
    let ratio = arena.div(b2, four_a);
    let neg = arena.neg(ratio);
    let e = arena.exp(neg);
    Some(arena.mul(&[e_c0, half, root, e]))
}

/// Fresnel: `sin(c x²)`, `cos(c x²)` → `½ √(π/(2|c|))` (sign of `c` for sin).
fn entry_fresnel(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    if s.exp_arg.is_some() || !s.recip.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    if s.trig.len() != 1 || !arena.is_zero_structural(h.p) {
        return None;
    }
    let (kind, targ, n) = s.trig[0].clone();
    if n != Ratio::one() {
        return None;
    }
    let [c0, c1, c2] = quadratic_coeffs(arena, targ, x)?;
    if !arena.is_zero_structural(c0) || !arena.is_zero_structural(c1) {
        return None;
    }
    let c_sign = sign_of(arena, c2)?;
    if c_sign == Ordering::Equal {
        return None;
    }
    let abs_c = arena.abs(c2);
    let two = arena.int(2);
    let two_c = arena.mul(&[two, abs_c]);
    let pi = arena.pi();
    let ratio = arena.div(pi, two_c);
    let root = arena.sqrt(ratio);
    let half = arena.rational(1, 2);
    let v = arena.mul(&[half, root]);
    if kind == Trig::Sin && c_sign == Ordering::Less {
        return Some(arena.neg(v));
    }
    Some(v)
}

/// `sin(bx)/x` → `(π/2) sign(b)`;  `sin²(bx)/x²` → `π|b|/2`;
/// `x^p sin(bx)` → `Γ(p+1) sin(π(p+1)/2)/b^{p+1}` (−2 < p < 0, non-integer);
/// `x^p cos(bx)` → `Γ(p+1) cos(π(p+1)/2)/b^{p+1}` (−1 < p < 0, non-integer).
fn entry_trig_over_power(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    if s.exp_arg.is_some() || !s.recip.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    if s.trig.len() != 1 {
        return None;
    }
    let (kind, targ, n) = s.trig[0].clone();
    let bcoef = pure_linear_coeff(arena, targ, x)?;
    // sin(bx) grows like e^{|Im b| x}: every entry needs a real b.
    if !is_real(arena, bcoef) {
        return None;
    }
    let two = Ratio::from_integer(BigInt::from(2));
    let pi = arena.pi();
    let half = arena.rational(1, 2);
    if kind == Trig::Sin && n == Ratio::one() && h.p == arena.neg_one() {
        let sg = arena.sign(bcoef);
        return Some(arena.mul(&[half, pi, sg]));
    }
    if kind == Trig::Sin && n == two && h.p == arena.int(-2) {
        let ab = arena.abs(bcoef);
        return Some(arena.mul(&[half, pi, ab]));
    }
    if n != Ratio::one() || !is_positive(arena, bcoef) {
        return None;
    }
    let pr = as_ratio(arena, h.p)?;
    if pr.is_integer() {
        return None;
    }
    let lower = if kind == Trig::Sin { -2 } else { -1 };
    if pr <= Ratio::from_integer(BigInt::from(lower)) || pr >= Ratio::zero() {
        return None;
    }
    let g = gamma_of(arena, h.p1)?;
    let ang = arena.mul(&[half, pi, h.p1]);
    let tr = match kind {
        Trig::Sin => arena.sin(ang),
        Trig::Cos => arena.cos(ang),
    };
    let neg_p1 = arena.neg(h.p1);
    let b_pow = arena.pow(bcoef, neg_p1);
    Some(arena.mul(&[g, tr, b_pow]))
}

/// `x^p (k + c x^n)^{−m}` → `(1/n) k^{s−m} c^{−s} B(s, m−s)` with
/// `s = (p+1)/n`, `0 < s < m`, `k, c > 0`.
fn entry_algebraic_beta(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    if s.exp_arg.is_some() || !s.trig.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    if s.recip.len() != 1 {
        return None;
    }
    let (base, m) = s.recip[0].clone();
    let (k, c, n) = as_binomial_in_x(arena, base, x)?;
    if !n.is_positive() || !is_positive(arena, k) || !is_positive(arena, c) {
        return None;
    }
    let pr = as_ratio(arena, h.p)?;
    let s_val = (pr + Ratio::one()) / n.clone();
    if !s_val.is_positive() || s_val >= m {
        return None;
    }
    let s_id = ratio_expr(arena, &s_val);
    let m_minus_s = ratio_expr(arena, &(m.clone() - s_val.clone()));
    let beta = beta_of(arena, s_id, m_minus_s)?;
    let inv_n = ratio_expr(arena, &(Ratio::one() / n));
    let k_exp = ratio_expr(arena, &(s_val.clone() - m));
    let k_pow = arena.pow(k, k_exp);
    let c_exp = ratio_expr(arena, &(-s_val));
    let c_pow = arena.pow(c, c_exp);
    Some(arena.mul(&[inv_n, k_pow, c_pow, beta]))
}

/// `cos(bx)/(c x² + k)` → `π e^{−ab}/(2ac)`;  `x sin(bx)/(c x² + k)` →
/// `π e^{−ab}/(2c)` with `a = √(k/c)`, `b > 0`.
fn entry_trig_over_quadratic(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    if s.exp_arg.is_some() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    if s.recip.len() != 1 || s.trig.len() != 1 {
        return None;
    }
    let (base, m) = s.recip[0].clone();
    if m != Ratio::one() {
        return None;
    }
    let (k, c, n) = as_binomial_in_x(arena, base, x)?;
    let two = Ratio::from_integer(BigInt::from(2));
    if n != two || !is_positive(arena, k) || !is_positive(arena, c) {
        return None;
    }
    let (kind, targ, tn) = s.trig[0].clone();
    if tn != Ratio::one() {
        return None;
    }
    let bcoef = pure_linear_coeff(arena, targ, x)?;
    if !is_positive(arena, bcoef) {
        return None;
    }
    let k_over_c = arena.div(k, c);
    let a_val = arena.sqrt(k_over_c);
    let ab = arena.mul(&[a_val, bcoef]);
    let neg_ab = arena.neg(ab);
    let e = arena.exp(neg_ab);
    let half = arena.rational(1, 2);
    let pi = arena.pi();
    let neg_one = arena.neg_one();
    let inv_c = arena.pow(c, neg_one);
    let one = arena.one();
    match kind {
        Trig::Cos if arena.is_zero_structural(h.p) => {
            let inv_a = arena.pow(a_val, neg_one);
            Some(arena.mul(&[half, pi, e, inv_a, inv_c]))
        }
        Trig::Sin if h.p == one => Some(arena.mul(&[half, pi, e, inv_c])),
        _ => None,
    }
}

/// `ln(x)/(c x² + k)` → `π ln(k/c) / (4 √(kc))`.
fn entry_ln_over_quadratic(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    let (larg, ln_n) = s.ln.clone()?;
    if s.exp_arg.is_some() || !s.trig.is_empty() || s.hyp_recip.is_some() {
        return None;
    }
    if s.recip.len() != 1 || !arena.is_zero_structural(h.p) || larg != x || ln_n != Ratio::one() {
        return None;
    }
    let (base, m) = s.recip[0].clone();
    if m != Ratio::one() {
        return None;
    }
    let (k, c, n) = as_binomial_in_x(arena, base, x)?;
    let two = Ratio::from_integer(BigInt::from(2));
    if n != two || !is_positive(arena, k) || !is_positive(arena, c) {
        return None;
    }
    let kc = arena.mul(&[k, c]);
    let root = arena.sqrt(kc);
    let four = arena.int(4);
    let denom = arena.mul(&[four, root]);
    let k_over_c = arena.div(k, c);
    let l = arena.ln(k_over_c);
    let pi = arena.pi();
    let num = arena.mul(&[pi, l]);
    Some(arena.div(num, denom))
}

/// Bose–Einstein / Fermi–Dirac: `x^s/(e^{ax} − 1)` → `Γ(s+1) ζ(s+1)/a^{s+1}`,
/// `x^s/(e^{ax} + 1)` → `(1 − 2^{−s}) Γ(s+1) ζ(s+1)/a^{s+1}` for odd `s`.
fn entry_bose_fermi(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    if s.exp_arg.is_some() || !s.trig.is_empty() || s.ln.is_some() || s.hyp_recip.is_some() {
        return None;
    }
    if s.recip.len() != 1 {
        return None;
    }
    let (base, m) = s.recip[0].clone();
    if m != Ratio::one() {
        return None;
    }
    let terms: Vec<ExprId> = match arena.node(base).clone() {
        ExprNode::Add(ch) if ch.len() == 2 => ch.to_vec(),
        _ => return None,
    };
    let mut exp_arg = None;
    let mut const_term = None;
    for t in terms {
        if let ExprNode::Exp(arg) = arena.node(t).clone()
            && walk::contains(arena, arg, x)
        {
            exp_arg = Some(arg);
        } else if !walk::contains(arena, t, x) {
            const_term = Some(t);
        }
    }
    let (arg, kterm) = (exp_arg?, const_term?);
    let kr = as_ratio(arena, kterm)?;
    let fermi = if kr == Ratio::one() {
        true
    } else if kr == -Ratio::one() {
        false
    } else {
        return None;
    };
    let alpha = pure_linear_coeff(arena, arg, x)?;
    if !is_positive(arena, alpha) {
        return None;
    }
    let sr = as_ratio(arena, h.p)?;
    if !sr.is_integer() || !sr.is_positive() {
        return None;
    }
    let s_int = sr.to_integer().to_u32()?;
    let zeta = zeta_even(arena, s_int + 1)?;
    let g = gamma_exact(arena, &(sr + Ratio::one()))?;
    let neg_s1 = arena.neg(h.p1);
    let a_pow = arena.pow(alpha, neg_s1);
    let mut parts = vec![g, zeta, a_pow];
    if fermi {
        let two_pow = Ratio::new(BigInt::one(), BigInt::from(2).pow(s_int));
        let factor = ratio_expr(arena, &(Ratio::one() - two_pow));
        parts.push(factor);
    }
    Some(arena.mul(&parts))
}

/// `1/cosh(ax)` → `π/(2a)`;  `x/sinh(ax)` → `π²/(4a²)`   (`a > 0`).
fn entry_hyperbolic_recip(arena: &mut Arena, h: &HalfLine, x: ExprId) -> Option<ExprId> {
    let s = &h.s;
    let (is_cosh, harg, m) = s.hyp_recip.clone()?;
    if s.exp_arg.is_some() || !s.trig.is_empty() || !s.recip.is_empty() || s.ln.is_some() {
        return None;
    }
    if m != Ratio::one() {
        return None;
    }
    let alpha = pure_linear_coeff(arena, harg, x)?;
    if !is_positive(arena, alpha) {
        return None;
    }
    let pi = arena.pi();
    let one = arena.one();
    if is_cosh && arena.is_zero_structural(h.p) {
        let two = arena.int(2);
        let two_a = arena.mul(&[two, alpha]);
        return Some(arena.div(pi, two_a));
    }
    if !is_cosh && h.p == one {
        let pi2 = arena.mul(&[pi, pi]);
        let four = arena.int(4);
        let a2 = arena.mul(&[alpha, alpha]);
        let denom = arena.mul(&[four, a2]);
        return Some(arena.div(pi2, denom));
    }
    None
}

/// Table for `∫₋∞^∞`: general Gaussian; otherwise even integrands reduce
/// to twice the half-line table (odd ones are handled by the parity
/// shortcut before we get here).
fn table_full_line(arena: &mut Arena, dep: &[ExprId], x: ExprId) -> Option<ExprId> {
    let s = parse_shape(arena, dep, x);
    if !s.other
        && let Some(arg) = s.exp_arg
        && s.trig.is_empty()
        && s.recip.is_empty()
        && s.ln.is_none()
        && s.hyp_recip.is_none()
    {
        let p = s.x_pow.unwrap_or(arena.zero());
        let [c0, c1, c2] = quadratic_coeffs(arena, arg, x)?;
        let a = arena.neg(c2);
        let a = safe_eval(arena, a);
        if !is_positive(arena, a) {
            return None;
        }
        // ∫ e^{−a x² + c1 x + c0} = √(π/a) e^{c0 + c1²/(4a)}
        let pi = arena.pi();
        let pi_over_a = arena.div(pi, a);
        let root = arena.sqrt(pi_over_a);
        let c1_sq = arena.mul(&[c1, c1]);
        let four_a = {
            let four = arena.int(4);
            arena.mul(&[four, a])
        };
        let shift = arena.div(c1_sq, four_a);
        let e_arg = arena.add(&[c0, shift]);
        let e = arena.exp(e_arg);
        if arena.is_zero_structural(p) {
            return Some(arena.mul(&[root, e]));
        }
        // x e^{−a x² + c1 x + c0} = (c1/(2a)) √(π/a) e^{…}
        if p == arena.one() {
            let two_a = {
                let two = arena.int(2);
                arena.mul(&[two, a])
            };
            let mean = arena.div(c1, two_a);
            return Some(arena.mul(&[mean, root, e]));
        }
        return None;
    }
    None
}

/// Table for `∫₀¹`.
fn table_zero_one(arena: &mut Arena, dep: &[ExprId], x: ExprId) -> Option<ExprId> {
    let s = parse_shape(arena, dep, x);
    let p = s.x_pow.unwrap_or(arena.zero());
    let one = arena.one();
    let p_plus_1 = arena.add(&[p, one]);
    let p_plus_1 = safe_eval(arena, p_plus_1);

    // ── x^p ln(x)^n → (−1)^n n! / (p+1)^{n+1} ─────────────────────
    if let Some((larg, n)) = s.ln.clone()
        && larg == x
        && s.recip.is_empty()
        && s.trig.is_empty()
        && s.exp_arg.is_none()
        && s.hyp_recip.is_none()
        && !s.other
    {
        if !n.is_integer() || !n.is_positive() || !is_positive(arena, p_plus_1) {
            return None;
        }
        let ni = n.to_integer().to_u32()?;
        if ni > 40 {
            return None;
        }
        let mut fact = BigInt::one();
        for j in 2..=ni {
            fact *= BigInt::from(j);
        }
        let sign = if ni % 2 == 0 { 1 } else { -1 };
        let c = ratio_expr(arena, &Ratio::from_integer(fact * BigInt::from(sign)));
        let neg_n1 = arena.int(-(ni as i64) - 1);
        let denom = arena.pow(p_plus_1, neg_n1);
        return Some(arena.mul(&[c, denom]));
    }

    // ── Beta: x^p (1 − x^n)^q  (possibly with q = 1 as a bare Add) ──
    // Collect the non-power factors as (base, q).
    let mut beta_factors: Vec<(ExprId, ExprId)> = Vec::new();
    let mut ok = true;
    for &d in dep {
        if as_x_power(arena, d, x).is_some() {
            continue;
        }
        match arena.node(d).clone() {
            ExprNode::Pow(base, exp) if !walk::contains(arena, exp, x) => {
                beta_factors.push((base, exp))
            }
            ExprNode::Add(_) => beta_factors.push((d, one)),
            _ => ok = false,
        }
    }
    if ok && beta_factors.len() == 1 {
        let (base, q) = beta_factors[0];
        // base = β − β x^n  (β > 0)
        let coeffs = calculus_util::poly_coeffs_symbolic(arena, base, x)?;
        let n = coeffs.len().checked_sub(1)?;
        if n >= 1 && coeffs[1..n].iter().all(|&c| arena.is_zero_structural(c)) {
            let beta_c = coeffs[0];
            let lead = coeffs[n];
            let neg_lead = arena.neg(lead);
            let neg_lead = safe_eval(arena, neg_lead);
            if cmp_points(arena, beta_c, neg_lead) == Some(Ordering::Equal)
                && is_positive(arena, beta_c)
            {
                let q_plus_1 = arena.add(&[q, one]);
                let q_plus_1 = safe_eval(arena, q_plus_1);
                if !is_positive(arena, q_plus_1) || !is_positive(arena, p_plus_1) {
                    return None;
                }
                let n_r = Ratio::from_integer(BigInt::from(n));
                let inv_n = ratio_expr(arena, &(Ratio::one() / n_r.clone()));
                let s_arg = arena.mul(&[inv_n, p_plus_1]);
                let s_arg = safe_eval(arena, s_arg);
                let b = beta_of(arena, s_arg, q_plus_1)?;
                let beta_pow = arena.pow(beta_c, q);
                return Some(arena.mul(&[inv_n, beta_pow, b]));
            }
        }
        return None;
    }

    // ── Dilogarithm-type: ln(L₁)/L₂ with linear/quadratic L ────────
    // The denominator is either a reciprocal factor or a bare x⁻¹.
    let dilog_denominator: Option<ExprId> = if s.recip.len() == 1 && s.x_pow.is_none() {
        if s.recip[0].1 == Ratio::one() {
            Some(s.recip[0].0)
        } else {
            None
        }
    } else if s.recip.is_empty() && s.x_pow == Some(arena.neg_one()) {
        Some(x)
    } else {
        None
    };
    if let Some((larg, n)) = s.ln.clone()
        && n == Ratio::one()
        && let Some(base) = dilog_denominator
        && s.trig.is_empty()
        && s.exp_arg.is_none()
        && s.hyp_recip.is_none()
        && !s.other
    {
        // Classify the log argument: x, 1−x, 1+x.
        #[derive(PartialEq, Clone, Copy)]
        enum L {
            X,
            OneMinusX,
            OnePlusX,
        }
        let classify = |arena: &mut Arena, e: ExprId| -> Option<(L, ExprId)> {
            if e == x {
                return Some((L::X, arena.one()));
            }
            let c = calculus_util::poly_coeffs_symbolic(arena, e, x)?;
            match c.len() {
                2 => {
                    let c0 = as_ratio(arena, c[0])?;
                    let c1 = as_ratio(arena, c[1])?;
                    if c0.is_zero() {
                        // c1 x
                        return Some((L::X, c[1]));
                    }
                    if c0 == -c1.clone() {
                        return Some((L::OneMinusX, c[0]));
                    }
                    if c0 == c1 {
                        return Some((L::OnePlusX, c[0]));
                    }
                    None
                }
                _ => None,
            }
        };
        let (lk, lscale) = classify(arena, larg)?;
        if !arena.is_one_structural(lscale) && lk != L::X {
            return None; // ln(k(1−x)) splits into ln k + ln(1−x); skip
        }
        // Denominator: k·x, k(1−x), k(1+x), k(1−x²)
        let dc = calculus_util::poly_coeffs_symbolic(arena, base, x)?;
        let pi = arena.pi();
        let pi2 = arena.mul(&[pi, pi]);
        let value_over = |arena: &mut Arena, num: i64, den: i64, k: ExprId| -> Option<ExprId> {
            let r = arena.rational(num, den);
            let v = arena.mul(&[r, pi2]);
            let out = arena.div(v, k);
            Some(safe_eval(arena, out))
        };
        return match dc.len() {
            2 => {
                let d0 = as_ratio(arena, dc[0])?;
                let d1 = as_ratio(arena, dc[1])?;
                if d0.is_zero() {
                    // ln(L)/(k x)
                    let k = dc[1];
                    return match lk {
                        L::OneMinusX => value_over(arena, -1, 6, k),
                        L::OnePlusX => value_over(arena, 1, 12, k),
                        L::X => None, // divergent (ln²)
                    };
                }
                if d0 == -d1.clone() {
                    // k(1 − x)
                    let k = dc[0];
                    return match lk {
                        L::X => {
                            if arena.is_one_structural(lscale) {
                                value_over(arena, -1, 6, k)
                            } else {
                                None
                            }
                        }
                        L::OnePlusX => {
                            // ∫₀¹ ln(1+x)/(1−x) diverges
                            None
                        }
                        L::OneMinusX => None, // divergent
                    };
                }
                if d0 == d1 {
                    // k(1 + x)
                    let k = dc[0];
                    return match lk {
                        L::X if arena.is_one_structural(lscale) => value_over(arena, -1, 12, k),
                        _ => None,
                    };
                }
                None
            }
            3 => {
                let d0 = as_ratio(arena, dc[0])?;
                let d1 = as_ratio(arena, dc[1])?;
                let d2 = as_ratio(arena, dc[2])?;
                if d1.is_zero()
                    && d0 == -d2.clone()
                    && lk == L::X
                    && arena.is_one_structural(lscale)
                {
                    // ln(x)/(k(1 − x²)) → −π²/(8k)
                    let k = dc[0];
                    return value_over(arena, -1, 8, k);
                }
                None
            }
            _ => None,
        };
    }

    None
}

/// Double factorial `n!!` (with `(−1)!! = 0!! = 1`).
fn double_factorial(n: i64) -> BigInt {
    let mut acc = BigInt::one();
    let mut k = n;
    while k > 1 {
        acc *= BigInt::from(k);
        k -= 2;
    }
    acc
}

/// Wallis: `∫₀^{π/2} sin^m cos^n = ((m−1)!!(n−1)!!/(m+n)!!) · (π/2 if m, n even)`
/// for non-negative integers; `½ B((m+1)/2, (n+1)/2)` otherwise (m, n > −1).
fn wallis(arena: &mut Arena, m: &Q, n: &Q) -> Option<ExprId> {
    if m.is_integer() && n.is_integer() && !m.is_negative() && !n.is_negative() {
        let mi = m.to_integer().to_i64()?;
        let ni = n.to_integer().to_i64()?;
        if mi + ni > 400 {
            return None;
        }
        let num = double_factorial(mi - 1) * double_factorial(ni - 1);
        let den = double_factorial(mi + ni);
        let r = Ratio::new(num, den);
        let r_id = ratio_expr(arena, &r);
        if mi % 2 == 0 && ni % 2 == 0 {
            let half = arena.rational(1, 2);
            let pi = arena.pi();
            return Some(arena.mul(&[r_id, half, pi]));
        }
        return Some(r_id);
    }
    let one = Ratio::one();
    let two = Ratio::from_integer(BigInt::from(2));
    let mp = (m + &one) / &two;
    let np = (n + &one) / &two;
    if !mp.is_positive() || !np.is_positive() {
        return None;
    }
    let mp_id = ratio_expr(arena, &mp);
    let np_id = ratio_expr(arena, &np);
    let b = beta_of(arena, mp_id, np_id)?;
    let half = arena.rational(1, 2);
    Some(arena.mul(&[half, b]))
}

/// Table for trigonometric intervals `[0, π/2]`, `[0, π]`, `[0, 2π]`, `[−π, π]`.
fn table_trig(arena: &mut Arena, dep: &[ExprId], x: ExprId, iv: IntervalShape) -> Option<ExprId> {
    let s = parse_shape(arena, dep, x);
    if s.other
        || s.exp_arg.is_some()
        || s.ln.is_some()
        || s.hyp_recip.is_some()
        || s.x_pow.is_some()
    {
        return None;
    }

    // ── Powers of sin / cos with argument x, or a single power of
    //    sin(kx) / cos(kx) with integer k over a (half) period ────────
    let single_multiple: Option<i64> =
        if s.recip.is_empty() && s.trig.len() == 1 && s.trig[0].1 != x {
            let kc = pure_linear_coeff(arena, s.trig[0].1, x);
            kc.and_then(|k| as_ratio(arena, k))
                .filter(|k| k.is_integer() && !k.is_zero())
                .and_then(|k| k.to_integer().to_i64())
        } else {
            None
        };
    if s.recip.is_empty()
        && !s.trig.is_empty()
        && (s.trig.iter().all(|(_, arg, _)| *arg == x) || single_multiple.is_some())
    {
        let mut m = Ratio::zero();
        let mut n = Ratio::zero();
        for (kind, _, k) in &s.trig {
            match kind {
                Trig::Sin => m += k,
                Trig::Cos => n += k,
            }
        }
        let neg_one = -Ratio::one();
        if m <= neg_one || n <= neg_one {
            return None;
        }
        let both_int = m.is_integer() && n.is_integer();
        let m_odd = both_int && m.to_integer().to_i64()? % 2 != 0;
        let n_odd = both_int && n.to_integer().to_i64()? % 2 != 0;
        if let Some(k) = single_multiple {
            // sinⁿ(kx) / cosⁿ(kx): substitution u = kx.  Over a full
            // period the value equals the k = 1 value; over [0, π] as well,
            // except that odd powers of sin(kx) with even k cancel.  Only
            // integer powers are supported (sign changes otherwise).
            if !both_int || iv == IntervalShape::ZeroHalfPi {
                return None;
            }
            if iv == IntervalShape::ZeroPi && m_odd {
                // ∫₀^π sinᵐ(kx) = (1/k) Σⱼ (−1)ʲ ∫₀^π sinᵐ: zero for even k,
                // W_π/k for odd k.
                if k % 2 == 0 {
                    return Some(arena.zero());
                }
                let w = wallis(arena, &m, &n)?;
                let two_over_k = arena.rational(2, k);
                return Some(arena.mul(&[two_over_k, w]));
            }
        }
        return match iv {
            IntervalShape::ZeroHalfPi => wallis(arena, &m, &n),
            IntervalShape::ZeroPi => {
                // cos changes sign on (π/2, π): need integer n.
                if !n.is_integer() {
                    return None;
                }
                if n_odd {
                    return Some(arena.zero());
                }
                let w = wallis(arena, &m, &n)?;
                let two = arena.int(2);
                Some(arena.mul(&[two, w]))
            }
            IntervalShape::ZeroTwoPi | IntervalShape::NegPiPi => {
                if !both_int {
                    return None;
                }
                if m_odd || n_odd {
                    return Some(arena.zero());
                }
                let w = wallis(arena, &m, &n)?;
                let four = arena.int(4);
                Some(arena.mul(&[four, w]))
            }
            _ => None,
        };
    }

    // ── Orthogonality: sin(mx)sin(nx), cos(mx)cos(nx), sin(mx)cos(nx) ──
    if s.recip.is_empty() && s.trig.len() == 2 {
        let (k1, a1, n1) = s.trig[0].clone();
        let (k2, a2, n2) = s.trig[1].clone();
        if n1 != Ratio::one() || n2 != Ratio::one() {
            return None;
        }
        let m1 = pure_linear_coeff(arena, a1, x)?;
        let m2 = pure_linear_coeff(arena, a2, x)?;
        let m1 = as_ratio(arena, m1)?;
        let m2 = as_ratio(arena, m2)?;
        if !m1.is_integer() || !m2.is_integer() {
            return None;
        }
        let (mi, ni) = (m1.to_integer().to_i64()?, m2.to_integer().to_i64()?);
        if mi == 0 || ni == 0 {
            return None;
        }
        let pi = arena.pi();
        let same_kind = k1 == k2;
        return match iv {
            IntervalShape::ZeroTwoPi | IntervalShape::NegPiPi => {
                if !same_kind {
                    return Some(arena.zero());
                }
                if mi.abs() == ni.abs() {
                    // sin·sin: π·sign(mn); cos·cos: π
                    if k1 == Trig::Sin && (mi > 0) != (ni > 0) {
                        return Some(arena.neg(pi));
                    }
                    return Some(pi);
                }
                Some(arena.zero())
            }
            IntervalShape::ZeroPi => {
                if same_kind {
                    if mi.abs() == ni.abs() {
                        let half = arena.rational(1, 2);
                        let v = arena.mul(&[half, pi]);
                        if k1 == Trig::Sin && (mi > 0) != (ni > 0) {
                            return Some(arena.neg(v));
                        }
                        return Some(v);
                    }
                    return Some(arena.zero());
                }
                // sin(m x) cos(n x) on [0, π]: (1 − (−1)^{m+n}) m/(m² − n²)
                let (sm, cn) = if k1 == Trig::Sin { (mi, ni) } else { (ni, mi) };
                if sm.abs() == cn.abs() {
                    return Some(arena.zero());
                }
                if (sm + cn) % 2 == 0 {
                    return Some(arena.zero());
                }
                let r = Ratio::new(BigInt::from(2 * sm), BigInt::from(sm * sm - cn * cn));
                Some(ratio_expr(arena, &r))
            }
            _ => None,
        };
    }

    // ── 1/(a + b cos x)^m, 1/(a + b sin x)^m on a full period ───────
    if s.trig.is_empty() && s.recip.len() == 1 {
        let (base, m) = s.recip[0].clone();
        let terms: Vec<ExprId> = match arena.node(base).clone() {
            ExprNode::Add(ch) if ch.len() >= 2 => ch.to_vec(),
            _ => return None,
        };
        // The constant part may be a sum (`b + 1 + cos x`).
        let free: Vec<ExprId> = terms
            .iter()
            .copied()
            .filter(|&t| !walk::contains(arena, t, x))
            .collect();
        if free.is_empty() || terms.len() - free.len() != 1 {
            return None;
        }
        let a_c: Option<ExprId> = Some(arena.add(&free));
        let mut b_c: Option<(ExprId, Trig)> = None;
        for t in terms {
            if !walk::contains(arena, t, x) {
                continue;
            }
            let (coeff, d) = factorize(arena, t, x);
            if d.len() != 1 {
                return None;
            }
            let (kind, arg, n) = as_trig_power(arena, d[0])?;
            if arg != x || n != Ratio::one() {
                return None;
            }
            let bc = if coeff.is_empty() {
                arena.one()
            } else {
                let mm = arena.mul(&coeff);
                safe_eval(arena, mm)
            };
            b_c = Some((bc, kind));
        }
        let (a_c, (b_c, kind)) = (a_c?, b_c?);
        // Need a > |b|.
        let abs_b = arena.abs(b_c);
        let diff_ab = arena.sub(a_c, abs_b);
        let diff_ab = safe_eval(arena, diff_ab);
        if !is_positive(arena, diff_ab) {
            return None;
        }
        let a2 = arena.mul(&[a_c, a_c]);
        let b2 = arena.mul(&[b_c, b_c]);
        let disc = arena.sub(a2, b2);
        let pi = arena.pi();
        let full = match iv {
            IntervalShape::ZeroTwoPi | IntervalShape::NegPiPi => true,
            IntervalShape::ZeroPi if kind == Trig::Cos => false,
            _ => return None,
        };
        if m == Ratio::one() {
            // 2π/√(a²−b²)   (π/√(a²−b²) on [0, π] for cos)
            let root = arena.sqrt(disc);
            let c = if full { arena.int(2) } else { arena.one() };
            let num = arena.mul(&[c, pi]);
            return Some(arena.div(num, root));
        }
        if m == Ratio::from_integer(BigInt::from(2)) {
            // 2π a/(a²−b²)^{3/2}
            let e = arena.rational(-3, 2);
            let pw = arena.pow(disc, e);
            let c = if full { arena.int(2) } else { arena.one() };
            return Some(arena.mul(&[c, pi, a_c, pw]));
        }
        return None;
    }

    None
}
