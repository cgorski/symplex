//! sin, cos, tan and cot at rational multiples of π, in radicals (part of [`eval`](super)).

use super::*;

// ── sin, cos, tan and cot at rational multiples of π, in radicals ───────────
//
// SymPy 1.14's `cos.eval` (and `sin.eval`, which reduces to it, and so
// `sec`/`csc`) gives `cos(pπ/n)` in square roots for exactly the
// denominators `n` of [`COS_PI_DENOMINATORS`]: the classical ones (`n | 4`,
// `n | 6`), `n = 5` (Chebyshev polynomials of `cos(π/5) = (1 + √5)/4`),
// `n = 8, 10` (half angles), and `n = 12, 15, 20, 24, 30, 40, 60, 120`
// (`cos(x − y)` for the pairs of its `_table2`, `pπ/12 = pπ/3 − pπ/4`, …).
// `tan.eval` and `cot.eval` add twice those through the half-angle formula
// (`n = 16, 48, 80, 240`).  Other denominators (`cos(π/7)`, `cos(π/16)`)
// stay, as there; `.rewrite(sqrt)` (Fermat primes, `n = 17`, `257`, …) is
// not automatic in SymPy and not done here.  The values are built by
// SymPy's formulas through the canonical arithmetic, so they come out in
// its normal forms (`cos(π/15)` = `−1/8 + √5/8 + √3·√(√5/8 + 5/8)/2`),
// except for the square roots of differences ([`sqrt_of_surd`]: `sin(π/5)`
// is `√5/(4·√(√5/8 + 5/8))`, SymPy's `sqrt(5/8 - sqrt(5)/8)`).
// Before 0.43 only the classical denominators were evaluated, and an
// equality like `sin(π/12) = (√6 − √2)/4` was left undecided.

/// The denominators `n` at which SymPy's `cos.eval`/`sin.eval` evaluate
/// `cos(pπ/n)`, `sin(pπ/n)` (`gcd(p, n) = 1`).
pub(super) const COS_PI_DENOMINATORS: [i64; 16] =
    [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120];

/// The denominators at which SymPy's `tan.eval`/`cot.eval` evaluate: those
/// of [`COS_PI_DENOMINATORS`] and, by `tan(x/2) = (1 − cos x)/sin x`, twice
/// them.
const TAN_PI_DENOMINATORS: [i64; 20] = [
    1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 240,
];

/// Is the denominator of `q` one of the classical `1, 2, 3, 4, 6`, whose
/// values the explicit tables of [`eval_sin`], [`eval_cos`] and [`eval_tan`]
/// give?
pub(super) fn is_classical_denominator(q: &Q) -> bool {
    denominator_divides(q, 4) || denominator_divides(q, 6)
}

/// The denominator of `q` when it is one of `denominators`.
fn listed_denominator(q: &Q, denominators: &[i64]) -> Option<i64> {
    let n = q.denom().to_i64()?;
    denominators.contains(&n).then_some(n)
}

/// `q mod m` (`m > 0`) as a multiple of `1/n`, `n` the denominator of `q`.
fn numerator_mod(q: &Q, m: i64) -> Option<i64> {
    use num_integer::Integer;
    q.numer().mod_floor(&(q.denom() * BigInt::from(m))).to_i64()
}

/// `cos(q·π)` in radicals at the denominators of [`COS_PI_DENOMINATORS`]
/// (SymPy's `cos.eval`); `None` at any other.
pub(super) fn cos_pi_radical(arena: &mut Arena, q: &Q) -> Option<ExprId> {
    let n = listed_denominator(q, &COS_PI_DENOMINATORS)?;
    let p = numerator_mod(q, 2)?;
    Some(cos_pi_reduced(arena, p, n))
}

/// `sin(q·π) = cos((q − 1/2)·π)`: the denominator of `q − 1/2` is in
/// [`COS_PI_DENOMINATORS`] exactly when that of `q` is (`n` and `2n` for an
/// odd `n` of the list, `n` and `n/2` for an even one).
pub(super) fn sin_pi_radical(arena: &mut Arena, q: &Q) -> Option<ExprId> {
    listed_denominator(q, &COS_PI_DENOMINATORS)?;
    let shifted = q - Ratio::new(BigInt::from(1), BigInt::from(2));
    cos_pi_radical(arena, &shifted)
}

/// `cos(pπ/n)` for `0 ≤ p < 2n`, `p/n` in lowest terms and `n` in
/// [`COS_PI_DENOMINATORS`].  The recursion is on the denominator (at most
/// two reflections, then `n` falls to a proper divisor of the table: depth
/// ≤ 12 for `n = 120`).
fn cos_pi_reduced(arena: &mut Arena, p: i64, n: i64) -> ExprId {
    // cos x = −cos(x − π) on (π, 2π), cos x = −cos(π − x) on (π/2, π].
    if p > n {
        let v = cos_pi_reduced(arena, p - n, n);
        return arena.neg(v);
    }
    if 2 * p > n {
        let v = cos_pi_reduced(arena, n - p, n);
        return arena.neg(v);
    }
    let sqrt_of = |arena: &mut Arena, k: i64| {
        let k = arena.int(k);
        let half = arena.rational(1, 2);
        arena.pow(k, half)
    };
    // 0 ≤ p/n ≤ 1/2.
    match n {
        1 => arena.one,
        2 => arena.zero,
        3 => arena.rational(1, 2),
        4 | 6 => {
            let r = sqrt_of(arena, if n == 4 { 2 } else { 3 });
            let half = arena.rational(1, 2);
            arena.mul(&[half, r])
        }
        // T_p((1 + √5)/4) expanded: (1 + √5)/4, (√5 − 1)/4.
        5 => {
            let r5 = sqrt_of(arena, 5);
            let quarter = arena.rational(1, 4);
            let a = arena.rational(if p == 1 { 1 } else { -1 }, 4);
            let b = arena.mul(&[quarter, r5]);
            arena.add(&[a, b])
        }
        // √((1 + cos 2x)/2), positive on [0, π/2]: √(1/2 ± √2/4) at π/8,
        // 3π/8 and √(5/8 ± √5/8) at π/10, 3π/10 (see `sqrt_of_surd`).
        8 | 10 => {
            let r = |p: i64, n: i64| Ratio::new(BigInt::from(p), BigInt::from(n));
            let (a, b, c) = if n == 8 {
                (r(1, 2), r(1, 4), 2)
            } else {
                (r(5, 8), r(1, 8), 5)
            };
            // cos 2x = cos(pπ/(n/2)) is positive exactly when 4p < n.
            let b = if 4 * p < n { b } else { -b };
            sqrt_of_surd(arena, a, b, c)
        }
        // cos(x − y) = cos x·cos y + cos(π/2 − x)·cos(π/2 − y) for
        // x = pπ/a, y = pπ/b, 1/n = 1/a − 1/b.
        _ => {
            let (a, b) = match n {
                12 => (3, 4),
                15 => (6, 10),
                20 => (4, 5),
                24 => (6, 8),
                30 => (5, 6),
                40 => (8, 10),
                60 => (20, 30),
                _ => (40, 60),
            };
            let half = Ratio::new(BigInt::from(1), BigInt::from(2));
            let x = Ratio::new(BigInt::from(p), BigInt::from(a));
            let y = Ratio::new(BigInt::from(p), BigInt::from(b));
            let cos_of = |arena: &mut Arena, r: Q| {
                // Every argument here has a listed denominator (a divisor of
                // `a` or `b`, or twice an odd one).
                cos_pi_radical(arena, &r).unwrap_or_else(|| {
                    let angle = pi_times(arena, r);
                    arena.cos(angle)
                })
            };
            let (cx, cy) = (cos_of(arena, x.clone()), cos_of(arena, y.clone()));
            let (sx, sy) = (cos_of(arena, &half - x), cos_of(arena, &half - y));
            let first = arena.mul(&[cx, cy]);
            let second = arena.mul(&[sx, sy]);
            arena.add(&[first, second])
        }
    }
}

/// `√(a + b·√c)` for rationals `a > |b|·√c`, an integer `c > 1`: as it
/// reads when `b > 0`, and as `√(a² − b²c)/√(a − b·√c)` when `b < 0`
/// (SymPy prints `sqrt(1/2 - sqrt(2)/4)`; here `√2/4/√(√2/4 + 1/2)`).  The
/// radicand of a square root then is a sum of positive terms, which the
/// assumptions see is positive — so the value is known real, `re`/`im`/
/// `|·|` of it decompose (the integrals over the roots of `x⁸ + 1` stay
/// compact), and the root is the same as its partner's (`sin(π/8)` and
/// `cos(π/8)` share `√(√2/4 + 1/2)`), which keeps the zero tests of
/// identities between them small.
fn sqrt_of_surd(arena: &mut Arena, a: Q, b: Q, c: i64) -> ExprId {
    let half = arena.rational(1, 2);
    let c_id = arena.int(c);
    let root_c = arena.pow(c_id, half);
    let positive = |arena: &mut Arena, b: Q| {
        let (a_id, b_id) = (arena.num_ratio(a.clone()), arena.num_ratio(b));
        let b_root = arena.mul(&[b_id, root_c]);
        arena.add(&[a_id, b_root])
    };
    if !b.is_negative() {
        let radicand = positive(arena, b);
        return arena.pow(radicand, half);
    }
    let norm = &a * &a - &b * &b * Ratio::from_integer(BigInt::from(c));
    let norm_id = arena.num_ratio(norm);
    let numerator = arena.pow(norm_id, half);
    let conjugate = positive(arena, -b);
    let minus_half = arena.rational(-1, 2);
    let denominator = arena.pow(conjugate, minus_half);
    arena.mul(&[numerator, denominator])
}

/// `tan(q·π)` in radicals at the denominators of [`TAN_PI_DENOMINATORS`]
/// (SymPy's `tan.eval`); `None` at any other.  The classical denominators
/// are [`eval_tan`]'s.
pub(super) fn tan_pi_radical(arena: &mut Arena, q: &Q) -> Option<ExprId> {
    let n = listed_denominator(q, &TAN_PI_DENOMINATORS)?;
    let p = numerator_mod(q, 1)?;
    if p == 0 {
        return Some(arena.zero);
    }
    if is_classical_denominator(q) {
        let angle = pi_times(arena, q.clone());
        return eval_tan(arena, angle);
    }
    let r = |p: i64, n: i64| Ratio::new(BigInt::from(p), BigInt::from(n));
    match n {
        // SymPy's `table10`: √(1 ∓ 2√5/5), √(5 ∓ 2√5) at π/10, π/5, 3π/10,
        // 2π/5 (the differences as in [`sqrt_of_surd`]).
        5 | 10 => {
            let k = 10 * p / n;
            let (k, sign) = if k > 5 { (10 - k, -1) } else { (k, 1) };
            let (a, b) = match k {
                1 => (r(1, 1), r(-2, 5)),
                2 => (r(5, 1), r(-2, 1)),
                3 => (r(1, 1), r(2, 5)),
                _ => (r(5, 1), r(2, 1)),
            };
            let v = sqrt_of_surd(arena, a, b, 5);
            Some(if sign < 0 { arena.neg(v) } else { v })
        }
        // A rational tangent of a non-classical angle does not exist
        // (Niven), so 15 is the only odd denominator left: tan(x − y) for
        // x = pπ/6, y = pπ/10.
        15 => {
            let ta = tan_pi_radical(arena, &r(p, 6))?;
            let tb = tan_pi_radical(arena, &r(p, 10))?;
            let num = arena.sub(ta, tb);
            let prod = arena.mul(&[ta, tb]);
            let den = arena.add(&[arena.one, prod]);
            Some(arena.div(num, den))
        }
        // tan(x/2) = 1/sin x − cos x/sin x, x = 2pπ/n.
        _ => {
            let x = r(2 * p, n);
            let c = cos_pi_radical(arena, &x)?;
            let s = sin_pi_radical(arena, &x)?;
            if arena.is_zero_structural(s) {
                return Some(arena.complex_infinity);
            }
            let inv_s = arena.pow(s, arena.neg_one);
            let c_over_s = arena.mul(&[c, inv_s]);
            Some(arena.sub(inv_s, c_over_s))
        }
    }
}

/// `cot(q·π)` in radicals at the non-classical denominators of
/// [`TAN_PI_DENOMINATORS`] (SymPy's `cot.eval`): `cot` is the quotient
/// `cos/sin` here, which [`eval_cot_quotients`] folds whole, so that
/// `cot(π/8)` is SymPy's `1 + √2` (not `√(√2/4 + 1/2)/√(1/2 − √2/4)`) and
/// `cot(π/16)` is evaluated as `tan(π/16)` is.
fn cot_pi_radical(arena: &mut Arena, q: &Q) -> Option<ExprId> {
    let n = listed_denominator(q, &TAN_PI_DENOMINATORS)?;
    if is_classical_denominator(q) {
        return None;
    }
    let p = numerator_mod(q, 1)?;
    let r = |p: i64, n: i64| Ratio::new(BigInt::from(p), BigInt::from(n));
    match n {
        // cot x = tan(π/2 − x).
        5 | 10 => tan_pi_radical(arena, &(r(1, 2) - r(p, n))),
        // cot(x − y) = (1 + cot x·cot y)/(cot y − cot x), x = pπ/6, y = pπ/10.
        15 => {
            let ca = cot_pi_classical_or_radical(arena, &r(p, 6))?;
            let cb = cot_pi_classical_or_radical(arena, &r(p, 10))?;
            let prod = arena.mul(&[ca, cb]);
            let num = arena.add(&[arena.one, prod]);
            let den = arena.sub(cb, ca);
            Some(arena.div(num, den))
        }
        // cot(x/2) = 1/sin x + cos x/sin x, x = 2pπ/n.
        _ => {
            let x = r(2 * p, n);
            let c = cos_pi_radical(arena, &x)?;
            let s = sin_pi_radical(arena, &x)?;
            if arena.is_zero_structural(s) {
                return Some(arena.complex_infinity);
            }
            let inv_s = arena.pow(s, arena.neg_one);
            let c_over_s = arena.mul(&[c, inv_s]);
            Some(arena.add(&[inv_s, c_over_s]))
        }
    }
}

/// `cot(q·π)` at any denominator of [`TAN_PI_DENOMINATORS`]: `1/tan` at the
/// classical ones (`cot(π/6) = √3`, `cot(π/3) = √3/3`).
fn cot_pi_classical_or_radical(arena: &mut Arena, q: &Q) -> Option<ExprId> {
    if is_classical_denominator(q) {
        let angle = pi_times(arena, q.clone());
        let t = eval_tan(arena, angle)?;
        if arena.is_zero_structural(t) {
            return Some(arena.complex_infinity);
        }
        return Some(arena.pow(t, arena.neg_one));
    }
    cot_pi_radical(arena, q)
}

/// `cos(u)^k·sin(u)^(−k)` → `cot(u)^k` and `cosh(u)^k·sinh(u)^(−k)` →
/// `coth(u)^k = (−i·cot(u/i))^k` (`k ≥ 1`) among the factors `children` of a
/// product, for `u = qπ` (`u = i·qπ`) with `q` at a non-classical
/// denominator of [`TAN_PI_DENOMINATORS`]: `cot` and `coth` are these
/// quotients, and the quotient of the separately evaluated `cos` and `sin`
/// is not SymPy's `cot.eval` form (or not evaluated at all: `cos(π/16)` and
/// `sin(π/16)` are not, `cot(π/16)` is).  `None` when no pair folds.  The
/// arguments are compared after evaluation (`cache`).
pub(super) fn eval_cot_quotients(
    arena: &mut Arena,
    children: &[ExprId],
    cache: &FxHashMap<ExprId, ExprId>,
) -> Option<ExprId> {
    // (argument, hyperbolic, integer exponent) of a factor `f(u)^k`, `f`
    // one of cos, sin, cosh, sinh.
    let factor = |arena: &Arena, c: ExprId| -> Option<(ExprId, bool, bool, i64)> {
        let (b, k) = match *arena.node(c) {
            ExprNode::Pow(b, e) => {
                let e = arena.as_num(e)?;
                if !e.is_integer() {
                    return None;
                }
                (b, e.to_integer().to_i64()?)
            }
            _ => (c, 1),
        };
        let (u, hyperbolic, cosine) = match *arena.node(b) {
            ExprNode::Cos(u) => (u, false, true),
            ExprNode::Sin(u) => (u, false, false),
            ExprNode::Cosh(u) => (u, true, true),
            ExprNode::Sinh(u) => (u, true, false),
            _ => return None,
        };
        Some((cache.get(&u).copied().unwrap_or(u), hyperbolic, cosine, k))
    };
    if children.len() < 2 || !children.iter().any(|&c| factor(arena, c).is_some()) {
        return None;
    }
    let mut used = vec![false; children.len()];
    let mut values: smallvec::SmallVec<[ExprId; 4]> = smallvec::SmallVec::new();
    for i in 0..children.len() {
        let Some((u, hyperbolic, true, k)) = factor(arena, children[i]) else {
            continue;
        };
        if k < 1 || used[i] {
            continue;
        }
        let partner = (0..children.len()).find(|&j| {
            !used[j] && j != i && factor(arena, children[j]) == Some((u, hyperbolic, false, -k))
        });
        let Some(j) = partner else { continue };
        let q = if hyperbolic {
            as_imaginary_pi_multiple(arena, u)
        } else {
            as_pi_multiple(arena, u)
        };
        let Some(cot) = q.and_then(|q| cot_pi_radical(arena, &q)) else {
            continue;
        };
        let value = if hyperbolic {
            let minus_i = arena.neg(arena.i_unit);
            arena.mul(&[minus_i, cot])
        } else {
            cot
        };
        let kk = arena.int(k);
        values.push(arena.pow(value, kk));
        used[i] = true;
        used[j] = true;
    }
    if values.is_empty() {
        return None;
    }
    for (c, u) in children.iter().zip(&used) {
        if !u {
            values.push(cache.get(c).copied().unwrap_or(*c));
        }
    }
    Some(arena.mul(&values))
}

/// Evaluate `exp(inner)` for known special values.
pub(super) fn eval_exp(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // exp(0) = 1
    if inner == arena.zero {
        return Some(arena.one);
    }
    // exp(1) = E
    if inner == arena.one {
        return Some(arena.e_const);
    }
    // exp(ln w) = w for every w ≠ 0 on the principal branch (the logarithm
    // is defined exactly there), as SymPy folds it.  A symbolic `w` is
    // generic (`x/x = 1` likewise); a constant one must be known non-zero:
    // `ln 0 = zoo` and `exp(zoo)` has no value, and before 0.30
    // `exp(ln((1 + √2)² − 3 − 2√2))` (a zero the canonical form does not
    // see) evaluated to 0.
    if let ExprNode::Ln(w) = *arena.node(inner)
        && (!walk::free_symbols(arena, w).is_empty() || constant_is_nonzero(arena, w))
    {
        return Some(w);
    }
    // exp(k·ln z) = z^k, the principal power by definition.
    if let Some(power) = log_multiple_power(arena, inner) {
        return Some(power);
    }

    // Euler's formula: exp(i*k*π) = cos(kπ) + i*sin(kπ)
    // Detect if inner is i * (something that's a π-multiple)
    // Only at the classical denominators (`e^{iπ/4} = (1 + i)·√2/2`):
    // SymPy keeps `exp(I*pi/5)`, and the radical values of `cos`/`sin` at
    // the other denominators would only make it longer.
    if let Some(pi_coeff) = as_imaginary_pi_multiple(arena, inner)
        && is_classical_denominator(&pi_coeff)
    {
        // We have exp(i * coeff * π)
        // = cos(coeff*π) + i*sin(coeff*π)
        // Build the angle as coeff*π and evaluate sin/cos
        let coeff_id = {
            let nid = arena.intern_num(pi_coeff.clone());
            arena.intern(ExprNode::Num(nid))
        };
        let angle = arena.mul(&[coeff_id, arena.pi]);

        let cos_val = eval_cos(arena, angle);
        let sin_val = eval_sin(arena, angle);

        if let (Some(c), Some(s)) = (cos_val, sin_val) {
            // cos(kπ) + i*sin(kπ)
            if s == arena.zero {
                return Some(c); // Pure real
            }
            if c == arena.zero {
                // Pure imaginary: i*sin(kπ)
                return Some(arena.mul(&[arena.i_unit, s]));
            }
            // General: cos + i*sin
            let i_sin = arena.mul(&[arena.i_unit, s]);
            return Some(arena.add(&[c, i_sin]));
        }
    }

    // exp(i·q·π) with `q` reduced to (−1, 1] (period 2πi), alone or beside
    // other terms, as SymPy's `exp.eval` does (`exp(23πi/9)` →
    // `exp(5πi/9)`, `exp(2 − 17πi/5)` → `exp(2 + 3πi/5)`); up to 0.46 they
    // stayed, and `exp(37πi/8) − exp(−11πi/8)` was not 0.
    if let Some(reduced) = principal_imaginary_pi(arena, inner) {
        return Some(eval_exp(arena, reduced).unwrap_or_else(|| arena.exp(reduced)));
    }

    // exp(w + i·k·π) = exp(w)·exp(i·k·π) for a sum with an imaginary
    // π-multiple term whose exponential folds (k integer or half-integer):
    // exp(ln 2 − 3πi) = −2 exactly.  Otherwise the numeric evaluator meets
    // e^{ln 2}·(cos 3π − i sin 3π) with a rounding residue in the imaginary
    // part whose sign picks the side of the next branch cut (0.24,
    // fuzz_simplify: √((e^{ln(−2)})^{−3}) came out −(√2/4)i at 30 digits).
    if let ExprNode::Add(terms) = arena.node(inner).clone() {
        for (k, &t) in terms.iter().enumerate() {
            if let Some(pi_coeff) = as_imaginary_pi_multiple(arena, t)
                && denominator_divides(&pi_coeff, 2)
                && let Some(unit) = eval_exp(arena, t)
            {
                let rest: smallvec::SmallVec<[ExprId; 6]> = terms
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != k)
                    .map(|(_, &c)| c)
                    .collect();
                let rest = arena.add(&rest);
                let rest_exp = eval_exp(arena, rest).unwrap_or_else(|| arena.exp(rest));
                return Some(arena.mul(&[unit, rest_exp]));
            }
        }
        // exp(Σ kⱼ·ln zⱼ + w) = ∏ zⱼ^kⱼ · exp(w) for real constants `kⱼ`, as SymPy's
        // `exp.eval` splits a sum (`exp(x + 2*log(3))` → `9*exp(x)`,
        // `exp(log(-8)/3)` → `2*exp(I*pi/3)`): `exp(a + b) = exp(a)·exp(b)` on
        // all of ℂ.  A symbolic `zⱼ` is generic ([`log_multiple_power`]:
        // `exp(x + 2·ln y) = y²·eˣ`, as in SymPy).
        let mut powers: smallvec::SmallVec<[ExprId; 4]> = smallvec::SmallVec::new();
        let mut rest: smallvec::SmallVec<[ExprId; 6]> = smallvec::SmallVec::new();
        for &t in &terms {
            match log_multiple_power(arena, t) {
                Some(p) => powers.push(p),
                None => rest.push(t),
            }
        }
        if !powers.is_empty() {
            let rest = arena.add(&rest);
            let rest_exp = eval_exp(arena, rest).unwrap_or_else(|| arena.exp(rest));
            powers.push(rest_exp);
            return Some(arena.mul(&powers));
        }
    }

    None
}

/// `inner` with its term `i·q·π` (`q` rational) moved into `(−1, 1]` by a
/// multiple of 2, when it is outside; `None` otherwise.
fn principal_imaginary_pi(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let terms: smallvec::SmallVec<[ExprId; 6]> = match arena.node(inner) {
        ExprNode::Add(terms) => terms.clone(),
        _ => smallvec::smallvec![inner],
    };
    let two = Q::from_integer(BigInt::from(2));
    for (k, &t) in terms.iter().enumerate() {
        let Some(q) = as_imaginary_pi_multiple(arena, t) else {
            continue;
        };
        // (−1, 1]: q − 2·⌈(q − 1)/2⌉.
        let shift = ((&q - Q::one()) / &two).ceil() * &two;
        if shift.is_zero() {
            return None;
        }
        let r = &q - &shift;
        let r = arena.num_ratio(r);
        let reduced = arena.mul(&[r, arena.i_unit, arena.pi]);
        let rest: smallvec::SmallVec<[ExprId; 6]> = terms
            .iter()
            .enumerate()
            .map(|(j, &c)| if j == k { reduced } else { c })
            .collect();
        return Some(if rest.len() == 1 {
            rest[0]
        } else {
            arena.add(&rest)
        });
    }
    None
}

/// `z^k` for a term `t = k·ln z` of an exponent, `z ≠ 0` (a constant, or
/// nonzero by the assumptions) and `k` a finite real constant (a product of
/// real constant factors, as SymPy's `exp.eval` asks of the cofactor of its
/// one `log`): `exp(k·Log z) = z^k` is the definition of the principal
/// power.  So `exp(2·ln 10) = 100`, `exp(−ln(2)·3/2) = √2/4`, `exp(2·ln(√2))
/// = 2`, `exp(π·ln 2) = 2^π`, `exp(ln(2 + i)/2) = √(2 + i)`, `exp(2·ln x) =
/// x²`; a symbolic or non-real `k` stays (`exp(x·ln 2)`, `exp(i·ln 2)`, as
/// in SymPy).  Up to 0.43 all stayed, and in 0.44 a symbolic `z` stayed
/// (the Gruntz rewrite then mishandled `exp(−ln(1/ω))` folding to `ω`,
/// fixed in 0.44).  A symbolic `z` is generic, as for `exp(ln z) = z` in
/// [`eval_exp`] (and `x/x = 1`): up to 0.46 `exp(2·ln x)` folded to `x²`
/// only for an `x` known nonzero while `exp(ln x)` folded to `x` for every
/// `x` (SymPy folds both); a constant `z` must still be known nonzero (a
/// zero the canonical form does not see).
fn log_multiple_power(arena: &mut Arena, t: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let children = match arena.node(t).clone() {
        ExprNode::Mul(children) => children,
        // A bare `ln z` in a sum (`exp(ln 2 + ln 3) = 6`).
        ExprNode::Ln(z) => return constant_is_nonzero(arena, z).then_some(z),
        _ => return None,
    };
    let mut log_arg = None;
    let mut coeffs: smallvec::SmallVec<[ExprId; 4]> = smallvec::SmallVec::new();
    for &c in &children {
        match *arena.node(c) {
            ExprNode::Ln(z) if log_arg.is_none() => log_arg = Some(z),
            ExprNode::Ln(_) => return None,
            _ => coeffs.push(c),
        }
    }
    let z = log_arg?;
    if walk::free_symbols(arena, z).is_empty() && !constant_is_nonzero(arena, z) {
        return None;
    }
    let k = arena.mul(&coeffs);
    if !walk::free_symbols(arena, k).is_empty()
        || AssumptionCache::new().query(arena, k, Props::FINITE) != Some(true)
        || !known_real(arena, k)
    {
        return None;
    }
    Some(arena.pow(z, k))
}

/// Is `w` provably non-zero: by the assumption engine, or, for a constant
/// (symbol-free) `w`, by a certified 16-digit value whose error ball
/// excludes 0?
pub(super) fn constant_is_nonzero(arena: &Arena, w: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    if AssumptionCache::new().query(arena, w, Props::NONZERO) == Some(true) {
        return true;
    }
    walk::free_symbols(arena, w).is_empty()
        && matches!(
        crate::transforms::evalf::evalf_settled(
            arena,
            w,
            16,
            crate::transforms::evalf::ZeroSearch::Cap
        ),
            Ok((ref z, crate::transforms::evalf::Settled::Certified))
                if !(z.0.is_zero() && z.1.is_zero())
        )
}

/// Check if `id` is of the form `i * k * π` for some rational `k`.
/// Returns `Some(k)` if so, `None` otherwise.
fn as_imaginary_pi_multiple(arena: &mut Arena, id: ExprId) -> Option<Q> {
    // The canonical form of i * k * π is Mul([k, π, I]) or variations
    // (sorted by sort key: Num < Pi < ImaginaryUnit).
    // We need to check if the expression is a product containing exactly
    // one ImaginaryUnit factor and the rest forming a π-multiple.
    let children = match arena.node(id).clone() {
        ExprNode::Mul(children) => children,
        _ => return None,
    };

    let mut has_i = false;
    let mut remaining: Vec<ExprId> = Vec::new();
    for &child in &children {
        if child == arena.i_unit {
            if has_i {
                return None;
            } // multiple i's
            has_i = true;
        } else {
            remaining.push(child);
        }
    }
    if !has_i {
        return None;
    }

    // Build the product of the remaining factors and check if it's a π-multiple
    let product = if remaining.is_empty() {
        // Just i alone — not a π-multiple
        return None;
    } else if remaining.len() == 1 {
        remaining[0]
    } else {
        arena.mul(&remaining)
    };

    as_pi_multiple(arena, product)
}

/// Evaluate `ln(inner)` for known special values.
pub(super) fn eval_ln(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // ln(1) = 0
    if inner == arena.one {
        return Some(arena.zero);
    }
    // ln(E) = 1
    if is_e(arena, inner) {
        return Some(arena.one);
    }

    // ln(-1) = i*π
    if inner == arena.neg_one {
        let i_pi = arena.mul(&[arena.i_unit, arena.pi]);
        return Some(i_pi);
    }

    // ln(negative rational) = ln(|r|) + i*π  (principal branch)
    if let Some(r) = arena.as_num(inner)
        && r.is_negative()
    {
        let abs_r = -r.clone();
        if abs_r.is_one() {
            // Already handled above: ln(-1)
        } else {
            let abs_id = {
                let nid = arena.intern_num(abs_r);
                arena.intern(ExprNode::Num(nid))
            };
            let ln_abs = arena.ln(abs_id);
            let i_pi = arena.mul(&[arena.i_unit, arena.pi]);
            return Some(arena.add(&[ln_abs, i_pi]));
        }
    }

    if let ExprNode::Exp(y) = *arena.node(inner)
        && let Some(v) = ln_of_exp(arena, y)
    {
        return Some(v);
    }

    // ln w = ln(−w) + iπ for any negative real constant `w`, as for the
    // rationals above (SymPy's `log.eval`: `log(-E)` → 1 + I*pi, `log(1 -
    // sqrt(2))` → `log(-1 + sqrt(2)) + I*pi`).  Up to 0.43 they stayed.
    if arena.as_num(inner).is_none() && constant_is_negative(arena, inner) {
        let w = arena.neg(inner);
        let ln_w = eval_ln(arena, w).unwrap_or_else(|| arena.ln(w));
        let i_pi = arena.mul(&[arena.i_unit, arena.pi]);
        return Some(arena.add(&[ln_w, i_pi]));
    }

    ln_of_special_complex(arena, inner)
}

/// Is the finite constant `w` negative: by the assumptions, or by a
/// certified 16-digit value that is real with a negative real part?
fn constant_is_negative(arena: &mut Arena, w: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    if !walk::free_symbols(arena, w).is_empty() {
        return false;
    }
    let fact = crate::base::arena::FactKey::Negative;
    let negative = crate::base::truth::SignSet::NEGATIVE;
    constant_fact(arena, w, fact, negative, |arena| {
        let mut cache = AssumptionCache::new();
        if cache.query(arena, w, Props::FINITE) == Some(false) {
            return false;
        }
        match cache.query(arena, w, Props::NEGATIVE) {
            Some(negative) => negative && cache.query(arena, w, Props::FINITE) == Some(true),
            // The value first: realness may take a minimal polynomial.
            None => {
                matches!(
                    evalf_settled(arena, w, 16, ZeroSearch::Cap),
                    Ok((z, Settled::Certified)) if z.1.is_zero() && z.0.is_negative()
                ) && known_real(arena, w)
            }
        }
    })
}

/// `Log(e^y) = y − 2πik` for a constant `y`, with the integer `k` that puts
/// the imaginary part in `(−π, π]` (SymPy's `log.eval`: `log(exp(5*I))` →
/// `-2*I*pi + 5*I`, `log(exp(2 + 3*I))` → `2 + 3*I`): exactly for an
/// imaginary part `q·π`, else by a 16-digit value away from the odd
/// multiples of `π`.  A real `y` folds when built.  Up to 0.43 they stayed.
fn ln_of_exp(arena: &mut Arena, y: ExprId) -> Option<ExprId> {
    if !walk::free_symbols(arena, y).is_empty() {
        return None;
    }
    if log_exp_is_identity(arena, y) {
        return Some(y);
    }
    let parts = crate::base::complex::decompose(arena, y);
    if !parts.exact {
        return None;
    }
    let im = eval(arena, parts.im);
    let turns = if let Some(q) = as_pi_multiple(arena, im) {
        // q − r for the r ∈ (−1, 1] with q ≡ r (mod 2), in units of 2π.
        let two = Ratio::from_integer(BigInt::from(2));
        let shifted = mod_positive(&(q.clone() + Ratio::one()), &two);
        let r = if shifted.is_zero() {
            Ratio::one()
        } else {
            shifted - Ratio::one()
        };
        (q - r) / two
    } else {
        let v = crate::transforms::evalf::evalf_f64(arena, im).ok()?;
        let tau = 2.0 * std::f64::consts::PI;
        if !v.is_finite() || v.abs() > 1e6 {
            return None;
        }
        let k = (v / tau).round();
        if (v - k * tau).abs() >= std::f64::consts::PI * (1.0 - 1e-9) {
            return None;
        }
        Ratio::from_integer(BigInt::from(k as i64))
    };
    if turns.is_zero() {
        return Some(y);
    }
    let shift = pi_times(arena, turns * Ratio::from_integer(BigInt::from(-2)));
    let i_shift = arena.mul(&[arena.i_unit, shift]);
    Some(arena.add(&[y, i_shift]))
}

/// `Log z = ln|z| + i·Arg z` for a constant `z = a + b·i` (`b ≠ 0`, exact
/// parts built from rationals and square roots) whose argument is a
/// rational multiple of π the special-angle table knows (`tan(Arg z)` = 1,
/// `√3`, `2 − √3`, …; for `a = 0` it is `±π/2`).  SymPy folds these:
/// `log(I)` → `I*pi/2`, `log(1 + I)` → `log(sqrt(2)) + I*pi/4`, `log(1/2 +
/// sqrt(3)*I/2)` → `I*pi/3`.  Before 0.39 they stayed `ln(…)`, and `solve`
/// returned `exp(ix) = i` as `−ln(I)·I`.
fn ln_of_special_complex(arena: &mut Arena, z: ExprId) -> Option<ExprId> {
    if crate::base::complex::is_real_node(arena, z)
        || !crate::base::walk::free_symbols(arena, z).is_empty()
    {
        return None;
    }
    let parts = crate::base::complex::decompose(arena, z);
    if !parts.exact || arena.is_zero_structural(parts.im) {
        return None;
    }
    let (re, im) = (eval(arena, parts.re), eval(arena, parts.im));
    let tower =
        |arena: &Arena, v: ExprId| arena.as_num(v).is_some() || is_square_root_tower(arena, v);
    if !tower(arena, re) || !tower(arena, im) {
        return None;
    }
    let im_sign = crate::poly::algebraic::sign_checked(arena, im)?;
    // `q·π`, the angle of `|b|/|a|` in [0, π/2].
    let (re_sign, q) = if arena.is_zero_structural(re) {
        (0, Ratio::new(BigInt::from(1), BigInt::from(2)))
    } else {
        let re_sign = crate::poly::algebraic::sign_checked(arena, re)?;
        let ratio = arena.div(im, re);
        let ratio = eval(arena, ratio);
        let q = match arena.as_num(ratio) {
            // Niven: a rational tangent of a rational multiple of π is 0 or ±1.
            Some(r) if r.abs().is_one() => Ratio::new(BigInt::from(1), BigInt::from(4)),
            Some(_) => return None,
            None => special_angle_of(arena, ratio, SpecialTable::Tan)?.1,
        };
        (re_sign, q)
    };
    if im_sign == 0 || (re_sign != 0 && q.is_zero()) {
        return None;
    }
    let pi_q = if re_sign >= 0 { q } else { Ratio::one() - q };
    let arg = if im_sign > 0 { pi_q } else { -pi_q };
    let i_arg = {
        let t = pi_times(arena, arg);
        arena.mul(&[arena.i_unit, t])
    };
    // `ln|z| = ln √(a² + b²)`, the squares expanded: on the unit circle
    // `a² + b²` is 1 and the logarithm 0 (`log(cos(7π/12) + i·sin(7π/12))`
    // → `7πi/12`, as SymPy gives it; up to 0.46 `ln √((√2/4 − √6/4)² +
    // …)` stayed beside it).
    let two = arena.int(2);
    let (re2, im2) = (arena.pow(re, two), arena.pow(im, two));
    let m2 = arena.add(&[re2, im2]);
    let m2 = crate::transforms::expand::expand(arena, m2);
    let m2 = eval(arena, m2);
    let half = arena.rational(1, 2);
    let modulus = arena.pow(m2, half);
    let modulus = eval(arena, modulus);
    let ln_abs = arena.ln(modulus);
    let out = arena.add(&[ln_abs, i_arg]);
    Some(eval(arena, out))
}

/// `(e^f)^g = e^(f·g)` on the principal branch?  Always for an integer
/// `g`; for any `g` when `f` is real (then `e^f > 0`, so `ln(e^f) = f`).
/// In general it needs `Im f ∈ (−π, π]`: `√(e^{4i}) = −e^{2i}`.  Before
/// 0.23 the merge was unconditional, which also corrupted `eval_decimal`
/// of such an expression (it calls `eval` first).
pub(super) fn exp_pow_merges(arena: &Arena, f: ExprId, g: ExprId) -> bool {
    if arena.as_num(g).is_some_and(|r| r.is_integer()) {
        return true;
    }
    let mut cache = crate::base::assumptions::AssumptionCache::new();
    cache.query(arena, g, crate::base::assumptions::Props::INTEGER) == Some(true)
        || cache.query(arena, f, crate::base::assumptions::Props::REAL) == Some(true)
}

/// Evaluate `Pow(base, 1/n)` for a rational `base` and an integer `n ≥ 2`:
/// the perfect `n`-th power factor of a positive base comes out
/// ([`positive_root`]), and a negative base takes the principal branch.
pub(super) fn eval_pow_root(arena: &mut Arena, base: ExprId, exp: ExprId) -> Option<ExprId> {
    if arena.as_num(base).is_none() {
        return negative_constant_power(arena, base, exp);
    }
    let base_r = arena.as_num(base)?.clone();
    let exp_r = arena.as_num(exp)?;

    // (−r)^q = r^q·(−1)^q for a negative rational base and any non-integer
    // rational q on the principal branch (arg(−r) = π), each factor in its
    // normal form (SymPy: `(-8)**(7/3)` → `128*(-1)**(1/3)`, `(-8)**(-1/3)`
    // → `-(-1)**(2/3)/2`).  Up to 0.44 only `(−r)^(1/n)` was split, and
    // `(−8)^(7/3)` stayed.
    if !exp_r.numer().is_one() {
        if exp_r.is_integer() || !base_r.is_negative() || base_r == -Q::one() {
            return None;
        }
        let abs_id = arena.num_ratio(-base_r);
        let magnitude = arena.pow(abs_id, exp);
        let unit = arena.pow(arena.neg_one, exp);
        return Some(arena.mul(&[magnitude, unit]));
    }
    let n: u32 = exp_r.denom().to_u32()?;
    if n < 2 {
        return None;
    }

    // Roots of negative rationals take the principal branch, like every
    // other evaluator: (−r)^(1/k) = r^(1/k)·(−1)^(1/k), exact because
    // arg(−r) = π (so ∛(−8) = 2·(−1)^(1/3) = 1 + √3·i, SymPy's
    // `2*(-1)**(1/3)`).  Before 0.23 odd roots of negative integers folded to
    // the real root (∛(−8) → −2) while `evalf` took the principal one; the
    // real root is `Ex::real_root`.
    if base_r.is_negative() && base_r != -Ratio::from_integer(BigInt::from(1)) {
        let abs_r = -base_r;
        // Before 0.29 `r^(1/k)` was built and handed to a fresh `eval`
        // pass, which extracted its radical again (through this function).
        let abs_root = positive_root(arena, &abs_r, exp, n).unwrap_or_else(|| {
            let abs_id = arena.num_ratio(abs_r);
            arena.pow(abs_id, exp)
        });
        let neg_one_root = arena.pow(arena.neg_one, exp);
        return Some(arena.mul(&[abs_root, neg_one_root]));
    }
    positive_root(arena, &base_r, exp, n)
}

/// `w^q = (−w)^q·(−1)^q` for a negative real constant `w` that is not a
/// rational and a non-integer rational `q`, on the principal branch
/// (`arg w = π`):
///
/// * a product with a negative coefficient that the assumptions know
///   negative, as SymPy's `Mul._eval_power` (`sqrt(-pi)` → `I*sqrt(pi)`,
///   `(-pi)**(1/3)` → `(-1)**(1/3)*pi**(1/3)`);
/// * a constant whose realness the assumptions leave open (it has non-real
///   parts, as the real root of a cubic by Cardano's formula) and which is
///   real ([`known_real`]: its minimal polynomial) and certified negative.
///   `evalf` cannot take `√w` or `ln w` on the cut when the imaginary part
///   of `w` cancels only exactly; `(−w)^q` it can.  Up to 0.44 `√λ` of
///   such an eigenvalue was refused (`PrecisionExhausted`).
///
/// A real sum such as `1 − √2` keeps its power, as in SymPy.
///
/// A product with a rational coefficient `c`, `|c| ≠ 1`, whose rest
/// `sgn(c)·z` is not known positive gives up its magnitude first, `(c·z)^q
/// = |c|^q·(sgn(c)·z)^q` (scaling by `|c| > 0` keeps the argument):
/// `(2i)^(7/3) = −4·∛2·∛i`, as SymPy writes it (`√(2π)` keeps its
/// form).
fn negative_constant_power(arena: &mut Arena, w: ExprId, exp: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    if arena.as_num(exp)?.is_integer() || !walk::free_symbols(arena, w).is_empty() {
        return None;
    }
    let mut cache = AssumptionCache::new();
    if let ExprNode::Mul(cs) = arena.node(w)
        && let Some(c) = cs.first().and_then(|&c| arena.as_num(c)).cloned()
        && !c.abs().is_one()
    {
        let magnitude = arena.num_ratio(c.abs());
        let rest = arena.div(w, magnitude);
        if cache.query(arena, rest, Props::POSITIVE) == Some(true) {
            return None;
        }
        let outer = arena.pow(magnitude, exp);
        let inner =
            negative_constant_power(arena, rest, exp).unwrap_or_else(|| arena.pow(rest, exp));
        return Some(arena.mul(&[outer, inner]));
    }

    let negative = match arena.node(w) {
        ExprNode::Mul(cs) => {
            cs.first()
                .is_some_and(|&c| arena.as_num(c).is_some_and(Signed::is_negative))
                && cache.query(arena, w, Props::NEGATIVE) == Some(true)
                && cache.query(arena, w, Props::FINITE) == Some(true)
        }
        ExprNode::Add(_) => {
            cache.query(arena, w, Props::REAL).is_none() && constant_is_negative(arena, w)
        }
        _ => false,
    };
    if !negative {
        return None;
    }
    let minus_w = arena.neg(w);
    let magnitude = arena.pow(minus_w, exp);
    let unit = arena.pow(arena.neg_one, exp);
    Some(arena.mul(&[magnitude, unit]))
}

/// `r^(1/n)` (`exp` is the node `1/n`) for a non-negative rational `r`, when
/// it simplifies: `0`, `1`, a perfect power (an exact rational), or an
/// integer `k^n·m` with `k > 1` (`k·m^(1/n)`), each factor split once by
/// [`crate::base::canon::split_perfect_power`] (which recognises exact
/// powers by an integer root first).  `None` otherwise.
fn positive_root(arena: &mut Arena, r: &Q, exp: ExprId, n: u32) -> Option<ExprId> {
    use crate::base::canon::split_perfect_power;
    if r.is_zero() {
        return Some(arena.zero);
    }
    if r.is_one() {
        return Some(arena.one);
    }
    if r.is_negative() {
        return None;
    }
    let (kn, mn) = split_perfect_power(r.numer(), n);
    if r.is_integer() {
        if mn.is_one() {
            return Some(arena.big_int(kn)); // perfect power
        }
        if kn.is_one() {
            return None;
        }
        // n^(1/q) = k · m^(1/q)
        let k_id = arena.big_int(kn);
        let m_id = arena.big_int(mn);
        let root = arena.pow(m_id, exp);
        return Some(arena.mul(&[k_id, root]));
    }
    // p/q: only a perfect power of both folds.
    if !mn.is_one() {
        return None;
    }
    let (kd, md) = split_perfect_power(r.denom(), n);
    if !md.is_one() {
        return None;
    }
    Some(arena.num_ratio(Ratio::new(kn, kd)))
}

/// Evaluate `abs(inner)` for known numeric values.
pub(super) fn eval_abs(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // abs(i) = 1
    if inner == arena.i_unit {
        return Some(arena.one);
    }
    if let Some(r) = arena.as_num(inner) {
        let r = r.clone();
        if r.is_negative() {
            let pos = -r;
            let nid = arena.intern_num(pos);
            return Some(arena.intern(ExprNode::Num(nid)));
        } else {
            return Some(inner);
        }
    }

    // `|e^z| = e^(Re z)` on all of ℂ (SymPy's `Abs.eval` takes
    // `exp(re(arg))`): `|exp(i)| = 1`.  Before 0.35 it stayed `abs(exp(i))`,
    // and the norm of the matrix `[[exp(i)]]` printed `√(sin²1 + cos²1)`.
    // For a constant exponent whose real part is exact.
    if let ExprNode::Exp(z) = *arena.node(inner)
        && crate::base::walk::free_symbols(arena, z).is_empty()
    {
        let parts = crate::base::complex::decompose(arena, z);
        if parts.exact {
            let re = eval(arena, parts.re);
            let e = arena.exp(re);
            return Some(eval(arena, e));
        }
    }

    // Complex modulus of a numeric constant: |a + b·i| = √(a² + b²) whenever
    // the real/imaginary decomposition is exact, both parts are arithmetic
    // constants (numbers, π, e, … combined with + − × ^, so radicals are
    // included) and the imaginary part is structurally non-zero:
    // |3 + 4i| = 5, |1 + i| = √2, |1 + √3·i| = 2, |e + πi| = √(e² + π²).
    // Real constants are left to the sign-aware simplifier (`√(π²)` would
    // be a step backwards), as are transcendental parts (|e^{i}| stays put
    // rather than becoming √(sin²1 + cos²1)).
    if !crate::base::complex::is_real_node(arena, inner)
        && crate::base::walk::free_symbols(arena, inner).is_empty()
    {
        let parts = crate::base::complex::decompose(arena, inner);
        if parts.exact
            && parts.im != arena.zero
            && is_arithmetic_constant(arena, parts.re)
            && is_arithmetic_constant(arena, parts.im)
        {
            tracing::trace!("eval_abs: folding modulus of a numeric complex constant");
            let two = arena.int(2);
            let re2 = arena.pow(parts.re, two);
            let im2 = arena.pow(parts.im, two);
            let sum = arena.add(&[re2, im2]);
            let sum = eval(arena, sum);
            let half = arena.rational(1, 2);
            return Some(arena.pow(sum, half));
        }
    }

    // `|c·t| = |c|·|t|` for a rational `c`: the canonical `Abs` takes the
    // coefficient out, and the rules above apply to `t` (before 0.41
    // `eval(|−e^π|)` stopped at `|e^π|`, which a second `eval` folded).
    let (coeff, term) = arena.as_coeff_term(inner);
    if term != inner && !coeff.is_zero() && !coeff.is_one() {
        let t = eval_abs(arena, term)?;
        return Some(arena.make_coeff_term(coeff.abs(), t));
    }

    None
}

/// True when `id` is built only from numbers, named real constants and
/// `Add`/`Mul`/`Neg`/`Pow` — i.e. an exact constant whose square is again
/// such a constant (`3`, `√3`, `π/2`, `1 + √2`), with no symbols and no
/// transcendental function applications.
fn is_arithmetic_constant(arena: &Arena, id: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, id)
        .iter()
        .all(|&n| {
            matches!(
                arena.node(n),
                ExprNode::Num(_)
                    | ExprNode::Pi
                    | ExprNode::E
                    | ExprNode::EulerGamma
                    | ExprNode::Catalan
                    | ExprNode::GoldenRatio
                    | ExprNode::Add(_)
                    | ExprNode::Mul(_)
                    | ExprNode::Neg(_)
                    | ExprNode::Pow(_, _)
            )
        })
}

/// `sign(inner)` for the evaluated argument `inner` (`original` before
/// evaluation, which the exact zero tests also try, see [`Difference`]).
pub(super) fn eval_sign(arena: &mut Arena, inner: ExprId, original: ExprId) -> Option<ExprId> {
    if let Some(r) = arena.as_num(inner) {
        use num_traits::{Signed, Zero};
        if r.is_positive() {
            return Some(arena.one);
        }
        if r.is_negative() {
            return Some(arena.neg_one);
        }
        if r.is_zero() {
            return Some(arena.zero);
        }
    }
    // A real constant of certified sign (SymPy: `sign(1 - sqrt(2))` → -1);
    // up to 0.44 only rationals folded, and `sign(λ)` of a negative root
    // by Cardano's formula stayed though `λ < 0` was decided.
    match real_constant_sign(arena, inner) {
        Some(std::cmp::Ordering::Greater) => Some(arena.one),
        Some(_) => Some(arena.neg_one),
        // A constant that is exactly 0 without being written 0: `sign(cos(π/16)
        // − √(2 + √(2 + √2))/2)` stayed up to 0.44.  The exact tests run
        // only for a value that is 0 to the precision of the evaluation; they
        // are those of `Eq(inner, 0)` ([`constant_zero_decision`]): up to
        // 0.45 the minimal polynomial and the arctangent sums were missing
        // here, and `sign(atan(1/2) + atan(1/3) − π/4)` stayed.
        None => {
            let zero = match constant_zero_decision(arena, inner) {
                None if original != inner && walk::free_symbols(arena, original).is_empty() => {
                    exact_zero_test(arena, original)
                }
                decided => decided,
            };
            (zero == Some(true)).then_some(arena.zero)
        }
    }
}

/// The sign of the symbol-free constant `x` when it is real by
/// [`known_real`] (the assumptions, a certified exact zero imaginary part,
/// or its minimal polynomial) and its certified value settles nonzero
/// ([`certified_real_sign`]); `None` otherwise.
pub(super) fn real_constant_sign(arena: &mut Arena, x: ExprId) -> Option<std::cmp::Ordering> {
    use crate::base::assumptions::{AssumptionCache, Props};
    // A certified value is finite: finiteness the assumptions do not know
    // (`W₋₁(−ln(2)/(5·⁵√2))`) does not keep the sign open (up to 0.45
    // `sign` stayed there while `eval` decided `x < 0`).
    if !walk::free_symbols(arena, x).is_empty()
        || AssumptionCache::new().query(arena, x, Props::FINITE) == Some(false)
    {
        return None;
    }
    // The value first: realness may take a minimal polynomial.
    let sign = certified_real_sign(arena, x)?;
    known_real(arena, x).then_some(sign)
}

/// The signs of the symbol-free constant `x`, decided on its evaluated form
/// `v = eval(x)` as `eval` decides relations (`W(√3·e^{√3}) − √3` is `0`):
/// the certified 16-digit value of `v` first (a nonzero imaginary part, or
/// a nonzero real part and realness by [`known_real`]), the exact zero
/// tests of `Eq(v, 0)` when the value is 0 to its precision.  Memoised per
/// arena ([`FactKey::Class`](crate::base::arena::FactKey::Class)).
///
/// One of eight sets: `POSITIVE`, `NEGATIVE` (real by [`known_real`], with
/// a certified value of that sign), `ZERO` (exactly 0, structurally or by
/// [`exact_zero_test`]), `NONREAL` (a certified nonzero imaginary part),
/// `NONZERO_REAL` (real and not 0 by an exact test, sign not decided),
/// `NONZERO` (a certified nonzero value or an exact test, realness not
/// decided), `REAL` (nothing else decided) and `UNKNOWN`.
fn constant_class(arena: &mut Arena, x: ExprId) -> crate::base::truth::SignSet {
    use crate::base::arena::FactKey;
    use crate::base::truth::SignSet;
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    if let Some(class) = arena.constant_fact(x, FactKey::Class) {
        return class;
    }
    let v = eval(arena, x);
    let class = if arena.is_zero_structural(v) {
        SignSet::ZERO
    } else {
        let value = evalf_settled(arena, v, 16, ZeroSearch::Cap);
        let (nonzero, nonreal) = match &value {
            Ok((z, Settled::Certified)) => (!(z.0.is_zero() && z.1.is_zero()), !z.1.is_zero()),
            _ => (false, false),
        };
        if nonreal {
            SignSet::NONREAL
        } else if nonzero {
            match (known_real(arena, v), value) {
                (true, Ok((z, _))) if z.0.is_negative() => SignSet::NEGATIVE,
                (true, _) => SignSet::POSITIVE,
                (false, _) => SignSet::NONZERO,
            }
        } else {
            // A real is finite: a value 0 to its precision is, but without a
            // value realness by [`known_real`] (an imaginary part 0 wherever
            // defined) is no claim unless the assumptions know `v` finite.
            // Up to 0.50 `1/max(Γ(1/3)·Γ(2/3) − 2π/√3, 0)`, a `1/0` that
            // `eval` keeps, was real.
            let finite = value.is_ok() || {
                use crate::base::assumptions::{AssumptionCache, Props};
                use crate::base::truth::Truth;
                Truth::from(AssumptionCache::new().query(arena, v, Props::FINITE)).is_true()
            };
            let real = finite && known_real(arena, v);
            let zero = exact_zero_test(arena, v)
                .or_else(|| (v != x).then(|| exact_zero_test(arena, x)).flatten());
            match (zero, real) {
                (Some(true), _) => SignSet::ZERO,
                (Some(false), true) => SignSet::NONZERO_REAL,
                (Some(false), false) => SignSet::NONZERO,
                (None, true) => SignSet::REAL,
                (None, false) => SignSet::UNKNOWN,
            }
        }
    };
    arena.remember_constant_fact(x, FactKey::Class, class);
    class
}

/// The sign, zero and realness facts `eval` decides about the symbol-free
/// constant `x` (empty for an expression with symbols or one known not to
/// be finite): what the assumption queries of the API add to the
/// structural rules of [`AssumptionCache`](crate::base::assumptions::AssumptionCache)
/// for constants, so that `cos(π/16).is_positive()` agrees with `eval` of
/// `cos(π/16) > 0` (both `True`; the query was `None` up to 0.45), and
/// `(cos(π/16) − √(2 + √(2 + √2))/2).is_zero()` with `Eq` and `sign`.
///
/// A certified 16-digit value decides the sign of a constant real by
/// [`known_real`] and shows a nonzero imaginary part; a value 0 to its
/// precision goes to the exact zero tests of `Eq(x, 0)`
/// ([`exact_zero_test`]).  The work is memoised per arena
/// ([`constant_class`], [`Arena::constant_facts`]) and nothing recurses
/// over `x`; the structural cache is left alone, so a query answers the
/// same whatever was asked before.  The facts are closed under the
/// implication rules.
pub(crate) fn constant_facts(
    arena: &mut Arena,
    x: ExprId,
) -> crate::base::assumptions::Assumptions {
    use crate::base::assumptions::{AssumptionCache, Assumptions, Props};
    let mut facts = Assumptions::default();
    if !walk::free_symbols(arena, x).is_empty()
        || AssumptionCache::new().query(arena, x, Props::FINITE) == Some(false)
    {
        return facts;
    }
    use crate::base::truth::SignSet;
    match constant_class(arena, x) {
        SignSet::POSITIVE => facts.known_true |= Props::REAL | Props::POSITIVE,
        SignSet::NEGATIVE => facts.known_true |= Props::REAL | Props::NEGATIVE,
        SignSet::ZERO => facts.known_true |= Props::ZERO,
        SignSet::NONREAL => {
            facts.known_true |= Props::NONZERO | Props::COMPLEX;
            facts.known_false |= Props::REAL | Props::EXTENDED_REAL;
        }
        SignSet::NONZERO_REAL => facts.known_true |= Props::NONZERO | Props::REAL,
        SignSet::NONZERO => facts.known_true |= Props::NONZERO,
        SignSet::REAL => facts.known_true |= Props::REAL,
        // `UNKNOWN`; [`constant_class`] gives no other set.
        _ => {}
    }
    facts.forward_chain();
    facts
}

pub(super) fn eval_asin(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // asin(0) = 0
    if inner == arena.one {
        // asin(1) = π/2
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, arena.pi]));
    }
    if inner == arena.neg_one {
        // asin(-1) = -π/2
        let neg_half = arena.rational(-1, 2);
        return Some(arena.mul(&[neg_half, arena.pi]));
    }
    // asin(1/2) = π/6
    if let Some(r) = arena.as_num(inner) {
        let half = Ratio::new(1.into(), 2.into());
        if *r == half {
            let sixth = arena.rational(1, 6);
            return Some(arena.mul(&[sixth, arena.pi]));
        }
        let neg_half = Ratio::new((-1).into(), 2.into());
        if *r == neg_half {
            let neg_sixth = arena.rational(-1, 6);
            return Some(arena.mul(&[neg_sixth, arena.pi]));
        }
    }
    if let Some(v) = inverse_of_trig_at_pi_multiple(arena, inner, InverseTrig::Asin) {
        return Some(v);
    }
    // asin(i·c) = i·asinh(c) on all of ℂ (SymPy's `asin.eval`): asin(±i) =
    // ±i·ln(1 + √2), and `acsc(±i) = asin(∓i)`.  Up to 0.43 they stayed.
    if let Some(c) = imaginary_real_cofactor(arena, inner) {
        let v = eval_asinh(arena, c)?;
        return Some(arena.mul(&[arena.i_unit, v]));
    }
    // asin(v) = π/2 − acos(v) = s·(1/2 − q)·π for |v| = cos(qπ).
    if let Some((s, q)) = special_angle_of(arena, inner, SpecialTable::Cos) {
        let half = Ratio::new(BigInt::from(1), BigInt::from(2));
        return Some(pi_times(arena, (half - q) * s));
    }
    let pos = negated_constant(arena, inner)?;
    let v = eval_asin(arena, pos).unwrap_or_else(|| arena.asin(pos));
    Some(arena.neg(v))
}

pub(super) fn eval_acos(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.one {
        return Some(arena.zero);
    } // acos(1) = 0
    if inner == arena.zero {
        // acos(0) = π/2
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, arena.pi]));
    }
    if inner == arena.neg_one {
        return Some(arena.pi);
    } // acos(-1) = π
    // acos(1/2) = π/3
    if let Some(r) = arena.as_num(inner) {
        let half = Ratio::new(1.into(), 2.into());
        if *r == half {
            let third = arena.rational(1, 3);
            return Some(arena.mul(&[third, arena.pi]));
        }
        let neg_half = Ratio::new((-1).into(), 2.into());
        if *r == neg_half {
            // acos(-1/2) = 2π/3
            let two_thirds = arena.rational(2, 3);
            return Some(arena.mul(&[two_thirds, arena.pi]));
        }
    }
    if let Some(v) = inverse_of_trig_at_pi_multiple(arena, inner, InverseTrig::Acos) {
        return Some(v);
    }
    // acos(i·c) = π/2 − i·asinh(c) (`acos = π/2 − asin`; SymPy: `acos(I)` →
    // pi/2 - I*log(1 + sqrt(2))), and `asec(±i) = acos(∓i)`.
    if let Some(c) = imaginary_real_cofactor(arena, inner) {
        let v = eval_asinh(arena, c)?;
        let i_v = arena.mul(&[arena.i_unit, v]);
        let minus_i_v = arena.neg(i_v);
        let quarter_turn = half_pi(arena);
        return Some(arena.add(&[quarter_turn, minus_i_v]));
    }
    // acos(±cos(qπ)) = qπ or (1 − q)π (principal range [0, π]).
    if let Some((s, q)) = special_angle_of(arena, inner, SpecialTable::Cos) {
        let angle = if s.is_negative() { Ratio::one() - q } else { q };
        return Some(pi_times(arena, angle));
    }
    // acos(−u) = π − acos(u) on all of ℂ, used when acos(u) folds (SymPy's
    // `acos.eval` for `−cos`, `−sin`): `acos(−cos(3π/80)) = 77π/80`,
    // `acos(−sin(9π/22)) = 10π/11`.  Since 0.50 `eval` reduces `cos(77π/80)`
    // to `−cos(3π/80)` first, and `acos(cos(77π/80))`, `asec(sec(5π/9))`
    // stayed unevaluated (0.49: `77π/80`, `5π/9`).
    let pos = negated_constant(arena, inner)?;
    let v = eval_acos(arena, pos)?;
    let minus_v = arena.neg(v);
    Some(arena.add(&[arena.pi, minus_v]))
}

/// The value at `c = 0` of the reciprocal function `name` (`acot`, `acoth`,
/// `asech`), whose node is `atan`, `atanh`, `acosh` of `1/c`
/// ([`reciprocal_function`](crate::base::canon::reciprocal_function)):
/// `acot(0) = π/2`, `acoth(0) = iπ/2`, `asech(0) = ∞`, when the argument
/// `original` is written as a reciprocal `k·∏ bⱼ^(−mⱼ)` and evaluated to
/// `zoo` because `c = k⁻¹·∏ bⱼ^mⱼ` evaluated to 0 (the bases' values are
/// in `cache`).  `reciprocal_function` takes these values for a literal
/// `0`, and `subs` for a substituted one; up to 0.49 a `c` that became 0
/// only here was lost: `acot(min(W(π), 0))` was `atan(zoo)` (`compile`:
/// NaN), `asech(min(W(π), 0))` was `zoo`.  `None` anywhere else.
pub(super) fn reciprocal_at_zero(
    arena: &mut Arena,
    name: &str,
    original: ExprId,
    evaluated: ExprId,
    cache: &FxHashMap<ExprId, ExprId>,
) -> Option<ExprId> {
    if ValueClass::special_atom(arena, evaluated) != Some(ValueClass::ComplexInf) {
        return None;
    }
    let factors: Vec<ExprId> = match arena.node(original) {
        ExprNode::Mul(fs) => fs.to_vec(),
        _ => vec![original],
    };
    let mut c_factors: Vec<ExprId> = Vec::with_capacity(factors.len());
    for (i, &f) in factors.iter().enumerate() {
        if i == 0
            && let Some(k) = arena.as_num(f)
        {
            if k.is_zero() {
                return None;
            }
            let inv = k.recip();
            c_factors.push(arena.num_ratio(inv));
            continue;
        }
        let ExprNode::Pow(b, m) = *arena.node(f) else {
            return None;
        };
        if !arena.as_num(m).is_some_and(Signed::is_negative) {
            return None;
        }
        let b = cache.get(&b).copied().unwrap_or(b);
        let m = arena.neg(m);
        c_factors.push(arena.pow(b, m));
    }
    let c = arena.mul(&c_factors);
    let zero = arena.zero;
    (c == zero).then(|| crate::base::canon::reciprocal_function(arena, name, zero))
}

pub(super) fn eval_atan(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // atan(0) = 0
    // atan(±∞) = ±π/2 (SymPy: `atan(oo)` → pi/2).  Before 0.30 it stayed,
    // and `evalf` refused it (an infinite argument).
    // At `±∞ + c` too (any finite `c`), and at `c ± i·∞` it is ±π/2 by the
    // half plane, `sign(Re c)·π/2`, SymPy's `atan(±oo*I) = ±pi/2` on the
    // cut (mpmath `atan(mpc(-1, 1e20))` = −1.5707…).
    if let Some(inf) = crate::base::canon::infinite_argument(arena, inner) {
        let positive = if inf.imaginary {
            match inf.side? {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => !inf.negative,
            }
        } else {
            !inf.negative
        };
        let half = arena.rational(if positive { 1 } else { -1 }, 2);
        return Some(arena.mul(&[half, arena.pi]));
    }
    if inner == arena.one {
        // atan(1) = π/4
        let quarter = arena.rational(1, 4);
        return Some(arena.mul(&[quarter, arena.pi]));
    }
    if inner == arena.neg_one {
        // atan(-1) = -π/4
        let neg_quarter = arena.rational(-1, 4);
        return Some(arena.mul(&[neg_quarter, arena.pi]));
    }
    if let Some(v) = inverse_of_trig_at_pi_multiple(arena, inner, InverseTrig::Atan) {
        return Some(v);
    }
    // atan(±tan(qπ)) = ±qπ (principal range (−π/2, π/2)).
    if let Some((s, q)) = special_angle_of(arena, inner, SpecialTable::Tan) {
        return Some(pi_times(arena, q * s));
    }
    let pos = negated_constant(arena, inner)?;
    let v = eval_atan(arena, pos).unwrap_or_else(|| arena.atan(pos));
    Some(arena.neg(v))
}

/// Which inverse function [`inverse_of_trig_at_pi_multiple`] applies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum InverseTrig {
    Asin,
    Acos,
    Atan,
}

/// The principal value of `asin`, `acos` or `atan` at `sin(qπ)`, `cos(qπ)`
/// or (for `atan`) `tan(qπ)` with `q` rational, an application `eval` kept
/// (`q` off SymPy's denominators): `asin(sin(7π/102)) = 7π/102`,
/// `acos(cos(14π/9)) = 4π/9`, `asin(cos(49π/85)) = −13π/170`,
/// `atan(tan(−34π/43)) = 9π/43` — SymPy's `asin.eval`, `acos.eval`,
/// `atan.eval` rules for a trigonometric argument (there for any comparable
/// angle, here for rational multiples of `π`, exactly).  Before 0.43 they
/// stayed.
fn inverse_of_trig_at_pi_multiple(
    arena: &mut Arena,
    inner: ExprId,
    which: InverseTrig,
) -> Option<ExprId> {
    // `cot` is `cos/sin` or `1/tan` here (`acot(tan x) = atan(1/tan x)`):
    // atan(cot(qπ)) = atan(tan((1/2 − q)π)).
    let cot_argument = |arena: &Arena| -> Option<ExprId> {
        match arena.node(inner) {
            ExprNode::Pow(b, e) if *e == arena.neg_one => match *arena.node(*b) {
                ExprNode::Tan(u) => Some(u),
                _ => None,
            },
            ExprNode::Mul(cs) if cs.len() == 2 => {
                let (c, s) = (arena.node(cs[0]), arena.node(cs[1]));
                match (c, s) {
                    (ExprNode::Cos(u), ExprNode::Pow(b, e))
                    | (ExprNode::Pow(b, e), ExprNode::Cos(u))
                        if *e == arena.neg_one
                            && matches!(*arena.node(*b), ExprNode::Sin(v) if v == *u) =>
                    {
                        Some(*u)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    };
    let (u, f, cot) = match *arena.node(inner) {
        ExprNode::Sin(u) => (u, InverseTrig::Asin, false),
        ExprNode::Cos(u) => (u, InverseTrig::Acos, false),
        ExprNode::Tan(u) if which == InverseTrig::Atan => (u, InverseTrig::Atan, false),
        _ if which == InverseTrig::Atan => (cot_argument(arena)?, InverseTrig::Atan, true),
        _ => return None,
    };
    let q = as_pi_multiple(arena, u)?;
    let q = if cot {
        Ratio::new(BigInt::from(1), BigInt::from(2)) - q
    } else {
        q
    };
    let one = Q::one();
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    // asin(sin(qπ)) in [−π/2, π/2] and acos(cos(qπ)) in [0, π], in units of π.
    let asin_of_sin = |q: &Q| {
        let mut r = mod_positive(q, &Ratio::from_integer(BigInt::from(2)));
        if r > one {
            r = &one - r;
        }
        if r > half {
            r = &one - r;
        } else if r < -half.clone() {
            r = -&one - r;
        }
        r
    };
    let acos_of_cos = |q: &Q| {
        let two = Ratio::from_integer(BigInt::from(2));
        let r = mod_positive(q, &two);
        if r > one { two - r } else { r }
    };
    let r = match (which, f) {
        (InverseTrig::Atan, InverseTrig::Atan) => {
            let r = mod_positive(&q, &one);
            // A pole of `tan` (only reached through `cot`, at a zero of `sin`).
            if r == half {
                return None;
            }
            if r > half { r - &one } else { r }
        }
        (InverseTrig::Asin, InverseTrig::Asin) => asin_of_sin(&q),
        (InverseTrig::Acos, InverseTrig::Acos) => acos_of_cos(&q),
        // asin(x) + acos(x) = π/2.
        (InverseTrig::Asin, InverseTrig::Acos) => &half - acos_of_cos(&q),
        (InverseTrig::Acos, InverseTrig::Asin) => &half - asin_of_sin(&q),
        _ => return None,
    };
    Some(pi_times(arena, r))
}

/// `r·π` as a canonical node.
fn pi_times(arena: &mut Arena, r: Q) -> ExprId {
    if r.is_zero() {
        return arena.zero;
    }
    let c = arena.num_ratio(r);
    arena.mul(&[c, arena.pi])
}

/// Which trigonometric function a [`SPECIAL_COS`] / [`SPECIAL_TAN`] lookup
/// inverts.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SpecialTable {
    Cos,
    Tan,
}

/// A value `(a + b·√m)/d`, or `√(a + b·√m)/d` when `root`, of a
/// trigonometric function at the angle `kπ/n`.
pub(super) struct SpecialValue {
    pub(super) k: i64,
    pub(super) n: i64,
    a: i64,
    b: i64,
    m: i64,
    d: i64,
    root: bool,
}

const fn sv(k: i64, n: i64, a: i64, b: i64, m: i64, d: i64, root: bool) -> SpecialValue {
    SpecialValue {
        k,
        n,
        a,
        b,
        m,
        d,
        root,
    }
}

/// `cos(kπ/n)` for `0 ≤ k/n ≤ 1/2`, `n ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12}`
/// (the angles whose cosines are square-root towers of degree ≤ 4).
pub(super) const SPECIAL_COS: &[SpecialValue] = &[
    sv(0, 1, 1, 0, 0, 1, false),   // 1
    sv(1, 12, 2, 1, 3, 2, true),   // √(2 + √3)/2 = (√6 + √2)/4
    sv(1, 10, 10, 2, 5, 4, true),  // √(10 + 2√5)/4
    sv(1, 8, 2, 1, 2, 2, true),    // √(2 + √2)/2
    sv(1, 6, 0, 1, 3, 2, false),   // √3/2
    sv(1, 5, 1, 1, 5, 4, false),   // (1 + √5)/4
    sv(1, 4, 0, 1, 2, 2, false),   // √2/2
    sv(3, 10, 10, -2, 5, 4, true), // √(10 − 2√5)/4
    sv(1, 3, 1, 0, 0, 2, false),   // 1/2
    sv(3, 8, 2, -1, 2, 2, true),   // √(2 − √2)/2
    sv(2, 5, -1, 1, 5, 4, false),  // (√5 − 1)/4
    sv(5, 12, 2, -1, 3, 2, true),  // √(2 − √3)/2 = (√6 − √2)/4
    sv(1, 2, 0, 0, 0, 1, false),   // 0
];

/// `tan(kπ/n)` for `0 ≤ k/n < 1/2`, the same denominators.
pub(super) const SPECIAL_TAN: &[SpecialValue] = &[
    sv(0, 1, 0, 0, 0, 1, false),    // 0
    sv(1, 12, 2, -1, 3, 1, false),  // 2 − √3
    sv(1, 10, 25, -10, 5, 5, true), // √(25 − 10√5)/5
    sv(1, 8, -1, 1, 2, 1, false),   // √2 − 1
    sv(1, 6, 0, 1, 3, 3, false),    // √3/3
    sv(1, 5, 5, -2, 5, 1, true),    // √(5 − 2√5)
    sv(1, 4, 1, 0, 0, 1, false),    // 1
    sv(3, 10, 25, 10, 5, 5, true),  // √(25 + 10√5)/5
    sv(1, 3, 0, 1, 3, 1, false),    // √3
    sv(3, 8, 1, 1, 2, 1, false),    // 1 + √2
    sv(2, 5, 5, 2, 5, 1, true),     // √(5 + 2√5)
    sv(5, 12, 2, 1, 3, 1, false),   // 2 + √3
];

/// Least common multiple of the table denominators.
const SPECIAL_LCM: i64 = 120;

/// Build the arena node of a table value.
pub(super) fn special_value_node(arena: &mut Arena, e: &SpecialValue) -> ExprId {
    let a = arena.int(e.a);
    let mut inner = if e.b == 0 {
        a
    } else {
        let m = arena.int(e.m);
        let half = arena.rational(1, 2);
        let sqrt_m = arena.pow(m, half);
        let b = arena.int(e.b);
        let b_sqrt_m = arena.mul(&[b, sqrt_m]);
        arena.add(&[a, b_sqrt_m])
    };
    if e.root {
        let half = arena.rational(1, 2);
        inner = arena.pow(inner, half);
    }
    let inv_d = arena.rational(1, e.d);
    let v = arena.mul(&[inv_d, inner]);
    eval(arena, v)
}

/// Is `id` a real constant built from rationals, `φ` and square roots
/// (`+ − ×`, integer and half-integer powers), with at least one square
/// root and at most 64 nodes?  The table values are of this form; anything
/// else (symbols, `π`, `i`, cube roots, functions) cannot be one.
fn is_square_root_tower(arena: &Arena, id: ExprId) -> bool {
    // Explicit stack, stopping at the first node that disqualifies (the
    // common case: an argument with a symbol) or past 64 nodes.
    let mut stack = vec![id];
    let mut visited = 0usize;
    let mut irrational = false;
    while let Some(n) = stack.pop() {
        visited += 1;
        if visited > 64 {
            return false;
        }
        match arena.node(n) {
            ExprNode::Num(_) => {}
            ExprNode::Add(cs) | ExprNode::Mul(cs) => stack.extend(cs.iter().copied()),
            ExprNode::Neg(a) => stack.push(*a),
            ExprNode::GoldenRatio => irrational = true,
            ExprNode::Pow(b, e) => {
                match arena.as_num(*e) {
                    Some(q) if q.is_integer() => {}
                    Some(q) if *q.denom() == BigInt::from(2) => irrational = true,
                    _ => return false,
                }
                stack.push(*b);
            }
            _ => return false,
        }
    }
    irrational
}

/// Monic minimal polynomial over ℚ of an algebraic constant.
fn monic_minimal_polynomial(arena: &mut Arena, id: ExprId) -> Option<crate::poly::Poly> {
    let p = crate::poly::algebraic::minimal_polynomial(arena, id)?;
    (!p.is_zero()).then(|| p.make_monic())
}

/// Recognise `v = s·f(qπ)` for a table angle `q ∈ [0, 1/2]` (`f = cos`) or
/// `q ∈ [0, 1/2)` (`f = tan`) and a sign `s = ±1`; returns `(s, q)`.
///
/// The candidate angle comes from the certified `f64` value of `v` (the
/// nearest multiple of `π/120`, within `10⁻⁶` of a step); it is accepted
/// only when `|v|` is exactly the table value: structurally equal after
/// [`eval`], or with the same minimal polynomial over ℚ.  Equal minimal
/// polynomials and close values identify the number, because the
/// the roots of every table polynomial are more than `0.3` apart (they are
/// the conjugates `±f(jqπ)`, `gcd(j, n) = 1`) while the tolerance allows at
/// most `4·10⁻⁷` (`tan` at `5π/12`).  So the reduction does not depend
/// on the radical form of the input: `(√6 + √2)/4`, `√(2 + √3)/2` and
/// `1/(√6 − √2)` all give `acos = π/12`.
fn special_angle_of(arena: &mut Arena, v: ExprId, table: SpecialTable) -> Option<(Q, Q)> {
    if !is_square_root_tower(arena, v) {
        return None;
    }
    let x = crate::transforms::evalf::evalf_f64(arena, v).ok()?;
    if !x.is_finite() || x.abs() < 1e-12 {
        return None;
    }
    let theta = match table {
        SpecialTable::Cos if x.abs() <= 1.0 => x.abs().acos(),
        SpecialTable::Cos => return None,
        SpecialTable::Tan => x.abs().atan(),
    } / std::f64::consts::PI;
    let steps = theta * SPECIAL_LCM as f64;
    let r = steps.round();
    if (steps - r).abs() > 1e-6 {
        return None;
    }
    let q = Ratio::new(BigInt::from(r as i64), BigInt::from(SPECIAL_LCM));
    let entries = match table {
        SpecialTable::Cos => SPECIAL_COS,
        SpecialTable::Tan => SPECIAL_TAN,
    };
    let entry = entries
        .iter()
        .find(|e| Ratio::new(BigInt::from(e.k), BigInt::from(e.n)) == q)?;
    let abs_v = if x < 0.0 {
        let neg = arena.neg(v);
        eval(arena, neg)
    } else {
        v
    };
    let target = special_value_node(arena, entry);
    if abs_v != target {
        let p = monic_minimal_polynomial(arena, abs_v)?;
        if monic_minimal_polynomial(arena, target)? != p {
            return None;
        }
    }
    let s: Q = if x < 0.0 { -Ratio::one() } else { Ratio::one() };
    Some((s, q))
}

/// Evaluate `atan2(y, x)` with quadrant-aware logic.
///
/// When both arguments are numeric rationals, returns an exact symbolic
/// result.  Returns `None` when no simplification is possible.
pub(crate) fn eval_atan2(arena: &mut Arena, y: ExprId, x: ExprId) -> Option<ExprId> {
    if let Some(angle) = eval_atan2_infinite(arena, y, x) {
        return Some(angle);
    }
    // Clone numeric values up-front so we can use arena mutably below.
    let y_val = arena.as_num(y).cloned();
    let x_val = arena.as_num(x).cloned();

    match (y_val, x_val) {
        (Some(yv), Some(xv)) => {
            let y_zero = yv.is_zero();
            let x_zero = xv.is_zero();
            let y_pos = !yv.is_negative() && !y_zero;
            let x_pos = !xv.is_negative() && !x_zero;
            let y_neg = yv.is_negative();
            let x_neg = xv.is_negative();

            if y_zero && x_zero {
                // atan2(0, 0) is the argument of 0: undefined (SymPy: nan),
                // as `arg(0)` is.  Before 0.30 it was 0 "by convention", and
                // `atan2(sin(π/4) − cos(π/4), 0)` evaluated to 0.
                Some(arena.nan)
            } else if y_zero && x_pos {
                // atan2(0, x>0) = 0
                Some(arena.zero)
            } else if y_zero && x_neg {
                // atan2(0, x<0) = π
                Some(arena.pi)
            } else if y_pos && x_zero {
                // atan2(y>0, 0) = π/2
                let half = arena.rational(1, 2);
                Some(arena.mul(&[half, arena.pi]))
            } else if y_neg && x_zero {
                // atan2(y<0, 0) = -π/2
                let neg_half = arena.rational(-1, 2);
                Some(arena.mul(&[neg_half, arena.pi]))
            } else if x_pos {
                // Quadrant I or IV: atan2(y, x) = atan(y/x)
                let ratio = arena.div(y, x);
                let atan_node = arena.atan(ratio);
                Some(eval_atan(arena, ratio).unwrap_or(atan_node))
            } else if x_neg && !y_neg {
                // Quadrant II (y >= 0, x < 0): atan2(y, x) = atan(y/x) + π
                let ratio = arena.div(y, x);
                let atan_val = eval_atan(arena, ratio).unwrap_or_else(|| arena.atan(ratio));
                Some(arena.add(&[atan_val, arena.pi]))
            } else {
                // Quadrant III (y < 0, x < 0): atan2(y, x) = atan(y/x) - π
                let ratio = arena.div(y, x);
                let atan_val = eval_atan(arena, ratio).unwrap_or_else(|| arena.atan(ratio));
                let neg_pi = arena.neg(arena.pi);
                Some(arena.add(&[atan_val, neg_pi]))
            }
        }
        _ => eval_atan2_special(arena, y, x),
    }
}

/// The least (`min`) or greatest of the symbol-free real constants `args`
/// (`Min(π, 3) = 3`, `Max(√2, 3/2) = 3/2`; SymPy compares them the same
/// way), each comparison decided as `eval` decides `a ≥ b`
/// ([`order_by_difference`]: a certified sign of the difference, else its
/// exact zero tests; of equal values a number, else the first, so that
/// `Min(atan(1/2) + atan(1/3) − π/4, 0)` is 0, which up to 0.50 stayed a
/// `Min` that hid a 0 from the rules for `1/c`).  `None` when an argument
/// is not such a constant or a comparison is not decided.
pub(super) fn extremum_of_real_constants(
    arena: &mut Arena,
    args: &[ExprId],
    min: bool,
) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    use crate::transforms::evalf::{ZeroSearch, evalf_settled};
    if args.len() < 2 || args.len() > 8 {
        return None;
    }
    let mut cache = AssumptionCache::new();
    for &a in args {
        if !crate::base::walk::free_symbols(arena, a).is_empty() {
            return None;
        }
        // Finite by the assumptions, else by a value (`Ai(√2)`,
        // `fresnels(15)`: the table claims nothing for a constant argument
        // of a function whose special values are not all folded).
        if !Truth::from(cache.query(arena, a, Props::FINITE)).is_true()
            && evalf_settled(arena, a, 16, ZeroSearch::Cap).is_err()
        {
            return None;
        }
    }
    // Real by the assumptions, else by `known_real` (a root by Cardano's
    // formula is, by its minimal polynomial; up to 0.44 `Max(λ, −4)` stayed).
    for &a in args {
        if cache.query(arena, a, Props::REAL) != Some(true) && !known_real(arena, a) {
            return None;
        }
    }
    let first = args
        .iter()
        .position(|&a| arena.as_num(a).is_some())
        .unwrap_or(0);
    let mut best = args[first];
    for (_, &a) in args.iter().enumerate().filter(|&(i, _)| i != first) {
        let diff = Difference {
            a,
            b: best,
            orig: (a, best),
        };
        // `a < best` is `!(a ≥ best)`; `a > best` is decided directly.
        let beyond = if min {
            !order_by_difference(arena, diff, false)?
        } else {
            order_by_difference(arena, diff, true)?
        };
        if beyond {
            best = a;
        }
    }
    Some(best)
}

/// `atan2(y, x)` with one argument `±∞` and the other real and finite (as
/// SymPy: `atan2(1, oo)` → 0, `atan2(-1, -oo)` → -pi, `atan2(0, -oo)` →
/// pi, `atan2(oo, 1)` → pi/2): the limit of the angle.  `arg(∞ + i)` is
/// `atan2(1, ∞) = 0`.  Both infinite, or the other's sign unknown: `None`.
fn eval_atan2_infinite(arena: &mut Arena, y: ExprId, x: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    use std::cmp::Ordering;
    let real_infinity =
        |id: ExprId| ValueClass::special_atom(arena, id).and_then(ValueClass::real_infinity);
    let (y_inf, x_inf) = (real_infinity(y), real_infinity(x));
    if y_inf.is_some() == x_inf.is_some() {
        return None;
    }
    let finite = if y_inf.is_some() { x } else { y };
    let mut cache = AssumptionCache::new();
    // (`REAL` is finite and real)
    if cache.query(arena, finite, Props::REAL) != Some(true) {
        return None;
    }
    let sign = if let Some(q) = arena.as_num(finite) {
        q.cmp(&Q::zero())
    } else if cache.query(arena, finite, Props::POSITIVE) == Some(true) {
        Ordering::Greater
    } else if cache.query(arena, finite, Props::NEGATIVE) == Some(true) {
        Ordering::Less
    } else if cache.query(arena, finite, Props::ZERO) == Some(true) {
        Ordering::Equal
    } else {
        arena.sign_of_real_constant(finite)?
    };
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    let angle: Q = match (y_inf, x_inf) {
        (_, Some(value_class::Sign::Positive)) => Q::zero(),
        (_, Some(value_class::Sign::Negative)) if sign == Ordering::Less => -Q::one(),
        (_, Some(value_class::Sign::Negative)) => Q::one(),
        (Some(value_class::Sign::Positive), None) => half,
        (Some(value_class::Sign::Negative), None) => -half,
        (None, None) => return None,
    };
    Some(pi_times(arena, angle))
}

/// `atan2(y, x)` for real square-root towers (at least one irrational) whose
/// quotient is a special tangent: the quadrant from the certified signs,
/// the angle from [`eval_atan`] of `y/x` (`atan2(√3/2, 1/2) = π/3`,
/// `atan2(−√2, −√6) = −5π/6`).  `None` unless the angle is a rational
/// multiple of `π`.
fn eval_atan2_special(arena: &mut Arena, y: ExprId, x: ExprId) -> Option<ExprId> {
    let tower_or_rational =
        |arena: &Arena, id: ExprId| arena.as_num(id).is_some() || is_square_root_tower(arena, id);
    if !tower_or_rational(arena, y) || !tower_or_rational(arena, x) {
        return None;
    }
    let sign_of = |arena: &Arena, id: ExprId| -> Option<i8> {
        if let Some(r) = arena.as_num(id) {
            return Some(if r.is_zero() {
                0
            } else if r.is_negative() {
                -1
            } else {
                1
            });
        }
        // An irrational tower is not zero; its sign is certified at 1e-12.
        let v = crate::transforms::evalf::evalf_f64(arena, id).ok()?;
        (v.is_finite() && v.abs() >= 1e-12).then_some(if v < 0.0 { -1 } else { 1 })
    };
    let sy = sign_of(arena, y)?;
    let sx = sign_of(arena, x)?;
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    let angle: Q = match (sy, sx) {
        (0, 0) => return None,
        (0, 1) => Ratio::zero(),
        (0, _) => Ratio::one(),
        (1, 0) => half,
        (_, 0) => -half,
        _ => {
            let ratio = arena.div(y, x);
            let ratio = eval(arena, ratio);
            let a = eval_atan(arena, ratio)?;
            let q = as_pi_multiple(arena, a)?;
            match (sy, sx) {
                (_, 1) => q,
                (1, _) => q + Ratio::one(),
                _ => q - Ratio::one(),
            }
        }
    };
    Some(pi_times(arena, angle))
}

pub(super) fn eval_sinh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Odd function: sinh(-x) = -sinh(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        let sinh_pos = eval_sinh(arena, pos_inner).unwrap_or_else(|| arena.sinh(pos_inner));
        return Some(arena.neg(sinh_pos));
    }

    if inner == arena.zero {
        return Some(arena.zero);
    } // sinh(0) = 0
    // sinh(i·qπ) = i·sin(qπ) where that evaluates (SymPy's `sinh.eval`:
    // `sinh(I*pi/8)` → `I*sqrt(1/2 - sqrt(2)/4)`); before 0.43 only the
    // rational values folded (in the canonical constructor).
    // At any other rational `q` the sine stays (`sinh(iπ/105)` →
    // `i·sin(π/105)`, as SymPy writes it; up to 0.46 the `sinh` stayed).
    let q = as_imaginary_pi_multiple(arena, inner)?;
    let angle = pi_times(arena, q);
    let s = eval_sin(arena, angle).unwrap_or_else(|| arena.sin(angle));
    Some(arena.mul(&[arena.i_unit, s]))
}

pub(super) fn eval_cosh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Even function: cosh(-x) = cosh(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        return Some(eval_cosh(arena, pos_inner).unwrap_or_else(|| arena.cosh(pos_inner)));
    }

    if inner == arena.zero {
        return Some(arena.one);
    } // cosh(0) = 1
    // cosh(i·qπ) = cos(qπ) (SymPy: `cosh(I*pi/5)` → `1/4 + sqrt(5)/4`).
    // (`cosh(iπ/105)` → `cos(π/105)` at any other rational `q`.)
    let q = as_imaginary_pi_multiple(arena, inner)?;
    let angle = pi_times(arena, q);
    Some(eval_cos(arena, angle).unwrap_or_else(|| arena.cos(angle)))
}

pub(super) fn eval_tanh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Odd function: tanh(-x) = -tanh(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        let tanh_pos = eval_tanh(arena, pos_inner).unwrap_or_else(|| arena.tanh(pos_inner));
        return Some(arena.neg(tanh_pos));
    }

    if inner == arena.zero {
        return Some(arena.zero);
    } // tanh(0) = 0
    // tanh(i·qπ) = i·tan(qπ) (SymPy: `tanh(I*pi/12)` → `I*(2 - sqrt(3))`).
    // (`tanh(iπ/105)` → `i·tan(π/105)` at any other rational `q`.)
    let q = as_imaginary_pi_multiple(arena, inner)?;
    let angle = pi_times(arena, q);
    let t = eval_tan(arena, angle).unwrap_or_else(|| arena.tan(angle));
    Some(arena.mul(&[arena.i_unit, t]))
}

/// `c` for an argument `i·c` with `c` a real constant (`c = 1` for `i`
/// itself, `−1` for `−i`): the argument of the identities between the
/// inverse trigonometric and hyperbolic functions on the imaginary axis.
fn imaginary_real_cofactor(arena: &mut Arena, z: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let c = if z == arena.i_unit {
        arena.one
    } else {
        as_pure_imaginary(arena, z)?
    };
    (walk::free_symbols(arena, c).is_empty()
        && AssumptionCache::new().query(arena, c, Props::REAL) == Some(true))
    .then_some(c)
}

/// `ln(1 + √2) = asinh(1)`.
fn ln_one_plus_sqrt2(arena: &mut Arena) -> ExprId {
    let two = arena.int(2);
    let sqrt2 = arena.sqrt(two);
    let sum = arena.add(&[arena.one, sqrt2]);
    arena.ln(sum)
}

/// The special values of `asinh`, SymPy's `asinh.eval`: `asinh(±1) =
/// ±ln(1 + √2)` and `asinh(i·c) = i·asin(c)` for a real constant `c` whose
/// `asin` folds (`asinh(±i) = ±iπ/2`, `asinh(i√3/2) = iπ/3`); the identity
/// holds on all of ℂ (`asinh z = ln(z + √(1 + z²))`, `asin z = −i·ln(iz +
/// √(1 − z²))`).  Through `acsch(z) = asinh(1/z)` they give `acsch(±1)` and
/// `acsch(±i) = ∓iπ/2`.  Up to 0.43 only `asinh(0)` folded.
pub(super) fn eval_asinh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // asinh(0) = 0
    if inner == arena.one || inner == arena.neg_one {
        let v = ln_one_plus_sqrt2(arena);
        return Some(if inner == arena.one { v } else { arena.neg(v) });
    }
    // asinh(sinh z) = z for a real constant z (SymPy: `asinh(sinh(3))` → 3).
    if let ExprNode::Sinh(z) = *arena.node(inner)
        && walk::free_symbols(arena, z).is_empty()
        && known_real(arena, z)
    {
        return Some(z);
    }
    if let Some(c) = imaginary_real_cofactor(arena, inner)
        && let Some(v) = eval_asin(arena, c)
    {
        return Some(arena.mul(&[arena.i_unit, v]));
    }
    let pos = negated_constant(arena, inner)?;
    let v = eval_asinh(arena, pos).unwrap_or_else(|| arena.asinh(pos));
    Some(arena.neg(v))
}

/// The special values of `acosh` (SymPy's `acosh.eval` and its table):
/// `acosh(x) = i·acos(x)` for a real `x ∈ [−1, 1]` whose `acos` folds
/// (`acosh(0) = iπ/2`, `acosh(−1) = iπ`, `acosh(−1/2) = 2iπ/3`), and
/// `acosh(i·c) = asinh(|c|) + sign(c)·iπ/2` for a real `c` whose `asinh`
/// folds (`acosh(±i) = ln(1 + √2) ± iπ/2`), both on the principal branch
/// `acosh z = ln(z + √(z + 1)·√(z − 1))` (mpmath: `acosh(0.5)` = 1.0471…j,
/// `acosh(-2j)` = 1.4436… − 1.5707…j).  Through `asech(z) = acosh(1/z)`
/// they give `asech(−1) = iπ`, `asech(2) = iπ/3`, `asech(±∞) = iπ/2`.  Up
/// to 0.43 only `acosh(1)` folded, and `series(tanh(acosh(x)), x, 0, 3)`
/// kept `tanh(acosh(0))`, a pole, in its coefficients.
pub(super) fn eval_acosh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.one {
        return Some(arena.zero);
    } // acosh(1) = 0
    // acosh(cosh z) = |z| for a real constant z (SymPy: `acosh(cosh(-2))` → 2).
    if let ExprNode::Cosh(z) = *arena.node(inner)
        && walk::free_symbols(arena, z).is_empty()
        && known_real(arena, z)
    {
        let a = arena.abs(z);
        return Some(eval_abs(arena, z).unwrap_or(a));
    }
    if known_real(arena, inner) {
        // `acos` folds only to `qπ` with `q ∈ [0, 1]`, so `x = cos(qπ)` is
        // in [−1, 1]; anything else is left alone.
        let v = eval_acos(arena, inner)?;
        let q = as_pi_multiple(arena, v)?;
        if q.is_negative() || q > Ratio::one() {
            return None;
        }
        return Some(arena.mul(&[arena.i_unit, v]));
    }
    let c = imaginary_real_cofactor(arena, inner)?;
    let r = arena.as_num(c)?.clone();
    if r.is_zero() {
        return None;
    }
    let abs_c = arena.num_ratio(r.abs());
    let s = eval_asinh(arena, abs_c)?;
    let half = arena.rational(if r.is_positive() { 1 } else { -1 }, 2);
    let i_half_pi = arena.mul(&[half, arena.i_unit, arena.pi]);
    Some(arena.add(&[s, i_half_pi]))
}

pub(super) fn eval_atanh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // atanh(0) = 0
    // atanh(±∞) = ∓iπ/2, the limit along the real axis, where the
    // principal value is `atanh(x) = acoth(x) ∓ iπ/2` for `±x > 1` (mpmath:
    // `atanh(2)` = 0.549… − 1.5708…j; SymPy: `atanh(oo)` → -I*pi/2).
    // Before 0.30 it stayed, and `evalf` refused it.
    //
    // Off the real axis the side decides: atanh(z) → sign(Im z)·iπ/2 as |z|
    // grows, so atanh(±∞ + c) = sign(Im c)·iπ/2 and atanh(c ± i·∞) = ±iπ/2
    // (mpmath `atanh(mpc(1e20, 1))` = 1.0e-20 + 1.5707…j).  Up to 0.40 `∞ +
    // i` was `∞` and `atanh(∞ + i)` was −iπ/2.
    if let Some(inf) = crate::base::canon::infinite_argument(arena, inner) {
        let positive = if inf.imaginary {
            !inf.negative
        } else {
            match inf.side? {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => inf.negative,
            }
        };
        let half = arena.rational(if positive { 1 } else { -1 }, 2);
        return Some(arena.mul(&[half, arena.i_unit, arena.pi]));
    }
    // atanh(tanh z) = z for a real constant z (SymPy: `atanh(tanh(1))` → 1).
    if let ExprNode::Tanh(z) = *arena.node(inner)
        && walk::free_symbols(arena, z).is_empty()
        && known_real(arena, z)
    {
        return Some(z);
    }
    // atanh(i·c) = i·atan(c) on all of ℂ (SymPy's `atanh.eval`): `atanh(±i)
    // = ±iπ/4`, `atanh(i√3) = iπ/3`, and through `acoth(z) = atanh(1/z)`,
    // `acoth(±i) = ∓iπ/4`.  Up to 0.43 they stayed.
    if let Some(c) = imaginary_real_cofactor(arena, inner)
        && let Some(v) = eval_atan(arena, c)
    {
        return Some(arena.mul(&[arena.i_unit, v]));
    }
    let pos = negated_constant(arena, inner)?;
    let v = eval_atanh(arena, pos).unwrap_or_else(|| arena.atanh(pos));
    Some(arena.neg(v))
}
