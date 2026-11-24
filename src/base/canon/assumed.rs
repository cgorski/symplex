//! Values under the assumptions on symbols (part of [`canon`](super)).

use super::*;

// ── Values under the assumptions on symbols ──────────────────────────────────
//
// SymPy's `eval` classmethods fold an application whose value follows from
// the assumptions on the symbols of its argument when it is built:
// `sin(pi*n)` is 0 for an integer `n` (`trigonometric.py`, `sin.eval`:
// `if pi_coeff.is_integer`), `exp(2*pi*I*n)` is 1 (`exp.eval`), `(-1)**m`
// is −1 for an odd `m` (`Pow.eval`), `Abs(p)` is `p` and `sign(p)` is 1 for
// a positive `p`, `sqrt(p**2)` is `p` (`Pow._eval_power`), `Abs(r)**2` is
// `r**2` for a real `r` (`Abs._eval_power`), `Max(p, 0)` is `p`
// (`miscellaneous.py`).  Up to 0.41 symplex kept them all, so the zero they
// stand for was invisible to the canonical arithmetic and to the zero
// tests: `((x + 1)² − x² − 2x − 1)/sin(πn)` was simplified to 0, where
// SymPy (and symplex now) builds `0·zoo = nan`.

/// The parity of an integer-valued `e` (`Some(true)` odd, `Some(false)`
/// even): a number, a term the assumptions decide (`n` odd, `2n` even), or
/// a sum of such terms (`2n + 1` odd), which the assumption system leaves
/// undecided.  One level: the terms of a canonical sum are not sums.
fn parity(
    arena: &Arena,
    cache: &mut crate::base::assumptions::AssumptionCache,
    e: ExprId,
) -> Option<bool> {
    use crate::base::assumptions::{AssumptionCache, Props};
    use num_integer::Integer;
    let one_term = |cache: &mut AssumptionCache, t: ExprId| -> Option<bool> {
        if let Some(q) = arena.as_num(t) {
            return q.is_integer().then(|| q.numer().is_odd());
        }
        if cache.query(arena, t, Props::ODD) == Some(true) {
            return Some(true);
        }
        if cache.query(arena, t, Props::EVEN) == Some(true) {
            return Some(false);
        }
        None
    };
    match arena.node(e) {
        ExprNode::Add(terms) => {
            let mut odd = false;
            for &t in terms.iter() {
                odd ^= one_term(cache, t)?;
            }
            Some(odd)
        }
        _ => one_term(cache, e),
    }
}

/// Is `e` known `≥ 0` (`> 0` when `strict`; `≤ 0`, `< 0` when `upper`):
/// by the assumptions, or as a number plus terms of known sign whose bounds
/// decide (a positive integer is at least 1), as SymPy's `_monotonic_sign`:
/// `j − 1 ≥ 0` for a positive integer `j`, which the assumption system
/// leaves open.  One level: the terms of a canonical sum are not sums.
fn known_sign_by_bounds(
    arena: &Arena,
    cache: &mut crate::base::assumptions::AssumptionCache,
    e: ExprId,
    strict: bool,
    upper: bool,
) -> bool {
    use crate::base::assumptions::Props;
    let (sign, weak, unit) = if upper {
        (Props::NEGATIVE, Props::NONPOSITIVE, -Q::one())
    } else {
        (Props::POSITIVE, Props::NONNEGATIVE, Q::one())
    };
    if cache.query(arena, e, if strict { sign } else { weak }) == Some(true) {
        return true;
    }
    let ExprNode::Add(terms) = arena.node(e) else {
        return false;
    };
    let Some(c) = arena.as_num(terms[0]) else {
        return false;
    };
    // The bound of the sum (towards 0 from the side of `sign`), and whether
    // a term keeps it from being attained.
    let mut bound = c.clone();
    let mut gap = false;
    for &t in &terms[1..] {
        if cache.query(arena, t, sign) == Some(true) {
            if cache.query(arena, t, Props::INTEGER) == Some(true) {
                bound = q_add(&bound, &unit);
            } else {
                gap = true;
            }
        } else if cache.query(arena, t, weak) != Some(true) {
            return false;
        }
    }
    let bound = if upper { -bound } else { bound };
    bound.is_positive() || (bound.is_zero() && (gap || !strict))
}

/// `q·c` with the number distributed over a sum, also over the sum in a
/// product `k·S` (`2·(n + ½)/1 = 2n + 1`, `½·(2n + 1) − ½` stays a
/// product), so that [`parity`] reads it term by term.
pub(super) fn scaled(arena: &mut Arena, q: &Q, c: ExprId) -> ExprId {
    let (k, rest) = arena.as_coeff_term(c);
    let k = q_mul(q, &k);
    if let ExprNode::Add(terms) = arena.node(rest).clone() {
        let parts: SmallVec<[ExprId; 6]> = terms
            .iter()
            .map(|&t| {
                let (kt, r) = arena.as_coeff_term(t);
                arena.make_coeff_term(q_mul(&k, &kt), r)
            })
            .collect();
        return canon_add(arena, &parts);
    }
    arena.make_coeff_term(k, rest)
}

/// The coefficient `c` of an argument `c·π` (`c·π·i` when `imaginary`)
/// with a symbolic `c`: a product with the factor `π` (and `i`) and a
/// non-numeric factor beside them, as SymPy's `_pi_coeff`; the number is
/// distributed over a sum ([`scaled`]: `π·(2n + 1)/2` has `c = n + ½`).
/// `None` for every other argument (a numeric multiple of `π` is the
/// business of the tables).
fn symbolic_pi_coefficient(arena: &mut Arena, a: ExprId, imaginary: bool) -> Option<ExprId> {
    let ExprNode::Mul(children) = arena.node(a) else {
        return None;
    };
    let (pi, i) = (arena.pi, arena.i_unit);
    if children.iter().filter(|&&c| c == pi).count() != 1
        || children.iter().filter(|&&c| c == i).count() != usize::from(imaginary)
    {
        return None;
    }
    let rest: SmallVec<[ExprId; 4]> = children
        .iter()
        .copied()
        .filter(|&c| c != pi && c != i)
        .collect();
    if rest.iter().all(|&c| arena.as_num(c).is_some()) {
        return None;
    }
    let c = canon_mul(arena, &rest);
    Some(scaled(arena, &Q::one(), c))
}

/// Where `c·π` lies for the trigonometric functions.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PiPoint {
    /// `c` an integer: `sin` and `tan` vanish, `cos = (−1)^c`.
    Integer,
    /// `c` an odd multiple of `½`: `cos` vanishes, `sin = (−1)^(c − ½)`,
    /// `tan` has a pole.
    HalfOdd,
}

fn pi_point(arena: &mut Arena, c: ExprId) -> Option<PiPoint> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let mut cache = AssumptionCache::new();
    if cache.query(arena, c, Props::INTEGER) == Some(true) {
        return Some(PiPoint::Integer);
    }
    // By the parity of `2c`: odd is a half-odd `c`, even an integer one that
    // the assumptions do not see (`k/2` for an even `k`).
    let two_c = scaled(arena, &Q::from_integer(BigInt::from(2)), c);
    parity(arena, &mut cache, two_c).map(|odd| {
        if odd {
            PiPoint::HalfOdd
        } else {
            PiPoint::Integer
        }
    })
}

/// `sin`, `cos` or `tan` (`f` = 0, 1, 2) at `c·π` for a `c` at `point`.
fn trig_at_pi_point(arena: &mut Arena, f: u8, point: PiPoint, c: ExprId) -> ExprId {
    match (f, point) {
        (0 | 2, PiPoint::Integer) | (1, PiPoint::HalfOdd) => arena.zero,
        (1, PiPoint::Integer) => canon_pow(arena, arena.neg_one, c),
        (0, PiPoint::HalfOdd) => {
            let minus_half = arena.rational(-1, 2);
            let e = canon_add(arena, &[c, minus_half]);
            canon_pow(arena, arena.neg_one, e)
        }
        _ => arena.complex_infinity,
    }
}

/// The value of an application that follows from the assumptions on the
/// symbols of its argument (see the section comment): `sin`/`cos`/`tan` at
/// `c·π` for an integer or half-odd `c`, `sinh`/`cosh`/`tanh` at `c·π·i`
/// likewise (`sinh(i·x) = i·sin x`, `cosh(i·x) = cos x`, `tanh(i·x) =
/// i·tan x`), `e^(c·π·i)` for `c` or `c + ½` of known parity (`1`, `−1`,
/// `∓i`), `|x| = ±x` for an `x` of known sign, `sign x` for an `x` known
/// positive, negative or zero, `⌊n⌋ = ⌈n⌉ = n` for an integer `n`.  `None`
/// otherwise (cheaply for every other node: [`canon_function`] asks for
/// each node it interns).
pub(super) fn assumed_value(arena: &mut Arena, node: &ExprNode) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    match *node {
        ExprNode::Sin(a) | ExprNode::Cos(a) | ExprNode::Tan(a) => {
            let f = match node {
                ExprNode::Sin(_) => 0,
                ExprNode::Cos(_) => 1,
                _ => 2,
            };
            if let Some(c) = symbolic_pi_coefficient(arena, a, false) {
                let point = pi_point(arena, c)?;
                return Some(trig_at_pi_point(arena, f, point, c));
            }
            shifted_by_periods(arena, f, a, false)
        }
        ExprNode::Sinh(a) | ExprNode::Cosh(a) | ExprNode::Tanh(a) => {
            let f = match node {
                ExprNode::Sinh(_) => 0,
                ExprNode::Cosh(_) => 1,
                _ => 2,
            };
            let Some(c) = symbolic_pi_coefficient(arena, a, true) else {
                return shifted_by_periods(arena, f, a, true);
            };
            let point = pi_point(arena, c)?;
            Some(match f {
                1 => trig_at_pi_point(arena, 1, point, c),
                _ => {
                    let v = trig_at_pi_point(arena, f, point, c);
                    canon_mul(arena, &[arena.i_unit, v])
                }
            })
        }
        ExprNode::Exp(a) => {
            let Some(c) = symbolic_pi_coefficient(arena, a, true) else {
                return exp_shifted_by_periods(arena, a);
            };
            let mut cache = AssumptionCache::new();
            if let Some(odd) = parity(arena, &mut cache, c) {
                return Some(if odd { arena.neg_one } else { arena.one });
            }
            let half = arena.rational(1, 2);
            let shifted = canon_add(arena, &[c, half]);
            let odd = parity(arena, &mut cache, shifted)?;
            let i = arena.i_unit;
            Some(if odd { i } else { canon_neg(arena, i) })
        }
        // ln(e^r) = r for a real r (SymPy's `log.eval`).
        ExprNode::Ln(a) => match *arena.node(a) {
            ExprNode::Exp(r)
                if AssumptionCache::new().query(arena, r, Props::REAL) == Some(true) =>
            {
                Some(r)
            }
            _ => None,
        },
        ExprNode::Abs(a) => {
            let mut cache = AssumptionCache::new();
            if known_sign_by_bounds(arena, &mut cache, a, false, false) {
                Some(a)
            } else if known_sign_by_bounds(arena, &mut cache, a, false, true) {
                Some(canon_neg(arena, a))
            } else if cache.query(arena, a, Props::IMAGINARY) == Some(true) {
                // |i·y| = ±y for a real `y` of known sign (SymPy: `Abs(I*p)` → `p`).
                let minus_i = canon_neg(arena, arena.i_unit);
                let y = canon_mul(arena, &[minus_i, a]);
                if cache.query(arena, y, Props::NONNEGATIVE) == Some(true) {
                    Some(y)
                } else if cache.query(arena, y, Props::NONPOSITIVE) == Some(true) {
                    Some(canon_neg(arena, y))
                } else {
                    None
                }
            } else {
                None
            }
        }
        ExprNode::Sign(a) => {
            let mut cache = AssumptionCache::new();
            if known_sign_by_bounds(arena, &mut cache, a, true, false) {
                Some(arena.one)
            } else if known_sign_by_bounds(arena, &mut cache, a, true, true) {
                Some(arena.neg_one)
            } else if cache.query(arena, a, Props::ZERO) == Some(true) {
                Some(arena.zero)
            } else if cache.query(arena, a, Props::IMAGINARY) == Some(true) {
                // sign(i·y) = ±i (SymPy: `sign(I*p)` → `I`).
                let minus_i = canon_neg(arena, arena.i_unit);
                let y = canon_mul(arena, &[minus_i, a]);
                if cache.query(arena, y, Props::POSITIVE) == Some(true) {
                    Some(arena.i_unit)
                } else if cache.query(arena, y, Props::NEGATIVE) == Some(true) {
                    Some(canon_neg(arena, arena.i_unit))
                } else {
                    None
                }
            } else {
                None
            }
        }
        // H(p) = 1, H(q) = 0 for a positive `p`, a negative `q`; δ(x) = 0 for
        // a real `x ≠ 0` (SymPy's `Heaviside.eval`, `DiracDelta.eval`).
        ExprNode::Heaviside(a) => {
            let mut cache = AssumptionCache::new();
            if known_sign_by_bounds(arena, &mut cache, a, true, false) {
                Some(arena.one)
            } else if known_sign_by_bounds(arena, &mut cache, a, true, true) {
                Some(arena.zero)
            } else {
                None
            }
        }
        ExprNode::DiracDelta(a) => {
            let mut cache = AssumptionCache::new();
            (cache.query(arena, a, Props::REAL) == Some(true)
                && cache.query(arena, a, Props::NONZERO) == Some(true))
            .then_some(arena.zero)
        }
        // atan2(0, p) = 0 for a positive `p`.
        ExprNode::Atan2(y, x) => (y == arena.zero
            && AssumptionCache::new().query(arena, x, Props::POSITIVE) == Some(true))
        .then_some(arena.zero),
        ExprNode::Floor(a) | ExprNode::Ceiling(a) => {
            let mut cache = AssumptionCache::new();
            if cache.query(arena, a, Props::INTEGER) == Some(true) {
                return Some(a);
            }
            // ⌊n + q⌋ = n + ⌊q⌋ for integer terms `n` and a number `q`
            // (SymPy: `floor(n + 1/2)` → `n`).
            let ExprNode::Add(terms) = arena.node(a).clone() else {
                return None;
            };
            let q = arena.as_num(terms[0])?.clone();
            if terms[1..]
                .iter()
                .any(|&t| cache.query(arena, t, Props::INTEGER) != Some(true))
            {
                return None;
            }
            let r = if matches!(node, ExprNode::Floor(_)) {
                q.floor()
            } else {
                q.ceil()
            };
            let nid = arena.intern_num(r);
            let mut parts: SmallVec<[ExprId; 6]> = terms[1..].iter().copied().collect();
            parts.push(arena.intern(ExprNode::Num(nid)));
            Some(canon_add(arena, &parts))
        }
        _ => None,
    }
}

/// `sin`, `cos`, `tan` (`f` = 0, 1, 2) at a sum with terms `c·π` of an
/// integer `c` — `sinh`, `cosh`, `tanh` at terms `c·π·i` when `imaginary`
/// — shifted by those periods, as SymPy's `_peeloff_pi`: `sin(x + 2πn) =
/// sin x`, `cos(x + πm) = −cos x` for an odd `m`, `cos(x + πn) = (−1)ⁿ·cos
/// x`, `tan(x + πn) = tan x`.  `None` without such a term.
fn shifted_by_periods(arena: &mut Arena, f: u8, a: ExprId, imaginary: bool) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let ExprNode::Add(terms) = arena.node(a).clone() else {
        return None;
    };
    let mut cache = AssumptionCache::new();
    let mut periods: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut rest: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &t in terms.iter() {
        match symbolic_pi_coefficient(arena, t, imaginary) {
            Some(c) if cache.query(arena, c, Props::INTEGER) == Some(true) => periods.push(c),
            _ => rest.push(t),
        }
    }
    if periods.is_empty() {
        return None;
    }
    let k = canon_add(arena, &periods);
    let rest = canon_add(arena, &rest);
    let g = arena.intern(match (f, imaginary) {
        (0, false) => ExprNode::Sin(rest),
        (1, false) => ExprNode::Cos(rest),
        (_, false) => ExprNode::Tan(rest),
        (0, true) => ExprNode::Sinh(rest),
        (1, true) => ExprNode::Cosh(rest),
        (_, true) => ExprNode::Tanh(rest),
    });
    if f == 2 {
        return Some(g);
    }
    Some(match parity(arena, &mut cache, k) {
        Some(false) => g,
        Some(true) => canon_neg(arena, g),
        None => {
            let sign = canon_pow(arena, arena.neg_one, k);
            canon_mul(arena, &[sign, g])
        }
    })
}

/// `e^(w + c·π·i)` for terms `c·π·i` of known parity: `e^(x + 2πin) =
/// eˣ`, `e^(x + iπm) = −eˣ` for an odd `m` (SymPy's `exp.eval` of a sum).
/// `None` without such a term.
fn exp_shifted_by_periods(arena: &mut Arena, a: ExprId) -> Option<ExprId> {
    let ExprNode::Add(terms) = arena.node(a).clone() else {
        return None;
    };
    let mut cache = crate::base::assumptions::AssumptionCache::new();
    let mut odd = false;
    let mut rest: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &t in terms.iter() {
        match symbolic_pi_coefficient(arena, t, true).and_then(|c| parity(arena, &mut cache, c)) {
            Some(o) => odd ^= o,
            None => rest.push(t),
        }
    }
    if rest.len() == terms.len() {
        return None;
    }
    let rest = canon_add(arena, &rest);
    let e = arena.intern(ExprNode::Exp(rest));
    Some(if odd { canon_neg(arena, e) } else { e })
}

/// A power whose value follows from the assumptions on the symbols of its
/// base and exponent (see the section comment): `(−1)^e = ±1` for an `e` of
/// known parity, `((−1)^a)^k = (−1)^(a·k)` for an integer `k` (valid for
/// every `a`: `z^k` is single-valued), `|r|^(2j) = r^(2j)` for a real `r`,
/// `(b^f)^e = b^(f·e)` for a non-negative `b` and numbers `f`, `e`, and
/// `(b^(2j))^e = |b|^(2j·e)` for a real `b` (`√(r²) = |r|`, `√(p²) = p`).
pub(super) fn assumed_power(arena: &mut Arena, base: ExprId, exp: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    use num_integer::Integer;
    if base == arena.neg_one && arena.as_num(exp).is_none() {
        let mut cache = AssumptionCache::new();
        if let Some(odd) = parity(arena, &mut cache, exp) {
            return Some(if odd { arena.neg_one } else { arena.one });
        }
        // (−1)^(c + t) = (−1)^c·(−1)^t for an integer number `c` and an
        // integer `t` (SymPy: `(-1)**(n + 1)` → `-(-1)**n`).
        let ExprNode::Add(terms) = arena.node(exp).clone() else {
            return None;
        };
        let c = arena.as_num(terms[0])?.clone();
        if !c.is_integer() {
            return None;
        }
        let t = canon_add(arena, &terms[1..]);
        if cache.query(arena, t, Props::INTEGER) != Some(true) {
            return None;
        }
        let unit = canon_pow(arena, base, t);
        return Some(if c.numer().is_odd() {
            canon_neg(arena, unit)
        } else {
            unit
        });
    }
    let e = arena.as_num(exp)?.clone();
    match arena.node(base).clone() {
        // (e^f)^k for an integer `k` when e^(k·f) folds: `(e^(iπn))² = 1`.
        ExprNode::Exp(f) if e.is_integer() => {
            let product = canon_mul(arena, &[f, exp]);
            let product = scaled(arena, &Q::one(), product);
            assumed_value(arena, &ExprNode::Exp(product))
        }
        ExprNode::Pow(b, a) if b == arena.neg_one && e.is_integer() => {
            let product = canon_mul(arena, &[a, exp]);
            let product = scaled(arena, &Q::one(), product);
            Some(canon_pow(arena, b, product))
        }
        ExprNode::Abs(r) if e.is_integer() && e.numer().is_even() => {
            let mut cache = AssumptionCache::new();
            (cache.query(arena, r, Props::REAL) == Some(true)).then(|| canon_pow(arena, r, exp))
        }
        ExprNode::Pow(b, f) if !e.is_integer() => {
            let fq = arena.as_num(f)?.clone();
            let mut cache = AssumptionCache::new();
            let b = if known_sign_by_bounds(arena, &mut cache, b, false, false) {
                b
            } else if fq.is_integer()
                && fq.numer().is_even()
                && cache.query(arena, b, Props::REAL) == Some(true)
            {
                arena.abs(b)
            } else {
                return None;
            };
            let nid = arena.intern_num(q_mul(&fq, &e));
            let product = arena.intern(ExprNode::Num(nid));
            Some(canon_pow(arena, b, product))
        }
        ExprNode::Mul(children) if !e.is_integer() => assumed_mul_power(arena, &children, exp, &e),
        _ => None,
    }
}

/// `(c·∏ fⱼ)^e` for a non-integer rational `e` when some factor `f = b^k`
/// is non-negative by the assumptions (`b ≥ 0`, or `b` real and `k` even)
/// and `k·e` is an integer, so its root disappears: those factors come out
/// with the coefficient's magnitude, `(A·B)^e = A^e·B^e` for `A ≥ 0` on
/// the principal branch (`arg(A·B) = arg B`).  `√(4b²) = 2|b|`, `√(8r²x) =
/// 2√2·|r|·√x`, `√(−9p²) = 3ip` (SymPy's `Pow._eval_expand_power_base`, run
/// by `Mul._eval_power`, which takes out every non-negative factor).  A
/// product with no such factor is left alone (`√(4x)`, `√(4x²)` for an
/// unassumed `x`).  Up to 0.43 only a lone `√(r²) = |r|` folded.
fn assumed_mul_power(arena: &mut Arena, children: &[ExprId], exp: ExprId, e: &Q) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    use num_integer::Integer;
    let mut cache = AssumptionCache::new();
    let mut comes_out = |arena: &Arena, f: ExprId| -> bool {
        let ExprNode::Pow(b, k) = *arena.node(f) else {
            return false;
        };
        let Some(kq) = arena.as_num(k) else {
            return false;
        };
        q_mul(kq, e).is_integer()
            && (known_sign_by_bounds(arena, &mut cache, b, false, false)
                || (kq.is_integer()
                    && kq.numer().is_even()
                    && cache.query(arena, b, Props::REAL) == Some(true)))
    };
    let mut out: SmallVec<[ExprId; 6]> = SmallVec::new();
    let mut kept: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &f in children {
        if comes_out(arena, f) {
            out.push(f);
        } else {
            kept.push(f);
        }
    }
    if out.is_empty() {
        return None;
    }
    let mut values: SmallVec<[ExprId; 6]> = out.iter().map(|&f| canon_pow(arena, f, exp)).collect();
    // The coefficient's magnitude comes out too; its sign stays inside.
    if let Some(&first) = kept.first()
        && let Some(c) = arena.as_num(first).cloned()
    {
        let magnitude = c.abs();
        if !magnitude.is_one() {
            let m = arena.intern_num(magnitude);
            let m = arena.intern(ExprNode::Num(m));
            values.push(canon_pow(arena, m, exp));
            if c.is_negative() {
                kept[0] = arena.neg_one;
            } else {
                kept.remove(0);
            }
        }
    }
    if !kept.is_empty() {
        let rest = if kept.len() == 1 {
            kept[0]
        } else {
            canon_mul(arena, &kept)
        };
        values.push(canon_pow(arena, rest, exp));
    }
    Some(canon_mul(arena, &values))
}

/// `Min`/`Max` of numbers and infinities fold when they are built, as
/// SymPy's: the rational arguments are replaced by their least (greatest),
/// `+∞` is dropped from a `Min` and `−∞` from a `Max` (identities), `−∞`
/// absorbs a `Min` and `+∞` a `Max` none of whose other arguments is known
/// to be non-real, `nan` and `zoo` make `nan`, and a single remaining
/// argument is the value: `Min(0, 10) = 0`, `Max(3, 0, x) = Max(3, x)`,
/// `Min(∞, x) = x`, `Max(∞, x) = ∞`.  Up to 0.40 `Min(0, 10)` stayed a node
/// until `eval`, and the zero it hid let `ln(x·Min(0, 10))` pass for finite
/// and `0·∞`-like products for 0.  `None` when nothing changes.
///
/// The arguments of `Min`/`Max` are real where they are defined (the order
/// is the reals'), so `±∞ + c` for a finite `c` (`y + ∞`, which the sum
/// keeps for the imaginary part of a plain `y`) is `±∞` there, and `+∞`
/// absorbs a `Max` whose other arguments may be real (SymPy: `Max(oo, y)` →
/// `oo`): `Max(−3, ∞, y) = ∞`, `Min(y + ∞, y) = y`.  Up to 0.56 only exact
/// infinities folded, and a `Max` with `∞` absorbed only arguments known
/// real, so `tanh(Max(−3, ∞, y))` stayed (it is 1) and `solve` refused
/// equations holding them.  An argument certified non-real (`i`, `ζ(3/7 +
/// i)`, a symbol declared non-real) leaves no point where the order is
/// defined: `nan`, as for `zoo` (SymPy raises "not comparable"; up to 0.57
/// `Max(1, i)` and `Min(∞, i)` stayed nodes, which `0·Max(1, i)` read as 0).
pub(super) fn fold_min_max(arena: &mut Arena, args: &[ExprId], is_min: bool) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    if args.is_empty() {
        return None;
    }
    let (identity, absorbing) = if is_min {
        (arena.infinity, arena.neg_infinity)
    } else {
        (arena.neg_infinity, arena.infinity)
    };
    // `zoo` has no order (SymPy: "The argument 'zoo' is not comparable").
    if args.iter().any(|&a| {
        matches!(
            ValueClass::special_atom(arena, a),
            Some(ValueClass::NaN | ValueClass::ComplexInf)
        )
    }) {
        return Some(arena.nan);
    }
    // An argument known non-real (`∞ + p·i` for a positive `p`, `i`, a
    // constant with a certified nonzero imaginary part) leaves no point where
    // the order is defined (up to 0.56 `Min(∞, i)` was `i`).
    let non_real = {
        use crate::base::truth::Truth;
        let mut cache = AssumptionCache::new();
        args.iter().any(|&a| {
            let real = Truth::from(cache.query(arena, a, Props::EXTENDED_REAL));
            real.is_false()
                || (!real.is_true() && arena.is_symbol_free(a) && certified_nonreal(arena, a))
        })
    };
    if non_real {
        return Some(arena.nan);
    }
    let mut best: Option<(usize, Q)> = None;
    let mut kept: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut changed = false;
    let mut has_absorbing = false;
    for &a in args {
        match real_infinity(arena, a) {
            // `+∞` in a `Min`, `−∞` in a `Max`.
            Some(d) if (d == InfDir::Pos) == is_min => {
                changed = true;
                continue;
            }
            Some(_) => {
                changed |= a != absorbing;
                has_absorbing = true;
                continue;
            }
            None => {}
        }
        if let Some(q) = arena.as_num(a) {
            match &mut best {
                Some((_, b)) => {
                    changed = true;
                    if (is_min && q < b) || (!is_min && q > b) {
                        *b = q.clone();
                    }
                }
                None => {
                    best = Some((kept.len(), q.clone()));
                    kept.push(a);
                }
            }
            continue;
        }
        if kept.contains(&a) {
            changed = true;
            continue;
        }
        kept.push(a);
    }
    if has_absorbing {
        return Some(absorbing);
    }
    if let Some((idx, q)) = best {
        let nid = arena.intern_num(q);
        kept[idx] = arena.intern(ExprNode::Num(nid));
    }
    if drop_dominated(arena, &mut kept, is_min) {
        changed = true;
    }
    match kept.len() {
        0 => Some(identity),
        1 => Some(kept[0]),
        _ if changed => Some(arena.intern(if is_min {
            ExprNode::Min(kept)
        } else {
            ExprNode::Max(kept)
        })),
        _ => None,
    }
}

/// The direction (`Pos`, `Neg`) of `+∞` and `+∞ + c`, `−∞` and `−∞ + c`, `c`
/// finite ([`ValueClass::is_finite_proved`] for each of its terms; `Min`
/// and `Max` read `c` as real); `None` otherwise.
fn real_infinity(arena: &Arena, a: ExprId) -> Option<InfDir> {
    if let Some(sign) = ValueClass::special_atom(arena, a).and_then(ValueClass::real_infinity) {
        return Some(match sign {
            value_class::Sign::Positive => InfDir::Pos,
            value_class::Sign::Negative => InfDir::Neg,
        });
    }
    if !matches!(arena.node(a), ExprNode::Add(_)) {
        return None;
    }
    match classify_arg(arena, a) {
        Arg::InfinitePlus(d @ (InfDir::Pos | InfDir::Neg), c)
            if c.iter()
                .all(|&t| ValueClass::of(arena, t).is_finite_proved()) =>
        {
            Some(d)
        }
        _ => None,
    }
}

/// Is the symbol-free constant `x` certified non-real by its value (an
/// imaginary part certified nonzero: [`Arena::part_signs`])?  `Min`/`Max`
/// ask it at every construction of a node with such an argument the
/// assumptions do not decide (each `eval`, `subs` and rebuild), and the
/// answer is an evaluation: it is memoised under
/// [`FactKey::NonReal`](crate::base::arena::FactKey::NonReal) (`NONREAL`
/// when proved, `UNKNOWN` when not).  Up to 0.57 it was evaluated anew each
/// time (about 1% of the CPU time of the golden corpus).
fn certified_nonreal(arena: &Arena, x: ExprId) -> bool {
    use crate::base::arena::FactKey;
    use crate::base::truth::SignSet;
    if let Some(signs) = arena.constant_fact(x, FactKey::NonReal) {
        return signs.is_subset(SignSet::NONREAL);
    }
    let nonreal = arena.part_signs(x).is_some_and(|s| s.im.is_some());
    let signs = if nonreal {
        SignSet::NONREAL
    } else {
        SignSet::UNKNOWN
    };
    arena.remember_constant_fact(x, FactKey::NonReal, signs);
    nonreal
}

/// Drop the arguments of a `Min` (`Max`) that another one is known to be at
/// most (at least) by the assumptions on their difference, as SymPy's
/// `MinMaxBase._collapse_arguments`/`_is_connected`: `Max(p, 0) = p` and
/// `Min(p, 0) = 0` for a positive `p`, `Min(q, q + 1) = q`.  Only real
/// arguments (SymPy rejects the others) and at most eight of them, with a
/// symbol among them (numbers were merged before).  `true` when an
/// argument was dropped.
fn drop_dominated(arena: &mut Arena, kept: &mut SmallVec<[ExprId; 4]>, is_min: bool) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    if kept.len() < 2 || kept.len() > 8 || kept.iter().all(|&a| arena.as_num(a).is_some()) {
        return false;
    }
    let mut cache = AssumptionCache::new();
    if !kept
        .iter()
        .all(|&a| cache.query(arena, a, Props::REAL) == Some(true))
    {
        return false;
    }
    // `a` dominates `b` when `a − b ≥ 0` for a `Max`, `≤ 0` for a `Min`.
    let mut dropped = vec![false; kept.len()];
    for i in 0..kept.len() {
        for j in 0..kept.len() {
            if i == j || dropped[i] || dropped[j] {
                continue;
            }
            let minus_b = canon_neg(arena, kept[j]);
            let d = canon_add(arena, &[kept[i], minus_b]);
            if known_sign_by_bounds(arena, &mut cache, d, false, is_min) {
                dropped[j] = true;
            }
        }
    }
    if !dropped.contains(&true) {
        return false;
    }
    let mut k = 0;
    kept.retain(|_| {
        k += 1;
        !dropped[k - 1]
    });
    true
}

/// Is `a` one of the positive real constants `π`, `e`, `γ`, `G`, `φ`?
pub(super) fn is_positive_constant(arena: &Arena, a: ExprId) -> bool {
    matches!(
        arena.node(a),
        ExprNode::Pi
            | ExprNode::E
            | ExprNode::EulerGamma
            | ExprNode::Catalan
            | ExprNode::GoldenRatio
    )
}

/// `i·v` for a folded real value `v` (0, a rational, or a pole).
pub(super) fn times_i(v: Folded) -> Option<Folded> {
    match v {
        Folded::Rational(q) if q.is_zero() => Some(Folded::int(0)),
        Folded::Rational(q) => Some(Folded::Imaginary(q)),
        Folded::ComplexInfinity => Some(Folded::ComplexInfinity),
        _ => None,
    }
}

/// `−1/2` (`true`) or `1/2`: the imaginary part over `π` of a value at
/// `∓i·∞` such as `ψ(±i·∞) = ∞ ± iπ/2`.
pub(super) fn signed_half(negative: bool) -> Q {
    Q::new(BigInt::from(if negative { -1 } else { 1 }), BigInt::from(2))
}

/// Is `id` the canonical `−exp(−1)`, i.e. `−1/e`?
pub(super) fn is_minus_inverse_e(arena: &Arena, id: ExprId) -> bool {
    let ExprNode::Mul(children) = arena.node(id) else {
        return false;
    };
    matches!(**children, [c, e] if c == arena.neg_one
        && matches!(arena.node(e), ExprNode::Exp(m) if *m == arena.neg_one))
}

/// Is `B(a, b) = Γ(a)Γ(b)/Γ(a+b)` at rationals a pole: exactly one of
/// `Γ(a)`, `Γ(b)` has one and `Γ(a+b)` does not?
pub(super) fn beta_has_pole(a: &Q, b: &Q) -> bool {
    is_nonpositive_int(a) != is_nonpositive_int(b) && !is_nonpositive_int(&q_add(a, b))
}

/// The zeros, poles and `nan`s of the library functions at exact numbers
/// (their other values are left to `eval`): `J_ν(0) = I_ν(0) = 0` for
/// `ν > 0` and for every integer `ν ≠ 0` (`J_{−n} = (−1)ⁿ J_n`,
/// `I_{−n} = I_n`; before 0.30 `J_{−3}(0)` stayed), `zoo` for a negative
/// non-integer `ν` (`J_ν(z) ~ (z/2)^ν/Γ(ν + 1)`), as SymPy's
/// `besselj.eval`/`besseli.eval`, `Y_ν(0)` and `K_ν(0)` infinite (`−∞`/`∞`
/// for `ν = 0`, `zoo` otherwise, as SymPy's `bessely.eval`/`besselk.eval`)
/// except `Y_ν(0) = 0` at a negative half-integer `ν` (where SymPy is wrong),
/// `erfi(0)`, `erfinv(0)`, `erfinv(±1) = ±∞`, `erfcinv(1) = 0`,
/// `erfcinv(0) = ∞`, `erfcinv(2) = −∞`, `Shi(0)`, `Chi(0) = zoo`,
/// `S(0) = C(0) = 0`, `K(1) = zoo`, `Li_s(0) = 0`, and `nan` for a `nan`
/// argument.
///
/// `Chi(z) = γ + ln z + ∫₀ᶻ (cosh t − 1)/t dt` has the singularity of
/// `ln z` at 0, and takes its value there, `zoo` (as `Ci(0)`; SymPy:
/// `Chi(0)`, `Ci(0)`, `log(0)` → zoo): approaching 0 along the negative
/// reals the value is `Chi(|x|) + iπ` (mpmath `chi(-1e-30)` =
/// `-68.50… + 3.14…j`).  `Ei` and `li` are real on both sides of their
/// singular points and tend to `−∞` from both, so `Ei(0) = li(1) = −∞`
/// (mpmath `ei(±1e-30)` = `-68.50…`).  Before 0.30 `Chi(0)` was `−∞`.
pub(super) fn lib_function_value(
    arena: &Arena,
    f: crate::base::libfn::LibFn,
    args: &[ExprId],
) -> Option<Folded> {
    use crate::base::libfn::LibFn;
    // Cheap filter: some argument must be a number or a special atom.
    let special = |id: ExprId| {
        matches!(
            arena.node(id),
            ExprNode::Num(_)
                | ExprNode::NaN
                | ExprNode::Infinity
                | ExprNode::NegInfinity
                | ExprNode::ComplexInfinity
        )
    };
    if !args.iter().any(|&a| special(a)) {
        return None;
    }
    if args
        .iter()
        .any(|&a| ValueClass::special_atom(arena, a) == Some(ValueClass::NaN))
    {
        return Some(Folded::NaN);
    }
    let num = |i: usize| args.get(i).and_then(|&a| arena.as_num(a));
    let is = |i: usize, v: i64| num(i).is_some_and(|q| *q == Q::from_integer(BigInt::from(v)));
    // The special atom argument `i` is, if any.
    let class = |i: usize| {
        args.get(i)
            .and_then(|&a| ValueClass::special_atom(arena, a))
    };
    let neg_infinity = |i: usize| class(i) == Some(ValueClass::NegInf);
    // Argument `i` is `+∞` (or `−∞` too, with `both`).
    let at_infinity = |i: usize, both: bool| match class(i) {
        Some(ValueClass::PosInf) => true,
        Some(ValueClass::NegInf) => both,
        _ => false,
    };
    // Argument `i` (an order) is finite ([`ValueClass::is_finite_proved`]:
    // no infinity in it, and a number, an argument with symbols or a
    // constant known finite or with a certified value, as in
    // [`at_infinity`]).  Up to 0.54 a constant without an infinity in it
    // counted as finite: `besselj(elliptic_k(besseli(0, 0)), ∞)` and
    // `uppergamma(elliptic_k(besseli(0, 0)), ∞)` were built as 0, though
    // `K(1) = ∞`.
    let finite_order = |i: usize| {
        args.get(i)
            .is_some_and(|&a| ValueClass::of(arena, a).is_finite_proved())
    };
    Some(match f {
        LibFn::BesselJ | LibFn::BesselI if is(1, 0) && num(0)?.is_positive() => Folded::int(0),
        LibFn::BesselJ | LibFn::BesselI
            if is(1, 0) && num(0)?.is_integer() && !num(0)?.is_zero() =>
        {
            Folded::int(0)
        }
        LibFn::BesselJ | LibFn::BesselI if is(1, 0) && num(0)?.is_negative() => {
            Folded::ComplexInfinity
        }
        LibFn::BesselY if is(1, 0) && num(0)?.is_zero() => Folded::NegInfinity,
        // Y_{−n−1/2} = (−1)ⁿ J_{n+1/2} (`cos νπ = 0` drops J_ν from
        // `Y_ν = (J_ν cos νπ − J_{−ν})/sin νπ`) vanishes at 0: mpmath
        // `bessely(-0.5, 0)` = 0, `bessely(-1.5, 1e-20)` = −2.66·10⁻³¹.
        // Before 0.41 `zoo`, as SymPy 1.14's `bessely(-1/2, 0)`.
        LibFn::BesselY
            if is(1, 0) && num(0)?.is_negative() && num(0)?.denom() == &BigInt::from(2) =>
        {
            Folded::int(0)
        }
        LibFn::BesselK if is(1, 0) && num(0)?.is_zero() => Folded::Infinity,
        LibFn::BesselY | LibFn::BesselK if is(1, 0) && num(0).is_some() => Folded::ComplexInfinity,
        LibFn::Erfi | LibFn::ErfInv | LibFn::Shi | LibFn::FresnelS | LibFn::FresnelC
            if is(0, 0) =>
        {
            Folded::int(0)
        }
        LibFn::ErfInv if is(0, 1) => Folded::Infinity,
        LibFn::ErfInv if is(0, -1) => Folded::NegInfinity,
        LibFn::ErfcInv if is(0, 1) => Folded::int(0),
        LibFn::ErfcInv if is(0, 0) => Folded::Infinity,
        LibFn::ErfcInv if is(0, 2) => Folded::NegInfinity,
        LibFn::Chi if is(0, 0) => Folded::ComplexInfinity,
        LibFn::EllipticK if is(0, 1) => Folded::ComplexInfinity,
        LibFn::PolyLog if is(1, 0) => Folded::int(0),
        // F(0 | m) = 0 for every `m` (as `eval`; SymPy `elliptic_f(0, m)` → 0):
        // up to 0.53 `x·elliptic_f(0, ∞)` stayed a product with the root 0.
        LibFn::EllipticF if is(0, 0) => Folded::int(0),
        // The functions that tend to 0 at `±∞` (as `eval` folds them), folded
        // when built so that `0/f(∞)` is `0/0`: `J_ν`, `Y_ν` at `±∞`, `K_ν`,
        // `E_ν`, `Γ(s, x)` at `+∞`, `Ai`, `K(m)` at `±∞`, `Ai′` at `+∞`, `Bi`
        // at `−∞` (SymPy 1.14: `besselj(nu, oo)`, `bessely(nu, -oo)`,
        // `besselk(nu, oo)`, `airyai(-oo)`, `elliptic_k(oo)` → 0).  Up to 0.51
        // only `eval` folded them, and `0/besselj(1, ∞)` was `0`.
        LibFn::BesselJ | LibFn::BesselY if at_infinity(1, true) && finite_order(0) => {
            Folded::int(0)
        }
        LibFn::BesselK | LibFn::ExpInt | LibFn::UpperGamma
            if at_infinity(1, false) && finite_order(0) =>
        {
            Folded::int(0)
        }
        LibFn::AiryAi | LibFn::EllipticK if at_infinity(0, true) => Folded::int(0),
        LibFn::AiryAiPrime if at_infinity(0, false) => Folded::int(0),
        LibFn::AiryBi if neg_infinity(0) => Folded::int(0),
        // Likewise the infinite values `eval` knows, so that `0·f(∞)` is
        // `0·∞`: `Bi`, `Bi′`, `Chi` at `+∞`, `erfi`, `Shi` at `±∞`, `I_ν`
        // of a rational order at `+∞` (SymPy 1.14: `airybi(oo)`, `Chi(oo)`,
        // `besseli(nu, oo)` → oo, `erfi(-oo)`, `Shi(-oo)` → -oo).
        LibFn::AiryBi | LibFn::AiryBiPrime | LibFn::Chi | LibFn::Erfi | LibFn::Shi
            if at_infinity(0, false) =>
        {
            Folded::Infinity
        }
        LibFn::BesselI if at_infinity(1, false) && num(0).is_some() => Folded::Infinity,
        LibFn::Erfi | LibFn::Shi if at_infinity(0, true) => Folded::NegInfinity,
        // The values at `−∞` of a rational order, by the principal branches
        // `evalf` uses (on a cut the value from above: `I_ν(−x) =
        // e^{iπν}I_ν(x)`), and `E(±∞)`: see the helpers.  Up to 0.52 they
        // stayed, and `0·besseli(1, −∞)`, `0·Chi(−∞)` were 0.
        LibFn::BesselI if neg_infinity(1) && num(0).is_some() => {
            bessel_i_at_minus_infinity(num(0)?)
        }
        // `K_ν(−x) = e^{−iπν}K_ν(x) − iπI_ν(x)` (DLMF 10.34.2): the first term
        // tends to 0, the second to `−i·∞` for every real `ν`.
        LibFn::BesselK if neg_infinity(1) && num(0).is_some() => Folded::ImaginaryInfinity(true),
        LibFn::ExpInt if neg_infinity(1) && num(0).is_some() => expint_at_minus_infinity(num(0)?),
        LibFn::UpperGamma if neg_infinity(1) && num(0).is_some() => {
            upper_gamma_at_minus_infinity(num(0)?)
        }
        LibFn::LowerGamma if neg_infinity(1) && num(0).is_some_and(Signed::is_positive) => {
            lower_gamma_at_minus_infinity(num(0)?)
        }
        // `γ(s, ∞) = Γ(s)`, `S(±∞) = C(±∞) = ±1/2`, `η(∞) = 1` (as `eval`).
        LibFn::LowerGamma if at_infinity(1, false) && finite_order(0) => {
            Folded::Gamma(*args.first()?)
        }
        LibFn::FresnelS | LibFn::FresnelC if at_infinity(0, false) => Folded::ratio(1, 2),
        LibFn::FresnelS | LibFn::FresnelC if at_infinity(0, true) => Folded::ratio(-1, 2),
        LibFn::DirichletEta if at_infinity(0, false) => Folded::int(1),
        // `E(m) ~ √(−m)` for `m → −∞`; for `m → +∞` the real part tends to 0
        // and the imaginary part to `±∞` by the side of the cut `[1, ∞)`
        // (mpmath `ellipe(10**6)` = 0.000785 + 999.996i; `evalf` takes no
        // side there), so only the undirected `zoo` holds.
        LibFn::EllipticE if at_infinity(0, false) => Folded::ComplexInfinity,
        LibFn::EllipticE if at_infinity(0, true) => Folded::Infinity,
        // `Chi(−x) = Chi(x) + iπ` (the `ln` in `Chi`; mpmath `chi(-1000)` =
        // 9.86·10⁴³⁰ + 3.14159i): `∞ + iπ`, as `ln(−∞)` (SymPy 1.14: `oo`).
        LibFn::Chi if neg_infinity(0) => Folded::InfinitePlus(InfDir::Pos, Q::one(), Q::zero()),
        // `W_k(0) = −∞` for a branch `k ≠ 0` (as `eval`; SymPy 1.14
        // `LambertW(0, 1)` → -oo), so that `0·W₁(0)` is `0·∞`: up to 0.51
        // `(x·W₁(x))` at `x = 0` was `0`.
        LibFn::LambertW if is(0, 0) && num(1).is_some_and(|k| k.is_integer() && !k.is_zero()) => {
            Folded::NegInfinity
        }
        _ => return None,
    })
}

/// `I_ν(−∞)` for a rational order: `I_ν(−x) = e^{iπν}·I_ν(x)` with `I_ν(x) ~
/// e^x/√(2πx)` (DLMF 10.40.1), so `(−1)ⁿ·∞` for an integer `n` (SymPy 1.14:
/// `besseli(1, -oo)` → `-oo`), `(−1)ᵏ·i·∞` for `ν = k + 1/2` (the real
/// part is 0: `I_{1/2}(z) = √(2/(πz))·sinh z`), and `zoo` for any other
/// order (the direction `e^{iπν}` is off the axes).
pub(super) fn bessel_i_at_minus_infinity(nu: &Q) -> Folded {
    use num_integer::Integer;
    if nu.is_integer() {
        return if nu.to_integer().is_even() {
            Folded::Infinity
        } else {
            Folded::NegInfinity
        };
    }
    match half_integer_part(nu) {
        Some(k) => Folded::ImaginaryInfinity(k.is_odd()),
        None => Folded::ComplexInfinity,
    }
}

/// `E_ν(−∞)` for a rational order.  `E_ν(−x) ~ −e^x/x` for every `ν` (DLMF
/// 8.20.2); the imaginary part is `−π` for `ν = 1` (`E₁(−x) = −Ei(x) −
/// iπ`), 0 for an integer `ν ≤ 0` (`E_ν` is elementary and real there),
/// `x^{ν−1}·sin(π(ν−1))·Γ(1−ν) → 0` for a non-integer `ν < 1`, and
/// unbounded for `ν > 1` (`−πx^{n−1}/(n−1)!` for an integer `n`; mpmath
/// `expint(2, -1000)` = −1.97·10⁴³¹ − 3141.59i): `−∞ − iπ`, `−∞`, `zoo`.
pub(super) fn expint_at_minus_infinity(nu: &Q) -> Folded {
    if nu.is_one() {
        Folded::InfinitePlus(InfDir::Neg, -Q::one(), Q::zero())
    } else if *nu < Q::one() {
        Folded::NegInfinity
    } else {
        Folded::ComplexInfinity
    }
}

/// `Γ(s, −∞)` for a rational `s`, from `Γ(s, −x) = Γ(s) + e^{iπ(s−1)}·R(x)`
/// with a real `R(x) ~ x^{s−1}e^x → +∞` (`γ(s, −x) = (−x)^s·Σ xᵏ/(k!(s +
/// k))`, DLMF 8.7.1): `(−1)^{n−1}·∞` for an integer `n ≥ 1` (SymPy 1.14:
/// `uppergamma(3, -oo)` → nan, from its closed form); for `s = −m ≤ 0`
/// (DLMF 8.4.15, `E₁(−x) = −Ei(x) − iπ`) `(−1)^{m+1}·(∞ + iπ/m!)` (SymPy:
/// `uppergamma(-1, -oo)` → 0); for `s = k + 1/2` the factor is `∓i`, so
/// `Γ(s) − (−1)ᵏ·i·∞` with the exact real part `Γ(s)` (`Γ(1/2, −x) =
/// √π − i√π·erfi(√x)`); `zoo` otherwise.  `|m|`, `|k|` above 100 give `zoo`.
pub(super) fn upper_gamma_at_minus_infinity(s: &Q) -> Folded {
    use num_integer::Integer;
    if s.is_integer() {
        let n = s.to_integer();
        if n.is_positive() {
            return if n.is_odd() {
                Folded::Infinity
            } else {
                Folded::NegInfinity
            };
        }
        let Some(m) = (-n).to_u32().filter(|&m| m <= 100) else {
            return Folded::ComplexInfinity;
        };
        let factorial = (1..=m).fold(BigInt::one(), |acc, j| acc * j);
        let (direction, sign) = if m % 2 == 1 {
            (InfDir::Pos, BigInt::one())
        } else {
            (InfDir::Neg, -BigInt::one())
        };
        return Folded::InfinitePlus(direction, Q::new(sign, factorial), Q::zero());
    }
    let Some(k) = half_integer_part(s) else {
        return Folded::ComplexInfinity;
    };
    let Some(gamma) = k.to_i64().and_then(half_integer_gamma_over_sqrt_pi) else {
        return Folded::ComplexInfinity;
    };
    let direction = if k.is_even() {
        InfDir::NegImag
    } else {
        InfDir::PosImag
    };
    Folded::SqrtPiPlus(gamma, direction)
}

/// `γ(s, −∞)` for a positive rational `s`: `γ(s, −x) = −e^{iπ(s−1)}·R(x)`
/// (see [`upper_gamma_at_minus_infinity`]), so `(−1)ⁿ·∞` for an integer
/// `n` (SymPy 1.14: `lowergamma(2, -oo)` → `oo`), `(−1)ᵏ·i·∞` for `s = k +
/// 1/2` (no real part; SymPy: `lowergamma(1/2, -oo)` → `oo*I`), `zoo`
/// otherwise.
pub(super) fn lower_gamma_at_minus_infinity(s: &Q) -> Folded {
    use num_integer::Integer;
    if s.is_integer() {
        return if s.to_integer().is_even() {
            Folded::Infinity
        } else {
            Folded::NegInfinity
        };
    }
    match half_integer_part(s) {
        Some(k) => Folded::ImaginaryInfinity(k.is_odd()),
        None => Folded::ComplexInfinity,
    }
}

/// `k` when `q = k + 1/2` for an integer `k`.
fn half_integer_part(q: &Q) -> Option<BigInt> {
    (q.denom() == &BigInt::from(2)).then(|| (q - Q::new(1.into(), 2.into())).to_integer())
}

/// `Γ(k + 1/2)/√π` for an integer `|k| ≤ 100`, from `Γ(1/2) = √π` and
/// `Γ(s + 1) = s·Γ(s)`: `1/2` for `k = 1`, `−2` for `k = −1`.
fn half_integer_gamma_over_sqrt_pi(k: i64) -> Option<Q> {
    if k.unsigned_abs() > 100 {
        return None;
    }
    let half = Q::new(1.into(), 2.into());
    let mut q = Q::one();
    for j in 0..k.unsigned_abs() {
        let j = Q::from_integer(BigInt::from(j));
        q = if k > 0 {
            q_mul(&q, &(j + &half))
        } else {
            // Γ(s − 1) = Γ(s)/(s − 1) from s = 1/2 − j.
            q / (-(j + &half))
        };
    }
    Some(q)
}
