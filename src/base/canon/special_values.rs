//! Function applications at exact special arguments (part of [`canon`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Function applications at exact special arguments
// ═══════════════════════════════════════════════════════════════════════════

/// A value that a function application folds to at construction.
pub(super) enum Folded {
    Rational(Q),
    /// `q·i` for a non-zero rational `q`.
    Imaginary(Q),
    Infinity,
    NegInfinity,
    /// The directed infinity `−i·∞` (`true`) or `i·∞` (`false`).
    ImaginaryInfinity(bool),
    /// `d·∞ + q·π·u + r` for the direction `d`, with `u = i` when `d` is
    /// real and `u = 1` when it is imaginary: the finite part of an infinite
    /// value that a bare infinity would lose (`asin(∞) = π/2 − i·∞`,
    /// `ln(−∞) = ∞ + iπ`, `erfc(i·∞) = 1 − i·∞`).
    InfinitePlus(InfDir, Q, Q),
    /// A value already built (a directed infinity `e^c·∞`).
    Built(ExprId),
    /// `Γ(s)` for the argument `s` (`lowergamma(s, ∞)`).
    Gamma(ExprId),
    /// `q·√π + d·∞` (`uppergamma(1/2, −∞) = √π − i·∞`).
    SqrtPiPlus(Q, InfDir),
    /// `q·π·i` for a non-zero rational `q` (`Chi(i·∞) = iπ/2`).
    PiImaginary(Q),
    ComplexInfinity,
    NaN,
}

/// The direction of `±∞` or `±i·∞`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum InfDir {
    Pos,
    Neg,
    PosImag,
    NegImag,
}

impl Folded {
    pub(super) fn int(n: i64) -> Self {
        Folded::Rational(Q::from_integer(BigInt::from(n)))
    }

    pub(super) fn ratio(n: i64, d: i64) -> Self {
        Folded::Rational(Q::new(BigInt::from(n), BigInt::from(d)))
    }
}

/// The exact constant an argument stands for, as far as the tables of
/// [`canon_function`] need it (the canonical spellings of `q`, `q·π`,
/// `q·π·i`, `q·i`, `e` and the special atoms).
pub(super) enum Arg {
    Rational(Q),
    PiMultiple(Q),
    ImaginaryPiMultiple(Q),
    ImaginaryMultiple(Q),
    E,
    Infinity,
    NegInfinity,
    /// `−i·∞` (`true`) or `i·∞` (`false`).
    ImaginaryInfinity(bool),
    /// A sum of `±∞` or `±i·∞` and finite terms (the second field, at
    /// least one; [`is_infinite_sum`]): `∞ + i`, `π/2 + i·∞`.
    InfinitePlus(InfDir, SmallVec<[ExprId; 4]>),
    ComplexInfinity,
    NaN,
    Other,
}

pub(super) fn classify_arg(arena: &Arena, id: ExprId) -> Arg {
    if let Some(q) = inverse_trig_pi_multiple(arena, id) {
        return Arg::PiMultiple(q);
    }
    match arena.node(id) {
        ExprNode::Num(n) => Arg::Rational(arena.num(*n).clone()),
        // ln(−1) = iπ on the principal branch.
        ExprNode::Ln(w) if *w == arena.neg_one => Arg::ImaginaryPiMultiple(Q::one()),
        ExprNode::Pi => Arg::PiMultiple(Q::one()),
        ExprNode::ImaginaryUnit => Arg::ImaginaryMultiple(Q::one()),
        ExprNode::E => Arg::E,
        ExprNode::Infinity => Arg::Infinity,
        ExprNode::NegInfinity => Arg::NegInfinity,
        ExprNode::ComplexInfinity => Arg::ComplexInfinity,
        ExprNode::NaN => Arg::NaN,
        ExprNode::Mul(children) => {
            let (q, rest) = match children.first().map(|&c| arena.node(c)) {
                Some(ExprNode::Num(n)) => (arena.num(*n).clone(), &children[1..]),
                _ => (Q::one(), &children[..]),
            };
            let (pi, i) = (arena.pi, arena.i_unit);
            match *rest {
                [a] if a == pi => Arg::PiMultiple(q),
                [a] if inverse_trig_pi_multiple(arena, a).is_some() => Arg::PiMultiple(q_mul(
                    &q,
                    &inverse_trig_pi_multiple(arena, a).unwrap_or_default(),
                )),
                [a] if matches!(arena.node(a), ExprNode::Ln(w) if *w == arena.neg_one) => {
                    Arg::ImaginaryPiMultiple(q)
                }
                [a] if a == i => Arg::ImaginaryMultiple(q),
                [a, b] if (a == pi && b == i) || (a == i && b == pi) => Arg::ImaginaryPiMultiple(q),
                // A directed infinity keeps only the sign of its coefficient.
                // (identity: the spelling `q·i·∞`)
                [a, b] if a == i && b == arena.infinity => Arg::ImaginaryInfinity(q.is_negative()),
                _ => Arg::Other,
            }
        }
        ExprNode::Add(children) if is_infinite_sum(arena, id) => {
            let mut direction = None;
            let mut finite: SmallVec<[ExprId; 4]> = SmallVec::new();
            for &c in children.iter() {
                let d = match *arena.node(c) {
                    ExprNode::Infinity => Some(InfDir::Pos),
                    ExprNode::NegInfinity => Some(InfDir::Neg),
                    _ if is_directed_infinity(arena, c) => match classify_arg(arena, c) {
                        Arg::ImaginaryInfinity(neg) => Some(if neg {
                            InfDir::NegImag
                        } else {
                            InfDir::PosImag
                        }),
                        // `x·∞ + 1`: no table entry.
                        _ => return Arg::Other,
                    },
                    _ => None,
                };
                match d {
                    Some(d) => direction = Some(d),
                    None => finite.push(c),
                }
            }
            match direction {
                Some(d) => Arg::InfinitePlus(d, finite),
                None => Arg::Other,
            }
        }
        _ => Arg::Other,
    }
}

/// An argument `±∞`, `±i·∞`, or a sum of one of them and finite terms
/// `c` (`∞ + i`, `π/2 − i·∞`): see [`infinite_argument`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct InfiniteArgument {
    /// The infinite term is `±i·∞` (else `±∞`).
    pub(crate) imaginary: bool,
    /// The infinite term is `−∞` or `−i·∞`.
    pub(crate) negative: bool,
    /// The sign of the finite part across the direction — of `Im c` for
    /// `±∞ + c`, of `Re c` for `c ± i·∞`; `Some(Equal)` for a bare
    /// infinity, `None` when unknown.
    pub(crate) side: Option<Ordering>,
}

/// The direction of an infinite argument and the side of it its finite
/// part lies on ([`InfiniteArgument`]), which picks the side of a branch cut
/// along the direction (`atanh(∞ + i) = iπ/2`, `atanh(∞) = −iπ/2`).  The
/// reciprocal functions built from quotients (`cot = cos/sin`, `coth =
/// cosh/sinh`) take their limits there before dividing: the quotient of
/// the two infinities is `nan`.
pub(crate) fn infinite_argument(arena: &Arena, id: ExprId) -> Option<InfiniteArgument> {
    let (d, side) = match classify_arg(arena, id) {
        Arg::Infinity => (InfDir::Pos, Some(Ordering::Equal)),
        Arg::NegInfinity => (InfDir::Neg, Some(Ordering::Equal)),
        Arg::ImaginaryInfinity(neg) => (
            if neg {
                InfDir::NegImag
            } else {
                InfDir::PosImag
            },
            Some(Ordering::Equal),
        ),
        Arg::InfinitePlus(d, c) => {
            let imaginary = matches!(d, InfDir::Pos | InfDir::Neg);
            (d, finite_part_sign(arena, &c, imaginary))
        }
        _ => return None,
    };
    Some(InfiniteArgument {
        imaginary: matches!(d, InfDir::PosImag | InfDir::NegImag),
        negative: matches!(d, InfDir::Neg | InfDir::NegImag),
        side,
    })
}

/// The reciprocal trigonometric and hyperbolic functions and their inverses
/// (`name` lower case: `cot`, `sec`, `csc`, `coth`, `sech`, `csch`, `acot`,
/// `asec`, `acsc`, `acoth`, `asech`, `acsch`; anything else is returned
/// unchanged), which have no node of their own: `cot x = cos x/sin x`,
/// `sec x = 1/cos x`, `acot x = atan(1/x)`, `acoth x = atanh(1/x)`, … (the
/// principal definitions; `asec(x).rewrite(acos)` → `acos(1/x)` in SymPy).
/// The parser, [`Ex::cot`](crate::api::expr::Ex::cot) & co. build them here.
///
/// The values the quotient or the reciprocal loses are taken first, as
/// SymPy's `acot.eval` & co.: `acot(0) = π/2` (the quotient form is
/// `atan(zoo)`), `acoth(0) = iπ/2`, `asech(0) = ∞`, `coth(±∞ + c) = ±1`
/// and `cot(c ± i·∞) = ∓i` for a finite `c` (the quotients of two
/// infinities are `nan`), and `asech` at the infinities (below).  Up to
/// 0.40 `acot(0)` was `atan(zoo)`, `coth(∞)` and `cot(−i·∞)` were `nan`.
/// `subs` rebuilds the quotient forms through this function when the
/// substituted argument is such a point (`coth(x)` at `x = ∞` is `1`;
/// up to 0.43 it was `nan`).
pub(crate) fn reciprocal_function(arena: &mut Arena, name: &str, arg: ExprId) -> ExprId {
    let one = arena.one;
    let infinite = infinite_argument(arena, arg);
    match name {
        "cot" => {
            if let Some(inf) = infinite.filter(|inf| inf.imaginary) {
                let i = arena.i_unit;
                return if inf.negative { i } else { arena.neg(i) };
            }
            let c = arena.cos(arg);
            let s = arena.sin(arg);
            arena.div(c, s)
        }
        "sec" => {
            let c = arena.cos(arg);
            arena.div(one, c)
        }
        "csc" => {
            let s = arena.sin(arg);
            arena.div(one, s)
        }
        "coth" => {
            if let Some(inf) = infinite.filter(|inf| !inf.imaginary) {
                return if inf.negative { arena.neg_one } else { one };
            }
            let c = arena.cosh(arg);
            let s = arena.sinh(arg);
            arena.div(c, s)
        }
        "sech" => {
            let c = arena.cosh(arg);
            arena.div(one, c)
        }
        "csch" => {
            let s = arena.sinh(arg);
            arena.div(one, s)
        }
        "acot" if arg == arena.zero => {
            let half = arena.rational(1, 2);
            arena.mul(&[half, arena.pi])
        }
        "acoth" if arg == arena.zero => {
            let half = arena.rational(1, 2);
            arena.mul(&[half, arena.pi, arena.i_unit])
        }
        "asech" if arg == arena.zero => arena.infinity,
        // asech(z) = acosh(w), w = 1/z → 0, on the cut (−∞, 1) of `acosh`:
        // iπ/2 from above and on it, −iπ/2 from below (`acosh(ε ∓ iδ)` →
        // ∓iπ/2).  So asech(±∞) = iπ/2, asech(±i·∞) = ∓iπ/2 (SymPy's
        // `_asech_table`), asech(±∞ + c) = ∓sign(Im c)·iπ/2 for `Im c ≠ 0`;
        // `zoo` and an unknown side have no single limit (`nan`).  Up to 0.43
        // all were `acosh(0)` (whose value is iπ/2).
        "asech" if ValueClass::special_atom(arena, arg) == Some(ValueClass::ComplexInf) => {
            arena.nan
        }
        "asech" if infinite.is_some() => {
            let upper = infinite.and_then(|inf| {
                if inf.imaginary {
                    Some(inf.negative)
                } else {
                    inf.side.map(|s| s != Ordering::Greater)
                }
            });
            match upper {
                Some(up) => {
                    let half = arena.rational(if up { 1 } else { -1 }, 2);
                    arena.mul(&[half, arena.pi, arena.i_unit])
                }
                None => arena.nan,
            }
        }
        "acot" | "asec" | "acsc" | "acoth" | "asech" | "acsch" => {
            let inv = arena.div(one, arg);
            match name {
                "acot" => arena.atan(inv),
                "asec" => arena.acos(inv),
                "acsc" => arena.asin(inv),
                "acoth" => arena.atanh(inv),
                "asech" => arena.acosh(inv),
                _ => arena.asinh(inv),
            }
        }
        _ => arg,
    }
}

/// The sign of the real (`imaginary == false`) or imaginary part of the
/// sum of the finite `terms`, when every term's contribution is known: a
/// real term of known sign, `i` times a product of factors of known sign.
/// `Some(Equal)` when the part is 0.
fn finite_part_sign(arena: &Arena, terms: &[ExprId], imaginary: bool) -> Option<Ordering> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let mut cache = AssumptionCache::new();
    let mut total = Ordering::Equal;
    for &t in terms {
        let factors: &[ExprId] = match arena.node(t) {
            ExprNode::Mul(children) => children,
            _ => std::slice::from_ref(&t),
        };
        let i_count = factors.iter().filter(|&&f| f == arena.i_unit).count();
        let (re_sign, im_sign) = if i_count == 1 {
            let mut sign = Ordering::Greater;
            for &f in factors.iter().filter(|&&f| f != arena.i_unit) {
                let s = match arena.as_num(f) {
                    Some(q) => q.cmp(&Q::zero()),
                    None => known_sign(arena, f)?,
                };
                if s == Ordering::Less {
                    sign = sign.reverse();
                }
            }
            (Ordering::Equal, sign)
        } else if cache.query(arena, t, Props::REAL) == Some(true) {
            let s = match arena.as_num(t) {
                Some(q) => q.cmp(&Q::zero()),
                None => known_sign(arena, t)?,
            };
            (s, Ordering::Equal)
        } else {
            return None;
        };
        let s = if imaginary { im_sign } else { re_sign };
        match (total, s) {
            (_, Ordering::Equal) => {}
            (Ordering::Equal, s) => total = s,
            (t, s) if t != s => return None,
            _ => {}
        }
    }
    Some(total)
}

/// `outer·exp(inner·c)·∞` for the sum `c` of the finite `terms` of an
/// infinite argument: the directed infinity of `e^(∞ + c) = e^c·∞` and of
/// the trigonometric and hyperbolic functions built from it (`cos(c + i·∞)
/// = e^(−ic)·∞`).
fn exp_direction(arena: &mut Arena, terms: &[ExprId], inner: ExprId, outer: ExprId) -> Folded {
    let c = canon_add(arena, terms);
    let scaled = canon_mul(arena, &[inner, c]);
    let e = arena.exp(scaled);
    let inf = arena.infinity;
    Folded::Built(canon_mul(arena, &[outer, e, inf]))
}

/// `sign·π·q`-style helper: `q` if `s` is `Greater`, `−q` if `Less`.
fn signed(s: Ordering, q: Q) -> Q {
    if s == Ordering::Less { -q } else { q }
}

/// `q` when `id` is an inverse trigonometric function at a point where its
/// principal value is `q·π`: `asin(±1) = ±π/2`, `asin(±1/2) = ±π/6`,
/// `acos(0) = π/2`, `acos(−1) = π`, `acos(±1/2) = π/3, 2π/3`,
/// `atan(±1) = ±π/4`, `atan(±∞) = ±π/2`.  These atoms keep their form (the
/// value is irrational), but a function of them folds: `tan(2·atan(1))` is
/// `tan(π/2) = zoo`, `sin(2·asin(1)) = 0`.
fn inverse_trig_pi_multiple(arena: &Arena, id: ExprId) -> Option<Q> {
    let (a, table): (ExprId, &[(i64, i64, i64, i64)]) = match *arena.node(id) {
        // (argument p/q, value r/s · π)
        ExprNode::Asin(a) => (
            a,
            &[(1, 1, 1, 2), (-1, 1, -1, 2), (1, 2, 1, 6), (-1, 2, -1, 6)],
        ),
        ExprNode::Acos(a) => (
            a,
            &[(0, 1, 1, 2), (-1, 1, 1, 1), (1, 2, 1, 3), (-1, 2, 2, 3)],
        ),
        ExprNode::Atan(a) => match ValueClass::special_atom(arena, a) {
            Some(ValueClass::PosInf) => return Some(Q::new(BigInt::from(1), BigInt::from(2))),
            Some(ValueClass::NegInf) => return Some(Q::new(BigInt::from(-1), BigInt::from(2))),
            // (`atan(zoo)` has no limit; `nan` is folded by the constructor)
            Some(_) => return None,
            None => (a, &[(1, 1, 1, 4), (-1, 1, -1, 4)]),
        },
        _ => return None,
    };
    let v = arena.as_num(a)?;
    table.iter().find_map(|&(p, q, r, s)| {
        (*v == Q::new(BigInt::from(p), BigInt::from(q)))
            .then(|| Q::new(BigInt::from(r), BigInt::from(s)))
    })
}

/// `sign`, `|·|` and `arg` of a directed infinity and of an exponential,
/// which only depend on a direction:
///
/// - `|c·∞| = ∞` and `sign(c·∞) = sign(c)` for factors `c` known to be
///   non-zero (SymPy: `Abs(exp(I)*oo)` → `oo`, `sign(exp(I)*oo)` →
///   `sign(exp(I))`);
/// - `sign(e^z) = e^(i·Im z)` (`|e^z| = e^(Re z)`): `sign(e^i) = e^i`,
///   `sign(e^x) = 1` for a real `x` (SymPy keeps `sign(exp(I))`, whose
///   value this is);
/// - `arg(e^(a + ic)) = c − 2πk` for real `a` and a real constant `c`, `k`
///   the integer putting it in `(−π, π]` (SymPy: `arg(exp(4*I))` → `4 -
///   2*pi`), decided by certified signs; `None` when they are not.
///
/// Before 0.43 all of these stayed (`arg(exp(2i))` too), and `sign(e^i·∞)`
/// was not the direction of `e^i·∞`.
fn sign_abs_arg_of_directions(arena: &mut Arena, node: &ExprNode) -> Option<ExprId> {
    match *node {
        ExprNode::Abs(a) => {
            nonzero_factors_of_directed_infinity(arena, a)?;
            Some(arena.infinity)
        }
        ExprNode::Sign(a) => {
            if let Some(factors) = nonzero_factors_of_directed_infinity(arena, a) {
                let c = canon_mul(arena, &factors);
                return Some(arena.sign(c));
            }
            let ExprNode::Exp(z) = *arena.node(a) else {
                return None;
            };
            let (real, imaginary) = split_real_imaginary_terms(arena, z)?;
            if imaginary.is_empty() {
                return Some(arena.one);
            }
            if real.is_empty() {
                return Some(a);
            }
            let c = canon_add(arena, &imaginary);
            let i = arena.i_unit;
            let ic = canon_mul(arena, &[i, c]);
            Some(arena.exp(ic))
        }
        ExprNode::Arg(a) => {
            let ExprNode::Exp(z) = *arena.node(a) else {
                return None;
            };
            let (_, imaginary) = split_real_imaginary_terms(arena, z)?;
            if imaginary.is_empty() {
                return None;
            }
            let c = canon_add(arena, &imaginary);
            principal_angle(arena, c)
        }
        _ => None,
    }
}

/// The factors other than `∞` of the directed infinity `a` (`Mul([…, ∞])`),
/// when each is known to be non-zero (then finite: an infinite factor would
/// have been absorbed).
fn nonzero_factors_of_directed_infinity(arena: &Arena, a: ExprId) -> Option<SmallVec<[ExprId; 4]>> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let ExprNode::Mul(children) = arena.node(a) else {
        return None;
    };
    if !is_directed_infinity(arena, a) {
        return None;
    }
    let mut cache = AssumptionCache::new();
    // (identity: the factors of the direction, beside `∞`)
    let factors: SmallVec<[ExprId; 4]> = children
        .iter()
        .copied()
        .filter(|&c| c != arena.infinity)
        .collect();
    factors
        .iter()
        .all(|&c| {
            arena.as_num(c).is_some_and(|q| !q.is_zero())
                || is_nonzero_gaussian_rational(arena, c)
                || cache.query(arena, c, Props::NONZERO) == Some(true)
        })
        .then_some(factors)
}

/// Is `x` the product of `i` and real factors (`i`, `(14/3)·i`, `r·i` for a
/// real symbol `r`)?
fn is_real_times_i(arena: &Arena, x: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    if x == arena.i_unit {
        return true;
    }
    let ExprNode::Mul(children) = arena.node(x) else {
        return false;
    };
    let mut cache = AssumptionCache::new();
    children.iter().filter(|&&c| c == arena.i_unit).count() == 1
        && children.iter().all(|&c| {
            c == arena.i_unit
                || arena.as_num(c).is_some()
                || cache.query(arena, c, Props::REAL) == Some(true)
        })
}

/// Is `c` a canonical `a + b·i` with rational `a`, `b` and `b ≠ 0` (so
/// `c ≠ 0`: the assumption engine leaves `1 + i` open)?
fn is_nonzero_gaussian_rational(arena: &Arena, c: ExprId) -> bool {
    let imaginary = |t: ExprId| {
        t == arena.i_unit
            || matches!(arena.node(t), ExprNode::Mul(ch)
                if ch.len() == 2 && arena.as_num(ch[0]).is_some() && ch[1] == arena.i_unit)
    };
    match arena.node(c) {
        ExprNode::Add(terms) => {
            terms.len() == 2
                && ((arena.as_num(terms[0]).is_some() && imaginary(terms[1]))
                    || (imaginary(terms[0]) && arena.as_num(terms[1]).is_some()))
        }
        _ => imaginary(c),
    }
}

/// The terms of `z` split into real ones and the coefficients `c` of the
/// pure imaginary ones `i·c` (`c` real); `None` if a term is neither.
#[allow(clippy::type_complexity)]
fn split_real_imaginary_terms(
    arena: &mut Arena,
    z: ExprId,
) -> Option<(SmallVec<[ExprId; 4]>, SmallVec<[ExprId; 4]>)> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let terms: SmallVec<[ExprId; 4]> = match arena.node(z) {
        ExprNode::Add(children) => children.iter().copied().collect(),
        _ => smallvec![z],
    };
    let mut cache = AssumptionCache::new();
    let (mut real, mut imaginary) = (SmallVec::new(), SmallVec::new());
    for t in terms {
        if t == arena.i_unit {
            imaginary.push(arena.one);
            continue;
        }
        if let ExprNode::Mul(children) = arena.node(t)
            && children.iter().filter(|&&c| c == arena.i_unit).count() == 1
        {
            let rest: SmallVec<[ExprId; 4]> = children
                .iter()
                .copied()
                .filter(|&c| c != arena.i_unit)
                .collect();
            let c = canon_mul(arena, &rest);
            if cache.query(arena, c, Props::REAL) != Some(true) {
                return None;
            }
            imaginary.push(c);
        } else if cache.query(arena, t, Props::REAL) == Some(true) {
            real.push(t);
        } else {
            return None;
        }
    }
    Some((real, imaginary))
}

/// `c − 2πk ∈ (−π, π]` for a real symbol-free constant `c`: `k` from a
/// 20-digit value of `(c − π)/(2π)`, confirmed by the certified signs of
/// `c − 2πk ± π`; `None` when a sign is not certified (or `|k|` is huge).
fn principal_angle(arena: &mut Arena, c: ExprId) -> Option<ExprId> {
    if !crate::base::walk::free_symbols(arena, c).is_empty() {
        return None;
    }
    let pi = arena.pi;
    let minus_pi = canon_neg(arena, pi);
    let shifted = canon_add(arena, &[c, minus_pi]);
    let two_pi_inv = {
        let two = arena.int(2);
        let two_pi = canon_mul(arena, &[two, pi]);
        let neg_one = arena.neg_one;
        canon_pow(arena, two_pi, neg_one)
    };
    let turns = canon_mul(arena, &[shifted, two_pi_inv]);
    let estimate = arena.const_f64(turns, 20)?;
    if !estimate.is_finite() || estimate.abs() > 1e12 {
        return None;
    }
    let k = estimate.ceil() as i64;
    let reduced = if k == 0 {
        c
    } else {
        let factor = arena.int(-2 * k);
        let shift = canon_mul(arena, &[factor, pi]);
        canon_add(arena, &[c, shift])
    };
    let above = canon_add(arena, &[reduced, pi]);
    let below = canon_add(arena, &[reduced, minus_pi]);
    (arena.sign_of_real_constant(above)? == Ordering::Greater
        && arena.sign_of_real_constant(below)? == Ordering::Less)
        .then_some(reduced)
}

/// `q mod m` in `[0, m)` for `m > 0`: `(a mod b·m)/b` for `q = a/b`, in
/// lowest terms as `q` is (`gcd(a mod b·m, b) = gcd(a, b)`).  Integer
/// arithmetic, linear in the size of `a`: the `Ratio` operators reduce
/// with a gcd that is quadratic in it even against a one-word operand
/// (building `exp(q·π·i)` for `q = a/6` with a 5,000-digit `a` took
/// 1.5 ms, now 0.2 ms; the folding tables are consulted at every node).
fn mod_rational(q: &Q, m: i64) -> Q {
    use num_integer::Integer;
    let r = q.numer().mod_floor(&(q.denom() * BigInt::from(m)));
    if r.is_zero() {
        Q::zero()
    } else {
        Ratio::new_raw(r, q.denom().clone())
    }
}

/// `q` as a multiple of `1/n` (`q·n` when that is an integer), for
/// `n > 0`.
fn in_units(q: &Q, n: i64) -> Option<i64> {
    // `q·n = a·(n/b)` is an integer exactly when `b | n` (`q = a/b` in
    // lowest terms).
    let b = q.denom().to_i64()?;
    if b <= 0 || n % b != 0 {
        return None;
    }
    (q.numer() * BigInt::from(n / b)).to_i64()
}

/// `sin(q·π)` when it is rational.
fn sin_pi(q: &Q) -> Option<Folded> {
    Some(match in_units(&mod_rational(q, 2), 6)? {
        0 | 6 => Folded::int(0),
        1 | 5 => Folded::ratio(1, 2),
        3 => Folded::int(1),
        7 | 11 => Folded::ratio(-1, 2),
        9 => Folded::int(-1),
        _ => return None,
    })
}

/// `cos(q·π)` when it is rational.
fn cos_pi(q: &Q) -> Option<Folded> {
    Some(match in_units(&mod_rational(q, 2), 6)? {
        0 => Folded::int(1),
        2 | 10 => Folded::ratio(1, 2),
        3 | 9 => Folded::int(0),
        4 | 8 => Folded::ratio(-1, 2),
        6 => Folded::int(-1),
        _ => return None,
    })
}

/// `tan(q·π)` when it is rational or a pole.
fn tan_pi(q: &Q) -> Option<Folded> {
    Some(match in_units(&mod_rational(q, 1), 4)? {
        0 => Folded::int(0),
        1 => Folded::int(1),
        2 => Folded::ComplexInfinity,
        3 => Folded::int(-1),
        _ => return None,
    })
}

/// Is `q` an integer `≤ 0` (a pole of Γ)?
pub(super) fn is_nonpositive_int(q: &Q) -> bool {
    q.is_integer() && !q.is_positive()
}

/// The canonical value of a function application at an exact constant
/// argument, when that value is a **rational number (or a rational multiple
/// of `i`), `±∞`, `zoo` or `nan`**;
/// `None` for every other node (including every application whose value is
/// irrational, such as `sin(π/4)`, `exp(1)` or `acos(0)`, which stay for
/// `eval` to rewrite).
///
/// [`Arena::intern`] consults this for every node it is asked to create,
/// so no function application with such a value ever exists in an arena,
/// whichever path builds it (the constructors, substitution, parsing,
/// differentiation, …).  This is what makes the canonical arithmetic sound
/// at constants: `x·x⁻¹ → 1`, `x − x → 0`, `x⁰ → 1` and `0·x → 0` hold only
/// for a finite non-zero `x`, so before 0.30 `sin(0)/sin(0)` was `1`,
/// `ln(0) − ln(0)` was `0`, `(sin(8n)/(8·sin n))` at `n = 0` was `1/8` (the
/// limit, not the value) and `(1 − cos x)/x²` at `x = 0` was `zoo` (the
/// hidden zero `1 − cos 0` times `0⁻²`).  With the atom folded they are
/// `0·zoo = nan`, like SymPy's (whose `Function.eval` classmethods fold
/// the same special values when the application is built).
///
/// Rational values are folded, not only zeros and poles, because a
/// rational-valued atom is a non-canonical spelling of a number — like an
/// unreduced fraction — that hides zeros from the arithmetic: `1 − cos 0`,
/// `ln(cos 0)`, `sinh(iπ/2) − i`.  The tables are exact and structural (no
/// evaluation, no recursion): `sin`/`cos`/`tan` at rational multiples of `π`
/// with rational values (the argument may be `asin(1)`, `acos(½)`,
/// `atan(1)`, … times a rational: `tan(2·atan(1))` is `tan(π/2) = zoo`),
/// the hyperbolic functions at `0` and at `i·π·q` (`ln(−1) = iπ`),
/// `exp(0)`, `exp(iπk/2) = iᵏ`, `ln(0) = zoo`, `ln(1)`, `ln(e)`, the inverse functions at
/// their zeros and poles, `Γ`/`ln Γ`/`ψ`/`n!`/`B(a, b)` at their poles and
/// `ln Γ(1) = ln Γ(2) = 0`, `C(n, k)` where it vanishes,
/// `erf`/`erfc`/`W`/`⌊·⌋`/`⌈·⌉`/`H`/`δ` at numbers, `|·|` and `sign` of
/// numbers and of `π`, `e`, …, the values at `±∞` that are infinite or
/// rational, `f(nan) = nan`, and the zeros and poles of the library
/// functions (`J_ν(0)`, `erfinv(±1)`, …).  The positive-integer values of
/// `Γ`, `n!`, `C(n, k)` and `B(a, b)` (numbers that can be huge) are left
/// to `eval`, which bounds them by the digit guard of exact results.
///
/// Then the values that follow from the assumptions on the symbols of the
/// argument ([`assumed_value`]): `sin(nπ) = 0` for an integer `n`, `|p| =
/// p` for a positive `p`, …
pub(crate) fn canon_function(arena: &mut Arena, node: &ExprNode) -> Option<ExprId> {
    canon_function_exact(arena, node).or_else(|| assumed_value(arena, node))
}

/// The exact tables of [`canon_function`].
fn canon_function_exact(arena: &mut Arena, node: &ExprNode) -> Option<ExprId> {
    if let ExprNode::Min(args) | ExprNode::Max(args) = node {
        return fold_min_max(arena, args, matches!(node, ExprNode::Min(_)));
    }
    if let Some(v) = sign_abs_arg_of_directions(arena, node) {
        return Some(v);
    }
    // |c| = c for the positive constants π, e, γ, G, φ (`abs(π)` stayed an
    // atom and hid `sin(|π|/2) = 1` from the tables).
    if let ExprNode::Abs(a) = *node
        && is_positive_constant(arena, a)
    {
        return Some(a);
    }
    if let ExprNode::Sign(a) = *node
        && is_positive_constant(arena, a)
    {
        return Some(arena.one);
    }
    // |i^q| = |(−1)^q| = |e^(iπq)| = 1 for a rational q; |∏ cⱼ| = ∏ |cⱼ|
    // for a product of such units, numbers, positive constants and their
    // rational powers (`|π·i| = π`).
    if let ExprNode::Abs(a) = *node {
        let is_unit = |c: ExprId| match *arena.node(c) {
            ExprNode::ImaginaryUnit => true,
            ExprNode::Pow(b, e) => {
                let unit_base = b == arena.i_unit
                    || b == arena.neg_one
                    || matches!(classify_arg(arena, b), Arg::ImaginaryMultiple(q) if (-&q).is_one());
                unit_base && arena.as_num(e).is_some()
            }
            // |e^{ic}| = 1 for a real c (SymPy: `Abs(sqrt(2)*exp(14*I/3))`
            // → `sqrt(2)`; before 0.43 only for c a rational multiple of π).
            ExprNode::Exp(x) => {
                matches!(classify_arg(arena, x), Arg::ImaginaryPiMultiple(_))
                    || is_real_times_i(arena, x)
            }
            _ => false,
        };
        let is_positive = |c: ExprId| match *arena.node(c) {
            // |i·∞| = ∞ (SymPy: `Abs(oo*I)` → oo).
            ExprNode::Infinity => true,
            ExprNode::Pow(b, e) => {
                (is_positive_constant(arena, b) || arena.as_num(b).is_some_and(|q| q.is_positive()))
                    && arena.as_num(e).is_some()
            }
            _ => is_positive_constant(arena, c),
        };
        if is_unit(a) {
            return Some(arena.one);
        }
        if let ExprNode::Mul(children) = arena.node(a) {
            // (factor, |factor|): a number by its magnitude, a positive
            // factor by itself, a unit by 1; `None` for any other factor.
            let mut kept: SmallVec<[(ExprId, Option<Q>); 6]> = SmallVec::new();
            let mut all_known = true;
            for &c in children.iter() {
                if let Some(q) = arena.as_num(c) {
                    kept.push((c, Some(q.abs())));
                } else if is_positive(c) {
                    kept.push((c, None));
                } else if !is_unit(c) {
                    all_known = false;
                    break;
                }
            }
            if all_known {
                let factors: SmallVec<[ExprId; 6]> = kept
                    .into_iter()
                    .map(|(c, q)| match q {
                        Some(q) => {
                            let nid = arena.intern_num(q);
                            arena.intern(ExprNode::Num(nid))
                        }
                        None => c,
                    })
                    .collect();
                return Some(canon_mul(arena, &factors));
            }
        }
    }
    let folded = match *node {
        // At `±i∞` (SymPy: sin(oo*I) → oo*I, cos(oo*I) → oo, tan(oo*I) → I,
        // exp(oo*I) → nan, log(oo*I) → oo, asin(oo*I) → oo*I,
        // acos(oo*I) → pi/2 - oo*I, asinh(oo*I) → oo, acosh(oo*I) →
        // oo + I*pi/2, erf(oo*I) → oo*I, erfc(oo*I) → -oo*I, sign(oo*I) → I;
        // a finite term next to an infinite one is absorbed).
        //
        // At `c ± i·∞` and `±∞ + c` for a finite `c` (a sum the canonical
        // arithmetic keeps since 0.41, see `infinity_plus_terms`) the
        // functions take the limits along `c + i·t`, `±t + c` (mpmath at
        // t = 10²⁰, 10⁴⁰, 10⁸⁰): `sin(c + i·∞) = i·e^(−ic)·∞`,
        // `cos(c ± i·∞) = e^(∓ic)·∞` (`cos(π/2 + i·∞) = −i·∞`; SymPy:
        // nan), `e^(∞ + c) = e^c·∞` (SymPy: `exp(oo + I)` → `oo*exp(I)`),
        // `tanh(±∞ + c) = ±1`, `ln(c ± i·∞) = ∞ ± iπ/2`.
        ExprNode::Sin(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::PiMultiple(q) => sin_pi(&q)?,
            Arg::ImaginaryInfinity(neg) => Folded::ImaginaryInfinity(neg),
            Arg::InfinitePlus(InfDir::PosImag, c) => {
                let (minus_i, i) = (canon_neg(arena, arena.i_unit), arena.i_unit);
                exp_direction(arena, &c, minus_i, i)
            }
            Arg::InfinitePlus(InfDir::NegImag, c) => {
                let (minus_i, i) = (canon_neg(arena, arena.i_unit), arena.i_unit);
                exp_direction(arena, &c, i, minus_i)
            }
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Cos(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(1),
            Arg::PiMultiple(q) => cos_pi(&q)?,
            Arg::ImaginaryInfinity(_) => Folded::Infinity,
            Arg::InfinitePlus(d @ (InfDir::PosImag | InfDir::NegImag), c) => {
                let i = arena.i_unit;
                let inner = if d == InfDir::PosImag {
                    canon_neg(arena, i)
                } else {
                    i
                };
                exp_direction(arena, &c, inner, arena.one)
            }
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Tan(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::PiMultiple(q) => tan_pi(&q)?,
            Arg::ImaginaryInfinity(neg) => {
                Folded::Imaginary(if neg { -Q::one() } else { Q::one() })
            }
            Arg::InfinitePlus(InfDir::PosImag, _) => Folded::Imaginary(Q::one()),
            Arg::InfinitePlus(InfDir::NegImag, _) => Folded::Imaginary(-Q::one()),
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Exp(a) => match classify_arg(arena, a) {
            Arg::InfinitePlus(InfDir::Pos, c) => exp_direction(arena, &c, arena.one, arena.one),
            Arg::InfinitePlus(InfDir::Neg, _) => Folded::int(0),
            Arg::InfinitePlus(_, _) => Folded::NaN,
            Arg::Rational(q) if q.is_zero() => Folded::int(1),
            // e^{iπk/2} = i^k: 1, i, −1, −i.
            Arg::ImaginaryPiMultiple(q) => match in_units(&mod_rational(&q, 2), 2)? {
                0 => Folded::int(1),
                1 => Folded::Imaginary(Q::one()),
                2 => Folded::int(-1),
                3 => Folded::Imaginary(-Q::one()),
                _ => return None,
            },
            Arg::Infinity => Folded::Infinity,
            Arg::NegInfinity => Folded::int(0),
            Arg::ImaginaryInfinity(_) | Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        // ln(−∞) = ∞ + iπ, ln(±i·∞) = ∞ ± iπ/2 (mpmath `log(-1e20)` =
        // 46.05… + 3.14…j; SymPy: `log(-oo)` → oo, which loses the
        // imaginary part; up to 0.40 symplex too); ln(−∞ + c) = ∞ ± iπ by
        // the side of the cut `c` puts it on.
        ExprNode::Ln(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::ComplexInfinity,
            Arg::Rational(q) if q.is_one() => Folded::int(0),
            Arg::E => Folded::int(1),
            Arg::Infinity | Arg::InfinitePlus(InfDir::Pos, _) => Folded::Infinity,
            Arg::NegInfinity => Folded::InfinitePlus(InfDir::Pos, Q::one(), Q::zero()),
            Arg::InfinitePlus(InfDir::Neg, c) => match finite_part_sign(arena, &c, true)? {
                Ordering::Equal => return None,
                s => Folded::InfinitePlus(InfDir::Pos, signed(s, Q::one()), Q::zero()),
            },
            Arg::ImaginaryInfinity(true) | Arg::InfinitePlus(InfDir::NegImag, _) => {
                Folded::InfinitePlus(InfDir::Pos, Q::new((-1).into(), 2.into()), Q::zero())
            }
            Arg::ImaginaryInfinity(false) | Arg::InfinitePlus(InfDir::PosImag, _) => {
                Folded::InfinitePlus(InfDir::Pos, Q::new(1.into(), 2.into()), Q::zero())
            }
            Arg::ComplexInfinity => Folded::ComplexInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Sinh(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            // sinh(iπq) = i·sin(πq).
            Arg::ImaginaryPiMultiple(q) => times_i(sin_pi(&q)?)?,
            Arg::Infinity => Folded::Infinity,
            Arg::NegInfinity => Folded::NegInfinity,
            // sinh(±∞ + c) = ±e^(±c)·∞.
            Arg::InfinitePlus(InfDir::Pos, c) => exp_direction(arena, &c, arena.one, arena.one),
            Arg::InfinitePlus(InfDir::Neg, c) => {
                exp_direction(arena, &c, arena.neg_one, arena.neg_one)
            }
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Cosh(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(1),
            // cosh(iπq) = cos(πq).
            Arg::ImaginaryPiMultiple(q) => cos_pi(&q)?,
            Arg::Infinity | Arg::NegInfinity => Folded::Infinity,
            // cosh(±∞ + c) = e^(±c)·∞.
            Arg::InfinitePlus(InfDir::Pos, c) => exp_direction(arena, &c, arena.one, arena.one),
            Arg::InfinitePlus(InfDir::Neg, c) => exp_direction(arena, &c, arena.neg_one, arena.one),
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Tanh(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            // tanh(iπq) = i·tan(πq).
            Arg::ImaginaryPiMultiple(q) => times_i(tan_pi(&q)?)?,
            Arg::Infinity | Arg::InfinitePlus(InfDir::Pos, _) => Folded::int(1),
            Arg::NegInfinity | Arg::InfinitePlus(InfDir::Neg, _) => Folded::int(-1),
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        // SymPy: atan(I) → oo*I, atan(-I) → -oo*I, acos(oo) → oo*I (before
        // 0.30 the hunts recorded them as `zoo`, without their direction).
        // On the cuts the values are the limits along them, SymPy's
        // numerical convention (`N(asin(10**20))` = 1.5707… − 46.74…i):
        // asin(±∞) = ±π/2 ∓ i·∞, acos(−∞) = π − i·∞, acos(±i·∞) = π/2 ∓
        // i·∞ (SymPy: `asin(oo)` → -oo*I, `acos(-oo)` → -oo*I, dropping the
        // real part; up to 0.40 symplex too); off them the side decides:
        // asin(±∞ + c) = ±π/2 + sign(Im c)·i·∞, asin(c ± i·∞) = ±i·∞.
        ExprNode::Asin(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::Infinity => {
                Folded::InfinitePlus(InfDir::NegImag, Q::new(1.into(), 2.into()), Q::zero())
            }
            Arg::NegInfinity => {
                Folded::InfinitePlus(InfDir::PosImag, Q::new((-1).into(), 2.into()), Q::zero())
            }
            Arg::InfinitePlus(d @ (InfDir::Pos | InfDir::Neg), c) => {
                let half = Q::new(if d == InfDir::Pos { 1 } else { -1 }.into(), 2.into());
                match finite_part_sign(arena, &c, true)? {
                    Ordering::Greater => Folded::InfinitePlus(InfDir::PosImag, half, Q::zero()),
                    Ordering::Less => Folded::InfinitePlus(InfDir::NegImag, half, Q::zero()),
                    Ordering::Equal => return None,
                }
            }
            Arg::ImaginaryInfinity(neg) => Folded::ImaginaryInfinity(neg),
            Arg::InfinitePlus(d, _) => Folded::ImaginaryInfinity(d == InfDir::NegImag),
            Arg::ComplexInfinity => Folded::ComplexInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        // acos = π/2 − asin.
        ExprNode::Acos(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_one() => Folded::int(0),
            Arg::Infinity => Folded::ImaginaryInfinity(false),
            Arg::NegInfinity => Folded::InfinitePlus(InfDir::NegImag, Q::one(), Q::zero()),
            Arg::InfinitePlus(d @ (InfDir::Pos | InfDir::Neg), c) => {
                let q = if d == InfDir::Pos {
                    Q::zero()
                } else {
                    Q::one()
                };
                match finite_part_sign(arena, &c, true)? {
                    Ordering::Greater => Folded::InfinitePlus(InfDir::NegImag, q, Q::zero()),
                    Ordering::Less => Folded::InfinitePlus(InfDir::PosImag, q, Q::zero()),
                    Ordering::Equal => return None,
                }
            }
            // π/2 ∓ i∞.
            Arg::ImaginaryInfinity(neg) => {
                let d = if neg {
                    InfDir::PosImag
                } else {
                    InfDir::NegImag
                };
                Folded::InfinitePlus(d, Q::new(1.into(), 2.into()), Q::zero())
            }
            Arg::InfinitePlus(d, _) => {
                let d = if d == InfDir::NegImag {
                    InfDir::PosImag
                } else {
                    InfDir::NegImag
                };
                Folded::InfinitePlus(d, Q::new(1.into(), 2.into()), Q::zero())
            }
            Arg::ComplexInfinity => Folded::ComplexInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Atan(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            // The logarithmic branch points atan(±i) = ±i∞.
            Arg::ImaginaryMultiple(q) if q.is_one() || (-&q).is_one() => {
                Folded::ImaginaryInfinity(q.is_negative())
            }
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Asinh(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::Infinity => Folded::Infinity,
            Arg::NegInfinity => Folded::NegInfinity,
            Arg::InfinitePlus(InfDir::Pos, _) => Folded::Infinity,
            Arg::InfinitePlus(InfDir::Neg, _) => Folded::NegInfinity,
            // asinh(±i·∞) = ±∞ ± iπ/2 (SymPy's `N(asinh(10**20*I))` = 46.74… +
            // 1.5707…i; SymPy and up to 0.40 symplex: ±∞); off the cut the
            // sign of `Re c` picks the side: asinh(c + i·∞) = ±∞ + iπ/2.
            Arg::ImaginaryInfinity(neg) => {
                let (d, q) = if neg {
                    (InfDir::Neg, Q::new((-1).into(), 2.into()))
                } else {
                    (InfDir::Pos, Q::new(1.into(), 2.into()))
                };
                Folded::InfinitePlus(d, q, Q::zero())
            }
            Arg::InfinitePlus(d, c) => {
                let up = d == InfDir::PosImag;
                let q = Q::new(if up { 1 } else { -1 }.into(), 2.into());
                let real = match finite_part_sign(arena, &c, false)? {
                    Ordering::Greater => InfDir::Pos,
                    Ordering::Less => InfDir::Neg,
                    Ordering::Equal if up => InfDir::Pos,
                    Ordering::Equal => InfDir::Neg,
                };
                Folded::InfinitePlus(real, q, Q::zero())
            }
            Arg::ComplexInfinity => Folded::ComplexInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        // acosh(−∞) = ∞ + iπ, acosh(±i·∞) = ∞ ± iπ/2 (SymPy: `acosh(-oo)`
        // → oo, `acosh(oo*I)` → oo + I*pi/2; up to 0.40 symplex: ∞ for
        // all), acosh(−∞ + c) = ∞ ± iπ by the sign of `Im c`.
        ExprNode::Acosh(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_one() => Folded::int(0),
            Arg::Infinity | Arg::InfinitePlus(InfDir::Pos, _) => Folded::Infinity,
            Arg::NegInfinity => Folded::InfinitePlus(InfDir::Pos, Q::one(), Q::zero()),
            Arg::InfinitePlus(InfDir::Neg, c) => match finite_part_sign(arena, &c, true)? {
                Ordering::Equal => return None,
                s => Folded::InfinitePlus(InfDir::Pos, signed(s, Q::one()), Q::zero()),
            },
            Arg::ImaginaryInfinity(true) | Arg::InfinitePlus(InfDir::NegImag, _) => {
                Folded::InfinitePlus(InfDir::Pos, Q::new((-1).into(), 2.into()), Q::zero())
            }
            Arg::ImaginaryInfinity(false) | Arg::InfinitePlus(InfDir::PosImag, _) => {
                Folded::InfinitePlus(InfDir::Pos, Q::new(1.into(), 2.into()), Q::zero())
            }
            Arg::ComplexInfinity => Folded::ComplexInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Atanh(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::Rational(q) if q.is_one() => Folded::Infinity,
            Arg::Rational(q) if (-&q).is_one() => Folded::NegInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Atan2(y, x) => match (classify_arg(arena, y), classify_arg(arena, x)) {
            (Arg::NaN, _) | (_, Arg::NaN) => Folded::NaN,
            // atan2(0, 0) is undefined (SymPy: nan); atan2(0, x > 0) = 0.
            (Arg::Rational(y), Arg::Rational(x)) if y.is_zero() && x.is_zero() => Folded::NaN,
            (Arg::Rational(y), Arg::Rational(x)) if y.is_zero() && x.is_positive() => {
                Folded::int(0)
            }
            _ => return None,
        },
        // Γ(±i·∞) = 0 (`|Γ(iy)| = √(π/(y·sinh πy))`, DLMF 5.4.3); ln Γ grows
        // without a direction at `−∞` (its imaginary part is `−π⌊−x⌋` between
        // the poles) and at `±i·∞` (`(iy − 1/2)·ln(iy) − iy`), as SymPy's
        // `loggamma(-oo)`, `loggamma(oo*I)` → zoo; `ψ(±i·∞) = ∞ ± iπ/2` (`ψ(z) ~
        // ln z`; mpmath `digamma(10**6*1j)` = 13.8155 + 1.5708i, SymPy: `oo`).
        // Up to 0.53 they stayed, and `0/Γ(i·∞)` was 0.
        ExprNode::Gamma(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if is_nonpositive_int(&q) => Folded::ComplexInfinity,
            Arg::Infinity => Folded::Infinity,
            Arg::ImaginaryInfinity(_) => Folded::int(0),
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::LogGamma(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if is_nonpositive_int(&q) => Folded::Infinity,
            Arg::Rational(q) if q.is_one() || in_units(&q, 1) == Some(2) => Folded::int(0),
            Arg::Infinity => Folded::Infinity,
            Arg::NegInfinity | Arg::ImaginaryInfinity(_) => Folded::ComplexInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Digamma(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if is_nonpositive_int(&q) => Folded::ComplexInfinity,
            Arg::Infinity => Folded::Infinity,
            Arg::ImaginaryInfinity(neg) => {
                Folded::InfinitePlus(InfDir::Pos, signed_half(neg), Q::zero())
            }
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Erf(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::Infinity | Arg::InfinitePlus(InfDir::Pos, _) => Folded::int(1),
            Arg::NegInfinity | Arg::InfinitePlus(InfDir::Neg, _) => Folded::int(-1),
            Arg::ImaginaryInfinity(neg) => Folded::ImaginaryInfinity(neg),
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Erfc(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(1),
            Arg::Infinity | Arg::InfinitePlus(InfDir::Pos, _) => Folded::int(0),
            Arg::NegInfinity | Arg::InfinitePlus(InfDir::Neg, _) => Folded::int(2),
            // erfc(±i·∞) = 1 ∓ i·∞ (`erfc(iy) = 1 − i·erfi(y)`; SymPy and up
            // to 0.40 symplex: ∓i·∞, real part lost).
            Arg::ImaginaryInfinity(neg) => {
                let d = if neg {
                    InfDir::PosImag
                } else {
                    InfDir::NegImag
                };
                Folded::InfinitePlus(d, Q::zero(), Q::one())
            }
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        // W(−∞) = ∞ + iπ, W(±i·∞) = ∞ ± iπ/2 (`W(z) ~ ln z − ln ln z`; mpmath
        // `lambertw(-10**6)` = 11.35 + 2.89i, tending to `π`; SymPy keeps
        // them).
        // At `±∞ + c`, `c ± i·∞` (`c` finite) as `ln`, whose imaginary part
        // `arg z` is `Im W`'s limit: `W(y + ∞) = ∞` (up to 0.56 it stayed,
        // and `solve(x + W(y + ∞))` refused).
        ExprNode::LambertW(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::E => Folded::int(1),
            Arg::Infinity => Folded::Infinity,
            Arg::NegInfinity => Folded::InfinitePlus(InfDir::Pos, Q::one(), Q::zero()),
            Arg::ImaginaryInfinity(neg) => {
                Folded::InfinitePlus(InfDir::Pos, signed_half(neg), Q::zero())
            }
            Arg::InfinitePlus(d, ref c)
                if c.iter()
                    .all(|&t| ValueClass::of(arena, t).is_finite_proved()) =>
            {
                match d {
                    InfDir::Pos => Folded::Infinity,
                    InfDir::Neg => match finite_part_sign(arena, c, true)? {
                        Ordering::Equal => return None,
                        s => Folded::InfinitePlus(InfDir::Pos, signed(s, Q::one()), Q::zero()),
                    },
                    InfDir::NegImag => {
                        Folded::InfinitePlus(InfDir::Pos, Q::new((-1).into(), 2.into()), Q::zero())
                    }
                    InfDir::PosImag => {
                        Folded::InfinitePlus(InfDir::Pos, Q::new(1.into(), 2.into()), Q::zero())
                    }
                }
            }
            Arg::NaN => Folded::NaN,
            // W(−1/e) = −1: the argument is `−exp(−1)`.
            Arg::Other if is_minus_inverse_e(arena, a) => Folded::int(-1),
            _ => return None,
        },
        ExprNode::Floor(a) => match classify_arg(arena, a) {
            Arg::Rational(q) => Folded::Rational(q.floor()),
            Arg::Infinity => Folded::Infinity,
            Arg::NegInfinity => Folded::NegInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Ceiling(a) => match classify_arg(arena, a) {
            Arg::Rational(q) => Folded::Rational(q.ceil()),
            Arg::Infinity => Folded::Infinity,
            Arg::NegInfinity => Folded::NegInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Heaviside(a) => match classify_arg(arena, a) {
            // H(0) = 1/2, the convention of `eval`.
            Arg::Rational(q) if q.is_zero() => Folded::ratio(1, 2),
            Arg::Rational(q) if q.is_positive() => Folded::int(1),
            Arg::Rational(_) | Arg::NegInfinity => Folded::int(0),
            Arg::Infinity => Folded::int(1),
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::DiracDelta(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if !q.is_zero() => Folded::int(0),
            Arg::Infinity | Arg::NegInfinity => Folded::int(0),
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Abs(a) => match classify_arg(arena, a) {
            Arg::Rational(q) => Folded::Rational(q.abs()),
            Arg::Infinity | Arg::NegInfinity | Arg::ComplexInfinity => Folded::Infinity,
            // |∞ + i| = ∞ (SymPy: `Abs(oo + I)` → oo).
            Arg::InfinitePlus(_, _) => Folded::Infinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Sign(a) => match classify_arg(arena, a) {
            Arg::Rational(q) => Folded::Rational(q.signum()),
            Arg::Infinity => Folded::int(1),
            Arg::NegInfinity => Folded::int(-1),
            Arg::ImaginaryInfinity(neg) => {
                Folded::Imaginary(if neg { -Q::one() } else { Q::one() })
            }
            // sign(z) = z/|z| → the direction of the infinite term.
            Arg::InfinitePlus(InfDir::Pos, _) => Folded::int(1),
            Arg::InfinitePlus(InfDir::Neg, _) => Folded::int(-1),
            Arg::InfinitePlus(InfDir::PosImag, _) => Folded::Imaginary(Q::one()),
            Arg::InfinitePlus(InfDir::NegImag, _) => Folded::Imaginary(-Q::one()),
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Factorial(a) => match classify_arg(arena, a) {
            // n! = Γ(n + 1): poles at the negative integers, 0 at `±i·∞`.
            Arg::Rational(q) if q.is_integer() && q.is_negative() => Folded::ComplexInfinity,
            Arg::Infinity => Folded::Infinity,
            Arg::ImaginaryInfinity(_) => Folded::int(0),
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Binomial(n, k) => match (classify_arg(arena, n), classify_arg(arena, k)) {
            (Arg::NaN, _) | (_, Arg::NaN) => Folded::NaN,
            // C(x, k) at `x = iᵗ·∞`: the polynomial `x(x − 1)⋯(x − k + 1)/k!`
            // for an integer `k ≥ 0` (0 for `k < 0`), `~ x^k/Γ(k + 1)` at `+∞`
            // otherwise (mpmath `binomial(10**6, 0.5)` = 1128.4; SymPy: nan);
            // `C(n, ±∞) = 0` for `n > −1` (`|C(n, k)| ~ |k|^{−n−1}/|Γ(−n)|`).
            (Arg::Infinity | Arg::NegInfinity | Arg::ImaginaryInfinity(_), Arg::Rational(k))
                if k.is_integer() =>
            {
                let turns = match classify_arg(arena, n) {
                    Arg::Infinity => 0,
                    Arg::ImaginaryInfinity(false) => 1,
                    Arg::NegInfinity => 2,
                    _ => 3,
                };
                if k.is_negative() {
                    Folded::int(0)
                } else {
                    polynomial_at_infinity(&k.to_integer(), false, turns)
                }
            }
            (Arg::Infinity, Arg::Rational(k)) if k.is_negative() => Folded::int(0),
            (Arg::Infinity, Arg::Rational(_)) => Folded::Infinity,
            (Arg::Rational(n), Arg::Infinity | Arg::NegInfinity) if n > -Q::one() => Folded::int(0),
            (Arg::Rational(n), Arg::Rational(k)) if k.is_integer() => {
                // C(n, k) = Γ(n+1)/(Γ(k+1)·Γ(n−k+1)).  For n ≥ 0 an integer
                // the value is 0 unless 0 ≤ k ≤ n; for a non-integer n,
                // 1/Γ(k+1) = 0 at a negative k.  A negative integer k gives 0
                // for every n (Graham–Knuth–Patashnik, as `eval` since 0.53:
                // Pascal's rule for all integers); a negative integer n with
                // k ≥ 0 is left to `eval`.
                let vanishes = if n.is_integer() {
                    k.is_negative() || (!n.is_negative() && k > n)
                } else {
                    k.is_negative()
                };
                if !vanishes {
                    return None;
                }
                Folded::int(0)
            }
            _ => return None,
        },
        ExprNode::Beta(a, b) => match (classify_arg(arena, a), classify_arg(arena, b)) {
            (Arg::NaN, _) | (_, Arg::NaN) => Folded::NaN,
            (Arg::Rational(a), Arg::Rational(b)) if beta_has_pole(&a, &b) => {
                Folded::ComplexInfinity
            }
            // B(a, x) ~ Γ(a)·x^{−a} for `x → ∞` off the negative axis: 0 for
            // `a > 0` (SymPy: `beta(1, oo)` → 0), `Γ(a)·∞` for `a < 0` at `+∞`
            // (mpmath `beta(-0.5, 10**6)` = −3544.9), `zoo` at a pole of `Γ(a)`.
            (Arg::Rational(q), Arg::Infinity | Arg::ImaginaryInfinity(_))
            | (Arg::Infinity | Arg::ImaginaryInfinity(_), Arg::Rational(q))
                if q.is_positive() =>
            {
                Folded::int(0)
            }
            (Arg::Rational(q), Arg::Infinity) | (Arg::Infinity, Arg::Rational(q)) => {
                if is_nonpositive_int(&q) {
                    Folded::ComplexInfinity
                } else if gamma_is_negative(&q) {
                    Folded::NegInfinity
                } else {
                    Folded::Infinity
                }
            }
            _ => return None,
        },
        ExprNode::Lib(f, ref args) => match lib_function_value(arena, f, args) {
            Some(v) => v,
            None => match at_complex_infinity(arena, f, args) {
                Some(AtComplexInfinity::ComplexInfinity) => Folded::ComplexInfinity,
                Some(AtComplexInfinity::Integer(k)) => Folded::int(k),
                Some(AtComplexInfinity::Constant) | None => match at_infinity(arena, f, args) {
                    AtInfinity::Value(v) if off_the_pole(arena, f, args) => v,
                    AtInfinity::Value(_) | AtInfinity::NoFiniteValue | AtInfinity::Untabulated => {
                        return None;
                    }
                },
            },
        },
        _ => return None,
    };
    Some(match folded {
        Folded::Rational(q) => {
            if q.is_zero() {
                arena.zero
            } else if q.is_one() {
                arena.one
            } else {
                let nid = arena.intern_num(q);
                arena.intern(ExprNode::Num(nid))
            }
        }
        Folded::Imaginary(q) => {
            let i = arena.i_unit;
            if q.is_one() {
                i
            } else {
                let nid = arena.intern_num(q);
                let c = arena.intern(ExprNode::Num(nid));
                canon_mul(arena, &[c, i])
            }
        }
        Folded::Infinity => arena.infinity,
        Folded::NegInfinity => arena.neg_infinity,
        Folded::ImaginaryInfinity(neg) => {
            let i = arena.i_unit;
            let direction = if neg { canon_neg(arena, i) } else { i };
            canon_mul(arena, &[direction, arena.infinity])
        }
        Folded::InfinitePlus(d, q, r) => {
            let i = arena.i_unit;
            let mut terms: SmallVec<[ExprId; 3]> = SmallVec::new();
            terms.push(match d {
                InfDir::Pos => arena.infinity,
                InfDir::Neg => arena.neg_infinity,
                InfDir::PosImag => canon_mul(arena, &[i, arena.infinity]),
                InfDir::NegImag => canon_mul(arena, &[arena.neg_one, i, arena.infinity]),
            });
            if !q.is_zero() {
                let nid = arena.intern_num(q);
                let c = arena.intern(ExprNode::Num(nid));
                let pi = arena.pi;
                terms.push(if matches!(d, InfDir::Pos | InfDir::Neg) {
                    canon_mul(arena, &[c, pi, i])
                } else {
                    canon_mul(arena, &[c, pi])
                });
            }
            if !r.is_zero() {
                let nid = arena.intern_num(r);
                terms.push(arena.intern(ExprNode::Num(nid)));
            }
            canon_add(arena, &terms)
        }
        Folded::Built(id) => id,
        Folded::Gamma(s) => arena.gamma(s),
        Folded::SqrtPiPlus(q, d) => {
            let half = arena.intern_num(Q::new(1.into(), 2.into()));
            let half = arena.intern(ExprNode::Num(half));
            let root_pi = arena.pow(arena.pi, half);
            let c = arena.intern_num(q);
            let c = arena.intern(ExprNode::Num(c));
            let finite = canon_mul(arena, &[c, root_pi]);
            let i = arena.i_unit;
            let infinite = match d {
                InfDir::Pos => arena.infinity,
                InfDir::Neg => arena.neg_infinity,
                InfDir::PosImag => canon_mul(arena, &[i, arena.infinity]),
                InfDir::NegImag => canon_mul(arena, &[arena.neg_one, i, arena.infinity]),
            };
            canon_add(arena, &[finite, infinite])
        }
        Folded::PiImaginary(q) => {
            let c = arena.intern_num(q);
            let c = arena.intern(ExprNode::Num(c));
            let (pi, i) = (arena.pi, arena.i_unit);
            canon_mul(arena, &[c, pi, i])
        }
        Folded::ComplexInfinity => arena.complex_infinity,
        Folded::NaN => arena.nan,
    })
}
