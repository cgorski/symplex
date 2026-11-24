//! Canonical-form constructors for arithmetic nodes.
//!
//! Each function here is the "canonical" entry-point called by the public
//! [`Arena::add`], [`Arena::mul`], [`Arena::pow`], and [`Arena::neg`] methods.
//!
//! # Canonicalization rules
//!
//! ## Add
//!
//! 1. Flatten nested `Add` (explicit stack, no recursion).
//! 2. Combine like terms: `2*x + 3*x → 5*x`.
//! 3. Numeric constant collected separately, placed first if nonzero.
//! 4. Remaining terms sorted by [`SortKey`](crate::base::sort_key::SortKey).
//! 5. Zero-coefficient terms dropped.
//! 6. `NaN` propagation: any `NaN` term ⟹ result is `NaN`.
//! 7. `zoo` absorbs every finite term; `±∞` (and a directed infinity of
//!    real direction) only the real ones, `∞ + i` stays a sum, whose
//!    negative powers are 0 (see [`infinity_plus_terms`], [`canon_pow`]); a
//!    directed infinity `x·∞` whose coefficients cancel is `NaN` (see
//!    [`add_directed_infinities`]), and so are cancelling like terms of an
//!    infinite sum ([`cancels_infinite_terms`]).
//!
//! ## Mul
//!
//! 1. Flatten nested `Mul` (explicit stack, no recursion).
//! 2. Collect running numeric coefficient.
//! 3. Combine like bases: `x * x → x²`, `x² * x³ → x⁵`.  Not the base 0:
//!    `0^y` is 0, 1 or `zoo` by the sign of `re y`, so its exponents add only
//!    where the powers are of one kind (`0^y·0^(2y) = 0^(3y)`); `0^y·0^z`
//!    stays, and `0^y·0^(−y)` is `Piecewise((1, Eq(y, 0)), (nan, True))`
//!    (see `push_zero_power_factors`).
//! 4. Numeric coefficient placed first if ≠ 1.
//! 5. Remaining factors sorted by [`SortKey`](crate::base::sort_key::SortKey).
//! 6. Zero propagation: any zero factor ⟹ result is `0`, except that
//!    `0 × (±∞ | zoo | nan) ⟹ NaN` regardless of argument order.  A function
//!    application at an exact argument where it is infinite (`ln(0)`,
//!    `exp(∞)`) never reaches here: it is folded when interned (see
//!    *Function applications* below).  Other applications (`Γ(zoo)`,
//!    `f(x)`) count as finite.  A denominator that is 0 on a region of its
//!    symbols (`0^y` on a half plane, a step `H(y)`, `⌊y⌋`, `max(y, 0)`, a
//!    delta) is not a generic nonzero value: `0/0^y` and `0/H(y)` are a
//!    `Piecewise` (see `zero_times_rest`), and so is `zoo` times such a
//!    factor (`handle_mul_with_zoo`).
//! 7. `NaN` propagation.
//! 8. `±∞` absorbs factors of known sign; factors of unknown direction
//!    stay: `x·∞`, `−i·∞` (a *directed infinity*, see
//!    [`directed_infinity`]); `zoo` absorbs every non-zero factor.
//!
//! ## Function applications
//!
//! [`canon_function`], consulted by `Arena::intern` for every node, folds
//! an application at an exact constant argument whose value is rational,
//! `±∞`, `zoo` or `nan`: `sin(0) → 0`, `cos(π) → −1`, `ln(0) → zoo`,
//! `tan(π/2) → zoo`, `Γ(0) → zoo`, `exp(−∞) → 0`.  Irrational special
//! values (`sin(π/4)`, `exp(1)`) are left to `eval`.
//!
//! ## Pow
//!
//! - `x⁰ → 1`, `x¹ → x`, `1ˣ → 1`, `0^(pos) → 0`.
//! - `(0^e)^q → 0^(q·e)` for a rational `q ≠ 0` (one form for the powers
//!   of 0: `1/0^y` is `0^(−y)`).
//! - Numeric base/exp evaluated when exp is integer with `|exp| ≤ max_pow_exponent`.
//! - `NaN` propagation.
//!
//! ## Neg
//!
//! - `Neg(Neg(x)) → x`, `Neg(Num(n)) → Num(−n)`.
//! - `Neg(Add(…))` distributes: `−(a+b) → (−a)+(−b)`.
//! - Otherwise normalises to `Mul(−1, x)`.

use std::cmp::Ordering;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Pow as NumPow, Signed, ToPrimitive, Zero};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{SmallVec, smallvec};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::{Q, q_add, q_mul};
use crate::base::value_class::{self, ValueClass};

mod assumed;
mod infinite_args;
mod mul_vanishing;
mod sets;
mod special_values;
mod verify;
use assumed::*;
use infinite_args::*;
pub(crate) use mul_vanishing::*;
pub(crate) use sets::*;
pub(crate) use special_values::*;
pub(crate) use verify::*;

// ═══════════════════════════════════════════════════════════════════════════
// Add
// ═══════════════════════════════════════════════════════════════════════════

/// Build a canonical `Add` node from the given summands.
///
/// Uses an explicit stack for flattening (never recurses) and an
/// [`FxHashMap`] for like-term collection.
pub(crate) fn canon_add(arena: &mut Arena, args: &[ExprId]) -> ExprId {
    // Fast path: 0 or 1 arguments
    if args.is_empty() {
        return arena.zero;
    }
    if args.len() == 1 {
        return args[0];
    }

    // Fast path: 2 numeric args → skip HashMap
    if args.len() == 2 {
        let (a, b) = (args[0], args[1]);
        if let (Some(na), Some(nb)) = (arena.as_num(a), arena.as_num(b)) {
            // `q_add`, not `Ratio`'s `+`: its gcd is quadratic in the size
            // of a huge integer even against 1 (see `numeric::q_add`).
            let sum = q_add(na, nb);
            if sum.is_zero() {
                tracing::debug!("canon_add fast path: two numerics sum to zero");
                return arena.zero;
            }
            let nid = arena.intern_num(sum);
            return arena.intern(ExprNode::Num(nid));
        }
    }

    // Running numeric constant (the "coefficient of 1").
    let mut constant: Q = Ratio::zero();

    // Map: symbolic_key → accumulated coefficient.
    let mut terms: FxHashMap<ExprId, Q> = FxHashMap::default();

    // Track whether we've seen infinity / neg-infinity to handle oo − oo → NaN.
    let mut has_pos_inf = false;
    let mut has_neg_inf = false;
    let mut has_zoo = false;

    // Keys that collected a second term (see `merges_undefined_terms`).
    let mut merged: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut collect = |terms: &mut FxHashMap<ExprId, Q>, key: ExprId, c: Q| match terms.entry(key) {
        std::collections::hash_map::Entry::Occupied(mut slot) => {
            let sum = q_add(slot.get(), &c);
            *slot.get_mut() = sum;
            merged.push(key);
        }
        std::collections::hash_map::Entry::Vacant(slot) => {
            slot.insert(c);
        }
    };

    // Explicit stack for iterative flattening.
    let mut stack: SmallVec<[ExprId; 16]> = SmallVec::from_slice(args);

    while let Some(id) = stack.pop() {
        match arena.node(id).clone() {
            ExprNode::Add(children) => {
                // Flatten: push children back onto the stack.
                stack.extend_from_slice(&children);
            }

            ExprNode::NaN => {
                return arena.nan;
            }

            ExprNode::Infinity => {
                if has_neg_inf || has_zoo {
                    return arena.nan; // oo + (-oo) → NaN, oo + zoo → NaN
                }
                has_pos_inf = true;
            }

            ExprNode::NegInfinity => {
                if has_pos_inf || has_zoo {
                    return arena.nan; // (-oo) + oo → NaN, (-oo) + zoo → NaN
                }
                has_neg_inf = true;
            }

            ExprNode::ComplexInfinity => {
                if has_zoo || has_pos_inf || has_neg_inf {
                    return arena.nan; // zoo + any infinity → NaN
                }
                has_zoo = true;
            }

            ExprNode::Num(nid) => {
                constant = q_add(&constant, arena.num(nid));
            }

            ExprNode::Neg(inner) => {
                // −x has coefficient −1, term x.
                let (c, key) = arena.as_coeff_term(inner);
                let combined = -c;
                if key == arena.one {
                    constant = q_add(&constant, &combined);
                } else {
                    collect(&mut terms, key, combined);
                }
            }

            _ => {
                let (c, key) = arena.as_coeff_term(id);
                if key == arena.one {
                    constant = q_add(&constant, &c);
                } else {
                    collect(&mut terms, key, c);
                }
            }
        }
    }

    if (has_pos_inf || has_neg_inf || has_zoo) && has_identically_infinite_term(arena, &terms) {
        tracing::debug!("canon_add: infinity plus a term over an identically zero sum → nan");
        return arena.nan;
    }

    if !merged.is_empty() && cancels_infinite_terms(arena, &terms, &merged) {
        tracing::debug!("canon_add: like infinite terms cancel → nan");
        return arena.nan;
    }

    // A directed infinity `x·∞` is an infinite term too.
    if terms.keys().any(|&k| is_directed_infinity(arena, k)) {
        let infinities = Infinities {
            pos: has_pos_inf,
            neg: has_neg_inf,
            zoo: has_zoo,
        };
        return add_directed_infinities(arena, &terms, &constant, infinities);
    }

    // `zoo` absorbs every finite term; `±∞` the real ones only (see
    // `infinity_plus_terms`).
    if has_zoo {
        return arena.complex_infinity;
    }
    if has_pos_inf || has_neg_inf {
        let inf = if has_pos_inf {
            arena.infinity
        } else {
            arena.neg_infinity
        };
        return infinity_plus_terms(arena, &terms, &constant, &[inf], true);
    }

    if !merged.is_empty() && merges_undefined_terms(arena, &merged) {
        tracing::debug!("canon_add: like terms undefined at every point collected → nan");
        return arena.nan;
    }

    // Collect non-zero terms.
    let non_zero_terms: SmallVec<[(ExprId, Q); 8]> =
        terms.into_iter().filter(|(_, c)| !c.is_zero()).collect();

    // Build the result argument list.
    let mut result_args: SmallVec<[ExprId; 6]> = SmallVec::new();

    // Numeric constant goes first (if nonzero).
    if !constant.is_zero() {
        let nid = arena.intern_num(constant);
        result_args.push(arena.intern(ExprNode::Num(nid)));
    }

    // Reconstruct symbolic terms, then sort by the RECONSTRUCTED expr's
    // sort key so that e.g. `x**2` (Pow, rank 20) sorts before `2*x`
    // (Mul, rank 30).
    let mut symbolic_args: SmallVec<[ExprId; 6]> = non_zero_terms
        .into_iter()
        .map(|(key, coeff)| arena.make_coeff_term(coeff, key))
        .collect();
    symbolic_args.sort_by(|a, b| arena.sort_key(*a).cmp(arena.sort_key(*b)));
    result_args.extend(symbolic_args);

    // Final assembly.
    let result = match result_args.len() {
        0 => arena.zero,
        1 => result_args[0],
        _ => arena.intern(ExprNode::Add(result_args)),
    };
    #[cfg(debug_assertions)]
    {
        let errors = verify_canonical_shallow(arena, result);
        if !errors.is_empty() {
            tracing::debug!("canon_add: non-canonical result: {:?}", errors);
        }
    }
    result
}

/// Is one of the `keys` under which [`canon_add`] collected two or more
/// terms (`c₁·k + c₂·k → (c₁ + c₂)·k`) infinite or undefined at every point?
/// Then each term is `zoo` (or `nan`) everywhere and their sum `zoo + zoo =
/// nan`, not the collected `(c₁ + c₂)·k = zoo` (nor `0` when the
/// coefficients cancel).  A key is so when it has a factor `d⁻ᵏ` whose
/// non-atomic base vanishes ([`factor_vanishes`]) or is undefined, or a sum
/// factor that is `zoo` or `nan` at every point by a term over such a
/// denominator ([`everywhere`]).  Up to 0.33 `x/d + (x + 3)²/d` with
/// `d = (x + y)² − x² − 2xy − y²`, substituted `x = 3/19`, collected
/// `(3657/361)/d = zoo`, and a later division made the expression `0`
/// instead of `nan`.  Only keys that merged, only their negative powers of
/// non-atoms and their sum factors are looked at, and not inside a
/// confirmation ([`vanishing_check_active`]).
fn merges_undefined_terms(arena: &mut Arena, keys: &[ExprId]) -> bool {
    if vanishing_check_active() {
        return false;
    }
    // A set, not a list: a sum of n terms collected into few keys merges
    // about n times, and a linear `contains` made this quadratic (the
    // expansion of a power in Gruntz's rewrite of `1/(ln x·atanh(atanh x))`
    // at `x = 1` spent 40 s here).
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    let mut memo = EverywhereMemo::default();
    for &key in keys {
        if !seen.insert(key) {
            continue;
        }
        if term_infinite_or_undefined(arena, &mut memo, key) {
            return true;
        }
    }
    false
}

/// Is the term `t` of a sum (a product, or a single factor) `zoo` or `nan`
/// at every point by one of its factors: a negative power `d⁻ᵏ` of a
/// non-atom `d` that is `0` or `nan` there, or a sum (or a positive power
/// of one) that is `zoo` or `nan` there ([`everywhere`])?  The factors
/// themselves only: a function factor is not looked into.
fn term_infinite_or_undefined(arena: &mut Arena, memo: &mut EverywhereMemo, t: ExprId) -> bool {
    let factors: SmallVec<[ExprId; 6]> = match arena.node(t) {
        ExprNode::Mul(children) => children.clone(),
        _ => smallvec![t],
    };
    factors.into_iter().any(|f| match *arena.node(f) {
        ExprNode::Pow(base, e)
            if !arena.node(base).is_atom() && arena.as_num(e).is_some_and(Signed::is_negative) =>
        {
            memo.zero_or_undefined(arena, base)
        }
        ExprNode::Pow(base, e)
            if matches!(arena.node(base), ExprNode::Add(_))
                && arena.as_num(e).is_some_and(Signed::is_positive) =>
        {
            memo.infinite_or_undefined(arena, base)
        }
        ExprNode::Add(_) => memo.infinite_or_undefined(arena, f),
        _ => false,
    })
}

/// Is one of the `terms` of a sum that also holds `±∞` or `zoo` infinite
/// (or undefined) at every point ([`term_infinite_or_undefined`]): a
/// product with a factor `d⁻ᵏ` whose base vanishes identically
/// ([`factor_vanishes`]) or is undefined, or a sum factor that is `zoo` or
/// `nan`?  Then the sum is `zoo + zoo` or `∞ + zoo`, `nan`, not the
/// infinity: `zoo + 2/((x + y)² − x² − 2xy − y²)` (up to 0.32 `zoo`, which
/// `subs` of a number into `x/(x² + 2x − (x + 1)² + 1) + 2/((x + y)² − x²
/// − 2xy − y²)` gave while the expression is `nan` at every point).  Only
/// on this path (a sum with an infinity); the directed infinities `x·∞`
/// are [`add_directed_infinities`]'s.
fn has_identically_infinite_term(arena: &mut Arena, terms: &FxHashMap<ExprId, Q>) -> bool {
    let keys: SmallVec<[ExprId; 6]> = terms
        .keys()
        .copied()
        .filter(|&t| !is_directed_infinity(arena, t))
        .collect();
    let mut memo = EverywhereMemo::default();
    keys.into_iter()
        .any(|t| term_infinite_or_undefined(arena, &mut memo, t))
}

/// The bare infinities a sum holds.
#[derive(Clone, Copy)]
struct Infinities {
    pos: bool,
    neg: bool,
    zoo: bool,
}

/// The sum of collected `terms` (coefficient per key) of which at least one
/// is a directed infinity `x·∞` (see [`directed_infinity`]), plus the bare
/// infinities seen and the rational `constant`.  A symbol is finite, so
/// `x·∞` is `nan` (at `x = 0`) or infinite in the direction of `x`:
///
/// * a finite term stays unless it is real and some infinite term has a
///   real direction ([`infinity_plus_terms`]): `x·∞ + 1` is `1 + i·∞` at
///   `x = i`, whose real part is 1 (SymPy keeps `oo*x + 1` and `1 + oo*I`);
///   up to 0.40 every finite term was absorbed, `π/2 + i·∞` was `i·∞` and
///   `cos(π/2 + i·∞)` became `cos(i·∞) = ∞` (the limit is `−i·∞`);
/// * a directed infinity whose coefficients cancel is `nan`
///   (`x·∞ − x·∞`; SymPy `oo*x - oo*x` → nan); other coefficients keep only
///   their sign (`x·∞ + x·∞ = x·∞`);
/// * `zoo` plus anything infinite or `nan` is `nan`;
/// * `±∞` and directed infinities in other directions stay a sum
///   (`∞ + x·∞` is `∞` for `x > 0` and `nan` otherwise), like SymPy's
///   `oo*x + oo`.
fn add_directed_infinities(
    arena: &mut Arena,
    terms: &FxHashMap<ExprId, Q>,
    constant: &Q,
    infinities: Infinities,
) -> ExprId {
    if infinities.zoo {
        return arena.nan;
    }
    let mut out: SmallVec<[ExprId; 6]> = SmallVec::new();
    let mut real_direction = infinities.pos || infinities.neg;
    for (&key, c) in terms {
        if !is_directed_infinity(arena, key) {
            continue;
        }
        if c.is_zero() {
            tracing::debug!("canon_add: directed infinities cancel → nan");
            return arena.nan;
        }
        real_direction |= has_real_direction(arena, key);
        // `canon_mul` keeps only the sign of the coefficient.
        out.push(arena.make_coeff_term(c.clone(), key));
    }
    if infinities.pos {
        out.push(arena.infinity);
    }
    if infinities.neg {
        out.push(arena.neg_infinity);
    }
    infinity_plus_terms(arena, terms, constant, &out, real_direction)
}

/// The sum of the `infinite` terms of a sum (`±∞`, directed infinities;
/// at least one) and those of its finite terms — the rational `constant`
/// and the collected `terms` that are not directed infinities — that are
/// not absorbed.  With `absorb_real` (some infinite term is `±∞` or has a
/// real direction) a term known to be real is absorbed (`∞ + π = ∞`, `x·∞ +
/// 1 = x·∞` for a real `x`); every other finite term stays: `∞ + i` is not
/// `∞`, its imaginary part is 1 and `atanh(∞ + i) = iπ/2` while `atanh(∞)
/// = −iπ/2` (SymPy's `Add.flatten` absorbs only real terms into `oo`: `oo +
/// I` and `oo + x` stay).  Up to 0.40 every finite term was absorbed: `oo +
/// I` was `oo`, so `im(oo + I)` was 0, `oo + I` claimed positive and real.
fn infinity_plus_terms(
    arena: &mut Arena,
    terms: &FxHashMap<ExprId, Q>,
    constant: &Q,
    infinite: &[ExprId],
    absorb_real: bool,
) -> ExprId {
    use crate::base::assumptions::{AssumptionCache, Props};
    let mut out: SmallVec<[ExprId; 6]> = SmallVec::from_slice(infinite);
    if !absorb_real && !constant.is_zero() {
        let nid = arena.intern_num(constant.clone());
        out.push(arena.intern(ExprNode::Num(nid)));
    }
    let mut cache = AssumptionCache::new();
    for (&key, c) in terms {
        if c.is_zero() || is_directed_infinity(arena, key) {
            continue;
        }
        if absorb_real && cache.query(arena, key, Props::REAL) == Some(true) {
            continue;
        }
        out.push(arena.make_coeff_term(c.clone(), key));
    }
    if out.len() == 1 {
        return out[0];
    }
    out.sort_by(|a, b| arena.sort_key(*a).cmp(arena.sort_key(*b)));
    let result = arena.intern(ExprNode::Add(out));
    debug_assert!(
        verify_canonical_shallow(arena, result).is_empty(),
        "canon_add: non-canonical sum with an infinite term: {:?}",
        verify_canonical_shallow(arena, result)
    );
    result
}

/// Is the direction of the directed infinity `key` (its factors other than
/// `∞`) known to be real?  Then it is `±∞` (or `nan`) at every point.
fn has_real_direction(arena: &Arena, key: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    let ExprNode::Mul(children) = arena.node(key) else {
        return false;
    };
    let mut cache = AssumptionCache::new();
    children.iter().all(|&c| {
        // (identity: the factor `∞` itself, beside the direction)
        c == arena.infinity
            || arena.as_num(c).is_some()
            || cache.query(arena, c, Props::REAL) == Some(true)
    })
}

/// Is `id` a sum with exactly one infinite term — `±∞` or a directed
/// infinity `x·∞` — beside finite ones (`∞ + i`, `1 + i·∞`)?  Its magnitude
/// is infinite at every point where it is defined, so its reciprocal is 0
/// ([`canon_pow`]) and like terms of it cannot cancel to 0
/// ([`cancels_infinite_terms`]).
pub(crate) fn is_infinite_sum(arena: &Arena, id: ExprId) -> bool {
    let ExprNode::Add(children) = arena.node(id) else {
        return false;
    };
    let mut infinite = 0usize;
    for &c in children.iter() {
        match arena.node(c) {
            ExprNode::Infinity | ExprNode::NegInfinity => infinite += 1,
            ExprNode::ComplexInfinity | ExprNode::NaN => return false,
            _ if is_directed_infinity(arena, c) => infinite += 1,
            _ => {}
        }
    }
    infinite == 1
}

/// Did like terms of a sum whose key holds an infinite sum factor
/// ([`is_infinite_sum`], or a positive power of one) cancel?  `y·(∞ + i) −
/// y·(∞ + i)` is `∞ − ∞ = nan`, not 0.  Only the `merged` keys (those that
/// collected two or more terms) whose coefficient became 0 are looked at.
fn cancels_infinite_terms(arena: &Arena, terms: &FxHashMap<ExprId, Q>, merged: &[ExprId]) -> bool {
    merged.iter().any(|key| {
        terms.get(key).is_some_and(Zero::is_zero) && {
            let factors: &[ExprId] = match arena.node(*key) {
                ExprNode::Mul(children) => children,
                _ => std::slice::from_ref(key),
            };
            factors.iter().any(|&f| match *arena.node(f) {
                ExprNode::Pow(b, e) => {
                    arena.as_num(e).is_some_and(Signed::is_positive) && is_infinite_sum(arena, b)
                }
                _ => is_infinite_sum(arena, f),
            })
        }
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Mul
// ═══════════════════════════════════════════════════════════════════════════

/// Result of `0 × (all other factors)` inside [`canon_mul`].
///
/// The rule is independent of argument order:
///
/// * `0 × (±∞ | zoo | nan) → nan`, also when the special value is hidden
///   inside a nested `Mul`, `Neg`, or (non-canonical) `Add` among the
///   factors that have not been processed yet, or among the bases already
///   collected;
/// * otherwise `0 × anything → 0`.  Applications at exact arguments where
///   they are infinite (`ln(0)`, `exp(∞)`, `Γ(−1)`, `besseli(1, −∞)`) were
///   folded to `zoo`/`±∞` when interned ([`canon_function`]); a library
///   function left unevaluated at an exact infinity (`±∞`, `±i·∞`) where
///   it has no finite value the constructors know makes the product `nan`
///   (also as a denominator: [`library_value_unknown_at_infinity`]); a
///   function left unevaluated at `zoo` makes the product `nan`
///   ([`unevaluated_at_complex_infinity`]); any other application (`f(x)`,
///   a user function `f(zoo)`) is treated as finite here;
/// * `0 × d⁻ᵏ → nan` for a denominator `d` (among the factors, or in a
///   product or sum among them) that vanishes identically
///   ([`factor_vanishes`]: also a constant that is 0, `sin² 1 + cos² 1 −
///   1`) or is undefined at every point ([`everywhere`]): `0/((x + y)² −
///   x² − 2xy − y²)` is `0/0`.  Up to 0.32 it was `0` (SymPy 1.14 too), and
///   `subs` of `x = 11/7` into `(x·(y + 3) − xy − 3x)·(c/((x + y)² − x² −
///   2xy − y²) + 2)/x`, whose first factor becomes `0` while the
///   denominator stays a sum in `y`, gave `0` instead of `nan`.  Only a
///   denominator met on this path gets the residue test;
/// * `0 × f → nan` for a factor `f` that is infinite or undefined at every
///   point: a positive power of a sum with a term over such a denominator,
///   `(1/d + 2)²`, or a function at such an argument, `sin(c/d)` with `c/d
///   = zoo` ([`everywhere`]).  Up to 0.34 `0·sin((√2 + e)/(sin² 1 + cos² 1 −
///   1))` was `0` (`0·nan`; SymPy 1.14 gives `0`, deliberately different);
/// * `0 × δ(c) → nan`, `0 × H(c) → nan` (also over them) for a constant `c`
///   proved non-real, where they have no value
///   ([`distribution_without_value`]);
/// * a denominator `0^e` whose exponent's sign the constructors cannot
///   decide (`0^y`; also a factor `0^(−y)`, its reciprocal) is not a
///   generic nonzero value: it is 0 on the half plane `re e > 0`.  The
///   quotient is `0` only where every such denominator is nonzero, and `nan`
///   elsewhere ([`zero_power_is_nonzero`]): `0/0^y` is `Piecewise((0, (0 >
///   `re(y)) | Eq(y, 0)), (nan, True))`.  Up to 0.57 it was 0, right only for
///   `re y < 0` (and `y = 0`).  (Any other factor `0^e` keeps the canonical
///   `0·f = 0` of a finite-looking factor: the exact zero coefficients of an
///   expansion at `0^e` rely on it.)
/// * any other denominator that may vanish on a region of its symbols
///   ([`may_vanish_on_a_region`]: a step `H(y)`, `⌊y⌋`, `max(y, 0)`, a delta,
///   a sum with a power `0^y` in it) is not a generic nonzero value either:
///   the quotient is `0` where each such factor of it is nonzero
///   ([`push_nonzero_conditions`]) and `nan` elsewhere, `0/H(y)` is
///   `Piecewise((0, y >= 0), (nan, True))`.  Up to 0.58 it was 0: wrong for
///   `y < 0`, and for almost every `y` over `KroneckerDelta(y, 1)` or `δ(y)`.
/// * a `Piecewise` factor or denominator distributes the zero over its
///   pieces ([`times_piecewise`]).
///
/// `saw_infinity` reports whether an infinity was already consumed from the
/// factor list before the coefficient became zero; `remaining` are the
/// other factors (bases with a positive exponent among them), `denominators`
/// the bases collected with a negative exponent, `zero_exponents` the
/// exponents of the factors `0^e` collected (under the base 0).
fn zero_times_rest<'a>(
    arena: &mut Arena,
    saw_infinity: bool,
    remaining: impl IntoIterator<Item = &'a ExprId>,
    denominators: impl IntoIterator<Item = ExprId>,
    zero_exponents: impl IntoIterator<Item = ExprId>,
) -> ExprId {
    if saw_infinity {
        return arena.nan;
    }
    let mut denominators: SmallVec<[ExprId; 4]> = denominators.into_iter().collect();
    let mut applications: SmallVec<[ExprId; 4]> = SmallVec::new();
    // The exponents `e` of the denominators `0^e` met, also as a factor
    // `0^(−e)` ([`reciprocal_zero_power`]).
    let mut zero_powers: SmallVec<[ExprId; 2]> = SmallVec::new();
    for e in zero_exponents {
        if let Some(d) = reciprocal_zero_power(arena, e) {
            zero_powers.push(d);
        }
    }
    // The `Piecewise` factors (`true`) and denominators (`false`) met.
    let mut piecewise: SmallVec<[(ExprId, bool); 2]> = SmallVec::new();
    // (node, outside every negative power?): only there is a negative
    // power a denominator of the product (`(1/d + 2)⁻¹` is finite).
    let mut stack: SmallVec<[(ExprId, bool); 16]> =
        remaining.into_iter().map(|&id| (id, true)).collect();
    stack.extend(denominators.iter().map(|&d| (d, false)));
    while let Some((id, outside)) = stack.pop() {
        match arena.node(id) {
            ExprNode::NaN
            | ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity => {
                tracing::debug!("canon_mul: 0 * ∞ → NaN");
                return arena.nan;
            }
            ExprNode::Mul(children) | ExprNode::Add(children) => {
                stack.extend(children.iter().map(|&c| (c, outside)));
            }
            ExprNode::Neg(inner) => stack.push((*inner, outside)),
            ExprNode::Pow(base, e)
                if outside && arena.as_num(*e).is_some_and(Signed::is_negative) =>
            {
                denominators.push(*base);
            }
            ExprNode::Pow(base, e) if outside && arena.as_num(*e).is_some() => {
                stack.push((*base, outside));
            }
            ExprNode::Pow(base, e) if outside && arena.is_zero_structural(*base) => {
                let e = *e;
                if let Some(d) = reciprocal_zero_power(arena, e) {
                    zero_powers.push(d);
                }
            }
            ExprNode::Piecewise(_) if outside => piecewise.push((id, true)),
            ExprNode::DiracDelta(_) | ExprNode::Heaviside(_)
                if outside && distribution_without_value(arena, id) =>
            {
                tracing::debug!("canon_mul: 0 times δ or H at a non-real constant → nan");
                return arena.nan;
            }
            node if outside && application_arguments(arena, node).is_some() => {
                applications.push(id);
            }
            // `0·s = nan` for a symbol declared infinite (SymPy:
            // `Symbol('s', infinite=True)*0` → nan).
            ExprNode::Symbol(_) if outside && symbol_is_infinite(arena, id) => {
                return arena.nan;
            }
            _ => {}
        }
    }
    let mut memo = EverywhereMemo::default();
    // The conditions under which the denominators that may vanish on a
    // region are nonzero.
    let mut conditions: SmallVec<[ExprId; 2]> = SmallVec::new();
    for d in denominators {
        if library_value_unknown_at_infinity(arena, d)
            || distribution_without_value(arena, d)
            || without_limit_at_complex_infinity(arena, d)
            || memo.zero_or_undefined(arena, d)
            || vanishes_by_function_identity(arena, d)
        {
            tracing::debug!("canon_mul: 0 over an identically zero or undefined denominator → nan");
            return arena.nan;
        }
        if matches!(arena.node(d), ExprNode::Piecewise(_)) {
            piecewise.push((d, false));
        } else {
            push_nonzero_conditions(arena, d, &mut conditions);
        }
    }
    for f in applications {
        if memo.infinite_or_undefined(arena, f)
            || not_finite_at_infinity(arena, f)
            || unevaluated_at_complex_infinity(arena, f)
        {
            tracing::debug!("canon_mul: 0 times a function undefined at every point → nan");
            return arena.nan;
        }
    }
    for e in zero_powers {
        let condition = zero_power_is_nonzero(arena, e);
        conditions.push(condition);
    }
    let zero = where_defined(arena, arena.zero, conditions);
    times_piecewise(arena, arena.zero, zero, &piecewise)
}

/// `z` (`absorbing`, or the `Piecewise` of [`where_defined`]) times the
/// `Piecewise` factors (`true`) and denominators (`false`) of a product
/// with the coefficient `absorbing` = 0: `0·Piecewise((vᵢ, cᵢ))` is
/// `Piecewise((0·vᵢ, cᵢ))` (one value when they agree), and so is
/// `0/Piecewise(…)` with `0/vᵢ`.  A `Piecewise` is not a generic finite value: `0·(0/0^y)` is
/// `0/0^y`'s `Piecewise((0, …), (nan, True))` itself, as when the product is
/// built at once, and `0/(0/0^y)` is `nan` (`0/0`, `0/nan`).  The same for
/// the absorbing factor `zoo` in place of 0 ([`handle_mul_with_zoo`]):
/// `zoo·Piecewise((1, Eq(y, 0)), (nan, True))` is `zoo` only at `y = 0`.
fn times_piecewise(
    arena: &mut Arena,
    absorbing: ExprId,
    z: ExprId,
    piecewise: &[(ExprId, bool)],
) -> ExprId {
    let mut result = z;
    for &(p, factor) in piecewise {
        let ExprNode::Piecewise(pairs) = arena.node(p).clone() else {
            continue;
        };
        let mut values: SmallVec<[(ExprId, ExprId); 3]> = SmallVec::new();
        for (v, c) in pairs {
            let v = if factor {
                v
            } else {
                canon_pow(arena, v, arena.neg_one)
            };
            values.push((canon_mul(arena, &[absorbing, v]), c));
        }
        let Some(&(first, _)) = values.first() else {
            continue;
        };
        let q = if values.iter().all(|&(v, _)| v == first) {
            first
        } else {
            arena.piecewise(&values)
        };
        if q == absorbing {
            continue;
        }
        result = if result == absorbing {
            q
        } else {
            canon_mul(arena, &[result, q])
        };
    }
    result
}

/// How many of the terms of the exponent `e` (a sum, or one term) have a
/// negative numeric coefficient, and how many terms there are: `−y − 1` is
/// (2, 2), `z − y` (1, 2), `2y` (0, 1).
fn negative_terms(arena: &Arena, e: ExprId) -> (usize, usize) {
    let negative = |t: ExprId| match arena.node(t) {
        ExprNode::Neg(_) => true,
        ExprNode::Num(n) => arena.num(*n).is_negative(),
        ExprNode::Mul(children) => children
            .first()
            .and_then(|&c| arena.as_num(c))
            .is_some_and(Signed::is_negative),
        _ => false,
    };
    match arena.node(e) {
        ExprNode::Add(terms) => (terms.iter().filter(|&&t| negative(t)).count(), terms.len()),
        _ => (usize::from(negative(e)), 1),
    }
}

/// `−e` when a term of the exponent `e` of a factor `0^e` has a negative
/// numeric coefficient: `0^(−y)` is the reciprocal of `0^y` (`zoo` for a
/// positive `re y`, 0 for a negative one, 1 at `y = 0`), and `0·0^(−y)` is
/// `0/0^y` (the form [`canon_pow`] gives `1/0^y`; so are `1/0^(y + 1)`, which
/// is `0^(−y − 1)`, and `1/0^(y − z)`, which is `0^(z − y)`).  Any other
/// factor `0^e` keeps the generic reading of a factor (`0·0^y = 0`; the exact
/// zero coefficients of an expansion at `0^y` rely on it).
fn reciprocal_zero_power(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    (negative_terms(arena, e).0 > 0).then(|| canon_neg(arena, e))
}

/// The condition under which the power `0^e` (an `e` whose sign the
/// constructors cannot decide: [`canon_pow`] folds the others) is nonzero
/// and has a value.  `0^e` is 0 for `re e > 0`, `zoo` for `re e < 0`, 1 at `e
/// = 0` and `nan` elsewhere on `re e = 0`: nonzero for `re e < 0` or `e = 0`,
/// written `(0 > re(e)) | Eq(e, 0)`; for an exponent with a negative
/// coefficient, `e = −d`, in terms of `d`: `(re(d) > 0) | Eq(d, 0)`.
fn zero_power_is_nonzero(arena: &mut Arena, e: ExprId) -> ExprId {
    let zero = arena.zero;
    let (negative, terms) = negative_terms(arena, e);
    let mut either = if negative == terms {
        let d = canon_neg(arena, e);
        let re = arena.re(d);
        [arena.gt(re, zero), arena.eq_(d, zero)]
    } else {
        let re = arena.re(e);
        [arena.gt(zero, re), arena.eq_(e, zero)]
    };
    either.sort_by(|a, b| arena.sort_key(*a).cmp(arena.sort_key(*b)));
    arena.or(&either)
}

/// The conditions under which the factors of `d` that may vanish on a
/// region of their symbols ([`may_vanish_on_a_region`]) are nonzero and
/// have a value, pushed onto `conditions`; nothing for a `d` that may not.
/// The factors are those of a product of positive numeric powers, looked
/// at through `−f`, `|f|`, `sign f`, `f̄` (zero exactly where `f` is):
///
/// * `0^e`: [`zero_power_is_nonzero`];
/// * `KroneckerDelta(a, b)`: `Eq(a, b)`; `δ(a)`: `Eq(a, 0)` (the generic
///   value of both is 0); `H(a)`: `a >= 0` (`H(0) = 1/2`);
/// * any other (`⌊y⌋`, `max(y, 0)`, `arg y`, a sum or a function with a
///   power `0^y` or a delta in it), unless the assumptions know it nonzero:
///   `|f| > 0`.  Not `f != 0`: `eval` decides
///   `Ne(nan, 0)` true, and at a point where `f` has no value (`max(i, 0)`,
///   `0^i − 1`) the quotient has none either; `|nan| > 0` is undecided there.
///
/// The generic convention does not cover such factors (they are never
/// divided by: CONTRIBUTING, "Piecewise parameters").
fn push_nonzero_conditions(arena: &mut Arena, d: ExprId, conditions: &mut SmallVec<[ExprId; 2]>) {
    if !may_vanish_on_a_region(arena, d) {
        return;
    }
    let mut steps: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut stack: SmallVec<[ExprId; 8]> = smallvec![d];
    while let Some(f) = stack.pop() {
        match *arena.node(f) {
            ExprNode::Mul(ref children) => stack.extend(children.iter().copied()),
            ExprNode::Neg(g) | ExprNode::Abs(g) | ExprNode::Sign(g) | ExprNode::Conjugate(g) => {
                stack.push(g);
            }
            ExprNode::Pow(base, n)
                if !arena.is_zero_structural(base) && arena.as_num(n).is_some() =>
            {
                if arena.as_num(n).is_some_and(Signed::is_positive) {
                    stack.push(base);
                } else if holds_zero_power(arena, base) {
                    // A reciprocal is 0 where its base is infinite, which
                    // only a base with a power `0^e` in it is on a region
                    // (`1/(0^y − 1)` for `re y < 0`); the reciprocal of a
                    // step that vanishes is infinite, not 0.
                    steps.push(f);
                }
            }
            _ if may_vanish_on_a_region(arena, f) => steps.push(f),
            _ => {}
        }
    }
    let zero = arena.zero;
    for f in steps {
        let condition = match *arena.node(f) {
            ExprNode::Pow(base, e) if arena.is_zero_structural(base) => {
                zero_power_is_nonzero(arena, e)
            }
            // (the symbolic argument first: `y == 1`)
            ExprNode::KroneckerDelta(a, b) if arena.as_num(a).is_some() => arena.eq_(b, a),
            ExprNode::KroneckerDelta(a, b) => arena.eq_(a, b),
            ExprNode::DiracDelta(a) => arena.eq_(a, zero),
            ExprNode::Heaviside(a) => arena.ge(a, zero),
            _ => {
                // A sum the assumptions know nonzero (`KroneckerDelta(y, 1)
                // + 1`) needs no condition.
                use crate::base::assumptions::{AssumptionCache, Props};
                use crate::base::truth::Truth;
                let nonzero = AssumptionCache::new().query(arena, f, Props::NONZERO);
                if Truth::from(nonzero).is_true() {
                    continue;
                }
                let modulus = arena.abs(f);
                arena.gt(modulus, zero)
            }
        };
        conditions.push(condition);
    }
}

/// Does `e` hold a power `0^x` (of an exponent whose sign the constructors
/// could not decide)?  Such a power is infinite on a half plane of its
/// exponent, so an expression with one may be infinite on a region.
fn holds_zero_power(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::contains(arena, e, arena.zero)
        && crate::base::walk::post_order_ids(arena, e)
            .into_iter()
            .any(|id| matches!(*arena.node(id), ExprNode::Pow(base, _) if base == arena.zero))
}

/// `value` (0 or `zoo`) where every one of the `conditions` holds, `nan`
/// elsewhere: `Piecewise((value, c₁ ∧ … ∧ cₙ), (nan, True))`; `value` itself
/// without conditions.  A value right on a part of the parameter space only
/// is not folded to.
fn where_defined(
    arena: &mut Arena,
    value: ExprId,
    mut conditions: SmallVec<[ExprId; 2]>,
) -> ExprId {
    if conditions.is_empty() {
        return value;
    }
    tracing::debug!("canon_mul: 0 or zoo times a factor 0 on a region → Piecewise");
    conditions.sort_by(|a, b| arena.sort_key(*a).cmp(arena.sort_key(*b)));
    conditions.dedup();
    let condition = arena.and(&conditions);
    let (nan, otherwise) = (arena.nan, arena.bool_true);
    arena.piecewise(&[(value, condition), (nan, otherwise)])
}

/// May `e`, a quantity with symbols, be 0 on a set of values of its
/// symbols of positive measure without being 0 identically?  The generic
/// convention — a quantity with symbols that is not identically zero is
/// nonzero for almost every value of them, so `x/x = 1` and a series may
/// divide by a parameter `a` — holds for the analytic functions of the
/// symbols and fails for the piecewise ones: `0^y` is `0` on the half plane
/// `re y > 0` (and `zoo`, `nan` elsewhere), `δ(1/2, y)` is `0` except at
/// `y = 1/2` (its generic value is 0), `H(y)` is `0` for `y < 0`, `⌊y⌋` on
/// `[0, 1)`, `max(y, 0)` for `y ≤ 0`, `arg(y)` for a real `y > 0`.  Nonzero at
/// a sample point proves nothing there.
///
/// A factor of a product may vanish on a region when it is such a step
/// (`0^y`, `δ`, `H`, `⌊·⌋`, `⌈·⌉`), a `min`, `max` or `Piecewise` with a
/// value 0 or one that may (`max(y, 1)` may not), `arg f` of a real `f`, or
/// `|f|`, `sign f`, `re f`, `im f`, `f̄`, `arg f` or a power `fⁿ` of an `f` that
/// may; any other factor — a sum or a function — when `0^y`, `δ` or a
/// Kronecker delta (0 on a half plane or almost everywhere) occurs in it.
/// A product of symbols, numbers and analytic functions of them may not.
/// (A sum whose step functions cancel a constant on a region, `H(y) − 1`,
/// `⌊y⌋ − 1`, or whose parts do for a real argument, `|y| − y`, is not
/// detected: it keeps the generic reading.)  Up to 0.56 the series engine
/// and the limits took `0^y` and `δ(1/2, y)` for nonzero parameters
/// (`series(sin(x)/(x + 0^y), x)` was `x/0^y + …`, wrong for `y > 0`), and
/// `cancel` divided them out; up to 0.58 the constructors folded `0/d` and
/// `zoo·d` over such a `d` ([`zero_times_rest`], [`handle_mul_with_zoo`]).
/// (In `simplify::identically_zero` up to 0.58; the zero tests name it
/// there.)
pub(crate) fn may_vanish_on_a_region(arena: &Arena, e: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    // Zero on a half plane or almost everywhere: also inside sums.
    let generically_zero = |arena: &Arena, id: ExprId| match *arena.node(id) {
        ExprNode::Pow(b, x) => arena.is_zero_structural(b) && arena.as_num(x).is_none(),
        ExprNode::KroneckerDelta(..) | ExprNode::DiracDelta(_) => !arena.is_symbol_free(id),
        _ => false,
    };
    let step = |arena: &Arena, id: ExprId| {
        generically_zero(arena, id)
            || (matches!(
                arena.node(id),
                ExprNode::Heaviside(_) | ExprNode::Floor(_) | ExprNode::Ceiling(_)
            ) && !arena.is_symbol_free(id))
    };
    let mut stack = vec![e];
    while let Some(f) = stack.pop() {
        if arena.is_symbol_free(f) {
            continue;
        }
        match *arena.node(f) {
            ExprNode::Mul(ref fs) => stack.extend(fs.iter().copied()),
            ExprNode::Neg(g)
            | ExprNode::Abs(g)
            | ExprNode::Sign(g)
            | ExprNode::Re(g)
            | ExprNode::Im(g)
            | ExprNode::Conjugate(g) => stack.push(g),
            ExprNode::Arg(g) => {
                if Truth::from(AssumptionCache::new().query(arena, g, Props::REAL)).is_true() {
                    return true;
                }
                stack.push(g);
            }
            ExprNode::Pow(b, n) if arena.as_num(n).is_some() => stack.push(b),
            ExprNode::Min(ref args) | ExprNode::Max(ref args) => {
                if args.iter().any(|&a| arena.is_zero_structural(a)) {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            ExprNode::Piecewise(ref pairs) => {
                if pairs.iter().any(|&(v, _)| arena.is_zero_structural(v)) {
                    return true;
                }
                stack.extend(pairs.iter().map(|&(v, _)| v));
            }
            ExprNode::Symbol(_) => {}
            _ if step(arena, f) => return true,
            _ => {
                if crate::base::walk::post_order_ids(arena, f)
                    .into_iter()
                    .any(|id| generically_zero(arena, id))
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Does a sum of the denominator `d` ([`zero_candidate_sums`]) vanish by an
/// identity of its functions (`sin²x + cos²x − 1`), confirmed by
/// [`Arena::vanishes_by_identity`] (certified samples, then an exact
/// reduction)?  Asked only on the `0·d⁻¹` path of [`zero_times_rest`] and
/// only for a sum with a function in it that the residue test cannot tell
/// from 0 ([`Arena::may_vanish_identically`]) — the multiplied-out
/// confirmation of [`sum_vanishes_identically`] does not relate `sin` and
/// `cos`.  Since 0.42 a numerator that is 0 for the assumptions folds when
/// it is built (`⌈n⌉ − n` for an integer `n`), before `simplify` could see
/// the `0/0`: `(⌈n⌉ − n)/(sin²x + cos²x − 1)` would have been 0 (SymPy
/// 1.14: 0); it is `nan`.  Not nested in another such confirmation.
fn vanishes_by_function_identity(arena: &mut Arena, d: ExprId) -> bool {
    let d = with_pinned_values(arena, d).unwrap_or(d);
    let candidates: SmallVec<[ExprId; 4]> = zero_candidate_sums(arena, d)
        .into_iter()
        .filter(|&s| {
            matches!(arena.node(s), ExprNode::Add(_))
                && may_vanish_by_identity(arena, s)
                && arena.may_vanish_identically(s)
                && !arena.may_have_vanishing_denominator(s)
        })
        .collect();
    if candidates.is_empty() {
        return false;
    }
    let Some(_guard) = VanishingCheckGuard::enter() else {
        return false;
    };
    candidates
        .into_iter()
        .any(|s| arena.vanishes_by_identity(s))
}

/// Is `f` a `δ(c)` or `H(c)` at a constant `c` proved non-real (by the
/// assumptions, or a certified nonzero imaginary part)?  The distributions
/// have values only at real arguments (SymPy 1.14: `Heaviside(I)` raises
/// "defined only for real values"), so `0·δ(i)` is not 0 (up to 0.52 it
/// was, and `x·δ(i)` vanished at `x = 0`).  At an argument with symbols (a
/// generic real point) or of undecided realness the product keeps folding
/// to 0.  An infinite argument off the real axis (`±i·∞`, `1 + i·∞`) is
/// not real either: up to 0.53 `0·H(i·∞)` was 0.
fn distribution_without_value(arena: &Arena, f: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    let (ExprNode::DiracDelta(a) | ExprNode::Heaviside(a)) = *arena.node(f) else {
        return false;
    };
    if infinite_direction(arena, a).is_some_and(|(_, im)| !im.is_zero()) {
        return true;
    }
    is_numeric_constant(arena, a)
        && (Truth::from(AssumptionCache::new().query(arena, a, Props::REAL)).is_false()
            || arena.part_signs(a).is_some_and(|s| s.im.is_some()))
}

fn symbol_is_infinite(arena: &Arena, id: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    AssumptionCache::new().query(arena, id, Props::INFINITE) == Some(true)
}

/// The complex rational `a + b·i` that `id` is: a number, `i`, `q·i`, or a
/// sum of them (the direction of a directed infinity, `(1 + i)·∞`).
fn gaussian_value(arena: &Arena, id: ExprId) -> Option<(Q, Q)> {
    let term = |t: ExprId| -> Option<(Q, Q)> {
        if let Some(q) = arena.as_num(t) {
            return Some((q.clone(), Q::zero()));
        }
        if t == arena.i_unit {
            return Some((Q::zero(), Q::one()));
        }
        match arena.node(t) {
            ExprNode::Mul(cs) => match **cs {
                [c, u] if u == arena.i_unit => Some((Q::zero(), arena.as_num(c)?.clone())),
                _ => None,
            },
            _ => None,
        }
    };
    match arena.node(id) {
        ExprNode::Add(terms) => {
            let (mut re, mut im) = (Q::zero(), Q::zero());
            for &t in terms.iter() {
                let (a, b) = term(t)?;
                re = q_add(&re, &a);
                im = q_add(&im, &b);
            }
            Some((re, im))
        }
        _ => term(id),
    }
}

/// The direction `(a, b)` of an argument that is infinite in the direction
/// `a + b·i` with rational `a`, `b`: `±∞`, a directed infinity `c·∞` of
/// a Gaussian rational `c`, or a sum of one of them and finite terms (which
/// do not change the growth of the functions [`not_finite_at_infinity`]
/// looks at).  `None` for every other argument, among them a direction of
/// unknown value (`x·∞`).
fn infinite_direction(arena: &Arena, arg: ExprId) -> Option<(Q, Q)> {
    let direction = |t: ExprId| -> Option<(Q, Q)> {
        match arena.node(t) {
            ExprNode::Infinity => Some((Q::one(), Q::zero())),
            ExprNode::NegInfinity => Some((-Q::one(), Q::zero())),
            ExprNode::Mul(cs) if is_directed_infinity(arena, t) => {
                let (mut re, mut im) = (Q::one(), Q::zero());
                // (identity: the factors of the direction, beside `∞`)
                for &c in cs.iter().filter(|&&c| c != arena.infinity) {
                    let (a, b) = gaussian_value(arena, c)?;
                    (re, im) = (
                        q_add(&q_mul(&re, &a), &-q_mul(&im, &b)),
                        q_add(&q_mul(&re, &b), &q_mul(&im, &a)),
                    );
                }
                Some((re, im))
            }
            _ => None,
        }
    };
    match arena.node(arg) {
        ExprNode::Add(terms) => {
            let mut found = None;
            for &t in terms.iter() {
                if let Some(d) = direction(t) {
                    if found.is_some() {
                        return None;
                    }
                    found = Some(d);
                } else if value_class::holds_real_infinity(arena, t) {
                    return None;
                }
            }
            found
        }
        _ => direction(arg),
    }
}

/// Is the elementary function application `f`, at an argument with an
/// infinite term, not known finite?  Then `0·f` is `nan` (or is `0·∞`)
/// rather than 0.  Up to 0.41 every application left unevaluated counted
/// as finite: `Max(0, 2/∞)·cosh((1 + i)·∞ − 3/2)` was 0, where the cosine
/// grows without bound (SymPy 1.14: `cosh(3/2 - oo*(1 + I)).is_finite` is
/// False and the product `nan`).  By the direction `a + b·i` of the
/// infinite term: `e^z`, `cosh`, `sinh` grow for `a ≠ 0` (`e^z` tends to 0
/// for `a < 0`) and stay bounded on the imaginary axis; `sin`, `cos` grow
/// for `b ≠ 0` and are bounded on the real axis; `tan` (`tanh`) has a
/// finite limit off the real (imaginary) axis and poles along it; `|z|`,
/// `ln`, the inverse sine and cosine functions and `⌊·⌋`, `⌈·⌉` grow;
/// `atan`, `atanh` and `sign` stay bounded; `erf`/`erfc` tend to constants
/// for `|a| ≥ |b|` and grow otherwise; `Γ(z)` and `z!` grow for `a > 0`,
/// tend to 0 for `a = 0` and for `a < 0`, `b ≠ 0`, and meet the poles for
/// `a < 0`, `b = 0`; `ln Γ` and `ψ` grow in every direction (`ψ(−∞)` meets
/// the poles; up to 0.53 `0·ψ(−∞)` was 0); `ζ` tends to 1 for `a > 0` and
/// grows without a limit otherwise (`ζ(−∞)`, `ζ(i·∞)`).  A direction of
/// unknown value (`x·∞`) is not known finite.  A library function whose values at `±∞` fold, left unevaluated
/// at `±∞`, is not known finite either
/// ([`library_value_unknown_at_infinity`]).  Any other function (`ln Γ`,
/// `atan`, user functions) keeps counting as finite unless the assumptions
/// say otherwise.
fn not_finite_at_infinity(arena: &Arena, f: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    let Some(a) = arena.node(f).unary_arg_if(examined_at_infinity) else {
        if library_value_unknown_at_infinity(arena, f) {
            return true;
        }
        let infinite_argument = application_arguments(arena, arena.node(f)).is_some_and(|args| {
            args.iter()
                .any(|&a| value_class::holds_real_infinity(arena, a))
        });
        return infinite_argument
            && AssumptionCache::new().query(arena, f, Props::FINITE) == Some(false);
    };
    if !value_class::holds_real_infinity(arena, a) {
        return false;
    }
    if AssumptionCache::new().query(arena, f, Props::FINITE) == Some(true) {
        return false;
    }
    let Some((re, im)) = infinite_direction(arena, a) else {
        return true;
    };
    let bounded = match *arena.node(f) {
        ExprNode::Exp(_) => !re.is_positive(),
        ExprNode::Cosh(_) | ExprNode::Sinh(_) => re.is_zero(),
        ExprNode::Sin(_) | ExprNode::Cos(_) => im.is_zero(),
        ExprNode::Tan(_) => !im.is_zero(),
        ExprNode::Tanh(_) => !re.is_zero(),
        ExprNode::Erf(_) | ExprNode::Erfc(_) => re.abs() >= im.abs(),
        // Stirling: |Γ(z)| grows for Re z → +∞ and tends to 0 along the
        // imaginary axis and off the real axis to the left (by the
        // reflection formula); along the negative real axis it meets the
        // poles.
        ExprNode::Gamma(_) | ExprNode::Factorial(_) => {
            re.is_zero() || (re.is_negative() && !im.is_zero())
        }
        ExprNode::Zeta(_) => re.is_positive(),
        _ => false,
    };
    !bounded
}

/// Is `f` an elementary, special or library function application (not a
/// user function, not a power) left unevaluated at an argument `zoo`, and
/// not known finite?  `zoo` has no direction, so no table value applies:
/// what the constructors fold there is folded (`ln(zoo) = zoo`, `sin(zoo) =
/// nan`, `Li_s(0) = 0` for every `s`), and what stays is `zoo` or has no
/// limit (`Γ(zoo)`, `atan(zoo)`, `uppergamma(zoo, ∞)`).  Either way `0·f`
/// is `nan` (up to 0.56 it was 0; SymPy 1.14 also gives 0, against its
/// `0·zoo = nan`).  A quotient `0/f` is 0 where `f` tends to `zoo`
/// (`W(zoo)`) and has no value where `f` has no limit
/// ([`without_limit_at_complex_infinity`]).  An orthogonal polynomial of
/// degree 0 is a constant at every argument (`eval` folds `legendre(−1,
/// zoo)` and `jacobi(2, −3, 0, zoo)` to 1: [`AtComplexInfinity::Constant`]),
/// and of degree 0 in its variable at every parameter (`legendre(0, zoo)`,
/// `jacobi(0, zoo, 1, x)`): not counted.
fn unevaluated_at_complex_infinity(arena: &Arena, f: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::libfn::LibFn;
    use crate::base::truth::Truth;
    let node = arena.node(f);
    let constant_polynomial = match node {
        ExprNode::Lib(
            LibFn::Legendre
            | LibFn::ChebyshevT
            | LibFn::ChebyshevU
            | LibFn::Hermite
            | LibFn::Laguerre
            | LibFn::Gegenbauer
            | LibFn::Jacobi
            | LibFn::AssocLaguerre,
            args,
        ) if args.first() == Some(&arena.zero) => true,
        ExprNode::Lib(lib, args) => matches!(
            at_complex_infinity(arena, *lib, args),
            Some(AtComplexInfinity::Constant)
        ),
        _ => false,
    };
    !constant_polynomial
        && !matches!(node, ExprNode::Apply(..) | ExprNode::Pow(..))
        && application_arguments(arena, node).is_some_and(|args| {
            args.iter()
                .any(|&a| ValueClass::special_atom(arena, a) == Some(ValueClass::ComplexInf))
        })
        && !Truth::from(AssumptionCache::new().query(arena, f, Props::FINITE)).is_true()
}

/// Is the denominator `d` a function left unevaluated at `zoo`
/// ([`unevaluated_at_complex_infinity`]) that does not tend to `zoo` there
/// ([`tends_to_complex_infinity`])?  Then `0/d` has no value: `Γ` grows
/// along the positive axis and tends to 0 along the imaginary one, `atan`,
/// `erf` tend to different constants in different directions,
/// `uppergamma(zoo, ∞)` has no order: `0/Γ(zoo)`, `0/atan(zoo)` are `nan`
/// (up to 0.57 they were 0, as in SymPy 1.14).  Where `|d| → ∞` in every
/// direction `0/d` is `0/zoo = 0`.
fn without_limit_at_complex_infinity(arena: &Arena, d: ExprId) -> bool {
    unevaluated_at_complex_infinity(arena, d) && !tends_to_complex_infinity(arena, d)
}

/// Does `f` tend to `zoo` (`|f| → ∞` in every direction) as its argument
/// `zoo` is approached: `W_k(z) ∼ ln z` (any branch `k` known finite),
/// `asin`, `acos`, `asinh`, `acosh` (`∼ ±ln 2z` up to `±i`), and
/// `assoc_laguerre` of a positive integer degree with a symbolic finite `α`
/// (leading coefficient `(−1)ⁿ/n!`; with numeric parameters it, and the
/// other orthogonal polynomials, fold to `zoo` when built:
/// [`at_complex_infinity`])?  Only the variable slot may be `zoo`.
fn tends_to_complex_infinity(arena: &Arena, f: ExprId) -> bool {
    use crate::base::libfn::LibFn;
    let zoo = |a: ExprId| ValueClass::special_atom(arena, a) == Some(ValueClass::ComplexInf);
    let finite = |a: ExprId| ValueClass::of(arena, a).is_finite_proved();
    let positive_degree = |n: ExprId| {
        arena
            .as_num(n)
            .is_some_and(|q| q.is_integer() && q.is_positive())
    };
    match arena.node(f) {
        ExprNode::LambertW(z)
        | ExprNode::Asin(z)
        | ExprNode::Acos(z)
        | ExprNode::Asinh(z)
        | ExprNode::Acosh(z) => zoo(*z),
        ExprNode::Lib(LibFn::LambertW, args) => match **args {
            [z] => zoo(z),
            [z, k] => zoo(z) && finite(k),
            _ => false,
        },
        ExprNode::Lib(LibFn::AssocLaguerre, args) => {
            matches!(**args, [n, a, z] if positive_degree(n) && finite(a) && zoo(z))
        }
        _ => false,
    }
}

/// What a library function is at `zoo` ([`at_complex_infinity`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AtComplexInfinity {
    /// `zoo`: `abs(f) → ∞` in every direction.
    ComplexInfinity,
    /// The integer it tends to in every direction (`Li₀(zoo) = −1`,
    /// `Li_{−k}(zoo) = 0`, the zero polynomial 0).
    Integer(i64),
    /// A polynomial of degree 0 in its variable: its value anywhere, which
    /// `eval` takes at 0.  The node stays when it is built, and counts as
    /// finite ([`unevaluated_at_complex_infinity`]).
    Constant,
}

/// The value of the library function `f` at `args` when an argument is
/// `zoo`, the point at infinity without a direction, where the function
/// tends to one value in every direction; `None` where the table has no
/// entry (the node stays, and `0·f` is `nan`:
/// [`unevaluated_at_complex_infinity`]).  Folded when the node is built
/// ([`canon_function`]) and by `eval` (`eval::at_complex_infinity`, which
/// also takes the constant of a polynomial of degree 0), so that the two
/// read `0·f`, `0/f` and `f − f` alike.
///
/// | function at `zoo` | value |
/// |---|---|
/// | `P_n`, `T_n`, `U_n`, `H_n`, `L_n`, `L_n^(α)`, `C_n^(λ)`, `P_n^(a,b)` with numeric parameters, `P_n^m` (`0 ≤ m ≤ n`) | `zoo` for a degree `d ≥ 1` (`abs(p(z)) → ∞` in every direction; `abs((1 − z²)^(m/2))` grows like `abs(z)^m`); [`AtComplexInfinity::Constant`] for `d = 0`; 0 for the zero polynomial ([`polynomial_degree`]) |
/// | `B_(x₁, x₂)(a, b)`, `I_(x₁, x₂)(a, b)`, positive integer shapes, one limit `zoo`, the other finite | `zoo` (a polynomial of degree `a + b − 1` in the limit) |
/// | `Li_s(z)`, integer `s` | `−1` for `s = 0` (`z/(1 − z)`), `0` for `s < 0` (a rational function of degree `−s` over `(1 − z)^(1−s)`), `zoo` for `s ≥ 1` (`abs(ln(−z))^s/s!`) |
/// | `F(φ \| 0) = φ`, `Π(n \| 0) = π/(2√(1 − n))` | `zoo` at `φ = zoo`, `0` at `n = zoo` |
///
/// Up to 0.57 the closed forms met `zoo + zoo` and gave `nan`:
/// `legendre(3, zoo)`, `chebyshev_t(3, zoo)`, `betainc(1, 2, 0, zoo)`,
/// `polylog(−1, zoo)` (which tends to 0) were `nan` (mpmath:
/// `abs(P₃(10⁸e^{it}))` = `2.5·10²⁴` in every direction,
/// `abs(Li₋₁(10⁸e^{it}))` ≤ `10⁻⁸`).  The constructors fold the table as
/// `eval` does: had only `eval` folded it, `0·polylog(−1, zoo)` would be
/// `nan` when built (`eval`: `0·0 = 0`) and `legendre(3, zoo) − legendre(3,
/// zoo)` 0 (`eval`: `zoo − zoo = nan`).  The incomplete gamma functions and
/// `E₀(z) = e^{−z}/z` have no limit at `zoo` (`e^{−z}` grows for `Re z <
/// 0`): they stay.
pub(crate) fn at_complex_infinity(
    arena: &Arena,
    f: crate::base::libfn::LibFn,
    args: &[ExprId],
) -> Option<AtComplexInfinity> {
    use crate::base::libfn::LibFn;
    let zoo = |a: ExprId| ValueClass::special_atom(arena, a) == Some(ValueClass::ComplexInf);
    if !args.iter().any(|&a| zoo(a)) {
        return None;
    }
    let (&last, front) = args.split_last()?;
    if zoo(last) && front.iter().all(|&a| !zoo(a)) {
        let max = BigInt::from(arena.config.max_pow_exponent);
        if let Some(degree) = polynomial_degree(arena, f, front, &max) {
            return Some(match degree {
                Some(d) if d >= BigInt::one() => AtComplexInfinity::ComplexInfinity,
                Some(_) => AtComplexInfinity::Constant,
                None => AtComplexInfinity::Integer(0),
            });
        }
    }
    // A numeric integer argument.
    let int = |a: ExprId| {
        arena
            .as_num(a)
            .filter(|q| q.is_integer())
            .map(|q| q.to_integer())
    };
    match f {
        LibFn::BetaInc | LibFn::BetaIncRegularized => {
            let positive_integer = |a: ExprId| int(a).is_some_and(|k| k.is_positive());
            let &[a, b, x1, x2] = args else {
                return None;
            };
            (positive_integer(a)
                && positive_integer(b)
                && (zoo(x1) != zoo(x2))
                && [x1, x2]
                    .iter()
                    .all(|&x| zoo(x) || ValueClass::of(arena, x).is_finite_proved()))
            .then_some(AtComplexInfinity::ComplexInfinity)
        }
        LibFn::PolyLog if zoo(last) && front.len() == 1 => {
            let s = int(args[0])?;
            Some(if s.is_zero() {
                AtComplexInfinity::Integer(-1)
            } else if s.is_negative() {
                AtComplexInfinity::Integer(0)
            } else {
                AtComplexInfinity::ComplexInfinity
            })
        }
        // `F(φ | 0) = φ`, `Π(n | 0) = π/(2√(1 − n))` (DLMF 19.6.1).
        LibFn::EllipticF => match *args {
            [phi, m] if zoo(phi) && arena.is_zero_structural(m) => {
                Some(AtComplexInfinity::ComplexInfinity)
            }
            _ => None,
        },
        LibFn::EllipticPi => match *args {
            [n, m] if zoo(n) && arena.is_zero_structural(m) => Some(AtComplexInfinity::Integer(0)),
            _ => None,
        },
        _ => None,
    }
}

/// The degree in its variable of the orthogonal polynomial `f` with the
/// numeric degree and parameters `params` (all the arguments but the
/// variable), up to `max`: `Some(Some(d))`, `Some(None)` for the zero
/// polynomial, `None` when `f` is no polynomial there (`H_{−m}`, `L_{−m}` and
/// the rest, a symbolic parameter, a degree beyond `max`).
///
/// Negative degrees as `eval::classical_at_negative_degree` and
/// `eval::zero_polynomial_degree` read them (`P_{−m} = P_{m−1}`, `T_{−m} =
/// T_m`, `U_{−m} = −U_{m−2}`, `U_{−1} = 0`; `C`, `P^(a,b)`, `L^(α)` vanish).
/// The parametrised families from their coefficients, which are
/// polynomials in the parameters: `C_n^(λ)(x) = Σ_k (−1)^k
/// (λ)_{n−k}/(k!(n − 2k)!)·(2x)^{n−2k}` (DLMF 18.5.10), `P_n^(a,b)(x) =
/// Σ_m (−n)_m (n + a + b + 1)_m (a + m + 1)_{n−m}/(n! m!)·((1 − x)/2)^m`
/// (DLMF 18.5.7 with `(a + 1)_n/(a + 1)_m` cancelled), `L_n^(α)` of leading
/// coefficient `(−1)ⁿ/n!`; `P_n^m` for `0 ≤ m ≤ n` grows like `abs(z)^n`,
/// and vanishes for `m > n ≥ 0`.  Up to 0.57 `gegenbauer(3, −2, zoo)` was
/// `nan` (`C₃^(−2)(x) = −4x`).
fn polynomial_degree(
    arena: &Arena,
    f: crate::base::libfn::LibFn,
    params: &[ExprId],
    max: &BigInt,
) -> Option<Option<BigInt>> {
    use crate::base::libfn::LibFn;
    let n = arena
        .as_num(*params.first()?)
        .filter(|q| q.is_integer())?
        .to_integer();
    if &n.abs() > max {
        return None;
    }
    // Degree 0 is the constant 1 at every parameter (`P₀ = T₀ = U₀ = H₀ =
    // L₀ = C₀^(λ) = P₀^(a,b) = L₀^(α) = 1`).
    if n.is_zero()
        && matches!(
            f,
            LibFn::Legendre
                | LibFn::ChebyshevT
                | LibFn::ChebyshevU
                | LibFn::Hermite
                | LibFn::Laguerre
                | LibFn::Gegenbauer
                | LibFn::Jacobi
                | LibFn::AssocLaguerre
        )
    {
        return Some(Some(n));
    }
    let q = |k: usize| params.get(k).and_then(|&p| arena.as_num(p));
    // `(r)_j = 0`: `r` an integer `≤ 0` with `j > −r`.
    let vanishes = |r: &Q, j: &BigInt| r.is_integer() && !r.is_positive() && *j > -r.to_integer();
    let degree = |d: BigInt| Some(Some(d).filter(|d| !d.is_negative()));
    match f {
        _ if !n.is_negative()
            && matches!(
                f,
                LibFn::Legendre
                    | LibFn::ChebyshevT
                    | LibFn::ChebyshevU
                    | LibFn::Hermite
                    | LibFn::Laguerre
            ) =>
        {
            degree(n)
        }
        LibFn::Legendre => degree(-n - 1),
        LibFn::ChebyshevT => degree(-n),
        LibFn::ChebyshevU => degree(-n - 2),
        LibFn::AssocLaguerre => {
            q(1)?;
            degree(if n.is_negative() { BigInt::from(-1) } else { n })
        }
        LibFn::Gegenbauer => {
            let l = q(1)?;
            if n.is_negative() {
                return Some(None);
            }
            // The first `k` with `(λ)_{n−k} ≠ 0`.
            let k = if l.is_integer() && !l.is_positive() {
                (&n + l.to_integer()).max(BigInt::zero())
            } else {
                BigInt::zero()
            };
            degree(if BigInt::from(2) * &k > n {
                BigInt::from(-1)
            } else {
                &n - BigInt::from(2) * k
            })
        }
        LibFn::Jacobi => {
            let (a, b) = (q(1)?, q(2)?);
            if n.is_negative() {
                return Some(None);
            }
            let s = Q::from_integer(&n + BigInt::one()) + a + b;
            let top = n.to_u64()?;
            let m = (0..=top).rev().find(|&m| {
                let m = BigInt::from(m);
                let shifted = a + Q::from_integer(&m + BigInt::one());
                !vanishes(&s, &m) && !vanishes(&shifted, &(&n - &m))
            });
            degree(m.map_or(BigInt::from(-1), BigInt::from))
        }
        LibFn::AssocLegendre => {
            let m = q(1).filter(|m| m.is_integer())?.to_integer();
            if m.is_negative() {
                return None;
            }
            // `P^m_{−ν−1} = P^m_ν` (DLMF 14.9.5), as `eval` reads it.
            let n = if n.is_negative() {
                -n - BigInt::one()
            } else {
                n
            };
            degree(if m > n { BigInt::from(-1) } else { n })
        }
        _ => None,
    }
}

/// Is `f` a library function application at an exact infinity in an axis
/// direction (`±∞`, `±i·∞`) that has no finite value the constructors
/// know ([`at_infinity`]: [`AtInfinity::NoFiniteValue`]) and is not known
/// finite?  Every finite value of a tabulated pattern folds when the node
/// is built ([`lib_function_value`], [`at_infinity`]: `J_ν(∞) = 0`,
/// `Chi(i·∞) = iπ/2`, `H₀(∞) = 1`); what stays is infinite or has no
/// limit (`I_ν(−∞)` of a symbolic order, `Ai′(−∞)`, which oscillates
/// without bound, `H_n(∞)` of a symbolic degree), so `0·f` and `0/f` are
/// not folded to 0: they are `nan`, as `0·cosh(x·∞)` is.  Up to 0.52
/// `0·besseli(nu, −∞)` and `0·airyaiprime(−∞)` were 0, up to 0.53
/// `0·besselj(1, i·∞)`.  An untabulated pattern (the degree slot of an
/// orthogonal polynomial, `B_x(a, b)`, `F(i·∞ | m) = i·K(1 − m)`) and an
/// argument that is not an exact infinity (`x − ∞`) keep counting as
/// finite: an unfolded one is no sign of an infinite value.
fn library_value_unknown_at_infinity(arena: &Arena, f: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::base::truth::Truth;
    let ExprNode::Lib(lib, args) = arena.node(f) else {
        return false;
    };
    matches!(at_infinity(arena, *lib, args), AtInfinity::NoFiniteValue)
        && !Truth::from(AssumptionCache::new().query(arena, f, Props::FINITE)).is_true()
}

/// What a subexpression is at every point, as far as the rare paths of the
/// canonical constructors can tell ([`everywhere`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Everywhere {
    /// `0` at every point.
    Zero,
    /// `zoo` at every point: a negative power of a base that vanishes
    /// identically, not multiplied by a zero.  Undirected only: `±∞` and
    /// `|zoo| = ∞` are [`Everywhere::Unknown`] (`∞ + ∞` is not `nan`).
    Infinite,
    /// `nan` at every point: `0·zoo`, `zoo + zoo`, a function at an
    /// undefined argument or at `zoo` where it has no value (`sin(zoo)`).
    Undefined,
    /// None of these known (almost every expression).
    Unknown,
}

/// The [`Everywhere`] values and the zero tests ([`factor_vanishes`]) of
/// one rare path, shared by its calls of [`everywhere`].
#[derive(Default)]
struct EverywhereMemo {
    value: FxHashMap<ExprId, Everywhere>,
    vanishes: FxHashMap<ExprId, bool>,
}

impl EverywhereMemo {
    /// [`factor_vanishes`], once per node (`false` for an atom).
    fn vanishes(&mut self, arena: &mut Arena, f: ExprId) -> bool {
        if let Some(&known) = self.vanishes.get(&f) {
            return known;
        }
        let zero =
            (!arena.node(f).is_atom() || pinned_zero_symbol(arena, f)) && factor_vanishes(arena, f);
        self.vanishes.insert(f, zero);
        zero
    }

    /// Is `d` (a denominator) `0` or `nan` at every point: [`everywhere`],
    /// then the zero test of `d` itself.
    fn zero_or_undefined(&mut self, arena: &mut Arena, d: ExprId) -> bool {
        match everywhere(arena, d, self) {
            Everywhere::Zero | Everywhere::Undefined => true,
            Everywhere::Infinite => false,
            Everywhere::Unknown => self.vanishes(arena, d),
        }
    }

    /// Is `f` `zoo` or `nan` at every point ([`everywhere`])?
    fn infinite_or_undefined(&mut self, arena: &mut Arena, f: ExprId) -> bool {
        matches!(
            everywhere(arena, f, self),
            Everywhere::Infinite | Everywhere::Undefined
        )
    }
}

/// The arguments of `node` when it is a function application into which
/// [`everywhere`] looks: the elementary and special functions, library
/// functions (`Lib`) and user functions (`Apply`), and a power with a
/// symbolic exponent.
fn application_arguments(arena: &Arena, node: &ExprNode) -> Option<SmallVec<[ExprId; 2]>> {
    match node {
        ExprNode::Pow(b, e) if arena.as_num(*e).is_none() => Some(smallvec![*b, *e]),
        ExprNode::Apply(_, args) | ExprNode::Lib(_, args) => Some(args.iter().copied().collect()),
        _ => match node.func(arena) {
            Some((f, args)) if looked_into(f) => Some(args.into_iter().collect()),
            _ => None,
        },
    }
}

/// The node-backed functions into whose arguments [`everywhere`] looks:
/// all but the variadic `min`/`max`, the parts `re`, `im`, `conjugate`,
/// `arg` and the distributions `H`, `δ`, `KroneckerDelta` (until 0.49 an
/// or-pattern of the 34 others).
fn looked_into(f: crate::base::func::Func) -> bool {
    use crate::base::func::Func;
    !matches!(
        f,
        Func::Min
            | Func::Max
            | Func::Re
            | Func::Im
            | Func::Conjugate
            | Func::Arg
            | Func::Heaviside
            | Func::DiracDelta
            | Func::KroneckerDelta
    )
}

/// What `root` is at every point ([`Everywhere`]), bottom-up over its sums,
/// products, negations, numeric powers and function arguments (an explicit
/// post-order): a negative power is `zoo` when its base vanishes
/// identically ([`factor_vanishes`]) and `0` when its base is `zoo`; a
/// product is `nan` with a `nan` factor or with a `zoo` beside a factor
/// that vanishes (`0·zoo`), else `zoo` with a `zoo`, `0` with a `0`; a sum is
/// `nan` with a `nan` term or two `zoo` terms, `zoo` with one; a function
/// is `nan` at a `nan` argument and otherwise takes the canonical value at
/// `zoo` or `0` (`sin(zoo) = nan`, `ln(zoo) = zoo`, `sin 0 = 0`).  Up to
/// 0.34 only the terms of a sum were looked at, one level deep: in `0/(c +
/// (1/(x·d₁) + 1/d₂)/d₃)` with `d₁`, `d₂`, `d₃` zero the denominator was
/// taken for `zoo` (a term over `d₃`) and the quotient for `0`, while the
/// numerator `zoo + zoo` of that term makes it `nan`.  `root` itself is
/// not tested for `0` (the caller does that where a zero matters); the
/// zero tests run only for the bases of negative powers and beside a `zoo`
/// factor, so a tree without a negative power costs one walk.
fn everywhere(arena: &mut Arena, root: ExprId, memo: &mut EverywhereMemo) -> Everywhere {
    if let Some(&known) = memo.value.get(&root) {
        return known;
    }
    // (node, children pushed?)
    let mut stack: SmallVec<[(ExprId, bool); 16]> = smallvec![(root, false)];
    while let Some(&(id, expanded)) = stack.last() {
        if memo.value.contains_key(&id) {
            stack.pop();
            continue;
        }
        let node = arena.node(id).clone();
        if !expanded {
            if let Some(top) = stack.last_mut() {
                top.1 = true;
            }
            let children: SmallVec<[ExprId; 6]> = match &node {
                ExprNode::Add(ch) | ExprNode::Mul(ch) => ch.clone(),
                ExprNode::Neg(c) => smallvec![*c],
                ExprNode::Pow(b, e) if arena.as_num(*e).is_some() => smallvec![*b],
                n => application_arguments(arena, n)
                    .map(|a| a.into_iter().collect())
                    .unwrap_or_default(),
            };
            stack.extend(
                children
                    .into_iter()
                    .filter(|c| !memo.value.contains_key(c))
                    .map(|c| (c, false)),
            );
            continue;
        }
        stack.pop();
        let get = |memo: &EverywhereMemo, c: ExprId| {
            memo.value.get(&c).copied().unwrap_or(Everywhere::Unknown)
        };
        let value = match &node {
            ExprNode::NaN => Everywhere::Undefined,
            ExprNode::ComplexInfinity => Everywhere::Infinite,
            ExprNode::Num(nid) if arena.num(*nid).is_zero() => Everywhere::Zero,
            ExprNode::Neg(c) => get(memo, *c),
            ExprNode::Pow(b, e) if arena.as_num(*e).is_some() => {
                let negative = arena.as_num(*e).is_some_and(Signed::is_negative);
                match (get(memo, *b), negative) {
                    (Everywhere::Undefined, _) => Everywhere::Undefined,
                    (Everywhere::Zero, false) | (Everywhere::Infinite, true) => Everywhere::Zero,
                    (Everywhere::Zero, true) | (Everywhere::Infinite, false) => {
                        Everywhere::Infinite
                    }
                    (Everywhere::Unknown, true) if memo.vanishes(arena, *b) => Everywhere::Infinite,
                    (Everywhere::Unknown, _) => Everywhere::Unknown,
                }
            }
            ExprNode::Mul(children) => {
                let values: SmallVec<[Everywhere; 6]> =
                    children.iter().map(|&c| get(memo, c)).collect();
                let infinite = values.contains(&Everywhere::Infinite);
                if values.contains(&Everywhere::Undefined)
                    || (infinite && values.contains(&Everywhere::Zero))
                {
                    Everywhere::Undefined
                } else if infinite {
                    // `0·zoo` for a factor that vanishes identically.
                    let mut undefined = false;
                    for (&c, &v) in children.iter().zip(&values) {
                        let positive = match arena.node(c) {
                            ExprNode::Pow(_, e) => arena.as_num(*e).is_none_or(Signed::is_positive),
                            _ => true,
                        };
                        if v == Everywhere::Unknown && positive && memo.vanishes(arena, c) {
                            undefined = true;
                            break;
                        }
                    }
                    if undefined {
                        Everywhere::Undefined
                    } else {
                        Everywhere::Infinite
                    }
                } else if values.contains(&Everywhere::Zero) {
                    Everywhere::Zero
                } else {
                    Everywhere::Unknown
                }
            }
            ExprNode::Add(children) => {
                let values: SmallVec<[Everywhere; 6]> =
                    children.iter().map(|&c| get(memo, c)).collect();
                let infinite = values
                    .iter()
                    .filter(|&&v| v == Everywhere::Infinite)
                    .count();
                if values.contains(&Everywhere::Undefined) || infinite > 1 {
                    Everywhere::Undefined
                } else if infinite == 1 {
                    Everywhere::Infinite
                } else if values.iter().all(|&v| v == Everywhere::Zero) {
                    Everywhere::Zero
                } else {
                    Everywhere::Unknown
                }
            }
            n => match application_arguments(arena, n) {
                Some(args) => {
                    let values: SmallVec<[Everywhere; 2]> =
                        args.iter().map(|&c| get(memo, c)).collect();
                    if values.contains(&Everywhere::Undefined) {
                        Everywhere::Undefined
                    } else if values
                        .iter()
                        .any(|v| matches!(v, Everywhere::Infinite | Everywhere::Zero))
                    {
                        application_at_special_values(arena, id, &args, &values)
                    } else {
                        Everywhere::Unknown
                    }
                }
                None => Everywhere::Unknown,
            },
        };
        memo.value.insert(id, value);
    }
    memo.value
        .get(&root)
        .copied()
        .unwrap_or(Everywhere::Unknown)
}

/// [`everywhere`] for each of `ids` (one memo for all): `ratsimp` asks it
/// of its generators, which are `nan` at every point when they are
/// functions at such arguments (`sin(c/d)` with `d` a constant that is 0).
/// The zero tests are those of the canonical constructors
/// ([`factor_vanishes`]); a tree without a negative power costs one walk.
/// [`Everywhere::Undefined`] is final (`nan` absorbs whatever else is
/// undetected); `Zero` and `Infinite` may be `nan` where a zero the tests
/// do not see (`sin²x + cos²x − 1`) meets them.
///
/// `zero_bases` are known to vanish identically — by an identity of their
/// functions that the residue test of [`factor_vanishes`] does not see
/// (`tan x·cos x − sin x`), decided by the caller: their negative powers
/// are `zoo`.
pub(crate) fn everywhere_values(
    arena: &mut Arena,
    ids: &[ExprId],
    zero_bases: &[ExprId],
) -> SmallVec<[Everywhere; 4]> {
    let mut memo = EverywhereMemo::default();
    for &b in zero_bases {
        memo.vanishes.insert(b, true);
    }
    ids.iter()
        .map(|&id| everywhere(arena, id, &mut memo))
        .collect()
}

/// The function application `id` with each argument that is `zoo` or `0`
/// at every point (`values`, parallel to `args`) replaced by that value,
/// classified by its canonical form: `sin(zoo) = nan` is
/// [`Everywhere::Undefined`], `ln(zoo) = zoo` and `ln 0 = zoo`
/// [`Everywhere::Infinite`], `sin 0 = 0` [`Everywhere::Zero`], anything
/// else — left unevaluated (`atan(zoo)`, `f(zoo)`) or a directed infinity
/// (`|zoo| = ∞`) — [`Everywhere::Unknown`].
fn application_at_special_values(
    arena: &mut Arena,
    id: ExprId,
    args: &[ExprId],
    values: &[Everywhere],
) -> Everywhere {
    let (zoo, zero) = (arena.complex_infinity, arena.zero);
    let replacement: SmallVec<[(ExprId, ExprId); 2]> = args
        .iter()
        .zip(values)
        .filter_map(|(&a, v)| match v {
            Everywhere::Infinite => Some((a, zoo)),
            Everywhere::Zero => Some((a, zero)),
            _ => None,
        })
        .collect();
    let at = crate::base::walk::rebuild_with(arena, id, &|c| {
        replacement
            .iter()
            .find(|(a, _)| *a == c)
            .map_or(c, |&(_, v)| v)
    });
    match arena.node(at) {
        ExprNode::NaN => Everywhere::Undefined,
        ExprNode::ComplexInfinity => Everywhere::Infinite,
        ExprNode::Num(nid) if arena.num(*nid).is_zero() => Everywhere::Zero,
        _ => Everywhere::Unknown,
    }
}

/// The bases collected in `canon_mul`, split for [`zero_times_rest`].
struct SplitBases {
    /// The bases without a negative numeric exponent.
    positive: SmallVec<[ExprId; 8]>,
    /// The bases with one: the denominators.
    negative: SmallVec<[ExprId; 4]>,
    /// The exponents `e` of the factors `0^e` (collected under the base 0,
    /// which is no factor itself).
    zero_exponents: SmallVec<[ExprId; 2]>,
}

/// The bases collected in `canon_mul`: those without and those with a
/// negative numeric exponent, and the exponents of the factors `0^e`.
fn split_bases(arena: &Arena, bases: &FxHashMap<ExprId, SmallVec<[ExprId; 4]>>) -> SplitBases {
    let mut split = SplitBases {
        positive: SmallVec::new(),
        negative: SmallVec::new(),
        zero_exponents: SmallVec::new(),
    };
    for (&b, exps) in bases {
        if arena.is_zero_structural(b) {
            split.zero_exponents.extend(exps.iter().copied());
        } else if exps
            .iter()
            .any(|&e| arena.as_num(e).is_some_and(Signed::is_negative))
        {
            split.negative.push(b);
        } else {
            split.positive.push(b);
        }
    }
    split
}

/// `r·sign(r) → |r|` among the collected `bases` of a product (both factors
/// to the power 1), for a real `r`: `r·sign(r) = r²/|r| = |r|` for `r ≠ 0`,
/// and `0` at `r = 0`.  For a non-real `r` the product is `r²/|r| ≠ |r|`, so
/// the assumptions must say real.  Before 0.43 the product stayed, and
/// `|m| − m·sign(m)` was a hidden zero: `ratsimp(((x + 1)² − x² − 2x −
/// 1)/(|m| − m·sign(m)))` gave 0 for a real `m` (the value is `0/0 =
/// nan`; SymPy keeps `Abs(m) - m*sign(m)` and `simplify` gives 0).  Only a
/// factor `|r|` that is a single base is produced (`|m − 1|`); an `r` whose
/// modulus takes a coefficient out is left.
fn fold_real_times_sign(arena: &mut Arena, bases: &mut FxHashMap<ExprId, SmallVec<[ExprId; 4]>>) {
    use crate::base::assumptions::{AssumptionCache, Props};
    let one = arena.one;
    let pairs: SmallVec<[ExprId; 2]> = bases
        .iter()
        .filter_map(|(&b, exps)| match *arena.node(b) {
            ExprNode::Sign(r)
                if exps.as_slice() == [one]
                    && bases.get(&r).is_some_and(|e| e.as_slice() == [one]) =>
            {
                Some(b)
            }
            _ => None,
        })
        .collect();
    if pairs.is_empty() {
        return;
    }
    let mut cache = AssumptionCache::new();
    for s in pairs {
        let ExprNode::Sign(r) = *arena.node(s) else {
            continue;
        };
        if cache.query(arena, r, Props::REAL) != Some(true) {
            continue;
        }
        let modulus = arena.abs(r);
        if matches!(arena.node(modulus), ExprNode::Mul(_) | ExprNode::Num(_)) {
            continue;
        }
        bases.remove(&s);
        bases.remove(&r);
        let (base, exp) = arena.as_base_exp(modulus);
        bases.entry(base).or_default().push(exp);
    }
}

/// `E^a·exp(b) → exp(a + b)` among the combined `factors` of a product
/// (base, exponent), for rational exponents: the factors that are powers of
/// *e* with a rational exponent — the constant `E`, `exp(q)` and their
/// rational powers — become one factor `(E, Σ)`, which `canon_mul` builds
/// as `1`, `E` or `exp(Σ)`.  SymPy's `Mul.flatten` gathers them the same
/// way (`exp(q).as_base_exp()` is `(E, q)`, and exponents with the same
/// term `1` add).  Before 0.50 `E·exp(−1)` and `exp(2)·exp(−2)` stayed
/// products — ones nothing reduced, `simplify(E·exp(−1))` included.
///
/// A symbolic exponent stays apart, as in SymPy: `E·exp(x)` is what
/// `expand(exp(x + 1))` gives, and merging it back would undo the
/// expansion.  Only distinct factors merge: `exp(q)²` (one base) keeps the
/// form `canon_pow` builds.
fn merge_constant_e_powers(arena: &mut Arena, factors: &mut SmallVec<[(ExprId, ExprId); 8]>) {
    // The rational `q` of a base `E` or `exp(q)`, with a rational exponent.
    fn e_power(arena: &Arena, base: ExprId, exp: ExprId) -> Option<(Option<&Q>, &Q)> {
        let k = arena.as_num(exp)?;
        match *arena.node(base) {
            ExprNode::E => Some((None, k)),
            ExprNode::Exp(q) => Some((Some(arena.as_num(q)?), k)),
            _ => None,
        }
    }
    let count = factors
        .iter()
        .filter(|&&(b, e)| e_power(arena, b, e).is_some())
        .count();
    if count < 2 {
        return;
    }
    let mut total = Q::zero();
    factors.retain(|&mut (b, e)| match e_power(arena, b, e) {
        Some((q, k)) => {
            total = match q {
                Some(q) => q_add(&total, &q_mul(q, k)),
                None => q_add(&total, k),
            };
            false
        }
        None => true,
    });
    tracing::trace!(%total, "canon_mul: merged rational powers of e");
    let nid = arena.intern_num(total);
    let exponent = arena.intern(ExprNode::Num(nid));
    factors.push((arena.e_const, exponent));
}

/// The factors (base, exponent) a product gets from its powers `0^e` of the
/// base 0 with the `exponents` given (two or more; each of undecided sign:
/// [`canon_pow`] folds the others), pushed onto `factors`.
///
/// `0^e` is piecewise in its exponent — 0 for `re e > 0`, 1 at `e = 0`,
/// `zoo` for `re e < 0`, `nan` elsewhere — so the exponents add only where
/// the powers are of one kind: `0^y·0^z` is `0·zoo = nan` for `re y > 0 > re
/// z`, where `0^(y + z)` is 0, 1 or `zoo`.  The exponents are grouped by
/// their term `t` without its rational coefficient (`e = c·t`):
///
/// * those of one `t` and one sign of `c` are of one kind at every `t` and
///   add: `0^y·0^(2y) = 0^(3y)`;
/// * a `t` met with both signs, `0^(a·t)·0^(−b·t)`, is `0·zoo = nan` wherever
///   `t ≠ 0` and `1` at `t = 0`: the factor `Piecewise((1, Eq(t, 0)), (nan,
///   True))` (one for all such `t`);
/// * powers of different `t` stay separate factors: `0^y·0^z` is kept.
///
/// Up to 0.58 all exponents were added like those of any other base:
/// `0^y·0^(−y)` was `1` and `0^y·0^z` was `0^(y + z)`.
fn push_zero_power_factors(
    arena: &mut Arena,
    exponents: &[ExprId],
    factors: &mut SmallVec<[(ExprId, ExprId); 8]>,
) {
    // (term, sum of its positive coefficients, sum of its negative ones)
    let mut groups: SmallVec<[(ExprId, Q, Q); 2]> = SmallVec::new();
    for &e in exponents {
        let (c, t) = arena.as_coeff_term(e);
        let slot = match groups.iter().position(|g| g.0 == t) {
            Some(slot) => slot,
            None => {
                groups.push((t, Q::zero(), Q::zero()));
                groups.len() - 1
            }
        };
        let group = &mut groups[slot];
        if c.is_negative() {
            group.2 = q_add(&group.2, &c);
        } else {
            group.1 = q_add(&group.1, &c);
        }
    }
    let zero = arena.zero;
    let mut cancelled: SmallVec<[ExprId; 2]> = SmallVec::new();
    for (t, positive, negative) in groups {
        if !positive.is_zero() && !negative.is_zero() {
            cancelled.push(arena.eq_(t, zero));
        } else {
            let exponent = arena.make_coeff_term(q_add(&positive, &negative), t);
            factors.push((zero, exponent));
        }
    }
    if !cancelled.is_empty() {
        tracing::debug!("canon_mul: 0^t·0^(−t) → Piecewise((1, t = 0), (nan, True))");
        let guard = where_defined(arena, arena.one, cancelled);
        factors.push((guard, arena.one));
    }
}

/// Build a canonical `Mul` node from the given factors.
///
/// Uses an explicit stack for flattening and an [`FxHashMap`] for
/// like-base combination.
pub(crate) fn canon_mul(arena: &mut Arena, args: &[ExprId]) -> ExprId {
    // Fast path: 0 or 1 arguments
    if args.is_empty() {
        return arena.one;
    }
    if args.len() == 1 {
        return args[0];
    }

    // Fast path: 2 numeric args → skip HashMap
    if args.len() == 2 {
        let (a, b) = (args[0], args[1]);
        if let (Some(na), Some(nb)) = (arena.as_num(a), arena.as_num(b)) {
            let product = q_mul(na, nb);
            if product.is_zero() {
                tracing::debug!("canon_mul fast path: two numerics multiply to zero");
                return arena.zero;
            }
            if product.is_one() {
                return arena.one;
            }
            let nid = arena.intern_num(product);
            return arena.intern(ExprNode::Num(nid));
        }
    }

    // Running numeric coefficient.
    let mut coeff: Q = Ratio::one();

    // Map: base → list of exponents to be summed.
    let mut bases: FxHashMap<ExprId, SmallVec<[ExprId; 4]>> = FxHashMap::default();

    // Track special values.
    let mut saw_infinity = false; // any kind of infinity (oo, -oo, zoo)

    // Explicit stack for iterative flattening.
    let mut stack: SmallVec<[ExprId; 16]> = SmallVec::from_slice(args);

    while let Some(id) = stack.pop() {
        match arena.node(id).clone() {
            ExprNode::Mul(children) => {
                // Flatten: push children back onto the stack.
                stack.extend_from_slice(&children);
            }

            ExprNode::NaN => {
                return arena.nan;
            }

            ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity => {
                saw_infinity = true;
                // Track sign for directed infinities.
                match arena.node(id) {
                    ExprNode::NegInfinity => {
                        coeff = -coeff;
                    }
                    ExprNode::ComplexInfinity => {
                        // zoo absorbs sign information.
                        return handle_mul_with_zoo(arena, &mut stack, &bases, &coeff);
                    }
                    _ => {} // Infinity — no sign change.
                }
            }

            ExprNode::Num(nid) => {
                coeff = q_mul(&coeff, arena.num(nid));
                if coeff.is_zero() {
                    // 0 × rest: `nan` if any infinity/NaN is involved
                    // (already seen, still on the stack, or hidden in a
                    // collected base), otherwise `0`.
                    let split = split_bases(arena, &bases);
                    return zero_times_rest(
                        arena,
                        saw_infinity,
                        stack.iter().chain(split.positive.iter()),
                        split.negative,
                        split.zero_exponents,
                    );
                }
            }

            ExprNode::Neg(inner) => {
                // −x contributes a factor of −1 to the coefficient
                // and pushes x back for further processing.
                coeff = -coeff;
                stack.push(inner);
            }

            _ => {
                let (base, exp) = arena.as_base_exp(id);
                bases.entry(base).or_default().push(exp);
            }
        }
    }

    // If coefficient became zero during processing.
    if coeff.is_zero() {
        let split = split_bases(arena, &bases);
        return zero_times_rest(
            arena,
            saw_infinity,
            split.positive.iter(),
            split.negative,
            split.zero_exponents,
        );
    }

    if cancels_vanishing_sum(arena, &bases) {
        tracing::debug!("canon_mul: s·s⁻¹ with s identically 0 → nan");
        return arena.nan;
    }

    fold_real_times_sign(arena, &mut bases);

    // Build the combined factors.
    let mut factors: SmallVec<[(ExprId, ExprId); 8]> = SmallVec::new();
    for (base, exponents) in &bases {
        let combined_exp: ExprId = if exponents.len() == 1 {
            exponents[0]
        } else if *base == arena.zero {
            // The exponents of the base 0 do not add (`0^y·0^(−y)` is not
            // `0⁰`): see [`push_zero_power_factors`].
            push_zero_power_factors(arena, exponents, &mut factors);
            continue;
        } else {
            canon_add(arena, exponents)
        };
        factors.push((*base, combined_exp));
    }

    merge_constant_e_powers(arena, &mut factors);

    // ── Exponent-grouping pass for positive numeric bases ──────────
    //
    // Combine factors like 2^{1/2} · 3^{1/2} → 6^{1/2} = √6.
    //
    // For factors whose bases are positive rational numbers and whose
    // exponents are non-integer rationals, group by exponent and multiply
    // the bases.  This is unconditionally valid for positive reals (no
    // branch-cut issues).  Matches SymPy's `pnum_rat` grouping in
    // `Mul.flatten`.
    //
    // Only numeric bases are combined — symbolic bases (x, y) are left
    // for `powsimp_base` with assumption checking.
    {
        let mut combined_factors: SmallVec<[(ExprId, ExprId); 8]> = SmallVec::new();
        // Map: exponent ExprId → (product of bases as Ratio, indices consumed).
        // The product is formed only once a second base joins a group
        // (`q_mul` is linear for integers; a lone radical pays nothing).
        let mut exp_groups: FxHashMap<ExprId, (Q, usize)> = FxHashMap::default();
        let mut factor_used: SmallVec<[bool; 8]> = smallvec::smallvec![false; factors.len()];

        for (i, &(base, exp_id)) in factors.iter().enumerate() {
            // Only group: base is a positive rational number, exponent is a
            // non-integer rational (fractional like 1/2, 1/3, 2/3, -1/2).
            let dominated = 'check: {
                let base_r = match arena.as_num(base) {
                    Some(r) if r.is_positive() => r.clone(),
                    _ => break 'check false,
                };
                let exp_r = match arena.as_num(exp_id) {
                    Some(r) if !r.is_integer() => r,
                    _ => break 'check false,
                };
                let _ = exp_r; // used only for the is_integer check

                match exp_groups.entry(exp_id) {
                    std::collections::hash_map::Entry::Occupied(mut slot) => {
                        let entry = slot.get_mut();
                        entry.0 = q_mul(&entry.0, &base_r);
                        entry.1 += 1;
                    }
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert((base_r, 1));
                    }
                }
                factor_used[i] = true;
                true
            };
            let _ = dominated;
        }

        // Check if any group actually combined multiple bases.
        let any_combined = exp_groups.values().any(|&(_, count)| count > 1);

        if any_combined {
            tracing::trace!(
                n_groups = exp_groups.len(),
                "canon_mul: exponent-grouping combined positive numeric bases"
            );

            // Keep non-grouped factors as-is.
            for (i, &(base, exp_id)) in factors.iter().enumerate() {
                if !factor_used[i] {
                    combined_factors.push((base, exp_id));
                }
            }

            // Emit combined factors (and single-entry groups).
            for (exp_id, (product, count)) in &exp_groups {
                if *count <= 1 {
                    // Single entry — find and re-emit the original factor.
                    for (i, &(_, e)) in factors.iter().enumerate() {
                        if factor_used[i] && e == *exp_id {
                            combined_factors.push(factors[i]);
                            break;
                        }
                    }
                } else {
                    // Multiple bases combined.  Build Pow(product, exp).
                    // canon_pow will simplify further (e.g., 8^{1/3} → 2).
                    let prod_nid = arena.intern_num(product.clone());
                    let prod_base = arena.intern(ExprNode::Num(prod_nid));
                    combined_factors.push((prod_base, *exp_id));
                }
            }

            factors = combined_factors;
        }
    }

    // Sort factors by base SortKey.
    factors.sort_by(|(a, _), (b, _)| arena.sort_key(*a).cmp(arena.sort_key(*b)));

    // Build result argument list.
    //
    // We defer adding the numeric coefficient until AFTER processing all
    // factors, because a factor with exp == 1 whose base is a Num, or a
    // `canon_pow` call that fully evaluates to a Num (e.g. 2^3 → 8),
    // must be absorbed back into the running coefficient — not pushed as
    // a separate Mul child.  Without this, expressions like
    //     `rationalize_denom(1/(1+√2))`
    // can produce `Mul([1, -1, √2])` (two Num children) instead of
    // `Mul([-1, √2])`.
    let mut result_args: SmallVec<[ExprId; 6]> = SmallVec::new();
    // Set when `canon_pow` hands back a product (see below).
    let mut has_nested_mul = false;

    // A combined factor can also be an infinity or `nan` —
    // `(−∞)^(1/2)·(−∞)^(1/2) → −∞` — which only the flattening pass knows how
    // to multiply (the sign of the other factors, `0·∞ = nan`); before 0.30
    // it stayed an ordinary factor, `x·sin(f(x))·(−∞)`, whose display parsed
    // as a different expression (found by `fuzz_roundtrip`).  Such a factor
    // triggers the re-flattening below.
    let singular = |arena: &Arena, id: ExprId| {
        matches!(
            arena.node(id),
            ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN
        )
    };
    for (base, exp) in factors {
        if exp == arena.one {
            // If the base is itself a numeric literal, absorb it into the
            // running coefficient instead of adding a second Num child.
            if let Some(val) = arena.as_num(base) {
                coeff = q_mul(&coeff, val);
                continue;
            }
            if singular(arena, base) {
                has_nested_mul = true;
            }
            // Exponents that sum to 1 on a `Pow(Mul(…), e)` base expose the
            // product itself: `√(-ω²)·√(-ω²) → -ω²`.
            if matches!(arena.node(base), ExprNode::Mul(_)) {
                has_nested_mul = true;
            }
            result_args.push(base);
        } else if arena.is_zero_structural(exp) {
            // base^0 = 1 — skip this factor entirely.
            continue;
        } else {
            let pow_id = canon_pow(arena, base, exp);
            if pow_id == arena.one {
                continue;
            }
            // canon_pow may have fully evaluated to a number
            // (e.g. 2^3 → 8).  Absorb it into the coefficient.
            if let Some(val) = arena.as_num(pow_id) {
                coeff = q_mul(&coeff, val);
                continue;
            }
            if matches!(arena.node(pow_id), ExprNode::Mul(_)) || singular(arena, pow_id) {
                has_nested_mul = true;
            }
            result_args.push(pow_id);
        }
    }

    // A factor whose exponents combined can also expose a base another
    // factor has: `W^(−1/3)·W^(4/3) → W` for `W = (x·y)^(−3/2)`, next to
    // `(x·y)^(3/2)` — the two are powers of `x·y`, and the product is 1.
    // (Before 0.30 such a product stayed `(x·y)^(−3/2)·(x·y)^(3/2)`,
    // which is not canonical: its display parsed back as 1; found by
    // `fuzz_roundtrip`.)  Regroup once when two factors share a base.
    // (Two powers of 0 are not regrouped: their exponents do not add.)
    if !has_nested_mul && result_args.len() > 1 {
        let mut seen: FxHashSet<ExprId> = FxHashSet::default();
        for &r in &result_args {
            let base = arena.as_base_exp(r).0;
            if base != arena.zero && !seen.insert(base) {
                has_nested_mul = true;
                break;
            }
        }
    }

    // A factor can turn out to be a *product*: `canon_pow` performs radical
    // extraction after exponent grouping (`√6·√3 → 18^(1/2) → 3·√2`),
    // handles negative bases (`(-4)^(1/2) → 2·i`), distributes integer
    // powers over a `Mul` base after exponents combined
    // (`(x·y)^(1/2)·(x·y)^(3/2) → x²·y²`), and a `Mul` base whose exponents
    // sum to 1 is exposed directly.  Its children must be merged with the
    // other factors (they may share bases or carry a numeric coefficient),
    // so re-canonicalise the whole product once.  The recursive call sees
    // only flat, already-canonical factors and therefore terminates.
    if has_nested_mul && !coeff.is_zero() {
        tracing::trace!("canon_mul: canon_pow produced a Mul factor; re-flattening");
        let mut all: SmallVec<[ExprId; 8]> = SmallVec::new();
        if !coeff.is_one() {
            let nid = arena.intern_num(coeff);
            all.push(arena.intern(ExprNode::Num(nid)));
        }
        all.extend(result_args);
        if saw_infinity {
            // Sign was already folded into `coeff` when `-∞` was consumed.
            all.push(arena.infinity);
        }
        return canon_mul(arena, &all);
    }

    // Re-check for zero after absorbing numeric factors.
    if coeff.is_zero() {
        return zero_times_rest(arena, saw_infinity, result_args.iter(), [], []);
    }

    // Now prepend the numeric coefficient (if not 1, or if there are
    // no symbolic factors left).
    if !coeff.is_one() || result_args.is_empty() {
        let nid = arena.intern_num(coeff.clone());
        result_args.insert(0, arena.intern(ExprNode::Num(nid)));
    }

    if saw_infinity {
        return directed_infinity(arena, coeff.is_negative(), &result_args);
    }

    // Re-sort result_args by the ACTUAL sort key of each entry.
    //
    // The `factors` list was sorted by *base* sort key, but
    // `canon_pow(base, exp)` may return a node with a different type
    // (and thus rank) than the base alone.  For example, `cos(x)` is a
    // Function (rank 50) but `Pow(cos(x), -1)` is a Pow (rank 20).
    // Sorting after construction ensures the final Mul children obey
    // the canonical sort order — mirroring what `canon_add` already
    // does for its reconstructed symbolic terms.
    result_args.sort_by(|a, b| arena.sort_key(*a).cmp(arena.sort_key(*b)));

    // ── Distribution: Number * Add → distributed sum ──────────────
    //
    // When the final result is exactly `[Number, Add]` (a single numeric
    // coefficient times a sum), distribute the coefficient over the
    // Add's terms.  This matches SymPy's `Mul.flatten` final step and
    // is required for correct like-term cancellation in `canon_add`.
    //
    // Without this, `Mul(-1, Add(-1, x))` would remain as a single
    // opaque term inside a sum, preventing cancellation with the bare
    // terms `-1` and `x`.
    //
    // Only `Number * Add` distributes — symbolic products like
    // `x * (y + z)` are never distributed (that's `.expand()`).
    //
    // This does mean that `2*(x+1)` → `2 + 2*x` while `y*2*(x+1)`
    // stays as `2*y*(x+1)`.  These are different canonical forms for
    // expressions built with different grouping — which is consistent
    // with SymPy's 20-year-proven approach.  Full normalisation across
    // groupings is the job of `.expand()` (Stage 7).
    if result_args.len() == 2 {
        let first = result_args[0];
        let second = result_args[1];
        if let ExprNode::Num(_) = arena.node(first)
            && let ExprNode::Add(add_children) = arena.node(second).clone()
        {
            let distributed: SmallVec<[ExprId; 6]> = add_children
                .iter()
                .map(|&child| canon_mul(arena, &[first, child]))
                .collect();
            return canon_add(arena, &distributed);
        }
    }

    // Final assembly.
    let result = match result_args.len() {
        0 => arena.one,
        1 => result_args[0],
        _ => arena.intern(ExprNode::Mul(result_args)),
    };
    debug_assert!(
        verify_canonical_shallow(arena, result).is_empty(),
        "canon_mul produced non-canonical result: {:?}",
        verify_canonical_shallow(arena, result)
    );
    result
}

/// `±∞` times the non-numeric `factors` (a non-zero coefficient of sign
/// `negative` already taken out): a *directed infinity*.
///
/// `±∞` absorbs the factors whose sign is known (a positive one keeps the
/// direction, a negative one reverses it: `π·∞ = ∞`, `(3 − π)·∞ = −∞`).  A
/// factor of unknown sign or not real (`x`, `i`, `exp(z)`, `1 + i`) carries
/// the direction, which canonical arithmetic cannot pick: the product stays
/// `Mul([−1?, f₁, …, fₖ, ∞])` — only the sign of the coefficient survives,
/// `−∞` never appears inside a product, and the factors are sorted as in
/// any product (`∞` last).  So `(x·∞)` at `x = 2`, `−1`, `i`, `0` is `∞`,
/// `−∞`, `i·∞`, `0·∞ = nan`, the values SymPy gives (it too keeps
/// `oo*x` and decides at substitution time; its outputs were the oracle,
/// not its code), and [`canon_add`] makes `x·∞ − x·∞ = nan`.
///
/// Canonical products may assume a *generic* symbol (`x·x⁻¹ = 1`,
/// `x·zoo = zoo`), but a direction has no generic value.  Before 0.30 every
/// factor was absorbed (`x·∞ = ∞`, wrong at `x = −1`); the 0.30 hunts then
/// made the product `zoo`, which lost the direction (`i·∞`, `atan(i)`) and
/// gave `zoo` instead of `nan` at `x = 0`.
fn directed_infinity(arena: &mut Arena, negative: bool, factors: &[ExprId]) -> ExprId {
    let mut negative = negative;
    let mut kept: SmallVec<[ExprId; 6]> = SmallVec::new();
    for &f in factors {
        if arena.as_num(f).is_some() {
            continue;
        }
        match known_sign_memo(arena, f) {
            Some(Ordering::Greater) => {}
            Some(Ordering::Less) => negative = !negative,
            _ => kept.push(f),
        }
    }
    if kept.is_empty() {
        return if negative {
            arena.neg_infinity
        } else {
            arena.infinity
        };
    }
    if negative {
        kept.push(arena.neg_one);
    }
    kept.push(arena.infinity);
    kept.sort_by(|a, b| arena.sort_key(*a).cmp(arena.sort_key(*b)));
    let result = arena.intern(ExprNode::Mul(kept));
    debug_assert!(
        verify_canonical_shallow(arena, result).is_empty(),
        "directed infinity not canonical: {:?}",
        verify_canonical_shallow(arena, result)
    );
    result
}

/// Is `id` a directed infinity `Mul([−1?, f₁, …, fₖ, ∞])` (see
/// [`directed_infinity`])?  Its coefficient-free part is the key under which
/// [`canon_add`] collects it.
pub(crate) fn is_directed_infinity(arena: &Arena, id: ExprId) -> bool {
    // (identity: the shape `c·∞`; its class is `ValueClass::of`'s)
    matches!(arena.node(id), ExprNode::Mul(ch) if ch.contains(&arena.infinity))
}

/// The sign memoised under `(key, fact)` ([`Arena::constant_facts`]; the
/// keys of [`known_sign_memo`] are `FactKey::Sign` for a real constant
/// factor `f`, keyed by `f`, and `FactKey::RePart`/`FactKey::ImPart` for
/// `re(g)`/`im(g)`, keyed by `g`): `None` when not memoised, `Some(None)`
/// when memoised as unknown.
fn sign_fact(
    arena: &Arena,
    key: ExprId,
    fact: crate::base::arena::FactKey,
) -> Option<Option<Ordering>> {
    arena
        .constant_fact(key, fact)
        .map(|signs| signs.to_ordering())
}

fn remember_sign(
    arena: &mut Arena,
    key: ExprId,
    fact: crate::base::arena::FactKey,
    sign: Option<Ordering>,
) {
    use crate::base::truth::SignSet;
    let signs = sign.map_or(SignSet::UNKNOWN, SignSet::from_ordering);
    arena.remember_constant_fact(key, fact, signs);
}

/// [`known_sign`], memoised per arena for a symbol-free factor: canonical
/// products rebuild `f·∞` on every pass over an expression and the parser
/// on every parse, and the certified value of `im(Γ(z))` evaluates `Γ` at
/// a complex point each time.  `re(g)` and `im(g)` take their signs from one
/// certified value of `g` ([`Arena::part_signs`]) when it settles both.
/// Inputs with `re(…)·∞` and `im(…)·∞` of the same complex Γ chain were
/// 2–2.5 times slower in 0.46 than in 0.43.
fn known_sign_memo(arena: &mut Arena, f: ExprId) -> Option<Ordering> {
    let part = match *arena.node(f) {
        ExprNode::Re(g) => Some((g, true)),
        ExprNode::Im(g) => Some((g, false)),
        _ => None,
    };
    use crate::base::arena::FactKey;
    let (key, fact) = match part {
        Some((g, true)) => (g, FactKey::RePart),
        Some((g, false)) => (g, FactKey::ImPart),
        None => (f, FactKey::Sign),
    };
    if let Some(sign) = sign_fact(arena, key, fact) {
        return sign;
    }
    if !crate::base::walk::free_symbols(arena, f).is_empty() {
        return known_sign(arena, f);
    }
    if let Some((g, re)) = part {
        use crate::base::assumptions::{AssumptionCache, Props};
        let mut cache = AssumptionCache::new();
        let decided = cache.query(arena, f, Props::POSITIVE) == Some(true)
            || cache.query(arena, f, Props::NEGATIVE) == Some(true);
        // A part the shared value does not settle is evaluated on its own
        // (an imaginary part far below the real one).
        if !decided && let Some(signs) = arena.part_signs(g) {
            if signs.re.is_some() {
                remember_sign(arena, g, FactKey::RePart, signs.re);
            }
            if signs.im.is_some() {
                remember_sign(arena, g, FactKey::ImPart, signs.im);
            }
            let sign = if re { signs.re } else { signs.im };
            if sign.is_some() {
                return sign;
            }
        }
    }
    let sign = known_sign(arena, f);
    remember_sign(arena, key, fact, sign);
    sign
}

/// The sign of a real factor, when it is known: from the assumption engine
/// (`π`, a positive symbol, `exp` of a real), else, for a real constant, from
/// a certified numeric value (`ln 2`, `√2 − 1`); `None` for a factor that
/// may be zero, negative or complex.
fn known_sign(arena: &Arena, f: ExprId) -> Option<Ordering> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let mut cache = AssumptionCache::new();
    if cache.query(arena, f, Props::POSITIVE) == Some(true) {
        return Some(Ordering::Greater);
    }
    if cache.query(arena, f, Props::NEGATIVE) == Some(true) {
        return Some(Ordering::Less);
    }
    if cache.query(arena, f, Props::REAL) != Some(true) {
        return None;
    }
    arena.sign_of_real_constant(f)
}

/// Helper for `Mul` when `zoo` (ComplexInfinity) is encountered.
///
/// `zoo * nonzero → zoo`, `zoo * 0 → NaN`.  `remaining` are the factors
/// not yet processed, `bases` those already collected (base → exponents).
///
/// A factor that vanishes identically ([`factor_vanishes`]) is a zero
/// factor too: `(x·(x + 1) − x² − x)/0` is `0/0 = nan`, and so
/// is `|x·(x + 1) − x² − x|·zoo`.  Up to 0.32 they were `zoo` — the generic
/// `x·zoo = zoo` applied to a factor that is 0 for every value of `x`
/// (SymPy keeps `zoo*(-x**2 + x*(x + 1) - x)`, which is `nan` at every
/// point).  So is a negative power of a sum that is `zoo` (or undefined)
/// at every point by a term over such a denominator ([`everywhere`]):
/// `zoo/(y/((x + y)² − x² − 2xy − y²) + 2)` is `zoo·0`, and a factor
/// undefined at every point.  Only on this path (a product with a literal
/// `zoo`), so ordinary products pay nothing.
///
/// A factor that may vanish on a region of its symbols
/// ([`may_vanish_on_a_region`]: `0^y`, `H(y)`, `⌊y⌋`, a delta) is not a
/// generic nonzero value: the product is `zoo` where each such factor is
/// nonzero ([`push_nonzero_conditions`]) and `nan` elsewhere, `zoo·0^y` is
/// `Piecewise((zoo, (0 > re(y)) | Eq(y, 0)), (nan, True))`.  Up to 0.58 it
/// was `zoo` (`0·zoo` on the half plane `re y > 0`).
fn handle_mul_with_zoo(
    arena: &mut Arena,
    remaining: &mut SmallVec<[ExprId; 16]>,
    bases: &FxHashMap<ExprId, SmallVec<[ExprId; 4]>>,
    coeff: &Q,
) -> ExprId {
    if coeff.is_zero() {
        return arena.nan;
    }
    let mut factors: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut denominators: SmallVec<[ExprId; 4]> = SmallVec::new();
    // The exponents of the factors `0^e`.
    let mut zero_exponents: SmallVec<[ExprId; 2]> = SmallVec::new();
    for (&b, exps) in bases {
        if arena.is_zero_structural(b) {
            zero_exponents.extend(exps.iter().copied());
            continue;
        }
        let numeric: SmallVec<[&Q; 4]> = exps.iter().filter_map(|&e| arena.as_num(e)).collect();
        if numeric.iter().any(|q| q.is_positive()) {
            factors.push(b);
        }
        if numeric.iter().any(|q| q.is_negative()) {
            denominators.push(b);
        }
    }
    // Drain remaining to check for zeros or NaN.
    while let Some(id) = remaining.pop() {
        match arena.node(id).clone() {
            ExprNode::NaN => return arena.nan,
            ExprNode::Num(nid) if arena.num(nid).is_zero() => {
                return arena.nan;
            }
            ExprNode::Mul(children) => {
                remaining.extend_from_slice(&children);
            }
            ExprNode::Neg(inner) => remaining.push(inner),
            ExprNode::Num(_) => {}
            ExprNode::Pow(base, e) if arena.as_num(e).is_some_and(Signed::is_negative) => {
                denominators.push(base);
            }
            ExprNode::Pow(base, e) if arena.is_zero_structural(base) => zero_exponents.push(e),
            _ => factors.push(id),
        }
    }
    let mut memo = EverywhereMemo::default();
    let mut conditions: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut piecewise: SmallVec<[(ExprId, bool); 2]> = SmallVec::new();
    for f in factors {
        // A sum undefined at every point (`0/d + 1/d` with `d` vanishing)
        // makes `zoo·nan = nan` too.
        if memo.zero_or_undefined(arena, f) {
            tracing::debug!("canon_mul: zoo times an identically zero or undefined factor → nan");
            return arena.nan;
        }
        if matches!(arena.node(f), ExprNode::Piecewise(_)) {
            piecewise.push((f, true));
        } else {
            push_nonzero_conditions(arena, f, &mut conditions);
        }
    }
    for d in denominators {
        if memo.infinite_or_undefined(arena, d) {
            tracing::debug!("canon_mul: zoo over an identically infinite sum → nan");
            return arena.nan;
        }
        // A denominator infinite on a region (`0^y − 1` for `re y < 0`):
        // `zoo/d` is `zoo·0` there.
        if holds_zero_power(arena, d) {
            let reciprocal = canon_pow(arena, d, arena.neg_one);
            push_nonzero_conditions(arena, reciprocal, &mut conditions);
        }
    }
    for e in zero_exponents {
        let condition = zero_power_is_nonzero(arena, e);
        conditions.push(condition);
    }
    let zoo = arena.intern(ExprNode::ComplexInfinity);
    let value = where_defined(arena, zoo, conditions);
    times_piecewise(arena, zoo, value, &piecewise)
}

// ═══════════════════════════════════════════════════════════════════════════
// Pow
// ═══════════════════════════════════════════════════════════════════════════

/// Build a canonical `Pow` node.
///
/// Applies simple algebraic identities and, when both base and exponent
/// are numeric with a small-enough integer exponent, evaluates the result.
pub(crate) fn canon_pow(arena: &mut Arena, base: ExprId, exp: ExprId) -> ExprId {
    // Pow(E, x) → Exp(x) — canonicalize e^x to exp(x)
    if base == arena.e_const {
        return arena.intern(ExprNode::Exp(exp));
    }

    // Pow(Pow(a, b), c) → Pow(a, b*c) when c is an integer and b a number.
    //
    // Valid for every complex a and b on the principal branch: z^c is
    // single-valued for integer c, and Log(a^b) = b·Log a + 2πik, so
    // (a^b)^c = exp(c·b·Log a)·exp(2πikc) = a^(bc).  SymPy's `Pow._eval_power`
    // applies the same rule (`s = 1` for an integer exponent).  Critical for
    // the Gruntz algorithm (1/(1/x) = Pow(Pow(x,-1),-1) → x) and for
    // (√y)² → y: before 0.25 only integer b or a positive rational a were
    // flattened, and u-substitution kept wrapping `sqrt((sqrt(c))²)` around
    // `∫ x^(3/2)/(sin(−1) − 2x) dx`, a new integrand every round (1.3 s,
    // 850 000 nodes, to give up).
    if let ExprNode::Pow(inner_base, inner_exp) = arena.node(base).clone()
        && let (Some(b), Some(c)) = (arena.as_num(inner_exp), arena.as_num(exp))
        && c.is_integer()
    {
        tracing::trace!("canon_pow: flattening Pow(Pow(a,b),c) → Pow(a,b*c)");
        let product = q_mul(b, c);
        let prod_id = arena.intern_num(product);
        let prod_expr = arena.intern(ExprNode::Num(prod_id));
        return canon_pow(arena, inner_base, prod_expr);
    }

    // (0^e)^q → 0^(q·e) for a rational q ≠ 0: an identity at every `e` (`0^e`
    // is 0, 1, `zoo` or `nan` by the sign of `re e`, and `q` keeps or swaps
    // the first and the third), so that the powers of 0 have one form —
    // `1/0^y` is `0^(−y)` — and a product sees all of them under the base 0
    // ([`push_zero_power_factors`]).  Up to 0.58 `0^y·(0^y)⁻¹` stayed, and
    // `simplify` cancelled it to 1 (`0/0` or `zoo/zoo` for every `y ≠ 0`).
    if let ExprNode::Pow(inner_base, inner_exp) = *arena.node(base)
        && inner_base == arena.zero
        && arena.as_num(exp).is_some_and(|q| !q.is_zero())
    {
        let product = canon_mul(arena, &[exp, inner_exp]);
        return canon_pow(arena, inner_base, product);
    }

    // Pow(Mul(children), n) → Mul(Pow(child_i, n)) when n is integer, |n| ≤ 10.
    //
    // This is the standard algebraic identity (a·b·c)^n = a^n · b^n · c^n,
    // which is unconditionally valid for integer exponents and commutative
    // bases (no branch-cut issues).  The threshold prevents expression swell
    // for very large exponents.
    //
    // Each Pow(child_i, n) recurses through canon_pow, enabling further
    // simplification: e.g., (½·√5)^2 → (½)^2 · (√5)^2 → ¼ · 5 = 5/4.
    //
    // Matches SymPy's Mul._eval_power auto-expansion for integer exponents.
    if let ExprNode::Mul(ref children) = arena.node(base).clone()
        && let Some(exp_r) = arena.as_num(exp)
        && exp_r.is_integer()
    {
        let n_i64: i64 = exp_r.to_integer().try_into().unwrap_or(i64::MAX);
        if n_i64.unsigned_abs() <= 10 && children.len() >= 2 {
            tracing::trace!(
                n = n_i64,
                n_factors = children.len(),
                "canon_pow: distributing integer power over Mul"
            );
            let distributed: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&child| canon_pow(arena, child, exp))
                .collect();
            return canon_mul(arena, &distributed);
        }
        // Above the threshold, still pull a numeric coefficient out:
        // (c·m)^n → c^n · m^n never swells and keeps `(-x)^11` from being
        // stuck as an opaque power (it must be `-x^11` for polynomial
        // bucketing to see a monomial).
        if children.len() >= 2
            && let Some(&first) = children.first()
            && matches!(arena.node(first), ExprNode::Num(_))
            && n_i64 != i64::MAX
        {
            let coeff_pow = canon_pow(arena, first, exp);
            let rest: SmallVec<[ExprId; 6]> = children[1..].iter().copied().collect();
            let rest_id = if rest.len() == 1 {
                rest[0]
            } else {
                canon_mul(arena, &rest)
            };
            let rest_pow = canon_pow(arena, rest_id, exp);
            return canon_mul(arena, &[coeff_pow, rest_pow]);
        }
    }

    // i^n reduction: i^0=1, i^1=i, i^2=-1, i^3=-i, then repeats with period 4.
    if base == arena.i_unit
        && let Some(exp_r) = arena.as_num(exp)
        && exp_r.is_integer()
    {
        use num_integer::Integer;
        let exp_int = exp_r.to_integer();
        let four = BigInt::from(4);
        let remainder = exp_int.mod_floor(&four);
        let r: u32 = remainder.try_into().unwrap_or(0);
        // `mod_floor(4)` is 0..=3.
        return match r {
            0 => arena.one,
            1 => arena.i_unit,
            2 => arena.neg_one,
            _ => arena.neg(arena.i_unit),
        };
    }

    // (-1)^(n/2) → I^n, then reduce via mod-4 arithmetic.
    if base == arena.neg_one
        && let Some(exp_r) = arena.as_num(exp)
    {
        let denom = exp_r.denom();
        let numer = exp_r.numer();
        // Check if denominator is 2 (i.e., exponent is n/2)
        if *denom == BigInt::from(2) {
            // (-1)^(n/2) = I^n, then reduce I^n via mod-4
            let n = numer.clone();
            let four = BigInt::from(4);
            use num_integer::Integer;
            let remainder = n.mod_floor(&four);
            let r: u32 = (&remainder).try_into().unwrap_or(0);
            // `mod_floor(4)` is 0..=3.
            return match r {
                0 => arena.one,
                1 => arena.i_unit,
                2 => arena.neg_one,
                _ => arena.neg(arena.i_unit),
            };
        }
        // (−1)^(i + r/d) = (−1)^i·(−1)^(r/d) with 0 < r < d: the
        // exponent comes into (0, 1), as SymPy's `NegativeOne._eval_power`
        // writes it (`(-1)**(59/3)` → `-(-1)**(2/3)`, `(-1)**(-1/3)` →
        // `-(-1)**(2/3)`).  Exact on the principal branch, `(−1)^q =
        // e^{iπq}` for every rational `q`.  Up to 0.44 the exponent stayed,
        // and equal values had different forms (`(−1)^(−59/3)`, `(−1)^(1/3)`).
        if !exp_r.is_integer() {
            use num_integer::Integer;
            let denom = denom.clone();
            let (whole, rest) = numer.div_mod_floor(&denom);
            if !whole.is_zero() {
                let nid = arena.intern_num(Ratio::new(rest, denom));
                let reduced = arena.intern(ExprNode::Num(nid));
                let unit = canon_pow(arena, base, reduced);
                return if whole.is_odd() {
                    canon_neg(arena, unit)
                } else {
                    unit
                };
            }
        }
    }

    // i^(2k + r) = (−1)^k·i^r with 0 < r < 2 for a non-integer rational
    // exponent, as SymPy's `ImaginaryUnit._eval_power` (`I**(7/3)` →
    // `-I**(1/3)`): `i^q = e^{iπq/2}` on the principal branch.
    if base == arena.i_unit
        && let Some(exp_r) = arena.as_num(exp)
        && !exp_r.is_integer()
    {
        use num_integer::Integer;
        let denom = exp_r.denom().clone();
        let two_denom = &denom * BigInt::from(2);
        let (half_turns, rest) = exp_r.numer().div_mod_floor(&two_denom);
        if !half_turns.is_zero() {
            let nid = arena.intern_num(Ratio::new(rest, denom));
            let reduced = arena.intern(ExprNode::Num(nid));
            let unit = canon_pow(arena, base, reduced);
            return if half_turns.is_odd() {
                canon_neg(arena, unit)
            } else {
                unit
            };
        }
    }

    // (-n)^(1/2) → i * sqrt(n) for negative numeric n
    // More generally, negative_rational^(1/2) → i * |negative_rational|^(1/2)
    if let Some(base_r) = arena.as_num(base)
        && base_r.is_negative()
        && let Some(exp_r) = arena.as_num(exp)
        && *exp_r == Ratio::new(1.into(), 2.into())
    {
        // base is negative, exp is 1/2
        // result = i * |base|^(1/2)
        let abs_base = {
            let abs_val = -base_r.clone();
            let nid = arena.intern_num(abs_val);
            arena.intern(ExprNode::Num(nid))
        };
        let sqrt_abs = canon_pow(arena, abs_base, exp);
        return arena.mul(&[arena.i_unit, sqrt_abs]);
    }

    // NaN propagation.
    let (base_class, exp_class) = (
        ValueClass::special_atom(arena, base),
        ValueClass::special_atom(arena, exp),
    );
    if base_class == Some(ValueClass::NaN) || exp_class == Some(ValueClass::NaN) {
        return arena.nan;
    }

    // A sum with one infinite term beside finite ones (`∞ + i`) has an
    // infinite magnitude: its negative powers are 0 (SymPy: `1/(oo + I)` →
    // 0), its zeroth power is `nan` like `∞⁰`.  So `B·B⁻¹` is `B·0 = nan`,
    // never the `1` of a finite base.
    if let Some(e) = arena.as_num(exp)
        && !e.is_positive()
        && is_infinite_sum(arena, base)
    {
        return if e.is_zero() { arena.nan } else { arena.zero };
    }

    // x^0 → 1, except for indeterminate forms ∞^0, (-∞)^0, zoo^0.
    if exp == arena.zero {
        return match base_class {
            Some(_) => arena.nan,
            None => arena.one,
        };
    }

    // x^1 → x.
    if exp == arena.one {
        return base;
    }

    // 1^x → 1, also for `x = ±∞, zoo` (the exact power of the number 1, as
    // IEEE `pow(1, ∞)`; the limit form `1^∞` is the business of `limit`).
    if base == arena.one {
        return arena.one;
    }

    // 0^q for a numeric q ≠ 0, 0^±∞, 0^zoo, and 0^z for a non-numeric
    // exponent whose real part has a known sign.
    if base == arena.zero {
        if let Some(e) = arena.as_num(exp) {
            if e.is_positive() {
                return arena.zero;
            }
            // (division by zero)
            if e.is_negative() {
                return arena.complex_infinity;
            }
        }
        match exp_class {
            Some(ValueClass::PosInf) => return arena.zero,
            Some(ValueClass::NegInf) => return arena.complex_infinity,
            // (indeterminate)
            Some(ValueClass::ComplexInf) => return arena.nan,
            _ => {}
        }
        if let Some(value) = zero_power(arena, exp) {
            return value;
        }
    }

    // The powers of `∞`, `−∞`, `zoo`.
    if let Some(class) = base_class
        && let Some(value) = special_atom_power(arena, class, exp, exp_class)
    {
        return value;
    }

    // Both numeric → try to evaluate.
    if let (Some(b), Some(e)) = (arena.as_num(base).cloned(), arena.as_num(exp).cloned())
        && let Some(result) = eval_numeric_pow(arena, &b, &e)
    {
        return result;
    }

    // Radical normal form for positive rational bases with fractional
    // exponents (see [`canon_radical`]): `√12 → 2√3`, `√(4/9) → 2/3`,
    // `√(1/2) = 2^(-1/2) = √2/2`, `∛54 → 3∛2`.
    if let (Some(base_r), Some(exp_r)) = (arena.as_num(base).cloned(), arena.as_num(exp).cloned())
        && base_r.is_positive()
        && !exp_r.is_integer()
        && let Some(result) = canon_radical(arena, base, &base_r, exp, &exp_r)
    {
        return result;
    }

    // Values under the assumptions on symbols (`(−1)^m = −1` for an odd
    // `m`, `√(p²) = p`, `|r|² = r²`; see [`assumed_power`]).
    if let Some(value) = assumed_power(arena, base, exp) {
        return value;
    }

    let result = arena.intern(ExprNode::Pow(base, exp));
    #[cfg(debug_assertions)]
    {
        let errors = verify_canonical_shallow(arena, result);
        if !errors.is_empty() {
            tracing::debug!("canon_pow: non-canonical result: {:?}", errors);
        }
    }
    result
}

/// `0^z` for a non-numeric exponent `z`: `0` for a real `z > 0`, `zoo` for a
/// real `z < 0` (the limits along every path), and `nan` for a non-real `z`
/// (`0^z = e^(z·ln 0)` has no value), as SymPy (`0**pi`, `0**(-pi)`,
/// `0**I`, `0**(1+I)`) and mpmath (`mpc(0)**mpc(1, 1)` is `nan`) define it.
/// `None` when the exponent's sign or realness is not known (`0^x` for an
/// unknown `x` stays).  Before 0.30 `0^I`, `0^π` and `0^(−1 + i)` stayed
/// as atoms, and `evalf` gave `0` for the first and the last.
/// `b^e` for a special atom `b` (`∞`, `−∞`, `zoo`; a `nan` is propagated
/// before): by the sign of a rational `e`, `∞^e` is `∞` or 0, `zoo^e` is
/// `zoo` or 0, `(−∞)^e` is 0 or `∞` in the direction `(−1)^e`; `∞^∞ = ∞`,
/// `∞^(−∞) = 0`; an infinite exponent of `−∞` or `zoo`, and a `zoo`
/// exponent of any infinity, is `nan` (`e^{e·ln b}` has no limit; up to
/// 0.57 `∞^zoo` and `(−∞)^zoo` stayed).  `None` for any other exponent.
fn special_atom_power(
    arena: &mut Arena,
    base: ValueClass,
    exp: ExprId,
    exp_class: Option<ValueClass>,
) -> Option<ExprId> {
    let e = arena.as_num(exp).cloned();
    let positive = e.as_ref().is_some_and(Signed::is_positive);
    let negative = e.as_ref().is_some_and(Signed::is_negative);
    match base {
        ValueClass::PosInf => match exp_class {
            _ if positive => Some(arena.infinity),
            _ if negative => Some(arena.zero),
            Some(ValueClass::PosInf) => Some(arena.infinity),
            Some(ValueClass::NegInf) => Some(arena.zero),
            Some(ValueClass::ComplexInf) => Some(arena.nan),
            _ => None,
        },
        ValueClass::NegInf => match exp_class {
            // (−∞)^n → ±∞ by the parity of a positive integer n.
            _ if positive && e.as_ref().is_some_and(Ratio::is_integer) => {
                Some(if num_integer::Integer::is_even(&e?.to_integer()) {
                    arena.infinity
                } else {
                    arena.neg_infinity
                })
            }
            // (−∞)^q for a positive non-integer rational q: ∞ in the
            // direction of (−1)^q on the principal branch, `√(−∞) = i·∞`,
            // `(−∞)^(1/3) = (−1)^(1/3)·∞` (SymPy: `sqrt(-oo)` → `oo*I`,
            // `(-oo)**Rational(1, 3)` → `oo*(-1)**(1/3)`).  Before 0.30 the
            // power stayed an atom.
            _ if positive => {
                let unit = canon_pow(arena, arena.neg_one, exp);
                Some(canon_mul(arena, &[unit, arena.infinity]))
            }
            // (magnitude → 0 regardless of direction)
            _ if negative => Some(arena.zero),
            // (indeterminate: ∞ is not an integer; `zoo` has no direction)
            Some(ValueClass::PosInf | ValueClass::NegInf | ValueClass::ComplexInf) => {
                Some(arena.nan)
            }
            _ => None,
        },
        ValueClass::ComplexInf => match exp_class {
            _ if positive => Some(arena.complex_infinity),
            _ if negative => Some(arena.zero),
            Some(ValueClass::PosInf | ValueClass::NegInf | ValueClass::ComplexInf) => {
                Some(arena.nan)
            }
            _ => None,
        },
        ValueClass::Finite
        | ValueClass::DirectedInf
        | ValueClass::NaN
        | ValueClass::Infinite
        | ValueClass::Unknown => None,
    }
}

fn zero_power(arena: &mut Arena, exp: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    if arena.as_num(exp).is_some() {
        return None;
    }
    // The real part: the exponent without its imaginary-multiple terms.
    let terms: SmallVec<[ExprId; 6]> = match arena.node(exp) {
        ExprNode::Add(t) => t.clone(),
        _ => smallvec![exp],
    };
    let is_imaginary_multiple =
        |id: ExprId| matches!(classify_arg(arena, id), Arg::ImaginaryMultiple(_));
    let real: SmallVec<[ExprId; 6]> = terms
        .iter()
        .copied()
        .filter(|&t| !is_imaginary_multiple(t))
        .collect();
    let has_imaginary = real.len() < terms.len();
    let re = canon_add(arena, &real);
    let mut cache = AssumptionCache::new();
    if re == arena.zero || cache.query(arena, re, Props::REAL) != Some(true) {
        // `re + q·i` with a real `re` and a non-zero `q` (canonical terms
        // have non-zero coefficients) is not real.
        return (has_imaginary && re == arena.zero).then_some(arena.nan);
    }
    if has_imaginary {
        return Some(arena.nan);
    }
    if cache.query(arena, re, Props::POSITIVE) == Some(true) {
        Some(arena.zero)
    } else if cache.query(arena, re, Props::NEGATIVE) == Some(true) {
        Some(arena.complex_infinity)
    } else {
        None
    }
}

/// Try to evaluate `b ^ e` when both are rational, returning `None` if the
/// exponent is not a suitably small integer.
fn eval_numeric_pow(arena: &mut Arena, b: &Q, e: &Q) -> Option<ExprId> {
    // Only evaluate when the exponent is an integer.
    if !e.is_integer() {
        return None;
    }

    let exp_int: BigInt = e.to_integer();

    // `(±1)^n` costs nothing whatever `n`: `1` or `(−1)^(n mod 2)` (SymPy
    // folds `(−1)**430587161543285117552609` to `−1`).  Before 0.37 an
    // exponent over the guard below left it a power, and
    // `(−∞)·(−1)^430587161543285117552609` stayed a product instead of `∞`.
    if b.is_one() {
        return Some(arena.one);
    }
    if *b == -Q::one() {
        let odd = num_integer::Integer::is_odd(&exp_int);
        return Some(if odd { arena.neg_one } else { arena.one });
    }

    // Guard: don't evaluate if exponent is too large.
    let max_exp = arena.config.max_pow_exponent as u64;
    let abs_exp: BigInt = exp_int.abs();
    if abs_exp > BigInt::from(max_exp) {
        return None;
    }

    // b == 0 is handled by the caller.
    if b.is_zero() {
        return None;
    }

    let exp_u32: u32 = match abs_exp.try_into() {
        Ok(v) => v,
        Err(_) => return None,
    };

    // A power the digit guard below would reject is not computed.  `b` is
    // in lowest terms, so `numerⁿ/denomⁿ` is too and the guard counts
    // exactly their digits (plus a sign); an integer of `k ≥ 1` bits has at
    // least `⌊(k − 1)·log₁₀ 2⌋ + 1` digits, and `xⁿ` at least
    // `n·(k − 1) + 1` bits.  So this lower bound exceeding the limit implies
    // rejection, and the output is unchanged.  (Computing
    // `(999999999999999/10¹⁵)¹⁰⁰⁰` — two 50,000-bit integers, their gcd
    // and their decimal strings — only to reject it took 0.55 s in a debug
    // build, at every construction of the power.)
    // `b^(±1)` has exactly the digits of `b`: exempt, since a rational
    // already beyond the guard (from a literal, or a product of numbers)
    // must still invert (0.29 kept `Pow(q, −1)` for such a `q`, a form a
    // re-parse of its display folded; found by `fuzz_roundtrip`).
    if exp_u32 != 1
        && min_pow_digits(b.numer(), exp_u32) + min_pow_digits(b.denom(), exp_u32)
            > arena.config.max_result_digits as f64 + 2.0
    {
        return None;
    }

    // Compute |exp| power of numerator and denominator.
    let numer: BigInt = b.numer().clone();
    let denom: BigInt = b.denom().clone();

    let pow_n: BigInt = NumPow::pow(numer, exp_u32);
    let pow_d: BigInt = NumPow::pow(denom, exp_u32);

    // `numerⁿ/denomⁿ` is in lowest terms already (`b` is): no gcd, which
    // `Ratio::new` computes in time quadratic in the digits even when
    // `denomⁿ = 1` (see `numeric::q_add`).
    let result = if exp_int.is_negative() {
        // b^(-n) = (denom^n) / (numer^n)
        if pow_n.is_zero() {
            // Would be division by zero → leave unevaluated.
            return None;
        }
        if pow_n.is_negative() {
            Ratio::new_raw(-pow_d, -pow_n)
        } else {
            Ratio::new_raw(pow_d, pow_n)
        }
    } else {
        Ratio::new_raw(pow_n, pow_d)
    };

    // Check the digit count doesn't exceed the guard.
    if exp_u32 != 1 {
        let digit_count = result.numer().to_string().len() + result.denom().to_string().len();
        if digit_count > arena.config.max_result_digits {
            return None;
        }
    }

    let nid = arena.intern_num(result);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// A lower bound on the decimal digits of `xⁿ`: an integer of `k ≥ 1` bits
/// has at least `⌊(k − 1)·log₁₀ 2⌋ + 1` digits, and `xⁿ` at least
/// `n·(k − 1) + 1` bits.
fn min_pow_digits(x: &BigInt, n: u32) -> f64 {
    let bits = x.bits().saturating_sub(1) as f64 * f64::from(n);
    (bits * std::f64::consts::LOG10_2).floor() + 1.0
}

/// Would the integer power `xⁿ` certainly exceed the digit guard
/// (`max_result_digits`, as [`eval_numeric_pow`] applies it)?
fn pow_exceeds_digit_guard(arena: &Arena, x: &BigInt, n: u32) -> bool {
    min_pow_digits(x, n) > arena.config.max_result_digits as f64 + 1.0
}

/// Normal form of `r^(a/b)` for a positive rational `r` and a non-integer
/// rational exponent `a/b` (`b > 1`), or `None` when nothing changes.
///
/// The canonical form has an **integer base ≥ 2**, a **positive** exponent
/// in `(0, ∞)` and a **radicand free of `b`-th power factors** (as far as
/// the bounded factorisation can tell):
///
/// * `(p/q)^(a/b) → p^(a/b) · q^(-a/b)` — the rational base is split;
/// * `n^(-a/b) → n^(-k-1) · n^((b-s)/b)` with `a = k·b + s`, `0 < s < b` —
///   the denominator is rationalised (`2^(-1/2) = √2/2`), matching SymPy;
/// * `n^(a/b) → outside^a · inside^(a/b)` with `n = outside^b · inside`
///   (`√12 = 2√3`, `√(4/9) = 2/3`, `∛54 = 3∛2`).
///
/// Consequently `√(1/2)`, `1/√2`, `2^(-1/2)` and `√2/2` all canonicalise
/// to the same `Mul(1/2, Pow(2, 1/2))`.
fn canon_radical(
    arena: &mut Arena,
    base: ExprId,
    base_r: &Q,
    exp: ExprId,
    exp_r: &Q,
) -> Option<ExprId> {
    debug_assert!(base_r.is_positive() && !exp_r.is_integer());

    // (p/q)^(a/b) → p^(a/b) · q^(-a/b).
    if !base_r.is_integer() {
        tracing::trace!("canon_radical: splitting rational base");
        let p = arena.big_int(base_r.numer().clone());
        let q = arena.big_int(base_r.denom().clone());
        let neg_exp = {
            let nid = arena.intern_num(-exp_r.clone());
            arena.intern(ExprNode::Num(nid))
        };
        let p_pow = canon_pow(arena, p, exp);
        let q_pow = canon_pow(arena, q, neg_exp);
        return Some(arena.mul(&[p_pow, q_pow]));
    }

    // Integer base n ≥ 2 from here on (n = 1 was folded by the caller).
    let n = base_r.to_integer();
    if n <= BigInt::one() {
        return None;
    }

    // n^(-a/b) → n^(-(k+1)) · n^((b-s)/b),  a = k·b + s.
    if exp_r.is_negative() {
        let a = -exp_r.numer().clone();
        let b = exp_r.denom().clone();
        let (k, s) = num_integer::Integer::div_rem(&a, &b);
        // Only when n^(k+1) folds to a number: otherwise both factors stay
        // powers of n, `mul` adds the exponents back to −a/b and this
        // recursed without end (`(37/5)^((387/5)^74)`, found by
        // `fuzz_parser` as a stack overflow).
        if (&k + BigInt::one()) > BigInt::from(arena.config.max_pow_exponent) {
            return None;
        }
        tracing::trace!("canon_radical: rationalising negative fractional exponent");
        let pos_exp = {
            let nid = arena.intern_num(Ratio::new(&b - &s, b.clone()));
            arena.intern(ExprNode::Num(nid))
        };
        let int_exp = {
            let nid = arena.intern_num(Ratio::from_integer(-(k + BigInt::one())));
            arena.intern(ExprNode::Num(nid))
        };
        let coeff = canon_pow(arena, base, int_exp);
        // The exponent bound above is necessary but not sufficient: the
        // digit guard of `eval_numeric_pow` also refuses a power with too
        // many digits (`9991999999^590`, ~5,900 digits), and then the same
        // endless recursion followed (`2/9991999999^589.99919999`, found by
        // `fuzz_parser` as a stack overflow).  Rationalise only when the
        // coefficient did fold.
        arena.as_num(coeff)?;
        let radical = canon_pow(arena, base, pos_exp);
        return Some(arena.mul(&[coeff, radical]));
    }

    // n^(a/b), a > 0: pull perfect b-th powers out of n.
    let b: u32 = exp_r.denom().to_u32()?;
    let a: u32 = exp_r.numer().to_u32()?;
    let (outside, inside) = split_perfect_power(&n, b);
    if outside.is_one() {
        // n^(a/b) → n^k · n^(s/b) with a = k·b + s, 0 < s < b: the radical
        // keeps a proper exponent (`15^(3/2) = 15·√15`, as SymPy writes
        // it).  Before 0.26 `15^(3/2)` and `15·√15` were different
        // canonical forms of one number, so equal coefficients did not
        // combine (and a `polylog` term that should cancel stayed).
        // The integer part must fold within the digit guard, as any power
        // does: `(10¹⁰⁰⁰ + 7)^(1999/2)` built a million-digit `n⁹⁹⁹` here
        // (15.9 s in a release build) before the guard ever saw it.
        if a > b
            && (a / b) <= arena.config.max_pow_exponent as u32
            && !pow_exceeds_digit_guard(arena, &n, a / b)
        {
            let k = a / b;
            let s = a % b;
            let integer_part = arena.big_int(NumPow::pow(n.clone(), k));
            let frac_exp = {
                let nid = arena.intern_num(Ratio::new(BigInt::from(s), BigInt::from(b)));
                arena.intern(ExprNode::Num(nid))
            };
            let radical = arena.intern(ExprNode::Pow(base, frac_exp));
            return Some(arena.mul(&[integer_part, radical]));
        }
        return None;
    }
    // outside^a must fold to a number of bounded size (a can be ~4·10⁹).
    if a > arena.config.max_pow_exponent as u32 || pow_exceeds_digit_guard(arena, &outside, a) {
        return None;
    }
    tracing::trace!("canon_radical: extracted perfect power factor");
    let outside_pow = NumPow::pow(outside, a);
    let outside_expr = arena.big_int(outside_pow);
    if inside.is_one() {
        return Some(outside_expr);
    }
    let inside_base = arena.big_int(inside);
    // `inside` has no b-th power factor, so this only splits off the
    // integer part of the exponent (no further recursion).
    let inside_radical = if a > b {
        canon_pow(arena, inside_base, exp)
    } else {
        arena.intern(ExprNode::Pow(inside_base, exp))
    };
    Some(arena.mul(&[outside_expr, inside_radical]))
}

/// Composite cofactors with more bits than this are not factored during
/// canonicalisation (`√n` is then left with the cofactor inside the
/// radical).  About 25 decimal digits: Pollard rho on such a semiprime is
/// still a few million machine-word steps, which is the most we are willing
/// to spend at construction time.
const RADICAL_FACTOR_MAX_BITS: u64 = 84;

/// Integers with at most this many bits are not memoised in
/// [`RADICAL_PARTS`]: trial division up to `√n < 2¹⁶` on machine words
/// decomposes them in microseconds.
const RADICAL_MEMO_MIN_BITS: u64 = 32;

/// Entries kept in [`RADICAL_PARTS`] before it is emptied and refilled.
const RADICAL_MEMO_CAPACITY: usize = 256;

/// A squarefree decomposition `(parts, cofactor)` as returned by
/// [`crate::domains::ntheory::squarefree_parts_bounded`].
type RadicalParts = (Vec<(BigInt, u32)>, BigInt);

thread_local! {
    /// Squarefree decompositions of the integers whose radicals were
    /// canonicalised on this thread, keyed by the integer.
    ///
    /// A radical is canonicalised again whenever a node holding it is
    /// rebuilt, and `eval` extracts its powers once more: in 0.28.0
    /// `√78243492961199594876179935 .simplify()` decomposed that integer 15
    /// times (~1 s, debug build), `pearson_test` on 11 pairs of 28-bit
    /// integers one 73-bit cofactor 45 times (~0.75 s).  The decomposition is a pure
    /// function of the integer, so a warm entry is exactly what a cold
    /// computation returns and canonical forms do not depend on the memo.
    static RADICAL_PARTS: std::cell::RefCell<FxHashMap<BigInt, std::rc::Rc<RadicalParts>>> =
        std::cell::RefCell::new(FxHashMap::default());

    /// [`split_perfect_power`]`(n, k) = (outside, inside)` of the integers
    /// above [`RADICAL_MEMO_MIN_BITS`] split on this thread, keyed by
    /// `(n, k)`.
    static RADICAL_SPLITS: std::cell::RefCell<RadicalSplits> =
        std::cell::RefCell::new(FxHashMap::default());
}

/// The memo behind [`split_perfect_power`]: `(n, k) → (outside, inside)`.
type RadicalSplits = FxHashMap<(BigInt, u32), std::rc::Rc<(BigInt, BigInt)>>;

/// [`crate::domains::ntheory::squarefree_parts_bounded`] of `n > 1` at
/// [`RADICAL_FACTOR_MAX_BITS`], through the thread's memo.
fn radical_parts(n: &BigInt) -> std::rc::Rc<RadicalParts> {
    let compute = || {
        std::rc::Rc::new(crate::domains::ntheory::squarefree_parts_bounded(
            n,
            RADICAL_FACTOR_MAX_BITS,
        ))
    };
    if n.bits() <= RADICAL_MEMO_MIN_BITS {
        return compute();
    }
    let cached = RADICAL_PARTS
        .try_with(|memo| memo.try_borrow().ok().and_then(|m| m.get(n).cloned()))
        .ok()
        .flatten();
    if let Some(parts) = cached {
        return parts;
    }
    let parts = compute();
    // Best effort: a memo that cannot be reached (thread teardown) is skipped.
    let _ = RADICAL_PARTS.try_with(|memo| {
        if let Ok(mut memo) = memo.try_borrow_mut() {
            if memo.len() >= RADICAL_MEMO_CAPACITY {
                memo.clear();
            }
            memo.insert(n.clone(), std::rc::Rc::clone(&parts));
        }
    });
    parts
}

/// Split a positive integer as `n = outside^k · inside` where `inside` has no
/// prime factor with exponent `≥ k` among the factors found by the bounded
/// factorisation ([`crate::domains::ntheory::factorint_bounded`]).
///
/// The factorisation itself is not needed, only the exponents: this reads
/// them from the squarefree decomposition
/// ([`crate::domains::ntheory::squarefree_parts_bounded`]), which carries
/// the same exponents for the same primes, memoised per thread.  (Over
/// 2,048 bits the base `b` of a perfect-power remainder `bᵉ` counts with
/// its exponent whether it is prime or not, as SymPy's `factorint` with a
/// limit lists it: `√(3·(p·q)⁴) = (p·q)²·√3`.)
///
/// Exact `k`-th powers are recognised first via an integer root, so
/// `√(p²)` folds even for huge primes `p`; residues rule most non-powers
/// out before the root is taken.  The split is memoised per thread too.
///
/// # Examples
///
/// - `split_perfect_power(12, 2)` → `(2, 3)` because 12 = 2²·3
/// - `split_perfect_power(8, 3)`  → `(2, 1)` because 8 = 2³
/// - `split_perfect_power(7, 2)`  → `(1, 7)` (7 is square-free)
pub(crate) fn split_perfect_power(n: &BigInt, k: u32) -> (BigInt, BigInt) {
    if k < 2 || *n <= BigInt::one() {
        return (BigInt::one(), n.clone());
    }
    if n.bits() <= RADICAL_MEMO_MIN_BITS {
        return split_perfect_power_uncached(n, k);
    }
    // A radical is split again whenever a product holding it is rebuilt
    // (`canon_mul` re-canonicalises every power): memoised per thread like
    // its decomposition, a pure function of `(n, k)`.
    let key = (n.clone(), k);
    let cached = RADICAL_SPLITS
        .try_with(|memo| memo.try_borrow().ok().and_then(|m| m.get(&key).cloned()))
        .ok()
        .flatten();
    if let Some(split) = cached {
        return (*split).clone();
    }
    let split = std::rc::Rc::new(split_perfect_power_uncached(n, k));
    let _ = RADICAL_SPLITS.try_with(|memo| {
        if let Ok(mut memo) = memo.try_borrow_mut() {
            if memo.len() >= RADICAL_MEMO_CAPACITY {
                memo.clear();
            }
            memo.insert(key, std::rc::Rc::clone(&split));
        }
    });
    (*split).clone()
}

/// [`split_perfect_power`] without the memo.
fn split_perfect_power_uncached(n: &BigInt, k: u32) -> (BigInt, BigInt) {
    // The exact root is a Newton iteration of full-size divisions
    // (quadratic in the digits): skipped when residues rule a `k`-th power
    // out, as they do for all but a few non-powers.
    if crate::domains::ntheory::may_be_kth_power(n, k) {
        let root = n.nth_root(k);
        if NumPow::pow(root.clone(), k) == *n {
            return (root, BigInt::one());
        }
    }
    let parts = radical_parts(n);
    let (pieces, cofactor) = &*parts;
    let mut outside = BigInt::one();
    let mut inside = cofactor.clone();
    for (d, e) in pieces {
        if *e >= k {
            outside *= NumPow::pow(d.clone(), e / k);
        }
        if e % k > 0 {
            inside *= NumPow::pow(d.clone(), e % k);
        }
    }
    (outside, inside)
}

// ═══════════════════════════════════════════════════════════════════════════
// Neg
// ═══════════════════════════════════════════════════════════════════════════

/// Build a canonical negation.
///
/// Normalises `Neg` away wherever possible — the canonical form of a
/// negated expression is typically `Mul(−1, expr)` or a folded numeric
/// literal.
pub(crate) fn canon_neg(arena: &mut Arena, expr: ExprId) -> ExprId {
    let result = match arena.node(expr).clone() {
        // −(−x) → x (double negation).
        ExprNode::Neg(inner) => inner,

        // −(Num(n)) → Num(−n).
        ExprNode::Num(nid) => {
            let val = arena.num(nid).clone();
            let neg_val = -val;
            let neg_nid = arena.intern_num(neg_val);
            arena.intern(ExprNode::Num(neg_nid))
        }

        // −NaN → NaN.
        ExprNode::NaN => arena.nan,

        // −∞ → −∞, −(−∞) → ∞
        ExprNode::Infinity => arena.neg_infinity,
        ExprNode::NegInfinity => arena.infinity,

        // −zoo → zoo.
        ExprNode::ComplexInfinity => arena.intern(ExprNode::ComplexInfinity),

        // −(a + b + …) → (−a) + (−b) + …  (distribute negation).
        ExprNode::Add(children) => {
            let negated: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&child| canon_neg(arena, child))
                .collect();
            canon_add(arena, &negated)
        }

        // −(c * a * b * …) where c is numeric → (−c) * a * b * …
        ExprNode::Mul(ref children) if !children.is_empty() => {
            if let ExprNode::Num(nid) = arena.node(children[0]) {
                let nid = *nid;
                let val = arena.num(nid).clone();
                let neg_val = -val;
                let neg_nid = arena.intern_num(neg_val);
                let neg_coeff = arena.intern(ExprNode::Num(neg_nid));
                let mut new_args: SmallVec<[ExprId; 6]> = smallvec![neg_coeff];
                new_args.extend_from_slice(&children[1..]);
                // Re-canonicalise in case the negated coefficient is 1 or 0.
                canon_mul(arena, &new_args)
            } else {
                // No numeric leading factor — prepend −1.
                let neg_one = arena.neg_one;
                let mut all: SmallVec<[ExprId; 6]> = smallvec![neg_one];
                let children = children.clone();
                all.extend_from_slice(&children);
                canon_mul(arena, &all)
            }
        }

        // General case: represent as Mul(−1, expr).
        _ => {
            let neg_one = arena.neg_one;
            canon_mul(arena, &[neg_one, expr])
        }
    };
    #[cfg(debug_assertions)]
    {
        let errors = verify_canonical_shallow(arena, result);
        if !errors.is_empty() {
            tracing::debug!("canon_neg: non-canonical result: {:?}", errors);
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    // ── helpers ─────────────────────────────────────────────────────────
    fn s(arena: &mut Arena, name: &str) -> ExprId {
        arena.symbol(name)
    }
    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    /// `application_arguments` looks into exactly the 34 node-backed
    /// functions its 0.49 or-pattern listed (W6c: the list became the
    /// `looked_into` predicate).
    #[test]
    fn looked_into_is_the_old_list() {
        use crate::base::func::{Family, Func};
        use std::collections::HashSet;
        let old: HashSet<Func> = [
            Func::Floor,
            Func::Ceiling,
            Func::Sin,
            Func::Cos,
            Func::Tan,
            Func::Exp,
            Func::Ln,
            Func::Abs,
            Func::Asin,
            Func::Acos,
            Func::Atan,
            Func::Sinh,
            Func::Cosh,
            Func::Tanh,
            Func::Asinh,
            Func::Acosh,
            Func::Atanh,
            Func::Sign,
            Func::Gamma,
            Func::LogGamma,
            Func::Digamma,
            Func::Erf,
            Func::Erfc,
            Func::LambertW0,
            Func::Si,
            Func::Ci,
            Func::Ei,
            Func::Li,
            Func::Zeta,
            Func::Factorial,
            Func::Atan2,
            Func::Beta,
            Func::Polygamma,
            Func::Binomial,
        ]
        .into_iter()
        .collect();
        let new: HashSet<Func> = Func::ALL
            .iter()
            .copied()
            .filter(|&f| f.family() == Family::Node && looked_into(f))
            .collect();
        assert_eq!(new, old);
    }

    /// `not_finite_at_infinity`, `zero_candidate_sums` and
    /// `sum_under_zero_chain` select exactly the functions of their 0.49
    /// or-patterns (W6d: the lists became `examined_at_infinity` and
    /// `zero_through`), and since 0.54 `ln Γ`, `ψ` and `ζ` at infinity.
    #[test]
    fn examined_at_infinity_and_zero_through_are_the_old_lists() {
        use crate::base::func::{Func::*, selected};
        let old_infinity = [
            Exp, Cosh, Sinh, Sin, Cos, Tan, Tanh, Abs, Ln, Asin, Acos, Asinh, Acosh, Floor,
            Ceiling, Erf, Erfc, Gamma, Factorial, LogGamma, Digamma, Zeta,
        ];
        assert_eq!(
            selected(examined_at_infinity),
            old_infinity.into_iter().collect()
        );
        let old_zero = [
            Sin, Tan, Sinh, Tanh, Asin, Atan, Asinh, Atanh, Abs, Sign, LambertW0, Floor, Ceiling,
        ];
        assert_eq!(selected(zero_through), old_zero.into_iter().collect());
    }

    // ── Add canonicalization ────────────────────────────────────────────

    #[test]
    fn add_combines_like_terms() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let three = a.int(3);
        let three_x = a.mul(&[three, x]);
        let result = a.add(&[two_x, three_x]);
        assert_eq!(display(&a, result), "5*x");
    }

    #[test]
    fn add_combines_identical_symbols() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let result = a.add(&[x, x]);
        assert_eq!(display(&a, result), "2*x");
    }

    #[test]
    fn add_flattens_nested_add() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let y = s(&mut a, "y");
        let z = s(&mut a, "z");
        let inner = a.intern(ExprNode::Add(smallvec![x, y]));
        let result = a.add(&[inner, z]);
        // Should be a flat x + y + z, not (x+y) + z.
        assert_eq!(display(&a, result), "x + y + z");
    }

    #[test]
    fn add_drops_zeros() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let zero = a.zero;
        let result = a.add(&[x, zero]);
        assert_eq!(result, x);
    }

    #[test]
    fn add_evaluates_numeric_sum() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let result = a.add(&[two, three]);
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn add_nan_propagates() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let nan = a.nan;
        let result = a.add(&[x, nan]);
        assert_eq!(result, a.nan);
    }

    #[test]
    fn add_cancellation_to_zero() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let neg_x = a.neg(x);
        let result = a.add(&[x, neg_x]);
        assert_eq!(result, a.zero);
    }

    #[test]
    fn add_numeric_and_symbolic() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let three = a.int(3);
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let result = a.add(&[x, two_x, three]);
        assert_eq!(display(&a, result), "3*x + 3");
    }

    #[test]
    fn add_multiple_like_terms() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let ab = {
            let aa = s(&mut a, "a");
            let bb = s(&mut a, "b");
            a.mul(&[aa, bb])
        };
        let e1 = ab;
        let two = a.int(2);
        let e2 = a.mul(&[two, ab]);
        let five = a.int(5);
        let e3 = a.mul(&[five, ab]);
        let result = a.add(&[e1, e2, e3, x]);
        assert_eq!(display(&a, result), "8*a*b + x");
    }

    #[test]
    fn add_oo_minus_oo_is_nan() {
        let mut a = Arena::new();
        let result = a.add(&[a.infinity, a.neg_infinity]);
        assert_eq!(result, a.nan);
    }

    /// `∞` absorbs the real finite terms only (SymPy's `Add.flatten`): `∞ +
    /// x` for an unassumed `x` is `∞ + i` at `x = i` and stays a sum (up to
    /// 0.40 it was `∞`, which this test pinned).
    #[test]
    fn add_oo_plus_finite_is_oo() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let result = a.add(&[a.infinity, x]);
        assert!(
            matches!(a.node(result), ExprNode::Add(ch) if ch.contains(&a.infinity) && ch.contains(&x))
        );
        let three = a.int(3);
        assert_eq!(a.add(&[a.infinity, three]), a.infinity);
        let pi = a.pi;
        assert_eq!(a.add(&[a.neg_infinity, pi]), a.neg_infinity);
    }

    // ── Mul canonicalization ────────────────────────────────────────────

    #[test]
    fn mul_combines_like_bases() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let result = a.mul(&[x, x]);
        assert_eq!(display(&a, result), "x^2");
    }

    #[test]
    fn mul_combines_powers() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let result = a.mul(&[x2, x3]);
        assert_eq!(display(&a, result), "x^5");
    }

    #[test]
    fn mul_flattens_nested_mul() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let y = s(&mut a, "y");
        let z = s(&mut a, "z");
        let inner = a.intern(ExprNode::Mul(smallvec![x, y]));
        let result = a.mul(&[inner, z]);
        assert_eq!(display(&a, result), "x*y*z");
    }

    #[test]
    fn mul_collects_numeric_coefficient() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);
        let result = a.mul(&[two, x, three]);
        assert_eq!(display(&a, result), "6*x");
    }

    #[test]
    fn mul_by_zero() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let zero = a.zero;
        let result = a.mul(&[x, zero]);
        assert_eq!(result, a.zero);
    }

    #[test]
    fn mul_by_one() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let one = a.one;
        let result = a.mul(&[one, x]);
        assert_eq!(result, x);
    }

    #[test]
    fn mul_nan_propagates() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let nan = a.nan;
        let result = a.mul(&[x, nan]);
        assert_eq!(result, a.nan);
    }

    #[test]
    fn mul_zero_times_infinity_is_nan() {
        let mut a = Arena::new();
        let result = a.mul(&[a.zero, a.infinity]);
        assert_eq!(result, a.nan);
    }

    #[test]
    fn mul_rational_coefficients() {
        let mut a = Arena::new();
        let r1 = a.rational(2, 3);
        let r2 = a.rational(3, 4);
        let result = a.mul(&[r1, r2]);
        assert_eq!(display(&a, result), "1/2");
    }

    #[test]
    fn mul_neg_neg_is_positive() {
        let mut a = Arena::new();
        let neg1 = a.int(-1);
        let result = a.mul(&[neg1, neg1]);
        assert_eq!(display(&a, result), "1");
    }

    // ── Pow canonicalization ────────────────────────────────────────────

    #[test]
    fn pow_x_zero_is_one() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let result = a.pow(x, a.zero);
        assert_eq!(result, a.one);
    }

    #[test]
    fn pow_x_one_is_x() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let result = a.pow(x, a.one);
        assert_eq!(result, x);
    }

    #[test]
    fn pow_one_x_is_one() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let result = a.pow(a.one, x);
        assert_eq!(result, a.one);
    }

    #[test]
    fn pow_zero_positive_is_zero() {
        let mut a = Arena::new();
        let base = a.zero;
        let exp = a.int(5);
        let result = a.pow(base, exp);
        assert_eq!(result, a.zero);
    }

    #[test]
    fn pow_evaluates_numeric() {
        let mut a = Arena::new();
        let base = a.int(2);
        let exp = a.int(10);
        let result = a.pow(base, exp);
        assert_eq!(display(&a, result), "1024");
    }

    #[test]
    fn pow_evaluates_rational() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let exp = a.int(3);
        let result = a.pow(half, exp);
        assert_eq!(display(&a, result), "1/8");
    }

    #[test]
    fn pow_evaluates_negative_exponent() {
        let mut a = Arena::new();
        let base = a.int(2);
        let exp = a.int(-3);
        let result = a.pow(base, exp);
        assert_eq!(display(&a, result), "1/8");
    }

    #[test]
    fn pow_nan_propagates_base() {
        let mut a = Arena::new();
        let base = a.nan;
        let exp = a.int(2);
        let result = a.pow(base, exp);
        assert_eq!(result, a.nan);
    }

    #[test]
    fn pow_nan_propagates_exp() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let result = a.pow(x, a.nan);
        assert_eq!(result, a.nan);
    }

    #[test]
    fn pow_perfect_square_simplifies() {
        let mut a = Arena::new();
        let base = a.int(4);
        let exp = a.rational(1, 2);
        let result = a.pow(base, exp);
        // 4^(1/2) = 2 via radical simplification (4 = 2²).
        assert_eq!(display(&a, result), "2");
    }

    #[test]
    fn pow_huge_exponent_stays_unevaluated() {
        let mut a = Arena::new();
        let base = a.int(2);
        let exp = a.int(5000);
        let result = a.pow(base, exp);
        // Exceeds max_pow_exponent (default 1000).
        assert_eq!(display(&a, result), "2^5000");
    }

    // ── Neg canonicalization ────────────────────────────────────────────

    #[test]
    fn neg_double_negation() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let neg_x = a.neg(x);
        let neg_neg_x = a.neg(neg_x);
        assert_eq!(neg_neg_x, x);
    }

    #[test]
    fn neg_numeric() {
        let mut a = Arena::new();
        let three = a.int(3);
        let result = a.neg(three);
        assert_eq!(display(&a, result), "-3");
    }

    #[test]
    fn neg_distributes_over_add() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let y = s(&mut a, "y");
        let sum = a.add(&[x, y]);
        let result = a.neg(sum);
        // -(x + y) → -x - y  which canonically is -x + (-y) = Add(-x, -y)
        // Display: -x - y
        // canon_neg distributes over Add: -(x+y) → Add(Mul(-1,x), Mul(-1,y))
        // display detects Mul(-1, ...) as subtraction notation.
        assert_eq!(display(&a, result), "-x - y");
    }

    #[test]
    fn neg_of_mul() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let result = a.neg(two_x);
        assert_eq!(display(&a, result), "-2*x");
    }

    #[test]
    fn neg_of_symbol() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let result = a.neg(x);
        assert_eq!(display(&a, result), "-x");
    }

    // ── Mixed / integration-style tests ─────────────────────────────────

    #[test]
    fn mixed_a_times_b_plus_b_times_a() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let y = s(&mut a, "y");
        let xy = a.mul(&[x, y]);
        let yx = a.mul(&[y, x]);
        let result = a.add(&[xy, yx]);
        assert_eq!(display(&a, result), "2*x*y");
    }

    #[test]
    fn mixed_a_minus_a_is_zero() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let result = a.sub(x, x);
        assert_eq!(result, a.zero);
    }

    #[test]
    fn mixed_division() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let y = s(&mut a, "y");
        let result = a.div(x, y);
        assert_eq!(display(&a, result), "x/y");
    }

    #[test]
    fn mixed_x_plus_2x_plus_3() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let three = a.int(3);
        let result = a.add(&[x, two_x, three]);
        assert_eq!(display(&a, result), "3*x + 3");
    }

    #[test]
    fn mixed_polynomial_canonical_form() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let exp2 = a.int(2);
        let x2 = a.pow(x, exp2);
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let one = a.int(1);
        let result = a.add(&[x2, two_x, one]);
        assert_eq!(display(&a, result), "x^2 + 2*x + 1");
    }

    #[test]
    fn mixed_compound_collection() {
        // a*b + b*a + a*b → 3*a*b  (SymPy test_arit0 inspired)
        let mut a_arena = Arena::new();
        let aa = s(&mut a_arena, "a");
        let bb = s(&mut a_arena, "b");
        let ab = a_arena.mul(&[aa, bb]);
        let ba = a_arena.mul(&[bb, aa]);
        let result = a_arena.add(&[ab, ba, ab]);
        assert_eq!(display(&a_arena, result), "3*a*b");
    }

    #[test]
    fn mixed_subtract_to_zero() {
        // b*a − b − a*b + b → 0  (SymPy test_arit0 inspired)
        let mut a = Arena::new();
        let aa = s(&mut a, "a");
        let bb = s(&mut a, "b");
        let ba = a.mul(&[bb, aa]);
        let neg_b = a.neg(bb);
        let ab = a.mul(&[aa, bb]);
        let neg_ab = a.neg(ab);
        let result = a.add(&[ba, neg_b, neg_ab, bb]);
        assert_eq!(result, a.zero);
    }

    #[test]
    fn mixed_rational_arithmetic_in_add() {
        // Rational(2) + a + Rational(5) → 7 + a
        let mut a = Arena::new();
        let sym_a = s(&mut a, "a");
        let two = a.int(2);
        let five = a.int(5);
        let result = a.add(&[two, sym_a, five]);
        assert_eq!(display(&a, result), "a + 7");
    }

    #[test]
    fn mixed_mul_abc_times_2() {
        let mut a = Arena::new();
        let aa = s(&mut a, "a");
        let bb = s(&mut a, "b");
        let cc = s(&mut a, "c");
        let two = a.int(2);
        let result = a.mul(&[aa, bb, cc, two]);
        assert_eq!(display(&a, result), "2*a*b*c");
    }

    #[test]
    fn dedup_after_canonicalization() {
        // Two separately constructed but mathematically equal expressions
        // should produce the same ExprId.
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let e1 = a.add(&[x, x]); // 2*x
        let two = a.int(2);
        let e2 = a.mul(&[two, x]); // 2*x
        assert_eq!(e1, e2, "canonical forms should hash-cons to same ExprId");
    }

    // ── Non-auto-evaluation tests ───────────────────────────────────────
    // These verify our principle: constructors do NOT expand or evaluate
    // beyond basic canonicalization.

    #[test]
    fn no_auto_expand_pow() {
        // (x+1)^2 should NOT expand to x^2 + 2*x + 1.
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let sum = a.add(&[x, a.one]);
        let exp = a.int(2);
        let result = a.pow(sum, exp);
        assert_eq!(display(&a, result), "(x + 1)^2");
    }

    #[test]
    fn no_auto_distribute_mul_over_add() {
        // x*(y + z) should NOT distribute to x*y + x*z.
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let y = s(&mut a, "y");
        let z = s(&mut a, "z");
        let sum = a.add(&[y, z]);
        let result = a.mul(&[x, sum]);
        assert_eq!(display(&a, result), "x*(y + z)");
    }

    #[test]
    fn constant_applications_with_rational_or_singular_values_fold() {
        // Before 0.30 construction kept `sin(0)`, `cos(π)` and `ln(0)` as
        // atoms, and `sin(0)/sin(0)` was 1.  Values: SymPy 1.14 `sin(0)`,
        // `cos(pi)`, `log(0)`, `tan(pi/2)`, `gamma(0)` → 0, -1, zoo, zoo, zoo.
        let mut a = Arena::new();
        let r = a.sin(a.zero);
        assert_eq!(r, a.zero);
        let r = a.cos(a.pi);
        assert_eq!(r, a.neg_one);
        let r = a.ln(a.zero);
        assert_eq!(r, a.complex_infinity);
        let half = a.rational(1, 2);
        let half_pi = a.mul(&[half, a.pi]);
        let r = a.tan(half_pi);
        assert_eq!(r, a.complex_infinity);
        let r = a.gamma(a.zero);
        assert_eq!(r, a.complex_infinity);
        let s = a.sin(a.zero);
        let inv = a.pow(s, a.neg_one);
        let q = a.mul(&[s, inv]);
        assert_eq!(q, a.nan, "sin(0)/sin(0) is 0/0");
    }

    #[test]
    fn irrational_special_values_stay_for_eval() {
        // sin(π/4) = √2/2 is not folded at construction (eval does it).
        let mut a = Arena::new();
        let quarter = a.rational(1, 4);
        let arg = a.mul(&[quarter, a.pi]);
        let result = a.sin(arg);
        assert_eq!(display(&a, result), "sin(1/4*pi)");
    }

    #[test]
    fn no_auto_cancel_fraction() {
        // (x^2 - 1) / (x - 1) should NOT auto-cancel to x + 1.
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let exp = a.int(2);
        let x2 = a.pow(x, exp);
        let numer = a.sub(x2, a.one);
        let denom = a.sub(x, a.one);
        let result = a.div(numer, denom);
        // Should be (x^2 - 1) * (x - 1)^(-1), NOT x + 1.
        let d = display(&a, result);
        assert!(!d.contains("x + 1"), "should not auto-cancel: got '{d}'");
    }

    // ── Scaling tests ───────────────────────────────────────────────────

    #[test]
    fn scaling_100_term_sum() {
        let mut a = Arena::new();
        let symbols: Vec<ExprId> = (0..100).map(|i| s(&mut a, &format!("x{i}"))).collect();
        let result = a.add(&symbols);
        // Just verify it doesn't blow up and has the right number of terms.
        if let ExprNode::Add(args) = a.node(result) {
            assert_eq!(args.len(), 100);
        } else {
            panic!("expected Add with 100 terms");
        }
    }

    #[test]
    fn scaling_1000_term_sum() {
        let mut a = Arena::new();
        let symbols: Vec<ExprId> = (0..1000).map(|i| s(&mut a, &format!("x{i}"))).collect();
        let timer = crate::base::perf_assert::CpuTimer::start();
        let result = a.add(&symbols);
        timer.assert_within(std::time::Duration::from_millis(500), "1000-term sum");
        if let ExprNode::Add(args) = a.node(result) {
            assert_eq!(args.len(), 1000);
        }
    }

    #[test]
    fn scaling_like_term_collection() {
        // 1*x + 2*x + 3*x + … + 100*x → 5050*x
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let terms: Vec<ExprId> = (1..=100)
            .map(|i| {
                let coeff = a.int(i);
                a.mul(&[coeff, x])
            })
            .collect();
        let result = a.add(&terms);
        assert_eq!(display(&a, result), "5050*x");
    }

    // ── i^n reduction ──────────────────────────────────────────────

    #[test]
    fn i_squared_is_neg_one() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let two = a.int(2);
        let result = a.pow(i, two);
        assert_eq!(result, a.neg_one);
    }

    #[test]
    fn i_cubed_is_neg_i() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let three = a.int(3);
        let result = a.pow(i, three);
        assert_eq!(display(&a, result), "-I");
    }

    #[test]
    fn i_fourth_is_one() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let four = a.int(4);
        let result = a.pow(i, four);
        assert_eq!(result, a.one);
    }

    #[test]
    fn i_to_neg_one() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let neg1 = a.int(-1);
        let result = a.pow(i, neg1);
        assert_eq!(display(&a, result), "-I");
    }

    #[test]
    fn i_to_neg_two() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let neg2 = a.int(-2);
        let result = a.pow(i, neg2);
        assert_eq!(result, a.neg_one);
    }

    #[test]
    fn i_to_100() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let hundred = a.int(100);
        let result = a.pow(i, hundred);
        assert_eq!(result, a.one);
    }

    #[test]
    fn one_plus_i_squared() {
        // (1+i)^2 = 1 + 2i + i^2 = 1 + 2i - 1 = 2*I
        let mut a = Arena::new();
        let one = a.one;
        let i = a.i_unit;
        let two = a.int(2);
        let term1 = a.mul(&[one, one]); // 1
        let term2 = a.mul(&[one, i]); // I
        let term3 = a.mul(&[i, one]); // I
        let term4 = a.pow(i, two); // i^2 = -1
        let result = a.add(&[term1, term2, term3, term4]);
        assert_eq!(display(&a, result), "2*I");
    }

    // ── (-1)^(n/2) reduction ───────────────────────────────────────────

    #[test]
    fn neg_one_to_half_is_i() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let result = a.pow(a.neg_one, half);
        assert_eq!(result, a.i_unit);
    }

    #[test]
    fn neg_one_to_three_halves_is_neg_i() {
        let mut a = Arena::new();
        let three_halves = a.rational(3, 2);
        let result = a.pow(a.neg_one, three_halves);
        assert_eq!(display(&a, result), "-I");
    }

    #[test]
    fn sqrt_neg_one_is_i() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let result = a.pow(a.neg_one, half);
        assert_eq!(result, a.i_unit);
    }

    #[test]
    fn neg_one_to_one_is_neg_one() {
        let mut a = Arena::new();
        let one = a.one;
        let result = a.pow(a.neg_one, one);
        assert_eq!(result, a.neg_one);
    }

    // ── (-n)^(1/2) → i * sqrt(n) ──────────────────────────────────────

    #[test]
    fn sqrt_neg_4_is_2i() {
        let mut a = Arena::new();
        let neg4 = a.int(-4);
        let half = a.rational(1, 2);
        let result = a.pow(neg4, half);
        assert_eq!(display(&a, result), "2*I");
    }

    #[test]
    fn sqrt_neg_2_is_i_sqrt_2() {
        let mut a = Arena::new();
        let neg2 = a.int(-2);
        let half = a.rational(1, 2);
        let result = a.pow(neg2, half);
        let s = display(&a, result);
        assert!(s.contains("I"), "expected I in {s}");
        assert!(s.contains("sqrt(2)"), "expected sqrt(2) in {s}");
    }

    #[test]
    fn sqrt_neg_9_is_3i() {
        let mut a = Arena::new();
        let neg9 = a.int(-9);
        let half = a.rational(1, 2);
        let result = a.pow(neg9, half);
        assert_eq!(display(&a, result), "3*I");
    }

    // ── verify_canonical tests ─────────────────────────────────────────

    #[test]
    fn verify_canonical_add_is_clean() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let y = s(&mut a, "y");
        let expr = a.add(&[x, y]);
        let errors = verify_canonical(&mut a, expr);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn verify_canonical_mul_is_clean() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let two = a.int(2);
        let expr = a.mul(&[two, x]);
        let errors = verify_canonical(&mut a, expr);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn verify_canonical_complex_expr() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let y = s(&mut a, "y");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let xy = a.mul(&[x, y]);
        let expr = a.add(&[x2, xy, y]);
        let errors = verify_canonical(&mut a, expr);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    // ── Radical simplification ─────────────────────────────────────────

    #[test]
    fn sqrt_12_simplifies() {
        let mut a = Arena::new();
        let twelve = a.int(12);
        let half = a.rational(1, 2);
        let result = a.pow(twelve, half);
        assert_eq!(display(&a, result), "2*sqrt(3)");
    }

    #[test]
    fn sqrt_8_simplifies() {
        let mut a = Arena::new();
        let base = a.int(8);
        let exp = a.rational(1, 2);
        let result = a.pow(base, exp);
        assert_eq!(display(&a, result), "2*sqrt(2)");
    }

    #[test]
    fn sqrt_7_stays() {
        let mut a = Arena::new();
        let base = a.int(7);
        let exp = a.rational(1, 2);
        let result = a.pow(base, exp);
        assert_eq!(display(&a, result), "sqrt(7)");
    }

    #[test]
    fn sqrt_18_simplifies() {
        let mut a = Arena::new();
        let base = a.int(18);
        let exp = a.rational(1, 2);
        let result = a.pow(base, exp);
        assert_eq!(display(&a, result), "3*sqrt(2)");
    }

    #[test]
    fn sqrt_9_simplifies_to_3() {
        let mut a = Arena::new();
        let base = a.int(9);
        let exp = a.rational(1, 2);
        let result = a.pow(base, exp);
        assert_eq!(display(&a, result), "3");
    }

    #[test]
    fn cbrt_24_simplifies() {
        let mut a = Arena::new();
        let base = a.int(24);
        let exp = a.rational(1, 3);
        let result = a.pow(base, exp);
        // 24 = 2³·3, so cbrt(24) = 2·cbrt(3)
        assert_eq!(display(&a, result), "2*cbrt(3)");
    }

    #[test]
    fn cbrt_8_simplifies_to_2() {
        let mut a = Arena::new();
        let base = a.int(8);
        let exp = a.rational(1, 3);
        let result = a.pow(base, exp);
        assert_eq!(display(&a, result), "2");
    }

    #[test]
    fn split_perfect_power_basic() {
        let sp = |n: i64, k: u32| {
            let (o, i) = split_perfect_power(&BigInt::from(n), k);
            (o.try_into().unwrap_or(-1i64), i.try_into().unwrap_or(-1i64))
        };
        assert_eq!(sp(12, 2), (2, 3));
        assert_eq!(sp(8, 2), (2, 2));
        assert_eq!(sp(18, 2), (3, 2));
        assert_eq!(sp(9, 2), (3, 1));
        assert_eq!(sp(7, 2), (1, 7));
        assert_eq!(sp(8, 3), (2, 1));
        assert_eq!(sp(24, 3), (2, 3));
        assert_eq!(sp(1, 2), (1, 1));
        // 7 · 23641997² — used to trial-divide up to 2.4·10⁷.
        assert_eq!(sp(3_912_608_155_036_063, 2), (23_641_997, 7));
    }

    #[test]
    fn split_perfect_power_leaves_large_composite_alone() {
        // (10^20 + 39)(10^20 + 5559) · 4: only the 2² is pulled out.
        let n = BigInt::parse_bytes(b"40000000000000002239200000000000000867204", 10).unwrap();
        let (o, i) = split_perfect_power(&n, 2);
        assert_eq!(o, BigInt::from(2));
        assert_eq!(&i * 4, n);
        // but an exact square of a huge prime is still recognised.
        let p = BigInt::parse_bytes(b"10000000000000000051", 10).unwrap();
        let (o, i) = split_perfect_power(&(&p * &p * 3), 2);
        assert_eq!(o, p);
        assert_eq!(i, BigInt::from(3));
    }

    /// `split_perfect_power(n, k)` as of 0.28.0, from the full bounded
    /// factorisation `(factors, cofactor)` of `n > 1`: the reference the
    /// squarefree decomposition must reproduce exactly (canonical forms are
    /// pinned byte for byte).
    fn split_perfect_power_by_factorint(
        n: &BigInt,
        k: u32,
        (factors, cofactor): &(Vec<(BigInt, u32)>, BigInt),
    ) -> (BigInt, BigInt) {
        let root = n.nth_root(k);
        if NumPow::pow(root.clone(), k) == *n {
            return (root, BigInt::one());
        }
        let mut outside = BigInt::one();
        let mut inside = cofactor.clone();
        for (p, e) in factors {
            if *e >= k {
                outside *= NumPow::pow(p.clone(), e / k);
            }
            if e % k > 0 {
                inside *= NumPow::pow(p.clone(), e % k);
            }
        }
        (outside, inside)
    }

    #[test]
    fn split_perfect_power_matches_the_factorint_split() {
        use crate::domains::ntheory::nextprime;
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut prime = |bits: u32| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let offset = BigInt::from(state) % (BigInt::one() << (bits - 2));
            nextprime((BigInt::one() << (bits - 1)) + offset)
        };
        let mut inputs: Vec<BigInt> = Vec::new();
        for _ in 0..4 {
            let (a, b, c) = (prime(17), prime(24), prime(33));
            let (d, e) = (prime(40), prime(44));
            let big = prime(46) * prime(47); // 92-bit composite, never factored
            inputs.extend([
                &a * &b,
                &a * &a * &b,
                &a * &b * &b * 12,
                NumPow::pow(&a, 3u32) * &b,
                &a * &b * &c,
                &c * &c * &a * 45,
                (&a * &b).pow(2u32) * &c,
                &d * &e,
                &d * &e * 65_521,
                &c * &d,
                &c * &c * 7,
                &big * 3,
                &big * &big * 3,
                NumPow::pow(&big, 3u32) * 4,
                &c * &c * &big,
                prime(90),
                prime(90) * &a * &a,
            ]);
        }
        for n in &inputs {
            let factored = crate::domains::ntheory::factorint_bounded(n, RADICAL_FACTOR_MAX_BITS);
            for k in [2, 3, 4, 6] {
                assert_eq!(
                    split_perfect_power(n, k),
                    split_perfect_power_by_factorint(n, k, &factored),
                    "n = {n}, k = {k}"
                );
            }
        }
    }
}
