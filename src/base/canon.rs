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
//! 3. Combine like bases: `x * x → x²`, `x² * x³ → x⁵`.
//! 4. Numeric coefficient placed first if ≠ 1.
//! 5. Remaining factors sorted by [`SortKey`](crate::base::sort_key::SortKey).
//! 6. Zero propagation: any zero factor ⟹ result is `0`, except that
//!    `0 × (±∞ | zoo | nan) ⟹ NaN` regardless of argument order.  A function
//!    application at an exact argument where it is infinite (`ln(0)`,
//!    `exp(∞)`) never reaches here: it is folded when interned (see
//!    *Function applications* below).  Other applications (`Γ(zoo)`,
//!    `f(x)`) count as finite.  A denominator `0^y` (0 on a half plane) is
//!    not a generic nonzero value: `0/0^y` is a `Piecewise` (see
//!    `zero_times_rest`).
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
///   elsewhere ([`zero_times_zero_powers`]): `0/0^y` is `Piecewise((0, (0 >
///   re(y)) | Eq(y, 0)), (nan, True))`.  Up to 0.58 it was 0, right only for
///   `re y < 0` (and `y = 0`).  (Any other factor `0^e` keeps the canonical
///   `0·f = 0` of a finite-looking factor: the exact zero coefficients of an
///   expansion at `0^e` rely on it.)
/// * a `Piecewise` factor or denominator distributes the zero over its
///   pieces ([`zero_times_piecewise`]).
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
        push_zero_powers(arena, d, &mut zero_powers);
        if matches!(arena.node(d), ExprNode::Piecewise(_)) {
            piecewise.push((d, false));
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
    let zero = zero_times_zero_powers(arena, &zero_powers);
    zero_times_piecewise(arena, zero, &piecewise)
}

/// `z` (0, or the `Piecewise` of [`zero_times_zero_powers`]) times the
/// `Piecewise` factors (`true`) and denominators (`false`) of a product
/// with the coefficient 0: `0·Piecewise((vᵢ, cᵢ))` is `Piecewise((0·vᵢ,
/// cᵢ))` (one value when they agree), and so is `0/Piecewise(…)` with
/// `0/vᵢ`.  A `Piecewise` is not a generic finite value: `0·(0/0^y)` is
/// `0/0^y`'s `Piecewise((0, …), (nan, True))` itself, as when the product is
/// built at once, and `0/(0/0^y)` is `nan` (`0/0`, `0/nan`).
fn zero_times_piecewise(arena: &mut Arena, z: ExprId, piecewise: &[(ExprId, bool)]) -> ExprId {
    let zero = arena.zero;
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
            values.push((canon_mul(arena, &[zero, v]), c));
        }
        let Some(&(first, _)) = values.first() else {
            continue;
        };
        let q = if values.iter().all(|&(v, _)| v == first) {
            first
        } else {
            arena.piecewise(&values)
        };
        if q == zero {
            continue;
        }
        result = if result == zero {
            q
        } else {
            canon_mul(arena, &[result, q])
        };
    }
    result
}

/// `−e` when the exponent `e` of a factor `0^e` has a negative numeric
/// coefficient: `0^(−y)` is the reciprocal of `0^y` (`zoo` for `re y > 0`,
/// 0 for `re y < 0`, 1 at `y = 0`), and `0·0^(−y)` is `0/0^y`.  Any other
/// factor `0^e` keeps the generic reading of a factor (`0·0^y = 0`; the exact
/// zero coefficients of an expansion at `0^y` rely on it).
fn reciprocal_zero_power(arena: &mut Arena, e: ExprId) -> Option<ExprId> {
    let negated = match arena.node(e) {
        ExprNode::Neg(_) => true,
        ExprNode::Mul(children) => children
            .first()
            .and_then(|&c| arena.as_num(c))
            .is_some_and(Signed::is_negative),
        _ => false,
    };
    negated.then(|| canon_neg(arena, e))
}

/// The exponents `e` of the powers `0^e` among the factors of the
/// denominator `d` (a product of positive numeric powers: `(0^y)^(1/2)·x`).
/// (`canon_pow` folds `0^e` for a numeric `e` and for an `e` whose real
/// part has a known sign: what stays is undecided.)
fn push_zero_powers(arena: &Arena, d: ExprId, zero_powers: &mut SmallVec<[ExprId; 2]>) {
    let mut stack: SmallVec<[ExprId; 8]> = smallvec![d];
    while let Some(id) = stack.pop() {
        match *arena.node(id) {
            ExprNode::Mul(ref children) => stack.extend(children.iter().copied()),
            ExprNode::Pow(base, e) if arena.is_zero_structural(base) => zero_powers.push(e),
            ExprNode::Pow(base, e) if arena.as_num(e).is_some_and(Signed::is_positive) => {
                stack.push(base);
            }
            _ => {}
        }
    }
}

/// `0` over finite denominators and the powers `0^e` of [`zero_times_rest`]
/// (their exponents `e`): `0^e` is 0 for `re e > 0`, `zoo` for `re e < 0`, 1
/// at `e = 0` and `nan` elsewhere on `re e = 0`, so the quotient is `0`
/// where each is nonzero and has a value (`re e < 0` or `e = 0`), and `nan`
/// elsewhere: `Piecewise((0, condition), (nan, True))`.  No such power: `0`.
/// The generic convention does not cover them (they vanish on a half plane
/// and are never divided by: CONTRIBUTING, "Piecewise parameters"), and a
/// value right on one half plane only is not folded to.
fn zero_times_zero_powers(arena: &mut Arena, zero_powers: &[ExprId]) -> ExprId {
    if zero_powers.is_empty() {
        return arena.zero;
    }
    tracing::debug!("canon_mul: 0 over a power 0^e of undecided sign → Piecewise");
    let zero = arena.zero;
    let mut conditions: SmallVec<[ExprId; 2]> = SmallVec::new();
    for &e in zero_powers {
        let re = arena.re(e);
        let half_plane = arena.gt(zero, re);
        let at_zero = arena.eq_(e, zero);
        let mut either = [half_plane, at_zero];
        either.sort_by(|a, b| arena.sort_key(*a).cmp(arena.sort_key(*b)));
        let condition = arena.or(&either);
        if !conditions.contains(&condition) {
            conditions.push(condition);
        }
    }
    conditions.sort_by(|a, b| arena.sort_key(*a).cmp(arena.sort_key(*b)));
    let condition = arena.and(&conditions);
    let (nan, otherwise) = (arena.nan, arena.bool_true);
    arena.piecewise(&[(zero, condition), (nan, otherwise)])
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
/// (up to 0.58 they were 0, as in SymPy 1.14).  Where `|d| → ∞` in every
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
/// Up to 0.58 the closed forms met `zoo + zoo` and gave `nan`:
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
/// and vanishes for `m > n ≥ 0`.  Up to 0.58 `gegenbauer(3, −2, zoo)` was
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
    if !has_nested_mul && result_args.len() > 1 {
        let mut seen: FxHashSet<ExprId> = FxHashSet::default();
        for &r in &result_args {
            if !seen.insert(arena.as_base_exp(r).0) {
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
    for (&b, exps) in bases {
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
            _ => factors.push(id),
        }
    }
    let mut memo = EverywhereMemo::default();
    for f in factors {
        // A sum undefined at every point (`0/d + 1/d` with `d` vanishing)
        // makes `zoo·nan = nan` too.
        if memo.zero_or_undefined(arena, f) {
            tracing::debug!("canon_mul: zoo times an identically zero or undefined factor → nan");
            return arena.nan;
        }
    }
    for d in denominators {
        if memo.infinite_or_undefined(arena, d) {
            tracing::debug!("canon_mul: zoo over an identically infinite sum → nan");
            return arena.nan;
        }
    }
    arena.intern(ExprNode::ComplexInfinity)
}

/// The sums whose vanishing identically makes `f` vanish identically:
/// `f` itself if it is a sum, those of the factors of a product, of the
/// base of a positive power, and of the argument `a` of `g(a)` for a `g`
/// with `g(0) = 0` (`sin`, `tan`, `sinh`, `tanh`, their inverses, `|·|`,
/// `sign`, `W`, `floor`, `ceiling`).  An explicit walk down those nodes.
/// `expand` and `ratsimp` test them by an identity beside a pole
/// (`√(tan x·cos x − sin x)` is 0 because `tan x·cos x − sin x` is): there
/// a factor that is undefined rather than 0 (a product with an infinite
/// factor inside) makes the product `nan` all the same.
pub(crate) fn zero_candidate_sums(arena: &Arena, f: ExprId) -> SmallVec<[ExprId; 4]> {
    let mut sums: SmallVec<[ExprId; 4]> = SmallVec::new();
    match arena.node(f) {
        ExprNode::Add(_) => {
            sums.push(f);
            return sums;
        }
        node if node.is_atom() => return sums,
        _ => {}
    }
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: SmallVec<[ExprId; 8]> = smallvec![f];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Add(_) => sums.push(id),
            ExprNode::Mul(children) => stack.extend_from_slice(children),
            ExprNode::Neg(inner) => stack.push(*inner),
            ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(Signed::is_positive) => {
                stack.push(*base);
            }
            node => stack.extend(node.unary_arg_if(zero_through)),
        }
    }
    sums
}

/// The functions [`not_finite_at_infinity`] examines at an infinite
/// argument: the elementary functions but `atan`, `atanh` (bounded), and
/// `|·|`, `⌊·⌋`, `⌈·⌉`, `erf`, `erfc`, `Γ`, `ln Γ`, `ψ`, `ζ`, `n!`.
fn examined_at_infinity(f: crate::base::func::Func) -> bool {
    use crate::base::func::Func;
    (f.elementary().is_some() && !matches!(f, Func::Atan | Func::Atanh))
        || matches!(
            f,
            Func::Abs
                | Func::Floor
                | Func::Ceiling
                | Func::Erf
                | Func::Erfc
                | Func::Gamma
                | Func::LogGamma
                | Func::Digamma
                | Func::Zeta
                | Func::Factorial
        )
}

/// The one-argument functions a zero of the argument passes through
/// (`f(0) = 0`) for [`zero_candidate_sums`] and [`sum_under_zero_chain`]:
/// the odd elementary functions (`sin`, `tan`, `asin`, `atan` and their
/// hyperbolic kin) and `|·|`, `sign`, `W`, `⌊·⌋`, `⌈·⌉`.
fn zero_through(f: crate::base::func::Func) -> bool {
    use crate::base::func::{Func, Parity};
    (f.elementary().is_some() && matches!(f.spec().parity, Parity::Odd(_)))
        || matches!(
            f,
            Func::Abs | Func::Sign | Func::LambertW0 | Func::Floor | Func::Ceiling
        )
}

/// `false` when `f` certainly does not vanish identically by one of its
/// [`zero_candidate_sums`] (the residue test, no expansion).
pub(crate) fn may_vanish_identically_as_factor(arena: &Arena, f: ExprId) -> bool {
    zero_candidate_sums(arena, f)
        .into_iter()
        .any(|s| arena.may_vanish_identically(s))
}

/// Does `f` vanish identically because one of its [`zero_candidate_sums`]
/// does ([`sum_vanishes_identically`])?  `x·(x + 1) − x² − x` does, so do
/// `sin(−x/(x + 1) + x·(−x/(x + 1) + 1))` and `√((x + 2)·((x + y)² − x²
/// − 2xy − y²))`; `cos(x·(x + 1) − x² − x)` and `sin²x + cos²x − 1` (not
/// 0 as a rational function of `sin x`, `cos x`) do not.
pub(crate) fn vanishes_identically_as_factor(arena: &mut Arena, f: ExprId) -> bool {
    zero_candidate_sums(arena, f)
        .into_iter()
        .any(|s| sum_vanishes_identically(arena, s))
}

/// Does a base of the product meet a positive and a negative power of
/// itself (`s·s⁻¹`, `s²·s⁻³`), while it vanishes identically
/// ([`factor_vanishes`])?  Then the product is `0·zoo =
/// nan` at every point, not the cancelled `s⁰ = 1`: up to 0.32
/// `(x·(x + 1) − x² − x)/(x·(x + 1) − x² − x)` was `1` (SymPy 1.14 too;
/// symplex's `together`, `ratsimp` and `simplify` give `nan` for `0/0`),
/// and up to 0.33 so were `(exp(2x) − exp(x)²)/(exp(2x) − exp(x)²)` (SymPy:
/// `nan`, it folds `exp(x)**2`) and `(sin² 1 + cos² 1 − 1)/(sin² 1 + cos² 1 − 1)`.
/// A base that vanishes by an identity of its functions
/// ([`vanishes_by_function_identity`]) too: up to 0.52 `(sin²x + cos²x −
/// 1)·(y + 1)/(sin²x + cos²x − 1)` was `y + 1` in every operation.
/// Only a base whose exponents have both signs is tested (a symbol at no
/// cost: it holds no sum; one declared zero is 0, so `z·z⁻¹` is `nan`).
fn cancels_vanishing_sum(
    arena: &mut Arena,
    bases: &FxHashMap<ExprId, SmallVec<[ExprId; 4]>>,
) -> bool {
    let mixed: SmallVec<[ExprId; 2]> = bases
        .iter()
        .filter(|&(&b, exps)| {
            exps.len() > 1
                && (!arena.node(b).is_atom() || pinned_zero_symbol(arena, b))
                && exps
                    .iter()
                    .any(|&e| arena.as_num(e).is_some_and(Signed::is_positive))
                && exps
                    .iter()
                    .any(|&e| arena.as_num(e).is_some_and(Signed::is_negative))
        })
        .map(|(&b, _)| b)
        .collect();
    mixed
        .into_iter()
        .any(|b| factor_vanishes(arena, b) || vanishes_by_function_identity(arena, b))
}

/// Does `d` hold a function application or another non-rational generator
/// (`sin x`, `tan x`, `eˣ`, `√x`) in its rational skeleton, so that it could
/// vanish by an identity of its functions (`tan x·cos x − sin x`) that the
/// residue test does not see?  Not an atom, and not an expression in
/// numbers and symbols only (there the residue test is exact).
fn may_vanish_by_identity(arena: &Arena, d: ExprId) -> bool {
    if arena.node(d).is_atom() {
        return false;
    }
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: SmallVec<[ExprId; 16]> = smallvec![d];
    let mut function = false;
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => return false,
            ExprNode::Num(_) | ExprNode::Symbol(_) => {}
            ExprNode::Add(children) | ExprNode::Mul(children) => stack.extend_from_slice(children),
            ExprNode::Neg(inner) => stack.push(*inner),
            ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(|q| q.is_integer()) => {
                stack.push(*base);
            }
            _ => function = true,
        }
    }
    function
}

/// The denominators of the `0/0` candidates among the products `nodes`:
/// the bases `d` of negative powers that could vanish by an identity of
/// their functions ([`may_vanish_by_identity`]: `tan x·cos x − sin x`), in a
/// product with another factor that may vanish identically by the residue
/// test ([`may_vanish_identically_as_factor`]: `(x + 1)² − x² − 2x − 1`).
/// Only those denominators are worth the numeric identity test
/// ([`Arena::vanishes_by_identity`]) where it is not affordable for every
/// denominator (`expand`, `trigsimp`): multiplying out the numerator would
/// give `0·d⁻¹ = 0` where the value is `0/0`.  Residue tests only.
pub(crate) fn identity_denominator_candidates(
    arena: &Arena,
    nodes: &[ExprId],
) -> SmallVec<[ExprId; 2]> {
    let negative_base = |c: ExprId| match arena.node(c) {
        ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(Signed::is_negative) => Some(*base),
        _ => None,
    };
    let mut transcendental: FxHashMap<ExprId, bool> = FxHashMap::default();
    let mut may_vanish: FxHashMap<ExprId, bool> = FxHashMap::default();
    let mut out: SmallVec<[ExprId; 2]> = SmallVec::new();
    for &id in nodes {
        let ExprNode::Mul(children) = arena.node(id) else {
            continue;
        };
        let mut denominators: SmallVec<[ExprId; 2]> = SmallVec::new();
        for &c in children {
            if let Some(b) = negative_base(c)
                && *transcendental
                    .entry(b)
                    .or_insert_with(|| may_vanish_by_identity(arena, b))
            {
                denominators.push(b);
            }
        }
        if denominators.is_empty() {
            continue;
        }
        let zero_candidate = children.iter().any(|&c| {
            negative_base(c).is_none()
                && *may_vanish
                    .entry(c)
                    .or_insert_with(|| may_vanish_identically_as_factor(arena, c))
        });
        if !zero_candidate {
            continue;
        }
        for d in denominators {
            if !out.contains(&d) {
                out.push(d);
            }
        }
    }
    out
}

/// The nodes among `order` (a post-order of an expression) that are `zoo`
/// (or `nan`) at every point because a negative power of a base that
/// vanishes (`zero`, decided by the caller) is in them outside every other
/// negative power: that power, and the sums, products, negations and
/// positive numeric powers above it (`y/s − 2` and its square for `s`
/// zero; not `1/(y/s − 2)`).  A factor beside one of them that vanishes
/// makes its product `0·zoo = nan`; `ratsimp` and `expand` test the
/// factors beside these, not only beside the powers themselves.
/// Structural; one pass over `order`.
pub(crate) fn nodes_over_vanishing_bases(
    arena: &Arena,
    order: &[ExprId],
    zero: impl Fn(ExprId) -> bool,
) -> FxHashSet<ExprId> {
    let mut infinite: FxHashSet<ExprId> = FxHashSet::default();
    for &id in order {
        let is_infinite = match arena.node(id) {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => ch.iter().any(|c| infinite.contains(c)),
            ExprNode::Neg(c) => infinite.contains(c),
            ExprNode::Pow(b, e) => match arena.as_num(*e) {
                Some(q) if q.is_negative() => zero(*b),
                Some(_) => infinite.contains(b),
                None => false,
            },
            _ => false,
        };
        if is_infinite {
            infinite.insert(id);
        }
    }
    infinite
}

/// The `zoo + zoo` candidates among the sums `nodes`: in a sum with exactly
/// one term over a denominator that vanishes (`vanishes`, decided by the
/// caller), the denominators of the other terms that could vanish by an
/// identity of their functions ([`may_vanish_by_identity`]): if one does,
/// the sum is `zoo + zoo = nan`, not the `zoo` of the first term (whose
/// reciprocal is a `0` that hides the `nan`).  `((x + 6)²/((x + y)² − x² −
/// 2xy − y²) + (cos x + 2)/(sin 2x − 2·sin x·cos x))`.  Structural tests
/// only; empty unless a denominator vanishes.
pub(crate) fn sum_partner_denominators(
    arena: &Arena,
    nodes: &[ExprId],
    vanishes: impl Fn(ExprId) -> bool,
) -> SmallVec<[ExprId; 2]> {
    let denominators = |t: ExprId| -> SmallVec<[ExprId; 2]> {
        let factors: &[ExprId] = match arena.node(t) {
            ExprNode::Mul(children) => children,
            _ => std::slice::from_ref(&t),
        };
        factors
            .iter()
            .filter_map(|&f| match arena.node(f) {
                ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(Signed::is_negative) => {
                    Some(*base)
                }
                _ => None,
            })
            .collect()
    };
    let mut out: SmallVec<[ExprId; 2]> = SmallVec::new();
    for &id in nodes {
        let ExprNode::Add(terms) = arena.node(id) else {
            continue;
        };
        let per_term: SmallVec<[SmallVec<[ExprId; 2]>; 6]> =
            terms.iter().map(|&t| denominators(t)).collect();
        let infinite: SmallVec<[bool; 6]> = per_term
            .iter()
            .map(|ds| ds.iter().any(|&d| vanishes(d)))
            .collect();
        if infinite.iter().filter(|&&b| b).count() != 1 {
            continue;
        }
        for (ds, &inf) in per_term.iter().zip(&infinite) {
            if inf {
                continue;
            }
            for &d in ds {
                if !out.contains(&d) && may_vanish_by_identity(arena, d) {
                    out.push(d);
                }
            }
        }
    }
    out
}

/// Does the factor `f` vanish identically: by one of its sums as a
/// rational function of its generators ([`vanishes_identically_as_factor`],
/// which relates `exp(2x)` and `exp(x)²`), or, for a constant, exactly
/// ([`constant_vanishes`]), itself or by the constant sum under its
/// positive powers and functions with `g(0) = 0` ([`sum_under_zero_chain`]:
/// `√(tan 2·cos 2 − sin 2)`, which the numeric test does not confirm as a
/// whole)?  The test of the rare paths of the canonical constructors
/// (`0·d⁻¹`, `zoo·f`, `s·s⁻¹`, like terms over `d`).
///
/// A symbol whose assumptions pin it to a value is that value
/// ([`with_pinned_values`]): `n − 2` and `n! − 2` vanish for an even prime
/// `n`, and a symbol declared zero vanishes.
pub(crate) fn factor_vanishes(arena: &mut Arena, f: ExprId) -> bool {
    let f = with_pinned_values(arena, f).unwrap_or(f);
    arena.is_zero_structural(f)
        || vanishes_identically_as_factor(arena, f)
        || constant_vanishes(arena, f)
        || sum_under_zero_chain(arena, f).is_some_and(|s| s != f && constant_vanishes(arena, s))
}

/// `f` with every free symbol whose assumptions pin it to one value
/// ([`Assumptions::pinned_value`](crate::base::assumptions::Assumptions::pinned_value):
/// `0` for `zero`, `2` for `even ∧ prime`, `1` for a positive integer
/// neither prime nor composite) replaced by that value, rebuilt by the
/// canonical constructors: what the zero tests of the rare paths
/// ([`factor_vanishes`], [`vanishes_by_function_identity`]) and
/// `identically_zero::vanishes_by_identity` decide, since the residue test
/// takes a symbol for an indeterminate.  Up to 0.53 an even prime `n` was
/// not known to be 2: `(⌈n⌉ − n)/(n − 2)` was 0 and `(x + 3)(n² − 4)/((n² −
/// 4)(y + 1))` was `(x + 3)/(y + 1)` (both `nan`; SymPy 1.14 gives 0 and
/// the quotient too).  `None` when `f` has no such symbol, or a binder
/// (whose bound variable the rebuild would replace too).
pub(crate) fn with_pinned_values(arena: &mut Arena, f: ExprId) -> Option<ExprId> {
    if arena.free_symbol_mask(f) == 0 {
        return None;
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for s in crate::base::walk::free_symbols(arena, f) {
        if let ExprNode::Symbol(sid) = *arena.node(s)
            && let Some(v) = arena.symbol_assumptions(sid).pinned_value()
        {
            let value = arena.int(v);
            cache.insert(s, value);
        }
    }
    if cache.is_empty() || arena.facts(f).has_binder() {
        return None;
    }
    for id in crate::base::walk::post_order_ids(arena, f) {
        if cache.contains_key(&id) {
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        if rebuilt != id {
            cache.insert(id, rebuilt);
        }
    }
    cache.get(&f).copied()
}

/// Is `id` an atom that may vanish: a symbol pinned to the value 0
/// ([`with_pinned_values`])?  The rare paths skip every other atom.
fn pinned_zero_symbol(arena: &Arena, id: ExprId) -> bool {
    matches!(*arena.node(id), ExprNode::Symbol(sid)
        if arena.symbol_assumptions(sid).pinned_value() == Some(0))
}

/// The sum `s` that `f` is a positive power of, or a function with `g(0) =
/// 0` of (`sin`, `|·|`, … as in [`zero_candidate_sums`]), possibly nested:
/// `f` vanishes with `s`.  Not through a product, whose other factors may
/// be infinite where `s` is 0 (`sin(s·w)` with `w = 1/(s/s′ + 2)` is
/// `sin(0·zoo)`, `nan`).  `None` when there is no such sum.
fn sum_under_zero_chain(arena: &Arena, f: ExprId) -> Option<ExprId> {
    let mut id = f;
    // A chain, not a tree: one step per node of `f`.
    loop {
        id = match arena.node(id) {
            ExprNode::Add(_) => return Some(id),
            ExprNode::Neg(inner) => *inner,
            ExprNode::Pow(base, e) if arena.as_num(*e).is_some_and(Signed::is_positive) => *base,
            node => node.unary_arg_if(zero_through)?,
        };
    }
}

/// Is `f` a constant: no free symbol, no infinity or `nan`, no unevaluated
/// node?  An explicit walk that stops at the first symbol.
fn is_numeric_constant(arena: &Arena, f: ExprId) -> bool {
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: SmallVec<[ExprId; 16]> = smallvec![f];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Symbol(_)
            | ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => return false,
            node => node.for_each_child(|c| stack.push(c)),
        }
    }
    !crate::base::walk::has_unevaluated(arena, f)
}

/// Is the constant `f` (no free symbol: `sin² 1 + cos² 1 − 1`, `e² −
/// exp(1)²`, `(1 + √2)² − 3 − 2√2`) exactly 0?  A constant is decided
/// numerically: not when the assumptions know it nonzero (`1 + √2`, `π`)
/// or a certified evaluation is nonzero (almost every constant, at the cost
/// of one evaluation); 0 when it is zero to that precision **and** exactly
/// 0 by the exact tests of `simplify` (algebraic numbers, rational
/// functions of exponentials; see
/// [`Arena::vanishes_by_identity`]).  Up to 0.33 `0/(sin² 1 + cos² 1 −
/// 1)` was `0` (SymPy 1.14 too) — so `eval` of `(e² − exp(1)²)/(sin² 1 +
/// cos² 1 − 1)`, whose numerator evaluates to 0, was `0`, and so was
/// `subs` of a point into `((x + 1)² − x² − 2x − 1)/(sin² x + cos² x −
/// 1)`; both are `nan`.  Only on the rare paths of [`factor_vanishes`], not
/// for an atom, and not inside a confirmation ([`vanishing_check_active`]).
///
/// A constant too costly to evaluate ([`costly_to_evaluate`]) is 0 only
/// by an identity that holds whatever the values of its costly parts,
/// which are never evaluated: the identity test takes them for symbols
/// ([`with_costly_parts_as_symbols`]), so `sin²u + cos²u − 1` is 0 for
/// `u = besselj(10⁵, 1)`.  Up to 0.34 such a constant was never taken for
/// 0, and `(sin²u + cos²u − 1)·zoo` was `zoo` (it is `0·zoo = nan`).  It
/// is tested only when the residue test does not rule it out
/// ([`Arena::may_vanish_identically`], which relates `sin u`, `cos u`, `tan
/// u` whatever `u` is), so a costly constant that is not 0 costs no more
/// than before.
pub(crate) fn constant_vanishes(arena: &mut Arena, f: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    if arena.node(f).is_atom() || vanishing_check_active() || !is_numeric_constant(arena, f) {
        return false;
    }
    if nonzero_by_form(arena, f) {
        return false;
    }
    if AssumptionCache::new().query(arena, f, Props::NONZERO) == Some(true) {
        return false;
    }
    if costly_to_evaluate(arena, f) && !arena.may_vanish_identically(f) {
        return false;
    }
    arena.vanishes_by_identity(f)
}

/// Is the constant `f` nonzero by its form alone: `Γ(z)` and `z! = Γ(z +
/// 1)` (`Γ` has no zeros), `eᶻ`, and a power `bᶻ` of such a constant or of
/// a nonzero number (`e^{z·ln b}`), whatever the (finite) `z`?  The numeric
/// test of [`constant_vanishes`] evaluated them: `((i!)!)!…` times `zoo`
/// (38 nested factorials of complex numbers, a parser fuzz input) took 0.15
/// s at rising precision once complex `Γ` evaluated (0.44), `33^{(i!)!…}`
/// 0.05 s (release builds); 0.43 refused the evaluation at once.  A chain of
/// bases, one step per node.
fn nonzero_by_form(arena: &Arena, f: ExprId) -> bool {
    let mut id = f;
    loop {
        match arena.node(id) {
            ExprNode::Gamma(_) | ExprNode::Factorial(_) | ExprNode::Exp(_) => return true,
            ExprNode::Pow(base, _) => match arena.as_num(*base) {
                Some(q) => return !q.is_zero(),
                None => id = *base,
            },
            _ => return false,
        }
    }
}

/// `f` with each of its costly parts — the outermost subexpressions that
/// [`is_costly_node`] (`besselj(10⁵, 1)`, `jacobi(999, …)`) — replaced by a
/// symbol of its own (`__costly_constant_k`), so that a zero test of the
/// result never evaluates them: an identity of the result holds for every
/// value of those parts, so at theirs — unless their value is a
/// singularity of a term (`tan u·cos u − sin u` at `cos u = 0`), where `f`
/// is undefined and the rare paths that ask give `nan` either way.  The
/// zero tests of `simplify` ([`Arena::vanishes_by_identity`]) and of the
/// canonical constructors ([`sum_vanishes_identically`]) use it.  `None`
/// when `f` has no costly part (or one is left).  One walk.
pub(crate) fn with_costly_parts_as_symbols(arena: &mut Arena, f: ExprId) -> Option<ExprId> {
    let mut costly: SmallVec<[ExprId; 2]> = SmallVec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: SmallVec<[ExprId; 16]> = smallvec![f];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if is_costly_node(arena, id) {
            costly.push(id);
        } else {
            arena.node(id).for_each_child(|c| stack.push(c));
        }
    }
    if costly.is_empty() {
        return None;
    }
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for (k, &c) in costly.iter().enumerate() {
        let symbol = arena.symbol(&format!("__costly_constant_{k}"));
        cache.insert(c, symbol);
    }
    for id in crate::base::walk::post_order_ids(arena, f) {
        if cache.contains_key(&id) || arena.node(id).is_atom() {
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        if rebuilt != id {
            cache.insert(id, rebuilt);
        }
    }
    let general = cache.get(&f).copied().unwrap_or(f);
    (general != f && !costly_to_evaluate(arena, general)).then_some(general)
}

/// The magnitude of a numeric parameter from which [`costly_to_evaluate`]
/// declines the numeric zero test (`besselj(10⁸, x)` took 5 s before 0.34).
const COSTLY_PARAMETER: i64 = 100_000;

/// The degree of an orthogonal polynomial from which [`costly_to_evaluate`]
/// declines the zero test: the exact test of its confirmation computes the
/// polynomial's rational value (`gegenbauer(999, 1/3, 1/3)` 0.75 s,
/// `jacobi(999, 1/3, 1/5, 1/7)` 2.8 s; at degree 199 at most 0.16 s).
const COSTLY_DEGREE: i64 = 200;

/// Is the numeric zero test of the constant `f` ([`constant_vanishes`])
/// beyond the budget of a canonical constructor, which runs on every node
/// built: does `f` apply a special function (`Γ`, `ψ⁽ⁿ⁾`, `ζ`, Bessel, the
/// incomplete `Γ`, … — anything but arithmetic, powers and the elementary
/// functions) to a number of magnitude [`COSTLY_PARAMETER`] or more, or an
/// orthogonal polynomial of degree [`COSTLY_DEGREE`] or more?  The cost of
/// evaluating some of those grows with their orders; such a constant is not
/// tested (not taken for 0, as before 0.34).  Up to 0.34-dev building
/// `cos(besselj(8345185991999992, 61/10²⁷))/sin(…)` never finished: the
/// test evaluated the Bessel function, a multiplication per unit of its
/// order (`evalf` takes it at once now; this bounds the next such kernel).
/// An explicit walk.
pub(crate) fn costly_to_evaluate(arena: &Arena, f: ExprId) -> bool {
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: SmallVec<[ExprId; 16]> = smallvec![f];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if is_costly_node(arena, id) {
            return true;
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    false
}

/// Is the node `id` itself what makes [`costly_to_evaluate`] true: a
/// special function with a numeric argument of magnitude
/// [`COSTLY_PARAMETER`] or more, or an orthogonal polynomial of degree
/// [`COSTLY_DEGREE`] or more?
fn is_costly_node(arena: &Arena, id: ExprId) -> bool {
    use crate::base::libfn::LibFn;
    let node = arena.node(id);
    let special = match node {
        ExprNode::Apply(..) | ExprNode::Lib(..) | ExprNode::Internal(..) => true,
        _ => node.func(arena).is_some_and(|(f, _)| f.is_special()),
    };
    if !special {
        return false;
    }
    if let ExprNode::Lib(
        LibFn::Legendre
        | LibFn::ChebyshevT
        | LibFn::ChebyshevU
        | LibFn::Hermite
        | LibFn::Laguerre
        | LibFn::Gegenbauer
        | LibFn::Jacobi
        | LibFn::AssocLegendre
        | LibFn::AssocLaguerre,
        args,
    ) = node
    {
        let degree_bound = Q::from_integer(BigInt::from(COSTLY_DEGREE));
        if args
            .first()
            .and_then(|&n| arena.as_num(n))
            .is_some_and(|q| q.abs() >= degree_bound)
        {
            return true;
        }
    }
    let bound = Q::from_integer(BigInt::from(COSTLY_PARAMETER));
    let mut costly = false;
    node.for_each_child(|c| costly |= arena.as_num(c).is_some_and(|q| q.abs() >= bound));
    costly
}

thread_local! {
    /// Set while [`sum_vanishes_identically`] confirms a zero by multiplying
    /// out: the products built meanwhile are not tested again, and `expand`
    /// does not look for vanishing denominators
    /// ([`vanishing_check_active`]).
    static IN_VANISHING_CHECK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Is [`sum_vanishes_identically`] confirming a zero on this thread?
pub(crate) fn vanishing_check_active() -> bool {
    IN_VANISHING_CHECK
        .try_with(std::cell::Cell::get)
        .unwrap_or(true)
}

/// Clears [`IN_VANISHING_CHECK`] when dropped (also on unwinding).
struct VanishingCheckGuard;

impl VanishingCheckGuard {
    /// `None` when a check is already running on this thread.
    fn enter() -> Option<Self> {
        IN_VANISHING_CHECK
            .try_with(|active| !active.replace(true))
            .unwrap_or(false)
            .then_some(VanishingCheckGuard)
    }
}

impl Drop for VanishingCheckGuard {
    fn drop(&mut self) {
        let _ = IN_VANISHING_CHECK.try_with(|active| active.set(false));
    }
}

/// Is the sum `s` identically 0 as a rational function of its generators
/// (`x·(x + 1) − x² − x`, `x/(x + 1) + 1/(x + 1) − 1`)?  The residue at a
/// pseudo-random point ([`Arena::may_vanish_identically`], one pass, no
/// expansion) rejects almost every sum; only a residue 0 is confirmed, by
/// the numerator of `s` as one fraction being 0 once multiplied out
/// ([`Arena::vanishes_identically`]).  `false` for anything but a sum,
/// inside another such confirmation, and for a sum with a denominator that
/// may vanish identically itself ([`Arena::may_have_vanishing_denominator`]):
/// `1/((x + y)² − x² − 2xy − y²) + 2` is `zoo + 2 = zoo`, not 0.
///
/// The confirmation evaluates (`eval` of the numerator multiplied out),
/// which computes an orthogonal polynomial of high degree exactly: a sum
/// with a costly part ([`costly_to_evaluate`]) is confirmed with its
/// costly parts as symbols ([`with_costly_parts_as_symbols`]).  Up to
/// 0.34 the residue test rejected `sin²u + cos²u − 1` before any
/// confirmation; it relates `sin` and `cos` now, and building `(sin²u +
/// cos²u − 1)·zoo` for `u = jacobi(999, 1/3, 1/5, 1/7)` would have computed
/// `u` (seconds).
pub(crate) fn sum_vanishes_identically(arena: &mut Arena, s: ExprId) -> bool {
    if !matches!(arena.node(s), ExprNode::Add(_))
        || !arena.may_vanish_identically(s)
        || arena.may_have_vanishing_denominator(s)
    {
        return false;
    }
    let Some(_guard) = VanishingCheckGuard::enter() else {
        return false;
    };
    let s = if costly_to_evaluate(arena, s) {
        match with_costly_parts_as_symbols(arena, s) {
            Some(general) => general,
            None => return false,
        }
    } else {
        s
    };
    let (numerator, _) = arena.as_numer_denom_expr(s);
    arena.vanishes_identically(numerator)
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
// Function applications at exact special arguments
// ═══════════════════════════════════════════════════════════════════════════

/// A value that a function application folds to at construction.
enum Folded {
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
enum InfDir {
    Pos,
    Neg,
    PosImag,
    NegImag,
}

impl Folded {
    fn int(n: i64) -> Self {
        Folded::Rational(Q::from_integer(BigInt::from(n)))
    }

    fn ratio(n: i64, d: i64) -> Self {
        Folded::Rational(Q::new(BigInt::from(n), BigInt::from(d)))
    }
}

/// The exact constant an argument stands for, as far as the tables of
/// [`canon_function`] need it (the canonical spellings of `q`, `q·π`,
/// `q·π·i`, `q·i`, `e` and the special atoms).
enum Arg {
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

fn classify_arg(arena: &Arena, id: ExprId) -> Arg {
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
fn is_nonpositive_int(q: &Q) -> bool {
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
fn scaled(arena: &mut Arena, q: &Q, c: ExprId) -> ExprId {
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
fn assumed_value(arena: &mut Arena, node: &ExprNode) -> Option<ExprId> {
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
fn assumed_power(arena: &mut Arena, base: ExprId, exp: ExprId) -> Option<ExprId> {
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
/// defined: `nan`, as for `zoo` (SymPy raises "not comparable"; up to 0.58
/// `Max(1, i)` and `Min(∞, i)` stayed nodes, which `0·Max(1, i)` read as 0).
fn fold_min_max(arena: &mut Arena, args: &[ExprId], is_min: bool) -> Option<ExprId> {
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
/// when proved, `UNKNOWN` when not).  Up to 0.58 it was evaluated anew each
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
fn is_positive_constant(arena: &Arena, a: ExprId) -> bool {
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
fn times_i(v: Folded) -> Option<Folded> {
    match v {
        Folded::Rational(q) if q.is_zero() => Some(Folded::int(0)),
        Folded::Rational(q) => Some(Folded::Imaginary(q)),
        Folded::ComplexInfinity => Some(Folded::ComplexInfinity),
        _ => None,
    }
}

/// `−1/2` (`true`) or `1/2`: the imaginary part over `π` of a value at
/// `∓i·∞` such as `ψ(±i·∞) = ∞ ± iπ/2`.
fn signed_half(negative: bool) -> Q {
    Q::new(BigInt::from(if negative { -1 } else { 1 }), BigInt::from(2))
}

/// Is `id` the canonical `−exp(−1)`, i.e. `−1/e`?
fn is_minus_inverse_e(arena: &Arena, id: ExprId) -> bool {
    let ExprNode::Mul(children) = arena.node(id) else {
        return false;
    };
    matches!(**children, [c, e] if c == arena.neg_one
        && matches!(arena.node(e), ExprNode::Exp(m) if *m == arena.neg_one))
}

/// Is `B(a, b) = Γ(a)Γ(b)/Γ(a+b)` at rationals a pole: exactly one of
/// `Γ(a)`, `Γ(b)` has one and `Γ(a+b)` does not?
fn beta_has_pole(a: &Q, b: &Q) -> bool {
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
fn lib_function_value(
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
fn bessel_i_at_minus_infinity(nu: &Q) -> Folded {
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
fn expint_at_minus_infinity(nu: &Q) -> Folded {
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
fn upper_gamma_at_minus_infinity(s: &Q) -> Folded {
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
fn lower_gamma_at_minus_infinity(s: &Q) -> Folded {
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

// ── Library functions at infinite arguments ────────────────────────────────────────────

/// An argument of a library function as [`at_infinity`] reads it: finite
/// ([`ValueClass::is_finite_proved`]: no infinity, `zoo` or `nan` in it, and
/// a number, an argument with symbols or a constant known finite or with a
/// certified value), an exact infinity `iᵏ·∞` in an axis direction (`k`
/// quarter turns: `+∞`, `i·∞`, `−∞`, `−i·∞`), or anything else (`x + ∞`,
/// `(1 + i)·∞`, `zoo`, an unevaluated pole `elliptic_pi(π, 1)`).  Up to H1b
/// `elliptic_f(elliptic_pi(π, 1), ∞)` folded to 0 (`Π(π | 1) = −∞`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Finite,
    Infinite(u8),
    Other,
}

/// The [`Slot`] of the argument `a`, from its [`ValueClass`].
fn slot(arena: &Arena, a: ExprId) -> Slot {
    match ValueClass::of(arena, a) {
        ValueClass::Finite => Slot::Finite,
        ValueClass::PosInf => Slot::Infinite(0),
        ValueClass::NegInf => Slot::Infinite(2),
        ValueClass::DirectedInf => match classify_arg(arena, a) {
            Arg::ImaginaryInfinity(false) => Slot::Infinite(1),
            Arg::ImaginaryInfinity(true) => Slot::Infinite(3),
            _ => Slot::Other,
        },
        ValueClass::ComplexInf | ValueClass::NaN | ValueClass::Infinite | ValueClass::Unknown => {
            Slot::Other
        }
    }
}

/// `Π(n | iᵗ·∞) = Π(iᵗ·∞ | m) = 0` ([`at_infinity`]) holds off the pole
/// of the other argument at 1.  `at_infinity` reads the arguments as built
/// and so excludes only the literal 1; before a constant argument folds,
/// its difference from 1 must have a certified nonzero value
/// ([`Arena::part_signs`]), so that one equal to 1 (`besseli(0, 0)`) does
/// not.  An argument with symbols is a generic point.  Up to H1b
/// `elliptic_pi(besseli(0, 0), ∞)` was built as 0 (`Π(1 | m) = zoo`: 0.53
/// evaluated it to `zoo`).
fn off_the_pole(arena: &mut Arena, f: crate::base::libfn::LibFn, args: &[ExprId]) -> bool {
    if f != crate::base::libfn::LibFn::EllipticPi {
        return true;
    }
    let constants: SmallVec<[ExprId; 2]> = args
        .iter()
        .copied()
        .filter(|&a| {
            arena.as_num(a).is_none()
                && arena.is_symbol_free(a)
                && !value_class::holds_real_infinity(arena, a)
        })
        .collect();
    constants.into_iter().all(|a| {
        let minus_one = arena.neg_one;
        let d = canon_add(arena, &[a, minus_one]);
        arena.as_num(d).is_some_and(|q| !q.is_zero())
            || arena
                .part_signs(d)
                .is_some_and(|s| s.re.is_some() || s.im.is_some())
    })
}

/// What a library function application is when one of its arguments is an
/// exact infinity in an axis direction ([`at_infinity`]).
enum AtInfinity {
    /// Its value there (the limit along the direction), folded when the
    /// node is built.
    Value(Folded),
    /// No finite value is known: it is infinite in a direction the table
    /// does not build, has no limit (`Ai′(−∞)`, `γ(1, i·∞)`), or depends
    /// on a parameter without a value (`besseli(ν, −∞)`, `hermite(n, ∞)`).
    /// `0·f` and `0/f` are `nan` ([`library_value_unknown_at_infinity`]).
    NoFiniteValue,
    /// Not tabulated: no exact infinite argument, or a finite value the
    /// table does not build (`F(i·∞ | m) = i·K(1 − m)`) or may exist in a
    /// slot not analysed (the degree of an orthogonal polynomial, `B_x(a,
    /// b)`): `0·f` keeps folding to 0.
    Untabulated,
}

/// `iᵏ·∞` for `k` quarter turns.
fn directed(turns: u8) -> Folded {
    match turns % 4 {
        0 => Folded::Infinity,
        1 => Folded::ImaginaryInfinity(false),
        2 => Folded::NegInfinity,
        _ => Folded::ImaginaryInfinity(true),
    }
}

/// `n mod 4`, the quarter turns of `iⁿ`.
fn quarter_turns(n: &BigInt) -> u8 {
    use num_integer::Integer;
    n.mod_floor(&BigInt::from(4)).to_u8().unwrap_or(0)
}

/// The value at `iᵗ·∞` of a polynomial of degree `degree` whose leading
/// coefficient is negative (`negative`) or positive: `±i^{t·degree}·∞`,
/// and `±1` for degree 0 (every constant [`at_infinity`] meets is `±1`).
fn polynomial_at_infinity(degree: &BigInt, negative: bool, t: u8) -> Folded {
    if degree.is_zero() {
        return Folded::int(if negative { -1 } else { 1 });
    }
    let turns = u32::from(t) * u32::from(quarter_turns(degree)) + if negative { 2 } else { 0 };
    directed((turns % 4) as u8)
}

/// Is `Γ(x) < 0` for a rational `x` that is not a pole?  For `x ∈ (−k, 1 −
/// k)` the sign is `(−1)ᵏ`.
fn gamma_is_negative(x: &Q) -> bool {
    use num_integer::Integer;
    x.is_negative() && (-x).ceil().to_integer().is_odd()
}

/// Is the rising factorial `(a)ₙ = a(a + 1)⋯(a + n − 1)` negative, for a
/// rational `a` none of whose factors is 0?  An odd count `min(n, ⌈−a⌉)`
/// of negative factors.
fn rising_is_negative(a: &Q, n: &BigInt) -> bool {
    use num_integer::Integer;
    if !a.is_negative() {
        return false;
    }
    let count = (-a).ceil().to_integer().min(n.clone());
    count.is_odd()
}

/// The values at infinite arguments of the classical orthogonal
/// polynomials `Pν`, `Tν`, `Uν`, `Hν`, `Lν` of a rational degree `nu` (the
/// argument `iᵗ·∞`).  An integer degree with a polynomial: the sign of
/// the leading coefficient (`Pₙ`, `Tₙ`, `Uₙ`, `Hₙ` positive, `Lₙ` of sign
/// `(−1)ⁿ`) and the parity of the degree give the direction, the constant
/// is the value of degree 0; the polynomial identities `P_{−n−1} = Pₙ`,
/// `T_{−n} = Tₙ`, `U₋₁ = 0`, `U_{−n} = −U_{n−2}`.  Otherwise by the
/// asymptotic forms (DLMF 12.7, 14.8, 13.7, 18.5): `Hν(z) ~ (2z)^ν` off
/// the negative axis (`0` for `ν < 0`, `zoo` off the axes) and `Hν(−x) ~
/// √π·e^{x²}x^{−ν−1}/Γ(−ν)`; `Lν = ₁F₁(−ν; 1; ·)` (so `L_{−m}(x) =
/// eˣ L_{m−1}(−x)`) with `Lν(x) ~ eˣx^{−ν−1}/Γ(−ν)` and `Lν(−x) ~
/// x^ν/Γ(1 + ν)`; `P_{−1/2} → 0` and `Uν → 0` for `−2 < ν < 0`, every
/// other degree infinite.  mpmath at `|x| = 10³, 10⁶`: `hermite(−1/2,
/// 10⁶j)` = 7.07·10⁻⁴·e^{−iπ/4}, `hermite(1/2, −10⁶)` = −8.9·10^{434294481893},
/// `laguerre(−1/2, 0, −10⁶)` = 5.6·10⁻⁴.
fn classical_at_infinity(f: crate::base::libfn::LibFn, nu: &Q, t: u8) -> AtInfinity {
    use crate::base::libfn::LibFn;
    use num_integer::Integer;
    if nu.is_integer() {
        let n = nu.to_integer();
        let polynomial = match f {
            LibFn::Legendre if n.is_negative() => (-&n - 1, false),
            LibFn::ChebyshevT => (n.abs(), false),
            LibFn::ChebyshevU if n == BigInt::from(-1) => {
                return AtInfinity::Value(Folded::int(0));
            }
            LibFn::ChebyshevU if n.is_negative() => (-&n - 2, true),
            LibFn::Hermite if n.is_negative() => {
                return AtInfinity::Value(if t == 2 {
                    Folded::Infinity
                } else {
                    Folded::int(0)
                });
            }
            LibFn::Laguerre if n.is_negative() => {
                return match t {
                    0 => AtInfinity::Value(Folded::Infinity),
                    2 => AtInfinity::Value(Folded::int(0)),
                    _ => AtInfinity::NoFiniteValue,
                };
            }
            LibFn::Laguerre => (n.clone(), n.is_odd()),
            _ => (n, false),
        };
        return AtInfinity::Value(polynomial_at_infinity(&polynomial.0, polynomial.1, t));
    }
    let negative = nu.is_negative();
    let minus_half = Q::new(BigInt::from(-1), BigInt::from(2));
    let infinite = |negative_direction: bool| {
        AtInfinity::Value(if negative_direction {
            Folded::NegInfinity
        } else {
            Folded::Infinity
        })
    };
    match f {
        LibFn::Legendre if *nu == minus_half => AtInfinity::Value(Folded::int(0)),
        LibFn::ChebyshevU if negative && *nu > Q::from_integer(BigInt::from(-2)) => {
            AtInfinity::Value(Folded::int(0))
        }
        LibFn::Hermite => match t {
            2 => infinite(gamma_is_negative(&-nu)),
            _ if negative => AtInfinity::Value(Folded::int(0)),
            0 => infinite(false),
            _ => AtInfinity::Value(Folded::ComplexInfinity),
        },
        LibFn::Laguerre => match t {
            0 => infinite(gamma_is_negative(&-nu)),
            2 if negative => AtInfinity::Value(Folded::int(0)),
            2 => infinite(false),
            _ if !negative => AtInfinity::Value(Folded::ComplexInfinity),
            _ if *nu > -Q::one() => AtInfinity::Value(Folded::int(0)),
            _ => AtInfinity::NoFiniteValue,
        },
        _ => AtInfinity::NoFiniteValue,
    }
}

/// The coefficients (the constant first) of the Jacobi polynomial
/// `Pₙ^{(a,b)}(x) = Σ_ℓ (n + a + b + 1)_ℓ (a + ℓ + 1)_{n−ℓ}/(ℓ!(n − ℓ)!)
/// ((x − 1)/2)^ℓ` (DLMF 18.5.7), for a degree small enough to expand.
fn jacobi_coefficients(n: usize, a: &Q, b: &Q) -> Vec<Q> {
    let int = |k: usize| Q::from_integer(BigInt::from(k));
    let c = int(n) + a + b + Q::one();
    let mut coefficients = vec![Q::zero(); n + 1];
    let mut factorials = vec![Q::one(); n + 1];
    for k in 1..=n {
        factorials[k] = &factorials[k - 1] * int(k);
    }
    for l in 0..=n {
        let mut term = Q::one();
        for j in 0..l {
            term *= &c + int(j);
        }
        for j in 0..(n - l) {
            term *= a + int(l + 1 + j);
        }
        term /= &factorials[l] * &factorials[n - l];
        term /= Q::from_integer(BigInt::one() << l);
        // ((x − 1)/2)^ℓ = 2^{−ℓ}·Σⱼ C(ℓ, j)·(−1)^{ℓ−j}·xʲ.
        let mut binomial = Q::one();
        for (j, coefficient) in coefficients.iter_mut().enumerate().take(l + 1) {
            let signed = if (l - j) % 2 == 1 {
                -(&term * &binomial)
            } else {
                &term * &binomial
            };
            *coefficient += signed;
            binomial = binomial * int(l - j) / int(j + 1);
        }
    }
    coefficients
}

/// Largest degree whose Jacobi coefficients [`at_infinity`] expands when
/// the leading one vanishes.
const MAX_JACOBI_EXPANSION: usize = 64;

/// The values at `iᵗ·∞` of the parametrised orthogonal polynomials of a
/// non-negative integer degree `n` with rational parameters: the leading
/// term.  `Cₙ^{(α)}` has the leading coefficient `2ⁿ(α)ₙ/n!`, and for `α =
/// −m` with `0 ≤ m < n` the terms `x^{n−2k}` with `k ≥ n − m` only: 0 when
/// `2(n − m) > n` (`Cₙ^{(0)} = 0`), else the degree `2m − n` with the sign
/// `(−1)ⁿ` (`C₂^{(−1)} = 1`, `C₃^{(−2)} = −4x`).  `Pₙ^{(a,b)}` has the
/// leading coefficient `(n + a + b + 1)ₙ/(2ⁿn!)`; when it vanishes the
/// coefficients are expanded ([`jacobi_coefficients`]).  `Lₙ^{(α)}` has
/// `(−1)ⁿ/n!` for every `α`.  `None` when a vanishing leading
/// coefficient leaves a degree too large to expand.
fn parametrised_at_infinity(
    f: crate::base::libfn::LibFn,
    n: &BigInt,
    params: &[&Q],
    t: u8,
) -> Option<Folded> {
    use crate::base::libfn::LibFn;
    use num_integer::Integer;
    if n.is_zero() {
        return Some(Folded::int(1));
    }
    match (f, params) {
        (LibFn::Gegenbauer, [alpha]) => {
            let m = -*alpha;
            if m.is_integer() && !m.is_negative() && m.to_integer() < *n {
                let m = m.to_integer();
                if (n - &m) * 2 > *n {
                    return Some(Folded::int(0));
                }
                return Some(polynomial_at_infinity(&(&m * 2 - n), n.is_odd(), t));
            }
            Some(polynomial_at_infinity(n, rising_is_negative(alpha, n), t))
        }
        (LibFn::Jacobi, [a, b]) => {
            let c = Q::from_integer(n.clone()) + *a + *b + Q::one();
            let vanishes = c.is_integer() && !c.is_positive() && -c.to_integer() < *n;
            if !vanishes {
                return Some(polynomial_at_infinity(n, rising_is_negative(&c, n), t));
            }
            let degree = n.to_usize().filter(|&d| d <= MAX_JACOBI_EXPANSION)?;
            let coefficients = jacobi_coefficients(degree, a, b);
            Some(match coefficients.iter().rposition(|c| !c.is_zero()) {
                None => Folded::int(0),
                Some(0) => Folded::Rational(coefficients[0].clone()),
                Some(d) => {
                    polynomial_at_infinity(&BigInt::from(d), coefficients[d].is_negative(), t)
                }
            })
        }
        (LibFn::AssocLaguerre, _) => Some(polynomial_at_infinity(n, n.is_odd(), t)),
        _ => None,
    }
}

/// What the library function `f` is at `args` when one of them is an exact
/// infinity in an axis direction: the value the constructors fold, or
/// whether a finite value may exist ([`AtInfinity`]).  Every finite value
/// a tabulated pattern has is folded, so that what stays unevaluated there
/// is not taken for finite (`0·f` is `nan`; up to 0.53 `0·besselj(1,
/// i·∞)` and `0·hermite(1, ∞)` were 0, and `x·hermite(1, ∞)` had the
/// root 0).  Values by DLMF and mpmath at `|x| = 10³, 10⁶` (30 and 60
/// digits), SymPy 1.14 where it has them:
///
/// * orthogonal polynomials at `iᵗ·∞` ([`classical_at_infinity`],
///   [`parametrised_at_infinity`]): `H₁(∞) = ∞`, `L₃(∞) = −∞`, `P₃(i·∞) =
///   −i·∞`, `C₂^{(−1)}(∞) = 1` (the degree slot is not tabulated);
/// * Bessel functions at `±i·∞`: `Jν(±iy) = e^{±iνπ/2}Iν(y)`, `Yν(±iy) ~
///   e^{±i(ν+1)π/2}Iν(y)` (DLMF 10.27.6, 10.27.11: a quarter turn per unit
///   of an integer order, `zoo` for any other), `Iν(±i·∞) = Kν(±i·∞) =
///   0`; at an infinite order `J∞(x) = I∞(x) = 0` (`(x/2)^ν/Γ(ν + 1)`),
///   `K_{±i∞}(x) = 0` for `x > 0`;
/// * `Eν(±i·∞) = 0`, `E_∞(z) = E_{±i∞}(z) = 0` (`e^{−z}/(z + ν)`, DLMF
///   8.20); `Γ(s, ±i·∞) = 0` and `γ(s, ±i·∞) = Γ(s)` for `s < 1` (no
///   limit for `s ≥ 1`: `Γ(1, iy) = e^{−iy}`), `Γ(−∞, x) = 0` for `x ≥
///   1`, `γ(∞, x) = 0` for `|x| ≤ 1`, `Γ(±i∞, x) = γ(±i∞, x) = 0` for
///   `x > 0`;
/// * `Ai`, `Bi`, `Ai′`, `Bi′` at `±i·∞`: `zoo` (`|e^{∓ζ}|` grows while
///   its phase turns); `K(±i·∞) = 0`, `E(±i·∞) = zoo` (`√(∓i)·∞`, off
///   the axes); `Chi(±i·∞) = Shi(±i·∞) = ±iπ/2`, `erfi(±i·∞) = ±i`,
///   `S(±i·∞) = ∓i/2`, `C(±i·∞) = ±i/2` (SymPy agrees);
/// * `W_k(iᵗ·∞) = ∞ + iπ(2k + arg)`, `arg ∈ {0, 1/2, 1, −1/2}` (`W_k(z) ~
///   ln z + 2πik − ln(ln z + 2πik)`: `lambertw(−10⁶, 1)` = 14.2 + 8.8i,
///   tending to `3π`);
/// * `Li_s(iᵗ·∞) = −1` for `s = 0`, `0` for `s < 0`, `−∞` for `0 < s < 1`
///   and at `−∞`, `−ln(1 − z)` for `s = 1` (`−∞ − iπ` at `+∞`), `zoo`
///   otherwise; `Li_∞(z) = z` for `|z| ≤ 1`;
/// * `F(φ | iᵗ·∞) = 0` (SymPy: `elliptic_f(z, oo)` → 0), `Π(n | iᵗ·∞) =
///   Π(iᵗ·∞ | m) = 0` when the other argument is known not to be 1
///   ([`off_the_pole`]; SymPy agrees at `±∞`): up to 0.53
///   `euler_number(5)/elliptic_pi(−∞, π⁴)` was 0;
/// * `(x)ₙ` and `x^{(n)}` at `x = iᵗ·∞`: `i^{tn}·∞` for an integer `n >
///   0`, 1 for `n = 0`, 0 for `n < 0` (SymPy: `rf(oo, −1)` → oo, wrong),
///   `x^ν` otherwise; at an infinite order `(x)_{±i∞} = 0`, `(x)_∞ =
///   Γ(x)⁻¹·∞`, `(−m)_{±∞} = 0`, `x^{(−∞)} = 0`;
/// * the integer sequences at `+∞` (`F_∞ = L_∞ = H_∞ = C_∞ = B_∞ = p(∞)
///   = ∞!! = !∞ = ∞`), `H_{±i∞} = ∞ ± iπ/2`, `C_{±i∞} = 0`, `B_{−∞} =
///   ∞` (`−s·ζ(1 − s)`);
/// * no finite value: `erfinv`, `erfcinv` (defined on `(−1, 1)`), `η` off
///   `+∞`, `F(±∞ | m)`, Bernoulli numbers at `+∞`.
///
/// The values at `±∞` of the functions folded since 0.51–0.53 (Bessel
/// functions, `Eν`, `Γ(s, x)`, `γ(s, x)`, Airy functions, `K`, `E`, `Chi`,
/// `Shi`, `erfi`, Fresnel integrals, `η`) are built by
/// [`lib_function_value`]; what it leaves there is [`AtInfinity::NoFiniteValue`].
fn at_infinity(arena: &Arena, f: crate::base::libfn::LibFn, args: &[ExprId]) -> AtInfinity {
    use crate::base::libfn::LibFn;
    use AtInfinity::{NoFiniteValue, Untabulated, Value};
    // (a cheap filter: `±∞`, or a product with `∞` that may be `±i·∞`)
    let candidate = |a: ExprId| {
        ValueClass::special_atom(arena, a).is_some_and(|c| c.real_infinity().is_some())
            || is_directed_infinity(arena, a)
    };
    if !args.iter().any(|&a| candidate(a)) {
        return Untabulated;
    }
    let slots: SmallVec<[Slot; 4]> = args.iter().map(|&a| slot(arena, a)).collect();
    if !slots.iter().any(|s| matches!(s, Slot::Infinite(_))) || slots.contains(&Slot::Other) {
        return Untabulated;
    }
    let num = |i: usize| args.get(i).and_then(|&a| arena.as_num(a));
    let s = |i: usize| slots.get(i).copied().unwrap_or(Slot::Other);
    let int = |i: i64| Q::from_integer(BigInt::from(i));
    let fin = Slot::Finite;
    let value = Value;
    match f {
        LibFn::Legendre
        | LibFn::ChebyshevT
        | LibFn::ChebyshevU
        | LibFn::Hermite
        | LibFn::Laguerre => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(t)) => match num(0) {
                Some(nu) => classical_at_infinity(f, nu, t),
                None => NoFiniteValue,
            },
            _ => Untabulated,
        },
        LibFn::Gegenbauer | LibFn::Jacobi | LibFn::AssocLaguerre => {
            let last = args.len() - 1;
            let Slot::Infinite(t) = s(last) else {
                return Untabulated;
            };
            if slots[..last].iter().any(|&p| p != fin) {
                return Untabulated;
            }
            let Some(n) = num(0) else {
                return NoFiniteValue;
            };
            if !n.is_integer() || n.is_negative() {
                return Untabulated;
            }
            if n.is_zero() {
                return value(Folded::int(1));
            }
            let params: SmallVec<[&Q; 2]> = match f {
                LibFn::AssocLaguerre => SmallVec::new(),
                _ => match (1..last).map(num).collect::<Option<SmallVec<[&Q; 2]>>>() {
                    Some(p) => p,
                    None => return NoFiniteValue,
                },
            };
            match parametrised_at_infinity(f, &n.to_integer(), &params, t) {
                Some(v) => value(v),
                None => Untabulated,
            }
        }
        LibFn::BesselJ | LibFn::BesselY | LibFn::BesselI | LibFn::BesselK => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(t)) => bessel_at_infinite_argument(f, num(0), t),
            (Slot::Infinite(0), Slot::Finite) => match f {
                LibFn::BesselJ | LibFn::BesselI => value(Folded::int(0)),
                _ => NoFiniteValue,
            },
            (Slot::Infinite(2), Slot::Finite) => NoFiniteValue,
            (Slot::Infinite(_), Slot::Finite) => match num(1) {
                Some(x) if x.is_positive() => match f {
                    LibFn::BesselK => value(Folded::int(0)),
                    _ => NoFiniteValue,
                },
                _ => Untabulated,
            },
            _ => NoFiniteValue,
        },
        LibFn::ExpInt => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(2)) => match num(0) {
                Some(nu) => value(expint_at_minus_infinity(nu)),
                None => NoFiniteValue,
            },
            (Slot::Finite, Slot::Infinite(_)) => value(Folded::int(0)),
            (Slot::Infinite(2), Slot::Finite) => NoFiniteValue,
            (Slot::Infinite(_), Slot::Finite) => value(Folded::int(0)),
            _ => NoFiniteValue,
        },
        LibFn::UpperGamma => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(0)) => value(Folded::int(0)),
            (Slot::Finite, Slot::Infinite(2)) => match num(0) {
                Some(s) => value(upper_gamma_at_minus_infinity(s)),
                None => NoFiniteValue,
            },
            (Slot::Finite, Slot::Infinite(_)) => match num(0) {
                Some(s) if *s < Q::one() => value(Folded::int(0)),
                _ => NoFiniteValue,
            },
            (Slot::Infinite(2), Slot::Finite) => match num(1) {
                Some(x) if *x >= Q::one() => value(Folded::int(0)),
                _ => NoFiniteValue,
            },
            (Slot::Infinite(t), Slot::Finite) if t % 2 == 1 => match num(1) {
                Some(x) if !x.is_negative() => value(Folded::int(0)),
                Some(_) => Untabulated,
                None => NoFiniteValue,
            },
            _ => NoFiniteValue,
        },
        LibFn::LowerGamma => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(0)) => value(Folded::Gamma(args[0])),
            (Slot::Finite, Slot::Infinite(2)) => match num(0) {
                Some(s) if s.is_positive() => value(lower_gamma_at_minus_infinity(s)),
                _ => NoFiniteValue,
            },
            (Slot::Finite, Slot::Infinite(_)) => match num(0) {
                Some(s) if is_nonpositive_int(s) => value(Folded::ComplexInfinity),
                Some(s) if *s < Q::one() => value(Folded::Gamma(args[0])),
                _ => NoFiniteValue,
            },
            (Slot::Infinite(0), Slot::Finite) => match num(1) {
                Some(x) if x.abs() <= Q::one() => value(Folded::int(0)),
                _ => NoFiniteValue,
            },
            (Slot::Infinite(t), Slot::Finite) if t % 2 == 1 => match num(1) {
                Some(x) if x.is_positive() => value(Folded::int(0)),
                Some(x) if x.is_negative() => Untabulated,
                _ => NoFiniteValue,
            },
            _ => NoFiniteValue,
        },
        LibFn::AiryAi | LibFn::AiryBi | LibFn::AiryAiPrime | LibFn::AiryBiPrime => match s(0) {
            Slot::Infinite(0) => match f {
                LibFn::AiryAi | LibFn::AiryAiPrime => value(Folded::int(0)),
                _ => value(Folded::Infinity),
            },
            Slot::Infinite(2) => match f {
                LibFn::AiryAi | LibFn::AiryBi => value(Folded::int(0)),
                _ => NoFiniteValue,
            },
            _ => value(Folded::ComplexInfinity),
        },
        LibFn::EllipticK => value(Folded::int(0)),
        LibFn::EllipticE => value(match s(0) {
            Slot::Infinite(2) => Folded::Infinity,
            _ => Folded::ComplexInfinity,
        }),
        LibFn::Chi | LibFn::Shi | LibFn::Erfi | LibFn::FresnelS | LibFn::FresnelC => {
            let Slot::Infinite(t) = s(0) else {
                return Untabulated;
            };
            let half = Q::new(BigInt::one(), BigInt::from(2));
            let up = t == 1;
            let sign = |q: Q| if up { q } else { -q };
            value(match (f, t) {
                (LibFn::FresnelS | LibFn::FresnelC, 0) => Folded::Rational(half),
                (LibFn::FresnelS | LibFn::FresnelC, 2) => Folded::Rational(-half),
                (LibFn::FresnelS, _) => Folded::Imaginary(sign(-half)),
                (LibFn::FresnelC, _) => Folded::Imaginary(sign(half)),
                (_, 0) => Folded::Infinity,
                (LibFn::Chi, 2) => Folded::InfinitePlus(InfDir::Pos, Q::one(), Q::zero()),
                (_, 2) => Folded::NegInfinity,
                (LibFn::Erfi, _) => Folded::Imaginary(sign(Q::one())),
                _ => Folded::PiImaginary(sign(half)),
            })
        }
        LibFn::DirichletEta => match s(0) {
            Slot::Infinite(0) => value(Folded::int(1)),
            _ => NoFiniteValue,
        },
        LibFn::ErfInv | LibFn::ErfcInv => NoFiniteValue,
        LibFn::LambertW => {
            let Slot::Infinite(t) = s(0) else {
                return NoFiniteValue;
            };
            let k = match args.get(1) {
                None => Q::zero(),
                Some(_) => match num(1) {
                    Some(k) if k.is_integer() && s(1) == fin => k.clone(),
                    _ => return NoFiniteValue,
                },
            };
            let turn = Q::new(BigInt::from(i64::from(t)), BigInt::from(2));
            let arg = if t == 3 { turn - int(2) } else { turn };
            value(Folded::InfinitePlus(
                InfDir::Pos,
                k * int(2) + arg,
                Q::zero(),
            ))
        }
        LibFn::PolyLog => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(t)) => match num(0) {
                Some(s) if s.is_zero() => value(Folded::int(-1)),
                Some(s) if s.is_negative() => value(Folded::int(0)),
                Some(_) if t == 2 => value(Folded::NegInfinity),
                Some(s) if s.is_one() => {
                    let q = match t {
                        0 => -Q::one(),
                        1 => Q::new(BigInt::one(), BigInt::from(2)),
                        _ => Q::new(BigInt::from(-1), BigInt::from(2)),
                    };
                    value(Folded::InfinitePlus(InfDir::Neg, q, Q::zero()))
                }
                Some(s) if *s < Q::one() => value(Folded::NegInfinity),
                Some(_) => value(Folded::ComplexInfinity),
                None => NoFiniteValue,
            },
            (Slot::Infinite(0), Slot::Finite) => match num(1) {
                Some(z) if z.abs() <= Q::one() => value(Folded::Rational(z.clone())),
                Some(_) => NoFiniteValue,
                None => Untabulated,
            },
            _ => NoFiniteValue,
        },
        LibFn::EllipticF => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(_)) => value(Folded::int(0)),
            (Slot::Infinite(t), Slot::Finite) if t % 2 == 1 => Untabulated,
            _ => NoFiniteValue,
        },
        LibFn::EllipticPi => match (s(0), s(1)) {
            (Slot::Finite, Slot::Infinite(_)) if args[0] != arena.one => value(Folded::int(0)),
            (Slot::Infinite(_), Slot::Finite) if args[1] != arena.one => value(Folded::int(0)),
            _ => NoFiniteValue,
        },
        LibFn::RisingFactorial | LibFn::FallingFactorial => {
            let rising = f == LibFn::RisingFactorial;
            match (s(0), s(1)) {
                (Slot::Infinite(t), Slot::Finite) => match num(1) {
                    Some(n) if n.is_integer() && n.is_positive() => {
                        value(polynomial_at_infinity(&n.to_integer(), false, t))
                    }
                    Some(n) if n.is_zero() => value(Folded::int(1)),
                    Some(n) if n.is_integer() => value(Folded::int(0)),
                    Some(n) => match t {
                        2 => NoFiniteValue,
                        _ if n.is_negative() => value(Folded::int(0)),
                        0 => value(Folded::Infinity),
                        _ => value(Folded::ComplexInfinity),
                    },
                    None => NoFiniteValue,
                },
                (Slot::Finite, Slot::Infinite(t)) => factorial_order_at_infinity(rising, num(0), t),
                _ => NoFiniteValue,
            }
        }
        LibFn::Fibonacci
        | LibFn::Lucas
        | LibFn::Harmonic
        | LibFn::Catalan
        | LibFn::Bell
        | LibFn::PartitionCount
        | LibFn::Factorial2
        | LibFn::Subfactorial
        | LibFn::Bernoulli => match (f, s(0)) {
            (LibFn::Bernoulli, Slot::Infinite(0)) => NoFiniteValue,
            (LibFn::Bernoulli, Slot::Infinite(2)) | (_, Slot::Infinite(0)) => {
                value(Folded::Infinity)
            }
            (LibFn::PartitionCount | LibFn::Subfactorial | LibFn::Bernoulli, _) => Untabulated,
            (_, Slot::Infinite(2)) => NoFiniteValue,
            (LibFn::Catalan, _) => value(Folded::int(0)),
            (LibFn::Harmonic, Slot::Infinite(t)) => value(Folded::InfinitePlus(
                InfDir::Pos,
                Q::new(BigInt::from(if t == 1 { 1 } else { -1 }), BigInt::from(2)),
                Q::zero(),
            )),
            _ => Untabulated,
        },
        LibFn::AssocLegendre => match (s(0), s(1), s(2), num(0), num(1)) {
            // (`m < −n`: a Ferrers function, not a polynomial; not tabulated.)
            (Slot::Finite, Slot::Finite, Slot::Infinite(t), Some(n), Some(m))
                if n.is_integer() && !n.is_negative() && m.is_integer() && *m >= -n =>
            {
                value(assoc_legendre_at_infinity(
                    &n.to_integer(),
                    &m.to_integer(),
                    t,
                ))
            }
            _ => Untabulated,
        },
        LibFn::EulerNumber
        | LibFn::Stirling1
        | LibFn::Stirling2
        | LibFn::BetaInc
        | LibFn::BetaIncRegularized => Untabulated,
    }
}

/// `Pₙᵐ(iᵗ·∞)` for integers `n ≥ 0` and `m`, by the definition `eval`
/// expands (Condon–Shortley phase, the principal `(1 − x²)^{m/2}`):
/// `Pₙᵏ(x) = (−1)ᵏ(2n)!/(2ⁿn!(n − k)!)·x^{n−k}(1 − x²)^{k/2} + …` for `k
/// ≥ 0` and `Pₙ^{−k} = (−1)ᵏ(n − k)!/(n + k)!·Pₙᵏ`, where `1 − x² → −∞`
/// on the real axis (`√(−∞) = i·∞`) and `→ +∞` on the imaginary one;
/// 0 for `|m| > n`.  `P₂¹(∞) = −i·∞`, `P₂²(∞) = −∞`, as `eval`'s
/// closed forms gave them before they stopped being taken at infinite
/// arguments.
fn assoc_legendre_at_infinity(n: &BigInt, m: &BigInt, t: u8) -> Folded {
    use num_integer::Integer;
    let k = m.abs();
    if k > *n {
        return Folded::int(0);
    }
    if n.is_zero() {
        return Folded::int(1);
    }
    let negative = k.is_odd() && !m.is_negative();
    let x_turns = u32::from(t) * u32::from(quarter_turns(&(n - &k)));
    let root_turns = if t.is_multiple_of(2) {
        u32::from(quarter_turns(&k))
    } else {
        0
    };
    let turns = x_turns + root_turns + if negative { 2 } else { 0 };
    directed((turns % 4) as u8)
}

/// The Bessel functions at `x = iᵗ·∞` for a finite order `nu` (a rational
/// when `Some`): see [`at_infinity`].  At `±∞` the values are those of
/// [`lib_function_value`] (`Jν(±∞) = Yν(±∞) = Kν(∞) = 0`, `Iν(∞) = ∞`,
/// `Iν(−∞) = e^{iπν}·∞`, `Kν(−∞) = −i·∞` for a rational order).
fn bessel_at_infinite_argument(f: crate::base::libfn::LibFn, nu: Option<&Q>, t: u8) -> AtInfinity {
    use crate::base::libfn::LibFn;
    let value = AtInfinity::Value;
    match (f, t) {
        (LibFn::BesselJ | LibFn::BesselY, 0 | 2) | (LibFn::BesselK, 0) => value(Folded::int(0)),
        (LibFn::BesselI | LibFn::BesselK, 1 | 3) => value(Folded::int(0)),
        (LibFn::BesselI, 0) => match nu {
            Some(_) => value(Folded::Infinity),
            None => AtInfinity::NoFiniteValue,
        },
        (LibFn::BesselI, _) => match nu {
            Some(nu) => value(bessel_i_at_minus_infinity(nu)),
            None => AtInfinity::NoFiniteValue,
        },
        (LibFn::BesselK, _) => match nu {
            Some(_) => value(Folded::ImaginaryInfinity(true)),
            None => AtInfinity::NoFiniteValue,
        },
        // J at ±i·∞: e^{±iνπ/2}·∞; Y: e^{±i(ν+1)π/2}·∞.
        _ => match nu {
            Some(nu) if nu.is_integer() => {
                let shift = if f == LibFn::BesselY { 1 } else { 0 };
                let turns = quarter_turns(&(nu.to_integer() + shift));
                value(directed(if t == 1 { turns } else { (4 - turns) % 4 }))
            }
            Some(_) => value(Folded::ComplexInfinity),
            None => AtInfinity::NoFiniteValue,
        },
    }
}

/// The rising (`rising`) or falling factorial of a finite `x` (a rational
/// when `Some`) at the order `iᵗ·∞`: see [`at_infinity`].  `(x)ν =
/// Γ(x + ν)/Γ(x)` grows as `Γ(x)⁻¹·∞` for `ν → ∞`, is 0 for `x` a pole
/// of `Γ` (`(−m)_ν = 0` off the poles of `Γ(ν − m)`) and tends to 0 along
/// the imaginary axis (`|Γ(x + iy)| ~ e^{−π|y|/2}`); towards `−∞` it meets
/// the poles of `Γ(x + ν)`.  `x^{(ν)} = Γ(x + 1)/Γ(x + 1 − ν)` tends to 0
/// for `ν → −∞` (`x` not a negative integer) and has no limit otherwise
/// (`1/Γ` grows between its zeros; mpmath `ff(2, 10⁶)` = 0 at the integers
/// only).
fn factorial_order_at_infinity(rising: bool, x: Option<&Q>, t: u8) -> AtInfinity {
    let value = AtInfinity::Value;
    let pole = |x: &Q| x.is_integer() && !x.is_positive();
    match (rising, t, x) {
        (true, 1 | 3, _) => value(Folded::int(0)),
        (true, 0 | 2, Some(x)) if pole(x) => value(Folded::int(0)),
        (true, 0, Some(x)) => value(if gamma_is_negative(x) {
            Folded::NegInfinity
        } else {
            Folded::Infinity
        }),
        (false, 2, Some(x)) if !(x.is_integer() && x.to_integer().is_negative()) => {
            value(Folded::int(0))
        }
        _ => AtInfinity::NoFiniteValue,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Set constructors (canonical)
// ═══════════════════════════════════════════════════════════════════════════

/// Build a canonical `Interval` node.
///
/// Handles degenerate cases:
/// - If both endpoints are numeric and start > end → EmptySet.
/// - If both endpoints are numeric and start == end:
///   - both closed → FiniteSet({start})
///   - any open → EmptySet
pub(crate) fn canon_interval(
    arena: &mut Arena,
    start: ExprId,
    end: ExprId,
    kind: crate::base::interval::IntervalKind,
) -> ExprId {
    tracing::debug!(
        "canon_interval: start={:?}, end={:?}, flags={}",
        start,
        end,
        kind.flag_bits()
    );

    // Try to compare endpoints numerically.
    if let (Some(s), Some(e)) = (arena.as_num(start), arena.as_num(end)) {
        let s = s.clone();
        let e = e.clone();
        use std::cmp::Ordering;
        match s.cmp(&e) {
            Ordering::Greater => {
                // start > end → empty set
                return arena.intern(ExprNode::EmptySet);
            }
            Ordering::Equal => {
                // start == end
                if kind.lower_open() || kind.upper_open() {
                    // Any open endpoint on a point interval → empty
                    return arena.intern(ExprNode::EmptySet);
                } else {
                    // Both closed, single point → FiniteSet({start})
                    return arena.intern(ExprNode::FiniteSet(smallvec![start]));
                }
            }
            Ordering::Less => {
                // start < end: valid interval, fall through
            }
        }
    }

    arena.intern(ExprNode::Interval(start, end, kind))
}

/// Build a canonical `FiniteSet` node.
///
/// Sorts elements by sort key and deduplicates.
/// An empty set of elements returns EmptySet.
pub(crate) fn canon_finite_set(arena: &mut Arena, elements: &[ExprId]) -> ExprId {
    tracing::debug!("canon_finite_set: {} elements", elements.len());

    if elements.is_empty() {
        return arena.intern(ExprNode::EmptySet);
    }

    // Deduplicate (ExprIds are hash-consed, so equality is ExprId equality).
    let mut deduped: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut seen = FxHashSet::default();
    for &elem in elements {
        if seen.insert(elem) {
            deduped.push(elem);
        }
    }

    // Sort by sort key for canonical ordering.
    deduped.sort_by(|&a, &b| arena.sort_key(a).cmp(arena.sort_key(b)));

    if deduped.len() == 1 {
        // A single-element set is still a FiniteSet — don't collapse.
    }

    arena.intern(ExprNode::FiniteSet(deduped))
}

/// Build a canonical `SetUnion` node.
///
/// Follows the LatticeOp pattern:
/// 1. Flatten nested SetUnion.
/// 2. Remove EmptySet children (identity for union).
/// 3. Short-circuit on UniversalSet (absorbing element for union).
/// 4. Deduplicate and sort children by sort key, then put the real
///    intervals in ascending order of their lower ends
///    ([`order_real_intervals`]): `(−∞, −2) ∪ (2, ∞)`, `(0, 1) ∪ [2, 3]`.
/// 5. Single child → return that child directly.
///
/// Up to 0.51 the order was the sort key's alone, which prints `(2, ∞) ∪
/// (−∞, −2)` (the key of an interval starts with its kind, and `−∞` sorts
/// after the numbers), so the set evaluator kept its ascending unions as
/// raw nodes, not `==` to the canonical union of the same parts: `(0, 1) ∪
/// [2, 3]` and its JSON round trip `[2, 3] ∪ (0, 1)` compared unequal.
pub(crate) fn canon_set_union(arena: &mut Arena, sets: &[ExprId]) -> ExprId {
    tracing::debug!("canon_set_union: {} children", sets.len());

    let mut flat: SmallVec<[ExprId; 4]> = SmallVec::new();

    // Flatten and filter.
    let mut stack: SmallVec<[ExprId; 8]> = sets.iter().copied().collect();
    while let Some(s) = stack.pop() {
        match arena.node(s).clone() {
            ExprNode::SetUnion(children) => {
                // Flatten nested union.
                for &child in &children {
                    stack.push(child);
                }
            }
            ExprNode::EmptySet => {
                // Identity element: skip.
            }
            ExprNode::UniversalSet => {
                // Absorbing element: union with UniversalSet = UniversalSet.
                return arena.intern(ExprNode::UniversalSet);
            }
            _ => {
                flat.push(s);
            }
        }
    }

    if flat.is_empty() {
        return arena.intern(ExprNode::EmptySet);
    }

    // Deduplicate.
    let mut seen = FxHashSet::default();
    flat.retain(|id| seen.insert(*id));

    // Sort by sort key.
    flat.sort_by(|&a, &b| arena.sort_key(a).cmp(arena.sort_key(b)));

    if flat.len() == 1 {
        return flat[0];
    }

    order_real_intervals(arena, &mut flat);
    arena.intern(ExprNode::SetUnion(flat))
}

/// The lower end of an interval in a union, for [`order_real_intervals`]:
/// `−∞`, a finite real number (its `f64` value, and the exact value of a
/// rational end), or `+∞`.
enum LowerEnd {
    NegInf,
    Finite(f64, Option<Q>),
    PosInf,
}

impl LowerEnd {
    /// The end `start` when it is `±∞`, a number, or a constant with a
    /// finite real value (`√2`, `π/2`, `ln 3`, from 17 digits of `evalf`);
    /// `None` for one with symbols or without a real value.
    fn of(arena: &Arena, start: ExprId) -> Option<LowerEnd> {
        match arena.node(start) {
            ExprNode::NegInfinity => Some(LowerEnd::NegInf),
            ExprNode::Infinity => Some(LowerEnd::PosInf),
            ExprNode::Num(_) => {
                let q = arena.as_num(start)?.clone();
                let huge = if q.is_negative() { f64::MIN } else { f64::MAX };
                let f = crate::base::numeric::ratio_to_f64(&q).unwrap_or(huge);
                Some(LowerEnd::Finite(f, Some(q)))
            }
            _ if arena.is_symbol_free(start) => {
                let f = arena.const_f64(start, 17)?;
                f.is_finite().then_some(LowerEnd::Finite(f, None))
            }
            _ => None,
        }
    }

    /// Ascending order: by value, and of two ends with one `f64` value the
    /// rational one first (exactly between two rationals).
    fn cmp_ascending(&self, other: &LowerEnd) -> Ordering {
        match (self, other) {
            (LowerEnd::NegInf, LowerEnd::NegInf) | (LowerEnd::PosInf, LowerEnd::PosInf) => {
                Ordering::Equal
            }
            (LowerEnd::NegInf, _) | (_, LowerEnd::PosInf) => Ordering::Less,
            (_, LowerEnd::NegInf) | (LowerEnd::PosInf, _) => Ordering::Greater,
            (LowerEnd::Finite(a, qa), LowerEnd::Finite(b, qb)) => {
                // `+ 0.0` makes `−0.0` equal to `0.0` under `total_cmp`
                (a + 0.0).total_cmp(&(b + 0.0)).then(match (qa, qb) {
                    (Some(x), Some(y)) => x.cmp(y),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => Ordering::Equal,
                })
            }
        }
    }
}

/// Put the real intervals among the sort-key-ordered `children` of a union
/// in ascending order of their lower ends ([`LowerEnd`]), in the places the
/// sort key gave them (all intervals are adjacent in that order: their keys
/// share the set rank and the interval sub-rank); the other children
/// (finite sets, image sets, intervals with symbolic ends) keep their
/// places.  Ties (equal lower ends, or two constants with one `f64` value)
/// go by sort key, so the order is a total order that depends on the parts
/// only.  For disjoint intervals the order of the lower ends is the order
/// on the line, which the set evaluator's normal form uses (intervals
/// ascending, then the finite set of isolated points).  The `f64` values
/// only order the parts of a set, which is the same set in any order.
fn order_real_intervals(arena: &Arena, children: &mut [ExprId]) {
    let intervals = children
        .iter()
        .filter(|&&c| matches!(arena.node(c), ExprNode::Interval(..)))
        .count();
    if intervals < 2 {
        return;
    }
    let mut slots: SmallVec<[usize; 4]> = SmallVec::new();
    let mut keyed: Vec<(LowerEnd, ExprId)> = Vec::with_capacity(intervals);
    for (i, &c) in children.iter().enumerate() {
        if let ExprNode::Interval(start, _, _) = *arena.node(c)
            && let Some(end) = LowerEnd::of(arena, start)
        {
            slots.push(i);
            keyed.push((end, c));
        }
    }
    if keyed.len() < 2 {
        return;
    }
    keyed.sort_by(|(ea, a), (eb, b)| {
        ea.cmp_ascending(eb)
            .then_with(|| arena.sort_key(*a).cmp(arena.sort_key(*b)))
    });
    for (slot, (_, c)) in slots.into_iter().zip(keyed) {
        children[slot] = c;
    }
}

/// Build a canonical `SetIntersection` node.
///
/// Follows the LatticeOp pattern:
/// 1. Flatten nested SetIntersection.
/// 2. Remove UniversalSet children (identity for intersection).
/// 3. Short-circuit on EmptySet (absorbing element for intersection).
/// 4. Deduplicate and sort children by sort key.
/// 5. Single child → return that child directly.
pub(crate) fn canon_set_intersection(arena: &mut Arena, sets: &[ExprId]) -> ExprId {
    tracing::debug!("canon_set_intersection: {} children", sets.len());

    let mut flat: SmallVec<[ExprId; 4]> = SmallVec::new();

    // Flatten and filter.
    let mut stack: SmallVec<[ExprId; 8]> = sets.iter().copied().collect();
    while let Some(s) = stack.pop() {
        match arena.node(s).clone() {
            ExprNode::SetIntersection(children) => {
                // Flatten nested intersection.
                for &child in &children {
                    stack.push(child);
                }
            }
            ExprNode::UniversalSet => {
                // Identity element: skip.
            }
            ExprNode::EmptySet => {
                // Absorbing element: intersection with EmptySet = EmptySet.
                return arena.intern(ExprNode::EmptySet);
            }
            _ => {
                flat.push(s);
            }
        }
    }

    if flat.is_empty() {
        // Intersection of no sets: by convention, return UniversalSet
        // (the identity element).
        return arena.intern(ExprNode::UniversalSet);
    }

    // Deduplicate.
    let mut seen = FxHashSet::default();
    flat.retain(|id| seen.insert(*id));

    // Sort by sort key.
    flat.sort_by(|&a, &b| arena.sort_key(a).cmp(arena.sort_key(b)));

    if flat.len() == 1 {
        return flat[0];
    }

    arena.intern(ExprNode::SetIntersection(flat))
}

// ═══════════════════════════════════════════════════════════════════════════
// Verify canonical form
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that an expression is in canonical form.
///
/// Returns a list of violations found. An empty list means the
/// expression is properly canonical.
///
/// Every node reachable from `id` is checked once (the walk is over the
/// hash-consed DAG with a visited set, using an explicit stack), so the
/// cost is proportional to the DAG, not the unfolded tree.
///
/// This is intended for use in `debug_assert!` and property-based tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn verify_canonical(arena: &mut Arena, id: ExprId) -> Vec<String> {
    let mut errors = Vec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: Vec<ExprId> = vec![id];
    while let Some(cur) = stack.pop() {
        if !visited.insert(cur) {
            continue;
        }
        verify_node(arena, cur, &mut errors);
        arena.node(cur).for_each_child(|c| stack.push(c));
    }
    errors
}

/// Check the canonical-form invariants of the node `id` itself (not its
/// descendants).  Constructors call this in debug builds on the node they
/// just built: the children were verified when *they* were built.
pub(crate) fn verify_canonical_shallow(arena: &mut Arena, id: ExprId) -> Vec<String> {
    let mut errors = Vec::new();
    verify_node(arena, id, &mut errors);
    errors
}

/// The per-node invariants behind [`verify_canonical`]; does not recurse.
fn verify_node(arena: &mut Arena, id: ExprId, errors: &mut Vec<String>) {
    match arena.node(id).clone() {
        ExprNode::Add(ref children) => {
            // 1. Must have >= 2 children
            if children.len() < 2 {
                errors.push(format!("Add with {} children (need >= 2)", children.len()));
            }
            // 2. No child should be zero
            for (i, &child) in children.iter().enumerate() {
                if child == arena.zero {
                    errors.push(format!("Add child {i} is zero"));
                }
            }
            // 3. No child should be an Add (must be flattened)
            for (i, &child) in children.iter().enumerate() {
                if matches!(arena.node(child), ExprNode::Add(_)) {
                    errors.push(format!("Add child {i} is nested Add (not flattened)"));
                }
            }
            // 4. Children must be sorted by SortKey
            for i in 1..children.len() {
                let key_prev = arena.sort_key(children[i - 1]);
                let key_curr = arena.sort_key(children[i]);
                if key_prev > key_curr {
                    errors.push(format!(
                        "Add children {}/{} not sorted: {:?} > {:?}",
                        i - 1,
                        i,
                        key_prev,
                        key_curr
                    ));
                }
            }
            // 5. No two children should have the same term key
            //    (like terms should be merged)
            {
                let mut seen_keys = FxHashSet::default();
                for (idx, &child) in children.iter().enumerate() {
                    let (_, key) = arena.as_coeff_term(child);
                    if !seen_keys.insert(key) {
                        errors.push(format!(
                            "Add child {idx} has duplicate term key (like terms not merged)"
                        ));
                    }
                }
            }
        }
        ExprNode::Mul(ref children) => {
            // 1. Must have >= 2 children
            if children.len() < 2 {
                errors.push(format!("Mul with {} children (need >= 2)", children.len()));
            }
            // 2. No child should be one
            for (i, &child) in children.iter().enumerate() {
                if child == arena.one {
                    errors.push(format!("Mul child {i} is one"));
                }
            }
            // 3. No child should be a Mul (must be flattened)
            for (i, &child) in children.iter().enumerate() {
                if matches!(arena.node(child), ExprNode::Mul(_)) {
                    errors.push(format!("Mul child {i} is nested Mul (not flattened)"));
                }
            }
            // 4. Children must be sorted by SortKey
            for i in 1..children.len() {
                let key_prev = arena.sort_key(children[i - 1]);
                let key_curr = arena.sort_key(children[i]);
                if key_prev > key_curr {
                    errors.push(format!("Mul children {}/{} not sorted", i - 1, i));
                }
            }
            // 5. At most one Num child
            let num_count = children
                .iter()
                .filter(|&&c| matches!(arena.node(c), ExprNode::Num(_)))
                .count();
            if num_count > 1 {
                errors.push(format!(
                    "Mul has {num_count} Num children (should be at most 1)"
                ));
            }
        }
        ExprNode::Neg(inner) => {
            // 1. No double negation
            if matches!(arena.node(inner), ExprNode::Neg(_)) {
                errors.push("Double negation Neg(Neg(...))".to_string());
            }
            // 2. Inner should not be zero
            if inner == arena.zero {
                errors.push("Neg(0) should be 0".to_string());
            }
            // 3. Inner should not be a Num (should be folded into negative Num)
            if matches!(arena.node(inner), ExprNode::Num(_)) {
                errors.push("Neg(Num) should be folded into negative Num".to_string());
            }
        }
        ExprNode::Pow(base, exp) => {
            // 1. exp should not be 0 (should be 1)
            if exp == arena.zero {
                errors.push("Pow(x, 0) should be 1".to_string());
            }
            // 2. exp should not be 1 (should be base)
            if exp == arena.one {
                errors.push("Pow(x, 1) should be x".to_string());
            }
            // 3. base should not be 1 (should be 1)
            if base == arena.one {
                errors.push("Pow(1, x) should be 1".to_string());
            }
        }
        ExprNode::And(ref children) | ExprNode::Or(ref children) => {
            let label = if matches!(arena.node(id), ExprNode::And(_)) {
                "And"
            } else {
                "Or"
            };
            // 1. Must have >= 2 children
            if children.len() < 2 {
                errors.push(format!(
                    "{} with {} children (need >= 2)",
                    label,
                    children.len()
                ));
            }
            // 2. Children must be sorted by SortKey
            for i in 1..children.len() {
                let key_prev = arena.sort_key(children[i - 1]);
                let key_curr = arena.sort_key(children[i]);
                if key_prev > key_curr {
                    errors.push(format!(
                        "{} children {}/{} not sorted: {:?} > {:?}",
                        label,
                        i - 1,
                        i,
                        key_prev,
                        key_curr
                    ));
                }
            }
        }
        // Atoms and functions carry no ordering invariants of their own.
        _ => {}
    }
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
