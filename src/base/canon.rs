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
//! 7. Infinities absorb finite terms; a directed infinity `x·∞` whose
//!    coefficients cancel is `NaN` (see [`add_directed_infinities`]).
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
//!    `f(x)`) count as finite.
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
use crate::base::numeric::Q;

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
            let sum = na.clone() + nb.clone();
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
            *slot.get_mut() += c;
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
                constant += arena.num(nid).clone();
            }

            ExprNode::Neg(inner) => {
                // −x has coefficient −1, term x.
                let neg_one: Q = -Ratio::one();
                let (c, key) = arena.as_coeff_term(inner);
                let combined = neg_one * c;
                if key == arena.one {
                    constant += combined;
                } else {
                    collect(&mut terms, key, combined);
                }
            }

            _ => {
                let (c, key) = arena.as_coeff_term(id);
                if key == arena.one {
                    constant += c;
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

    // A directed infinity `x·∞` is an infinite term too.
    if terms.keys().any(|&k| is_directed_infinity(arena, k)) {
        return add_directed_infinities(arena, &terms, has_pos_inf, has_neg_inf, has_zoo);
    }

    // Handle infinities: if we saw oo / -oo / zoo, they dominate finite terms.
    if has_pos_inf {
        return arena.infinity;
    }
    if has_neg_inf {
        return arena.neg_infinity;
    }
    if has_zoo {
        return arena.complex_infinity;
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
    let mut seen: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut memo = EverywhereMemo::default();
    for &key in keys {
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
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

/// The sum of collected `terms` (coefficient per key) of which at least one
/// is a directed infinity `x·∞` (see [`directed_infinity`]), plus the bare
/// infinities seen.  A symbol is finite, so `x·∞` is `nan` (at `x = 0`) or
/// infinite in the direction of `x`:
///
/// * finite terms are absorbed (`x·∞ + 1 = x·∞`), as by a bare `±∞`;
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
    has_pos_inf: bool,
    has_neg_inf: bool,
    has_zoo: bool,
) -> ExprId {
    if has_zoo {
        return arena.nan;
    }
    let mut out: SmallVec<[ExprId; 6]> = SmallVec::new();
    for (&key, c) in terms {
        if !is_directed_infinity(arena, key) {
            continue;
        }
        if c.is_zero() {
            tracing::debug!("canon_add: directed infinities cancel → nan");
            return arena.nan;
        }
        // `canon_mul` keeps only the sign of the coefficient.
        out.push(arena.make_coeff_term(c.clone(), key));
    }
    if has_pos_inf {
        out.push(arena.infinity);
    }
    if has_neg_inf {
        out.push(arena.neg_infinity);
    }
    if out.len() == 1 {
        return out[0];
    }
    out.sort_by(|a, b| arena.sort_key(*a).cmp(arena.sort_key(*b)));
    let result = arena.intern(ExprNode::Add(out));
    debug_assert!(
        verify_canonical_shallow(arena, result).is_empty(),
        "canon_add: non-canonical sum of directed infinities: {:?}",
        verify_canonical_shallow(arena, result)
    );
    result
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
///   they are infinite (`ln(0)`, `exp(∞)`, `Γ(−1)`) were folded to `zoo`/`∞`
///   when interned ([`canon_function`]); any other application (`Γ(zoo)`,
///   `f(x)`) is treated as finite here;
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
///   1))` was `0` (`0·nan`; SymPy 1.14 gives `0`, deliberately different).
///
/// `saw_infinity` reports whether an infinity was already consumed from the
/// factor list before the coefficient became zero; `remaining` are the
/// other factors (bases with a positive exponent among them), `denominators`
/// the bases collected with a negative exponent.
fn zero_times_rest<'a>(
    arena: &mut Arena,
    saw_infinity: bool,
    remaining: impl IntoIterator<Item = &'a ExprId>,
    denominators: impl IntoIterator<Item = ExprId>,
) -> ExprId {
    if saw_infinity {
        return arena.nan;
    }
    let mut denominators: SmallVec<[ExprId; 4]> = denominators.into_iter().collect();
    let mut applications: SmallVec<[ExprId; 4]> = SmallVec::new();
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
            node if outside && application_arguments(arena, node).is_some() => {
                applications.push(id);
            }
            _ => {}
        }
    }
    let mut memo = EverywhereMemo::default();
    for d in denominators {
        if memo.zero_or_undefined(arena, d) {
            tracing::debug!("canon_mul: 0 over an identically zero or undefined denominator → nan");
            return arena.nan;
        }
    }
    for f in applications {
        if memo.infinite_or_undefined(arena, f) {
            tracing::debug!("canon_mul: 0 times a function undefined at every point → nan");
            return arena.nan;
        }
    }
    arena.zero
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
        let zero = !arena.node(f).is_atom() && factor_vanishes(arena, f);
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
/// [`everywhere`] looks: the elementary and special functions, library and
/// user functions (`Apply`), and a power with a symbolic exponent.
fn application_arguments(arena: &Arena, node: &ExprNode) -> Option<SmallVec<[ExprId; 2]>> {
    match node {
        ExprNode::Floor(a)
        | ExprNode::Ceiling(a)
        | ExprNode::Sin(a)
        | ExprNode::Cos(a)
        | ExprNode::Tan(a)
        | ExprNode::Exp(a)
        | ExprNode::Ln(a)
        | ExprNode::Abs(a)
        | ExprNode::Asin(a)
        | ExprNode::Acos(a)
        | ExprNode::Atan(a)
        | ExprNode::Sinh(a)
        | ExprNode::Cosh(a)
        | ExprNode::Tanh(a)
        | ExprNode::Asinh(a)
        | ExprNode::Acosh(a)
        | ExprNode::Atanh(a)
        | ExprNode::Sign(a)
        | ExprNode::Gamma(a)
        | ExprNode::LogGamma(a)
        | ExprNode::Digamma(a)
        | ExprNode::Erf(a)
        | ExprNode::Erfc(a)
        | ExprNode::LambertW(a)
        | ExprNode::Si(a)
        | ExprNode::Ci(a)
        | ExprNode::Ei(a)
        | ExprNode::Li(a)
        | ExprNode::Zeta(a)
        | ExprNode::Factorial(a) => Some(smallvec![*a]),
        ExprNode::Atan2(a, b)
        | ExprNode::Beta(a, b)
        | ExprNode::Polygamma(a, b)
        | ExprNode::Binomial(a, b) => Some(smallvec![*a, *b]),
        ExprNode::Pow(b, e) if arena.as_num(*e).is_none() => Some(smallvec![*b, *e]),
        ExprNode::Apply(_, args) => Some(args.iter().copied().collect()),
        _ => None,
    }
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
pub(crate) fn everywhere_values(arena: &mut Arena, ids: &[ExprId]) -> SmallVec<[Everywhere; 4]> {
    let mut memo = EverywhereMemo::default();
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

/// The bases collected in `canon_mul`: those without and those with a
/// negative numeric exponent.
fn split_bases(
    arena: &Arena,
    bases: &FxHashMap<ExprId, SmallVec<[ExprId; 4]>>,
) -> (SmallVec<[ExprId; 8]>, SmallVec<[ExprId; 4]>) {
    let mut positive: SmallVec<[ExprId; 8]> = SmallVec::new();
    let mut negative: SmallVec<[ExprId; 4]> = SmallVec::new();
    for (&b, exps) in bases {
        if exps
            .iter()
            .any(|&e| arena.as_num(e).is_some_and(Signed::is_negative))
        {
            negative.push(b);
        } else {
            positive.push(b);
        }
    }
    (positive, negative)
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
            let product = na.clone() * nb.clone();
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
                let val = arena.num(nid).clone();
                coeff *= val;
                if coeff.is_zero() {
                    // 0 × rest: `nan` if any infinity/NaN is involved
                    // (already seen, still on the stack, or hidden in a
                    // collected base), otherwise `0`.
                    let (positive, negative) = split_bases(arena, &bases);
                    return zero_times_rest(
                        arena,
                        saw_infinity,
                        stack.iter().chain(positive.iter()),
                        negative,
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
        let (positive, negative) = split_bases(arena, &bases);
        return zero_times_rest(arena, saw_infinity, positive.iter(), negative);
    }

    if cancels_vanishing_sum(arena, &bases) {
        tracing::debug!("canon_mul: s·s⁻¹ with s identically 0 → nan");
        return arena.nan;
    }

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
        // Map: exponent ExprId → (product of bases as Ratio, indices consumed)
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

                let entry = exp_groups
                    .entry(exp_id)
                    .or_insert_with(|| (Ratio::one(), 0));
                entry.0 *= &base_r;
                entry.1 += 1;
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
                coeff *= val.clone();
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
                coeff *= val.clone();
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
        return zero_times_rest(arena, saw_infinity, result_args.iter(), []);
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
    for &f in factors.iter().filter(|&&f| arena.as_num(f).is_none()) {
        match known_sign(arena, f) {
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
    matches!(arena.node(id), ExprNode::Mul(ch) if ch.contains(&arena.infinity))
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
            ExprNode::Sin(a)
            | ExprNode::Tan(a)
            | ExprNode::Sinh(a)
            | ExprNode::Tanh(a)
            | ExprNode::Asin(a)
            | ExprNode::Atan(a)
            | ExprNode::Asinh(a)
            | ExprNode::Atanh(a)
            | ExprNode::Abs(a)
            | ExprNode::Sign(a)
            | ExprNode::LambertW(a)
            | ExprNode::Floor(a)
            | ExprNode::Ceiling(a) => stack.push(*a),
            _ => {}
        }
    }
    sums
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
/// Only a base whose exponents have both signs is tested (a symbol at no
/// cost: it holds no sum).
fn cancels_vanishing_sum(
    arena: &mut Arena,
    bases: &FxHashMap<ExprId, SmallVec<[ExprId; 4]>>,
) -> bool {
    let mixed: SmallVec<[ExprId; 2]> = bases
        .iter()
        .filter(|&(&b, exps)| {
            exps.len() > 1
                && !arena.node(b).is_atom()
                && exps
                    .iter()
                    .any(|&e| arena.as_num(e).is_some_and(Signed::is_positive))
                && exps
                    .iter()
                    .any(|&e| arena.as_num(e).is_some_and(Signed::is_negative))
        })
        .map(|(&b, _)| b)
        .collect();
    mixed.into_iter().any(|b| factor_vanishes(arena, b))
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
pub(crate) fn factor_vanishes(arena: &mut Arena, f: ExprId) -> bool {
    vanishes_identically_as_factor(arena, f)
        || constant_vanishes(arena, f)
        || sum_under_zero_chain(arena, f).is_some_and(|s| s != f && constant_vanishes(arena, s))
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
            ExprNode::Sin(a)
            | ExprNode::Tan(a)
            | ExprNode::Sinh(a)
            | ExprNode::Tanh(a)
            | ExprNode::Asin(a)
            | ExprNode::Atan(a)
            | ExprNode::Asinh(a)
            | ExprNode::Atanh(a)
            | ExprNode::Abs(a)
            | ExprNode::Sign(a)
            | ExprNode::LambertW(a)
            | ExprNode::Floor(a)
            | ExprNode::Ceiling(a) => *a,
            _ => return None,
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
    if AssumptionCache::new().query(arena, f, Props::NONZERO) == Some(true) {
        return false;
    }
    if costly_to_evaluate(arena, f) && !arena.may_vanish_identically(f) {
        return false;
    }
    arena.vanishes_by_identity(f)
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
    let special = matches!(
        node,
        ExprNode::Gamma(_)
            | ExprNode::LogGamma(_)
            | ExprNode::Digamma(_)
            | ExprNode::Polygamma(..)
            | ExprNode::Zeta(_)
            | ExprNode::Beta(..)
            | ExprNode::Factorial(_)
            | ExprNode::Binomial(..)
            | ExprNode::Erf(_)
            | ExprNode::Erfc(_)
            | ExprNode::Ei(_)
            | ExprNode::Li(_)
            | ExprNode::Si(_)
            | ExprNode::Ci(_)
            | ExprNode::LambertW(_)
            | ExprNode::Apply(..)
    );
    if !special {
        return false;
    }
    if let ExprNode::Apply(sid, args) = node
        && matches!(
            arena.lib_fn(*sid),
            Some(
                LibFn::Legendre
                    | LibFn::ChebyshevT
                    | LibFn::ChebyshevU
                    | LibFn::Hermite
                    | LibFn::Laguerre
                    | LibFn::Gegenbauer
                    | LibFn::Jacobi
                    | LibFn::AssocLegendre
                    | LibFn::AssocLaguerre
            )
        )
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
        let product = b * c;
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
    if base == arena.nan || exp == arena.nan {
        return arena.nan;
    }

    // x^0 → 1, except for indeterminate forms ∞^0, (-∞)^0, zoo^0.
    if exp == arena.zero {
        if base == arena.infinity || base == arena.neg_infinity || base == arena.complex_infinity {
            return arena.nan;
        }
        return arena.one;
    }

    // x^1 → x.
    if exp == arena.one {
        return base;
    }

    // 1^x → 1.
    if base == arena.one {
        return arena.one;
    }

    // 0^(positive numeric) → 0.
    if base == arena.zero
        && let Some(e) = arena.as_num(exp)
        && e.is_positive()
    {
        return arena.zero;
    }
    // 0^(negative numeric) → ComplexInfinity (division by zero).
    if base == arena.zero
        && let Some(e) = arena.as_num(exp)
        && e.is_negative()
    {
        return arena.complex_infinity;
    }
    // 0^∞ → 0.
    if base == arena.zero && exp == arena.infinity {
        return arena.zero;
    }
    // 0^(-∞) → ComplexInfinity.
    if base == arena.zero && exp == arena.neg_infinity {
        return arena.complex_infinity;
    }
    // 0^zoo → NaN (indeterminate).
    if base == arena.zero && exp == arena.complex_infinity {
        return arena.nan;
    }
    // 0^z for a non-numeric exponent whose real part has a known sign.
    if base == arena.zero
        && let Some(value) = zero_power(arena, exp)
    {
        return value;
    }

    // ── Positive infinity base ─────────────────────────────────────
    // ∞^(positive numeric) → ∞.
    if base == arena.infinity
        && let Some(e) = arena.as_num(exp)
        && e.is_positive()
    {
        return arena.infinity;
    }
    // ∞^(negative numeric) → 0.
    if base == arena.infinity
        && let Some(e) = arena.as_num(exp)
        && e.is_negative()
    {
        return arena.zero;
    }
    // ∞^∞ → ∞.
    if base == arena.infinity && exp == arena.infinity {
        return arena.infinity;
    }
    // ∞^(-∞) → 0.
    if base == arena.infinity && exp == arena.neg_infinity {
        return arena.zero;
    }

    // ── Negative infinity base ─────────────────────────────────────
    // (-∞)^(positive integer) → ±∞ by parity.
    if base == arena.neg_infinity
        && let Some(e) = arena.as_num(exp)
        && e.is_integer()
        && e.is_positive()
    {
        let n: i64 = e.to_integer().try_into().unwrap_or(0);
        return if n % 2 == 0 {
            arena.infinity
        } else {
            arena.neg_infinity
        };
    }
    // (−∞)^q for a positive non-integer rational q: ∞ in the direction of
    // (−1)^q on the principal branch, `√(−∞) = i·∞`, `(−∞)^(1/3) =
    // (−1)^(1/3)·∞` (SymPy: `sqrt(-oo)` → `oo*I`, `(-oo)**Rational(1, 3)` →
    // `oo*(-1)**(1/3)`).  Before 0.30 the power stayed an atom.
    if base == arena.neg_infinity
        && let Some(e) = arena.as_num(exp)
        && e.is_positive()
        && !e.is_integer()
    {
        let unit = canon_pow(arena, arena.neg_one, exp);
        return canon_mul(arena, &[unit, arena.infinity]);
    }
    // (-∞)^(negative numeric) → 0 (magnitude → 0 regardless of direction).
    if base == arena.neg_infinity
        && let Some(e) = arena.as_num(exp)
        && e.is_negative()
    {
        return arena.zero;
    }
    // (-∞)^∞ → NaN, (-∞)^(-∞) → NaN (indeterminate: ∞ is not an integer).
    if base == arena.neg_infinity && (exp == arena.infinity || exp == arena.neg_infinity) {
        return arena.nan;
    }

    // ── Complex infinity (zoo) base ────────────────────────────────
    // zoo^(positive numeric) → zoo.
    if base == arena.complex_infinity
        && let Some(e) = arena.as_num(exp)
        && e.is_positive()
    {
        return arena.complex_infinity;
    }
    // zoo^(negative numeric) → 0.
    if base == arena.complex_infinity
        && let Some(e) = arena.as_num(exp)
        && e.is_negative()
    {
        return arena.zero;
    }
    // zoo^zoo, zoo^∞, zoo^(-∞) → NaN.
    if base == arena.complex_infinity
        && (exp == arena.complex_infinity || exp == arena.infinity || exp == arena.neg_infinity)
    {
        return arena.nan;
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

    let result = if exp_int.is_negative() {
        // b^(-n) = (denom^n) / (numer^n)
        if pow_n.is_zero() {
            // Would be division by zero → leave unevaluated.
            return None;
        }
        Ratio::new(pow_d, pow_n)
    } else {
        Ratio::new(pow_n, pow_d)
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
}

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
/// the same exponents for the same primes, memoised per thread.
///
/// Exact `k`-th powers are recognised first via an integer root, so
/// `√(p²)` folds even for huge primes `p`.
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
    let root = n.nth_root(k);
    if NumPow::pow(root.clone(), k) == *n {
        return (root, BigInt::one());
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
    ComplexInfinity,
    NaN,
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
                [a] if inverse_trig_pi_multiple(arena, a).is_some() => {
                    Arg::PiMultiple(q * inverse_trig_pi_multiple(arena, a).unwrap_or_default())
                }
                [a] if matches!(arena.node(a), ExprNode::Ln(w) if *w == arena.neg_one) => {
                    Arg::ImaginaryPiMultiple(q)
                }
                [a] if a == i => Arg::ImaginaryMultiple(q),
                [a, b] if (a == pi && b == i) || (a == i && b == pi) => Arg::ImaginaryPiMultiple(q),
                // A directed infinity keeps only the sign of its coefficient.
                [a, b] if a == i && b == arena.infinity => Arg::ImaginaryInfinity(q.is_negative()),
                _ => Arg::Other,
            }
        }
        _ => Arg::Other,
    }
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
        ExprNode::Atan(a) => {
            if a == arena.infinity {
                return Some(Q::new(BigInt::from(1), BigInt::from(2)));
            }
            if a == arena.neg_infinity {
                return Some(Q::new(BigInt::from(-1), BigInt::from(2)));
            }
            (a, &[(1, 1, 1, 4), (-1, 1, -1, 4)])
        }
        _ => return None,
    };
    let v = arena.as_num(a)?;
    table.iter().find_map(|&(p, q, r, s)| {
        (*v == Q::new(BigInt::from(p), BigInt::from(q)))
            .then(|| Q::new(BigInt::from(r), BigInt::from(s)))
    })
}

/// `q mod m` in `[0, m)`.
fn mod_rational(q: &Q, m: i64) -> Q {
    let m = Q::from_integer(BigInt::from(m));
    q - (q / &m).floor() * &m
}

/// `q` as a multiple of `1/n` (`q·n` when that is an integer).
fn in_units(q: &Q, n: i64) -> Option<i64> {
    let t = q * Q::from_integer(BigInt::from(n));
    if t.is_integer() {
        t.to_integer().to_i64()
    } else {
        None
    }
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
pub(crate) fn canon_function(arena: &mut Arena, node: &ExprNode) -> Option<ExprId> {
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
            ExprNode::Exp(x) => matches!(classify_arg(arena, x), Arg::ImaginaryPiMultiple(_)),
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
        ExprNode::Sin(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::PiMultiple(q) => sin_pi(&q)?,
            Arg::ImaginaryInfinity(neg) => Folded::ImaginaryInfinity(neg),
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Cos(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(1),
            Arg::PiMultiple(q) => cos_pi(&q)?,
            Arg::ImaginaryInfinity(_) => Folded::Infinity,
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Tan(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::PiMultiple(q) => tan_pi(&q)?,
            Arg::ImaginaryInfinity(neg) => {
                Folded::Imaginary(if neg { -Q::one() } else { Q::one() })
            }
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Exp(a) => match classify_arg(arena, a) {
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
        ExprNode::Ln(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::ComplexInfinity,
            Arg::Rational(q) if q.is_one() => Folded::int(0),
            Arg::E => Folded::int(1),
            Arg::Infinity | Arg::NegInfinity | Arg::ImaginaryInfinity(_) => Folded::Infinity,
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
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Cosh(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(1),
            // cosh(iπq) = cos(πq).
            Arg::ImaginaryPiMultiple(q) => cos_pi(&q)?,
            Arg::Infinity | Arg::NegInfinity => Folded::Infinity,
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Tanh(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            // tanh(iπq) = i·tan(πq).
            Arg::ImaginaryPiMultiple(q) => times_i(tan_pi(&q)?)?,
            Arg::Infinity => Folded::int(1),
            Arg::NegInfinity => Folded::int(-1),
            Arg::ComplexInfinity | Arg::NaN => Folded::NaN,
            _ => return None,
        },
        // SymPy: asin(oo) → -oo*I, asin(-oo) → oo*I, acos(oo) → oo*I,
        // acos(-oo) → -oo*I, atan(I) → oo*I, atan(-I) → -oo*I (before 0.30
        // the hunts recorded them as `zoo`, without their direction).
        ExprNode::Asin(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::Infinity => Folded::ImaginaryInfinity(true),
            Arg::NegInfinity => Folded::ImaginaryInfinity(false),
            Arg::ImaginaryInfinity(neg) => Folded::ImaginaryInfinity(neg),
            Arg::ComplexInfinity => Folded::ComplexInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Acos(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_one() => Folded::int(0),
            Arg::Infinity => Folded::ImaginaryInfinity(false),
            Arg::NegInfinity => Folded::ImaginaryInfinity(true),
            // π/2 ∓ i∞.
            Arg::ImaginaryInfinity(neg) => Folded::ImaginaryInfinity(!neg),
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
            Arg::ImaginaryInfinity(true) => Folded::NegInfinity,
            Arg::ImaginaryInfinity(false) => Folded::Infinity,
            Arg::ComplexInfinity => Folded::ComplexInfinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Acosh(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_one() => Folded::int(0),
            // ∞ ± iπ/2 at ±i∞.
            Arg::Infinity | Arg::NegInfinity | Arg::ImaginaryInfinity(_) => Folded::Infinity,
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
        ExprNode::Gamma(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if is_nonpositive_int(&q) => Folded::ComplexInfinity,
            Arg::Infinity => Folded::Infinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::LogGamma(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if is_nonpositive_int(&q) => Folded::Infinity,
            Arg::Rational(q) if q.is_one() || in_units(&q, 1) == Some(2) => Folded::int(0),
            Arg::Infinity => Folded::Infinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Digamma(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if is_nonpositive_int(&q) => Folded::ComplexInfinity,
            Arg::Infinity => Folded::Infinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Erf(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::Infinity => Folded::int(1),
            Arg::NegInfinity => Folded::int(-1),
            Arg::ImaginaryInfinity(neg) => Folded::ImaginaryInfinity(neg),
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Erfc(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(1),
            Arg::Infinity => Folded::int(0),
            Arg::NegInfinity => Folded::int(2),
            // 1 ∓ i∞.
            Arg::ImaginaryInfinity(neg) => Folded::ImaginaryInfinity(!neg),
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::LambertW(a) => match classify_arg(arena, a) {
            Arg::Rational(q) if q.is_zero() => Folded::int(0),
            Arg::E => Folded::int(1),
            Arg::Infinity => Folded::Infinity,
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
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Factorial(a) => match classify_arg(arena, a) {
            // n! = Γ(n + 1): poles at the negative integers.
            Arg::Rational(q) if q.is_integer() && q.is_negative() => Folded::ComplexInfinity,
            Arg::Infinity => Folded::Infinity,
            Arg::NaN => Folded::NaN,
            _ => return None,
        },
        ExprNode::Binomial(n, k) => match (classify_arg(arena, n), classify_arg(arena, k)) {
            (Arg::NaN, _) | (_, Arg::NaN) => Folded::NaN,
            (Arg::Rational(n), Arg::Rational(k)) if k.is_integer() => {
                // C(n, k) = Γ(n+1)/(Γ(k+1)·Γ(n−k+1)).  For n ≥ 0 an integer
                // the value is 0 unless 0 ≤ k ≤ n; for a non-integer n,
                // 1/Γ(k+1) = 0 at a negative k.  A negative integer n is left
                // alone (the pole of Γ(n+1) makes it a limit; see `eval`).
                let vanishes = if n.is_integer() {
                    !n.is_negative() && (k.is_negative() || k > n)
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
            _ => return None,
        },
        ExprNode::Apply(head, ref args) => lib_function_value(arena, head, args)?,
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
        Folded::ComplexInfinity => arena.complex_infinity,
        Folded::NaN => arena.nan,
    })
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
    is_nonpositive_int(a) != is_nonpositive_int(b) && !is_nonpositive_int(&(a + b))
}

/// The zeros, poles and `nan`s of the library functions at exact numbers
/// (their other values are left to `eval`): `J_ν(0) = I_ν(0) = 0` for
/// `ν > 0` and for every integer `ν ≠ 0` (`J_{−n} = (−1)ⁿ J_n`,
/// `I_{−n} = I_n`; before 0.30 `J_{−3}(0)` stayed), `zoo` for a negative
/// non-integer `ν` (`J_ν(z) ~ (z/2)^ν/Γ(ν + 1)`), as SymPy's
/// `besselj.eval`/`besseli.eval`, `Y_ν(0)` and `K_ν(0)` infinite (`−∞`/`∞`
/// for `ν = 0`, `zoo` otherwise, as SymPy's `bessely.eval`/`besselk.eval`),
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
    head: crate::base::node::SymbolId,
    args: &[ExprId],
) -> Option<Folded> {
    use crate::base::libfn::LibFn;
    // Cheap filter before the name lookup: some argument must be a number
    // or a special atom.
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
    let f = arena.lib_fn(head)?;
    if args.contains(&arena.nan) {
        return Some(Folded::NaN);
    }
    let num = |i: usize| args.get(i).and_then(|&a| arena.as_num(a));
    let is = |i: usize, v: i64| num(i).is_some_and(|q| *q == Q::from_integer(BigInt::from(v)));
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
        _ => return None,
    })
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
pub(crate) fn canon_interval(arena: &mut Arena, start: ExprId, end: ExprId, flags: u8) -> ExprId {
    use crate::base::node::{INTERVAL_LEFT_OPEN, INTERVAL_RIGHT_OPEN};

    tracing::debug!(
        "canon_interval: start={:?}, end={:?}, flags={}",
        start,
        end,
        flags
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
                if flags & (INTERVAL_LEFT_OPEN | INTERVAL_RIGHT_OPEN) != 0 {
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

    arena.intern(ExprNode::Interval(start, end, flags))
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
/// 4. Deduplicate and sort children by sort key.
/// 5. Single child → return that child directly.
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

    arena.intern(ExprNode::SetUnion(flat))
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

    #[test]
    fn add_oo_plus_finite_is_oo() {
        let mut a = Arena::new();
        let x = s(&mut a, "x");
        let result = a.add(&[a.infinity, x]);
        assert_eq!(result, a.infinity);
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
        let start = std::time::Instant::now();
        let result = a.add(&symbols);
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 500,
            "1000-term sum took {elapsed:?}, expected < 500ms"
        );
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
