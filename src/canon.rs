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
//! 4. Remaining terms sorted by [`SortKey`].
//! 5. Zero-coefficient terms dropped.
//! 6. `NaN` propagation: any `NaN` term ⟹ result is `NaN`.
//!
//! ## Mul
//!
//! 1. Flatten nested `Mul` (explicit stack, no recursion).
//! 2. Collect running numeric coefficient.
//! 3. Combine like bases: `x * x → x²`, `x² * x³ → x⁵`.
//! 4. Numeric coefficient placed first if ≠ 1.
//! 5. Remaining factors sorted by [`SortKey`].
//! 6. Zero propagation: any zero factor ⟹ result is `0` (unless ∞ involved ⟹ `NaN`).
//! 7. `NaN` propagation.
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

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Pow as NumPow, Signed, Zero};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{SmallVec, smallvec};

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};

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

    // Running numeric constant (the "coefficient of 1").
    let mut constant: Ratio<BigInt> = Ratio::zero();

    // Map: symbolic_key → accumulated coefficient.
    let mut terms: FxHashMap<ExprId, Ratio<BigInt>> = FxHashMap::default();

    // Track whether we've seen infinity / neg-infinity to handle oo − oo → NaN.
    let mut has_pos_inf = false;
    let mut has_neg_inf = false;
    let mut has_zoo = false;

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
                let neg_one: Ratio<BigInt> = -Ratio::one();
                let (c, key) = arena.as_coeff_term(inner);
                let combined = neg_one * c;
                if key == arena.one {
                    constant += combined;
                } else {
                    *terms.entry(key).or_insert_with(Ratio::zero) += combined;
                }
            }

            _ => {
                let (c, key) = arena.as_coeff_term(id);
                if key == arena.one {
                    constant += c;
                } else {
                    *terms.entry(key).or_insert_with(Ratio::zero) += c;
                }
            }
        }
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

    // Collect non-zero terms.
    let non_zero_terms: SmallVec<[(ExprId, Ratio<BigInt>); 8]> =
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
        let errors = verify_canonical(arena, result);
        if !errors.is_empty() {
            tracing::debug!("canon_add: non-canonical result: {:?}", errors);
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Mul
// ═══════════════════════════════════════════════════════════════════════════

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

    // Running numeric coefficient.
    let mut coeff: Ratio<BigInt> = Ratio::one();

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
                        return handle_mul_with_zoo(arena, &mut stack, &coeff);
                    }
                    _ => {} // Infinity — no sign change.
                }
            }

            ExprNode::Num(nid) => {
                let val = arena.num(nid).clone();
                coeff *= val;
                if coeff.is_zero() {
                    // 0 * anything: check for infinity → NaN.
                    if saw_infinity {
                        return arena.nan;
                    }
                    // Short-circuit: the rest doesn't matter.
                    return arena.zero;
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
        if saw_infinity {
            return arena.nan;
        }
        return arena.zero;
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

    for (base, exp) in factors {
        if exp == arena.one {
            // If the base is itself a numeric literal, absorb it into the
            // running coefficient instead of adding a second Num child.
            if let Some(val) = arena.as_num(base) {
                coeff *= val.clone();
                continue;
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
            result_args.push(pow_id);
        }
    }

    // Re-check for zero after absorbing numeric factors.
    if coeff.is_zero() {
        if saw_infinity {
            return arena.nan;
        }
        return arena.zero;
    }

    // Now prepend the numeric coefficient (if not 1, or if there are
    // no symbolic factors left).
    if !coeff.is_one() || result_args.is_empty() {
        let nid = arena.intern_num(coeff.clone());
        result_args.insert(0, arena.intern(ExprNode::Num(nid)));
    }

    // If infinity was seen and coefficient is nonzero.
    if saw_infinity {
        if coeff.is_negative() {
            return arena.neg_infinity;
        }
        return arena.infinity;
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
        verify_canonical(arena, result).is_empty(),
        "canon_mul produced non-canonical result: {:?}",
        verify_canonical(arena, result)
    );
    result
}

/// Helper for `Mul` when `zoo` (ComplexInfinity) is encountered.
///
/// `zoo * nonzero → zoo`, `zoo * 0 → NaN`.
fn handle_mul_with_zoo(
    arena: &mut Arena,
    remaining: &mut SmallVec<[ExprId; 16]>,
    coeff: &Ratio<BigInt>,
) -> ExprId {
    if coeff.is_zero() {
        return arena.nan;
    }
    // Drain remaining to check for zeros or NaN.
    while let Some(id) = remaining.pop() {
        match arena.node(id).clone() {
            ExprNode::NaN => return arena.nan,
            ExprNode::Num(nid) => {
                if arena.num(nid).is_zero() {
                    return arena.nan;
                }
            }
            ExprNode::Mul(children) => {
                remaining.extend_from_slice(&children);
            }
            _ => {}
        }
    }
    arena.intern(ExprNode::ComplexInfinity)
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

    // Pow(Pow(a, b), c) → Pow(a, b*c) when both b and c are integers.
    // This is mathematically safe for integer exponents (no branch cut issues).
    // Critical for the Gruntz algorithm: ensures 1/(1/x) = Pow(Pow(x,-1),-1) → x.
    // Matches SymPy's auto-simplification of nested powers at construction time.
    if let ExprNode::Pow(inner_base, inner_exp) = arena.node(base).clone()
        && let (Some(b), Some(c)) = (arena.as_num(inner_exp), arena.as_num(exp)) {
            let b = b.clone();
            let c = c.clone();
            if b.is_integer() && c.is_integer() {
                tracing::trace!("canon_pow: flattening Pow(Pow(a,b),c) → Pow(a,b*c)");
                let product = b * c;
                let prod_id = arena.intern_num(product);
                let prod_expr = arena.intern(ExprNode::Num(prod_id));
                return canon_pow(arena, inner_base, prod_expr);
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
        return match r {
            0 => arena.one,
            1 => arena.i_unit,
            2 => arena.neg_one,
            3 => arena.neg(arena.i_unit),
            _ => unreachable!(),
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
            return match r {
                0 => arena.one,
                1 => arena.i_unit,
                2 => arena.neg_one,
                3 => arena.neg(arena.i_unit),
                _ => unreachable!(),
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
    // 0^0 was caught above (exp==zero). 0^negative → zoo? We leave
    // it unevaluated for safety.

    // Both numeric → try to evaluate.
    if let (Some(b), Some(e)) = (arena.as_num(base).cloned(), arena.as_num(exp).cloned())
        && let Some(result) = eval_numeric_pow(arena, &b, &e)
    {
        return result;
    }

    let result = arena.intern(ExprNode::Pow(base, exp));
    #[cfg(debug_assertions)]
    {
        let errors = verify_canonical(arena, result);
        if !errors.is_empty() {
            tracing::debug!("canon_pow: non-canonical result: {:?}", errors);
        }
    }
    result
}

/// Try to evaluate `b ^ e` when both are rational, returning `None` if the
/// exponent is not a suitably small integer.
fn eval_numeric_pow(arena: &mut Arena, b: &Ratio<BigInt>, e: &Ratio<BigInt>) -> Option<ExprId> {
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
    let digit_count = result.numer().to_string().len() + result.denom().to_string().len();
    if digit_count > arena.config.max_result_digits {
        return None;
    }

    let nid = arena.intern_num(result);
    Some(arena.intern(ExprNode::Num(nid)))
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
        let errors = verify_canonical(arena, result);
        if !errors.is_empty() {
            tracing::debug!("canon_neg: non-canonical result: {:?}", errors);
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Verify canonical form
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that an expression is in canonical form.
///
/// Returns a list of violations found. An empty list means the
/// expression is properly canonical.
///
/// This is intended for use in `debug_assert!` and property-based tests.
pub(crate) fn verify_canonical(arena: &mut Arena, id: ExprId) -> Vec<String> {
    let mut errors = Vec::new();
    verify_node(arena, id, &mut errors);
    errors
}

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
            // Recurse into children
            for &child in children {
                verify_node(arena, child, errors);
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
            // Recurse
            for &child in children {
                verify_node(arena, child, errors);
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
            verify_node(arena, inner, errors);
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
            verify_node(arena, base, errors);
            verify_node(arena, exp, errors);
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
            // Recurse into children
            for &child in children {
                verify_node(arena, child, errors);
            }
        }
        // Atoms and functions: recurse into children
        _ => {
            for &child in arena.node(id).children().iter() {
                verify_node(arena, child, errors);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

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
    fn pow_non_integer_exp_stays_unevaluated() {
        let mut a = Arena::new();
        let base = a.int(4);
        let exp = a.rational(1, 2);
        let result = a.pow(base, exp);
        // 4^(1/2) should NOT evaluate to 2 — that's simplification, not
        // canonicalization.
        assert_eq!(display(&a, result), "sqrt(4)");
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
        assert_eq!(display(&a, result), "x*1/y");
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
    fn no_auto_eval_sin() {
        // sin(0) should NOT auto-evaluate to 0.
        let mut a = Arena::new();
        let result = a.sin(a.zero);
        assert_eq!(display(&a, result), "sin(0)");
    }

    #[test]
    fn no_auto_eval_cos_pi() {
        // cos(pi) should NOT auto-evaluate to -1.
        let mut a = Arena::new();
        let result = a.cos(a.pi);
        assert_eq!(display(&a, result), "cos(pi)");
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
        assert_eq!(display(&a, result), "sqrt(4)*I");
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
        assert_eq!(display(&a, result), "sqrt(9)*I");
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
}
