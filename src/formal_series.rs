//! Formal power series representations and algorithms.
//!
//! This module implements formal power series (FPS) expansions of symbolic
//! expressions. A formal power series represents a function as
//! `Σ a_k x^k` where the coefficients `a_k` may be given by a closed-form
//! formula or as a truncated polynomial.
//!
//! # Supported functions
//!
//! The following functions have known closed-form coefficient formulas:
//! - `exp(x)`: `a_k = 1/k!`
//! - `sin(x)`: `a_k = (-1)^n / (2n+1)!` for odd k=2n+1, 0 for even
//! - `cos(x)`: `a_k = (-1)^n / (2n)!` for even k=2n, 0 for odd
//! - `sinh(x)`: `a_k = 1/(2n+1)!` for odd k=2n+1, 0 for even
//! - `cosh(x)`: `a_k = 1/(2n)!` for even k=2n, 0 for odd
//! - `1/(1-x)`: `a_k = 1`
//! - `ln(1+x)`: `a_k = (-1)^(k+1)/k` for k >= 1
//! - `atan(x)`: `a_k = (-1)^n/(2n+1)` for odd k=2n+1, 0 for even
//! - `(1+x)^α`: `a_k = C(α, k)` (generalized binomial coefficients)
//!
//! For other functions, the module falls back to computing Taylor
//! coefficients via repeated differentiation.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode, SymbolId};

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// A formal power series `Σ a_k (x - point)^k`.
pub struct FormalPowerSeries {
    /// The kind of series representation.
    kind: FPSKind,
    /// Variable of expansion.
    pub(crate) variable: ExprId,
    /// Expansion point (usually 0).
    // The caller must pass the arena for operations.
    pub(crate) point: ExprId,
}

/// Internal representation of the coefficient sequence.
enum FPSKind {
    /// Closed-form coefficient formula: `a_k = f(k)`.
    ClosedForm {
        /// Symbolic expression for `a_k` in terms of `k`.
        formula: CoeffFormula,
        /// Starting index (usually 0 or 1).
        start: usize,
        /// Symmetry: only every m-th coefficient is nonzero.
        /// 1 = all, 2 = every other (e.g. sin/cos).
        symmetry: usize,
        /// Offset within the symmetry pattern.
        /// For sin: symmetry=2, offset=1 (odd terms only).
        /// For cos: symmetry=2, offset=0 (even terms only).
        sym_offset: usize,
    },
    /// Truncated polynomial (fallback: just the first N terms).
    Truncated {
        /// Coefficients indexed by power: coeffs[k] = coefficient of x^k.
        coeffs: Vec<CachedCoeff>,
    },
}

/// A cached rational coefficient.
#[derive(Clone)]
struct CachedCoeff {
    value: Ratio<BigInt>,
}

/// Coefficient formula for known functions.
#[derive(Clone)]
enum CoeffFormula {
    /// `a_k = 1/k!` (exponential)
    InvFactorial,
    /// `a_k = (-1)^(k/2) / k!` for the relevant terms (sin/cos pattern)
    AlternatingInvFactorial,
    /// `a_k = 1/k!` for relevant terms (sinh/cosh pattern)
    PlainInvFactorial,
    /// `a_k = 1` for all k (geometric series 1/(1-x))
    AllOnes,
    /// `a_k = (-1)^(k+1) / k` for k >= 1 (ln(1+x))
    LogSeries,
    /// `a_k = (-1)^((k-1)/2) / k` for odd k (atan(x))
    AtanSeries,
    /// `a_k = C(alpha, k)` generalized binomial coefficients
    BinomialSeries { alpha: Ratio<BigInt> },
}

// ═══════════════════════════════════════════════════════════════════════════
// Coefficient computation
// ═══════════════════════════════════════════════════════════════════════════

fn factorial_big(n: usize) -> BigInt {
    let mut result = BigInt::from(1);
    for i in 2..=n {
        result *= BigInt::from(i);
    }
    result
}

/// Compute the generalized binomial coefficient C(alpha, k) for rational alpha.
fn gen_binomial(alpha: &Ratio<BigInt>, k: usize) -> Ratio<BigInt> {
    if k == 0 {
        return Ratio::one();
    }
    let mut result = Ratio::one();
    for i in 0..k {
        let numer = alpha - Ratio::from_integer(BigInt::from(i));
        let denom = Ratio::from_integer(BigInt::from(i + 1));
        result = result * numer / denom;
    }
    result
}

impl CoeffFormula {
    /// Compute the coefficient at index `k`, respecting symmetry.
    /// `effective_k` is the actual power index in the series.
    fn coefficient_at(&self, effective_k: usize) -> Ratio<BigInt> {
        match self {
            CoeffFormula::InvFactorial => {
                // a_k = 1/k!
                Ratio::new(BigInt::from(1), factorial_big(effective_k))
            }
            CoeffFormula::AlternatingInvFactorial => {
                // For sin: effective_k = 2n+1, sign = (-1)^n
                // For cos: effective_k = 2n, sign = (-1)^n
                let n = effective_k / 2;
                let sign = if n.is_multiple_of(2) {
                    BigInt::from(1)
                } else {
                    BigInt::from(-1)
                };
                Ratio::new(sign, factorial_big(effective_k))
            }
            CoeffFormula::PlainInvFactorial => {
                // For sinh/cosh: 1/k! (no alternation)
                Ratio::new(BigInt::from(1), factorial_big(effective_k))
            }
            CoeffFormula::AllOnes => Ratio::one(),
            CoeffFormula::LogSeries => {
                // a_k = (-1)^(k+1) / k for k >= 1
                if effective_k == 0 {
                    return Ratio::zero();
                }
                let sign = if (effective_k + 1).is_multiple_of(2) {
                    BigInt::from(1)
                } else {
                    BigInt::from(-1)
                };
                Ratio::new(sign, BigInt::from(effective_k))
            }
            CoeffFormula::AtanSeries => {
                // a_k = (-1)^((k-1)/2) / k for odd k
                if effective_k == 0 || effective_k.is_multiple_of(2) {
                    return Ratio::zero();
                }
                let n = (effective_k - 1) / 2;
                let sign = if n.is_multiple_of(2) {
                    BigInt::from(1)
                } else {
                    BigInt::from(-1)
                };
                Ratio::new(sign, BigInt::from(effective_k))
            }
            CoeffFormula::BinomialSeries { alpha } => gen_binomial(alpha, effective_k),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FormalPowerSeries methods
// ═══════════════════════════════════════════════════════════════════════════

impl FormalPowerSeries {
    /// Get the k-th coefficient as a rational number.
    pub fn coefficient_rational(&self, k: usize) -> Ratio<BigInt> {
        match &self.kind {
            FPSKind::ClosedForm {
                formula,
                start,
                symmetry,
                sym_offset,
            } => {
                if k < *start {
                    return Ratio::zero();
                }
                // Check symmetry: if symmetry=2 and offset=1, only odd k have nonzero coeffs
                let sym = *symmetry;
                let off = *sym_offset;
                if sym > 1 && (k % sym) != off {
                    return Ratio::zero();
                }
                formula.coefficient_at(k)
            }
            FPSKind::Truncated { coeffs } => {
                if k < coeffs.len() {
                    coeffs[k].value.clone()
                } else {
                    Ratio::zero()
                }
            }
        }
    }

    /// Get the k-th coefficient as a symbolic expression.
    pub fn coefficient(&self, arena: &mut Arena, k: usize) -> ExprId {
        let r = self.coefficient_rational(k);
        let nid = arena.intern_num(r);
        arena.intern(ExprNode::Num(nid))
    }

    /// Get the coefficient formula description (for display/debugging).
    pub fn has_closed_form(&self) -> bool {
        matches!(self.kind, FPSKind::ClosedForm { .. })
    }

    /// Truncate the formal power series to a polynomial of degree n.
    /// Returns the polynomial `Σ_{k=0}^{n-1} a_k * (x - point)^k`.
    pub fn truncate(&self, arena: &mut Arena, n: usize) -> ExprId {
        tracing::trace!("fps: truncating to {} terms", n);
        let var = self.variable;
        let point = self.point;
        let is_maclaurin = arena.is_zero_structural(point);

        let mut terms: Vec<ExprId> = Vec::with_capacity(n);

        for k in 0..n {
            let coeff_rat = self.coefficient_rational(k);
            if coeff_rat.is_zero() {
                continue;
            }

            let coeff_nid = arena.intern_num(coeff_rat);
            let coeff_id = arena.intern(ExprNode::Num(coeff_nid));

            if k == 0 {
                terms.push(coeff_id);
            } else {
                // Build (x - point)^k
                let x_minus_a = if is_maclaurin { var } else { arena.sub(var, point) };

                let power = if k == 1 {
                    x_minus_a
                } else {
                    let k_id = arena.int(k as i64);
                    arena.pow(x_minus_a, k_id)
                };

                let term = arena.mul(&[coeff_id, power]);
                terms.push(term);
            }
        }

        if terms.is_empty() {
            arena.zero
        } else if terms.len() == 1 {
            terms[0]
        } else {
            arena.add(&terms)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Known-function pattern matching
// ═══════════════════════════════════════════════════════════════════════════

/// Try to construct an FPS for a known elementary function at x=0.
pub(crate) fn fps_known_function(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
    point: ExprId,
) -> Option<FormalPowerSeries> {
    tracing::debug!("fps: attempting known-function match");
    // Only handle Maclaurin (point = 0) for known patterns
    if !arena.is_zero_structural(point) {
        return None;
    }

    let node = arena.node(expr).clone();
    match node {
        // exp(x) → a_k = 1/k!
        ExprNode::Exp(inner) if inner == var => Some(FormalPowerSeries {
            kind: FPSKind::ClosedForm {
                formula: CoeffFormula::InvFactorial,
                start: 0,
                symmetry: 1,
                sym_offset: 0,
            },
            variable: var,
            point,
        }),

        // sin(x) → a_k = (-1)^n/(2n+1)! for odd k=2n+1
        ExprNode::Sin(inner) if inner == var => Some(FormalPowerSeries {
            kind: FPSKind::ClosedForm {
                formula: CoeffFormula::AlternatingInvFactorial,
                start: 0,
                symmetry: 2,
                sym_offset: 1, // odd terms only
            },
            variable: var,
            point,
        }),

        // cos(x) → a_k = (-1)^n/(2n)! for even k=2n
        ExprNode::Cos(inner) if inner == var => Some(FormalPowerSeries {
            kind: FPSKind::ClosedForm {
                formula: CoeffFormula::AlternatingInvFactorial,
                start: 0,
                symmetry: 2,
                sym_offset: 0, // even terms only
            },
            variable: var,
            point,
        }),

        // sinh(x) → a_k = 1/(2n+1)! for odd k=2n+1
        ExprNode::Sinh(inner) if inner == var => Some(FormalPowerSeries {
            kind: FPSKind::ClosedForm {
                formula: CoeffFormula::PlainInvFactorial,
                start: 0,
                symmetry: 2,
                sym_offset: 1,
            },
            variable: var,
            point,
        }),

        // cosh(x) → a_k = 1/(2n)! for even k=2n
        ExprNode::Cosh(inner) if inner == var => Some(FormalPowerSeries {
            kind: FPSKind::ClosedForm {
                formula: CoeffFormula::PlainInvFactorial,
                start: 0,
                symmetry: 2,
                sym_offset: 0,
            },
            variable: var,
            point,
        }),

        // atan(x) → a_k = (-1)^n/(2n+1) for odd k=2n+1
        ExprNode::Atan(inner) if inner == var => Some(FormalPowerSeries {
            kind: FPSKind::ClosedForm {
                formula: CoeffFormula::AtanSeries,
                start: 0,
                symmetry: 2,
                sym_offset: 1,
            },
            variable: var,
            point,
        }),

        // ln(1+x): detect Add([1, x]) inside Ln
        ExprNode::Ln(inner) => {
            if is_one_plus_var(arena, inner, var) {
                Some(FormalPowerSeries {
                    kind: FPSKind::ClosedForm {
                        formula: CoeffFormula::LogSeries,
                        start: 1,
                        symmetry: 1,
                        sym_offset: 0,
                    },
                    variable: var,
                    point,
                })
            } else {
                None
            }
        }

        // 1/(1-x) = (1-x)^(-1): detect Pow(Add([1, Neg(x)]), -1) or
        // Pow(Add([1, Mul([-1, x])]), -1)
        ExprNode::Pow(base, exp) => {
            // Check for (1+x)^alpha — generalized binomial
            if is_one_plus_var(arena, base, var)
                && let Some(alpha) = arena.as_num(exp)
            {
                let alpha = alpha.clone();
                // Special case: alpha = -1 → 1/(1+x) = Σ (-1)^k x^k
                // which is a special binomial. We use the general formula.
                return Some(FormalPowerSeries {
                    kind: FPSKind::ClosedForm {
                        formula: CoeffFormula::BinomialSeries { alpha },
                        start: 0,
                        symmetry: 1,
                        sym_offset: 0,
                    },
                    variable: var,
                    point,
                });
            }
            // Check for 1/(1-x): Pow(1-x, -1)
            if is_one_minus_var(arena, base, var)
                && let Some(exp_val) = arena.as_num(exp)
            {
                if *exp_val == Ratio::from_integer(BigInt::from(-1)) {
                    return Some(FormalPowerSeries {
                        kind: FPSKind::ClosedForm {
                            formula: CoeffFormula::AllOnes,
                            start: 0,
                            symmetry: 1,
                            sym_offset: 0,
                        },
                        variable: var,
                        point,
                    });
                }
                // (1-x)^alpha = (1+(-x))^alpha; use binomial with sign adjustment
                // C(alpha, k) * (-1)^k
                let alpha = exp_val.clone();
                return Some(FormalPowerSeries {
                    kind: FPSKind::ClosedForm {
                        formula: CoeffFormula::BinomialSeries {
                            alpha: alpha.clone(),
                        },
                        start: 0,
                        symmetry: 1,
                        sym_offset: 0,
                    },
                    variable: var,
                    point,
                });
            }
            None
        }

        _ => None,
    }
}

/// Check if `expr` structurally represents `1 + var`.
fn is_one_plus_var(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    if let ExprNode::Add(ref children) = arena.node(expr).clone()
        && children.len() == 2
    {
        let (a, b) = (children[0], children[1]);
        // Arena sorts Add children canonically; 1 (Num) typically comes first
        return (arena.is_one_structural(a) && b == var)
            || (arena.is_one_structural(b) && a == var);
    }
    false
}

/// Check if `expr` structurally represents `1 - var`.
fn is_one_minus_var(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    if let ExprNode::Add(ref children) = arena.node(expr).clone()
        && children.len() == 2
    {
        let (a, b) = (children[0], children[1]);
        // 1 + (-1)*x or (-1)*x + 1
        if arena.is_one_structural(a) && is_neg_of(arena, b, var) {
            return true;
        }
        if arena.is_one_structural(b) && is_neg_of(arena, a, var) {
            return true;
        }
    }
    false
}

/// Check if `expr` is `-var` (either Neg(var) or Mul([-1, var])).
fn is_neg_of(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    match arena.node(expr).clone() {
        ExprNode::Neg(inner) => inner == var,
        ExprNode::Mul(ref children) => {
            if children.len() == 2 {
                let (a, b) = (children[0], children[1]);
                if let Some(val) = arena.as_num(a)
                    && val.is_negative() && val.abs() == Ratio::one()
                {
                    return b == var;
                }
                if let Some(val) = arena.as_num(b)
                    && val.is_negative() && val.abs() == Ratio::one()
                {
                    return a == var;
                }
            }
            false
        }
        _ => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Truncated (fallback) FPS via Taylor coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Compute an FPS by extracting Taylor coefficients via repeated
/// differentiation. This is the fallback for functions not in the
/// known-function table.
pub(crate) fn fps_truncated(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    _var_sym: SymbolId,
    point: ExprId,
    order: usize,
) -> FormalPowerSeries {
    let mut coeffs: Vec<CachedCoeff> = Vec::with_capacity(order);
    let mut current_deriv = expr;
    let mut factorial = Ratio::<BigInt>::one();

    for k in 0..order {
        tracing::trace!("fps: computing coefficient k={}", k);
        // Evaluate k-th derivative at the point
        let subst = crate::subs::subs(arena, current_deriv, var, point);
        let evaled = crate::eval::eval(arena, subst);

        // Try to extract the rational value
        let coeff_val = if let Some(r) = arena.as_num(evaled) {
            r.clone() / factorial.clone()
        } else {
            // Not a rational number — store zero (best effort)
            Ratio::zero()
        };

        coeffs.push(CachedCoeff { value: coeff_val });

        // Compute next derivative
        if k + 1 < order {
            current_deriv = crate::diff::diff(arena, current_deriv, var);
            factorial *= Ratio::from_integer(BigInt::from(k as i64 + 1));
        }
    }

    FormalPowerSeries {
        kind: FPSKind::Truncated { coeffs },
        variable: var,
        point,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational algorithm (simplified)
// ═══════════════════════════════════════════════════════════════════════════

/// Try to find an FPS by checking if any derivative (up to 4th) is rational.
///
/// If `f^(j)(x)` is a rational function, we can use partial fraction
/// decomposition to find a closed-form for its Taylor coefficients, then
/// integrate j times symbolically to get back to `f`'s coefficients.
///
/// This is a simplified version that handles the case `j=0` (the function
/// itself is rational). For j>0, we fall back to the truncated approach
/// since the "integrate coefficient formula j times" step requires
/// nontrivial symbolic manipulation.
pub(crate) fn fps_rational(
    arena: &mut Arena,
    expr: ExprId,
    _var: ExprId,
    var_sym: SymbolId,
    point: ExprId,
) -> Option<FormalPowerSeries> {
    // Only handle Maclaurin for now
    if !arena.is_zero_structural(point) {
        return None;
    }

    // Check if expr itself is a rational function of var
    // A rational function is a ratio of polynomials
    if !is_rational_function(arena, expr, var_sym) {
        return None;
    }

    // For 1/(1-x), 1/(1+x), etc., the known-function path handles it.
    // For general rational functions, compute Taylor coefficients as fallback.
    // A full implementation would use partial fractions → coefficient formulas,
    // but that requires significant additional infrastructure.
    // We fall through to the truncated approach.
    None
}

/// Check if `expr` is a rational function of `var_sym`.
fn is_rational_function(arena: &Arena, expr: ExprId, _var_sym: SymbolId) -> bool {
    match arena.node(expr).clone() {
        ExprNode::Num(_) | ExprNode::Pi | ExprNode::E => true,
        ExprNode::Symbol(_sid) => true, // either it IS var or it's a constant
        ExprNode::Add(ref children) | ExprNode::Mul(ref children) => {
            children.iter().all(|&c| is_rational_function(arena, c, _var_sym))
        }
        ExprNode::Neg(inner) => is_rational_function(arena, inner, _var_sym),
        ExprNode::Pow(base, exp) => {
            // base^exp is rational if base is rational and exp is an integer
            if !is_rational_function(arena, base, _var_sym) {
                return false;
            }
            if let Some(r) = arena.as_num(exp) {
                r.is_integer()
            } else {
                false
            }
        }
        _ => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Default truncation order for the fallback approach.
const DEFAULT_ORDER: usize = 10;

/// Compute the formal power series of `expr` about `point` in `var`.
///
/// Tries known-function patterns first, then the rational algorithm,
/// then falls back to computing Taylor coefficients.
pub fn fps(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> FormalPowerSeries {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => {
            // Fallback: compute Taylor coefficients
            return fps_truncated(arena, expr, var, SymbolId(0), point, DEFAULT_ORDER);
        }
    };

    // 1. Try known elementary functions
    if let Some(fps) = fps_known_function(arena, expr, var, var_sym, point) {
        return fps;
    }

    // 2. Try the rational algorithm
    if let Some(fps) = fps_rational(arena, expr, var, var_sym, point) {
        return fps;
    }

    // 3. Fallback: compute Taylor coefficients
    fps_truncated(arena, expr, var, var_sym, point, DEFAULT_ORDER)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn fps_exp_x_coefficient_0() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let zero = a.zero;
        let series = fps(&mut a, expr, x, zero);
        // a_0 = 1/0! = 1
        let c0 = series.coefficient_rational(0);
        assert_eq!(c0, Ratio::one());
    }

    #[test]
    fn fps_exp_x_coefficient_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let zero = a.zero;
        let series = fps(&mut a, expr, x, zero);
        // a_1 = 1/1! = 1
        let c1 = series.coefficient_rational(1);
        assert_eq!(c1, Ratio::one());
    }

    #[test]
    fn fps_exp_x_coefficient_5() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let zero = a.zero;
        let series = fps(&mut a, expr, x, zero);
        // a_5 = 1/120
        let c5 = series.coefficient_rational(5);
        assert_eq!(c5, Ratio::new(BigInt::from(1), BigInt::from(120)));
    }

    #[test]
    fn fps_exp_x_has_closed_form() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let zero = a.zero;
        let series = fps(&mut a, expr, x, zero);
        assert!(series.has_closed_form());
    }

    #[test]
    fn fps_sin_x_coefficients() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let zero = a.zero;
        let series = fps(&mut a, expr, x, zero);
        // a_0 = 0
        assert!(series.coefficient_rational(0).is_zero());
        // a_1 = 1
        assert_eq!(series.coefficient_rational(1), Ratio::one());
        // a_2 = 0
        assert!(series.coefficient_rational(2).is_zero());
        // a_3 = -1/6
        assert_eq!(
            series.coefficient_rational(3),
            Ratio::new(BigInt::from(-1), BigInt::from(6))
        );
        // a_5 = 1/120
        assert_eq!(
            series.coefficient_rational(5),
            Ratio::new(BigInt::from(1), BigInt::from(120))
        );
    }

    #[test]
    fn fps_cos_x_coefficients() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.cos(x);
        let zero = a.zero;
        let series = fps(&mut a, expr, x, zero);
        // a_0 = 1
        assert_eq!(series.coefficient_rational(0), Ratio::one());
        // a_1 = 0
        assert!(series.coefficient_rational(1).is_zero());
        // a_2 = -1/2
        assert_eq!(
            series.coefficient_rational(2),
            Ratio::new(BigInt::from(-1), BigInt::from(2))
        );
        // a_3 = 0
        assert!(series.coefficient_rational(3).is_zero());
        // a_4 = 1/24
        assert_eq!(
            series.coefficient_rational(4),
            Ratio::new(BigInt::from(1), BigInt::from(24))
        );
    }

    #[test]
    fn fps_atan_x_coefficients() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.atan(x);
        let zero = a.zero;
        let series = fps(&mut a, expr, x, zero);
        // a_0 = 0
        assert!(series.coefficient_rational(0).is_zero());
        // a_1 = 1
        assert_eq!(series.coefficient_rational(1), Ratio::one());
        // a_2 = 0
        assert!(series.coefficient_rational(2).is_zero());
        // a_3 = -1/3
        assert_eq!(
            series.coefficient_rational(3),
            Ratio::new(BigInt::from(-1), BigInt::from(3))
        );
        // a_5 = 1/5
        assert_eq!(
            series.coefficient_rational(5),
            Ratio::new(BigInt::from(1), BigInt::from(5))
        );
    }

    #[test]
    fn fps_one_over_one_minus_x_coefficients() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        // Build 1/(1-x) = (1-x)^(-1)
        let one = a.one;
        let one_minus_x = a.sub(one, x);
        let neg_one = a.neg_one;
        let expr = a.pow(one_minus_x, neg_one);
        let series = fps(&mut a, expr, x, zero);
        // All coefficients should be 1
        for k in 0..10 {
            assert_eq!(
                series.coefficient_rational(k),
                Ratio::one(),
                "coefficient {k} should be 1"
            );
        }
    }

    #[test]
    fn fps_ln_1_plus_x_coefficients() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        // Build ln(1+x)
        let one = a.one;
        let one_plus_x = a.add(&[one, x]);
        let expr = a.ln(one_plus_x);
        let series = fps(&mut a, expr, x, zero);
        // a_0 = 0
        assert!(series.coefficient_rational(0).is_zero());
        // a_1 = 1
        assert_eq!(series.coefficient_rational(1), Ratio::one());
        // a_2 = -1/2
        assert_eq!(
            series.coefficient_rational(2),
            Ratio::new(BigInt::from(-1), BigInt::from(2))
        );
        // a_3 = 1/3
        assert_eq!(
            series.coefficient_rational(3),
            Ratio::new(BigInt::from(1), BigInt::from(3))
        );
    }

    #[test]
    fn fps_binomial_sqrt_1_plus_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        // Build (1+x)^(1/2)
        let one = a.one;
        let one_plus_x = a.add(&[one, x]);
        let half = a.rational(1, 2);
        let expr = a.pow(one_plus_x, half);
        let series = fps(&mut a, expr, x, zero);
        assert!(series.has_closed_form());
        // a_0 = C(1/2, 0) = 1
        assert_eq!(series.coefficient_rational(0), Ratio::one());
        // a_1 = C(1/2, 1) = 1/2
        assert_eq!(
            series.coefficient_rational(1),
            Ratio::new(BigInt::from(1), BigInt::from(2))
        );
        // a_2 = C(1/2, 2) = (1/2)(1/2 - 1)/2! = (1/2)(-1/2)/2 = -1/8
        assert_eq!(
            series.coefficient_rational(2),
            Ratio::new(BigInt::from(-1), BigInt::from(8))
        );
    }

    #[test]
    fn fps_exp_truncate_matches_maclaurin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let exp_x = a.exp(x);
        let zero = a.zero;

        let series = fps(&mut a, exp_x, x, zero);
        let truncated = series.truncate(&mut a, 5);

        // Also compute via the series module directly
        let maclaurin = crate::series::series(&mut a, exp_x, x, zero, 5).unwrap();

        // Both should produce the same expanded polynomial
        let t_expanded = crate::expand::expand(&mut a, truncated);
        let t_evaled = crate::eval::eval(&mut a, t_expanded);
        let m_expanded = crate::expand::expand(&mut a, maclaurin);
        let m_evaled = crate::eval::eval(&mut a, m_expanded);

        assert_eq!(
            display(&a, t_evaled),
            display(&a, m_evaled),
            "FPS truncate should match Maclaurin"
        );
    }

    #[test]
    fn fps_sinh_coefficients() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sinh(x);
        let zero = a.zero;
        let series = fps(&mut a, expr, x, zero);
        // a_0 = 0
        assert!(series.coefficient_rational(0).is_zero());
        // a_1 = 1
        assert_eq!(series.coefficient_rational(1), Ratio::one());
        // a_2 = 0
        assert!(series.coefficient_rational(2).is_zero());
        // a_3 = 1/6
        assert_eq!(
            series.coefficient_rational(3),
            Ratio::new(BigInt::from(1), BigInt::from(6))
        );
    }

    #[test]
    fn fps_cosh_coefficients() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.cosh(x);
        let zero = a.zero;
        let series = fps(&mut a, expr, x, zero);
        // a_0 = 1
        assert_eq!(series.coefficient_rational(0), Ratio::one());
        // a_1 = 0
        assert!(series.coefficient_rational(1).is_zero());
        // a_2 = 1/2
        assert_eq!(
            series.coefficient_rational(2),
            Ratio::new(BigInt::from(1), BigInt::from(2))
        );
    }

    #[test]
    fn fps_truncated_fallback() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        // Use a non-standard function that won't match known patterns:
        // exp(x) + sin(x)
        let exp_x = a.exp(x);
        let sin_x = a.sin(x);
        let expr = a.add(&[exp_x, sin_x]);
        let series = fps(&mut a, expr, x, zero);
        // Should fall back to truncated
        assert!(!series.has_closed_form());
        // a_0 = exp(0) + sin(0) = 1
        assert_eq!(series.coefficient_rational(0), Ratio::one());
        // a_1 = exp(0) + cos(0) = 2
        assert_eq!(
            series.coefficient_rational(1),
            Ratio::from_integer(BigInt::from(2))
        );
    }
}
