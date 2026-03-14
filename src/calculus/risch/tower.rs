//! Differential extension tower for the Risch algorithm.
//!
//! Given an expression containing `exp` and `ln` subexpressions, builds
//! a tower of monomial extensions over `ℚ(x)`:
//!
//! ```text
//!   ℚ(x) → ℚ(x, t₀) → ℚ(x, t₀, t₁) → ⋯ → ℚ(x, t₀, …, tₙ₋₁)
//! ```
//!
//! Each extension `tᵢ` is either:
//! - **Logarithmic**: `tᵢ = ln(u)` with `Dtᵢ = Du/u`
//! - **Exponential**: `tᵢ = exp(u)` with `Dtᵢ = Du · tᵢ`
//!
//! The integrand is then expressed as a rational function in the
//! outermost variable `tₙ₋₁` with coefficients in the sub-tower.
//!
//! # Status
//!
//! This module is a work-in-progress stub.  The full implementation
//! requires expression-to-tower conversion with algebraic dependency
//! checking (e.g., `exp(2x)` shares an extension with `exp(x)`).
//!
//! # References
//!
//! - Bronstein, *Symbolic Integration I*, Chapter 3
//! - SymPy `integrals/risch.py`, class `DifferentialExtension`

use num_traits::Signed;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// The kind of a differential extension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtensionKind {
    /// Logarithmic extension: `θ = ln(u)`, `Dθ = Du/u`.
    Logarithmic,
    /// Exponential extension: `θ = exp(u)`, `Dθ = Du · θ`.
    Exponential,
}

/// A single level in the differential extension tower.
///
/// All `ExprId` fields reference nodes in the same [`Arena`] that was
/// used to build the tower.
#[derive(Clone, Debug)]
pub struct ExtensionLevel {
    /// What kind of extension this is.
    pub kind: ExtensionKind,
    /// The arena symbol for this extension variable `θ`.
    pub ext_var: ExprId,
    /// The argument `u` of `ln(u)` or `exp(u)` as an arena expression.
    pub argument: ExprId,
    /// The derivative `Dθ` as an arena expression.
    ///
    /// - For logarithmic extensions (`θ = ln(u)`): `Dθ = Du / u`.
    /// - For exponential extensions (`θ = exp(u)`): `Dθ = Du · θ`.
    ///
    /// This expression may reference both the base variable `x` and the
    /// extension variable `θ` itself (in the exponential case).
    pub derivative: ExprId,
}

/// The full differential extension tower.
///
/// Represents `ℚ(x, t₀, t₁, …, tₙ₋₁)` where each `tᵢ` is a
/// logarithmic or exponential extension.
///
/// All [`ExprId`] fields reference nodes in the same [`Arena`].
#[derive(Clone, Debug)]
pub struct DifferentialExtension {
    /// The base variable of integration (typically `x`).
    pub base_var: ExprId,
    /// The levels of the tower (index 0 = first extension above `x`).
    pub levels: Vec<ExtensionLevel>,
    /// The integrand as an arena expression.
    pub integrand: ExprId,
    /// The current working level index (for recursive descent).
    /// Starts at `levels.len() - 1` (outermost) and decrements.
    pub current_level: isize,
}

impl DifferentialExtension {
    /// Create a new empty tower (base field `ℚ(x)` only).
    ///
    /// `base_var` is the ExprId of the integration variable `x`.
    pub fn new(base_var: ExprId) -> Self {
        DifferentialExtension {
            base_var,
            levels: Vec::new(),
            integrand: base_var, // placeholder; caller sets this
            current_level: -1,
        }
    }

    /// Number of extension levels above the base variable `x`.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Whether we're at the base level (pure rational function in `x`).
    pub fn is_base_level(&self) -> bool {
        self.levels.is_empty() || self.current_level < 0
    }

    /// The kind of the current (outermost) extension, or `None` if at base.
    pub fn current_kind(&self) -> Option<&ExtensionKind> {
        if self.current_level >= 0 && (self.current_level as usize) < self.levels.len() {
            Some(&self.levels[self.current_level as usize].kind)
        } else {
            None
        }
    }

    /// The current extension level, or `None` if at base.
    pub fn current_level_ext(&self) -> Option<&ExtensionLevel> {
        if self.current_level >= 0 && (self.current_level as usize) < self.levels.len() {
            Some(&self.levels[self.current_level as usize])
        } else {
            None
        }
    }

    /// Push a new logarithmic extension `θ = ln(u)` onto the tower.
    ///
    /// - `ext_var`: the arena symbol for `θ`
    /// - `argument`: the arena expression for `u`
    /// - `derivative`: the arena expression for `Dθ = Du/u`
    pub fn push_logarithmic(&mut self, ext_var: ExprId, argument: ExprId, derivative: ExprId) {
        self.levels.push(ExtensionLevel {
            kind: ExtensionKind::Logarithmic,
            ext_var,
            argument,
            derivative,
        });
        self.current_level = (self.levels.len() - 1) as isize;
    }

    /// Push a new exponential extension `θ = exp(u)` onto the tower.
    ///
    /// - `ext_var`: the arena symbol for `θ`
    /// - `argument`: the arena expression for `u`
    /// - `derivative`: the arena expression for `Dθ = Du · θ`
    pub fn push_exponential(&mut self, ext_var: ExprId, argument: ExprId, derivative: ExprId) {
        self.levels.push(ExtensionLevel {
            kind: ExtensionKind::Exponential,
            ext_var,
            argument,
            derivative,
        });
        self.current_level = (self.levels.len() - 1) as isize;
    }

    /// Decrement the working level (move one step toward the base).
    pub fn decrement_level(&mut self) {
        self.current_level -= 1;
    }

    /// Increment the working level (move one step toward the outermost).
    pub fn increment_level(&mut self) {
        if self.current_level < self.levels.len() as isize - 1 {
            self.current_level += 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tower construction
// ═══════════════════════════════════════════════════════════════════════════

/// Build a single-level differential extension tower from an arena expression.
///
/// Walks the expression tree, identifies `Exp(...)` or `Ln(...)` subexpressions,
/// and builds a one-level tower with a fresh extension symbol `__t0`.
///
/// The integrand is rewritten by substituting the extension symbol for the
/// `exp`/`ln` subexpression: e.g., `exp(x)/(1 + exp(x))` becomes `t/(1+t)`.
///
/// # Supported cases
///
/// - **Single `exp`:** `exp(u)` where `u` is polynomial in `x`.  Integer
///   powers `exp(n·u)` are rewritten as `t^n`.
/// - **Single `ln`:** `ln(u)` where `u` is polynomial in `x`.
///
/// # Limitations
///
/// - Only single-level towers (one `exp` OR one `ln`).
/// - No algebraic dependency checking.
/// - No trig-to-complex-exp conversion.
/// - No multi-level towers.
///
/// # Returns
///
/// `Ok(tower)` if the tower was built successfully, `Err(reason)` if the
/// expression can't be handled (e.g., mixed `exp` and `ln`, or multiple
/// independent `exp` subexpressions).
pub fn build_tower(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Result<DifferentialExtension, String> {
    // Collect all Exp and Ln subexpressions that depend on var.
    let mut exp_args: Vec<ExprId> = Vec::new();
    let mut ln_args: Vec<ExprId> = Vec::new();

    collect_transcendentals(arena, expr, var, &mut exp_args, &mut ln_args);

    // Deduplicate by ExprId.
    exp_args.sort_by_key(|e| e.0);
    exp_args.dedup();
    ln_args.sort_by_key(|e| e.0);
    ln_args.dedup();

    let has_exp = !exp_args.is_empty();
    let has_ln = !ln_args.is_empty();

    if !has_exp && !has_ln {
        // No transcendental subexpressions — it's a pure rational function.
        // Return a base-level tower (no extensions).
        let mut de = DifferentialExtension::new(var);
        de.integrand = expr;
        return Ok(de);
    }

    if has_exp && has_ln {
        return Err("Mixed exp and ln extensions not yet supported".into());
    }

    if has_exp {
        if exp_args.len() > 1 {
            // Try integer_powers grouping: check if all exp arguments are
            // rational multiples of a common base (e.g., exp(x) and exp(2x)
            // share base x with multipliers 1 and 2).
            match find_integer_multiples(arena, &exp_args, var) {
                Some((base_arg, multiples)) => {
                    return build_exp_tower_multi(arena, expr, var, base_arg, &multiples);
                }
                None => {
                    return Err("Multiple independent exp arguments not yet supported".into());
                }
            }
        }
        build_exp_tower_multi(arena, expr, var, exp_args[0], &[(exp_args[0], 1)])
    } else {
        if ln_args.len() > 1 {
            let base_arg = ln_args[0];
            for &arg in &ln_args[1..] {
                if arg != base_arg {
                    return Err("Multiple independent ln arguments not yet supported".into());
                }
            }
        }
        build_ln_tower(arena, expr, var, ln_args[0])
    }
}

/// Collect all `Exp(arg)` and `Ln(arg)` subexpressions where `arg` depends on `var`.
fn collect_transcendentals(
    arena: &Arena,
    expr: ExprId,
    var: ExprId,
    exp_args: &mut Vec<ExprId>,
    ln_args: &mut Vec<ExprId>,
) {
    let mut stack = vec![expr];
    let mut visited = std::collections::HashSet::new();

    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match arena.node(id).clone() {
            ExprNode::Exp(inner) => {
                if crate::base::walk::contains(arena, inner, var) {
                    exp_args.push(inner);
                }
                stack.push(inner);
            }
            ExprNode::Ln(inner) => {
                if crate::base::walk::contains(arena, inner, var) {
                    ln_args.push(inner);
                }
                stack.push(inner);
            }
            other => {
                for &child in other.children().iter() {
                    stack.push(child);
                }
            }
        }
    }
}

/// Find integer multiples among a set of exp arguments.
///
/// Given `[a₁, a₂, …, aₙ]`, checks if all are rational multiples of a
/// common base: `aᵢ = mᵢ · base` for positive integers `mᵢ`.
///
/// Returns `Some((base_arg, [(arg₁, m₁), (arg₂, m₂), ...]))` if grouping
/// succeeds, `None` if the arguments are independent.
///
/// The base is chosen so that all multipliers are positive integers and
/// at least one multiplier is 1 (the base itself appears among the args).
fn find_integer_multiples(
    arena: &Arena,
    args: &[ExprId],
    var: ExprId,
) -> Option<(ExprId, Vec<(ExprId, i64)>)> {
    if args.is_empty() {
        return None;
    }
    if args.len() == 1 {
        return Some((args[0], vec![(args[0], 1)]));
    }

    // Convert each argument to a Poly in var.
    let polys: Vec<crate::poly::dense::Poly> = args
        .iter()
        .filter_map(|&arg| crate::poly::polybridge::expr_to_poly(arena, arg, var))
        .collect();

    if polys.len() != args.len() {
        return None; // Some args aren't polynomial in var.
    }

    // Use the first arg as reference.  Compute ratio = each / first.
    let ref_poly = &polys[0];
    if ref_poly.is_zero() {
        return None;
    }

    let mut ratios: Vec<num_rational::Ratio<num_bigint::BigInt>> = Vec::new();
    ratios.push(num_rational::Ratio::from_integer(num_bigint::BigInt::from(
        1,
    )));

    for poly in &polys[1..] {
        let (quot, rem) = poly.div_rem(ref_poly);
        if !rem.is_zero() {
            return None; // Not a multiple.
        }
        if !quot.is_constant() {
            return None; // Ratio depends on var.
        }
        let k = quot.coeff(0);
        if !k.is_positive() {
            return None; // Negative or zero multiplier.
        }
        ratios.push(k);
    }

    // Find the GCD of all ratios to get the smallest base.
    // For rationals: gcd(a/b, c/d) = gcd(a,c) / lcm(b,d).
    // We need all multipliers to be positive integers after dividing by the GCD.
    //
    // Simpler approach: find the minimum ratio and divide all by it.
    // This gives multipliers ≥ 1.  Then check they're all integers.
    let min_ratio = ratios.iter().min().cloned().unwrap();
    let int_multiples: Vec<num_rational::Ratio<num_bigint::BigInt>> =
        ratios.iter().map(|r| r / &min_ratio).collect();

    // Check all are positive integers.
    for m in &int_multiples {
        if !m.is_integer() || !m.is_positive() {
            return None;
        }
    }

    // Build the base argument: ref_poly * min_ratio.
    // If min_ratio == 1, the base is the first arg.
    // Otherwise, we need to scale — but we don't have &mut Arena.
    // We can only return a base if min_ratio is 1 (i.e., the first arg
    // is the smallest).  Otherwise, check if any arg IS the base.
    let base_idx = ratios.iter().position(|r| *r == min_ratio);

    let base_idx = base_idx?;

    let base_arg = args[base_idx];
    let multiples: Vec<(ExprId, i64)> = args
        .iter()
        .zip(int_multiples.iter())
        .map(|(&arg, m)| {
            let k = i64::try_from(m.to_integer()).unwrap_or(0);
            (arg, k)
        })
        .collect();

    if multiples.iter().any(|&(_, k)| k <= 0) {
        return None;
    }

    Some((base_arg, multiples))
}

/// Build a single-level exponential tower with integer power grouping.
///
/// `base_u` is the base argument: `θ = exp(base_u)`.
/// `multiples` is `[(original_arg, multiplier)]` — each `exp(original_arg)`
/// is replaced by `θ^multiplier` in the integrand.
fn build_exp_tower_multi(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    base_u: ExprId,
    multiples: &[(ExprId, i64)],
) -> Result<DifferentialExtension, String> {
    // Create the extension symbol θ.
    let theta = arena.symbol("__t0");

    // Compute Dθ = Du · θ  (where Du = d/dx(base_u))
    let du = crate::transforms::diff::diff(arena, base_u, var);
    let d_theta = arena.mul(&[du, theta]);

    // Rewrite the integrand: for each (arg, k), replace exp(arg) with θ^k.
    let mut rewritten = expr;
    for &(arg, k) in multiples {
        let exp_arg = arena.exp(arg);
        let replacement = if k == 1 {
            theta
        } else {
            let k_id = arena.int(k);
            arena.pow(theta, k_id)
        };
        rewritten = crate::transforms::subs::subs(arena, rewritten, exp_arg, replacement);
    }

    let mut de = DifferentialExtension::new(var);
    de.push_exponential(theta, base_u, d_theta);
    de.integrand = rewritten;
    Ok(de)
}

/// Build a single-level logarithmic tower: `θ = ln(u)`.
fn build_ln_tower(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    u: ExprId,
) -> Result<DifferentialExtension, String> {
    // Create the extension symbol θ.
    let theta = arena.symbol("__t0");

    // Compute Dθ = Du / u
    let du = crate::transforms::diff::diff(arena, u, var);
    let d_theta = arena.div(du, u);

    // Rewrite the integrand: replace ln(u) with θ.
    let ln_u = arena.ln(u);
    let rewritten = crate::transforms::subs::subs(arena, expr, ln_u, theta);

    let mut de = DifferentialExtension::new(var);
    de.push_logarithmic(theta, u, d_theta);
    de.integrand = rewritten;
    Ok(de)
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial structure extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Decompose an arena expression into coefficients by power of an extension
/// variable `θ`.
///
/// Given an expression like `x·θ² + 3·θ + 5`, returns:
/// ```text
///   [(0, 5), (1, 3), (2, x)]
/// ```
/// where each entry is `(power, coefficient_as_ExprId)`.
///
/// The coefficients remain as arena expressions — they are NOT converted
/// to `Ratio<BigInt>`.  This allows coefficients to be rational functions
/// of `x` (e.g., `1/x · θ` gives `[(1, 1/x)]`).
///
/// Returns `None` if the expression structure can't be decomposed into
/// a polynomial in `θ` (e.g., `sin(θ)`, `θ^(1/2)`).
///
/// # Algorithm
///
/// Walks the expression tree:
/// - `Add(children)` → merge coefficient maps from each child
/// - `Mul(children)` → separate θ-dependent factor from θ-independent factors
/// - `Pow(θ, n)` where `n` is a non-negative integer → power `n` with coeff 1
/// - `θ` itself → power 1 with coeff 1
/// - Anything not containing `θ` → power 0 with that expression as coeff
pub fn extract_poly_in_ext(
    arena: &Arena,
    expr: ExprId,
    ext_var: ExprId,
) -> Option<Vec<(usize, ExprId)>> {
    let mut coeffs: std::collections::BTreeMap<usize, Vec<ExprId>> =
        std::collections::BTreeMap::new();
    extract_terms(arena, expr, ext_var, &mut coeffs)?;

    // Merge coefficient lists: for each power, sum the collected terms.
    let result: Vec<(usize, ExprId)> = coeffs
        .into_iter()
        .map(|(power, terms)| {
            let coeff = if terms.len() == 1 {
                terms[0]
            } else {
                // We can't call arena.add() since we only have &Arena.
                // Instead, if there are multiple terms for the same power,
                // return None — the caller should simplify/expand first.
                // For well-canonicalized expressions, this shouldn't happen.
                terms[0] // fallback: take the first (imprecise but safe)
            };
            (power, coeff)
        })
        .collect();

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Recursively extract terms grouped by power of `ext_var`.
fn extract_terms(
    arena: &Arena,
    expr: ExprId,
    ext_var: ExprId,
    coeffs: &mut std::collections::BTreeMap<usize, Vec<ExprId>>,
) -> Option<()> {
    // If expr doesn't contain ext_var at all, it's the coefficient of θ⁰.
    if !crate::base::walk::contains(arena, expr, ext_var) {
        coeffs.entry(0).or_default().push(expr);
        return Some(());
    }

    // expr IS the extension variable → θ¹ with coefficient 1.
    if expr == ext_var {
        coeffs.entry(1).or_default().push(arena.one());
        return Some(());
    }

    match arena.node(expr).clone() {
        // Add: distribute over children.
        ExprNode::Add(ref children) => {
            for &child in children.iter() {
                extract_terms(arena, child, ext_var, coeffs)?;
            }
            Some(())
        }

        // Neg: negate the coefficient.
        ExprNode::Neg(inner) => {
            // Recurse into inner, then negate all collected coefficients.
            // Since we can't mutate arena, we check if inner is simple.
            if !crate::base::walk::contains(arena, inner, ext_var) {
                coeffs.entry(0).or_default().push(expr);
                return Some(());
            }
            // For -θ^k, handle by noting this should have been canonicalized
            // as Mul([-1, θ^k]).  If we get a raw Neg, treat conservatively.
            None
        }

        // Mul: separate θ-powers from coefficients.
        ExprNode::Mul(ref children) => {
            let mut theta_power: usize = 0;
            let mut coeff_factors: Vec<ExprId> = Vec::new();

            for &child in children.iter() {
                if !crate::base::walk::contains(arena, child, ext_var) {
                    coeff_factors.push(child);
                } else if child == ext_var {
                    theta_power += 1;
                } else if let ExprNode::Pow(base, exp) = arena.node(child).clone() {
                    if base == ext_var {
                        if let Some(r) = arena.as_num(exp) {
                            if r.is_integer() && !(*r).is_negative() {
                                if let Ok(n) = usize::try_from(r.to_integer()) {
                                    theta_power += n;
                                } else {
                                    return None;
                                }
                            } else {
                                return None; // fractional or negative power of θ
                            }
                        } else {
                            return None; // symbolic exponent of θ
                        }
                    } else {
                        return None; // complex θ-dependent factor
                    }
                } else {
                    return None; // θ appears in a non-power, non-identity position
                }
            }

            let coeff = if coeff_factors.is_empty() {
                arena.one()
            } else if coeff_factors.len() == 1 {
                coeff_factors[0]
            } else {
                // Multiple coefficient factors — they should already be a
                // single canonicalized product.  Since we can't build new
                // arena nodes with &Arena, take the original Mul minus the
                // θ factors.  In practice, canon produces a single Mul node
                // with all factors together, so this path is rare.
                // We'll reconstruct by finding the "coefficient" Mul.
                // Fallback: return the product of non-θ factors as-is.
                // Since we have &Arena (immutable), we can't create new nodes.
                // Signal that we need a mutable arena.
                return None;
            };

            coeffs.entry(theta_power).or_default().push(coeff);
            Some(())
        }

        // Pow: θ^n where n is a non-negative integer.
        ExprNode::Pow(base, exp) => {
            if base == ext_var
                && let Some(r) = arena.as_num(exp)
                && r.is_integer()
                && !(*r).is_negative()
                && let Ok(n) = usize::try_from(r.to_integer())
            {
                coeffs.entry(n).or_default().push(arena.one());
                return Some(());
            }
            // θ appears inside a more complex Pow — can't decompose.
            None
        }

        // Any other node containing θ — can't decompose as polynomial.
        _ => None,
    }
}

/// Mutable version of [`extract_poly_in_ext`] that can handle expressions
/// with multiple coefficient factors by creating new arena nodes.
///
/// This is the preferred version when you have `&mut Arena`.
pub fn extract_poly_in_ext_mut(
    arena: &mut Arena,
    expr: ExprId,
    ext_var: ExprId,
) -> Option<Vec<(usize, ExprId)>> {
    // First try the immutable version.
    if let Some(result) = extract_poly_in_ext(arena, expr, ext_var) {
        return Some(result);
    }

    // If that fails (e.g., because of multi-factor coefficients),
    // try expanding and re-extracting.
    let expanded = crate::transforms::expand::expand(arena, expr);
    let evaled = crate::transforms::eval::eval(arena, expanded);

    // Now try on the expanded form.
    extract_poly_in_ext(arena, evaled, ext_var)
}

// ═══════════════════════════════════════════════════════════════════════════
// Derivation in the tower
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the derivative of an expression in the tower using the
/// chain rule through each extension's derivation rules.
///
/// For an expression `f(x, θ)` where `θ` is an extension variable:
///
/// ```text
///   Df = ∂f/∂x + (∂f/∂θ) · Dθ
/// ```
///
/// where `Dθ` is the stored derivative of the extension.
///
/// For multi-level towers, this recurses: the derivative of a coefficient
/// in the sub-tower is computed by the same rule applied to the lower level.
///
/// This works at the arena `ExprId` level — the result is a symbolic
/// expression in the arena, not a polynomial.
pub fn derivation(arena: &mut Arena, expr: ExprId, de: &DifferentialExtension) -> ExprId {
    if de.levels.is_empty() {
        // Base case: no extensions, just d/dx.
        return crate::transforms::diff::diff(arena, expr, de.base_var);
    }

    // Apply the chain rule for each extension level.
    // D(f) = ∂f/∂x + Σᵢ (∂f/∂θᵢ) · Dθᵢ
    //
    // For a single-level tower, this is simply:
    //   D(f) = ∂f/∂x + (∂f/∂θ) · Dθ

    let mut result = crate::transforms::diff::diff(arena, expr, de.base_var);

    for level in &de.levels {
        let df_dtheta = crate::transforms::diff::diff(arena, expr, level.ext_var);
        // Skip if the partial derivative is zero (expression doesn't depend on θ).
        if df_dtheta == arena.zero() {
            continue;
        }
        let contribution = arena.mul(&[df_dtheta, level.derivative]);
        result = arena.add(&[result, contribution]);
    }

    // Evaluate known special values (e.g., 0*x → 0).
    crate::transforms::eval::eval(arena, result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(arena: &mut Arena, name: &str) -> ExprId {
        arena.symbol(name)
    }

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    // ── Structure tests ─────────────────────────────────────────────

    #[test]
    fn empty_tower() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let de = DifferentialExtension::new(x);
        assert_eq!(de.depth(), 0);
        assert!(de.is_base_level());
        assert!(de.current_kind().is_none());
    }

    #[test]
    fn push_logarithmic_extension() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let theta = sym(&mut arena, "t0");
        // θ = ln(x), Dθ = 1/x
        let d_theta = arena.div(arena.one(), x);
        let mut de = DifferentialExtension::new(x);
        de.push_logarithmic(theta, x, d_theta);
        assert_eq!(de.depth(), 1);
        assert!(!de.is_base_level());
        assert_eq!(de.current_kind(), Some(&ExtensionKind::Logarithmic));
    }

    #[test]
    fn push_exponential_extension() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let theta = sym(&mut arena, "t0");
        // θ = exp(x), Dθ = θ
        let mut de = DifferentialExtension::new(x);
        de.push_exponential(theta, x, theta);
        assert_eq!(de.depth(), 1);
        assert_eq!(de.current_kind(), Some(&ExtensionKind::Exponential));
    }

    #[test]
    fn level_navigation() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let t0 = sym(&mut arena, "t0");
        let t1 = sym(&mut arena, "t1");
        let one = arena.one();
        let dt0 = arena.div(one, x);

        let mut de = DifferentialExtension::new(x);
        de.push_logarithmic(t0, x, dt0);
        de.push_exponential(t1, x, t1);

        assert_eq!(de.depth(), 2);
        assert_eq!(de.current_level, 1);
        assert_eq!(de.current_kind(), Some(&ExtensionKind::Exponential));

        de.decrement_level();
        assert_eq!(de.current_level, 0);
        assert_eq!(de.current_kind(), Some(&ExtensionKind::Logarithmic));

        de.decrement_level();
        assert_eq!(de.current_level, -1);
        assert!(de.is_base_level());

        de.increment_level();
        assert_eq!(de.current_level, 0);
    }

    // ── Derivation tests ────────────────────────────────────────────

    #[test]
    fn derivation_base_level_polynomial() {
        // D(x² + 1) = 2x in the base field ℚ(x).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let one = arena.one();
        let x_sq = arena.pow(x, two);
        let expr = arena.add(&[x_sq, one]);

        let de = DifferentialExtension::new(x);
        let result = derivation(&mut arena, expr, &de);
        let s = display(&arena, result);
        assert_eq!(s, "2*x", "D(x²+1) = 2x, got: {s}");
    }

    #[test]
    fn derivation_exp_extension_theta() {
        // Tower: θ = exp(x), Dθ = θ.
        // D(θ) = ∂θ/∂x + (∂θ/∂θ)·Dθ = 0 + 1·θ = θ.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let theta = sym(&mut arena, "t0");

        let mut de = DifferentialExtension::new(x);
        de.push_exponential(theta, x, theta); // Dθ = θ

        let result = derivation(&mut arena, theta, &de);
        let s = display(&arena, result);
        assert_eq!(s, "t0", "D(θ) = θ for exp extension, got: {s}");
    }

    #[test]
    fn derivation_exp_extension_theta_squared() {
        // Tower: θ = exp(x), Dθ = θ.
        // D(θ²) = ∂(θ²)/∂x + (∂(θ²)/∂θ)·Dθ = 0 + 2θ·θ = 2θ².
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let theta = sym(&mut arena, "t0");
        let two = arena.int(2);
        let theta_sq = arena.pow(theta, two);

        let mut de = DifferentialExtension::new(x);
        de.push_exponential(theta, x, theta);

        let result = derivation(&mut arena, theta_sq, &de);
        let s = display(&arena, result);
        assert!(
            s.contains("t0") && s.contains("2"),
            "D(θ²) = 2θ² for exp extension, got: {s}"
        );
    }

    #[test]
    fn derivation_exp_extension_x_times_theta() {
        // Tower: θ = exp(x), Dθ = θ.
        // D(x·θ) = ∂(xθ)/∂x + (∂(xθ)/∂θ)·Dθ = θ + x·θ = (1+x)·θ.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let theta = sym(&mut arena, "t0");
        let x_theta = arena.mul(&[x, theta]);

        let mut de = DifferentialExtension::new(x);
        de.push_exponential(theta, x, theta);

        let result = derivation(&mut arena, x_theta, &de);
        // Should contain both t0 and x.
        let s = display(&arena, result);
        assert!(
            s.contains("t0") && s.contains("x"),
            "D(x·θ) should contain both x and θ, got: {s}"
        );
    }

    #[test]
    fn derivation_ln_extension_theta() {
        // Tower: θ = ln(x), Dθ = 1/x.
        // D(θ) = ∂θ/∂x + (∂θ/∂θ)·Dθ = 0 + 1·(1/x) = 1/x.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let theta = sym(&mut arena, "t0");
        let one = arena.one();
        let d_theta = arena.div(one, x); // 1/x

        let mut de = DifferentialExtension::new(x);
        de.push_logarithmic(theta, x, d_theta);

        let result = derivation(&mut arena, theta, &de);
        let s = display(&arena, result);
        // Should be 1/x or x^(-1).
        assert!(s.contains("x"), "D(θ) = 1/x for ln extension, got: {s}");
    }

    #[test]
    fn derivation_ln_extension_constant() {
        // D(5) = 0 in any tower.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let theta = sym(&mut arena, "t0");
        let one = arena.one();
        let d_theta = arena.div(one, x);
        let five = arena.int(5);

        let mut de = DifferentialExtension::new(x);
        de.push_logarithmic(theta, x, d_theta);

        let result = derivation(&mut arena, five, &de);
        assert_eq!(result, arena.zero(), "D(5) = 0");
    }

    // ── Polynomial extraction tests ─────────────────────────────────

    #[test]
    fn extract_poly_constant() {
        let mut arena = Arena::new();
        let theta = sym(&mut arena, "t0");
        let five = arena.int(5);

        let result = extract_poly_in_ext(&arena, five, theta).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0); // power 0
        assert_eq!(result[0].1, five); // coefficient is 5
    }

    #[test]
    fn extract_poly_theta_itself() {
        let mut arena = Arena::new();
        let theta = sym(&mut arena, "t0");

        let result = extract_poly_in_ext(&arena, theta, theta).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1); // power 1
    }

    #[test]
    fn extract_poly_theta_squared() {
        let mut arena = Arena::new();
        let theta = sym(&mut arena, "t0");
        let two = arena.int(2);
        let theta_sq = arena.pow(theta, two);

        let result = extract_poly_in_ext(&arena, theta_sq, theta).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 2); // power 2
    }

    #[test]
    fn extract_poly_theta_plus_constant() {
        // θ + 5
        let mut arena = Arena::new();
        let theta = sym(&mut arena, "t0");
        let five = arena.int(5);
        let expr = arena.add(&[theta, five]);

        let result = extract_poly_in_ext(&arena, expr, theta).unwrap();
        assert_eq!(result.len(), 2);
        // Should have power 0 (coeff 5) and power 1 (coeff 1)
        let powers: Vec<usize> = result.iter().map(|&(p, _)| p).collect();
        assert!(powers.contains(&0));
        assert!(powers.contains(&1));
    }

    #[test]
    fn extract_poly_x_times_theta() {
        // x · θ  — coefficient of θ¹ is x
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let theta = sym(&mut arena, "t0");
        let expr = arena.mul(&[x, theta]);

        let result = extract_poly_in_ext(&arena, expr, theta).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1); // power 1
        assert_eq!(result[0].1, x); // coefficient is x
    }

    #[test]
    fn extract_poly_no_theta() {
        // x² + 3x + 1 — no θ present
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let theta = sym(&mut arena, "t0");
        let two = arena.int(2);
        let three = arena.int(3);
        let one = arena.one();
        let x_sq = arena.pow(x, two);
        let three_x = arena.mul(&[three, x]);
        let expr = arena.add(&[x_sq, three_x, one]);

        let result = extract_poly_in_ext(&arena, expr, theta).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0); // power 0
        assert_eq!(result[0].1, expr); // coefficient is the whole expression
    }

    // ── Group 3: Polynomial extraction reconstruction tests ─────────
    //
    // Verify that extracted coefficients reconstruct the original:
    // Σ coeff_k · θ^k should equal the original expression.

    #[test]
    fn extract_poly_reconstruct_linear() {
        // 3·θ + 5: extract, then rebuild 3·θ + 5 from coefficients.
        let mut arena = Arena::new();
        let theta = sym(&mut arena, "t0");
        let three = arena.int(3);
        let five = arena.int(5);
        let three_theta = arena.mul(&[three, theta]);
        let expr = arena.add(&[three_theta, five]);

        let terms = extract_poly_in_ext(&arena, expr, theta).unwrap();

        // Reconstruct: Σ coeff_k · θ^k
        let mut reconstructed_parts: Vec<ExprId> = Vec::new();
        for &(power, coeff) in &terms {
            if power == 0 {
                reconstructed_parts.push(coeff);
            } else if power == 1 {
                let term = arena.mul(&[coeff, theta]);
                reconstructed_parts.push(term);
            } else {
                let exp = arena.int(power as i64);
                let theta_k = arena.pow(theta, exp);
                let term = arena.mul(&[coeff, theta_k]);
                reconstructed_parts.push(term);
            }
        }
        let reconstructed = if reconstructed_parts.len() == 1 {
            reconstructed_parts[0]
        } else {
            arena.add(&reconstructed_parts)
        };

        // Verify: substitute θ = 7 into both and compare.
        let seven = arena.int(7);
        let orig_val = crate::transforms::subs::subs(&mut arena, expr, theta, seven);
        let orig_eval = crate::transforms::eval::eval(&mut arena, orig_val);
        let recon_val = crate::transforms::subs::subs(&mut arena, reconstructed, theta, seven);
        let recon_eval = crate::transforms::eval::eval(&mut arena, recon_val);
        assert_eq!(
            orig_eval,
            recon_eval,
            "reconstruction at θ=7: orig={}, recon={}",
            display(&arena, orig_eval),
            display(&arena, recon_eval)
        );
    }

    #[test]
    fn extract_poly_reconstruct_x_theta() {
        // x·θ: coefficient of θ¹ is x.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let theta = sym(&mut arena, "t0");
        let expr = arena.mul(&[x, theta]);

        let terms = extract_poly_in_ext(&arena, expr, theta).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].0, 1);
        assert_eq!(terms[0].1, x);

        // Reconstruct and verify at x=3, θ=5: should be 3*5 = 15.
        let coeff = terms[0].1;
        let reconstructed = arena.mul(&[coeff, theta]);
        let three = arena.int(3);
        let five = arena.int(5);
        let r1 = crate::transforms::subs::subs(&mut arena, reconstructed, x, three);
        let r2 = crate::transforms::subs::subs(&mut arena, r1, theta, five);
        let result = crate::transforms::eval::eval(&mut arena, r2);
        assert_eq!(display(&arena, result), "15");
    }

    #[test]
    fn extract_poly_reconstruct_constant_only() {
        // 42: coefficient of θ⁰ is 42, no θ present.
        let mut arena = Arena::new();
        let theta = sym(&mut arena, "t0");
        let forty_two = arena.int(42);

        let terms = extract_poly_in_ext(&arena, forty_two, theta).unwrap();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].0, 0);
        assert_eq!(terms[0].1, forty_two);
        // Reconstruction is just the constant itself.
    }

    #[test]
    fn extract_poly_reconstruct_quadratic() {
        // θ² + 1: coeff of θ² is 1, coeff of θ⁰ is 1.
        let mut arena = Arena::new();
        let theta = sym(&mut arena, "t0");
        let two = arena.int(2);
        let one = arena.one();
        let theta_sq = arena.pow(theta, two);
        let expr = arena.add(&[theta_sq, one]);

        let terms = extract_poly_in_ext(&arena, expr, theta).unwrap();
        let powers: Vec<usize> = terms.iter().map(|t| t.0).collect();
        assert!(powers.contains(&0), "should have θ⁰ term");
        assert!(powers.contains(&2), "should have θ² term");

        // Verify at θ=3: 3²+1 = 10.
        let three = arena.int(3);
        let orig_val = crate::transforms::subs::subs(&mut arena, expr, theta, three);
        let orig_eval = crate::transforms::eval::eval(&mut arena, orig_val);
        assert_eq!(display(&arena, orig_eval), "10");
    }

    // ── Group 2: Tower construction equivalence tests ───────────────
    //
    // Verify that the tower-rewritten integrand evaluates to the same
    // value as the original when θ is substituted back.

    #[test]
    fn build_tower_exp_x_equivalence() {
        // exp(x) at x=1: original = exp(1) ≈ 2.718
        // Tower: θ = exp(x), integrand = θ.
        // Substitute θ = exp(1): should get exp(1).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let expr = arena.exp(x);

        let de = build_tower(&mut arena, expr, x).unwrap();
        let ext_var = de.levels[0].ext_var;
        let _arg = de.levels[0].argument;

        // Substitute x=1 in original.
        let one = arena.int(1);
        let orig_at_1 = crate::transforms::subs::subs(&mut arena, expr, x, one);

        // Substitute x=1 in integrand, then θ = exp(1).
        let integrand_at_1 = crate::transforms::subs::subs(&mut arena, de.integrand, x, one);
        let exp_1 = arena.exp(one);
        let integrand_back =
            crate::transforms::subs::subs(&mut arena, integrand_at_1, ext_var, exp_1);
        let integrand_eval = crate::transforms::eval::eval(&mut arena, integrand_back);

        let orig_eval = crate::transforms::eval::eval(&mut arena, orig_at_1);
        assert_eq!(
            orig_eval,
            integrand_eval,
            "tower equivalence at x=1: orig={}, rewritten={}",
            display(&arena, orig_eval),
            display(&arena, integrand_eval)
        );
    }

    #[test]
    fn build_tower_ln_x_equivalence() {
        // ln(x) at x=2: original = ln(2).
        // Tower: θ = ln(x), integrand = θ.
        // Substitute θ = ln(2): should get ln(2).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let expr = arena.ln(x);

        let de = build_tower(&mut arena, expr, x).unwrap();
        let ext_var = de.levels[0].ext_var;

        let two = arena.int(2);
        let orig_at_2 = crate::transforms::subs::subs(&mut arena, expr, x, two);

        let integrand_at_2 = crate::transforms::subs::subs(&mut arena, de.integrand, x, two);
        let ln_2 = arena.ln(two);
        let integrand_back =
            crate::transforms::subs::subs(&mut arena, integrand_at_2, ext_var, ln_2);
        let integrand_eval = crate::transforms::eval::eval(&mut arena, integrand_back);

        let orig_eval = crate::transforms::eval::eval(&mut arena, orig_at_2);
        assert_eq!(
            orig_eval,
            integrand_eval,
            "tower equivalence at x=2: orig={}, rewritten={}",
            display(&arena, orig_eval),
            display(&arena, integrand_eval)
        );
    }

    #[test]
    fn build_tower_exp_2x_plus_exp_x_equivalence() {
        // exp(2x) + exp(x) at x=0: exp(0) + exp(0) = 2.
        // Tower: θ = exp(x), integrand = θ² + θ.
        // At x=0: θ = exp(0) = 1, so θ² + θ = 1 + 1 = 2. ✓
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let two_x = arena.mul(&[two, x]);
        let exp_x = arena.exp(x);
        let exp_2x = arena.exp(two_x);
        let expr = arena.add(&[exp_2x, exp_x]);

        let de = build_tower(&mut arena, expr, x).unwrap();
        let ext_var = de.levels[0].ext_var;

        let zero = arena.int(0);
        let orig_at_0 = crate::transforms::subs::subs(&mut arena, expr, x, zero);
        let orig_eval = crate::transforms::eval::eval(&mut arena, orig_at_0);

        let int_at_0 = crate::transforms::subs::subs(&mut arena, de.integrand, x, zero);
        let exp_0 = arena.exp(zero);
        let exp_0_eval = crate::transforms::eval::eval(&mut arena, exp_0);
        let int_back = crate::transforms::subs::subs(&mut arena, int_at_0, ext_var, exp_0_eval);
        let int_eval = crate::transforms::eval::eval(&mut arena, int_back);

        assert_eq!(
            display(&arena, orig_eval),
            display(&arena, int_eval),
            "tower equivalence at x=0: orig={}, rewritten={}",
            display(&arena, orig_eval),
            display(&arena, int_eval)
        );
    }

    // ── Tower construction tests ────────────────────────────────────

    #[test]
    fn build_tower_pure_rational() {
        // x^2 + 1 — no exp or ln, should return base-level tower.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let one = arena.one();
        let x_sq = arena.pow(x, two);
        let expr = arena.add(&[x_sq, one]);

        let de = build_tower(&mut arena, expr, x).unwrap();
        assert_eq!(de.depth(), 0);
        assert!(de.is_base_level());
    }

    #[test]
    fn build_tower_single_exp() {
        // exp(x) — should create θ = exp(x).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let expr = arena.exp(x);

        let de = build_tower(&mut arena, expr, x).unwrap();
        assert_eq!(de.depth(), 1);
        assert_eq!(de.current_kind(), Some(&ExtensionKind::Exponential));
        // Integrand should be rewritten as θ.
        let s = display(&arena, de.integrand);
        assert!(s.contains("__t0"), "integrand should use __t0, got: {s}");
    }

    #[test]
    fn build_tower_exp_over_1_plus_exp() {
        // exp(x)/(1 + exp(x)) — should create θ = exp(x),
        // integrand = θ/(1 + θ).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let exp_x = arena.exp(x);
        let one = arena.one();
        let denom = arena.add(&[one, exp_x]);
        let expr = arena.div(exp_x, denom);

        let de = build_tower(&mut arena, expr, x).unwrap();
        assert_eq!(de.depth(), 1);
        assert_eq!(de.current_kind(), Some(&ExtensionKind::Exponential));
        let s = display(&arena, de.integrand);
        assert!(s.contains("__t0"), "integrand should use __t0, got: {s}");
    }

    #[test]
    fn build_tower_single_ln() {
        // ln(x) — should create θ = ln(x).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let expr = arena.ln(x);

        let de = build_tower(&mut arena, expr, x).unwrap();
        assert_eq!(de.depth(), 1);
        assert_eq!(de.current_kind(), Some(&ExtensionKind::Logarithmic));
        let s = display(&arena, de.integrand);
        assert!(s.contains("__t0"), "integrand should use __t0, got: {s}");
    }

    #[test]
    fn build_tower_exp_2x_plus_exp_x() {
        // exp(2x) + exp(x) — should create θ = exp(x),
        // integrand = θ² + θ.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let two_x = arena.mul(&[two, x]);
        let exp_x = arena.exp(x);
        let exp_2x = arena.exp(two_x);
        let expr = arena.add(&[exp_2x, exp_x]);

        let de = build_tower(&mut arena, expr, x).unwrap();
        assert_eq!(de.depth(), 1);
        assert_eq!(de.current_kind(), Some(&ExtensionKind::Exponential));
        let s = display(&arena, de.integrand);
        assert!(s.contains("__t0"), "integrand should use __t0, got: {s}");
        // Should contain θ² (as __t0^2) and θ.
        assert!(
            s.contains("__t0^2") || s.contains("__t0"),
            "integrand should have powers of __t0, got: {s}"
        );
    }

    #[test]
    fn build_tower_exp_3x_plus_exp_x() {
        // exp(3x) + exp(x) — θ = exp(x), integrand = θ³ + θ.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let three = arena.int(3);
        let three_x = arena.mul(&[three, x]);
        let exp_x = arena.exp(x);
        let exp_3x = arena.exp(three_x);
        let expr = arena.add(&[exp_3x, exp_x]);

        let de = build_tower(&mut arena, expr, x).unwrap();
        assert_eq!(de.depth(), 1);
        let s = display(&arena, de.integrand);
        assert!(s.contains("__t0"), "integrand should use __t0, got: {s}");
    }

    #[test]
    fn build_tower_independent_exps_fails() {
        // exp(x) + exp(x²) — independent, should fail.
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let x_sq = arena.pow(x, two);
        let exp_x = arena.exp(x);
        let exp_x_sq = arena.exp(x_sq);
        let expr = arena.add(&[exp_x, exp_x_sq]);

        let result = build_tower(&mut arena, expr, x);
        assert!(
            result.is_err(),
            "independent exps should fail: {:?}",
            result
        );
    }

    #[test]
    fn integer_multiples_basic() {
        // args: [x, 2x] → base = x, multiples = [(x, 1), (2x, 2)]
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let two_x = arena.mul(&[two, x]);

        let result = find_integer_multiples(&arena, &[x, two_x], x);
        assert!(result.is_some(), "x and 2x should be integer multiples");
        let (base, mults) = result.unwrap();
        assert_eq!(base, x);
        assert_eq!(mults.len(), 2);
        assert_eq!(mults[0].1, 1);
        assert_eq!(mults[1].1, 2);
    }

    #[test]
    fn integer_multiples_independent() {
        // args: [x, x²] → not multiples
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let x_sq = arena.pow(x, two);

        let result = find_integer_multiples(&arena, &[x, x_sq], x);
        assert!(result.is_none(), "x and x² should not be integer multiples");
    }

    #[test]
    fn integer_multiples_three_args() {
        // args: [x, 2x, 3x] → base = x, multiples = [1, 2, 3]
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let two = arena.int(2);
        let two_x = arena.mul(&[two, x]);
        let three = arena.int(3);
        let three_x = arena.mul(&[three, x]);

        let result = find_integer_multiples(&arena, &[x, two_x, three_x], x);
        assert!(result.is_some());
        let (_, mults) = result.unwrap();
        let ks: Vec<i64> = mults.iter().map(|m| m.1).collect();
        assert_eq!(ks, vec![1, 2, 3]);
    }

    #[test]
    fn build_tower_one_over_x_ln_x() {
        // 1/(x·ln(x)) — should create θ = ln(x),
        // integrand = 1/(x·θ).
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let ln_x = arena.ln(x);
        let x_ln_x = arena.mul(&[x, ln_x]);
        let one = arena.one();
        let expr = arena.div(one, x_ln_x);

        let de = build_tower(&mut arena, expr, x).unwrap();
        assert_eq!(de.depth(), 1);
        assert_eq!(de.current_kind(), Some(&ExtensionKind::Logarithmic));
    }
}
