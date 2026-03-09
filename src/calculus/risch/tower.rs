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
    pub fn push_logarithmic(
        &mut self,
        ext_var: ExprId,
        argument: ExprId,
        derivative: ExprId,
    ) {
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
    pub fn push_exponential(
        &mut self,
        ext_var: ExprId,
        argument: ExprId,
        derivative: ExprId,
    ) {
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
            // Check if they're integer multiples of a common argument.
            // For now, only handle the case where all are the same argument.
            // TODO: integer_powers grouping
            let base_arg = exp_args[0];
            for &arg in &exp_args[1..] {
                if arg != base_arg {
                    return Err("Multiple independent exp arguments not yet supported".into());
                }
            }
        }
        build_exp_tower(arena, expr, var, exp_args[0])
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

/// Build a single-level exponential tower: `θ = exp(u)`.
fn build_exp_tower(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    u: ExprId,
) -> Result<DifferentialExtension, String> {
    // Create the extension symbol θ.
    let theta = arena.symbol("__t0");

    // Compute Dθ = Du · θ  (where Du = d/dx(u))
    let du = crate::transforms::diff::diff(arena, u, var);
    let d_theta = arena.mul(&[du, theta]);

    // Rewrite the integrand: replace exp(u) with θ.
    let exp_u = arena.exp(u);
    let rewritten = crate::transforms::subs::subs(arena, expr, exp_u, theta);

    let mut de = DifferentialExtension::new(var);
    de.push_exponential(theta, u, d_theta);
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
pub fn derivation(
    arena: &mut Arena,
    expr: ExprId,
    de: &DifferentialExtension,
) -> ExprId {
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
        assert!(
            s.contains("x"),
            "D(θ) = 1/x for ln extension, got: {s}"
        );
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
        assert!(
            s.contains("__t0"),
            "integrand should use __t0, got: {s}"
        );
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
        assert!(
            s.contains("__t0"),
            "integrand should use __t0, got: {s}"
        );
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
        assert!(
            s.contains("__t0"),
            "integrand should use __t0, got: {s}"
        );
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
