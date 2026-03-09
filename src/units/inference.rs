//! Runtime dimension inference for symbolic expressions.
//!
//! Given a mapping from variable names to physical dimensions, computes
//! the dimension of an arbitrary expression by walking the expression tree.
//!
//! This is used for:
//! - Validating `from_ex` assertions in debug builds
//! - Debugging dimension errors interactively
//! - Cross-checking symbolic derivations

use std::collections::HashMap;

use crate::prelude::Ex;
use super::dim::ConstDim;

// ---------------------------------------------------------------------------
// DimMap — variable → dimension mapping
// ---------------------------------------------------------------------------

/// Maps symbolic variable names to their physical dimensions.
///
/// # Examples
///
/// ```
/// use symplex::units::*;
///
/// let dims = DimMap::new()
///     .with("m", ConstDim::MASS)
///     .with("a", ConstDim::ACCELERATION)
///     .with("g", ConstDim::ACCELERATION)
///     .with("x", ConstDim::LENGTH);
/// ```
#[derive(Clone, Debug, Default)]
pub struct DimMap {
    map: HashMap<String, ConstDim>,
}

impl DimMap {
    /// Create an empty dimension map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable-dimension mapping. Chainable.
    pub fn with(mut self, name: &str, dim: ConstDim) -> Self {
        self.map.insert(name.to_string(), dim);
        self
    }

    /// Add a mapping from an expression's display representation.
    ///
    /// Extracts the variable name from the `Ex`'s `Display` output.
    pub fn with_var(mut self, var: &Ex, dim: ConstDim) -> Self {
        let name = format!("{}", var);
        self.map.insert(name, dim);
        self
    }

    /// Look up a variable's dimension.
    pub fn get(&self, name: &str) -> Option<&ConstDim> {
        self.map.get(name)
    }

    /// Insert a mapping in-place (non-chaining).
    pub fn insert(&mut self, name: &str, dim: ConstDim) {
        self.map.insert(name.to_string(), dim);
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the map has no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ConstDim extensions — pow and name
// ---------------------------------------------------------------------------

impl core::fmt::Display for ConstDim {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let n = self.name();
        if n != "Unknown" {
            return write!(f, "{}", n);
        }
        // Fall back to raw exponent notation
        write!(
            f,
            "L^{} M^{} T^{} I^{} Θ^{} N^{} J^{}",
            self.l, self.m, self.t, self.i, self.th, self.n, self.j
        )
    }
}

// ---------------------------------------------------------------------------
// infer_dimension — walk an expression tree and compute its dimension
// ---------------------------------------------------------------------------

/// Infer the physical dimension of an expression.
///
/// Returns `Ok(dim)` if the expression is dimensionally consistent,
/// or `Err(message)` describing the dimensional error.
///
/// # Dimension Rules
///
/// | Operation         | Rule                                                   |
/// |-------------------|--------------------------------------------------------|
/// | Number            | Dimensionless                                          |
/// | Symbol            | Look up in `DimMap`                                    |
/// | Constant (π, e)   | Dimensionless                                          |
/// | Add(a, b)         | `a` and `b` must have same dimension                   |
/// | Mul(a, b)         | Dimensions multiply (exponents add)                    |
/// | Pow(base, exp)    | `exp` must be dimensionless; base dimension scaled     |
/// | Neg(a)            | Same dimension as `a`                                  |
/// | sin, cos, exp, ln | Argument must be dimensionless; result is dimensionless |
/// | Derivative(f, x)  | `dim(f) / dim(x)`                                     |
/// | Integral(f, x)    | `dim(f) × dim(x)`                                     |
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::units::*;
///
/// let ctx = Context::new();
/// let ctx = ctx.clone(); symplex::syms!(ctx; m, a);
/// let dims = DimMap::new()
///     .with("m", ConstDim::MASS)
///     .with("a", ConstDim::ACCELERATION);
///
/// let d = infer_dimension(&(&m * &a), &dims).unwrap();
/// assert!(d.eq(ConstDim::FORCE));
/// ```
pub fn infer_dimension(expr: &Ex, dims: &DimMap) -> Result<ConstDim, String> {
    use crate::prelude::ExprType;

    match expr.expr_type() {
        ExprType::Number => Ok(ConstDim::DIMENSIONLESS),

        ExprType::Symbol => {
            let name = format!("{}", expr);
            dims.get(&name)
                .copied()
                .ok_or_else(|| {
                    format!(
                        "Unknown variable '{}' — not in dimension map",
                        name
                    )
                })
        }

        ExprType::Constant => {
            // Physical constants (c, h, k_B, …) display as their symbol name.
            // Check the DimMap first so callers can assign dimensions to them.
            let name = format!("{}", expr);
            if let Some(&dim) = dims.get(&name) {
                return Ok(dim);
            }
            // π, e, i, ∞ — all dimensionless
            Ok(ConstDim::DIMENSIONLESS)
        }

        ExprType::Add => {
            let args = expr.args();
            if args.is_empty() {
                return Ok(ConstDim::DIMENSIONLESS);
            }
            let first_dim = infer_dimension(&args[0], dims)?;
            for (i, arg) in args[1..].iter().enumerate() {
                let arg_dim = infer_dimension(arg, dims)?;
                if !first_dim.eq(arg_dim) {
                    return Err(format!(
                        "Dimension mismatch in addition: term 0 has dimension {} \
                         but term {} has dimension {}",
                        first_dim,
                        i + 1,
                        arg_dim,
                    ));
                }
            }
            Ok(first_dim)
        }

        ExprType::Mul => {
            let args = expr.args();
            let mut result = ConstDim::DIMENSIONLESS;
            for arg in &args {
                let arg_dim = infer_dimension(arg, dims)?;
                result = result.mul(arg_dim);
            }
            Ok(result)
        }

        ExprType::Pow => {
            let args = expr.args();
            if args.len() != 2 {
                return Err(format!(
                    "Pow must have exactly 2 arguments, got {}",
                    args.len()
                ));
            }
            let base_dim = infer_dimension(&args[0], dims)?;
            let exp_dim = infer_dimension(&args[1], dims)?;

            // Exponent must be dimensionless
            if !exp_dim.eq(ConstDim::DIMENSIONLESS) {
                return Err(format!(
                    "Exponent must be dimensionless, got {}",
                    exp_dim,
                ));
            }

            // If the base is already dimensionless, short-circuit
            if base_dim.eq(ConstDim::DIMENSIONLESS) {
                return Ok(ConstDim::DIMENSIONLESS);
            }

            // Try to extract an integer exponent for dimension scaling
            if let Ok(val) = args[1].eval_f64() {
                let n = val.round() as i8;
                if (val - f64::from(n)).abs() < 1e-10 {
                    return Ok(base_dim.pow(n));
                }
            }

            // Non-integer power of a dimensioned quantity — not allowed
            Err(format!(
                "Non-integer power of dimensioned quantity (base dimension: {})",
                base_dim,
            ))
        }

        ExprType::Neg => {
            let args = expr.args();
            if args.is_empty() {
                return Ok(ConstDim::DIMENSIONLESS);
            }
            infer_dimension(&args[0], dims)
        }

        ExprType::Function => {
            // sin, cos, tan, exp, ln, abs, etc.
            // Argument(s) must be dimensionless; result is dimensionless.
            let args = expr.args();
            for (i, arg) in args.iter().enumerate() {
                let dim = infer_dimension(arg, dims)?;
                if !dim.eq(ConstDim::DIMENSIONLESS) {
                    return Err(format!(
                        "Function argument {} must be dimensionless, got {} \
                         (in expression {})",
                        i, dim, expr,
                    ));
                }
            }
            Ok(ConstDim::DIMENSIONLESS)
        }

        ExprType::Apply => {
            // User-defined function application — treat like Function
            let args = expr.args();
            for (i, arg) in args.iter().enumerate() {
                let dim = infer_dimension(arg, dims)?;
                if !dim.eq(ConstDim::DIMENSIONLESS) {
                    return Err(format!(
                        "Applied function argument {} must be dimensionless, got {}",
                        i, dim,
                    ));
                }
            }
            Ok(ConstDim::DIMENSIONLESS)
        }

        ExprType::Derivative => {
            // d(body)/d(var) → dim(body) / dim(var)
            let args = expr.args();
            if args.len() >= 2 {
                let body_dim = infer_dimension(&args[0], dims)?;
                let var_dim = infer_dimension(&args[1], dims)?;
                Ok(body_dim.div(var_dim))
            } else {
                Err("Derivative must have at least body and variable".to_string())
            }
        }

        ExprType::Integral => {
            // ∫ body d(var) → dim(body) × dim(var)
            let args = expr.args();
            if args.len() >= 2 {
                let body_dim = infer_dimension(&args[0], dims)?;
                let var_dim = infer_dimension(&args[1], dims)?;
                Ok(body_dim.mul(var_dim))
            } else {
                Err("Integral must have at least body and variable".to_string())
            }
        }

        // Set expressions and any future variants — treat as dimensionless
        _ => Ok(ConstDim::DIMENSIONLESS),
    }
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Check whether an expression is dimensionally consistent without
/// returning the computed dimension.
///
/// Returns `Ok(())` on success, or `Err(message)` on dimensional mismatch.
pub fn check_dimensions(expr: &Ex, dims: &DimMap) -> Result<(), String> {
    infer_dimension(expr, dims).map(|_| ())
}

/// Infer the dimension and assert it matches the expected dimension.
///
/// Returns `Ok(())` if it matches, or `Err(message)` describing the mismatch.
pub fn assert_dimension(
    expr: &Ex,
    dims: &DimMap,
    expected: ConstDim,
) -> Result<(), String> {
    let actual = infer_dimension(expr, dims)?;
    if actual.eq(expected) {
        Ok(())
    } else {
        Err(format!(
            "Expected dimension {} but expression has dimension {}",
            expected, actual,
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a standard dimension map for tests.
    fn dims() -> DimMap {
        DimMap::new()
            .with("m", ConstDim::MASS)
            .with("a", ConstDim::ACCELERATION)
            .with("g", ConstDim::ACCELERATION)
            .with("v", ConstDim::VELOCITY)
            .with("t", ConstDim::TIME)
            .with("x", ConstDim::LENGTH)
            .with("k", ConstDim::STIFFNESS)
            .with("F", ConstDim::FORCE)
            .with("R", ConstDim::RESISTANCE)
            .with("I", ConstDim::CURRENT)
    }

    // --- Basic inference ---

    #[test]
    fn infer_mass_times_accel_is_force() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; m, a);
        let expr = &m * &a;
        let d = infer_dimension(&expr, &dims()).unwrap();
        assert!(d.eq(ConstDim::FORCE), "Expected Force, got {}", d);
    }

    #[test]
    fn infer_half_mv_squared_is_energy() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; m, v);
        // (1/2) * m * v^2
        let half = ctx.int(1) / ctx.int(2);
        let expr = &half * &m * v.powi(2);
        let d = infer_dimension(&expr, &dims()).unwrap();
        assert!(d.eq(ConstDim::ENERGY), "Expected Energy, got {}", d);
    }

    #[test]
    fn infer_add_mismatch_is_error() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; m, a);
        let expr = &m + &a;
        let result = infer_dimension(&expr, &dims());
        assert!(
            result.is_err(),
            "Adding Mass + Acceleration should be an error"
        );
    }

    #[test]
    fn infer_voltage_is_current_times_resistance() {
        let ctx = crate::api::context::Context::new();
        let i_var = ctx.symbol("I");
        let r_var = ctx.symbol("R");
        let expr = &i_var * &r_var;
        let d = infer_dimension(&expr, &dims()).unwrap();
        assert!(d.eq(ConstDim::VOLTAGE), "Expected Voltage, got {}", d);
    }

    #[test]
    fn infer_spring_force() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; k, x);
        let expr = &k * &x;
        let d = infer_dimension(&expr, &dims()).unwrap();
        assert!(d.eq(ConstDim::FORCE), "Expected Force, got {}", d);
    }

    #[test]
    fn infer_pure_number_is_dimensionless() {
        let ctx = crate::api::context::Context::new();
        let d = infer_dimension(&ctx.int(42), &dims()).unwrap();
        assert!(d.eq(ConstDim::DIMENSIONLESS));
    }

    #[test]
    fn infer_unknown_variable_is_error() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; unknown);
        let result = infer_dimension(&unknown, &dims());
        assert!(result.is_err(), "Unknown variable should produce an error");
    }

    // --- ConstDim::pow ---

    #[test]
    fn pow_length_squared_is_area() {
        let d = ConstDim::LENGTH.pow(2);
        assert!(d.eq(ConstDim::AREA));
    }

    #[test]
    fn pow_length_cubed_is_volume() {
        let d = ConstDim::LENGTH.pow(3);
        assert!(d.eq(ConstDim::VOLUME));
    }

    #[test]
    fn pow_zero_is_dimensionless() {
        let d = ConstDim::FORCE.pow(0);
        assert!(d.eq(ConstDim::DIMENSIONLESS));
    }

    #[test]
    fn pow_one_is_identity() {
        let d = ConstDim::VELOCITY.pow(1);
        assert!(d.eq(ConstDim::VELOCITY));
    }

    // --- ConstDim::name ---

    #[test]
    fn name_known_dimensions() {
        assert_eq!(ConstDim::FORCE.name(), "Force");
        assert_eq!(ConstDim::ENERGY.name(), "Energy");
        assert_eq!(ConstDim::VOLTAGE.name(), "Voltage");
        assert_eq!(ConstDim::DIMENSIONLESS.name(), "Dimensionless");
        assert_eq!(ConstDim::MASS.name(), "Mass");
        assert_eq!(ConstDim::LENGTH.name(), "Length");
        assert_eq!(ConstDim::TIME.name(), "Time");
    }

    #[test]
    fn name_unknown_dimension() {
        // An exotic dimension that doesn't match any named constant
        let exotic = ConstDim::new(3, 2, -1, 0, 0, 0, 0);
        assert_eq!(exotic.name(), "Unknown");
    }

    // --- DimMap ---

    #[test]
    fn dimmap_with_var() {
        let ctx = crate::api::context::Context::new();
        let x = ctx.symbol("x");
        let dm = DimMap::new().with_var(&x, ConstDim::LENGTH);
        assert_eq!(dm.get("x"), Some(&ConstDim::LENGTH));
    }

    #[test]
    fn dimmap_len_and_empty() {
        let dm = DimMap::new();
        assert!(dm.is_empty());
        assert_eq!(dm.len(), 0);

        let dm = dm.with("x", ConstDim::LENGTH);
        assert!(!dm.is_empty());
        assert_eq!(dm.len(), 1);
    }

    // --- Addition dimensional consistency ---

    #[test]
    fn infer_add_consistent_is_ok() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; m, a, g);
        // m*a + m*g — both are Force
        let expr = &m * &a + &m * &g;
        let d = infer_dimension(&expr, &dims()).unwrap();
        assert!(d.eq(ConstDim::FORCE), "Expected Force, got {}", d);
    }

    // --- Negation ---

    #[test]
    #[allow(non_snake_case)]
    fn infer_negation_preserves_dimension() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; F);
        let expr = -&F;
        let d = infer_dimension(&expr, &dims()).unwrap();
        assert!(d.eq(ConstDim::FORCE), "Expected Force, got {}", d);
    }

    // --- assert_dimension helper ---

    #[test]
    fn assert_dimension_ok() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; m, a);
        let expr = &m * &a;
        assert!(assert_dimension(&expr, &dims(), ConstDim::FORCE).is_ok());
    }

    #[test]
    fn assert_dimension_mismatch() {
        let ctx = crate::api::context::Context::new(); crate::syms!(ctx; m, a);
        let expr = &m * &a;
        assert!(assert_dimension(&expr, &dims(), ConstDim::ENERGY).is_err());
    }
}
