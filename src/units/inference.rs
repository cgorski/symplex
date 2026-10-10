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

use num_traits::{One, ToPrimitive};
use rustc_hash::FxHashMap;

use super::dim::ConstDim;
use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::prelude::Ex;

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
/// | Operation                   | Rule                                                        |
/// |-----------------------------|-------------------------------------------------------------|
/// | Number, π, e, i, ∞          | Dimensionless                                               |
/// | Symbol                      | Look up in `DimMap`                                         |
/// | Physical constant, `f(x)`   | Looked up by display name in `DimMap`, else as below        |
/// | Add, Min, Max, Piecewise    | All operands (values) must have the same dimension          |
/// | Mul                         | Dimensions multiply (exponents add)                         |
/// | Pow(base, p/q)              | Exponent dimensionless; every exponent of `base` times `p/q` must be whole (`√(k/m)` is a frequency) |
/// | Neg, abs, re, im, conj      | Same dimension as the argument                              |
/// | sign, Heaviside, arg        | Any argument; dimensionless                                 |
/// | DiracDelta(a)               | `1 / dim(a)`                                                |
/// | atan2(y, x), relations      | Operands of one dimension; dimensionless                    |
/// | sin, exp, ln, floor, …      | Argument must be dimensionless; result is dimensionless     |
/// | Derivative(f, x)            | `dim(f) / dim(x)`                                           |
/// | Integral(f, x), ∫ₐᵇ f dx    | `dim(f) × dim(x)` (bounds of the dimension of `x`)          |
/// | Sum(f, k, a, b), Limit, Subs| `dim(f)` (index and bounds dimensionless)                   |
/// | Product(f, k, a, b)         | `dim(f)^(b − a + 1)` for constant integer bounds             |
/// | Series(f, x, x₀, n)         | `dim(f)` (`x₀` of the dimension of `x`)                       |
/// | Residue(f, z, z₀)           | `dim(f) × dim(z)`                                            |
/// | LaplaceTransform(f, t, s)   | `dim(f) × dim(t)` with `dim(s) = 1/dim(t)`; the inverse transform `dim(F) × dim(s)` |
///
/// Exponents that leave the range of `i8` are an error (they used to
/// overflow: a panic in debug builds, a wrapped dimension in release).
/// The walk is iterative, so deep expressions do not exhaust the stack.
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
    let inner = expr.inner.read();
    infer_in_arena(&inner.arena, expr.raw_id(), dims)
}

/// `d` raised to the rational power `r`, when every exponent stays whole
/// and within `i8`.
fn dim_pow(d: ConstDim, r: &Q) -> Result<ConstDim, String> {
    let scale = |e: i8| -> Result<i8, String> {
        let v = Q::from_integer(e.into()) * r;
        if !v.is_integer() {
            return Err(format!(
                "Non-integer power of dimensioned quantity: ({d})^{r} is not a whole dimension"
            ));
        }
        v.to_integer()
            .to_i8()
            .ok_or_else(|| format!("Dimension exponent overflow in ({d})^{r}"))
    };
    Ok(ConstDim {
        l: scale(d.l)?,
        m: scale(d.m)?,
        t: scale(d.t)?,
        i: scale(d.i)?,
        th: scale(d.th)?,
        n: scale(d.n)?,
        j: scale(d.j)?,
    })
}

/// `a·b` (exponents added), refusing an exponent outside `i8`.
fn dim_mul(a: ConstDim, b: ConstDim) -> Result<ConstDim, String> {
    let add = |x: i8, y: i8| {
        x.checked_add(y)
            .ok_or_else(|| format!("Dimension exponent overflow in ({a})·({b})"))
    };
    Ok(ConstDim {
        l: add(a.l, b.l)?,
        m: add(a.m, b.m)?,
        t: add(a.t, b.t)?,
        i: add(a.i, b.i)?,
        th: add(a.th, b.th)?,
        n: add(a.n, b.n)?,
        j: add(a.j, b.j)?,
    })
}

/// `a/b`, refusing an exponent outside `i8`.
fn dim_div(a: ConstDim, b: ConstDim) -> Result<ConstDim, String> {
    dim_mul(a, dim_pow(b, &-Q::one())?)
}

/// The exact rational value of an exponent node (`p/q`, or `−p/q` as a
/// negation), else an integer value certified by the evaluator.
fn exponent_value(arena: &Arena, e: ExprId) -> Option<Q> {
    if let Some(r) = arena.as_num(e) {
        return Some(r.clone());
    }
    if let ExprNode::Neg(inner) = arena.node(e)
        && let Some(r) = arena.as_num(*inner)
    {
        return Some(-r.clone());
    }
    let v = crate::transforms::evalf::evalf_f64(arena, e).ok()?;
    let n = v.round();
    ((v - n).abs() < 1e-10 && n.abs() < 1e6).then(|| Q::from_integer((n as i64).into()))
}

/// [`infer_dimension`] on the arena: one bottom-up pass over the post-order
/// of `root` (no recursion).
fn infer_in_arena(arena: &Arena, root: ExprId, dims: &DimMap) -> Result<ConstDim, String> {
    let dimensionless = ConstDim::DIMENSIONLESS;
    let mut dim_of: FxHashMap<ExprId, ConstDim> = FxHashMap::default();
    for id in walk::post_order_ids(arena, root) {
        let get = |c: &ExprId| -> Result<ConstDim, String> {
            dim_of
                .get(c)
                .copied()
                .ok_or_else(|| "internal error: operand without a dimension".to_string())
        };
        // All operands of one dimension (Add, Min, Max, relations, bounds,
        // …).  `0` and the infinities fit every dimension (`t > 0`,
        // `∫₀^T`, `max(x, 0)`, `x → ∞`).
        let polymorphic = |c: ExprId| {
            arena.is_zero_structural(c)
                || matches!(
                    arena.node(c),
                    ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity
                )
        };
        let same = |ops: &[ExprId], what: &str| -> Result<ConstDim, String> {
            let mut first: Option<(usize, ConstDim)> = None;
            for (i, &c) in ops.iter().enumerate() {
                if polymorphic(c) {
                    continue;
                }
                let d = get(&c)?;
                match first {
                    None => first = Some((i, d)),
                    Some((i0, d0)) if !d0.eq(d) => {
                        return Err(format!(
                            "Dimension mismatch in {what}: term {i0} has dimension {d0} \
                             but term {i} has dimension {d}"
                        ));
                    }
                    Some(_) => {}
                }
            }
            Ok(first.map_or(dimensionless, |(_, d)| d))
        };
        let node = arena.node(id);
        let d = match node {
            ExprNode::Num(_) => dimensionless,
            ExprNode::Symbol(sid) => {
                let name = arena.symbol_name(*sid);
                *dims
                    .get(name)
                    .ok_or_else(|| format!("Unknown variable '{name}' — not in dimension map"))?
            }
            // Physical constants (c, h, k_B, …) display as their symbol
            // name; the map may give them a dimension.
            ExprNode::PhysicalConstant(..) => dims
                .get(&arena.display(id).to_string())
                .copied()
                .unwrap_or(dimensionless),
            ExprNode::Add(ops) => same(ops, "addition")?,
            ExprNode::Min(ops) | ExprNode::Max(ops) => same(ops, "min/max")?,
            ExprNode::Mul(ops) => {
                let mut acc = dimensionless;
                for c in ops.iter() {
                    acc = dim_mul(acc, get(c)?)?;
                }
                acc
            }
            ExprNode::Pow(b, e) => {
                let exp_dim = get(e)?;
                if !exp_dim.eq(dimensionless) {
                    return Err(format!("Exponent must be dimensionless, got {exp_dim}"));
                }
                let base_dim = get(b)?;
                if base_dim.eq(dimensionless) {
                    dimensionless
                } else {
                    let r = exponent_value(arena, *e).ok_or_else(|| {
                        format!(
                            "Non-constant power of a dimensioned quantity (base dimension: {base_dim})"
                        )
                    })?;
                    dim_pow(base_dim, &r)?
                }
            }
            ExprNode::Neg(a)
            | ExprNode::Abs(a)
            | ExprNode::Re(a)
            | ExprNode::Im(a)
            | ExprNode::Conjugate(a) => get(a)?,
            // The argument (phase) of a quantity is dimensionless, like its sign.
            ExprNode::Sign(a) | ExprNode::Heaviside(a) | ExprNode::Arg(a) => {
                get(a)?;
                dimensionless
            }
            ExprNode::DiracDelta(a) => dim_div(dimensionless, get(a)?)?,
            ExprNode::Atan2(y, x) => {
                same(&[*y, *x], "atan2")?;
                dimensionless
            }
            ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
                same(&[*a, *b], "a relation")?;
                dimensionless
            }
            ExprNode::And(_)
            | ExprNode::Or(_)
            | ExprNode::Not(_)
            | ExprNode::BoolTrue
            | ExprNode::BoolFalse => dimensionless,
            ExprNode::Piecewise(pieces) => {
                let values: Vec<ExprId> = pieces.iter().map(|&(v, _)| v).collect();
                same(&values, "a piecewise expression")?
            }
            ExprNode::Derivative(f, x) => dim_div(get(f)?, get(x)?)?,
            ExprNode::Integral(f, x) => dim_mul(get(f)?, get(x)?)?,
            ExprNode::DefiniteIntegral(f, x, lo, hi) => {
                same(&[*x, *lo, *hi], "the bounds of an integral")?;
                dim_mul(get(f)?, get(x)?)?
            }
            ExprNode::Sum(f, k, lo, hi) => {
                for c in [k, lo, hi] {
                    let d = get(c)?;
                    if !d.eq(dimensionless) {
                        return Err(format!(
                            "Summation index and bounds must be dimensionless, got {d}"
                        ));
                    }
                }
                get(f)?
            }
            ExprNode::Product_(f, k, lo, hi) => {
                for c in [k, lo, hi] {
                    let d = get(c)?;
                    if !d.eq(dimensionless) {
                        return Err(format!(
                            "Product index and bounds must be dimensionless, got {d}"
                        ));
                    }
                }
                let df = get(f)?;
                if df.eq(dimensionless) {
                    dimensionless
                } else {
                    // ∏_{k=lo}^{hi} f has dim(f)^(hi − lo + 1).
                    let count = exponent_value(arena, *hi)
                        .zip(exponent_value(arena, *lo))
                        .map(|(h, l)| h - l + Q::one())
                        .filter(|n| n.is_integer() && *n >= Q::from_integer(0.into()))
                        .ok_or_else(|| {
                            format!("Product of a dimensioned factor ({df}) over a symbolic range")
                        })?;
                    dim_pow(df, &count)?
                }
            }
            ExprNode::Limit(f, x, point) => {
                same(&[*x, *point], "a limit point")?;
                get(f)?
            }
            ExprNode::Series(f, x, point, order) => {
                same(&[*x, *point], "a series expansion point")?;
                let d = get(order)?;
                if !d.eq(dimensionless) {
                    return Err(format!("Series order must be dimensionless, got {d}"));
                }
                get(f)?
            }
            ExprNode::Residue(f, z, point) => {
                same(&[*z, *point], "a residue point")?;
                dim_mul(get(f)?, get(z)?)?
            }
            // ∫₀^∞ f(t)·e^{−st} dt: dim(f)·dim(t), with s of dimension 1/t
            // (st is an exponent).  The inverse transform divides again.
            ExprNode::LaplaceTransform(f, t, s) | ExprNode::InverseLaplaceTransform(f, s, t) => {
                let (dt, ds) = (get(t)?, get(s)?);
                if !dim_mul(dt, ds)?.eq(dimensionless) {
                    return Err(format!(
                        "Laplace variables must have reciprocal dimensions, got {dt} and {ds}"
                    ));
                }
                let wrt = if matches!(node, ExprNode::LaplaceTransform(..)) {
                    dt
                } else {
                    ds
                };
                dim_mul(get(f)?, wrt)?
            }
            ExprNode::Subs(f, x, value) => {
                same(&[*x, *value], "a substitution")?;
                get(f)?
            }
            ExprNode::Apply(..) if dims.get(&arena.display(id).to_string()).is_some() => *dims
                .get(&arena.display(id).to_string())
                .unwrap_or(&dimensionless),
            // Constants, sets and the remaining forms: dimensionless.
            ExprNode::Pi
            | ExprNode::E
            | ExprNode::ImaginaryUnit
            | ExprNode::EulerGamma
            | ExprNode::Catalan
            | ExprNode::GoldenRatio
            | ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN
            | ExprNode::EmptySet
            | ExprNode::UniversalSet
            | ExprNode::Interval(..)
            | ExprNode::FiniteSet(_)
            | ExprNode::SetUnion(_)
            | ExprNode::SetIntersection(_)
            | ExprNode::SetComplement(..)
            | ExprNode::ConditionSet(..)
            | ExprNode::ImageSet(..) => dimensionless,
            // Every other function (sin, exp, ln, floor, Γ, f(x), …): its
            // arguments must be dimensionless.
            other => {
                let mut bad: Option<(usize, ConstDim)> = None;
                let mut k = 0usize;
                other.for_each_child(|c| {
                    let d = dim_of.get(&c).copied().unwrap_or(dimensionless);
                    if bad.is_none() && !d.eq(dimensionless) {
                        bad = Some((k, d));
                    }
                    k += 1;
                });
                if let Some((i, d)) = bad {
                    return Err(if matches!(other, ExprNode::Apply(..)) {
                        format!("Applied function argument {i} must be dimensionless, got {d}")
                    } else {
                        format!(
                            "Function argument {i} must be dimensionless, got {d} \
                             (in expression {})",
                            arena.display(id)
                        )
                    });
                }
                dimensionless
            }
        };
        dim_of.insert(id, d);
    }
    dim_of
        .get(&root)
        .copied()
        .ok_or_else(|| "internal error: no dimension for the expression".to_string())
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
pub fn assert_dimension(expr: &Ex, dims: &DimMap, expected: ConstDim) -> Result<(), String> {
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
        let ctx = crate::api::context::Context::new();
        crate::syms!(ctx; m, a);
        let expr = &m * &a;
        let d = infer_dimension(&expr, &dims()).unwrap();
        assert!(d.eq(ConstDim::FORCE), "Expected Force, got {}", d);
    }

    #[test]
    fn infer_half_mv_squared_is_energy() {
        let ctx = crate::api::context::Context::new();
        crate::syms!(ctx; m, v);
        // (1/2) * m * v^2
        let half = ctx.int(1) / ctx.int(2);
        let expr = &half * &m * v.powi(2);
        let d = infer_dimension(&expr, &dims()).unwrap();
        assert!(d.eq(ConstDim::ENERGY), "Expected Energy, got {}", d);
    }

    #[test]
    fn infer_add_mismatch_is_error() {
        let ctx = crate::api::context::Context::new();
        crate::syms!(ctx; m, a);
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
        let ctx = crate::api::context::Context::new();
        crate::syms!(ctx; k, x);
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
        let ctx = crate::api::context::Context::new();
        crate::syms!(ctx; unknown);
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
        let ctx = crate::api::context::Context::new();
        crate::syms!(ctx; m, a, g);
        // m*a + m*g — both are Force
        let expr = &m * &a + &m * &g;
        let d = infer_dimension(&expr, &dims()).unwrap();
        assert!(d.eq(ConstDim::FORCE), "Expected Force, got {}", d);
    }

    // --- Negation ---

    #[test]
    #[allow(non_snake_case)]
    fn infer_negation_preserves_dimension() {
        let ctx = crate::api::context::Context::new();
        crate::syms!(ctx; F);
        let expr = -&F;
        let d = infer_dimension(&expr, &dims()).unwrap();
        assert!(d.eq(ConstDim::FORCE), "Expected Force, got {}", d);
    }

    // --- assert_dimension helper ---

    #[test]
    fn assert_dimension_ok() {
        let ctx = crate::api::context::Context::new();
        crate::syms!(ctx; m, a);
        let expr = &m * &a;
        assert!(assert_dimension(&expr, &dims(), ConstDim::FORCE).is_ok());
    }

    #[test]
    fn assert_dimension_mismatch() {
        let ctx = crate::api::context::Context::new();
        crate::syms!(ctx; m, a);
        let expr = &m * &a;
        assert!(assert_dimension(&expr, &dims(), ConstDim::ENERGY).is_err());
    }
}
