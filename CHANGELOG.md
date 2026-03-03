# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

### Added

**Core Expression System**
- Arena-interned expression DAG with hash-consing and O(1) equality
- 28 `ExprNode` variants: numbers, symbols, arithmetic, 15 math functions, calculus forms
- Canonical ordering with precomputed sort keys
- Thread-safe `Ex` handle (`Send + Sync` via `Arc<RwLock>`)

**Mathematical Functions**
- Trigonometric: `sin`, `cos`, `tan`
- Inverse trigonometric: `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `tanh`
- Exponential/logarithmic: `exp`, `ln`
- Other: `sqrt`, `abs`, `pow`, `powi`

**Calculus**
- Symbolic differentiation with chain rule, product rule, all function types
- Indefinite integration: power rule, trig, exp, linearity, constant factor, integration by parts, u-substitution
- Definite integrals
- Taylor/Maclaurin series expansion with pole detection
- Symbolic limits via direct substitution, L'Hôpital's rule, and series fallback
- Fundamental theorem of calculus: `d/dx(∫ f dx) = f`

**Algebra**
- Algebraic expansion (distribute products, expand powers)
- Polynomial factoring with content extraction and multiplicity
- Collect by variable, common denominator (`together`)
- Fraction cancellation via polynomial GCD
- Equation solving: linear, quadratic, higher-degree via rational root theorem
- Linear system solving via Gaussian elimination over exact rationals

**Simplification**
- 9 rewrite rules: Pythagorean identities, inverse function pairs, sqrt/abs, hyperbolic identity
- Sub-expression matching in Add nodes
- Fixpoint iteration (`full_simplify`)
- Exact evaluation of 20+ special trig/exp/ln values

**Numerical Evaluation**
- Arbitrary-precision via `astro-float` (`evalf`)
- Convenience `f64` evaluation (`evalf_f64`)

**Assumption System**
- 23 mathematical properties with forward-chaining inference
- Fluent `.assume()` method for setting assumptions

**Expression Construction**
- Operator overloading for `Ex`, `&Ex`, `i64` (all combinations)
- `expr!` macro for natural math syntax
- `rule!` macro for rewrite rule definition
- `syms!` / `vars!` macros for symbol declaration
- Global default context (`symplex::var`, `symplex::int`, etc.)
- Runtime string parser (`symplex::parse::parse`)

**Introspection & Queries**
- `free_symbols`, `contains`, `degree`, `coeffs`, `coeff`, `term_count`
- `is_zero`, `is_positive`, `is_negative`, `is_real`, `is_integer`, `is_nonzero`, `is_finite`
- `is_polynomial`, `is_constant`, `as_numer_denom`
- `equals` with expand fallback

**Serialization**
- `ExprTree` serde type for JSON interchange
- `to_tree`, `to_json`, `to_json_pretty`
- `Context::from_tree`, `Context::from_json`

**Error Handling**
- `#[non_exhaustive]` error enum with `FreeSymbol`, `Unevaluable`, `ComputationFailed`, `PrecisionExhausted`, `ContradictoryAssumptions`
- `limit`, `solve`, `series` return `Result` for honest failure reporting
- Convenience `_or_self` / `_or_empty` methods for chainable usage

**Infrastructure**
- Zero-cost diagnostic logging via `tracing`
- REPL example (`cargo run --example repl`)
- 1091 tests including 33 proptest properties
- Criterion benchmarks for all core operations
- GitHub Actions CI (test + clippy + fmt)