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
- Inverse hyperbolic: `asinh`, `acosh`, `atanh` with derivatives and eval
- `cbrt()` (cube root) and `nthroot(n)` convenience methods
- `Sqrt` node removed — `sqrt(x)` is now `Pow(x, 1/2)` with display detection

**Calculus**
- Symbolic differentiation with chain rule, product rule, all function types
- Indefinite integration: power rule, trig, exp, linearity, constant factor, integration by parts, u-substitution
- Definite integrals
- Taylor/Maclaurin series expansion with pole detection
- Symbolic limits via direct substitution, L'Hôpital's rule, and series fallback
- Fundamental theorem of calculus: `d/dx(∫ f dx) = f`
- Numerical root finding via Newton's method (`nsolve`)

**Algebra**
- Algebraic expansion (distribute products, expand powers)
- Polynomial factoring with content extraction and multiplicity
- Collect by variable, common denominator (`together`)
- Fraction cancellation via polynomial GCD
- Equation solving: linear, quadratic, higher-degree via rational root theorem
- Linear system solving via Gaussian elimination over exact rationals
- Partial fraction decomposition (`apart`)
- Trigonometric expansion (`expand_trig`) — sin(a+b), cos(a+b) formulas
- Logarithm expansion (`expand_log`) — ln(a*b)→ln(a)+ln(b), ln(a^n)→n*ln(a)
- Polynomial GCD and LCM exposed on Ex
- u-substitution in integration for sin(ax+b), cos(ax+b), exp(ax+b)

**Simplification**
- 13 rewrite rules: Pythagorean identities, inverse function pairs, sqrt/abs, hyperbolic identity
- Sub-expression matching in Add nodes
- Fixpoint iteration (`full_simplify`)
- Exact evaluation of 20+ special trig/exp/ln values
- Power-of-power: `(a^m)^n → a^(m*n)`
- Inverse hyperbolic compositions: `asinh(sinh(w))→w`, `acosh(cosh(w))→w`, `atanh(tanh(w))→w`
- Odd/even function detection for all trig and hyperbolic functions
- Perfect nth root evaluation: `8^(1/3)→2`, `27^(1/3)→3`, `16^(1/4)→2`
- Irrational trig special values: sin(π/4)=√2/2, cos(π/6)=√3/2
- `Pow(E, x)` canonicalizes to `Exp(x)` — eliminates representational fork

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
- `limit`, `solve`, `series` return `Result` with `ComputationFailed` variant
- Convenience `_or_self` / `_or_empty` methods for chainable usage

**Infrastructure**
- Zero-cost diagnostic logging via `tracing`
- REPL example (`cargo run --example repl`)
- 1159 tests including 33 proptest properties
- 30 Criterion benchmarks for all core operations
- `exp_fn` renamed to `exp` for consistency
- `cargo clippy` clean, `cargo fmt` clean
- GitHub Actions CI (test + clippy + fmt)

**Sprint A-D: Math Depth & Ergonomics**
- Integration: `∫ tan(x) dx`, `∫ ln(x) dx`, `∫ tanh(x) dx`, `∫ 1/(x²+1) dx → atan(x)`, `∫ 1/√(1-x²) dx → asin(x)`, `∫ 1/√(x²+1) dx → asinh(x)`, `∫ 1/√(x²-1) dx → acosh(x)`, `∫ 1/(1-x²) dx → atanh(x)`, partial fraction → integrate pipeline, extended by-parts for `ln(x)·polynomial`
- Simplification: `sin(w)/cos(w) → tan(w)`, `sinh(w)/cosh(w) → tanh(w)`, `exp(a)*exp(b) → exp(a+b)` (16 rules total)
- Mul sub-expression matching in rewrite rule engine (matches Add sub-match)
- `logcombine()` method — inverse of `expand_log()`: `ln(a)+ln(b) → ln(a·b)`, `n·ln(a) → ln(aⁿ)`
- Complete unit circle evaluation: all 16 standard sin/cos angles via quadrant reduction, `tan(π/6)=√3/3`, `tan(π/3)=√3`
- `Ex::zero()` and `Ex::one()` class methods via global default context
- `Ex::expr_type() → ExprType` structural classification enum
- `Ex::replace(closure)` user-provided transformation walk via `walk_and_rebuild`
- `examples/calculus.rs` comprehensive workflow example