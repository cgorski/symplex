# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — Unreleased

### Added

**0.2.0: Type-Safe Boolean Expressions + Logic + Piecewise**
- BREAKING: `Ex` is now a type alias for `Expr<Numeric>`, not a standalone struct
- BREAKING: `gt()`, `ge()`, `lt()`, `le()`, `eq_expr()`, `ne_expr()` now return `BoolEx` (was `Ex`)
- BREAKING: `and_expr()`, `or_expr()`, `not_expr()` renamed to `and()`, `or()`, `not()` and moved to `BoolEx`
- BREAKING: `piecewise()` now takes `&[(&Ex, &BoolEx)]` (conditions must be boolean)
- New `Expr<S: Sort>` phantom-typed expression handle — compile-time sort safety
- New `BoolEx` type alias for `Expr<Boolean>` — boolean expressions
- New `Sort` trait with `Numeric` and `Boolean` marker types
- New ExprNode variants: `BoolTrue`, `BoolFalse`, `Gt`, `Ge`, `Eq_`, `Ne`, `And`, `Or`, `Not`, `Piecewise`
- Boolean evaluation: `5 > 3` → `True`, `And(True, False)` → `False`
- Piecewise differentiation: d/dx(Piecewise) differentiates each piece
- `BoolEx::and()`, `BoolEx::or()`, `BoolEx::not()` — boolean operations
- `BoolEx::eval()`, `BoolEx::simplify()`, `BoolEx::subs()` — sort-preserving
- `BoolEx::into_ex()`, `BoolEx::as_ex()` — escape hatches
- `expr!` macro: `>`, `<`, `>=`, `<=`, `!=`, `&&`, `||`, `!` operators
- Canonicalization: And/Or flatten, sort, deduplicate; Not double-negation
- Display: `True`, `False`, `x > 0`, `x & y`, `x | y`, `!x`, `Piecewise(...)`

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

**Cycle 6-8: Complex Numbers, Math Depth & SymPy-Inspired Improvements**
- Complex number canonicalization: `i²=-1` via mod-4, `(-1)^(1/2)→I`, `(-n)^(1/2)→I·√n`
- Complex quadratic roots: `x²+1=0 → [I, -I]`, `x²+2x+5=0 → [-1+2I, -1-2I]`
- Euler's formula: `exp(i·k·π) = cos(kπ) + i·sin(kπ)`, all unit circle angles
- Transcendental equation solving: `exp(x)=5→x=ln(5)`, `sin(x)=½→x=asin(½)`, `sqrt(x)=3→x=9`
- Mul-factor solving: `x·(x-1)·(x+2)=0 → {0, 1, -2}`
- Integer sqrt simplification: `√8→2√2`, `√12→2√3`, `√50→5√2`
- Trig-hyperbolic bridge: `sin(ix)=i·sinh(x)`, `cos(ix)=cosh(x)`
- Integration: `∫ asin(x)`, `∫ acos(x)`, `∫ atan(x)`, `∫ (ax+b)^n dx`, expand-then-retry
- sinh/cosh linear u-substitution
- `ln(-1)=iπ`, `ln(negative)=ln(|r|)+iπ`, `asin(1/2)=π/6`, `acos(1/2)=π/3`
- 6 forward-direction inverse function simplification rules (23 total)
- Assumption handlers for all 9 previously-unhandled function types
- Imaginary inference in `compute_mul`: `real·imaginary→imaginary`
- Node rebuilding fix in trig_expand, log_expand, log_combine
- `Poly::content()` bug fix (was returning leading_coeff)
- `expand()` now recurses through all 15 unary function types
- Global convenience: `symplex::pi()`, `symplex::e()`, `symplex::i_unit()`, `symplex::infinity()`
- `Ex::args()`, `Ex::diff_n()`, `Ex::log()`, `Ex::is_imaginary()`, `Ex::is_complex()`, `Ex::is_rational()`, `Ex::is_nonnegative()`, `Ex::is_nonpositive()`

**Cycles 9-18: Advanced Features**
- General u-substitution in integration (∫ 2x·exp(x²) dx, ∫ cos(x)·exp(sin(x)) dx)
- Trig power integration: ∫ sin^n(x) dx, ∫ cos^n(x) dx via recursive reduction
- Trig product-to-sum: sin(a)·cos(b) → ½[sin(a+b)+sin(a-b)] via `trig_combine()`
- Change-of-variable solver: exp(2x)-3·exp(x)+2=0 via t=exp(x) substitution
- Parser: float literals (3.14), implicit multiplication (2x), constant recognition (pi, I, E)
- Complex numerical evaluation (evalf Tier 3): full (real, imag) pair arithmetic
- `as_real_imag` decomposition: `re()` / `im()` methods
- `factor_terms()`: extract GCD of numeric coefficients from sums
- `From<i64/i32/u8/...>` for `Ex`, `Sum`/`Product` trait implementations
- Symbolic `Matrix` type: construct, transpose, add, matmul, determinant, trace, Jacobian
- `lambdify()`: compile expressions to `Box<dyn Fn(&[f64]) -> f64>`
- Common subexpression elimination (`cse()`)
- ODE solver (`dsolve`): separable, first-order linear, second-order constant-coefficient
- `Equation` type with solve/subs/simplify and `eq!` macro
- Factorial and Binomial node types with arbitrary-precision evaluation
- Canonical invariant checker (`verify_canonical`) with debug_assert integration
- Fixed `canon_mul` sort-order bug (sort by result sort key, not base sort key)
- Code hardening: unwrap→expect with documented invariants, display.rs defensive fallback
- `smart_simplify()` multi-strategy orchestrator with `count_ops()` measure
- Denominator rationalization (`rationalize_denom()`)
- `expr!` macro: constants (pi, E, I, oo), rationals (1/2), log(x, base)
- `matrix!` macro for natural matrix construction
- `eq!` macro for equation construction
- `rule!` macro conditional guards (`if condition`)

**Hardening: Correctness, Safety, and Testing**
- BREAKING: `replace()` now takes `Fn(ExprView<'_>) -> Option<Ex>` instead of `Fn(&Ex) -> Option<Ex>` — `ExprView` is a non-locking view type that makes deadlock structurally impossible at compile time
- BREAKING: `Piecewise` node uses `SmallVec<[(ExprId, ExprId); 3]>` pairs instead of flat `SmallVec<[ExprId; 6]>` — invalid odd-length states are now unrepresentable
- BREAKING: `ExprTree::Piecewise` serialization format changed from `Vec<ExprTree>` to `Vec<(ExprTree, ExprTree)>`
- Fixed: `smart_simplify()` no longer drops GCD factor from `factor_terms` (was silently returning wrong results)
- Fixed: Integration by-parts uses LIATE ordering with depth limit (20) — `∫ x·ln(x) dx` no longer stack overflows
- Fixed: `pow_pow` rule now requires at least one integer exponent (was producing wrong results for negative bases with fractional exponents)
- Fixed: Removed `asin(sin(w))→w`, `acos(cos(w))→w`, `atan(tan(w))→w` rules (incorrect outside principal branch)
- Fixed: `acosh(cosh(w))` now correctly returns `|w|` instead of `w`
- Fixed: `binom()` in trig power integration uses `BigInt` (was silently overflowing `i64`)
- Fixed: `nsolve()` uses `Ratio::from_float` instead of lossy `(x * 1e15) as i64` cast
- Fixed: `From<u64>` and `From<usize>` for `Ex` handle values > `i64::MAX` via `BigInt`
- Fixed: `ExprId`/`NumId`/`SymbolId` allocation uses `try_from().expect()` instead of silent `as u32` truncation
- Fixed: `cancel()` now cancels constant content factors (e.g., `(2x+2)/2 → x+1`)
- Fixed: `expand()` now propagates into `Derivative`, `Integral`, and `Apply` nodes
- Fixed: `eval_tan` quadrant reduction handles all standard angles (e.g., `tan(2π/3) = -√3`)
- Fixed: Odd roots of negative integers evaluate correctly (`(-8)^(1/3) = -2`)
- Fixed: `as_negated` handles 3+ child `Mul` nodes (`sin(-x*y) → -sin(x*y)`)
- Fixed: Parser uses `BigInt`/`Ratio<BigInt>` for arbitrary-precision input (was limited to `i64`)
- Fixed: Parser recursion depth limited to 256 (was unlimited, could stack overflow)
- New rule: `cos(w)/sin(w) → 1/tan(w)` (complement to existing sin/cos→tan)
- New rule: `exp(a*ln(b)) → b^a` (exp-log denesting)
- New: `expand_trig` handles `sin(n*x)` and `cos(n*x)` for integer n ≥ 2
- New: `trig_combine` runs eval pass to fold `sin(0)→0` artifacts
- New: `together()` uses polynomial LCM for common denominator (with product fallback)
- New: `ExprView` type for non-locking expression inspection
- New: `unary_compose_rule()` helper reduces rule definition boilerplate
- New: Arena constant fields changed from `pub` to `pub(crate)` with accessor methods
- New: `verify_canonical` uses O(n) `FxHashSet` instead of O(n²) nested loop
- New: `rebuild_with_cache` macro reduces inverse trig/hyperbolic boilerplate
- New: `Factorial`/`Binomial` supported in pattern `match_recursive`
- Testing: 5 concurrency tests (multi-thread construction, read+write, global context)
- Testing: 4 proptest value-preservation properties (simplify, smart_simplify, full_simplify, FTC)
- Testing: 15 per-rule numerical validation tests
- Testing: 11 negative tests verifying rules don't fire incorrectly
- Testing: 6 regression tests for critical bugs
- Testing: 3 test helper macros (`assert_simplifies_to!`, `assert_simplify_unchanged!`, `assert_simplify_preserves_value!`)
- Testing: Size assertion tests for ExprNode and Ex handle sizes
- Testing: Parser fuzz target (`fuzz/fuzz_targets/fuzz_parser.rs`)
- Testing: 30 bc-verified arbitrary-precision parser tests
- Dev-dependencies: added `trybuild`, `static_assertions`

**Gruntz Algorithm for Limits at Infinity**
- Complete implementation of the Gruntz algorithm (~1500 lines in `gruntz.rs`)
- MRV (Most Rapidly Varying) set computation with SubsSet tracking
- Expression rewriting in terms of ω → 0 with dependency-ordered substitutions
- Leading term extraction: structural + function-expansion-as-series for Laurent-like expressions
- Growth rate comparison via mutual recursion (limitinf ↔ compare)
- Sign determination with base cases for x, exp, polynomial powers
- Handles all elementary exp-log functions: `exp(-x)→0`, `ln(x)/x→0`, `x·exp(-x)→0`
- Gruntz is primary for limits at infinity; L'Hôpital remains primary for finite points
- 18 unit tests covering polynomial, exponential, logarithmic, and finite-point limits

**Canonicalization: Pow Flattening**
- `canon_pow` now flattens `Pow(Pow(a,b),c) → Pow(a,b*c)` for integer exponents
- Matches SymPy's auto-simplification at construction time
- Critical for Gruntz algorithm: ensures `1/(1/x) = x`
- `(x²)³ = x⁶` at construction time (was staying as `(x²)³`)

**SymPy Cross-Validation (263 fixtures)**
- Comprehensive cross-validation against SymPy 1.14.0
- 37 categories: diff (59), integrate (34), definite_integral (15), simplify (21),
  expand (14), solve (23), eval (15), series (10), limit (15), matrix (22),
  algebra (28), evalf (4), special_func (3)
- All comparisons are numerical (evaluate at same points, compare within tolerance)
- Parser accepts SymPy syntax natively (`**` for power, `Abs()`, `log()`)
- 252 pass, 0 fail, 2 not-implemented, 9 no-API
- scripts/generate_sympy_fixtures.py generates fixtures from SymPy
- tests/fixtures/sympy_cross_validation.json checked into repo

**Tracing Instrumentation**
- Added comprehensive tracing to 6 modules: pattern.rs, simplify_engine.rs,
  integrate.rs, limit.rs, gruntz.rs, canon.rs
- Enable with `RUST_LOG=symplex=debug` or per-module (e.g., `symplex::gruntz=trace`)
- Gruntz tracing shows expression values, MRV sets, rewrite steps, leadterm extraction
- Pattern tracing shows rule firings and sub-expression matches
- Integration tracing shows strategy selection and LIATE ordering
- Added `tracing-subscriber` and `tracing-test` as dev-dependencies

**Waves A–Y: Feature Parity Sprint**
- Reciprocal trig/hyperbolic: sec, csc, cot, acot, asec, acsc, coth, sech, csch, acoth, asech, acsch, sinc (Wave A)
- Atan2 ExprNode variant with full quadrant eval; arg(), conjugate() complex methods (Wave O)
- Assumption query methods: is_even, is_odd, is_prime, is_composite, is_algebraic, is_transcendental, is_irrational, is_hermitian (Wave Q)
- Cubic (Cardano) and quartic (Ferrari) polynomial solving (Wave C)
- 11 combinatorial functions via Apply nodes: fibonacci, lucas, bernoulli, harmonic, catalan, bell, euler_number, subfactorial, factorial2, rising_factorial, falling_factorial (Wave R)
- Matrix inverse, cofactor, adjugate, char_poly, eigenvals (Wave D)
- LU, QR decomposition; RREF, rank, nullspace, columnspace; norm, cross, dot, hstack, vstack, is_symmetric (Wave E)
- Rust code generation: to_rust_fn() with CSE (Wave F)
- trigsimp (6-strategy choice-set), powsimp (symbolic exponent merging), rewrite_as_exp/rewrite_as_trig (Euler's formula) (Wave K)
- check_solution, classify_ode (OdeType enum), checkodesol (Wave P)
- Vector calculus: gradient, divergence, curl, laplacian, is_conservative, is_solenoidal (Wave M)
- Logic connectives on BoolEx: xor, implies, equivalent, nand, nor, ite (Wave N)
- heaviside, dirac_delta, lambertw special functions (Wave S)
- Floor, Ceiling ExprNode variants; frac(), rem() convenience; Min, Max n-ary nodes (Wave B)
- Symbolic Sum and Product nodes with finite evaluation (Wave G)
- Special functions: Gamma, LogGamma, Digamma, Erf, Erfc, Beta ExprNode variants with eval/diff rules (Wave J)
- Residue computation via limit; Fourier series via integration (Wave T)
- Negative trig power integration (sec², csc², sec⁴, ...); cyclic IBP for exp·sin, exp·cos (Wave U)
- LU-based determinant for large matrices; 5 new proptests (Wave V)
- combsimp (factorial/binomial simplification), nsimplify (closed-form detection from floats) (Wave X)
- Laplace transform (forward table + structural rules) and inverse Laplace (partial fractions + table) (Wave Y)
- `expr!` macro expanded to 50+ functions including atan2, beta, gamma, erf, etc.

**Set Types and Inequality Solving**
- New `Expr<SetValued>` (aliased `SetEx`) — third phantom-typed sort for set-valued expressions
- New ExprNode variants: `EmptySet`, `UniversalSet`, `Interval(lower, upper, flags)`, `FiniteSet(SmallVec)`, `SetUnion(SmallVec)`, `SetIntersection(SmallVec)`, `SetComplement(set, universe)` — 7 set variants total
- Interval construction: `ex.closed_interval(&end)`, `ex.open_interval(&end)`
- Set canonicalization: flatten nested unions/intersections, sort, dedup, identity/annihilator rules (LatticeOp pattern)
- `Ex::solveset(&var)` — solve returning `SetEx` (FiniteSet) instead of `Vec<Ex>`
- `Ex::solve_gt(&var)`, `solve_ge(&var)`, `solve_lt(&var)`, `solve_le(&var)` — polynomial inequality solving returning `SetEx` (unions of intervals)
- New `src/inequalities.rs` module for polynomial and rational inequality solving

**Additional Methods**
- `Ex::ratsimp()` — rational simplification (cancel + together pipeline)
- `Ex::separatevars(&[&x, &y])` — separate multiplicative variable dependencies
- `Ex::evalf_complex64()` — complex numerical evaluation returning `(f64, f64)` pair
- `Ex::closed_form_sum()` — attempt closed-form evaluation of symbolic sums
- `Ex::is_convergent(&var)` — test series convergence

**Note:** The SymPy cross-validation suite (tests/test_sympy_cross_validation) should be re-run to capture improvements from Waves U–Y (sec² integration, cyclic IBP, Laplace transforms) and the set type additions.