# Symplex

Symbolic mathematics library for Rust.

## Features

- **Expression building** — operator overloading (`+`, `-`, `*`, `/`, unary `-`), method chaining (`.pow()`, `.sin()`, `.diff()`), automatic canonicalization (flatten, sort, combine like terms)
- **Proc macros** — `expr!(x^2 + 2*x + 1)` for natural math syntax with constants (`pi`, `E`, `I`) and rationals (`1/2`); `rule!(arena, "name", LHS => RHS)` for rewrite rules; `matrix!` and `eq!` for matrices and equations
- **18 math functions** — sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, ln, abs, sqrt, cbrt, nthroot
- **Differentiation** — all elementary functions, chain rule, product rule, n-ary generalization, higher-order derivatives, partial derivatives
- **Integration** — power rule, trig, exp, tan, ln, linearity, constant factor, integration by parts, u-substitution (`sin(ax+b)`, `cos(ax+b)`, `exp(ax+b)`), inverse trig/hyperbolic standard forms, partial fraction decomposition pipeline, definite integrals, inverse trig antiderivatives (asin, acos, atan), general linear substitution (ax+b)^n, expand-then-integrate fallback
- **Taylor series** — expansion around any point with configurable order and pole detection
- **Limits** — direct substitution, L'Hôpital's rule (0/0 and ∞/∞), series fallback
- **Equation solving** — polynomial (linear, quadratic, higher-degree via rational root theorem), complex roots, linear systems (Gaussian elimination), numerical root finding (Newton's method), transcendental equations via inversion peeling (exp, ln, sin, cos, tan, sqrt) with change-of-variable, Mul-factor solving
- **Simplification** — 23 rewrite rules (incl. sin/cos→tan ratio, exp combining, abs-positive) with sub-expression matching, fixpoint iteration via `full_simplify()`, multi-strategy `smart_simplify()`
- **Algebraic manipulation** — expand, factor, collect, together, cancel, partial fractions (`apart`), trig expansion (`expand_trig`), log expansion (`expand_log`), logcombine
- **Exact evaluation** — 30+ special values for trig/exp/ln including irrational values (√2/2, √3/2), perfect nth root evaluation, odd/even function detection, integer sqrt simplification (√8→2√2), trig-hyperbolic bridge (sin(ix)=i·sinh(x))
- **Complex numbers** — i²=-1 canonicalization, (-1)^(1/2)→i, (-n)^(1/2)→i√n, complex quadratic roots, Euler's formula exp(iπ)=-1
- **Complex decomposition** — `re()` / `im()` split expressions into real and imaginary parts
- **Symbolic matrices** — construct, multiply, transpose, determinant, trace, Jacobian computation
- **ODE solver** — separable, first-order linear, second-order constant-coefficient
- **lambdify** — compile expressions to `Fn(f64) -> f64` closures for fast numerical evaluation
- **Common subexpression elimination (CSE)** — extract shared subexpressions for code generation
- **Factorial and binomial coefficients** — `n!` and `C(n,k)` with arbitrary-precision evaluation
- **Denominator rationalization** — clear square roots from denominators
- **Arbitrary-precision numerical evaluation** — via `astro-float`, any number of decimal digits
- **Assumption system** — 23 mathematical properties (positive, real, integer, etc.) with forward-chaining inference
- **Pattern matching** — wild symbols, named rewrite rules, simplification with trace
- **Polynomial algebra** — dense univariate over ℚ, arithmetic, Euclidean GCD/LCM, degree, coefficients
- **Serde serialization** — `ExprTree` for JSON interchange with round-trip support
- **Runtime parser** — `symplex::parse::parse(&ctx, "x^2 + 1")` for REPL and dynamic construction
- **Zero-cost tracing** — diagnostic logging via the `tracing` crate
- **Thread safety** — `Ex` is `Send + Sync`; the arena uses `parking_lot::RwLock`

## Quick Start

```rust
use symplex::prelude::*;

// ── Quick setup — no Context boilerplate needed ────────────────
let x = symplex::var("x");
let y = symplex::var("y");

// ── Build expressions with natural math syntax ─────────────────
let f = expr!(x^2 + 2*x + 1);
println!("{f}");                             // 1 + x^2 + 2*x

// ── Differentiate and integrate ────────────────────────────────
let deriv = expr!(x^3).diff(&x);
println!("{deriv}");                         // 3*x^2

let anti = expr!(x^2).integrate(&x);
println!("{anti}");                          // 1/3*x^3

// ── Evaluate at a point ────────────────────────────────────────
let at_2 = deriv.subs_i64(&x, 2);
println!("{at_2}");                          // 12

// ── Taylor series ──────────────────────────────────────────────
let s = x.sin().maclaurin(&x, 4).unwrap();
println!("{}", s.expand().eval());           // x - 1/6*x^3

// ── Limits ─────────────────────────────────────────────────────
let lim = (&x.sin() / &x).limit(&x, &symplex::int(0)).unwrap();
println!("{lim}");                           // 1

// ── Solve equations ────────────────────────────────────────────
let roots = expr!(x^2 - 5*x + 6).solve(&x).unwrap();
for r in &roots { println!("x = {r}"); }    // x = 2, x = 3

// ── Numerical root finding ─────────────────────────────────────
let root = (&x - &x.cos()).nsolve(&x, 1.0, 50, 1e-12).unwrap();
println!("x = {root:.10}");                 // x = 0.7390851332

// ── Factor and simplify ────────────────────────────────────────
let factored = (&x.powi(2) - 1).factor(&x);
println!("{factored}");                      // (-1 + x)*(1 + x)

let trig = expr!(sin(x)^2 + cos(x)^2);
println!("{}", trig.simplify());             // 1

// ── Matrices ─────────────────────────────────────────────────
let m = matrix![[x, 1], [0, x^2]];
println!("det = {}", m.det());              // x^3

// ── Complex numbers ─────────────────────────────────────────
let i = ctx.i_unit();
assert_eq!(format!("{}", i.powi(2)), "-1");                // i² = -1
assert_eq!(format!("{}", ctx.int(-1).sqrt()), "I");        // √(-1) = i

// Euler's formula
let euler = (&i * &ctx.pi()).exp().eval();
assert_eq!(format!("{euler}"), "-1");                       // e^(iπ) = -1

// ── Advanced: explicit Context for custom configuration ────────
let ctx = Context::new();
let t = ctx.symbol_with("t", &[Assumption::Positive, Assumption::Real]);
assert_eq!(t.is_positive(), Some(true));
```

## API Reference

### Context

```rust
Context::new()                               // default configuration
Context::with_config(EvalConfig { ... })     // custom limits

ctx.symbol("x")                              // symbolic variable
ctx.symbol_with("t", &[Assumption::Positive]) // variable with assumptions
ctx.int(5)                                   // integer
ctx.rational(1, 3)                           // exact fraction 1/3
ctx.pi() / ctx.e() / ctx.i_unit()           // constants
ctx.infinity() / ctx.nan()                   // special values
ctx.query(&expr, Props::POSITIVE)            // query assumption
ctx.with_arena_mut(|arena| { ... })          // direct arena access
ctx.solve_system(&[eq1, eq2], &[x, y])      // linear system solving
ctx.from_tree(&tree) / ctx.from_json(json)   // deserialization
ctx.node_count()                             // arena info
```

### Expression Methods

```rust
// ── Math functions ─────────────────────────────────────────────
ex.pow(&exp)    ex.powi(3)     ex.sin()      ex.cos()
ex.tan()        ex.exp()       ex.ln()       ex.sqrt()
ex.abs()        ex.asin()      ex.acos()     ex.atan()
ex.sinh()       ex.cosh()      ex.tanh()     ex.asinh()
ex.acosh()      ex.atanh()     ex.cbrt()     ex.nthroot(n)

// ── Calculus ───────────────────────────────────────────────────
ex.diff(&x)                                  // symbolic derivative
ex.diff_n(&x, n)                             // nth derivative
ex.integrate(&x)                             // indefinite integral
ex.definite_integral(&x, &lower, &upper)     // definite integral
ex.series(&x, &point, order)                 // Taylor series → Result
ex.maclaurin(&x, order)                      // Maclaurin series → Result
ex.limit(&x, &point)                         // symbolic limit → Result

// ── Algebra ────────────────────────────────────────────────────
ex.expand()                                  // distribute products
ex.factor(&var)                              // polynomial factoring
ex.collect(&var)                             // group by powers
ex.together()                                // common denominator
ex.cancel(&var)                              // cancel common factors
ex.apart(&var)                               // partial fractions
ex.expand_trig()                             // sin(a+b) → sin(a)cos(b)+...
ex.expand_log()                              // ln(a*b) → ln(a)+ln(b)
ex.logcombine()                              // ln(a)+ln(b) → ln(a*b)
ex.log(&base)                                // arbitrary-base logarithm
ex.factor_terms()                            // extract GCD of coefficients
ex.rationalize_denom()                       // clear radicals from denominators
ex.solve(&var)                               // solve expr=0 → Result
ex.nsolve(&var, guess, max_iter, tol)        // numerical root → Result

// ── Simplification ─────────────────────────────────────────────
ex.simplify()                                // one-pass rewrite rules
ex.simplify_trace()                          // with step-by-step trace
ex.full_simplify()                           // fixpoint: eval+expand+simplify
ex.full_simplify_trace()                     // with accumulated trace
ex.smart_simplify()                          // multi-strategy simplification
ex.eval()                                    // evaluate special values
ex.count_ops()                               // expression complexity

// ── Complex decomposition ──────────────────────────────────────
ex.re()                                      // real part
ex.im()                                      // imaginary part

// ── Code generation ────────────────────────────────────────────
ex.lambdify(&["x", "y"])                     // compile to Fn(f64) → f64 closure
ex.cse()                                     // common subexpression elimination

// ── Substitution ───────────────────────────────────────────────
ex.subs(&old, &new)                          // structural substitution
ex.subs_i64(&old, n)                         // substitute with integer
ex.subs_map(&[(&x, &a), (&y, &b)])          // simultaneous substitution

// ── Numerical evaluation ───────────────────────────────────────
ex.evalf(50)                                 // arbitrary precision → Result<String>
ex.evalf_f64()                               // f64 convenience → Result<f64>

// ── Queries ────────────────────────────────────────────────────
ex.is_zero()        ex.is_positive()         ex.is_negative()
ex.is_real()        ex.is_integer()          ex.is_nonzero()
ex.is_finite()      ex.query(Props::...)     ex.equals(&other)
ex.is_imaginary()   ex.is_complex()          ex.is_rational()
ex.is_nonnegative() ex.is_nonpositive()
ex.is_zero_structural()    ex.is_one_structural()
ex.is_constant()    ex.is_polynomial(&var)
ex.expr_type()                               // → ExprType

// ── Introspection ──────────────────────────────────────────────
ex.free_symbols()                            // → Vec<Ex>
ex.contains(&sub)                            // → bool
ex.degree(&var)                              // → Option<usize>
ex.coeffs(&var)                              // → Option<Vec<Ex>>
ex.coeff(&var, n)                            // → Option<Ex>
ex.as_numer_denom()                          // → (Ex, Ex)
ex.term_count()                              // → usize
ex.poly_gcd(&other, &var)                    // → Option<Ex>
ex.poly_lcm(&other, &var)                    // → Option<Ex>
ex.args()                                    // → Vec<Ex> (children)

// ── Assumptions ────────────────────────────────────────────────
ex.assume(Assumption::Positive)              // fluent chaining

// ── Serialization ──────────────────────────────────────────────
ex.to_tree()                                 // → ExprTree (serde)
ex.to_json()                                 // → String (JSON)
ex.to_json_pretty()                          // → String (formatted)

// ── Collection ─────────────────────────────────────────────────
Ex::sum_of(&ctx, iter)                       // sum expressions
Ex::product_of(&ctx, iter)                   // multiply expressions

// ── Class Methods ──────────────────────────────────────────────
Ex::zero()                                   // additive identity
Ex::one()                                    // multiplicative identity

// ── Utilities ──────────────────────────────────────────────────
ex.apply_until_stable(max, f)                // generic fixpoint
ex.replace(closure)                          // user transformation walk

// ── Convenience (return fallback on failure) ───────────────────
ex.limit_or_self(&x, &a)                    // limit or unchanged
ex.solve_or_empty(&x)                       // solve or []
ex.series_or_self(&x, &a, n)               // series or unchanged
ex.maclaurin_or_self(&x, n)                 // maclaurin or unchanged
```

### Macros

```rust
expr!(x^2 + 2*x + 1)                        // build expression (supports pi, E, I, 1/2)
rule!(arena, "name", LHS => RHS)             // define rewrite rule
syms!(ctx; x, y, z)                          // declare symbols (with context)
sym!(ctx; t, Positive, Real)                 // symbol with assumptions
vars!(x, y, z)                               // declare symbols (global context)
matrix![[a, b], [c, d]]                      // build Matrix
eq!(lhs = rhs)                               // build Equation
```

### Matrix

```rust
Matrix::new(rows)                            // from nested vecs
Matrix::identity(n)                          // n×n identity
Matrix::zeros(n, m)                          // zero matrix
m.transpose()                                // transpose
m.det()                                      // determinant
m.trace()                                    // trace
m.matmul(&other)                             // matrix multiply
m.diff(&var)                                 // element-wise differentiation
m.subs(&old, &new)                           // element-wise substitution
jacobian(&[f1, f2], &[x, y])                // Jacobian matrix
```

### Equation

```rust
Equation::new(lhs, rhs)                     // create equation
eq.solve(&var)                               // solve for variable → Result
eq.solve_or_empty(&var)                      // solve or []
eq.subs(&old, &new)                          // substitute in both sides
eq.simplify()                                // simplify both sides
```

### Free-Standing Functions

```rust
symplex::var("x")                            // global context symbol
symplex::symbol("x")                         // alias for var
symplex::int(5)                              // global context integer
symplex::rational(1, 2)                      // global context rational
symplex::default_context()                   // access global context
symplex::pi()                                // global context π
symplex::e()                                 // global context e
symplex::i_unit()                            // global context imaginary unit
symplex::infinity()                          // global context ∞
symplex::parse::parse(&ctx, "x^2 + 1")      // runtime parser
```

## Serialization

Expressions can be serialized to JSON for interchange:

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.symbol("x");
let expr = &x.powi(2) + 1;

// Serialize
let json = expr.to_json();

// Deserialize
let back = ctx.from_json(&json).unwrap();
assert_eq!(format!("{expr}"), format!("{back}"));
```

For LaTeX, Markdown, or Typst rendering, a separate `symplex-format` crate is planned.

## Simplification Rules (23)

| # | Rule | Identity |
|---|------|----------|
| 1 | `sin(w)^2 + cos(w)^2 → 1` | Pythagorean |
| 2 | `exp(ln(w)) → w` | Inverse pair |
| 3 | `ln(exp(w)) → w` | Inverse pair |
| 4 | `abs(abs(w)) → abs(w)` | Idempotent |
| 5 | `(w^2)^(1/2) → abs(w)` | Square root of square |
| 6 | `asin(sin(w)) → w` | Inverse trig |
| 7 | `acos(cos(w)) → w` | Inverse trig |
| 8 | `atan(tan(w)) → w` | Inverse trig |
| 9 | `cosh(w)^2 - sinh(w)^2 → 1` | Hyperbolic Pythagorean |
| 10 | `(a^m)^n → a^(m*n)` | Power of power |
| 11 | `asinh(sinh(w)) → w` | Inverse hyperbolic |
| 12 | `acosh(cosh(w)) → w` | Inverse hyperbolic |
| 13 | `atanh(tanh(w)) → w` | Inverse hyperbolic |
| 14 | `sin(asin(w)) → w` | Forward-inverse trig |
| 15 | `cos(acos(w)) → w` | Forward-inverse trig |
| 16 | `tan(atan(w)) → w` | Forward-inverse trig |
| 17 | `sinh(asinh(w)) → w` | Forward-inverse hyperbolic |
| 18 | `cosh(acosh(w)) → w` | Forward-inverse hyperbolic |
| 19 | `tanh(atanh(w)) → w` | Forward-inverse hyperbolic |
| 20 | `sin(w)/cos(w) → tan(w)` | Trig ratio |
| 21 | `sinh(w)/cosh(w) → tanh(w)` | Hyperbolic ratio |
| 22 | `exp(a)*exp(b) → exp(a+b)` | Exp combining |
| 23 | `abs(w) → w` (when w positive) | Abs-positive |

All rules support sub-expression matching in Add and Mul (e.g., `3 + sin²(x) + cos²(x) → 4`).

## Dependencies

All dependencies are MIT or Apache-2.0 licensed. No C bindings. No LGPL.

| Crate | Purpose |
|-------|---------|
| `num-bigint` | Arbitrary-precision integers |
| `num-rational` | Exact rational numbers |
| `num-integer` | GCD, LCM |
| `num-traits` | Numeric trait vocabulary |
| `smallvec` | Inline small vectors |
| `rustc-hash` | Fast hash maps |
| `bitflags` | Assumption property flags |
| `parking_lot` | Fast locks |
| `thiserror` | Error types |
| `astro-float` | Arbitrary-precision floats |
| `serde` / `serde_json` | Serialization |
| `tracing` | Zero-cost diagnostic logging |
| `symplex-macros` | Proc macros (`expr!`, `rule!`) |

## Dependencies Policy

Symplex avoids feature flags unless absolutely necessary. All core capabilities are always included.

## Requirements

- Rust 1.93+ (Edition 2024)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.