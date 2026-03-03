# Symplex

Symbolic mathematics library for Rust.

## Features

- **Expression building** — operator overloading (`+`, `-`, `*`, `/`, unary `-`), method chaining (`.pow()`, `.sin()`, `.diff()`), automatic canonicalization (flatten, sort, combine like terms)
- **Differentiation** — all elementary functions, chain rule, product rule, n-ary generalization, higher-order derivatives, partial derivatives
- **Algebraic expansion** — distribute products over sums, expand integer powers of sums
- **Exact evaluation** — known special values of trig/exp/ln at multiples of π, 0, 1, e
- **Arbitrary-precision numerical evaluation** — via `astro-float`, any number of decimal digits
- **Assumption system** — 23 mathematical properties (positive, real, integer, etc.) with forward-chaining inference
- **Structural substitution** — exact node replacement, simultaneous multi-substitution
- **Pattern matching** — wild symbols, named rewrite rules, simplification with trace
- **Polynomial algebra** — dense univariate over ℚ, arithmetic, Euclidean GCD
- **Fraction cancellation** — GCD-based common factor elimination for rational expressions
- **Equation solving** — linear, quadratic, and higher-degree polynomial equations via rational root theorem
- **Thread safety** — `Ex` is `Send + Sync`; the arena uses `parking_lot::RwLock`

## Quick Start

```rust
use symplex::prelude::*;
use symplex::syms;

let ctx = Context::new();
syms!(ctx; x, y);

// Arithmetic with operator overloading
let expr = &x * &x + &x * 2 + 1;
println!("{expr}");                          // 1 + x**2 + 2*x

// Expand powers
let cubed = (&x + 1).powi(3);
println!("{}", cubed.expand());              // 1 + x**3 + 3*x + 3*x**2

// Differentiate
let deriv = x.powi(3).diff(&x);
println!("{deriv}");                         // 3*x**2

// Evaluate derivative at a point
let at_2 = deriv.subs(&x, &ctx.int(2));
println!("{at_2}");                          // 12

// Evaluate special values
let cos_pi = ctx.pi().cos().eval();
println!("{cos_pi}");                        // -1

// Arbitrary-precision numerical evaluation
let pi_50 = ctx.pi().evalf(50).unwrap();
println!("{pi_50}");                         // 3.1415926535897932384626433832795...

// Simplify trig identities
let trig = &x.sin().powi(2) + &x.cos().powi(2);
println!("{}", trig.simplify());             // 1

// Cancel common factors
let frac = (&x.powi(2) - 1) / (&x - 1);
println!("{}", frac.cancel(&x));             // 1 + x

// Solve equations
let roots = (&x.powi(2) - &x * 5 + 6).solve(&x);
for r in &roots {
    println!("x = {r}");                     // x = 2, x = 3
}

// Assumptions
let t = ctx.symbol_with("t", &[Assumption::Positive, Assumption::Real]);
assert_eq!((&t + 1).is_positive(), Some(true));
```

## API Reference

### Context

```rust
let ctx = Context::new();
let ctx = Context::with_config(EvalConfig { max_pow_exponent: 500, ..Default::default() });

ctx.symbol("x")                              // symbolic variable
ctx.symbol_with("t", &[Assumption::Positive]) // variable with assumptions
ctx.int(5)                                   // integer
ctx.rational(1, 3)                           // exact fraction 1/3
ctx.pi()                                     // π
ctx.e()                                      // Euler's number
ctx.query(&expr, Props::POSITIVE)            // query assumption → Option<bool>
ctx.display(&expr)                           // format as String
```

### Expression Methods

```rust
// Arithmetic: +, -, *, / for Ex, &Ex, i64 (all combinations)

// Functions
ex.pow(&exp)    ex.powi(3)    ex.sin()     ex.cos()
ex.tan()        ex.exp_fn()   ex.ln()      ex.sqrt()    ex.abs()

// Calculus
ex.diff(&x)                                  // symbolic derivative

// Transformation
ex.subs(&old, &new)                          // structural substitution
ex.subs_map(&[(&x, &a), (&y, &b)])          // simultaneous substitution
ex.expand()                                  // distribute products, expand powers
ex.eval()                                    // evaluate known special values
ex.simplify()                                // apply rewrite rules
ex.simplify_trace()                          // simplify with step-by-step trace
ex.cancel(&var)                              // cancel common polynomial factors
ex.solve(&var)                               // solve expr = 0 for var

// Numerical
ex.evalf(50)                                 // → Result<String, SymplexError>

// Queries
ex.is_zero()                                 // → Option<bool>
ex.is_positive()                             // → Option<bool>
ex.query(Props::INTEGER)                     // → Option<bool>
ex.equals(&other)                            // → Option<bool>
ex.is_zero_structural()                      // → bool (O(1))
```

### Macros

```rust
syms!(ctx; x, y, z);                        // declare multiple symbols
sym!(ctx; t, Positive, Real);               // declare with assumptions
```

## Architecture

Expressions are stored in an arena-interned DAG with hash-consing. Every expression is an `ExprId` (4-byte index). Structurally identical expressions share the same `ExprId`, making equality comparison O(1).

The user-facing `Ex` type holds an `Arc<RwLock<ContextInner>>` and an `ExprId`. It is 16 bytes, `Clone`, `Send`, and `Sync`.

All tree traversals use explicit stacks (no recursion), so stack overflow cannot occur regardless of expression depth.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed architecture documentation.

## Design Decisions

**Construction is cheap; evaluation is explicit.** Building `x + y` canonicalizes (flatten, sort, combine like terms) but does not expand, evaluate functions, or apply identities. Call `.eval()`, `.expand()`, or `.simplify()` when you want those transformations.

**No implicit float conversion.** There is no `Float` node type and no `From<f64>` for `Ex`. All symbolic computation uses exact `Ratio<BigInt>`. Floating-point results come only from `.evalf()`.

**Structural substitution by default.** `.subs()` replaces exact node matches only. `(1/x).subs(x², 1)` returns `1/x` unchanged because `x²` does not appear as a node in `x⁻¹`. This prevents silent mathematical errors.

**Number × Add distributes.** `2*(x+1)` canonicalizes to `2*x + 2`. This is required for `a - a = 0` to hold structurally. Symbolic products like `y*(x+1)` do not distribute — that is `.expand()`.

## Dependencies

All dependencies are MIT or Apache-2.0 licensed. There are no C bindings or LGPL dependencies.

| Crate | Purpose |
|-------|---------|
| `num-bigint` | Arbitrary-precision integers |
| `num-rational` | Exact rational numbers |
| `num-integer` | GCD, LCM |
| `num-traits` | Numeric trait vocabulary |
| `smallvec` | Inline small vectors for expression children |
| `rustc-hash` | Fast hash maps |
| `bitflags` | Assumption property flags |
| `parking_lot` | Fast locks |
| `thiserror` | Error types |
| `astro-float` | Arbitrary-precision floats (optional, feature `evalf`) |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `evalf` | yes | Numerical evaluation via `astro-float`. Disable for smaller binaries. |

## Requirements

- Rust 1.93+ (Edition 2024)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.