# Symplex

Symbolic mathematics library for Rust.

## Features

- **Expression building** — operator overloading (`+`, `-`, `*`, `/`, unary `-`), method chaining (`.pow()`, `.sin()`, `.diff()`), automatic canonicalization (flatten, sort, combine like terms)
- **Proc macros** — `expr!(x^2 + 2*x + 1)` for natural math syntax with auto-borrowing; `rule!(arena, "name", LHS => RHS)` for one-line rewrite rule definitions
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
- **Integration** — antiderivatives for polynomials, trig, exp; linearity, constant factor extraction
- **Taylor series** — expansion around a point with configurable order, pole detection
- **Polynomial factoring** — rational root finding with content extraction and multiplicity handling
- **Expression collection** — group terms by powers of a variable via `collect(var)`
- **Common denominators** — combine fractions via `together()`
- **Structural introspection** — `free_symbols()`, `contains()` for expression analysis
- **Thread safety** — `Ex` is `Send + Sync`; the arena uses `parking_lot::RwLock`

## Quick Start

```rust
use symplex::prelude::*;

// ── Quick setup with free-standing functions ───────────────────────
// No Context boilerplate needed for common usage.
let x = symplex::var("x");
let y = symplex::var("y");

// Or declare multiple variables at once:
// use symplex::vars;
// vars!(x, y, z);

// ── Build expressions with natural math syntax ─────────────────────
let f = expr!(x^2 + 2*x + 1);
println!("{f}");                             // 1 + x^2 + 2*x

// ── Expand, differentiate, integrate ───────────────────────────────
let cubed = expr!((x + 1)^3);
println!("{}", cubed.expand());              // 1 + x^3 + 3*x + 3*x^2

let deriv = expr!(x^3).diff(&x);
println!("{deriv}");                         // 3*x^2

let anti = expr!(x^2).integrate(&x);
println!("{anti}");                          // 1/3*x^3

// ── Evaluate ───────────────────────────────────────────────────────
let at_2 = deriv.subs_i64(&x, 2);
println!("{at_2}");                          // 12

let cos_pi = symplex::default_context().pi().cos().eval();
println!("{cos_pi}");                        // -1

// Arbitrary-precision numerical evaluation
let pi_50 = symplex::default_context().pi().evalf(50).unwrap();
println!("{pi_50}");                         // 3.14159265358979...

// ── Simplify trig identities ───────────────────────────────────────
let trig = expr!(sin(x)^2 + cos(x)^2);
println!("{}", trig.simplify());             // 1

// ── Solve equations ────────────────────────────────────────────────
let eq = expr!(x^2 - 5*x + 6);
let roots = eq.solve(&x);
for r in &roots {
    println!("x = {r}");                     // x = 2, x = 3
}

// ── Factor polynomials ─────────────────────────────────────────────
let factored = (&x.powi(2) - 1).factor(&x);
println!("{factored}");                      // (-1 + x)*(1 + x)

// ── Taylor series ──────────────────────────────────────────────────
let s = x.sin().maclaurin(&x, 4);
println!("{}", s.expand().eval());           // x - 1/6*x^3

// ── Structural introspection ───────────────────────────────────────
let expr = &x.powi(2) + &y;
println!("{:?}", expr.free_symbols().iter().map(|s| format!("{s}")).collect::<Vec<_>>());

// ── Advanced: explicit Context for custom configuration ────────────
let ctx = Context::new();
let t = ctx.symbol_with("t", &[Assumption::Positive, Assumption::Real]);
assert_eq!(t.is_positive(), Some(true));
assert_eq!(t.is_real(), Some(true));
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
ctx.with_arena_mut(|arena| { ... })          // direct arena access (for rule!)
```

### Expression Methods

```rust
// Arithmetic: +, -, *, / for Ex, &Ex, i64 (all combinations)

// Functions
ex.pow(&exp)    ex.powi(3)    ex.sin()     ex.cos()
ex.tan()        ex.exp_fn()   ex.ln()      ex.sqrt()    ex.abs()

// Calculus
ex.diff(&x)                                  // symbolic derivative
ex.integrate(&x)                             // indefinite integral

// Transformation
ex.subs(&old, &new)                          // structural substitution
ex.subs_map(&[(&x, &a), (&y, &b)])          // simultaneous substitution
ex.expand()                                  // distribute products, expand powers
ex.eval()                                    // evaluate known special values
ex.simplify()                                // apply rewrite rules (with sub-expression matching)
ex.simplify_trace()                          // simplify with step-by-step trace
ex.cancel(&var)                              // cancel common polynomial factors
ex.collect(&var)                             // group by powers of var
ex.together()                                // common denominator for fractions
ex.factor(&var)                              // factor polynomial into linear factors
ex.solve(&var)                               // solve expr = 0 for var
ex.series(&var, &point, order)               // Taylor series expansion

// Numerical
ex.evalf(50)                                 // → Result<String, SymplexError>

// Queries
ex.is_zero()                                 // → Option<bool>
ex.is_positive()                             // → Option<bool>
ex.is_negative()                             // → Option<bool>
ex.is_real()                                 // → Option<bool>
ex.is_integer()                              // → Option<bool>
ex.is_nonzero()                              // → Option<bool>
ex.is_finite()                               // → Option<bool>
ex.query(Props::INTEGER)                     // → Option<bool>
ex.equals(&other)                            // → Option<bool> (with expand fallback)
ex.is_zero_structural()                      // → bool (O(1))
ex.free_symbols()                            // → Vec<Ex>
ex.contains(&sub)                            // → bool
```

### Macros

```rust
// Declarative macros
syms!(ctx; x, y, z);                        // declare multiple symbols
sym!(ctx; t, Positive, Real);               // declare with assumptions

// Proc macros — natural math syntax
expr!(x^2 + 2*x + 1)                        // build Ex with ^ for power
expr!(sin(x)^2 + cos(x)^2)                  // function calls
expr!((x + 1)^3 * y)                        // grouping with parens

// Rewrite rule definition (inside ctx.with_arena_mut)
rule!(arena, "pythagorean", sin(w_)^2 + cos(w_)^2 => 1)
rule!(arena, "exp_ln", exp(ln(w_)) => w_)
rule!(arena, "sqrt_sq", sqrt(w_^2) => abs(w_))
```

The `expr!` macro auto-borrows identifiers (no `&` needed) and rewrites `^` to `.powi()` or `.pow()`. Integer literals stay as `i64`. Note: `expr!(1/2)` is a compile error — use `ctx.rational(1, 2)` for exact fractions.

The `rule!` macro builds `Pattern`/`Rule` structs. Identifiers ending in `_` are wilds (match anything). Known constants (`pi`, `E`, `I`, `oo`, `nan`) are recognized. Unknown bare identifiers produce a compile error with a helpful message.

## Architecture

Expressions are stored in an arena-interned DAG with hash-consing. Every expression is an `ExprId` (4-byte index). Structurally identical expressions share the same `ExprId`, making equality comparison O(1).

The user-facing `Ex` type holds an `Arc<RwLock<ContextInner>>` and an `ExprId`. It is 16 bytes, `Clone`, `Send`, and `Sync`.

All tree traversals use explicit stacks (no recursion), so stack overflow cannot occur regardless of expression depth. Verified with 10,000-deep nested expressions.

The codebase is ~17,500 lines across 25 modules with 732 tests. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed architecture, module reference, concurrency model, and next steps.

## Design Decisions

**Construction is cheap; evaluation is explicit.** Building `x + y` canonicalizes (flatten, sort, combine like terms) but does not expand, evaluate functions, or apply identities. Call `.eval()`, `.expand()`, or `.simplify()` when you want those transformations.

**No implicit float conversion.** There is no `Float` node type and no `From<f64>` for `Ex`. All symbolic computation uses exact `Ratio<BigInt>`. Floating-point results come only from `.evalf()`.

**Structural substitution by default.** `.subs()` replaces exact node matches only. `(1/x).subs(x², 1)` returns `1/x` unchanged because `x²` does not appear as a node in `x⁻¹`. This prevents silent mathematical errors.

**Number × Add distributes.** `2*(x+1)` canonicalizes to `2*x + 2`. This is required for `a - a = 0` to hold structurally. Symbolic products like `y*(x+1)` do not distribute — that is `.expand()`.

## Dependencies

All dependencies are MIT or Apache-2.0 licensed. No C bindings. No LGPL.

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
| `symplex-macros` | Proc macros (`expr!`, `rule!`); uses `syn`, `quote`, `proc-macro2` |
| `astro-float` | Arbitrary-precision floats |

## Dependencies Policy

Symplex avoids feature flags unless absolutely necessary (e.g., a dependency requires a C toolchain or adds significant platform-specific constraints). All core capabilities, including numerical evaluation via `astro-float`, are always included. This keeps the maintenance burden low, eliminates conditional compilation complexity, and ensures every user gets the full API without configuration.

## Requirements

- Rust 1.93+ (Edition 2024)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.