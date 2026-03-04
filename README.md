# Symplex

Symbolic mathematics library for Rust.

## Features

- **Type-safe expression system** — `Expr<Numeric>` (aliased `Ex`) and `Expr<Boolean>` (aliased `BoolEx`) prevent mixing boolean and numeric expressions at compile time
- **Expression building** — operator overloading (`+`, `-`, `*`, `/`, unary `-`), method chaining (`.pow()`, `.sin()`, `.diff()`), automatic canonicalization (flatten, sort, combine like terms, `Pow(Pow(a,b),c)→Pow(a,b*c)` for integer exponents — matching SymPy)
- **Proc macros** — `expr!(x^2 + 2*x + 1)` for natural math syntax with constants (`pi`, `E`, `I`) and rationals (`1/2`); `rule!(arena, "name", LHS => RHS)` for rewrite rules; `matrix!` and `eq!` for matrices and equations
- **Boolean expressions** — relational comparisons (`>`, `<`, `>=`, `<=`, `==`, `!=`), logical connectives (`and`, `or`, `not`), `True`/`False` atoms
- **Piecewise functions** — `Piecewise(value if condition, ...)` with differentiation and condition evaluation
- **18 math functions** — sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, ln, abs, sqrt, cbrt, nthroot
- **Differentiation** — all elementary functions, chain rule, product rule, n-ary generalization, higher-order derivatives, partial derivatives
- **Integration** — power rule, trig, exp, tan, ln, linearity, constant factor, integration by parts (LIATE-ordered with recursion depth limit), u-substitution (`sin(ax+b)`, `cos(ax+b)`, `exp(ax+b)`), inverse trig/hyperbolic standard forms, partial fraction decomposition pipeline, definite integrals, inverse trig antiderivatives (asin, acos, atan), general linear substitution (ax+b)^n, expand-then-integrate fallback
- **Taylor series** — expansion around any point with configurable order and pole detection
- **Limits** — direct substitution, L'Hôpital's rule (0/0 and ∞/∞), series fallback, Gruntz algorithm for limits at infinity (exponential, logarithmic, polynomial growth rates)
- **Equation solving** — polynomial (linear, quadratic, higher-degree via rational root theorem), complex roots, linear systems (Gaussian elimination), numerical root finding (Newton's method), transcendental equations via inversion peeling (exp, ln, sin, cos, tan, sqrt) with change-of-variable, Mul-factor solving
- **Simplification** — 24 rewrite rules with condition guards (Pythagorean, inverse pairs, exp combining, exp-log denesting, trig ratios, abs-positive) with sub-expression matching in Add and Mul, fixpoint iteration via `full_simplify()`, multi-strategy `smart_simplify()`
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
- **Arbitrary-precision parser** — `symplex::parse::parse` lexes integers and decimals of any size into `BigInt`/`Ratio` tokens, with recursion depth protection — `0.1 + 0.2 = 3/10` exactly
- **Tracing** — zero-cost `tracing` instrumentation throughout all core operations (simplification rules, integration strategies, limit algorithm steps, canonicalization). Enable with `RUST_LOG=symplex=debug`.
- **Gruntz algorithm** — the gold standard for computing symbolic limits at infinity, handling `exp(-x)→0`, `ln(x)/x→0`, `x·exp(-x)→0`, and all elementary exp-log functions
- **Thread safety** — `Ex` is `Send + Sync`; the arena uses `parking_lot::RwLock`

## Quick Start

```rust
use symplex::prelude::*;

// ── Declare variables (one line) ───────────────────────────────
vars!(x, y);

// ── Natural math syntax with expr! ─────────────────────────────
let f = expr!(x^2 + 2*x + 1);
println!("{f}");                             // x^2 + 2*x + 1

// ── Calculus ───────────────────────────────────────────────────
let df = f.diff(&x);                        // 2 + 2*x
let anti = expr!(x^2).integrate(&x);        // 1/3*x^3
let series = x.sin().maclaurin(&x, 4).unwrap();
println!("{}", series.expand());             // x - 1/6*x^3

// ── Exact fractions (no floating-point) ────────────────────────
let half_x = expr!(1/2 * x^2);              // 1/2*x^2

// ── Solve equations ────────────────────────────────────────────
let equation = eq!(x^2 - 5*x + 6 = 0);
let roots = equation.solve(&x).unwrap();
for r in &roots { println!("x = {r}"); }    // x = 2, x = 3

// ── Simplify ───────────────────────────────────────────────────
let trig = expr!(sin(x)^2 + cos(x)^2);
println!("{}", trig.simplify());             // 1

// ── Matrices ───────────────────────────────────────────────────
let m = matrix![[x, 1], [0, x^2]];
println!("det = {}", m.det());               // x^3

// ── Complex numbers ────────────────────────────────────────────
println!("{}", expr!(I^2));                  // -1
println!("{}", expr!(exp(I * pi) + 1).eval()); // 0  (Euler's identity)
println!("{}", symplex::int(-4).sqrt());     // 2*I

// ── Type-safe booleans (compile-time guarantees) ───────────────
let cond: BoolEx = expr!(x > 0 && x < 10);  // boolean expression
let pw = Ex::piecewise(&[                    // piecewise function
    (&x, &expr!(x > 0)),
    (&(-&x), &expr!(x <= 0)),
]);
// cond + 1;  // ← COMPILE ERROR: can't add boolean to number
// cond.sin(); // ← COMPILE ERROR: sin() only on numeric expressions

// ── Compile expressions to fast closures ───────────────────────
let f = expr!(x^2 + sin(x));
let fast_f = f.lambdify(&["x"]).unwrap();
println!("{:.6}", fast_f(&[1.0]));           // 1.841471

// ── ODE solving ────────────────────────────────────────────────
let ode = expr!(diff(y, x) + 2*y);          // y' + 2y = 0
if let Some((sol, _)) = ode.dsolve(&y, &x) {
    println!("y = {sol}");                   // C1*exp(-2*x)
}

// ── Arbitrary-precision evaluation ─────────────────────────────
println!("{}", symplex::pi().evalf(50).unwrap());
// 3.1415926535897932384626433832795028841971693993751
```

### Ergonomic tips

| Pattern | Recommended | Avoid |
|---------|-------------|-------|
| Build expressions | `expr!(x^2 + 1)` | `&x.powi(2) + 1` |
| Create symbols | `vars!(x, y, z);` | `let x = ctx.symbol("x");` |
| Fractions | `expr!(1/2)` | `ctx.rational(1, 2)` |
| Equations | `eq!(x^2 = 4)` | `Equation::new(x.powi(2), int(4))` |
| Matrices | `matrix![[a, b], [c, d]]` | `Matrix::new(vec![...])` |
| Constants | `expr!(pi)`, `expr!(I)` | `ctx.pi()`, `ctx.i_unit()` |
| Solve (no panic) | `eq.solve_or_empty(&x)` | `eq.solve(&x).unwrap()` |
| Sum collection | `terms.into_iter().sum()` | manual loop with `&` |
| Quick variables | `symplex::var("x")` | `Context::new()` + `ctx.symbol("x")` |

**Note:** The `expr!` macro uses the global default context. If you use `Context::new()` for custom configuration, build expressions with method calls instead of `expr!`, or use `symplex::var()` / `symplex::int()` for all symbols and constants.

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

### Boolean Expressions (`BoolEx`)

```rust
// ── Construction (returns BoolEx) ──────────────────────────
ex.gt(&other)   ex.ge(&other)   ex.lt(&other)    ex.le(&other)
ex.eq_expr(&other)               ex.ne_expr(&other)

// ── Logic (on BoolEx) ──────────────────────────────────────
bex.and(&other)                  // logical AND
bex.or(&other)                   // logical OR
bex.not()                        // logical NOT

// ── Sort-preserving (on BoolEx) ────────────────────────────
bex.eval()     bex.simplify()   bex.subs(&old, &new)
bex.free_symbols()               bex.contains(&sub)

// ── Escape hatches ─────────────────────────────────────────
bex.into_ex()                    // convert to Ex (loses type safety)
bex.as_ex()                      // borrow as Ex
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

## Simplification Rules (24)

| # | Rule | Identity |
|---|------|----------|
| 1 | `sin(w)^2 + cos(w)^2 → 1` | Pythagorean |
| 2 | `exp(ln(w)) → w` | Inverse pair |
| 3 | `ln(exp(w)) → w` | Inverse pair |
| 4 | `abs(abs(w)) → abs(w)` | Idempotent |
| 5 | `(w^2)^(1/2) → abs(w)` | Square root of square |
| 6 | `cosh(w)^2 - sinh(w)^2 → 1` | Hyperbolic Pythagorean |
| 7 | `(a^m)^n → a^(m*n)` | Power of power (when m or n is integer) |
| 8 | `asinh(sinh(w)) → w` | Inverse hyperbolic |
| 9 | `acosh(cosh(w)) → abs(w)` | Inverse hyperbolic |
| 10 | `atanh(tanh(w)) → w` | Inverse hyperbolic |
| 11 | `sin(asin(w)) → w` | Forward-inverse trig |
| 12 | `cos(acos(w)) → w` | Forward-inverse trig |
| 13 | `tan(atan(w)) → w` | Forward-inverse trig |
| 14 | `sinh(asinh(w)) → w` | Forward-inverse hyperbolic |
| 15 | `cosh(acosh(w)) → w` | Forward-inverse hyperbolic |
| 16 | `tanh(atanh(w)) → w` | Forward-inverse hyperbolic |
| 17 | `sin(w)/cos(w) → tan(w)` | Trig ratio |
| 18 | `cos(w)/sin(w) → 1/tan(w)` | Trig ratio (complement) |
| 19 | `sinh(w)/cosh(w) → tanh(w)` | Hyperbolic ratio |
| 20 | `exp(a)*exp(b) → exp(a+b)` | Exp combining |
| 21 | `exp(a*ln(b)) → b^a` | Exp-log denesting |
| 22 | `abs(w) → w` (when w positive) | Abs-positive |

All rules support sub-expression matching in Add and Mul. Rules with mathematical preconditions use condition guards (e.g., `pow_pow` requires at least one integer exponent; `abs_positive` requires the argument to be known positive).

## Feature Comparison with SymPy

symplex covers the core algebra–calculus pipeline with a Rust-native architecture.
SymPy has 20+ years of development and broader coverage. This matrix tracks both
what symplex has and what's missing.

### What symplex does that SymPy doesn't

| Feature | Description |
|---------|-------------|
| Arena hash-consing | O(1) equality, structural sharing, cache-friendly |
| Compile-time sort safety | `Expr<Numeric>` vs `Expr<Boolean>` — mixing is a compile error |
| Thread safety | `Ex` is `Send + Sync` — parallel computation safe |
| No recursion | All tree walks use explicit stacks — no stack overflow |
| Proc macro DSL | `expr!()`, `rule!()`, `matrix!()`, `eq!()` |
| Canonical invariant checker | Structural correctness verified in debug builds |
| Compiled lambdify | Bytecode VM, not interpreted |
| Arbitrary-precision parser | `0.1 + 0.2 = 3/10` exactly — no floating-point |
| Type-safe ExprView | `replace()` closure gets non-locking view — deadlock impossible at compile time |
| Gruntz algorithm in Rust | First Rust implementation of the Gruntz limit algorithm |

### Core features

| Feature | symplex | SymPy | Gap |
|---------|:---:|:---:|-----|
| Expression tree | ✅ | ✅ | symplex: arena; SymPy: Python objects |
| Assumptions (23 properties) | ✅ | ✅ | Comparable |
| Simplification (24 rules with condition guards) | ✅ | ✅ (hundreds) | SymPy has more rules |
| Arbitrary-precision eval | ✅ | ✅ | Both via external lib |
| Complex numbers | ✅ | ✅ | Both complete for Tier 1-2 |
| Boolean expressions | ✅ | ✅ | symplex: typed; SymPy: runtime |
| Piecewise | ✅ | ✅ | |
| JSON serialization | ✅ | ❌ | |

### Calculus

| Feature | symplex | SymPy | Gap |
|---------|:---:|:---:|-----|
| Differentiation (all elementary) | ✅ | ✅ | |
| Integration (power, trig, exp, ln) | ✅ | ✅ | |
| Integration (by-parts LIATE-ordered, u-sub) | ✅ | ✅ | |
| Integration (trig powers) | ✅ | ✅ | |
| Integration (partial fractions) | ✅ | ✅ | |
| Integration (completing square) | ✅ | ✅ | |
| Taylor/Maclaurin series | ✅ | ✅ | symplex: known-coefficient fast paths |
| Limits (L'Hôpital, series) | ✅ | ✅ | |
| Limits at infinity | ✅ Gruntz | ✅ Gruntz | symplex: Gruntz algorithm; SymPy: also Gruntz |
| Definite integrals | ✅ basic | ✅ full | SymPy handles improper integrals |
| Risch algorithm | ❌ | ✅ | Not planned |
| Integral transforms (Laplace, Fourier) | ❌ | ✅ | Future |
| Trig substitution | ❌ | ✅ | Planned |

### Algebra & Solving

| Feature | symplex | SymPy | Gap |
|---------|:---:|:---:|-----|
| Expand / Factor / Collect | ✅ | ✅ | |
| Together / Cancel / Apart | ✅ | ✅ | symplex: polynomial LCM for common denominator |
| Trig/Log expand & combine | ✅ | ✅ | |
| Polynomial solve (linear, quadratic) | ✅ | ✅ | |
| Complex roots | ✅ | ✅ | |
| Higher-degree (rational roots) | ✅ | ✅ | |
| Cubic/quartic formulas | ❌ | ✅ | Planned |
| Transcendental solving | ✅ partial | ✅ full | |
| Change-of-variable solve | ✅ | ✅ | |
| Linear systems | ✅ | ✅ | |
| Inequality solving | ❌ | ✅ | Planned |
| Multivariate polynomials | ❌ | ✅ | Planned |
| Full factoring (Hensel) | ❌ | ✅ | Planned |
| Gröbner bases | ❌ | ✅ | Future |
| Sets / Intervals | ❌ | ✅ | Planned |

### Matrices & Linear Algebra

| Feature | symplex | SymPy | Gap |
|---------|:---:|:---:|-----|
| Dense symbolic matrix | ✅ | ✅ | |
| Determinant / Inverse / Trace | ✅ | ✅ | |
| Matrix multiply | ✅ | ✅ | |
| Jacobian | ✅ | ✅ | |
| Eigenvalues / Eigenvectors | ❌ | ✅ | Planned |
| Matrix decompositions (LU, QR) | ❌ | ✅ | Future |
| Sparse matrices | ❌ | ✅ | Future |

### Special Features

| Feature | symplex | SymPy | Gap |
|---------|:---:|:---:|-----|
| Factorial / Binomial | ✅ | ✅ | symplex: arbitrary precision, no limit |
| Sign function | ✅ | ✅ | |
| ODE solver (basic) | ✅ | ✅ (full) | SymPy has dozens of methods |
| CSE | ✅ | ✅ | |
| lambdify | ✅ | ✅ | symplex: bytecode VM |
| Symbolic sums (Σ) | ❌ | ✅ | Planned |
| Special functions (gamma, erf) | ❌ | ✅ | Future |
| Vector calculus (grad, div, curl) | ❌ | ✅ | Planned |
| Number theory | ❌ | ✅ | Not planned |
| Statistics / Probability | ❌ | ✅ | Not planned |
| LaTeX output | ❌ (serde) | ✅ | Separate crate planned |
| Code generation (C, Python) | ❌ | ✅ | Separate crate planned |

### Macros (Rust-specific, no SymPy equivalent)

| Macro | What it does |
|-------|-------------|
| `expr!(x^2 + sin(x))` | Natural math syntax → Ex |
| `expr!(x > 0 && y < 1)` | Comparisons → BoolEx |
| `rule!(arena, "name", LHS => RHS)` | Pattern rewrite rules |
| `matrix![[a, b], [c, d]]` | Matrix construction |
| `eq!(x^2 + x = 6)` | Equation construction |
| `expr!(1/2)` | Exact rationals |
| `expr!(pi)`, `expr!(I)` | Mathematical constants |

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
| `tracing-subscriber` | *(dev)* Subscriber for log output with env-filter |
| `tracing-test` | *(dev)* Test harness for tracing assertions |

## Dependencies Policy

Symplex avoids feature flags unless absolutely necessary. All core capabilities are always included.

## Requirements

- Rust 1.93+ (Edition 2024)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.