# Symplex

Symbolic mathematics library for Rust. Provides differentiation, integration, equation solving, matrix algebra, and code generation with thread-safe, arena-based expression management. Expressions use hash-consing for O(1) structural equality and automatic canonicalization. The type system separates numeric (`Ex`) and boolean (`BoolEx`) expressions at compile time.

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

## Installation

```sh
cargo add symplex
```

## Features

- **Expression system** — type-safe `Ex` (numeric) and `BoolEx` (boolean) with operator overloading, method chaining, and automatic canonicalization. Proc macros for natural math syntax: `expr!()`, `rule!()`, `matrix![]`, `eq!()`.
- **Calculus** — symbolic differentiation (chain/product rule, higher-order, partial), integration (by-parts, u-substitution, partial fractions, trig powers, cyclic IBP), Taylor series, and limits (L'Hôpital, Gruntz algorithm).
- **Algebra & solving** — expand, factor, collect, together, cancel, apart, trig/log expansion. Polynomial solving through quartic (Cardano/Ferrari), transcendental equation solving, linear systems, numerical root finding.
- **Simplification** — 24 rewrite rules with condition guards, fixpoint iteration (`full_simplify`), multi-strategy `smart_simplify`, trigsimp, powsimp, combsimp, nsimplify, and rewrite protocol (trig↔exp via Euler's formula).
- **Matrices & linear algebra** — determinant, inverse, eigenvalues, LU/QR decomposition, RREF, nullspace, rank, Jacobian, vector calculus (gradient, divergence, curl, laplacian).
- **Transforms & special functions** — forward/inverse Laplace, Fourier series, ODE solver (separable, linear, constant-coefficient). Gamma, Erf, Beta, LambertW, Heaviside, DiracDelta, and 11 combinatorial functions.
- **Numerical & code generation** — `lambdify` compiles expressions to `Fn(f64) -> f64` closures, CSE for code generation, `to_rust_fn()`, arbitrary-precision evaluation via `astro-float`, runtime parser.
- **Architecture** — arena hash-consing with O(1) equality, thread-safe (`Send + Sync`), no recursion (explicit stacks), assumption system (23 properties), pattern matching, serde serialization, zero-cost `tracing` instrumentation.

<details>
<summary>Full feature list</summary>

- **Type-safe expression system** — `Expr<Numeric>` (aliased `Ex`) and `Expr<Boolean>` (aliased `BoolEx`) prevent mixing boolean and numeric expressions at compile time
- **Expression building** — operator overloading (`+`, `-`, `*`, `/`, unary `-`), method chaining (`.pow()`, `.sin()`, `.diff()`), automatic canonicalization (flatten, sort, combine like terms, `Pow(Pow(a,b),c)→Pow(a,b*c)` for integer exponents)
- **Proc macros** — `expr!(x^2 + 2*x + 1)` for natural math syntax with constants (`pi`, `E`, `I`) and rationals (`1/2`); `rule!(arena, "name", LHS => RHS)` for rewrite rules; `matrix!` and `eq!` for matrices and equations
- **Boolean expressions** — relational comparisons (`>`, `<`, `>=`, `<=`, `==`, `!=`), logical connectives (`and`, `or`, `not`), `True`/`False` atoms
- **Piecewise functions** — `Piecewise(value if condition, ...)` with differentiation and condition evaluation
- **31+ math functions** — sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, ln, abs, sqrt, cbrt, nthroot, sec, csc, cot, acot, asec, acsc, coth, sech, csch, acoth, asech, acsch, sinc, atan2
- **Differentiation** — all elementary functions, chain rule, product rule, n-ary generalization, higher-order derivatives, partial derivatives
- **Integration** — power rule, trig, exp, tan, ln, linearity, constant factor, integration by parts (LIATE-ordered with recursion depth limit), u-substitution, inverse trig/hyperbolic standard forms, partial fraction decomposition, definite integrals, inverse trig antiderivatives, general linear substitution, expand-then-integrate fallback, negative trig powers, cyclic IBP (exp·sin, exp·cos)
- **Taylor series** — expansion around any point with configurable order and pole detection
- **Limits** — direct substitution, L'Hôpital's rule (0/0 and ∞/∞), series fallback, Gruntz algorithm for limits at infinity
- **Equation solving** — polynomial (linear, quadratic, cubic via Cardano, quartic via Ferrari, higher-degree via rational root theorem), complex roots, linear systems (Gaussian elimination), numerical root finding (Newton's method), transcendental equations via inversion peeling
- **Simplification** — 24 rewrite rules with condition guards, fixpoint iteration via `full_simplify()`, multi-strategy `smart_simplify()`, trigsimp (6-strategy choice-set), powsimp, combsimp, nsimplify, rewrite protocol (trig↔exp via Euler's formula)
- **Algebraic manipulation** — expand, factor, collect, together, cancel, partial fractions (`apart`), trig expansion (`expand_trig`), log expansion (`expand_log`), logcombine
- **Laplace transforms** — forward (table-based + structural rules) and inverse (partial fractions + table)
- **Fourier series** — computation via integration
- **Special functions** — Gamma, LogGamma, Digamma, Erf, Erfc, Beta with eval/diff rules; Heaviside, DiracDelta, LambertW
- **Combinatorial functions** — 11 functions via Apply nodes: fibonacci, lucas, bernoulli, harmonic, catalan, bell, euler_number, subfactorial, factorial2, rising_factorial, falling_factorial
- **Vector calculus** — gradient, divergence, curl, laplacian, is_conservative, is_solenoidal
- **Logic connectives** — xor, implies, equivalent, nand, nor, ite on BoolEx
- **Floor/Ceiling/Min/Max** — ExprNode variants with exact rational eval and canonicalization
- **Symbolic sums and products** — ExprNode variants with finite evaluation
- **Rust code generation** — `to_rust_fn()` with CSE, piecewise, all elementary functions
- **Exact evaluation** — 86+ special values for trig/exp/ln including irrational values, perfect nth root evaluation, odd/even function detection, integer sqrt simplification, trig-hyperbolic bridge
- **Complex numbers** — i²=-1 canonicalization, (-1)^(1/2)→i, complex quadratic roots, Euler's formula
- **Complex decomposition** — `re()` / `im()` split expressions into real and imaginary parts
- **Symbolic matrices** — construct, multiply, transpose, determinant, inverse, trace, Jacobian, eigenvalues, char_poly, cofactor, adjugate, LU/QR decomposition, RREF, rank, nullspace, columnspace, norm, cross, dot, hstack, vstack, is_symmetric
- **ODE solver** — separable, first-order linear, second-order constant-coefficient
- **lambdify** — compile expressions to `Fn(f64) -> f64` closures for numerical evaluation
- **Common subexpression elimination (CSE)** — extract shared subexpressions for code generation
- **Factorial and binomial coefficients** — `n!` and `C(n,k)` with arbitrary-precision evaluation
- **Denominator rationalization** — clear square roots from denominators
- **Arbitrary-precision numerical evaluation** — via `astro-float`, any number of decimal digits
- **Assumption system** — 23 mathematical properties (positive, real, integer, etc.) with forward-chaining inference
- **Pattern matching** — wild symbols, named rewrite rules, simplification with trace
- **Polynomial algebra** — dense univariate over ℚ, arithmetic, Euclidean GCD/LCM, degree, coefficients
- **Serde serialization** — `ExprTree` for JSON interchange with round-trip support
- **Runtime parser** — `symplex::parse::parse(&ctx, "x^2 + 1")` for REPL and dynamic construction
- **Arbitrary-precision parser** — lexes integers and decimals of any size into `BigInt`/`Ratio` tokens, with recursion depth protection — `0.1 + 0.2 = 3/10` exactly
- **Tracing** — zero-cost `tracing` instrumentation throughout all core operations. Enable with `RUST_LOG=symplex=debug`.
- **Gruntz algorithm** — computing symbolic limits at infinity, handling `exp(-x)→0`, `ln(x)/x→0`, `x·exp(-x)→0`, and all elementary exp-log functions
- **Thread safety** — `Ex` is `Send + Sync`; the arena uses `parking_lot::RwLock`

</details>

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

### Expression Methods (172 public methods)

```rust
// ── Math functions (40+) ───────────────────────────────────────
ex.pow(&exp)    ex.powi(3)     ex.sin()      ex.cos()
ex.tan()        ex.exp()       ex.ln()       ex.sqrt()
ex.abs()        ex.asin()      ex.acos()     ex.atan()
ex.sinh()       ex.cosh()      ex.tanh()     ex.asinh()
ex.acosh()      ex.atanh()     ex.cbrt()     ex.nthroot(n)
ex.sec()        ex.csc()       ex.cot()      ex.sinc()
ex.acot()       ex.asec()      ex.acsc()     ex.atan2(&x)
ex.coth()       ex.sech()      ex.csch()
ex.acoth()      ex.asech()     ex.acsch()
ex.sign()       ex.floor()     ex.ceiling()  ex.frac()
ex.rem(&other)  ex.log(&base)
Ex::min_of(ctx, iter)          Ex::max_of(ctx, iter)
ex.min_with(&other)            ex.max_with(&other)

// ── Special functions ──────────────────────────────────────────
ex.gamma()      ex.log_gamma()  ex.digamma()
ex.erf()        ex.erfc()       ex.beta(&other)
ex.factorial()  ex.binomial(&k) ex.factorial2()
ex.subfactorial()               ex.rising_factorial(&n)
ex.falling_factorial(&n)        ex.fibonacci()
ex.lucas()      ex.bernoulli_number()  ex.harmonic()
ex.catalan_number()  ex.bell()  ex.euler_number()
ex.heaviside()  ex.dirac_delta()  ex.lambertw()

// ── Symbolic sums & products ───────────────────────────────────
Ex::symbolic_sum(body, var, lo, hi)          // unevaluated Σ
Ex::symbolic_product(body, var, lo, hi)      // unevaluated Π
ex.closed_form_sum()                         // try closed-form evaluation
ex.is_convergent(&var)                       // convergence test

// ── Calculus ───────────────────────────────────────────────────
ex.diff(&x)                                  // symbolic derivative
ex.diff_n(&x, n)                             // nth derivative
ex.formal_diff(&x)                           // formal derivative (no eval)
ex.integrate(&x)                             // indefinite integral
ex.definite_integral(&x, &lower, &upper)     // definite integral
ex.series(&x, &point, order)                 // Taylor series → Result
ex.maclaurin(&x, order)                      // Maclaurin series → Result
ex.limit(&x, &point)                         // symbolic limit → Result
ex.residue(&x, &point)                       // residue via limit → Result
ex.fourier_series(&x, n_terms)               // Fourier series
ex.laplace(&t, &s)                           // forward Laplace → Result
ex.inverse_laplace(&s, &t)                   // inverse Laplace → Result

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
ex.trig_combine()                            // sin(a)cos(b) → sum form
ex.factor_terms()                            // extract GCD of coefficients
ex.rationalize_denom()                       // clear radicals from denominators
ex.ratsimp()                                 // rational simplification
ex.separatevars(&[&x, &y])                  // separate variable dependencies
ex.solve(&var)                               // solve expr=0 → Result
ex.solve_or_empty(&var)                      // solve or []
ex.nsolve(&var, guess, max_iter, tol)        // numerical root → Result

// ── Inequality solving (returns SetEx) ─────────────────────────
ex.solve_gt(&var)                            // solve expr > 0 → SetEx
ex.solve_ge(&var)                            // solve expr ≥ 0 → SetEx
ex.solve_lt(&var)                            // solve expr < 0 → SetEx
ex.solve_le(&var)                            // solve expr ≤ 0 → SetEx
ex.solveset(&var)                            // solve → FiniteSet (SetEx)

// ── ODE solving ────────────────────────────────────────────────
ex.dsolve(&func, &var)                       // → Option<(Ex, Vec<Ex>)>
ex.check_solution(&var, &val)                // verify solution
ex.classify_ode(&func, &var)                 // → OdeType
ex.checkodesol(&sol, &func, &var)            // verify ODE solution

// ── Simplification ─────────────────────────────────────────────
ex.simplify()                                // one-pass rewrite rules
ex.simplify_trace()                          // with step-by-step trace
ex.full_simplify()                           // fixpoint: eval+expand+simplify
ex.full_simplify_trace()                     // with accumulated trace
ex.smart_simplify()                          // multi-strategy simplification
ex.trigsimp()                                // trig simplification (6 strategies)
ex.powsimp()                                 // symbolic exponent merging
ex.combsimp()                                // factorial/binomial simplification
ex.nsimplify(tolerance)                      // closed-form from floats
ex.rewrite_as_exp()                          // trig → exp (Euler's formula)
ex.rewrite_as_trig()                         // exp → trig
ex.eval()                                    // evaluate special values
ex.count_ops()                               // expression complexity

// ── Complex decomposition ──────────────────────────────────────
ex.re()                                      // real part
ex.im()                                      // imaginary part
ex.arg()                                     // complex argument
ex.conjugate()                               // complex conjugate

// ── Code generation ────────────────────────────────────────────
ex.lambdify(&["x", "y"])                     // compile to Fn(f64) → f64 closure
ex.cse()                                     // common subexpression elimination
ex.to_rust_fn("name", &["x", "y"])           // Rust source code generation

// ── Substitution ───────────────────────────────────────────────
ex.subs(&old, &new)                          // structural substitution
ex.subs_i64(&old, n)                         // substitute with integer
ex.subs_map(&[(&x, &a), (&y, &b)])          // simultaneous substitution

// ── Numerical evaluation ───────────────────────────────────────
ex.evalf(50)                                 // arbitrary precision → Result<String>
ex.evalf_f64()                               // f64 convenience → Result<f64>
ex.evalf_complex64()                         // complex (f64,f64) → Result

// ── Queries ────────────────────────────────────────────────────
ex.is_zero()        ex.is_positive()         ex.is_negative()
ex.is_real()        ex.is_integer()          ex.is_nonzero()
ex.is_finite()      ex.query(Props::...)     ex.equals(&other)
ex.is_imaginary()   ex.is_complex()          ex.is_rational()
ex.is_nonnegative() ex.is_nonpositive()
ex.is_even()        ex.is_odd()              ex.is_prime()
ex.is_composite()   ex.is_algebraic()        ex.is_transcendental()
ex.is_irrational()  ex.is_hermitian()
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

// ── Set construction (returns SetEx) ───────────────────────────
ex.closed_interval(&end)                     // [a, b]
ex.open_interval(&end)                       // (a, b)

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
bex.xor(&other)                  // exclusive OR
bex.implies(&other)              // logical implication
bex.equivalent(&other)           // logical equivalence (biconditional)
bex.nand(&other)                 // NOT AND
bex.nor(&other)                  // NOT OR
bex.ite(&then_ex, &else_ex)     // if-then-else

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

### Matrix (44 methods)

```rust
Matrix::new(rows)                            // from nested vecs
Matrix::identity(n)                          // n×n identity
Matrix::zeros(n, m)                          // zero matrix
m.transpose()                                // transpose
m.det()                                      // determinant (LU-based for large)
m.trace()                                    // trace
m.matmul(&other)                             // matrix multiply
m.inv()                                      // matrix inverse
m.eigenvals(&var)                            // eigenvalues via char poly
m.eigenvects(&var)                           // eigenvectors via null space
m.char_poly(&var)                            // characteristic polynomial
m.lu()                                       // LU decomposition
m.qr()                                       // QR decomposition
m.rref()                                     // row-reduced echelon form
m.rank()                                     // matrix rank
m.nullspace()                                // null space basis
m.columnspace()                              // column space basis
m.cofactor(i, j)                             // cofactor at (i,j)
m.adjugate()                                 // adjugate matrix
m.norm()                                     // Frobenius norm
m.cross(&other)                              // cross product (3-vectors)
m.dot(&other)                                // dot product
Matrix::hstack(&[m1, m2])                    // horizontal concatenation
Matrix::vstack(&[m1, m2])                    // vertical concatenation
m.is_symmetric()                             // symmetry test
m.diff(&var)                                 // element-wise differentiation
m.subs(&old, &new)                           // element-wise substitution
jacobian(&[f1, f2], &[x, y])                // Jacobian matrix
gradient(&f, &[x, y, z])                     // gradient vector
divergence(&field, &[x, y, z])               // divergence scalar
curl(&field, &[x, y, z])                     // curl vector
laplacian(&f, &[x, y, z])                    // Laplacian scalar
```

### Equation

```rust
Equation::new(lhs, rhs)                     // create equation
eq.solve(&var)                               // solve for variable → Result
eq.solve_or_empty(&var)                      // solve or []
eq.subs(&old, &new)                          // substitute in both sides
eq.simplify()                                // simplify both sides
eq.check(&var, &val)                         // verify a solution
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
gradient(&f, &[x, y, z])                     // gradient (free-standing)
divergence(&field, &[x, y, z])               // divergence (free-standing)
curl(&field, &[x, y, z])                     // curl (free-standing)
laplacian(&f, &[x, y, z])                    // laplacian (free-standing)
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

## Simplification Rules (24 + dedicated simplifiers)

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

All rules support sub-expression matching in Add and Mul. Rules with mathematical preconditions use condition guards (e.g., `pow_pow` requires at least one integer exponent; `abs_positive` requires the argument to be known positive). Beyond pattern rules, dedicated simplifiers include: `trigsimp()` (6-strategy choice-set), `powsimp()` (symbolic exponent merging), `combsimp()` (factorial/binomial), `nsimplify()` (closed-form detection), `rewrite_as_exp()`/`rewrite_as_trig()` (Euler's formula), and `smart_simplify()` (multi-strategy orchestrator selecting lowest `count_ops`).

<details>
<summary>Feature Comparison with SymPy</summary>

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
| Laplace transform in Rust | First Rust CAS with bidirectional Laplace (forward + inverse) |

### Core features

| Feature | symplex | SymPy | Gap |
|---------|:---:|:---:|-----|
| Expression tree | ✅ | ✅ | symplex: arena; SymPy: Python objects |
| Assumptions (23 properties) | ✅ | ✅ | Comparable |
| Simplification (24 rules + trigsimp + powsimp + combsimp + nsimplify) | ✅ | ✅ (hundreds) | SymPy has more rules |
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
| Integration (trig powers sin^n, cos^n) | ✅ | ✅ | |
| Integration (negative trig powers: sec², csc², sec⁴, …) | ✅ | ✅ | |
| Integration (cyclic IBP: exp·sin, exp·cos) | ✅ | ✅ | |
| Integration (partial fractions) | ✅ | ✅ | |
| Integration (completing square) | ✅ | ✅ | |
| Taylor/Maclaurin series | ✅ | ✅ | symplex: known-coefficient fast paths |
| Limits (L'Hôpital, series) | ✅ | ✅ | |
| Limits at infinity | ✅ Gruntz | ✅ Gruntz | symplex: Gruntz algorithm; SymPy: also Gruntz |
| Definite integrals | ✅ basic | ✅ full | SymPy handles improper integrals |
| Laplace / inverse Laplace transform | ✅ basic | ✅ full | symplex: table-based; SymPy: algorithmic |
| Fourier series | ✅ basic | ✅ full | symplex: via integration |
| Risch algorithm | ❌ | ✅ | Not planned |
| Trig substitution | ❌ | ✅ | Planned |

### Algebra & Solving

| Feature | symplex | SymPy | Gap |
|---------|:---:|:---:|-----|
| Expand / Factor / Collect | ✅ | ✅ | |
| Together / Cancel / Apart | ✅ | ✅ | symplex: polynomial LCM for common denominator |
| Trig/Log expand & combine | ✅ | ✅ | |
| Polynomial solve (linear, quadratic) | ✅ | ✅ | |
| Cubic/quartic formulas | ✅ | ✅ | Cardano + Ferrari |
| Complex roots | ✅ | ✅ | |
| Higher-degree (rational roots) | ✅ | ✅ | |
| Transcendental solving | ✅ partial | ✅ full | |
| Change-of-variable solve | ✅ | ✅ | |
| Linear systems | ✅ | ✅ | |
| Inequality solving | ❌ | ✅ | Planned (blocked on Set types) |
| Multivariate polynomials | ❌ | ✅ | Planned |
| Full factoring (Hensel) | ❌ | ✅ | Planned |
| Gröbner bases | ❌ | ✅ | Future |
| Sets / Intervals | ❌ | ✅ | Planned |

### Matrices & Linear Algebra

| Feature | symplex | SymPy | Gap |
|---------|:---:|:---:|-----|
| Dense symbolic matrix | ✅ | ✅ | |
| Determinant (LU-based for large) | ✅ | ✅ | |
| Matrix inverse | ✅ | ✅ | |
| Matrix multiply | ✅ | ✅ | |
| Jacobian | ✅ | ✅ | |
| Eigenvalues / Eigenvectors | ✅ | ✅ | |
| Matrix decompositions (LU, QR) | ✅ | ✅ | |
| RREF, nullspace, columnspace, rank | ✅ | ✅ | |
| Sparse matrices | ❌ | ✅ | Future |

### Special Features

| Feature | symplex | SymPy | Gap |
|---------|:---:|:---:|-----|
| Factorial / Binomial | ✅ | ✅ | symplex: arbitrary precision, no limit |
| Sign function | ✅ | ✅ | |
| ODE solver (basic) | ✅ | ✅ (full) | SymPy has dozens of methods |
| CSE | ✅ | ✅ | |
| lambdify | ✅ | ✅ | symplex: bytecode VM |
| Special functions (gamma, erf) | ✅ basic | ✅ | Gamma, LogGamma, Digamma, Erf, Erfc, Beta |
| Vector calculus (grad, div, curl) | ✅ | ✅ | gradient, divergence, curl, laplacian |
| Symbolic sums (Σ) | ✅ | ✅ | Finite evaluation |
| Floor / Ceiling / Min / Max | ✅ | ✅ | |
| Laplace transforms | ✅ | ✅ | symplex: table-based forward + inverse |
| Combinatorial (fibonacci, bernoulli, …) | ✅ | ✅ | 11 functions via Apply nodes |
| Logic (Xor, Implies, Equivalent) | ✅ | ✅ | + nand, nor, ite |
| combsimp | ✅ | ✅ | Factorial/binomial simplification |
| nsimplify | ✅ | ✅ | Closed-form detection from floats |
| trigsimp (choice-set) | ✅ | ✅ | 6-strategy selection |
| powsimp | ✅ | ✅ | Symbolic exponent merging |
| rewrite protocol (trig↔exp) | ✅ | ✅ | Euler's formula |
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

</details>

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