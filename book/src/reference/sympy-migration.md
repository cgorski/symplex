# Migrating from SymPy

This page is a reference for developers who know SymPy and want to find the equivalent operations in symplex. It is organized by task, with SymPy on the left and symplex on the right.

## Key Differences

Before the translation table, a few structural differences to be aware of:

| Concept | SymPy | symplex |
|---------|-------|---------|
| State | Global implicit state; symbols are free-standing objects | Every expression belongs to a `Context`; no global state |
| Types | Everything is an `Expr` at runtime | `Ex` (numeric), `BoolEx` (boolean), `SetEx` (set-valued) — distinct at compile time |
| Arithmetic | Python operators on SymPy objects | Rust operators on `&Ex` references (or use `expr!` macro) |
| Evaluation | `simplify()` is the catch-all | `eval()` for exact reduction, `simplify()` for rewrite rules, `full_simplify()` for multi-strategy |
| Failure | Returns unevaluated or raises exception | Returns unevaluated `Ex` (Pattern 1) or `Result` (Patterns 2–5); never raises |
| Floats | `Float` type exists alongside exact | No float type in expressions; floats only via `eval_f64()` |
| Printing | `pprint()`, `latex()`, `str()` | `println!("{expr}")`, `expr.to_latex()` |

## Setup

| SymPy | symplex |
|-------|---------|
| `from sympy import *` | `use symplex::prelude::*;` |
| `x, y, z = symbols('x y z')` | `symplex::syms!(ctx; x, y, z);` |
| `x = Symbol('x')` | `let x = ctx.symbol("x");` |
| `x = Symbol('x', positive=True)` | `let x = sym!(ctx; x, Positive);` |
| `x = Symbol('x', integer=True)` | `let x = sym!(ctx; x, Integer);` |

## Building Expressions

| SymPy | symplex |
|-------|---------|
| `x**2 + 2*x + 1` | `&x.powi(2) + &x * 2 + 1` |
| `x**2 + 2*x + 1` | `expr!(ctx, x^2 + 2*x + 1)` |
| `Rational(1, 3)` | `ctx.rational(1, 3)` |
| `Integer(42)` | `ctx.int(42)` |
| `pi` | `ctx.pi()` |
| `E` | `ctx.e()` |
| `I` | `ctx.i_unit()` |
| `oo` | `ctx.infinity()` |
| `sin(x)` | `x.sin()` |
| `cos(x)` | `x.cos()` |
| `exp(x)` | `x.exp()` |
| `log(x)` | `x.ln()` |
| `sqrt(x)` | `x.sqrt()` |
| `Abs(x)` | `x.abs_val()` |
| `x**y` | `x.pow(&y)` |
| `x**5` | `x.powi(5)` |
| `factorial(n)` | `n.factorial()` |
| `binomial(n, k)` | `n.binomial(&k)` |
| `gamma(x)` | `x.gamma()` |
| `erf(x)` | `x.erf()` |
| `Piecewise((a, cond1), (b, cond2))` | `Ex::piecewise(&[(&a, &cond1), (&b, &cond2)])` |
| `Matrix([[1, 2], [3, 4]])` | `matrix![ctx, [1, 2], [3, 4]]` |

## Calculus

| SymPy | symplex |
|-------|---------|
| `diff(f, x)` | `f.diff(&x)` |
| `diff(f, x, 3)` | `f.diff_n(&x, 3)` |
| `diff(f, x, y)` | `f.diff(&x).diff(&y)` |
| `integrate(f, x)` | `f.integrate(&x)` |
| `integrate(f, (x, 0, 1))` | `f.integrate_definite(&x, &ctx.int(0), &ctx.int(1))` |
| `limit(f, x, 0)` | `f.limit(&x, &ctx.int(0))` |
| `limit(f, x, oo)` | `f.limit(&x, &ctx.infinity())` |
| `series(f, x, 0, 5)` | `f.series(&x, &ctx.int(0), 5)` |
| `f.series(x, 0, 5).removeO()` | `f.maclaurin(&x, 5)` |
| `laplace_transform(f, t, s)` | `f.laplace(&t, &s)` |
| `inverse_laplace_transform(F, s, t)` | `F.inverse_laplace(&s, &t)` |
| `summation(f, (k, 0, n))` | `Ex::symbolic_sum(&f, &k, &ctx.int(0), &n).gosper_sum(&k)` |

## Simplification

| SymPy | symplex |
|-------|---------|
| `simplify(expr)` | `expr.simplify()` or `expr.full_simplify()` |
| `expand(expr)` | `expr.expand()` |
| `factor(expr)` | `expr.factor(&x)` |
| `cancel(expr)` | `expr.cancel(&x)` |
| `apart(expr, x)` | `expr.partial_fractions(&x)` |
| `together(expr)` | `expr.together()` |
| `collect(expr, x)` | `expr.collect(&x)` |
| `trigsimp(expr)` | `expr.simplify_trig()` |
| `expand_trig(expr)` | `expr.expand_trig()` |
| `logcombine(expr)` | `expr.log_combine()` |
| `expand_log(expr)` | `expr.expand_log()` |
| `powsimp(expr)` | `expr.simplify_powers()` |
| `combsimp(expr)` | `expr.simplify_combinatorial()` |
| `radsimp(expr)` | `expr.rationalize_denom(&x)` |
| `nsimplify(expr)` | Not available |

## Solving

| SymPy | symplex |
|-------|---------|
| `solve(f, x)` | `f.solve(&x)` → `Result<Vec<Ex>>` |
| `solve(f, x)` (ignoring errors) | `f.solve_or_empty(&x)` → `Vec<Ex>` |
| `solveset(f > 0, x)` | `f.solve_gt(&x)` → `SetEx` |
| `solveset(f >= 0, x)` | `f.solve_ge(&x)` → `SetEx` |
| `dsolve(ode, y(x))` | `ode.solve_ode(&y, &x)` |
| `nsolve(f, x, x0)` | `f.solve_numeric(&x, x0, max_iter, tol)` |
| `solve([eq1, eq2], [x, y])` | `symplex::polysys::solve_system_ex(&[eq1, eq2], &[x, y])` |

## Linear Algebra

| SymPy | symplex |
|-------|---------|
| `M = Matrix([[1,2],[3,4]])` | `let m = matrix![ctx, [1, 2], [3, 4]];` |
| `M.det()` | `m.det().unwrap()` |
| `M.inv()` | `m.inv().unwrap()` |
| `M.eigenvals()` | `m.eigenvals(&lam).unwrap()` |
| `M.charpoly(x)` | `m.char_poly(&x).unwrap()` |
| `M.T` | `m.transpose()` |
| `M * N` | `m.matmul(&n).unwrap()` |
| `M.trace()` | `m.trace().unwrap()` |
| `M.rank()` | `m.rank().unwrap()` |
| `M.nullspace()` | `m.nullspace().unwrap()` |
| `M.jordan_form()` | `m.jordan_form(&lam).unwrap()` |
| `M.exp()` | `m.matrix_exp(&t).unwrap()` |

## Evaluation and Substitution

| SymPy | symplex |
|-------|---------|
| `expr.subs(x, 3)` | `expr.subs(&x, &ctx.int(3))` |
| `expr.subs(x, 3)` (integer shorthand) | `expr.subs_i64(&x, 3)` |
| `expr.subs([(x, 1), (y, 2)])` | `expr.subs(&x, &ctx.int(1)).subs(&y, &ctx.int(2))` |
| `expr.evalf()` | `expr.eval_f64()` → `Result<f64>` |
| `expr.evalf(50)` | `expr.eval_decimal(50)` → `Result<String>` |
| `expr.is_number` | `expr.eval_f64().is_ok()` |
| `expr.free_symbols` | `expr.free_symbols()` |

## Code Generation

| SymPy | symplex |
|-------|---------|
| `lambdify([x], expr)` | `expr.compile(&["x"])` → `Option<Box<dyn Fn(&[f64]) -> f64>>` |
| `rust_code(expr)` | `expr.to_rust_fn("name", &["x"])` → `Result<String>` |
| `cse([expr1, expr2])` | `expr.cse()` → `(Vec<(Ex, Ex)>, Ex)` |
| `latex(expr)` | `expr.to_latex()` → `String` |
| — | `expr.to_json()` → `Result<String>` |

## Number Theory

| SymPy | symplex |
|-------|---------|
| `isprime(n)` | `symplex::ntheory::isprime(n)` |
| `factorint(n)` | `symplex::ntheory::factorint(n)` |
| `nextprime(n)` | `symplex::ntheory::nextprime(n)` |
| `prevprime(n)` | `symplex::ntheory::prevprime(n)` |
| `divisors(n)` | `symplex::ntheory::divisors(n)` |
| `totient(n)` | `symplex::ntheory::totient(n)` |
| `mobius(n)` | `symplex::ntheory::mobius(n)` |
| `mod_inverse(a, m)` | `symplex::ntheory::mod_inverse(a, m)` |
| `crt([r1,r2], [m1,m2])` | `symplex::ntheory::crt_i64(&[r1,r2], &[m1,m2])` |
| `pow(a, e, m)` (3-arg pow) | `symplex::ntheory::mod_pow(a, e, m)` |
| `primerange(2, 50)` | `symplex::ntheory::primes_up_to(50)` |
| `legendre_symbol(a, p)` | `symplex::ntheory::legendre_symbol(a, p)` |

## Combinatorics

| SymPy | symplex |
|-------|---------|
| `stirling(n, k, kind=2)` | `symplex::combinatorics::stirling2(n, k)` → `Option<BigInt>` |
| `stirling(n, k, kind=1, signed=True)` | `symplex::combinatorics::stirling1(n, k)` → `Option<BigInt>` |
| `npartitions(n)` / `partition(n)` | `symplex::combinatorics::partition_count(n)` → `Option<BigInt>` |
| `multinomial_coefficients(n, k)` | `symplex::combinatorics::multinomial(n, &ks)` → `Option<BigInt>` |
| `bell(n)` | `ctx.int(n).bell().eval()` |
| `catalan(n)` | `ctx.int(n).catalan_number().eval()` |
| `fibonacci(n)` | `ctx.int(n).fibonacci().eval()` |
| `bernoulli(n)` | `ctx.int(n).bernoulli_number().eval()` |

## Special Functions

| SymPy | symplex |
|-------|---------|
| `gamma(x)` | `x.gamma()` |
| `erf(x)` | `x.erf()` |
| `erfc(x)` | `x.erfc()` |
| `beta(a, b)` | `a.beta(&b)` |
| `besselj(n, x)` | `x.bessel_j(&n)` |
| `bessely(n, x)` | `x.bessel_y(&n)` |
| `LambertW(x)` | `x.lambertw()` |
| `DiracDelta(x)` | `x.dirac_delta()` |
| `Heaviside(x)` | `x.heaviside()` |
| `digamma(x)` | `x.digamma()` |
| `loggamma(x)` | `x.log_gamma()` |
| `legendre(n, x)` | `x.legendre(&n)` |
| `chebyshevt(n, x)` | `x.chebyshev_t(&n)` |
| `hermite(n, x)` | `x.hermite(&n)` |
| `laguerre(n, x)` | `x.laguerre(&n)` |

## Dimensional Analysis

SymPy does not have a built-in compile-time unit system. symplex provides one:

```rust
use symplex::units::*;

let ctx = Context::new();
let m = Mass::symbol(&ctx, "m");
let a = Acceleration::symbol(&ctx, "a");

// Type-safe: the compiler verifies the dimension
let force = dim!(ctx, Force: &m * &a);

// Typed calculus: d(Length)/d(Time) → Velocity
let t = Time::symbol(&ctx, "t");
let pos = Length::from_ex(expr!(ctx, 1/2 * a * t^2));
let vel: Velocity = pos.diff_wrt(&t);
```

There is no SymPy equivalent for this. The closest is SymPy's `physics.units` module, which performs dimensional analysis at runtime rather than compile time.

## ODE Solving

| SymPy | symplex |
|-------|---------|
| `dsolve(Eq(f(x).diff(x), f(x)), f(x))` | (see below) |
| `classify_ode(ode)` | `ode_expr.classify_ode(&y, &x)` |

ODE solving in symplex uses a different interface from SymPy. Instead of wrapping the ODE in `Eq()` and using `Function('f')`, you build the ODE as an expression involving `y.formal_diff(&x)`:

```rust
let ctx = Context::new();
symplex::syms!(ctx; x);
let y = ctx.symbol("y");
let dy = y.formal_diff(&x);

// y' + 2y = 0
let ode = &dy + &y * 2;
let solution = ode.solve_ode(&y, &x);
println!("{solution}");   // y = C1*exp(-2*x)
```

symplex supports 13 ODE classes: simple separable, full separable, first-order linear (constant and variable coefficient), exact, integrating factor, Bernoulli, Riccati, Euler-Cauchy, second-order constant-coefficient (homogeneous and non-homogeneous), reduction of order, and variation of parameters.

## Things That Don't Have Direct Equivalents

### In SymPy but not symplex

- `Permutation`, `PermutationGroup`, and abstract algebra
- `geometry` module (Point, Line, Circle, Polygon)
- `stats` module (probability distributions)
- `tensor` module (indexed tensors, Einstein summation)
- `physics.quantum` module
- `pdsolve` (PDE solving)
- `Diophantine` equation solving
- `rsolve` (recurrence relation solving)
- `Eq` type (symplex equations are expressions set equal to zero)
- Pretty-printing with Unicode box drawing (`pprint`)
- `O()` notation for series remainders

### In symplex but not SymPy

- Compile-time dimensional analysis (`dim!` macro, quantity types)
- Compile-time expression type safety (`Ex` / `BoolEx` / `SetEx`)
- Thread-safe contexts (`Send + Sync`, no GIL)
- Optimized Rust code generation with CSE (`to_rust_fn`)
- Compiled closures for fast numerical evaluation (`compile`)
- `EvalConfig` for user-controllable computation limits
- `build.rs` code generation pipeline for embedded targets
- WASM compilation target (`symplex-wasm`)