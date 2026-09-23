# Symplex

Symbolic mathematics for Rust.

[![Crates.io](https://img.shields.io/crates/v/symplex.svg)](https://crates.io/crates/symplex)
[![docs.rs](https://docs.rs/symplex/badge.svg)](https://docs.rs/symplex)
[![License](https://img.shields.io/crates/l/symplex.svg)](LICENSE-MIT)

> **Pre-release (0.23).** The API is stabilising but not stable: 0.7.0 reshaped the
> certificate API and 0.10.0 added variants to three fresh enums/structs; every
> breaking change is listed first, under `### Breaking`, in the release's entry in
> [CHANGELOG.md](CHANGELOG.md), and the larger ones have a one-line fix in the book's
> migration pages ([0.11 → 0.12](book/src/reference/migrating-0.12.md),
> [0.6 → 0.7](book/src/reference/migrating-0.7.md), [0.3 → 0.4](book/src/reference/migrating-0.4.md),
> [0.1 → 0.2](book/src/reference/migrating-0.2.md)). Option structs are `#[non_exhaustive]` so that adding
> an option is never a break again. Feedback welcome.
>
> Contributing? See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture, the no-panic
> policy (enforced by a ratchet test), and conventions. The user guide is
> [The Symplex Book](book/src/SUMMARY.md); its "What's New" pages cover the releases
> up to 0.14; later releases are described in [CHANGELOG.md](CHANGELOG.md).

---

## What This Is

symplex is a symbolic computation library — a computer algebra system for Rust with SymPy as the coverage reference. It manipulates mathematical expressions exactly (arbitrary-precision rational arithmetic, never floating point unless you ask) and can differentiate, integrate, sum, solve equations, systems, ODEs and recurrences, simplify, transform, factor, compute Gröbner bases and minimal polynomials, analyse functions, work with random variables and distributions exactly, do exact linear algebra, linear programming and polytope geometry, and generate Rust, C99, Python, NumPy or Julia code from symbolic results.

Two things set it apart from a port of SymPy: **exact certificates** — LP duals and Farkas vectors, Handelman / Pólya / sum-of-squares proofs of polynomial inequalities that are re-verified with exact arithmetic — and **Lean 4 / Mathlib export** of those proofs, so a Rust program can produce a theorem a proof assistant checks. It is designed for Rust developers in robotics, control, physics and signal processing, and for anyone generating machine-checked mathematics.

## Quick Example

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Build an expression and differentiate
    let f = expr!(ctx, x^3 - 2*x + 1);
    let df = f.diff(&x);
    println!("f'(x) = {df}");                        // 3*x^2 - 2

    // Solve an equation (identities and contradictions are errors, not [])
    let roots = expr!(ctx, x^2 - 5*x + 6).solve(&x).unwrap();
    println!("roots: {roots:?}");                     // [Ex(3), Ex(2)]

    // Definite integral — improper and divergent cases are handled honestly
    let gauss = expr!(ctx, exp(-x^2)).integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity());
    println!("∫ e^(-x²) dx = {gauss}");                // sqrt(pi)

    // Simplify a trig identity
    println!("{}", expr!(ctx, sin(x)^2 + cos(x)^2).simplify());   // 1

    // Generate optimized Rust code from a symbolic result
    let code = df.to_rust_fn("gradient", &["x"]).unwrap();
    println!("{code}");
    // → pub fn gradient(x: f64) -> f64 { 3_f64.mul_add(x.powi(2), -2_f64) }

    // Or compile to a callable — no codegen, just fast evaluation
    let grad = df.compile(&["x"]).unwrap();
    println!("f'(2) = {}", grad(&[2.0]));             // 10.0
}
```

```sh
cargo add symplex
```

---

## When to Use This

- You need symbolic differentiation, integration, summation, or equation solving and want to stay in Rust.
- You are generating numerical code from symbolic derivations — Jacobians, transfer functions, filter coefficients, control laws — as Rust or C99.
- You need compile-time dimensional analysis for physical quantities.
- You need thread-safe symbolic computation without a GIL or global interpreter lock.
- You want exact rational arithmetic (`1/3` stays as `1/3`, not `0.33333...`).
- You want a *proof*, not a number: a certificate that an inequality holds on a region, exportable to Lean 4 / Mathlib and compiled there.
- You need exact probability — `P(Binomial(5, 1/3) > 2) = 17/81`, `E[X | X > 0] = √(2/π)` — rather than Monte-Carlo estimates.

## When Not to Use This

- You need a mature CAS with decades of community validation — use [SymPy](https://www.sympy.org/). It has broader coverage, more special functions, and a much larger test corpus.
- You need geometry, tensor algebra, quantum mechanics, or PDE solving — these are not available (planar/space geometry is next on the roadmap; random variables arrived in 0.11 and the data-statistics modules — tests, estimation, agreement, survival, … — in 0.13–0.21).
- You need interactive notebook-style exploration — symplex is a library, not an application. (Though see `cargo run --example repl` for a basic REPL.)
- You need results verified against extensive known-answer databases — symplex has ~12,900 tests including SymPy cross-validation fixtures and every 0.9+ test cites its SymPy reference value, but SymPy has orders of magnitude more coverage.
- You need large-scale or sparse numerical optimisation — the exact simplex is dense (`O(m·n)` integer operations per pivot, in `i64`/`i128`/256-bit/`BigInt` as the numbers grow; hundreds of rows, not hundreds of thousands), and the `f64` routines are the classic derivative-free methods, not a replacement for a dedicated optimisation library.

---

## What You Can Do

### Calculus

Differentiation handles the chain rule, product rule, all elementary functions, and the special functions (Bessel, orthogonal polynomials, `digamma → polygamma`). Indefinite integration uses 15+ strategies including by-parts, u-substitution, partial fractions, trig substitution, the Risch algorithm, Lazard–Rioboo–Trager log-to-real conversion, and heuristic integration. Radical coefficients (e.g., `√5` from cyclotomic denominators) are handled exactly via algebraic number field arithmetic.

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; x);

expr!(ctx, sin(x^2)).diff(&x);                      // 2*x*cos(x^2)
expr!(ctx, x * exp(x)).integrate(&x);               // x*exp(x) - exp(x)
expr!(ctx, sin(x) / x).limit(&x, &ctx.int(0));      // 1  (Gruntz algorithm)
expr!(ctx, exp(x)).series(&x, &ctx.int(0), 5);      // 1 + x + x^2/2 + x^3/6 + x^4/24

// One-sided limits; the two-sided limit stays a `Limit` node when they disagree
(1 / &x).limit_right(&x, &ctx.int(0));              // oo
(1 / &x).limit_left(&x, &ctx.int(0));               // -oo
```

### Definite, Improper and Numeric Integration

`integrate_definite` locates interior singularities, treats infinite bounds and endpoint singularities as improper integrals via one-sided limits, resolves `Abs`/`Heaviside`/`DiracDelta`/`Piecewise` integrands, and consults a table of ~30 classical improper integrals (with symbolic parameters under assumptions). Divergence is reported, never hidden.

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; x);
let (zero, one, inf) = (ctx.int(0), ctx.int(1), ctx.infinity());

x.powi(2).integrate_definite(&x, &zero, &one);                  // 1/3
(-&x).exp().integrate_definite(&x, &zero, &inf);                // 1
(&x.sin() / &x).integrate_definite(&x, &zero, &inf);            // 1/2*pi
x.ln().integrate_definite(&x, &zero, &one);                     // -1
x.abs().integrate_definite(&x, &ctx.int(-2), &ctx.int(3));      // 13/2

// ∫₋₁¹ dx/x² diverges: F(1) − F(−1) = −2 would be wrong, so it is an error
let r = x.powi(-2).try_integrate_definite(&x, &ctx.int(-1), &one);
assert!(matches!(r, Err(SymplexError::Divergent { .. })));

// Adaptive Gauss–Kronrod (G7/K15) quadrature when there is no closed form
let v = x.powi(2).exp().integrate_numeric(&x, &zero, &one).unwrap();   // 1.4626517459…

// Residues at poles of any order, and at infinity
let z = ctx.symbol("z");
(&z.exp() / &z.powi(3)).residue(&z, &zero);                     // 1/2
(1 / (&z.powi(2) + 1)).residue_at_infinity(&z);                 // 0
```

### Summation, Products and Series

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; k, x);
let n = ctx.symbol_with("n", &[Assumption::Integer, Assumption::Positive]);
let (zero, one, inf) = (ctx.int(0), ctx.int(1), ctx.infinity());

k.powi(5).summation(&k, &one, &n);                    // 1/6*n^6 + 1/2*n^5 + 5/12*n^4 - 1/12*n^2
(&k * &ctx.int(2).pow(&k)).summation(&k, &zero, &n);  // 2^(n + 1)*(n - 1) + 2   (Gosper)
n.binomial(&k).summation(&k, &zero, &n);              // 2^n
k.powi(-2).summation(&k, &one, &inf);                 // 1/6*pi^2
k.powi(-3).summation(&k, &one, &inf);                 // zeta(3)   (odd p: symbolic)
(&x.pow(&k) / &k.factorial()).summation(&k, &zero, &inf);   // exp(x)
(1 - k.powi(-2)).product_over(&k, &ctx.int(2), &inf); // 1/2

(1 / &k).is_convergent(&k);                           // Some(false)

// Formal power series: lazy exact coefficients and closed-form general terms
let s = x.sin().fps_maclaurin(&x);
s.coefficient(51);                                    // -1/1551118753287382280224243016469303211063259720016986112000000000000
s.general_term(&k);                                   // Some(sin(1/2*k*pi)/k!)
s.reversion().unwrap().coefficients(6);               // asin: [0, 1, 0, 1/6, 0, 3/40]
```

### Function Analysis

SymPy's `calculus.util` on `Ex` (0.9): singularities, stationary points, extrema on an interval or union of intervals (one-sided limits at open or infinite endpoints, `±∞` allowed), monotonicity and convexity (exact for polynomial and rational derivatives via Sturm sequences; three-valued, never a guess), periodicity and the range of a function.

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; x);
let f = &x.powi(3) - 3 * &x;
let interval = ctx.interval(&ctx.int(-2), &ctx.int(2), IntervalKind::Closed);
f.stationary_points(&x, None).unwrap();                          // {-1, 1}
f.maximum(&x, &interval).unwrap();                               // 2
(1 / (&x.powi(2) - 1)).singularities(&x, None).unwrap();          // {-1, 1}
x.powi(3).is_increasing(&x, &ctx.reals());                       // Some(true)
((2 * &x).sin() + (3 * &x).cos()).periodicity(&x).unwrap();      // 2*pi
```

Also: `minimum`, `is_decreasing`, `is_strictly_increasing`/`_decreasing`, `is_monotonic`, `is_convex`, `function_range`.

### Complex Analysis and Special Functions

`re`, `im`, `conjugate`, `arg` are honest about unknown realness: with no assumption on `z`, `z.re()` is the unevaluated `re(z)`.

```rust
# use symplex::prelude::*;
let ctx = Context::new();
let z = ctx.symbol("z");
let x = ctx.symbol_with("x", &[Assumption::Real]);
let y = ctx.symbol_with("y", &[Assumption::Real]);
let i = ctx.i_unit();

let w = &x + &i * &y;
w.conjugate();                                        // x - y*I
w.abs_squared();                                      // x^2 + y^2
w.exp().as_real_imag();                               // (cos(y)*exp(x), sin(y)*exp(x))
z.re();                                               // re(z)   — not assumed real
z.exp().re();                                         // cos(im(z))*exp(re(z))
(1 / &ctx.int(0)).eval();                             // zoo   (complex infinity)

// New constants and special functions with exact values and evalf
ctx.int(1).digamma().eval();                          // -EulerGamma
ctx.int(4).zeta().eval();                             // 1/90*pi^4
ctx.int(1).polygamma(&ctx.int(1)).eval();             // 1/6*pi^2
ctx.infinity().si().eval();                           // 1/2*pi
ctx.catalan().eval_decimal(30).unwrap();              // 0.915965594177219015054603514932
```

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
// 0.9: 24 more special functions — exact values, derivative rules, arbitrary-precision evalf
// (checked at 40 digits against mpmath), Display/LaTeX/parse, and integration results
x.powi(2).exp().integrate(&x);                        // 1/2*sqrt(pi)*erfi(x)      (was unevaluated before 0.9)
(x.sinh() / &x).integrate(&x);                        // Shi(x)
ctx.rational(1, 2).polylog(&ctx.int(2)).eval();       // -1/2*ln(2)^2 + 1/12*pi^2   (Li₂(½))
ctx.int(0).elliptic_k().eval();                       // 1/2*pi
x.airyai().diff(&x);                                  // airyaiprime(x)
x.assoc_legendre(&ctx.int(2), &ctx.int(1)).eval();    // -3*x*sqrt(-x^2 + 1)
expr!(ctx, airyai(x) + polylog(2, x));                // the macro knows them too
```

Also: Gamma, log-gamma, `lowergamma`/`uppergamma`, erf/erfc/`erfi`/`erfinv`/`erfcinv`, Beta, Lambert W, Bessel J/Y/I/K, Airy Ai/Bi and derivatives, elliptic K/E/F/Π, `expint`/`E1`, `Shi`/`Chi`, Fresnel S/C, `polylog`/`dirichlet_eta`, Legendre/Chebyshev/Hermite/Laguerre and the associated/Gegenbauer/Jacobi families, `Si`/`Ci`/`Ei`/`li`, Kronecker delta — all with arbitrary-precision evaluation.

### Algebra and Factoring

Polynomial operations work over ℚ using arbitrary-precision rational arithmetic. Univariate factoring over ℤ uses Berlekamp–Zassenhaus (any degree); multivariate factoring uses Kronecker substitution. Gröbner bases use Buchberger's algorithm with FGLM order conversion.

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; x, y);

expr!(ctx, x^12 - 1).factor(&x);                      // (x - 1)*(x + 1)*(x^2 + x + 1)*(x^2 + 1)*(x^2 - x + 1)*(x^4 - x^2 + 1)
expr!(ctx, x^3 - x*y^2 + x^2 - y^2).factor_all();     // (x + 1)*(x + y)*(x - y)
expr!(ctx, (x + 1)^3).expand();                       // x^3 + 3*x^2 + 3*x + 1
expr!(ctx, (x^2 - 1) / (x - 1)).cancel(&x);           // x + 1

// Polynomial algebra on Ex: resultant, discriminant, division, gcdex, real-root isolation, …
expr!(ctx, x^3 - x).discriminant(&x);                 // Some(4)
expr!(ctx, x^5 - x - 1).count_real_roots(&x);         // Some(1)
expr!(ctx, x^4 + 1).is_irreducible(&x);               // Some(true)

// 0.9: algebraic numbers, variable-free gcd, Gröbner bases, real roots as RootOf, GF(p), symbolic resultants
syms!(ctx; a, b, c);
(ctx.int(2).sqrt() + ctx.int(3).sqrt()).minimal_polynomial(&x).unwrap();   // x^4 - 10*x^2 + 1
(&x.powi(2) - &y.powi(2)).gcd_all(&(&x - &y)).unwrap();                    // x - y
Ex::groebner(&[&x.powi(2) + &y.powi(2) - 1, &x - &y], &[x.clone(), y.clone()], MonomialOrder::Lex).unwrap();
                                                      // [x - y, y^2 - 1/2]
(&x.powi(3) - 2 * &x).real_roots(&x).unwrap();        // [RootOf(x^2 - 2, 0), 0, RootOf(x^2 - 2, 1)]  (ascending)
(&x.powi(2) + 1).factor_mod(&x, 5).unwrap();          // (1, [(x + 2, 1), (x + 3, 1)])
(&a * &x.powi(2) + &b * &x + &c).discriminant_symbolic(&x).unwrap();       // -4*a*c + b^2
```

### Polynomials as Data and Rational Normal Forms

`Poly` (0.3) views an expression as a sparse polynomial in an explicit list of generators. Coefficients are exact rationals *or* symbolic parameter expressions, terms come back in SymPy's lex-descending order, and nothing is approximated. `degree`/`coeffs`/`leading_coeff` on `Ex` accept symbolic coefficients too. `ratsimp` is a rational-function normal form — one cancelled fraction with integer-primitive numerator and denominator — and `solve` uses it for parametric linear and quadratic equations.

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; x, y, a, j, r);

// Polynomial introspection on Ex with symbolic (var-free) coefficients
let e = &a * &x.powi(2) + &x * (&a + 1) + 3;
e.degree(&x);                                         // Some(2)
e.coeffs(&x);                                         // Some([3, a + 1, a])   (ascending)
e.leading_coeff(&x);                                  // Some(a)

// Poly: sparse terms over explicit generators, exact evaluation, calculus
let p = (&a * &x.powi(2) + &x * &y * 3 - &y + 1).as_poly(&[&x, &y]).unwrap();
p.terms();                                            // [([2, 0], a), ([1, 1], 3), ([0, 1], -1), ([0, 0], 1)]
p.coeff_monomial(&[1, 1]).unwrap();                   // 3
p.total_degree();                                     // Some(2)
p.eval_gen(&x, &ctx.int(2)).unwrap();                 // Poly(5*y + 4*a + 1, y)
p.derivative(&x).unwrap().to_ex();                    // 2*a*x + 3*y

// Rational normal form: nested fractions collapse to one cancelled fraction
(1 / (&x + 1 / &y) + 1 / (1 / &x + &y)).ratsimp();   // (x + y)/(x*y + 1)
((&r * 3 - 1) / (&j + 1) - (&r + 1) / (&j * 2)).solve(&r);   // Ok([(3*j + 1)/(5*j - 1)])

// Exact sign of a rational-coefficient polynomial on an interval (square-free part + Sturm)
(&x.powi(3) - &x).poly_is_nonnegative_on(&x, &ctx.int(2), &ctx.infinity());               // Some(true)
(&x.powi(2) - &x * 2 + 1).poly_is_positive_on(&x, &ctx.neg_infinity(), &ctx.infinity());  // Some(false) — touches 0 at x = 1

// Linear certificates: (x + 1)² = λ₁·(x + 1) + λ₂·(x² − 1) as an exact linear system
let (h1, h2) = ((&x + 1).as_poly(&[&x]).unwrap(), (&x.powi(2) - 1).as_poly(&[&x]).unwrap());
let goal = (&x + 1).powi(2).as_poly(&[&x]).unwrap();
let basis = Poly::monomial_basis(&[&h1, &h2, &goal]).unwrap();    // [[2], [1], [0]]
let m = Poly::coefficient_matrix(&[&h1, &h2], &basis).unwrap();   // [[0, 1], [1, 0], [1, -1]]
let b = Poly::coefficient_matrix(&[&goal], &basis).unwrap();      // [[1], [2], [1]]
linsolve_matrix(&m, &b);                              // Ok(Unique([(x1, 2), (x2, 1)]))
```

Also: `Poly::{from_terms, all_coeffs, degree_list, eval, add/sub/mul/pow/scale, content_and_primitive, monic, to_multipoly, nroots}`, and on `MultiPoly` a heuristic multivariate `gcd`/`lcm`, `integer_content`, `clear_denominators`.

### Simplification and the Rule Engine

`simplify()` tries a dozen strategies and iterates to a fixpoint. The pattern-matching engine behind it is public in 0.2: build your own rules (symbols ending in `_` are wildcards, `rest__` absorbs the rest of a sum or product), rewrite with them, trace what fired, and interleave them with the built-in simplifier.

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; x, y);
let (a, b) = (ctx.symbol("a_"), ctx.symbol("b_"));

let rules = RuleSet::from_rules(vec![
    Rule::new("sin_sq", &a.sin().powi(2), &(1 - &a.cos().powi(2))),
    Rule::new("ln_add", &(&a.ln() + &b.ln()), &(&a * &b).ln()),
]);
(&x.sin().powi(2) + 3).rewrite(&rules);               // -cos(x)^2 + 4
(&x.ln() + &y.ln()).rewrite(&rules);                  // ln(x*y)

let (result, steps) = (&x.sin().powi(2) + &x.cos().powi(2)).simplify_traced(&SimplifyOpts::default());
// result = 1; steps name the strategy and the rules that fired

x.powi(4).subs_algebraic(&x.powi(2), &y);             // y^2   (plain subs would leave x^4)
(ctx.int(5) + ctx.int(24).sqrt()).sqrt().sqrtdenest();  // sqrt(2) + sqrt(3)
```

### Equation Solving

Polynomial equations are solved through quartic by radicals; degree ≥ 5 produces `RootOf` nodes with numerical evaluation. Transcendental equations use inversion peeling and Lambert W. `solve` never lies: identities are `Err(InfiniteSolutions)`, contradictions and range violations are `Err(NoSolution)`.

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; x, y, z);

expr!(ctx, x^2 - 5*x + 6).solve(&x);                  // Ok([3, 2])
(&x.sin() - &ctx.rational(1, 2)).solve(&x);           // Ok([1/6*pi, 5/6*pi])
(&x.sin() - 2).solve(&x);                             // Err(NoSolution)
(&x - &x).solve(&x);                                  // Err(InfiniteSolutions)

// All periodic solutions, with an integer parameter
let fam = (&x.sin() - &ctx.rational(1, 2)).solve_general(&x).unwrap();
// fam.solutions = [2*n*pi + 1/6*pi, 2*n*pi + 5/6*pi], fam.parameters = [n]

// Linear systems: unique / parametric / inconsistent, symbolic coefficients allowed
let sol = linsolve(&[&x + &y + &z - 6, &x - &y - 2], &[x.clone(), y.clone(), z.clone()]).unwrap();
// LinearSolution::Parametric { solution: [x = -z/2 + 4, y = -z/2 + 2, z = z], free: [z] }

// Polynomial systems via Gröbner bases (algebraic solutions)
symplex::polysys::solve_system_ex(&[&x.powi(2) + &y.powi(2) - 1, &x - &y], &[x.clone(), y.clone()]);
// Ok([[√2/2, √2/2], [-√2/2, -√2/2]])

// Inequalities (sign-chart method), including absolute values
expr!(ctx, x^2 - 4).solve_gt(&x);                     // (-oo, -2) ∪ (2, oo)
(&(&x - 1).abs() - 2).solve_lt(&x);                   // (-1, 3)
```

### Differential Equations and Recurrences

16 ODE classes (separable, linear, Bernoulli, Riccati, Euler–Cauchy, exact, integrating factor, Clairaut, nth-order constant-coefficient, variation of parameters, systems via matrix exponential, …), initial-value problems, and linear recurrences.

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; x, n);
let y = ctx.symbol("y");
let (d1, d2) = (y.formal_diff(&x), y.formal_diff(&x).formal_diff(&x));

(&d2 + &y).solve_ode(&y, &x);                         // y = C1*sin(x) + C2*cos(x)
(&d2 + &y).solve_ode_ivp(&y, &x, &[                   // y(0) = 0, y'(0) = 1
    InitialCondition { order: 0, x: ctx.int(0), value: ctx.int(0) },
    InitialCondition { order: 1, x: ctx.int(0), value: ctx.int(1) },
]);                                                   // Ok(sin(x))
(&d1 * &x - &y - &d1.powi(2)).classify_ode(&y, &x);   // Clairaut

// a(n+2) = a(n+1) + a(n), a(0) = 0, a(1) = 1  →  Binet's formula
symplex::rsolve::rsolve_linear(&[ctx.int(-1), ctx.int(-1), ctx.int(1)], None, &n, &[ctx.int(0), ctx.int(1)]);
```

### Sets and Logic

`SetEx` and `BoolEx` are first-class: intervals, finite sets, unions with a normal form, three-valued queries, and boolean normal forms with a DPLL satisfiability check.

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; x, p, q);

let a = ctx.interval(&ctx.int(0), &ctx.int(5), IntervalKind::Closed);    // [0, 5]
let b = ctx.interval(&ctx.int(3), &ctx.int(10), IntervalKind::LeftOpen); // (3, 10]
a.intersection(&b).simplify();                        // (3, 5]
a.symmetric_difference(&b);                           // [0, 3] ∪ (5, 10]
a.contains(&ctx.int(7));                              // Some(false)
a.contains(&x);                                       // None
a.union(&b).measure();                                // Some(10)

let conds = [x.gt(&ctx.int(0)), x.le(&ctx.int(5)), (&x.powi(2) - 4).gt(&ctx.int(0))];
reduce_inequalities(&conds, &x);                      // Ok((2, 5])

let (pp, qq) = (p.gt(&ctx.int(0)), q.gt(&ctx.int(0)));
pp.and(&qq).or(&pp).simplify();                       // p > 0
pp.and(&qq).not().to_nnf();                           // 0 >= p | 0 >= q
pp.or(&pp.not()).is_tautology();                      // Some(true)
```

### Linear Algebra

Symbolic matrices with exact decompositions. The eigen family needs no dummy variable in 0.2, structure tests are three-valued, and preconditions are `Result`s.

```rust
# use symplex::prelude::*;
# use symplex::syms;
use symplex::linprog::q;   // exact rational literal: q(1, 2) = 1/2

let ctx = Context::new();
syms!(ctx; t, n);
let m = matrix![ctx, [2, 1], [1, 2]];

m.det().unwrap();                                     // 3
m.eigenvals().unwrap();                               // [3, 1]
m.char_poly(&ctx.symbol("λ")).unwrap();               // λ^2 - 4*λ + 3
m.diagonalize().unwrap();                             // Diagonalization { p, d } with A = P·D·P⁻¹
m.matrix_exp_t(&t).unwrap();                          // [[e^(3t)/2 + e^t/2, …], …]
m.matrix_pow_symbolic(&n).unwrap();                   // [[3^n/2 + 1/2, 3^n/2 - 1/2], …]
m.matrix_sqrt().unwrap();

let spd = matrix![ctx, [4, 12, -16], [12, 37, -43], [-16, -43, 98]];
spd.cholesky().unwrap();                              // [[2,0,0],[6,1,0],[-8,5,3]]
spd.is_positive_definite();                           // Some(true)
matrix![ctx, [1, 1, 0], [1, 0, 1], [0, 1, 1]].qr().unwrap();   // Qr { q, r }, exact radicals

// Irreducible characteristic polynomials give exact, evaluable RootOf eigenvalues
matrix![ctx, [0, 1, 0], [0, 0, 1], [1, 1, 0]].eigenvals().unwrap();   // [RootOf(λ^3 - λ - 1, 0), …]

// 0.3: index-list extraction, exact rationals in and out, three-valued structure tests
m.extract(&[1, 0], &[0]).unwrap();                    // [[1], [2]]
Matrix::from_ratio(&ctx, &[vec![q(1, 2), q(3, 1)]]).unwrap();   // [[1/2, 3]]
(&m - &m.transpose()).is_zero();               // Some(true)   (m is symmetric)

// 0.3.5: QMatrix / ZMatrix — plain exact matrices over ℚ / ℤ, no expression arena.
// Fraction-free (Bareiss) elimination: a 30×30 rational inverse takes 10 ms, not 470.
let h = QMatrix::from_fn(4, 4, |i, j| q(1, (i + j + 1) as i64));   // Hilbert matrix
h.det().unwrap();                                     // 1/6048000
assert_eq!(h.inv().unwrap()[(3, 3)], q(2800, 1));     // the inverse is integral
let (r, pivots) = QMatrix::from_i64(&[&[1, 2, 3], &[4, 5, 6]]).unwrap().rref();
// r = [[1, 0, -1], [0, 1, 2]], pivots = [0, 1]
ZMatrix::from_i64(&[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]).unwrap().smith_normal_form();
```

`Matrix::{rref, rank, nullspace, det, inv, solve}`, `linsolve`/`linsolve_matrix` and the normal forms route through `QMatrix`/`ZMatrix` automatically whenever every entry is a rational literal, so existing code gets the speed-up without changes.

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
// 0.9: singular values and condition number, a pseudo-inverse defined for every matrix,
// rank decomposition, Hessenberg form, permanent, companion / Jordan blocks, exact LLL
matrix![ctx, [1, 2], [3, 4]].singular_values().unwrap();     // [sqrt(sqrt(221) + 15), sqrt(-sqrt(221) + 15)]
matrix![ctx, [1, 2], [2, 4]].pinv().unwrap();                // [[1/25, 2/25], [2/25, 4/25]]  (rank-deficient)
matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]].rank_decomposition().unwrap();   // RankDecomposition { c, f } with A = C·F
matrix![ctx, [1, 2], [3, 4]].permanent().unwrap();           // 10
matrix![ctx, [1, 2], [3, 4]].inv_mod(5).unwrap();            // [[3, 1], [4, 2]]
ZMatrix::from_i64(&[&[1, 1, 1], &[-1, 0, 2], &[3, 5, 6]]).unwrap().lll_default().unwrap();   // [[0, 1, 0], [1, 0, 1], [-1, 0, 2]]
```

Also: LU, LDLᵀ, Gram–Schmidt, Jordan form, `matrix_log`, Kronecker product, rank/nullspace/rowspace, norms, least squares, Hessian, Wronskian, Casoratian, `hessenberg`, `companion`, `jordan_block`, row/column insert/delete/permute, quaternions, vector calculus in Cartesian/cylindrical/spherical coordinates, state-space ↔ transfer function; `select_rows`/`select_cols`, `from_bigint`/`from_f64_rows`, `to_rational_rows`/`to_bigint_rows`, `is_integer_matrix`, `subs_map`, `nnz`.

### Exact Optimization and Integer Lattices

Linear programs are solved over ℚ by a two-phase simplex: optima, shadow prices and Farkas infeasibility certificates are exact, never "infeasible to within tolerance". The tableau is fraction-free (integers with a common denominator) and runs on `i64`, then `i128`, then 256-bit, then `BigInt` cells as the numbers grow — the same pivot path in every width, so results are identical and small problems never touch the heap. Dantzig's rule is used until twelve consecutive degenerate pivots, then Bland's until the next improving step (finite, and not condemned to Bland's slow walk on the degenerate certificate LPs). A `Budget` (deadline and/or pivot cap, checked at every pivot) turns a runaway solve into `LpStatus::BudgetExhausted`. Integer matrices get Hermite and Smith normal forms with unimodular transforms, ℤ-bases of integer kernels, and exact LLL reduction.

```rust
# use symplex::prelude::*;
use symplex::linprog::{feasible_nonneg, q, qi};
use symplex::normalforms::hermite_normal_form_with_transform;
let ctx = Context::new();

// max 5x + 4y  s.t.  6x + 4y ≤ 24,  x + 2y ≤ 6,  x, y ≥ 0
let sol = LpProblem::maximize(vec![qi(5), qi(4)])
    .le(vec![qi(6), qi(4)], qi(24))
    .le(vec![qi(1), qi(2)], qi(6))
    .solve().unwrap();
sol.status;                                           // Optimal
sol.x;                                                // [3, 3/2]
sol.objective;                                        // Some(21)
sol.duals;                                            // [3/4, 1/2]   shadow prices: yᵀb = 21 = cᵀx*

// x + y ≤ 1 and x + y ≥ 2 cannot both hold — here is the proof
let bad = LpProblem::minimize(vec![qi(0), qi(0)])
    .le(vec![qi(1), qi(1)], qi(1))
    .ge(vec![qi(1), qi(1)], qi(2))
    .solve().unwrap();
bad.status;                                           // Infeasible
bad.farkas;                                           // Some([1, -1])   Aᵀy = 0, yᵀb = −1 < 0

// "Is there μ ≥ 0 with Aμ = b?", exactly (Farkas / Carathéodory searches)
feasible_nonneg(&[vec![q(1, 3), q(1, 7)], vec![qi(1), qi(-1)]], &[qi(1), qi(0)]);   // Ok(Some([21/10, 21/10]))

// Integer normal forms: H = U·A (row style), S = U·A·V, ℤ-basis of the kernel
let a = matrix![ctx, [2, 4, 4], [-6, 6, 12], [10, -4, -16]];
let HermiteNormalForm { h, u } = hermite_normal_form_with_transform(&a).unwrap();
// h = [[2, 4, 4], [0, 6, 0], [0, 0, 12]]
assert_eq!((&u * &a).eval(), h);                      // det U = −1
a.smith_normal_form().unwrap();                       // [[2, 0, 0], [0, 6, 0], [0, 0, 12]]
matrix![ctx, [2, 1, 1]].integer_nullspace().unwrap(); // [(1, 0, −2)ᵀ, (0, 1, −1)ᵀ] — generates every integer solution
```

Also: `linprog` (SciPy-shaped), `linprog_matrix` (from `Matrix` data), per-variable bounds and free variables, `nonneg_combination` / `feasible_nonneg_certified` (cone membership with the separating Farkas vector on failure), `column_hermite_normal_form` (SymPy's convention), `smith_normal_form_with_transforms`, `is_unimodular`, `lattice_determinant`.

### Certified Inequalities and Lean Export

`certificates::prove_nonnegative_on_box` proves `goal ≥ 0` on a box with a Handelman certificate — an exact identity `goal = Σ λₖ·Π(xᵢ − lᵢ)^a(uᵢ − xᵢ)^b` with `λ ≥ 0` found by the exact LP and **re-verified with exact polynomial arithmetic** — or refutes the claim with an exact counterexample. The certificate exports as a Lean 4 / Mathlib theorem whose proof is `nlinarith` over exactly those products; `Ex::to_lean()` renders any elementary expression in Mathlib syntax.

```rust
# use symplex::prelude::*;
# use symplex::syms;
use symplex::certificates::{prove_nonnegative_on_box, BoxBound, BoxOutcome};
let ctx = Context::new();
syms!(ctx; x, y);
let square = [
    BoxBound { var: x.clone(), lo: ctx.int(0), hi: ctx.int(1) },
    BoxBound { var: y.clone(), lo: ctx.int(0), hi: ctx.int(1) },
];

let cert = match prove_nonnegative_on_box(&(1 - &x * &y), &square, 2).unwrap() {
    BoxOutcome::Proved(c) => c,
    other => panic!("{other:?}"),
};
cert.to_string();                                     // -x*y + 1 = -y + y*(-x + 1) + 1, 0 ≤ x ≤ 1, 0 ≤ y ≤ 1
cert.verify();                                        // true — exact re-check, independent of the LP
cert.to_lean("one_minus_xy").unwrap();
// theorem one_minus_xy (x y : ℝ) (_h_x_lo : (0 : ℝ) ≤ x) (h_x_hi : x ≤ (1 : ℝ)) (h_y_lo : (0 : ℝ) ≤ y)
//     (h_y_hi : y ≤ (1 : ℝ)) :
//     0 ≤ -(x * y) + 1 := by
//   nlinarith [sub_nonneg.mpr h_y_hi, mul_nonneg (sub_nonneg.mpr h_x_hi) (sub_nonneg.mpr h_y_lo)]

prove_nonnegative_on_box(&(&x * &y - ctx.rational(1, 2)), &square, 2).unwrap();
                                                      // Refuted { point: [(x, 0), (y, 0)], value: -1/2, .. }
((&x - 1) / (2 * &x)).to_lean().unwrap();             // "(x - 1) / (2 * x)"
x.sqrt().gt(&ctx.int(0)).to_lean().unwrap();          // "0 < Real.sqrt x"

// 0.4: a polyhedron whose facets depend on a parameter j ≥ j₀.  On { t ≥ r, t + j·r ≥ j + 1 }
// the goal t − 1 ≥ 0 needs the multiplier λ(j) = 1 + j: (j + 1)(t − 1) = j·h₀ + h₁.
use symplex::certificates::{prove_nonnegative_on_polyhedron, ParamBound, PolyhedronOpts};
syms!(ctx; j, r, t);
let hyps = [&t - &r, &t + &j * &r - &j - 1];
let j_bound = ParamBound { var: j.clone(), lower: ctx.int(0) };
let out = prove_nonnegative_on_polyhedron(&(&t - 1), &hyps, Some(&j_bound), &PolyhedronOpts::default()).unwrap();
out.certificate().unwrap().to_string();               // (j + 1)*(t - 1) = j*h0 + h1; h0 = -r + t, h1 = j*r - j + t - 1; j ≥ 0
out.certificate().unwrap().to_lean("needs_lambda").unwrap();
//   … have h0J := mul_nonneg hJ0 h0
//   have hg : (0 : ℝ) ≤ (j + 1) * (t - 1) := by linarith only [h0J, h1]
//   have hg' := nonneg_of_mul_nonneg_right hg (by linarith only [hJ0])
//   linarith only [hg']
```

```rust
# use symplex::prelude::*;
# use symplex::syms;
# let ctx = Context::new();
// 0.6: sums of squares — non-negativity on all of ℝⁿ, interior zeros included.  The Gram SDP is
// solved by a built-in interior-point method, rounded, projected and checked exactly (rational LDLᵀ).
use symplex::certificates::{prove_sos, SosOpts};
syms!(ctx; x, y, z);
let amgm = x.powi(4) + y.powi(4) + z.powi(4) - &x * &y * &z * 4 + 1;
let cert = prove_sos(&amgm, &[x.clone(), y.clone(), z.clone()], &SosOpts::default()).unwrap();
cert.certificate().unwrap().to_string();
// x^4 + y^4 + z^4 - 4*x*y*z + 1 = (-1/3*x^2 - 1/3*y^2 - 1/3*z^2 + 1)^2 + 2/3*(-y*z + x)^2 + 2/3*(-x*z + y)^2
//   + 2/3*(-x*y + z)^2 + 2/3*(-x^2 + y^2)^2 + 8/9*(-1/2*x^2 - 1/2*y^2 + z^2)^2
cert.certificate().unwrap().to_lean("amgm3").unwrap();   // have h : … := by ring;  rw [h];  positivity
```

Every prover returns one `Outcome<C, U>` — `Proved(certificate)`, `Refuted { point, value, .. }` with an exact counterexample, or `Unknown(what was tried)`, never a wrong `Proved` — and every certificate type implements the `Certificate` trait (`goal`, `verify`, `to_lean`, JSON round trip with re-verification). A prover call can be given a **budget** (`PolyhedronOpts::default().with_time_limit(Duration::from_secs(150))`, or `with_max_pivots`) and returns `Unknown` naming the limit rather than running on. For assembling many certificates into one lemma, `lean::{Block, Tactic, Decl}` is a small structured model of a tactic proof that renders `have`s, bullets and `by` blocks from their tactic column — no hand-counted indentation — and its output for a real generator's proof shape is pinned to text that compiled against Mathlib.

Also: `prove_polyhedron_empty` (the same identity with goal `−1`: a cell is empty for every `j`), `PolyhedronProver` (parse the hypotheses once, prove many goals; `prove_poly` takes an exact `Poly` directly; `used_hyps` names the hypotheses a certificate needs), `prove_nonnegative_on_halfline` / `prove_nonnegative_on_reals` (univariate, Pólya multipliers and square factors), `lean_steps` / `block` / `lean_hints` for dropping a proof into an existing skeleton, `LeanOpts::{prefer_subtraction, single_fraction, symbol_text}`, and `symplex::polytope::{Polytope, ParametricPolytope}` for the exact geometry of the cells — vertices in integer arithmetic with tight sets, `clip` (both sides of a cut from the cached vertices, no re-enumeration), volume in any dimension, `is_full_dimensional` (one LP), redundancy.

### Transforms

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; t, w, s, x);
let a = ctx.symbol_with("a", &[Assumption::Positive]);

(-&a * t.abs()).exp().fourier_transform(&t, &w);      // Ok(2*a/(a^2 + w^2))
(-t.powi(2)).exp().fourier_transform(&t, &w);         // Ok(sqrt(pi)*exp(-1/4*w^2))
(1 / (1 + &x)).mellin_transform(&x, &s);              // Ok((pi/sin(s*pi), re(s) > 0 & 1 > re(s)))
t.sin().laplace(&t, &s);                              // 1/(s^2 + 1)
((&s * -2).exp() / &s).inverse_laplace(&s, &t);       // H(t - 2)
x.sign().fourier_series_on(&x, &(-ctx.pi()), &ctx.pi(), 5).unwrap().truncate(5);
                                                      // 4*sin(x)/pi + 4*sin(3*x)/(3*pi) + 4*sin(5*x)/(5*pi)
```

### Number Theory and Combinatorics

Pollard–Brent rho + ECM factorization, BPSW primality, modular square roots and discrete logarithms, continued fractions, Diophantine equations, and integer sequences.

```rust
use symplex::ntheory::*;
use symplex::diophantine;
use symplex::combinatorics::*;
use symplex::linprog::qi;
use symplex::num_bigint::BigInt;

isprime(561);                                         // false (Carmichael number)
factorint(1_099_532_599_387u64);                      // [(1048583, 1), (1048589, 1)]  — ~1 ms
sqrt_mod(2, 7);                                       // Some(3)
discrete_log(3, 13, 17);                              // Some(4)
primepi(1_000_000);                                   // Some(78498)
continued_fraction_periodic(23);                      // Some(PeriodicContinuedFraction { pre_period: [4], period: [1, 3, 1, 8] })
diophantine::pell(61);                                // Some((1766319049, 226153980))
diophantine::sum_of_two_squares(65);                  // Some((4, 7))
stirling2(10, 4);                                     // Some(34105)
partition_count(100);                                 // Some(190569292)
crt_i64(&[2, 3, 2], &[3, 5, 7]);                      // Some(23)
igcd(&[12i64, 18, 30]);                               // 6    (gcd_many / lcm_many take BigInt slices)
ilcm(&[4i64, 6, 10]);                                 // 60

// 0.10: n-th roots and polynomial congruences for any modulus, and exact discrete transforms
nthroot_mod(11, 4, 19, true);                         // Some([8, 11])
polynomial_congruence(&[1, 0, -3, 5].map(BigInt::from), 1000003);   // [488045, 745229, 766732]  (x³ − 3x + 5 mod p)
is_carmichael(561);                                   // true
symplex::discrete::convolution(&[qi(1), qi(2), qi(3)], &[qi(4), qi(5), qi(6)]);       // [4, 13, 28, 27, 18]
symplex::discrete::ntt(&[1, 2, 3, 4].map(BigInt::from), BigInt::from(998244353));    // number-theoretic transform
```

Also: `quadratic_residues`, `is_nthpow_residue`, `multiplicity`, `primenu`/`primeomega`, `primorial`, `continued_fraction_reduce` (finite and periodic → quadratic surd), `is_amicable`, `binomial_coefficients`; `discrete::{convolution_cyclic, convolution_subset, intt, fwht, mobius_transform}`.

### Numerical Toolbox

Deterministic, budgeted `f64` routines — bracketed roots, derivative-free minimisation, global search in a box, least-squares fits — usable on plain closures or directly on expressions (which are `compile`d first). Bad input is `Err(InvalidArgument)`, a non-finite value is `Err(ComputationFailed)`; nothing panics.

```rust
# use symplex::prelude::*;
# use symplex::syms;
use symplex::optimize::{DeOpts, brent_root, nelder_mead, poly_fit};

let ctx = Context::new();
syms!(ctx; x, y);

// Bracketed roots (Brent–Dekker), on a closure or on a compiled expression
brent_root(|t| t * t - 2.0, 0.0, 2.0, &RootOpts::default()).unwrap();   // 1.41421356237…
(x.cos() - &x).find_root_bracket(&x, 0.0, 1.0).unwrap();                 // 0.739085133215…

// Nelder–Mead: local minimum from a starting point
nelder_mead(|p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2), &[0.0, 0.0], &MinimizeOpts::default()).unwrap();   // x ≈ [1, −2]
let rosen = (1 - &x).powi(2) + 100 * (&y - &x.powi(2)).powi(2);
let r = rosen.minimize_numeric(&[&x, &y], &[-1.2, 1.0]).unwrap();       // r.x ≈ [1, 1], r.fun ≈ 1e-18, r.converged

// Differential evolution: global minimum in a closed box, deterministic for a given seed
let himmelblau = (&x.powi(2) + &y - 11).powi(2) + (&x + &y.powi(2) - 7).powi(2);
let square = [Interval::closed(-5.0, 5.0), Interval::closed(-5.0, 5.0)];
himmelblau.minimize_global_numeric(&[&x, &y], &square, &DeOpts::default()).unwrap();   // fun < 1e-8

// Brent scalar minimisation, and least-squares fits (f64 via Householder QR, or exact rational)
let m = (&x * x.ln()).minimize_scalar_numeric(&x, 0.1, 2.0).unwrap();    // m.x = 0.36787944… (1/e), m.value = −0.36787944… (−1/e)
poly_fit(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 9.0, 19.0], 2).unwrap();     // ≈ [1, 0, 2]   (ascending: 1 + 2x²)
let pts = [(ctx.int(0), ctx.int(1)), (ctx.int(1), ctx.int(0)), (ctx.int(2), ctx.int(4)), (ctx.int(3), ctx.int(2))];
Ex::poly_fit_points(&ctx, &pts, &x, 1).unwrap();                          // 7/10*x + 7/10   (exact least-squares line)
```

Also: `bisect`, `newton_root`, `golden_section`, `minimize_scalar` (→ `ScalarMinimum { x, value }`), `poly_fit_exact`, `linear_fit` (→ `LinearFit { slope, intercept }`), `trapezoid`, `eval_poly`; `RootOpts`/`MinimizeOpts`/`DeOpts` for tolerances, budgets and seeds.

### Code Generation: Rust, C99 and Compiled Closures

Symbolic expressions compile to optimized Rust or C functions with common subexpression elimination, `mul_add`/`fma`, integer powers as multiplications, optional domain assertions, and a self-contained special-function runtime.

```rust
# use symplex::prelude::*;
# use symplex::syms;
let ctx = Context::new();
syms!(ctx; x, y);
let f = &x.sin().powi(2) + &(&x * 2 + &y).exp() * 3;

f.to_rust_fn("f", &["x", "y"]).unwrap();
// pub fn f(x: f64, y: f64) -> f64 { 3_f64.mul_add(2_f64.mul_add(x, y).exp(), x.sin().powi(2)) }

f.to_c_fn("f", &["x", "y"]).unwrap();
// #include <math.h>
// double f(double x, double y) { return fma(3.0, exp(fma(2.0, x, y)), pow(sin(x), 2.0)); }

// Special functions embed only the helpers they need (Rust: `mod symplex_rt`; C: `static inline`)
x.lambertw().to_c_fn("w0", &["x"]).unwrap();          // contains symplex_lambert_w0

let opts = CodegenOptions { precision: Precision::F32, checked_domain: true, ..Default::default() };
x.ln().to_c_fn_with_options("g", &["x"], &opts).unwrap();   // float g(float x) { return assert(x > 0.0f), logf(x); }

// Compiled closures: Result, arity-checked, Send + Sync; gradients share one CSE pass
let cf = f.compile(&["x", "y"]).unwrap();
cf(&[0.5, 0.25]);
let grad = Ex::compile_many(&[&f.diff(&x), &f.diff(&y)], &["x", "y"]).unwrap();
grad.call_vec(&[0.5, 0.25]);

f.to_latex();                                          // \sin^{2}\left(x\right) + 3\exp\left(2x + y\right)

// 0.10: more targets and interchange formats
(x.sin().powi(2) + x.exp()).to_python().unwrap();      // math.sin(x)**2 + math.exp(x)   (executable; tested against eval_f64)
x.sin().to_numpy().unwrap();                           // numpy.sin(x)
(x.exp() + x.sin().powi(2)).to_julia().unwrap();       // sin(x)^2 + exp(x)
(&x.powi(2) + 1).to_mathml().unwrap();                 // <math xmlns=…><mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>1</mn></mrow></math>
(2 * &x + 1).to_srepr();                               // Add(Integer(1), Mul(Integer(2), Symbol('x')))
f.to_dot();                                            // Graphviz digraph of the expression tree

// and the other direction: relations, Boolean connectives, implicit multiplication
ctx.parse_bool("x > 0 and x < 1").unwrap().to_lean().unwrap();   // 0 < x ∧ x < 1
ctx.parse_implicit("2x + 3(x - 1)").unwrap();          // 5*x - 3
```

For `build.rs` pipelines and `no_std` targets see [`symplex-build`](symplex-build/README.md); for the browser see [`symplex-wasm`](symplex-wasm/README.md).

### Probability and Statistics

`symplex::stats` (0.11) is the counterpart of `sympy.stats`: a `RandomVariable` is a symbol with a `Distribution`, and mean, variance, moments, probabilities of events, density, CDF, moment generating function, quantile and entropy are computed *exactly* — closed-form moments for polynomial expectations, the family's CDF for probabilities, exact integration or summation over the support otherwise. Fifteen continuous families (Normal, Uniform, Exponential, Gamma, χ², Beta, Cauchy, Laplace, Logistic, LogNormal, Student t, `FDistribution`, Weibull, Pareto, Triangular), eight discrete ones (Bernoulli, Binomial, Poisson, Geometric, Negative Binomial, Hypergeometric, DiscreteUniform, and explicit finite tables — a die is a finite table), independence algebra over several variables, and seeded sampling.

```rust
# use symplex::prelude::*;
use symplex::stats::{self, Distribution, RandomVariable, Rng};
let ctx = Context::new();
let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
let y = RandomVariable::new(&ctx, "Y", Distribution::exponential(ctx.int(3)));
let b = RandomVariable::new(&ctx, "B", Distribution::binomial(ctx.int(5), ctx.rational(1, 3)));

x.expectation(&(x.symbol().powi(2) + 3 * x.symbol()));   // 1
y.probability(&y.symbol().gt(&ctx.int(1))).unwrap();     // exp(-3)
b.probability(&b.symbol().gt(&ctx.int(2))).unwrap();     // 17/81
(y.skewness(), b.kurtosis());                            // (2, 27/10)
x.cdf(&ctx.symbol("t"));                                 // 1/2*erf(1/2*t*sqrt(2)) + 1/2
y.quantile(&ctx.symbol("p")).unwrap();                   // -1/3*ln(-p + 1)
stats::conditional_expectation(&x, x.symbol(), &x.symbol().gt(&ctx.int(0))).unwrap();   // sqrt(2)*pi^(-1/2)
stats::covariance(&[&x], x.symbol(), &(2 * x.symbol())).unwrap();                        // 2
let z = RandomVariable::new(&ctx, "Z", Distribution::normal(ctx.int(1), ctx.int(2)));
stats::sum_distribution(&x, &z).unwrap();                // Normal(1, sqrt(5))
x.entropy();                                             // 1/2*ln(2*pi*E)
let coin = Distribution::try_finite(&ctx, vec![(ctx.int(1), ctx.rational(2, 3)), (ctx.int(0), ctx.rational(1, 3))]).unwrap();
RandomVariable::new(&ctx, "C", coin).variance();         // 2/9
y.sample(20_000, &mut Rng::new(1)).unwrap();             // reproducible f64 samples; mean ≈ 0.33
```

Also: `std`, `moment(n)`, `central_moment`, `median`, `mgf`, `characteristic_function`, `density`, `support`; `stats::{expectation, variance, correlation, probability}` over several independent variables (rectangles and `X < Y`), `conditional_probability`; every constructor has a `try_` twin validating numeric parameters. All test values come from SymPy 1.14.

Beside the random variables, `stats` has a data layer that works on observed samples (`Q` = exact rationals, or `f64` where the reference library is numeric), one module per question:

- `data` — descriptive statistics and measures of association (Pearson, Spearman, Kendall's τ, Goodman–Kruskal's γ, Somers' D), exactly.
- `estimation` — maximum likelihood, method of moments, conjugate Bayesian updating; confidence and credible intervals (`proportion_interval*`, `confidence_interval_mean*`).
- `hypothesis` — t/z/χ²/exact/rank tests, effect sizes, multiple-comparison corrections, resampling, power and sample size.
- `anova` — one-way, factorial and repeated-measures ANOVA with post-hoc comparisons.
- `agreement` — inter-rater agreement: Cohen's/Fleiss' κ, Scott's π, Krippendorff's α, Gwet's AC₁, ICC, Kendall's W, Cochran's Q.
- `reliability` — Cronbach's α and relatives, split-half, KR-20, item analysis.
- `aggregation` — majority/plurality/weighted votes, Dawid–Skene, Bradley–Terry, worker-quality helpers.
- `regression` — exact OLS/WLS over ℚ and logistic regression in `f64`.
- `survival`, `cox` — Kaplan–Meier, log-rank and Cox proportional hazards with right censoring.
- `markov` — finite discrete-time Markov chains with exact transition matrices.
- `information` — entropies, divergences and mutual information on finite distributions, exactly.
- `sequential` — Wald's SPRT.
- `multivariate` — the multivariate normal, covariance/correlation matrices, principal components.
- `order` — order statistics as a `Family` of their own.
- `numdist` — the `f64` reference distributions (`scipy.stats.<dist>.{cdf, sf, ppf}`) the tests are checked against.

The book chapters [Statistics](book/src/guide/statistics.md) and [Response analysis](book/src/guide/response-analysis.md) walk through them.

### Compile-Time Dimensional Analysis

Physical quantity types are checked at compile time. Adding a `Mass` to a `Length` is a compiler error. Differentiation respects dimensions: `d(Length)/d(Time)` produces `Velocity`.

```rust
# use symplex::prelude::*;
# use symplex::syms;
use symplex::units::*;

let ctx = Context::new();
let m = Mass::symbol(&ctx, "m");
let a = Acceleration::symbol(&ctx, "a");

// dim! macro: the `: Force` annotation is a compile-time assertion
let force = dim!(ctx, Force: m * a);               // F = m·a [N]

// Typed calculus: d(Length)/d(Time) → Velocity
syms!(ctx; g, t);                                  // raw symbols for expr!
let t_var = Time::symbol(&ctx, "t");
let position = Length::from_ex(expr!(ctx, 1/2 * g * t^2));
let velocity: Velocity = position.diff_wrt(&t_var);   // g·t [m/s]

// 30 named quantity types, ~100 unit conversions (all exact rationals)
// Mass + Length → compile error
```

---

## Design Principles

1. **Exact by default.** Every number is `Ratio<BigInt>`. No floating-point contamination. `0.1 + 0.2 == 3/10`, not `0.30000000000000004`. Floats only appear on explicit `eval_f64()`, `compile()`, or `integrate_numeric()`. `Context::from_f64` converts a float to its exact dyadic rational; `from_f64_approx` to the nearest bounded-denominator rational.

2. **Explicit contexts.** Every expression belongs to a `Context`. No hidden global state. Mixing expressions from different contexts is caught immediately (compiler-enforced private field + runtime guard).

3. **Type-safe expressions.** `Ex` (numeric), `BoolEx` (boolean), `SetEx` (set-valued) are distinct types. `sin(bool_expr)` is a compile error.

4. **Thread-safe.** `Context` is `Clone` (Arc-based), `Ex` is `Send + Sync`. Multiple threads can share a context safely.

5. **No recursion.** All tree traversals use explicit stacks. Deep expressions don't blow the call stack.

6. **Never silently wrong.** Numerical evaluation returns `Result`. Operations that can't produce a closed form return unevaluated symbolic nodes — `∫x^x dx` returns `Integral(x^x, x)`, not garbage. `∫₋₁¹ dx/x²` is `Err(Divergent)`, not `−2`. `re(z)` stays `re(z)` unless `z` is known to be real. A certificate prover says `Unknown` with what it tried, never a wrong `Proved`.

7. **One domain: ℂ, principal branch.** A symbol without assumptions may be complex, every multivalued function takes its principal branch (`∛(−8) = 1 + √3·i`; the real root is `real_root`), and a rewrite is applied only where it preserves the value: `ln x + ln y` stays unless the arguments are known positive, `√(x²)` is `|x|` only for real `x`. The table of identities and their conditions, and the one exception (generated `f64` code takes real odd roots), is in [Key Concepts](book/src/getting-started/key-concepts.md#the-domain-model). `fuzz_simplify` checks it at real and complex points every night.

8. **No panics in library code.** Failure is a `Result`, absence an `Option`, invariants `debug_assert!`. The library never calls `unwrap`/`expect`/`panic!`/`unreachable!` on user data (ratchet `tests/unit/test_no_panics.rs`); the remaining `assert!`s on caller-supplied *shapes* (e.g. `Matrix::zeros(0, n)`, `Context::symbol("")`, a non-prime modulus to `legendre_symbol`) are documented under `# Panics` on each item and counted by the same ratchet — see CONTRIBUTING.md for the policy and why error plumbing costs nothing measurable.

9. **One crate, no knobs.** There are no Cargo features to combine; every capability is always present. Compile time is not the constraint, capability is.

10. **Named positions, not tuples.** Where two values share a type, the API says which is which: `Interval { lower, upper, kind }` rather than `(f64, f64)`, `Qr { q, r }` rather than `(Matrix, Matrix)`, `ExtendedGcd { gcd, x, y }` rather than `(BigInt, BigInt, BigInt)`, `DhLink { theta, d, a, alpha }`, `Bounds::at_least(0)` for an LP variable. Universal conventions stay tuples (`(x, y)` points, `(re, im)`, `(numer, denom)`, `(quotient, remainder)`, `shape() -> (rows, cols)`). Polynomial coefficient vectors are ascending (`c[i]` multiplies `x^i`) everywhere except `Poly::all_coeffs`, which is SymPy's highest-first by name. A ratchet test keeps new same-typed tuples off the public surface — see CONTRIBUTING.md, "Tuples versus structs".

---

## The API Model

Every symbolic operation that might not produce a closed-form result has two entry points:

| Intent | Method | Returns | When to use |
|--------|--------|---------|-------------|
| Give me math | `integrate(&x)` | `Ex` (always — may contain `Integral` nodes) | Interactive exploration, chaining |
| Fail if you can't | `try_integrate(&x)` | `Result<Ex>` | Pipelines, codegen, safety-critical |

`try_` twins exist for `diff`, `integrate`, `integrate_definite`, `limit`, `limit_left/right/dir`, `series`, `series_at_infinity`, `summation`, `product_over`, `laplace`, `inverse_laplace`, `residue`, `gosper_sum`, `solve_ode`, `solve_gt/ge/lt/le`, `char_poly`, `wronskian`, and every `stats::Distribution` constructor. Check any expression for unevaluated forms:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
# let hard_expr = x.pow(&x);   // ∫ x^x dx has no elementary antiderivative
let anti = hard_expr.integrate(&x);
if anti.has_unevaluated() {
    println!("integration produced formal result: {anti}");
}
```

(`RootOf` and `RootSum` are *not* unevaluated: they are complete algebraic answers.)

Operations that always succeed (`simplify`, `expand`, `eval`, `factor`, `subs`, `rewrite`) return `Ex` with no `try_` variant — "unchanged" is a valid answer.

**`Result` boundaries.** Crossing from symbols to numbers (`eval_f64`, `eval_decimal`, `compile`, `to_rust_fn`, `to_c_fn`, `integrate_numeric`) always returns `Result`. So do operations with structural preconditions (`Matrix::inv`, `cholesky`, `lu`, `minor`, `matmul`) and solvers whose failure is a mathematical fact: `solve` returns `Err(InfiniteSolutions)` for identities and `Err(NoSolution)` for contradictions, `try_integrate_definite` returns `Err(Divergent)`, `laplace_final_value` returns `Err(Divergent)` for unstable poles. Transform APIs without an unevaluated node (`fourier_transform`, `mellin_transform`, `z_transform`) are `Result`-only.

**Three-valued queries.** `is_positive`, `equals`, `is_convergent`, `SetEx::contains`, `is_subset`, `Matrix::is_symmetric`, `is_diagonalizable`, `is_positive_definite`, `BoolEx::is_tautology`, `vector::is_conservative`, `is_increasing`, `is_convex` … return `Option<bool>`: yes, no, or unknown. `degree`, `resultant`, `discriminant`, `hypergeometric_ratio`, `minimal_polynomial`, `periodicity` return `Option<T>`.

**Certificate outcomes.** The five inequality provers (`prove_nonnegative_on_box`, `prove_nonnegative_on_halfline`, `prove_nonnegative_on_reals`, `prove_nonnegative_on_polyhedron`, `prove_sos`) return `Outcome<C, U>`: `Proved(C)` (re-verified), `Refuted { point, value, .. }` (an exact point where the goal is negative) or `Unknown(U)` (what was tried, including a budget that ran out). `is_proved()`, `certificate()`, `refutation()`, `unknown()` and `map_certificate` are shared; `Refuted` and the `Unknown` payloads are `#[non_exhaustive]`.

**Budgets.** Long-running exact algorithms accept a deadline and/or pivot cap (`linprog::Budget`, `PolyhedronOpts::with_time_limit`, `SosOpts::with_time_limit`); running out is a *status*, not an error.

**Intervals.** `Interval<T>` (prelude) is the one interval type: two endpoints and an `IntervalKind` (`[a, b]`, `(a, b)`, `(a, b]`, `[a, b)`), with `contains`, `is_empty`, `intersect`, `hull`, `width`, `Display`, and `From<a..=b>`. Confidence and credible intervals are `Interval<f64>`; root isolation returns `(lo, hi]` cells beside exact `[r, r]` hits; `SetEx::as_intervals` and `stats::Support` pieces are `Interval<Ex>` (membership through the set layer, three-valued). `Bounds<T>` (`lower: Option<T>, upper: Option<T>`) is a closed constraint side that may be absent — LP variable bounds, bounding boxes. Oriented limits (`∫ₐᵇ`, `Σ`) stay separate `lower`, `upper` arguments because reversing them flips the sign.

---

## Comparison with SymPy

| Feature | symplex 0.23 | SymPy 1.14 |
|---------|--------------|------------|
| Arithmetic | Exact `Ratio<BigInt>` | Exact (similar) |
| Differentiation | Complete, incl. Bessel/Airy/orthogonal/polygamma/erf family | Complete |
| Indefinite integration | 15+ strategies incl. Risch + LRT log-to-real; results in `erf`/`erfi`/`Si`/`Shi`/`Ei`/… | Risch + heurisch + Meijer G (broader) |
| Definite / improper integration | Singularity detection, ~30-entry improper table, divergence reported as `Err` | Meijer G-based; much broader table |
| Numeric integration | Adaptive G7/K15 quadrature | via mpmath (more algorithms) |
| Summation | Faulhaber, Gosper, telescoping, binomial, p-series, power-series recognition | + Zeilberger, hypergeometric closed forms (broader) |
| Polynomial solving | Through quartic + `RootOf`; `real_roots` as ordered `RootOf`s | Through quartic + `CRootOf` |
| General solutions | `solve_general` (periodic families) | `solveset` with `ImageSet` |
| Linear systems | `linsolve` (unique / parametric / inconsistent, symbolic) | `linsolve` (similar) |
| Polynomial systems | Gröbner + FGLM, algebraic solutions; `Ex::groebner`/`reduce_modulo` | Gröbner, more strategies |
| Polynomial algebra | `Poly` with symbolic coefficients, `MultiPoly` over ℚ, resultants/discriminants (also with symbolic coefficients), `gcd_all`, `factor_mod` over GF(p), `minimal_polynomial` | `Poly` with domains, algebraic extensions, `primitive_element` (broader) |
| Function analysis | `singularities`, `stationary_points`, `maximum`/`minimum`, monotonicity/convexity (exact via Sturm), `periodicity`, `function_range` | `calculus.util` (similar) |
| Rational simplification | `ratsimp`/`cancel`: heuristic multivariate GCD | `cancel`, `ratsimp`, `together`, `apart` (similar) |
| Linear programming | Exact simplex on hybrid `i64`/`i128`/256-bit/`BigInt` cells; duals, Farkas certificates, budgets | `sympy.solvers.simplex` (exact; optimum and argmin only) |
| Polynomial inequalities | Handelman (boxes), Pólya (half-lines), parametric polyhedra with a goal multiplier, sums of squares (built-in SDP + exact rounding); all re-verified exactly; Lean export | — |
| Integer normal forms | Row/column HNF with transform, SNF with transforms, integer nullspace, lattice index, exact LLL | `hermite_normal_form`, `smith_normal_form` (no transforms), `lll` |
| Polytopes | Exact vertices (integer arithmetic, tight sets), `clip`, volume in any dimension, parametric families | — |
| Numerical optimisation | Brent, bisection, Newton, Nelder–Mead, differential evolution, QR least squares, exact rational fits | Defers to SciPy / mpmath; far broader via SciPy |
| Series expansion | Taylor / Laurent / at ∞ / formal power series with general terms | + `O()` notation, Puiseux |
| Limits | Gruntz with work budget, one-sided | Gruntz (more mature) |
| Simplification | Multi-strategy fixpoint + public rule engine with AC matching, tracing | More strategies; `replace`/`Wild` patterns |
| Factoring | Berlekamp–Zassenhaus (any degree), multivariate via Kronecker, GF(p) | Zassenhaus + Wang (faster multivariate), algebraic extensions |
| Matrices | Eigen/Jordan/exp/log/sqrt/pow, QR, Cholesky, LDL, LU, Hessenberg, singular values, condition number, rank-deficient pseudo-inverse, permanent, `RootOf` eigenvalues; exact `QMatrix`/`ZMatrix` (Bareiss) | SVD, Schur, sparse, matrix expressions (broader) |
| Statistics | 23 distribution families + finite tables; exact moments, probabilities, CDF/MGF/quantile/entropy; independence algebra, conditional expectation, sum closures; seeded sampling | `sympy.stats` (broader: joint/compound/stochastic processes, matrix distributions) |
| ODE solving | 16 classes, IVPs, systems | More classes, hints, series solutions |
| Recurrences | Linear constant-coefficient, first-order | `rsolve` (poly/rational/hyper) |
| Transforms | Laplace, Fourier (3 conventions), Mellin (with strip), Z, Fourier series; discrete: convolutions, NTT, Walsh–Hadamard, Möbius | Broader tables, Hankel, cosine/sine; `discrete` (+ float FFT) |
| Sets & logic | Interval algebra, three-valued queries, NNF/CNF/DNF, DPLL; `parse_bool` | Richer set types (`ImageSet`, `ConditionSet`), `satisfiable` models |
| Number theory | rho/ECM, BPSW, `sqrt_mod`, `nthroot_mod` (any modulus), `polynomial_congruence`, dlog, CRT, Pell, two/four squares, continued fractions, Carmichael/amicable | Broader (quadratic forms, general Diophantine) |
| Combinatorics | Stirling, Bell, partitions, derangements, multinomial | Broader (permutation groups, etc.) |
| Special functions | Γ family (incl. incomplete), ψ⁽ⁿ⁾, erf/erfi/erfinv, B, W, Bessel, Airy, elliptic K/E/F/Π, `expint`, Si/Ci/Shi/Chi/Ei/li, Fresnel, polylog/η/ζ, classical + associated orthogonal polynomials — all arbitrary precision | Many more (hypergeometric, Meijer G, Mathieu, …) |
| Algebraic numbers | `ℚ(α)` field with exact zero/sign testing; `minimal_polynomial` | `AlgebraicNumber` + `ANP`, `primitive_element` |
| Code generation | Rust, C99, Python, NumPy, Julia with CSE and an embedded special-function runtime (Rust/C); compiled closures | Python / C / Fortran / Rust / Julia / Octave / JS via `codegen` |
| Interchange | LaTeX, Unicode pretty-print, Presentation MathML, `srepr`, Graphviz DOT, JSON tree, Lean 4 | LaTeX, MathML, `srepr`, `dotprint`, pretty |
| Parsing | Numeric expressions, relations and Boolean connectives, implicit multiplication | `sympify`/`parse_expr` (+ LaTeX, Mathematica) |
| Dimensional analysis | Compile-time type checking | Runtime `physics.units` |
| Thread safety | `Send + Sync`, no GIL | GIL-bound |
| Expression type safety | `Ex` / `BoolEx` / `SetEx` at compile time | Runtime only |
| Failure model | No panics (ratchet-enforced); `Result`/`Option`/unevaluated forms; budgets are statuses | Exceptions |
| Language | Rust (compiled, ~245K lines, 92 node types) | Python (interpreted) |

**Where SymPy is stronger:** geometry, tensor algebra, quantum mechanics, general Diophantine equations, PDE solving, hypergeometric/Meijer-G machinery, stochastic processes and joint distributions beyond independence, and 30 years of community contributions and testing.

**Where symplex is different:** exact certificates as first-class results (LP duals and Farkas vectors, Handelman/Pólya/SOS proofs re-verified with exact arithmetic, Sturm-verified polynomial signs, unimodular HNF/SNF transforms) and their export to Lean 4 / Mathlib as theorems that compile; compile-time dimensional analysis; thread safety; Rust *and* C code generation with an embedded runtime; algebraic number field arithmetic with exact zero/sign testing; a library verified not to panic. Operations that can't complete return honest unevaluated forms, `Err`, or `Unknown` rather than guessing.

---

## Migrating

Every breaking change has a one-line fix in the book:

| From → to | Page | The gist |
|---|---|---|
| 0.12 → 0.22 | [CHANGELOG](CHANGELOG.md) | breaking changes listed per release under `### Breaking`; the transitional `stats` re-exports (`aggregation::proportion_interval*`, `hypothesis::{anova_one_way, confidence_interval_mean}`, `reliability::{kappa*, somers_d}`) were removed in 0.22 in favour of `stats::{estimation, anova, agreement, data}` |
| 0.11 → 0.12 | [migrating-0.12](book/src/reference/migrating-0.12.md) | `Distribution` is a struct wrapping a `Family` trait (`downcast_ref::<Normal>()` instead of matching enum variants), `Support` is a typed region (`Support::interval`/`integers`/`points`), `Distribution::try_finite` takes `&ctx` |
| 0.6 → 0.7 | [migrating-0.7](book/src/reference/migrating-0.7.md) | one `Outcome<C, U>` for every prover (patterns need `..`; `Unknown(u)`), the Handelman struct is `BoxCertificate` and `Certificate` is the trait, `LeanOpts`/`PolyhedronOpts`/`SosOpts` are `#[non_exhaustive]` (builders instead of literals), `control`/`robotics`/`dynamics` shape failures return `Result` |
| 0.9 → 0.10 | [CHANGELOG](CHANGELOG.md) | `LpStatus::BudgetExhausted`, `Tactic::Apply`, `Decl.preamble` (use `Decl::new` + builders) |
| 0.3 → 0.4 | [migrating-0.4](book/src/reference/migrating-0.4.md) | `roots_count_real` removed, `LeanOpts` gained a field |
| 0.1 → 0.2 | [migrating-0.2](book/src/reference/migrating-0.2.md) | `Result` boundaries, three-valued queries, `integrate_definite`, eigen family without dummy variables |

0.2 → 0.3, 0.4 → 0.6, 0.7 → 0.9 and 0.10 → 0.11 were additive (`cargo semver-checks` clean).

---

## Examples

Every example is self-contained and runs in a few seconds; CI runs all of them.

**Getting started:**
```sh
cargo run --example quickstart              # Tour of core operations
cargo run --example repl                    # Interactive expression evaluation
cargo run --example readme_snippets         # Every code block in this README, executed
```

**Certificates, Lean and exact geometry (0.3–0.10):**
```sh
cargo run --example certificates_to_lean    # Handelman / half-line / SOS certificates exported as Mathlib theorems (writes a .lean file)
cargo run --example polyhedron_certificates # Parametric polyhedra: λ(j) goal multiplier, staged exact LP, emptiness, lean_steps
cargo run --example exact_matrices          # QMatrix / ZMatrix: fraction-free elimination, rank, nullspace, Smith form
cargo run --example polynomials             # Poly views with symbolic coefficients, ratsimp, linear certificates, exact sign on an interval
cargo run --example exact_lp                # Exact simplex: optima, shadow prices, Farkas certificates, feasible_nonneg, linprog_matrix
cargo run --example integer_lattices        # Row/column HNF with transforms, Smith normal form, integer nullspace, unimodularity, lattice index
cargo run --example numeric_optimization    # Brent/Newton roots, Nelder–Mead, differential evolution, polynomial fits (f64 and exact)
```

**New in 0.2:**
```sh
cargo run --example definite_integration    # Improper integrals, divergence detection, quadrature, residues
cargo run --example summation_and_series    # Σ/Π closed forms, convergence, formal power series, finite differences
cargo run --example complex_analysis        # re/im/conjugate/arg, new constants, Si/Ci/Ei/ζ/polygamma
cargo run --example rule_engine             # Custom rewrite rules, tracing, subs_algebraic, targeted simplifiers
cargo run --example linear_systems_and_ivp  # solve semantics, linsolve, solve_general, ODE IVPs, rsolve
cargo run --example sets_and_logic          # Interval algebra, reduce_inequalities, CNF/DNF, tautology
cargo run --example factoring_and_ntheory   # Zassenhaus factoring, factorint, sqrt_mod, Pell, continued fractions
cargo run --example transforms              # Fourier, Mellin, Laplace, Fourier series, Z, one-sided limits
cargo run --example matrix_decompositions   # QR, Cholesky, LDL, Jordan, matrix_exp_t, RootOf eigenvalues
cargo run --example c_codegen               # C99 backend, embedded runtime, compile_many (compiles the C if cc exists)
```

**Engineering workflows:**
```sh
cargo run --example pid_controller          # PID design → stability → Rust codegen
cargo run --example robotics_codegen        # DH parameters → Jacobian → optimized Rust
cargo run --example control_system          # State-space, transfer functions, pole placement, ZOH
cargo run --example signal_filter           # Bilinear transform → digital filter → codegen
cargo run --example dynamics                # Lagrangian mechanics, equations of motion
cargo run --example inverse_kinematics      # 2-DOF IK via Gröbner bases
```

**Mathematics and science:**
```sh
cargo run --example calculus                # Differentiation, integration, limits, series
cargo run --example equation_solving        # Polynomial, transcendental, system solving
cargo run --example matrix_algebra          # Eigenvalues, Jordan form, codegen
cargo run --example ode_solving             # ODE classification and solving
cargo run --example combinatorics_counting  # Stirling numbers, partitions, multinomials
cargo run --example optimization            # Gradient, Hessian, critical points
cargo run --example complex_numbers         # Euler's formula, complex roots
cargo run --example laplace_transforms      # Forward, inverse, z-transforms
```

**Applied problems:**
```sh
cargo run --example gradient_descent        # Symbolic gradient → compiled optimization loop
cargo run --example crypto_rsa              # RSA with number theory primitives
cargo run --example number_theory           # Primality, factorization, CRT
```

**Dimensional analysis:**
```sh
cargo run --example units_physics           # Compile-time unit checking
cargo run --example units_electrical        # Circuit analysis with units
cargo run --example units_engineering       # Motor design, imperial conversions
cargo run --example units_kinematics        # Kinematics with typed quantities
cargo run --example units_lagrangian        # Lagrangian mechanics with units
```

**Output:**
```sh
cargo run --example latex_output            # LaTeX rendering
cargo run --example physics_constants       # Physical constants (symbolic + exact)
```

---

## Dependencies

All MIT or Apache-2.0 licensed. No C bindings. No LGPL.

Core: `num-bigint`, `num-rational`, `num-complex`, `num-traits`, `num-integer`, `smallvec`, `rustc-hash`, `bitflags`, `parking_lot`, `thiserror`, `astro-float`, `serde`, `serde_json`, `tracing`, `typenum`.

Proc macros: `syn`, `quote`, `proc-macro2`.

Companion crates: [`symplex-build`](symplex-build/README.md) (build-time codegen for `no_std` firmware), [`symplex-wasm`](symplex-wasm/README.md) (browser bindings).

## Requirements

Rust 1.93+ (Edition 2024). No Cargo features by design; pure Rust on every platform Rust targets, including `wasm32-unknown-unknown`. ~12,900 tests (`cargo nextest run`), ~1,290 doctests — every Rust block in this README and in the book is compiled and run as a doctest too (`cargo test --doc -- doctests::`); every emitted Lean shape is pinned to text compiled against Mathlib (Lean 4.30).

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE).

symplex is an independent project, not affiliated with or endorsed by SymPy.
SymPy, mpmath, SciPy and statsmodels (all BSD-3-Clause) are run as test
oracles; the few modules that follow SymPy's implementation of a published
algorithm, and SymPy's licence notice, are listed in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
