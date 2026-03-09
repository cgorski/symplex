# Chapter 18: Formal Power Series

A formal power series (FPS) represents a function as an infinite sum $\sum_{k=0}^{\infty} a_k x^k$ — but unlike a Taylor series, the emphasis is on the **coefficient sequence** rather than convergence. Symplex's FPS module gives you closed-form formulas for coefficients of known functions, lazy coefficient extraction, and truncation to finite polynomials.

## FPS vs. Maclaurin Series

You might wonder: doesn't symplex already have `.maclaurin()`? Yes, but the two tools solve different problems:

- **`.maclaurin(&x, n)`** computes a truncated polynomial of degree $n$. You get a concrete expression, but no information about the pattern of coefficients.
- **`.fps_maclaurin(&x)`** returns a `FormalPowerSeries` object that knows the **formula** for the $k$-th coefficient. You can ask for any coefficient without recomputing the entire series.

Here's the difference in practice:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

// Maclaurin: gives you a polynomial
let poly = x.exp().maclaurin(&x, 5);
println!("Maclaurin: {}", poly.expand().eval());
// 1 + x + x^2/2 + x^3/6 + x^4/24

// FPS: gives you a coefficient oracle
let fps = x.exp().fps_maclaurin(&x);
println!("Has closed form: {}", fps.has_closed_form());
// Has closed form: true

// Ask for any coefficient — no need to recompute from scratch
println!("a_0  = {}", fps.coefficient_rational(0));   // 1
println!("a_5  = {}", fps.coefficient_rational(5));   // 1/120
println!("a_10 = {}", fps.coefficient_rational(10));  // 1/3628800
println!("a_20 = {}", fps.coefficient_rational(20));  // 1/2432902008176640000
```

The `.coefficient_rational(k)` method returns the exact rational coefficient as a `Ratio<BigInt>` — no floating-point, no overflow, no matter how large $k$ is.

## Constructing a Formal Power Series

### About Zero (Maclaurin)

Use `.fps_maclaurin(&var)` for expansion about $x = 0$:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let fps_sin = x.sin().fps_maclaurin(&x);
let fps_cos = x.cos().fps_maclaurin(&x);
let fps_exp = x.exp().fps_maclaurin(&x);

println!("All have closed-form coefficient formulas:");
println!("  sin(x): {}", fps_sin.has_closed_form());  // true
println!("  cos(x): {}", fps_cos.has_closed_form());  // true
println!("  exp(x): {}", fps_exp.has_closed_form());  // true
```

### About an Arbitrary Point

Use `.fps(&var, &point)` for expansion about any point:

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.symbol("x");
let zero = ctx.int(0);

let fps = x.exp().fps(&x, &zero);
assert!(fps.has_closed_form());
```

Currently, closed-form recognition works best for Maclaurin series (expansion about 0). For non-zero expansion points, symplex falls back to truncated Taylor coefficients.

## Known Function Patterns

Symplex recognizes the following elementary functions and assigns closed-form coefficient formulas:

### Exponential

$$e^x = \sum_{k=0}^{\infty} \frac{1}{k!} x^k$$

```rust
vars!(x);
let fps = x.exp().fps_maclaurin(&x);
assert!(fps.has_closed_form());
// a_k = 1/k!
println!("a_0 = {}", fps.coefficient_rational(0));  // 1
println!("a_1 = {}", fps.coefficient_rational(1));  // 1
println!("a_2 = {}", fps.coefficient_rational(2));  // 1/2
println!("a_3 = {}", fps.coefficient_rational(3));  // 1/6
println!("a_4 = {}", fps.coefficient_rational(4));  // 1/24
```

### Sine and Cosine

$$\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!} x^{2n+1}$$

$$\cos(x) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n)!} x^{2n}$$

Only odd (resp. even) coefficients are nonzero — the FPS object encodes this symmetry:

```rust
vars!(x);

let fps_sin = x.sin().fps_maclaurin(&x);
// Even coefficients are zero
println!("a_0 = {}", fps_sin.coefficient_rational(0));  // 0
println!("a_2 = {}", fps_sin.coefficient_rational(2));  // 0
println!("a_4 = {}", fps_sin.coefficient_rational(4));  // 0
// Odd coefficients: alternating sign / factorial
println!("a_1 = {}", fps_sin.coefficient_rational(1));  // 1
println!("a_3 = {}", fps_sin.coefficient_rational(3));  // -1/6
println!("a_5 = {}", fps_sin.coefficient_rational(5));  // 1/120
println!("a_7 = {}", fps_sin.coefficient_rational(7));  // -1/5040

let fps_cos = x.cos().fps_maclaurin(&x);
// Odd coefficients are zero
println!("a_1 = {}", fps_cos.coefficient_rational(1));  // 0
println!("a_3 = {}", fps_cos.coefficient_rational(3));  // 0
// Even coefficients: alternating sign / factorial
println!("a_0 = {}", fps_cos.coefficient_rational(0));  // 1
println!("a_2 = {}", fps_cos.coefficient_rational(2));  // -1/2
println!("a_4 = {}", fps_cos.coefficient_rational(4));  // 1/24
println!("a_6 = {}", fps_cos.coefficient_rational(6));  // -1/720
```

### Hyperbolic Sine and Cosine

Same structure as sin/cos but without the alternating sign:

$$\sinh(x) = \sum_{n=0}^{\infty} \frac{1}{(2n+1)!} x^{2n+1}$$
$$\cosh(x) = \sum_{n=0}^{\infty} \frac{1}{(2n)!} x^{2n}$$

```rust
vars!(x);

let fps_sinh = x.sinh().fps_maclaurin(&x);
assert!(fps_sinh.has_closed_form());
println!("a_1 = {}", fps_sinh.coefficient_rational(1));  // 1
println!("a_3 = {}", fps_sinh.coefficient_rational(3));  // 1/6  (positive!)
println!("a_5 = {}", fps_sinh.coefficient_rational(5));  // 1/120

let fps_cosh = x.cosh().fps_maclaurin(&x);
assert!(fps_cosh.has_closed_form());
println!("a_0 = {}", fps_cosh.coefficient_rational(0));  // 1
println!("a_2 = {}", fps_cosh.coefficient_rational(2));  // 1/2  (positive!)
println!("a_4 = {}", fps_cosh.coefficient_rational(4));  // 1/24
```

### Logarithm: ln(1 + x)

$$\ln(1+x) = \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k} x^k$$

Note: the series starts at $k = 1$ (since $\ln(1) = 0$):

```rust
let ctx = Context::new();
let x = ctx.symbol("x");
let one = ctx.int(1);
let zero = ctx.int(0);

let fps = (&one + &x).ln().fps(&x, &zero);
assert!(fps.has_closed_form());
println!("a_0 = {}", fps.coefficient_rational(0));  // 0
println!("a_1 = {}", fps.coefficient_rational(1));  // 1
println!("a_2 = {}", fps.coefficient_rational(2));  // -1/2
println!("a_3 = {}", fps.coefficient_rational(3));  // 1/3
println!("a_4 = {}", fps.coefficient_rational(4));  // -1/4
```

### Geometric Series: 1/(1 − x)

$$\frac{1}{1-x} = \sum_{k=0}^{\infty} x^k$$

Every coefficient is 1:

```rust
let ctx = Context::new();
let x = ctx.symbol("x");
let one = ctx.int(1);
let zero = ctx.int(0);

let expr = (&one - &x).powi(-1);
let fps = expr.fps(&x, &zero);
for k in 0..10 {
    println!("a_{k} = {}", fps.coefficient_rational(k));
    // All print: 1
}
```

### Arctangent

$$\arctan(x) = \sum_{n=0}^{\infty} \frac{(-1)^n}{2n+1} x^{2n+1}$$

```rust
vars!(x);
let fps = x.atan().fps_maclaurin(&x);
assert!(fps.has_closed_form());
println!("a_1 = {}", fps.coefficient_rational(1));  // 1
println!("a_3 = {}", fps.coefficient_rational(3));  // -1/3
println!("a_5 = {}", fps.coefficient_rational(5));  // 1/5
println!("a_7 = {}", fps.coefficient_rational(7));  // -1/7
// Even coefficients are all zero
println!("a_2 = {}", fps.coefficient_rational(2));  // 0
```

### Binomial Series: (1 + x)^α

$$( 1+x)^\alpha = \sum_{k=0}^{\infty} \binom{\alpha}{k} x^k$$

where the generalized binomial coefficient is $\binom{\alpha}{k} = \frac{\alpha(\alpha-1)\cdots(\alpha-k+1)}{k!}$.

```rust
let ctx = Context::new();
let x = ctx.symbol("x");
let one = ctx.int(1);
let zero = ctx.int(0);
let half = ctx.rational(1, 2);

// √(1+x) = (1+x)^(1/2)
let fps = (&one + &x).pow(&half).fps(&x, &zero);
assert!(fps.has_closed_form());
println!("a_0 = {}", fps.coefficient_rational(0));  // 1
println!("a_1 = {}", fps.coefficient_rational(1));  // 1/2
println!("a_2 = {}", fps.coefficient_rational(2));  // -1/8
println!("a_3 = {}", fps.coefficient_rational(3));  // 1/16
```

## Truncation: FPS to Polynomial

When you need a concrete polynomial (for plotting, numerical evaluation, or further symbolic manipulation), truncate the FPS:

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.symbol("x");
let zero = ctx.int(0);

let fps = x.exp().fps(&x, &zero);

// Truncate to 5 terms: 1 + x + x²/2 + x³/6 + x⁴/24
let poly = ctx.with_arena_mut(|arena| {
    let p = fps.truncate(arena, 5);
    ctx.wrap(p)
});
println!("exp(x) ≈ {}", poly.expand().eval());
```

The `truncate` method requires arena access via `ctx.with_arena_mut()` because it builds new expression nodes. The result is a standard `Ex` expression that you can simplify, differentiate, evaluate, or pass to code generation.

## Non-Elementary Functions: Truncated Fallback

For expressions that don't match a single known pattern, symplex falls back to computing Taylor coefficients numerically (via repeated differentiation) and stores them in a truncated representation:

```rust
vars!(x);

// exp(x) + sin(x) doesn't match a single known function
let fps = (&x.exp() + &x.sin()).fps_maclaurin(&x);

// No closed-form formula for the combined coefficients
println!("Closed form: {}", fps.has_closed_form());
// Closed form: false

// But coefficients are still available (from Taylor expansion)
println!("a_0 = {}", fps.coefficient_rational(0));  // 1 (exp(0) + sin(0))
println!("a_1 = {}", fps.coefficient_rational(1));  // 2 (1 + 1)
println!("a_2 = {}", fps.coefficient_rational(2));  // 1/2
println!("a_3 = {}", fps.coefficient_rational(3));  // 1/6 + (-1/6) = 0?
```

The truncated representation stores a finite number of coefficients. Requesting a coefficient beyond the stored range returns zero, so be aware of the truncation boundary.

## Aspirational Features

The following capabilities are planned but not yet implemented:

```rust
// PLANNED: Hypergeometric algorithm (Koepf)
// Automatically find closed-form coefficient formulas for a wider
// class of functions using the hypergeometric algorithm.
// let fps = expr.fps_hypergeometric(&x);

// PLANNED: FPS arithmetic
// Addition, multiplication (Cauchy product), and composition
// of formal power series, with lazy coefficient computation.
// let product = fps_sin.mul(&fps_cos);  // Cauchy product
// let composed = fps_exp.compose(&fps_sin);  // exp(sin(x))

// PLANNED: Recurrence relation representation
// Some series are best described by a recurrence: a_{k+1} = f(a_k, k).
// let fps = FormalPowerSeries::from_recurrence(|a_k, k| a_k / (k + 1));

// PLANNED: D-finite (holonomic) function recognition
// Recognize functions satisfying a linear ODE with polynomial coefficients,
// and use the ODE to generate coefficients efficiently.
// let fps = expr.fps_holonomic(&x);
// assert!(fps.is_d_finite());
```

## Summary

| Operation | API | Notes |
|-----------|-----|-------|
| FPS about 0 | `expr.fps_maclaurin(&x)` | Shorthand for `fps(&x, &zero)` |
| FPS about a point | `expr.fps(&x, &point)` | General expansion point |
| $k$-th coefficient (rational) | `fps.coefficient_rational(k)` | Exact `Ratio<BigInt>` |
| Closed-form check | `fps.has_closed_form()` | `true` for known functions |
| Truncate to polynomial | `fps.truncate(arena, n)` | Needs arena access |

### Known Closed-Form Patterns

| Function | Coefficient $a_k$ | Symmetry |
|----------|-------------------|----------|
| $e^x$ | $1/k!$ | All terms |
| $\sin(x)$ | $(-1)^n/(2n+1)!$ | Odd only |
| $\cos(x)$ | $(-1)^n/(2n)!$ | Even only |
| $\sinh(x)$ | $1/(2n+1)!$ | Odd only |
| $\cosh(x)$ | $1/(2n)!$ | Even only |
| $\ln(1+x)$ | $(-1)^{k+1}/k$ | Starts at $k=1$ |
| $1/(1-x)$ | $1$ | All terms |
| $\arctan(x)$ | $(-1)^n/(2n+1)$ | Odd only |
| $(1+x)^\alpha$ | $\binom{\alpha}{k}$ | All terms |

## What's Next

For computing with the polynomials that result from FPS truncation — factoring, root finding, Gröbner bases — revisit [Chapter 6: Solving Equations](06-solving.md). For generating numerical code from truncated series, see [Chapter 8: Code Generation](08-code-generation.md).

---

*[← Chapter 17: Vector Calculus](17-vector-calculus.md) | [Back to Table of Contents](index.md) | [Chapter 19: Optimization →](19-optimization.md)*