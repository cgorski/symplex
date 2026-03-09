# Chapter 4: Simplification

Simplification is the heart of any computer algebra system. Symplex provides a layered toolkit: from broad "just make it simpler" functions to targeted transformations that do exactly one thing. This chapter covers all of them.

## The Big Picture

Symplex expressions are **canonicalized** on construction — like terms are collected, constants are folded, and sums/products are sorted. But canonicalization doesn't apply mathematical identities or factor polynomials. That's what the simplification functions are for.

Here's the hierarchy, from broadest to most specific:

| Level | Function | What it does |
|-------|----------|--------------|
| Broadest | `full_simplify()` | Iterates eval + expand + simplify to fixpoint |
| Broad | `simplify()` | Single pass of rewrite rules |
| Targeted | `expand()`, `factor()`, `simplify_trig()`, etc. | One specific transformation |
| Lowest | `eval()` | Only evaluates known special values |

**Rule of thumb:** Use the most specific function you can. If you know you need to expand a product, call `.expand()`. If you know you have a trig identity, call `.simplify_trig()`. Reserve `simplify()` and `full_simplify()` for exploratory work or when you genuinely don't know what transformation is needed.

## `eval()` — Exact Special-Value Evaluation

`.eval()` replaces function applications with their exact values when the arguments are known constants:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let pi = ctx.pi();

// Known special values
println!("{}", pi.sin().eval());          // 0
println!("{}", pi.cos().eval());          // -1
println!("{}", ctx.int(0).exp().eval()); // 1
println!("{}", ctx.int(1).ln().eval());  // 0
println!("{}", ctx.int(4).sqrt().eval()); // 2

// Not a known special value — returned unchanged
println!("{}", x.sin().eval());           // sin(x)
```

`.eval()` is the most conservative transformation. It never rearranges terms, never expands products, and never applies identities. It only replaces what it knows for certain.

## `simplify()` — Single-Pass Rewrite Rules

`.simplify()` applies a set of built-in rewrite rules in one bottom-up pass:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Pythagorean identity
let expr = expr!(sin(x)^2 + cos(x)^2);
println!("{}", expr.simplify()); // 1

// exp/ln inverse
let expr = x.ln().exp();
println!("{}", expr.simplify()); // x

let expr = x.exp().ln();
println!("{}", expr.simplify()); // x
```

Because it's a single pass, it can miss simplifications that require multiple rounds of expansion and rule application. For example:

```rust
let ctx = Context::new();
syms!(ctx; x);

let expr = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
println!("{}", expr.simplify());      // might not reduce to 1
println!("{}", expr.full_simplify()); // 1
```

### Simplify with Trace

Want to see which rules fired? Use `.simplify_trace()`:

```rust
let ctx = Context::new();
syms!(ctx; x);

let expr = expr!(sin(x)^2 + cos(x)^2);
let (result, steps) = expr.simplify_trace();
println!("Result: {result}");
for step in &steps {
    println!("  Rule: {step:?}");
}
```

## `full_simplify()` — Fixpoint Simplification

`.full_simplify()` repeatedly applies `.eval()`, `.expand()`, and `.simplify()` until the expression stops changing (or 10 iterations, whichever comes first):

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Needs multiple passes to fully reduce
let expr = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
println!("{}", expr.full_simplify()); // 1
```

This is the "just make it simpler" button. It's slower than a single `.simplify()` call but more thorough.

There's also `.full_simplify_trace()` for debugging.

## `expand()` — Algebraic Expansion

`.expand()` distributes multiplication over addition and expands integer powers of sums:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

// Distribute
let expr = &x * &(&y + 1);
println!("{}", expr.expand()); // x*y + x

// Expand powers
let expr = (&x + &y).powi(2);
println!("{}", expr.expand()); // x^2 + 2*x*y + y^2

let expr = (&x + 1).powi(3);
println!("{}", expr.expand()); // x^3 + 3*x^2 + 3*x + 1
```

`.expand()` does *not* evaluate functions or apply trig identities. It's purely algebraic distribution.

## `factor()` — Polynomial Factoring

`.factor(&var)` factors a polynomial expression into a product of linear factors (when possible):

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Difference of squares
let expr = expr!(x^2 - 1);
println!("{}", expr.factor(&x)); // (x - 1)*(x + 1)

// Quadratic with integer roots
let expr = expr!(x^2 - 5*x + 6);
println!("{}", expr.factor(&x)); // (x - 2)*(x - 3)

// Perfect square
let expr = expr!(x^2 + 2*x + 1);
println!("{}", expr.factor(&x)); // (x + 1)^2

// Higher degree
let expr = expr!(x^4 - 1);
println!("{}", expr.factor(&x)); // (x - 1)*(x + 1)*(x^2 + 1)
```

The factoring algorithm finds rational roots via the polynomial solver and extracts them one at a time. It handles multiplicities and GCD of coefficients.

If no rational roots exist, the expression is returned unchanged:

```rust
let ctx = Context::new();
syms!(ctx; x);
let expr = expr!(x^2 + 1); // roots are ±i, not rational
println!("{}", expr.factor(&x)); // x^2 + 1 (unchanged)
```

### `factor_terms()` — Factor Out GCD of Coefficients

For factoring out common numeric factors from a sum:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let expr = expr!(6*x^2 + 4*x + 2);
let factored = expr.factor_terms();
println!("{factored}"); // 2*(3*x^2 + 2*x + 1)
```

## Trigonometric Simplification

### `simplify_trig()` — Trig Identity Simplification

Applies trigonometric identities (Pythagorean, double-angle, etc.):

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Pythagorean identity
let expr = expr!(sin(x)^2 + cos(x)^2);
println!("{}", expr.simplify_trig()); // 1

// Works with more complex expressions too
let expr = &x.sin().powi(2) + &x.cos().powi(2) + &x;
println!("{}", expr.simplify_trig()); // x + 1
```

### `expand_trig()` — Trig Expansion

Expands compound trig expressions using double-angle, sum-to-product, and similar identities:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Double angle → expanded form
let expr = (&x * 2).sin();
let expanded = expr.expand_trig();
println!("{expanded}"); // 2*sin(x)*cos(x)

let expr = (&x * 2).cos();
let expanded = expr.expand_trig();
println!("{expanded}"); // 2*cos(x)^2 - 1 (or equivalent form)
```

### `trig_combine()` — Combine Trig Products

The inverse of `expand_trig()` — combines products of trig functions into single terms:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let expr = &x.sin() * &x.cos();
let combined = expr.trig_combine();
println!("{combined}"); // 1/2*sin(2*x) (or equivalent)
```

### `rewrite_as_exp()` and `rewrite_as_trig()`

Convert between trigonometric and exponential forms:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Trig → exponential (Euler's formula)
let expr = x.sin();
let as_exp = expr.rewrite_as_exp();
println!("{as_exp}"); // expression in terms of exp(I*x)

// Exponential → trig
let expr = (&ctx.i_unit() * &x).exp();
let as_trig = expr.rewrite_as_trig();
println!("{as_trig}"); // cos(x) + I*sin(x)
```

## Logarithmic Simplification

### `expand_log()` — Log Expansion

Expands logarithms of products and powers:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

// ln(x*y) → ln(x) + ln(y)
let expr = (&x * &y).ln();
let expanded = expr.expand_log();
println!("{expanded}"); // ln(x) + ln(y)

// ln(x^2) → 2*ln(x)
let expr = x.powi(2).ln();
let expanded = expr.expand_log();
println!("{expanded}"); // 2*ln(x)
```

### `log_combine()` — Log Combination

The inverse of `expand_log()` — combines sums of logarithms:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

// ln(x) + ln(y) → ln(x*y)
let expr = &x.ln() + &y.ln();
let combined = expr.log_combine();
println!("{combined}"); // ln(x*y)
```

## Power Simplification

### `simplify_powers()`

Simplifies expressions involving powers — combines like bases, simplifies nested powers:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// (x^2)^3 → x^6
let expr = x.powi(2).powi(3);
let simplified = expr.simplify_powers();
println!("{simplified}"); // x^6
```

### `rationalize_denom()`

Eliminates radicals from the denominator:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// 1/sqrt(x) → sqrt(x)/x
let expr = 1 / &x.sqrt();
let rationalized = expr.rationalize_denom();
println!("{rationalized}");
```

## Rational Expression Simplification

### `simplify_rational()` — Cancel Common Factors

Cancels common polynomial factors between numerator and denominator:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// (x^2 - 1) / (x - 1) → x + 1
let expr = &expr!(x^2 - 1) / &expr!(x - 1);
let simplified = expr.simplify_rational();
println!("{simplified}"); // x + 1
```

### `partial_fractions()` — Partial Fraction Decomposition

Decomposes a rational expression into a sum of simpler fractions:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// 1/(x^2 - 1) → 1/(2*(x-1)) - 1/(2*(x+1))
let expr = 1 / &(expr!(x^2 - 1));
let decomposed = expr.partial_fractions(&x);
println!("{decomposed}");
```

Partial fractions are essential for:
- Integration of rational functions
- Inverse Laplace transforms
- Transfer function analysis in control theory

### `together()` and `cancel()`

`.together()` combines fractions over a common denominator. `.cancel()` cancels common polynomial factors:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Combine separate fractions
let expr = &(1 / &x) + &(1 / &(&x + 1));
let combined = expr.together();
println!("{combined}"); // (2*x + 1)/(x*(x + 1)) or equivalent

// Cancel common factors
let numer = expr!(x^2 - 1);
let denom = expr!(x - 1);
let expr = &numer / &denom;
let cancelled = expr.cancel(&x);
println!("{cancelled}"); // x + 1
```

### `as_numer_denom()` — Extract Numerator and Denominator

Split a rational expression into its parts:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let expr = &expr!(x + 1) / &expr!(x - 1);
let (numer, denom) = expr.as_numer_denom();
println!("Numerator: {numer}");   // x + 1
println!("Denominator: {denom}"); // x - 1
```

## Polynomial Utilities

### `collect()` — Collect Terms

Groups terms by powers of a variable:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

let expr = expr!(x*y + x + y + 1);
let collected = expr.collect(&x);
println!("{collected}"); // x*(y + 1) + y + 1 (or equivalent)
```

### `degree()` and `coeffs()`

Inspect polynomial structure:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let poly = expr!(3*x^3 + 2*x + 1);
println!("Degree: {}", poly.degree(&x)); // 3

let coefficients = poly.coeffs(&x);
for (i, c) in coefficients.iter().enumerate() {
    println!("  x^{i}: {c}");
}
```

### `coeff()` — Extract a Single Coefficient

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let poly = expr!(3*x^2 + 5*x + 7);
println!("{}", poly.coeff(&x, 2)); // 3  (coefficient of x^2)
println!("{}", poly.coeff(&x, 1)); // 5  (coefficient of x)
println!("{}", poly.coeff(&x, 0)); // 7  (constant term)
```

### `poly_gcd()` and `poly_lcm()`

Greatest common divisor and least common multiple of polynomials:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let a = expr!(x^2 - 1);            // (x-1)(x+1)
let b = expr!(x^2 - 2*x + 1);     // (x-1)^2
let gcd = a.poly_gcd(&b, &x);
println!("GCD: {gcd}"); // x - 1 (or equivalent)
```

### Query Functions

Check properties of an expression without transforming it:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let poly = expr!(x^2 + 1);
println!("Is constant? {}", poly.is_constant());      // false
println!("Is polynomial? {}", poly.is_polynomial(&x)); // true

let c = ctx.int(5);
println!("Is constant? {}", c.is_constant()); // true
```

## Separate Variables

`.separate_vars()` attempts to factor an expression into independent pieces:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

let expr = &x.sin() * &y.exp();
if let Some(factors) = expr.separate_vars(&[&x, &y]) {
    for (i, factor) in factors.iter().enumerate() {
        println!("Factor {i}: {factor}");
    }
}
```

## Choosing the Right Tool

Here's a decision tree:

1. **You have a trig expression** → Try `.simplify_trig()` first, then `.expand_trig()` if you want to decompose compound angles.

2. **You have a rational expression** → Use `.simplify_rational()` to cancel, `.partial_fractions(&x)` to decompose.

3. **You need to expand a product or power** → `.expand()`.

4. **You need to factor a polynomial** → `.factor(&x)`.

5. **You have logarithms** → `.expand_log()` to split, `.log_combine()` to merge.

6. **You just want it simpler and don't know the structure** → `.full_simplify()`.

7. **You need to check if two expressions are equal** → Compute `(a - b).full_simplify()` and check if the result is zero, or use `a.equals(&b)`.

---

*[← Chapter 3: Common Gotchas](03-gotchas.md) | [Back to Table of Contents](index.md) | [Chapter 5: Calculus →](05-calculus.md)*