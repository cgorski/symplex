# Chapter 11: Assumptions

In symbolic math, `x` is just a symbol — the system doesn't know whether it's positive, real, an integer, or something else entirely. The **assumption system** lets you attach mathematical properties to symbols so that symplex can answer questions like "is this expression positive?" or "is this product real?"

## Why Assumptions Matter

Consider `√(x²)`. Is that `x` or `|x|`? It depends:

- If `x` is positive, `√(x²) = x`
- If `x` is negative, `√(x²) = -x = |x|`
- If `x` is complex, things get even more subtle

Without assumptions, symplex can't simplify `√(x²)` — it doesn't know which case applies. With assumptions, it has the information it needs (though as we'll see, symplex doesn't yet *use* assumptions to guide simplification — that's a major planned feature).

## The Assumption Enum

The `Assumption` enum has 31 variants covering the standard mathematical property hierarchy. Here's the full list:

| Assumption | Meaning |
|------------|---------|
| `Commutative` | Commutes under multiplication |
| `Complex` | Element of ℂ |
| `Real` | Element of ℝ |
| `Rational` | Element of ℚ |
| `Integer` | Element of ℤ |
| `Algebraic` | Root of a polynomial with rational coefficients |
| `Transcendental` | Not algebraic (like π, e) |
| `Irrational` | Real but not rational |
| `Imaginary` | Pure imaginary (nonzero, real part is zero) |
| `Positive` | Strictly greater than zero |
| `Negative` | Strictly less than zero |
| `NonNegative` | Greater than or equal to zero |
| `NonPositive` | Less than or equal to zero |
| `Zero` | Equal to zero |
| `NonZero` | Not equal to zero |
| `Even` | Divisible by 2 (integer) |
| `Odd` | Not divisible by 2 (integer) |
| `Prime` | A prime number |
| `Composite` | A composite number |
| `Finite` | Bounded in absolute value |
| `Infinite` | Unbounded (±∞) |
| `Hermitian` | Equal to its own conjugate transpose |
| `AntiHermitian` | Equal to the negation of its conjugate transpose |

There are also negated forms for common denials: `NotReal`, `NotComplex`, `NotInteger`, `NotRational`, `NotPositive`, `NotNegative`, `NotZero`, `NotFinite`.

## Setting Assumptions

Use `.assume()` on a symbol. It returns the same `Ex` (fluent API), so you can chain:

```rust
use symplex::prelude::*;

let ctx = Context::new();
let t = ctx.var("t")
    .assume(Assumption::Positive)
    .assume(Assumption::Real);

assert_eq!(t.is_positive(), Some(true));
assert_eq!(t.is_real(), Some(true));
```

You can also set assumptions at creation time using the `sym!` macro:

```rust
use symplex::prelude::*;
use symplex::sym;

let ctx = Context::new();
sym!(ctx; t, Positive, Real);

assert_eq!(t.is_positive(), Some(true));
assert_eq!(t.is_real(), Some(true));
```

Or use the `syms!` macro for multiple symbols (without assumptions) and then add them:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

let x = x.assume(Assumption::Positive);
let y = y.assume(Assumption::Integer);
```

## Querying Assumptions

Every property has a dedicated query method that returns `Option<bool>`:

- `Some(true)` — the property is provably true
- `Some(false)` — the property is provably false
- `None` — unknown

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.var("x").assume(Assumption::Positive);

assert_eq!(x.is_positive(), Some(true));
assert_eq!(x.is_negative(), Some(false)); // positive → not negative
assert_eq!(x.is_real(), Some(true));       // positive → real
assert_eq!(x.is_even(), None);             // don't know
```

Here's the full list of query methods:

| Method | Property |
|--------|----------|
| `.is_positive()` | Strictly > 0 |
| `.is_negative()` | Strictly < 0 |
| `.is_nonnegative()` | ≥ 0 |
| `.is_nonpositive()` | ≤ 0 |
| `.is_zero()` | = 0 |
| `.is_nonzero()` | ≠ 0 |
| `.is_real()` | ∈ ℝ |
| `.is_complex()` | ∈ ℂ |
| `.is_integer()` | ∈ ℤ |
| `.is_rational()` | ∈ ℚ |
| `.is_imaginary()` | Pure imaginary |
| `.is_finite()` | Bounded |
| `.is_even()` | Divisible by 2 |
| `.is_odd()` | Not divisible by 2 |
| `.is_prime()` | Prime |
| `.is_composite()` | Composite |
| `.is_algebraic()` | Algebraic number |
| `.is_transcendental()` | Transcendental number |
| `.is_irrational()` | Irrational number |
| `.is_hermitian()` | Hermitian |

For any property not covered by a dedicated method, use `.query(Props::FLAG)`:

```rust
use symplex::prelude::*;
use symplex::base::assumptions::Props;

let ctx = Context::new();
let x = ctx.var("x").assume(Assumption::AntiHermitian);
assert_eq!(x.query(Props::ANTIHERMITIAN), Some(true));
```

## Forward-Chaining Inference

The real power of the assumption system is **automatic inference**. When you assert one property, symplex deduces everything that logically follows.

### Positive → the full chain

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.var("x").assume(Assumption::Positive);

// Direct assertion:
assert_eq!(x.is_positive(), Some(true));

// Inferred:
assert_eq!(x.is_nonnegative(), Some(true));  // positive → nonnegative
assert_eq!(x.is_nonzero(), Some(true));       // positive → nonzero
assert_eq!(x.is_real(), Some(true));          // positive → real
assert_eq!(x.is_complex(), Some(true));       // real → complex
assert_eq!(x.is_finite(), Some(true));        // real → finite
assert_eq!(x.is_hermitian(), Some(true));     // real → hermitian

// Negative inferences:
assert_eq!(x.is_negative(), Some(false));     // positive → not negative
assert_eq!(x.is_zero(), Some(false));         // positive → not zero
assert_eq!(x.is_imaginary(), Some(false));    // real → not imaginary
assert_eq!(x.is_infinite(), Some(false));     // finite → not infinite
```

A single `Positive` assertion yields at least 12 derived facts. This is because the inference engine applies implication rules in a loop until no new facts emerge — typically converging in 2–4 iterations.

### Integer → rational → real → complex

```rust
use symplex::prelude::*;

let ctx = Context::new();
let n = ctx.var("n").assume(Assumption::Integer);

assert_eq!(n.is_integer(), Some(true));
assert_eq!(n.is_rational(), Some(true));      // integer → rational
assert_eq!(n.is_algebraic(), Some(true));     // rational → algebraic
assert_eq!(n.is_real(), Some(true));          // rational → real
assert_eq!(n.is_complex(), Some(true));       // real → complex
assert_eq!(n.is_finite(), Some(true));        // integer → finite
assert_eq!(n.is_transcendental(), Some(false)); // algebraic → not transcendental
assert_eq!(n.is_imaginary(), Some(false));    // real → not imaginary
```

### Prime → integer + positive

```rust
use symplex::prelude::*;

let ctx = Context::new();
let p = ctx.var("p").assume(Assumption::Prime);

assert_eq!(p.is_prime(), Some(true));
assert_eq!(p.is_integer(), Some(true));       // prime → integer
assert_eq!(p.is_positive(), Some(true));      // prime → positive
assert_eq!(p.is_nonzero(), Some(true));       // positive → nonzero
assert_eq!(p.is_composite(), Some(false));    // prime → not composite
```

### Join rules: two facts combine

Some inferences require two facts together:

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.var("x")
    .assume(Assumption::NonNegative)
    .assume(Assumption::NonZero);

// Neither alone implies positive, but together they do:
assert_eq!(x.is_positive(), Some(true)); // nonneg ∧ nonzero → positive
```

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.var("x")
    .assume(Assumption::NonNegative)
    .assume(Assumption::NonPositive);

// The only number that's both nonneg and nonpos is zero:
assert_eq!(x.is_zero(), Some(true));
```

## Assumptions on Computed Expressions

The assumption system doesn't just work on symbols — it propagates through operations:

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.var("x").assume(Assumption::Positive);
let y = ctx.var("y").assume(Assumption::Positive);

// Sum of positives is positive
let sum = &x + &y;
assert_eq!(sum.is_positive(), Some(true));

// Product of positive and negative is negative
let z = ctx.var("z").assume(Assumption::Negative);
let prod = &x * &z;
assert_eq!(prod.is_negative(), Some(true));
```

The cache system also handles built-in constants:

```rust
use symplex::prelude::*;

let ctx = Context::new();
let pi = ctx.pi();
assert_eq!(pi.is_positive(), Some(true));
assert_eq!(pi.is_real(), Some(true));
assert_eq!(pi.is_transcendental(), Some(true));
assert_eq!(pi.is_irrational(), Some(true));

let e = ctx.e();
assert_eq!(e.is_positive(), Some(true));
assert_eq!(e.is_transcendental(), Some(true));

let i = ctx.i_unit();
assert_eq!(i.is_imaginary(), Some(true));
assert_eq!(i.is_real(), Some(false));
```

## The Honesty Section: What Assumptions Don't Do (Yet)

Symplex stores and queries assumptions but **does not yet use them to guide simplification**. This is a major planned feature — arguably the single biggest gap between symplex and mature CAS tools like SymPy.

Here's what that means concretely:

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.var("x").assume(Assumption::Positive);

// We KNOW x is positive...
assert_eq!(x.is_positive(), Some(true));

// ...but simplify() doesn't use that knowledge:
let expr = x.powi(2).sqrt(); // sqrt(x²)
let simplified = expr.simplify();
// PLANNED: should give `x` since x is positive
// Currently: returns sqrt(x^2) unchanged
println!("{simplified}"); // sqrt(x^2), not x
```

The assumption queries work. The inference engine works. But the simplification engine doesn't consult them yet. You can *ask* "is x positive?" and get the right answer, but `simplify()` doesn't ask.

## What's Next

The assumption infrastructure is in place and waiting for the simplification engine to use it. Here's what's planned:

- **`refine(expr, assumptions)`** — simplify an expression using assumption knowledge:
  - `sqrt(x²)` → `x` when x is positive
  - `Abs(x)` → `x` when x is positive
  - `(-1)^(2n)` → `1` when n is integer
- **`ask(query)`** — a programmatic boolean query interface (like SymPy's `ask(Q.positive(x))`)
- **Assumptions in integration** — `∫ exp(-a·x) dx` with `a > 0` should converge, and the integrator should know that
- **`with_assuming()` context** — a temporary scope where assumptions are active, then automatically rolled back:

```rust
// PLANNED: temporary assumption scope
let x = ctx.var("x");
let result = with_assuming(&[(&x, Assumption::Positive)], || {
    x.powi(2).sqrt().simplify() // would give x
});
```

For now, use assumptions for querying properties and let the rest of your code make decisions based on those queries. Head to [Chapter 12: Plotting](12-plotting.md) to learn how to visualize your expressions.

---

*[← Chapter 10: Sets and Logic](10-sets-and-logic.md) | [Back to Table of Contents](index.md) | [Chapter 12: Plotting →](12-plotting.md)*