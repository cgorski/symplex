# Key Concepts

This chapter explains the design decisions that affect how you write code with symplex. Understanding these concepts will help you read error messages, choose the right method variants, and structure your programs effectively.

## Contexts and Ownership

Every expression in symplex belongs to a `Context`. The context owns an arena (a pool of expression nodes) and manages symbol names, assumptions, and evaluation settings.

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.symbol("x");
let f = x.powi(2) + 1;   // this expression lives in ctx's arena
```

### Why contexts exist

Contexts serve two purposes:

1. **Isolation.** Different contexts have independent symbol tables and assumptions. A server handling multiple users can give each one a separate context without interference.

2. **Safety.** Expressions from different contexts cannot be mixed. If you try to add an expression from `ctx_a` to one from `ctx_b`, the library detects this and panics with a clear message. This is the only panic in the symbolic layer — it guards against a logic error analogous to indexing out of bounds.

```rust
let ctx_a = Context::new();
let ctx_b = Context::new();
let x = ctx_a.symbol("x");
let y = ctx_b.symbol("y");

// This panics: "cannot combine expressions from different contexts"
// let bad = &x + &y;
```

### Cloning and thread safety

`Context` is `Clone` — cloning shares the underlying arena via `Arc<RwLock<...>>`. Multiple threads can hold clones of the same context and create expressions concurrently. `Ex` (the expression handle type) is `Send + Sync`.

```rust
let ctx = Context::new();
let ctx2 = ctx.clone();  // shares the same arena

std::thread::spawn(move || {
    let y = ctx2.symbol("y");
    println!("{}", y.powi(2));
});
```

## The Expression Type: `Ex`

`Ex` is a type alias for `Expr<Numeric>`. It is the primary expression handle. Internally, it holds:

- A reference to the context (via `Arc`)
- An opaque index into the arena (`ExprId`)

`Ex` is `Clone` (cheap — it's just an Arc bump and a u32 copy) but not `Copy`. You will often work with `&Ex` references to avoid unnecessary clones.

There are also `BoolEx` (for boolean expressions like `x > 0`) and `SetEx` (for set-valued expressions like solution sets). These are distinct types — you cannot pass a `BoolEx` where an `Ex` is expected.

## The Five API Patterns

Every operation in symplex follows one of five patterns. Knowing which pattern a method uses tells you what to expect from its return type.

### Pattern 1: Always returns `Ex`

Operations where "unchanged" or "unevaluated" is a valid result. These never fail — they always return something meaningful.

```rust
expr.simplify()       // might return input unchanged
expr.expand()         // might return input unchanged
expr.eval()           // sin(0) → 0; symbolic expr → unchanged
expr.diff(&x)         // might return Derivative(expr, x) if it can't differentiate
expr.integrate(&x)    // might return Integral(expr, x) if no closed form
expr.limit(&x, &a)    // might return Limit(expr, x, a)
```

### Pattern 2: `try_` variant returns `Result`

For users who need guaranteed closed-form results (e.g., in a code generation pipeline), every Pattern 1 method that can produce an unevaluated form has a `try_` twin:

```rust
// CAS-style: always returns something
let anti = expr.integrate(&x);

// Pipeline-style: Err if unevaluated
let anti = expr.try_integrate(&x)?;
```

The `try_` variant calls the base method, then checks `has_unevaluated()`. There is zero code duplication between the two.

Available: `try_diff`, `try_integrate`, `try_limit`, `try_series`, `try_laplace`, `try_inverse_laplace`, `try_residue`, `try_gosper_sum`, `try_solve_ode`.

### Pattern 3: Numeric boundary → `Result`

Operations that cross from symbolic to numeric always return `Result`, because the conversion can fail if free symbols remain:

```rust
expr.eval_f64()          // Err if free symbols remain
expr.eval_complex64()    // Err if can't evaluate
expr.eval_decimal(30)    // Err if precision exhausted
expr.compile(&["x"])     // None if unsupported nodes
expr.to_rust_fn("f", &["x"])  // Err if can't generate code
```

### Pattern 4: Queries → `Option`

Three-valued queries return `Option` — the answer might be yes, no, or "can't determine":

```rust
expr.is_positive()    // Some(true), Some(false), or None
expr.degree(&x)       // Some(3) or None (not a polynomial)
expr.equals(&other)   // Some(true), Some(false), or None
```

### Pattern 5: Structural preconditions → `Result`

Operations with structural requirements (e.g., matrix operations that require specific shapes):

```rust
matrix.det()           // Err if non-square
matrix.inv()           // Err if singular
matrix.matmul(&other)  // Err if dimensions don't match
```

## Unevaluated Forms

When symplex cannot compute a closed-form result, it returns an unevaluated symbolic node. This is a deliberate design choice — the library never returns a wrong answer or silently drops a computation.

```rust
let ctx = Context::new();
let x = ctx.symbol("x");

// No closed-form antiderivative exists for exp(x²)
let result = expr!(ctx, exp(x^2)).integrate(&x);
println!("{result}");   // Integral(exp(x^2), x)
```

The expression `Integral(exp(x^2), x)` is not an error — it is a truthful representation of the mathematical object "the integral of exp(x²) with respect to x." It can be:

- Displayed as text or LaTeX
- Substituted into larger expressions
- Checked with `.has_unevaluated()`
- Rejected with `try_integrate()` if you need a closed form

Common unevaluated forms:

| Node | Meaning |
|------|---------|
| `Derivative(f, x)` | Derivative that couldn't be computed |
| `Integral(f, x)` | Antiderivative not found |
| `Limit(f, x, a)` | Limit couldn't be determined |
| `Series(f, x, a, n)` | Series expansion failed |
| `RootOf(poly, index)` | Root of a polynomial of degree ≥ 5 |
| `stirling2(n, k)` | Stirling number with symbolic arguments |

## Evaluation Configuration

You can control computational limits via `EvalConfig`:

```rust
let config = EvalConfig {
    max_pow_exponent: 1000,    // don't auto-evaluate 2^5000
    max_result_digits: 5000,   // cap result size
    max_evalf_precision: 10_000, // max bits for numerical eval
};
let ctx = Context::with_config(config);
```

When a computation exceeds these limits, the result stays in unevaluated form rather than consuming unbounded memory. For example, `2^5000` with `max_pow_exponent = 1000` remains as `2^5000` (a `Pow` node) instead of computing a 1,505-digit number.

## Assumptions

You can declare properties of symbols to help the simplifier:

```rust
let x = sym!(ctx; x, Positive);    // x > 0
let n = sym!(ctx; n, Integer);     // n ∈ ℤ
```

With `x` declared positive, `sqrt(x²)` simplifies to `x` (without the assumption, the result is `|x|` or stays as `sqrt(x²)`).

Available assumptions: `Positive`, `Negative`, `NonNegative`, `NonPositive`, `Integer`, `Real`, `Complex`, `Even`, `Odd`, `Prime`, `Finite`, `Zero`, `NonZero`.

## Next Steps

You now understand the core abstractions. The [Guide](../guide/calculus.md) chapters cover each mathematical domain in depth, and the [Cookbook](../cookbook/pid-controller.md) shows complete worked solutions to real problems.