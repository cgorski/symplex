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

2. **Safety.** Expressions from different contexts cannot be mixed. If you try to add an expression from `ctx_a` to one from `ctx_b`, the library detects this and panics with a clear message — it guards against a logic error analogous to indexing out of bounds. The library never calls `unwrap`/`expect`/`panic!`/`unreachable!` on user data (ratchet `tests/unit/test_no_panics.rs`); the remaining `assert!`s on caller-supplied *shapes* (e.g. `Matrix::zeros(0, n)`, `Context::symbol("")`) are documented under `# Panics` on each item and counted by the same ratchet.

```rust
# use symplex::prelude::*;
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
# use symplex::prelude::*;
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
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
# let a = ctx.int(0);
# let expr = x.sin();
expr.simplify();       // might return input unchanged
expr.expand();         // might return input unchanged
expr.eval();           // sin(0) → 0; symbolic expr → unchanged
expr.diff(&x);         // might return Derivative(expr, x) if it can't differentiate
expr.integrate(&x);    // might return Integral(expr, x) if no closed form
expr.limit(&x, &a);    // might return Limit(expr, x, a)
```

### Pattern 2: `try_` variant returns `Result`

For users who need guaranteed closed-form results (e.g., in a code generation pipeline), every Pattern 1 method that can produce an unevaluated form has a `try_` twin:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
# let expr = x.sin();
// CAS-style: always returns something
let anti = expr.integrate(&x);

// Pipeline-style: Err if unevaluated
let anti = expr.try_integrate(&x)?;
# Ok::<(), SymplexError>(())
```

The `try_` variant calls the base method, then checks `has_unevaluated()`. There is zero code duplication between the two.

Available: `try_diff`, `try_integrate`, `try_integrate_definite`, `try_limit`, `try_limit_left`/`right`/`dir`, `try_series`, `try_series_at_infinity`, `try_maclaurin`, `try_summation`, `try_product_over`, `try_laplace`, `try_inverse_laplace`, `try_residue`, `try_gosper_sum`, `try_solve_ode`, `try_solve_gt`/`ge`/`lt`/`le`.

Some `try_` variants carry extra information in the error: `try_integrate_definite` returns `Err(SymplexError::Divergent { .. })` when the integral is *proven* to diverge, as opposed to `Err(ComputationFailed)` when no closed form was found.

### Pattern 3: Numeric boundary → `Result`

Operations that cross from symbolic to numeric always return `Result`, because the conversion can fail if free symbols remain:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
# let (a, b) = (ctx.int(0), ctx.int(1));
# let expr = x.sin();
expr.eval_f64();                  // Err if free symbols remain
expr.eval_complex64();            // Err if can't evaluate
expr.eval_decimal(30);            // Err if precision exhausted
expr.compile(&["x"]);             // Err(FreeSymbol / NotImplemented) → Result<CompiledFn>
expr.to_rust_fn("f", &["x"]);     // Err if can't generate code
expr.to_c_fn("f", &["x"]);        // same, C99
expr.integrate_numeric(&x, &a, &b);   // Err if the quadrature does not converge
```

Solvers whose failure is a mathematical fact also use `Result`: `solve` returns `Err(InfiniteSolutions)` for an identity and `Err(NoSolution)` for a contradiction or range violation (`sin x = 2`); `linsolve` returns `Ok(LinearSolution::Inconsistent)` because inconsistency is a legitimate answer, but `Err(InvalidArgument)` for non-linear input.

### Pattern 4: Queries → `Option`

Three-valued queries return `Option` — the answer might be yes, no, or "can't determine":

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
# let k = ctx.symbol("k");
# let expr = x.powi(3);
# let other = x.powi(3);
# let elem = ctx.int(1);
# let set = ctx.reals();
# let matrix = Matrix::identity(&ctx, 2).unwrap();
expr.is_positive();        // Some(true), Some(false), or None
expr.degree(&x);           // Some(3) or None (not a polynomial)
expr.equals(&other);       // Some(true), Some(false), or None
expr.is_convergent(&k);    // decisive answers only
set.contains(&elem);       // set membership
matrix.is_symmetric();     // structure tests on matrices are three-valued too
```

### Pattern 5: Structural preconditions → `Result`

Operations with structural requirements (e.g., matrix operations that require specific shapes):

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let matrix = Matrix::identity(&ctx, 2).unwrap();
# let other = Matrix::identity(&ctx, 2).unwrap();
matrix.det();           // Err if non-square
matrix.inv();           // Err if singular
matrix.matmul(&other);  // Err if dimensions don't match
matrix.cholesky();      // Err if not symmetric / not positive definite
matrix.minor(0, 0);     // Err if out of range (the sub-matrix is minor_matrix)
```

## Unevaluated Forms

When symplex cannot compute a closed-form result, it returns an unevaluated symbolic node. This is a deliberate design choice — the library never returns a wrong answer or silently drops a computation.

```rust
# use symplex::prelude::*;
let ctx = Context::new();
let x = ctx.symbol("x");

// No closed-form antiderivative exists for x^x
let result = expr!(ctx, x^x).integrate(&x);
println!("{result}");   // Integral(x^x, x)
```

The expression `Integral(x^x, x)` is not an error — it is a truthful representation of the mathematical object "the integral of x^x with respect to x." It can be:

- Displayed as text or LaTeX
- Substituted into larger expressions
- Checked with `.has_unevaluated()`
- Rejected with `try_integrate()` if you need a closed form

Common unevaluated forms:

| Node | Meaning |
|------|---------|
| `Derivative(f, x)` | Derivative that couldn't be computed |
| `Integral(f, x)` | Antiderivative not found |
| `Integral(f, x, a, b)` | Definite integral that could not be decided (`DefiniteIntegral` node; `eval_f64` evaluates it numerically) |
| `Limit(f, x, a)` | Limit couldn't be determined (including a two-sided limit whose one-sided limits differ) |
| `Series(f, x, a, n)` | Series expansion failed |
| `Sum(f, k, a, b)` / `Product_(f, k, a, b)` | No closed form for the sum / product |
| `LaplaceTransform(f, t, s)` | Not in the Laplace table |
| `re(z)`, `im(z)`, `conjugate(z)`, `arg(z)` | Realness of `z` unknown |
| `stirling2(n, k)` | Stirling number with symbolic arguments |

`RootOf(poly, index)` and `RootSum(poly, body, var)` are **not** unevaluated: they are complete, exact descriptions of algebraic numbers (with numerical evaluation), so `has_unevaluated()` returns `false` for them and `try_` methods accept them.

## Evaluation Configuration

You can control computational limits via `EvalConfig`:

```rust
# use symplex::prelude::*;
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
# use symplex::prelude::*;
# use symplex::sym;
# let ctx = Context::new();
sym!(ctx; x, Positive);    // x > 0
sym!(ctx; n, Integer);     // n ∈ ℤ
# Ok::<(), SymplexError>(())
```

With `x` declared positive, `sqrt(x²)` simplifies to `x`; declared `Real`, to `|x|`; without an assumption it stays `sqrt(x²)` (see the domain model below).

Assumptions matter for correctness, not just for prettier output. Without `Real`, `z.re()` stays `re(z)`, `√(z²)` does not become `|z|`, and `∫₀^∞ e^(−a x) dx` will not simplify to `1/a` (it needs `a > 0`).

Available assumptions include `Positive`, `Negative`, `NonNegative`, `NonPositive`, `Integer`, `Real`, `ExtendedReal`, `Complex`, `Even`, `Odd`, `Prime`, `Finite`, `Zero`, `NonZero`, and their negations (`NotPositive`, `NotZero`, …). `ctx.symbol_with("a", &[Assumption::Positive])` is the non-macro form; `Assumptions::implies` and `Assumption::negate` let you reason about them programmatically.

## The Domain Model

symplex is a CAS over the complex numbers, on the principal branch:

1. **A symbol without assumptions may be complex.**  `Real`, `Positive`, `Integer`, … narrow it.
2. **Every multivalued function takes its principal branch** — `ln z = ln|z| + i·arg z` with `arg z ∈ (−π, π]`, `z^a = e^(a·ln z)`, and the inverse trigonometric and hyperbolic functions as in DLMF (and mpmath, and SymPy's `N`).  So `∛(−8) = 2·(−1)^(1/3) = 1 + √3·i`; the real cube root is `ctx.int(-8).real_root(3)? = −2`.  The exact evaluator (`eval`), the numeric one (`eval_f64`, `eval_complex64`, `eval_decimal`) and every rewrite agree on it.
3. **A rewrite preserves the value everywhere.**  An identity of real analysis is applied only where the assumptions make it true:

| Identity | Holds when | Applied by default when | Unconditionally |
|---|---|---|---|
| `ln(a·b) = ln a + ln b` | `arg a + arg b ∈ (−π, π]` | a factor is known positive (it is split off), or known negative (split off as `ln(−f)`, its sign kept) | `expand_log_with(true)` |
| `ln a + ln b = ln(a·b)` | the same | the logarithms of known-positive arguments, together with at most one other (`ln 2 + ln x = ln(2x)` for every `x`) | `log_combine_with(true)` |
| `ln(a^c) = c·ln a` | `c·arg a ∈ (−π, π]` | `a > 0` and `c` real, or `−1 < c ≤ 1` (`ln √x = ½ ln x` for every `x`) | `expand_log_with(true)` / `log_combine_with(true)` |
| `ln(e^w) = w` | `Im w ∈ (−π, π]` | `w` known real | `expand_log_with(true)` |
| `√(w²) = ∣w∣` | `w` real | `w` known real (`√(i²) = i`) | `powdenest(true)` gives `w` |
| `(x^a)^b = x^(a·b)` | `b ∈ ℤ`, or `x ≥ 0`, or `−1 < a ≤ 1` | the same | `powdenest(true)` |
| `(e^f)^g = e^(f·g)` | `g ∈ ℤ`, or `Im f ∈ (−π, π]` | `g` integer or `f` known real (`√(e^{4i}) ≠ e^{2i}`) | — |
| `asinh(sinh w) = w`, `atanh(tanh w) = w`, `acosh(cosh w) = ∣w∣` | `w` real (for `asinh`, `∣Im w∣ ≤ π/2`) | `w` known real | — |

```rust
# use symplex::prelude::*;
let ctx = Context::new();
let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
assert_eq!((&x.ln() + &y.ln()).simplify(), &x.ln() + &y.ln());   // x = y = −1: 2πi ≠ 0
assert_eq!(x.powi(2).sqrt().simplify(), x.powi(2).sqrt());      // x = i:  i ≠ 1
let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
assert_eq!(r.powi(2).sqrt().simplify(), r.abs());
let (p, q) = (ctx.symbol_with("p", &[Assumption::Positive]).unwrap(), ctx.symbol_with("q", &[Assumption::Positive]).unwrap());
assert_eq!(format!("{}", (&p.ln() + &q.ln()).simplify()), "ln(p*q)");
assert_eq!(format!("{}", (ctx.int(2).ln() + x.ln()).log_combine()), "ln(2*x)");
assert_eq!(ctx.int(-8).real_root(3)?, ctx.int(-2));
# Ok::<(), SymplexError>(())
```

**Where a variable is real by construction**, symplex uses the fact:

- *Limits* approach along the real axis: the limit variable is a positive dummy inside the Gruntz algorithm (as in SymPy), so `lim_{x→∞} e^x^(1/x) = e`.
- *Integration* is over a real variable, with the real-variable antiderivative `∫ dx/x = ln|x|`: an antiderivative is valid on each real interval where the integrand is continuous, and `√(x²)` is `|x|` for the integration variable itself.

**The one documented exception is generated numeric code.**  `compile()` and the Rust, C, Python, NumPy and Julia back ends work in `f64` reals, so `x^(p/q)` with an odd denominator `q` is the *real* root there (`sign(x)·|x|^(p/q)` for odd `p`, `|x|^(p/q)` for even `p`), as it has been since 0.11.1 — a principal-branch value would be complex, which an `f64` cannot hold.  Even denominators follow `f64` semantics (`NaN` for a negative base).

Before 0.23 some of these rewrites fired for every symbol not known to be *non*-real, `expand_log`/`log_combine` were the forced forms, and `eval` took the real odd root; the changelog lists every change.

## Next Steps

You now understand the core abstractions. The [Guide](../guide/calculus.md) chapters cover each mathematical domain in depth, and the [Cookbook](../cookbook/pid-controller.md) shows complete worked solutions to real problems.