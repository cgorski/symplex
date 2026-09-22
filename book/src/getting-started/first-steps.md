# First Steps

This chapter walks through the basics of creating and manipulating symbolic expressions in symplex. By the end, you will know how to build expressions, evaluate them, differentiate, and integrate.

## Creating a Context

Every expression in symplex belongs to a `Context`. The context owns the expression storage (an arena) and manages symbol names and assumptions. You create one at the start of your program:

```rust
use symplex::prelude::*;

let ctx = Context::new();
```

You can create multiple contexts if you need isolated environments (e.g., separate user sessions on a server). Expressions from different contexts cannot be mixed — the library enforces this at runtime.

## Defining Symbols

Symbols are named unknowns. They represent the variables in your expressions:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
let x = ctx.symbol("x");
let y = ctx.symbol("y");
```

There is also a convenience macro for declaring multiple symbols at once:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
symplex::syms!(ctx; x, y, z);
```

Symbols are symbolic — they don't have a value until you substitute one. Calling `x.eval_f64()` on a bare symbol returns `Err` because there is nothing to evaluate.

## Building Expressions

### Arithmetic operators

Expressions support the standard Rust arithmetic operators. Because `Ex` is not `Copy` (it is `Clone` and `Send + Sync`), you typically work with references:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
let f = &x * &x + &x * 2 + 1;   // x² + 2x + 1
let g = &x.powi(3) - &y;         // x³ - y
```

Integer literals (`1`, `2`, etc.) are automatically converted to exact rational constants. There is no floating-point contamination.

### The `expr!` macro

For more complex expressions, the `expr!` macro provides mathematical notation:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
let f = expr!(ctx, x^2 + 2*x + 1);
let g = expr!(ctx, sin(x)^2 + cos(x)^2);
let h = expr!(ctx, x^3 - 3*x^2 + 2*x);
```

The macro recognizes standard mathematical functions (`sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `abs`, `gamma`, `erf`, and many others), the constants `pi` and `E`, and the caret `^` for exponentiation.

### Exact rationals

Constants are exact. The rational number 1/3 is stored as the ratio of two arbitrary-precision integers, not as `0.33333...`:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
let half = ctx.rational(1, 2);     // exactly 1/2
let third = ctx.rational(1, 3);    // exactly 1/3
let big = ctx.int(1_000_000_007);  // arbitrary-precision integer
```

### Functions

Standard mathematical functions are methods on `Ex`:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
let a = x.sin();          // sin(x)
let b = x.exp();          // exp(x)
let c = x.ln();           // ln(x)
let d = x.sqrt();         // √x
let e = x.powi(5);        // x⁵
let f = x.pow(&y);        // x^y
let g = x.factorial();    // x!
let h = x.gamma();        // Γ(x)
```

## Displaying Expressions

Expressions implement `Display` for plain-text output and have a `to_latex()` method for LaTeX:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
let f = expr!(ctx, x^2 + 2*x + 1);
println!("{f}");                    // x^2 + 2*x + 1
println!("{}", f.to_latex());       // x^{2} + 2x + 1
```

## Evaluating Expressions

### Symbolic evaluation

`.eval()` applies exact simplification rules — reducing `sin(0)` to `0`, `exp(ln(x))` to `x`, computing `5!` to `120`, and so on — without any floating-point approximation:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
let a = ctx.int(5).factorial().eval();
println!("{a}");   // 120

let b = expr!(ctx, sin(pi)).eval();
println!("{b}");   // 0
```

### Numerical evaluation

`.eval_f64()` converts a fully determined expression (no free symbols) to an `f64`. It returns `Result` because the conversion can fail:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
let val = expr!(ctx, sin(1) + cos(1)).eval_f64().unwrap();
println!("{val:.6}");   // 1.381773
```

If free symbols remain, you get an error:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
let result = x.sin().eval_f64();
assert!(result.is_err());   // "expression contains free symbol 'x'"
```

### Substitution

Use `.subs()` to replace a symbol with a value or another expression:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
let f = expr!(ctx, x^2 + 1);

// Substitute x = 3 (exact integer)
let at_3 = f.subs(&x, &ctx.int(3)).eval();
println!("{at_3}");   // 10

// Substitute x = y + 1 (symbolic)
let shifted = f.subs(&x, &(&y + 1));
println!("{shifted}"); // (y + 1)^2 + 1
```

For quick numerical substitution there is a convenience method:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
# let f = expr!(ctx, x^2 + 1);
let val = f.eval_f64_with(&[(&x, 3)]).unwrap();
println!("{val}");   // 10.0
```

## Differentiation

`.diff(&var)` computes the symbolic derivative with respect to a variable. It handles the chain rule, product rule, quotient rule, and all elementary functions:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
let f = expr!(ctx, x^3 - 3*x^2 + 2*x);
let df = f.diff(&x);
println!("{df}");   // 3*x^2 - 6*x + 2

// Second derivative
let d2f = df.diff(&x);
println!("{d2f}");  // 6*x - 6
```

Higher-order derivatives have a convenience method:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
let d4 = expr!(ctx, x^6).diff_n(&x, 4);
println!("{d4}");   // 360*x^2
```

Partial derivatives work the same way — just specify which variable:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
let g = expr!(ctx, x^2 * y + y^3);
println!("∂g/∂x = {}", g.diff(&x));   // 2*x*y
println!("∂g/∂y = {}", g.diff(&y));   // x^2 + 3*y^2
```

## Integration

`.integrate(&var)` computes the indefinite integral. The library uses multiple strategies (polynomial, u-substitution, by-parts, partial fractions, trigonometric, Risch algorithm, heuristic integration):

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
let f = expr!(ctx, x^2);
let anti = f.integrate(&x);
println!("{anti}");   // 1/3*x^3
```

Definite integrals:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
let zero = ctx.int(0);
let one = ctx.int(1);
let area = expr!(ctx, x^2).integrate_definite(&x, &zero, &one);
println!("{area}");   // 1/3
```

Definite integration is *not* a naive `F(b) − F(a)`: it looks for singularities inside the interval, handles infinite bounds, and reports divergence instead of returning a wrong number. See [Definite Integration and Quadrature](../guide/definite-integration.md).

When integration cannot find a closed form, it returns an unevaluated `Integral` node rather than failing silently:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
let hard = expr!(ctx, x^x);
let result = hard.integrate(&x);
println!("{result}");   // Integral(x^x, x)
```

You can check whether a result contains unevaluated forms:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
# let result = expr!(ctx, x^x).integrate(&x);
if result.has_unevaluated() {
    println!("no closed form found");
}
```

Or use the `try_integrate` variant, which returns `Err` if the result is not fully evaluated:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
# let hard = expr!(ctx, x^x);
match hard.try_integrate(&x) {
    Ok(anti) => println!("closed form: {anti}"),
    Err(_) => println!("no closed form"),
}
```

## Simplification

`.simplify()` tries a dozen strategies (evaluation, expansion, factoring, trigonometric, logarithmic, power and radical rules, assumption-aware refinement, …), keeps the result with the fewest operations, and iterates to a fixpoint:

```rust
# use symplex::prelude::*;
# let ctx = Context::new();
# let x = ctx.symbol("x");
let expr = expr!(ctx, sin(x)^2 + cos(x)^2);
println!("{}", expr.simplify());   // 1

let expr = (&x + 1).powi(2) - &x.powi(2) - &x * 2;
println!("{}", expr.simplify());   // 1
```

`.simplify_with(&SimplifyOpts::single_pass())` runs one pass; `.simplify_traced(&SimplifyOpts::default())` also returns the strategies and rules that fired. Targeted simplifiers (`simplify_trig`, `expand_log`, `sqrtdenest`, `powdenest`, …) and your own rewrite rules are covered in [Algebra](../guide/algebra.md) and [The Rule Engine](../guide/rule-engine.md).

## Putting It Together

Here is a complete example that finds the critical points of a polynomial:

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let f = expr!(ctx, x^3 - 3*x + 1);
    let df = f.diff(&x);

    println!("f(x)  = {f}");
    println!("f'(x) = {df}");

    // Critical points: f'(x) = 0
    let crits = df.solve_or_empty(&x);
    for c in &crits {
        let val = f.subs(&x, c).eval();
        println!("  f({c}) = {val}");
    }
}
```

## Next Steps

Continue to [Key Concepts](./key-concepts.md) for a deeper understanding of how the library works — context ownership, the five API patterns, and how to handle unevaluated results.