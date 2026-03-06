# Chapter 2: Getting Started

## Installation

Add symplex to your project:

```bash
cargo add symplex
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
symplex = "0.1"
```

**Minimum Supported Rust Version (MSRV):** 1.93.0

## The Prelude and Variables

Every symplex session starts with two imports:

```rust
use symplex::prelude::*;
use symplex::vars;
```

The prelude brings in the core types — `Ex` (the expression type), `Matrix`, `Context`, `StateSpace`, `TransferFunction`, and the proc macros `expr!`, `matrix!`, and `eq!`.

The `vars!` macro is a separate `#[macro_export]` macro that creates symbolic variables in the global default context:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x, y, z);
// x, y, z are now `Ex` values bound to symbols named "x", "y", "z"
```

Under the hood, `vars!(x)` expands to `let x = symplex::var("x");`. You can also create variables directly:

```rust
let theta = symplex::var("theta");
let omega = symplex::var("omega");
```

### Using a Context

For more control, create an explicit `Context`:

```rust
use symplex::prelude::*;
use symplex::syms;

let ctx = Context::new();
syms!(ctx; x, y, z);
```

The `syms!` macro works like `vars!` but takes a context. You can also attach assumptions to symbols:

```rust
use symplex::prelude::*;
use symplex::sym;

let ctx = Context::new();
sym!(ctx; t, Positive, Real);
// t is now known to be positive and real
```

For most use cases, the global default context via `vars!` is sufficient.

## Building Expressions

### The `expr!` Macro

The `expr!` macro is the most convenient way to build expressions. It parses a mathematical expression at compile time and generates the corresponding symplex API calls:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x, y);

let f = expr!(x^2 + 2*x + 1);
let g = expr!(sin(x)^2 + cos(x)^2);
let h = expr!(x^2 * y + y^3);
```

The macro supports:
- Arithmetic: `+`, `-`, `*`, `/`, `^` (power)
- Functions: `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `abs`, `asin`, `acos`, `atan`, and more
- Fraction literals: `expr!(1/2)` creates an exact rational `1/2`
- Nested expressions: `expr!(sin(x^2 + 1))`
- Formal derivatives: `expr!(diff(y, x))` creates an unevaluated derivative node

### Manual Construction

You can also build expressions using method calls:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let f = x.powi(2) + &x * 2 + 1;        // x² + 2x + 1
let g = x.sin().powi(2) + x.cos().powi(2); // sin²(x) + cos²(x)
let h = x.exp() + x.ln();                // eˣ + ln(x)
```

### Constants and Rationals

```rust
let zero = symplex::int(0);
let five = symplex::int(5);
let half = symplex::rational(1, 2);  // exact 1/2
let third = symplex::rational(1, 3); // exact 1/3

// Convenience functions
let half = symplex::half();           // 1/2
let third = symplex::third();         // 1/3
let quarter = symplex::quarter();     // 1/4

// Mathematical constants
let pi = symplex::pi();               // π
let e = symplex::e();                 // Euler's number e
let i = symplex::i_unit();            // imaginary unit i
let inf = symplex::infinity();        // +∞
let neg_inf = symplex::neg_infinity(); // -∞
```

## Display and LaTeX

Every expression implements `Display`:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let f = expr!(x^2 + 2*x + 1);
println!("f(x) = {f}");        // f(x) = x^2 + 2*x + 1

let g = expr!(sin(x)^2);
println!("g(x) = {g}");        // g(x) = sin(x)^2
```

For publication-quality output, use `.to_latex()`:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let f = expr!(x^2 + 2*x + 1);
println!("{}", f.to_latex());   // x^{2} + 2 x + 1

let g = expr!(sin(x)^2);
println!("{}", g.to_latex());   // \sin^{2}\left(x\right)
```

There are also wrappers for inline and display math modes:

```rust
println!("{}", f.to_latex_inline());   // $x^{2} + 2 x + 1$
println!("{}", f.to_latex_display());  // $$x^{2} + 2 x + 1$$
```

Matrices have their own LaTeX rendering:

```rust
use symplex::prelude::*;

let m = matrix![[1, 2], [3, 4]];
println!("{}", m.to_latex());
// \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
```

## Substitution

### Single Substitution

Replace a variable with another expression using `.subs()`:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let f = expr!(x^2 + 1);
let at_3 = f.subs(&x, &symplex::int(3));
println!("{at_3}"); // 10

// Substitute with another expression
let g = f.subs(&x, &expr!(x + 1));
println!("{g}"); // (x + 1)^2 + 1
```

### Integer Substitution Shorthand

For the common case of substituting an integer:

```rust
let result = f.subs_i64(&x, 5);
println!("{result}"); // 26
```

### Multiple Simultaneous Substitutions

Replace several variables at once with `.subs_map()`:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x, y);

let f = expr!(x^2 + y^2);
let result = f.subs_map(&[
    (&x, &symplex::int(3)),
    (&y, &symplex::int(4)),
]);
println!("{result}"); // 25
```

Note: `.subs_map()` applies all substitutions simultaneously — earlier substitutions don't affect later ones.

## Numerical Evaluation

### Quick `f64` Evaluation

For expressions with free variables, use `.eval_f64_with()`:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let f = expr!(sin(x) + cos(x));
let val = f.eval_f64_with(&[(&x, 1)]).unwrap();
println!("{val:.6}"); // 1.381773
```

The argument is a slice of `(&Ex, i64)` pairs — variable references mapped to integer values. This converts each integer to `f64` internally.

For expressions that are already fully numeric (no free variables), use `.eval_f64()`:

```rust
use symplex::prelude::*;

let pi = symplex::default_context().pi();
let val = pi.sin().eval_f64().unwrap();
println!("{val}"); // approximately 0.0 (within floating-point precision)
```

### Arbitrary-Precision Evaluation

Need more than 16 digits? Use `.eval_decimal()`:

```rust
use symplex::prelude::*;

let pi = symplex::default_context().pi();
let s = pi.eval_decimal(50).unwrap();
println!("π = {s}");
// π = 3.14159265358979323846264338327950288419716939937510
```

This uses arbitrary-precision arithmetic internally and returns a decimal string.

### Complex Evaluation

For expressions involving the imaginary unit:

```rust
use symplex::prelude::*;

let i = symplex::i_unit();
let expr = &i.powi(2);          // i² = -1
let (re, im) = expr.eval_complex64().unwrap();
println!("({re}, {im})");       // (-1.0, 0.0)
```

## Compiled Functions

For repeated numerical evaluation in tight loops, `.compile()` converts an expression into a native Rust closure — no expression-tree walking at evaluation time:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let f = expr!(x^2 + 2*x + 1);
let compiled = f.compile(&["x"]).expect("compilation failed");

// Now `compiled` is a Box<dyn Fn(&[f64]) -> f64 + Send + Sync>
let result = compiled(&[3.0]);
println!("{result}"); // 16.0

// Use it in a hot loop
for i in 0..1000 {
    let val = compiled(&[i as f64 * 0.001]);
    // ... use val ...
}
```

The variable names passed to `.compile()` determine the positional mapping: `&["x"]` means `args[0]` is `x`. For multivariate functions:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x, y);

let f = expr!(x^2 + y^2);
let compiled = f.compile(&["x", "y"]).unwrap();
let result = compiled(&[3.0, 4.0]); // 25.0
```

## Code Generation

The most powerful evaluation method: generate a standalone Rust function as source code.

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let f = expr!(x^2 + 2*x + 1);
let code = f.to_rust_fn("quadratic", &["x"]).unwrap();
println!("{code}");
```

This prints something like:

```rust
#[must_use]
pub fn quadratic(x: f64) -> f64 {
    x * x + 2.0 * x + 1.0
}
```

The generated code includes common subexpression elimination (CSE) for complex expressions — shared subexpressions are computed once and reused. See [Chapter 8: Code Generation](08-code-generation.md) for the full story.

## Putting It All Together

Here's a complete mini-workflow: define a function, find its derivative, solve for critical points, evaluate, and generate code:

```rust
use symplex::prelude::*;
use symplex::vars;

fn main() {
    vars!(x);

    // Define a polynomial
    let f = expr!(x^3 - 3*x^2 + 2*x);
    println!("f(x) = {f}");

    // Differentiate
    let df = f.diff(&x);
    println!("f'(x) = {df}");

    // Find where f'(x) = 0
    let critical = df.solve_or_empty(&x);
    for pt in &critical {
        let val = f.eval_f64_with(&[(&x, 1)]).unwrap();
        println!("f'({pt}) = 0, f({pt}) ≈ {val:.4}");
    }

    // Integrate
    let anti = f.integrate(&x);
    println!("∫f dx = {anti}");

    // Definite integral
    let area = f.definite_integral(&x, &symplex::int(0), &symplex::int(1));
    println!("∫₀¹ f dx = {area}");

    // LaTeX
    println!("LaTeX: {}", df.to_latex());

    // Generate code
    let code = df.to_rust_fn("df", &["x"]).unwrap();
    println!("\nGenerated:\n{code}");
}
```

## Next Steps

Now that you know how to create, display, substitute, evaluate, and generate code from expressions, head to [Chapter 3: Common Gotchas](03-gotchas.md) to learn about the sharp edges you'll want to avoid.

---

*[← Chapter 1: Introduction](01-introduction.md) | [Back to Table of Contents](index.md) | [Chapter 3: Common Gotchas →](03-gotchas.md)*