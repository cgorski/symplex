# Chapter 16: Complex Numbers

Symplex has first-class support for the imaginary unit and complex arithmetic. Complex numbers arise naturally when solving polynomials, applying Euler's formula, and bridging trigonometric and hyperbolic functions.

## The Imaginary Unit

The imaginary unit `I` (where I² = −1) is available as a function or through the `expr!` macro:

```rust
use symplex::prelude::*;

let i = symplex::i_unit();
println!("I = {i}");

// Powers of I cycle with period 4
let i2 = i.powi(2).eval();
let i3 = i.powi(3).eval();
let i4 = i.powi(4).eval();
println!("I² = {i2}");   // -1
println!("I³ = {i3}");   // -I
println!("I⁴ = {i4}");   // 1

// In the expr! macro, I is a recognized constant
let z = expr!(3 + 4*I);
println!("z = {z}"); // 3 + 4*I
```

## Complex Arithmetic

Build complex expressions with standard operators:

```rust
use symplex::prelude::*;

let i = symplex::i_unit();

let z1 = &symplex::int(2) + &(&i * 3);  // 2 + 3I
let z2 = &symplex::int(1) - &(&i * 2);  // 1 - 2I

let sum = (&z1 + &z2).eval();
println!("z₁ + z₂ = {sum}"); // 3 + I

// (2+3I)(1-2I) = 2 - 4I + 3I - 6I² = 2 - I + 6 = 8 - I
let product = (&z1 * &z2).expand().eval();
println!("z₁ · z₂ = {product}"); // 8 - I
```

## Euler's Formula

Euler's formula — `exp(I·x) = cos(x) + I·sin(x)` — is one of the deepest connections in mathematics. Symplex knows it:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);
let i = symplex::i_unit();

// Rewrite exp(I·x) as trig functions
let eix = (&i * &x).exp();
let as_trig = eix.rewrite_as_trig();
println!("exp(I·x) = {as_trig}"); // cos(x) + I·sin(x)

// Euler's identity: exp(I·π) + 1 = 0
let pi = symplex::default_context().pi();
let euler = &(&i * &pi).exp() + 1;
let result = euler.eval().simplify();
println!("exp(I·π) + 1 = {result}"); // 0
```

You can also go the other direction — rewrite trig as exponentials:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let sin_as_exp = x.sin().rewrite_as_exp();
println!("sin(x) = {sin_as_exp}"); // (exp(I*x) - exp(-I*x))/(2*I)

let cos_as_exp = x.cos().rewrite_as_exp();
println!("cos(x) = {cos_as_exp}"); // (exp(I*x) + exp(-I*x))/2
```

## Complex Quadratic Roots

Symplex's solver returns complex roots when the discriminant is negative:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

// x² + 1 = 0 → x = I, -I
let roots = expr!(x^2 + 1).solve_or_empty(&x);
println!("x² + 1 = 0:");
for r in &roots {
    println!("  x = {r}");
}

// x² + 2x + 5 = 0 → x = -1 + 2I, -1 - 2I
let roots = expr!(x^2 + 2*x + 5).solve_or_empty(&x);
println!("\nx² + 2x + 5 = 0:");
for r in &roots {
    println!("  x = {r}");
}

// x⁴ - 1 = 0 → x = 1, -1, I, -I
let roots = expr!(x^4 - 1).solve_or_empty(&x);
println!("\nx⁴ - 1 = 0:");
for r in &roots {
    println!("  x = {r}");
}
```

## Real and Imaginary Parts

`.re()` and `.im()` decompose an expression into its real and imaginary components. Unadorned symbols are assumed real:

```rust
use symplex::prelude::*;

let i = symplex::i_unit();
let z = &symplex::int(3) + &(&symplex::int(4) * &i);

println!("re(3 + 4I) = {}", z.re()); // 3
println!("im(3 + 4I) = {}", z.im()); // 4
```

The decomposition works on symbolic expressions too:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);
let i = symplex::i_unit();

let eix = (&i * &x).exp();
let re = eix.re();
let im = eix.im();
println!("re(exp(I·x)) = {re}"); // cos(x)
println!("im(exp(I·x)) = {im}"); // sin(x)
```

## Complex Conjugate

`.conjugate()` computes the complex conjugate — if `z = a + bi`, then `conjugate(z) = a - bi`:

```rust
use symplex::prelude::*;

let i = symplex::i_unit();
let z = &symplex::int(3) + &(&symplex::int(4) * &i);
let z_bar = z.conjugate();
println!("conjugate(3 + 4I) = {z_bar}"); // 3 - 4*I

// z · conjugate(z) = |z|² = a² + b² (always real)
let product = (&z * &z_bar).expand().eval();
println!("(3+4I)(3-4I) = {product}"); // 25
```

## Numerical Complex Evaluation

`.eval_complex64()` returns a `Result<(f64, f64), SymplexError>` pair of (real, imaginary):

```rust
use symplex::prelude::*;

let i = symplex::i_unit();

// Pure imaginary
let (re, im) = i.eval_complex64().unwrap();
println!("I → ({re}, {im})"); // (0, 1)

// I² = -1
let (re, im) = i.powi(2).eval().eval_complex64().unwrap();
println!("I² → ({re}, {im})"); // (-1, 0)

// 3 + 4I
let z = &symplex::int(3) + &(&symplex::int(4) * &i);
if let Ok((re, im)) = z.eval_complex64() {
    println!("3 + 4I → ({re}, {im})"); // (3, 4)
}

// Numerical values of complex roots
let x = symplex::var("x");
let roots = expr!(x^2 + 2*x + 5).solve_or_empty(&x);
for r in &roots {
    if let Ok((re, im)) = r.eval_complex64() {
        if im.abs() > 1e-10 {
            println!("  {re:.6} + {im:.6}i");
        } else {
            println!("  {re:.6}");
        }
    }
}
// -1.000000 + 2.000000i
// -1.000000 + -2.000000i
```

## Trig–Hyperbolic Bridge

Complex numbers connect circular and hyperbolic functions. These identities follow directly from Euler's formula:

- `cos(I·x) = cosh(x)`
- `sin(I·x) = I·sinh(x)`

You can verify this numerically in symplex:

```rust
use symplex::prelude::*;

let i = symplex::i_unit();

// cos(2I) should equal cosh(2) ≈ 3.7622
let cos_2i = (&i * 2).cos().eval();
if let Ok((re, im)) = cos_2i.eval_complex64() {
    println!("cos(2I) = ({re:.6}, {im:.6})");
    println!("cosh(2)  = {:.6}", 2.0_f64.cosh());
    // Both ≈ 3.762196
}
```

## Limitation: Code Generation

The code generation pipeline (`to_rust_fn()` and related methods) does **not** support complex-valued expressions. If you attempt to generate Rust code from an expression containing `I`, the result will be incorrect or the call will fail. This is a known limitation — complex codegen requires emitting `num::Complex<f64>` types, which is not yet implemented.

```rust
use symplex::prelude::*;

let i = symplex::i_unit();
let z = &symplex::int(3) + &(&symplex::int(4) * &i);

// This will NOT produce correct code — complex codegen is unsupported
// let code = z.to_rust_fn("complex_val", &[]);
```

Workaround: decompose into real and imaginary parts, generate code for each part separately, and recombine in your numerical code.

## What's Next

Several complex number features are planned for future releases:

```rust
// Polar form: r·exp(I·θ)
// let (r, theta) = z.to_polar();

// Principal argument
// let angle = z.arg_principal(); // returns angle in (-π, π]

// Principal branch for multi-valued functions
// let w = z.sqrt_principal(); // principal square root

// Complex-aware simplification
// let simplified = expr.simplify_complex();
```

---

*[← Chapter 9: Number Theory](09-number-theory.md) | [Back to Table of Contents](index.md) | [Chapter 19: Data Export →](19-data-export.md)*