# Chapter 1: Introduction

## What Is Symbolic Math?

Most programmers are used to **numerical** math — you pick a type like `f64`, do some arithmetic, and get an approximate answer. Symbolic math is different: you work with *expressions* as structured objects, manipulating them algebraically before ever plugging in numbers.

Here's the moment that makes it click. In numerical Rust:

```rust
let x: f64 = 8.0_f64.sqrt();
println!("{x}"); // 2.8284271247461903
```

That's a 16-digit approximation. In symplex:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);
let result = symplex::int(8).sqrt().simplify();
println!("{result}"); // 2*sqrt(2)
```

The answer is **exact**: `2√2`. No rounding, no truncation, no accumulated error. The library knows that `√8 = √(4·2) = 2√2` and rewrites it symbolically.

## The Floating-Point Problem

Every engineer has run into this:

```rust
let a: f64 = 0.1 + 0.2;
println!("{a}");          // 0.30000000000000004
println!("{}", a == 0.3); // false
```

IEEE 754 doubles cannot represent `0.1` or `0.2` exactly. The error is tiny — about 5 × 10⁻¹⁷ — but it compounds across long computations, and it makes equality testing unreliable.

In symplex, rationals are exact:

```rust
use symplex::prelude::*;

let a = symplex::rational(1, 10); // exactly 1/10
let b = symplex::rational(2, 10); // exactly 2/10 = 1/5
let sum = &a + &b;
println!("{sum}"); // 3/10
```

The result is `3/10` — exactly, with no representation error. The `expr!` macro makes this even more natural:

```rust
use symplex::prelude::*;

let sum = expr!(1/10 + 2/10);
println!("{sum}"); // 3/10
```

When you detect `a/b` where both `a` and `b` are integer literals, `expr!` automatically constructs a rational number. No floating-point involved.

## A Taste of Symplex

Here's what ten lines of symplex look like in practice:

```rust
use symplex::prelude::*;
use symplex::vars;

fn main() {
    vars!(x);

    // Build a polynomial
    let f = expr!(x^3 - 3*x^2 + 2*x);
    println!("f(x) = {f}");           // x^3 - 3*x^2 + 2*x

    // Differentiate
    let df = f.diff(&x);
    println!("f'(x) = {df}");         // 3*x^2 - 6*x + 2

    // Find critical points
    let critical = df.solve_or_empty(&x);
    for pt in &critical {
        println!("critical point: x = {pt}");
    }

    // Integrate
    let anti = f.integrate(&x);
    println!("∫f dx = {anti}");        // 1/4*x^4 - x^3 + x^2

    // Simplify a trig identity
    let identity = expr!(sin(x)^2 + cos(x)^2);
    println!("{identity} = {}", identity.simplify()); // 1

    // Generate Rust code
    let code = df.to_rust_fn("df", &["x"]).unwrap();
    println!("{code}");
    // Outputs a complete `pub fn df(x: f64) -> f64 { ... }`
}
```

That's the workflow: **build** an expression, **transform** it symbolically (differentiate, simplify, factor, solve), then optionally **generate code** for numerical evaluation.

## Why Symplex?

There are mature computer algebra systems in the world — SymPy, Mathematica, Maple. Why another one in Rust?

### Exact by Default

Every operation preserves exactness until you explicitly ask for a float. Rationals stay rational. Square roots stay symbolic. `π` is `π`, not `3.14159…`.

### Type-Safe

`Ex` (the expression type) is a regular Rust type with a well-defined API. No stringly-typed interfaces. No runtime "is this a matrix or a scalar?" checks. The compiler catches misuse before your code runs.

### Thread-Safe

`Ex` is `Send + Sync`. You can build expressions on one thread and simplify them on another. There's no global interpreter lock, no garbage collector pauses. Parallel workloads — computing Jacobians for multiple robot configurations, for instance — just work.

### Rust Code Generation

This is the killer feature. Symplex can take a symbolic expression — say, a 6-DOF robot Jacobian derived from DH parameters — and emit an optimized Rust function with common subexpression elimination. You derive once symbolically, then generate code that runs at full compiled speed in your real-time control loop.

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);
let f = expr!(x^2 + 2*x + 1);
let code = f.to_rust_fn("quadratic", &["x"]).unwrap();
println!("{code}");
```

This prints a complete, compilable Rust function. No runtime overhead. No expression-tree walking at evaluation time. Just `f64` arithmetic.

### Build-Script Integration

The `symplex-build` crate (planned) will let you derive symbolic expressions in `build.rs` and emit generated Rust source files at compile time. Your symbolic math happens *once*, at build time, and your final binary contains only the optimized numerical code.

## Who Is Symplex For?

**Robotics engineers** who derive kinematics and dynamics symbolically, then need fast numerical code for real-time control.

**Controls engineers** who work with state-space models, transfer functions, and stability analysis — and want exact pole locations, not floating-point approximations.

**Physics researchers** who solve ODEs, compute series expansions, and need results they can trust across coordinate transforms.

**Compiler and language researchers** who want a CAS they can embed, extend, and reason about with Rust's type system.

**Anyone** who's been burned by floating-point rounding and wants exact answers.

## What Symplex Is Not

Symplex is not a replacement for numerical linear algebra libraries like `nalgebra` or `ndarray`. Those are designed for fast numerical computation on large dense matrices. Symplex's `Matrix` type is for small, symbolic matrices — the kind you encounter in kinematics (4×4 homogeneous transforms) and controls (state-space models with a handful of states).

Symplex is not a general-purpose plotting tool or a notebook environment. It's a Rust library: you write code, compile it, and run it.

Symplex is young. It handles polynomials through degree 4, common transcendental equations, and a growing set of ODE types. It does not yet have the breadth of a system like SymPy (which has had 20+ years of development). See the [SymPy comparison](../sympy-comparison.md) for an honest assessment.

## Next Steps

Ready to write some code? Head to [Chapter 2: Getting Started](02-getting-started.md) to install symplex and build your first expressions.

---

*[← Back to Table of Contents](index.md) | [Chapter 2: Getting Started →](02-getting-started.md)*