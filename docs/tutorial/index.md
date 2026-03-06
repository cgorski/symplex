# Symplex Tutorial

Welcome to the **symplex** tutorial — a guided tour through symbolic mathematics in Rust.

Symplex lets you build, transform, and solve mathematical expressions exactly — no floating-point surprises, no silent precision loss. Whether you're deriving robot kinematics, simplifying control-system equations, or generating optimized Rust code from symbolic formulas, this tutorial will get you productive quickly.

## Table of Contents

### [Chapter 1: Introduction](01-introduction.md)
What is symbolic math? Why does it matter? A taste of what symplex can do in ten lines of Rust.

### [Chapter 2: Getting Started](02-getting-started.md)
Installation, creating variables and expressions, display and LaTeX output, substitution, numerical evaluation, and compiled functions.

### [Chapter 3: Common Gotchas](03-gotchas.md)
Fraction literals, `Clone` vs `Copy`, structural vs mathematical equality, the `^` operator, and when `simplify()` isn't enough.

### [Chapter 4: Simplification](04-simplification.md)
The full simplification toolkit: `simplify()`, `full_simplify()`, `expand()`, `factor()`, trig identities, logarithm rules, partial fractions, and more.

### [Chapter 5: Calculus](05-calculus.md)
Differentiation, integration, definite integrals, limits, Taylor/Maclaurin series, implicit differentiation, and Laplace transforms.

### [Chapter 6: Solving Equations](06-solving.md)
Polynomial roots through quartic, transcendental equations, systems via Gröbner bases, numerical fallback, inequality solving, and ODE solving.

### [Chapter 7: Matrices](07-matrices.md)
Construction, determinants, inverses, eigenvalues, LU/QR decomposition, RREF, rank, nullspace, Jacobians, and matrix LaTeX.

### [Chapter 8: Code Generation](08-code-generation.md)
**The killer feature.** Derive symbolically, generate optimized Rust code, compile, and run. DH parameters → forward kinematics → Jacobian → codegen. Options for `no_std`, embedded `f32`, and cross-entry CSE.

---

## Prerequisites

- Rust 1.93.0 or later (the minimum supported Rust version)
- Basic familiarity with Rust syntax (`let`, references, `use`)
- Some comfort with algebra and calculus (the math, not the library)

## Conventions

Throughout this tutorial, all examples assume:

```rust
use symplex::prelude::*;
use symplex::vars;
```

Code blocks show the expected output in comments where helpful. Every example in this tutorial uses the real symplex API and can be pasted into a Rust file to run.