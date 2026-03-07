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

### [Chapter 9: Complex Numbers](09-complex-numbers.md)
The imaginary unit, polar form, Euler's formula, complex arithmetic, and simplification of complex expressions.

### [Chapter 10: Assumptions](10-assumptions.md)
Declaring that a variable is real, positive, integer, or bounded — and how assumptions guide simplification and solving.

### [Chapter 11: Sets and Intervals](11-sets-and-intervals.md)
Finite sets, intervals, set operations (union, intersection, complement), and solution sets from equation solving.

### [Chapter 12: Quaternions](12-quaternions.md)
Quaternion algebra for 3D rotations: construction, multiplication, conjugation, rotation of vectors, and conversion to/from rotation matrices.

### [Chapter 13: Control Systems](13-control-systems.md)
State-space models, transfer functions, pole-zero analysis, Routh-Hurwitz stability, Ackermann pole placement, ZOH discretization, and Laplace transforms — a complete controls engineering workflow.

### [Chapter 14: Robotics](14-robotics.md)
DH parameters, forward kinematics, Jacobians, inverse kinematics via Gröbner bases, Lagrangian dynamics (mass matrix, Coriolis, gravity), and Rust code generation for real-time control loops.

### [Chapter 15: ODE Solving](15-ode-solving.md)
Symbolic ODE classification and solving: separable, linear, exact, Bernoulli, Euler-Cauchy, second-order constant-coefficient, variation of parameters, and linear ODE systems.

### [Chapter 16: Number Theory](16-number-theory.md)
Primality testing, integer factorization, divisors, GCD/LCM, modular arithmetic, Chinese Remainder Theorem, and discrete logarithms.

### [Chapter 17: Vector Calculus](17-vector-calculus.md)
Gradient, divergence, curl, Laplacian, and line/surface integrals in Cartesian, cylindrical, and spherical coordinates.

### [Chapter 18: Formal Power Series](18-formal-power-series.md)
Formal power series with closed-form coefficient formulas for known functions (exp, sin, cos, ln, binomial), lazy coefficient extraction, and truncation to polynomials.

### [Chapter 19: Optimization](19-optimization.md)
Symbolic optimization: critical points, Lagrange multipliers, KKT conditions, and convexity analysis.

### [Chapter 20: Data Export and Visualization](20-data-export.md)
Exporting expressions and numerical data to CSV, JSON, LaTeX tables, HTML, and Markdown for reports and plotting.

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