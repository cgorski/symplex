# Chapter 20: Finite Differences

Finite difference methods approximate derivatives from discrete data — essential when you have tabulated values instead of a closed-form expression, or when you need to convert symbolic derivatives into numerical stencils. Symplex implements the Fornberg algorithm for computing exact rational weights on arbitrary grids.

## The Idea

A derivative is a limit:

```text
f'(x) = lim_{h→0} (f(x+h) - f(x)) / h
```

For a finite (nonzero) step size `h`, this becomes an *approximation*. The question is: given a set of grid points, what linear combination of function values best approximates the derivative? Fornberg's algorithm (1988) answers this exactly, producing optimal weights for any derivative order on any grid.

## Computing Weights with `finite_diff_weights`

`finite_diff_weights` is the core function. It takes a derivative order, a list of grid points, and the evaluation point — all as symbolic `ExprId` values in the arena — and returns exact rational weights.

The return type is a 3D structure `result[m][n]` where:
- `m` is the derivative order (0 through the requested order)
- `n` is the number of grid points used (0 through N)
- `result[m][n]` is a `Vec<ExprId>` of weights, one per grid point

The final weights for derivative order `m` using all grid points are in `result[m][N]` where `N = x_list.len() - 1`.

### Forward Difference (First Derivative)

The simplest finite difference uses two points: `x₀` and `x₀ + h`.

```rust
use symplex::prelude::*;

let ctx = Context::new();
fn main() {
    let x = ctx.var("x");
    let h = ctx.var("h");

    // Two-point forward difference for f'(x)
    // Grid: [x, x+h], evaluate at x
    // Weights: [-1/h, 1/h]
    // Approximation: f'(x) ≈ (-f(x) + f(x+h)) / h

    // This is the classic formula:
    //   f'(x) ≈ (f(x+h) - f(x)) / h
    println!("Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h");
    println!("  Error: O(h)  — first-order accurate");
}
```

### Central Difference (First Derivative)

Using three symmetric points gives a more accurate formula:

```rust
use symplex::prelude::*;

fn main() {
    // Three-point central difference for f'(x)
    // Grid: [x-h, x, x+h], evaluate at x
    // Weights: [-1/(2h), 0, 1/(2h)]
    // Approximation: f'(x) ≈ (-f(x-h) + f(x+h)) / (2h)

    println!("Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)");
    println!("  Error: O(h²) — second-order accurate");
    println!("  Note: the weight at the center point is exactly 0");
}
```

### Second Derivative Stencil

The classic three-point second derivative formula:

```rust
use symplex::prelude::*;

fn main() {
    // Three-point central difference for f''(x)
    // Grid: [x-h, x, x+h], evaluate at x
    // Weights: [1/h², -2/h², 1/h²]
    // Approximation: f''(x) ≈ (f(x-h) - 2f(x) + f(x+h)) / h²

    println!("Second derivative: f''(x) ≈ (f(x-h) - 2f(x) + f(x+h)) / h²");
    println!("  Error: O(h²) — second-order accurate");
}
```

## Applying Weights to Data

`apply_finite_diff` combines the weight computation and application in one step. Given grid points `x_list`, function values `y_list`, a derivative order, and an evaluation point `x0`, it computes `Σ wᵢ · yᵢ`:

```rust
use symplex::prelude::*;

fn main() {
    // Approximate f'(1) for f(x) = x² using three points
    // Points: x = [0, 1, 2], y = [0, 1, 4]
    //
    // The exact derivative is f'(x) = 2x, so f'(1) = 2.
    //
    // With the Fornberg weights on this grid, the central
    // difference gives: (-0 + 4) / 2 = 2  — exact for polynomials!

    // In symplex, this is done symbolically through the arena.
    // The result is an exact rational expression.
    println!("f(x) = x² on grid [0, 1, 2]");
    println!("f'(1) ≈ Σ wᵢ·f(xᵢ) = 2  (exact for degree ≤ 2)");
}
```

The key insight: Fornberg weights on N+1 points give exact derivatives for polynomials of degree ≤ N. For non-polynomial functions, you get an approximation whose error shrinks as you add points or reduce the step size.

## Equispaced Grids

The `equispaced_grid` helper creates a symmetric grid centered at `x0`:

```rust
use symplex::prelude::*;

fn main() {
    // equispaced_grid(arena, x0, h, half_width) creates
    // [x0 - half_width*h, ..., x0, ..., x0 + half_width*h]
    //
    // half_width = 1 → 3 points: [x-h, x, x+h]
    // half_width = 2 → 5 points: [x-2h, x-h, x, x+h, x+2h]

    // More points = higher-order accuracy for the same derivative
    println!("5-point first derivative stencil:");
    println!("  f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)");
    println!("  Error: O(h⁴) — fourth-order accurate");
}
```

## Symbolic Differentiation with Finite Differences

`Ex::differentiate_finite(&var)` replaces formal derivative nodes in an expression with central finite difference approximations. When it encounters `Derivative(f, x)`, it substitutes:

```text
f'(x) ≈ (f(x + h/2) - f(x - h/2)) / h
```

where `h` is introduced as the symbol `_h`.

```rust
use symplex::prelude::*;

let ctx = Context::new();
let x = ctx.var("x");

// Create a formal derivative: d/dx(x²)
let formal = x.powi(2).formal_diff(&x);
println!("Formal: {formal}"); // Derivative(x^2, x)

// Replace with finite difference approximation
let finite = formal.differentiate_finite(&x);
println!("Finite: {finite}");
// ((x + _h/2)^2 - (x - _h/2)^2) / _h

// The result contains the step-size symbol _h.
// You can substitute a concrete value:
let h = ctx.var("_h");
let concrete = finite.subs(&h, &ctx.rational(1, 100));
println!("With h = 0.01: {concrete}");
```

This is useful for converting analytical ODE formulations into numerical schemes:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

// Build an ODE: y' + 2y = 0
let dy = y.formal_diff(&x);
let ode = &dy + &(&y * 2);
println!("ODE: {ode} = 0");

// Convert to a finite difference equation
let fd_ode = ode.differentiate_finite(&x);
println!("FD form: {fd_ode} = 0");
// This gives you a recurrence relation in terms of y(x+h/2),
// y(x-h/2), and the step size _h.
```

## Approximating Derivatives from Tabulated Data

Here's a complete workflow: given tabulated data, approximate derivatives using Fornberg weights.

```rust
use symplex::prelude::*;

fn main() {
    // Suppose we have measurements of position vs. time:
    //   t = [0.0, 0.1, 0.2, 0.3, 0.4]
    //   x = [0.0, 0.0998, 0.1987, 0.2955, 0.3894]
    //
    // These look like x(t) ≈ sin(t). Let's estimate the velocity
    // x'(t) at t = 0.2 using the central three points.
    //
    // Central difference with h = 0.1:
    //   x'(0.2) ≈ (x(0.3) - x(0.1)) / (2 × 0.1)
    //           = (0.2955 - 0.0998) / 0.2
    //           = 0.9785
    //
    // Exact: cos(0.2) = 0.9801
    // Error: ~0.16% — good for such coarse data!

    let v_approx = (0.2955 - 0.0998) / 0.2;
    let v_exact = 0.2_f64.cos();
    println!("Estimated velocity at t=0.2: {v_approx:.4}");
    println!("Exact velocity (cos 0.2):    {v_exact:.4}");
    println!("Relative error: {:.4}%", (v_approx - v_exact).abs() / v_exact * 100.0);

    // Using all five points would give a fourth-order estimate,
    // reducing the error further. That's what apply_finite_diff
    // does internally with the Fornberg weights.
}
```

## The Fornberg Algorithm

Under the hood, Fornberg's algorithm fills a 3D recurrence table. The key properties:

1. **Arbitrary grids** — points don't need to be equally spaced
2. **Exact rational arithmetic** — weights are computed symbolically, no floating-point error
3. **All orders at once** — a single call computes weights for derivatives 0 through the requested order
4. **Optimal** — for N+1 points, the weights are exact for polynomials of degree ≤ N

The zeroth-order "derivative" weights are actually interpolation weights — they reconstruct the function value at `x0` from the grid values. This makes the same algorithm useful for polynomial interpolation.

## What's Next

Several finite difference features are planned for future releases:

```rust
// Higher-order stencils with automatic order selection
// let weights = finite_diff::optimal_stencil(order, accuracy, h);

// Non-uniform grid support with error estimation
// let (deriv, error_bound) = finite_diff::apply_nonuniform(
//     order, &x_irregular, &y_values, x0
// );

// Richardson extrapolation for improved accuracy
// let improved = finite_diff::richardson_extrapolate(
//     |h| central_diff(f, x, h), h0, num_refinements
// );
```

---

*[← Chapter 19: Data Export](19-data-export.md) | [Back to Table of Contents](index.md)*