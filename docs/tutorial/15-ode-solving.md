# Chapter 15: ODE Solving

Symplex can classify and solve ordinary differential equations symbolically — no numerical stepping, no discretization error. You get closed-form solutions with arbitrary constants, ready for initial-condition substitution or code generation.

This chapter covers the full range of ODE types that symplex handles, from trivial separable equations to second-order systems solved by variation of parameters.

## The ODE API

The workflow for every ODE is the same:

1. **Build** the ODE as an expression equal to zero, using `formal_diff` to create derivative nodes
2. **Classify** it with `.classify_ode(&y, &x)` to see what type symplex recognizes
3. **Solve** it with `.solve_ode(&y, &x)` to get the general solution and integration constants
4. **Verify** the solution with `.check_ode_solution(&particular, &y, &x)`

```rust
use symplex::prelude::*;
use symplex::ode::OdeType;
use symplex::vars;

vars!(x, y);

// Build: y' - x = 0  (i.e., y' = x)
let dy = y.formal_diff(&x);
let ode = &dy - &x;

// Classify
let ode_type = ode.classify_ode(&y, &x);
println!("Type: {ode_type:?}");
// Type: SimpleSeparable

// Solve
if let Some((sol, constants)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
    // y = x^2/2 + C1
    println!("Constants: {:?}",
        constants.iter().map(|c| format!("{c}")).collect::<Vec<_>>()
    );
}

// Verify: does y = x²/2 satisfy y' = x?
let particular = &x.powi(2) / 2;
let verified = ode.check_ode_solution(&particular, &y, &x);
println!("Verified: {verified}");
// Verified: true
```

### Creating Derivative Nodes

The key is `formal_diff`. Unlike `.diff()`, which computes the derivative, `.formal_diff()` creates an unevaluated derivative node — a symbolic placeholder for $y'$:

```rust
vars!(x, y);

let dy = y.formal_diff(&x);       // Derivative(y, x) — represents y'
let d2y = dy.formal_diff(&x);     // Derivative(Derivative(y, x), x) — represents y''

// Or use the expr! macro
let dy_macro = expr!(diff(y, x));  // same as y.formal_diff(&x)
```

For second-order ODEs, chain two `formal_diff` calls. The ODE solver recognizes nested derivative nodes and handles them accordingly.

## First-Order ODEs

### SimpleSeparable: y' = f(x)

The simplest case — the right-hand side depends only on $x$, so the solution is a direct integral:

```rust
vars!(x, y);

// y' = x  →  y = x²/2 + C1
let ode = expr!(diff(y, x) - x);
assert_eq!(ode.classify_ode(&y, &x), OdeType::SimpleSeparable);

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}

// y' = sin(x)  →  y = -cos(x) + C1
let ode2 = &y.formal_diff(&x) - &x.sin();
if let Some((sol, _)) = ode2.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

### FirstOrderLinearCC: y' + ay = 0

First-order linear with constant coefficients — the exponential decay/growth equation:

```rust
vars!(x, y);

// y' + 2y = 0  →  y = C1·exp(-2x)
let ode = expr!(diff(y, x) + 2 * y);
assert_eq!(ode.classify_ode(&y, &x), OdeType::FirstOrderLinearCC);

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
    // y = C1*exp(-2*x)
}

// Verify: y = exp(-2x) satisfies y' + 2y = 0
let particular = (-&x * 2).exp();
let verified = ode.check_ode_solution(&particular, &y, &x);
println!("Verified: {verified}");
// Verified: true
```

### FirstOrderLinearVC: y' + P(x)y = Q(x)

When the coefficients are variable (functions of $x$), symplex uses the integrating factor method:

```rust
vars!(x, y);

// y' + (1/x)·y = x
// Integrating factor: μ = exp(∫1/x dx) = x
// Solution: y = x/2·x + C1/x  (after integrating)
let ode = &y.formal_diff(&x) + &(&y / &x) - &x;
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: FirstOrderLinearVC

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

### FullSeparable: y' = f(x)·g(y)

When the ODE separates into a product of a function of $x$ and a function of $y$:

```rust
vars!(x, y);

// y' = x·y  →  dy/y = x dx  →  ln|y| = x²/2  →  y = C1·exp(x²/2)
let ode = &y.formal_diff(&x) - &(&x * &y);
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: FullSeparable

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

### ExactFirstOrder: M(x,y) + N(x,y)·y' = 0

An exact ODE satisfies $\partial M/\partial y = \partial N/\partial x$. The solution is an implicit function $F(x,y) = C$ found by integrating $M$ with respect to $x$ and $N$ with respect to $y$:

```rust
vars!(x, y);

// (2xy + 3) + x²·y' = 0
// M = 2xy + 3, N = x²
// ∂M/∂y = 2x = ∂N/∂x  →  exact!
let dy = y.formal_diff(&x);
let ode = &(&(&x * &y * 2) + 3) + &(&x.powi(2) * &dy);
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: ExactFirstOrder

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

### HomogeneousCoefficient: y' = f(y/x)

When the right-hand side is a function of $y/x$ alone, the substitution $v = y/x$ reduces the ODE to a separable equation:

```rust
vars!(x, y);

// y' = (x + y)/x = 1 + y/x
let dy = y.formal_diff(&x);
let ode = &dy - &(&(&x + &y) / &x);
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: HomogeneousCoefficient

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

### Bernoulli: y' + P(x)·y = Q(x)·yⁿ

The Bernoulli equation is nonlinear but reducible to a linear ODE via the substitution $v = y^{1-n}$:

```rust
vars!(x, y);

// y' + y = y²  (Bernoulli with n=2)
// Substitution v = y^(-1): v' - v = -1
let dy = y.formal_diff(&x);
let ode = &(&dy + &y) - &y.powi(2);
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: Bernoulli

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

## Second-Order ODEs

### SecondOrderLinearCCHomogeneous: ay'' + by' + cy = 0

The workhorse of mechanical and electrical engineering. The solution depends on the discriminant of the characteristic equation $ar^2 + br + c = 0$:

- **Two real roots** ($b^2 > 4ac$): $y = C_1 e^{r_1 x} + C_2 e^{r_2 x}$
- **Complex conjugate roots** ($b^2 < 4ac$): $y = e^{\alpha x}(C_1\cos\beta x + C_2\sin\beta x)$
- **Repeated root** ($b^2 = 4ac$): $y = (C_1 + C_2 x) e^{rx}$

```rust
vars!(x, y);
let dy = y.formal_diff(&x);
let d2y = dy.formal_diff(&x);

// Case 1: Pure oscillation — y'' + y = 0
// Characteristic: r² + 1 = 0 → r = ±i
// Solution: y = C1·cos(x) + C2·sin(x)
let ode = &d2y + &y;
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: SecondOrderLinearCCHomogeneous

if let Some((sol, constants)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
    println!("Constants: {:?}",
        constants.iter().map(|c| format!("{c}")).collect::<Vec<_>>()
    );
}

// Verify both particular solutions
println!("cos(x) works: {}", ode.check_ode_solution(&x.cos(), &y, &x));
println!("sin(x) works: {}", ode.check_ode_solution(&x.sin(), &y, &x));
// Both: true

// Wrong solution: exp(x) does NOT satisfy y'' + y = 0
println!("exp(x) works: {}", ode.check_ode_solution(&x.exp(), &y, &x));
// false
```

```rust
// Case 2: Overdamped — y'' + 3y' + 2y = 0
// Characteristic: r² + 3r + 2 = 0 → r = -1, -2
// Solution: y = C1·exp(-x) + C2·exp(-2x)
vars!(x, y);
let dy = y.formal_diff(&x);
let d2y = dy.formal_diff(&x);

let ode = &(&d2y + &(&dy * 3)) + &(&y * 2);
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: SecondOrderLinearCCHomogeneous

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}

// Verify each fundamental solution
println!("exp(-x) works:  {}", ode.check_ode_solution(&(-&x).exp(), &y, &x));
println!("exp(-2x) works: {}", ode.check_ode_solution(&(-&x * 2).exp(), &y, &x));
```

### SecondOrderLinearCCNonHomogeneous: ay'' + by' + cy = f(x)

When there's a forcing function on the right-hand side, the solution is the sum of the homogeneous solution and a particular solution found via undetermined coefficients:

```rust
vars!(x, y);
let dy = y.formal_diff(&x);
let d2y = dy.formal_diff(&x);

// y'' + y = sin(x)
// Homogeneous: C1·cos(x) + C2·sin(x)
// Particular solution via undetermined coefficients (resonance case!)
let ode = &(&d2y + &y) - &x.sin();
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: SecondOrderLinearCCNonHomogeneous

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

This is the resonance case — the forcing frequency matches the natural frequency. The particular solution grows linearly with $x$.

### EulerCauchy: x²y'' + bxy' + cy = 0

The Euler-Cauchy (or equidimensional) equation has variable coefficients, but the substitution $x = e^t$ converts it to a constant-coefficient ODE:

```rust
vars!(x, y);
let dy = y.formal_diff(&x);
let d2y = dy.formal_diff(&x);

// x²y'' - 2y = 0
// Trial solution y = x^r → r(r-1) - 2 = 0 → r = 2, -1
// Solution: y = C1·x² + C2·x⁻¹
let ode = &(&x.powi(2) * &d2y) - &(&y * 2);
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: EulerCauchy

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

### VariationOfParameters: y'' + py' + qy = g(x)

When the forcing function doesn't fit the undetermined-coefficients template (exponentials, sines, polynomials), variation of parameters handles the general case:

```rust
vars!(x, y);
let dy = y.formal_diff(&x);
let d2y = dy.formal_diff(&x);

// y'' + y = tan(x)
// Undetermined coefficients can't handle tan(x),
// but variation of parameters can.
let ode = &(&d2y + &y) - &x.tan();
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: VariationOfParameters

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

### NthOrderReducible: F(y, y', y'') = 0

When the independent variable $x$ doesn't appear explicitly in the ODE, the substitution $p = y'$ (treating $p$ as a function of $y$) reduces the order:

```rust
vars!(x, y);
let dy = y.formal_diff(&x);
let d2y = dy.formal_diff(&x);

// y'' = y'  (no explicit x)
// Let p = y': dp/dx = p → p = C1·exp(x) → y = C1·exp(x) + C2
let ode = &d2y - &dy;
println!("Type: {:?}", ode.classify_ode(&y, &x));
// Type: NthOrderReducible

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

## Using the `expr!` Macro

You can build ODEs more concisely with the `expr!` macro, which understands `diff(y, x)`:

```rust
vars!(x, y);

let ode = expr!(diff(y, x) + y);
println!("Type: {:?}", ode.classify_ode(&y, &x));

if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
    // y = C1*exp(-x)
}
```

For second-order ODEs, nest two `diff` calls or build the derivative nodes manually as shown above.

## Classification at a Glance

Here's a quick way to classify a batch of ODEs without solving them:

```rust
vars!(x, y);
let dy = y.formal_diff(&x);
let d2y = dy.formal_diff(&x);

let odes: Vec<(&str, Ex)> = vec![
    ("y' = x",           expr!(diff(y, x) - x)),
    ("y' + 2y = 0",      expr!(diff(y, x) + 2 * y)),
    ("y' = x·y",         &dy - &(&x * &y)),
    ("y'' + y = 0",       &d2y + &y),
    ("y'' + 3y' + 2y = 0", &(&d2y + &(&dy * 3)) + &(&y * 2)),
];

for (desc, ode) in &odes {
    let cls = ode.classify_ode(&y, &x);
    let solvable = ode.solve_ode(&y, &x).is_some();
    println!("{desc:30} → {cls:?} (solvable: {solvable})");
}
```

## ODE Systems: ẋ = Ax

For linear constant-coefficient systems $\dot{\mathbf{x}} = A\mathbf{x}$, symplex solves via eigenvalue decomposition or matrix exponential series:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(t);

// ẋ = Ax where A = [[0, 1], [-2, -3]]
// This is the mass-spring-damper from Chapter 13
let a = matrix![[0, 1], [-2, -3]];

let sol = symplex::ode::solve_ode_system(&a, &t)
    .expect("system should be solvable");

println!("x₁(t) = {}", sol[0]);
println!("x₂(t) = {}", sol[1]);
// Solutions are linear combinations of C1·exp(λ₁t) + C2·exp(λ₂t)
// where λ₁, λ₂ are eigenvalues of A
```

The solver tries three strategies in order:
1. **Diagonal systems** — each component is independent: $x_i = C_i e^{a_{ii}t}$
2. **Eigenvalue decomposition** — find eigenvalues and eigenvectors for the exact fundamental matrix
3. **Matrix exponential series** — truncated Taylor expansion of $e^{At}$ as a fallback

```rust
// A diagonal system is trivial
let a_diag = matrix![[symplex::int(-1), symplex::int(0)],
                      [symplex::int(0), symplex::int(-3)]];
let sol = symplex::ode::solve_ode_system(&a_diag, &t).unwrap();
println!("x₁(t) = {}", sol[0]);  // C1*exp(-t)
println!("x₂(t) = {}", sol[1]);  // C2*exp(-3*t)
```

## Aspirational Features

The following capabilities are planned but not yet implemented:

```rust
// PLANNED: Initial conditions
// let sol = ode.solve_ode_with_ics(&y, &x, &[(0, 1), (0, 0)]);

// PLANNED: Boundary value problems
// let sol = ode.solve_bvp(&y, &x, (0.0, 1.0), (1.0, 0.0));

// PLANNED: classify_ode returning ALL applicable methods
// let methods = ode.classify_ode_all(&y, &x);
// // → [Bernoulli, FirstOrderLinearVC, ExactFirstOrder]

// PLANNED: Riccati equations: y' = P(x) + Q(x)·y + R(x)·y²
// PLANNED: Liouville equations
// PLANNED: Hypergeometric ODEs

// PLANNED: PDE solving
// let pde = expr!(diff(u, t) - diff(diff(u, x), x));  // heat equation
// let sol = pde.solve_pde(&u, &[&x, &t]);
```

## Summary

| ODE Type | Form | Method |
|----------|------|--------|
| `SimpleSeparable` | $y' = f(x)$ | Direct integration |
| `FullSeparable` | $y' = f(x)g(y)$ | Separation of variables |
| `FirstOrderLinearCC` | $y' + ay = f(x)$ | Exponential solution |
| `FirstOrderLinearVC` | $y' + P(x)y = Q(x)$ | Integrating factor |
| `ExactFirstOrder` | $M + Ny' = 0$, $M_y = N_x$ | Potential function |
| `HomogeneousCoefficient` | $y' = f(y/x)$ | Substitution $v = y/x$ |
| `Bernoulli` | $y' + Py = Qy^n$ | Substitution $v = y^{1-n}$ |
| `SecondOrderLinearCCHomogeneous` | $ay'' + by' + cy = 0$ | Characteristic equation |
| `SecondOrderLinearCCNonHomogeneous` | $ay'' + by' + cy = f(x)$ | Undetermined coefficients |
| `EulerCauchy` | $x^2y'' + bxy' + cy = 0$ | Trial $y = x^r$ |
| `VariationOfParameters` | $y'' + py' + qy = g(x)$ | Wronskian method |
| `NthOrderReducible` | $F(y, y', y'') = 0$ | Order reduction |

| API | Purpose |
|-----|---------|
| `y.formal_diff(&x)` | Create unevaluated derivative node $y'$ |
| `expr!(diff(y, x))` | Same, via macro |
| `ode.classify_ode(&y, &x)` | Identify ODE type |
| `ode.solve_ode(&y, &x)` | General solution + constants |
| `ode.check_ode_solution(&sol, &y, &x)` | Verify a particular solution |
| `symplex::ode::solve_ode_system(&A, &t)` | Solve $\dot{\mathbf{x}} = A\mathbf{x}$ |

## What's Next

For generating efficient numerical code from your symbolic ODE solutions (or any expression), revisit [Chapter 8: Code Generation](08-code-generation.md). For power series representations of solutions, continue to [Chapter 18: Formal Power Series](18-formal-power-series.md).

---

*[← Chapter 14: Robotics](14-robotics.md) | [Back to Table of Contents](index.md) | [Chapter 16: Number Theory →](16-number-theory.md)*