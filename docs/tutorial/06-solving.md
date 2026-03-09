# Chapter 6: Solving Equations

Symplex solves polynomial equations through degree 4, transcendental equations via pattern matching, polynomial systems via Gröbner bases, and ODEs through several standard classes. When symbolic methods fail, numerical fallback is available.

## Polynomial Equations

### Linear and Quadratic

The `.solve(&var)` method finds all roots of `self = 0`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// Linear: 3x + 6 = 0 → x = -2
let roots = expr!(3*x + 6).solve_or_empty(&x);
println!("3x + 6 = 0 → x = {}", roots[0]); // -2

// Quadratic: x² - 5x + 6 = 0 → x = 2, 3
let roots = expr!(x^2 - 5*x + 6).solve_or_empty(&x);
for r in &roots {
    println!("x = {r}");
}
```

`.solve()` returns `Result<Vec<Ex>, SymplexError>`. Use `.solve_or_empty()` when you don't need the error:

```rust
let ctx = Context::new();
syms!(ctx; x);

// solve() gives you the error if it fails
match expr!(x^2 + 1).solve(&x) {
    Ok(roots) => println!("roots: {}", roots.len()),
    Err(e) => println!("solver error: {e}"),
}

// solve_or_empty() returns vec![] on failure
let roots = expr!(x^2 + 1).solve_or_empty(&x);
println!("roots found: {}", roots.len());
```

### Cubic

Symplex solves cubics using the cubic formula (Cardano's method) and rational-root shortcuts:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// x³ - 6x² + 11x - 6 = 0 → x = 1, 2, 3
let roots = expr!(x^3 - 6*x^2 + 11*x - 6).solve_or_empty(&x);
println!("Cubic roots:");
for r in &roots {
    println!("  x = {r}");
}
```

### Quartic

Degree-4 polynomials are also handled:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// x⁴ - 1 = 0
let roots = expr!(x^4 - 1).solve_or_empty(&x);
println!("x⁴ - 1 = 0:");
for r in &roots {
    println!("  x = {r}");
}
// Real roots: 1, -1
// Complex roots may also appear (±i)
```

### Complex Roots

The solver may return complex-valued roots for equations like `x² + 1 = 0`. These are represented symbolically using the imaginary unit `I`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let roots = expr!(x^2 + 1).solve_or_empty(&x);
for r in &roots {
    println!("x = {r}"); // I, -I
}
```

You can evaluate complex roots numerically with `.eval_complex64()`:

```rust
for r in &roots {
    if let Ok((re, im)) = r.eval_complex64() {
        println!("  numerical: ({re:.4}, {im:.4})");
    }
}
```

### Verifying Solutions

Use `.check_solution()` to verify that a value is actually a root:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let poly = expr!(x^2 - 4);
assert!(poly.check_solution(&x, &ctx.int(2)));
assert!(poly.check_solution(&x, &ctx.int(-2)));
assert!(!poly.check_solution(&x, &ctx.int(3)));
```

This substitutes the value, evaluates, and checks if the result is zero (structurally or numerically within tolerance).

### Factoring

Related to solving: `.factor(&x)` expresses a polynomial as a product of its roots:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

println!("{}", expr!(x^2 - 5*x + 6).factor(&x));  // (x - 2)*(x - 3)
println!("{}", expr!(x^2 - 1).factor(&x));          // (x - 1)*(x + 1)
println!("{}", expr!(x^4 - 1).factor(&x));           // (x - 1)*(x + 1)*(x^2 + 1)
println!("{}", expr!(x^2 + 2*x + 1).factor(&x));     // (x + 1)^2
```

## Transcendental Equations

Some equations involving `sin`, `cos`, `exp`, etc. can be solved symbolically. The solver detects patterns and applies appropriate techniques:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// sin(x) = 1/2 → x = π/6 (principal value)
let eq = &x.sin() - &ctx.rational(1, 2);
let roots = eq.solve_or_empty(&x);
for r in &roots {
    println!("sin(x) = 1/2 → x = {r}");
}
```

The solver uses the Gröbner basis machinery with a sin/cos polynomial ring to find exact solutions when the equation can be reduced to a polynomial system in `sin(x)` and `cos(x)`.

Note: transcendental equations may have infinitely many solutions (e.g., `sin(x) = 0` has solutions at every integer multiple of π). The solver returns a finite set of principal solutions.

## Polynomial Systems (Gröbner Bases)

### Two-Variable Systems

`symplex::solve_system()` solves systems of polynomial equations using Gröbner bases:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

// Circle and line intersection:
// x² + y² = 1
// x + y = 1
let solutions = symplex::solve_system(
    &[expr!(x^2 + y^2 - 1), expr!(x + y - 1)],
    &[x.clone(), y.clone()],
).unwrap();

for (i, sol) in solutions.iter().enumerate() {
    println!("Solution {}: x = {}, y = {}", i + 1, sol[0], sol[1]);
}
// Solution 1: x = 0, y = 1
// Solution 2: x = 1, y = 0
```

### Nonlinear Systems

The Gröbner basis approach handles genuinely nonlinear systems:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

// Two conics: x² + y² = 5, xy = 2
let solutions = symplex::solve_system(
    &[expr!(x^2 + y^2 - 5), expr!(x*y - 2)],
    &[x.clone(), y.clone()],
).unwrap();

println!("Found {} intersection points:", solutions.len());
for (i, sol) in solutions.iter().enumerate() {
    println!("  ({}, {})", sol[0], sol[1]);
}
```

### Univariate via solve_system

You can also use `solve_system` for single-variable equations — useful when you want the system-solving infrastructure:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let solutions = symplex::solve_system(
    &[expr!(x^3 - 6*x^2 + 11*x - 6)],
    std::slice::from_ref(&x),
).unwrap();

for sol in &solutions {
    println!("x = {}", sol[0]);
}
```

### When Gröbner Fails

Gröbner basis computation can be expensive for large systems or high-degree polynomials. If `solve_system` returns an error, consider:

1. Simplifying the system first
2. Reducing the number of variables by hand
3. Using numerical methods

The solver also uses a symbolic fallback for systems with irrational roots — it attempts to find roots via the univariate polynomial solver after elimination.

## Numerical Solving

### `solve_numeric()` — Newton's Method

When symbolic methods can't handle an equation, use numerical root-finding:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// x = cos(x) — the Dottie number
// Rewrite as x - cos(x) = 0
let f = &x - &x.cos();
match f.solve_numeric(&x, 1.0, 50, 1e-12) {
    Ok(root) => println!("x = cos(x) at x ≈ {root:.10}"),
    Err(e) => println!("failed: {e}"),
}
// x ≈ 0.7390851332
```

The arguments are:
- `&x` — the variable to solve for
- `1.0` — initial guess
- `50` — maximum iterations
- `1e-12` — convergence tolerance

### Tips for Numerical Solving

**Good initial guesses matter.** Newton's method converges quadratically near a root but can diverge or find the wrong root with a bad guess. If you're looking for a root near a specific value, use that as your guess.

**Multiple roots:** Call `solve_numeric` with different initial guesses to find different roots:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// x³ - 6x² + 11x - 6 = 0 has roots at 1, 2, 3
let f = expr!(x^3 - 6*x^2 + 11*x - 6);

for guess in [0.5, 1.5, 3.5] {
    if let Ok(root) = f.solve_numeric(&x, guess, 50, 1e-12) {
        println!("root near {guess}: x ≈ {root:.6}");
    }
}
```

**Verify numerically:** After finding a numerical root, check it:

```rust
let ctx = Context::new();
syms!(ctx; x);
let f = &x - &x.cos();
if let Ok(root) = f.solve_numeric(&x, 1.0, 50, 1e-12) {
    let residual = f.eval_f64_with(&[(&x, root as i64)]);
    println!("root: {root:.10}");
    // For more precise verification, substitute the f64 manually
    println!("verify: {:.2e}", root - root.cos());
}
```

## Inequality Solving

Symplex solves polynomial inequalities using the sign-chart method — it finds roots, tests the sign in each region, and returns a union of intervals.

### Basic Inequalities

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

// x² - 4 > 0 → x < -2 or x > 2
let result = expr!(x^2 - 4).solve_gt(&x).unwrap();
println!("x² - 4 > 0: {result}");

// x² - 4 >= 0 → x <= -2 or x >= 2
let result = expr!(x^2 - 4).solve_ge(&x).unwrap();
println!("x² - 4 ≥ 0: {result}");

// x² - 4 < 0 → -2 < x < 2
let result = expr!(x^2 - 4).solve_lt(&x).unwrap();
println!("x² - 4 < 0: {result}");

// x² - 4 <= 0 → -2 <= x <= 2
let result = expr!(x^2 - 4).solve_le(&x).unwrap();
println!("x² - 4 ≤ 0: {result}");
```

### Set-Valued Solutions

`.solve_as_set()` returns solutions as a `FiniteSet`:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let result = expr!(x^2 - 5*x + 6).solve_as_set(&x);
println!("{result}"); // {2, 3}
```

The inequality solvers (`solve_gt`, `solve_ge`, `solve_lt`, `solve_le`) return `SetEx` values that may represent intervals, unions of intervals, or finite sets.

## ODE Solving

Symplex solves several classes of ordinary differential equations.

### Building an ODE

ODEs are built using `.formal_diff()` to create unevaluated derivative nodes:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

// y' is a formal (unevaluated) derivative
let dy = y.formal_diff(&x);

// Build the ODE: y' + 2y = 0
let ode = &dy + &(&y * 2);
```

You can also use `expr!` with `diff`:

```rust
let ctx = Context::new();
syms!(ctx; x, y);
let ode = expr!(diff(y, x) + 2*y); // y' + 2y = 0
```

### Solving First-Order ODEs

`.solve_ode(&func, &var)` returns `Ex` — the solution expression (or an unevaluated `DSolve` node if the ODE cannot be solved):

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

// y' - x = 0 → y = x²/2 + C1
let dy = y.formal_diff(&x);
let ode = &dy - &x;
if let Some((sol, constants)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
    println!("Constants: {:?}", constants.iter().map(|c| format!("{c}")).collect::<Vec<_>>());
}
```

### Supported ODE Types

The solver recognizes these classes:

**Simple separable:** `y' = f(x)` (no y dependence)
```rust
let ctx = Context::new();
syms!(ctx; x, y);
let ode = expr!(diff(y, x) - x); // y' = x → y = x²/2 + C1
let (sol, _) = ode.solve_ode(&y, &x).unwrap();
println!("y = {sol}");
```

**Full separable:** `y' = f(x)·g(y)`
```rust
let ctx = Context::new();
syms!(ctx; x, y);
// y' = x*y → y = C1·exp(x²/2)
let ode = &y.formal_diff(&x) - &(&x * &y);
if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

**First-order linear (constant coefficient):** `y' + a·y = f(x)`
```rust
let ctx = Context::new();
syms!(ctx; x, y);
// y' + 2y = 0 → y = C1·exp(-2x)
let ode = expr!(diff(y, x) + 2*y);
let (sol, _) = ode.solve_ode(&y, &x).unwrap();
println!("y = {sol}");
```

**First-order linear (variable coefficient):** `y' + P(x)·y = Q(x)`
```rust
let ctx = Context::new();
syms!(ctx; x, y);
// y' + 2x·y = 0 → integrating factor μ = exp(x²)
let ode = &y.formal_diff(&x) + &(&x * &y * 2);
if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

**Exact first-order:** `M(x,y) + N(x,y)·y' = 0` where `∂M/∂y = ∂N/∂x`

**Second-order linear constant-coefficient (homogeneous):** `a·y'' + b·y' + c·y = 0`
```rust
let ctx = Context::new();
syms!(ctx; x, y);
// y'' + y = 0 → y = C1·cos(x) + C2·sin(x)
let dy = y.formal_diff(&x);
let d2y = dy.formal_diff(&x);
let ode = &d2y + &y;
if let Some((sol, _)) = ode.solve_ode(&y, &x) {
    println!("y = {sol}");
}
```

**Second-order linear constant-coefficient (nonhomogeneous):** `a·y'' + b·y' + c·y = f(x)`

### Classifying ODEs

Use `.classify_ode()` to determine the type without solving:

```rust
use symplex::prelude::*;
use symplex::ode::OdeType;

let ctx = Context::new();
syms!(ctx; x, y);

let ode = expr!(diff(y, x) - x);
let classification = ode.classify_ode(&y, &x);
println!("{:?}", classification); // SimpleSeparable
```

The `OdeType` enum includes:
- `SimpleSeparable`
- `FullSeparable`
- `FirstOrderLinearCC`
- `FirstOrderLinearVC`
- `ExactFirstOrder`
- `SecondOrderLinearCCHomogeneous`
- `SecondOrderLinearCCNonHomogeneous`
- `Unknown`

### Verifying ODE Solutions

Use `.check_ode_solution()` to verify that a proposed solution satisfies the ODE:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

// ODE: y' - x = 0
let ode = expr!(diff(y, x) - x);

// Proposed solution: y = x²/2
let sol = &x.powi(2) / 2;
assert!(ode.check_ode_solution(&sol, &y, &x));

// Wrong solution: y = x
let wrong = x.clone();
assert!(!ode.check_ode_solution(&wrong, &y, &x));
```

This substitutes the solution for `y` and its derivative for `y'`, then checks if the residual is zero.

## Solving Workflow Patterns

### Pattern 1: Symbolic → Verify

Always verify solutions, especially for complex equations:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let poly = expr!(x^3 - 6*x^2 + 11*x - 6);
let roots = poly.solve_or_empty(&x);

println!("Roots of x³ - 6x² + 11x - 6 = 0:");
for r in &roots {
    let verified = poly.check_solution(&x, r);
    println!("  x = {r} (verified: {verified})");
}
```

### Pattern 2: Symbolic → Numerical Fallback

Try symbolic first, fall back to numerical:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x);

let f = &x - &x.cos(); // x = cos(x)

// Try symbolic
let symbolic_roots = f.solve_or_empty(&x);
if symbolic_roots.is_empty() {
    println!("No symbolic solution — trying numerical...");
    match f.solve_numeric(&x, 1.0, 50, 1e-12) {
        Ok(root) => println!("x ≈ {root:.10}"),
        Err(e) => println!("Numerical solve also failed: {e}"),
    }
} else {
    for r in &symbolic_roots {
        println!("x = {r}");
    }
}
```

### Pattern 3: System → Verify Numerically

For systems, verify each solution by substituting back:

```rust
use symplex::prelude::*;

let ctx = Context::new();
syms!(ctx; x, y);

let eq1 = expr!(x^2 + y^2 - 1);
let eq2 = expr!(x + y - 1);

let solutions = symplex::solve_system(
    &[eq1.clone(), eq2.clone()],
    &[x.clone(), y.clone()],
).unwrap();

for (i, sol) in solutions.iter().enumerate() {
    let r1 = eq1.subs(&x, &sol[0]).subs(&y, &sol[1]).eval();
    let r2 = eq2.subs(&x, &sol[0]).subs(&y, &sol[1]).eval();
    println!("Solution {}: x={}, y={} — residuals: ({r1}, {r2})", i+1, sol[0], sol[1]);
}
```

## Limitations

- **Degree > 4:** The solver uses closed-form formulas through quartic. Degree 5 and above have no general algebraic solution (Abel–Ruffini theorem). Use `.solve_numeric()` for these.

- **Transcendental equations:** Only certain patterns are recognized (sin/cos reducible to polynomial systems). For general transcendental equations, numerical methods are recommended.

- **Systems:** Gröbner basis computation has worst-case doubly-exponential complexity. Large or high-degree systems may time out. Reduce the system by hand when possible.

- **ODEs:** The solver handles the standard textbook classes listed above. Nonlinear ODEs, PDEs, and delay differential equations are not supported.

## Summary

| Task | Method |
|------|--------|
| Polynomial roots | `.solve(&x)` or `.solve_or_empty(&x)` |
| Factor polynomial | `.factor(&x)` |
| System of equations | `symplex::solve_system(&eqs, &vars)` |
| Numerical root | `.solve_numeric(&x, guess, iters, tol)` |
| Inequality | `.solve_gt(&x)`, `.solve_ge(&x)`, `.solve_lt(&x)`, `.solve_le(&x)` |
| Set-valued solution | `.solve_as_set(&x)` |
| ODE solving | `.solve_ode(&y, &x)` |
| ODE classification | `.classify_ode(&y, &x)` |
| Verify root | `.check_solution(&x, &val)` |
| Verify ODE solution | `.check_ode_solution(&sol, &y, &x)` |

---

*[← Chapter 5: Calculus](05-calculus.md) | [Back to Table of Contents](index.md) | [Chapter 7: Matrices →](07-matrices.md)*