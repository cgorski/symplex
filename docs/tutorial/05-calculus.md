# Chapter 5: Calculus

Symplex provides a complete symbolic calculus toolkit: differentiation, integration, definite integrals, limits, series expansions, Laplace transforms, and implicit differentiation. This chapter walks through each one with practical examples.

## Differentiation

### Basic Derivatives

The `.diff(&var)` method computes the symbolic derivative with respect to a variable:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

// Power rule
let f = expr!(x^3);
println!("{}", f.diff(&x)); // 3*x^2

// Chain rule (automatic)
let f = expr!(sin(x^2));
println!("{}", f.diff(&x)); // 2*x*cos(x^2)

// Product rule (automatic)
let f = &x * &x.sin();
println!("{}", f.diff(&x)); // sin(x) + x*cos(x)

// Quotient / chain combinations
let f = &x.sin() / &x;
let df = f.diff(&x);
println!("{df}"); // (x*cos(x) - sin(x))/x^2 or equivalent
```

Symplex knows the derivatives of all built-in functions: `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, and their inverses.

### Partial Derivatives

For multivariate expressions, `.diff()` differentiates with respect to the specified variable, treating all other variables as constants:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x, y);

let f = expr!(x^2 * y + y^3);
println!("∂f/∂x = {}", f.diff(&x)); // 2*x*y
println!("∂f/∂y = {}", f.diff(&y)); // x^2 + 3*y^2

// Mixed partial: ∂²f/∂x∂y
let fxy = f.diff(&x).diff(&y);
println!("∂²f/∂x∂y = {fxy}"); // 2*x
```

### Higher-Order Derivatives

Chain `.diff()` calls or use `.diff_n()`:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let f = expr!(x^5);

// Chain calls
let f_prime = f.diff(&x);        // 5*x^4
let f_double = f_prime.diff(&x); // 20*x^3
let f_triple = f_double.diff(&x); // 60*x^2

// Or use diff_n for the nth derivative
let f_4th = f.diff_n(&x, 4);
println!("f⁽⁴⁾(x) = {f_4th}"); // 120*x
```

### Implicit Differentiation

When `y` depends on `x` but isn't given explicitly, use `.diff_with_dependent()`:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x, y);

// Circle equation: x² + y² = r²
// Differentiate x² + y² with y depending on x
let expr = expr!(x^2 + y^2);
let result = expr.diff_with_dependent(&x, &[&y]);
println!("{result}"); // 2*x + 2*y*Derivative(y, x)
// This represents 2x + 2y·(dy/dx) = 0
// So dy/dx = -x/y
```

This is how you derive relationships like dy/dx = -x/y from implicit equations. The `Derivative(y, x)` in the output is a formal derivative node — it represents dy/dx as a symbol that can be solved for.

### Formal Derivatives (for ODEs)

`.formal_diff()` creates an unevaluated derivative node instead of computing the derivative. This is essential for constructing ODEs:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x, y);

// Create y' as a formal symbol
let dy_dx = y.formal_diff(&x);
println!("{dy_dx}"); // Derivative(y, x)

// Build an ODE: y' + 2y = 0
let ode = &dy_dx + &(&y * 2);
println!("{ode}"); // Derivative(y, x) + 2*y
```

You can also use the `expr!` macro with `diff()`:

```rust
vars!(x, y);
let dy = expr!(diff(y, x));  // same as y.formal_diff(&x)
```

See [Chapter 6: Solving](06-solving.md) for how to solve ODEs once you've constructed them.

## Integration

### Indefinite Integrals

The `.integrate(&var)` method computes the antiderivative:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

// Power rule
let f = expr!(x^2);
println!("∫x² dx = {}", f.integrate(&x)); // 1/3*x^3

// Trig functions
println!("∫sin(x) dx = {}", x.sin().integrate(&x)); // -cos(x)
println!("∫cos(x) dx = {}", x.cos().integrate(&x)); // sin(x)

// Exponential
println!("∫eˣ dx = {}", x.exp().integrate(&x)); // exp(x)

// Natural log
println!("∫1/x dx = {}", (1 / &x).integrate(&x)); // ln(x)

// Polynomial
let f = expr!(3*x^2 + 2*x + 1);
println!("∫(3x²+2x+1) dx = {}", f.integrate(&x)); // x^3 + x^2 + x
```

Note: symplex does not add an explicit `+ C` constant to indefinite integrals. If you need the general antiderivative, add a constant yourself:

```rust
vars!(x);
let c = symplex::var("C");
let general = &expr!(x^2).integrate(&x) + &c;
println!("{general}"); // 1/3*x^3 + C
```

### Roundtrip Verification

A useful check: differentiating the antiderivative should give back the original:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let f = expr!(x^3 - 3*x^2 + 2*x);
let anti = f.integrate(&x);
let roundtrip = anti.diff(&x);
println!("f(x) = {f}");
println!("∫f dx = {anti}");
println!("d/dx(∫f dx) = {roundtrip}");
// roundtrip should equal f
```

### Definite Integrals

`.definite_integral(&var, &lower, &upper)` computes the definite integral using the fundamental theorem of calculus:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let zero = symplex::int(0);
let one = symplex::int(1);

// ∫₀¹ x² dx = 1/3
let result = expr!(x^2).definite_integral(&x, &zero, &one);
println!("∫₀¹ x² dx = {result}"); // 1/3

// ∫₀^π sin(x) dx = 2
let pi = symplex::default_context().pi();
let result = x.sin().definite_integral(&x, &zero, &pi);
println!("∫₀^π sin(x) dx = {}", result.eval()); // 2

// Area under a cubic from 0 to 1
let f = expr!(x^3 - 3*x^2 + 2*x);
let area = f.definite_integral(&x, &zero, &one);
println!("∫₀¹ (x³-3x²+2x) dx = {area}"); // 1/4
```

Internally, this computes `F(upper) - F(lower)` where `F` is the antiderivative.

## Limits

### Basic Limits

`.limit(&var, &point)` computes the limit as `var` approaches `point`:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let zero = symplex::int(0);

// The classic: lim(x→0) sin(x)/x = 1
let expr = &x.sin() / &x;
let lim = expr.limit(&x, &zero);
println!("lim(x→0) sin(x)/x = {lim}"); // 1

// lim(x→0) (eˣ - 1)/x = 1
let expr = &(&x.exp() - 1) / &x;
let lim = expr.limit(&x, &zero);
println!("lim(x→0) (eˣ-1)/x = {lim}"); // 1
```

The limit engine uses:
1. Direct substitution
2. L'Hôpital's rule for 0/0 and ∞/∞ forms
3. Series expansion as a fallback

### Limits at Infinity

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

// lim(x→∞) 1/x = 0
let inf = symplex::infinity();
let lim = (1 / &x).limit(&x, &inf);
println!("lim(x→∞) 1/x = {lim}"); // 0

// lim(x→-∞) 1/x = 0
let neg_inf = symplex::neg_infinity();
let lim = (1 / &x).limit(&x, &neg_inf);
println!("lim(x→-∞) 1/x = {lim}"); // 0
```

### Fallible variant: `try_limit()`

If you need to detect when a limit can't be computed, use `try_limit()` which returns `Result<Ex>`:

```rust
vars!(x);
match expr.try_limit(&x, &symplex::int(0)) {
    Ok(lim) => println!("limit = {lim}"),
    Err(e) => println!("could not compute limit: {e}"),
}
```

## Series Expansions

### Maclaurin Series (Around 0)

`.maclaurin(&var, order)` computes the Taylor series around x = 0:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

// sin(x) ≈ x - x³/6 + x⁵/120
let sin_series = x.sin().maclaurin(&x, 5);
println!("sin(x) ≈ {}", sin_series.expand().eval());

// cos(x) ≈ 1 - x²/2 + x⁴/24
let cos_series = x.cos().maclaurin(&x, 5);
println!("cos(x) ≈ {}", cos_series.expand().eval());

// eˣ ≈ 1 + x + x²/2 + x³/6 + x⁴/24
let exp_series = x.exp().maclaurin(&x, 5);
println!("eˣ ≈ {}", exp_series.expand().eval());
```

The `order` parameter controls how many terms to include.

### Taylor Series (Around Any Point)

`.series(&var, &point, order)` computes the Taylor series around an arbitrary point:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

// Taylor series of sin(x) around x = π/2
let pi_half = &symplex::default_context().pi() / 2;
let series = x.sin().series(&x, &pi_half, 4);
println!("sin(x) around π/2: {}", series.expand().eval());
```

### Fallible variants: `try_maclaurin()` and `try_series()`

These return `Result<Ex>`, giving `Err` if the series computation fails:

```rust
vars!(x);
match x.sin().try_maclaurin(&x, 5) {
    Ok(s) => println!("{}", s.expand().eval()),
    Err(e) => println!("series failed: {e}"),
}
```

### Fourier Series

For periodic functions, `.fourier_series()` computes the Fourier expansion:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let f = x.clone(); // f(x) = x on [-π, π]
let pi = symplex::default_context().pi();
let neg_pi = -&pi;
let result = f.fourier_series(&x, &neg_pi, &pi, 3);
if let Ok(series) = result {
    println!("Fourier of x: {series}");
}
```

## Laplace Transforms

The Laplace transform converts time-domain functions to the frequency domain — essential for control theory and signal processing.

### Forward Laplace Transform

`.laplace(&t, &s)` computes L{f(t)} = F(s):

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(t, s);

// L{1} = 1/s
let result = symplex::int(1).laplace(&t, &s).unwrap();
println!("L{{1}} = {result}"); // 1/s

// L{t} = 1/s²
let result = t.laplace(&t, &s).unwrap();
println!("L{{t}} = {result}"); // s^(-2) or 1/s^2

// L{exp(2t)} = 1/(s-2)
let result = (&t * 2).exp().laplace(&t, &s).unwrap();
println!("L{{e^2t}} = {result}"); // 1/(s - 2)

// L{sin(t)} = 1/(s² + 1)
let result = t.sin().laplace(&t, &s).unwrap();
println!("L{{sin(t)}} = {result}");

// L{cos(t)} = s/(s² + 1)
let result = t.cos().laplace(&t, &s).unwrap();
println!("L{{cos(t)}} = {result}");
```

### Inverse Laplace Transform

`.inverse_laplace(&s, &t)` computes L⁻¹{F(s)} = f(t):

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(t, s);

// L⁻¹{1/s} = 1
let expr = 1 / &s;
let result = expr.inverse_laplace(&s, &t).unwrap();
println!("L⁻¹{{1/s}} = {result}"); // 1

// L⁻¹{1/(s-2)} = e^(2t)
let expr = 1 / &(&s - 2);
let result = expr.inverse_laplace(&s, &t).unwrap();
println!("L⁻¹{{1/(s-2)}} = {result}"); // exp(2*t)
```

The inverse Laplace transform uses partial fraction decomposition followed by table lookup for each term — the same approach you'd use by hand.

### Workflow: Solving Differential Equations via Laplace

The Laplace transform turns ODEs into algebraic equations. Here's the pattern:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(t, s);

// Consider: y'' + y = 0, y(0) = 0, y'(0) = 1
// Laplace: s²Y(s) - s·y(0) - y'(0) + Y(s) = 0
//          s²Y + Y = 1
//          Y(s) = 1/(s² + 1)

let y_s = 1 / &(expr!(s^2 + 1));
println!("Y(s) = {y_s}");

// Inverse Laplace to get y(t)
let y_t = y_s.inverse_laplace(&s, &t).unwrap();
println!("y(t) = {y_t}"); // sin(t)
```

## Z-Transforms

For discrete-time systems, symplex provides z-transforms:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(n, z);

// Z{(1/2)^n} = z/(z - 1/2)
let half = symplex::rational(1, 2);
let result = half.pow(&n).z_transform(&n, &z).unwrap();
println!("Z{{(1/2)^n}} = {result}");

// Inverse: Z⁻¹{z/(z-2)} = 2^n
let xz = &z / &(&z - 2);
let result = xz.inverse_z_transform(&z, &n).unwrap();
println!("Z⁻¹{{z/(z-2)}} = {result}"); // 2^n
```

## Residues

For complex analysis, compute residues at poles:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

// Residue of 1/x at x = 0
let expr = 1 / &x;
let res = expr.residue(&x, &symplex::int(0));
if let Ok(r) = res {
    println!("Res(1/x, x=0) = {r}"); // 1
}
```

## Numerical Root Finding

When symbolic methods can't solve an equation, use `.solve_numeric()`:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

// Find the fixed point: x = cos(x)
let f = &x - &x.cos();
match f.solve_numeric(&x, 1.0, 50, 1e-12) {
    Ok(root) => println!("x = cos(x) at x ≈ {root:.10}"),
    Err(e) => println!("Failed: {e}"),
}
```

Arguments: `solve_numeric(&var, initial_guess, max_iterations, tolerance)`.

## Arbitrary-Precision Evaluation

For high-precision numerical results:

```rust
use symplex::prelude::*;

let pi = symplex::default_context().pi();

// π to 30 decimal digits
match pi.eval_decimal(30) {
    Ok(s) => println!("π = {s}"),
    Err(e) => println!("Error: {e}"),
}

// e to 30 digits
match symplex::e().eval_decimal(30) {
    Ok(s) => println!("e = {s}"),
    Err(e) => println!("Error: {e}"),
}
```

## Practical Workflow: Analyzing a Function

Here's a complete calculus workflow — the kind of thing you'd do in a homework problem or an engineering analysis:

```rust
use symplex::prelude::*;
use symplex::vars;

fn main() {
    vars!(x);

    // Define the function
    let f = expr!(x^3 - 3*x^2 + 2*x);
    println!("f(x) = {f}");

    // First and second derivatives
    let df = f.diff(&x);
    let d2f = df.diff(&x);
    println!("f'(x)  = {df}");
    println!("f''(x) = {d2f}");

    // Critical points
    let critical = df.solve_or_empty(&x);
    println!("\nCritical points:");
    for pt in &critical {
        println!("  x = {pt}");
    }

    // Classify with second-derivative test
    println!("\nClassification:");
    for pt in &critical {
        let d2_val = d2f.subs(&x, pt);
        println!("  f''({pt}) = {d2_val}");
    }

    // Integrate
    let anti = f.integrate(&x);
    println!("\n∫f(x) dx = {anti}");

    // Verify: d/dx(∫f dx) = f
    let roundtrip = anti.diff(&x);
    println!("d/dx(∫f dx) = {roundtrip}");

    // Definite integral
    let area = f.definite_integral(&x, &symplex::int(0), &symplex::int(1));
    println!("\n∫₀¹ f(x) dx = {area}");

    // Taylor series around x = 0
    let series = f.maclaurin(&x, 5);
    println!("\nTaylor (should match f): {}", series.expand().eval());

    // Factor
    let factored = f.factor(&x);
    println!("\nFactored: {factored}");

    // Limit at x → 0
    let ratio = &f / &x;
    let lim = ratio.limit(&x, &symplex::int(0));
    println!("\nlim(x→0) f(x)/x = {lim}"); // 2

    // LaTeX output
    println!("\nLaTeX:");
    println!("  f(x) = {}", f.to_latex());
    println!("  f'(x) = {}", df.to_latex());
}
```

## Summary

| Operation | Method | Example |
|-----------|--------|---------|
| Derivative | `.diff(&x)` | `f.diff(&x)` |
| nth derivative | `.diff_n(&x, n)` | `f.diff_n(&x, 3)` |
| Formal derivative | `.formal_diff(&x)` | `y.formal_diff(&x)` |
| Implicit differentiation | `.diff_with_dependent(&x, &[&y])` | chain rule with deps |
| Indefinite integral | `.integrate(&x)` | `f.integrate(&x)` |
| Definite integral | `.definite_integral(&x, &a, &b)` | `f.definite_integral(&x, &zero, &one)` |
| Limit | `.limit(&x, &pt)` | `f.limit(&x, &zero)` |
| Limit (fallible) | `.try_limit(&x, &pt)` | `f.try_limit(&x, &zero)?` |
| Maclaurin series | `.maclaurin(&x, order)` | `f.maclaurin(&x, 5)` |
| Maclaurin (fallible) | `.try_maclaurin(&x, order)` | `f.try_maclaurin(&x, 5)?` |
| Taylor series | `.series(&x, &pt, order)` | `f.series(&x, &a, 5)` |
| Taylor (fallible) | `.try_series(&x, &pt, order)` | `f.try_series(&x, &a, 5)?` |
| Laplace transform | `.laplace(&t, &s)` | `f.laplace(&t, &s)` |
| Inverse Laplace | `.inverse_laplace(&s, &t)` | `F.inverse_laplace(&s, &t)` |
| Z-transform | `.z_transform(&n, &z)` | `f.z_transform(&n, &z)` |
| Inverse Z-transform | `.inverse_z_transform(&z, &n)` | `F.inverse_z_transform(&z, &n)` |
| Residue | `.residue(&x, &pt)` | `f.residue(&x, &zero)` |
| Numerical root | `.solve_numeric(&x, guess, iters, tol)` | `f.solve_numeric(&x, 1.0, 50, 1e-12)` |

---

*[← Chapter 4: Simplification](04-simplification.md) | [Back to Table of Contents](index.md) | [Chapter 6: Solving Equations →](06-solving.md)*