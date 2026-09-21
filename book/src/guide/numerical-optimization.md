# Numerical Optimisation

`symplex::optimize` is a small, dependable set of `f64` routines — bracketed root finding, derivative-free minimisation, global search in a box, least-squares fitting, the trapezoidal rule — plus `Ex` methods that compile an expression with [`compile`](./code-generation.md) and hand the closure to the matching routine. It fills the gap between "I have an exact symbolic answer" and "I need a number now and the equation has no closed form".

Three conventions hold everywhere in the module:

- **Deterministic and bounded.** Every routine has an explicit iteration budget; `differential_evolution` draws its random numbers from a local SplitMix64 generator seeded by `DeOpts::seed`, so identical inputs give bit-identical results.
- **Nothing panics.** Bad input (a bracket without a sign change, `degree ≥ len`, reversed bounds) is `InvalidArgument`; running out of iterations or meeting a non-finite value is `ComputationFailed`. The minimisers that return a `MinimizeResult` report an exhausted budget through `converged == false` *instead* of an error, so the best point found is never thrown away.
- **Polynomial coefficients are ascending**: `[c₀, c₁, …, c_d]` means `c₀ + c₁x + … + c_d xᵈ`. (NumPy's `polyfit` is highest-degree first.)

## Bracketing roots

`brent_root(f, a, b, &opts)` is Brent–Dekker: inverse quadratic interpolation, secant and bisection steps chosen adaptively, so it converges superlinearly on smooth functions and never slower than bisection. `bisect` is the bullet-proof fallback. Both require `f(a)·f(b) < 0` and return a point within `xtol + rtol·|x|` of a sign change (`RootOpts::default()` is `xtol = 2e-12`, `rtol = 4ε`, `max_iter = 100`). `newton_root(f, df, x0, &opts)` polishes from a point and detects divergence instead of looping.

On an `Ex`, `find_root_bracket(&x, a, b)` compiles and brackets in one call; free symbols other than `x` are a `FreeSymbol` error, not a silent `NaN`.

```rust
use std::f64::consts::PI;
use symplex::prelude::*;
use symplex::optimize::{RootOpts, bisect, brent_root, newton_root};

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    println!("{:.15}", brent_root(|t| t * t - 2.0, 0.0, 2.0, &RootOpts::default()).unwrap());   // 1.414213562373136
    println!("{:.15}", bisect(|t| t * t - 2.0, 0.0, 2.0, &RootOpts::default()).unwrap());       // 1.414213562372424
    println!("{:.15}", newton_root(|t| t * t * t - 2.0, |t| 3.0 * t * t, 1.0, &RootOpts::default()).unwrap());
    // 1.259921049894873
    println!("{}", brent_root(|t| t * t + 1.0, -1.0, 1.0, &RootOpts::default()).unwrap_err());
    // brent_root: invalid argument: f(a) and f(b) must have opposite signs: f(-1) = 2, f(1) = 2
    // Newton on atan(x) from x₀ = 2 diverges — reported, not looped:
    assert!(newton_root(f64::atan, |t| 1.0 / (1.0 + t * t), 2.0, &RootOpts::default()).is_err());

    // Kepler's equation E − 0.3·sin E = 1, defined symbolically.
    let kepler = &x - x.sin() * ctx.rational(3, 10) - 1;
    let e = kepler.find_root_bracket(&x, 0.0, PI).unwrap();
    println!("{e:.15}");                                                  // 1.288091313212269
    println!("{:.2e}", kepler.compile(&["x"]).unwrap().call(&[e]));       // 3.95e-13   (residual)

    // Newton with a *symbolically* differentiated derivative agrees:
    let f = kepler.compile(&["x"]).unwrap();
    let df = kepler.diff(&x).compile(&["x"]).unwrap();
    let n = newton_root(|t| f.call(&[t]), |t| df.call(&[t]), 1.0, &RootOpts::default()).unwrap();
    println!("{:.1e}", (n - e).abs());                                    // 4.3e-13

    let loose = RootOpts { xtol: 1e-6, ..RootOpts::default() };
    println!("{:.7}", (x.cos() - &x).find_root_bracket_with(&x, 0.0, 1.0, &loose).unwrap());   // 0.7390851

    let a = ctx.symbol("a");
    println!("{}", (&x.powi(2) - &a).find_root_bracket(&x, 0.0, 2.0).unwrap_err());
    // expression contains free symbol 'a'
}
```

For a system of equations, `solve_numeric_system` (damped Newton with a symbolic Jacobian) is in [Solving Equations](./solving.md#numeric-systems).

## Nelder–Mead

`nelder_mead(f, &x0, &opts)` is the downhill-simplex method with the standard reflect/expand/contract/shrink steps; for more than two variables it uses the dimension-adaptive coefficients that keep the method usable in higher dimensions. `NaN` objective values are treated as `+∞`, so the simplex simply moves away from regions where `f` is undefined. It returns a `MinimizeResult { x, fun, iterations, evaluations, converged }`.

`MinimizeOpts::default()` is `xtol = 1e-8`, `ftol = 1e-12`, `max_iter = 0` (meaning `200·n`) and `initial_step = 0.0` (SciPy's 5 % perturbation of each coordinate of `x0`).

```rust
use symplex::prelude::*;
use symplex::optimize::{MinimizeOpts, nelder_mead};

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    let rosen = |p: &[f64]| (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2);
    let opts = MinimizeOpts { max_iter: 2000, ..MinimizeOpts::default() };
    let r = nelder_mead(rosen, &[-1.2, 1.0], &opts).unwrap();
    println!("x = ({:.6}, {:.6}), f = {:.2e}", r.x[0], r.x[1], r.fun);   // x = (1.000000, 1.000000), f = 1.10e-18
    println!("{} {} {}", r.iterations, r.evaluations, r.converged);      // 116 219 true

    // Exhausting the budget is not an error: you still get the best vertex.
    let tight = MinimizeOpts { max_iter: 20, ..MinimizeOpts::default() };
    let r = nelder_mead(rosen, &[-1.2, 1.0], &tight).unwrap();
    println!("{} {:.4}", r.converged, r.fun);                            // false 2.0022

    // The same problem as an Ex: `minimize_numeric` / `minimize_numeric_with`.
    let rosen_ex = (1 - &x).powi(2) + 100 * (&y - &x.powi(2)).powi(2);
    let r = rosen_ex.minimize_numeric_with(&[&x, &y], &[-1.2, 1.0], &opts).unwrap();
    println!("x = ({:.6}, {:.6}), f = {:.2e}", r.x[0], r.x[1], r.fun);   // x = (1.000000, 1.000000), f = 1.10e-18

    let bowl = (&x - 1).powi(2) + (&y + 2).powi(2);
    let r = bowl.minimize_numeric(&[&x, &y], &[0.0, 0.0]).unwrap();
    println!("x = ({:.6}, {:.6}), f = {:.2e}, converged = {}", r.x[0], r.x[1], r.fun, r.converged);
    // x = (1.000000, -2.000000), f = 5.36e-18, converged = true
    println!("{}", bowl.minimize_numeric(&[&x], &[0.0]).unwrap_err());   // expression contains free symbol 'y'
}
```

## Scalar minimisation

`minimize_scalar(f, a, b, &opts)` is Brent's `localmin` (golden-section steps plus parabolic interpolation) and `golden_section` is the pure golden-section search — slower but immune to parabolic mis-steps. Both return a `ScalarMinimum { x, value }` (the minimiser and the objective there); the interval may be reversed. On an `Ex`: `minimize_scalar_numeric(&x, a, b)`.

```rust
use symplex::prelude::*;
use symplex::optimize::{MinimizeOpts, golden_section, minimize_scalar};

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    // x·ln x has its minimum −1/e at x = 1/e.
    let g = |t: f64| t * t.ln();
    let brent = minimize_scalar(g, 0.1, 2.0, &MinimizeOpts::default()).unwrap();
    let golden = golden_section(g, 0.1, 2.0, &MinimizeOpts::default()).unwrap();
    println!("{:.10} {:.12}", brent.x, brent.value);    // 0.3678794418 -0.367879441171
    println!("{:.10} {:.12}", golden.x, golden.value);  // 0.3678794415 -0.367879441171
    println!("{:.10}", (-1.0f64).exp());                // 0.3678794412

    // Γ has its minimum on (0, ∞) near 1.4616.
    let m = x.gamma().minimize_scalar_numeric(&x, 1.0, 2.0).unwrap();
    println!("{:.8} {:.10}", m.x, m.value);             // 1.46163212 0.8856031944
}
```

The location is only resolved to about `√ε·|x| ≈ 1e-8` relative — the objective is flat to rounding on that scale, which is why `x` above agrees with `1/e` to ten digits but not fifteen, while `value` is correct to twelve.

## Differential evolution (deterministic)

`differential_evolution(f, &bounds, &opts)` is `DE/rand/1/bin` — Latin-hypercube initialisation, one trial vector per member from three distinct others, binomial crossover, clipping to the box — followed by a Nelder–Mead polish of the best member. `bounds` is a slice of closed `Interval<f64>`s, one per coordinate (`Interval::closed(lo, hi)` or `(lo..=hi).into()`; an open or half-open kind is rejected, since trial points are clamped onto the endpoints). Every evaluation point, including during the polish, lies inside `bounds`. `DeOpts::default()` is population `max(15n, 8)`, 300 generations, `CR = 0.7`, `F = 0.8`, `tol = 1e-8`, `seed = 0`. On an `Ex`: `minimize_global_numeric(&vars, &bounds, &opts)`.

```rust
use std::f64::consts::PI;
use symplex::prelude::*;
use symplex::optimize::{DeOpts, differential_evolution};

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    // Rastrigin: many local minima, global minimum 0 at the origin.
    let rastrigin = |p: &[f64]| {
        10.0 * p.len() as f64 + p.iter().map(|v| v * v - 10.0 * (2.0 * PI * v).cos()).sum::<f64>()
    };
    let bounds = [Interval::closed(-5.12, 5.12), Interval::closed(-5.12, 5.12)];
    let r = differential_evolution(rastrigin, &bounds, &DeOpts::default()).unwrap();
    println!("f = {:.2e}, |x| < 1e-6: {}, generations {}, evaluations {}, converged {}",
        r.fun, r.x.iter().all(|v| v.abs() < 1e-6), r.iterations, r.evaluations, r.converged);
    // f = 3.55e-15, |x| < 1e-6: true, generations 85, evaluations 2629, converged true

    // Same seed, same inputs → identical result (MinimizeResult is PartialEq).
    let seeded = DeOpts { seed: 7, ..DeOpts::default() };
    let a = differential_evolution(rastrigin, &bounds, &seeded).unwrap();
    let b = differential_evolution(rastrigin, &bounds, &seeded).unwrap();
    println!("{}", a == b);                                                // true

    // Himmelblau's function has four global minima with f = 0.
    let h = (&x.powi(2) + &y - 11).powi(2) + (&x + &y.powi(2) - 7).powi(2);
    let square = [Interval::closed(-5.0, 5.0), Interval::closed(-5.0, 5.0)];
    let r = h.minimize_global_numeric(&[&x, &y], &square, &DeOpts::default()).unwrap();
    println!("f = {:.2e} at ({:.4}, {:.4})", r.fun, r.x[0], r.x[1]);       // f = 4.52e-16 at (3.0000, 2.0000)

    println!("{}", differential_evolution(rastrigin, &[Interval::closed(1.0, -1.0)], &DeOpts::default()).unwrap_err());
    // differential_evolution: invalid argument: each bound must be a finite interval with lower <= upper, got [1, -1]
    println!("{}", differential_evolution(rastrigin, &[Interval::open(-1.0, 1.0)], &DeOpts::default()).unwrap_err());
    // differential_evolution: invalid argument: each bound must be a closed interval [lower, upper], got (-1, 1)
}
```

Which of Himmelblau's four minima is found depends on the seed; the values printed above are for `seed = 0`. Floating-point transcendental functions can differ in the last bit between platforms, so the *trajectory* is reproducible on one machine rather than universally — the converged optimum is the same.

## Fitting: floating point versus exact

`poly_fit(&xs, &ys, degree)` is a backward-stable least-squares fit (column-scaled Vandermonde, Householder QR; the normal equations are never formed) returning **ascending** coefficients; `eval_poly(&c, x)` evaluates them by Horner's rule and `linear_fit` returns a `LinearFit { slope, intercept }`. `poly_fit_exact(&points, degree)` solves the normal equations over ℚ, so for consistent data it recovers the exact polynomial, and for inconsistent data the exact least-squares solution. `Ex::poly_fit_points(&ctx, &points, &x, degree)` is the same thing returning an `Ex`.

```rust
use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::prelude::*;
use symplex::optimize::{LinearFit, eval_poly, linear_fit, poly_fit, poly_fit_exact};

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    // Six samples of x²/3 − x/2 + 1/7.
    let xs: Vec<f64> = (-2..=3).map(f64::from).collect();
    let ys: Vec<f64> = xs.iter().map(|t| t * t / 3.0 - t / 2.0 + 1.0 / 7.0).collect();
    let c = poly_fit(&xs, &ys, 2).unwrap();
    println!("[{:.12}, {:.12}, {:.12}]", c[0], c[1], c[2]);
    // [0.142857142857, -0.500000000000, 0.333333333333]      ← c₀, c₁, c₂ (ascending)
    println!("{:.6}", eval_poly(&c, 10.0));                             // 28.476190
    println!("{}", poly_fit(&[0.0, 1.0], &[0.0, 1.0], 2).unwrap_err());
    // poly_fit: invalid argument: degree 2 needs at least 3 points, got 2

    let LinearFit { slope, intercept } = linear_fit(&[0.0, 1.0, 2.0, 3.0], &[1.0, 0.0, 4.0, 2.0]).unwrap();
    println!("{slope:.12} {intercept:.12}");                            // 0.700000000000 0.700000000000

    // The same six samples over ℚ: exact recovery.
    let q = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
    let pts: Vec<(Ratio<BigInt>, Ratio<BigInt>)> = (-2..=3)
        .map(|i| {
            let t = q(i, 1);
            (t.clone(), &t * &t / q(3, 1) - &t / q(2, 1) + q(1, 7))
        })
        .collect();
    let c = poly_fit_exact(&pts, 2).unwrap();
    println!("{} {} {}", c[0], c[1], c[2]);                             // 1/7 -1/2 1/3

    // …and as an Ex.
    let pts_ex: Vec<(Ex, Ex)> = (-2..=3)
        .map(|i| {
            let xi = ctx.int(i);
            let yi = &xi.powi(2) * ctx.rational(1, 3) - &xi * ctx.rational(1, 2) + ctx.rational(1, 7);
            (xi, yi.eval())
        })
        .collect();
    println!("{}", Ex::poly_fit_points(&ctx, &pts_ex, &x, 2).unwrap());  // 1/3*x^2 - 1/2*x + 1/7

    // Inconsistent data: the exact least-squares line, and the interpolating cubic.
    let noisy = [(ctx.int(0), ctx.int(1)), (ctx.int(1), ctx.int(0)), (ctx.int(2), ctx.int(4)), (ctx.int(3), ctx.int(2))];
    println!("{}", Ex::poly_fit_points(&ctx, &noisy, &x, 1).unwrap());   // 7/10*x + 7/10
    println!("{}", Ex::poly_fit_points(&ctx, &noisy, &x, 3).unwrap());   // -11/6*x^3 + 8*x^2 - 43/6*x + 1
}
```

Use the exact fit when the data *are* exact (tabulated values, coefficients recovered from a known-degree polynomial, interpolation) and the floating-point fit when the data are measurements. `Ex::poly_interpolate` ([Algebra](./algebra.md#polynomial-algebra-on-ex)) is the special case `degree + 1 == points.len()`.

## Trapezoidal rule

`trapezoid(&ys, &xs)` integrates sampled data on an arbitrary (non-uniform) grid: `Σ ½·(xᵢ₊₁ − xᵢ)·(yᵢ + yᵢ₊₁)`.

```rust
use std::f64::consts::PI;
use symplex::optimize::trapezoid;

fn main() {
    let grid: Vec<f64> = (0..=1000).map(|i| i as f64 / 1000.0).collect();
    let samples: Vec<f64> = grid.iter().map(|t| t * t).collect();
    println!("{:.9}", trapezoid(&samples, &grid).unwrap());                 // 0.333333500
    let sin_samples: Vec<f64> = grid.iter().map(|t| (PI * t).sin()).collect();
    println!("{:.9} {:.9}", trapezoid(&sin_samples, &grid).unwrap(), 2.0 / PI);   // 0.636619249 0.636619772
    println!("{}", trapezoid(&[1.0], &[0.0, 1.0]).unwrap_err());
    // trapezoid: invalid argument: ys and xs must have the same length, got 1 and 2
}
```

When you have the integrand as an expression rather than samples, `integrate_numeric` (adaptive Gauss–Kronrod, [Definite Integration](./definite-integration.md)) is both faster and far more accurate.

## When to prefer the symbolic solvers

Reach for `symplex::optimize` when the problem is genuinely numerical: a transcendental equation with no closed form, a black-box objective, measured data. Prefer the exact machinery when it applies, because it answers a different (better) question:

| You want | Numeric | Exact |
|----------|---------|-------|
| Roots of a polynomial | `find_root_bracket` (one root, needs a bracket) | `solve` (all roots, radicals/`RootOf`), `real_roots_isolate`, `nroots` |
| Roots of a transcendental equation | `find_root_bracket`, `newton_root` | `solve` (Lambert W, inversion), `solve_general` for families |
| Systems of equations | `solve_numeric_system` | `linsolve`, `polysys::solve_system_ex` |
| A minimum of a differentiable function | `nelder_mead`, `minimize_scalar` | `diff` + `solve`, `hessian` for classification |
| A global minimum in a box | `differential_evolution` | `poly_is_nonnegative_on` for *proving* a bound in 1-D; [LP certificates](../cookbook/polynomial-certificates.md) in several |
| A feasible point / optimum of a **linear** program | — | `linprog` ([Exact Linear Programming](./exact-lp.md)) |
| A polynomial through points | `poly_fit` | `poly_fit_exact`, `poly_interpolate` |
| An integral | `trapezoid` (samples), `integrate_numeric` (expression) | `integrate_definite` |

A numeric answer tells you *where* a root is to twelve digits; the exact answer tells you *how many* roots there are and that none was missed. When both are available, use the exact form to decide and the numeric form to display.

See `cargo run --example numeric_optimization` for the complete program.
