# Solving Equations

Single equations, general (periodic) solutions, linear and polynomial systems, numeric systems, inequalities, ordinary differential equations with initial conditions, and recurrences.

## `solve`

`solve(&x)` returns `Result<Vec<Ex>>`. Polynomials are solved through quartic by radicals; degree ≥ 5 gives `RootOf` nodes (exact, numerically evaluable, and *not* counted as unevaluated). Transcendental equations use inversion peeling and Lambert W. Results are evaluated, so `asin(1/2)` comes back as `π/6`.

The 0.2 contract is that `solve` never lies:

| Situation | Result |
|-----------|--------|
| Finitely many solutions | `Ok(vec![…])` |
| Identity (`x − x = 0`) | `Err(SymplexError::InfiniteSolutions { .. })` |
| Contradiction (`0·x + 1 = 0`) or range violation (`sin x = 2`, `eˣ = −1`, `|x| = −1`) | `Err(SymplexError::NoSolution { .. })` |
| Solver has no method | `Err(SymplexError::ComputationFailed { .. })` |

`solve_or_empty` maps every error to an empty vector when you do not care why.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    println!("{:?}", expr!(ctx, x^2 - 5*x + 6).solve(&x).unwrap());          // [Ex(3), Ex(2)]
    println!("{:?}", (&x.sin() - &ctx.rational(1, 2)).solve(&x).unwrap());  // [Ex(1/6*pi), Ex(5/6*pi)]
    println!("{:?}", (&x.exp() - 5).solve(&x).unwrap());                     // [Ex(ln(5))]
    println!("{:?}", (&x.powi(2) + 1).solve(&x).unwrap());                   // [Ex(I), Ex(-I)]
    println!("{}", (&x.powi(5) - &x - 1).solve(&x).unwrap()[0]);             // RootOf(x^5 - x - 1, 0)

    assert!(matches!((&x - &x).solve(&x), Err(SymplexError::InfiniteSolutions { .. })));
    assert!(matches!((&x.sin() - 2).solve(&x), Err(SymplexError::NoSolution { .. })));
}
```

### Symbolic coefficients are returned in rational normal form

Since 0.3, when the coefficients of a linear or quadratic equation are themselves parameters, the solutions (and the quadratic discriminant) are passed through [`ratsimp`](./algebra.md#rational-normal-form-ratsimp). A parametric equation whose coefficients are fractions therefore comes back as **one cancelled fraction**, not a fraction of fractions. The values are the same as in 0.2; only the printed form changed.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; r, j, x, a, b, c);

    // (3r − 1)/(j + 1) = (r + 1)/(2j), solved for r
    let eqn = (&r * 3 - 1) / (&j + 1) - (&r + 1) / (&j * 2);
    println!("{}", eqn.solve(&r).unwrap()[0]);              // (3*j + 1)/(5*j - 1)

    for s in (&a * &x.powi(2) + &b * &x + &c).solve(&x).unwrap() {
        println!("{s}");
    }
    // (-b + sqrt(-4*a*c + b^2))/(2*a)
    // (-b - sqrt(-4*a*c + b^2))/(2*a)

    // x²/a + 2x + a = 0: the discriminant 4 − 4 simplifies to 0 → one double root
    for s in (&x.powi(2) / &a + &x * 2 + &a).solve(&x).unwrap() {
        println!("{s}");                                      // -a
    }
    let lin = &a * &x / (&a + 1) - &b / (&a - 1);
    println!("{}", lin.solve(&x).unwrap()[0]);              // (a*b + b)/(a^2 - a)
}
```

## General solutions

`solve` returns principal branches. `solve_general` returns the complete solution families of periodic equations, expressed with a fresh integer-assumed parameter (`n`, or `n1`, `n2`, … if `n` is taken). `GeneralSolution::instance(k)` substitutes a concrete integer.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let fam = (&x.sin() - &ctx.rational(1, 2)).solve_general(&x).unwrap();
    for s in &fam.solutions {
        println!("{s}");                 // 2*n*pi + 1/6*pi,  2*n*pi + 5/6*pi
    }
    println!("{:?}", fam.parameters);    // [Ex(n)]
    println!("{:?}", fam.instance(1));   // [Ex(13/6*pi), Ex(17/6*pi)]
    let tan = (&x.tan() - 1).solve_general(&x).unwrap();
    println!("{}", tan.solutions[0]);    // n*pi + 1/4*pi  (parameter name may differ)
}
```

## Linear systems

`linsolve(&eqs, &vars)` accepts `Ex` (meaning `expr = 0`) or `Equation` values, allows symbolic coefficients, and returns a `LinearSolution`:

- `Unique(Vec<(var, value)>)`,
- `Parametric { solution, free }` — every variable is given; pivots in terms of the free variables, free variables mapped to themselves,
- `Inconsistent` — a legitimate mathematical outcome, so it is a variant rather than an `Err` (which is reserved for malformed input such as non-linear equations).

`linsolve_matrix(&a, &b)` solves `A·x = b` for rectangular or singular `A` (unknowns are named `x1, x2, …`); `Context::solve_system` is the same solver.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z, a, b);
    let vars = [x.clone(), y.clone(), z.clone()];

    let sol = linsolve(&[&x + &y + &z - 6, &x - &y + 2 * &z - 5, &x * 2 + &y - &z - 1], &vars).unwrap();
    println!("{sol:?}");                          // Unique([(x, 1), (y, 2), (z, 3)])

    let sol = linsolve(&[&x + &y + &z - 6, &x - &y - 2], &vars).unwrap();
    if let LinearSolution::Parametric { solution, free } = &sol {
        println!("{solution:?} free {free:?}");   // x = -z/2 + 4, y = -z/2 + 2, z = z; free [z]
    }
    println!("{}", sol.get(&x).unwrap());         // -1/2*z + 4

    assert!(linsolve(&[&x + &y - 1, &x + &y - 2], &[x.clone(), y.clone()]).unwrap().is_inconsistent());

    // Equations and symbolic coefficients
    let sol = linsolve(&[eq!(ctx, a * x + y = 1), eq!(ctx, x - y = b)], &[x.clone(), y.clone()]).unwrap();
    println!("{}", sol.get(&x).unwrap());         // (b + 1)/(a + 1)

    let am = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let bm = Matrix::col_vector(vec![ctx.int(6), ctx.int(15), ctx.int(24)]);
    println!("{:?}", linsolve_matrix(&am, &bm).unwrap());   // Parametric: x1 = x3, x2 = -2*x3 + 3
}
```

## Polynomial systems

`symplex::polysys::solve_system_ex(&eqs, &vars)` uses Gröbner bases (Buchberger + FGLM) and returns **algebraic** solutions (radicals, not just rationals) for zero-dimensional systems; positive-dimensional systems return `Err(InfiniteSolutions)`.

```rust
use symplex::prelude::*;
use symplex::polysys::solve_system_ex;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let vars = [x.clone(), y.clone()];
    let sols = solve_system_ex(&[&x.powi(2) + &y.powi(2) - 1, &x - &y], &vars).unwrap();
    println!("{sols:?}");        // [[1/2*sqrt(2), 1/2*sqrt(2)], [-1/2*sqrt(2), -1/2*sqrt(2)]]
    let sols = solve_system_ex(&[&x.powi(2) + &y.powi(2) - 1, &x.powi(2) - &y], &vars).unwrap();
    println!("{} solutions, y = {}", sols.len(), sols[0][1]);   // 4, 1/2*sqrt(5) - 1/2
    assert!(matches!(solve_system_ex(&[&x + &y - 1], &vars), Err(SymplexError::InfiniteSolutions { .. })));
}
```

## Numeric systems

`solve_numeric_system(&eqs, &vars, &x0)` is a damped Newton method with a symbolic Jacobian; `solve_numeric_system_with` takes `NewtonOpts { tol, max_iter, .. }`. For a single equation, `solve_numeric(&x, x0, max_iter, tol)` exists on `Ex`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let f1 = &x.powi(2) + &y.powi(2) - 4;
    let f2 = &x.exp() + &y - 1;
    let root = solve_numeric_system(&[f1.clone(), f2], &[x.clone(), y.clone()], &[1.0, -1.0]).unwrap();
    println!("{:.10} {:.10}", root[0], root[1]);          // 1.0041687385 -1.7296372870
    println!("{:e}", f1.eval_f64_with(&[(&x, root[0]), (&y, root[1])]).unwrap());   // ~1e-16
}
```

## Inequalities

`solve_gt`/`ge`/`lt`/`le` return a `SetEx` (sign-chart method; absolute values supported). `reduce_inequalities` and `BoolEx::solve_for` handle conjunctions — see [Sets and Logic](./sets-and-logic.md).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    println!("{}", expr!(ctx, x^2 - 4).solve_gt(&x));         // (-oo, -2) ∪ (2, oo)
    println!("{}", (&(&x - 1).abs() - 2).solve_lt(&x));       // (-1, 3)
    println!("{}", (&x.abs() - 3).solve_ge(&x));              // (-oo, -3] ∪ [3, oo)
}
```

## Ordinary differential equations

Build the ODE as an expression in `y` and `y.formal_diff(&x)` (nested for higher orders) and call `solve_ode(&y, &x)`. `classify_ode` names the class; 0.2 supports 16: simple/full separable, first-order linear (constant/variable coefficient), exact, integrating factor, Bernoulli, Riccati, Euler–Cauchy, homogeneous-coefficient, second-order constant-coefficient (homogeneous/non-homogeneous), variation of parameters, reduction of order, **nth-order constant-coefficient**, and **Clairaut**.

`solve_ode_ivp(&y, &x, &[InitialCondition { order: k, x: x0, value }, …])` pins the constants with conditions `y^(k)(x0) = value`. `solve_riccati` takes a known particular solution. `ode::solve_ode_system_ivp(&A, &t, &x0)` solves `x' = A x` with initial state.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, t);
    let y = ctx.symbol("y");
    let d1 = y.formal_diff(&x);
    let d2 = d1.formal_diff(&x);
    let zero = ctx.int(0);

    let ode = &d2 + &y;
    println!("{:?}", ode.classify_ode(&y, &x));              // SecondOrderLinearCCHomogeneous
    println!("{}", ode.solve_ode(&y, &x));                   // C1*cos(x) + C2*sin(x)
    let ics = [
        InitialCondition { order: 0, x: zero.clone(), value: ctx.int(0) },   // y(0) = 0
        InitialCondition { order: 1, x: zero.clone(), value: ctx.int(1) },   // y'(0) = 1
    ];
    let sol = ode.solve_ode_ivp(&y, &x, &ics).unwrap();
    println!("{}", sol.simplify());                          // sin(x)

    let third = &d2.formal_diff(&x) - &d1;                   // y''' − y' = 0
    println!("{:?} {}", third.classify_ode(&y, &x), third.solve_ode(&y, &x));
    // NthOrderLinearConstCoeff C1 + C2*exp(x) + C3*exp(-x)

    let clairaut = &y - &x * &d1 - &d1.powi(2);              // y = x y' + (y')²
    println!("{:?} {}", clairaut.classify_ode(&y, &x), clairaut.solve_ode(&y, &x));   // Clairaut C1^2 + C1*x

    let riccati = &d1 - &y.powi(2) + &(&ctx.int(2) / &x.powi(2));
    println!("{}", riccati.solve_riccati(&y, &x, &(&ctx.int(1) / &x)).unwrap());
    // x^2/(-1/3*x^3 + C1) + 1/x

    let a = matrix![ctx, [0, 1], [-1, 0]];
    let sys = symplex::ode::solve_ode_system_ivp(&a, &t, &[ctx.int(1), ctx.int(0)]).unwrap();
    println!("{} {}", sys[0].simplify(), sys[1].simplify());   // cos(t) -sin(t)
}
```

`solve_ode` returns an unevaluated `DSolve` node when no method applies; `try_solve_ode` makes that an error, and `check_ode_solution` verifies a candidate.

## Recurrences

`rsolve::rsolve_linear(&coeffs, forcing, &n, &ics)` solves `c₀·a(n) + c₁·a(n+1) + … + c_k·a(n+k) = f(n)` for rational constants `cᵢ` and forcing terms that are sums of `c·n^d·bⁿ`; initial values `a(0), a(1), …` are optional (unused constants stay as `C1, C2, …`). `rsolve_first_order(&p, &q, &n, a0)` solves `a(n+1) = p(n)·a(n) + q(n)`.

```rust
use symplex::prelude::*;
use symplex::rsolve::{rsolve_first_order, rsolve_linear};

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; n);
    // Fibonacci: a(n+2) − a(n+1) − a(n) = 0
    let fib = rsolve_linear(&[ctx.int(-1), ctx.int(-1), ctx.int(1)], None, &n, &[ctx.int(0), ctx.int(1)]).unwrap();
    println!("{}", fib.subs_i64(&n, 10).eval().simplify());                  // 55
    // Towers of Hanoi: a(n+1) − 2a(n) = 1
    println!("{}", rsolve_linear(&[ctx.int(-2), ctx.int(1)], Some(&ctx.int(1)), &n, &[ctx.int(0)]).unwrap());  // 2^n - 1
    println!("{}", rsolve_linear(&[ctx.int(6), ctx.int(-5), ctx.int(1)], None, &n, &[]).unwrap());  // C1*3^n + C2*2^n
    println!("{}", rsolve_first_order(&(&n + 1), &ctx.int(0), &n, Some(&ctx.int(1))).unwrap());     // n!
}
```

See `cargo run --example linear_systems_and_ivp`, `equation_solving` and `ode_solving`.
