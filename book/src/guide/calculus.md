# Calculus

This chapter covers differentiation, indefinite integration, limits and series. Definite and improper integration have [their own chapter](./definite-integration.md), as do [summation and formal power series](./summation.md).

Every code block is a complete program; run it with `cargo run` in a crate that depends on `symplex`.

## Differentiation

`diff(&x)` handles the chain, product and quotient rules, all elementary functions, and the special functions (Gamma, digamma → polygamma, erf, Bessel, orthogonal polynomials, Si/Ci/Ei/li). `diff_n` takes higher derivatives; partial derivatives are just `diff` with respect to another symbol.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    println!("{}", expr!(ctx, sin(x^2)).diff(&x));          // 2*x*cos(x^2)
    println!("{}", expr!(ctx, x^6).diff_n(&x, 4));          // 360*x^2
    println!("{}", expr!(ctx, x^2 * y + y^3).diff(&y));     // x^2 + 3*y^2
    println!("{}", x.digamma().diff(&x));                   // polygamma(1, x)
    println!("{}", x.bessel_j(&ctx.int(0)).diff(&x));       // -1/2*besselj(1, x) + 1/2*besselj(-1, x)
    println!("{}", x.si().diff(&x));                        // sin(x)/x
}
```

### Formal derivatives

`formal_diff` builds a `Derivative` node without evaluating it. This is how you write differential equations (see [Solving Equations](./solving.md)) and finite-difference stencils:

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, h);
    let y = ctx.symbol("y");

    let ode = &y.formal_diff(&x) + &y;            // y' + y  (unevaluated)
    println!("{ode}");                            // y + Derivative(y, x)
    println!("{}", ode.solve_ode(&y, &x));        // C1*exp(-x)

    // Fornberg finite differences: central stencil {x-h, x, x+h}
    let stencil = [&x - &h, x.clone(), &x + &h];
    let d = x.powi(3).differentiate_finite(&x, &stencil, 1).expand();
    println!("{d}");                              // h^2 + 3*x^2
}
```

## Indefinite integration

`integrate(&x)` tries polynomial, u-substitution, by-parts, partial fractions, trigonometric, Risch, Rothstein–Trager, Lazard–Rioboo–Trager and heuristic strategies. When nothing applies you get an unevaluated `Integral` node; use `try_integrate` if that should be an error.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    println!("{}", expr!(ctx, x * exp(x)).integrate(&x));        // x*exp(x) - exp(x)
    println!("{}", expr!(ctx, 1 / (x^2 - 1)).integrate(&x));     // partial fractions
    println!("{}", expr!(ctx, sin(x)^3).integrate(&x));
    println!("{}", expr!(ctx, 1 / (x^2 + 1)).integrate(&x));     // atan(x)

    // No elementary antiderivative: honest unevaluated form
    let hard = expr!(ctx, exp(x^2)).integrate(&x);
    println!("{hard}   unevaluated: {}", hard.has_unevaluated());
    assert!(expr!(ctx, exp(x^2)).try_integrate(&x).is_err());
}
```

## Limits

`limit(&x, &point)` uses the Gruntz algorithm (with a work budget so it cannot hang). 0.2 adds **one-sided limits**: `limit_left`, `limit_right`, `limit_dir(&x, &a, Direction::Left)`. The two-sided `limit` returns an unevaluated `Limit` node when the one-sided limits differ.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);

    println!("{}", expr!(ctx, sin(x) / x).limit(&x, &zero));               // 1
    println!("{}", (1 + 1 / &x).pow(&x).limit(&x, &ctx.infinity()));       // E
    println!("{}", ((1 - &x.cos()) / &x.powi(2)).limit(&x, &zero));        // 1/2

    println!("{}", (1 / &x).limit_right(&x, &zero));                       // oo
    println!("{}", (1 / &x).limit_left(&x, &zero));                        // -oo
    println!("{}", (1 / &x).limit(&x, &zero));                             // Limit(1/x, x, 0)
    println!("{}", (&x * &x.ln()).limit_right(&x, &zero));                 // 0
    println!("{}", x.floor().limit_dir(&x, &ctx.int(1), Direction::Left)); // 0
}
```

## Series

`series(&x, &point, n)` gives a Taylor or Laurent expansion with `n` terms (Puiseux series are refused rather than approximated). `maclaurin(&x, n)` is the expansion at zero; `series_at_infinity(&x, n)` is the asymptotic expansion.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    println!("{}", x.exp().series(&x, &ctx.int(0), 5));
    // 1/24*x^4 + 1/6*x^3 + 1/2*x^2 + x + 1
    println!("{}", (1 / &x.sin()).series(&x, &ctx.int(0), 4));     // Laurent: 1/x + x/6 + …
    println!("{}", (&(&x.powi(2) + 1).sqrt() - &x).series_at_infinity(&x, 4));
    // -1/8*x^(-3) + 1/2*1/x
}
```

For exact coefficients of arbitrary order and closed-form general terms, see `FormalPowerSeries` in [Summation and Series](./summation.md#formal-power-series).

## Residues

`residue(&z, &point)` works at poles of any order; `residue_at_infinity` gives `Res_{z=∞}`, so the sum of all residues can be checked to vanish.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let f = 1 / (&z.powi(2) + 1);
    println!("{}", f.residue(&z, &ctx.i_unit()));               // -1/2*I
    println!("{}", (&z.exp() / &z.powi(3)).residue(&z, &ctx.int(0)));   // 1/2 (third-order pole)
    println!("{}", f.residue_at_infinity(&z));                   // 0
}
```

## Where to go next

- [Definite Integration and Quadrature](./definite-integration.md) — `integrate_definite`, improper integrals, divergence detection, Gauss–Kronrod.
- [Summation and Series](./summation.md) — `summation`, `product_over`, convergence, `FormalPowerSeries`.
- `cargo run --example calculus` for a longer tour.
