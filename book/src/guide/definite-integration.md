# Definite Integration and Quadrature

New in 0.2. The 0.1 method `definite_integral` computed `F(b) − F(a)` from an antiderivative, which is wrong whenever the integrand has a singularity inside `[a, b]` (`∫₋₁¹ dx/x²` came out as `−2`). It has been replaced by three methods with precise contracts.

| Method | Returns | Use when |
|--------|---------|----------|
| `integrate_definite(&x, &a, &b)` | `Ex` — a closed form, or an unevaluated `DefiniteIntegral` node (`Integral(f, x, a, b)`) | Exploring; chaining |
| `try_integrate_definite(&x, &a, &b)` | `Result<Ex>` — `Err(Divergent)`, `Err(ComputationFailed)`, `Err(InvalidArgument)` | Pipelines that must distinguish "diverges" from "don't know" |
| `integrate_numeric(&x, &a, &b)` | `Result<f64>` | You want a number and accept floating point |

## What `integrate_definite` does

1. Evaluates the integrand and locates its real singularities inside `[a, b]`.
2. Splits at interior singularities and treats each piece as an improper integral via one-sided limits of the antiderivative; a piece with an infinite one-sided limit makes the whole integral **divergent**.
3. Handles infinite bounds the same way.
4. Applies symmetry shortcuts (odd integrand on a symmetric interval, periodicity).
5. Resolves `Abs`, `Sign`, `Heaviside`, `DiracDelta` and `Piecewise` integrands by splitting at their breakpoints.
6. If no antiderivative exists, consults a table of ~30 classical improper integrals (Gaussian, Dirichlet, Fresnel, `xⁿe⁻ˣ`, `x/(eˣ−1)`, `ln x`, `1/(1+x⁴)`, …) whose entries may carry symbolic parameters guarded by assumptions.

If none of this decides the integral, the result is an unevaluated `DefiniteIntegral` node, displayed `Integral(f, x, a, b)` — never a guessed finite number.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let (zero, one, inf) = (ctx.int(0), ctx.int(1), ctx.infinity());

    // Proper integrals
    println!("{}", x.powi(2).integrate_definite(&x, &zero, &one));               // 1/3
    println!("{}", x.sin().integrate_definite(&x, &zero, &ctx.pi()));           // 2

    // Improper integrals
    println!("{}", (-&x).exp().integrate_definite(&x, &zero, &inf));            // 1
    println!("{}", (-x.powi(2)).exp().integrate_definite(&x, &ctx.neg_infinity(), &inf)); // sqrt(pi)
    println!("{}", (&x.sin() / &x).integrate_definite(&x, &zero, &inf));        // 1/2*pi
    println!("{}", x.ln().integrate_definite(&x, &zero, &one));                 // -1
    println!("{}", (1 / &x.sqrt()).integrate_definite(&x, &zero, &one));        // 2
    println!("{}", (&x / (&x.exp() - 1)).integrate_definite(&x, &zero, &inf));  // 1/6*pi^2

    // Non-smooth integrands
    println!("{}", x.abs().integrate_definite(&x, &ctx.int(-2), &ctx.int(3)));  // 13/2
    println!("{}", (&x.dirac_delta() * &x.cos()).integrate_definite(&x, &ctx.int(-1), &one)); // 1
    let pw = Ex::piecewise(&[(&x.powi(2), &x.lt(&one)), (&(2 - &x), &x.ge(&one))]);
    println!("{}", pw.integrate_definite(&x, &zero, &ctx.int(2)));               // 5/6
}
```

## Symbolic parameters need assumptions

`∫₀^∞ e^(−a x) dx = 1/a` is only true for `a > 0`. The table checks such conditions through the assumption system and refuses otherwise.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let (zero, inf) = (ctx.int(0), ctx.infinity());
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let n = ctx.symbol_with("n", &[Assumption::Positive]).unwrap();

    println!("{}", (-(&a * &x)).exp().integrate_definite(&x, &zero, &inf));           // 1/a
    println!("{}", (-(&a * &x.powi(2))).exp().integrate_definite(&x, &ctx.neg_infinity(), &inf)); // sqrt(pi/a)
    println!("{}", (&x.sin() * &(-(&a * &x)).exp()).integrate_definite(&x, &zero, &inf)); // 1/(a^2 + 1)
    println!("{}", (&x.pow(&(&n - 1)) * &(-&x).exp()).integrate_definite(&x, &zero, &inf)); // Gamma(n)

    // Unknown sign: stays unevaluated instead of assuming a > 0
    let b = ctx.symbol("b");
    let r = (-(&b * &x)).exp().integrate_definite(&x, &zero, &inf);
    assert!(r.has_unevaluated());
}
```

## Divergence is an error, not a number

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let one = ctx.int(1);

    match x.powi(-2).try_integrate_definite(&x, &ctx.int(-1), &one) {
        Err(SymplexError::Divergent { reason, .. }) => println!("diverges: {reason}"),
        other => println!("unexpected {other:?}"),
    }
    assert!(matches!(
        (1 / &x).try_integrate_definite(&x, &one, &ctx.infinity()),
        Err(SymplexError::Divergent { .. })
    ));

    // The non-try variant keeps it symbolic.
    let kept = x.powi(-2).integrate_definite(&x, &ctx.int(-1), &one);
    assert!(kept.has_unevaluated());
}
```

An undecided definite integral keeps its bounds: the `DefiniteIntegral` node prints as `Integral(f, x, a, b)` (LaTeX `\int_a^b f\,dx`), round-trips through `parse`, binds `x` for `free_symbols`/`subs`, differentiates by the Leibniz rule, and `eval_f64()` evaluates it by quadrature. `Ex::eval_integrals()` re-attempts every formal definite integral inside an expression. Use `try_integrate_definite` when you need to know *why* it was undecided.

## Numeric quadrature

`integrate_numeric` compiles the integrand (so it must contain no free symbol other than the variable and only nodes `compile` supports) and runs adaptive Gauss–Kronrod G7/K15 quadrature. Infinite bounds are mapped to a finite interval. `integrate_numeric_with(&x, &a, &b, &QuadOpts)` returns a `QuadResult { value, error }` (the estimate and its estimated absolute error) and lets you set tolerances and the subdivision limit.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let (zero, one) = (ctx.int(0), ctx.int(1));

    let v = x.sin().integrate_numeric(&x, &zero, &ctx.pi()).unwrap();
    assert!((v - 2.0).abs() < 1e-12);

    // No elementary antiderivative — numerics is the right tool
    let opts = QuadOpts { rel_tol: 1e-12, ..QuadOpts::default() };
    let q = x.powi(2).exp().integrate_numeric_with(&x, &zero, &one, &opts).unwrap();
    println!("∫₀¹ e^(x²) dx ≈ {:.15} ± {:.1e}", q.value, q.error);   // 1.462651745907181

    // A divergent integral does not converge and is reported as an error.
    assert!(x.powi(-2).integrate_numeric(&x, &ctx.int(-1), &one).is_err());
}
```

Conditionally convergent oscillatory tails such as `∫₀^∞ sin(x)/x dx` are beyond plain adaptive quadrature and also return an error; use `integrate_definite` (which knows the closed form) for those.

For integrating a plain Rust closure, `symplex::definite::quadrature(&f, a, b, &opts)` exposes the same algorithm and returns the same `QuadResult`.

## Residues

Residues are the contour-integration counterpart and live on `Ex` as well:

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let g = (&z + 2) / (&z * (&z - 1));
    println!("{}", g.residue(&z, &ctx.int(0)));      // -2
    println!("{}", g.residue(&z, &ctx.int(1)));      // 3
    println!("{}", g.residue_at_infinity(&z));       // -1  (= −(−2 + 3))
}
```

See `cargo run --example definite_integration` for the full tour.
