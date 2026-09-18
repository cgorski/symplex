# Summation and Series

New in 0.2: a summation engine on `Ex` (`summation`, `product_over`), convergence tests, asymptotic series, and an `Ex`-based `FormalPowerSeries`.

## Finite sums

`summation(&k, &lower, &upper)` tries, in order: polynomial (Faulhaber) sums of any degree, geometric and arithmetico-geometric sums, telescoping (partial fractions in `k`), binomial identities (`Σ P(k)·C(n,k)·xᵏ` for any polynomial `P`), and Gosper's algorithm for hypergeometric terms. When nothing applies the result is an unevaluated `Sum` node; `try_summation` returns `Err(ComputationFailed)` instead.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; k, x);
    // Assumptions on the bound let the engine use n! and 2^n freely.
    let n = ctx.symbol_with("n", &[Assumption::Integer, Assumption::Positive]);
    let (zero, one) = (ctx.int(0), ctx.int(1));

    println!("{}", k.summation(&k, &one, &n));                       // 1/2*n^2 + 1/2*n
    println!("{}", k.powi(5).summation(&k, &one, &n));               // 1/6*n^6 + 1/2*n^5 + 5/12*n^4 - 1/12*n^2
    println!("{}", ctx.int(2).pow(&k).summation(&k, &zero, &n));     // 2^(n + 1) - 1
    println!("{}", (1 / (&k * (&k + 1))).summation(&k, &one, &n));   // -1/(n + 1) + 1
    println!("{}", n.binomial(&k).summation(&k, &zero, &n));         // 2^n
    println!("{}", (&k * &n.binomial(&k)).summation(&k, &zero, &n)); // n*2^(n - 1)
    println!("{}", (&k * &ctx.int(2).pow(&k)).summation(&k, &zero, &n)); // 2^(n + 1)*(n - 1) + 2
    println!("{}", (1 / &k).summation(&k, &one, &n));                // harmonic(n)

    // A geometric sum with symbolic ratio is only valid for x ≠ 1:
    println!("{}", x.pow(&k).summation(&k, &zero, &n));
    // Piecewise(n + 1 if x == 1, (-x^(n + 1) + 1)/(-x + 1) if x != 1)

    assert!(k.sin().try_summation(&k, &one, &n).is_err());
}
```

## Infinite series

Infinite upper bounds go through p-series (`ζ(2m)` exact; odd `p` gives a symbolic `zeta(p)`; alternating variants give `ln 2`, Catalan's constant), power-series recognition (`Σ xᵏ/k! = eˣ`, geometric series), telescoping limits, and Gosper tails. Divergent series evaluate to `oo` where the divergence is provable.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; k, x);
    let (zero, one, inf) = (ctx.int(0), ctx.int(1), ctx.infinity());

    println!("{}", k.powi(-2).summation(&k, &one, &inf));                          // 1/6*pi^2
    println!("{}", k.powi(-4).summation(&k, &one, &inf));                          // 1/90*pi^4
    println!("{}", k.powi(-3).summation(&k, &one, &inf));                          // zeta(3)
    println!("{}", (ctx.int(-1).pow(&k) / (&k * 2 + 1).powi(2)).summation(&k, &zero, &inf)); // Catalan
    println!("{}", (ctx.int(-1).pow(&k) / &k).summation(&k, &one, &inf));          // -ln(2)
    println!("{}", (1 / &k.factorial()).summation(&k, &zero, &inf));               // E
    println!("{}", (&x.pow(&k) / &k.factorial()).summation(&k, &zero, &inf));      // exp(x)
    println!("{}", (1 / (&k * (&k + 1))).summation(&k, &one, &inf));               // 1
    println!("{}", (1 / &k).summation(&k, &one, &inf));                            // oo
}
```

## Convergence

`is_convergent(&k)` and `is_absolutely_convergent(&k)` apply the p-series, ratio, root, alternating-series and comparison tests and return `Some(true)`/`Some(false)` only when a test is decisive; otherwise `None`. `hypergeometric_ratio(&k)` returns `t(k+1)/t(k)` as a rational function when the term is hypergeometric.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; k);
    let alt = ctx.int(-1).pow(&k) / &k;

    assert_eq!(k.powi(-2).is_convergent(&k), Some(true));
    assert_eq!((1 / &k).is_convergent(&k), Some(false));
    assert_eq!(alt.is_convergent(&k), Some(true));
    assert_eq!(alt.is_absolutely_convergent(&k), Some(false));

    let ratio = (ctx.int(2).pow(&k) / &k.factorial()).hypergeometric_ratio(&k).unwrap();
    println!("{ratio}");                                                    // 2/(k + 1)
}
```

## Products

`product_over` handles constant, polynomial-factorable, telescoping and factorial-type products, plus classical infinite products.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; k);
    let n = ctx.symbol_with("n", &[Assumption::Integer, Assumption::Positive]);

    println!("{}", k.product_over(&k, &ctx.int(1), &n));                  // n!
    println!("{}", (1 + 1 / &k).product_over(&k, &ctx.int(1), &n));       // n + 1
    println!("{}", (1 - k.powi(-2)).product_over(&k, &ctx.int(2), &ctx.infinity())); // 1/2
}
```

## Formal power series

`fps(&x, &point)` and `fps_maclaurin(&x)` return a `FormalPowerSeries`: coefficients are computed lazily and exactly on demand, and `general_term(&k)` returns a closed form for the *k*-th coefficient when one is recognised. Series support `add`, `sub`, `mul`, `scale`, `compose`, `inverse` (1/f), `reversion` (compositional inverse), `derivative`, `integral`, and `truncate(n)` back to an `Ex`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, k);

    let sin = x.sin().fps_maclaurin(&x);
    let cos = x.cos().fps_maclaurin(&x);
    let exp = x.exp().fps_maclaurin(&x);

    println!("{:?}", sin.coefficients(8).iter().map(|c| c.to_string()).collect::<Vec<_>>());
    // ["0", "1", "0", "-1/6", "0", "1/120", "0", "-1/5040"]
    println!("{}", sin.coefficient(51));              // -1/51!  (exact)
    println!("{}", sin.general_term(&k).unwrap());    // sin(1/2*k*pi)/k!
    println!("{}", exp.general_term(&k).unwrap());    // 1/k!

    println!("{}", sin.mul(&cos).unwrap().truncate(6));               // 2/15*x^5 - 2/3*x^3 + x
    println!("{:?}", cos.inverse().unwrap().coefficients(7).iter().map(|c| c.to_string()).collect::<Vec<_>>());
    // sec x: ["1", "0", "1/2", "0", "5/24", "0", "61/720"]
    println!("{:?}", sin.reversion().unwrap().coefficients(6).iter().map(|c| c.to_string()).collect::<Vec<_>>());
    // asin x: ["0", "1", "0", "1/6", "0", "3/40"]
    let gauss = exp.compose(&(-&x.powi(2)).fps_maclaurin(&x)).unwrap();
    println!("{:?}", gauss.coefficients(7).iter().map(|c| c.to_string()).collect::<Vec<_>>());
    // e^(-x²): ["1", "0", "-1", "0", "1/2", "0", "-1/6"]
}
```

## Finite differences

`finite_diff::finite_diff_weights(order, &points, &x0)` returns exact Fornberg weights; `Ex::differentiate_finite(&x, &points, order)` applies them to an expression (formal `Derivative` nodes inside are replaced by their stencils; `order = 0` does just that replacement).

```rust
use symplex::prelude::*;
use symplex::finite_diff::finite_diff_weights;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, h);
    let stencil = [&x - &h, x.clone(), &x + &h];
    let w = finite_diff_weights(1, &stencil, &x);
    println!("{:?}", w.iter().map(|e| e.to_string()).collect::<Vec<_>>());
    // ["-1/(2*h)", "0", "1/(2*h)"]
    println!("{}", x.powi(3).differentiate_finite(&x, &stencil, 1).expand());   // h^2 + 3*x^2
    println!("{}", x.powi(4).differentiate_finite(&x, &stencil, 2).expand());   // 2*h^2 + 12*x^2
}
```

See `cargo run --example summation_and_series` for the full tour.
