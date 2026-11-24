# Transforms

Laplace, Fourier, Mellin and Z transforms, Fourier series, and one-sided limits. Transforms that have an unevaluated node (`LaplaceTransform`, `InverseLaplaceTransform`) return `Ex` and have `try_` twins; the others (`fourier_transform`, `mellin_transform`, `z_transform`) are `Result`-only because no unevaluated node exists for them.

## Laplace

`laplace(&t, &s)` and `inverse_laplace(&s, &t)` are table-driven with linearity, shifting, scaling, differentiation, integration and `f(t)/t` rules. The tables include Bessel `J₀`, `tⁿe^(−at)`, `sinh`/`cosh`, `erf` where possible, `1/√s` and delayed step functions; `laplace_initial_value` and `laplace_final_value` are the initial- and final-value theorems.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; t, s);
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    println!("{}", t.sin().laplace(&t, &s));                                  // 1/(s^2 + 1)
    println!("{}", (&t.powi(2) * &(&t * -3).exp()).laplace(&t, &s));          // 2*(s + 3)^(-3)
    println!("{}", ((&t * 2).sin() / &t).laplace(&t, &s));                    // atan(2/s)
    println!("{}", (&t - 2).heaviside().laplace(&t, &s));                     // exp(-2*s)/s
    println!("{}", t.bessel_j(&ctx.int(0)).laplace(&t, &s));                  // 1/sqrt(s^2 + 1)
    println!("{}", (&a * &t).sinh().laplace(&t, &s));                         // a/(-a^2 + s^2)
    println!("{}", t.erf().laplace(&t, &s));                                  // LaplaceTransform(erf(t), t, s) — not in table

    println!("{}", (&s / (&s.powi(2) + &s * 2 + 5)).inverse_laplace(&s, &t)); // -1/2*sin(2*t)*exp(-t) + cos(2*t)*exp(-t)
    println!("{}", ((&s * -2).exp() / &s).inverse_laplace(&s, &t));           // H(t - 2)
    println!("{}", (1 / &s.sqrt()).inverse_laplace(&s, &t));                  // 1/(sqrt(t)*sqrt(pi))

    let f = (&s + 1) / (&s * (&s.powi(2) + &s * 2 + 5));
    println!("{}", f.laplace_initial_value(&s).unwrap());                     // 0   (f(0⁺))
    println!("{}", f.laplace_final_value(&s).unwrap());                       // 1/5 (f(∞))
    assert!(matches!((1 / (&s * (&s - 1))).laplace_final_value(&s), Err(SymplexError::Divergent { .. })));
}
```

## Fourier transform

`fourier_transform(&t, &w)` uses the non-unitary angular convention `F(ω) = ∫ f(t) e^(−iωt) dt`; `fourier_transform_with(&t, &w, FourierConvention::{NonUnitaryAngular, UnitaryAngular, Ordinary})` selects another. The table covers δ, constants, `H(t)`, `sign`, `1/t`, `|t|`, rectangular windows, `e^(−a|t|)`, Gaussians, `tⁿe^(−at)H(t)`, `cos`/`sin`, `sinc`, extended by linearity, shift, modulation, scaling, the derivative rule and `t·f(t) → iF′(ω)`. Symbols other than `t` and `ω` are treated as real parameters; a required sign condition that cannot be proven is an **error**, not an assumption.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; t, w, nu);
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let i = ctx.i_unit();

    println!("{}", (-&a * t.abs()).exp().fourier_transform(&t, &w).unwrap());   // 2*a/(a^2 + w^2)
    println!("{}", ((&t + 1).heaviside() - (&t - 1).heaviside()).fourier_transform(&t, &w).unwrap()); // 2*sin(w)/w
    println!("{}", ((&t * -2).exp() * t.heaviside()).fourier_transform(&t, &w).unwrap());  // 1/(w*I + 2)
    println!("{}", (-t.powi(2)).exp().fourier_transform(&t, &w).unwrap());     // sqrt(pi)*exp(-1/4*w^2)
    println!("{}", (&t * 3).cos().fourier_transform(&t, &w).unwrap());         // DiracDelta(w - 3)*pi + DiracDelta(w + 3)*pi
    println!("{}", (-(ctx.pi() * t.powi(2))).exp()
        .fourier_transform_with(&t, &nu, FourierConvention::Ordinary).unwrap()); // exp(-nu^2*pi)  (self-dual)
    println!("{}", (1 / (&i * &w + 2)).inverse_fourier_transform(&w, &t).unwrap()); // exp(-2*t)*H(t)

    let b = ctx.symbol("b");
    assert!((-&b * t.abs()).exp().fourier_transform(&t, &w).is_err());       // sign of b unknown
}
```

## Mellin transform

`mellin_transform(&x, &s)` returns `(F(s), strip)` where the strip is the `BoolEx` condition on `re(s)` for convergence; `inverse_mellin_transform(&s, &x)` inverts by table lookup (choosing the strip to the right of the pole when ambiguous).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, s);
    for f in [(-&x).exp(), 1 / (1 + &x), 1 / (1 + &x).powi(3), x.powi(2) * (&x * -3).exp(), x.sin()] {
        let (m, strip) = f.mellin_transform(&x, &s).unwrap();
        println!("M{{{f}}} = {m}   on {strip}");
    }
    // Gamma(s) on re(s) > 0
    // pi/sin(s*pi) on re(s) > 0 & 1 > re(s)
    // B(s, -s + 3) on re(s) > 0 & 3 > re(s)
    // 3^(-s - 2)*Gamma(s + 2) on re(s) > -2
    // sin(1/2*s*pi)*Gamma(s) on re(s) > -1 & 1 > re(s)
    println!("{}", s.gamma().inverse_mellin_transform(&s, &x).unwrap());     // exp(-x)
}
```

## A variable bound inside the expression

The index of a `Sum` or `Product`, the variable of a definite `Integral` and the variable of a `Limit` are *bound*: inside that node the name does not mean the transform variable, and the node as a whole is a constant of the transform. The Laplace, Fourier, Mellin and Z transforms (and their inverses) treat it so: they do not rewrite inside such a node, and they do not count its bound occurrences as dependence on the variable.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; t, s, x);
    // The sum over t is a constant (9/2); only sin(3t) is transformed.
    let f = ctx.parse("Sum(Heaviside(t + 1), t, -3, 3)*sin(3*t)").unwrap();
    println!("{}", f.laplace(&t, &s));                          // 3*Sum(H(t + 1), t=-3..3)/(s^2 + 9)
    let g = ctx.parse("Sum(Heaviside(x + 1), x, -3, 3)*exp(-x)").unwrap();
    println!("{}", g.mellin_transform(&x, &s).unwrap().0);      // 9/2*Gamma(s)
    // A sum over another index is transformed term by term.
    let h = ctx.parse("Sum(exp(-k*t), k, 1, 3)").unwrap();
    println!("{}", h.laplace(&t, &s));                          // 1/(s + 1) + 1/(s + 2) + 1/(s + 3)
}
```

Up to 0.59 the first two came out wrong (`21/(s^2 + 9)` and `7*Gamma(s)`): the step functions inside the sums were read at the transform variable. One limit remains: the Fourier and Mellin transforms refuse a result that would still hold an unevaluated node, even a constant one — `Sum(sin(k^2), k, 1, m)*exp(-x^2)` has no `fourier_transform`, while `laplace` keeps such a constant as a factor.

## Fourier series

`fourier_series_on(&x, &lower, &upper, n_terms)` computes exact coefficients as definite integrals, so `sign`, `|x|`, sawtooth and piecewise waves work. The `FourierSeries` exposes `a0`, `coefficient_a(k)`, `coefficient_b(k)`, complex `coefficient_c(k)`, `omega0()`, `n_terms()` and `truncate(n)`. `fourier_series(&x, n)` is the `[−π, π]` shorthand.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let pi = ctx.pi();
    let square = x.sign().fourier_series_on(&x, &(-&pi), &pi, 5).unwrap();
    println!("{}", square.truncate(5));      // 4*sin(x)/pi + 4*sin(3*x)/(3*pi) + 4*sin(5*x)/(5*pi)
    println!("{} {}", square.coefficient_b(1), square.coefficient_b(2));   // 4/pi 0
    let tri = x.abs().fourier_series_on(&x, &(-&pi), &pi, 3).unwrap();
    println!("{} {}", tri.a0, tri.truncate(3));   // pi -4*cos(x)/pi - 4*cos(3*x)/(9*pi) + 1/2*pi
    let parab = x.powi(2).fourier_series_on(&x, &ctx.int(-1), &ctx.int(1), 2).unwrap();
    println!("{}", parab.truncate(2));       // -4*pi^(-2)*cos(x*pi) + pi^(-2)*cos(2*x*pi) + 1/3
}
```

## Z-transform

`z_transform(&n, &z)` and `inverse_z_transform(&z, &n)` are table-driven (`aⁿ`, `n`, `n²`, `n·aⁿ`, `cos(ωn)`, `1/n!`, …) with linearity and shift rules.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; n, z, w);
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    println!("{}", a.pow(&n).z_transform(&n, &z).unwrap());          // z/(-a + z)
    println!("{}", n.powi(2).z_transform(&n, &z).unwrap());          // (z - 1)^(-3)*(z^2 + z)
    println!("{}", (&w * &n).cos().z_transform(&n, &z).unwrap());    // z*(z - cos(w))/(z^2 - 2*z*cos(w) + 1)
    println!("{}", (&z / (&z - 1).powi(2)).inverse_z_transform(&z, &n).unwrap());   // n
}
```

## One-sided limits

`limit_left`, `limit_right` and `limit_dir(&x, &a, Direction::{Left, Right})` take the limit from one side; each has a `try_` twin. The two-sided `limit` returns an unevaluated `Limit` node when the one-sided limits disagree, rather than picking one.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);
    for f in [1 / &x, &x.abs() / &x, (-1 / &x).exp()] {
        println!("{f}: {} | {} | {}", f.limit_left(&x, &zero), f.limit_right(&x, &zero), f.limit(&x, &zero));
    }
    // 1/x: -oo | oo | Limit(1/x, x, 0)
    // abs(x)/x: -1 | 1 | Limit(abs(x)/x, x, 0)
    // exp(-1/x): oo | 0 | Limit(exp(-1/x), x, 0)
    println!("{}", (&x * &x.ln()).limit_right(&x, &zero));               // 0
    println!("{}", x.tan().limit_left(&x, &(ctx.pi() / 2)));             // oo
    println!("{}", x.floor().limit_dir(&x, &ctx.int(1), Direction::Left)); // 0
}
```

See `cargo run --example transforms` and `laplace_transforms`.
