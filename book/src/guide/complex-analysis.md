# Complex Analysis and Special Functions

New in 0.2. symplex is a CAS over ℂ: a symbol with no assumptions may be complex, and the library refuses to pretend otherwise.  Every multivalued function takes its principal branch — in `eval`, in numerical evaluation and in every rewrite (since 0.23; before, `eval` took the real odd root of a negative number and some simplifications assumed real arguments).  [The Domain Model](../getting-started/key-concepts.md#the-domain-model) lists which identities need which assumptions.

## re, im, conjugate, arg

`Ex::{re, im, conjugate, arg}` are constructed with as much evaluation as the structure and assumptions allow. For a symbol declared `Real` the parts are immediate; for an unassumed symbol you get an unevaluated `re(z)` node (0.1 silently assumed every symbol real — a wrong answer for `conjugate(z)`).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let y = ctx.symbol_with("y", &[Assumption::Real]);
    let z = ctx.symbol("z");          // may be complex
    let i = ctx.i_unit();

    let w = &x + &i * &y;
    println!("{}", w.re());                       // x
    println!("{}", w.im());                       // y
    println!("{}", w.conjugate());                // x - y*I
    println!("{}", w.abs_squared());              // x^2 + y^2
    println!("{}", w.arg());                      // atan2(y, x)

    println!("{}", z.re());                       // re(z)          — not assumed real
    println!("{}", z.conjugate());                // conjugate(z)
    println!("{}", (&z.powi(2) + 1).conjugate()); // conjugate(z)^2 + 1  (distributes)
    println!("{}", z.exp().conjugate());          // exp(conjugate(z))   (commutes with exp)
    println!("{}", (&i * &z).re());               // -im(z)
    println!("{}", z.exp().re());                 // cos(im(z))*exp(re(z))
    println!("{:?} {:?}", (&x.powi(2) + 1).is_real_valued(), z.is_real_valued());   // Some(true) None
}
```

## Splitting into real and imaginary parts

`as_real_imag()` returns `(re, im)` as a pair; `expand_complex()` rewrites the expression as `re + im·I`; `polar()` returns a `Polar { modulus, argument }` struct (`symplex::expr_complex::Polar`).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let y = ctx.symbol_with("y", &[Assumption::Real]);
    let w = &x + &ctx.i_unit() * &y;

    let (re, im) = w.powi(2).as_real_imag();
    println!("({re}) + ({im})i");                 // (x^2 - y^2) + (2*x*y)i
    let (re, im) = w.exp().as_real_imag();
    println!("({re}) + ({im})i");                 // (cos(y)*exp(x)) + (sin(y)*exp(x))i
    let (re, im) = w.sin().as_real_imag();
    println!("({re}) + ({im})i");                 // (sin(x)*cosh(y)) + (cos(x)*sinh(y))i
    let (re, im) = (1 / &w).as_real_imag();
    println!("({re}) + ({im})i");                 // (x/(x^2 + y^2)) + (-y/(x^2 + y^2))i
    println!("{}", (&ctx.i_unit() * &x).cos().expand_complex());   // cosh(x)
}
```

## Concrete complex numbers

Gaussian rationals are exact. Use `expand()` (or `expand_complex()`) to multiply out and `as_real_imag()` to divide; `abs_squared()` avoids the square root.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let a = ctx.complex(&ctx.int(3), &ctx.int(4));    // 3 + 4i
    let b = ctx.int(1) - &i * 2;                       // 1 − 2i

    println!("{}", (&a * &b).expand());               // -2*I + 11
    let (re, im) = (&a / &b).as_real_imag();
    println!("{re} + {im}i");                         // -1 + 2i
    println!("{}", a.abs_squared());                  // 25
    println!("{}", (ctx.int(1) + &i).arg().eval());   // 1/4*pi
    println!("{}", (ctx.int(1) + &i).powi(8).expand()); // 16
    println!("{}", (&i * &ctx.pi()).exp().eval());    // -1
    println!("{}", ctx.int(-1).ln().eval());          // pi*I
    println!("{}", ctx.int(-4).sqrt());               // 2*I
    println!("{}", ctx.int(-8).cbrt().eval());        // 2*cbrt(-1)   (principal: 1 + √3·i)
    println!("{}", ctx.int(-8).real_root(3).unwrap()); // -2          (the real cube root)
    println!("{}", i.exp().eval_complex64().unwrap()); // 0.5403…+0.8414…i  (a `Complex64`)
}
```

Current gaps, stated plainly: `abs(3 + 4i)` is not folded to `5` by `eval`/`simplify` (use `abs_squared()`), and `ln(i)` / `i^i` stay symbolic (evaluate with `eval_decimal`).

## Complex infinity

`1/0` evaluates to complex infinity `zoo` (`Context::complex_infinity()`), distinct from the signed real infinities `oo` and `-oo`. The parser accepts `zoo`.

## New constants

| Constant | `Context` method | Notes |
|----------|------------------|-------|
| γ (Euler–Mascheroni) | `euler_gamma()` | rationality unknown → assumption system leaves it open |
| G (Catalan) | `catalan()` | positive, real, finite |
| φ (golden ratio) | `golden_ratio()` | algebraic; `nsimplify` recognises `φ² − φ − 1 = 0`, `simplify` does not yet |
| ∞̃ | `complex_infinity()` | `zoo` |

All evaluate to arbitrary precision with `eval_decimal(digits)`.

## Special functions

0.2 adds `si`, `ci`, `ei`, `li`, `zeta`, `polygamma(n, x)` and `kronecker_delta(i, j)`, with exact special values, derivative rules and `evalf`; `Digamma` folds at positive integers and half-integers; Bessel `I`/`K` and orthogonal polynomials of any degree evaluate numerically; all Bessel functions and orthogonal polynomials have derivative rules.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);

    println!("{}", ctx.int(1).digamma().eval());                     // -EulerGamma
    println!("{}", ctx.rational(1, 2).digamma().eval());             // -EulerGamma - 2*ln(2)
    println!("{}", ctx.int(1).polygamma(&ctx.int(1)).eval());        // 1/6*pi^2
    println!("{}", ctx.int(1).polygamma(&ctx.int(2)).eval());        // -2*zeta(3)
    println!("{}", x.digamma().diff(&x));                            // polygamma(1, x)

    println!("{} {} {} {}", ctx.int(2).zeta().eval(), ctx.int(4).zeta().eval(),
        ctx.int(0).zeta().eval(), ctx.int(-1).zeta().eval());        // 1/6*pi^2 1/90*pi^4 -1/2 -1/12
    println!("{}", ctx.int(3).zeta().eval_decimal(30).unwrap());     // 1.20205690315959428539973816151

    println!("{} {}", ctx.int(0).si().eval(), ctx.infinity().si().eval());   // 0 1/2*pi
    println!("{} {}", x.si().diff(&x), x.li().diff(&x));             // sin(x)/x 1/ln(x)
    println!("{}", ctx.int(1).ei().eval_decimal(25).unwrap());       // 1.89511781635593675546652

    println!("{} {}", x.kronecker_delta(&x), ctx.int(1).kronecker_delta(&ctx.int(2)));   // 1 0
    println!("{}", ctx.int(2).bessel_k(&ctx.int(0)).eval_decimal(20).unwrap());  // 0.11389387274953343565
    println!("{}", ctx.rational(1, 3).legendre(&ctx.int(10)).eval()); // 13597/59049
}
```

The parser accepts all of these by name (`re`, `im`, `conjugate`/`conj`, `arg`, `si`, `ci`, `ei`, `li`, `zeta`, `polygamma`, `zoo`), and `compile()` / `to_rust_fn` / `to_c_fn` support Γ, lnΓ, ψ, erf/erfc, W, B, Bessel and orthogonal polynomials.

See `cargo run --example complex_analysis` for the full tour.

## More special functions (0.9)

0.9 adds the remaining SymPy special functions that show up as integration
results and in physics.  All of them are `Apply` nodes with SymPy's names, so
they print, parse (`ctx.parse("erfi(x)")`) and serialise like `besselj`; the
method lives on the *argument* and takes the parameters first, as with
`x.bessel_j(&nu)`.

| Function | Constructor | Exact values (`eval`) | Derivative |
|---|---|---|---|
| erfi, erf⁻¹, erfc⁻¹ | `x.erfi()`, `x.erfinv()`, `x.erfcinv()` | `erfi(0) = 0`, odd; `erfinv(±1) = ±∞`; `erfcinv(1) = 0` | `2e^{x²}/√π`; `(√π/2) e^{erfinv²}` |
| Eₙ, E₁ | `x.expint(&n)`, `x.e1()` | `Eₙ(0) = 1/(n−1)`, `E₀(x) = e^{−x}/x`, `Eₙ(∞) = 0` | `−Eₙ₋₁(x)` |
| Shi, Chi | `x.shi()`, `x.chi()` | `Shi(0) = 0`, odd; `Chi(0) = −∞` | `sinh x/x`, `cosh x/x` |
| Fresnel S, C | `x.fresnels()`, `x.fresnelc()` | `S(0) = 0`, `S(±∞) = ±1/2`, odd | `sin(πx²/2)`, `cos(πx²/2)` |
| γ(s,x), Γ(s,x) | `x.lowergamma(&s)`, `x.uppergamma(&s)` | closed forms for integer and half-integer `s` (`Γ(1,x) = e^{−x}`, `Γ(0,x) = E₁(x)`, `Γ(½,x) = √π erfc(√x)`, …) | `±x^{s−1}e^{−x}` |
| Liₛ(z) | `z.polylog(&s)` | `Liₛ(0) = 0`, `Liₛ(1) = ζ(s)`, `Liₛ(−1) = −η(s)`, `Li₁ = −ln(1−z)`, `Li₀ = z/(1−z)`, `Li₋ₙ` rational, `Li₂(½)` | `Liₛ₋₁(z)/z` |
| η(s) | `s.dirichlet_eta()` | `η(1) = ln 2`, `η(s) = (1−2^{1−s})ζ(s)` when `ζ(s)` folds | formal |
| Ai, Bi, Ai′, Bi′ | `x.airyai()`, `x.airybi()`, `x.airyaiprime()`, `x.airybiprime()` | values at `0` in terms of `Γ(⅓)`, `Γ(⅔)`; limits at `±∞` | `Ai′ = airyaiprime`, `Ai″ = x·Ai` |
| K(m), E(m) | `m.elliptic_k()`, `m.elliptic_e()` | `K(0) = E(0) = π/2`, `E(1) = 1`, `K(1) = z∞`, `K(½) = Γ(¼)²/(4√π)` | `(E − (1−m)K)/(2m(1−m))`, `(E − K)/(2m)` |
| F(φ\|m), Π(n\|m) | `phi.elliptic_f(&m)`, `n.elliptic_pi(&m)` | `F(0\|m) = 0`, `F(φ\|0) = φ`, `F(π/2\|m) = K(m)`; `Π(0\|m) = K(m)`, `Π(n\|0) = π/(2√(1−n))`, `Π(n\|n) = E(n)/(1−n)` | `∂φF = 1/√(1−m sin²φ)` (`∂ₘF` formal); both partials of `Π` |
| Cₙ^(a), Pₙ^(a,b), Pₙ^m, Lₙ^(a) | `x.gegenbauer(&n, &a)`, `x.jacobi(&n, &a, &b)`, `x.assoc_legendre(&n, &m)`, `x.assoc_laguerre(&n, &a)` | explicit polynomials for integer `n` (symbolic `a`, `b` allowed); `Cₙ^(½) = Pₙ`, `Cₙ^(1) = Uₙ`, `Pₙ^(0,0) = Pₙ^0 = Pₙ`, `Lₙ^(0) = Lₙ` | `2a Cₙ₋₁^(a+1)`, `(n+a+b+1)/2 Pₙ₋₁^(a+1,b+1)`, `(nxPₙ^m − (n+m)Pₙ₋₁^m)/(x²−1)`, `−Lₙ₋₁^(a+1)` |

`assoc_legendre` uses the Condon–Shortley phase like SymPy (`P₁¹(x) = −√(1−x²)`);
the elliptic integrals take the parameter `m = k²`.

Everything evaluates numerically to arbitrary precision (`eval_decimal(40)`
agrees with mpmath) for real arguments in the real domain: the error-function
family by series / asymptotics / Halley iteration, `Eₙ`, `γ`, `Γ` by series
and Legendre's continued fraction, `Liₛ` by the direct series or the
expansion in `ln z` (any real `z ≠ 1` for `s ≤ 0`, `|z| ≤ 1` otherwise),
Airy by Maclaurin series with guard bits or the large-`|x|` asymptotic
expansions, and the elliptic integrals by Carlson's symmetric forms
(`n < 1`, `m < 1`).  Arguments outside these domains (complex `Liₛ`, `K(m > 1)`,
`erfinv(|y| ≥ 1)`, …) return `Err(Unevaluable)`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let (x, z, a) = (ctx.symbol("x"), ctx.symbol("z"), ctx.symbol("a"));

    println!("{}", x.erfi().diff(&x));                                   // 2*exp(x^2)/sqrt(pi)
    println!("{}", ctx.rational(7, 10).erfi().eval_decimal(20).unwrap()); // 0.94028293383350747659
    println!("{}", x.uppergamma(&ctx.int(3)).eval());                    // x^2*exp(-x) + 2*x*exp(-x) + 2*exp(-x)
    println!("{}", ctx.int(1).polylog(&ctx.int(2)).eval());              // 1/6*pi^2
    println!("{}", z.polylog(&ctx.int(-1)).eval());                      // z*(-z + 1)^(-2)
    println!("{}", ctx.int(0).elliptic_k().eval());                       // 1/2*pi
    println!("{}", ctx.rational(1, 2).elliptic_k().eval_decimal(20).unwrap()); // 1.8540746773013719184
    println!("{}", x.gegenbauer(&ctx.int(2), &a).eval());                // 2*a^2*x^2 + 2*a*x^2 - a
    println!("{}", x.airyaiprime().diff(&x));                             // x*airyai(x)
    println!("{}", x.polylog(&ctx.int(2)).to_latex());                   // \operatorname{Li}_{2}\left(x\right)
}
```

`to_lean` returns `NotImplemented` for all of these (Mathlib has no standard
spelling), and `compile` / `to_rust_fn` / `to_c_fn` report the missing runtime
rather than generating code.
