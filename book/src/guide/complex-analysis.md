# Complex Analysis and Special Functions

New in 0.2. symplex is a CAS over ℂ: a symbol with no assumptions may be complex, and the library refuses to pretend otherwise.

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

`as_real_imag()` returns `(re, im)` as a pair; `expand_complex()` rewrites the expression as `re + im·I`; `polar()` returns `(modulus, argument)`.

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
    println!("{:?}", i.exp().eval_complex64());       // Ok((0.5403…, 0.8414…))
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
