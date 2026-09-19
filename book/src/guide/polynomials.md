# Polynomials as Data

An `Ex` is a tree. When you know an expression *is* a polynomial in some symbols, you usually want a different view of it: a finite list of `(monomial, coefficient)` pairs that you can index, iterate, multiply, evaluate exactly, and lay out as a matrix. In 0.3 that view is `Poly` (`symplex::poly_ex::Poly`), reached from any expression with `as_poly(&[&x, &y])`.

Two things distinguish `Poly` from the rational-coefficient machinery in `symplex::multipoly`:

- **Coefficients are `Ex`.** They may be exact rationals or symbolic *parameters* — anything free of the generators. `a·x² + (a + b)·x + 3` is a perfectly good polynomial in `x`.
- **Nothing is approximated.** Every operation is exact, and `to_ex()` rebuilds an expression equal to the (expanded) input.

Terms are always reported in **descending lexicographic order** of the exponent vectors — the order of SymPy's `Poly.terms()`.

## Viewing an expression as a polynomial

`as_poly` (equivalently `Poly::new`) expands the expression and collects it by monomial. It returns `None` if a generator appears in a non-polynomial position — inside a function, under a negative or fractional power, or in an exponent.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    let e = (&x + &y * 2).powi(2) * &x - &y.powi(3);
    let p = e.as_poly(&[&x, &y]).unwrap();
    println!("{p}");                                    // Poly(x^3 + 4*x^2*y + 4*x*y^2 - y^3, x, y)
    for (mono, coeff) in p.terms() {
        println!("x^{} y^{}  ·  {coeff}", mono[0], mono[1]);
    }
    // x^3 y^0  ·  1
    // x^2 y^1  ·  4
    // x^1 y^2  ·  4
    // x^0 y^3  ·  -1
    println!("{:?}", p.monoms());                       // [[3, 0], [2, 1], [1, 2], [0, 3]]
    println!("{:?}", p.coeffs());                       // [Ex(1), Ex(4), Ex(4), Ex(-1)]
    println!("{}", p.coeff_monomial(&[1, 2]).unwrap()); // 4
    println!("{}", p.coeff_monomial(&[5, 0]).unwrap()); // 0   (absent monomials are zero)
    println!("{:?} {:?} {:?}", p.total_degree(), p.degree_in(&x), p.degree_list());
    // Some(3) Some(3) [3, 3]
    println!("{} {:?}", p.leading_coeff(), p.leading_monomial());   // 1 Some([3, 0])
    println!("{} {} {}", p.num_terms(), p.is_homogeneous(), p.has_rational_coeffs());
    // 4 true true
    println!("{}", p.to_ex());                          // x^3 + 4*x*y^2 - y^3 + 4*y*x^2

    assert!(x.sin().as_poly(&[&x]).is_none());          // generator inside a function
    assert!((ctx.int(1) / &x).as_poly(&[&x]).is_none()); // negative power
    assert!(x.pow(&y).as_poly(&[&x]).is_none());        // generator in an exponent
}
```

Other structural queries: `is_zero`, `is_ground` (constant), `is_univariate`, `is_linear`, `gens()`, `num_gens()`, `leading_term()`, and `equals(&other)` (same generators, identical normalised coefficients). `Poly::from_terms(&ctx, &gens, vec![(exps, coeff), …])`, `Poly::zero`, `Poly::one` and `Poly::constant` build polynomials directly.

## Symbolic coefficients

Any symbol that is *not* a generator becomes part of the coefficients. The same expression can be viewed with different generator lists, and the plain `Ex` methods `degree`, `coeffs`, `coeff`, `leading_coeff` and `is_polynomial` now accept parameter coefficients too (in 0.2 they required rational coefficients).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, a, b);

    let e = &a * &x.powi(2) + (&a + &b) * &x + &x.powi(2) + 3;
    println!("{e}");                                     // a*x^2 + x^2 + x*(a + b) + 3

    // Ex methods — ascending order, like 0.2:
    println!("{:?}", e.degree(&x));                      // Some(2)
    let cs: Vec<String> = e.coeffs(&x).unwrap().iter().map(|c| c.to_string()).collect();
    println!("{cs:?}");                                  // ["3", "a + b", "a + 1"]
    println!("{}", e.coeff(&x, 1).unwrap());             // a + b
    println!("{}", e.leading_coeff(&x).unwrap());        // a + 1
    println!("{}", e.is_polynomial(&x));                 // true

    // Poly view in x alone — all_coeffs is dense and highest-degree first (SymPy order):
    let p = e.as_poly(&[&x]).unwrap();
    let dense: Vec<String> = p.all_coeffs().unwrap().iter().map(|c| c.to_string()).collect();
    println!("{dense:?}");                               // ["a + 1", "a + b", "3"]
    println!("{}", p.has_rational_coeffs());             // false

    // Promote a to a generator: now b is the only parameter.
    let q = e.as_poly(&[&x, &a]).unwrap();
    for (mono, coeff) in q.terms() {
        println!("x^{} a^{}  ·  {coeff}", mono[0], mono[1]);
    }
    // x^2 a^1  ·  1
    // x^2 a^0  ·  1
    // x^1 a^1  ·  1
    // x^1 a^0  ·  b
    // x^0 a^0  ·  3

    println!("{:?}", x.pow(&a).degree(&x));              // None  (x^a is not polynomial in x)
}
```

Note the two orderings: `Ex::coeffs` is **ascending** (`[a₀, a₁, …]`, unchanged from 0.2), while `Poly::all_coeffs` is **descending** with zeros filled in, matching SymPy's `all_coeffs()`.

## Exact evaluation and arithmetic

`eval` substitutes a value for every generator and evaluates; values may be rationals, radicals or expressions. `eval_gen` substitutes one generator (by a constant *or* a polynomial in the remaining generators) and returns a `Poly` with one generator fewer. Arithmetic (`add`, `sub`, `mul`, `neg`, `scale`, `pow`, `derivative`) requires identical generator lists and returns `Result`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    let p = (&x.powi(2) * &y - &y / 2 + ctx.rational(1, 3)).as_poly(&[&x, &y]).unwrap();
    println!("{}", p.eval(&[&ctx.rational(3, 2), &ctx.rational(-4, 5)]).unwrap());  // -16/15
    println!("{}", p.eval(&[&ctx.int(2).sqrt(), &ctx.int(1)]).unwrap());            // 11/6
    println!("{}", p.eval_gen(&x, &ctx.int(2)).unwrap());        // Poly(7/2*y + 1/3, y)
    println!("{}", p.eval_gen(&x, &(&y + 1)).unwrap());          // Poly(y^3 + 2*y^2 + 1/2*y + 1/3, y)
    println!("{}", p.derivative(&y).unwrap().to_ex());          // x^2 - 1/2

    let s = (&x + &y).as_poly(&[&x, &y]).unwrap();
    let d = (&x - &y).as_poly(&[&x, &y]).unwrap();
    println!("{}", s.mul(&d).unwrap().to_ex());                  // x^2 - y^2
    println!("{}", s.pow(3).unwrap().to_ex());                   // x^3 + 3*x*y^2 + y^3 + 3*y*x^2
    println!("{}", s.add(&d).unwrap().to_ex());                  // 2*x
    println!("{}", s.sub(&d).unwrap().to_ex());                  // 2*y
    println!("{}", s.neg().to_ex());                             // -x - y
    println!("{}", s.scale(&ctx.rational(1, 2)).unwrap().to_ex()); // 1/2*x + 1/2*y
    assert!(s.mul(&d).unwrap().equals(&(&x.powi(2) - &y.powi(2)).as_poly(&[&x, &y]).unwrap()));
    assert!(s.add(&(&x + 1).as_poly(&[&x]).unwrap()).is_err()); // different generators

    // Rational-coefficient helpers
    let q = (&x.powi(2) * -4 + &x * 6).as_poly(&[&x]).unwrap();
    let (c, prim) = q.content_and_primitive().unwrap();
    println!("{c} · ({})", prim.to_ex());                        // -2 · (2*x^2 - 3*x)
    println!("{}", q.monic().unwrap().to_ex());                  // x^2 - 3/2*x
}
```

## Coefficient matrices and exact linear systems

The reason `Poly` exists is to make questions like *"is `goal` a linear combination of `h₁, …, hₖ`?"* mechanical. `Poly::monomial_basis` collects every monomial that occurs in a family, and `Poly::coefficient_matrix` lays the family out with one **row per monomial** and one **column per polynomial**. The unknown multipliers `λ` then satisfy `M·λ = coefficients of goal`, which `linsolve_matrix` solves exactly — including the under- and over-determined cases.

```rust
use symplex::prelude::*;
use symplex::poly_ex::Poly;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let h1 = (&x + 1).as_poly(&[&x]).unwrap();
    let h2 = (&x.powi(2) - 1).as_poly(&[&x]).unwrap();
    let goal = (&x + 1).powi(2).as_poly(&[&x]).unwrap();

    let basis = Poly::monomial_basis(&[&h1, &h2, &goal]).unwrap();
    println!("{basis:?}");                                // [[2], [1], [0]]
    let m = Poly::coefficient_matrix(&[&h1, &h2], &basis).unwrap();
    println!("{m}");
    let rhs: Vec<Ex> = basis.iter().map(|mono| goal.coeff_monomial(mono).unwrap()).collect();
    match linsolve_matrix(&m, &Matrix::col_vector(rhs)).unwrap() {
        LinearSolution::Unique(pairs) => {
            for (var, val) in pairs {
                println!("{var} = {val}");                // x1 = 2, x2 = 1
            }
        }
        other => println!("{other:?}"),
    }
    // So (x + 1)² = 2·(x + 1) + 1·(x² − 1).

    // x² + x + 1 is not in the span:
    let goal2 = (&x.powi(2) + &x + 1).as_poly(&[&x]).unwrap();
    let rhs2: Vec<Ex> = basis.iter().map(|mono| goal2.coeff_monomial(mono).unwrap()).collect();
    println!("{:?}", linsolve_matrix(&m, &Matrix::col_vector(rhs2)).unwrap());   // Inconsistent
}
```

Output of the matrix:

```text
[
  [0,  1],
  [1,  0],
  [1, -1]
]
```

When the multipliers must be **non-negative** — the situation in every positivity certificate — feed the same matrix to the exact LP solver instead: `Matrix::to_rational_rows()` gives the rows in the form `linprog::feasible_nonneg` wants. The [Polynomial Inequality Certificates](../cookbook/polynomial-certificates.md) cookbook entry does this end to end, and [Exact Linear Programming](./exact-lp.md) describes the solver.

## Rational normal form: `ratsimp`

`ratsimp` puts a rational expression into a canonical `P/Q`: one fraction, common factors cancelled by a multivariate GCD, integer-primitive numerator and denominator, and a positive leading coefficient in `Q`. Maximal non-rational subexpressions (`sin x`, `π`, `√x`) are treated as opaque indeterminates, exactly as SymPy's `cancel` does. `simplify_rational` is now the same normal form.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, r, j);

    println!("{}", ((&x.powi(2) - &y.powi(2)) / (&x - &y)).ratsimp());   // x + y
    println!("{}", (ctx.int(1) / &x + ctx.int(1) / &y).ratsimp());         // (x + y)/(x*y)
    let nested = ctx.int(1) / (&x + ctx.int(1) / &y) + ctx.int(1) / (&y + ctx.int(1) / &x);
    println!("{nested}  →  {}", nested.ratsimp());
    // 1/(x + 1/y) + 1/(1/x + y)  →  (x + y)/(x*y + 1)
    println!("{}", (x.sin().powi(2) / x.sin()).ratsimp());                 // sin(x)
    println!("{}", ((x.sin().powi(2) - x.cos().powi(2)) / (x.sin() - x.cos())).ratsimp());
    // sin(x) + cos(x)   — a difference of squares, no trig identity involved
    let (n, d) = (ctx.int(1) / &x + ctx.int(1) / (&x + 1)).ratsimp().as_numer_denom();
    println!("{n}  /  {d}");                                               // 2*x + 1  /  x^2 + x
    let mixed = (&x.powi(2) * 2 + &x * 4) / (&x * 6 + 12) + ctx.rational(1, 3);
    println!("{mixed}  →  {}", mixed.ratsimp());
    // (2*x^2 + 4*x)/(6*x + 12) + 1/3  →  1/3*x + 1/3
    println!("{}", (&x + 1).ratsimp());                                    // x + 1   (already normal)
    println!("{}", x.exp().ratsimp());                                     // exp(x)  (unchanged)

    // `solve` with parameter coefficients returns ratsimp'd solutions.
    let eqn = (&r * 3 - 1) / (&j + 1) - (&r + 1) / (&j * 2);
    println!("{}", eqn.solve(&r).unwrap()[0]);                             // (3*j + 1)/(5*j - 1)
    println!("{}", eqn.simplify_rational());   // (5*j*r - 3*j - r - 1)/(2*j^2 + 2*j)
}
```

`ratsimp` is the right tool for *checking an identity*: `(lhs − rhs).ratsimp()` is structurally `0` exactly when the two sides agree as rational functions. `cancel(&x)` (single variable) and `together()` (no cancellation) remain available for lighter-weight jobs.

## Sign of a polynomial on an interval

`poly_is_nonnegative_on(&x, &lo, &hi)` and `poly_is_positive_on` decide, **exactly**, whether a univariate polynomial with rational coefficients is `≥ 0` (resp. `> 0`) on the closed interval `[lo, hi]`. The method is a square-free decomposition (to find the roots where the sign can change), a Sturm count to check that none lies strictly inside the interval, and one sample point. Endpoints must be rationals or `±∞`. The answer is `None` for non-polynomial input or symbolic coefficients.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, a);
    let (ninf, inf) = (ctx.neg_infinity(), ctx.infinity());

    let sq = &x.powi(2) - &x * 2 + 1;                                  // (x − 1)²
    println!("{:?} {:?}",
        sq.poly_is_nonnegative_on(&x, &ninf, &inf),
        sq.poly_is_positive_on(&x, &ninf, &inf));                      // Some(true) Some(false)

    let cubic = &x.powi(3) - &x;
    println!("{:?} {:?} {:?}",
        cubic.poly_is_nonnegative_on(&x, &ctx.int(2), &inf),
        cubic.poly_is_nonnegative_on(&x, &ctx.int(-2), &inf),
        cubic.poly_is_nonnegative_on(&x, &ctx.int(-1), &ctx.int(0)));  // Some(true) Some(false) Some(true)

    let wobble = &x.powi(4) - &x.powi(2) * 5 + 4;                      // (x² − 1)(x² − 4)
    println!("{:?} {:?} {:?}",
        wobble.poly_is_nonnegative_on(&x, &ctx.rational(-1, 1), &ctx.rational(1, 1)),
        wobble.poly_is_positive_on(&x, &ctx.rational(-1, 1), &ctx.rational(1, 1)),
        wobble.poly_is_nonnegative_on(&x, &ctx.rational(3, 2), &ctx.rational(7, 4)));
    // Some(true) Some(false) Some(false)

    println!("{:?} {:?}",
        (&x.powi(2) + 1).poly_is_positive_on(&x, &ninf, &inf),
        (&a * &x + 1).poly_is_nonnegative_on(&x, &ninf, &inf));        // Some(true) None
    println!("{:?}", x.sin().poly_is_nonnegative_on(&x, &ninf, &inf));  // None
    println!("{:?}", wobble.count_real_roots(&x));                       // Some(4)
}
```

For the roots themselves use `count_real_roots`, `real_roots_isolate` and `nroots` (see [Algebra](./algebra.md#polynomial-algebra-on-ex)).

## Numeric roots

`Poly::nroots(digits)` is `Ex::nroots` for the univariate, rational-coefficient case; anything else is an `InvalidArgument` error rather than a wrong answer.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, a);
    let p = (&x.powi(5) - &x - 1).as_poly(&[&x]).unwrap();
    for (re, im) in p.nroots(12).unwrap() {
        println!("{re:.10} {im:+.10}i");
    }
    // -0.7648844336 -0.3524715460i
    // -0.7648844336 +0.3524715460i
    //  0.1812324445 -1.0839541013i
    //  0.1812324445 +1.0839541013i
    //  1.1673039783 +0.0000000000i
    println!("{}", (&a * &x + 1).as_poly(&[&x]).unwrap().nroots(10).unwrap_err());
    // Poly::nroots: invalid argument: polynomial must have rational coefficients
    println!("{}", (&x * &a).as_poly(&[&x, &a]).unwrap().nroots(10).unwrap_err());
    // Poly::nroots: invalid argument: polynomial must be univariate
}
```

## Bridge to `MultiPoly` and Gröbner bases

`symplex::groebner` and `symplex::polysys` work on `MultiPoly<GrevLex>`, a rational-coefficient sparse polynomial indexed by variable *position*. `to_multipoly()` converts a `Poly` whose coefficients are all rational (`None` otherwise), and `Poly::from_multipoly(&ctx, &gens, &mp)` converts back, so you can move between the two worlds without touching the low-level representation.

```rust
use symplex::prelude::*;
use symplex::groebner::groebner_basis;
use symplex::poly_ex::Poly;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    let f1 = (&x.powi(2) + &y.powi(2) - 1).as_poly(&[&x, &y]).unwrap();
    let f2 = (&x - &y).as_poly(&[&x, &y]).unwrap();
    let gb = groebner_basis(&[f1.to_multipoly().unwrap(), f2.to_multipoly().unwrap()]);
    for g in &gb {
        println!("{}", Poly::from_multipoly(&ctx, &[&x, &y], g).unwrap().to_ex());
    }
    // y^2 - 1/2
    // x - y

    // Normal form of x³ modulo the ideal:
    let target = x.powi(3).as_poly(&[&x, &y]).unwrap().to_multipoly().unwrap();
    let refs: Vec<_> = gb.iter().collect();
    let rem = target.reduce(&refs);
    println!("{}", Poly::from_multipoly(&ctx, &[&x, &y], &rem).unwrap().to_ex());   // 1/2*y

    let a = ctx.symbol("a");
    println!("{:?}", (&a * &x).as_poly(&[&x]).unwrap().to_multipoly().is_none());    // true
}
```

`symplex::polysys::solve_system_ex` (see [Solving Equations](./solving.md#polynomial-systems)) does the whole pipeline — Gröbner basis, triangularisation, algebraic back-substitution — when what you want is the solution set rather than the basis.

See `cargo run --example polynomials` for the complete program these snippets are drawn from.

## Algebraic numbers and polynomial algebra (0.9)

0.9 adds a layer of `Ex` methods over the exact engines above, so the common polynomial-algebra questions no longer need a detour through `Poly`/`MultiPoly`. Every method takes and returns `Ex`; the ones that answer a *query* return `Option` (`None` when the input is not of the required shape), the ones that validate caller-supplied structure return `Result`.

### Minimal polynomials

`minimal_polynomial(&var)` (SymPy `minimal_polynomial`) returns the minimal polynomial over ℚ of an algebraic constant built from rationals, radicals, `i`, `φ`, sums, products, integer powers and reciprocals. The result is integer-primitive with a positive leading coefficient, exactly as SymPy prints it; `None` means the number was not recognised as algebraic (`π`, `e`, free symbols, transcendental functions).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let a = ctx.int(2).sqrt() + ctx.int(3).sqrt();
    println!("{}", a.minimal_polynomial(&x).unwrap());          // x^4 - 10*x^2 + 1
    let cbrt2 = ctx.int(2).pow(&ctx.rational(1, 3));
    println!("{}", cbrt2.minimal_polynomial(&x).unwrap());      // x^3 - 2
    println!("{}", ctx.rational(3, 4).minimal_polynomial(&x).unwrap());   // 4*x - 3
    let b = ctx.int(1) / (ctx.int(1) + ctx.int(2).sqrt());
    println!("{}", b.minimal_polynomial(&x).unwrap());          // x^2 + 2*x - 1
    assert!(ctx.pi().minimal_polynomial(&x).is_none());
}
```

### Multivariate gcd and lcm without naming variables

`gcd_all` / `lcm_all` (SymPy `gcd(f, g)` / `lcm(f, g)`) treat both inputs as polynomials over ℚ in *all* of their free symbols. The result follows `MultiPoly::gcd`'s normalisation: integer coefficients, positive leading coefficient (grevlex), integer content equal to the gcd of the inputs' contents — for polynomials over ℤ that is the ordinary gcd over ℤ, `gcd(2x, 4x) = 2x`. Non-polynomial input (`sin x`, `1/x`, `π·x`) gives `None`.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);

    let f = &x.powi(2) - &y.powi(2);
    println!("{}", f.gcd_all(&(&x - &y)).unwrap());              // x - y
    println!("{}", f.lcm_all(&(&x - &y)).unwrap());              // x^2 - y^2
    println!("{}", (&x * &y * &z + &x * &y).gcd_all(&(&x * &z + &x)).unwrap());   // x*z + x
    println!("{}", (&x * 2).gcd_all(&(&x * 4)).unwrap());        // 2*x
    assert!(x.sin().gcd_all(&x).is_none());
}
```

### Gröbner bases and normal forms from `Ex`

`Ex::groebner(&polys, &vars, order)` computes the reduced (monic) Gröbner basis in the given variables under `MonomialOrder::Lex` or `MonomialOrder::GrevLex` (`symplex::multipoly::MonomialOrder`); `reduce_modulo(&basis, &vars, order)` is the remainder of multivariate division — the unique normal form when `basis` is a Gröbner basis for that order, so it is zero exactly for members of the ideal. Variables must be distinct symbols and every polynomial must have rational coefficients; anything else is an `InvalidArgument` error naming the offending input.

```rust
use symplex::multipoly::MonomialOrder;
use symplex::prelude::*;

fn main() -> Result<(), SymplexError> {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let vars = [x.clone(), y.clone()];

    let gens = [&x.powi(2) + &y.powi(2) - 1, &x - &y];
    let lex = Ex::groebner(&gens, &vars, MonomialOrder::Lex)?;
    for g in &lex {
        println!("{g}");
    }
    // x - y
    // y^2 - 1/2
    let grevlex = Ex::groebner(&gens, &vars, MonomialOrder::GrevLex)?;   // [y^2 - 1/2, x - y]

    println!("{}", x.powi(3).reduce_modulo(&lex, &vars, MonomialOrder::Lex)?);   // 1/2*y
    let member = (&x.powi(2) - &y.powi(2)).reduce_modulo(&grevlex, &vars, MonomialOrder::GrevLex)?;
    assert!(member.is_zero_structural());
    Ok(())
}
```

### Exact real roots

`real_roots(&var)` lists the distinct real roots of a rational-coefficient polynomial in increasing order (decided exactly with Sturm sequences): rational roots as numbers, every other root as the `RootOf(g, k)` node that `solve` already uses for degree ≥ 5, where `g` is the irreducible factor over ℤ and `k` its index among `g`'s complex roots. `RootOf` evaluates numerically (`eval_f64`, `eval_decimal`) and prints as such; `root_of(&var, k)` is the `k`-th real root (0-based). Unlike SymPy's `real_roots`, a repeated root is listed once (as in `count_real_roots`).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let roots = (&x.powi(3) - &x * 2).real_roots(&x).unwrap();
    for r in &roots {
        println!("{r}  ≈ {}", r.eval_f64().unwrap());
    }
    // RootOf(x^2 - 2, 0)  ≈ -1.41421356…
    // 0  ≈ 0
    // RootOf(x^2 - 2, 1)  ≈ 1.41421356…
    let largest = (&x.powi(3) - &x * 2).root_of(&x, 2).unwrap();   // RootOf(x^2 - 2, 1)
    assert_eq!(largest, roots[2]);
    assert_eq!((&x.powi(2) + 1).real_roots(&x), Some(vec![]));     // no real roots
}
```

### Factoring modulo a prime

`factor_mod(&var, p)` (SymPy `factor_list(f, modulus=p)`) factors a rational-coefficient polynomial over `GF(p)` into `(lc, [(monic irreducible factor, multiplicity)])` with coefficients in `[0, p)`. A composite `p` is an `InvalidArgument` error, as is a coefficient whose denominator is divisible by `p`; the finite-field arithmetic supports odd primes below 2³¹.

```rust
use symplex::prelude::*;

fn main() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let (lc, factors) = (&x.powi(2) + 1).factor_mod(&x, 5)?;
    println!("{lc}: {:?}", factors.iter().map(|(f, m)| format!("({f})^{m}")).collect::<Vec<_>>());
    // 1: ["(x + 2)^1", "(x + 3)^1"]
    let (_, factors) = (&x.powi(2) + 1).factor_mod(&x, 3)?;     // irreducible mod 3
    assert_eq!(factors.len(), 1);
    assert!((&x.powi(2) + 1).factor_mod(&x, 6).is_err());
    Ok(())
}
```

### Resultants and discriminants with symbolic coefficients

`resultant` and `discriminant` (0.2) require every coefficient to be rational. `resultant_symbolic` and `discriminant_symbolic` accept parameter coefficients: they build the Sylvester matrix over `Ex` entries, take its determinant with `Matrix::det`, and expand, so the answer is a polynomial in the parameters. `discriminant_symbolic` is division-free (the leading coefficient is eliminated with one row operation before the determinant).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, a, b, c, p, q);

    let quad = &a * &x.powi(2) + &b * &x + &c;
    println!("{}", quad.discriminant_symbolic(&x).unwrap());          // -4*a*c + b^2
    let cubic = &x.powi(3) + &p * &x + &q;
    println!("{}", cubic.discriminant_symbolic(&x).unwrap());         // -4*p^3 - 27*q^2
    println!("{}", (&x - &a).resultant_symbolic(&(&x - &b), &x).unwrap());   // a - b
}
```
