# Algebra

Expansion, factoring, rational functions, polynomial algebra, and simplification. The pattern-matching engine that powers `simplify` is covered in [The Rule Engine](./rule-engine.md); the `Poly` view of an expression as explicit `(monomial, coefficient)` data is in [Polynomials as Data](./polynomials.md).

## Expansion and collection

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, t);

    println!("{}", expr!(ctx, (x + 1)^3).expand());               // x^3 + 3*x^2 + 3*x + 1
    println!("{}", (&x + &y).powi(3).expand_multinomial());        // x^3 + 3*x*y^2 + y^3 + 3*y*x^2
    println!("{}", (&x * &y + &x * &t + &y * &t + &x).rcollect(&[&x, &y]));   // t*y + x*(t + y + 1)
    println!("{}", expr!(ctx, x^2 * y + x * y^2).collect(&x));

    // expand_with: deep = false stops at function boundaries
    let opts = ExpandOpts { deep: false, ..ExpandOpts::default() };
    let e = (&x + 1) * &((&y + 1).powi(2)).sin();
    println!("{}", e.expand_with(&opts));          // x*sin((y + 1)^2) + sin((y + 1)^2)
}
```

`expand()` no longer splits `(x·y)^a` for symbols of unknown sign — that identity fails over ℂ. `expand_power_base(true)` forces it when you know it is safe.

## Factoring

Univariate factoring over ℤ uses **Berlekamp–Zassenhaus** (any degree; 0.1 was limited to small Kronecker degrees). Multivariate factoring uses Kronecker substitution. `factor_list` returns the content and `(factor, multiplicity)` pairs.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    println!("{}", expr!(ctx, x^12 - 1).factor(&x));
    // (x - 1)*(x + 1)*(x^2 + x + 1)*(x^2 + 1)*(x^2 - x + 1)*(x^4 - x^2 + 1)
    let p = ((&x.powi(5) - &x - 1) * (&x.powi(4) + &x + 1) * (&x.powi(2) + 1)).expand();
    println!("{}", p.factor(&x));                  // (x^5 - x - 1)*(x^4 + x + 1)*(x^2 + 1)
    let (content, factors) = p.factor_list(&x);
    println!("{content} {:?}", factors.iter().map(|(f, m)| format!("({f})^{m}")).collect::<Vec<_>>());

    println!("{}", expr!(ctx, x^3 - x*y^2 + x^2 - y^2).factor_all());   // (x + 1)*(x + y)*(x - y)
    println!("{}", expr!(ctx, x^2 + 1).factor(&x));                    // x^2 + 1 (irreducible over ℤ)
    println!("{:?}", expr!(ctx, x^4 + 1).is_irreducible(&x));           // Some(true)
}
```

## Rational functions

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    println!("{}", expr!(ctx, (x^2 - 1) / (x - 1)).cancel(&x));       // x + 1
    println!("{}", expr!(ctx, 1 / (x^2 - 1)).partial_fractions(&x));  // -1/(2*(x + 1)) + 1/(2*(x - 1))
    println!("{}", (1 / &x + 1 / (&x + 1)).together());
    let (num, den) = expr!(ctx, (x + 1) / (x - 1)).as_numer_denom();
    println!("{num} / {den}");
}
```

### Rational normal form: `ratsimp`

New in 0.3, `ratsimp` is the canonical form for rational expressions in *all* variables at once: a single fraction `P/Q` with common polynomial factors cancelled (multivariate GCD), integer-primitive numerator and denominator, and a positive leading coefficient in `Q`. Non-rational subexpressions (`sin x`, `π`, `√x`) are treated as opaque indeterminates, exactly like SymPy's `cancel`. `simplify_rational` now produces the same normal form, and `solve` uses it for solutions with symbolic coefficients.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, a);

    println!("{}", (ctx.int(1) / (&x + ctx.int(1) / &y) + ctx.int(1) / (&y + ctx.int(1) / &x)).ratsimp());
    // (x + y)/(x*y + 1)
    println!("{}", ((&x.powi(2) - &y.powi(2)) / (&x - &y)).ratsimp());   // x + y
    println!("{}", (ctx.int(1) / &x + ctx.int(1) / (&x + 1)).ratsimp());  // (2*x + 1)/(x^2 + x)

    // degree / coeffs / coeff / leading_coeff / is_polynomial accept parameter coefficients
    let e = &a * &x.powi(2) + (&a + 1) * &x + 3;
    println!("{:?}", e.degree(&x));                                        // Some(2)
    println!("{:?}", e.coeffs(&x).map(|cs| cs.iter().map(|c| c.to_string()).collect::<Vec<_>>()));
    // Some(["3", "a + 1", "a"])
    println!("{}", e.leading_coeff(&x).unwrap());                          // a
    println!("{}", e.is_polynomial(&x));                                   // true
}
```

`(lhs - rhs).ratsimp()` is `0` exactly when two rational expressions agree — the cheapest way to check an identity. For the polynomial *data* behind these expressions (terms, coefficient matrices, exact evaluation, sign on an interval) see [Polynomials as Data](./polynomials.md).

## Polynomial algebra on `Ex`

New in 0.2 (rational coefficients unless noted): `resultant`, `discriminant`, `sqf_list`, `square_free_part`, `is_squarefree`, `poly_div`/`poly_quo`/`poly_rem`, `poly_gcdex`, `poly_gcd`/`poly_lcm`, `decompose`, `content_primitive`, `leading_coeff`, `monic`, `poly_compose`, `poly_shift`, `poly_reverse`, `poly_interpolate`, `count_real_roots`, `real_roots_isolate`, `nroots`. All return `Option`/`Result` and give `None` for non-polynomial input. Since 0.3, `degree`, `coeffs`, `coeff`, `leading_coeff` and `is_polynomial` also accept symbolic (parameter) coefficients, and `poly_is_nonnegative_on` / `poly_is_positive_on` decide the sign of a polynomial on an interval exactly (see [Polynomials as Data](./polynomials.md#sign-of-a-polynomial-on-an-interval)).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    println!("{}", (&x.powi(2) + 1).resultant(&(&x.powi(2) - 2), &x).unwrap());     // 9
    println!("{}", (&x.powi(3) - &x).discriminant(&x).unwrap());                    // 4
    let (q, r) = (&x.powi(3) + &x * 2 + 1).poly_div(&(&x.powi(2) + 1), &x).unwrap();
    println!("q = {q}, r = {r}");                                                   // q = x, r = x + 1
    let e = (&x.powi(2) - 1).poly_gcdex(&(&x.powi(2) - &x * 2 + 1), &x).unwrap();   // ExtendedGcd { gcd, x, y }
    println!("{}·f + {}·g = {}", e.x, e.y, e.gcd);                                   // 1/2·f + -1/2·g = x - 1
    let sq = ((&x - 1).powi(2) * (&x + 2).powi(3) * &x).expand();
    let (_, sqf) = sq.sqf_list(&x).unwrap();
    println!("{:?}", sqf.iter().map(|(f, m)| format!("({f})^{m}")).collect::<Vec<_>>());
    // ["(x)^1", "(x - 1)^2", "(x + 2)^3"]
    println!("{:?}", (&x.powi(4) + &x.powi(2) * 2 + 1).decompose(&x).iter().map(|e| e.to_string()).collect::<Vec<_>>());
    // ["x^2 + 2*x + 1", "x^2"]   (outer ∘ inner)
    let pts = [(ctx.int(0), ctx.int(1)), (ctx.int(1), ctx.int(3)), (ctx.int(2), ctx.int(9))];
    println!("{}", Ex::poly_interpolate(&pts, &x).unwrap());                        // 2*x^2 + 1

    let q5 = &x.powi(5) - &x - 1;
    println!("{:?}", q5.count_real_roots(&x));                                      // Some(1)
    for Complex64 { re, im } in q5.nroots(&x, 12).unwrap() {
        println!("{re:.10} {im:+.10}i");
    }
    // Isolating intervals carry their kind: a Sturm cell is `(lo, hi]`, a root
    // hit exactly by the bisection is the point `[r, r]`.
    println!("{:?}", (&x.powi(3) - &x * 2 - 5).real_roots_isolate(&x).iter()
        .map(|iv| iv.to_string()).collect::<Vec<_>>());
    // ["(17157/8192, 4291/2048]"]
}
```

Since 0.18 these scale to high degree: root isolation and counting
evaluate signs in `ℤ[x]` and `real_roots` / `root_of` start the numeric
root finder behind a `RootOf` index from the Newton polygon of the
coefficients, so the root in `(0, 1)` of a degree-50 binomial-tail
polynomial (`stats::aggregation::proportion_interval_exact`) is named in
a few seconds even in a debug build — previously `root_of` gave `None`
at degree ≥ 40.  `root_of` names only the requested root, so prefer it to
`real_roots(&x)[k]` for one root of a large polynomial.  Should the
factorisation over ℤ not be certified complete (an exhausted
recombination budget, not a matter of degree), `RootOf(g, k)` names a
square-free but possibly reducible `g`: it still evaluates correctly but
is not a canonical form.  `nroots` returns exactly `0.0` for a real part
that is below the iteration's convergence tolerance (`10⁻³⁰` relative to
`max(1, |z|)`), so `x² + 1` gives `±i` rather than `-7.7e-93 ± i`, while
genuinely tiny roots (`x² − 10⁻⁴⁰` → `±10⁻²⁰`) are kept; whether a root
is *real* is still decided exactly.

## Simplification

`simplify()` runs a dozen strategies (eval, expand, factor, trig, log, cancel, power, radical, assumption-aware refinement, …), keeps the result with the fewest operations, and iterates to a fixpoint. `simplify_with(&SimplifyOpts)` controls the iteration count; `simplify_traced` shows what fired. Targeted simplifiers are available when you know which identity you want:

| Method | Does |
|--------|------|
| `simplify_trig`, `fu`, `trig_power_linearize`, `trig_half_angle` | Trigonometric identities (Fu's algorithm) |
| `expand_trig`, `trig_combine` | `sin(a+b)` ↔ products |
| `expand_log`, `log_combine`, `expand_log_with(force)`, `log_combine_with(force)` | Logarithm rules (guarded by assumptions unless forced) |
| `simplify_powers`, `powdenest(force)`, `expand_power_base(force)`, `expand_power_exp(force)` | Power rules |
| `sqrtdenest` | `√(5 + 2√6)` → `√2 + √3` |
| `signsimp` | Canonicalise signs: `−t·(−x − y)` → `t·(x + y)` |
| `rationalize_denom`, `simplify_rational` | Radical and rational denominators |
| `simplify_combinatorial` | Factorials, binomials, Gamma |
| `nsimplify(tol)`, `nsimplify_with_constants(&[&pi], tol)` | Recognise a float as a rational or a rational multiple of a constant |
| `refine`, `refine_with` | Apply assumptions (`√(x²)` → `x` for `x ≥ 0`) |
| `rewrite_as_exp`, `rewrite_as_trig` | Change representation |
| `separate_vars`, `separate_vars_additive`, `separate_vars_dict` | Split products/sums by variable groups |
| `subs_algebraic(&old, &new)` | Substitute `x²` → `u` inside `x⁴`, `x³`, `1/x²` |

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, t);

    println!("{}", (&x.sin().powi(2) + &x.cos().powi(2)).simplify());          // 1
    println!("{}", (ctx.int(5) + ctx.int(24).sqrt()).sqrt().sqrtdenest());     // sqrt(2) + sqrt(3)
    println!("{}", ((-&x - &y) * (-&t)).signsimp());                           // t*(x + y)
    println!("{} / {}", x.powi(2).sqrt().powdenest(false), x.powi(2).sqrt().powdenest(true)); // abs(x) / x
    println!("{}", ctx.from_f64(0.333333333333).unwrap().nsimplify(1e-9));    // 1/3
    println!("{}", ctx.from_f64(std::f64::consts::PI / 2.0).unwrap()
        .nsimplify_with_constants(&[&ctx.pi()], 1e-12));                       // 1/2*pi
    println!("{}", x.powi(4).subs_algebraic(&x.powi(2), &y));                  // y^2
    println!("{}", (&x * 2).exp().subs_algebraic(&x.exp(), &y));               // y^2
}
```

## Gröbner bases

`symplex::groebner` computes reduced Gröbner bases (Buchberger with FGLM order conversion) over sparse multivariate polynomials (`symplex::multipoly`); `symplex::polysys::solve_system_ex` builds on it (see [Solving Equations](./solving.md#polynomial-systems)).
