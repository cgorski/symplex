# Migrating from 0.1 to 0.2

symplex 0.2 is a breaking release. Most changes are mechanical (a `Result` where there was an `Option`, a dropped dummy argument); a few change *results* because 0.1 was wrong. The full list is in the [CHANGELOG](https://github.com/cgorski/symplex/blob/main/CHANGELOG.md#breaking); this page shows the code.

## `compile` returns `Result`

```rust,ignore
// 0.1
let f = expr.compile(&["x"]).expect("unsupported node");   // Option<Box<dyn Fn>>
// 0.2
let f = expr.compile(&["x"])?;                              // Result<CompiledFn, SymplexError>
f(&[2.0]);                 // still callable
f.arity();                 // new
f.try_call(&[2.0])?;       // new: arity-checked
```

`CompiledFn` is `Clone + Send + Sync`. The error tells you *why*: `FreeSymbol { name }` or `NotImplemented(node)`. Coverage grew to every numerically evaluable node (special functions, Bessel, orthogonal polynomials, piecewise), so expressions that were `None` in 0.1 now compile.

## `definite_integral` → `integrate_definite`

```rust,ignore
// 0.1  — computed F(b) − F(a) blindly: ∫₋₁¹ dx/x² gave −2
let v = expr.definite_integral(&x, &a, &b);
// 0.2
let v = expr.integrate_definite(&x, &a, &b);          // Ex; Integral node if undecided
let v = expr.try_integrate_definite(&x, &a, &b)?;     // Err(Divergent) / Err(ComputationFailed)
```

If you relied on `F(b) − F(a)` for a proper integral, results are unchanged. For integrals across a pole you now get `Err(Divergent)` (or an unevaluated node), which is the correct answer.

## `solve` semantics

```rust,ignore
// 0.1: identities and contradictions both gave Ok(vec![]) (or a guess)
// 0.2:
match expr.solve(&x) {
    Ok(roots) => …,
    Err(SymplexError::InfiniteSolutions { .. }) => …,   // x − x = 0
    Err(SymplexError::NoSolution { .. }) => …,          // sin x = 2, eˣ = −1, |x| = −1
    Err(e) => …,
}
```

Roots are now `eval`'d: `asin(1/2)` comes back as `π/6`. If you matched on strings, update the expected text. `solve_or_empty` still returns `Vec<Ex>` and swallows every error.

## Matrices

```rust,ignore
// 0.1                                     // 0.2
m.eigenvals(&lam)?                         m.eigenvals()?
m.eigenvects(&lam)?                        m.eigenvects()?
m.diagonalize(&lam)?                       m.diagonalize()?
m.jordan_form(&lam)?                       m.jordan_form()?
m.matrix_exp(&t)?                          m.matrix_exp_t(&t)?      // or matrix_exp() for e^A
m.is_diagonalizable(&lam) -> bool          m.is_diagonalizable() -> Option<bool>
m.is_symmetric() -> bool                   m.is_symmetric() -> Option<bool>
m.cholesky() -> Option<Matrix>             m.cholesky() -> Result<Matrix>
m.lu() -> (L, U, perm)                     m.lu() -> Result<(L, U, perm)>
m.minor(i, j) -> Matrix                    m.minor_matrix(i, j)?    // sub-matrix
                                           m.minor(i, j)?           // Result<Ex>: its determinant
Matrix::from_i64(&ctx, rows) -> Matrix     Matrix::from_i64(&ctx, rows)?
m.add_elementwise(&n) / sub_elementwise    m.add(&n)? / m.sub(&n)?   (or &m + &n)
Matrix::try_identity / try_zeros           Matrix::identity / zeros
```

`char_poly(&lam)` still takes the variable you want in the output. Structure tests on symbolic matrices return `None` when undecidable — replace `if m.is_symmetric()` with `if m.is_symmetric() == Some(true)`.

## Linear systems

```rust,ignore
// 0.1
let values: Vec<Ex> = ctx.solve_system(&eqs, &vars);
// 0.2
match ctx.solve_system(&eqs, &vars)? {
    LinearSolution::Unique(pairs) => …,
    LinearSolution::Parametric { solution, free } => …,
    LinearSolution::Inconsistent => …,
}
// or: let sol = linsolve(&eqs, &vars)?; sol.get(&x)
```

`polysys::solve_system_ex` now returns algebraic solutions (radicals) where 0.1 returned only rational ones, and `Err(InfiniteSolutions)` for positive-dimensional systems.

## Complex parts

```rust,ignore
let z = ctx.symbol("z");
z.re()          // 0.1: z        (assumed real — wrong)
                // 0.2: re(z)    (unevaluated until z is known real)
z.conjugate()   // 0.1: z        // 0.2: conjugate(z)
```

Declare `ctx.symbol_with("z", &[Assumption::Real])` to recover the 0.1 behaviour where it was intended.

## `has_unevaluated` and `RootOf`

`RootOf` / `RootSum` no longer count as unevaluated, so `try_integrate`, `try_solve_ode`, … succeed on results containing them. If you used `has_unevaluated()` to detect degree-≥5 roots, check for the node instead: `expr_type()` still reports `ExprType::Unevaluated` for a `RootOf` at the root of an expression, so `root.expr_type() == ExprType::Unevaluated` keeps working for the solutions returned by `solve`.

## Other signature changes

| 0.1 | 0.2 |
|-----|-----|
| `StateSpace::poles(&s)` | `StateSpace::poles()` |
| `vector::is_conservative(…) -> bool` | `-> Option<bool>` (also `is_irrotational`, `is_solenoidal`) |
| `SetEx::contains(&e)` structural | set membership, `Option<bool>` |
| `expr.textplot(…) -> String` | `-> Result<String>` (all plotting methods) |
| `Ex::differentiate_finite(...)` | `differentiate_finite(&var, &points, order)` |
| `FormalPowerSeries` over `Ratio` | over `Ex` (`coefficient(k) -> Ex`, `coefficient_rational(k) -> Option<Ratio>`) |
| `finite_diff::*` over `Ratio` | over `Ex` |
| `iter.sum::<Ex>()` on empty → `0` | panics; use `ctx.sum(iter)` or `Option<Ex>` |
| `Debug for Ex` prints ids | prints `Ex(x^2 + 1)` |
| `expand()` splits `(x·y)^a` | no longer for unknown-sign symbols; `expand_power_base(true)` |
| `Assumption` enum | new variants `ExtendedReal`, `NotPositive`, `NotZero`, … — add a `_ =>` arm |
| `OdeType` enum | new variants — add a `_ =>` arm |
| `d/dx digamma(x)` → formal derivative | `polygamma(1, x)` |
| `Digamma(5)` stays | folds to `-EulerGamma + 25/12` |

## Results that changed because 0.1 was wrong

- `fourier_series` of `|x|`, `sign(x)` and piecewise inputs (coefficients are now exact definite integrals).
- One-sided limits: `limit` returns a `Limit` node when the two one-sided limits differ, instead of one of them.
- Several Gruntz limits of `exp`/`ln` towers.
- `matrix_exp` with numeric complex eigenvalues (a `sin(−1)` parity error).
- Factoring is no longer truncated at small degrees: `factor` may now split polynomials that 0.1 left whole.
- Shifted alternating half-integer p-series had a sign error.

## New things worth adopting

- `Context::from_f64` (exact dyadic) / `from_f64_approx(v, max_denominator)` for ingesting floats — `symplex-build` and `symplex-wasm` now use these for DH parameters (`0.3` → `3/10`).
- `solve_general` for periodic equations; `linsolve` for linear systems; `solve_ode_ivp` for initial-value problems.
- `to_c_fn` for C targets; `compile_many` for gradients.
- `simplify_traced` / `rewrite_traced` when a simplification surprises you.
