# API Patterns

Every public operation in symplex follows one of a small number of patterns. Once you know which pattern a method uses, you know its return type and what a "failure" looks like.

## Pattern 1 — Always returns `Ex`

Operations for which "unchanged" or "unevaluated" is a legitimate answer never fail:

```rust,ignore
expr.simplify()          expr.expand()           expr.eval()
expr.factor(&x)          expr.subs(&x, &v)       expr.rewrite(&rules)
expr.diff(&x)            // Derivative(f, x) if it cannot differentiate
expr.integrate(&x)       // Integral(f, x) if no closed form
expr.integrate_definite(&x, &a, &b)   // Integral node if undecided
expr.limit(&x, &a)       // Limit(f, x, a)
expr.summation(&k, &a, &b)            // Sum node
expr.laplace(&t, &s)     // LaplaceTransform node
expr.solve_ode(&y, &x)   // DSolve node
```

Check with `has_unevaluated()`. Note that `RootOf` and `RootSum` are exact algebraic answers and are **not** counted.

## Pattern 2 — `try_` twin returns `Result<Ex>`

Every Pattern-1 method that can produce an unevaluated form has a `try_` twin that returns `Err` instead. The twin calls the base method and checks `has_unevaluated()`, so there is no behavioural drift between the two.

```rust,ignore
let anti = expr.try_integrate(&x)?;                       // Err(ComputationFailed) if unevaluated
let val  = expr.try_integrate_definite(&x, &a, &b)?;      // Err(Divergent) if proven divergent
let lim  = expr.try_limit_right(&x, &a)?;
let sum  = expr.try_summation(&k, &lo, &hi)?;
```

Available twins: `try_diff`, `try_integrate`, `try_integrate_definite`, `try_limit`, `try_limit_left`, `try_limit_right`, `try_limit_dir`, `try_series`, `try_maclaurin`, `try_series_at_infinity`, `try_summation`, `try_product_over`, `try_laplace`, `try_inverse_laplace`, `try_residue`, `try_gosper_sum`, `try_solve_ode`, `try_solve_gt`/`ge`/`lt`/`le`.

## Pattern 3 — Numeric boundary → `Result`

Crossing from symbols to numbers can fail (free symbols, unsupported node, precision exhausted, non-convergence):

```rust,ignore
expr.eval_f64()                        // Result<f64>
expr.eval_complex64()                  // Result<Complex64>  (num_complex; in the prelude)
expr.eval_decimal(50)                  // Result<String>
expr.compile(&["x"])                   // Result<CompiledFn>
Ex::compile_many(&[&a, &b], &["x"])    // Result<CompiledFnVec>
expr.to_rust_fn("f", &["x"])           // Result<String>
expr.to_c_fn("f", &["x"])              // Result<String>
expr.integrate_numeric(&x, &a, &b)     // Result<f64>
expr.nroots(&x, 12)                    // Result<Vec<Complex64>>
expr.textplot(&x, a, b)                // Result<String>  (all plotting methods)
```

## Pattern 4 — Queries → `Option`

Three-valued questions return `Option<bool>` (yes / no / cannot decide) and structural queries return `Option<T>`:

```rust,ignore
expr.is_positive()  expr.is_real()  expr.is_integer()  expr.equals(&other)
expr.is_convergent(&k)  expr.is_absolutely_convergent(&k)  expr.is_real_valued()
set.contains(&e)  set.is_subset(&t)  set.is_disjoint(&t)  set.is_empty()  set.is_open()
matrix.is_symmetric()  matrix.is_orthogonal()  matrix.is_positive_definite()  matrix.is_diagonalizable()
bool_ex.is_tautology()  bool_ex.satisfiable()
vector::is_conservative(&f, &vars)

expr.degree(&x)  expr.coeff(&x, 2)  expr.resultant(&g, &x)  expr.discriminant(&x)
expr.hypergeometric_ratio(&k)  expr.as_i64()  expr.as_rational()  set.inf()  set.measure()
```

`None` is a real answer — do not `unwrap()` it. A symbolic entry usually means the question cannot be decided without assumptions.

## Pattern 5 — Structural preconditions → `Result`

Operations whose input must have a particular shape:

```rust,ignore
matrix.det()            matrix.inv()          matrix.matmul(&other)
matrix.cholesky()       matrix.lu()           matrix.minor(i, j)
matrix.eigenvals()      matrix.jordan_form()  matrix.qr()
Matrix::new(rows)       Matrix::from_i64(&ctx, rows)
Rule::try_new(...)      bool_ex.truth_table(&atoms)
```

## Pattern 6 — Mathematical outcomes as `Err` or enum variants

Solvers distinguish "no method" from "the answer is: none" or "the answer is: all":

| Call | Outcome | Representation |
|------|---------|----------------|
| `solve` | identity | `Err(SymplexError::InfiniteSolutions { .. })` |
| `solve` | contradiction / range violation | `Err(SymplexError::NoSolution { .. })` |
| `solve_system_ex` | positive-dimensional | `Err(InfiniteSolutions)` |
| `linsolve` | contradictory system | `Ok(LinearSolution::Inconsistent)` |
| `linsolve` | under-determined | `Ok(LinearSolution::Parametric { .. })` |
| `try_integrate_definite` | divergent | `Err(SymplexError::Divergent { .. })` |
| `laplace_final_value` | unstable pole | `Err(Divergent)` |
| `fourier_transform`, `mellin_transform`, `z_transform` | not in table / missing sign assumption | `Err(ComputationFailed)` (no unevaluated node exists for these) |

## Ownership and references

`Ex` is `Clone` (cheap: an `Arc` bump and a `u32`) but not `Copy`. Operators are implemented on references and values (`&x + &y`, `&x * 2`, `x.clone() / 3`, `2 * &x`, `x += 1`), and scalars of type `i32`, `i64`, `u32`, `u64`, `i128`, `f64`, `BigInt`, `Ratio<BigInt>` are accepted through the `Scalar`/`ToEx` traits. Methods take `&Ex` arguments. Collections use `Context::sum(iter)` / `Context::product(iter)` or `Option<Ex>` — `iter.sum::<Ex>()` panics on an empty iterator because there is no context to build `0` in.

## Contexts

Everything belongs to a `Context`. Mixing expressions from different contexts panics with a clear message (the only panic in the symbolic layer, treated as a logic error like indexing out of bounds). `Context` is `Clone`; clones share the arena. `Context::compact(&roots)` garbage-collects into a fresh context.

## Naming conventions

| Suffix / prefix | Meaning | Example |
|-----------------|---------|---------|
| `try_` | `Result` twin of a Pattern-1 method | `try_integrate` |
| `_with` | same operation with an options struct | `simplify_with(&SimplifyOpts)`, `rewrite_with(&rules, &RewriteOpts)`, `integrate_numeric_with(…, &QuadOpts)` |
| `_traced` | also returns `Vec<Step>` | `simplify_traced`, `rewrite_traced` |
| `_or_empty` | swallow the error into an empty `Vec` | `solve_or_empty` |
| `_general` | complete solution family | `solve_general` |
| `_ivp` | with initial conditions | `solve_ode_ivp` |
| `_all` | all variables (multivariate) | `factor_all`, `sqrt_mod_all` |
| `is_*` | three-valued query | `is_positive`, `is_symmetric` |
| `as_*` | cheap structural view | `as_rational`, `as_numer_denom`, `as_intervals` |
| `from_*` | constructor on `Context`/types | `from_f64`, `from_ratio`, `from_coefficients` |
