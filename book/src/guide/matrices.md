# Matrices

`Matrix` is a dense matrix of `Ex` entries — exact rationals, radicals, or symbols. Construct one with `matrix![ctx, [1, 2], [3, 4]]`, `Matrix::new(rows)?`, `Matrix::identity(&ctx, n)`, `Matrix::zeros(&ctx, m, n)`, `Matrix::from_fn(m, n, |i, j| …)`, `Matrix::diag`, `Matrix::col_vector`/`row_vector`, `Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]])?`, or `Matrix::try_from(vec_of_rows)?`.

## Basics and ergonomics

Shape-sensitive operations return `Result`; entries are indexed with `m[(i, j)]` (and `IndexMut`); scalars multiply from either side.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let mut m = matrix![ctx, [1, 2], [3, 4]];

    println!("{}", m.det().unwrap());                    // -2
    println!("{}", m.inv().unwrap());                    // [[-2, 1], [3/2, -1/2]]
    println!("{}", m.transpose());
    println!("{}", m.trace().unwrap());                  // 5
    println!("{}", m.rank());                            // 2

    m[(0, 1)] = ctx.int(7);                              // IndexMut
    println!("{} {}", m[(0, 1)], m.get(1, 0));           // 7 3
    println!("{}", 2 * &m);                              // scalar on the left
    println!("{}", m.clone() / 2);
    println!("{}", &m * &m);                             // matrix product (also m.matmul(&m)?)
    println!("{}", &m + &m);
    println!("{}", -m.clone());
    println!("{}", m.hadamard(&m).unwrap());             // element-wise product
    println!("{}", Matrix::block_diag(&[&m, &Matrix::identity(&ctx, 1)]).unwrap());
    println!("{} {}", m.minor(0, 0).unwrap(), m.minor_matrix(0, 0).unwrap());   // 4 [[4]]
    println!("{:?}", m.eval_f64().unwrap());             // [[1.0, 7.0], [3.0, 4.0]]
    println!("{:?}", m.equals(&m));                      // Some(true)
    assert!(Matrix::try_from(vec![vec![ctx.int(1)], vec![ctx.int(2), ctx.int(3)]]).is_err());
}
```

Element-wise helpers: `map`, `map_indexed`, `subs`, `subs_map`, `eval`, `expand`, `simplify`, `diff`, `integrate`, `col`, `row`, `diagonal`, `submatrix`, `set`, `iter`, `to_vec`, `vec` (column-major vectorisation), `hstack`/`vstack`, `kronecker`.

## Selecting sub-matrices and exact conversion

New in 0.3: `extract(&rows, &cols)` (SymPy's `Matrix.extract`; indices may repeat or reorder), `select_rows`, `select_cols`, `delete_row`, `delete_col`; the three-valued structure test `is_integer_matrix` (a companion to the existing `is_zero`); `nnz` (structurally non-zero entries); and lossless conversions to and from the `num` types — `to_rational_rows`, `to_bigint_rows`, `Matrix::from_ratio`, `Matrix::from_bigint`, `Matrix::from_f64_rows` (each `f64` becomes the *exact* dyadic rational it represents) — which is how a `Matrix` is handed to the exact [LP solver](./exact-lp.md) and the [integer normal forms](./integer-lattices.md). (As elsewhere on this page, multi-row matrix output is compacted onto one line in the comments; a single-row matrix really does print as `[[…]]`.)

```rust
use symplex::prelude::*;
use num_bigint::BigInt;
use num_rational::Ratio;

fn main() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    println!("{}", m.extract(&[2, 0], &[0, 2]).unwrap());              // [[7, 9], [1, 3]]
    println!("{}", m.select_rows(&[0, 2]).unwrap());                    // [[1, 2, 3], [7, 8, 9]]
    println!("{}", m.select_cols(&[1]).unwrap());                       // [[2], [5], [8]]
    println!("{}", m.delete_row(1).unwrap().delete_col(1).unwrap());     // [[1, 3], [7, 9]]
    println!("{} {:?} {:?}", m.nnz(), m.is_zero(), m.is_integer_matrix());   // 9 Some(false) Some(true)
    println!("{:?}", m.to_bigint_rows().unwrap()[2]);                    // [7, 8, 9]
    println!("{}", m.extract(&[3], &[0]).unwrap_err());
    // extract: invalid argument: row index 3 out of range for 3 rows

    let f = Matrix::from_f64_rows(&ctx, &[vec![0.5, 0.1]]).unwrap();
    println!("{f}");                          // [[1/2, 3602879701896397/36028797018963968]]
    println!("{:?} {:?}", f.is_integer_matrix(), f.to_bigint_rows());   // Some(false) None
    let q = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
    let r = Matrix::from_ratio(&ctx, &[vec![q(1, 2), q(3, 1)]]).unwrap();
    println!("{r} {:?}", r.to_rational_rows().unwrap()[0]);
    // [[1/2, 3]] [Ratio { numer: 1, denom: 2 }, Ratio { numer: 3, denom: 1 }]
    println!("{}", Matrix::from_bigint(&ctx, &[vec![BigInt::from(1), BigInt::from(-2)]]).unwrap());   // [[1, -2]]

    symplex::syms!(ctx; x, y);
    let s = Matrix::new(vec![vec![x.clone(), y.clone()]]).unwrap();
    println!("{}", s.subs_map(&[(&x, &y), (&y, &x)]));                 // [[y, x]]   (simultaneous)
    println!("{:?} {:?} {}", s.is_zero(), s.is_integer_matrix(), s.nnz());   // None None 2
    let z = Matrix::new(vec![vec![&(&x + 1).powi(2) - &(&x.powi(2) + &x * 2 + 1)]]).unwrap();
    println!("{:?} {}", z.is_zero(), z.nnz());                   // Some(true) 1
}
```

`is_zero` simplifies each entry, so it recognises `(x + 1)² − x² − 2x − 1` as zero; `nnz` is purely structural and counts that entry. `from_f64_rows` is deliberately exact — use `Context::from_f64_nice` entry-wise when you want `0.1` read as `1/10`.

## Eigenvalues, eigenvectors, Jordan form

In 0.2 the eigen family takes **no dummy variable**. `eigenvals` returns eigenvalues with repetition, `eigenvals_with_multiplicity` returns `(value, multiplicity)` pairs, `eigenvects` returns `(value, multiplicity, basis)`, `diagonalize` returns `Diagonalization { p, d }`, `jordan_form` returns `JordanForm { p, j }`, and `is_diagonalizable` is `Option<bool>`. Use `char_poly(&λ)` / `char_poly_coeffs()` when you want the polynomial itself.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let s = matrix![ctx, [2, 1], [1, 2]];
    println!("{:?}", s.eigenvals().unwrap());            // [Ex(3), Ex(1)]
    for (val, mult, vecs) in s.eigenvects().unwrap() {
        println!("λ = {val} ×{mult}: {}", vecs[0].transpose());   // 3: [[1, 1]], 1: [[-1, 1]]
    }
    let Diagonalization { p, d } = s.diagonalize().unwrap();
    assert_eq!(p.matmul(&d).unwrap().matmul(&p.inv().unwrap()).unwrap().equals(&s), Some(true));

    let j = matrix![ctx, [5, 4, 2, 1], [0, 1, -1, -1], [-1, -1, 3, 0], [1, 1, -1, 2]];
    println!("{:?}", j.eigenvals_with_multiplicity().unwrap());   // [(4, 2), (2, 1), (1, 1)]
    println!("{:?}", j.is_diagonalizable());                      // Some(false)
    let jordan = j.jordan_form().unwrap().j;
    println!("{jordan}");                                          // [[4,1,0,0],[0,4,0,0],[0,0,2,0],[0,0,0,1]]
}
```

### `RootOf` eigenvalues

When the characteristic polynomial has an irreducible cubic or quartic factor without a compact radical form, 0.2 returns exact `RootOf(poly, index)` eigenvalues instead of Cardano/Ferrari expressions with nested complex cube roots (which made every downstream step swell). They evaluate numerically and are not counted as unevaluated. An `EXPRESSION_BUDGET` guard aborts computations whose intermediate expressions grow beyond bounds.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let c = matrix![ctx, [0, 1, 0], [0, 0, 1], [1, 1, 0]];      // char poly λ³ − λ − 1
    for ev in c.eigenvals().unwrap() {
        let Complex64 { re, im } = ev.eval_complex64().unwrap();
        println!("{ev} ≈ {re:.6} {im:+.6}i");       // RootOf(λ^3 - λ - 1, k) ≈ …
    }
}
```

## Decompositions

Every factorisation returns a named struct from `symplex::decompositions` (all in the prelude) rather than a tuple, so `qr.q`/`qr.r` or `let Qr { q, r } = …` say which factor is which; each struct documents the identity it satisfies.

| Method | Returns | Preconditions (→ `Err`) |
|--------|---------|-------------------------|
| `lu()` | `Lu { l, u, perm }` with `P·A = L·U` | square |
| `cholesky()` | `L` with `L·Lᵀ = A` | symmetric, positive definite |
| `ldl()` | `Ldl { l, d }` with `A = L·D·Lᵀ` | symmetric |
| `qr()` | `Qr { q, r }` with `A = Q·R`, exact radicals | — |
| `matrix_decomp::gram_schmidt(&vectors, normalize)` | orthogonal (or orthonormal) basis | linearly independent input |
| `rref()` | `(R, pivot_columns)` | — |
| `pinv()` | Moore–Penrose pseudo-inverse (any rank, 0.9) | — |
| `diagonalize()` | `Diagonalization { p, d }` with `A = P·D·P⁻¹` | square, diagonalizable |
| `jordan_form()` | `JordanForm { p, j }` with `A = P·J·P⁻¹` | square |
| `rank_decomposition()` (0.9) | `RankDecomposition { c, f }` with `A = C·F`, `rank A` columns/rows | non-zero |
| `hessenberg()` (0.9) | `Hessenberg { h, p }` with `H = P⁻¹AP` upper Hessenberg, no radicals | square |

```rust
use symplex::prelude::*;
use symplex::matrix_decomp::gram_schmidt;

fn main() {
    let ctx = Context::new();
    let spd = matrix![ctx, [4, 12, -16], [12, 37, -43], [-16, -43, 98]];
    let l = spd.cholesky().unwrap();
    println!("{l}");                                             // [[2,0,0],[6,1,0],[-8,5,3]]
    assert_eq!(l.matmul(&l.transpose()).unwrap().equals(&spd), Some(true));
    let Ldl { l, d } = spd.ldl().unwrap();
    println!("{l} {d}");
    assert!(matrix![ctx, [1, 2], [3, 4]].cholesky().is_err());  // not symmetric

    let m = matrix![ctx, [1, 1, 0], [1, 0, 1], [0, 1, 1]];
    let Qr { q, r } = m.qr().unwrap();
    println!("{q}\n{r}");                                        // sqrt(1/2), sqrt(2/3), …
    assert_eq!(q.is_orthogonal(), Some(true));
    assert_eq!(q.matmul(&r).unwrap().simplify().equals(&m), Some(true));

    let v1 = Matrix::col_vector(vec![ctx.int(1), ctx.int(1), ctx.int(0)]);
    let v2 = Matrix::col_vector(vec![ctx.int(1), ctx.int(0), ctx.int(1)]);
    for b in gram_schmidt(&[v1, v2], true).unwrap() {
        println!("{}", b.transpose());
    }
    let Lu { l: lu_l, u: lu_u, perm } = matrix![ctx, [2, 1], [4, 3]].lu().unwrap();
    println!("{lu_l} {lu_u} {perm:?}");
}
```

## Matrix functions

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; t, n);
    let rot = matrix![ctx, [0, -1], [1, 0]];
    println!("{}", rot.matrix_exp().unwrap());              // [[cos(1), -sin(1)], [sin(1), cos(1)]]
    println!("{}", rot.matrix_exp_t(&t).unwrap());          // [[cos(t), -sin(t)], [sin(t), cos(t)]]
    println!("{}", matrix![ctx, [2, 1], [0, 2]].matrix_exp_t(&t).unwrap());   // [[e^(2t), t e^(2t)], [0, e^(2t)]]
    let s = matrix![ctx, [2, 1], [1, 2]];
    println!("{}", s.matrix_pow_symbolic(&n).unwrap());     // [[3^n/2 + 1/2, 3^n/2 - 1/2], …]
    println!("{}", s.matrix_sqrt().unwrap());               // [[√3/2 + 1/2, √3/2 - 1/2], …]
    println!("{}", s.powi(3).unwrap());
    println!("{}", s.exp_series(6).unwrap());               // truncated Taylor series
}
```

## Structure tests and norms

All structure tests return `Option<bool>` (`None` when a symbolic entry cannot be decided): `is_symmetric`, `is_skew_symmetric`, `is_hermitian`, `is_orthogonal`, `is_unitary`, `is_upper_triangular`, `is_lower_triangular`, `is_diagonal`, `is_identity`, `is_zero`, `is_nilpotent`, `is_positive_definite`, `is_positive_semidefinite`, `is_diagonalizable`. Norms: `norm_1`, `norm_inf`, `norm_frobenius` (= `norm`), `norm_p(&p)` for vectors.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; theta);
    let r = Matrix::new(vec![vec![theta.cos(), -theta.sin()], vec![theta.sin(), theta.cos()]]).unwrap();
    println!("{:?} {}", r.is_orthogonal(), r.det().unwrap().simplify());   // Some(true) 1
    println!("{:?}", r.eigenvals().unwrap());   // [sin(theta)*I + cos(theta), -sin(theta)*I + cos(theta)]
    let i = ctx.i_unit();
    let h = Matrix::new(vec![vec![ctx.int(2), &ctx.int(1) + &i], vec![&ctx.int(1) - &i, ctx.int(3)]]).unwrap();
    println!("{:?} {}", h.is_hermitian(), h.adjoint());     // Some(true) …
    let m = matrix![ctx, [1, -2], [3, 4]];
    println!("{} {} {}", m.norm_1(), m.norm_inf(), m.norm_frobenius());    // 6 7 sqrt(30)
}
```

## Subspaces and least squares

`rank`, `nullspace`, `columnspace`, `rowspace`, `left_nullspace` return bases as column vectors; `solve(&b)` solves a square system; `solve_least_squares(&b)` solves the normal equations for over-determined systems; `linsolve_matrix` (see [Solving](./solving.md#linear-systems)) handles singular and inconsistent systems with free variables.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 1], [1, 2], [1, 3]];
    let b = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(2)]);
    println!("{}", a.solve_least_squares(&b).unwrap().transpose());     // [[2/3, 1/2]]
    let r1 = matrix![ctx, [1, 2], [2, 4]];
    println!("{} {} {}", r1.rank(), r1.rowspace()[0], r1.left_nullspace()[0].transpose());
    // 1 [[1, 2]] [[-2, 1]]
}
```

## Exact matrices over ℚ and ℤ: `QMatrix` and `ZMatrix`

`Matrix` stores expressions, and every entry operation goes through the expression arena (canonicalisation, hash-consing, a write lock). That is what you want for symbolic matrices and pure overhead for numeric ones. 0.3.5 adds `QMatrix` (entries `Ratio<BigInt>`) and `ZMatrix` (entries `BigInt`) — both in the prelude and in `symplex::matrix` — as plain row-major `Vec<T>` matrices with the same shape rules, indexing, `Display`/`Debug` layout and `Result`-returning arithmetic as `Matrix`.

All rational eliminations are **fraction-free**: the rows are scaled to integers and reduced with Bareiss's Gauss–Jordan variant, whose intermediate entries are minors of the input, so every division is exact and no gcd runs in the inner loop. On a 60×72 integer matrix that is about 40× faster than Gauss–Jordan over `Ratio<BigInt>` and two orders of magnitude faster than the same elimination over expressions.

```rust
use symplex::prelude::*;
use symplex::linprog::q;            // exact rational literal: q(1, 2) = 1/2

fn main() {
    let a = QMatrix::from_i64(&[&[2, 1], &[1, 3]]).unwrap();
    let b = QMatrix::new(vec![vec![q(1, 2)], vec![q(1, 3)]]).unwrap();
    println!("{}", a.solve(&b).unwrap().transpose());       // [[7/30, 1/30]]
    println!("{}", a.det().unwrap());                        // 5
    println!("{}", a.inv().unwrap());                        // [[3/5, -1/5], [-1/5, 2/5]]

    // The 4×4 Hilbert matrix: det = 1/6048000, integral inverse.
    let h = QMatrix::from_fn(4, 4, |i, j| q(1, (i + j + 1) as i64));
    println!("{} {}", h.det().unwrap(), h.inv().unwrap().is_integer());   // 1/6048000 true

    let s = QMatrix::from_i64(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]).unwrap();
    let (r, pivots) = s.rref();
    println!("{r:?} {pivots:?}");
    // QMatrix(3×3, [[1, 0, -1], [0, 1, 2], [0, 0, 0]]) [0, 1]
    println!("{}", s.nullspace()[0].transpose());            // [[1, -2, 1]]

    // ℤ: Bareiss determinant, Hermite and Smith forms, integer kernels.
    let z = ZMatrix::from_i64(&[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]).unwrap();
    let HermiteNormalForm { h: hnf, u } = z.hermite_normal_form_with_transform();
    println!("{hnf:?} det U = {}", u.det().unwrap());
    // ZMatrix(3×3, [[2, 4, 4], [0, 6, 0], [0, 0, 12]]) det U = -1
    println!("{:?}", z.smith_normal_form().diagonal());     // [2, 6, 12]
    println!("{:?}", ZMatrix::from_i64(&[&[2, 1, 1]]).unwrap().integer_nullspace());
    // [ZMatrix(3×1, [[1], [0], [-2]]), ZMatrix(3×1, [[0], [1], [-1]])]
}
```

`QMatrix` has `rref`, `rank`, `nullspace`, `columnspace`, `rowspace`, `det`, `inv`, `solve` (square, several right-hand sides), `clear_denominators` (`(Z, s)` with `Z = s·A` integral) and `to_zmatrix`; `ZMatrix` has `det`, `rank`, `content`, `hermite_normal_form[_with_transform]`, `column_hermite_normal_form`, `smith_normal_form[_with_transforms]`, `integer_nullspace`, `is_unimodular`, `lattice_determinant` and `to_qmatrix`. Both have `transpose`, `submatrix`, `hstack`/`vstack`, `map`, `scale`, `trace`, the operators `+ − *`, and `is_zero`/`is_identity`.

Conversions are explicit and lossless. `ZMatrix::try_from(&m)` / `QMatrix::try_from(&m)` accept a `Matrix` whose entries are all integer / rational literals (constant arithmetic such as `1/3 + 1/6` is folded first; a symbol is an `InvalidArgument` error, not an approximation), and `to_matrix(&ctx)` goes back:

```rust
use symplex::prelude::*;
use symplex::linprog::q;

fn main() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    let z = ZMatrix::try_from(&m).unwrap();
    println!("{}", z.to_qmatrix().inv().unwrap().to_matrix(&ctx));   // [[-2, 1], [3/2, -1/2]]
    let (zc, s) = QMatrix::new(vec![vec![q(1, 2), q(1, 3)], vec![q(2, 1), q(-1, 6)]])
        .unwrap()
        .clear_denominators();
    println!("{zc:?} {s}");        // ZMatrix(2×2, [[3, 2], [12, -1]]) 6
    let half = Matrix::new(vec![vec![ctx.rational(1, 2)]]).unwrap();
    println!("{}", ZMatrix::try_from(&half).unwrap_err());
    // ZMatrix::try_from: invalid argument: every entry must be an integer literal (fractions and symbolic entries are not allowed)
}
```

You rarely need to convert by hand: `Matrix::{rref, rank, nullspace, columnspace, rowspace, left_nullspace, det, inv, solve, solve_least_squares, pinv}` (and, since 0.21, `char_poly_coeffs`, `matmul`, `trace`, `lu`), `linsolve` / `linsolve_matrix` and every function in `symplex::normalforms` detect all-rational input and route through `QMatrix`/`ZMatrix` themselves, returning the same `Matrix` results as before (the RREF is unique, so pivots and entries are identical). A single symbolic entry sends the whole matrix down the expression path. Use the exact types directly when the data is numeric from the start — LP formulations, coefficient matrices from `Poly::coefficient_matrix`, lattices — to skip the arena round trip.

## Integer normal forms

For a matrix of integer literals, 0.3 adds `hermite_normal_form` (row style, `H = U·A`), `smith_normal_form` (`S = U·A·V`, invariant factors) and `integer_nullspace` (a ℤ-basis of the integer kernel) as methods, with the transform-returning and column-convention variants in `symplex::normalforms`. Non-integer entries are an `InvalidArgument` error, not a rounding.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 4, 4], [-6, 6, 12], [10, -4, -16]];
    println!("{}", a.hermite_normal_form().unwrap());     // [[2, 4, 4], [0, 6, 0], [0, 0, 12]]
    println!("{}", a.smith_normal_form().unwrap());       // [[2, 0, 0], [0, 6, 0], [0, 0, 12]]
    let ker: Vec<String> = matrix![ctx, [2, 1, 1]]
        .integer_nullspace()
        .unwrap()
        .iter()
        .map(|k| k.transpose().to_string())
        .collect();
    println!("{ker:?}");                                  // ["[[1, 0, -2]]", "[[0, 1, -1]]"]
}
```

Conventions, the SymPy-compatible column HNF, unimodularity tests and lattice determinants are covered in [Integer Lattices and Normal Forms](./integer-lattices.md).

## Calculus helpers

`matrix::jacobian(&funcs, &vars)`, `matrix_decomp::hessian(&f, &vars)`, `matrix_decomp::wronskian(&funcs, &x)`, and `Matrix::{diff, integrate}` element-wise.

```rust
use symplex::prelude::*;
use symplex::matrix_decomp::{hessian, wronskian};

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let f = &x.powi(3) * &y + &x * &y.powi(2);
    println!("{}", hessian(&f, &[&x, &y]));                       // [[6xy, 3x²+2y], [3x²+2y, 2x]]
    println!("{}", wronskian(&[&x.sin(), &x.cos()], &x).simplify());   // -1
    println!("{}", symplex::matrix::jacobian(&[&f], &[&x, &y]));
}
```

## Related types

- **Quaternions** (`symplex::quaternion::Quaternion`, in the prelude): arithmetic operators, `conjugate`, `norm`, `inverse`, `normalize`, `to_rotation_matrix`/`from_rotation_matrix`, `from_axis_angle`/`to_axis_angle`, `from_euler`/`to_euler`, `rotate_vector`, `slerp`, `exp`/`ln`/`pow`.
- **Vector calculus** (`symplex::vector`): `gradient`, `divergence`, `curl`, `laplacian`, and their `_in(&CoordinateSystem)` variants for cylindrical and spherical coordinates; `directional_derivative`, `line_integral_scalar`, `line_integral_vector`, `scalar_potential`; `is_conservative`/`is_irrotational`/`is_solenoidal` return `Option<bool>`.
- **Control** (`symplex::control`): `StateSpace` (poles, stability, controllability, observability, ZOH discretisation, `to_transfer_function`) and `TransferFunction` (series/parallel/feedback algebra, Routh–Hurwitz, `to_state_space`).

See `cargo run --example matrix_decompositions`, `matrix_algebra`, `exact_matrices`, `control_system` and `integer_lattices`.

## More decompositions and utilities (0.9)

0.9 fills in the remaining everyday SymPy matrix methods. As elsewhere, rational input is routed through `QMatrix`/`ZMatrix` and is exact; symbolic input follows the same *structural* pivoting rules as `rref` and `lu` (a symbolic pivot whose value cannot be decided is assumed non-zero, so the result holds generically).

### Singular values and condition number

`singular_values()` returns the `ncols` square roots of the eigenvalues of `AᵀA` (computed from the smaller Gram matrix `AᵀA` or `AAᵀ` and padded with zeros), sorted in descending order whenever the values can be compared numerically. The result is exact whenever `eigenvals` can solve the Gram matrix's characteristic polynomial — always for rational matrices, as radicals when the irreducible factors have degree ≤ 2 (or a compact radical form) and as `RootOf` values otherwise. `condition_number()` is `σ_max/σ_min` in the 2-norm; a singular matrix is a `ComputationFailed` error whose reason mentions "singular" (SymPy returns `zoo`).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    println!("{:?}", m.singular_values().unwrap());
    // [Ex(sqrt(sqrt(221) + 15)), Ex(sqrt(-sqrt(221) + 15))]   SymPy: [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221))]
    println!("{}", m.condition_number().unwrap().eval_f64().unwrap());   // 14.93303437365925
    println!("{:?}", matrix![ctx, [3, 0, 0], [0, 4, 0]].singular_values().unwrap());   // [Ex(4), Ex(3), Ex(0)]
    println!("{}", matrix![ctx, [2, 0], [0, 3]].condition_number().unwrap());        // 3/2
    assert!(matrix![ctx, [1, 2], [2, 4]].condition_number().is_err());
}
```

### Pseudo-inverse for any rank, rank factorisation

`pinv()` no longer requires full column rank. It uses the full-rank factorisation `A = C·F` returned by `rank_decomposition()` — `C` holds the pivot columns of `A`, `F` the non-zero rows of `rref(A)` — and `A⁺ = Fᵀ(FFᵀ)⁻¹(CᵀC)⁻¹Cᵀ`; full-column-rank matrices still take the classical `(AᵀA)⁻¹Aᵀ`, and the zero matrix maps to the zero matrix of the transposed shape. `rank_decomposition` itself is an error only for the zero matrix (rank 0 has no non-empty factors).

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2], [2, 4]];                    // rank 1
    println!("{}", a.pinv().unwrap());                       // [[1/25, 2/25], [2/25, 4/25]]
    let RankDecomposition { c, f } = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]].rank_decomposition().unwrap();
    println!("{c} {f}");                                     // [[1, 2], [4, 5], [7, 8]]  [[1, 0, -1], [0, 1, 2]]
    symplex::syms!(ctx; x);
    let s = Matrix::new(vec![vec![x.clone(), x.clone()], vec![x.clone(), x.clone()]]).unwrap();
    println!("{}", s.pinv().unwrap());                       // [[1/(4x), 1/(4x)], [1/(4x), 1/(4x)]]
}
```

### Hessenberg form

`hessenberg()` returns `Hessenberg { h, p }` with `H = P⁻¹AP` upper Hessenberg (`h_ij = 0` for `i > j + 1`), computed by Gaussian similarity transforms — row eliminations paired with the compensating column operations, with a symmetric row/column swap when the sub-diagonal entry is zero. Unlike SymPy's Householder-based `upper_hessenberg_decomposition` the result stays in the field of the entries: exact rationals for rational input, no radicals.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 10]];
    let Hessenberg { h, p } = a.hessenberg().unwrap();
    println!("{h}");             // [[1, 29/4, 3], [4, 31/2, 6], [0, -13/8, -1/2]]
    println!("{p}");             // [[1, 0, 0], [0, 1, 0], [0, 7/4, 1]]
    assert_eq!(&a * &p, &p * &h);
    assert_eq!(h.char_poly(&ctx.symbol("λ")).unwrap(), a.char_poly(&ctx.symbol("λ")).unwrap());
}
```

### Constructors and small utilities

| Method | SymPy | Notes |
|--------|-------|-------|
| `Matrix::companion(&[c₀, …, c_{n−1}])` | `Matrix.companion(Poly)` | monic `xⁿ + c_{n−1}xⁿ⁻¹ + … + c₀`, coefficients **ascending**; ones on the sub-diagonal, `−cᵢ` in the last column |
| `Matrix::jordan_block(&λ, size)` | `Matrix.jordan_block(size, λ)` | `Err` for `size == 0` |
| `permanent()` | `Matrix.per()` | Ryser on `QMatrix` for rational input, subset DP for symbolic; **exponential**, `n ≤ 20` |
| `row_insert(pos, &rows)`, `col_insert(pos, &cols)` | `row_insert`, `col_insert` | `pos == nrows/ncols` appends; `Err` on bad position or shape |
| `permute_rows(&perm)`, `permute_cols(&perm)` | `permute_rows`, `permute_cols` | result row `i` is input row `perm[i]`; `perm` must be a genuine permutation |
| `row_del(i)`, `col_del(j)` | `row_del`, `col_del` | SymPy names for `delete_row` / `delete_col` |
| `Matrix::casoratian(&seqs, &n)` | `casoratian(seqs, n, zero=False)` | `det[fⱼ(n+i)]`; SymPy's default `zero=True` is `.subs(&n, &ctx.int(0))` |
| `inv_mod(m)` | `Matrix.inv_mod(m)` | integer matrices, `adj(A)·det(A)⁻¹ mod m`; `Err` unless `gcd(det A, m) = 1` |
| `matrix_log()` | `Matrix.log()` | via the Jordan form: `P·log(J)·P⁻¹`, `log J_k(λ) = ln λ·I + Σ (−1)^{d+1}N^d/(dλ^d)`; `Err` for singular matrices |

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, n);
    let c = Matrix::companion(&[ctx.int(4), ctx.int(3), ctx.int(2)]).unwrap();   // x³ + 2x² + 3x + 4
    println!("{c}");                                          // [[0, 0, -4], [1, 0, -3], [0, 1, -2]]
    println!("{}", -c.char_poly(&x).unwrap());               // x^3 + 2*x^2 + 3*x + 4  (char_poly is det(C − xI))
    println!("{}", Matrix::jordan_block(&ctx.int(2), 3).unwrap());   // [[2, 1, 0], [0, 2, 1], [0, 0, 2]]

    let m = matrix![ctx, [1, 2], [3, 4]];
    println!("{}", m.permanent().unwrap());                                   // 10
    println!("{}", matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]].permanent().unwrap());   // 450
    println!("{}", m.row_insert(1, &matrix![ctx, [5, 6]]).unwrap());         // [[1, 2], [5, 6], [3, 4]]
    println!("{}", m.col_insert(1, &matrix![ctx, [5], [6]]).unwrap());       // [[1, 5, 2], [3, 6, 4]]
    println!("{}", matrix![ctx, [1], [2], [3]].permute_rows(&[2, 0, 1]).unwrap().transpose());   // [[3, 1, 2]]
    println!("{}", m.inv_mod(5).unwrap());                                    // [[3, 1], [4, 2]]

    let w = Matrix::casoratian(&[ctx.int(2).pow(&n), ctx.int(3).pow(&n)], &n).unwrap();
    println!("{}", w.simplify());                                             // 6^n
    println!("{}", matrix![ctx, [2, 0], [0, 3]].matrix_log().unwrap());      // [[ln(2), 0], [0, ln(3)]]
    println!("{}", matrix![ctx, [1, 1], [0, 1]].matrix_log().unwrap());      // [[0, 1], [0, 0]]
}
```

### LLL lattice reduction

`ZMatrix::lll(delta)` (`delta` a `num_rational::Rational64`; `lll_default()` uses `δ = 3/4`, `lll_with_transform` also returns the unimodular `T` with `T·A = R` as `LllReduction { reduced, transform }`) reduces the lattice basis formed by the **rows**, with exact rational Gram–Schmidt data. The output satisfies the size condition `|μ_ij| ≤ 1/2` and the Lovász condition `‖b*_k‖² ≥ (δ − μ²_{k,k−1})‖b*_{k−1}‖²` exactly, spans the same lattice (same Hermite normal form), and — because the reduction order and rounding follow SymPy's `DomainMatrix.lll` — coincides with SymPy's output. `δ` must lie in `(1/4, 1)` and the rows must be linearly independent (a lattice basis); anything else is an `InvalidArgument` error. The same is available on `Matrix` for integer literals (`Matrix::lll`, `Matrix::lll_default`, `normalforms::lll`, `normalforms::lll_with_transform`).

```rust
use symplex::prelude::*;
use symplex::num_rational::Ratio;

fn main() {
    let ctx = Context::new();
    let b = ZMatrix::from_i64(&[&[1, 1, 1], &[-1, 0, 2], &[3, 5, 6]]).unwrap();
    let r = b.lll_default().unwrap();
    println!("{r:?}");                          // ZMatrix(3×3, [[0, 1, 0], [1, 0, 1], [-1, 0, 2]])  (= SymPy's .lll())
    assert_eq!(r.hermite_normal_form(), b.hermite_normal_form());   // same lattice
    let LllReduction { reduced: r2, transform: t } = b.lll_with_transform(Ratio::new(3, 4)).unwrap();
    assert_eq!(&t * &b, r2);
    assert!(t.is_unimodular());
    println!("{}", matrix![ctx, [1, 0, 0, 1345], [0, 1, 0, 35], [0, 0, 1, 154]].lll_default().unwrap());
    // [[0, 9, -2, 7], [1, 1, -9, -6], [1, -3, -8, 8]]
    assert!(ZMatrix::from_i64(&[&[1, 2], &[2, 4]]).unwrap().lll_default().is_err());   // dependent rows
}
```
