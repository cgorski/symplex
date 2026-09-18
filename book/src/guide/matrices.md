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

In 0.2 the eigen family takes **no dummy variable**. `eigenvals` returns eigenvalues with repetition, `eigenvals_with_multiplicity` returns `(value, multiplicity)` pairs, `eigenvects` returns `(value, multiplicity, basis)`, `diagonalize` returns `(P, D)`, `jordan_form` returns `(P, J)`, and `is_diagonalizable` is `Option<bool>`. Use `char_poly(&λ)` / `char_poly_coeffs()` when you want the polynomial itself.

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let s = matrix![ctx, [2, 1], [1, 2]];
    println!("{:?}", s.eigenvals().unwrap());            // [Ex(3), Ex(1)]
    for (val, mult, vecs) in s.eigenvects().unwrap() {
        println!("λ = {val} ×{mult}: {}", vecs[0].transpose());   // 3: [[1, 1]], 1: [[-1, 1]]
    }
    let (p, d) = s.diagonalize().unwrap();
    assert_eq!(p.matmul(&d).unwrap().matmul(&p.inv().unwrap()).unwrap().equals(&s), Some(true));

    let j = matrix![ctx, [5, 4, 2, 1], [0, 1, -1, -1], [-1, -1, 3, 0], [1, 1, -1, 2]];
    println!("{:?}", j.eigenvals_with_multiplicity().unwrap());   // [(4, 2), (2, 1), (1, 1)]
    println!("{:?}", j.is_diagonalizable());                      // Some(false)
    let (_p, jordan) = j.jordan_form().unwrap();
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
        let (re, im) = ev.eval_complex64().unwrap();
        println!("{ev} ≈ {re:.6} {im:+.6}i");       // RootOf(λ^3 - λ - 1, k) ≈ …
    }
}
```

## Decompositions

| Method | Returns | Preconditions (→ `Err`) |
|--------|---------|-------------------------|
| `lu()` | `(L, U, permutation)` | square |
| `cholesky()` | `L` with `L·Lᵀ = A` | symmetric, positive definite |
| `ldl()` | `(L, D)` | symmetric |
| `qr()` | `(Q, R)` with exact radicals | — |
| `matrix_decomp::gram_schmidt(&vectors, normalize)` | orthogonal (or orthonormal) basis | linearly independent input |
| `rref()` | `(R, pivot_columns)` | — |
| `pinv()` | Moore–Penrose pseudo-inverse | — |

```rust
use symplex::prelude::*;
use symplex::matrix_decomp::gram_schmidt;

fn main() {
    let ctx = Context::new();
    let spd = matrix![ctx, [4, 12, -16], [12, 37, -43], [-16, -43, 98]];
    let l = spd.cholesky().unwrap();
    println!("{l}");                                             // [[2,0,0],[6,1,0],[-8,5,3]]
    assert_eq!(l.matmul(&l.transpose()).unwrap().equals(&spd), Some(true));
    let (l, d) = spd.ldl().unwrap();
    println!("{l} {d}");
    assert!(matrix![ctx, [1, 2], [3, 4]].cholesky().is_err());  // not symmetric

    let m = matrix![ctx, [1, 1, 0], [1, 0, 1], [0, 1, 1]];
    let (q, r) = m.qr().unwrap();
    println!("{q}\n{r}");                                        // sqrt(1/2), sqrt(2/3), …
    assert_eq!(q.is_orthogonal(), Some(true));
    assert_eq!(q.matmul(&r).unwrap().simplify().equals(&m), Some(true));

    let v1 = Matrix::col_vector(vec![ctx.int(1), ctx.int(1), ctx.int(0)]);
    let v2 = Matrix::col_vector(vec![ctx.int(1), ctx.int(0), ctx.int(1)]);
    for b in gram_schmidt(&[v1, v2], true).unwrap() {
        println!("{}", b.transpose());
    }
    let (lu_l, lu_u, perm) = matrix![ctx, [2, 1], [4, 3]].lu().unwrap();
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
    let (hnf, u) = z.hermite_normal_form_with_transform();
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

You rarely need to convert by hand: `Matrix::{rref, rank, nullspace, columnspace, rowspace, left_nullspace, det, inv, solve, solve_least_squares, pinv}`, `linsolve` / `linsolve_matrix` and every function in `symplex::normalforms` detect all-rational input and route through `QMatrix`/`ZMatrix` themselves, returning the same `Matrix` results as before (the RREF is unique, so pivots and entries are identical). A single symbolic entry sends the whole matrix down the expression path. Use the exact types directly when the data is numeric from the start — LP formulations, coefficient matrices from `Poly::coefficient_matrix`, lattices — to skip the arena round trip.

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
