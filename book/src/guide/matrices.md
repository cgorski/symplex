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

Element-wise helpers: `map`, `map_indexed`, `subs`, `eval`, `expand`, `simplify`, `diff`, `integrate`, `col`, `row`, `diagonal`, `submatrix`, `set`, `iter`, `to_vec`, `vec` (column-major vectorisation), `hstack`/`vstack`, `kronecker`.

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

See `cargo run --example matrix_decompositions`, `matrix_algebra` and `control_system`.
