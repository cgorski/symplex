# Chapter 7: Matrices

Symplex provides a symbolic `Matrix` type for small, exact linear algebra — the kind you encounter in robotics (4×4 homogeneous transforms), controls (state-space models), and physics (inertia tensors). This chapter covers construction, operations, decompositions, and code generation.

## Construction

### The `matrix!` Macro

The most convenient way to build a matrix:

```rust
use symplex::prelude::*;

// 2×2 numeric matrix
let a = matrix![[1, 2], [3, 4]];
println!("{a}");

// 3×3 identity-like
let b = matrix![[1, 0, 0], [0, 1, 0], [0, 0, 1]];
```

The macro accepts integer literals, which are converted to exact symbolic integers internally. You can also use symbolic expressions:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x, y);

let m = matrix![[x, 1], [0, y]];
println!("{m}");
// [x, 1]
// [0, y]
```

### Programmatic Construction

For matrices built from computed values:

```rust
use symplex::prelude::*;

// From nested Vec
let rows = vec![
    vec![symplex::int(1), symplex::int(2)],
    vec![symplex::int(3), symplex::int(4)],
];
let a = Matrix::new(rows);

// Zeros
let z = Matrix::zeros(3, 3);

// Identity
let eye = Matrix::identity(4);

// Diagonal
let d = Matrix::diag(&[symplex::int(1), symplex::int(2), symplex::int(3)]);

// From a function
let m = Matrix::from_fn(3, 3, |i, j| {
    if i == j { symplex::int(1) } else { symplex::int(0) }
});

// Row and column vectors
let row = Matrix::row_vector(vec![symplex::int(1), symplex::int(2), symplex::int(3)]);
let col = Matrix::col_vector(vec![symplex::int(4), symplex::int(5), symplex::int(6)]);
```

### Shape and Access

```rust
use symplex::prelude::*;

let m = matrix![[1, 2, 3], [4, 5, 6]];
println!("Shape: {:?}", m.shape());  // (2, 3)
println!("Rows: {}", m.nrows());     // 2
println!("Cols: {}", m.ncols());     // 3
println!("m[0,1] = {}", m.get(0, 1)); // 2

// Get an entire row
let row = m.row(0); // [1, 2, 3]
```

## Arithmetic

Matrices support the standard arithmetic operators via references:

```rust
use symplex::prelude::*;

let a = matrix![[1, 2], [3, 4]];
let b = matrix![[5, 6], [7, 8]];

// Addition and subtraction
let sum = &a + &b;
let diff = &a - &b;

// Matrix multiplication
let product = &a * &b;

// Scalar multiplication
let scaled = &a * &symplex::int(3);
let scaled_i64 = &a * 3;

// Negation
let neg = -&a;

// Transpose
let at = a.transpose();
println!("Aᵀ = {at}");
```

All operations produce new matrices — nothing is modified in place.

### Matrix Power

Raise a square matrix to an integer power:

```rust
use symplex::prelude::*;

let a = matrix![[1, 1], [0, 1]];
let a3 = a.powi(3);
println!("A³ = {a3}");
// [1, 3]
// [0, 1]
```

### Kronecker Product

```rust
use symplex::prelude::*;

let a = matrix![[1, 2], [3, 4]];
let b = matrix![[0, 1], [1, 0]];
let kron = a.kronecker(&b);
println!("A ⊗ B = {kron}"); // 4×4 matrix
```

## Fundamental Operations

### Determinant

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let a = matrix![[2, 1], [1, 3]];
println!("det(A) = {}", a.det()); // 5

// Symbolic determinant
let b = matrix![[x, 1], [1, x]];
println!("det(B) = {}", b.det()); // x^2 - 1
```

The determinant uses LU decomposition for larger matrices and Bareiss's algorithm for exact integer arithmetic — no floating-point division.

### Trace

```rust
use symplex::prelude::*;

let a = matrix![[2, 1], [1, 3]];
println!("tr(A) = {}", a.trace()); // 5
```

### Inverse

```rust
use symplex::prelude::*;

let a = matrix![[2, 1], [1, 3]];
if let Some(inv) = a.inv() {
    println!("A⁻¹ = {inv}");
    // Verify: A · A⁻¹ = I
    let product = &a * &inv;
    println!("A·A⁻¹ = {product}");
}
```

`.inv()` returns `Option<Matrix>` — it's `None` if the matrix is singular.

### Pseudoinverse

For non-square or singular matrices:

```rust
use symplex::prelude::*;

let a = matrix![[1, 2], [3, 4], [5, 6]]; // 3×2
let pinv = a.pinv();
println!("A⁺ = {pinv}");
```

## Characteristic Polynomial and Eigenvalues

### Characteristic Polynomial

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let a = matrix![[2, 1], [1, 3]];
let poly = a.char_poly(&x);
println!("char poly: {poly}"); // x^2 - 5*x + 5 (or equivalent)
```

The variable `x` here plays the role of λ in the characteristic equation `det(A - λI) = 0`.

### Eigenvalues

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let a = matrix![[2, 1], [1, 3]];
let eigenvals = a.eigenvals(&x);
println!("Eigenvalues:");
for ev in &eigenvals {
    println!("  λ = {ev}");
}
```

Eigenvalues are found by solving the characteristic polynomial, so they're exact for matrices up to 4×4 (the polynomial solver handles up to degree 4).

## Decompositions

### LU Decomposition

```rust
use symplex::prelude::*;

let a = matrix![[2, 1], [1, 3]];
let (l, u, perm) = a.lu();
println!("L = {l}");
println!("U = {u}");
println!("Permutation: {perm:?}");

// Verify: P·A = L·U
let lu_product = &l * &u;
println!("L·U = {lu_product}");
```

The LU decomposition returns lower-triangular `L`, upper-triangular `U`, and a permutation vector.

### Cholesky Decomposition

For symmetric positive-definite matrices:

```rust
use symplex::prelude::*;

let a = matrix![[4, 2], [2, 3]];
if let Some(l) = a.cholesky() {
    println!("L = {l}");
    // Verify: A = L·Lᵀ
    let product = &l * &l.transpose();
    println!("L·Lᵀ = {product}");
}
```

### RREF (Reduced Row Echelon Form)

```rust
use symplex::prelude::*;

let a = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
let rref = a.rref();
println!("RREF = {rref}");
```

### Rank

```rust
use symplex::prelude::*;

let a = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
println!("rank = {}", a.rank()); // 2 (rows are linearly dependent)

let b = matrix![[1, 0], [0, 1]];
println!("rank = {}", b.rank()); // 2
```

### Nullspace

```rust
use symplex::prelude::*;

let a = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
let null = a.nullspace();
println!("Nullspace basis vectors:");
for (i, v) in null.iter().enumerate() {
    println!("  v{} = {:?}", i, (0..v.ncols()).map(|j| format!("{}", v.get(0, j))).collect::<Vec<_>>());
}
```

### Column Space

```rust
use symplex::prelude::*;

let a = matrix![[1, 2], [3, 4], [5, 6]];
let colspace = a.columnspace();
println!("Column space basis vectors: {}", colspace.len());
```

## Symbolic Matrix Operations

Matrices can contain symbolic expressions and support symbolic transformations:

### Differentiation

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let m = matrix![[x, x.powi(2)], [x.sin(), x.exp()]];
let dm = m.diff(&x);
println!("dM/dx = {dm}");
// [1, 2*x]
// [cos(x), exp(x)]
```

### Substitution

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let m = matrix![[x, 1], [0, x]];
let m_at_3 = m.subs(&x, &symplex::int(3));
println!("M(3) = {m_at_3}");
// [3, 1]
// [0, 3]
```

### Evaluation, Expansion, and Simplification

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let m = matrix![[x.sin().powi(2) + x.cos().powi(2), 0], [0, 1]];
let simplified = m.simplify();
println!("{simplified}");
// [1, 0]
// [0, 1]

let m2 = matrix![[(&x + 1).powi(2), 0], [0, 1]];
let expanded = m2.expand();
println!("{expanded}");
// [x^2 + 2*x + 1, 0]
// [0, 1]
```

## Jacobian Computation

The Jacobian matrix of a vector-valued function is critical in robotics and optimization. Use `symplex::matrix::jacobian()`:

```rust
use symplex::prelude::*;
use symplex::matrix::jacobian;
use symplex::vars;

vars!(x, y);

// f(x,y) = [x²+y, x*y²]
let f1 = expr!(x^2 + y);
let f2 = expr!(x * y^2);

let jac = jacobian(&[&f1, &f2], &[&x, &y]);
println!("J = {jac}");
// [2*x, 1    ]
// [y^2, 2*x*y]

println!("det(J) = {}", jac.det());
```

### Jacobian for Robotics

A common pattern: compute the FK position, then take the Jacobian with respect to joint angles:

```rust
use symplex::prelude::*;
use symplex::matrix::jacobian;
use symplex::robotics::*;
use symplex::vars;

vars!(theta1, theta2);

let l1 = symplex::rational(1, 1);
let l2 = symplex::rational(1, 1);
let zero = symplex::int(0);

let dh: [(&Ex, &Ex, &Ex, &Ex); 2] = [
    (&theta1, &zero, &l1, &zero),
    (&theta2, &zero, &l2, &zero),
];

let (px, py, _pz) = fk_position(&dh);
let jac = jacobian(&[&px, &py], &[&theta1, &theta2]);
println!("Robot Jacobian (2×2):");
println!("{jac}");
```

## Stacking and Combining

### Horizontal Stack

```rust
use symplex::prelude::*;

let a = matrix![[1, 2], [3, 4]];
let b = matrix![[5], [6]];
let combined = Matrix::hstack(&[&a, &b]);
println!("{combined}");
// [1, 2, 5]
// [3, 4, 6]
```

### Vertical Stack

```rust
use symplex::prelude::*;

let a = matrix![[1, 2]];
let b = matrix![[3, 4], [5, 6]];
let combined = Matrix::vstack(&[&a, &b]);
println!("{combined}");
// [1, 2]
// [3, 4]
// [5, 6]
```

## Norms and Properties

```rust
use symplex::prelude::*;

let a = matrix![[1, 2], [3, 4]];
println!("Is square: {}", a.is_square());     // true
println!("Is symmetric: {}", a.is_symmetric()); // false

let sym = matrix![[1, 2], [2, 1]];
println!("Is symmetric: {}", sym.is_symmetric()); // true
```

### Matrix Exponential (Series Approximation)

For the matrix exponential via Taylor series:

```rust
use symplex::prelude::*;

let a = matrix![[0, 1], [-1, 0]];
let exp_a = a.exp_series(6); // 6-term Taylor approximation
println!("exp(A) ≈ {exp_a}");
```

## LaTeX Output

```rust
use symplex::prelude::*;

let m = matrix![[1, 2], [3, 4]];
println!("{}", m.to_latex());
// \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
```

Symbolic matrices render cleanly too:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let m = matrix![[x.sin(), x.cos()], [-x.cos(), x.sin()]];
println!("{}", m.to_latex());
```

## Code Generation

Generate optimized Rust functions from symbolic matrices:

```rust
use symplex::prelude::*;
use symplex::vars;

vars!(x);

let m = matrix![[x.sin(), x.cos()], [-x.cos(), x.sin()]];
let code = m.to_rust_fn("rotation_2d", &["x"]).expect("codegen");
println!("{code}");
```

The generated function returns a flat array `[f64; R*C]` in row-major order. Cross-entry common subexpression elimination (CSE) ensures that shared computations like `sin(x)` are computed once and reused across all matrix entries.

### With Options

```rust
use symplex::prelude::*;
use symplex::matrix::{CodegenOptions, Precision};
use symplex::vars;

vars!(x);

let m = Matrix::new(vec![
    vec![x.sin(), x.cos()],
    vec![-x.cos(), x.sin()],
]);
let opts = CodegenOptions { precision: Precision::F32, ..Default::default() };
let code = m.to_rust_fn_with_options("rotation_2d_f32", &["x"], &opts).expect("codegen");
println!("{code}");
```

See [Chapter 8: Code Generation](08-code-generation.md) for the full code generation story.

## Vector Operations

Symplex provides dot product and cross product for column vectors:

```rust
use symplex::prelude::*;
use symplex::matrix::{dot, cross};
use symplex::vars;

vars!(x, y, z);

let a = Matrix::col_vector(vec![symplex::int(1), symplex::int(2), symplex::int(3)]);
let b = Matrix::col_vector(vec![symplex::int(4), symplex::int(5), symplex::int(6)]);

let d = dot(&a, &b);
println!("a · b = {d}"); // 32

let c = cross(&a, &b);
println!("a × b = {c}"); // [-3, 6, -3]
```

## Summary

| Operation | Method | Notes |
|-----------|--------|-------|
| Create | `matrix![[...], [...]]` | Macro for literals |
| Zeros/Identity | `Matrix::zeros(m,n)`, `Matrix::identity(n)` | |
| Determinant | `.det()` | Exact via Bareiss |
| Inverse | `.inv()` | Returns `Option` |
| Eigenvalues | `.eigenvals(&x)` | Up to 4×4 (polynomial degree) |
| Char polynomial | `.char_poly(&x)` | |
| LU decomp | `.lu()` | Returns `(L, U, perm)` |
| Cholesky | `.cholesky()` | SPD matrices only |
| RREF | `.rref()` | |
| Rank | `.rank()` | |
| Nullspace | `.nullspace()` | |
| Jacobian | `jacobian(&fns, &vars)` | Free function |
| Transpose | `.transpose()` | |
| Power | `.powi(n)` | |
| Differentiate | `.diff(&x)` | Element-wise |
| Substitute | `.subs(&x, &val)` | Element-wise |
| LaTeX | `.to_latex()` | bmatrix format |
| Code gen | `.to_rust_fn(name, args)` | With CSE |

---

*[← Chapter 6: Solving Equations](06-solving.md) | [Back to Table of Contents](index.md) | [Chapter 8: Code Generation →](08-code-generation.md)*