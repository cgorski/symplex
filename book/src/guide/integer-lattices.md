# Integer Lattices and Normal Forms

`symplex::normalforms` works with matrices over **ℤ**: Hermite normal form (in both the row and the column convention), Smith normal form, integer kernels, unimodularity and lattice determinants. Everything is exact `BigInt` arithmetic. Inputs are ordinary `Matrix` values whose entries must be integer literals — a fraction, a symbol or an unevaluated constant is an `InvalidArgument` error, never a silent rounding — and results come back as integer `Matrix` values in the same context.

The most common operations are also available as methods: `Matrix::hermite_normal_form`, `Matrix::smith_normal_form`, `Matrix::integer_nullspace`.

Since 0.3.5 the algorithms live on [`ZMatrix`](./matrices.md#exact-matrices-over-ℚ-and-ℤ-qmatrix-and-zmatrix), a plain `BigInt` matrix with no expression arena behind it; the `normalforms` functions convert a `Matrix` to a `ZMatrix`, run the same code and convert back. When your data is already integer, call `ZMatrix::hermite_normal_form()` and friends directly — they return `ZMatrix` values and need no `Context`, and the row/column conventions are identical.

## Hermite normal form (row style): `H = U·A`

`hermite_normal_form(&a)` returns the row-style HNF: a row-echelon matrix with **positive pivots**, every entry **above** a pivot reduced into `[0, pivot)`, and zero rows at the bottom. There is a unimodular `U` (`det U = ±1`) with `H = U·A`; `hermite_normal_form_with_transform` returns `HermiteNormalForm { h, u }`. Because this `H` is unique, the function is idempotent and `HNF(V·A) = HNF(A)` for every unimodular `V` — the rows of `H` are a canonical basis of the row lattice of `A`.

```rust
use symplex::prelude::*;
use symplex::normalforms::{hermite_normal_form, hermite_normal_form_with_transform, is_unimodular};

fn main() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 4, 4], [-6, 6, 12], [10, -4, -16]];
    let h = hermite_normal_form(&a).unwrap();
    println!("{h}");
    let HermiteNormalForm { h: h2, u } = hermite_normal_form_with_transform(&a).unwrap();
    assert_eq!(h, h2);
    println!("{u}");
    println!("{}", (&u * &a).eval() == h);                     // true   — H = U·A, verified
    println!("{} {}", u.det().unwrap(), is_unimodular(&u).unwrap());   // -1 true

    println!("{}", h.hermite_normal_form().unwrap() == h);       // true   — idempotent
    let v = matrix![ctx, [1, 3, 0], [0, 1, 0], [2, 0, 1]];       // unimodular
    println!("{}", (&v * &a).eval().hermite_normal_form().unwrap() == h);   // true

    println!("{}", hermite_normal_form(&matrix![ctx, [1, 2], [2, 4]]).unwrap());     // rank 1
    println!("{}", hermite_normal_form(&matrix![ctx, [3, 1], [1, 2]]).unwrap());
    println!("{}", hermite_normal_form(&matrix![ctx, [0, 2, 3], [0, 4, 5]]).unwrap());
}
```

```text
[
  [2, 4,  4],
  [0, 6,  0],
  [0, 0, 12]
]
[
  [ 1,  0,  0],
  [-1,  3,  2],
  [ 3, -4, -3]
]
true
-1 true
true
true
[
  [1, 2],
  [0, 0]
]
[
  [1, 2],
  [0, 5]
]
[
  [0, 2, 0],
  [0, 0, 1]
]
```

The number of nonzero rows of `H` is the rank, and for a square nonsingular `A` the product of the pivots is `|det A|` (`2·6·12 = 144 = |−144|` here). `U` is not unique when `A` is rank-deficient; the one returned is what the elimination produced.

## Hermite normal form (column style): `H = A·V`

SymPy's `hermite_normal_form` uses the *column* convention of Cohen's Algorithm 2.4.5: column operations, `H = A·V`, the pivot of each nonzero column is its **lowest** nonzero entry, pivot rows increase strictly from left to right (so a square nonsingular matrix gives an **upper**-triangular `H`), pivots are positive, and entries to the *right* of a pivot in its row lie in `[0, pivot)`. `column_hermite_normal_form` implements exactly this, with one difference: SymPy drops leading zero columns, symplex keeps them so that `H = A·V` holds with a square `V`.

```rust
use symplex::prelude::*;
use symplex::normalforms::{column_hermite_normal_form, hermite_normal_form};

fn main() {
    let ctx = Context::new();
    // SymPy: hermite_normal_form(Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]]))
    //        == Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]])
    let m = matrix![ctx, [12, 6, 4], [3, 9, 6], [2, 16, 14]];
    println!("{}", column_hermite_normal_form(&m).unwrap());
    println!("{}", hermite_normal_form(&m).unwrap());           // the row form is a different matrix
    println!("{}", column_hermite_normal_form(&matrix![ctx, [2, 4], [1, 2]]).unwrap());   // zero column kept

    let half = Matrix::new(vec![vec![ctx.rational(1, 2), ctx.int(1)]]).unwrap();
    println!("{}", hermite_normal_form(&half).unwrap_err());
}
```

```text
[
  [10,  0, 2],
  [ 0, 15, 3],
  [ 0,  0, 2]
]
[
  [1, 23,  2],
  [0, 30,  0],
  [0,  0, 10]
]
[
  [0, 2],
  [0, 1]
]
hermite_normal_form: invalid argument: every entry must be an integer literal (fractions and symbolic entries are not allowed)
```

Which convention you want depends on what the rows and columns *mean*: the row form canonicalises the lattice spanned by the **rows** (`ℤ`-module generated by row vectors), the column form canonicalises the lattice spanned by the **columns**. Transposing alone does *not* turn one into the other — the row HNF of `Aᵀ`, transposed back, is lower-triangular with pivots at the top of each column, which is a third normalisation. `column_hermite_normal_form` handles the row/column reversal for you.

## Smith normal form: `S = U·A·V`

`smith_normal_form` returns `diag(d₁, …, dᵣ, 0, …)` with `dᵢ > 0` and `dᵢ | dᵢ₊₁`. The `dᵢ` (the *invariant factors*) are unique: `d₁⋯dₖ` is the gcd of the `k×k` minors of `A`, and for a square nonsingular matrix `d₁⋯dₙ = |det A|`. `smith_normal_form_with_transforms` returns `SmithNormalForm { s, u, v }` with `S = U·A·V` and both transforms unimodular.

```rust
use symplex::prelude::*;
use symplex::normalforms::{smith_normal_form, smith_normal_form_with_transforms};

fn main() {
    let ctx = Context::new();
    let m = matrix![ctx, [12, 6, 4], [3, 9, 6], [2, 16, 14]];
    println!("{}", smith_normal_form(&m).unwrap());
    let SmithNormalForm { s, u, v } = smith_normal_form_with_transforms(&m).unwrap();
    println!("{}", (&(&u * &m) * &v).eval() == s);          // true
    println!("{} {}", u.det().unwrap(), v.det().unwrap());   // 1 1
    println!("{}", m.det().unwrap());                        // 300 = 1·10·30

    let rect = matrix![ctx, [2, 4, 6, 8], [3, 6, 9, 15]];
    println!("{}", rect.smith_normal_form().unwrap());

    // The abelian group ℤ²/⟨(2, 4), (4, 2)⟩ is ℤ/2 ⊕ ℤ/6.
    println!("{}", smith_normal_form(&matrix![ctx, [2, 4], [4, 2]]).unwrap());
}
```

```text
[
  [1,  0,  0],
  [0, 10,  0],
  [0,  0, 30]
]
true
1 1
300
[
  [1, 0, 0, 0],
  [0, 6, 0, 0]
]
[
  [2, 0],
  [0, 6]
]
```

The last example is the classical use: the Smith form of a relation matrix reads off the structure of a finitely generated abelian group (here `ℤ/2 ⊕ ℤ/6`, not `ℤ/12` and not `ℤ/3 ⊕ ℤ/4` — the divisibility chain matters).

## Integer kernels

`integer_nullspace(&a)` returns a **ℤ-basis** of `{x ∈ ℤⁿ : A·x = 0}` as column vectors: `n − rank(A)` vectors that generate *every* integer solution. This is stronger than scaling the rational `nullspace()` to integers, which in general only spans a sublattice of finite index.

```rust
use symplex::prelude::*;
use symplex::normalforms::integer_nullspace;

fn main() {
    let ctx = Context::new();
    let k = matrix![ctx, [2, 1, 1]];
    for b in integer_nullspace(&k).unwrap() {
        println!("{}   A·k = {}", b.transpose(), (&k * &b).eval());
    }
    // [[1, 0, -2]]   A·k = [[0]]
    // [[0, 1, -1]]   A·k = [[0]]
    for v in k.nullspace() {
        println!("{}", v.transpose());
    }
    // [[-1/2, 1, 0]]
    // [[-1/2, 0, 1]]
    println!("{}", matrix![ctx, [1, 0], [0, 1], [1, 1]].integer_nullspace().unwrap().len());   // 0
    for b in matrix![ctx, [1, 2, 3], [4, 5, 6]].integer_nullspace().unwrap() {
        println!("{}", b.transpose());                                                        // [[1, -2, 1]]
    }
}
```

Clearing denominators in the rational basis gives `(−1, 2, 0)` and `(−1, 0, 2)`, which generate only the even-coordinate part of the kernel — `(0, 1, −1)` is an integer solution they cannot reach. The ℤ-basis is *saturated*: it reaches all of them.

## Unimodularity and lattice determinants

`is_unimodular(&a)` is `true` for a square integer matrix with `det = ±1` (an automorphism of `ℤⁿ`; non-square gives `false`). `lattice_determinant(&a)` is the index `[ℤᵐ : A·ℤⁿ]` of the lattice spanned by the **columns** of an `m×n` matrix of full row rank — the product of the column-HNF pivots, equal to the gcd of all `m×m` minors, and `|det A|` in the square case. A rank-deficient matrix has a column lattice of infinite index and is an error.

```rust
use symplex::prelude::*;
use symplex::normalforms::{is_unimodular, lattice_determinant};

fn main() {
    let ctx = Context::new();
    println!("{} {} {}",
        is_unimodular(&matrix![ctx, [2, 1], [1, 1]]).unwrap(),
        is_unimodular(&matrix![ctx, [2, 0], [0, 1]]).unwrap(),
        is_unimodular(&matrix![ctx, [1, 2, 3]]).unwrap());                  // true false false
    println!("{}", lattice_determinant(&matrix![ctx, [2, 0], [0, 3]]).unwrap());        // 6
    println!("{}", lattice_determinant(&matrix![ctx, [2, 0, 1], [0, 3, 1]]).unwrap());  // 1
    println!("{}", lattice_determinant(&matrix![ctx, [2, 0, 2], [0, 4, 2]]).unwrap());  // 4
    println!("{}", lattice_determinant(&matrix![ctx, [1, 2], [2, 4]]).unwrap_err());
    // lattice_determinant: invalid argument: matrix must have full row rank (rank 1 of 2 rows); the column lattice has infinite index otherwise
}
```

The columns `(2, 0), (0, 3), (1, 1)` generate all of `ℤ²` (index 1) even though no two of them do — the `2×2` minors are `6, 2, −3`, whose gcd is `1`.

## Integer helpers

`symplex::ntheory` gained the list-valued gcd/lcm functions these algorithms need, which are also handy for clearing denominators in a certificate:

```rust
use num_bigint::BigInt;
use symplex::linprog::q;
use symplex::ntheory::{gcd_many, igcd, ilcm, lcm_many, rational_lcm_of_denominators};

fn main() {
    let v: Vec<BigInt> = [12, 18, 30].iter().map(|&n| BigInt::from(n)).collect();
    println!("{} {}", gcd_many(&v), lcm_many(&v));                        // 6 180
    println!("{} {}", igcd(&[-4i64, 6]), ilcm(&[4i64, 6, 10]));           // 2 60
    println!("{}", rational_lcm_of_denominators(&[q(1, 2), q(2, 3), q(5, 4)]));   // 12
}
```

`gcd_many(&[])` is `0` and `lcm_many(&[])` is `1` (the empty product); `igcd`/`ilcm` accept any integer type convertible to `BigInt`.

See `cargo run --example integer_lattices` for the complete program.
