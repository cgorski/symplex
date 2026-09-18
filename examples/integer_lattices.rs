//! Integer matrix normal forms and lattices (symplex 0.3).
//!
//! Demonstrates:
//! - row-style Hermite normal form `H = U·A` with its unimodular transform
//!   `U`, verified by multiplying back;
//! - the column-style form `H = A·V` (SymPy's convention), on SymPy's own
//!   documented example;
//! - Smith normal form `S = U·A·V` and its invariant factors, again
//!   verified by multiplying back;
//! - the integer kernel `{x ∈ ℤⁿ : A·x = 0}` and why it differs from the
//!   rational nullspace scaled to integers;
//! - `is_unimodular` and `lattice_determinant`.
//!
//! Run with: `cargo run --example integer_lattices`

use symplex::normalforms::{
    column_hermite_normal_form, hermite_normal_form_with_transform, integer_nullspace,
    is_unimodular, lattice_determinant, smith_normal_form_with_transforms,
};
use symplex::prelude::*;

fn main() {
    println!("=== Integer normal forms and lattices ===\n");
    let ctx = Context::new();

    // ── 1. Row Hermite normal form ──────────────────────────────────────
    let a = matrix![ctx, [2, 4, 4], [-6, 6, 12], [10, -4, -16]];
    println!("A =\n{a}");
    let (h, u) = hermite_normal_form_with_transform(&a).unwrap();
    println!("row HNF H = U·A:\nH =\n{h}\nU =\n{u}");
    let back = (&u * &a).eval();
    assert_eq!(back, h);
    println!("check U·A == H : {}", back == h);
    println!("det U          : {}", u.det().unwrap());
    println!(
        "det A          : {}  (= ± product of pivots 2·6·12)",
        a.det().unwrap()
    );
    // Unique ⇒ idempotent and invariant under unimodular left factors.
    assert_eq!(h.hermite_normal_form().unwrap(), h);
    let v = matrix![ctx, [1, 3, 0], [0, 1, 0], [2, 0, 1]];
    assert_eq!((&v * &a).eval().hermite_normal_form().unwrap(), h);
    println!("HNF(HNF(A)) == HNF(A) and HNF(V·A) == HNF(A) for unimodular V: ok");

    // ── 2. Column Hermite normal form (SymPy convention) ────────────────
    println!("\n--- column HNF, SymPy's documented example ---");
    let m = matrix![ctx, [12, 6, 4], [3, 9, 6], [2, 16, 14]];
    let hc = column_hermite_normal_form(&m).unwrap();
    println!("M =\n{m}\nH = M·V =\n{hc}");
    println!("(SymPy: hermite_normal_form(Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]]))");
    println!("        == Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]]))");
    assert_eq!(hc, matrix![ctx, [10, 0, 2], [0, 15, 3], [0, 0, 2]]);

    // ── 3. Smith normal form ────────────────────────────────────────────
    println!("\n--- Smith normal form ---");
    let (s, us, vs) = smith_normal_form_with_transforms(&m).unwrap();
    println!("S = U·M·V =\n{s}");
    println!("U =\n{us}\nV =\n{vs}");
    let back = (&(&us * &m) * &vs).eval();
    assert_eq!(back, s);
    println!("check U·M·V == S : {}", back == s);
    println!(
        "invariant factors 1 | 10 | 30, product 300 = |det M| = {}",
        m.det().unwrap()
    );
    println!("(SymPy: smith_normal_form(...) == Matrix([[1, 0, 0], [0, 10, 0], [0, 0, 30]]))");

    let rect = matrix![ctx, [2, 4, 6, 8], [3, 6, 9, 15]];
    println!(
        "\nrectangular:\n{rect}\nSNF =\n{}",
        rect.smith_normal_form().unwrap()
    );

    // ── 4. Integer kernel ───────────────────────────────────────────────
    println!("\n--- integer nullspace of [2 1 1] ---");
    let k = matrix![ctx, [2, 1, 1]];
    let basis = integer_nullspace(&k).unwrap();
    for (i, b) in basis.iter().enumerate() {
        let img = (&k * b).eval();
        assert_eq!(img.is_zero_matrix(), Some(true));
        println!("k{i} = {:?}   A·k{i} = {img}", b.col(0));
    }
    // Rational nullspace scaled to integers spans only an index-2 sublattice.
    let rational: Vec<String> = k
        .nullspace()
        .iter()
        .map(|v| format!("{:?}", v.col(0)))
        .collect();
    println!("rational nullspace basis: {}", rational.join(", "));
    println!(
        "scaled to integers, (−1,2,0), (−1,0,2) miss (0,1,−1): the ℤ-basis above is saturated."
    );

    // ── 5. Unimodularity and lattice determinants ───────────────────────
    println!("\n--- unimodular? ---");
    for mm in [
        matrix![ctx, [2, 1], [1, 1]],
        matrix![ctx, [2, 0], [0, 1]],
        matrix![ctx, [1, 5, -3], [0, 1, 4], [0, 0, -1]],
    ] {
        println!("{:?}  → {}", mm.to_vec(), is_unimodular(&mm).unwrap());
    }
    println!("\n--- lattice determinant (index of the column lattice in ℤᵐ) ---");
    let l = matrix![ctx, [2, 0, 1], [0, 3, 1]];
    println!(
        "columns (2,0), (0,3), (1,1): index {}  (gcd of 2×2 minors 6, 2, −3)",
        lattice_determinant(&l).unwrap()
    );
    let l2 = matrix![ctx, [2, 0, 2], [0, 4, 2]];
    println!(
        "columns (2,0), (0,4), (2,2): index {}",
        lattice_determinant(&l2).unwrap()
    );
    println!(
        "rank-deficient [[1,2],[2,4]] → {}",
        lattice_determinant(&matrix![ctx, [1, 2], [2, 4]]).unwrap_err()
    );

    println!("\nDone.");
}
