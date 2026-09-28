//! Linear algebra over `ℚ(√p…, i)(params)`: QR with the Hermitian inner
//! product, the Gram–Schmidt of symbolic complex matrices in exact field
//! arithmetic, eigenvalues and eigenvectors of 3×3 matrices with algebraic
//! entries.

use std::time::{Duration, Instant};
use symplex::decompositions::Qr;
use symplex::num_complex::Complex64;
use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

fn mat(ctx: &Context, rows: &[&[&str]]) -> Matrix {
    Matrix::new(
        rows.iter()
            .map(|r| r.iter().map(|s| p(ctx, s)).collect())
            .collect(),
    )
    .unwrap()
}

fn c64(e: &Ex) -> Complex64 {
    e.eval_complex64()
        .unwrap_or_else(|err| panic!("{e}: {err}"))
}

fn num(m: &Matrix, bind: Option<(&Ex, &Ex)>) -> Vec<Vec<Complex64>> {
    (0..m.nrows())
        .map(|i| {
            (0..m.ncols())
                .map(|j| {
                    let e = m.get(i, j);
                    c64(&match bind {
                        Some((s, v)) => e.subs(s, v),
                        None => e.clone(),
                    })
                })
                .collect()
        })
        .collect()
}

/// `QᴴQ = I`, `Q·R = A`, `R` upper triangular with a real positive
/// diagonal, numerically (with the parameter bound).
fn assert_unitary_qr(a: &Matrix, q: &Matrix, r: &Matrix, bind: Option<(&Ex, &Ex)>) {
    let (an, qn, rn) = (num(a, bind), num(q, bind), num(r, bind));
    let (m, n) = (an.len(), an[0].len());
    let scale = an.iter().flatten().map(|z| z.norm()).fold(1.0, f64::max);
    for i in 0..n {
        assert!(
            rn[i][i].im.abs() < 1e-12 * scale && rn[i][i].re > 0.0,
            "R[{i}][{i}] = {}",
            rn[i][i]
        );
        for (j, z) in rn[i].iter().enumerate().take(i) {
            assert!(z.norm() < 1e-12 * scale, "R[{i}][{j}] = {z}");
        }
        for j in 0..n {
            let s: Complex64 = (0..m).map(|k| qn[k][i].conj() * qn[k][j]).sum();
            let id = if i == j { 1.0 } else { 0.0 };
            assert!((s - id).norm() < 1e-12, "(QᴴQ)[{i}][{j}] = {s}");
        }
    }
    for i in 0..m {
        for j in 0..n {
            let s: Complex64 = (0..n).map(|k| qn[i][k] * rn[k][j]).sum();
            assert!(
                (s - an[i][j]).norm() < 1e-12 * scale,
                "(QR)[{i}][{j}] = {s} vs {}",
                an[i][j]
            );
        }
    }
}

/// `qr` of the column `[1, i]ᵀ` was refused as "linearly dependent": the
/// bilinear product `1 + i² = 0`.  QR now uses the Hermitian product, as
/// SymPy (`u.dot(v, hermitian=True)` in `_QRdecomposition_optional`).
///
/// SymPy: `Matrix([1, I]).QRdecomposition()` → `Q = [sqrt(2)/2,
/// sqrt(2)*I/2]`, `R = [sqrt(2)]`; `Matrix([[1, I], [I, 1]])` → `Q =
/// [[sqrt(2)/2, sqrt(2)*I/2], [sqrt(2)*I/2, sqrt(2)/2]]`, `R = sqrt(2)·I`.
#[test]
fn qr_uses_the_hermitian_inner_product() {
    let ctx = Context::new();
    let a = mat(&ctx, &[&["1"], &["I"]]);
    let Qr { q, r } = a.qr().unwrap();
    assert_eq!(q.get(0, 0).to_string(), "1/2*sqrt(2)");
    assert_eq!(q.get(1, 0).to_string(), "1/2*sqrt(2)*I");
    assert_eq!(r.get(0, 0).to_string(), "sqrt(2)");
    assert_eq!(
        q.adjoint().matmul(&q).unwrap().eval(),
        Matrix::identity(&ctx, 1).unwrap()
    );

    let a = mat(&ctx, &[&["1", "I"], &["I", "1"]]);
    let Qr { q, r } = a.qr().unwrap();
    let want_q = mat(
        &ctx,
        &[&["sqrt(2)/2", "sqrt(2)*I/2"], &["sqrt(2)*I/2", "sqrt(2)/2"]],
    );
    assert_eq!(q, want_q.eval());
    assert_eq!(r, mat(&ctx, &[&["sqrt(2)", "0"], &["0", "sqrt(2)"]]).eval());
    assert_eq!(q.is_unitary(), Some(true));
}

/// The 3×2 complex case of 0.31 (`v30_linalg`) gave a `Q` with `QᵀQ = I`;
/// now exactly SymPy's factors.
///
/// SymPy: `Matrix([[0, -2/5], [1/2, -3], [-I, 2*I]]).QRdecomposition()` →
/// `Q = [[0, -sqrt(21)/21], [sqrt(5)/5, -4*sqrt(21)/21], [-2*sqrt(5)*I/5,
/// -2*sqrt(21)*I/21]]`, `R = [[sqrt(5)/2, -7*sqrt(5)/5], [0, 2*sqrt(21)/5]]`.
#[test]
fn complex_qr_matches_sympy() {
    let ctx = Context::new();
    let a = mat(&ctx, &[&["0", "-2/5"], &["1/2", "-3"], &["-I", "2*I"]]);
    let Qr { q, r } = a.qr().unwrap();
    let want_q = mat(
        &ctx,
        &[
            &["0", "-sqrt(21)/21"],
            &["sqrt(5)/5", "-4*sqrt(21)/21"],
            &["-2*sqrt(5)*I/5", "-2*sqrt(21)*I/21"],
        ],
    );
    let want_r = mat(
        &ctx,
        &[&["sqrt(5)/2", "-7*sqrt(5)/5"], &["0", "2*sqrt(21)/5"]],
    );
    assert_eq!(q, want_q.eval());
    assert_eq!(r, want_r.eval());
}

/// Real matrices keep their factors (the bilinear and the Hermitian
/// products agree; the real path is unchanged).
///
/// SymPy: `Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]]).QRdecomposition()` →
/// `Q = [[sqrt(2)/2, sqrt(6)/6, -sqrt(3)/3], [sqrt(2)/2, -sqrt(6)/6,
/// sqrt(3)/3], [0, sqrt(6)/3, sqrt(3)/3]]`.
#[test]
fn real_qr_is_unchanged() {
    let ctx = Context::new();
    let a = mat(
        &ctx,
        &[&["1", "1", "0"], &["1", "0", "1"], &["0", "1", "1"]],
    );
    let Qr { q, r } = a.qr().unwrap();
    assert_eq!(
        q.to_string(),
        "[\n  [1/2*sqrt(2),  1/6*sqrt(6), -1/3*sqrt(3)],\n  [1/2*sqrt(2), -1/6*sqrt(6),  1/3*sqrt(3)],\n  [          0,  1/3*sqrt(6),  1/3*sqrt(3)]\n]"
    );
    assert_eq!(
        r.to_string(),
        "[\n  [sqrt(2), 1/2*sqrt(2), 1/2*sqrt(2)],\n  [      0, 1/2*sqrt(6), 1/6*sqrt(6)],\n  [      0,           0, 2/3*sqrt(3)]\n]"
    );
}

/// The 4×4 matrix of the 0.31 hand-off in one complex parameter: 53 s in a
/// debug build (every projection `simplify`d; `Q` printed 307 KB, `R` 245
/// KB).  Now fraction-free Gram–Schmidt over `ℚ(√2, √3, i)(a, ā)`: 0.14 s,
/// 17 KB and 16 KB.
///
/// Oracle: the defining identities at parameter values (SymPy's
/// `QRdecomposition()` of this matrix did not finish in 60 s).
#[test]
fn symbolic_complex_qr_is_exact_and_fast() {
    let ctx = Context::new();
    let a = mat(
        &ctx,
        &[
            &["1", "0", "0", "-3"],
            &["3", "a^2*(sqrt(2)+2)", "0", "3"],
            &["1", "a^2*(-sqrt(2)-sqrt(3))-1", "0", "-2"],
            &[
                "-3",
                "-a*I-I",
                "-sqrt(3)*a^2-2*sqrt(2)",
                "a^2*(sqrt(2)+sqrt(3))+3",
            ],
        ],
    );
    let t = Instant::now();
    let Qr { q, r } = a.qr().unwrap();
    assert!(t.elapsed() < Duration::from_secs(10), "{:?}", t.elapsed());
    assert!(q.to_string().len() + r.to_string().len() < 60_000);
    let s = ctx.symbol("a");
    for v in ["3/7 + 2/5*I", "-5/3 + 1/9*I", "2", "-1/2*I"] {
        assert_unitary_qr(&a, &q, &r, Some((&s, &p(&ctx, v))));
    }
}

/// A symbol without a real assumption is complex: `Q` carries
/// `conjugate(x)` (SymPy's convention), and `QᴴQ = I` holds at complex `x`
/// (the bilinear `x/√(x² + 1)` of 0.31 did not).
///
/// SymPy: `Matrix([[x, 1], [1, 0]]).QRdecomposition()` → `Q[0, 0] =
/// x/sqrt(Abs(x)**2 + 1)`, `R[0, 1] = sqrt(Abs(x)**2 + 1)*conjugate(x)/(x*conjugate(x) + 1)`.
#[test]
fn qr_with_a_complex_symbol_carries_conjugates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = mat(&ctx, &[&["x", "1"], &["1", "0"]]);
    let Qr { q, r } = a.qr().unwrap();
    assert_eq!(q.get(0, 0).to_string(), "x/sqrt(x*conjugate(x) + 1)");
    assert_eq!(
        r.get(0, 1).to_string(),
        "conjugate(x)/sqrt(x*conjugate(x) + 1)"
    );
    for v in ["1 + 2*I", "-3/2*I", "5"] {
        assert_unitary_qr(&a, &q, &r, Some((&x, &p(&ctx, v))));
    }
    // A real symbol keeps the real path (no conjugates).
    let y = ctx.symbol_with("y", &[Assumption::Real]).unwrap();
    let b = Matrix::new(vec![
        vec![y.clone(), ctx.int(1)],
        vec![ctx.int(1), ctx.int(0)],
    ])
    .unwrap();
    let Qr { q, .. } = b.qr().unwrap();
    assert!(!q.to_string().contains("conjugate"), "{q}");
}

/// `gram_schmidt` orthogonalises with the Hermitian product too.
///
/// SymPy: `GramSchmidt([Matrix([1, I]), Matrix([1, 0])], True)` →
/// `[[sqrt(2)/2, sqrt(2)*I/2], [sqrt(2)/2, -sqrt(2)*I/2]]`.
#[test]
fn gram_schmidt_is_hermitian() {
    let ctx = Context::new();
    let v1 = mat(&ctx, &[&["1"], &["I"]]);
    let v2 = mat(&ctx, &[&["1"], &["0"]]);
    let q = symplex::matrix_decomp::gram_schmidt(&[v1, v2], true).unwrap();
    assert_eq!(q[0], mat(&ctx, &[&["sqrt(2)/2"], &["sqrt(2)*I/2"]]).eval());
    assert_eq!(q[1], mat(&ctx, &[&["sqrt(2)/2"], &["-sqrt(2)*I/2"]]).eval());
}

/// `solve_least_squares` used `AᵀA` for complex `A` too (not the minimiser
/// of `‖Ax − b‖`); now `AᴴA`, as SymPy's `M.H`.
///
/// numpy: `lstsq([[1, 1j], [2, 3], [1, 1]], [1, 2, 3])` →
/// `[1.5 - 0.125j, -0.125 + 0.125j]`; SymPy's `solve_least_squares` agrees.
#[test]
fn least_squares_uses_the_conjugate_transpose() {
    let ctx = Context::new();
    let a = mat(&ctx, &[&["1", "I"], &["2", "3"], &["1", "1"]]);
    let b = mat(&ctx, &[&["1"], &["2"], &["3"]]);
    let x = a.solve_least_squares(&b).unwrap();
    assert_eq!(
        x.eval(),
        mat(&ctx, &[&["3/2 - I/8"], &["-1/8 + I/8"]]).eval()
    );
}

/// `singular_values` took the eigenvalues of `AᵀA` for complex `A` too: the
/// "singular values" of `[[i, 0], [0, 2]]` were `i` and `2`.  Now `AᴴA`, as
/// SymPy's `_singular_values` (`M.H.multiply(M).eigenvals()`).
///
/// SymPy: `Matrix([[I, 0], [0, 2]]).singular_values()` → `[2, 1]`;
/// `Matrix([[1, I], [2, 3]]).singular_values()` → `[sqrt(sqrt(173)/2 +
/// 15/2), sqrt(15/2 - sqrt(173)/2)]`; numpy `svd` → `3.7518626332773635,
/// 0.9610030078085334`.
#[test]
fn singular_values_of_complex_matrices() {
    let ctx = Context::new();
    let sv = mat(&ctx, &[&["I", "0"], &["0", "2"]])
        .singular_values()
        .unwrap();
    assert_eq!(sv, vec![ctx.int(2), ctx.int(1)]);
    let sv = mat(&ctx, &[&["1", "I"], &["2", "3"]])
        .singular_values()
        .unwrap();
    for (s, w) in sv.iter().zip([3.7518626332773635, 0.9610030078085334]) {
        assert!((c64(s) - Complex64::new(w, 0.0)).norm() < 1e-13, "{s}");
    }
}

/// A real 4×4 matrix over `ℚ(√2, √3)` exceeded the expression budget on the
/// real path (hunter `zz_hunt_la2 qr` seed 186); it now falls back to the
/// exact field arithmetic.
///
/// Oracle: the defining identities (numerically).
#[test]
fn real_algebraic_qr_falls_back_to_exact_arithmetic() {
    let ctx = Context::new();
    let a = mat(
        &ctx,
        &[
            &["-sqrt(2) - sqrt(3)", "-sqrt(3)", "sqrt(2) + 2", "-2"],
            &["-sqrt(2)", "-3", "1", "1"],
            &["-1", "-1/5", "-sqrt(2)", "0"],
            &["0", "-1", "0", "-1"],
        ],
    );
    let Qr { q, r } = a.qr().unwrap();
    assert_unitary_qr(&a, &q, &r, None);
}

/// The characteristic polynomial of this matrix is `λ·(λ² + bλ + c)` over
/// `ℚ(√2, √3, i)` (its determinant is 0); `eigenvals` was refused ("could
/// not solve the characteristic polynomial"): the solver takes only
/// binomials with algebraic coefficients.  Now: square-free factors over
/// `K`, split by their norms over `ℚ`, then the quadratic formula (Cardano
/// for a cubic), as SymPy's `roots`.
///
/// SymPy (0.15 s): `Matrix([[-3*sqrt(3), -3*sqrt(2) - 3, 3*sqrt(2) +
/// 3*sqrt(3)], [sqrt(3), sqrt(2) + 1, -sqrt(2) - sqrt(3)], [2 + I, 1/4,
/// sqrt(2) + 1]]).eigenvals()` → `{0: 1, -3*sqrt(3)/2 + 1 + sqrt(2) ±
/// sqrt(27 + sqrt(2)*(23 + 12*I) + sqrt(3)*(23 + 12*I))/2: 1}`; numpy
/// `eigvals` → `-5.25411042e+00-9.30801959e-01j, 4.88638512e+00+9.30801959e-01j, ≈0`.
#[test]
fn eigenvalues_of_an_algebraic_3x3_with_zero_determinant() {
    let ctx = Context::new();
    let a = mat(
        &ctx,
        &[
            &["-3*sqrt(3)", "-3*sqrt(2) - 3", "3*sqrt(2) + 3*sqrt(3)"],
            &["sqrt(3)", "sqrt(2) + 1", "-sqrt(2) - sqrt(3)"],
            &["2 + I", "1/4", "sqrt(2) + 1"],
        ],
    );
    let ev = a.eigenvals_with_multiplicity().unwrap();
    assert_eq!(ev.len(), 3);
    assert!(ev.iter().all(|(_, m)| *m == 1));
    let s = p(
        &ctx,
        "sqrt(27 + sqrt(2)*(23 + 12*I) + sqrt(3)*(23 + 12*I))/2",
    );
    let base = p(&ctx, "-3*sqrt(3)/2 + 1 + sqrt(2)");
    let want = [ctx.int(0), (&base + &s).eval(), (&base - &s).eval()];
    for w in &want {
        assert!(
            ev.iter()
                .any(|(l, _)| (l - w).eval().is_zero_structural() || c64(&(l - w)).norm() < 1e-13),
            "{w} not among {ev:?}"
        );
    }
    assert_eigenvalues_exact(&a, &ev);
}

/// `det(A − λI) = 0` for each eigenvalue by a certified evaluation at 30
/// digits (`eval_decimal` prints `0` for a value that is zero to the
/// precision reached).
fn assert_eigenvalues_exact(a: &Matrix, ev: &[(Ex, usize)]) {
    let ctx = a.get(0, 0).context();
    let lam = ctx.symbol("lambda_test");
    let cp = a.char_poly_coeffs().unwrap();
    let poly = cp
        .iter()
        .enumerate()
        .fold(ctx.int(0), |acc, (k, c)| &acc + &(c * &lam.powi(k as i64)));
    for (l, _) in ev {
        assert_eq!(poly.subs(&lam, l).eval_decimal(30).unwrap(), "0", "p({l})");
    }
}

/// An irreducible cubic over `ℚ(√2, √3, i)`: refused before; now Cardano's
/// formulas (SymPy's `roots_cubic`), every eigenvalue certified and equal
/// to numpy's.
///
/// SymPy: `Matrix([[1, sqrt(2), 0], [0, 2, I], [1, 0, sqrt(3)]]).eigenvals()`
/// → three Cardano roots `0.53084168953861773445 + 0.51353646941893664564i`,
/// `1.5880374690632759704 − 1.0431340564906015038i`,
/// `2.6131716489669835887 + 0.52959758707166485814i`; numpy `eigvals` →
/// `0.5308416895386172+0.5135364694189363j`,
/// `1.5880374690632757-1.0431340564906013j`, `2.6131716489669836+0.529597587071665j`.
#[test]
fn cardano_eigenvalues_of_an_irreducible_algebraic_cubic() {
    let ctx = Context::new();
    let a = mat(
        &ctx,
        &[
            &["1", "sqrt(2)", "0"],
            &["0", "2", "I"],
            &["1", "0", "sqrt(3)"],
        ],
    );
    let t = Instant::now();
    let ev = a.eigenvals_with_multiplicity().unwrap();
    assert!(t.elapsed() < Duration::from_secs(10));
    let numpy = [
        Complex64::new(0.5308416895386172, 0.5135364694189363),
        Complex64::new(1.5880374690632757, -1.0431340564906013),
        Complex64::new(2.6131716489669836, 0.529597587071665),
    ];
    assert_eq!(ev.len(), 3);
    for w in numpy {
        assert!(
            ev.iter()
                .any(|(l, m)| *m == 1 && (c64(l) - w).norm() < 1e-12 * w.norm()),
            "{w} not among {ev:?}"
        );
    }
    assert_eigenvalues_exact(&a, &ev);
    // Eigenvectors: exactly over K[t]/(g) (before, the elimination on the
    // Cardano expressions found no pivot decision: no vector).
    let an = num(&a, None);
    for (l, m, vs) in a.eigenvects().unwrap() {
        assert_eq!((m, vs.len()), (1, 1), "{l}");
        let z = c64(&l);
        let v: Vec<Complex64> = (0..3).map(|i| c64(vs[0].get(i, 0))).collect();
        for i in 0..3 {
            let s: Complex64 = (0..3).map(|j| an[i][j] * v[j]).sum::<Complex64>() - z * v[i];
            assert!(s.norm() < 1e-12, "A·v − λv = {s} for λ = {z}");
        }
    }
}

/// A defective matrix over `ℚ(√2, i)` with a double eigenvalue: the
/// multiplicities come from the square-free decomposition over `K`.
///
/// SymPy: `(P*J*P.inv())` for `P = [[1, 2, 0], [0, 1, 1], [1, 0, 1]]`,
/// `J = [[sqrt(2), 1, 0], [0, sqrt(2), 0], [0, 0, I]]`: `eigenvals()` →
/// `{sqrt(2): 2, I: 1}`, `eigenvects()` → `[(sqrt(2), 2, [[1, 0, 1]]),
/// (I, 1, [[0, 1, 1]])]`.
#[test]
fn repeated_algebraic_eigenvalue() {
    let ctx = Context::new();
    let a = mat(
        &ctx,
        &[
            &["1/3 + sqrt(2)", "1/3", "-1/3"],
            &["sqrt(2)/3 - I/3", "sqrt(2)/3 + 2*I/3", "-sqrt(2)/3 + I/3"],
            &[
                "1/3 + sqrt(2)/3 - I/3",
                "-2*sqrt(2)/3 + 1/3 + 2*I/3",
                "-1/3 + 2*sqrt(2)/3 + I/3",
            ],
        ],
    );
    let mut ev = a.eigenvals_with_multiplicity().unwrap();
    ev.sort_by_key(|(_, m)| *m);
    assert_eq!(ev[0].0, ctx.i_unit());
    assert_eq!(ev[1], (ctx.int(2).sqrt(), 2));
    for (l, m, vs) in a.eigenvects().unwrap() {
        assert_eq!(vs.len(), 1, "{l} (multiplicity {m})");
        let v: Vec<Complex64> = (0..3).map(|i| c64(vs[0].get(i, 0))).collect();
        let want = if m == 2 {
            [1.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 1.0]
        };
        // proportional to SymPy's vector
        let k = (0..3).find(|&i| want[i] != 0.0).unwrap();
        for i in 0..3 {
            assert!((v[i] - v[k] * want[i]).norm() < 1e-12, "{l}: {v:?}");
        }
    }
}
