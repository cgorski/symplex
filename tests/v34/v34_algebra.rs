//! Cholesky and LDL of Hermitian matrices: `A = L·Lᴴ` and `A = L·D·Lᴴ`,
//! as SymPy's `cholesky` and `LDLdecomposition` with `hermitian=True`
//! (their default).  Before, both refused every Hermitian matrix that is
//! not symmetric ("matrix is not symmetric").  And `solve` of a polynomial
//! over `ℚ(√p…, i)` given as a product, which returned the roots of the
//! factors it could solve alone.  Every reference value cites its SymPy
//! 1.14 or mpmath 1.3 call.

use symplex::prelude::*;

/// Rows of a matrix as strings.
type Rows<'a> = &'a [&'a [&'a str]];

fn mat(ctx: &Context, rows: Rows<'_>) -> Matrix {
    Matrix::new(
        rows.iter()
            .map(|r| r.iter().map(|s| ctx.parse(s).unwrap()).collect())
            .collect(),
    )
    .unwrap()
}

/// `solve` of a product whose factors are polynomials over `ℚ(√p…, i)`:
/// the solver solves a product factor by factor and dropped a factor it
/// could not solve (degree 3 or 4 with algebraic coefficients) from the
/// union, so `solve(x·(x⁴ + √3·x + 1))` was `[0]`, `(x² + 1)·(x³ + √2·x +
/// i)` gave `±i` alone and `(x − 1)²·(x³ + √2·x + 1)` gave `[1]`, though the
/// expanded polynomials have all their roots in radicals.  SymPy 1.14
/// `solve(x*(x**4 + sqrt(3)*x + 1), x)` has 5 roots; mpmath (`mp.dps = 30`
/// and 50) `polyroots` of the expanded coefficients: `0`, `−0.75874495677599
/// ± 0.070695986873045i`, `0.75874495677599 ± 1.070695986873045i`; `±i`,
/// `±0.404404995193889 − 0.725202721660379i`, `1.450405443320758i`; `1`
/// (double), `−0.573634552759303`, `0.286817276379651 ± 1.288800222091314i`.
/// A factor of degree 5 over the field is reported instead of a partial
/// list: `(x² + i)·(x⁵ − √2·x − 1)` (SymPy returns the two roots of `x² + i`
/// alone).
#[test]
fn solve_of_a_product_over_an_algebraic_field_has_every_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: &[(&str, &[(f64, f64)])] = &[
        (
            "x*(x^4 + sqrt(3)*x + 1)",
            &[
                (0.0, 0.0),
                (-0.75874495677599, -0.070695986873045),
                (-0.75874495677599, 0.070695986873045),
                (0.75874495677599, -1.070695986873045),
                (0.75874495677599, 1.070695986873045),
            ],
        ),
        (
            "(x^2 + 1)*(x^3 + sqrt(2)*x + I)",
            &[
                (0.0, -1.0),
                (0.0, 1.0),
                (-0.404404995193889, -0.725202721660379),
                (0.404404995193889, -0.725202721660379),
                (0.0, 1.450405443320758),
            ],
        ),
        (
            "(x - 1)^2*(x^3 + sqrt(2)*x + 1)",
            &[
                (1.0, 0.0),
                (-0.573634552759303, 0.0),
                (0.286817276379651, -1.288800222091314),
                (0.286817276379651, 1.288800222091314),
            ],
        ),
    ];
    for (src, want) in cases {
        let roots = ctx.parse(src).unwrap().solve(&x).unwrap();
        assert_eq!(roots.len(), want.len(), "{src}: {roots:?}");
        let mut pool: Vec<(f64, f64)> = want.to_vec();
        for r in &roots {
            let z = r.eval_complex64().unwrap();
            let k = pool
                .iter()
                .position(|&(a, b)| (z.re - a).abs() < 1e-12 && (z.im - b).abs() < 1e-12)
                .unwrap_or_else(|| panic!("{src}: root {r} = {z} not among {pool:?}"));
            pool.swap_remove(k);
        }
    }
    let err = ctx
        .parse("(x^2 + I)*(x^5 - sqrt(2)*x - 1)")
        .unwrap()
        .solve(&x)
        .unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err:?}"
    );
    assert!(
        err.to_string()
            .contains("7 distinct roots, of which 2 were found"),
        "{err}"
    );
}

fn assert_entries(m: &Matrix, want: &Matrix, what: &str) {
    for i in 0..want.nrows() {
        for j in 0..want.ncols() {
            let d = (m.get(i, j) - want.get(i, j)).simplify();
            assert!(
                d.is_zero_structural(),
                "{what}[{i}][{j}] = {} (want {})",
                m.get(i, j),
                want.get(i, j)
            );
        }
    }
}

/// `Matrix([[2, I], [-I, 2]]).cholesky()` = `[[sqrt(2), 0], [-sqrt(2)*I/2,
/// sqrt(6)/2]]`; `Matrix([[9, 3*I], [-3*I, 5]]).cholesky()` = `[[3, 0],
/// [-I, 2]]`; `Matrix([[4, 1+I, 0], [1-I, 3, I], [0, -I, 2]]).cholesky()`
/// simplified = `[[2, 0, 0], [1/2 - I/2, sqrt(10)/2, 0], [0,
/// -sqrt(10)*I/5, 2*sqrt(10)/5]]`; `Matrix([[2, sqrt(2)*I], [-sqrt(2)*I,
/// 3]]).cholesky()` = `[[sqrt(2), 0], [-I, sqrt(2)]]`.  Each was refused as
/// "not symmetric" before.
#[test]
fn cholesky_of_hermitian_matrices_is_l_times_l_adjoint() {
    let ctx = Context::new();
    let cases: &[(Rows<'_>, Rows<'_>)] = &[
        (
            &[&["2", "I"], &["-I", "2"]],
            &[&["sqrt(2)", "0"], &["-sqrt(2)*I/2", "sqrt(6)/2"]],
        ),
        (
            &[&["9", "3*I"], &["-3*I", "5"]],
            &[&["3", "0"], &["-I", "2"]],
        ),
        (
            &[&["4", "1+I", "0"], &["1-I", "3", "I"], &["0", "-I", "2"]],
            &[
                &["2", "0", "0"],
                &["1/2 - I/2", "sqrt(10)/2", "0"],
                &["0", "-sqrt(10)*I/5", "2*sqrt(10)/5"],
            ],
        ),
        (
            &[&["2", "sqrt(2)*I"], &["-sqrt(2)*I", "3"]],
            &[&["sqrt(2)", "0"], &["-I", "sqrt(2)"]],
        ),
    ];
    for (a, l_want) in cases {
        let a = mat(&ctx, a);
        let l = a.cholesky().unwrap_or_else(|e| panic!("{a}: {e}"));
        assert_entries(&l, &mat(&ctx, l_want), "L");
        assert_eq!(l.is_lower_triangular(), Some(true));
        let back = l.matmul(&l.adjoint()).unwrap().simplify();
        assert_entries(&back, &a, "L·Lᴴ");
    }
}

/// `Matrix([[9, 3*I], [-3*I, 5]]).LDLdecomposition()` = `L = [[1, 0],
/// [-I/3, 1]]`, `D = diag(9, 4)`; `Matrix([[2, I], [-I,
/// 2]]).LDLdecomposition()` = `L = [[1, 0], [-I/2, 1]]`, `D = diag(2,
/// 3/2)`; the 3×3 above: `L = [[1, 0, 0], [1/4 - I/4, 1, 0], [0, -2*I/5,
/// 1]]`, `D = diag(4, 5/2, 8/5)` (simplified).  Refused as "not symmetric"
/// before.
#[test]
fn ldl_of_hermitian_matrices_is_l_d_l_adjoint() {
    let ctx = Context::new();
    let cases: &[(Rows<'_>, Rows<'_>, &[&str])] = &[
        (
            &[&["9", "3*I"], &["-3*I", "5"]],
            &[&["1", "0"], &["-I/3", "1"]],
            &["9", "4"],
        ),
        (
            &[&["2", "I"], &["-I", "2"]],
            &[&["1", "0"], &["-I/2", "1"]],
            &["2", "3/2"],
        ),
        (
            &[&["4", "1+I", "0"], &["1-I", "3", "I"], &["0", "-I", "2"]],
            &[
                &["1", "0", "0"],
                &["1/4 - I/4", "1", "0"],
                &["0", "-2*I/5", "1"],
            ],
            &["4", "5/2", "8/5"],
        ),
    ];
    for (a, l_want, d_want) in cases {
        let a = mat(&ctx, a);
        let Ldl { l, d } = a.ldl().unwrap_or_else(|e| panic!("{a}: {e}"));
        assert_entries(&l, &mat(&ctx, l_want), "L");
        let d_want: Vec<Ex> = d_want.iter().map(|s| ctx.parse(s).unwrap()).collect();
        assert_entries(&d, &Matrix::diag(&d_want).unwrap(), "D");
        let back = l
            .matmul(&d)
            .unwrap()
            .matmul(&l.adjoint())
            .unwrap()
            .simplify();
        assert_entries(&back, &a, "L·D·Lᴴ");
    }
}

/// A Hermitian matrix with a real parameter: SymPy, `x = Symbol('x',
/// real=True)`, `Matrix([[x**2 + 1, I*x], [-I*x, 2]]).cholesky()` simplified
/// = `[[sqrt(x**2 + 1), 0], [-I*x/sqrt(x**2 + 1), sqrt(x**2 + 2)/sqrt(x**2 +
/// 1)]]`.  The second pivot `2 − x²/(x² + 1)` is decided positive over one
/// denominator, `(x² + 2)/(x² + 1)`.  Checked at `x = 3/2`: `L₁₀ = −1.5i/√3.25`,
/// `L₁₁ = √(17/13)`.
#[test]
fn cholesky_of_a_hermitian_matrix_with_a_real_parameter() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let i = ctx.i_unit();
    let a = Matrix::new(vec![
        vec![&x.powi(2) + 1, &i * &x],
        vec![-(&i * &x), ctx.int(2)],
    ])
    .unwrap();
    let l = a.cholesky().unwrap();
    let at = |e: &Ex| e.subs(&x, &ctx.rational(3, 2)).eval_complex64().unwrap();
    let l10 = at(l.get(1, 0));
    assert!(
        l10.re.abs() < 1e-15 && (l10.im + 1.5 / 3.25f64.sqrt()).abs() < 1e-15,
        "{l10}"
    );
    let l11 = at(l.get(1, 1));
    assert!(
        (l11.re - (17.0f64 / 13.0).sqrt()).abs() < 1e-15 && l11.im == 0.0,
        "{l11}"
    );
    assert_eq!(l.get(0, 0), &(&x.powi(2) + 1).sqrt());
}

/// What stays as it was: a matrix provably symmetric keeps `A = L·Lᵀ` — a
/// real one (`Matrix([[25, 15, -5], [15, 18, 0], [-5, 0, 11]]).cholesky()`
/// = `[[5, 0, 0], [3, 3, 0], [-1, 1, 3]]`) and a complex symmetric one,
/// which SymPy decomposes only with `hermitian=False`
/// (`Matrix([[2, I], [I, 2]]).cholesky(hermitian=False)` = `[[sqrt(2), 0],
/// [sqrt(2)*I/2, sqrt(10)/2]]`; the default raises "Matrix must be
/// Hermitian").  An indefinite Hermitian matrix is not positive definite
/// for `cholesky` (SymPy: `NonPositiveDefiniteMatrixError`), while `ldl`
/// decomposes it as it does an indefinite real symmetric one (SymPy refuses
/// this one too): `[[1, 2i], [−2i, 1]] = L·diag(1, −3)·Lᴴ`, `L₁₀ = −2i`.
/// Neither symmetric nor Hermitian is an invalid argument.
#[test]
fn symmetric_matrices_keep_l_times_l_transpose() {
    let ctx = Context::new();
    let real = mat(
        &ctx,
        &[&["25", "15", "-5"], &["15", "18", "0"], &["-5", "0", "11"]],
    );
    assert_eq!(
        real.cholesky().unwrap(),
        mat(
            &ctx,
            &[&["5", "0", "0"], &["3", "3", "0"], &["-1", "1", "3"]]
        )
    );
    let csym = mat(&ctx, &[&["2", "I"], &["I", "2"]]);
    let l = csym.cholesky().unwrap();
    assert_entries(
        &l,
        &mat(&ctx, &[&["sqrt(2)", "0"], &["sqrt(2)*I/2", "sqrt(10)/2"]]),
        "L",
    );
    assert_entries(&l.matmul(&l.transpose()).unwrap().simplify(), &csym, "L·Lᵀ");

    let indefinite = mat(&ctx, &[&["1", "2*I"], &["-2*I", "1"]]);
    let err = indefinite.cholesky().unwrap_err();
    assert!(err.to_string().contains("not positive definite"), "{err}");
    let Ldl { l, d } = indefinite.ldl().unwrap();
    assert_entries(&l, &mat(&ctx, &[&["1", "0"], &["-2*I", "1"]]), "L");
    assert_entries(&d, &mat(&ctx, &[&["1", "0"], &["0", "-3"]]), "D");

    let neither = mat(&ctx, &[&["1", "2"], &["3", "4"]]);
    for err in [
        neither.cholesky().unwrap_err(),
        neither.ldl().map(|_| ()).unwrap_err(),
    ] {
        assert!(
            matches!(err, SymplexError::InvalidArgument { .. }),
            "{err:?}"
        );
        assert!(
            err.to_string().contains("neither symmetric nor Hermitian"),
            "{err}"
        );
    }
}
