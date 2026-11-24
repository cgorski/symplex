//! Tests of session 17, wave 18 (s17b_spectrum): an incomplete spectrum is an error; stability decided from certified signs.
//! Every test says what was wrong before and cites its oracle.

use symplex::control::{StateSpace, is_routh_stable};
use symplex::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// The companion matrix of the monic `xⁿ + c[n−1]·xⁿ⁻¹ + … + c[0]`: ones
/// above the diagonal, `−c` in the last row.  Its characteristic polynomial
/// is that polynomial.
fn companion(ctx: &Context, c: &[Ex]) -> Matrix {
    let n = c.len();
    Matrix::from_fn(n, n, |i, j| {
        if i + 1 < n {
            if j == i + 1 { ctx.one() } else { ctx.zero() }
        } else {
            -&c[j]
        }
    })
    .unwrap()
}

fn block_diag(ctx: &Context, blocks: &[Matrix]) -> Matrix {
    let n: usize = blocks.iter().map(|b| b.nrows()).sum();
    let mut rows = vec![vec![ctx.zero(); n]; n];
    let mut at = 0;
    for b in blocks {
        for i in 0..b.nrows() {
            for j in 0..b.ncols() {
                rows[at + i][at + j] = b.get(i, j).clone();
            }
        }
        at += b.nrows();
    }
    Matrix::new(rows).unwrap()
}

/// `P·B·P⁻¹` for the dense unimodular `P = L·Lᵀ` (`L` lower triangular of
/// ones): a similar matrix that is one block, so that its eigenvalues come
/// from the whole characteristic polynomial.
fn conjugated(ctx: &Context, b: &Matrix) -> Matrix {
    let n = b.nrows();
    let l = Matrix::from_fn(n, n, |i, j| if j <= i { ctx.one() } else { ctx.zero() }).unwrap();
    let p = l.matmul(&l.transpose()).unwrap();
    let p_inv = p.inv().unwrap();
    p.matmul(b)
        .unwrap()
        .matmul(&p_inv)
        .unwrap()
        .map(|e| e.expand())
}

fn ints(ctx: &Context, c: &[i64]) -> Vec<Ex> {
    c.iter().map(|&k| ctx.int(k)).collect()
}

/// A single-input, single-output system with the state matrix `a`.
fn system(a: Matrix) -> StateSpace {
    let ctx = a.get(0, 0).context();
    let n = a.nrows();
    let b = Matrix::from_fn(n, 1, |_, _| ctx.one()).unwrap();
    let c = Matrix::from_fn(1, n, |_, _| ctx.one()).unwrap();
    StateSpace::new(a, b, c, matrix![ctx, [0]]).unwrap()
}

/// The 6×6 matrix of the survey's recipe (S3 §1.5): `P·diag(1, C)·P⁻¹` with
/// `C` the companion matrix of `λ⁵ − a·λ − 1`.
fn one_and_a_quintic(ctx: &Context, a: &Ex) -> Matrix {
    let quintic = [ctx.int(-1), -a, ctx.int(0), ctx.int(0), ctx.int(0)];
    let b = block_diag(ctx, &[matrix![ctx, [1]], companion(ctx, &quintic)]);
    conjugated(ctx, &b)
}

/// Is the multiset `got` the multiset `want` (pairs `(re, im)`) to `1e-12`?
fn same_values(got: &[Ex], want: &[(f64, f64)]) -> bool {
    let mut left: Vec<(f64, f64)> = want.to_vec();
    for g in got {
        let z = g.eval_complex64().unwrap();
        let Some(k) = left
            .iter()
            .position(|(re, im)| (z.re - re).abs() < 1e-12 && (z.im - im).abs() < 1e-12)
        else {
            return false;
        };
        left.swap_remove(k);
    }
    left.is_empty()
}

// ── 1. An incomplete spectrum is an error (B-C11-01) ────────────────────────────────────

/// The reproducer of B-C11-01, built from the survey's recipe.  The matrix
/// is one block with the characteristic polynomial `(λ − 1)·(λ⁵ − a·λ −
/// 1)` (checked here by expanding: an exact identity); the quintic has no
/// roots in closed form.
///
/// Before (0.59.0): `eigenvals_with_multiplicity()` was `Ok([(1, 1)])` — one
/// eigenvalue for a 6×6 matrix — `eigenvals()` `Ok([1])`, `eigenvects()` one
/// vector, and `StateSpace::poles()` `[1]`.  Now each is the error of an
/// incomplete spectrum, which names the eigenvalue found.
#[test]
fn a_spectrum_with_an_unsolved_factor_is_an_error() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let m = one_and_a_quintic(&ctx, &a);
    let lam = ctx.symbol("t");
    let expected = (&(&lam - 1) * &(&lam.powi(5) - &(&a * &lam) - 1)).expand();
    assert_eq!(m.char_poly(&lam).unwrap().expand(), expected);

    let err = m.eigenvals_with_multiplicity().unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err:?}"
    );
    let text = err.to_string();
    assert!(text.contains("found 1 of its 6 roots (1)"), "{text}");
    assert!(matches!(
        m.eigenvals(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert!(matches!(
        m.eigenvects(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    let ss = system(m);
    assert!(ss.poles().is_empty());
    assert!(matches!(
        ss.try_poles(),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

/// What is built on the spectrum inherits the error, not the gap.
///
/// Before (0.59.0), for the matrix above: `diagonalize()` was
/// `Err(NotDiagonalizable { reason: "insufficient eigenvectors found" })` —
/// a verdict, through the variant callers branch on, about a matrix that is
/// diagonalizable for all but finitely many `a` (its six eigenvalues are
/// distinct unless the discriminant of the quintic or its value at 1
/// vanishes).  Now `ComputationFailed`; `is_diagonalizable()` stays `None`,
/// and `jordan_form`, `matrix_exp` and `singular_values` (of a matrix
/// whose Gram matrix has such a factor) are errors as before.
#[test]
fn functions_built_on_the_spectrum_report_the_gap() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let m = one_and_a_quintic(&ctx, &a);
    let err = m.diagonalize().unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err:?}"
    );
    assert_eq!(m.is_diagonalizable(), None);
    assert!(matches!(
        m.jordan_form(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert!(matches!(
        m.matrix_exp(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert!(matches!(
        m.matrix_exp_t(&ctx.symbol("t")),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

/// With a number for the parameter every factor has exact roots, and the
/// six eigenvalues are there.
///
/// Oracle: mpmath, `polyroots([1, 0, 0, 0, -3, -1])` at 30 digits:
/// `−1.2146480426984618`, `−0.33473414194335269`, `0.080295100117280154 ±
/// 1.3283551098206541i`, `1.3887919844072542`; and the eigenvalue 1.
#[test]
fn the_same_matrix_with_a_number_has_six_eigenvalues() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let m = one_and_a_quintic(&ctx, &a).subs(&a, &ctx.int(3));
    let ev = m.eigenvals().unwrap();
    assert_eq!(ev.len(), 6);
    let want = [
        (1.0, 0.0),
        (-1.214_648_042_698_461_8, 0.0),
        (-0.334_734_141_943_352_7, 0.0),
        (0.080_295_100_117_280_15, 1.328_355_109_820_654),
        (0.080_295_100_117_280_15, -1.328_355_109_820_654),
        (1.388_791_984_407_254_3, 0.0),
    ];
    assert!(same_values(&ev, &want), "{ev:?}");
    // one eigenvector each: the matrix is diagonalizable
    let vects = m.eigenvects().unwrap();
    assert_eq!(vects.iter().map(|(_, k, _)| *k).sum::<usize>(), 6);
    assert!(vects.iter().all(|(_, k, vs)| vs.len() == *k));
    assert_eq!(m.is_diagonalizable(), Some(true));
}

/// Neighbours: a repeated solvable factor and a quadratic with the
/// parameter next to the unsolved quintic, and blocks instead of one block.
///
/// Before (0.59.0): `[(1, 2), (2, 1)]` for the 8×8 matrix with the
/// characteristic polynomial `(λ − 1)²·(λ − 2)·(λ⁵ − a·λ − 1)` and `[±√a]`
/// for the 7×7 one with `(λ² − a)·(λ⁵ − a·λ − 1)`.  The block-diagonal
/// `diag(1, C)` was an error already (its message did not name the 1).
#[test]
fn neighbours_of_the_reproducer() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let quintic = [ctx.int(-1), -&a, ctx.int(0), ctx.int(0), ctx.int(0)];
    let c = companion(&ctx, &quintic);

    let b = block_diag(
        &ctx,
        &[
            matrix![ctx, [1]],
            matrix![ctx, [1]],
            matrix![ctx, [2]],
            c.clone(),
        ],
    );
    let err = conjugated(&ctx, &b)
        .eigenvals_with_multiplicity()
        .unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("found 3 of its 8 roots"), "{err}");

    let quadratic = companion(&ctx, &[-&a, ctx.int(0)]);
    let b = block_diag(&ctx, &[quadratic, c.clone()]);
    let err = conjugated(&ctx, &b).eigenvals().unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("found 2 of its 7 roots"), "{err}");

    let blocks = block_diag(&ctx, &[matrix![ctx, [1]], c.clone()]);
    let err = blocks.eigenvals().unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err:?}"
    );
    assert!(
        err.to_string().contains("found 1 of its 6 roots (1)"),
        "{err}"
    );

    // alone, nothing is found: the message of 0.59
    let err = c.eigenvals().unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err:?}"
    );
}

/// A rational matrix always has its whole spectrum: an irreducible factor
/// of any degree has the exact roots `RootOf(factor, k)`, each with the
/// multiplicity of the factor.
///
/// Oracle: mpmath, `polyroots([1, 0, 0, 0, -1, -1])` at 30 digits:
/// `−0.76488443360058473 ± 0.35247154603172625i`, `0.18123244446987538 ±
/// 1.0839541013177107i`, `1.1673039782614187`.  The 12×12 matrix is
/// `P·diag(1, 1, C, C)·P⁻¹` with `C` the companion matrix of `λ⁵ − λ − 1`:
/// every root of the quintic twice, the eigenvalue 1 twice.
#[test]
fn a_rational_matrix_has_its_whole_spectrum() {
    let ctx = Context::new();
    let quintic = [
        (-0.764_884_433_600_584_8, 0.352_471_546_031_726_26),
        (-0.764_884_433_600_584_8, -0.352_471_546_031_726_26),
        (0.181_232_444_469_875_38, 1.083_954_101_317_710_7),
        (0.181_232_444_469_875_38, -1.083_954_101_317_710_7),
        (1.167_303_978_261_418_7, 0.0),
    ];
    let c = companion(&ctx, &ints(&ctx, &[-1, -1, 0, 0, 0]));
    let ev = c.eigenvals().unwrap();
    assert_eq!(ev.len(), 5);
    assert!(same_values(&ev, &quintic), "{ev:?}");

    let one = matrix![ctx, [1]];
    let b = block_diag(&ctx, &[one.clone(), one, c.clone(), c]);
    let pairs = conjugated(&ctx, &b).eigenvals_with_multiplicity().unwrap();
    assert_eq!(pairs.len(), 6);
    assert!(pairs.iter().all(|(_, k)| *k == 2), "{pairs:?}");
    let distinct: Vec<Ex> = pairs.into_iter().map(|(v, _)| v).collect();
    let mut want = quintic.to_vec();
    want.push((1.0, 0.0));
    assert!(same_values(&distinct, &want), "{distinct:?}");

    // degrees 6 and 7 (λ⁶ + λ + 3 and λ⁷ − 2λ + 2, irreducible)
    for coeffs in [vec![3, 1, 0, 0, 0, 0], vec![2, -2, 0, 0, 0, 0, 0]] {
        let n = coeffs.len();
        let m = companion(&ctx, &ints(&ctx, &coeffs));
        let pairs = m.eigenvals_with_multiplicity().unwrap();
        assert_eq!(pairs.iter().map(|(_, k)| *k).sum::<usize>(), n);
        // Σλ = tr(A) = 0
        let sum: num_complex::Complex64 =
            pairs.iter().map(|(v, _)| v.eval_complex64().unwrap()).sum();
        assert!(sum.norm() < 1e-12, "{sum}");
    }
}

// ── 2. Stability from proved signs ──────────────────────────────────────────────────────

/// A pole below the range of `f64`.  `e^(−1000) = 5.0759588975494568·10⁻⁴³⁵`
/// (mpmath, `exp(-1000)`), so `−e^(−1000)` is in the open left half-plane.
///
/// Before (0.59.0): `is_stable()` read `eval_f64()`, the correctly rounded
/// `−0.0`, and `−0.0 >= 0.0` holds: `Some(false)` for the stable system, and
/// for the pair `−e^(−1000) ± i` (through `eval_complex64().re`).  The
/// unstable twins were `Some(false)` by the same accident.
#[test]
fn poles_below_the_range_of_f64() {
    let ctx = Context::new();
    let tiny = ctx.parse("exp(-1000)").unwrap();
    let one = ctx.int(1);

    let stable = system(Matrix::new(vec![vec![-&tiny]]).unwrap());
    assert_eq!(stable.is_stable(), Some(true));
    let unstable = system(Matrix::new(vec![vec![tiny.clone()]]).unwrap());
    assert_eq!(unstable.is_stable(), Some(false));

    // [[x, 1], [−1, x]] has the eigenvalues x ± i.
    let pair = |x: &Ex| {
        system(Matrix::new(vec![vec![x.clone(), one.clone()], vec![-&one, x.clone()]]).unwrap())
    };
    assert_eq!(pair(&-&tiny).is_stable(), Some(true));
    assert_eq!(pair(&tiny).is_stable(), Some(false));
}

/// A pole exactly on the imaginary axis is decided by a proved zero.
///
/// * `A = [[0]]`, the rotation `[[0, 1], [−1, 0]]` (poles `±i`): not
///   asymptotically stable.
/// * `√2 + √3 − √(5 + 2√6)` is exactly 0 (`(√2 + √3)² = 5 + 2√6`); mpmath
///   at 30 digits prints `−3.9·10⁻³¹`, a rounding residue of the sign that
///   would read "stable".
/// * The companion matrix of `s⁶ + 6s⁴ + 9s² + 1` (irreducible): its six
///   poles are `RootOf` values on the axis — mpmath, `polyroots([1, 0, 6, 0,
///   9, 0, 1])`: `±0.3472963553338607i`, `±1.5320888862379561i`,
///   `±1.8793852415718168i`, real parts 0 — whose computed real parts are
///   rounding noise.  Decided by the zero row of the Routh table (exact).
#[test]
fn poles_on_the_imaginary_axis() {
    let ctx = Context::new();
    assert_eq!(system(matrix![ctx, [0]]).is_stable(), Some(false));
    assert_eq!(
        system(matrix![ctx, [0, 1], [-1, 0]]).is_stable(),
        Some(false)
    );

    let hidden = ctx
        .parse("sqrt(2) + sqrt(3) - sqrt(5 + 2*sqrt(6))")
        .unwrap();
    let a = Matrix::new(vec![
        vec![ctx.int(-1), ctx.int(1)],
        vec![ctx.int(0), hidden],
    ])
    .unwrap();
    assert_eq!(system(a).is_stable(), Some(false));

    let sextic = companion(&ctx, &ints(&ctx, &[1, 0, 9, 0, 6, 0]));
    assert_eq!(system(sextic).is_stable(), Some(false));
}

/// A part of the spectrum proves "not stable" and never "stable".
///
/// The 6-state system of the reproducer has the pole 1 and five poles that
/// are not found: `Some(false)` (the pole 1 is a pole).  With the block
/// `[−1]` in its place the pole found is `−1`; the five others are the roots
/// of `s⁵ − a·s − 1`, which sum to 0, so one of them has a real part `≥ 0`:
/// the system is not stable for any `a`.  For a real `a` that is proved
/// (the characteristic polynomial `s⁶ + s⁵ − a·s² − (a + 1)·s − 1` has a
/// zero coefficient); for a plain, possibly complex `a` the answer is not
/// `Some(true)`.
///
/// Before (0.59.0): `Some(false)` in both cases, the second one from the
/// real-polynomial rule applied to a polynomial not known to be real.
#[test]
fn a_partial_spectrum_never_proves_stability() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    assert_eq!(system(one_and_a_quintic(&ctx, &a)).is_stable(), Some(false));

    let with_minus_one = |a: &Ex| {
        let quintic = [ctx.int(-1), -a, ctx.int(0), ctx.int(0), ctx.int(0)];
        let b = block_diag(&ctx, &[matrix![ctx, [-1]], companion(&ctx, &quintic)]);
        system(conjugated(&ctx, &b))
    };
    assert_ne!(with_minus_one(&a).is_stable(), Some(true));
    let r = ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    assert_eq!(with_minus_one(&r).is_stable(), Some(false));
}

/// The real part of a symbolic pole is decided by the assumption system.
///
/// For `k > 0`: `A = [[−1 + i·k]]` has the pole `−1 + i·k` (real part `−1`),
/// `[[1 + i·k]]` real part 1, `[[i·k]]` real part 0, `[[i·k − k]]` real
/// part `−k < 0`.
///
/// Before (0.59.0): `None`, `None`, `Some(false)`, `None` — a pole that
/// did not evaluate was left to the Routh table, read for a polynomial with
/// complex coefficients.
#[test]
fn symbolic_poles_by_the_sign_of_their_real_part() {
    let ctx = Context::new();
    let k = ctx.symbol_with("k", &[Assumption::Positive]).unwrap();
    let ik = &ctx.i_unit() * &k;
    let one_pole = |p: Ex| system(Matrix::new(vec![vec![p]]).unwrap()).is_stable();
    assert_eq!(one_pole(&ik - 1), Some(true));
    assert_eq!(one_pole(&ik + 1), Some(false));
    assert_eq!(one_pole(ik.clone()), Some(false));
    assert_eq!(one_pole(&ik - &k), Some(true));
}

/// `is_routh_stable` reads signs of real numbers only.
///
/// * `i·s + (1 − i)`: the root is `−(1 − i)/i = 1 + i`.  Before: `Some(true)`
///   (`i` and `1 − i` are "not positive and not zero", read as two negative
///   entries without a sign change).
/// * `s + (1 + i)`: the root is `−1 − i`.  Before: `Some(false)`.
/// * `i·s² + 2i·s + (i − 100) = i·((s + 1)² + 100i)`: the roots are `−1 ±
///   10·√(−i) = −1 ± (5√2 − 5√2·i)`, real parts `6.07…` and `−8.07…`.
///   Before: `Some(true)`.
/// * `s² + 2s + (1 + 100i)`: the same roots.  `Some(false)` before too.
/// * `s² + (2 + i)·s + (1 + i) = (s + 1)·(s + 1 + i)`: stable.
///
/// Each complex case is decided through `p·p̄`, a real polynomial with the
/// same real parts of roots.
#[test]
fn routh_with_complex_coefficients() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let int = |k: i64| ctx.int(k);
    assert_eq!(is_routh_stable(&[i.clone(), &int(1) - &i]), Some(false));
    assert_eq!(is_routh_stable(&[int(1), &int(1) + &i]), Some(true));
    assert_eq!(
        is_routh_stable(&[i.clone(), &int(2) * &i, &i - 100]),
        Some(false)
    );
    assert_eq!(
        is_routh_stable(&[int(1), int(2), &int(1) + &(&int(100) * &i)]),
        Some(false)
    );
    assert_eq!(
        is_routh_stable(&[int(1), &int(2) + &i, &int(1) + &i]),
        Some(true)
    );
    // a nonzero constant has no roots, real or not
    assert_eq!(is_routh_stable(std::slice::from_ref(&i)), Some(true));
    // unchanged for real coefficients
    assert_eq!(is_routh_stable(&ints(&ctx, &[1, 3, 2])), Some(true));
    assert_eq!(is_routh_stable(&ints(&ctx, &[1, 0, 1])), Some(false));
    assert_eq!(is_routh_stable(&ints(&ctx, &[1, 1, 2, 2, 3])), Some(false));
}

/// Coefficients below the range of `f64`, and a symbol that may be complex.
///
/// `s + e^(−1000)` has the root `−e^(−1000) < 0`; `s − e^(−1000)` the root
/// `e^(−1000) > 0` (both were right in 0.59: these signs were already
/// certified).  A plain symbol may be complex, so `s² + s + k` is not
/// decided; `s² − s + k` is unstable for every complex `k` (the roots sum
/// to 1), and for `k > 0` the first is stable.
#[test]
fn routh_signs_are_proved() {
    let ctx = Context::new();
    let tiny = ctx.parse("exp(-1000)").unwrap();
    assert_eq!(is_routh_stable(&[ctx.int(1), tiny.clone()]), Some(true));
    assert_eq!(is_routh_stable(&[ctx.int(1), -&tiny]), Some(false));
    let k = ctx.symbol("k");
    assert_eq!(is_routh_stable(&[ctx.int(1), ctx.int(1), k.clone()]), None);
    assert_eq!(is_routh_stable(&[ctx.int(1), ctx.int(-1), k]), Some(false));
    let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    assert_eq!(is_routh_stable(&[ctx.int(1), ctx.int(1), p]), Some(true));
}
