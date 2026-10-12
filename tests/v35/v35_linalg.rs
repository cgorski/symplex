//! The linear-algebra hunt after 0.43 (session 10): symbolic parameters,
//! algebraic entries, defective and singular matrices.  Reference values
//! cite the oracle call that produced them (numpy 2 / scipy, SymPy 1.14).

use symplex::control::StateSpace;
use symplex::prelude::*;

/// `|z − want| ≤ tol·(1 + |want|)` for a complex value of `e`.
fn close_c(e: &Ex, re: f64, im: f64, tol: f64, label: &str) {
    let z = e
        .eval_complex64()
        .unwrap_or_else(|err| panic!("{label}: `{e}` does not evaluate: {err}"));
    let d = ((z.re - re).powi(2) + (z.im - im).powi(2)).sqrt();
    let scale = 1.0 + (re * re + im * im).sqrt();
    assert!(d <= tol * scale, "{label}: got {z}, want {re} + {im}i");
}

/// The eigenvalues of `m` at `subs`, as complex numbers sorted by (re, im).
fn sorted_values(vals: &[Ex], subs: &[(&Ex, &Ex)]) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = vals
        .iter()
        .map(|v| {
            let z = v
                .subs_map(subs)
                .eval_complex64()
                .unwrap_or_else(|e| panic!("`{v}` does not evaluate: {e}"));
            (z.re, z.im)
        })
        .collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

fn assert_values(got: &[(f64, f64)], want: &[(f64, f64)], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: {got:?}");
    for (g, w) in got.iter().zip(want) {
        let d = ((g.0 - w.0).powi(2) + (g.1 - w.1).powi(2)).sqrt();
        // The references carry 8 to 15 digits.
        assert!(
            d < 1e-7 * (1.0 + w.0.abs() + w.1.abs()),
            "{label}: got {got:?}, want {want:?}"
        );
    }
}

/// `A·v = λ·v` for every eigenpair, checked at the parameter values `subs`.
fn assert_eigenpairs(m: &Matrix, subs: &[(&Ex, &Ex)], label: &str) {
    let evs = m.eigenvects().unwrap_or_else(|e| panic!("{label}: {e}"));
    let n = m.nrows();
    let total: usize = evs.iter().map(|(_, k, _)| *k).sum();
    assert_eq!(total, n, "{label}: multiplicities");
    let ms = m.subs_map(subs);
    for (lam, _, vecs) in &evs {
        assert!(!vecs.is_empty(), "{label}: no eigenvector for {lam}");
        let l = lam.subs_map(subs);
        for v in vecs {
            let vs = v.subs_map(subs);
            let r = ms.matmul(&vs).unwrap().sub(&vs.scale(&l)).unwrap();
            let size: f64 = vs.iter().map(|e| e.eval_complex64().unwrap().norm()).sum();
            assert!(size > 1e-6, "{label}: zero eigenvector for {lam}");
            for e in r.iter() {
                let z = e.eval_complex64().unwrap();
                assert!(
                    z.norm() < 1e-9 * (1.0 + size),
                    "{label}: |Av - λv| = {z} for {lam}"
                );
            }
        }
    }
}

/// The reported matrix: `k > 0`, irreducible cubic characteristic
/// polynomial `−λ³ − (k − 1)λ² + (4k + 12)λ − 2k³ − 10k² − 22k − 6`.
/// Before 0.44 `eigenvals` was refused ("could not solve the characteristic
/// polynomial; symbolic matrices larger than 2×2 need a factorable
/// characteristic polynomial"), and with it `eigenvects`, `diagonalize`,
/// `matrix_exp` and `StateSpace::poles`.  SymPy 1.14 returns the Cardano
/// roots (`Matrix(...).eigenvals()`); reference values at `k = 2` from
/// `numpy.linalg.eigvals` (`−6.53837545585681`, `2.76918772792841 ±
/// 2.92293986824647i`) and `scipy.linalg.expm(A)[0, 0] = −13.645534429499241`.
#[test]
fn symbolic_cubic_eigenvalues_in_radicals() {
    let ctx = Context::new();
    let k = ctx.symbol_with("k", &[Assumption::Positive]).unwrap();
    let m = Matrix::new(vec![
        vec![ctx.int(2), &k + 1, &k - 1],
        vec![ctx.int(2), -&k - 3, &k + 1],
        vec![-&k - 3, &k - 1, ctx.int(2)],
    ])
    .unwrap();
    let ev = m.eigenvals().unwrap();
    assert_eq!(ev.len(), 3);
    let two = ctx.int(2);
    let at2 = [(&k, &two)];
    assert_values(
        &sorted_values(&ev, &at2),
        &[
            (-6.53837545585681, 0.0),
            (2.76918772792841, -2.92293986824647),
            (2.76918772792841, 2.92293986824647),
        ],
        "eigenvals at k = 2",
    );
    // Σλ = tr A = 1 − k and Πλ = det A, at another point.
    let k7 = ctx.rational(7, 3);
    let sum: Ex = ev.iter().fold(ctx.zero(), |acc, v| &acc + v);
    close_c(&sum.subs(&k, &k7), 1.0 - 7.0 / 3.0, 0.0, 1e-10, "trace");
    let prod: Ex = ev.iter().fold(ctx.one(), |acc, v| &acc * v);
    let det = m.det().unwrap().subs(&k, &k7).eval_f64().unwrap();
    close_c(&prod.subs(&k, &k7), det, 0.0, 1e-10, "det");
    assert_eigenpairs(&m, &at2, "k = 2");
    assert_eigenpairs(&m, &[(&k, &k7)], "k = 7/3");
    // Diagonalizable, and e^A from the eigenvalues.
    assert_eq!(m.is_diagonalizable(), Some(true));
    let e = m.matrix_exp().unwrap();
    close_c(
        &e[(0, 0)].subs(&k, &two),
        -13.645534429499241,
        0.0,
        1e-9,
        "expm",
    );
}

/// The poles of a state-space system with that matrix: before 0.44
/// `StateSpace::poles` was empty (the eigenvalue refusal); numpy as above.
#[test]
fn state_space_poles_of_a_symbolic_3x3_system() {
    let ctx = Context::new();
    let k = ctx.symbol_with("k", &[Assumption::Positive]).unwrap();
    let a = Matrix::new(vec![
        vec![ctx.int(2), &k + 1, &k - 1],
        vec![ctx.int(2), -&k - 3, &k + 1],
        vec![-&k - 3, &k - 1, ctx.int(2)],
    ])
    .unwrap();
    let b = Matrix::col_vector(vec![ctx.int(1), ctx.int(0), ctx.int(0)]).unwrap();
    let c = Matrix::row_vector(vec![ctx.int(1), ctx.int(0), ctx.int(0)]).unwrap();
    let d = Matrix::new(vec![vec![ctx.int(0)]]).unwrap();
    let ss = StateSpace::new(a, b, c, d).unwrap();
    let poles = ss.poles();
    let two = ctx.int(2);
    assert_values(
        &sorted_values(&poles, &[(&k, &two)]),
        &[
            (-6.53837545585681, 0.0),
            (2.76918772792841, -2.92293986824647),
            (2.76918772792841, 2.92293986824647),
        ],
        "poles at k = 2",
    );
}

/// Two plain symbols (no assumptions), irreducible cubic `λ³ − (x + y +
/// 2)λ² + (xy + 2x + 2y)λ − 2xy − 1` (SymPy `factor(charpoly)`): refused
/// before 0.44.  numpy at `x = 3/7, y = −5/3`: `2.15195389, 0.13093773,
/// −1.52098685`; at the complex point `x = 1 + i, y = 2`: `0.92697278 +
/// 1.33345383i, 1.36638234 − 0.49726504i, 2.70664488 + 0.16381121i`.
#[test]
fn symbolic_cubic_without_assumptions() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let m = Matrix::new(vec![
        vec![x.clone(), ctx.int(1), ctx.int(0)],
        vec![ctx.int(0), y.clone(), ctx.int(1)],
        vec![ctx.int(1), ctx.int(0), ctx.int(2)],
    ])
    .unwrap();
    let (xv, yv) = (ctx.rational(3, 7), ctx.rational(-5, 3));
    let at = [(&x, &xv), (&y, &yv)];
    assert_values(
        &sorted_values(&m.eigenvals().unwrap(), &at),
        &[
            (-1.52098685436884, 0.0),
            (0.130937725493088, 0.0),
            (2.15195389078052, 0.0),
        ],
        "eigenvals",
    );
    assert_eigenpairs(&m, &at, "real point");
    let xc = &ctx.int(1) + &ctx.i_unit();
    let y2 = ctx.int(2);
    let atc = [(&x, &xc), (&y, &y2)];
    assert_values(
        &sorted_values(&m.eigenvals().unwrap(), &atc),
        &[
            (0.92697278, 1.33345383),
            (1.36638234, -0.49726504),
            (2.70664488, 0.16381121),
        ],
        "complex point",
    );
    assert_eigenpairs(&m, &atc, "complex point");
}

/// The companion matrix of the irreducible biquadratic `λ⁴ + xλ² + 1`
/// (SymPy's `roots_quartic` takes its biquadratic branch, `±√((−x ± √(x² −
/// 4))/2)`): refused before 0.44.  numpy at `x = 3/7`:
/// `numpy.roots([1, 0, 3/7, 0, 1])` = `±0.62678317 ± 0.77919372i`, at `x =
/// −5`: `±2.18890106, ±0.45685025`.
///
/// The general symbolic quartic stays refused: Ferrari's roots take square
/// roots of numbers that are exactly negative real wherever the quartic has
/// complex roots, with imaginary parts that cancel only exactly, so they do
/// not evaluate there (`λ⁴ + (1 − x)λ³ − xλ² − 2λ + 2x − 1` below, SymPy
/// `factor(charpoly)`).
#[test]
fn symbolic_biquadratic_eigenvalues_and_the_general_quartic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let z = || ctx.int(0);
    let o = || ctx.int(1);
    let m = Matrix::new(vec![
        vec![z(), z(), z(), ctx.int(-1)],
        vec![o(), z(), z(), z()],
        vec![z(), o(), z(), -&x],
        vec![z(), z(), o(), z()],
    ])
    .unwrap();
    let xv = ctx.rational(3, 7);
    let (re, im) = (0.62678317, 0.77919372);
    assert_values(
        &sorted_values(&m.eigenvals().unwrap(), &[(&x, &xv)]),
        &[(-re, -im), (-re, im), (re, -im), (re, im)],
        "eigenvals at x = 3/7",
    );
    let xw = ctx.int(-5);
    assert_values(
        &sorted_values(&m.eigenvals().unwrap(), &[(&x, &xw)]),
        &[
            (-2.18890106, 0.0),
            (-0.45685025, 0.0),
            (0.45685025, 0.0),
            (2.18890106, 0.0),
        ],
        "eigenvals at x = -5",
    );
    assert_eigenpairs(&m, &[(&x, &xv)], "x = 3/7");
    assert_eigenpairs(&m, &[(&x, &xw)], "x = -5");
    let general = Matrix::new(vec![
        vec![x.clone(), o(), z(), z()],
        vec![z(), z(), o(), z()],
        vec![z(), z(), z(), o()],
        vec![o(), ctx.int(2), z(), ctx.int(-1)],
    ])
    .unwrap();
    let err = general.eigenvals().unwrap_err().to_string();
    assert!(err.contains("general quartic"), "{err}");
}

/// A 4×4 matrix similar to `diag(C, 3)` with `C` a symbolic 3×3 block of
/// irreducible cubic characteristic polynomial, by a unimodular change of
/// basis (so the eigenvalue problem does not split into blocks): the
/// characteristic polynomial factors as `(λ − 3)·cubic` and before 0.44 only
/// the eigenvalue 3 was returned (the multiplicities summed to 1 for a 4×4
/// matrix).  numpy at `x = 3/7, y = −5/3`: the cubic's roots as above and 3.
#[test]
fn symbolic_cubic_factor_of_a_4x4_characteristic_polynomial() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let i = |v: i64| ctx.int(v);
    let b = Matrix::new(vec![
        vec![x.clone(), i(1), i(0), i(0)],
        vec![i(0), y.clone(), i(1), i(0)],
        vec![i(1), i(0), i(2), i(0)],
        vec![i(0), i(0), i(0), i(3)],
    ])
    .unwrap();
    let p = Matrix::new(vec![
        vec![i(1), i(0), i(0), i(1)],
        vec![i(0), i(1), i(0), i(1)],
        vec![i(0), i(0), i(1), i(0)],
        vec![i(1), i(0), i(0), i(2)],
    ])
    .unwrap();
    let m = p
        .matmul(&b)
        .unwrap()
        .matmul(&p.inv().unwrap())
        .unwrap()
        .expand();
    let pairs = m.eigenvals_with_multiplicity().unwrap();
    assert_eq!(pairs.iter().map(|(_, k)| *k).sum::<usize>(), 4, "{pairs:?}");
    let (xv, yv) = (ctx.rational(3, 7), ctx.rational(-5, 3));
    let at = [(&x, &xv), (&y, &yv)];
    let vals: Vec<Ex> = pairs.iter().map(|(v, _)| v.clone()).collect();
    assert_values(
        &sorted_values(&vals, &at),
        &[
            (-1.52098685436884, 0.0),
            (0.130937725493088, 0.0),
            (2.15195389078052, 0.0),
            (3.0, 0.0),
        ],
        "eigenvals",
    );
    assert_eigenpairs(&m, &at, "x = 3/7, y = -5/3");
}

/// The matrix `P·diag(i, √2, −√2)·P⁻¹` over `ℚ(√2, i)` splits into a 1×1
/// and a 2×2 block whose quadratic formula needs `√(1 + 2√2·i)`, which is
/// `√2 + i`.  Before 0.44 that square root stayed nested, the eigenvalue
/// `−√2` came out as `(−√2 + i − √(1 + 2√2·i))/2` — on the negative real
/// axis only up to an imaginary part no evaluation can prove zero — and
/// `matrix_sqrt` / `matrix_log` returned entries that did not evaluate
/// (`PrecisionExhausted`).  Checked: `S² = A`, and the traces are the sums of
/// the principal values, `tr √A = 2^(1/4) + (1 + i)/√2 + i·2^(1/4)` and `tr
/// log A = ln 2 + 3πi/2` (SymPy 1.14's `sqrt`/`log` of the eigenvalues;
/// `scipy.linalg.sqrtm` is off the principal branch here: the floating-point
/// eigenvalue `−√2 − 1.8·10⁻¹⁵i` lies below the cut).
#[test]
fn principal_sqrt_and_log_through_a_denested_complex_discriminant() {
    let ctx = Context::new();
    let cells = [
        "3*I + 2*sqrt(2)",
        "0",
        "-2*I - 2*sqrt(2)",
        "-6*sqrt(2) + 6*I",
        "sqrt(2)",
        "4*sqrt(2) - 4*I",
        "3*I + 3*sqrt(2)",
        "0",
        "-2*I - 3*sqrt(2)",
    ];
    let e: Vec<Ex> = cells.iter().map(|s| ctx.parse(s).unwrap()).collect();
    let a = Matrix::new(vec![e[0..3].to_vec(), e[3..6].to_vec(), e[6..9].to_vec()]).unwrap();
    let mut ev: Vec<String> = a
        .eigenvals()
        .unwrap()
        .iter()
        .map(|v| v.to_string())
        .collect();
    ev.sort();
    assert_eq!(ev, ["-sqrt(2)", "I", "sqrt(2)"]);
    let s = a.matrix_sqrt().unwrap();
    let s2 = s.matmul(&s).unwrap().sub(&a).unwrap();
    for x in s2.iter() {
        assert!(x.eval_complex64().unwrap().norm() < 1e-12, "S² − A: {x}");
    }
    let q = 2f64.powf(0.25);
    let h = std::f64::consts::FRAC_1_SQRT_2;
    close_c(&s.trace().unwrap(), q + h, h + q, 1e-12, "tr sqrt");
    let l = a.matrix_log().unwrap();
    close_c(
        &l.trace().unwrap(),
        2f64.ln(),
        1.5 * std::f64::consts::PI,
        1e-12,
        "tr log",
    );
}

/// `p > 0`, eigenvalues `1, p·i, −p`: the quadratic formula needs `√(2p²·i)
/// = p + p·i`, which stayed nested before 0.44 (`√(2p²·i)`), so `−p` was
/// `−p(1 + i)/2 − p/2 + p·i/2` in disguise and `matrix_log` / `matrix_sqrt`
/// did not evaluate.  scipy at `p = 10/11` (`scipy.linalg.logm`):
/// `log(B)[0, 1] = −1.5707963267949i`, `log(B)[0, 2] = 0.0953101798043 −
/// 4.71238898038469i`, `log(B)[1, 1] = −0.0953101798043`.
#[test]
fn matrix_log_with_a_positive_parameter_on_the_negative_axis() {
    let ctx = Context::new();
    let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    let i = ctx.i_unit();
    let b = Matrix::new(vec![
        vec![ctx.int(1), &p + &(&p * &i), &(&(&p * 2) + &(&p * &i)) + 1],
        vec![
            ctx.int(0),
            &p + &(&p * &(&i * 2)),
            &(&p * 2) + &(&p * &(&i * 2)),
        ],
        vec![ctx.int(0), -&p - &(&p * &i), -&(&p * 2) - &(&p * &i)],
    ])
    .unwrap();
    let l = b.matrix_log().unwrap();
    let pv = ctx.rational(10, 11);
    let at = |e: &Ex| e.subs(&p, &pv);
    close_c(
        &at(&l[(0, 1)]),
        0.0,
        -std::f64::consts::FRAC_PI_2,
        1e-12,
        "log[0,1]",
    );
    close_c(
        &at(&l[(0, 2)]),
        0.0953101798043249,
        -4.71238898038469,
        1e-12,
        "log[0,2]",
    );
    close_c(&at(&l[(1, 1)]), -0.0953101798043249, 0.0, 1e-12, "log[1,1]");
    let s = b.matrix_sqrt().unwrap().subs(&p, &pv);
    let bs = b.subs(&p, &pv);
    for x in s.matmul(&s).unwrap().sub(&bs).unwrap().iter() {
        assert!(x.eval_complex64().unwrap().norm() < 1e-12, "S² − B: {x}");
    }
}

/// The condition number of a symbolic 3×3 matrix: its Gram matrix `AᴴA`
/// has an irreducible cubic characteristic polynomial (refused before
/// 0.44).  The singular values are real, but in Cardano's form their
/// imaginary parts cancel only exactly, so `max`/`min` of them would not
/// evaluate ("min requires real arguments"); they are taken through
/// `re(σ) = σ`.  numpy at `x = 3/7, y = −5/3`: `numpy.linalg.cond(A)` =
/// `28.196856136365184`, `svd` = `2.56273069, 1.8400006, 0.09088711`.
/// `matrix_sqrt` is refused for such a spectrum (at real values of the
/// symbols a negative eigenvalue sits on the branch cut undecidably), as it
/// was before.
#[test]
fn condition_number_of_a_symbolic_3x3_matrix() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let m = Matrix::new(vec![
        vec![x.clone(), ctx.int(1), ctx.int(0)],
        vec![ctx.int(0), y.clone(), ctx.int(1)],
        vec![ctx.int(1), ctx.int(0), ctx.int(2)],
    ])
    .unwrap();
    let (xv, yv) = (ctx.rational(3, 7), ctx.rational(-5, 3));
    let at = [(&x, &xv), (&y, &yv)];
    let c = m.condition_number().unwrap().subs_map(&at);
    close_c(&c, 28.196856136365184, 0.0, 1e-9, "cond");
    let sv = m.singular_values().unwrap();
    assert_values(
        &sorted_values(&sv, &at),
        &[(0.09088711, 0.0), (1.8400006, 0.0), (2.56273069, 0.0)],
        "singular values",
    );
    let err = m.matrix_sqrt().unwrap_err().to_string();
    assert!(err.contains("branch"), "{err}");
}

/// A 2×2 block over `ℚ(√2, i)` with the eigenvalues `1 − i` and `−√2`: the
/// quadratic formula needs `√(2 + 2√2 − (2 + 2√2)i) = 1 + √2 − i`, whose
/// modulus is `√(24 + 16√2) = 4 + 2√2`.  `sqrtdenest` found that root, but
/// the exactness check expanded to a structural `0`, which the tower's zero
/// test leaves undecided, so before 0.44 it was dropped: the eigenvalues
/// stayed `(1 − √2 − i ± √(2 + 2√2 − (2 + 2√2)i))/2`, and `matrix_sqrt` /
/// `matrix_log` of the 3×3 matrix around the block did not evaluate.
#[test]
fn eigenvalues_through_a_denested_modulus() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let block = Matrix::new(vec![
        vec![p("-2*(1 - I) - 3*sqrt(2)"), p("-3*(1 - I) - 3*sqrt(2)")],
        vec![p("2*(1 - I) + 2*sqrt(2)"), p("3*(1 - I) + 2*sqrt(2)")],
    ])
    .unwrap();
    let mut ev: Vec<String> = block
        .eigenvals()
        .unwrap()
        .iter()
        .map(|v| v.to_string())
        .collect();
    ev.sort();
    assert_eq!(ev, ["-I + 1", "-sqrt(2)"]);
    let a = Matrix::new(vec![
        vec![
            p("-2*(1 - I) - 3*sqrt(2)"),
            ctx.int(0),
            p("-3*(1 - I) - 3*sqrt(2)"),
        ],
        vec![ctx.int(-2), p("1 - I"), ctx.int(-3)],
        vec![
            p("2*(1 - I) + 2*sqrt(2)"),
            ctx.int(0),
            p("3*(1 - I) + 2*sqrt(2)"),
        ],
    ])
    .unwrap();
    let s = a.matrix_sqrt().unwrap();
    for x in s.matmul(&s).unwrap().sub(&a).unwrap().iter() {
        assert!(x.eval_complex64().unwrap().norm() < 1e-12, "S² − A: {x}");
    }
    let l = a.matrix_log().unwrap();
    for x in l.iter() {
        assert!(
            x.eval_complex64().is_ok(),
            "log entry does not evaluate: {x}"
        );
    }
}
