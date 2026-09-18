//! SymPy oracle, 0.2 surface — matrices: `eigenvals` (multiset), `qr`
//! (properties + |diag R|), `cholesky`, `jordan_form` (structure +
//! reconstruction), `matrix_exp`, `pinv`, `rank`/`nullspace`, `charpoly`.

use super::v02_oracle_common;

use symplex::prelude::*;
use v02_oracle_common::*;

const KNOWN_BUGS: &[KnownBug] = &[];

fn matrix_in(ctx: &Context, fx: &Fixture) -> Result<Matrix, Status> {
    parse_matrix(ctx, fx, "matrix").map_err(Status::NotImplemented)
}

fn compare_matrix_entries(got: &Matrix, fx: &Fixture, key: &str, tol: f64) -> Status {
    let Some(want) = parse_num_matrix(fx, key) else {
        return Status::SkippedOracle(format!("no {key}"));
    };
    let g = match matrix_c64(got) {
        Ok(g) => g,
        Err(e) => return Status::NotImplemented(e),
    };
    if g.len() != want.len() || g.first().map(Vec::len) != want.first().map(Vec::len) {
        return Status::Fail(format!(
            "shape: symplex={}x{} sympy={}x{}",
            g.len(),
            g.first().map_or(0, Vec::len),
            want.len(),
            want.first().map_or(0, Vec::len)
        ));
    }
    for (i, (gr, wr)) in g.iter().zip(&want).enumerate() {
        for (j, (gv, wv)) in gr.iter().zip(wr).enumerate() {
            let Num::Finite(re, im) = wv else {
                return Status::SkippedOracle(format!("non-finite oracle entry ({i},{j})"));
            };
            if !complex_matches(*gv, (*re, *im), tol) {
                return Status::Fail(format!(
                    "entry ({i},{j}): symplex={gv:?} sympy=({re}, {im})"
                ));
            }
        }
    }
    Status::Pass
}

#[test]
fn matrix_eigenvals_multiset() {
    run_with_known_bugs("matrix", "eigenvals", KNOWN_BUGS, |ctx, fx| {
        let m = match matrix_in(ctx, fx) {
            Ok(m) => m,
            Err(s) => return s,
        };
        let want: Vec<Num> = fx
            .field("eigenvalues")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(Num::from_json).collect())
            .unwrap_or_default();
        let Some(want) = nums_to_complex(&want) else {
            return Status::SkippedOracle("non-finite eigenvalue".into());
        };
        let ev = match m.eigenvals() {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        let mut got = Vec::new();
        for e in &ev {
            match e.eval_complex64() {
                Ok(c) => got.push(c),
                Err(err) => {
                    return Status::NotImplemented(format!(
                        "eigenvalue {} not numeric: {err}",
                        truncate(e)
                    ));
                }
            }
        }
        compare_complex_multisets(got, want, 1e-8)
    });
}

#[test]
fn matrix_qr_decomposition() {
    run_with_known_bugs("matrix", "qr", KNOWN_BUGS, |ctx, fx| {
        let m = match matrix_in(ctx, fx) {
            Ok(m) => m,
            Err(s) => return s,
        };
        let (q, r) = match m.qr() {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        // Q·R = A
        let prod = q.matmul(&r).unwrap_or_else(|e| panic!("{e}"));
        let (pa, ma) = match (matrix_c64(&prod), matrix_c64(&m)) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => return Status::NotImplemented(e),
        };
        for (i, (pr, mr)) in pa.iter().zip(&ma).enumerate() {
            for (j, (pv, mv)) in pr.iter().zip(mr).enumerate() {
                if !complex_matches(*pv, *mv, 1e-9) {
                    return Status::Fail(format!("Q·R != A at ({i},{j}): {pv:?} vs {mv:?}"));
                }
            }
        }
        // Qᵀ·Q = I
        let qtq = match q.transpose().matmul(&q).map(|x| matrix_c64(&x)) {
            Ok(Ok(v)) => v,
            _ => return Status::NotImplemented("cannot evaluate QᵀQ".into()),
        };
        for (i, row) in qtq.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                if !complex_matches(*v, (want, 0.0), 1e-9) {
                    return Status::Fail(format!("Q not orthonormal: (QᵀQ)[{i},{j}] = {v:?}"));
                }
            }
        }
        // R upper triangular with |diag| matching SymPy.
        let rr = match matrix_c64(&r) {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(e),
        };
        for (i, row) in rr.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                if j < i && v.0.hypot(v.1) > 1e-9 {
                    return Status::Fail(format!("R not upper triangular at ({i},{j}) = {v:?}"));
                }
            }
        }
        let want_diag: Vec<f64> = fx
            .field("r_diag_abs")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();
        for (i, w) in want_diag.iter().enumerate() {
            let g = rr[i][i].0.hypot(rr[i][i].1);
            if !approx_eq_tol(g, *w, 1e-8) {
                return Status::Fail(format!("|R[{i},{i}]| = {g}, SymPy {w}"));
            }
        }
        Status::Pass
    });
}

#[test]
fn matrix_cholesky() {
    run_with_known_bugs("matrix", "cholesky", KNOWN_BUGS, |ctx, fx| {
        let m = match matrix_in(ctx, fx) {
            Ok(m) => m,
            Err(s) => return s,
        };
        match m.cholesky() {
            Ok(l) => compare_matrix_entries(&l, fx, "result", 1e-9),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

#[test]
fn matrix_jordan_form() {
    run_with_known_bugs("matrix", "jordan_form", KNOWN_BUGS, |ctx, fx| {
        let m = match matrix_in(ctx, fx) {
            Ok(m) => m,
            Err(s) => return s,
        };
        let (p, j) = match m.jordan_form() {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        let jn = match matrix_c64(&j) {
            Ok(v) => v,
            Err(e) => return Status::NotImplemented(e),
        };
        let n = jn.len();
        // Structure: only diagonal and super-diagonal (0/1) entries.
        let mut ones = 0usize;
        for (i, row) in jn.iter().enumerate() {
            for (k, v) in row.iter().enumerate() {
                if k == i {
                    continue;
                }
                if k == i + 1 {
                    if complex_matches(*v, (1.0, 0.0), 1e-9) {
                        ones += 1;
                    } else if v.0.hypot(v.1) > 1e-9 {
                        return Status::Fail(format!(
                            "J super-diagonal entry ({i},{k}) = {v:?} is neither 0 nor 1"
                        ));
                    }
                } else if v.0.hypot(v.1) > 1e-9 {
                    return Status::Fail(format!("J has an off-Jordan entry at ({i},{k}) = {v:?}"));
                }
            }
        }
        let want_ones = fx.u64("superdiag_ones").unwrap_or(0) as usize;
        if ones != want_ones {
            return Status::Fail(format!(
                "number of Jordan-block couplings: symplex={ones} sympy={want_ones} (J = {:?})",
                jn
            ));
        }
        let diag: Vec<(f64, f64)> = (0..n).map(|i| jn[i][i]).collect();
        let want: Vec<Num> = fx
            .field("jordan_diag")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(Num::from_json).collect())
            .unwrap_or_default();
        let Some(want) = nums_to_complex(&want) else {
            return Status::SkippedOracle("non-finite oracle eigenvalue".into());
        };
        if let Status::Fail(r) = compare_complex_multisets(diag, want, 1e-8) {
            return Status::Fail(format!("Jordan diagonal: {r}"));
        }
        // Reconstruction P·J·P⁻¹ = A.
        let pinv = match p.inv() {
            Ok(v) => v,
            Err(e) => return Status::Fail(format!("P is singular: {e}")),
        };
        let recon = p
            .matmul(&j)
            .and_then(|pj| pj.matmul(&pinv))
            .unwrap_or_else(|e| panic!("{e}"));
        let (ra, ma) = match (matrix_c64(&recon), matrix_c64(&m)) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => return Status::NotImplemented(e),
        };
        for (i, (rr, mr)) in ra.iter().zip(&ma).enumerate() {
            for (k, (rv, mv)) in rr.iter().zip(mr).enumerate() {
                if !complex_matches(*rv, *mv, 1e-8) {
                    return Status::Fail(format!("P·J·P⁻¹ != A at ({i},{k}): {rv:?} vs {mv:?}"));
                }
            }
        }
        Status::Pass
    });
}

#[test]
fn matrix_exponential() {
    run_with_known_bugs("matrix", "exp", KNOWN_BUGS, |ctx, fx| {
        let m = match matrix_in(ctx, fx) {
            Ok(m) => m,
            Err(s) => return s,
        };
        match m.matrix_exp() {
            Ok(e) => compare_matrix_entries(&e, fx, "result", 1e-9),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

#[test]
fn matrix_pseudoinverse() {
    run_with_known_bugs("matrix", "pinv", KNOWN_BUGS, |ctx, fx| {
        let m = match matrix_in(ctx, fx) {
            Ok(m) => m,
            Err(s) => return s,
        };
        match m.pinv() {
            Ok(p) => compare_matrix_entries(&p, fx, "result", 1e-9),
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}

#[test]
fn matrix_rank_and_nullspace() {
    run_with_known_bugs("matrix", "rank_nullspace", KNOWN_BUGS, |ctx, fx| {
        let m = match matrix_in(ctx, fx) {
            Ok(m) => m,
            Err(s) => return s,
        };
        let want_rank = fx.u64("rank").unwrap_or(0) as usize;
        let want_nullity = fx.u64("nullity").unwrap_or(0) as usize;
        let rank = m.rank();
        if rank != want_rank {
            return Status::Fail(format!("rank: symplex={rank} sympy={want_rank}"));
        }
        let ns = m.nullspace();
        if ns.len() != want_nullity {
            return Status::Fail(format!(
                "nullity: symplex={} sympy={want_nullity}",
                ns.len()
            ));
        }
        for v in &ns {
            let av = m.matmul(v).unwrap_or_else(|e| panic!("{e}"));
            match matrix_c64(&av) {
                Ok(rows) => {
                    if rows.iter().flatten().any(|c| c.0.hypot(c.1) > 1e-9) {
                        return Status::Fail(format!(
                            "A·v != 0 for nullspace vector {:?}",
                            matrix_c64(v)
                        ));
                    }
                }
                Err(e) => return Status::NotImplemented(e),
            }
            if matrix_c64(v).map(|r| r.iter().flatten().all(|c| c.0.hypot(c.1) < 1e-12)) == Ok(true)
            {
                return Status::Fail("nullspace contains the zero vector".into());
            }
        }
        Status::Pass
    });
}

#[test]
fn matrix_characteristic_polynomial() {
    run_with_known_bugs("matrix", "charpoly", KNOWN_BUGS, |ctx, fx| {
        let m = match matrix_in(ctx, fx) {
            Ok(m) => m,
            Err(s) => return s,
        };
        // SymPy: det(λI − A), highest degree first.
        let want: Vec<Num> = fx
            .field("coeffs")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(Num::from_json).collect())
            .unwrap_or_default();
        let Some(want) = nums_to_complex(&want) else {
            return Status::SkippedOracle("non-finite coefficient".into());
        };
        // symplex: det(A − λI), lowest degree first → reverse, scale by (−1)ⁿ.
        let coeffs = match m.char_poly_coeffs() {
            Ok(c) => c,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        let n = m.nrows();
        let sign = if n % 2 == 1 { -1.0 } else { 1.0 };
        let mut got = Vec::new();
        for c in coeffs.iter().rev() {
            match c.eval_complex64() {
                Ok(v) => got.push((sign * v.0, sign * v.1)),
                Err(e) => {
                    return Status::NotImplemented(format!(
                        "coefficient {} not numeric: {e}",
                        truncate(c)
                    ));
                }
            }
        }
        if got.len() != want.len() {
            return Status::Fail(format!(
                "coefficient count: symplex={} sympy={}",
                got.len(),
                want.len()
            ));
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            if !complex_matches(*g, *w, 1e-9) {
                return Status::Fail(format!(
                    "coefficient {i}: symplex={g:?} sympy={w:?} (all: {got:?} vs {want:?})"
                ));
            }
        }
        Status::Pass
    });
}
