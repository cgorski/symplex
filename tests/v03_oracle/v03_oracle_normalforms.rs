//! SymPy oracle, 0.3 surface — integer matrix normal forms
//! (`symplex::normalforms`) and `gcd_many` / `lcm_many`.
//!
//! Fixtures: `tests/fixtures/v03_cross_validation.json`
//! (`scripts/generate_v03_fixtures.py`); the runner is shared with the 0.2
//! oracle (`v02_oracle_common/mod.rs`).
//!
//! Convention differences handled here (all documented in
//! `symplex::normalforms`):
//!
//! * **Column HNF.**  SymPy's `hermite_normal_form(M)` is column-style
//!   (`H = A·V`) and *drops* the zero columns; symplex's
//!   `column_hermite_normal_form` keeps them (leading).  The comparison
//!   removes symplex's zero columns and then requires exact equality — for a
//!   full-rank square input the two matrices are identical as-is.
//! * **Row HNF.**  SymPy has no row-style HNF; the reference is derived
//!   from SymPy's column HNF by the reverse/transpose identity
//!   `R(B) = T·P·Q·Cc(P·T·B)` (T transpose, P reverse rows, Q reverse
//!   columns), padded with zero rows.  This reproduces symplex's convention
//!   (positive pivots, entries above a pivot in `[0, pivot)`, zero rows last)
//!   exactly, so the comparison is exact equality; `H = U·A` with `|det U| = 1`
//!   and idempotence are verified in addition.
//! * **Smith normal form.**  Invariant factors are compared as a sorted
//!   multiset of positive integers; SymPy's sign/order choices are
//!   normalised in the generator.  The divisibility chain and
//!   `S = U·A·V` are verified on the symplex side.
//! * **Integer kernel.**  SymPy's `nullspace` is over ℚ; only the rank /
//!   nullity are taken from it.  Each symplex kernel vector is checked to
//!   satisfy `A k = 0`, and the basis is checked to be a ℤ-basis (the row
//!   lattice it spans has index 1 in its saturation, i.e.
//!   `lattice_determinant(basis rows) == 1`).

use super::oracle_common;

use std::sync::OnceLock;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use oracle_common::*;
use serde_json::Value;
use symplex::normalforms::{
    column_hermite_normal_form, hermite_normal_form, hermite_normal_form_with_transform,
    integer_nullspace, is_unimodular, lattice_determinant, smith_normal_form,
    smith_normal_form_with_transforms,
};
use symplex::ntheory::{gcd_many, lcm_many};
use symplex::prelude::*;

const KNOWN_BUGS: &[KnownBug] = &[];

// ── v03 fixture file ───────────────────────────────────────────────────

static V03_FILE: OnceLock<FixtureFile> = OnceLock::new();

fn v03() -> &'static FixtureFile {
    V03_FILE.get_or_init(|| {
        let json = include_str!("../fixtures/v03_cross_validation.json");
        let file: FixtureFile = serde_json::from_str(json).expect("v03 fixture JSON must parse");
        assert_eq!(
            file.fixture_count,
            file.fixtures.len(),
            "fixture_count is stale"
        );
        file
    })
}

fn run_v03(
    category: &str,
    subcategory: &str,
    process: impl Fn(&Context, &Fixture) -> Status + Send + Sync + 'static,
) {
    let file = v03();
    run_fixtures(
        &file.fixtures,
        &file.generated_by,
        category,
        Some(subcategory),
        KNOWN_BUGS,
        process,
    );
}

// ── Helpers ────────────────────────────────────────────────────────────

type Rows = Vec<Vec<BigInt>>;

fn int_rows(v: Option<&Value>) -> Result<Rows, String> {
    let arr = v.and_then(Value::as_array).ok_or("missing matrix")?;
    arr.iter()
        .map(|row| {
            row.as_array()
                .ok_or("row is not an array")?
                .iter()
                .map(|c| {
                    c.as_str()
                        .ok_or("cell is not a string")?
                        .parse::<BigInt>()
                        .map_err(|e| format!("bad integer: {e}"))
                })
                .collect()
        })
        .collect()
}

fn matrix_in(ctx: &Context, fx: &Fixture) -> Result<(Matrix, Rows), Status> {
    let rows = int_rows(fx.field("matrix")).map_err(Status::SkippedOracle)?;
    let m = Matrix::from_bigint(ctx, &rows).map_err(|e| Status::NotImplemented(format!("{e}")))?;
    Ok((m, rows))
}

fn bigint_rows_of(m: &Matrix, what: &str) -> Result<Rows, Status> {
    m.to_bigint_rows()
        .ok_or_else(|| Status::Fail(format!("{what} has a non-integer entry: {m}")))
}

fn fmt_rows(r: &Rows) -> String {
    format!(
        "[{}]",
        r.iter()
            .map(|row| format!(
                "[{}]",
                row.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn drop_zero_columns(rows: &Rows) -> Rows {
    let ncols = rows.first().map_or(0, Vec::len);
    let keep: Vec<usize> = (0..ncols)
        .filter(|&j| rows.iter().any(|r| !r[j].is_zero()))
        .collect();
    rows.iter()
        .map(|r| keep.iter().map(|&j| r[j].clone()).collect())
        .collect()
}

fn det_is_unit(m: &Matrix) -> Result<(), String> {
    let d = m.det().map_err(|e| format!("det: {e}"))?;
    match d.eval().as_i64() {
        Some(v) if v.abs() == 1 => Ok(()),
        other => Err(format!("det = {other:?}, expected ±1")),
    }
}

fn product(a: &Matrix, b: &Matrix) -> Result<Rows, Status> {
    let p = a
        .matmul(b)
        .map_err(|e| Status::Fail(format!("matmul: {e}")))?
        .eval();
    bigint_rows_of(&p, "product")
}

// ═══════════════════════════════════════════════════════════════════════

#[test]
fn normalforms_column_hnf_matches_sympy_without_zero_columns() {
    run_v03("normalforms", "hnf", |ctx, fx| {
        let (m, rows) = match matrix_in(ctx, fx) {
            Ok(v) => v,
            Err(s) => return s,
        };
        let want = match int_rows(fx.field("hnf")) {
            Ok(w) => w,
            Err(e) => return Status::SkippedOracle(e),
        };
        let h = match column_hermite_normal_form(&m) {
            Ok(h) => h,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        let hr = match bigint_rows_of(&h, "column HNF") {
            Ok(r) => r,
            Err(s) => return s,
        };
        // Same shape as the input; zero columns (if any) are the leading ones.
        if hr.len() != rows.len() || hr.first().map(Vec::len) != rows.first().map(Vec::len) {
            return Status::Fail(format!(
                "shape: symplex {}x{} input {}x{}",
                hr.len(),
                hr.first().map_or(0, Vec::len),
                rows.len(),
                rows.first().map_or(0, Vec::len)
            ));
        }
        let ncols = hr.first().map_or(0, Vec::len);
        let is_zero_col = |j: usize| hr.iter().all(|r| r[j].is_zero());
        let first_nonzero = (0..ncols).find(|&j| !is_zero_col(j)).unwrap_or(ncols);
        if (first_nonzero..ncols).any(is_zero_col) {
            return Status::Fail(format!("zero columns are not leading in {}", fmt_rows(&hr)));
        }
        let got = drop_zero_columns(&hr);
        if got != want {
            return Status::Fail(format!(
                "column HNF: symplex={} (zero columns dropped) sympy={}",
                fmt_rows(&got),
                fmt_rows(&want)
            ));
        }
        // Full-rank square input: identical as-is, and rank = #nonzero columns.
        if let Some(rank) = fx.u64("rank") {
            if got.first().map_or(0, Vec::len) as u64 != rank {
                return Status::Fail(format!(
                    "{} nonzero columns for rank {rank}",
                    got.first().map_or(0, Vec::len)
                ));
            }
            if rows.len() == ncols && rank as usize == ncols && hr != want {
                return Status::Fail("full-rank square input but HNF differs from SymPy".into());
            }
        }
        Status::Pass
    });
}

#[test]
fn normalforms_row_hnf_matches_derived_reference_and_is_unimodular() {
    run_v03("normalforms", "hnf_row", |ctx, fx| {
        let (m, rows) = match matrix_in(ctx, fx) {
            Ok(v) => v,
            Err(s) => return s,
        };
        let want = match int_rows(fx.field("hnf_row")) {
            Ok(w) => w,
            Err(e) => return Status::SkippedOracle(e),
        };
        let h = match hermite_normal_form(&m) {
            Ok(h) => h,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        let hr = match bigint_rows_of(&h, "row HNF") {
            Ok(r) => r,
            Err(s) => return s,
        };
        if hr != want {
            return Status::Fail(format!(
                "row HNF: symplex={} sympy-derived={}",
                fmt_rows(&hr),
                fmt_rows(&want)
            ));
        }
        // Nonzero rows = rank.
        if let Some(rank) = fx.u64("rank") {
            let nz = hr.iter().filter(|r| r.iter().any(|v| !v.is_zero())).count();
            if nz as u64 != rank {
                return Status::Fail(format!("{nz} nonzero rows for rank {rank}"));
            }
        }
        // H = U·A with U unimodular.
        let HermiteNormalForm { h: h2, u } = match hermite_normal_form_with_transform(&m) {
            Ok(v) => v,
            Err(e) => return Status::Fail(format!("with_transform: {e}")),
        };
        if bigint_rows_of(&h2, "H").ok().as_ref() != Some(&hr) {
            return Status::Fail("hermite_normal_form_with_transform gives a different H".into());
        }
        match product(&u, &m) {
            Ok(ua) if ua == hr => {}
            Ok(ua) => {
                return Status::Fail(format!("U·A = {} != H = {}", fmt_rows(&ua), fmt_rows(&hr)));
            }
            Err(s) => return s,
        }
        if let Err(e) = det_is_unit(&u) {
            return Status::Fail(format!("U not unimodular: {e}"));
        }
        match is_unimodular(&u) {
            Ok(true) => {}
            other => return Status::Fail(format!("is_unimodular(U) = {other:?}")),
        }
        // Idempotence.
        match hermite_normal_form(&h).map(|hh| hh.to_bigint_rows()) {
            Ok(Some(hh)) if hh == hr => {}
            Ok(Some(hh)) => {
                return Status::Fail(format!(
                    "hnf(hnf(A)) = {} != hnf(A) = {}",
                    fmt_rows(&hh),
                    fmt_rows(&hr)
                ));
            }
            Ok(None) => return Status::Fail("hnf(hnf(A)) has non-integer entries".into()),
            Err(e) => return Status::Fail(format!("hnf(hnf(A)): {e}")),
        }
        let _ = rows;
        Status::Pass
    });
}

#[test]
fn normalforms_smith_invariant_factors() {
    run_v03("normalforms", "snf", |ctx, fx| {
        let (m, rows) = match matrix_in(ctx, fx) {
            Ok(v) => v,
            Err(s) => return s,
        };
        let want: Vec<BigInt> = fx.str_list("snf_diag").iter().map(|s| bigint(s)).collect();
        let s = match smith_normal_form(&m) {
            Ok(s) => s,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        let sr = match bigint_rows_of(&s, "SNF") {
            Ok(r) => r,
            Err(st) => return st,
        };
        let (nr, nc) = (sr.len(), sr.first().map_or(0, Vec::len));
        if (nr, nc) != (rows.len(), rows.first().map_or(0, Vec::len)) {
            return Status::Fail(format!("SNF shape {nr}x{nc} differs from input"));
        }
        // Diagonal, positive, divisibility chain, zeros last.
        let mut diag: Vec<BigInt> = Vec::new();
        for (i, row) in sr.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                if i != j && !v.is_zero() {
                    return Status::Fail(format!("off-diagonal entry ({i},{j}) = {v}"));
                }
            }
            if i < nc {
                diag.push(row[i].clone());
            }
        }
        let nz: Vec<BigInt> = diag.iter().filter(|d| !d.is_zero()).cloned().collect();
        if diag.iter().skip(nz.len()).any(|d| !d.is_zero()) {
            return Status::Fail(format!("zero invariant factors not last: {diag:?}"));
        }
        for d in &nz {
            if !d.is_positive() {
                return Status::Fail(format!("non-positive invariant factor {d}"));
            }
        }
        for w in nz.windows(2) {
            if !w[1].is_multiple_of(&w[0]) {
                return Status::Fail(format!("{} does not divide {}", w[0], w[1]));
            }
        }
        let mut got_sorted = nz.clone();
        got_sorted.sort();
        if got_sorted != want {
            return Status::Fail(format!(
                "invariant factors: symplex={got_sorted:?} sympy={want:?} ({})",
                fx.str("sympy_result").unwrap_or("?")
            ));
        }
        if let Some(rank) = fx.u64("rank")
            && nz.len() as u64 != rank
        {
            return Status::Fail(format!("{} invariant factors for rank {rank}", nz.len()));
        }
        // S = U·A·V with unimodular U, V.
        let SmithNormalForm { s: s2, u, v } = match smith_normal_form_with_transforms(&m) {
            Ok(t) => t,
            Err(e) => return Status::Fail(format!("with_transforms: {e}")),
        };
        if bigint_rows_of(&s2, "S").ok().as_ref() != Some(&sr) {
            return Status::Fail("smith_normal_form_with_transforms gives a different S".into());
        }
        let ua = match u.matmul(&m) {
            Ok(p) => p.eval(),
            Err(e) => return Status::Fail(format!("U·A: {e}")),
        };
        match product(&ua, &v) {
            Ok(uav) if uav == sr => {}
            Ok(uav) => return Status::Fail(format!("U·A·V = {} != S", fmt_rows(&uav))),
            Err(st) => return st,
        }
        if let Err(e) = det_is_unit(&u) {
            return Status::Fail(format!("U not unimodular: {e}"));
        }
        if let Err(e) = det_is_unit(&v) {
            return Status::Fail(format!("V not unimodular: {e}"));
        }
        Status::Pass
    });
}

#[test]
fn normalforms_integer_nullspace_is_a_z_basis() {
    run_v03("normalforms", "nullspace_rank", |ctx, fx| {
        let (m, rows) = match matrix_in(ctx, fx) {
            Ok(v) => v,
            Err(s) => return s,
        };
        let want_rank = fx.u64("rank").unwrap_or(0) as usize;
        let want_nullity = fx.u64("nullity").unwrap_or(0) as usize;
        let rank = m.rank();
        if rank != want_rank {
            return Status::Fail(format!("rank: symplex={rank} sympy={want_rank}"));
        }
        let basis = match integer_nullspace(&m) {
            Ok(b) => b,
            Err(e) => return Status::NotImplemented(format!("{e}")),
        };
        if basis.len() != want_nullity {
            return Status::Fail(format!(
                "kernel dimension: symplex={} sympy={want_nullity}",
                basis.len()
            ));
        }
        let ncols = rows.first().map_or(0, Vec::len);
        let mut basis_rows: Rows = Vec::with_capacity(basis.len());
        for k in &basis {
            let kr = match bigint_rows_of(k, "kernel vector") {
                Ok(r) => r,
                Err(s) => return s,
            };
            if kr.len() != ncols || kr.iter().any(|r| r.len() != 1) {
                return Status::Fail(format!(
                    "kernel vector has shape {}x{}, expected {ncols}x1",
                    kr.len(),
                    kr.first().map_or(0, Vec::len)
                ));
            }
            let flat: Vec<BigInt> = kr.iter().map(|r| r[0].clone()).collect();
            if flat.iter().all(Zero::is_zero) {
                return Status::Fail("kernel basis contains the zero vector".into());
            }
            for (i, row) in rows.iter().enumerate() {
                let s: BigInt = row.iter().zip(&flat).map(|(a, b)| a * b).sum();
                if !s.is_zero() {
                    return Status::Fail(format!("A·k != 0: row {i} gives {s} for k = {flat:?}"));
                }
            }
            basis_rows.push(flat);
        }
        if !basis_rows.is_empty() {
            // A ℤ-basis of the (saturated) kernel lattice has primitive
            // maximal minors: the index of its row lattice in ℤⁿ ∩ span is 1.
            let b = match Matrix::from_bigint(ctx, &basis_rows) {
                Ok(b) => b,
                Err(e) => return Status::Fail(format!("{e}")),
            };
            match lattice_determinant(&b) {
                Ok(d) if d.is_one() => {}
                Ok(d) => {
                    return Status::Fail(format!(
                        "kernel basis is not saturated: gcd of maximal minors = {d} (basis rows {})",
                        fmt_rows(&basis_rows)
                    ));
                }
                Err(e) => {
                    return Status::Fail(format!(
                        "kernel basis rows are dependent: {e} (basis rows {})",
                        fmt_rows(&basis_rows)
                    ));
                }
            }
        }
        Status::Pass
    });
}

#[test]
fn normalforms_lattice_determinant_is_gcd_of_maximal_minors() {
    run_v03("normalforms", "lattice_det", |ctx, fx| {
        let (m, _rows) = match matrix_in(ctx, fx) {
            Ok(v) => v,
            Err(s) => return s,
        };
        let want = fx.str("lattice_det").map(bigint);
        match (lattice_determinant(&m), want) {
            (Ok(d), Some(w)) => {
                if d == w {
                    Status::Pass
                } else {
                    Status::Fail(format!("lattice determinant: symplex={d} sympy={w}"))
                }
            }
            (Ok(d), None) => Status::Fail(format!(
                "matrix does not have full row rank but symplex returned {d}"
            )),
            (Err(_), None) => Status::Pass,
            (Err(e), Some(w)) => {
                Status::Fail(format!("full-row-rank matrix rejected: {e} (sympy: {w})"))
            }
        }
    });
}

#[test]
fn normalforms_gcd_many_lcm_many() {
    run_v03("normalforms", "igcd_ilcm", |_ctx, fx| {
        let vals: Vec<BigInt> = fx.str_list("values").iter().map(|s| bigint(s)).collect();
        let (Some(want_g), Some(want_l)) = (fx.str("gcd"), fx.str("lcm")) else {
            return Status::SkippedOracle("no gcd/lcm".into());
        };
        let g = gcd_many(&vals);
        let l = lcm_many(&vals);
        if g.to_string() != want_g {
            return Status::Fail(format!("gcd_many({vals:?}) = {g}, sympy {want_g}"));
        }
        if l.to_string() != want_l {
            return Status::Fail(format!("lcm_many({vals:?}) = {l}, sympy {want_l}"));
        }
        Status::Pass
    });
}
