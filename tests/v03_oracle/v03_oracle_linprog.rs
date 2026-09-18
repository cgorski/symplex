//! SymPy oracle, 0.3 surface — exact linear programming
//! (`symplex::linprog`) against `sympy.solvers.simplex.lpmin` / `lpmax`.
//!
//! Fixtures: `tests/fixtures/v03_cross_validation.json`
//! (`scripts/generate_v03_fixtures.py`); the runner is shared with the 0.2
//! oracle (`v02_oracle_common/mod.rs`).
//!
//! Comparison rules (an LP optimum is not unique in general, so SymPy's
//! vertex is never required):
//!
//! * the **status** (optimal / infeasible / unbounded) must agree;
//! * for an optimal problem the **objective value** must agree exactly, and
//!   symplex's point must be **feasible** (every `≤` / `=` row and every
//!   bound, exact rationals) and **attain** that objective; the duals must
//!   satisfy complementary slackness `yᵢ·(aᵢ·x − bᵢ) = 0`;
//! * for an infeasible problem symplex's Farkas certificate (when present) is
//!   verified against the definition in the `linprog` module docs;
//! * `feasible_nonneg` must return a genuinely non-negative exact solution
//!   of `A x = b`, or `None` exactly when SymPy says the system is
//!   infeasible.
//!
//! Oracle note (see the generator): SymPy 1.14's `linprog` mishandles
//! non-default `bounds`, so the reference comes from `lpmin`/`lpmax` with
//! explicit relational constraints; `linprog` is cross-checked by the
//! generator for default-bound problems only.

use super::oracle_common;

use std::sync::OnceLock;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};
use oracle_common::*;
use serde_json::Value;
use symplex::linprog::{
    LpProblem, LpSolution, LpStatus, Objective, feasible_nonneg, linprog, linprog_matrix,
};
use symplex::prelude::*;

type Q = Ratio<BigInt>;

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

// ── Fixture decoding ───────────────────────────────────────────────────

fn parse_q(s: &str) -> Result<Q, String> {
    s.trim()
        .parse::<Q>()
        .map_err(|e| format!("bad rational {s:?}: {e}"))
}

fn q_vec(v: Option<&Value>) -> Result<Vec<Q>, String> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|s| parse_q(s.as_str().unwrap_or("")))
                .collect()
        })
        .unwrap_or_else(|| Ok(Vec::new()))
}

fn q_rows(v: Option<&Value>) -> Result<Vec<Vec<Q>>, String> {
    v.and_then(Value::as_array)
        .map(|a| a.iter().map(|row| q_vec(Some(row))).collect())
        .unwrap_or_else(|| Ok(Vec::new()))
}

type Bound = (Option<Q>, Option<Q>);

fn bounds_of(v: Option<&Value>) -> Result<Vec<Bound>, String> {
    let Some(arr) = v.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    arr.iter()
        .map(|b| {
            let pair = b.as_array().ok_or("bound is not a pair")?;
            let side = |i: usize| -> Result<Option<Q>, String> {
                match pair.get(i) {
                    Some(Value::Null) | None => Ok(None),
                    Some(s) => parse_q(s.as_str().unwrap_or("")).map(Some),
                }
            };
            Ok((side(0)?, side(1)?))
        })
        .collect()
}

struct Lp {
    c: Vec<Q>,
    a_ub: Vec<Vec<Q>>,
    b_ub: Vec<Q>,
    a_eq: Vec<Vec<Q>>,
    b_eq: Vec<Q>,
    bounds: Vec<Bound>,
    maximize: bool,
}

impl Lp {
    fn from_fixture(fx: &Fixture) -> Result<Lp, String> {
        Ok(Lp {
            c: q_vec(fx.field("c"))?,
            a_ub: q_rows(fx.field("A_ub"))?,
            b_ub: q_vec(fx.field("b_ub"))?,
            a_eq: q_rows(fx.field("A_eq"))?,
            b_eq: q_vec(fx.field("b_eq"))?,
            bounds: bounds_of(fx.field("bounds"))?,
            maximize: fx.str("sense") == Some("max"),
        })
    }

    fn effective_bounds(&self) -> Vec<Bound> {
        if self.bounds.is_empty() {
            vec![(Some(Q::zero()), None); self.c.len()]
        } else {
            self.bounds.clone()
        }
    }

    fn solve(&self) -> Result<LpSolution, SymplexError> {
        if self.maximize {
            let mut p = LpProblem::maximize(self.c.clone());
            for (row, rhs) in self.a_ub.iter().zip(&self.b_ub) {
                p = p.le(row.clone(), rhs.clone());
            }
            for (row, rhs) in self.a_eq.iter().zip(&self.b_eq) {
                p = p.eq(row.clone(), rhs.clone());
            }
            for (j, (lo, hi)) in self.bounds.iter().enumerate() {
                p = p.bounds(j, lo.clone(), hi.clone());
            }
            p.solve()
        } else {
            linprog(
                &self.c,
                &self.a_ub,
                &self.b_ub,
                &self.a_eq,
                &self.b_eq,
                &self.bounds,
            )
        }
    }

    /// `linprog_matrix` on the same data (default bounds only).
    fn solve_matrix(&self, ctx: &Context) -> Result<LpSolution, SymplexError> {
        let col = |v: &[Q]| -> Result<Matrix, SymplexError> {
            let rows: Vec<Vec<Q>> = v.iter().map(|q| vec![q.clone()]).collect();
            Matrix::from_ratio(ctx, &rows)
        };
        let c = col(&self.c)?;
        let a_ub = if self.a_ub.is_empty() {
            None
        } else {
            Some(Matrix::from_ratio(ctx, &self.a_ub)?)
        };
        let b_ub = if self.b_ub.is_empty() {
            None
        } else {
            Some(col(&self.b_ub)?)
        };
        let a_eq = if self.a_eq.is_empty() {
            None
        } else {
            Some(Matrix::from_ratio(ctx, &self.a_eq)?)
        };
        let b_eq = if self.b_eq.is_empty() {
            None
        } else {
            Some(col(&self.b_eq)?)
        };
        let objective = if self.maximize {
            Objective::Maximize
        } else {
            Objective::Minimize
        };
        linprog_matrix(
            objective,
            &c,
            a_ub.as_ref(),
            b_ub.as_ref(),
            a_eq.as_ref(),
            b_eq.as_ref(),
        )
    }
}

fn dot(a: &[Q], b: &[Q]) -> Q {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

fn status_name(s: &LpStatus) -> &'static str {
    match s {
        LpStatus::Optimal => "optimal",
        LpStatus::Infeasible => "infeasible",
        LpStatus::Unbounded => "unbounded",
    }
}

/// Exact feasibility of `x` for the problem, or a description of the
/// violated constraint.
fn check_feasible(lp: &Lp, x: &[Q]) -> Result<(), String> {
    if x.len() != lp.c.len() {
        return Err(format!(
            "point has {} entries for {} variables",
            x.len(),
            lp.c.len()
        ));
    }
    for (i, (row, rhs)) in lp.a_ub.iter().zip(&lp.b_ub).enumerate() {
        let lhs = dot(row, x);
        if lhs > *rhs {
            return Err(format!("row {i}: {lhs} ≤ {rhs} violated"));
        }
    }
    for (i, (row, rhs)) in lp.a_eq.iter().zip(&lp.b_eq).enumerate() {
        let lhs = dot(row, x);
        if lhs != *rhs {
            return Err(format!("equality {i}: {lhs} = {rhs} violated"));
        }
    }
    for (j, ((lo, hi), xj)) in lp.effective_bounds().iter().zip(x).enumerate() {
        if let Some(lo) = lo
            && xj < lo
        {
            return Err(format!("x[{j}] = {xj} < lower bound {lo}"));
        }
        if let Some(hi) = hi
            && xj > hi
        {
            return Err(format!("x[{j}] = {xj} > upper bound {hi}"));
        }
    }
    Ok(())
}

/// Farkas certificate check per the `linprog` module docs: with `g = Aᵀy`,
/// `Σⱼ inf{gⱼ xⱼ : lⱼ ≤ xⱼ ≤ uⱼ} > yᵀb`, `yᵢ ≥ 0` on `≤` rows, free on `=`
/// rows, every infimum finite.
fn check_farkas(lp: &Lp, y: &[Q]) -> Result<(), String> {
    let m_ub = lp.a_ub.len();
    let m = m_ub + lp.a_eq.len();
    if y.len() != m {
        return Err(format!("certificate has {} entries for {m} rows", y.len()));
    }
    for (i, yi) in y.iter().take(m_ub).enumerate() {
        if yi.is_negative() {
            return Err(format!("y[{i}] = {yi} < 0 on a ≤ row"));
        }
    }
    let n = lp.c.len();
    let rows: Vec<&Vec<Q>> = lp.a_ub.iter().chain(lp.a_eq.iter()).collect();
    let rhs: Vec<&Q> = lp.b_ub.iter().chain(lp.b_eq.iter()).collect();
    let mut g = vec![Q::zero(); n];
    for (row, yi) in rows.iter().zip(y) {
        for (gj, aij) in g.iter_mut().zip(row.iter()) {
            *gj += aij * yi;
        }
    }
    let ytb: Q = rhs.iter().zip(y).map(|(b, yi)| *b * yi).sum();
    let mut inf_sum = Q::zero();
    for (j, ((lo, hi), gj)) in lp.effective_bounds().iter().zip(&g).enumerate() {
        if gj.is_positive() {
            let lo = lo
                .as_ref()
                .ok_or_else(|| format!("g[{j}] = {gj} > 0 but x[{j}] has no lower bound"))?;
            inf_sum += gj * lo;
        } else if gj.is_negative() {
            let hi = hi
                .as_ref()
                .ok_or_else(|| format!("g[{j}] = {gj} < 0 but x[{j}] has no upper bound"))?;
            inf_sum += gj * hi;
        }
    }
    if inf_sum <= ytb {
        return Err(format!(
            "certificate does not separate: Σ inf = {inf_sum} ≤ yᵀb = {ytb} (y = {y:?})"
        ));
    }
    Ok(())
}

/// Shared verification of a solution against the oracle fields.
fn compare_solution(ctx: &Context, lp: &Lp, fx: &Fixture, sol: &LpSolution) -> Status {
    let Some(want_status) = fx.str("status") else {
        return Status::SkippedOracle("no status".into());
    };
    let got_status = status_name(&sol.status);
    if got_status != want_status {
        return Status::Fail(format!(
            "status: symplex={got_status} sympy={want_status}{}",
            fx.str("objective")
                .map(|o| format!(" (sympy objective {o}, x = {:?})", fx.str_list("x")))
                .unwrap_or_default()
        ));
    }
    match sol.status {
        LpStatus::Optimal => {
            let want_obj = match parse_q(fx.str("objective").unwrap_or("")) {
                Ok(q) => q,
                Err(e) => return Status::SkippedOracle(e),
            };
            let Some(got_obj) = sol.objective.as_ref() else {
                return Status::Fail("optimal without an objective value".into());
            };
            if *got_obj != want_obj {
                return Status::Fail(format!(
                    "objective: symplex={got_obj} sympy={want_obj} (symplex x = {:?}, sympy x = {:?})",
                    sol.x.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    fx.str_list("x")
                ));
            }
            if let Err(e) = check_feasible(lp, &sol.x) {
                return Status::Fail(format!("optimal point infeasible: {e}"));
            }
            let attained = dot(&lp.c, &sol.x);
            if attained != want_obj {
                return Status::Fail(format!(
                    "cᵀx = {attained} does not attain the reported objective {want_obj}"
                ));
            }
            // Complementary slackness on every row.
            let rows: Vec<&Vec<Q>> = lp.a_ub.iter().chain(lp.a_eq.iter()).collect();
            let rhs: Vec<&Q> = lp.b_ub.iter().chain(lp.b_eq.iter()).collect();
            if sol.duals.len() != rows.len() {
                return Status::Fail(format!(
                    "{} duals for {} constraints",
                    sol.duals.len(),
                    rows.len()
                ));
            }
            for (i, ((row, b), yi)) in rows.iter().zip(&rhs).zip(&sol.duals).enumerate() {
                let slack = dot(row, &sol.x) - *b;
                if !(yi * &slack).is_zero() {
                    return Status::Fail(format!(
                        "complementary slackness violated on row {i}: y = {yi}, slack = {slack}"
                    ));
                }
            }
            // x_ex must be the same exact point as expressions.
            let xs = sol.x_ex(ctx);
            for (j, (e, q)) in xs.iter().zip(&sol.x).enumerate() {
                if e.as_rational().as_ref() != Some(q) {
                    return Status::Fail(format!("x_ex[{j}] = {e} != {q}"));
                }
            }
            Status::Pass
        }
        LpStatus::Infeasible => {
            if !sol.x.is_empty() || sol.objective.is_some() {
                return Status::Fail("infeasible status with a point / objective".into());
            }
            match &sol.farkas {
                Some(y) => match check_farkas(lp, y) {
                    Ok(()) => Status::Pass,
                    Err(e) => Status::Fail(format!("invalid Farkas certificate: {e}")),
                },
                // Allowed only when the bounds alone are contradictory.
                None => {
                    let bounds_bad = lp
                        .effective_bounds()
                        .iter()
                        .any(|(lo, hi)| matches!((lo, hi), (Some(l), Some(h)) if l > h));
                    if bounds_bad {
                        Status::Pass
                    } else {
                        Status::Fail("infeasible but no Farkas certificate returned".into())
                    }
                }
            }
        }
        LpStatus::Unbounded => {
            if !sol.x.is_empty() || sol.objective.is_some() {
                return Status::Fail("unbounded status with a point / objective".into());
            }
            Status::Pass
        }
    }
}

fn run_lp(ctx: &Context, fx: &Fixture) -> Status {
    let lp = match Lp::from_fixture(fx) {
        Ok(lp) => lp,
        Err(e) => return Status::SkippedOracle(e),
    };
    let sol = match lp.solve() {
        Ok(s) => s,
        Err(e) => return Status::NotImplemented(format!("{e}")),
    };
    let st = compare_solution(ctx, &lp, fx, &sol);
    if !matches!(st, Status::Pass) {
        return st;
    }
    // The Matrix front end must agree with the slice front end.
    if lp.bounds.is_empty() {
        match lp.solve_matrix(ctx) {
            Ok(sol2) => {
                if sol2.status != sol.status || sol2.objective != sol.objective {
                    return Status::Fail(format!(
                        "linprog_matrix disagrees: status {} objective {:?} vs {} / {:?}",
                        status_name(&sol2.status),
                        sol2.objective,
                        status_name(&sol.status),
                        sol.objective
                    ));
                }
            }
            Err(e) => return Status::Fail(format!("linprog_matrix rejected numeric data: {e}")),
        }
    }
    Status::Pass
}

// ═══════════════════════════════════════════════════════════════════════

#[test]
fn linprog_lpmax_objective_and_feasible_optimum() {
    run_v03("linprog", "lpmax", run_lp);
}

#[test]
fn linprog_lpmin_objective_and_feasible_optimum() {
    run_v03("linprog", "lpmin", run_lp);
}

#[test]
fn linprog_infeasible_with_farkas_certificate() {
    run_v03("linprog", "infeasible", run_lp);
}

#[test]
fn linprog_unbounded_detected() {
    run_v03("linprog", "unbounded", run_lp);
}

#[test]
fn linprog_feasible_nonneg_equality_systems() {
    run_v03("linprog", "feasible_nonneg", |_ctx, fx| {
        let (a_eq, b_eq) = match (q_rows(fx.field("A_eq")), q_vec(fx.field("b_eq"))) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => return Status::SkippedOracle(e),
        };
        let Some(want) = fx.bool("feasible") else {
            return Status::SkippedOracle("no verdict".into());
        };
        match feasible_nonneg(&a_eq, &b_eq) {
            Ok(Some(x)) => {
                if !want {
                    return Status::Fail(format!(
                        "symplex found x = {:?} but SymPy says infeasible",
                        x.iter().map(ToString::to_string).collect::<Vec<_>>()
                    ));
                }
                if let Some((j, bad)) = x.iter().enumerate().find(|(_, v)| v.is_negative()) {
                    return Status::Fail(format!("x[{j}] = {bad} < 0"));
                }
                for (i, (row, rhs)) in a_eq.iter().zip(&b_eq).enumerate() {
                    let lhs = dot(row, &x);
                    if lhs != *rhs {
                        return Status::Fail(format!("equation {i}: A x = {lhs} != {rhs}"));
                    }
                }
                Status::Pass
            }
            Ok(None) => {
                if want {
                    Status::Fail(format!(
                        "symplex says infeasible, SymPy found x = {:?}",
                        fx.str_list("x")
                    ))
                } else {
                    Status::Pass
                }
            }
            Err(e) => Status::NotImplemented(format!("{e}")),
        }
    });
}
