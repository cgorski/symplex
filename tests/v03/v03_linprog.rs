//! symplex 0.3 — exact linear programming: two-phase simplex over ℚ,
//! shadow prices, Farkas certificates, wrappers.
//!
//! Every `Optimal` result is checked against the full KKT conditions
//! exactly (primal feasibility, dual signs, complementary slackness,
//! reduced-cost signs at bounds, strong duality); every `Infeasible` result
//! has its Farkas inequality verified; every `Unbounded` result is shown to
//! be feasible and to improve without limit as a box bound grows.

use std::time::{Duration, Instant};

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};
use proptest::prelude::*;
use symplex::linprog::{
    Budget, Feasibility, LpProblem, LpSolution, LpStatus, Objective, Q, feasible_nonneg,
    feasible_nonneg_certified, linprog, linprog_matrix, nonneg_combination, q, qi,
};
use symplex::matrix::Matrix;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Test-side problem description + exact certificate checking
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rel {
    Le,
    Eq,
    Ge,
}

#[derive(Clone, Debug)]
struct Spec {
    maximize: bool,
    c: Vec<Q>,
    rows: Vec<(Vec<Q>, Rel, Q)>,
    bounds: Vec<(Option<Q>, Option<Q>)>,
}

fn dot(a: &[Q], b: &[Q]) -> Q {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Spec {
    fn min(c: Vec<Q>) -> Self {
        let n = c.len();
        Spec {
            maximize: false,
            c,
            rows: Vec::new(),
            bounds: vec![(Some(Q::zero()), None); n],
        }
    }

    fn max(c: Vec<Q>) -> Self {
        Spec {
            maximize: true,
            ..Spec::min(c)
        }
    }

    fn le(mut self, row: Vec<Q>, rhs: Q) -> Self {
        self.rows.push((row, Rel::Le, rhs));
        self
    }

    fn ge(mut self, row: Vec<Q>, rhs: Q) -> Self {
        self.rows.push((row, Rel::Ge, rhs));
        self
    }

    fn eq(mut self, row: Vec<Q>, rhs: Q) -> Self {
        self.rows.push((row, Rel::Eq, rhs));
        self
    }

    fn bounds(mut self, j: usize, lo: Option<Q>, hi: Option<Q>) -> Self {
        self.bounds[j] = (lo, hi);
        self
    }

    fn free(self, j: usize) -> Self {
        self.bounds(j, None, None)
    }

    fn build(&self) -> LpProblem {
        let mut p = if self.maximize {
            LpProblem::maximize(self.c.clone())
        } else {
            LpProblem::minimize(self.c.clone())
        };
        for (row, rel, rhs) in &self.rows {
            p = match rel {
                Rel::Le => p.le(row.clone(), rhs.clone()),
                Rel::Eq => p.eq(row.clone(), rhs.clone()),
                Rel::Ge => p.ge(row.clone(), rhs.clone()),
            };
        }
        for (j, (lo, hi)) in self.bounds.iter().enumerate() {
            p = p.bounds(
                j,
                Bounds {
                    lower: lo.clone(),
                    upper: hi.clone(),
                },
            );
        }
        p
    }

    fn solve(&self) -> LpSolution {
        self.build().solve().expect("well-formed LP")
    }

    /// Solve and verify whichever certificate the status carries.
    fn solve_checked(&self) -> LpSolution {
        let sol = self.solve();
        match sol.status {
            LpStatus::Optimal => self.check_optimal(&sol),
            LpStatus::Infeasible => self.check_farkas(sol.farkas.as_deref().expect("certificate")),
            LpStatus::Unbounded => self.check_unbounded(),
            LpStatus::BudgetExhausted => panic!("no budget was set"),
        }
        sol
    }

    fn is_feasible(&self, x: &[Q]) -> bool {
        if x.len() != self.c.len() {
            return false;
        }
        for (j, (lo, hi)) in self.bounds.iter().enumerate() {
            if lo.as_ref().is_some_and(|l| &x[j] < l) || hi.as_ref().is_some_and(|h| &x[j] > h) {
                return false;
            }
        }
        self.rows.iter().all(|(row, rel, rhs)| {
            let lhs = dot(row, x);
            match rel {
                Rel::Le => lhs <= *rhs,
                Rel::Eq => lhs == *rhs,
                Rel::Ge => lhs >= *rhs,
            }
        })
    }

    /// Full KKT check of an `Optimal` solution.
    fn check_optimal(&self, sol: &LpSolution) {
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!(sol.is_optimal());
        let x = &sol.x;
        let y = &sol.duals;
        assert_eq!(x.len(), self.c.len(), "x length");
        assert_eq!(y.len(), self.rows.len(), "duals length");
        assert!(
            sol.farkas.is_none(),
            "no Farkas certificate for an optimal LP"
        );
        assert!(self.is_feasible(x), "primal infeasible: x = {x:?}");
        assert_eq!(
            sol.objective.as_ref(),
            Some(&dot(&self.c, x)),
            "objective = cᵀx"
        );

        // Work in minimisation form: c' = dir·c, y' = dir·y.
        let dir = if self.maximize { qi(-1) } else { qi(1) };
        let cp: Vec<Q> = self.c.iter().map(|v| v * &dir).collect();
        let yp: Vec<Q> = y.iter().map(|v| v * &dir).collect();

        let mut rhs_total = Q::zero();
        for (i, (row, rel, rhs)) in self.rows.iter().enumerate() {
            match rel {
                Rel::Le => assert!(
                    !yp[i].is_positive(),
                    "y'[{i}] = {} must be ≤ 0 on a ≤ row",
                    yp[i]
                ),
                Rel::Ge => assert!(
                    !yp[i].is_negative(),
                    "y'[{i}] = {} must be ≥ 0 on a ≥ row",
                    yp[i]
                ),
                Rel::Eq => {}
            }
            let slack = &dot(row, x) - rhs;
            assert!(
                (&y[i] * &slack).is_zero(),
                "complementary slackness violated on row {i}: y = {}, slack = {slack}",
                y[i]
            );
            rhs_total += &y[i] * rhs;
        }

        // Reduced costs r' = c' − Aᵀy'.
        let n = self.c.len();
        let mut bound_total = Q::zero();
        for j in 0..n {
            let aty: Q = self
                .rows
                .iter()
                .zip(&yp)
                .map(|((row, _, _), yi)| &row[j] * yi)
                .sum();
            let r = &cp[j] - &aty;
            let (lo, hi) = &self.bounds[j];
            let at_lo = lo.as_ref() == Some(&x[j]);
            let at_hi = hi.as_ref() == Some(&x[j]);
            if at_lo && at_hi {
                // Fixed variable: any multiplier.
            } else if at_lo {
                assert!(
                    !r.is_negative(),
                    "r'[{j}] = {r} must be ≥ 0 at a lower bound"
                );
            } else if at_hi {
                assert!(
                    !r.is_positive(),
                    "r'[{j}] = {r} must be ≤ 0 at an upper bound"
                );
            } else {
                assert!(
                    r.is_zero(),
                    "r'[{j}] = {r} must vanish strictly inside the bounds"
                );
            }
            // Back to the caller's sign for the strong-duality identity.
            bound_total += &(&r * &dir) * &x[j];
        }
        let primal = dot(&self.c, x);
        assert_eq!(
            primal,
            &rhs_total + &bound_total,
            "strong duality: cᵀx = yᵀb + Σ rⱼxⱼ"
        );
    }

    /// Verify the documented Farkas inequality.
    fn check_farkas(&self, y: &[Q]) {
        assert_eq!(y.len(), self.rows.len(), "farkas length");
        let mut ytb = Q::zero();
        for (i, (_, rel, rhs)) in self.rows.iter().enumerate() {
            match rel {
                Rel::Le => assert!(
                    !y[i].is_negative(),
                    "farkas y[{i}] = {} must be ≥ 0 on ≤",
                    y[i]
                ),
                Rel::Ge => assert!(
                    !y[i].is_positive(),
                    "farkas y[{i}] = {} must be ≤ 0 on ≥",
                    y[i]
                ),
                Rel::Eq => {}
            }
            ytb += &y[i] * rhs;
        }
        let n = self.c.len();
        let mut inf_total = Q::zero();
        for j in 0..n {
            let g: Q = self
                .rows
                .iter()
                .zip(y)
                .map(|((row, _, _), yi)| &row[j] * yi)
                .sum();
            let (lo, hi) = &self.bounds[j];
            if g.is_positive() {
                let l = lo
                    .as_ref()
                    .unwrap_or_else(|| panic!("g[{j}] > 0 needs a finite lower bound"));
                inf_total += &g * l;
            } else if g.is_negative() {
                let h = hi
                    .as_ref()
                    .unwrap_or_else(|| panic!("g[{j}] < 0 needs a finite upper bound"));
                inf_total += &g * h;
            }
        }
        assert!(
            inf_total > ytb,
            "Farkas inequality fails: {inf_total} ≯ {ytb}"
        );
    }

    /// An unbounded LP must be feasible, and boxing the variables at ±M
    /// must give optima that keep improving as M grows.
    fn check_unbounded(&self) {
        let feas = Spec {
            maximize: false,
            c: vec![Q::zero(); self.c.len()],
            rows: self.rows.clone(),
            bounds: self.bounds.clone(),
        }
        .solve();
        assert_eq!(
            feas.status,
            LpStatus::Optimal,
            "unbounded LP must be feasible"
        );
        // Box around a point known to be feasible: the feasible region need
        // not meet a fixed ±50 box (one random case forces x₃ ≥ 52), so the
        // small box is the feasibility LP's own vertex, rounded outwards.
        let reach: i64 = feas
            .x
            .iter()
            .map(|v| (v.abs().ceil().to_integer()).try_into().unwrap_or(i64::MAX))
            .max()
            .unwrap_or(0);
        let base = reach.max(50);
        let boxed = |m: i64| {
            let mut s = self.clone();
            for (lo, hi) in s.bounds.iter_mut() {
                if lo.is_none() {
                    *lo = Some(qi(-m));
                }
                if hi.is_none() {
                    *hi = Some(qi(m));
                }
            }
            let sol = s.solve();
            assert_eq!(
                sol.status,
                LpStatus::Optimal,
                "boxed LP (M = {m}) must be optimal"
            );
            s.check_optimal(&sol);
            sol.objective.unwrap()
        };
        let (small, large) = (boxed(base), boxed(base.saturating_mul(100)));
        if self.maximize {
            assert!(
                large > small,
                "objective must grow with the box: {small} vs {large}"
            );
        } else {
            assert!(
                large < small,
                "objective must fall with the box: {small} vs {large}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Textbook problems with known exact optima
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn textbook_max_3x_2y() {
    // max 3x + 2y  s.t.  x + y ≤ 4,  x + 3y ≤ 6,  x, y ≥ 0  →  (4, 0), 12
    let s = Spec::max(vec![qi(3), qi(2)])
        .le(vec![qi(1), qi(1)], qi(4))
        .le(vec![qi(1), qi(3)], qi(6));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(4), qi(0)]);
    assert_eq!(sol.objective, Some(qi(12)));
    assert_eq!(sol.duals, vec![qi(3), qi(0)]);
}

#[test]
fn textbook_min_fractional_optimum() {
    // min x + y  s.t.  x + 2y ≥ 1,  3x + y ≥ 1  →  (1/5, 2/5), 3/5
    // (vertices: (1,0) → 1, (0,1) → 1, intersection → 3/5)
    let s = Spec::min(vec![qi(1), qi(1)])
        .ge(vec![qi(1), qi(2)], qi(1))
        .ge(vec![qi(3), qi(1)], qi(1));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![q(1, 5), q(2, 5)]);
    assert_eq!(sol.objective, Some(q(3, 5)));
    // Both constraints binding: y₁ + 3y₂ = 1, 2y₁ + y₂ = 1 → (2/5, 1/5).
    assert_eq!(sol.duals, vec![q(2, 5), q(1, 5)]);
}

#[test]
fn reddy_mikks_paint_problem() {
    // max 5x + 4y  s.t.  6x + 4y ≤ 24,  x + 2y ≤ 6,  −x + y ≤ 1,  y ≤ 2  →  (3, 3/2), 21
    let s = Spec::max(vec![qi(5), qi(4)])
        .le(vec![qi(6), qi(4)], qi(24))
        .le(vec![qi(1), qi(2)], qi(6))
        .le(vec![qi(-1), qi(1)], qi(1))
        .le(vec![qi(0), qi(1)], qi(2));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(3), q(3, 2)]);
    assert_eq!(sol.objective, Some(qi(21)));
    assert_eq!(sol.duals, vec![q(3, 4), q(1, 2), qi(0), qi(0)]);
}

#[test]
fn fractional_data_fractional_optimum() {
    // max x + y  s.t.  x/4 + y/5 ≤ 1,  x/7 + y/2 ≤ 1
    // Vertices: (4, 0) → 4, (0, 2) → 2, intersection (28/9, 10/9) → 38/9.
    let s = Spec::max(vec![qi(1), qi(1)])
        .le(vec![q(1, 4), q(1, 5)], qi(1))
        .le(vec![q(1, 7), q(1, 2)], qi(1));
    let sol = s.solve_checked();
    let (x, y) = (&sol.x[0], &sol.x[1]);
    assert_eq!(x / qi(4) + y / qi(5), qi(1));
    assert_eq!(x / qi(7) + y / qi(2), qi(1));
    assert_eq!(*x, q(28, 9));
    assert_eq!(*y, q(10, 9));
    assert_eq!(sol.objective, Some(q(38, 9)));
}

#[test]
fn fractional_data_vertex_on_axis() {
    // Same polytope, objective x/2 + y/3: the axis vertex (4, 0) wins with 2.
    let s = Spec::max(vec![q(1, 2), q(1, 3)])
        .le(vec![q(1, 4), q(1, 5)], qi(1))
        .le(vec![q(1, 7), q(1, 2)], qi(1));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(4), qi(0)]);
    assert_eq!(sol.objective, Some(qi(2)));
    assert_eq!(sol.duals, vec![qi(2), qi(0)]);
}

#[test]
fn diet_style_min_cost() {
    // min 2x + 3y + z  s.t.  x + y + z ≥ 10,  x − y ≥ 0,  z ≤ 4 → z = 4, x = 6, y = 0: 16
    let s = Spec::min(vec![qi(2), qi(3), qi(1)])
        .ge(vec![qi(1), qi(1), qi(1)], qi(10))
        .ge(vec![qi(1), qi(-1), qi(0)], qi(0))
        .le(vec![qi(0), qi(0), qi(1)], qi(4));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(6), qi(0), qi(4)]);
    assert_eq!(sol.objective, Some(qi(16)));
}

#[test]
fn equality_constrained_transport_like() {
    // Supplies 5 and 7 to demands 4 and 8; costs [[1,3],[2,1]].
    let s = Spec::min(vec![qi(1), qi(3), qi(2), qi(1)])
        .eq(vec![qi(1), qi(1), qi(0), qi(0)], qi(5))
        .eq(vec![qi(0), qi(0), qi(1), qi(1)], qi(7))
        .eq(vec![qi(1), qi(0), qi(1), qi(0)], qi(4))
        .eq(vec![qi(0), qi(1), qi(0), qi(1)], qi(8));
    let sol = s.solve_checked();
    // x11 = 4, x12 = 1, x21 = 0, x22 = 7 → 4 + 3 + 0 + 7 = 14
    assert_eq!(sol.x, vec![qi(4), qi(1), qi(0), qi(7)]);
    assert_eq!(sol.objective, Some(qi(14)));
}

// ═══════════════════════════════════════════════════════════════════════════
// Degeneracy (Bland's rule must terminate)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn beale_cycling_example_terminates_with_correct_optimum() {
    // Beale's example cycles under the naive most-negative rule.
    // min −3/4 x₄ + 20 x₅ − 1/2 x₆ + 6 x₇
    // x₁ + 1/4 x₄ − 8 x₅ − x₆ + 9 x₇ = 0
    // x₂ + 1/2 x₄ − 12 x₅ − 1/2 x₆ + 3 x₇ = 0
    // x₃ + x₆ = 1
    let s = Spec::min(vec![qi(0), qi(0), qi(0), q(-3, 4), qi(20), q(-1, 2), qi(6)])
        .eq(
            vec![qi(1), qi(0), qi(0), q(1, 4), qi(-8), qi(-1), qi(9)],
            qi(0),
        )
        .eq(
            vec![qi(0), qi(1), qi(0), q(1, 2), qi(-12), q(-1, 2), qi(3)],
            qi(0),
        )
        .eq(vec![qi(0), qi(0), qi(1), qi(0), qi(0), qi(1), qi(0)], qi(1));
    let t0 = std::time::Instant::now();
    let sol = s.solve_checked();
    assert!(t0.elapsed() < std::time::Duration::from_secs(2));
    assert_eq!(sol.objective, Some(q(-5, 4)));
    assert_eq!(
        sol.x,
        vec![q(3, 4), qi(0), qi(0), qi(1), qi(0), qi(1), qi(0)]
    );
}

#[test]
fn degenerate_vertex_cube_corner() {
    // max x + y + z with x ≤ 1, y ≤ 1, z ≤ 1, x + y + z ≤ 3 (the corner is
    // over-determined), plus redundant x + y ≤ 2.
    let s = Spec::max(vec![qi(1), qi(1), qi(1)])
        .le(vec![qi(1), qi(0), qi(0)], qi(1))
        .le(vec![qi(0), qi(1), qi(0)], qi(1))
        .le(vec![qi(0), qi(0), qi(1)], qi(1))
        .le(vec![qi(1), qi(1), qi(1)], qi(3))
        .le(vec![qi(1), qi(1), qi(0)], qi(2));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(1), qi(1), qi(1)]);
    assert_eq!(sol.objective, Some(qi(3)));
}

#[test]
fn degenerate_origin_start() {
    // Constraints all pass through the origin: the initial vertex is degenerate.
    let s = Spec::max(vec![qi(1), qi(2)])
        .le(vec![qi(1), qi(-1)], qi(0))
        .le(vec![qi(-1), qi(1)], qi(0))
        .le(vec![qi(1), qi(1)], qi(4));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(2), qi(2)]);
    assert_eq!(sol.objective, Some(qi(6)));
}

#[test]
fn many_redundant_rows_terminate() {
    let mut s = Spec::max(vec![qi(1), qi(1)]);
    for k in 1..=12 {
        s = s.le(vec![qi(k), qi(k)], qi(2 * k)); // all the same halfplane x + y ≤ 2
    }
    let sol = s.solve_checked();
    assert_eq!(sol.objective, Some(qi(2)));
}

// ═══════════════════════════════════════════════════════════════════════════
// Infeasible problems and their Farkas certificates
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infeasible_two_halfplanes() {
    let s = Spec::min(vec![qi(1), qi(1)])
        .le(vec![qi(1), qi(1)], qi(1))
        .ge(vec![qi(1), qi(1)], qi(2));
    let sol = s.solve_checked();
    assert_eq!(sol.status, LpStatus::Infeasible);
    assert!(sol.x.is_empty());
    assert!(sol.objective.is_none());
    assert!(sol.duals.is_empty());
    let y = sol.farkas.unwrap();
    // Explicit textbook check: y ≥ 0 on ≤, y ≤ 0 on ≥, Aᵀy ≥ 0, yᵀb < 0.
    assert!(!y[0].is_negative() && !y[1].is_positive());
    assert!(!(&y[0] + &y[1]).is_negative());
    assert!((&y[0] + &(&y[1] * qi(2))).is_negative());
}

#[test]
fn infeasible_against_default_lower_bound() {
    // x ≤ −1 with x ≥ 0.
    let s = Spec::min(vec![qi(1)]).le(vec![qi(1)], qi(-1));
    let sol = s.solve_checked();
    assert_eq!(sol.status, LpStatus::Infeasible);
    let y = sol.farkas.unwrap();
    assert!(y[0].is_positive());
}

#[test]
fn infeasible_equalities() {
    let s = Spec::min(vec![qi(1), qi(1)])
        .eq(vec![qi(1), qi(1)], qi(1))
        .eq(vec![qi(2), qi(2)], qi(3));
    let sol = s.solve_checked();
    assert_eq!(sol.status, LpStatus::Infeasible);
    let y = sol.farkas.unwrap();
    // Aᵀy = (y₀ + 2y₁, y₀ + 2y₁) must be ≥ 0 and yᵀb = y₀ + 3y₁ < 0.
    assert!(!(&y[0] + &(&y[1] * qi(2))).is_negative());
    assert!((&y[0] + &(&y[1] * qi(3))).is_negative());
}

#[test]
fn infeasible_with_upper_bounds_uses_box_infimum() {
    // x + y ≥ 5 with 0 ≤ x ≤ 2, 0 ≤ y ≤ 2.
    let s = Spec::min(vec![qi(1), qi(1)])
        .ge(vec![qi(1), qi(1)], qi(5))
        .bounds(0, Some(qi(0)), Some(qi(2)))
        .bounds(1, Some(qi(0)), Some(qi(2)));
    let sol = s.solve_checked();
    assert_eq!(sol.status, LpStatus::Infeasible);
}

#[test]
fn infeasible_free_variable() {
    let s = Spec::min(vec![qi(1)])
        .le(vec![qi(1)], qi(0))
        .ge(vec![qi(1)], qi(1))
        .free(0);
    let sol = s.solve_checked();
    assert_eq!(sol.status, LpStatus::Infeasible);
    let y = sol.farkas.unwrap();
    // Free variable ⇒ Aᵀy = 0 exactly.
    assert!((&y[0] + &y[1]).is_zero());
}

#[test]
fn infeasible_negative_rhs_rows() {
    // −x − y ≤ −3 (i.e. x + y ≥ 3) and x + y ≤ 2.
    let s = Spec::max(vec![qi(1), qi(0)])
        .le(vec![qi(-1), qi(-1)], qi(-3))
        .le(vec![qi(1), qi(1)], qi(2));
    let sol = s.solve_checked();
    assert_eq!(sol.status, LpStatus::Infeasible);
}

#[test]
fn infeasible_mixed_three_rows() {
    let s = Spec::min(vec![qi(1), qi(2), qi(3)])
        .ge(vec![qi(1), qi(1), qi(0)], qi(4))
        .le(vec![qi(1), qi(0), qi(1)], qi(1))
        .le(vec![qi(0), qi(1), qi(-1)], qi(1));
    let sol = s.solve_checked();
    assert_eq!(sol.status, LpStatus::Infeasible);
}

#[test]
fn inconsistent_bounds_alone_give_infeasible_without_certificate() {
    let s = Spec::min(vec![qi(1)]).bounds(0, Some(qi(3)), Some(qi(2)));
    let sol = s.solve();
    assert_eq!(sol.status, LpStatus::Infeasible);
    assert!(sol.farkas.is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// Unbounded problems
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn unbounded_ray() {
    let s = Spec::max(vec![qi(1), qi(1)]).le(vec![qi(1), qi(-1)], qi(1));
    let sol = s.solve_checked();
    assert_eq!(sol.status, LpStatus::Unbounded);
    assert!(sol.x.is_empty() && sol.objective.is_none());
}

#[test]
fn unbounded_free_variable_no_constraints() {
    let s = Spec::min(vec![qi(1)]).free(0);
    assert_eq!(s.solve_checked().status, LpStatus::Unbounded);
}

#[test]
fn unbounded_only_after_phase_one() {
    // Feasible region needs phase 1 (≥ row) and is unbounded in y.
    let s = Spec::max(vec![qi(0), qi(1)])
        .ge(vec![qi(1), qi(1)], qi(2))
        .le(vec![qi(1), qi(0)], qi(5));
    assert_eq!(s.solve_checked().status, LpStatus::Unbounded);
}

#[test]
fn unbounded_below_with_upper_bound_only() {
    // min x with x ≤ 5 and no lower bound.
    let s = Spec::min(vec![qi(1)]).bounds(0, None, Some(qi(5)));
    assert_eq!(s.solve_checked().status, LpStatus::Unbounded);
}

// ═══════════════════════════════════════════════════════════════════════════
// Bounds and free variables
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn free_variable_negative_optimum() {
    let s = Spec::min(vec![qi(1)]).ge(vec![qi(1)], qi(-3)).free(0);
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(-3)]);
    assert_eq!(sol.duals, vec![qi(1)]);
}

#[test]
fn free_variables_fractional() {
    // min 2x + 3y  s.t.  x − y = −1/2,  x + y ≥ 0,  x, y free  →  (−1/4, 1/4), 1/4
    let s = Spec::min(vec![qi(2), qi(3)])
        .eq(vec![qi(1), qi(-1)], q(-1, 2))
        .ge(vec![qi(1), qi(1)], qi(0))
        .free(0)
        .free(1);
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![q(-1, 4), q(1, 4)]);
    assert_eq!(sol.objective, Some(q(1, 4)));
}

#[test]
fn free_variable_reduced_cost_is_zero() {
    let s = Spec::max(vec![qi(1), qi(-1)])
        .le(vec![qi(1), qi(0)], qi(3))
        .ge(vec![qi(0), qi(1)], qi(-2))
        .free(0)
        .free(1);
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(3), qi(-2)]);
    assert_eq!(sol.objective, Some(qi(5)));
}

#[test]
fn nonzero_lower_bound_shift() {
    let s = Spec::min(vec![qi(1), qi(1)])
        .ge(vec![qi(1), qi(1)], qi(1))
        .bounds(0, Some(qi(2)), None)
        .bounds(1, Some(qi(-1)), None);
    let sol = s.solve_checked();
    // x ≥ 2, y ≥ −1: min x + y = 1 at the constraint, e.g. (2, −1).
    assert_eq!(sol.objective, Some(qi(1)));
    assert_eq!(sol.x, vec![qi(2), qi(-1)]);
}

#[test]
fn upper_bound_only_is_mirrored() {
    // x ≤ 1 with no lower bound, 0 ≤ y ≤ 10:  max 2x + y  s.t.  x + y ≤ 4
    // On the binding line 2x + y = x + 4 grows with x → x = 1, y = 3.
    let s = Spec::max(vec![qi(2), qi(1)])
        .le(vec![qi(1), qi(1)], qi(4))
        .bounds(0, None, Some(qi(1)))
        .bounds(1, Some(qi(0)), Some(qi(10)));
    let sol = s.solve_checked();
    assert_eq!(sol.objective, Some(qi(5)));
    assert_eq!(sol.x, vec![qi(1), qi(3)]);
}

#[test]
fn upper_bound_only_negative_optimum() {
    // min x + 2y with x ≤ 1 (no lower bound), y ≥ 0, and x ≥ −5 as a row.
    let s = Spec::min(vec![qi(1), qi(2)])
        .ge(vec![qi(1), qi(0)], qi(-5))
        .bounds(0, None, Some(qi(1)));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(-5), qi(0)]);
    assert_eq!(sol.objective, Some(qi(-5)));
}

#[test]
fn two_sided_bounds_active_at_upper() {
    let s = Spec::max(vec![qi(3), qi(1)])
        .le(vec![qi(1), qi(1)], qi(10))
        .bounds(0, Some(qi(0)), Some(qi(2)))
        .bounds(1, Some(qi(0)), Some(qi(5)));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(2), qi(5)]);
    assert_eq!(sol.objective, Some(qi(11)));
    // Row not binding (7 < 10): dual 0.
    assert_eq!(sol.duals, vec![qi(0)]);
}

#[test]
fn fixed_variable_lo_equals_hi() {
    let s = Spec::min(vec![qi(1), qi(1)])
        .ge(vec![qi(1), qi(1)], qi(3))
        .bounds(0, Some(qi(2)), Some(qi(2)));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(2), qi(1)]);
    assert_eq!(sol.objective, Some(qi(3)));
}

#[test]
fn bounds_only_problem_no_constraints() {
    let s = Spec::min(vec![qi(2), qi(-3)])
        .bounds(0, Some(qi(-1)), Some(qi(4)))
        .bounds(1, Some(qi(-2)), Some(qi(5)));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(-1), qi(5)]);
    assert_eq!(sol.objective, Some(qi(-17)));
    assert!(sol.duals.is_empty());
}

#[test]
fn negative_rhs_equality_dual_sign() {
    // min x + y s.t. −x − y = −3  →  3; ∂z/∂b = −1.
    let s = Spec::min(vec![qi(1), qi(1)]).eq(vec![qi(-1), qi(-1)], qi(-3));
    let sol = s.solve_checked();
    assert_eq!(sol.objective, Some(qi(3)));
    assert_eq!(sol.duals, vec![qi(-1)]);
}

#[test]
fn redundant_equalities_are_handled() {
    let s = Spec::min(vec![qi(1), qi(2)])
        .eq(vec![qi(1), qi(1)], qi(1))
        .eq(vec![qi(2), qi(2)], qi(2))
        .eq(vec![qi(-3), qi(-3)], qi(-3));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(1), qi(0)]);
    assert_eq!(sol.objective, Some(qi(1)));
}

#[test]
fn overdetermined_but_consistent_equalities() {
    let s = Spec::min(vec![qi(0), qi(0)])
        .eq(vec![qi(1), qi(0)], qi(2))
        .eq(vec![qi(0), qi(1)], qi(3))
        .eq(vec![qi(1), qi(1)], qi(5));
    let sol = s.solve_checked();
    assert_eq!(sol.x, vec![qi(2), qi(3)]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Maximise / minimise consistency and duality
// ═══════════════════════════════════════════════════════════════════════════

fn duality_suite() -> Vec<Spec> {
    vec![
        Spec::max(vec![qi(3), qi(2)])
            .le(vec![qi(1), qi(1)], qi(4))
            .le(vec![qi(1), qi(3)], qi(6)),
        Spec::min(vec![qi(1), qi(1)])
            .ge(vec![qi(1), qi(2)], qi(1))
            .ge(vec![qi(3), qi(1)], qi(1)),
        Spec::max(vec![qi(5), qi(4)])
            .le(vec![qi(6), qi(4)], qi(24))
            .le(vec![qi(1), qi(2)], qi(6))
            .le(vec![qi(-1), qi(1)], qi(1)),
        Spec::min(vec![qi(2), qi(3), qi(1)])
            .ge(vec![qi(1), qi(1), qi(1)], qi(10))
            .ge(vec![qi(1), qi(-1), qi(0)], qi(0))
            .le(vec![qi(0), qi(0), qi(1)], qi(4)),
        Spec::max(vec![qi(1), qi(2), qi(3)])
            .eq(vec![qi(1), qi(1), qi(1)], qi(6))
            .le(vec![qi(1), qi(0), qi(2)], qi(8))
            .ge(vec![qi(0), qi(1), qi(0)], qi(1)),
    ]
}

#[test]
fn duality_primal_equals_dual_objective_on_five_problems() {
    for (k, s) in duality_suite().iter().enumerate() {
        let sol = s.solve_checked();
        assert_eq!(sol.status, LpStatus::Optimal, "problem {k}");
        // Default bounds x ≥ 0 and rⱼxⱼ = 0 ⇒ cᵀx = yᵀb exactly.
        let ytb: Q = s
            .rows
            .iter()
            .zip(&sol.duals)
            .map(|((_, _, b), y)| b * y)
            .sum();
        assert_eq!(
            sol.objective.as_ref(),
            Some(&ytb),
            "problem {k}: strong duality"
        );
    }
}

#[test]
fn complementary_slackness_on_five_problems() {
    for (k, s) in duality_suite().iter().enumerate() {
        let sol = s.solve_checked();
        for (i, (row, _, b)) in s.rows.iter().enumerate() {
            let slack = &dot(row, &sol.x) - b;
            assert!((&slack * &sol.duals[i]).is_zero(), "problem {k} row {i}");
        }
    }
}

#[test]
fn maximize_equals_negated_minimize_of_negated_objective() {
    for s in duality_suite() {
        let flipped = Spec {
            maximize: !s.maximize,
            c: s.c.iter().map(|v| -v).collect(),
            ..s.clone()
        };
        let a = s.solve_checked();
        let b = flipped.solve_checked();
        assert_eq!(a.status, b.status);
        assert_eq!(a.x, b.x);
        assert_eq!(a.objective.as_ref().map(|v| -v), b.objective);
        // Shadow prices flip sign with the objective.
        let neg: Vec<Q> = b.duals.iter().map(|v| -v).collect();
        assert_eq!(a.duals, neg);
    }
}

#[test]
fn duals_are_shadow_prices_under_perturbation() {
    // Perturb each rhs by a small ε and check Δz = yᵢ·ε (non-degenerate problem).
    let s = Spec::max(vec![qi(5), qi(4)])
        .le(vec![qi(6), qi(4)], qi(24))
        .le(vec![qi(1), qi(2)], qi(6));
    let base = s.solve_checked();
    let eps = q(1, 100);
    for i in 0..2 {
        let mut t = s.clone();
        t.rows[i].2 = &t.rows[i].2 + &eps;
        let pert = t.solve_checked();
        let dz = pert.objective.unwrap() - base.objective.clone().unwrap();
        assert_eq!(dz, &base.duals[i] * &eps, "row {i}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// feasible_nonneg
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn feasible_nonneg_thirds_and_sevenths() {
    // x/3 + y/7 + z/11 = 1,  x − y = 0,  y − z = 0  →  x = y = z = 231/131
    let a = vec![
        vec![q(1, 3), q(1, 7), q(1, 11)],
        vec![qi(1), qi(-1), qi(0)],
        vec![qi(0), qi(1), qi(-1)],
    ];
    let b = vec![qi(1), qi(0), qi(0)];
    let x = feasible_nonneg(&a, &b).unwrap().expect("feasible");
    assert_eq!(x, vec![q(231, 131); 3]);
    for (row, rhs) in a.iter().zip(&b) {
        assert_eq!(dot(row, &x), *rhs);
    }
}

#[test]
fn feasible_nonneg_returns_exact_solution_for_wide_system() {
    let a = vec![vec![q(1, 3), q(1, 7), q(2, 5)], vec![qi(1), qi(1), qi(1)]];
    let b = vec![qi(1), qi(4)];
    let x = feasible_nonneg(&a, &b).unwrap().expect("feasible");
    assert!(x.iter().all(|v| !v.is_negative()));
    for (row, rhs) in a.iter().zip(&b) {
        assert_eq!(dot(row, &x), *rhs);
    }
}

#[test]
fn feasible_nonneg_none_when_infeasible() {
    assert!(
        feasible_nonneg(&[vec![qi(1), qi(1)]], &[qi(-1)])
            .unwrap()
            .is_none()
    );
    assert!(
        feasible_nonneg(&[vec![qi(1), qi(1)], vec![qi(1), qi(1)]], &[qi(1), qi(2)])
            .unwrap()
            .is_none()
    );
}

#[test]
fn feasible_nonneg_errors_on_malformed() {
    assert!(feasible_nonneg(&[], &[]).is_err());
    assert!(feasible_nonneg(&[vec![]], &[qi(1)]).is_err());
    assert!(feasible_nonneg(&[vec![qi(1)]], &[qi(1), qi(2)]).is_err());
    assert!(feasible_nonneg(&[vec![qi(1), qi(2)], vec![qi(1)]], &[qi(1), qi(2)]).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// linprog (SciPy-shaped) and linprog_matrix
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn linprog_scipy_shape_known_answer() {
    // SciPy docs example: min −x + 4y s.t. −3x + y ≤ 6, x + 2y ≤ 4, y ≥ −3, x free
    //   → x = (10, −3), fun = −22.
    let sol = linprog(
        &[qi(-1), qi(4)],
        &[vec![qi(-3), qi(1)], vec![qi(1), qi(2)]],
        &[qi(6), qi(4)],
        &[],
        &[],
        &[Bounds::free(), Bounds::at_least(qi(-3))],
    )
    .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    assert_eq!(sol.x, vec![qi(10), qi(-3)]);
    assert_eq!(sol.objective, Some(qi(-22)));
}

#[test]
fn linprog_with_equalities_and_default_bounds() {
    let sol = linprog(
        &[qi(1), qi(2), qi(3)],
        &[vec![qi(1), qi(1), qi(0)]],
        &[qi(5)],
        &[vec![qi(1), qi(1), qi(1)]],
        &[qi(6)],
        &[],
    )
    .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    // x + y ≤ 5 forces z ≥ 1; cheapest: x = 5, z = 1 → 8.
    assert_eq!(sol.x, vec![qi(5), qi(0), qi(1)]);
    assert_eq!(sol.objective, Some(qi(8)));
    assert_eq!(sol.duals.len(), 2);
}

#[test]
fn linprog_shape_errors() {
    assert!(linprog(&[qi(1)], &[vec![qi(1)]], &[], &[], &[], &[]).is_err());
    assert!(linprog(&[qi(1)], &[], &[], &[vec![qi(1)]], &[qi(1), qi(2)], &[]).is_err());
    assert!(linprog(&[qi(1), qi(2)], &[], &[], &[], &[], &[Bounds::free()]).is_err());
    assert!(linprog(&[qi(1)], &[vec![qi(1), qi(2)]], &[qi(1)], &[], &[], &[]).is_err());
    assert!(linprog(&[], &[], &[], &[], &[], &[]).is_err());
}

#[test]
fn linprog_matrix_with_ex_entries() {
    let ctx = Context::new();
    let c = matrix![ctx, [3], [2]];
    let a = matrix![ctx, [1, 1], [1, 3]];
    let b = matrix![ctx, [4], [6]];
    let sol = linprog_matrix(Objective::Maximize, &c, Some(&a), Some(&b), None, None).unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    assert_eq!(sol.x_ex(&ctx), vec![ctx.int(4), ctx.int(0)]);
    assert_eq!(sol.objective, Some(qi(12)));
}

#[test]
fn linprog_matrix_row_vector_objective_and_rational_entries() {
    let ctx = Context::new();
    let c = matrix![ctx, [1, 1]];
    let a = Matrix::new(vec![
        vec![ctx.rational(1, 2), ctx.rational(1, 3)],
        vec![ctx.int(1), ctx.int(0)],
    ])
    .unwrap();
    let b = matrix![ctx, [1], [1]];
    let sol = linprog_matrix(Objective::Maximize, &c, Some(&a), Some(&b), None, None).unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    // x ≤ 1, x/2 + y/3 ≤ 1: vertices (1, 3/2) → 5/2 and (0, 3) → 3.
    assert_eq!(sol.x, vec![qi(0), qi(3)]);
    assert_eq!(sol.objective, Some(qi(3)));

    // With objective 2x + y the other vertex wins: 2 + 3/2 = 7/2 > 3.
    let c2 = matrix![ctx, [2, 1]];
    let sol2 = linprog_matrix(Objective::Maximize, &c2, Some(&a), Some(&b), None, None).unwrap();
    assert_eq!(sol2.x, vec![qi(1), q(3, 2)]);
    assert_eq!(sol2.objective, Some(q(7, 2)));
}

#[test]
fn linprog_matrix_folds_constant_expressions() {
    let ctx = Context::new();
    let c = matrix![ctx, [1]];
    let a = Matrix::new(vec![vec![&ctx.rational(1, 2) + &ctx.rational(1, 2)]]).unwrap();
    let b = Matrix::new(vec![vec![&ctx.int(2) * &ctx.int(3)]]).unwrap();
    let sol = linprog_matrix(Objective::Maximize, &c, Some(&a), Some(&b), None, None).unwrap();
    assert_eq!(sol.x, vec![qi(6)]);
}

#[test]
fn linprog_matrix_symbolic_entry_is_invalid_argument() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let c = matrix![ctx, [1], [1]];
    let a = Matrix::new(vec![vec![x.clone(), ctx.int(1)]]).unwrap();
    let b = matrix![ctx, [1]];
    let err = linprog_matrix(Objective::Minimize, &c, Some(&a), Some(&b), None, None).unwrap_err();
    assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
    assert!(err.to_string().contains("numeric literal"), "{err}");
    let pi_c = Matrix::new(vec![vec![ctx.pi()], vec![ctx.int(1)]]).unwrap();
    assert!(linprog_matrix(Objective::Minimize, &pi_c, None, None, None, None).is_err());
}

#[test]
fn linprog_matrix_shape_and_pairing_errors() {
    let ctx = Context::new();
    let c = matrix![ctx, [1], [1]];
    let a = matrix![ctx, [1, 1]];
    let b = matrix![ctx, [1]];
    assert!(linprog_matrix(Objective::Minimize, &c, Some(&a), None, None, None).is_err());
    assert!(linprog_matrix(Objective::Minimize, &c, None, Some(&b), None, None).is_err());
    let wide_c = matrix![ctx, [1, 1], [1, 1]];
    assert!(linprog_matrix(Objective::Minimize, &wide_c, None, None, None, None).is_err());
    let a3 = matrix![ctx, [1, 1, 1]];
    assert!(linprog_matrix(Objective::Minimize, &c, Some(&a3), Some(&b), None, None).is_err());
    let b2 = matrix![ctx, [1], [2]];
    assert!(linprog_matrix(Objective::Minimize, &c, Some(&a), Some(&b2), None, None).is_err());
}

#[test]
fn linprog_matrix_no_constraints_is_bounded_by_default_bounds() {
    let ctx = Context::new();
    let c = matrix![ctx, [1], [2]];
    let sol = linprog_matrix(Objective::Minimize, &c, None, None, None, None).unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    assert_eq!(sol.x, vec![qi(0), qi(0)]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Certificate search: non-negative combinations of Ex column vectors
// ═══════════════════════════════════════════════════════════════════════════

/// Build the `n×k` matrix whose columns are the given `Ex` vectors.
fn columns(cols: &[Vec<Ex>]) -> Matrix {
    let n = cols[0].len();
    Matrix::from_fn(n, cols.len(), |i, j| cols[j][i].clone()).unwrap()
}

#[test]
fn nonnegative_combination_certificate_found() {
    let ctx = Context::new();
    let v1 = vec![ctx.int(1), ctx.int(0), ctx.int(1)];
    let v2 = vec![ctx.int(0), ctx.int(1), ctx.int(1)];
    let v3 = vec![ctx.int(1), ctx.int(1), ctx.int(0)];
    let a = columns(&[v1, v2, v3]);
    // target = 2·v₁ + 3·v₂ + 0·v₃ = (2, 3, 5)
    let target = matrix![ctx, [2], [3], [5]];
    let c = matrix![ctx, [0], [0], [0]];
    let sol = linprog_matrix(Objective::Minimize, &c, None, None, Some(&a), Some(&target)).unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    let mu = Matrix::col_vector(sol.x_ex(&ctx)).unwrap();
    assert_eq!((&a * &mu).eval(), target);
    assert!(sol.x.iter().all(|v| !v.is_negative()));
}

#[test]
fn nonnegative_combination_certificate_with_rational_weights() {
    let ctx = Context::new();
    let v1 = vec![ctx.int(2), ctx.int(0)];
    let v2 = vec![ctx.int(0), ctx.int(3)];
    let a = columns(&[v1, v2]);
    let target = matrix![ctx, [1], [1]];
    let c = matrix![ctx, [1], [1]];
    let sol = linprog_matrix(Objective::Minimize, &c, None, None, Some(&a), Some(&target)).unwrap();
    assert_eq!(sol.x, vec![q(1, 2), q(1, 3)]);
}

#[test]
fn nonnegative_combination_impossible_gives_separating_hyperplane() {
    let ctx = Context::new();
    let v1 = vec![ctx.int(1), ctx.int(0), ctx.int(1)];
    let v2 = vec![ctx.int(0), ctx.int(1), ctx.int(1)];
    let v3 = vec![ctx.int(1), ctx.int(1), ctx.int(0)];
    let a = columns(&[v1, v2, v3]);
    let target = matrix![ctx, [-1], [0], [0]]; // outside the cone
    let c = matrix![ctx, [0], [0], [0]];
    let sol = linprog_matrix(Objective::Minimize, &c, None, None, Some(&a), Some(&target)).unwrap();
    assert_eq!(sol.status, LpStatus::Infeasible);
    let y = sol.farkas.unwrap();
    // Separating hyperplane: yᵀvᵢ ≥ 0 for every generator, yᵀ·target < 0.
    let rows = a.to_rational_rows().unwrap();
    for j in 0..3 {
        let g: Q = rows.iter().zip(&y).map(|(row, yi)| &row[j] * yi).sum();
        assert!(!g.is_negative(), "generator {j}");
    }
    let t = target.to_rational_rows().unwrap();
    let ytb: Q = t.iter().zip(&y).map(|(row, yi)| &row[0] * yi).sum();
    assert!(ytb.is_negative());
}

// ═══════════════════════════════════════════════════════════════════════════
// Misc API
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn q_and_qi_constructors() {
    assert_eq!(q(6, 4), Ratio::new(BigInt::from(3), BigInt::from(2)));
    assert_eq!(qi(-7), Ratio::from_integer(BigInt::from(-7)));
    assert_eq!(q(-1, -2), q(1, 2));
}

#[test]
fn builder_reports_num_vars_and_constraints() {
    let p = LpProblem::minimize(vec![qi(1), qi(2), qi(3)])
        .le(vec![qi(1), qi(1), qi(1)], qi(1))
        .eq(vec![qi(1), qi(0), qi(0)], qi(0));
    assert_eq!(p.num_vars(), 3);
    assert_eq!(p.num_constraints(), 2);
}

#[test]
fn builder_errors_for_malformed_input() {
    assert!(matches!(
        LpProblem::minimize(vec![]).solve(),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        LpProblem::minimize(vec![qi(1)])
            .le(vec![qi(1), qi(1)], qi(1))
            .solve(),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        LpProblem::minimize(vec![qi(1)])
            .bounds(1, Bounds::free())
            .solve(),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        LpProblem::minimize(vec![qi(1)]).free(7).solve(),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn x_ex_produces_exact_rationals_in_context() {
    let ctx = Context::new();
    let sol = Spec::min(vec![qi(1), qi(1)])
        .ge(vec![qi(1), qi(2)], qi(1))
        .ge(vec![qi(3), qi(1)], qi(1))
        .solve();
    assert_eq!(sol.x_ex(&ctx), vec![ctx.rational(1, 5), ctx.rational(2, 5)]);
}

#[test]
fn duals_ex_matches_duals() {
    let ctx = Context::new();
    let sol = LpProblem::maximize(vec![qi(3), qi(2)])
        .le(vec![qi(1), qi(1)], qi(4))
        .le(vec![qi(1), qi(3)], qi(6))
        .solve()
        .unwrap();
    let d = sol.duals_ex(&ctx);
    assert_eq!(d.len(), sol.duals.len());
    for (e, r) in d.iter().zip(&sol.duals) {
        assert_eq!(e.as_rational().as_ref(), Some(r));
    }
}

#[test]
fn display_summarises_every_status() {
    let opt = LpProblem::maximize(vec![qi(3), qi(2)])
        .le(vec![qi(1), qi(1)], qi(4))
        .le(vec![qi(1), qi(3)], qi(6))
        .solve()
        .unwrap();
    assert_eq!(
        opt.to_string(),
        "Optimal: x = (4, 0), objective = 12, duals = (3, 0)"
    );
    let frac = LpProblem::minimize(vec![qi(1), qi(1)])
        .ge(vec![qi(1), qi(2)], qi(1))
        .ge(vec![qi(3), qi(1)], qi(1))
        .solve()
        .unwrap();
    assert!(
        frac.to_string()
            .starts_with("Optimal: x = (1/5, 2/5), objective = 3/5"),
        "{frac}"
    );
    let inf = LpProblem::minimize(vec![qi(1), qi(1)])
        .le(vec![qi(1), qi(1)], qi(1))
        .ge(vec![qi(1), qi(1)], qi(2))
        .solve()
        .unwrap();
    assert!(
        inf.to_string()
            .starts_with("Infeasible: Farkas certificate y = ("),
        "{inf}"
    );
    let unb = LpProblem::maximize(vec![qi(1), qi(1)])
        .le(vec![qi(1), qi(-1)], qi(1))
        .solve()
        .unwrap();
    assert_eq!(unb.to_string(), "Unbounded");
    let out = LpProblem::maximize(vec![qi(1), qi(1)])
        .le(vec![qi(1), qi(-1)], qi(1))
        .with_budget(Budget::within(Duration::ZERO))
        .solve()
        .unwrap();
    assert_eq!(out.to_string(), "Budget exhausted");
}

// ═══════════════════════════════════════════════════════════════════════════
// Budgets (0.9.1)
// ═══════════════════════════════════════════════════════════════════════════

/// A problem that needs at least two pivots (two `≥` rows, both
/// artificials must leave the basis in phase 1).
fn two_pivot_problem() -> LpProblem {
    LpProblem::minimize(vec![qi(1), qi(1)])
        .ge(vec![qi(1), qi(2)], qi(1))
        .ge(vec![qi(3), qi(1)], qi(1))
}

#[test]
fn budget_max_pivots_stops_a_solve_that_needs_more() {
    let p = two_pivot_problem();
    let sol = p
        .clone()
        .with_budget(Budget::max_pivots(1))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::BudgetExhausted);
    assert!(sol.x.is_empty());
    assert_eq!(sol.objective, None);
    assert!(sol.duals.is_empty());
    assert_eq!(sol.farkas, None);
    assert!(!sol.is_optimal());
    // Zero pivots allowed: still an answer, not an error.
    let sol = p
        .clone()
        .with_budget(Budget::max_pivots(0))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::BudgetExhausted);
    // A budget the solve fits inside changes nothing.
    let free = p.solve().unwrap();
    let roomy = p
        .clone()
        .with_budget(Budget::max_pivots(1_000))
        .solve()
        .unwrap();
    assert_eq!(roomy.status, LpStatus::Optimal);
    assert_eq!(roomy.x, free.x);
    assert_eq!(roomy.x, vec![q(1, 5), q(2, 5)]);
    assert_eq!(roomy.objective, free.objective);
    assert_eq!(roomy.duals, free.duals);
}

#[test]
fn budget_deadline_is_absolute_and_checked_before_the_first_pivot() {
    let p = two_pivot_problem();
    let started = Instant::now();
    let sol = p
        .clone()
        .with_budget(Budget::within(Duration::ZERO))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::BudgetExhausted);
    assert!(started.elapsed() < Duration::from_secs(1));
    let sol = p
        .clone()
        .with_budget(Budget::deadline(Instant::now() - Duration::from_millis(1)))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::BudgetExhausted);
    // A deadline comfortably in the future is never hit by a small LP.
    let sol = p
        .with_budget(Budget::within(Duration::from_secs(60)))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    assert_eq!(sol.x, vec![q(1, 5), q(2, 5)]);
}

#[test]
fn budget_builders_combine_and_default_is_unlimited() {
    let b = Budget::default();
    assert_eq!(b.deadline, None);
    assert_eq!(b.max_pivots, None);
    assert_eq!(*two_pivot_problem().budget(), Budget::default());
    let at = Instant::now() + Duration::from_secs(30);
    let b = Budget::deadline(at).with_max_pivots(7);
    assert_eq!(b.deadline, Some(at));
    assert_eq!(b.max_pivots, Some(7));
    let b = Budget::max_pivots(7).with_deadline(at);
    assert_eq!((b.deadline, b.max_pivots), (Some(at), Some(7)));
    let p = two_pivot_problem().with_budget(b.clone());
    assert_eq!(p.budget(), &b);
    assert_eq!(p.solve().unwrap().status, LpStatus::Optimal);
    // The cap binds even when the deadline is generous.
    let sol = two_pivot_problem()
        .with_budget(Budget::within(Duration::from_secs(60)).with_max_pivots(1))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::BudgetExhausted);
    // Malformed input is still an error under a budget.
    assert!(
        LpProblem::minimize(vec![qi(1)])
            .le(vec![qi(1), qi(2)], qi(1))
            .with_budget(Budget::max_pivots(0))
            .solve()
            .is_err()
    );
    // Contradictory bounds are decided before any pivot: no budget needed.
    let sol = LpProblem::minimize(vec![qi(1)])
        .bounds(0, Bounds::closed(qi(2), qi(1)))
        .with_budget(Budget::max_pivots(0))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::Infeasible);
}

/// The budget is shared by the whole solve: an `i64` attempt that
/// overflows into `BigInt` does not get its pivots back.
#[test]
fn budget_pivots_count_across_the_cell_type_fallback() {
    let big = |k: i64| qi(k) * qi(1 << 40);
    let p = LpProblem::minimize(vec![qi(1), qi(1), qi(1)])
        .ge(vec![big(3), big(1), big(2)], big(7))
        .ge(vec![big(1), big(5), big(1)], big(11))
        .le(vec![big(2), big(1), big(3)], big(40));
    let free = p.solve().unwrap();
    assert_eq!(free.status, LpStatus::Optimal);
    // Find the smallest cap that lets the solve finish; every smaller cap
    // is exhausted, and the answer under the smallest sufficient cap is the
    // unbudgeted one.
    let mut needed = None;
    for cap in 0..200usize {
        let sol = p
            .clone()
            .with_budget(Budget::max_pivots(cap))
            .solve()
            .unwrap();
        match sol.status {
            LpStatus::BudgetExhausted => assert!(needed.is_none()),
            LpStatus::Optimal => {
                assert_eq!(sol.x, free.x);
                assert_eq!(sol.objective, free.objective);
                needed.get_or_insert(cap);
            }
            other => panic!("{other:?}"),
        }
    }
    let needed = needed.expect("a cap under 200 suffices");
    // The i64 attempt performs at least one pivot before it overflows, so
    // the shared count exceeds what the BigInt run alone would need: the
    // cap must cover both, hence strictly more than the pivots of a
    // problem of this shape with small coefficients.
    let small = LpProblem::minimize(vec![qi(1), qi(1), qi(1)])
        .ge(vec![qi(3), qi(1), qi(2)], qi(7))
        .ge(vec![qi(1), qi(5), qi(1)], qi(11))
        .le(vec![qi(2), qi(1), qi(3)], qi(40));
    let mut small_needed = None;
    for cap in 0..200usize {
        if small
            .clone()
            .with_budget(Budget::max_pivots(cap))
            .solve()
            .unwrap()
            .status
            == LpStatus::Optimal
        {
            small_needed = Some(cap);
            break;
        }
    }
    let small_needed = small_needed.expect("a cap under 200 suffices");
    assert!(
        needed > small_needed,
        "shared budget: {needed} pivots for the overflowing problem vs {small_needed} for the small one"
    );
}

#[test]
fn feasible_nonneg_certified_agrees_with_feasible_nonneg_and_certifies() {
    // Feasible: x/3 + y/7 = 1, x − y = 0.
    let a = [vec![q(1, 3), q(1, 7)], vec![qi(1), qi(-1)]];
    let b = [qi(1), qi(0)];
    let plain = feasible_nonneg(&a, &b).unwrap().unwrap();
    match feasible_nonneg_certified(&a, &b).unwrap() {
        Feasibility::Feasible(x) => {
            assert_eq!(x, plain);
            assert_eq!(x, vec![q(21, 10), q(21, 10)]);
        }
        other => panic!("expected feasible, got {other:?}"),
    }

    // Infeasible: x + y = 1 and x + y = 2.  Verify the Farkas inequality
    // Aᵀy ≥ 0, yᵀb < 0 exactly.
    let a = [vec![qi(1), qi(1)], vec![qi(1), qi(1)]];
    let b = [qi(1), qi(2)];
    assert!(feasible_nonneg(&a, &b).unwrap().is_none());
    let res = feasible_nonneg_certified(&a, &b).unwrap();
    assert!(!res.is_feasible());
    assert!(res.witness().is_none());
    let Feasibility::Infeasible { farkas: Some(y) } = res else {
        panic!("expected a Farkas certificate, got {res:?}");
    };
    assert_eq!(y.len(), 2);
    for j in 0..2 {
        let g: Q = a.iter().zip(&y).map(|(row, yi)| &row[j] * yi).sum();
        assert!(g >= Q::zero(), "(Aᵀy)_{j} = {g}");
    }
    let yb: Q = y.iter().zip(&b).map(|(u, v)| u * v).sum();
    assert!(yb < Q::zero(), "yᵀb = {yb}");
}

#[test]
fn nonneg_combination_is_cone_membership_by_columns() {
    // (1, 1) = ½·(2, 0) + 1·(0, 1)
    let cone = [vec![qi(2), qi(0)], vec![qi(0), qi(1)]];
    assert_eq!(
        nonneg_combination(&cone, &[qi(1), qi(1)]).unwrap(),
        Feasibility::Feasible(vec![q(1, 2), qi(1)])
    );
    // A target with a negative first coordinate is separated by y = (1, 0):
    // y·(2,0) = 2 ≥ 0, y·(0,1) = 0 ≥ 0, y·(−1, 1) = −1 < 0.
    match nonneg_combination(&cone, &[qi(-1), qi(1)]).unwrap() {
        Feasibility::Infeasible { farkas: Some(y) } => {
            for v in &cone {
                assert!(dot(&y, v) >= Q::zero());
            }
            assert!(dot(&y, &[qi(-1), qi(1)]) < Q::zero());
        }
        other => panic!("expected a certificate, got {other:?}"),
    }
    // Three linearly independent generators in ℚ³: the decomposition is
    // unique, so membership is decided by the sign of the unique solution.
    let gens = [
        vec![qi(1), qi(0), qi(0)],
        vec![qi(1), qi(1), qi(0)],
        vec![qi(1), qi(1), qi(1)],
    ];
    // (3, 2, 1) = 1·(1,0,0) + 1·(1,1,0) + 1·(1,1,1)
    assert_eq!(
        nonneg_combination(&gens, &[qi(3), qi(2), qi(1)]).unwrap(),
        Feasibility::Feasible(vec![qi(1), qi(1), qi(1)])
    );
    // (1, 2, 3) = 0·(1,0,0) − 1·(1,1,0) + 3·(1,1,1) needs a negative
    // coefficient, so it is outside the cone; the certificate separates it.
    match nonneg_combination(&gens, &[qi(1), qi(2), qi(3)]).unwrap() {
        Feasibility::Infeasible { farkas: Some(y) } => {
            for v in &gens {
                assert!(dot(&y, v) >= Q::zero());
            }
            assert!(dot(&y, &[qi(1), qi(2), qi(3)]) < Q::zero());
        }
        other => panic!("expected a certificate, got {other:?}"),
    }
    // Errors: jagged / empty input.
    assert!(nonneg_combination(&[], &[qi(1)]).is_err());
    assert!(nonneg_combination(&[vec![qi(1)]], &[qi(1), qi(2)]).is_err());
}

/// The hybrid arithmetic (`i64 → i128 → 256-bit → BigInt` cells) is
/// invisible from outside: the same LP with every row, right-hand side and
/// cost scaled by `2^k` has the same feasible set, hence the same optimal
/// point and duals and a `2^k`-fold objective, for `k` that lands the
/// fraction-free tableau (minors of the scaled data, up to four-fold
/// products here) in each cell type.  Every run passes the exact KKT
/// check and the wide-coefficient runs stay fast.
#[test]
fn scaled_data_lands_in_every_cell_type_and_agrees() {
    let base = |scale: &Q| {
        let s = |k: i64| qi(k) * scale;
        Spec::min(vec![s(3), s(1), s(4), s(2)])
            .ge(vec![s(3), s(1), s(2), s(1)], s(7))
            .ge(vec![s(1), s(5), s(1), s(2)], s(11))
            .le(vec![s(2), s(1), s(3), s(4)], s(40))
            .eq(vec![s(1), s(-1), s(1), s(0)], s(1))
    };
    let reference = base(&qi(1)).solve_checked();
    assert_eq!(reference.status, LpStatus::Optimal);
    // Roughly: 2^0 stays in i64, 2^20 needs i128, 2^50 the 256-bit cells,
    // 2^100 BigInt.
    for bits in [0u32, 20, 50, 100] {
        let scale = Ratio::from_integer(BigInt::from(1) << bits);
        let t0 = Instant::now();
        let sol = base(&scale).solve_checked();
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "2^{bits}: {:?}",
            t0.elapsed()
        );
        assert_eq!(sol.status, LpStatus::Optimal, "2^{bits}");
        assert_eq!(sol.x, reference.x, "2^{bits}: the optimum is scale-free");
        assert_eq!(
            sol.objective,
            reference.objective.as_ref().map(|v| v * &scale),
            "2^{bits}"
        );
        assert_eq!(
            sol.duals, reference.duals,
            "2^{bits}: shadow prices are scale-free"
        );
    }
    // With fractions too (row scaling by denominators is part of the
    // conversion), at a scale only BigInt can hold.
    let scale = Ratio::new(BigInt::from(1) << 300u32, BigInt::from(7));
    let sol = base(&scale).solve_checked();
    assert_eq!(sol.x, reference.x);
    assert_eq!(
        sol.objective,
        reference.objective.as_ref().map(|v| v * &scale)
    );
    assert_eq!(sol.duals, reference.duals);
}

#[test]
fn larger_transportation_problem_is_fast_and_integral() {
    // 4 sources × 5 sinks, balanced: 20 variables, 9 equalities.
    let supply = [15i64, 25, 20, 30];
    let demand = [10i64, 20, 25, 15, 20];
    let cost = [
        [4i64, 8, 8, 6, 5],
        [6, 2, 4, 9, 7],
        [3, 5, 6, 4, 8],
        [7, 6, 3, 5, 2],
    ];
    let n = 20;
    let c: Vec<Q> = cost.iter().flatten().map(|&v| qi(v)).collect();
    let mut s = Spec::min(c);
    for (i, &sup) in supply.iter().enumerate() {
        let mut row = vec![qi(0); n];
        for j in 0..5 {
            row[i * 5 + j] = qi(1);
        }
        s = s.eq(row, qi(sup));
    }
    for (j, &dem) in demand.iter().enumerate() {
        let mut row = vec![qi(0); n];
        for i in 0..4 {
            row[i * 5 + j] = qi(1);
        }
        s = s.eq(row, qi(dem));
    }
    let t0 = std::time::Instant::now();
    let sol = s.solve_checked();
    assert!(
        t0.elapsed() < std::time::Duration::from_secs(5),
        "{:?}",
        t0.elapsed()
    );
    assert_eq!(sol.status, LpStatus::Optimal);
    assert!(
        sol.x.iter().all(|v| v.is_integer()),
        "transportation optima are integral"
    );
    // Hand-built feasible plan gives an upper bound on the optimum.
    let plan = [
        [10i64, 0, 0, 5, 0],
        [0, 20, 5, 0, 0],
        [0, 0, 10, 10, 0],
        [0, 0, 10, 0, 20],
    ];
    let plan_cost: i64 = (0..4)
        .map(|i| (0..5).map(|j| plan[i][j] * cost[i][j]).sum::<i64>())
        .sum();
    assert!(sol.objective.clone().unwrap() <= qi(plan_cost));
}

// ═══════════════════════════════════════════════════════════════════════════
// Property tests: random small LPs, exact KKT / Farkas / unboundedness
// ═══════════════════════════════════════════════════════════════════════════

fn random_spec(
    nvars: usize,
    nrows: usize,
    maximize: bool,
    data: &[i64],
    rels: &[u8],
    free_mask: u8,
) -> Spec {
    let mut idx = 0;
    let mut next = || {
        let v = data[idx % data.len()];
        idx += 1;
        qi(v)
    };
    let c: Vec<Q> = (0..nvars).map(|_| next()).collect();
    let mut s = if maximize { Spec::max(c) } else { Spec::min(c) };
    for r in 0..nrows {
        let row: Vec<Q> = (0..nvars).map(|_| next()).collect();
        let rhs = next();
        s = match rels[r % rels.len()] % 3 {
            0 => s.le(row, rhs),
            1 => s.ge(row, rhs),
            _ => s.eq(row, rhs),
        };
    }
    for j in 0..nvars {
        if free_mask & (1 << j) != 0 {
            s = s.free(j);
        }
    }
    s
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(160))]

    /// Every verdict on a random small LP carries an exactly verifiable
    /// certificate: KKT (Optimal), Farkas (Infeasible), or growth under
    /// boxing plus feasibility (Unbounded).
    #[test]
    fn prop_random_lp_certificates(
        nvars in 2usize..=4,
        nrows in 2usize..=4,
        maximize in any::<bool>(),
        data in prop::collection::vec(-5i64..=5, 40),
        rels in prop::collection::vec(0u8..3, 4),
        free_mask in 0u8..16,
    ) {
        let s = random_spec(nvars, nrows, maximize, &data, &rels, free_mask);
        let sol = s.solve_checked();
        match sol.status {
            LpStatus::Optimal => prop_assert!(s.is_feasible(&sol.x)),
            LpStatus::Infeasible => prop_assert!(sol.farkas.is_some()),
            LpStatus::Unbounded => prop_assert!(sol.x.is_empty()),
            LpStatus::BudgetExhausted => prop_assert!(false, "no budget was set"),
        }
    }

    /// A budget never changes an answer that fits inside it: with a pivot
    /// cap at least the number of pivots the solve needs, the result is the
    /// unbudgeted one; below it, `BudgetExhausted` and nothing else.
    #[test]
    fn prop_budget_is_monotone_and_never_alters_an_answer(
        nvars in 1usize..=3,
        nrows in 1usize..=3,
        maximize in any::<bool>(),
        data in prop::collection::vec(-5i64..=5, 40),
        rels in prop::collection::vec(0u8..3, 4),
        free_mask in 0u8..16,
    ) {
        let s = random_spec(nvars, nrows, maximize, &data, &rels, free_mask);
        let free = s.solve();
        let mut finished = false;
        for cap in 0..64usize {
            let sol = s.build().with_budget(Budget::max_pivots(cap)).solve().expect("well-formed LP");
            if sol.status == LpStatus::BudgetExhausted {
                prop_assert!(!finished, "exhausted at cap {cap} after finishing at a smaller one");
                prop_assert!(sol.x.is_empty() && sol.objective.is_none() && sol.duals.is_empty() && sol.farkas.is_none());
            } else {
                finished = true;
                prop_assert_eq!(&sol.status, &free.status);
                prop_assert_eq!(&sol.x, &free.x);
                prop_assert_eq!(&sol.objective, &free.objective);
                prop_assert_eq!(&sol.duals, &free.duals);
                prop_assert_eq!(&sol.farkas, &free.farkas);
            }
        }
        prop_assert!(finished, "a 3×3 LP needs fewer than 64 pivots");
    }

    /// The optimum is at least as good as any random feasible point found
    /// by rejection sampling of the box [0, 6]ⁿ.
    #[test]
    fn prop_optimum_dominates_random_feasible_points(
        nvars in 2usize..=3,
        nrows in 2usize..=3,
        data in prop::collection::vec(-4i64..=4, 30),
        rels in prop::collection::vec(0u8..2, 3),
        samples in prop::collection::vec(prop::collection::vec(0i64..=6, 3), 40),
    ) {
        let s = random_spec(nvars, nrows, false, &data, &rels, 0);
        let sol = s.solve();
        if sol.status != LpStatus::Optimal {
            // Infeasible/unbounded: nothing to dominate; covered by the
            // other property.
            return Ok(());
        }
        let best = sol.objective.clone().unwrap();
        for p in &samples {
            let x: Vec<Q> = p.iter().take(nvars).map(|&v| qi(v)).collect();
            if s.is_feasible(&x) {
                prop_assert!(dot(&s.c, &x) >= best, "point {x:?} beats the optimum {best}");
            }
        }
    }

    /// max f = −min(−f) on random data, including the duals.
    #[test]
    fn prop_max_min_consistency(
        nvars in 2usize..=3,
        nrows in 1usize..=3,
        data in prop::collection::vec(-5i64..=5, 30),
        rels in prop::collection::vec(0u8..3, 3),
    ) {
        let s = random_spec(nvars, nrows, true, &data, &rels, 0);
        let flipped = Spec { maximize: false, c: s.c.iter().map(|v| -v).collect(), ..s.clone() };
        let a = s.solve();
        let b = flipped.solve();
        prop_assert_eq!(&a.status, &b.status);
        prop_assert_eq!(&a.x, &b.x);
        prop_assert_eq!(a.objective.as_ref().map(|v| -v), b.objective);
        let neg: Vec<Q> = b.duals.iter().map(|v| -v).collect();
        prop_assert_eq!(a.duals, neg);
    }
}
