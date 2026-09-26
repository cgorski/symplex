//! After 0.29 — the algebra hunter: polynomials, Gröbner bases, exact
//! matrices, rational functions, inequalities and sets checked against
//! exact oracles (an own arithmetic over ℚ: Euclid, Sturm, Sylvester
//! determinants, Buchberger, Gaussian elimination, evaluation in number
//! fields), SymPy 1.14 on samples.
//!
//! Each test says what was wrong before; every reference value cites the
//! identity checked or the SymPy call that produced it.

use symplex::num_bigint::BigInt;
use symplex::num_rational::BigRational;
use symplex::prelude::*;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

/// `true` when the rational function `e` (in `vars`) vanishes identically:
/// `e` is evaluated exactly at three rational points; a nonzero rational
/// function of the low degrees used here vanishes at none of them.
fn vanishes_at_points(e: &Ex, vars: &[&Ex]) -> bool {
    let pts = [q(7919, 13), q(-104_729, 17), q(3, 1_000_003)];
    (0..3).all(|k| {
        let subs: Vec<(&Ex, Ex)> = vars
            .iter()
            .enumerate()
            .map(|(i, v)| (*v, e.context().from_ratio(pts[(k + i) % 3].clone())))
            .collect();
        let pairs: Vec<(&Ex, &Ex)> = subs.iter().map(|(v, x)| (*v, x)).collect();
        e.subs_map(&pairs).as_rational() == Some(q(0, 1))
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Eigenvectors of rational matrices: exact pivots
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eigenvectors_of_a_matrix_with_small_entries_are_exact() {
    // Hunter matrix seed 123.  The characteristic polynomial is an
    // irreducible quartic (four simple eigenvalues `RootOf(g, k)`, of sizes
    // 1e-17 … 5e-11), so each eigenspace has dimension 1 (identity: a simple
    // eigenvalue has geometric multiplicity 1).  Before, the pivots of
    // `A − λI` were zero-tested by an `f64` value below 1e-10: every
    // eigenvalue but one got two "eigenvectors" (not in the kernel), the
    // largest got none.
    let ctx = Context::new();
    let rows = [
        "5/100000000002, 3/50000000002, 3/100000000000000000000000000003, 2/5000000000000001",
        "6/10000000000000000000000007, -5/1000000000000000000000000000001, -6/100000000000000000009, 5/100000000000000003",
        "7/10000000002, -7/100000000000000000000006, -1/1000000000000000005, -7/100000000000000000006",
        "-1/1000000000000008, -1/500000000001, 3/5000000000000000000000000003, -2/333333333333333333333333333335",
    ];
    let m = Matrix::new(
        rows.iter()
            .map(|r| r.split(',').map(|e| ctx.parse(e.trim()).unwrap()).collect())
            .collect(),
    )
    .unwrap();
    let evs = m.eigenvects().unwrap();
    assert_eq!(evs.len(), 4);
    for (lam, alg, vecs) in &evs {
        assert_eq!(*alg, 1, "{lam}");
        assert_eq!(vecs.len(), 1, "λ = {lam}: {} eigenvectors", vecs.len());
        // Av − λv at 40 digits, relative to |A|·|v| (the old vectors were
        // off by 100 % of |λv|).
        let v = &vecs[0];
        let av = m.matmul(v).unwrap();
        for i in 0..4 {
            let r = av.get(i, 0) - &(lam * v.get(i, 0));
            let s = r.eval_decimal(40).unwrap();
            let val: f64 = s
                .split_whitespace()
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(1.0);
            assert!(val.abs() < 1e-40, "(Av − λv)[{i}] = {s} for λ = {lam}");
        }
    }
}

#[test]
fn eigenvectors_of_integer_matrices_are_unchanged() {
    // The exact number-field pivots take the same decisions as before on
    // matrices the old zero test got right: `A v = λ v` (checked by
    // simplification to 0) and one vector per simple eigenvalue.
    let ctx = Context::new();
    let m = Matrix::from_i64(
        &ctx,
        &[
            &[1, 3, -1, 5],
            &[3, -4, 2, -1],
            &[-1, 2, -3, 3],
            &[5, -1, 3, 0],
        ],
    )
    .unwrap();
    let evs = m.eigenvects().unwrap();
    let total: usize = evs.iter().map(|(_, k, _)| *k).sum();
    assert_eq!(total, 4);
    for (lam, _, vecs) in &evs {
        assert_eq!(vecs.len(), 1, "{lam}");
        let av = m.matmul(&vecs[0]).unwrap();
        for i in 0..4 {
            let r = (av.get(i, 0) - &(lam * vecs[0].get(i, 0)))
                .eval_complex64()
                .unwrap();
            assert!(r.norm() < 1e-9, "(Av − λv)[{i}] = {r} for λ = {lam}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic matrices: pivots that are identically zero
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rank_of_a_symbolic_matrix_sees_hidden_zero_pivots() {
    // Rows 2 and 3 are multiples of row 1 (factor 3/(t − 1)), so the rank is
    // 1 — SymPy `Matrix([[2*t-2, 3*t-3], [6, 9], [-6, -9]]).rank()` = 1.
    // Before, every structurally nonzero entry was a pivot: rank 2, an
    // empty null space, and an RREF whose second row was not zero.
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let m = Matrix::new(vec![
        vec![&t * 2 - 2, &t * 3 - 3],
        vec![ctx.int(6), ctx.int(9)],
        vec![ctx.int(-6), ctx.int(-9)],
    ])
    .unwrap();
    assert_eq!(m.rank(), 1);
    let (r, pivots) = m.rref();
    assert_eq!(pivots, vec![0]);
    assert!(
        r.get(1, 1).is_zero_structural() && r.get(2, 1).is_zero_structural(),
        "{r}"
    );
    let ns = m.nullspace();
    assert_eq!(ns.len(), 1);
    // A·v = 0 identically in t.
    let av = m.matmul(&ns[0]).unwrap();
    for i in 0..3 {
        assert!(
            vanishes_at_points(av.get(i, 0), &[&t]),
            "(Av)[{i}] = {}",
            av.get(i, 0)
        );
    }
}

#[test]
fn nullspace_of_a_rank_one_symbolic_matrix_is_in_the_kernel() {
    // Hunter seed (smatrix): u·vᵀ with polynomial entries, generic rank 1
    // (identity: every 2×2 minor of an outer product is 0).  Before: rank
    // 3 and null-space vectors with A·v ≠ 0.
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let u = [&t * 1 - 1, ctx.int(2), -(&t * 1) - 1];
    let v = [&t * 2 + 1, -(&t * 2) + 3, ctx.int(2), t.clone()];
    let rows: Vec<Vec<Ex>> = u
        .iter()
        .map(|a| v.iter().map(|b| (a * b).expand()).collect())
        .collect();
    let m = Matrix::new(rows).unwrap();
    assert_eq!(m.rank(), 1);
    let ns = m.nullspace();
    assert_eq!(ns.len(), 3);
    for w in &ns {
        let aw = m.matmul(w).unwrap();
        for i in 0..3 {
            assert!(
                vanishes_at_points(aw.get(i, 0), &[&t]),
                "(Aw)[{i}] = {}",
                aw.get(i, 0)
            );
        }
    }
}

#[test]
fn generic_symbolic_rank_is_fast_and_full() {
    // A 6×6 matrix of independent symbols has rank 6 (its determinant is a
    // nonzero polynomial); the exact pivot test must stay cheap on the
    // nested fractions of the elimination.
    let ctx = Context::new();
    let rows: Vec<Vec<Ex>> = (0..6)
        .map(|i| (0..6).map(|j| ctx.symbol(&format!("a{i}{j}"))).collect())
        .collect();
    let m = Matrix::new(rows).unwrap();
    let t0 = std::time::Instant::now();
    assert_eq!(m.rank(), 6);
    assert!(t0.elapsed().as_secs() < 20, "{:?}", t0.elapsed());
}

// ═══════════════════════════════════════════════════════════════════════════
// linsolve with symbolic coefficients
// ═══════════════════════════════════════════════════════════════════════════

fn system(ctx: &Context, n: usize, eqs: &[&str]) -> (Vec<Ex>, Vec<Ex>) {
    let xs: Vec<Ex> = (0..n).map(|i| ctx.symbol(&format!("x{i}"))).collect();
    let es = eqs.iter().map(|s| ctx.parse(s).unwrap()).collect();
    (es, xs)
}

#[test]
fn linsolve_rank_deficient_symbolic_system_is_parametric() {
    // Hunter linsys seed 4189: equation 4 is −2 × equation 1.  SymPy
    // `linsolve(S1, [x0, x1, x2, x3])` is parametric in x3.  Before: a
    // "unique" solution of NaNs (division by an identically-zero pivot).
    let ctx = Context::new();
    let p = ctx.symbol("p");
    let (eqs, xs) = system(
        &ctx,
        4,
        &[
            "p*x0 + x1*(-p - 3/2) + 4*x2 - 4*x3 - 3",
            "5*x0 + x1*(-2*p - 9/4) - 5*x2 + 9/5*x3 + 4",
            "-7/2*x0 + 2*x1 - x2 + 2*x3 + 1",
            "-2*p*x0 + x1*(2*p + 3) - 8*x2 + 8*x3 + 6",
        ],
    );
    match linsolve(&eqs, &xs).unwrap() {
        LinearSolution::Parametric { solution, free } => {
            assert_eq!(free, vec![xs[3].clone()]);
            let pairs: Vec<(&Ex, &Ex)> = solution.iter().map(|(v, e)| (v, e)).collect();
            for e in &eqs {
                assert!(
                    vanishes_at_points(&e.subs_map(&pairs), &[&p, &xs[3]]),
                    "{e}"
                );
            }
        }
        other => panic!("expected a parametric solution, got {other:?}"),
    }
}

#[test]
fn linsolve_consistent_overdetermined_symbolic_system_is_unique() {
    // Hunter linsys seed 7301: equation 4 is −1 × equation 1.  SymPy
    // `linsolve(S2, [x0, x1, x2])` at p = 2 gives (1101/646, 1155/646,
    // 1053/1292).  Before: reported inconsistent.
    let ctx = Context::new();
    let p = ctx.symbol("p");
    let (eqs, xs) = system(
        &ctx,
        3,
        &[
            "-2*p + x0*(-p + 5) - 1/6*x1 - x2",
            "x0*(p + 2/3) - 5/2*x1 - 5*x2 + 4",
            "x0*(2*p + 1) - 4*x1 + 2*x2 - 3",
            "2*p + x0*(p - 5) + 1/6*x1 + x2",
        ],
    );
    match linsolve(&eqs, &xs).unwrap() {
        LinearSolution::Unique(pairs) => {
            let want = [q(1101, 646), q(1155, 646), q(1053, 1292)];
            for ((_, v), w) in pairs.iter().zip(want) {
                assert_eq!(v.subs(&p, &ctx.int(2)).as_rational(), Some(w), "{v}");
            }
        }
        other => panic!("expected a unique solution, got {other:?}"),
    }
}

#[test]
fn linsolve_inconsistent_symbolic_system_is_inconsistent() {
    // Hunter linsys seed 18461: equation 1 + equation 3 is `−1 = 0`.
    // SymPy `linsolve(S3, [x0, x1, x2, x3])` = EmptySet.  Before: a
    // "parametric" solution x0 = x1 = x2 = 0 that satisfies no equation.
    let ctx = Context::new();
    let (eqs, xs) = system(
        &ctx,
        4,
        &[
            "x0*(-2*p - 5) + x1 + x2*(2*p - 5) - 5*x3 - 8",
            "2*p*x0 + 4*x1 + 2*x3 - 3/2",
            "x0*(2*p + 5) - x1 + x2*(-2*p + 5) + 5*x3 + 7",
        ],
    );
    assert!(matches!(
        linsolve(&eqs, &xs).unwrap(),
        LinearSolution::Inconsistent
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// Inequalities: `≠` and `¬` keep the domain
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn not_equal_excludes_the_poles() {
    // SymPy `reduce_inequalities([Ne((x + 2)/(x - 4), 0)], x)` =
    // (x < −2) ∨ (−2 < x < 4) ∨ (4 < x).  Before, `≠` was the complement
    // of the `=` set and contained the pole x = 4.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ((&x + 2) / (&x - 4))
        .ne_expr(&ctx.zero())
        .solve_for(&x)
        .unwrap();
    assert_eq!(s.to_string(), "(-oo, -2) ∪ (-2, 4) ∪ (4, oo)");
    assert_eq!(s.contains(&ctx.int(4)), Some(false));
    assert_eq!(s.contains(&ctx.int(0)), Some(true));
}

#[test]
fn not_equal_keeps_the_domain_of_a_square_root() {
    // SymPy `solveset(Ne(sqrt(x), 1), x, S.Reals)` = [0, 1) ∪ (1, ∞).
    // Before: the complement of {1}, negative numbers included.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.sqrt().ne_expr(&ctx.one()).solve_for(&x).unwrap();
    assert_eq!(s.to_string(), "[0, 1) ∪ (1, oo)");
    assert_eq!(s.contains(&ctx.int(-1)), Some(false));
}

#[test]
fn negation_is_the_negated_relation() {
    // SymPy: `Not(1/x > 0)` evaluates to `1/x <= 0`, and
    // `solve_univariate_inequality(Not(1/x > 0), x)` = (−∞, 0).  Before,
    // `¬` was the set complement and contained the pole 0.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = (ctx.one() / &x)
        .gt(&ctx.zero())
        .not()
        .solve_for(&x)
        .unwrap();
    assert_eq!(s.to_string(), "(-oo, 0)");
    // De Morgan through `And` (identity): ¬(x ≥ 0 ∧ x < 3) = (x < 0) ∨ (x ≥ 3).
    let s = x
        .ge(&ctx.zero())
        .and(&x.lt(&ctx.int(3)))
        .not()
        .solve_for(&x)
        .unwrap();
    assert_eq!(s.to_string(), "(-oo, 0) ∪ [3, oo)");
}

#[test]
fn equation_of_a_rational_function_without_roots_is_empty() {
    // SymPy `solveset(Eq(-1/(x**2 + 7*x + 12), 0), x, S.Reals)` =
    // EmptySet.  Before: "could not solve the equation".
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.int(-1) / (x.powi(2) + &x * 7 + 12);
    let s = f.eq_expr(&ctx.zero()).solve_for(&x).unwrap();
    assert_eq!(s.is_empty(), Some(true), "{s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// LDLᵀ with a zero pivot that needs no division
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ldl_accepts_a_zero_pivot_over_a_zero_column() {
    // `[[0]] = 1·0·1` and `[[1, 1], [1, 1]] = L·diag(1, 0)·Lᵀ` with
    // L = [[1, 0], [1, 1]] (identity A = L·D·Lᵀ, checked below).  Before:
    // "zero pivot … needs pivoting", although nothing is divided by it.
    let ctx = Context::new();
    for a in [
        Matrix::from_i64(&ctx, &[&[0]]).unwrap(),
        Matrix::from_i64(&ctx, &[&[1, 1], &[1, 1]]).unwrap(),
        Matrix::from_i64(&ctx, &[&[2, 0, 2], &[0, 0, 0], &[2, 0, 3]]).unwrap(),
    ] {
        let Ldl { l, d } = a.ldl().unwrap();
        let back = (&(&l * &d) * &l.transpose()).eval();
        assert_eq!(back, a, "L = {l}, D = {d}");
    }
    // A zero pivot with a nonzero entry below it still needs pivoting.
    assert!(
        Matrix::from_i64(&ctx, &[&[0, 1], &[1, 0]])
            .unwrap()
            .ldl()
            .is_err()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Multivariate GCD
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gcd_of_four_variable_polynomials_finds_the_common_factor() {
    // f = 996·(7z + 34)(47z + 83)(165w + 13x + z + 49) and
    // g = −1464·(7z + 34)(2w² + x³ − 100x + yz − 11) (SymPy `factor`).
    // SymPy `gcd(f, g)` = 84z + 408.  Before, the heuristic GCD failed at
    // every evaluation point for four variables (a recursion-depth guard
    // counted the variables left, not the variables eliminated) and
    // returned 1; `lcm_all` was then f·g.
    let ctx = Context::new();
    let f = ctx
        .parse(
            "4259892*x*z^2 + 28213692*x*z + 36539256*x + 327684*z^3 + 54067860*z^2*w \
             + 18226800*z^2 + 358096860*z*w + 109154628*z + 463767480*w + 137724888",
        )
        .unwrap();
    let g = ctx
        .parse(
            "-10248*x^3*z - 49776*x^3 + 1024800*x*z + 4977600*x - 10248*y*z^2 - 49776*y*z \
             - 20496*z*w^2 + 112728*z - 99552*w^2 + 547536",
        )
        .unwrap();
    let z = ctx.symbol("z");
    assert_eq!(f.gcd_all(&g).unwrap(), &z * 84 + 408);
    // lcm·gcd = f·g up to sign (identity): total degree 3 + 4 − 1 = 6.
    let l = f.lcm_all(&g).unwrap();
    let (x, y, w) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("w"));
    let p = l.as_poly(&[&x, &y, &z, &w]).unwrap();
    assert_eq!(p.total_degree(), Some(6), "lcm = {l}");
}

#[test]
fn polynomial_system_with_radical_solutions_does_not_hang() {
    // Hunter groebner seed 26126.  The eliminant 108z⁶ − 540z⁵ + … factors
    // as (3z² − 4z + 2)(36z⁴ − 132z³ + 169z² − 115z + 16) (SymPy `factor`);
    // the system has 6 solutions (the number of distinct roots of the
    // eliminant of a generic linear form, SymPy `groebner`), and every
    // tuple satisfies the equations.  Before, back-substitution simplified
    // expressions with more than three radical generators through the
    // failing heuristic GCD and did not finish in a minute.
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let eqs = [
        ctx.parse("-2*x*y + 2*y*z - 3*z^2 + 4*z - 2").unwrap(),
        ctx.parse("-4*x*y + 3*x").unwrap(),
        ctx.parse("x^2 - 4*y*z").unwrap(),
    ];
    let t0 = std::time::Instant::now();
    let sols = symplex::polysys::solve_system_ex(&eqs, &[x.clone(), y.clone(), z.clone()]).unwrap();
    assert!(t0.elapsed().as_secs() < 30, "{:?}", t0.elapsed());
    assert_eq!(sols.len(), 6);
    for s in &sols {
        for e in &eqs {
            let r = e.subs_map(&[(&x, &s[0]), (&y, &s[1]), (&z, &s[2])]);
            let v = r.eval_complex64().unwrap();
            assert!(v.norm() < 1e-9, "{e} at {s:?} = {v}");
        }
    }
}
