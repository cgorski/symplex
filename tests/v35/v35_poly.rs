//! Polynomial algebra: the 0.39 hunt for silent wrong answers.  Each test
//! says what was wrong before and cites its oracle: an exact identity
//! (the factors multiplied back, the solutions substituted back), SymPy
//! 1.14 (`factor_list`, `groebner`), or the number of distinct solutions
//! of a zero-dimensional system (the quotient dimension of its radical).

use symplex::num_complex::Complex64;
use symplex::polysys::solve_system_ex;
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `content · ∏ fᵢ^mᵢ` expands to `e`.
fn assert_product(e: &Ex, content: &Ex, factors: &[(Ex, u32)]) {
    let mut prod = content.clone();
    for (f, m) in factors {
        prod = &prod * &f.powi(i64::from(*m));
    }
    assert!(
        (&prod - e).expand().is_zero_structural(),
        "{content} * {factors:?} is not {e}"
    );
}

/// Every tuple satisfies every equation (residual below `1e-9` at 16
/// digits), and the tuples are pairwise distinct numerically.
fn assert_solutions(eqs: &[Ex], vars: &[Ex], sols: &[Vec<Ex>]) {
    let mut points: Vec<Vec<Complex64>> = Vec::new();
    for s in sols {
        let pairs: Vec<(&Ex, &Ex)> = vars.iter().zip(s.iter()).collect();
        for e in eqs {
            let r = e.subs_map(&pairs).eval_complex64().unwrap();
            assert!(r.norm() < 1e-9, "{e} at {s:?} is {r}");
        }
        let p: Vec<Complex64> = s.iter().map(|v| v.eval_complex64().unwrap()).collect();
        assert!(
            !points
                .iter()
                .any(|q| q.iter().zip(&p).all(|(a, b)| (a - b).norm() < 1e-9)),
            "duplicate solution {s:?}"
        );
        points.push(p);
    }
}

/// `factor_list_all` (and `factor_all`) left a reducible polynomial whole
/// when its Kronecker image had degree above 96 — the only multivariate
/// method: the product
/// `(x²z + 5xz + 8z − 4)(7x²y + 8yz − z − 9)²(7x² − x + 6z² − 6z + 3)`
/// was returned as one factor of multiplicity 1, and so was
/// `(9x − 7y − z − 1)²(7y − 9)(7xy − z + 1)²`.  Now evaluation, Hensel
/// lifting and recombination take over (SymPy 1.14 `factor_list` gives
/// the same factors).
#[test]
fn multivariate_factorization_beyond_the_kronecker_bound() {
    let ctx = Context::new();
    let cases: [(&str, usize, u32); 3] = [
        (
            "(x^2*z + 5*x*z + 8*z - 4)*(7*x^2*y + 8*y*z - z - 9)^2*(7*x^2 - x + 6*z^2 - 6*z + 3)",
            3,
            4,
        ),
        ("(9*x - 7*y - z - 1)^2*(7*y - 9)*(7*x*y - z + 1)^2", 3, 5),
        ("(3*x - 17)*(z + 6)^2*(x^8 - y^6)", 4, 5),
    ];
    for (src, distinct, total) in cases {
        let e = parse(&ctx, src).expand();
        let (c, fs) = e.factor_list_all();
        assert_product(&e, &c, &fs);
        assert_eq!(fs.len(), distinct, "{src}: {fs:?}");
        assert_eq!(
            fs.iter().map(|(_, m)| m).sum::<u32>(),
            total,
            "{src}: {fs:?}"
        );
        for (f, _) in &fs {
            let (_, sub) = f.factor_list_all();
            assert_eq!(sub.len(), 1, "{f} is reducible: {sub:?}");
        }
        let f = e.factor_all();
        assert!((&f - &e).expand().is_zero_structural());
    }
}

/// `solve_system_ex` returned `Ok([])` — "no solution" — for systems whose
/// back-substitution meets a fibre polynomial with algebraic coefficients
/// the univariate solver cannot solve: `y² = 2, x³ + yx + 1 = 0` has 6
/// solutions (`res_y = (x³ + 1)² − 2x²` is square-free of degree 6),
/// `y² = 2, x⁵ + yx + 1 = 0` 10, `z² = 3, y² = z, x² + xy + 1 = 0` 8.  The
/// number of distinct solutions is now computed exactly (the dimension of
/// the quotient by the radical) and, when back-substitution falls short,
/// every solution comes from the rational univariate representation.
#[test]
fn polynomial_systems_with_algebraic_fibres_are_solved_completely() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let cases: [(&[&str], usize, usize); 4] = [
        (&["y^2 - 2", "x^3 + y*x + 1"], 2, 6),
        (&["y^2 - 2", "x^5 + y*x + 1"], 2, 10),
        (&["y^2 + 1", "x^5 - x - y"], 2, 10),
        (&["z^2 - 3", "y^2 - z", "x^2 + x*y + 1"], 3, 8),
    ];
    for (src, nv, count) in cases {
        let vars: Vec<Ex> = [&x, &y, &z][..nv].iter().map(|v| (*v).clone()).collect();
        let eqs: Vec<Ex> = src.iter().map(|s| parse(&ctx, s)).collect();
        let sols = solve_system_ex(&eqs, &vars).unwrap();
        assert_eq!(sols.len(), count, "{src:?}: {sols:?}");
        assert_solutions(&eqs, &vars, &sols);
    }
}

/// `solve_polynomial_system` (rational solutions) took the roots of the
/// first polynomial of the lex basis in the current variable and ignored
/// the others, so a non-solution came back: `x² − 1, y² − 1, (x − 1)(y −
/// 1)` gave `(−1, −1)` besides the three solutions (at `y = −1` the basis
/// holds `x² − 1` and `−2x + 2`).  Oracle: each solution substituted into
/// every equation, and the three points `(1, 1), (−1, 1), (1, −1)` by
/// hand.
#[test]
fn rational_system_solutions_satisfy_every_equation() {
    use symplex::multipoly::{GrevLex, MultiPoly};
    use symplex::num_rational::Ratio;
    let x: MultiPoly<GrevLex> = MultiPoly::var(2, 0);
    let y: MultiPoly<GrevLex> = MultiPoly::var(2, 1);
    let one = MultiPoly::from_int(2, 1);
    let system = [
        x.mul(&x).sub(&one),
        y.mul(&y).sub(&one),
        x.sub(&one).mul(&y.sub(&one)),
    ];
    let mut sols = symplex::polysys::solve_polynomial_system(&system).unwrap();
    sols.sort();
    let q = |n: i64| Ratio::from_integer(n.into());
    assert_eq!(
        sols,
        vec![vec![q(-1), q(1)], vec![q(1), q(-1)], vec![q(1), q(1)]]
    );
    for s in &sols {
        for p in &system {
            assert_eq!(p.eval(s).unwrap(), q(0));
        }
    }
}

/// The lex Gröbner basis of a unit ideal ran Buchberger in lex order for
/// minutes (over 55 s in a debug build) although the grevlex basis is `[1]`
/// after 0.6 s: FGLM does not apply to the unit ideal (it is not
/// zero-dimensional), and the fallback recomputed it from scratch.  SymPy
/// 1.14 `groebner(F, x, y, z, order='lex')` is `[1]`.
#[test]
fn lex_basis_of_the_unit_ideal_is_immediate() {
    use symplex::multipoly::MonomialOrder;
    let ctx = Context::new();
    let vars = [ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z")];
    let polys: Vec<Ex> = [
        "10*y*z - 8*y - 3*y^4 - 3*z",
        "9*x - 7*y^2 + 1 - 5*x^2*y - y",
        "-2*x*y*z - y^2 - x^3 + 2 - 7*y",
        "x^4 + 6*x + 8*z",
    ]
    .iter()
    .map(|s| parse(&ctx, s))
    .collect();
    let start = std::time::Instant::now();
    let basis = Ex::groebner(&polys, &vars, MonomialOrder::Lex).unwrap();
    assert_eq!(basis, vec![ctx.int(1)]);
    assert!(start.elapsed().as_secs() < 20, "{:?}", start.elapsed());
}

/// A power `x^1000000000` was built by a billion multiplications in the
/// multivariate conversions, so `factor_all`, `gcd_all`, `groebner` and
/// `solve_system_ex` hung; beyond degree 10,000 such a power is now not a
/// polynomial for them, as for the univariate conversion.  `groebner` in
/// lex order of `x⁶⁰ − 1, y⁶⁰ − 1` (already a lex basis) built dense
/// 3600 × 3600 FGLM matrices for over a minute and 2 GB; and
/// `solve_system_ex` of `x¹⁰⁰⁰⁰⁰ = 1, y = 1` ran out of memory.  Oracle:
/// the basis is the input (coprime leading monomials, Buchberger's first
/// criterion), as SymPy 1.14 `groebner` says.
#[test]
fn huge_degrees_do_not_hang() {
    use symplex::multipoly::MonomialOrder;
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let start = std::time::Instant::now();
    let big = parse(&ctx, "x^1000000000 - y^2");
    assert_eq!(big.factor_all(), big);
    assert!(big.gcd_all(&(&x - 1)).is_none());
    let vars = [x.clone(), y.clone()];
    assert!(Ex::groebner(&[big.clone(), &y - 1], &vars, MonomialOrder::GrevLex).is_err());
    assert!(solve_system_ex(&[big, &y - 1], &vars).is_err());
    let f = parse(&ctx, "x^60 - 1");
    let g = parse(&ctx, "y^60 - 1");
    let basis = Ex::groebner(&[f.clone(), g.clone()], &vars, MonomialOrder::Lex).unwrap();
    assert_eq!(basis, vec![f, g]);
    let many = solve_system_ex(&[parse(&ctx, "x^5000 - 1"), &y - 1], &vars);
    assert!(many.is_err(), "{many:?}");
    assert!(start.elapsed().as_secs() < 20, "{:?}", start.elapsed());
}

/// The Swinnerton-Dyer polynomial `SD₅ = ∏(x ± √2 ± √3 ± √5 ± √7 ±
/// √11)` (SymPy `swinnerton_dyer_poly(5)`).
const SD5: &str = "x^32 - 448*x^30 + 84864*x^28 - 9028096*x^26 + 602397952*x^24 \
    - 26625650688*x^22 + 801918722048*x^20 - 16665641517056*x^18 + 239210760462336*x^16 \
    - 2349014746136576*x^14 + 15459151516270592*x^12 - 65892492886671360*x^10 \
    + 172580952324702208*x^8 - 255690851718529024*x^6 + 183876928237731840*x^4 \
    - 44660812492570624*x^2 + 2000989041197056";

/// `is_irreducible` said `Some(true)` for `SD₅(x)·SD₅(x + 1)`: modulo every
/// prime it splits into 32 quadratics, the Zassenhaus recombination budget
/// runs out long before the subsets of size 16 that form the true factors,
/// and the unsplit product was taken as irreducible.  Now `None` (not
/// certified); `SD₅` alone is still certified irreducible.  Oracle: the
/// product of two non-constant polynomials is reducible.
#[test]
fn irreducibility_is_not_claimed_without_a_certificate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sd5 = parse(&ctx, SD5);
    assert_eq!(sd5.is_irreducible(&x), Some(true));
    let product = (&sd5 * &sd5.subs(&x, &(&x + 1))).expand();
    assert_ne!(product.is_irreducible(&x), Some(true));
}

/// Systems the back-substitution solves completely keep their radical
/// forms (the count certificate only adds a check).
#[test]
fn complete_back_substitution_is_unchanged() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let eqs = [&x.powi(2) + &y.powi(2) - 1, &y - &x];
    let sols = solve_system_ex(&eqs, &[x.clone(), y.clone()]).unwrap();
    let h = ctx.int(2).sqrt() / 2;
    assert_eq!(sols, vec![vec![h.clone(), h.clone()], vec![-&h, -&h]]);
}
