//! Linear algebra with symbolic and algebraic entries after 0.30 (the
//! differential hunt of `examples/zz_hunt_linalg.rs`), and the equation
//! refusals closed with it.

use symplex::polysys::{LinearSolution, linsolve};
use symplex::prelude::*;

fn c64(e: &Ex) -> Complex64 {
    e.eval_complex64()
        .unwrap_or_else(|err| panic!("{e} does not evaluate: {err}"))
}

fn close(a: Complex64, b: Complex64) -> bool {
    (a - b).norm() <= 1e-12 * b.norm().max(1.0)
}

/// `true` if the two lists hold the same numbers (in any order).
fn same_set(got: &[Ex], want: &[Complex64]) -> bool {
    got.len() == want.len() && want.iter().all(|w| got.iter().any(|g| close(c64(g), *w)))
}

/// A pivot mixing `√2` and a symbol was decided by `simplify`, which did not
/// bring the zero `(a·x + y − 2z − 1) + (√2·x + y + z) − …` to 0: the
/// consistent system was reported "Inconsistent".  Every pivot is now
/// decided exactly over `ℚ(√2)(a)`.
///
/// SymPy: `linsolve([a*x + y - 2*z - 1, sqrt(2)*x + y + z, -2*x + y + z,
/// (a*x+y-2*z-1)+(sqrt(2)*x+y+z)], [x, y, z])` → `{(0, 1/3, -1/3)}`.
#[test]
fn linsolve_with_sqrt2_and_a_symbol_is_consistent() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let eqs = [
        p("a*x + y - 2*z - 1"),
        p("sqrt(2)*x + y + z"),
        p("-2*x + y + z"),
        p("(a*x+y-2*z-1)+(sqrt(2)*x+y+z)"),
    ];
    let vars = [p("x"), p("y"), p("z")];
    match linsolve(&eqs, &vars).unwrap() {
        LinearSolution::Unique(pairs) => {
            let got: Vec<String> = pairs.iter().map(|(_, v)| v.to_string()).collect();
            assert_eq!(got, ["0", "1/3", "-1/3"]);
        }
        other => panic!("expected the unique solution, got {other:?}"),
    }
}

/// A 4×4 system over `ℚ(√2)(a, b)` took over 8 s (release) and returned
/// nested unsimplified quotients; it is now solved by fraction-free
/// elimination over `ℚ(√2)[a, b]` (tens of milliseconds in debug).
///
/// SymPy (219 s): `linsolve([a*x+(a+1)*y+(2*a-1)*z+3*w-(a-b),
/// 2*x+sqrt(2)*y-z+2*w+4, 2*x+2*y+(a+1)*z-w-3,
/// a**2*x+(a-b)*z+(a-b)*w-(2*a-1)], [x, y, z, w])`, substituted at
/// `a = 3/7, b = -5/11`: `x = -2964464727623/1786152503069 -
/// 1572924044328*sqrt(2)/1786152503069`, `y = 1042405760487*sqrt(2)/
/// 1786152503069 + 4206373086864/1786152503069`, `z = 571602843084*sqrt(2)/
/// 1786152503069 + 119870814854/162377500279`, `w = -90086959755/
/// 162377500279 - 244461077562*sqrt(2)/1786152503069`.
#[test]
fn linsolve_four_by_four_over_sqrt2_and_two_parameters() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let eqs = [
        p("a*x+(a+1)*y+(2*a-1)*z+3*w-(a-b)"),
        p("2*x+sqrt(2)*y-z+2*w+4"),
        p("2*x+2*y+(a+1)*z-w-3"),
        p("a^2*x+(a-b)*z+(a-b)*w-(2*a-1)"),
    ];
    let vars = [p("x"), p("y"), p("z"), p("w")];
    let LinearSolution::Unique(pairs) = linsolve(&eqs, &vars).unwrap() else {
        panic!("expected a unique solution");
    };
    let want = [
        "-2964464727623/1786152503069 - 1572924044328*sqrt(2)/1786152503069",
        "1042405760487*sqrt(2)/1786152503069 + 4206373086864/1786152503069",
        "571602843084*sqrt(2)/1786152503069 + 119870814854/162377500279",
        "-90086959755/162377500279 - 244461077562*sqrt(2)/1786152503069",
    ];
    for ((v, value), w) in pairs.iter().zip(want) {
        let at = value.subs(&p("a"), &p("3/7")).subs(&p("b"), &p("-5/11"));
        // The difference is 0 to the precision of the zero search.
        let diff = (&at - &p(w)).eval_decimal(40).unwrap();
        assert_eq!(diff, "0", "{v} = {value}: at the point {at}");
    }
}

/// `rank`, `rref` and `nullspace` decided a pivot with radicals only
/// structurally: row 2 is `√3·(−1)·` row 1 through `√6 = √2·√3`.
///
/// SymPy: `Matrix([[sqrt(2)-3, -2], [-sqrt(6)+3*sqrt(3), 2*sqrt(3)]]).rank()`
/// → 1; `Matrix([[1, sqrt(2), a], [sqrt(2), 2, sqrt(2)*a]]).rank()` → 1
/// with a two-vector null space.
#[test]
fn rank_of_matrices_dependent_through_radicals() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let m = Matrix::new(vec![
        vec![p("sqrt(2) - 3"), p("-2")],
        vec![p("-sqrt(6) + 3*sqrt(3)"), p("2*sqrt(3)")],
    ])
    .unwrap();
    assert_eq!(m.rank(), 1);
    let m = Matrix::new(vec![
        vec![p("1"), p("sqrt(2)"), p("a")],
        vec![p("sqrt(2)"), p("2"), p("sqrt(2)*a")],
    ])
    .unwrap();
    assert_eq!(m.rank(), 1);
    let ns = m.nullspace();
    assert_eq!(ns.len(), 2);
    for v in &ns {
        let r = (&m * v).expand().simplify();
        assert!(r.iter().all(Ex::is_zero_structural), "A·v = {r}");
    }
}

/// `inv` tested the determinant `2a − 2a` of a matrix with `√2` and a
/// symbol by `simplify` only, did not see the zero, and divided by it.
///
/// SymPy: `Matrix([[sqrt(2), 2], [a, sqrt(2)*a]]).inv()` raises
/// `NonInvertibleMatrixError`.
#[test]
fn inverse_of_a_singular_matrix_with_radicals_is_refused() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let m = Matrix::new(vec![
        vec![p("sqrt(2)"), p("2")],
        vec![p("a"), p("sqrt(2)*a")],
    ])
    .unwrap();
    assert!(m.inv().is_err());
}

/// Complex triangular matrices: the eigenvalues came from the quadratic
/// formula as `√(−3 − 4i)/2 + 1/2 + i` and `√(2i)/2 + i/2 − 1/2`, and no
/// zero pivot of `A − λI` could be decided (no eigenvectors at all).  The
/// block-triangular split gives the diagonal entries, as SymPy.
///
/// SymPy: `Matrix([[1, 0], [0, 2*I]]).eigenvects()` →
/// `[(1, 1, [[1, 0]]), (2*I, 1, [[0, 1]])]`;
/// `Matrix([[-1, 0], [1+I, I]]).eigenvects()` →
/// `[(-1, 1, [[-1, 1]]), (I, 1, [[0, 1]])]`.
#[test]
fn eigenvects_of_complex_triangular_matrices() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    for (rows, want) in [
        ([["1", "0"], ["0", "2*I"]], ["1", "2*I"]),
        ([["-1", "0"], ["1+I", "I"]], ["-1", "I"]),
    ] {
        let m = Matrix::new(
            rows.iter()
                .map(|r| r.iter().map(|s| p(s)).collect())
                .collect(),
        )
        .unwrap();
        let ev = m.eigenvects().unwrap();
        let vals: Vec<String> = ev.iter().map(|(v, _, _)| v.to_string()).collect();
        assert_eq!(vals, want);
        for (lam, mult, vecs) in &ev {
            assert_eq!((*mult, vecs.len()), (1, 1), "λ = {lam}");
            let r = (&(&m * &vecs[0]) - &vecs[0].scale(lam)).expand().simplify();
            assert!(r.iter().all(Ex::is_zero_structural), "λ = {lam}: {r}");
        }
    }
}

/// A non-triangular complex 2×2: the eigenvalues are nested radicals
/// `(−3 − i ± √(−28 − 6i))/2`, and the zero test of the null-space
/// elimination now decides `P + Q·√D` exactly (`P² = Q²D` and the
/// principal sign).  Before, `A − λI` had full rank: no eigenvectors.
///
/// SymPy: `[N(v, 20) for v in Matrix([[-3, -3], [3, -I]]).eigenvals()]` →
/// `[-1.7818781045400445583 + 2.1607245753401625408*I,
/// -1.2181218954599554417 - 3.1607245753401625408*I]`.
#[test]
fn eigenvects_with_nested_complex_radicals() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let m = Matrix::new(vec![vec![p("-3"), p("-3")], vec![p("3"), p("-I")]]).unwrap();
    let ev = m.eigenvects().unwrap();
    let vals: Vec<Ex> = ev.iter().map(|(v, _, _)| v.clone()).collect();
    let want = [
        Complex64::new(-1.781_878_104_540_044_6, 2.160_724_575_340_162_5),
        Complex64::new(-1.218_121_895_459_955_4, -3.160_724_575_340_162_5),
    ];
    assert!(same_set(&vals, &want), "{vals:?}");
    for (lam, _, vecs) in &ev {
        assert_eq!(vecs.len(), 1, "λ = {lam}");
        let (l, v0, v1) = (c64(lam), c64(vecs[0].get(0, 0)), c64(vecs[0].get(1, 0)));
        let r0 = Complex64::new(-3.0, 0.0) * v0 - 3.0 * v1 - l * v0;
        let r1 = 3.0 * v0 - Complex64::new(0.0, 1.0) * v1 - l * v1;
        assert!(r0.norm() < 1e-12 && r1.norm() < 1e-12, "λ = {lam}");
    }
}

/// The eigenvector of the eigenvalue `≈ −10⁻⁴⁰` was the expression of the
/// elimination on `A − λI` (nested quotients cancelling 40 digits), and
/// `eval_complex64` of its entries failed with `PrecisionExhausted`.  The
/// entries are now read from the exact elimination in `ℚ[t]/(g)`:
/// `c₀ + c₁λ + c₂λ²`, which evaluate directly.
///
/// SymPy: `[N(e, 30) for e in v]` for the eigenvector of that eigenvalue of
/// `Matrix([[3,-3,3],[6+Rational(1,10**39),-6,6],[5,4,-3]])` →
/// `[-0.111111111111111111111111111111, 0.888888888888888888888888888889, 1]`.
#[test]
fn eigenvector_of_a_tiny_eigenvalue_evaluates() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let m = Matrix::new(vec![
        vec![p("3"), p("-3"), p("3")],
        vec![p("6 + 10^(-39)"), p("-6"), p("6")],
        vec![p("5"), p("4"), p("-3")],
    ])
    .unwrap();
    let ev = m.eigenvects().unwrap();
    let (_, _, vecs) = ev
        .iter()
        .find(|(l, _, _)| c64(l).norm() < 1e-30)
        .expect("the eigenvalue ≈ −1e-40");
    let got: Vec<Complex64> = (0..3).map(|i| c64(vecs[0].get(i, 0))).collect();
    let want = [-1.0 / 9.0, 8.0 / 9.0, 1.0];
    for (g, w) in got.iter().zip(want) {
        assert!(close(*g, Complex64::new(w, 0.0)), "{got:?}");
    }
}

/// `Q·R` was `−A` in a column whose bilinear norm² is negative: `√(1/n)` is
/// `−1/√n` for `n < 0`.
///
/// SymPy: `Q, R = Matrix([[0, -2/5], [1/2, -3], [-I, 2*I]]).QRdecomposition();
/// simplify(Q*R)` → the matrix itself.
#[test]
fn qr_reproduces_a_complex_matrix() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let a = Matrix::new(vec![
        vec![p("0"), p("-2/5")],
        vec![p("1/2"), p("-3")],
        vec![p("-I"), p("2*I")],
    ])
    .unwrap();
    let symplex::decompositions::Qr { q, r } = a.qr().unwrap();
    let qr = &q * &r;
    for i in 0..3 {
        for j in 0..2 {
            assert!(
                close(c64(qr.get(i, j)), c64(a.get(i, j))),
                "({i},{j}): {qr}"
            );
        }
    }
}

/// `lu` chose pivots structurally: the entry `1/(√2 − 1) − √2 − 1` (zero)
/// was a pivot and `L` held `5/0`.  Now the rows are swapped past it.
///
/// SymPy: `Matrix([[1, 2, 3], [sqrt(2), 2*sqrt(2) + 1/(sqrt(2) - 1) -
/// sqrt(2) - 1, 5], [1, 7, 1]]).LUdecomposition()` → the row swap
/// `[[1, 2]]`, `U = [[1, 2, 3], [0, 5, -2], [0, 0, …]]` (the last entry is
/// `5 − 3√2`).
#[test]
fn lu_skips_a_hidden_zero_pivot() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let a = Matrix::new(vec![
        vec![p("1"), p("2"), p("3")],
        vec![
            p("sqrt(2)"),
            p("2*sqrt(2) + 1/(sqrt(2) - 1) - sqrt(2) - 1"),
            p("5"),
        ],
        vec![p("1"), p("7"), p("1")],
    ])
    .unwrap();
    let symplex::decompositions::Lu { l, u, perm } = a.lu().unwrap();
    assert_eq!(perm, [0, 2, 1]);
    assert_eq!(u.get(1, 1).to_string(), "5");
    let prod = &l * &u;
    for (i, &row) in perm.iter().enumerate() {
        for j in 0..3 {
            let want = c64(a.get(row, j));
            assert!(close(c64(prod.get(i, j)), want), "({i},{j}): L = {l}");
        }
    }
}

/// `harmonic(n)` at a negative integer (a pole of `ψ(n + 1) + γ`) stayed
/// unevaluated.  SymPy: `harmonic(-1)` → `zoo`, `harmonic(-5)` → `zoo`.
#[test]
fn harmonic_at_negative_integers_is_complex_infinity() {
    let ctx = Context::new();
    for n in [-1, -2, -5] {
        let h = ctx.int(n).harmonic().eval();
        assert_eq!(h.to_string(), "zoo", "harmonic({n})");
    }
    assert_eq!(ctx.int(3).harmonic().eval().to_string(), "11/6");
}

/// Equations `solve` refused ("transcendental solver could not find
/// solutions"): a sum of sine and cosine (half-angle substitution),
/// exponentials of one rate, exponentials with a common base, a multiple
/// angle, a sum of logarithms, a rational equation.
///
/// SymPy 1.14 `solve(e, x)`: `sin(x)+cos(x)-1` → `[0, pi/2]`;
/// `sin(x)+cos(x)-1/2` → `[2*atan(2/3 - sqrt(7)/3), 2*atan(2/3 + sqrt(7)/3)]`;
/// `exp(x)+exp(-x)-3` → `[log(3/2 - sqrt(5)/2), log(sqrt(5)/2 + 3/2)]`;
/// `exp(x)+exp(-x)-1` → `[-I*pi/3, I*pi/3]`;
/// `2**(2*x)-5*2**x+6` → `[1, log(3)/log(2)]`;
/// `3**x+9**x-12` → `[1, (log(4) + I*pi)/log(3)]`;
/// `log(x)+log(x+1)-log(6)` → `[2]`; `x+1/x-3` → `[3/2 - sqrt(5)/2,
/// sqrt(5)/2 + 3/2]`.
#[test]
fn solve_trig_exponential_log_and_rational_equations() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s2 = 2f64.sqrt();
    let s3 = 3f64.sqrt();
    let s5 = 5f64.sqrt();
    let s7 = 7f64.sqrt();
    let pi = std::f64::consts::PI;
    let re = |v: f64| Complex64::new(v, 0.0);
    let cases: Vec<(&str, Vec<Complex64>)> = vec![
        ("sin(x) + cos(x) - 1", vec![re(0.0), re(pi / 2.0)]),
        (
            "sin(x) + cos(x) - 1/2",
            vec![
                re(2.0 * (2.0 / 3.0 - s7 / 3.0).atan()),
                re(2.0 * (2.0 / 3.0 + s7 / 3.0).atan()),
            ],
        ),
        (
            "exp(x) + exp(-x) - 3",
            vec![re((1.5 - s5 / 2.0).ln()), re((1.5 + s5 / 2.0).ln())],
        ),
        (
            "exp(x) + exp(-x) - 1",
            vec![
                Complex64::new(0.0, -pi / 3.0),
                Complex64::new(0.0, pi / 3.0),
            ],
        ),
        (
            "2^(2*x) - 5*2^x + 6",
            vec![re(1.0), re(3f64.ln() / 2f64.ln())],
        ),
        (
            "3^x + 9^x - 12",
            vec![re(1.0), Complex64::new(4f64.ln(), pi) / 3f64.ln()],
        ),
        ("log(x) + log(x + 1) - log(6)", vec![re(2.0)]),
        ("x + 1/x - 3", vec![re(1.5 - s5 / 2.0), re(1.5 + s5 / 2.0)]),
        ("2*sin(x) + 3*cos(x) - 1", {
            let t = |v: f64| re(2.0 * v.atan());
            vec![t(0.5 - s3 / 2.0), t(0.5 + s3 / 2.0)]
        }),
        ("sin(x) + cos(x) - sqrt(2)", vec![re(pi / 4.0)]),
    ];
    let _ = s2;
    for (src, want) in cases {
        let e = ctx.parse(src).unwrap();
        let got = e.solve(&x).unwrap_or_else(|err| panic!("{src}: {err}"));
        assert!(same_set(&got, &want), "{src}: {got:?}");
    }
}

/// The half-angle substitution does not reach `t = tan(x/2) = ∞`; `x = π`
/// is added when the equation holds there.  SymPy misses it:
/// `solve(sin(2*x) - sin(x))` → `[0, -pi/3, pi/3]` and
/// `solve(sin(x) + cos(x) + 1)` → `[-pi/2]`, while `sin 2π − sin π = 0` and
/// `sin π + cos π + 1 = 0`.
#[test]
fn half_angle_substitution_keeps_the_root_pi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    let re = |v: f64| Complex64::new(v, 0.0);
    for (src, want) in [
        (
            "sin(2*x) - sin(x)",
            vec![re(0.0), re(-pi / 3.0), re(pi / 3.0), re(pi)],
        ),
        ("sin(x) + cos(x) + 1", vec![re(-pi / 2.0), re(pi)]),
    ] {
        let got = ctx.parse(src).unwrap().solve(&x).unwrap();
        assert!(same_set(&got, &want), "{src}: {got:?}");
    }
}
