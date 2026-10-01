//! `solve` and eigenvalues over algebraic coefficients `K = ℚ(√p…, i)`:
//! Cardano for cubics and SymPy's `roots_quartic` (Ferrari / Descartes–
//! Euler) for quartics, exact in `K`.

use num_complex::Complex64;
use std::f64::consts::SQRT_2;
use symplex::prelude::*;

/// Each of `got` is within `1e-13` of a distinct member of `want`.
fn same_roots(got: &[Ex], want: &[(f64, f64)], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: {got:?}");
    let mut pool: Vec<Complex64> = want
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    for r in got {
        let z = r.eval_complex64().unwrap();
        let (k, d) = pool
            .iter()
            .enumerate()
            .map(|(k, w)| (k, (w - z).norm()))
            .fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
        assert!(d < 1e-13, "{what}: root {r} = {z} is not one of {pool:?}");
        pool.remove(k);
    }
}

/// `p(r) = 0` to 30 digits for every root `r` (a zero to the precision of
/// the deep zero search).
fn roots_verify(p: &Ex, x: &Ex, roots: &[Ex]) {
    for r in roots {
        let residual = p.subs(x, r);
        assert_eq!(residual.eval_decimal(30).unwrap(), "0", "{p} at {r}");
    }
}

/// Before: `ComputationFailed { "expression is not polynomial in the given
/// variable and transcendental solver could not find solutions" }` — it is
/// a polynomial; `solve` took degrees 1 and 2 and binomials only when the
/// coefficients are not rational.  SymPy 1.14: `solve(x**3 + sqrt(2)*x + I,
/// x)` gives three radical roots; mpmath `mp.dps=40; polyroots([1, 0,
/// sqrt(2), 1j])` = `±0.40440499519388892257 − 0.72520272166037922084i`,
/// `1.4504054433207584417i`.
#[test]
fn a_cubic_with_algebraic_coefficients_is_solved() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = ctx.parse("x^3 + sqrt(2)*x + I").unwrap();
    let roots = p.solve(&x).unwrap();
    same_roots(
        &roots,
        &[
            (-0.40440499519388895, -0.7252027216603792),
            (0.40440499519388895, -0.7252027216603792),
            (0.0, 1.4504054433207585),
        ],
        "x^3 + sqrt(2)*x + I",
    );
    roots_verify(&p, &x, &roots);
}

/// Before: refused as "not polynomial" (`solve`).  Quartics over `K` in
/// radicals, every case of SymPy's `roots_quartic`: Ferrari (the first and
/// the last), a biquadratic (`f = 0`), and a factor in `K` found by the
/// norm (`x⁴ + √2·x + i` has the root `(1 − i)/√2`).  mpmath `mp.dps=40;
/// polyroots(...)`, rounded to `f64`.
#[test]
fn quartics_with_algebraic_coefficients_are_solved() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: [(&str, [(f64, f64); 4]); 4] = [
        (
            "x^4 - sqrt(2)*x^3 + x^2 + I*x + 5",
            [
                (-0.5823787918452015, 1.028485701063668),
                (-0.7696731949955787, -1.0714784026546893),
                (1.5086200326136483, -1.1148124852325119),
                (1.257645516600227, 1.157805186823533),
            ],
        ),
        (
            "x^4 + sqrt(3)*x^2 + 1",
            [
                (-0.25881904510252074, -0.9659258262890683),
                (-0.25881904510252074, 0.9659258262890683),
                (0.25881904510252074, -0.9659258262890683),
                (0.25881904510252074, 0.9659258262890683),
            ],
        ),
        (
            "x^4 + I*x^3 + sqrt(2)",
            [
                (-0.7324003236470864, 0.5783629796903536),
                (0.7324003236470864, 0.5783629796903536),
                (-0.6789405115064243, -1.0783629796903536),
                (0.6789405115064243, -1.0783629796903536),
            ],
        ),
        (
            "x^4 + (1+I)*x^3 + sqrt(3)*x + 2",
            [
                (-0.8035533374537546, 0.22928873339849404),
                (-1.3488661388390624, -0.8672337778005427),
                (0.6806600047581207, 0.8868637206949264),
                (0.4717594715346962, -1.2489186762928777),
            ],
        ),
    ];
    for (s, want) in cases {
        let p = ctx.parse(s).unwrap();
        let roots = p.solve(&x).unwrap();
        same_roots(&roots, &want, s);
        roots_verify(&p, &x, &roots);
    }
}

/// The other cases of `roots_quartic`: quasi-symmetric (`(c/a)² = d`:
/// `c/a = i`, `d = −1`) and Descartes–Euler (the resolvent `64R³ + 32√3·R²
/// + (12 − 16(65/4 + 4√3))·R − 16` has the rational root `R = 2`).
/// Before: refused as "not polynomial".  mpmath `mp.dps=40; polyroots(...)`.
#[test]
fn quasi_symmetric_and_descartes_euler_quartics() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: [(&str, [(f64, f64); 4]); 2] = [
        (
            "x^4 + sqrt(2)*x^3 + x^2 + sqrt(2)*I*x - 1",
            [
                (0.6541113369834176, -0.2295041938920957),
                (-1.3612181181699652, -0.4776025872944518),
                (-0.2295041938920957, -0.6541113369834176),
                (-0.4776025872944518, 1.3612181181699652),
            ],
        ),
        (
            "x^4 + sqrt(3)*x^2 + 4*x + 65/4 + 4*sqrt(3)",
            // The real parts are ±√2.
            [
                (-SQRT_2, 1.4693259075500884),
                (-SQRT_2, -1.4693259075500884),
                (SQRT_2, -1.8902730450839598),
                (SQRT_2, 1.8902730450839598),
            ],
        ),
    ];
    for (s, want) in cases {
        let p = ctx.parse(s).unwrap();
        let roots = p.solve(&x).unwrap();
        same_roots(&roots, &want, s);
        roots_verify(&p, &x, &roots);
    }
}

/// Before: `eigenvals` found no eigenvalue — the characteristic polynomial
/// is an irreducible quartic over `ℚ(√2, √3, i)` and the radical roots
/// stopped at degree 3.  numpy `linalg.eigvals` (and mpmath `mp.dps=40;
/// eig(...)`): `2.824114670913325736 + 0.666657329985900932i`,
/// `0.176243268793561019 + 0.930281898069045357i`,
/// `1.167999819486183942 − 1.138382130399158437i`,
/// `1.563693048375806596 − 0.458557097655787852i`.
#[test]
fn a_four_by_four_algebraic_matrix_has_its_eigenvalues() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let rows = [
        ["1", "sqrt(2)", "0", "I"],
        ["0", "2", "I", "1"],
        ["1", "0", "sqrt(3)", "0"],
        ["I", "1", "0", "1"],
    ];
    let m = Matrix::new(
        rows.iter()
            .map(|r| r.iter().map(|s| p(s)).collect())
            .collect(),
    )
    .unwrap();
    let ev = m.eigenvals().unwrap();
    same_roots(
        &ev,
        &[
            (2.8241146709133256, 0.6666573299859009),
            (0.17624326879356103, 0.9302818980690454),
            (1.167999819486184, -1.1383821303991584),
            (1.5636930483758067, -0.45855709765578784),
        ],
        "eigenvals",
    );
    let size: usize = ev.iter().map(|e| e.to_string().len()).sum();
    assert!(size < 50_000, "{size} characters");
}

/// A polynomial whose roots are not found says so; a non-polynomial
/// equation keeps its message.  Before, both said "not polynomial".
#[test]
fn the_error_tells_a_polynomial_from_a_non_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (s, want) in [
        ("x^5 + sqrt(2)*x + 1", "polynomial of degree 5"),
        ("x^3 + pi*x + 1", "polynomial of degree 3"),
        ("exp(x) + x^3 + sqrt(2)", "not polynomial"),
    ] {
        match ctx.parse(s).unwrap().solve(&x) {
            Err(SymplexError::ComputationFailed { reason, .. }) => {
                assert!(reason.contains(want), "{s}: {reason}");
            }
            other => panic!("{s}: {other:?}"),
        }
    }
}
