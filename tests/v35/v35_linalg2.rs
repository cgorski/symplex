//! Eigenvalues of symbolic 4×4 matrices by Ferrari's formulas, and the
//! principal square root and logarithm of real matrices with a negative
//! eigenvalue in Cardano form (the eigen hunt after 0.44, agent E).  Every
//! reference value cites its oracle: mpmath 1.3 at 50 digits (`polyroots`,
//! `eig`), numpy 2 for the multisets.

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s)
        .unwrap_or_else(|e| panic!("`{s}` does not parse: {e}"))
}

fn matrix(ctx: &Context, rows: &[&[&str]]) -> Matrix {
    Matrix::new(
        rows.iter()
            .map(|r| r.iter().map(|s| p(ctx, s)).collect())
            .collect(),
    )
    .unwrap()
}

/// Every value of `want` is matched by one of `got` (as multisets) to
/// `1e-10` relative.
fn same_values(got: &[Ex], want: &[(f64, f64)], what: &str) {
    let mut z: Vec<num_complex::Complex64> = got
        .iter()
        .map(|e| {
            e.eval_complex64()
                .unwrap_or_else(|err| panic!("{what}: {e} does not evaluate: {err}"))
        })
        .collect();
    assert_eq!(z.len(), want.len(), "{what}: {z:?}");
    for &(re, im) in want {
        let w = num_complex::Complex64::new(re, im);
        let k = (0..z.len())
            .min_by(|&a, &b| (z[a] - w).norm().total_cmp(&(z[b] - w).norm()))
            .unwrap();
        assert!(
            (z[k] - w).norm() <= 1e-10 * w.norm().max(1.0),
            "{what}: {w} not among {z:?}"
        );
        z.remove(k);
    }
}

/// Before (0.44): the eigenvalues of a 4×4 matrix with a parameter whose
/// characteristic polynomial `t⁴ + 4t³ + x·t² − 2t + 5` is a general
/// quartic were refused ("a general quartic one (whose radical roots do not
/// evaluate where it has complex roots)").  Now they are Ferrari's roots
/// (SymPy's `roots_quartic`), and they evaluate at `x = 1/2` (two real
/// roots) and at `x = 3` (two complex pairs: the radicands are negative
/// reals written with complex cube roots, decided real exactly).
///
/// Oracle: mpmath `polyroots([1, 4, 1/2, -2, 5])` →
/// `-3.59952026842732282153594906164939471`,
/// `-1.5670157668818220274189475178688504`,
/// `0.583268017654568424477448289759122553 ±
/// 0.739083127027217270207827747348092348j`; `polyroots([1, 4, 3, -2,
/// 5])` → `-2.439748469536361512867538324270333269393 ±
/// 0.7568854969757944141518026513549220253436j`,
/// `0.4397484695363615128675383242703332693925 ±
/// 0.7568854969757944141518026513549220253436j`.
#[test]
fn ferrari_eigenvalues_of_a_parametric_companion_matrix() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let a = matrix(
        &ctx,
        &[
            &["0", "1", "0", "0"],
            &["0", "0", "1", "0"],
            &["0", "0", "0", "1"],
            &["-5", "2", "-x", "-4"],
        ],
    );
    let ev = a.eigenvals().unwrap();
    assert_eq!(ev.len(), 4);
    let at = |v: &str| -> Vec<Ex> { ev.iter().map(|l| l.subs(&x, &p(&ctx, v))).collect() };
    same_values(
        &at("1/2"),
        &[
            (-3.59952026842732, 0.0),
            (-1.56701576688182, 0.0),
            (0.583268017654568, -0.739083127027217),
            (0.583268017654568, 0.739083127027217),
        ],
        "x = 1/2",
    );
    same_values(
        &at("3"),
        &[
            (-2.43974846953636, -0.756885496975794),
            (-2.43974846953636, 0.756885496975794),
            (0.439748469536362, -0.756885496975794),
            (0.439748469536362, 0.756885496975794),
        ],
        "x = 3",
    );
    // The eigenvectors are columns of adj(λI − A): A·v = λ·v at a value.
    let xv = p(&ctx, "3");
    let an = a.subs(&x, &xv);
    for (l, m, vs) in a.eigenvects().unwrap() {
        assert_eq!((m, vs.len()), (1, 1));
        let l = l.subs(&x, &xv).eval_complex64().unwrap();
        let v: Vec<_> = (0..4)
            .map(|i| vs[0][(i, 0)].subs(&x, &xv).eval_complex64().unwrap())
            .collect();
        let norm: f64 = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        assert!(norm > 1e-6);
        for i in 0..4 {
            let mut r = -l * v[i];
            for (j, vj) in v.iter().enumerate() {
                r += an[(i, j)].eval_complex64().unwrap() * vj;
            }
            assert!(r.norm() <= 1e-9 * norm * (1.0 + l.norm()), "row {i}: {r}");
        }
    }
}

/// Before (0.44): the poles of `1/(s⁴ + 4x·s + 4x² − x)` were missing (the
/// companion matrix was refused, and `solve` found no root).  Its resolvent
/// `64R³ − 16(4x² − x)·R − 16x²` has the root `R = x`, so the poles are
/// Descartes–Euler's `±√x ∓ √(A ± B)` (square roots only).
///
/// Oracle: mpmath `polyroots([1, 0, 0, 8, 14])` (`x = 2`) →
/// `-1.41421356237309504880168872420969807857 ±
/// 0.7653668647301795434569199680607977335227j`,
/// `1.41421356237309504880168872420969807857 ±
/// 1.84775906502257256366378793576573645j`.
#[test]
fn descartes_euler_poles_of_a_parametric_transfer_function() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let s = ctx.symbol("s");
    let g = TransferFunction::new(ctx.one(), p(&ctx, "s^4 + 4*x*s + 4*x^2 - x"), s);
    let poles = g.poles();
    assert_eq!(poles.len(), 4, "{poles:?}");
    assert!(
        poles.iter().all(|e| !e.to_string().contains("cbrt")),
        "square roots only: {poles:?}"
    );
    let xv = p(&ctx, "2");
    let at: Vec<Ex> = poles.iter().map(|l| l.subs(&x, &xv)).collect();
    same_values(
        &at,
        &[
            (-std::f64::consts::SQRT_2, -0.76536686473018),
            (-std::f64::consts::SQRT_2, 0.76536686473018),
            (std::f64::consts::SQRT_2, -1.84775906502257),
            (std::f64::consts::SQRT_2, 1.84775906502257),
        ],
        "x = 2",
    );
}

/// Before (0.44): `matrix_sqrt` and `matrix_log` of the real (not
/// Hermitian) matrix `[[3, 1 + √2, 2], [3, −2, √2/2], [√2/2, −2, √3]]` gave
/// entries that do not evaluate (`PrecisionExhausted`): its eigenvalue
/// `λ ≈ −3.2731` comes out of Cardano's formula with complex cube roots, so
/// `√λ` sits on the branch cut.  `λ̄` is an eigenvalue of the real matrix
/// too and no other eigenvalue lies near it, so `λ` is real: `√λ =
/// i·√(−λ)`, `ln λ = ln(−λ) + iπ`.
///
/// Oracle: mpmath at 50 digits, `V·diag(f(λ))·V⁻¹` from `mp.eig` with the
/// principal branch: eigenvalues `-3.2730942988980187952030611554`,
/// `3.93564575772980553032073667283`, `2.06949934873709055840977082408`;
/// `√A[0][0]` = `1.7352211107693470009367746355622808 +
/// 0.36575921101158350248546085869636915j`, `√A[1][2]` =
/// `0.211584740743989045089067388274711047 +
/// 0.07348092299140044976788323614201027658579j`; `ln A[0][0]` =
/// `1.512514285023704337963627224034036069731 +
/// 0.6351347659421727685813216712674810595044j`, `ln A[2][1]` =
/// `0.1618138209541746931891468179608423809763 +
/// 1.16266245742011425415688061937842215j` (scipy `sqrtm`/`logm` agree).
#[test]
fn principal_root_and_logarithm_of_a_real_matrix_with_a_cardano_spectrum() {
    let ctx = Context::new();
    let a = matrix(
        &ctx,
        &[
            &["3", "1 + sqrt(2)", "2"],
            &["3", "-2", "sqrt(2)/2"],
            &["sqrt(2)/2", "-2", "sqrt(3)"],
        ],
    );
    let s = a.matrix_sqrt().unwrap();
    let close = |e: &Ex, re: f64, im: f64| {
        let z = e.eval_complex64().unwrap();
        assert!(
            (z.re - re).abs() < 1e-12 && (z.im - im).abs() < 1e-12,
            "{z} vs {re} + {im}i"
        );
    };
    close(&s[(0, 0)], 1.7352211107693, 0.36575921101158);
    close(&s[(1, 2)], 0.211584740743989, 0.0734809229914004);
    let l = a.matrix_log().unwrap();
    close(&l[(0, 0)], 1.5125142850237, 0.635134765942173);
    close(&l[(2, 1)], 0.161813820954175, 1.16266245742011);
    // Every entry evaluates.
    for e in s.iter().chain(l.iter()) {
        e.eval_complex64().unwrap();
    }
}
