//! 0.21 track: the shared dense `f64` kernel (`base::dense_f64`).
//!
//! One Cholesky / Jacobi / partial-pivot / Householder implementation now
//! sits behind the SOS interior point, `logit`/`mnlogit`/`ologit`,
//! `cox_ph`, `MultivariateNormal::sample`, `pca_f64`, `poly_fit`,
//! `solve_numeric_system` and heurisch.  The kernel itself is
//! `pub(crate)`, so its direct tests (hand-checkable 3×3/4×4 SPD matrices
//! against `numpy.linalg.{cholesky,inv,solve,eigh,lstsq}`, the relative
//! versus absolute tolerances, a non-SPD matrix returning `None`, an
//! overdetermined `lstsq`) live in `src/base/dense_f64.rs`; this file
//! pins one documented value per switched caller so the consolidation is
//! output-preserving.  Reference values cite statsmodels 0.14 / scipy /
//! numpy (`symplex/.venv/bin/python`).

use symplex::certificates::{SosOpts, prove_sos};
use symplex::linprog::q;
use symplex::optimize::poly_fit;
use symplex::polysys::solve_numeric_system;
use symplex::prelude::*;
use symplex::stats::Rng;
use symplex::stats::cox::{CoxOpts, cox_ph};
use symplex::stats::multivariate::{MultivariateNormal, pca_f64};
use symplex::stats::regression::{LogitOpts, logit, mnlogit};
use symplex::stats::survival::Observation;

fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-9 * expected.abs().max(1.0),
        "{label}: {actual} vs {expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// stats: Cholesky + Wald summary (information_cholesky / wald_summary)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn logit_doc_values_unchanged() {
    // Ten controls with 3 successes, ten treated with 7 (the `logit` doc).
    let y: Vec<bool> = (0..20).map(|i| matches!(i, 7..=9 | 13..=19)).collect();
    let x: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![if i < 10 { 0.0 } else { 1.0 }])
        .collect();
    let fit = logit(&y, &x, true, &LogitOpts::default()).unwrap();
    // statsmodels Logit(y, add_constant(x)).fit(): params = [ln(3/7), ln(49/9)]
    close(fit.coefficients[0], (3.0f64 / 7.0).ln(), "β₀");
    close(fit.coefficients[1], (49.0f64 / 9.0).ln(), "β₁");
    // bse = [0.6900655593423543, 0.9759000729485332] = [√(10/21), √(20/21)]
    close(fit.standard_errors[0], 0.6900655593423543, "se₀");
    close(fit.standard_errors[1], 0.9759000729485332, "se₁");
    // pvalues = [0.21950281228300073, 0.08248537711586468]
    close(fit.p_values[0], 0.21950281228300073, "p₀");
    close(fit.p_values[1], 0.08248537711586468, "p₁");
    // cov_params() = [[10/21, -10/21], [-10/21, 20/21]]
    close(fit.cov_params[0][0], 10.0 / 21.0, "cov₀₀");
    close(fit.cov_params[0][1], -10.0 / 21.0, "cov₀₁");
    close(fit.cov_params[1][0], -10.0 / 21.0, "cov₁₀");
    close(fit.cov_params[1][1], 20.0 / 21.0, "cov₁₁");
    close(fit.log_likelihood, -12.217286041097868, "llf");
    // A collinear design is still rejected by the relative pivot test.
    let collinear: Vec<Vec<f64>> = x.iter().map(|r| vec![r[0], 2.0 * r[0]]).collect();
    assert!(matches!(
        logit(&y, &collinear, true, &LogitOpts::default()),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn mnlogit_doc_values_unchanged() {
    let y = [
        0, 0, 1, 0, 0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 1, 2, 1, 2, 2, 1, 2, 0, 2, 2,
    ];
    let x: Vec<Vec<f64>> = (1..=24).map(|i| vec![f64::from(i)]).collect();
    let fit = mnlogit(&y, &x, true, &LogitOpts::default()).unwrap();
    assert!(fit.converged);
    // statsmodels MNLogit(y, add_constant(x)).fit(method='newton'):
    // params.T = [[-0.9175964754951935, 0.10985702371911016], [-2.875029245270399, 0.2536273262970427]]
    close(fit.coefficients[0][0], -0.9175964754951935, "β₁₀");
    close(fit.coefficients[0][1], 0.10985702371911016, "β₁₁");
    close(fit.coefficients[1][0], -2.875029245270399, "β₂₀");
    close(fit.coefficients[1][1], 0.2536273262970427, "β₂₁");
    // bse.T = [[1.0202923117484362, 0.09311682562597019], [1.4432124419606307, 0.1081682371602044]]
    close(fit.standard_errors[0][0], 1.0202923117484362, "se₁₀");
    close(fit.standard_errors[0][1], 0.09311682562597019, "se₁₁");
    close(fit.standard_errors[1][0], 1.4432124419606307, "se₂₀");
    close(fit.standard_errors[1][1], 0.1081682371602044, "se₂₁");
    // pvalues.T = [[0.36846804499407193, 0.23808919923338745], [0.04635965070751905, 0.019039911083865633]]
    close(fit.p_values[0][0], 0.36846804499407193, "p₁₀");
    close(fit.p_values[1][1], 0.019039911083865633, "p₂₁");
    close(fit.log_likelihood, -22.117379463121267, "llf");
}

#[test]
fn cox_doc_values_unchanged() {
    // Ten subjects, one covariate, three censored (the `stats::cox` doc).
    let obs = Observation::from_i64(
        &[4, 7, 2, 9, 12, 5, 15, 3, 11, 8],
        &[
            true, true, true, false, true, true, false, true, true, false,
        ],
    );
    let x: Vec<Vec<f64>> = [3.0, 1.0, 5.0, 2.0, 0.0, 4.0, 1.0, 6.0, 2.0, 3.0]
        .iter()
        .map(|&v| vec![v])
        .collect();
    let fit = cox_ph(&obs, &x, &CoxOpts::default()).unwrap();
    // statsmodels PHReg(t, x, status=s, ties='efron').fit(): params [0.8759809887649096],
    // bse [0.3809028296542367], llf -8.465216276861835
    close(fit.coefficients[0], 0.875_980_988_764_909_6, "β");
    close(fit.standard_errors[0], 0.380_902_829_654_236_7, "se");
    close(fit.log_likelihood, -8.465_216_276_861_835, "llf");
    // se² is the single entry of cov_params; z and the two-sided p follow.
    close(
        fit.cov_params[0][0],
        0.380_902_829_654_236_7 * 0.380_902_829_654_236_7,
        "cov",
    );
    close(
        fit.z_values[0],
        0.875_980_988_764_909_6 / 0.380_902_829_654_236_7,
        "z",
    );
    assert!(
        fit.p_values[0] > 0.02 && fit.p_values[0] < 0.022,
        "{}",
        fit.p_values[0]
    );
    assert_eq!(fit.concordance().unwrap(), q(31, 38));
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::multivariate: Cholesky sampling and Jacobi PCA
// ═══════════════════════════════════════════════════════════════════════════

/// `N((1, 2), [[2, 1], [1, 3]])`.
fn mvn2(ctx: &Context) -> MultivariateNormal {
    MultivariateNormal::try_new(vec![ctx.int(1), ctx.int(2)], matrix![ctx, [2, 1], [1, 3]]).unwrap()
}

#[test]
fn mvn_precision_density_and_sampling_unchanged() {
    let ctx = Context::new();
    let mvn = mvn2(&ctx);
    // Σ⁻¹ = [[3/5, −1/5], [−1/5, 2/5]] exactly.
    let prec = mvn.precision().unwrap();
    assert_eq!(*prec.get(0, 0), ctx.rational(3, 5));
    assert_eq!(*prec.get(0, 1), ctx.rational(-1, 5));
    assert_eq!(*prec.get(1, 1), ctx.rational(2, 5));
    // scipy.stats.multivariate_normal([1, 2], [[2, 1], [1, 3]]).pdf([1.5, 2.5]) = 0.06603330634679298
    let d = mvn
        .density(&[ctx.rational(3, 2), ctx.rational(5, 2)])
        .unwrap()
        .eval_f64()
        .unwrap();
    close(d, 0.06603330634679298, "pdf(1.5, 2.5)");
    // Sampling goes through the kernel's Cholesky of Σ: deterministic for
    // a seed, with the right first and second moments.
    let n = 4000;
    let draws = mvn.sample(n, &mut Rng::new(42)).unwrap();
    assert_eq!(draws, mvn.sample(n, &mut Rng::new(42)).unwrap());
    let mean = |j: usize| draws.iter().map(|x| x[j]).sum::<f64>() / n as f64;
    let cov = |i: usize, j: usize| {
        let (mi, mj) = (mean(i), mean(j));
        draws.iter().map(|x| (x[i] - mi) * (x[j] - mj)).sum::<f64>() / (n as f64 - 1.0)
    };
    assert!((mean(0) - 1.0).abs() < 0.1 && (mean(1) - 2.0).abs() < 0.1);
    assert!((cov(0, 0) - 2.0).abs() < 0.25 && (cov(1, 1) - 3.0).abs() < 0.25);
    assert!((cov(0, 1) - 1.0).abs() < 0.25);
    // A singular covariance cannot be sampled (Cholesky pivot ≤ 0).
    let degenerate =
        MultivariateNormal::new(vec![ctx.int(0), ctx.int(0)], matrix![ctx, [1, 1], [1, 1]]);
    assert!(matches!(
        degenerate.sample(1, &mut Rng::new(1)),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn pca_f64_matches_numpy_eigh() {
    let cov = vec![
        vec![4.0, 1.0, 0.5, 0.0],
        vec![1.0, 3.0, 0.25, 0.5],
        vec![0.5, 0.25, 2.0, 1.0],
        vec![0.0, 0.5, 1.0, 5.0],
    ];
    let p = pca_f64(&cov).unwrap();
    // numpy.linalg.eigh eigenvalues (descending): 5.5446889571830384, 4.5024046887344715,
    // 2.3618374429898186, 1.5910689110926703
    let expected = [
        5.5446889571830384,
        4.5024046887344715,
        2.3618374429898186,
        1.5910689110926703,
    ];
    for (i, e) in expected.iter().enumerate() {
        close(p.eigenvalues[i], *e, &format!("λ{i}"));
    }
    // Leading eigh column, signed so its first coordinate is positive.
    let v0 = [
        0.30225746834694595,
        0.3150155112535152,
        0.3037565246562271,
        0.8468397866343743,
    ];
    for (j, c) in v0.iter().enumerate() {
        close(p.components[0][j], *c, &format!("v0{j}"));
    }
    close(p.explained_variance_ratio[0], 0.39604921122735987, "ratio₀");
    // 2×2 with equal diagonal entries (a zero rotation angle): eigenvalues 1 and 3.
    let e = pca_f64(&[vec![2.0, -1.0], vec![-1.0, 2.0]]).unwrap();
    close(e.eigenvalues[0], 3.0, "λ₀");
    close(e.eigenvalues[1], 1.0, "λ₁");
    assert!(
        pca_f64(&[vec![1.0, 2.0], vec![3.0, 4.0]]).is_err(),
        "not symmetric"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// optimize / polysys / heurisch: Householder and partial pivoting
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn poly_fit_is_householder_least_squares() {
    // numpy.polyfit([0, 1, 2, 3], [1, 3, 2, 5], 1) = [1.1, 1.1] (descending).
    let c = poly_fit(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 2.0, 5.0], 1).unwrap();
    close(c[0], 1.1, "intercept");
    close(c[1], 1.1, "slope");
    // Exact data (the `poly_fit` doc): 1 + 2x + 3x².
    let xs: Vec<f64> = (0..6).map(f64::from).collect();
    let ys: Vec<f64> = xs.iter().map(|x| 1.0 + 2.0 * x + 3.0 * x * x).collect();
    let c = poly_fit(&xs, &ys, 2).unwrap();
    close(c[0], 1.0, "c₀");
    close(c[1], 2.0, "c₁");
    close(c[2], 3.0, "c₂");
    // Two distinct abscissae cannot carry a quadratic.
    assert!(matches!(
        poly_fit(&[0.0, 1.0, 0.0, 1.0], &[0.0, 1.0, 0.0, 1.0], 2),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn newton_system_solves_through_partial_pivoting() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // Unit circle ∩ y = x near (1, 1) (the `solve_numeric_system` doc).
    let eqs = [&x.powi(2) + &y.powi(2) - 1, &y - &x];
    let sol = solve_numeric_system(&eqs, &[x.clone(), y.clone()], &[1.0, 1.0]).unwrap();
    let r = std::f64::consts::FRAC_1_SQRT_2;
    assert!((sol[0] - r).abs() < 1e-10 && (sol[1] - r).abs() < 1e-10);
    // Jacobian [[0, 1], [1, 1]]: the zero leading pivot forces a row swap.
    let eqs = [&y - 1, &x + &y - 3];
    let sol = solve_numeric_system(&eqs, &[x.clone(), y.clone()], &[0.0, 0.0]).unwrap();
    assert!((sol[0] - 2.0).abs() < 1e-10 && (sol[1] - 1.0).abs() < 1e-10);
    // A singular Jacobian is reported, not divided by.
    let eqs = [&x + &y, 2 * (&x + &y) - 1];
    assert!(matches!(
        solve_numeric_system(&eqs, &[x, y], &[0.0, 0.0]),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn heurisch_outcomes_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // These reach heurisch's least-squares stage (table and Risch pass) and
    // are left unevaluated, exactly as before the switch to Householder.
    for e in [x.ln().sin(), x.exp() * x.cos().powi(2), x.exp() * x.tan()] {
        let s = e.integrate(&x).to_string();
        assert!(s.starts_with("Integral("), "{e}: {s}");
    }
    // …while the surrounding pipeline still closes ordinary integrands.
    assert_eq!(
        (&x * x.exp()).integrate(&x).to_string(),
        "x*exp(x) - exp(x)"
    );
    assert_eq!(
        (x.ln().ln() / &x).integrate(&x).to_string(),
        "-ln(x) + ln(x)*ln(ln(x))"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// certificates::sos: the interior point's Cholesky / Jacobi / Schur solve
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn amgm_sos_certificate_is_byte_identical() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let amgm = x.powi(4) + y.powi(4) + z.powi(4) - &x * &y * &z * 4 + 1;
    let out = prove_sos(&amgm, &[x, y, z], &SosOpts::default()).unwrap();
    let sos = out.certificate().unwrap();
    assert!(sos.verify());
    // The rounded Gram matrix (hence the printed identity) depends on the
    // numerical SDP path: this is `examples/readme_snippets.rs`' output.
    assert_eq!(
        sos.to_string(),
        "x^4 + y^4 + z^4 - 4*x*y*z + 1 = (-1/3*x^2 - 1/3*y^2 - 1/3*z^2 + 1)^2 + 2/3*(-y*z + x)^2 + 2/3*(-x*z + y)^2 + 2/3*(-x*y + z)^2 + 2/3*(-x^2 + y^2)^2 + 8/9*(-1/2*x^2 - 1/2*y^2 + z^2)^2"
    );
}
