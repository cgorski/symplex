//! symplex 0.14 — multivariate_order: `stats::multivariate`, `stats::order`,
//! `stats::information`, `stats::sequential`.  Reference values cite scipy
//! 1.18.1 / numpy 2.5.3 / SymPy 1.14.0 / Python `fractions`+`math`
//! (`symplex/.venv/bin/python`).  Every rational quantity is asserted
//! exactly; roots, logarithms and floats to 1e-9.

use symplex::linprog::{q, qi};
use symplex::prelude::*;
use symplex::stats::data::{Ddof, Q};
use symplex::stats::information::{
    self as info, Base, Given, Norm, bhattacharyya_coefficient, bhattacharyya_distance,
    conditional_entropy, cross_entropy, entropy, hellinger, information_gain, joint_entropy,
    joint_from_counts, js_divergence, kl_divergence, mutual_information,
    normalized_mutual_information, perplexity, probability_vector, total_variation,
};
use symplex::stats::multivariate::{
    MultivariateNormal, correlation_matrix, covariance_matrix, pca, pca_f64,
};
use symplex::stats::order::{OrderStatistic, maximum_of, minimum_of, order_statistic};
use symplex::stats::sequential::{
    Decision, Sprt, expected_sample_size_bernoulli, operating_characteristic_bernoulli,
    wald_boundaries,
};
use symplex::stats::{Distribution, Finite, Normal, Rng};

fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-9 * expected.abs().max(1.0),
        "{label}: {actual} vs {expected}"
    );
}

fn ex_close(actual: &Ex, expected: f64, label: &str) {
    let v = actual
        .eval_f64()
        .unwrap_or_else(|e| panic!("{label}: `{actual}` did not evaluate: {e}"));
    close(v, expected, label);
}

fn is_true(actual: &Ex, expected: &Ex, label: &str) {
    assert_eq!(
        actual.equals(expected),
        Some(true),
        "{label}: `{actual}` vs `{expected}`"
    );
}

/// `N((1, 2), [[2, 1], [1, 3]])`: the running two-dimensional example.
fn mvn2(ctx: &Context) -> MultivariateNormal {
    MultivariateNormal::try_new(vec![ctx.int(1), ctx.int(2)], matrix![ctx, [2, 1], [1, 3]])
        .expect("valid")
}

/// `N(0, [[4, 2, 1], [2, 3, 1], [1, 1, 2]])`.
fn mvn3(ctx: &Context) -> MultivariateNormal {
    MultivariateNormal::try_new(
        vec![ctx.int(0), ctx.int(0), ctx.int(0)],
        matrix![ctx, [4, 2, 1], [2, 3, 1], [1, 1, 2]],
    )
    .expect("valid")
}

/// Five observations of three variables (column 1 = 2 × column 0).
fn data5x3() -> Vec<Vec<Q>> {
    [[1, 2, 3], [2, 4, 1], [3, 6, 4], [4, 8, 2], [6, 12, 5]]
        .iter()
        .map(|row| row.iter().map(|&v| qi(v)).collect())
        .collect()
}

/// Cover & Thomas, *Elements of Information Theory*, Example 2.2.1 (rows
/// and columns as laid out there; here rows are the first variable).
fn cover_thomas_joint() -> Vec<Vec<Q>> {
    vec![
        vec![q(1, 8), q(1, 16), q(1, 32), q(1, 32)],
        vec![q(1, 16), q(1, 8), q(1, 32), q(1, 32)],
        vec![q(1, 16), q(1, 16), q(1, 16), q(1, 16)],
        vec![q(1, 4), q(0, 1), q(0, 1), q(0, 1)],
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::multivariate — MultivariateNormal
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mvn_density_matches_scipy_pdf() {
    let ctx = Context::new();
    let mvn = mvn2(&ctx);
    // scipy.stats.multivariate_normal([1, 2], [[2, 1], [1, 3]]).pdf([1.5, 2.5]) = 0.06603330634679298
    let d = mvn
        .density(&[ctx.rational(3, 2), ctx.rational(5, 2)])
        .unwrap();
    ex_close(&d, 0.06603330634679298, "pdf(1.5, 2.5)");
    // .pdf([0, 0]) = 0.03534508188501653
    let d0 = mvn.density(&[ctx.int(0), ctx.int(0)]).unwrap();
    ex_close(&d0, 0.03534508188501653, "pdf(0, 0)");
    assert_eq!(mvn.dim(), 2);
    assert!(mvn.density(&[ctx.int(0)]).is_err(), "wrong dimension");
}

#[test]
fn mvn_exact_density_at_mean_and_entropy() {
    let ctx = Context::new();
    let mvn = mvn2(&ctx);
    // 1/√((2π)² det Σ) = 1/(2π√5); scipy .pdf([1, 2]) = 0.07117625434171772
    let at_mean = mvn.density(&[ctx.int(1), ctx.int(2)]).unwrap();
    is_true(
        &at_mean,
        &(ctx.one() / (2 * ctx.pi() * ctx.int(5).sqrt())),
        "density at μ",
    );
    ex_close(&at_mean, 0.07117625434171772, "pdf(μ)");
    // ½ ln((2πe)² · 5); scipy .entropy() = 3.6425960226263956
    ex_close(&mvn.entropy().unwrap(), 3.6425960226263956, "entropy");
}

#[test]
fn mvn_symbolic_density_matches_sympy() {
    let ctx = Context::new();
    let mvn = mvn2(&ctx);
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let ours = mvn.density(&[x.clone(), y.clone()]).unwrap();
    // sympy.stats.density(MultivariateNormal('X', [1, 2], [[2, 1], [1, 3]]))(x, y)
    //   = sqrt(5)*exp(-3*x**2/10 + x*y/5 + x/5 - y**2/5 + 3*y/5 - 7/10)/(10*pi)
    let exponent =
        ctx.rational(-3, 10) * x.powi(2) + ctx.rational(1, 5) * &x * &y + ctx.rational(1, 5) * &x
            - ctx.rational(1, 5) * y.powi(2)
            + ctx.rational(3, 5) * &y
            - ctx.rational(7, 10);
    let sympy = ctx.int(5).sqrt() * exponent.exp() / (10 * ctx.pi());
    assert_ne!(ours.equals(&sympy), Some(false));
    for (xv, yv) in [(q(3, 10), q(-6, 5)), (qi(2), qi(5)), (q(-1, 2), q(7, 3))] {
        let (xe, ye) = (ctx.from_ratio(xv), ctx.from_ratio(yv));
        let a = ours.subs(&x, &xe).subs(&y, &ye).eval_f64().unwrap();
        let b = sympy.subs(&x, &xe).subs(&y, &ye).eval_f64().unwrap();
        close(a, b, "symbolic density vs SymPy at a point");
    }
}

#[test]
fn mvn_try_new_rejects_bad_parameters() {
    let ctx = Context::new();
    let zero2 = vec![ctx.int(0), ctx.int(0)];
    // indefinite [[1, 2], [2, 1]] (eigenvalues 3, −1)
    assert!(MultivariateNormal::try_new(zero2.clone(), matrix![ctx, [1, 2], [2, 1]]).is_err());
    // singular [[1, 1], [1, 1]] is only semidefinite
    assert!(MultivariateNormal::try_new(zero2.clone(), matrix![ctx, [1, 1], [1, 1]]).is_err());
    // not symmetric
    assert!(MultivariateNormal::try_new(zero2.clone(), matrix![ctx, [2, 1], [0, 3]]).is_err());
    // shape mismatch
    assert!(MultivariateNormal::try_new(zero2.clone(), matrix![ctx, [2]]).is_err());
    assert!(MultivariateNormal::try_new(vec![], matrix![ctx, [2]]).is_err());
    // symbolic covariance with undecidable sign is accepted (caller's promise)
    let s = ctx.symbol("s");
    let sym = Matrix::new(vec![
        vec![s.clone(), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();
    assert!(MultivariateNormal::try_new(zero2, sym).is_ok());
}

#[test]
fn mvn_marginals_match_sympy() {
    let ctx = Context::new();
    let mvn = mvn2(&ctx);
    // marginal_distribution(X, X[1])(y) = sqrt(6)*exp(-(y - 2)**2/6)/(6*sqrt(pi)): Normal(2, √3)
    let m1 = mvn.marginal(&[1]).unwrap();
    assert_eq!(m1.mean, vec![ctx.int(2)]);
    assert_eq!(m1.cov, matrix![ctx, [3]]);
    let x1 = mvn.marginal_1d(1).unwrap();
    assert_eq!(x1.mean(), ctx.int(2));
    assert_eq!(x1.variance(), ctx.int(3));
    // marginal_distribution(X, X[0])(x) = exp(-(x - 1)**2/4)/(2*sqrt(pi)): Normal(1, √2)
    let x0 = mvn.marginal_1d(0).unwrap();
    let n = x0.downcast_ref::<Normal>().expect("a Normal");
    assert_eq!(n.mean, ctx.int(1));
    is_true(&n.std, &ctx.int(2).sqrt(), "σ₀");
    let x = ctx.symbol("x");
    let sympy = (-(&x - 1).powi(2) / 4).exp() / (2 * ctx.pi().sqrt());
    is_true(&x0.density(&x), &sympy, "marginal density vs SymPy");
    // Reordering the coordinates permutes the parameters.
    let swapped = mvn.marginal(&[1, 0]).unwrap();
    assert_eq!(swapped.mean, vec![ctx.int(2), ctx.int(1)]);
    assert_eq!(swapped.cov, matrix![ctx, [3, 1], [1, 2]]);
    assert!(mvn.marginal(&[]).is_err());
    assert!(mvn.marginal(&[0, 0]).is_err());
    assert!(mvn.marginal(&[2]).is_err());
}

#[test]
fn mvn_conditional_2d_is_schur_complement() {
    let ctx = Context::new();
    let mvn = mvn2(&ctx);
    let y = ctx.symbol("y");
    // μ₀ + Σ₀₁ Σ₁₁⁻¹ (y − μ₁) = 1 + (y − 2)/3 = y/3 + 1/3 (sympy.simplify);
    // Σ₀₀ − Σ₀₁ Σ₁₁⁻¹ Σ₁₀ = 2 − 1/3 = 5/3 (Fraction)
    let cond = mvn.conditional(&[(1, y.clone())]).unwrap();
    assert_eq!(cond.dim(), 1);
    is_true(
        &cond.mean[0],
        &(&y / 3 + ctx.rational(1, 3)),
        "conditional mean",
    );
    assert_eq!(cond.cov.shape(), (1, 1));
    assert_eq!(cond.cov[(0, 0)], ctx.rational(5, 3));
    // Numeric conditioning value: X₀ | X₁ = 5 has mean 2.
    let at5 = mvn.conditional(&[(1, ctx.int(5))]).unwrap();
    assert_eq!(at5.mean, vec![ctx.int(2)]);
    assert!(
        mvn.conditional(&[(0, ctx.int(1)), (1, ctx.int(2))])
            .is_err(),
        "nothing left"
    );
    assert!(mvn.conditional(&[]).is_err());
}

#[test]
fn mvn_3d_conditional_and_density_match_oracles() {
    let ctx = Context::new();
    let mvn = mvn3(&ctx);
    // Fraction Schur complement: X₀ | X₁ = 1, X₂ = 2 has mean 1, variance 13/5
    let cond = mvn
        .conditional(&[(1, ctx.int(1)), (2, ctx.int(2))])
        .unwrap();
    assert_eq!(cond.mean, vec![ctx.int(1)]);
    assert_eq!(cond.cov[(0, 0)], ctx.rational(13, 5));
    // Conditioning on the middle coordinate leaves (X₀, X₂), in that order:
    // Σ_AA − Σ_AB Σ_BA / 3 = [[8/3, 1/3], [1/3, 5/3]] (Fraction)
    let two = mvn.conditional(&[(1, ctx.int(0))]).unwrap();
    assert_eq!(two.dim(), 2);
    assert_eq!(two.mean, vec![ctx.int(0), ctx.int(0)]);
    assert_eq!(two.cov[(0, 0)], ctx.rational(8, 3));
    assert_eq!(two.cov[(0, 1)], ctx.rational(1, 3));
    assert_eq!(two.cov[(1, 0)], ctx.rational(1, 3));
    assert_eq!(two.cov[(1, 1)], ctx.rational(5, 3));
    // scipy multivariate_normal(0, Σ).pdf([1, 2, -1]) = 0.003929314568571803
    let d = mvn.density(&[ctx.int(1), ctx.int(2), ctx.int(-1)]).unwrap();
    ex_close(&d, 0.003929314568571803, "3-D pdf");
}

#[test]
fn mvn_mahalanobis_is_exact() {
    let ctx = Context::new();
    let mvn = mvn2(&ctx);
    // Σ⁻¹ = [[3, −1], [−1, 2]]/5, x − μ = (1, 1): d² = 3/5;
    // scipy: −2·ln(pdf([2, 3]) / pdf([1, 2])) = 0.6
    let d2 = mvn.mahalanobis_squared(&[ctx.int(2), ctx.int(3)]).unwrap();
    assert_eq!(d2, ctx.rational(3, 5));
    is_true(
        &mvn.mahalanobis(&[ctx.int(2), ctx.int(3)]).unwrap(),
        &ctx.rational(3, 5).sqrt(),
        "d",
    );
    assert_eq!(
        mvn.mahalanobis_squared(&[ctx.int(1), ctx.int(2)]).unwrap(),
        ctx.zero()
    );
    assert_eq!(
        mvn.precision().unwrap(),
        matrix![ctx, [3, -1], [-1, 2]].map(|e| e / 5)
    );
}

#[test]
fn mvn_affine_image() {
    let ctx = Context::new();
    let mvn = mvn2(&ctx);
    // numpy: A @ μ + b = [3, 0], A @ Σ @ A.T = [[7, -1], [-1, 3]] for A = [[1, 1], [1, -1]], b = (0, 1)
    let a = matrix![ctx, [1, 1], [1, -1]];
    let img = mvn.affine(&a, &[ctx.int(0), ctx.int(1)]).unwrap();
    assert_eq!(img.mean, vec![ctx.int(3), ctx.int(0)]);
    assert_eq!(img.cov, matrix![ctx, [7, -1], [-1, 3]]);
    // A 1 × 2 projection onto the sum: Normal(3, √7)
    let sum = mvn.affine(&matrix![ctx, [1, 1]], &[ctx.int(0)]).unwrap();
    assert_eq!(sum.marginal_1d(0).unwrap().variance(), ctx.int(7));
    assert!(mvn.affine(&matrix![ctx, [1, 1, 1]], &[ctx.int(0)]).is_err());
    assert!(mvn.affine(&a, &[ctx.int(0)]).is_err());
}

#[test]
fn mvn_sampling_is_deterministic_with_correct_moments() {
    let ctx = Context::new();
    let mvn = mvn2(&ctx);
    let n = 4000;
    let draws = mvn.sample(n, &mut Rng::new(42)).unwrap();
    assert_eq!(draws.len(), n);
    let again = mvn.sample(n, &mut Rng::new(42)).unwrap();
    assert_eq!(draws, again, "same seed, same draws");
    let mean = |j: usize| draws.iter().map(|x| x[j]).sum::<f64>() / n as f64;
    let (m0, m1) = (mean(0), mean(1));
    assert!((m0 - 1.0).abs() < 0.1, "mean₀ {m0}");
    assert!((m1 - 2.0).abs() < 0.1, "mean₁ {m1}");
    let cov = |i: usize, j: usize| {
        let (mi, mj) = (mean(i), mean(j));
        draws.iter().map(|x| (x[i] - mi) * (x[j] - mj)).sum::<f64>() / (n as f64 - 1.0)
    };
    assert!((cov(0, 0) - 2.0).abs() < 0.25, "var₀ {}", cov(0, 0));
    assert!((cov(1, 1) - 3.0).abs() < 0.25, "var₁ {}", cov(1, 1));
    assert!((cov(0, 1) - 1.0).abs() < 0.25, "cov {}", cov(0, 1));
    // A symbolic parameter cannot be sampled.
    let sym = MultivariateNormal::new(
        vec![ctx.symbol("m"), ctx.int(0)],
        matrix![ctx, [1, 0], [0, 1]],
    );
    assert!(sym.sample(1, &mut Rng::new(1)).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::multivariate — data
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn covariance_matrix_matches_numpy_cov() {
    let data = data5x3();
    // numpy.cov(data, rowvar=False, ddof=1) = [[3.7, 7.4, 1.75], [7.4, 14.8, 3.5], [1.75, 3.5, 2.5]]
    let c = covariance_matrix(&data, Ddof::Sample).unwrap();
    assert_eq!(
        c,
        QMatrix::new(vec![
            vec![q(37, 10), q(37, 5), q(7, 4)],
            vec![q(37, 5), q(74, 5), q(7, 2)],
            vec![q(7, 4), q(7, 2), q(5, 2)],
        ])
        .unwrap()
    );
    // ddof=0: [[2.96, 5.92, 1.4], [5.92, 11.84, 2.8], [1.4, 2.8, 2.0]]
    let p = covariance_matrix(&data, Ddof::Population).unwrap();
    assert_eq!(p[(0, 0)], q(74, 25));
    assert_eq!(p[(0, 1)], q(148, 25));
    assert_eq!(p[(1, 1)], q(296, 25));
    assert_eq!(p[(0, 2)], q(7, 5));
    assert_eq!(p[(2, 2)], qi(2));
    assert!(covariance_matrix(&[], Ddof::Sample).is_err());
    assert!(
        covariance_matrix(&[vec![qi(1), qi(2)]], Ddof::Sample).is_err(),
        "n = 1"
    );
    assert!(
        covariance_matrix(&[vec![qi(1)], vec![qi(1), qi(2)]], Ddof::Sample).is_err(),
        "jagged"
    );
}

#[test]
fn correlation_matrix_matches_numpy_corrcoef() {
    let ctx = Context::new();
    let data = data5x3();
    // numpy.corrcoef(data, rowvar=False): r₀₁ = 1 (column 1 = 2·column 0),
    // r₀₂ = r₁₂ = 0.5753964555687505 = 7/(2√37)
    let r = correlation_matrix(&ctx, &data).unwrap();
    assert_eq!(r[(0, 0)], ctx.one());
    assert_eq!(r[(0, 1)], ctx.one());
    is_true(&r[(0, 2)], &(ctx.int(7) / (2 * ctx.int(37).sqrt())), "r₀₂");
    ex_close(&r[(0, 2)], 0.5753964555687505, "r₀₂");
    assert_eq!(r[(1, 2)], r[(2, 1)]);
    ex_close(&r[(1, 2)], 0.5753964555687505, "r₁₂");
    // A constant column has no correlation.
    let constant = vec![vec![qi(1), qi(5)], vec![qi(2), qi(5)], vec![qi(3), qi(5)]];
    assert!(correlation_matrix(&ctx, &constant).is_err());
}

#[test]
fn pca_exact_with_rational_eigenvalues() {
    let ctx = Context::new();
    // numpy.linalg.eigh([[2, 1], [1, 2]]) = (1, 3), vectors (−1, 1)/√2, (1, 1)/√2
    let p = pca(&ctx, &QMatrix::from_i64(&[&[2, 1], &[1, 2]]).unwrap()).unwrap();
    assert_eq!(p.eigenvalues, vec![ctx.int(3), ctx.int(1)]);
    assert_eq!(
        p.explained_variance_ratio,
        vec![ctx.rational(3, 4), ctx.rational(1, 4)]
    );
    let r2 = ctx.int(2).sqrt() / 2;
    is_true(&p.components[0][0], &r2, "c₀₀");
    is_true(&p.components[0][1], &r2, "c₀₁");
    is_true(&p.components[1][0], &r2, "c₁₀");
    is_true(&p.components[1][1], &(-&r2), "c₁₁");
    // Not PSD → error; zero → error.
    assert!(pca(&ctx, &QMatrix::from_i64(&[&[1, 2], &[2, 1]]).unwrap()).is_err());
    assert!(pca(&ctx, &QMatrix::from_i64(&[&[0, 0], &[0, 0]]).unwrap()).is_err());
    assert!(pca(&ctx, &QMatrix::from_i64(&[&[1, 2], &[3, 4]]).unwrap()).is_err());
}

#[test]
fn pca_exact_with_surd_eigenvalues_matches_numpy_eigh() {
    let ctx = Context::new();
    // eigh([[2, 1], [1, 3]]) = (1.381966011250105, 3.618033988749895) = (5 ∓ √5)/2;
    // vectors (0.5257311121191335, 0.8506508083520399) for the larger eigenvalue
    let p = pca(&ctx, &QMatrix::from_i64(&[&[2, 1], &[1, 3]]).unwrap()).unwrap();
    let r5 = ctx.int(5).sqrt();
    is_true(&p.eigenvalues[0], &((ctx.int(5) + &r5) / 2), "λ₀");
    is_true(&p.eigenvalues[1], &((ctx.int(5) - &r5) / 2), "λ₁");
    ex_close(&p.eigenvalues[0], 3.618033988749895, "λ₀");
    ex_close(&p.eigenvalues[1], 1.381966011250105, "λ₁");
    is_true(
        &p.explained_variance_ratio[0],
        &(ctx.rational(1, 2) + &r5 / 10),
        "ratio₀",
    );
    ex_close(&p.components[0][0], 0.5257311121191335, "v₀₀");
    ex_close(&p.components[0][1], 0.8506508083520399, "v₀₁");
    ex_close(&p.components[1][0], 0.8506508083520399, "v₁₀");
    ex_close(&p.components[1][1], -0.5257311121191335, "v₁₁");
    let ratios = p.explained_variance_ratio_f64().unwrap();
    close(ratios[0] + ratios[1], 1.0, "ratios sum to 1");
}

#[test]
fn pca_exact_3x3_gives_ordered_rootof_eigenvalues() {
    let ctx = Context::new();
    // Characteristic polynomial λ³ − 9λ² + 20λ − 13 is irreducible (casus irreducibilis):
    // eigh([[4, 2, 1], [2, 3, 1], [1, 1, 2]]) = (1.3079785283699037, 1.6431041321077904, 6.048917339522303)
    let p = pca(
        &ctx,
        &QMatrix::from_i64(&[&[4, 2, 1], &[2, 3, 1], &[1, 1, 2]]).unwrap(),
    )
    .unwrap();
    ex_close(&p.eigenvalues[0], 6.048917339522303, "λ₀");
    ex_close(&p.eigenvalues[1], 1.6431041321077904, "λ₁");
    ex_close(&p.eigenvalues[2], 1.3079785283699037, "λ₂");
    // ratios λ/9: 0.6721019266135893, 0.18256712578975448, 0.14533094759665596
    let ratios = p.explained_variance_ratio_f64().unwrap();
    close(ratios[0], 0.6721019266135893, "ratio₀");
    close(ratios[1], 0.18256712578975448, "ratio₁");
    close(ratios[2], 0.14533094759665596, "ratio₂");
    let sum: Ex = p.eigenvalues.iter().fold(ctx.zero(), |acc, v| acc + v);
    ex_close(&sum, 9.0, "trace");
    // eigh columns (first coordinate made positive), evaluated from the exact RootOf vectors:
    let vectors = [
        [0.7369762290995784, 0.5910090485061031, 0.32798527760568164],
        [0.5910090485061029, -0.327985277605681, -0.7369762290995792],
        [0.32798527760568214, -0.7369762290995786, 0.5910090485061028],
    ];
    for (i, v) in vectors.iter().enumerate() {
        for (j, c) in v.iter().enumerate() {
            ex_close(&p.components[i][j], *c, &format!("v{i}{j}"));
        }
    }
}

#[test]
fn pca_f64_matches_numpy_eigh_4x4() {
    let cov = vec![
        vec![4.0, 1.0, 0.5, 0.0],
        vec![1.0, 3.0, 0.25, 0.5],
        vec![0.5, 0.25, 2.0, 1.0],
        vec![0.0, 0.5, 1.0, 5.0],
    ];
    let p = pca_f64(&cov).unwrap();
    // numpy.linalg.eigh eigenvalues (ascending): 1.5910689110926703, 2.3618374429898186,
    // 4.5024046887344715, 5.5446889571830384
    let expected = [
        5.5446889571830384,
        4.5024046887344715,
        2.3618374429898186,
        1.5910689110926703,
    ];
    for (i, e) in expected.iter().enumerate() {
        close(p.eigenvalues[i], *e, &format!("λ{i}"));
    }
    // explained ratios: 0.39604921122735987, 0.3216003349096051, 0.16870267449927276, 0.11364777936376216
    close(p.explained_variance_ratio[0], 0.39604921122735987, "ratio₀");
    close(p.explained_variance_ratio[1], 0.3216003349096051, "ratio₁");
    close(p.explained_variance_ratio[2], 0.16870267449927276, "ratio₂");
    close(p.explained_variance_ratio[3], 0.11364777936376216, "ratio₃");
    // eigh columns, signed so the first coordinate is positive:
    let vectors = [
        [
            0.30225746834694595,
            0.3150155112535152,
            0.3037565246562271,
            0.8468397866343743,
        ],
        [
            0.8058106750956385,
            0.3930746524744864,
            0.023536817851704948,
            -0.44227535731644474,
        ],
        [
            0.45176861940811536,
            -0.8575150098403882,
            0.2348891461876611,
            0.07348613079864326,
        ],
        [
            0.23497806971820137,
            -0.10452537084453019,
            -0.9230412130221585,
            0.2861025561995268,
        ],
    ];
    for (i, v) in vectors.iter().enumerate() {
        for (j, c) in v.iter().enumerate() {
            close(p.components[i][j], *c, &format!("v{i}{j}"));
        }
    }
    assert!(
        pca_f64(&[vec![1.0, 2.0], vec![3.0, 4.0]]).is_err(),
        "not symmetric"
    );
    assert!(pca_f64(&[vec![1.0, 2.0]]).is_err(), "not square");
    assert!(pca_f64(&[]).is_err());
}

#[test]
fn pca_f64_agrees_with_exact_pca() {
    let ctx = Context::new();
    let exact = pca(&ctx, &QMatrix::from_i64(&[&[2, 1], &[1, 3]]).unwrap()).unwrap();
    let float = pca_f64(&[vec![2.0, 1.0], vec![1.0, 3.0]]).unwrap();
    for i in 0..2 {
        close(
            float.eigenvalues[i],
            exact.eigenvalues[i].eval_f64().unwrap(),
            "λ",
        );
        for j in 0..2 {
            close(
                float.components[i][j],
                exact.components[i][j].eval_f64().unwrap(),
                "v",
            );
        }
    }
    // Also on the data covariance: eigh gives 19.4057566…, 1.5942434…, 0 (rank 2)
    let c = covariance_matrix(&data5x3(), Ddof::Sample).unwrap();
    let rows: Vec<Vec<f64>> = c
        .rows()
        .map(|r| {
            r.iter()
                .map(|v| ctx.from_ratio(v.clone()).eval_f64().unwrap())
                .collect()
        })
        .collect();
    let p = pca_f64(&rows).unwrap();
    close(
        p.eigenvalues[0] + p.eigenvalues[1] + p.eigenvalues[2],
        21.0,
        "trace 37/10 + 74/5 + 5/2",
    );
    assert!(
        p.eigenvalues[2].abs() < 1e-12,
        "rank-deficient: {}",
        p.eigenvalues[2]
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::order
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn uniform_max_of_three_has_density_3x2_and_exact_moments() {
    let ctx = Context::new();
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    let m = maximum_of(&u, 3).unwrap();
    let x = ctx.symbol("x");
    assert_eq!(m.density(&x), 3 * x.powi(2));
    assert_eq!(m.density(&ctx.rational(1, 2)).eval(), ctx.rational(3, 4));
    // Beta(3, 1): mean 3/4, variance 3/80 (scipy.stats.beta(3, 1).var() = 0.0375)
    assert_eq!(m.mean(), ctx.rational(3, 4));
    assert_eq!(m.variance(), ctx.rational(3, 80));
    assert_eq!(m.support(), u.support());
    assert_eq!(m.cdf(&ctx.rational(1, 2)), ctx.rational(1, 8));
}

#[test]
fn uniform_kth_of_n_has_mean_k_over_n_plus_1() {
    let ctx = Context::new();
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    // X_(2:5) ~ Beta(2, 4): density 20x(1−x)³, mean 1/3, variance 2/63
    // (scipy.stats.beta(2, 4).var() = 0.031746031746031744)
    let o = order_statistic(&u, 5, 2).unwrap();
    let x = ctx.symbol("x");
    is_true(
        &o.density(&x),
        &(20 * &x * (ctx.one() - &x).powi(3)),
        "density",
    );
    assert_eq!(o.density(&ctx.rational(1, 4)).eval(), ctx.rational(135, 64));
    assert_eq!(o.mean(), ctx.rational(1, 3));
    assert_eq!(o.variance(), ctx.rational(2, 63));
    for (n, k) in [(1usize, 1usize), (4, 1), (4, 4), (7, 3)] {
        let d = order_statistic(&u, n, k).unwrap();
        assert_eq!(
            d.mean(),
            ctx.rational(k as i64, n as i64 + 1),
            "E[U_({k}:{n})]"
        );
    }
}

#[test]
fn order_statistic_cdf_is_regularized_incomplete_beta() {
    let ctx = Context::new();
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    let o = order_statistic(&u, 5, 2).unwrap();
    // scipy.special.betainc(2, 4, 0.5) = 0.8125 = Σ_{j≥2} C(5, j)/32 = 13/16
    assert_eq!(o.cdf(&ctx.rational(1, 2)), ctx.rational(13, 16));
    assert_eq!(o.cdf(&ctx.int(-1)), ctx.zero());
    assert_eq!(o.cdf(&ctx.int(2)), ctx.one());
    // P(1/4 < X_(2) ≤ 1/2) through the CDF
    let f14 = o.cdf(&ctx.rational(1, 4));
    let region = symplex::stats::Support::interval(ctx.rational(1, 4), ctx.rational(1, 2));
    assert_eq!(
        o.probability_of(&region).unwrap(),
        (ctx.rational(13, 16) - f14).simplify()
    );
}

#[test]
fn exponential_min_of_n_is_exponential_n_lambda() {
    let ctx = Context::new();
    let e = Distribution::exponential(ctx.int(2));
    let m = minimum_of(&e, 3).unwrap();
    let x = ctx.symbol("x");
    // n(1−F)^{n−1} f = 3·e^{−4x}·2e^{−2x} = 6e^{−6x}
    assert_eq!(m.density(&x).simplify(), 6 * (-6 * &x).exp());
    let target = Distribution::exponential(ctx.int(6));
    is_true(
        &m.density(&x),
        &target.density(&x),
        "density = Exponential(6)",
    );
    // scipy.stats.expon(scale=1/6): pdf(0.1) = 3.2928698165641586, cdf(1) = 0.9975212478233336
    ex_close(
        &m.density(&ctx.rational(1, 10)),
        3.2928698165641586,
        "pdf(0.1)",
    );
    ex_close(&m.cdf(&ctx.int(1)), 0.9975212478233336, "cdf(1)");
    assert_eq!(m.mean(), ctx.rational(1, 6));
    assert_eq!(m.support(), e.support());
}

#[test]
fn normal_median_of_three_matches_scipy() {
    let ctx = Context::new();
    let n = Distribution::normal(ctx.int(0), ctx.int(1));
    let med = order_statistic(&n, 3, 2).unwrap();
    // 6·Φ(0.3)(1 − Φ(0.3))·φ(0.3) = 0.5402668775986347; I_{Φ(0.3)}(2, 2) = 0.6735884638744767
    ex_close(
        &med.density(&ctx.rational(3, 10)),
        0.5402668775986347,
        "density(0.3)",
    );
    ex_close(
        &med.cdf(&ctx.rational(3, 10)),
        0.6735884638744767,
        "cdf(0.3)",
    );
    // Symmetric about 0: F(0) = 1/2 exactly through I_{1/2}(2, 2) = 1/2
    assert_eq!(med.cdf(&ctx.int(0)), ctx.rational(1, 2));
}

#[test]
fn die_max_and_min_of_two_are_exact_finite_tables() {
    let ctx = Context::new();
    let die = Distribution::die(ctx.int(6));
    // P(max = v) = (2v − 1)/36: 1/36, 1/12, 5/36, 7/36, 1/4, 11/36; mean 161/36
    let mx = maximum_of(&die, 2).unwrap();
    let table = &mx.downcast_ref::<Finite>().expect("a Finite table").table;
    assert_eq!(table.len(), 6);
    for v in 1..=6 {
        assert_eq!(
            mx.density(&ctx.int(v)).eval(),
            ctx.rational(2 * v - 1, 36),
            "max = {v}"
        );
    }
    assert_eq!(mx.mean(), ctx.rational(161, 36));
    // P(min = v) = (13 − 2v)/36: 11/36, 1/4, 7/36, 5/36, 1/12, 1/36; mean 91/36
    let mn = minimum_of(&die, 2).unwrap();
    for v in 1..=6 {
        assert_eq!(
            mn.density(&ctx.int(v)).eval(),
            ctx.rational(13 - 2 * v, 36),
            "min = {v}"
        );
    }
    assert_eq!(mn.mean(), ctx.rational(91, 36));
    // E[min] + E[max] = 2·E[X] = 7
    assert_eq!(mx.mean() + mn.mean(), ctx.int(7));
}

#[test]
fn finite_table_median_of_three_by_exact_enumeration() {
    let ctx = Context::new();
    let fin = Distribution::finite(
        &ctx,
        vec![
            (ctx.int(1), ctx.rational(1, 2)),
            (ctx.int(2), ctx.rational(1, 3)),
            (ctx.int(3), ctx.rational(1, 6)),
        ],
    );
    // Fraction enumeration over all 27 outcomes: P(median = 1, 2, 3) = 1/2, 23/54, 2/27
    let med = order_statistic(&fin, 3, 2).unwrap();
    assert_eq!(med.density(&ctx.int(1)).eval(), ctx.rational(1, 2));
    assert_eq!(med.density(&ctx.int(2)).eval(), ctx.rational(23, 54));
    assert_eq!(med.density(&ctx.int(3)).eval(), ctx.rational(2, 27));
    let total = (1..=3).fold(ctx.zero(), |acc, v| acc + med.density(&ctx.int(v)).eval());
    assert_eq!(total, ctx.one());
    // Binomial(2, 1/2) is a finite lattice: max of two has P(max = 2) = 1 − (3/4)² = 7/16
    let b = Distribution::binomial(ctx.int(2), ctx.rational(1, 2));
    let bm = maximum_of(&b, 2).unwrap();
    assert_eq!(bm.density(&ctx.int(2)).eval(), ctx.rational(7, 16));
    assert_eq!(bm.density(&ctx.int(0)).eval(), ctx.rational(1, 16));
}

#[test]
fn order_statistic_sampling_means() {
    let ctx = Context::new();
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    let m = maximum_of(&u, 3).unwrap();
    let draws = m.sample(4000, &mut Rng::new(7)).unwrap();
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    assert!((mean - 0.75).abs() < 0.02, "E[max of 3] ≈ 0.75, got {mean}");
    assert!(draws.iter().all(|&x| (0.0..=1.0).contains(&x)));
    let again = m.sample(4000, &mut Rng::new(7)).unwrap();
    assert_eq!(draws, again, "deterministic");
    // The exactly enumerated die table samples through the generic route.
    let dm = maximum_of(&Distribution::die(ctx.int(6)), 2).unwrap();
    let d = dm.sample(4000, &mut Rng::new(3)).unwrap();
    let dmean = d.iter().sum::<f64>() / d.len() as f64;
    assert!(
        (dmean - 161.0 / 36.0).abs() < 0.1,
        "E[die max of 2] = 4.4722, got {dmean}"
    );
}

#[test]
fn order_statistic_rejects_bad_ranks() {
    let ctx = Context::new();
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    assert!(order_statistic(&u, 5, 0).is_err());
    assert!(order_statistic(&u, 5, 6).is_err());
    assert!(order_statistic(&u, 0, 0).is_err());
    assert!(OrderStatistic::try_new(u.clone(), 3, 4).is_err());
    assert!(maximum_of(&u, 0).is_err());
}

#[test]
fn order_statistic_family_metadata_and_min_max_of_two() {
    let ctx = Context::new();
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    let o = order_statistic(&u, 5, 2).unwrap();
    assert_eq!(o.name(), "OrderStatistic");
    let params = o.parameters();
    assert_eq!(params[0], ("n", ctx.int(5)));
    assert_eq!(params[1], ("k", ctx.int(2)));
    assert!(format!("{o}").starts_with("OrderStatistic(k=2, n=5, "));
    let fam = o.downcast_ref::<OrderStatistic>().expect("family");
    assert_eq!((fam.n, fam.k), (5, 2));
    assert_eq!(o, order_statistic(&u, 5, 2).unwrap());
    assert_ne!(o, order_statistic(&u, 5, 3).unwrap());
    // E[min of 2] = 1/3, E[max of 2] = 2/3 for Uniform(0, 1)
    assert_eq!(minimum_of(&u, 2).unwrap().mean(), ctx.rational(1, 3));
    assert_eq!(maximum_of(&u, 2).unwrap().mean(), ctx.rational(2, 3));
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::information
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn entropy_exact_cases_and_scipy_values() {
    let ctx = Context::new();
    let p = [q(1, 2), q(1, 4), q(1, 4)];
    // scipy.stats.entropy([.5, .25, .25], base=2) = 1.5; entropy(...) = 1.0397207708399179
    assert_eq!(entropy(&ctx, &p, Base::Bits).unwrap(), ctx.rational(3, 2));
    let nats = entropy(&ctx, &p, Base::Nats).unwrap();
    assert_eq!(nats, ctx.rational(3, 2) * ctx.int(2).ln());
    ex_close(&nats, 1.0397207708399179, "H nats");
    // scipy.stats.entropy([0.2, 0.3, 0.5]) = 1.0296530140645737
    let r = [q(1, 5), q(3, 10), q(1, 2)];
    ex_close(
        &entropy(&ctx, &r, Base::Nats).unwrap(),
        1.0296530140645737,
        "H(1/5, 3/10, 1/2)",
    );
    // 0·ln 0 = 0; a fair coin is one bit; a fair die ln 6 nats
    assert_eq!(
        entropy(&ctx, &[qi(1), qi(0)], Base::Nats).unwrap(),
        ctx.zero()
    );
    assert_eq!(
        entropy(&ctx, &[q(1, 2), q(1, 2)], Base::Bits).unwrap(),
        ctx.one()
    );
    // … in the canonical prime-split form ln 2 + ln 3 (= ln 6 = 1.791759469228055)
    let die = entropy(&ctx, &vec![q(1, 6); 6], Base::Nats).unwrap();
    assert_eq!(die, ctx.int(2).ln() + ctx.int(3).ln());
    ex_close(&die, 1.791759469228055, "ln 6");
    // H(1/3, 2/3) = ln 3 − (2/3) ln 2 nats = log₂3 − 2/3 bits
    let h = entropy(&ctx, &[q(1, 3), q(2, 3)], Base::Bits).unwrap();
    ex_close(&h, 0.9182958340544896, "H(1/3, 2/3) bits");
    is_true(
        &h,
        &(ctx.int(3).ln() / ctx.int(2).ln() - ctx.rational(2, 3)),
        "H(1/3, 2/3) form",
    );
}

#[test]
fn kl_divergence_matches_scipy_entropy_pk_qk() {
    let ctx = Context::new();
    let p = [q(1, 2), q(1, 4), q(1, 4)];
    let u = [q(1, 3), q(1, 3), q(1, 3)];
    // scipy.stats.entropy(p, u) = 0.05889151782819178 = ln 3 − (3/2) ln 2
    let kl = kl_divergence(&ctx, &p, &u, Base::Nats).unwrap();
    ex_close(&kl, 0.05889151782819178, "KL(p‖u)");
    assert_eq!(kl, ctx.int(3).ln() - ctx.rational(3, 2) * ctx.int(2).ln());
    // scipy.stats.entropy(u, p) = 0.056633012265132426 — not symmetric
    ex_close(
        &kl_divergence(&ctx, &u, &p, Base::Nats).unwrap(),
        0.056633012265132426,
        "KL(u‖p)",
    );
    // In bits: log₂3 − 3/2
    let bits = kl_divergence(&ctx, &p, &u, Base::Bits).unwrap();
    ex_close(
        &bits,
        0.05889151782819178 / std::f64::consts::LN_2,
        "KL bits",
    );
    assert_eq!(kl_divergence(&ctx, &p, &p, Base::Nats).unwrap(), ctx.zero());
}

#[test]
fn kl_and_cross_entropy_are_infinite_when_q_vanishes() {
    let ctx = Context::new();
    let p = [q(1, 2), q(1, 2)];
    let q0 = [qi(1), qi(0)];
    assert!(kl_divergence(&ctx, &p, &q0, Base::Nats).is_err());
    assert!(cross_entropy(&ctx, &p, &q0, Base::Nats).is_err());
    // …but zero mass in p is fine: KL((1, 0) ‖ (1/2, 1/2)) = ln 2
    assert_eq!(
        kl_divergence(&ctx, &q0, &p, Base::Nats).unwrap(),
        ctx.int(2).ln()
    );
    assert_eq!(kl_divergence(&ctx, &q0, &p, Base::Bits).unwrap(), ctx.one());
}

#[test]
fn js_divergence_is_square_of_scipy_jensenshannon() {
    let ctx = Context::new();
    let p = [q(1, 2), q(1, 4), q(1, 4)];
    let u = [q(1, 3), q(1, 3), q(1, 3)];
    // scipy.spatial.distance.jensenshannon(p, u) = 0.11984403015647774 (a distance, base e);
    // squared: 0.014362591564146746 = the divergence
    let js = js_divergence(&ctx, &p, &u, Base::Nats).unwrap();
    ex_close(&js, 0.11984403015647774_f64.powi(2), "JS = jensenshannon²");
    ex_close(&js, 0.014362591564146746, "JS nats");
    ex_close(
        &js_divergence(&ctx, &p, &u, Base::Bits).unwrap(),
        0.02072083962390817,
        "JS bits",
    );
    assert_eq!(
        js,
        js_divergence(&ctx, &u, &p, Base::Nats).unwrap(),
        "symmetric"
    );
    assert_eq!(js_divergence(&ctx, &p, &p, Base::Nats).unwrap(), ctx.zero());
    // Disjoint supports: one bit exactly, no error.
    assert_eq!(
        js_divergence(&ctx, &[qi(1), qi(0)], &[qi(0), qi(1)], Base::Bits).unwrap(),
        ctx.one()
    );
}

#[test]
fn distances_total_variation_hellinger_bhattacharyya() {
    let ctx = Context::new();
    let p = [q(1, 2), q(1, 4), q(1, 4)];
    let u = [q(1, 3), q(1, 3), q(1, 3)];
    // ½ Σ|pᵢ − qᵢ| = ½(1/6 + 1/12 + 1/12) = 1/6
    assert_eq!(total_variation(&p, &u).unwrap(), q(1, 6));
    assert_eq!(
        total_variation(&[qi(1), qi(0)], &[qi(0), qi(1)]).unwrap(),
        qi(1)
    );
    // BC = Σ√(pᵢqᵢ) = √(1/6) + 2√(1/12) = 0.9855985596534887;
    // Hellinger √(1 − BC) = 0.12000600129373219; −ln BC = 0.014506147594484509
    let bc = bhattacharyya_coefficient(&ctx, &p, &u).unwrap();
    ex_close(&bc, 0.9855985596534887, "BC");
    is_true(
        &bc,
        &(ctx.rational(1, 6).sqrt() + 2 * ctx.rational(1, 12).sqrt()),
        "BC form",
    );
    ex_close(
        &hellinger(&ctx, &p, &u).unwrap(),
        0.12000600129373219,
        "Hellinger",
    );
    ex_close(
        &bhattacharyya_distance(&ctx, &p, &u).unwrap(),
        0.014506147594484509,
        "Bhattacharyya",
    );
    assert_eq!(hellinger(&ctx, &p, &p).unwrap(), ctx.zero());
    assert_eq!(
        hellinger(&ctx, &[qi(1), qi(0)], &[qi(0), qi(1)]).unwrap(),
        ctx.one()
    );
    assert!(bhattacharyya_distance(&ctx, &[qi(1), qi(0)], &[qi(0), qi(1)]).is_err());
}

#[test]
fn cross_entropy_equals_entropy_plus_kl() {
    let ctx = Context::new();
    let p = [q(1, 2), q(1, 4), q(1, 4)];
    let u = [q(1, 3), q(1, 3), q(1, 3)];
    // −Σ pᵢ ln(1/3) = ln 3 = 1.0986122886681098 = H(p) + KL(p‖u)
    let ce = cross_entropy(&ctx, &p, &u, Base::Nats).unwrap();
    assert_eq!(ce, ctx.int(3).ln());
    ex_close(&ce, 1.0986122886681098, "H(p, u)");
    let h_plus_kl =
        entropy(&ctx, &p, Base::Nats).unwrap() + kl_divergence(&ctx, &p, &u, Base::Nats).unwrap();
    is_true(&ce, &h_plus_kl, "H(p, q) = H(p) + D(p‖q)");
    assert_eq!(
        cross_entropy(&ctx, &p, &p, Base::Bits).unwrap(),
        ctx.rational(3, 2)
    );
}

#[test]
fn mutual_information_cover_thomas_example() {
    let ctx = Context::new();
    let joint = cover_thomas_joint();
    // I(X; Y) = 3/8 bit = 0.25993019270997947 nats
    assert_eq!(
        mutual_information(&ctx, &joint, Base::Bits).unwrap(),
        ctx.rational(3, 8)
    );
    let nats = mutual_information(&ctx, &joint, Base::Nats).unwrap();
    assert_eq!(nats, ctx.rational(3, 8) * ctx.int(2).ln());
    ex_close(&nats, 0.25993019270997947, "I nats");
    assert_eq!(
        information_gain(&ctx, &joint, Base::Bits).unwrap(),
        mutual_information(&ctx, &joint, Base::Bits).unwrap()
    );
    // I = H(X) + H(Y) − H(X, Y) = 2 + 7/4 − 27/8
    let (rows, cols) = info::marginals(&joint).unwrap();
    assert_eq!(rows, vec![q(1, 4), q(1, 4), q(1, 4), q(1, 4)]);
    assert_eq!(cols, vec![q(1, 2), q(1, 4), q(1, 8), q(1, 8)]);
    let hx = entropy(&ctx, &rows, Base::Bits).unwrap();
    let hy = entropy(&ctx, &cols, Base::Bits).unwrap();
    let hxy = joint_entropy(&ctx, &joint, Base::Bits).unwrap();
    assert_eq!(
        (hx, hy, hxy),
        (ctx.int(2), ctx.rational(7, 4), ctx.rational(27, 8))
    );
}

#[test]
fn conditional_entropies_cover_thomas_example() {
    let ctx = Context::new();
    let joint = cover_thomas_joint();
    // H(col | row) = 27/8 − 2 = 11/8; H(row | col) = 27/8 − 7/4 = 13/8 (bits)
    assert_eq!(
        conditional_entropy(&ctx, &joint, Given::Row, Base::Bits).unwrap(),
        ctx.rational(11, 8)
    );
    assert_eq!(
        conditional_entropy(&ctx, &joint, Given::Column, Base::Bits).unwrap(),
        ctx.rational(13, 8)
    );
    // Chain rule: H(X, Y) = H(row) + H(col | row)
    let (rows, _) = info::marginals(&joint).unwrap();
    let chain = entropy(&ctx, &rows, Base::Nats).unwrap()
        + conditional_entropy(&ctx, &joint, Given::Row, Base::Nats).unwrap();
    is_true(
        &chain,
        &joint_entropy(&ctx, &joint, Base::Nats).unwrap(),
        "chain rule",
    );
    // I = H(row) − H(row | col) = 2 − 13/8 = 3/8
    let ig = ctx.int(2) - conditional_entropy(&ctx, &joint, Given::Column, Base::Bits).unwrap();
    assert_eq!(ig, ctx.rational(3, 8));
}

#[test]
fn normalized_mutual_information_variants() {
    let ctx = Context::new();
    let joint = cover_thomas_joint();
    // I = 3/8, H(X) = 2, H(Y) = 7/4 (bits; the ratios are base-free):
    // arithmetic 0.2, geometric 3/(8√(7/2)) = 3√14/56 = 0.2004459314343183, min 3/14, max 3/16
    assert_eq!(
        normalized_mutual_information(&ctx, &joint, Norm::Arithmetic).unwrap(),
        ctx.rational(1, 5)
    );
    let geo = normalized_mutual_information(&ctx, &joint, Norm::Geometric).unwrap();
    is_true(&geo, &(3 * ctx.int(14).sqrt() / 56), "geometric");
    ex_close(&geo, 0.2004459314343183, "geometric");
    assert_eq!(
        normalized_mutual_information(&ctx, &joint, Norm::Min).unwrap(),
        ctx.rational(3, 14)
    );
    assert_eq!(
        normalized_mutual_information(&ctx, &joint, Norm::Max).unwrap(),
        ctx.rational(3, 16)
    );
    // Perfect dependence: NMI = 1 under every normalisation.
    let diag = vec![vec![q(1, 2), qi(0)], vec![qi(0), q(1, 2)]];
    for n in [Norm::Arithmetic, Norm::Geometric, Norm::Min, Norm::Max] {
        assert_eq!(
            normalized_mutual_information(&ctx, &diag, n).unwrap(),
            ctx.one(),
            "{n:?}"
        );
    }
    // A degenerate margin has zero entropy: Min is undefined.
    let degenerate = vec![vec![q(1, 2), q(1, 2)]];
    assert!(normalized_mutual_information(&ctx, &degenerate, Norm::Min).is_err());
}

#[test]
fn joint_from_counts_and_independence() {
    let ctx = Context::new();
    // counts [[2, 1], [1, 4]] / 8
    let joint = joint_from_counts(&[vec![2, 1], vec![1, 4]]).unwrap();
    assert_eq!(joint, vec![vec![q(1, 4), q(1, 8)], vec![q(1, 8), q(1, 2)]]);
    // Σ pᵢⱼ ln(pᵢⱼ / (pᵢ· p·ⱼ)) = 0.1101189103360598 nats (Fraction + math.log)
    let i = mutual_information(&ctx, &joint, Base::Nats).unwrap();
    ex_close(&i, 0.1101189103360598, "I from counts");
    // An outer product has I = 0 exactly and H(Y|X) = H(Y).
    let p = [q(1, 2), q(1, 3), q(1, 6)];
    let r = [q(1, 4), q(3, 4)];
    let indep: Vec<Vec<Q>> = p
        .iter()
        .map(|a| r.iter().map(|b| a * b).collect())
        .collect();
    assert_eq!(
        mutual_information(&ctx, &indep, Base::Bits).unwrap(),
        ctx.zero()
    );
    assert_eq!(
        conditional_entropy(&ctx, &indep, Given::Row, Base::Nats).unwrap(),
        entropy(&ctx, &r, Base::Nats).unwrap()
    );
    assert!(joint_from_counts(&[vec![0, 0]]).is_err());
    assert!(joint_from_counts(&[vec![1], vec![1, 2]]).is_err());
    assert!(joint_from_counts(&[]).is_err());
}

#[test]
fn perplexity_and_probability_vector() {
    let ctx = Context::new();
    // exp(H) for a fair four-sided die is 4; for (½, ¼, ¼) it is 2^{3/2} = 2.8284271247461903
    assert_eq!(perplexity(&ctx, &vec![q(1, 4); 4]).unwrap(), ctx.int(4));
    let pp = perplexity(&ctx, &[q(1, 2), q(1, 4), q(1, 4)]).unwrap();
    is_true(&pp, &ctx.int(2).pow(&ctx.rational(3, 2)), "2^{3/2}");
    ex_close(&pp, 2.8284271247461903, "perplexity");
    // A die and a Binomial(2, 1/2) as probability vectors
    let die = probability_vector(&Distribution::die(ctx.int(6))).unwrap();
    assert_eq!(die, vec![q(1, 6); 6]);
    assert_eq!(
        entropy(&ctx, &die, Base::Nats).unwrap(),
        ctx.int(2).ln() + ctx.int(3).ln()
    );
    let b = probability_vector(&Distribution::binomial(ctx.int(2), ctx.rational(1, 2))).unwrap();
    assert_eq!(b, vec![q(1, 4), q(1, 2), q(1, 4)]);
    assert_eq!(entropy(&ctx, &b, Base::Bits).unwrap(), ctx.rational(3, 2));
    let fin = Distribution::finite(
        &ctx,
        vec![
            (ctx.int(7), ctx.rational(2, 3)),
            (ctx.int(9), ctx.rational(1, 3)),
        ],
    );
    assert_eq!(probability_vector(&fin).unwrap(), vec![q(2, 3), q(1, 3)]);
    assert!(probability_vector(&Distribution::uniform(ctx.int(0), ctx.int(1))).is_err());
    assert!(probability_vector(&Distribution::poisson(ctx.int(2))).is_err());
}

#[test]
fn information_functions_validate_their_inputs() {
    let ctx = Context::new();
    assert!(
        entropy(&ctx, &[q(1, 2), q(1, 3)], Base::Nats).is_err(),
        "sums to 5/6"
    );
    assert!(
        entropy(&ctx, &[q(3, 2), q(-1, 2)], Base::Nats).is_err(),
        "negative"
    );
    assert!(entropy(&ctx, &[], Base::Nats).is_err(), "empty");
    assert!(
        kl_divergence(&ctx, &[q(1, 2), q(1, 2)], &[qi(1)], Base::Nats).is_err(),
        "lengths"
    );
    assert!(total_variation(&[q(1, 2), q(1, 2)], &[q(1, 2), q(1, 3)]).is_err());
    assert!(
        mutual_information(&ctx, &[vec![q(1, 2)], vec![q(1, 2), qi(0)]], Base::Nats).is_err(),
        "jagged"
    );
    assert!(
        mutual_information(&ctx, &[vec![q(1, 2), q(1, 3)]], Base::Nats).is_err(),
        "sum"
    );
    assert!(joint_entropy(&ctx, &[], Base::Nats).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::sequential
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sprt_boundaries_match_wald() {
    // A = ln(β/(1−α)) = ln(0.1/0.95) = −2.251291798606495; B = ln((1−β)/α) = ln(0.9/0.05) = 2.8903717578961645
    let (a, b) = wald_boundaries(0.05, 0.10).unwrap();
    close(a, -2.251291798606495, "A");
    close(b, 2.8903717578961645, "B");
    let test = Sprt::bernoulli(q(7, 10), q(9, 10), 0.05, 0.10).unwrap();
    assert_eq!(test.boundaries(), (a, b));
    assert_eq!((test.alpha(), test.beta()), (0.05, 0.10));
    assert_eq!(test.decision(), Decision::Continue);
    assert!(wald_boundaries(0.0, 0.1).is_err());
    assert!(wald_boundaries(0.6, 0.5).is_err(), "α + β ≥ 1");
}

#[test]
fn sprt_good_worker_is_accepted_after_twelve_successes() {
    // ln(9/7) = 0.25131442828090617 per success; ⌈2.8903717578961645 / 0.25131442828090617⌉ = 12
    let mut test = Sprt::bernoulli(q(7, 10), q(9, 10), 0.05, 0.10).unwrap();
    let mut n = 0;
    loop {
        n += 1;
        match test.update(true) {
            Decision::Continue => assert!(n < 12, "still undecided after {n}"),
            Decision::AcceptH1 => break,
            Decision::AcceptH0 => panic!("a perfect worker was rejected"),
        }
    }
    assert_eq!(n, 12);
    assert_eq!(test.observations(), 12);
    assert_eq!(test.successes(), 12);
    assert_eq!(test.failures(), 0);
    close(
        test.log_likelihood_ratio_f64(),
        12.0 * 0.25131442828090617,
        "Λ₁₂",
    );
}

#[test]
fn sprt_bad_worker_is_rejected_after_three_failures() {
    // ln(1/3) = −1.0986122886681098 per failure; ⌈2.251291798606495 / 1.0986122886681098⌉ = 3
    let mut test = Sprt::bernoulli(q(7, 10), q(9, 10), 0.05, 0.10).unwrap();
    assert_eq!(test.update(false), Decision::Continue);
    assert_eq!(test.update(false), Decision::Continue);
    assert_eq!(test.update(false), Decision::AcceptH0);
    assert_eq!(test.observations(), 3);
    assert_eq!(test.successes(), 0);
}

#[test]
fn sprt_mixed_sequence_decisions_match_python_replay() {
    // T T F T T F F: Continue ×6, then AcceptH0 at the 7th observation (Python replay of Wald's rule)
    let mut test = Sprt::bernoulli(q(7, 10), q(9, 10), 0.05, 0.10).unwrap();
    let seq = [true, true, false, true, true, false, false];
    let decisions: Vec<Decision> = seq.iter().map(|&s| test.update(s)).collect();
    assert_eq!(&decisions[..6], &[Decision::Continue; 6]);
    assert_eq!(decisions[6], Decision::AcceptH0);
    assert_eq!((test.successes(), test.failures()), (4, 3));
}

#[test]
fn sprt_log_likelihood_ratio_is_exact() {
    let ctx = Context::new();
    let mut test = Sprt::bernoulli(q(7, 10), q(9, 10), 0.05, 0.10).unwrap();
    for s in [true, true, false, true, false] {
        test.update(s);
    }
    // 3·ln(9/7) + 2·ln(1/3) = −1.443281292493501
    let llr = test.log_likelihood_ratio(&ctx);
    let exact = 3 * ctx.rational(9, 7).ln() + 2 * ctx.rational(1, 3).ln();
    is_true(&llr, &exact, "Λ");
    ex_close(&llr, -1.443281292493501, "Λ numeric");
    close(test.log_likelihood_ratio_f64(), -1.443281292493501, "Λ f64");
    assert_eq!(test.decision(), Decision::Continue);
    // Before any observation the ratio is exactly 0.
    let fresh = Sprt::bernoulli(q(7, 10), q(9, 10), 0.05, 0.10).unwrap();
    assert_eq!(fresh.log_likelihood_ratio(&ctx), ctx.zero());
}

#[test]
fn sprt_reset_counters_and_determinism() {
    let ctx = Context::new();
    let mut a = Sprt::bernoulli(q(7, 10), q(9, 10), 0.05, 0.10).unwrap();
    let mut b = a.clone();
    let seq = [
        true, false, true, true, true, false, true, true, true, true, true, true, true, true,
    ];
    let da: Vec<Decision> = seq.iter().map(|&s| a.update(s)).collect();
    let db: Vec<Decision> = seq.iter().map(|&s| b.update(s)).collect();
    assert_eq!(da, db, "identical designs, identical decisions");
    assert_eq!(a, b);
    assert_eq!(a.observations(), seq.len());
    assert_eq!(a.successes(), 12);
    assert_eq!(*a.sum(), qi(12));
    a.reset();
    assert_eq!(a.observations(), 0);
    assert_eq!(a.successes(), 0);
    assert_eq!(a.log_likelihood_ratio(&ctx), ctx.zero());
    assert_eq!(a.decision(), Decision::Continue);
    assert_eq!(
        a.boundaries(),
        b.boundaries(),
        "the design survives a reset"
    );
    // observe() accepts only 0/1 for a Bernoulli test.
    assert_eq!(a.observe(qi(1)).unwrap(), Decision::Continue);
    assert!(a.observe(q(1, 2)).is_err());
    assert_eq!(a.successes(), 1);
}

#[test]
fn sprt_normal_mean_llr_is_exact_in_the_data() {
    let ctx = Context::new();
    // μ₀ = 0, μ₁ = 1, σ = 1: Λ = Σ(xᵢ − ½); observations 1/2, 3/2 give Λ = 1 exactly
    let mut test = Sprt::normal_mean(qi(0), qi(1), qi(1), 0.05, 0.10).unwrap();
    assert_eq!(test.observe(q(1, 2)).unwrap(), Decision::Continue);
    assert_eq!(test.log_likelihood_ratio(&ctx), ctx.zero());
    assert_eq!(test.observe(q(3, 2)).unwrap(), Decision::Continue);
    assert_eq!(test.log_likelihood_ratio(&ctx), ctx.one());
    close(test.log_likelihood_ratio_f64(), 1.0, "Λ");
    assert_eq!(test.observations(), 2);
    assert_eq!(*test.sum(), qi(2));
    // Large observations cross B = 2.8904 quickly: x = 4 adds 3.5.
    assert_eq!(test.observe(qi(4)).unwrap(), Decision::AcceptH1);
    assert!(
        Sprt::normal_mean(qi(0), qi(1), qi(0), 0.05, 0.10).is_err(),
        "σ = 0"
    );
    assert!(
        Sprt::normal_mean(qi(1), qi(1), qi(1), 0.05, 0.10).is_err(),
        "μ₀ = μ₁"
    );
}

#[test]
fn expected_sample_size_and_oc_match_wald_formulas() {
    let (p0, p1) = (q(7, 10), q(9, 10));
    // At p₀: ((1−α)A + αB) / E_{p₀}[Z] = 12.977756554177112; L(p₀) = 1 − α
    close(
        expected_sample_size_bernoulli(0.7, &p0, &p1, 0.05, 0.10).unwrap(),
        12.977756554177112,
        "E_p0[N]",
    );
    close(
        operating_characteristic_bernoulli(0.7, &p0, &p1, 0.05, 0.10).unwrap(),
        0.95,
        "L(p0)",
    );
    // At p₁: (βA + (1−β)B) / E_{p₁}[Z] = 20.427867253612252; L(p₁) = β
    close(
        expected_sample_size_bernoulli(0.9, &p0, &p1, 0.05, 0.10).unwrap(),
        20.427867253612252,
        "E_p1[N]",
    );
    close(
        operating_characteristic_bernoulli(0.9, &p0, &p1, 0.05, 0.10).unwrap(),
        0.10,
        "L(p1)",
    );
    // At p = 0.8 (scipy.optimize.brentq for h = 0.13281371651409804): L = 0.6442212119279843, E[N] = 22.601836457327206
    close(
        operating_characteristic_bernoulli(0.8, &p0, &p1, 0.05, 0.10).unwrap(),
        0.6442212119279843,
        "L(0.8)",
    );
    close(
        expected_sample_size_bernoulli(0.8, &p0, &p1, 0.05, 0.10).unwrap(),
        22.601836457327206,
        "E_0.8[N]",
    );
    assert!(expected_sample_size_bernoulli(1.5, &p0, &p1, 0.05, 0.10).is_err());
}

#[test]
fn sprt_rejects_invalid_parameters() {
    assert!(
        Sprt::bernoulli(q(7, 10), q(7, 10), 0.05, 0.10).is_err(),
        "p0 = p1"
    );
    assert!(
        Sprt::bernoulli(qi(0), q(9, 10), 0.05, 0.10).is_err(),
        "p0 = 0"
    );
    assert!(
        Sprt::bernoulli(q(7, 10), qi(1), 0.05, 0.10).is_err(),
        "p1 = 1"
    );
    assert!(
        Sprt::bernoulli(q(7, 10), q(9, 10), 1.0, 0.10).is_err(),
        "α = 1"
    );
    assert!(
        Sprt::bernoulli(q(7, 10), q(9, 10), 0.05, -0.1).is_err(),
        "β < 0"
    );
    assert!(
        Sprt::bernoulli(q(7, 10), q(9, 10), 0.5, 0.5).is_err(),
        "α + β = 1"
    );
    // p₁ < p₀ is a valid design too (the boundaries are the same, the increments flip).
    let mut reversed = Sprt::bernoulli(q(9, 10), q(7, 10), 0.05, 0.10).unwrap();
    assert_eq!(reversed.update(false), Decision::Continue);
    assert!(reversed.log_likelihood_ratio_f64() > 0.0);
}
