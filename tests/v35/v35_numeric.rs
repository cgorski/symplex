//! Numerical root finding and fitting, integer normal forms and physical
//! units after the 0.40 hunt.  Every test names what was wrong before and
//! the oracle of its reference: SciPy (`scipy.optimize.brentq`, `bisect`,
//! `numpy.trapezoid`), exact rational least squares (SymPy `Matrix.solve`
//! of the normal equations), SymPy 1.14 `hermite_normal_form` /
//! `smith_normal_form`, or an exact property that pins the answer (a sign
//! change between adjacent floats, the lattice of `A` equal to that of `H`,
//! the chain rule).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};
use symplex::matrix::{QMatrix, ZMatrix};
use symplex::optimize::{
    RootOpts, bisect, brent_root, linear_fit, poly_fit, poly_fit_exact, trapezoid,
};
use symplex::prelude::*;
use symplex::units::*;

fn q(x: f64) -> Ratio<BigInt> {
    Ratio::from_float(x).unwrap()
}

// ─────────────────────────────── root finding ───────────────────────────────

/// A sign change through a pole is not a root.  Before: `brent_root(tan,
/// 1, 2)` returned `1.570796326793586` and `bisect` `1.5707963267950618`
/// (π/2, where |tan| ≈ 10¹²), and `1/(x − 0.3) + 2` on `[0.25, 0.31]`
/// returned `0.3` — SciPy 1.x `brentq` returns the same poles; mpmath
/// `findroot` finds no root there.  The bracket around the genuine root π
/// still converges.
#[test]
fn bracketing_root_finders_refuse_a_pole() {
    let opts = RootOpts::default();
    for r in [
        brent_root(f64::tan, 1.0, 2.0, &opts),
        bisect(f64::tan, 1.0, 2.0, &opts),
    ] {
        assert!(
            matches!(r, Err(SymplexError::ComputationFailed { .. })),
            "{r:?}"
        );
    }
    let recip = |x: f64| 1.0 / (x - 0.3) + 2.0;
    assert!(brent_root(recip, 0.25, 0.31, &opts).is_err());
    assert!(bisect(recip, 0.25, 0.31, &opts).is_err());
    // The root of recip is 0.3 − 1/2 = −0.2; tan's root π.
    assert!((brent_root(recip, -1.0, 0.25, &opts).unwrap() + 0.2).abs() < 1e-12);
    assert!((brent_root(f64::tan, 3.0, 3.3, &opts).unwrap() - std::f64::consts::PI).abs() < 1e-12);
    assert!((bisect(f64::tan, 3.0, 3.3, &opts).unwrap() - std::f64::consts::PI).abs() < 1e-11);
}

/// A bracket reaching `±f64::MAX`.  Before: `brent_root` returned `−∞`
/// (`0.5·(c − b)` overflowed and the iterate jumped to infinity, where
/// `atan` is finite), `bisect` `+∞` for a steep `atan`.  Oracle: the sign
/// of `atan(k(x − r))` is that of `x − r`.
#[test]
fn root_finders_survive_brackets_at_the_float_limits() {
    let opts = RootOpts::default();
    let r = 3.691_732_523_024_782_6;
    let f = |x: f64| (0.005_101_480_868_353_455 * (x - r)).atan();
    let x = brent_root(f, -f64::MAX, f64::MAX, &opts).unwrap();
    assert!((x - r).abs() < 1e-11, "{x}");
    let g = |x: f64| (7_738_637.162_667_296 * (x - 2.290_885_218_528_304)).atan();
    let x = bisect(g, -f64::MAX, f64::MAX, &opts).unwrap();
    assert!((x - 2.290_885_218_528_304).abs() < 1e-11, "{x}");
}

/// A triple root, where Brent's interpolation creeps.  Before: "did not
/// converge within 100 iterations" (SciPy `brentq` on the same bracket:
/// "Failed to converge after 100 iterations").  The function
/// `s·(x − r)³` changes sign exactly at `r`.
#[test]
fn brent_converges_on_a_triple_root() {
    let opts = RootOpts::default();
    for (r, s, a, b) in [
        (
            -3.732_691_433_039,
            -46.993_907_865_85,
            -4.233_456_654_491,
            -3.730_208_542_931,
        ),
        (
            0.202_733_970_612,
            10.603_170_319_504,
            -81.679_041_210_82,
            10.874_921_150_73,
        ),
    ] {
        let x = brent_root(|x: f64| s * (x - r).powi(3), a, b, &opts).unwrap();
        assert!(
            (x - r).abs() <= opts.xtol + opts.rtol * r.abs(),
            "{x} vs {r}"
        );
    }
}

/// `xtol = rtol = 0` asks for the closest floats around the sign change.
/// Before: `brent_root` and `bisect` stalled on two adjacent floats and
/// reported "did not converge within 100 iterations".  Oracle: `f`
/// changes sign between the answer and one of its neighbouring floats.
#[test]
fn zero_tolerance_ends_at_adjacent_floats() {
    let opts = RootOpts {
        xtol: 0.0,
        rtol: 0.0,
        ..RootOpts::default()
    };
    let f = |x: f64| x.exp() - 0.016_703_290_333_597_003;
    let up = |x: f64| f64::from_bits(x.to_bits() - 1); // towards −∞ for x < 0
    let down = |x: f64| f64::from_bits(x.to_bits() + 1);
    for x in [
        brent_root(f, -5.535_953_805_763_644, -4.091_989_668_365_889, &opts).unwrap(),
        bisect(f, -5.535_953_805_763_644, -4.091_989_668_365_889, &opts).unwrap(),
    ] {
        let changes = |y: f64| (f(x) > 0.0) != (f(y) > 0.0) || f(x) == 0.0;
        assert!(changes(up(x)) || changes(down(x)), "{x}");
    }
}

// ─────────────────────────────── fitting ───────────────────────────────

/// `Σ (c_j − e_j)·x^j` over the samples, relative to `‖y‖`, in exact
/// arithmetic: how far the fitted values of `c` are from those of the
/// exact least-squares coefficients `e`.
fn fitted_value_error(xs: &[f64], ys: &[f64], c: &[f64], e: &[Ratio<BigInt>]) -> f64 {
    let (mut err2, mut y2) = (Ratio::<BigInt>::zero(), Ratio::<BigInt>::zero());
    for (&x, &y) in xs.iter().zip(ys) {
        let (mut pw, mut d) = (Ratio::<BigInt>::one(), Ratio::<BigInt>::zero());
        for (cj, ej) in c.iter().zip(e) {
            d += (q(*cj) - ej) * &pw;
            pw *= q(x);
        }
        err2 += &d * &d;
        y2 += q(y) * q(y);
    }
    (err2.to_f64().unwrap() / y2.to_f64().unwrap()).sqrt()
}

/// Abscissae far from the origin (timestamps).  Before: the powers of `x`
/// themselves were factored, so `linear_fit` of 13 samples at
/// `x = 10⁹ + 60k` gave slope `0.04166666663038663` (relative error
/// 9·10⁻¹⁰) and intercept `−41666670.1303`; a cubic at `x = 10⁴ + k/100`
/// was refused as "rank deficient: fewer than 4 distinct abscissae" (13
/// are distinct).  Oracle: exact rational least squares (SymPy
/// `(AᵀA).solve(Aᵀy)` on the binary values of the data: slope
/// `0.041666666666666664`, intercept `−41666670.166589744`), also
/// `poly_fit_exact`.
#[test]
fn polynomial_fits_far_from_the_origin_are_accurate() {
    let ys: Vec<f64> = (0..13)
        .map(|k| {
            let t = f64::from(k);
            2.0 - 0.5 * t + 0.25 * t * t + if k % 2 == 0 { 1e-3 } else { -1e-3 }
        })
        .collect();
    let xs: Vec<f64> = (0..13).map(|k| 1e9 + 60.0 * f64::from(k)).collect();
    let fit = linear_fit(&xs, &ys).unwrap();
    assert!(
        (fit.slope - 0.041_666_666_666_666_664).abs() < 1e-16,
        "{}",
        fit.slope
    );
    assert!(
        (fit.intercept + 41_666_670.166_589_744).abs() < 1e-6,
        "{}",
        fit.intercept
    );

    let xs: Vec<f64> = (0..13).map(|k| 1e4 + 0.01 * f64::from(k)).collect();
    let pts: Vec<_> = xs.iter().zip(&ys).map(|(&x, &y)| (q(x), q(y))).collect();
    let exact = poly_fit_exact(&pts, 3).unwrap();
    let c = poly_fit(&xs, &ys, 3).unwrap();
    // The exact coefficients rounded to f64 reach 1.3·10⁻⁶ here.
    assert!(fitted_value_error(&xs, &ys, &c, &exact) < 2e-6);
    let xs: Vec<f64> = (0..13).map(|k| 1000.0 + 0.1 * f64::from(k)).collect();
    let pts: Vec<_> = xs.iter().zip(&ys).map(|(&x, &y)| (q(x), q(y))).collect();
    let exact = poly_fit_exact(&pts, 2).unwrap();
    let c = poly_fit(&xs, &ys, 2).unwrap();
    // Before: 1.2·10⁻⁹ (the exact coefficients rounded: 1.1·10⁻¹⁰).
    assert!(fitted_value_error(&xs, &ys, &c, &exact) < 3e-10);
}

/// Unsorted abscissae.  Before: `trapezoid(&[0, 2, 1], &[0, 2, 1])` summed
/// the panels of a path that doubles back and returned `0.5` (as
/// `numpy.trapezoid` does), not the integral `2` of `y = x` over `[0, 2]`.
/// Sorted (either way) and repeated abscissae are integrated as before.
#[test]
fn trapezoid_refuses_unsorted_abscissae() {
    assert!(matches!(
        trapezoid(&[0.0, 2.0, 1.0], &[0.0, 2.0, 1.0]),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(trapezoid(&[1.0, 2.0], &[f64::NAN, 1.0]).is_err());
    assert_eq!(trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]).unwrap(), 2.0);
    assert_eq!(trapezoid(&[2.0, 1.0, 0.0], &[2.0, 1.0, 0.0]).unwrap(), -2.0);
    // A jump at x = 1: ∫₀² (0 on [0,1), 1 on [1,2]) = 1.
    assert_eq!(
        trapezoid(&[0.0, 0.0, 1.0, 1.0], &[0.0, 1.0, 1.0, 2.0]).unwrap(),
        1.0
    );
}

// ─────────────────────────────── normal forms ───────────────────────────────

fn lcg_matrix(n: usize, seed: u64) -> ZMatrix {
    let mut s = seed;
    let rows: Vec<Vec<BigInt>> = (0..n)
        .map(|_| {
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    BigInt::from(i64::try_from((s >> 33) % 19).unwrap() - 9)
                })
                .collect()
        })
        .collect();
    ZMatrix::new(rows).unwrap()
}

fn is_integral(m: &QMatrix) -> bool {
    m.iter().all(|v| v.is_integer())
}

/// A random 40×40 matrix with entries in `[−9, 9]`.  Before: the
/// eliminating HNF let intermediate entries grow exponentially — over 300 s
/// for `hermite_normal_form` and 32 s for `smith_normal_form` in a debug
/// build (SymPy 1.14: 2.2 s plain, 0.16 s with `D = |det|`).  Now every
/// entry stays below the determinant (Cohen's Algorithm 2.4.8).  Oracle:
/// `H` is in Hermite form, `H·A⁻¹` and `A·H⁻¹` are integral (same row
/// lattice, so `H` is *the* HNF) and `∏ hᵢᵢ = |det A|`; `S` equals the
/// `S` of the transform path, which also checks `U·H·V = S`.
#[test]
fn normal_forms_of_a_40_by_40_matrix_are_fast_and_exact() {
    let a = lcg_matrix(40, 7);
    let h = a.hermite_normal_form();
    let n = 40;
    let mut diag = BigInt::one();
    for i in 0..n {
        let p = h.get(i, i);
        assert!(p.is_positive(), "pivot {i}");
        diag *= p;
        for k in 0..i {
            assert!(h.get(i, k).is_zero(), "below diagonal ({i},{k})");
            assert!(
                !h.get(k, i).is_negative() && h.get(k, i) < p,
                "not reduced ({k},{i})"
            );
        }
    }
    assert_eq!(diag, a.det().unwrap().abs());
    let (aq, hq) = (a.to_qmatrix(), h.to_qmatrix());
    assert!(is_integral(&(&hq * &aq.inv().unwrap())));
    assert!(is_integral(&(&aq * &hq.inv().unwrap())));

    let s = a.smith_normal_form();
    let t = h.smith_normal_form_with_transforms();
    assert_eq!(s, t.s);
    assert_eq!(&(&t.u * &h) * &t.v, t.s);
}

/// The integer kernel of a 30×40 matrix.  Before: `integer_nullspace`
/// (the HNF of `[Aᵀ | I]`) ran for over 50 s; 20×30 took 0.22 s, now
/// 0.04 s.  Oracle: ten vectors with `A·k = 0` that generate a saturated
/// lattice (the Smith form of the basis is all ones).
#[test]
fn integer_kernel_of_a_wide_matrix_is_fast_and_saturated() {
    let square = lcg_matrix(40, 11);
    let a = ZMatrix::new(square.to_rows().into_iter().take(30).collect()).unwrap();
    let kernel = a.integer_nullspace();
    assert_eq!(kernel.len(), 10);
    for k in &kernel {
        assert!((&a * k).is_zero());
    }
    let rows: Vec<Vec<BigInt>> = kernel
        .iter()
        .map(|k| (0..40).map(|i| k.get(i, 0).clone()).collect())
        .collect();
    let s = ZMatrix::new(rows).unwrap().smith_normal_form();
    assert!((0..10).all(|i| s.get(i, i).is_one()), "{s:?}");
}

// ─────────────────────────────── units ───────────────────────────────

/// Calculus with respect to a converted variable.  Before: `diff_qty`
/// with respect to `Time::minutes(τ)` (the expression `60·τ`) returned
/// `0` — differentiating with respect to a non-symbol gave zero — and so
/// did the heat capacity `d(3ϑ J)/d(Temperature::from_celsius(ϑ))`;
/// `integrate_qty(5 N, Length::kilometers(ξ))` stayed
/// `Integral(5, 1000*ξ)`; the forwarding methods `Length::diff`,
/// `Qty::diff` and `integrate` did the same.  Oracle: the chain rule
/// `(dE/dτ)/(dt/dτ)` and the substitution `∫ f·(dv/dξ) dξ`.
#[test]
fn dimensional_calculus_with_converted_variables() {
    let ctx = Context::new();
    let tau = ctx.symbol("tau");
    let xi = ctx.symbol("xi");
    let theta = ctx.symbol("theta");
    let position = Length::meters(&(&tau * &tau)).as_qty();
    let v = diff_qty(&position, &Time::minutes(&tau).as_qty());
    assert!(
        (v.inner() - &tau / 30).expand().is_zero_structural(),
        "{}",
        v.inner()
    );
    let w = integrate_qty(
        &Force::newtons(&ctx.int(5)).as_qty(),
        &Length::kilometers(&xi).as_qty(),
    );
    assert!(
        (w.inner() - &xi * 5000).expand().is_zero_structural(),
        "{}",
        w.inner()
    );
    let minutes = Time::minutes(&tau);
    let by_method = Length::meters(&(&tau * &tau)).diff(&minutes);
    assert!(
        (&by_method - &tau / 30).expand().is_zero_structural(),
        "{by_method}"
    );
    let by_qty = Length::meters(&(&tau * &tau)).as_qty().integrate(&minutes);
    assert!(
        (&by_qty - tau.powi(3) * 20).expand().is_zero_structural(),
        "{by_qty}"
    );
    let heat = Energy::joules(&(&theta * 3)).as_qty();
    let capacity = diff_qty(&heat, &Temperature::from_celsius(&theta).as_qty());
    assert_eq!(capacity.inner(), &ctx.int(3));
    // A plain symbol is differentiated as before.
    let t = Time::symbol(&ctx, "t").as_qty();
    let x = Length::meters(&ctx.symbol("t").powi(2)).as_qty();
    assert_eq!(format!("{}", diff_qty(&x, &t).inner()), "2*t");
}

/// Dimensions of `arg`, products, series and Laplace transforms.  Before:
/// each was refused with "Function argument 0 must be dimensionless" for a
/// dimensioned argument.  Oracle: `arg z` is a phase (dimensionless, like
/// `sign`); `∏_{k=1}^{3} x` is `x³`; a series of `f(t)` has the dimension
/// of `f`; `∫₀^∞ f(t)e^{−st} dt` has `dim f · dim t` with `s` of dimension
/// `1/t`.
#[test]
fn dimension_inference_of_arg_product_series_and_laplace() {
    let ctx = Context::new();
    let (x, t, k, s) = (
        ctx.symbol("x"),
        ctx.symbol("t"),
        ctx.symbol("k"),
        ctx.symbol("s"),
    );
    let f = ctx.apply("f", &[&t]).unwrap();
    let dims = DimMap::new()
        .with("x", ConstDim::LENGTH)
        .with("t", ConstDim::TIME)
        .with("k", ConstDim::DIMENSIONLESS)
        .with("s", ConstDim::FREQUENCY)
        .with("f(t)", ConstDim::LENGTH);
    let d = |e: &Ex| infer_dimension(e, &dims);
    assert_eq!(d(&x.arg()), Ok(ConstDim::DIMENSIONLESS));
    let product = Ex::symbolic_product(&x, &k, &ctx.int(1), &ctx.int(3));
    assert_eq!(d(&product), Ok(ConstDim::VOLUME));
    assert!(d(&Ex::symbolic_product(&x, &k, &ctx.int(1), &k)).is_err());
    assert_eq!(d(&f.series(&t, &ctx.int(0), 3)), Ok(ConstDim::LENGTH));
    assert_eq!(
        d(&f.laplace(&t, &s)),
        Ok(ConstDim::LENGTH.mul(ConstDim::TIME))
    );
    let wrong = DimMap::new()
        .with("t", ConstDim::TIME)
        .with("s", ConstDim::TIME)
        .with("f(t)", ConstDim::LENGTH);
    assert!(infer_dimension(&f.laplace(&t, &s), &wrong).is_err());
}
