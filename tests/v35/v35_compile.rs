//! `compile()` of the exponential, logarithmic, sine and cosine integrals,
//! erfi and the Fresnel integrals (0.48).
//!
//! Up to 0.47 `x.si().compile(&["x"])` (and `ci`, `ei`, `li`, `shi`, `chi`,
//! `erfi`, `fresnels`, `fresnelc`) failed with `NotImplemented: cannot
//! compile `Si` to a numerical function` (`… `Apply(Shi, …)` …` for the
//! library functions).  They now run through `f64` routines in the shared
//! numeric runtime built from DLMF series, continued fractions and
//! asymptotic expansions.  Oracle: mpmath 1.3 at 30–40 digits at the exact
//! double of each literal (`mpmath.si(mpf(x))`, …); a dense-grid sweep of
//! ~395,000 points (tiny, huge, negative, near the zeros) found at most
//! 9.2 ulps of relative error (li near 10¹⁹), see the 0.48 CHANGELOG.
//!
//! SymPy 1.14's `lambdify(x, f(x))` (default modules) is the comparison:
//! `li`, `Shi` and `Chi` raise `NameError` when called, `Ei` loses two
//! digits next to its zero (relative error 9e-3 at x = 0.37250741078136670)
//! and returns `nan` for `Ei(1e5)`, `erfi(30)` is `nan`, and with
//! `modules="math"`/`"numpy"` none of the nine functions works.

// The reference values are mpmath's, quoted to 20 digits.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

type Builder = fn(&Ex) -> Ex;

const FUNCS: [(&str, Builder); 9] = [
    ("Si", |e| e.si()),
    ("Ci", |e| e.ci()),
    ("Ei", |e| e.ei()),
    ("li", |e| e.li()),
    ("Shi", |e| e.shi()),
    ("Chi", |e| e.chi()),
    ("erfi", |e| e.erfi()),
    ("fresnels", |e| e.fresnels()),
    ("fresnelc", |e| e.fresnelc()),
];

fn builder(name: &str) -> Builder {
    FUNCS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|&(_, b)| b)
        .unwrap_or_else(|| panic!("unknown function {name}"))
}

fn compiled(ctx: &Context, name: &str) -> CompiledFn {
    let x = ctx.symbol("x");
    builder(name)(&x)
        .compile(&["x"])
        .unwrap_or_else(|e| panic!("{name}(x) does not compile: {e}"))
}

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let err = if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    };
    assert!(
        err <= tol,
        "{what}: got {got:.17e}, want {want:.17e} (rel err {err:.2e})"
    );
}

/// Each of the nine functions compiles and agrees with mpmath to 4e-15
/// relative (≈ 18 ulps; the worst case seen on dense grids is 9.2 ulps) at
/// tiny, moderate, large and negative arguments, on both sides of every
/// switch between power series, continued fraction and asymptotic
/// expansion.  Before (0.47): `NotImplemented: cannot compile …` for all
/// nine.  Oracle: mpmath 1.3, e.g. `mpmath.si(10)` = 1.6583475942188740493,
/// `mpmath.ei(-50)` = -3.7832640295504590187e-24, `mpmath.li(1e300)` =
/// 1.4497500526693363651e297, `mpmath.erfi(26.7)` = 8.4998672612689850586e307,
/// `mpmath.fresnels(1000)` = 0.49968169011381630608.
#[test]
fn integrals_compile_and_match_mpmath() {
    let cases: &[(&str, f64, f64)] = &[
        ("Si", 1e-300, 1e-300),
        ("Si", 0.5, 0.49310741804306668916),
        ("Si", 10.0, 1.6583475942188740493),
        ("Si", 1e10, 1.570796326707584657),
        ("Si", -3.0, -1.8486525279994682564),
        ("Ci", 1e-300, -690.19831223331217232),
        ("Ci", 0.5, -0.17778407880661290134),
        ("Ci", 2.5, 0.28587119636538349539),
        ("Ci", 64.5, 0.015446400671808082775),
        ("Ci", 1e10, -4.8750602517482265379e-11),
        ("Ei", -700.0, -1.4065187662340329228e-307),
        ("Ei", -50.0, -3.7832640295504590187e-24),
        ("Ei", -0.5, -0.55977359477616081175),
        ("Ei", 1.0, 1.8951178163559367555),
        ("Ei", 44.9, 7.2043442487032991969e17),
        ("Ei", 716.0, 1.2605029106040893555e308),
        ("li", 1e-300, -1.4455586289196509272e-303),
        ("li", 0.5, -0.37867104306108797673),
        ("li", 1.0000001, -15.540879935472920138),
        ("li", 1e6, 78627.54915946218192),
        ("li", 1e300, 1.4497500526693363651e297),
        ("Shi", 0.5, 0.50699674981966719583),
        ("Shi", 45.1, 4.3797274869974543461e17),
        ("Shi", -3.0, -4.9734404758598067977),
        ("Chi", 1.0, 0.83786694098020824089),
        ("Chi", 10.0, 1246.1144860424544147),
        ("Chi", 700.0, 7.2548936802628042631e300),
        ("erfi", 1e-300, 1.1283791670955126022e-300),
        ("erfi", 3.0, 1629.9946226015656511),
        ("erfi", 6.6, 7.1595491037404315298e17),
        ("erfi", 26.7, 8.4998672612689850586e307),
        ("erfi", -2.0, -18.564802414575552599),
        ("fresnels", 1e-100, 5.2359877559829890448e-301),
        ("fresnels", 1.5, 0.69750496008209301308),
        ("fresnels", 6.5, 0.54537645524323358023),
        ("fresnels", 1000.0, 0.49968169011381630608),
        ("fresnels", -2.0, -0.3434156783636982422),
        ("fresnelc", 0.5, 0.49234422587144639288),
        ("fresnelc", 1.7, 0.32382687600390026138),
        ("fresnelc", 6.3, 0.47600445535306793248),
        ("fresnelc", 1e8, 0.5),
    ];
    let ctx = Context::new();
    for &(name, x, want) in cases {
        let f = compiled(&ctx, name);
        assert_rel(f(&[x]), want, 4e-15, &format!("{name}({x:e})"));
    }
}

/// Next to their zeros Ei, li and Chi (one real zero each) and Ci (first
/// zero) keep their relative accuracy: Taylor series about the zero
/// replace the cancelling defining series.  At the later zeros of Ci the
/// accuracy is absolute (≤ 1e-16 here).  Oracle: mpmath 1.3 at the double
/// nearest each zero, `mpmath.ei(0.3725074107813666)` =
/// -5.1196989365556847021e-17, `mpmath.li(1.451369234883381)` =
/// 1.2958497702998692421e-16, `mpmath.chi(0.5238225713898644)` =
/// 5.8063560101731740179e-17, `mpmath.ci(0.6165054856207163)` =
/// 5.5715489456128631287e-17, `mpmath.ci(0.62)` = 0.0046059848815549665688,
/// `mpmath.ci(3.384180422551186)` = 1.8395124042672971097e-16.
/// (SymPy 1.14's `lambdify(x, Ei(x))(0.3725074107813667)` is off by 9e-3
/// relative.)
#[test]
fn integrals_keep_relative_accuracy_at_their_zeros() {
    let ctx = Context::new();
    let cases: &[(&str, f64, f64)] = &[
        ("Ei", 0.3725074107813666, -5.1196989365556847021e-17),
        ("Ei", 0.38, 0.029011221419283038485),
        ("li", 1.451369234883381, 1.2958497702998692421e-16),
        ("li", 1.5, 0.12506498631529635599),
        ("Chi", 0.5238225713898644, 5.8063560101731740179e-17),
        ("Chi", 0.53, 0.01338946755889949865),
        ("Ci", 0.6165054856207163, 5.5715489456128631287e-17),
        ("Ci", 0.62, 0.0046059848815549665688),
    ];
    for &(name, x, want) in cases {
        let f = compiled(&ctx, name);
        assert_rel(f(&[x]), want, 4e-15, &format!("{name}({x:e})"));
    }
    let ci = compiled(&ctx, "Ci");
    assert!((ci(&[3.384180422551186]) - 1.8395124042672971097e-16).abs() < 1e-16);
}

/// `compile`'s rule: the value `eval_f64` gives, or NaN where it is not
/// real.  `Ci`, `Chi` and `li` are not real for x < 0 (`eval_f64` refuses
/// with "nonzero imaginary part"; mpmath: `ci(-1)` = 0.3374 + 3.1416j,
/// `chi(-1)` = 0.8379 + 3.1416j, `li(-1)` = 0.0737 + 3.4227j), so the
/// compiled functions return NaN there; `Si`, `Ei`, `Shi`, `erfi` and the
/// Fresnel integrals are real on the whole line.  At the singular points
/// the compiled value is the real-axis limit from the real side: `Ei(0)`
/// and `li(1)` are `-oo` symbolically, `Ci(0)` and `Chi(0)` are `-∞` as in
/// mpmath (`mpmath.ci(0)` = `-inf`; SymPy's and symplex's symbolic value is
/// `zoo`, which a real function cannot return), and the limits at ±∞ are
/// those `eval` gives (`Si(oo)` = π/2, `Ei(-oo)` = 0, `fresnels(oo)` = 1/2).
#[test]
fn integrals_follow_the_nan_rule_and_limits() {
    let ctx = Context::new();
    for name in ["Ci", "Chi", "li"] {
        let f = compiled(&ctx, name);
        for x in [-1.0, -1e-300, -1e300, f64::NEG_INFINITY] {
            assert!(f(&[x]).is_nan(), "{name}({x:e}) should be NaN");
        }
        let v = ctx.from_f64(-1.0).unwrap();
        assert!(
            builder(name)(&v).eval_f64().is_err(),
            "{name}(-1) is not real"
        );
    }
    for name in ["Si", "Ei", "Shi", "erfi", "fresnels", "fresnelc"] {
        let f = compiled(&ctx, name);
        assert!(f(&[-1.0]).is_finite(), "{name}(-1) is real");
    }
    let at = |name: &str, x: f64| compiled(&ctx, name)(&[x]);
    // `Chi(-60)` = 9.68e23 + πi (mpmath `chi(-60)`): the value is not real,
    // and the compiled `Chi` is NaN on the whole negative axis rather than
    // switching to real values below x ≈ −38.  Since 0.53 `eval_f64` refuses
    // it too (its imaginary part is certified nonzero, D3); up to 0.52 it
    // returned the real part, the imaginary part lying below the 16th digit.
    let v = ctx.from_f64(-60.0).unwrap();
    assert!(v.chi().eval_f64().is_err());
    assert!(at("Chi", -60.0).is_nan());
    assert_eq!(at("Ci", 0.0), f64::NEG_INFINITY);
    assert_eq!(at("Chi", 0.0), f64::NEG_INFINITY);
    assert_eq!(at("Ei", 0.0), f64::NEG_INFINITY);
    assert_eq!(at("li", 1.0), f64::NEG_INFINITY);
    assert_eq!(at("li", 0.0), 0.0);
    assert_eq!(at("Si", f64::INFINITY), std::f64::consts::FRAC_PI_2);
    assert_eq!(at("Ci", f64::INFINITY), 0.0);
    assert_eq!(at("Ei", f64::NEG_INFINITY), 0.0);
    assert_eq!(at("Ei", f64::INFINITY), f64::INFINITY);
    assert_eq!(at("Shi", f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert_eq!(at("erfi", 30.0), f64::INFINITY);
    assert_eq!(at("fresnels", f64::NEG_INFINITY), -0.5);
    assert_eq!(at("fresnelc", f64::INFINITY), 0.5);
    for name in [
        "Si", "Ci", "Ei", "li", "Shi", "Chi", "erfi", "fresnels", "fresnelc",
    ] {
        assert!(at(name, f64::NAN).is_nan(), "{name}(NaN)");
    }
}

/// The compiled values agree with `eval_f64` (the certified evaluator,
/// correctly rounded here) to 4e-15 relative wherever `eval_f64` succeeds,
/// over points spread across the regions of every algorithm — except
/// where the value is not real: `Ci`, `Chi` and `li` of a negative argument
/// are NaN (see `integrals_follow_the_nan_rule_and_limits` for `Chi(-60)`,
/// whose imaginary part `eval_f64` drops as negligible).
#[test]
fn integrals_agree_with_eval_f64() {
    let ctx = Context::new();
    let points = [
        -60.0, -7.3, -1.5, -0.4, 1e-12, 0.05, 0.3, 0.9, 1.25, 1.9, 2.1, 3.7, 6.4, 6.6, 12.5, 33.0,
        46.0, 63.0, 65.0, 300.0, 1e5, 1e12,
    ];
    for (name, build) in FUNCS {
        let f = compiled(&ctx, name);
        for x in points {
            if x < 0.0 && matches!(name, "Ci" | "Chi" | "li") {
                assert!(f(&[x]).is_nan(), "{name}({x:e})");
                continue;
            }
            let v = ctx.from_f64(x).unwrap();
            let Ok(want) = build(&v).eval_f64() else {
                assert!(f(&[x]).is_nan() || f(&[x]).is_infinite(), "{name}({x:e})");
                continue;
            };
            if want.is_infinite() {
                assert_eq!(f(&[x]), want, "{name}({x:e})");
            } else {
                assert_rel(f(&[x]), want, 4e-15, &format!("{name}({x:e}) vs eval_f64"));
            }
        }
    }
}

/// Antiderivatives that `integrate` writes with these functions now
/// compile: ∫ sin(x)/x dx = Si(x), ∫ eˣ/x dx = Ei(x), ∫ 1/ln x dx = li(x),
/// ∫ sin(πx²/2) dx = S(x).  Their compiled values match the definitions
/// (mpmath: `quad(lambda t: sin(t)/t, [0, 2])` = 1.6054129768026948486,
/// `ei(2)` = 4.9542343560018901634, `li(3)` = 2.1635885946671919729,
/// `fresnels(1.2)` = 0.62340091854624963806).
#[test]
fn integrate_results_compile() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        (x.sin() / &x, 2.0, 1.6054129768026948486),
        (x.exp() / &x, 2.0, 4.9542343560018901634),
        (x.ln().powi(-1), 3.0, 2.1635885946671919729),
    ];
    for (integrand, at, want) in cases {
        let anti = integrand.integrate(&x);
        let f = anti
            .compile(&["x"])
            .unwrap_or_else(|e| panic!("∫ {integrand} = {anti} does not compile: {e}"));
        assert_rel(f(&[at]), want, 4e-15, &format!("∫ {integrand} at {at}"));
    }
    let s = (&(&ctx.pi() / 2) * &x.powi(2)).sin().integrate(&x);
    let f = s.compile(&["x"]).unwrap();
    assert_rel(f(&[1.2]), 0.62340091854624963806, 4e-15, "∫ sin(πx²/2)");
}
