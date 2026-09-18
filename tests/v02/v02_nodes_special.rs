//! 0.2 special-function nodes: `Si`, `Ci`, `Ei`, `Li`, `Zeta`, `Polygamma`,
//! `KroneckerDelta` — exact values, differentiation, arbitrary-precision
//! evaluation, output formats, and `has_unevaluated` semantics.

use symplex::prelude::*;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Central finite difference of `f` at `x0` (step `h`).
fn fd(f: &Ex, x: &Ex, x0: f64, h: f64) -> f64 {
    let ctx = f.context();
    let at = |v: f64| {
        let r = ctx.rational((v * 1e9).round() as i64, 1_000_000_000);
        f.subs(x, &r).eval_f64().unwrap()
    };
    (at(x0 + h) - at(x0 - h)) / (2.0 * h)
}

fn check_derivative(f: &Ex, x: &Ex, points: &[f64]) {
    let d = f.diff(x);
    assert!(
        !d.has_unevaluated(),
        "derivative of {f} should be closed form, got {d}"
    );
    for &x0 in points {
        let numeric = fd(f, x, x0, 1e-4);
        let r = f
            .context()
            .rational((x0 * 1e9).round() as i64, 1_000_000_000);
        let symbolic = d.subs(x, &r).eval_f64().unwrap();
        assert!(
            approx(numeric, symbolic, 1e-5 * (1.0 + symbolic.abs())),
            "d/dx {f} at {x0}: finite diff {numeric} vs symbolic {symbolic} ({d})"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact values at construction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn si_exact_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(ctx.zero().si().is_zero_structural());
    assert_eq!(format!("{}", ctx.infinity().si()), "1/2*pi");
    assert_eq!(format!("{}", ctx.neg_infinity().si()), "-1/2*pi");
    assert_eq!((-&x).si(), -&x.si());
    assert_eq!(ctx.int(-3).si(), -&ctx.int(3).si());
    assert_eq!((&ctx.int(-2) * &x).si(), -&(&ctx.int(2) * &x).si());
    assert_eq!(format!("{}", x.si()), "Si(x)");
}

#[test]
fn ci_ei_li_exact_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(ctx.infinity().ci().is_zero_structural());
    assert_eq!(ctx.zero().ci(), ctx.neg_infinity());
    assert!(ctx.neg_infinity().ei().is_zero_structural());
    assert_eq!(ctx.infinity().ei(), ctx.infinity());
    assert_eq!(ctx.zero().ei(), ctx.neg_infinity());
    assert!(ctx.zero().li().is_zero_structural());
    assert_eq!(ctx.one().li(), ctx.neg_infinity());
    assert_eq!(ctx.infinity().li(), ctx.infinity());
    assert_eq!(x.exp().li(), x.ei(), "li(e^x) = Ei(x)");
    assert_eq!(ctx.e().li(), ctx.one().ei());
    assert_eq!(format!("{}", x.ci()), "Ci(x)");
    assert_eq!(format!("{}", x.ei()), "Ei(x)");
    assert_eq!(format!("{}", x.li()), "li(x)");
    // Ci(-x) is *not* rewritten (it changes real/complex character).
    assert_eq!(format!("{}", (-&x).ci()), "Ci(-x)");
}

#[test]
fn zeta_exact_values() {
    let ctx = Context::new();
    assert_eq!(ctx.int(1).zeta(), ctx.complex_infinity());
    assert_eq!(format!("{}", ctx.int(0).zeta()), "-1/2");
    assert_eq!(format!("{}", ctx.int(-1).zeta()), "-1/12");
    assert_eq!(format!("{}", ctx.int(-3).zeta()), "1/120");
    assert_eq!(format!("{}", ctx.int(-5).zeta()), "-1/252");
    assert_eq!(format!("{}", ctx.int(-7).zeta()), "1/240");
    for k in [-2i64, -4, -6, -8, -100] {
        assert!(ctx.int(k).zeta().is_zero_structural(), "zeta({k}) = 0");
    }
    assert_eq!(format!("{}", ctx.int(2).zeta()), "1/6*pi^2");
    assert_eq!(format!("{}", ctx.int(4).zeta()), "1/90*pi^4");
    assert_eq!(format!("{}", ctx.int(6).zeta()), "1/945*pi^6");
    assert_eq!(format!("{}", ctx.int(8).zeta()), "1/9450*pi^8");
    assert_eq!(format!("{}", ctx.int(10).zeta()), "1/93555*pi^10");
    assert_eq!(format!("{}", ctx.int(12).zeta()), "691/638512875*pi^12");
    assert_eq!(ctx.infinity().zeta(), ctx.one());
    // odd arguments and non-integers stay symbolic
    assert_eq!(format!("{}", ctx.int(3).zeta()), "zeta(3)");
    assert_eq!(format!("{}", ctx.rational(1, 2).zeta()), "zeta(1/2)");
    let s = ctx.symbol("s");
    assert_eq!(format!("{}", s.zeta()), "zeta(s)");
    // Numerical consistency of the exact even values
    let v = ctx.int(12).zeta().eval_f64().unwrap();
    assert!(approx(v, 1.000_246_086_553_308, 1e-15));
}

#[test]
fn polygamma_exact_values() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let two = ctx.int(2);
    let half = ctx.rational(1, 2);
    let x = ctx.symbol("x");
    // n = 0 → digamma
    assert_eq!(x.polygamma(&ctx.zero()), x.digamma());
    // trigamma(1) = ζ(2), ψ''(1) = −2ζ(3)
    assert_eq!(one.polygamma(&one), two.zeta());
    assert_eq!(format!("{}", one.polygamma(&two)), "-2*zeta(3)");
    assert_eq!(format!("{}", one.polygamma(&ctx.int(3))), "1/15*pi^4");
    // ψ'(1/2) = π²/2, ψ''(1/2) = −14 ζ(3)
    assert_eq!(format!("{}", half.polygamma(&one)), "1/2*pi^2");
    assert_eq!(format!("{}", half.polygamma(&two)), "-14*zeta(3)");
    // recurrence: ψ'(2) = π²/6 − 1, ψ'(3) = π²/6 − 5/4, ψ'(3/2) = π²/2 − 4
    assert_eq!(format!("{}", two.polygamma(&one)), "1/6*pi^2 - 1");
    assert_eq!(format!("{}", ctx.int(3).polygamma(&one)), "1/6*pi^2 - 5/4");
    assert_eq!(
        format!("{}", ctx.rational(3, 2).polygamma(&one)),
        "1/2*pi^2 - 4"
    );
    // poles
    assert_eq!(ctx.zero().polygamma(&one), ctx.complex_infinity());
    assert_eq!(ctx.int(-3).polygamma(&two), ctx.complex_infinity());
    // symbolic
    assert_eq!(format!("{}", x.polygamma(&one)), "polygamma(1, x)");
    let n = ctx.symbol("n");
    assert_eq!(format!("{}", x.polygamma(&n)), "polygamma(n, x)");
    // digamma folding via eval
    assert_eq!(
        format!("{}", ctx.int(3).digamma().eval()),
        "-EulerGamma + 3/2"
    );
    assert_eq!(
        format!("{}", half.digamma().eval()),
        "-EulerGamma - 2*ln(2)"
    );
    let psi_3_2 = ctx.rational(3, 2).digamma().eval();
    assert_eq!(
        psi_3_2,
        &(&ctx.int(2) - &ctx.euler_gamma()) - &(&ctx.int(2) * &ctx.int(2).ln())
    );
    assert!(approx(
        psi_3_2.eval_f64().unwrap(),
        0.036_489_973_978_576_52,
        1e-14
    ));
    assert_eq!(ctx.int(-1).digamma().eval(), ctx.complex_infinity());
}

#[test]
fn kronecker_delta_rules() {
    let ctx = Context::new();
    let i = ctx.symbol("i");
    let j = ctx.symbol("j");
    assert!(i.kronecker_delta(&i).is_one_structural());
    assert!(ctx.int(2).kronecker_delta(&ctx.int(2)).is_one_structural());
    assert!(ctx.int(2).kronecker_delta(&ctx.int(3)).is_zero_structural());
    assert!(
        ctx.rational(1, 2)
            .kronecker_delta(&ctx.int(1))
            .is_zero_structural()
    );
    assert!((&i + 1).kronecker_delta(&i).is_zero_structural());
    assert!((&i + &j).kronecker_delta(&(&j + &i)).is_one_structural());
    let d = i.kronecker_delta(&j);
    assert_eq!(d, j.kronecker_delta(&i), "symmetric");
    assert_eq!(format!("{d}"), "KroneckerDelta(i, j)");
    assert_eq!(d.to_latex(), r"\delta_{i j}");
    assert_eq!(d.is_integer(), Some(true));
    assert_eq!(d.is_negative(), Some(false));
    // derivative is zero
    assert!(d.diff(&i).is_zero_structural());
    // substitution folds
    assert!(d.subs(&j, &i).is_one_structural());
    assert!(d.subs_i64(&i, 1).subs_i64(&j, 2).is_zero_structural());
    // numerics
    assert_eq!(d.subs_i64(&i, 4).subs_i64(&j, 4).eval_f64().unwrap(), 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Output formats and parsing
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_latex_parse_tree() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let n = ctx.symbol("n");
    let nodes = [
        x.si(),
        x.ci(),
        x.ei(),
        x.li(),
        x.zeta(),
        x.polygamma(&n),
        x.kronecker_delta(&n),
    ];
    let displays = [
        "Si(x)",
        "Ci(x)",
        "Ei(x)",
        "li(x)",
        "zeta(x)",
        "polygamma(n, x)",
        "KroneckerDelta(n, x)",
    ];
    let latex = [
        r"\operatorname{Si}\left(x\right)",
        r"\operatorname{Ci}\left(x\right)",
        r"\operatorname{Ei}\left(x\right)",
        r"\operatorname{li}\left(x\right)",
        r"\zeta\left(x\right)",
        r"\psi^{(n)}\left(x\right)",
        r"\delta_{n x}",
    ];
    for ((e, d), l) in nodes.iter().zip(displays).zip(latex) {
        assert_eq!(format!("{e}"), d);
        assert_eq!(e.to_latex(), l);
        // Display → parse round trip
        assert_eq!(ctx.parse(d).unwrap(), *e, "parse({d})");
        // JSON round trip
        let json = serde_json::to_string(&e.to_tree()).unwrap();
        let tree: symplex::tree::ExprTree = serde_json::from_str(&json).unwrap();
        assert_eq!(ctx.from_tree(&tree), *e, "json({d})");
        // Not "unevaluated"
        assert!(
            !e.has_unevaluated(),
            "{d} is a function, not an unevaluated form"
        );
        assert_eq!(e.expr_type(), symplex::expr::ExprType::Function);
    }
    // Parsing is case-insensitive for function names and folds exact values.
    assert_eq!(format!("{}", ctx.parse("zeta(2)").unwrap()), "1/6*pi^2");
    assert_eq!(format!("{}", ctx.parse("si(0)").unwrap()), "0");
    assert_eq!(ctx.parse("kronecker_delta(3, 3)").unwrap(), ctx.one());
    // Unicode pretty printer
    assert_eq!(x.zeta().pretty().trim(), "ζ(x)");
    assert!(x.polygamma(&n).pretty().contains('ψ'));
}

// ═══════════════════════════════════════════════════════════════════════════
// Differentiation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn derivative_rules_symbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(format!("{}", x.si().diff(&x)), "sin(x)/x");
    assert_eq!(format!("{}", x.ci().diff(&x)), "cos(x)/x");
    assert_eq!(format!("{}", x.ei().diff(&x)), "exp(x)/x");
    assert_eq!(format!("{}", x.li().diff(&x)), "1/ln(x)");
    assert_eq!(format!("{}", x.digamma().diff(&x)), "polygamma(1, x)");
    assert_eq!(
        format!("{}", x.polygamma(&ctx.int(2)).diff(&x)),
        "polygamma(3, x)"
    );
    // chain rule
    let d = x.powi(2).si().diff(&x);
    assert_eq!(format!("{d}"), "2*sin(x^2)/x");
    // zeta has no elementary derivative
    let d = x.zeta().diff(&x);
    assert_eq!(format!("{d}"), "Derivative(zeta(x), x)");
    assert!(d.has_unevaluated());
    assert!(x.zeta().try_diff(&x).is_err());
    // constants
    assert!(ctx.int(3).zeta().diff(&x).is_zero_structural());
    let y = ctx.symbol("y");
    assert!(y.si().diff(&x).is_zero_structural());
}

#[test]
fn derivative_rules_numeric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_derivative(&x.si(), &x, &[0.5, 1.3, 4.0]);
    check_derivative(&x.ci(), &x, &[0.5, 1.3, 4.0]);
    check_derivative(&x.ei(), &x, &[0.5, 1.3, 2.5]);
    check_derivative(&x.li(), &x, &[2.0, 3.5, 0.5]);
    check_derivative(&x.digamma(), &x, &[0.7, 1.5, 3.0]);
    check_derivative(&x.polygamma(&ctx.int(1)), &x, &[0.7, 1.5, 3.0]);
    check_derivative(&(&x * &x.powi(2).si()), &x, &[0.5, 1.2]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Arbitrary-precision evaluation
// ═══════════════════════════════════════════════════════════════════════════

fn assert_prefix(e: &Ex, digits: u32, expected: &str) {
    let s = e.eval_decimal(digits).unwrap();
    assert!(
        s.starts_with(expected),
        "{e} → {s}, expected prefix {expected}"
    );
}

#[test]
fn si_ci_ei_li_reference_values() {
    let ctx = Context::new();
    let one = ctx.int(1);
    assert_prefix(&one.si(), 30, "0.94608307036718301494135331382");
    assert_prefix(&one.ci(), 30, "0.33740392290096813466264620388");
    assert_prefix(&one.ei(), 30, "1.89511781635593675546652093433");
    assert_prefix(&ctx.int(2).li(), 30, "1.04516378011749278484458888919");
    assert_prefix(&ctx.int(-1).ei(), 25, "-0.2193839343955202736771");
    assert_prefix(&ctx.int(10).ei(), 25, "2492.22897624187775913844");
    // Large arguments (asymptotic regime) and moderate ones (series with
    // cancellation guard) agree with known values.
    assert_prefix(&ctx.int(50).si(), 15, "1.55161707248594");
    assert_prefix(&ctx.int(50).ci(), 12, "-0.0056283863241");
    assert_prefix(&ctx.int(-100).ei(), 15, "-3.6835977616820");
    assert_prefix(&ctx.int(300).si(), 20, "1.5708810882137495193");
    // Si is odd; Ci of a negative argument is complex.
    assert_prefix(&ctx.int(-1).si(), 20, "-0.946083070367183014");
    let (re, im) = ctx.int(-1).ci().eval_complex64().unwrap();
    assert!(approx(re, 0.337_403_922_900_968_1, 1e-14));
    assert!(approx(im, std::f64::consts::PI, 1e-14));
    // li(1/2) via Ei(ln 1/2)
    assert_prefix(&ctx.rational(1, 2).li(), 20, "-0.37867104306108797");
    // 100 digits of Si(1)
    assert_prefix(
        &one.si(),
        100,
        "0.946083070367183014941353313823179657812337954738111790471",
    );
}

#[test]
fn zeta_reference_values() {
    let ctx = Context::new();
    assert_prefix(&ctx.int(3).zeta(), 30, "1.20205690315959428539973816151");
    assert_prefix(
        &ctx.int(3).zeta(),
        100,
        "1.2020569031595942853997381615114499907649862923404988817922715553418382057863130901864558736093352",
    );
    assert_prefix(&ctx.int(5).zeta(), 25, "1.036927755143369926331365");
    assert_prefix(
        &ctx.rational(1, 2).zeta(),
        25,
        "-1.460354508809586812889499",
    );
    assert_prefix(&ctx.rational(5, 2).zeta(), 20, "1.3414872572509171798");
    assert_prefix(&ctx.rational(-1, 2).zeta(), 20, "-0.2078862249773545660");
    assert_prefix(&ctx.rational(-7, 2).zeta(), 15, "0.00444101133547943");
    // Exact rational values agree with the numerical path
    let v = ctx.int(-3).zeta().eval_f64().unwrap();
    assert!(approx(v, 1.0 / 120.0, 1e-16));
    // Trivial zero through the numerical path (symbol substituted late)
    let s = ctx.symbol("s");
    let z = s.zeta().subs_i64(&s, -4);
    assert!(z.is_zero_structural());
    // ζ(1) via evalf is a pole
    assert!(s.zeta().subs_i64(&s, 1).eval_f64().is_err());
}

#[test]
fn polygamma_reference_values() {
    let ctx = Context::new();
    let one = ctx.int(1);
    // ψ'(1) = π²/6
    assert_prefix(&one.polygamma(&one), 25, "1.644934066848226436472415");
    // Force the numerical path with a non-special argument.
    let x = ctx.symbol("x");
    let tri = x.polygamma(&one);
    assert_prefix(&tri.subs_i64(&x, 1), 25, "1.644934066848226436472415");
    assert_prefix(
        &tri.subs(&x, &ctx.rational(7, 3)),
        25,
        "0.533097125427094081792004",
    );
    let tetra = x.polygamma(&ctx.int(2));
    assert_prefix(
        &tetra.subs(&x, &ctx.rational(7, 3)),
        20,
        "-0.27837239940160755246",
    );
    // ψ'(−1/2) = π²/2 + 4
    assert_prefix(
        &x.polygamma(&one).subs(&x, &ctx.rational(-1, 2)),
        20,
        "8.9348022005446793094",
    );
    // ψ⁽⁵⁾(3/4)
    assert_prefix(
        &x.polygamma(&ctx.int(5)).subs(&x, &ctx.rational(3, 4)),
        17,
        "678.75334287457971",
    );
    // digamma numeric and exact agree
    assert_prefix(&ctx.rational(7, 3).digamma(), 20, "0.617966219979193677");
    assert_prefix(&one.digamma(), 20, "-0.577215664901532860");
    // poles error out numerically
    assert!(x.polygamma(&one).subs_i64(&x, 0).eval_f64().is_err());
}

#[test]
fn try_variants_treat_special_functions_as_closed_form() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Si(x) is a closed form: differentiating it is fine…
    assert!(x.si().try_diff(&x).is_ok());
    // …and the special functions never trip has_unevaluated.
    let e = &x.si() + &(&x.zeta() * &x.polygamma(&ctx.int(1)));
    assert!(!e.has_unevaluated());
}
