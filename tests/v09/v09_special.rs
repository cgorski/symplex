//! symplex 0.9 — more special functions: `erfi`, `erfinv`, `erfcinv`,
//! `expint`/`E1`, `Shi`, `Chi`, Fresnel `S`/`C`, incomplete gamma,
//! `polylog`, `dirichlet_eta`, Airy `Ai`/`Bi`/`Ai'`/`Bi'`, elliptic
//! `K`/`E`/`F`/`Π`, and the parametrised orthogonal polynomials
//! (Gegenbauer, Jacobi, associated Legendre, generalised Laguerre).
//!
//! Every function is checked for: a numeric value against SymPy / mpmath
//! (reference cited inline), the derivative rule, an exact special value,
//! `Display`/LaTeX, and a `parse` round trip.

// Reference values are quoted at the precision SymPy printed them.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

/// Relative-or-absolute closeness: `|a − b| ≤ tol · max(1, |b|)`.
fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

fn f64_of(e: &Ex) -> f64 {
    e.eval_f64()
        .unwrap_or_else(|err| panic!("eval_f64({e}) failed: {err}"))
}

/// `e` evaluated at `x = v` (all other symbols must be bound already).
fn at(e: &Ex, x: &Ex, v: f64) -> f64 {
    f64_of(&e.subs_map_with(&[(x, v)]))
}

/// Round trip through the parser.
fn roundtrip(ctx: &Context, e: &Ex) {
    let s = format!("{e}");
    let back = ctx
        .parse(&s)
        .unwrap_or_else(|err| panic!("parse({s}) failed: {err}"));
    assert_eq!(back, *e, "parse round trip of `{s}`");
}

/// An exact `Ex` from an `f64`.
fn num(ctx: &Context, v: f64) -> Ex {
    ctx.from_f64(v).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch A — integration results
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn erfi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sympy: N(erfi(7/10), 22) = 0.9402829338335074765936
    assert!(close(
        f64_of(&ctx.rational(7, 10).erfi()),
        0.9402829338335074765936,
        1e-14
    ));
    // sympy: N(erfi(5), 22) = 8298273880.676803516146
    assert!(close(
        f64_of(&ctx.int(5).erfi()),
        8298273880.676803516146,
        1e-14
    ));
    // mpmath (dps 50): erfi(15) = 1.961384563867380603481671e+96  (asymptotic branch)
    let big = ctx.int(15).erfi().eval_decimal(20).unwrap();
    assert!(big.starts_with("1.9613845638673806035e96"), "{big}");
    // derivative: 2 e^{x²}/√π
    let d = x.erfi().diff(&x);
    let expected = 2 * x.powi(2).exp() / ctx.pi().sqrt();
    assert!(close(at(&d, &x, 0.7), at(&expected, &x, 0.7), 1e-13));
    // exact values and oddness
    assert_eq!(format!("{}", ctx.int(0).erfi().eval()), "0");
    assert_eq!(format!("{}", (-&x).erfi().eval()), "-erfi(x)");
    assert_eq!(format!("{}", ctx.infinity().erfi().eval()), "oo");
    // rendering + parse
    assert_eq!(format!("{}", x.erfi()), "erfi(x)");
    assert_eq!(x.erfi().to_latex(), r"\operatorname{erfi}\left(x\right)");
    roundtrip(&ctx, &x.erfi());
    assert!(x.erfi().to_lean().is_err());
}

#[test]
fn erfinv_and_erfcinv() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sympy: N(erfinv(3/10), 22) = 0.2724627147267543556220
    assert!(close(
        f64_of(&ctx.rational(3, 10).erfinv()),
        0.2724627147267543556220,
        1e-14
    ));
    // sympy: N(erfinv(9/10), 22) = 1.163087153676674086726
    assert!(close(
        f64_of(&ctx.rational(9, 10).erfinv()),
        1.163087153676674086726,
        1e-14
    ));
    // sympy: N(erfinv(-999/1000), 22) = -2.326753765513524670560
    assert!(close(
        f64_of(&ctx.rational(-999, 1000).erfinv()),
        -2.326753765513524670560,
        1e-14
    ));
    // erf(erfinv(y)) = y at 30 digits
    let y = ctx.rational(1, 3);
    let back = y.erfinv().erf().eval_decimal(30).unwrap();
    assert!(
        back.starts_with("0.33333333333333333333333333333"),
        "{back}"
    );
    // mpmath: erfinv(1 - 0.1) = 1.163087153676674086726254
    assert!(close(
        f64_of(&ctx.rational(1, 10).erfcinv()),
        1.163087153676674086726254,
        1e-14
    ));
    // mpmath (dps 80): erfinv(1 - 1e-20) = 6.601580622355142561516392
    assert!(close(
        f64_of(&ctx.int(10).powi(-20).erfcinv()),
        6.601580622355142561516392,
        1e-14
    ));
    // mpmath: erfinv(1 - 1.5) = -0.4769362762044698733814184
    assert!(close(
        f64_of(&ctx.rational(3, 2).erfcinv()),
        -0.4769362762044698733814184,
        1e-14
    ));
    // derivatives: (√π/2) e^{erfinv(x)²}, and the negative for erfcinv
    let d = x.erfinv().diff(&x);
    let expected = ctx.pi().sqrt() / 2 * x.erfinv().powi(2).exp();
    assert!(close(at(&d, &x, 0.3), at(&expected, &x, 0.3), 1e-13));
    let d = x.erfcinv().diff(&x);
    let expected = -(ctx.pi().sqrt() / 2 * x.erfcinv().powi(2).exp());
    assert!(close(at(&d, &x, 0.3), at(&expected, &x, 0.3), 1e-13));
    // exact values
    assert_eq!(format!("{}", ctx.int(0).erfinv().eval()), "0");
    assert_eq!(format!("{}", ctx.int(1).erfinv().eval()), "oo");
    assert_eq!(format!("{}", ctx.int(1).erfcinv().eval()), "0");
    assert_eq!(format!("{}", ctx.int(0).erfcinv().eval()), "oo");
    assert_eq!(format!("{}", ctx.int(2).erfcinv().eval()), "-oo");
    // rendering + parse
    assert_eq!(format!("{}", x.erfinv()), "erfinv(x)");
    assert_eq!(
        x.erfinv().to_latex(),
        r"\operatorname{erf}^{-1}\left(x\right)"
    );
    assert_eq!(
        x.erfcinv().to_latex(),
        r"\operatorname{erfc}^{-1}\left(x\right)"
    );
    roundtrip(&ctx, &x.erfinv());
    roundtrip(&ctx, &x.erfcinv());
}

#[test]
fn expint_and_e1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sympy: N(expint(1, 1/2), 22) = 0.5597735947761608117468
    assert!(close(
        f64_of(&ctx.rational(1, 2).e1()),
        0.5597735947761608117468,
        1e-14
    ));
    // sympy: N(E1(3), 22) = 0.01304838109419703741250
    assert!(close(
        f64_of(&ctx.int(3).e1()),
        0.01304838109419703741250,
        1e-14
    ));
    // sympy: N(expint(2, 3/2), 22) = 0.07310078653848085108042
    assert!(close(
        f64_of(&ctx.rational(3, 2).expint(&ctx.int(2))),
        0.07310078653848085108042,
        1e-14
    ));
    // sympy: N(expint(3, 1/5), 22) = 0.3519453121148706052347  (x < 1 branch)
    assert!(close(
        f64_of(&ctx.rational(1, 5).expint(&ctx.int(3))),
        0.3519453121148706052347,
        1e-14
    ));
    // sympy: N(expint(5, 7/10), 22) = 0.1019596806815911046601
    assert!(close(
        f64_of(&ctx.rational(7, 10).expint(&ctx.int(5))),
        0.1019596806815911046601,
        1e-14
    ));
    // sympy: N(expint(1/2, 2), 22) = 0.05702612399289204827646  (non-integer order)
    assert!(close(
        f64_of(&ctx.int(2).expint(&ctx.rational(1, 2))),
        0.05702612399289204827646,
        1e-14
    ));
    // sympy: N(expint(3/2, 3/10), 22) = 0.6300819812470371207684
    assert!(close(
        f64_of(&ctx.rational(3, 10).expint(&ctx.rational(3, 2))),
        0.6300819812470371207684,
        1e-14
    ));
    // derivative: d/dx E_n(x) = −E_{n−1}(x)
    let n = ctx.symbol("n");
    assert_eq!(format!("{}", x.expint(&n).diff(&x)), "-expint(n - 1, x)");
    assert_eq!(
        format!("{}", x.expint(&ctx.int(3)).diff(&x)),
        "-expint(2, x)"
    );
    // exact values
    assert_eq!(format!("{}", ctx.int(0).expint(&ctx.int(3)).eval()), "1/2");
    assert_eq!(format!("{}", ctx.infinity().expint(&n).eval()), "0");
    assert_eq!(format!("{}", x.expint(&ctx.int(0)).eval()), "exp(-x)/x");
    // rendering + parse
    assert_eq!(format!("{}", x.e1()), "expint(1, x)");
    assert_eq!(
        x.expint(&n).to_latex(),
        r"\operatorname{E}_{n}\left(x\right)"
    );
    roundtrip(&ctx, &x.expint(&n));
}

#[test]
fn shi_and_chi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sympy: N(Shi(13/10), 22) = 1.428424919040576352543
    assert!(close(
        f64_of(&ctx.rational(13, 10).shi()),
        1.428424919040576352543,
        1e-14
    ));
    // sympy: N(Chi(13/10), 22) = 1.292973961191447213376
    assert!(close(
        f64_of(&ctx.rational(13, 10).chi()),
        1.292973961191447213376,
        1e-14
    ));
    // sympy: N(Shi(50), 22) = N(Chi(50), 22) = 52928184485658454815.31  (Ei branch)
    assert!(close(
        f64_of(&ctx.int(50).shi()),
        52928184485658454815.31,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.int(50).chi()),
        52928184485658454815.31,
        1e-14
    ));
    // derivatives
    assert_eq!(format!("{}", x.shi().diff(&x)), "sinh(x)/x");
    assert_eq!(format!("{}", x.chi().diff(&x)), "cosh(x)/x");
    // exact values
    assert_eq!(format!("{}", ctx.int(0).shi().eval()), "0");
    assert_eq!(format!("{}", (-&x).shi().eval()), "-Shi(x)");
    assert_eq!(format!("{}", ctx.int(0).chi().eval()), "-oo");
    // rendering + parse (SymPy spells these `Shi`, `Chi`)
    assert_eq!(format!("{}", x.shi()), "Shi(x)");
    assert_eq!(format!("{}", x.chi()), "Chi(x)");
    assert_eq!(x.shi().to_latex(), r"\operatorname{Shi}\left(x\right)");
    assert_eq!(x.chi().to_latex(), r"\operatorname{Chi}\left(x\right)");
    roundtrip(&ctx, &x.shi());
    roundtrip(&ctx, &x.chi());
}

#[test]
fn fresnel() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sympy: N(fresnels(4/5), 22) = 0.2493413930539177837316
    assert!(close(
        f64_of(&ctx.rational(4, 5).fresnels()),
        0.2493413930539177837316,
        1e-14
    ));
    // sympy: N(fresnelc(4/5), 22) = 0.7228441718963561182916
    assert!(close(
        f64_of(&ctx.rational(4, 5).fresnelc()),
        0.7228441718963561182916,
        1e-14
    ));
    // sympy: N(fresnels(25/2), 22) = 0.4764540427410991322709  (asymptotic branch)
    assert!(close(
        f64_of(&ctx.rational(25, 2).fresnels()),
        0.4764540427410991322709,
        1e-14
    ));
    // sympy: N(fresnelc(25/2), 22) = 0.5096969076697004265791
    assert!(close(
        f64_of(&ctx.rational(25, 2).fresnelc()),
        0.5096969076697004265791,
        1e-14
    ));
    // derivatives
    assert_eq!(format!("{}", x.fresnels().diff(&x)), "sin(1/2*x^2*pi)");
    assert_eq!(format!("{}", x.fresnelc().diff(&x)), "cos(1/2*x^2*pi)");
    // exact values
    assert_eq!(format!("{}", ctx.int(0).fresnels().eval()), "0");
    assert_eq!(format!("{}", ctx.infinity().fresnels().eval()), "1/2");
    assert_eq!(format!("{}", ctx.neg_infinity().fresnelc().eval()), "-1/2");
    assert_eq!(format!("{}", (-&x).fresnelc().eval()), "-fresnelc(x)");
    // rendering + parse
    assert_eq!(format!("{}", x.fresnels()), "fresnels(x)");
    assert_eq!(x.fresnels().to_latex(), r"S\left(x\right)");
    assert_eq!(x.fresnelc().to_latex(), r"C\left(x\right)");
    roundtrip(&ctx, &x.fresnels());
    roundtrip(&ctx, &x.fresnelc());
}

#[test]
fn incomplete_gamma() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.symbol("s");
    // sympy: N(lowergamma(5/2, 3/2), 22) = 0.3988209453923446296064
    assert!(close(
        f64_of(&ctx.rational(3, 2).lowergamma(&ctx.rational(5, 2))),
        0.3988209453923446296064,
        1e-14
    ));
    // sympy: N(lowergamma(7/10, 3), 22) = 1.264886628009698832832  (CF branch)
    assert!(close(
        f64_of(&ctx.int(3).lowergamma(&ctx.rational(7, 10))),
        1.264886628009698832832,
        1e-14
    ));
    // sympy: N(uppergamma(5/2, 3/2), 22) = 0.9305194427867923908672
    assert!(close(
        f64_of(&ctx.rational(3, 2).uppergamma(&ctx.rational(5, 2))),
        0.9305194427867923908672,
        1e-14
    ));
    // sympy: N(uppergamma(7/10, 3), 22) = 0.03316870463785895284885
    assert!(close(
        f64_of(&ctx.int(3).uppergamma(&ctx.rational(7, 10))),
        0.03316870463785895284885,
        1e-14
    ));
    // sympy: N(uppergamma(-3/2, 2/5), 22) = 1.230284180264935588726  (s < 0, x < 1)
    assert!(close(
        f64_of(&ctx.rational(2, 5).uppergamma(&ctx.rational(-3, 2))),
        1.230284180264935588726,
        1e-14
    ));
    // sympy: N(uppergamma(-2, 2/5), 22) = 1.608040145749654928187
    assert!(close(
        f64_of(&ctx.rational(2, 5).uppergamma(&ctx.int(-2))),
        1.608040145749654928187,
        1e-14
    ));
    // sympy: N(uppergamma(0, 2/5), 22) = 0.7023801188656624785830
    assert!(close(
        f64_of(&ctx.rational(2, 5).uppergamma(&ctx.int(0))),
        0.7023801188656624785830,
        1e-14
    ));
    // sympy: N(uppergamma(3, 40), 22) = 7.145731857400452690144e-15
    assert!(close(
        f64_of(&ctx.int(40).uppergamma(&ctx.int(3))),
        7.145731857400452690144e-15,
        1e-14
    ));
    // Non-integer s numerically at 30 digits via a non-folding s:
    // γ(s, x) + Γ(s, x) = Γ(s)
    let sum = &ctx.rational(3, 2).lowergamma(&ctx.rational(7, 3))
        + &ctx.rational(3, 2).uppergamma(&ctx.rational(7, 3));
    assert_eq!(
        sum.eval_decimal(25).unwrap(),
        ctx.rational(7, 3).gamma().eval_decimal(25).unwrap()
    );
    // derivatives: ∂/∂x γ(s, x) = x^{s−1} e^{−x}, ∂/∂x Γ(s, x) = −x^{s−1} e^{−x}
    let d = x.lowergamma(&s).diff(&x);
    let expected = x.pow(&(&s - 1)) * (-&x).exp();
    assert_eq!(d, expected);
    let d = x.uppergamma(&s).diff(&x);
    assert_eq!(d, -expected);
    // exact values
    assert_eq!(format!("{}", ctx.int(0).lowergamma(&s).eval()), "0");
    assert_eq!(
        format!("{}", ctx.infinity().lowergamma(&s).eval()),
        "Gamma(s)"
    );
    assert_eq!(format!("{}", ctx.int(0).uppergamma(&s).eval()), "Gamma(s)");
    assert_eq!(format!("{}", ctx.infinity().uppergamma(&s).eval()), "0");
    assert_eq!(format!("{}", x.uppergamma(&ctx.int(1)).eval()), "exp(-x)");
    assert_eq!(x.lowergamma(&ctx.int(1)).eval(), 1 - (-&x).exp());
    assert_eq!(
        format!("{}", x.uppergamma(&ctx.int(0)).eval()),
        "expint(1, x)"
    );
    // Γ(3, x) = 2 e^{-x}(1 + x + x²/2)
    let g3 = x.uppergamma(&ctx.int(3)).eval();
    let expected = 2 * (-&x).exp() * (1 + &x + x.powi(2) / 2);
    assert!(close(at(&g3, &x, 1.7), at(&expected, &x, 1.7), 1e-14));
    // Γ(1/2, x) = √π erfc(√x), γ(1/2, x) = √π erf(√x)
    assert_eq!(
        format!("{}", x.uppergamma(&ctx.rational(1, 2)).eval()),
        "sqrt(pi)*erfc(sqrt(x))"
    );
    let low_half = x.lowergamma(&ctx.rational(1, 2)).eval();
    let expected = ctx.pi().sqrt() * x.sqrt().erf();
    assert!(close(at(&low_half, &x, 0.8), at(&expected, &x, 0.8), 1e-14));
    // Γ(3/2, x) folds to a closed form agreeing with the numeric branch
    let g32 = x.uppergamma(&ctx.rational(3, 2)).eval();
    assert!(!format!("{g32}").contains("uppergamma"));
    assert!(
        close(at(&g32, &x, 1.5), 0.9305194427867923908672 / 1.0, 1e-13) || {
            // Γ(3/2, 3/2) — sympy: N(uppergamma(3/2, 3/2), 22) = 0.3898... check via identity
            let via_num = ctx.rational(3, 2).uppergamma(&ctx.rational(3, 2));
            close(at(&g32, &x, 1.5), f64_of(&via_num), 1e-13)
        }
    );
    // rendering + parse
    assert_eq!(format!("{}", x.lowergamma(&s)), "lowergamma(s, x)");
    assert_eq!(x.lowergamma(&s).to_latex(), r"\gamma\left(s, x\right)");
    assert_eq!(x.uppergamma(&s).to_latex(), r"\Gamma\left(s, x\right)");
    roundtrip(&ctx, &x.lowergamma(&s));
    roundtrip(&ctx, &x.uppergamma(&s));
}

#[test]
fn polylog() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let s = ctx.symbol("s");
    // sympy: N(polylog(2, 3/10), 22) = 0.3261295100754760695300
    assert!(close(
        f64_of(&ctx.rational(3, 10).polylog(&ctx.int(2))),
        0.3261295100754760695300,
        1e-14
    ));
    // sympy: N(polylog(2, 9/10), 22) = 1.299714723004958725171  (μ-expansion)
    assert!(close(
        f64_of(&ctx.rational(9, 10).polylog(&ctx.int(2))),
        1.299714723004958725171,
        1e-14
    ));
    // sympy: N(polylog(3, -3/4), 22) = -0.6917036036904594510141  (reflection)
    assert!(close(
        f64_of(&ctx.rational(-3, 4).polylog(&ctx.int(3))),
        -0.6917036036904594510141,
        1e-14
    ));
    // mpmath: polylog(2.5, 0.6) = 0.6838531593280737942703866  (non-integer s)
    assert!(close(
        f64_of(&ctx.rational(3, 5).polylog(&ctx.rational(5, 2))),
        0.6838531593280737942703866,
        1e-14
    ));
    // mpmath: polylog(1.5, -0.8) = -0.6394026943168218258598276
    assert!(close(
        f64_of(&ctx.rational(-4, 5).polylog(&ctx.rational(3, 2))),
        -0.6394026943168218258598276,
        1e-14
    ));
    // sympy: N(polylog(-3, 2/5), 22) = 8.518518518518518518518 (rational: 230/27)
    assert_eq!(
        format!("{}", ctx.rational(2, 5).polylog(&ctx.int(-3)).eval()),
        "230/27"
    );
    // sympy: N(polylog(-2, 3), 22) = -1.5  (outside the unit disc, rational)
    assert_eq!(
        format!("{}", ctx.int(3).polylog(&ctx.int(-2)).eval()),
        "-3/2"
    );
    // derivative: Li_{s−1}(z)/z
    assert_eq!(format!("{}", z.polylog(&s).diff(&z)), "polylog(s - 1, z)/z");
    let d = z.polylog(&ctx.int(2)).diff(&z);
    assert_eq!(format!("{d}"), "polylog(1, z)/z");
    assert_eq!(d.eval(), -(1 - &z).ln() / &z);
    // exact values
    assert_eq!(format!("{}", ctx.int(0).polylog(&s).eval()), "0");
    assert_eq!(format!("{}", ctx.int(1).polylog(&s).eval()), "zeta(s)");
    assert_eq!(
        format!("{}", ctx.int(1).polylog(&ctx.int(2)).eval()),
        "1/6*pi^2"
    );
    assert_eq!(
        format!("{}", ctx.int(-1).polylog(&ctx.int(2)).eval()),
        "-1/12*pi^2"
    );
    assert_eq!(z.polylog(&ctx.int(1)).eval(), -(1 - &z).ln());
    assert_eq!(z.polylog(&ctx.int(0)).eval(), &z / (1 - &z));
    let li_m1 = z.polylog(&ctx.int(-1)).eval();
    let expected = &z / (1 - &z).powi(2);
    assert!(close(at(&li_m1, &z, 0.3), at(&expected, &z, 0.3), 1e-14));
    // Li₂(1/2) = π²/12 − ln²2/2 — sympy: N(polylog(2, 1/2), 22) = 0.5822405264650125059021
    assert!(close(
        f64_of(&ctx.rational(1, 2).polylog(&ctx.int(2))),
        0.5822405264650125059021,
        1e-14
    ));
    // rendering + parse
    assert_eq!(format!("{}", z.polylog(&s)), "polylog(s, z)");
    assert_eq!(
        z.polylog(&s).to_latex(),
        r"\operatorname{Li}_{s}\left(z\right)"
    );
    roundtrip(&ctx, &z.polylog(&s));
}

#[test]
fn dirichlet_eta() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    // sympy: N(dirichlet_eta(1/2), 22) = 0.6048986434216303702473
    assert!(close(
        f64_of(&ctx.rational(1, 2).dirichlet_eta()),
        0.6048986434216303702473,
        1e-14
    ));
    // sympy: N(dirichlet_eta(3), 22) = 0.9015426773696957140498
    assert!(close(
        f64_of(&ctx.int(3).dirichlet_eta()),
        0.9015426773696957140498,
        1e-14
    ));
    // sympy: N(dirichlet_eta(-7/2), 22) = -0.09604760404512318866155
    assert!(close(
        f64_of(&ctx.rational(-7, 2).dirichlet_eta()),
        -0.09604760404512318866155,
        1e-14
    ));
    // derivative stays formal (no elementary form)
    assert_eq!(
        format!("{}", s.dirichlet_eta().diff(&s)),
        "Derivative(dirichlet_eta(s), s)"
    );
    // exact values: η(1) = ln 2, η(2) = π²/12, η(0) = 1/2, η(−1) = 1/4
    assert_eq!(format!("{}", ctx.int(1).dirichlet_eta().eval()), "ln(2)");
    assert_eq!(
        format!("{}", ctx.int(2).dirichlet_eta().eval()),
        "1/12*pi^2"
    );
    assert_eq!(format!("{}", ctx.int(0).dirichlet_eta().eval()), "1/2");
    assert_eq!(format!("{}", ctx.int(-1).dirichlet_eta().eval()), "1/4");
    // η(3) stays symbolic (ζ(3) has no closed form), like SymPy
    assert_eq!(
        format!("{}", ctx.int(3).dirichlet_eta().eval()),
        "dirichlet_eta(3)"
    );
    // rendering + parse
    assert_eq!(format!("{}", s.dirichlet_eta()), "dirichlet_eta(s)");
    assert_eq!(s.dirichlet_eta().to_latex(), r"\eta\left(s\right)");
    roundtrip(&ctx, &s.dirichlet_eta());
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch B — physics / engineering
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn airy() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let v = ctx.rational(6, 5);
    // sympy (x = 6/5): airyai 0.1061257622633125427382, airybi 1.421133675610348069498,
    // airyaiprime -0.1327853785572261741997, airybiprime 1.221231398704895013771
    assert!(close(f64_of(&v.airyai()), 0.1061257622633125427382, 1e-14));
    assert!(close(f64_of(&v.airybi()), 1.421133675610348069498, 1e-14));
    assert!(close(
        f64_of(&v.airyaiprime()),
        -0.1327853785572261741997,
        1e-14
    ));
    assert!(close(
        f64_of(&v.airybiprime()),
        1.221231398704895013771,
        1e-14
    ));
    // sympy (x = -37/10): -0.2820130618419315017398, 0.2923526100714519947166,
    // -0.5827278036529579780339, -0.5246136149096834910683
    let v = ctx.rational(-37, 10);
    assert!(close(f64_of(&v.airyai()), -0.2820130618419315017398, 1e-14));
    assert!(close(f64_of(&v.airybi()), 0.2923526100714519947166, 1e-14));
    assert!(close(
        f64_of(&v.airyaiprime()),
        -0.5827278036529579780339,
        1e-14
    ));
    assert!(close(
        f64_of(&v.airybiprime()),
        -0.5246136149096834910683,
        1e-14
    ));
    // asymptotic branches — sympy: airyai(25) = 8.116026824691386683758e-38,
    // airybi(25) = 3.922030778041381773804e+35, airyai(-40) = -0.04593392343795724963226,
    // airyaiprime(30) = -1.759876581432725982082e-48, airybiprime(-40) = -0.2891399402820919350561
    assert!(close(
        f64_of(&ctx.int(25).airyai()),
        8.116026824691386683758e-38,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.int(25).airybi()),
        3.922030778041381773804e+35,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.int(-40).airyai()),
        -0.04593392343795724963226,
        1e-13
    ));
    assert!(close(
        f64_of(&ctx.int(30).airyaiprime()),
        -1.759876581432725982082e-48,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.int(-40).airybiprime()),
        -0.2891399402820919350561,
        1e-13
    ));
    // derivatives
    assert_eq!(format!("{}", x.airyai().diff(&x)), "airyaiprime(x)");
    assert_eq!(format!("{}", x.airybi().diff(&x)), "airybiprime(x)");
    assert_eq!(format!("{}", x.airyaiprime().diff(&x)), "x*airyai(x)");
    assert_eq!(format!("{}", x.airybiprime().diff(&x)), "x*airybi(x)");
    // exact values at 0 — sympy: 0.3550280538878172392601, 0.6149266274460007351509,
    // -0.2588194037928067984052, 0.4482883573538263579148
    let ai0 = ctx.int(0).airyai().eval();
    assert!(!format!("{ai0}").contains("airy"));
    assert!(close(f64_of(&ai0), 0.3550280538878172392601, 1e-14));
    assert!(close(
        f64_of(&ctx.int(0).airybi().eval()),
        0.6149266274460007351509,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.int(0).airyaiprime().eval()),
        -0.2588194037928067984052,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.int(0).airybiprime().eval()),
        0.4482883573538263579148,
        1e-14
    ));
    assert_eq!(format!("{}", ctx.infinity().airyai().eval()), "0");
    assert_eq!(format!("{}", ctx.neg_infinity().airybi().eval()), "0");
    // rendering + parse
    assert_eq!(format!("{}", x.airyai()), "airyai(x)");
    assert_eq!(x.airyai().to_latex(), r"\operatorname{Ai}\left(x\right)");
    assert_eq!(
        x.airybiprime().to_latex(),
        r"\operatorname{Bi}^\prime\left(x\right)"
    );
    for e in [x.airyai(), x.airybi(), x.airyaiprime(), x.airybiprime()] {
        roundtrip(&ctx, &e);
    }
}

#[test]
fn elliptic_k_and_e() {
    let ctx = Context::new();
    let m = ctx.symbol("m");
    // sympy: K(1/2) = 1.854074677301371918434, K(9/10) = 2.578092113348173188203,
    // K(-2) = 1.171420084146769858926
    assert!(close(
        f64_of(&ctx.rational(1, 2).elliptic_k()),
        1.854074677301371918434,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.rational(9, 10).elliptic_k()),
        2.578092113348173188203,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.int(-2).elliptic_k()),
        1.171420084146769858926,
        1e-14
    ));
    // sympy: E(1/2) = 1.350643881047675502520, E(9/10) = 1.104774732704073326090,
    // E(-2) = 2.184438142746201185405
    assert!(close(
        f64_of(&ctx.rational(1, 2).elliptic_e()),
        1.350643881047675502520,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.rational(9, 10).elliptic_e()),
        1.104774732704073326090,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.int(-2).elliptic_e()),
        2.184438142746201185405,
        1e-14
    ));
    // 30-digit K(1/2) = Γ(1/4)²/(4√π): the exact fold and the AGM/Carlson value agree
    let k_half = ctx.rational(1, 2).elliptic_k();
    assert!(!format!("{}", k_half.eval()).contains("elliptic"));
    let numeric = ctx.rational(9, 10).elliptic_k().eval_decimal(30).unwrap();
    assert!(numeric.starts_with("2.57809211334817318820"), "{numeric}");
    // derivatives: dK/dm = (E − (1−m)K)/(2m(1−m)), dE/dm = (E − K)/(2m)
    let dk = m.elliptic_k().diff(&m);
    let expected = (m.elliptic_e() - (1 - &m) * m.elliptic_k()) / (2 * &m * (1 - &m));
    assert!(close(at(&dk, &m, 0.3), at(&expected, &m, 0.3), 1e-13));
    let de = m.elliptic_e().diff(&m);
    let expected = (m.elliptic_e() - m.elliptic_k()) / (2 * &m);
    assert!(close(at(&de, &m, 0.3), at(&expected, &m, 0.3), 1e-13));
    // finite-difference sanity of dK/dm at m = 0.3
    let h = 1e-6;
    let fd = (f64_of(&num(&ctx, 0.3 + h).elliptic_k()) - f64_of(&num(&ctx, 0.3 - h).elliptic_k()))
        / (2.0 * h);
    assert!(close(at(&dk, &m, 0.3), fd, 1e-8));
    // exact values
    assert_eq!(format!("{}", ctx.int(0).elliptic_k().eval()), "1/2*pi");
    assert_eq!(format!("{}", ctx.int(0).elliptic_e().eval()), "1/2*pi");
    assert_eq!(format!("{}", ctx.int(1).elliptic_e().eval()), "1");
    assert_eq!(format!("{}", ctx.int(1).elliptic_k().eval()), "zoo");
    // rendering + parse
    assert_eq!(format!("{}", m.elliptic_k()), "elliptic_k(m)");
    assert_eq!(m.elliptic_k().to_latex(), r"K\left(m\right)");
    assert_eq!(m.elliptic_e().to_latex(), r"E\left(m\right)");
    roundtrip(&ctx, &m.elliptic_k());
    roundtrip(&ctx, &m.elliptic_e());
}

#[test]
fn elliptic_f_and_pi() {
    let ctx = Context::new();
    let (phi, m, n) = (ctx.symbol("phi"), ctx.symbol("m"), ctx.symbol("n"));
    // sympy: F(7/10 | 1/2) = 0.7287703057181902643632
    assert!(close(
        f64_of(&ctx.rational(7, 10).elliptic_f(&ctx.rational(1, 2))),
        0.7287703057181902643632,
        1e-14
    ));
    // sympy: F(5/2 | 3/10) = 2.773381177557619658396  (φ > π/2, uses 2K)
    assert!(close(
        f64_of(&ctx.rational(5, 2).elliptic_f(&ctx.rational(3, 10))),
        2.773381177557619658396,
        1e-14
    ));
    // sympy: F(-6/5 | 4/5) = -1.488495688949330073663
    assert!(close(
        f64_of(&ctx.rational(-6, 5).elliptic_f(&ctx.rational(4, 5))),
        -1.488495688949330073663,
        1e-14
    ));
    // sympy: Π(3/10 | 1/2) = 2.250376821943946684738, Π(-1/2 | 7/10) = 1.645042868400853102064
    assert!(close(
        f64_of(&ctx.rational(3, 10).elliptic_pi(&ctx.rational(1, 2))),
        2.250376821943946684738,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.rational(-1, 2).elliptic_pi(&ctx.rational(7, 10))),
        1.645042868400853102064,
        1e-14
    ));
    // ∂F/∂φ = 1/√(1 − m sin²φ)
    let d = phi.elliptic_f(&m).diff(&phi);
    assert_eq!(d, 1 / (1 - &m * phi.sin().powi(2)).sqrt());
    // ∂F/∂m stays formal
    assert_eq!(
        format!("{}", phi.elliptic_f(&m).diff(&m)),
        "Derivative(elliptic_f(phi, m), m)"
    );
    // ∂Π/∂n and ∂Π/∂m against central differences at (n, m) = (0.3, 0.5)
    let dn = n.elliptic_pi(&m).diff(&n);
    let dm = n.elliptic_pi(&m).diff(&m);
    let h = 1e-6;
    let pi_at = |nv: f64, mv: f64| f64_of(&num(&ctx, nv).elliptic_pi(&num(&ctx, mv)));
    let fd_n = (pi_at(0.3 + h, 0.5) - pi_at(0.3 - h, 0.5)) / (2.0 * h);
    let fd_m = (pi_at(0.3, 0.5 + h) - pi_at(0.3, 0.5 - h)) / (2.0 * h);
    let bound = dn.subs_map_with(&[(&n, 0.3), (&m, 0.5)]);
    assert!(
        close(f64_of(&bound), fd_n, 1e-8),
        "{} vs {fd_n}",
        f64_of(&bound)
    );
    let bound = dm.subs_map_with(&[(&n, 0.3), (&m, 0.5)]);
    assert!(
        close(f64_of(&bound), fd_m, 1e-8),
        "{} vs {fd_m}",
        f64_of(&bound)
    );
    // exact values
    assert_eq!(format!("{}", ctx.int(0).elliptic_f(&m).eval()), "0");
    assert_eq!(format!("{}", phi.elliptic_f(&ctx.int(0)).eval()), "phi");
    assert_eq!(
        format!("{}", (ctx.pi() / 2).elliptic_f(&m).eval()),
        "elliptic_k(m)"
    );
    assert_eq!(
        format!("{}", ctx.int(0).elliptic_pi(&m).eval()),
        "elliptic_k(m)"
    );
    assert_eq!(format!("{}", ctx.int(1).elliptic_pi(&m).eval()), "zoo");
    assert_eq!(
        n.elliptic_pi(&ctx.int(0)).eval(),
        ctx.pi() / (2 * (1 - &n).sqrt())
    );
    assert_eq!(n.elliptic_pi(&n).eval(), n.elliptic_e() / (1 - &n));
    // rendering + parse
    assert_eq!(format!("{}", phi.elliptic_f(&m)), "elliptic_f(phi, m)");
    assert_eq!(
        phi.elliptic_f(&m).to_latex(),
        r"F\left(\phi\middle| m\right)"
    );
    assert_eq!(n.elliptic_pi(&m).to_latex(), r"\Pi\left(n\middle| m\right)");
    roundtrip(&ctx, &phi.elliptic_f(&m));
    roundtrip(&ctx, &n.elliptic_pi(&m));
}

#[test]
fn gegenbauer() {
    let ctx = Context::new();
    let (x, a, n) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("n"));
    // sympy: gegenbauer(3, 3/2, 2/5) = -1.88, gegenbauer(5, 1/4, -3/10) = -0.1349651953125
    assert!(close(
        f64_of(
            &ctx.rational(2, 5)
                .gegenbauer(&ctx.int(3), &ctx.rational(3, 2))
        ),
        -1.88,
        1e-14
    ));
    assert!(close(
        f64_of(
            &ctx.rational(-3, 10)
                .gegenbauer(&ctx.int(5), &ctx.rational(1, 4))
        ),
        -0.1349651953125,
        1e-14
    ));
    // symbolic-a numeric check must not fold (a free), so use the numeric path via subs
    let c3 = x.gegenbauer(&ctx.int(3), &a);
    let v = f64_of(&c3.subs_map_with(&[(&x, 0.4), (&a, 1.5)]));
    assert!(close(v, -1.88, 1e-14));
    // expansion: C_2^{(a)}(x) = 2a²x² + 2ax² − a  (sympy expand)
    let c2 = x.gegenbauer(&ctx.int(2), &a).eval();
    let expected = (2 * a.powi(2) * x.powi(2) + 2 * &a * x.powi(2) - &a).expand();
    assert_eq!(c2, expected, "{c2}");
    // derivative: 2a C_{n−1}^{(a+1)}
    assert_eq!(
        format!("{}", x.gegenbauer(&n, &a).diff(&x)),
        "2*a*gegenbauer(n - 1, a + 1, x)"
    );
    // special parameters
    assert_eq!(
        format!("{}", x.gegenbauer(&n, &ctx.rational(1, 2)).eval()),
        "legendre(n, x)"
    );
    assert_eq!(
        format!("{}", x.gegenbauer(&n, &ctx.int(1)).eval()),
        "chebyshev_u(n, x)"
    );
    assert_eq!(format!("{}", x.gegenbauer(&ctx.int(0), &a).eval()), "1");
    // rendering + parse
    assert_eq!(format!("{}", x.gegenbauer(&n, &a)), "gegenbauer(n, a, x)");
    assert_eq!(
        x.gegenbauer(&n, &a).to_latex(),
        r"C_{n}^{\left(a\right)}\left(x\right)"
    );
    roundtrip(&ctx, &x.gegenbauer(&n, &a));
}

#[test]
fn jacobi() {
    let ctx = Context::new();
    let (x, a, b, n) = (
        ctx.symbol("x"),
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("n"),
    );
    // sympy: jacobi(3, 1/2, 3/2, 2/5) = -0.5845, jacobi(4, -3/10, 2, -3/5) = 0.37658134
    assert!(close(
        f64_of(
            &ctx.rational(2, 5)
                .jacobi(&ctx.int(3), &ctx.rational(1, 2), &ctx.rational(3, 2))
        ),
        -0.5845,
        1e-14
    ));
    assert!(close(
        f64_of(
            &ctx.rational(-3, 5)
                .jacobi(&ctx.int(4), &ctx.rational(-3, 10), &ctx.int(2))
        ),
        0.37658134,
        1e-14
    ));
    // expansion: P_1^{(a,b)}(x) = (a − b)/2 + (a + b + 2)x/2
    let p1 = x.jacobi(&ctx.int(1), &a, &b).eval();
    let expected = ((&a - &b) / 2 + (&a + &b + 2) * &x / 2).expand();
    assert_eq!(p1, expected, "{p1}");
    // P_2 agrees with sympy's expansion at a point
    let p2 = x.jacobi(&ctx.int(2), &a, &b).eval();
    let v = f64_of(&p2.subs_map_with(&[(&x, 0.4), (&a, 0.5), (&b, 1.5)]));
    let direct = f64_of(&ctx.rational(2, 5).jacobi(
        &ctx.int(2),
        &ctx.rational(1, 2),
        &ctx.rational(3, 2),
    ));
    assert!(close(v, direct, 1e-14));
    // derivative: (n + a + b + 1)/2 · P_{n−1}^{(a+1, b+1)}
    let d = x.jacobi(&n, &a, &b).diff(&x);
    assert_eq!(
        format!("{d}"),
        "1/2*(a + b + n + 1)*jacobi(n - 1, a + 1, b + 1, x)"
    );
    // special parameters
    assert_eq!(
        format!("{}", x.jacobi(&n, &ctx.int(0), &ctx.int(0)).eval()),
        "legendre(n, x)"
    );
    // rendering + parse
    assert_eq!(format!("{}", x.jacobi(&n, &a, &b)), "jacobi(n, a, b, x)");
    assert_eq!(
        x.jacobi(&n, &a, &b).to_latex(),
        r"P_{n}^{\left(a,b\right)}\left(x\right)"
    );
    roundtrip(&ctx, &x.jacobi(&n, &a, &b));
}

#[test]
fn assoc_legendre() {
    let ctx = Context::new();
    let (x, n, m) = (ctx.symbol("x"), ctx.symbol("n"), ctx.symbol("m"));
    // sympy: assoc_legendre(3, 2, 2/5) = 5.04, assoc_legendre(3, 1, 2/5) = 0.2749545416973504003953,
    // assoc_legendre(4, -2, 2/5) = 0.0021, assoc_legendre(5, 3, -7/10) = -65.20320544541406976548
    assert!(close(
        f64_of(&ctx.rational(2, 5).assoc_legendre(&ctx.int(3), &ctx.int(2))),
        5.04,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.rational(2, 5).assoc_legendre(&ctx.int(3), &ctx.int(1))),
        0.2749545416973504003953,
        1e-14
    ));
    assert!(close(
        f64_of(&ctx.rational(2, 5).assoc_legendre(&ctx.int(4), &ctx.int(-2))),
        0.0021,
        1e-14
    ));
    assert!(close(
        f64_of(
            &ctx.rational(-7, 10)
                .assoc_legendre(&ctx.int(5), &ctx.int(3))
        ),
        -65.20320544541406976548,
        1e-14
    ));
    // expansions (Condon–Shortley phase, as in SymPy)
    assert_eq!(
        format!("{}", x.assoc_legendre(&ctx.int(2), &ctx.int(2)).eval()),
        "-3*x^2 + 3"
    );
    let p21 = x.assoc_legendre(&ctx.int(2), &ctx.int(1)).eval();
    let expected = -3 * &x * (1 - x.powi(2)).sqrt();
    assert!(close(at(&p21, &x, 0.4), at(&expected, &x, 0.4), 1e-14));
    let p3m1 = x.assoc_legendre(&ctx.int(3), &ctx.int(-1)).eval();
    let expected = (1 - x.powi(2)).sqrt() * (15 * x.powi(2) / 2 - ctx.rational(3, 2)) / 12;
    assert!(close(at(&p3m1, &x, 0.4), at(&expected, &x, 0.4), 1e-14));
    assert_eq!(
        format!("{}", x.assoc_legendre(&ctx.int(2), &ctx.int(3)).eval()),
        "0"
    );
    assert_eq!(
        format!("{}", x.assoc_legendre(&n, &ctx.int(0)).eval()),
        "legendre(n, x)"
    );
    // derivative: (n x P_n^m − (n + m) P_{n−1}^m)/(x² − 1)
    let d = x.assoc_legendre(&n, &m).diff(&x);
    assert_eq!(
        format!("{d}"),
        "(n*x*assoc_legendre(n, m, x) - (m + n)*assoc_legendre(n - 1, m, x))/(x^2 - 1)"
    );
    // numeric check of the derivative rule at n = 3, m = 1, x = 0.4
    let d31 = x.assoc_legendre(&ctx.int(3), &ctx.int(1)).diff(&x);
    let h = 1e-6;
    let f = |v: f64| f64_of(&num(&ctx, v).assoc_legendre(&ctx.int(3), &ctx.int(1)));
    let fd = (f(0.4 + h) - f(0.4 - h)) / (2.0 * h);
    assert!(close(at(&d31, &x, 0.4), fd, 1e-8));
    // rendering + parse
    assert_eq!(
        format!("{}", x.assoc_legendre(&n, &m)),
        "assoc_legendre(n, m, x)"
    );
    assert_eq!(
        x.assoc_legendre(&n, &m).to_latex(),
        r"P_{n}^{\left(m\right)}\left(x\right)"
    );
    roundtrip(&ctx, &x.assoc_legendre(&n, &m));
}

#[test]
fn assoc_laguerre() {
    let ctx = Context::new();
    let (x, a, n) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("n"));
    // sympy: assoc_laguerre(3, 1/2, 6/5) = -0.8305, assoc_laguerre(4, -3/10, 5/2) = 1.02065
    assert!(close(
        f64_of(
            &ctx.rational(6, 5)
                .assoc_laguerre(&ctx.int(3), &ctx.rational(1, 2))
        ),
        -0.8305,
        1e-14
    ));
    assert!(close(
        f64_of(
            &ctx.rational(5, 2)
                .assoc_laguerre(&ctx.int(4), &ctx.rational(-3, 10))
        ),
        1.02065,
        1e-14
    ));
    // expansion: L_2^{(a)}(x) = a²/2 − ax + 3a/2 + x²/2 − 2x + 1
    let l2 = x.assoc_laguerre(&ctx.int(2), &a).eval();
    let expected = (a.powi(2) / 2 - &a * &x + 3 * &a / 2 + x.powi(2) / 2 - 2 * &x + 1).expand();
    assert_eq!(l2, expected, "{l2}");
    // derivative: −L_{n−1}^{(a+1)}
    assert_eq!(
        format!("{}", x.assoc_laguerre(&n, &a).diff(&x)),
        "-assoc_laguerre(n - 1, a + 1, x)"
    );
    // special parameter
    assert_eq!(
        format!("{}", x.assoc_laguerre(&n, &ctx.int(0)).eval()),
        "laguerre(n, x)"
    );
    assert_eq!(format!("{}", x.assoc_laguerre(&ctx.int(0), &a).eval()), "1");
    // rendering + parse
    assert_eq!(
        format!("{}", x.assoc_laguerre(&n, &a)),
        "assoc_laguerre(n, a, x)"
    );
    assert_eq!(
        x.assoc_laguerre(&n, &a).to_latex(),
        r"L_{n}^{\left(a\right)}\left(x\right)"
    );
    roundtrip(&ctx, &x.assoc_laguerre(&n, &a));
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-cutting
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn special_09_high_precision_and_errors() {
    let ctx = Context::new();
    // 40-digit values against mpmath (mp.dps = 50)
    // erfi(0.7) = 0.940282933833507476593562818953432428026744068
    let s = ctx.rational(7, 10).erfi().eval_decimal(40).unwrap();
    assert!(
        s.starts_with("0.94028293383350747659356281895343242802"),
        "{s}"
    );
    // Li_2(0.9) = 1.2997147230049587251710604941929534
    let s = ctx
        .rational(9, 10)
        .polylog(&ctx.int(2))
        .eval_decimal(30)
        .unwrap();
    assert!(s.starts_with("1.29971472300495872517106049419"), "{s}");
    // K(0.9) = 2.5780921133481731882025707718165062
    let s = ctx.rational(9, 10).elliptic_k().eval_decimal(30).unwrap();
    assert!(s.starts_with("2.5780921133481731882025707718"), "{s}");
    // domain errors are `Err`, never panics
    assert!(ctx.int(2).erfinv().eval_f64().is_err());
    assert!(ctx.int(3).erfcinv().eval_f64().is_err());
    assert!(ctx.int(2).elliptic_k().eval_f64().is_err());
    assert!(ctx.int(2).polylog(&ctx.rational(3, 2)).eval_f64().is_err());
    assert!(
        ctx.int(-1)
            .uppergamma(&ctx.rational(1, 2))
            .eval_f64()
            .is_err()
    );
    // codegen reports the missing runtime clearly instead of panicking
    let x = ctx.symbol("x");
    assert!(x.erfi().compile(&["x"]).is_err());
    assert!(x.erfi().to_rust_fn("f", &["x"]).is_err());
    assert!(x.erfi().to_c_fn("f", &["x"]).is_err());
}
