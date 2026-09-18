//! v0.2 indefinite integration battery: ≥ 80 standard integrals, each
//! verified by differentiating the antiderivative back and comparing with
//! the integrand numerically at several points.

use symplex::prelude::*;

/// Points at which the FTC check is performed (avoid 0, ±1 and small
/// integers where common integrands have poles or branch points).
const POINTS: &[(i64, i64)] = &[(3, 10), (7, 10), (13, 10), (23, 10), (-7, 10)];

/// Integrate, then differentiate back and compare with the integrand.
///
/// Points where the integrand itself cannot be evaluated (domain
/// restrictions) are skipped; at least two points must succeed.
fn verify_one(label: &str, f: &Ex, x: &Ex) -> Result<(), String> {
    let anti = f.integrate(x);
    if anti.has_unevaluated() {
        return Err(format!(
            "{label}: integration returned an unevaluated form: {anti}"
        ));
    }
    let d = anti.diff(x);
    let ctx = f.context();
    let mut checked = 0;
    for &(p, q) in POINTS {
        let pt = ctx.rational(p, q);
        let fv = f.subs(x, &pt).eval().eval_f64();
        let dv = d.subs(x, &pt).eval().eval_f64();
        match (fv, dv) {
            (Ok(fv), Ok(dv)) if fv.is_finite() => {
                checked += 1;
                let scale = fv.abs().max(dv.abs()).max(1.0);
                if (fv - dv).abs() >= 1e-8 * scale {
                    return Err(format!(
                        "{label}: d/dx({anti}) = {d} ≠ integrand at x={p}/{q}: {dv} vs {fv}"
                    ));
                }
            }
            (Ok(fv), Err(e)) if fv.is_finite() => {
                return Err(format!(
                    "{label}: integrand = {fv} at x={p}/{q} but derivative {d} failed: {e}"
                ));
            }
            _ => {}
        }
    }
    if checked < 2 {
        return Err(format!(
            "{label}: fewer than two evaluation points succeeded"
        ));
    }
    Ok(())
}

/// Collects failures so that one report lists every failing integrand.
struct Battery {
    failures: Vec<String>,
    total: usize,
}

impl Battery {
    fn new() -> Self {
        Self {
            failures: Vec::new(),
            total: 0,
        }
    }
    fn verify(&mut self, label: &str, f: &Ex, x: &Ex) {
        self.total += 1;
        if let Err(e) = verify_one(label, f, x) {
            self.failures.push(e);
        }
    }
    fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "{} of {} integrals failed:\n{}",
            self.failures.len(),
            self.total,
            self.failures.join("\n")
        );
    }
}

// ── Powers, logs, exponentials ─────────────────────────────────────────

#[test]
fn battery_powers_logs_exponentials() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut b = Battery::new();
    b.verify("x^n ln x", &(&x.powi(3) * &x.ln()), &x);
    b.verify("ln x / x^2", &(&x.ln() / &x.powi(2)), &x);
    b.verify("1/(x ln x)", &(&ctx.int(1) / &(&x * &x.ln())), &x);
    b.verify("ln(x)^2", &x.ln().powi(2), &x);
    b.verify("ln(x)^3", &x.ln().powi(3), &x);
    b.verify("ln(x^2+1)", &(&x.powi(2) + 1).ln(), &x);
    b.verify("x ln(x^2+1)", &(&x * &(&x.powi(2) + 1).ln()), &x);
    b.verify("x e^{x^2}", &(&x * &x.powi(2).exp()), &x);
    b.verify("x^3 e^{x^2}", &(&x.powi(3) * &x.powi(2).exp()), &x);
    b.verify("x^2 e^{-x}", &(&x.powi(2) * &(-&x).exp()), &x);
    b.verify("e^{ax} cos(bx)", &(&(&x * 2).exp() * &(&x * 3).cos()), &x);
    b.verify("e^{ax} sin(bx)", &(&(-&x).exp() * &(&x * 2).sin()), &x);
    b.verify("x e^x sin x", &(&(&x * &x.exp()) * &x.sin()), &x);
    b.verify("1/(1+e^x)", &(&ctx.int(1) / &(&x.exp() + 1)), &x);
    b.verify("e^x/(1+e^{2x})", &(&x.exp() / &(&(&x * 2).exp() + 1)), &x);
    b.verify("e^x/(1+e^x)", &(&x.exp() / &(&x.exp() + 1)), &x);
    b.verify("e^{sqrt x}", &x.sqrt().exp(), &x);
    b.verify("x^{1/3}", &x.pow(&ctx.rational(1, 3)), &x);
    b.verify(
        "1/(sqrt x (1+sqrt x))",
        &(&ctx.int(1) / &(&x.sqrt() * &(&x.sqrt() + 1))),
        &x,
    );
    b.verify("x sqrt(x+1)", &(&x * &(&x + 1).sqrt()), &x);
    b.verify("sqrt(x) ln x", &(&x.sqrt() * &x.ln()), &x);
    b.verify("e^{-x^2}", &(-x.powi(2)).exp(), &x);
    b.verify("erf(x)", &x.erf(), &x);
    b.verify("x e^{-x^2}", &(&x * &(-x.powi(2)).exp()), &x);
    b.verify("2^x", &ctx.int(2).pow(&x), &x);
    b.verify("x 2^x", &(&x * &ctx.int(2).pow(&x)), &x);
    b.finish();
}

// ── Rational functions ─────────────────────────────────────────────────

#[test]
fn battery_rational() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut b = Battery::new();
    b.verify("1/(x^2-a^2)", &(&ctx.int(1) / &(&x.powi(2) - 4)), &x);
    b.verify(
        "1/(a^2x^2+b^2)",
        &(&ctx.int(1) / &(&(&x.powi(2) * 9) + 4)),
        &x,
    );
    b.verify(
        "1/(x(x^2+1))",
        &(&ctx.int(1) / &(&x * &(&x.powi(2) + 1))),
        &x,
    );
    b.verify(
        "(x+1)/(x^2+2x+5)",
        &(&(&x + 1) / &(&(&x.powi(2) + &(&x * 2)) + 5)),
        &x,
    );
    b.verify("1/(x^3+1)", &(&ctx.int(1) / &(&x.powi(3) + 1)), &x);
    b.verify("x/(x^4+1)", &(&x / &(&x.powi(4) + 1)), &x);
    b.verify("1/(x^4+1)", &(&ctx.int(1) / &(&x.powi(4) + 1)), &x);
    b.verify(
        "1/(x^2+x+1)",
        &(&ctx.int(1) / &(&(&x.powi(2) + &x) + 1)),
        &x,
    );
    b.verify("x^2/(x^2+1)", &(&x.powi(2) / &(&x.powi(2) + 1)), &x);
    b.verify("1/(x^2-1)^2", &(&x.powi(2) - 1).powi(-2), &x);
    b.verify(
        "(3x+2)/(x^2-x-2)",
        &(&(&(&x * 3) + 2) / &(&(&x.powi(2) - &x) - 2)),
        &x,
    );
    b.verify("x^3/(x^2+1)", &(&x.powi(3) / &(&x.powi(2) + 1)), &x);
    b.verify(
        "1/(x^2(x+1))",
        &(&ctx.int(1) / &(&x.powi(2) * &(&x + 1))),
        &x,
    );
    b.verify("1/(x^2+1)^2", &(&x.powi(2) + 1).powi(-2), &x);
    b.finish();
}

// ── Trigonometric ──────────────────────────────────────────────────────

#[test]
fn battery_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut b = Battery::new();
    b.verify("sec^3", &x.cos().powi(-3), &x);
    b.verify("tan^2", &x.tan().powi(2), &x);
    b.verify("sin^2 cos^2", &(&x.sin().powi(2) * &x.cos().powi(2)), &x);
    b.verify("sin^3", &x.sin().powi(3), &x);
    b.verify("cos^4", &x.cos().powi(4), &x);
    b.verify("tan sec^2", &(&x.tan() / &x.cos().powi(2)), &x);
    b.verify("1/(sin cos)", &(&ctx.int(1) / &(&x.sin() * &x.cos())), &x);
    b.verify("x sin x cos x", &(&(&x * &x.sin()) * &x.cos()), &x);
    b.verify("sin(ax) sin(bx)", &(&(&x * 2).sin() * &(&x * 5).sin()), &x);
    b.verify("sin(ax) cos(bx)", &(&(&x * 3).sin() * &(&x * 4).cos()), &x);
    b.verify("x^2 cos x", &(&x.powi(2) * &x.cos()), &x);
    b.verify("sec x", &(&ctx.int(1) / &x.cos()), &x);
    b.verify("csc x", &(&ctx.int(1) / &x.sin()), &x);
    b.verify("sec^2 x", &x.cos().powi(-2), &x);
    b.verify("cot x", &(&x.cos() / &x.sin()), &x);
    b.verify("sin^2 x", &x.sin().powi(2), &x);
    b.verify("cos(sqrt x)", &x.sqrt().cos(), &x);
    b.verify("sin x / (1 + cos x)", &(&x.sin() / &(&x.cos() + 1)), &x);
    b.verify("1/(2 + cos x)", &(&ctx.int(1) / &(&x.cos() + 2)), &x);
    b.verify("sin(x) e^{cos x}", &(&x.sin() * &x.cos().exp()), &x);
    b.verify("tan^3", &x.tan().powi(3), &x);
    b.verify("cos(2x) sin(x)", &(&(&x * 2).cos() * &x.sin()), &x);
    b.finish();
}

// ── Inverse trig and hyperbolic ────────────────────────────────────────

#[test]
fn battery_inverse_trig_hyperbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut b = Battery::new();
    b.verify("arctan x", &x.atan(), &x);
    b.verify("x arctan x", &(&x * &x.atan()), &x);
    b.verify("arcsin x", &x.asin(), &x);
    b.verify("x arcsin x", &(&x * &x.asin()), &x);
    b.verify("arctan(2x)", &(&x * 2).atan(), &x);
    b.verify("sinh cosh", &(&x.sinh() * &x.cosh()), &x);
    b.verify("1/cosh x", &(&ctx.int(1) / &x.cosh()), &x);
    b.verify("tanh^2", &x.tanh().powi(2), &x);
    b.verify("cosh^2", &x.cosh().powi(2), &x);
    b.verify("x sinh x", &(&x * &x.sinh()), &x);
    b.verify("1/sinh x", &(&ctx.int(1) / &x.sinh()), &x);
    b.verify("sinh^2", &x.sinh().powi(2), &x);
    b.finish();
}

// ── Algebraic / radicals ───────────────────────────────────────────────

#[test]
fn battery_algebraic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut b = Battery::new();
    b.verify("sqrt(x^2+a^2)", &(&x.powi(2) + 4).sqrt(), &x);
    b.verify(
        "sqrt(x^2-a^2)",
        &(&x.powi(2) - &ctx.rational(1, 100)).sqrt(),
        &x,
    );
    b.verify(
        "1/sqrt(x^2+a^2)",
        &(&x.powi(2) + 1).pow(&ctx.rational(-1, 2)),
        &x,
    );
    b.verify(
        "x^2/sqrt(1-x^2)",
        &(&x.powi(2) / &(&ctx.int(1) - &x.powi(2)).sqrt()),
        &x,
    );
    b.verify("sqrt(1-x^2)", &(&ctx.int(1) - &x.powi(2)).sqrt(), &x);
    b.verify(
        "1/(x^2 sqrt(x^2-1))",
        &(&ctx.int(1) / &(&x.powi(2) * &(&x.powi(2) - &ctx.rational(1, 100)).sqrt())),
        &x,
    );
    b.verify("x/sqrt(x^2+1)", &(&x / &(&x.powi(2) + 1).sqrt()), &x);
    b.verify("x sqrt(x^2+1)", &(&x * &(&x.powi(2) + 1).sqrt()), &x);
    b.verify(
        "1/sqrt(4-x^2)",
        &(&ctx.int(4) - &x.powi(2)).pow(&ctx.rational(-1, 2)),
        &x,
    );
    b.verify("sqrt(2x+3)", &(&(&x * 2) + 3).sqrt(), &x);
    b.verify(
        "1/(x sqrt(x^2+1))",
        &(&ctx.int(1) / &(&x * &(&x.powi(2) + 1).sqrt())),
        &x,
    );
    b.verify("x^2 sqrt(x+1)", &(&x.powi(2) * &(&x + 1).sqrt()), &x);
    b.finish();
}

// ── Distributions and piecewise ────────────────────────────────────────

#[test]
fn battery_piecewise_like() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut b = Battery::new();
    // |x| → x|x|/2
    b.verify("abs(x)", &x.abs(), &x);
    // sign(x) → |x|
    b.verify("sign(x)", &x.sign(), &x);
    // |2x − 1|
    b.verify("abs(2x-1)", &(&(&x * 2) - 1).abs(), &x);
    // x |x|
    b.verify("x abs(x)", &(&x * &x.abs()), &x);
    // Heaviside
    b.verify("H(x-1)", &(&x - 1).heaviside(), &x);
    b.verify("x H(x)", &(&x * &x.heaviside()), &x);
    // Piecewise antiderivative
    let cond = x.lt(&ctx.int(0));
    let f = Ex::piecewise(&[(&x.powi(2), &cond), (&x, &ctx.int(1).ge(&ctx.int(0)))]);
    let anti = f.integrate(&x);
    assert!(!anti.has_unevaluated(), "piecewise: {anti}");
    let s = format!("{anti}");
    assert!(s.contains("Piecewise") || s.contains("piecewise"), "{s}");
    // |x² − 1| has real roots → must stay unevaluated rather than guess
    let g = (&x.powi(2) - 1).abs().integrate(&x);
    assert!(g.has_unevaluated(), "{g}");
    // |x² + 1| = x² + 1 (no real roots) → integrates
    b.verify("abs(x^2+1)", &(&x.powi(2) + 1).abs(), &x);
    b.finish();
}
