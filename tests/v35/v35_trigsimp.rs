//! Trigonometric and hyperbolic identities in `simplify` (0.48).
//!
//! Up to 0.47 Fu (`src/simplify/fu.rs`) chose by its measure whether to
//! write `tan` as `sin/cos` (TR2); that raises the trig count, so it never
//! did, and the Pythagorean substitutions never saw the `sin²x` inside
//! `tan²x`: `sec²x − tan²x` stayed while `csc²x − cot²x` was `1` (`cot` has
//! no node of its own, it is built as `cos/sin`, so no `tan` was in the
//! way).  SymPy's `fu` applies TR2 unconditionally and keeps the `tan` form
//! as a candidate; so does Fu now.  Hyperbolic functions reached no
//! identity beyond the pattern rules `cosh² − sinh² = 1`, `2·sinh·cosh =
//! sinh 2x`, `cosh² + sinh² = cosh 2x`: they now go through Fu by Osborne's
//! rule (SymPy's `hyper_as_trig`), and next to exponentials through their
//! exponential form.  Powers of `sin` and `cos` are linearized (TRpower,
//! the power case of SymPy's TR8) as one more Fu alternative.
//!
//! Oracles: every expected value is an exact identity; the hunter's oracle
//! checked each result against the input with mpmath 1.3.0 at 30 digits at
//! three complex points.  SymPy 1.14 `simplify` values are quoted per test.

use symplex::prelude::*;

fn simp(s: &str) -> String {
    let ctx = Context::new();
    ctx.parse(s).unwrap().simplify().to_string()
}

/// `sec²x − tan²x = 1` in its guises.  Up to 0.47 `sec(x)^2 - tan(x)^2`
/// and `1/cos(x)^2 - tan(x)^2` gave `-tan(x)^2 + cos(x)^(-2)`, `tan(x)^2 +
/// 1` stayed, `sec(x)^4 - tan(x)^4 - 2*tan(x)^2` was `-tan(x)^4 -
/// 2*tan(x)^2 + cos(x)^(-4)` and `sec(x)^4 - tan(x)^4` was `-tan(x)^4 +
/// cos(x)^(-4)`.  SymPy 1.14: `simplify(sec(x)**2 - tan(x)**2)` → `1`,
/// `simplify(tan(x)**2 + 1)` → `cos(x)**(-2)`, `simplify(sec(x)**4 -
/// tan(x)**4)` → `-tan(x)**4 + sec(x)**4` (unchanged; `(sec² − tan²)·(sec²
/// + tan²) = 2·sec²x − 1`).
#[test]
fn sec_tan_pythagorean_identity() {
    assert_eq!(simp("sec(x)^2 - tan(x)^2"), "1");
    assert_eq!(simp("1/cos(x)^2 - tan(x)^2"), "1");
    assert_eq!(simp("csc(x)^2 - cot(x)^2"), "1");
    assert_eq!(simp("tan(x)^2 + 1"), "cos(x)^(-2)");
    assert_eq!(simp("sec(x)^4 - tan(x)^4 - 2*tan(x)^2"), "1");
    assert_eq!(simp("sec(x)^4 - tan(x)^4"), "2*cos(x)^(-2) - 1");
}

/// The same at an angle that TR11 expands (`2x`, `x + 1`): the Pythagorean
/// substitutions now also see the unexpanded angle.  Up to 0.47 both gave
/// `-tan(…)^2 + cos(…)^(-2)`.  SymPy 1.14: `simplify(sec(2*x)**2 -
/// tan(2*x)**2)` → `1`, but `simplify(sec(x + 1)**2 - tan(x + 1)**2)` →
/// `-tan(x + 1)**2 + sec(x + 1)**2` (unsimplified).
#[test]
fn sec_tan_pythagorean_identity_at_compound_angles() {
    assert_eq!(simp("sec(2*x)^2 - tan(2*x)^2"), "1");
    assert_eq!(simp("sec(x + 1)^2 - tan(x + 1)^2"), "1");
    assert_eq!(simp("sec(2*x)^2 - tan(2*x)^2 + 1/x"), "1/x + 1");
}

/// The hyperbolic Pythagorean identity with `tanh`, `sech`, `coth`,
/// `csch`.  Up to 0.47 `tanh(x)^2 + sech(x)^2` (and `tanh(x)^2 +
/// 1/cosh(x)^2`) gave `tanh(x)^2 + cosh(x)^(-2)`, `coth(x)^2 - csch(x)^2`
/// gave `sinh(x)^(-2)*(cosh(x)^2 - 1)`, `1 - tanh(x)^2` stayed.  SymPy
/// 1.14 `simplify`: `1`, `1`, `cosh(x)**(-2)`, and `coth(x)**2 - 1` →
/// `sinh(x)**(-2)`.
#[test]
fn hyperbolic_pythagorean_reciprocals() {
    assert_eq!(simp("tanh(x)^2 + sech(x)^2"), "1");
    assert_eq!(simp("tanh(x)^2 + 1/cosh(x)^2"), "1");
    assert_eq!(simp("coth(x)^2 - csch(x)^2"), "1");
    assert_eq!(simp("1 - tanh(x)^2"), "cosh(x)^(-2)");
    assert_eq!(simp("coth(x)^2 - 1"), "sinh(x)^(-2)");
}

/// The hyperbolic identities at a compound angle, beside a constant, and
/// beside trigonometric functions (masked while the hyperbolic ones are
/// read as trigonometric, `sin x` shared with `cos(sin x)` included).  Up to
/// 0.47 all stayed (`tanh(x + 1)^2 + cosh(x + 1)^(-2)`, `-tanh(x)^2 + 2`,
/// …).  SymPy 1.14 `simplify`: `tanh(x + 1)**2 + sech(x + 1)**2` and `2 -
/// tanh(x)**2` (unsimplified; `2 − tanh²x = 1 + sech²x` is shorter in
/// symplex's count), `sin(x) + 1`, `sin(x) + cos(sin(x)) + cosh(2*x)`.
#[test]
fn hyperbolic_pythagorean_in_context() {
    assert_eq!(simp("tanh(x + 1)^2 + sech(x + 1)^2"), "1");
    assert_eq!(simp("2 - tanh(x)^2"), "cosh(x)^(-2) + 1");
    assert_eq!(simp("tanh(x)^2 + sech(x)^2 + sin(x)"), "sin(x) + 1");
    assert_eq!(
        simp("cos(sin(x)) + sin(x) + cosh(x)^4 - sinh(x)^4"),
        "sin(x) + cos(sin(x)) + cosh(2*x)"
    );
}

/// Double and triple angles of hyperbolic functions, and a constant in
/// the argument (read back as `cosh(u + i·k)` from the trigonometric angle
/// `u·d + k`, here with `k = 0` after the identities).  Up to 0.47 all of
/// these stayed except `cosh(x)^2 + sinh(x)^2` (a pattern rule).  SymPy
/// 1.14 `simplify`: `0`, `cosh(2*x)`, `cosh(3*x)`, `sinh(x)`, `-1`.
#[test]
fn hyperbolic_multiple_angles() {
    assert_eq!(simp("cosh(2*x) - cosh(x)^2 - sinh(x)^2"), "0");
    assert_eq!(simp("cosh(x)^4 - sinh(x)^4"), "cosh(2*x)");
    assert_eq!(simp("4*cosh(x)^3 - 3*cosh(x)"), "cosh(3*x)");
    assert_eq!(simp("tanh(x)*cosh(x)"), "sinh(x)");
    assert_eq!(simp("sinh(x + 1)^2 - cosh(x + 1)^2"), "-1");
    assert_eq!(simp("cosh(x)^2 + sinh(x)^2"), "cosh(2*x)");
}

/// Hyperbolic functions next to exponentials, and `cosh ± sinh`.  Up to
/// 0.47 these stayed (`-1/2*exp(2*x) + exp(x)*cosh(x) - 1/2` for the
/// first).  SymPy 1.14: `simplify(exp(x)*cosh(x) - (exp(2*x) + 1)/2)` →
/// `-exp(2*x)/2 + exp(x)*cosh(x) - 1/2` (unsimplified); the others → `0`,
/// `0`, `exp(x)`, `exp(-x)`.
#[test]
fn exponential_forms_of_hyperbolic_functions() {
    assert_eq!(simp("exp(x)*cosh(x) - (exp(2*x) + 1)/2"), "0");
    assert_eq!(simp("cosh(x) - (exp(x) + exp(-x))/2"), "0");
    assert_eq!(simp("sinh(x) - (exp(x) - exp(-x))/2"), "0");
    assert_eq!(simp("sinh(x) + cosh(x)"), "exp(x)");
    assert_eq!(simp("cosh(x) - sinh(x)"), "exp(-x)");
}

/// Multiple angles from powers.  Up to 0.47 `4*cos(x)^3 - 3*cos(x)` and
/// `3*sin(x) - 4*sin(x)^3` stayed, and `sin(x)^4 - cos(x)^4` was
/// `2*sin(x)^2 - 1` (4 operations; `-cos(2*x)` has 3).  SymPy 1.14
/// `simplify`: `cos(3*x)`, `sin(3*x)`, `-cos(2*x)`; `cos(3*x) + 3*cos(x)`
/// stays in SymPy (`3*cos(x) + cos(3*x)`), symplex gives the shorter
/// `4*cos(x)^3` (0.47 already).
#[test]
fn multiple_angles_from_powers() {
    assert_eq!(simp("4*cos(x)^3 - 3*cos(x)"), "cos(3*x)");
    assert_eq!(simp("3*sin(x) - 4*sin(x)^3"), "sin(3*x)");
    assert_eq!(simp("sin(x)^4 - cos(x)^4"), "-cos(2*x)");
    assert_eq!(simp("cos(3*x) + 3*cos(x)"), "4*cos(x)^3");
}

/// Power reduction on its own (CTR1) next to a polynomial: up to 0.47
/// `x^2 + 2*cos(x)^2` stayed (the TR11 inside RL2's third alternative
/// expanded the double angle back).  SymPy 1.14: `x**2 + cos(2*x) + 1`.
#[test]
fn power_reduction_beside_a_polynomial() {
    assert_eq!(simp("x^2 + 2*cos(x)^2"), "x^2 + cos(2*x) + 1");
}

/// `Ex::fu` itself (TR2 applied whenever `tan` remains): up to 0.47
/// `fu(sec(x)^2 - tan(x)^2)` returned its input.
#[test]
fn fu_writes_tan_as_sin_over_cos() {
    let ctx = Context::new();
    let e = ctx.parse("sec(x)^2 - tan(x)^2").unwrap();
    assert_eq!(e.fu().to_string(), "1");
    let t = ctx.parse("tan(x)").unwrap();
    assert_eq!(t.fu().to_string(), "tan(x)");
}

/// `simplify` never returns a longer expression, and leaves alone what no
/// identity shortens (the Osborne and exponential strategies included).
#[test]
fn never_longer() {
    let ctx = Context::new();
    for s in [
        "sinh(x)",
        "cosh(x)^2",
        "x*sinh(x) + cosh(x)",
        "sin(x)^3",
        "sin(x)^3*cos(x)^3",
        "tan(x) + sin(x)",
        "sech(x)^2 + csch(x)^2",
        "cosh(x) + exp(x)",
        "sinh(x)*exp(2*x)",
        "tanh(x)^3 + cosh(y)",
    ] {
        let e = ctx.parse(s).unwrap();
        let r = e.simplify();
        assert!(
            r.count_ops() <= e.count_ops(),
            "simplify({s}) = {r} is longer"
        );
    }
}
