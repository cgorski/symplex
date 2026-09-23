//! symplex 0.2 — simplification gap-fill: `sqrtdenest`, `signsimp`,
//! `expand_with`/`ExpandOpts`, guarded `expand_power_base` / `expand_power_exp`,
//! `collect` over symbolic powers, `rcollect`, `collect_const`, trig identity
//! rules, `powdenest(force)`, guarded log combine/expand, `nsimplify` with
//! constants, and additive / dictionary variable separation.
//!
//! Every transformation is checked numerically at several rational points.

use symplex::macros::ExpandOpts;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

const POINTS: &[(i64, i64)] = &[(3, 7), (-5, 3), (11, 4), (1, 9), (-13, 6), (7, 2)];
const POS_POINTS: &[(i64, i64)] = &[(3, 7), (5, 3), (11, 4), (1, 9), (13, 6), (7, 2)];

fn assert_same_value_at(a: &Ex, b: &Ex, points: &[(i64, i64)], label: &str) {
    let ctx = a.context();
    let mut syms = a.free_symbols();
    for s in b.free_symbols() {
        if !syms.contains(&s) {
            syms.push(s);
        }
    }
    let mut checked = 0;
    for (i, &(p, q)) in points.iter().enumerate() {
        let mut ea = a.clone();
        let mut eb = b.clone();
        for (j, s) in syms.iter().enumerate() {
            let (pp, qq) = points[(i + j) % points.len()];
            let v = ctx.rational(pp + p, qq + q);
            ea = ea.subs(s, &v);
            eb = eb.subs(s, &v);
        }
        match (ea.eval_f64(), eb.eval_f64()) {
            (Ok(va), Ok(vb)) => {
                assert!(
                    (va - vb).abs() <= 1e-9 * (1.0 + va.abs().max(vb.abs())),
                    "{label}: value mismatch at point {i}: {va} vs {vb}\n  before: {a}\n  after:  {b}"
                );
                checked += 1;
            }
            (Err(_), Err(_)) => {}
            (ra, rb) => panic!(
                "{label}: one side failed to evaluate: {ra:?} vs {rb:?}\n  before: {a}\n  after: {b}"
            ),
        }
    }
    assert!(
        checked >= 2,
        "{label}: too few evaluable points ({checked})"
    );
}

fn assert_same_value(a: &Ex, b: &Ex, label: &str) {
    assert_same_value_at(a, b, POINTS, label);
}

fn assert_same_value_positive(a: &Ex, b: &Ex, label: &str) {
    assert_same_value_at(a, b, POS_POINTS, label);
}

fn s(e: &Ex) -> String {
    format!("{e}")
}

// ═══════════════════════════════════════════════════════════════════════════
// sqrtdenest
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sqrtdenest_three_plus_two_sqrt_two() {
    let ctx = Context::new();
    let e = (&ctx.int(2).sqrt() * 2 + 3).sqrt();
    let d = e.sqrtdenest();
    assert_eq!(s(&d), "sqrt(2) + 1");
    let (a, b) = (e.eval_f64().unwrap(), d.eval_f64().unwrap());
    assert!((a - b).abs() < 1e-12);
}

#[test]
fn sqrtdenest_five_minus_two_sqrt_six() {
    let ctx = Context::new();
    let e = (5 - &ctx.int(6).sqrt() * 2).sqrt();
    let d = e.sqrtdenest();
    assert_eq!(s(&d), "sqrt(3) - sqrt(2)");
    assert!((e.eval_f64().unwrap() - d.eval_f64().unwrap()).abs() < 1e-12);
}

#[test]
fn sqrtdenest_rational_coefficients() {
    // √(3/2 + √2) = (√2 + 1)/√2 ... a = 3/2, b = 1, c = 2: d = 9/4 - 2 = 1/4
    let ctx = Context::new();
    let e = (&ctx.int(2).sqrt() + ctx.rational(3, 2)).sqrt();
    let d = e.sqrtdenest();
    assert_ne!(d, e, "should denest: {d}");
    assert!(
        !s(&d).contains("sqrt(sqrt") && !s(&d).contains("+ sqrt(2))^(1/2)"),
        "{d}"
    );
    assert!((e.eval_f64().unwrap() - d.eval_f64().unwrap()).abs() < 1e-12);
}

#[test]
fn sqrtdenest_depth_two() {
    // √(2 + 4·√(3 + 2√2)) → inner = 1 + √2 → √(6 + 4√2) → 2 + √2
    let ctx = Context::new();
    let inner = (&ctx.int(2).sqrt() * 2 + 3).sqrt();
    let e = (&inner * 4 + 2).sqrt();
    let d = e.sqrtdenest();
    assert_eq!(s(&d), "sqrt(2) + 2");
    assert!((e.eval_f64().unwrap() - d.eval_f64().unwrap()).abs() < 1e-12);
}

#[test]
fn sqrtdenest_leaves_non_denestable_alone() {
    let ctx = Context::new();
    // √(2 + √2): d = 4 - 2 = 2, not a square.
    let e = (&ctx.int(2).sqrt() + 2).sqrt();
    assert_eq!(e.sqrtdenest(), e);
    // Negative a: √(-3 + 2√2) is imaginary — untouched.
    let f = (&ctx.int(2).sqrt() * 2 - 3).sqrt();
    assert_eq!(f.sqrtdenest(), f);
    // No radical at all.
    let x = ctx.symbol("x");
    assert_eq!((&x + 1).sqrt().sqrtdenest(), (&x + 1).sqrt());
}

#[test]
fn sqrtdenest_inverse_and_odd_half_powers() {
    let ctx = Context::new();
    let base = &ctx.int(2).sqrt() * 2 + 3;
    let inv = base.pow(&ctx.rational(-1, 2));
    let d = inv.sqrtdenest();
    assert_eq!(s(&d), "1/(sqrt(2) + 1)");
    assert!((inv.eval_f64().unwrap() - d.eval_f64().unwrap()).abs() < 1e-12);
    let three_halves = base.pow(&ctx.rational(3, 2));
    let d3 = three_halves.sqrtdenest();
    assert_eq!(s(&d3), "(sqrt(2) + 1)^3");
    assert!((three_halves.eval_f64().unwrap() - d3.eval_f64().unwrap()).abs() < 1e-10);
}

#[test]
fn sqrtdenest_wired_into_simplify() {
    let ctx = Context::new();
    let e = (&ctx.int(2).sqrt() * 2 + 3).sqrt();
    let simplified = e.simplify();
    assert_eq!(s(&simplified), "sqrt(2) + 1");
    let x = ctx.symbol("x");
    let f = &x * &(5 - &ctx.int(6).sqrt() * 2).sqrt();
    let g = f.simplify();
    assert!(
        !s(&g).contains("sqrt(5") && !s(&g).contains("^(1/2)"),
        "{g}"
    );
    assert_same_value(&f, &g, "simplify sqrtdenest");
}

// ═══════════════════════════════════════════════════════════════════════════
// signsimp
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn signsimp_even_power() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (&y - &x).powi(2);
    let r = e.signsimp();
    assert_eq!(s(&r), "(x - y)^2");
    assert_same_value(&e, &r, "signsimp even");
}

#[test]
fn signsimp_odd_power() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (&y - &x).powi(3);
    let r = e.signsimp();
    assert_eq!(s(&r), "-(x - y)^3");
    assert_same_value(&e, &r, "signsimp odd");
}

#[test]
fn signsimp_product_factor() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let e = &z * &(&y - &x);
    let r = e.signsimp();
    assert_eq!(s(&r), "-z*(x - y)");
    assert_same_value(&e, &r, "signsimp mul");
    // Two flips cancel.
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let f = &(-&x - &y) * &(&b - &a);
    let rf = f.signsimp();
    assert_eq!(s(&rf), "(a - b)*(x + y)");
    assert_same_value(&f, &rf, "signsimp two flips");
}

#[test]
fn signsimp_is_idempotent_and_leaves_normal_forms() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (&x - &y).powi(2) * &(&x + 1);
    assert_eq!(e.signsimp(), e);
    let f = (&y - &x).powi(5) / &(&y - &x).powi(2);
    let r = f.signsimp();
    assert_eq!(r.signsimp(), r);
    assert_same_value(&f, &r, "signsimp idempotent");
    // Fractional exponents are not touched (branch would change).
    let g = (&y - &x).sqrt();
    assert_eq!(g.signsimp(), g);
}

// ═══════════════════════════════════════════════════════════════════════════
// expand_with / ExpandOpts
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_default_matches_expand() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (&x + &y).powi(3) * &(&x - 1) + &(&x * &y).ln();
    assert_eq!(e.expand_with(&ExpandOpts::default()), e.expand());
}

#[test]
fn expand_mul_only_keeps_powers_of_sums() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &(&x + &y).powi(2) * &(&x + 1);
    let r = e.expand_with(&ExpandOpts::none().with_mul(true));
    assert_eq!(s(&r), "x*(x + y)^2 + (x + y)^2");
    assert_same_value(&e, &r, "mul only");
}

#[test]
fn expand_multinomial_only_keeps_products() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &(&x + &y).powi(2) * &(&x + 1);
    let r = e.expand_multinomial();
    assert_eq!(s(&r), "(x + 1)*(x^2 + 2*x*y + y^2)");
    assert_same_value(&e, &r, "multinomial only");
}

#[test]
fn expand_deep_false_skips_function_arguments() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &(&x + &y).powi(2).sin() + &(&x + &y).powi(2);
    let shallow = e.expand_with(&ExpandOpts::default().deep(false));
    assert_eq!(s(&shallow), "x^2 + 2*x*y + y^2 + sin((x + y)^2)");
    let deep = e.expand();
    assert!(s(&deep).contains("sin(x^2"), "{deep}");
    assert_same_value(&e, &shallow, "deep=false");
}

#[test]
fn expand_log_hint_guarded_and_forced() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (&x * &y.powi(2)).ln();
    assert_eq!(e.expand_with(&ExpandOpts::default().log(true)), e);
    let forced = e.expand_with(&ExpandOpts::default().log(true).force(true));
    assert_eq!(s(&forced), "2*ln(y) + ln(x)");
    assert_same_value_positive(&e, &forced, "log forced");
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let q = ctx.symbol_with("q", &[Assumption::Positive]);
    let g = (&p * &q).ln();
    assert_eq!(
        s(&g.expand_with(&ExpandOpts::default().log(true))),
        "ln(p) + ln(q)"
    );
}

#[test]
fn expand_trig_hint() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (&x + &y).sin();
    assert_eq!(e.expand(), e);
    let r = e.expand_with(&ExpandOpts::default().trig(true));
    assert_eq!(s(&r), "sin(x)*cos(y) + sin(y)*cos(x)");
    assert_same_value(&e, &r, "trig hint");
}

#[test]
fn expand_power_base_guard() {
    let ctx = Context::new();
    let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
    let e = (&x * &y).pow(&a);
    assert_eq!(e.expand_power_base(false), e);
    assert_eq!(e.expand(), e, "default expand must not split (x*y)^a");
    assert_eq!(s(&e.expand_power_base(true)), "x^a*y^a");
    // Integer exponent is always fine.
    let f = (&x * &y).powi(-3);
    assert_eq!(s(&f.expand_power_base(false)), "x^(-3)*y^(-3)");
    assert_same_value(&f, &f.expand_power_base(false), "power_base int");
    // Positive symbols are fine.
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let q = ctx.symbol_with("q", &[Assumption::Positive]);
    let g = (&p * &q).sqrt();
    assert_eq!(s(&g.expand_power_base(false)), "sqrt(p)*sqrt(q)");
    assert_same_value_positive(&g, &g.expand_power_base(false), "power_base positive");
}

#[test]
fn expand_power_base_unsound_case_is_blocked() {
    // For x = y = -1: sqrt(x*y) = 1 but sqrt(x)*sqrt(y) = i*i = -1.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (&x * &y).sqrt();
    assert_eq!(e.expand(), e);
    assert_eq!(e.expand_power_base(false), e);
}

#[test]
fn expand_power_exp_exp_split() {
    let ctx = Context::new();
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let e = (&a + &b * 2).exp();
    let r = e.expand_power_exp(false);
    assert_eq!(s(&r), "exp(a)*exp(2*b)");
    assert_same_value(&e, &r, "exp split");
    // Default expand splits too (SymPy parity); powsimp recombines.
    assert_eq!(e.expand(), r);
    assert_eq!(r.simplify_powers(), e);
}

#[test]
fn expand_power_exp_integer_summands_via_assumptions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = ctx.symbol_with("m", &[Assumption::Integer]);
    let n = ctx.symbol_with("n", &[Assumption::Integer]);
    // x^(m+n) → x^m * x^n is valid, but canonicalisation re-merges; the
    // observable guarantee is that the result is value-equal.
    let e = x.pow(&(&m + &n));
    let r = e.expand_power_exp(false);
    assert_same_value(&e, &r, "power_exp int");
}

// ═══════════════════════════════════════════════════════════════════════════
// collect / rcollect / collect_const
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn collect_symbolic_exponents() {
    let ctx = Context::new();
    let (x, y, z, a) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("z"),
        ctx.symbol("a"),
    );
    let e = &y * &x.pow(&a) + &z * &x.pow(&a) + &x.powi(2) * 3;
    let c = e.collect(&x);
    assert_eq!(s(&c), "3*x^2 + x^a*(y + z)");
    assert_same_value_positive(&e, &c, "collect symbolic");
}

#[test]
fn collect_negative_exponents() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let e = &y / &x + &z / &x + &y;
    let c = e.collect(&x);
    assert_eq!(s(&c), "(y + z)/x + y");
    assert_same_value(&e, &c, "collect neg exp");
}

#[test]
fn collect_polynomial_unchanged_path() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &x * &y + &x.powi(2) + &y;
    assert_eq!(s(&e.collect(&x)), "x^2 + x*y + y");
}

#[test]
fn rcollect_two_variables() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let e = &x * &y + &x * &z + &y.powi(2) * &x + &y.powi(2) * &z;
    let c = e.rcollect(&[&x, &y]);
    assert_eq!(s(&c), "z*y^2 + x*(y^2 + y + z)");
    assert_same_value(&e, &c, "rcollect");
    // Recurses into nested sums.
    let f = (&x * &y + &x * &z).sin();
    let cf = f.rcollect(&[&x]);
    assert_eq!(s(&cf), "sin(x*(y + z))");
}

#[test]
fn collect_const_inside_products_and_powers() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let e = &z * &(&x * 2 + &y * 4);
    let c = e.collect_const();
    assert_eq!(s(&c), "2*z*(x + 2*y)");
    assert_same_value(&e, &c, "collect_const mul");
    let f = (&x * 2 + &y * 4).powi(2);
    let cf = f.collect_const();
    assert_eq!(s(&cf), "4*(x + 2*y)^2");
    assert_same_value(&f, &cf, "collect_const pow");
    // Top-level sum: not representable, unchanged; factor_terms gives the pair.
    let g = &x * 2 + &y * 4;
    assert_eq!(g.collect_const(), g);
    let (gcd, inner) = g.factor_terms();
    assert_eq!(s(&gcd), "2");
    assert_eq!(s(&inner), "x + 2*y");
}

// ═══════════════════════════════════════════════════════════════════════════
// trigsimp identities
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trigsimp_sin_sum_and_difference() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &x.sin() * &y.cos() + &x.cos() * &y.sin();
    let r = e.simplify_trig();
    assert_eq!(s(&r), "sin(x + y)");
    assert_same_value(&e, &r, "sin_add");
    let f = &x.sin() * &y.cos() - &x.cos() * &y.sin();
    let rf = f.simplify_trig();
    assert_eq!(s(&rf), "sin(x - y)");
    assert_same_value(&f, &rf, "sin_sub");
}

#[test]
fn trigsimp_cos_sum_and_difference() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &x.cos() * &y.cos() - &x.sin() * &y.sin();
    let r = e.simplify_trig();
    assert_eq!(s(&r), "cos(x + y)");
    assert_same_value(&e, &r, "cos_add");
    let f = &x.cos() * &y.cos() + &x.sin() * &y.sin();
    let rf = f.simplify_trig();
    assert_eq!(s(&rf), "cos(x - y)");
    assert_same_value(&f, &rf, "cos_sub");
}

#[test]
fn trigsimp_double_angle_forms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        (&x.cos().powi(2) - &x.sin().powi(2), "cos(2*x)"),
        (1 - &x.sin().powi(2) * 2, "cos(2*x)"),
        (&x.cos().powi(2) * 2 - 1, "cos(2*x)"),
        (&x.sin() * &x.cos() * 2, "sin(2*x)"),
    ];
    for (e, expected) in cases {
        let r = e.simplify_trig();
        assert_eq!(s(&r), expected, "from {e}");
        assert_same_value(&e, &r, expected);
    }
}

#[test]
fn trigsimp_sin_over_cos_and_pythagorean_variants() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin() / &x.cos();
    assert_eq!(s(&e.simplify_trig()), "tan(x)");
    let f = 1 - &x.cos().powi(2);
    let rf = f.simplify_trig();
    assert_eq!(s(&rf), "sin(x)^2");
    assert_same_value(&f, &rf, "1 - cos^2");
}

#[test]
fn trigsimp_hyperbolic_identities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sinh() * &x.cosh() * 2;
    let r = e.simplify();
    assert_eq!(s(&r), "sinh(2*x)");
    assert_same_value(&e, &r, "sinh_double");
    let f = &x.cosh().powi(2) + &x.sinh().powi(2);
    let rf = f.simplify();
    assert_eq!(s(&rf), "cosh(2*x)");
    assert_same_value(&f, &rf, "cosh_double");
    let g = &x.cosh().powi(2) - &x.sinh().powi(2);
    assert_eq!(s(&g.simplify()), "1");
}

#[test]
fn trigsimp_identities_inside_larger_expression() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let e = &z + &x.sin() * &y.cos() + &x.cos() * &y.sin() + &z.powi(2);
    let r = e.simplify();
    assert_eq!(s(&r), "z^2 + z + sin(x + y)");
    assert_same_value(&e, &r, "sin_add embedded");
}

#[test]
fn trigsimp_does_not_fire_on_mismatched_arguments() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let e = &x.sin() * &y.cos() + &x.cos() * &z.sin();
    let r = e.simplify_trig();
    assert_same_value(&e, &r, "mismatch");
    assert!(
        !s(&r).contains("sin(x + y)") && !s(&r).contains("sin(x + z)"),
        "{r}"
    );
}

#[test]
fn trigsimp_value_preservation_sweep() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let exprs = [
        &x.sin() * &y.cos() * 3 + &x.cos() * &y.sin() * 3,
        &x.cos().powi(2) * 5 - &x.sin().powi(2) * 5 + &y,
        &x.sin() * &x.cos() * 6 + 1,
        &x.cosh().powi(2) * 2 + &x.sinh().powi(2) * 2,
        (&x + &y).sin().powi(2) + (&x + &y).cos().powi(2),
        &(&x * 2).sin() / &(&x * 2).cos() + &x.sinh() / &x.cosh(),
    ];
    for e in &exprs {
        let r = e.simplify();
        assert_same_value(e, &r, &format!("simplify {e}"));
        let t = e.simplify_trig();
        assert_same_value(e, &t, &format!("simplify_trig {e}"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// powdenest
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn powdenest_nested_symbolic_needs_force_or_assumptions() {
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    let e = x.pow(&a).pow(&b);
    assert_eq!(e.powdenest(false), e);
    assert_eq!(s(&e.powdenest(true)), "x^(a*b)");
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!(s(&p.pow(&r).pow(&b).powdenest(false)), "p^(b*r)");
}

#[test]
fn powdenest_integer_outer_exponent_always_valid() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let e = x.pow(&a).powi(3);
    let r = e.powdenest(false);
    assert_eq!(s(&r), "x^(3*a)");
    assert_same_value_positive(&e, &r, "powdenest int");
    // Fractional inner and outer: (x^(1/2))^3 = x^(3/2) for all x? For x < 0
    // both sides are (√x)^3 — an integer power, so exact.
    let f = x.sqrt().powi(3);
    let rf = f.powdenest(false);
    assert_eq!(s(&rf), "x^(3/2)");
}

#[test]
fn powdenest_sqrt_of_square() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(2).sqrt();
    // 0.23: unassumed x may be complex (√(i²) = i ≠ |i|): only a known-real
    // argument gives |x|.
    assert_eq!(e.powdenest(false), e);
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!(s(&r.powi(2).sqrt().powdenest(false)), "abs(r)");
    assert_eq!(s(&e.powdenest(true)), "x");
    let p = ctx.symbol_with("p", &[Assumption::NonNegative]);
    assert_eq!(s(&p.powi(2).sqrt().powdenest(false)), "p");
    let i = ctx.i_unit();
    let z = ctx.symbol_with("z", &[Assumption::Imaginary]);
    // Known non-real: sqrt(z²) is not |z| — untouched.
    assert_eq!(z.powi(2).sqrt().powdenest(false), z.powi(2).sqrt());
    let _ = i;
}

#[test]
fn powdenest_product_base_via_assumptions() {
    let ctx = Context::new();
    let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a"));
    let e = (&x * &y).pow(&a);
    assert_eq!(e.powdenest(false), e);
    assert_eq!(s(&e.powdenest(true)), "x^a*y^a");
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let q = ctx.symbol_with("q", &[Assumption::Positive]);
    assert_eq!(s(&(&p * &q).pow(&a).powdenest(false)), "p^a*q^a");
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumption-aware logarithms and ln(exp(x))
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_combine_with_guard_vs_force() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &x.ln() + &y.ln() * 2;
    assert_eq!(e.log_combine_with(false), e);
    let forced = e.log_combine_with(true);
    assert_eq!(s(&forced), "ln(x*y^2)");
    // 0.23: the default is the guarded form.
    assert_eq!(e.log_combine(), e);
    assert_same_value_positive(&e, &forced, "log_combine force");
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let q = ctx.symbol_with("q", &[Assumption::Positive]);
    let g = &p.ln() + &q.ln() * 2;
    assert_eq!(s(&g.log_combine_with(false)), "ln(p*q^2)");
}

#[test]
fn log_combine_with_guard_mixes_known_and_unknown_terms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let q = ctx.symbol_with("q", &[Assumption::Positive]);
    let e = &p.ln() + &q.ln() + &x.ln();
    let r = e.log_combine_with(false);
    // 0.23: logarithms of positive arguments also absorb one other
    // logarithm — arg(p·q) = 0, so ln(pq) + ln x = ln(pqx) for every x.
    assert_eq!(s(&r), "ln(p*q*x)");
    let y = ctx.symbol("y");
    let two_unknown = &e + &y.ln();
    assert_eq!(
        s(&two_unknown.log_combine_with(false)),
        "ln(x) + ln(y) + ln(p*q)"
    );
}

#[test]
fn expand_log_with_guard_vs_force() {
    let ctx = Context::new();
    let (x, n) = (ctx.symbol("x"), ctx.symbol("n"));
    let e = x.pow(&n).ln();
    assert_eq!(e.expand_log_with(false), e);
    assert_eq!(s(&e.expand_log_with(true)), "n*ln(x)");
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!(s(&p.pow(&r).ln().expand_log_with(false)), "r*ln(p)");
    // Positive base, non-real exponent: blocked.
    let z = ctx.symbol_with("z", &[Assumption::Imaginary]);
    assert_eq!(p.pow(&z).ln().expand_log_with(false), p.pow(&z).ln());
}

#[test]
fn ln_exp_blocked_for_known_non_real_argument() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let z = ctx.symbol_with("z", &[Assumption::Imaginary]);
    // ln(exp(z)) with imaginary z: the principal log may differ by 2πi·k.
    let e = z.exp().ln();
    assert_eq!(e.simplify(), e);
    // Real argument: fires.
    assert_eq!(x.exp().ln().simplify(), x);
    // Compound imaginary argument i·x (x real) is known non-real → blocked.
    let f = (&ctx.i_unit() * &x).exp().ln();
    assert_ne!(
        f.simplify(),
        &ctx.i_unit() * &x,
        "ln(exp(i x)) must not become i x"
    );
}

#[test]
fn exp_ln_fires_for_any_argument() {
    let ctx = Context::new();
    let z = ctx.symbol_with("z", &[Assumption::Imaginary]);
    assert_eq!(z.ln().exp().simplify(), z);
}

// ═══════════════════════════════════════════════════════════════════════════
// nsimplify with constants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nsimplify_rational_multiple_of_constant() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let e = ctx.e();
    let v = ctx.rational(471238898038469, 100000000000000); // 3π/2
    assert_eq!(s(&v.nsimplify_with_constants(&[&pi, &e], 1e-9)), "3/2*pi");
    let w = ctx.rational(-5436563656918091, 1000000000000000); // -2e
    assert_eq!(s(&w.nsimplify_with_constants(&[&pi, &e], 1e-9)), "-2*E");
}

#[test]
fn nsimplify_rational_offset_of_constant() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let v = ctx.rational(3641592653589793, 1000000000000000); // π + 1/2
    assert_eq!(s(&v.nsimplify_with_constants(&[&pi], 1e-9)), "1/2 + pi");
}

#[test]
fn nsimplify_rational_power_of_constant() {
    let ctx = Context::new();
    let e = ctx.e();
    let v = ctx.rational(1648721270700128, 1000000000000000); // e^(1/2)
    assert_eq!(s(&v.nsimplify_with_constants(&[&e], 1e-9)), "exp(1/2)");
    let two = ctx.int(2);
    let w = ctx.rational(1259921049894873, 1000000000000000); // 2^(1/3)
    assert_eq!(s(&w.nsimplify_with_constants(&[&two], 1e-9)), "cbrt(2)");
}

#[test]
fn nsimplify_default_table() {
    let ctx = Context::new();
    let sqrt3 = ctx.rational(1732050807568877, 1000000000000000);
    assert_eq!(s(&sqrt3.nsimplify(1e-9)), "sqrt(3)");
    let phi2 = ctx.rational(3236067977499790, 1000000000000000); // 2φ
    assert_eq!(s(&phi2.nsimplify(1e-9)), "2*GoldenRatio");
    let ln2 = ctx.rational(6931471805599453, 10000000000000000);
    assert_eq!(s(&ln2.nsimplify(1e-9)), "ln(2)");
    let sqrt5_half = ctx.rational(1118033988749895, 1000000000000000); // √5/2
    assert_eq!(s(&sqrt5_half.nsimplify(1e-9)), "1/2*sqrt(5)");
}

#[test]
fn nsimplify_plain_rational_and_no_match() {
    let ctx = Context::new();
    let third = ctx.rational(333333333, 1000000000);
    assert_eq!(s(&third.nsimplify(1e-6)), "1/3");
    // Nothing recognisable at tight tolerance: unchanged.
    let odd = ctx.rational(1234567891, 1000000000);
    assert_eq!(odd.nsimplify(1e-12), odd);
    // Symbolic input: unchanged.
    let x = ctx.symbol("x");
    assert_eq!((&x + 1).nsimplify(1e-9), &x + 1);
}

#[test]
fn nsimplify_recognised_values_are_numerically_accurate() {
    let ctx = Context::new();
    let inputs = [
        ctx.rational(628318530717958, 100000000000000),
        ctx.rational(1414213562373095, 1000000000000000),
        ctx.rational(3718281828459045, 1000000000000000),
    ];
    for v in &inputs {
        let r = v.nsimplify(1e-9);
        assert_ne!(&r, v, "{v} should be recognised");
        let (a, b) = (v.eval_f64().unwrap(), r.eval_f64().unwrap());
        assert!((a - b).abs() < 1e-9, "{v} → {r}: {a} vs {b}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// separate_vars
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn separate_vars_additive_groups_terms() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &x.sin() + &x.powi(2) + &y.exp() + &x * &y + 3;
    let groups = e.separate_vars_additive(&[&x, &y]);
    assert_eq!(groups.len(), 4);
    let find = |n: usize| {
        groups
            .iter()
            .filter(|(d, _)| d.len() == n)
            .collect::<Vec<_>>()
    };
    assert_eq!(s(&find(0)[0].1), "3");
    assert_eq!(s(&find(2)[0].1), "x*y");
    let x_only = groups
        .iter()
        .find(|(d, _)| d.len() == 1 && d[0] == x)
        .unwrap();
    assert_eq!(s(&x_only.1), "x^2 + sin(x)");
    // The groups sum back to the original.
    let total = groups.iter().fold(ctx.int(0), |acc, (_, g)| acc + g);
    assert_eq!(total, e);
}

#[test]
fn separate_vars_additive_non_sum() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &x * &y;
    let groups = e.separate_vars_additive(&[&x, &y]);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].0.len(), 2);
}

#[test]
fn separate_vars_dict_product() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &x.sin() * &y.powi(2) * 2 * &x;
    let d = e.separate_vars_dict(&[&x, &y]).unwrap();
    assert_eq!(d.len(), 2);
    assert_eq!(d[0].0, x);
    assert_eq!(s(&d[0].1), "2*x*sin(x)");
    assert_eq!(d[1].0, y);
    assert_eq!(s(&d[1].1), "y^2");
    let product = &d[0].1 * &d[1].1;
    assert_eq!(product, e);
}

#[test]
fn separate_vars_dict_not_separable() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    assert!((&x + &y).separate_vars_dict(&[&x, &y]).is_none());
    assert!((&x * &y).sin().separate_vars_dict(&[&x, &y]).is_none());
    assert!(x.separate_vars_dict(&[]).is_none());
}

#[test]
fn separate_vars_dict_factors_sums_first() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // x*y + x*y^2 = x*(y + y^2)
    let e = &x * &y + &x * &y.powi(2);
    let d = e.separate_vars_dict(&[&x, &y]).unwrap();
    assert_eq!(s(&d[0].1), "x");
    assert_eq!(s(&d[1].1), "y*(y + 1)");
    assert_same_value(&e, &(&d[0].1 * &d[1].1), "separate dict factored");
}

#[test]
fn separate_vars_dict_missing_variable_gets_one() {
    let ctx = Context::new();
    let (x, y, t) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("t"));
    let e = &x.exp() * 3;
    let d = e.separate_vars_dict(&[&x, &y, &t]).unwrap();
    assert_eq!(s(&d[0].1), "3*exp(x)");
    assert_eq!(s(&d[1].1), "1");
    assert_eq!(s(&d[2].1), "1");
}
