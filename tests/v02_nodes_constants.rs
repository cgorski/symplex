//! 0.2 named constants: `EulerGamma` (γ), `Catalan` (G), `GoldenRatio` (φ)
//! and the `Context::complex_infinity()` constructor.

use symplex::prelude::*;

/// γ to 250 decimal places (OEIS A001620).
const EULER_GAMMA_250: &str = "0.\
5772156649015328606065120900824024310421593359399235988057672348848677\
2677766467093694706329174674951463144724980708248096050401448654283622\
4173997644923536253500333742937337737673942792595258247094916008735203\
9481656708532331517766115286211995015079";

/// G to 105 decimal places (OEIS A006752).
const CATALAN_105: &str = "0.\
9159655941772190150546035149323841107741493742816721342664981196217630\
19776254769479356512926115106248574";

/// φ = (1+√5)/2 to 100 decimal places (OEIS A001622).
const GOLDEN_RATIO_100: &str = "1.\
6180339887498948482045868343656381177203091798057628621354486227052604\
628189024497072072041893911374";

fn digits_agree(computed: &str, reference: &str, n: usize) {
    let c: String = computed.chars().take(n).collect();
    let r: String = reference.chars().take(n).collect();
    assert_eq!(
        c, r,
        "first {n} characters differ:\n  got  {computed}\n  want {reference}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction, display, LaTeX, parse
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn constants_are_singletons() {
    let ctx = Context::new();
    assert_eq!(ctx.euler_gamma(), ctx.euler_gamma());
    assert_eq!(ctx.catalan(), ctx.catalan());
    assert_eq!(ctx.golden_ratio(), ctx.golden_ratio());
    assert_ne!(ctx.euler_gamma(), ctx.catalan());
    assert_ne!(ctx.golden_ratio(), ctx.pi());
    assert_eq!(ctx.complex_infinity(), ctx.complex_infinity());
    assert_ne!(ctx.complex_infinity(), ctx.infinity());
}

#[test]
fn display_and_latex() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.euler_gamma()), "EulerGamma");
    assert_eq!(format!("{}", ctx.catalan()), "Catalan");
    assert_eq!(format!("{}", ctx.golden_ratio()), "GoldenRatio");
    assert_eq!(format!("{}", ctx.complex_infinity()), "zoo");
    assert_eq!(ctx.euler_gamma().to_latex(), r"\gamma");
    assert_eq!(ctx.catalan().to_latex(), "G");
    assert_eq!(ctx.golden_ratio().to_latex(), r"\phi");
    assert_eq!(ctx.euler_gamma().pretty().trim(), "γ");
    assert_eq!(ctx.golden_ratio().pretty().trim(), "φ");
    assert_eq!(ctx.euler_gamma().pretty_ascii().trim(), "EulerGamma");
    // Constants sort like other constants (after symbols) inside sums.
    let x = ctx.symbol("x");
    assert_eq!(format!("{}", &x + &ctx.euler_gamma()), "x + EulerGamma");
    assert_eq!(format!("{}", &ctx.golden_ratio() * &x), "x*GoldenRatio");
}

#[test]
fn parse_names() {
    let ctx = Context::new();
    assert_eq!(ctx.parse("EulerGamma").unwrap(), ctx.euler_gamma());
    assert_eq!(ctx.parse("euler_gamma").unwrap(), ctx.euler_gamma());
    assert_eq!(ctx.parse("Catalan").unwrap(), ctx.catalan());
    assert_eq!(ctx.parse("GoldenRatio").unwrap(), ctx.golden_ratio());
    assert_eq!(ctx.parse("golden_ratio").unwrap(), ctx.golden_ratio());
    assert_eq!(ctx.parse("zoo").unwrap(), ctx.complex_infinity());
    // `phi` and `gamma` remain ordinary symbols / the Gamma function.
    assert_eq!(ctx.parse("phi").unwrap(), ctx.symbol("phi"));
    let e = ctx.parse("2*GoldenRatio + EulerGamma^2").unwrap();
    assert_eq!(
        e,
        &(&ctx.int(2) * &ctx.golden_ratio()) + &ctx.euler_gamma().powi(2)
    );
}

#[test]
fn tree_json_round_trip() {
    let ctx = Context::new();
    let e = &(&ctx.euler_gamma() + &ctx.catalan()) * &ctx.golden_ratio();
    let json = serde_json::to_string(&e.to_tree()).unwrap();
    assert!(
        json.contains("EulerGamma") && json.contains("Catalan") && json.contains("GoldenRatio")
    );
    let tree: symplex::tree::ExprTree = serde_json::from_str(&json).unwrap();
    assert_eq!(ctx.from_tree(&tree), e);
}

#[test]
fn compact_transfers_constants() {
    let ctx = Context::new();
    let e = &ctx.euler_gamma() + &(&ctx.catalan() * &ctx.golden_ratio());
    let (ctx2, roots) = ctx.compact(std::slice::from_ref(&e));
    assert_eq!(format!("{}", roots[0]), format!("{e}"));
    assert_eq!(
        roots[0],
        &ctx2.euler_gamma() + &(&ctx2.catalan() * &ctx2.golden_ratio())
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumptions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn assumptions() {
    let ctx = Context::new();
    for c in [ctx.euler_gamma(), ctx.catalan(), ctx.golden_ratio()] {
        assert_eq!(c.is_positive(), Some(true), "{c}");
        assert_eq!(c.is_real(), Some(true), "{c}");
        assert_eq!(c.is_finite(), Some(true), "{c}");
        assert_eq!(c.is_integer(), Some(false), "{c}");
        assert_eq!(c.is_zero(), Some(false), "{c}");
        assert_eq!(c.is_imaginary(), Some(false), "{c}");
    }
    // φ is algebraic and irrational.
    let phi = ctx.golden_ratio();
    assert_eq!(phi.is_algebraic(), Some(true));
    assert_eq!(phi.is_irrational(), Some(true));
    assert_eq!(phi.is_rational(), Some(false));
    assert_eq!(phi.is_transcendental(), Some(false));
    // γ and G: (ir)rationality is an open problem → unknown.
    for c in [ctx.euler_gamma(), ctx.catalan()] {
        assert_eq!(c.is_rational(), None, "{c}");
        assert_eq!(c.is_irrational(), None, "{c}");
        assert_eq!(c.is_algebraic(), None, "{c}");
        assert_eq!(c.is_transcendental(), None, "{c}");
    }
    // Complex infinity is not finite and not real.
    assert_eq!(ctx.complex_infinity().is_finite(), Some(false));
}

// ═══════════════════════════════════════════════════════════════════════════
// Calculus
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn constants_differentiate_to_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for c in [ctx.euler_gamma(), ctx.catalan(), ctx.golden_ratio()] {
        assert!(c.diff(&x).is_zero_structural());
        // d/dx (c·x²) = 2 c x
        let d = (&c * &x.powi(2)).diff(&x);
        assert_eq!(d, &(&ctx.int(2) * &c) * &x);
    }
    let e = (&ctx.golden_ratio() * &x).exp();
    assert_eq!(e.diff(&x), &ctx.golden_ratio() * &e);
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn euler_gamma_to_250_digits() {
    let ctx = Context::new();
    let g = ctx.euler_gamma();
    for digits in [15u32, 50, 100, 200, 250] {
        let s = g.eval_decimal(digits).unwrap();
        // `eval_decimal(d)` yields d significant digits with the last one
        // *rounded* (and trailing zeros trimmed), while the reference is a
        // truncated expansion — so compare all but the final digit.
        digits_agree(&s, EULER_GAMMA_250, digits as usize);
    }
    let v = g.eval_f64().unwrap();
    assert!((v - 0.577_215_664_901_532_9).abs() < 1e-15);
}

#[test]
fn catalan_to_105_digits() {
    let ctx = Context::new();
    let g = ctx.catalan();
    for digits in [15u32, 50, 100, 105] {
        let s = g.eval_decimal(digits).unwrap();
        digits_agree(&s, CATALAN_105, digits as usize);
    }
    let v = g.eval_f64().unwrap();
    assert!((v - 0.915_965_594_177_219).abs() < 1e-15);
}

#[test]
fn golden_ratio_to_100_digits() {
    let ctx = Context::new();
    let phi = ctx.golden_ratio();
    for digits in [15u32, 50, 100] {
        let s = phi.eval_decimal(digits).unwrap();
        digits_agree(&s, GOLDEN_RATIO_100, digits as usize);
    }
    // φ² = φ + 1 numerically to high precision
    let lhs = phi.powi(2).eval_decimal(60).unwrap();
    let rhs = (&phi + 1).eval_decimal(60).unwrap();
    assert_eq!(lhs, rhs);
    let v = phi.eval_f64().unwrap();
    assert!((v - 1.618_033_988_749_895).abs() < 1e-15);
}

#[test]
fn constants_inside_expressions() {
    let ctx = Context::new();
    // ψ(1) = −γ  and  ψ(2) = 1 − γ  (exact folding through eval)
    let psi1 = ctx.int(1).digamma().eval();
    assert_eq!(psi1, -&ctx.euler_gamma());
    let psi2 = ctx.int(2).digamma().eval();
    assert_eq!(psi2, &ctx.int(1) - &ctx.euler_gamma());
    let v = psi2.eval_f64().unwrap();
    assert!((v - 0.422_784_335_098_467_1).abs() < 1e-14);
    // γ + G + φ numerically
    let s = (&(&ctx.euler_gamma() + &ctx.catalan()) + &ctx.golden_ratio())
        .eval_f64()
        .unwrap();
    assert!(
        (s - (0.577_215_664_901_532_9 + 0.915_965_594_177_219 + 1.618_033_988_749_895)).abs()
            < 1e-14
    );
}

#[test]
fn complex_infinity_from_poles() {
    let ctx = Context::new();
    let zoo = ctx.complex_infinity();
    assert_eq!(ctx.int(1).zeta(), zoo);
    assert_eq!(ctx.int(0).digamma().eval(), zoo);
    assert_eq!(ctx.int(-2).polygamma(&ctx.int(3)), zoo);
    assert!(zoo.eval_f64().is_err());
}

#[test]
fn codegen_and_lambdify_emit_literals() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&ctx.euler_gamma() * &x) + &(&ctx.catalan() + &ctx.golden_ratio());
    let f = e.compile(&["x"]).expect("constants should be lambdifiable");
    let v = f(&[2.0]);
    let expected = 0.577_215_664_901_532_9 * 2.0 + 0.915_965_594_177_219 + 1.618_033_988_749_895;
    assert!((v - expected).abs() < 1e-12);
    let src = e.to_rust_fn("f", &["x"]).unwrap();
    assert!(src.contains("0.5772156649015329"), "{src}");
    assert!(src.contains("0.915965594177219"), "{src}");
    assert!(src.contains("1.618033988749895"), "{src}");
}
