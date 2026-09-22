//! symplex 0.12 — numeric_summation.  Reference values cite SymPy 1.14 / mpmath 1.3
//! (`symplex/.venv/bin/python`).
//!
//! * `erfinv` / `erfcinv` in `Ex::compile` (and the Rust code generator);
//!   references are `mpmath.erfinv(mpf(x))` at dps 60, evaluated at the exact
//!   double nearest each literal, and `erfcinv` solved from `erfc(z) = y` the
//!   same way.
//! * Summation identities surfaced by the 0.11 statistics work: the
//!   negative-binomial series `Σ C(k+c,k) xᵏ = (1−x)^{−(c+1)}`, the binomial
//!   theorem with a symbolic exponent offset `(1−p)^{n−k}`, Poisson /
//!   Geometric totals and moments.  Sums the engine still leaves formal are
//!   asserted as such with `has_unevaluated()`.

// Reference values are quoted with the full digits printed by mpmath so they
// can be re-checked against the source; the compiler rounds them for us.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

/// Relative error of `got` against `want` (absolute when `want == 0`).
fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        rel(got, want) <= tol,
        "{what}: got {got:.17e}, want {want:.17e}, rel err {:.3e} > {tol:.0e}",
        rel(got, want)
    );
}

/// `a` and `b` agree as polynomials (their difference expands to 0).
fn assert_poly_eq(a: &Ex, b: &Ex, label: &str) {
    let d = (a - b).expand().simplify();
    assert!(
        d.is_zero_structural(),
        "{label}: `{a}` ≠ `{b}` (difference `{d}`)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 1 — erfinv / erfcinv through `Ex::compile`
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compiled_erfinv_matches_mpmath() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.erfinv().compile(&["x"]).unwrap();
    // mpmath (dps 60): erfinv(mpf(x)) at the exact double nearest each literal.
    let cases: [(f64, f64); 12] = [
        (0.1, 0.088855990494257691974),
        (0.3, 0.27246271472675434502),
        (0.46875, 0.44271885732435436322),
        (0.5, 0.47693627620446987338),
        (0.6, 0.59511608144999482198),
        (0.75, 0.81341984759761854169),
        (0.9, 1.1630871536766741628),
        (0.99, 1.8213863677184494559),
        (0.999, 2.3267537655135244939),
        (0.99999, 3.123413274341570864),
        (1.0 - 1e-9, 4.3200053881053620459),
        (1.0 - 1e-12, 5.0420318985726961301),
    ];
    for &(xv, want) in &cases {
        assert_rel(f(&[xv]), want, 1e-15, &format!("erfinv({xv})"));
        assert_rel(f(&[-xv]), -want, 1e-15, &format!("erfinv(-{xv})"));
    }
    // Deep tail: 1 − 1e−15 and 1 − 2⁻⁵² (mpmath erfinv(1 - 2**-52)).
    assert_rel(
        f(&[1.0 - 1e-15]),
        5.6759157397447131788,
        1e-14,
        "erfinv(1-1e-15)",
    );
    assert_rel(
        f(&[1.0 - f64::EPSILON]),
        5.8050186831934533002,
        1e-14,
        "erfinv(1-2^-52)",
    );
    // Tiny arguments: mpmath erfinv(1e-300) = 8.8622692545275803586e-301 (≈ (√π/2)·x).
    assert_rel(
        f(&[1e-300]),
        8.8622692545275803586e-301,
        1e-15,
        "erfinv(1e-300)",
    );
    assert_eq!(f(&[0.0]), 0.0);
    // Boundaries and domain.
    assert_eq!(f(&[1.0]), f64::INFINITY);
    assert_eq!(f(&[-1.0]), f64::NEG_INFINITY);
    assert!(f(&[1.5]).is_nan());
    assert!(f(&[-1.0000001]).is_nan());
    assert!(f(&[f64::NAN]).is_nan());
}

#[test]
fn compiled_erfcinv_matches_mpmath() {
    let ctx = Context::new();
    let y = ctx.symbol("y");
    let f = y.erfcinv().compile(&["y"]).unwrap();
    // mpmath (dps 60): z solving erfc(z) = mpf(y) at the exact double nearest y.
    let cases: [(f64, f64); 10] = [
        (0.5, 0.47693627620446987338),
        (1.5, -0.47693627620446987338),
        (0.1, 1.1630871536766740677),
        (1.9, -1.1630871536766737823),
        (0.999, 0.00088622715746655289169),
        (1e-5, 3.1234132743408750177),
        (1e-10, 4.5728249673894852748),
        (1e-20, 6.6015806223551425656),
        (1e-100, 15.065574702592645704),
        (1e-300, 26.209469960516123886),
    ];
    for &(yv, want) in &cases {
        assert_rel(f(&[yv]), want, 1e-15, &format!("erfcinv({yv})"));
    }
    // Where erfc itself underflows (subnormal y) the inverse keeps relative accuracy.
    assert_rel(
        f(&[5e-324]),
        27.213293210812948815,
        1e-15,
        "erfcinv(5e-324)",
    );
    assert_eq!(f(&[1.0]), 0.0);
    assert_eq!(f(&[0.0]), f64::INFINITY);
    assert_eq!(f(&[2.0]), f64::NEG_INFINITY);
    assert!(f(&[-0.5]).is_nan());
    assert!(f(&[2.5]).is_nan());
}

#[test]
fn compiled_normal_quantile() {
    // Normal(μ, σ) quantile μ + σ√2·erfinv(2p − 1).
    // mpmath (dps 40): sqrt(2)*erfinv(2*mpf(0.975) - 1) = 1.9599639845400538556,
    //                  1 + 2*sqrt(2)*erfinv(2*mpf(0.975) - 1) = 4.9199279690801077112,
    //                  sqrt(2)*erfinv(2*mpf(0.999) - 1) = 3.0902323061678132778.
    let ctx = Context::new();
    let p = ctx.symbol("p");
    let mu = ctx.symbol("mu");
    let sigma = ctx.symbol("sigma");
    let q = &mu + &sigma * ctx.int(2).sqrt() * (ctx.int(2) * &p - 1).erfinv();
    let f = q.compile(&["mu", "sigma", "p"]).unwrap();
    assert_eq!(f.arity(), 3);
    assert_rel(
        f(&[0.0, 1.0, 0.975]),
        1.9599639845400538556,
        1e-15,
        "z_{0.975}",
    );
    assert_rel(
        f(&[1.0, 2.0, 0.975]),
        4.9199279690801077112,
        1e-15,
        "N(1,2) quantile",
    );
    assert_rel(
        f(&[0.0, 1.0, 0.999]),
        3.0902323061678132778,
        1e-15,
        "z_{0.999}",
    );
    assert_eq!(f(&[0.0, 1.0, 0.5]), 0.0);
    // Symmetry and the boundaries p = 0, 1.
    assert_eq!(f(&[0.0, 1.0, 0.025]), -f(&[0.0, 1.0, 0.975]));
    assert_eq!(f(&[0.0, 1.0, 1.0]), f64::INFINITY);
    assert_eq!(f(&[0.0, 1.0, 0.0]), f64::NEG_INFINITY);
    // LogNormal quantile exp(μ + σ√2·erfinv(2p − 1)).
    let g = q.exp().compile(&["mu", "sigma", "p"]).unwrap();
    assert_rel(
        g(&[1.0, 2.0, 0.975]),
        4.9199279690801077112f64.exp(),
        1e-15,
        "LogNormal(1,2) quantile",
    );
}

#[test]
fn erfinv_rust_codegen_uses_the_shared_runtime() {
    let ctx = Context::new();
    let p = ctx.symbol("p");
    let q = ctx.int(2).sqrt() * (ctx.int(2) * &p - 1).erfinv();
    let code = q.to_rust_fn("quantile", &["p"]).unwrap();
    assert!(code.contains("symplex_rt::erfinv("), "{code}");
    assert!(
        code.contains("pub fn erfinv("),
        "runtime section not embedded:\n{code}"
    );
    assert!(
        code.contains("fn erfcx_large("),
        "erf section dependency missing:\n{code}"
    );
    // erfcinv likewise.
    let code = p.erfcinv().to_rust_fn("f", &["p"]).unwrap();
    assert!(code.contains("symplex_rt::erfcinv("), "{code}");
    // The C backend mirrors the runtime with `symplex_erfinv` (0.21).
    let code = p.erfinv().to_c_fn("f", &["p"]).unwrap();
    assert!(code.contains("symplex_erfinv("), "{code}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 2a — negative-binomial series Σ C(k+c, k) xᵏ
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn negative_binomial_series_numeric_c() {
    // SymPy: summation(binomial(k+3,k)*Rational(1,3)**k, (k,0,oo)) = 81/16
    //        summation(k*binomial(k+3,k)*Rational(1,3)**k, (k,0,oo)) = 81/8
    //        summation(k**2*binomial(k+3,k)*Rational(1,3)**k, (k,0,oo)) = 567/16
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let third = ctx.rational(1, 3);
    let bin = (&k + ctx.int(3)).binomial(&k);
    let s = (&bin * third.pow(&k)).summation(&k, &zero, &inf);
    assert_eq!(s, ctx.rational(81, 16), "{s}");
    // C(k+3, 3) is the same coefficient.
    let bin3 = (&k + ctx.int(3)).binomial(&ctx.int(3));
    let s = (&bin3 * third.pow(&k)).summation(&k, &zero, &inf);
    assert_eq!(s, ctx.rational(81, 16), "{s}");
    // Polynomial multipliers go through the Euler operator P(x d/dx).
    let s = (&k * &bin * third.pow(&k)).summation(&k, &zero, &inf);
    assert_eq!(s, ctx.rational(81, 8), "{s}");
    let s = (k.powi(2) * &bin * third.pow(&k)).summation(&k, &zero, &inf);
    assert_eq!(s, ctx.rational(567, 16), "{s}");
}

#[test]
fn negative_binomial_pmf_total_mass_and_moments() {
    // NegativeBinomial(r = 3, p = 1/2): pmf C(k+2, k)·(1/2)³·(1/2)ᵏ.
    // SymPy: summation(binomial(k+2,k)*Rational(1,2)**3*Rational(1,2)**k,(k,0,oo)) = 1,
    //        …*k = 3 (E[N]), …*k**2 = 15 (E[N²]),
    //        summation(binomial(k+2,k)*Rational(1,2)**(k+3),(k,3,oo)) = 1/2 (P(N > 2)).
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let half = ctx.rational(1, 2);
    let pmf = (&k + ctx.int(2)).binomial(&k) * half.powi(3) * half.pow(&k);
    assert_eq!(pmf.summation(&k, &zero, &inf), ctx.int(1));
    assert_eq!((&k * &pmf).summation(&k, &zero, &inf), ctx.int(3));
    assert_eq!((k.powi(2) * &pmf).summation(&k, &zero, &inf), ctx.int(15));
    // A positive lower bound subtracts the skipped leading terms.
    assert_eq!(pmf.summation(&k, &ctx.int(3), &inf), ctx.rational(1, 2));
    // The `C(k+r−1, k)` spelling with symbolic r and numeric p: p^r·(1−p)^{−r} = 1.
    // SymPy: summation(binomial(k+r-1,k)*Rational(1,2)**r*Rational(1,2)**k,(k,0,oo)).simplify() = 1
    let r = ctx.symbol("r");
    let pmf_r = (&k + &r - 1).binomial(&k) * half.pow(&r) * half.pow(&k);
    assert_eq!(pmf_r.summation(&k, &zero, &inf).simplify(), ctx.int(1));
}

#[test]
fn negative_binomial_series_symbolic_c() {
    // SymPy: summation(binomial(k+c,k)*Rational(1,3)**k,(k,0,oo)) is a Piecewise
    // whose main branch is (3/2)**(c + 1) = (2/3)^(−c−1) — the generalised binomial
    // series (1 − x)^(−c−1) holds for every c when |x| < 1.
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let c = ctx.symbol("c");
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let third = ctx.rational(1, 3);
    let s = ((&k + &c).binomial(&k) * third.pow(&k)).summation(&k, &zero, &inf);
    assert!(!s.has_unevaluated(), "{s}");
    let expected = ctx.rational(2, 3).pow(&(-&c - 1));
    assert_eq!(s, expected, "{s}");
    // C(k+c, c) spelling.
    let s2 = ((&k + &c).binomial(&c) * third.pow(&k)).summation(&k, &zero, &inf);
    assert_eq!(s2, expected, "{s2}");
    // Σ k·C(k+c,k) xᵏ = (c+1)·x·(1−x)^(−c−2).
    // SymPy (simplified, factorial form): 3*3**c*(c + 1)/(4*2**c); at c = 5 both give 2187/64.
    let s3 = (&k * (&k + &c).binomial(&k) * third.pow(&k)).summation(&k, &zero, &inf);
    assert!(!s3.has_unevaluated(), "{s3}");
    assert_eq!(s3.subs_i64(&c, 5).eval(), ctx.rational(2187, 64), "{s3}");
    // Numeric spot check at c = 2: (2/3)^(-3) = 27/8.
    assert_eq!(s.subs_i64(&c, 2).eval(), ctx.rational(27, 8));
}

#[test]
fn negative_binomial_series_symbolic_ratio_stays_formal() {
    // Convention of the summation engine (see `calculus::summation` docs): with a
    // symbolic ratio there is no |x| < 1 assumption, so the sum stays a formal `Sum`.
    // SymPy instead returns Piecewise(((1 - x)**(-c - 1), Abs(x) < 1), (Sum(…), True)).
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let c = ctx.symbol("c");
    let x = ctx.symbol("x");
    let p = ctx.symbol("p");
    let r = ctx.symbol("r");
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let s = ((&k + &c).binomial(&k) * x.pow(&k)).summation(&k, &zero, &inf);
    assert!(s.has_unevaluated(), "{s}");
    let s = ((&k + ctx.int(3)).binomial(&k) * x.pow(&k)).summation(&k, &zero, &inf);
    assert!(s.has_unevaluated(), "{s}");
    // NegativeBinomial pmf with both parameters symbolic.
    let pmf = (&k + &r - 1).binomial(&k) * p.pow(&r) * (ctx.one() - &p).pow(&k);
    assert!(pmf.summation(&k, &zero, &inf).has_unevaluated());
    // |x| ≥ 1 with numeric c ≥ 0 is divergent (SymPy leaves this Sum unevaluated;
    // the terms C(k+3,k)·3ᵏ → ∞).
    let s = ((&k + ctx.int(3)).binomial(&k) * ctx.int(3).pow(&k)).summation(&k, &zero, &inf);
    assert_eq!(s, ctx.infinity(), "{s}");
}

#[test]
fn negative_binomial_finite_sum_via_gosper() {
    // Σ_{k=0}^{n} C(k+2,k)(1/2)ᵏ with symbolic n closes through Gosper once the
    // hypergeometric ratio of C(k+c, k) is known: (k+c+1)/(k+1).
    // SymPy: [summation(binomial(k+2,k)*Rational(1,2)**k,(k,0,N)) for N in (0,1,4,7)]
    //        = [1, 5/2, 99/16, 121/16]
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let half = ctx.rational(1, 2);
    let body = (&k + ctx.int(2)).binomial(&k) * half.pow(&k);
    let s = body.summation(&k, &ctx.int(0), &n);
    assert!(!s.has_unevaluated(), "{s}");
    for (nv, want) in [
        (0, ctx.int(1)),
        (1, ctx.rational(5, 2)),
        (4, ctx.rational(99, 16)),
        (7, ctx.rational(121, 16)),
    ] {
        let v = s.subs_i64(&n, nv).eval().simplify();
        assert_eq!(v, want, "n = {nv}: {v}");
    }
    // Concrete bounds are enumerated exactly (SymPy: 99/16).
    assert_eq!(
        body.summation(&k, &ctx.int(0), &ctx.int(4)),
        ctx.rational(99, 16)
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 2b — sums with symbolic parameters from the statistics families
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn poisson_total_mass_and_moments_symbolic_rate() {
    // SymPy: summation(lam**k*exp(-lam)/factorial(k),(k,0,oo)) = 1
    //        simplify(summation(k**2*lam**k*exp(-lam)/factorial(k),(k,0,oo))) = lam*(lam + 1)
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let lam = ctx.symbol("lambda");
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let pmf = lam.pow(&k) * (-&lam).exp() / k.factorial();
    let total = pmf.summation(&k, &zero, &inf);
    assert!(!total.has_unevaluated(), "{total}");
    assert_eq!(total.simplify(), ctx.int(1), "{total}");
    let mean = (&k * &pmf).summation(&k, &zero, &inf);
    assert_eq!(mean.simplify(), lam, "{mean}");
    let second = (k.powi(2) * &pmf).summation(&k, &zero, &inf);
    assert_poly_eq(&second.simplify(), &(lam.powi(2) + &lam), "Poisson E[X²]");
}

#[test]
fn geometric_mean_numeric_p_closes_symbolic_p_stays_formal() {
    // SymPy: summation(k*Rational(2,3)**(k-1)*Rational(1,3),(k,1,oo)) = 3,
    //        summation(k**2*Rational(2,3)**(k-1)*Rational(1,3),(k,1,oo)) = 15.
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let one = ctx.int(1);
    let inf = ctx.infinity();
    let pmf = ctx.rational(2, 3).pow(&(&k - 1)) * ctx.rational(1, 3);
    assert_eq!(pmf.summation(&k, &one, &inf), ctx.int(1));
    assert_eq!((&k * &pmf).summation(&k, &one, &inf), ctx.int(3));
    assert_eq!((k.powi(2) * &pmf).summation(&k, &one, &inf), ctx.int(15));
    // Symbolic p: SymPy gives p*Piecewise((p**(-2), Abs(p - 1) < 1), (Sum(…), True));
    // symplex has no |1 − p| < 1 assumption and keeps the formal Sum.
    let p = ctx.symbol("p");
    let pmf_p = (ctx.one() - &p).pow(&(&k - 1)) * &p;
    assert!(pmf_p.summation(&k, &one, &inf).has_unevaluated());
    assert!((&k * &pmf_p).summation(&k, &one, &inf).has_unevaluated());
    // Likewise the bare geometric series Σ rᵏ.
    let r = ctx.symbol("r");
    assert!(r.pow(&k).summation(&k, &ctx.int(0), &inf).has_unevaluated());
}

#[test]
fn binomial_theorem_symbolic_n_and_p() {
    // SymPy: simplify(summation(binomial(n,k)*p**k*(1-p)**(n-k),(k,0,n))) has main
    // branch (-1/(p - 1))**n*(1 - p)**n = 1;
    //        …*k → n*p*(-1/(p - 1))**(n - 1)*(1 - p)**(n - 1) = n*p;
    //        …*k**2 → n*p*(n*p - p + 1)  (expand: n**2*p**2 - n*p**2 + n*p).
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("n");
    let p = ctx.symbol("p");
    let zero = ctx.int(0);
    let pmf = n.binomial(&k) * p.pow(&k) * (ctx.one() - &p).pow(&(&n - &k));
    let total = pmf.summation(&k, &zero, &n);
    assert_eq!(total, ctx.int(1), "{total}");
    let mean = (&k * &pmf).summation(&k, &zero, &n);
    assert_eq!(mean.simplify(), &n * &p, "{mean}");
    let second = (k.powi(2) * &pmf).summation(&k, &zero, &n);
    assert!(!second.has_unevaluated(), "{second}");
    let want = n.powi(2) * p.powi(2) - &n * p.powi(2) + &n * &p;
    assert_poly_eq(&second, &want, "Binomial E[X²]");
    // Numeric p with symbolic n: SymPy summation(binomial(n,k)*Rational(1,3)**k*Rational(2,3)**(n-k),(k,0,n))
    // = (2/3)**n * (3/2)**n = 1.
    let pmf_num = n.binomial(&k) * ctx.rational(1, 3).pow(&k) * ctx.rational(2, 3).pow(&(&n - &k));
    assert_eq!(pmf_num.summation(&k, &zero, &n), ctx.int(1));
}

#[test]
fn binomial_coefficient_with_k_greater_than_n_is_zero_on_every_route() {
    // SymPy: binomial(1, 2) = 0, binomial(2, 5) = 0, binomial(0, 1) = 0.
    // 0.12: `eval` folds the node, `eval_f64` short-circuits before the
    // Γ(n − k + 1) pole, and the f64 runtime behind `compile` agrees.
    let ctx = Context::new();
    let c12 = ctx.int(1).binomial(&ctx.int(2));
    assert_eq!(c12.eval(), ctx.int(0));
    assert_eq!(c12.eval_f64().unwrap(), 0.0);
    assert_eq!(ctx.int(2).binomial(&ctx.int(5)).eval(), ctx.int(0));
    assert_eq!(ctx.int(0).binomial(&ctx.int(1)).eval(), ctx.int(0));
    // Still a genuine value when k ≤ n.
    assert_eq!(ctx.int(5).binomial(&ctx.int(2)).eval(), ctx.int(10));
    let n = ctx.symbol("n");
    let k = ctx.symbol("k");
    let f = n.binomial(&k).compile(&["n", "k"]).unwrap();
    assert_eq!(f(&[1.0, 2.0]), 0.0);
    assert_eq!(f(&[2.0, 5.0]), 0.0);
    assert_eq!(f(&[0.0, 1.0]), 0.0);
}
