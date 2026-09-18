//! 0.2 numeric back-ends: `Ex::compile` / `Ex::compile_many`.
//!
//! Accuracy of every supported node is checked against the library's own
//! arbitrary-precision evaluator (`eval_decimal(30)`) where it supports the
//! function, and against independently computed (mpmath, 30 digits)
//! reference values otherwise.

// Reference constants are quoted with more digits than f64 holds on purpose.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use num_traits::ToPrimitive;
use symplex::prelude::*;

/// The exact rational value of a finite `f64`, as an expression.
fn exact(ctx: &Context, p: f64) -> Ex {
    let r = symplex::numeric::f64_to_ratio_exact(p).expect("finite input");
    let n = r.numer().to_i64().expect("numerator fits i64");
    let d = r.denom().to_i64().expect("denominator fits i64");
    ctx.rational(n, d)
}

/// Parse the decimal string produced by `eval_decimal`.
fn reference(e: &Ex) -> f64 {
    let s = e.eval_decimal(30).expect("eval_decimal should succeed");
    s.trim()
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("could not parse eval_decimal output {s:?}"))
}

fn rel_err(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        rel_err(got, want) <= tol,
        "{what}: got {got:.17e}, want {want:.17e}, rel err {:.2e} > {tol:.0e}",
        rel_err(got, want)
    );
}

/// Compile `f(x)` and compare with `eval_decimal` at each point.
fn check_unary_vs_evalf(name: &str, make: impl Fn(&Ex) -> Ex, points: &[(f64, f64)]) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = make(&x).compile(&["x"]).expect(name);
    assert_eq!(f.arity(), 1);
    for &(p, tol) in points {
        let want = reference(&make(&exact(&ctx, p)));
        assert_close(f(&[p]), want, tol, &format!("{name}({p})"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Special functions vs eval_decimal(30)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gamma_matches_evalf() {
    check_unary_vs_evalf(
        "gamma",
        |x| x.gamma(),
        &[
            (0.5, 2e-15),
            (1.5, 2e-15),
            (3.75, 2e-15),
            (12.25, 2e-15),
            (-0.5, 2e-15),
            (-2.5, 2e-15),
            (-7.3, 4e-15), // negative non-integer, reflection
            (0.001, 2e-15),
            (150.5, 2e-15),
        ],
    );
    // eval_decimal(30) returns "0" for Γ(170.25) (overflow in the
    // arbitrary-precision path); use an independent reference.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.gamma().compile(&["x"]).unwrap();
    assert_close(
        f(&[170.25]),
        1.5406560227188190329e305,
        3e-15,
        "gamma(170.25)",
    );
}

#[test]
fn gamma_poles_and_integers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.gamma().compile(&["x"]).unwrap();
    assert_eq!(f(&[1.0]), 1.0);
    assert_eq!(f(&[5.0]), 24.0);
    assert_eq!(f(&[21.0]), 2_432_902_008_176_640_000.0);
    assert!(f(&[0.0]).is_infinite());
    assert!(f(&[-1.0]).is_nan(), "negative integer pole → NaN");
    assert!(f(&[-3.0]).is_nan());
    assert_eq!(f(&[200.0]), f64::INFINITY);
}

#[test]
fn loggamma_matches_evalf() {
    check_unary_vs_evalf(
        "loggamma",
        |x| x.log_gamma(),
        &[
            (0.5, 1e-14),
            (3.5, 1e-14),
            (10.0, 1e-14),
            (25.5, 1e-15),
            (100.5, 1e-15),
            (1000.25, 1e-15),
            (-0.5, 1e-14),
            (-4.5, 1e-14),
        ],
    );
}

#[test]
fn digamma_matches_evalf() {
    check_unary_vs_evalf(
        "digamma",
        |x| x.digamma(),
        &[
            (0.5, 1e-14),
            (1.0, 1e-14),
            (2.75, 1e-14),
            (10.5, 1e-14),
            (200.0, 1e-15),
            (-0.5, 1e-13),
            (-3.25, 1e-13),
        ],
    );
}

#[test]
fn erf_erfc_match_evalf() {
    check_unary_vs_evalf(
        "erf",
        |x| x.erf(),
        &[
            (0.0, 1e-15),
            (0.1, 1e-15),
            (0.5, 1e-15),
            (1.0, 1e-15),
            (2.5, 1e-15),
            (-1.7, 1e-15),
            (4.0, 1e-15),
        ],
    );
    check_unary_vs_evalf(
        "erfc",
        |x| x.erfc(),
        &[(0.25, 1e-15), (1.0, 1e-15), (3.0, 1e-15), (-2.0, 1e-15)],
    );
    // For large arguments `eval_decimal(30)` computes erfc as 1 − erf and
    // loses all significance (erfc(8) ≈ 1e-29), so compare against
    // independent 30-digit references instead.  Relative precision must
    // survive all the way down to the underflow threshold.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.erfc().compile(&["x"]).unwrap();
    for &(p, want) in &[
        (5.0, 1.5374597944280348502e-12),
        (8.0, 1.122429717298292708e-29),
        (12.0, 1.3562611692059042128e-64),
        (20.0, 5.3958656116079009289e-176),
        (26.0, 5.6631924088561428465e-296),
    ] {
        assert_close(f(&[p]), want, 1e-15, &format!("erfc({p})"));
    }
    assert_eq!(f(&[30.0]), 0.0);
    assert_eq!(f(&[-30.0]), 2.0);
}

#[test]
fn lambertw_matches_evalf() {
    check_unary_vs_evalf(
        "lambertw",
        |x| x.lambertw(),
        &[
            (-0.3, 2e-15),
            (-0.1, 2e-15),
            (0.25, 2e-15),
            (1.0, 2e-15),
            (2.718281828459045, 2e-15),
            (10.0, 2e-15),
            (1e6, 2e-15),
            // Near the branch point the problem is ill-conditioned
            // (dW/dx → ∞); the input rounding alone costs ~1e-13.
            (-0.3678, 1e-12),
        ],
    );
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let w = x.lambertw().compile(&["x"]).unwrap();
    assert!(w(&[-0.5]).is_nan(), "outside domain → NaN");
    assert_eq!(w(&[0.0]), 0.0);
    // Functional identity W e^W = x, including very close to −1/e.
    for &p in &[-0.36787, -0.35, -0.2, 0.0, 0.5, 3.0, 1e3, 1e100] {
        let v = w(&[p]);
        assert_close(v * v.exp(), p, 1e-12, &format!("W({p}) e^W"));
    }
}

#[test]
fn beta_matches_evalf() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let f = a.beta(&b).compile(&["a", "b"]).unwrap();
    assert_eq!(f.arity(), 2);
    for &(pa, pb, tol) in &[
        (2.0, 3.0, 1e-15),
        (0.5, 0.5, 2e-15),
        (2.5, 1.5, 4e-15),
        (7.25, 0.75, 4e-15),
        (-0.5, 1.5, 1e-14),
        (30.0, 40.0, 1e-13),
    ] {
        let want = reference(&exact(&ctx, pa).beta(&exact(&ctx, pb)));
        assert_close(f(&[pa, pb]), want, tol, &format!("beta({pa},{pb})"));
    }
}

#[test]
fn binomial_and_factorial() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let k = ctx.symbol("k");
    let c = n.binomial(&k).compile(&["n", "k"]).unwrap();
    assert_eq!(c(&[10.0, 3.0]), 120.0);
    assert_eq!(c(&[52.0, 5.0]), 2_598_960.0);
    assert_eq!(c(&[5.0, 7.0]), 0.0);
    assert_eq!(c(&[5.0, -1.0]), 0.0);
    assert_eq!(c(&[0.5, 2.0]), -0.125);
    assert_eq!(c(&[-3.0, 2.0]), 6.0);
    // general real arguments vs evalf
    let want = reference(&ctx.rational(1, 2).binomial(&ctx.rational(1, 4)));
    assert_close(c(&[0.5, 0.25]), want, 1e-14, "C(1/2,1/4)");
    let want = reference(&ctx.int(200).binomial(&ctx.int(100)));
    assert_close(c(&[200.0, 100.0]), want, 1e-13, "C(200,100)");

    let f = n.factorial().compile(&["n"]).unwrap();
    assert_eq!(f(&[0.0]), 1.0);
    assert_eq!(f(&[5.0]), 120.0);
    assert_eq!(f(&[20.0]), 2_432_902_008_176_640_000.0);
    let want = reference(&ctx.rational(3, 2).gamma()); // (1/2)! = Γ(3/2)
    assert_close(f(&[0.5]), want, 2e-15, "0.5!");
    assert_eq!(f(&[171.0]), f64::INFINITY);
}

#[test]
fn bessel_j_matches_evalf_small_x() {
    // `eval_decimal` is only reliable for J_n at small x (it disagrees with
    // mpmath by O(1) for x ≳ 15, and its Y_n is wrong even at x = 0.5), so
    // this cross-check is restricted to J_n with x ≤ 10; see
    // `bessel_j_y_reference_values` for independent references.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for order in [0i64, 1, 2, 5] {
        let n = ctx.int(order);
        let j = x.bessel_j(&n).compile(&["x"]).expect("besselj compiles");
        for &p in &[0.5, 1.0, 3.7, 8.25, 10.0] {
            let arg = exact(&ctx, p);
            let wj = reference(&arg.bessel_j(&n));
            // Oscillatory: use error relative to max(|value|, 0.05) so zeros
            // don't dominate; target is 1e-12, we deliver ~1e-14.
            let ej = (j(&[p]) - wj).abs() / wj.abs().max(0.05);
            assert!(
                ej < 1e-12,
                "J_{order}({p}): got {}, want {wj}, err {ej:e}",
                j(&[p])
            );
        }
    }
}

#[test]
fn bessel_j_y_reference_values() {
    // mpmath (30 digits): (order, x, J_n(x), Y_n(x)) spanning the series,
    // Miller-recurrence and Hankel-asymptotic regimes.
    let cases: &[(i64, f64, f64, f64)] = &[
        (0, 0.5, 0.93846980724081290423, -0.44451873350670655715),
        (0, 3.7, -0.39923020337119111533, 0.10607431532035411027),
        (0, 8.25, 0.10920747150610137554, 0.25514960509864492878),
        (0, 15.5, -0.10923065090005016848, 0.17064491122943461749),
        (0, 24.0, -0.056230274166859267015, -0.15283402879758777874),
        (0, 30.5, -0.019389754517762152066, -0.14315731617410765097),
        (0, 60.0, -0.091471804089061869531, 0.047358952209449399203),
        (1, 0.5, 0.24226845767487388638, -1.4714723926702430692),
        (1, 3.7, 0.053833987745461790513, 0.41667437268380749329),
        (1, 8.25, 0.26220355199274381872, -0.093994487064634487199),
        (1, 15.5, 0.16721318035174714327, 0.11478614251334232744),
        (1, 24.0, -0.15403806518312122128, 0.053059776121202168863),
        (1, 30.5, -0.14349430015097094111, 0.017046142883876454172),
        (1, 60.0, 0.046598383758166317869, 0.091869609369866895264),
        (2, 0.5, 0.030604023458682641307, -5.4413708371742657196),
        (2, 3.7, 0.42832965620657586556, 0.11915507531954182124),
        (2, 8.25, -0.045642974053314995242, -0.27793614741734419841),
        (2, 15.5, 0.13080654513898528374, -0.15583379606642270427),
        (2, 24.0, 0.043393768734932498575, 0.15725567680768795948),
        (2, 30.5, 0.0099802922127804510095, 0.14427509603534545124),
        (2, 60.0, 0.09302508354766741346, -0.044296631897120502694),
        (5, 0.5, 8.053627241357474086e-6, -7946.3014788074733418),
        (5, 1.0, 0.00024975773021123443138, -260.40586662581222072),
        (5, 3.7, 0.09948541700833390963, -0.97906506823354205704),
        (5, 8.25, 0.12807165041063199603, 0.28152921270165615678),
        (5, 15.5, 0.039280041041026650998, 0.20446365724861588135),
        (5, 24.0, -0.16229575288623108409, -0.027805603670412992178),
        (5, 30.5, -0.13994926793930187173, -0.039621071791740274083),
        (5, 60.0, 0.02745474422834409975, 0.099464632840450885642),
    ];
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for &(order, p, wj, wy) in cases {
        let n = ctx.int(order);
        let j = x.bessel_j(&n).compile(&["x"]).unwrap();
        let y = x.bessel_y(&n).compile(&["x"]).unwrap();
        let ej = (j(&[p]) - wj).abs() / wj.abs().max(0.05);
        let ey = (y(&[p]) - wy).abs() / wy.abs().max(0.05);
        assert!(
            ej < 1e-13,
            "J_{order}({p}): got {}, want {wj}, err {ej:e}",
            j(&[p])
        );
        assert!(
            ey < 1e-13,
            "Y_{order}({p}): got {}, want {wy}, err {ey:e}",
            y(&[p])
        );
    }
    // Negative arguments / orders.
    let j1 = x.bessel_j(&ctx.int(1)).compile(&["x"]).unwrap();
    assert_close(j1(&[-1.0]), -0.44005058574493351596, 1e-15, "J1(-1)");
    let jm1 = x.bessel_j(&ctx.int(-1)).compile(&["x"]).unwrap();
    assert_close(jm1(&[1.0]), -0.44005058574493351596, 1e-15, "J-1(1)");
    let y0 = x.bessel_y(&ctx.int(0)).compile(&["x"]).unwrap();
    assert!(y0(&[-1.0]).is_nan());
    assert_eq!(y0(&[0.0]), f64::NEG_INFINITY);
}

#[test]
fn bessel_i_k_reference_values() {
    // mpmath (30 digits) references — evalf does not support I/K yet.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: &[(i64, f64, f64, f64)] = &[
        // (order, x, I_n(x), K_n(x))
        (0, 1.0, 1.2660658777520083356, 0.42102443824070833334),
        (1, 1.0, 0.56515910399248502721, 0.60190723019723457474),
        (0, 10.0, 2815.7166284662544715, 1.7780062316167651811e-5),
        (2, 3.0, 2.2452124409299511546, 0.061510458471742037657),
        (0, 0.1, 1.0025015629340956, 2.4270690247020165578),
        (5, 2.0, 0.009825679323131702, 9.4310491005964674428),
        (0, 50.0, 2.9325537838493363267e20, 3.4101677497894956e-23),
    ];
    for &(order, p, want_i, want_k) in cases {
        let n = ctx.int(order);
        let i = x.bessel_i(&n).compile(&["x"]).unwrap();
        let k = x.bessel_k(&n).compile(&["x"]).unwrap();
        assert_close(i(&[p]), want_i, 1e-13, &format!("I_{order}({p})"));
        assert_close(k(&[p]), want_k, 1e-13, &format!("K_{order}({p})"));
    }
    let k0 = x.bessel_k(&ctx.int(0)).compile(&["x"]).unwrap();
    assert!(k0(&[-1.0]).is_nan());
    assert!(k0(&[0.0]).is_infinite());
}

#[test]
fn orthogonal_polynomials_match_exact_eval() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = [-0.9, -0.3, 0.25, 0.5, 1.7];
    for deg in [0i64, 1, 2, 3, 6, 11] {
        let n = ctx.int(deg);
        let fams: [(&str, Ex); 5] = [
            ("legendre", x.legendre(&n)),
            ("chebyshev_t", x.chebyshev_t(&n)),
            ("chebyshev_u", x.chebyshev_u(&n)),
            ("hermite", x.hermite(&n)),
            ("laguerre", x.laguerre(&n)),
        ];
        for (name, expr) in fams {
            let f = expr
                .compile(&["x"])
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            for &p in &pts {
                // Exact rational evaluation of the polynomial as reference.
                let want = expr
                    .subs(&x, &exact(&ctx, p))
                    .eval()
                    .eval_f64()
                    .unwrap_or_else(|e| panic!("{name}({deg}, {p}) eval: {e}"));
                // Polynomials evaluated near their roots cancel; judge the
                // error relative to max(|value|, 1).
                let err = (f(&[p]) - want).abs() / want.abs().max(1.0);
                assert!(
                    err < 1e-13,
                    "{name}_{deg}({p}): got {}, want {want}, err {err:e}",
                    f(&[p])
                );
            }
        }
    }
    // Symbolic degree is rejected with a clear error.
    let m = ctx.symbol("m");
    match x.legendre(&m).compile(&["x", "m"]) {
        Err(SymplexError::NotImplemented(msg)) => {
            assert!(
                msg.contains("legendre"),
                "message should name the function: {msg}"
            )
        }
        other => panic!("expected NotImplemented, got {other:?}"),
    }
}

#[test]
fn integer_sequences_and_factorials() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let fib = n.fibonacci().compile(&["n"]).unwrap();
    for (k, want) in [
        (0.0, 0.0),
        (1.0, 1.0),
        (10.0, 55.0),
        (30.0, 832040.0),
        (-5.0, 5.0),
        (-6.0, -8.0),
    ] {
        assert_eq!(fib(&[k]), want, "F({k})");
    }
    assert_eq!(fib(&[78.0]), 8_944_394_323_791_464.0);
    assert!(fib(&[2.5]).is_nan());
    // Exact values via the symbolic evaluator for larger n.
    for k in [40i64, 90, 150] {
        let exact = ctx.int(k).fibonacci().eval().eval_f64().unwrap();
        assert_close(fib(&[k as f64]), exact, 1e-15, &format!("F({k})"));
    }
    let luc = n.lucas().compile(&["n"]).unwrap();
    assert_eq!(luc(&[0.0]), 2.0);
    assert_eq!(luc(&[1.0]), 1.0);
    assert_eq!(luc(&[10.0]), 123.0);
    assert_eq!(luc(&[-3.0]), -4.0);
    let h = n.harmonic().compile(&["n"]).unwrap();
    assert_eq!(h(&[0.0]), 0.0);
    assert_eq!(h(&[1.0]), 1.0);
    assert_close(h(&[5.0]), 137.0 / 60.0, 1e-15, "H5");
    for k in [50i64, 100, 101, 250] {
        let want = ctx.int(k).harmonic().eval().eval_f64().unwrap();
        assert_close(h(&[k as f64]), want, 2e-15, &format!("H({k})"));
    }
    // Large n uses ψ(n+1)+γ (mpmath reference).
    assert_close(h(&[12345.0]), 9.9982625683616251106, 1e-15, "H(12345)");
    assert_close(h(&[1e6]), 14.392726722865723631, 1e-15, "H(1e6)");
    let f2 = n.factorial2().compile(&["n"]).unwrap();
    assert_eq!(f2(&[7.0]), 105.0);
    assert_eq!(f2(&[8.0]), 384.0);
    assert_eq!(f2(&[0.0]), 1.0);
    assert_eq!(f2(&[-1.0]), 1.0);
    assert_eq!(f2(&[-3.0]), -1.0);

    let xs = ctx.symbol("x");
    let rf = xs.rising_factorial(&n).compile(&["x", "n"]).unwrap();
    assert_eq!(rf(&[3.0, 4.0]), 360.0);
    assert_eq!(rf(&[0.5, 3.0]), 1.875);
    assert_eq!(rf(&[-3.0, 5.0]), 0.0);
    let ff = xs.falling_factorial(&n).compile(&["x", "n"]).unwrap();
    assert_eq!(ff(&[5.0, 2.0]), 20.0);
    assert_eq!(ff(&[0.5, 2.0]), -0.25);
    // non-integer n → Γ ratio
    let want = reference(&ctx.rational(5, 2).gamma()) / reference(&ctx.rational(3, 4).gamma());
    assert_close(rf(&[0.75, 1.75]), want, 1e-14, "(3/4)_{7/4}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Elementary / piecewise / boolean nodes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn elementary_nodes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = x.sign().compile(&["x"]).unwrap();
    assert_eq!((f(&[2.0]), f(&[-2.0]), f(&[0.0])), (1.0, -1.0, 0.0));
    let f = x.heaviside().compile(&["x"]).unwrap();
    assert_eq!((f(&[2.0]), f(&[-2.0]), f(&[0.0])), (1.0, 0.0, 0.5));
    let f = x.dirac_delta().compile(&["x"]).unwrap();
    assert_eq!(f(&[0.3]), 0.0, "DiracDelta is 0 pointwise");
    let f = x.floor().compile(&["x"]).unwrap();
    assert_eq!(f(&[-2.5]), -3.0);
    let f = x.ceiling().compile(&["x"]).unwrap();
    assert_eq!(f(&[-2.5]), -2.0);
    let f = x.abs().compile(&["x"]).unwrap();
    assert_eq!(f(&[-2.5]), 2.5);
    let f = y.atan2(&x).compile(&["x", "y"]).unwrap();
    assert_close(
        f(&[-1.0, 1.0]),
        3.0 * std::f64::consts::FRAC_PI_4,
        1e-15,
        "atan2(1,-1)",
    );
    let f = x.min_with(&y).compile(&["x", "y"]).unwrap();
    assert_eq!(f(&[3.0, -1.0]), -1.0);
    let f = x.max_with(&y).compile(&["x", "y"]).unwrap();
    assert_eq!(f(&[3.0, -1.0]), 3.0);
}

#[test]
fn piecewise_with_boolean_conditions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.zero();
    let one = ctx.int(1);
    let two = ctx.int(2);
    // |x| via piecewise
    let abs_pw = Ex::piecewise(&[(&x, &x.ge(&zero)), (&(-&x), &x.lt(&zero))]);
    let f = abs_pw.compile(&["x"]).unwrap();
    assert_eq!(f(&[3.5]), 3.5);
    assert_eq!(f(&[-3.5]), 3.5);
    assert_eq!(f(&[0.0]), 0.0);
    // ordered branches with And/Or/Not and a catch-all
    let in_unit = x.gt(&zero).and(&x.lt(&one));
    let big = x.gt(&two).or(&x.lt(&(-&two)));
    let pw = Ex::piecewise(&[
        (&ctx.int(10), &in_unit),
        (&ctx.int(20), &big),
        (&ctx.int(30), &in_unit.not()),
    ]);
    let f = pw.compile(&["x"]).unwrap();
    assert_eq!(f(&[0.5]), 10.0);
    assert_eq!(f(&[5.0]), 20.0);
    assert_eq!(f(&[-9.0]), 20.0);
    assert_eq!(f(&[1.5]), 30.0);
    // No branch matches → NaN (never silently wrong).
    let pw2 = Ex::piecewise(&[(&x, &x.gt(&one))]);
    let g = pw2.compile(&["x"]).unwrap();
    assert!(g(&[0.0]).is_nan());
    assert_eq!(g(&[4.0]), 4.0);
    // Equality / inequality tests
    let eqpw = Ex::piecewise(&[(&one, &x.eq_expr(&two)), (&zero, &x.ne_expr(&two))]);
    let h = eqpw.compile(&["x"]).unwrap();
    assert_eq!(h(&[2.0]), 1.0);
    assert_eq!(h(&[2.5]), 0.0);
}

#[test]
fn physical_constants_and_special_values() {
    let ctx = Context::new();
    let c = ctx.physical_constant("c", ctx.int(299_792_458));
    let f = (&c * 2).compile(&[]).unwrap();
    assert_eq!(f(&[]), 599_584_916.0);
    let f = ctx.pi().compile(&[]).unwrap();
    assert_eq!(f(&[]), std::f64::consts::PI);
    let f = ctx.infinity().compile(&[]).unwrap();
    assert_eq!(f(&[]), f64::INFINITY);
}

// ═══════════════════════════════════════════════════════════════════════════
// Errors, ergonomics, compile_many
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn errors_are_specific() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    match (&x + &y).compile(&["x"]) {
        Err(SymplexError::FreeSymbol { name }) => assert_eq!(name, "y"),
        other => panic!("expected FreeSymbol, got {other:?}"),
    }
    match x.integrate(&x).compile(&["x"]) {
        // x.integrate(x) evaluates to x^2/2 — use an unevaluable integrand instead
        Ok(_) => {}
        Err(e) => panic!("unexpected: {e}"),
    }
    let hard = (x.sin().sin()).exp().integrate(&x);
    match hard.compile(&["x"]) {
        Err(SymplexError::NotImplemented(msg)) => {
            assert!(
                msg.contains("Integral"),
                "message should name the node: {msg}"
            )
        }
        Ok(_) => panic!("unevaluated integral must not compile"),
        Err(e) => panic!("unexpected error kind: {e}"),
    }
    match ctx.i_unit().compile(&[]) {
        Err(SymplexError::NotImplemented(msg)) => assert!(msg.contains("ImaginaryUnit")),
        other => panic!("expected NotImplemented, got {other:?}"),
    }
    match x.compile(&["x", "x"]) {
        Err(SymplexError::InvalidArgument { operation, .. }) => assert_eq!(operation, "compile"),
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    let n = ctx.symbol("n");
    match x.bessel_j(&n).compile(&["x", "n"]) {
        Err(SymplexError::NotImplemented(msg)) => assert!(msg.contains("besselj")),
        other => panic!("expected NotImplemented, got {other:?}"),
    }
    match x.bessel_j(&ctx.rational(1, 2)).compile(&["x"]) {
        Err(SymplexError::NotImplemented(_)) => {}
        other => panic!("half-integer order not supported: {other:?}"),
    }
}

fn assert_send_sync<T: Send + Sync>(_: &T) {}

#[test]
fn compiled_fn_ergonomics() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (x.sin() + x.powi(2)).compile(&["x"]).unwrap();
    assert_send_sync(&f);
    let g = f.clone();
    assert_eq!(f(&[0.7]), g.call(&[0.7]));
    assert_eq!(f.try_call(&[0.7]).unwrap(), f(&[0.7]));
    assert!(f.call(&[]).is_nan());
    assert!(f(&[1.0, 2.0]).is_nan());
    assert!(matches!(
        f.try_call(&[1.0, 2.0]),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(f.instruction_count() > 0);
    let dbg = format!("{f:?}");
    assert!(dbg.contains("arity"));
    // Usable from another thread.
    let h = std::thread::spawn(move || g(&[0.7])).join().unwrap();
    assert_eq!(h, f(&[0.7]));
    // Works as a plain closure argument via `&*f`.
    fn apply(k: &dyn Fn(&[f64]) -> f64) -> f64 {
        k(&[2.0])
    }
    assert_eq!(apply(&*f), f(&[2.0]));
}

#[test]
fn compile_many_jacobian() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f1 = &x.sin() * &y.exp();
    let f2 = &x.powi(3) + &(&x * &y).cos();
    let jac = Ex::compile_many(
        &[&f1.diff(&x), &f1.diff(&y), &f2.diff(&x), &f2.diff(&y)],
        &["x", "y"],
    )
    .unwrap();
    assert_eq!(jac.len(), 4);
    assert_eq!(jac.arity(), 2);
    let (px, py) = (0.4, -1.3);
    let mut out = [0.0; 4];
    jac.try_call(&[px, py], &mut out).unwrap();
    let want = [
        px.cos() * py.exp(),
        px.sin() * py.exp(),
        3.0 * px * px - py * (px * py).sin(),
        -px * (px * py).sin(),
    ];
    for (i, (g, w)) in out.iter().zip(want.iter()).enumerate() {
        assert_close(*g, *w, 1e-14, &format!("J[{i}]"));
    }
    assert_eq!(jac.call_vec(&[px, py]), out.to_vec());
    // Mismatched output buffer → NaNs / error.
    let mut small = [0.0; 2];
    jac.call(&[px, py], &mut small);
    assert!(small.iter().all(|v| v.is_nan()));
    assert!(jac.try_call(&[px], &mut out).is_err());
    assert_send_sync(&jac);
    let empty = Ex::compile_many(&[], &["a", "b"]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.arity(), 2);
    assert!(empty.call_vec(&[1.0, 2.0]).is_empty());
    // Cross-checks with scalar compile.
    let s = f2.diff(&x).compile(&["x", "y"]).unwrap();
    assert_eq!(s(&[px, py]), out[2]);
}

#[test]
fn cse_does_not_change_results() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.sin();
    let e = (&s.powi(2) + &s.powi(3) + &(&s * &x.cos()) + &s.exp()).simplify();
    let f = e.compile(&["x"]).unwrap();
    for p in [-2.0f64, -0.5, 0.0, 0.3, 1.9, 7.0] {
        let sv = p.sin();
        let want = sv * sv + sv * sv * sv + sv * p.cos() + sv.exp();
        assert_close(f(&[p]), want, 1e-14, &format!("cse({p})"));
    }
}

#[test]
fn deep_expression_compiles_iteratively() {
    // 10 000 nested function applications: the compiler must not recurse.
    // (Nested `Add` chains are avoided here because the base-layer
    // canonicaliser itself overflows on ~600 nested sums.)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut e = x.clone();
    for i in 0..10_000 {
        e = match i % 3 {
            0 => e.sin(),
            1 => e.exp(),
            _ => e.atan(),
        };
    }
    let f = e.compile(&["x"]).expect("deep expression compiles");
    assert!(f(&[0.5]).is_finite());
}

#[test]
fn numerical_precision_patterns() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(x) - 1 → exp_m1 keeps precision for tiny x
    let f = (x.exp() - 1).compile(&["x"]).unwrap();
    assert_close(f(&[1e-10]), 1e-10 + 5e-21, 1e-12, "exp_m1");
    // ln(1 + x) → ln_1p
    let g = (&x + 1).ln().compile(&["x"]).unwrap();
    assert_close(g(&[1e-10]), 1e-10 - 5e-21, 1e-12, "ln_1p");
    // odd roots of negatives are real
    let h = x.pow(&ctx.rational(1, 3)).compile(&["x"]).unwrap();
    assert_eq!(h(&[-8.0]), -2.0);
    // Real roots with odd denominator: (-32)^(2/5) = ((-32)^(1/5))^2 = 4,
    // (-32)^(3/5) = -8.
    let k = x.pow(&ctx.rational(2, 5)).compile(&["x"]).unwrap();
    assert_close(k(&[-32.0]), 4.0, 1e-15, "(-32)^(2/5)");
    let k3 = x.pow(&ctx.rational(3, 5)).compile(&["x"]).unwrap();
    assert_close(k3(&[-32.0]), -8.0, 1e-15, "(-32)^(3/5)");
    // division peephole: exactly one rounding
    let d = (&x / &ctx.int(3)).compile(&["x"]).unwrap();
    assert_eq!(d(&[1.0]), 1.0 / 3.0);
}
