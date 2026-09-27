//! After 0.29 — the ODE, series and limit hunts, real-root isolation by the
//! Descartes method, and `compile()` of constants that are not real.
//!
//! Each test says what was wrong before; every reference value cites the
//! oracle call that produced it (SymPy 1.14 `dsolve`/`limit`, mpmath 1.3.0
//! `taylor` at `mp.dps = 30`; the scripts are `ode_oracle.py`,
//! `rr_oracle.py` of the hunt).  A returned ODE solution is verified by
//! substitution: the residual, with the constants set to rational values,
//! must vanish at rational points to 30 digits.

use symplex::matrix::Matrix;
use symplex::prelude::*;

/// The ODE text with `D1Y`, `D2Y` standing for `y′`, `y″` in `x`.
fn ode(ctx: &Context, text: &str) -> Ex {
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let d1 = y.formal_diff(&x);
    let d2 = d1.formal_diff(&x);
    ctx.parse(text)
        .unwrap()
        .subs(&ctx.symbol("D2Y"), &d2)
        .subs(&ctx.symbol("D1Y"), &d1)
}

/// `|z|` of a decimal string from `eval_decimal` (real or `a ± b*i`).
fn magnitude(s: &str) -> f64 {
    let t = s.replace(' ', "");
    if let Some(body) = t.strip_suffix("*i") {
        let k = body
            .char_indices()
            .skip(1)
            .filter(|&(i, c)| (c == '+' || c == '-') && !body[..i].ends_with('e'))
            .map(|(i, _)| i)
            .last();
        let (re, im) = match k {
            Some(k) => (
                body[..k].parse::<f64>().unwrap(),
                body[k..].parse::<f64>().unwrap(),
            ),
            None => (0.0, body.parse::<f64>().unwrap()),
        };
        re.hypot(im)
    } else {
        t.parse::<f64>().unwrap().abs()
    }
}

/// The explicit solution `sol` of `text = 0` (in `D1Y`, `D2Y`) satisfies it
/// at `points` for the constant values `cs` (to 30 digits), and carries
/// exactly the constants `names`.
fn assert_solves(
    ctx: &Context,
    text: &str,
    sol: &Ex,
    names: &[&str],
    points: &[(i64, i64)],
    cs: &[(i64, i64)],
) {
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    assert!(!sol.contains(&y) && !sol.has_unevaluated(), "{text}: {sol}");
    for n in names {
        assert!(sol.contains(&ctx.symbol(n)), "{text}: {sol} lacks {n}");
    }
    let s1 = sol.diff(&x);
    let s2 = s1.diff(&x);
    let resid = ctx
        .parse(text)
        .unwrap()
        .subs(&ctx.symbol("D2Y"), &s2)
        .subs(&ctx.symbol("D1Y"), &s1)
        .subs(&y, sol);
    for &(p, q) in points {
        let mut r = resid.subs(&x, &ctx.rational(p, q));
        for (n, &(a, b)) in names.iter().zip(cs) {
            r = r.subs(&ctx.symbol(n), &ctx.rational(a, b));
        }
        let v = r
            .eval_decimal(30)
            .unwrap_or_else(|e| panic!("{text}: {r}: {e}"));
        assert!(
            magnitude(&v) < 1e-25,
            "{text}: residual {v} at {p}/{q}: {sol}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// compile() of constants that are not real
// ═══════════════════════════════════════════════════════════════════════════

/// `compile()` returned a function that is NaN everywhere for a constant
/// that is not real (`atanh(9)·x`, `asin(2) + x`), while `to_rust_fn`,
/// `to_c_fn` and the Python emitters refuse it.  It refuses with the same
/// message now.  A real constant sharing a non-real part with another one
/// was split by CSE around the shared `atanh(9)` and compiled to NaN:
/// `|atanh(9)| + |atanh(9) + 1|·x` at `x = 1` is 3.49906996579188119022…
/// (mpmath: `abs(atanh(9)) + abs(atanh(9) + 1)`).
#[test]
fn compile_refuses_constants_that_are_not_real() {
    let ctx = Context::new();
    for bad in ["atanh(9)*x", "asin(2) + x", "x*acosh(1/2)"] {
        let e = ctx.parse(bad).unwrap();
        let err = e.compile(&["x"]).expect_err(bad);
        assert!(err.to_string().contains("is not real"), "{bad}: {err}");
        let err = e.to_rust_fn("f", &["x"]).expect_err(bad);
        assert!(err.to_string().contains("is not real"), "{bad}: {err}");
    }
    let shared = ctx.parse("abs(atanh(9)) + abs(atanh(9) + 1)*x").unwrap();
    let f = shared.compile(&["x"]).unwrap();
    assert!(
        (f(&[1.0]) - 3.499_069_965_791_881).abs() < 1e-14,
        "{}",
        f(&[1.0])
    );
    // 0.30 (decision D2): the odd root of a negative constant is its
    // principal value, `(-8)^(1/3)` = 1 + √3·i, not real, and refused like
    // the others (it compiled to the real root −2 before); the real root is
    // `real_root`.  Real constants still fold.
    let err = ctx
        .parse("(-8)^(1/3)*x")
        .unwrap()
        .compile(&["x"])
        .expect_err("(-8)^(1/3) is not real");
    assert!(err.to_string().contains("is not real"), "{err}");
    let odd = ctx.int(-8).real_root(3).unwrap() * ctx.symbol("x");
    assert_eq!(odd.compile(&["x"]).unwrap()(&[1.0]), -2.0);
    let abs = ctx
        .parse("abs(atanh(9))*x")
        .unwrap()
        .compile(&["x"])
        .unwrap();
    assert!((abs(&[1.0]) - 1.574_753_746_271_339_7).abs() < 1e-15);
}

// ═══════════════════════════════════════════════════════════════════════════
// Real-root isolation
// ═══════════════════════════════════════════════════════════════════════════

/// Real-root isolation is the Descartes method now; the old Sturm chain
/// took 4.3 s (release) for `count_real_roots` of a degree-40 polynomial
/// with 41 random 30-digit rational coefficients, all in its subresultant
/// sequence.  Two real roots, in (−2, −1) and (1, 2): SymPy
/// `Poly(p, x).intervals()`; mpmath `polyroots` at 200 digits.
#[test]
fn real_roots_with_huge_denominators_are_fast() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut state = 0x1234_5678_9ABC_DEF1u64;
    let mut digits = |n: usize| {
        let mut s = String::new();
        for i in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let d = (state % 10) as u8;
            s.push(char::from(b'0' + if i == 0 { d % 9 + 1 } else { d }));
        }
        s
    };
    let text = (0..=40)
        .map(|i| {
            let (n, d) = (digits(30), digits(30));
            let sign = if i % 3 == 1 { "-" } else { "" };
            format!("({sign}{n}/{d})*x^{i}")
        })
        .collect::<Vec<_>>()
        .join(" + ");
    let p = ctx.parse(&text).unwrap();
    let start = std::time::Instant::now();
    assert_eq!(p.count_real_roots(&x), Some(2));
    let ivs = p.real_roots_isolate(&x);
    assert_eq!(ivs.len(), 2);
    assert!(start.elapsed().as_secs() < 60, "took {:?}", start.elapsed());
    let (lo0, hi0) = (
        ivs[0].lower.eval_f64().unwrap(),
        ivs[0].upper.eval_f64().unwrap(),
    );
    let (lo1, hi1) = (
        ivs[1].lower.eval_f64().unwrap(),
        ivs[1].upper.eval_f64().unwrap(),
    );
    assert!(
        -2.0 < lo0 && hi0 < -1.0 && 1.0 < lo1 && hi1 < 2.0,
        "{ivs:?}"
    );
}

/// The isolating intervals are those of the Sturm-chain bisection, byte
/// for byte (the certificates built on root isolation depend on it): the
/// intervals below are the output of 0.30's Sturm implementation for the
/// same polynomials (a Mignotte polynomial with two roots `2·10⁻⁵` apart,
/// and rational roots that bisection points hit).
#[test]
fn isolating_intervals_are_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: [(&str, &[(&str, &str)]); 2] = [
        (
            "x^9 - 2*(9*x - 1)^2",
            &[
                ("233003/2097152", "466047/4194304"),
                ("466047/4194304", "58261/524288"),
                ("133373/65536", "66707/32768"),
            ],
        ),
        (
            "(x + 6)*(x - 2)*(4*x - 2)*(3*x - 1)*(3*x + 6)",
            &[
                ("-196625/32768", "-294905/49152"),
                ("-196625/98304", "-4095/2048"),
                ("1365/4096", "32825/98304"),
                ("4095/8192", "49205/98304"),
                ("4095/2048", "196625/98304"),
            ],
        ),
    ];
    for (text, want) in cases {
        let ivs = ctx.parse(text).unwrap().expand().real_roots_isolate(&x);
        let got: Vec<(String, String)> = ivs
            .iter()
            .map(|iv| (iv.lower.to_string(), iv.upper.to_string()))
            .collect();
        let want: Vec<(String, String)> = want
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        assert_eq!(got, want, "{text}");
        assert!(ivs.iter().all(|iv| iv.kind == IntervalKind::LeftOpen));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ODEs
// ═══════════════════════════════════════════════════════════════════════════

/// An exact equation whose potential cannot be solved for `y` came back as
/// `F(x, y)` without its constant (`2xy + (x² + 3y²)y′ = 0` gave `y³ + x²y`);
/// now `F − C1` (= 0), like the other implicit solutions.  SymPy:
/// `Eq(x**2*y + y**3, C1)` (hint `1st_exact`).
#[test]
fn exact_implicit_solution_carries_its_constant() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let sol = ode(&ctx, "2*x*y + (x^2 + 3*y^2)*D1Y").solve_ode(&y, &x);
    let c1 = ctx.symbol("C1");
    assert!(sol.contains(&c1) && sol.contains(&y), "{sol}");
    // F − C1 with F_x·N − F_y·M = 0: d/dx F along solutions vanishes.
    let expected = ctx.parse("x^2*y + y^3 - C1").unwrap();
    assert_eq!(
        (&sol - &expected).expand().simplify().to_string(),
        "0",
        "{sol}"
    );
}

/// `F(y′, y″) = 0` and `F(y, y′, y″) = 0`: the reduced first-order solution
/// was used even when implicit, leaking the dummy (`y″ + 2y′² = 0` gave
/// `−1/4·ln(−2·__p²·(C2 + x))`, one constant).  SymPy: `y = C1 +
/// log(C2 + 2x)/2`; `x·y″ + y′ = 0` (no `y`, reduced through `p(x)`):
/// `y = C1 + C2·log(x)`.
#[test]
fn reducible_equations_have_two_constants_and_no_dummy() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for text in ["D2Y + 2*D1Y^2", "x*D2Y + D1Y", "D2Y - D1Y^2"] {
        let sol = ode(&ctx, text).solve_ode(&y, &x);
        assert!(!sol.to_string().contains("__"), "{text}: {sol}");
        assert_solves(
            &ctx,
            text,
            &sol,
            &["C1", "C2"],
            &[(1, 3), (7, 5)],
            &[(-5, 1), (2, 3)],
        );
    }
    // y y″ − y′² = 0 through p(y) with the factor p divided out.
    let sol = ode(&ctx, "y*D2Y - D1Y^2").solve_ode(&y, &x);
    assert!(
        sol.contains(&ctx.symbol("C1")) && sol.contains(&ctx.symbol("C2")),
        "{sol}"
    );
    assert!(!sol.to_string().contains("__"), "{sol}");
}

/// First-order linear equations with a coefficient on `y′`: a number other
/// than 1 with a constant `P` (`2y′ + y = cos x`) and a function of `x`
/// (`x·y′ + 2y = x²`) were not solved at all, nor `2y′ = e⁻ˣ`.  SymPy:
/// `y = C1·exp(−x/2) + 2 sin(x)/5 + cos(x)/5`; `y = C1 − exp(−x)/2`.
#[test]
fn linear_equations_with_a_coefficient_on_the_derivative() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for text in [
        "2*D1Y + y - cos(x)",
        "2*D1Y - exp(-x)",
        "(1 + x^2)*D1Y - x^2",
        "x*D1Y + 2*y - x^2",
    ] {
        let sol = ode(&ctx, text).solve_ode(&y, &x);
        assert_solves(&ctx, text, &sol, &["C1"], &[(1, 3), (7, 5)], &[(2, 3)]);
    }
    let sol = ode(&ctx, "2*D1Y + y - cos(x)").solve_ode(&y, &x);
    let sympy = ctx.parse("C1*exp(-x/2) + 2*sin(x)/5 + cos(x)/5").unwrap();
    assert_eq!(
        (&sol - &sympy).expand().simplify().to_string(),
        "0",
        "{sol}"
    );
}

/// Homogeneous right-hand sides that are quotients of sums were refused:
/// the homogeneity test cancelled `x` with the univariate `cancel`, whose
/// coefficients cannot involve `v` (`(3vx − x)/(vx + x)`).  An integrating
/// factor `μ(x)` was missed the same way.  The implicit solutions satisfy
/// `F_x + F_y·y′ = 0`; SymPy: `y = ±x·sqrt(C1·x − 1)` for the second one
/// (equivalent to our implicit `ln(1 + y²/x²) − ln|x| − C1 = 0`).
#[test]
fn homogeneous_and_integrating_factor_quotients() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for (text, rhs) in [
        ("D1Y - (3*y - x)/(x + y)", "(3*y - x)/(x + y)"),
        ("D1Y - (x^2 + 3*y^2)/(2*x*y)", "(x^2 + 3*y^2)/(2*x*y)"),
        ("D1Y - 2*x*y/(x^2 - y^2)", "2*x*y/(x^2 - y^2)"),
        ("(2*x + y)/x + ((2*y + x)/x)*D1Y", "-(2*x + y)/(2*y + x)"),
    ] {
        let sol = ode(&ctx, text).solve_ode(&y, &x);
        assert!(
            !sol.has_unevaluated() && sol.contains(&ctx.symbol("C1")),
            "{text}: {sol}"
        );
        let f = ctx.parse(rhs).unwrap();
        let relation = &sol.diff(&x) + &(&sol.diff(&y) * &f);
        for (px, py) in [((1, 3), (5, 2)), ((7, 5), (1, 3))] {
            let v = relation
                .subs(&x, &ctx.rational(px.0, px.1))
                .subs(&y, &ctx.rational(py.0, py.1))
                .eval_decimal(30)
                .unwrap();
            assert!(magnitude(&v) < 1e-25, "{text}: {sol}: {v}");
        }
    }
}

/// Forced Cauchy–Euler equations were refused; `x = eᵗ` reduces them to
/// constant coefficients.  SymPy: `x²y″ − 3xy′ + 3y = ln x` has
/// `y = C1·x + C2·x³ + log(x)/3 + 4/9`; `x²y″ + xy′ − 4y = x²` has
/// `y = (C1 + x⁴(C2 + log x)/4)/x²`.
#[test]
fn forced_cauchy_euler_equations() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "x^2*D2Y - 3*x*D1Y + 3*y - log(x)";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert_solves(
        &ctx,
        text,
        &sol,
        &["C1", "C2"],
        &[(1, 3), (5, 2)],
        &[(2, 3), (-3, 7)],
    );
    let at = |e: &Ex| {
        e.subs(&ctx.symbol("C1"), &ctx.int(0))
            .subs(&ctx.symbol("C2"), &ctx.int(0))
    };
    assert_eq!(at(&sol).simplify().to_string(), "1/3*ln(x) + 4/9", "{sol}");
    let text = "x^2*D2Y + x*D1Y - 4*y - x^2";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert_solves(
        &ctx,
        text,
        &sol,
        &["C1", "C2"],
        &[(1, 3), (5, 2)],
        &[(2, 3), (-3, 7)],
    );
    // Initial values are fitted as for the other linear equations.
    let ivp = ode(&ctx, text)
        .solve_ode_ivp(
            &y,
            &x,
            &[
                InitialCondition {
                    order: 0,
                    x: ctx.int(1),
                    value: ctx.int(0),
                },
                InitialCondition {
                    order: 1,
                    x: ctx.int(1),
                    value: ctx.int(1),
                },
            ],
        )
        .unwrap();
    assert_eq!(ivp.subs(&x, &ctx.int(1)).simplify().to_string(), "0");
    assert_eq!(
        ivp.diff(&x).subs(&x, &ctx.int(1)).simplify().to_string(),
        "1"
    );
}

/// Variation of parameters built `x·e⁻ˣ·e⁻ˣ/x²/e⁻²ˣ` and could not
/// integrate it; the exponentials are merged first now.  SymPy:
/// `y = (C1 + C2·x − log(x))·exp(−x)`.
#[test]
fn variation_of_parameters_merges_exponentials() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D2Y + 2*D1Y + y - exp(-x)/x^2";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert_solves(
        &ctx,
        text,
        &sol,
        &["C1", "C2"],
        &[(1, 3), (5, 4)],
        &[(2, 3), (-3, 7)],
    );
}

/// An implicit solution from an earlier method no longer wins over an
/// explicit one from a later method: `y′ = y − y²` came back from the
/// integrating factor as `x − ln|y| + ln|y − 1|` (no constant).  SymPy:
/// `y = 1/(C1·exp(−x) + 1)`.
#[test]
fn explicit_solutions_are_preferred() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D1Y - y + y^2";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert_solves(&ctx, text, &sol, &["C1"], &[(1, 3), (7, 5)], &[(2, 3)]);
    // Separable: ∫ dy/(1 + y²) was never attempted.  SymPy: y = −tan(C1 − x).
    let sol = ode(&ctx, "D1Y - 1 - y^2").solve_ode(&y, &x);
    assert_solves(&ctx, "D1Y - 1 - y^2", &sol, &["C1"], &[(1, 3)], &[(1, 5)]);
}

/// `x′ = A·x` for a defective or not-diagonalisable-by-the-old-route `A`
/// returned the degree-12 Taylor polynomial of `e^{At}·c` as the solution.
/// Now the exact exponential (SymPy: `f1 = C1·e⁻ᵗ + C2·t·e⁻ᵗ`, `f2 = C2·e⁻ᵗ`
/// for `[[−1, 1], [0, −1]]`), and complex eigenvalues give real modes
/// (SymPy for `[[0, −1], [2, −2]]`: `e⁻ᵗ` times `cos t`, `sin t`).
#[test]
fn linear_systems_are_exact() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    for rows in [
        vec![vec![-1, 1], vec![0, -1]],
        vec![vec![0, -1], vec![2, -2]],
        vec![vec![2, 3], vec![-3, -2]],
        vec![vec![3, -1], vec![-3, -2]],
        vec![vec![1, 1, 0], vec![0, 1, 0], vec![0, 0, 2]],
    ] {
        let n = rows.len();
        let a = Matrix::new(
            rows.iter()
                .map(|r| r.iter().map(|&v| ctx.int(v)).collect())
                .collect(),
        )
        .unwrap();
        let sol = symplex::ode::solve_ode_system(&a, &t).unwrap();
        for i in 0..n {
            assert!(!sol[i].contains(&ctx.i_unit()), "{rows:?}: {}", sol[i]);
            let mut r = sol[i].diff(&t);
            for k in 0..n {
                r = &r - &(&ctx.int(rows[i][k]) * &sol[k]);
            }
            for (j, tv) in [(1, 3), (7, 5)].iter().enumerate() {
                let mut v = r.subs(&t, &ctx.rational(tv.0, tv.1));
                for c in 1..=n {
                    v = v.subs(
                        &ctx.symbol(&format!("C{c}")),
                        &ctx.rational(c as i64 + j as i64, 3),
                    );
                }
                let d = v.eval_decimal(30).unwrap();
                assert!(magnitude(&d) < 1e-25, "{rows:?} row {i}: residual {d}");
            }
        }
        // x' = A x + b with the exact exponentials.
        let b: Vec<Ex> = (0..n)
            .map(|i| if i == 0 { ctx.int(1) } else { t.clone() })
            .collect();
        let nsol = symplex::ode::solve_ode_system_nonhomogeneous(&a, &b, &t).unwrap();
        for i in 0..n {
            let mut r = &nsol[i].diff(&t) - &b[i];
            for k in 0..n {
                r = &r - &(&ctx.int(rows[i][k]) * &nsol[k]);
            }
            let mut v = r.subs(&t, &ctx.rational(1, 3));
            for c in 1..=n {
                v = v.subs(&ctx.symbol(&format!("C{c}")), &ctx.rational(c as i64, 3));
            }
            let d = v.eval_decimal(30).unwrap();
            assert!(
                magnitude(&d) < 1e-25,
                "{rows:?} nonhomogeneous row {i}: residual {d}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Series
// ═══════════════════════════════════════════════════════════════════════════

/// Bessel functions at 0: `Y₀`, `K₀` (logarithmic singularity) came back as
/// polynomials in `Y₀(0) = −∞`, and `J₁/₂` (a square root) in
/// `J₋₁/₂(0) = ∞`; now a formal `Series`.  Integer orders of `J`, `I` expand
/// from their power series (before, in terms of the unevaluated `J₋ₖ(0)`).
/// mpmath `taylor(lambda z: besselj(3, z), 0, 7)`: `x³/48 − x⁵/768 + …`;
/// `taylor(lambda z: besseli(1, 2z), 0, 6)`: `x + x³/2 + x⁵/12`.  An
/// essential singularity without a real asymptotic expansion stays formal;
/// `exp(−1/x²)` is `0 + O(xⁿ)` on the real line (SymPy:
/// `series(exp(-1/x**2), x, 0, 4)` = `O(x**4)`).
#[test]
fn series_at_essential_and_logarithmic_singularities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    assert_eq!(
        ctx.parse("exp(-1/x^2)")
            .unwrap()
            .series(&x, &zero, 4)
            .to_string(),
        "0"
    );
    for text in [
        "exp(1/x)",
        "sin(1/x)",
        "bessely(0, x)",
        "besselk(0, x)",
        "besselj(1/2, x)",
    ] {
        let s = ctx.parse(text).unwrap().series(&x, &zero, 4);
        assert!(s.has_unevaluated(), "{text}: {s}");
    }
    let j3 = ctx.parse("besselj(3, x)").unwrap().series(&x, &zero, 7);
    assert_eq!(j3.to_string(), "-1/768*x^5 + 1/48*x^3");
    let i1 = ctx.parse("besseli(1, 2*x)").unwrap().series(&x, &zero, 6);
    assert_eq!(i1.to_string(), "1/12*x^5 + 1/2*x^3 + x");
    let jm3 = ctx.parse("besselj(-3, x)").unwrap().series(&x, &zero, 7);
    assert_eq!(jm3.to_string(), "1/768*x^5 - 1/48*x^3");
}

// ═══════════════════════════════════════════════════════════════════════════
// Limits with a parameter
// ═══════════════════════════════════════════════════════════════════════════

/// The limit methods take a symbolic coefficient to be non-zero, and the
/// answer was returned for every value of it: `x²/(a + x²) → 0` at `x = 0`
/// also for `a = 0` (the limit is 1), `(a·x + 1)/(a·x + 2) → 1` at `∞` also
/// for `a = 0` (it is 1/2).  SymPy 1.14 returns the same generic answers
/// (`limit(x**2/(a + x**2), x, 0)` = 0, `limit((a*x+1)/(a*x+2), x, oo)` =
/// 1); the degenerate values now get their own case.
#[test]
fn parametric_limits_split_at_degenerate_values() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let l = ctx.parse("x^2/(a + x^2)").unwrap().limit(&x, &ctx.int(0));
    assert_eq!(l.subs(&a, &ctx.int(0)).eval().to_string(), "1", "{l}");
    assert_eq!(l.subs(&a, &ctx.int(3)).eval().to_string(), "0", "{l}");
    let l = ctx
        .parse("(a*x + 1)/(a*x + 2)")
        .unwrap()
        .limit(&x, &ctx.infinity());
    assert_eq!(l.subs(&a, &ctx.int(0)).eval().to_string(), "1/2", "{l}");
    assert_eq!(l.subs(&a, &ctx.int(-2)).eval().to_string(), "1", "{l}");
    // Answers that hold for every value (or are undefined where they do
    // not) are unchanged.
    assert_eq!(
        ctx.parse("sin(a*x)/x")
            .unwrap()
            .limit(&x, &ctx.int(0))
            .to_string(),
        "a"
    );
    assert_eq!(
        ctx.parse("x/(a*x + 1)")
            .unwrap()
            .limit(&x, &ctx.infinity())
            .to_string(),
        "1/a"
    );
}
