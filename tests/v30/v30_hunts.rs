//! Differential hunts after 0.30 (0.31): polynomial systems and
//! transcendental `solve`, sums and products with symbolic limits, and
//! piecewise inputs through the Laplace, Fourier, Z and Mellin transforms.
//! Each test says what was wrong before and cites its oracle.

use symplex::num_complex::Complex64;
use symplex::polysys::solve_system_ex;
use symplex::prelude::*;

fn c64(e: &Ex) -> Complex64 {
    e.eval_complex64()
        .unwrap_or_else(|err| panic!("{e} does not evaluate: {err}"))
}

fn close(a: Complex64, b: Complex64, tol: f64) -> bool {
    (a - b).norm() <= tol * (1.0 + b.norm())
}

// ─────────────────────────── systems and solve ───────────────────────────

/// `(y³ − 3)² = 0, x = 3y/2 − 3` in the shifted variables `y = v − u + 1`,
/// `x = u − 2v − 2`: the two complex solutions were dropped.  The residual
/// of the original equations at an exact complex candidate is a true zero
/// that `evalf` refuses to certify (`PrecisionExhausted`), and the old
/// check (an `evalf` string against `10⁻⁸·(1 + max|v|)^deg`) counted the
/// refusal as "unverified".  Found by the systems hunter (seed 617).
///
/// SymPy: `solve([e1, e2], [u, v])` → 3 solutions,
/// `(−2.04787349607593, −1.60562392576852)`,
/// `(5.52393674803796 ∓ 4.37158668269192i, 3.80281196288426 ∓ 3.12256191620852i)`.
#[test]
fn polynomial_system_keeps_complex_solutions_of_a_repeated_factor() {
    let ctx = Context::new();
    let (u, v) = (ctx.symbol("u"), ctx.symbol("v"));
    let e1 = ctx
        .parse(
            "3*(((-u + v + 1)^3 - 3)^2 + (-3*(-u + v + 1) + 2)*((u - 2*v - 2) - (-3 + 3/2*(-u + v + 1))))",
        )
        .unwrap();
    let e2 = ctx
        .parse("-((u - 2*v - 2) - (-3 + 3/2*(-u + v + 1)))")
        .unwrap();
    let sols = solve_system_ex(&[e1.clone(), e2.clone()], &[u.clone(), v.clone()]).unwrap();
    assert_eq!(sols.len(), 3, "{sols:?}");
    let want = [
        (
            Complex64::new(-2.04787349607593, 0.0),
            Complex64::new(-1.60562392576852, 0.0),
        ),
        (
            Complex64::new(5.52393674803796, -4.37158668269192),
            Complex64::new(3.80281196288426, -3.12256191620852),
        ),
        (
            Complex64::new(5.52393674803796, 4.37158668269192),
            Complex64::new(3.80281196288426, 3.12256191620852),
        ),
    ];
    for (wu, wv) in want {
        assert!(
            sols.iter()
                .any(|s| close(c64(&s[0]), wu, 1e-12) && close(c64(&s[1]), wv, 1e-12)),
            "missing ({wu}, {wv}) in {sols:?}"
        );
    }
    for s in &sols {
        for e in [&e1, &e2] {
            let r = e.subs(&u, &s[0]).subs(&v, &s[1]).simplify();
            assert!(r.is_zero_structural(), "residual {r} at {s:?}");
        }
    }
}

/// The range restrictions of inversion (`exp f = c ≤ 0`, `|cos f| > 1`,
/// `cosh f < 1`) were applied whatever the argument, so equations with
/// real solutions were reported as having none: `exp(i·x) = −1` (`x = π`),
/// `cos(i·x) = 2` (`x = ±acosh 2`), `exp(√x) = −1` (`x = −π²`).  They hold
/// only for an argument that is real for real `x`.
///
/// SymPy: `solve(exp(I*x) + 1, x)` → `[pi]`; `solve(cos(I*x) - 2, x)` →
/// `[log(2 - sqrt(3)), log(sqrt(3) + 2)]`; `solve(exp(sqrt(x)) + 1, x)` →
/// `[-pi**2]`.
#[test]
fn range_restrictions_only_for_real_arguments() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.parse("exp(I*x) + 1").unwrap().solve(&x).unwrap();
    assert_eq!(s.len(), 1);
    assert!(
        close(c64(&s[0]), Complex64::new(std::f64::consts::PI, 0.0), 1e-14),
        "{s:?}"
    );

    let s = ctx.parse("cos(I*x) - 2").unwrap().solve(&x).unwrap();
    let acosh2 = 2f64.acosh();
    for want in [acosh2, -acosh2] {
        assert!(
            s.iter()
                .any(|r| close(c64(r), Complex64::new(want, 0.0), 1e-14)),
            "{s:?}"
        );
    }

    let s = ctx.parse("exp(sqrt(x)) + 1").unwrap().solve(&x).unwrap();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0], ctx.parse("-pi^2").unwrap());

    // exp(x) = −1 still has no real solution (the documented contract).
    assert!(matches!(
        ctx.parse("exp(x) + 1").unwrap().solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
}

/// `solve_general` missed the periodic families of non-real arguments:
/// `(−1)ˣ = 1` gave only `0` (the family is `2n`), `(−1)ˣ = −1` only `1`,
/// `exp(i·x) = 1` only `0` (`2πn`).
///
/// mpmath: `power(-1, 2*k) = 1`, `power(-1, 2*k + 1) = -1`,
/// `exp(2j*pi*k) = 1` for integers `k` (checked below at `k = −3..3`).
#[test]
fn general_solution_families_of_complex_exponentials() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, want) in [
        ("(-1)^x - 1", "2*n"),
        ("(-1)^x + 1", "2*n + 1"),
        ("exp(I*x) - 1", "2*n*pi"),
    ] {
        let e = ctx.parse(src).unwrap();
        let g = e.solve_general(&x).unwrap();
        assert_eq!(g.parameters.len(), 1, "{src}: {g:?}");
        assert_eq!(g.solutions.len(), 1, "{src}: {g:?}");
        let n = &g.parameters[0];
        let want = ctx.parse(want).unwrap().subs(&ctx.symbol("n"), n);
        assert_eq!(g.solutions[0], want, "{src}");
        for k in -3..=3 {
            for r in g.instance(k) {
                assert!(c64(&e.subs(&x, &r)).norm() < 1e-12, "{src} at k = {k}");
            }
        }
    }
    // exp(i·x) = −1/2: x = π + i·ln 2 + 2πn.
    let e = ctx.parse("exp(I*x) + 1/2").unwrap();
    let g = e.solve_general(&x).unwrap();
    for k in -2..=2 {
        for r in g.instance(k) {
            assert!(c64(&e.subs(&x, &r)).norm() < 1e-12, "k = {k}: {r}");
        }
    }
}

/// A candidate of a non-polynomial equation was kept when its residual was
/// below `10⁻¹⁰`: `√x = −10⁻¹²` returned `x = 10⁻²⁴` (where `√x + 10⁻¹²` is
/// `2·10⁻¹²`), `asin x = π/2 + 10⁻¹¹` returned `sin(π/2 + 10⁻¹¹)`.  A
/// certified nonzero digit now rejects a candidate whatever its size.  A
/// family none of whose members solves the equation is rejected too
/// (`exp(√x) = −1/2`: `√x = ln(1/2) + (2n + 1)πi` has a negative real part).
///
/// SymPy: `solve(sqrt(x) + S(10)**-12, x)` → `[]`;
/// `solve(asin(x) - pi/2 - S(10)**-11, x)` → `[]`;
/// `solve(exp(sqrt(x)) + S(1)/2, x)` → `[]`.
#[test]
fn tiny_residuals_do_not_make_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "sqrt(x) + 10^(-12)",
        "asin(x) - pi/2 - 10^(-11)",
        "exp(sqrt(x)) + 1/2",
    ] {
        let e = ctx.parse(src).unwrap();
        assert!(
            matches!(e.solve(&x), Err(SymplexError::NoSolution { .. })),
            "{src}: {:?}",
            e.solve(&x)
        );
        assert!(
            matches!(e.solve_general(&x), Err(SymplexError::NoSolution { .. })),
            "{src}: {:?}",
            e.solve_general(&x)
        );
    }
}

/// A constant equation `c = 0` with `|evalf(c)| < 10⁻¹⁵` was an identity:
/// `solve(exp(−40), x)` reported every `x` as a solution.
///
/// SymPy: `solve(exp(-40), x)` → `[]`.
#[test]
fn tiny_nonzero_constant_is_not_an_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.parse("exp(-40)").unwrap();
    assert!(
        matches!(e.solve(&x), Err(SymplexError::NoSolution { .. })),
        "{:?}",
        e.solve(&x)
    );
    let z = ctx.parse("sqrt(2)*sqrt(3) - sqrt(6)").unwrap();
    assert!(matches!(
        z.solve(&x),
        Err(SymplexError::InfiniteSolutions { .. })
    ));
}

// ─────────────────────────── inequalities ───────────────────────────

/// A zero of the numerator outside the natural domain was a solution:
/// `(x + 2)/(ln x + 2) ≥ 0` contained the isolated point `x = −2`, where
/// `ln(−2)` is not real (the expression folded to `0` there because the
/// numerator is `0`).  Critical points now satisfy the domain constraints
/// of their branch, as sample points did.  Found by the sets hunter (15 of
/// 3,000 cases).
///
/// SymPy: `reduce_inequalities([(x + 2)/(log(x) + 2) >= 0, x > -3, x < 1], x)`
/// → `(-3 < x) & (x < 1) & (exp(-2) < x)`.
#[test]
fn inequality_zero_outside_the_domain_is_excluded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("(x + 2)/(ln(x) + 2)").unwrap();
    let set = reduce_inequalities(
        &[f.ge(&ctx.int(0)), x.gt(&ctx.int(-3)), x.lt(&ctx.int(1))],
        &x,
    )
    .unwrap();
    assert_eq!(set.contains(&ctx.int(-2)), Some(false), "{set}");
    assert_eq!(set.contains(&ctx.rational(1, 2)), Some(true), "{set}");
    assert_eq!(set.contains(&ctx.rational(1, 10)), Some(false), "{set}");
    assert_eq!(format!("{set}"), "(exp(-2), 1)");
}

// ─────────────────────────── products ───────────────────────────

/// A finite product with a factor that vanishes inside the range was `0`
/// for every `n` (the Gamma ratio `Γ(n − 1)/Γ(−2)` has a pole in its
/// denominator): `Π_{k=0}^{n}(k − 2)` is `−2` at `n = 0` and `2` at `n = 1`.
/// It is now the rising factorial.  Found by the sums hunter (62 of 400
/// product cases).
///
/// SymPy: `product(k - 2, (k, 0, n))` → `RisingFactorial(-2, n + 1)`, values
/// `[-2, 2, 0, 0]` at `n = 0..3`; `product(1 - 1/k**2, (k, 1, n))` →
/// `RisingFactorial(0, n)*RisingFactorial(2, n)/factorial(n)**2`, values
/// `[1, 0, 0, 0]`.
#[test]
fn products_through_a_zero_factor() {
    let ctx = Context::new();
    let (k, n) = (ctx.symbol("k"), ctx.symbol("n"));
    let p = (&k - 2).product_over(&k, &ctx.int(0), &n);
    let vals: Vec<Ex> = (0..4).map(|m| p.subs_i64(&n, m).eval()).collect();
    assert_eq!(
        vals,
        vec![ctx.int(-2), ctx.int(2), ctx.int(0), ctx.int(0)],
        "{p}"
    );

    let p = (ctx.int(1) - k.powi(-2)).product_over(&k, &ctx.int(1), &n);
    let vals: Vec<Ex> = (0..4).map(|m| p.subs_i64(&n, m).eval()).collect();
    assert_eq!(
        vals,
        vec![ctx.int(1), ctx.int(0), ctx.int(0), ctx.int(0)],
        "{p}"
    );

    // The infinite product keeps its limit.
    let p = (ctx.int(1) - k.powi(-2)).product_over(&k, &ctx.int(2), &ctx.infinity());
    assert_eq!(p, ctx.rational(1, 2));
}

/// `Σ 1/(k + β)^m` with `m ≥ 2` and a symbolic upper limit was refused (the
/// generalised harmonic number was not available).  It is now
/// `(−1)^{m−1} ψ^{(m−1)}(n + β + 1)/(m − 1)!` relative to the lower limit.
///
/// SymPy: `summation(1/k**2, (k, 1, n))` → `harmonic(n, 2)`;
/// `summation(1/k**3, (k, 1, n))` → `harmonic(n, 3)`;
/// `summation(1/k**2, (k, 2, n))` → `harmonic(n, 2) - 1`.  Checked against
/// the explicit sums at `n = 0..6`.
#[test]
fn generalized_harmonic_sums() {
    let ctx = Context::new();
    let (k, n) = (ctx.symbol("k"), ctx.symbol("n"));
    for (m, lo) in [(2, 1), (3, 1), (2, 2), (4, 3)] {
        let term = k.powi(-m);
        let closed = term.try_summation(&k, &ctx.int(lo), &n).unwrap();
        for nv in (lo - 1)..=6 {
            let mut acc = ctx.int(0);
            for j in lo..=nv {
                acc = (&acc + &term.subs_i64(&k, j)).eval();
            }
            let d = (&closed.subs_i64(&n, nv) - &acc).simplify();
            assert!(
                d.is_zero_structural(),
                "m = {m}, lo = {lo}, n = {nv}: {closed} vs {acc}"
            );
        }
    }
}

// ─────────────────────────── transforms ───────────────────────────

/// Every `Piecewise` input was refused by the Laplace transform, and so
/// were falling steps `H(c − t)`: the triangle, `H(1 − t)`,
/// `sin t·H(π − t)`.
///
/// SymPy: `laplace_transform(Piecewise((t, t<1), (2-t, t<2), (0, True)), t, s)`
/// → `(exp(2*s) - 2*exp(s) + 1)*exp(-2*s)/s**2`;
/// `laplace_transform(Heaviside(1 - t), t, s)` → `(1 - exp(-s))/s`;
/// `laplace_transform(sin(t)*Heaviside(pi - t), t, s)` →
/// `(exp(pi*s) + 1)*exp(-pi*s)/(s**2 + 1)`.
#[test]
fn laplace_of_piecewise_and_falling_steps() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let cases = [
        (
            "Piecewise(t if t < 1, 2 - t if t < 2, 0 if True)",
            "(1 - exp(-s))^2/s^2",
        ),
        ("Heaviside(1 - t)", "(1 - exp(-s))/s"),
        ("sin(t)*Heaviside(pi - t)", "(1 + exp(-pi*s))/(s^2 + 1)"),
        (
            "Piecewise(1 if abs(t - 2) < 1, 0 if True)",
            "(exp(-s) - exp(-3*s))/s",
        ),
    ];
    for (f, want) in cases {
        let got = ctx.parse(f).unwrap().try_laplace(&t, &s).unwrap();
        let want = ctx.parse(want).unwrap();
        for sv in [ctx.rational(1, 2), ctx.int(2), ctx.rational(7, 3)] {
            let (a, b) = (c64(&got.subs(&s, &sv)), c64(&want.subs(&s, &sv)));
            assert!(close(a, b, 1e-14), "{f} at s = {sv}: {got}");
        }
    }
}

/// The inverse Laplace transform refused a delay with a constant in the
/// exponent (`e^{−2s−1}/(s + 1)`) and a product of sums of delays
/// (`(1 − e^{−s})²/s²`).
///
/// SymPy: `inverse_laplace_transform((1-exp(-s))**2/s**2, s, t)` →
/// `t*Heaviside(t) + (t - 2)*Heaviside(t - 2) - 2*(t - 1)*Heaviside(t - 1)`;
/// `inverse_laplace_transform(exp(-2*s-1)/(s+1), s, t)` →
/// `E*exp(-t)*Heaviside(t - 2)`.
#[test]
fn inverse_laplace_of_delays() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let f = ctx
        .parse("(1 - exp(-s))^2/s^2")
        .unwrap()
        .try_inverse_laplace(&s, &t)
        .unwrap();
    for (tv, want) in [((1, 2), 0.5), ((3, 2), 0.5), ((3, 1), 0.0)] {
        let v = c64(&f.subs(&t, &ctx.rational(tv.0, tv.1)));
        assert!(close(v, Complex64::new(want, 0.0), 1e-14), "{f} at {tv:?}");
    }
    let f = ctx
        .parse("exp(-2*s - 1)/(s + 1)")
        .unwrap()
        .try_inverse_laplace(&s, &t)
        .unwrap();
    let v = c64(&f.subs(&t, &ctx.int(3)));
    assert!(close(v, Complex64::new((-2f64).exp(), 0.0), 1e-14), "{f}");
    assert!(c64(&f.subs(&t, &ctx.int(1))).norm() < 1e-15, "{f}");
}

/// The Fourier transform of a modulated window kept impulse terms whose
/// coefficients cancel (`±sin²1·πi/2`, `±cos²1·πi/2`, …), and the
/// transform of a window times anything but a constant was refused (the
/// triangle, `t·(H(t − 1) − H(t − 2))`, `e^{−t}` on a window).  Found by
/// the transforms hunter (154 of 250 refused).
///
/// mpmath: `quad(lambda t: sin(2*t)*exp(-1j*w*t), [1/2, 5/2])` at `w = 3/7`
/// → `0.32012066421768887217815011847 + 0.21419506854768733239331192j`;
/// `quad(lambda t: tri(t)*exp(-1j*w*t), [0, 1, 2])` at `w = 3/7` →
/// `0.895723466331136231792423864357 - 0.40924987745880122027241312j`.
#[test]
fn fourier_of_windows() {
    let ctx = Context::new();
    let (t, w) = (ctx.symbol("t"), ctx.symbol("w"));
    let w0 = ctx.rational(3, 7);
    let f = ctx
        .parse("sin(2*t)*(Heaviside(t - 1/2) - Heaviside(t - 5/2))")
        .unwrap()
        .fourier_transform(&t, &w)
        .unwrap();
    assert!(!f.to_string().contains("DiracDelta"), "{f}");
    let v = c64(&f.subs(&w, &w0));
    assert!(
        close(
            v,
            Complex64::new(0.32012066421768887, 0.21419506854768733),
            1e-13
        ),
        "{v}"
    );

    let tri = ctx
        .parse("Piecewise(0 if t < 0, t if t < 1, 2 - t if t < 2, 0 if True)")
        .unwrap()
        .fourier_transform(&t, &w)
        .unwrap();
    let v = c64(&tri.subs(&w, &w0));
    assert!(
        close(
            v,
            Complex64::new(0.8957234663311362, -0.4092498774588012),
            1e-13
        ),
        "{tri}"
    );

    // A window reaching t < 0 is shifted: ∫_{−1}^{3} e^{−t} e^{−iωt} dt.
    let f = ctx
        .parse("exp(-t)*(Heaviside(t + 1) - Heaviside(t - 3))")
        .unwrap()
        .fourier_transform(&t, &w)
        .unwrap();
    let z = Complex64::new(1.0, 3.0 / 7.0);
    let want = ((z * 1.0).exp() - (-z * 3.0).exp()) / z;
    assert!(close(c64(&f.subs(&w, &w0)), want, 1e-13), "{f}");
}

/// Every `Piecewise` sequence was refused by the Z-transform, and so was
/// `H(n − k)·aⁿ` (the delay shifted it to `a^(n+k)`, which the table did
/// not read); the inverse refused the transform of a finite window.
///
/// Direct partial sums (exact): `Σ_{n<3} 2^{−n} = 7/4`;
/// `Σ_{n≥2} (1/2)ⁿ 2^{−n} = 1/12`.
#[test]
fn z_transform_of_piecewise_sequences() {
    let ctx = Context::new();
    let (n, z) = (ctx.symbol("n"), ctx.symbol("z"));
    let x = ctx
        .parse("Piecewise(1 if n < 3, 0 if True)")
        .unwrap()
        .z_transform(&n, &z)
        .unwrap();
    assert_eq!(x.subs(&z, &ctx.int(2)).eval(), ctx.rational(7, 4), "{x}");
    let back = x.inverse_z_transform(&z, &n).unwrap();
    let vals: Vec<Ex> = (0..5).map(|j| back.subs_i64(&n, j).eval()).collect();
    assert_eq!(
        vals,
        vec![ctx.int(1), ctx.int(1), ctx.int(1), ctx.int(0), ctx.int(0)],
        "{back}"
    );

    let x = ctx
        .parse("Heaviside(n - 2)*(1/2)^n")
        .unwrap()
        .z_transform(&n, &z)
        .unwrap();
    assert_eq!(x.subs(&z, &ctx.int(2)).eval(), ctx.rational(1, 12), "{x}");

    let x = ctx
        .parse("Piecewise(n^2 if n <= 3, 0 if True)")
        .unwrap()
        .z_transform(&n, &z)
        .unwrap();
    // 0 + 1/2 + 4/4 + 9/8
    assert_eq!(x.subs(&z, &ctx.int(2)).eval(), ctx.rational(21, 8), "{x}");
}

/// The Mellin transform refused every `Piecewise` and every step with a
/// threshold other than 1; a compactly supported function written with
/// rising steps mixed strips `Re s < …` and `Re s > …` ("strips do not
/// overlap").
///
/// mpmath: `quad(lambda x: x**(-3/2), [5/2, inf])` → `1.26491106406735…`
/// (`M{H(x − 5/2)}(−1/2)`); `quad(lambda x: x**(1/2), [1, 2])` →
/// `1.21895141649746…` (window at `s = 3/2`); triangle at `s = 1/2` →
/// `1.10456949966158679680450326456`.
#[test]
fn mellin_of_steps_and_windows() {
    let ctx = Context::new();
    let (x, s) = (ctx.symbol("x"), ctx.symbol("s"));
    let (m, _strip) = ctx
        .parse("Heaviside(x - 5/2)")
        .unwrap()
        .mellin_transform(&x, &s)
        .unwrap();
    let v = c64(&m.subs(&s, &ctx.rational(-1, 2)));
    assert!(
        close(v, Complex64::new(1.2649110640673518, 0.0), 1e-14),
        "{m}"
    );

    let (m, _) = ctx
        .parse("Heaviside(x - 1) - Heaviside(x - 2)")
        .unwrap()
        .mellin_transform(&x, &s)
        .unwrap();
    let v = c64(&m.subs(&s, &ctx.rational(3, 2)));
    assert!(
        close(v, Complex64::new(1.21895141649746, 0.0), 1e-14),
        "{m}"
    );

    let (m, strip) = ctx
        .parse("Piecewise(x if x < 1, 2 - x if x < 2, 0 if True)")
        .unwrap()
        .mellin_transform(&x, &s)
        .unwrap();
    let v = c64(&m.subs(&s, &ctx.rational(1, 2)));
    assert!(
        close(v, Complex64::new(1.104569499661587, 0.0), 1e-14),
        "{m} on {strip}"
    );
}
