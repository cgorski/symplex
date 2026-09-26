//! Lambert W on every branch, the evalf kernels that were missing, and
//! the polynomial routines that hung (0.30).  Each test says what was wrong
//! before and cites its oracle: mpmath 1.3 with exact rational inputs
//! (`mpf(-1)/5`), quoted at the digits it printed at two working
//! precisions that agree (the dps are given with each value, the exponent
//! and a trailing `.0` rewritten in symplex's format), SymPy 1.14 for
//! `solve` and `count_roots`, and mpmath's `findroot` for the real roots of
//! the Lambert-type equations.

use symplex::prelude::*;

fn dec(ctx: &Context, s: &str, digits: u32) -> Result<String, SymplexError> {
    ctx.parse(s)
        .unwrap_or_else(|e| panic!("parse {s}: {e:?}"))
        .eval_decimal(digits)
}

fn check(ctx: &Context, s: &str, digits: u32, want: &str) {
    match dec(ctx, s, digits) {
        Ok(got) => assert_eq!(got, want, "{s} at {digits} digits"),
        Err(e) => panic!("{s} at {digits} digits: {e:?}, want {want}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Lambert W: the branches W_k
// ═══════════════════════════════════════════════════════════════════════════

/// Before: there was only the principal branch, `lambertw(x, k)` did not
/// parse, and evalf refused `W(x)` for `x < −1/e` ("argument < −1/e,
/// outside principal branch domain") and for every complex argument.
#[test]
fn lambertw_branches_evaluate_over_the_complex_plane() {
    let ctx = Context::new();
    // mpmath (dps 50, 70): lambertw(mpf(-1)/5, -1)
    check(&ctx, "W(-1/5, -1)", 30, "-2.54264135777352642429380615666");
    // mpmath (dps 50, 70): lambertw(mpf(-1))
    check(
        &ctx,
        "W(-1)",
        30,
        "-0.318131505204764135312654251588 + 1.33723570143068940890116214319*i",
    );
    // mpmath (dps 50, 70): lambertw(mpf(-1), -1) — the conjugate on the cut
    check(
        &ctx,
        "lambertw(-1, -1)",
        30,
        "-0.318131505204764135312654251588 - 1.33723570143068940890116214319*i",
    );
    // mpmath (dps 50, 70): lambertw(mpf(2), 1)
    check(
        &ctx,
        "W(2, 1)",
        30,
        "-0.834310366631110014694725298171 + 4.53026599855500829213136627984*i",
    );
    // mpmath (dps 50, 70): lambertw(mpc(1, 1), -2)
    check(
        &ctx,
        "LambertW(1 + I, -2)",
        30,
        "-1.97664835818958793205885837604 - 10.0153178860529412753107338178*i",
    );
    // mpmath (dps 50, 70): lambertw(mpc(0, 1))
    check(
        &ctx,
        "W(I)",
        30,
        "0.37469902073711749360597842876 + 0.576412723031435283148289239887*i",
    );
    // Next to the branch point −1/e (a 16-digit rational 1.1·10⁻¹⁷ from it):
    // mpmath (dps 50, 70): lambertw(mpf(-3678794411714423)/10**16, -1)
    check(
        &ctx,
        "W(-3678794411714423/10^16, -1)",
        30,
        "-1.0000000108353791133055811477",
    );
    // At the logarithmic singularity of W₋₁: mpmath (dps 50, 70):
    // lambertw(mpf(10)**-1000, -1)
    check(
        &ctx,
        "W(10^-1000, -1)",
        30,
        "-2310.33023967357255949262583202 - 3.14295304400116447046618890335*i",
    );
}

/// An argument whose imaginary part is only zero to within its error, on
/// the cut of `W₋₁` (`(−∞, 0]`): the side is undecidable and the value is
/// refused, never guessed; off the cut of `W₀` (`(−∞, −1/e]`) it evaluates.
#[test]
fn lambertw_undecidable_side_of_the_cut_is_refused() {
    let ctx = Context::new();
    match dec(&ctx, "W(-1/5 + I*(sin(1)^2 + cos(1)^2 - 1), -1)", 20) {
        Err(SymplexError::PrecisionExhausted { .. }) => {}
        other => panic!("want a refusal, got {other:?}"),
    }
    // mpmath (dps 50, 70): lambertw(mpf(-1)/5) = -0.25917110181907374506
    check(
        &ctx,
        "W(-1/5 + I*(sin(1)^2 + cos(1)^2 - 1))",
        20,
        "-0.25917110181907374506",
    );
}

/// Exact values (SymPy's `LambertW.eval`, and `W_k(w·eʷ) = w` on the real
/// branch that reaches `w`); `lambertw(x, 0)` is the principal-branch
/// node, so no existing canonical form changes.
#[test]
fn lambertw_exact_values_and_canonical_form() {
    let ctx = Context::new();
    let ev = |s: &str| ctx.parse(s).unwrap().eval().to_string();
    assert_eq!(ev("W(0)"), "0");
    assert_eq!(ev("W(0, 3)"), "-oo");
    assert_eq!(ev("W(-exp(-1))"), "-1");
    assert_eq!(ev("W(-exp(-1), -1)"), "-1");
    assert_eq!(ev("W(-2*exp(-2), -1)"), "-2");
    // −2e⁻² is not reached by W₀ at −2 (its values are ≥ −1): left alone.
    assert_eq!(ev("W(-2*exp(-2))"), "W(-2*exp(-2))");
    assert_eq!(ev("W(3*exp(3))"), "3");
    assert_eq!(ev("W(-ln(2)/2)"), "-ln(2)");
    assert_eq!(ev("W(-ln(2)/2, -1)"), "-2*ln(2)");
    assert_eq!(ev("W(-pi/2, -1)"), "-1/2*pi*I");
    let x = ctx.symbol("x");
    assert_eq!(x.lambertw_branch(&ctx.int(0)), x.lambertw());
    assert_eq!(ctx.parse("lambertw(x, 0)").unwrap(), x.lambertw());
    assert_eq!(x.lambertw().to_string(), "W(x)");
}

/// Display, parse, LaTeX, MathML, srepr and the tree of `W(x, k)`; the
/// principal branch prints as before.
#[test]
fn lambertw_branch_output_round_trips() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let w = x.lambertw_branch(&ctx.int(-1));
    assert_eq!(w.to_string(), "W(x, -1)");
    assert_eq!(ctx.parse("W(x, -1)").unwrap(), w);
    assert_eq!(ctx.parse("LambertW(x, -1)").unwrap(), w);
    assert_eq!(w.to_latex(), r"\operatorname{W}_{-1}\left(x\right)");
    assert_eq!(x.lambertw().to_latex(), r"\operatorname{W}\left(x\right)");
    assert_eq!(w.to_srepr(), "LambertW(Symbol('x'), Integer(-1))");
    let mathml = w.to_mathml().unwrap();
    assert!(mathml.contains("<msub><mi>W</mi>"), "{mathml}");
    assert_eq!(ctx.from_tree(&w.to_tree()), w);
}

/// `W_k′ = W_k/(x(1 + W_k))` on every branch (the chain rule applied).
#[test]
fn lambertw_branch_derivative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let x2 = x.powi(2);
    let w = x2.lambertw_branch(&ctx.int(-1));
    let want = 2 * &w / (&x * (&w + 1));
    assert_eq!(w.diff(&x), want);
}

/// `lambertw(x, -1)` compiles (the runtime's new `lambert_wm1`); the
/// other branches are complex on the whole real axis and are refused.
#[test]
fn lambertw_minus_one_compiles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let w = x.lambertw_branch(&ctx.int(-1));
    let f = w.compile(&["x"]).unwrap();
    // mpmath (dps 50, 70): lambertw(mpf(-1)/5, -1) = -2.5426413577735264243
    let v = f.call(&[-0.2]);
    assert!((v + 2.542_641_357_773_526_4).abs() < 1e-14, "{v}");
    let c = w.to_c_fn("f", &["x"]).unwrap();
    assert!(c.contains("return symplex_lambert_wm1(x);"), "{c}");
    assert!(x.lambertw_branch(&ctx.int(2)).to_c_fn("f", &["x"]).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Lambert W in solve
// ═══════════════════════════════════════════════════════════════════════════

/// The real roots of `f` on `[lo, hi]` found by a sign-change scan of its
/// compiled `f64` form (points where `f` is undefined are skipped).
fn sign_changes(f: &Ex, lo: f64, hi: f64) -> usize {
    let g = f.compile(&["x"]).unwrap();
    let n = 40_000;
    let mut prev: Option<f64> = None;
    let mut count = 0;
    for i in 0..=n {
        let t = lo + (hi - lo) * f64::from(i) / f64::from(n);
        let v = g.call(&[t]);
        if !v.is_finite() {
            continue;
        }
        if let Some(p) = prev
            && p * v < 0.0
        {
            count += 1;
        }
        prev = Some(v);
    }
    count
}

/// The largest magnitude among the numbers in an `eval_decimal` string
/// (`"a"`, `"a + b*i"`, `"a - b*i"`, `"b*i"`).
fn max_abs(s: &str) -> f64 {
    s.replace("*i", " ")
        .replace(" + ", " ")
        .replace(" - ", " ")
        .split_whitespace()
        .filter_map(|t| {
            if t == "i" {
                Some(1.0)
            } else {
                t.parse::<f64>().ok()
            }
        })
        .map(f64::abs)
        .fold(0.0, f64::max)
}

/// `solve(f)` returns exactly the real roots `want` (and nothing else
/// real), each certified by substitution, and a sign-change scan finds no
/// other real root.
fn lambert_roots(f_src: &str, lo: f64, hi: f64, want: &[&str]) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(f_src).unwrap();
    let sols = f.solve(&x).unwrap_or_else(|e| panic!("{f_src}: {e:?}"));
    let mut reals: Vec<f64> = Vec::new();
    for s in &sols {
        let v = s
            .eval_decimal(30)
            .unwrap_or_else(|e| panic!("{f_src}: {s}: {e:?}"));
        // Every returned root satisfies the equation.
        let r = f.subs(&x, s).eval_decimal(20).unwrap();
        assert!(max_abs(&r) < 1e-20, "{f_src} at {s}: residual {r}");
        if !v.contains('i') {
            reals.push(v.parse().unwrap());
        }
    }
    reals.sort_by(f64::total_cmp);
    let want: Vec<f64> = want.iter().map(|w| w.parse().unwrap()).collect();
    assert_eq!(reals.len(), want.len(), "{f_src}: {sols:?}");
    for (g, w) in reals.iter().zip(&want) {
        assert!(
            (g - w).abs() <= 1e-14 * w.abs().max(1.0),
            "{f_src}: {g} vs {w}"
        );
    }
    assert_eq!(sign_changes(&f, lo, hi), want.len(), "{f_src}: scan");
}

/// Before: `solve(exp(x) − 2x − π, x)` returned only `−1.45397665972…` —
/// the `W₋₁` root `1.95256170041956046 = −π/2 − W₋₁(−e^{−π/2}/2)` was
/// missing (SymPy 1.14 misses it too: `[-pi/2 - LambertW(-exp(-pi/2)/2)]`).
#[test]
fn solve_lambert_takes_both_real_branches() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("exp(x) - 2*x - pi").unwrap();
    let sols: Vec<String> = f
        .solve(&x)
        .unwrap()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        sols,
        [
            "-1/2*pi - W(-1/2*exp(-1/2*pi))",
            "-1/2*pi - W(-1/2*exp(-1/2*pi), -1)"
        ]
    );
    // mpmath (dps 50, 70): -pi/2 - lambertw(-exp(-pi/2)/2, -1)
    check(
        &ctx,
        "-1/2*pi - W(-1/2*exp(-1/2*pi), -1)",
        30,
        "1.95256170041956046003960541604",
    );
    // mpmath findroot (dps 40) from −1.45 and 1.95
    lambert_roots(
        "exp(x) - 2*x - pi",
        -20.0,
        20.0,
        &["-1.4539766597210248122", "1.95256170041956046"],
    );
}

/// A sample of Lambert-type equations against SymPy 1.14's `solve` (the
/// forms) and mpmath's `findroot` at dps 40 (the real roots, from starting
/// points next to each): every returned root is verified by substitution
/// and a sign-change scan finds no other real root.  SymPy misses the `W₋₁`
/// root of `exp(x) − 3x`, `x·e^{−x} − 1/10`, `exp(x) − x − 2` and
/// `ln x − x + 2` (it keeps `k = −1` only when `LambertW(arg, -1).is_real`
/// is decided symbolically); before 0.30 symplex missed them all, and did
/// not solve `x^x = c` or `x·ln x = c` at all.
#[test]
fn solve_lambert_sample_against_sympy_and_findroot() {
    // SymPy: [LambertW(-1/5), LambertW(-1/5, -1)]
    lambert_roots(
        "x*exp(x) + 1/5",
        -20.0,
        20.0,
        &["-2.5426413577735264243", "-0.25917110181907374506"],
    );
    // SymPy: [LambertW(3)]
    lambert_roots("x*exp(x) - 3", -20.0, 20.0, &["1.04990889496403996"]);
    // SymPy: [7/5 - LambertW(6*exp(14/5)/5)/2]
    lambert_roots(
        "3*exp(2*x) + 5*x - 7",
        -20.0,
        20.0,
        &["0.30210855013732074598"],
    );
    // SymPy: [-LambertW(1)]
    lambert_roots("exp(x) + x", -20.0, 20.0, &["-0.567143290409783873"]);
    // SymPy: [exp(LambertW(log(2)))]
    lambert_roots("x^x - 2", 1e-9, 10.0, &["1.55961046946236935"]);
    // SymPy: [exp(LambertW(-1/5)), exp(LambertW(-1/5, -1))]
    lambert_roots(
        "x*log(x) + 1/5",
        1e-9,
        10.0,
        &["0.078658360286851764532", "0.77169097401769414054"],
    );
    // SymPy: [exp(LambertW(3))]
    lambert_roots("x*log(x) - 3", 1e-9, 10.0, &["2.8573907835143656792"]);
    // SymPy: [exp(LambertW(log(9/10))), exp(LambertW(log(9/10), -1))]
    lambert_roots(
        "x^x - 9/10",
        1e-9,
        10.0,
        &["0.030065368489615144722", "0.88813532882643816359"],
    );
    // SymPy: [-1 + LambertW(6*exp(3))/3]
    lambert_roots(
        "2*exp(-3*x) - x - 1",
        -5.0,
        20.0,
        &["0.17678656808417997665"],
    );
    // SymPy: [-LambertW(-1/3)] (misses 1.5121…)
    lambert_roots(
        "exp(x) - 3*x",
        -20.0,
        20.0,
        &["0.61906128673594511215", "1.5121345516578424739"],
    );
    // SymPy: [-LambertW(-1/10)] (misses 3.5771…)
    lambert_roots(
        "x*exp(-x) - 1/10",
        -5.0,
        40.0,
        &["0.11183255915896296483", "3.5771520639572972184"],
    );
    // SymPy: [-2 - LambertW(-exp(-2))] (misses 1.1461…)
    lambert_roots(
        "exp(x) - x - 2",
        -20.0,
        20.0,
        &["-1.8414056604369606378", "1.1461932206205825852"],
    );
    // SymPy 1.14 solve(log(x) - x + 2): [-LambertW(-exp(-2))] (misses 3.1461…)
    lambert_roots(
        "log(x) - x + 2",
        1e-9,
        20.0,
        &["0.15859433956303936215", "3.1461932206205825852"],
    );
}

/// An argument outside `(−1/e, 0)` keeps the single principal-branch
/// solution, as before (`x·eˣ = −1` has none real: the principal complex
/// root, SymPy's `[LambertW(-1)]`).
#[test]
fn solve_lambert_principal_only_outside_the_real_interval() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = |src: &str| -> Vec<String> {
        ctx.parse(src)
            .unwrap()
            .solve(&x)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect()
    };
    assert_eq!(s("x*exp(x) - 3"), ["W(3)"]);
    assert_eq!(s("x*exp(x) + 1"), ["W(-1)"]);
    assert_eq!(s("x*exp(x) + exp(-1)"), ["-1"]);
}

// ═══════════════════════════════════════════════════════════════════════════
// evalf: kernels that were missing
// ═══════════════════════════════════════════════════════════════════════════

/// Before: "sub-expression e22 not in cache (likely contains free
/// symbols)" for `harmonic(7/2) + 1` and every non-integer
/// `rising_factorial`/`falling_factorial` — evalf had no kernel, and the
/// parent reported its child's missing value instead of the child's error.
#[test]
fn harmonic_and_pochhammer_at_non_integers() {
    let ctx = Context::new();
    // mpmath (dps 50, 70): harmonic(mpf(7)/2)
    check(&ctx, "harmonic(7/2)", 30, "1.96608659126106176211791670946");
    check(
        &ctx,
        "harmonic(7/2) + 1",
        30,
        "2.96608659126106176211791670946",
    );
    // mpmath (dps 50, 70): harmonic(mpf(-1)/2)
    check(
        &ctx,
        "harmonic(-1/2)",
        30,
        "-1.38629436111989061883446424292",
    );
    // Cancellation of ψ(1) = −γ: mpmath (dps 90, 110): harmonic(mpf(1)/10**30)
    check(
        &ctx,
        "harmonic(10^-30)",
        30,
        "1.64493406684822643647241516664e-30",
    );
    // mpmath (dps 50, 70): rf(mpf(1)/3, mpf(5)/2)
    check(
        &ctx,
        "rising_factorial(1/3, 5/2)",
        30,
        "0.643738450059492965699637124855",
    );
    // mpmath (dps 50, 70): 2*ff(mpf(1)/3, mpf(5)/2), 1 + ff(mpf(1)/3, mpf(5)/2)
    check(
        &ctx,
        "2*falling_factorial(1/3, 5/2)",
        30,
        "0.307648653690654959875118419498",
    );
    check(
        &ctx,
        "1 + falling_factorial(1/3, 5/2)",
        30,
        "1.15382432684532747993755920975",
    );
    // Log space: mpmath (dps 50, 70): rf(mpf(10)**20, mpf(1)/2),
    // rf(mpf(1)/3, 10**7), rf(mpf(-7)/2, 10**6)
    check(
        &ctx,
        "rising_factorial(10^20, 1/2)",
        30,
        "9999999999.9999999999875",
    );
    check(
        &ctx,
        "rising_factorial(1/3, 10^7)",
        30,
        "9.67003394751781956028414476959e65657053",
    );
    check(
        &ctx,
        "rising_factorial(-7/2, 10^6)",
        30,
        "3.05973996200947998273196279145e5565682",
    );
    // Poles: mpmath rf(-3, mpf(1)/2) = 0.0; rf(mpf(1)/2, mpf(-3)/2) = +inf.
    check(&ctx, "rising_factorial(-3, 1/2)", 20, "0");
    assert!(dec(&ctx, "rising_factorial(1/2, -3/2)", 20).is_err());
}

/// Before: "Gamma at non-positive integer pole" for `binomial(−7, 2500)`,
/// finite as the limit `(−1)^k·C(k − n − 1, k)` (mpmath's `gammaprod`).
#[test]
fn binomial_at_the_poles_of_gamma() {
    let ctx = Context::new();
    // mpmath (dps 50, 70): binomial(-7, 2500) = 341942019002818626.0
    check(&ctx, "binomial(-7, 2500)", 30, "341942019002818626");
    // mpmath (dps 50, 70): binomial(-7, -9) = 28.0
    check(&ctx, "binomial(-7, -9)", 20, "28");
    // mpmath: binomial(mpf(7)/2, -3) = 0.0; binomial(sqrt(2), -3) = 0.0
    check(&ctx, "binomial(7/2, -3)", 20, "0");
    check(&ctx, "binomial(sqrt(2), -3)", 20, "0");
    // mpmath: binomial(-7, mpf(5)/2) = +inf
    assert!(dec(&ctx, "binomial(-7, 5/2)", 20).is_err());
}

/// Before: "polygamma order must be a non-negative integer" for every
/// order above 10⁴, although `eval` leaves `polygamma(10⁵, 1/2)` symbolic.
#[test]
fn polygamma_of_large_order() {
    let ctx = Context::new();
    // mpmath (dps 50, 70): psi(100000, mpf(1)/2)
    check(
        &ctx,
        "polygamma(100000, 1/2)",
        30,
        "-5.64282217941032707556361125479e486676",
    );
    // mpmath (dps 50, 70): psi(100001, mpf(-7)/5)
    check(
        &ctx,
        "polygamma(100001, -7/5)",
        30,
        "1.76868924482138880751521692432e496373",
    );
    // At a half-integer left of 0 the terms (∓3/2)^{−s}, (∓1/2)^{−s} cancel
    // exactly; mpmath's `psi(100000, mpf(-3)/2)` loses them to cancellation
    // (it prints 1.4·10⁴³⁸⁹⁶⁴), so the reference is the identity
    // ψ⁽ⁿ⁾(−3/2) = −n!·ζ(n + 1, 5/2) for even n: mpmath (dps 50, 70):
    // -factorial(100000)*zeta(100001, mpf(5)/2)
    check(
        &ctx,
        "polygamma(100000, -3/2)",
        30,
        "-1.1274382335477814949538217849e416779",
    );
    // mpmath (dps 50, 70): psi(20000, 10**5), psi(12345, 3)
    check(
        &ctx,
        "polygamma(20000, 10^5)",
        30,
        "-1.00359361823872468633719254645e-22667",
    );
    check(
        &ctx,
        "polygamma(12345, 3)",
        30,
        "9.95422117553492134517139754101e39259",
    );
}

/// A node evalf has no routine for says so, at the root and below it.
#[test]
fn unsupported_nodes_say_so() {
    let ctx = Context::new();
    for s in ["catalan(1/2)", "catalan(1/2) + 1"] {
        match dec(&ctx, s, 20) {
            Err(SymplexError::Unevaluable { reason }) => {
                assert!(
                    reason.contains("no numerical routine for 'catalan'"),
                    "{s}: {reason}"
                );
            }
            other => panic!("{s}: {other:?}"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Real roots of polynomials with large coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// `Σ cᵢ xⁱ` with `cᵢ = ±(7919^(i+13) mod 10^digits)`, signs `(−1)^{i(i+1)/2}`.
fn family_a(ctx: &Context, degree: u32, digits: u32) -> Ex {
    let x = ctx.symbol("x");
    let modulus = num_bigint::BigInt::from(10).pow(digits);
    let mut f = ctx.int(0);
    for i in 0..=degree {
        let c = num_bigint::BigInt::from(7919).pow(i + 13) % &modulus;
        let c = if (i * (i + 1) / 2) % 2 == 1 { -c } else { c };
        f = &f + ctx.from_bigint(c) * x.powi(i64::from(i));
    }
    f
}

/// Before: the Sturm chain stripped the integer content of every remainder
/// (a quadratic binary gcd on thousand-bit coefficients): 2 s for the count
/// of a degree-40 polynomial with 100-digit coefficients in a release
/// build, twice over with the square-free part.  SymPy 1.14:
/// `Poly(...).count_roots()` gives 3, 2 and 4 (57 s for the last, with
/// `intervals()`).
#[test]
fn count_and_isolate_with_large_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (degree, digits, want) in [(25, 30, 3usize), (30, 50, 2), (40, 100, 4)] {
        let f = family_a(&ctx, degree, digits);
        assert_eq!(f.count_real_roots(&x), Some(want), "degree {degree}");
        let ivs = f.real_roots_isolate(&x);
        assert_eq!(ivs.len(), want, "degree {degree}");
        for iv in &ivs {
            assert_eq!(
                f.count_real_roots_in(&x, &iv.lower, &iv.upper),
                Some(1),
                "degree {degree}: {iv:?}"
            );
        }
    }
}

/// Before: the bisection stopped at depth 256 whatever the root bound, and
/// `Π (bᵢx − aᵢ)` with 20-digit `aᵢ`, 10-digit `bᵢ` (Cauchy bound about
/// 2⁶⁶⁰) came back as 2 "isolating" intervals for its 20 roots.
#[test]
fn isolation_under_a_huge_root_bound() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut f = ctx.int(1);
    let mut roots: Vec<Ex> = Vec::new();
    for i in 0u32..20 {
        let a = num_bigint::BigInt::from(3).pow(i + 40) % num_bigint::BigInt::from(10).pow(20);
        let a = if i % 2 == 1 { -a } else { a };
        let b = num_bigint::BigInt::from(7).pow(i + 11) % num_bigint::BigInt::from(10).pow(10) + 1;
        let (a, b) = (ctx.from_bigint(a), ctx.from_bigint(b));
        roots.push(&a / &b);
        f = &f * (&b * &x - &a);
    }
    let f = f.expand();
    let ivs = f.real_roots_isolate(&x);
    // SymPy: len(set(Rational(a, b) ...)) = 20 distinct roots.
    assert_eq!(ivs.len(), 20);
    for r in &roots {
        let inside = ivs.iter().filter(|iv| {
            let lo = (r - &iv.lower).eval_decimal(5).unwrap();
            let hi = (&iv.upper - r).eval_decimal(5).unwrap();
            !lo.starts_with('-') && !hi.starts_with('-')
        });
        assert_eq!(inside.count(), 1, "root {r}");
    }
}
