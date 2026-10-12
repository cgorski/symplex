//! Transcendental equations: the session-10 hunt for needless refusals in
//! `solve` / `solve_as_set` (Lambert W forms, multiple-angle tangents,
//! hyperbolic functions, equations over a polynomial denominator).  Each
//! test says what was refused or wrong before and cites its oracle: SymPy
//! 1.14 (`solve`, `solveset`), the solution substituted back, and a real
//! root scan where completeness is claimed.

use symplex::num_complex::Complex64;
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `e` vanishes at `x = r` (exactly, or `|value| < 1e-12` numerically).
fn assert_root(e: &Ex, x: &Ex, r: &Ex) {
    let at = e.subs(x, r).eval();
    if at.is_zero_structural() {
        return;
    }
    let v = at
        .eval_complex64()
        .unwrap_or_else(|err| panic!("{e} at {r}: {at} ({err})"));
    assert!(v.norm() < 1e-12, "{e} at {r} is {v}");
}

/// The roots of `src` over `x`, each checked by substitution, with their
/// numerical values.
fn roots(ctx: &Context, x: &Ex, src: &str) -> Vec<(String, Complex64)> {
    let e = parse(ctx, src);
    let found = e
        .solve(x)
        .unwrap_or_else(|err| panic!("{src}: refused: {err}"));
    found
        .iter()
        .map(|r| {
            assert_root(&e, x, r);
            (r.to_string(), r.eval_complex64().unwrap())
        })
        .collect()
}

/// The real values among `got` equal `want` (as sets, to 1e-9).
fn assert_real_roots(src: &str, got: &[(String, Complex64)], want: &[f64]) {
    let real: Vec<f64> = got
        .iter()
        .filter(|(_, z)| z.im.abs() < 1e-12)
        .map(|(_, z)| z.re)
        .collect();
    assert_eq!(real.len(), want.len(), "{src}: {got:?}");
    for w in want {
        assert!(
            real.iter().any(|r| (r - w).abs() < 1e-9 * (1.0 + w.abs())),
            "{src}: {w} missing from {got:?}"
        );
    }
}

fn shown(got: &[(String, Complex64)]) -> Vec<String> {
    let mut s: Vec<String> = got.iter().map(|(s, _)| s.clone()).collect();
    s.sort();
    s
}

/// `x^a = b^x` and `x^a·e^(b·x) = c` were refused ("not polynomial … could
/// not find solutions") before 0.44.  SymPy 1.14 `solve(x**2 - 2**x, x)` →
/// `[2, 4, -2*LambertW(log(2)/2)/log(2)]`; `solve(x**2*exp(x) - 3, x)` →
/// `[2*LambertW(-sqrt(3)/2), 2*LambertW(sqrt(3)/2)]` (the first complex:
/// `−√3/2 < −1/e`, so `W₋₁` is not real and not taken); `solve(x**3 -
/// 3**x, x)` → `[3, -3*LambertW(-log(3)/3)/log(3)]` (`W₋₁(−ln 3/3) = −ln 3`
/// gives `3`).
#[test]
fn power_equals_exponential_by_lambert_w() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let got = roots(&ctx, &x, "x^2 - 2^x");
    assert_eq!(shown(&got), ["-2*W(1/2*ln(2))/ln(2)", "2", "4"]);
    assert_real_roots("x^2 - 2^x", &got, &[2.0, 4.0, -0.766_664_695_962_123]);

    let got = roots(&ctx, &x, "x^2*exp(x) - 3");
    assert_eq!(shown(&got), ["2*W(-1/2*sqrt(3))", "2*W(1/2*sqrt(3))"]);
    assert_real_roots("x^2*exp(x) - 3", &got, &[1.033_230_903_717_66]);

    let got = roots(&ctx, &x, "x^3 - 3^x");
    assert_eq!(shown(&got), ["-3*W(-1/3*ln(3))/ln(3)", "3"]);
    assert_real_roots("x^3 - 3^x", &got, &[3.0, 2.478_052_680_288_30]);

    // x^n = c·e^x: x·e^(−x/2) = ±1/√3, both branches of W at −1/(2√3),
    // W₀ at +1/(2√3) (SymPy `solve(x**(-2)*exp(x) - 3)`: 0.9100, −0.4590,
    // 3.7331).
    let got = roots(&ctx, &x, "x^(-2)*exp(x) - 3");
    assert_real_roots(
        "x^(-2)*exp(x) - 3",
        &got,
        &[
            0.910_007_572_488_709,
            -0.458_962_267_536_949,
            3.733_079_028_632_81,
        ],
    );
    // A rational power: √x = 2ˣ/4 (SymPy: −W(−ln 2/8)/(2 ln 2) and the
    // W₋₁ branch, 0.06875 and 2.72245).
    let got = roots(&ctx, &x, "sqrt(x) - 2^x/4");
    assert_real_roots(
        "sqrt(x) - 2^x/4",
        &got,
        &[0.068_749_814_389_850_6, 2.722_453_777_305_10],
    );
}

/// The Lambert solutions respect the declared domain of `x` (0.43's rule):
/// SymPy with `Symbol('x', positive=True)` gives `[2, 4]` for `x² = 2ˣ`; a
/// negative `x` the third root alone.  For a real `x` the complex
/// `2·W(−√3/2)` of `x²·eˣ = 3` is dropped, and `x²·eˣ = −3` (no real root:
/// `x²·eˣ ≥ 0`) is the empty set; refused before 0.44.
#[test]
fn lambert_roots_respect_assumptions() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert_eq!(shown(&roots(&ctx, &x, "x^2 - 2^x")), ["2", "4"]);

    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Negative]).unwrap();
    let got = roots(&ctx, &x, "x^2 - 2^x");
    assert_real_roots("x^2 - 2^x", &got, &[-0.766_664_695_962_123]);
    assert_eq!(got.len(), 1);

    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    assert_eq!(
        shown(&roots(&ctx, &x, "x^2*exp(x) - 3")),
        ["2*W(1/2*sqrt(3))"]
    );
    assert_eq!(
        parse(&ctx, "x^2 - 2^x").solve_as_set(&x).to_string(),
        "{2, 4, -2*W(1/2*ln(2))/ln(2)}"
    );
    assert_eq!(
        parse(&ctx, "x^2*exp(x) + 3").solve_as_set(&x).to_string(),
        "EmptySet"
    );
    // Over ℂ (a plain x) the set of a Lambert equation is not the finite
    // list of real branches: it stays unknown.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(
        parse(&ctx, "x^2 - 2^x")
            .solve_as_set(&x)
            .to_string()
            .starts_with("ConditionSet")
    );
}

/// A linear polynomial times an exponential, also when expanded, and a
/// common power factor: refused before 0.44.  SymPy: `solve((x + 1)*exp(x)
/// - 2)` and `solve(x*exp(x) + exp(x) - 2)` → `[-1 + LambertW(2*E)]`;
/// `solve(x**2*2**x - x**2/10)` → `[0, -log(10)/log(2)]`;
/// `solve(x**(-2)*exp(-x) - exp(2)/4)` → `[-2, 2*LambertW(exp(-1))]`, of
/// which a negative `x` keeps `−2` (`W(−1/e) = −1` at the branch point,
/// where the argument came out as `−√(4e⁻²)/2`: it took 15 s and gave an
/// unevaluable `W` before the branch point was recognised).
#[test]
fn linear_base_and_merged_exponential_terms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["(x + 1)*exp(x) - 2", "x*exp(x) + exp(x) - 2"] {
        assert_eq!(shown(&roots(&ctx, &x, src)), ["W(2*E) - 1"], "{src}");
    }
    assert_eq!(
        shown(&roots(&ctx, &x, "x^2*2^x - x^2/10")),
        ["-ln(10)/ln(2)", "0"]
    );
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Negative]).unwrap();
    assert_eq!(shown(&roots(&ctx, &x, "x^(-2)*exp(-x) - exp(2)/4")), ["-2"]);
}

/// Logarithmic forms: a power of `x` with `ln x`, `ln` over a power, and
/// `ln` of a shifted linear argument; refused before 0.44.  SymPy:
/// `solve(x**2*log(x) - 1)` → `[exp(LambertW(2)/2)]`; `solve(log(x)/x**2 -
/// 1/10)` → `[exp(-LambertW(-1/5)/2), exp(-LambertW(-1/5, -1)/2)]`;
/// `solve(log(4*x - 2) - 2*x + 2)` → `[1/2 - LambertW(-exp(-1)/2)/2, 1/2 -
/// LambertW(-exp(-1)/2, -1)/2]`; `solve(log(x)/x - 1/4)` → `[-4*LambertW(-1/4),
/// -4*LambertW(-1/4, -1)]`.
#[test]
fn logarithmic_lambert_forms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(shown(&roots(&ctx, &x, "x^2*ln(x) - 1")), ["exp(1/2*W(2))"]);
    assert_eq!(
        shown(&roots(&ctx, &x, "ln(x)/x^2 - 1/10")),
        ["exp(-1/2*W(-1/5))", "exp(-1/2*W(-1/5, -1))"]
    );
    let got = roots(&ctx, &x, "ln(4*x - 2) - 2*x + 2");
    assert_real_roots(
        "ln(4*x - 2) - 2*x + 2",
        &got,
        &[0.615_980_476_493_267, 1.839_173_495_008_33],
    );
    assert_eq!(
        shown(&roots(&ctx, &x, "ln(x)/x - 1/4")),
        ["-4*W(-1/4)", "-4*W(-1/4, -1)"]
    );
}

/// `x^x = c` at integers: `exp(W(ln 27))` is `3` (SymPy `[3]`), `x^x = 1`
/// also at `0` (`0⁰ = 1`; SymPy `[0, 1]`), and the negative integers where
/// `x^x` is real, which no real branch of `W` reaches and SymPy misses:
/// `(−1)^(−1) = −1`, `(−2)^(−2) = 1/4` (SymPy returns only the complex
/// `exp(LambertW(I*pi))`, `exp(LambertW(-log(4)))`).  Before 0.44 `x^x + 1`
/// gave `[]` for a real `x` and `x^x − 1` missed `0`.
#[test]
fn self_power_integer_roots() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    assert_eq!(shown(&roots(&ctx, &x, "x^x - 27")), ["3"]);
    assert_eq!(shown(&roots(&ctx, &x, "x^x - 1")), ["0", "1"]);
    assert_eq!(shown(&roots(&ctx, &x, "x^x + 1")), ["-1"]);
    assert_eq!(shown(&roots(&ctx, &x, "x^x - 1/4")), ["-2"]);
}

/// `tan(k·x)` with `k ≥ 2` was refused (`expand_trig` leaves `tan 2x`
/// alone, and the half-angle substitution then saw an atom it could not
/// rewrite).  `tan 2x = tan x` holds exactly at `x = nπ`, away from the
/// poles `π/4 + nπ/2` of `tan 2x` and `π/2 + nπ` of `tan x`: SymPy
/// `solveset(tan(2*x) - tan(x), x, S.Reals)` → `{2nπ} ∪ {2nπ + π}`, `solve`
/// → `[0]` (it misses `π`, as for `sin 2x = sin x`).
#[test]
fn tangent_of_multiple_angles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "tan(2*x) - tan(x)");
    assert_eq!(shown(&roots(&ctx, &x, "tan(2*x) - tan(x)")), ["0", "pi"]);
    let set = e.solve_as_set(&x);
    assert_eq!(set.to_string(), "ImageSet(Lambda(_n, _n*pi), Integers)");
    assert_eq!(set.contains(&(ctx.pi() * 5)), Some(true));
    assert_eq!(set.contains(&(ctx.pi() / 4)), Some(false));
    for (src, set) in [
        (
            "tan(2*x) - 2*tan(x)",
            "ImageSet(Lambda(_n, _n*pi), Integers)",
        ),
        (
            "tan(2*x) + tan(x)",
            "ImageSet(Lambda(_n, 1/3*_n*pi), Integers)",
        ),
        ("tan(3*x) - tan(x)", "ImageSet(Lambda(_n, _n*pi), Integers)"),
    ] {
        assert_eq!(parse(&ctx, src).solve_as_set(&x).to_string(), set, "{src}");
    }
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert_eq!(shown(&roots(&ctx, &x, "tan(2*x) - tan(x)")), ["pi"]);
}

/// A substitution took the roots the solver found of its polynomial in `t`
/// as all of them: with radicals in the coefficients it finds some only
/// (degree above 2 over `ℚ(√p)`), and `sin 4x = √3·sin x` (in `t =
/// tan(x/2)` a degree-9 polynomial over `ℚ(√3)`) gave `[π]` and the set
/// `{2nπ + π}`, without `0` (`sin 0 = 0`) and the roots `±acos(…)`; likewise
/// `sin 3x = √2·sin x`.  Wrong before 0.44; now refused as incomplete
/// (SymPy 1.14 `solveset(sin(4*x) - sqrt(3)*sin(x), x, S.Reals)` contains
/// `{2nπ}`).
#[test]
fn incomplete_substituted_polynomial_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    for src in ["sin(4*x) - sqrt(3)*sin(x)", "sin(3*x) - sqrt(2)*sin(x)"] {
        let e = parse(&ctx, src);
        assert_root(&e, &x, &ctx.int(0));
        assert!(
            matches!(e.solve(&x), Err(SymplexError::ComputationFailed { .. })),
            "{src}: {:?}",
            e.solve(&x)
        );
        let set = e.solve_as_set(&x);
        assert_ne!(set.contains(&ctx.int(0)), Some(false), "{src}: {set}");
    }
}

/// Equations in several hyperbolic functions were refused (only a single
/// `sinh x = c`, `cosh x = c` was inverted); rewritten in exponentials they
/// are polynomial in `eˣ`.  SymPy: `solve(sinh(x) + cosh(x) - 3)` →
/// `[log(3)]`, `solve(sinh(2*x) - sinh(x))` → `[0, -I*pi/3, I*pi/3, I*pi]`,
/// `solve(cosh(2*x) - 3*cosh(x) + 2)` → `[0, -I*pi/3, I*pi/3]`.  For a real
/// `x` the complex roots go; `sinh x − 3·cosh x = 3` (`−eˣ − 2e⁻ˣ = 3`, no
/// real root) is no solution.
#[test]
fn hyperbolic_equations_through_exponentials() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(shown(&roots(&ctx, &x, "sinh(x) + cosh(x) - 3")), ["ln(3)"]);
    let got = roots(&ctx, &x, "sinh(2*x) - sinh(x)");
    assert_real_roots("sinh(2*x) - sinh(x)", &got, &[0.0]);
    let got = roots(&ctx, &x, "cosh(2*x) - 3*cosh(x) + 2");
    assert_real_roots("cosh(2*x) - 3*cosh(x) + 2", &got, &[0.0]);
    let set = parse(&ctx, "sinh(2*x) - sinh(x)").solve_as_set(&x);
    let i_pi = parse(&ctx, "I*pi");
    assert_eq!(set.contains(&i_pi), Some(true));
    assert_eq!(set.contains(&(&i_pi / 3)), Some(true));

    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    assert_eq!(shown(&roots(&ctx, &x, "sinh(2*x) - sinh(x)")), ["0"]);
    let e = parse(&ctx, "sinh(x) - 3*cosh(x) - 3");
    assert!(matches!(e.solve(&x), Err(SymplexError::NoSolution { .. })));
    assert_eq!(e.solve_as_set(&x).to_string(), "EmptySet");
}

/// Equations whose exponential factors or numerator have no real root were
/// refused rather than answered: `(e^(x/2) + 1)·(e^(x/2) + 2) = 0` and
/// `eˣ/x + 1/(3x) = 0` (numerator `eˣ + 1/3`).  For a real `x` both are
/// empty (SymPy `solveset(…, x, S.Reals)` → `EmptySet`).
#[test]
fn exponential_equations_without_real_roots() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    for src in ["(exp(x/2) + 1)*(exp(x/2) + 2)", "exp(x)/x + 1/(3*x)"] {
        let e = parse(&ctx, src);
        assert!(
            matches!(e.solve(&x), Err(SymplexError::NoSolution { .. })),
            "{src}"
        );
        assert_eq!(e.solve_as_set(&x).to_string(), "EmptySet", "{src}");
    }
}

/// No closed form: `x·eˣ − x − 1 = 0` (SymPy raises `NotImplementedError`,
/// "multiple generators [x, exp(x)]") stays a refusal and an unknown set,
/// though it has real roots; and the reference `2ˣ − x − 2` (0.39) keeps
/// its two roots.
#[test]
fn no_closed_form_stays_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "x*exp(x) - x - 1");
    assert!(matches!(
        e.solve(&x),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert!(e.solve_as_set(&x).to_string().starts_with("ConditionSet"));
    let got = roots(&ctx, &x, "2^x - x - 2");
    assert_real_roots("2^x - x - 2", &got, &[2.0, -1.690_093_067_619_31]);
}
