//! Equation solving, second 0.45 hunt (agent C): Lambert forms whose
//! exponent is a power of the unknown, substitutions whose polynomial has
//! algebraic coefficients, polynomials over fields beyond square roots,
//! Lambert values on the boundary of a domain, and inequalities that use
//! them.  Each test says what was wrong before and cites its oracle: the
//! solution substituted back, SymPy 1.14 (`solve`, `solveset`, `factor`)
//! or mpmath 1.3 (`findroot`, `polyroots`) at 25 digits.

use symplex::num_complex::Complex64;
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn real_x(ctx: &Context) -> Ex {
    ctx.symbol_with("x", &[Assumption::Real]).unwrap()
}

/// `e` at `x = r` is zero: structurally, or below `1e-12` relative to the
/// terms at 16 digits.
fn assert_root(e: &Ex, x: &Ex, r: &Ex) {
    let at = e.subs(x, r).eval();
    if at.is_zero_structural() {
        return;
    }
    let v = at
        .eval_complex64()
        .unwrap_or_else(|err| panic!("{e} at {r}: {at} ({err})"));
    let scale = e
        .expand()
        .args()
        .iter()
        .filter_map(|t| t.subs(x, r).eval_complex64().ok())
        .map(|z| z.norm())
        .fold(1.0f64, f64::max);
    assert!(v.norm() < 1e-12 * scale, "{e} at {r} is {v}");
}

/// The roots of `src`, each checked by substitution.
fn roots(ctx: &Context, x: &Ex, src: &str) -> Vec<Ex> {
    let e = parse(ctx, src);
    let got = e
        .solve(x)
        .unwrap_or_else(|err| panic!("solve({src}): {err}"));
    for r in &got {
        assert_root(&e, x, r);
    }
    got
}

fn numeric(v: &[Ex]) -> Vec<Complex64> {
    v.iter().map(|r| r.eval_complex64().unwrap()).collect()
}

/// The real values among `v`, sorted.
fn real_values(v: &[Ex]) -> Vec<f64> {
    let mut out: Vec<f64> = numeric(v)
        .into_iter()
        .filter(|z| z.im.abs() < 1e-12 * (1.0 + z.re.abs()))
        .map(|z| z.re)
        .collect();
    out.sort_by(f64::total_cmp);
    out
}

fn assert_values(src: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{src}: {got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() < 1e-12 * (1.0 + w.abs()),
            "{src}: {got:?} vs {want:?}"
        );
    }
}

/// `x^a·e^(b·xⁿ) = c` with `n ≥ 2` (and `n = 1/2`) was refused ("not
/// polynomial … transcendental solver could not find solutions"): the
/// power–exponential Lambert path took only exponents linear in `x`.  With
/// `w = xⁿ` the equation raised to the power `n` is `w^a·e^(b·n·w) = cⁿ`,
/// and for `a = 1` each `w` gives `x = c·e^(−b·w)`.  Oracles: SymPy
/// `solve(x*exp(x**2) - 3)` → `[3*exp(-LambertW(18)/2)]`, `solve(x*exp(x**2)
/// - c)` → `[c*exp(-LambertW(2*c**2)/2)]`, `solve(x**2*exp(x**2) - 3)` →
/// `±sqrt(3)*exp(-LambertW(3)/2)` (±1.02465062092600…), `solve(x**3*exp(2*x**2)
/// - 5)` → `[5**(1/3)*exp(-LambertW(4*5**(2/3)/3)/2)]` (0.944007657633738195…);
/// SymPy raises `NotImplementedError` for the others, whose values are
/// mpmath `findroot`: `x·e^(−x²) = 1/4` at 0.268720443153763172… and
/// 1.277044573336516076…, `(x + 1)·e^((x+1)²) = 2` at −0.103949779219887252…,
/// `x²·e^(√x/2) = 2` at 1.089410300203039063….
#[test]
fn lambert_forms_with_a_power_in_the_exponent() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let got = roots(&ctx, &x, "x*exp(x^2) - 3");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].to_string(), "3*exp(-1/2*W(18))");
    {
        // a plain `x`, in a context of its own
        let ctx = Context::new();
        let plain = ctx.symbol("x");
        let got = parse(&ctx, "x*exp(x^2) - c").solve(&plain).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].to_string(), "c*exp(-1/2*W(2*c^2))");
        let c = ctx.symbol("c");
        for k in [3, -2] {
            let e = parse(&ctx, &format!("x*exp(x^2) - ({k})"));
            assert_root(&e, &plain, &got[0].subs(&c, &ctx.int(k)));
        }
        // `x²·e^(x²) = c` is `w·e^w = c` in `w = x²` without squaring: the
        // two roots `±√W(c)`, not also those of `w·e^w = −c` (a parameter
        // leaves nothing to check them against).
        let got = parse(&ctx, "x^2*exp(x^2) - c").solve(&plain).unwrap();
        assert_eq!(got.len(), 2, "{got:?}");
        let e = parse(&ctx, "x^2*exp(x^2) - 3");
        for r in &got {
            assert_root(&e, &plain, &r.subs(&c, &ctx.int(3)));
        }
    }
    for (src, want) in [
        (
            "x^2*exp(x^2) - 3",
            vec![-1.024_650_620_926_001_3, 1.024_650_620_926_001_3],
        ),
        ("x^3*exp(2*x^2) - 5", vec![0.944_007_657_633_738_2]),
        (
            "x*exp(-x^2) - 1/4",
            vec![0.268_720_443_153_763_17, 1.277_044_573_336_516],
        ),
        (
            "(x + 1)*exp((x + 1)^2) - 2",
            vec![-0.103_949_779_219_887_25],
        ),
        ("x^2*exp(sqrt(x)/2) - 2", vec![1.089_410_300_203_039]),
    ] {
        assert_values(src, &real_values(&roots(&ctx, &x, src)), &want);
    }
}

/// `x·ln²x = c` was refused: the logarithmic Lambert forms took `ln x` to
/// the first power only.  Each `r` with `r² = c` gives `√x·ln x = r`, so
/// `x = exp(2·W_k(r/2))` on the real branches.  Oracle: SymPy
/// `solve(x*log(x)**2 - 2)` → `[exp(2*LambertW(-sqrt(2)/2)),
/// exp(2*LambertW(sqrt(2)/2))]` (the first is not real: `−√2/2 < −1/e`), and
/// `solve(x*log(x)**2 - 1/10)` → three real roots, 0.0029442599144069385…,
/// 0.68183623631086466…, 1.3172290092437950….
#[test]
fn squared_logarithm_times_a_power() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let got = roots(&ctx, &x, "x*ln(x)^2 - 2");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].to_string(), "exp(2*W(1/2*sqrt(2)))");
    let src = "x*ln(x)^2 - 1/10";
    assert_values(
        src,
        &real_values(&roots(&ctx, &x, src)),
        &[
            0.002_944_259_914_406_938_5,
            0.681_836_236_310_864_7,
            1.317_229_009_243_795,
        ],
    );
}

/// A substitution whose polynomial has radical coefficients was refused
/// when the solver found only some of its roots (0.44; in 0.43
/// `sin 4x = √3·sin x` gave `[π]`, without `0` and `±π/6`): the root
/// finding over number fields lived above the solver's layer.  Now the
/// polynomial is factored over the field of its coefficients (Trager) and
/// each factor solved (quadratic formula, Cardano, a factor in `t²`, the
/// norm).  Oracle: SymPy `factor(expand_trig(sin(4*x) - sqrt(3)*sin(x)),
/// extension=sqrt(3))` → `-8*(sin(x)**2*cos(x) - cos(x)/2 + sqrt(3)/8)*sin(x)`,
/// i.e. `sin x·(2cos x − √3)·(4cos²x + 2√3·cos x + 1)` up to a constant, whose
/// last factor has no real zero (discriminant `12 − 16 < 0`): the real
/// solutions are `nπ` and `±π/6 + 2nπ`.  SymPy's `solve` gives `[0, pi]` and
/// `solveset(…, Interval(-4, 4))` a `ConditionSet`.
#[test]
fn trigonometric_substitution_over_a_number_field() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let src = "sin(4*x) - sqrt(3)*sin(x)";
    let pi = std::f64::consts::PI;
    assert_values(
        src,
        &real_values(&roots(&ctx, &x, src)),
        &[-pi / 6.0, 0.0, pi / 6.0, pi],
    );
    let set = parse(&ctx, src).solve_as_set(&x);
    for member in ["0", "7*pi", "pi/6 + 2*pi", "-pi/6 - 4*pi", "11*pi/6"] {
        assert_eq!(
            set.contains(&parse(&ctx, member)),
            Some(true),
            "{member} in {set}"
        );
    }
    for other in ["pi/3", "5*pi/6", "1"] {
        assert_eq!(
            set.contains(&parse(&ctx, other)),
            Some(false),
            "{other} in {set}"
        );
    }
}

/// More substitutions with radical coefficients, refused before: in
/// `t = tan(x/2)`, `sin 3x = √2·sin x` (`sin x·(4cos²x − 1 − √2)`, roots `0`,
/// `π`, `±0.681089818242874006…`, `±2.460502835346919232…`, mpmath
/// `findroot`); in `t = cos x`, `cos³x − √2·cos x + 1/4 = 0`, whose cubic has
/// the three real roots −1.269318…, 0.180967389455538259…,
/// 1.088351… (mpmath `polyroots`), of which only the middle one is a
/// cosine: `x = ±acos(0.18096…)`; in `t = eˣ`, `e^(3x) − √2·eˣ = 1/4` at
/// 0.238480198144593906… and `(e^(3x) − 3)/(eˣ + 1) = √2` at
/// 0.655099738060320463… (mpmath `findroot`).
#[test]
fn substitutions_with_radical_coefficients() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let src = "sin(3*x) - sqrt(2)*sin(x)";
    let pi = std::f64::consts::PI;
    let (a, b) = (0.681_089_818_242_874, 2.460_502_835_346_919);
    assert_values(
        src,
        &real_values(&roots(&ctx, &x, src)),
        &[-b, -a, 0.0, a, b, pi],
    );
    let src = "cos(x)^3 - sqrt(2)*cos(x) + 1/4";
    let c = 0.180_967_389_455_538_26_f64.acos();
    assert_values(src, &real_values(&roots(&ctx, &x, src)), &[-c, c]);
    let src = "exp(3*x) - sqrt(2)*exp(x) - 1/4";
    assert_values(
        src,
        &real_values(&roots(&ctx, &x, src)),
        &[0.238_480_198_144_593_9],
    );
    let src = "(exp(3*x) - 3)/(exp(x) + 1) - sqrt(2)";
    assert_values(
        src,
        &real_values(&roots(&ctx, &x, src)),
        &[0.655_099_738_060_320_5],
    );
}

/// Polynomials whose coefficients generate a field beyond square roots
/// (`∛2`), or with an irreducible quintic factor over `ℚ(√2)`, were refused
/// ("roots were not found in closed form").  Oracles: the construction
/// `(x − ∛2)·(x − i)²` (two distinct roots) and `(x − ∛2)·(x² − √3)` (three:
/// `∛2`, `±3^(1/4)`); mpmath `polyroots([1, 0, 0, 0, sqrt(2), 1])` for
/// `x⁵ + √2·x + 1`: −0.634429162285747458…, 0.906243619705781110 ±
/// 0.794590256304714116i, −0.589029038562907381 ± 0.859132825640299824i.
#[test]
fn polynomials_over_fields_beyond_square_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "(x - 2^(1/3))*(x - I)^2").expand();
    let got = e.solve(&x).unwrap();
    assert_eq!(got.len(), 2, "{got:?}");
    for want in ["2^(1/3)", "I"] {
        let w = parse(&ctx, want);
        assert!(
            got.iter().any(|g| (g - &w).eval().is_zero_structural()),
            "{want} in {got:?}"
        );
    }
    let e = parse(&ctx, "(x - 2^(1/3))*(x^2 - sqrt(3))").expand();
    let got = e.solve(&x).unwrap();
    for r in &got {
        assert_root(&e, &x, r);
    }
    let r4 = 3f64.powf(0.25);
    assert_values(
        "(x - ∛2)(x² − √3)",
        &real_values(&got),
        &[-r4, 2f64.cbrt(), r4],
    );
    let e = parse(&ctx, "x^5 + sqrt(2)*x + 1");
    let got = e.solve(&x).unwrap();
    for r in &got {
        assert_root(&e, &x, r);
    }
    let want = [
        Complex64::new(-0.634_429_162_285_747_5, 0.0),
        Complex64::new(0.906_243_619_705_781, 0.794_590_256_304_714_1),
        Complex64::new(0.906_243_619_705_781, -0.794_590_256_304_714_1),
        Complex64::new(-0.589_029_038_562_907_4, 0.859_132_825_640_299_8),
        Complex64::new(-0.589_029_038_562_907_4, -0.859_132_825_640_299_8),
    ];
    let got = numeric(&got);
    assert_eq!(got.len(), 5, "{got:?}");
    for w in want {
        assert!(got.iter().any(|g| (g - w).norm() < 1e-12), "{w} in {got:?}");
    }
}

/// A Lambert root that is exactly on the boundary of the declared domain
/// was kept: `2·2ˣ + 4x − 2 = 0` for a positive `x` gave `[1/2 −
/// W(√2·ln(2)/2)/ln 2]`, and `(5/2)ˣ = (x + 2)²/4` for a negative `x`
/// included `−2·W(−2·ln(5/2)/5)/ln(5/2) − 2`, both `0` — no evaluation
/// decides their sign.  The `W` values are now recognised exactly:
/// `W₀(√2·ln(2)/2) = ln(2)/2` since `(ln(2)/2)·e^(ln(2)/2) = √2·ln(2)/2` and
/// `ln(2)/2 ≥ −1` (the definition of `W₀`), and `W₀(−2·ln(5/2)/5) =
/// −ln(5/2)` likewise.  Oracle: substitution (`2·2⁰ + 0 − 2 = 0`,
/// `(5/2)⁰ = 2²/4`), and mpmath `findroot` for the negative root
/// −2.606045369336472695….
#[test]
fn lambert_values_on_the_domain_boundary() {
    let ctx = Context::new();
    let pos = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let e = parse(&ctx, "2*2^x + 4*x - 2");
    assert!(e.solve(&pos).unwrap().is_empty());
    assert_eq!(e.solve_as_set(&pos).to_string(), "EmptySet");
    let ctx = Context::new();
    let real = real_x(&ctx);
    let got = parse(&ctx, "2*2^x + 4*x - 2").solve(&real).unwrap();
    assert_eq!(got.len(), 1);
    assert!(got[0].is_zero_structural(), "{got:?}");
    let ctx = Context::new();
    let neg = ctx.symbol_with("x", &[Assumption::Negative]).unwrap();
    let src = "(5/2)^x - (x + 2)^2/4";
    assert_values(
        src,
        &real_values(&roots(&ctx, &neg, src)),
        &[-2.606_045_369_336_473],
    );
}

/// A constant root whose sign no evaluation decides because it is `0` to
/// every precision was returned for a positive unknown: `x − atan(1/2) −
/// atan(1/3) + π/4` (`atan(1/2) + atan(1/3) = π/4`, Euler's identity, so
/// the root is `0`) gave `[−π/4 + atan(1/2) + atan(1/3)]`.  Now refused,
/// with the reason; the set keeps the undecided intersection.  Oracle: the
/// identity `tan(atan(1/2) + atan(1/3)) = (1/2 + 1/3)/(1 − 1/6) = 1`.
#[test]
fn undecided_zero_member_is_refused() {
    let ctx = Context::new();
    let pos = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let e = parse(&ctx, "x - atan(1/2) - atan(1/3) + pi/4");
    match e.solve(&pos) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("cannot decide"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
    let set = e.solve_as_set(&pos).to_string();
    assert!(set.starts_with("(0, oo) ∩"), "{set}");
}

/// An equation in `x²` only (beyond the single power and exponential of
/// the Lambert forms) was refused: `(x² + 1)·e^(x²) = 3` is `(w + 1)·e^w =
/// 3` in `w = x²`, and `x⁴ = 2^(x²)` is `w² = 2^w` (`w = 2, 4` and a
/// negative `w` that no real `x` reaches).  Oracles: SymPy
/// `solve((x**2 + 1)*exp(x**2) - 3)` → `[-sqrt(-1 + LambertW(3*E)), sqrt(-1 +
/// LambertW(3*E))]` (±0.785902326485978987…, mpmath `findroot`); SymPy raises
/// `NotImplementedError` for `x**4 - 2**(x**2)`, whose real roots `±√2`,
/// `±2` substitute exactly (`4 = 2²`, `16 = 2⁴`).
#[test]
fn equations_in_a_power_of_the_unknown() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let src = "(x^2 + 1)*exp(x^2) - 3";
    let r = 0.785_902_326_485_979;
    assert_values(src, &real_values(&roots(&ctx, &x, src)), &[-r, r]);
    let src = "x^4 - 2^(x^2)";
    let s2 = std::f64::consts::SQRT_2;
    assert_values(
        src,
        &real_values(&roots(&ctx, &x, src)),
        &[-2.0, -s2, s2, 2.0],
    );
}

/// The solution set of `sin(x²) = 1/2` for a real `x` held imaginary
/// numbers: its families `±√(2nπ + π/6)`, `±√(5π/6 + 2nπ)` over all
/// integers `n` were taken for real progressions (their members at `n = 0,
/// 1` are real), and the real line dropped from `ℝ ∩ {…}`, so
/// `i·√(11π/6)` (`n = −1`) was a member (SymPy 1.14 `solveset(sin(x**2) -
/// 1/2, x, S.Reals)` gives the same four image sets over all integers).
/// Oracle: `(i·√(11π/6))² = −11π/6`, whose sine is `1/2`: a root, but not
/// real; `√(13π/6)` is a real root.
#[test]
fn real_solution_set_of_a_family_with_imaginary_members() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let e = parse(&ctx, "sin(x^2) - 1/2");
    let set = e.solve_as_set(&x);
    let imaginary = parse(&ctx, "I*sqrt(11*pi/6)");
    assert_eq!(set.contains(&imaginary), Some(false), "{set}");
    for member in ["sqrt(pi/6)", "-sqrt(13*pi/6)", "sqrt(5*pi/6)"] {
        assert_eq!(
            set.contains(&parse(&ctx, member)),
            Some(true),
            "{member} in {set}"
        );
    }
    // over ℂ (a plain symbol, in a context of its own) it is a root
    let ctx = Context::new();
    let plain = ctx.symbol("x");
    let over_c = parse(&ctx, "sin(x^2) - 1/2").solve_as_set(&plain);
    let imaginary = parse(&ctx, "I*sqrt(11*pi/6)");
    assert_eq!(over_c.contains(&imaginary), Some(true), "{over_c}");
}

/// `x·e^(x²) − 1 > 0` was refused (its zero was not found), and once the
/// zero is found the sign chart's sample point `1001` took seconds: the
/// value `1001·e^(1001²)`, about `3.2·10^435166`, became an exact rational
/// whose `gcd` ran in the realness test.  Oracle: `x·e^(x²)` is increasing
/// (derivative `(1 + 2x²)·e^(x²) > 0`), so the set is `(x₀, ∞)` with `x₀ =
/// e^(−W(2)/2)`, 0.652918640419204715… (SymPy `solve(x*exp(x**2) - 1)`
/// → `[exp(-LambertW(2)/2)]`).
#[test]
fn inequality_with_a_lambert_monomial_zero() {
    let ctx = Context::new();
    let x = real_x(&ctx);
    let e = parse(&ctx, "x*exp(x^2) - 1");
    let t0 = std::time::Instant::now();
    let set = e.solve_gt(&x);
    assert_eq!(set.to_string(), "(exp(-1/2*W(2)), oo)");
    assert!(t0.elapsed().as_secs_f64() < 5.0, "{:?}", t0.elapsed());
    assert_eq!(set.contains(&ctx.int(1)), Some(true));
    assert_eq!(set.contains(&ctx.rational(13, 20)), Some(false));
    assert_eq!(set.contains(&ctx.rational(131, 200)), Some(true));
}
