//! Equation solving, inequalities and sets: the 0.39 hunt for silent wrong
//! answers.  Each test says what was wrong before and cites its oracle: the
//! solution substituted back (exact or certified evaluation), SymPy 1.14
//! (`solve`, `solveset`, `reduce_inequalities`), or the principal-branch
//! definition `z^(p/q) = exp((p/q)·Log z)` with `Log` the principal
//! logarithm (`−π < Im Log z ≤ π`).

use symplex::num_complex::Complex64;
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `e` at `x = r` is exactly zero (structurally, or `|value| < 1e-12` at
/// 16 digits for a residual in radicals).
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

fn numeric(v: &[Ex]) -> Vec<Complex64> {
    v.iter().map(|r| r.eval_complex64().unwrap()).collect()
}

fn same_set(got: &[Complex64], want: &[(f64, f64)]) -> bool {
    got.len() == want.len()
        && want.iter().all(|&(re, im)| {
            got.iter()
                .any(|g| (g - Complex64::new(re, im)).norm() < 1e-9 * (1.0 + g.norm()))
        })
}

/// `solve` takes principal branches, and when every principal candidate
/// was a pole it reported "no solution" (`SymplexError::NoSolution`, "the
/// equation is provably unsatisfiable") though other members of the
/// periodic family solve the equation: `tan(x)/x = 0` at `x = π`
/// (`tan π/π = 0`), `(cos x − 1)/x = 0` and `(1 − cos x)/x² = 0` at `2π`,
/// `sin x/(x(x − π)) = 0` at `2π`, `(e^(ix) − 1)/x = 0` at `2π`.  Now
/// `ComputationFailed` naming such a member (SymPy 1.14 `solve` returns
/// `[]` for `tan(x)/x`, `[2*pi]` for `(1 - cos(x))/x**2`).  An equation all
/// of whose family members are poles keeps `NoSolution`: `(1 − cos x)/sin x`
/// is `0/0` on every `2nπ` (SymPy `solveset(…, S.Reals)` → `EmptySet`).
#[test]
fn principal_poles_do_not_prove_there_is_no_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, member) in [
        ("tan(x)/x", "pi"),
        ("(cos(x) - 1)/x", "2*pi"),
        ("(1 - cos(x))/x^2", "2*pi"),
        ("sin(x)/(x*(x - pi))", "2*pi"),
        ("(exp(I*x) - 1)/x", "2*pi"),
    ] {
        let e = parse(&ctx, src);
        assert_root(&e, &x, &parse(&ctx, member));
        match e.solve(&x) {
            Err(SymplexError::ComputationFailed { reason, .. }) => {
                assert!(reason.contains("other branches"), "{src}: {reason}");
            }
            other => panic!("{src}: expected a refusal, got {other:?}"),
        }
    }
    let e = parse(&ctx, "(1 - cos(x))/sin(x)");
    assert!(matches!(e.solve(&x), Err(SymplexError::NoSolution { .. })));
    // sin(x)/x keeps its principal root π (SymPy: [pi]).
    let roots = parse(&ctx, "sin(x)/x").solve(&x).unwrap();
    assert_eq!(roots.len(), 1);
    assert_root(&parse(&ctx, "sin(x)/x"), &x, &roots[0]);
}

/// `solve_general` returned families containing poles: `sin(x)/x = 0` gave
/// `2nπ` and `2nπ + π`, whose member `n = 0` is `0` (`sin 0/0` is
/// undefined; SymPy: `solveset(sin(x)/x, x, S.Reals)` =
/// `Complement(ImageSet(Lambda(n, 2*n*pi), Integers), {0}) ∪ …`), and a
/// family over all integers cannot exclude it — now refused; likewise
/// `sin(x)·ln x` (`0·ln 0`), `tan(x)/x`, `sin(x)/(x² − π²)` (members `±π`),
/// and `sin(x)/(x − 10π)`, whose bad member `n = 5` lies outside the
/// sampled ones (found from the pole `x = 10π`).  When the equation is
/// periodic the bad members are periodic too, and the family is split
/// exactly: `sin(3x)/sin(x) = 0` gives `π/3, 2π/3, 4π/3, 5π/3 + 2nπ`
/// (before: `2nπ/3` and `2nπ/3 + π/3`, with the poles `nπ`), every
/// instance a root (substitution), and they are all the zeros in `[−7, 7]`
/// (SymPy `solveset(sin(3*x)/sin(x), x, Interval(-7, 7))`: the 8 points
/// `kπ/3`, `3 ∤ k`, `|k| ≤ 5`).
#[test]
fn general_families_do_not_contain_poles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "sin(x)/x",
        "sin(x)*ln(x)",
        "tan(x)/x",
        "sin(x)/(x^2 - pi^2)",
        "sin(x)/(x - 10*pi)",
    ] {
        match parse(&ctx, src).solve_general(&x) {
            Err(SymplexError::ComputationFailed { reason, .. }) => {
                assert!(reason.contains("not a solution"), "{src}: {reason}");
            }
            other => panic!("{src}: expected a refusal, got {other:?}"),
        }
    }
    let e = parse(&ctx, "sin(3*x)/sin(x)");
    let fam = e.solve_general(&x).unwrap();
    assert_eq!(fam.solutions.len(), 4, "{:?}", fam.solutions);
    let mut members: Vec<f64> = Vec::new();
    for k in -3..=3 {
        for s in fam.instance(k) {
            assert_root(&e, &x, &s);
            let v = s.eval_f64().unwrap();
            if v.abs() <= 7.0 {
                members.push(v);
            }
        }
    }
    members.sort_by(f64::total_cmp);
    let want: Vec<f64> = [-5, -4, -2, -1, 1, 2, 4, 5]
        .iter()
        .map(|&k| f64::from(k) * std::f64::consts::PI / 3.0)
        .collect();
    assert_eq!(members.len(), want.len(), "{members:?}");
    for (g, w) in members.iter().zip(&want) {
        assert!((g - w).abs() < 1e-12, "{members:?}");
    }
}

/// `x^(p/q) = c` takes the roots `u` of `u^p = c` and `x = u^q`; only `u`
/// in the principal sector `−π/q < arg u ≤ π/q` give solutions, which a
/// numeric check decided — and could not, at a candidate equal to `−1`
/// written as `√3/2·i^(5/3) + i^(2/3)/4 + 3/4·i^(8/3)`, so `x^(3/2) = i`
/// returned it besides `i^(2/3)` (principal power: `(−1)^(3/2) =
/// e^(3πi/2) = −i`).  The sector is now decided exactly for a Gaussian
/// rational `c` whose argument is a rational multiple of `π`.  SymPy:
/// `solve(x**Rational(3, 2) - I)` → `[(sqrt(3)/2 + I/2)**2]`.  The other
/// cases, by the same definition: `x^(5/2) = −i` has `e^(−iπ/5)` and
/// `e^(3iπ/5)`; `x^(2/5) = −1 + i` none (`arg x^(2/5) ∈ (−2π/5, 2π/5]`);
/// `x^(−2/3) = i` has `e^(−3iπ/4)`.
#[test]
fn rational_powers_keep_the_principal_sector() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    let cis = |t: f64| (t.cos(), t.sin());
    for (src, want) in [
        ("x^(3/2) - I", vec![cis(pi / 3.0)]),
        ("x^(5/2) + I", vec![cis(-pi / 5.0), cis(3.0 * pi / 5.0)]),
        ("x^(-2/3) - I", vec![cis(-3.0 * pi / 4.0)]),
        (
            "x^(3/4) - 1 - I",
            vec![(0.7937005259840998, 1.3747296369986026)],
        ),
    ] {
        let e = parse(&ctx, src);
        let roots = e.solve(&x).unwrap();
        for r in &roots {
            assert_root(&e, &x, r);
        }
        assert!(same_set(&numeric(&roots), &want), "{src}: {roots:?}");
    }
    assert!(matches!(
        parse(&ctx, "x^(2/5) + 1 - I").solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
}

/// `exp(√x) = i`: `√x = ln i + 2nπi`, and the principal `√` reaches only
/// `iπ/2 + 2nπi` with `n ≥ 0` — the member `n = −1`, `x = −9π²/4`, gives
/// `exp(√x) = e^(3πi/2) = −i`.  `solve_general` kept the whole family
/// `(2nπi + ln i)²`: `ln i` was left unevaluated and the residual at the
/// member sat on the branch cut of `√`, undecided.  Now `ln i = iπ/2`
/// (principal logarithm of a Gaussian rational), the member is rejected and
/// the family refused; `solve` gives `−π²/4` (`√(−π²/4) = iπ/2`).
/// `exp(ix) = i` gives `π/2` (it was `−ln(i)·i`), as SymPy.
#[test]
fn logarithms_of_gaussian_rationals_are_folded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "exp(sqrt(x)) - I");
    let roots = e.solve(&x).unwrap();
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0], parse(&ctx, "-pi^2/4"));
    assert!(matches!(
        e.solve_general(&x),
        Err(SymplexError::ComputationFailed { .. })
    ));
    let e = parse(&ctx, "exp(I*x) - I");
    assert_eq!(e.solve(&x).unwrap(), vec![parse(&ctx, "pi/2")]);
    let fam = e.solve_general(&x).unwrap();
    for k in -2..=2 {
        for s in fam.instance(k) {
            assert_root(&e, &x, &s);
        }
    }
}

/// A rational equation whose numerator and denominator share an
/// irreducible factor of degree ≥ 3: the roots of the factor are poles
/// (`0/0`), but substituting a nested-radical or `RootOf` root into the
/// denominator did not give a recognisable zero, and they were returned as
/// roots: `(x⁴ − 2x³ − 3x² + 7x − 2)/(x³ − 3x + 1)` (`= x − 2` off the
/// poles) gave `2` and the three roots of the cubic; `(x⁶ − 2x⁵ − x² + x +
/// 2)/(x⁵ − x − 1)` gave `2` and five `RootOf`s.  The common factors are
/// now divided out exactly (polynomial gcd).  SymPy: `solve(…)` → `[2]`
/// for both; `(x² − 1)/(x − 1) = 2` has no solution (`[]`; was refused).
#[test]
fn rational_equations_drop_common_irrational_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "(x^4 - 2*x^3 - 3*x^2 + 7*x - 2)/(x^3 - 3*x + 1)",
        "(x^6 - 2*x^5 - x^2 + x + 2)/(x^5 - x - 1)",
    ] {
        assert_eq!(
            parse(&ctx, src).solve(&x).unwrap(),
            vec![ctx.int(2)],
            "{src}"
        );
    }
    assert!(matches!(
        parse(&ctx, "(x^2 - 1)/(x - 1) - 2").solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
    // The equation atom of `reduce_inequalities` used the same roots.
    let c = ctx
        .parse_bool("(x^4 - 2*x^3 - 3*x^2 + 7*x - 2)/(x^3 - 3*x + 1) == 0")
        .unwrap();
    assert_eq!(
        c.solve_for(&x).unwrap().as_finite_set(),
        Some(vec![ctx.int(2)])
    );
}

/// Rational equations with parameters or irrational coefficients returned
/// roots of the numerator that are poles: `(x² − a²)/(x − a)` gave
/// `±√(a²)`, `(x² − (a + b)x + ab)/(x − a)` both `a` and `b`,
/// `1/(x − a) − 1/(x² − a²)` the pole `a` besides `1 − a`, and
/// `(2x² + (2 + 2√2)x + 2√2)/(x² + (√2 − 1 − √3)x − √6 − √2)` (=
/// `2(x + 1)/(x − 1 − √3)` off the pole `−√2`) a nested-radical form of
/// `−√2` besides `− 1`.  Common factors are now divided out over
/// `ℚ[x, parameters]` (multivariate gcd), and a root of a factor is tested
/// against the other factors' denominators exactly.  SymPy 1.14 `solve`:
/// `[-a]`, `[b]`, `[1 - a]`, `[]` for `(x − a)²/(x² − a²)`, `[-1]`, and
/// `[]` for `(x − √3)/(x² + (1 − √3)x − √3)`.
#[test]
fn rational_equations_with_parameters_do_not_return_poles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, want) in [
        ("(x^2 - a^2)/(x - a)", "-a"),
        ("(x^2 - a*x - b*x + a*b)/(x - a)", "b"),
        ("1/(x - a) - 1/(x^2 - a^2)", "1 - a"),
    ] {
        let roots = parse(&ctx, src).solve(&x).unwrap();
        assert_eq!(roots, vec![parse(&ctx, want)], "{src}");
    }
    // The root comes as `−√2/2 + √(12 − 8√2)/4 − 1/2`, which is `−1`.
    let e = parse(
        &ctx,
        "(2*x^2 + 2*x + 2*sqrt(2)*x + 2*sqrt(2))/(x^2 - sqrt(3)*x - x + sqrt(2)*x - sqrt(6) - sqrt(2))",
    );
    let roots = e.solve(&x).unwrap();
    assert!(same_set(&numeric(&roots), &[(-1.0, 0.0)]), "{roots:?}");
    assert_root(&e, &x, &roots[0]);
    for src in [
        "(x^2 - 2*a*x + a^2)/(x^2 - a^2)",
        "(x - sqrt(3))/(x^2 - sqrt(3)*x + x - sqrt(3))",
    ] {
        assert!(
            matches!(
                parse(&ctx, src).solve(&x),
                Err(SymplexError::NoSolution { .. })
            ),
            "{src}"
        );
    }
}

/// `reduce_inequalities` took the principal solutions of an equation atom
/// as its solution set: `sin x = 0 ∧ −4 < x < 4` gave `{0, π}` without
/// `−π`, and `sin x = 0` alone `{0, π}` (SymPy 1.14's `reduce_inequalities`
/// gives the same wrong sets).  A periodic equation is now
/// `{x | sin x = 0}`, and the members of its families inside the bounded
/// hull of the other conditions are listed (each checked, so the pole `0`
/// of `sin(x)/x` is left out).  Oracle: substitution, and SymPy
/// `solveset(sin(x), x, Interval.open(-4, 4))` → `{-pi, 0, pi}`.
#[test]
fn periodic_equations_in_reduce_inequalities_are_complete() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    let cases: &[(&str, &[f64])] = &[
        ("sin(x) == 0 & x > -4 & x < 4", &[-pi, 0.0, pi]),
        (
            "tan(x) == 1 & x > -4 & x < 4",
            &[-3.0 * pi / 4.0, pi / 4.0, 5.0 * pi / 4.0],
        ),
        (
            "sin(x)/x == 0 & x > -7 & x < 7",
            &[-2.0 * pi, -pi, pi, 2.0 * pi],
        ),
        (
            "cos(x) == 1/2 & x >= -pi/3 & x <= 5*pi/3",
            &[-pi / 3.0, pi / 3.0, 5.0 * pi / 3.0],
        ),
        (
            "(sin(x) + cos(x) == 1 | tan(x)/x == 0) & x > -2 & x < 2",
            &[0.0, pi / 2.0],
        ),
    ];
    for (src, want) in cases {
        let set = ctx.parse_bool(src).unwrap().solve_for(&x).unwrap();
        let pts = set
            .as_finite_set()
            .unwrap_or_else(|| panic!("{src}: {set}"));
        let mut got: Vec<f64> = pts.iter().map(|p| p.eval_f64().unwrap()).collect();
        got.sort_by(f64::total_cmp);
        assert_eq!(got.len(), want.len(), "{src}: {set}");
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-12, "{src}: {set}");
        }
    }
    // Unbounded: the exact condition set.
    let set = ctx
        .parse_bool("sin(x) == 0")
        .unwrap()
        .solve_for(&x)
        .unwrap();
    assert_eq!(set.contains(&parse(&ctx, "5*pi")), Some(true));
    assert_eq!(set.contains(&parse(&ctx, "-7*pi")), Some(true));
    assert_eq!(set.contains(&ctx.int(1)), Some(false));
}

/// `sin(x)/cos(x) − tan(x) = 0` holds wherever it is defined, but the
/// substitution `t = tan(x/2)` turned it into `0 = 0` and only the value
/// `x = π` checked apart from it came back: `solve` gave `[π]` and
/// `solve_general` the family `π + 2nπ`, missing e.g. `x = 1`
/// (`sin 1/cos 1 − tan 1 = 0`).  Now both refuse.
#[test]
fn an_identity_in_the_half_angle_tangent_is_not_a_root_list() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "sin(x)/cos(x) - tan(x)");
    assert_root(&e, &x, &ctx.int(1));
    assert!(e.solve(&x).is_err());
    assert!(e.solve_general(&x).is_err());
}

/// Radicals of a linear expression, and sums of two square roots, were
/// refused ("not polynomial … transcendental solver could not find
/// solutions").  SymPy 1.14: `solve(sqrt(x + 1) - (x - 1))` → `[3]` (the
/// candidate `0` of the squared equation fails: `√1 = 1 ≠ −1`),
/// `solve(x - 2*sqrt(x - 1))` → `[2]`, `solve(sqrt(x + 5) + sqrt(x) - 3)`
/// → `[4/9]`, `solve(sqrt(x**2 + 1) - x - 2)` → `[-3/4]`,
/// `solve(sqrt(x)*sqrt(x + 1) - 2)` → `[-1/2 + sqrt(17)/2]`; and
/// `x^(1/3)` principal: `cbrt(x + 1) = x − 1` has the one real root of
/// `t³ − t − 2` cubed minus one (mpmath `findroot`: 2.52137970680456…),
/// the complex `t` being outside `−π/3 < arg t ≤ π/3`.
#[test]
fn radical_equations_are_solved_and_extraneous_roots_rejected() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: &[(&str, &[(f64, f64)])] = &[
        ("sqrt(x + 1) - (x - 1)", &[(3.0, 0.0)]),
        ("x - 2*sqrt(x - 1)", &[(2.0, 0.0)]),
        ("sqrt(x + 5) + sqrt(x) - 3", &[(4.0 / 9.0, 0.0)]),
        ("sqrt(x^2 + 1) - x - 2", &[(-0.75, 0.0)]),
        ("sqrt(x)*sqrt(x + 1) - 2", &[(1.5615528128088303, 0.0)]),
        ("cbrt(x + 1) - x + 1", &[(2.521379706804567, 0.0)]),
        (
            "sqrt(2*x - 4) - (2*x - 1)",
            &[(0.75, 0.82915619758885), (0.75, -0.82915619758885)],
        ),
    ];
    for (src, want) in cases {
        let e = parse(&ctx, src);
        let roots = e.solve(&x).unwrap_or_else(|err| panic!("{src}: {err}"));
        for r in &roots {
            assert_root(&e, &x, r);
        }
        assert!(same_set(&numeric(&roots), want), "{src}: {roots:?}");
    }
}

/// A cubic with symbolic coefficients that factors over `ℤ[x, a]` was
/// refused ("coefficients not all in ℚ(√p…, i)"); it is factored first,
/// as SymPy does: `solve(expand(a*(x - a)*(x - a - 1)*(x + 1)), x)` →
/// `[-1, a, a + 1]`.  And the radical inequality `√(x + 1) ≥ x − 1`, whose
/// sign chart needs the zero `3` of `√(x + 1) − x + 1`, was refused;
/// SymPy `reduce_inequalities(sqrt(x + 1) >= x - 1)` (real `x`) →
/// `(-1 <= x) & (x <= 3)`, and `< ` → `3 < x`.
#[test]
fn parametric_cubics_and_radical_inequalities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "a^3*x + a^3 - 2*a^2*x^2 - a^2*x + a^2 + a*x^3 - a*x");
    let mut roots: Vec<String> = e.solve(&x).unwrap().iter().map(|r| r.to_string()).collect();
    roots.sort();
    assert_eq!(roots, vec!["-1", "a", "a + 1"]);
    let ge = ctx
        .parse_bool("sqrt(x + 1) >= x - 1")
        .unwrap()
        .solve_for(&x)
        .unwrap();
    assert_eq!(ge.to_string(), "[-1, 3]");
    let lt = ctx
        .parse_bool("sqrt(x + 1) < x - 1")
        .unwrap()
        .solve_for(&x)
        .unwrap();
    assert_eq!(lt.to_string(), "(3, oo)");
}

/// `a^x = b·x + c` with a constant base was refused; it is
/// `exp(x·ln a) = b·x + c`, a Lambert W form with both real branches
/// (`−ln 2/3 ∈ (−1/e, 0)`): `−W₀(−ln 2/3)/ln 2 = 0.457822373232055` and
/// `−W₋₁(−ln 2/3)/ln 2 = 3.313178380475635` (mpmath `lambertw`; SymPy
/// 1.14 `solve(2**x - 3*x)` returns the first only).
#[test]
fn constant_base_exponentials_reach_lambert_w() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "2^x - 3*x");
    let roots = e.solve(&x).unwrap();
    for r in &roots {
        assert_root(&e, &x, r);
    }
    assert!(same_set(
        &numeric(&roots),
        &[(0.457822373232055, 0.0), (3.313178380475635, 0.0)]
    ));
}
