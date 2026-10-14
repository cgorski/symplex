//! Tests of the 0.50 refactor (b3): open bugs in canon, eval and solve.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn is_no_solution<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::NoSolution { .. }))
}

/// `|e(x = r)| < 1e-9` in complex double precision.
fn assert_numeric_root(e: &Ex, x: &Ex, r: &Ex) {
    let z = e.subs(x, r).eval_complex64().unwrap();
    assert!(z.norm() < 1e-9, "{e} at {x} = {r} is {z}");
}

/// The 30-digit value of `e` is `digits` (mpmath at 30 digits).
fn assert_digits(e: &Ex, digits: &str) {
    assert_eq!(e.eval_decimal(30).unwrap(), digits, "{e}");
}

/// Item 1.  `E` is `exp(1)`, but a product did not merge it with `exp(q)`:
/// `E·exp(−1)` stayed `exp(-1)*E` (also under `simplify`), `E/E` was
/// `exp(-1)*E`, `exp(2)·exp(−2)` was `exp(-2)*exp(2)` and `E·exp(1/2)` was
/// `exp(1/2)*E`.  Oracle: the exact identity `e^a·e^b = e^(a+b)` (SymPy's
/// `Mul.flatten` gathers `E` and `exp(q)` the same way: `E*exp(-1)` is `1`).
/// A symbolic exponent stays apart, as in SymPy (`E*exp(x)`): it is what
/// `expand(exp(x + 1))` gives.
#[test]
fn rational_powers_of_e_merge_in_a_product() {
    let ctx = Context::new();
    assert_eq!(parse(&ctx, "E*exp(-1)").to_string(), "1");
    assert_eq!(parse(&ctx, "E*exp(-1)").simplify().to_string(), "1");
    assert_eq!(parse(&ctx, "E/E").to_string(), "1");
    assert_eq!(parse(&ctx, "exp(2)*exp(-2)").to_string(), "1");
    assert_eq!(parse(&ctx, "E*exp(1/2)").to_string(), "exp(3/2)");
    assert_eq!(parse(&ctx, "2*E*exp(-1)*x").to_string(), "2*x");
    assert_eq!(parse(&ctx, "E*exp(x)").to_string(), "exp(x)*E");
}

/// Item 1, consequences.  With `E/E ≠ 1` and `exp(−3/4)·exp(3/4) ≠ 1`
/// hidden in them: `elliptic_k(E/E)` was not `elliptic_k(1)`;
/// `binomial(−2E/E, 5)` did not evaluate (`precision exhausted`; it is
/// `C(−2, 5) = −6`); and the real solve of
/// `e^(x/2 + 3/4) + e^(−x/2 − 3/4) + 2` returned the non-real `2πi − 3/2`
/// twice (through `√(−4·exp(−3/4)·exp(3/4) + 4)`, a hidden 0).  Oracles:
/// `C(−2, 5) = (−2)(−3)(−4)(−5)(−6)/5! = −6`; `e^u + e^(−u) + 2 =
/// e^(−u)(e^u + 1)²` vanishes only at `e^u = −1`, which no real `u` reaches.
#[test]
fn hidden_ones_of_e_no_longer_block_evaluation_and_solving() {
    let ctx = Context::new();
    assert_eq!(
        parse(&ctx, "binomial(E*(-2)/E, 5)").eval().to_string(),
        "-6"
    );
    let xr = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let e = parse(&ctx, "exp(x/2 + 3/4) + exp(-(x/2 + 3/4)) + 2");
    assert!(is_no_solution(&e.solve(&xr)), "{:?}", e.solve(&xr));
}

/// Item 1, consequence.  `solve_as_set(csc(tanh(x/E)))` was
/// `{2nπei − πei/2} ∪ {2nπei + πei/2}`, the poles of `tanh(x/e)`: the
/// check of a candidate `x = πei/2` met `tanh(πi/2·E·exp(−1))`, which it
/// did not see to be a pole.  Oracle: `csc w = 1/sin w` is never 0 for a
/// finite `w` (`sin` is entire) and is undefined at the poles of `tanh`,
/// so there is no solution (SymPy: `solveset(csc(tanh(x/E)), x)` is
/// `EmptySet`).
#[test]
fn poles_of_an_inner_tanh_are_not_roots_of_csc() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "csc(tanh(x/E))");
    assert_eq!(e.solve_as_set(&x).to_string(), "EmptySet");
}

/// Item 2.  `polylog(3, 1/2)`, `polylog(2, ±i)`, `polylog(2, 2)`,
/// `polylog(2, 1 ± i)`, `polylog(2, (1 ± i)/2)` stayed unevaluated (SymPy
/// folds all but `Li₃(1/2)` and `Li₂((1 + i)/2)`).  Oracles: Lewin
/// (*Polylogarithms and Associated Functions*, ch. 6) `Li₃(1/2) = 7ζ(3)/8 −
/// π² ln 2/12 + ln³2/6`; SymPy's `_dilogtable`; each
/// closed form agrees with `mpmath.polylog` to 1e-31 at 30 digits and
/// 1e-51 at 50 digits (`target/scratch/probe/polylog_oracle.py`); the
/// digits below are mpmath's at 30 digits.
#[test]
fn classical_polylogarithm_values_fold() {
    let ctx = Context::new();
    let cases = [
        (
            "polylog(3, 1/2)",
            "1/6*ln(2)^3 - 1/12*pi^2*ln(2) + 7/8*zeta(3)",
            "0.537213193608040200940623225595",
        ),
        (
            "polylog(2, I)",
            "-1/48*pi^2 + I*Catalan",
            "-0.205616758356028304559051895831 + 0.915965594177219015054603514932*i",
        ),
        (
            "polylog(2, -I)",
            "-1/48*pi^2 - I*Catalan",
            "-0.205616758356028304559051895831 - 0.915965594177219015054603514932*i",
        ),
        (
            "polylog(2, 2)",
            "1/4*pi^2 - ln(2)*pi*I",
            "2.46740110027233965470862274997 - 2.17758609030360213050068889824*i",
        ),
        (
            "polylog(2, 1 - I)",
            "1/16*pi^2 - 1/4*ln(2)*pi*I - I*Catalan",
            "0.616850275068084913677155687492 - 1.46036211675311954767977573949*i",
        ),
        (
            "polylog(2, (1 + I)/2)",
            "-1/8*ln(2)^2 + 5/96*pi^2 - 1/8*ln(2)*pi*I + I*Catalan",
            "0.453985269150295583314241923786 + 0.643767332889268748742017402653*i",
        ),
    ];
    for (src, closed, digits) in cases {
        let v = parse(&ctx, src).eval();
        assert_eq!(v.to_string(), closed, "{src}");
        assert_digits(&v, digits);
    }
}

/// Item 2.  The four golden-ratio dilogarithms of SymPy's `_dilogtable`
/// stayed unevaluated: `Li₂((√5 − 1)/2) = π²/10 − ln²((√5 − 1)/2)`,
/// `Li₂((3 − √5)/2) = π²/15 − ln²((√5 − 1)/2)`, `Li₂((1 − √5)/2) = −π²/15 +
/// ln²((√5 − 1)/2)/2`, `Li₂(−(1 + √5)/2) = −π²/10 − ln²((1 + √5)/2)` (Lewin,
/// ch. 1).  Oracle: mpmath at 30 and 50 digits (to 1e-31 and
/// 1e-51); the digits below are mpmath's `polylog(2, z)` at 30 digits.
/// `Li₂(−1/2)` has no such form (Landen's identity relates it to
/// `Li₂(1/3)` only) and stays.
#[test]
fn golden_ratio_dilogarithms_fold() {
    let ctx = Context::new();
    let cases = [
        (
            "polylog(2, (sqrt(5) - 1)/2)",
            "0.755395619531741469386520028756",
        ),
        (
            "polylog(2, (3 - sqrt(5))/2)",
            "0.426408806162096182092036995427",
        ),
        (
            "polylog(2, (1 - sqrt(5))/2)",
            "-0.542191216450693378340501531043",
        ),
        (
            "polylog(2, -(1 + sqrt(5))/2)",
            "-1.21852526068613025438037817122",
        ),
    ];
    for (src, digits) in cases {
        let v = parse(&ctx, src).eval();
        assert!(!v.to_string().contains("polylog"), "{src} -> {v}");
        assert_digits(&v, digits);
    }
    let stays = parse(&ctx, "polylog(2, -1/2)").eval();
    assert_eq!(stays.to_string(), "polylog(2, -1/2)");
}

/// Item 3.  Equations in `u = a·x + b` with `b ≠ 0` were refused
/// (`ComputationFailed`) where the same equations in `a·x` were solved: the
/// substitutions look for `sin(a·x)`, `e^(a·x)`, … of the unknown itself.
/// The session-12 hunt counted 522 such refusals; the B3 hunt (16 families
/// × 7 slopes × 6 shifts, 3,000 seeds, each beside its unshifted twin)
/// 1,025, all solved now, no wrong root.  Oracle: each root substituted
/// (complex double precision here; the hunt: mpmath at 30 and 60 digits),
/// and the solutions are those of the twin shifted by `−b/a` (`sin(2x) +
/// cos(2x) + 1/2` has `atan(2 ± √7)`; `sinh u = tanh u` at `u = nπi`;
/// `cosh u ∈ {−1, −2}` has no real `u`).
#[test]
fn shifted_arguments_are_solved_in_the_shifted_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "sin(2*x - 1) + cos(2*x - 1) + 1/2");
    let roots = e.solve(&x).unwrap();
    let shown: Vec<String> = roots.iter().map(|r| r.to_string()).collect();
    assert_eq!(
        shown,
        ["atan(-sqrt(7) + 2) + 1/2", "atan(sqrt(7) + 2) + 1/2"]
    );
    for r in &roots {
        assert_numeric_root(&e, &x, r);
    }
    for src in [
        "sinh(2*x + 3) - tanh(2*x + 3)",
        "cosh(2*x + 1)^2 + 3*cosh(2*x + 1) + 2",
        "sin(2*x - 1) + sin(4*x - 2)",
        "sinh(x/2 + 1/3) + 2*tanh(x/2 + 1/3)",
        "cos(x + 1)^2 - 3*cos(x + 1) + 2",
    ] {
        let e = parse(&ctx, src);
        let roots = e.solve(&x).unwrap();
        assert!(!roots.is_empty(), "{src}");
        for r in &roots {
            assert_numeric_root(&e, &x, r);
        }
        let fam = e.solve_general(&x).unwrap();
        for k in -2..=2 {
            for s in fam.instance(k) {
                assert_numeric_root(&e, &x, &s);
            }
        }
    }
    let e = parse(&ctx, "sinh(2*x + 3) - tanh(2*x + 3)");
    assert_eq!(
        e.solve_as_set(&x).to_string(),
        "ImageSet(Lambda(_n, 1/2*_n*pi*I - 3/2), Integers)"
    );
    let ctx = Context::new();
    let xr = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let e = parse(&ctx, "cosh(2*x + 1)^2 + 3*cosh(2*x + 1) + 2");
    assert!(is_no_solution(&e.solve(&xr)), "{:?}", e.solve(&xr));
    let e = parse(&ctx, "sinh(2*x + 3) - tanh(2*x + 3)");
    let real: Vec<String> = e
        .solve(&xr)
        .unwrap()
        .iter()
        .map(|r| r.to_string())
        .collect();
    assert_eq!(real, ["-3/2"]);
}

/// Item 3, the limits of the class: two different shifts
/// (`sin(x + 1) + cos(x + 2)`) or the unknown also outside the forms
/// (`x + sin(x + 1)`) are not one equation in a shifted unknown, and stay
/// refused (no root is claimed).  Oracle: SymPy refuses both as well.
#[test]
fn different_shifts_are_not_one_substitution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["sin(x + 1) + cos(x + 2)", "x + sin(x + 1)"] {
        let r = parse(&ctx, src).solve(&x);
        assert!(
            matches!(r, Err(SymplexError::ComputationFailed { .. })),
            "{src}: {r:?}"
        );
    }
}

/// Item 4 (W3c found-not-fixed).  `solve_general(sinh(x)² + 1)` listed
/// three families, `2nπi + πi/2`, `2nπi − πi/2` and `2nπi + 3πi/2`; the
/// last is the second (`3πi/2 − (−πi/2) = 2πi`, one step), and
/// `solve_general(sin(x)² − 1)` likewise listed `2nπ + 3π/2` beside `2nπ −
/// π/2`.  Oracle: `a + d·n` and `b + d·n` are one set when `(a − b)/d` is an
/// integer; `sinh(±πi/2) = ±i`.
#[test]
fn congruent_families_are_listed_once() {
    for (src, want) in [
        (
            "sinh(x)^2 + 1",
            ["2*n*pi*I + 1/2*pi*I", "2*n*pi*I - 1/2*pi*I"],
        ),
        ("sin(x)^2 - 1", ["2*n*pi + 1/2*pi", "2*n*pi - 1/2*pi"]),
        ("cos(x)^2 - 1", ["2*n*pi", "2*n*pi + pi"]),
    ] {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let e = parse(&ctx, src);
        let fam = e.solve_general(&x).unwrap();
        let shown: Vec<String> = fam.solutions.iter().map(|s| s.to_string()).collect();
        assert_eq!(shown, want, "{src}");
        for k in -2..=2 {
            for s in fam.instance(k) {
                assert_numeric_root(&e, &x, &s);
            }
        }
    }
}

/// Item 4 (W3c found-not-fixed).  `reduce_inequalities` of `e^(ix) + 1 =
/// 0` for an integer `x` was `ℤ ∩ {2nπ + π : n ∈ ℤ}` (and `sin x = 0` was
/// `ℤ ∩ {nπ}`, `tan x = 1` was `ℤ ∩ {nπ + π/4}`), where `solve_as_set` gave
/// `∅`, `{0}`, `∅`.  Oracle: `π` is irrational, so `kπ + c` with rational
/// `c/π` is an integer only for `kπ + c = 0`.
#[test]
fn integer_members_of_families_are_decided_in_reduce_inequalities() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Integer]).unwrap();
    let zero = ctx.int(0);
    for (src, want) in [
        ("exp(I*x) + 1", "EmptySet"),
        ("cos(x) + 1", "EmptySet"),
        ("sin(x)", "{0}"),
        ("tan(x) - 1", "EmptySet"),
    ] {
        let e = parse(&ctx, src);
        let set = reduce_inequalities(&[e.eq_expr(&zero)], &x).unwrap();
        assert_eq!(set.to_string(), want, "{src}");
        assert_eq!(e.solve_as_set(&x).to_string(), want, "{src}");
    }
}
