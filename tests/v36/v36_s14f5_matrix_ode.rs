//! Tests of session 14 (s14f5): matrix function performance, ODE Lambert branches.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `y′ − rhs` for `y(x)`.
fn ode(ctx: &Context, rhs: &str) -> (Ex, Ex, Ex) {
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &y.formal_diff(&x) - &parse(ctx, rhs);
    (e, x, y)
}

fn decimal(e: &Ex, digits: u32) -> f64 {
    e.eval_decimal(digits).unwrap().parse::<f64>().unwrap()
}

/// P1.  `matrix_exp`, `matrix_sqrt`, `matrix_log`, `matrix_pow_symbolic` of
/// the 6×6 unitary matrix of sweep B's seed 775 (over `ℚ(√2, √3, i)`) took
/// 8.9 s (release, one core of a c7i) where 0.48 had taken 6.3 s, and
/// seed 360's `singular_values` refusal 8.4 s instead of 6.7 s.  The
/// profile blamed the allocator (6 % → 30 % of the time, 35 M → 194 M
/// allocations), but the code had not changed: the builds of 0.48 for the
/// comparison resolved num-bigint 0.4.8, whose `BigUint` keeps a one-digit
/// value inline, while the repository's (untracked) lock file pinned
/// 0.4.6, which allocates for every one-digit number (`Ratio::one()`, each
/// clone of a small coefficient).  With 0.4.8 the same functions take
/// 5.6–6.6 s and 31 M allocations, the outputs byte for byte the same (the
/// 200,000-case golden output is unchanged).  `Cargo.toml` now requires
/// num-bigint 0.4.8; `BigInt::ONE` exists from 0.4.7 on, so this test stops
/// compiling if the requirement is lowered.  A timing is not tested here
/// (a debug build takes minutes for these matrices).
#[test]
fn num_bigint_keeps_one_digit_values_inline() {
    const ONE: num_bigint::BigInt = num_bigint::BigInt::ONE;
    assert_eq!(ONE, num_bigint::BigInt::from(1));
}

/// R2.  `y′ = 1/(2x + y + 2)`, `y(5/2) = 2/3` failed ("could not solve
/// −1/2·W(−2·C1·e^(−11)) − 49/6 = 0 for C1"): the general solution was
/// the principal-branch family `−2x − W(−2·C1·e^(−4x−1))/2 − 5/2`, and
/// `W₀ ≥ −1` never equals `−49/3`.  The initial value lies on the lower
/// branch: `y = −2x − W₋₁(−49/3·e^(−4x−19/3))/2 − 5/2`.  SymPy 1.14's
/// `dsolve` gives only the principal family and its `ics` fit fails
/// ("Couldn't solve for initial conditions").  Oracle: mpmath
/// `odefun(lambda t, v: 1/(2t + v + 2), 5/2, 2/3)` at 30 and 60 digits,
/// `y(3) = 0.727743525658916633126132832260…`, `y(4) =
/// 0.830380061830792835309701266733…`; `y(5/2) = 2/3` exactly since
/// `W₋₁(w·eʷ) = w` for `w ≤ −1`.
#[test]
fn lambert_ivp_on_the_lower_branch() {
    let ctx = Context::new();
    let (e, x, y) = ode(&ctx, "1/(2*x + y + 2)");
    let ic = InitialCondition {
        order: 0,
        x: ctx.rational(5, 2),
        value: ctx.rational(2, 3),
    };
    let sol = e.solve_ode_ivp(&y, &x, &[ic]).unwrap();
    assert!(e.check_ode_solution(&sol, &y, &x), "{sol}");
    assert_eq!(
        sol.subs(&x, &ctx.rational(5, 2)).eval(),
        ctx.rational(2, 3),
        "{sol}"
    );
    for (at, want) in [
        (3, 0.727_743_525_658_916_6_f64),
        (4, 0.830_380_061_830_792_8),
    ] {
        let v = decimal(&sol.subs(&x, &ctx.int(at)), 25);
        assert!((v - want).abs() < 1e-15, "y({at}) = {v}: {sol}");
    }
}

/// R2.  An explicit family through the principal Lambert W carries its
/// twin through `W₋₁`: for `−1/e ≤ z < 0` both branches are real and both
/// solve the equation (`W′ = W/(z·(1 + W))` on every branch); before, only
/// the principal family was returned, so a real solution on the lower
/// branch had no constant.  Oracle: mpmath at 30 and 60 digits, at `x = 0`,
/// `C1 = 1/10` (`z = −e⁻¹/5`): `−W₀(z)/2 − 5/2 = −2.46016091974426174…`,
/// `−W₋₁(z)/2 − 5/2 = −0.50284582649893895…`; `check_ode_solution` proves
/// each family.
#[test]
fn lambert_families_carry_both_real_branches() {
    let ctx = Context::new();
    let (e, x, y) = ode(&ctx, "1/(2*x + y + 2)");
    let fams = e.solve_ode_all(&y, &x).unwrap();
    assert_eq!(fams.len(), 2, "{fams:?}");
    let c1 = ctx.symbol("C1");
    let mut values = Vec::new();
    for f in &fams {
        assert!(e.check_ode_solution(f, &y, &x), "{f}");
        let at = f.subs(&x, &ctx.int(0)).subs(&c1, &ctx.rational(1, 10));
        values.push(decimal(&at, 25));
    }
    assert!(
        (values[0] + 2.460_160_919_744_261_7).abs() < 1e-15,
        "{values:?}"
    );
    assert!(
        (values[1] + 0.502_845_826_498_938_9).abs() < 1e-15,
        "{values:?}"
    );
    // The first family is the one `solve_ode` returned before.
    assert_eq!(fams[0], e.solve_ode(&y, &x));
}

/// R2.  `y′ = (3y − x)/(x + y)`: `check_ode_solution(C1·e^(W(2x/C1)) + x)`
/// was `false` although it is a solution, and since 0.51 (stricter family
/// verification) `solve_ode` fell back to the implicit `−C1 + 2/(y/x − 1) −
/// ln|x| − ln|y/x − 1|` (0.50 returned the explicit family).  Oracle (exact
/// identity): with `z = 2x/C1` and `e^(W(z)) = z/W(z)` (every branch solves
/// `W·e^W = z`), `y′ = 1 + 2/(1 + W) = (3 + W)/(1 + W)` and `(3y − x)/(x + y)
/// = (6x/W + 2x)/(2x/W + 2x) = (3 + W)/(1 + W)`.  A wrong family
/// (`C1·e^(W(2x/C1)) + 2x`: `y′ = (4 + 2W)/(1 + W)` against `(6 + 5W)/(2 + 3W)`)
/// is still rejected.  SymPy 1.14's `dsolve` hits its recursion limit on this
/// equation.  The lower-branch twin, mpmath at 30 and 60 digits at `x = 1`,
/// `C1 = −6`: `−6·e^(W₋₁(−1/3)) + 1 = −0.32263362265433443…`.
#[test]
fn homogeneous_lambert_solution_verifies() {
    let ctx = Context::new();
    let (e, x, y) = ode(&ctx, "(3*y - x)/(x + y)");
    let good = parse(&ctx, "C1*exp(lambertw(2*x/C1)) + x");
    assert!(e.check_ode_solution(&good, &y, &x));
    let lower = parse(&ctx, "C1*exp(lambertw(2*x/C1, -1)) + x");
    assert!(e.check_ode_solution(&lower, &y, &x));
    let wrong = parse(&ctx, "C1*exp(lambertw(2*x/C1)) + 2*x");
    assert!(!e.check_ode_solution(&wrong, &y, &x));
    let sol = e.solve_ode(&y, &x);
    assert_eq!(sol, good, "{sol}");
    let fams = e.solve_ode_all(&y, &x).unwrap();
    assert_eq!(fams, vec![good, lower.clone()]);
    let c1 = ctx.symbol("C1");
    let at = lower.subs(&x, &ctx.int(1)).subs(&c1, &ctx.int(-6));
    assert!((decimal(&at, 25) + 0.322_633_622_654_334_43).abs() < 1e-15);
}

/// R2.  The exact equation `y·e^(xy) + (x·e^(xy) + 1)·y′ = 0` has the
/// families `C1 − W_k(x·e^(C1·x))/x`, `k = 0, −1`; `check_ode_solution`
/// returned `false` for the principal one (only the principal one was
/// returned).  Oracle (exact identity): with `z = x·e^(C1·x)` and
/// `e^(−W) = W/z`, `e^(xy) = e^(C1·x)·e^(−W) = W/x`, so `y·e^(xy) =
/// C1·W/x − W²/x²` and `(x·e^(xy) + 1)·y′ = (W + 1)·y′ = (W² −
/// C1·x·W)/x²`, which cancel.
#[test]
fn exact_equation_lambert_families_verify() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e =
        &(&y * &parse(&ctx, "exp(x*y)")) + &(&parse(&ctx, "x*exp(x*y) + 1") * &y.formal_diff(&x));
    let fams = e.solve_ode_all(&y, &x).unwrap();
    let want = [
        parse(&ctx, "C1 - lambertw(x*exp(C1*x))/x"),
        parse(&ctx, "C1 - lambertw(x*exp(C1*x), -1)/x"),
    ];
    assert_eq!(fams, want);
    for f in &fams {
        assert!(e.check_ode_solution(f, &y, &x), "{f}");
    }
}

/// R2.  `W₋₁(f) = c` was refused by `solve` ("transcendental solver could
/// not find solutions"), so an initial value could not be fitted on a
/// lower-branch family; it is inverted as `W₀` is, `f = c·e^c`, and the
/// candidate is checked against the branch's range.  Oracle (exact):
/// `W₋₁(w·eʷ) = w` for real `w ≤ −1`, so `W₋₁(x) = −2` has the root
/// `−2e⁻²` (mpmath `lambertw(-2*exp(-2), -1) = -2.0` at 30 and 60
/// digits); `1` is not a value of `W₋₁` (the candidate `e` gives mpmath
/// `lambertw(e, -1) = −0.532092121986379905… − 4.597158013302573277…i`),
/// no solution.  SymPy 1.14 raises NotImplementedError for both.
///
/// The inversion decides `W_k(c·e^c) = c` itself, since a factor of a
/// product and a candidate with parameters are not checked later (a first
/// version of this fix gave these wrong roots, found by the golden-output
/// diff): `W₋₁(x) = −asin(1/π)` has none (`W₋₁` is real only `≤ −1`; the
/// candidate gives `W₋₁ = −1.5286…`, mpmath), `W₁(x + 1/4) = 0` none
/// (`W_k(0) = −∞` for `k ≠ 0`), so `laguerre(−1/2, y)·W₁(x + 1/4)` has
/// none, and `x·W₋₁(x)` none (`0·W₋₁(0) = 0·(−∞)`, a pole, not a
/// root), while `x·W₋₁(x − G)` (Catalan's `G`) has the root 0 and
/// `W₋₁(x)·(x + Ai(y))` only `−Ai(y)`.  `cosh(W₋₁(x)) = 0` has the root
/// `−π/2` (`W₋₁(−π/2) = −iπ/2`; mpmath `lambertw(-pi/2, -1)` =
/// `-1.5707963267948966192j`), and `W₋₁(x) = y` is refused (a root only
/// for `y` in the range of `W₋₁`).
#[test]
fn solve_inverts_the_lower_lambert_branch() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let rs = parse(&ctx, "lambertw(x, -1) + 2").solve(&x).unwrap();
    assert_eq!(rs, vec![parse(&ctx, "-2*exp(-2)")]);
    for src in [
        "lambertw(x, -1) - 1",
        "lambertw(x, -1) + acsc(pi)",
        "laguerre(-1/2, y)*lambertw(x + 1/4, 1)",
        "x*lambertw(x, -1)",
        "x*lambertw(x, 1)",
    ] {
        let r = parse(&ctx, src).solve(&x);
        assert!(
            matches!(r, Err(SymplexError::NoSolution { .. })),
            "{src}: {r:?}"
        );
    }
    let roots = |src: &str| parse(&ctx, src).solve(&x).unwrap();
    assert_eq!(roots("x*lambertw(x - Catalan, -1)"), vec![ctx.int(0)]);
    assert_eq!(
        roots("lambertw(x, -1)*(x + airyai(y))"),
        vec![parse(&ctx, "-airyai(y)")]
    );
    assert_eq!(roots("cosh(lambertw(x, -1))"), vec![parse(&ctx, "-pi/2")]);
    let r = parse(&ctx, "lambertw(x, -1) - y").solve(&x);
    assert!(
        matches!(r, Err(SymplexError::ComputationFailed { .. })),
        "{r:?}"
    );
}

/// `W_k(0) = −∞` for a branch `k ≠ 0` is folded when the node is built, as
/// `eval` folded it, so `0·W₁(0)` is `0·(−∞)`, undefined: up to 0.51
/// `(x·W₁(x))` at `x = 0` was `0` (the product folded `0·W₁(0)` before
/// `W₁(0)` was evaluated).  SymPy 1.14: `LambertW(0, 1)` → `-oo`,
/// `(x*LambertW(x, 1)).subs(x, 0)` → `nan`.
#[test]
fn lambert_branches_at_zero_are_infinite_when_built() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for k in ["1", "-1", "5"] {
        let w = parse(&ctx, &format!("lambertw(0, {k})"));
        assert_eq!(w.to_string(), "-oo", "k = {k}");
    }
    let e = parse(&ctx, "x*lambertw(x, 1)").subs(&x, &ctx.int(0));
    assert_eq!(e.eval().to_string(), "nan");
}
