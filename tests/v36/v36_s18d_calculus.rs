//! Tests of session 18. Lane D of the 0.60.1 block: calculus (inverse Mellin transforms, `diff_with_dependent`,
//! internal symbols of the limit engine, one-sided limits, `check_ode_solution`,
//! `probably_equal`).
//! Every test says what was wrong before and cites its oracle.

use symplex::prelude::*;

// ── W26: `diff_with_dependent` by something that is not a symbol ─────────────

/// A derivative with respect to a number, a product or a sum is not defined.  Up to 0.60.0
/// `diff_with_dependent` returned `0` for it — a value — where `diff` returns the formal
/// `Derivative(…, var)` and `try_diff` an error.  Now the three agree: `diff_with_dependent`
/// returns the same unevaluated node as `diff`.
///
/// Oracle: SymPy `diff(x*y + sin(x), 2)` raises `ValueError: First variable cannot be a number:
/// 2`, `diff(x*y + sin(x), 2*x)`: `ValueError: Can't calculate derivative wrt 2*x.`
#[test]
fn diff_with_dependent_by_a_non_symbol_is_the_refusal_of_diff() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &(&x * &y) + &x.sin();
    for var in [
        "2", "0", "1/2", "pi", "2*x", "x + 1", "x^2", "sin(x)", "x*y", "-x",
    ] {
        let v = ctx.parse(var).unwrap();
        let d = e.diff_with_dependent(&v, &[&y]);
        assert!(!d.is_zero_structural(), "by {var}");
        assert!(d.has_unevaluated(), "by {var}: {d}");
        assert_eq!(d, e.diff(&v), "by {var}");
        assert_eq!(d, e.formal_diff(&v), "by {var}");
        assert!(e.try_diff(&v).is_err(), "by {var}");
        // Without dependent symbols, too.
        assert_eq!(e.diff_with_dependent(&v, &[]), e.diff(&v), "by {var}");
    }
}

/// The fix of W26 leaves the derivative by a symbol as it was: the chain rule with a formal
/// `Derivative(y, x)` for a dependent symbol.
///
/// Oracle: SymPy `diff(x*y(x) + sin(x), x)` = `x*Derivative(y(x), x) + y(x) + cos(x)`;
/// `diff(x**2 + y(x)**2, x)` = `2*x + 2*y(x)*Derivative(y(x), x)`.
#[test]
fn diff_with_dependent_by_a_symbol_is_the_chain_rule() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let dy = y.formal_diff(&x);
    let e = &(&x * &y) + &x.sin();
    assert_eq!(
        e.diff_with_dependent(&x, &[&y]),
        &(&(&x * &dy) + &y) + &x.cos()
    );
    let circle = &x.powi(2) + &y.powi(2);
    assert_eq!(
        circle.diff_with_dependent(&x, &[&y]),
        &(&x * 2) + &(&(&y * 2) * &dy)
    );
    // No dependent symbol: the plain derivative.
    assert_eq!(e.diff_with_dependent(&x, &[]), e.diff(&x));
}

// ── W28: the library's internal symbols ──────────────────────────────────────

/// The internal names that the engines declare positive or real (`grep '"__' src/`).
const DECLARED_DUMMIES: [&str; 20] = [
    "__lim_r",
    "__lim_w",
    "__lim_t",
    "__limit_t",
    "__lim_dir_t",
    "__gwp0",
    "__gzt_x",
    "__gz_posv1",
    "__gz_real0",
    "__gz_realv0",
    "__gruntz_lnw",
    "__ft_pos_w",
    "__exp_unit",
    // declared by `solve`, the inequalities and `summation` (moved in the follow-up of 0.60.1)
    "__sum_index",
    "__abs_modulus",
    "__abs_modulus_0",
    "__abs_real",
    "__u_scaled",
    "__u_shifted",
    "__ineq_real",
];

/// Limits, a series, a Fourier transform and a zero test that make the engines create their
/// dummies.
fn use_the_engines(ctx: &Context) {
    let x = ctx.symbol("x");
    let oo = ctx.parse("oo").unwrap();
    for (f, point, side) in [
        ("sin(x)/x", "0", 0),
        ("abs(x)/x", "0", 1),
        ("abs(x)/x", "0", -1),
        ("(1 + 1/x)^x", "oo", 0),
        ("sqrt(x^2 + x) - x", "oo", 0),
        ("x^x", "0", 1),
        ("floor(x)*x", "1", -1),
        ("(exp(x) + x)^(1/x)", "oo", 0),
        ("gamma(x + 1)/gamma(x) - x", "oo", 0),
        ("(a*x + sin(b*x))/x", "0", 0),
        ("x^a*exp(-x)", "oo", 0),
        ("floor(a + x)", "0", 1),
    ] {
        let e = ctx.parse(f).unwrap();
        let p = ctx.parse(point).unwrap();
        let _ = match side {
            1 => e.limit_right(&x, &p),
            -1 => e.limit_left(&x, &p),
            _ => e.limit(&x, &p),
        };
    }
    let _ = ctx.parse("gamma(x)").unwrap().series(&x, &oo, 3);
    let _ = ctx
        .parse("exp(-abs(x))*cos(x)")
        .unwrap()
        .fourier_transform(&x, &ctx.symbol("w"));
    let _ = ctx.parse("exp(2*a) - exp(a)^2").unwrap().is_zero();
    let (k, n) = (ctx.symbol("k"), ctx.symbol("n"));
    let _ = ctx
        .parse("k^2 + 2^k")
        .unwrap()
        .summation(&k, &ctx.int(1), &n);
    for equation in [
        "abs(x + I) - 2",
        "abs(x - 1) + abs(x + 2) - 5",
        "abs(2*x + 1) - abs(x)",
        "abs(x^2 - 1) - x",
        "sin(3*x) - 1/2",
    ] {
        let _ = ctx.parse(equation).unwrap().solve(&x);
    }
}

/// A symbol of the user's that has the name of one of the engines' dummies carries no
/// assumption, whether it is made before the engines run or after.  Up to 0.60.0 the dummy
/// *was* that symbol (one name space): for 11 of these 20 names `sqrt(s^2)` was built as `s`
/// or as `abs(s)`, and `abs(s)` as `s`, after the calls below (`__lim_r`, `__lim_w`, `__lim_t`,
/// `__limit_t`, `__gwp0`, `__gzt_x` positive; `__gz_real0`, `__gruntz_lnw`, `__sum_index`,
/// `__abs_modulus`, `__abs_modulus_0` real).
///
/// Oracle: exact, `sqrt((-1)^2) = 1 ≠ −1` and `|−1| = 1`.
#[test]
fn a_users_symbol_with_the_name_of_a_dummy_keeps_no_assumption() {
    for name in DECLARED_DUMMIES {
        for made_before in [true, false] {
            let ctx = Context::new();
            let before = made_before.then(|| ctx.symbol(name));
            use_the_engines(&ctx);
            let s = before.unwrap_or_else(|| ctx.symbol(name));
            let case = format!("{name}, made before: {made_before}");
            assert_eq!(s.is_positive(), None, "{case}");
            assert_eq!(s.query(Props::REAL), None, "{case}");
            let root = (&s * &s).sqrt();
            assert_eq!(root.to_string(), format!("sqrt({name}^2)"), "{case}");
            assert_eq!(s.abs().to_string(), format!("abs({name})"), "{case}");
            assert_eq!(root.subs(&s, &ctx.int(-1)).eval(), ctx.int(1), "{case}");
        }
    }
}

/// A parameter of the user's named like a dummy of the limit engine is a parameter: the limit
/// is what it is with the parameter named `q`.  Up to 0.60.0 the dummy captured it:
/// `((__lim_w*x + sin(x))/x).limit(x, 0)` was `oo`, `((x^2 + __gwp0*x)/(2*x^2 +
/// 1)).limit(x, oo)` was `1`, `(x*sin(__limit_t/x)).limit(x, oo)` was `0`,
/// `(__gw2*x*log(x) + __gw2).limit(x, 0)` was `0`, and with `__lim_r` every limit at a finite
/// point came back unevaluated.
///
/// Oracle: SymPy `limit((q*x + sin(x))/x, x, 0)` = `q + 1`; `limit((x**2 + q*x)/(2*x**2 + 1),
/// x, oo)` = `1/2`; `limit(x*sin(q/x), x, oo)` = `q`; `limit(q*x*log(x) + q, x, 0)` = `q`;
/// `limit((1 + q/x)**x, x, oo)` = `exp(q)`.
#[test]
fn a_parameter_named_like_a_limit_dummy_is_a_parameter() {
    for name in [
        "q",
        "__lim_r",
        "__lim_w",
        "__lim_t",
        "__limit_t",
        "__lim_dir_t",
        "__gwp0",
        "__gwp1",
        "__gw0",
        "__gw1",
        "__gw2",
        "__gw3",
        "__gruntz_lnw",
        "__gzt_x",
    ] {
        let ctx = Context::new();
        let (x, q) = (ctx.symbol("x"), ctx.symbol(name));
        let zero = ctx.int(0);
        let oo = ctx.parse("oo").unwrap();
        let build = |src: &str| ctx.parse(src).unwrap().subs(&ctx.symbol("Q"), &q);
        assert_eq!(build("(Q*x + sin(x))/x").limit(&x, &zero), &q + 1, "{name}");
        assert_eq!(
            build("(x^2 + Q*x)/(2*x^2 + 1)").limit(&x, &oo),
            ctx.rational(1, 2),
            "{name}"
        );
        assert_eq!(build("x*sin(Q/x)").limit(&x, &oo), q, "{name}");
        assert_eq!(build("Q*x*log(x) + Q").limit(&x, &zero), q, "{name}");
        assert_eq!(build("(1 + Q/x)^x").limit(&x, &oo), q.exp(), "{name}");
    }
}

/// A parameter named like a dummy of the ODE solvers: the solution is the one found with the
/// parameter named `q`.  Up to 0.60.0 it was not for `__v` (3 of the equations below),
/// `__clairaut_p` (1), `__r_sym` (4) and `__p` (1): the solver's dummy was the user's
/// parameter.
///
/// Oracle: exact — renaming a parameter commutes with solving.
#[test]
fn a_parameter_named_like_an_ode_dummy_is_a_parameter() {
    let equations = [
        "D2 + Q*D1",
        "D2 - Q^2*y",
        "D1 - Q*y",
        "D1 + y - Q*y^2",
        "x^2*D2 + Q*x*D1 + y",
        "D1 - Q - y/x",
        "D2 + y - Q*sin(x)",
        "D2 - 3*D1 + 2*y - Q*exp(x)",
        "y - x*D1 - Q*D1^2",
        "y*D2 - Q*D1^2",
        "x*D1 + y - Q*x^2*y^2",
        "D1 - (x + Q*y)/x",
        "D2 - 2*Q*D1 + Q^2*y",
    ];
    let solve = |name: &str, equation: &str| -> String {
        let ctx = Context::new();
        let (x, y, p) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol(name));
        let d1 = y.formal_diff(&x);
        let ode = ctx
            .parse(equation)
            .unwrap()
            .subs(&ctx.symbol("D2"), &d1.formal_diff(&x))
            .subs(&ctx.symbol("D1"), &d1)
            .subs(&ctx.symbol("Q"), &p);
        ode.solve_ode(&y, &x).subs(&p, &ctx.symbol("q")).to_string()
    };
    for equation in equations {
        let expected = solve("q", equation);
        for name in [
            "__v",
            "__r",
            "__p",
            "__t",
            "__Y",
            "__clairaut_p",
            "__riccati_v",
            "__r_sym",
            "__r_cc",
            "__t_cc",
            "__s_green",
            "__d1",
            "__d2",
            "__u_lin",
            "__p_low",
        ] {
            assert_eq!(solve(name, equation), expected, "{equation} with {name}");
        }
    }
}

// ── W29: an unevaluated one-sided limit keeps its side ───────────────────────

/// `abs(x)/x*Sum(1/k^(x + 2), k, 1, oo)` jumps at 0 and the engine evaluates neither side.
/// Up to 0.60.0 `limit_left`, `limit_right` and `limit` returned one node, `Limit(…, x, 0)`:
/// `left == right` and `left - right` was `0`.  Now the three are different expressions that
/// print their side, and each reads back through `parse` and through JSON as it was.
///
/// Oracle: the two values differ — SymPy `limit(Abs(x)/x*zeta(x + 2), x, 0, '-')` =
/// `-pi**2/6`, `'+'`: `pi**2/6`; mpmath `sign(x)*zeta(2 + x)` at `x = ∓1e-30`:
/// `∓1.644934066848226436472415` (20 and 40 digits).
#[test]
fn the_three_unevaluated_limits_at_a_jump_are_three_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let e = ctx.parse("abs(x)/x*Sum(1/k^(x + 2), k, 1, oo)").unwrap();
    let (left, right, both) = (
        e.limit_left(&x, &zero),
        e.limit_right(&x, &zero),
        e.limit(&x, &zero),
    );
    assert_ne!(left, right);
    assert_ne!(left, both);
    assert_ne!(right, both);
    assert!(!(&left - &right).is_zero_structural());
    assert_eq!((&left - &left), zero);
    let body = "abs(x)*Sum(1/k^(x + 2), k=1..oo)/x";
    assert_eq!(left.to_string(), format!("Limit({body}, x, 0, left)"));
    assert_eq!(right.to_string(), format!("Limit({body}, x, 0, right)"));
    assert_eq!(both.to_string(), format!("Limit({body}, x, 0)"));
    for node in [&left, &right, &both] {
        assert!(node.has_unevaluated());
        assert_eq!(&ctx.parse(&node.to_string()).unwrap(), node, "{node}");
        let json = node.to_json().unwrap();
        assert_eq!(&ctx.from_json(&json).unwrap(), node, "{node}");
        assert!(node.eval_f64().is_err(), "{node}");
        assert_eq!(&node.simplify(), node);
        assert_eq!(&node.eval(), node);
    }
    assert!(left.to_latex().starts_with(r"\lim_{x \to 0^{-}} "));
    assert!(right.to_latex().starts_with(r"\lim_{x \to 0^{+}} "));
    assert!(both.to_latex().starts_with(r"\lim_{x \to 0} "));
}

/// The side of an unevaluated one-sided limit survives a substitution into its point, and at
/// `±∞`, where there is one way of approach, the one-sided limits are the two-sided node (as
/// `limit_dir` documents: "for `point = ±∞` the direction is irrelevant").
///
/// Oracle: the definition — `lim_{x→a⁻}` at `a = 0` is `lim_{x→0⁻}`; SymPy
/// `Limit(f(x), x, a, '-').subs(a, 0)` = `Limit(f(x), x, 0, dir='-')`.
#[test]
fn the_side_of_an_unevaluated_limit_survives_substitution() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let zero = ctx.int(0);
    let at_a = ctx
        .parse("abs(x - a)/(x - a)*Sum(1/k^(x + 2), k, 1, oo)")
        .unwrap();
    let at_zero = ctx.parse("abs(x)/x*Sum(1/k^(x + 2), k, 1, oo)").unwrap();
    let left = at_a.limit_left(&x, &a);
    assert!(left.to_string().ends_with(", x, a, left)"), "{left}");
    assert_eq!(left.subs(&a, &zero), at_zero.limit_left(&x, &zero));
    assert_ne!(left.subs(&a, &zero), at_zero.limit_right(&x, &zero));
    let right = at_a.limit_right(&x, &a);
    assert_eq!(right.subs(&a, &zero), at_zero.limit_right(&x, &zero));
    // A compound point is printed whole, and in parentheses in LaTeX.
    let shifted = ctx
        .parse("abs(x - a - 1)/(x - a - 1)*Sum(1/k^(x + 2), k, 1, oo)")
        .unwrap()
        .limit_right(&x, &(&a + 1));
    assert!(
        shifted.to_string().ends_with(", x, a + 1, right)"),
        "{shifted}"
    );
    assert!(
        shifted
            .to_latex()
            .starts_with(r"\lim_{x \to \left(a + 1\right)^{+}} ")
    );
    assert_eq!(ctx.parse(&shifted.to_string()).unwrap(), shifted);
    // At infinity: one node.
    let oo = ctx.parse("oo").unwrap();
    let f = ctx.parse("sin(x)*Sum(1/k^(x + 2), k, 1, oo)").unwrap();
    assert!(f.limit(&x, &oo).has_unevaluated());
    assert_eq!(f.limit_left(&x, &oo), f.limit(&x, &oo));
    assert_eq!(f.limit_right(&x, &oo), f.limit(&x, &oo));
}

/// The parser reads the printed form of a one-sided limit and nothing else in the fourth
/// place.  Up to 0.60.0 a `Limit` with four arguments was "unknown 4-argument function".
///
/// Oracle: the round trip `parse(print(e)) == e`.
#[test]
fn parse_reads_the_side_of_a_limit() {
    let ctx = Context::new();
    let left = ctx.parse("Limit(sin(x)/x, x, 0, left)").unwrap();
    let right = ctx.parse("Limit(sin(x)/x, x, 0, right)").unwrap();
    let both = ctx.parse("Limit(sin(x)/x, x, 0)").unwrap();
    assert_eq!(left.to_string(), "Limit(sin(x)/x, x, 0, left)");
    assert_eq!(right.to_string(), "Limit(sin(x)/x, x, 0, right)");
    assert_ne!(left, right);
    assert_ne!(left, both);
    assert!(ctx.parse("Limit(sin(x)/x, x, 0, up)").is_err());
    assert!(ctx.parse("Limit(sin(x)/x, x, 0, 1)").is_err());
    // At infinity the side says nothing.
    assert_eq!(
        ctx.parse("Limit(1/x, x, oo, left)").unwrap(),
        ctx.parse("Limit(1/x, x, oo)").unwrap()
    );
}

// ── W11: `check_ode_solution` ────────────────────────────────────────────────

/// `self = 0` with `D1` for `y'` and `D2` for `y''`, and the proposed solution.
fn checks(ode: &str, solution: &str) -> bool {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let d1 = y.formal_diff(&x);
    let ode = ctx
        .parse(ode)
        .unwrap()
        .subs(&ctx.symbol("D2"), &d1.formal_diff(&x))
        .subs(&ctx.symbol("D1"), &d1);
    ode.check_ode_solution(&ctx.parse(solution).unwrap(), &y, &x)
}

/// The six correct solutions of the review findings (of 24 tried) for which
/// `check_ode_solution` was `false` up to 0.60.0, and more of the same kind: an antiderivative in
/// another form.  The residual now goes through the zero test of expressions with symbols.
///
/// Oracle: SymPy `diff(solution, x) - rhs` at `x = 3/10, 11/20, 4/5, -2/5` to 40 digits: 0 at
/// every point where it is finite, for each pair (`gen_w11.py` of the block's scratch; of its
/// 253 correct pairs 54 were `false` before and 12 are now).
#[test]
fn check_ode_solution_proves_an_antiderivative_in_another_form() {
    for (rhs, solution) in [
        // the six
        ("1/cosh(x)", "2*atan(exp(x))"),
        ("1/cosh(x)", "2*atan(tanh(x/2))"),
        ("1/cos(x)", "log(tan(x/2 + pi/4))"),
        ("1/cos(x)", "2*atanh(tan(x/2))"),
        ("1/(1 + x^2)", "atan((x - 1)/(x + 1))"),
        ("1/sqrt(1 - x^2)", "2*atan(x/(1 + sqrt(1 - x^2)))"),
        // more
        ("1/cosh(x)", "atan(sinh(x))"),
        ("1/cos(x)", "log(1/cos(x) + tan(x))"),
        ("1/cos(x)", "atanh(sin(x))"),
        ("1/(1 + x^2)", "atan((1 + x)/(1 - x))"),
        ("1/sin(x)", "log(tan(x/2))"),
        ("1/sin(x)", "-atanh(cos(x))"),
        ("tanh(x)", "x + log(1 + exp(-2*x))"),
        ("1/(1 + cos(x))", "(1 - cos(x))/sin(x)"),
        ("1/(1 + sin(x))", "tan(x/2 - pi/4)"),
        ("exp(x)/(1 + exp(2*x))", "-atan(exp(-x))"),
    ] {
        assert!(
            checks(&format!("D1 - ({rhs})"), solution),
            "y' = {rhs}, y = {solution}"
        );
    }
    for (ode, solution) in [
        ("D2 + 4*y", "tan(x)/(1 + tan(x)^2)"),
        ("D2 - y", "tanh(x)*cosh(x)"),
        ("D1 - 1 - y^2", "(tan(x) + tan(1))/(1 - tan(x)*tan(1))"),
        ("D1 - y*(1 - y)", "(1 + tanh(x/2))/2"),
    ] {
        assert!(checks(ode, solution), "{ode} = 0, y = {solution}");
    }
}

/// `true` stays a proof: a wrong solution, and a formula that solves the equation on a part
/// of the line only, are `false` (as before).
///
/// Oracle: as above — the residual is nonzero at all four points for the first four, and at
/// some of them for the last three (`cos(x)` in `y' = sqrt(1 - y^2)`: `-sin(x) - |sin(x)|`,
/// which is `-2*sin(3/10)` at `3/10`).
#[test]
fn check_ode_solution_does_not_accept_a_wrong_or_a_partial_solution() {
    for (ode, solution) in [
        ("D1 - 1/cosh(x)", "2*atan(exp(x)) + x/7"),
        ("D1 - 1/cosh(x)", "8*(2*atan(exp(x)))/7"),
        ("D1 - 1/(1 + x^2)", "atan((x - 1)/(x + 1)) + x^2/9"),
        ("D2 + y", "cos(x)^3 - 3*cos(x)*sin(x)^2"),
        ("D1 - sqrt(1 - y^2)", "cos(x)"),
        ("D1 - sqrt(y)", "x^2/4"),
        ("D1 - 1/(x*sqrt(x^2 - 1))", "acos(1/x)"),
    ] {
        assert!(!checks(ode, solution), "{ode} = 0, y = {solution}");
    }
}

// ── W09: `probably_equal` samples the declared domain ────────────────────────

/// `lhs.probably_equal(rhs, samples)` with the symbol `v` declared.
fn probably(assumptions: &[Assumption], lhs: &str, rhs: &str, samples: usize) -> Option<bool> {
    let ctx = Context::new();
    let _v = if assumptions.is_empty() {
        ctx.symbol("v")
    } else {
        ctx.symbol_with("v", assumptions).unwrap()
    };
    ctx.parse(lhs)
        .unwrap()
        .probably_equal(&ctx.parse(rhs).unwrap(), samples)
}

/// The 14 identities of a declared domain (of 29 tried in the review findings) that
/// `probably_equal` refuted up to 0.60.0 at a rational outside the domain: none is
/// `Some(false)` now, at 1, 8 and 32 samples.
///
/// Oracle: SymPy/mpmath `N(lhs - rhs, 30)` at the integers `-9..9`, resp. the positive
/// rationals `1/3, 5/7, 11/4, 2, 9, 64/3, 1/8`, resp. their negatives: below `1e-20` at every
/// point where it is finite (`gen_w09.py`; 810 such pairs with wrappers: 168 refuted at 8
/// samples before, 0 now).
#[test]
fn probably_equal_does_not_refute_identities_of_the_declared_domain() {
    use Assumption::{Integer, Negative, Positive};
    for (domain, lhs, rhs) in [
        (Integer, "(-1)^(v*(v + 1))", "1"),
        (Integer, "(-1)^(v^2)", "(-1)^v"),
        (Integer, "floor(v/2) + floor((v + 1)/2)", "v"),
        (Integer, "floor(v/2) + ceiling(v/2)", "v"),
        (Integer, "exp(I*pi*v^2)", "exp(I*pi*v)"),
        (Integer, "gamma(v + 1/2)*gamma(1/2 - v)", "pi*(-1)^v"),
        (Positive, "atan(v) + atan(1/v)", "pi/2"),
        (Positive, "acos(1/sqrt(v^2 + 1))", "atan(v)"),
        (Positive, "acosh(v^2 + 1)", "log(v^2 + 1 + v*sqrt(v^2 + 2))"),
        (Positive, "atan2(v, v)", "pi/4"),
        (Positive, "acos(v/sqrt(v^2 + 1))", "atan(1/v)"),
        (Negative, "atan(v) + atan(1/v)", "-pi/2"),
        (Negative, "log(-v) + I*pi", "log(v)"),
        (Negative, "acos(1/sqrt(v^2 + 1))", "-atan(v)"),
    ] {
        for samples in [1, 8, 32] {
            assert_ne!(
                probably(&[domain], lhs, rhs, samples),
                Some(false),
                "{lhs} = {rhs} for {domain:?} v, {samples} samples"
            );
        }
    }
}

/// What is false on the declared domain is still refuted, and what is false on it but true
/// on the rationals is refuted now: for an imaginary `v`, `abs(v)^2 = v^2` was `Some(true)`
/// up to 0.60.0 (it holds at every real sample; at `v = 3i/5` the sides are `9/25` and
/// `-9/25`).
///
/// Oracle: `asin(2v/(1 + v²)) − 2·atan(v)` at `v = 2`: mpmath `-1.2870022175865687`;
/// `(-1)^3 = -1 ≠ 1`; `|3i/5|² = 9/25`, `(3i/5)² = -9/25`.
#[test]
fn probably_equal_refutes_on_the_declared_domain() {
    use Assumption::{Imaginary, Integer, Positive};
    assert_eq!(
        probably(&[Positive], "asin(2*v/(1 + v^2))", "2*atan(v)", 8),
        Some(false)
    );
    assert_eq!(probably(&[Integer], "(-1)^v", "1", 8), Some(false));
    assert_eq!(probably(&[Integer], "floor(v/2)", "v/2", 8), Some(false));
    assert_eq!(probably(&[Imaginary], "abs(v)^2", "v^2", 8), Some(false));
    // Without assumptions the samples are the rationals they were.
    assert_eq!(probably(&[], "sqrt(v^2)", "v", 8), Some(false));
    assert_eq!(probably(&[], "(v + 1)^2", "v^2 + 2*v + 1", 8), Some(true));
}

/// A symbol that cancels in the difference of the two sides takes a sample value like the
/// others.  Up to 0.60.0 a point at which the sides kept such a symbol was skipped as if it
/// were a domain error, so the points where the sides agree decided: for a real `v`,
/// `sign(v) - t^2` against `1 - t^2` was `Some(true)` at 8 samples.
///
/// Oracle: at `v = -1`, `t = 0`: `-1 ≠ 1`.
#[test]
fn probably_equal_samples_a_symbol_that_cancels_in_the_difference() {
    assert_eq!(
        probably(&[Assumption::Real], "sign(v) - t^2", "1 - t^2", 8),
        Some(false)
    );
    assert_eq!(
        probably(&[Assumption::Integer], "(-1)^v - t^2", "1 - t^2", 8),
        Some(false)
    );
}

/// When no point of the declared domain can be drawn there is no verdict: an infinite symbol
/// has no finite value to take.
///
/// Oracle: the contract (`None`: "no sample point could be evaluated").
#[test]
fn probably_equal_of_a_domain_without_a_sample_is_none() {
    let ctx = Context::new();
    let w = ctx.symbol_with("w", &[Assumption::Infinite]).unwrap();
    assert_eq!(w.sin().powi(2).probably_equal(&w.sin(), 8), None);
}

// ── W37: the inverse Mellin transform on a given strip ───────────────────────

/// Is `a = b` at `x = 1/3, 3/4, 3/2, 3, 7` (to `1e-12`)?
fn same_function(ctx: &Context, x: &Ex, a: &Ex, b: &Ex) -> bool {
    [(1, 3), (3, 4), (3, 2), (3, 1), (7, 1)]
        .iter()
        .all(|&(p, q)| {
            let at = ctx.rational(p, q);
            match (a.subs(x, &at).eval_f64(), b.subs(x, &at).eval_f64()) {
                (Ok(u), Ok(v)) => (u - v).abs() <= 1e-12 * (1.0 + v.abs()),
                _ => false,
            }
        })
}

/// The round trip `mellin_transform` → `inverse_mellin_transform_in_strip` on the transform's
/// own strip returns the function: the seven cases of the review findings (four strips left
/// of the pole, where the two-argument inverse returns the function of the other strip —
/// asserted — and three right or two-sided) and more.
///
/// Oracle: SymPy `mellin_transform(f, x, s)` gives the same transform and strip for each,
/// e.g. `(-1/s, (-oo, 0), True)` for `Heaviside(x - 1)`, and `inverse_mellin_transform(F, s,
/// x, strip)` the function (`Heaviside(x - 1)` for `(-1/s, (None, 0))`).
#[test]
fn the_inverse_mellin_transform_on_the_transforms_own_strip_is_the_function() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let s = ctx.symbol("s");
    let cases = [
        // left of the pole
        ("heaviside(x - 1)", "-oo", "0", true),
        ("heaviside(x - 1)/x^2", "-oo", "2", true),
        ("heaviside(x - 1)/x^3", "-oo", "3", true),
        ("heaviside(x - 2)/x", "-oo", "1", true),
        // right of the pole, or between two
        ("exp(-x)", "0", "oo", false),
        ("1/(1 + x)", "0", "1", false),
        ("heaviside(1 - x)*x", "-1", "oo", false),
        // more
        ("heaviside(x - 1/2)*sqrt(x)", "-oo", "-1/2", true),
        (
            "heaviside(x - 1)/x^2 + heaviside(x - 2)/x^3",
            "-oo",
            "2",
            true,
        ),
        (
            "2*heaviside(x - 1)/x^2 - 3*heaviside(x - 3)/x",
            "-oo",
            "1",
            true,
        ),
        ("x^2*exp(-3*x)", "-2", "oo", false),
        ("exp(-1/x)", "-oo", "0", false),
        ("1/(1 + x)^3", "0", "3", false),
        ("x/(1 + x)^3", "-1", "2", false),
        ("sin(x)", "-1", "1", false),
        ("cos(3*x)", "0", "1", false),
        ("log(1 + x)", "-1", "0", false),
        ("heaviside(1 - x)*(1 - x)^2", "0", "oo", false),
        ("exp(-x) + 1/(1 + x)", "0", "1", false),
    ];
    for (source, lower, upper, left) in cases {
        let f = ctx.parse(source).unwrap();
        let (transform, strip) = f.mellin_transform(&x, &s).unwrap();
        let (lo, hi) = (ctx.parse(lower).unwrap(), ctx.parse(upper).unwrap());
        // The strip quoted here is the transform's.
        let expected = match (lower, upper) {
            ("-oo", u) => format!("{u} > re(s)"),
            (l, "oo") => format!("re(s) > {l}"),
            (l, u) => format!("re(s) > {l} & {u} > re(s)"),
        };
        assert_eq!(strip.to_string(), expected, "{source}");
        let back = transform
            .inverse_mellin_transform_in_strip(&s, &x, &lo, &hi)
            .unwrap();
        assert!(same_function(&ctx, &x, &back, &f), "{source}: {back}");
        let two = transform.inverse_mellin_transform(&s, &x).unwrap();
        assert_eq!(!same_function(&ctx, &x, &two, &f), left, "{source}: {two}");
    }
}

/// On another strip the function is another one: the table's, minus the residues of
/// `F(s)*x^(-s)` at the poles crossed to the left, plus those crossed to the right.
///
/// Oracle: mpmath, the Mellin–Barnes integral `(1/2π) ∫ F(c + it) x^(-c - it) dt` on the
/// middle line of the strip at `x = 1/2` and `x = 2` (25 digits, `quad` on `[-40, 40]`),
/// equal to the closed forms below to `1e-25` for `gamma(s)`, `pi/sin(pi*s)` and `beta(s, 3 -
/// s)` (`gamma(s)` on `c = -3/2` at `x = 2`: `1.13533528324` = `exp(-2) - 1 + 2`); for
/// `gamma(s)*sin(pi*s/2)` on `c = -2` and `gamma(s)*cos(pi*s/2)` on `c = -1`, which decay like
/// a power only, `quad` on `[-3000, 3000]`: equal to `sin(x) - x` and `cos(x) - 1` to `1e-6`;
/// SymPy `inverse_mellin_transform(pi/sin(pi*s), s, x, (1, 2))` = `-1/(x*(x + 1))`, `(-1/s, s,
/// x, (0, None))` = `-Heaviside(1 - x)`, `(-1/s, s, x, (-3, -2))` = `Heaviside(x - 1)`,
/// `(1/(s + 2), s, x, (None, -2))` = `-x**2*Heaviside(x - 1)`.
#[test]
fn the_inverse_mellin_transform_on_another_strip_crosses_the_poles() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let s = ctx.symbol("s");
    for (transform, lower, upper, function) in [
        ("gamma(s)", "0", "oo", "exp(-x)"),
        ("gamma(s)", "-1", "0", "exp(-x) - 1"),
        ("gamma(s)", "-2", "-1", "exp(-x) - 1 + x"),
        ("gamma(s)", "-3", "-2", "exp(-x) - 1 + x - x^2/2"),
        ("gamma(s)", "-5/2", "-2", "exp(-x) - 1 + x - x^2/2"),
        ("pi/sin(pi*s)", "0", "1", "1/(1 + x)"),
        ("pi/sin(pi*s)", "1", "2", "-1/(x*(1 + x))"),
        ("pi/sin(pi*s)", "-1", "0", "1/(1 + x) - 1"),
        ("pi/sin(pi*s)", "-2", "-1", "1/(1 + x) - 1 + x"),
        ("gamma(s)*sin(pi*s/2)", "-3", "-1", "sin(x) - x"),
        ("gamma(s)*cos(pi*s/2)", "-2", "0", "cos(x) - 1"),
        ("beta(s, 3 - s)", "-1", "0", "1/(1 + x)^3 - 1"),
        ("beta(s, 3 - s)", "3", "4", "1/(1 + x)^3 - 1/x^3"),
        ("-1/s", "0", "oo", "-heaviside(1 - x)"),
        ("-1/s", "-oo", "0", "heaviside(x - 1)"),
        ("-1/s", "-3", "-2", "heaviside(x - 1)"),
        ("1/(s + 2)", "-oo", "-2", "-x^2*heaviside(x - 1)"),
    ] {
        let big = ctx.parse(transform).unwrap();
        let (lo, hi) = (ctx.parse(lower).unwrap(), ctx.parse(upper).unwrap());
        let f = big
            .inverse_mellin_transform_in_strip(&s, &x, &lo, &hi)
            .unwrap_or_else(|e| panic!("{transform} on ({lower}, {upper}): {e}"));
        assert!(
            same_function(&ctx, &x, &f, &ctx.parse(function).unwrap()),
            "{transform} on ({lower}, {upper}): {f}"
        );
    }
}

/// No function has the transform on a strip with a pole inside, on an empty strip, or beyond
/// an end of the table's strip that is not a pole; and a strip that cannot be placed among
/// the poles is not guessed.  Each is an error.
///
/// Oracle: `Γ(s)` has poles at `0, -1, -2, …`, and the Mellin–Barnes integrals of `Γ(s)` on
/// the two sides of the pole 0 differ by its residue (mpmath at `x = 1/2`: `0.606530659713`
/// on `c = 1/2`, `-0.393469340287` on `c = -1/2`), so no one function belongs to a strip
/// around it.  (SymPy 1.14 `inverse_mellin_transform(gamma(s), s, x, (-1, 1))` answers
/// `exp(-x)`, the inverse on `Re s > 0`: it does not look at the strip there; on `(-1, 0)` it
/// raises `Gammas partially over the strip`.)  `Γ(s)·cos(πs/2)` is the transform of `cos x` on
/// `0 < Re s < 1` only (`∫ x^(s-1) cos x dx` diverges for `Re s ≥ 1`) and is analytic for `Re
/// s > 0`: beyond 1 it is the transform of no function.
#[test]
fn the_inverse_mellin_transform_refuses_a_strip_without_a_function() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let s = ctx.symbol("s");
    let on = |transform: &str, lower: &str, upper: &str| {
        ctx.parse(transform)
            .unwrap()
            .inverse_mellin_transform_in_strip(
                &s,
                &x,
                &ctx.parse(lower).unwrap(),
                &ctx.parse(upper).unwrap(),
            )
    };
    // a pole inside
    assert!(on("gamma(s)", "-1", "1").is_err());
    assert!(on("gamma(s)", "-oo", "0").is_err());
    assert!(on("-1/s", "-1", "1").is_err());
    assert!(on("pi/sin(pi*s)", "1/2", "3/2").is_err());
    // empty
    assert!(matches!(
        on("gamma(s)", "3", "2"),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        on("gamma(s)", "1", "1"),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // beyond an end that is not a pole
    assert!(on("gamma(s)*cos(pi*s/2)", "1", "2").is_err());
    assert!(on("gamma(s)*sin(pi*s/2)", "1", "2").is_err());
    // a pole at `-a` of unknown position
    assert!(on("1/(s + a)", "0", "oo").is_err());
    // … and of known position
    let _a = ctx.symbol_with("b", &[Assumption::Positive]).unwrap();
    assert!(same_function(
        &ctx,
        &x,
        &on("1/(s + b)", "0", "oo")
            .unwrap()
            .subs(&ctx.symbol("b"), &ctx.int(2)),
        &ctx.parse("x^2*heaviside(1 - x)").unwrap()
    ));
    // bounds that are no strip
    assert!(matches!(
        on("gamma(s)", "s", "1"),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        on("gamma(s)", "0", "nan"),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ── The follow-up: the dummies of `solve`, of the integrator and of `summation` ──

/// A parameter named like a dummy of `solve` or of the integrator is a parameter: the answer
/// is the one found with the parameter named `q`.  Up to 0.60.1 the dummy was the user's
/// symbol: `solve(sqrt(x) + x^(1/3) - __t_radical, x)` returned `[0, 9 - 4*sqrt(5), 9 +
/// 4*sqrt(5)]` (with `q`: no root found), and `1/(sin(x) + __wt)`, `1/(__wt + cos(x))`,
/// `1/(x^5 - x + __rs_t)`, `1/(x^3 + __rs_t)`, `1/(x*sqrt(x + __mt))`, `1/(exp(x) + __eu)`,
/// `1/(exp(x) + __elu)` came back unevaluated.  The bound variable of a `RootSum` is part of
/// the result and steps aside (`__rs_t_1`) when the integrand holds a symbol `__rs_t`.
///
/// Oracle: exact — renaming a parameter commutes with the operation (and with the name of a
/// bound variable).
#[test]
fn a_parameter_named_like_a_dummy_of_solve_or_the_integrator_is_a_parameter() {
    let answer = |name: &str, kind: &str, source: &str| -> String {
        let ctx = Context::new();
        let (x, p) = (ctx.symbol("x"), ctx.symbol(name));
        let e = ctx.parse(source).unwrap().subs(&ctx.symbol("Q"), &p);
        let results = match kind {
            "solve" => e.solve(&x).unwrap_or_default(),
            _ => vec![e.integrate(&x)],
        };
        let q = ctx.symbol("q");
        results
            .iter()
            .map(|r| r.subs(&p, &q).to_string())
            .collect::<Vec<_>>()
            .join(" ; ")
            .replace(&format!("{name}_1"), name)
    };
    let operations = [
        ("solve", "x^(1/2) + x^(1/3) - Q"),
        ("solve", "sqrt(x + Q) - x"),
        ("integrate", "1/(sin(x) + Q)"),
        ("integrate", "1/(Q + cos(x))"),
        ("integrate", "1/(x^5 - x + Q)"),
        ("integrate", "1/(x^3 + Q)"),
        ("integrate", "1/(x*sqrt(x + Q))"),
        ("integrate", "1/(exp(x) + Q)"),
        ("integrate", "x/(x^2 + Q)^2"),
    ];
    for (kind, source) in operations {
        let expected = answer("q", kind, source);
        for name in [
            "__t_radical",
            "__wt",
            "__rs_t",
            "__mt",
            "__eu",
            "__elu",
            "__prr_u",
        ] {
            assert_eq!(
                answer(name, kind, source),
                expected,
                "{kind} {source} with {name}"
            );
        }
    }
    // The integrals above are evaluated, not refused.
    assert!(!answer("__wt", "integrate", "1/(sin(x) + Q)").contains("Integral("));
    assert!(answer("__rs_t", "integrate", "1/(x^5 - x + Q)").starts_with("RootSum("));
}

/// `eval_derivatives` of a derivative by something that is not a symbol leaves it as it is.
/// Up to 0.60.1 `(x*y + sin(x)).diff(&2)`, the unevaluated `Derivative(x*y + sin(x), 2)`,
/// became `0` under `eval_derivatives` — a value where `diff` and `try_diff` refuse (W26, one
/// level up).  A derivative by a symbol is evaluated as before.
///
/// Oracle: SymPy `Derivative(x*y + sin(x), 2)` raises `ValueError: First variable cannot be a
/// number: 2`; `Derivative(x*y + sin(x), x).doit()` = `y + cos(x)`.
#[test]
fn eval_derivatives_leaves_a_derivative_by_a_non_symbol_unevaluated() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = &(&x * &y) + &x.sin();
    for var in ["2", "0", "pi", "2*x", "x + 1", "sin(x)"] {
        let v = ctx.parse(var).unwrap();
        let formal = e.diff(&v);
        assert_eq!(formal, e.formal_diff(&v), "by {var}");
        let evaluated = formal.eval_derivatives();
        assert!(!evaluated.is_zero_structural(), "by {var}");
        assert_eq!(evaluated, formal, "by {var}");
        // Inside an expression, and through `diff_with_dependent`.
        assert_eq!((&formal + &x).eval_derivatives(), &formal + &x, "by {var}");
        assert_eq!(
            e.diff_with_dependent(&v, &[&y]).eval_derivatives(),
            formal,
            "by {var}"
        );
    }
    // An inner derivative by a symbol is evaluated inside an outer one that stays.
    let inner = (&x * &x).formal_diff(&x);
    let outer = inner.formal_diff(&ctx.int(2));
    assert_eq!(outer.eval_derivatives(), (&x * 2).formal_diff(&ctx.int(2)));
    // By a symbol: as before.
    assert_eq!(e.formal_diff(&x).eval_derivatives(), &y + &x.cos());
    assert_eq!(
        e.formal_diff(&x).formal_diff(&x).eval_derivatives(),
        -x.sin()
    );
}
