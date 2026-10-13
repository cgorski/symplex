//! Special values of the inverse functions, `exp`/`log` compositions over
//! ℂ, radicals with assumed factors, substitution into the quotient forms
//! of the reciprocal functions, and `simplify` of relations between
//! constants (the eval hunt after 0.43).  Every reference value cites its
//! oracle: SymPy 1.14, or mpmath 1.3 at 50 digits (along `oo → 10ᵏ` for an
//! infinite point).

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s)
        .unwrap_or_else(|e| panic!("`{s}` does not parse: {e}"))
}

/// `s.eval()` equals `want.eval()` and agrees with `want` at 40 digits.
fn evals_to(ctx: &Context, s: &str, want: &str) {
    let got = p(ctx, s).eval();
    let expected = p(ctx, want).eval();
    assert_eq!(got, expected, "{s}: {got} != {expected}");
    assert_eq!(
        got.eval_decimal(40).unwrap(),
        p(ctx, want).eval_decimal(40).unwrap(),
        "{s}"
    );
}

/// Before: `acosh(0)`, `acosh(-1)`, `acosh(±I)`, `asinh(±1)`, `asinh(±I)`,
/// `atanh(±I)`, `asin(±I)`, `acos(±I)` stayed unevaluated, and so did the
/// reciprocal forms built on them (`asech(-1) = acosh(-1)`, `acsch(1) =
/// asinh(1)`, `acoth(I) = atanh(-I)`, `asech(oo) = acosh(0)`).
///
/// Oracle: SymPy 1.14 `acosh(0)` → `I*pi/2`, `acosh(-1)` → `I*pi`,
/// `acosh(I)` → `log(1 + sqrt(2)) + I*pi/2`, `acosh(-I)` →
/// `log(1 + sqrt(2)) - I*pi/2`, `acosh(S(1)/2)` → `I*pi/3`,
/// `acosh(-sqrt(3)/2)` → `5*I*pi/6`, `asinh(1)` → `log(1 + sqrt(2))`,
/// `asinh(-1)` → `log(-1 + sqrt(2))`, `asinh(I)` → `I*pi/2`,
/// `asinh(sqrt(3)*I/2)` → `I*pi/3`, `atanh(I)` → `I*pi/4`,
/// `atanh(sqrt(3)*I)` → `I*pi/3`, `acoth(I)` → `-I*pi/4`, `asech(-1)` →
/// `I*pi`, `asech(2)` → `I*pi/3`, `asech(I)` → `log(1 + sqrt(2)) - I*pi/2`,
/// `acsch(1)` → `log(1 + sqrt(2))`, `acsch(I)` → `-I*pi/2`, `acsch(2*I)`
/// → `-I*pi/6`, `asin(I)` → `I*log(1 + sqrt(2))`, `acos(I)` →
/// `pi/2 - I*log(1 + sqrt(2))`; mpmath `acosh(0)` = 1.5707963…j,
/// `acosh(1j)` = 0.8813735… + 1.5707963…j.
#[test]
fn inverse_hyperbolic_special_values() {
    let ctx = Context::new();
    for (s, want) in [
        ("acosh(0)", "I*pi/2"),
        ("acosh(-1)", "I*pi"),
        ("acosh(I)", "log(1 + sqrt(2)) + I*pi/2"),
        ("acosh(-I)", "log(1 + sqrt(2)) - I*pi/2"),
        ("acosh(1/2)", "I*pi/3"),
        ("acosh(-sqrt(3)/2)", "5*I*pi/6"),
        ("asinh(1)", "log(1 + sqrt(2))"),
        ("asinh(-1)", "-log(1 + sqrt(2))"),
        ("asinh(I)", "I*pi/2"),
        ("asinh(-I)", "-I*pi/2"),
        ("asinh(sqrt(3)*I/2)", "I*pi/3"),
        ("atanh(I)", "I*pi/4"),
        ("atanh(-I)", "-I*pi/4"),
        ("atanh(sqrt(3)*I)", "I*pi/3"),
        ("acoth(I)", "-I*pi/4"),
        ("acoth(-I)", "I*pi/4"),
        ("asech(-1)", "I*pi"),
        ("asech(2)", "I*pi/3"),
        ("asech(I)", "log(1 + sqrt(2)) - I*pi/2"),
        ("acsch(1)", "log(1 + sqrt(2))"),
        ("acsch(-1)", "-log(1 + sqrt(2))"),
        ("acsch(I)", "-I*pi/2"),
        ("acsch(2*I)", "-I*pi/6"),
        ("asin(I)", "I*log(1 + sqrt(2))"),
        ("asin(-I)", "-I*log(1 + sqrt(2))"),
        ("acos(I)", "pi/2 - I*log(1 + sqrt(2))"),
        ("asec(I)", "pi/2 + I*log(1 + sqrt(2))"),
        ("acsc(I)", "-I*log(1 + sqrt(2))"),
    ] {
        evals_to(&ctx, s, want);
    }
    // No closed form: unchanged (SymPy keeps `acosh(2*I)`, `asinh(1/2)`).
    for s in ["acosh(2*I)", "asinh(1/2)", "acosh(2)", "atanh(2*I)"] {
        assert_eq!(p(&ctx, s).eval(), p(&ctx, s), "{s}");
    }
}

/// Before: `series(tanh(acosh(x)), x, 0, 3)` kept `tanh(acosh(0))` (a pole)
/// in its coefficients, because `acosh(0)` did not evaluate.
///
/// Oracle: SymPy 1.14 `series(tanh(acosh(x)), x, 0, 3)` → `I/x - I*x/2 +
/// O(x**3)`, `series(acosh(x), x, 0, 4)` → `I*pi/2 - I*x - I*x**3/6`.
#[test]
fn series_through_acosh_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let s = p(&ctx, "tanh(acosh(x))").series(&x, &zero, 3);
    assert_eq!(s, p(&ctx, "I/x - I*x/2"), "{s}");
    let s = p(&ctx, "acosh(x)").series(&x, &zero, 4);
    assert_eq!(s, p(&ctx, "I*pi/2 - I*x - I*x^3/6"), "{s}");
}

/// Before: `asech(±I·oo)`, `asech(±oo)`, `asech(zoo)` were all `acosh(0)`
/// (value iπ/2), and `asech(x)` at `x = I·oo` likewise.  `asech z =
/// acosh(1/z)` approaches `acosh`'s cut at 0 from the side opposite to
/// `Im z`.
///
/// Oracle: SymPy 1.14 `asech(oo)` → I*pi/2, `asech(-oo)` → I*pi/2,
/// `asech(oo*I)` → -I*pi/2, `asech(-oo*I)` → I*pi/2, `asech(zoo)` →
/// I*AccumBounds(-pi/2, pi/2) (no single value: `nan` here); mpmath
/// `acosh(1/mpc(10, 1))` = 0.0099… − 1.4716…j (towards −iπ/2 for `oo + I`),
/// `acosh(1/mpc(1, 1e30))` = 1e-30 − 1.5707963…j.
#[test]
fn asech_at_the_infinities() {
    let ctx = Context::new();
    for (s, want) in [
        ("asech(oo)", "I*pi/2"),
        ("asech(-oo)", "I*pi/2"),
        ("asech(I*oo)", "-I*pi/2"),
        ("asech(-I*oo)", "I*pi/2"),
        ("asech(oo + I)", "-I*pi/2"),
        ("asech(oo - I)", "I*pi/2"),
        ("asech(1 + I*oo)", "-I*pi/2"),
    ] {
        evals_to(&ctx, s, want);
    }
    assert_eq!(p(&ctx, "asech(zoo)"), ctx.nan());
    let x = ctx.symbol("x");
    assert_eq!(
        p(&ctx, "asech(x)").subs(&x, &p(&ctx, "I*oo")),
        p(&ctx, "-I*pi/2")
    );
    assert_eq!(
        p(&ctx, "asech(x)").subs(&x, &p(&ctx, "-oo")).eval(),
        p(&ctx, "I*pi/2")
    );
    // `(oo + I)^2` has lost the side: no value is claimed.
    assert_eq!(
        p(&ctx, "asech(x^2)").subs(&x, &p(&ctx, "oo + I")),
        ctx.nan()
    );
}

/// Before: `exp(2*log(10))`, `exp(-3*log(2)/2)`, `exp(x + 2*log(3))`
/// stayed, and `exp(2*log(1 + I))` was `exp(2*ln(sqrt(2)))*I`.
/// `exp(k·Log z) = z^k` is the principal power; for a constant `z` it folds
/// (a symbolic one stays: the limit engine depends on it).
///
/// Oracle: SymPy 1.14 `exp(2*log(10))` → 100, `exp(-3*log(2)/2)` →
/// sqrt(2)/4, `exp(pi*log(2))` → 2**pi, `exp(x + 2*log(3))` → 9*exp(x),
/// `exp(2*log(1 + I))` → 2*I, `exp(log(-8)/3)` → 2*exp(I*pi/3) = 1 +
/// sqrt(3)*I, `exp(x*log(2))` and `exp(I*log(2))` unchanged; mpmath
/// `exp(log(2+1j)/2)` = 1.4553466… + 0.3435607…j = `sqrt(2+1j)`.
#[test]
fn exp_of_a_multiple_of_a_logarithm() {
    let ctx = Context::new();
    for (s, want) in [
        ("exp(2*log(10))", "100"),
        ("exp(-3*log(2)/2)", "sqrt(2)/4"),
        ("exp(2*log(1 + I))", "2*I"),
        ("exp(log(-8)/3)", "1 + sqrt(3)*I"),
        ("exp(pi*log(2))", "2^pi"),
        ("exp(log(2 + I)/2)", "sqrt(2 + I)"),
        ("exp(2*log(27) + log(3))", "2187"),
        ("exp(4*log(sqrt(3)/2))", "9/16"),
    ] {
        evals_to(&ctx, s, want);
    }
    assert_eq!(p(&ctx, "exp(x + 2*log(3))").eval(), p(&ctx, "9*exp(x)"));
    assert_eq!(p(&ctx, "exp(log(2) + log(3))").eval(), ctx.int(6));
    assert_eq!(p(&ctx, "exp(2*log(10))").simplify(), ctx.int(100));
    for s in ["exp(x*log(2))", "exp(I*log(2))"] {
        assert_eq!(p(&ctx, s).eval(), p(&ctx, s), "{s}");
    }
    // A symbolic base is generic, as for `exp(log(x)) = x` (0.47; SymPy:
    // `exp(2*log(x))` → `x**2`).
    assert_eq!(p(&ctx, "exp(2*log(x))").eval(), p(&ctx, "x^2"));
}

/// Before: `log(exp(I))`, `log(exp(5*I))`, `log(exp(2 + 5*pi*I/3))`,
/// `log(-E)`, `log(1 - sqrt(2))` (a negative constant that is not a
/// rational) and `asinh(sinh(3))`, `acosh(cosh(-2))`, `atanh(tanh(1))`
/// stayed.
/// `Log(e^y)` is `y` moved into the strip `−π < Im ≤ π`; the inverse
/// hyperbolic functions invert theirs on the reals.
///
/// Oracle: SymPy 1.14 `log(exp(I))` → I, `log(exp(5*I))` → `-2*I*pi +
/// 5*I`, `log(exp(2 + 3*I))` → 2 + 3*I, `log(exp(2 + 5*I*pi/3))` → `2 -
/// I*pi/3`, `asinh(sinh(3))` → 3, `acosh(cosh(-2))` → 2, `atanh(tanh(1))`
/// → 1, `log(-E)` → 1 + I*pi, `log(-pi)` → log(pi) + I*pi, `log(1 -
/// sqrt(2))` → log(-1 + sqrt(2)) + I*pi; mpmath `log(exp(5j))` =
/// −1.2831853…j.
#[test]
fn logarithm_of_exponential_and_inverse_hyperbolic_of_hyperbolic() {
    let ctx = Context::new();
    for (s, want) in [
        ("log(exp(I))", "I"),
        ("log(exp(5*I))", "5*I - 2*pi*I"),
        ("log(exp(-7*I))", "-7*I + 2*pi*I"),
        ("log(exp(2 + 3*I))", "2 + 3*I"),
        ("log(exp(2 + 5*pi*I/3))", "2 - pi*I/3"),
        ("asinh(sinh(3))", "3"),
        ("acosh(cosh(-2))", "2"),
        ("atanh(tanh(1))", "1"),
        ("log(-E)", "1 + pi*I"),
        ("log(-pi)", "log(pi) + pi*I"),
        ("log(1 - sqrt(2))", "log(sqrt(2) - 1) + pi*I"),
        ("log(exp(1 - pi*I))", "1 + pi*I"),
    ] {
        evals_to(&ctx, s, want);
    }
    // A non-real or unknown argument stays (`log(exp(x))`, `asinh(sinh(x))`).
    for s in ["log(exp(x))", "asinh(sinh(x))", "acosh(cosh(2*I))"] {
        assert_eq!(p(&ctx, s).eval(), p(&ctx, s), "{s}");
    }
}

/// Before: `sqrt(4*b^2)` stayed for an integer `b` (only a lone
/// `sqrt(r^2) = |r|` folded).  Non-negative factors whose root disappears
/// come out of a power with the magnitude of the coefficient.
///
/// Oracle: SymPy 1.14 with `b` integer, `r` real, `p` positive, `n`
/// negative: `sqrt(4*b**2)` → 2*Abs(b), `sqrt(8*r**2)` → 2*sqrt(2)*Abs(r),
/// `sqrt(9*p**2)` → 3*p, `sqrt(-9*p**2)` → 3*I*p, `sqrt(-4*b**2*x)` →
/// 2*sqrt(-x)*Abs(b), `(4*b**2)**(3/2)` → 8*Abs(b)**3, `sqrt(4*n**2)` →
/// -2*n, `(8*r**6)**(1/3)` → 2*r**2, `sqrt(4*x**2)` → 2*sqrt(x**2) (here
/// unchanged), `sqrt(4*x)` → 2*sqrt(x) (here unchanged).
#[test]
fn square_roots_of_assumed_squares() {
    let ctx = Context::new();
    ctx.symbol_with("b", &[Assumption::Integer]).unwrap();
    ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    ctx.symbol_with("n", &[Assumption::Negative]).unwrap();
    for (s, want) in [
        ("sqrt(4*b^2)", "2*abs(b)"),
        ("sqrt(8*r^2)", "2*sqrt(2)*abs(r)"),
        ("sqrt(9*p^2)", "3*p"),
        ("sqrt(-9*p^2)", "3*I*p"),
        ("sqrt(-4*b^2*x)", "2*sqrt(-x)*abs(b)"),
        ("(4*b^2)^(3/2)", "8*abs(b)^3"),
        ("(4*b^2)^(-1/2)", "1/(2*abs(b))"),
        ("sqrt(4*n^2)", "-2*n"),
        ("(8*r^6)^(1/3)", "2*r^2"),
        ("sqrt(b^2/4)", "abs(b)/2"),
    ] {
        assert_eq!(p(&ctx, s), p(&ctx, want), "{s}");
    }
    for s in ["sqrt(4*x)", "sqrt(4*x^2)", "cbrt(4*b^2)"] {
        assert_eq!(p(&ctx, s).to_string(), s, "{s}");
    }
    // Numerically, at b = -5, x = 2 - i: sqrt(-4·25·(2 - i)) = sqrt(-200 + 100i)
    // (mpmath: 3.4356074972… + 14.553466902…j).
    let v = p(&ctx, "sqrt(-4*b^2*x)")
        .subs(&p(&ctx, "b"), &ctx.int(-5))
        .subs(&p(&ctx, "x"), &p(&ctx, "2 - I"));
    assert_eq!(
        v.eval_decimal(30).unwrap(),
        p(&ctx, "sqrt(-200 + 100*I)").eval_decimal(30).unwrap()
    );
}

/// Before: `coth(x).subs(x, oo)` was `nan`, as were `cot(x)` at `x = ±I·oo`,
/// `1/coth(x)` at `oo` and `x*cot(x)` at `I·oo`: the quotient forms became
/// `oo·0`.  `acot(x)`, `acoth(x)` at `x = 0` were `atan(zoo)`,
/// `atanh(zoo)`, `asech(x)` at 0 was `zoo`.
///
/// Oracle: SymPy 1.14 `coth(x).subs(x, oo)` → 1, `coth(x).subs(x, -oo)`
/// → -1, `cot(x).subs(x, I*oo)` → -I, `cot(x).subs(x, -I*oo)` → I,
/// `tanh(oo)` → 1 (= `1/coth`), `acot(0)` → pi/2, `acoth(0)` → I*pi/2,
/// `asech(0)` → oo; mpmath `coth(mpc(1e20, 1))` = 1, `z*cot(z)` at `z =
/// 1e20j` = 1e20, `coth(z)*csch(z)` at `z = 1e20` = 0 at 50 digits.
#[test]
fn substitution_into_reciprocal_quotients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let at = |s: &str, pt: &str| p(&ctx, s).subs(&x, &p(&ctx, pt));
    assert_eq!(at("coth(x)", "oo"), ctx.one());
    assert_eq!(at("coth(x)", "-oo"), ctx.int(-1));
    assert_eq!(at("coth(x)", "oo + I"), ctx.one());
    assert_eq!(at("coth(2*x + 1)", "oo"), ctx.one());
    assert_eq!(at("cot(x)", "I*oo"), -ctx.i_unit());
    assert_eq!(at("cot(x)", "-I*oo"), ctx.i_unit());
    assert_eq!(at("1/coth(x)", "oo"), ctx.one());
    assert_eq!(at("1/cot(x)", "-I*oo"), -ctx.i_unit());
    assert_eq!(at("coth(x)^2", "-oo"), ctx.one());
    assert_eq!(at("coth(x)*csch(x)", "oo"), ctx.zero());
    assert_eq!(at("x*cot(x)", "I*oo"), ctx.infinity());
    assert_eq!(at("exp(-x)*coth(x)", "oo"), ctx.zero());
    assert_eq!(at("acot(x)", "0"), p(&ctx, "pi/2"));
    assert_eq!(at("acot(2*x)", "0"), p(&ctx, "pi/2"));
    assert_eq!(at("acoth(x)", "0"), p(&ctx, "I*pi/2"));
    assert_eq!(at("asech(x)", "0"), ctx.infinity());
    // Elsewhere: the quotient forms as before.
    assert_eq!(at("coth(x)", "y"), p(&ctx, "cosh(y)/sinh(y)"));
    assert_eq!(at("coth(x)", "0"), ctx.complex_infinity());
    assert_eq!(at("cot(x)", "oo"), p(&ctx, "cos(oo)/sin(oo)"));
}

/// Before (0.43): `Eq(cos(5*pi/8), 2*cos(5*pi/16)^2 - 1).simplify()` was
/// left as the relation (0.42: `True`), and `simplify(tan(pi/24)*cos(pi/24))`
/// stayed (0.42: `sin(pi/24)`): `eval` writes `cos(5π/8)`, `tan(π/24)` in
/// radicals, after which neither the double angle nor Fu applies.
///
/// Oracle: SymPy 1.14 `simplify(Eq(cos(5*pi/8), 2*cos(5*pi/16)**2 - 1))`
/// → True; mpmath `tan(pi/24)*cos(pi/24) - sin(pi/24)` = 0 at 50 digits,
/// `2*cos(5*pi/16)**2 - 1 - cos(5*pi/8)` = 0.
#[test]
fn simplify_of_trigonometric_constants() {
    let ctx = Context::new();
    let rel = ctx
        .parse_bool("Eq(cos(5*pi/8), 2*cos(5*pi/16)^2 - 1)")
        .unwrap();
    assert_eq!(rel.simplify().to_string(), "True");
    assert_eq!(
        p(&ctx, "tan(pi/24)*cos(pi/24)").simplify(),
        p(&ctx, "sin(pi/24)")
    );
    assert_eq!(
        p(&ctx, "2*cos(5*pi/16)^2 - 1").simplify(),
        p(&ctx, "cos(5*pi/8)")
    );
    assert_eq!(
        p(&ctx, "x + tan(pi/24)*cos(pi/24)").simplify(),
        p(&ctx, "x + sin(pi/24)")
    );
}
