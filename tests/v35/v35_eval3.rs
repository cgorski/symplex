//! Powers of `−1`, `i` and negative constants with rational exponents,
//! Boolean `simplify` of relations between constants, real and imaginary
//! parts of radicals of known sign, the sign of constants with a hidden
//! imaginary part, and roots and logarithms of negative real algebraic
//! numbers written with complex radicals (the eval hunt after 0.44).
//! Every reference value cites its oracle: SymPy 1.14 for the forms,
//! mpmath 1.3 at 50 digits for the values.

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s)
        .unwrap_or_else(|e| panic!("`{s}` does not parse: {e}"))
}

/// The leading `digits` significant characters of a decimal agree.
fn agrees(got: &str, want: &str, digits: usize) {
    let norm = |s: &str| -> String { s.chars().filter(|c| !c.is_whitespace()).collect() };
    let (g, w) = (norm(got), norm(want));
    assert!(
        g.len() >= digits && w.len() >= digits && g[..digits] == w[..digits],
        "{got} vs {want}"
    );
}

/// Before (0.44): `(−1)^(59/3)` stayed `(−1)^(59/3)`, so equal values had
/// different forms (`(−1)^(−59/3)` and `(−1)^(1/3)`), and a series
/// coefficient came out as six such terms; `I^(7/3)` stayed as well.
///
/// Oracle: SymPy 1.14 `(-1)**(59/3)` → `-(-1)**(2/3)`, `(-1)**(-59/3)` →
/// `(-1)**(1/3)`, `(-1)**(-1/3)` → `-(-1)**(2/3)`, `(-1)**(4/3)` →
/// `-(-1)**(1/3)`, `I**(7/3)` → `-I**(1/3)`, `I**(-1/3)` → `-I**(5/3)`
/// (`NegativeOne._eval_power`, `ImaginaryUnit._eval_power`); mpmath
/// `power(-1, mpf(59)/3)` = `0.5 − 0.8660254037844386467637231707529361834714j`,
/// `power(1j, mpf(7)/3)` = `−0.8660254037844386467637231707529361834714 − 0.5j`.
#[test]
fn powers_of_minus_one_and_i_reduce() {
    let ctx = Context::new();
    for (s, want) in [
        ("(-1)^(59/3)", "-(-1)^(2/3)"),
        ("(-1)^(-59/3)", "(-1)^(1/3)"),
        ("(-1)^(-1/3)", "-(-1)^(2/3)"),
        ("(-1)^(4/3)", "-(-1)^(1/3)"),
        ("(-1)^(2/3)*(-1)^(2/3)", "-(-1)^(1/3)"),
        ("(-1)^(25/2)", "I"),
        ("I^(7/3)", "-I^(1/3)"),
        ("I^(-1/3)", "-I^(5/3)"),
        ("I^(5/3)", "I^(5/3)"),
    ] {
        assert_eq!(p(&ctx, s).to_string(), p(&ctx, want).to_string(), "{s}");
    }
    agrees(
        &p(&ctx, "(-1)^(59/3)").eval_decimal(40).unwrap(),
        "0.5 - 0.8660254037844386467637231707529361834714*i",
        40,
    );
    agrees(
        &p(&ctx, "I^(7/3)").eval_decimal(40).unwrap(),
        "-0.8660254037844386467637231707529361834714 - 0.5*i",
        40,
    );
}

/// Before (0.44): `eval` split only `(−r)^(1/n)` for a negative rational:
/// `(−8)^(7/3)`, `(−8)^(−1/3)`, `√(−π)`, `(−π)^(1/3)`, `(2i)^(7/3)` stayed.
///
/// Oracle: SymPy 1.14 `(-8)**(7/3)` → `128*(-1)**(1/3)`, `(-8)**(-1/3)` →
/// `-(-1)**(2/3)/2`, `sqrt(-pi)` → `I*sqrt(pi)`, `(-pi)**(1/3)` →
/// `(-1)**(1/3)*pi**(1/3)`, `(2*I)**(7/3)` → `4*2**(1/3)*(-I**(1/3))`;
/// mpmath `power(-8, mpf(7)/3)` = `64 + 110.8512516844081467857565658563758314843j`,
/// `power(2j, mpf(7)/3)` = `−4.364494543886885614240290456759235525303 −
/// 2.519842099789746329534421214556456701141j`.
#[test]
fn powers_of_negative_constants_split_on_the_principal_branch() {
    let ctx = Context::new();
    for (s, want) in [
        ("(-8)^(7/3)", "128*(-1)^(1/3)"),
        ("(-8)^(-1/3)", "-(-1)^(2/3)/2"),
        ("sqrt(-pi)", "I*sqrt(pi)"),
        ("(-pi)^(1/3)", "(-1)^(1/3)*pi^(1/3)"),
        ("(2*I)^(7/3)", "-4*2^(1/3)*I^(1/3)"),
    ] {
        assert_eq!(p(&ctx, s).eval(), p(&ctx, want), "{s}");
    }
    agrees(
        &p(&ctx, "(-8)^(7/3)").eval_decimal(40).unwrap(),
        "64 + 110.8512516844081467857565658563758314843*i",
        40,
    );
    agrees(
        &p(&ctx, "(2*I)^(7/3)").eval_decimal(40).unwrap(),
        "-4.364494543886885614240290456759235525303 - 2.519842099789746329534421214556456701141*i",
        40,
    );
}

/// Before (0.44): Boolean `simplify` simplified the two sides of a relation
/// separately and never ran `eval`'s decision, so
/// `Eq(cos(pi/8), sqrt(2 + sqrt(2))/2).simplify()` stayed a relation while
/// `eval` gave `True` (the 0.44 eval hunt counted 45 such relations).
///
/// Oracle: SymPy 1.14 `Eq(cos(pi/8), sqrt(2 + sqrt(2))/2)` → `True`,
/// `sqrt(10 - 2*sqrt(5))/4 > sin(pi/5)` → `False`,
/// `Ne(sqrt(3 + 2*sqrt(2)), 1 + sqrt(2))` → `False`; mpmath
/// `sqrt(2 + sqrt(2 + sqrt(3)))/2 − cos(pi/24)` = 0 to 50 digits.
#[test]
fn boolean_simplify_decides_relations_like_eval() {
    let ctx = Context::new();
    ctx.symbol("x");
    for (s, want) in [
        ("Eq(cos(pi/8), sqrt(2 + sqrt(2))/2)", "True"),
        ("sqrt(10 - 2*sqrt(5))/4 > sin(pi/5)", "False"),
        ("sqrt(2 + sqrt(2 + sqrt(3)))/2 > cos(pi/24)", "False"),
        ("sqrt(2 + sqrt(2 + sqrt(3)))/2 >= cos(pi/24)", "True"),
        ("Ne(sqrt(3 + 2*sqrt(2)), 1 + sqrt(2))", "False"),
        ("Eq(cos(pi/8), sqrt(2 + sqrt(2))/2) & (x > 0)", "x > 0"),
        ("Ne(cos(pi/8), sqrt(2 + sqrt(2))/2) | (x > 0)", "x > 0"),
    ] {
        let b = ctx.parse_bool(s).unwrap();
        assert_eq!(b.simplify().to_string(), want, "{s}");
        assert_eq!(b.simplify().to_string(), b.eval().to_string(), "{s}");
    }
}

/// Before (0.44): `as_real_imag` left `re(∛(28/27 − √87/9))` although the
/// radicand is positive, and `re(√(−p))`, `re((−p)^(1/3))`, `re(n^(1/3))`
/// for a positive `p` and a negative `n`, and `re((−1)^(1/3))` stayed
/// opaque.  (A power of a negative constant *sum* keeps its opaque parts:
/// `apart` builds the real quadratic term of `±√(−√5/2 − 1/2)` from both
/// roots when it sees them.)
///
/// Oracle: SymPy 1.14 `cbrt(28/27 - sqrt(87)/9).as_real_imag()` →
/// `((28/27 - sqrt(87)/9)**(1/3), 0)`, `sqrt(-p).as_real_imag()` → `(0,
/// sqrt(p))`, `((-p)**(1/3)).as_real_imag()` → `(p**(1/3)/2,
/// sqrt(3)*p**(1/3)/2)`, `(n**(1/3)).as_real_imag()` → `((-n)**(1/3)/2,
/// sqrt(3)*(-n)**(1/3)/2)`, `re((-1)**(1/3))` → `1/2`,
/// `((-p)**(3/2)).as_real_imag()` → `(0, -p**(3/2))`.
#[test]
fn real_and_imaginary_parts_of_radicals_of_known_sign() {
    let ctx = Context::new();
    ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    ctx.symbol_with("n", &[Assumption::Negative]).unwrap();
    for (s, re, im) in [
        ("cbrt(28/27 - sqrt(87)/9)", "cbrt(28/27 - sqrt(87)/9)", "0"),
        ("sqrt(-p)", "0", "sqrt(p)"),
        ("(-p)^(1/3)", "p^(1/3)/2", "sqrt(3)*p^(1/3)/2"),
        ("n^(1/3)", "(-n)^(1/3)/2", "sqrt(3)*(-n)^(1/3)/2"),
        ("(-1)^(1/3)", "1/2", "sqrt(3)/2"),
        ("(-p)^(3/2)", "0", "-p^(3/2)"),
    ] {
        let (got_re, got_im) = p(&ctx, s).as_real_imag();
        assert_eq!(got_re, p(&ctx, re), "re {s}");
        assert_eq!(got_im, p(&ctx, im), "im {s}");
    }
    // An unassumed base keeps its opaque parts.
    let (re, _) = p(&ctx, "sqrt(x)").as_real_imag();
    assert_eq!(re.to_string(), "re(sqrt(x))");
}

/// A product with `∞` is oriented by the sign of its constant factor only
/// when that factor is real: `−1 − i·(e^{10⁻³⁰} − 1 − 10⁻³⁰)` evaluates to
/// `−1` at any working precision below 61 digits (its imaginary part is
/// `−5·10⁻⁶¹`), and must stay a directed infinity, never `−∞`.  (Up to
/// 0.44 `sign_of_real_constant` gave it the sign `−`; canonical
/// multiplication asked it only after the assumptions proved the factor
/// real, so the product was already right.)
///
/// Oracle: mpmath `exp(mpf(10)**-30) - 1 - mpf(10)**-30` =
/// `5.000000000000000000000000000001666…·10⁻⁶¹` (nonzero), SymPy 1.14
/// `(-1 - I*(exp(S(10)**-30) - 1 - S(10)**-30))*oo` stays a product.
#[test]
fn infinity_times_a_constant_with_a_hidden_imaginary_part() {
    let ctx = Context::new();
    let e = p(&ctx, "(-1 - I*(exp(10^(-30)) - 1 - 10^(-30)))*oo").eval();
    assert_ne!(e, p(&ctx, "-oo"));
    assert_ne!(e, p(&ctx, "oo"));
    assert!(e.to_string().ends_with("*oo"), "{e}");
    assert_eq!(p(&ctx, "ln(2)*oo"), p(&ctx, "oo"));
    assert_eq!(p(&ctx, "(sqrt(2) - 2)*oo"), p(&ctx, "-oo"));
}

/// The eigenvalue `λ ≈ −3.2546` of `[[x, x, y], [x − 1, y, −x], [1 − x, 1 −
/// x, y]]` at `x = −18/13`, `y = −5/13`, by Cardano's formula: its complex
/// cube roots cancel only exactly.  Before (0.44): `√λ` and `ln λ` were
/// refused (`PrecisionExhausted`), which blocked `matrix_sqrt`/`matrix_log`
/// of such matrices.  Now `λ` is decided real by its minimal polynomial
/// `169λ³ + 364λ² − 756λ − 490` and negative by its certified value, and
/// `√λ = i·√(−λ)`, `ln λ = ln(−λ) + iπ`.
///
/// Oracle: mpmath `polyroots([169, 364, -756, -490])` →
/// `−3.2545986779106937129333779303066882570999683529323`; `sqrt` of it
/// = `1.8040506306394767347866637855871248812330911850229j`, `log` =
/// `1.1800689740225447847578350980264237970694097210352 +
/// 3.1415926535897932384626433832795028841971693993751j`, `cbrt` =
/// `0.74097317465558342727262858716614320512736303328064 +
/// 1.2834031855490780364378902509809827728192379505743j`.
#[test]
fn roots_and_logarithm_of_a_negative_cardano_eigenvalue() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let one = ctx.int(1);
    let m = Matrix::new(vec![
        vec![x.clone(), x.clone(), y.clone()],
        vec![&x - &one, y.clone(), -&x],
        vec![&one - &x, &one - &x, y.clone()],
    ])
    .unwrap();
    let (xv, yv) = (ctx.rational(-18, 13), ctx.rational(-5, 13));
    let lambda = m
        .eigenvals()
        .unwrap()
        .into_iter()
        .map(|l| l.subs(&x, &xv).subs(&y, &yv))
        .find(|l| l.eval_decimal(5).unwrap().starts_with("-3.2546"))
        .expect("the eigenvalue near -3.2546");
    agrees(
        &lambda.sqrt().eval_decimal(40).unwrap(),
        "1.804050630639476734786663785587124881233*i",
        40,
    );
    agrees(
        &lambda.ln().eval_decimal(40).unwrap(),
        "1.180068974022544784757835098026423797069 + 3.141592653589793238462643383279502884197*i",
        40,
    );
    let cbrt = p(&ctx, "z^(1/3)").subs(&ctx.symbol("z"), &lambda);
    agrees(
        &cbrt.eval_decimal(30).unwrap(),
        "0.740973174655583427272628587166 + 1.28340318554907803643789025098*i",
        30,
    );
}
