//! Tests of session 18. Lane G of the 0.60.1 block: questions that were left undecided without need
//! (`is_identity`, `is_prime_value`, two-property queries, `Context::query`) and printed
//! forms that did not read back (`RootSum`, MathML of `RootOf`).
//! Every test says what was wrong before and cites its oracle.

use symplex::prelude::*;

/// The 24 properties.
const PROPS: [Props; 24] = [
    Props::COMMUTATIVE,
    Props::COMPLEX,
    Props::REAL,
    Props::RATIONAL,
    Props::INTEGER,
    Props::ALGEBRAIC,
    Props::TRANSCENDENTAL,
    Props::IRRATIONAL,
    Props::IMAGINARY,
    Props::POSITIVE,
    Props::NEGATIVE,
    Props::NONNEGATIVE,
    Props::NONPOSITIVE,
    Props::ZERO,
    Props::NONZERO,
    Props::EVEN,
    Props::ODD,
    Props::PRIME,
    Props::COMPOSITE,
    Props::FINITE,
    Props::INFINITE,
    Props::HERMITIAN,
    Props::ANTIHERMITIAN,
    Props::EXTENDED_REAL,
];

/// The three-valued conjunction (Kleene): false as soon as one part is false, true when both
/// are true, unknown otherwise.
fn conjunction(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

/// Every `<tag>` of `xml` has its `</tag>` in the right order.
fn assert_balanced(xml: &str) {
    let mut open: Vec<&str> = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find('<') {
        let end = start + rest[start..].find('>').expect("an unclosed tag");
        let tag = &rest[start + 1..end];
        if let Some(name) = tag.strip_prefix('/') {
            assert_eq!(open.pop(), Some(name), "in {xml}");
        } else if !tag.ends_with('/') {
            open.push(tag.split(' ').next().unwrap());
        }
        rest = &rest[end + 1..];
    }
    assert!(open.is_empty(), "unclosed {open:?} in {xml}");
}

// ── W15: a query of several properties ───────────────────────────────────────

/// W15.  A query of two properties is their conjunction.  Up to 0.60.0 it was `Some(false)`
/// only when both were known false: of the pairs below every one with one true (or unknown)
/// and one false part was `None` (`POSITIVE | NEGATIVE` of a positive `x`, `POSITIVE |
/// INTEGER` of `1/2`).
///
/// Oracle: logic — the Kleene conjunction of the two one-property answers, for every pair of
/// the 24 properties (276 pairs per expression).
#[test]
fn a_query_of_two_properties_is_the_conjunction_of_the_two_answers() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let n = ctx
        .symbol_with("n", &[Assumption::Integer, Assumption::Negative])
        .unwrap();
    let z = ctx.symbol("z");
    let mut exprs = vec![
        x.clone(),
        n.clone(),
        z.clone(),
        &x + &n,
        &x * &z,
        x.powi(2) + 1,
    ];
    for src in [
        "0",
        "1",
        "-3",
        "7",
        "1/2",
        "-22/7",
        "sqrt(2)",
        "pi",
        "I",
        "1 + I",
        "cos(pi/16)",
        "exp(1) - 3",
        "log(1/2)",
        "oo",
        "zoo",
    ] {
        exprs.push(ctx.parse(src).unwrap());
    }
    let mut false_with_an_open_or_true_part = 0;
    for e in &exprs {
        for (i, &p) in PROPS.iter().enumerate() {
            for &q in &PROPS[i + 1..] {
                let (a, b) = (e.query(p), e.query(q));
                assert_eq!(e.query(p | q), conjunction(a, b), "{e}: {p:?} | {q:?}");
                if conjunction(a, b) == Some(false) && (a != Some(false) || b != Some(false)) {
                    false_with_an_open_or_true_part += 1;
                }
            }
        }
    }
    // The pairs that were `None` up to 0.60.0.
    assert!(false_with_an_open_or_true_part > 1000);
}

/// W15, the cases of the finding and their neighbours.  Up to 0.60.0: `None`, `None`, and
/// `None` for the third (two false parts were `Some(false)` already).
///
/// Oracle: logic; the one-property answers are asserted beside each.
#[test]
fn a_false_part_makes_the_conjunction_false() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert_eq!(x.query(Props::POSITIVE), Some(true));
    assert_eq!(x.query(Props::NEGATIVE), Some(false));
    assert_eq!(x.query(Props::INTEGER), None);
    assert_eq!(x.query(Props::POSITIVE | Props::NEGATIVE), Some(false));
    assert_eq!(x.query(Props::INTEGER | Props::NEGATIVE), Some(false));
    assert_eq!(x.query(Props::POSITIVE | Props::REAL), Some(true));
    assert_eq!(x.query(Props::POSITIVE | Props::INTEGER), None);
    assert_eq!(
        x.query(Props::POSITIVE | Props::REAL | Props::NONZERO | Props::ZERO),
        Some(false)
    );
    // No property asked: the empty conjunction, as before.
    assert_eq!(x.query(Props::empty()), Some(true));
    let half = ctx.rational(1, 2);
    assert_eq!(half.query(Props::POSITIVE | Props::INTEGER), Some(false));
    assert_eq!(half.query(Props::POSITIVE | Props::RATIONAL), Some(true));
}

/// W15 at its source, `Assumptions::query`.  Up to 0.60.0 `POSITIVE | NEGATIVE` of the facts
/// of a positive number was `None`.
///
/// Oracle: logic.
#[test]
fn the_stored_facts_answer_a_conjunction() {
    let mut facts = Assumptions::default();
    facts.assert_true(Props::POSITIVE);
    facts.assert_true(Props::REAL);
    assert_eq!(facts.query(Props::POSITIVE), Some(true));
    assert_eq!(facts.query(Props::NEGATIVE), Some(false));
    assert_eq!(facts.query(Props::POSITIVE | Props::NEGATIVE), Some(false));
    assert_eq!(facts.query(Props::POSITIVE | Props::REAL), Some(true));
    assert_eq!(facts.query(Props::POSITIVE | Props::INTEGER), None);
    assert_eq!(facts.query(Props::NEGATIVE | Props::ZERO), Some(false));
}

/// W15 with a part decided by the value of a constant.  `cos(pi/16)` is positive by its value
/// and not known to be an integer by the rules, so `NEGATIVE | INTEGER` is false and `POSITIVE
/// | REAL` true; up to 0.60.0 both were `None` (a mask with a property outside the
/// value-decided ones was not passed to the value at all).
///
/// Oracle: mpmath `cos(pi/16)` = 0.98078528040323044913 (30 and 50 digits agree).
#[test]
fn a_conjunction_reads_the_value_of_a_constant() {
    let ctx = Context::new();
    let c = ctx.parse("cos(pi/16)").unwrap();
    assert_eq!(c.query(Props::POSITIVE), Some(true));
    assert_eq!(c.query(Props::NEGATIVE | Props::INTEGER), Some(false));
    assert_eq!(c.query(Props::POSITIVE | Props::REAL), Some(true));
    assert_eq!(c.query(Props::POSITIVE | Props::NONZERO), Some(true));
}

// ── W17: one question, one answer ────────────────────────────────────────────

/// W17.  `Context::query` and `Ex::query` are one implementation.  Up to 0.60.0
/// `Context::query` gave the structural answer alone: `None` for the sign of each of these
/// eight constants, which `Ex::query` decided.
///
/// Oracle: mpmath (30 and 50 digits agree): `cos(pi/16)` = 0.98078528, `cos(1)` =
/// 0.54030231, `sin(3)` = 0.14112001, `pi - 3` = 0.14159265, `exp(1) - 3` = -0.28171817,
/// `sqrt(2) - 1` = 0.41421356, `log(2)` = 0.69314718, `zeta(3) - 1` = 0.20205690.
#[test]
fn context_query_is_ex_query() {
    let ctx = Context::new();
    for (src, positive) in [
        ("cos(pi/16)", true),
        ("cos(1)", true),
        ("sin(3)", true),
        ("pi - 3", true),
        ("exp(1) - 3", false),
        ("sqrt(2) - 1", true),
        ("log(2)", true),
        ("zeta(3) - 1", true),
    ] {
        let e = ctx.parse(src).unwrap();
        assert_eq!(ctx.query(&e, Props::POSITIVE), Some(positive), "{src}");
        assert_eq!(ctx.query(&e, Props::NEGATIVE), Some(!positive), "{src}");
        for p in PROPS {
            assert_eq!(ctx.query(&e, p), e.query(p), "{src}: {p:?}");
        }
        assert_eq!(
            ctx.query(&e, Props::POSITIVE | Props::NEGATIVE),
            Some(false),
            "{src}"
        );
    }
    // With symbols too (the readings of `Ex::query`: finiteness is the pole detector).
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let inv = ctx.one() / &x;
    for p in PROPS {
        assert_eq!(ctx.query(&inv, p), inv.query(p), "1/x: {p:?}");
    }
    assert_eq!(ctx.query(&inv, Props::REAL), Some(true));
    assert_eq!(ctx.query(&inv, Props::FINITE), None);
}

/// W17: the cross-context guard of `Context::query` is kept.
#[test]
#[should_panic(expected = "different contexts")]
fn context_query_of_a_foreign_expression_panics() {
    let (a, b) = (Context::new(), Context::new());
    let _ = a.query(&b.int(-7), Props::POSITIVE);
}

// ── W13: an equation that fails at a point ───────────────────────────────────

/// W13.  An equation that fails at a point of the declared domain is no identity.  Up to
/// 0.60.0 every one of these was `None` (only `Ex::equals` was asked).
///
/// Oracle: substitution.  `x = 3` at 0: `0 != 3`.  `x = y` at (0, 1).  `sqrt(x^2) = x` and
/// `abs(x) = x` at -1: `1 != -1`.  `floor(x) = x` for a real `x` at 1/2: `0 != 1/2`.
/// `atan(tan(x)) = x` at 2: mpmath `atan(tan(2))` = -1.1415926535897932 (= 2 - pi).
/// `sin(x) = x` at 1: mpmath `sin(1)` = 0.84147098480789650665.  `x^2 = 2^x` at 1: `1 != 2`
/// (it holds at 2 and at 4).
#[test]
fn an_equation_that_fails_at_a_point_is_no_identity() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let eq = |l: &str, r: &str| Equation::new(ctx.parse(l).unwrap(), ctx.parse(r).unwrap());
    assert_eq!(
        Equation::new(x.clone(), ctx.int(3)).is_identity(),
        Some(false)
    );
    assert_eq!(
        Equation::new(x.clone(), y.clone()).is_identity(),
        Some(false)
    );
    assert_eq!(eq("sqrt(x^2)", "x").is_identity(), Some(false));
    assert_eq!(eq("abs(x)", "x").is_identity(), Some(false));
    assert_eq!(eq("floor(r)", "r").is_identity(), Some(false));
    assert_eq!(eq("atan(tan(x))", "x").is_identity(), Some(false));
    assert_eq!(eq("sin(x)", "x").is_identity(), Some(false));
    assert_eq!(eq("x^2", "2^x").is_identity(), Some(false));
    assert_eq!(eq("x*y", "x + y").is_identity(), Some(false));
    // `is_satisfied` is another question: are the sides equal?  It is `None` for an equation
    // that holds for some values only, and `Some(false)` for one that holds nowhere.
    assert_eq!(Equation::new(x.clone(), ctx.int(3)).is_satisfied(), None);
    assert_eq!(Equation::new(x.clone(), &x + 1).is_satisfied(), Some(false));
    // What was decided before stays decided.
    assert_eq!(Equation::new(x.clone(), &x + 1).is_identity(), Some(false));
    assert_eq!(eq("sin(x)^2 + cos(x)^2", "1").is_identity(), Some(true));
    assert_eq!(eq("(x + 1)^2", "x^2 + 2*x + 1").is_identity(), Some(true));
}

/// W13: only a point of the *declared* domain refutes.  `sqrt(p^2) = p` and `abs(p) = p` hold
/// for a positive `p` (they fail at -1, which is no value of `p`); `floor(n) = n` holds for an
/// integer `n` (it fails at 1/2); `sin(pi*n) = 0` holds for an integer `n` (it fails at 1/2);
/// `log(exp(r)) = r` holds for a real `r` (it fails at `4i`).  None of them may be
/// `Some(false)`.
///
/// Oracle: the identities, each on its domain (textbook).
#[test]
fn a_point_outside_the_declared_domain_refutes_nothing() {
    let ctx = Context::new();
    ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    ctx.symbol_with("n", &[Assumption::Integer]).unwrap();
    ctx.symbol_with("r", &[Assumption::Real]).unwrap();
    let eq = |l: &str, r: &str| Equation::new(ctx.parse(l).unwrap(), ctx.parse(r).unwrap());
    for (l, r) in [
        ("sqrt(p^2)", "p"),
        ("abs(p)", "p"),
        ("floor(n)", "n"),
        ("sin(pi*n)", "0"),
        ("log(exp(r))", "r"),
        ("exp(log(p))", "p"),
        ("cos(2*pi*n)", "1"),
    ] {
        assert_ne!(eq(l, r).is_identity(), Some(false), "{l} = {r}");
    }
    assert_eq!(eq("sqrt(p^2)", "p").is_identity(), Some(true));
}

/// W13: what is neither proved nor refuted stays `None`.  An undefined function has no value
/// at a point; a symbol whose declared domain holds none of the values tried (a
/// non-commutative one) is not given one; a point at which a side has no finite value (`1/x`
/// at 0) is skipped, and the next point refutes.
///
/// Oracle: for `1/x = x`, substitution at 2: `1/2 != 2`.
#[test]
fn an_equation_without_a_usable_point_stays_undecided() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.apply("f", &[&x]).unwrap();
    assert_eq!(Equation::new(f.clone(), ctx.int(3)).is_identity(), None);
    assert_eq!(Equation::new(f, x.clone()).is_identity(), None);
    let a = ctx.symbol_with("A", &[Assumption::NotCommutative]).unwrap();
    assert_eq!(Equation::new(a, ctx.int(3)).is_identity(), None);
    assert_eq!(
        Equation::new(ctx.one() / &x, x.clone()).is_identity(),
        Some(false)
    );
}

// ── W14: primes ──────────────────────────────────────────────────────────────

/// W14.  A non-integer is no prime, and neither is an integer below 2.  Up to 0.60.0
/// `is_prime_value` was `None` for `1/2` and for `pi` (while `is_prime` was `Some(false)`), and
/// `is_prime` and `is_composite` were `None` for `1` and for `-7` (while `is_prime_value` was
/// `Some(false)`).
///
/// Oracle: SymPy 1.14 `Rational(1, 2).is_prime` = False, `pi.is_prime` = False,
/// `Integer(1).is_prime` = False, `Integer(1).is_composite` = False, `Integer(-7).is_prime` =
/// False, `Integer(-7).is_composite` = False, `Integer(0).is_prime` = False.
#[test]
fn a_fraction_and_an_integer_below_two_are_not_prime() {
    let ctx = Context::new();
    for src in ["1/2", "-7/3", "pi", "1", "0", "-1", "-7", "-2"] {
        let e = ctx.parse(src).unwrap();
        assert_eq!(e.is_prime(), Some(false), "is_prime({src})");
        assert_eq!(e.is_prime_value(), Some(false), "is_prime_value({src})");
        assert_eq!(e.is_composite(), Some(false), "is_composite({src})");
        assert_eq!(ctx.query(&e, Props::PRIME), Some(false), "{src}");
    }
    // Decided before, and still: 7, 8, and sums that fold.
    assert_eq!(ctx.int(7).is_prime(), Some(true));
    assert_eq!(ctx.int(7).is_prime_value(), Some(true));
    assert_eq!(ctx.int(8).is_prime(), Some(false));
    assert_eq!(ctx.int(8).is_composite(), Some(true));
    assert_eq!(ctx.parse("3 + 4").unwrap().is_prime_value(), Some(true));
    // Nothing is known about a plain symbol; a symbol declared prime is one.
    assert_eq!(ctx.symbol("x").is_prime_value(), None);
    let p = ctx.symbol_with("p", &[Assumption::Prime]).unwrap();
    assert_eq!(p.is_prime(), Some(true));
    assert_eq!(p.is_prime_value(), Some(true));
}

/// W14.  An integer literal beyond `u64` is decided where the number-theory test is a proof.
/// Up to 0.60.0 `is_prime` was `None` for every integer above 2^64 − 1.
///
/// * Below 3.3·10²⁴ `ntheory::isprime` is deterministic: `2^64 + 13` is prime, `2^67 − 1` is
///   composite.
/// * Beyond it a Miller–Rabin witness proves a composite: `(2^89 − 1)(2^61 − 1)` is no prime.
/// * `2^89 − 1` (6.2·10²⁶) lies above the proved range.  There the test is Baillie–PSW, which
///   is not a proof, so `is_prime` (and `is_composite`) is `None`: a query answers only by a
///   proof.  `is_prime_value` reports the Baillie–PSW verdict, `Some(true)`, as before.
///
/// Oracle: SymPy 1.14 `isprime(2**64 + 13)` = True, `nextprime(2**64)` = 18446744073709551629;
/// `factorint(2**67 - 1)` = {193707721: 1, 761838257287: 1}; `isprime(2**89 - 1)` = True and
/// `isprime(2**61 - 1)` = True (Mersenne primes M89, M61).
#[test]
fn an_integer_beyond_u64_is_decided_where_the_test_is_a_proof() {
    let ctx = Context::new();
    let prime = ctx.parse("2^64 + 13").unwrap();
    assert_eq!(prime.is_prime(), Some(true));
    assert_eq!(prime.is_composite(), Some(false));
    assert_eq!(prime.is_prime_value(), Some(true));
    let cole = ctx.parse("2^67 - 1").unwrap();
    assert_eq!(cole.is_prime(), Some(false));
    assert_eq!(cole.is_composite(), Some(true));
    assert_eq!(cole.is_prime_value(), Some(false));
    let product = ctx.parse("(2^89 - 1)*(2^61 - 1)").unwrap();
    assert_eq!(product.is_prime(), Some(false));
    assert_eq!(product.is_composite(), Some(true));
    assert_eq!(product.is_prime_value(), Some(false));
    let m89 = ctx.parse("2^89 - 1").unwrap();
    assert_eq!(m89.is_prime(), None);
    assert_eq!(m89.is_composite(), None);
    assert_eq!(m89.is_prime_value(), Some(true));
    for e in [&prime, &cole, &product, &m89] {
        assert_eq!(ctx.query(e, Props::PRIME), e.is_prime(), "{e}");
        assert_eq!(e.query(Props::PRIME | Props::POSITIVE), e.is_prime(), "{e}");
    }
}

// ── W39: printed forms that read back ────────────────────────────────────────

/// W39.  The printed form of a `RootSum` reads back as the same expression.  Up to 0.60.0
/// `parse` refused it ("expected expression, got Gt": there was no `RootSum` form).
///
/// Oracle: the round trip, `parse(print(e)) == e`; and for the value, the derivative of the
/// integral read back at `x = 1/2` against the integrand, `1/(1/32 - 1/2 + 1)` = 32/17.
#[test]
fn the_printed_form_of_a_rootsum_reads_back() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "1/(x^5 - x + 1)",
        "x/(x^5 - x + 1)",
        "1/(x^5 + 2*x + 3)",
        "(x^2 + 1)/(x^7 + x + 1)",
    ] {
        let integral = ctx.parse(src).unwrap().integrate(&x);
        let text = integral.to_string();
        assert!(text.contains("RootSum("), "{src}: {text}");
        assert_eq!(ctx.parse(&text).ok().as_ref(), Some(&integral), "{src}");
        // In a context that has never seen the integral.
        let fresh = Context::new();
        assert_eq!(fresh.parse(&text).unwrap().to_string(), text, "{src}");
    }
    let integral = ctx.parse("1/(x^5 - x + 1)").unwrap().integrate(&x);
    let back = ctx.parse(&integral.to_string()).unwrap();
    let slope = back
        .diff(&x)
        .subs(&x, &ctx.rational(1, 2))
        .eval_f64()
        .unwrap();
    assert!((slope - 32.0 / 17.0).abs() < 1e-9, "{slope}");
}

/// W39.  The form itself: `RootSum(poly, var -> expr)`, as printed; the variable is one name.
/// Up to 0.60.0 every `RootSum(…)` was an "unknown function".
///
/// Oracle: the printer (`Display`), whose form the parser reads.
#[test]
fn rootsum_is_read_in_its_printed_form() {
    let ctx = Context::new();
    let src = "RootSum(t^5 - t + 1, t -> t*ln(-t + x))";
    let e = ctx.parse(src).unwrap();
    assert_eq!(e.to_string(), src);
    assert_eq!(
        ctx.parse("rootsum(t^5 - t + 1, t->t*ln(x - t))").unwrap(),
        e
    );
    assert_eq!(ctx.parse(&format!("2*{src} + 1")).unwrap(), &e * 2 + 1);
    // The bound variable is not a free symbol of the sum.
    assert_eq!(e.free_symbols(), vec![ctx.symbol("x")]);
    for bad in [
        "RootSum(t^5 - t + 1, t, t*ln(x - t))",
        "RootSum(t^5 - t + 1, 2 -> t)",
        "RootSum(t^5 - t + 1, t - t)",
        "RootSum(t^5 - t + 1)",
    ] {
        assert!(ctx.parse(bad).is_err(), "{bad}");
    }
}

/// W39, second half.  `RootOf` and `RootSum` have a MathML form, a call under their name as
/// in `to_latex`.  Up to 0.60.0 `to_mathml` refused both ("no presentation for `RootOf`"),
/// although `solve` and `integrate` return them.
///
/// Oracle: the other printer (`to_latex`: `\operatorname{RootOf}\left(x^{5} - x + 1, 0\right)`,
/// `\operatorname{RootSum}\left(…,\, t \mapsto …\right)`) and well-formedness (every tag
/// closed in order).
#[test]
fn rootof_and_rootsum_have_a_mathml_form() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let roots = ctx.parse("x^5 - x + 1").unwrap().solve(&x).unwrap();
    assert_eq!(roots.len(), 5);
    for (k, r) in roots.iter().enumerate() {
        assert!(r.to_latex().contains(r"\operatorname{RootOf}"), "{r}");
        let xml = r.to_mathml().unwrap();
        assert_balanced(&xml);
        assert!(
            xml.contains("<mi>RootOf</mi><mo>&#x2061;</mo>")
                && xml.ends_with(&format!(
                    "<mo>,</mo><mn>{k}</mn><mo>)</mo></mrow></mrow></math>"
                )),
            "{xml}"
        );
    }
    // The variable is written where the polynomial has another symbol.
    let with_var = ctx.parse("RootOf(x^5 - a*x - 1, x, 2)").unwrap();
    let xml = with_var.to_mathml().unwrap();
    assert_balanced(&xml);
    assert!(
        xml.ends_with("<mo>,</mo><mi>x</mi><mo>,</mo><mn>2</mn><mo>)</mo></mrow></mrow></math>"),
        "{xml}"
    );
    // A RootSum, as `integrate` returns it and inside a larger expression.
    let integral = ctx.parse("1/(x^5 - x + 1)").unwrap().integrate(&x);
    assert!(integral.to_latex().contains(r"\operatorname{RootSum}"));
    for e in [integral.clone(), &integral * &x + 1] {
        let xml = e.to_mathml().unwrap();
        assert_balanced(&xml);
        assert!(
            xml.contains("<mi>RootSum</mi><mo>&#x2061;</mo>") && xml.contains("<mo>&#x21A6;</mo>"),
            "{xml}"
        );
    }
}
