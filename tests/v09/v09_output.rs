//! symplex 0.9.1 — more output targets and interchange: relation parsing
//! (`Context::parse_bool`), implicit function application
//! (`Context::parse_implicit`), Presentation MathML (`to_mathml`), the
//! `srepr`/DOT tree dumps (`to_srepr`, `to_dot`), and Python / NumPy /
//! Julia code generation (`to_python`, `to_numpy`, `to_julia`, `*_fn`).
//!
//! SymPy references (1.14, `symplex/.venv`):
//!
//! ```text
//! >>> mathml(x**2+1, printer='presentation')
//! <mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>1</mn></mrow>
//! >>> srepr(2*x+1)
//! Add(Mul(Integer(2), Symbol('x')), Integer(1))
//! >>> pycode(sin(x)**2+exp(x))
//! math.exp(x) + math.sin(x)**2
//! >>> NumPyPrinter().doprint(sin(x))
//! numpy.sin(x)
//! >>> julia_code(sin(x)**2+exp(x))
//! exp(x) + sin(x) .^ 2
//! >>> parse_expr('2x + 3(y-1)', transformations=T)   # implicit_multiplication_application
//! 2*x + 3*y - 3
//! >>> parse_expr('sin x cos y', transformations=T)
//! sin(x*cos(y))        # symplex reads sin(x)*cos(y); see parse_implicit docs
//! ```
//!
//! The Python code is executed with the venv interpreter when it is
//! present (`symplex/.venv/bin/python`); otherwise those checks are skipped
//! with a note, never failed.

use std::process::Command;

use symplex::prelude::*;

const PYTHON: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/.venv/bin/python");

/// Run `code` (a Python program) with the venv interpreter and return its
/// trimmed stdout, or `None` (with a note) when the interpreter is absent.
fn run_python(code: &str) -> Option<String> {
    if !std::path::Path::new(PYTHON).is_file() {
        eprintln!("note: {PYTHON} not found; skipping the Python execution check");
        return None;
    }
    let out = Command::new(PYTHON)
        .arg("-c")
        .arg(code)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {PYTHON}: {e}"));
    assert!(
        out.status.success(),
        "python failed on:\n{code}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Every `<tag>` in `xml` is closed by a matching `</tag>` in LIFO order.
fn assert_well_formed(xml: &str) {
    let mut stack: Vec<&str> = Vec::new();
    let mut rest = xml;
    while let Some(open) = rest.find('<') {
        let close = rest[open..]
            .find('>')
            .unwrap_or_else(|| panic!("unterminated tag in {xml}"));
        let tag = &rest[open + 1..open + close];
        if let Some(name) = tag.strip_prefix('/') {
            assert_eq!(stack.pop(), Some(name), "tag mismatch in {xml}");
        } else {
            stack.push(tag.split(' ').next().unwrap_or(tag));
        }
        rest = &rest[open + close + 1..];
    }
    assert!(stack.is_empty(), "unclosed tags {stack:?} in {xml}");
    // Only numeric character references: no `&` outside `&#…;`/`&gt;`/`&lt;`/`&amp;`.
    for (i, _) in xml.match_indices('&') {
        let tail = &xml[i..];
        assert!(
            tail.starts_with("&#")
                || tail.starts_with("&gt;")
                || tail.starts_with("&lt;")
                || tail.starts_with("&amp;"),
            "bare `&` in {xml}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Parsing relations and Boolean connectives
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_bool_relations_round_trip_through_display_and_lean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = ctx.parse_bool("x > 0 & x < 1").unwrap();
    // Structurally the same as the API constructors …
    assert_eq!(p, x.gt(&ctx.int(0)).and(&x.lt(&ctx.int(1))));
    // … and round-trips through `Display` (`x < 1` is stored as `1 > x`)
    // and the Lean printer.
    assert_eq!(p.to_string(), "x > 0 & 1 > x");
    assert_eq!(ctx.parse_bool(&p.to_string()).unwrap(), p);
    assert_eq!(p.to_lean().unwrap(), "0 < x ∧ x < 1");
}

#[test]
fn parse_bool_precedence_is_mathematical_not_pythons() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let zero = ctx.int(0);
    // Comparisons bind tighter than `and`, which binds tighter than `or`;
    // `not` is prefix and takes one relation.
    let p = ctx.parse_bool("not x == 0 and y >= 0 or x != y").unwrap();
    let expected = x.eq_expr(&zero).not().and(&y.ge(&zero)).or(&x.ne_expr(&y));
    assert_eq!(p, expected);
    assert_eq!(p.to_string(), "!(x == 0) & y >= 0 | x != y");
    // Alternative spellings and SymPy's function forms.
    assert_eq!(
        ctx.parse_bool("~(x == 0) && y >= 0 || x != y").unwrap(),
        expected
    );
    assert_eq!(
        ctx.parse_bool("Or(And(Not(Eq(x, 0)), Ge(y, 0)), Ne(x, y))")
            .unwrap(),
        expected
    );
    // Arithmetic inside relations, parentheses regroup.
    assert_eq!(
        ctx.parse_bool("(x + 1)^2 <= 2*y | (x > 0 | y > 0) & x < 1")
            .unwrap(),
        (&x + 1)
            .powi(2)
            .le(&(2 * &y))
            .or(&x.gt(&zero).or(&y.gt(&zero)).and(&x.lt(&ctx.int(1))))
    );
    // Errors are `Result`s, not panics.
    assert!(ctx.parse_bool("x + 1").is_err());
    assert!(ctx.parse_bool("0 < x < 1").is_err());
    assert!(ctx.parse_bool("(x > 0) + 1").is_err());
    // The numeric parser is unchanged: relations are rejected, words stay symbols.
    assert!(ctx.parse("x > 0").is_err());
    assert_eq!(ctx.parse("and").unwrap(), ctx.symbol("and"));
}

#[test]
fn parse_implicit_products_and_application() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    // SymPy: parse_expr('2x + 3(y-1)') == 2*x + 3*y - 3
    let e = ctx.parse_implicit("2x + 3(y-1)").unwrap();
    assert_eq!(e, 2 * &x + 3 * (&y - 1));
    assert_eq!(e.expand(), 2 * &x + 3 * &y - 3);
    assert_eq!(ctx.parse_implicit("x y z").unwrap(), &x * &y * &z);
    assert_eq!(
        ctx.parse_implicit("(x+1)(x-1)").unwrap(),
        (&x + 1) * (&x - 1)
    );
    assert_eq!(ctx.parse_implicit("x(x+1)").unwrap(), &x * (&x + 1));
    // Function application without parentheses.
    assert_eq!(ctx.parse_implicit("sin x").unwrap(), x.sin());
    assert_eq!(ctx.parse_implicit("2 sin x").unwrap(), 2 * &x.sin());
    assert_eq!(ctx.parse_implicit("sin 2x").unwrap(), (2 * &x).sin());
    assert_eq!(ctx.parse_implicit("sin x^2").unwrap(), x.powi(2).sin());
    assert_eq!(ctx.parse_implicit("sin x + 1").unwrap(), &x.sin() + 1);
    // Resolved ambiguity: the argument stops at the next function name.
    assert_eq!(
        ctx.parse_implicit("sin x cos y").unwrap(),
        &x.sin() * &y.cos()
    );
    assert_eq!(
        ctx.parse_implicit("e^x sin x").unwrap(),
        &x.exp() * &x.sin()
    );
    // A textbook function name needs an argument; short ambiguous names
    // (`re`, `im`, `arg`, …) are symbols unless parenthesised.
    assert!(ctx.parse_implicit("sin + 1").is_err());
    assert_eq!(ctx.parse_implicit("re x").unwrap(), ctx.symbol("re") * &x);
    // The strict parser is unchanged: `sin x` is the product of the symbol
    // `sin` and `x` there, and `x(x+1)` is an unknown-function error.
    assert_eq!(ctx.parse("sin x").unwrap(), ctx.symbol("sin") * &x);
    assert!(ctx.parse("x(x+1)").is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Presentation MathML
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mathml_presentation_structure() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // SymPy: <mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>1</mn></mrow>
    let xml = (&x.powi(2) + 1).to_mathml().unwrap();
    assert_eq!(
        xml,
        "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">\
         <mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>1</mn></mrow></math>"
    );
    assert_well_formed(&xml);

    // Fractions, roots, function application, Greek letters, subscripts.
    let e = ctx.parse("sin(alpha_1)/2 + sqrt(x) - 1/x^3").unwrap();
    let xml = e.to_mathml().unwrap();
    assert_well_formed(&xml);
    for needle in [
        "<mfrac>",
        "<msqrt><mi>x</mi></msqrt>",
        "<mi>sin</mi><mo>&#x2061;</mo>",
        "<msub><mi>&#x3B1;</mi><mn>1</mn></msub>",
        "<mfrac><mn>1</mn><msup><mi>x</mi><mn>3</mn></msup></mfrac>",
        "<mo>-</mo>",
    ] {
        assert!(xml.contains(needle), "missing {needle} in {xml}");
    }
    assert!(
        !xml.contains("<mfenced"),
        "explicit parentheses only: {xml}"
    );

    // Parenthesisation mirrors `to_latex`: compound bases and sums inside
    // products are wrapped in `<mo>(</mo>…<mo>)</mo>`.
    let e = ctx.parse("(x+1)^2 * y").unwrap();
    assert_eq!(e.to_latex(), r"y \left(x + 1\right)^{2}");
    let xml = e.to_mathml().unwrap();
    assert!(
        xml.contains("<msup><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow><mn>2</mn></msup>"),
        "{xml}"
    );
    assert_well_formed(&xml);

    // Relations render for `BoolEx`.
    let p = x.gt(&ctx.int(0)).and(&x.lt(&ctx.int(1)));
    let xml = p.to_mathml().unwrap();
    assert!(
        xml.contains("<mrow><mi>x</mi><mo>&gt;</mo><mn>0</mn></mrow><mo>&#x2227;</mo><mrow><mn>1</mn><mo>&gt;</mo><mi>x</mi></mrow>"),
        "{xml}"
    );
    assert_well_formed(&xml);

    // Unrenderable nodes are `NotImplemented`, not a guess.
    let r = ctx.parse("RootOf(x^5 - x - 1, 0)").unwrap();
    assert!(matches!(
        r.to_mathml(),
        Err(SymplexError::NotImplemented(_))
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. srepr and DOT
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn srepr_is_the_exact_constructor_form() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // SymPy: Add(Mul(Integer(2), Symbol('x')), Integer(1)); symplex prints
    // the arena's canonical child order (number first).
    assert_eq!(
        (2 * &x + 1).to_srepr(),
        "Add(Integer(1), Mul(Integer(2), Symbol('x')))"
    );
    assert_eq!(
        (&x.sin().powi(2) + &x.exp()).to_srepr(),
        "Add(Pow(sin(Symbol('x')), Integer(2)), exp(Symbol('x')))"
    );
    assert_eq!(
        (&x / 2 - &y.sqrt()).to_srepr(),
        "Add(Mul(Integer(-1), Pow(Symbol('y'), Rational(1, 2))), Mul(Rational(1, 2), Symbol('x')))"
    );
    assert_eq!(
        (ctx.pi() + ctx.e() + ctx.i_unit()).to_srepr(),
        "Add(pi, E, I)"
    );
    assert_eq!(
        x.gt(&ctx.int(0)).to_srepr(),
        "StrictGreaterThan(Symbol('x'), Integer(0))"
    );
    assert_eq!(
        ctx.parse("Integral(exp(-x^2), x, 0, inf)")
            .unwrap()
            .to_srepr(),
        "DefiniteIntegral(exp(Mul(Integer(-1), Pow(Symbol('x'), Integer(2)))), Symbol('x'), Integer(0), oo)"
    );
    assert_eq!(
        x.bessel_j(&ctx.int(0)).to_srepr(),
        "besselj(Integer(0), Symbol('x'))"
    );
    let pw = Ex::piecewise(&[(&x, &x.gt(&ctx.int(0))), (&(-&x), &x.le(&ctx.int(0)))]);
    assert_eq!(
        pw.to_srepr(),
        "Piecewise((Symbol('x'), StrictGreaterThan(Symbol('x'), Integer(0))), \
         (Mul(Integer(-1), Symbol('x')), GreaterThan(Integer(0), Symbol('x'))))"
    );
    // Total: every node kind prints (the tree is the same one `to_json` uses).
    let deep = ctx
        .parse("Limit(sin(x)/x, x, 0) + Sum(k^2, k=1..n) + RootOf(x^5 - x - 1, 0)")
        .unwrap();
    let s = deep.to_srepr();
    assert!(
        s.contains("Limit(") && s.contains("Sum(") && s.contains("RootOf("),
        "{s}"
    );
}

#[test]
fn dot_is_a_deterministic_digraph_of_the_tree() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let dot = (2 * &x + 1).to_dot();
    assert_eq!(
        dot,
        "digraph {\n\
         \x20   ordering=out;\n\
         \x20   rankdir=TD;\n\
         \x20   n0 [label=\"Add\"];\n\
         \x20   n1 [label=\"Integer(1)\"];\n\
         \x20   n2 [label=\"Mul\"];\n\
         \x20   n3 [label=\"Integer(2)\"];\n\
         \x20   n4 [label=\"Symbol('x')\"];\n\
         \x20   n0 -> n1;\n\
         \x20   n0 -> n2;\n\
         \x20   n2 -> n3;\n\
         \x20   n2 -> n4;\n\
         }\n"
    );
    // One node per tree position, one edge per child; repeated
    // subexpressions appear once per occurrence (a tree, not the DAG).
    let e = &x.sin() * &x.sin() + &x.sin();
    let dot = e.to_dot();
    assert_eq!(dot.matches("[label=\"sin\"]").count(), 2, "{dot}");
    assert_eq!(
        dot.matches(" -> ").count(),
        dot.matches("[label=").count() - 1
    );
    assert_eq!(e.to_dot(), dot, "deterministic");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Python / NumPy / Julia code generation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn to_python_matches_sympy_pycode_shape() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // SymPy: math.exp(x) + math.sin(x)**2 (SymPy's term order differs).
    assert_eq!(
        (&x.sin().powi(2) + &x.exp()).to_python().unwrap(),
        "math.sin(x)**2 + math.exp(x)"
    );
    assert_eq!((&x.powi(2) + 1).to_python().unwrap(), "x**2 + 1");
    assert_eq!((&x / 2).to_python().unwrap(), "x/2");
    assert_eq!(x.sqrt().to_python().unwrap(), "math.sqrt(x)");
    assert_eq!((ctx.pi() * ctx.e()).to_python().unwrap(), "math.pi*math.e");
    assert_eq!(
        (&x.gamma() + &x.erf()).to_python().unwrap(),
        "math.gamma(x) + math.erf(x)"
    );
    assert_eq!((&x / &y).to_python().unwrap(), "x/y");
    assert_eq!((1 / &x).to_python().unwrap(), "1/x");
    assert_eq!((-&x).to_python().unwrap(), "-x");
    assert_eq!((&x - &y).to_python().unwrap(), "x - y");
    assert_eq!(x.powi(-2).to_python().unwrap(), "x**(-2)");
    assert_eq!(
        x.gt(&ctx.int(0))
            .and(&x.lt(&ctx.int(1)))
            .to_python()
            .unwrap(),
        "x > 0 and 1 > x"
    );
    // Unsupported → NotImplemented, never a wrong formula.
    assert!(matches!(
        x.bessel_j(&ctx.int(0)).to_python(),
        Err(SymplexError::NotImplemented(_))
    ));
    assert!(matches!(
        ctx.parse("Integral(x, x)").unwrap().to_python(),
        Err(SymplexError::NotImplemented(_))
    ));
    assert!(matches!(
        (&x + ctx.i_unit()).to_python(),
        Err(SymplexError::NotImplemented(_))
    ));
}

#[test]
fn to_python_is_executable_and_agrees_with_eval_f64() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let cases: Vec<Ex> = vec![
        &x.sin().powi(2) + &x.exp(),
        (&x + 1) / (&y - 3) - &x.powi(3) / 2,
        (&x * ctx.pi()).sqrt() * &y.ln() + &x.atan2(&y),
        (&x.gamma() + &x.erf() - &x.floor()) * ctx.e(),
        Ex::piecewise(&[
            (&x.powi(2), &x.lt(&ctx.int(0))),
            (&x.sqrt(), &x.ge(&ctx.int(0))),
        ]),
        (2 * &x + 3 * &y).powi(2).cos() - &x.abs() / 3,
        x.exp().pow(&y) + &x.sign(),
    ];
    let (xv, yv) = (0.7_f64, 1.3_f64);
    let mut program = String::from("import math\n");
    program.push_str(&format!("x = {xv:?}\ny = {yv:?}\n"));
    let mut expected: Vec<f64> = Vec::new();
    for e in &cases {
        let code = e.to_python().unwrap();
        program.push_str(&format!("print(repr(float({code})))\n"));
        expected.push(e.subs_map_with(&[(&x, xv), (&y, yv)]).eval_f64().unwrap());
    }
    // A function with CSE temporaries, called on the same point.
    let f = &x.sin().powi(2) + &x.sin() * &y + (&x.sin() + &y).exp();
    let def = f.to_python_fn("f", &["x", "y"]).unwrap();
    assert!(
        def.starts_with("def f(x, y):\n    t0 = math.sin(x)\n"),
        "{def}"
    );
    program.push_str(&def);
    program.push_str("print(repr(float(f(x, y))))\n");
    expected.push(f.subs_map_with(&[(&x, xv), (&y, yv)]).eval_f64().unwrap());

    let Some(out) = run_python(&program) else {
        return;
    };
    let got: Vec<f64> = out
        .lines()
        .map(|l| l.parse::<f64>().unwrap_or_else(|e| panic!("{l}: {e}")))
        .collect();
    assert_eq!(got.len(), expected.len(), "{out}");
    for ((g, e), c) in got.iter().zip(&expected).zip(
        cases
            .iter()
            .map(|c| c.to_python().unwrap())
            .chain(std::iter::once(def.clone())),
    ) {
        assert!(
            (g - e).abs() <= 1e-12 * e.abs().max(1.0),
            "python {g} vs symplex {e} for\n{c}"
        );
    }
}

#[test]
fn to_numpy_and_to_julia_use_their_libraries() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // SymPy: NumPyPrinter().doprint(sin(x)) == 'numpy.sin(x)'
    assert_eq!(x.sin().to_numpy().unwrap(), "numpy.sin(x)");
    assert_eq!(
        (&x.sin().powi(2) + &x.exp()).to_numpy().unwrap(),
        "numpy.sin(x)**2 + numpy.exp(x)"
    );
    assert_eq!((&x.powi(2) + 1).to_numpy().unwrap(), "x**2 + 1");
    assert_eq!(ctx.pi().to_numpy().unwrap(), "numpy.pi");
    assert_eq!(x.sqrt().to_numpy().unwrap(), "numpy.sqrt(x)");
    assert_eq!(x.asin().to_numpy().unwrap(), "numpy.arcsin(x)");
    let pw = Ex::piecewise(&[(&x, &x.gt(&ctx.int(0))), (&(-&x), &x.le(&ctx.int(0)))]);
    assert_eq!(
        pw.to_numpy().unwrap(),
        "numpy.select([numpy.greater(x, 0), numpy.greater_equal(0, x)], [x, -x], default=numpy.nan)"
    );
    assert_eq!(
        x.gt(&ctx.int(0))
            .and(&y.gt(&ctx.int(0)))
            .to_numpy()
            .unwrap(),
        "numpy.logical_and(numpy.greater(x, 0), numpy.greater(y, 0))"
    );
    assert_eq!(
        x.exp().to_numpy_fn("g", &["x"]).unwrap(),
        "def g(x):\n    return numpy.exp(x)\n"
    );
    // NumPy itself has no gamma: refused rather than a non-vectorised guess.
    assert!(matches!(
        x.gamma().to_numpy(),
        Err(SymplexError::NotImplemented(_))
    ));

    // SymPy: julia_code(sin(x)**2+exp(x)) == 'exp(x) + sin(x) .^ 2'
    assert_eq!(
        (&x.sin().powi(2) + &x.exp()).to_julia().unwrap(),
        "sin(x)^2 + exp(x)"
    );
    assert_eq!((&x.powi(2) + ctx.pi()).to_julia().unwrap(), "x^2 + pi");
    assert_eq!((ctx.e() * &x).to_julia().unwrap(), "x*ℯ");
    assert_eq!(pw.to_julia().unwrap(), "(x > 0 ? x : (0 >= x ? -x : NaN))");
    assert_eq!(
        (&x.sin().powi(2) + &x.sin() * &y)
            .to_julia_fn("f", &["x", "y"])
            .unwrap(),
        "function f(x, y)\n    t0 = sin(x)\n    return t0^2 + t0*y\nend\n"
    );
    assert!(matches!(
        (&x + &y).to_julia_fn("f", &["x"]),
        Err(SymplexError::FreeSymbol { .. })
    ));
}
