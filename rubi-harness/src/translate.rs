//! Maxima expression text → text that `symplex::parse` accepts.
//!
//! The Maxima source is parsed into a small AST (so that argument extents
//! and arities are known) and printed back in symplex syntax:
//!
//! - `%e`, `%pi`, `%i`, `%gamma`, `%phi` become `E`, `pi`, `I`,
//!   `EulerGamma`, `GoldenRatio`;
//! - a bare identifier is a *symbol*.  Names that symplex's parser reads as
//!   constants (`e`, `i`, `E`, `I`, `pi`, `inf`, …) are renamed by
//!   appending `_` (`e` → `e_`), so the parameter `e` of the Rubi files does
//!   not become Euler's number;
//! - `log` → `ln`; `asec`/`acsc`/`acoth`/`asech`/`acsch` of `u` are
//!   rewritten as `acos`/`asin`/`atanh`/`acosh`/`asinh` of `1/u` (symplex
//!   has no dedicated nodes for them; these are also the Mathematica and
//!   Maxima definitions);
//! - special functions are mapped to symplex's names where the conventions
//!   agree (`Si`, `Ci`, `Shi`, `Chi`, `Ei`, `Li` → `li`, `FresnelS` →
//!   `fresnels`, `ProductLog` → `lambertw`, `GAMMA(a, z)` → `uppergamma`,
//!   `polylog`, `elliptic_f(φ, m)`, …);
//! - anything else (hypergeometric and Appell functions, incomplete
//!   `elliptic_e`/`elliptic_pi`, abstract functions `f(x)`, formal
//!   `Derivative`s, Maxima lists, …) is [`Unsupported`], with a reason.

use std::collections::BTreeSet;

/// Why an expression could not be translated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported {
    /// Short aggregatable key, e.g. `function hypergeometric/3`.
    pub reason: String,
    /// Free-form detail.
    pub detail: String,
}

impl Unsupported {
    fn new(reason: impl Into<String>, detail: impl Into<String>) -> Self {
        Unsupported {
            reason: reason.into(),
            detail: detail.into(),
        }
    }
}

/// A translated expression.
#[derive(Debug, Clone)]
pub struct Translation {
    /// Text for `symplex::parse`.
    pub text: String,
    /// The (renamed) symbol names that occur in `text`.
    pub symbols: BTreeSet<String>,
}

/// Translate one Maxima expression.
pub fn translate(src: &str) -> Result<Translation, Unsupported> {
    let node = parse(src)?;
    let mut out = String::with_capacity(src.len() + src.len() / 4);
    let mut symbols = BTreeSet::new();
    emit(&node, &mut out, &mut symbols)?;
    Ok(Translation { text: out, symbols })
}

/// Translate the variable field (a plain identifier).
pub fn translate_variable(src: &str) -> Result<String, Unsupported> {
    match parse(src)? {
        Node::Sym(name) => Ok(rename(&name)),
        _ => Err(Unsupported::new(
            "variable is not a symbol",
            src.to_string(),
        )),
    }
}

/// Names symplex's parser treats as constants; as Maxima symbols they are
/// renamed.
const COLLIDING: &[&str] = &[
    "e",
    "E",
    "i",
    "I",
    "pi",
    "Pi",
    "PI",
    "inf",
    "Inf",
    "oo",
    "zoo",
    "nan",
    "EulerGamma",
    "euler_gamma",
    "Catalan",
    "GoldenRatio",
    "golden_ratio",
];

/// The symplex name of the Maxima symbol `name`.
pub fn rename(name: &str) -> String {
    if COLLIDING.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

/// The Maxima name of a (possibly renamed) symplex symbol.
pub fn original_name(name: &str) -> &str {
    match name.strip_suffix('_') {
        Some(base) if COLLIDING.contains(&base) => base,
        _ => name,
    }
}

// ─── AST ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Node {
    /// A number already in symplex syntax (an integer or a decimal, or a
    /// parenthesised `(m*10^(k))` for an exponent float).
    Num(String),
    /// A Maxima identifier used as a symbol (not yet renamed).
    Sym(String),
    /// A constant, spelled as symplex spells it.
    Const(&'static str),
    Neg(Box<Node>),
    /// Terms with their sign: `true` = subtracted.
    Add(Vec<(bool, Node)>),
    /// Factors with their operator: `true` = divided.
    Mul(Vec<(bool, Node)>),
    Pow(Box<Node>, Box<Node>),
    Call(String, Vec<Node>),
    List(Vec<Node>),
    Factorial(Box<Node>),
}

// ─── Tokenizer ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(String),
    Ident(String),
    Op(char),
    Eof,
}

fn tokenize(src: &str) -> Result<Vec<Tok>, Unsupported> {
    let b = src.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit))
        {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            let int_part = &src[start..i];
            let mut frac_part = "";
            let mut is_float = false;
            if i < b.len() && b[i] == b'.' {
                is_float = true;
                i += 1;
                let fs = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                frac_part = &src[fs..i];
            }
            let mut exponent: Option<String> = None;
            if i < b.len() && matches!(b[i], b'e' | b'E' | b'b' | b'B' | b'd' | b'D') {
                let mut j = i + 1;
                let mut sign = "";
                if j < b.len() && matches!(b[j], b'+' | b'-') {
                    if b[j] == b'-' {
                        sign = "-";
                    }
                    j += 1;
                }
                let ds = j;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                if j > ds {
                    exponent = Some(format!("{sign}{}", &src[ds..j]));
                    i = j;
                }
            }
            let int_part = if int_part.is_empty() { "0" } else { int_part };
            let mantissa = if is_float && !frac_part.is_empty() {
                format!("{int_part}.{frac_part}")
            } else {
                int_part.to_string()
            };
            toks.push(Tok::Num(match exponent {
                Some(e) => format!("({mantissa}*10^({e}))"),
                None => mantissa,
            }));
        } else if c.is_ascii_alphabetic() || c == b'_' || c == b'%' {
            let start = i;
            i += 1;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            toks.push(Tok::Ident(src[start..i].to_string()));
        } else if c == b'*' && b.get(i + 1) == Some(&b'*') {
            toks.push(Tok::Op('^'));
            i += 2;
        } else if b"+-*/^()[],!".contains(&c) {
            toks.push(Tok::Op(c as char));
            i += 1;
        } else {
            let ch = src[i..].chars().next().unwrap_or('?');
            return Err(Unsupported::new(
                format!("maxima syntax `{ch}`"),
                format!("unexpected character `{ch}` at byte {i}"),
            ));
        }
    }
    toks.push(Tok::Eof);
    Ok(toks)
}

// ─── Parser ─────────────────────────────────────────────────────────────

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    depth: usize,
}

/// Deeper nesting is refused rather than risking the stack.
const MAX_DEPTH: usize = 500;

fn parse(src: &str) -> Result<Node, Unsupported> {
    let mut p = Parser {
        toks: tokenize(src)?,
        pos: 0,
        depth: 0,
    };
    let n = p.expr()?;
    if p.peek() != &Tok::Eof {
        return Err(p.err(format!("trailing {:?}", p.peek())));
    }
    Ok(n)
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos]
    }

    fn next(&mut self) -> Tok {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn is_op(&self, c: char) -> bool {
        self.peek() == &Tok::Op(c)
    }

    fn err(&self, msg: String) -> Unsupported {
        Unsupported::new("maxima parse error", format!("{msg} (token {})", self.pos))
    }

    fn expect(&mut self, c: char) -> Result<(), Unsupported> {
        if self.is_op(c) {
            self.next();
            Ok(())
        } else {
            Err(self.err(format!("expected `{c}`, found {:?}", self.peek())))
        }
    }

    fn enter(&mut self) -> Result<(), Unsupported> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            Err(Unsupported::new("nesting too deep", String::new()))
        } else {
            Ok(())
        }
    }

    /// sum := term (('+'|'-') term)*
    fn expr(&mut self) -> Result<Node, Unsupported> {
        self.enter()?;
        let first = self.term()?;
        let mut terms = vec![(false, first)];
        while self.is_op('+') || self.is_op('-') {
            let neg = self.next() == Tok::Op('-');
            terms.push((neg, self.term()?));
        }
        self.depth -= 1;
        Ok(if terms.len() == 1 {
            terms.pop().map(|t| t.1).unwrap_or(Node::Num("0".into()))
        } else {
            Node::Add(terms)
        })
    }

    /// term := unary (('*'|'/') unary)*
    fn term(&mut self) -> Result<Node, Unsupported> {
        let first = self.unary()?;
        let mut factors = vec![(false, first)];
        while self.is_op('*') || self.is_op('/') {
            let div = self.next() == Tok::Op('/');
            factors.push((div, self.unary()?));
        }
        Ok(if factors.len() == 1 {
            factors.pop().map(|t| t.1).unwrap_or(Node::Num("1".into()))
        } else {
            Node::Mul(factors)
        })
    }

    /// unary := ('-'|'+') unary | power
    fn unary(&mut self) -> Result<Node, Unsupported> {
        if self.is_op('-') {
            self.next();
            self.enter()?;
            let n = self.unary()?;
            self.depth -= 1;
            return Ok(Node::Neg(Box::new(n)));
        }
        if self.is_op('+') {
            self.next();
            return self.unary();
        }
        self.power()
    }

    /// power := postfix ('^' unary)?   (right-associative; `x^-2` allowed)
    fn power(&mut self) -> Result<Node, Unsupported> {
        let base = self.postfix()?;
        if self.is_op('^') {
            self.next();
            self.enter()?;
            let exp = self.unary()?;
            self.depth -= 1;
            return Ok(Node::Pow(Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn postfix(&mut self) -> Result<Node, Unsupported> {
        let mut n = self.atom()?;
        while self.is_op('!') {
            self.next();
            n = Node::Factorial(Box::new(n));
        }
        Ok(n)
    }

    fn atom(&mut self) -> Result<Node, Unsupported> {
        match self.next() {
            Tok::Num(s) => Ok(Node::Num(s)),
            Tok::Ident(name) => {
                if self.is_op('(') {
                    self.next();
                    let args = self.args(')')?;
                    if self.is_op('(') {
                        // `Derivative(1)(f)(x)`: an applied operator.
                        return Err(Unsupported::new(
                            format!("formal {name}(..)(..) application"),
                            String::new(),
                        ));
                    }
                    return Ok(Node::Call(name, args));
                }
                match name.as_str() {
                    "%e" => Ok(Node::Const("E")),
                    "%pi" => Ok(Node::Const("pi")),
                    "%i" => Ok(Node::Const("I")),
                    "%gamma" => Ok(Node::Const("EulerGamma")),
                    "%phi" => Ok(Node::Const("GoldenRatio")),
                    _ if name.starts_with('%') => Err(Unsupported::new(
                        format!("maxima constant {name}"),
                        String::new(),
                    )),
                    _ => Ok(Node::Sym(name)),
                }
            }
            Tok::Op('(') => {
                let n = self.expr()?;
                self.expect(')')?;
                Ok(n)
            }
            Tok::Op('[') => Ok(Node::List(self.args(']')?)),
            t => Err(self.err(format!("unexpected {t:?}"))),
        }
    }

    /// Comma-separated expressions up to the closing `close` (consumed).
    fn args(&mut self, close: char) -> Result<Vec<Node>, Unsupported> {
        let mut args = Vec::new();
        if self.is_op(close) {
            self.next();
            return Ok(args);
        }
        loop {
            args.push(self.expr()?);
            if self.is_op(',') {
                self.next();
                continue;
            }
            self.expect(close)?;
            return Ok(args);
        }
    }
}

// ─── Printer ────────────────────────────────────────────────────────────

fn prec(n: &Node) -> u8 {
    match n {
        Node::Add(_) => 1,
        Node::Mul(_) => 2,
        Node::Neg(_) => 3,
        Node::Pow(..) => 4,
        _ => 5,
    }
}

fn emit_wrapped(
    n: &Node,
    wrap: bool,
    out: &mut String,
    syms: &mut BTreeSet<String>,
) -> Result<(), Unsupported> {
    if wrap {
        out.push('(');
        emit(n, out, syms)?;
        out.push(')');
        Ok(())
    } else {
        emit(n, out, syms)
    }
}

fn emit(n: &Node, out: &mut String, syms: &mut BTreeSet<String>) -> Result<(), Unsupported> {
    match n {
        Node::Num(s) => out.push_str(s),
        Node::Sym(name) => {
            let r = rename(name);
            out.push_str(&r);
            syms.insert(r);
        }
        Node::Const(c) => out.push_str(c),
        Node::Neg(x) => {
            out.push('-');
            emit_wrapped(x, prec(x) < 4, out, syms)?;
        }
        Node::Add(terms) => {
            for (i, (neg, t)) in terms.iter().enumerate() {
                if i > 0 {
                    out.push_str(if *neg { " - " } else { " + " });
                } else if *neg {
                    out.push('-');
                }
                let wrap = match t {
                    Node::Add(_) => i > 0 || *neg,
                    Node::Neg(_) => i > 0,
                    _ => false,
                };
                emit_wrapped(t, wrap, out, syms)?;
            }
        }
        Node::Mul(factors) => {
            for (i, (div, f)) in factors.iter().enumerate() {
                if i > 0 {
                    out.push(if *div { '/' } else { '*' });
                }
                // Later factors keep parentheses around sums, products and
                // negations (`a/(b*c)`, `a*(-b)`).
                let wrap = if i == 0 { prec(f) < 2 } else { prec(f) < 4 };
                emit_wrapped(f, wrap, out, syms)?;
            }
        }
        Node::Pow(b, e) => {
            emit_wrapped(b, prec(b) < 5 || is_signed_or_float(b), out, syms)?;
            out.push('^');
            emit_wrapped(e, prec(e) < 5 || is_signed_or_float(e), out, syms)?;
        }
        Node::Factorial(x) => {
            out.push_str("factorial(");
            emit(x, out, syms)?;
            out.push(')');
        }
        Node::List(_) => {
            return Err(Unsupported::new(
                "maxima list",
                "list outside a known function",
            ));
        }
        Node::Call(name, args) => emit_call(name, args, out, syms)?,
    }
    Ok(())
}

/// A decimal literal as a power's base or exponent is parenthesised for
/// readability (`(0.5)^x`); exponent floats are already parenthesised.
fn is_signed_or_float(n: &Node) -> bool {
    matches!(n, Node::Num(s) if s.contains('.'))
}

fn emit_args(
    name: &str,
    args: &[Node],
    out: &mut String,
    syms: &mut BTreeSet<String>,
) -> Result<(), Unsupported> {
    out.push_str(name);
    out.push('(');
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        emit(a, out, syms)?;
    }
    out.push(')');
    Ok(())
}

fn emit_call(
    name: &str,
    args: &[Node],
    out: &mut String,
    syms: &mut BTreeSet<String>,
) -> Result<(), Unsupported> {
    let k = args.len();
    // Same name, one argument.
    const DIRECT1: &[&str] = &[
        "sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth", "sech", "csch",
        "asin", "acos", "atan", "acot", "asinh", "acosh", "atanh", "exp", "sqrt", "erf", "erfc",
        "erfi", "abs", "floor",
    ];
    if k == 1 && DIRECT1.contains(&name) {
        return emit_args(name, args, out, syms);
    }
    // Reciprocal inverse functions: f(u) = g(1/u).
    let reciprocal = match name {
        "asec" => Some("acos"),
        "acsc" => Some("asin"),
        "acoth" => Some("atanh"),
        "asech" => Some("acosh"),
        "acsch" => Some("asinh"),
        _ => None,
    };
    if let (Some(g), 1) = (reciprocal, k) {
        out.push_str(g);
        out.push_str("(1/");
        emit_wrapped(&args[0], prec(&args[0]) < 4, out, syms)?;
        out.push(')');
        return Ok(());
    }
    // Mathematica's two-argument `ArcTan[x, y]`, kept in that order by the
    // suite's converter, is `atan2(y, x)`.
    if name == "atan" && k == 2 {
        let swapped = [args[1].clone(), args[0].clone()];
        return emit_args("atan2", &swapped, out, syms);
    }
    let mapped: Option<&str> = match (name, k) {
        ("log", 1) => Some("ln"),
        ("Si" | "expintegral_si", 1) => Some("Si"),
        ("Ci" | "expintegral_ci", 1) => Some("Ci"),
        ("Shi" | "expintegral_shi", 1) => Some("Shi"),
        ("Chi" | "expintegral_chi", 1) => Some("Chi"),
        ("Ei" | "expintegral_ei", 1) => Some("Ei"),
        ("Li" | "expintegral_li", 1) => Some("li"),
        ("expintegral_e1", 1) => Some("E1"),
        // Mathematica's `ExpIntegralE[n, z]` = E_n(z), written `Ei(n, z)`
        // by the suite's converter (∫E₁(bx)dx = −E₂(bx)/b in the suite).
        ("expintegral_e" | "Ei", 2) => Some("expint"),
        ("FresnelS" | "fresnel_s", 1) => Some("fresnels"),
        ("FresnelC" | "fresnel_c", 1) => Some("fresnelc"),
        ("ProductLog" | "lambert_w", 1) => Some("lambertw"),
        ("GAMMA" | "gamma", 1) => Some("gamma"),
        ("GAMMA" | "gamma_incomplete", 2) => Some("uppergamma"),
        ("lnGAMMA" | "log_gamma", 1) => Some("loggamma"),
        ("Psi" | "psi", 1) => Some("digamma"),
        ("Psi", 2) => Some("polygamma"),
        ("Zeta" | "zeta", 1) => Some("zeta"),
        ("polylog" | "PolyLog", 2) => Some("polylog"),
        // Parameter convention m = k² in Maxima, Mathematica and symplex.
        ("elliptic_f" | "EllipticF", 2) => Some("elliptic_f"),
        ("elliptic_kc" | "EllipticK", 1) => Some("elliptic_k"),
        ("elliptic_ec" | "EllipticE", 1) => Some("elliptic_e"),
        ("EllipticPi", 2) => Some("elliptic_pi"),
        ("atan2", 2) => Some("atan2"),
        ("Factorial" | "factorial", 1) => Some("factorial"),
        ("signum", 1) => Some("sign"),
        ("ceiling", 1) => Some("ceiling"),
        _ => None,
    };
    if let Some(m) = mapped {
        return emit_args(m, args, out, syms);
    }
    let reason = match name {
        "Unintegrable" | "CannotIntegrate" | "int" | "Int" | "integrate" | "nintegrable" => {
            "unevaluated integral".to_string()
        }
        "Derivative" => "formal Derivative".to_string(),
        "hypergeometric" | "HypergeometricPFQ" | "Hypergeometric2F1" | "AppellF1"
        | "HurwitzLerchPhi" | "elliptic_e" | "elliptic_pi" | "EllipticE" | "EllipticF"
        | "EllipticPi" | "Zeta" | "zeta" | "ProductLog" | "GAMMA" | "Psi" => {
            format!("special function {name}/{k}")
        }
        _ => format!("unknown function {name}/{k}"),
    };
    Err(Unsupported::new(reason, String::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(s: &str) -> String {
        translate(s).unwrap().text
    }

    #[test]
    fn constants_and_renaming() {
        assert_eq!(tr("%e^x+%pi*%i"), "E^x + pi*I");
        assert_eq!(tr("e*x+i"), "e_*x + i_");
        assert_eq!(original_name("e_"), "e");
        assert_eq!(original_name("x_"), "x_");
    }

    #[test]
    fn precedence_is_preserved() {
        assert_eq!(tr("-x^2"), "-x^2");
        assert_eq!(tr("a-(b+c)"), "a - (b + c)");
        assert_eq!(tr("a/(b*c)"), "a/(b*c)");
        assert_eq!(tr("a^-2*b"), "a^(-2)*b");
        assert_eq!(tr("(a^b)^c"), "(a^b)^c");
        assert_eq!(tr("a^b^c"), "a^(b^c)");
        assert_eq!(tr("(-1)/x"), "-1/x");
        assert_eq!(tr("a/(-b)"), "a/(-b)");
        assert_eq!(tr("a*(b*c)"), "a*(b*c)");
        assert_eq!(tr("2^-x^2"), "2^(-x^2)");
    }

    #[test]
    fn functions() {
        assert_eq!(tr("log(x)"), "ln(x)");
        assert_eq!(tr("asec(a+b*x)"), "acos(1/(a + b*x))");
        assert_eq!(tr("acoth(x)"), "atanh(1/x)");
        assert_eq!(tr("GAMMA(a,x)"), "uppergamma(a, x)");
        assert_eq!(tr("Li(c*x)"), "li(c*x)");
        assert_eq!(tr("atan(a,b)"), "atan2(b, a)");
        assert_eq!(tr("Ei(1,b*x)"), "expint(1, b*x)");
        assert_eq!(
            translate("Derivative(1)(f)(x)").unwrap_err().reason,
            "formal Derivative(..)(..) application"
        );
        assert!(translate("f(x)").is_err());
        assert_eq!(
            translate("hypergeometric([1],[2],x)").unwrap_err().reason,
            "special function hypergeometric/3"
        );
    }

    #[test]
    fn numbers() {
        assert_eq!(tr("1.5*x"), "1.5*x");
        assert_eq!(tr("2.0e-3"), "(2.0*10^(-3))");
        assert_eq!(tr("x^0.5"), "x^(0.5)");
    }
}
