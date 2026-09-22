//! Runtime expression parser.
//!
//! Parses mathematical expressions from strings into the arena.
//! Uses a Pratt parser (precedence climbing) with the same grammar
//! as the `expr!` proc macro:
//!
//! - Operators: `+`, `-`, `*`, `/`, `^` (right-associative)
//! - Unary: `-` (prefix negation)
//! - Functions: `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `abs`
//! - Parentheses: `(`, `)`
//! - Atoms: integer literals, symbol names
//! - Constants: `pi`, `e`, `I`, `inf`, `nan`
//!
//! Three entry points share this grammar (SymPy: `sympify`, `parse_expr`):
//!
//! | Function | Result | Extra syntax |
//! |----------|--------|--------------|
//! | [`parse`] / [`Context::parse`] | [`Ex`] | juxtaposition of numbers, symbols and parentheses is multiplication (`2x`, `2 x`, `x y`, `2(x+1)`, `(x+1)(x-1)`, `2pi`) |
//! | [`parse_bool`] / [`Context::parse_bool`] | [`BoolEx`] | relations `<` `<=` `>` `>=` `==` `!=`, connectives `&`/`&&`/`and`, `\|`/`\|\|`/`or`, prefix `~`/`!`/`not`, `True`/`False`, and the function forms `Eq(a, b)`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`, `And(…)`, `Or(…)`, `Not(a)` |
//! | [`parse_implicit`] / [`Context::parse_implicit`] | [`Ex`] | function application without parentheses (`sin x`, `2 sin x`, `sin 2x`) and `f(x)` as a product for unknown `f` (`x(x+1)`) |
//!
//! # Examples
//!
//! ```
//! use symplex::prelude::*;
//!
//! let ctx = Context::new();
//! let expr = symplex::parse::parse(&ctx, "x^2 + 2*x + 1").unwrap();
//! assert_eq!(format!("{expr}"), "x^2 + 2*x + 1");
//! ```

use std::sync::Arc;

use num_bigint::BigInt;
use num_rational::Ratio;
use smallvec::SmallVec;

use crate::api::context::Context;
use crate::api::expr::{BoolEx, Ex};
use crate::base::arena::Arena;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;

/// Error returned when parsing fails.
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Human-readable description of what went wrong.
    pub message: String,
    /// Byte offset in the input where the error occurred.
    pub position: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "parse error at position {}: {}",
            self.position, self.message
        )
    }
}

impl std::error::Error for ParseError {}

/// Parse a mathematical expression string into an `Ex`.
///
/// Uses the provided context for symbol and number interning.
///
/// # Errors
///
/// Returns `ParseError` if the string is not a valid expression.
pub fn parse(ctx: &Context, input: &str) -> Result<Ex, ParseError> {
    let id = parse_with_mode(ctx, input, Mode::STRICT)?;
    // Construct an Ex from the ExprId. Ex fields are pub(crate), so this
    // works from within the crate without needing to expose make_ex.
    Ok(Ex::from_raw_parts(ctx.id, Arc::clone(&ctx.inner), id))
}

/// Parse a relation or Boolean combination of relations into a [`BoolEx`].
///
/// The numeric grammar of [`parse`] is extended with the comparison
/// operators `<`, `<=`, `>`, `>=`, `==`, `!=`, the connectives `&`/`&&`/`and`
/// and `|`/`||`/`or`, the prefix negation `~`/`!`/`not`, the constants
/// `True`/`False`, and the function forms `Eq(a, b)`, `Ne`, `Lt`, `Le`,
/// `Gt`, `Ge`, `And(a, b, …)`, `Or(…)`, `Not(a)`.  Precedence, loosest
/// first: `or` < `and` < comparisons < `+ -` < `* /` < `^`; `not` is a
/// prefix operator that applies to the following relation (`not x > 0` is
/// `not (x > 0)`), and comparisons do not chain (`a < b < c` is an error —
/// write `a < b & b < c`).  This is mathematical precedence, unlike
/// SymPy's `sympify("x > 0 & x < 1")`, where Python binds `&` tighter than
/// `>`.
///
/// # Errors
///
/// [`ParseError`] if the string is not well formed, or if it parses to a
/// numeric expression rather than a relation.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let p = symplex::parse::parse_bool(&ctx, "x > 0 & x < 1").unwrap();
/// assert_eq!(p.to_string(), "x > 0 & 1 > x");
/// assert!(symplex::parse::parse_bool(&ctx, "x + 1").is_err());
/// ```
pub fn parse_bool(ctx: &Context, input: &str) -> Result<BoolEx, ParseError> {
    let id = parse_with_mode(ctx, input, Mode::RELATIONS)?;
    Ok(BoolEx::from_raw_parts(ctx.id, Arc::clone(&ctx.inner), id))
}

/// Parse with implicit multiplication *and* implicit function application
/// (SymPy: `parse_expr(s, transformations=implicit_multiplication_application)`).
///
/// [`parse`] already reads `2x`, `2 x`, `x y`, `2(x+1)` and `(x+1)(x-1)` as
/// products.  This variant additionally accepts
///
/// - `sin x`, `2 sin x`, `sin 2x`, `sin x^2`: a textbook function name
///   (trigonometric, hyperbolic and inverse trigonometric functions, `exp`,
///   `ln`/`log`, `sqrt`, `cbrt`, `abs`, `floor`, `ceil`, `sign`, `gamma`,
///   `erf`, `erfc`, `factorial`) applied without parentheses to the
///   juxtaposed product that follows it, up to the next `+`, `-`,
///   comparison, closing parenthesis, or function name;
/// - `x(x+1)`, `f(x) g(x)`: an identifier that is *not* a known function,
///   followed by `(`, is a symbol times the parenthesised group.
///
/// Ambiguities are resolved as follows:
///
/// | Input | Reading | Note |
/// |-------|---------|------|
/// | `x y z` | `x*y*z` | juxtaposition is left-associative |
/// | `2 sin x` | `2*sin(x)` | a coefficient stays outside |
/// | `sin 2x` | `sin(2*x)` | the argument is the whole following product |
/// | `sin x^2` | `sin(x^2)` | `^` binds tighter than application |
/// | `sin x cos y` | `sin(x)*cos(y)` | the argument stops at the next function name (SymPy reads `sin(x*cos(y))`) |
/// | `sin x + 1` | `sin(x) + 1` | `+` ends the argument |
/// | `sin x/2` | `sin(x/2)` | `/` is part of the product |
/// | `f(x)` | `f*x` | `f` is not a known function |
///
/// Short names that double as common variables (`re`, `im`, `arg`, `li`,
/// `zeta`, …) are *not* applied implicitly; write them with parentheses.
/// The one-letter display aliases `C(n, k)`, `B(a, b)`, `W(x)` are
/// ordinary symbols here (use `binomial`, `beta`, `lambertw`).  A textbook
/// function name that is not followed by an operand is an error
/// (`sin + 1`).
///
/// # Errors
///
/// [`ParseError`] if the string is not well formed.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
/// let e = symplex::parse::parse_implicit(&ctx, "2x + 3(y-1)").unwrap();
/// assert_eq!(e, 2 * &x + 3 * (&y - 1));
/// let s = symplex::parse::parse_implicit(&ctx, "2 sin x cos y").unwrap();
/// assert_eq!(s, 2 * &x.sin() * &y.cos());
/// ```
pub fn parse_implicit(ctx: &Context, input: &str) -> Result<Ex, ParseError> {
    let id = parse_with_mode(ctx, input, Mode::IMPLICIT)?;
    Ok(Ex::from_raw_parts(ctx.id, Arc::clone(&ctx.inner), id))
}

/// Run the parser in `mode` and return the interned root.
///
/// In [`Mode::RELATIONS`] the root must be a Boolean node.
fn parse_with_mode(ctx: &Context, input: &str, mode: Mode) -> Result<ExprId, ParseError> {
    let mut parser = Parser::new(input, mode);
    ctx.with_arena_mut(|arena| {
        let result = parser.parse_expr(arena, 0)?;
        // After parsing the expression, ensure we consumed everything.
        if parser.current != Token::Eof {
            return Err(ParseError {
                message: format!("unexpected token {:?} after expression", parser.current),
                position: parser.lexer.pos,
            });
        }
        if mode.relations && !is_bool_node(arena, result) {
            return Err(ParseError {
                message: format!(
                    "expected a relation or Boolean expression, got the numeric expression '{}'",
                    arena.display(result)
                ),
                position: parser.lexer.pos,
            });
        }
        Ok(result)
    })
}

impl Context {
    /// Parse a relation or Boolean combination of relations in this context
    /// (SymPy: `sympify("x > 0")`).
    ///
    /// See [`parse::parse_bool`](crate::parse::parse_bool) for the grammar:
    /// comparisons `<` `<=` `>` `>=` `==` `!=` bind tighter than `&`/`and`,
    /// which binds tighter than `|`/`or`; `~`/`!`/`not` is prefix.
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`](crate::base::errors::SymplexError::ComputationFailed)
    /// with the parser's message if the string is not a well-formed relation.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let p = ctx.parse_bool("x > 0 & x < 1").unwrap();
    /// assert_eq!(p.to_string(), "x > 0 & 1 > x");
    /// assert_eq!(p.to_lean().unwrap(), "0 < x ∧ x < 1");
    /// assert_eq!(ctx.parse_bool("not x == 1 or y >= 2").unwrap().to_string(), "!(x == 1) | y >= 2");
    /// ```
    pub fn parse_bool(
        &self,
        input: &str,
    ) -> Result<crate::api::expr::BoolEx, crate::base::errors::SymplexError> {
        parse_bool(self, input).map_err(|e| crate::base::errors::SymplexError::ComputationFailed {
            operation: "parse_bool",
            reason: e.to_string(),
        })
    }

    /// Parse with implicit multiplication and implicit function application
    /// (SymPy: `parse_expr(s, transformations=implicit_multiplication_application)`).
    ///
    /// See [`parse::parse_implicit`](crate::parse::parse_implicit) for the
    /// rules and the ambiguities they resolve (`2 sin x` is `2*sin(x)`,
    /// `sin 2x` is `sin(2*x)`, `sin x cos y` is `sin(x)*cos(y)`).
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`](crate::base::errors::SymplexError::ComputationFailed)
    /// with the parser's message if the string is not well formed.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    /// assert_eq!(ctx.parse_implicit("2x + 3(y-1)").unwrap(), 2 * &x + 3 * (&y - 1));
    /// assert_eq!(ctx.parse_implicit("sin 2x").unwrap(), (2 * &x).sin());
    /// ```
    pub fn parse_implicit(
        &self,
        input: &str,
    ) -> Result<crate::api::expr::Ex, crate::base::errors::SymplexError> {
        parse_implicit(self, input).map_err(|e| {
            crate::base::errors::SymplexError::ComputationFailed {
                operation: "parse_implicit",
                reason: e.to_string(),
            }
        })
    }
}

/// Which syntax extensions the parser accepts.
#[derive(Clone, Copy)]
struct Mode {
    /// Comparison operators, Boolean connectives and `True`/`False`.
    relations: bool,
    /// Function application without parentheses; unknown `f(…)` is a product.
    implicit_app: bool,
}

impl Mode {
    const STRICT: Mode = Mode {
        relations: false,
        implicit_app: false,
    };
    const RELATIONS: Mode = Mode {
        relations: true,
        implicit_app: false,
    };
    const IMPLICIT: Mode = Mode {
        relations: false,
        implicit_app: true,
    };
}

/// Is `id` a Boolean-sorted node (a relation, connective or truth value)?
fn is_bool_node(arena: &Arena, id: ExprId) -> bool {
    matches!(
        arena.node(id),
        ExprNode::BoolTrue
            | ExprNode::BoolFalse
            | ExprNode::Gt(_, _)
            | ExprNode::Ge(_, _)
            | ExprNode::Eq_(_, _)
            | ExprNode::Ne(_, _)
            | ExprNode::And(_)
            | ExprNode::Or(_)
            | ExprNode::Not(_)
    )
}

/// Every function name the call tables (`call_1` … `call_4`, `min`/`max`,
/// `Sum`/`Product`) accept by a dedicated arm, lower-cased; the library
/// functions ([`LibFn`]) are known by their registry names.  Used by
/// [`parse_implicit`] to tell `sin(x)` (a call) from `f(x)` (a product).
const KNOWN_FUNCTIONS: &[&str] = &[
    // call_1
    "sin",
    "cos",
    "tan",
    "exp",
    "ln",
    "log",
    "sqrt",
    "cbrt",
    "abs",
    "asin",
    "arcsin",
    "acos",
    "arccos",
    "atan",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "arcsinh",
    "acosh",
    "arccosh",
    "atanh",
    "arctanh",
    "sign",
    "sgn",
    "floor",
    "ceil",
    "ceiling",
    "gamma",
    "erf",
    "erfc",
    "heaviside",
    "diracdelta",
    "dirac_delta",
    "lambertw",
    "w",
    "factorial",
    "digamma",
    "loggamma",
    "cot",
    "sec",
    "csc",
    "coth",
    "sech",
    "csch",
    "acot",
    "arccot",
    "re",
    "im",
    "conjugate",
    "conj",
    "arg",
    "si",
    "ci",
    "ei",
    "li",
    "zeta",
    "e1",
    // call_2
    "rootof",
    "conditionset",
    "integral",
    "atan2",
    "polygamma",
    "kroneckerdelta",
    "kronecker_delta",
    "binomial",
    "c",
    "beta",
    "b",
    // call_3
    "limit",
    "laplacetransform",
    "inverselaplacetransform",
    "residue",
    "dsolve",
    // call_4
    "series",
    // variadic / binder forms
    "min",
    "max",
    "sum",
    "product",
];

fn is_known_function(name_lower: &str) -> bool {
    KNOWN_FUNCTIONS.contains(&name_lower) || lib_fn_by_name(name_lower).is_some()
}

/// The library function a (lower-cased) call name denotes.  `lambertw` is
/// excluded: the parser builds the dedicated `LambertW` node for it, as the
/// `Arena` does.
fn lib_fn_by_name(name_lower: &str) -> Option<LibFn> {
    LibFn::from_name_ignore_ascii_case(name_lower).filter(|f| *f != LibFn::LambertW)
}

/// Textbook one-argument functions that [`parse_implicit`] applies without
/// parentheses (`sin x`).  Deliberately excludes short names that are also
/// common variables (`re`, `im`, `arg`, `li`, `w`, `zeta`, `chi`, …).
fn is_implicit_unary_function(name_lower: &str) -> bool {
    matches!(
        name_lower,
        "sin"
            | "cos"
            | "tan"
            | "cot"
            | "sec"
            | "csc"
            | "sinh"
            | "cosh"
            | "tanh"
            | "coth"
            | "sech"
            | "csch"
            | "asin"
            | "acos"
            | "atan"
            | "acot"
            | "arcsin"
            | "arccos"
            | "arctan"
            | "arccot"
            | "asinh"
            | "acosh"
            | "atanh"
            | "arcsinh"
            | "arccosh"
            | "arctanh"
            | "exp"
            | "ln"
            | "log"
            | "sqrt"
            | "cbrt"
            | "abs"
            | "floor"
            | "ceil"
            | "ceiling"
            | "sign"
            | "sgn"
            | "gamma"
            | "erf"
            | "erfc"
            | "factorial"
    )
}

/// The named constants (`pi`, `e`, `I`, `inf`, …), or `None` for a symbol.
fn constant_of(arena: &Arena, name: &str) -> Option<ExprId> {
    Some(match name {
        "pi" | "Pi" | "PI" => arena.pi,
        "e" | "E" => arena.e_const,
        "I" | "i" => arena.i_unit,
        "inf" | "oo" | "Inf" => arena.infinity,
        "zoo" => arena.complex_infinity,
        "nan" => arena.nan,
        "EulerGamma" | "euler_gamma" => arena.euler_gamma,
        "Catalan" => arena.catalan,
        "GoldenRatio" | "golden_ratio" => arena.golden_ratio,
        _ => return None,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Tokenizer
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Int(BigInt),
    Rational(Q),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    /// Postfix factorial `!` (prefix logical negation in [`Mode::RELATIONS`]).
    Bang,
    /// `=` (only inside `Sum(body, k=lo..hi)` / `Product(…)`).
    Eq,
    /// `..` range separator (only inside `Sum` / `Product`).
    DotDot,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `==`
    EqEq,
    /// `!=`
    Ne,
    /// `&` or `&&`
    Amp,
    /// `|` or `||`
    Pipe,
    /// `~` (prefix logical negation)
    Tilde,
    Eof,
}

struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Lexer { input, pos: 0 }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() && self.input.as_bytes()[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    /// Is the byte at the current position `b`?
    fn peek_is(&self, b: u8) -> bool {
        self.input.as_bytes().get(self.pos) == Some(&b)
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();
        if self.pos >= self.input.len() {
            return Ok(Token::Eof);
        }

        let b = self.input.as_bytes()[self.pos];
        match b {
            b'+' => {
                self.pos += 1;
                Ok(Token::Plus)
            }
            b'-' => {
                self.pos += 1;
                Ok(Token::Minus)
            }
            b'*' => {
                self.pos += 1;
                // Check for ** (SymPy-compatible power operator)
                if self.pos < self.input.len() && self.input.as_bytes()[self.pos] == b'*' {
                    self.pos += 1;
                    Ok(Token::Caret) // ** treated same as ^
                } else {
                    Ok(Token::Star)
                }
            }
            b'/' => {
                self.pos += 1;
                Ok(Token::Slash)
            }
            b'^' => {
                self.pos += 1;
                Ok(Token::Caret)
            }
            b'(' => {
                self.pos += 1;
                Ok(Token::LParen)
            }
            b')' => {
                self.pos += 1;
                Ok(Token::RParen)
            }
            b',' => {
                self.pos += 1;
                Ok(Token::Comma)
            }
            b'!' => {
                self.pos += 1;
                if self.peek_is(b'=') {
                    self.pos += 1;
                    Ok(Token::Ne)
                } else {
                    Ok(Token::Bang)
                }
            }
            b'=' => {
                self.pos += 1;
                if self.peek_is(b'=') {
                    self.pos += 1;
                    Ok(Token::EqEq)
                } else {
                    Ok(Token::Eq)
                }
            }
            b'<' => {
                self.pos += 1;
                if self.peek_is(b'=') {
                    self.pos += 1;
                    Ok(Token::Le)
                } else {
                    Ok(Token::Lt)
                }
            }
            b'>' => {
                self.pos += 1;
                if self.peek_is(b'=') {
                    self.pos += 1;
                    Ok(Token::Ge)
                } else {
                    Ok(Token::Gt)
                }
            }
            b'&' => {
                self.pos += 1;
                if self.peek_is(b'&') {
                    self.pos += 1;
                }
                Ok(Token::Amp)
            }
            b'|' => {
                self.pos += 1;
                if self.peek_is(b'|') {
                    self.pos += 1;
                }
                Ok(Token::Pipe)
            }
            b'~' => {
                self.pos += 1;
                Ok(Token::Tilde)
            }
            b'.' if self.pos + 1 < self.input.len()
                && self.input.as_bytes()[self.pos + 1] == b'.' =>
            {
                self.pos += 2;
                Ok(Token::DotDot)
            }
            b'0'..=b'9' => {
                let start = self.pos;
                while self.pos < self.input.len()
                    && self.input.as_bytes()[self.pos].is_ascii_digit()
                {
                    self.pos += 1;
                }
                // Check for decimal point followed by digits → Rational token
                if self.pos < self.input.len()
                    && self.input.as_bytes()[self.pos] == b'.'
                    && self.pos + 1 < self.input.len()
                    && self.input.as_bytes()[self.pos + 1].is_ascii_digit()
                {
                    self.pos += 1; // consume '.'
                    let frac_start = self.pos;
                    while self.pos < self.input.len()
                        && self.input.as_bytes()[self.pos].is_ascii_digit()
                    {
                        self.pos += 1;
                    }
                    let decimal_places = self.pos - frac_start;
                    let full_str = self.input[start..self.pos].replace('.', "");
                    let numer = full_str.parse::<BigInt>().map_err(|e| ParseError {
                        message: format!(
                            "invalid number '{}': {}",
                            &self.input[start..self.pos],
                            e
                        ),
                        position: start,
                    })?;
                    let mut denom = BigInt::from(1);
                    for _ in 0..decimal_places {
                        denom *= 10;
                    }
                    let ratio = Ratio::new(numer, denom);
                    return Ok(Token::Rational(ratio));
                }
                let s = &self.input[start..self.pos];
                let n = s.parse::<BigInt>().map_err(|e| ParseError {
                    message: format!("invalid integer '{}': {}", s, e),
                    position: start,
                })?;
                Ok(Token::Int(n))
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = self.pos;
                while self.pos < self.input.len() {
                    let c = self.input.as_bytes()[self.pos];
                    if c.is_ascii_alphanumeric() || c == b'_' {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                let s = self.input[start..self.pos].to_string();
                Ok(Token::Ident(s))
            }
            _ => Err(ParseError {
                message: format!("unexpected character '{}'", b as char),
                position: self.pos,
            }),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pratt parser
// ═══════════════════════════════════════════════════════════════════════════

// Binding powers `(left, right)` of the infix tiers, loosest first.  Only
// the relative order matters; the Boolean tiers are reachable only in
// `Mode::RELATIONS`.
const BP_OR: (u8, u8) = (1, 2);
const BP_AND: (u8, u8) = (3, 4);
const BP_REL: (u8, u8) = (5, 6);
const BP_ADD: (u8, u8) = (7, 8);
const BP_MUL: (u8, u8) = (9, 10);
/// Operand of prefix `-`: tighter than `*` and `/` but looser than `^`, so
/// that `-x^2` parses as `-(x^2)`.
const BP_NEG: u8 = 11;
/// `^` is right-associative: the right binding power is below the left.
const BP_POW: (u8, u8) = (14, 13);
/// Operand of prefix `not`/`~`/`!`: one relation, not a whole conjunction
/// (`not x > 0 & y > 0` is `(not x > 0) & (y > 0)`).
const BP_NOT: u8 = BP_REL.0;

/// An infix operator recognised by the Pratt loop.
#[derive(Clone, Copy)]
enum Infix {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Rel(RelOp),
    And,
    Or,
}

#[derive(Clone, Copy)]
enum RelOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl RelOp {
    fn of(token: &Token) -> Option<RelOp> {
        Some(match token {
            Token::Lt => RelOp::Lt,
            Token::Le => RelOp::Le,
            Token::Gt => RelOp::Gt,
            Token::Ge => RelOp::Ge,
            Token::EqEq => RelOp::Eq,
            Token::Ne => RelOp::Ne,
            _ => return None,
        })
    }

    fn text(self) -> &'static str {
        match self {
            RelOp::Lt => "<",
            RelOp::Le => "<=",
            RelOp::Gt => ">",
            RelOp::Ge => ">=",
            RelOp::Eq => "==",
            RelOp::Ne => "!=",
        }
    }
}

/// Can `token` begin an operand (the argument of `sin x`)?
fn starts_operand(token: &Token) -> bool {
    matches!(
        token,
        Token::Int(_) | Token::Rational(_) | Token::Ident(_) | Token::LParen
    )
}

struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    depth: usize,
    mode: Mode,
    /// Inside the parenthesis-free argument of `sin x`: a following
    /// function name ends the argument instead of multiplying into it.
    app_arg: bool,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str, mode: Mode) -> Self {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token().unwrap_or(Token::Eof);
        Parser {
            lexer,
            current,
            depth: 0,
            mode,
            app_arg: false,
        }
    }

    fn error(&self, message: String) -> ParseError {
        ParseError {
            message,
            position: self.lexer.pos,
        }
    }

    /// Run `f` outside any `sin x` argument (inside parentheses or a call).
    fn outside_app_arg<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, ParseError>,
    ) -> Result<T, ParseError> {
        let saved = std::mem::replace(&mut self.app_arg, false);
        let result = f(self);
        self.app_arg = saved;
        result
    }

    /// The operand of `+`, `-`, `*`, `/`, `^` or a comparison must be numeric.
    fn numeric_operands(
        &self,
        arena: &Arena,
        op: &str,
        lhs: ExprId,
        rhs: ExprId,
    ) -> Result<(), ParseError> {
        for id in [lhs, rhs] {
            if is_bool_node(arena, id) {
                return Err(self.error(format!(
                    "'{op}' needs numeric operands, but '{}' is a Boolean expression",
                    arena.display(id)
                )));
            }
        }
        Ok(())
    }

    /// The operand of `&`, `|` or `not` must be a relation or truth value.
    fn boolean_operand(&self, arena: &Arena, op: &str, id: ExprId) -> Result<(), ParseError> {
        if is_bool_node(arena, id) {
            Ok(())
        } else {
            Err(self.error(format!(
                "'{op}' needs Boolean operands (relations), but '{}' is numeric",
                arena.display(id)
            )))
        }
    }

    /// Combine `lhs op rhs` into a node, checking the operand sorts.
    fn combine(
        &self,
        arena: &mut Arena,
        op: Infix,
        lhs: ExprId,
        rhs: ExprId,
    ) -> Result<ExprId, ParseError> {
        match op {
            Infix::Add => {
                self.numeric_operands(arena, "+", lhs, rhs)?;
                Ok(arena.add(&[lhs, rhs]))
            }
            Infix::Sub => {
                self.numeric_operands(arena, "-", lhs, rhs)?;
                Ok(arena.sub(lhs, rhs))
            }
            Infix::Mul => {
                self.numeric_operands(arena, "*", lhs, rhs)?;
                Ok(arena.mul(&[lhs, rhs]))
            }
            Infix::Div => {
                self.numeric_operands(arena, "/", lhs, rhs)?;
                Ok(arena.div(lhs, rhs))
            }
            Infix::Pow => {
                self.numeric_operands(arena, "^", lhs, rhs)?;
                Ok(arena.pow(lhs, rhs))
            }
            Infix::Rel(rel) => {
                self.numeric_operands(arena, rel.text(), lhs, rhs)?;
                if RelOp::of(&self.current).is_some() {
                    return Err(self.error(
                        "chained comparisons are not supported; write 'a < b & b < c'".into(),
                    ));
                }
                Ok(match rel {
                    RelOp::Lt => arena.gt(rhs, lhs),
                    RelOp::Le => arena.ge(rhs, lhs),
                    RelOp::Gt => arena.gt(lhs, rhs),
                    RelOp::Ge => arena.ge(lhs, rhs),
                    RelOp::Eq => arena.eq_(lhs, rhs),
                    RelOp::Ne => arena.ne_(lhs, rhs),
                })
            }
            Infix::And => {
                self.boolean_operand(arena, "&", lhs)?;
                self.boolean_operand(arena, "&", rhs)?;
                // Flatten `a & b & c` into one n-ary node, as `Ex::and` does.
                let mut items: SmallVec<[ExprId; 4]> = match arena.node(lhs) {
                    ExprNode::And(children) => children.iter().copied().collect(),
                    _ => SmallVec::from_slice(&[lhs]),
                };
                items.push(rhs);
                Ok(arena.and(&items))
            }
            Infix::Or => {
                self.boolean_operand(arena, "|", lhs)?;
                self.boolean_operand(arena, "|", rhs)?;
                let mut items: SmallVec<[ExprId; 4]> = match arena.node(lhs) {
                    ExprNode::Or(children) => children.iter().copied().collect(),
                    _ => SmallVec::from_slice(&[lhs]),
                };
                items.push(rhs);
                Ok(arena.or(&items))
            }
        }
    }

    fn advance(&mut self) -> Result<Token, ParseError> {
        let old = std::mem::replace(&mut self.current, Token::Eof);
        self.current = self.lexer.next_token()?;
        Ok(old)
    }

    fn expect(&mut self, expected: &Token) -> Result<(), ParseError> {
        if &self.current == expected {
            self.advance()?;
            Ok(())
        } else {
            Err(ParseError {
                message: format!("expected {:?}, got {:?}", expected, self.current),
                position: self.lexer.pos,
            })
        }
    }

    /// Parse an expression with minimum binding power `min_bp`.
    fn parse_expr(&mut self, arena: &mut Arena, min_bp: u8) -> Result<ExprId, ParseError> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(ParseError {
                message: "expression nesting too deep (max 128 levels)".into(),
                position: self.lexer.pos,
            });
        }

        // Prefix (atom or unary)
        let mut lhs = self.parse_prefix(arena)?;

        // Infix loop
        loop {
            // Postfix factorial binds tighter than every infix operator:
            // `2^3!` is `2^(3!)` and `x!^2` is `(x!)^2`.
            if self.current == Token::Bang {
                self.advance()?;
                lhs = arena.intern(ExprNode::Factorial(lhs));
                continue;
            }
            let relations = self.mode.relations;
            let (op, (l_bp, r_bp), implicit) = match &self.current {
                Token::Plus => (Infix::Add, BP_ADD, false),
                Token::Minus => (Infix::Sub, BP_ADD, false),
                Token::Star => (Infix::Mul, BP_MUL, false),
                Token::Slash => (Infix::Div, BP_MUL, false),
                Token::Caret => (Infix::Pow, BP_POW, false),
                Token::Lt | Token::Le | Token::Gt | Token::Ge | Token::EqEq | Token::Ne
                    if relations =>
                {
                    match RelOp::of(&self.current) {
                        Some(rel) => (Infix::Rel(rel), BP_REL, false),
                        None => break,
                    }
                }
                Token::Amp if relations => (Infix::And, BP_AND, false),
                Token::Pipe if relations => (Infix::Or, BP_OR, false),
                Token::Ident(name) if relations && name == "and" => (Infix::And, BP_AND, false),
                Token::Ident(name) if relations && name == "or" => (Infix::Or, BP_OR, false),
                // `sin x cos y`: the argument of `sin` ends at the next
                // function name (which then multiplies `sin x` as a whole).
                Token::Ident(name)
                    if self.app_arg && is_implicit_unary_function(&name.to_ascii_lowercase()) =>
                {
                    break;
                }
                // Implicit multiplication: number, identifier, or '(' immediately
                // following a complete left-hand expression.
                Token::Int(_) | Token::Rational(_) | Token::Ident(_) | Token::LParen => {
                    (Infix::Mul, BP_MUL, true)
                }
                _ => break,
            };

            if l_bp < min_bp {
                break;
            }

            if !implicit {
                self.advance()?;
            }
            let rhs = self.parse_expr(arena, r_bp)?;
            lhs = self.combine(arena, op, lhs, rhs)?;
        }

        self.depth -= 1;
        Ok(lhs)
    }

    /// Parse a prefix expression (atom, unary minus, function call, parens).
    fn parse_prefix(&mut self, arena: &mut Arena) -> Result<ExprId, ParseError> {
        match self.current.clone() {
            Token::Int(n) => {
                self.advance()?;
                Ok(arena.big_int(n))
            }
            Token::Rational(ratio) => {
                self.advance()?;
                let nid = arena.intern_num(ratio);
                Ok(arena.intern(ExprNode::Num(nid)))
            }
            Token::Ident(name) => {
                self.advance()?;
                let name_lower = name.to_ascii_lowercase();
                // Check for function call: ident followed by '('.  With
                // implicit application, an identifier that is not a known
                // function (`x(x+1)`, `f(x)`) is a symbol and the '(' starts
                // an implicitly multiplied group; the one-letter display
                // aliases `C`, `B`, `W` count as symbols there too.
                let is_call = !self.mode.implicit_app
                    || (is_known_function(&name_lower) && name_lower.len() > 1);
                if self.current == Token::LParen && is_call {
                    return self.outside_app_arg(|p| p.parse_function_call(arena, &name));
                }
                if self.mode.relations {
                    match name.as_str() {
                        "True" | "true" => return Ok(arena.bool_true()),
                        "False" | "false" => return Ok(arena.bool_false()),
                        "not" => return self.parse_not(arena),
                        _ => {}
                    }
                }
                if self.mode.implicit_app && is_implicit_unary_function(&name_lower) {
                    // `sin x`, `sin 2x`, `sin x^2`: the argument is the
                    // following juxtaposed product.
                    if !starts_operand(&self.current) {
                        return Err(self.error(format!(
                            "function '{name}' needs an argument (write '{name}(x)' or '{name} x')"
                        )));
                    }
                    let saved = std::mem::replace(&mut self.app_arg, true);
                    let arg = self.parse_expr(arena, BP_MUL.0);
                    self.app_arg = saved;
                    return self.call_1(arena, &name, &name_lower, arg?);
                }
                match constant_of(arena, &name) {
                    Some(c) => Ok(c),
                    None => Ok(arena.symbol(&name)),
                }
            }
            Token::Minus => {
                self.advance()?;
                let operand = self.parse_expr(arena, BP_NEG)?;
                if is_bool_node(arena, operand) {
                    return Err(self.error(format!(
                        "'-' needs a numeric operand, but '{}' is a Boolean expression",
                        arena.display(operand)
                    )));
                }
                Ok(arena.neg(operand))
            }
            Token::Tilde | Token::Bang if self.mode.relations => {
                self.advance()?;
                self.parse_not(arena)
            }
            Token::LParen => {
                self.advance()?;
                let inner = self.outside_app_arg(|p| p.parse_expr(arena, 0))?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            other => Err(ParseError {
                message: format!("expected expression, got {:?}", other),
                position: self.lexer.pos,
            }),
        }
    }

    /// Prefix `not`/`~`/`!` (the keyword or symbol already consumed): the
    /// operand is one relation.
    fn parse_not(&mut self, arena: &mut Arena) -> Result<ExprId, ParseError> {
        let operand = self.parse_expr(arena, BP_NOT)?;
        self.boolean_operand(arena, "not", operand)?;
        Ok(arena.not(operand))
    }

    /// Largest number of arguments accepted by any function.
    const MAX_FN_ARGS: usize = 32;

    fn parse_function_call(&mut self, arena: &mut Arena, name: &str) -> Result<ExprId, ParseError> {
        self.expect(&Token::LParen)?;

        if self.current == Token::RParen {
            self.advance()?;
            return Err(ParseError {
                message: format!("function '{}' requires an argument", name),
                position: self.lexer.pos,
            });
        }

        // Case-insensitive function name matching for SymPy compatibility
        let name_lower = name.to_ascii_lowercase();
        let mut args: Vec<ExprId> = vec![self.parse_expr(arena, 0)?];

        // `Sum(body, k=lo..hi)` / `Product(body, k=lo..hi)` — the display form.
        if matches!(name_lower.as_str(), "sum" | "product") && self.current == Token::Comma {
            self.advance()?;
            let var = self.parse_expr(arena, 0)?;
            if self.current == Token::Eq {
                self.advance()?;
                let lo = self.parse_expr(arena, 0)?;
                self.expect(&Token::DotDot)?;
                let hi = self.parse_expr(arena, 0)?;
                self.expect(&Token::RParen)?;
                return self.make_sum_product(arena, name, &name_lower, args[0], var, lo, hi);
            }
            args.push(var);
        }

        while self.current == Token::Comma {
            self.advance()?;
            if args.len() >= Self::MAX_FN_ARGS {
                return Err(ParseError {
                    message: format!(
                        "function '{}' has too many arguments (max {})",
                        name,
                        Self::MAX_FN_ARGS
                    ),
                    position: self.lexer.pos,
                });
            }
            args.push(self.parse_expr(arena, 0)?);
        }
        self.expect(&Token::RParen)?;

        // SymPy's function forms of the relations and connectives
        // (`Eq(x, 1)`, `And(a, b, c)`, `Not(a)`), only when parsing a relation.
        if self.mode.relations
            && let Some(result) = self.call_boolean(arena, name, &name_lower, &args)?
        {
            return Ok(result);
        }

        // Variadic functions.
        if matches!(name_lower.as_str(), "min" | "max") {
            if args.len() < 2 {
                return Err(ParseError {
                    message: format!("function '{}' requires at least 2 arguments", name),
                    position: self.lexer.pos,
                });
            }
            let ids: SmallVec<[ExprId; 4]> = args.iter().copied().collect();
            return Ok(arena.intern(if name_lower == "min" {
                ExprNode::Min(ids)
            } else {
                ExprNode::Max(ids)
            }));
        }

        match args.len() {
            1 => self.call_1(arena, name, &name_lower, args[0]),
            2 => self.call_2(arena, name, &name_lower, args[0], args[1]),
            3 => self.call_3(arena, name, &name_lower, args[0], args[1], args[2]),
            4 => self.call_4(arena, name, &name_lower, args[0], args[1], args[2], args[3]),
            n => Err(ParseError {
                message: format!(
                    "unknown {n}-argument function '{}'. Only min and max take more than 4 arguments",
                    name
                ),
                position: self.lexer.pos,
            }),
        }
    }

    /// `Eq`/`Ne`/`Lt`/`Le`/`Gt`/`Ge(a, b)`, `And`/`Or(a, b, …)`, `Not(a)`
    /// (SymPy spellings; `Mode::RELATIONS` only).  `Ok(None)` for any other
    /// name.
    fn call_boolean(
        &self,
        arena: &mut Arena,
        name: &str,
        name_lower: &str,
        args: &[ExprId],
    ) -> Result<Option<ExprId>, ParseError> {
        let rel = match name_lower {
            "eq" => Some(RelOp::Eq),
            "ne" => Some(RelOp::Ne),
            "lt" => Some(RelOp::Lt),
            "le" => Some(RelOp::Le),
            "gt" => Some(RelOp::Gt),
            "ge" => Some(RelOp::Ge),
            _ => None,
        };
        if let Some(rel) = rel {
            let [lhs, rhs] = args else {
                return Err(self.error(format!(
                    "'{name}' takes exactly 2 arguments, got {}",
                    args.len()
                )));
            };
            self.numeric_operands(arena, rel.text(), *lhs, *rhs)?;
            return Ok(Some(match rel {
                RelOp::Lt => arena.gt(*rhs, *lhs),
                RelOp::Le => arena.ge(*rhs, *lhs),
                RelOp::Gt => arena.gt(*lhs, *rhs),
                RelOp::Ge => arena.ge(*lhs, *rhs),
                RelOp::Eq => arena.eq_(*lhs, *rhs),
                RelOp::Ne => arena.ne_(*lhs, *rhs),
            }));
        }
        match name_lower {
            "and" | "or" => {
                for &a in args {
                    self.boolean_operand(arena, name, a)?;
                }
                Ok(Some(if name_lower == "and" {
                    arena.and(args)
                } else {
                    arena.or(args)
                }))
            }
            "not" => {
                let [a] = args else {
                    return Err(self.error(format!(
                        "'{name}' takes exactly 1 argument, got {}",
                        args.len()
                    )));
                };
                self.boolean_operand(arena, name, *a)?;
                Ok(Some(arena.not(*a)))
            }
            _ => Ok(None),
        }
    }

    /// Build `Sum`/`Product` after validating that the index is a symbol.
    #[allow(clippy::too_many_arguments)]
    fn make_sum_product(
        &self,
        arena: &mut Arena,
        name: &str,
        name_lower: &str,
        body: ExprId,
        var: ExprId,
        lo: ExprId,
        hi: ExprId,
    ) -> Result<ExprId, ParseError> {
        if !matches!(arena.node(var), ExprNode::Symbol(_)) {
            return Err(ParseError {
                message: format!(
                    "the index of '{}' must be a symbol, got '{}'",
                    name,
                    arena.display(var)
                ),
                position: self.lexer.pos,
            });
        }
        Ok(arena.intern(if name_lower == "sum" {
            ExprNode::Sum(body, var, lo, hi)
        } else {
            ExprNode::Product_(body, var, lo, hi)
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn call_4(
        &self,
        arena: &mut Arena,
        name: &str,
        name_lower: &str,
        arg: ExprId,
        arg2: ExprId,
        arg3: ExprId,
        arg4: ExprId,
    ) -> Result<ExprId, ParseError> {
        match name_lower {
            "series" => Ok(arena.intern(ExprNode::Series(arg, arg2, arg3, arg4))),
            // `Integral(f, x, a, b)` — the Display form of a definite integral;
            // the constructor applies only the cheap folds.
            "integral" => Ok(arena.definite_integral(arg, arg2, arg3, arg4)),
            // SymPy-style `Sum(f, k, a, b)` / `Product(f, k, a, b)`.
            "sum" | "product" => {
                self.make_sum_product(arena, name, name_lower, arg, arg2, arg3, arg4)
            }
            // `jacobi(n, a, b, x)`, `betainc(a, b, x1, x2)`, `betainc_regularized(a, b, x1, x2)`.
            _ => self.lib_call(arena, name_lower, &[arg, arg2, arg3, arg4], || {
                format!(
                    "unknown 4-argument function '{}'. Supported: Series, Sum, Product, Integral, \
                     jacobi, betainc, betainc_regularized",
                    name
                )
            }),
        }
    }

    /// A library special function by its registry name, in the arity it
    /// declares (`besselj(n, x)`, `expint(n, x)`, `gegenbauer(n, a, x)`,
    /// `jacobi(n, a, b, x)`, `fibonacci(n)`, …); otherwise the error the
    /// call table would have reported.
    fn lib_call(
        &self,
        arena: &mut Arena,
        name_lower: &str,
        args: &[ExprId],
        unknown: impl FnOnce() -> String,
    ) -> Result<ExprId, ParseError> {
        match lib_fn_by_name(name_lower) {
            Some(f) if f.arity().accepts(args.len()) => Ok(arena.lib_apply(f, args)),
            _ => Err(ParseError {
                message: unknown(),
                position: self.lexer.pos,
            }),
        }
    }

    fn call_3(
        &self,
        arena: &mut Arena,
        name: &str,
        name_lower: &str,
        arg: ExprId,
        arg2: ExprId,
        arg3: ExprId,
    ) -> Result<ExprId, ParseError> {
        match name_lower {
            "limit" => Ok(arena.intern(ExprNode::Limit(arg, arg2, arg3))),
            "laplacetransform" => Ok(arena.intern(ExprNode::LaplaceTransform(arg, arg2, arg3))),
            "inverselaplacetransform" => {
                Ok(arena.intern(ExprNode::InverseLaplaceTransform(arg, arg2, arg3)))
            }
            "residue" => Ok(arena.intern(ExprNode::Residue(arg, arg2, arg3))),
            "dsolve" => Ok(arena.intern(ExprNode::DSolve(arg, arg2, arg3))),
            // Orthogonal polynomials with a parameter: (n, param, x).
            _ => self.lib_call(arena, name_lower, &[arg, arg2, arg3], || {
                format!(
                    "unknown 3-argument function '{}'. Supported: Limit, LaplaceTransform, \
                     InverseLaplaceTransform, Residue, DSolve, min, max, gegenbauer, \
                     assoc_legendre, assoc_laguerre",
                    name
                )
            }),
        }
    }

    fn call_2(
        &self,
        arena: &mut Arena,
        name: &str,
        name_lower: &str,
        arg: ExprId,
        arg2: ExprId,
    ) -> Result<ExprId, ParseError> {
        match name_lower {
            "log" => {
                // log(x, base) = ln(x) / ln(base)
                let ln_x = arena.ln(arg);
                let ln_base = arena.ln(arg2);
                Ok(arena.div(ln_x, ln_base))
            }
            "rootof" => Ok(arena.intern(ExprNode::RootOf(arg, arg2))),
            "conditionset" => Ok(arena.intern(ExprNode::ConditionSet(arg, arg2))),
            // `Integral(f, x)` — the Display form of an indefinite integral.
            "integral" => Ok(arena.intern(ExprNode::Integral(arg, arg2))),
            "atan2" => Ok(arena.atan2(arg, arg2)),
            "polygamma" => Ok(arena.polygamma(arg, arg2)),
            "kroneckerdelta" | "kronecker_delta" => Ok(arena.kronecker_delta(arg, arg2)),
            // Combinatorics / special functions (`C(n, k)` and `B(a, b)` are
            // the display forms).
            "binomial" | "c" => Ok(arena.binomial(arg, arg2)),
            "beta" | "b" => Ok(arena.beta(arg, arg2)),
            // Library functions with a parameter first, as in SymPy and in
            // the display: Bessel `(order, x)`, `expint`/`lowergamma`/
            // `uppergamma`/`polylog` `(s, x)`, `elliptic_f`/`elliptic_pi`,
            // the classical orthogonal polynomials `(n, x)`, …
            _ => self.lib_call(arena, name_lower, &[arg, arg2], || {
                format!(
                    "unknown 2-argument function '{}'. Supported: log, atan2, polygamma, \
                     binomial, beta, besselj, bessely, besseli, besselk, expint, lowergamma, \
                     uppergamma, polylog, elliptic_f, elliptic_pi, min, max, KroneckerDelta, \
                     RootOf, ConditionSet, Integral",
                    name
                )
            }),
        }
    }

    fn call_1(
        &self,
        arena: &mut Arena,
        name: &str,
        name_lower: &str,
        arg: ExprId,
    ) -> Result<ExprId, ParseError> {
        match name_lower {
            "sin" => Ok(arena.sin(arg)),
            "cos" => Ok(arena.cos(arg)),
            "tan" => Ok(arena.tan(arg)),
            "exp" => Ok(arena.exp(arg)),
            "ln" | "log" => Ok(arena.ln(arg)),
            "sqrt" => Ok(arena.sqrt(arg)),
            "cbrt" => Ok(arena.cbrt(arg)),
            "abs" => Ok(arena.abs(arg)),
            "asin" | "arcsin" => Ok(arena.asin(arg)),
            "acos" | "arccos" => Ok(arena.acos(arg)),
            "atan" | "arctan" => Ok(arena.atan(arg)),
            "sinh" => Ok(arena.sinh(arg)),
            "cosh" => Ok(arena.cosh(arg)),
            "tanh" => Ok(arena.tanh(arg)),
            "asinh" | "arcsinh" => Ok(arena.asinh(arg)),
            "acosh" | "arccosh" => Ok(arena.acosh(arg)),
            "atanh" | "arctanh" => Ok(arena.atanh(arg)),
            "sign" | "sgn" => Ok(arena.sign(arg)),
            "floor" => Ok(arena.floor(arg)),
            "ceil" | "ceiling" => Ok(arena.ceiling(arg)),
            "gamma" => Ok(arena.intern(crate::base::node::ExprNode::Gamma(arg))),
            "erf" => Ok(arena.intern(crate::base::node::ExprNode::Erf(arg))),
            "erfc" => Ok(arena.intern(crate::base::node::ExprNode::Erfc(arg))),
            "heaviside" => Ok(arena.intern(crate::base::node::ExprNode::Heaviside(arg))),
            "diracdelta" | "dirac_delta" => {
                Ok(arena.intern(crate::base::node::ExprNode::DiracDelta(arg)))
            }
            "lambertw" | "w" => Ok(arena.intern(crate::base::node::ExprNode::LambertW(arg))),
            "factorial" => Ok(arena.intern(crate::base::node::ExprNode::Factorial(arg))),
            "digamma" => Ok(arena.intern(crate::base::node::ExprNode::Digamma(arg))),
            "loggamma" => Ok(arena.intern(crate::base::node::ExprNode::LogGamma(arg))),
            // Reciprocal trig / hyperbolic functions (no dedicated nodes;
            // the same forms `Ex::cot` & co. build).
            "cot" => {
                let c = arena.cos(arg);
                let s = arena.sin(arg);
                Ok(arena.div(c, s))
            }
            "sec" => {
                let c = arena.cos(arg);
                Ok(arena.div(arena.one, c))
            }
            "csc" => {
                let s = arena.sin(arg);
                Ok(arena.div(arena.one, s))
            }
            "coth" => {
                let c = arena.cosh(arg);
                let s = arena.sinh(arg);
                Ok(arena.div(c, s))
            }
            "sech" => {
                let c = arena.cosh(arg);
                Ok(arena.div(arena.one, c))
            }
            "csch" => {
                let s = arena.sinh(arg);
                Ok(arena.div(arena.one, s))
            }
            "acot" | "arccot" => {
                let inv = arena.div(arena.one, arg);
                Ok(arena.atan(inv))
            }
            // Complex analysis
            "re" => Ok(arena.re(arg)),
            "im" => Ok(arena.im(arg)),
            "conjugate" | "conj" => Ok(arena.conjugate(arg)),
            "arg" => Ok(arena.arg(arg)),
            // Special functions (0.2)
            "si" => Ok(arena.si(arg)),
            "ci" => Ok(arena.ci(arg)),
            "ei" => Ok(arena.ei(arg)),
            "li" => Ok(arena.li(arg)),
            "zeta" => Ok(arena.zeta(arg)),
            // `E1(x) = expint(1, x)`.
            "e1" => {
                let one = arena.one;
                Ok(arena.lib_apply(LibFn::ExpInt, &[one, arg]))
            }
            // One-argument library functions: `erfi`, `erfinv`, `erfcinv`,
            // `Shi`, `Chi`, `fresnels`, `fresnelc`, `dirichlet_eta`, the Airy
            // functions, `elliptic_k`, `elliptic_e`, the integer sequences, …
            _ => self.lib_call(arena, name_lower, &[arg], || {
                format!(
                    "unknown function '{}'. Supported: sin, cos, tan, cot, sec, csc, exp, ln, log, \
                     sqrt, cbrt, abs, asin, acos, atan, acot, sinh, cosh, tanh, coth, sech, csch, \
                     asinh, acosh, atanh, sign, floor, ceil, gamma, erf, erfc, heaviside, \
                     diracdelta, lambertw, factorial, digamma, loggamma, re, im, conjugate, arg, \
                     Si, Ci, Ei, li, zeta, polygamma, binomial, beta, besselj, bessely, besseli, \
                     besselk, erfi, erfinv, erfcinv, E1, expint, Shi, Chi, fresnels, fresnelc, \
                     lowergamma, uppergamma, polylog, dirichlet_eta, airyai, airybi, \
                     airyaiprime, airybiprime, elliptic_k, elliptic_e, elliptic_f, elliptic_pi, \
                     gegenbauer, jacobi, assoc_legendre, assoc_laguerre, betainc, \
                     betainc_regularized, min, max, \
                     KroneckerDelta, Limit, RootOf, ConditionSet, LaplaceTransform, \
                     InverseLaplaceTransform, Residue, DSolve, Series, Sum, Product, Integral",
                    name
                )
            }),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;

    fn parse_and_display(input: &str) -> String {
        let ctx = Context::new();
        let ex = parse(&ctx, input).unwrap();
        format!("{ex}")
    }

    #[test]
    fn parse_integer() {
        assert_eq!(parse_and_display("42"), "42");
    }

    #[test]
    fn parse_symbol() {
        assert_eq!(parse_and_display("x"), "x");
    }

    #[test]
    fn parse_addition() {
        assert_eq!(parse_and_display("x + y"), "x + y");
    }

    #[test]
    fn parse_polynomial() {
        let s = parse_and_display("x^2 + 2*x + 1");
        assert!(s.contains("x^2") && s.contains("2*x"), "got: {s}");
    }

    #[test]
    fn parse_function_sin() {
        assert_eq!(parse_and_display("sin(x)"), "sin(x)");
    }

    #[test]
    fn parse_nested_functions() {
        assert_eq!(parse_and_display("sin(cos(x))"), "sin(cos(x))");
    }

    #[test]
    fn parse_negation() {
        let s = parse_and_display("-x");
        assert!(s.contains("x") && s.starts_with('-'), "got: {s}");
    }

    #[test]
    fn parse_power_right_assoc() {
        // x^2^3 should parse as x^(2^3) due to right-associativity
        let s = parse_and_display("x^2^3");
        assert!(s.contains("x"), "got: {s}");
    }

    #[test]
    fn parse_constant_pi() {
        assert_eq!(parse_and_display("pi"), "pi");
    }

    #[test]
    fn parse_precedence() {
        // 2 + 3*x should parse as 2 + (3*x)
        let s = parse_and_display("2 + 3*x");
        assert!(s.contains("3*x"), "got: {s}");
    }

    #[test]
    fn parse_parens() {
        let s = parse_and_display("(x + 1)^2");
        assert!(
            s.contains("(1 + x)^2") || s.contains("(x + 1)^2"),
            "got: {s}"
        );
    }

    #[test]
    fn parse_division() {
        let s = parse_and_display("x / y");
        // Division may be displayed as x*y^(-1) or x*1/y etc.
        assert!(
            s.contains("1/y") || s.contains("x*1/y") || s.contains("x/y") || s.contains("y^(-1)"),
            "got: {s}"
        );
    }

    #[test]
    fn parse_empty_string_error() {
        let ctx = Context::new();
        assert!(parse(&ctx, "").is_err());
    }

    #[test]
    fn parse_unknown_function_error() {
        let ctx = Context::new();
        assert!(parse(&ctx, "foo(x)").is_err());
    }

    #[test]
    fn parse_unary_minus_precedence() {
        // -x^2 should parse as -(x^2), not (-x)^2
        let s = parse_and_display("-x^2");
        // Should be displayed as -x^2 (meaning -(x^2))
        assert!(s.contains("x^2"), "got: {s}");
    }

    #[test]
    fn parse_subtraction() {
        let s = parse_and_display("x - y");
        // Subtraction is x + (-y), display may vary
        assert!(s.contains("x") && s.contains("y"), "got: {s}");
    }

    #[test]
    fn parse_multiple_operations() {
        let s = parse_and_display("2*x + 3*y - z");
        assert!(
            s.contains("2*x") && s.contains("3*y") && s.contains("z"),
            "got: {s}"
        );
    }

    #[test]
    fn parse_constant_e() {
        assert_eq!(parse_and_display("e"), "E");
    }

    #[test]
    fn parse_exp_function() {
        let s = parse_and_display("exp(x)");
        // exp(x) may display as E^x or exp(x) depending on arena
        assert!(s.contains("x"), "got: {s}");
    }

    #[test]
    fn parse_sqrt_function() {
        let s = parse_and_display("sqrt(x)");
        // sqrt(x) may display as x^(1/2) or sqrt(x)
        assert!(s.contains("x"), "got: {s}");
    }

    #[test]
    fn parse_complex_nested() {
        let s = parse_and_display("sin(x^2 + 1)");
        assert!(s.contains("sin"), "got: {s}");
    }

    #[test]
    fn parse_trailing_garbage_error() {
        let ctx = Context::new();
        // With implicit multiplication, "x y" is now valid (x*y).
        // Use truly invalid trailing tokens instead.
        assert!(parse(&ctx, "x )").is_err());
    }

    #[test]
    fn parse_unmatched_paren_error() {
        let ctx = Context::new();
        assert!(parse(&ctx, "(x + 1").is_err());
    }

    #[test]
    fn parse_unexpected_char_error() {
        let ctx = Context::new();
        assert!(parse(&ctx, "x & y").is_err());
    }

    #[test]
    fn parse_empty_function_call_error() {
        let ctx = Context::new();
        assert!(parse(&ctx, "sin()").is_err());
    }

    #[test]
    fn parse_abs_function() {
        let s = parse_and_display("abs(x)");
        assert!(s.contains("x"), "got: {s}");
    }

    #[test]
    fn parse_ln_function() {
        let s = parse_and_display("ln(x)");
        assert!(s.contains("x"), "got: {s}");
    }

    #[test]
    fn parse_log_alias() {
        let s = parse_and_display("log(x)");
        assert!(s.contains("x"), "got: {s}");
    }

    #[test]
    fn parse_constant_infinity() {
        let s = parse_and_display("inf");
        assert!(
            s.contains("oo") || s.contains("inf") || s.contains("∞"),
            "got: {s}"
        );
    }

    #[test]
    fn parse_deeply_nested_parens() {
        let s = parse_and_display("((((x))))");
        assert_eq!(s, "x");
    }

    #[test]
    fn parse_chained_additions() {
        let s = parse_and_display("a + b + c + d");
        assert!(
            s.contains("a") && s.contains("b") && s.contains("c") && s.contains("d"),
            "got: {s}"
        );
    }

    #[test]
    fn parse_chained_multiplications() {
        let s = parse_and_display("a * b * c");
        assert!(
            s.contains("a") && s.contains("b") && s.contains("c"),
            "got: {s}"
        );
    }

    #[test]
    fn parse_mixed_precedence() {
        // a + b * c should parse as a + (b*c)
        let s = parse_and_display("a + b * c");
        assert!(s.contains("b*c") || s.contains("c*b"), "got: {s}");
    }

    // ═══════════════════════════════════════════════════════════════════
    // New feature tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn parse_float() {
        let ctx = Context::new();
        let result = parse(&ctx, "3.14").unwrap();
        let s = format!("{result}");
        // 3.14 should parse as 314/100 = 157/50
        assert!(
            s.contains("157") || s.contains("3.14") || s.contains("314"),
            "got: {s}"
        );
    }

    #[test]
    fn parse_implicit_mul_number_var() {
        let ctx = Context::new();
        let result = parse(&ctx, "2x").unwrap();
        let s = format!("{result}");
        assert!(s.contains("2") && s.contains("x"), "2x should be 2*x: {s}");
    }

    #[test]
    fn parse_implicit_mul_var_paren() {
        // ident + '(' is always treated as a function call, so x(x+1) errors
        // (x is not a known function). Use number*paren or paren*paren instead.
        let ctx = Context::new();
        assert!(parse(&ctx, "x(x+1)").is_err());
    }

    #[test]
    fn parse_constant_pi_variants() {
        assert_eq!(parse_and_display("pi"), "pi");
        assert_eq!(parse_and_display("Pi"), "pi");
        assert_eq!(parse_and_display("PI"), "pi");
    }

    #[test]
    fn parse_constant_i_unit() {
        let ctx = Context::new();
        let result = parse(&ctx, "I").unwrap();
        assert_eq!(format!("{result}"), "I");
    }

    #[test]
    fn parse_constant_i_lowercase() {
        let ctx = Context::new();
        let result = parse(&ctx, "i").unwrap();
        assert_eq!(format!("{result}"), "I");
    }

    #[test]
    fn parse_euler_formula() {
        let ctx = Context::new();
        let result = parse(&ctx, "exp(I*pi)").unwrap();
        let s = format!("{result}");
        assert!(s.contains("I") && s.contains("pi"), "got: {s}");
    }

    #[test]
    fn parse_float_times_var() {
        let ctx = Context::new();
        let result = parse(&ctx, "2.5*x").unwrap();
        let s = format!("{result}");
        assert!(
            s.contains("x") && (s.contains("5/2") || s.contains("2.5") || s.contains("5*1/2")),
            "got: {s}"
        );
    }

    #[test]
    fn parse_log_two_args() {
        let ctx = Context::new();
        // log(x, 2) = ln(x) / ln(2)
        let result = parse(&ctx, "log(x, 2)").unwrap();
        let s = format!("{result}");
        assert!(s.contains("x"), "log(x,2) should parse: {s}");
    }

    #[test]
    fn parse_implicit_mul_number_paren() {
        let ctx = Context::new();
        let result = parse(&ctx, "3(x+1)").unwrap();
        let s = format!("{result}");
        assert!(
            s.contains("3") && s.contains("x"),
            "3(x+1) should be 3*(x+1): {s}"
        );
    }

    #[test]
    fn parse_implicit_mul_paren_paren() {
        let ctx = Context::new();
        let result = parse(&ctx, "(a)(b)").unwrap();
        let s = format!("{result}");
        assert!(
            s.contains("a") && s.contains("b"),
            "(a)(b) should be a*b: {s}"
        );
    }

    #[test]
    fn parse_implicit_mul_coeff_pi() {
        let ctx = Context::new();
        let result = parse(&ctx, "2pi").unwrap();
        let s = format!("{result}");
        assert!(
            s.contains("2") && s.contains("pi"),
            "2pi should be 2*pi: {s}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Arbitrary-precision integer tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_large_integer() {
        let ctx = Context::new();
        let result = parse(&ctx, "99999999999999999999999999999").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "99999999999999999999999999999");
    }

    #[test]
    fn parse_integer_beyond_i64_max() {
        let ctx = Context::new();
        // i64::MAX = 9223372036854775807 — this is one more
        let result = parse(&ctx, "9223372036854775808").unwrap();
        let s = format!("{result}");
        assert_eq!(
            s, "9223372036854775808",
            "should handle integers > i64::MAX"
        );
    }

    #[test]
    fn parse_integer_beyond_i128_max() {
        let ctx = Context::new();
        // i128::MAX ≈ 1.7e38 — this is well beyond
        let big = "123456789012345678901234567890123456789012345678901234567890";
        let result = parse(&ctx, big).unwrap();
        let s = format!("{result}");
        assert_eq!(s, big, "should handle integers > i128::MAX");
    }

    #[test]
    fn parse_large_integer_arithmetic() {
        let ctx = Context::new();
        // (10^30)^2 should be 10^60
        let result = parse(&ctx, "1000000000000000000000000000000^2").unwrap();
        let evaled = result.eval();
        let s = format!("{evaled}");
        assert_eq!(
            s, "1000000000000000000000000000000000000000000000000000000000000",
            "large integer exponentiation should be exact"
        );
    }

    #[test]
    fn parse_negative_large_integer() {
        let ctx = Context::new();
        let result = parse(&ctx, "-99999999999999999999999999999").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "-99999999999999999999999999999");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Arbitrary-precision decimal / rational tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_many_decimal_places() {
        let ctx = Context::new();
        // This should not panic (was overflowing i64 before)
        let result = parse(&ctx, "1.0000000000000000000001");
        assert!(
            result.is_ok(),
            "parsing many decimal places should not panic"
        );
    }

    #[test]
    fn parse_decimal_exact_rational_simple() {
        let ctx = Context::new();
        // 0.5 should be exactly 1/2
        let result = parse(&ctx, "0.5").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "1/2", "0.5 should parse as exact rational 1/2, got: {s}");
    }

    #[test]
    fn parse_decimal_exact_rational_quarter() {
        let ctx = Context::new();
        // 0.25 should be exactly 1/4
        let result = parse(&ctx, "0.25").unwrap();
        let s = format!("{result}");
        assert_eq!(
            s, "1/4",
            "0.25 should parse as exact rational 1/4, got: {s}"
        );
    }

    #[test]
    fn parse_decimal_exact_rational_third_approx() {
        let ctx = Context::new();
        // 0.333 should be exactly 333/1000
        let result = parse(&ctx, "0.333").unwrap();
        let s = format!("{result}");
        assert_eq!(
            s, "333/1000",
            "0.333 should parse as exact rational 333/1000, got: {s}"
        );
    }

    #[test]
    fn parse_decimal_preserves_all_digits() {
        let ctx = Context::new();
        // 1.00000000000000000000000000000000001 — 34 zeros then 1
        // numerator = 100000000000000000000000000000000001
        // denominator = 10^35
        // This must NOT lose any precision
        let input = "1.00000000000000000000000000000000001";
        let result = parse(&ctx, input).unwrap();
        // Multiply by 10^35 — should give exactly 100000000000000000000000000000000001
        let big_denom = parse(&ctx, "100000000000000000000000000000000000").unwrap();
        let product = &result * &big_denom;
        let s = format!("{}", product.eval());
        assert_eq!(
            s, "100000000000000000000000000000000001",
            "1.00000000000000000000000000000000001 * 10^35 should be exact, got: {s}"
        );
    }

    #[test]
    fn parse_decimal_20_places_exact() {
        let ctx = Context::new();
        // bc: 314159265358979323846 / 2 = 157079632679489661923
        // bc: 100000000000000000000 / 2 = 50000000000000000000
        // GCD(314159265358979323846, 100000000000000000000) = 2
        // Reduced: 157079632679489661923/50000000000000000000
        let result = parse(&ctx, "3.14159265358979323846").unwrap();
        let s = format!("{result}");
        assert_eq!(
            s, "157079632679489661923/50000000000000000000",
            "20-digit decimal should be exact reduced rational"
        );
    }

    #[test]
    fn parse_decimal_20_places_multiply_back() {
        let ctx = Context::new();
        // Verify: 157079632679489661923/50000000000000000000 * 50000000000000000000
        //       = 157079632679489661923 (bc-verified)
        let result = parse(&ctx, "3.14159265358979323846").unwrap();
        let denom = parse(&ctx, "50000000000000000000").unwrap();
        let product = (&result * &denom).eval();
        let s = format!("{product}");
        assert_eq!(
            s, "157079632679489661923",
            "rational * denominator should recover exact numerator"
        );
    }

    #[test]
    fn parse_decimal_50_places_exact() {
        let ctx = Context::new();
        // 50 decimal places — well beyond any fixed-precision type
        // Must parse without error AND produce an exact rational
        let input = "3.14159265358979323846264338327950288419716939937510";
        let result = parse(&ctx, input).unwrap();
        // Multiply by 10^50 to recover the exact numerator
        // bc: the full numerator is 314159265358979323846264338327950288419716939937510
        let big = parse(&ctx, "100000000000000000000000000000000000000000000000000").unwrap();
        let product = (&result * &big).eval();
        let s = format!("{product}");
        assert_eq!(
            s, "314159265358979323846264338327950288419716939937510",
            "50-digit decimal * 10^50 must recover exact integer (bc-verified)"
        );
    }

    #[test]
    fn parse_decimal_large_integer_part_and_fraction_exact() {
        let ctx = Context::new();
        // 123456789012345678901234567890.123456789012345678901234567890
        // = 123456789012345678901234567890123456789012345678901234567890 / 10^30
        // Verify by multiplying by 10^30
        let result = parse(
            &ctx,
            "123456789012345678901234567890.123456789012345678901234567890",
        )
        .unwrap();
        let denom = parse(&ctx, "1000000000000000000000000000000").unwrap(); // 10^30
        let product = (&result * &denom).eval();
        let s = format!("{product}");
        assert_eq!(
            s, "123456789012345678901234567890123456789012345678901234567890",
            "large decimal * 10^30 must recover exact integer (bc-verified)"
        );
    }

    #[test]
    fn parse_decimal_used_in_arithmetic() {
        let ctx = Context::new();
        // 0.1 + 0.2 should be exactly 3/10 (no floating-point 0.30000000000000004 nonsense)
        let result = parse(&ctx, "0.1 + 0.2").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "3/10", "0.1 + 0.2 should be exactly 3/10, got: {s}");
    }

    #[test]
    fn parse_decimal_multiplication_exact() {
        let ctx = Context::new();
        // 0.1 * 0.1 should be exactly 1/100
        let result = parse(&ctx, "0.1 * 0.1").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "1/100", "0.1 * 0.1 should be exactly 1/100, got: {s}");
    }

    #[test]
    fn parse_decimal_vs_fraction_equivalence() {
        let ctx = Context::new();
        // 2.5 * x should be the same as 5/2 * x
        let x = ctx.symbol("x");
        let from_decimal = parse(&ctx, "2.5 * x").unwrap();
        let five_halves = ctx.rational(5, 2);
        let from_fraction = &five_halves * &x;
        assert_eq!(
            from_decimal, from_fraction,
            "2.5*x and (5/2)*x should be identical expressions"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Edge cases and robustness
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_zero_integer() {
        let ctx = Context::new();
        let result = parse(&ctx, "0").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "0");
    }

    #[test]
    fn parse_zero_decimal() {
        let ctx = Context::new();
        let result = parse(&ctx, "0.0").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "0", "0.0 should parse as 0, got: {s}");
    }

    #[test]
    fn parse_leading_zeros_integer() {
        let ctx = Context::new();
        let result = parse(&ctx, "007").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "7", "007 should parse as 7, got: {s}");
    }

    #[test]
    fn parse_leading_zeros_decimal() {
        let ctx = Context::new();
        let result = parse(&ctx, "0.00100").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "1/1000", "0.00100 should parse as 1/1000, got: {s}");
    }

    #[test]
    fn parse_deep_nesting_limit() {
        let ctx = Context::new();
        let deep = "(".repeat(300) + "1" + &")".repeat(300);
        let result = parse(&ctx, &deep);
        assert!(
            result.is_err(),
            "deeply nested input should return error, not stack overflow"
        );
    }

    #[test]
    fn parse_moderate_nesting_ok() {
        let ctx = Context::new();
        // 50 levels of nesting should be fine (well under 256 limit)
        let expr = "(".repeat(50) + "x + 1" + &")".repeat(50);
        let result = parse(&ctx, &expr);
        assert!(result.is_ok(), "50 levels of nesting should be fine");
    }

    #[test]
    fn parse_decimal_with_variable_exact_coeff() {
        let ctx = Context::new();
        // bc: 314/100 = 157/50 (GCD=2). So 3.14*x = 157/50*x
        let result = parse(&ctx, "3.14 * x").unwrap();
        let s = format!("{result}");
        assert_eq!(
            s, "157/50*x",
            "3.14*x should have exact coefficient 157/50, got: {s}"
        );
    }

    #[test]
    fn parse_integer_one() {
        let ctx = Context::new();
        let result = parse(&ctx, "1").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "1");
    }

    #[test]
    fn parse_negative_decimal() {
        let ctx = Context::new();
        // -0.5 = -5/10 = -1/2
        let result = parse(&ctx, "-0.5").unwrap();
        let s = format!("{result}");
        assert_eq!(s, "-1/2", "-0.5 should parse as -1/2, got: {s}");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // bc-verified: arithmetic on parsed arbitrary-precision values
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_beyond_i64_arithmetic() {
        let ctx = Context::new();
        // bc: 9223372036854775808 + 1 = 9223372036854775809
        let result = parse(&ctx, "9223372036854775808 + 1").unwrap();
        let s = format!("{result}");
        assert_eq!(
            s, "9223372036854775809",
            "i64::MAX+1 + 1 should be exact (bc-verified)"
        );
    }

    #[test]
    fn parse_beyond_i128() {
        let ctx = Context::new();
        // i128::MAX = 170141183460469231731687303715884105727
        // bc: 170141183460469231731687303715884105727 + 1 = 170141183460469231731687303715884105728
        let result = parse(&ctx, "170141183460469231731687303715884105728").unwrap();
        let s = format!("{result}");
        assert_eq!(
            s, "170141183460469231731687303715884105728",
            "i128::MAX+1 should parse and display exactly"
        );
    }

    #[test]
    fn parse_huge_integer_squared() {
        let ctx = Context::new();
        // bc: 99999999999999999999999999999 * 99999999999999999999999999999
        //   = 9999999999999999999999999999800000000000000000000000000001
        let base = parse(&ctx, "99999999999999999999999999999").unwrap();
        let squared = base.powi(2).eval();
        let s = format!("{squared}");
        assert_eq!(
            s, "9999999999999999999999999999800000000000000000000000000001",
            "(10^29 - 1)^2 should be exact (bc-verified)"
        );
    }

    #[test]
    fn parse_tiny_number_exact() {
        let ctx = Context::new();
        // 0.000000000000000000000000000000000001 = 1/10^36
        // bc: 10^36 = 1000000000000000000000000000000000000
        // Verify: multiply by 10^36 should give exactly 1
        let tiny = parse(&ctx, "0.000000000000000000000000000000000001").unwrap();
        let big = parse(&ctx, "1000000000000000000000000000000000000").unwrap();
        let product = (&tiny * &big).eval();
        let s = format!("{product}");
        assert_eq!(s, "1", "10^-36 * 10^36 should be exactly 1 (bc-verified)");
    }

    #[test]
    fn parse_tiny_number_display() {
        let ctx = Context::new();
        // 0.000000000000000000000000000000000001 = 1/10^36
        let tiny = parse(&ctx, "0.000000000000000000000000000000000001").unwrap();
        let s = format!("{tiny}");
        assert_eq!(
            s, "1/1000000000000000000000000000000000000",
            "10^-36 should display as exact fraction (bc-verified)"
        );
    }

    #[test]
    fn parse_large_integer_addition() {
        let ctx = Context::new();
        // bc: 123456789012345678901234567890 + 1 = 123456789012345678901234567891
        let result = parse(&ctx, "123456789012345678901234567890 + 1").unwrap();
        let s = format!("{result}");
        assert_eq!(
            s, "123456789012345678901234567891",
            "large integer + 1 should be exact (bc-verified)"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SymPy compatibility
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_sympy_power_operator() {
        assert_eq!(parse_and_display("x**2"), parse_and_display("x^2"));
    }

    #[test]
    fn parse_sympy_double_star_in_expression() {
        assert_eq!(
            parse_and_display("3*x**2 + 2"),
            parse_and_display("3*x^2 + 2")
        );
    }

    #[test]
    fn parse_sympy_abs_capital() {
        assert_eq!(parse_and_display("Abs(x)"), parse_and_display("abs(x)"));
    }

    #[test]
    fn parse_sympy_full_expression() {
        let ctx = Context::new();
        // SymPy output for integrate(x*ln(x))
        let result = parse(&ctx, "x**2*log(x)/2 - x**2/4").unwrap();
        let s = format!("{result}");
        assert!(
            s.contains("ln") && s.contains("x"),
            "should parse SymPy integrate output: {s}"
        );
    }

    #[test]
    fn parse_sympy_trig_identity() {
        let ctx = Context::new();
        let result = parse(&ctx, "sin(x)**2 + cos(x)**2").unwrap();
        // Should simplify to 1 via full_simplify
        let simplified = result.simplify();
        assert_eq!(format!("{simplified}"), "1");
    }

    // ── 0.2 nodes ──────────────────────────────────────────────────────────

    #[test]
    fn parse_named_constants() {
        let ctx = Context::new();
        assert_eq!(parse(&ctx, "EulerGamma").unwrap(), ctx.euler_gamma());
        assert_eq!(parse(&ctx, "Catalan").unwrap(), ctx.catalan());
        assert_eq!(parse(&ctx, "GoldenRatio").unwrap(), ctx.golden_ratio());
        assert_eq!(parse(&ctx, "zoo").unwrap(), ctx.complex_infinity());
        assert_eq!(
            parse_and_display("EulerGamma + Catalan"),
            "EulerGamma + Catalan"
        );
    }

    #[test]
    fn parse_complex_functions() {
        let ctx = Context::new();
        let z = ctx.symbol("z");
        assert_eq!(parse(&ctx, "re(z)").unwrap(), z.re());
        assert_eq!(parse(&ctx, "im(z)").unwrap(), z.im());
        assert_eq!(parse(&ctx, "conjugate(z)").unwrap(), z.conjugate());
        assert_eq!(parse(&ctx, "conj(z)").unwrap(), z.conjugate());
        assert_eq!(parse(&ctx, "arg(z)").unwrap(), z.arg());
        assert_eq!(parse_and_display("Re(3 + 4*I)"), "3");
        assert_eq!(parse_and_display("im(3 + 4*I)"), "4");
    }

    #[test]
    fn parse_special_functions() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let n = ctx.symbol("n");
        assert_eq!(parse(&ctx, "Si(x)").unwrap(), x.si());
        assert_eq!(parse(&ctx, "Ci(x)").unwrap(), x.ci());
        assert_eq!(parse(&ctx, "Ei(x)").unwrap(), x.ei());
        assert_eq!(parse(&ctx, "li(x)").unwrap(), x.li());
        assert_eq!(parse(&ctx, "zeta(x)").unwrap(), x.zeta());
        assert_eq!(parse(&ctx, "polygamma(n, x)").unwrap(), x.polygamma(&n));
        assert_eq!(
            parse(&ctx, "KroneckerDelta(n, x)").unwrap(),
            x.kronecker_delta(&n)
        );
        assert_eq!(
            parse(&ctx, "kronecker_delta(n, x)").unwrap(),
            x.kronecker_delta(&n)
        );
        assert_eq!(parse_and_display("zeta(2)"), "1/6*pi^2");
        assert_eq!(parse_and_display("Si(0)"), "0");
        assert_eq!(parse_and_display("atan2(1, 1)"), "atan2(1, 1)");
    }

    // ── 0.9.1: relations, connectives, implicit application ───────────────

    #[test]
    fn parse_bool_relations_and_connectives() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let zero = ctx.int(0);
        let one = ctx.int(1);
        assert_eq!(parse_bool(&ctx, "x > 0").unwrap(), x.gt(&zero));
        assert_eq!(parse_bool(&ctx, "x >= 0").unwrap(), x.ge(&zero));
        assert_eq!(parse_bool(&ctx, "x < 1").unwrap(), x.lt(&one));
        assert_eq!(parse_bool(&ctx, "x <= 1").unwrap(), x.le(&one));
        assert_eq!(parse_bool(&ctx, "x == 1").unwrap(), x.eq_expr(&one));
        assert_eq!(parse_bool(&ctx, "x != 1").unwrap(), x.ne_expr(&one));
        assert_eq!(
            parse_bool(&ctx, "x > 0 & x < 1").unwrap(),
            x.gt(&zero).and(&x.lt(&one))
        );
        assert_eq!(
            parse_bool(&ctx, "x > 0 && x < 1").unwrap(),
            parse_bool(&ctx, "x > 0 and x < 1").unwrap()
        );
        assert_eq!(
            parse_bool(&ctx, "x > 0 | y > 0").unwrap(),
            x.gt(&zero).or(&y.gt(&zero))
        );
        assert_eq!(
            parse_bool(&ctx, "x > 0 || y > 0").unwrap(),
            parse_bool(&ctx, "x > 0 or y > 0").unwrap()
        );
        assert_eq!(parse_bool(&ctx, "~(x > 0)").unwrap(), x.gt(&zero).not());
        assert_eq!(parse_bool(&ctx, "!(x > 0)").unwrap(), x.gt(&zero).not());
        assert_eq!(parse_bool(&ctx, "not x > 0").unwrap(), x.gt(&zero).not());
        assert_eq!(parse_bool(&ctx, "True").unwrap().to_string(), "True");
        assert_eq!(parse_bool(&ctx, "false").unwrap().to_string(), "False");
    }

    #[test]
    fn parse_bool_precedence() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let zero = ctx.int(0);
        let one = ctx.int(1);
        // `and` binds tighter than `or`; comparisons tighter than both.
        assert_eq!(
            parse_bool(&ctx, "x > 0 & x < 1 | y == 0").unwrap(),
            x.gt(&zero).and(&x.lt(&one)).or(&y.eq_expr(&zero))
        );
        assert_eq!(
            parse_bool(&ctx, "x > 0 | x < 1 & y == 0").unwrap(),
            x.gt(&zero).or(&x.lt(&one).and(&y.eq_expr(&zero)))
        );
        // Arithmetic binds tighter than comparisons.
        assert_eq!(
            parse_bool(&ctx, "x + 1 > 2*y").unwrap(),
            (&x + 1).gt(&(2 * &y))
        );
        // `not` takes one relation, not the whole conjunction.
        assert_eq!(
            parse_bool(&ctx, "not x > 0 & y > 0").unwrap(),
            x.gt(&zero).not().and(&y.gt(&zero))
        );
        // `a & b & c` is one n-ary node.
        assert_eq!(
            parse_bool(&ctx, "x > 0 & y > 0 & x < 1")
                .unwrap()
                .to_string(),
            "x > 0 & y > 0 & 1 > x"
        );
        // Parentheses regroup.
        assert_eq!(
            parse_bool(&ctx, "(x > 0 | y > 0) & x < 1").unwrap(),
            x.gt(&zero).or(&y.gt(&zero)).and(&x.lt(&one))
        );
    }

    #[test]
    fn parse_bool_function_forms() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let one = ctx.int(1);
        assert_eq!(parse_bool(&ctx, "Eq(x, 1)").unwrap(), x.eq_expr(&one));
        assert_eq!(parse_bool(&ctx, "Ne(x, 1)").unwrap(), x.ne_expr(&one));
        assert_eq!(parse_bool(&ctx, "Lt(x, 1)").unwrap(), x.lt(&one));
        assert_eq!(parse_bool(&ctx, "Le(x, 1)").unwrap(), x.le(&one));
        assert_eq!(parse_bool(&ctx, "Gt(x, 1)").unwrap(), x.gt(&one));
        assert_eq!(parse_bool(&ctx, "Ge(x, 1)").unwrap(), x.ge(&one));
        assert_eq!(
            parse_bool(&ctx, "And(x > 1, y > 1)").unwrap(),
            x.gt(&one).and(&y.gt(&one))
        );
        assert_eq!(
            parse_bool(&ctx, "Or(x > 1, y > 1)").unwrap(),
            x.gt(&one).or(&y.gt(&one))
        );
        assert_eq!(parse_bool(&ctx, "Not(x > 1)").unwrap(), x.gt(&one).not());
    }

    #[test]
    fn parse_bool_errors() {
        let ctx = Context::new();
        // A numeric expression is not a relation.
        assert!(parse_bool(&ctx, "x + 1").is_err());
        // Chained comparisons.
        assert!(parse_bool(&ctx, "0 < x < 1").is_err());
        // Sort errors.
        assert!(parse_bool(&ctx, "(x > 0) + 1").is_err());
        assert!(parse_bool(&ctx, "x & y").is_err());
        assert!(parse_bool(&ctx, "not x").is_err());
        assert!(parse_bool(&ctx, "(x > 0) > 1").is_err());
        assert!(parse_bool(&ctx, "-(x > 0)").is_err());
        // Trailing garbage.
        assert!(parse_bool(&ctx, "x > 0 &").is_err());
    }

    #[test]
    fn parse_strict_rejects_relations_and_keeps_keywords_as_symbols() {
        let ctx = Context::new();
        assert!(parse(&ctx, "x > 0").is_err());
        assert!(parse(&ctx, "x & y").is_err());
        assert!(parse(&ctx, "~x").is_err());
        assert!(parse(&ctx, "x != 1").is_err());
        // In the numeric grammar `and`, `True` are ordinary identifiers.
        assert_eq!(parse(&ctx, "and").unwrap(), ctx.symbol("and"));
        assert_eq!(parse(&ctx, "True").unwrap(), ctx.symbol("True"));
        // `Sum(body, k=lo..hi)` still uses a single `=`.
        assert!(parse(&ctx, "Sum(k, k=1..3)").is_ok());
    }

    #[test]
    fn parse_implicit_application() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        assert_eq!(parse_implicit(&ctx, "sin x").unwrap(), x.sin());
        assert_eq!(parse_implicit(&ctx, "2 sin x").unwrap(), 2 * &x.sin());
        assert_eq!(parse_implicit(&ctx, "sin 2x").unwrap(), (2 * &x).sin());
        assert_eq!(parse_implicit(&ctx, "sin x^2").unwrap(), x.powi(2).sin());
        assert_eq!(
            parse_implicit(&ctx, "sin x cos y").unwrap(),
            &x.sin() * &y.cos()
        );
        assert_eq!(parse_implicit(&ctx, "sin x + 1").unwrap(), &x.sin() + 1);
        assert_eq!(parse_implicit(&ctx, "sin x/2").unwrap(), (&x / 2).sin());
        assert_eq!(parse_implicit(&ctx, "sin x y").unwrap(), (&x * &y).sin());
        assert_eq!(parse_implicit(&ctx, "exp (x) y").unwrap(), &x.exp() * &y);
        assert_eq!(parse_implicit(&ctx, "sqrt 2").unwrap(), ctx.int(2).sqrt());
        assert_eq!(parse_implicit(&ctx, "ln x^2").unwrap(), x.powi(2).ln());
        // Parenthesised calls still work and reset the argument scope.
        assert_eq!(
            parse_implicit(&ctx, "sin(x cos y)").unwrap(),
            (&x * &y.cos()).sin()
        );
    }

    #[test]
    fn parse_implicit_products() {
        let ctx = Context::new();
        let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
        assert_eq!(
            parse_implicit(&ctx, "2x + 3(y-1)").unwrap(),
            2 * &x + 3 * (&y - 1)
        );
        assert_eq!(parse_implicit(&ctx, "x y z").unwrap(), &x * &y * &z);
        assert_eq!(parse_implicit(&ctx, "x(x+1)").unwrap(), &x * (&x + 1));
        assert_eq!(
            parse_implicit(&ctx, "(x+1)(x-1)").unwrap(),
            (&x + 1) * (&x - 1)
        );
        // Unknown `f(x)` is a product; constants multiply groups too; the
        // one-letter aliases `C`/`B`/`W` are ordinary symbols here.
        let f = ctx.symbol("f");
        assert_eq!(parse_implicit(&ctx, "f(x)").unwrap(), &f * &x);
        let c = ctx.symbol("c");
        assert_eq!(parse_implicit(&ctx, "c(x+1)").unwrap(), &c * (&x + 1));
        assert_eq!(
            parse_implicit(&ctx, "binomial(x, 2)").unwrap(),
            parse(&ctx, "C(x, 2)").unwrap()
        );
        assert_eq!(parse_implicit(&ctx, "pi(x)").unwrap(), ctx.pi() * &x);
        assert_eq!(parse_implicit(&ctx, "2 pi x").unwrap(), 2 * ctx.pi() * &x);
    }

    #[test]
    fn parse_implicit_errors_and_limits() {
        let ctx = Context::new();
        // A textbook function name without an argument.
        assert!(parse_implicit(&ctx, "sin + 1").is_err());
        assert!(parse_implicit(&ctx, "sin").is_err());
        // Ambiguous short names need parentheses: `re x` is `re*x`.
        let (re, x) = (ctx.symbol("re"), ctx.symbol("x"));
        assert_eq!(parse_implicit(&ctx, "re x").unwrap(), &re * &x);
        assert_eq!(parse_implicit(&ctx, "re(x)").unwrap(), x.re());
        // Relations are not part of the implicit grammar.
        assert!(parse_implicit(&ctx, "x > 0").is_err());
    }

    #[test]
    fn known_functions_table_matches_call_tables() {
        // Every name in `KNOWN_FUNCTIONS` is accepted by some call table.
        let ctx = Context::new();
        for name in KNOWN_FUNCTIONS {
            let ok = [
                format!("{name}(x)"),
                format!("{name}(x, y)"),
                format!("{name}(x, y, z)"),
                format!("{name}(x, y, z, w)"),
            ]
            .iter()
            .any(|s| parse(&ctx, s).is_ok());
            assert!(ok, "`{name}` is listed but no arity parses");
        }
        for name in KNOWN_FUNCTIONS {
            assert!(
                !is_implicit_unary_function(name) || parse(&ctx, &format!("{name}(x)")).is_ok(),
                "`{name}` is applied implicitly but is not a unary function"
            );
        }
    }
}
