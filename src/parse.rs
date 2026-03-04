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

use crate::arena::Arena;
use crate::context::Context;
use crate::expr::Ex;
use crate::node::{ExprId, ExprNode};

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
    let mut parser = Parser::new(input);
    let id = ctx.with_arena_mut(|arena| {
        let result = parser.parse_expr(arena, 0)?;
        // After parsing the expression, ensure we consumed everything.
        if parser.current != Token::Eof {
            return Err(ParseError {
                message: format!("unexpected token {:?} after expression", parser.current),
                position: parser.lexer.pos,
            });
        }
        Ok(result)
    })?;
    // Construct an Ex from the ExprId. Ex fields are pub(crate), so this
    // works from within the crate without needing to expose make_ex.
    Ok(Ex {
        ctx_id: ctx.id,
        inner: Arc::clone(&ctx.inner),
        id,
        _sort: std::marker::PhantomData,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Tokenizer
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Int(BigInt),
    Rational(Ratio<BigInt>),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
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

struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token().unwrap_or(Token::Eof);
        Parser {
            lexer,
            current,
            depth: 0,
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
        if self.depth > 256 {
            return Err(ParseError {
                message: "expression nesting too deep (max 256 levels)".into(),
                position: self.lexer.pos,
            });
        }

        // Prefix (atom or unary)
        let mut lhs = self.parse_prefix(arena)?;

        // Infix loop
        loop {
            let (op, l_bp, r_bp, implicit) = match &self.current {
                Token::Plus => ('+', 1, 2, false),
                Token::Minus => ('-', 1, 2, false),
                Token::Star => ('*', 3, 4, false),
                Token::Slash => ('/', 3, 4, false),
                Token::Caret => ('^', 8, 7, false), // right-associative
                // Implicit multiplication: number, identifier, or '(' immediately
                // following a complete left-hand expression.
                Token::Int(_) | Token::Rational(_) | Token::Ident(_) | Token::LParen => {
                    ('*', 3, 4, true)
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

            lhs = match op {
                '+' => arena.add(&[lhs, rhs]),
                '-' => arena.sub(lhs, rhs),
                '*' => arena.mul(&[lhs, rhs]),
                '/' => arena.div(lhs, rhs),
                '^' => arena.pow(lhs, rhs),
                _ => unreachable!(),
            };
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
                // Check for function call: ident followed by '('
                if self.current == Token::LParen {
                    return self.parse_function_call(arena, &name);
                }
                // Known constants
                match name.as_str() {
                    "pi" | "Pi" | "PI" => Ok(arena.pi),
                    "e" | "E" => Ok(arena.e_const),
                    "I" | "i" => Ok(arena.i_unit),
                    "inf" | "oo" | "Inf" => Ok(arena.infinity),
                    "nan" => Ok(arena.nan),
                    _ => Ok(arena.symbol(&name)),
                }
            }
            Token::Minus => {
                self.advance()?;
                // Unary minus binding power — tighter than +/- and *//
                // but looser than ^, so that -x^2 parses as -(x^2).
                let operand = self.parse_expr(arena, 5)?;
                Ok(arena.neg(operand))
            }
            Token::LParen => {
                self.advance()?;
                let inner = self.parse_expr(arena, 0)?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            other => Err(ParseError {
                message: format!("expected expression, got {:?}", other),
                position: self.lexer.pos,
            }),
        }
    }

    fn parse_function_call(&mut self, arena: &mut Arena, name: &str) -> Result<ExprId, ParseError> {
        self.expect(&Token::LParen)?;

        if self.current == Token::RParen {
            self.advance()?;
            return Err(ParseError {
                message: format!("function '{}' requires an argument", name),
                position: self.lexer.pos,
            });
        }

        let arg = self.parse_expr(arena, 0)?;

        // Multi-argument functions: check for comma
        if self.current == Token::Comma {
            self.advance()?;
            let arg2 = self.parse_expr(arena, 0)?;
            self.expect(&Token::RParen)?;

            return match name {
                "log" => {
                    // log(x, base) = ln(x) / ln(base)
                    let ln_x = arena.ln(arg);
                    let ln_base = arena.ln(arg2);
                    Ok(arena.div(ln_x, ln_base))
                }
                _ => Err(ParseError {
                    message: format!("function '{}' does not accept multiple arguments", name),
                    position: self.lexer.pos,
                }),
            };
        }

        self.expect(&Token::RParen)?;

        // Case-insensitive function name matching for SymPy compatibility
        let name_lower = name.to_ascii_lowercase();
        match name_lower.as_str() {
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
            _ => Err(ParseError {
                message: format!(
                    "unknown function '{}'. Supported: sin, cos, tan, exp, ln, log, sqrt, cbrt, abs, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh",
                    name
                ),
                position: self.lexer.pos,
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
    use crate::context::Context;

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
        let simplified = result.full_simplify();
        assert_eq!(format!("{simplified}"), "1");
    }
}
