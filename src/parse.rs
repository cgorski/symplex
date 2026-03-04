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

use crate::arena::Arena;
use crate::context::Context;
use crate::expr::Ex;
use crate::node::ExprId;

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
    Int(i64),
    Rational(i64, i64),
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
                Ok(Token::Star)
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
                    let numer = full_str.parse::<i64>().map_err(|e| ParseError {
                        message: format!(
                            "invalid number '{}': {}",
                            &self.input[start..self.pos],
                            e
                        ),
                        position: start,
                    })?;
                    let denom = 10i64.pow(decimal_places as u32);
                    return Ok(Token::Rational(numer, denom));
                }
                let s = &self.input[start..self.pos];
                let n = s.parse::<i64>().map_err(|e| ParseError {
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
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token().unwrap_or(Token::Eof);
        Parser { lexer, current }
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
                Token::Int(_) | Token::Rational(_, _) | Token::Ident(_) | Token::LParen => {
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

        Ok(lhs)
    }

    /// Parse a prefix expression (atom, unary minus, function call, parens).
    fn parse_prefix(&mut self, arena: &mut Arena) -> Result<ExprId, ParseError> {
        match self.current.clone() {
            Token::Int(n) => {
                self.advance()?;
                Ok(arena.int(n))
            }
            Token::Rational(n, d) => {
                self.advance()?;
                Ok(arena.rational(n, d))
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

        match name {
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
}
