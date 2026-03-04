//! Pratt parser for math expressions over `syn::ParseStream`.
//!
//! This module provides [`MathExpr`], a simple AST for mathematical
//! expressions, and [`parse_math_expr`], a precedence-climbing parser
//! that builds a `MathExpr` from a `syn` token stream.
//!
//! The grammar handled:
//!
//! ```text
//! expr    := logic_or
//! logic_or  := logic_and (('||') logic_and)*
//! logic_and := comparison (('&&') comparison)*
//! comparison := addition (('>' | '<' | '>=' | '<=' | '==' | '!=') addition)*
//! addition  := term (('+' | '-') term)*
//! term      := unary (('*' | '/') unary)*
//! unary     := '-' unary | '!' unary | power
//! power     := primary ('^' power)?          // right-associative
//! primary   := INT | IDENT | IDENT '(' args ')' | '(' expr ')'
//! args      := expr (',' expr)*
//! ```
//!
//! Operator precedence (ascending):
//!
//! | Level | Operators          | Associativity |
//! |-------|--------------------|---------------|
//! | 1     | `\|\|`             | left          |
//! | 2     | `&&`               | left          |
//! | 3     | `==`, `!=`         | left          |
//! | 4     | `>`, `<`, `>=`,`<=`| left          |
//! | 5     | `+`, `-`           | left          |
//! | 6     | `*`, `/`           | left          |
//! | 7     | unary `-`, `!`     | prefix        |
//! | 8     | `^`                | right         |

use proc_macro2::Span;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitInt, Token};

// ═══════════════════════════════════════════════════════════════════════════
// AST
// ═══════════════════════════════════════════════════════════════════════════

/// A node in the math expression AST.
#[derive(Debug, Clone)]
pub enum MathExpr {
    /// Integer literal: `0`, `1`, `42`.
    Int(i64, Span),

    /// Identifier: `x`, `y`, `w_` (wilds end in `_`).
    Ident(Ident),

    /// Binary operation.
    BinOp {
        op: BinOp,
        lhs: Box<MathExpr>,
        rhs: Box<MathExpr>,
    },

    /// Unary negation: `-expr`.
    Neg(Box<MathExpr>),

    /// Unary logical NOT: `!expr`.
    LogicalNot(Box<MathExpr>),

    /// Function call: `sin(x)`, `cos(x + 1)`, etc.
    Func {
        name: String,
        span: Span,
        args: Vec<MathExpr>,
    },
}

/// Binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    // Comparison operators
    Gt,
    Lt,
    Ge,
    Le,
    EqEq,
    Ne,
    // Logical operators
    AndAnd,
    OrOr,
}

// ═══════════════════════════════════════════════════════════════════════════
// Known functions
// ═══════════════════════════════════════════════════════════════════════════

/// The set of built-in function names recognised by the parser.
pub const KNOWN_FUNCTIONS: &[&str] = &[
    // Original trig
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    // Exp/log
    "exp",
    "ln",
    "sqrt",
    "cbrt",
    "abs",
    "sign",
    // Wave A: reciprocal trig/hyp
    "sec",
    "csc",
    "cot",
    "acot",
    "asec",
    "acsc",
    "coth",
    "sech",
    "csch",
    "acoth",
    "asech",
    "acsch",
    "sinc",
    // Wave O: complex
    "arg",
    "conjugate",
    // Wave R: combinatorial (1-arg)
    "fibonacci",
    "lucas",
    "catalan_number",
    "bell",
    "euler_number",
    "harmonic",
    "subfactorial",
    "factorial2",
    "bernoulli_number",
    // Wave S: special elementary
    "heaviside",
    "dirac_delta",
    "lambertw",
    // Wave J: special functions
    "gamma",
    "log_gamma",
    "digamma",
    "erf",
    "erfc",
];

/// Returns `true` if `name` is a known built-in function.
pub fn is_known_function(name: &str) -> bool {
    KNOWN_FUNCTIONS.contains(&name)
}

// ═══════════════════════════════════════════════════════════════════════════
// Known constants (for rule! macro)
// ═══════════════════════════════════════════════════════════════════════════

/// The set of known constant names for the `rule!` macro.
pub const KNOWN_CONSTANTS: &[&str] = &["pi", "E", "I", "oo", "nan", "zoo"];

/// Returns `true` if `name` is a known constant.
pub fn is_known_constant(name: &str) -> bool {
    KNOWN_CONSTANTS.contains(&name)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

impl MathExpr {
    /// Returns `true` if this expression is an integer literal.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            MathExpr::Int(n, _) => Some(*n),
            _ => None,
        }
    }

    /// Returns `true` if this is an identifier ending in `_`.
    #[cfg(test)]
    pub fn is_wild(&self) -> bool {
        match self {
            MathExpr::Ident(id) => id.to_string().ends_with('_'),
            _ => false,
        }
    }

    /// Collect all unique wild identifiers in this expression.
    pub fn collect_wilds(&self) -> Vec<Ident> {
        let mut wilds = Vec::new();
        self.collect_wilds_inner(&mut wilds);
        wilds
    }

    fn collect_wilds_inner(&self, wilds: &mut Vec<Ident>) {
        match self {
            MathExpr::Ident(id) if id.to_string().ends_with('_') => {
                if !wilds.iter().any(|w| w == id) {
                    wilds.push(id.clone());
                }
            }
            MathExpr::BinOp { lhs, rhs, .. } => {
                lhs.collect_wilds_inner(wilds);
                rhs.collect_wilds_inner(wilds);
            }
            MathExpr::Neg(inner) => inner.collect_wilds_inner(wilds),
            MathExpr::LogicalNot(inner) => inner.collect_wilds_inner(wilds),
            MathExpr::Func { args, .. } => {
                for arg in args {
                    arg.collect_wilds_inner(wilds);
                }
            }
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pratt parser
// ═══════════════════════════════════════════════════════════════════════════

/// Binding power for each precedence level.
///
/// A higher number means tighter binding.  For left-associative ops,
/// `left_bp` < `right_bp`.  For right-associative ops (like `^`),
/// `left_bp` > `right_bp` (or equal, with a different check).
///
/// Precedence levels (ascending):
///   1: ||            (left-assoc)
///   2: &&            (left-assoc)
///   3: == !=         (left-assoc)
///   4: > < >= <=     (left-assoc)
///   5: + -           (left-assoc)
///   6: * /           (left-assoc)
///   7: unary - !     (prefix)
///   8: ^             (right-assoc)
fn infix_bp(op: BinOp) -> (u8, u8) {
    match op {
        BinOp::OrOr => (1, 2),                                   // level 1, left-assoc
        BinOp::AndAnd => (3, 4),                                 // level 2, left-assoc
        BinOp::EqEq | BinOp::Ne => (5, 6),                       // level 3, left-assoc
        BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le => (7, 8), // level 4, left-assoc
        BinOp::Add | BinOp::Sub => (9, 10),                      // level 5, left-assoc
        BinOp::Mul | BinOp::Div => (11, 12),                     // level 6, left-assoc
        BinOp::Pow => (16, 15),                                  // level 8, right-assoc
    }
}

/// Prefix binding power for unary minus and logical NOT.
fn prefix_bp() -> u8 {
    13 // between Mul/Div (11,12) and Pow (15,16)
}

/// Parse a math expression from a `syn::ParseStream`.
///
/// This is the entry point.  It parses the entire remaining input as
/// a math expression with standard operator precedence.
pub fn parse_math_expr(input: ParseStream) -> syn::Result<MathExpr> {
    parse_expr_bp(input, 0)
}

/// Parse an expression with a minimum binding power.
///
/// This is the core of the Pratt parser.
fn parse_expr_bp(input: ParseStream, min_bp: u8) -> syn::Result<MathExpr> {
    // Parse the left-hand side (prefix or primary).
    let mut lhs = parse_prefix(input)?;

    // Parse infix operators as long as they bind tightly enough.
    loop {
        // Peek at the next token to see if it's a binary operator.
        let op = match peek_binop(input) {
            Some(op) => op,
            None => break, // No more infix operators — done.
        };

        let (left_bp, right_bp) = infix_bp(op);
        if left_bp < min_bp {
            break; // This operator doesn't bind tightly enough.
        }

        // Consume the operator token(s).
        consume_binop(input, op)?;

        // Parse the right-hand side with the right binding power.
        let rhs = parse_expr_bp(input, right_bp)?;

        lhs = MathExpr::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
    }

    Ok(lhs)
}

/// Parse a prefix expression (unary minus, logical NOT, or a primary).
fn parse_prefix(input: ParseStream) -> syn::Result<MathExpr> {
    if input.peek(Token![-]) {
        let _: Token![-] = input.parse()?;
        let operand = parse_expr_bp(input, prefix_bp())?;
        Ok(MathExpr::Neg(Box::new(operand)))
    } else if input.peek(Token![!]) {
        let _: Token![!] = input.parse()?;
        let operand = parse_expr_bp(input, prefix_bp())?;
        Ok(MathExpr::LogicalNot(Box::new(operand)))
    } else {
        parse_primary(input)
    }
}

/// Parse a primary expression: integer, identifier, function call,
/// or parenthesised expression.
fn parse_primary(input: ParseStream) -> syn::Result<MathExpr> {
    if input.peek(syn::token::Paren) {
        // Parenthesised expression.
        let content;
        syn::parenthesized!(content in input);
        parse_math_expr(&content)
    } else if input.peek(LitInt) {
        // Integer literal.
        let lit: LitInt = input.parse()?;
        let value: i64 = lit.base10_parse()?;
        Ok(MathExpr::Int(value, lit.span()))
    } else if input.peek(Ident) {
        let ident: Ident = input.parse()?;
        let name = ident.to_string();

        // Check if this is a function call: IDENT '(' ... ')'.
        if input.peek(syn::token::Paren) {
            // Function call.
            let content;
            syn::parenthesized!(content in input);
            let args = parse_arg_list(&content)?;
            Ok(MathExpr::Func {
                name,
                span: ident.span(),
                args,
            })
        } else {
            // Plain identifier.
            Ok(MathExpr::Ident(ident))
        }
    } else {
        Err(input.error("expected integer, identifier, function call, or '('"))
    }
}

/// Parse a comma-separated list of arguments.
fn parse_arg_list(input: ParseStream) -> syn::Result<Vec<MathExpr>> {
    let mut args = Vec::new();
    if input.is_empty() {
        return Ok(args);
    }
    args.push(parse_math_expr(input)?);
    while input.peek(Token![,]) {
        let _: Token![,] = input.parse()?;
        args.push(parse_math_expr(input)?);
    }
    Ok(args)
}

/// Peek at the next token to see if it's a binary operator.
/// Returns `None` if it's not.
///
/// Two-character operators (`>=`, `<=`, `==`, `!=`, `&&`, `||`) are
/// checked before their single-character prefixes (`>`, `<`, `=`, etc.)
/// so that `>=` is not misread as `>` followed by `=`.
fn peek_binop(input: ParseStream) -> Option<BinOp> {
    // Two-character operators first
    if input.peek(Token![&&]) {
        return Some(BinOp::AndAnd);
    }
    if input.peek(Token![||]) {
        return Some(BinOp::OrOr);
    }
    if input.peek(Token![>=]) {
        return Some(BinOp::Ge);
    }
    if input.peek(Token![<=]) {
        return Some(BinOp::Le);
    }
    if input.peek(Token![==]) {
        return Some(BinOp::EqEq);
    }
    if input.peek(Token![!=]) {
        return Some(BinOp::Ne);
    }
    // Single-character operators
    if input.peek(Token![>]) {
        return Some(BinOp::Gt);
    }
    if input.peek(Token![<]) {
        return Some(BinOp::Lt);
    }
    if input.peek(Token![+]) {
        Some(BinOp::Add)
    } else if input.peek(Token![-]) {
        Some(BinOp::Sub)
    } else if input.peek(Token![*]) {
        Some(BinOp::Mul)
    } else if input.peek(Token![/]) {
        Some(BinOp::Div)
    } else if input.peek(Token![^]) {
        Some(BinOp::Pow)
    } else {
        None
    }
}

/// Consume the token(s) for a binary operator.
fn consume_binop(input: ParseStream, op: BinOp) -> syn::Result<()> {
    match op {
        BinOp::Add => {
            let _: Token![+] = input.parse()?;
        }
        BinOp::Sub => {
            let _: Token![-] = input.parse()?;
        }
        BinOp::Mul => {
            let _: Token![*] = input.parse()?;
        }
        BinOp::Div => {
            let _: Token![/] = input.parse()?;
        }
        BinOp::Pow => {
            let _: Token![^] = input.parse()?;
        }
        BinOp::Gt => {
            let _: Token![>] = input.parse()?;
        }
        BinOp::Lt => {
            let _: Token![<] = input.parse()?;
        }
        BinOp::Ge => {
            let _: Token![>=] = input.parse()?;
        }
        BinOp::Le => {
            let _: Token![<=] = input.parse()?;
        }
        BinOp::EqEq => {
            let _: Token![==] = input.parse()?;
        }
        BinOp::Ne => {
            let _: Token![!=] = input.parse()?;
        }
        BinOp::AndAnd => {
            let _: Token![&&] = input.parse()?;
        }
        BinOp::OrOr => {
            let _: Token![||] = input.parse()?;
        }
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Top-level parse wrappers (for use from proc macro entry points)
// ═══════════════════════════════════════════════════════════════════════════

/// Input for the `expr!` macro: just a math expression.
pub struct ExprMacroInput {
    pub expr: MathExpr,
}

impl Parse for ExprMacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let expr = parse_math_expr(input)?;
        Ok(ExprMacroInput { expr })
    }
}

/// Input for the `rule!` macro:
/// `arena, "name", LHS => RHS`
pub struct RuleMacroInput {
    pub arena: Ident,
    pub name: syn::LitStr,
    pub lhs: MathExpr,
    pub rhs: MathExpr,
    pub condition: Option<syn::Expr>,
}

impl Parse for RuleMacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let arena: Ident = input.parse()?;
        let _: Token![,] = input.parse()?;
        let name: syn::LitStr = input.parse()?;
        let _: Token![,] = input.parse()?;

        // Parse LHS until we see `=>`
        let lhs = parse_math_expr(input)?;

        let _: Token![=>] = input.parse()?;

        // Parse RHS (rest of input).
        let rhs = parse_math_expr(input)?;

        let condition = if input.peek(Token![if]) {
            input.parse::<Token![if]>()?;
            Some(input.parse::<syn::Expr>()?)
        } else {
            None
        };

        Ok(RuleMacroInput {
            arena,
            name,
            lhs,
            rhs,
            condition,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix and Equation macro inputs
// ═══════════════════════════════════════════════════════════════════════════

/// Input for the `matrix!` macro: `[[expr, expr], [expr, expr]]`.
pub struct MatrixMacroInput {
    pub rows: Vec<Vec<MathExpr>>,
}

impl Parse for MatrixMacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // When invoked as `matrix![[r0], [r1]]`, the outer `[]` is the
        // macro-invocation delimiter so the input stream is `[r0], [r1]`.
        // We therefore iterate directly over bracket-delimited rows
        // without consuming an additional outer bracket group.

        let mut rows = Vec::new();
        while !input.is_empty() {
            let row_content;
            syn::bracketed!(row_content in input);

            let mut row = Vec::new();
            loop {
                row.push(parse_math_expr(&row_content)?);
                if row_content.is_empty() {
                    break;
                }
                row_content.parse::<Token![,]>()?;
                if row_content.is_empty() {
                    break; // trailing comma
                }
            }
            rows.push(row);

            if input.is_empty() {
                break;
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        // Validate: all rows must have the same length
        if let Some(first_len) = rows.first().map(|r| r.len()) {
            for (i, row) in rows.iter().enumerate() {
                if row.len() != first_len {
                    return Err(syn::Error::new(
                        Span::call_site(),
                        format!(
                            "matrix! row {} has {} columns, but row 0 has {} columns",
                            i,
                            row.len(),
                            first_len
                        ),
                    ));
                }
            }
        }

        Ok(MatrixMacroInput { rows })
    }
}

/// Input for the `eq!` macro: `LHS = RHS`.
pub struct EqMacroInput {
    pub lhs: MathExpr,
    pub rhs: MathExpr,
}

impl Parse for EqMacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lhs = parse_math_expr(input)?;
        input.parse::<Token![=]>()?;
        let rhs = parse_math_expr(input)?;
        Ok(EqMacroInput { lhs, rhs })
    }
}
