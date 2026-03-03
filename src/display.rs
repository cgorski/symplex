//! Pretty-printing for symbolic expressions.
//!
//! This module implements human-readable formatting of expression trees stored
//! in an [`Arena`].  The core entry point is [`fmt_expr`], which recursively
//! walks the DAG and writes a string representation to a [`std::fmt::Formatter`].
//!
//! # Precedence
//!
//! Parentheses are inserted only when necessary, governed by a simple numeric
//! precedence scheme.  Each expression kind has an associated precedence level;
//! when a sub-expression's precedence is strictly less than its parent's, it is
//! wrapped in parentheses.
//!
//! # Example
//!
//! ```text
//! let a = Arena::new();
//! // ... build some expression ...
//! println!("{}", a.display(expr_id));
//! ```

use std::fmt;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed};

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};

// ---------------------------------------------------------------------------
// Precedence constants
// ---------------------------------------------------------------------------

/// Precedence for atoms — never need parentheses.
pub(crate) const PREC_ATOM: u8 = 100;

/// Precedence for exponentiation (`**`).
pub(crate) const PREC_POW: u8 = 80;

/// Precedence for unary negation and function application.
pub(crate) const PREC_UNARY: u8 = 70;

/// Precedence for multiplication.
pub(crate) const PREC_MUL: u8 = 60;

/// Precedence for addition / subtraction.
pub(crate) const PREC_ADD: u8 = 40;

// ---------------------------------------------------------------------------
// Helper: precedence of a node
// ---------------------------------------------------------------------------

/// Returns the precedence level of the given expression node.
fn prec_of(node: &ExprNode) -> u8 {
    match node {
        // Atoms
        ExprNode::Num(_)
        | ExprNode::Symbol(_)
        | ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
        | ExprNode::Infinity
        | ExprNode::NegInfinity
        | ExprNode::ComplexInfinity
        | ExprNode::NaN => PREC_ATOM,

        // Operators
        ExprNode::Add(_) => PREC_ADD,
        ExprNode::Mul(_) => PREC_MUL,
        ExprNode::Pow(_, _) => PREC_POW,
        ExprNode::Neg(_) => PREC_UNARY,

        // Functions — their argument is always inside `(…)`, so the node
        // itself behaves like an atom from a parenthesization standpoint.
        ExprNode::Sin(_)
        | ExprNode::Cos(_)
        | ExprNode::Tan(_)
        | ExprNode::Exp(_)
        | ExprNode::Ln(_)
        | ExprNode::Sqrt(_)
        | ExprNode::Abs(_)
        | ExprNode::Apply(_, _) => PREC_ATOM,

        // Calculus forms — rendered with explicit delimiters, so atom-like.
        ExprNode::Derivative(_, _) | ExprNode::Integral(_, _) => PREC_ATOM,
    }
}

// ---------------------------------------------------------------------------
// Helper: format a rational number
// ---------------------------------------------------------------------------

/// Writes a `Ratio<BigInt>` to `f`.
///
/// * If the denominator is 1, prints just the integer part (e.g. `3`, `-7`).
/// * Otherwise prints `p/q` (e.g. `1/2`, `-5/7`).
fn fmt_ratio(f: &mut fmt::Formatter<'_>, r: &Ratio<BigInt>) -> fmt::Result {
    if r.denom().is_one() {
        write!(f, "{}", r.numer())
    } else {
        write!(f, "{}/{}", r.numer(), r.denom())
    }
}

// ---------------------------------------------------------------------------
// Core recursive formatter
// ---------------------------------------------------------------------------

/// Recursively formats the expression identified by `id`, writing to `f`.
///
/// `parent_prec` is the precedence of the context in which this expression
/// appears.  If the current expression's precedence is strictly less than
/// `parent_prec`, the output is wrapped in parentheses.
///
/// # Panics
///
/// Panics if `id` refers to a node that does not exist in `arena`.
pub(crate) fn fmt_expr(
    arena: &Arena,
    f: &mut fmt::Formatter<'_>,
    id: ExprId,
    parent_prec: u8,
) -> fmt::Result {
    let node = arena.node(id);
    let my_prec = prec_of(node);
    let need_parens = my_prec < parent_prec;

    if need_parens {
        write!(f, "(")?;
    }

    match node {
        // -- atoms -----------------------------------------------------------
        ExprNode::Num(num_id) => {
            let r = arena.num(*num_id);
            fmt_ratio(f, r)?;
        }

        ExprNode::Symbol(sym_id) => {
            write!(f, "{}", arena.symbol_name(*sym_id))?;
        }

        ExprNode::Pi => write!(f, "pi")?,
        ExprNode::E => write!(f, "E")?,
        ExprNode::ImaginaryUnit => write!(f, "I")?,
        ExprNode::Infinity => write!(f, "oo")?,
        ExprNode::NegInfinity => write!(f, "-oo")?,
        ExprNode::ComplexInfinity => write!(f, "zoo")?,
        ExprNode::NaN => write!(f, "nan")?,

        // -- Add -------------------------------------------------------------
        //
        // Join terms with ` + `.  If a term is `Neg(x)`, display ` - x`
        // instead of ` + -x`.  The first negative term is shown as `-x`
        // (no leading ` + `).
        ExprNode::Add(args) => {
            let args = args.clone();
            for (i, &arg) in args.iter().enumerate() {
                let child_node = arena.node(arg);

                // Check for Neg(inner) — canonical only for raw nodes
                if let ExprNode::Neg(inner) = child_node {
                    let inner = *inner;
                    if i == 0 {
                        write!(f, "-")?;
                        fmt_expr(arena, f, inner, PREC_UNARY)?;
                    } else {
                        write!(f, " - ")?;
                        fmt_expr(arena, f, inner, PREC_UNARY)?;
                    }
                }
                // Check for Mul([-1, rest...]) — canonical form of negation
                else if let ExprNode::Mul(mul_args) = child_node {
                    let is_neg_one_mul = if !mul_args.is_empty() {
                        if let ExprNode::Num(nid) = arena.node(mul_args[0]) {
                            let r = arena.num(*nid);
                            r == &Ratio::from(BigInt::from(-1))
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if is_neg_one_mul && mul_args.len() >= 2 {
                        // Display as " - rest" where rest is the factors without -1
                        if i == 0 {
                            write!(f, "-")?;
                        } else {
                            write!(f, " - ")?;
                        }
                        // Print the remaining factors (skip the -1 coefficient)
                        fmt_mul_factors(arena, f, &mul_args[1..])?;
                    } else if i == 0 {
                        fmt_expr(arena, f, arg, PREC_ADD)?;
                    } else {
                        write!(f, " + ")?;
                        fmt_expr(arena, f, arg, PREC_ADD)?;
                    }
                } else if i == 0 {
                    fmt_expr(arena, f, arg, PREC_ADD)?;
                } else {
                    write!(f, " + ")?;
                    fmt_expr(arena, f, arg, PREC_ADD)?;
                }
            }
        }

        // -- Mul -------------------------------------------------------------
        //
        // Join factors with `*`.  Special cases:
        //   - Leading `1` coefficient is omitted (unless it's the only factor).
        //   - Leading `-1` coefficient displays as `-` prefix.
        ExprNode::Mul(args) => {
            let args = args.clone();
            if args.is_empty() {
                // Degenerate empty product — shouldn't occur in practice.
                write!(f, "1")?;
            } else {
                let first_node = arena.node(args[0]);
                let is_neg_one = if let ExprNode::Num(nid) = first_node {
                    let r = arena.num(*nid);
                    r == &Ratio::from(BigInt::from(-1))
                } else {
                    false
                };
                let is_pos_one = if let ExprNode::Num(nid) = first_node {
                    let r = arena.num(*nid);
                    r == &Ratio::from(BigInt::from(1))
                } else {
                    false
                };

                if args.len() == 1 {
                    // Single factor — just print it.
                    fmt_expr(arena, f, args[0], PREC_MUL)?;
                } else if is_neg_one {
                    // `-1 * rest` → `-rest`
                    write!(f, "-")?;
                    fmt_mul_factors(arena, f, &args[1..])?;
                } else if is_pos_one {
                    // `1 * rest` → `rest`
                    fmt_mul_factors(arena, f, &args[1..])?;
                } else {
                    fmt_mul_factors(arena, f, &args)?;
                }
            }
        }

        // -- Pow -------------------------------------------------------------
        //
        // Display as `base**exp`.
        //
        // The base needs parentheses if it is Add, Mul, Neg, or another Pow.
        // The exponent needs parentheses if it is Add, Mul, or Neg.
        ExprNode::Pow(base, exp) => {
            let base = *base;
            let exp = *exp;

            // Determine required parenthesization for the base.
            let base_prec = {
                let bn = arena.node(base);
                match bn {
                    ExprNode::Add(_)
                    | ExprNode::Mul(_)
                    | ExprNode::Neg(_)
                    | ExprNode::Pow(_, _) => PREC_POW + 1,
                    _ => PREC_POW,
                }
            };
            fmt_expr(arena, f, base, base_prec)?;

            write!(f, "**")?;

            // Determine required parenthesization for the exponent.
            let exp_prec = {
                let en = arena.node(exp);
                match en {
                    ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => PREC_POW + 1,
                    // Negative numbers or non-integer rationals need parens.
                    // Num has PREC_ATOM (100), so we must exceed that to force parens.
                    ExprNode::Num(nid) => {
                        let r = arena.num(*nid);
                        if r.is_negative() || !r.denom().is_one() {
                            PREC_ATOM + 1
                        } else {
                            PREC_POW
                        }
                    }
                    _ => PREC_POW,
                }
            };
            fmt_expr(arena, f, exp, exp_prec)?;
        }

        // -- Neg -------------------------------------------------------------
        ExprNode::Neg(x) => {
            let x = *x;
            write!(f, "-")?;
            // Wrap the operand if it's an Add (so `-a + b` doesn't become
            // ambiguous).  Other cases (Mul, Pow, atoms) are fine without
            // extra parens at unary precedence.
            let child_prec = match arena.node(x) {
                ExprNode::Add(_) => PREC_UNARY + 1,
                _ => PREC_UNARY,
            };
            fmt_expr(arena, f, x, child_prec)?;
        }

        // -- Built-in functions ----------------------------------------------
        //
        // Display as `name(x)`.  The argument is inside function-call parens
        // so it never needs extra parenthesization (we pass `0`).
        ExprNode::Sin(x) => {
            let x = *x;
            write!(f, "sin(")?;
            fmt_expr(arena, f, x, 0)?;
            write!(f, ")")?;
        }

        ExprNode::Cos(x) => {
            let x = *x;
            write!(f, "cos(")?;
            fmt_expr(arena, f, x, 0)?;
            write!(f, ")")?;
        }

        ExprNode::Tan(x) => {
            let x = *x;
            write!(f, "tan(")?;
            fmt_expr(arena, f, x, 0)?;
            write!(f, ")")?;
        }

        ExprNode::Exp(x) => {
            let x = *x;
            write!(f, "exp(")?;
            fmt_expr(arena, f, x, 0)?;
            write!(f, ")")?;
        }

        ExprNode::Ln(x) => {
            let x = *x;
            write!(f, "ln(")?;
            fmt_expr(arena, f, x, 0)?;
            write!(f, ")")?;
        }

        ExprNode::Sqrt(x) => {
            let x = *x;
            write!(f, "sqrt(")?;
            fmt_expr(arena, f, x, 0)?;
            write!(f, ")")?;
        }

        ExprNode::Abs(x) => {
            let x = *x;
            write!(f, "abs(")?;
            fmt_expr(arena, f, x, 0)?;
            write!(f, ")")?;
        }

        // -- Apply -----------------------------------------------------------
        //
        // User-defined / library function: `func(arg1, arg2, ...)`.
        ExprNode::Apply(sym_id, args) => {
            let sym_id = *sym_id;
            let args = args.clone();
            write!(f, "{}(", arena.symbol_name(sym_id))?;
            for (i, &arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_expr(arena, f, arg, 0)?;
            }
            write!(f, ")")?;
        }

        // -- Calculus forms --------------------------------------------------
        ExprNode::Derivative(body, var) => {
            let body = *body;
            let var = *var;
            write!(f, "Derivative(")?;
            fmt_expr(arena, f, body, 0)?;
            write!(f, ", ")?;
            fmt_expr(arena, f, var, 0)?;
            write!(f, ")")?;
        }

        ExprNode::Integral(body, var) => {
            let body = *body;
            let var = *var;
            write!(f, "Integral(")?;
            fmt_expr(arena, f, body, 0)?;
            write!(f, ", ")?;
            fmt_expr(arena, f, var, 0)?;
            write!(f, ")")?;
        }
    }

    if need_parens {
        write!(f, ")")?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: format a slice of multiplicative factors joined by `*`
// ---------------------------------------------------------------------------

/// Writes a sequence of factors joined by `*`.
///
/// Each factor is parenthesized according to the standard precedence rule
/// (i.e. wrapped when its precedence is strictly less than [`PREC_MUL`]).
fn fmt_mul_factors(arena: &Arena, f: &mut fmt::Formatter<'_>, factors: &[ExprId]) -> fmt::Result {
    for (i, &fid) in factors.iter().enumerate() {
        if i > 0 {
            write!(f, "*")?;
        }
        fmt_expr(arena, f, fid, PREC_MUL)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Arena::display
// ---------------------------------------------------------------------------

impl Arena {
    /// Returns a displayable wrapper for the expression identified by `id`.
    ///
    /// The returned value implements [`std::fmt::Display`] and can be used
    /// directly with `format!`, `println!`, `write!`, etc.
    ///
    /// # Example
    ///
    /// ```text
    /// let arena = Arena::new();
    /// let x = arena.symbol("x");
    /// println!("{}", arena.display(x));
    /// ```
    ///
    /// Uses [`std::fmt::from_fn`] (stabilised in Rust 1.93) to avoid a
    /// separate wrapper struct.
    pub fn display(&self, id: ExprId) -> impl fmt::Display + '_ {
        fmt::from_fn(move |f| fmt_expr(self, f, id, 0))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Shorthand: build an arena, construct an expression, and format it.
    macro_rules! assert_display {
        ($arena:expr, $id:expr, $expected:expr) => {
            assert_eq!(format!("{}", $arena.display($id)), $expected);
        };
    }

    // -- atoms ---------------------------------------------------------------

    #[test]
    fn display_integer() {
        let mut a = Arena::new();
        let n = a.int(42);
        assert_display!(a, n, "42");
    }

    #[test]
    fn display_negative_integer() {
        let mut a = Arena::new();
        let n = a.int(-3);
        assert_display!(a, n, "-3");
    }

    #[test]
    fn display_zero() {
        let mut a = Arena::new();
        let z = a.int(0);
        assert_display!(a, z, "0");
    }

    #[test]
    fn display_rational() {
        let mut a = Arena::new();
        let r = a.rational(1, 2);
        assert_display!(a, r, "1/2");
    }

    #[test]
    fn display_negative_rational() {
        let mut a = Arena::new();
        let r = a.rational(-5, 7);
        assert_display!(a, r, "-5/7");
    }

    #[test]
    fn display_symbol() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        assert_display!(a, x, "x");
    }

    #[test]
    fn display_constants() {
        let a = Arena::new();
        assert_display!(a, a.pi, "pi");
        assert_display!(a, a.e_const, "E");
        assert_display!(a, a.i_unit, "I");
    }

    #[test]
    fn display_special_values() {
        let a = Arena::new();
        assert_display!(a, a.infinity, "oo");
        assert_display!(a, a.neg_infinity, "-oo");
        assert_display!(a, a.nan, "nan");
    }

    // -- Add -----------------------------------------------------------------

    #[test]
    fn display_add_two() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let sum = a.add(&[x, y]);
        assert_display!(a, sum, "x + y");
    }

    #[test]
    fn display_add_with_neg_term() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let neg_y = a.neg(y);
        let expr = a.add(&[x, neg_y]);
        // canon_neg turns neg(y) into Mul(-1, y); display detects this as subtraction.
        assert_display!(a, expr, "x - y");
    }

    #[test]
    fn display_add_leading_neg() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let neg_x = a.neg(x);
        let expr = a.add(&[neg_x, y]);
        // Canonical Add sorts: y (symbol, rank 10) before Mul(-1, x) (Mul, rank 30).
        assert_display!(a, expr, "y - x");
    }

    // -- Mul -----------------------------------------------------------------

    #[test]
    fn display_mul_two() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let prod = a.mul(&[x, y]);
        assert_display!(a, prod, "x*y");
    }

    #[test]
    fn display_mul_neg_one_coefficient() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let neg1 = a.int(-1);
        let prod = a.mul(&[neg1, x]);
        assert_display!(a, prod, "-x");
    }

    #[test]
    fn display_mul_one_coefficient() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let one = a.int(1);
        let prod = a.mul(&[one, x]);
        assert_display!(a, prod, "x");
    }

    // -- Pow -----------------------------------------------------------------

    #[test]
    fn display_pow() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let n = a.int(2);
        let p = a.pow(x, n);
        assert_display!(a, p, "x**2");
    }

    #[test]
    fn display_pow_add_base_gets_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let sum = a.add(&[x, y]);
        let n = a.int(2);
        let p = a.pow(sum, n);
        assert_display!(a, p, "(x + y)**2");
    }

    #[test]
    fn display_pow_add_exp_gets_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let a_sym = a.symbol("a");
        let b_sym = a.symbol("b");
        let sum = a.add(&[a_sym, b_sym]);
        let p = a.pow(x, sum);
        assert_display!(a, p, "x**(a + b)");
    }

    // -- Neg -----------------------------------------------------------------

    #[test]
    fn display_neg_symbol() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let neg = a.neg(x);
        assert_display!(a, neg, "-x");
    }

    #[test]
    fn display_neg_add_gets_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let sum = a.add(&[x, y]);
        let neg = a.neg(sum);
        // canon_neg distributes over Add: -(x + y) → Add(Mul(-1,x), Mul(-1,y))
        // display detects Mul(-1, ...) as subtraction notation.
        assert_display!(a, neg, "-x - y");
    }

    // -- Functions -----------------------------------------------------------

    #[test]
    fn display_sin() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let s = a.sin(x);
        assert_display!(a, s, "sin(x)");
    }

    #[test]
    fn display_cos() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let c = a.cos(x);
        assert_display!(a, c, "cos(x)");
    }

    #[test]
    fn display_nested_function() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let s = a.sin(x);
        let c = a.cos(s);
        assert_display!(a, c, "cos(sin(x))");
    }

    // -- Derivative / Integral -----------------------------------------------

    #[test]
    fn display_derivative() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let s = a.sin(x);
        let d = a.intern(ExprNode::Derivative(s, x));
        assert_display!(a, d, "Derivative(sin(x), x)");
    }

    #[test]
    fn display_integral() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let s = a.sin(x);
        let i = a.intern(ExprNode::Integral(s, x));
        assert_display!(a, i, "Integral(sin(x), x)");
    }

    // -- Composite expressions -----------------------------------------------

    #[test]
    fn display_mul_in_add_no_extra_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let z = a.symbol("z");
        let xy = a.mul(&[x, y]);
        let sum = a.add(&[xy, z]);
        // Canonical Add sorts: z (symbol, rank 10) before x*y (Mul, rank 30).
        assert_display!(a, sum, "z + x*y");
    }

    #[test]
    fn display_add_in_mul_gets_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let z = a.symbol("z");
        let sum = a.add(&[x, y]);
        let prod = a.mul(&[sum, z]);
        // Canonical Mul sorts: z (symbol, rank 10) before (x+y) (Add, rank 40).
        assert_display!(a, prod, "z*(x + y)");
    }
}
