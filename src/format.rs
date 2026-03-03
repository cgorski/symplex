//! Expression formatting for multiple output modes.
//!
//! This module provides [`PrintMode`] and [`PrintOptions`] for rendering
//! expressions as plain text, LaTeX, or Markdown. The core function
//! [`format_expr`] uses exhaustive matching on both [`ExprNode`] and
//! [`PrintMode`] to guarantee compile-time completeness.

use std::fmt::Write;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Signed;

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};

/// Output format modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintMode {
    /// Plain text: `x^2 + 2*x`, `sin(x)`, `1/x`
    PlainText,
    /// LaTeX: `x^{2} + 2x`, `\sin(x)`, `\frac{1}{x}`
    LaTeX,
    /// Markdown with Unicode: `x² + 2x`, `sin(x)`, `1/x`
    Markdown,
}

/// Configuration for expression formatting.
#[derive(Debug, Clone)]
pub struct PrintOptions {
    /// The output format mode.
    pub mode: PrintMode,
}

impl Default for PrintOptions {
    fn default() -> Self {
        PrintOptions {
            mode: PrintMode::PlainText,
        }
    }
}

impl PrintOptions {
    /// Create options for LaTeX output.
    pub fn latex() -> Self {
        PrintOptions {
            mode: PrintMode::LaTeX,
        }
    }

    /// Create options for Markdown output.
    pub fn markdown() -> Self {
        PrintOptions {
            mode: PrintMode::Markdown,
        }
    }
}

/// Format an expression as a string using the given options.
pub fn format_expr(arena: &Arena, id: ExprId, opts: &PrintOptions) -> String {
    let mut out = String::new();
    write_expr(arena, id, opts, &mut out, 0);
    out
}

// Precedence constants (same as display.rs)
const PREC_ADD: u8 = 40;
const PREC_MUL: u8 = 60;
const PREC_POW: u8 = 80;
#[allow(dead_code)]
const PREC_ATOM: u8 = 100;

fn write_expr(arena: &Arena, id: ExprId, opts: &PrintOptions, out: &mut String, parent_prec: u8) {
    let node = arena.node(id).clone();
    match node {
        ExprNode::Num(nid) => {
            let r = arena.num(nid);
            if r.denom() == &BigInt::from(1) {
                write!(out, "{}", r.numer()).unwrap();
            } else {
                match opts.mode {
                    PrintMode::LaTeX => {
                        write!(out, "\\frac{{{}}}{{{}}}", r.numer(), r.denom()).unwrap();
                    }
                    _ => {
                        write!(out, "{}/{}", r.numer(), r.denom()).unwrap();
                    }
                }
            }
        }

        ExprNode::Symbol(sid) => {
            out.push_str(arena.symbol_name(sid));
        }

        ExprNode::Pi => match opts.mode {
            PrintMode::LaTeX => out.push_str("\\pi"),
            PrintMode::Markdown => out.push_str("π"),
            PrintMode::PlainText => out.push_str("pi"),
        },

        ExprNode::E => match opts.mode {
            PrintMode::LaTeX => out.push_str("e"),
            _ => out.push_str("E"),
        },

        ExprNode::ImaginaryUnit => match opts.mode {
            PrintMode::LaTeX => out.push_str("i"),
            _ => out.push_str("I"),
        },

        ExprNode::Infinity => match opts.mode {
            PrintMode::LaTeX => out.push_str("\\infty"),
            PrintMode::Markdown => out.push_str("∞"),
            PrintMode::PlainText => out.push_str("oo"),
        },

        ExprNode::NegInfinity => match opts.mode {
            PrintMode::LaTeX => out.push_str("-\\infty"),
            PrintMode::Markdown => out.push_str("-∞"),
            PrintMode::PlainText => out.push_str("-oo"),
        },

        ExprNode::ComplexInfinity => match opts.mode {
            PrintMode::LaTeX => out.push_str("\\tilde{\\infty}"),
            PrintMode::Markdown => out.push_str("∞̃"),
            PrintMode::PlainText => out.push_str("zoo"),
        },

        ExprNode::NaN => out.push_str("nan"),

        ExprNode::Add(ref children) => {
            let need_parens = PREC_ADD < parent_prec;
            if need_parens {
                out.push('(');
            }
            for (i, &child) in children.iter().enumerate() {
                if i > 0 {
                    out.push_str(" + ");
                }
                write_expr(arena, child, opts, out, PREC_ADD);
            }
            if need_parens {
                out.push(')');
            }
        }

        ExprNode::Mul(ref children) => {
            let need_parens = PREC_MUL < parent_prec;
            if need_parens {
                out.push('(');
            }
            for (i, &child) in children.iter().enumerate() {
                if i > 0 {
                    match opts.mode {
                        PrintMode::LaTeX => out.push_str(" \\cdot "),
                        _ => out.push('*'),
                    }
                }
                write_expr(arena, child, opts, out, PREC_MUL);
            }
            if need_parens {
                out.push(')');
            }
        }

        ExprNode::Pow(base, exp) => {
            let need_parens = PREC_POW < parent_prec;
            if need_parens {
                out.push('(');
            }
            // Check for x^(-1) → fraction display
            let is_neg_one = if let ExprNode::Num(nid) = arena.node(exp) {
                *arena.num(*nid) == Ratio::from(BigInt::from(-1))
            } else {
                false
            };

            if is_neg_one {
                match opts.mode {
                    PrintMode::LaTeX => {
                        out.push_str("\\frac{1}{");
                        write_expr(arena, base, opts, out, 0);
                        out.push('}');
                    }
                    _ => {
                        out.push_str("1/");
                        write_expr(arena, base, opts, out, PREC_MUL);
                    }
                }
            } else {
                write_expr(arena, base, opts, out, PREC_POW + 1);
                match opts.mode {
                    PrintMode::LaTeX => {
                        out.push_str("^{");
                        write_expr(arena, exp, opts, out, 0);
                        out.push('}');
                    }
                    _ => {
                        out.push('^');
                        // Parens for negative/fractional exponents
                        let exp_needs_parens = match arena.node(exp) {
                            ExprNode::Num(nid) => {
                                let r = arena.num(*nid);
                                r.is_negative() || !r.is_integer()
                            }
                            ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
                            _ => false,
                        };
                        if exp_needs_parens {
                            out.push('(');
                        }
                        write_expr(arena, exp, opts, out, PREC_POW);
                        if exp_needs_parens {
                            out.push(')');
                        }
                    }
                }
            }
            if need_parens {
                out.push(')');
            }
        }

        ExprNode::Neg(inner) => {
            out.push('-');
            write_expr(arena, inner, opts, out, PREC_MUL);
        }

        // All unary functions — exhaustive on function type AND print mode
        ExprNode::Sin(x) => write_func(arena, x, opts, out, parent_prec, "sin", "\\sin"),
        ExprNode::Cos(x) => write_func(arena, x, opts, out, parent_prec, "cos", "\\cos"),
        ExprNode::Tan(x) => write_func(arena, x, opts, out, parent_prec, "tan", "\\tan"),
        ExprNode::Exp(x) => write_func(arena, x, opts, out, parent_prec, "exp", "\\exp"),
        ExprNode::Ln(x) => write_func(arena, x, opts, out, parent_prec, "ln", "\\ln"),
        ExprNode::Sqrt(x) => match opts.mode {
            PrintMode::LaTeX => {
                out.push_str("\\sqrt{");
                write_expr(arena, x, opts, out, 0);
                out.push('}');
            }
            PrintMode::Markdown => {
                out.push_str("√(");
                write_expr(arena, x, opts, out, 0);
                out.push(')');
            }
            PrintMode::PlainText => write_func_plain(arena, x, opts, out, "sqrt"),
        },
        ExprNode::Abs(x) => match opts.mode {
            PrintMode::LaTeX => {
                out.push_str("\\left|");
                write_expr(arena, x, opts, out, 0);
                out.push_str("\\right|");
            }
            _ => {
                out.push_str("abs(");
                write_expr(arena, x, opts, out, 0);
                out.push(')');
            }
        },
        ExprNode::Asin(x) => write_func(arena, x, opts, out, parent_prec, "asin", "\\arcsin"),
        ExprNode::Acos(x) => write_func(arena, x, opts, out, parent_prec, "acos", "\\arccos"),
        ExprNode::Atan(x) => write_func(arena, x, opts, out, parent_prec, "atan", "\\arctan"),
        ExprNode::Sinh(x) => write_func(arena, x, opts, out, parent_prec, "sinh", "\\sinh"),
        ExprNode::Cosh(x) => write_func(arena, x, opts, out, parent_prec, "cosh", "\\cosh"),
        ExprNode::Tanh(x) => write_func(arena, x, opts, out, parent_prec, "tanh", "\\tanh"),

        ExprNode::Apply(sym_id, ref args) => {
            out.push_str(arena.symbol_name(sym_id));
            out.push('(');
            for (i, &arg) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_expr(arena, arg, opts, out, 0);
            }
            out.push(')');
        }

        ExprNode::Derivative(body, var) => match opts.mode {
            PrintMode::LaTeX => {
                out.push_str("\\frac{d}{d");
                write_expr(arena, var, opts, out, 0);
                out.push_str("}\\left(");
                write_expr(arena, body, opts, out, 0);
                out.push_str("\\right)");
            }
            _ => {
                out.push_str("Derivative(");
                write_expr(arena, body, opts, out, 0);
                out.push_str(", ");
                write_expr(arena, var, opts, out, 0);
                out.push(')');
            }
        },

        ExprNode::Integral(body, var) => match opts.mode {
            PrintMode::LaTeX => {
                out.push_str("\\int ");
                write_expr(arena, body, opts, out, 0);
                out.push_str(" \\, d");
                write_expr(arena, var, opts, out, 0);
            }
            _ => {
                out.push_str("Integral(");
                write_expr(arena, body, opts, out, 0);
                out.push_str(", ");
                write_expr(arena, var, opts, out, 0);
                out.push(')');
            }
        },
    }
}

/// Helper: write a standard function call `name(arg)` with mode-aware name.
fn write_func(
    arena: &Arena,
    arg: ExprId,
    opts: &PrintOptions,
    out: &mut String,
    _parent_prec: u8,
    plain_name: &str,
    latex_name: &str,
) {
    match opts.mode {
        PrintMode::LaTeX => {
            out.push_str(latex_name);
            out.push_str("\\left(");
            write_expr(arena, arg, opts, out, 0);
            out.push_str("\\right)");
        }
        _ => write_func_plain(arena, arg, opts, out, plain_name),
    }
}

fn write_func_plain(arena: &Arena, arg: ExprId, opts: &PrintOptions, out: &mut String, name: &str) {
    out.push_str(name);
    out.push('(');
    write_expr(arena, arg, opts, out, 0);
    out.push(')');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    fn fmt(arena: &Arena, id: ExprId, mode: PrintMode) -> String {
        format_expr(arena, id, &PrintOptions { mode })
    }

    #[test]
    fn plain_text_basic() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let expr = a.pow(x, two);
        assert_eq!(fmt(&a, expr, PrintMode::PlainText), "x^2");
    }

    #[test]
    fn latex_power() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let expr = a.pow(x, two);
        assert_eq!(fmt(&a, expr, PrintMode::LaTeX), "x^{2}");
    }

    #[test]
    fn latex_fraction() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let neg_one = a.int(-1);
        let expr = a.pow(x, neg_one);
        assert_eq!(fmt(&a, expr, PrintMode::LaTeX), "\\frac{1}{x}");
    }

    #[test]
    fn latex_sin() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let expr = a.sin(x);
        assert_eq!(fmt(&a, expr, PrintMode::LaTeX), "\\sin\\left(x\\right)");
    }

    #[test]
    fn latex_sqrt() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let expr = a.sqrt(x);
        assert_eq!(fmt(&a, expr, PrintMode::LaTeX), "\\sqrt{x}");
    }

    #[test]
    fn latex_abs() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let expr = a.abs(x);
        assert_eq!(fmt(&a, expr, PrintMode::LaTeX), "\\left|x\\right|");
    }

    #[test]
    fn latex_pi() {
        let a = Arena::new();
        assert_eq!(fmt(&a, a.pi, PrintMode::LaTeX), "\\pi");
    }

    #[test]
    fn latex_derivative() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let expr = a.sin(x);
        let d = a.intern(ExprNode::Derivative(expr, x));
        assert_eq!(
            fmt(&a, d, PrintMode::LaTeX),
            "\\frac{d}{dx}\\left(\\sin\\left(x\\right)\\right)"
        );
    }

    #[test]
    fn markdown_pi() {
        let a = Arena::new();
        assert_eq!(fmt(&a, a.pi, PrintMode::Markdown), "π");
    }

    #[test]
    fn markdown_sqrt() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let expr = a.sqrt(x);
        assert_eq!(fmt(&a, expr, PrintMode::Markdown), "√(x)");
    }

    #[test]
    fn latex_inverse_trig() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let asin_x = a.asin(x);
        let acos_x = a.acos(x);
        let atan_x = a.atan(x);
        assert_eq!(
            fmt(&a, asin_x, PrintMode::LaTeX),
            "\\arcsin\\left(x\\right)"
        );
        assert_eq!(
            fmt(&a, acos_x, PrintMode::LaTeX),
            "\\arccos\\left(x\\right)"
        );
        assert_eq!(
            fmt(&a, atan_x, PrintMode::LaTeX),
            "\\arctan\\left(x\\right)"
        );
    }

    #[test]
    fn latex_hyperbolic() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let sinh_x = a.sinh(x);
        let cosh_x = a.cosh(x);
        let tanh_x = a.tanh(x);
        assert_eq!(fmt(&a, sinh_x, PrintMode::LaTeX), "\\sinh\\left(x\\right)");
        assert_eq!(fmt(&a, cosh_x, PrintMode::LaTeX), "\\cosh\\left(x\\right)");
        assert_eq!(fmt(&a, tanh_x, PrintMode::LaTeX), "\\tanh\\left(x\\right)");
    }

    #[test]
    fn latex_rational() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        assert_eq!(fmt(&a, half, PrintMode::LaTeX), "\\frac{1}{2}");
    }

    #[test]
    fn latex_integral() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let body = a.sin(x);
        let i = a.intern(ExprNode::Integral(body, x));
        assert_eq!(
            fmt(&a, i, PrintMode::LaTeX),
            "\\int \\sin\\left(x\\right) \\, dx"
        );
    }
}
