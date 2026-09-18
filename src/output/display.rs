//! Expression display / pretty-printing.
//!
//! Converts an expression tree into a human-readable string with correct
//! operator precedence and parenthesisation.
//!
//! # Design
//!
//! The formatter uses an **explicit work-stack** rather than recursive
//! function calls.  This guarantees stack safety for arbitrarily deep
//! expression trees (Principle 5: no recursive tree walks).
//!
//! Each entry on the work stack is either a *literal* string to emit, or
//! an *expression id* that still needs to be expanded.  The main loop
//! pops one item at a time, and for expression items it pushes the
//! constituent pieces (open-paren, children, operators, close-paren)
//! back onto the stack in **reverse** order so that they come out
//! left-to-right when popped.
//!
//! Example for display of `Add(x, Neg(y))`:
//!
//! ```text
//! stack (top → bottom):
//!   Expr(Add(x, Neg(y)), parent_prec=0)
//!   ↓ expand Add
//!   Expr(x, PREC_ADD)          ← first term
//!   Lit(" - ")                 ← separator (Neg detected)
//!   Expr(y, PREC_UNARY)       ← inner of Neg
//! ```

use std::fmt;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Signed;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};

use super::common::{display_sort_key, is_neg_coeff_mul, is_neg_one_mul};

// ═══════════════════════════════════════════════════════════════════════════
// Precedence constants
// ═══════════════════════════════════════════════════════════════════════════

/// Atoms never need parentheses.
pub(crate) const PREC_ATOM: u8 = 100;

/// Exponentiation.
pub(crate) const PREC_POW: u8 = 80;

/// Unary minus and function application.
pub(crate) const PREC_UNARY: u8 = 70;

/// Multiplication / division.
pub(crate) const PREC_MUL: u8 = 60;

/// Addition / subtraction (lowest precedence for composed expressions).
pub(crate) const PREC_ADD: u8 = 40;

/// Does a numeric literal need parentheses when it is the base of a power?
///
/// `-2^x` re-parses as `-(2^x)` and `2/3^x` as `2/(3^x)`, so negative and
/// non-integer rationals must be wrapped: `(-2)^x`, `(2/3)^x`.
fn num_base_needs_parens(arena: &Arena, id: ExprId) -> bool {
    if let ExprNode::Num(nid) = arena.node(id) {
        let r = arena.num(*nid);
        r.is_negative() || !r.is_integer()
    } else {
        false
    }
}

/// Return the precedence of a node.
fn prec_of(node: &ExprNode) -> u8 {
    match node {
        ExprNode::Add(_) => PREC_ADD,
        ExprNode::Mul(_) => PREC_MUL,
        ExprNode::Pow(_, _) => PREC_POW,
        ExprNode::Neg(_) => PREC_UNARY,
        // Functions and calculus forms have their arguments inside
        // delimiters (parens), so they behave like atoms w.r.t.
        // outer precedence.
        ExprNode::Sin(_)
        | ExprNode::Cos(_)
        | ExprNode::Tan(_)
        | ExprNode::Exp(_)
        | ExprNode::Ln(_)
        | ExprNode::Abs(_)
        | ExprNode::Asin(_)
        | ExprNode::Acos(_)
        | ExprNode::Atan(_)
        | ExprNode::Atan2(_, _)
        | ExprNode::Sinh(_)
        | ExprNode::Cosh(_)
        | ExprNode::Tanh(_)
        | ExprNode::Asinh(_)
        | ExprNode::Acosh(_)
        | ExprNode::Atanh(_)
        | ExprNode::Gamma(_)
        | ExprNode::LogGamma(_)
        | ExprNode::Digamma(_)
        | ExprNode::Erf(_)
        | ExprNode::Erfc(_)
        | ExprNode::LambertW(_)
        | ExprNode::Beta(_, _)
        | ExprNode::Re(_)
        | ExprNode::Im(_)
        | ExprNode::Conjugate(_)
        | ExprNode::Arg(_)
        | ExprNode::Si(_)
        | ExprNode::Ci(_)
        | ExprNode::Ei(_)
        | ExprNode::Li(_)
        | ExprNode::Zeta(_)
        | ExprNode::Polygamma(_, _)
        | ExprNode::KroneckerDelta(_, _)
        | ExprNode::Floor(_)
        | ExprNode::Ceiling(_)
        | ExprNode::Min(_)
        | ExprNode::Max(_)
        | ExprNode::Sum(_, _, _, _)
        | ExprNode::Product_(_, _, _, _)
        | ExprNode::Apply(_, _)
        | ExprNode::Derivative(_, _)
        | ExprNode::Integral(_, _)
        | ExprNode::DefiniteIntegral(_, _, _, _)
        | ExprNode::Limit(_, _, _)
        | ExprNode::Series(_, _, _, _)
        | ExprNode::LaplaceTransform(_, _, _)
        | ExprNode::InverseLaplaceTransform(_, _, _)
        | ExprNode::Residue(_, _, _)
        | ExprNode::RootOf(_, _)
        | ExprNode::DSolve(_, _, _)
        | ExprNode::RootSum(_, _, _)
        | ExprNode::ConditionSet(_, _) => PREC_ATOM,
        ExprNode::Or(_) => 10,
        ExprNode::And(_) => 15,
        ExprNode::Gt(_, _) | ExprNode::Ge(_, _) | ExprNode::Eq_(_, _) | ExprNode::Ne(_, _) => 20,
        ExprNode::Not(_) => PREC_UNARY,
        ExprNode::Piecewise(_) | ExprNode::BoolTrue | ExprNode::BoolFalse => PREC_ATOM,
        // Everything else is an atom.
        _ => PREC_ATOM,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational number formatting

// ═══════════════════════════════════════════════════════════════════════════
// Work-stack items
// ═══════════════════════════════════════════════════════════════════════════

/// An item on the display work-stack.
enum WorkItem {
    /// A literal string to write verbatim.
    Lit(&'static str),
    /// A dynamically-built string to write.
    Owned(String),
    /// An expression that needs to be expanded, with the parent
    /// precedence that determines whether it needs parentheses.
    Expr(ExprId, u8),
}

// ═══════════════════════════════════════════════════════════════════════════
// Core formatting function (iterative)
// ═══════════════════════════════════════════════════════════════════════════

/// Format an expression rooted at `id` into `f`.
///
/// `parent_prec` is the precedence of the enclosing context; the
/// expression is wrapped in parentheses when its own precedence is
/// lower.
///
/// **This function uses an explicit stack — it never recurses.**
pub(crate) fn fmt_expr(
    arena: &Arena,
    f: &mut fmt::Formatter<'_>,
    id: ExprId,
    parent_prec: u8,
) -> fmt::Result {
    // We push work items in *reverse* order so they come out
    // left-to-right when popped from the end.
    let mut stack: Vec<WorkItem> = Vec::with_capacity(32);
    stack.push(WorkItem::Expr(id, parent_prec));

    while let Some(item) = stack.pop() {
        match item {
            WorkItem::Lit(s) => write!(f, "{s}")?,
            WorkItem::Owned(s) => write!(f, "{s}")?,
            WorkItem::Expr(eid, par_prec) => {
                expand_expr(arena, eid, par_prec, &mut stack)?;
            }
        }
    }

    Ok(())
}

/// Convenience wrapper: format an expression to a `String`.
///
/// Equivalent to calling [`fmt_expr`] with precedence 0 and collecting
/// the output into a heap-allocated string.
#[allow(dead_code)]
pub(crate) fn format_expr(arena: &Arena, id: ExprId) -> String {
    struct FmtAdapter<'a>(&'a Arena, ExprId);
    impl std::fmt::Display for FmtAdapter<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt_expr(self.0, f, self.1, 0)
        }
    }
    format!("{}", FmtAdapter(arena, id))
}

/// Expand a single expression node into work items on the stack.
///
/// For atoms this directly writes to `f` (via a pushed `Owned`).
/// For composites it pushes child `Expr` items and literal separators.
///
/// Items are pushed in **reverse** display order so that popping
/// yields left-to-right output.
fn expand_expr(
    arena: &Arena,
    id: ExprId,
    parent_prec: u8,
    stack: &mut Vec<WorkItem>,
) -> fmt::Result {
    let node = arena.node(id);
    let my_prec = prec_of(node);
    let need_parens = my_prec < parent_prec;

    // Clone the node so we can release the borrow on `arena`.
    let node = node.clone();

    // Helper: push close-paren if needed (pushed first = emitted last).
    if need_parens {
        stack.push(WorkItem::Lit(")"));
    }

    match node {
        // ── Atoms ──────────────────────────────────────────────────
        ExprNode::Num(num_id) => {
            let r = arena.num(num_id);
            // Build the string eagerly for atoms.
            let s = if r.denom() == &BigInt::from(1) {
                format!("{}", r.numer())
            } else {
                format!("{}/{}", r.numer(), r.denom())
            };
            stack.push(WorkItem::Owned(s));
        }

        ExprNode::Symbol(sym_id) => {
            let name = arena.symbol_name(sym_id).to_owned();
            stack.push(WorkItem::Owned(name));
        }

        ExprNode::Pi => stack.push(WorkItem::Lit("pi")),
        ExprNode::E => stack.push(WorkItem::Lit("E")),
        ExprNode::ImaginaryUnit => stack.push(WorkItem::Lit("I")),
        ExprNode::EulerGamma => stack.push(WorkItem::Lit("EulerGamma")),
        ExprNode::Catalan => stack.push(WorkItem::Lit("Catalan")),
        ExprNode::GoldenRatio => stack.push(WorkItem::Lit("GoldenRatio")),
        ExprNode::PhysicalConstant(name_id, _) => {
            stack.push(WorkItem::Owned(arena.symbol_name(name_id).to_owned()));
        }
        ExprNode::Infinity => stack.push(WorkItem::Lit("oo")),
        ExprNode::NegInfinity => stack.push(WorkItem::Lit("-oo")),
        ExprNode::ComplexInfinity => stack.push(WorkItem::Lit("zoo")),
        ExprNode::NaN => stack.push(WorkItem::Lit("nan")),

        // ── Add ────────────────────────────────────────────────────
        //
        // Terms joined with " + ".  A term that is Neg(x) or
        // Mul([-1, ...]) is rendered with " - " instead.
        ExprNode::Add(args) => {
            // Sort children by display key (polynomial degree descending, then functions, then constants)
            let mut display_order: SmallVec<[ExprId; 6]> = args.clone();
            display_order.sort_by_key(|a| display_sort_key(arena, *a));

            // Push children in reverse (last child pushed first).
            for (i, &arg) in display_order.iter().enumerate().rev() {
                let child_node = arena.node(arg);

                if let ExprNode::Neg(inner) = child_node {
                    let inner = *inner;
                    if i == 0 {
                        // Leading negative: "-x"
                        stack.push(WorkItem::Expr(inner, PREC_UNARY));
                        stack.push(WorkItem::Lit("-"));
                    } else {
                        stack.push(WorkItem::Expr(inner, PREC_UNARY));
                        stack.push(WorkItem::Lit(" - "));
                    }
                } else if is_neg_one_mul(arena, arg) {
                    // Mul([-1, rest...]) → display as " - rest..."
                    // We extract the non-(-1) factors and push them
                    // directly to avoid the double-negation bug that
                    // occurs when mul_without_neg_one can't intern a
                    // new Mul node for the 3+-factor case.
                    let rest: SmallVec<[ExprId; 6]> =
                        if let ExprNode::Mul(children) = arena.node(arg) {
                            SmallVec::from_slice(&children[1..])
                        } else {
                            unreachable!("is_neg_one_mul confirmed Mul")
                        };
                    if i == 0 {
                        // Leading negative: "-rest..."
                        if rest.len() == 1 {
                            stack.push(WorkItem::Expr(rest[0], PREC_UNARY));
                        } else {
                            push_mul_factors(arena, &rest, PREC_MUL, stack);
                        }
                        stack.push(WorkItem::Lit("-"));
                    } else {
                        push_mul_factors(arena, &rest, PREC_MUL, stack);
                        stack.push(WorkItem::Lit(" - "));
                    }
                } else if is_neg_coeff_mul(arena, arg) {
                    let display_str = neg_coeff_mul_display(arena, arg);
                    if i == 0 {
                        stack.push(WorkItem::Owned(format!("-{display_str}")));
                    } else {
                        stack.push(WorkItem::Owned(format!(" - {display_str}")));
                    }
                } else if let ExprNode::Num(nid) = child_node
                    && arena.num(*nid).is_negative()
                    && i > 0
                {
                    // Negative numeric literal: render "x + -3" as "x - 3"
                    let r = arena.num(*nid);
                    let pos_r = -r.clone();
                    let abs_str = if pos_r.denom() == &BigInt::from(1) {
                        format!("{}", pos_r.numer())
                    } else {
                        format!("{}/{}", pos_r.numer(), pos_r.denom())
                    };
                    stack.push(WorkItem::Owned(format!(" - {abs_str}")));
                } else if i == 0 {
                    stack.push(WorkItem::Expr(arg, PREC_ADD));
                } else {
                    stack.push(WorkItem::Expr(arg, PREC_ADD));
                    stack.push(WorkItem::Lit(" + "));
                }
            }
        }

        // ── Mul ────────────────────────────────────────────────────
        //
        // Factors joined with "*".
        // Leading 1 is omitted; leading -1 becomes "-".
        ExprNode::Mul(args) => {
            if args.is_empty() {
                stack.push(WorkItem::Lit("1"));
            } else {
                let first_node = arena.node(args[0]);
                let is_neg_one = if let ExprNode::Num(nid) = first_node {
                    let r = arena.num(*nid);
                    *r == Ratio::from(BigInt::from(-1))
                } else {
                    false
                };
                let is_pos_one = if let ExprNode::Num(nid) = first_node {
                    let r = arena.num(*nid);
                    *r == Ratio::from(BigInt::from(1))
                } else {
                    false
                };

                if args.len() == 1 {
                    stack.push(WorkItem::Expr(args[0], PREC_MUL));
                } else if is_neg_one {
                    // "-1 * rest" → "-rest"
                    push_mul_factors(arena, &args[1..], PREC_MUL, stack);
                    stack.push(WorkItem::Lit("-"));
                } else if is_pos_one {
                    // "1 * rest" → "rest"
                    push_mul_factors(arena, &args[1..], PREC_MUL, stack);
                } else {
                    push_mul_factors(arena, &args, PREC_MUL, stack);
                }
            }
        }

        // ── Pow ────────────────────────────────────────────────────
        ExprNode::Pow(base, exp) => {
            // Detect Pow(x, 1/2) → display as sqrt(x)
            let is_half_exp = if let ExprNode::Num(nid) = arena.node(exp) {
                let r = arena.num(*nid);
                *r == Ratio::new(BigInt::from(1), BigInt::from(2))
            } else {
                false
            };

            // Detect Pow(x, 1/3) → display as cbrt(x)
            let is_third_exp = if let ExprNode::Num(nid) = arena.node(exp) {
                let r = arena.num(*nid);
                *r == Ratio::new(BigInt::from(1), BigInt::from(3))
            } else {
                false
            };

            // Special case: x^(-1) → "1/x" for cleaner display.
            let is_neg_one_exp = if let ExprNode::Num(nid) = arena.node(exp) {
                *arena.num(*nid) == Ratio::from(BigInt::from(-1))
            } else {
                false
            };

            if is_half_exp {
                stack.push(WorkItem::Lit(")"));
                stack.push(WorkItem::Expr(base, 0));
                stack.push(WorkItem::Lit("sqrt("));
            } else if is_third_exp {
                stack.push(WorkItem::Lit(")"));
                stack.push(WorkItem::Expr(base, 0));
                stack.push(WorkItem::Lit("cbrt("));
            } else if is_neg_one_exp {
                let base_prec = match arena.node(base) {
                    ExprNode::Add(_)
                    | ExprNode::Mul(_)
                    | ExprNode::Neg(_)
                    | ExprNode::Pow(_, _) => PREC_MUL + 1,
                    _ if num_base_needs_parens(arena, base) => PREC_ATOM + 1,
                    _ => PREC_MUL,
                };
                stack.push(WorkItem::Expr(base, base_prec));
                stack.push(WorkItem::Lit("1/"));
            } else {
                // Determine parenthesization for base.
                let base_prec = match arena.node(base) {
                    ExprNode::Add(_)
                    | ExprNode::Mul(_)
                    | ExprNode::Neg(_)
                    | ExprNode::Pow(_, _) => PREC_POW + 1,
                    // `(-2)^x`, `(2/3)^x` — a bare `-2^x` / `2/3^x` re-parses
                    // with the wrong grouping.
                    _ if num_base_needs_parens(arena, base) => PREC_ATOM + 1,
                    _ => PREC_POW,
                };

                // Determine parenthesization for exponent.
                let exp_prec = match arena.node(exp) {
                    ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => PREC_POW + 1,
                    ExprNode::Num(nid) => {
                        let r = arena.num(*nid);
                        if r.is_negative() || !r.is_integer() {
                            PREC_ATOM + 1
                        } else {
                            PREC_POW
                        }
                    }
                    _ => PREC_POW,
                };

                // Push in reverse: base ^ exp
                stack.push(WorkItem::Expr(exp, exp_prec));
                stack.push(WorkItem::Lit("^"));
                stack.push(WorkItem::Expr(base, base_prec));
            }
        }

        // ── Neg ────────────────────────────────────────────────────
        ExprNode::Neg(inner) => {
            let child_prec = match arena.node(inner) {
                ExprNode::Add(_) => PREC_UNARY + 1,
                _ => PREC_UNARY,
            };
            stack.push(WorkItem::Expr(inner, child_prec));
            stack.push(WorkItem::Lit("-"));
        }

        // ── Built-in functions ─────────────────────────────────────
        ExprNode::Sin(x) => push_func("sin", x, stack),
        ExprNode::Cos(x) => push_func("cos", x, stack),
        ExprNode::Tan(x) => push_func("tan", x, stack),
        ExprNode::Exp(x) => push_func("exp", x, stack),
        ExprNode::Ln(x) => push_func("ln", x, stack),
        ExprNode::Abs(x) => push_func("abs", x, stack),
        ExprNode::Asin(x) => push_func("asin", x, stack),
        ExprNode::Acos(x) => push_func("acos", x, stack),
        ExprNode::Atan(x) => push_func("atan", x, stack),
        ExprNode::Atan2(y, x) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(x, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(y, 0));
            stack.push(WorkItem::Lit("atan2("));
        }
        ExprNode::Sinh(x) => push_func("sinh", x, stack),
        ExprNode::Cosh(x) => push_func("cosh", x, stack),
        ExprNode::Tanh(x) => push_func("tanh", x, stack),
        ExprNode::Asinh(x) => push_func("asinh", x, stack),
        ExprNode::Acosh(x) => push_func("acosh", x, stack),
        ExprNode::Atanh(x) => push_func("atanh", x, stack),
        ExprNode::Sign(x) => push_func("sign", x, stack),
        ExprNode::Heaviside(x) => push_func("H", x, stack),
        ExprNode::DiracDelta(x) => push_func("DiracDelta", x, stack),
        ExprNode::Gamma(x) => push_func("Gamma", x, stack),
        ExprNode::LogGamma(x) => push_func("LogGamma", x, stack),
        ExprNode::Digamma(x) => push_func("Digamma", x, stack),
        ExprNode::Erf(x) => push_func("erf", x, stack),
        ExprNode::Erfc(x) => push_func("erfc", x, stack),
        ExprNode::LambertW(x) => push_func("W", x, stack),
        ExprNode::Beta(a, b) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(b, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(a, 0));
            stack.push(WorkItem::Lit("B("));
        }

        // ── Complex analysis ────────────────────────────────────────────
        ExprNode::Re(x) => push_func("re", x, stack),
        ExprNode::Im(x) => push_func("im", x, stack),
        ExprNode::Conjugate(x) => push_func("conjugate", x, stack),
        ExprNode::Arg(x) => push_func("arg", x, stack),

        // ── Special functions (0.2) ──────────────────────────────────────
        ExprNode::Si(x) => push_func("Si", x, stack),
        ExprNode::Ci(x) => push_func("Ci", x, stack),
        ExprNode::Ei(x) => push_func("Ei", x, stack),
        ExprNode::Li(x) => push_func("li", x, stack),
        ExprNode::Zeta(x) => push_func("zeta", x, stack),
        ExprNode::Polygamma(n, x) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(x, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(n, 0));
            stack.push(WorkItem::Lit("polygamma("));
        }
        ExprNode::KroneckerDelta(i, j) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(j, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(i, 0));
            stack.push(WorkItem::Lit("KroneckerDelta("));
        }
        ExprNode::Floor(x) => push_func("floor", x, stack),
        ExprNode::Ceiling(x) => push_func("ceiling", x, stack),

        // ── Min / Max ──────────────────────────────────────────────
        ExprNode::Min(ref args) => {
            stack.push(WorkItem::Lit(")"));
            for (i, &arg) in args.iter().enumerate().rev() {
                stack.push(WorkItem::Expr(arg, 0));
                if i > 0 {
                    stack.push(WorkItem::Lit(", "));
                }
            }
            stack.push(WorkItem::Lit("min("));
        }
        ExprNode::Max(ref args) => {
            stack.push(WorkItem::Lit(")"));
            for (i, &arg) in args.iter().enumerate().rev() {
                stack.push(WorkItem::Expr(arg, 0));
                if i > 0 {
                    stack.push(WorkItem::Lit(", "));
                }
            }
            stack.push(WorkItem::Lit("max("));
        }

        // ── Sum / Product ──────────────────────────────────────────
        ExprNode::Sum(body, var, lo, hi) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(hi, 0));
            stack.push(WorkItem::Lit(".."));
            stack.push(WorkItem::Expr(lo, 0));
            stack.push(WorkItem::Lit("="));
            stack.push(WorkItem::Expr(var, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit("Sum("));
        }
        ExprNode::Product_(body, var, lo, hi) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(hi, 0));
            stack.push(WorkItem::Lit(".."));
            stack.push(WorkItem::Expr(lo, 0));
            stack.push(WorkItem::Lit("="));
            stack.push(WorkItem::Expr(var, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit("Product("));
        }

        // ── Combinatorial ──────────────────────────────────────────
        ExprNode::Factorial(inner) => {
            // Display as "inner!" with parentheses if inner is compound
            let needs_parens = !arena.node(inner).is_atom();
            if needs_parens {
                stack.push(WorkItem::Lit(")!"));
                stack.push(WorkItem::Expr(inner, 0));
                stack.push(WorkItem::Lit("("));
            } else {
                stack.push(WorkItem::Lit("!"));
                stack.push(WorkItem::Expr(inner, 0));
            }
        }
        ExprNode::Binomial(n, k) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(k, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(n, 0));
            stack.push(WorkItem::Lit("C("));
        }

        // ── Apply (user-defined function) ──────────────────────────
        ExprNode::Apply(sym_id, ref args) => {
            let name = arena.symbol_name(sym_id).to_owned();
            // Push in reverse: name ( arg1 , arg2 , ... )
            stack.push(WorkItem::Lit(")"));
            for (i, &arg) in args.iter().enumerate().rev() {
                stack.push(WorkItem::Expr(arg, 0));
                if i > 0 {
                    stack.push(WorkItem::Lit(", "));
                }
            }
            stack.push(WorkItem::Lit("("));
            stack.push(WorkItem::Owned(name));
        }

        // ── Calculus forms ─────────────────────────────────────────
        ExprNode::Derivative(body, var) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(var, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit("Derivative("));
        }

        ExprNode::Integral(body, var) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(var, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit("Integral("));
        }

        // `Integral(body, var, lo, hi)` — the 4-argument form the parser
        // accepts, so Display round-trips.
        ExprNode::DefiniteIntegral(body, var, lo, hi) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(hi, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(lo, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(var, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit("Integral("));
        }

        // ── Limit ──────────────────────────────────────────────────
        ExprNode::Limit(body, var, point) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(point, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(var, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit("Limit("));
        }

        // ── Series ─────────────────────────────────────────────────
        ExprNode::Series(body, var, point, order) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(order, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(point, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(var, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit("Series("));
        }

        // ── Laplace Transform ──────────────────────────────────────
        ExprNode::LaplaceTransform(body, t, s) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(s, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(t, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit("LaplaceTransform("));
        }

        // ── Inverse Laplace Transform ──────────────────────────────
        ExprNode::InverseLaplaceTransform(body, s, t) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(t, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(s, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit("InverseLaplaceTransform("));
        }

        // ── Residue ────────────────────────────────────────────────
        ExprNode::Residue(body, var, point) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(point, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(var, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit("Residue("));
        }

        // ── RootOf ─────────────────────────────────────────────────
        ExprNode::RootOf(poly, index) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(index, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(poly, 0));
            stack.push(WorkItem::Lit("RootOf("));
        }

        // ── DSolve ─────────────────────────────────────────────────
        ExprNode::DSolve(expr, func, var) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(var, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(func, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(expr, 0));
            stack.push(WorkItem::Lit("DSolve("));
        }

        // ── RootSum ────────────────────────────────────────────────
        // Display as: RootSum(poly, sumvar -> body)
        ExprNode::RootSum(poly, body, sumvar) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(body, 0));
            stack.push(WorkItem::Lit(" -> "));
            stack.push(WorkItem::Expr(sumvar, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(poly, 0));
            stack.push(WorkItem::Lit("RootSum("));
        }

        // ── ConditionSet ───────────────────────────────────────────
        ExprNode::ConditionSet(var, condition) => {
            stack.push(WorkItem::Lit(")"));
            stack.push(WorkItem::Expr(condition, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(var, 0));
            stack.push(WorkItem::Lit("ConditionSet("));
        }

        // ── Boolean atoms ──────────────────────────────────────────
        ExprNode::BoolTrue => stack.push(WorkItem::Lit("True")),
        ExprNode::BoolFalse => stack.push(WorkItem::Lit("False")),

        // ── Relational operators ───────────────────────────────────
        ExprNode::Gt(lhs, rhs) => {
            stack.push(WorkItem::Expr(rhs, 21));
            stack.push(WorkItem::Lit(" > "));
            stack.push(WorkItem::Expr(lhs, 21));
        }
        ExprNode::Ge(lhs, rhs) => {
            stack.push(WorkItem::Expr(rhs, 21));
            stack.push(WorkItem::Lit(" >= "));
            stack.push(WorkItem::Expr(lhs, 21));
        }
        ExprNode::Eq_(lhs, rhs) => {
            stack.push(WorkItem::Expr(rhs, 21));
            stack.push(WorkItem::Lit(" == "));
            stack.push(WorkItem::Expr(lhs, 21));
        }
        ExprNode::Ne(lhs, rhs) => {
            stack.push(WorkItem::Expr(rhs, 21));
            stack.push(WorkItem::Lit(" != "));
            stack.push(WorkItem::Expr(lhs, 21));
        }

        // ── Logical connectives ────────────────────────────────────
        ExprNode::And(children) => {
            for (i, &child) in children.iter().enumerate().rev() {
                stack.push(WorkItem::Expr(child, 16));
                if i > 0 {
                    stack.push(WorkItem::Lit(" & "));
                }
            }
        }
        ExprNode::Or(children) => {
            for (i, &child) in children.iter().enumerate().rev() {
                stack.push(WorkItem::Expr(child, 11));
                if i > 0 {
                    stack.push(WorkItem::Lit(" | "));
                }
            }
        }
        ExprNode::Not(inner) => {
            stack.push(WorkItem::Expr(inner, PREC_UNARY));
            stack.push(WorkItem::Lit("!"));
        }

        // ── Piecewise ──────────────────────────────────────────────
        ExprNode::Piecewise(children) => {
            stack.push(WorkItem::Lit(")"));
            for (i, &(val, cond)) in children.iter().enumerate().rev() {
                stack.push(WorkItem::Expr(cond, 0));
                stack.push(WorkItem::Lit(" if "));
                stack.push(WorkItem::Expr(val, 0));
                if i > 0 {
                    stack.push(WorkItem::Lit(", "));
                }
            }
            stack.push(WorkItem::Lit("Piecewise("));
        }

        // ── Set atoms ──────────────────────────────────────────────
        ExprNode::EmptySet => stack.push(WorkItem::Lit("EmptySet")),
        ExprNode::UniversalSet => stack.push(WorkItem::Lit("UniversalSet")),

        // ── Interval ───────────────────────────────────────────────
        ExprNode::Interval(a, b, flags) => {
            let left = if flags & crate::base::node::INTERVAL_LEFT_OPEN != 0 {
                "("
            } else {
                "["
            };
            let right = if flags & crate::base::node::INTERVAL_RIGHT_OPEN != 0 {
                ")"
            } else {
                "]"
            };
            stack.push(WorkItem::Owned(right.to_string()));
            stack.push(WorkItem::Expr(b, 0));
            stack.push(WorkItem::Lit(", "));
            stack.push(WorkItem::Expr(a, 0));
            stack.push(WorkItem::Owned(left.to_string()));
        }

        // ── Finite set ─────────────────────────────────────────────
        ExprNode::FiniteSet(ref elems) => {
            stack.push(WorkItem::Lit("}"));
            for (i, &elem) in elems.iter().enumerate().rev() {
                stack.push(WorkItem::Expr(elem, 0));
                if i > 0 {
                    stack.push(WorkItem::Lit(", "));
                }
            }
            stack.push(WorkItem::Lit("{"));
        }

        // ── Set union ──────────────────────────────────────────────
        ExprNode::SetUnion(ref sets) => {
            for (i, &set) in sets.iter().enumerate().rev() {
                stack.push(WorkItem::Expr(set, 0));
                if i > 0 {
                    stack.push(WorkItem::Lit(" ∪ "));
                }
            }
        }

        // ── Set intersection ───────────────────────────────────────
        ExprNode::SetIntersection(ref sets) => {
            for (i, &set) in sets.iter().enumerate().rev() {
                stack.push(WorkItem::Expr(set, 0));
                if i > 0 {
                    stack.push(WorkItem::Lit(" ∩ "));
                }
            }
        }

        // ── Set complement ─────────────────────────────────────────
        ExprNode::SetComplement(a, b) => {
            stack.push(WorkItem::Expr(b, 0));
            stack.push(WorkItem::Lit(" \\ "));
            stack.push(WorkItem::Expr(a, 0));
        }
    }

    // Open paren (pushed last = emitted first).
    if need_parens {
        stack.push(WorkItem::Lit("("));
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Push a function call `name(arg)` onto the stack in reverse order.
fn push_func(name: &'static str, arg: ExprId, stack: &mut Vec<WorkItem>) {
    stack.push(WorkItem::Lit(")"));
    stack.push(WorkItem::Expr(arg, 0));
    stack.push(WorkItem::Lit("("));
    stack.push(WorkItem::Lit(name));
}

/// Push Mul factors joined by "*" onto the stack in reverse order.
///
/// Each factor that is an Add gets a parent precedence of `PREC_MUL + 1`
/// to force parenthesisation.
fn push_mul_factors(arena: &Arena, factors: &[ExprId], _mul_prec: u8, stack: &mut Vec<WorkItem>) {
    // A leading numeric literal is the coefficient; it is folded into the
    // fraction (numerator `p`, denominator `q`) when inverse factors exist.
    if let Some((&first, rest)) = factors.split_first()
        && let ExprNode::Num(nid) = arena.node(first)
    {
        push_mul_with_coeff(arena, Some(arena.num(*nid).clone()), rest, stack);
    } else {
        push_mul_with_coeff(arena, None, factors, stack);
    }
}

/// Push `coeff * factors` onto the stack in reverse order.
///
/// The factors are split into numerator factors and denominator bases
/// (`Pow(base, -1)`), and a rational coefficient `p/q` contributes `p` to
/// the numerator and `q` to the denominator, so that
/// `1/2 * (j - 1) * j⁻¹` renders as `(j - 1)/(2*j)` and `1/x * 1/y * (x+y)`
/// as `(x + y)/(x*y)` rather than `1/2*1/j*(j - 1)` / `1/x*1/y*(x + y)`.
/// Without inverse factors the coefficient is printed as-is: `1/2*j`.
fn push_mul_with_coeff(
    arena: &Arena,
    coeff: Option<Ratio<BigInt>>,
    factors: &[ExprId],
    stack: &mut Vec<WorkItem>,
) {
    let mut numer: SmallVec<[ExprId; 6]> = SmallVec::new();
    let mut denom_bases: SmallVec<[ExprId; 6]> = SmallVec::new();

    for &f in factors {
        if let ExprNode::Pow(base, exp) = arena.node(f)
            && let ExprNode::Num(nid) = arena.node(*exp)
            && *arena.num(*nid) == Ratio::from(BigInt::from(-1))
        {
            denom_bases.push(*base);
            continue;
        }
        numer.push(f);
    }

    // No inverse factors: plain product `coeff*f1*f2*…`.
    if denom_bases.is_empty() {
        if !numer.is_empty() {
            push_plain_mul_factors(arena, &numer, stack);
        }
        match coeff {
            Some(c) if numer.is_empty() => stack.push(WorkItem::Owned(rational_literal(&c))),
            Some(c) if c == Ratio::from(BigInt::from(-1)) => stack.push(WorkItem::Lit("-")),
            Some(c) if c != Ratio::from(BigInt::from(1)) => {
                stack.push(WorkItem::Lit("*"));
                stack.push(WorkItem::Owned(rational_literal(&c)));
            }
            Some(_) => {}
            None if numer.is_empty() => stack.push(WorkItem::Lit("1")),
            None => {}
        }
        return;
    }

    let (p, q) = match coeff {
        Some(c) => (c.numer().clone(), c.denom().clone()),
        None => (BigInt::from(1), BigInt::from(1)),
    };
    let q_is_one = q == BigInt::from(1);

    // ── Denominator ────────────────────────────────────────────
    if q_is_one && denom_bases.len() == 1 {
        let base = denom_bases[0];
        // Parenthesise compound bases: .../(x + y), .../(a*b), etc.
        let base_prec = match arena.node(base) {
            ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) | ExprNode::Pow(_, _) => {
                PREC_MUL + 1
            }
            _ => PREC_MUL,
        };
        stack.push(WorkItem::Expr(base, base_prec));
    } else {
        // Several denominator factors (possibly including `q`): .../(q*a*b)
        stack.push(WorkItem::Lit(")"));
        push_plain_mul_factors(arena, &denom_bases, stack);
        if !q_is_one {
            stack.push(WorkItem::Lit("*"));
            stack.push(WorkItem::Owned(q.to_string()));
        }
        stack.push(WorkItem::Lit("("));
    }

    stack.push(WorkItem::Lit("/"));

    // ── Numerator ──────────────────────────────────────────────
    let p_is_one = p == BigInt::from(1);
    let p_is_neg_one = p == BigInt::from(-1);
    if numer.is_empty() {
        stack.push(WorkItem::Owned(p.to_string()));
        return;
    }
    if numer.len() == 1 {
        let f = numer[0];
        let prec = match arena.node(f) {
            ExprNode::Add(_) => PREC_MUL + 1,
            _ => PREC_MUL,
        };
        stack.push(WorkItem::Expr(f, prec));
    } else {
        push_plain_mul_factors(arena, &numer, stack);
    }
    if p_is_neg_one {
        stack.push(WorkItem::Lit("-"));
    } else if !p_is_one {
        stack.push(WorkItem::Lit("*"));
        stack.push(WorkItem::Owned(p.to_string()));
    }
}

/// `p` or `p/q` for a rational literal.
fn rational_literal(r: &Ratio<BigInt>) -> String {
    if r.denom() == &BigInt::from(1) {
        format!("{}", r.numer())
    } else {
        format!("{}/{}", r.numer(), r.denom())
    }
}

/// Render `coeff * factors` to a string with the same fraction folding as
/// [`push_mul_with_coeff`].  Used where the caller needs an owned string
/// (negative-coefficient terms inside a sum).
fn render_mul_with_coeff(
    arena: &Arena,
    coeff: Option<Ratio<BigInt>>,
    factors: &[ExprId],
) -> String {
    let mut stack: Vec<WorkItem> = Vec::with_capacity(16);
    push_mul_with_coeff(arena, coeff, factors, &mut stack);
    let mut out = String::new();
    while let Some(item) = stack.pop() {
        match item {
            WorkItem::Lit(s) => out.push_str(s),
            WorkItem::Owned(s) => out.push_str(&s),
            WorkItem::Expr(eid, par_prec) => {
                // `expand_expr` only pushes onto the stack; it cannot fail.
                let _ = expand_expr(arena, eid, par_prec, &mut stack);
            }
        }
    }
    out
}

/// Push Mul factors joined by `*` without fraction splitting.
fn push_plain_mul_factors(arena: &Arena, factors: &[ExprId], stack: &mut Vec<WorkItem>) {
    for (i, &factor) in factors.iter().enumerate().rev() {
        // Add factors need parens inside Mul: x*(a + b).
        let child_prec = match arena.node(factor) {
            ExprNode::Add(_) => PREC_MUL + 1,
            _ => PREC_MUL,
        };
        stack.push(WorkItem::Expr(factor, child_prec));
        if i > 0 {
            stack.push(WorkItem::Lit("*"));
        }
    }
}

/// Given a Mul whose first factor is a negative number (not -1),
/// return the display string with the coefficient negated.
/// Since we have &Arena (read-only), we build an Owned string representation.
fn neg_coeff_mul_display(arena: &Arena, id: ExprId) -> String {
    if let ExprNode::Mul(children) = arena.node(id)
        && let Some(&first) = children.first()
        && let ExprNode::Num(nid) = arena.node(first)
    {
        let pos_r = -arena.num(*nid).clone();
        render_mul_with_coeff(arena, Some(pos_r), &children[1..])
    } else {
        arena.display(id).to_string()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arena::display
// ═══════════════════════════════════════════════════════════════════════════

impl Arena {
    /// Create a displayable wrapper for an expression.
    ///
    /// Returns an opaque type that implements [`Display`](fmt::Display).
    ///
    /// Uses `fmt::from_fn` (Rust 1.93+) — no wrapper struct needed.
    ///
    /// ```text
    /// let mut arena = Arena::new();
    /// let x = arena.symbol("x");
    /// let two = arena.int(2);
    /// let expr = arena.pow(x, two);
    /// assert_eq!(arena.display(expr).to_string(), "x^2");
    /// ```
    pub fn display(&self, id: ExprId) -> impl fmt::Display + '_ {
        fmt::from_fn(move |f| fmt_expr(self, f, id, 0))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use crate::base::arena::Arena;
    use crate::base::node::ExprNode;
    use smallvec::smallvec;

    macro_rules! assert_display {
        ($arena:expr, $id:expr, $expected:expr) => {
            assert_eq!($arena.display($id).to_string(), $expected);
        };
    }

    // ── Atoms ────────────────────────────────────────────────────────

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
        let a = Arena::new();
        assert_display!(a, a.zero, "0");
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

    /// A rational coefficient is folded into the fraction when inverse
    /// factors are present, and left alone otherwise.
    #[test]
    fn display_rational_coefficient_folds_into_fraction() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let j = a.symbol("j");
        let half = a.rational(1, 2);
        let neg_half = a.rational(-1, 2);
        let two = a.int(2);
        let three = a.int(3);
        let neg_one = a.neg_one;

        // (j - 1) * j^-1 * 1/2  →  (j - 1)/(2*j)
        let jm1 = a.sub(j, a.one);
        let inv_j = a.pow(j, neg_one);
        let e = a.mul(&[half, inv_j, jm1]);
        assert_display!(a, e, "(j - 1)/(2*j)");

        // 1/2 * x^-1  →  1/(2*x);  -1/2 * x^-1  →  -1/(2*x)
        let inv_x = a.pow(x, neg_one);
        let e = a.mul(&[half, inv_x]);
        assert_display!(a, e, "1/(2*x)");
        let e = a.mul(&[neg_half, inv_x]);
        assert_display!(a, e, "-1/(2*x)");

        // Several denominator factors: x * y^-1 * j^-1 * 2/3  →  2*x/(3*j*y)
        let inv_y = a.pow(y, neg_one);
        let two_thirds = a.rational(2, 3);
        let e = a.mul(&[two_thirds, x, inv_y, inv_j]);
        assert_display!(a, e, "2*x/(3*j*y)");

        // Integer coefficient: unchanged behaviour.
        let e = a.mul(&[three, x, inv_y]);
        assert_display!(a, e, "3*x/y");
        let e = a.mul(&[two, inv_y]);
        assert_display!(a, e, "2/y");

        // No inverse factor: the coefficient stays in front.
        let e = a.mul(&[half, x]);
        assert_display!(a, e, "1/2*x");
        let e = a.mul(&[neg_half, x]);
        assert_display!(a, e, "-1/2*x");

        // Inside a sum, a negative fractional term prints as " - a/(b)".
        let t = a.mul(&[neg_half, inv_x]);
        let s = a.add(&[x, t]);
        assert_display!(a, s, "x - 1/(2*x)");
        let t = a.mul(&[three, y, inv_x]);
        let nt = a.neg(t);
        let s = a.add(&[x, nt]);
        assert_display!(a, s, "-3*y/x + x");
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

    // ── Add ──────────────────────────────────────────────────────────

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
        // x - y  (canonical form: x + (-1)*y via neg distribution)
        let diff = a.sub(x, y);
        assert_display!(a, diff, "x - y");
    }

    #[test]
    fn display_add_leading_neg() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let neg_x = a.neg(x);
        let one = a.one;
        let sum = a.add(&[neg_x, one]);
        let s = a.display(sum).to_string();
        // Could be "-x + 1" or "1 - x" depending on sort order.
        assert!(
            s.contains('x') && s.contains('1'),
            "should contain both x and 1, got: {s}"
        );
    }

    // ── Mul ──────────────────────────────────────────────────────────

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
        let neg_x = a.neg(x);
        assert_display!(a, neg_x, "-x");
    }

    #[test]
    fn display_mul_one_coefficient() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        // Mul(1, x) should canonicalize to just x.
        let expr = a.mul(&[a.one, x]);
        assert_display!(a, expr, "x");
    }

    // ── Pow ──────────────────────────────────────────────────────────

    #[test]
    fn display_pow() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let p = a.pow(x, two);
        assert_display!(a, p, "x^2");
    }

    #[test]
    fn display_pow_add_base_gets_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let one = a.one;
        let sum = a.add(&[one, x]);
        let two = a.int(2);
        let p = a.pow(sum, two);
        assert_display!(a, p, "(x + 1)^2");
    }

    #[test]
    fn display_pow_add_exp_gets_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let sum = a.add(&[x, y]);
        let p = a.pow(x, sum);
        assert_display!(a, p, "x^(x + y)");
    }

    #[test]
    fn display_pow_rational_exp_gets_parens() {
        let mut a = Arena::new();
        // Use a non-perfect-square so radical simplification doesn't reduce it
        let x = a.int(5);
        let half = a.rational(1, 2);
        let p = a.pow(x, half);
        assert_display!(a, p, "sqrt(5)");
    }

    #[test]
    fn display_pow_negative_exp_gets_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let neg_one = a.int(-1);
        let p = a.pow(x, neg_one);
        assert_display!(a, p, "1/x");
    }

    // ── Neg ──────────────────────────────────────────────────────────

    #[test]
    fn display_neg_symbol() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        // neg(x) canonicalizes to Mul(-1, x), displayed as "-x"
        let neg = a.neg(x);
        assert_display!(a, neg, "-x");
    }

    #[test]
    fn display_neg_add_gets_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let sum = a.add(&[x, y]);
        // neg(x + y) distributes to -x - y
        let neg = a.neg(sum);
        assert_display!(a, neg, "-x - y");
    }

    // ── Functions ────────────────────────────────────────────────────

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
        let sin_x = a.sin(x);
        let cos_sin = a.cos(sin_x);
        assert_display!(a, cos_sin, "cos(sin(x))");
    }

    // ── Calculus forms ───────────────────────────────────────────────

    #[test]
    fn display_derivative() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let sin_x = a.sin(x);
        let d = a.intern(ExprNode::Derivative(sin_x, x));
        assert_display!(a, d, "Derivative(sin(x), x)");
    }

    #[test]
    fn display_integral() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let x_sq = {
            let two = a.int(2);
            a.pow(x, two)
        };
        let i = a.intern(ExprNode::Integral(x_sq, x));
        assert_display!(a, i, "Integral(x^2, x)");
    }

    #[test]
    fn display_definite_integral() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let x_sq = {
            let two = a.int(2);
            a.pow(x, two)
        };
        let one = a.one;
        let i = a.intern(ExprNode::DefiniteIntegral(x_sq, x, a.zero, one));
        assert_display!(a, i, "Integral(x^2, x, 0, 1)");
    }

    // ── Composite ────────────────────────────────────────────────────

    #[test]
    fn display_mul_in_add_no_extra_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let z = a.symbol("z");
        let xy = a.mul(&[x, y]);
        let sum = a.add(&[z, xy]);
        // Sort: x*y (degree 2) comes before z (degree 1) in display order
        assert_display!(a, sum, "x*y + z");
    }

    #[test]
    fn display_add_in_mul_gets_parens() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let z = a.symbol("z");
        let sum = a.add(&[x, y]);
        let prod = a.mul(&[z, sum]);
        // z sorts before (x + y)
        assert_display!(a, prod, "z*(x + y)");
    }

    // ── Deep expression (stack safety) ───────────────────────────────

    #[test]
    fn display_deep_expression_no_stack_overflow() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        // Build sin(sin(sin(...sin(x)...))) 10,000 levels deep.
        let mut expr = x;
        for _ in 0..10_000 {
            expr = a.sin(expr);
        }
        // Should not stack-overflow.
        let s = a.display(expr).to_string();
        assert!(s.starts_with("sin("), "should start with sin(");
        assert!(s.contains('x'), "should contain x somewhere");
    }

    // ── Regression: raw Add/Mul (non-canonical) ─────────────────────

    #[test]
    fn display_raw_add() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let raw = a.intern(ExprNode::Add(smallvec![x, y]));
        assert_display!(a, raw, "x + y");
    }

    #[test]
    fn display_raw_mul() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let raw = a.intern(ExprNode::Mul(smallvec![x, y]));
        assert_display!(a, raw, "x*y");
    }

    #[test]
    fn display_polynomial_order() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let one = a.one;
        let x2 = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[one, x2, two_x]);
        let s = a.display(expr).to_string();
        // Should display as x^2 + 2*x + 1 (descending degree), not 1 + x^2 + 2*x
        assert!(
            s.starts_with("x^2"),
            "polynomial should display in descending degree order: {s}"
        );
    }

    // ── Subtraction display (negative-term cleanup) ─────────────────

    #[test]
    fn display_subtraction_clean() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let three = a.int(3);
        let neg_three = a.neg(three);
        let expr = a.add(&[x, neg_three]);
        assert_eq!(a.display(expr).to_string(), "x - 3");
    }

    #[test]
    fn display_subtraction_mul_neg_coeff() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let neg_y = a.neg(y);
        let expr = a.add(&[x, neg_y]);
        assert_eq!(a.display(expr).to_string(), "x - y");
    }

    #[test]
    fn display_leading_negative() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let neg_x = a.neg(x);
        let one = a.int(1);
        let expr = a.add(&[neg_x, one]);
        // Leading negative: "-x + 1" is acceptable
        let s = a.display(expr).to_string();
        assert!(!s.contains("+ -"), "should not have '+ -' pattern: {s}");
    }

    // ── 0.2 nodes ──────────────────────────────────────────────────────────

    #[test]
    fn display_named_constants() {
        let a = Arena::new();
        assert_display!(a, a.euler_gamma, "EulerGamma");
        assert_display!(a, a.catalan, "Catalan");
        assert_display!(a, a.golden_ratio, "GoldenRatio");
    }

    #[test]
    fn display_complex_and_special_nodes() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let n = a.symbol("n");
        let cases = [
            (a.intern(ExprNode::Re(x)), "re(x)"),
            (a.intern(ExprNode::Im(x)), "im(x)"),
            (a.intern(ExprNode::Conjugate(x)), "conjugate(x)"),
            (a.intern(ExprNode::Arg(x)), "arg(x)"),
            (a.intern(ExprNode::Si(x)), "Si(x)"),
            (a.intern(ExprNode::Ci(x)), "Ci(x)"),
            (a.intern(ExprNode::Ei(x)), "Ei(x)"),
            (a.intern(ExprNode::Li(x)), "li(x)"),
            (a.intern(ExprNode::Zeta(x)), "zeta(x)"),
            (a.intern(ExprNode::Polygamma(n, x)), "polygamma(n, x)"),
            (
                a.intern(ExprNode::KroneckerDelta(n, x)),
                "KroneckerDelta(n, x)",
            ),
        ];
        for (id, expected) in cases {
            assert_display!(a, id, expected);
        }
        // Function nodes behave as atoms w.r.t. precedence: no extra parens.
        let re_x = a.intern(ExprNode::Re(x));
        let two = a.int(2);
        let prod = a.mul(&[two, re_x]);
        assert_display!(a, prod, "2*re(x)");
        let sq = a.pow(re_x, two);
        assert_display!(a, sq, "re(x)^2");
    }
}
