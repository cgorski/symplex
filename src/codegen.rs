//! Rust source code generation from symbolic expressions.
//!
//! Converts an expression tree into a Rust function body string.
//! Uses CSE (common subexpression elimination) for efficiency.

use crate::arena::Arena;
use crate::errors::SymplexError;
use crate::node::{ExprId, ExprNode};
use num_traits::ToPrimitive;

/// Generate a Rust function as a string from an expression.
///
/// The generated function takes `f64` arguments and returns `f64`.
///
/// # Example output
/// ```text
/// pub fn my_func(x: f64, y: f64) -> f64 {
///     let t0 = x * x;
///     t0 * y.sin() + x.cos()
/// }
/// ```
pub(crate) fn to_rust_fn(
    arena: &mut Arena,
    expr: ExprId,
    name: &str,
    args: &[&str],
) -> Result<String, SymplexError> {
    // Run CSE first to factor out common subexpressions
    let cse_result = crate::cse::cse(arena, expr);

    let mut lines = Vec::new();

    // Function signature
    let params: Vec<String> = args.iter().map(|a| format!("{a}: f64")).collect();
    lines.push(format!("pub fn {name}({}) -> f64 {{", params.join(", ")));

    // CSE bindings — each extracted subexpression becomes a `let` binding
    for (i, (_, binding_expr)) in cse_result.bindings.iter().enumerate() {
        let code = expr_to_rust(arena, *binding_expr, args)?;
        lines.push(format!("    let t{i} = {code};"));
    }

    // Final expression (the function body's return value)
    let result_code = expr_to_rust(arena, cse_result.expr, args)?;
    lines.push(format!("    {result_code}"));
    lines.push("}".to_string());

    Ok(lines.join("\n"))
}

/// Convert a single expression node to Rust source code.
fn expr_to_rust(arena: &Arena, id: ExprId, var_names: &[&str]) -> Result<String, SymplexError> {
    match arena.node(id).clone() {
        ExprNode::Num(nid) => {
            let r = arena.num(nid);
            if r.is_integer() {
                let n = r.numer();
                // Use the suffix form for readability
                Ok(format!("{n}_f64"))
            } else {
                let num = r.numer();
                let den = r.denom();
                Ok(format!("({num}_f64 / {den}_f64)"))
            }
        }
        ExprNode::Symbol(sid) => {
            let sym_name = arena.symbols.name(sid);
            // Check if it's a known variable argument
            if var_names.contains(&sym_name) {
                Ok(sym_name.to_string())
            } else if sym_name.starts_with("__cse_") {
                // CSE temporary variable — map __cse_N to tN
                let idx_str = sym_name.strip_prefix("__cse_").unwrap();
                let idx: usize = idx_str.parse().map_err(|_| {
                    SymplexError::NotImplemented(format!("invalid CSE variable name: {sym_name}"))
                })?;
                Ok(format!("t{idx}"))
            } else {
                Err(SymplexError::FreeSymbol {
                    name: sym_name.to_string(),
                })
            }
        }
        ExprNode::Pi => Ok("std::f64::consts::PI".to_string()),
        ExprNode::E => Ok("std::f64::consts::E".to_string()),
        ExprNode::Infinity => Ok("f64::INFINITY".to_string()),
        ExprNode::NegInfinity => Ok("f64::NEG_INFINITY".to_string()),
        ExprNode::NaN | ExprNode::ComplexInfinity => Ok("f64::NAN".to_string()),
        ExprNode::Add(ref children) => {
            if children.is_empty() {
                return Ok("0_f64".to_string());
            }
            let parts: Result<Vec<String>, _> = children
                .iter()
                .map(|&c| expr_to_rust(arena, c, var_names))
                .collect();
            Ok(format!("({})", parts?.join(" + ")))
        }
        ExprNode::Mul(ref children) => {
            if children.is_empty() {
                return Ok("1_f64".to_string());
            }
            let parts: Result<Vec<String>, _> = children
                .iter()
                .map(|&c| expr_to_rust(arena, c, var_names))
                .collect();
            Ok(format!("({})", parts?.join(" * ")))
        }
        ExprNode::Pow(base, exp) => {
            let b = expr_to_rust(arena, base, var_names)?;
            // Check if exponent is a small integer — use powi for efficiency
            if let Some(r) = arena.as_num(exp) {
                if r.is_integer() {
                    if let Some(n) = r.numer().to_i64() {
                        if (0..=10).contains(&n) {
                            return Ok(format!("{b}.powi({n})"));
                        }
                        if (-10..0).contains(&n) {
                            return Ok(format!("{b}.powi({n})"));
                        }
                    }
                }
                // Check for sqrt: exponent == 1/2
                if *r.numer() == 1.into() && *r.denom() == 2.into() {
                    return Ok(format!("{b}.sqrt()"));
                }
                // Check for cbrt: exponent == 1/3
                if *r.numer() == 1.into() && *r.denom() == 3.into() {
                    return Ok(format!("{b}.cbrt()"));
                }
            }
            let e = expr_to_rust(arena, exp, var_names)?;
            Ok(format!("{b}.powf({e})"))
        }
        ExprNode::Neg(inner) => {
            let inner_code = expr_to_rust(arena, inner, var_names)?;
            Ok(format!("(-{inner_code})"))
        }
        ExprNode::Sin(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.sin()"))
        }
        ExprNode::Cos(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.cos()"))
        }
        ExprNode::Tan(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.tan()"))
        }
        ExprNode::Exp(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.exp()"))
        }
        ExprNode::Ln(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.ln()"))
        }
        ExprNode::Abs(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.abs()"))
        }
        ExprNode::Asin(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.asin()"))
        }
        ExprNode::Acos(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.acos()"))
        }
        ExprNode::Atan(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.atan()"))
        }
        ExprNode::Atan2(y, x) => {
            let y_code = expr_to_rust(arena, y, var_names)?;
            let x_code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{y_code}.atan2({x_code})"))
        }
        ExprNode::Sinh(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.sinh()"))
        }
        ExprNode::Cosh(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.cosh()"))
        }
        ExprNode::Tanh(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.tanh()"))
        }
        ExprNode::Asinh(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.asinh()"))
        }
        ExprNode::Acosh(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.acosh()"))
        }
        ExprNode::Atanh(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.atanh()"))
        }
        ExprNode::Sign(x) => {
            let code = expr_to_rust(arena, x, var_names)?;
            Ok(format!("{code}.signum()"))
        }
        // Node types that cannot be meaningfully compiled to Rust f64 code
        ExprNode::ImaginaryUnit => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for imaginary unit".to_string(),
        )),
        ExprNode::Factorial(_) | ExprNode::Binomial(_, _) => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for combinatorial functions".to_string(),
        )),
        ExprNode::Apply(_, _) => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for user-defined Apply nodes".to_string(),
        )),
        ExprNode::Derivative(_, _) => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for unevaluated Derivative".to_string(),
        )),
        ExprNode::Integral(_, _) => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for unevaluated Integral".to_string(),
        )),
        ExprNode::Piecewise(ref branches) => {
            // Generate a chain of if/else expressions
            codegen_piecewise(arena, branches, var_names)
        }
        ExprNode::BoolTrue
        | ExprNode::BoolFalse
        | ExprNode::Gt(_, _)
        | ExprNode::Ge(_, _)
        | ExprNode::Eq_(_, _)
        | ExprNode::Ne(_, _)
        | ExprNode::And(_)
        | ExprNode::Or(_)
        | ExprNode::Not(_) => Err(SymplexError::NotImplemented(format!(
            "cannot generate Rust f64 code for boolean/relational node: {:?}",
            arena.node(id)
        ))),
    }
}

/// Generate Rust code for a piecewise expression as a chain of if/else.
fn codegen_piecewise(
    arena: &Arena,
    branches: &[(ExprId, ExprId)],
    var_names: &[&str],
) -> Result<String, SymplexError> {
    if branches.is_empty() {
        return Ok("f64::NAN".to_string());
    }

    let mut parts = Vec::new();
    for (i, &(value, cond)) in branches.iter().enumerate() {
        let val_code = expr_to_rust(arena, value, var_names)?;
        let cond_code = bool_to_rust(arena, cond, var_names)?;
        if i == 0 {
            parts.push(format!("if {cond_code} {{ {val_code} }}"));
        } else {
            parts.push(format!("else if {cond_code} {{ {val_code} }}"));
        }
    }
    parts.push("else { f64::NAN }".to_string());
    Ok(parts.join(" "))
}

/// Convert a boolean expression node to Rust source code.
fn bool_to_rust(arena: &Arena, id: ExprId, var_names: &[&str]) -> Result<String, SymplexError> {
    match arena.node(id).clone() {
        ExprNode::BoolTrue => Ok("true".to_string()),
        ExprNode::BoolFalse => Ok("false".to_string()),
        ExprNode::Gt(a, b) => {
            let la = expr_to_rust(arena, a, var_names)?;
            let lb = expr_to_rust(arena, b, var_names)?;
            Ok(format!("({la} > {lb})"))
        }
        ExprNode::Ge(a, b) => {
            let la = expr_to_rust(arena, a, var_names)?;
            let lb = expr_to_rust(arena, b, var_names)?;
            Ok(format!("({la} >= {lb})"))
        }
        ExprNode::Eq_(a, b) => {
            let la = expr_to_rust(arena, a, var_names)?;
            let lb = expr_to_rust(arena, b, var_names)?;
            Ok(format!("({la} == {lb})"))
        }
        ExprNode::Ne(a, b) => {
            let la = expr_to_rust(arena, a, var_names)?;
            let lb = expr_to_rust(arena, b, var_names)?;
            Ok(format!("({la} != {lb})"))
        }
        ExprNode::And(ref children) => {
            let parts: Result<Vec<String>, _> = children
                .iter()
                .map(|&c| bool_to_rust(arena, c, var_names))
                .collect();
            Ok(format!("({})", parts?.join(" && ")))
        }
        ExprNode::Or(ref children) => {
            let parts: Result<Vec<String>, _> = children
                .iter()
                .map(|&c| bool_to_rust(arena, c, var_names))
                .collect();
            Ok(format!("({})", parts?.join(" || ")))
        }
        ExprNode::Not(inner) => {
            let code = bool_to_rust(arena, inner, var_names)?;
            Ok(format!("(!{code})"))
        }
        _ => Err(SymplexError::NotImplemented(format!(
            "cannot generate Rust boolean code for node: {:?}",
            arena.node(id)
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    #[test]
    fn codegen_simple_number() {
        let mut a = Arena::new();
        let five = a.int(5);
        let code = expr_to_rust(&a, five, &[]).unwrap();
        assert_eq!(code, "5_f64");
    }

    #[test]
    fn codegen_rational_number() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let code = expr_to_rust(&a, half, &[]).unwrap();
        assert_eq!(code, "(1_f64 / 2_f64)");
    }

    #[test]
    fn codegen_variable() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let code = expr_to_rust(&a, x, &["x"]).unwrap();
        assert_eq!(code, "x");
    }

    #[test]
    fn codegen_free_symbol_is_error() {
        let mut a = Arena::new();
        let y = sym(&mut a, "y");
        let result = expr_to_rust(&a, y, &["x"]);
        assert!(result.is_err());
    }

    #[test]
    fn codegen_pi_and_e() {
        let a = Arena::new();
        let pi_code = expr_to_rust(&a, a.pi, &[]).unwrap();
        assert_eq!(pi_code, "std::f64::consts::PI");

        let e_code = expr_to_rust(&a, a.e_const, &[]).unwrap();
        assert_eq!(e_code, "std::f64::consts::E");
    }

    #[test]
    fn codegen_sin_cos() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let code = expr_to_rust(&a, sin_x, &["x"]).unwrap();
        assert!(code.contains(".sin()"));

        let cos_x = a.cos(x);
        let code = expr_to_rust(&a, cos_x, &["x"]).unwrap();
        assert!(code.contains(".cos()"));
    }

    #[test]
    fn codegen_pow_integer_uses_powi() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let code = expr_to_rust(&a, x2, &["x"]).unwrap();
        assert!(code.contains("powi(2)"), "expected powi(2), got: {code}");
    }

    #[test]
    fn codegen_pow_half_uses_sqrt() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let half = a.rational(1, 2);
        let sqrt_x = a.pow(x, half);
        let code = expr_to_rust(&a, sqrt_x, &["x"]).unwrap();
        assert!(code.contains(".sqrt()"), "expected sqrt(), got: {code}");
    }

    #[test]
    fn codegen_neg() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_x = a.neg(x);
        let code = expr_to_rust(&a, neg_x, &["x"]).unwrap();
        assert!(code.contains('-'), "expected negation, got: {code}");
    }

    #[test]
    fn codegen_to_rust_fn_produces_valid_fn() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let code = to_rust_fn(&mut a, x2, "square", &["x"]).unwrap();
        assert!(code.contains("pub fn square(x: f64) -> f64 {"));
        assert!(code.contains('}'));
    }

    #[test]
    fn codegen_cse_temp_variable() {
        let mut a = Arena::new();
        let cse_var = sym(&mut a, "__cse_0");
        let code = expr_to_rust(&a, cse_var, &["x"]).unwrap();
        assert_eq!(code, "t0");
    }

    #[test]
    fn codegen_exp_ln() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let exp_x = a.exp(x);
        let code = expr_to_rust(&a, exp_x, &["x"]).unwrap();
        assert!(code.contains(".exp()"));

        let ln_x = a.ln(x);
        let code = expr_to_rust(&a, ln_x, &["x"]).unwrap();
        assert!(code.contains(".ln()"));
    }

    #[test]
    fn codegen_abs_sign() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let abs_x = a.abs(x);
        let code = expr_to_rust(&a, abs_x, &["x"]).unwrap();
        assert!(code.contains(".abs()"));

        let sign_x = a.sign(x);
        let code = expr_to_rust(&a, sign_x, &["x"]).unwrap();
        assert!(code.contains(".signum()"));
    }

    #[test]
    fn codegen_hyperbolic_functions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        let sinh_x = a.sinh(x);
        assert!(
            expr_to_rust(&a, sinh_x, &["x"])
                .unwrap()
                .contains(".sinh()")
        );

        let cosh_x = a.cosh(x);
        assert!(
            expr_to_rust(&a, cosh_x, &["x"])
                .unwrap()
                .contains(".cosh()")
        );

        let tanh_x = a.tanh(x);
        assert!(
            expr_to_rust(&a, tanh_x, &["x"])
                .unwrap()
                .contains(".tanh()")
        );
    }

    #[test]
    fn codegen_inverse_trig() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        let asin_x = a.asin(x);
        assert!(
            expr_to_rust(&a, asin_x, &["x"])
                .unwrap()
                .contains(".asin()")
        );

        let acos_x = a.acos(x);
        assert!(
            expr_to_rust(&a, acos_x, &["x"])
                .unwrap()
                .contains(".acos()")
        );

        let atan_x = a.atan(x);
        assert!(
            expr_to_rust(&a, atan_x, &["x"])
                .unwrap()
                .contains(".atan()")
        );
    }

    #[test]
    fn codegen_add_mul() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        let sum = a.add(&[x, y]);
        let code = expr_to_rust(&a, sum, &["x", "y"]).unwrap();
        assert!(code.contains('+'), "expected +, got: {code}");

        let prod = a.mul(&[x, y]);
        let code = expr_to_rust(&a, prod, &["x", "y"]).unwrap();
        assert!(code.contains('*'), "expected *, got: {code}");
    }

    #[test]
    fn codegen_two_param_fn() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let code = to_rust_fn(&mut a, sum, "add_xy", &["x", "y"]).unwrap();
        assert!(code.contains("x: f64, y: f64"));
        assert!(code.contains("pub fn add_xy"));
    }

    #[test]
    fn codegen_imaginary_is_error() {
        let a = Arena::new();
        let result = expr_to_rust(&a, a.i_unit, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn codegen_atan2() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let at2 = a.atan2(y, x);
        let code = expr_to_rust(&a, at2, &["x", "y"]).unwrap();
        assert!(code.contains(".atan2("), "expected atan2, got: {code}");
    }

    #[test]
    fn codegen_negative_powi() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_two = a.int(-2);
        let x_neg2 = a.pow(x, neg_two);
        let code = expr_to_rust(&a, x_neg2, &["x"]).unwrap();
        assert!(code.contains("powi(-2)"), "expected powi(-2), got: {code}");
    }

    #[test]
    fn codegen_inverse_hyperbolic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        let asinh_x = a.asinh(x);
        assert!(
            expr_to_rust(&a, asinh_x, &["x"])
                .unwrap()
                .contains(".asinh()")
        );

        let acosh_x = a.acosh(x);
        assert!(
            expr_to_rust(&a, acosh_x, &["x"])
                .unwrap()
                .contains(".acosh()")
        );

        let atanh_x = a.atanh(x);
        assert!(
            expr_to_rust(&a, atanh_x, &["x"])
                .unwrap()
                .contains(".atanh()")
        );
    }
}
