//! Compile symbolic expressions to callable Rust closures.
//!
//! [`lambdify`] converts a symbolic expression into a `Box<dyn Fn(&[f64]) -> f64>`
//! that evaluates numerically at given variable values.

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use num_traits::{One, ToPrimitive};
use std::collections::HashMap;

/// Result type for [`lambdify`]: a boxed, thread-safe closure from `&[f64]` to `f64`,
/// or `None` when the expression cannot be compiled.
pub(crate) type LambdifyResult = Option<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>;

/// Compile `expr` into a closure that evaluates it numerically.
///
/// `var_names` maps variable names to their positional index in the
/// input array. For example, `["x", "y"]` means the closure takes
/// `&[f64]` where index 0 is `x` and index 1 is `y`.
///
/// Returns `None` if the expression contains unresolvable nodes
/// (e.g., unevaluated Integrals, user-defined Apply nodes).
pub(crate) fn lambdify(arena: &Arena, expr: ExprId, var_names: &[&str]) -> LambdifyResult {
    // Build a variable index map
    let var_map: HashMap<String, usize> = var_names
        .iter()
        .enumerate()
        .map(|(i, &name)| (name.to_string(), i))
        .collect();

    // Compile to an instruction list (bytecode-like)
    let mut instructions = Vec::new();
    compile_recursive(arena, expr, &var_map, &mut instructions)?;

    Some(Box::new(move |args: &[f64]| -> f64 {
        execute(&instructions, args)
    }))
}

/// A simple stack-based instruction for numerical evaluation.
#[derive(Clone, Debug)]
enum Instruction {
    PushConst(f64),
    PushVar(usize), // index into args array
    Add,
    Mul,
    #[allow(dead_code)]
    Sub, // top - second
    Neg,
    #[allow(dead_code)]
    Div, // second / top
    Pow, // second ^ top
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    #[allow(dead_code)]
    Sqrt,
    Cbrt,
    Powi(i32),
    Abs,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Sign,
    Heaviside,
    DiracDelta,
    Atan2,
    Floor,
    Ceiling,
    Min2,
    Max2,
}

/// Recursively compile an expression into a list of stack-based instructions.
///
/// Uses an explicit match on every [`ExprNode`] variant.  Returns `None` if
/// any sub-expression is not numerically evaluable (e.g. `ImaginaryUnit`,
/// `Factorial`, `Apply`, `Derivative`, `Integral`).
fn compile_recursive(
    arena: &Arena,
    id: ExprId,
    var_map: &HashMap<String, usize>,
    out: &mut Vec<Instruction>,
) -> Option<()> {
    match arena.node(id).clone() {
        ExprNode::Num(nid) => {
            let r = arena.num(nid);
            let numer_f: f64 = r.numer().to_f64().unwrap_or(f64::NAN);
            let denom_f: f64 = r.denom().to_f64().unwrap_or(1.0);
            out.push(Instruction::PushConst(numer_f / denom_f));
        }
        ExprNode::Symbol(sid) => {
            let name = arena.symbol_name(sid);
            let idx = var_map.get(name)?;
            out.push(Instruction::PushVar(*idx));
        }
        ExprNode::Pi => out.push(Instruction::PushConst(std::f64::consts::PI)),
        ExprNode::E => out.push(Instruction::PushConst(std::f64::consts::E)),
        ExprNode::ImaginaryUnit => return None, // Can't lambdify complex
        ExprNode::PhysicalConstant(_, value_id) => {
            // Recursively compile the stored exact value
            compile_recursive(arena, value_id, var_map, out)?;
        }
        ExprNode::Infinity => out.push(Instruction::PushConst(f64::INFINITY)),
        ExprNode::NegInfinity => out.push(Instruction::PushConst(f64::NEG_INFINITY)),
        ExprNode::NaN | ExprNode::ComplexInfinity => {
            out.push(Instruction::PushConst(f64::NAN));
        }

        ExprNode::Add(ref children) => {
            if children.is_empty() {
                out.push(Instruction::PushConst(0.0));
                return Some(());
            }
            compile_recursive(arena, children[0], var_map, out)?;
            for &child in &children[1..] {
                compile_recursive(arena, child, var_map, out)?;
                out.push(Instruction::Add);
            }
        }
        ExprNode::Mul(ref children) => {
            if children.is_empty() {
                out.push(Instruction::PushConst(1.0));
                return Some(());
            }
            compile_recursive(arena, children[0], var_map, out)?;
            for &child in &children[1..] {
                compile_recursive(arena, child, var_map, out)?;
                out.push(Instruction::Mul);
            }
        }
        ExprNode::Pow(base, exp) => {
            // Mirror the codegen path's special-casing for common exponents.
            // This avoids f64::powf(negative, frac) → NaN for odd roots.
            if let Some(r) = arena.as_num(exp) {
                // Integer exponent → powi (handles negative bases correctly)
                if r.is_integer() {
                    if let Ok(n) = i32::try_from(r.to_integer()) {
                        compile_recursive(arena, base, var_map, out)?;
                        out.push(Instruction::Powi(n));
                        return Some(());
                    }
                }
                let (numer, denom) = (r.numer().clone(), r.denom().clone());
                // exp == 1/2 → sqrt
                if numer == 1.into() && denom == 2.into() {
                    compile_recursive(arena, base, var_map, out)?;
                    out.push(Instruction::Sqrt);
                    return Some(());
                }
                // exp == 1/3 → cbrt (handles negative bases correctly!)
                if numer == 1.into() && denom == 3.into() {
                    compile_recursive(arena, base, var_map, out)?;
                    out.push(Instruction::Cbrt);
                    return Some(());
                }
                // General odd-denominator fractional exponent →
                // sign(base) * |base|^(p/q)  (sign-preserving real root)
                if !r.denom().is_one() && r.denom() % num_bigint::BigInt::from(2) != num_bigint::BigInt::from(0) {
                    compile_recursive(arena, base, var_map, out)?;
                    out.push(Instruction::Sign);
                    compile_recursive(arena, base, var_map, out)?;
                    out.push(Instruction::Abs);
                    compile_recursive(arena, exp, var_map, out)?;
                    out.push(Instruction::Pow);
                    out.push(Instruction::Mul); // sign * |base|^exp
                    return Some(());
                }
            }
            // General case: standard powf
            compile_recursive(arena, base, var_map, out)?;
            compile_recursive(arena, exp, var_map, out)?;
            out.push(Instruction::Pow);
        }
        ExprNode::Neg(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Neg);
        }
        ExprNode::Sin(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Sin);
        }
        ExprNode::Cos(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Cos);
        }
        ExprNode::Tan(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Tan);
        }
        ExprNode::Exp(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Exp);
        }
        ExprNode::Ln(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Ln);
        }
        ExprNode::Abs(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Abs);
        }
        ExprNode::Asin(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Asin);
        }
        ExprNode::Acos(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Acos);
        }
        ExprNode::Atan(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Atan);
        }
        ExprNode::Atan2(y, x) => {
            compile_recursive(arena, y, var_map, out)?;
            compile_recursive(arena, x, var_map, out)?;
            out.push(Instruction::Atan2);
        }
        ExprNode::Sinh(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Sinh);
        }
        ExprNode::Cosh(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Cosh);
        }
        ExprNode::Tanh(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Tanh);
        }
        ExprNode::Asinh(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Asinh);
        }
        ExprNode::Acosh(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Acosh);
        }
        ExprNode::Atanh(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Atanh);
        }
        ExprNode::Sign(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Sign);
        }
        ExprNode::Heaviside(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Heaviside);
        }
        ExprNode::DiracDelta(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::DiracDelta);
        }
        ExprNode::Floor(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Floor);
        }
        ExprNode::Ceiling(inner) => {
            compile_recursive(arena, inner, var_map, out)?;
            out.push(Instruction::Ceiling);
        }
        ExprNode::Min(ref children) => {
            if children.is_empty() {
                out.push(Instruction::PushConst(f64::INFINITY));
                return Some(());
            }
            compile_recursive(arena, children[0], var_map, out)?;
            for &child in &children[1..] {
                compile_recursive(arena, child, var_map, out)?;
                out.push(Instruction::Min2);
            }
        }
        ExprNode::Max(ref children) => {
            if children.is_empty() {
                out.push(Instruction::PushConst(f64::NEG_INFINITY));
                return Some(());
            }
            compile_recursive(arena, children[0], var_map, out)?;
            for &child in &children[1..] {
                compile_recursive(arena, child, var_map, out)?;
                out.push(Instruction::Max2);
            }
        }

        ExprNode::Factorial(_)
        | ExprNode::Binomial(_, _)
        | ExprNode::Gamma(_)
        | ExprNode::LogGamma(_)
        | ExprNode::Digamma(_)
        | ExprNode::Erf(_)
        | ExprNode::Erfc(_)
        | ExprNode::LambertW(_)
        | ExprNode::Beta(_, _) => return None,
        ExprNode::Apply(_, _)
        | ExprNode::Derivative(_, _)
        | ExprNode::Integral(_, _)
        | ExprNode::Sum(_, _, _, _)
        | ExprNode::Product_(_, _, _, _)
        | ExprNode::Limit(_, _, _)
        | ExprNode::Series(_, _, _, _)
        | ExprNode::LaplaceTransform(_, _, _)
        | ExprNode::InverseLaplaceTransform(_, _, _)
        | ExprNode::Residue(_, _, _)
        | ExprNode::RootOf(_, _)
        | ExprNode::DSolve(_, _, _)
        | ExprNode::RootSum(_, _, _)
        | ExprNode::ConditionSet(_, _) => {
            return None;
        }
        ExprNode::BoolTrue
        | ExprNode::BoolFalse
        | ExprNode::Gt(_, _)
        | ExprNode::Ge(_, _)
        | ExprNode::Eq_(_, _)
        | ExprNode::Ne(_, _)
        | ExprNode::And(_)
        | ExprNode::Or(_)
        | ExprNode::Not(_)
        | ExprNode::Piecewise(_) => return None,
        ExprNode::EmptySet
        | ExprNode::UniversalSet
        | ExprNode::Interval(_, _, _)
        | ExprNode::FiniteSet(_)
        | ExprNode::SetUnion(_)
        | ExprNode::SetIntersection(_)
        | ExprNode::SetComplement(_, _) => return None,
    }
    Some(())
}

/// Execute a compiled instruction sequence on a stack.
fn execute(instructions: &[Instruction], args: &[f64]) -> f64 {
    let mut stack: Vec<f64> = Vec::with_capacity(32);
    for inst in instructions {
        match inst {
            Instruction::PushConst(v) => stack.push(*v),
            Instruction::PushVar(idx) => {
                stack.push(args.get(*idx).copied().unwrap_or(f64::NAN));
            }
            Instruction::Add => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a + b);
            }
            Instruction::Mul => {
                let b = stack.pop().unwrap_or(1.0);
                let a = stack.pop().unwrap_or(1.0);
                stack.push(a * b);
            }
            Instruction::Sub => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a - b);
            }
            Instruction::Neg => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(-a);
            }
            Instruction::Div => {
                let b = stack.pop().unwrap_or(1.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a / b);
            }
            Instruction::Pow => {
                let b = stack.pop().unwrap_or(1.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.powf(b));
            }
            Instruction::Sin => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.sin());
            }
            Instruction::Cos => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.cos());
            }
            Instruction::Tan => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.tan());
            }
            Instruction::Exp => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.exp());
            }
            Instruction::Ln => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.ln());
            }
            Instruction::Sqrt => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.sqrt());
            }
            Instruction::Cbrt => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.cbrt());
            }
            Instruction::Powi(n) => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.powi(*n));
            }
            Instruction::Abs => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.abs());
            }
            Instruction::Asin => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.asin());
            }
            Instruction::Acos => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.acos());
            }
            Instruction::Atan => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.atan());
            }
            Instruction::Sinh => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.sinh());
            }
            Instruction::Cosh => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.cosh());
            }
            Instruction::Tanh => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.tanh());
            }
            Instruction::Asinh => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.asinh());
            }
            Instruction::Acosh => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.acosh());
            }
            Instruction::Atanh => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.atanh());
            }
            Instruction::Sign => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 });
            }
            Instruction::Heaviside => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(if a > 0.0 {
                    1.0
                } else if a < 0.0 {
                    0.0
                } else {
                    0.5
                });
            }
            Instruction::DiracDelta => {
                // Distributional: pointwise evaluation is always 0
                let _a = stack.pop().unwrap_or(0.0);
                stack.push(0.0);
            }
            Instruction::Atan2 => {
                let x = stack.pop().unwrap_or(0.0);
                let y = stack.pop().unwrap_or(0.0);
                stack.push(y.atan2(x));
            }
            Instruction::Floor => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.floor());
            }
            Instruction::Ceiling => {
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a.ceil());
            }
            Instruction::Min2 => {
                let b = stack.pop().unwrap_or(f64::INFINITY);
                let a = stack.pop().unwrap_or(f64::INFINITY);
                stack.push(a.min(b));
            }
            Instruction::Max2 => {
                let b = stack.pop().unwrap_or(f64::NEG_INFINITY);
                let a = stack.pop().unwrap_or(f64::NEG_INFINITY);
                stack.push(a.max(b));
            }
        }
    }
    stack.pop().unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambdify_polynomial() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let three = a.int(3);
        let x2 = a.pow(x, two);
        let three_x = a.mul(&[three, x]);
        let one = a.int(1);
        let expr = a.add(&[x2, three_x, one]);
        // f(x) = x^2 + 3x + 1
        let f = lambdify(&a, expr, &["x"]).unwrap();
        assert!((f(&[2.0]) - 11.0).abs() < 1e-10); // 4 + 6 + 1 = 11
        assert!((f(&[0.0]) - 1.0).abs() < 1e-10);
        assert!((f(&[-1.0]) - (-1.0)).abs() < 1e-10); // 1 - 3 + 1 = -1
    }

    #[test]
    fn lambdify_trig() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let expr = a.sin(x);
        let f = lambdify(&a, expr, &["x"]).unwrap();
        assert!((f(&[0.0]) - 0.0).abs() < 1e-10);
        assert!((f(&[std::f64::consts::FRAC_PI_2]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn lambdify_two_vars() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let xy = a.mul(&[x, y]);
        let one = a.int(1);
        let expr = a.add(&[xy, one]);
        // f(x,y) = x*y + 1
        let f = lambdify(&a, expr, &["x", "y"]).unwrap();
        assert!((f(&[3.0, 4.0]) - 13.0).abs() < 1e-10);
    }

    #[test]
    fn lambdify_exp_sin() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let sin_x = a.sin(x);
        let expr = a.exp(sin_x);
        let f = lambdify(&a, expr, &["x"]).unwrap();
        let expected = 0.0f64.sin().exp(); // exp(sin(0)) = exp(0) = 1
        assert!((f(&[0.0]) - expected).abs() < 1e-10);
    }

    #[test]
    fn lambdify_pi_e() {
        let a = Arena::new();
        let f = lambdify(&a, a.pi, &[]).unwrap();
        assert!((f(&[]) - std::f64::consts::PI).abs() < 1e-10);

        let f2 = lambdify(&a, a.e_const, &[]).unwrap();
        assert!((f2(&[]) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn lambdify_complex_fails() {
        let a = Arena::new();
        let result = lambdify(&a, a.i_unit, &[]);
        assert!(result.is_none(), "complex should fail lambdify");
    }
}
