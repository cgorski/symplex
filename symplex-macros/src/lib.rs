//! Proc macros for the symplex symbolic mathematics library.
//!
//! This crate provides two proc macros:
//!
//! - [`expr!`] — build symbolic expressions with natural math syntax.
//! - [`rule!`] — define rewrite rules with pattern/template syntax.
//!
//! This crate should not be used directly.  Instead, depend on `symplex`
//! which re-exports these macros.

mod parse;

use parse::{BinOp, DimMacroInput, EqMacroInput, ExprMacroInput, MathExpr, MatrixMacroInput, RuleMacroInput};
use parse::{KNOWN_FUNCTIONS, is_known_constant, is_known_function};

use proc_macro::TokenStream;
use proc_macro2::Span;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::Ident;

// ═══════════════════════════════════════════════════════════════════════════
// expr! macro
// ═══════════════════════════════════════════════════════════════════════════

/// Build a symbolic expression using natural math syntax.
///
/// All identifiers are treated as existing Rust variables of type `Ex`
/// (or `&Ex`).  They are automatically borrowed with `&`.
///
/// # Syntax
///
/// - Operators: `+`, `-`, `*`, `/`, `^` (power)
/// - Functions: `sin(x)`, `cos(x)`, `tan(x)`, `asin(x)`, `acos(x)`, `atan(x)`,
///   `sinh(x)`, `cosh(x)`, `tanh(x)`, `exp(x)`, `ln(x)`, `sqrt(x)`, `abs(x)`
/// - Parentheses for grouping
/// - Integer literals
/// - Unary minus: `-x`
///
/// # Constants
///
/// The following identifiers are recognized as mathematical constants:
/// - `pi`, `Pi`, `PI` → π
/// - `E` → Euler's number e
/// - `I` → imaginary unit i
/// - `oo`, `inf` → positive infinity
///
/// # Rationals
///
/// `1/2`, `3/4`, etc. produce exact rational numbers (not Rust integer division).
///
/// # Multi-argument functions
///
/// - `log(x, base)` → logarithm of x with given base
/// - `diff(f, x)` → formal derivative of f with respect to x (unevaluated)
/// - `factorial(n)` → n!
/// - `binomial(n, k)` or `C(n, k)` → binomial coefficient C(n,k)
///
/// # Examples
///
/// ```ignore
/// use symplex::prelude::*;
/// use symplex::expr;
///
/// let ctx = Context::new();
/// syms!(ctx; x, y);
/// let e = expr!(x^2 + 2*x + 1);
/// let f = expr!(sin(x)^2 + cos(x)^2);
/// ```
///
/// # Limitations
///
/// - Implicit multiplication (`2x`) is not supported.  Write `2*x`.
/// - Only the listed built-in functions are recognised.
#[proc_macro]
pub fn expr(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as ExprMacroInput);
    match generate_expr(&input.expr) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Generate Rust code for an `expr!` invocation.
///
/// Every identifier is emitted as `(&ident)`.  Integer literals stay
/// as `i64`.  `^` becomes `.powi(n)` for integer RHS or `.pow(&rhs)`
/// for expression RHS.  Known function names become method calls.
fn generate_expr(expr: &MathExpr) -> syn::Result<TokenStream2> {
    match expr {
        MathExpr::Int(n, _span) => Ok(quote! { #n }),

        MathExpr::Ident(id) => {
            let name = id.to_string();
            match name.as_str() {
                "pi" | "Pi" | "PI" => Ok(quote! { __ctx.pi() }),
                "E" => Ok(quote! { __ctx.e() }),
                "I" => Ok(quote! { __ctx.i_unit() }),
                "oo" | "inf" => Ok(quote! { __ctx.infinity() }),
                _ => Ok(quote! { (&#id) }),
            }
        }

        MathExpr::Neg(inner) => {
            let inner_code = generate_expr(inner)?;
            Ok(quote! { (-(#inner_code)) })
        }

        MathExpr::LogicalNot(inner) => {
            let inner_code = generate_expr(inner)?;
            Ok(quote! { (#inner_code).not() })
        }

        MathExpr::BinOp { op, lhs, rhs } => {
            match op {
                BinOp::Pow => {
                    let lhs_code = generate_expr(lhs)?;
                    // If RHS is an integer literal, use .powi(n).
                    // If RHS is Neg(Int), use .powi(-n).
                    if let Some(n) = rhs.as_int() {
                        Ok(quote! { (#lhs_code).powi(#n) })
                    } else if let MathExpr::Neg(inner) = rhs.as_ref() {
                        if let Some(n) = inner.as_int() {
                            let neg_n = -n;
                            Ok(quote! { (#lhs_code).powi(#neg_n) })
                        } else {
                            let rhs_code = generate_expr(rhs)?;
                            Ok(quote! { (#lhs_code).pow(&(#rhs_code)) })
                        }
                    } else {
                        let rhs_code = generate_expr(rhs)?;
                        Ok(quote! { (#lhs_code).pow(&(#rhs_code)) })
                    }
                }
                BinOp::Div => {
                    if let (Some(p), Some(q)) = (lhs.as_int(), rhs.as_int()) {
                        if q == 0 {
                            return Err(syn::Error::new(
                                Span::call_site(),
                                "division by zero in expr!()",
                            ));
                        }
                        return Ok(quote! { __ctx.rational(#p, #q) });
                    }
                    // Handle -Int / Int → rational(-n, q).
                    // Due to precedence, `-1/2` parses as `Neg(1) / 2`.
                    // Without this, the codegen emits `(-(1)) / (2)` which
                    // is Rust integer division yielding 0.
                    if let MathExpr::Neg(inner_lhs) = lhs.as_ref() {
                        if let (Some(p), Some(q)) = (inner_lhs.as_int(), rhs.as_int()) {
                            if q == 0 {
                                return Err(syn::Error::new(
                                    Span::call_site(),
                                    "division by zero in expr!()",
                                ));
                            }
                            let neg_p = -p;
                            return Ok(quote! { __ctx.rational(#neg_p, #q) });
                        }
                    }
                    let lhs_code = generate_expr(lhs)?;
                    let rhs_code = generate_expr(rhs)?;
                    Ok(quote! { ((#lhs_code) / (#rhs_code)) })
                }
                BinOp::Gt => {
                    let lhs_code = generate_expr_as_ex(lhs)?;
                    let rhs_code = generate_expr_as_ex(rhs)?;
                    Ok(quote! { (#lhs_code).gt(&(#rhs_code)) })
                }
                BinOp::Lt => {
                    let lhs_code = generate_expr_as_ex(lhs)?;
                    let rhs_code = generate_expr_as_ex(rhs)?;
                    Ok(quote! { (#lhs_code).lt(&(#rhs_code)) })
                }
                BinOp::Ge => {
                    let lhs_code = generate_expr_as_ex(lhs)?;
                    let rhs_code = generate_expr_as_ex(rhs)?;
                    Ok(quote! { (#lhs_code).ge(&(#rhs_code)) })
                }
                BinOp::Le => {
                    let lhs_code = generate_expr_as_ex(lhs)?;
                    let rhs_code = generate_expr_as_ex(rhs)?;
                    Ok(quote! { (#lhs_code).le(&(#rhs_code)) })
                }
                BinOp::EqEq => {
                    let lhs_code = generate_expr_as_ex(lhs)?;
                    let rhs_code = generate_expr_as_ex(rhs)?;
                    Ok(quote! { (#lhs_code).eq_expr(&(#rhs_code)) })
                }
                BinOp::Ne => {
                    let lhs_code = generate_expr_as_ex(lhs)?;
                    let rhs_code = generate_expr_as_ex(rhs)?;
                    Ok(quote! { (#lhs_code).ne_expr(&(#rhs_code)) })
                }
                BinOp::AndAnd => {
                    let lhs_code = generate_expr(lhs)?;
                    let rhs_code = generate_expr(rhs)?;
                    Ok(quote! { (#lhs_code).and(&(#rhs_code)) })
                }
                BinOp::OrOr => {
                    let lhs_code = generate_expr(lhs)?;
                    let rhs_code = generate_expr(rhs)?;
                    Ok(quote! { (#lhs_code).or(&(#rhs_code)) })
                }
                _ => {
                    let lhs_code = generate_expr(lhs)?;
                    let rhs_code = generate_expr(rhs)?;
                    let op_token = match op {
                        BinOp::Add => quote! { + },
                        BinOp::Sub => quote! { - },
                        BinOp::Mul => quote! { * },
                        _ => unreachable!(),
                    };
                    Ok(quote! { ((#lhs_code) #op_token (#rhs_code)) })
                }
            }
        }

        MathExpr::Func { name, span, args } => {
            // Multi-argument functions
            if name == "log" && args.len() == 2 {
                let arg_code = generate_expr(&args[0])?;
                // The base must be an Ex; bare integer literals from
                // generate_expr would be i64, so promote them.
                let base_code = generate_expr_as_ex(&args[1])?;
                return Ok(quote! { (#arg_code).log(&(#base_code)) });
            }

            // diff(f, x) → formal derivative node (unevaluated)
            if name == "diff" && args.len() == 2 {
                let f_code = generate_expr(&args[0])?;
                let var_code = generate_expr(&args[1])?;
                return Ok(quote! { (#f_code).formal_diff(&(#var_code)) });
            }

            // factorial(n) → n.factorial()
            if name == "factorial" && args.len() == 1 {
                let arg_code = generate_expr_as_ex(&args[0])?;
                return Ok(quote! { (#arg_code).factorial() });
            }

            // binomial(n, k) or C(n, k) → n.binomial(&k)
            if (name == "binomial" || name == "C") && args.len() == 2 {
                let n_code = generate_expr_as_ex(&args[0])?;
                let k_code = generate_expr_as_ex(&args[1])?;
                return Ok(quote! { (#n_code).binomial(&(#k_code)) });
            }

            // atan2(y, x) → y.atan2(&x)
            if name == "atan2" && args.len() == 2 {
                let y_code = generate_expr_as_ex(&args[0])?;
                let x_code = generate_expr_as_ex(&args[1])?;
                return Ok(quote! { (#y_code).atan2(&(#x_code)) });
            }

            // rising_factorial(x, n) → x.rising_factorial(&n)
            if name == "rising_factorial" && args.len() == 2 {
                let x_code = generate_expr_as_ex(&args[0])?;
                let n_code = generate_expr_as_ex(&args[1])?;
                return Ok(quote! { (#x_code).rising_factorial(&(#n_code)) });
            }

            // falling_factorial(x, n) → x.falling_factorial(&n)
            if name == "falling_factorial" && args.len() == 2 {
                let x_code = generate_expr_as_ex(&args[0])?;
                let n_code = generate_expr_as_ex(&args[1])?;
                return Ok(quote! { (#x_code).falling_factorial(&(#n_code)) });
            }

            // beta(a, b) → a.beta(&b)
            if name == "beta" && args.len() == 2 {
                let a_code = generate_expr_as_ex(&args[0])?;
                let b_code = generate_expr_as_ex(&args[1])?;
                return Ok(quote! { (#a_code).beta(&(#b_code)) });
            }

            // min(a, b) → a.min_with(&b)
            if name == "min" && args.len() == 2 {
                let a = generate_expr_as_ex(&args[0])?;
                let b = generate_expr_as_ex(&args[1])?;
                return Ok(quote! { (#a).min_with(&(#b)) });
            }

            // max(a, b) → a.max_with(&b)
            if name == "max" && args.len() == 2 {
                let a = generate_expr_as_ex(&args[0])?;
                let b = generate_expr_as_ex(&args[1])?;
                return Ok(quote! { (#a).max_with(&(#b)) });
            }

            if !is_known_function(name)
                && ![
                    "log",
                    "diff",
                    "factorial",
                    "binomial",
                    "C",
                    "atan2",
                    "rising_factorial",
                    "falling_factorial",
                    "beta",
                    "min",
                    "max",
                ]
                .contains(&name.as_str())
            {
                return Err(syn::Error::new(
                    *span,
                    format!(
                        "unknown function '{}' in expr!(). Supported: {}, log, diff, factorial, binomial, C, atan2, rising_factorial, falling_factorial, beta, min, max",
                        name,
                        KNOWN_FUNCTIONS.join(", ")
                    ),
                ));
            }
            if args.len() != 1 {
                return Err(syn::Error::new(
                    *span,
                    format!("{}() takes exactly 1 argument in expr!()", name),
                ));
            }
            let arg_code = generate_expr_as_ex(&args[0])?;
            let method = match name.as_str() {
                "sin" => quote! { sin },
                "cos" => quote! { cos },
                "tan" => quote! { tan },
                "asin" => quote! { asin },
                "acos" => quote! { acos },
                "atan" => quote! { atan },
                "sinh" => quote! { sinh },
                "cosh" => quote! { cosh },
                "tanh" => quote! { tanh },
                "asinh" => quote! { asinh },
                "acosh" => quote! { acosh },
                "atanh" => quote! { atanh },
                "exp" => quote! { exp },
                "ln" => quote! { ln },
                "sqrt" => quote! { sqrt },
                "cbrt" => quote! { cbrt },
                "abs" => quote! { abs },
                "sign" => quote! { sign },
                "floor" => quote! { floor },
                "ceiling" => quote! { ceiling },
                // Wave A: reciprocal trig/hyp
                "sec" => quote! { sec },
                "csc" => quote! { csc },
                "cot" => quote! { cot },
                "acot" => quote! { acot },
                "asec" => quote! { asec },
                "acsc" => quote! { acsc },
                "coth" => quote! { coth },
                "sech" => quote! { sech },
                "csch" => quote! { csch },
                "acoth" => quote! { acoth },
                "asech" => quote! { asech },
                "acsch" => quote! { acsch },
                "sinc" => quote! { sinc },
                // Wave O: complex
                "arg" => quote! { arg },
                "conjugate" => quote! { conjugate },
                // Wave R: combinatorial (1-arg)
                "fibonacci" => quote! { fibonacci },
                "lucas" => quote! { lucas },
                "catalan_number" => quote! { catalan_number },
                "bell" => quote! { bell },
                "euler_number" => quote! { euler_number },
                "harmonic" => quote! { harmonic },
                "subfactorial" => quote! { subfactorial },
                "factorial2" => quote! { factorial2 },
                "bernoulli_number" => quote! { bernoulli_number },
                // Wave S: special elementary
                "heaviside" => quote! { heaviside },
                "dirac_delta" => quote! { dirac_delta },
                "lambertw" => quote! { lambertw },
                // Wave J: special functions (1-arg)
                "gamma" => quote! { gamma },
                "log_gamma" => quote! { log_gamma },
                "digamma" => quote! { digamma },
                "erf" => quote! { erf },
                "erfc" => quote! { erfc },
                _ => unreachable!(),
            };
            Ok(quote! { (#arg_code).#method() })
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// dim! macro
// ═══════════════════════════════════════════════════════════════════════════

/// Build a dimension-checked physical quantity using natural math syntax.
///
/// # Syntax
///
/// ```ignore
/// dim!(OutputType: math_expression)
/// ```
///
/// The macro parses the math expression (same syntax as [`expr!`]), generates
/// code that operates on `Qty<D>` values (preserving compile-time dimension
/// tracking), and converts the result to `OutputType` via [`FromDimExpr`].
///
/// If the computed dimension doesn't match `OutputType`, the compiler emits
/// a clear error message.
///
/// # How it works
///
/// - Identifiers refer to named quantity variables (e.g. `Mass`, `Length`).
///   They are cloned and converted to `Qty<D>` via `.as_qty()`.
/// - Integer literals become `Dimensionless::constant(n).as_qty()`.
/// - `+`, `-`, `*`, `/` use the `Qty` operator impls which track dimensions
///   at the type level.
/// - `x^n` for small integer `n` (0–8) expands to repeated multiplication,
///   preserving type-level dimension tracking. For larger or non-literal
///   exponents, the macro falls back to extracting the inner `Ex` and using
///   `.powi()` / `.pow()`, which loses dimension tracking (treats result as
///   dimensionless).
/// - Functions like `sin`, `cos`, `exp`, `ln` extract the inner `Ex`, call
///   the method, and wrap the result as `Dimensionless`.
///
/// # Examples
///
/// ```ignore
/// use symplex::prelude::*;
/// use symplex::units::*;
///
/// let m = Mass::symbol("m");
/// let g = Acceleration::symbol("g");
/// let h = Length::symbol("h");
/// let pe = symplex::dim!(Energy: m * g * h);
/// ```
///
/// [`FromDimExpr`]: ::symplex::units::qty::FromDimExpr
#[proc_macro]
pub fn dim(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as DimMacroInput);
    let output_type = &input.output_type;
    match generate_dim_expr(&input.expr) {
        Ok(expr_tokens) => quote! {
            <#output_type as ::symplex::units::qty::FromDimExpr<_>>::from_dim_expr(#expr_tokens)
        }
        .into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Generate Rust code for a `dim!` invocation.
///
/// Each `MathExpr` node is translated to code producing a `Qty<D>`,
/// where the dimension `D` is computed at the type level by Rust's
/// type system via the `Qty` arithmetic operator impls.
fn generate_dim_expr(expr: &MathExpr) -> syn::Result<TokenStream2> {
    match expr {
        MathExpr::Int(n, _span) => Ok(quote! {
            ::symplex::units::Dimensionless::constant(#n).as_qty()
        }),

        MathExpr::Ident(id) => {
            let name = id.to_string();
            match name.as_str() {
                "pi" | "Pi" | "PI" => Ok(quote! {
                    ::symplex::units::Dimensionless::from_ex(__ctx.pi()).as_qty()
                }),
                "E" => Ok(quote! {
                    ::symplex::units::Dimensionless::from_ex(__ctx.e()).as_qty()
                }),
                _ => Ok(quote! { (#id).clone().as_qty() }),
            }
        }

        MathExpr::Neg(inner) => {
            let inner_code = generate_dim_expr(inner)?;
            Ok(quote! { (-(#inner_code)) })
        }

        MathExpr::LogicalNot(_) => Err(syn::Error::new(
            Span::call_site(),
            "logical NOT (!) is not supported in dim!()",
        )),

        MathExpr::BinOp { op, lhs, rhs } => match op {
            BinOp::Add => {
                let l = generate_dim_expr(lhs)?;
                let r = generate_dim_expr(rhs)?;
                Ok(quote! { ((#l) + (#r)) })
            }
            BinOp::Sub => {
                let l = generate_dim_expr(lhs)?;
                let r = generate_dim_expr(rhs)?;
                Ok(quote! { ((#l) - (#r)) })
            }
            BinOp::Mul => {
                let l = generate_dim_expr(lhs)?;
                let r = generate_dim_expr(rhs)?;
                Ok(quote! { ((#l) * (#r)) })
            }
            BinOp::Div => {
                // int / int → exact rational (dimensionless)
                if let (Some(p), Some(q)) = (lhs.as_int(), rhs.as_int()) {
                    if q == 0 {
                        return Err(syn::Error::new(
                            Span::call_site(),
                            "division by zero in dim!()",
                        ));
                    }
                    return Ok(quote! {
                        ::symplex::units::Dimensionless::rational(#p, #q).as_qty()
                    });
                }
                // -int / int → rational(-n, q)
                if let MathExpr::Neg(inner_lhs) = lhs.as_ref() {
                    if let (Some(p), Some(q)) = (inner_lhs.as_int(), rhs.as_int()) {
                        if q == 0 {
                            return Err(syn::Error::new(
                                Span::call_site(),
                                "division by zero in dim!()",
                            ));
                        }
                        let neg_p = -p;
                        return Ok(quote! {
                            ::symplex::units::Dimensionless::rational(#neg_p, #q).as_qty()
                        });
                    }
                }
                let l = generate_dim_expr(lhs)?;
                let r = generate_dim_expr(rhs)?;
                Ok(quote! { ((#l) / (#r)) })
            }
            BinOp::Pow => {
                // Integer exponents: expand to repeated multiplication for
                // type-level dimension tracking.
                if let Some(n) = rhs.as_int() {
                    return generate_dim_pow(lhs, n);
                }
                // Negative integer exponent: x^(-n) = 1 / x^n
                if let MathExpr::Neg(inner_rhs) = rhs.as_ref() {
                    if let Some(n) = inner_rhs.as_int() {
                        let pow_code = generate_dim_pow(lhs, n)?;
                        return Ok(quote! {
                            (::symplex::units::Dimensionless::constant(1).as_qty() / (#pow_code))
                        });
                    }
                }
                // Non-integer exponent: fall back to inner Ex operations.
                // This loses dimension tracking — result is Dimensionless.
                let b = generate_dim_expr(lhs)?;
                let e = generate_dim_expr(rhs)?;
                Ok(quote! {
                    ::symplex::units::Dimensionless::from_ex(
                        (#b).into_inner().pow(&#e.into_inner())
                    ).as_qty()
                })
            }
            _ => Err(syn::Error::new(
                Span::call_site(),
                format!("operator {:?} is not supported in dim!()", op),
            )),
        },

        MathExpr::Func { name, span, args } => {
            // Transcendental functions produce dimensionless results.
            // Extract inner Ex, call the method, wrap as Dimensionless.
            let func_str = name.as_str();
            if args.len() == 1 {
                let arg = generate_dim_expr(&args[0])?;
                let method = match func_str {
                    "sin" => quote! { sin },
                    "cos" => quote! { cos },
                    "tan" => quote! { tan },
                    "asin" => quote! { asin },
                    "acos" => quote! { acos },
                    "atan" => quote! { atan },
                    "sinh" => quote! { sinh },
                    "cosh" => quote! { cosh },
                    "tanh" => quote! { tanh },
                    "exp" => quote! { exp },
                    "ln" => quote! { ln },
                    "sqrt" => quote! { sqrt },
                    "abs" => quote! { abs },
                    _ => {
                        return Err(syn::Error::new(
                            *span,
                            format!("dim!: unsupported function '{}'", func_str),
                        ));
                    }
                };
                Ok(quote! {
                    ::symplex::units::Dimensionless::from_ex(
                        (#arg).into_inner().#method()
                    ).as_qty()
                })
            } else {
                Err(syn::Error::new(
                    *span,
                    format!(
                        "dim!: function '{}' with {} args is not supported",
                        func_str,
                        args.len()
                    ),
                ))
            }
        }
    }
}

/// Generate code for `base^n` where `n` is a known integer literal.
///
/// For small `n` (0–8), this expands to repeated multiplication so the
/// type system tracks the resulting dimension.  For larger `n`, it falls
/// back to `.powi()` on the inner `Ex` (losing dimension tracking).
fn generate_dim_pow(base: &MathExpr, n: i64) -> syn::Result<TokenStream2> {
    if n == 0 {
        return Ok(quote! { ::symplex::units::Dimensionless::constant(1).as_qty() });
    }
    if n == 1 {
        return generate_dim_expr(base);
    }
    if n >= 2 && n <= 8 {
        // Expand x^n = x * x * ... * x  (n factors).
        // Each factor is an independent evaluation of `base` so the
        // type-level dimension products compose correctly.
        let mut factors = Vec::new();
        for _ in 0..n {
            factors.push(generate_dim_expr(base)?);
        }
        let mut result = factors.remove(0);
        for factor in factors {
            result = quote! { ((#result) * (#factor)) };
        }
        return Ok(result);
    }
    // n > 8: fall back to extracting inner Ex and using powi.
    // Dimension tracking is lost — the result is treated as Dimensionless.
    let base_code = generate_dim_expr(base)?;
    Ok(quote! {
        ::symplex::units::Dimensionless::from_ex(
            (#base_code).into_inner().powi(#n)
        ).as_qty()
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// rule! macro
// ═══════════════════════════════════════════════════════════════════════════

/// Define a rewrite rule with pattern/template syntax.
///
/// # Syntax
///
/// ```ignore
/// rule!(arena, "rule_name", LHS_PATTERN => RHS_TEMPLATE)
/// rule!(arena, "rule_name", LHS_PATTERN => RHS_TEMPLATE if condition_expr)
/// ```
///
/// - `arena` — an expression of type `&mut Arena`.
/// - `"rule_name"` — a string literal used for tracing.
/// - Identifiers ending in `_` are **wilds** (pattern variables):
///   they match any sub-expression and bind it.
/// - Known constants: `pi`, `E`, `I`, `oo`, `nan`, `zoo`.
/// - Integer literals: `0`, `1`, `2`, `-3`, etc.
/// - Functions: `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `abs`.
/// - `=>` separates the pattern (LHS) from the template (RHS).
/// - An optional `if <expr>` after the RHS specifies a condition
///   function: `fn(&Arena, &Substitution) -> bool`.
///
/// # Examples
///
/// ```ignore
/// use symplex::prelude::*;
/// use symplex::rule;
///
/// let rules = vec![
///     rule!(arena, "pythagorean", sin(w_)^2 + cos(w_)^2 => 1),
///     rule!(arena, "exp_ln", exp(ln(w_)) => w_),
///     rule!(arena, "ln_exp", ln(exp(w_)) => w_),
/// ];
/// ```
///
/// # Limitations
///
/// - Only wilds (`w_`), integers, known constants, and known functions
///   are allowed.  Bare identifiers that are not wilds or constants
///   produce a compile error.
#[proc_macro]
pub fn rule(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as RuleMacroInput);
    match generate_rule(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Generate Rust code for a `rule!` invocation.
///
/// All sub-expressions are bound to flat `let __tN = arena.method(...)`
/// temporaries to avoid double-mutable-borrow errors.
fn generate_rule(input: &RuleMacroInput) -> syn::Result<TokenStream2> {
    let arena = &input.arena;
    let name = &input.name;

    // Collect wilds from LHS.
    let lhs_wilds = input.lhs.collect_wilds();

    // Validate that all RHS wilds also appear in LHS.
    for w in input.rhs.collect_wilds() {
        if !lhs_wilds.iter().any(|existing| existing == &w) {
            return Err(syn::Error::new_spanned(
                &input.name,
                format!(
                    "wild '{}' appears in RHS but not in LHS — it will never be bound by matching",
                    w
                ),
            ));
        }
    }

    let all_wilds = lhs_wilds;

    let mut codegen = RuleCodeGen {
        arena: arena.clone(),
        temp_counter: 0,
        bindings: Vec::new(),
        wild_expr_idents: Vec::new(),
        wild_wid_idents: Vec::new(),
        wild_names: Vec::new(),
    };

    // Generate wild declarations.
    for wild in &all_wilds {
        let wild_name = wild.to_string();
        let expr_ident = format_ident!("__wild_expr_{}", wild_name);
        let wid_ident = format_ident!("__wild_wid_{}", wild_name);
        codegen.bindings.push(quote! {
            let (#expr_ident, #wid_ident) = #arena.wild();
        });
        codegen.wild_expr_idents.push(expr_ident);
        codegen.wild_wid_idents.push(wid_ident);
        codegen.wild_names.push(wild.clone());
    }

    // Generate LHS expression tree.
    let lhs_temp = codegen.generate_arena_expr(&input.lhs)?;

    // Generate RHS expression tree.
    let rhs_temp = codegen.generate_arena_expr(&input.rhs)?;

    // Build the wilds map.
    let wild_inserts: Vec<TokenStream2> = codegen
        .wild_names
        .iter()
        .zip(
            codegen
                .wild_expr_idents
                .iter()
                .zip(codegen.wild_wid_idents.iter()),
        )
        .map(|(_, (expr_id, wid_id))| {
            quote! { __wilds.insert(#expr_id, #wid_id); }
        })
        .collect();

    let bindings = &codegen.bindings;

    let condition_code = if let Some(cond) = &input.condition {
        quote! { Some(#cond) }
    } else {
        quote! { None }
    };

    Ok(quote! {
        {
            #(#bindings)*

            let mut __wilds = ::symplex::__macro_support::FxHashMap::default();
            #(#wild_inserts)*

            ::symplex::__macro_support::Rule {
                name: #name,
                pattern: ::symplex::__macro_support::Pattern {
                    root: #lhs_temp,
                    wilds: __wilds,
                },
                template: #rhs_temp,
                condition: #condition_code,
            }
        }
    })
}

/// Code generator for `rule!` that emits flat let-bindings for every
/// sub-expression, avoiding double-mutable-borrow issues.
struct RuleCodeGen {
    arena: Ident,
    temp_counter: usize,
    bindings: Vec<TokenStream2>,
    wild_expr_idents: Vec<Ident>,
    wild_wid_idents: Vec<Ident>,
    wild_names: Vec<Ident>,
}

impl RuleCodeGen {
    /// Allocate a fresh temporary name.
    fn fresh_temp(&mut self) -> Ident {
        let id = format_ident!("__t{}", self.temp_counter);
        self.temp_counter += 1;
        id
    }

    /// Generate arena method calls for a MathExpr, returning the
    /// identifier of the temporary holding the final ExprId.
    fn generate_arena_expr(&mut self, expr: &MathExpr) -> syn::Result<Ident> {
        let arena = self.arena.clone();

        match expr {
            MathExpr::Int(n, _) => {
                let temp = self.fresh_temp();
                match *n {
                    0 => {
                        self.bindings.push(quote! { let #temp = #arena.zero(); });
                    }
                    1 => {
                        self.bindings.push(quote! { let #temp = #arena.one(); });
                    }
                    -1 => {
                        self.bindings.push(quote! { let #temp = #arena.neg_one(); });
                    }
                    _ => {
                        self.bindings.push(quote! { let #temp = #arena.int(#n); });
                    }
                }
                Ok(temp)
            }

            MathExpr::Ident(id) => {
                let name = id.to_string();

                // Wild: return the pre-declared wild expression id.
                if name.ends_with('_') {
                    for (i, wn) in self.wild_names.iter().enumerate() {
                        if wn == id {
                            return Ok(self.wild_expr_idents[i].clone());
                        }
                    }
                    return Err(syn::Error::new(id.span(), format!("unknown wild '{name}'")));
                }

                // Known constant.
                if is_known_constant(&name) {
                    let temp = self.fresh_temp();
                    let access = match name.as_str() {
                        "pi" => quote! { #arena.pi() },
                        "E" => quote! { #arena.e_const() },
                        "I" => quote! { #arena.i_unit() },
                        "oo" => quote! { #arena.infinity() },
                        "nan" => quote! { #arena.nan() },
                        "zoo" => quote! { #arena.complex_infinity() },
                        _ => unreachable!(),
                    };
                    self.bindings.push(quote! { let #temp = #access; });
                    return Ok(temp);
                }

                // Unknown identifier — error.
                Err(syn::Error::new(
                    id.span(),
                    format!(
                        "unknown identifier '{}' in rule!(). \
                         Use '{}' for a wild, or a known constant (pi, E, I, oo, nan, zoo), \
                         or an integer literal.",
                        name,
                        format!("{}_", name)
                    ),
                ))
            }

            MathExpr::Neg(inner) => {
                let inner_temp = self.generate_arena_expr(inner)?;
                let temp = self.fresh_temp();
                self.bindings
                    .push(quote! { let #temp = #arena.neg(#inner_temp); });
                Ok(temp)
            }

            MathExpr::LogicalNot(inner) => {
                let inner_temp = self.generate_arena_expr(inner)?;
                let temp = self.fresh_temp();
                self.bindings
                    .push(quote! { let #temp = #arena.not(#inner_temp); });
                Ok(temp)
            }

            MathExpr::BinOp { op, lhs, rhs } => {
                let lhs_temp = self.generate_arena_expr(lhs)?;
                let rhs_temp = self.generate_arena_expr(rhs)?;
                let temp = self.fresh_temp();

                let call = match op {
                    BinOp::Add => quote! { #arena.add(&[#lhs_temp, #rhs_temp]) },
                    BinOp::Sub => quote! { #arena.sub(#lhs_temp, #rhs_temp) },
                    BinOp::Mul => quote! { #arena.mul(&[#lhs_temp, #rhs_temp]) },
                    BinOp::Div => quote! { #arena.div(#lhs_temp, #rhs_temp) },
                    BinOp::Pow => quote! { #arena.pow(#lhs_temp, #rhs_temp) },
                    BinOp::Gt => quote! { #arena.gt(#lhs_temp, #rhs_temp) },
                    BinOp::Lt => quote! { #arena.gt(#rhs_temp, #lhs_temp) },
                    BinOp::Ge => quote! { #arena.ge(#lhs_temp, #rhs_temp) },
                    BinOp::Le => quote! { #arena.ge(#rhs_temp, #lhs_temp) },
                    BinOp::EqEq => quote! { #arena.eq_(#lhs_temp, #rhs_temp) },
                    BinOp::Ne => quote! { #arena.ne_(#lhs_temp, #rhs_temp) },
                    BinOp::AndAnd => quote! { #arena.and(&[#lhs_temp, #rhs_temp]) },
                    BinOp::OrOr => quote! { #arena.or(&[#lhs_temp, #rhs_temp]) },
                };

                self.bindings.push(quote! { let #temp = #call; });
                Ok(temp)
            }

            MathExpr::Func { name, span, args } => {
                if !is_known_function(name) {
                    return Err(syn::Error::new(
                        *span,
                        format!(
                            "unknown function '{}' in rule!(). Supported: {}",
                            name,
                            KNOWN_FUNCTIONS.join(", ")
                        ),
                    ));
                }

                // Binary functions: beta(a, b), atan2(y, x)
                if name == "beta" && args.len() == 2 {
                    let a_temp = self.generate_arena_expr(&args[0])?;
                    let b_temp = self.generate_arena_expr(&args[1])?;
                    let temp = self.fresh_temp();
                    self.bindings
                        .push(quote! { let #temp = #arena.beta(#a_temp, #b_temp); });
                    return Ok(temp);
                }
                if name == "atan2" && args.len() == 2 {
                    let y_temp = self.generate_arena_expr(&args[0])?;
                    let x_temp = self.generate_arena_expr(&args[1])?;
                    let temp = self.fresh_temp();
                    self.bindings
                        .push(quote! { let #temp = #arena.atan2(#y_temp, #x_temp); });
                    return Ok(temp);
                }

                if args.len() != 1 {
                    return Err(syn::Error::new(
                        *span,
                        format!("{}() takes exactly 1 argument in rule!()", name),
                    ));
                }

                let arg_temp = self.generate_arena_expr(&args[0])?;
                let temp = self.fresh_temp();

                let call = match name.as_str() {
                    // Core trig
                    "sin" => quote! { #arena.sin(#arg_temp) },
                    "cos" => quote! { #arena.cos(#arg_temp) },
                    "tan" => quote! { #arena.tan(#arg_temp) },
                    "asin" => quote! { #arena.asin(#arg_temp) },
                    "acos" => quote! { #arena.acos(#arg_temp) },
                    "atan" => quote! { #arena.atan(#arg_temp) },
                    "sinh" => quote! { #arena.sinh(#arg_temp) },
                    "cosh" => quote! { #arena.cosh(#arg_temp) },
                    "tanh" => quote! { #arena.tanh(#arg_temp) },
                    "asinh" => quote! { #arena.asinh(#arg_temp) },
                    "acosh" => quote! { #arena.acosh(#arg_temp) },
                    "atanh" => quote! { #arena.atanh(#arg_temp) },
                    // Exp/log/root
                    "exp" => quote! { #arena.exp(#arg_temp) },
                    "ln" => quote! { #arena.ln(#arg_temp) },
                    "sqrt" => quote! { #arena.sqrt(#arg_temp) },
                    "cbrt" => quote! { #arena.cbrt(#arg_temp) },
                    "abs" => quote! { #arena.abs(#arg_temp) },
                    "sign" => quote! { #arena.sign(#arg_temp) },
                    // Wave B: Floor, Ceiling
                    "floor" => quote! { #arena.floor(#arg_temp) },
                    "ceiling" => quote! { #arena.ceiling(#arg_temp) },
                    // Wave J: Special functions
                    "gamma" => quote! { #arena.gamma(#arg_temp) },
                    "log_gamma" => quote! { #arena.log_gamma(#arg_temp) },
                    "digamma" => quote! { #arena.digamma(#arg_temp) },
                    "erf" => quote! { #arena.erf(#arg_temp) },
                    "erfc" => quote! { #arena.erfc(#arg_temp) },
                    // Wave S/δ: Heaviside, DiracDelta, LambertW
                    "heaviside" => quote! { #arena.heaviside(#arg_temp) },
                    "dirac_delta" => quote! { #arena.dirac_delta(#arg_temp) },
                    "lambertw" => quote! { #arena.lambertw(#arg_temp) },
                    // Wave R: Combinatorial (1-arg, Apply-based but have arena methods)
                    "fibonacci" => quote! { #arena.fibonacci(#arg_temp) },
                    "lucas" => quote! { #arena.lucas(#arg_temp) },
                    "catalan_number" => quote! { #arena.catalan_number(#arg_temp) },
                    "bell" => quote! { #arena.bell(#arg_temp) },
                    "euler_number" => quote! { #arena.euler_number(#arg_temp) },
                    "harmonic" => quote! { #arena.harmonic(#arg_temp) },
                    "subfactorial" => quote! { #arena.subfactorial(#arg_temp) },
                    "factorial2" => quote! { #arena.factorial2(#arg_temp) },
                    "bernoulli_number" => quote! { #arena.bernoulli_number(#arg_temp) },
                    other => {
                        return Err(syn::Error::new(
                            *span,
                            format!(
                                "function '{}' is recognised but not yet supported in rule!()",
                                other
                            ),
                        ));
                    }
                };

                self.bindings.push(quote! { let #temp = #call; });
                Ok(temp)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// matrix! macro
// ═══════════════════════════════════════════════════════════════════════════

/// Build a symbolic matrix using natural math syntax.
///
/// # Examples
///
/// ```ignore
/// use symplex::prelude::*;
/// use symplex::matrix;
///
/// let x = symplex::var("x");
/// let m = matrix![[x, 1], [0, x]];
/// ```
#[proc_macro]
pub fn matrix(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as MatrixMacroInput);
    match generate_matrix(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn generate_matrix(input: &MatrixMacroInput) -> syn::Result<TokenStream2> {
    let mut row_codes = Vec::new();
    for row in &input.rows {
        let mut cell_codes = Vec::new();
        for cell in row {
            let cell_expr = generate_expr_as_ex(cell)?;
            cell_codes.push(quote! { #cell_expr });
        }
        row_codes.push(quote! { vec![#(#cell_codes),*] });
    }
    Ok(quote! {
        ::symplex::matrix::Matrix::new(vec![#(#row_codes),*])
            .expect("matrix! macro: invalid literal data")
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// eq! macro
// ═══════════════════════════════════════════════════════════════════════════

/// Build a symbolic equation using natural math syntax.
///
/// # Examples
///
/// ```ignore
/// use symplex::prelude::*;
/// use symplex::eq;
///
/// let x = symplex::var("x");
/// let equation = eq!(x^2 + x = 6);
/// ```
#[proc_macro]
pub fn eq(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as EqMacroInput);
    match generate_eq(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn generate_eq(input: &EqMacroInput) -> syn::Result<TokenStream2> {
    let lhs_code = generate_expr_as_ex(&input.lhs)?;
    let rhs_code = generate_expr_as_ex(&input.rhs)?;
    Ok(quote! {
        ::symplex::eq::Equation::new(#lhs_code, #rhs_code)
    })
}

/// Like [`generate_expr`] but guarantees the result is an `Ex`, even for
/// bare integer literals (which `generate_expr` emits as plain `i64`).
///
/// Function calls (including `diff`, `factorial`, `binomial`, `C`, `log`,
/// and all single-arg functions) always return `Ex`, so we delegate
/// directly to [`generate_expr`] for those.
fn generate_expr_as_ex(expr: &MathExpr) -> syn::Result<TokenStream2> {
    match expr {
        MathExpr::Int(n, _) => Ok(quote! { __ctx.int(#n) }),
        MathExpr::Neg(inner) => {
            if let Some(n) = inner.as_int() {
                let neg_n = -n;
                Ok(quote! { __ctx.int(#neg_n) })
            } else {
                let code = generate_expr(expr)?;
                Ok(quote! { { let __v: ::symplex::expr::Ex = (#code).clone(); __v } })
            }
        }
        MathExpr::Func { .. } => generate_expr(expr),
        _ => {
            let code = generate_expr(expr)?;
            Ok(quote! { { let __v: ::symplex::expr::Ex = (#code).clone(); __v } })
        }
    }
}
