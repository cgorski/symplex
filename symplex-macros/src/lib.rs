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

use parse::{BinOp, ExprMacroInput, MathExpr, RuleMacroInput};
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
/// - `1/2` is Rust integer division (= 0).  Use `ctx.rational(1, 2)` for fractions.
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

        MathExpr::Ident(id) => Ok(quote! { (&#id) }),

        MathExpr::Neg(inner) => {
            let inner_code = generate_expr(inner)?;
            Ok(quote! { (-(#inner_code)) })
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
                    // Check for Int / Int — emit compile error.
                    if lhs.as_int().is_some() && rhs.as_int().is_some() {
                        return Err(syn::Error::new(
                            Span::call_site(),
                            "integer / integer inside expr!() is Rust integer division, \
                             which truncates. Use ctx.rational(p, q) for exact fractions.",
                        ));
                    }
                    let lhs_code = generate_expr(lhs)?;
                    let rhs_code = generate_expr(rhs)?;
                    Ok(quote! { ((#lhs_code) / (#rhs_code)) })
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
            if !is_known_function(name) {
                return Err(syn::Error::new(
                    *span,
                    format!(
                        "unknown function '{}' in expr!(). Supported: {}",
                        name,
                        KNOWN_FUNCTIONS.join(", ")
                    ),
                ));
            }
            if args.len() != 1 {
                return Err(syn::Error::new(
                    *span,
                    format!("{}() takes exactly 1 argument", name),
                ));
            }
            let arg_code = generate_expr(&args[0])?;
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
                _ => unreachable!(),
            };
            Ok(quote! { (#arg_code).#method() })
        }
    }
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
/// - Conditions (`if ...`) are not yet supported.
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

    Ok(quote! {
        {
            #(#bindings)*

            let mut __wilds = ::symplex::__macro_support::FxHashMap::default();
            #(#wild_inserts)*

            ::symplex::__macro_support::Rule::new(
                #name,
                ::symplex::__macro_support::Pattern {
                    root: #lhs_temp,
                    wilds: __wilds,
                },
                #rhs_temp,
            )
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
                        self.bindings.push(quote! { let #temp = #arena.zero; });
                    }
                    1 => {
                        self.bindings.push(quote! { let #temp = #arena.one; });
                    }
                    -1 => {
                        self.bindings.push(quote! { let #temp = #arena.neg_one; });
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
                        "pi" => quote! { #arena.pi },
                        "E" => quote! { #arena.e_const },
                        "I" => quote! { #arena.i_unit },
                        "oo" => quote! { #arena.infinity },
                        "nan" => quote! { #arena.nan },
                        "zoo" => quote! { #arena.complex_infinity },
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
                if args.len() != 1 {
                    return Err(syn::Error::new(
                        *span,
                        format!("{}() takes exactly 1 argument", name),
                    ));
                }

                let arg_temp = self.generate_arena_expr(&args[0])?;
                let temp = self.fresh_temp();

                let call = match name.as_str() {
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
                    "exp" => quote! { #arena.exp(#arg_temp) },
                    "ln" => quote! { #arena.ln(#arg_temp) },
                    "sqrt" => quote! { #arena.sqrt(#arg_temp) },
                    "cbrt" => quote! { #arena.cbrt(#arg_temp) },
                    "abs" => quote! { #arena.abs(#arg_temp) },
                    _ => unreachable!(),
                };

                self.bindings.push(quote! { let #temp = #call; });
                Ok(temp)
            }
        }
    }
}
