//! Rust source code generation from symbolic expressions.
//!
//! Converts an expression tree into a Rust function body string.
//! Uses CSE (common subexpression elimination) for efficiency.

use crate::arena::Arena;
use crate::errors::SymplexError;
use crate::node::{ExprId, ExprNode};
use num_traits::ToPrimitive;
use rustc_hash::FxHashMap;

// ═══════════════════════════════════════════════════════════════════════════
// CodegenOptions types
// ═══════════════════════════════════════════════════════════════════════════

/// Math function dispatch strategy for generated code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathBackend {
    /// Use `f64::sin()` method syntax. Works on std targets.
    Std,
    /// Use `libm::sin(x)` function syntax. Works on no_std with libm.
    Libm,
    /// Emit cfg-gated wrapper module. Works everywhere.
    CfgGated,
}

/// Floating-point precision for generated code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// 64-bit floating point.
    F64,
    /// 32-bit floating point.
    F32,
}

/// Configuration for Rust code generation.
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    /// Math function dispatch strategy.
    pub math_backend: MathBackend,
    /// Floating-point precision.
    pub precision: Precision,
    /// Add #[inline] annotation to generated functions.
    pub inline: bool,
    /// Add #[must_use] annotation.
    pub must_use: bool,
    /// Run CSE before code generation.
    pub cse: bool,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            math_backend: MathBackend::Std,
            precision: Precision::F64,
            inline: false,
            must_use: true,
            cse: true,
        }
    }
}

impl CodegenOptions {
    /// Configuration for no_std embedded targets.
    /// Uses cfg-gated math backend and adds #[inline].
    pub fn no_std() -> Self {
        Self {
            math_backend: MathBackend::CfgGated,
            inline: true,
            ..Default::default()
        }
    }

    /// Configuration for embedded f32 targets (e.g., Cortex-M4F).
    /// Uses cfg-gated math backend, f32 precision, and #[inline].
    pub fn embedded_f32() -> Self {
        Self {
            math_backend: MathBackend::CfgGated,
            precision: Precision::F32,
            inline: true,
            ..Default::default()
        }
    }
}

impl Precision {
    /// The float type name, e.g. `"f64"` or `"f32"`.
    fn type_name(self) -> &'static str {
        match self {
            Precision::F64 => "f64",
            Precision::F32 => "f32",
        }
    }

    /// The suffix for float literals, e.g. `"_f64"` or `"_f32"`.
    fn suffix(self) -> &'static str {
        match self {
            Precision::F64 => "_f64",
            Precision::F32 => "_f32",
        }
    }

    /// The consts module path, e.g. `"std::f64::consts"` or `"std::f32::consts"`.
    fn consts_mod(self) -> &'static str {
        match self {
            Precision::F64 => "std::f64::consts",
            Precision::F32 => "std::f32::consts",
        }
    }

    /// The infinity constant, e.g. `"f64::INFINITY"`.
    fn infinity(self) -> &'static str {
        match self {
            Precision::F64 => "f64::INFINITY",
            Precision::F32 => "f32::INFINITY",
        }
    }

    /// The negative infinity constant.
    fn neg_infinity(self) -> &'static str {
        match self {
            Precision::F64 => "f64::NEG_INFINITY",
            Precision::F32 => "f32::NEG_INFINITY",
        }
    }

    /// The NaN constant.
    fn nan(self) -> &'static str {
        match self {
            Precision::F64 => "f64::NAN",
            Precision::F32 => "f32::NAN",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

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
    to_rust_fn_with_options(arena, expr, name, args, &CodegenOptions::default())
}

/// Generate a Rust function as a string from an expression, with custom options.
pub(crate) fn to_rust_fn_with_options(
    arena: &mut Arena,
    expr: ExprId,
    name: &str,
    args: &[&str],
    options: &CodegenOptions,
) -> Result<String, SymplexError> {
    let float_type = options.precision.type_name();

    // Run CSE if requested
    let (bindings_list, final_expr) = if options.cse {
        let cse_result = crate::cse::cse(arena, expr);
        (cse_result.bindings, cse_result.expr)
    } else {
        (Vec::new(), expr)
    };

    // Post-CSE constant propagation: evaluate pure-constant bindings
    let mut cse_constants: FxHashMap<usize, f64> = FxHashMap::default();
    let mut kept_bindings: Vec<(usize, ExprId)> = Vec::new();
    for (i, (_, binding_expr)) in bindings_list.iter().enumerate() {
        if is_pure_constant(arena, *binding_expr, &cse_constants)
            && let Some(val) = eval_constant_f64(arena, *binding_expr, &cse_constants)
        {
            cse_constants.insert(i, val);
            continue;
        }
        kept_bindings.push((i, *binding_expr));
    }

    let mut lines = Vec::new();

    // Cfg-gated math module
    if options.math_backend == MathBackend::CfgGated {
        append_cfg_gated_module(&mut lines, options.precision);
        lines.push(String::new());
    }

    // Annotations
    if options.inline {
        lines.push("#[inline]".to_string());
    }
    if options.must_use {
        lines.push("#[must_use]".to_string());
    }

    // Function signature
    let params: Vec<String> = args.iter().map(|a| format!("{a}: {float_type}")).collect();
    lines.push(format!(
        "pub fn {name}({}) -> {float_type} {{",
        params.join(", ")
    ));

    // CSE bindings (skip pure-constant ones)
    for &(i, binding_expr) in &kept_bindings {
        let code = expr_to_rust_cse(arena, binding_expr, args, options, &cse_constants)?;
        lines.push(format!("    let t{i} = {code};"));
    }

    // Final expression (strip unnecessary outer parens)
    let result_code = expr_to_rust_cse(arena, final_expr, args, options, &cse_constants)?;
    let result_code = strip_outer_parens(&result_code);
    lines.push(format!("    {result_code}"));
    lines.push("}".to_string());

    Ok(lines.join("\n"))
}

/// Generate a Rust function that computes a matrix and returns a flat array.
///
/// Uses `cse_multi` across all matrix entries for cross-entry CSE.
pub(crate) fn matrix_to_rust_fn(
    arena: &mut Arena,
    entries: &[ExprId],
    nrows: usize,
    ncols: usize,
    name: &str,
    args: &[&str],
    options: &CodegenOptions,
) -> Result<String, SymplexError> {
    let float_type = options.precision.type_name();
    let total = nrows * ncols;

    // Run multi-expression CSE if requested
    let (bindings_list, final_entries) = if options.cse {
        let cse_result = crate::cse::cse_multi(arena, entries);
        (cse_result.bindings, cse_result.exprs)
    } else {
        (Vec::new(), entries.to_vec())
    };

    // Post-CSE constant propagation: evaluate pure-constant bindings
    let mut cse_constants: FxHashMap<usize, f64> = FxHashMap::default();
    let mut kept_bindings: Vec<(usize, ExprId)> = Vec::new();
    for (i, (_, binding_expr)) in bindings_list.iter().enumerate() {
        if is_pure_constant(arena, *binding_expr, &cse_constants)
            && let Some(val) = eval_constant_f64(arena, *binding_expr, &cse_constants)
        {
            cse_constants.insert(i, val);
            continue;
        }
        kept_bindings.push((i, *binding_expr));
    }

    let mut lines = Vec::new();

    // Cfg-gated math module
    if options.math_backend == MathBackend::CfgGated {
        append_cfg_gated_module(&mut lines, options.precision);
        lines.push(String::new());
    }

    // Annotations
    if options.inline {
        lines.push("#[inline]".to_string());
    }
    if options.must_use {
        lines.push("#[must_use]".to_string());
    }

    // Function signature: returns a flat array
    let params: Vec<String> = args.iter().map(|a| format!("{a}: {float_type}")).collect();
    lines.push(format!(
        "pub fn {name}({}) -> [{float_type}; {total}] {{",
        params.join(", ")
    ));

    // CSE bindings (skip pure-constant ones)
    for &(i, binding_expr) in &kept_bindings {
        let code = expr_to_rust_cse(arena, binding_expr, args, options, &cse_constants)?;
        lines.push(format!("    let t{i} = {code};"));
    }

    // Matrix entries
    for (i, &entry_id) in final_entries.iter().enumerate() {
        let code = expr_to_rust_cse(arena, entry_id, args, options, &cse_constants)?;
        if i == 0 {
            lines.push(format!("    [{code},"));
        } else if i + 1 == total {
            lines.push(format!("     {code}]"));
        } else {
            lines.push(format!("     {code},"));
        }
    }

    lines.push("}".to_string());

    Ok(lines.join("\n"))
}

// ═══════════════════════════════════════════════════════════════════════════
// Cfg-gated math module
// ═══════════════════════════════════════════════════════════════════════════

fn append_cfg_gated_module(lines: &mut Vec<String>, precision: Precision) {
    let ft = precision.type_name();
    lines.push("#[cfg(feature = \"std\")]".to_string());
    lines.push("mod math {".to_string());
    for func in &[
        "sin", "cos", "tan", "exp", "ln", "abs", "sqrt", "cbrt", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "floor", "ceil", "signum",
    ] {
        let method = *func;
        lines.push(format!(
            "    #[inline] pub fn {method}(x: {ft}) -> {ft} {{ x.{method}() }}"
        ));
    }
    lines.push(format!(
        "    #[inline] pub fn atan2(y: {ft}, x: {ft}) -> {ft} {{ y.atan2(x) }}"
    ));
    lines.push(format!(
        "    #[inline] pub fn powf(base: {ft}, exp: {ft}) -> {ft} {{ base.powf(exp) }}"
    ));
    lines.push(format!(
        "    #[inline] pub fn powi(base: {ft}, exp: i32) -> {ft} {{ base.powi(exp) }}"
    ));
    lines.push(format!(
        "    #[inline] pub fn min(a: {ft}, b: {ft}) -> {ft} {{ a.min(b) }}"
    ));
    lines.push(format!(
        "    #[inline] pub fn max(a: {ft}, b: {ft}) -> {ft} {{ a.max(b) }}"
    ));
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push("#[cfg(not(feature = \"std\"))]".to_string());
    lines.push("mod math {".to_string());
    for func in &[
        "sin", "cos", "tan", "exp", "abs", "sqrt", "cbrt", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "floor", "ceil",
    ] {
        let method = *func;
        lines.push(format!(
            "    #[inline] pub fn {method}(x: {ft}) -> {ft} {{ libm::{method}({} as f64) as {ft} }}",
            "x"
        ));
    }
    // ln maps to log in libm
    lines.push(format!(
        "    #[inline] pub fn ln(x: {ft}) -> {ft} {{ libm::log(x as f64) as {ft} }}"
    ));
    lines.push(format!(
        "    #[inline] pub fn signum(x: {ft}) -> {ft} {{ if x > 0.0 {{ 1.0 }} else if x < 0.0 {{ -1.0 }} else {{ 0.0 }} }}"
    ));
    lines.push(format!(
        "    #[inline] pub fn atan2(y: {ft}, x: {ft}) -> {ft} {{ libm::atan2(y as f64, x as f64) as {ft} }}"
    ));
    lines.push(format!(
        "    #[inline] pub fn powf(base: {ft}, exp: {ft}) -> {ft} {{ libm::pow(base as f64, exp as f64) as {ft} }}"
    ));
    lines.push(format!(
        "    #[inline] pub fn powi(base: {ft}, exp: i32) -> {ft} {{ libm::pow(base as f64, exp as f64) as {ft} }}"
    ));
    lines.push(format!(
        "    #[inline] pub fn min(a: {ft}, b: {ft}) -> {ft} {{ libm::fmin(a as f64, b as f64) as {ft} }}"
    ));
    lines.push(format!(
        "    #[inline] pub fn max(a: {ft}, b: {ft}) -> {ft} {{ libm::fmax(a as f64, b as f64) as {ft} }}"
    ));
    lines.push("}".to_string());
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression-to-Rust code generator
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
fn expr_to_rust(
    arena: &Arena,
    id: ExprId,
    var_names: &[&str],
    options: &CodegenOptions,
) -> Result<String, SymplexError> {
    expr_to_rust_cse(arena, id, var_names, options, &FxHashMap::default())
}

fn expr_to_rust_cse(
    arena: &Arena,
    id: ExprId,
    var_names: &[&str],
    options: &CodegenOptions,
    cse_constants: &FxHashMap<usize, f64>,
) -> Result<String, SymplexError> {
    let prec = options.precision;
    let suffix = prec.suffix();

    match arena.node(id).clone() {
        ExprNode::Num(nid) => {
            let r = arena.num(nid);
            if r.is_integer() {
                let n = r.numer();
                Ok(format!("{n}{suffix}"))
            } else {
                let n = r.numer().to_f64().unwrap_or(0.0);
                let d = r.denom().to_f64().unwrap_or(1.0);
                let val = n / d;
                Ok(format!("{}{suffix}", format_float(val)))
            }
        }
        ExprNode::Symbol(sid) => {
            let sym_name = arena.symbols.name(sid);
            if var_names.contains(&sym_name) {
                Ok(sym_name.to_string())
            } else if sym_name.starts_with("__cse_") {
                let idx_str = sym_name.strip_prefix("__cse_").unwrap();
                let idx: usize = idx_str.parse().map_err(|_| {
                    SymplexError::NotImplemented(format!("invalid CSE variable name: {sym_name}"))
                })?;
                // Check if this CSE variable was resolved to a constant
                if let Some(&val) = cse_constants.get(&idx) {
                    Ok(format!("{}{suffix}", format_float(val)))
                } else {
                    Ok(format!("t{idx}"))
                }
            } else {
                Err(SymplexError::FreeSymbol {
                    name: sym_name.to_string(),
                })
            }
        }
        ExprNode::Pi => Ok(format!("{}::PI", prec.consts_mod())),
        ExprNode::E => Ok(format!("{}::E", prec.consts_mod())),
        ExprNode::Infinity => Ok(prec.infinity().to_string()),
        ExprNode::NegInfinity => Ok(prec.neg_infinity().to_string()),
        ExprNode::NaN | ExprNode::ComplexInfinity => Ok(prec.nan().to_string()),
        ExprNode::Add(ref children) => {
            if children.is_empty() {
                return Ok(format!("0.0{suffix}"));
            }
            // Filter out zero-valued children (from CSE constant propagation)
            let live: Vec<ExprId> = children
                .iter()
                .copied()
                .filter(|&c| {
                    try_resolve_constant(arena, c, cse_constants)
                        .is_none_or(|v| v != 0.0)
                })
                .collect();
            if live.is_empty() {
                return Ok(format!("0.0{suffix}"));
            }
            if live.len() == 1 {
                return expr_to_rust_cse(arena, live[0], var_names, options, cse_constants);
            }
            // Build parts with subtraction detection
            let mut parts = Vec::new();
            for (i, &child) in live.iter().enumerate() {
                let (is_neg, code) = if matches!(arena.node(child), ExprNode::Neg(_)) {
                    if let ExprNode::Neg(inner) = arena.node(child).clone() {
                        (true, expr_to_rust_cse(arena, inner, var_names, options, cse_constants)?)
                    } else {
                        unreachable!()
                    }
                } else if is_neg_one_mul_codegen(arena, child) {
                    (true, emit_mul_without_neg_one(arena, child, var_names, options, cse_constants)?)
                } else {
                    (false, expr_to_rust_cse(arena, child, var_names, options, cse_constants)?)
                };
                if i == 0 {
                    if is_neg {
                        parts.push(format!("-{code}"));
                    } else {
                        parts.push(code);
                    }
                } else if is_neg {
                    parts.push(format!(" - {code}"));
                } else {
                    parts.push(format!(" + {code}"));
                }
            }
            Ok(format!("({})", parts.join("")))
        }
        ExprNode::Mul(ref children) => {
            if children.is_empty() {
                return Ok(format!("1.0{suffix}"));
            }
            // Check for zero-valued children → whole product is 0
            for &c in children.iter() {
                if let Some(v) = try_resolve_constant(arena, c, cse_constants)
                    && v == 0.0
                {
                    return Ok(format!("0.0{suffix}"));
                }
            }
            // Filter out 1.0-valued children
            let live: Vec<ExprId> = children
                .iter()
                .copied()
                .filter(|&c| {
                    try_resolve_constant(arena, c, cse_constants)
                        .is_none_or(|v| v != 1.0)
                })
                .collect();
            if live.is_empty() {
                return Ok(format!("1.0{suffix}"));
            }
            if live.len() == 1 {
                return expr_to_rust_cse(arena, live[0], var_names, options, cse_constants);
            }
            // Detect leading -1 coefficient: emit -(rest) instead of (-1_f64 * rest)
            if live.len() >= 2
                && let Some(r) = arena.as_num(live[0])
                && r.is_integer()
                && r.numer().to_i64() == Some(-1)
            {
                let rest: Result<Vec<String>, _> = live[1..]
                    .iter()
                    .map(|&c| expr_to_rust_cse(arena, c, var_names, options, cse_constants))
                    .collect();
                let rest = rest?;
                return if rest.len() == 1 {
                    Ok(format!("(-{})", rest[0]))
                } else {
                    Ok(format!("-({})", rest.join(" * ")))
                };
            }
            let parts: Result<Vec<String>, _> = live
                .iter()
                .map(|&c| expr_to_rust_cse(arena, c, var_names, options, cse_constants))
                .collect();
            Ok(format!("({})", parts?.join(" * ")))
        }
        ExprNode::Pow(base, exp) => {
            let b = expr_to_rust_cse(arena, base, var_names, options, cse_constants)?;
            if let Some(r) = arena.as_num(exp) {
                if r.is_integer()
                    && let Some(n) = r.numer().to_i64()
                    && (-10..=10).contains(&n)
                {
                    return emit_powi(&b, n, options);
                }
                // sqrt: exponent == 1/2
                if *r.numer() == 1.into() && *r.denom() == 2.into() {
                    return emit_unary_call(&b, "sqrt", options);
                }
                // cbrt: exponent == 1/3
                if *r.numer() == 1.into() && *r.denom() == 3.into() {
                    return emit_unary_call(&b, "cbrt", options);
                }
            }
            let e = expr_to_rust_cse(arena, exp, var_names, options, cse_constants)?;
            emit_powf(&b, &e, options)
        }
        ExprNode::Neg(inner) => {
            let inner_code = expr_to_rust_cse(arena, inner, var_names, options, cse_constants)?;
            Ok(format!("(-{inner_code})"))
        }
        ExprNode::Sin(x) => emit_unary(arena, x, "sin", var_names, options, cse_constants),
        ExprNode::Cos(x) => emit_unary(arena, x, "cos", var_names, options, cse_constants),
        ExprNode::Tan(x) => emit_unary(arena, x, "tan", var_names, options, cse_constants),
        ExprNode::Exp(x) => emit_unary(arena, x, "exp", var_names, options, cse_constants),
        ExprNode::Ln(x) => emit_unary(arena, x, "ln", var_names, options, cse_constants),
        ExprNode::Abs(x) => emit_unary(arena, x, "abs", var_names, options, cse_constants),
        ExprNode::Asin(x) => emit_unary(arena, x, "asin", var_names, options, cse_constants),
        ExprNode::Acos(x) => emit_unary(arena, x, "acos", var_names, options, cse_constants),
        ExprNode::Atan(x) => emit_unary(arena, x, "atan", var_names, options, cse_constants),
        ExprNode::Atan2(y, x) => {
            let y_code = expr_to_rust_cse(arena, y, var_names, options, cse_constants)?;
            let x_code = expr_to_rust_cse(arena, x, var_names, options, cse_constants)?;
            emit_atan2(&y_code, &x_code, options)
        }
        ExprNode::Sinh(x) => emit_unary(arena, x, "sinh", var_names, options, cse_constants),
        ExprNode::Cosh(x) => emit_unary(arena, x, "cosh", var_names, options, cse_constants),
        ExprNode::Tanh(x) => emit_unary(arena, x, "tanh", var_names, options, cse_constants),
        ExprNode::Asinh(x) => emit_unary(arena, x, "asinh", var_names, options, cse_constants),
        ExprNode::Acosh(x) => emit_unary(arena, x, "acosh", var_names, options, cse_constants),
        ExprNode::Atanh(x) => emit_unary(arena, x, "atanh", var_names, options, cse_constants),
        ExprNode::Sign(x) => emit_unary(arena, x, "signum", var_names, options, cse_constants),
        ExprNode::Heaviside(x) => {
            let code = expr_to_rust_cse(arena, x, var_names, options, cse_constants)?;
            let s = options.precision.suffix();
            Ok(format!(
                "(if {code} > 0.0{s} {{ 1.0{s} }} else if {code} < 0.0{s} {{ 0.0{s} }} else {{ 0.5{s} }})"
            ))
        }
        ExprNode::DiracDelta(_x) => {
            let s = options.precision.suffix();
            Ok(format!("0.0{s}"))
        }
        ExprNode::Floor(x) => emit_unary(arena, x, "floor", var_names, options, cse_constants),
        ExprNode::Ceiling(x) => emit_unary(arena, x, "ceil", var_names, options, cse_constants),
        ExprNode::Min(ref children) => {
            if children.is_empty() {
                return Ok(prec.infinity().to_string());
            }
            let mut code = expr_to_rust_cse(arena, children[0], var_names, options, cse_constants)?;
            for &c in &children[1..] {
                let c_code = expr_to_rust_cse(arena, c, var_names, options, cse_constants)?;
                code = emit_min(&code, &c_code, options);
            }
            Ok(code)
        }
        ExprNode::Max(ref children) => {
            if children.is_empty() {
                return Ok(prec.neg_infinity().to_string());
            }
            let mut code = expr_to_rust_cse(arena, children[0], var_names, options, cse_constants)?;
            for &c in &children[1..] {
                let c_code = expr_to_rust_cse(arena, c, var_names, options, cse_constants)?;
                code = emit_max(&code, &c_code, options);
            }
            Ok(code)
        }
        // Unsupported nodes
        ExprNode::ImaginaryUnit => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for imaginary unit".to_string(),
        )),
        ExprNode::Factorial(_) | ExprNode::Binomial(_, _) => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for combinatorial functions".to_string(),
        )),
        ExprNode::Gamma(_)
        | ExprNode::LogGamma(_)
        | ExprNode::Digamma(_)
        | ExprNode::Erf(_)
        | ExprNode::Erfc(_)
        | ExprNode::Beta(_, _) => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for special functions (gamma, erf, beta)".to_string(),
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
        ExprNode::Sum(_, _, _, _) => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for symbolic Sum".to_string(),
        )),
        ExprNode::Product_(_, _, _, _) => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for symbolic Product".to_string(),
        )),
        ExprNode::Piecewise(ref branches) => {
            codegen_piecewise(arena, branches, var_names, options, cse_constants)
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
        ExprNode::EmptySet
        | ExprNode::UniversalSet
        | ExprNode::Interval(_, _, _)
        | ExprNode::FiniteSet(_)
        | ExprNode::SetUnion(_)
        | ExprNode::SetIntersection(_)
        | ExprNode::SetComplement(_, _) => Err(SymplexError::NotImplemented(
            "cannot generate Rust code for set-valued expressions".to_string(),
        )),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Constant-folding helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Try to extract a concrete `f64` value from an expression node.
///
/// Returns `Some(v)` when the node is a numeric literal (`Num`), `Pi`, or `E`.
fn try_const_eval_f64(arena: &Arena, id: ExprId) -> Option<f64> {
    match arena.node(id) {
        ExprNode::Num(nid) => {
            let r = arena.num(*nid);
            let n = r.numer().to_f64()?;
            let d = r.denom().to_f64()?;
            Some(n / d)
        }
        ExprNode::Pi => Some(std::f64::consts::PI),
        ExprNode::E => Some(std::f64::consts::E),
        _ => None,
    }
}

/// Evaluate a named unary math function on a concrete `f64` value.
fn eval_unary_f64(func: &str, val: f64) -> Option<f64> {
    Some(match func {
        "sin" => val.sin(),
        "cos" => val.cos(),
        "tan" => val.tan(),
        "exp" => val.exp(),
        "ln" => val.ln(),
        "abs" => val.abs(),
        "asin" => val.asin(),
        "acos" => val.acos(),
        "atan" => val.atan(),
        "sinh" => val.sinh(),
        "cosh" => val.cosh(),
        "tanh" => val.tanh(),
        "asinh" => val.asinh(),
        "acosh" => val.acosh(),
        "atanh" => val.atanh(),
        "signum" => val.signum(),
        "floor" => val.floor(),
        "ceil" => val.ceil(),
        "sqrt" => val.sqrt(),
        "cbrt" => val.cbrt(),
        _ => return None,
    })
}

/// Format an `f64` as a Rust literal (always includes a decimal point).
fn format_float(v: f64) -> String {
    if v == 0.0 {
        "0.0".to_string()
    } else if v == 1.0 {
        "1.0".to_string()
    } else if v == -1.0 {
        "-1.0".to_string()
    } else {
        // Ensure a decimal point so Rust sees it as a float literal.
        let s = format!("{v}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CSE constant propagation helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if an expression tree consists entirely of constant leaves.
///
/// Returns `true` when every leaf is a `Num`, `Pi`, or `E` node — or a CSE
/// symbol that was already resolved to a constant value.
fn is_pure_constant(arena: &Arena, id: ExprId, resolved: &FxHashMap<usize, f64>) -> bool {
    let mut stack = vec![id];
    while let Some(cur) = stack.pop() {
        match arena.node(cur) {
            ExprNode::Num(_) | ExprNode::Pi | ExprNode::E => {}
            ExprNode::Symbol(sid) => {
                let name = arena.symbols.name(*sid);
                if let Some(idx_str) = name.strip_prefix("__cse_")
                    && let Ok(idx) = idx_str.parse::<usize>()
                    && resolved.contains_key(&idx)
                {
                    continue;
                }
                return false;
            }
            other => {
                for child in other.children() {
                    stack.push(child);
                }
            }
        }
    }
    true
}

/// Recursively evaluate a pure-constant expression to `f64`.
fn eval_constant_f64(
    arena: &Arena,
    id: ExprId,
    cse_constants: &FxHashMap<usize, f64>,
) -> Option<f64> {
    match arena.node(id).clone() {
        ExprNode::Num(nid) => {
            let r = arena.num(nid);
            let n = r.numer().to_f64()?;
            let d = r.denom().to_f64()?;
            Some(n / d)
        }
        ExprNode::Pi => Some(std::f64::consts::PI),
        ExprNode::E => Some(std::f64::consts::E),
        ExprNode::Symbol(sid) => {
            let name = arena.symbols.name(sid);
            let idx_str = name.strip_prefix("__cse_")?;
            let idx: usize = idx_str.parse().ok()?;
            cse_constants.get(&idx).copied()
        }
        ExprNode::Add(ref children) => {
            let mut sum = 0.0;
            for &c in children {
                sum += eval_constant_f64(arena, c, cse_constants)?;
            }
            Some(sum)
        }
        ExprNode::Mul(ref children) => {
            let mut prod = 1.0;
            for &c in children {
                prod *= eval_constant_f64(arena, c, cse_constants)?;
            }
            Some(prod)
        }
        ExprNode::Neg(inner) => eval_constant_f64(arena, inner, cse_constants).map(|v| -v),
        ExprNode::Pow(base, exp) => {
            let b = eval_constant_f64(arena, base, cse_constants)?;
            let e = eval_constant_f64(arena, exp, cse_constants)?;
            Some(b.powf(e))
        }
        ExprNode::Sin(x) => eval_constant_f64(arena, x, cse_constants).map(|v| v.sin()),
        ExprNode::Cos(x) => eval_constant_f64(arena, x, cse_constants).map(|v| v.cos()),
        ExprNode::Tan(x) => eval_constant_f64(arena, x, cse_constants).map(|v| v.tan()),
        ExprNode::Exp(x) => eval_constant_f64(arena, x, cse_constants).map(|v| v.exp()),
        ExprNode::Ln(x) => eval_constant_f64(arena, x, cse_constants).map(|v| v.ln()),
        ExprNode::Abs(x) => eval_constant_f64(arena, x, cse_constants).map(|v| v.abs()),
        _ => None,
    }
}

/// Try to resolve an expression to a known constant `f64`.
///
/// Checks direct numeric literals, Pi, E, and CSE symbols that were previously
/// resolved to constant values.
fn try_resolve_constant(
    arena: &Arena,
    id: ExprId,
    cse_constants: &FxHashMap<usize, f64>,
) -> Option<f64> {
    if let Some(v) = try_const_eval_f64(arena, id) {
        return Some(v);
    }
    if let ExprNode::Symbol(sid) = arena.node(id)
        && let Some(idx_str) = arena.symbols.name(*sid).strip_prefix("__cse_")
        && let Ok(idx) = idx_str.parse::<usize>()
    {
        return cse_constants.get(&idx).copied();
    }
    None
}

/// Check if `id` is a `Mul` node whose first factor is −1.
fn is_neg_one_mul_codegen(arena: &Arena, id: ExprId) -> bool {
    if let ExprNode::Mul(children) = arena.node(id)
        && let Some(&first) = children.first()
        && let ExprNode::Num(nid) = arena.node(first)
    {
        let r = arena.num(*nid);
        return r.is_integer() && r.numer().to_i64() == Some(-1);
    }
    false
}

/// Emit the non-neg-one part of a `Mul([-1, rest...])` node.
fn emit_mul_without_neg_one(
    arena: &Arena,
    id: ExprId,
    var_names: &[&str],
    options: &CodegenOptions,
    cse_constants: &FxHashMap<usize, f64>,
) -> Result<String, SymplexError> {
    if let ExprNode::Mul(ref children) = arena.node(id).clone() {
        let rest: Result<Vec<String>, _> = children[1..]
            .iter()
            .map(|&c| expr_to_rust_cse(arena, c, var_names, options, cse_constants))
            .collect();
        let rest = rest?;
        if rest.len() == 1 {
            Ok(rest.into_iter().next().unwrap())
        } else {
            Ok(format!("({})", rest.join(" * ")))
        }
    } else {
        expr_to_rust_cse(arena, id, var_names, options, cse_constants)
    }
}

/// Strip one level of outer parentheses when they wrap the entire string.
#[must_use]
fn strip_outer_parens(s: &str) -> &str {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'(') || bytes.last() != Some(&b')') {
        return s;
    }
    let inner = &s[1..s.len() - 1];
    let mut depth: i32 = 0;
    for c in inner.chars() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth < 0 {
                    return s;
                }
            }
            _ => {}
        }
    }
    if depth == 0 { inner } else { s }
}

// ═══════════════════════════════════════════════════════════════════════════
// Backend-aware emission helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a unary math function call, resolving the argument first.
///
/// When the argument is a compile-time constant (numeric literal, π, or e),
/// the function is evaluated eagerly and the result emitted as a float literal
/// (constant folding).
fn emit_unary(
    arena: &Arena,
    arg: ExprId,
    func: &str,
    var_names: &[&str],
    options: &CodegenOptions,
    cse_constants: &FxHashMap<usize, f64>,
) -> Result<String, SymplexError> {
    // Constant folding: evaluate at codegen time when the argument is known.
    if let Some(val) = try_const_eval_f64(arena, arg)
        && let Some(result) = eval_unary_f64(func, val)
        && (result.is_finite() || result == 0.0)
    {
        let suffix = options.precision.suffix();
        return Ok(format!("{}{}", format_float(result), suffix));
    }
    let code = expr_to_rust_cse(arena, arg, var_names, options, cse_constants)?;
    emit_unary_call(&code, func, options)
}

/// Emit a unary math function call on an already-resolved code string.
fn emit_unary_call(
    arg_code: &str,
    func: &str,
    options: &CodegenOptions,
) -> Result<String, SymplexError> {
    Ok(match options.math_backend {
        MathBackend::Std => format!("{arg_code}.{func}()"),
        MathBackend::Libm => {
            let libm_fn = libm_function_name(func);
            format!("libm::{libm_fn}({arg_code} as f64) as {}", options.precision.type_name())
        }
        MathBackend::CfgGated => format!("math::{func}({arg_code})"),
    })
}

/// Emit a powi call.
fn emit_powi(
    base_code: &str,
    exp: i64,
    options: &CodegenOptions,
) -> Result<String, SymplexError> {
    Ok(match options.math_backend {
        MathBackend::Std => format!("{base_code}.powi({exp})"),
        MathBackend::Libm => {
            let ft = options.precision.type_name();
            format!("libm::pow({base_code} as f64, {exp}_f64) as {ft}")
        }
        MathBackend::CfgGated => format!("math::powi({base_code}, {exp})"),
    })
}

/// Emit a powf call.
fn emit_powf(
    base_code: &str,
    exp_code: &str,
    options: &CodegenOptions,
) -> Result<String, SymplexError> {
    Ok(match options.math_backend {
        MathBackend::Std => format!("{base_code}.powf({exp_code})"),
        MathBackend::Libm => {
            let ft = options.precision.type_name();
            format!("libm::pow({base_code} as f64, {exp_code} as f64) as {ft}")
        }
        MathBackend::CfgGated => format!("math::powf({base_code}, {exp_code})"),
    })
}

/// Emit an atan2 call.
fn emit_atan2(
    y_code: &str,
    x_code: &str,
    options: &CodegenOptions,
) -> Result<String, SymplexError> {
    Ok(match options.math_backend {
        MathBackend::Std => format!("{y_code}.atan2({x_code})"),
        MathBackend::Libm => {
            let ft = options.precision.type_name();
            format!("libm::atan2({y_code} as f64, {x_code} as f64) as {ft}")
        }
        MathBackend::CfgGated => format!("math::atan2({y_code}, {x_code})"),
    })
}

/// Emit a min call.
fn emit_min(a_code: &str, b_code: &str, options: &CodegenOptions) -> String {
    match options.math_backend {
        MathBackend::Std => format!("{a_code}.min({b_code})"),
        MathBackend::Libm => {
            let ft = options.precision.type_name();
            format!("libm::fmin({a_code} as f64, {b_code} as f64) as {ft}")
        }
        MathBackend::CfgGated => format!("math::min({a_code}, {b_code})"),
    }
}

/// Emit a max call.
fn emit_max(a_code: &str, b_code: &str, options: &CodegenOptions) -> String {
    match options.math_backend {
        MathBackend::Std => format!("{a_code}.max({b_code})"),
        MathBackend::Libm => {
            let ft = options.precision.type_name();
            format!("libm::fmax({a_code} as f64, {b_code} as f64) as {ft}")
        }
        MathBackend::CfgGated => format!("math::max({a_code}, {b_code})"),
    }
}

/// Map our internal function names to libm function names.
fn libm_function_name(func: &str) -> &str {
    match func {
        "ln" => "log",
        "abs" => "fabs",
        "signum" => "copysign", // we handle signum specially below
        _ => func,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Piecewise & boolean codegen
// ═══════════════════════════════════════════════════════════════════════════

/// Generate Rust code for a piecewise expression as a chain of if/else.
fn codegen_piecewise(
    arena: &Arena,
    branches: &[(ExprId, ExprId)],
    var_names: &[&str],
    options: &CodegenOptions,
    cse_constants: &FxHashMap<usize, f64>,
) -> Result<String, SymplexError> {
    if branches.is_empty() {
        return Ok(options.precision.nan().to_string());
    }

    let mut parts = Vec::new();
    for (i, &(value, cond)) in branches.iter().enumerate() {
        let val_code = expr_to_rust_cse(arena, value, var_names, options, cse_constants)?;
        let cond_code = bool_to_rust(arena, cond, var_names, options, cse_constants)?;
        if i == 0 {
            parts.push(format!("if {cond_code} {{ {val_code} }}"));
        } else {
            parts.push(format!("else if {cond_code} {{ {val_code} }}"));
        }
    }
    parts.push(format!("else {{ {} }}", options.precision.nan()));
    Ok(parts.join(" "))
}

/// Convert a boolean expression node to Rust source code.
fn bool_to_rust(
    arena: &Arena,
    id: ExprId,
    var_names: &[&str],
    options: &CodegenOptions,
    cse_constants: &FxHashMap<usize, f64>,
) -> Result<String, SymplexError> {
    match arena.node(id).clone() {
        ExprNode::BoolTrue => Ok("true".to_string()),
        ExprNode::BoolFalse => Ok("false".to_string()),
        ExprNode::Gt(a, b) => {
            let la = expr_to_rust_cse(arena, a, var_names, options, cse_constants)?;
            let lb = expr_to_rust_cse(arena, b, var_names, options, cse_constants)?;
            Ok(format!("({la} > {lb})"))
        }
        ExprNode::Ge(a, b) => {
            let la = expr_to_rust_cse(arena, a, var_names, options, cse_constants)?;
            let lb = expr_to_rust_cse(arena, b, var_names, options, cse_constants)?;
            Ok(format!("({la} >= {lb})"))
        }
        ExprNode::Eq_(a, b) => {
            let la = expr_to_rust_cse(arena, a, var_names, options, cse_constants)?;
            let lb = expr_to_rust_cse(arena, b, var_names, options, cse_constants)?;
            Ok(format!("({la} == {lb})"))
        }
        ExprNode::Ne(a, b) => {
            let la = expr_to_rust_cse(arena, a, var_names, options, cse_constants)?;
            let lb = expr_to_rust_cse(arena, b, var_names, options, cse_constants)?;
            Ok(format!("({la} != {lb})"))
        }
        ExprNode::And(ref children) => {
            let parts: Result<Vec<String>, _> = children
                .iter()
                .map(|&c| bool_to_rust(arena, c, var_names, options, cse_constants))
                .collect();
            Ok(format!("({})", parts?.join(" && ")))
        }
        ExprNode::Or(ref children) => {
            let parts: Result<Vec<String>, _> = children
                .iter()
                .map(|&c| bool_to_rust(arena, c, var_names, options, cse_constants))
                .collect();
            Ok(format!("({})", parts?.join(" || ")))
        }
        ExprNode::Not(inner) => {
            let code = bool_to_rust(arena, inner, var_names, options, cse_constants)?;
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

    // Helper that uses default options (Std backend, f64)
    fn default_opts() -> CodegenOptions {
        CodegenOptions::default()
    }

    #[test]
    fn codegen_simple_number() {
        let mut a = Arena::new();
        let five = a.int(5);
        let code = expr_to_rust(&a, five, &[], &default_opts()).unwrap();
        assert_eq!(code, "5_f64");
    }

    #[test]
    fn codegen_rational_number() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let code = expr_to_rust(&a, half, &[], &default_opts()).unwrap();
        assert_eq!(code, "0.5_f64");
    }

    #[test]
    fn codegen_variable() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let code = expr_to_rust(&a, x, &["x"], &default_opts()).unwrap();
        assert_eq!(code, "x");
    }

    #[test]
    fn codegen_free_symbol_is_error() {
        let mut a = Arena::new();
        let y = sym(&mut a, "y");
        let result = expr_to_rust(&a, y, &["x"], &default_opts());
        assert!(result.is_err());
    }

    #[test]
    fn codegen_pi_and_e() {
        let a = Arena::new();
        let pi_code = expr_to_rust(&a, a.pi, &[], &default_opts()).unwrap();
        assert_eq!(pi_code, "std::f64::consts::PI");

        let e_code = expr_to_rust(&a, a.e_const, &[], &default_opts()).unwrap();
        assert_eq!(e_code, "std::f64::consts::E");
    }

    #[test]
    fn codegen_sin_cos() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let code = expr_to_rust(&a, sin_x, &["x"], &default_opts()).unwrap();
        assert!(code.contains(".sin()"));

        let cos_x = a.cos(x);
        let code = expr_to_rust(&a, cos_x, &["x"], &default_opts()).unwrap();
        assert!(code.contains(".cos()"));
    }

    #[test]
    fn codegen_pow_integer_uses_powi() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let code = expr_to_rust(&a, x2, &["x"], &default_opts()).unwrap();
        assert!(code.contains("powi(2)"), "expected powi(2), got: {code}");
    }

    #[test]
    fn codegen_pow_half_uses_sqrt() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let half = a.rational(1, 2);
        let sqrt_x = a.pow(x, half);
        let code = expr_to_rust(&a, sqrt_x, &["x"], &default_opts()).unwrap();
        assert!(code.contains(".sqrt()"), "expected sqrt(), got: {code}");
    }

    #[test]
    fn codegen_neg() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_x = a.neg(x);
        let code = expr_to_rust(&a, neg_x, &["x"], &default_opts()).unwrap();
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
        let code = expr_to_rust(&a, cse_var, &["x"], &default_opts()).unwrap();
        assert_eq!(code, "t0");
    }

    #[test]
    fn codegen_exp_ln() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let exp_x = a.exp(x);
        let code = expr_to_rust(&a, exp_x, &["x"], &default_opts()).unwrap();
        assert!(code.contains(".exp()"));

        let ln_x = a.ln(x);
        let code = expr_to_rust(&a, ln_x, &["x"], &default_opts()).unwrap();
        assert!(code.contains(".ln()"));
    }

    #[test]
    fn codegen_abs_sign() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let abs_x = a.abs(x);
        let code = expr_to_rust(&a, abs_x, &["x"], &default_opts()).unwrap();
        assert!(code.contains(".abs()"));

        let sign_x = a.sign(x);
        let code = expr_to_rust(&a, sign_x, &["x"], &default_opts()).unwrap();
        assert!(code.contains(".signum()"));
    }

    #[test]
    fn codegen_hyperbolic_functions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        let sinh_x = a.sinh(x);
        assert!(
            expr_to_rust(&a, sinh_x, &["x"], &default_opts())
                .unwrap()
                .contains(".sinh()")
        );

        let cosh_x = a.cosh(x);
        assert!(
            expr_to_rust(&a, cosh_x, &["x"], &default_opts())
                .unwrap()
                .contains(".cosh()")
        );

        let tanh_x = a.tanh(x);
        assert!(
            expr_to_rust(&a, tanh_x, &["x"], &default_opts())
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
            expr_to_rust(&a, asin_x, &["x"], &default_opts())
                .unwrap()
                .contains(".asin()")
        );

        let acos_x = a.acos(x);
        assert!(
            expr_to_rust(&a, acos_x, &["x"], &default_opts())
                .unwrap()
                .contains(".acos()")
        );

        let atan_x = a.atan(x);
        assert!(
            expr_to_rust(&a, atan_x, &["x"], &default_opts())
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
        let code = expr_to_rust(&a, sum, &["x", "y"], &default_opts()).unwrap();
        assert!(code.contains('+'), "expected +, got: {code}");

        let prod = a.mul(&[x, y]);
        let code = expr_to_rust(&a, prod, &["x", "y"], &default_opts()).unwrap();
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
        let result = expr_to_rust(&a, a.i_unit, &[], &default_opts());
        assert!(result.is_err());
    }

    #[test]
    fn codegen_atan2() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let at2 = a.atan2(y, x);
        let code = expr_to_rust(&a, at2, &["x", "y"], &default_opts()).unwrap();
        assert!(code.contains(".atan2("), "expected atan2, got: {code}");
    }

    #[test]
    fn codegen_negative_powi() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_two = a.int(-2);
        let x_neg2 = a.pow(x, neg_two);
        let code = expr_to_rust(&a, x_neg2, &["x"], &default_opts()).unwrap();
        assert!(code.contains("powi(-2)"), "expected powi(-2), got: {code}");
    }

    #[test]
    fn codegen_inverse_hyperbolic() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        let asinh_x = a.asinh(x);
        assert!(
            expr_to_rust(&a, asinh_x, &["x"], &default_opts())
                .unwrap()
                .contains(".asinh()")
        );

        let acosh_x = a.acosh(x);
        assert!(
            expr_to_rust(&a, acosh_x, &["x"], &default_opts())
                .unwrap()
                .contains(".acosh()")
        );

        let atanh_x = a.atanh(x);
        assert!(
            expr_to_rust(&a, atanh_x, &["x"], &default_opts())
                .unwrap()
                .contains(".atanh()")
        );
    }

    // ── New tests for CodegenOptions ───────────────────────────────────

    #[test]
    fn codegen_f32_precision() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let opts = CodegenOptions {
            precision: Precision::F32,
            ..Default::default()
        };
        let code = to_rust_fn_with_options(&mut a, x2, "square", &["x"], &opts).unwrap();
        assert!(
            code.contains("f32"),
            "expected f32 in output, got:\n{code}"
        );
        assert!(
            code.contains("pub fn square(x: f32) -> f32"),
            "expected f32 signature, got:\n{code}"
        );
    }

    #[test]
    fn codegen_f32_consts() {
        let a = Arena::new();
        let opts = CodegenOptions {
            precision: Precision::F32,
            ..Default::default()
        };
        let pi_code = expr_to_rust(&a, a.pi, &[], &opts).unwrap();
        assert_eq!(pi_code, "std::f32::consts::PI");
    }

    #[test]
    fn codegen_inline_annotation() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let opts = CodegenOptions {
            inline: true,
            ..Default::default()
        };
        let code = to_rust_fn_with_options(&mut a, x, "id", &["x"], &opts).unwrap();
        assert!(
            code.contains("#[inline]"),
            "expected #[inline], got:\n{code}"
        );
    }

    #[test]
    fn codegen_must_use_annotation() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let opts = CodegenOptions {
            must_use: true,
            ..Default::default()
        };
        let code = to_rust_fn_with_options(&mut a, x, "id", &["x"], &opts).unwrap();
        assert!(
            code.contains("#[must_use]"),
            "expected #[must_use], got:\n{code}"
        );
    }

    #[test]
    fn codegen_no_must_use() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let opts = CodegenOptions {
            must_use: false,
            ..Default::default()
        };
        let code = to_rust_fn_with_options(&mut a, x, "id", &["x"], &opts).unwrap();
        assert!(
            !code.contains("#[must_use]"),
            "should not have #[must_use], got:\n{code}"
        );
    }

    #[test]
    fn codegen_libm_backend_sin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let opts = CodegenOptions {
            math_backend: MathBackend::Libm,
            ..Default::default()
        };
        let code = expr_to_rust(&a, sin_x, &["x"], &opts).unwrap();
        assert!(
            code.contains("libm::sin("),
            "expected libm::sin, got: {code}"
        );
    }

    #[test]
    fn codegen_cfg_gated_backend() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let opts = CodegenOptions {
            math_backend: MathBackend::CfgGated,
            ..Default::default()
        };
        let code = to_rust_fn_with_options(&mut a, sin_x, "f", &["x"], &opts).unwrap();
        assert!(
            code.contains("mod math {"),
            "expected cfg-gated math module, got:\n{code}"
        );
        assert!(
            code.contains("math::sin("),
            "expected math::sin call, got:\n{code}"
        );
    }

    #[test]
    fn codegen_no_cse_option() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let two = a.int(2);
        let sin_sq = a.pow(sin_x, two);
        let expr = a.add(&[sin_sq, sin_x]);
        let opts = CodegenOptions {
            cse: false,
            ..Default::default()
        };
        let code = to_rust_fn_with_options(&mut a, expr, "f", &["x"], &opts).unwrap();
        // With CSE off, there should be no let tN bindings
        assert!(
            !code.contains("let t"),
            "expected no CSE bindings with cse=false, got:\n{code}"
        );
    }

    #[test]
    fn codegen_matrix_fn() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sin_x = a.sin(x);
        let cos_y = a.cos(y);
        let one = a.int(1);
        let zero = a.int(0);
        let entries = vec![sin_x, cos_y, zero, one];
        let opts = CodegenOptions::default();
        let code =
            matrix_to_rust_fn(&mut a, &entries, 2, 2, "rot", &["x", "y"], &opts).unwrap();
        assert!(
            code.contains("[f64; 4]"),
            "expected [f64; 4] return type, got:\n{code}"
        );
        assert!(
            code.contains("pub fn rot("),
            "expected function name, got:\n{code}"
        );
    }

    // ── Constant folding tests ─────────────────────────────────────

    #[test]
    fn codegen_constant_folds_cos_zero() {
        let mut a = Arena::new();
        let zero = a.int(0);
        let cos_zero = a.cos(zero);
        let code = expr_to_rust(&a, cos_zero, &[], &default_opts()).unwrap();
        assert!(!code.contains(".cos()"), "should constant-fold cos(0): {code}");
        assert!(code.contains("1.0"), "should produce 1.0: {code}");
    }

    #[test]
    fn codegen_constant_folds_sin_zero() {
        let mut a = Arena::new();
        let zero = a.int(0);
        let sin_zero = a.sin(zero);
        let code = expr_to_rust(&a, sin_zero, &[], &default_opts()).unwrap();
        assert!(!code.contains(".sin()"), "should constant-fold sin(0): {code}");
        assert!(code.contains("0.0"), "should produce 0.0: {code}");
    }

    #[test]
    fn codegen_constant_folds_exp_zero() {
        let mut a = Arena::new();
        let zero = a.int(0);
        let exp_zero = a.exp(zero);
        let code = expr_to_rust(&a, exp_zero, &[], &default_opts()).unwrap();
        assert!(!code.contains(".exp()"), "should constant-fold exp(0): {code}");
        assert!(code.contains("1.0"), "should produce 1.0: {code}");
    }

    #[test]
    fn codegen_constant_folds_sin_pi() {
        let mut a = Arena::new();
        let pi = a.pi;
        let sin_pi = a.sin(pi);
        let code = expr_to_rust(&a, sin_pi, &[], &default_opts()).unwrap();
        // sin(pi) ≈ 0 — should be constant-folded (not a .sin() call)
        assert!(!code.contains(".sin()"), "should constant-fold sin(pi): {code}");
    }

    #[test]
    fn codegen_no_fold_for_variable() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let code = expr_to_rust(&a, sin_x, &["x"], &default_opts()).unwrap();
        assert!(code.contains(".sin()"), "variable arg should NOT be folded: {code}");
    }

    #[test]
    fn codegen_neg_one_mul_emits_negation() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_one = a.int(-1);
        let product = a.mul(&[neg_one, x]);
        let code = expr_to_rust(&a, product, &["x"], &default_opts()).unwrap();
        assert!(!code.contains("-1"), "should not contain -1 literal: {code}");
        assert!(code.contains("(-x)"), "should emit (-x): {code}");
    }
}
