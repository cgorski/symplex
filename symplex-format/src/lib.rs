//! LaTeX and MathML rendering for symplex symbolic expressions.
//!
//! This crate provides the [`ToLatex`] and [`ToMathML`] traits for converting
//! symplex expressions into LaTeX and MathML markup strings.
//!
//! # Quick Start
//!
//! ```rust
//! use symplex::prelude::*;
//! use symplex_format::ToLatex;
//!
//! let x = symplex::var("x");
//! let expr = x.powi(2);
//! assert_eq!(expr.to_latex(), r"x^{2}");
//! ```

use symplex::expr::{Ex, ExprType};
use symplex::matrix::Matrix;

// ═══════════════════════════════════════════════════════════════════════════
// Traits
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for converting symplex types to LaTeX strings.
pub trait ToLatex {
    /// Render as a LaTeX math expression (no delimiters).
    fn to_latex(&self) -> String;

    /// Render as an inline LaTeX expression with `$` delimiters.
    fn to_latex_inline(&self) -> String {
        format!("${}$", self.to_latex())
    }

    /// Render as a display LaTeX expression with `$$` delimiters.
    fn to_latex_display(&self) -> String {
        format!("$${}$$", self.to_latex())
    }
}

/// Trait for converting symplex types to MathML strings.
pub trait ToMathML {
    /// Render as MathML markup.
    fn to_mathml(&self) -> String;
}

// ═══════════════════════════════════════════════════════════════════════════
// Greek letter mapping
// ═══════════════════════════════════════════════════════════════════════════

/// Greek letter names that should be rendered as LaTeX commands.
const GREEK_LETTERS: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    // Uppercase variants
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
    "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
    // Common variants
    "varepsilon", "varphi", "vartheta", "varrho", "varsigma",
];

/// Check if a name is a Greek letter and return the LaTeX command if so.
fn greek_to_latex(name: &str) -> Option<String> {
    // Check for subscript patterns like "alpha_1" — handle the base name
    if let Some(idx) = name.find('_') {
        let base = &name[..idx];
        let sub = &name[idx + 1..];
        if let Some(latex_base) = greek_to_latex(base) {
            return Some(format!("{}_{{{}}}", latex_base, sub));
        }
        return None;
    }

    for &greek in GREEK_LETTERS {
        if name == greek {
            return Some(format!("\\{}", greek));
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// LaTeX function name mapping
// ═══════════════════════════════════════════════════════════════════════════

/// Standard math functions that get a backslash in LaTeX.
const LATEX_FUNC_NAMES: &[&str] = &[
    "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "coth",
    "arcsin", "arccos", "arctan",
    "exp", "ln", "log", "lg",
    "det", "dim", "ker", "hom",
    "lim", "sup", "inf", "max", "min",
    "arg", "deg", "gcd",
];

/// Map a function display name to LaTeX representation.
fn func_name_to_latex(name: &str) -> String {
    // Handle inverse trig: asin → \arcsin, acos → \arccos, atan → \arctan
    match name {
        "asin" => return r"\arcsin".to_string(),
        "acos" => return r"\arccos".to_string(),
        "atan" => return r"\arctan".to_string(),
        "atan2" => return r"\operatorname{atan2}".to_string(),
        "asinh" => return r"\operatorname{asinh}".to_string(),
        "acosh" => return r"\operatorname{acosh}".to_string(),
        "atanh" => return r"\operatorname{atanh}".to_string(),
        "abs" => return String::new(), // handled specially with \left| ... \right|
        "sign" => return r"\operatorname{sgn}".to_string(),
        "floor" => return String::new(), // handled specially with \lfloor ... \rfloor
        "ceiling" => return String::new(), // handled specially with \lceil ... \rceil
        "erf" => return r"\operatorname{erf}".to_string(),
        "erfc" => return r"\operatorname{erfc}".to_string(),
        "Gamma" => return r"\Gamma".to_string(),
        "LogGamma" => return r"\ln \Gamma".to_string(),
        "Digamma" => return r"\psi".to_string(),
        "H" => return r"\operatorname{H}".to_string(),
        "DiracDelta" => return r"\delta".to_string(),
        _ => {}
    }

    for &f in LATEX_FUNC_NAMES {
        if name == f {
            return format!("\\{}", name);
        }
    }

    // Unknown function → \operatorname{name}
    format!("\\operatorname{{{}}}", name)
}

// ═══════════════════════════════════════════════════════════════════════════
// Display string parsing helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Extract the function name from a Display string like "sin(x + 1)".
/// Returns the function name and whether it was found.
fn extract_func_name(display: &str) -> Option<&str> {
    // Look for pattern: name(
    if let Some(paren_pos) = display.find('(') {
        let candidate = &display[..paren_pos];
        // Make sure the candidate is a valid identifier (no operators/spaces before the paren)
        if !candidate.is_empty()
            && !candidate.contains(' ')
            && !candidate.contains('+')
            && !candidate.contains('-')
            && !candidate.contains('*')
            && !candidate.contains('/')
            && !candidate.contains('^')
        {
            return Some(candidate);
        }
    }
    None
}

/// Check if a display string represents a negative number (like "-3" or "-5/7").
fn is_negative_display(s: &str) -> bool {
    s.starts_with('-')
}

// ═══════════════════════════════════════════════════════════════════════════
// Core rendering
// ═══════════════════════════════════════════════════════════════════════════

impl ToLatex for Ex {
    fn to_latex(&self) -> String {
        render_latex(self, false)
    }
}

/// Render an expression to LaTeX.
/// `parent_is_tight` indicates whether the parent context needs compact rendering
/// (e.g., inside a fraction numerator/denominator).
fn render_latex(expr: &Ex, _parent_is_tight: bool) -> String {
    match expr.expr_type() {
        ExprType::Number => render_number(expr),
        ExprType::Symbol => render_symbol(expr),
        ExprType::Constant => render_constant(expr),
        ExprType::Add => render_add(expr),
        ExprType::Mul => render_mul(expr),
        ExprType::Pow => render_pow(expr),
        ExprType::Neg => render_neg(expr),
        ExprType::Function => render_function(expr),
        ExprType::Apply => render_apply(expr),
        ExprType::Derivative => render_derivative(expr),
        ExprType::Integral => render_integral(expr),
        ExprType::Set => format!("{expr}"), // Fallback for sets
    }
}

// ── Number ─────────────────────────────────────────────────────────────

fn render_number(expr: &Ex) -> String {
    let s = format!("{expr}");

    // Check for rational: "p/q"
    if let Some(slash_pos) = s.find('/') {
        let numer = &s[..slash_pos];
        let denom = &s[slash_pos + 1..];
        // Handle negative rationals like "-5/7" → render as -\frac{5}{7}
        if let Some(stripped) = numer.strip_prefix('-') {
            return format!("-\\frac{{{}}}{{{}}}", stripped, denom);
        }
        return format!("\\frac{{{}}}{{{}}}", numer, denom);
    }

    // Plain integer
    s
}

// ── Symbol ─────────────────────────────────────────────────────────────

fn render_symbol(expr: &Ex) -> String {
    let name = format!("{expr}");

    // Check for Greek letter
    if let Some(latex) = greek_to_latex(&name) {
        return latex;
    }

    // Handle subscripts: "x_1" → "x_{1}", "x_12" → "x_{12}"
    if let Some(idx) = name.find('_') {
        let base = &name[..idx];
        let sub = &name[idx + 1..];
        let latex_base = greek_to_latex(base).unwrap_or_else(|| base.to_string());
        return format!("{}_{{{}}}", latex_base, sub);
    }

    name
}

// ── Constant ───────────────────────────────────────────────────────────

fn render_constant(expr: &Ex) -> String {
    let s = format!("{expr}");
    match s.as_str() {
        "pi" => r"\pi".to_string(),
        "E" => "e".to_string(),
        "I" => "i".to_string(),
        "oo" => r"\infty".to_string(),
        "-oo" => r"-\infty".to_string(),
        "zoo" => r"\tilde{\infty}".to_string(),
        "nan" => r"\text{NaN}".to_string(),
        "True" => r"\text{True}".to_string(),
        "False" => r"\text{False}".to_string(),
        _ => s,
    }
}

// ── Add ────────────────────────────────────────────────────────────────

fn render_add(expr: &Ex) -> String {
    let args = expr.args();
    if args.is_empty() {
        return "0".to_string();
    }

    let mut result = String::new();

    for (i, arg) in args.iter().enumerate() {
        let arg_type = arg.expr_type();
        let arg_display = format!("{arg}");

        if arg_type == ExprType::Neg {
            // This is a negated term: render as " - inner"
            let inner_args = arg.args();
            if !inner_args.is_empty() {
                let inner_latex = render_latex(&inner_args[0], false);
                if i == 0 {
                    result.push_str(&format!("-{}", inner_latex));
                } else {
                    result.push_str(&format!(" - {}", inner_latex));
                }
            } else {
                // Fallback
                let latex = render_latex(arg, false);
                if i == 0 {
                    result.push_str(&latex);
                } else {
                    result.push_str(&format!(" + {}", latex));
                }
            }
        } else if is_neg_coefficient_mul(arg) {
            // Mul with negative leading coefficient: e.g., -2*x
            let latex = render_neg_coeff_mul_as_subtraction(arg);
            if i == 0 {
                result.push_str(&format!("-{}", latex));
            } else {
                result.push_str(&format!(" - {}", latex));
            }
        } else if i == 0 {
            // First non-negative term
            let latex = render_latex(arg, false);
            result.push_str(&latex);
        } else {
            let latex = render_latex(arg, false);
            // Check if the rendered latex starts with "-" (e.g., negative number)
            if is_negative_display(&arg_display) && arg_type == ExprType::Number {
                // Render as subtraction
                let pos_latex = render_number_abs(arg);
                result.push_str(&format!(" - {}", pos_latex));
            } else {
                result.push_str(&format!(" + {}", latex));
            }
        }
    }

    result
}

/// Check if an expression is a Mul with a negative leading coefficient.
fn is_neg_coefficient_mul(expr: &Ex) -> bool {
    if expr.expr_type() != ExprType::Mul {
        return false;
    }
    let args = expr.args();
    if args.is_empty() {
        return false;
    }
    let first = &args[0];
    if first.expr_type() == ExprType::Number {
        let s = format!("{first}");
        return s.starts_with('-');
    }
    // Check if first arg is Neg
    first.expr_type() == ExprType::Neg
}

/// Render a Mul with a negative coefficient as a subtraction (without the minus sign).
/// E.g., Mul(-2, x) → "2 x" (the caller adds the " - " prefix).
fn render_neg_coeff_mul_as_subtraction(expr: &Ex) -> String {
    let args = expr.args();
    if args.is_empty() {
        return String::new();
    }
    let first = &args[0];
    if first.expr_type() == ExprType::Number {
        let s = format!("{first}");
        if let Some(stripped) = s.strip_prefix('-') {
            // Rebuild the mul with positive coefficient
            if stripped == "1" {
                // -1 * rest → just render rest
                if args.len() == 2 {
                    return render_latex(&args[1], false);
                } else {
                    let rest: Vec<String> = args[1..].iter().map(|a| render_mul_factor(a)).collect();
                    return rest.join(" ");
                }
            } else {
                // -N * rest → N * rest
                let coeff = stripped.to_string();
                if args.len() == 1 {
                    return coeff;
                }
                let rest: Vec<String> = args[1..].iter().map(|a| render_mul_factor(a)).collect();
                let mut parts = vec![coeff];
                parts.extend(rest);
                return parts.join(" ");
            }
        }
    }
    // Fallback: render normally but strip leading minus
    let full = render_latex(expr, false);
    full.strip_prefix('-').unwrap_or(&full).to_string()
}

/// Render the absolute value of a number (strip leading minus).
fn render_number_abs(expr: &Ex) -> String {
    let s = render_number(expr);
    if let Some(stripped) = s.strip_prefix('-') {
        stripped.to_string()
    } else {
        s
    }
}

// ── Mul ────────────────────────────────────────────────────────────────

fn render_mul(expr: &Ex) -> String {
    let args = expr.args();
    if args.is_empty() {
        return "1".to_string();
    }

    // Check for negative-one leading factor
    let first_display = format!("{}", args[0]);
    let (skip_first, prefix) = if args.len() > 1 && first_display == "-1" {
        (true, "-")
    } else if args.len() > 1 && first_display == "1" {
        (true, "")
    } else {
        (false, "")
    };

    let start_idx = if skip_first { 1 } else { 0 };

    // Separate numerator factors and denominator factors
    // Denominator factors: Pow(x, -n) where n > 0
    let mut numer_factors: Vec<String> = Vec::new();
    let mut denom_factors: Vec<String> = Vec::new();

    for arg in &args[start_idx..] {
        if let Some((base, neg_exp)) = extract_negative_power(arg) {
            // This is x^(-n) → goes to denominator as x^n
            if neg_exp == "1" {
                denom_factors.push(base);
            } else {
                denom_factors.push(format!("{}^{{{}}}", base, neg_exp));
            }
        } else {
            numer_factors.push(render_mul_factor(arg));
        }
    }

    if denom_factors.is_empty() {
        // Pure product — no fractions
        let body = if numer_factors.is_empty() {
            "1".to_string()
        } else {
            join_mul_factors(&numer_factors)
        };
        format!("{}{}", prefix, body)
    } else {
        // Has denominator factors → render as \frac
        let numer_str = if numer_factors.is_empty() {
            "1".to_string()
        } else {
            join_mul_factors(&numer_factors)
        };
        let denom_str = join_mul_factors(&denom_factors);
        if prefix == "-" {
            format!("-\\frac{{{}}}{{{}}}", numer_str, denom_str)
        } else {
            format!("\\frac{{{}}}{{{}}}", numer_str, denom_str)
        }
    }
}

/// Join multiplication factors with implicit multiplication (space) or \cdot.
fn join_mul_factors(factors: &[String]) -> String {
    if factors.len() == 1 {
        return factors[0].clone();
    }

    let mut result = String::new();
    for (i, f) in factors.iter().enumerate() {
        if i > 0 {
            // Use \cdot between two numbers, space otherwise
            let prev = &factors[i - 1];
            if looks_like_number(prev) && looks_like_number(f) {
                result.push_str(r" \cdot ");
            } else {
                result.push(' ');
            }
        }
        result.push_str(f);
    }
    result
}

/// Heuristic: does this rendered string look like a plain number?
fn looks_like_number(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let s = s.strip_prefix('-').unwrap_or(s);
    s.chars().all(|c| c.is_ascii_digit())
}

/// Render a single factor in a multiplication.
fn render_mul_factor(expr: &Ex) -> String {
    let latex = render_latex(expr, false);
    match expr.expr_type() {
        ExprType::Add => format!("\\left({}\\right)", latex),
        _ => latex,
    }
}

/// Check if an expression is Pow(base, negative_exponent) and return
/// (rendered_base, positive_exponent_str) if so.
fn extract_negative_power(expr: &Ex) -> Option<(String, String)> {
    if expr.expr_type() != ExprType::Pow {
        return None;
    }
    let args = expr.args();
    if args.len() != 2 {
        return None;
    }
    let exp = &args[1];
    let exp_display = format!("{exp}");

    // Check for negative integer exponent: "-2", "-3", etc.
    if let Some(stripped) = exp_display.strip_prefix('-') {
        if stripped.chars().all(|c| c.is_ascii_digit()) {
            let base_latex = render_pow_base(&args[0]);
            return Some((base_latex, stripped.to_string()));
        }
    }

    // Check for negative rational exponent: "-1/2", etc.
    if exp_display.starts_with('-') && exp_display.contains('/') {
        let pos_exp = &exp_display[1..];
        if let Some(slash) = pos_exp.find('/') {
            let n = &pos_exp[..slash];
            let d = &pos_exp[slash + 1..];
            let base_latex = render_pow_base(&args[0]);
            return Some((base_latex, format!("\\frac{{{}}}{{{}}}", n, d)));
        }
    }

    None
}

// ── Pow ────────────────────────────────────────────────────────────────

fn render_pow(expr: &Ex) -> String {
    let args = expr.args();
    if args.len() != 2 {
        return format!("{expr}");
    }

    let base = &args[0];
    let exp = &args[1];
    let exp_display = format!("{exp}");

    // Special case: exp = 1/2 → \sqrt{base}
    if exp_display == "1/2" {
        let base_latex = render_latex(base, false);
        return format!("\\sqrt{{{}}}", base_latex);
    }

    // Special case: exp = 1/3 → \sqrt[3]{base}
    if exp_display == "1/3" {
        let base_latex = render_latex(base, false);
        return format!("\\sqrt[3]{{{}}}", base_latex);
    }

    // Special case: exp = 1/n → \sqrt[n]{base}
    if exp.expr_type() == ExprType::Number {
        if let Some(slash_pos) = exp_display.find('/') {
            let numer = &exp_display[..slash_pos];
            let denom = &exp_display[slash_pos + 1..];
            if numer == "1" {
                let base_latex = render_latex(base, false);
                return format!("\\sqrt[{}]{{{}}}", denom, base_latex);
            }
        }
    }

    // Special case: exp = -1 → \frac{1}{base}
    if exp_display == "-1" {
        let base_latex = render_latex(base, false);
        return format!("\\frac{{1}}{{{}}}", base_latex);
    }

    // Special case: exp = -n (negative integer) → \frac{1}{base^{n}}
    if let Some(stripped) = exp_display.strip_prefix('-') {
        if stripped.chars().all(|c| c.is_ascii_digit()) && stripped != "1" {
            let base_latex = render_pow_base(base);
            return format!("\\frac{{1}}{{{}^{{{}}}}}", base_latex, stripped);
        }
    }

    // General case: base^{exp}
    let base_latex = render_pow_base(base);
    let exp_latex = render_latex(exp, false);
    format!("{}^{{{}}}", base_latex, exp_latex)
}

/// Render a base expression for use in Pow, adding parens if needed.
fn render_pow_base(base: &Ex) -> String {
    let latex = render_latex(base, false);
    match base.expr_type() {
        ExprType::Add | ExprType::Mul | ExprType::Neg => {
            format!("\\left({}\\right)", latex)
        }
        ExprType::Function => {
            // Functions are fine without parens in base position
            latex
        }
        _ => latex,
    }
}

// ── Neg ────────────────────────────────────────────────────────────────

fn render_neg(expr: &Ex) -> String {
    let args = expr.args();
    if args.is_empty() {
        return format!("{expr}");
    }
    let inner = &args[0];
    let inner_latex = render_latex(inner, false);
    match inner.expr_type() {
        ExprType::Add => format!("-\\left({}\\right)", inner_latex),
        _ => format!("-{}", inner_latex),
    }
}

// ── Function ───────────────────────────────────────────────────────────

fn render_function(expr: &Ex) -> String {
    let s = format!("{expr}");
    let args = expr.args();

    // Try to identify the function name from the display string
    let func_name = match extract_func_name(&s) {
        Some(name) => name,
        None => {
            // Might be a factorial: "n!" or "(n+1)!"
            if s.ends_with('!') {
                return render_factorial(expr, &s);
            }
            return s;
        }
    };

    // Special case: abs → \left| ... \right|
    if func_name == "abs" {
        let inner_latex = if !args.is_empty() {
            render_latex(&args[0], false)
        } else {
            extract_func_arg_display(&s, "abs")
        };
        return format!("\\left|{}\\right|", inner_latex);
    }

    // Special case: floor → \lfloor ... \rfloor
    if func_name == "floor" {
        let inner_latex = if !args.is_empty() {
            render_latex(&args[0], false)
        } else {
            extract_func_arg_display(&s, "floor")
        };
        return format!("\\lfloor {}\\rfloor", inner_latex);
    }

    // Special case: ceiling → \lceil ... \rceil
    if func_name == "ceiling" {
        let inner_latex = if !args.is_empty() {
            render_latex(&args[0], false)
        } else {
            extract_func_arg_display(&s, "ceiling")
        };
        return format!("\\lceil {}\\rceil", inner_latex);
    }

    // Special case: C(n, k) → \binom{n}{k}
    if func_name == "C" && args.len() == 2 {
        let n_latex = render_latex(&args[0], false);
        let k_latex = render_latex(&args[1], false);
        return format!("\\binom{{{}}}{{{}}}", n_latex, k_latex);
    }

    // Special case: B(a, b) → \mathrm{B}(a, b)
    if func_name == "B" && args.len() == 2 {
        let a_latex = render_latex(&args[0], false);
        let b_latex = render_latex(&args[1], false);
        return format!("\\mathrm{{B}}\\left({}, {}\\right)", a_latex, b_latex);
    }

    // Special case: Sum(body, var=lo..hi)
    if func_name == "Sum" && args.len() == 4 {
        let body_latex = render_latex(&args[0], false);
        let var_latex = render_latex(&args[1], false);
        let lo_latex = render_latex(&args[2], false);
        let hi_latex = render_latex(&args[3], false);
        return format!(
            "\\sum_{{{}={}}}^{{{}}} {}",
            var_latex, lo_latex, hi_latex, body_latex
        );
    }

    // Special case: Product(body, var=lo..hi)
    if func_name == "Product" && args.len() == 4 {
        let body_latex = render_latex(&args[0], false);
        let var_latex = render_latex(&args[1], false);
        let lo_latex = render_latex(&args[2], false);
        let hi_latex = render_latex(&args[3], false);
        return format!(
            "\\prod_{{{}={}}}^{{{}}} {}",
            var_latex, lo_latex, hi_latex, body_latex
        );
    }

    // Special case: min/max with multiple args
    if func_name == "min" || func_name == "max" {
        let latex_name = format!("\\{}", func_name);
        let inner: Vec<String> = args.iter().map(|a| render_latex(a, false)).collect();
        return format!("{}\\left({}\\right)", latex_name, inner.join(", "));
    }

    // Special case: Piecewise
    if func_name == "Piecewise" {
        return render_piecewise(expr);
    }

    // General function rendering
    let latex_name = func_name_to_latex(func_name);

    let inner_latex = if args.len() == 1 {
        render_latex(&args[0], false)
    } else if args.is_empty() {
        extract_func_arg_display(&s, func_name)
    } else {
        args.iter()
            .map(|a| render_latex(a, false))
            .collect::<Vec<_>>()
            .join(", ")
    };

    format!("{}\\left({}\\right)", latex_name, inner_latex)
}

/// Extract the argument text from a display string like "sin(x + 1)" → "x + 1".
fn extract_func_arg_display(display: &str, func_name: &str) -> String {
    let prefix = format!("{}(", func_name);
    if let Some(rest) = display.strip_prefix(&prefix) {
        if let Some(inner) = rest.strip_suffix(')') {
            return inner.to_string();
        }
    }
    display.to_string()
}

/// Render a factorial expression.
fn render_factorial(expr: &Ex, _display: &str) -> String {
    let args = expr.args();
    if !args.is_empty() {
        let inner = &args[0];
        let inner_latex = render_latex(inner, false);
        match inner.expr_type() {
            ExprType::Number | ExprType::Symbol | ExprType::Constant => {
                format!("{}!", inner_latex)
            }
            _ => format!("\\left({}\\right)!", inner_latex),
        }
    } else {
        format!("{expr}")
    }
}

/// Render a piecewise expression.
fn render_piecewise(expr: &Ex) -> String {
    let args = expr.args();
    // Piecewise args come in pairs: (value, condition)
    // Display format: Piecewise(val1 if cond1, val2 if cond2, ...)
    // We render as LaTeX cases environment
    if args.is_empty() {
        return format!("{expr}");
    }
    // args from the Piecewise node are pairs stored as: val1, cond1, val2, cond2, ...
    // But actually, looking at the display code, Piecewise stores (val, cond) tuples
    // in the node. Through args() these become flattened children.
    // Let's handle both even and odd lengths gracefully.
    let mut result = String::from("\\begin{cases}\n");
    let mut i = 0;
    while i + 1 < args.len() {
        let val = render_latex(&args[i], false);
        let cond = render_latex(&args[i + 1], false);
        result.push_str(&format!("  {} & \\text{{if }} {}", val, cond));
        if i + 2 < args.len() {
            result.push_str(" \\\\\n");
        } else {
            result.push('\n');
        }
        i += 2;
    }
    result.push_str("\\end{cases}");
    result
}

// ── Apply ──────────────────────────────────────────────────────────────

fn render_apply(expr: &Ex) -> String {
    let s = format!("{expr}");
    let args = expr.args();

    // Apply display: "f(x, y)"
    // Try to extract the function name
    if let Some(name) = extract_func_name(&s) {
        let inner: Vec<String> = args.iter().map(|a| render_latex(a, false)).collect();
        return format!("{}\\left({}\\right)", name, inner.join(", "));
    }

    // Fallback
    s
}

// ── Derivative ─────────────────────────────────────────────────────────

fn render_derivative(expr: &Ex) -> String {
    let args = expr.args();
    if args.len() >= 2 {
        let body = render_latex(&args[0], false);
        let var = render_latex(&args[1], false);
        format!("\\frac{{d}}{{d{}}} {}", var, body)
    } else {
        format!("{expr}")
    }
}

// ── Integral ───────────────────────────────────────────────────────────

fn render_integral(expr: &Ex) -> String {
    let args = expr.args();
    if args.len() >= 2 {
        let body = render_latex(&args[0], false);
        let var = render_latex(&args[1], false);
        format!("\\int {} \\, d{}", body, var)
    } else {
        format!("{expr}")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MathML implementation
// ═══════════════════════════════════════════════════════════════════════════

impl ToMathML for Ex {
    fn to_mathml(&self) -> String {
        let inner = render_mathml(self);
        format!("<math xmlns=\"http://www.w3.org/1998/Math/MathML\">{}</math>", inner)
    }
}

fn render_mathml(expr: &Ex) -> String {
    match expr.expr_type() {
        ExprType::Number => render_mathml_number(expr),
        ExprType::Symbol => render_mathml_symbol(expr),
        ExprType::Constant => render_mathml_constant(expr),
        ExprType::Add => render_mathml_add(expr),
        ExprType::Mul => render_mathml_mul(expr),
        ExprType::Pow => render_mathml_pow(expr),
        ExprType::Neg => render_mathml_neg(expr),
        ExprType::Function => render_mathml_function(expr),
        ExprType::Apply => render_mathml_apply(expr),
        ExprType::Derivative | ExprType::Integral | ExprType::Set => {
            // Fallback: wrap display text in <mtext>
            format!("<mtext>{}</mtext>", format!("{expr}"))
        }
    }
}

fn render_mathml_number(expr: &Ex) -> String {
    let s = format!("{expr}");
    if let Some(slash) = s.find('/') {
        let n = &s[..slash];
        let d = &s[slash + 1..];
        format!(
            "<mfrac><mn>{}</mn><mn>{}</mn></mfrac>",
            n, d
        )
    } else {
        format!("<mn>{}</mn>", s)
    }
}

fn render_mathml_symbol(expr: &Ex) -> String {
    let name = format!("{expr}");
    // Greek letters get <mi> with the Unicode character
    let display = match name.as_str() {
        "alpha" => "α",
        "beta" => "β",
        "gamma" => "γ",
        "delta" => "δ",
        "epsilon" => "ε",
        "zeta" => "ζ",
        "eta" => "η",
        "theta" => "θ",
        "lambda" => "λ",
        "mu" => "μ",
        "nu" => "ν",
        "xi" => "ξ",
        "pi" => "π",
        "rho" => "ρ",
        "sigma" => "σ",
        "tau" => "τ",
        "phi" => "φ",
        "chi" => "χ",
        "psi" => "ψ",
        "omega" => "ω",
        _ => &name,
    };
    format!("<mi>{}</mi>", display)
}

fn render_mathml_constant(expr: &Ex) -> String {
    let s = format!("{expr}");
    match s.as_str() {
        "pi" => "<mi>π</mi>".to_string(),
        "E" => "<mi>e</mi>".to_string(),
        "I" => "<mi>i</mi>".to_string(),
        "oo" => "<mi>∞</mi>".to_string(),
        "-oo" => "<mrow><mo>-</mo><mi>∞</mi></mrow>".to_string(),
        _ => format!("<mi>{}</mi>", s),
    }
}

fn render_mathml_add(expr: &Ex) -> String {
    let args = expr.args();
    let mut parts = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            if arg.expr_type() == ExprType::Neg {
                parts.push("<mo>-</mo>".to_string());
                let inner = arg.args();
                if !inner.is_empty() {
                    parts.push(render_mathml(&inner[0]));
                }
                continue;
            } else {
                parts.push("<mo>+</mo>".to_string());
            }
        }
        parts.push(render_mathml(arg));
    }
    format!("<mrow>{}</mrow>", parts.join(""))
}

fn render_mathml_mul(expr: &Ex) -> String {
    let args = expr.args();
    let mut parts = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            parts.push("<mo>⁢</mo>".to_string()); // invisible times
        }
        parts.push(render_mathml(arg));
    }
    format!("<mrow>{}</mrow>", parts.join(""))
}

fn render_mathml_pow(expr: &Ex) -> String {
    let args = expr.args();
    if args.len() != 2 {
        return format!("<mtext>{}</mtext>", format!("{expr}"));
    }
    let base = render_mathml(&args[0]);
    let exp_display = format!("{}", args[1]);

    // sqrt
    if exp_display == "1/2" {
        return format!("<msqrt>{}</msqrt>", render_mathml(&args[0]));
    }

    let exp = render_mathml(&args[1]);
    format!("<msup>{}{}</msup>", base, exp)
}

fn render_mathml_neg(expr: &Ex) -> String {
    let args = expr.args();
    if args.is_empty() {
        return format!("<mtext>{}</mtext>", format!("{expr}"));
    }
    let inner = render_mathml(&args[0]);
    format!("<mrow><mo>-</mo>{}</mrow>", inner)
}

fn render_mathml_function(expr: &Ex) -> String {
    let s = format!("{expr}");
    let args = expr.args();

    if let Some(name) = extract_func_name(&s) {
        if name == "abs" && !args.is_empty() {
            let inner = render_mathml(&args[0]);
            return format!("<mrow><mo>|</mo>{}<mo>|</mo></mrow>", inner);
        }

        let inner: Vec<String> = args.iter().map(|a| render_mathml(a)).collect();
        return format!(
            "<mrow><mi>{}</mi><mo>⁡</mo><mfenced>{}</mfenced></mrow>",
            name,
            inner.join("")
        );
    }

    format!("<mtext>{}</mtext>", s)
}

fn render_mathml_apply(expr: &Ex) -> String {
    let s = format!("{expr}");
    let args = expr.args();

    if let Some(name) = extract_func_name(&s) {
        let inner: Vec<String> = args.iter().map(|a| render_mathml(a)).collect();
        return format!(
            "<mrow><mi>{}</mi><mo>⁡</mo><mfenced>{}</mfenced></mrow>",
            name,
            inner.join("")
        );
    }

    format!("<mtext>{}</mtext>", s)
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix implementations
// ═══════════════════════════════════════════════════════════════════════════

impl ToLatex for Matrix {
    fn to_latex(&self) -> String {
        let mut s = String::from(r"\begin{bmatrix}");
        for i in 0..self.nrows() {
            if i > 0 {
                s.push_str(r" \\ ");
            } else {
                s.push(' ');
            }
            for j in 0..self.ncols() {
                if j > 0 {
                    s.push_str(" & ");
                }
                s.push_str(&self.get(i, j).to_latex());
            }
        }
        s.push_str(r" \end{bmatrix}");
        s
    }
}

impl ToMathML for Matrix {
    fn to_mathml(&self) -> String {
        let mut s = String::from("<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mfenced open=\"[\" close=\"]\"><mtable>");
        for i in 0..self.nrows() {
            s.push_str("<mtr>");
            for j in 0..self.ncols() {
                s.push_str("<mtd>");
                s.push_str(&render_mathml(self.get(i, j)));
                s.push_str("</mtd>");
            }
            s.push_str("</mtr>");
        }
        s.push_str("</mtable></mfenced></math>");
        s
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Number tests ───────────────────────────────────────────────

    #[test]
    fn latex_integer() {
        assert_eq!(symplex::int(42).to_latex(), "42");
    }

    #[test]
    fn latex_zero() {
        assert_eq!(symplex::int(0).to_latex(), "0");
    }

    #[test]
    fn latex_negative_integer() {
        // Negative integers are typically rendered as Neg(42)
        let neg = symplex::int(-3);
        let latex = neg.to_latex();
        assert!(latex == "-3" || latex == "-3", "got: {latex}");
    }

    #[test]
    fn latex_fraction() {
        let half = symplex::rational(1, 2);
        assert_eq!(half.to_latex(), r"\frac{1}{2}");
    }

    #[test]
    fn latex_negative_fraction() {
        let neg_frac = symplex::rational(-5, 7);
        let latex = neg_frac.to_latex();
        // Could be rendered as -\frac{5}{7} or \frac{-5}{7}
        assert!(
            latex == r"-\frac{5}{7}" || latex == r"\frac{-5}{7}",
            "got: {latex}"
        );
    }

    // ── Symbol tests ───────────────────────────────────────────────

    #[test]
    fn latex_symbol() {
        assert_eq!(symplex::var("x").to_latex(), "x");
    }

    #[test]
    fn latex_symbol_multichar() {
        assert_eq!(symplex::var("foo").to_latex(), "foo");
    }

    #[test]
    fn latex_greek_theta() {
        assert_eq!(symplex::var("theta").to_latex(), r"\theta");
    }

    #[test]
    fn latex_greek_alpha() {
        assert_eq!(symplex::var("alpha").to_latex(), r"\alpha");
    }

    #[test]
    fn latex_greek_omega() {
        assert_eq!(symplex::var("omega").to_latex(), r"\omega");
    }

    #[test]
    fn latex_greek_lambda() {
        assert_eq!(symplex::var("lambda").to_latex(), r"\lambda");
    }

    #[test]
    fn latex_symbol_subscript() {
        let x1 = symplex::var("x_1");
        assert_eq!(x1.to_latex(), "x_{1}");
    }

    // ── Constant tests ─────────────────────────────────────────────

    #[test]
    fn latex_pi() {
        assert_eq!(symplex::pi().to_latex(), r"\pi");
    }

    #[test]
    fn latex_e() {
        assert_eq!(symplex::e().to_latex(), "e");
    }

    #[test]
    fn latex_imaginary() {
        assert_eq!(symplex::i_unit().to_latex(), "i");
    }

    #[test]
    fn latex_infinity() {
        assert_eq!(symplex::infinity().to_latex(), r"\infty");
    }

    #[test]
    fn latex_neg_infinity() {
        assert_eq!(symplex::neg_infinity().to_latex(), r"-\infty");
    }

    // ── Pow tests ──────────────────────────────────────────────────

    #[test]
    fn latex_power() {
        let x = symplex::var("x");
        let expr = x.powi(2);
        assert_eq!(expr.to_latex(), r"x^{2}");
    }

    #[test]
    fn latex_power_cube() {
        let x = symplex::var("x");
        let expr = x.powi(3);
        assert_eq!(expr.to_latex(), r"x^{3}");
    }

    #[test]
    fn latex_sqrt() {
        let x = symplex::var("x");
        let expr = x.sqrt();
        assert_eq!(expr.to_latex(), r"\sqrt{x}");
    }

    #[test]
    fn latex_cbrt() {
        let x = symplex::var("x");
        let expr = x.cbrt();
        assert_eq!(expr.to_latex(), r"\sqrt[3]{x}");
    }

    #[test]
    fn latex_inverse() {
        let x = symplex::var("x");
        let expr = x.powi(-1);
        let latex = expr.to_latex();
        assert_eq!(latex, r"\frac{1}{x}");
    }

    // ── Neg tests ──────────────────────────────────────────────────

    #[test]
    fn latex_neg_symbol() {
        let x = symplex::var("x");
        let expr = -&x;
        assert_eq!(expr.to_latex(), "-x");
    }

    // ── Add tests ──────────────────────────────────────────────────

    #[test]
    fn latex_add_simple() {
        let x = symplex::var("x");
        let expr = &x + 1;
        let latex = expr.to_latex();
        // Order may vary due to canonicalization
        assert!(
            latex == "x + 1" || latex == "1 + x",
            "got: {latex}"
        );
    }

    #[test]
    fn latex_add_power_and_const() {
        let x = symplex::var("x");
        let expr = x.powi(2) + 1;
        let latex = expr.to_latex();
        assert!(
            latex == r"x^{2} + 1" || latex == r"1 + x^{2}",
            "got: {latex}"
        );
    }

    // ── Mul tests ──────────────────────────────────────────────────

    #[test]
    fn latex_mul_coefficient() {
        let x = symplex::var("x");
        let expr = &x * 2;
        let latex = expr.to_latex();
        assert!(
            latex == "2 x" || latex == "x \\cdot 2" || latex == "2x",
            "got: {latex}"
        );
    }

    // ── Trig function tests ────────────────────────────────────────

    #[test]
    fn latex_sin() {
        let x = symplex::var("x");
        let expr = x.sin();
        assert_eq!(expr.to_latex(), r"\sin\left(x\right)");
    }

    #[test]
    fn latex_cos() {
        let x = symplex::var("x");
        let expr = x.cos();
        assert_eq!(expr.to_latex(), r"\cos\left(x\right)");
    }

    #[test]
    fn latex_tan() {
        let x = symplex::var("x");
        let expr = x.tan();
        assert_eq!(expr.to_latex(), r"\tan\left(x\right)");
    }

    #[test]
    fn latex_sinh() {
        let x = symplex::var("x");
        let expr = x.sinh();
        assert_eq!(expr.to_latex(), r"\sinh\left(x\right)");
    }

    #[test]
    fn latex_cosh() {
        let x = symplex::var("x");
        let expr = x.cosh();
        assert_eq!(expr.to_latex(), r"\cosh\left(x\right)");
    }

    // ── Exp/Ln tests ───────────────────────────────────────────────

    #[test]
    fn latex_exp() {
        let x = symplex::var("x");
        let expr = x.exp();
        assert_eq!(expr.to_latex(), r"\exp\left(x\right)");
    }

    #[test]
    fn latex_ln() {
        let x = symplex::var("x");
        let expr = x.ln();
        assert_eq!(expr.to_latex(), r"\ln\left(x\right)");
    }

    // ── Abs test ───────────────────────────────────────────────────

    #[test]
    fn latex_abs() {
        let x = symplex::var("x");
        let expr = x.abs();
        assert_eq!(expr.to_latex(), r"\left|x\right|");
    }

    // ── Derivative test ────────────────────────────────────────────

    #[test]
    fn latex_derivative() {
        let x = symplex::var("x");
        let expr = x.powi(2).formal_diff(&x);
        let latex = expr.to_latex();
        assert_eq!(latex, r"\frac{d}{dx} x^{2}");
    }

    // ── Integral test ──────────────────────────────────────────────

    #[test]
    fn latex_integral() {
        let x = symplex::var("x");
        // sin(sin(x)) has no elementary antiderivative, so integrate()
        // returns an unevaluated Integral node.
        let expr = x.sin().sin();
        let integral = expr.integrate(&x);
        let latex = integral.to_latex();
        if integral.expr_type() == symplex::expr::ExprType::Integral {
            assert!(latex.starts_with(r"\int"), "got: {latex}");
            assert!(latex.ends_with(r"\, dx"), "got: {latex}");
        }
        // Either way, should produce valid non-empty LaTeX
        assert!(!latex.is_empty());
    }

    // ── Matrix test ────────────────────────────────────────────────

    #[test]
    fn latex_matrix_2x2() {
        let a = symplex::int(1);
        let b = symplex::int(2);
        let c = symplex::int(3);
        let d = symplex::int(4);
        let m = Matrix::new(vec![vec![a, b], vec![c, d]]);
        let latex = m.to_latex();
        assert_eq!(
            latex,
            r"\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}"
        );
    }

    #[test]
    fn latex_matrix_symbolic() {
        let x = symplex::var("x");
        let y = symplex::var("y");
        let m = Matrix::new(vec![vec![x, y]]);
        let latex = m.to_latex();
        assert_eq!(latex, r"\begin{bmatrix} x & y \end{bmatrix}");
    }

    // ── Delimiter tests ────────────────────────────────────────────

    #[test]
    fn latex_inline_delimiters() {
        let x = symplex::var("x");
        assert_eq!(x.to_latex_inline(), "$x$");
    }

    #[test]
    fn latex_display_delimiters() {
        let x = symplex::var("x");
        assert_eq!(x.to_latex_display(), "$$x$$");
    }

    // ── MathML tests ───────────────────────────────────────────────

    #[test]
    fn mathml_integer() {
        let expr = symplex::int(42);
        let mathml = expr.to_mathml();
        assert!(mathml.contains("<mn>42</mn>"), "got: {mathml}");
        assert!(mathml.starts_with("<math"), "got: {mathml}");
    }

    #[test]
    fn mathml_symbol() {
        let x = symplex::var("x");
        let mathml = x.to_mathml();
        assert!(mathml.contains("<mi>x</mi>"), "got: {mathml}");
    }

    #[test]
    fn mathml_greek_symbol() {
        let theta = symplex::var("theta");
        let mathml = theta.to_mathml();
        assert!(mathml.contains("<mi>θ</mi>"), "got: {mathml}");
    }

    #[test]
    fn mathml_pi_constant() {
        let pi = symplex::pi();
        let mathml = pi.to_mathml();
        assert!(mathml.contains("<mi>π</mi>"), "got: {mathml}");
    }

    #[test]
    fn mathml_sqrt() {
        let x = symplex::var("x");
        let expr = x.sqrt();
        let mathml = expr.to_mathml();
        assert!(mathml.contains("<msqrt>"), "got: {mathml}");
    }

    #[test]
    fn mathml_matrix() {
        let a = symplex::int(1);
        let b = symplex::int(2);
        let m = Matrix::new(vec![vec![a, b]]);
        let mathml = m.to_mathml();
        assert!(mathml.contains("<mtable>"), "got: {mathml}");
        assert!(mathml.contains("<mn>1</mn>"), "got: {mathml}");
        assert!(mathml.contains("<mn>2</mn>"), "got: {mathml}");
    }

    // ── Inverse trig ───────────────────────────────────────────────

    #[test]
    fn latex_asin() {
        let x = symplex::var("x");
        let expr = x.asin();
        assert_eq!(expr.to_latex(), r"\arcsin\left(x\right)");
    }

    #[test]
    fn latex_acos() {
        let x = symplex::var("x");
        let expr = x.acos();
        assert_eq!(expr.to_latex(), r"\arccos\left(x\right)");
    }

    #[test]
    fn latex_atan() {
        let x = symplex::var("x");
        let expr = x.atan();
        assert_eq!(expr.to_latex(), r"\arctan\left(x\right)");
    }

    // ── Compound expressions ───────────────────────────────────────

    #[test]
    fn latex_sin_squared() {
        let x = symplex::var("x");
        let expr = x.sin().powi(2);
        let latex = expr.to_latex();
        assert_eq!(latex, r"\sin\left(x\right)^{2}");
    }

    #[test]
    fn latex_nested_function() {
        let x = symplex::var("x");
        let expr = x.sin().exp();
        let latex = expr.to_latex();
        assert_eq!(latex, r"\exp\left(\sin\left(x\right)\right)");
    }
}
