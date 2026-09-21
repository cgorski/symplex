//! WebAssembly bindings for symplex symbolic mathematics.
//!
//! Exposes CAS functionality to JavaScript via wasm-bindgen.  Designed for
//! interactive math notebooks and robotics demos.
//!
//! # Two flavours of API
//!
//! * **Stateless functions** — `simplify("x^2 + 2*x + 1")`, `differentiate(
//!   "sin(x)", "x")`, …  Each call parses its string arguments in a fresh
//!   [`Context`] and returns a `String` (or a JSON string for list results).
//!   Errors surface as rejected `JsValue`s carrying a human-readable message.
//! * **[`Session`]** — a long-lived `Context` in which `define("a", "2*x")`
//!   binds names that later `eval`/`simplify`/`latex` calls can use.
//!
//! All arithmetic is exact: decimal literals such as `0.1` in an input
//! string are parsed as the rational `1/10`.

use std::collections::BTreeMap;

use symplex::prelude::*;
use symplex::robotics::DhLink;
use wasm_bindgen::prelude::*;

/// Initialize panic hook for better error messages in browser console.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers (plain `Result<_, String>`, testable on any target)
// ═══════════════════════════════════════════════════════════════════════════

fn js(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn parse_in(ctx: &Context, input: &str) -> Result<Ex, String> {
    symplex::parse::parse(ctx, input).map_err(|e| format!("Parse error: {e}"))
}

/// Split a comma-separated parameter list, trimming whitespace and
/// dropping empty entries.
fn split_params(params: &str) -> Vec<&str> {
    params
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

fn json_strings(items: impl IntoIterator<Item = String>) -> String {
    let v: Vec<String> = items.into_iter().collect();
    serde_json::to_string(&v).unwrap_or_else(|_| "[]".to_string())
}

mod core {
    use super::*;

    pub fn simplify(expr: &str) -> Result<String, String> {
        let ctx = Context::new();
        Ok(parse_in(&ctx, expr)?.simplify().to_string())
    }

    pub fn simplify_latex(expr: &str) -> Result<String, String> {
        let ctx = Context::new();
        Ok(parse_in(&ctx, expr)?.simplify().to_latex())
    }

    pub fn expand(expr: &str) -> Result<String, String> {
        let ctx = Context::new();
        Ok(parse_in(&ctx, expr)?.expand().to_string())
    }

    pub fn factor(expr: &str, var: &str) -> Result<String, String> {
        let ctx = Context::new();
        let e = parse_in(&ctx, expr)?;
        let v = ctx.symbol(var);
        Ok(e.factor(&v).to_string())
    }

    pub fn differentiate(expr: &str, var: &str) -> Result<String, String> {
        let ctx = Context::new();
        let e = parse_in(&ctx, expr)?;
        let v = ctx.symbol(var);
        Ok(e.diff(&v).to_string())
    }

    pub fn differentiate_latex(expr: &str, var: &str) -> Result<String, String> {
        let ctx = Context::new();
        let e = parse_in(&ctx, expr)?;
        let v = ctx.symbol(var);
        Ok(e.diff(&v).to_latex())
    }

    pub fn integrate(expr: &str, var: &str) -> Result<String, String> {
        let ctx = Context::new();
        let e = parse_in(&ctx, expr)?;
        let v = ctx.symbol(var);
        Ok(e.integrate(&v).to_string())
    }

    pub fn integrate_definite(expr: &str, var: &str, lo: &str, hi: &str) -> Result<String, String> {
        let ctx = Context::new();
        let e = parse_in(&ctx, expr)?;
        let v = ctx.symbol(var);
        let lo = parse_in(&ctx, lo)?;
        let hi = parse_in(&ctx, hi)?;
        Ok(e.integrate_definite(&v, &lo, &hi).to_string())
    }

    pub fn solve(expr: &str, var: &str) -> Result<String, String> {
        let ctx = Context::new();
        let e = parse_in(&ctx, expr)?;
        let v = ctx.symbol(var);
        let roots = e.solve(&v).map_err(|e| format!("Solve error: {e}"))?;
        Ok(json_strings(roots.iter().map(ToString::to_string)))
    }

    pub fn limit(expr: &str, var: &str, point: &str) -> Result<String, String> {
        let ctx = Context::new();
        let e = parse_in(&ctx, expr)?;
        let v = ctx.symbol(var);
        let p = parse_in(&ctx, point)?;
        Ok(e.limit(&v, &p).to_string())
    }

    pub fn series(expr: &str, var: &str, point: &str, order: u32) -> Result<String, String> {
        let ctx = Context::new();
        let e = parse_in(&ctx, expr)?;
        let v = ctx.symbol(var);
        let p = parse_in(&ctx, point)?;
        Ok(e.series(&v, &p, order).to_string())
    }

    pub fn to_latex(expr: &str) -> Result<String, String> {
        let ctx = Context::new();
        Ok(parse_in(&ctx, expr)?.to_latex())
    }

    pub fn pretty(expr: &str) -> Result<String, String> {
        let ctx = Context::new();
        Ok(parse_in(&ctx, expr)?.pretty())
    }

    pub fn eval_f64(expr: &str) -> Result<String, String> {
        let ctx = Context::new();
        let v = parse_in(&ctx, expr)?
            .eval_f64()
            .map_err(|e| format!("Evaluation error: {e}"))?;
        Ok(v.to_string())
    }

    pub fn eval_decimal(expr: &str, digits: u32) -> Result<String, String> {
        let ctx = Context::new();
        parse_in(&ctx, expr)?
            .eval_decimal(digits)
            .map_err(|e| format!("Evaluation error: {e}"))
    }

    pub fn eval_numeric(expr: &str) -> Result<String, String> {
        let ctx = Context::new();
        let evaled = parse_in(&ctx, expr)?.eval();
        Ok(match evaled.eval_f64() {
            Ok(v) => v.to_string(),
            Err(_) => evaled.to_string(),
        })
    }

    pub fn to_rust_fn(expr: &str, name: &str, params: &str) -> Result<String, String> {
        let ctx = Context::new();
        let e = parse_in(&ctx, expr)?;
        e.to_rust_fn(name, &split_params(params))
            .map_err(|e| format!("Codegen error: {e}"))
    }

    pub fn to_c_fn(expr: &str, name: &str, params: &str) -> Result<String, String> {
        let ctx = Context::new();
        let e = parse_in(&ctx, expr)?;
        e.to_c_fn(name, &split_params(params))
            .map_err(|e| format!("Codegen error: {e}"))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stateless exports
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a math expression string and return its LaTeX representation.
#[wasm_bindgen]
pub fn parse_to_latex(input: &str) -> Result<String, JsValue> {
    core::to_latex(input).map_err(js)
}

/// Alias of [`parse_to_latex`].
#[wasm_bindgen]
pub fn to_latex(input: &str) -> Result<String, JsValue> {
    core::to_latex(input).map_err(js)
}

/// Render an expression as Unicode "pretty" text (fractions, superscripts).
#[wasm_bindgen]
pub fn pretty(expr_str: &str) -> Result<String, JsValue> {
    core::pretty(expr_str).map_err(js)
}

/// Differentiate an expression and return the result as a string.
#[wasm_bindgen]
pub fn differentiate(expr_str: &str, var_name: &str) -> Result<String, JsValue> {
    core::differentiate(expr_str, var_name).map_err(js)
}

/// Differentiate an expression and return LaTeX.
#[wasm_bindgen]
pub fn differentiate_latex(expr_str: &str, var_name: &str) -> Result<String, JsValue> {
    core::differentiate_latex(expr_str, var_name).map_err(js)
}

/// Indefinite integral `∫ expr d(var)`; unevaluated `Integral(...)` if no
/// closed form is found.
#[wasm_bindgen]
pub fn integrate(expr_str: &str, var_name: &str) -> Result<String, JsValue> {
    core::integrate(expr_str, var_name).map_err(js)
}

/// Definite integral `∫_lo^hi expr d(var)`.  Bounds are expression strings
/// (`"0"`, `"pi/2"`, `"oo"`, `"-oo"`).
#[wasm_bindgen]
pub fn integrate_definite(
    expr_str: &str,
    var_name: &str,
    lo: &str,
    hi: &str,
) -> Result<String, JsValue> {
    core::integrate_definite(expr_str, var_name, lo, hi).map_err(js)
}

/// Simplify an expression and return the result as a string.
#[wasm_bindgen]
pub fn simplify(expr_str: &str) -> Result<String, JsValue> {
    core::simplify(expr_str).map_err(js)
}

/// Simplify an expression and return LaTeX.
#[wasm_bindgen]
pub fn simplify_latex(expr_str: &str) -> Result<String, JsValue> {
    core::simplify_latex(expr_str).map_err(js)
}

/// Expand an expression and return the result as a string.
#[wasm_bindgen]
pub fn expand(expr_str: &str) -> Result<String, JsValue> {
    core::expand(expr_str).map_err(js)
}

/// Factor a polynomial in `var`.
#[wasm_bindgen]
pub fn factor(expr_str: &str, var_name: &str) -> Result<String, JsValue> {
    core::factor(expr_str, var_name).map_err(js)
}

/// Solve `expr = 0` for a variable.  Returns a JSON array of solution
/// strings; an equation with no solution yields `[]`.
#[wasm_bindgen]
pub fn solve(expr_str: &str, var_name: &str) -> Result<String, JsValue> {
    core::solve(expr_str, var_name).map_err(js)
}

/// `lim_{var → point} expr`.  `point` may be `"oo"` / `"-oo"`.
#[wasm_bindgen]
pub fn limit(expr_str: &str, var_name: &str, point: &str) -> Result<String, JsValue> {
    core::limit(expr_str, var_name, point).map_err(js)
}

/// Taylor/Laurent series of `expr` in `var` about `point`, up to (and
/// excluding) `order`.
#[wasm_bindgen]
pub fn series(expr_str: &str, var_name: &str, point: &str, order: u32) -> Result<String, JsValue> {
    core::series(expr_str, var_name, point, order).map_err(js)
}

/// Evaluate a constant expression to an `f64`, returned as its shortest
/// round-trip decimal string.  Fails on free symbols or complex results.
#[wasm_bindgen]
pub fn eval_f64(expr_str: &str) -> Result<String, JsValue> {
    core::eval_f64(expr_str).map_err(js)
}

/// Evaluate a constant expression to `digits` significant decimal digits
/// (arbitrary precision).
#[wasm_bindgen]
pub fn eval_decimal(expr_str: &str, digits: u32) -> Result<String, JsValue> {
    core::eval_decimal(expr_str, digits).map_err(js)
}

/// Evaluate an expression numerically if possible, otherwise return the
/// exactly-evaluated symbolic form.
#[wasm_bindgen]
pub fn eval_numeric(expr_str: &str) -> Result<String, JsValue> {
    core::eval_numeric(expr_str).map_err(js)
}

/// Generate a Rust function `fn name(params…) -> f64` computing `expr`.
/// `params` is a comma-separated list of parameter names (`"x, y"`).
#[wasm_bindgen]
pub fn to_rust_fn(expr_str: &str, name: &str, params: &str) -> Result<String, JsValue> {
    core::to_rust_fn(expr_str, name, params).map_err(js)
}

/// Generate a C function `double name(double params…)` computing `expr`.
/// `params` is a comma-separated list of parameter names.
#[wasm_bindgen]
pub fn to_c_fn(expr_str: &str, name: &str, params: &str) -> Result<String, JsValue> {
    core::to_c_fn(expr_str, name, params).map_err(js)
}

// ═══════════════════════════════════════════════════════════════════════════
// Session — a persistent Context with named definitions
// ═══════════════════════════════════════════════════════════════════════════

/// A persistent symbolic session.
///
/// All expressions live in one [`Context`], so symbols keep their identity
/// (and assumptions) across calls, and `define` bindings are substituted
/// into later inputs.
///
/// ```js
/// const s = new Session();
/// s.define("r", "2");
/// s.define("area", "pi * r^2");
/// s.eval("area");          // "4*pi"
/// s.eval_f64("area");      // "12.566370614359172"
/// s.latex("area / 2");     // "2 \pi"
/// ```
#[wasm_bindgen]
#[derive(Default)]
pub struct Session {
    ctx: Context,
    defs: BTreeMap<String, Ex>,
}

impl Session {
    fn parse(&self, input: &str) -> Result<Ex, String> {
        parse_in(&self.ctx, input)
    }

    /// Parse `input` and substitute every definition (simultaneously).
    ///
    /// Definitions may themselves refer to earlier definitions; those were
    /// already resolved when they were stored.
    fn resolve(&self, input: &str) -> Result<Ex, String> {
        let e = self.parse(input)?;
        if self.defs.is_empty() {
            return Ok(e);
        }
        let syms: Vec<Ex> = self.defs.keys().map(|k| self.ctx.symbol(k)).collect();
        let pairs: Vec<(&Ex, &Ex)> = syms.iter().zip(self.defs.values()).collect();
        Ok(e.subs_map(&pairs))
    }

    fn define_impl(&mut self, name: &str, expr: &str) -> Result<String, String> {
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(format!(
                "invalid name {name:?}: use letters, digits and underscores"
            ));
        }
        let value = self.resolve(expr)?;
        if value.free_symbols().iter().any(|s| s.to_string() == name) {
            return Err(format!("definition of {name} refers to itself"));
        }
        let shown = value.to_string();
        self.defs.insert(name.to_string(), value);
        Ok(shown)
    }

    fn get_impl(&self, name: &str) -> Result<String, String> {
        self.defs
            .get(name)
            .map(ToString::to_string)
            .ok_or_else(|| format!("{name} is not defined"))
    }

    fn eval_impl(&self, expr: &str) -> Result<String, String> {
        Ok(self.resolve(expr)?.eval().to_string())
    }

    fn simplify_impl(&self, expr: &str) -> Result<String, String> {
        Ok(self.resolve(expr)?.simplify().to_string())
    }

    fn latex_impl(&self, expr: &str) -> Result<String, String> {
        Ok(self.resolve(expr)?.to_latex())
    }

    fn eval_f64_impl(&self, expr: &str) -> Result<String, String> {
        self.resolve(expr)?
            .eval_f64()
            .map(|v| v.to_string())
            .map_err(|e| format!("Evaluation error: {e}"))
    }

    fn diff_impl(&self, expr: &str, var: &str) -> Result<String, String> {
        let e = self.resolve(expr)?;
        let v = self.ctx.symbol(var);
        Ok(e.diff(&v).to_string())
    }

    fn solve_impl(&self, expr: &str, var: &str) -> Result<String, String> {
        let e = self.resolve(expr)?;
        let v = self.ctx.symbol(var);
        let roots = e.solve(&v).map_err(|e| format!("Solve error: {e}"))?;
        Ok(json_strings(roots.iter().map(ToString::to_string)))
    }
}

#[wasm_bindgen]
impl Session {
    /// Create an empty session.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Session {
        Session::default()
    }

    /// Bind `name` to the expression `expr_str` (parsed in this session, with
    /// existing definitions substituted).  Returns the stored value as a
    /// string.  Redefining a name replaces the old binding.
    pub fn define(&mut self, name: &str, expr_str: &str) -> Result<String, JsValue> {
        self.define_impl(name, expr_str).map_err(js)
    }

    /// The stored definition of `name`, or an error if it is not defined.
    pub fn get(&self, name: &str) -> Result<String, JsValue> {
        self.get_impl(name).map_err(js)
    }

    /// Remove a definition.  Returns `true` if it existed.
    pub fn undefine(&mut self, name: &str) -> bool {
        self.defs.remove(name).is_some()
    }

    /// Remove all definitions (symbols and their assumptions survive).
    pub fn clear(&mut self) {
        self.defs.clear();
    }

    /// JSON array of the currently defined names (sorted).
    pub fn names(&self) -> String {
        json_strings(self.defs.keys().cloned())
    }

    /// Parse `expr_str`, substitute definitions, evaluate exactly.
    pub fn eval(&self, expr_str: &str) -> Result<String, JsValue> {
        self.eval_impl(expr_str).map_err(js)
    }

    /// Parse, substitute definitions, and simplify.
    pub fn simplify(&self, expr_str: &str) -> Result<String, JsValue> {
        self.simplify_impl(expr_str).map_err(js)
    }

    /// Parse, substitute definitions, and render as LaTeX.
    pub fn latex(&self, expr_str: &str) -> Result<String, JsValue> {
        self.latex_impl(expr_str).map_err(js)
    }

    /// Parse, substitute definitions, and evaluate to an `f64` string.
    pub fn eval_f64(&self, expr_str: &str) -> Result<String, JsValue> {
        self.eval_f64_impl(expr_str).map_err(js)
    }

    /// Parse, substitute definitions, and differentiate with respect to `var`.
    pub fn diff(&self, expr_str: &str, var_name: &str) -> Result<String, JsValue> {
        self.diff_impl(expr_str, var_name).map_err(js)
    }

    /// Parse, substitute definitions, and solve `expr = 0` for `var`
    /// (JSON array of solution strings).
    pub fn solve(&self, expr_str: &str, var_name: &str) -> Result<String, JsValue> {
        self.solve_impl(expr_str, var_name).map_err(js)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Robotics demos
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a DH parameter given as `f64` to the exact rational a human
/// meant (`0.3 → 3/10`), via [`Context::from_f64_approx`].
fn dh_param(ctx: &Context, v: f64) -> Result<Ex, String> {
    ctx.from_f64_approx(v, 1_000_000)
        .map_err(|e| format!("invalid DH parameter {v}: {e}"))
}

fn build_dh(ctx: &Context, config: &DhConfig) -> Result<Vec<(Ex, Ex, Ex, Ex)>, String> {
    config
        .joints
        .iter()
        .map(|j| {
            Ok((
                ctx.symbol(&j.theta),
                dh_param(ctx, j.d)?,
                dh_param(ctx, j.a)?,
                dh_param(ctx, j.alpha)?,
            ))
        })
        .collect()
}

fn matrix_latex_json(m: &symplex::matrix::Matrix) -> String {
    let mut rows = Vec::new();
    for i in 0..m.nrows() {
        let mut row = Vec::new();
        for j in 0..m.ncols() {
            row.push(m.get(i, j).simplify().to_latex());
        }
        rows.push(row);
    }
    serde_json::to_string(&rows).unwrap_or_default()
}

fn compute_fk_impl(dh_json: &str) -> Result<String, String> {
    let config: DhConfig = serde_json::from_str(dh_json).map_err(|e| format!("JSON error: {e}"))?;
    let ctx = Context::new();
    let dh_params = build_dh(&ctx, &config)?;
    let dh_refs: Vec<DhLink<'_>> = dh_params
        .iter()
        .map(|(t, d, a, al)| DhLink {
            theta: t,
            d,
            a,
            alpha: al,
        })
        .collect();
    let fk = symplex::robotics::fk_chain(&dh_refs);
    Ok(matrix_latex_json(&fk))
}

fn compute_jacobian_impl(dh_json: &str) -> Result<String, String> {
    let config: DhConfig = serde_json::from_str(dh_json).map_err(|e| format!("JSON error: {e}"))?;
    let ctx = Context::new();
    let dh_params = build_dh(&ctx, &config)?;
    let dh_refs: Vec<DhLink<'_>> = dh_params
        .iter()
        .map(|(t, d, a, al)| DhLink {
            theta: t,
            d,
            a,
            alpha: al,
        })
        .collect();
    let (x, y, z) = symplex::robotics::fk_position(&dh_refs);
    let theta_vars: Vec<Ex> = config.joints.iter().map(|j| ctx.symbol(&j.theta)).collect();
    let theta_refs: Vec<&Ex> = theta_vars.iter().collect();
    let jac = symplex::matrix::jacobian(&[&x, &y, &z], &theta_refs);
    Ok(matrix_latex_json(&jac))
}

fn generate_jacobian_code_impl(dh_json: &str) -> Result<String, String> {
    let config: DhConfig = serde_json::from_str(dh_json).map_err(|e| format!("JSON error: {e}"))?;
    let ctx = Context::new();
    let dh_params = build_dh(&ctx, &config)?;
    let dh_refs: Vec<DhLink<'_>> = dh_params
        .iter()
        .map(|(t, d, a, al)| DhLink {
            theta: t,
            d,
            a,
            alpha: al,
        })
        .collect();
    let (x, y, z) = symplex::robotics::fk_position(&dh_refs);
    let theta_vars: Vec<Ex> = config.joints.iter().map(|j| ctx.symbol(&j.theta)).collect();
    let theta_refs: Vec<&Ex> = theta_vars.iter().collect();
    let param_names: Vec<&str> = config.joints.iter().map(|j| j.theta.as_str()).collect();
    let jac = symplex::matrix::jacobian(&[&x, &y, &z], &theta_refs);
    jac.to_rust_fn("jacobian", &param_names)
        .map_err(|e| format!("Codegen error: {e}"))
}

/// Compute forward kinematics from a DH parameter table (JSON).
///
/// Input JSON format:
/// ```json
/// {
///   "joints": [
///     {"theta": "theta1", "d": 0, "a": 0.3, "alpha": 0},
///     {"theta": "theta2", "d": 0, "a": 0.25, "alpha": 0}
///   ]
/// }
/// ```
///
/// Numeric parameters are converted to exact rationals (`0.3 → 3/10`).
/// Returns a JSON 2D array of LaTeX strings representing the 4×4 FK matrix.
#[wasm_bindgen]
pub fn compute_fk(dh_json: &str) -> Result<String, JsValue> {
    compute_fk_impl(dh_json).map_err(js)
}

/// Compute the Jacobian from DH parameters.
///
/// Takes the same JSON format as `compute_fk`. Returns a JSON 2D array of
/// LaTeX strings representing the 3×N Jacobian (position rows only).
#[wasm_bindgen]
pub fn compute_jacobian(dh_json: &str) -> Result<String, JsValue> {
    compute_jacobian_impl(dh_json).map_err(js)
}

/// Generate optimized Rust code for the Jacobian from DH parameters.
///
/// Takes the same JSON format as `compute_fk`. Returns a string containing
/// a Rust function that computes the Jacobian numerically.
#[wasm_bindgen]
pub fn generate_jacobian_code(dh_json: &str) -> Result<String, JsValue> {
    generate_jacobian_code_impl(dh_json).map_err(js)
}

// ── Internal types ─────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct DhConfig {
    joints: Vec<DhJoint>,
}

#[derive(serde::Deserialize)]
struct DhJoint {
    theta: String,
    #[serde(default)]
    d: f64,
    #[serde(default)]
    a: f64,
    #[serde(default)]
    alpha: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests (native; exercise the `Result<_, String>` layer)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplify_expand_factor() {
        assert_eq!(core::simplify("sin(x)^2 + cos(x)^2").unwrap(), "1");
        assert_eq!(core::expand("(x + 1)^2").unwrap(), "x^2 + 2*x + 1");
        assert_eq!(core::factor("x^2 - 1", "x").unwrap(), "(x - 1)*(x + 1)");
    }

    #[test]
    fn calculus() {
        assert_eq!(core::differentiate("x^3", "x").unwrap(), "3*x^2");
        assert_eq!(core::integrate("2*x", "x").unwrap(), "x^2");
        assert_eq!(core::integrate_definite("x", "x", "0", "2").unwrap(), "2");
        assert_eq!(
            core::integrate_definite("exp(-x^2)", "x", "-oo", "oo").unwrap(),
            "sqrt(pi)"
        );
        assert_eq!(core::limit("sin(x)/x", "x", "0").unwrap(), "1");
        let s = core::series("exp(x)", "x", "0", 3).unwrap();
        assert!(s.contains("1/2*x^2"), "{s}");
    }

    #[test]
    fn solve_returns_json_array() {
        let out = core::solve("x^2 - 4", "x").unwrap();
        let v: Vec<String> = serde_json::from_str(&out).unwrap();
        let mut v = v;
        v.sort();
        assert_eq!(v, vec!["-2", "2"]);
    }

    #[test]
    fn output_formats() {
        assert_eq!(core::to_latex("x^2/2").unwrap(), r"\frac{1}{2} x^{2}");
        let p = core::pretty("x/2").unwrap();
        assert!(p.lines().count() >= 2, "{p}");
        assert_eq!(core::eval_f64("1/4").unwrap(), "0.25");
        // evalf truncates rather than rounds the last digit.
        let pi = core::eval_decimal("pi", 10).unwrap();
        assert!(pi.starts_with("3.14159265"), "{pi}");
        assert_eq!(core::eval_numeric("2*x").unwrap(), "2*x");
        assert_eq!(core::eval_numeric("2*3").unwrap(), "6");
    }

    #[test]
    fn decimals_are_exact() {
        assert_eq!(core::simplify("0.1 + 0.2").unwrap(), "3/10");
    }

    #[test]
    fn codegen() {
        let rust = core::to_rust_fn("x^2 + y", "f", "x, y").unwrap();
        assert!(rust.contains("fn f(x: f64, y: f64) -> f64"), "{rust}");
        let c = core::to_c_fn("x^2", "sq", "x").unwrap();
        assert!(c.contains("double sq(double x)"), "{c}");
        assert!(core::to_rust_fn("x + y", "f", "x").is_err());
    }

    #[test]
    fn errors_are_strings() {
        assert!(
            core::simplify("x +")
                .unwrap_err()
                .starts_with("Parse error")
        );
        assert!(
            core::eval_f64("x")
                .unwrap_err()
                .starts_with("Evaluation error")
        );
        assert!(core::eval_f64("sqrt(-1)").is_err());
    }

    #[test]
    fn session_define_get_eval() {
        let mut s = Session::new();
        assert_eq!(s.define_impl("r", "2").unwrap(), "2");
        assert_eq!(s.define_impl("area", "pi*r^2").unwrap(), "4*pi");
        assert_eq!(s.get_impl("area").unwrap(), "4*pi");
        assert_eq!(s.eval_impl("area / 2").unwrap(), "2*pi");
        assert_eq!(s.eval_f64_impl("r + 1").unwrap(), "3");
        assert_eq!(s.names(), r#"["area","r"]"#);
        assert!(s.get_impl("nope").is_err());
    }

    #[test]
    fn session_symbols_persist_and_redefine() {
        let mut s = Session::new();
        s.define_impl("f", "x^2 + x").unwrap();
        assert_eq!(s.diff_impl("f", "x").unwrap(), "2*x + 1");
        assert_eq!(s.simplify_impl("f - x",).unwrap(), "x^2");
        s.define_impl("f", "x^3").unwrap();
        assert_eq!(s.eval_impl("f").unwrap(), "x^3");
        assert!(s.undefine("f"));
        assert!(!s.undefine("f"));
        assert_eq!(s.eval_impl("f").unwrap(), "f");
    }

    #[test]
    fn session_solve_and_latex() {
        let mut s = Session::new();
        s.define_impl("c", "9").unwrap();
        let roots: Vec<String> =
            serde_json::from_str(&s.solve_impl("x^2 - c", "x").unwrap()).unwrap();
        assert_eq!(roots.len(), 2);
        assert_eq!(s.latex_impl("c/2").unwrap(), r"\frac{9}{2}");
        s.clear();
        assert_eq!(s.names(), "[]");
    }

    #[test]
    fn session_rejects_bad_definitions() {
        let mut s = Session::new();
        assert!(s.define_impl("", "1").is_err());
        assert!(s.define_impl("a b", "1").is_err());
        assert!(s.define_impl("x", "x + 1").is_err());
        assert!(s.define_impl("y", "1 +").is_err());
    }

    #[test]
    fn fk_uses_exact_dh_params() {
        let json = r#"{"joints":[{"theta":"q1","a":0.3},{"theta":"q2","a":0.25}]}"#;
        let fk = compute_fk_impl(json).unwrap();
        let rows: Vec<Vec<String>> = serde_json::from_str(&fk).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].len(), 4);
        let joined = fk.replace(' ', "");
        assert!(joined.contains("3}{10}") || joined.contains("3/10"), "{fk}");
        let jac = compute_jacobian_impl(json).unwrap();
        let jrows: Vec<Vec<String>> = serde_json::from_str(&jac).unwrap();
        assert_eq!((jrows.len(), jrows[0].len()), (3, 2));
        let code = generate_jacobian_code_impl(json).unwrap();
        assert!(code.contains("fn jacobian("), "{code}");
        assert!(compute_fk_impl("{").is_err());
    }
}
