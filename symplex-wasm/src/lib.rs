//! WebAssembly bindings for symplex symbolic mathematics.
//!
//! Exposes CAS functionality to JavaScript via wasm-bindgen.
//! Designed for interactive math notebooks and robotics demos.

use wasm_bindgen::prelude::*;
use symplex::prelude::*;

/// Initialize panic hook for better error messages in browser console.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Parse a math expression string and return its LaTeX representation.
#[wasm_bindgen]
pub fn parse_to_latex(input: &str) -> Result<String, JsValue> {
    let ctx = Context::new();
    match symplex::parse::parse(&ctx, input) {
        Ok(expr) => Ok(expr.to_latex()),
        Err(e) => Err(JsValue::from_str(&format!("Parse error: {e}"))),
    }
}

/// Differentiate an expression and return the result as a string.
#[wasm_bindgen]
pub fn differentiate(expr_str: &str, var_name: &str) -> Result<String, JsValue> {
    let ctx = Context::new();
    let expr = symplex::parse::parse(&ctx, expr_str)
        .map_err(|e| JsValue::from_str(&format!("Parse error: {e}")))?;
    let var = ctx.symbol(var_name);
    let result = expr.diff(&var);
    Ok(format!("{result}"))
}

/// Differentiate an expression and return LaTeX.
#[wasm_bindgen]
pub fn differentiate_latex(expr_str: &str, var_name: &str) -> Result<String, JsValue> {
    let ctx = Context::new();
    let expr = symplex::parse::parse(&ctx, expr_str)
        .map_err(|e| JsValue::from_str(&format!("Parse error: {e}")))?;
    let var = ctx.symbol(var_name);
    let result = expr.diff(&var);
    Ok(result.to_latex())
}

/// Integrate an expression and return the result as a string.
#[wasm_bindgen]
pub fn integrate(expr_str: &str, var_name: &str) -> Result<String, JsValue> {
    let ctx = Context::new();
    let expr = symplex::parse::parse(&ctx, expr_str)
        .map_err(|e| JsValue::from_str(&format!("Parse error: {e}")))?;
    let var = ctx.symbol(var_name);
    let result = expr.integrate(&var);
    Ok(format!("{result}"))
}

/// Simplify an expression and return the result as a string.
#[wasm_bindgen]
pub fn simplify(expr_str: &str) -> Result<String, JsValue> {
    let ctx = Context::new();
    let expr = symplex::parse::parse(&ctx, expr_str)
        .map_err(|e| JsValue::from_str(&format!("Parse error: {e}")))?;
    let result = expr.simplify();
    Ok(format!("{result}"))
}

/// Simplify an expression and return LaTeX.
#[wasm_bindgen]
pub fn simplify_latex(expr_str: &str) -> Result<String, JsValue> {
    let ctx = Context::new();
    let expr = symplex::parse::parse(&ctx, expr_str)
        .map_err(|e| JsValue::from_str(&format!("Parse error: {e}")))?;
    let result = expr.simplify();
    Ok(result.to_latex())
}

/// Expand an expression and return the result as a string.
#[wasm_bindgen]
pub fn expand(expr_str: &str) -> Result<String, JsValue> {
    let ctx = Context::new();
    let expr = symplex::parse::parse(&ctx, expr_str)
        .map_err(|e| JsValue::from_str(&format!("Parse error: {e}")))?;
    let result = expr.expand();
    Ok(format!("{result}"))
}

/// Solve an equation (expr = 0) for a variable. Returns a JSON array of solution strings.
#[wasm_bindgen]
pub fn solve(expr_str: &str, var_name: &str) -> Result<String, JsValue> {
    let ctx = Context::new();
    let expr = symplex::parse::parse(&ctx, expr_str)
        .map_err(|e| JsValue::from_str(&format!("Parse error: {e}")))?;
    let var = ctx.symbol(var_name);
    let roots = expr.solve_or_empty(&var);
    let result: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    Ok(serde_json::to_string(&result).unwrap_or_default())
}

/// Evaluate an expression numerically.
#[wasm_bindgen]
pub fn eval_numeric(expr_str: &str) -> Result<String, JsValue> {
    let ctx = Context::new();
    let expr = symplex::parse::parse(&ctx, expr_str)
        .map_err(|e| JsValue::from_str(&format!("Parse error: {e}")))?;
    let evaled = expr.eval();
    match evaled.eval_f64() {
        Ok(v) => Ok(format!("{v}")),
        Err(_) => Ok(format!("{evaled}")),
    }
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
/// Returns a JSON 2D array of LaTeX strings representing the 4×4 FK matrix.
#[wasm_bindgen]
pub fn compute_fk(dh_json: &str) -> Result<String, JsValue> {
    let config: DhConfig = serde_json::from_str(dh_json)
        .map_err(|e| JsValue::from_str(&format!("JSON error: {e}")))?;

    let ctx = Context::new();
    let dh_params: Vec<(Ex, Ex, Ex, Ex)> = config
        .joints
        .iter()
        .map(|j| {
            let theta = ctx.symbol(&j.theta);
            let d = ctx.rational((j.d * 1000.0) as i64, 1000);
            let a = ctx.rational((j.a * 1000.0) as i64, 1000);
            let alpha = ctx.rational((j.alpha * 1000.0) as i64, 1000);
            (theta, d, a, alpha)
        })
        .collect();

    let dh_refs: Vec<(&Ex, &Ex, &Ex, &Ex)> = dh_params
        .iter()
        .map(|(t, d, a, al)| (t, d, a, al))
        .collect();

    let fk = symplex::robotics::fk_chain(&dh_refs);

    // Build result as matrix of LaTeX strings
    let mut rows = Vec::new();
    for i in 0..fk.nrows() {
        let mut row = Vec::new();
        for j in 0..fk.ncols() {
            let entry = fk.get(i, j).simplify();
            row.push(entry.to_latex());
        }
        rows.push(row);
    }

    Ok(serde_json::to_string(&rows).unwrap_or_default())
}

/// Compute the Jacobian from DH parameters.
///
/// Takes the same JSON format as `compute_fk`. Returns a JSON 2D array of
/// LaTeX strings representing the 3×N Jacobian (position rows only).
#[wasm_bindgen]
pub fn compute_jacobian(dh_json: &str) -> Result<String, JsValue> {
    let config: DhConfig = serde_json::from_str(dh_json)
        .map_err(|e| JsValue::from_str(&format!("JSON error: {e}")))?;

    let ctx = Context::new();
    let dh_params: Vec<(Ex, Ex, Ex, Ex)> = config
        .joints
        .iter()
        .map(|j| {
            let theta = ctx.symbol(&j.theta);
            let d = ctx.rational((j.d * 1000.0) as i64, 1000);
            let a = ctx.rational((j.a * 1000.0) as i64, 1000);
            let alpha = ctx.rational((j.alpha * 1000.0) as i64, 1000);
            (theta, d, a, alpha)
        })
        .collect();

    let dh_refs: Vec<(&Ex, &Ex, &Ex, &Ex)> = dh_params
        .iter()
        .map(|(t, d, a, al)| (t, d, a, al))
        .collect();

    let (x, y, z) = symplex::robotics::fk_position(&dh_refs);
    let theta_vars: Vec<Ex> = config.joints.iter().map(|j| ctx.symbol(&j.theta)).collect();
    let theta_refs: Vec<&Ex> = theta_vars.iter().collect();

    let jac = symplex::matrix::jacobian(&[&x, &y, &z], &theta_refs);

    let mut rows = Vec::new();
    for i in 0..jac.nrows() {
        let mut row = Vec::new();
        for j in 0..jac.ncols() {
            let entry = jac.get(i, j).simplify();
            row.push(entry.to_latex());
        }
        rows.push(row);
    }

    Ok(serde_json::to_string(&rows).unwrap_or_default())
}

/// Generate optimized Rust code for the Jacobian from DH parameters.
///
/// Takes the same JSON format as `compute_fk`. Returns a string containing
/// a Rust function that computes the Jacobian numerically.
#[wasm_bindgen]
pub fn generate_jacobian_code(dh_json: &str) -> Result<String, JsValue> {
    let config: DhConfig = serde_json::from_str(dh_json)
        .map_err(|e| JsValue::from_str(&format!("JSON error: {e}")))?;

    let ctx = Context::new();
    let dh_params: Vec<(Ex, Ex, Ex, Ex)> = config
        .joints
        .iter()
        .map(|j| {
            let theta = ctx.symbol(&j.theta);
            let d = ctx.rational((j.d * 1000.0) as i64, 1000);
            let a = ctx.rational((j.a * 1000.0) as i64, 1000);
            let alpha = ctx.rational((j.alpha * 1000.0) as i64, 1000);
            (theta, d, a, alpha)
        })
        .collect();

    let dh_refs: Vec<(&Ex, &Ex, &Ex, &Ex)> = dh_params
        .iter()
        .map(|(t, d, a, al)| (t, d, a, al))
        .collect();

    let (x, y, z) = symplex::robotics::fk_position(&dh_refs);
    let theta_vars: Vec<Ex> = config.joints.iter().map(|j| ctx.symbol(&j.theta)).collect();
    let theta_refs: Vec<&Ex> = theta_vars.iter().collect();
    let param_names: Vec<&str> = config.joints.iter().map(|j| j.theta.as_str()).collect();

    let jac = symplex::matrix::jacobian(&[&x, &y, &z], &theta_refs);

    jac.to_rust_fn("jacobian", &param_names)
        .map_err(|e| JsValue::from_str(&format!("Codegen error: {e}")))
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
