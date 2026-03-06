//! Build-time code generation for symplex.
//!
//! This crate is designed to be used as a `[build-dependency]` in firmware
//! projects. It runs the symplex CAS at build time to derive symbolic
//! equations (Jacobians, dynamics, etc.) and generates optimized numerical
//! Rust code for `no_std` embedded targets.
//!
//! # Quick Start
//!
//! ```toml
//! # Cargo.toml
//! [build-dependencies]
//! symplex-build = "0.1"
//! ```
//!
//! ```rust,no_run
//! // build.rs
//! use symplex_build::CodeGen;
//! use symplex::prelude::*;
//! use symplex::matrix::jacobian;
//! use symplex::robotics::*;
//!
//! fn main() {
//!     vars!(theta1, theta2);
//!     let l1 = symplex::rational(3, 10);  // 0.3m
//!     let l2 = symplex::rational(1, 4);   // 0.25m
//!
//!     let (x, y, _z) = fk_position(&[
//!         (&theta1, &symplex::int(0), &l1, &symplex::int(0)),
//!         (&theta2, &symplex::int(0), &l2, &symplex::int(0)),
//!     ]);
//!
//!     let j = jacobian(&[&x, &y], &[&theta1, &theta2]);
//!
//!     CodeGen::new()
//!         .add_matrix_fn("jacobian", &j, &["theta1", "theta2"])
//!         .write_to_out_dir("robot_math.rs")
//!         .unwrap();
//! }
//! ```

use symplex::matrix::{CodegenOptions, MathBackend, Matrix, Precision};
use symplex::prelude::*;

use std::fs;
use std::path::{Path, PathBuf};

// ═══════════════════════════════════════════════════════════════════════════
// CodeGen builder
// ═══════════════════════════════════════════════════════════════════════════

/// Builder for generating Rust source code from symbolic expressions.
///
/// Collects multiple functions (scalar and matrix) and writes them all
/// to a single output file with shared preamble (cfg-gated math module, etc.).
pub struct CodeGen {
    functions: Vec<GeneratedFn>,
    options: CodegenOptions,
    preamble: Vec<String>,
    test_points: Vec<Vec<f64>>,
    generate_tests: bool,
}

enum GeneratedFn {
    Scalar {
        name: String,
        expr: Ex,
        params: Vec<String>,
    },
    Matrix {
        name: String,
        matrix: Matrix,
        params: Vec<String>,
    },
}

impl CodeGen {
    /// Create a new `CodeGen` builder with default options.
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            options: CodegenOptions::default(),
            preamble: Vec::new(),
            test_points: Vec::new(),
            generate_tests: false,
        }
    }

    /// Set custom code generation options.
    pub fn options(mut self, options: CodegenOptions) -> Self {
        self.options = options;
        self
    }

    /// Enable or disable `no_std`-compatible output (uses the `CfgGated` math backend).
    pub fn no_std(mut self, enabled: bool) -> Self {
        if enabled {
            self.options.math_backend = MathBackend::CfgGated;
        } else {
            self.options.math_backend = MathBackend::Std;
        }
        self
    }

    /// Use `f32` precision for generated code.
    pub fn precision_f32(mut self) -> Self {
        self.options.precision = Precision::F32;
        self
    }

    /// Set whether to emit `#[inline]` annotations on generated functions.
    pub fn inline(mut self, enabled: bool) -> Self {
        self.options.inline = enabled;
        self
    }

    /// Add a scalar function to the output.
    ///
    /// The generated function will take the named parameters as `f64` (or `f32`)
    /// arguments and return the scalar result.
    pub fn add_scalar_fn(mut self, name: &str, expr: &Ex, params: &[&str]) -> Self {
        self.functions.push(GeneratedFn::Scalar {
            name: name.to_string(),
            expr: expr.clone(),
            params: params.iter().map(|s| s.to_string()).collect(),
        });
        self
    }

    /// Add a matrix function to the output.
    ///
    /// The generated function will take the named parameters and return
    /// a flat array `[f64; rows*cols]` in row-major order.
    pub fn add_matrix_fn(mut self, name: &str, matrix: &Matrix, params: &[&str]) -> Self {
        self.functions.push(GeneratedFn::Matrix {
            name: name.to_string(),
            matrix: matrix.clone(),
            params: params.iter().map(|s| s.to_string()).collect(),
        });
        self
    }

    /// Enable or disable companion test generation.
    ///
    /// When enabled, a `#[cfg(test)] mod generated_tests { ... }` block is
    /// appended with test functions that evaluate at any configured test points.
    pub fn with_tests(mut self, enabled: bool) -> Self {
        self.generate_tests = enabled;
        self
    }

    /// Add a test evaluation point.
    ///
    /// Each test point is a slice of `f64` values corresponding to the
    /// function parameters in order. During test generation, each function
    /// is called with each test point to verify it produces a finite result.
    pub fn add_test_point(mut self, point: &[f64]) -> Self {
        self.test_points.push(point.to_vec());
        self
    }

    /// Generate the full source file as a `String`.
    ///
    /// 1. If the math backend is `CfgGated`, emits the cfg-gated math module.
    /// 2. For each registered function, calls the appropriate symplex codegen method.
    /// 3. If test generation is enabled, emits a `#[cfg(test)]` module.
    pub fn generate(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut output = String::new();

        // File header
        output.push_str("// Auto-generated by symplex-build. Do not edit.\n\n");

        // Add any custom preamble lines
        for line in &self.preamble {
            output.push_str(line);
            output.push('\n');
        }

        // We delegate cfg-gated module emission to the per-function codegen,
        // but we want it emitted only once at the top if CfgGated is selected.
        // So we generate a dummy options set with Std backend for the individual
        // functions after emitting the shared module once.
        let emit_cfg_module = self.options.math_backend == MathBackend::CfgGated;

        if emit_cfg_module {
            // Emit the cfg-gated math module once at the top
            append_cfg_gated_module(&mut output, self.options.precision);
            output.push('\n');
        }

        // For individual function codegen, use the same options but avoid
        // re-emitting the cfg-gated module for each function.
        let fn_options = if emit_cfg_module {
            // Use Std backend so individual codegen doesn't re-emit the module,
            // but the generated code already references `math::sin()` etc.
            // Actually, we need to keep CfgGated so the expr_to_rust emits
            // `math::sin(x)` style calls. The module itself is already emitted.
            // Unfortunately the upstream codegen always prepends the module.
            // So we just use the original options and strip duplicate modules
            // from the per-function output.
            self.options.clone()
        } else {
            self.options.clone()
        };

        let mut first_fn = true;
        for gfn in &self.functions {
            if !first_fn {
                output.push('\n');
            }
            first_fn = false;

            let code = match gfn {
                GeneratedFn::Scalar {
                    name,
                    expr,
                    params,
                } => {
                    let param_refs: Vec<&str> = params.iter().map(|s| s.as_str()).collect();
                    expr.to_rust_fn_with_options(name, &param_refs, &fn_options)?
                }
                GeneratedFn::Matrix {
                    name,
                    matrix,
                    params,
                } => {
                    let param_refs: Vec<&str> = params.iter().map(|s| s.as_str()).collect();
                    matrix.to_rust_fn_with_options(name, &param_refs, &fn_options)?
                }
            };

            // If we already emitted the cfg-gated module at the top, strip it
            // from the per-function output to avoid duplicates.
            if emit_cfg_module {
                let stripped = strip_cfg_gated_module(&code);
                output.push_str(&stripped);
            } else {
                output.push_str(&code);
            }
            output.push('\n');
        }

        // Generate test module if requested
        if self.generate_tests && !self.test_points.is_empty() {
            output.push('\n');
            output.push_str("#[cfg(test)]\n");
            output.push_str("mod generated_tests {\n");
            output.push_str("    use super::*;\n\n");

            for (fn_idx, gfn) in self.functions.iter().enumerate() {
                let (fn_name, param_count) = match gfn {
                    GeneratedFn::Scalar { name, params, .. } => (name.as_str(), params.len()),
                    GeneratedFn::Matrix { name, params, .. } => (name.as_str(), params.len()),
                };

                for (pt_idx, point) in self.test_points.iter().enumerate() {
                    if point.len() != param_count {
                        continue;
                    }
                    output.push_str(&format!(
                        "    #[test]\n    fn test_{fn_name}_point_{pt_idx}() {{\n"
                    ));

                    let args: Vec<String> = point
                        .iter()
                        .map(|v| {
                            let float_ty = match self.options.precision {
                                Precision::F64 => "f64",
                                Precision::F32 => "f32",
                            };
                            format!("{v}_{float_ty}")
                        })
                        .collect();
                    let args_str = args.join(", ");

                    match &self.functions[fn_idx] {
                        GeneratedFn::Scalar { .. } => {
                            output.push_str(&format!(
                                "        let result = {fn_name}({args_str});\n"
                            ));
                            output.push_str("        assert!(result.is_finite(), \"expected finite result, got {}\", result);\n");
                        }
                        GeneratedFn::Matrix { matrix, .. } => {
                            let total = matrix.nrows() * matrix.ncols();
                            output.push_str(&format!(
                                "        let result = {fn_name}({args_str});\n"
                            ));
                            output.push_str(&format!(
                                "        for i in 0..{total} {{\n"
                            ));
                            output.push_str("            assert!(result[i].is_finite(), \"entry {} is not finite: {}\", i, result[i]);\n");
                            output.push_str("        }\n");
                        }
                    }

                    output.push_str("    }\n\n");
                }
            }

            output.push_str("}\n");
        }

        Ok(output)
    }

    /// Generate code and write it to `$OUT_DIR/<filename>`.
    ///
    /// Also prints `cargo:rerun-if-changed=build.rs` so Cargo knows when to
    /// re-run the build script.
    pub fn write_to_out_dir(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let out_dir = std::env::var("OUT_DIR")
            .map_err(|_| "OUT_DIR not set — this function must be called from a build script")?;
        let path = PathBuf::from(out_dir).join(filename);
        let code = self.generate()?;
        fs::write(&path, code)?;
        println!("cargo:rerun-if-changed=build.rs");
        Ok(())
    }

    /// Generate code and write it to an explicit path.
    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let code = self.generate()?;
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, code)?;
        Ok(())
    }
}

impl Default for CodeGen {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers for cfg-gated module emission
// ═══════════════════════════════════════════════════════════════════════════

/// Emit the cfg-gated math wrapper module into the given string buffer.
fn append_cfg_gated_module(out: &mut String, precision: Precision) {
    let ft = match precision {
        Precision::F64 => "f64",
        Precision::F32 => "f32",
    };

    let funcs = [
        "sin", "cos", "tan", "exp", "ln", "abs", "sqrt", "cbrt", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "floor", "ceil", "signum",
    ];

    // std version
    out.push_str("#[cfg(feature = \"std\")]\n");
    out.push_str("mod math {\n");
    for func in &funcs {
        out.push_str(&format!(
            "    #[inline] pub fn {func}(x: {ft}) -> {ft} {{ x.{func}() }}\n"
        ));
    }
    out.push_str(&format!(
        "    #[inline] pub fn atan2(y: {ft}, x: {ft}) -> {ft} {{ y.atan2(x) }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn powf(base: {ft}, exp: {ft}) -> {ft} {{ base.powf(exp) }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn powi(base: {ft}, exp: i32) -> {ft} {{ base.powi(exp) }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn min(a: {ft}, b: {ft}) -> {ft} {{ a.min(b) }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn max(a: {ft}, b: {ft}) -> {ft} {{ a.max(b) }}\n"
    ));
    out.push_str("}\n\n");

    // no_std (libm) version
    out.push_str("#[cfg(not(feature = \"std\"))]\n");
    out.push_str("mod math {\n");
    let libm_funcs = [
        "sin", "cos", "tan", "exp", "abs", "sqrt", "cbrt", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "floor", "ceil",
    ];
    for func in &libm_funcs {
        out.push_str(&format!(
            "    #[inline] pub fn {func}(x: {ft}) -> {ft} {{ libm::{func}(x as f64) as {ft} }}\n"
        ));
    }
    out.push_str(&format!(
        "    #[inline] pub fn ln(x: {ft}) -> {ft} {{ libm::log(x as f64) as {ft} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn signum(x: {ft}) -> {ft} {{ if x > 0.0 {{ 1.0 }} else if x < 0.0 {{ -1.0 }} else {{ 0.0 }} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn atan2(y: {ft}, x: {ft}) -> {ft} {{ libm::atan2(y as f64, x as f64) as {ft} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn powf(base: {ft}, exp: {ft}) -> {ft} {{ libm::pow(base as f64, exp as f64) as {ft} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn powi(base: {ft}, exp: i32) -> {ft} {{ libm::pow(base as f64, exp as f64) as {ft} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn min(a: {ft}, b: {ft}) -> {ft} {{ libm::fmin(a as f64, b as f64) as {ft} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn max(a: {ft}, b: {ft}) -> {ft} {{ libm::fmax(a as f64, b as f64) as {ft} }}\n"
    ));
    out.push_str("}\n");
}

/// Strip the cfg-gated module block from per-function generated code.
///
/// When the module has already been emitted at the file level, we need to
/// remove duplicates from individual codegen output that also contains it.
fn strip_cfg_gated_module(code: &str) -> String {
    let mut result = String::new();
    let mut lines = code.lines().peekable();

    while let Some(line) = lines.next() {
        if line.starts_with("#[cfg(") && line.contains("feature") {
            // Check if next line is "mod math {"
            if let Some(&next) = lines.peek() {
                if next.starts_with("mod math {") {
                    // Consume the "mod math {" line and skip the whole block
                    lines.next();
                    let mut brace_depth = 1;
                    while brace_depth > 0 {
                        if let Some(inner) = lines.next() {
                            for ch in inner.chars() {
                                if ch == '{' {
                                    brace_depth += 1;
                                } else if ch == '}' {
                                    brace_depth -= 1;
                                }
                            }
                        } else {
                            break;
                        }
                    }
                    // After closing brace, skip any blank line
                    if let Some(&next_after) = lines.peek() {
                        if next_after.trim().is_empty() {
                            lines.next();
                        }
                    }
                    continue;
                }
            }
        }

        result.push_str(line);
        result.push('\n');
    }

    // Remove leading blank lines
    let trimmed = result.trim_start_matches('\n');
    trimmed.to_string()
}

// ═══════════════════════════════════════════════════════════════════════════
// TOML robot config reader
// ═══════════════════════════════════════════════════════════════════════════

/// Robot configuration loaded from a TOML file.
#[derive(serde::Deserialize)]
struct RobotConfig {
    #[allow(dead_code)]
    robot: RobotInfo,
    joints: Vec<JointConfig>,
    generate: GenerateConfig,
}

/// Basic robot metadata.
#[derive(serde::Deserialize)]
struct RobotInfo {
    #[allow(dead_code)]
    name: String,
}

/// Configuration for a single joint using DH parameters.
#[derive(serde::Deserialize)]
struct JointConfig {
    theta: String,
    #[serde(default)]
    d: f64,
    #[serde(default)]
    a: f64,
    #[serde(default)]
    alpha: f64,
}

fn default_functions() -> Vec<String> {
    vec!["fk".to_string(), "jacobian".to_string()]
}

fn default_output() -> String {
    "robot_math.rs".to_string()
}

/// What to generate from the robot definition.
#[derive(serde::Deserialize)]
struct GenerateConfig {
    #[serde(default = "default_functions")]
    functions: Vec<String>,
    #[serde(default = "default_output")]
    #[allow(dead_code)]
    output: String,
}

/// Load a robot configuration from a TOML file and generate code.
///
/// Returns a `CodeGen` builder pre-populated with the functions requested
/// in the TOML config. Call `.write_to_out_dir()` or `.generate()` on the
/// result to produce the final source file.
///
/// # Example TOML
///
/// ```toml
/// [robot]
/// name = "two_link"
///
/// [[joints]]
/// theta = "theta1"
/// a = 0.3
///
/// [[joints]]
/// theta = "theta2"
/// a = 0.25
///
/// [generate]
/// functions = ["fk", "jacobian"]
/// output = "robot_math.rs"
/// ```
pub fn from_toml(path: impl AsRef<Path>) -> Result<CodeGen, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path.as_ref())?;
    let config: RobotConfig = toml::from_str(&content)?;

    // Build DH parameters from config — create symbolic variables for each theta
    let theta_vars: Vec<Ex> = config
        .joints
        .iter()
        .map(|j| symplex::var(&j.theta))
        .collect();

    let dh_params: Vec<(&Ex, &Ex, &Ex, &Ex)> = {
        // We need to keep owned expressions alive for d, a, alpha
        // so we collect them first.
        Vec::new()
    };

    // We need to hold the numeric constants in a vec so borrows are valid.
    let d_vals: Vec<Ex> = config.joints.iter().map(|j| float_to_expr(j.d)).collect();
    let a_vals: Vec<Ex> = config.joints.iter().map(|j| float_to_expr(j.a)).collect();
    let alpha_vals: Vec<Ex> = config
        .joints
        .iter()
        .map(|j| float_to_expr(j.alpha))
        .collect();

    let _ = dh_params; // drop the empty placeholder

    let dh_params: Vec<(&Ex, &Ex, &Ex, &Ex)> = theta_vars
        .iter()
        .enumerate()
        .map(|(i, theta)| {
            (
                theta as &Ex,
                &d_vals[i] as &Ex,
                &a_vals[i] as &Ex,
                &alpha_vals[i] as &Ex,
            )
        })
        .collect();

    let theta_names: Vec<&str> = config.joints.iter().map(|j| j.theta.as_str()).collect();

    let mut codegen = CodeGen::new();

    for func in &config.generate.functions {
        match func.as_str() {
            "fk" => {
                let (x, y, z) = symplex::robotics::fk_position(&dh_params);
                codegen = codegen.add_scalar_fn("fk_x", &x, &theta_names);
                codegen = codegen.add_scalar_fn("fk_y", &y, &theta_names);
                codegen = codegen.add_scalar_fn("fk_z", &z, &theta_names);
            }
            "jacobian" => {
                let (x, y, _z) = symplex::robotics::fk_position(&dh_params);
                let theta_refs: Vec<&Ex> = theta_vars.iter().collect();
                let j = symplex::matrix::jacobian(
                    &[&x, &y],
                    &theta_refs,
                );
                codegen = codegen.add_matrix_fn("jacobian", &j, &theta_names);
            }
            other => {
                return Err(format!("unknown generate function: {other}").into());
            }
        }
    }

    Ok(codegen)
}

/// Convert an `f64` to an exact symplex expression.
///
/// If the value is an integer, use `symplex::int()`. Otherwise approximate
/// with a rational via `symplex::rational()` scaled by 1_000_000.
fn float_to_expr(v: f64) -> Ex {
    if v == 0.0 {
        symplex::int(0)
    } else if v == (v as i64) as f64 {
        symplex::int(v as i64)
    } else {
        // Approximate the float as p/q with q = 1_000_000 for reasonable precision
        let scale = 1_000_000i64;
        let p = (v * scale as f64).round() as i64;
        symplex::rational(p, scale)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// robot_arm convenience builder
// ═══════════════════════════════════════════════════════════════════════════

/// Quick builder for a serial robot arm from DH parameters.
///
/// Each tuple is `(theta_name, d, a, alpha)`.
///
/// # Examples
///
/// ```rust,no_run
/// // build.rs
/// symplex_build::robot_arm(&[
///     ("theta1", 0.0, 0.3, 0.0),
///     ("theta2", 0.0, 0.25, 0.0),
/// ])
/// .generate_all()
/// .write_to_out_dir("arm.rs")
/// .unwrap();
/// ```
pub fn robot_arm(joints: &[(&str, f64, f64, f64)]) -> RobotArmBuilder {
    let owned: Vec<(String, f64, f64, f64)> = joints
        .iter()
        .map(|(name, d, a, alpha)| (name.to_string(), *d, *a, *alpha))
        .collect();
    RobotArmBuilder::new(owned)
}

/// Builder for generating code for a serial robot arm.
///
/// Created by [`robot_arm()`]. Accumulates requested functions and then
/// delegates to [`CodeGen`] for final output.
pub struct RobotArmBuilder {
    joints: Vec<(String, f64, f64, f64)>,
    codegen: CodeGen,
    generated_fk: bool,
    generated_jacobian: bool,
}

impl RobotArmBuilder {
    fn new(joints: Vec<(String, f64, f64, f64)>) -> Self {
        Self {
            joints,
            codegen: CodeGen::new(),
            generated_fk: false,
            generated_jacobian: false,
        }
    }

    /// Build the symbolic DH parameter tuples and theta variable list.
    fn build_dh(&self) -> (Vec<Ex>, Vec<Ex>, Vec<Ex>, Vec<Ex>) {
        let thetas: Vec<Ex> = self
            .joints
            .iter()
            .map(|(name, _, _, _)| symplex::var(name))
            .collect();
        let d_vals: Vec<Ex> = self
            .joints
            .iter()
            .map(|(_, d, _, _)| float_to_expr(*d))
            .collect();
        let a_vals: Vec<Ex> = self
            .joints
            .iter()
            .map(|(_, _, a, _)| float_to_expr(*a))
            .collect();
        let alpha_vals: Vec<Ex> = self
            .joints
            .iter()
            .map(|(_, _, _, alpha)| float_to_expr(*alpha))
            .collect();
        (thetas, d_vals, a_vals, alpha_vals)
    }

    fn theta_names_owned(&self) -> Vec<String> {
        self.joints.iter().map(|(name, _, _, _)| name.clone()).collect()
    }

    /// Generate forward kinematics position functions (`fk_x`, `fk_y`, `fk_z`).
    pub fn generate_fk(mut self, name: &str) -> Self {
        let (thetas, d_vals, a_vals, alpha_vals) = self.build_dh();
        let dh: Vec<(&Ex, &Ex, &Ex, &Ex)> = thetas
            .iter()
            .enumerate()
            .map(|(i, t)| (t as &Ex, &d_vals[i] as &Ex, &a_vals[i] as &Ex, &alpha_vals[i] as &Ex))
            .collect();
        let owned_names = self.theta_names_owned();
        let theta_names: Vec<&str> = owned_names.iter().map(|s| s.as_str()).collect();
        let (x, y, z) = symplex::robotics::fk_position(&dh);

        let name_x = format!("{name}_x");
        let name_y = format!("{name}_y");
        let name_z = format!("{name}_z");

        self.codegen = self.codegen.add_scalar_fn(&name_x, &x, &theta_names);
        self.codegen = self.codegen.add_scalar_fn(&name_y, &y, &theta_names);
        self.codegen = self.codegen.add_scalar_fn(&name_z, &z, &theta_names);
        self.generated_fk = true;
        self
    }

    /// Generate the Jacobian matrix function.
    pub fn generate_jacobian(mut self, name: &str) -> Self {
        let (thetas, d_vals, a_vals, alpha_vals) = self.build_dh();
        let dh: Vec<(&Ex, &Ex, &Ex, &Ex)> = thetas
            .iter()
            .enumerate()
            .map(|(i, t)| (t as &Ex, &d_vals[i] as &Ex, &a_vals[i] as &Ex, &alpha_vals[i] as &Ex))
            .collect();
        let owned_names = self.theta_names_owned();
        let theta_names: Vec<&str> = owned_names.iter().map(|s| s.as_str()).collect();
        let (x, y, _z) = symplex::robotics::fk_position(&dh);
        let theta_refs: Vec<&Ex> = thetas.iter().collect();
        let j = symplex::matrix::jacobian(&[&x, &y], &theta_refs);

        self.codegen = self.codegen.add_matrix_fn(name, &j, &theta_names);
        self.generated_jacobian = true;
        self
    }

    /// Generate all standard functions (FK position + Jacobian).
    pub fn generate_all(self) -> Self {
        let s = if !self.generated_fk {
            self.generate_fk("fk")
        } else {
            self
        };
        if !s.generated_jacobian {
            s.generate_jacobian("jacobian")
        } else {
            s
        }
    }

    /// Enable `no_std`-compatible output.
    pub fn no_std(mut self) -> Self {
        self.codegen = self.codegen.no_std(true);
        self
    }

    /// Write generated code to `$OUT_DIR/<filename>`.
    pub fn write_to_out_dir(self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.codegen.write_to_out_dir(filename)
    }

    /// Write generated code to an explicit path.
    pub fn write_to_path(
        self,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.codegen.write_to_path(path)
    }

    /// Get the inner [`CodeGen`] builder for further customization.
    pub fn into_codegen(self) -> CodeGen {
        self.codegen
    }
}
