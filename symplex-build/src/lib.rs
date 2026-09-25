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
//! symplex-build = "0.2"
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
//!     let ctx = Context::new();
//!     symplex::syms!(ctx; theta1, theta2);
//!     let zero = ctx.int(0);
//!     let l1 = ctx.rational(3, 10);  // 0.3m
//!     let l2 = ctx.rational(1, 4);   // 0.25m
//!
//!     let (x, y, _z) = fk_position(&[
//!         DhLink { theta: &theta1, d: &zero, a: &l1, alpha: &zero },
//!         DhLink { theta: &theta2, d: &zero, a: &l2, alpha: &zero },
//!     ]);
//!
//!     let j = jacobian(&[&x, &y], &[&theta1, &theta2]).unwrap();
//!
//!     CodeGen::new()
//!         .add_matrix_fn("jacobian", &j, &["theta1", "theta2"])
//!         .write_to_out_dir("robot_math.rs")
//!         .unwrap();
//! }
//! ```

use symplex::matrix::{CodegenOptions, MathBackend, Matrix, Precision};
use symplex::prelude::*;
use symplex::robotics::DhLink;

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
    ///
    /// Every registered function is emitted with these options.  The
    /// special-function runtime (`mod symplex_rt`, needed by `gamma`,
    /// `lambertw`, Bessel functions, …) is emitted **once** at the top of
    /// the file when [`CodegenOptions::emit_runtime`] is `true` (the
    /// default) and at least one function needs it.  Set `emit_runtime:
    /// false` to leave it out entirely — e.g. when several generated files
    /// share one copy of [`CodegenOptions::runtime_module`]:
    ///
    /// ```rust,no_run
    /// use symplex::matrix::{CodegenOptions, MathBackend};
    /// use symplex::prelude::*;
    /// use symplex_build::CodeGen;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let opts = CodegenOptions {
    ///     math_backend: MathBackend::CfgGated,
    ///     emit_runtime: false,
    ///     ..Default::default()
    /// };
    /// let body = CodeGen::new()
    ///     .options(opts.clone())
    ///     .add_scalar_fn("g", &x.gamma(), &["x"])
    ///     .add_scalar_fn("w", &x.lambertw(), &["x"])
    ///     .generate()
    ///     .unwrap();
    /// std::fs::write("robot_math.rs", body).unwrap();
    /// std::fs::write("symplex_rt.rs", opts.runtime_module()).unwrap();
    /// ```
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
    /// 1. If the math backend is `CfgGated`, emits the cfg-gated math module
    ///    (once; the per-function copies are stripped).
    /// 2. If [`CodegenOptions::emit_runtime`] is set and any registered
    ///    function uses a special function, emits the `mod symplex_rt`
    ///    runtime once, containing exactly the helpers the file needs.
    /// 3. For each registered function, calls the appropriate symplex codegen method.
    /// 4. If test generation is enabled, emits a `#[cfg(test)]` module.
    pub fn generate(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut output = String::new();

        // File header
        output.push_str("// Auto-generated by symplex-build. Do not edit.\n\n");

        // Add any custom preamble lines
        for line in &self.preamble {
            output.push_str(line);
            output.push('\n');
        }

        let emit_cfg_module = self.options.math_backend == MathBackend::CfgGated;

        if emit_cfg_module {
            // Emit the cfg-gated math module once at the top
            append_cfg_gated_module(&mut output, self.options.precision);
            output.push('\n');
        }

        // Per-function codegen never embeds the runtime: it is emitted once
        // for the whole file below, after we know which helpers are used.
        let fn_options = CodegenOptions {
            emit_runtime: false,
            ..self.options.clone()
        };

        let mut functions = String::new();
        let mut first_fn = true;
        for gfn in &self.functions {
            if !first_fn {
                functions.push('\n');
            }
            first_fn = false;

            let code = match gfn {
                GeneratedFn::Scalar { name, expr, params } => {
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
                functions.push_str(&stripped);
            } else {
                functions.push_str(&code);
            }
            functions.push('\n');
        }

        if self.options.emit_runtime
            && let Some(runtime) = self.options.runtime_module_for(&functions)
        {
            output.push_str(&runtime);
            output.push_str("\n\n");
        }
        output.push_str(&functions);

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
                            output.push_str(&format!("        for i in 0..{total} {{\n"));
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
///
/// The module must provide every `math::*` function the symplex Rust backend
/// can emit with [`MathBackend::CfgGated`]: the elementary functions,
/// `atan2`, `powf`/`powi`, `min`/`max`, the numerically-optimised forms
/// `expm1`, `log1p`, `log2`, `exp2`, the fused multiply-add `fma`, and
/// `sin_cos` (used when both `sin(x)` and `cos(x)` appear).  The `std`
/// variant delegates to inherent `f64`/`f32` methods; the `no_std` variant
/// delegates to the `libm` crate.
fn append_cfg_gated_module(out: &mut String, precision: Precision) {
    let ft = match precision {
        Precision::F64 => "f64",
        Precision::F32 => "f32",
    };

    let funcs = [
        "sin", "cos", "tan", "exp", "ln", "abs", "sqrt", "cbrt", "asin", "acos", "atan", "sinh",
        "cosh", "tanh", "asinh", "acosh", "atanh", "floor", "ceil", "signum",
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
    out.push_str(&format!(
        "    #[inline] pub fn expm1(x: {ft}) -> {ft} {{ x.exp_m1() }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn log1p(x: {ft}) -> {ft} {{ x.ln_1p() }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn log2(x: {ft}) -> {ft} {{ x.log2() }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn exp2(x: {ft}) -> {ft} {{ x.exp2() }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn fma(a: {ft}, b: {ft}, c: {ft}) -> {ft} {{ a.mul_add(b, c) }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn sin_cos(x: {ft}) -> ({ft}, {ft}) {{ x.sin_cos() }}\n"
    ));
    out.push_str("}\n\n");

    // no_std (libm) version.  `libm` names differ from the inherent methods
    // for `abs` (`fabs`) and `ln` (`log`); `signum` and `sin_cos` have no
    // direct counterpart and are composed.
    out.push_str("#[cfg(not(feature = \"std\"))]\n");
    out.push_str("mod math {\n");
    let libm_funcs = [
        "sin", "cos", "tan", "exp", "sqrt", "cbrt", "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "asinh", "acosh", "atanh", "floor", "ceil",
    ];
    for func in &libm_funcs {
        out.push_str(&format!(
            "    #[inline] pub fn {func}(x: {ft}) -> {ft} {{ libm::{func}(x as f64) as {ft} }}\n"
        ));
    }
    out.push_str(&format!(
        "    #[inline] pub fn abs(x: {ft}) -> {ft} {{ libm::fabs(x as f64) as {ft} }}\n"
    ));
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
    out.push_str(&format!(
        "    #[inline] pub fn expm1(x: {ft}) -> {ft} {{ libm::expm1(x as f64) as {ft} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn log1p(x: {ft}) -> {ft} {{ libm::log1p(x as f64) as {ft} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn log2(x: {ft}) -> {ft} {{ libm::log2(x as f64) as {ft} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn exp2(x: {ft}) -> {ft} {{ libm::exp2(x as f64) as {ft} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn fma(a: {ft}, b: {ft}, c: {ft}) -> {ft} {{ libm::fma(a as f64, b as f64, c as f64) as {ft} }}\n"
    ));
    out.push_str(&format!(
        "    #[inline] pub fn sin_cos(x: {ft}) -> ({ft}, {ft}) {{ (libm::sin(x as f64) as {ft}, libm::cos(x as f64) as {ft}) }}\n"
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
            if let Some(&next) = lines.peek()
                && next.starts_with("mod math {")
            {
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
                if let Some(&next_after) = lines.peek()
                    && next_after.trim().is_empty()
                {
                    lines.next();
                }
                continue;
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
/// functions = ["fk", "jacobian"]   # also: "fk_matrix" (full 4×4 transform)
/// output = "robot_math.rs"
/// ```
///
/// Numeric DH parameters are converted to exact rationals with
/// [`Context::from_f64_approx`] (`0.3` → `3/10`).
pub fn from_toml(path: impl AsRef<Path>) -> Result<CodeGen, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path.as_ref())?;
    let config: RobotConfig = toml::from_str(&content)?;

    // All symbolic work for this robot lives in a single private context.
    let ctx = Context::new();

    // Build DH parameters from config — create symbolic variables for each
    // theta (an empty name is an error, not a panic).
    let theta_vars: Vec<Ex> = config
        .joints
        .iter()
        .map(|j| joint_symbol(&ctx, &j.theta))
        .collect::<Result<_, _>>()?;

    // Hold the numeric constants in vecs so the borrows below stay valid.
    let d_vals: Vec<Ex> = config
        .joints
        .iter()
        .map(|j| float_to_expr(&ctx, j.d))
        .collect::<Result<_, _>>()?;
    let a_vals: Vec<Ex> = config
        .joints
        .iter()
        .map(|j| float_to_expr(&ctx, j.a))
        .collect::<Result<_, _>>()?;
    let alpha_vals: Vec<Ex> = config
        .joints
        .iter()
        .map(|j| float_to_expr(&ctx, j.alpha))
        .collect::<Result<_, _>>()?;

    let dh_params: Vec<DhLink<'_>> = theta_vars
        .iter()
        .enumerate()
        .map(|(i, theta)| DhLink {
            theta,
            d: &d_vals[i],
            a: &a_vals[i],
            alpha: &alpha_vals[i],
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
                let j = symplex::matrix::jacobian(&[&x, &y], &theta_refs)?;
                codegen = codegen.add_matrix_fn("jacobian", &j, &theta_names);
            }
            "fk_matrix" => {
                let t = symplex::robotics::fk_chain(&dh_params);
                codegen = codegen.add_matrix_fn("fk_matrix", &t, &theta_names);
            }
            other => {
                return Err(format!("unknown generate function: {other}").into());
            }
        }
    }

    Ok(codegen)
}

/// The joint variable named `name`.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`](symplex::errors::SymplexError::InvalidArgument)
/// for an empty name (a configuration error; [`Context::symbol`] would
/// panic on it).
fn joint_symbol(ctx: &Context, name: &str) -> Result<Ex, symplex::errors::SymplexError> {
    ctx.try_symbol(name).map_err(|_| {
        symplex::errors::SymplexError::invalid_argument(
            "symplex-build",
            "a joint's variable name must not be empty",
        )
    })
}

/// Convert an `f64` DH parameter to an exact symplex expression.
///
/// Uses [`Context::from_f64_approx`] with a denominator bound of one
/// million, so the value becomes the reduced rational a human-written robot
/// spec almost always means (`0.3` → `3/10`, `0.25` → `1/4`, `2.0` → `2`)
/// rather than the exact binary expansion of the float.
///
/// # Errors
///
/// [`SymplexError::InvalidArgument`](symplex::errors::SymplexError::InvalidArgument)
/// for a `NaN` or infinite entry: a configuration error (a `NaN`, which
/// TOML can spell `nan`, used to panic here).
fn float_to_expr(ctx: &Context, v: f64) -> Result<Ex, symplex::errors::SymplexError> {
    if !v.is_finite() {
        return Err(symplex::errors::SymplexError::invalid_argument(
            "symplex-build",
            format!("a DH parameter must be a finite number, got {v}"),
        ));
    }
    ctx.from_f64_approx(v, 1_000_000)
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

/// The symbolic DH table of a [`RobotArmBuilder`]: one entry per joint.
#[derive(Default)]
struct DhTable {
    thetas: Vec<Ex>,
    d: Vec<Ex>,
    a: Vec<Ex>,
    alpha: Vec<Ex>,
}

impl DhTable {
    /// The links, borrowing the table.
    fn links(&self) -> Vec<DhLink<'_>> {
        self.thetas
            .iter()
            .zip(&self.d)
            .zip(&self.a)
            .zip(&self.alpha)
            .map(|(((theta, d), a), alpha)| DhLink { theta, d, a, alpha })
            .collect()
    }
}

/// Builder for generating code for a serial robot arm.
///
/// Created by [`robot_arm()`]. Accumulates requested functions and then
/// delegates to [`CodeGen`] for final output.  All symbolic work happens
/// in a private [`Context`] owned by the builder.
///
/// A function that cannot be generated (a Jacobian for an arm without
/// joints; any function of an arm with an empty joint name or a `NaN` or
/// infinite DH parameter) is reported by
/// [`write_to_out_dir`](Self::write_to_out_dir) /
/// [`write_to_path`](Self::write_to_path);
/// [`into_codegen`](Self::into_codegen) returns the functions that could
/// be generated.
pub struct RobotArmBuilder {
    ctx: Context,
    joints: Vec<(String, f64, f64, f64)>,
    codegen: CodeGen,
    generated_fk: bool,
    generated_jacobian: bool,
    error: Option<symplex::errors::SymplexError>,
}

impl RobotArmBuilder {
    fn new(joints: Vec<(String, f64, f64, f64)>) -> Self {
        Self {
            ctx: Context::new(),
            joints,
            codegen: CodeGen::new(),
            generated_fk: false,
            generated_jacobian: false,
            error: None,
        }
    }

    /// Build the symbolic DH parameter tuples and theta variable list.
    ///
    /// # Errors
    ///
    /// An empty joint name or a non-finite DH parameter (see
    /// [`joint_symbol`], [`float_to_expr`]); the `generate_*` methods keep
    /// it for [`write_to_out_dir`](Self::write_to_out_dir) /
    /// [`write_to_path`](Self::write_to_path) to report.  (An empty name
    /// panicked in `Context::symbol`.)
    fn build_dh(&self) -> Result<DhTable, symplex::errors::SymplexError> {
        let ctx = &self.ctx;
        let mut table = DhTable::default();
        for (name, d, a, alpha) in &self.joints {
            table.thetas.push(joint_symbol(ctx, name)?);
            table.d.push(float_to_expr(ctx, *d)?);
            table.a.push(float_to_expr(ctx, *a)?);
            table.alpha.push(float_to_expr(ctx, *alpha)?);
        }
        Ok(table)
    }

    /// [`build_dh`](Self::build_dh), recording its error (the first one
    /// wins) and returning `None` so the caller generates nothing.
    fn dh_or_record(&mut self) -> Option<DhTable> {
        match self.build_dh() {
            Ok(table) => Some(table),
            Err(e) => {
                self.error = self.error.take().or(Some(e));
                None
            }
        }
    }

    fn theta_names_owned(&self) -> Vec<String> {
        self.joints
            .iter()
            .map(|(name, _, _, _)| name.clone())
            .collect()
    }

    /// Generate forward kinematics position functions (`fk_x`, `fk_y`, `fk_z`).
    pub fn generate_fk(mut self, name: &str) -> Self {
        let Some(table) = self.dh_or_record() else {
            self.generated_fk = true;
            return self;
        };
        let dh = table.links();
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

    /// Generate the full 4×4 homogeneous forward-kinematics transform as a
    /// matrix function `name(theta…) -> [f64; 16]` (row-major), via
    /// [`symplex::robotics::fk_chain`].
    ///
    /// The position functions from [`generate_fk`](Self::generate_fk) are
    /// the last column of this matrix; the upper-left 3×3 block is the
    /// end-effector rotation.
    pub fn generate_fk_matrix(mut self, name: &str) -> Self {
        let Some(table) = self.dh_or_record() else {
            return self;
        };
        let dh = table.links();
        let owned_names = self.theta_names_owned();
        let theta_names: Vec<&str> = owned_names.iter().map(|s| s.as_str()).collect();
        let t = symplex::robotics::fk_chain(&dh);
        self.codegen = self.codegen.add_matrix_fn(name, &t, &theta_names);
        self
    }

    /// Generate the Jacobian matrix function.
    pub fn generate_jacobian(mut self, name: &str) -> Self {
        let Some(table) = self.dh_or_record() else {
            self.generated_jacobian = true;
            return self;
        };
        let dh = table.links();
        let owned_names = self.theta_names_owned();
        let theta_names: Vec<&str> = owned_names.iter().map(|s| s.as_str()).collect();
        let (x, y, _z) = symplex::robotics::fk_position(&dh);
        let theta_refs: Vec<&Ex> = table.thetas.iter().collect();
        match symplex::matrix::jacobian(&[&x, &y], &theta_refs) {
            Ok(j) => self.codegen = self.codegen.add_matrix_fn(name, &j, &theta_names),
            Err(e) => self.error = self.error.or(Some(e)),
        }
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
    ///
    /// # Errors
    ///
    /// A function that could not be generated (see [`RobotArmBuilder`]), or
    /// the error of [`CodeGen::write_to_out_dir`].
    pub fn write_to_out_dir(self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(e) = self.error {
            return Err(e.into());
        }
        self.codegen.write_to_out_dir(filename)
    }

    /// Write generated code to an explicit path.
    ///
    /// # Errors
    ///
    /// A function that could not be generated (see [`RobotArmBuilder`]), or
    /// the error of [`CodeGen::write_to_path`].
    pub fn write_to_path(self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(e) = self.error {
            return Err(e.into());
        }
        self.codegen.write_to_path(path)
    }

    /// Get the inner [`CodeGen`] builder for further customization.
    pub fn into_codegen(self) -> CodeGen {
        self.codegen
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codegen_new_default() {
        let cg = CodeGen::new();
        // Should create without panic; default has no functions registered
        assert!(cg.functions.is_empty());
        assert!(!cg.generate_tests);
    }

    #[test]
    fn codegen_default_trait() {
        // CodeGen::default() and CodeGen::new() should behave identically
        let cg = CodeGen::default();
        assert!(cg.functions.is_empty());
    }

    #[test]
    fn codegen_generate_empty() {
        let cg = CodeGen::new();
        let code = cg.generate().unwrap();
        // Empty codegen should produce at least the file header
        assert!(
            code.contains("Auto-generated by symplex-build"),
            "expected header comment in generated output, got: {code}"
        );
        // No functions registered → no function bodies
        assert!(
            !code.contains("fn "),
            "expected no function definitions in empty codegen"
        );
    }

    #[test]
    fn float_to_expr_is_exact_and_reduced() {
        let ctx = Context::new();
        let show = |v: f64| format!("{}", float_to_expr(&ctx, v).unwrap());
        assert_eq!(show(0.0), "0");
        assert_eq!(show(2.0), "2");
        assert_eq!(show(-3.0), "-3");
        assert_eq!(show(0.3), "3/10");
        assert_eq!(show(0.25), "1/4");
        assert_eq!(show(0.1 + 0.2), "3/10");
        assert_eq!(show(1.0 / 3.0), "1/3");
        assert_eq!(show(0.123456), "1929/15625");
        // A NaN or infinite entry is an error (a NaN panicked).
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(float_to_expr(&ctx, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn robot_arm_generates_fk_matrix() {
        let code = robot_arm(&[("q1", 0.0, 0.3, 0.0), ("q2", 0.1, 0.25, 0.0)])
            .generate_fk_matrix("fk_t")
            .into_codegen()
            .generate()
            .unwrap();
        assert!(code.contains("fn fk_t("), "{code}");
        assert!(
            code.contains("q1: f64") && code.contains("q2: f64"),
            "{code}"
        );
        // 4×4 homogeneous transform → flat array of 16.
        assert!(code.contains("[f64; 16]"), "expected 4×4 matrix:\n{code}");
        // Exact rationals survive into the generated constants (no 0.30000000000000004).
        assert!(!code.contains("0.30000000000000004"), "{code}");
    }

    #[test]
    fn fk_matrix_last_column_matches_fk_position() {
        // The generated position functions must agree with the last column
        // of the full transform, so both entry points are consistent.
        let ctx = Context::new();
        let (q1, q2) = (ctx.symbol("q1"), ctx.symbol("q2"));
        let zero = ctx.int(0);
        let (l1, l2) = (
            float_to_expr(&ctx, 0.3).unwrap(),
            float_to_expr(&ctx, 0.25).unwrap(),
        );
        let dh = [
            DhLink {
                theta: &q1,
                d: &zero,
                a: &l1,
                alpha: &zero,
            },
            DhLink {
                theta: &q2,
                d: &zero,
                a: &l2,
                alpha: &zero,
            },
        ];
        let t = symplex::robotics::fk_chain(&dh);
        let (x, y, z) = symplex::robotics::fk_position(&dh);
        assert_eq!(t.get(0, 3).eval(), x);
        assert_eq!(t.get(1, 3).eval(), y);
        assert_eq!(t.get(2, 3).eval(), z);
    }

    #[test]
    fn from_toml_accepts_fk_matrix() {
        let dir = std::env::temp_dir().join(format!("symplex_build_fkm_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("robot.toml");
        fs::write(
            &path,
            r#"
[robot]
name = "one_link"
[[joints]]
theta = "q"
a = 0.5
[generate]
functions = ["fk_matrix"]
"#,
        )
        .unwrap();
        let code = from_toml(&path).unwrap().generate().unwrap();
        assert!(code.contains("fn fk_matrix("), "{code}");
        assert!(code.contains("[f64; 16]"), "{code}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn robot_arm_generates_fk_and_jacobian() {
        let code = robot_arm(&[("theta1", 0.0, 0.3, 0.0), ("theta2", 0.0, 0.25, 0.0)])
            .generate_all()
            .into_codegen()
            .generate()
            .unwrap();

        for name in ["fk_x", "fk_y", "fk_z", "jacobian"] {
            assert!(
                code.contains(&format!("fn {name}(")),
                "expected `{name}` in generated code:\n{code}"
            );
        }
        assert!(code.contains("theta1: f64") && code.contains("theta2: f64"));
        // Planar arm: the Jacobian is 2×2 → flat array of 4.
        assert!(code.contains("[f64; 4]"), "expected 2×2 Jacobian:\n{code}");
    }

    #[test]
    fn robot_arm_no_std_emits_single_math_module() {
        let code = robot_arm(&[("q", 0.0, 1.0, 0.0)])
            .no_std()
            .generate_all()
            .into_codegen()
            .generate()
            .unwrap();
        // The cfg-gated math module must be emitted exactly once (std + libm variants).
        assert_eq!(code.matches("mod math {").count(), 2, "{code}");
    }

    /// Two functions that each need the special-function runtime.
    fn two_special_fns(opts: CodegenOptions) -> CodeGen {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let y = ctx.symbol("y");
        CodeGen::new()
            .options(opts)
            .add_scalar_fn("g", &(x.gamma() + &y), &["x", "y"])
            .add_scalar_fn("e", &(x.erf() * &y), &["x", "y"])
    }

    #[test]
    fn generate_emits_runtime_module_once_for_two_special_functions() {
        let code = two_special_fns(CodegenOptions::default())
            .generate()
            .unwrap();
        assert_eq!(code.matches("mod symplex_rt {").count(), 1, "{code}");
        assert!(
            code.contains("pub fn gamma(") && code.contains("pub fn erf("),
            "{code}"
        );
        assert!(code.contains("fn g(") && code.contains("fn e("), "{code}");
        // The runtime precedes the functions that use it.
        assert!(code.find("mod symplex_rt {").unwrap() < code.find("fn g(").unwrap());
        // Only the helpers the file needs are embedded.
        assert!(!code.contains("pub fn bessel_k("), "{code}");
    }

    #[test]
    fn generate_emits_runtime_module_once_in_no_std_mode() {
        let code = two_special_fns(CodegenOptions::no_std())
            .generate()
            .unwrap();
        assert_eq!(code.matches("mod symplex_rt {").count(), 1, "{code}");
        assert_eq!(code.matches("mod math {").count(), 2, "{code}");
        // Order: mod math, mod symplex_rt, functions.
        let math_pos = code.find("mod math {").unwrap();
        let rt_pos = code.find("mod symplex_rt {").unwrap();
        let fn_pos = code.find("fn g(").unwrap();
        assert!(math_pos < rt_pos && rt_pos < fn_pos, "{code}");
    }

    #[test]
    fn generate_honours_emit_runtime_false() {
        let opts = CodegenOptions {
            emit_runtime: false,
            ..Default::default()
        };
        let code = two_special_fns(opts).generate().unwrap();
        assert_eq!(code.matches("mod symplex_rt {").count(), 0, "{code}");
        assert!(code.contains("symplex_rt::gamma("), "{code}");
    }

    #[test]
    fn generate_omits_runtime_when_unused() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let code = CodeGen::new()
            .add_scalar_fn("f", &(x.sin() + x.powi(2)), &["x"])
            .generate()
            .unwrap();
        assert!(!code.contains("mod symplex_rt"), "{code}");
    }

    /// The two-function file must compile as a library (skipped when
    /// `rustc` is not on the PATH).
    #[test]
    fn generated_file_with_two_special_functions_compiles() {
        let Ok(out) = std::process::Command::new("rustc")
            .arg("--version")
            .output()
        else {
            eprintln!("rustc not available; skipping compile check");
            return;
        };
        if !out.status.success() {
            return;
        }
        let code = two_special_fns(CodegenOptions::default())
            .generate()
            .unwrap();
        let dir = std::env::temp_dir().join(format!("symplex_build_rt_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let src = dir.join("gen.rs");
        fs::write(&src, format!("#![allow(dead_code)]\n{code}")).unwrap();
        let out = std::process::Command::new("rustc")
            .args(["--crate-type", "lib", "--edition", "2024", "-o"])
            .arg(dir.join("gen.rlib"))
            .arg(&src)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        let _ = fs::remove_dir_all(&dir);
        assert!(
            out.status.success(),
            "generated file failed to compile:\n{stderr}\n{code}"
        );
    }

    #[test]
    fn from_toml_round_trip() {
        let dir = std::env::temp_dir().join(format!("symplex_build_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("robot.toml");
        fs::write(
            &path,
            r#"
[robot]
name = "two_link"

[[joints]]
theta = "theta1"
a = 0.3

[[joints]]
theta = "theta2"
a = 0.25

[generate]
functions = ["fk", "jacobian"]
"#,
        )
        .unwrap();

        let code = from_toml(&path).unwrap().generate().unwrap();
        assert!(code.contains("fn fk_x("), "{code}");
        assert!(code.contains("fn jacobian("), "{code}");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_toml_rejects_unknown_function() {
        let dir = std::env::temp_dir().join(format!("symplex_build_bad_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("robot.toml");
        fs::write(
            &path,
            r#"
[robot]
name = "r"
[[joints]]
theta = "q"
[generate]
functions = ["dynamics"]
"#,
        )
        .unwrap();
        let err = from_toml(&path)
            .err()
            .expect("unknown function should error");
        assert!(err.to_string().contains("dynamics"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// An empty joint name panicked in `build_dh` (`Context::symbol("")`)
    /// from every `generate_*`; it is now the builder's error, reported by
    /// `write_to_path`, and nothing is generated for the arm.  A `NaN` DH
    /// parameter (which panicked in `float_to_expr`) likewise.
    #[test]
    fn robot_arm_reports_an_empty_joint_name_instead_of_panicking() {
        let dir = std::env::temp_dir().join(format!("symplex_build_empty_{}", std::process::id()));
        let path = dir.join("arm.rs");
        let err = robot_arm(&[("q1", 0.0, 0.3, 0.0), ("", 0.0, 0.25, 0.0)])
            .generate_all()
            .generate_fk_matrix("fk_t")
            .write_to_path(&path)
            .expect_err("an empty joint name is an error");
        assert!(err.to_string().contains("name must not be empty"), "{err}");
        assert!(!path.exists(), "nothing is written");
        let generated = robot_arm(&[("", 0.0, 0.3, 0.0)])
            .generate_fk("fk")
            .into_codegen();
        assert!(generated.functions.is_empty());
        let err = robot_arm(&[("q1", f64::NAN, 0.3, 0.0)])
            .generate_jacobian("j")
            .write_to_path(&path)
            .expect_err("a NaN DH parameter is an error");
        assert!(err.to_string().contains("finite"), "{err}");
        assert!(!path.exists(), "nothing is written");
    }

    /// `from_toml` with `theta = ""` (or `d = nan`) panicked the same way;
    /// it is now its error.
    #[test]
    fn from_toml_rejects_an_empty_joint_name_and_a_nan_parameter() {
        let dir =
            std::env::temp_dir().join(format!("symplex_build_empty_toml_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("robot.toml");
        for (joint, needle) in [
            ("theta = \"\"", "name must not be empty"),
            ("theta = \"q\"\nd = nan", "finite"),
        ] {
            fs::write(
                &path,
                format!(
                    "[robot]\nname = \"r\"\n[[joints]]\n{joint}\n[generate]\nfunctions = [\"fk\"]\n"
                ),
            )
            .unwrap();
            let err = from_toml(&path).err().expect("a bad joint is an error");
            assert!(err.to_string().contains(needle), "{err}");
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
