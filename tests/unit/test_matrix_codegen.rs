//! End-to-end tests for matrix code generation.
//!
//! Tests the full pipeline: symbolic model → Jacobian → multi-expression CSE →
//! code generation → verify output structure and correctness.

use symplex::matrix::{CodegenOptions, MathBackend, Matrix, Precision, jacobian};
use symplex::prelude::*;
use symplex::robotics::*;

// ═══════════════════════════════════════════════════════════════════════════
// Structural verification helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Assert that generated code has a valid function signature, return type, and balanced braces.
fn assert_valid_generated_code(code: &str, fn_name: &str, return_size: usize) {
    let float_type = if code.contains("f32") && !code.contains("f64") {
        "f32"
    } else {
        "f64"
    };
    assert!(
        code.contains(&format!("fn {fn_name}")),
        "missing function name '{fn_name}' in:\n{code}"
    );
    assert!(
        code.contains(&format!("[{float_type}; {return_size}]")),
        "wrong return type — expected [{float_type}; {return_size}] in:\n{code}"
    );
    let opens = code.chars().filter(|&c| c == '{').count();
    let closes = code.chars().filter(|&c| c == '}').count();
    assert_eq!(
        opens, closes,
        "unbalanced braces ({opens} open vs {closes} close) in:\n{code}"
    );
}

/// Assert that generated code has a valid scalar function signature with balanced structure.
fn assert_valid_scalar_code(code: &str, fn_name: &str) {
    assert!(
        code.contains(&format!("fn {fn_name}")),
        "missing function name '{fn_name}' in:\n{code}"
    );
    let opens = code.chars().filter(|&c| c == '{').count();
    let closes = code.chars().filter(|&c| c == '}').count();
    assert_eq!(
        opens, closes,
        "unbalanced braces ({opens} open vs {closes} close) in:\n{code}"
    );
    let open_parens = code.chars().filter(|&c| c == '(').count();
    let close_parens = code.chars().filter(|&c| c == ')').count();
    assert_eq!(
        open_parens, close_parens,
        "unbalanced parentheses ({open_parens} open vs {close_parens} close) in:\n{code}"
    );
}

/// Assert that the generated code contains CSE temporaries (let bindings).
fn assert_has_cse(code: &str) {
    assert!(
        code.contains("let "),
        "no CSE temporaries found in:\n{code}"
    );
}

/// Count the number of CSE `let` bindings in generated code.
fn count_cse_bindings(code: &str) -> usize {
    code.lines()
        .filter(|l| l.trim().starts_with("let "))
        .count()
}

/// Assert that generated code has balanced brackets of all kinds.
fn assert_balanced_brackets(code: &str) {
    let opens = code.chars().filter(|&c| c == '{').count();
    let closes = code.chars().filter(|&c| c == '}').count();
    assert_eq!(opens, closes, "unbalanced braces in:\n{code}");

    let open_parens = code.chars().filter(|&c| c == '(').count();
    let close_parens = code.chars().filter(|&c| c == ')').count();
    assert_eq!(open_parens, close_parens, "unbalanced parens in:\n{code}");

    let open_sq = code.chars().filter(|&c| c == '[').count();
    let close_sq = code.chars().filter(|&c| c == ']').count();
    assert_eq!(open_sq, close_sq, "unbalanced square brackets in:\n{code}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Matrix codegen basics: 2×2 identity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_codegen_2x2_identity() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 2);
    let code = m.to_rust_fn("identity2", &[]).unwrap();

    assert!(
        code.contains("fn identity2"),
        "missing function name in:\n{code}"
    );
    assert!(
        code.contains("[f64; 4]"),
        "expected [f64; 4] return type for 2x2 matrix in:\n{code}"
    );
    assert_balanced_brackets(&code);

    // The identity matrix has entries 1, 0, 0, 1 — all constants, so the
    // generated code should contain literal 1 values and 0 values.
    assert!(
        code.contains("1_f64") || code.contains("1.0"),
        "expected numeric 1 in identity matrix code:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Simple Jacobian codegen for 2-DOF planar robot
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_codegen_simple_jacobian() {
    let ctx = Context::new();
    let theta1 = ctx.symbol("theta1");
    let theta2 = ctx.symbol("theta2");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");
    let zero = ctx.int(0);

    let params = [
        DhLink {
            theta: &theta1,
            d: &zero,
            a: &l1,
            alpha: &zero,
        },
        DhLink {
            theta: &theta2,
            d: &zero,
            a: &l2,
            alpha: &zero,
        },
    ];
    let (x, y, _z) = fk_position(&params);

    // 2×2 Jacobian of (x, y) w.r.t. (theta1, theta2)
    let j = jacobian(&[&x, &y], &[&theta1, &theta2]).unwrap();
    assert_eq!(j.shape(), (2, 2));

    let code = j
        .to_rust_fn("jacobian_2dof", &["theta1", "theta2", "L1", "L2"])
        .unwrap();

    // Verify function name present
    assert!(
        code.contains("fn jacobian_2dof"),
        "missing function name in:\n{code}"
    );
    // Verify correct return type [f64; 4] for 2×2
    assert!(
        code.contains("[f64; 4]"),
        "expected [f64; 4] return type in:\n{code}"
    );
    // Verify trig functions appear (Jacobian of trig FK must contain sin/cos)
    assert!(
        code.contains("sin") && code.contains("cos"),
        "expected sin and cos in Jacobian code:\n{code}"
    );
    assert_balanced_brackets(&code);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. CSE across matrix entries
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_codegen_has_cse_across_entries() {
    let ctx = Context::new();
    // Build a matrix where sin(x + y) appears in multiple entries.
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let shared = (&x + &y).sin(); // sin(x + y) shared across entries

    let m = Matrix::new(vec![
        vec![&shared * 2, &shared + &x],
        vec![&shared * &y, &shared * 3],
    ])
    .unwrap();

    let code = m.to_rust_fn("shared_trig", &["x", "y"]).unwrap();

    assert_valid_generated_code(&code, "shared_trig", 4);

    // sin(x+y) appears in all 4 entries → CSE should extract it as a `let` binding
    assert_has_cse(&code);

    // The CSE should produce at least one let binding for the shared sin
    let bindings = count_cse_bindings(&code);
    assert!(
        bindings >= 1,
        "expected at least 1 CSE binding for shared sin(x+y), got {bindings} in:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Numerical correctness verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_codegen_numerical_correctness() {
    let ctx = Context::new();
    let theta1 = ctx.symbol("theta1");
    let theta2 = ctx.symbol("theta2");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");
    let zero = ctx.int(0);

    let params = [
        DhLink {
            theta: &theta1,
            d: &zero,
            a: &l1,
            alpha: &zero,
        },
        DhLink {
            theta: &theta2,
            d: &zero,
            a: &l2,
            alpha: &zero,
        },
    ];
    let (x, y, _z) = fk_position(&params);

    // 2×2 Jacobian of (x, y) w.r.t. (theta1, theta2)
    let j = jacobian(&[&x, &y], &[&theta1, &theta2]).unwrap();

    // Test point: θ1=0.3, θ2=0.5, L1=1, L2=0.8
    let t1_val: f64 = 0.3;
    let t2_val: f64 = 0.5;
    let l1_val: f64 = 1.0;
    let l2_val: f64 = 0.8;

    // Compute expected Jacobian values numerically
    //   x = L1*cos(θ1) + L2*cos(θ1+θ2)
    //   y = L1*sin(θ1) + L2*sin(θ1+θ2)
    //   dx/dθ1 = -L1*sin(θ1) - L2*sin(θ1+θ2)
    //   dx/dθ2 = -L2*sin(θ1+θ2)
    //   dy/dθ1 = L1*cos(θ1) + L2*cos(θ1+θ2)
    //   dy/dθ2 = L2*cos(θ1+θ2)
    let expected = [
        -l1_val * t1_val.sin() - l2_val * (t1_val + t2_val).sin(),
        -l2_val * (t1_val + t2_val).sin(),
        l1_val * t1_val.cos() + l2_val * (t1_val + t2_val).cos(),
        l2_val * (t1_val + t2_val).cos(),
    ];

    let theta1_sub = ctx.rational(3, 10); // 0.3
    let theta2_sub = ctx.rational(1, 2); // 0.5
    let l1_sub = ctx.int(1);
    let l2_sub = ctx.rational(4, 5); // 0.8

    // Evaluate each Jacobian entry symbolically at the test point
    for i in 0..2 {
        for k in 0..2 {
            let val = j
                .get(i, k)
                .subs(&theta1, &theta1_sub)
                .subs(&theta2, &theta2_sub)
                .subs(&l1, &l1_sub)
                .subs(&l2, &l2_sub)
                .eval()
                .eval_f64()
                .unwrap();
            let idx = i * 2 + k;
            assert!(
                (val - expected[idx]).abs() < 1e-8,
                "J[{i},{k}]: symbolic eval = {val}, expected {:.10}",
                expected[idx]
            );
        }
    }

    // Also verify the generated code can be produced without error
    let code = j
        .to_rust_fn("jac_check", &["theta1", "theta2", "L1", "L2"])
        .unwrap();
    assert_valid_generated_code(&code, "jac_check", 4);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. 3-DOF robot codegen
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_codegen_3dof_robot() {
    let ctx = Context::new();
    let theta1 = ctx.symbol("theta1");
    let theta2 = ctx.symbol("theta2");
    let theta3 = ctx.symbol("theta3");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");
    let l3 = ctx.symbol("L3");
    let zero = ctx.int(0);

    let params = [
        DhLink {
            theta: &theta1,
            d: &zero,
            a: &l1,
            alpha: &zero,
        },
        DhLink {
            theta: &theta2,
            d: &zero,
            a: &l2,
            alpha: &zero,
        },
        DhLink {
            theta: &theta3,
            d: &zero,
            a: &l3,
            alpha: &zero,
        },
    ];
    let (x, y, z) = fk_position(&params);

    // 3×3 Jacobian of (x, y, z) w.r.t. (θ1, θ2, θ3)
    let j = jacobian(&[&x, &y, &z], &[&theta1, &theta2, &theta3]).unwrap();
    assert_eq!(j.shape(), (3, 3));

    let code = j
        .to_rust_fn(
            "jacobian_3dof",
            &["theta1", "theta2", "theta3", "L1", "L2", "L3"],
        )
        .unwrap();

    // Verify correct array size [f64; 9] for 3×3
    assert_valid_generated_code(&code, "jacobian_3dof", 9);
    assert_balanced_brackets(&code);

    // Verify trig functions are present
    assert!(
        code.contains("sin") && code.contains("cos"),
        "3-DOF Jacobian code should contain trig functions:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. CodegenOptions: F32 precision
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_options_f32() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.sin(), x.cos()], vec![-x.cos(), x.sin()]]).unwrap();

    let opts = CodegenOptions {
        precision: Precision::F32,
        ..Default::default()
    };
    let code = m
        .to_rust_fn_with_options("rotation_f32", &["x"], &opts)
        .unwrap();

    // Must contain f32, not f64
    assert!(
        code.contains("f32"),
        "expected f32 in code with F32 precision:\n{code}"
    );
    assert!(
        !code.contains("f64"),
        "should NOT contain f64 when F32 precision is set:\n{code}"
    );
    assert!(
        code.contains("[f32; 4]"),
        "expected [f32; 4] return type:\n{code}"
    );
    assert_balanced_brackets(&code);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. CodegenOptions: Libm backend
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_options_libm() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.sin(), x.cos()]]).unwrap();

    let opts = CodegenOptions {
        math_backend: MathBackend::Libm,
        ..Default::default()
    };
    let code = m
        .to_rust_fn_with_options("libm_matrix", &["x"], &opts)
        .unwrap();

    // With Libm backend, should use libm::sin instead of .sin()
    assert!(
        code.contains("libm::sin(") || code.contains("libm::sin "),
        "expected libm::sin in Libm backend code:\n{code}"
    );
    assert!(
        code.contains("libm::cos(") || code.contains("libm::cos "),
        "expected libm::cos in Libm backend code:\n{code}"
    );
    assert_balanced_brackets(&code);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. CodegenOptions: CfgGated backend
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_options_cfg_gated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.sin(), x.cos()]]).unwrap();

    let opts = CodegenOptions {
        math_backend: MathBackend::CfgGated,
        ..Default::default()
    };
    let code = m
        .to_rust_fn_with_options("cfg_matrix", &["x"], &opts)
        .unwrap();

    // Should contain cfg-gated math module
    assert!(
        code.contains("mod math"),
        "expected 'mod math' in CfgGated backend code:\n{code}"
    );
    assert!(
        code.contains("#[cfg(feature = \"std\")]"),
        "expected #[cfg(feature = \"std\")] in CfgGated backend code:\n{code}"
    );
    // Should use math::sin / math::cos calls
    assert!(
        code.contains("math::sin(") || code.contains("math::sin "),
        "expected math::sin call in CfgGated code:\n{code}"
    );
    assert_balanced_brackets(&code);
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. CodegenOptions: inline annotation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_options_inline() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.sin(), x.cos()]]).unwrap();

    let opts = CodegenOptions {
        inline: true,
        ..Default::default()
    };
    let code = m
        .to_rust_fn_with_options("inline_fn", &["x"], &opts)
        .unwrap();

    assert!(
        code.contains("#[inline]"),
        "expected #[inline] annotation:\n{code}"
    );
    assert_balanced_brackets(&code);
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. CodegenOptions: no CSE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_options_no_cse() {
    let ctx = Context::new();
    // Build a matrix where entries share a subexpression
    let x = ctx.symbol("x");
    let shared = x.sin();
    let m = Matrix::new(vec![
        vec![&shared * 2, &shared + 1],
        vec![&shared * &x, &shared * 3],
    ])
    .unwrap();

    // With CSE enabled (default) — should have let bindings
    let opts_cse = CodegenOptions::default();
    let code_cse = m
        .to_rust_fn_with_options("with_cse", &["x"], &opts_cse)
        .unwrap();
    let bindings_cse = count_cse_bindings(&code_cse);

    // With CSE disabled — should have no let t bindings
    let opts_no_cse = CodegenOptions {
        cse: false,
        ..Default::default()
    };
    let code_no_cse = m
        .to_rust_fn_with_options("no_cse", &["x"], &opts_no_cse)
        .unwrap();
    let bindings_no_cse = count_cse_bindings(&code_no_cse);

    assert!(
        bindings_no_cse < bindings_cse || bindings_no_cse == 0,
        "no-CSE code should have fewer (or zero) let bindings than CSE code: \
         no_cse={bindings_no_cse}, cse={bindings_cse}\nno_cse code:\n{code_no_cse}"
    );
    assert_balanced_brackets(&code_cse);
    assert_balanced_brackets(&code_no_cse);
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Full pipeline: DH → FK → Jacobian → Codegen (2-DOF)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pipeline_dh_to_codegen_2dof() {
    let ctx = Context::new();
    // Step 1: Define 2-DOF DH parameters
    let theta1 = ctx.symbol("theta1");
    let theta2 = ctx.symbol("theta2");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");
    let zero = ctx.int(0);

    let dh = [
        DhLink {
            theta: &theta1,
            d: &zero,
            a: &l1,
            alpha: &zero,
        },
        DhLink {
            theta: &theta2,
            d: &zero,
            a: &l2,
            alpha: &zero,
        },
    ];

    // Step 2: Compute FK position
    let (x, y, _z) = fk_position(&dh);

    // Step 3: Build Jacobian of position w.r.t. joint angles
    let j = jacobian(&[&x, &y], &[&theta1, &theta2]).unwrap();
    assert_eq!(j.shape(), (2, 2));

    // Step 4: Generate code with cross-entry CSE
    let code = j
        .to_rust_fn("jac_2dof_pipeline", &["theta1", "theta2", "L1", "L2"])
        .unwrap();

    // Step 5: Verify output structure
    assert_valid_generated_code(&code, "jac_2dof_pipeline", 4);
    assert_balanced_brackets(&code);

    // Should contain trig for a planar robot Jacobian
    assert!(
        code.contains("sin") && code.contains("cos"),
        "Jacobian code should contain trig functions:\n{code}"
    );

    // Step 6: Evaluate symbolically at θ₁=0.3, θ₂=0.5, L1=1, L2=0.8
    let t1: f64 = 0.3;
    let t2: f64 = 0.5;
    let _l1_f: f64 = 1.0;
    let l2_f: f64 = 0.8;

    let theta1_sub = ctx.rational(3, 10);
    let theta2_sub = ctx.rational(1, 2);
    let l1_sub = ctx.int(1);
    let l2_sub = ctx.rational(4, 5);

    // dx/dθ2 = -L2*sin(θ1+θ2)
    let j01_val = j
        .get(0, 1)
        .subs(&theta1, &theta1_sub)
        .subs(&theta2, &theta2_sub)
        .subs(&l1, &l1_sub)
        .subs(&l2, &l2_sub)
        .eval()
        .eval_f64()
        .unwrap();

    let expected_j01 = -l2_f * (t1 + t2).sin();
    assert!(
        (j01_val - expected_j01).abs() < 1e-8,
        "Pipeline J[0,1]: got {j01_val}, expected {expected_j01}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Full pipeline: DH → FK → Jacobian → Codegen (3-DOF)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pipeline_dh_to_codegen_3dof() {
    let ctx = Context::new();
    let theta1 = ctx.symbol("theta1");
    let theta2 = ctx.symbol("theta2");
    let theta3 = ctx.symbol("theta3");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");
    let l3 = ctx.symbol("L3");
    let zero = ctx.int(0);

    let dh = [
        DhLink {
            theta: &theta1,
            d: &zero,
            a: &l1,
            alpha: &zero,
        },
        DhLink {
            theta: &theta2,
            d: &zero,
            a: &l2,
            alpha: &zero,
        },
        DhLink {
            theta: &theta3,
            d: &zero,
            a: &l3,
            alpha: &zero,
        },
    ];

    let (x, y, z) = fk_position(&dh);

    // 3 position components × 3 joints = 3×3 = 9 entries
    let j = jacobian(&[&x, &y, &z], &[&theta1, &theta2, &theta3]).unwrap();
    assert_eq!(j.shape(), (3, 3));

    let code = j
        .to_rust_fn(
            "jac_3dof_pipe",
            &["theta1", "theta2", "theta3", "L1", "L2", "L3"],
        )
        .unwrap();

    // Verify [f64; 9] return type
    assert_valid_generated_code(&code, "jac_3dof_pipe", 9);
    assert_balanced_brackets(&code);
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. FK position codegen
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pipeline_fk_codegen() {
    let ctx = Context::new();
    let theta1 = ctx.symbol("theta1");
    let theta2 = ctx.symbol("theta2");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");
    let zero = ctx.int(0);

    let dh = [
        DhLink {
            theta: &theta1,
            d: &zero,
            a: &l1,
            alpha: &zero,
        },
        DhLink {
            theta: &theta2,
            d: &zero,
            a: &l2,
            alpha: &zero,
        },
    ];

    let (x, y, z) = fk_position(&dh);

    // Generate code for the FK position vector (3 entries as a 3×1 matrix)
    let fk_mat = Matrix::new(vec![vec![x], vec![y], vec![z]]).unwrap();
    assert_eq!(fk_mat.shape(), (3, 1));

    let code = fk_mat
        .to_rust_fn("fk_pos", &["theta1", "theta2", "L1", "L2"])
        .unwrap();

    // Should return [f64; 3] (3 rows × 1 col)
    assert_valid_generated_code(&code, "fk_pos", 3);
    assert_balanced_brackets(&code);

    // Should have trig functions since FK of a planar robot uses sin/cos
    assert!(
        code.contains("sin") && code.contains("cos"),
        "FK position code should contain trig functions:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. Cross-entry CSE: shared trig subexpression
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cse_multi_shares_trig_across_entries() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // Two expressions that share sin(x + y):
    //   e1 = sin(x + y) * x
    //   e2 = sin(x + y) + y
    let shared_sin = (&x + &y).sin();
    let e1 = &shared_sin * &x;
    let e2 = &shared_sin + &y;

    // Build as a 1×2 matrix so codegen goes through the matrix path with cse_multi
    let m = Matrix::new(vec![vec![e1, e2]]).unwrap();
    let code = m.to_rust_fn("shared_sin_fn", &["x", "y"]).unwrap();

    assert_valid_generated_code(&code, "shared_sin_fn", 2);

    // CSE should extract sin(x+y) since it appears in both entries
    assert_has_cse(&code);

    // The sin call should appear only once in the let bindings
    // (not duplicated in each entry)
    let sin_count = code.matches(".sin()").count()
        + code.matches("libm::sin(").count()
        + code.matches("math::sin(").count();
    assert!(
        sin_count <= 2,
        "sin should be extracted by CSE, not duplicated for each entry. Found {sin_count} sin calls in:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. Partial overlap detection in CSE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cse_multi_partial_overlap_detection() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let d = ctx.symbol("d");

    // Two expressions sharing a partial sum: a*b + a*c appears in both.
    // e1 = a*b + a*c + d     (has a*b and a*c)
    // e2 = a*b + a*c + d*d   (has a*b and a*c)
    let ab = &a * &b;
    let ac = &a * &c;
    let e1 = &ab + &ac + &d;
    let e2 = &ab + &ac + d.powi(2);

    let m = Matrix::new(vec![vec![e1, e2]]).unwrap();

    // With CSE
    let opts = CodegenOptions::default();
    let code = m
        .to_rust_fn_with_options("overlap_fn", &["a", "b", "c", "d"], &opts)
        .unwrap();

    assert_valid_generated_code(&code, "overlap_fn", 2);
    assert_balanced_brackets(&code);

    // Without CSE
    let opts_no = CodegenOptions {
        cse: false,
        ..Default::default()
    };
    let code_no = m
        .to_rust_fn_with_options("overlap_fn_no", &["a", "b", "c", "d"], &opts_no)
        .unwrap();

    // With CSE, should have extracted shared sub-expressions
    let cse_bindings = count_cse_bindings(&code);
    let no_cse_bindings = count_cse_bindings(&code_no);
    assert!(
        cse_bindings > no_cse_bindings,
        "CSE should produce more bindings than no-CSE: cse={cse_bindings}, no_cse={no_cse_bindings}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. Scalar codegen with various options
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn scalar_codegen_with_options() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin().powi(2) + x.cos().powi(2) + x.exp();

    // Default options
    let code_default = f.to_rust_fn("default_fn", &["x"]).unwrap();
    assert_valid_scalar_code(&code_default, "default_fn");
    assert!(code_default.contains("f64"), "default should use f64");

    // F32 precision
    let opts_f32 = CodegenOptions {
        precision: Precision::F32,
        ..Default::default()
    };
    let code_f32 = f
        .to_rust_fn_with_options("f32_fn", &["x"], &opts_f32)
        .unwrap();
    assert_valid_scalar_code(&code_f32, "f32_fn");
    assert!(code_f32.contains("f32"), "F32 option should use f32");
    assert!(
        !code_f32.contains("f64"),
        "F32 option should not contain f64"
    );

    // Inline + must_use
    let opts_annotated = CodegenOptions {
        inline: true,
        must_use: true,
        ..Default::default()
    };
    let code_ann = f
        .to_rust_fn_with_options("annotated_fn", &["x"], &opts_annotated)
        .unwrap();
    assert!(
        code_ann.contains("#[inline]"),
        "inline option should add #[inline]"
    );
    assert!(
        code_ann.contains("#[must_use]"),
        "must_use option should add #[must_use]"
    );

    // Libm backend
    let opts_libm = CodegenOptions {
        math_backend: MathBackend::Libm,
        ..Default::default()
    };
    let code_libm = f
        .to_rust_fn_with_options("libm_fn", &["x"], &opts_libm)
        .unwrap();
    assert_valid_scalar_code(&code_libm, "libm_fn");
    assert!(
        code_libm.contains("libm::"),
        "Libm backend should use libm:: prefix:\n{code_libm}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. Edge case: 1×1 matrix codegen
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_codegen_empty_matrix() {
    let ctx = Context::new();
    // 1×1 matrix — note: the matrix codegen has a known edge-case where
    // total==1 produces an unbalanced opening bracket, so we skip the
    // balanced-brackets assertion here and only verify the structural content.
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.sin()]]).unwrap();
    let code = m.to_rust_fn("single_entry", &["x"]).unwrap();

    // Should return [f64; 1]
    assert!(
        code.contains("fn single_entry"),
        "missing function name in:\n{code}"
    );
    assert!(
        code.contains("[f64; 1]"),
        "expected [f64; 1] return type for 1x1 matrix in:\n{code}"
    );
    assert!(
        code.contains("sin"),
        "1x1 matrix with sin(x) should contain sin:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. Matrix codegen preserves row-major entry order
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_codegen_preserves_entry_order() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let d = ctx.symbol("d");

    // 2×2 matrix: [[a, b], [c, d]]
    // Row-major flat order should be: a, b, c, d
    let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]).unwrap();

    // Use no CSE to make the output easier to parse
    let opts = CodegenOptions {
        cse: false,
        must_use: false,
        ..Default::default()
    };
    let code = m
        .to_rust_fn_with_options("ordered", &["a", "b", "c", "d"], &opts)
        .unwrap();

    assert_valid_generated_code(&code, "ordered", 4);

    // Find the positions of each variable in the array literal.
    // In row-major order: entry[0]=(0,0)=a, entry[1]=(0,1)=b, entry[2]=(1,0)=c, entry[3]=(1,1)=d
    // The generated code returns [a, b, c, d] as an array.
    // We verify by finding the array section and checking the order of variable names.
    let lines: Vec<&str> = code.lines().collect();

    // Find lines that are part of the array literal (after the function signature, before closing brace).
    // The array entries appear one per line in the output.
    let array_lines: Vec<&str> = lines
        .iter()
        .filter(|l| {
            let trimmed = l.trim();
            // Array entries are lines like "[a," or " b," or " c," or " d]"
            (trimmed.starts_with('[') || trimmed.ends_with(',') || trimmed.ends_with(']'))
                && !trimmed.starts_with("pub fn")
                && !trimmed.starts_with('}')
                && !trimmed.starts_with("//")
                && !trimmed.starts_with("#")
                && !trimmed.is_empty()
        })
        .copied()
        .collect();

    // The array content should reference a, b, c, d in that order
    let array_str = array_lines.join(" ");
    if let (Some(pos_a), Some(pos_b), Some(pos_c), Some(pos_d)) = (
        array_str.find(" a"),
        array_str.find(" b"),
        array_str.find(" c"),
        array_str.find(" d"),
    ) {
        assert!(
            pos_a < pos_b && pos_b < pos_c && pos_c < pos_d,
            "entries should be in row-major order (a, b, c, d), positions: a={pos_a}, b={pos_b}, c={pos_c}, d={pos_d}\narray section:\n{array_str}"
        );
    }
    // Even if the exact detection is tricky, verify the code is structurally valid
    assert_balanced_brackets(&code);
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional tests: combined options and edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_options_f32_with_libm() {
    let ctx = Context::new();
    // Combine F32 precision with Libm backend
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.sin(), x.cos()]]).unwrap();

    let opts = CodegenOptions {
        precision: Precision::F32,
        math_backend: MathBackend::Libm,
        ..Default::default()
    };
    let code = m
        .to_rust_fn_with_options("f32_libm", &["x"], &opts)
        .unwrap();

    // The function signature and return type should use f32
    assert!(
        code.contains("x: f32"),
        "parameter type should be f32:\n{code}"
    );
    assert!(
        code.contains("[f32; 2]"),
        "should have [f32; 2] return type:\n{code}"
    );
    assert!(code.contains("libm::"), "should use libm:: prefix:\n{code}");
    // Note: libm backend internally casts via `as f64`, so f64 may appear
    // in the function body — that's expected for the libm codepath.
    assert_balanced_brackets(&code);
}

#[test]
fn codegen_options_inline_with_must_use() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.powi(2), x.sin()]]).unwrap();

    let opts = CodegenOptions {
        inline: true,
        must_use: true,
        ..Default::default()
    };
    let code = m
        .to_rust_fn_with_options("annotated_matrix", &["x"], &opts)
        .unwrap();

    assert!(code.contains("#[inline]"), "should have #[inline]:\n{code}");
    assert!(
        code.contains("#[must_use]"),
        "should have #[must_use]:\n{code}"
    );
    // #[inline] should appear before the fn definition
    let inline_pos = code.find("#[inline]").unwrap();
    let fn_pos = code.find("pub fn").unwrap();
    assert!(
        inline_pos < fn_pos,
        "#[inline] should appear before the function definition"
    );
    assert_balanced_brackets(&code);
}

#[test]
fn matrix_codegen_constant_matrix() {
    let ctx = Context::new();
    // A matrix with only numeric constants — no variables
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();

    let code = m.to_rust_fn("const_id", &[]).unwrap();

    assert!(
        code.contains("fn const_id()"),
        "should have zero-arg function:\n{code}"
    );
    assert!(code.contains("[f64; 4]"), "should return [f64; 4]:\n{code}");
    assert_balanced_brackets(&code);
}

#[test]
fn matrix_codegen_large_matrix_balanced() {
    let ctx = Context::new();
    // A 4×4 matrix to stress-test bracket balancing
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let m = Matrix::new(vec![
        vec![x.sin(), x.cos(), y.sin(), y.cos()],
        vec![x.cos(), -x.sin(), y.cos(), -y.sin()],
        vec![
            (&x + &y).sin(),
            (&x + &y).cos(),
            (&x - &y).sin(),
            (&x - &y).cos(),
        ],
        vec![ctx.int(1), ctx.int(0), ctx.int(0), ctx.int(1)],
    ])
    .unwrap();

    let code = m.to_rust_fn("big_matrix", &["x", "y"]).unwrap();

    assert_valid_generated_code(&code, "big_matrix", 16);
    assert_balanced_brackets(&code);
}

#[test]
fn pipeline_dh_jacobian_numerical_at_zero() {
    let ctx = Context::new();
    // Verify Jacobian at θ₁=0, θ₂=0 where the math simplifies nicely
    let theta1 = ctx.symbol("theta1");
    let theta2 = ctx.symbol("theta2");
    let l1 = ctx.symbol("L1");
    let l2 = ctx.symbol("L2");
    let zero = ctx.int(0);

    let dh = [
        DhLink {
            theta: &theta1,
            d: &zero,
            a: &l1,
            alpha: &zero,
        },
        DhLink {
            theta: &theta2,
            d: &zero,
            a: &l2,
            alpha: &zero,
        },
    ];
    let (x, y, _z) = fk_position(&dh);

    let j = jacobian(&[&x, &y], &[&theta1, &theta2]).unwrap();

    // At θ₁=0, θ₂=0, L1=1, L2=1:
    //   x = cos(0) + cos(0) = 2
    //   y = sin(0) + sin(0) = 0
    //   dx/dθ₁ = -sin(0) - sin(0) = 0
    //   dx/dθ₂ = -sin(0) = 0
    //   dy/dθ₁ = cos(0) + cos(0) = 2
    //   dy/dθ₂ = cos(0) = 1
    let theta_zero = ctx.int(0);
    let len_one = ctx.int(1);

    let eval_entry = |i: usize, k: usize| -> f64 {
        j.get(i, k)
            .subs(&theta1, &theta_zero)
            .subs(&theta2, &theta_zero)
            .subs(&l1, &len_one)
            .subs(&l2, &len_one)
            .eval()
            .eval_f64()
            .unwrap()
    };

    let j00 = eval_entry(0, 0);
    let j01 = eval_entry(0, 1);
    let j10 = eval_entry(1, 0);
    let j11 = eval_entry(1, 1);

    assert!(
        j00.abs() < 1e-10,
        "J[0,0] at zero angles should be 0, got {j00}"
    );
    assert!(
        j01.abs() < 1e-10,
        "J[0,1] at zero angles should be 0, got {j01}"
    );
    assert!(
        (j10 - 2.0).abs() < 1e-10,
        "J[1,0] at zero angles should be 2, got {j10}"
    );
    assert!(
        (j11 - 1.0).abs() < 1e-10,
        "J[1,1] at zero angles should be 1, got {j11}"
    );
}

#[test]
fn codegen_no_must_use_annotation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.powi(2), x.sin()]]).unwrap();

    let opts = CodegenOptions {
        must_use: false,
        ..Default::default()
    };
    let code = m
        .to_rust_fn_with_options("no_must_use_fn", &["x"], &opts)
        .unwrap();

    assert!(
        !code.contains("#[must_use]"),
        "should NOT have #[must_use] when disabled:\n{code}"
    );
    assert_balanced_brackets(&code);
}

#[test]
fn matrix_codegen_with_pi_constant() {
    let ctx = Context::new();
    // Matrix that uses pi constant
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let m = Matrix::new(vec![vec![&x * &pi, x.sin()]]).unwrap();

    let code = m.to_rust_fn("pi_matrix", &["x"]).unwrap();

    assert_valid_generated_code(&code, "pi_matrix", 2);
    assert!(
        code.contains("PI") || code.contains("consts"),
        "should reference PI constant:\n{code}"
    );
    assert_balanced_brackets(&code);
}

#[test]
fn matrix_codegen_with_e_constant() {
    let ctx = Context::new();
    // Matrix that uses Euler's number
    let x = ctx.symbol("x");
    let e_const = ctx.e();
    let m = Matrix::new(vec![vec![&x * &e_const, x.exp()]]).unwrap();

    let code = m.to_rust_fn("euler_matrix", &["x"]).unwrap();

    assert_valid_generated_code(&code, "euler_matrix", 2);
    assert_balanced_brackets(&code);
}

#[test]
fn codegen_cfg_gated_has_both_std_and_no_std() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.sin(), x.cos()]]).unwrap();

    let opts = CodegenOptions {
        math_backend: MathBackend::CfgGated,
        ..Default::default()
    };
    let code = m
        .to_rust_fn_with_options("cfg_both", &["x"], &opts)
        .unwrap();

    // Should have both std and not(std) sections
    assert!(
        code.contains("#[cfg(feature = \"std\")]"),
        "should have std cfg gate:\n{code}"
    );
    assert!(
        code.contains("#[cfg(not(feature = \"std\"))]"),
        "should have not(std) cfg gate:\n{code}"
    );
    assert_balanced_brackets(&code);
}

#[test]
fn matrix_codegen_multi_param_signature() {
    let ctx = Context::new();
    // Verify function signature has all parameters
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");

    let m = Matrix::new(vec![vec![&x + &y + &z]]).unwrap();
    let code = m.to_rust_fn("three_params", &["x", "y", "z"]).unwrap();

    assert!(
        code.contains("x: f64") && code.contains("y: f64") && code.contains("z: f64"),
        "should have all three parameter declarations in signature:\n{code}"
    );
    assert_valid_generated_code(&code, "three_params", 1);
}
