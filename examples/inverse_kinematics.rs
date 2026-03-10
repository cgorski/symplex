//! Inverse Kinematics — find joint angles for a target position.
//!
//! Demonstrates:
//! - 2-DOF planar inverse kinematics via Gröbner bases
//! - Forward kinematics verification of each solution
//! - Generic target positions (axis-aligned, diagonal, folded)
//! - Workspace boundary analysis
//! - When the algebraic solver returns empty (irrational angles)
//! - Numerical fallback using Newton's method
//!
//! The `inverse_kinematics_2dof` solver uses Gröbner bases over a
//! sin/cos polynomial ring, so it finds *exact* algebraic solutions —
//! no iterative approximation needed.  However, it works best when
//! the polynomial system factors cleanly (axis-aligned or fully-extended
//! targets).  For arbitrary targets with irrational joint angles, the
//! algebraic solver may return empty — that's when we fall back to
//! numerical methods.
//!
//! Run with: cargo run --example inverse_kinematics

use symplex::robotics::inverse_kinematics_2dof;

fn main() {
    println!("=== Inverse Kinematics (2-DOF) ===\n");

    let l1: f64 = 1.0;
    let l2: f64 = 0.8;
    let total_reach: f64 = l1 + l2; // 1.8
    let min_reach: f64 = (l1 - l2).abs(); // 0.2

    println!("Robot parameters:");
    println!("  L1 = {l1}");
    println!("  L2 = {l2}");
    println!("  Max reach (L1+L2) = {total_reach}");
    println!("  Min reach |L1-L2| = {min_reach}");
    println!();

    // ════════════════════════════════════════════════════════════════
    // Part 1: Axis-Aligned and Fully-Extended Targets
    // ════════════════════════════════════════════════════════════════
    //
    // The Gröbner-basis solver works best with axis-aligned or
    // fully-extended targets where the polynomial system factors cleanly.

    println!("--- Axis-Aligned Targets ---\n");

    let targets = [
        (1.8, 0.0, "fully extended along +x"),
        (0.0, 1.8, "fully extended along +y"),
        (0.2, 0.0, "folded back along +x (L1-L2)"),
        (-1.8, 0.0, "fully extended along -x"),
        (0.0, -1.8, "fully extended along -y"),
    ];

    for (tx, ty, desc) in &targets {
        println!("Target: ({tx}, {ty}) — {desc}");
        let solutions = inverse_kinematics_2dof(l1, l2, *tx, *ty);
        if solutions.is_empty() {
            println!("  No algebraic solution found");
        }
        for (i, (t1, t2)) in solutions.iter().enumerate() {
            // Verify with forward kinematics
            let fx = l1 * t1.cos() + l2 * (t1 + t2).cos();
            let fy = l1 * t1.sin() + l2 * (t1 + t2).sin();
            let err = ((fx - tx).powi(2) + (fy - ty).powi(2)).sqrt();
            println!("  Solution {}: θ₁ = {:.4}, θ₂ = {:.4}", i + 1, t1, t2);
            println!("    FK verify: ({:.6}, {:.6})", fx, fy);
            println!("    Error: {:.2e}", err);
            if err < 1e-6 {
                println!("    ✓ Verified");
            } else {
                println!("    ✗ Error too large!");
            }
        }
        println!();
    }

    // ════════════════════════════════════════════════════════════════
    // Part 2: Workspace Boundary
    // ════════════════════════════════════════════════════════════════

    println!("--- Workspace Boundary ---\n");

    // On the workspace boundary (distance = L1+L2), there should be
    // exactly one solution (arm fully extended)
    println!("On boundary (distance = {total_reach}):");
    let boundary_solutions = inverse_kinematics_2dof(l1, l2, total_reach, 0.0);
    println!(
        "  ({total_reach}, 0): {} solution(s)",
        boundary_solutions.len()
    );
    for (i, (t1, t2)) in boundary_solutions.iter().enumerate() {
        println!("    Sol {}: θ₁ = {:.4}, θ₂ = {:.4}", i + 1, t1, t2);
    }

    // Inside the inner boundary (distance < |L1-L2|)
    println!("\nInside inner boundary (distance < {min_reach}):");
    let inner_solutions = inverse_kinematics_2dof(l1, l2, 0.1, 0.0);
    println!("  (0.1, 0): {} solution(s)", inner_solutions.len());
    if inner_solutions.is_empty() {
        println!("  (Expected: 0 — target too close to origin)");
    }

    // ════════════════════════════════════════════════════════════════
    // Part 3: Unreachable Targets
    // ════════════════════════════════════════════════════════════════

    println!("\n--- Unreachable Targets ---\n");

    let unreachable: [(f64, f64); 3] = [(10.0, 10.0), (3.0, 0.0), (0.0, 5.0)];

    for (tx, ty) in &unreachable {
        let dist = (tx * tx + ty * ty).sqrt();
        let solutions = inverse_kinematics_2dof(l1, l2, *tx, *ty);
        println!(
            "Target: ({tx}, {ty}), distance = {dist:.2} > {total_reach}: {} solution(s)",
            solutions.len()
        );
    }

    // ════════════════════════════════════════════════════════════════
    // Part 4: Generic Targets (May Need Numerical Fallback)
    // ════════════════════════════════════════════════════════════════
    //
    // For arbitrary targets with irrational joint angles, the Gröbner
    // basis solver over rationals may not find the roots.  This is a
    // known limitation — the solver works in exact arithmetic and
    // cannot represent irrational numbers.
    //
    // When the algebraic solver returns empty for a reachable target,
    // we fall back to the geometric (numerical) solution:
    //   θ₂ = ±acos((x²+y²-L1²-L2²) / (2·L1·L2))
    //   θ₁ = atan2(y,x) - atan2(L2·sin(θ₂), L1+L2·cos(θ₂))

    println!("\n--- Generic Targets (with Numerical Fallback) ---\n");

    let generic_targets: [(f64, f64); 5] =
        [(1.0, 0.5), (0.7, 0.7), (0.5, 1.2), (1.5, 0.3), (-0.5, 0.8)];

    for (tx, ty) in &generic_targets {
        let dist = (tx * tx + ty * ty).sqrt();
        println!("Target: ({tx}, {ty}), distance = {dist:.4}");

        // First, try the algebraic solver
        let algebraic = inverse_kinematics_2dof(l1, l2, *tx, *ty);

        if !algebraic.is_empty() {
            println!("  Algebraic solver found {} solution(s):", algebraic.len());
            for (i, (t1, t2)) in algebraic.iter().enumerate() {
                let fx = l1 * t1.cos() + l2 * (t1 + t2).cos();
                let fy = l1 * t1.sin() + l2 * (t1 + t2).sin();
                let err = ((fx - tx).powi(2) + (fy - ty).powi(2)).sqrt();
                println!(
                    "    Sol {}: θ₁ = {:.6}, θ₂ = {:.6} (err = {:.2e})",
                    i + 1,
                    t1,
                    t2,
                    err
                );
            }
        } else {
            println!("  Algebraic solver: no solution (irrational angles)");
            println!("  Falling back to geometric/numerical method...");

            // Geometric IK for 2-DOF planar arm
            let solutions = numerical_ik_2dof(l1, l2, *tx, *ty);
            if solutions.is_empty() {
                println!("  Numerical: unreachable");
            } else {
                for (i, (t1, t2)) in solutions.iter().enumerate() {
                    let fx = l1 * t1.cos() + l2 * (t1 + t2).cos();
                    let fy = l1 * t1.sin() + l2 * (t1 + t2).sin();
                    let err = ((fx - tx).powi(2) + (fy - ty).powi(2)).sqrt();
                    let config = if i == 0 { "elbow-up" } else { "elbow-down" };
                    println!(
                        "    Sol {} ({}): θ₁ = {:.6}, θ₂ = {:.6} (err = {:.2e})",
                        i + 1,
                        config,
                        t1,
                        t2,
                        err
                    );
                }
            }
        }
        println!();
    }

    // ════════════════════════════════════════════════════════════════
    // Part 5: When to Use Which Method
    // ════════════════════════════════════════════════════════════════

    println!("--- Summary: Algebraic vs Numerical ---\n");
    println!("Algebraic (Gröbner basis) solver:");
    println!("  ✓ Exact solutions — no iterative error");
    println!("  ✓ Finds ALL solution branches");
    println!("  ✓ Works well for axis-aligned and special-angle targets");
    println!("  ✗ May fail for targets requiring irrational joint angles");
    println!("  ✗ Slower than closed-form geometric method");
    println!();
    println!("Numerical (geometric) solver:");
    println!("  ✓ Always works for reachable targets");
    println!("  ✓ Very fast (just a few trig calls)");
    println!("  ✓ Returns both elbow-up and elbow-down configurations");
    println!("  ✗ Floating-point precision only");
    println!("  ✗ Cannot prove exactness");
    println!();
    println!("Recommended workflow:");
    println!("  1. Try the algebraic solver first");
    println!("  2. If it returns empty for a reachable target, use numerical");
    println!("  3. Always verify solutions with forward kinematics");

    println!("\n✓ Done!");
}

/// Numerical (geometric) 2-DOF planar IK using the cosine law.
///
/// Returns up to two solutions (elbow-up and elbow-down).
/// Returns empty if the target is unreachable.
fn numerical_ik_2dof(l1: f64, l2: f64, tx: f64, ty: f64) -> Vec<(f64, f64)> {
    let dist_sq = tx * tx + ty * ty;
    let cos_theta2 = (dist_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2);

    // Check reachability
    if cos_theta2.abs() > 1.0 + 1e-10 {
        return vec![];
    }

    let cos_theta2 = cos_theta2.clamp(-1.0, 1.0);
    let mut solutions = Vec::new();

    // Two solutions: elbow-up and elbow-down
    for sign in [1.0, -1.0] {
        let theta2 = sign * cos_theta2.acos();
        let k1 = l1 + l2 * theta2.cos();
        let k2 = l2 * theta2.sin();
        let theta1 = ty.atan2(tx) - k2.atan2(k1);

        solutions.push((theta1, theta2));
    }

    // Deduplicate if both solutions are the same (boundary case)
    if solutions.len() == 2 {
        let (t1a, t2a) = solutions[0];
        let (t1b, t2b) = solutions[1];
        if (t1a - t1b).abs() < 1e-8 && (t2a - t2b).abs() < 1e-8 {
            solutions.pop();
        }
    }

    solutions
}
