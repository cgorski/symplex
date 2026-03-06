//! Inverse Kinematics — find joint angles for a target position.
//!
//! Uses the `inverse_kinematics_2dof` solver from `symplex::robotics`
//! to compute joint angles for a planar 2-link arm, then verifies each
//! solution with a forward-kinematics check.
//!
//! The solver uses Gröbner bases over a sin/cos polynomial ring, so it
//! finds *exact* algebraic solutions — no iterative approximation needed.
//!
//! Run with: cargo run --example inverse_kinematics

use symplex::robotics::inverse_kinematics_2dof;

fn main() {
    println!("=== Inverse Kinematics (2-DOF) ===\n");

    let l1 = 1.0;
    let l2 = 0.8;

    // Target positions to reach.
    // The Gröbner-basis solver works best with axis-aligned or
    // fully-extended targets where the polynomial system factors cleanly.
    let targets = [
        (1.8, 0.0),  // fully extended along +x
        (0.0, 1.8),  // fully extended along +y
        (0.2, 0.0),  // folded back along +x  (l1 - l2 = 0.2)
    ];

    for (tx, ty) in &targets {
        println!("Target: ({tx}, {ty})");
        let solutions = inverse_kinematics_2dof(l1, l2, *tx, *ty);
        if solutions.is_empty() {
            println!("  No solution (target may be out of reach or solver limit)");
        }
        for (i, (t1, t2)) in solutions.iter().enumerate() {
            // Verify with forward kinematics
            let fx = l1 * t1.cos() + l2 * (t1 + t2).cos();
            let fy = l1 * t1.sin() + l2 * (t1 + t2).sin();
            println!("  Solution {}: θ₁ = {:.4}, θ₂ = {:.4}", i + 1, t1, t2);
            println!("    FK verify: ({:.4}, {:.4}) ≈ ({tx}, {ty})", fx, fy);
        }
        println!();
    }

    // Unreachable target — well beyond the arm's total length (l1 + l2 = 1.8)
    println!("Target: (10.0, 10.0)");
    let solutions = inverse_kinematics_2dof(l1, l2, 10.0, 10.0);
    println!(
        "  Solutions: {} (expected: 0 — out of reach)",
        solutions.len()
    );

    println!("\n✓ Done!");
}
