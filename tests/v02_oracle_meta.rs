//! Consistency checks for the 0.2 oracle fixture file itself: every
//! `(category, subcategory)` present in `v02_cross_validation.json` must have
//! a consumer `#[test]` (listed here), ids are sequential, keys are unique,
//! and the file stays small enough to live in the repository.

mod v02_oracle_common;

use std::collections::BTreeSet;
use v02_oracle_common::*;

/// Every subcategory that some `tests/v02_oracle_*.rs` test consumes.
/// Adding fixtures to the generator without a consumer fails this test.
const CONSUMED: &[(&str, &str)] = &[
    // v02_oracle_calculus.rs
    ("definite_integral", "proper"),
    ("definite_integral", "improper"),
    ("definite_integral", "infinite"),
    ("definite_integral", "symmetric"),
    ("definite_integral", "parametric"),
    ("definite_integral", "numeric"),
    ("summation", "finite_symbolic"),
    ("summation", "infinite"),
    ("summation", "divergent"),
    ("summation", "convergence"),
    ("product", "finite_symbolic"),
    ("product", "infinite"),
    ("limit", "left"),
    ("limit", "right"),
    ("series_at_infinity", "laurent"),
    ("residue", "finite"),
    ("residue", "infinity"),
    // v02_oracle_transforms.rs
    ("laplace", "forward"),
    ("fourier", "ordinary"),
    ("mellin", "forward"),
    // v02_oracle_solve.rs
    ("solve", "general_periodic"),
    ("linsolve", "unique"),
    ("linsolve", "symbolic"),
    ("linsolve", "parametric"),
    ("linsolve", "inconsistent"),
    ("nonlinear_system", "polynomial"),
    ("ode_ivp", "linear"),
    ("rsolve", "linear"),
    // v02_oracle_poly.rs
    ("factor", "multivariate"),
    ("factor", "univariate_list"),
    ("resultant", "integer"),
    ("resultant", "symbolic"),
    ("discriminant", "integer"),
    ("discriminant", "symbolic"),
    ("sqf_list", "univariate"),
    ("nroots", "complex"),
    ("poly_gcd_lcm", "gcd"),
    ("poly_gcd_lcm", "lcm"),
    ("apart", "univariate"),
    ("poly_div", "univariate"),
    ("decompose", "univariate"),
    // v02_oracle_simplify.rs
    ("complex", "re"),
    ("complex", "im"),
    ("complex", "conjugate"),
    ("complex", "arg"),
    ("complex", "abs"),
    ("complex", "expand_complex"),
    ("special_func", "numeric"),
    ("simplify", "sqrtdenest"),
    ("simplify", "nsimplify"),
    ("simplify", "trigsimp"),
    ("simplify", "powsimp"),
    ("simplify", "powdenest"),
    ("simplify", "logcombine"),
    ("simplify", "expand_log"),
    // v02_oracle_sets_logic.rs
    ("sets", "interval_ops"),
    ("inequalities", "reduce"),
    ("logic", "boolean"),
    // v02_oracle_matrix.rs
    ("matrix", "eigenvals"),
    ("matrix", "qr"),
    ("matrix", "cholesky"),
    ("matrix", "jordan_form"),
    ("matrix", "exp"),
    ("matrix", "pinv"),
    ("matrix", "rank_nullspace"),
    ("matrix", "charpoly"),
    // v02_oracle_ntheory.rs
    ("ntheory", "factorint"),
    ("ntheory", "isprime"),
    ("ntheory", "sqrt_mod"),
    ("ntheory", "discrete_log"),
    ("ntheory", "primitive_root"),
    ("ntheory", "jacobi_symbol"),
    ("ntheory", "continued_fraction"),
    ("ntheory", "primepi"),
    ("ntheory", "totient"),
    ("ntheory", "mobius"),
    ("ntheory", "partition"),
    ("ntheory", "bell"),
    ("ntheory", "catalan"),
    ("ntheory", "fibonacci"),
    ("ntheory", "lucas"),
    ("ntheory", "bernoulli"),
    ("ntheory", "divisor_sigma"),
    ("ntheory", "stirling1"),
    ("ntheory", "stirling2"),
    ("diophantine", "linear"),
    ("diophantine", "pell"),
    // v02_oracle_codegen.rs
    ("codegen", "compile"),
];

#[test]
fn every_fixture_subcategory_has_a_consumer() {
    let file = fixture_file();
    let present: BTreeSet<(String, String)> = file
        .fixtures
        .iter()
        .map(|f| (f.category.clone(), f.subcategory.clone()))
        .collect();
    let consumed: BTreeSet<(String, String)> = CONSUMED
        .iter()
        .map(|(c, s)| (c.to_string(), s.to_string()))
        .collect();
    let orphaned: Vec<_> = present.difference(&consumed).collect();
    let stale: Vec<_> = consumed.difference(&present).collect();
    assert!(
        orphaned.is_empty(),
        "fixture subcategories without a consumer test: {orphaned:?}"
    );
    assert!(
        stale.is_empty(),
        "consumer entries without fixtures: {stale:?}"
    );
}

#[test]
fn fixture_ids_sequential_and_keys_unique() {
    let file = fixture_file();
    for (i, f) in file.fixtures.iter().enumerate() {
        assert_eq!(f.id, i + 1, "ids must be 1..=n in file order");
    }
    let mut seen = BTreeSet::new();
    for f in &file.fixtures {
        assert!(
            seen.insert((f.category.as_str(), f.subcategory.as_str(), f.key.as_str())),
            "duplicate key {} in {}:{}",
            f.key,
            f.category,
            f.subcategory
        );
    }
    assert!(
        file.fixtures.len() >= 1000,
        "expected ≥ 1000 fixtures, got {}",
        file.fixtures.len()
    );
}

#[test]
fn fixture_files_are_reasonably_sized() {
    // All oracle fixtures together must stay well under 2 MB.
    let sizes = [
        include_str!("fixtures/v02_cross_validation.json").len(),
        include_str!("fixtures/sympy_cross_validation.json").len(),
        include_str!("fixtures/new_capabilities.json").len(),
        include_str!("fixtures/new_features_cross_validation.json").len(),
    ];
    let total: usize = sizes.iter().sum();
    println!("fixture sizes (bytes): {sizes:?}, total {total}");
    assert!(
        total < 2 * 1024 * 1024,
        "fixtures total {total} bytes (> 2 MB)"
    );
}

#[test]
fn oracle_gaps_are_recorded_not_dropped() {
    // The generator keeps fixtures whose oracle computation failed; the
    // consumers report them as SKIPPED_ORACLE.  Make sure the bookkeeping
    // fields deserialise and that at least the one known SymPy-unevaluated
    // Laplace case is present (it documents an oracle limitation).
    let file = fixture_file();
    let gaps: Vec<_> = file
        .fixtures
        .iter()
        .filter(|f| f.oracle_missing().is_some())
        .collect();
    for g in &gaps {
        println!(
            "oracle gap: {}:{} [{}] — {}",
            g.category,
            g.subcategory,
            g.key,
            g.oracle_missing().unwrap()
        );
    }
    assert!(
        gaps.iter()
            .any(|g| g.category == "laplace" && g.sympy_unevaluated),
        "expected the SymPy-unevaluated Laplace fixture to be kept"
    );
}
