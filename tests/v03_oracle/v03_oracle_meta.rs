//! Consistency checks for the 0.3 oracle fixture file
//! (`tests/fixtures/v03_cross_validation.json`): every
//! `(category, subcategory)` present has a consumer `#[test]` (listed
//! here), `fixture_count` matches, ids are sequential, keys are unique within
//! `category:subcategory`, and the file stays small.

use super::oracle_common;

use std::collections::{BTreeMap, BTreeSet};

use oracle_common::*;

const JSON: &str = include_str!("../fixtures/v03_cross_validation.json");

fn v03() -> FixtureFile {
    serde_json::from_str(JSON).expect("v03 fixture JSON must parse")
}

/// Every subcategory that some `tests/v03_oracle_*.rs` test consumes.
/// Adding fixtures to the generator without a consumer fails this test.
const CONSUMED: &[(&str, &str)] = &[
    // v03_oracle_poly.rs
    ("poly", "as_dict"),
    ("poly", "degree"),
    ("poly", "LC"),
    ("poly", "all_coeffs"),
    ("poly", "eval"),
    ("poly", "nroots"),
    ("poly", "degree_symbolic"),
    ("poly", "coeff_symbolic"),
    ("ratsimp", "cancel"),
    ("optimize", "polyfit_exact"),
    ("optimize", "brent_root"),
    // v03_oracle_linprog.rs
    ("linprog", "lpmax"),
    ("linprog", "lpmin"),
    ("linprog", "infeasible"),
    ("linprog", "unbounded"),
    ("linprog", "feasible_nonneg"),
    // v03_oracle_normalforms.rs
    ("normalforms", "hnf"),
    ("normalforms", "hnf_row"),
    ("normalforms", "snf"),
    ("normalforms", "nullspace_rank"),
    ("normalforms", "lattice_det"),
    ("normalforms", "igcd_ilcm"),
];

#[test]
fn every_fixture_subcategory_has_a_consumer() {
    let file = v03();
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
fn fixture_count_ids_and_keys_are_consistent() {
    let file = v03();
    assert_eq!(
        file.fixture_count,
        file.fixtures.len(),
        "fixture_count is stale"
    );
    assert!(
        file.generated_by.starts_with("generate_v03_fixtures.py"),
        "unexpected generated_by: {}",
        file.generated_by
    );
    for (i, f) in file.fixtures.iter().enumerate() {
        assert_eq!(f.id, i + 1, "ids must be 1..=n in file order");
        assert!(
            !f.key.is_empty(),
            "fixture {} in {}:{} has an empty key",
            f.id,
            f.category,
            f.subcategory
        );
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
    // Target range from the campaign brief; every subcategory has ≥ 6 cases.
    assert!(
        (350..=600).contains(&file.fixtures.len()),
        "expected 350–600 fixtures, got {}",
        file.fixtures.len()
    );
    let mut per_sub: BTreeMap<(String, String), usize> = BTreeMap::new();
    for f in &file.fixtures {
        *per_sub
            .entry((f.category.clone(), f.subcategory.clone()))
            .or_default() += 1;
    }
    for (k, n) in &per_sub {
        println!("{}:{} {n}", k.0, k.1);
        assert!(*n >= 6, "{}:{} has only {n} fixtures", k.0, k.1);
    }
}

#[test]
fn fixture_file_is_reasonably_sized_and_gap_free() {
    let size = JSON.len();
    println!("v03 fixture size: {size} bytes");
    assert!(
        size < 600 * 1024,
        "v03 fixtures are {size} bytes (> 600 KB)"
    );
    // The generator keeps oracle failures rather than dropping them; report
    // them so a regression in the oracle is visible.
    let file = v03();
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
            g.oracle_missing().unwrap_or_default()
        );
    }
    assert!(
        gaps.len() * 20 < file.fixtures.len(),
        "{} of {} fixtures have no oracle value",
        gaps.len(),
        file.fixtures.len()
    );
}
