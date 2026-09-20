//! Ratchet over the layer graph of `src/`: a file must not gain an
//! **upward** dependency (a `crate::<layer>::…` path into a layer above
//! its own) beyond its allowlisted count of distinct upward targets, and an
//! allowlisted file must not lose one without the allowlist being
//! tightened.  See CONTRIBUTING.md, "Architecture → Dependency flow".
//!
//! The order is the *honest* one documented there: `base → poly →
//! transforms → simplify → calculus → output → plotting → domains → api →
//! units`, with the handle types `api::{expr, context, expr_ops,
//! expr_view, eq, macros}` treated as a hub every layer may name.  What
//! remains upward is real and listed below with its reason; the ratchet
//! keeps the list from growing silently.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const LAYERS: &[&str] = &[
    "base",
    "poly",
    "transforms",
    "simplify",
    "calculus",
    "output",
    "plotting",
    "domains",
    "api",
    "units",
];

/// Modules every layer may reach: the expression handle and its context.
const HUB: &[&str] = &[
    "api::expr",
    "api::context",
    "api::expr_ops",
    "api::expr_view",
    "api::eq",
    "api::macros",
];

/// Files with upward edges, with the exact number of *distinct* upward
/// target modules each names.  Paths are relative to `src/`.
const ALLOWLIST: &[(&str, usize)] = &[
    // The `Arena` façade: `arena.expand()`, `.integrate()`, … delegate to the algorithm layers.
    ("base/arena.rs", 35),
    // Assumption queries consult polynomial structure (Sturm, polybridge) and `poly_is_nonnegative_on`.
    ("base/assumptions.rs", 4),
    // `canon_pow` factors small integers through `ntheory`.
    ("base/canon.rs", 1),
    // `as_real_imag` evaluates and expands.
    ("base/complex.rs", 2),
    // Function analysis prints candidates for its error messages.
    ("calculus/calculus_util.rs", 1),
    // Numeric quadrature compiles the integrand.
    ("calculus/definite.rs", 1),
    // ODE systems use `Matrix` and `linsolve`.
    ("calculus/ode.rs", 2),
    // Certificates, polytopes and stats work on the public `Poly` view.
    ("domains/certificates.rs", 1),
    ("domains/certificates/outcome.rs", 1),
    ("domains/certificates/polyhedron.rs", 1),
    ("domains/certificates/sos.rs", 1),
    ("domains/polytope.rs", 1),
    ("domains/stats/joint.rs", 1),
    ("domains/stats/rv.rs", 1),
    // Robotics renders SI quantities.
    ("domains/robotics.rs", 1),
    // Algebraic numbers evaluate, substitute and solve.
    ("poly/algebraic.rs", 4),
    // The Ex ↔ polynomial bridge expands and evaluates coefficients.
    ("poly/polybridge.rs", 3),
    // Polynomial systems fall back to the general solver and re-export `expr_solve_ext` items (public API).
    ("poly/polysys.rs", 6),
    // `factor_terms` shares a sign helper with the printers.
    ("simplify/factor_terms.rs", 1),
    // `apart` rationalises coefficients.
    ("transforms/apart.rs", 1),
    // `eval` folds `Sum`/`Product` through the summation engine, `Integral` through Risch, and binomials through combinatorics.
    ("transforms/eval.rs", 3),
    // `evalf` integrates numerically, compiles, and evaluates combinatorial functions.
    ("transforms/evalf.rs", 3),
    // `expand` delegates trig/log expansion to the simplifiers.
    ("transforms/expand.rs", 2),
    // `integrate` calls Risch and the simplifier.
    ("transforms/integrate.rs", 2),
    // `logic` simplifies relational atoms with the numeric engine.
    ("transforms/logic.rs", 1),
    // The pattern matcher is the engine behind `RuleSet` and the simplifier.
    ("transforms/pattern.rs", 2),
    // `rsolve` uses `linsolve` and `Matrix`.
    ("transforms/rsolve.rs", 2),
    // `solve` normalises with `ratsimp`.
    ("transforms/solve.rs", 1),
];

fn layer_of(rel: &str) -> Option<usize> {
    let first = rel.split('/').next()?.trim_end_matches(".rs");
    LAYERS.iter().position(|l| *l == first)
}

/// Distinct upward target modules (`layer::module` or `layer`) named in a file.
fn upward_targets(rel: &str, text: &str) -> BTreeSet<String> {
    let own = layer_of(rel).unwrap_or(usize::MAX);
    let mut out = BTreeSet::new();
    for (i, _) in text.match_indices("crate::") {
        let rest = &text[i + "crate::".len()..];
        let ident = |s: &str| -> String {
            s.chars()
                .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_')
                .collect()
        };
        let layer = ident(rest);
        let Some(target) = LAYERS.iter().position(|l| *l == layer) else {
            continue;
        };
        let after = &rest[layer.len()..];
        let sub = after
            .strip_prefix("::")
            .map(ident)
            .filter(|s| !s.is_empty());
        let path = match &sub {
            Some(s) => format!("{layer}::{s}"),
            None => layer.clone(),
        };
        if HUB.contains(&path.as_str()) {
            continue;
        }
        if target > own {
            out.insert(path);
        }
    }
    out
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("readable directory entry").path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn layer_graph_has_no_unallowlisted_upward_edges() {
    let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    files.sort();
    let allow: BTreeMap<&str, usize> = ALLOWLIST.iter().copied().collect();
    let mut counts: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for path in &files {
        let rel = path
            .strip_prefix(src)
            .expect("under src/")
            .to_string_lossy()
            .replace('\\', "/");
        if rel == "lib.rs" || layer_of(&rel).is_none() {
            continue;
        }
        let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let up = upward_targets(&rel, &text);
        if !up.is_empty() {
            counts.insert(rel, up);
        }
    }
    let mut problems = Vec::new();
    for (rel, up) in &counts {
        let allowed = allow.get(rel.as_str()).copied().unwrap_or(0);
        if up.len() > allowed {
            problems.push(format!(
                "  {rel}: {} upward target(s), allowed {allowed} — {:?}\n    (move the code down a layer, route through the Arena façade, or extend the allowlist with a reason)",
                up.len(),
                up
            ));
        } else if up.len() < allowed {
            problems.push(format!(
                "  {rel}: {} upward target(s), allowlist says {allowed} — tighten the allowlist",
                up.len()
            ));
        }
    }
    for (rel, allowed) in &allow {
        if !counts.contains_key(*rel) {
            problems.push(format!(
                "  {rel}: no upward edges, allowlist says {allowed} — remove the entry"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "layer ratchet failed:\n{}\n\nper-file upward targets:\n{}",
        problems.join("\n"),
        counts
            .iter()
            .map(|(k, v)| format!("  {k}: {v:?}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
