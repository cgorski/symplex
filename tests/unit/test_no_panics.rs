//! Ratchet over `src/`: library code (everything before the first
//! `#[cfg(test)]` line of each file) must not gain a panicking construct
//! beyond its allowlisted count, and an allowlisted file must not lose one
//! without the allowlist being tightened.  See CONTRIBUTING.md, "No Panics
//! Rule".

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Files allowed to contain panicking constructs, with the exact count.
/// Paths are relative to `src/`.
const ALLOWLIST: &[(&str, usize)] = &[
    // Cross-context guard (`checked_id`): a documented logic error, like indexing a Vec out of bounds.
    ("api/expr.rs", 1),
    // `impl Sum`/`Product for Ex` on an empty iterator: no context to build 0/1 in (documented logic error).
    ("api/expr_ops.rs", 1),
    // `u32::try_from` on the node/number index: making `Arena::intern`/`intern_num` fallible touches ~800 call sites; tracked as a follow-up.
    ("base/arena.rs", 2),
    // Same index-space exhaustion for `SymbolTable::intern` (callers through `Arena::symbol`); follow-up with the arena.
    ("base/symbol.rs", 1),
    // `const_assert_dim!`: a `panic!` inside a `const` block only fires during const evaluation (a compile error, never at runtime).
    ("units/assert_macros.rs", 1),
];

/// Does `line` (already trimmed) contain a panicking construct?
fn is_panic_site(line: &str) -> bool {
    if line.starts_with("//") || line.contains("debug_assert") {
        return false;
    }
    if line.contains(".unwrap()") {
        return true;
    }
    // The parser's own `fn expect(&mut self, &Token) -> Result<…>` is not `Option::expect`.
    if line.contains(".expect(") && !line.contains(".expect(&Token") {
        return true;
    }
    ["panic!(", "unreachable!(", "todo!(", "unimplemented!("]
        .iter()
        .any(|m| line.starts_with(m))
}

/// Number of panic sites in the library region of one source file.
fn count_file(path: &Path) -> usize {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .take_while(|l| !l.trim_start().starts_with("#[cfg(test)]"))
        .filter(|l| is_panic_site(l.trim()))
        .count()
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
fn library_code_has_no_unallowlisted_panics() {
    let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    files.sort();

    let allowed: BTreeMap<&str, usize> = ALLOWLIST.iter().copied().collect();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for path in &files {
        let rel = path
            .strip_prefix(src)
            .expect("file is under src/")
            .to_string_lossy()
            .replace('\\', "/");
        counts.insert(rel, count_file(path));
    }

    let mut violations = Vec::new();
    for (rel, &count) in &counts {
        let allowance = allowed.get(rel.as_str()).copied().unwrap_or(0);
        if count > allowance {
            violations.push(format!(
                "{rel}: {count} panic site(s), allowed {allowance} — fix them (CONTRIBUTING.md, \"No Panics Rule\")"
            ));
        } else if count < allowance {
            violations.push(format!(
                "{rel}: {count} panic site(s), allowed {allowance} — tighten the allowlist"
            ));
        }
    }
    for (rel, &allowance) in &allowed {
        if !counts.contains_key(*rel) {
            violations.push(format!(
                "{rel}: allowlisted ({allowance}) but not found under src/ — tighten the allowlist"
            ));
        }
    }

    if !violations.is_empty() {
        let mut report = String::from("panic-site ratchet failed:\n");
        for v in &violations {
            report.push_str("  ");
            report.push_str(v);
            report.push('\n');
        }
        report.push_str("\nper-file counts (non-zero only):\n");
        for (rel, &count) in &counts {
            if count > 0 {
                report.push_str(&format!("  {rel}: {count}\n"));
            }
        }
        panic!("{report}");
    }
}
