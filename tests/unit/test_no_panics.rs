//! Ratchet over `src/`: library code (everything before the file's
//! `#[cfg(test)] mod …` test module) must not gain a panicking construct
//! beyond its allowlisted count, and an allowlisted file must not lose one
//! without the allowlist being tightened.  See CONTRIBUTING.md, "No Panics
//! Rule".
//!
//! A `#[cfg(test)]` attribute on a *single item* (a test-only helper `fn`
//! in the middle of a file) does not end the library region: 0.11.1 found
//! twelve panic sites hidden behind such attributes in three files.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Files allowed to contain panicking constructs, with the exact count.
/// Paths are relative to `src/`.
const ALLOWLIST: &[(&str, usize)] = &[
    // Cross-context guard (`checked_id`): a documented logic error, like indexing a Vec out of bounds.
    ("api/expr.rs", 1),
    // `Context::own_id`: the same cross-context guard, for expressions handed to a `Context` method.
    ("api/expr_ops.rs", 1),
    // `u32::try_from` on the node/number index: making `Arena::intern`/`intern_num` fallible touches ~800 call sites; tracked as a follow-up.
    ("base/arena.rs", 2),
    // Same index-space exhaustion for `SymbolTable::intern` (callers through `Arena::symbol`); follow-up with the arena.
    ("base/symbol.rs", 1),
    // `const_assert_dim!`: a `panic!` inside a `const` block only fires during const evaluation (a compile error, never at runtime).
    ("units/assert_macros.rs", 1),
    // Arithmetic operators on matrices of mismatched shapes (`&a + &b`): an operator cannot return a
    // `Result`, so like `Vec` indexing it panics, documented, with `add`/`sub`/`matmul` returning one.
    ("domains/exact_matrix.rs", 1),
    ("domains/matrix.rs", 1),
    // `Polytope::with_halfspace` with the wrong number of coefficients; `try_with_halfspace` returns a `Result`.
    ("domains/polytope.rs", 1),
];

/// Files allowed to contain runtime `assert!` / `assert_eq!` / `assert_ne!`
/// in library code, with the exact count (0.22).  Every one checks a
/// caller-supplied *shape* or precondition and is documented under
/// `# Panics` on its item; converting them to `Result`s is tracked as a
/// follow-up, and this list may only shrink.  `debug_assert!` is not counted.
const ASSERT_ALLOWLIST: &[(&str, usize)] = &[
    // `Context::symbol`: an empty name (`try_symbol` returns an error).
    ("api/context.rs", 1),
    // `Ex::replace`: the user's closure returned an expression from another context (cross-context logic error).
    ("api/expr_funcs.rs", 1),
    // `impl Sum`/`Product for Ex` on an empty iterator (no context to build 0/1 in; `Context::sum`/`product`
    // take one).
    ("api/expr_ops.rs", 4),
    // Exact matrices: index bounds in `get`, `get_mut` (the bodies of `m[(i, j)]`), `row`, `col`; the checked
    // siblings are `try_get`, `try_get_mut`, `try_row`, `try_col`.  The constructors return `Result` (0.29).
    ("domains/exact_matrix.rs", 4),
    // `Matrix`: index bounds in `get`, `get_mut` (the bodies of `m[(i, j)]`), `row`, `col`; the checked siblings are
    // `try_get`, `try_get_mut`, `try_row`, `try_col`.  The constructors return `Result` (0.29).
    ("domains/matrix.rs", 4),
    // `MultiPoly`: variable-count agreement and exponent overflow in `add`/`sub`/`mul`/`pow` (the bodies of the
    // `+`/`-`/`*` operators, which cannot return an error; `try_add`/`try_sub`/`try_mul`/`try_pow` return `None`),
    // variable index in `var`/`degree_in`/`partial_derivative`/`eval_var`/`substitute`, and `s_polynomial`.
    ("poly/multipoly.rs", 10),
    // `RationalFn`: zero denominator, division by zero, inverse of zero.
    ("poly/ratfn.rs", 3),
];

/// Does `line` (already trimmed) contain a runtime assertion macro?
/// `debug_assert*` and `const_assert_dim!` do not count.
fn is_assert_site(line: &str) -> bool {
    if line.starts_with("//") || line.contains("debug_assert") {
        return false;
    }
    ["assert!(", "assert_eq!(", "assert_ne!("].iter().any(|m| {
        line.match_indices(m).any(|(i, _)| {
            line[..i]
                .chars()
                .next_back()
                .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
        })
    })
}

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
    // Anywhere on the line, not only at its start: `Err(e) => panic!(…)`
    // and `_ => unreachable!()` went uncounted before 0.25.
    ["panic!(", "unreachable!(", "todo!(", "unimplemented!("]
        .iter()
        .any(|m| {
            line.match_indices(m).any(|(i, _)| {
                line[..i]
                    .chars()
                    .next_back()
                    .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
            })
        })
}

/// The library region of a source file: every line before its
/// `#[cfg(test)]` test *module* (`#[cfg(test)]` followed — possibly after
/// further attributes — by a `mod` line).  A `#[cfg(test)]` on a lone `fn`
/// keeps the region open.
fn library_region(text: &str) -> impl Iterator<Item = &str> {
    let lines: Vec<&str> = text.lines().collect();
    let mut end = lines.len();
    for (i, line) in lines.iter().enumerate() {
        if !line.trim_start().starts_with("#[cfg(test)]") {
            continue;
        }
        // Skip further attributes; the item decides.
        let item = lines[i + 1..]
            .iter()
            .map(|l| l.trim_start())
            .find(|l| !l.starts_with("#["));
        if item.is_some_and(|l| l.starts_with("mod ") || l.starts_with("pub(crate) mod ")) {
            end = i;
            break;
        }
    }
    lines.into_iter().take(end)
}

/// Number of lines matching `site` in the library region of one source file.
fn count_file(path: &Path, site: fn(&str) -> bool) -> usize {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    library_region(&text).filter(|l| site(l.trim())).count()
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
    ratchet("panic site", ALLOWLIST, is_panic_site);
}

#[test]
fn library_code_has_no_unallowlisted_asserts() {
    ratchet("assert", ASSERT_ALLOWLIST, is_assert_site);
}

#[test]
fn panic_site_detection() {
    assert!(is_panic_site("panic!(\"msg\");"));
    assert!(is_panic_site("Err(e) => panic!(\"{e}\"),"));
    assert!(is_panic_site("_ => unreachable!(),"));
    assert!(is_panic_site("let v = x.unwrap();"));
    assert!(!is_panic_site("// panic!(\"in a comment\")"));
    assert!(!is_panic_site("/// _ => unreachable!() in a doc comment"));
    assert!(!is_panic_site("my_panic!(x);"));
    assert!(!is_panic_site("debug_assert!(n > 0, \"no panic!(\");"));
}

#[test]
fn assert_site_detection() {
    assert!(is_assert_site("assert!(n > 0, \"msg\");"));
    assert!(is_assert_site("let _ = { assert_eq!(a, b); };"));
    assert!(is_assert_site("assert_ne!("));
    assert!(!is_assert_site("debug_assert!(n > 0);"));
    assert!(!is_assert_site("debug_assert_eq!(a, b);"));
    assert!(!is_assert_site("const_assert_dim!(D, LENGTH);"));
    assert!(!is_assert_site("/// assert!(x) in a doc comment"));
    assert!(!is_assert_site("my_assert!(x);"));
}

fn ratchet(what: &str, allowlist: &[(&str, usize)], site: fn(&str) -> bool) {
    let src = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    files.sort();

    let allowed: BTreeMap<&str, usize> = allowlist.iter().copied().collect();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for path in &files {
        let rel = path
            .strip_prefix(src)
            .expect("file is under src/")
            .to_string_lossy()
            .replace('\\', "/");
        counts.insert(rel, count_file(path, site));
    }

    let mut violations = Vec::new();
    for (rel, &count) in &counts {
        let allowance = allowed.get(rel.as_str()).copied().unwrap_or(0);
        if count > allowance {
            violations.push(format!(
                "{rel}: {count} {what}(s), allowed {allowance} — fix them (CONTRIBUTING.md, \"No Panics Rule\")"
            ));
        } else if count < allowance {
            violations.push(format!(
                "{rel}: {count} {what}(s), allowed {allowance} — tighten the allowlist"
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
        let mut report = format!("{what} ratchet failed:\n");
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
