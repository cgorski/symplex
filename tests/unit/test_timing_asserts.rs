//! Ratchet over `tests/`: wall-clock measurements outside the
//! `perf_assert` helper (session 15, M4).  See `tests/perf_assert/mod.rs`
//! and CONTRIBUTING.md, "Timeouts and granularity".
//!
//! A bound on the wall-clock time of a call (`assert!(t.elapsed() <
//! Duration::from_secs(2))`) measures the machine as much as the code: CI
//! failed on 2.02 s against 2 s, and full runs under a 56-core load failed
//! `v32_series` at 3.76 s against 3 s.  A budget goes through
//! `perf_assert` — `assert_fast!(budget, expr)`, `CpuTimer`, `cpu_time` —
//! which measures the CPU time of the test's thread and scales the budget
//! by `SYMPLEX_TEST_TIME_SCALE`.
//!
//! Two counts per file, both syntactic (a linter sees no types):
//!
//! | column | construct |
//! |---|---|
//! | `now` | a call of `Instant::now()` (any path ending in `Instant::now`): the start of every wall-clock measurement |
//! | `elapsed` | a method call `.elapsed()`, `.duration_since(…)`, `.checked_duration_since(…)` or `.saturating_duration_since(…)`, except `duration_since(UNIX_EPOCH)` (a timestamp, as in the names of temporary directories) |
//!
//! Every `.rs` file under `tests/` is scanned except `tests/perf_assert/`
//! (the helper).  Macro bodies are scanned as expressions when they parse
//! as a comma-separated list (`assert!`, `format!`), and token by token
//! otherwise (`proptest! { … }`).
//!
//! What may stay: an `Instant` built as a deadline for the library's
//! `Budget` API, and a measurement that is reported, not asserted, of work
//! done on another thread (the thread's CPU clock cannot see it).  Each
//! entry says which.  The counts are exact: a file may not gain a site, and
//! a file that loses one must have its entry tightened (the failure message
//! prints the table to paste).  Files without an entry must have no site.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::punctuated::Punctuated;
use syn::visit::Visit;

/// Per-file counts `[now, elapsed]` (the columns of the module doc).
/// Paths are relative to `tests/`.  At the wave-12 merge (0.57
/// development): 225 `now` and 301 `elapsed` sites in 85 files (97
/// wall-clock bounds in 80 of them, the rest the reports of benchmarks and
/// the interleaved comparisons of `v28_perf`); after the migration to
/// `perf_assert`, 9 and 2 in 5 files.
const ALLOWLIST: &[(&str, [usize; 2])] = &[
    // `Instant` deadlines (past, now, far future) for `Budget` / `SosOpts` /
    // `PolyhedronOpts`: the library's deadlines are wall-clock by contract.
    ("v03/v03_linprog.rs", [2, 0]),
    ("v04/v04_polyhedron.rs", [1, 0]),
    ("v04/v04_sos.rs", [1, 0]),
    ("v20/v20_base.rs", [4, 0]),
    // The statistics-oracle dump (`STATS_ORACLE_OUT`) logs each row's wall
    // time for `run_dump.py`; the row runs on a worker thread, and nothing
    // is asserted.
    ("v36/v36_s15l5_oracle.rs", [1, 2]),
];

const COLUMNS: [&str; 2] = ["now", "elapsed"];

/// The `Duration`-measuring methods of `Instant` and `SystemTime`.
const MEASURING_METHODS: [&str; 4] = [
    "elapsed",
    "duration_since",
    "checked_duration_since",
    "saturating_duration_since",
];

fn parse(path: &Path) -> syn::File {
    let text = fs::read_to_string(path).expect("test file is readable");
    syn::parse_file(&text)
        .unwrap_or_else(|e| panic!("{}: does not parse as Rust: {e}", path.display()))
}

fn is_instant_now(path: &syn::Path) -> bool {
    let names: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    names.len() >= 2 && names[names.len() - 2] == "Instant" && names[names.len() - 1] == "now"
}

/// `duration_since(UNIX_EPOCH)`: a timestamp, not a measurement.
fn is_unix_epoch(args: &Punctuated<syn::Expr, syn::Token![,]>) -> bool {
    match args.first() {
        Some(syn::Expr::Path(p)) => p
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "UNIX_EPOCH"),
        _ => false,
    }
}

/// Counts of the two columns in the token text of a macro body that does
/// not parse as expressions (`TokenStream`'s `Display` separates tokens by
/// spaces: `Instant :: now ()`, `t . elapsed ()`).
fn token_counts(text: &str) -> [usize; 2] {
    let now = text.matches("Instant :: now (").count();
    let mut elapsed = 0;
    for m in MEASURING_METHODS {
        let pattern = format!(". {m} (");
        for (at, _) in text.match_indices(&pattern) {
            let rest = &text[at + pattern.len()..];
            let epoch = rest.trim_start().starts_with("UNIX_EPOCH")
                || rest.trim_start().starts_with("std :: time :: UNIX_EPOCH");
            if !epoch {
                elapsed += 1;
            }
        }
    }
    [now, elapsed]
}

#[derive(Default)]
struct Scanner {
    /// The function being scanned (for the site labels).
    fn_names: Vec<String>,
    sites: [Vec<String>; 2],
}

impl Scanner {
    fn label(&self, what: &dyn ToTokens) -> String {
        let text = what.to_token_stream().to_string();
        let short: String = text.chars().take(90).collect();
        format!(
            "{}: {short}",
            self.fn_names.last().map_or("(item)", String::as_str)
        )
    }
}

impl<'ast> Visit<'ast> for Scanner {
    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        self.fn_names.push(f.sig.ident.to_string());
        syn::visit::visit_item_fn(self, f);
        self.fn_names.pop();
    }

    fn visit_impl_item_fn(&mut self, m: &'ast syn::ImplItemFn) {
        self.fn_names.push(m.sig.ident.to_string());
        syn::visit::visit_impl_item_fn(self, m);
        self.fn_names.pop();
    }

    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*c.func
            && is_instant_now(&p.path)
        {
            let l = self.label(c);
            self.sites[0].push(l);
        }
        syn::visit::visit_expr_call(self, c);
    }

    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        let name = m.method.to_string();
        if MEASURING_METHODS.contains(&name.as_str())
            && !(name == "duration_since" && is_unix_epoch(&m.args))
        {
            let l = self.label(m);
            self.sites[1].push(l);
        }
        syn::visit::visit_expr_method_call(self, m);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        match mac.parse_body_with(Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated) {
            Ok(exprs) => {
                for e in &exprs {
                    self.visit_expr(e);
                }
            }
            Err(_) => {
                let counts = token_counts(&mac.tokens.to_string());
                let name = mac.path.to_token_stream().to_string();
                for (col, n) in counts.into_iter().enumerate() {
                    for _ in 0..n {
                        let l = format!(
                            "{}: in `{name}!`",
                            self.fn_names.last().map_or("(item)", String::as_str)
                        );
                        self.sites[col].push(l);
                    }
                }
            }
        }
    }
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("tests/ is readable") {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn token_scan_counts_wall_clock_reads_in_macro_bodies() {
    assert_eq!(
        token_counts("let t = Instant :: now () ; f () ; assert ! (t . elapsed () < d)"),
        [1, 1]
    );
    assert_eq!(
        token_counts("SystemTime :: now () . duration_since (UNIX_EPOCH) . unwrap ()"),
        [0, 0]
    );
    assert_eq!(
        token_counts("b . saturating_duration_since (a) ; c . checked_duration_since (a)"),
        [0, 2]
    );
}

#[test]
fn wall_clock_measurements_in_tests_do_not_grow() {
    let tests = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests"));
    let helper = tests.join("perf_assert");
    let mut files = Vec::new();
    collect_rs_files(tests, &mut files);
    files.retain(|p| !p.starts_with(&helper));
    files.sort();

    let mut found: BTreeMap<String, [Vec<String>; 2]> = BTreeMap::new();
    for path in &files {
        let rel = path
            .strip_prefix(tests)
            .expect("file is under tests/")
            .to_string_lossy()
            .replace('\\', "/");
        let mut scanner = Scanner::default();
        scanner.visit_file(&parse(path));
        found.insert(rel, scanner.sites);
    }

    let allowed: BTreeMap<&str, [usize; 2]> = ALLOWLIST.iter().copied().collect();
    let mut violations = Vec::new();
    let mut table = Vec::new();
    let mut totals = [0usize; 2];
    for (rel, sites) in &found {
        let have = [sites[0].len(), sites[1].len()];
        totals[0] += have[0];
        totals[1] += have[1];
        if have.iter().any(|&n| n > 0) {
            table.push(format!("    (\"{rel}\", [{}, {}]),", have[0], have[1]));
        }
        let allowance = allowed.get(rel.as_str()).copied().unwrap_or([0; 2]);
        for col in 0..2 {
            let (n, a) = (have[col], allowance[col]);
            if n > a {
                violations.push(format!(
                    "{rel}: {n} `{}` site(s), allowed {a} — bound a test's time with \
                     `perf_assert::assert_fast!` / `CpuTimer::assert_within` (thread CPU time \
                     × SYMPLEX_TEST_TIME_SCALE), compare workloads with `cpu_time`:\n    {}",
                    COLUMNS[col],
                    sites[col].join("\n    ")
                ));
            } else if n < a {
                violations.push(format!(
                    "{rel}: {n} `{}` site(s), allowed {a} — tighten the allowlist",
                    COLUMNS[col]
                ));
            }
        }
    }
    for (rel, allowance) in &allowed {
        if !found.contains_key(*rel) {
            violations.push(format!(
                "{rel}: allowlisted ({allowance:?}) but not found under tests/ — remove the entry"
            ));
        }
    }

    if !violations.is_empty() {
        panic!(
            "timing ratchet: {} violation(s):\n\n{}\n\nCurrent table [now, elapsed] \
             (totals {} / {}, {} files):\n{}\n",
            violations.len(),
            violations.join("\n\n"),
            totals[0],
            totals[1],
            table.len(),
            table.join("\n")
        );
    }
}
