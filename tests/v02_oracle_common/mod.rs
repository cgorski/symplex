//! Shared plumbing for the symplex 0.2 SymPy oracle
//! (`tests/v02_oracle/v02_oracle_*.rs`; also reused by `tests/v03_oracle.rs`).
//!
//! Fixtures come from `tests/fixtures/v02_cross_validation.json`, generated
//! by `scripts/generate_v02_fixtures.py`.  Each consumer test file handles a
//! group of categories, with **one `#[test]` per `(category, subcategory)`**
//! so that a failure or a hang is attributable to a small set of cases.
//!
//! === HONESTY POLICY ===
//! PASS            symplex agrees with the oracle (numerically, within tolerance)
//! FAIL            symplex produced a result that is WRONG — this fails the test
//! NOT_IMPLEMENTED symplex returned an error / unevaluated form
//! UNSUPPORTED_API symplex has no public API for the operation
//! SKIPPED_ORACLE  SymPy itself timed out / errored / returned an unevaluated
//!                 object while generating the fixture (recorded, not hidden)
//!
//! KNOWN_BUG       a WRONG result that is already recorded in the consumer's
//!                 `KNOWN_BUGS` table (with a `// BUG:` reason and, where
//!                 possible, an `#[ignore]`d reproducer).  Strict xfail: if a
//!                 listed case starts passing the test FAILS so the entry is
//!                 removed.  Known bugs are printed on every run.
//!
//! `run()` asserts `fail == 0`, `unexpected_pass == 0` and `total > 0` for
//! every subcategory and prints a summary row plus every non-PASS case.
//!
//! Every fixture is processed on its own thread with a wall-clock budget
//! (`FIXTURE_TIMEOUT`); a fixture that does not finish counts as FAIL
//! ("HANG") — a non-terminating library call is a bug, and this keeps one
//! bad case from stalling CI.  A panic inside a fixture is also a FAIL.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde_json::Value;
use symplex::prelude::*;

pub const TOLERANCE: f64 = 1e-6;
/// Wall-clock budget per fixture (each `#[test]` must stay well under 10 s).
pub const FIXTURE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(4);

// ── Fixture file ───────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct FixtureFile {
    pub generated_by: String,
    pub fixture_count: usize,
    pub fixtures: Vec<Fixture>,
}

#[derive(serde::Deserialize)]
pub struct Fixture {
    pub id: usize,
    pub category: String,
    pub subcategory: String,
    /// Stable identifier within `category:subcategory` (defaults to empty for
    /// fixture files that key on the subcategory alone).
    #[serde(default)]
    pub key: String,
    #[serde(default)]
    pub sympy_timeout: bool,
    #[serde(default)]
    pub sympy_error: Option<String>,
    #[serde(default)]
    pub sympy_unevaluated: bool,
    #[serde(flatten)]
    pub fields: serde_json::Map<String, Value>,
}

static FILE: OnceLock<FixtureFile> = OnceLock::new();

pub fn fixture_file() -> &'static FixtureFile {
    FILE.get_or_init(|| {
        let json = include_str!("../fixtures/v02_cross_validation.json");
        let file: FixtureFile = serde_json::from_str(json).expect("v02 fixture JSON must parse");
        assert_eq!(
            file.fixture_count,
            file.fixtures.len(),
            "fixture_count is stale"
        );
        file
    })
}

// ── Numbers ────────────────────────────────────────────────────────────

/// A reference value from the oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Num {
    Finite(f64, f64),
    PosInf,
    NegInf,
    ComplexInf,
    NaN,
}

impl Num {
    pub fn from_json(v: &Value) -> Option<Num> {
        let obj = v.as_object()?;
        if let Some(sp) = obj.get("special").and_then(Value::as_str) {
            return Some(match sp {
                "oo" => Num::PosInf,
                "-oo" => Num::NegInf,
                "zoo" => Num::ComplexInf,
                _ => Num::NaN,
            });
        }
        let re = obj.get("re")?.as_f64()?;
        let im = obj.get("im").and_then(Value::as_f64).unwrap_or(0.0);
        Some(Num::Finite(re, im))
    }

    pub fn re(&self) -> Option<f64> {
        match self {
            Num::Finite(re, _) => Some(*re),
            _ => None,
        }
    }
}

impl std::fmt::Display for Num {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Num::Finite(re, im) if *im == 0.0 => write!(f, "{re}"),
            Num::Finite(re, im) => write!(f, "({re} + {im}i)"),
            Num::PosInf => write!(f, "oo"),
            Num::NegInf => write!(f, "-oo"),
            Num::ComplexInf => write!(f, "zoo"),
            Num::NaN => write!(f, "nan"),
        }
    }
}

pub struct EvalPoint {
    pub subs: BTreeMap<String, f64>,
    pub value: Option<Num>,
    /// For multi-valued fixtures (linsolve): variable → value.
    pub values: BTreeMap<String, Num>,
}

impl Fixture {
    pub fn str(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(Value::as_str)
    }
    pub fn field(&self, key: &str) -> Option<&Value> {
        self.fields.get(key)
    }
    pub fn num(&self, key: &str) -> Option<Num> {
        self.fields.get(key).and_then(Num::from_json)
    }
    pub fn bool(&self, key: &str) -> Option<bool> {
        self.fields.get(key).and_then(Value::as_bool)
    }
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.fields.get(key).and_then(Value::as_u64)
    }
    pub fn i64(&self, key: &str) -> Option<i64> {
        self.fields.get(key).and_then(Value::as_i64)
    }
    pub fn str_list(&self, key: &str) -> Vec<String> {
        self.fields
            .get(key)
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
    pub fn eval_points(&self, key: &str) -> Vec<EvalPoint> {
        let Some(arr) = self.fields.get(key).and_then(Value::as_array) else {
            return Vec::new();
        };
        arr.iter()
            .filter_map(|p| {
                let obj = p.as_object()?;
                let subs = obj
                    .get("subs")
                    .and_then(Value::as_object)
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                            .collect()
                    })
                    .unwrap_or_default();
                let value = obj.get("value").and_then(Num::from_json);
                let values = obj
                    .get("values")
                    .and_then(Value::as_object)
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, v)| Num::from_json(v).map(|n| (k.clone(), n)))
                            .collect()
                    })
                    .unwrap_or_default();
                Some(EvalPoint {
                    subs,
                    value,
                    values,
                })
            })
            .collect()
    }
    /// True if the oracle itself could not produce a reference value.
    pub fn oracle_missing(&self) -> Option<String> {
        if self.sympy_timeout {
            return Some("SymPy timed out".into());
        }
        if let Some(e) = &self.sympy_error {
            return Some(format!("SymPy error: {e}"));
        }
        if self.sympy_unevaluated {
            return Some("SymPy returned an unevaluated object".into());
        }
        None
    }
}

// ── Status / statistics ────────────────────────────────────────────────

pub enum Status {
    Pass,
    Fail(String),
    NotImplemented(String),
    UnsupportedApi(String),
    SkippedOracle(String),
}

#[derive(Default)]
pub struct Stats {
    pub pass: usize,
    pub fail: usize,
    pub not_impl: usize,
    pub no_api: usize,
    pub skipped_oracle: usize,
    pub known_bug: usize,
    pub unexpected_pass: usize,
}

impl Stats {
    pub fn total(&self) -> usize {
        self.pass + self.fail + self.not_impl + self.no_api + self.skipped_oracle + self.known_bug
    }
}

/// A recorded library bug: `(category, subcategory, fixture key, reason)`.
pub type KnownBug = (&'static str, &'static str, &'static str, &'static str);

/// Shared, thread-safe fixture processor.
type Processor = std::sync::Arc<dyn Fn(&Context, &Fixture) -> Status + Send + Sync>;

/// Run every fixture of `category:subcategory` through `process`, print a
/// summary row and all non-PASS cases, and assert `fail == 0`.
pub fn run(
    category: &str,
    subcategory: &str,
    process: impl Fn(&Context, &Fixture) -> Status + Send + Sync + 'static,
) {
    run_with_known_bugs(category, subcategory, &[], process);
}

/// Run `process` on a fresh context on its own thread with a wall-clock
/// budget.  Hangs and panics are reported as `Status::Fail`.
fn process_bounded(process: &Processor, fx: &'static Fixture) -> Status {
    let (tx, rx) = std::sync::mpsc::channel();
    let p = process.clone();
    std::thread::spawn(move || {
        let ctx = Context::new();
        let _ = tx.send(p(&ctx, fx));
    });
    match rx.recv_timeout(FIXTURE_TIMEOUT) {
        Ok(status) => status,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Status::Fail(format!(
            "HANG: no result within {:.0}s (thread abandoned)",
            FIXTURE_TIMEOUT.as_secs_f64()
        )),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Status::Fail("PANIC while processing the fixture (see stderr)".into())
        }
    }
}

/// Like [`run`], with a table of known bugs (strict xfail, see module docs).
pub fn run_with_known_bugs(
    category: &str,
    subcategory: &str,
    known_bugs: &[KnownBug],
    process: impl Fn(&Context, &Fixture) -> Status + Send + Sync + 'static,
) {
    let file = fixture_file();
    run_fixtures(
        &file.fixtures,
        &file.generated_by,
        category,
        Some(subcategory),
        known_bugs,
        process,
    );
}

/// Core runner over an arbitrary fixture slice.  `subcategory = None` runs a
/// whole category (used by fixture files keyed on category alone).
pub fn run_fixtures(
    fixtures: &'static [Fixture],
    generated_by: &str,
    category: &str,
    subcategory: Option<&str>,
    known_bugs: &[KnownBug],
    process: impl Fn(&Context, &Fixture) -> Status + Send + Sync + 'static,
) {
    let process: Processor = std::sync::Arc::new(process);
    let mut stats = Stats::default();
    let mut lines: Vec<String> = Vec::new();
    let start = std::time::Instant::now();
    let label = match subcategory {
        Some(s) => format!("{category}:{s}"),
        None => category.to_string(),
    };
    for fx in fixtures
        .iter()
        .filter(|f| f.category == category && subcategory.is_none_or(|s| f.subcategory == s))
    {
        let known = known_bugs
            .iter()
            .find(|(c, s, k, _)| {
                *c == category && (*s == fx.subcategory || *s == "*") && *k == fx.key
            })
            .map(|(_, _, _, reason)| *reason);
        let status = match fx.oracle_missing() {
            Some(reason) => Status::SkippedOracle(reason),
            None => process_bounded(&process, fx),
        };
        match status {
            Status::Pass if known.is_some() => {
                stats.unexpected_pass += 1;
                lines.push(format!(
                    "  UNEXPECTED_PASS id={} [{}] — listed as known bug ({}) but now passes: remove it from KNOWN_BUGS",
                    fx.id,
                    fx.key,
                    known.unwrap_or("")
                ));
            }
            Status::Pass => stats.pass += 1,
            Status::Fail(r) if known.is_some() => {
                stats.known_bug += 1;
                lines.push(format!(
                    "  KNOWN_BUG      id={} [{}] — {} (observed: {})",
                    fx.id,
                    fx.key,
                    known.unwrap_or(""),
                    r
                ));
            }
            Status::Fail(r) => {
                stats.fail += 1;
                lines.push(format!(
                    "  FAIL           id={} [{}] — {}",
                    fx.id, fx.key, r
                ));
            }
            Status::NotImplemented(r) => {
                stats.not_impl += 1;
                lines.push(format!(
                    "  NOT_IMPL       id={} [{}] — {}",
                    fx.id, fx.key, r
                ));
            }
            Status::UnsupportedApi(r) => {
                stats.no_api += 1;
                lines.push(format!(
                    "  NO_API         id={} [{}] — {}",
                    fx.id, fx.key, r
                ));
            }
            Status::SkippedOracle(r) => {
                stats.skipped_oracle += 1;
                lines.push(format!(
                    "  SKIPPED_ORACLE id={} [{}] — {}",
                    fx.id, fx.key, r
                ));
            }
        }
    }
    let elapsed = start.elapsed();
    println!(
        "{:<36} pass={:>3} fail={:>3} known_bug={:>2} not_impl={:>3} no_api={:>2} skipped_oracle={:>2}  ({:.2}s, {})",
        label,
        stats.pass,
        stats.fail,
        stats.known_bug,
        stats.not_impl,
        stats.no_api,
        stats.skipped_oracle,
        elapsed.as_secs_f64(),
        generated_by
    );
    for l in &lines {
        println!("{l}");
    }
    assert!(
        stats.total() > 0,
        "no fixtures found for {label} — generator/consumer out of sync"
    );
    assert_eq!(
        stats.unexpected_pass,
        0,
        "{} known-bug entr{} in {label} now pass — remove them from KNOWN_BUGS",
        stats.unexpected_pass,
        if stats.unexpected_pass == 1 {
            "y"
        } else {
            "ies"
        }
    );
    assert_eq!(
        stats.fail, 0,
        "{} fixture(s) in {label} produced WRONG results (see FAIL lines above); \
         known_bug={} not_impl={} no_api={} skipped_oracle={} are informational",
        stats.fail, stats.known_bug, stats.not_impl, stats.no_api, stats.skipped_oracle
    );
}

// ── Numeric helpers ────────────────────────────────────────────────────

pub fn approx_eq_tol(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    let diff = (a - b).abs();
    let denom = a.abs().max(b.abs()).max(1e-15);
    diff < tol || diff / denom < tol
}

pub fn approx_eq(a: f64, b: f64) -> bool {
    approx_eq_tol(a, b, TOLERANCE)
}

pub fn complex_matches(actual: (f64, f64), expected: (f64, f64), tol: f64) -> bool {
    // Compare with a tolerance relative to the magnitude of the whole number.
    let scale = (expected.0.hypot(expected.1))
        .max(actual.0.hypot(actual.1))
        .max(1.0);
    (actual.0 - expected.0).abs() / scale < tol && (actual.1 - expected.1).abs() / scale < tol
}

/// Parse a SymPy-syntax string; `oo` / `-oo` become ±∞.
pub fn parse(ctx: &Context, s: &str) -> Result<Ex, String> {
    match s.trim() {
        "oo" | "inf" => Ok(ctx.infinity()),
        "-oo" | "-inf" => Ok(ctx.neg_infinity()),
        "zoo" => Ok(ctx.complex_infinity()),
        other => {
            symplex::parse::parse(ctx, other).map_err(|e| format!("parse error in {other:?}: {e}"))
        }
    }
}

/// Build an exact expression for an `f64` sample point (integers stay
/// integers, everything else goes through the exact decimal parser).
pub fn f64_to_ex(ctx: &Context, v: f64) -> Option<Ex> {
    if v == v.floor() && v.abs() < 1e15 {
        Some(ctx.int(v as i64))
    } else {
        ctx.decimal_str(&format!("{v}")).ok()
    }
}

/// Substitute the sample point and evaluate to a complex `f64`.
pub fn eval_at(expr: &Ex, ctx: &Context, subs: &BTreeMap<String, f64>) -> Option<(f64, f64)> {
    let mut e = expr.clone();
    for (name, v) in subs {
        let sym = ctx.symbol(name);
        let val = f64_to_ex(ctx, *v)?;
        e = e.subs(&sym, &val);
    }
    e.eval_complex64().ok()
}

/// Classify an expression as an extended value, if it is one.
pub fn extended_value(ctx: &Context, e: &Ex) -> Option<Num> {
    let e = e.eval();
    if e == ctx.infinity() {
        return Some(Num::PosInf);
    }
    if e == ctx.neg_infinity() {
        return Some(Num::NegInf);
    }
    if e == ctx.complex_infinity() {
        return Some(Num::ComplexInf);
    }
    if e == ctx.nan() {
        return Some(Num::NaN);
    }
    None
}

/// Compare a constant symplex expression with the oracle value.
pub fn compare_constant(ctx: &Context, result: &Ex, expected: &Num, tol: f64) -> Status {
    if result.has_unevaluated() {
        return Status::NotImplemented(format!("unevaluated result: {}", truncate(result)));
    }
    if let Some(ext) = extended_value(ctx, result) {
        return if ext == *expected {
            Status::Pass
        } else {
            Status::Fail(format!("symplex={ext}, sympy={expected}"))
        };
    }
    match expected {
        Num::Finite(re, im) => match result.eval_complex64() {
            Ok(val) => {
                if complex_matches(val, (*re, *im), tol) {
                    Status::Pass
                } else {
                    Status::Fail(format!(
                        "symplex={} = ({}, {}i), sympy={}",
                        truncate(result),
                        val.0,
                        val.1,
                        expected
                    ))
                }
            }
            Err(e) => Status::NotImplemented(format!("cannot evaluate {}: {e}", truncate(result))),
        },
        other => Status::Fail(format!(
            "symplex={} (finite), sympy={other}",
            truncate(result)
        )),
    }
}

/// Compare `result(subs)` with the oracle at every eval point in `key`.
pub fn compare_eval_points(
    ctx: &Context,
    result: &Ex,
    fx: &Fixture,
    key: &str,
    tol: f64,
) -> Status {
    if result.has_unevaluated() {
        return Status::NotImplemented(format!("unevaluated result: {}", truncate(result)));
    }
    let pts = fx.eval_points(key);
    if pts.is_empty() {
        return Status::SkippedOracle("fixture has no eval points".into());
    }
    let mut mismatches = Vec::new();
    let mut evaluated = 0;
    for pt in &pts {
        let Some(Num::Finite(re, im)) = pt.value else {
            continue;
        };
        match eval_at(result, ctx, &pt.subs) {
            Some(val) => {
                evaluated += 1;
                if !complex_matches(val, (re, im), tol) {
                    mismatches.push(format!(
                        "at {:?}: symplex=({}, {}i) sympy=({}, {}i)",
                        pt.subs, val.0, val.1, re, im
                    ));
                }
            }
            None => mismatches.push(format!(
                "at {:?}: symplex could not evaluate {}",
                pt.subs,
                truncate(result)
            )),
        }
    }
    if evaluated == 0 {
        return Status::NotImplemented(format!(
            "could not evaluate {} at any point",
            truncate(result)
        ));
    }
    if mismatches.is_empty() {
        Status::Pass
    } else {
        Status::Fail(mismatches.join("; "))
    }
}

pub fn truncate(e: &Ex) -> String {
    let s = e.to_string();
    if s.chars().count() > 90 {
        let cut: String = s.chars().take(90).collect();
        format!("{cut}…")
    } else {
        s
    }
}

// ── Matrices ───────────────────────────────────────────────────────────

pub fn parse_matrix(ctx: &Context, fx: &Fixture, key: &str) -> Result<Matrix, String> {
    let rows = fx
        .field(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("fixture has no {key}"))?;
    let mut data = Vec::with_capacity(rows.len());
    for row in rows {
        let cells = row.as_array().ok_or("row is not an array")?;
        let mut r = Vec::with_capacity(cells.len());
        for c in cells {
            let s = c.as_str().ok_or("cell is not a string")?;
            r.push(parse(ctx, s)?);
        }
        data.push(r);
    }
    Matrix::new(data).map_err(|e| e.to_string())
}

/// Numeric `[[re, im]]` view of a matrix, or the entry that failed.
pub fn matrix_c64(m: &Matrix) -> Result<Vec<Vec<(f64, f64)>>, String> {
    let (nr, nc) = m.shape();
    let mut out = Vec::with_capacity(nr);
    for i in 0..nr {
        let mut row = Vec::with_capacity(nc);
        for j in 0..nc {
            let e = m.get(i, j);
            row.push(
                e.eval_complex64().map_err(|err| {
                    format!("entry ({i},{j}) = {} not numeric: {err}", truncate(e))
                })?,
            );
        }
        out.push(row);
    }
    Ok(out)
}

pub fn parse_num_matrix(fx: &Fixture, key: &str) -> Option<Vec<Vec<Num>>> {
    let rows = fx.field(key)?.as_array()?;
    rows.iter()
        .map(|r| r.as_array()?.iter().map(Num::from_json).collect())
        .collect()
}

pub fn sorted_complex(mut v: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    v.sort_by(|a, b| {
        let ka = ((a.0 * 1e9).round(), (a.1 * 1e9).round());
        let kb = ((b.0 * 1e9).round(), (b.1 * 1e9).round());
        ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
    });
    v
}

pub fn nums_to_complex(v: &[Num]) -> Option<Vec<(f64, f64)>> {
    v.iter()
        .map(|n| match n {
            Num::Finite(re, im) => Some((*re, *im)),
            _ => None,
        })
        .collect()
}

/// Compare two multisets of complex numbers (both sorted first).
pub fn compare_complex_multisets(got: Vec<(f64, f64)>, want: Vec<(f64, f64)>, tol: f64) -> Status {
    let got = sorted_complex(got);
    let want = sorted_complex(want);
    if got.len() != want.len() {
        return Status::Fail(format!(
            "count mismatch: symplex={} sympy={}",
            got.len(),
            want.len()
        ));
    }
    for (g, w) in got.iter().zip(&want) {
        if !complex_matches(*g, *w, tol) {
            return Status::Fail(format!("symplex={got:?} sympy={want:?}"));
        }
    }
    Status::Pass
}

/// Read a big integer from a decimal string.
pub fn bigint(s: &str) -> num_bigint::BigInt {
    s.parse::<num_bigint::BigInt>()
        .unwrap_or_else(|e| panic!("bad integer {s:?}: {e}"))
}
