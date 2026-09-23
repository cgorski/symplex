//! Output files and the ratchet.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::driver::FileRun;
use crate::protocol::Status;

/// Counts in the column order of `summary.tsv`.
#[derive(Debug, Clone, Default)]
pub struct Counts {
    pub total: usize,
    pub by_status: BTreeMap<Status, usize>,
    pub secs: f64,
}

impl Counts {
    pub fn of(f: &FileRun) -> Counts {
        let mut c = Counts {
            total: f.entries.len(),
            secs: f.secs,
            ..Counts::default()
        };
        for st in Status::ALL {
            c.by_status.insert(st, f.count(st));
        }
        c
    }

    pub fn get(&self, st: Status) -> usize {
        self.by_status.get(&st).copied().unwrap_or(0)
    }

    fn add(&mut self, o: &Counts) {
        self.total += o.total;
        self.secs += o.secs;
        for (st, n) in &o.by_status {
            *self.by_status.entry(*st).or_insert(0) += n;
        }
    }

    fn row(&self, label: &str) -> String {
        format!(
            "{label}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}",
            self.total,
            self.get(Status::Unsupported),
            self.get(Status::Verified),
            self.get(Status::RealVerified),
            self.get(Status::Unevaluated),
            self.get(Status::Wrong),
            self.get(Status::Undecided),
            self.get(Status::Timeout),
            self.get(Status::Panic),
            self.secs
        )
    }
}

const HEADER: &str = "path\ttotal\tunsupported\tverified\treal_verified\tunevaluated\twrong\t\
                      undecided\ttimeout\tpanic\tseconds";

/// Per-chapter totals (first path component), and the grand total.
pub fn chapter_totals(files: &[FileRun]) -> (BTreeMap<String, Counts>, Counts) {
    let mut chapters: BTreeMap<String, Counts> = BTreeMap::new();
    let mut all = Counts::default();
    for f in files {
        let c = Counts::of(f);
        chapters.entry(f.chapter().to_string()).or_default().add(&c);
        all.add(&c);
    }
    (chapters, all)
}

/// `summary.tsv`: one row per file, then `TOTAL <chapter>` rows and `TOTAL`.
pub fn write_summary(path: &Path, files: &[FileRun], header_note: &str) -> std::io::Result<()> {
    let mut s = String::new();
    let _ = writeln!(s, "# {header_note}");
    let _ = writeln!(s, "{HEADER}");
    for f in files {
        let _ = writeln!(s, "{}", Counts::of(f).row(&f.rel));
    }
    let (chapters, all) = chapter_totals(files);
    for (ch, c) in &chapters {
        let _ = writeln!(s, "{}", c.row(&format!("TOTAL {ch}")));
    }
    let _ = writeln!(s, "{}", all.row("TOTAL"));
    std::fs::write(path, s)
}

/// Console table of the chapter totals.
pub fn chapter_table(files: &[FileRun]) -> String {
    let (chapters, all) = chapter_totals(files);
    let mut s = String::new();
    let _ = writeln!(
        s,
        "| chapter | total | unsupported | verified | real_verified | unevaluated | wrong | undecided | timeout | panic |"
    );
    let _ = writeln!(s, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let line = |s: &mut String, name: &str, c: &Counts| {
        let _ = writeln!(
            s,
            "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            c.total,
            c.get(Status::Unsupported),
            c.get(Status::Verified),
            c.get(Status::RealVerified),
            c.get(Status::Unevaluated),
            c.get(Status::Wrong),
            c.get(Status::Undecided),
            c.get(Status::Timeout),
            c.get(Status::Panic)
        );
    };
    for (ch, c) in &chapters {
        line(&mut s, ch, c);
    }
    line(&mut s, "**all**", &all);
    s
}

/// `undecided.txt`: every undecided case with the reason at each point.
pub fn write_undecided(path: &Path, files: &[FileRun], header_note: &str) -> std::io::Result<()> {
    let mut s = format!("# {header_note}\n\n");
    for f in files {
        for (idx, r) in f.results.iter().enumerate() {
            let Some(r) = r.as_ref().filter(|r| r.status == Status::Undecided) else {
                continue;
            };
            let e = &f.entries[idx];
            let _ = writeln!(
                s,
                "=== UNDECIDED  {}  entry {} (line {})",
                f.rel,
                idx + 1,
                e.line
            );
            let _ = writeln!(s, "integrand (Maxima): {}", e.integrand());
            for key in ["reason", "F", "detail"] {
                if let Some(v) = r.get(key) {
                    let _ = writeln!(s, "{key}: {}", v.replace('\n', "\n    "));
                }
            }
            let _ = writeln!(s);
        }
    }
    std::fs::write(path, s)
}

const REPRODUCE: &str = "Reproduce with: let ctx = Context::new(); \
     let f = ctx.parse(<translated>).unwrap(); \
     let big_f = f.integrate(&ctx.symbol(<variable>));";

/// Tag the first `;`-separated clause of `v` for grouping (`"a (b); c"` → `"a"`).
fn head_clause(v: &str) -> &str {
    let first = v.split(';').next().unwrap_or(v);
    first.split(" (").next().unwrap_or(first)
}

/// `wrong.txt`: every wrong and panic case with what is needed to
/// reproduce it.
pub fn write_wrong(path: &Path, files: &[FileRun], header_note: &str) -> std::io::Result<()> {
    let mut s = String::new();
    let (mut n_wrong, mut n_panic) = (0, 0);
    let mut by_tag: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_rubi: BTreeMap<String, usize> = BTreeMap::new();
    for r in files.iter().flat_map(|f| f.results.iter().flatten()) {
        match r.status {
            Status::Wrong => {
                n_wrong += 1;
                *by_tag
                    .entry(r.get("tag").unwrap_or("?").to_string())
                    .or_insert(0) += 1;
                let rubi = r.get("rubi").unwrap_or("n/a: not run");
                *by_rubi.entry(head_clause(rubi).to_string()).or_insert(0) += 1;
            }
            Status::Panic => n_panic += 1,
            _ => {}
        }
    }
    let _ = writeln!(s, "# {header_note}");
    let _ = writeln!(
        s,
        "# {n_wrong} WRONG and {n_panic} PANIC cases.  {REPRODUCE}"
    );
    let _ = writeln!(
        s,
        "# WRONG = F' differs from f at a sample point where f is real, or anywhere if the \
         integrand contains %i.  Mismatches only where f is complex are in real_verified.txt."
    );
    let _ = writeln!(s, "# WRONG by classification:");
    for (t, k) in &by_tag {
        let _ = writeln!(s, "#   {k:>6}  {t}");
    }
    let _ = writeln!(
        s,
        "# WRONG by Rubi cross-check (Rubi's optimal antiderivative, same points):"
    );
    for (t, k) in &by_rubi {
        let _ = writeln!(s, "#   {k:>6}  {t}");
    }
    let _ = writeln!(s);
    write_cases(&mut s, files, &[Status::Wrong, Status::Panic]);
    std::fs::write(path, s)
}

/// `real_verified.txt`: every answer that mismatches only where the
/// integrand is complex (symplex's real-variable convention).
pub fn write_real_verified(
    path: &Path,
    files: &[FileRun],
    header_note: &str,
) -> std::io::Result<()> {
    let mut s = String::new();
    let mut by_tag: BTreeMap<String, usize> = BTreeMap::new();
    for r in files.iter().flat_map(|f| f.results.iter().flatten()) {
        if r.status == Status::RealVerified {
            let vacuous = r
                .get("tag")
                .is_some_and(|t| t.contains("no real sample point"));
            let key = if vacuous {
                "integrand complex at every evaluated point (vacuously real_verified)"
            } else {
                "agrees at one or more points where the integrand is real"
            };
            *by_tag.entry(key.to_string()).or_insert(0) += 1;
        }
    }
    let n: usize = by_tag.values().sum();
    let _ = writeln!(s, "# {header_note}");
    let _ = writeln!(
        s,
        "# {n} REAL_VERIFIED cases: F' = f at every sample point where f is real; F' differs \
         from f only where f is complex (e.g. ln|u| for u'/u with u complex), and the \
         integrand has no %i.  Not bugs under symplex's real-variable convention.  {REPRODUCE}"
    );
    for (t, k) in &by_tag {
        let _ = writeln!(s, "#   {k:>6}  {t}");
    }
    let _ = writeln!(s);
    write_cases(&mut s, files, &[Status::RealVerified]);
    std::fs::write(path, s)
}

/// One block per entry whose status is in `kinds`, numbered per kind.
fn write_cases(s: &mut String, files: &[FileRun], kinds: &[Status]) {
    let mut numbers: BTreeMap<Status, usize> = BTreeMap::new();
    for f in files {
        for (idx, r) in f.results.iter().enumerate() {
            let Some(r) = r else { continue };
            if !kinds.contains(&r.status) {
                continue;
            }
            let n = numbers.entry(r.status).or_insert(0);
            *n += 1;
            let n = *n;
            let e = &f.entries[idx];
            let kind = r.status.as_str().to_ascii_uppercase();
            let _ = writeln!(
                s,
                "=== {kind} #{n}  {}  entry {} (line {})",
                f.rel,
                idx + 1,
                e.line
            );
            let _ = writeln!(s, "integrand (Maxima): {}", e.integrand());
            let _ = writeln!(s, "variable:           {}", e.variable());
            if let Some(t) = r.get("translated") {
                let _ = writeln!(s, "translated:         {t}");
            }
            if let Some(o) = e.optimal() {
                let _ = writeln!(s, "Rubi optimal:       {o}");
            }
            for (key, label) in [
                ("tag", "classification"),
                ("reason", "reason"),
                ("panic", "panic"),
                ("note", "note"),
                ("F", "symplex F"),
                ("params", "parameters"),
            ] {
                if let Some(v) = r.get(key) {
                    let _ = writeln!(s, "{label:<20}{v}", label = format!("{label}:"));
                }
            }
            if let Some(p) = r.get("points") {
                let _ = writeln!(s, "points:");
                for l in p.lines() {
                    let _ = writeln!(s, "    {l}");
                }
            }
            if let Some(v) = r.get("rubi") {
                let _ = writeln!(s, "rubi cross-check:   {v}");
            }
            let _ = writeln!(s, "seconds:            {:.2}", r.secs);
            let _ = writeln!(s);
        }
    }
}

/// `entries.tsv`: one line per entry (status, seconds, reason).
pub fn write_entries(path: &Path, files: &[FileRun]) -> std::io::Result<()> {
    let mut s = String::from("path\tentry\tline\tstatus\tseconds\treason\n");
    for f in files {
        for (idx, r) in f.results.iter().enumerate() {
            let e = &f.entries[idx];
            let (st, secs, reason) = match r {
                Some(r) => (
                    r.status.as_str(),
                    r.secs,
                    r.get("reason").unwrap_or("").replace(['\t', '\n'], " "),
                ),
                None => ("missing", 0.0, String::new()),
            };
            let _ = writeln!(
                s,
                "{}\t{}\t{}\t{st}\t{secs:.3}\t{reason}",
                f.rel,
                idx + 1,
                e.line
            );
        }
    }
    std::fs::write(path, s)
}

/// Reasons for `unsupported` (and `undecided`) with counts, most common
/// first.
pub fn reason_counts(files: &[FileRun], st: Status) -> Vec<(String, usize)> {
    let mut m: BTreeMap<String, usize> = BTreeMap::new();
    for f in files {
        for r in f.results.iter().flatten() {
            if r.status == st {
                *m.entry(r.get("reason").unwrap_or("?").to_string())
                    .or_insert(0) += 1;
            }
        }
    }
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

pub fn write_reasons(path: &Path, files: &[FileRun]) -> std::io::Result<()> {
    let mut s = String::from("status\tcount\treason\n");
    for st in [
        Status::Unsupported,
        Status::Undecided,
        Status::Timeout,
        Status::Panic,
    ] {
        for (reason, n) in reason_counts(files, st) {
            let _ = writeln!(s, "{st}\t{n}\t{reason}");
        }
    }
    std::fs::write(path, s)
}

// ─── Ratchet ────────────────────────────────────────────────────────────

/// Per file: the minimum verified + real_verified count and the maximum
/// wrong + panic count.
pub type Ratchet = BTreeMap<String, (usize, usize)>;

pub fn read_ratchet(path: &Path) -> Result<Ratchet, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Ratchet::new()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let mut r = Ratchet::new();
    for (i, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.trim().is_empty() || line.starts_with("path\t") {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let parsed = (cols.len() == 3)
            .then(|| Some((cols[1].parse().ok()?, cols[2].parse().ok()?)))
            .flatten();
        match parsed {
            Some(v) => {
                r.insert(cols[0].to_string(), v);
            }
            None => return Err(format!("{}:{}: malformed line", path.display(), i + 1)),
        }
    }
    Ok(r)
}

pub fn current_ratchet(files: &[FileRun]) -> Ratchet {
    files
        .iter()
        .filter(|f| f.error.is_none())
        .map(|f| {
            (
                f.rel.clone(),
                (
                    f.count(Status::Verified) + f.count(Status::RealVerified),
                    f.count(Status::Wrong) + f.count(Status::Panic),
                ),
            )
        })
        .collect()
}

pub fn write_ratchet(path: &Path, r: &Ratchet) -> std::io::Result<()> {
    let mut s = String::from(
        "# Rubi harness ratchet: `cargo run --release -- --check` fails if a file's\n\
         # verified + real_verified count drops below min_verified or its wrong + panic\n\
         # count rises above max_wrong_panic.  Regenerate with `--update`.\n",
    );
    s.push_str("path\tmin_verified\tmax_wrong_panic\n");
    for (p, (v, w)) in r {
        let _ = writeln!(s, "{p}\t{v}\t{w}");
    }
    std::fs::write(path, s)
}

/// Violations of `old` by `files` (files absent from `old` are listed as
/// notes, not violations).
pub fn check_ratchet(old: &Ratchet, files: &[FileRun]) -> (Vec<String>, Vec<String>) {
    let mut violations = Vec::new();
    let mut notes = Vec::new();
    for (path, (v, w)) in current_ratchet(files) {
        match old.get(&path) {
            Some(&(min_v, max_w)) => {
                if v < min_v {
                    violations.push(format!(
                        "{path}: verified+real_verified {v} < ratchet {min_v}"
                    ));
                }
                if w > max_w {
                    violations.push(format!("{path}: wrong+panic {w} > ratchet {max_w}"));
                }
                if v > min_v || w < max_w {
                    notes.push(format!(
                        "{path}: improved (verified+real_verified {min_v} -> {v}, wrong+panic {max_w} -> {w}); run --update"
                    ));
                }
            }
            None => notes.push(format!("{path}: not in the ratchet")),
        }
    }
    (violations, notes)
}
