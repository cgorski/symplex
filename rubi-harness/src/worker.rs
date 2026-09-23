//! The worker process: integrates and checks a range of entries of one
//! file, streaming records on stdout (see [`crate::protocol`]).
//!
//! The driver enforces the wall-clock limits by killing this process, so
//! nothing here needs to be interruptible.  Each entry gets a fresh
//! `Context`; panics are caught per entry.  A stack overflow cannot be
//! caught — the process dies and the driver reports the entry as a panic.

use std::cell::RefCell;
use std::io::Write;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;

use symplex::prelude::{Context, Ex};

use crate::check::{self, Env, Outcome, PointReport};
use crate::mac::{self, Entry};
use crate::protocol::{Status, escape};
use crate::translate::{self, Unsupported};

/// Longest `F` text reported.
const MAX_TEXT: usize = 20_000;
/// Stack of the thread that runs symplex.
const STACK_BYTES: usize = 1 << 29;

thread_local! {
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn install_panic_hook() {
    panic::set_hook(Box::new(|info| {
        let payload = info.payload();
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "non-string panic payload".to_string()
        };
        let loc = info
            .location()
            .map(|l| format!(" at {}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_default();
        LAST_PANIC.with(|p| *p.borrow_mut() = Some(format!("{msg}{loc}")));
    }));
}

fn take_panic() -> String {
    LAST_PANIC
        .with(|p| p.borrow_mut().take())
        .unwrap_or_else(|| "unknown panic".into())
}

/// Line writer on stdout.
struct Out {
    idx: usize,
    stage: std::cell::Cell<&'static str>,
}

impl Out {
    /// Record the current stage (reported with panics, timeouts, crashes).
    fn stage(&self, s: &'static str) {
        self.stage.set(s);
        self.kv("stage", s);
    }
    fn line(&self, s: &str) {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        // A closed pipe means the driver is gone; nothing useful to do.
        let _ = writeln!(lock, "{s}");
        let _ = lock.flush();
    }

    fn begin(&self) {
        self.line(&format!("BEGIN\t{}", self.idx));
    }

    fn phase(&self, name: &str) {
        self.line(&format!("PHASE\t{}\t{name}", self.idx));
    }

    fn kv(&self, key: &str, value: &str) {
        self.line(&format!("KV\t{}\t{key}\t{}", self.idx, escape(value)));
    }

    fn end(&self, st: Status) {
        self.line(&format!("END\t{}\t{st}", self.idx));
    }

    fn unsupported(&self, u: &Unsupported, prefix: &str) -> Status {
        self.kv("reason", &format!("{prefix}{}", u.reason));
        if !u.detail.is_empty() {
            self.kv("detail", &u.detail);
        }
        Status::Unsupported
    }
}

/// Entry point of `rubi-harness --worker FILE START END [--selftest]`.
pub fn main(file: &Path, start: usize, end: usize, selftest: bool) -> i32 {
    install_panic_hook();
    let entries = match mac::read_file(file) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("worker: {e}");
            return 2;
        }
    };
    let end = end.min(entries.len());
    let handle = std::thread::Builder::new()
        .name("symplex".into())
        .stack_size(STACK_BYTES)
        .spawn(move || {
            for (idx, entry) in entries.iter().enumerate().take(end).skip(start) {
                run_entry(idx, entry, selftest);
            }
        });
    match handle.map(|h| h.join()) {
        Ok(Ok(())) => 0,
        _ => 3,
    }
}

/// Test hook for the driver's failure handling: `RUBI_HARNESS_FAULT=kind:idx`
/// with kind `panic`, `abort` (process dies) or `hang`, at entry `idx`
/// (0-based) of every file.
fn inject_fault(idx: usize) {
    let Ok(spec) = std::env::var("RUBI_HARNESS_FAULT") else {
        return;
    };
    let Some((kind, at)) = spec.split_once(':') else {
        return;
    };
    if at.parse::<usize>().ok() != Some(idx) {
        return;
    }
    match kind {
        "panic" => panic!("injected fault"),
        "abort" => std::process::abort(),
        "hang" => loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        },
        _ => {}
    }
}

fn run_entry(idx: usize, entry: &Entry, selftest: bool) {
    let out = Out {
        idx,
        stage: std::cell::Cell::new("startup"),
    };
    out.begin();
    let status = match panic::catch_unwind(AssertUnwindSafe(|| entry_body(&out, entry, selftest))) {
        Ok(st) => st,
        Err(_) => {
            out.kv("reason", &format!("panic during {}", out.stage.get()));
            out.kv("panic", &take_panic());
            Status::Panic
        }
    };
    out.end(status);
}

fn truncated(s: String) -> String {
    if s.len() <= MAX_TEXT {
        return s;
    }
    let mut cut = MAX_TEXT;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{} … [truncated; {} bytes in total]", &s[..cut], s.len())
}

/// Parse a translated Maxima expression in `ctx`.
fn parse_maxima(ctx: &Context, src: &str) -> Result<Ex, Unsupported> {
    let tr = translate::translate(src)?;
    ctx.parse(&tr.text).map_err(|e| Unsupported {
        reason: "symplex parse error".into(),
        detail: format!("{e}; input: {}", tr.text),
    })
}

fn entry_body(out: &Out, entry: &Entry, selftest: bool) -> Status {
    out.stage("translate");
    inject_fault(out.idx);
    let tr = match translate::translate(entry.integrand()) {
        Ok(t) => t,
        Err(u) => return out.unsupported(&u, ""),
    };
    out.kv("translated", &tr.text);
    let var = match translate::translate_variable(entry.variable()) {
        Ok(v) => v,
        Err(u) => return out.unsupported(&u, ""),
    };

    out.stage("parse");
    let ctx = Context::new();
    let f = match ctx.parse(&tr.text) {
        Ok(f) => f,
        Err(e) => {
            out.kv("reason", "symplex parse error");
            out.kv("detail", &e.to_string());
            return Status::Unsupported;
        }
    };
    let x = ctx.symbol(&var);

    out.stage("integrate");
    let big_f = if selftest {
        match entry.optimal().map(|o| parse_maxima(&ctx, o)) {
            Some(Ok(g)) => g,
            Some(Err(u)) => return out.unsupported(&u, "optimal: "),
            None => {
                out.kv(
                    "reason",
                    "optimal: missing (or Rubi failed: negative step count)",
                );
                return Status::Unsupported;
            }
        }
    } else {
        f.integrate(&x)
    };
    if big_f.has_unevaluated() {
        return Status::Unevaluated;
    }

    out.phase("check");
    out.stage("check");
    let mut env = Env::new();
    env.extend(&ctx, &x, &[&f, &big_f]);
    let reports = check::check(&ctx, &f, &big_f, &x, &env);

    let explicit_i = entry.integrand().contains("%i");
    let status = check::verdict(&reports, explicit_i);
    if matches!(status, Status::Wrong | Status::RealVerified) {
        // Sent first: a verdict survives a timeout while printing `F`.
        out.kv("verdict", status.as_str());
        out.kv("params", &env.desc);
        let points: Vec<String> = reports.iter().map(|r| r.describe(&var)).collect();
        out.kv("points", &points.join("\n"));
        let real_fail = reports
            .iter()
            .any(|r| r.failed() && r.integrand_real() == Some(true));
        let real_ok = check::real_agreements(&reports);
        let tag = if real_fail {
            "mismatch where the integrand is real".to_string()
        } else if status == Status::Wrong {
            "mismatch only where the integrand is complex, but the integrand contains %i \
             (complex by construction, so not a real-variable convention)"
                .to_string()
        } else if real_ok == 0 {
            "mismatch only where the integrand is complex; no real sample point evaluated \
             (real-variable convention, vacuously)"
                .to_string()
        } else {
            format!(
                "mismatch only where the integrand is complex; agrees at {real_ok} real \
                 point(s) (real-variable convention)"
            )
        };
        out.kv("tag", &tag);
        out.stage("display");
        out.kv("F", &truncated(big_f.to_string()));
        if status == Status::Wrong && !selftest {
            out.stage("rubi-check");
            let rubi = rubi_cross_check(&ctx, entry, &f, &x, &mut env, &reports);
            out.kv("rubi", &rubi);
        }
        return status;
    }
    if status == Status::Verified {
        return status;
    }
    out.kv("reason", "no point evaluated on both sides");
    let why: Vec<String> = reports.iter().map(|r| r.describe(&var)).collect();
    out.kv("detail", &why.join("\n"));
    out.kv("F", &truncated(big_f.to_string()));
    Status::Undecided
}

/// Differentiate Rubi's optimal antiderivative and compare it with the
/// integrand at the points where symplex's answer mismatched.  If Rubi's
/// (verified) answer also "mismatches", the harness — translation, branch
/// conventions, evaluation — is suspect rather than symplex.
fn rubi_cross_check(
    ctx: &Context,
    entry: &Entry,
    f: &Ex,
    x: &Ex,
    env: &mut Env,
    reports: &[PointReport],
) -> String {
    let Some(opt) = entry.optimal() else {
        return "n/a: no optimal antiderivative (Rubi failed on this entry)".into();
    };
    let g = match parse_maxima(ctx, opt) {
        Ok(g) => g,
        Err(u) => return format!("n/a: optimal antiderivative unsupported ({})", u.reason),
    };
    if g.has_unevaluated() {
        return "n/a: optimal antiderivative unevaluated".into();
    }
    env.extend(ctx, x, &[&g]);
    let f_s = env.apply(f);
    let dg = env.apply(&g).diff(x);
    let mut agree = 0;
    let mut differ = 0;
    let mut lines = Vec::new();
    for r in reports.iter().filter(|r| r.failed()) {
        let (p, q) = r.x;
        match check::compare_at(ctx, &dg, &f_s, x, r.x) {
            Outcome::Ok { lhs, .. } => {
                agree += 1;
                lines.push(format!("at {p}/{q}: Rubi's F'={lhs} agrees with f"));
            }
            Outcome::Fail { lhs, .. } => {
                differ += 1;
                lines.push(format!("at {p}/{q}: Rubi's F'={lhs} ALSO differs from f"));
            }
            Outcome::Skip(why) => lines.push(format!("at {p}/{q}: not evaluable ({why})")),
        }
    }
    let head = if differ > 0 {
        "RUBI ALSO MISMATCHES (suspect harness/branch artefact)"
    } else if agree > 0 {
        "Rubi's answer agrees with the integrand at these points"
    } else {
        "n/a: Rubi's derivative not evaluable"
    };
    format!("{head}; {}", lines.join("; "))
}
