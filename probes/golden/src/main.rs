//! `golden` — a deterministic golden-output harness for symplex.
//!
//! Prints one TSV line `case_id \t op \t status \t result` per (case, op),
//! flushed as it goes.  See `README.md` for the design and the driver.

mod alloc;
mod cases;
mod ops;

use std::io::Write;
use std::panic::{self, AssertUnwindSafe};
use std::process::exit;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

#[global_allocator]
static GLOBAL: alloc::Capped = alloc::Capped;

/// Worker stack (reserved address space, committed only as used).
const STACK_BYTES: usize = 1 << 29;
/// Results longer than this (in chars, after escaping) are hashed.
const MAX_RESULT: usize = 400;

const USAGE: &str = "\
usage: golden [--seeds A..B | --file PATH [--seeds A..B]] [--ops LIST|all]
              [--timeout-ms N] [--mem-mb N] [--after OP]
       golden --list-ops [--ops LIST|all]
       golden --list-fns
       golden --info

  --seeds A..B     generated cases with ids A <= id < B
  --file PATH      one expression per line; case id = 1-based line number
                   (blank lines and lines starting with # are skipped);
                   --seeds then selects line numbers A <= n < B
  --ops LIST       comma-separated op names, or all (default all); output
                   follows the canonical order of --list-ops; the pseudo-op
                   expr (the input string) is always included
  --timeout-ms N   per-op limit (default 5000); on expiry print TIMEOUT and
                   exit with code 3 (the abandoned worker cannot be stopped)
  --mem-mb N       abort (stderr 'golden: OOM') above N MiB live heap
                   (default 4096)
  --after OP       for the first case only, skip ops up to and including OP
                   (the driver resumes with it after a TIMEOUT or crash)
";

struct Args {
    seeds: Option<(u64, u64)>,
    file: Option<String>,
    ops: Vec<&'static str>,
    timeout_ms: u64,
    mem_mb: u64,
    after: Option<usize>,
    list_ops: bool,
    list_fns: bool,
    info: bool,
}

fn die(msg: &str) -> ! {
    eprintln!("golden: {msg}\n\n{USAGE}");
    exit(2)
}

fn parse_range(s: &str) -> (u64, u64) {
    let Some((a, b)) = s.split_once("..") else {
        die(&format!("bad range '{s}' (want A..B)"))
    };
    match (a.trim().parse::<u64>(), b.trim().parse::<u64>()) {
        (Ok(a), Ok(b)) if a <= b => (a, b),
        _ => die(&format!("bad range '{s}' (want A..B with A <= B)")),
    }
}

fn parse_args() -> Args {
    let mut a = Args {
        seeds: None,
        file: None,
        ops: ops::OPS.to_vec(),
        timeout_ms: 5000,
        mem_mb: 4096,
        after: None,
        list_ops: false,
        list_fns: false,
        info: false,
    };
    let mut after_name: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().unwrap_or_else(|| die(&format!("{name} needs a value")));
        match arg.as_str() {
            "--seeds" => a.seeds = Some(parse_range(&val("--seeds"))),
            "--file" => a.file = Some(val("--file")),
            "--ops" => {
                let v = val("--ops");
                if v != "all" {
                    let want: Vec<&str> = v.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
                    for w in &want {
                        if !ops::OPS.contains(w) {
                            die(&format!("unknown op '{w}'"));
                        }
                    }
                    a.ops = ops::OPS
                        .iter()
                        .copied()
                        .filter(|o| *o == "expr" || want.contains(o))
                        .collect();
                }
            }
            "--timeout-ms" => {
                a.timeout_ms = val("--timeout-ms").parse().unwrap_or_else(|_| die("bad --timeout-ms"))
            }
            "--mem-mb" => a.mem_mb = val("--mem-mb").parse().unwrap_or_else(|_| die("bad --mem-mb")),
            "--after" => after_name = Some(val("--after")),
            "--list-ops" => a.list_ops = true,
            "--list-fns" => a.list_fns = true,
            "--info" => a.info = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                exit(0)
            }
            other => die(&format!("unknown argument '{other}'")),
        }
    }
    if let Some(n) = after_name {
        match a.ops.iter().position(|o| *o == n) {
            Some(k) => a.after = Some(k),
            None => die(&format!("--after {n}: not among the selected ops")),
        }
    }
    a
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Escape control characters (TSV safety); hash long results.
fn clean(s: &str, hash_long: bool) -> String {
    let mut t = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\t' => t.push_str("\\t"),
            '\n' => t.push_str("\\n"),
            '\r' => t.push_str("\\r"),
            c if (c as u32) < 0x20 || c == '\u{7f}' => t.push_str(&format!("\\x{:02x}", c as u32)),
            c => t.push(c),
        }
    }
    if hash_long {
        let n = t.chars().count();
        if n > MAX_RESULT {
            return format!("len={n} fnv={:016x}", fnv1a64(t.as_bytes()));
        }
    }
    t
}

fn emit(out: &mut impl Write, id: u64, op: &str, status: &str, result: &str) {
    let line = format!("{id}\t{op}\t{status}\t{}\n", clean(result, op != "expr"));
    if out.write_all(line.as_bytes()).and_then(|()| out.flush()).is_err() {
        exit(4);
    }
}

enum Outcome {
    Ok(String),
    Err(String),
    Panic(String),
    Timeout,
}

/// Location and message of the last panic, recorded by the hook (the default
/// hook's stderr report is suppressed; `main` prints it for PANIC lines).
static LAST_PANIC: Mutex<Option<String>> = Mutex::new(None);

fn payload_text(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".into()
    }
}

fn install_panic_hook() {
    panic::set_hook(Box::new(|info| {
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".into());
        let msg = payload_text(info.payload());
        if let Ok(mut g) = LAST_PANIC.lock() {
            *g = Some(format!("{loc}: {msg}"));
        }
    }));
}

/// Test hook for the driver's failure handling: `GOLDEN_FAULT=kind:case:op`
/// with kind `panic`, `abort`, `hang` (worker never returns), `oom` (worker
/// allocates until the cap) or `stall` (the main thread stops printing).
fn fault(id: u64, op: &str) -> Option<&'static str> {
    static SPEC: OnceLock<Option<(String, u64, String)>> = OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        let v = std::env::var("GOLDEN_FAULT").ok()?;
        let mut it = v.splitn(3, ':');
        let kind = it.next()?.to_string();
        let case = it.next()?.parse().ok()?;
        Some((kind, case, it.next()?.to_string()))
    });
    match spec {
        Some((kind, case, o)) if *case == id && o == op => Some(match kind.as_str() {
            "panic" => "panic",
            "abort" => "abort",
            "hang" => "hang",
            "oom" => "oom",
            "stall" => "stall",
            _ => return None,
        }),
        _ => None,
    }
}

fn inject(kind: &str) {
    match kind {
        "panic" => panic!("injected fault"),
        "abort" => std::process::abort(),
        "hang" | "stall" => loop {
            std::thread::sleep(Duration::from_secs(1));
        },
        "oom" => {
            let mut hoard: Vec<Vec<u8>> = Vec::new();
            loop {
                hoard.push(vec![1u8; 1 << 26]);
            }
        }
        _ => {}
    }
}

fn run_op(id: u64, op: &'static str, src: &Arc<str>, timeout: Duration) -> Outcome {
    if let Ok(mut g) = LAST_PANIC.lock() {
        *g = None;
    }
    let injected = fault(id, op);
    if injected == Some("stall") {
        inject("stall");
    }
    let (tx, rx) = mpsc::sync_channel::<Outcome>(1);
    let s = Arc::clone(src);
    let spawned = std::thread::Builder::new()
        .name(format!("golden-{op}"))
        .stack_size(STACK_BYTES)
        .spawn(move || {
            // The context and everything in it are dropped inside `ops::run`,
            // before the result is sent.
            let o = match panic::catch_unwind(AssertUnwindSafe(|| {
                if let Some(kind) = injected {
                    inject(kind);
                }
                ops::run(op, &s)
            })) {
                Ok(Ok(v)) => Outcome::Ok(v),
                Ok(Err(e)) => Outcome::Err(e),
                Err(p) => Outcome::Panic(payload_text(p.as_ref())),
            };
            let _ = tx.send(o);
        });
    let handle = match spawned {
        Ok(h) => h,
        Err(e) => {
            eprintln!("golden: cannot spawn a worker thread: {e}");
            exit(5)
        }
    };
    match rx.recv_timeout(timeout) {
        Ok(o) => {
            let _ = handle.join();
            o
        }
        Err(RecvTimeoutError::Timeout) => Outcome::Timeout,
        Err(RecvTimeoutError::Disconnected) => {
            let _ = handle.join();
            Outcome::Panic("worker exited without a result".into())
        }
    }
}

fn list_fns() {
    use symplex::base::libfn::LibFn;
    for f in cases::DEDICATED {
        println!("dedicated\t{}\t{}", f.name, f.args.len());
    }
    for f in cases::LIBFNS {
        println!("libfn\t{}\t{}", f.name, f.args.len());
    }
    // Compare the frozen table with this tree's registry.
    let range = |f: LibFn| {
        let ns: Vec<u8> = (0u8..=8).filter(|n| f.arity().accepts(usize::from(*n))).collect();
        (ns.first().copied().unwrap_or(0), ns.last().copied().unwrap_or(0))
    };
    let mut problems = 0;
    for f in LibFn::ALL {
        let (lo, hi) = range(*f);
        match cases::LIBFNS.iter().find(|g| g.name == f.name()) {
            None => {
                println!("UNCOVERED\t{}\t{lo}..={hi}\t(new in this tree; not generated)", f.name());
                problems += 1;
            }
            Some(g) if g.lib != Some((lo, hi)) => {
                println!("ARITY\t{}\ttree {lo}..={hi}, table {:?}", f.name(), g.lib);
                problems += 1;
            }
            Some(_) => {}
        }
    }
    for g in cases::LIBFNS {
        if LibFn::from_name(g.name).is_none() {
            println!("GONE\t{}\t(in the table, not in this tree)", g.name);
            problems += 1;
        }
    }
    println!(
        "libfn-check\t{}\ttree {} / table {}",
        if problems == 0 { "OK" } else { "MISMATCH" },
        LibFn::ALL.len(),
        cases::LIBFNS.len()
    );
}

fn main() {
    let args = parse_args();
    if args.list_ops {
        for o in &args.ops {
            println!("{o}");
        }
        return;
    }
    if args.list_fns {
        list_fns();
        return;
    }
    if args.info {
        println!("gen_version\t{}", cases::GEN_VERSION);
        println!("funs\t{}", cases::n_funs());
        println!("ops\t{}", ops::OPS.len());
        return;
    }
    alloc::set_limit_mb(args.mem_mb);
    install_panic_hook();

    let cases: Box<dyn Iterator<Item = (u64, String)>> = match (&args.file, args.seeds) {
        (Some(path), seeds) => {
            let text = std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
            let (lo, hi) = seeds.unwrap_or((0, u64::MAX));
            let v: Vec<(u64, String)> = text
                .lines()
                .enumerate()
                .map(|(i, l)| (i as u64 + 1, l.trim().to_string()))
                .filter(|(n, l)| *n >= lo && *n < hi && !l.is_empty() && !l.starts_with('#'))
                .collect();
            Box::new(v.into_iter())
        }
        (None, Some((lo, hi))) => Box::new((lo..hi).map(|id| (id, cases::case(id)))),
        (None, None) => die("give --seeds A..B or --file PATH"),
    };

    let timeout = Duration::from_millis(args.timeout_ms);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut first = true;
    for (id, src) in cases {
        let src: Arc<str> = Arc::from(src.as_str());
        for (k, op) in args.ops.iter().copied().enumerate() {
            if first && args.after.is_some_and(|a| k <= a) {
                continue;
            }
            if op == "expr" {
                emit(&mut out, id, op, "OK", &src);
                continue;
            }
            match run_op(id, op, &src, timeout) {
                Outcome::Ok(s) => emit(&mut out, id, op, "OK", &s),
                Outcome::Err(s) => emit(&mut out, id, op, "ERR", &s),
                Outcome::Panic(m) => {
                    let at = LAST_PANIC.lock().ok().and_then(|g| g.clone()).unwrap_or_default();
                    eprintln!("golden: PANIC case {id} op {op}: {at} | input: {src}");
                    emit(&mut out, id, op, "PANIC", &m);
                }
                Outcome::Timeout => {
                    emit(&mut out, id, op, "TIMEOUT", &format!(">{}ms", args.timeout_ms));
                    // The worker cannot be stopped; leave, the driver resumes
                    // with `--after op`.
                    exit(3);
                }
            }
        }
        first = false;
    }
}
