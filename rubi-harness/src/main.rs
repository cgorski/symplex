//! Runs symplex's `Ex::integrate` on the Rubi integration test suite and
//! judges every answer by differentiation.  See `README.md`.

mod check;
mod driver;
mod mac;
mod protocol;
mod report;
mod translate;
mod worker;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use crate::driver::Config;
use crate::protocol::Status;

const USAGE: &str = "\
usage: rubi-harness [options]

  --jobs N              parallel worker processes (default: CPUs - 2)
  --only SUBSTR         only files whose path (relative to the suite) contains
                        SUBSTR; repeatable
  --check               exit 1 if a file's verified count fell below ratchet.tsv
                        or its wrong+panic count rose above it
  --update              rewrite ratchet.tsv from this run (rows of files that
                        were not run are kept)
  --timeout SECS        integration limit per entry (default 10)
  --check-timeout SECS  verification limit per entry (default 30)
  --mem-limit-mb MB     resident memory limit per worker (default 4000)
  --chunk N             entries per worker process (default 25)
  --out DIR             output directory (default: results; results/only with
                        --only; results/selftest with --selftest)
  --suite DIR           suite root (default: <crate>/suite)
  --selftest            judge Rubi's optimal antiderivatives instead of
                        symplex's (validates translation and checker)
  --negative-params     also check every answer with parameters with the
                        parameter values negated; a mismatch there counts as
                        wrong (default output: results/negative)
  --scan                translate and parse only (no integration); print
                        statistics
  --probe EXPR [VAR]    integrate one Maxima expression in-process and show
                        the translation, the answer and the check; logs to
                        stderr under RUST_LOG (RUST_LOG=symplex::stage=debug
                        lists the stages that were slow or allocated much)
";

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Mode {
    Report,
    Check,
    Update,
}

struct Args {
    jobs: usize,
    only: Vec<String>,
    mode: Mode,
    timeout: u64,
    check_timeout: u64,
    mem_limit_mb: u64,
    chunk: usize,
    out: Option<PathBuf>,
    suite: PathBuf,
    selftest: bool,
    negative_params: bool,
    scan: bool,
}

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let cpus = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut a = Args {
        jobs: cpus.saturating_sub(2).max(1),
        only: Vec::new(),
        mode: Mode::Report,
        timeout: 10,
        check_timeout: 30,
        mem_limit_mb: 4000,
        chunk: 25,
        out: None,
        suite: crate_dir().join("suite"),
        selftest: false,
        negative_params: false,
        scan: false,
    };
    let mut it = argv.iter();
    fn value<'a>(
        it: &mut impl Iterator<Item = &'a String>,
        flag: &str,
    ) -> Result<&'a String, String> {
        it.next().ok_or_else(|| format!("{flag} needs a value"))
    }
    fn num<T: std::str::FromStr>(s: &str, flag: &str) -> Result<T, String> {
        s.parse().map_err(|_| format!("{flag}: not a number: {s}"))
    }
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--jobs" | "-j" => a.jobs = num(value(&mut it, arg)?, arg)?,
            "--only" => a.only.push(value(&mut it, arg)?.clone()),
            "--check" => a.mode = Mode::Check,
            "--update" => a.mode = Mode::Update,
            "--timeout" => a.timeout = num(value(&mut it, arg)?, arg)?,
            "--check-timeout" => a.check_timeout = num(value(&mut it, arg)?, arg)?,
            "--mem-limit-mb" => a.mem_limit_mb = num(value(&mut it, arg)?, arg)?,
            "--chunk" => a.chunk = num::<usize>(value(&mut it, arg)?, arg)?.max(1),
            "--out" => a.out = Some(PathBuf::from(value(&mut it, arg)?)),
            "--suite" => a.suite = PathBuf::from(value(&mut it, arg)?),
            "--selftest" => a.selftest = true,
            "--negative-params" => a.negative_params = true,
            "--scan" => a.scan = true,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if a.selftest && a.mode != Mode::Report {
        return Err("--selftest cannot be combined with --check/--update".into());
    }
    if a.negative_params && a.mode == Mode::Update {
        return Err("--negative-params cannot be combined with --update".into());
    }
    Ok(a)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("--worker") {
        return worker_main(&argv[1..]);
    }
    if argv.first().map(String::as_str) == Some("--probe") {
        return probe(&argv[1..]);
    }
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}\n");
            }
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn worker_main(argv: &[String]) -> ExitCode {
    let (Some(file), Some(start), Some(end)) = (argv.first(), argv.get(1), argv.get(2)) else {
        eprintln!("usage: rubi-harness --worker FILE START END [--selftest] [--negative-params]");
        return ExitCode::from(2);
    };
    let (Ok(start), Ok(end)) = (start.parse(), end.parse()) else {
        eprintln!("worker: bad range");
        return ExitCode::from(2);
    };
    let selftest = argv.iter().any(|a| a == "--selftest");
    let negative_params = argv.iter().any(|a| a == "--negative-params");
    ExitCode::from(worker::main(Path::new(file), start, end, selftest, negative_params) as u8)
}

/// `--probe EXPR [VAR]`: the whole pipeline for one integrand, verbosely.
fn probe(argv: &[String]) -> ExitCode {
    use symplex::prelude::Context;
    let Some(src) = argv.first() else {
        eprintln!("usage: rubi-harness --probe EXPR [VAR]");
        return ExitCode::from(2);
    };
    let var = argv.get(1).map_or("x", String::as_str);
    if std::env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .with_ansi(false)
            .init();
    }
    let tr = match translate::translate(src) {
        Ok(t) => t,
        Err(u) => {
            println!("unsupported: {} {}", u.reason, u.detail);
            return ExitCode::from(1);
        }
    };
    println!("translated: {}", tr.text);
    let ctx = Context::new();
    let f = match ctx.parse(&tr.text) {
        Ok(f) => f,
        Err(e) => {
            println!("symplex parse error: {e}");
            return ExitCode::from(1);
        }
    };
    println!("parsed:     {f}");
    let x = ctx.symbol(&translate::rename(var));
    let t0 = Instant::now();
    let big_f = f.integrate(&x);
    println!("F ({:.3} s): {big_f}", t0.elapsed().as_secs_f64());
    println!("unevaluated: {}", big_f.has_unevaluated());
    let mut env = check::Env::new();
    env.extend(&ctx, &x, &[&f, &big_f]);
    println!("params:     {}", env.desc);
    let reports = check::check(&ctx, &f, &big_f, &x, &env, src.contains("%i"));
    for r in &reports {
        println!("  {}", r.describe(var));
    }
    if !big_f.has_unevaluated() {
        println!(
            "verdict:    {}",
            check::verdict(&reports, src.contains("%i"))
        );
    }
    if !env.is_empty() {
        let mut env_neg = check::Env::negated();
        env_neg.extend(&ctx, &x, &[&f, &big_f]);
        println!("params:     {} (negated set)", env_neg.desc);
        let reports_neg = check::check(&ctx, &f, &big_f, &x, &env_neg, src.contains("%i"));
        for r in &reports_neg {
            println!("  {}", r.describe(var));
        }
        if !big_f.has_unevaluated() {
            println!(
                "verdict:    {} (negated set)",
                check::verdict(&reports_neg, src.contains("%i"))
            );
        }
    }
    ExitCode::SUCCESS
}

/// The symplex version, read from the parent crate's manifest.
fn symplex_version() -> String {
    std::fs::read_to_string(crate_dir().join("../Cargo.toml"))
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("version"))
                .and_then(|l| l.split('"').nth(1).map(str::to_string))
        })
        .unwrap_or_else(|| "?".into())
}

fn run(args: &Args) -> Result<ExitCode, String> {
    let mut files = driver::discover(&args.suite)?;
    if files.is_empty() {
        return Err(format!("no .mac files under {}", args.suite.display()));
    }
    let full_suite = args.only.is_empty();
    if !full_suite {
        files.retain(|(rel, _)| args.only.iter().any(|s| rel.contains(s.as_str())));
        if files.is_empty() {
            return Err("--only matched no file".into());
        }
    }
    let mut files = driver::load(files);
    for f in &files {
        if let Some(e) = &f.error {
            eprintln!("warning: {e}");
        }
    }
    if args.scan {
        scan(&files);
        return Ok(ExitCode::SUCCESS);
    }

    let cfg = Config {
        jobs: args.jobs,
        timeout: Duration::from_secs(args.timeout),
        check_timeout: Duration::from_secs(args.check_timeout),
        chunk: args.chunk,
        mem_limit_mb: args.mem_limit_mb,
        selftest: args.selftest,
        negative_params: args.negative_params,
    };
    let n_entries: usize = files.iter().map(|f| f.entries.len()).sum();
    eprintln!(
        "rubi-harness: {} files, {n_entries} entries, {} jobs, timeout {} s (check {} s){}",
        files.len(),
        cfg.jobs,
        args.timeout,
        args.check_timeout,
        if cfg.selftest {
            ", SELFTEST (judging Rubi's answers)"
        } else {
            ""
        }
    );
    let t0 = Instant::now();
    driver::run(&mut files, &cfg);
    let wall = t0.elapsed().as_secs_f64();

    let out = args.out.clone().unwrap_or_else(|| {
        let base = crate_dir().join("results");
        if args.selftest {
            base.join("selftest")
        } else if args.negative_params {
            base.join("negative")
        } else if full_suite {
            base
        } else {
            base.join("only")
        }
    });
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let note = format!(
        "rubi-harness, symplex {}: {} files, {n_entries} entries, {} jobs, timeout {} s, check timeout {} s, wall {wall:.0} s{}",
        symplex_version(),
        files.len(),
        cfg.jobs,
        args.timeout,
        args.check_timeout,
        if cfg.selftest {
            ", selftest (Rubi's optimal antiderivatives judged)"
        } else {
            ""
        }
    ) + if cfg.negative_params {
        ", every answer also checked with the parameter values negated"
    } else {
        ""
    };
    let io = |r: std::io::Result<()>, what: &str| r.map_err(|e| format!("{what}: {e}"));
    io(
        report::write_summary(&out.join("summary.tsv"), &files, &note),
        "summary.tsv",
    )?;
    io(
        report::write_wrong(&out.join("wrong.txt"), &files, &note),
        "wrong.txt",
    )?;
    io(
        report::write_real_verified(&out.join("real_verified.txt"), &files, &note),
        "real_verified.txt",
    )?;
    io(
        report::write_entries(&out.join("entries.tsv"), &files),
        "entries.tsv",
    )?;
    io(
        report::write_undecided(&out.join("undecided.txt"), &files, &note),
        "undecided.txt",
    )?;
    io(
        report::write_reasons(&out.join("reasons.tsv"), &files),
        "reasons.tsv",
    )?;

    println!("{}", report::chapter_table(&files));
    let (_, all) = report::chapter_totals(&files);
    println!(
        "wall {wall:.0} s with {} jobs; {} REAL_VERIFIED, {} WRONG, {} PANIC; results in {}",
        cfg.jobs,
        all.get(Status::RealVerified),
        all.get(Status::Wrong),
        all.get(Status::Panic),
        out.display()
    );
    println!("\nmost common reasons for `unsupported`:");
    for (reason, n) in report::reason_counts(&files, Status::Unsupported)
        .iter()
        .take(15)
    {
        println!("  {n:>6}  {reason}");
    }

    let ratchet_path = crate_dir().join("ratchet.tsv");
    match args.mode {
        Mode::Report => Ok(ExitCode::SUCCESS),
        Mode::Update => {
            let mut r = report::read_ratchet(&ratchet_path)?;
            r.extend(report::current_ratchet(&files));
            io(report::write_ratchet(&ratchet_path, &r), "ratchet.tsv")?;
            println!("updated {}", ratchet_path.display());
            Ok(ExitCode::SUCCESS)
        }
        Mode::Check => {
            let old = report::read_ratchet(&ratchet_path)?;
            let (violations, notes) = report::check_ratchet(&old, &files);
            for n in &notes {
                println!("note: {n}");
            }
            if violations.is_empty() {
                println!("ratchet: OK ({} files checked)", files.len());
                Ok(ExitCode::SUCCESS)
            } else {
                for v in &violations {
                    println!("RATCHET VIOLATION: {v}");
                }
                Ok(ExitCode::from(1))
            }
        }
    }
}

/// `--scan`: translation statistics, and a symplex parse (no integration)
/// of every translated integrand.
fn scan(files: &[driver::FileRun]) {
    use symplex::prelude::Context;
    let mut reasons: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let mut opt_reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut symbols: BTreeMap<String, usize> = BTreeMap::new();
    let mut vars: BTreeMap<String, usize> = BTreeMap::new();
    let mut fields: BTreeMap<usize, usize> = BTreeMap::new();
    let (mut total, mut ok) = (0, 0);
    for f in files {
        for e in &f.entries {
            total += 1;
            *fields.entry(e.fields.len()).or_insert(0) += 1;
            *vars.entry(e.variable().to_string()).or_insert(0) += 1;
            let mut fail = |key: String, sample: String| {
                let slot = reasons.entry(key).or_insert((0, sample));
                slot.0 += 1;
            };
            match translate::translate(e.integrand()) {
                Err(u) => fail(u.reason, format!("{}: {}", f.rel, e.integrand())),
                Ok(t) => {
                    for s in &t.symbols {
                        *symbols.entry(s.clone()).or_insert(0) += 1;
                    }
                    let ctx = Context::new();
                    match ctx.parse(&t.text) {
                        Ok(_) => ok += 1,
                        Err(err) => fail(
                            "symplex parse error".into(),
                            format!("{}: {} => {} ({err})", f.rel, e.integrand(), t.text),
                        ),
                    }
                }
            }
            if let Some(o) = e.optimal() {
                let key = match translate::translate(o) {
                    Ok(t) => match Context::new().parse(&t.text) {
                        Ok(_) => "ok".to_string(),
                        Err(_) => "symplex parse error".into(),
                    },
                    Err(u) => u.reason,
                };
                *opt_reasons.entry(key).or_insert(0) += 1;
            }
        }
    }
    println!("{total} entries, {ok} integrands translate and parse");
    println!("fields per entry: {fields:?}");
    println!("variables: {vars:?}");
    println!("\nintegrand symbols (renamed):");
    for (s, n) in &symbols {
        println!("  {n:>6}  {s}");
    }
    println!("\nintegrand unsupported reasons:");
    let mut rv: Vec<_> = reasons.into_iter().collect();
    rv.sort_by_key(|a| std::cmp::Reverse(a.1.0));
    for (r, (n, sample)) in rv {
        println!("  {n:>6}  {r}\n          e.g. {sample}");
    }
    println!("\noptimal antiderivatives:");
    let mut ov: Vec<_> = opt_reasons.into_iter().collect();
    ov.sort_by_key(|a| std::cmp::Reverse(a.1));
    for (r, n) in ov {
        println!("  {n:>6}  {r}");
    }
}
