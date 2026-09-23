//! The driver: runs the suite over worker processes and collects results.
//!
//! Files are split into chunks of consecutive entries; a fixed pool of
//! threads takes chunks from a queue and runs each in a child process of
//! this binary (`--worker`).  The thread reads the child's records and
//! enforces the per-entry limits: `timeout` from `BEGIN` to `PHASE check`,
//! `check_timeout` from there to `END`, and a resident-memory cap.  On a
//! violation it kills the child, classifies the entry, and starts a new
//! child at the next entry.  A child that dies (stack overflow, abort) is
//! handled the same way, with the entry reported as a panic.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::mac::{self, Entry};
use crate::protocol::{Status, unescape};

/// Run configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub jobs: usize,
    pub timeout: Duration,
    pub check_timeout: Duration,
    pub chunk: usize,
    pub mem_limit_mb: u64,
    pub selftest: bool,
}

/// One entry's result.
#[derive(Debug, Clone)]
pub struct EntryResult {
    pub status: Status,
    /// Details (`reason`, `detail`, `F`, `points`, `rubi`, `panic`, …).
    pub kv: BTreeMap<String, String>,
    pub secs: f64,
}

impl EntryResult {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.kv.get(key).map(String::as_str)
    }
}

/// A suite file and its results.
#[derive(Debug)]
pub struct FileRun {
    /// Path relative to the suite root, `/`-separated.
    pub rel: String,
    pub abs: PathBuf,
    pub entries: Vec<Entry>,
    /// Set if the file could not be read.
    pub error: Option<String>,
    pub results: Vec<Option<EntryResult>>,
    /// Sum of the wall-clock time of this file's chunks.
    pub secs: f64,
}

impl FileRun {
    pub fn count(&self, st: Status) -> usize {
        self.results
            .iter()
            .filter(|r| r.as_ref().is_some_and(|r| r.status == st))
            .count()
    }

    pub fn chapter(&self) -> &str {
        self.rel.split('/').next().unwrap_or("")
    }
}

/// All `.mac` files under `root`, sorted by relative path.
pub fn discover(root: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for ent in rd {
            let ent = ent.map_err(|e| e.to_string())?;
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "mac") {
                let rel = p
                    .strip_prefix(root)
                    .map_err(|e| e.to_string())?
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push((rel, p));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Read the files (in the driver, for entry counts and report texts).
pub fn load(files: Vec<(String, PathBuf)>) -> Vec<FileRun> {
    files
        .into_iter()
        .map(|(rel, abs)| {
            let (entries, error) = match mac::read_file(&abs) {
                Ok(e) => (e, None),
                Err(e) => (Vec::new(), Some(e)),
            };
            let n = entries.len();
            FileRun {
                rel,
                abs,
                entries,
                error,
                results: vec![None; n],
                secs: 0.0,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Task {
    file: usize,
    start: usize,
    end: usize,
}

/// Run every entry of `files`, filling in their results.
pub fn run(files: &mut [FileRun], cfg: &Config) {
    let exe = std::env::current_exe().expect("cannot locate the harness executable");
    let mut tasks: Vec<Task> = Vec::new();
    for (fi, f) in files.iter().enumerate() {
        let n = f.entries.len();
        let mut s = 0;
        while s < n {
            let e = (s + cfg.chunk).min(n);
            tasks.push(Task {
                file: fi,
                start: s,
                end: e,
            });
            s = e;
        }
    }
    // Largest files first: their chunks spread over all threads early.
    tasks.sort_by_key(|t| {
        (
            std::cmp::Reverse(files[t.file].entries.len()),
            t.file,
            t.start,
        )
    });
    let chunks_left: Vec<AtomicUsize> = files
        .iter()
        .map(|f| AtomicUsize::new(f.entries.len().div_ceil(cfg.chunk.max(1))))
        .collect();
    let total_files = files.iter().filter(|f| !f.entries.is_empty()).count();
    let files_done = AtomicUsize::new(0);
    let queue = Mutex::new(tasks.into_iter().rev().collect::<Vec<_>>());
    let shared: Vec<Mutex<(Vec<Option<EntryResult>>, f64)>> = files
        .iter()
        .map(|f| Mutex::new((vec![None; f.entries.len()], 0.0)))
        .collect();
    let started = Instant::now();
    let meta: Vec<(String, PathBuf, usize)> = files
        .iter()
        .map(|f| (f.rel.clone(), f.abs.clone(), f.entries.len()))
        .collect();

    thread::scope(|scope| {
        for _ in 0..cfg.jobs.max(1) {
            scope.spawn(|| {
                loop {
                    let Some(task) = queue.lock().unwrap().pop() else {
                        break;
                    };
                    let (rel, abs, _) = &meta[task.file];
                    let t0 = Instant::now();
                    let res = run_chunk(&exe, abs, task.start, task.end, cfg);
                    let secs = t0.elapsed().as_secs_f64();
                    let mut slot = shared[task.file].lock().unwrap();
                    for (idx, r) in res {
                        if idx < slot.0.len() {
                            slot.0[idx] = Some(r);
                        }
                    }
                    slot.1 += secs;
                    if chunks_left[task.file].fetch_sub(1, Ordering::SeqCst) == 1 {
                        let done = files_done.fetch_add(1, Ordering::SeqCst) + 1;
                        let count = |st: Status| {
                            slot.0
                                .iter()
                                .filter(|r| r.as_ref().is_some_and(|r| r.status == st))
                                .count()
                        };
                        eprintln!(
                            "[{done:>3}/{total_files} {:>6.0}s] {rel}: {} entries, {} verified, {} real_verified, {} wrong, {} panic, {} timeout, {} unevaluated, {} unsupported ({:.0}s)",
                            started.elapsed().as_secs_f64(),
                            slot.0.len(),
                            count(Status::Verified),
                            count(Status::RealVerified),
                            count(Status::Wrong),
                            count(Status::Panic),
                            count(Status::Timeout),
                            count(Status::Unevaluated),
                            count(Status::Unsupported),
                            slot.1,
                        );
                    }
                }
            });
        }
    });

    for (f, s) in files.iter_mut().zip(shared) {
        let (results, secs) = s.into_inner().unwrap();
        f.results = results;
        f.secs = secs;
    }
}

/// The entry currently running in a worker.
struct Inflight {
    idx: usize,
    in_check: bool,
    started: Instant,
    deadline: Instant,
    kv: BTreeMap<String, String>,
}

enum Interrupt {
    Timeout,
    Memory(u64),
    Died(String),
}

/// Grace period for a worker to start and read its file.
const IDLE_LIMIT: Duration = Duration::from_secs(120);
/// Entries running longer than this have their memory watched.
const MEM_WATCH_AFTER: Duration = Duration::from_secs(2);

/// Run entries `start..end` of `file`, restarting the worker after every
/// timeout or crash.
fn run_chunk(
    exe: &Path,
    file: &Path,
    start: usize,
    end: usize,
    cfg: &Config,
) -> Vec<(usize, EntryResult)> {
    let mut results = Vec::new();
    let mut next = start;
    while next < end {
        let mut cmd = Command::new(exe);
        cmd.arg("--worker")
            .arg(file)
            .arg(next.to_string())
            .arg(end.to_string());
        if cfg.selftest {
            cmd.arg("--selftest");
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                for idx in next..end {
                    results.push((
                        idx,
                        failure(Status::Panic, "could not start worker", &e.to_string()),
                    ));
                }
                break;
            }
        };
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let (tx, rx) = mpsc::channel::<String>();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let err_tail = Arc::new(Mutex::new(String::new()));
        let err_reader = {
            let tail = Arc::clone(&err_tail);
            thread::spawn(move || drain_tail(stderr, &tail))
        };

        let first = next;
        let mut inflight: Option<Inflight> = None;
        let mut idle_since = Instant::now();
        let mut last_mem_check = Instant::now();
        let interrupt: Option<Interrupt> = loop {
            let now = Instant::now();
            let deadline = inflight
                .as_ref()
                .map_or(idle_since + IDLE_LIMIT, |f| f.deadline);
            let line = if now >= deadline {
                // Past the deadline: take what has already arrived, then stop.
                match rx.try_recv() {
                    Ok(l) => Ok(l),
                    Err(TryRecvError::Empty) => break Some(Interrupt::Timeout),
                    Err(TryRecvError::Disconnected) => Err(RecvTimeoutError::Disconnected),
                }
            } else {
                rx.recv_timeout((deadline - now).min(Duration::from_millis(500)))
            };
            match line {
                Ok(l) => {
                    if let Some(done) = handle_line(&l, &mut inflight, cfg) {
                        next = done.0 + 1;
                        results.push(done);
                        idle_since = Instant::now();
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if let Some(f) = &inflight
                        && f.started.elapsed() > MEM_WATCH_AFTER
                        && last_mem_check.elapsed() > Duration::from_secs(1)
                    {
                        last_mem_check = Instant::now();
                        if let Some(mb) = rss_mb(child.id())
                            && mb > cfg.mem_limit_mb
                        {
                            break Some(Interrupt::Memory(mb));
                        }
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break None,
            }
        };

        let status = match interrupt {
            Some(_) => kill(&mut child),
            None => child.wait().ok(),
        };
        let _ = reader.join();
        let _ = err_reader.join();
        let tail = err_tail.lock().map(|t| t.clone()).unwrap_or_default();
        let interrupt = interrupt.unwrap_or_else(|| Interrupt::Died(describe_exit(status, &tail)));

        match inflight.take() {
            Some(f) => {
                let idx = f.idx;
                results.push((idx, interrupted(f, &interrupt, cfg)));
                next = idx + 1;
            }
            // The worker stopped between entries without making any progress:
            // blame the next entry so that the loop terminates.  (After some
            // progress, the loop simply starts a new worker at `next`.)
            None if next < end && next == first => {
                let why = match &interrupt {
                    Interrupt::Died(d) => d.clone(),
                    Interrupt::Timeout => "worker idle too long".into(),
                    Interrupt::Memory(mb) => format!("worker used {mb} MB"),
                };
                results.push((
                    next,
                    failure(Status::Panic, "worker failed before the entry", &why),
                ));
                next += 1;
            }
            None => {}
        }
    }
    results
}

/// Apply one protocol line; returns a finished entry.
fn handle_line(
    line: &str,
    inflight: &mut Option<Inflight>,
    cfg: &Config,
) -> Option<(usize, EntryResult)> {
    let mut parts = line.splitn(4, '\t');
    let tag = parts.next()?;
    let idx: usize = parts.next()?.parse().ok()?;
    let now = Instant::now();
    match tag {
        "BEGIN" => {
            *inflight = Some(Inflight {
                idx,
                in_check: false,
                started: now,
                deadline: now + cfg.timeout,
                kv: BTreeMap::new(),
            });
        }
        "PHASE" => {
            if let Some(f) = inflight.as_mut().filter(|f| f.idx == idx) {
                f.in_check = true;
                f.deadline = now + cfg.check_timeout;
            }
        }
        "KV" => {
            let key = parts.next()?;
            let value = unescape(parts.next().unwrap_or(""));
            if let Some(f) = inflight.as_mut().filter(|f| f.idx == idx) {
                f.kv.insert(key.to_string(), value);
            }
        }
        "END" => {
            let status = Status::parse(parts.next()?)?;
            let f = inflight.take().filter(|f| f.idx == idx)?;
            let mut kv = f.kv;
            kv.remove("stage");
            return Some((
                idx,
                EntryResult {
                    status,
                    kv,
                    secs: f.started.elapsed().as_secs_f64(),
                },
            ));
        }
        _ => {}
    }
    None
}

/// Classify an entry whose worker was killed or died.
fn interrupted(f: Inflight, why: &Interrupt, cfg: &Config) -> EntryResult {
    let secs = f.started.elapsed().as_secs_f64();
    let mut kv = f.kv;
    let stage = kv.remove("stage").unwrap_or_else(|| "startup".into());
    let what = match why {
        Interrupt::Timeout => format!(
            "exceeded {} s",
            if f.in_check {
                cfg.check_timeout.as_secs()
            } else {
                cfg.timeout.as_secs()
            }
        ),
        Interrupt::Memory(mb) => format!(
            "exceeded the memory limit ({mb} MB > {} MB)",
            cfg.mem_limit_mb
        ),
        Interrupt::Died(d) => format!("worker died: {d}"),
    };
    // A verdict streamed before the interruption stands.
    let verdict = kv
        .get("verdict")
        .and_then(|v| Status::parse(v))
        .filter(|st| matches!(st, Status::Wrong | Status::RealVerified));
    let status = if let Some(st) = verdict {
        kv.insert(
            "note".into(),
            format!("details incomplete: {what} during {stage}"),
        );
        st
    } else {
        match why {
            Interrupt::Died(_) => {
                kv.insert(
                    "reason".into(),
                    format!("worker process died during {stage}"),
                );
                kv.insert("panic".into(), what);
                Status::Panic
            }
            _ if f.in_check => {
                kv.insert("reason".into(), format!("check {what}"));
                Status::Undecided
            }
            _ => {
                kv.insert("reason".into(), format!("{stage} {what}"));
                Status::Timeout
            }
        }
    };
    EntryResult { status, kv, secs }
}

fn failure(status: Status, reason: &str, detail: &str) -> EntryResult {
    let mut kv = BTreeMap::new();
    kv.insert("reason".to_string(), reason.to_string());
    kv.insert("panic".to_string(), detail.to_string());
    EntryResult {
        status,
        kv,
        secs: 0.0,
    }
}

fn kill(child: &mut Child) -> Option<ExitStatus> {
    let _ = child.kill();
    child.wait().ok()
}

fn describe_exit(status: Option<ExitStatus>, tail: &str) -> String {
    let st = match status {
        Some(s) => {
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                match (s.code(), s.signal()) {
                    (Some(c), _) => format!("exit code {c}"),
                    (None, Some(sig)) => format!("signal {sig}"),
                    _ => s.to_string(),
                }
            }
            #[cfg(not(unix))]
            {
                s.to_string()
            }
        }
        None => "unknown exit status".into(),
    };
    let tail = tail.trim();
    if tail.is_empty() {
        st
    } else {
        format!("{st}; stderr: {tail}")
    }
}

/// Keep the last few KB of a stream.
fn drain_tail(mut r: impl Read, tail: &Mutex<String>) {
    const KEEP: usize = 4000;
    let mut buf = [0u8; 4096];
    loop {
        match r.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let mut t = tail.lock().unwrap();
                t.push_str(&String::from_utf8_lossy(&buf[..n]));
                if t.len() > 2 * KEEP {
                    let mut cut = t.len() - KEEP;
                    while !t.is_char_boundary(cut) {
                        cut += 1;
                    }
                    t.drain(..cut);
                }
            }
        }
    }
}

/// Resident set size of `pid` in MB, via `ps`.
fn rss_mb(pid: u32) -> Option<u64> {
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let kb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kb / 1024)
}
