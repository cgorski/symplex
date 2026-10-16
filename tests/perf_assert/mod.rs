//! Performance assertions for tests: the CPU time of the running thread
//! against a budget scaled by `SYMPLEX_TEST_TIME_SCALE` (session 15, M4).
//!
//! A wall-clock bound (`assert!(t.elapsed() < Duration::from_secs(2))`)
//! measures the machine as much as the code: on a loaded machine the test
//! thread waits for a core, and the same work takes twice as long.  CI
//! failed once on 2.02 s against a 2 s bound, and `v32_series` took 3.76 s
//! against 3 s while 56 other processes ran.  The CPU time of the thread
//! (`clock_gettime(CLOCK_THREAD_CPUTIME_ID)`) counts only the cycles spent
//! on the test's own work, so a budget keeps its meaning — "this call does
//! not take the exponential path" — at any load.  symplex spawns no
//! threads, so the thread that calls the library is the thread that works.
//!
//! | item | use |
//! |---|---|
//! | `assert_fast!(budget, expr)` | evaluates `expr`, fails if it took more than `budget × scale` of CPU time, returns its value |
//! | `CpuTimer::start()`, `.assert_within(budget, what)` | the same when the timed region spans several statements |
//! | `cpu_time(f) -> (T, Duration)`, `CpuTimer::cpu()` | relative comparisons: two workloads timed in interleaved rounds, a cold call against a warm one |
//! | `time_scale()`, `scaled(budget)` | the factor from `SYMPLEX_TEST_TIME_SCALE` (default 1) and a budget multiplied by it |
//!
//! Budgets are for a debug build on a developer machine.  A slower machine
//! (a CI runner, an emulator, a sanitizer build) sets
//! `SYMPLEX_TEST_TIME_SCALE` — `2` multiplies every budget by two; relative
//! comparisons are not scaled.
//!
//! CPU time does not count waiting for a core, but it does grow while the
//! core's other hyperthread is busy and the caches are shared: on a
//! 32-core, 64-thread Xeon a series that takes under 0.9 s alone (`cargo
//! nextest run -j 1`) took 2.3 s of CPU time in a full parallel run of the
//! suite, and 1.8 s of `is_positive` took 3.9 s.  That factor is bounded —
//! once every hardware thread is busy, more load adds only waiting — while
//! wall-clock time grows with the load without limit.  So a budget of 5×
//! the CPU time measured alone holds at any load.
//!
//! Not covered:
//!
//! - Work done on another thread: measure inside that thread (a `CpuTimer`
//!   is `!Send`, so it cannot be started on one thread and read on another).
//! - Hang guards (a worker thread and `recv_timeout`): they ask whether a
//!   call returns at all, so they stay wall-clock; keep them generous.
//! - Targets without a per-thread CPU clock (anything but Linux, Android,
//!   macOS, iOS and FreeBSD): the timer falls back to the wall clock, and
//!   a failure message says so.
//!
//! The ratchet `tests/unit/test_timing_asserts.rs` keeps raw wall-clock
//! reads (`Instant::now()`, `SystemTime::now()`, `.elapsed()`) out of the
//! other test files.

#![allow(dead_code, unused_imports, unused_macros)]

use std::fmt;
use std::marker::PhantomData;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// The environment variable whose value multiplies every budget.
pub const SCALE_VAR: &str = "SYMPLEX_TEST_TIME_SCALE";

/// The scale given by a value of [`SCALE_VAR`]: `1` when unset or blank,
/// otherwise a finite positive number (`"2"`, `"1.5"`).
pub fn parse_time_scale(value: Option<&str>) -> Result<f64, String> {
    let Some(text) = value.map(str::trim).filter(|t| !t.is_empty()) else {
        return Ok(1.0);
    };
    match text.parse::<f64>() {
        Ok(scale) if scale.is_finite() && scale > 0.0 => Ok(scale),
        _ => Err(format!(
            "{SCALE_VAR} must be a finite positive number, got {text:?}"
        )),
    }
}

/// The factor every budget is multiplied by, read once from
/// [`SCALE_VAR`].  A malformed value fails the test that reads it.
pub fn time_scale() -> f64 {
    static SCALE: OnceLock<Result<f64, String>> = OnceLock::new();
    let scale = SCALE.get_or_init(|| match std::env::var(SCALE_VAR) {
        Ok(value) => parse_time_scale(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(1.0),
        Err(std::env::VarError::NotUnicode(raw)) => {
            Err(format!("{SCALE_VAR} is not valid Unicode: {raw:?}"))
        }
    });
    match scale {
        Ok(scale) => *scale,
        Err(e) => panic!("{e}"),
    }
}

/// `budget × time_scale()` (saturating at `Duration::MAX`).
pub fn scaled(budget: Duration) -> Duration {
    Duration::try_from_secs_f64(budget.as_secs_f64() * time_scale()).unwrap_or(Duration::MAX)
}

/// The CPU time the calling thread has used so far, or `None` where the
/// platform has no per-thread CPU clock.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd"
))]
fn thread_cpu_now() -> Option<Duration> {
    let mut ts = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: `ts` points to writable storage for one `timespec`, which
    // `clock_gettime` fills when it returns 0.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, ts.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: `clock_gettime` returned 0, so it initialised `ts`.
    let ts = unsafe { ts.assume_init() };
    let secs = u64::try_from(ts.tv_sec).ok()?;
    let nanos = u32::try_from(ts.tv_nsec).ok()?;
    Some(Duration::new(secs, nanos))
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd"
)))]
fn thread_cpu_now() -> Option<Duration> {
    None
}

/// Whether this platform has a per-thread CPU clock (otherwise every
/// measurement here is wall-clock time).
pub fn has_thread_cpu_clock() -> bool {
    thread_cpu_now().is_some()
}

/// A stopwatch of the CPU time of the thread that started it.
///
/// `!Send`: the clock it reads belongs to one thread.
pub struct CpuTimer {
    cpu_start: Option<Duration>,
    wall_start: Instant,
    _same_thread: PhantomData<*const ()>,
}

impl CpuTimer {
    /// Starts measuring the calling thread.
    pub fn start() -> Self {
        CpuTimer {
            cpu_start: thread_cpu_now(),
            wall_start: Instant::now(),
            _same_thread: PhantomData,
        }
    }

    /// The thread's CPU time since [`CpuTimer::start`] (the wall-clock time
    /// where the platform has no per-thread CPU clock).
    pub fn cpu(&self) -> Duration {
        match (self.cpu_start, thread_cpu_now()) {
            (Some(start), Some(now)) => now.saturating_sub(start),
            _ => self.wall_start.elapsed(),
        }
    }

    fn clock_name(&self) -> &'static str {
        if self.cpu_start.is_some() {
            "thread CPU time"
        } else {
            "wall-clock time (no per-thread CPU clock on this platform)"
        }
    }

    /// Fails the test if the CPU time since the start exceeds
    /// `budget × time_scale()`; `what` names the work in the message.
    #[track_caller]
    pub fn assert_within(&self, budget: Duration, what: impl fmt::Display) {
        let cpu = self.cpu();
        let limit = scaled(budget);
        if cpu > limit {
            panic!(
                "too slow: {what} took {cpu:.3?} of {} (wall {:.3?}); budget {budget:?} × \
                 {SCALE_VAR} {} = {limit:.3?}",
                self.clock_name(),
                self.wall_start.elapsed(),
                time_scale(),
            );
        }
    }
}

/// The value of `f()` and the CPU time the calling thread spent in it.
pub fn cpu_time<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let timer = CpuTimer::start();
    let value = f();
    (value, timer.cpu())
}

/// `assert_fast!(budget, expr)` evaluates `expr`, fails the test if that
/// took more than `budget × SYMPLEX_TEST_TIME_SCALE` of the thread's CPU
/// time, and returns the value.  An optional format string and arguments
/// name the work in the failure message (default: the expression).
///
/// ```ignore
/// let r = assert_fast!(Duration::from_secs(2), e.series(&x, &zero, 6));
/// assert_fast!(Duration::from_secs(1), f.integrate(&x), "∫ {f}");
/// ```
macro_rules! assert_fast {
    ($budget:expr, $body:expr $(,)?) => {{
        let timer = $crate::perf_assert::CpuTimer::start();
        let value = $body;
        timer.assert_within($budget, concat!("`", stringify!($body), "`"));
        value
    }};
    ($budget:expr, $body:expr, $($what:tt)+) => {{
        let timer = $crate::perf_assert::CpuTimer::start();
        let value = $body;
        timer.assert_within($budget, format_args!($($what)+));
        value
    }};
}
pub(crate) use assert_fast;
