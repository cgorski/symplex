//! Tests of session 15 (s15m4): robust performance assertions.
//! Every test says what was wrong before and cites its oracle.
//!
//! The helper is `tests/perf_assert/mod.rs`; its ratchet is
//! `tests/unit/test_timing_asserts.rs`.

#![allow(unused_imports)]
use std::hint::black_box;
use std::time::Duration;

use symplex::prelude::*;

use crate::perf_assert::{
    CpuTimer, assert_fast, cpu_time, has_thread_cpu_clock, parse_time_scale, scaled, time_scale,
};

/// `n` rounds of integer arithmetic the optimiser cannot remove.
fn busy(n: u64) -> u64 {
    let mut acc = 0x9E37_79B9_7F4A_7C15_u64;
    for i in 0..n {
        acc = black_box(acc.rotate_left(5) ^ i).wrapping_mul(0x0100_0000_01B3);
    }
    acc
}

/// A thread that waits costs no CPU time.  Before, every budget was a
/// wall-clock bound (`assert!(t.elapsed() < bound)`), which counts the time
/// a thread waits — for a core on a loaded machine, or here in `sleep`, which
/// returns after *at least* the requested 300 ms (`std::thread::sleep`
/// docs).  Oracle: POSIX `CLOCK_THREAD_CPUTIME_ID` advances only while the
/// thread runs, so the sleep measures a few microseconds.
#[test]
fn sleeping_costs_no_cpu_time() {
    if !has_thread_cpu_clock() {
        return;
    }
    let ((), cpu) = cpu_time(|| std::thread::sleep(Duration::from_millis(300)));
    assert!(cpu < Duration::from_millis(50), "{cpu:?}");
}

/// Work on another thread is invisible to the calling thread's clock, so a
/// timed region must run on the thread that measures it (a `CpuTimer` is
/// `!Send`, below).  Oracle: the same busy loop measured inside the worker
/// takes at least ten times what the joining thread spends.
#[test]
fn work_on_another_thread_is_measured_there() {
    if !has_thread_cpu_clock() {
        return;
    }
    let mut worker = Duration::ZERO;
    let ((), joining) = cpu_time(|| {
        worker = std::thread::spawn(|| cpu_time(|| busy(20_000_000)).1)
            .join()
            .expect("worker thread");
    });
    assert!(
        joining * 10 < worker,
        "joining {joining:?}, worker {worker:?}"
    );
}

/// CPU-bound work is measured, and `assert_fast!` passes its value through.
/// Oracle: the busy loop is deterministic (two runs agree) and takes
/// nonzero time.
#[test]
fn assert_fast_returns_the_value_and_measures_work() {
    let v = assert_fast!(Duration::from_secs(20), busy(2_000_000));
    assert_eq!(v, busy(2_000_000));
    let (w, cpu) = cpu_time(|| busy(2_000_000));
    assert_eq!(w, v);
    assert!(cpu > Duration::ZERO);
    let timer = CpuTimer::start();
    let _ = busy(1_000_000);
    assert!(timer.cpu() > Duration::ZERO);
    timer.assert_within(Duration::from_secs(20), "a million rounds");
}

/// A call over its budget fails the test, naming the work, the clock and
/// the scale.  Oracle: a million rounds take far more than a nanosecond at
/// any scale a CI machine sets.
#[test]
#[should_panic(expected = "too slow: a million rounds took")]
fn over_budget_fails_with_the_measurement() {
    let timer = CpuTimer::start();
    let _ = busy(1_000_000);
    timer.assert_within(Duration::from_nanos(1), "a million rounds");
}

/// The same through the macro, whose default message is the expression.
#[test]
#[should_panic(expected = "too slow: `busy(1_000_000)` took")]
fn assert_fast_over_budget_names_the_expression() {
    let _ = assert_fast!(Duration::from_nanos(1), busy(1_000_000));
}

/// `SYMPLEX_TEST_TIME_SCALE` is unset or blank (scale 1) or a finite
/// positive number; anything else is a configuration error, not a silent 1.
/// Before there was no scale: v02/v03 multiplied by 5 when `CI` was set,
/// every other bound by nothing.  Oracle: the values parse as `f64`.
#[test]
fn time_scale_parses_finite_positive_numbers() {
    assert_eq!(parse_time_scale(None), Ok(1.0));
    assert_eq!(parse_time_scale(Some("")), Ok(1.0));
    assert_eq!(parse_time_scale(Some("  ")), Ok(1.0));
    assert_eq!(parse_time_scale(Some("2")), Ok(2.0));
    assert_eq!(parse_time_scale(Some(" 1.5 ")), Ok(1.5));
    assert_eq!(parse_time_scale(Some("0.25")), Ok(0.25));
    for bad in ["0", "-1", "abc", "inf", "NaN", "2x"] {
        let e = parse_time_scale(Some(bad)).expect_err(bad);
        assert!(e.contains("SYMPLEX_TEST_TIME_SCALE"), "{e}");
    }
}

/// A budget is multiplied by the scale of this run (1 unless the
/// environment sets one).  Oracle: `Duration` arithmetic.
#[test]
fn budgets_are_scaled() {
    let s = time_scale();
    assert!(s.is_finite() && s > 0.0);
    assert_eq!(
        scaled(Duration::from_secs(2)),
        Duration::from_secs_f64(2.0 * s)
    );
    assert_eq!(scaled(Duration::ZERO), Duration::ZERO);
    if s >= 1.0 {
        // Saturates instead of overflowing.
        assert_eq!(scaled(Duration::MAX), Duration::MAX);
    }
}

/// A timer reads the clock of the thread that started it; moving it to
/// another thread would read that thread's clock instead.  Oracle: the
/// compiler (`static_assertions`).
#[test]
fn a_cpu_timer_stays_on_its_thread() {
    static_assertions::assert_not_impl_any!(CpuTimer: Send, Sync);
}

/// The helper on a symplex call.  Oracle: the expansion `(x + 1)³ = x³ +
/// 3x² + 3x + 1` (binomial theorem).
#[test]
fn a_library_call_under_a_budget() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cube = assert_fast!(Duration::from_secs(5), (&x + 1).powi(3).expand());
    assert_eq!(cube, ctx.parse("x^3 + 3*x^2 + 3*x + 1").unwrap());
}
