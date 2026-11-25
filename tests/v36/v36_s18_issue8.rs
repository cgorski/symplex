//! Tests of session 18: issue #8, `eval_decimal` at very high precision
//! (`π` to ten million digits took 411 s on the reporter's machine).
//! Every test says what was wrong before and cites its oracle.

use symplex::prelude::*;

use crate::perf_assert::cpu_time;

/// A context whose precision limit admits 100,000 digits (the default,
/// 10,000 bits, stops near 2,900).
fn wide() -> Context {
    Context::with_config(EvalConfig {
        max_evalf_precision: 400_000,
        ..Default::default()
    })
}

/// Expressions whose evaluation reads the cached constants — `π`, `e`,
/// `ln 2` and `ln 10` themselves, and functions that reduce their argument
/// by `π` or `ln 2` — each with the largest digit count it is evaluated to
/// here (the special functions are slow in a debug build).
const EXPRESSIONS: [(&str, u32); 14] = [
    ("pi", 2500),
    ("E", 2500),
    ("log(2)", 2500),
    ("log(10)", 2500),
    ("exp(1) + pi", 2500),
    ("sin(10^20)", 2500),
    ("cos(355/113)", 2500),
    ("atan(1/3) + sqrt(2)", 2500),
    ("exp(I*pi/3)", 2500),
    ("log(-7/2)", 2500),
    ("tan(pi/2 - 10^(-30))", 1000),
    ("gamma(1/3)", 300),
    ("zeta(3)", 300),
    ("sin(1)^2 + cos(1)^2 - 1 + sin(10^(-200))", 300),
];

/// What the caller sees of `text` at `digits` digits: the decimal string
/// (or the refusal) and the bits of `eval_f64` (or its refusal).
fn observed(ctx: &Context, text: &str, digits: u32) -> String {
    let e = ctx.parse(text).unwrap();
    format!(
        "{:?} / {:?}",
        e.eval_decimal(digits),
        e.eval_f64().map(f64::to_bits)
    )
}

/// Since 0.60.1 every numeric evaluation of a thread shares one cache of
/// `π`, `e`, `ln 2` and `ln 10` (`eval_decimal` created a new one per call
/// before, and recomputed `π` each time).  What is printed must not depend
/// on it: each evaluation on a thread that has evaluated nothing before
/// (cold) against the same evaluations on one thread whose caches were
/// first warmed at a higher precision, in falling and then rising order of
/// precision.
///
/// Oracle: the cold evaluation — what 0.60.0 printed, where every
/// `eval_decimal` started from an empty cache.
#[test]
fn evaluation_is_the_same_on_a_cold_and_a_warm_thread() {
    let digits = [16u32, 30, 50, 300, 1000, 2500];
    let mut cold = Vec::new();
    for (text, most) in EXPRESSIONS {
        for d in digits.into_iter().filter(|&d| d <= most) {
            let seen = std::thread::spawn(move || observed(&wide(), text, d))
                .join()
                .expect("the cold evaluation returns");
            cold.push((text, d, seen));
        }
    }
    let warm = std::thread::spawn(move || {
        let ctx = wide();
        // Warm every band these precisions use, beyond what they need.
        for text in ["pi", "E", "log(2)", "log(10)", "sin(1)"] {
            for d in [9000, 500] {
                ctx.parse(text).unwrap().eval_decimal(d).unwrap();
            }
        }
        let mut seen = Vec::new();
        let falling = digits.into_iter().rev();
        for d in falling.chain(digits) {
            for (text, most) in EXPRESSIONS {
                if d <= most {
                    seen.push((text, d, observed(&ctx, text, d)));
                }
            }
        }
        seen
    })
    .join()
    .expect("the warm evaluations return");
    assert_eq!(warm.len(), 2 * cold.len());
    for (text, d, seen) in warm {
        let want = cold
            .iter()
            .find(|(t, n, _)| (*t, *n) == (text, d))
            .map(|(_, _, s)| s);
        assert_eq!(Some(&seen), want, "{text} at {d} digits");
    }
}

/// `π` to 20,000 digits, and again: the second call copies the constant
/// instead of computing it, `Context::clear_numeric_cache` makes the next
/// call compute it once more, and the digits are the same all three times.
/// Up to 0.60.0 every call computed `π` anew (a million digits: 3.32 s,
/// and 3.31 s the second time).
///
/// Oracle: mpmath 1.4.0, `mp.dps = 20030; mp.nstr(mp.pi, 20000,
/// strip_zeros=False)` (the same string at `mp.dps = 20200`): 20,001
/// characters, the first `3.1415926535`, the last
/// `109418074487814049075518`.
#[test]
fn pi_is_computed_once_per_thread() {
    let ctx = wide();
    let pi = ctx.pi();
    let (first, cold) = cpu_time(|| pi.eval_decimal(20_000).unwrap());
    assert_eq!(first.len(), 20_001);
    assert!(first.starts_with("3.1415926535"), "{}", &first[..40]);
    assert!(
        first.ends_with("109418074487814049075518"),
        "{}",
        &first[first.len() - 40..]
    );
    let (second, warm) = cpu_time(|| pi.eval_decimal(20_000).unwrap());
    assert_eq!(second, first);
    Context::clear_numeric_cache();
    let (third, cold_again) = cpu_time(|| pi.eval_decimal(20_000).unwrap());
    assert_eq!(third, first);
    eprintln!("pi to 20,000 digits: cold {cold:?}, warm {warm:?}, after clearing {cold_again:?}");
    // Measured in a debug build (c7i.16xlarge): cold 108 ms, warm 12.5 ms,
    // after clearing 107 ms — the warm call is the decimal conversion alone.
    assert!(
        warm.as_secs_f64() < WARM_SHARE * cold.as_secs_f64()
            && warm.as_secs_f64() < WARM_SHARE * cold_again.as_secs_f64(),
        "the second call recomputed pi: cold {cold:?}, warm {warm:?}, after clearing {cold_again:?}"
    );
}

/// The share of a cold `eval_decimal(π)` that a warm one may take in
/// [`pi_is_computed_once_per_thread`] (measured: 0.12).
const WARM_SHARE: f64 = 0.5;
