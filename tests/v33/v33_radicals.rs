//! Radicals of integers with thousands of digits.

use num_bigint::BigInt;
use std::time::{Duration, Instant};
use symplex::prelude::*;

fn mersenne(ctx: &Context, p: u32) -> Ex {
    let m: BigInt = (BigInt::from(1) << p) - 1;
    ctx.parse(&m.to_string()).unwrap()
}

/// The nightly `fuzz_roundtrip` timeout of 2026-10-01 (33 s on CI; 4 s
/// locally): building `sqrt(p/q)` for a rational with 1,000-digit terms
/// splits perfect powers out of `p` and `q`, and the bounded squarefree
/// decomposition ran a BPSW primality test on every remainder over its
/// factoring bound (0.1 s for 1,200 digits, 0.3 s for 3,000) and took the
/// exact `q`-th root for every prime `q` up to the bit length (0.24 s at
/// 10,000 bits).  Neither changes the split of a remainder that is not a
/// perfect power; a residue filter now discards almost every exponent.
///
/// The Mersenne numbers `2^4423 − 1`, `2^3217 − 1` and `2^2203 − 1` are
/// prime (SymPy: `isprime(2**4423 - 1)` → `True`), so none of the radicals
/// below simplifies, except for the perfect power.
#[test]
fn radicals_of_huge_integers_are_fast() {
    let ctx = Context::new();
    let (a, b, c) = (
        mersenne(&ctx, 4423),
        mersenne(&ctx, 3217),
        mersenne(&ctx, 2203),
    );
    let t = Instant::now();
    let r1 = a.sqrt();
    let r2 = (&a / &b).sqrt();
    let r3 = (&a.powi(2) * &c).sqrt();
    let r4 = (&a * 3).cbrt();
    let r5 = (&a.powi(3) * 5).cbrt();
    let elapsed = t.elapsed();
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
    assert_eq!(r1.powi(2), a);
    assert_eq!(r2.powi(2), &a / &b);
    assert_eq!(r3.powi(2), &a.powi(2) * &c);
    assert_eq!(r4.powi(3), &a * 3);
    // A perfect power of a prime remainder is still pulled out (of a
    // composite one over the factoring bound it never was).
    assert_eq!(r5, &a * &ctx.int(5).cbrt());
}
