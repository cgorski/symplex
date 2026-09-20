//! 0.11.1 review fixes: the items outside the three review tracks — the
//! SOS budget start, the NTT plan, the fraction-free / GF(p) division
//! fallbacks and the ratchet.  Reference values cite SymPy 1.14
//! (`symplex/.venv/bin/python`).

use std::time::{Duration, Instant};

use num_bigint::BigInt;
use symplex::certificates::{SosOpts, SosOutcome, prove_sos};
use symplex::discrete::{convolution_ntt, intt, ntt};
use symplex::prelude::*;
use symplex::syms;

fn b(v: &[i64]) -> Vec<BigInt> {
    v.iter().map(|&t| BigInt::from(t)).collect()
}

/// A time limit is measured from the *start* of `prove_sos`: a goal whose
/// Newton-polytope pruning runs dozens of LPs comes back as `Unknown`
/// from the pruning stage instead of overshooting the limit by the whole
/// stage and the SDP assembly.  (0.11.0 fixed the deadline only after
/// pruning; the cheap refutation search stays unbudgeted by contract — a
/// counterexample is decisive.)
#[test]
fn sos_time_limit_covers_the_newton_pruning_stage() {
    let ctx = Context::new();
    syms!(ctx; x, y, z);
    // Dense degree-6 goal in three variables: 84 candidate monomials of
    // degree ≤ 3, one LP each, before the SDP would even start.
    let goal = (&x.powi(2) + &y.powi(2) + &z.powi(2) + 1).powi(3);
    let t0 = Instant::now();
    let out = prove_sos(
        &goal,
        &[x.clone(), y.clone(), z.clone()],
        &SosOpts::default().with_time_limit(Duration::ZERO),
    )
    .unwrap();
    let elapsed = t0.elapsed();
    match out {
        SosOutcome::Unknown(u) => assert_eq!(
            u.reason,
            "budget exhausted: deadline passed during the Newton-polytope pruning"
        ),
        other => panic!("expected Unknown(budget exhausted), got {other}"),
    }
    // Generous: the refutation grid (343 exact evaluations) runs first.
    assert!(
        elapsed < Duration::from_secs(10),
        "a zero limit took {elapsed:?}"
    );
    // Without a limit the same goal is proved — the pruning budget only
    // *keeps* monomials when it is hit, never drops one.
    let full = prove_sos(&goal, &[x, y, z], &SosOpts::default()).unwrap();
    assert!(matches!(full, SosOutcome::Proved(_)), "{full}");
}

/// The NTT plan (prime validated once, root derived once) gives the same
/// entries as before and as SymPy; negative inputs reduce modulo `p`.
#[test]
fn ntt_plan_matches_sympy_including_negative_entries() {
    // SymPy: ntt([-1, 2, -3, 4], prime=998244353) == [2, 173167438, 998244343, 825076919]
    let t = ntt(&b(&[-1, 2, -3, 4]), 998_244_353).unwrap();
    assert_eq!(t, b(&[2, 173_167_438, 998_244_343, 825_076_919]));
    assert_eq!(
        intt(&t, 998_244_353).unwrap(),
        b(&[998_244_352, 2, 998_244_350, 4])
    );
    // SymPy: ntt([7], prime=17) == [7]
    assert_eq!(ntt(&b(&[7]), 17).unwrap(), b(&[7]));
    assert_eq!(ntt(&[], 17).unwrap(), Vec::<BigInt>::new());
    // SymPy: convolution_ntt([1, -2, 3], [4, 5], prime=998244353) == [4, 998244350, 2, 15]
    assert_eq!(
        convolution_ntt(&b(&[1, -2, 3]), &b(&[4, 5]), 998_244_353).unwrap(),
        b(&[4, 998_244_350, 2, 15])
    );
}

/// The trivial product validates its modulus like `ntt` does (0.11.0
/// returned `Ok([])` for any modulus).
#[test]
fn convolution_ntt_validates_the_prime_even_for_an_empty_input() {
    assert!(convolution_ntt(&[], &b(&[1, 2]), 6).is_err());
    assert!(ntt(&[], 6).is_err());
    assert_eq!(
        convolution_ntt(&[], &b(&[1, 2]), 7).unwrap(),
        Vec::<BigInt>::new()
    );
}

/// A prime with too small a power of two in `p − 1` is rejected with the
/// length that failed.
#[test]
fn ntt_reports_the_missing_root_of_unity() {
    // 7 − 1 = 6: 4 ∤ 6.
    let err = ntt(&b(&[1, 2, 3, 4]), 7).unwrap_err().to_string();
    assert!(err.contains("4-th root of unity"), "{err}");
    // 2 is prime; a length-1 transform needs no root …
    assert_eq!(ntt(&b(&[5]), 2).unwrap(), b(&[1]));
    // … but a length-2 one needs 2 | p − 1 = 1.
    assert!(ntt(&b(&[1, 1]), 2).is_err());
}

/// Division by the zero polynomial is `None` at the public boundary (and,
/// since 0.11.1, a graceful `(0, a)` rather than an `assert!` in the
/// internal `GenPoly::div_rem` every other route goes through).
#[test]
fn polynomial_division_by_zero_is_none_not_a_panic() {
    let ctx = Context::new();
    syms!(ctx; x);
    let p = &x.powi(2) + 1;
    assert!(p.poly_div(&ctx.zero(), &x).is_none());
    assert!(p.poly_rem(&ctx.zero(), &x).is_none());
    // SymPy: div(x**2 + 1, x + 1) == (x - 1, 2)
    let (q, r) = p.poly_div(&(&x + 1), &x).unwrap();
    assert_eq!(q, &x - 1);
    assert_eq!(r, ctx.int(2));
}
