//! Tests of the 0.50 refactor (w12): performance of the core.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

/// A chain `e₀ = x`, `eₖ₊₁ = sin(eₖ) + cos(eₖ)` of `n` steps: a DAG of
/// about `3n` nodes whose unfolded tree has about `2ⁿ` nodes.
fn sin_cos_chain(ctx: &Context, n: usize) -> Ex {
    let mut e = ctx.symbol("x");
    for _ in 0..n {
        e = e.sin() + e.cos();
    }
    e
}

/// `ExprNode` is 40 bytes: smallvec's `union` feature lets the inline and
/// heap forms of the child lists share storage.
///
/// Before: 48 bytes (`size_of::<ExprNode>()`, measured by the S5 survey on
/// 0.48.0).  Oracle: `std::mem::size_of`.
#[cfg(target_pointer_width = "64")]
#[test]
fn expr_node_is_forty_bytes() {
    assert_eq!(
        std::mem::size_of::<symplex::__macro_support::ExprNode>(),
        40
    );
}

/// The structural queries answer from facts stored with each node instead
/// of walking the DAG on every call.
///
/// Before: `has_unevaluated`, `free_symbols`, `is_constant` and `contains`
/// of a symbol each walked the whole DAG (here 30,001 nodes); the 4 × 2,000
/// queries below took 69.7 s in a debug build of 0.49 (c6bd5ce) on m1, and
/// now take under 0.1 s.  Oracle: the answers are those of the definitions
/// (the chain contains `x`, no other symbol and no unevaluated node); the
/// time bound is 14 times below the old time and 50 times above the new.
#[test]
fn structural_queries_on_a_deep_dag_are_constant_time() {
    let ctx = Context::new();
    let e = sin_cos_chain(&ctx, 10_000);
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let start = std::time::Instant::now();
    for _ in 0..2_000 {
        assert!(!e.has_unevaluated());
        assert_eq!(e.free_symbols(), vec![x.clone()]);
        assert!(!e.is_constant());
        assert!(!e.contains(&y));
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "8,000 queries took {elapsed:?}"
    );
}

/// Free symbols when a context has more than 63 symbols: the first 63
/// `Symbol` nodes own one bit of the free-symbol mask each and the later
/// ones share bit 63, which a binder of such a symbol cannot clear; those
/// queries walk.  The answers are the definitions' (a bound variable is not
/// free; the order is the order of the depth-first walk).
///
/// Before: the same answers by a full walk per query (this test guards the
/// shared-bit path of the new facts).  Oracle: `walk::free_symbols`' binder
/// rules — `Σ_{k=lo}^{hi} body` binds `k` in `body` only — and the output of
/// 0.49 (c6bd5ce) for the same calls.
#[test]
fn free_symbols_beyond_sixty_three_symbols() {
    let ctx = Context::new();
    let s: Vec<Ex> = (0..70).map(|i| ctx.symbol(&format!("s{i}"))).collect();
    let f = |a: &Ex| ctx.apply("f", &[a]).unwrap();
    let (k, n, m) = (&s[69], &s[68], &s[67]);
    // Σ_{s69=1}^{s68} f(s69): s69 (shared bit) bound, s68 free.
    let sum = f(k).summation(k, &ctx.int(1), n);
    let names = |v: Vec<Ex>| v.iter().map(Ex::to_string).collect::<Vec<_>>();
    assert_eq!(names(sum.free_symbols()), ["s68"]);
    assert!(!sum.is_constant());
    // Σ_{s69=1}^{∞} f(s69): no free symbol, although bit 63 is set below.
    let closed = f(k).summation(k, &ctx.int(1), &ctx.infinity());
    assert!(closed.free_symbols().is_empty());
    assert!(closed.is_constant());
    assert!(closed.contains(k), "a bound occurrence is an occurrence");
    // s67·Σ + s3 + Σ_{s69=1}^{∞} f(s69): free symbols in walk order.
    let e = m * &sum + &s[3] + &closed;
    assert_eq!(names(e.free_symbols()), EXPECTED_ORDER);
    assert!(e.contains(k));
    assert!(!e.contains(&s[66]));
}

/// The order `free_symbols` lists the symbols of
/// `free_symbols_beyond_sixty_three_symbols`' last expression in, as
/// 0.49 (c6bd5ce) lists it.
const EXPECTED_ORDER: [&str; 3] = ["s68", "s67", "s3"];

/// A sum whose terms' sort keys are longer than 256 bytes and equal in
/// their first 256: their order is decided by the 64-bit `FxHasher` digest
/// of the whole pre-truncation key, which the sort-key pool must reproduce
/// bit for bit (a different digest reorders the printed terms).
///
/// Before: the same output from per-node `SortKey` objects (this pins it
/// for the pool).  Oracle: the output of 0.49 (c6bd5ce) on the same 64-bit
/// target (`rustc-hash` digests depend on the pointer width).
#[cfg(target_pointer_width = "64")]
#[test]
fn long_sort_keys_keep_their_order() {
    let ctx = Context::new();
    let a = (0..100).fold(ctx.int(0), |acc, i| acc + ctx.symbol(&format!("a{i}")));
    let terms: Vec<Ex> = (0..8)
        .map(|i| &a * ctx.symbol(&format!("b{i}")).sin())
        .collect();
    let e = terms.iter().fold(ctx.int(0), |acc, t| acc + t);
    let order: Vec<usize> = (0..8)
        .map(|i| {
            let s = e.to_string();
            s.find(&format!("sin(b{i})")).unwrap_or(usize::MAX)
        })
        .collect();
    let mut ranked: Vec<usize> = (0..8).collect();
    ranked.sort_by_key(|&i| order[i]);
    assert_eq!(ranked, EXPECTED_LONG_KEY_ORDER);
}

/// The order in which `long_sort_keys_keep_their_order` prints its terms
/// (by the index `i` of `sin(bᵢ)`), as 0.49 (c6bd5ce) prints it.
#[cfg(target_pointer_width = "64")]
const EXPECTED_LONG_KEY_ORDER: [usize; 8] = [5, 7, 4, 1, 6, 2, 3, 0];

/// Numbers sort by the decimal bytes of their value, which the key pool
/// now writes without the big-integer conversion when they fit a machine
/// word: a finite set prints its elements in key order.
///
/// Before: the same order (this pins the machine-word fast path at the
/// `i64` boundaries).  Oracle: the output of 0.49 (c6bd5ce), whose keys
/// were `Ratio::to_string()` bytes.
#[test]
fn number_keys_at_machine_word_boundaries() {
    let ctx = Context::new();
    let big = |s: &str| ctx.parse(s).unwrap();
    let elems = [
        big("9223372036854775807"),
        big("9223372036854775808"),
        big("-9223372036854775808"),
        big("-9223372036854775809"),
        big("1/9223372036854775807"),
        big("-7/9223372036854775808"),
        ctx.int(-1),
        ctx.int(10),
        ctx.int(9),
        ctx.rational(-3, 4),
    ];
    let set = ctx.finite_set(&elems);
    assert_eq!(set.to_string(), EXPECTED_NUMBER_SET);
}

/// `number_keys_at_machine_word_boundaries`' set as 0.49 (c6bd5ce) prints
/// it.
const EXPECTED_NUMBER_SET: &str = "{-1, -3/4, -7/9223372036854775808, -9223372036854775808, \
     -9223372036854775809, 1/9223372036854775807, 10, 9, 9223372036854775807, \
     9223372036854775808}";
