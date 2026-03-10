//! Combinatorial Counting — solve real counting problems with Stirling numbers,
//! partitions, and multinomial coefficients.
//!
//! This example shows how to use symplex's combinatorics module to answer
//! questions that come up in probability, CS, and discrete math courses:
//!
//!   1. Distributing distinct objects into identical bins (Stirling S(n,k))
//!   2. Counting permutations by cycle structure (Stirling s(n,k))
//!   3. Counting anagrams of words with repeated letters (multinomial)
//!   4. Making change / integer partitions (partition_count)
//!   5. Cross-validating Stirling numbers against Bell numbers
//!   6. Using the expression API for symbolic combinatorics
//!
//! Run with: `cargo run --example combinatorics_counting`

use num_bigint::BigInt;
use symplex::combinatorics::*;
use symplex::prelude::*;

fn main() {
    println!("=== Combinatorial Counting Problems ===\n");

    // ── 1. Distributing objects into groups ─────────────────────────
    //
    // "How many ways can 8 students be split into 3 non-empty project groups?"
    //
    // This is exactly the Stirling number of the second kind S(8, 3).
    // The groups are unordered (identical bins), the students are distinct.

    println!("--- Distributing Distinct Objects into Identical Bins ---");

    let ways = stirling2(8, 3).unwrap();
    println!("  8 students into 3 groups: S(8,3) = {ways}");
    // If the groups are LABELED (e.g., Team A, B, C), multiply by 3! = 6
    println!(
        "  Into 3 LABELED teams:     S(8,3)·3! = {}",
        &ways * BigInt::from(6)
    );

    // More examples
    for (n, k) in [(10, 4), (12, 3), (6, 2), (5, 5)] {
        let s = stirling2(n, k).unwrap();
        println!("  {n} objects into {k} groups: S({n},{k}) = {s}");
    }

    // Special case: S(n, 2) = 2^(n-1) - 1
    let n = 20;
    let s = stirling2(n, 2).unwrap();
    println!("\n  S({n}, 2) = 2^{} - 1 = {s}", n - 1);
    assert_eq!(s, BigInt::from(2u64.pow(19) - 1));

    // ── 2. Permutations by cycle structure ──────────────────────────
    //
    // "How many permutations of 6 elements have exactly 2 cycles?"
    //
    // The unsigned Stirling number |s(6, 2)| counts this.
    // The signed version s(6, 2) alternates in sign.

    println!("\n--- Permutations by Cycle Count ---");

    for k in 1..=6u64 {
        let s = stirling1(6, k).unwrap();
        let abs_s = if s < BigInt::from(0) { -&s } else { s.clone() };
        println!("  Perms of 6 elements with {k} cycle(s): |s(6,{k})| = {abs_s:>5}  (signed: {s})");
    }

    // The unsigned values should sum to 6! = 720
    let total: BigInt = (1..=6u64)
        .map(|k| {
            let s = stirling1(6, k).unwrap();
            if s < BigInt::from(0) { -s } else { s }
        })
        .sum();
    println!("  Sum = {total} (should be 6! = 720)");
    assert_eq!(total, BigInt::from(720));

    // ── 3. Counting anagrams (multinomial coefficients) ─────────────
    //
    // "How many distinct arrangements of the letters in MISSISSIPPI?"
    //
    // MISSISSIPPI has 11 letters: M×1, I×4, S×4, P×2
    // Answer: 11! / (1! · 4! · 4! · 2!) = multinomial(11, [1, 4, 4, 2])

    println!("\n--- Anagrams (Multinomial Coefficients) ---");

    let mississippi = multinomial(11, &[1, 4, 4, 2]).unwrap();
    println!("  MISSISSIPPI: 11!/(1!·4!·4!·2!) = {mississippi}");
    // Verify: 11! = 39916800, 1!·4!·4!·2! = 1·24·24·2 = 1152
    // 39916800 / 1152 = 34650
    assert_eq!(mississippi, BigInt::from(34650));

    // ABRACADABRA: A×5, B×2, R×2, C×1, D×1
    let abracadabra = multinomial(11, &[5, 2, 2, 1, 1]).unwrap();
    println!("  ABRACADABRA: 11!/(5!·2!·2!·1!·1!) = {abracadabra}");
    assert_eq!(abracadabra, BigInt::from(83160));

    // Simpler: BANANA has B×1, A×3, N×2 → 6!/(1!·3!·2!) = 60
    let banana = multinomial(6, &[1, 3, 2]).unwrap();
    println!("  BANANA:      6!/(1!·3!·2!) = {banana}");
    assert_eq!(banana, BigInt::from(60));

    // Dice: ways to roll 12 dice and get each face at least once,
    // then 6 specific additional faces: multinomial(12, [2,2,2,2,2,2])
    let dice = multinomial(12, &[2, 2, 2, 2, 2, 2]).unwrap();
    println!("  12 dice, each face exactly 2×: 12!/(2!)⁶ = {dice}");
    // 479001600 / 64 = 7484400
    assert_eq!(dice, BigInt::from(7484400));

    // ── 4. Integer partitions ───────────────────────────────────────
    //
    // "In how many ways can you write 50 as a sum of positive integers?"
    //
    // This is the partition function p(50).

    println!("\n--- Integer Partitions ---");

    for n in [5, 10, 20, 50, 100] {
        let p = partition_count(n).unwrap();
        println!("  p({n:>3}) = {p}");
    }

    // Fun fact: p(100) = 190,569,292 — almost 200 million ways
    // to write 100 as a sum of positive integers!
    let p100 = partition_count(100).unwrap();
    assert_eq!(p100, BigInt::from(190569292));

    // p(200) is much larger
    let p200 = partition_count(200).unwrap();
    println!("  p(200) = {p200}");
    println!("  (That's {} digits!)", p200.to_string().len());

    // ── 5. Cross-validation: Stirling ↔ Bell numbers ────────────────
    //
    // The Bell number B(n) = Σ_{k=0}^{n} S(n, k).
    // This is the total number of set partitions.

    println!("\n--- Stirling → Bell Cross-Validation ---");

    let bell_numbers = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975];
    for (n, &expected_bell) in bell_numbers.iter().enumerate() {
        let computed: BigInt = (0..=n)
            .map(|k| stirling2(n as u64, k as u64).unwrap())
            .sum();
        let ok = computed == BigInt::from(expected_bell);
        println!(
            "  B({n:>2}) = Σ S({n},k) = {computed:>8} (expected {expected_bell:>8}) {}",
            if ok { "✓" } else { "✗" }
        );
        assert_eq!(computed, BigInt::from(expected_bell));
    }

    // ── 6. Stirling orthogonality ───────────────────────────────────
    //
    // The first and second kind Stirling numbers are inverses:
    //   Σ_j s(n, j) · S(j, k) = δ(n, k)

    println!("\n--- Stirling Orthogonality: Σ s(n,j)·S(j,k) = δ(n,k) ---");

    let size = 6;
    let mut all_ok = true;
    for n in 0..size {
        for k in 0..size {
            let sum: BigInt = (0..size)
                .map(|j| stirling1(n, j).unwrap() * stirling2(j, k).unwrap())
                .sum();
            let expected = if n == k { 1 } else { 0 };
            if sum != BigInt::from(expected) {
                println!("  FAIL: n={n}, k={k}: got {sum}, expected {expected}");
                all_ok = false;
            }
        }
    }
    if all_ok {
        println!("  All {size}×{size} entries verified ✓");
    }

    // ── 7. Expression-level API ─────────────────────────────────────
    //
    // The combinatorial functions also work as symbolic expressions:

    println!("\n--- Symbolic Combinatorics (Expression API) ---");

    let ctx = Context::new();
    symplex::syms!(ctx; n);

    // Concrete evaluation through the expression layer
    let s53 = ctx.int(5).stirling2(&ctx.int(3)).eval();
    println!("  S(5, 3) = {s53}");
    assert_eq!(format!("{s53}"), "25");

    let s41 = ctx.int(4).stirling1(&ctx.int(1)).eval();
    println!("  s(4, 1) = {s41}");
    assert_eq!(format!("{s41}"), "-6");

    let p10 = ctx.int(10).partition_count().eval();
    println!("  p(10)   = {p10}");
    assert_eq!(format!("{p10}"), "42");

    // Symbolic: stays unevaluated when the argument is a variable
    let symbolic = n.stirling2(&ctx.int(3));
    println!("  S(n, 3) = {symbolic}  (symbolic — n is a free variable)");

    println!("\n✓ Done!");
}
