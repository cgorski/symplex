//! Polynomial factoring and number theory (symplex 0.2).
//!
//! Demonstrates:
//! - Berlekamp–Zassenhaus factoring over ℤ for any degree (`factor`,
//!   `factor_list`), multivariate factoring (`factor_all`),
//!   `is_irreducible`, `sqf_list`;
//! - polynomial algebra on `Ex`: `resultant`, `discriminant`, `poly_div`,
//!   `poly_gcdex`, `decompose`, `poly_interpolate`, `nroots`,
//!   `real_roots_isolate`, `count_real_roots`;
//! - `factorint` (Pollard–Brent rho + ECM) on a 40-bit and a 60-bit
//!   semiprime and on 2⁶⁴ + 1; BPSW `isprime` (no Carmichael false
//!   positives, `2¹²⁷ − 1` in under a millisecond);
//! - modular arithmetic: `sqrt_mod`, `discrete_log`, `primitive_root`,
//!   `jacobi_symbol`, `kronecker_symbol`, `primepi`;
//! - continued fractions and Egyptian fractions;
//! - Diophantine equations: `linear_diophantine`, `pell`,
//!   `sum_of_two_squares`, `pythagorean_triples`;
//! - integer sequences: Fibonacci, Bernoulli, Bell, derangements, partitions.
//!
//! Run with: `cargo run --example factoring_and_ntheory`

use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::diophantine;
use symplex::ntheory;
use symplex::prelude::*;

fn main() {
    println!("=== Factoring and Number Theory ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    // ── 1. Univariate factoring over ℤ (Berlekamp–Zassenhaus) ───────────
    println!("--- Polynomial factoring over ℤ ---");
    println!("x⁸ − 1  = {}", (&x.powi(8) - 1).factor(&x));
    println!("x¹² − 1 = {}", (&x.powi(12) - 1).factor(&x));
    // A degree-11 product of irreducibles; 0.1 needed degree ≤ MAX_KRONECKER_DEGREE.
    let p = ((&x.powi(5) - &x - 1) * (&x.powi(4) + &x + 1) * (&x.powi(2) + 1)).expand();
    println!("{p}\n        = {}", p.factor(&x));
    let (content, factors) = p.factor_list(&x);
    let listed: Vec<String> = factors.iter().map(|(f, m)| format!("({f})^{m}")).collect();
    println!(
        "factor_list: content {content}, factors {}",
        listed.join(" · ")
    );
    let t0 = std::time::Instant::now();
    let deg20 = ((&x.powi(10) - &x.powi(3) + 2) * (&x.powi(10) + &x + 3)).expand();
    let f20 = deg20.factor(&x);
    println!("degree 20: {f20}   [{:.0?}]", t0.elapsed());
    println!("x² + 1 over ℤ stays {}", (&x.powi(2) + 1).factor(&x));
    println!(
        "is_irreducible: x⁴ + 1 → {:?},  x⁴ − 1 → {:?}",
        (&x.powi(4) + 1).is_irreducible(&x),
        (&x.powi(4) - 1).is_irreducible(&x)
    );
    let sq = ((&x - 1).powi(2) * (&x + 2).powi(3) * &x).expand();
    let (c, sqf) = sq.sqf_list(&x).unwrap();
    let listed: Vec<String> = sqf.iter().map(|(f, m)| format!("({f})^{m}")).collect();
    println!("sqf_list of {sq}:\n        {c} · {}", listed.join(" · "));

    // ── 2. Multivariate factoring (Kronecker substitution) ──────────────
    println!("\n--- Multivariate factoring ---");
    let mv = ((&x.powi(2) - &y.powi(2)) * (&x + &y * 2 + 1)).expand();
    println!("{mv}\n        = {}", mv.factor_all());

    // ── 3. Polynomial algebra ───────────────────────────────────────────
    println!("\n--- Polynomial algebra (rational coefficients) ---");
    let f = &x.powi(2) + 1;
    let g = &x.powi(2) - 2;
    println!("res(x²+1, x²−2)       = {}", f.resultant(&g, &x).unwrap());
    println!(
        "disc(x³ − x)          = {}",
        (&x.powi(3) - &x).discriminant(&x).unwrap()
    );
    println!(
        "disc((x−1)²)          = {}",
        (&x - 1).powi(2).expand().discriminant(&x).unwrap()
    );
    let (q, r) = (&x.powi(3) + &x * 2 + 1).poly_div(&f, &x).unwrap();
    println!("(x³+2x+1) ÷ (x²+1)    = quotient {q}, remainder {r}");
    let (s, t, gcd) = (&x.powi(2) - 1)
        .poly_gcdex(&(&x.powi(2) - &x * 2 + 1), &x)
        .unwrap();
    println!("gcdex(x²−1, (x−1)²)   = s·f + t·g = gcd:  s = {s}, t = {t}, gcd = {gcd}");
    let comp = (&x.powi(4) + &x.powi(2) * 2 + 1).decompose(&x);
    let parts: Vec<String> = comp.iter().map(|e| e.to_string()).collect();
    println!(
        "decompose(x⁴+2x²+1)   = {}   (outer ∘ inner)",
        parts.join(" ∘ ")
    );
    let pts = [
        (ctx.int(0), ctx.int(1)),
        (ctx.int(1), ctx.int(3)),
        (ctx.int(2), ctx.int(9)),
    ];
    println!(
        "interpolate (0,1),(1,3),(2,9) = {}",
        Ex::poly_interpolate(&pts, &x).unwrap()
    );
    let quintic = &x.powi(5) - &x - 1;
    println!(
        "x⁵ − x − 1 has {} real root(s)",
        quintic.count_real_roots(&x).unwrap()
    );
    for (re, im) in quintic.nroots(&x, 12).unwrap() {
        if im.abs() < 1e-9 {
            println!("   root ≈ {re:.12}");
        } else {
            println!(
                "   root ≈ {re:.12} {} {:.12}i",
                if im < 0.0 { "−" } else { "+" },
                im.abs()
            );
        }
    }
    let iso = (&x.powi(3) - &x * 2 - 5).real_roots_isolate(&x);
    for iv in &iso {
        // `Display` shows the kind: `(lo, hi]` for a Sturm cell, `[r, r]` for an exact hit.
        println!("x³ − 2x − 5: real root isolated in {iv}");
    }

    // ── 4. Integer factorization and primality ──────────────────────────
    println!("\n--- factorint / isprime ---");
    let n40: u64 = 1_048_583 * 1_048_589; // two 21-bit primes → 41-bit semiprime
    let t0 = std::time::Instant::now();
    println!(
        "factorint({n40}) = {:?}   [{:.1?}]",
        ntheory::factorint(n40),
        t0.elapsed()
    );
    let n60 = BigInt::from(1_073_741_827u64) * BigInt::from(1_073_741_831u64);
    let t0 = std::time::Instant::now();
    println!(
        "factorint({n60}) = {:?}   [{:.1?}]",
        ntheory::factorint(n60.clone()),
        t0.elapsed()
    );
    let fermat6 = BigInt::from(2u128.pow(64) + 1);
    println!("factorint(2⁶⁴ + 1) = {:?}", ntheory::factorint(fermat6));
    println!(
        "isprime(561) [Carmichael number] = {}",
        ntheory::isprime(561)
    );
    let m127 = BigInt::parse_bytes(b"170141183460469231731687303715884105727", 10).unwrap();
    let t0 = std::time::Instant::now();
    println!(
        "isprime(2¹²⁷ − 1) = {}   [{:.1?}]",
        ntheory::isprime(m127),
        t0.elapsed()
    );
    println!("π(10⁶) = {}", ntheory::primepi(1_000_000).unwrap());
    // The same on Ex values:
    println!(
        "Ex: 97.is_prime_value() = {:?}",
        ctx.int(97).is_prime_value()
    );

    // ── 5. Modular arithmetic ───────────────────────────────────────────
    println!("\n--- Modular arithmetic ---");
    println!(
        "√2 mod 7        = {:?}   (all: {:?})",
        ntheory::sqrt_mod(2, 7),
        ntheory::sqrt_mod_all(2, 7)
    );
    println!(
        "√3 mod 7        = {:?}   (3 is not a quadratic residue mod 7)",
        ntheory::sqrt_mod(3, 7)
    );
    println!("√1 mod 15       = {:?}", ntheory::sqrt_mod_all(1, 15));
    println!(
        "3ˣ ≡ 13 (mod 17) → x = {:?}",
        ntheory::discrete_log(3, 13, 17)
    );
    println!("primitive root of 17 = {:?}", ntheory::primitive_root(17));
    println!(
        "(1001 / 9907) Jacobi = {:?}",
        ntheory::jacobi_symbol(1001, 9907)
    );
    println!("(3 / 8) Kronecker    = {}", ntheory::kronecker_symbol(3, 8));
    println!(
        "3⁻¹ mod 7 = {:?},   crt(x≡2 mod 3, x≡3 mod 5, x≡2 mod 7) = {:?}",
        ntheory::mod_inverse(3, 7),
        ntheory::crt_i64(&[2, 3, 2], &[3, 5, 7])
    );

    // ── 6. Continued fractions ──────────────────────────────────────────
    println!("\n--- Continued fractions ---");
    let r = Ratio::new(BigInt::from(415), BigInt::from(93));
    println!("415/93 = {:?}", ntheory::continued_fraction(&r));
    let (head, period) = ntheory::continued_fraction_periodic(23).unwrap();
    let fmt = |v: &[BigInt]| {
        v.iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!(
        "√23    = [{}; ({})]   (period {})",
        fmt(&head),
        fmt(&period),
        period.len()
    );
    let pi_terms: Vec<BigInt> = [3, 7, 15, 1].iter().map(|&k| BigInt::from(k)).collect();
    let conv: Vec<String> = ntheory::continued_fraction_convergents(&pi_terms)
        .iter()
        .map(|c| format!("{}/{}", c.numer(), c.denom()))
        .collect();
    println!("convergents of [3; 7, 15, 1] = {}", conv.join(", "));
    let r = Ratio::new(BigInt::from(4), BigInt::from(13));
    println!("4/13 as Egyptian fraction: 1/{}", {
        let d = ntheory::egyptian_fraction(&r).unwrap();
        d.iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(" + 1/")
    });

    // ── 7. Diophantine equations ────────────────────────────────────────
    println!("\n--- Diophantine ---");
    let (x0, y0, dx, dy) = diophantine::linear_diophantine(3, 5, 1).unwrap();
    println!(
        "3x + 5y = 1: x = {x0} + {dx}k, y = {y0} {} {}k",
        if dy < BigInt::from(0) { "−" } else { "+" },
        dy.magnitude()
    );
    let (px, py) = diophantine::pell(61).unwrap();
    println!("Pell x² − 61y² = 1: fundamental solution ({px}, {py})");
    println!(
        "Pell x² − 2y² = 1: first four {:?}",
        diophantine::pell_solutions(2, 4)
    );
    println!(
        "65 = a² + b²: {:?};   2021 = 43·47: {:?}",
        diophantine::sum_of_two_squares(65),
        diophantine::sum_of_two_squares(2021)
    );
    println!(
        "primitive Pythagorean triples with c ≤ 30: {:?}",
        diophantine::pythagorean_triples(30)
    );

    // ── 8. Sequences ────────────────────────────────────────────────────
    println!("\n--- Sequences ---");
    println!("F(100)          = {}", ntheory::fibonacci(100));
    println!("B(12)           = {}", ntheory::bernoulli(12).unwrap());
    println!(
        "Bell(10)        = {}",
        symplex::combinatorics::bell(10).unwrap()
    );
    println!(
        "derangements(10)= {}",
        symplex::combinatorics::derangements(10).unwrap()
    );
    println!(
        "partitions of 5 = {:?}",
        symplex::combinatorics::partitions(5).collect::<Vec<_>>()
    );
    let n = ctx.symbol("n");
    println!(
        "symbolic: fibonacci(n) = {},  fibonacci(30).eval() = {}",
        n.fibonacci(),
        ctx.int(30).fibonacci().eval()
    );

    println!("\n✓ Done!");
}
