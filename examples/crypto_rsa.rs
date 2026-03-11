//! RSA Encryption — key generation, encryption, and decryption using number theory.
//!
//! This example demonstrates a real cryptographic workflow using symplex's
//! number theory module:
//!
//!   1. Generate two large primes with `nextprime()`
//!   2. Compute the RSA modulus n = p·q and totient φ(n)
//!   3. Choose public exponent e, compute private key d = e⁻¹ mod φ(n)
//!   4. Encrypt a message: c = m^e mod n
//!   5. Decrypt: m = c^d mod n
//!   6. Verify the round-trip
//!   7. Sign and verify a message
//!
//! This is textbook RSA — NOT suitable for production use (no padding,
//! small keys, etc.) — but it demonstrates real number theory operations.
//!
//! Run with: `cargo run --example crypto_rsa`

use num_bigint::BigInt;
use symplex::ntheory::*;

fn main() {
    println!("=== RSA Encryption with symplex Number Theory ===\n");

    // ── 1. Key generation ──────────────────────────────────────────
    //
    // Choose two distinct primes p and q.
    // In real RSA these would be ~1024 bits each; here we use smaller
    // values for readability while still being too large for brute force.

    println!("--- Key Generation ---");

    let p = nextprime(1_000_000_007i64);
    let q = nextprime(2_000_000_011i64);
    println!("  p = {p}");
    println!("  q = {q}");
    assert!(isprime(p.clone()), "p should be prime");
    assert!(isprime(q.clone()), "q should be prime");
    assert_ne!(p, q, "p and q must be distinct");

    // RSA modulus
    let n = &p * &q;
    println!("  n = p·q = {n}");
    println!("  n has {} digits", n.to_string().len());

    // Euler's totient: φ(n) = (p-1)(q-1)
    let phi = (&p - BigInt::from(1)) * (&q - BigInt::from(1));
    println!("  φ(n) = (p-1)(q-1) = {phi}");

    // Public exponent — standard choice e = 65537
    let e = BigInt::from(65537i64);
    println!("  e = {e}");

    // Verify gcd(e, φ(n)) = 1 (required for RSA)
    let g = gcd(e.clone(), phi.clone());
    println!("  gcd(e, φ(n)) = {g}");
    assert!(g == BigInt::from(1), "e must be coprime to φ(n)");

    // Private exponent: d = e⁻¹ mod φ(n)
    let d = mod_inverse(e.clone(), phi.clone()).expect("modular inverse must exist");
    println!("  d = e⁻¹ mod φ(n) = {d}");

    // Verify: e·d ≡ 1 (mod φ(n))
    let _check = mod_pow(e.clone(), BigInt::from(1), phi.clone());
    let ed_mod_phi = (&e * &d) % &phi;
    println!("  e·d mod φ(n) = {ed_mod_phi}");
    assert_eq!(ed_mod_phi, BigInt::from(1), "e·d should be 1 mod φ(n)");

    println!("\n  Public key:  (e={e}, n={n})");
    println!("  Private key: (d={d}, n={n})");

    // ── 2. Encryption ──────────────────────────────────────────────
    //
    // To encrypt message m: c = m^e mod n

    println!("\n--- Encryption ---");

    let message = BigInt::from(42_000_000i64);
    println!("  Original message:  m = {message}");

    let ciphertext = mod_pow(message.clone(), e.clone(), n.clone());
    println!("  Encrypted:         c = m^e mod n = {ciphertext}");

    // ── 3. Decryption ──────────────────────────────────────────────
    //
    // To decrypt: m = c^d mod n

    println!("\n--- Decryption ---");

    let decrypted = mod_pow(ciphertext.clone(), d.clone(), n.clone());
    println!("  Decrypted:         m = c^d mod n = {decrypted}");

    assert_eq!(
        decrypted, message,
        "decryption should recover the original message"
    );
    println!("  ✓ Round-trip verified: decrypted matches original!");

    // ── 4. Try several messages ────────────────────────────────────

    println!("\n--- Multiple Messages ---");

    let messages = [
        BigInt::from(0i64),
        BigInt::from(1i64),
        BigInt::from(12345i64),
        BigInt::from(999_999_999i64),
        BigInt::from(1_500_000_000_000i64),
    ];

    for m in &messages {
        let c = mod_pow(m.clone(), e.clone(), n.clone());
        let m_back = mod_pow(c.clone(), d.clone(), n.clone());
        let ok = &m_back == m;
        println!(
            "  m = {:<15} → c = {:<25}… → m' = {:<15} {}",
            m,
            &c.to_string()[..c.to_string().len().min(25)],
            m_back,
            if ok { "✓" } else { "✗ FAIL" }
        );
        assert_eq!(&m_back, m, "round-trip failed for m={m}");
    }

    // ── 5. Digital signature ───────────────────────────────────────
    //
    // Sign: signature = hash^d mod n  (simplified — real RSA uses proper hashing)
    // Verify: hash' = signature^e mod n, check hash' == hash

    println!("\n--- Digital Signature ---");

    let document_hash = BigInt::from(314159265i64);
    println!("  Document hash: {document_hash}");

    // Sign with private key
    let signature = mod_pow(document_hash.clone(), d.clone(), n.clone());
    println!("  Signature:     {signature}");

    // Verify with public key
    let verified_hash = mod_pow(signature.clone(), e.clone(), n.clone());
    println!("  Verified hash: {verified_hash}");

    assert_eq!(
        verified_hash, document_hash,
        "signature verification failed"
    );
    println!("  ✓ Signature verified!");

    // ── 6. Key properties ──────────────────────────────────────────

    println!("\n--- Key Properties ---");
    println!(
        "  Key size:         {} bits (toy — real RSA uses 2048+)",
        n.bits()
    );
    println!("  p is prime:       {}", isprime(p.clone()));
    println!("  q is prime:       {}", isprime(q.clone()));
    println!("  n is composite:   {}", !isprime(n.clone()));
    let n_factors = factorint(n.clone());
    println!(
        "  n factors:        {}",
        n_factors
            .iter()
            .map(|(p, e)| if *e == 1 {
                format!("{p}")
            } else {
                format!("{p}^{e}")
            })
            .collect::<Vec<_>>()
            .join(" × ")
    );

    // ── 7. Chinese Remainder Theorem speedup ───────────────────────
    //
    // CRT lets us decrypt using smaller exponentiations mod p and mod q
    // separately, then combine — this is 4x faster for large keys.

    println!("\n--- CRT-Optimized Decryption ---");

    let dp = &d % (&p - BigInt::from(1));
    let dq = &d % (&q - BigInt::from(1));
    println!("  dp = d mod (p-1) = {dp}");
    println!("  dq = d mod (q-1) = {dq}");

    let m1 = mod_pow(ciphertext.clone(), dp, p.clone());
    let m2 = mod_pow(ciphertext.clone(), dq, q.clone());
    println!("  m1 = c^dp mod p = {m1}");
    println!("  m2 = c^dq mod q = {m2}");

    // Combine via CRT
    let remainders = vec![m1, m2];
    let moduli = vec![p, q];
    let m_crt = crt(&remainders, &moduli).expect("CRT should succeed");
    println!("  m_crt = CRT(m1, m2) = {m_crt}");

    assert_eq!(
        m_crt, message,
        "CRT decryption should match standard decryption"
    );
    println!("  ✓ CRT decryption matches standard decryption!");

    println!("\n✓ Done!");
}
