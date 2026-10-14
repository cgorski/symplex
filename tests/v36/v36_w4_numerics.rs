//! Tests of the 0.49 refactor (w4): checked float-to-integer conversions and the BigFloat boundary.
//! Every test says what was wrong before and cites its oracle.
//!
//! The second half is a small deterministic property test of the
//! astro-float primitives symplex builds on (the large run, 11.2 million
//! exact cases and 1.05 million mpmath cases, is in the 0.49 W4 report).
//! Measured semantics of astro-float 0.9.6, which the library documents
//! differently ("round half toward …"): `Up`, `Down`, `ToZero` and
//! `FromZero` are directed roundings, `ToEven`/`ToOdd` round to nearest,
//! `None` truncates.

use std::cmp::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use astro_float::{BigFloat, Consts, RoundingMode, Sign, WORD_BIT_SIZE};
use num_bigint::{BigInt, BigUint, Sign as BSign};
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};
use symplex::prelude::*;

/// Runs `f` on its own thread and fails the test if it has not returned
/// after `secs` seconds (the thread is abandoned).
fn within<T: Send + 'static>(secs: u64, what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .unwrap_or_else(|_| panic!("{what} did not return within {secs} s"))
}

fn definite(f: &str, lo: i64, hi: i64) -> String {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(f).unwrap();
    f.integrate_definite(&x, &ctx.int(lo), &ctx.int(hi))
        .to_string()
}

/// S4-B1: 0.48 returned `0` for `∫₋₁¹ floor(10¹⁹x)/10¹⁹ dx`.  `t.floor() as
/// i64` saturated at `i64::MIN`, the `− 1` wrapped to `i64::MAX`, the loop
/// over the jumps was empty, and the antiderivative was evaluated across
/// 2·10¹⁹ jumps.  Oracle, an exact identity: `floor(t) + floor(−t) = −1` off
/// the integers, so `∫₋₁¹ floor(Nx) dx = ∫₀¹ (floor(Nx) + floor(−Nx)) dx = −1`
/// and the integral is `−1/10¹⁹`.  Now refused (more than 10 000 jumps);
/// `N = 10` is still evaluated, to `−1/10`.
#[test]
fn floor_with_slope_beyond_i64_is_not_integrated_to_zero() {
    let r = definite("floor(10^19*x)/10^19", -1, 1);
    assert!(
        r == "-1/10000000000000000000" || r.starts_with("Integral("),
        "{r}"
    );
    assert_eq!(definite("floor(10*x)/10", -1, 1), "-1/10");
}

/// S4-B2: 0.48 never returned, with memory growing until killed, on
/// `∫₋₁¹ floor(9·10¹⁸x)/10¹⁹ dx`: both ends fit `i64` but the length guard
/// `n_max − n_min > 10 000` wrapped negative and the loop visited 1.8·10¹⁹
/// candidate jumps.  The same scan places the poles of Γ, ψ and lnΓ:
/// `∫₋₁¹ Γ(9·10¹⁸x) dx` and `∫₋₁¹ 1/Γ(9·10¹⁸x) dx` hung too (killed at a 3 GB
/// heap cap on the 0.48 tree).  Oracle for the floor case: `−1/10¹⁹` (the
/// identity of the previous test).  Now each is refused at once.
#[test]
fn breakpoint_scans_with_huge_slopes_return() {
    for f in [
        "floor(9*10^18*x)/10^19",
        "ceiling(9*10^18*x)/10^19",
        "gamma(9*10^18*x)",
        "1/gamma(9*10^18*x)",
    ] {
        let r = within(60, f, move || definite(f, -1, 1));
        assert!(
            r.starts_with("Integral(") || r == "-1/10000000000000000000",
            "{f}: {r}"
        );
    }
}

// ── astro-float property test ─────────────────────────────────────────────

type Q = Ratio<BigInt>;

const MODES: [RoundingMode; 7] = [
    RoundingMode::None,
    RoundingMode::Up,
    RoundingMode::Down,
    RoundingMode::ToZero,
    RoundingMode::FromZero,
    RoundingMode::ToEven,
    RoundingMode::ToOdd,
];

/// SplitMix64: a fixed, dependency-free stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

/// The exact value `m·2^e` of a finite BigFloat.
#[derive(Clone, Debug)]
struct Dyadic {
    m: BigInt,
    e: i64,
}

fn to_dyadic(x: &BigFloat) -> Option<Dyadic> {
    if x.is_zero() {
        return Some(Dyadic {
            m: BigInt::zero(),
            e: 0,
        });
    }
    let (words, _, sign, exp, _) = x.as_raw_parts()?;
    let mut m = BigUint::zero();
    for &w in words.iter().rev() {
        m = (m << WORD_BIT_SIZE) + BigUint::from(w);
    }
    let s = if sign == Sign::Neg {
        BSign::Minus
    } else {
        BSign::Plus
    };
    Some(Dyadic {
        m: BigInt::from_biguint(s, m),
        e: i64::from(exp) - (WORD_BIT_SIZE * words.len()) as i64,
    })
}

/// A BigFloat `±0.m × 2^exp` from 64-bit mantissa words, most significant
/// first (top bit set).
fn make_bf(top_first: &[u64], neg: bool, exp: i32) -> BigFloat {
    let mut m = BigUint::zero();
    for &w in top_first {
        m = (m << 64usize) + BigUint::from(w);
    }
    // Rebuild in the library's word size (32 bits on wasm32).
    let bits = 64 * top_first.len();
    let words: Vec<astro_float::Word> = (0..bits / WORD_BIT_SIZE)
        .map(|i| {
            let w = (&m >> (i * WORD_BIT_SIZE)) & BigUint::from(u64::MAX >> (64 - WORD_BIT_SIZE));
            w.to_u64().unwrap_or(0) as astro_float::Word
        })
        .collect();
    let s = if neg { Sign::Neg } else { Sign::Pos };
    BigFloat::from_raw_parts(&words, bits, s, exp, false)
}

fn pow2(k: i64) -> Q {
    if k >= 0 {
        Q::from_integer(BigInt::one() << (k as usize))
    } else {
        Q::new(BigInt::one(), BigInt::one() << ((-k) as usize))
    }
}

fn floor_log2(a: &Q) -> i64 {
    let l0 = a.numer().bits() as i64 - a.denom().bits() as i64;
    if *a >= pow2(l0) { l0 } else { l0 - 1 }
}

/// astro-float's rounding of the magnitude `n + frac` (`half`: `frac`
/// against 1/2; `exact`: `frac = 0`), as measured.
fn apply_rule(n: BigInt, exact: bool, half: Ordering, neg: bool, rm: RoundingMode) -> BigInt {
    if exact {
        return n;
    }
    let up = match rm {
        RoundingMode::None | RoundingMode::ToZero => false,
        RoundingMode::FromZero => true,
        RoundingMode::Up => !neg,
        RoundingMode::Down => neg,
        _ if half != Ordering::Equal => half == Ordering::Greater,
        RoundingMode::ToEven => n.bit(0),
        _ => !n.bit(0),
    };
    if up { n + 1 } else { n }
}

/// The expected result `m·2^e` of rounding the nonzero `q·2^e0` to `p` bits.
fn expect_round(q: &Q, e0: i64, p: usize, rm: RoundingMode) -> Dyadic {
    let neg = q.is_negative();
    let a = q.abs();
    let t = p as i64 - 1 - floor_log2(&a);
    let s = &a * pow2(t);
    let n = s.numer() / s.denom();
    let rem = s.numer() - &n * s.denom();
    let half = (&rem * BigInt::from(2)).cmp(s.denom());
    let r = apply_rule(n, rem.is_zero(), half, neg, rm);
    Dyadic {
        m: if neg { -r } else { r },
        e: e0 - t,
    }
}

/// The expected result of rounding `sqrt(m·2^e)` (`m > 0`) to `p` bits.
fn expect_sqrt(m: &BigInt, e: i64, p: usize, rm: RoundingMode) -> Dyadic {
    let (mut m, mut e) = (m.clone(), e);
    if e.rem_euclid(2) == 1 {
        m <<= 1usize;
        e -= 1;
    }
    let mu = m.to_biguint().unwrap();
    let ls = (mu.bits() as i64 - 1) / 2;
    let t = 2 * (p as i64 - ls - 1);
    // n = floor(sqrt(m·2^t)); compare 4·m·2^t with (2n + 1)².
    let (n, exact, half) = if t >= 0 {
        let s = &mu << (t as usize);
        let n = s.sqrt();
        let two_n1 = &n * 2u32 + 1u32;
        (
            n.clone(),
            &n * &n == s,
            (s * 4u32).cmp(&(&two_n1 * &two_n1)),
        )
    } else {
        let sh = (-t) as usize;
        let n = (&mu >> sh).sqrt();
        let two_n1 = &n * 2u32 + 1u32;
        let exact = (&n * &n) << sh == mu;
        (n, exact, (&mu * 4u32).cmp(&((&two_n1 * &two_n1) << sh)))
    };
    Dyadic {
        m: apply_rule(BigInt::from(n), exact, half, false, rm),
        e: e / 2 - t / 2,
    }
}

fn same_value(a: &Dyadic, b: &Dyadic) -> bool {
    if a.m.is_zero() || b.m.is_zero() {
        return a.m.is_zero() && b.m.is_zero();
    }
    let d = a.e - b.e;
    if d.abs() > 100_000 {
        return false;
    }
    if d >= 0 {
        (&a.m << (d as usize)) == b.m
    } else {
        a.m == (&b.m << ((-d) as usize))
    }
}

/// The exact sum; an operand far below the other's last bit (and below the
/// finest rounding position of any precision used) only decides a side, so
/// it is replaced by a signed epsilon there.
fn exact_add(a: &Dyadic, b: &Dyadic) -> Dyadic {
    if a.m.is_zero() {
        return b.clone();
    }
    if b.m.is_zero() {
        return a.clone();
    }
    let top = |d: &Dyadic| d.e + d.m.bits() as i64;
    let (big, small) = if top(a) >= top(b) { (a, b) } else { (b, a) };
    let floor_bit = big.e.min(top(big) - 1024);
    if top(small) < floor_bit - 2 {
        let eps = if small.m.is_negative() {
            -BigInt::one()
        } else {
            BigInt::one()
        };
        return Dyadic {
            m: (&big.m << ((big.e - floor_bit + 2) as usize)) + eps,
            e: floor_bit - 2,
        };
    }
    let e = a.e.min(b.e);
    Dyadic {
        m: (&a.m << ((a.e - e) as usize)) + (&b.m << ((b.e - e) as usize)),
        e,
    }
}

fn neg_dyadic(d: &Dyadic) -> Dyadic {
    Dyadic {
        m: -d.m.clone(),
        e: d.e,
    }
}

/// A random operand of `words` 64-bit words: a signed zero, a small integer,
/// a sparse mantissa, a value next to a power of two (0.5, 1, 2, …) or a full
/// random mantissa; exponents mostly moderate, some up to `±2²⁹`.
fn gen_bf(rng: &mut Rng, words: usize, extreme: bool) -> BigFloat {
    let neg = rng.below(2) == 0;
    let exp = match rng.below(10) {
        0..=6 => rng.range(-64, 64),
        7 | 8 => rng.range(-2000, 2000),
        _ if extreme => rng.range(-(1 << 29), 1 << 29),
        _ => rng.range(-20_000, 20_000),
    } as i32;
    let top = 1u64 << 63;
    let mut w = vec![0u64; words];
    match rng.below(100) {
        0..=4 => {
            let z = BigFloat::from_i32(0, words * 64);
            return if neg { z.neg() } else { z };
        }
        5..=14 => {
            let n = rng.range(1, 1000);
            return BigFloat::from_i64(if neg { -n } else { n }, words * 64);
        }
        15..=44 => {
            w[0] = top;
            for _ in 0..rng.below(4) {
                let bit = rng.below((words * 64) as u64) as usize;
                w[bit / 64] |= top >> (bit % 64);
            }
        }
        45..=59 => {
            if rng.below(2) == 0 {
                w[0] = top;
                w[words - 1] |= rng.below(4);
            } else {
                w.fill(u64::MAX);
                w[words - 1] -= rng.below(4);
            }
        }
        _ => {
            for x in &mut w {
                *x = rng.next();
            }
            w[0] |= top;
        }
    }
    make_bf(&w, neg, exp)
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
    Powi(usize),
    Reciprocal,
    Cmp,
}

/// The rounding of the exact value `q·2^e` that `r` should be.
fn judge(r: &BigFloat, exact: (Q, i64), rm: RoundingMode, desc: &str) -> Result<(), String> {
    let (q, e) = exact;
    let got = to_dyadic(r).ok_or_else(|| format!("{desc}: got {r:?}, want a finite value"))?;
    if q.is_zero() {
        return if got.m.is_zero() {
            Ok(())
        } else {
            Err(format!("{desc}: got {r:?}, want 0"))
        };
    }
    // Results next to astro-float's exponent limits (±2³¹) are not judged.
    if (floor_log2(&q.abs()) + e).abs() > (1i64 << 31) - 8192 {
        return Ok(());
    }
    let p = r.mantissa_max_bit_len().unwrap_or(64);
    let want = expect_round(&q, e, p, rm);
    if same_value(&got, &want) {
        Ok(())
    } else {
        Err(format!("{desc}: got {r:?}, want {want:?}"))
    }
}

/// One case: `Ok(())`, or the reason it failed (division by zero is not a
/// case).
fn check_case(
    op: Op,
    a: &BigFloat,
    b: &BigFloat,
    p: usize,
    rm: RoundingMode,
) -> Result<(), String> {
    let (da, db) = (to_dyadic(a).unwrap(), to_dyadic(b).unwrap());
    let desc = format!("{op:?} {rm:?} p={p} a={a:?} b={b:?}");
    let int = |d: Dyadic| (Q::from_integer(d.m), d.e);
    match op {
        Op::Cmp => {
            let want = exact_add(&da, &neg_dyadic(&db)).m.cmp(&BigInt::zero());
            let got = (a.partial_cmp(b), a.cmp(b).map(|s| s.cmp(&0)));
            if got == (Some(want), Some(want)) {
                Ok(())
            } else {
                Err(format!("{desc}: want {want:?}, got {got:?}"))
            }
        }
        Op::Add => judge(&a.add(b, p, rm), int(exact_add(&da, &db)), rm, &desc),
        Op::Sub => judge(
            &a.sub(b, p, rm),
            int(exact_add(&da, &neg_dyadic(&db))),
            rm,
            &desc,
        ),
        Op::Mul => judge(
            &a.mul(b, p, rm),
            (Q::from_integer(&da.m * &db.m), da.e + db.e),
            rm,
            &desc,
        ),
        Op::Div if db.m.is_zero() => Ok(()),
        Op::Div => judge(
            &a.div(b, p, rm),
            (Q::new(da.m, db.m), da.e - db.e),
            rm,
            &desc,
        ),
        Op::Sqrt => {
            let r = a.abs().sqrt(p, rm);
            if da.m.is_zero() {
                return judge(&r, (Q::zero(), 0), rm, &desc);
            }
            let want = expect_sqrt(&da.m.abs(), da.e, r.mantissa_max_bit_len().unwrap_or(p), rm);
            match to_dyadic(&r) {
                Some(got) if same_value(&got, &want) => Ok(()),
                _ => Err(format!("{desc}: got {r:?}, want {want:?}")),
            }
        }
        Op::Powi(0) => judge(&a.powi(0, p, rm), (Q::one(), 0), rm, &desc),
        Op::Powi(n) => {
            let want = (Q::from_integer(da.m.pow(n as u32)), da.e * n as i64);
            judge(&a.powi(n, p, rm), want, rm, &desc)
        }
        Op::Reciprocal if da.m.is_zero() => Ok(()),
        Op::Reciprocal => judge(
            &a.reciprocal(p, rm),
            (Q::new(BigInt::one(), da.m), -da.e),
            rm,
            &desc,
        ),
    }
}

/// A described check, run by [`run_with_deadline`].
type Case = (String, Box<dyn FnOnce() -> Result<(), String> + Send>);

/// Runs `cases` on a worker thread, each with a 2 s deadline: astro-float's
/// `pow` and `log` never return on some inputs, and a primitive that hangs
/// must fail the test, not stall it.
fn run_with_deadline(cases: Vec<Case>) -> Vec<String> {
    let (tx, rx) = mpsc::channel::<(String, Option<Result<(), String>>)>();
    std::thread::spawn(move || {
        for (desc, case) in cases {
            if tx.send((desc, None)).is_err() {
                return;
            }
            let out = case();
            if tx.send((String::new(), Some(out))).is_err() {
                return;
            }
        }
    });
    let mut failures = Vec::new();
    let mut current = String::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(2)) {
            Ok((desc, None)) => current = desc,
            Ok((_, Some(Err(e)))) => failures.push(e),
            Ok((_, Some(Ok(())))) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return failures,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                failures.push(format!("{current}: did not return within 2 s"));
                return failures;
            }
        }
    }
}

/// astro-float 0.9.6 `add`, `sub`, `mul`, `div`, `sqrt`, `powi`,
/// `reciprocal` and comparison against exact rational arithmetic, in every
/// rounding mode: 784 random cases (signed zeros, small integers, sparse
/// mantissas, values next to powers of two, exponents up to `±2²⁹`) and the
/// named corner cases (ties, exact results, zero on either side of `cmp`,
/// the class of upstream bug #44).  Oracle: the exact value rounded by hand
/// (`expect_round`/`expect_sqrt`, num-rational).  Nothing was wrong before:
/// this pins the library's behaviour, which `base::bigfloat` documents and
/// evalf's error bounds rely on.
#[test]
fn astro_float_arithmetic_is_correctly_rounded() {
    let mut cases: Vec<Case> = Vec::new();
    let mut rng = Rng(0x5EED_0049);
    let ops = [
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Sqrt,
        Op::Powi(0),
        Op::Reciprocal,
        Op::Cmp,
    ];
    for op in ops {
        for rm in MODES {
            for _ in 0..14 {
                let (wa, wb) = (1 + rng.below(4) as usize, 1 + rng.below(4) as usize);
                let p = 64 * (1 + rng.below(4) as usize);
                let extreme = !matches!(op, Op::Powi(_));
                let (a, b) = (gen_bf(&mut rng, wa, extreme), gen_bf(&mut rng, wb, extreme));
                let op = match op {
                    Op::Powi(_) => Op::Powi(rng.below(13) as usize),
                    op => op,
                };
                cases.push((
                    format!("{op:?} {rm:?}"),
                    Box::new(move || check_case(op, &a, &b, p, rm)),
                ));
            }
        }
    }
    // Named corner cases, in every mode.
    let f = |x: f64| BigFloat::from_f64(x, 64);
    let tiny = make_bf(&[1 << 63], false, -(1 << 29));
    let huge = make_bf(&[1 << 63], false, 1 << 29);
    let named: Vec<(Op, BigFloat, BigFloat)> = vec![
        // 1 ± 2⁻⁶⁴ is a tie at 64 bits; 1 + 2⁻⁶³ + 2⁻⁶⁴ another (odd neighbour).
        (Op::Add, f(1.0), f(2f64.powi(-64))),
        (Op::Add, f(-1.0), f(-(2f64.powi(-64)))),
        (
            Op::Add,
            make_bf(&[(1 << 63) | 1], false, 1),
            f(2f64.powi(-64)),
        ),
        (Op::Sub, f(1.0), f(2f64.powi(-66))),
        // Exact results.
        (Op::Sqrt, f(0.25), f(0.0)),
        (Op::Reciprocal, f(2.0), f(0.0)),
        (Op::Powi(3), f(0.5), f(0.0)),
        (Op::Div, f(1.0), f(3.0)),
        (Op::Mul, huge.clone(), tiny.clone()),
        // Zero on either side.
        (Op::Add, f(0.0), f(-1.5)),
        (Op::Add, f(2.5), f(-2.5)),
        (Op::Mul, f(0.0).neg(), f(3.0)),
        (Op::Div, f(0.0), f(-7.0)),
        (Op::Sqrt, f(0.0), f(0.0)),
        (Op::Cmp, f(0.0), tiny.clone()),
        (Op::Cmp, tiny.clone(), f(0.0)),
        (Op::Cmp, f(0.0), tiny.neg()),
        (Op::Cmp, tiny.neg(), f(0.0)),
        (Op::Cmp, f(0.0).neg(), f(0.0)),
        (Op::Cmp, huge.neg(), f(0.0).neg()),
        (Op::Cmp, f(0.5), make_bf(&[(1 << 63) | 1], false, 0)),
    ];
    for (op, a, b) in named {
        for rm in MODES {
            for p in [64, 128] {
                let (a, b) = (a.clone(), b.clone());
                cases.push((
                    format!("{op:?} {rm:?}"),
                    Box::new(move || check_case(op, &a, &b, p, rm)),
                ));
            }
        }
    }
    let failures = run_with_deadline(cases);
    assert!(
        failures.is_empty(),
        "{} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// `f(m·2^e)` to 256 bits as `(name, m, e, M, E)`, value `M·2^E`.  Oracle:
/// mpmath 1.3 at 600 bits, `M = nint(ldexp(f(ldexp(m, e)), 256 − k))`
/// (generator in the 0.49 W4 notes).
#[rustfmt::skip]
const TRANSCENDENTAL_REFS: &[(&str, i64, i64, &str, i64)] = &[
    ("exp", 1, -1, "95454440252185294983886966005124823429889078541787437355320193447306720442726", -255),
    ("exp", -3, -2, "109392620055422004122165636559352771634699419578393360891461908533673082809366", -257),
    ("exp", 13, -2, "93322416911010531982193812912365621667007410871246672581955111201369012513147", -251),
    ("exp", 1, -100, "57896044618658097711785492504389625852801583049014147170751193862719322193920", -255),
    ("exp", 801, -3, "79079610081053998509779055807933663233391535310882719355173021416715356755426", -111),
    ("exp", -15, -1, "65579821817776914312664714465818183888865466740249459967701025572823393030829", -266),
    ("exp", 3602879701896397, -55, "63985024784151028392850054603036435117232931914466217645204010493357300203050", -255),
    ("exp", -5, 3, "70893981353923957076997963707765917535320597212511661276234500681006323982879", -313),
    ("exp", 7, 0, "62002658467455071871257807570305645407377122216438165465771743699718318920357", -245),
    ("ln", 1, -1, "-80260960185991308862233904206310070533990667611589946606122867505419956976172", -256),
    ("ln", 13, -2, "68239462258200240530156000535905372407671454009327127545918930900226512531018", -255),
    ("ln", 1, -100, "-62703875145305710048620237661179742604680209071554645786033490238609341387634", -249),
    ("ln", 801, -3, "66673365856656612064642257285058155191181011739482499242678374402246822837829", -253),
    ("ln", 2097153, -1, "100326203683361486721057159184349484411903292415224210557535750445544340976934", -252),
    ("ln", 3602879701896397, -55, "-66655284641120135448625320594805415917006861080211897971514940196210398124061", -254),
    ("ln", 7, 0, "112660500813606048678782017959702492548347539027662435018932387300412814483741", -255),
    ("sin", 1, -1, "111027369497412508784314791148902648293994217576031053546226341313477387335312", -257),
    ("sin", -3, -2, "-78928376128235469093094568120636126943009786161916276723913149103058679254193", -256),
    ("sin", 13, -2, "-100225125380429920507241870340658379426987117019111406535626592735756909424516", -259),
    ("sin", 1, -100, "115792089237316195423570985008687907853269984665640564039457571998314123318613", -356),
    ("sin", 801, -3, "-91453850132583581994043476117267544771097629343257764718360650774535851420287", -257),
    ("sin", -15, -1, "-108612977015301080586905849516129966081692803571855865750987623675147135809979", -256),
    ("sin", 2097153, -1, "85978042155599370382192602017124607370761425527441189692415758172875745287220", -256),
    ("sin", 3602879701896397, -55, "92479359113885553871533267772861606806942099316434124776262051275384197576225", -259),
    ("sin", -5, 3, "-86278209570123457809914634589167438252065347435379097751452238443414292489935", -256),
    ("sin", 7, 0, "76073850866566872379693726844081292249867925337799757138455117534817261026956", -256),
    ("cos", 1, -1, "101617118319522600545601981648807607350213579319835970884288805016705398675944", -256),
    ("cos", -3, -2, "84723782798588416432790597400864961822150914155529938649700334111839829194179", -256),
    ("cos", 13, -2, "-115112352166182851487357552667718280813052010629696150277933871081064130707481", -256),
    ("cos", 1, -100, "115792089237316195423570985008687907853269984665640564039457547979116110675968", -256),
    ("cos", 801, -3, "106380713731034595766879597608141580365903195742700889639062843409490726639251", -256),
    ("cos", -15, -1, "80275255311117541698743097697784983823961813103993384425412411398314687960087", -257),
    ("cos", 2097153, -1, "77560197247251600008455909746428099868367323453118259480064444340605479939953", -256),
    ("cos", 3602879701896397, -55, "115213611097374472122247136373863334800459466102705645575940960144633601786872", -256),
    ("cos", -5, 3, "-77226151550601393910712827356673171956128418116933784851805084792647722546777", -256),
    ("cos", 7, 0, "87295917111133781935764738923471732212450910431832086487066734933178544079901", -256),
    ("atan", 1, -1, "107373450632179258931464468992502143670939545571605869416383036103061999339274", -257),
    ("atan", -3, -2, "-74512337813703903208653002396362787655757143092590345563003039455907447893819", -256),
    ("atan", 13, -2, "73660986761206374618732266610080875815763918831540109221831758484732571110585", -255),
    ("atan", 1, -100, "115792089237316195423570985008687907853269984665640564039457559988715116997291", -356),
    ("atan", 801, -3, "90364675799157810293586994867614582360577524110380805911031048752095357602359", -255),
    ("atan", -15, -1, "-83268684794821842061516012371184313900125316458050336062859612364874943267342", -255),
    ("atan", 2097153, -1, "90942839008997134822650292044396561695276470181819020477904755914466404272930", -255),
    ("atan", 3602879701896397, -55, "92326732027357529889865353663624595705775734052357208212438545407879424129111", -259),
    ("atan", -5, 3, "-89495794536346428810080467995690173982649426184828415957007055810629873903337", -255),
    ("atan", 7, 0, "82727616018322742139355869045397626659552743712344226526348038617696085755182", -255),
];

fn bigfloat_of(m: i64, e: i64, p: usize) -> BigFloat {
    let neg = m < 0;
    let mag = m.unsigned_abs();
    let shift = mag.leading_zeros();
    // 0.(mag << shift) × 2^(64 − shift + e) = mag·2^e.
    make_bf(&[mag << shift], neg, (64 - i64::from(shift) + e) as i32).add(
        &BigFloat::from_i32(0, p),
        p,
        RoundingMode::None,
    )
}

/// astro-float 0.9.6 `exp`, `ln`, `sin`, `cos`, `atan` at 128 bits against
/// mpmath, in every mode: within half an ulp for `ToEven`/`ToOdd`, within
/// one ulp on the mode's side for the directed modes and `None` (which may
/// land on either side: 3 of 25 000 `exp` results in the large run lay
/// above the true value); exact at the exactly representable points
/// (`exp(0)`, `ln(1)`, `sin(0)`, `cos(0)`, `atan(0)`).  Every call has a 2 s
/// deadline.  Nothing was wrong before; see the previous test.
#[test]
fn astro_float_transcendentals_are_within_an_ulp() {
    let p = 128;
    let mut cases: Vec<Case> = Vec::new();
    for &(name, m, e, big, big_e) in TRANSCENDENTAL_REFS {
        let reference = (Q::from_integer(big.parse::<BigInt>().unwrap()), big_e);
        for rm in MODES {
            let reference = reference.clone();
            let desc = format!("{name}({m}·2^{e}) {rm:?}");
            let d2 = desc.clone();
            cases.push((
                desc,
                Box::new(move || {
                    let mut cc = Consts::new().map_err(|e| format!("{e:?}"))?;
                    let x = bigfloat_of(m, e, p);
                    let r = match name {
                        "exp" => x.exp(p, rm, &mut cc),
                        "ln" => x.ln(p, rm, &mut cc),
                        "sin" => x.sin(p, rm, &mut cc),
                        "cos" => x.cos(p, rm, &mut cc),
                        _ => x.atan(p, rm, &mut cc),
                    };
                    let got = to_dyadic(&r).ok_or_else(|| format!("{d2}: {r:?}"))?;
                    let (q, qe) = reference;
                    // The error in ulps of the result's binade.
                    let k = floor_log2(&q.abs()) + 1 + qe;
                    let err = (Q::from_integer(got.m) * pow2(got.e - (k - p as i64))
                        - q * pow2(qe - (k - p as i64)))
                    .to_f64()
                    .unwrap_or(f64::INFINITY);
                    let negative = r.is_negative();
                    let ok = match rm {
                        RoundingMode::ToEven | RoundingMode::ToOdd => err.abs() <= 0.5 + 1e-30,
                        RoundingMode::Up => (-1e-30..=1.0).contains(&err),
                        RoundingMode::Down => (-1.0..=1e-30).contains(&err),
                        RoundingMode::ToZero => {
                            err.abs() <= 1.0 && (err == 0.0 || (err < 0.0) != negative)
                        }
                        RoundingMode::FromZero => {
                            err.abs() <= 1.0 && (err == 0.0 || (err > 0.0) != negative)
                        }
                        RoundingMode::None => err.abs() <= 1.0,
                    };
                    if ok {
                        Ok(())
                    } else {
                        Err(format!("{d2}: error {err} ulps"))
                    }
                }),
            ));
        }
    }
    for (name, x, want) in [
        ("exp", 0, 1),
        ("ln", 1, 0),
        ("sin", 0, 0),
        ("cos", 0, 1),
        ("atan", 0, 0),
    ] {
        for rm in MODES {
            let desc = format!("{name}({x}) {rm:?}");
            let d2 = desc.clone();
            cases.push((
                desc,
                Box::new(move || {
                    let mut cc = Consts::new().map_err(|e| format!("{e:?}"))?;
                    let x = BigFloat::from_i32(x, p);
                    let r = match name {
                        "exp" => x.exp(p, rm, &mut cc),
                        "ln" => x.ln(p, rm, &mut cc),
                        "sin" => x.sin(p, rm, &mut cc),
                        "cos" => x.cos(p, rm, &mut cc),
                        _ => x.atan(p, rm, &mut cc),
                    };
                    if r == BigFloat::from_i32(want, p) {
                        Ok(())
                    } else {
                        Err(format!("{d2}: {r:?}"))
                    }
                }),
            ));
        }
    }
    let failures = run_with_deadline(cases);
    assert!(
        failures.is_empty(),
        "{} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Scales of values far outside the `BigFloat` exponent range are `i64`
/// exponents; adding terms at such scales subtracted them unchecked.  The
/// 0.49 refactor's fuzz campaign (`fuzz_roundtrip`, built with debug
/// assertions as CI builds it) found `sin(1/sinh(3554090000043952550))`
/// panicking with "attempt to subtract with overflow" in
/// `evalf::extended::part_sum` (`kmax − window`, `k − kmax`); a release
/// build wrapped silently.  `1/sinh(N) = 2e^{−N}/(1 − e^{−2N})` with
/// N = 3554090000043952550 is 1.7133750762637643412·10^−1543521675206616638
/// (mpmath at 40 and 60 digits, via `log10(2) − N·log10(e)`), and `sin` of
/// it is the same number to all printed digits.  That is below the
/// exponent range evalf can print, so the documented answer is the
/// underflow refusal; the test (built with overflow checks) pins that the
/// sum no longer overflows on the way there.
#[test]
fn sums_at_scales_far_below_the_exponent_range_do_not_overflow() {
    let ctx = Context::new();
    // coth(N) for N ≈ 3.4·10¹⁸ is 1 to every printed digit, so the second
    // case is sin(1) = 0.841470984807897 (mpmath).
    for (s, want) in [
        ("sin(1/sinh(3554090000043952550))", "1.71337507626376"),
        (
            "sin(cosh(455555000300023003000033/133333)/sinh(455555000300023003000033/133333))",
            "0.84147098480789",
        ),
    ] {
        let e = ctx.parse(s).unwrap();
        let shown = within(60, s, move || e.eval_decimal(15).map_err(|e| e.to_string()));
        match shown {
            Ok(v) => assert!(v.starts_with(want), "{s}: {v}"),
            Err(why) => assert!(why.contains("underflows"), "{s}: {why}"),
        }
    }
}

/// `Γ(x)` above the exponent range is `e^{ln Γ(x)}` with `ln Γ(x)` computed
/// at the bits of its integer part, `log₂(x·ln x)`; `x·ln x` overflows `f64`
/// above `x ≈ 2.5·10³⁰⁵`, the cast to `usize` saturated and `prec +
/// int_bits` overflowed ("attempt to add with overflow" with debug
/// assertions, a wrapped precision in release).  Found by the 0.49
/// release's local `fuzz_parser` campaign (built with debug assertions as
/// CI builds it) on a parse of huge double factorials.  `ln Γ(3·10³⁰⁵)` is
/// 2.1004…·10³⁰⁸ (mpmath `loggamma(mpf('3e305'))`), so `Γ(3·10³⁰⁵)` is far
/// beyond the printable range: the documented answer is a refusal.
#[test]
fn gamma_beyond_the_exponent_range_does_not_overflow_its_precision() {
    let ctx = Context::new();
    let _ = within(60, "fuzz_parser input", || {
        let ctx = Context::new();
        ctx.parse("na+c\r0**e**-17!!**170!!*07!*-9**-7023!!7!2!!7!!0**-9<-\u{1c}!2a")
            .map(|e| e.to_string())
            .map_err(|e| e.to_string())
    });
    for s in [
        "gamma(3*10^305)",
        "factorial(3*10^305)",
        "log(gamma(10^306))",
    ] {
        let e = ctx.parse(s).unwrap();
        let _ = within(60, s, move || e.eval_decimal(15).map_err(|e| e.to_string()));
    }
}
