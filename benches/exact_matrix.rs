//! Benchmarks for the exact matrix core (`QMatrix` / `ZMatrix`) and the
//! `Matrix` fast paths that route rational input through it.
//!
//! Run with: `cargo bench --bench exact_matrix`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use num_bigint::BigInt;
use symplex::linprog::{LpProblem, Q, q, qi};
use symplex::prelude::*;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }

    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

fn random_z(n: usize, m: usize, seed: u64) -> ZMatrix {
    let mut g = Lcg(seed);
    ZMatrix::from_fn(n, m, |_, _| BigInt::from(g.range(-9, 9))).unwrap()
}

fn bench_qmatrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("qmatrix");
    for &n in &[10usize, 20, 40] {
        let a = random_z(n, n + n / 5, 11).to_qmatrix();
        group.bench_with_input(BenchmarkId::new("rref", n), &a, |b, a| {
            b.iter(|| black_box(a.rref()))
        });
        let s = random_z(n, n, 5).to_qmatrix();
        group.bench_with_input(BenchmarkId::new("det", n), &s, |b, s| {
            b.iter(|| black_box(s.det().unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("inv", n), &s, |b, s| {
            b.iter(|| black_box(s.inv().unwrap()))
        });
        // The row-operation HNF has exponential coefficient growth on random
        // input: 40×40 took two hours under `cargo test --benches` (debug) in
        // CI, so the largest size is measured only for the other kernels.
        if n <= 20 {
            let z = random_z(n, n, 9);
            group.bench_with_input(BenchmarkId::new("hnf", n), &z, |b, z| {
                b.iter(|| black_box(z.hermite_normal_form()))
            });
        }
    }
    group.finish();
}

/// A random `n × m` matrix of `bits`-bit entries (either sign).
fn random_wide(n: usize, m: usize, bits: u32, seed: u64) -> QMatrix {
    let mut g = Lcg(seed);
    ZMatrix::from_fn(n, m, |_, _| {
        let mut v = BigInt::from(0);
        let mut left = bits;
        while left > 0 {
            let take = left.min(50);
            v = (v << take) + BigInt::from(g.next() & ((1u64 << take) - 1));
            left -= take;
        }
        if g.next() & 1 == 1 { -v } else { v }
    })
    .unwrap()
    .to_qmatrix()
}

/// The fraction-free kernel's width escalation, one shape per stage it
/// ends on: `qmatrix/rref/*` above stays on `i64` (10), reaches `i128`
/// (20) and the 256-bit cells (40); these start beyond `i64` at once and
/// end on `i128`, on the 256-bit cells, or on `BigInt` (entries past 255
/// bits, where the kernel runs on `BigInt` from the first pivot).
fn bench_kernel_stages(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel");
    for &(name, n, bits) in &[
        ("rref_12x14_40bit_i128", 12usize, 40u32),
        ("rref_12x14_90bit_w256", 12, 90),
        ("rref_12x14_300bit_bigint", 12, 300),
    ] {
        let a = random_wide(n, n + 2, bits, 17);
        group.bench_with_input(BenchmarkId::new(name, n), &a, |b, a| {
            b.iter(|| black_box(a.rref()))
        });
        let s = random_wide(n, n, bits, 19);
        let dname = name.replacen("rref", "det", 1);
        group.bench_with_input(BenchmarkId::new(dname, n), &s, |b, s| {
            b.iter(|| black_box(s.det().unwrap()))
        });
    }
    group.finish();
}

fn bench_matrix_fast_paths(c: &mut Criterion) {
    let ctx = Context::new();
    let mut group = c.benchmark_group("matrix_rational");
    for &n in &[10usize, 20] {
        let a = random_z(n, n + n / 5, 11).to_matrix(&ctx);
        group.bench_with_input(BenchmarkId::new("rref", n), &a, |b, a| {
            b.iter(|| black_box(a.rref()))
        });
        let s = random_z(n, n, 5).to_matrix(&ctx);
        group.bench_with_input(BenchmarkId::new("inv", n), &s, |b, s| {
            b.iter(|| black_box(s.inv().unwrap()))
        });
        let rhs = random_z(n, 1, 6).to_matrix(&ctx);
        group.bench_with_input(
            BenchmarkId::new("linsolve_matrix", n),
            &(s, rhs),
            |b, (s, rhs)| b.iter(|| black_box(linsolve_matrix(s, rhs).unwrap())),
        );
    }
    group.finish();
}

fn bench_linprog(c: &mut Criterion) {
    let mut group = c.benchmark_group("linprog");
    group.sample_size(10);
    for &(m, n) in &[(10usize, 25usize), (20, 50), (40, 100)] {
        let mut g = Lcg(42);
        let cost: Vec<Q> = (0..n).map(|_| qi(g.range(1, 9))).collect();
        let mut p = LpProblem::maximize(cost);
        for _ in 0..m {
            let row: Vec<Q> = (0..n).map(|_| q(g.range(0, 9), g.range(1, 4))).collect();
            p = p.le(row, qi(g.range(50, 500)));
        }
        group.bench_with_input(
            BenchmarkId::new("maximize", format!("{m}x{n}")),
            &p,
            |b, p| b.iter(|| black_box(p.solve().unwrap())),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_qmatrix,
    bench_kernel_stages,
    bench_matrix_fast_paths,
    bench_linprog
);
criterion_main!(benches);
