//! Benchmarks for expression canonicalization.

use criterion::{criterion_group, criterion_main, Criterion};

fn canonicalization_benchmarks(_c: &mut Criterion) {
    // TODO: add benchmarks once the canonicalization module is implemented
}

criterion_group!(benches, canonicalization_benchmarks);
criterion_main!(benches);
