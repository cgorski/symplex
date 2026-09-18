//! Benchmarks disguised as tests (`tests/perf/*.rs`: the former
//! `simplify_perf_test.rs` and `perf_analysis.rs`). All `#[ignore]`d; run with
//! `cargo test --test perf --release -- --ignored --nocapture`.

#[path = "perf/perf_analysis.rs"]
mod perf_analysis;
#[path = "perf/simplify_perf_test.rs"]
mod simplify_perf_test;
