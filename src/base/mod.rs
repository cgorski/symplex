pub(crate) mod arena;
pub mod assumptions;
pub(crate) mod bernoulli;
pub(crate) mod bigcomplex;
pub(crate) mod bigfloat;
pub mod budget;
pub(crate) mod canon;
pub(crate) mod cast;
pub(crate) mod combinatorics;
pub(crate) mod compact;
pub(crate) mod complex;
pub mod config;

pub(crate) mod dense_f64;
pub mod errors;
pub mod extended;
pub(crate) mod func;
pub(crate) mod graph;
pub mod interval;
pub mod libfn;
pub(crate) mod node;
pub mod numeric;
/// The CPU-time assertions of the integration tests (`tests/perf_assert/`,
/// one file for both) for the `#[cfg(test)]` modules of `src/`:
/// `CpuTimer::start()` … `.assert_within(budget, what)`, `cpu_time`.  See
/// CONTRIBUTING.md, "Timeouts and granularity".
#[cfg(test)]
#[path = "../../tests/perf_assert/mod.rs"]
pub(crate) mod perf_assert;
pub mod rng;
pub(crate) mod sort_key;
pub(crate) mod stage;
pub(crate) mod symbol;
pub mod truth;
pub(crate) mod value_class;
pub(crate) mod walk;
