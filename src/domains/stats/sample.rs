//! The deterministic generator behind [`Distribution::sample`](super::Distribution::sample)
//! (SplitMix64): seeded and reproducible, so Monte-Carlo sanity checks of
//! exact results are repeatable — and the primitive `f64` draws (standard
//! normal, standard gamma, Poisson) the family samplers are built from.

use crate::output::codegen::numeric_rt::lgamma;

/// A deterministic pseudo-random generator (SplitMix64).  Not
/// cryptographic; seeded, reproducible, and good enough for Monte-Carlo
/// sanity checks of exact results.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// A generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `0..n` (`n > 0`; `0` for `n == 0`).
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Primitive draws behind the family samplers
// ═══════════════════════════════════════════════════════════════════════════
//
// Every routine here is exact in distribution (no normal approximation):
// a rejection or transformation of uniforms whose output law is the target
// law.  Parameters are already validated `f64`s; the families convert their
// `Ex` parameters once when the sampler is built.

/// Uniform in `(0, 1]` — for logarithms and roots that must not see `0`.
pub(crate) fn positive_uniform(rng: &mut Rng) -> f64 {
    1.0 - rng.next_f64()
}

/// A standard normal variate by Marsaglia's polar method: a point uniform
/// in the unit disc, `(u, v)` with `s = u² + v² ∈ (0, 1)`, gives
/// `u √(−2 ln s / s) ~ N(0, 1)` (Marsaglia & Bray 1964; Knuth TAOCP 2,
/// §3.4.1 algorithm P).  The second variate `v √(…)` is discarded so the
/// draw is a pure function of the stream.
pub(crate) fn standard_normal(rng: &mut Rng) -> f64 {
    loop {
        let u = 2.0 * rng.next_f64() - 1.0;
        let v = 2.0 * rng.next_f64() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

/// A `Gamma(shape, 1)` variate.  For `shape ≥ 1` the Marsaglia–Tsang
/// squeeze–rejection method (ACM TOMS 26(3), 2000): with `d = k − 1/3`,
/// `c = 1/√(9d)`, draw `x ~ N(0, 1)`, `v = (1 + cx)³ > 0`, `u ~ U(0, 1)`
/// and accept `d·v` when `u < 1 − 0.0331 x⁴` (the squeeze) or
/// `ln u < x²/2 + d(1 − v + ln v)`; the acceptance rate exceeds 95 %.
/// For `shape < 1` the boost `Gamma(k) = Gamma(k + 1) · U^{1/k}`
/// (same paper, §7) reduces to the first case.
pub(crate) fn standard_gamma(rng: &mut Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        let boost = positive_uniform(rng).powf(1.0 / shape);
        return standard_gamma(rng, shape + 1.0) * boost;
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let t = 1.0 + c * x;
        if t <= 0.0 {
            continue;
        }
        let v = t * t * t;
        let u = rng.next_f64();
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v;
        }
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// The rate below which a Poisson variate is drawn by Knuth's
/// multiplication method; above it, by Hörmann's PTRS.
const POISSON_KNUTH_LIMIT: f64 = 30.0;

/// A `Poisson(λ)` variate.  For `λ < 30`, Knuth's method (TAOCP 2, §3.4.1):
/// the number of unit-rate exponential inter-arrivals in `[0, λ]`, counted
/// by multiplying uniforms until the product drops below `e^{−λ}`
/// (`O(λ)` per draw).  For `λ ≥ 30`, the transformed rejection method
/// with squeeze PTRS of W. Hörmann, "The transformed rejection method for
/// generating Poisson random variables", Insurance: Mathematics and
/// Economics 12 (1993) 39–45 — valid for `λ ≥ 10`, `O(1)` per draw, and
/// exact: the final test compares against the exact log-pmf
/// `k ln λ − λ − ln Γ(k + 1)`; the two earlier tests are its proven
/// squeezes (the same constants NumPy's `random_poisson_ptrs` uses).
pub(crate) fn poisson(rng: &mut Rng, lambda: f64) -> f64 {
    if lambda < POISSON_KNUTH_LIMIT {
        let limit = (-lambda).exp();
        let mut k = 0.0;
        let mut product = 1.0;
        loop {
            product *= rng.next_f64();
            if product <= limit {
                return k;
            }
            k += 1.0;
        }
    }
    let log_lambda = lambda.ln();
    let b = 0.931 + 2.53 * lambda.sqrt();
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let v_r = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = rng.next_f64() - 0.5;
        let v = rng.next_f64();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
        if us >= 0.07 && v <= v_r {
            return k;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        // `ln v` with `v = 0` is `−∞` and accepts; `us = 0` (only when the
        // uniform was exactly 0) gives `k = −∞` and is rejected above.
        if v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln()
            <= -lambda + k * log_lambda - lgamma(k + 1.0)
        {
            return k;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(samples: &[f64]) -> (f64, f64) {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let var = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var)
    }

    #[test]
    fn standard_normal_moments() {
        let mut rng = Rng::new(3);
        let s: Vec<f64> = (0..20_000).map(|_| standard_normal(&mut rng)).collect();
        let (mean, var) = moments(&s);
        assert!(mean.abs() < 4.0 / 20_000f64.sqrt(), "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
    }

    #[test]
    fn standard_gamma_both_branches() {
        for shape in [0.3, 0.9, 1.0, 2.5, 17.0] {
            let mut rng = Rng::new(5);
            let s: Vec<f64> = (0..20_000)
                .map(|_| standard_gamma(&mut rng, shape))
                .collect();
            assert!(s.iter().all(|v| *v >= 0.0), "shape {shape}: negative draw");
            let (mean, var) = moments(&s);
            let se = shape.sqrt() / 20_000f64.sqrt();
            assert!(
                (mean - shape).abs() < 4.0 * se,
                "shape {shape}: mean {mean}"
            );
            assert!(
                (var - shape).abs() < 0.1 * shape.max(1.0),
                "shape {shape}: variance {var}"
            );
        }
    }

    #[test]
    fn poisson_both_branches() {
        for lambda in [0.5, 4.0, 29.9, 30.0, 150.0] {
            let mut rng = Rng::new(7);
            let s: Vec<f64> = (0..20_000).map(|_| poisson(&mut rng, lambda)).collect();
            assert!(
                s.iter().all(|v| *v >= 0.0 && v.fract() == 0.0),
                "λ = {lambda}: non-lattice draw"
            );
            let (mean, var) = moments(&s);
            let se = lambda.sqrt() / 20_000f64.sqrt();
            assert!(
                (mean - lambda).abs() < 4.0 * se,
                "λ = {lambda}: mean {mean}"
            );
            assert!(
                (var - lambda).abs() < 0.1 * lambda.max(1.0),
                "λ = {lambda}: variance {var}"
            );
        }
    }
}
