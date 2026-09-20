//! Sampling: a small deterministic generator (SplitMix64) so that samples
//! are reproducible, and `RandomVariable::sample` built on inverse
//! transform sampling where the quantile function has a closed form, or on
//! the pmf's cumulative sums for finite discrete supports.

use crate::api::expr::Ex;
use crate::base::errors::SymplexError;

use super::rv::{Distribution, RandomVariable, Support};

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
}

impl RandomVariable {
    /// `n` samples as `f64`, by inverse transform sampling through the
    /// quantile function (continuous families with a closed-form quantile)
    /// or by walking the pmf's cumulative sums (discrete families with a
    /// finite support); the distribution's parameters must evaluate
    /// numerically.  SymPy: `sample(X, size=n)`.
    ///
    /// # Errors
    ///
    /// [`SymplexError::NotImplemented`] if the family has neither route;
    /// [`SymplexError::Unevaluable`] if a parameter is symbolic.
    pub fn sample(&self, n: usize, rng: &mut Rng) -> Result<Vec<f64>, SymplexError> {
        let ctx = self.context();
        let x = ctx.symbol("_sample_x");
        match (self.distribution(), self.support()) {
            (Distribution::Continuous(_), _) => {
                let p = ctx.symbol("_sample_p");
                let q = self.quantile(&p).ok_or_else(|| {
                    SymplexError::NotImplemented(format!(
                        "sampling {}: no closed-form quantile function",
                        self.distribution().name()
                    ))
                })?;
                let q = q.compile(&["_sample_p"])?;
                Ok((0..n).map(|_| q.call(&[rng.next_f64()])).collect())
            }
            (
                Distribution::Discrete(_),
                Support::Discrete {
                    lo: Some(lo),
                    hi: Some(hi),
                },
            ) => {
                let lo_v = lo.eval_f64()?;
                let hi_v = hi.eval_f64()?;
                if !(lo_v.is_finite() && hi_v.is_finite() && hi_v >= lo_v) {
                    return Err(SymplexError::NotImplemented(format!(
                        "sampling {}: the support is not a finite integer range",
                        self.distribution().name()
                    )));
                }
                let pmf = self.density(&x).compile(&["_sample_x"])?;
                let values: Vec<f64> = {
                    let mut v = Vec::new();
                    let mut k = lo_v;
                    while k <= hi_v && v.len() < 1_000_000 {
                        v.push(k);
                        k += 1.0;
                    }
                    v
                };
                let mut cumulative = Vec::with_capacity(values.len());
                let mut acc = 0.0;
                for &k in &values {
                    acc += pmf.call(&[k]);
                    cumulative.push(acc);
                }
                Ok((0..n)
                    .map(|_| {
                        let u = rng.next_f64() * acc;
                        let idx = cumulative.partition_point(|c| *c < u);
                        values[idx.min(values.len().saturating_sub(1))]
                    })
                    .collect())
            }
            _ => Err(SymplexError::NotImplemented(format!(
                "sampling {}: no sampling route for an infinite discrete support",
                self.distribution().name()
            ))),
        }
    }

    /// A single sample; see [`sample`](Self::sample).
    ///
    /// # Errors
    ///
    /// As [`sample`](Self::sample).
    pub fn sample_one(&self, rng: &mut Rng) -> Result<f64, SymplexError> {
        Ok(self.sample(1, rng)?.first().copied().unwrap_or(f64::NAN))
    }

    /// The symbol used internally; exposed for tests of the sampler.
    #[doc(hidden)]
    pub fn sampling_symbol(&self) -> Ex {
        self.context().symbol("_sample_x")
    }
}
