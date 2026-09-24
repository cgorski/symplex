//! `stats::numdist` (the `f64` reference distributions): invariants that
//! hold for every valid parameter and argument, over the whole `f64` range.
//!
//! The 0.21 audit found a hang (discrete `ppf` above 2⁵³), a negative tail
//! (Temme's expansion when `erfc` underflowed), exact zeros where the
//! value was representable, and a NaN from `+∞` — all of them violations of
//! the checks below.  The checks:
//!
//! * `cdf`, `sf` never panic and lie in `[0, 1]` (never NaN) for valid
//!   parameters;
//! * `cdf + sf = 1` where neither tail is tiny (relative 1e-9);
//! * `cdf` is monotone: `cdf(x) ≤ cdf(x′)` for `x < x′` (up to 1e-9
//!   relative — TOMS 708 is not ulp-monotone, and far-tail values are good to ≈ 1e-11);
//! * `ppf(p)`/`isf(q)` terminate (libFuzzer's `-timeout` turns a hang into
//!   a crash), and when they return a finite `x` the matching tail is back
//!   at the level: `cdf(ppf(p)) ≈ p` for continuous families (relative
//!   1e-6 — loose enough for the flat tails, tight enough to catch a wrong
//!   branch); for discrete ones `cdf(k) ≥ p > cdf(k − 1)`.
//!
//! Parameters are drawn log-uniformly: shapes mostly from 1e-3 to 1e9, and
//! one draw in four from 1e-300 to 1e300 (1e15 for the beta and F) (0.27:
//! tiny shapes had negative tails, huge ones non-converging quantiles); `n`, rates from 1 to 1e17
//! (past 2⁵³ the discrete quantiles must refuse, not loop); `x` absolute or
//! relative to the shape; `p` from the full range including subnormals.
//! The discrete `isf` is checked like `ppf`: `sf(k) ≤ q < sf(k − 1)`.
#![no_main]

use libfuzzer_sys::fuzz_target;
use symplex::stats::numdist::{beta, binom, chi2, f, gamma, norm, poisson, t};

struct Bytes<'a>(&'a [u8], usize);

impl Bytes<'_> {
    fn u8(&mut self) -> u8 {
        let b = self.0.get(self.1).copied().unwrap_or(0);
        self.1 += 1;
        b
    }
    fn u16(&mut self) -> u16 {
        u16::from(self.u8()) << 8 | u16::from(self.u8())
    }
    /// Log-uniform in `[lo, hi]`.
    fn log_range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = f64::from(self.u16()) / 65535.0;
        (lo.ln() + u * (hi.ln() - lo.ln())).exp()
    }
    /// A shape: log-uniform on `[lo, hi]`, or on `[1e-300, extreme]` one
    /// time in four.
    fn shape(&mut self, lo: f64, hi: f64, extreme: f64) -> f64 {
        if self.u8() % 4 == 0 {
            self.log_range(1e-300, extreme)
        } else {
            self.log_range(lo, hi)
        }
    }
    /// A point: absolute on `[lo, hi]`, or `centre` times a factor in
    /// `[1e-3, 1e3]` (near the bulk of a huge or tiny shape).
    fn point(&mut self, lo: f64, hi: f64, centre: f64) -> f64 {
        if self.u8() % 2 == 0 {
            self.log_range(lo, hi)
        } else {
            (centre * self.log_range(1e-3, 1e3)).min(f64::MAX)
        }
    }
    /// A probability level: mostly moderate, often deep in a tail.
    fn level(&mut self) -> f64 {
        match self.u8() % 4 {
            0 => f64::from(self.u16()) / 65536.0 + 1.0 / 131072.0,
            1 => self.log_range(1e-300, 1e-3),
            2 => 1.0 - self.log_range(1e-15, 1e-3),
            _ => self.log_range(5e-324, 1e-300),
        }
    }
}

fn check_prob(label: &str, v: f64) {
    assert!(
        (0.0..=1.0).contains(&v),
        "{label} = {v:e} is not a probability"
    );
}

/// `cdf`/`sf` sanity at `x` and a slightly larger `x2`.
fn check_tails(label: &str, cdf: impl Fn(f64) -> f64, sf: impl Fn(f64) -> f64, x: f64, x2: f64) {
    let (c, s) = (cdf(x), sf(x));
    check_prob(&format!("{label} cdf({x:e})"), c);
    check_prob(&format!("{label} sf({x:e})"), s);
    if c > 1e-12 && s > 1e-12 {
        assert!(
            ((c + s) - 1.0).abs() <= 1e-9,
            "{label}: cdf + sf = {:e} at x = {x:e}",
            c + s
        );
    }
    let c2 = cdf(x2);
    assert!(
        c2 >= c * (1.0 - 1e-9) - 1e-300,
        "{label}: cdf decreases from {c:e} at {x:e} to {c2:e} at {x2:e}"
    );
}

fn check_quantile(
    label: &str,
    cdf: impl Fn(f64) -> f64,
    ppf: impl Fn(f64) -> Result<f64, symplex::prelude::SymplexError>,
    p: f64,
) {
    if let Ok(x) = ppf(p)
        && x.is_finite()
    {
        let back = cdf(x);
        // Only meaningful where the density is not vanishingly flat.  Where
        // the cdf is steeper than an ulp (beta(0.001, 0.001) puts half its
        // mass within 10^-3200 of 1) the true quantile is not representable;
        // then the neighbouring floats must bracket the level instead.
        if p > 1e-280 && p < 1.0 - 1e-12 && back > 0.0 {
            let close = ((back - p) / p).abs() <= 1e-6 || (back - p).abs() <= 1e-12;
            // `cdf` is increasing for `ppf` and decreasing (`sf`) for `isf`.
            let (below, above) = (cdf(x.next_down()), cdf(x.next_up()));
            let (lo, hi) = if below <= above {
                (below, above)
            } else {
                (above, below)
            };
            let bracketed = lo <= p * (1.0 + 1e-9) && hi >= p * (1.0 - 1e-9);
            assert!(
                close || bracketed,
                "{label}: cdf(ppf({p:e})) = {back:e} (x = {x:e})"
            );
        }
    }
}

/// `sf(k) ≤ q < sf(k − 1)` for a discrete `isf`.
fn check_discrete_isf(
    label: &str,
    sf: impl Fn(f64) -> f64,
    k: Result<f64, symplex::prelude::SymplexError>,
    q: f64,
) {
    if let Ok(k) = k
        && k.is_finite()
        && k < 9.0e15
        && q > 1e-280
    {
        assert!(
            sf(k) <= q * (1.0 + 1e-9),
            "{label}: sf({k}) = {:e} > q = {q:e}",
            sf(k)
        );
        if k >= 1.0 && q < 1.0 - 1e-12 {
            assert!(
                sf(k - 1.0) > q * (1.0 - 1e-9),
                "{label}: sf({}) = {:e} already ≤ q = {q:e}",
                k - 1.0,
                sf(k - 1.0)
            );
        }
    }
}

fn check_discrete_quantile(
    label: &str,
    cdf: impl Fn(f64) -> f64,
    k: Result<f64, symplex::prelude::SymplexError>,
    p: f64,
) {
    if let Ok(k) = k
        && k.is_finite()
        && k < 9.0e15
        && p < 1.0 - 1e-12
    {
        assert!(
            cdf(k) >= p * (1.0 - 1e-12),
            "{label}: cdf({k}) = {:e} < p = {p:e}",
            cdf(k)
        );
        if k >= 1.0 && p > 1e-280 {
            assert!(
                cdf(k - 1.0) < p * (1.0 + 1e-9),
                "{label}: cdf({}) = {:e} already ≥ p = {p:e}",
                k - 1.0,
                cdf(k - 1.0)
            );
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut b = Bytes(data, 0);
    let which = b.u8() % 8;
    let p = b.level();
    let q = b.level();
    let bump = 1.0 + b.log_range(1e-12, 1.0);
    match which {
        0 => {
            let x = (b.log_range(1e-3, 40.0)) * if b.u8() % 2 == 0 { 1.0 } else { -1.0 };
            check_tails(
                "norm",
                norm::cdf,
                norm::sf,
                x,
                if x < 0.0 { x / bump } else { x * bump },
            );
            check_quantile("norm", norm::cdf, norm::ppf, p);
            check_quantile("norm isf", norm::sf, norm::isf, q);
        }
        1 => {
            let df = b.shape(1e-2, 1e9, 1e300);
            let x = b.log_range(1e-3, 1e200) * if b.u8() % 2 == 0 { 1.0 } else { -1.0 };
            let (cdf, sf) = (|x| t::cdf(x, df), |x| t::sf(x, df));
            let l = format!("t({df:e})");
            check_tails(&l, cdf, sf, x, if x < 0.0 { x / bump } else { x * bump });
            check_quantile(&l, cdf, |p| t::ppf(p, df), p);
            check_quantile(&format!("{l} isf"), sf, |q| t::isf(q, df), q);
        }
        2 => {
            let df = b.shape(1e-2, 1e9, 1e300);
            let x = b.point(1e-6, 1e10, df);
            let (cdf, sf) = (|x| chi2::cdf(x, df), |x| chi2::sf(x, df));
            let l = format!("chi2({df:e})");
            check_tails(&l, cdf, sf, x, x * bump);
            check_quantile(&l, cdf, |p| chi2::ppf(p, df), p);
            check_quantile(&format!("{l} isf"), sf, |q| chi2::isf(q, df), q);
        }
        3 => {
            let (shape, scale) = (b.shape(1e-3, 1e9, 1e300), b.log_range(1e-3, 1e3));
            let x = b.point(1e-6, 1e12, shape * scale);
            let (cdf, sf) = (
                |x| gamma::cdf(x, shape, scale),
                |x| gamma::sf(x, shape, scale),
            );
            let l = format!("gamma({shape:e}, {scale:e})");
            check_tails(&l, cdf, sf, x, x * bump);
            check_quantile(&l, cdf, |p| gamma::ppf(p, shape, scale), p);
            check_quantile(&format!("{l} isf"), sf, |q| gamma::isf(q, shape, scale), q);
        }
        4 => {
            // Beta and F shapes beyond ~1e15 put the distribution inside the
            // rounding of its f64 argument (`(a + b)x`, `d₁x/(d₁x + d₂)`): the
            // computed cdf is then not monotone at the ulp level, and a
            // quantile can only be good to its relative accuracy (module docs).
            let (a, bb) = (b.shape(1e-3, 1e7, 1e15), b.shape(1e-3, 1e7, 1e15));
            let x = f64::from(b.u16()) / 65536.0;
            let (cdf, sf) = (|x| beta::cdf(x, a, bb), |x| beta::sf(x, a, bb));
            let label = format!("beta({a:e}, {bb:e})");
            check_tails(&label, cdf, sf, x, (x * bump).min(1.0));
            check_quantile(&label, cdf, |p| beta::ppf(p, a, bb), p);
            check_quantile(&format!("{label} isf"), sf, |q| beta::isf(q, a, bb), q);
        }
        5 => {
            let (d1, d2) = (b.shape(1e-2, 1e8, 1e15), b.shape(1e-2, 1e8, 1e15));
            let x = b.point(1e-6, 1e6, 1.0);
            let (cdf, sf) = (|x| f::cdf(x, d1, d2), |x| f::sf(x, d1, d2));
            let l = format!("f({d1:e}, {d2:e})");
            check_tails(&l, cdf, sf, x, x * bump);
            check_quantile(&l, cdf, |p| f::ppf(p, d1, d2), p);
            check_quantile(&format!("{l} isf"), sf, |q| f::isf(q, d1, d2), q);
        }
        6 => {
            let n = b.log_range(1.0, 1e17).floor();
            let pr = b.log_range(1e-9, 1.0).min(1.0 - 1e-9);
            let k = (n * f64::from(b.u16()) / 65535.0).floor();
            let (cdf, sf) = (|k| binom::cdf(k, n, pr), |k| binom::sf(k, n, pr));
            let l = format!("binom({n:e}, {pr:e})");
            check_tails(&l, cdf, sf, k, k + 1.0);
            check_discrete_quantile(&l, cdf, binom::ppf(p, n, pr), p);
            check_discrete_isf(&l, sf, binom::isf(q, n, pr), q);
        }
        _ => {
            let rate = if b.u8() % 4 == 0 {
                b.log_range(1e-300, 1e-6)
            } else {
                b.log_range(1e-6, 1e17)
            };
            let k = (rate * b.log_range(1e-3, 10.0)).floor();
            let (cdf, sf) = (|k| poisson::cdf(k, rate), |k| poisson::sf(k, rate));
            let l = format!("poisson({rate:e})");
            check_tails(&l, cdf, sf, k, k + 1.0);
            check_discrete_quantile(&l, cdf, poisson::ppf(p, rate), p);
            check_discrete_isf(&l, sf, poisson::isf(q, rate), q);
        }
    }
});
