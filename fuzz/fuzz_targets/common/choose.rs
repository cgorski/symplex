//! Byte-driven choices for the structure-aware generators of the
//! differential targets (`fuzz_evalf`, `fuzz_calculus`, `fuzz_compile`,
//! `fuzz_roundtrip`; `#[path]`-included by each, not a target itself).
//!
//! The generators were written against a seeded PRNG (the in-house
//! hunters); [`Src`] offers the same interface — `below`, `range`,
//! `chance`, `pick` — but every decision reads the next input byte(s), so
//! one mutated byte changes one decision and libFuzzer's coverage guidance
//! sees which decisions matter.  Choices among at most 256 alternatives
//! read one byte, wider ones as many as they need (little-endian).  Past
//! the end of the input every byte is `0`: the first alternative, which
//! every generator makes a leaf, so decoding always terminates.
#![allow(dead_code)]

pub struct Src<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Src<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Src { data, pos: 0 }
    }

    /// The next byte, `0` past the end.
    pub fn byte(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    /// Bytes consumed so far (may exceed the input length).
    pub fn consumed(&self) -> usize {
        self.pos
    }

    /// A value in `0..n` (`0` when `n ≤ 1`).
    pub fn below(&mut self, n: u64) -> u64 {
        if n <= 1 {
            return 0;
        }
        let mut v: u128 = 0;
        let mut span: u128 = 1;
        let mut shift = 0;
        while span < u128::from(n) {
            v |= u128::from(self.byte()) << shift;
            shift += 8;
            span <<= 8;
        }
        (v % u128::from(n)) as u64
    }

    /// A value in `lo..=hi` (`lo` when `hi < lo`).
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        lo + self.below((hi - lo + 1) as u64) as i64
    }

    /// `true` with probability ≈ `p` over a uniform byte (so `true` on the
    /// zero byte for every `p > 1/512`).
    pub fn chance(&mut self, p: f64) -> bool {
        (f64::from(self.byte()) + 0.5) / 256.0 < p
    }

    /// One of `xs` (which must not be empty).
    pub fn pick<'b, T>(&mut self, xs: &'b [T]) -> &'b T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}
