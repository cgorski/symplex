//! Real root counting and isolation for univariate polynomials over ℚ.
//!
//! [`SturmChain`] — the name is historical: until 0.30 it held a Sturm
//! chain — isolates the distinct real roots of a polynomial once, when it
//! is built, and answers every count and isolation query from those
//! isolating intervals.
//!
//! The isolation is the **Descartes method**, the bisection form of the
//! Vincent–Collins–Akritas algorithm (G. E. Collins, A. G. Akritas,
//! *Polynomial real root isolation using Descartes' rule of signs*, SYMSAC
//! 1976), on the integer square-free part `p₀`.  For a polynomial `q` whose
//! roots in `(0, 1)` are those of `p₀` in a dyadic cell, the number `v` of
//! sign variations in the coefficients of `(x + 1)ⁿ q(1/(x + 1))` bounds the
//! number of roots in the cell and has its parity, so `v = 0` means none
//! and `v = 1` exactly one; otherwise the cell is halved (`2ⁿ q(x/2)` and
//! its Taylor shift by 1), and a root on the midpoint is split off exactly.
//! For a square-free `q` every cell small enough is decided (the two-circle
//! theorem), so the loop terminates.  Only integer additions and shifts
//! are involved; the Sturm chain of 0.29–0.30 needed a subresultant PRS
//! whose coefficients grow to `deg · (coefficient size)` bits: 4.3 s
//! (release) for a degree-40 polynomial with 41 random 30-digit rational
//! coefficients (a 1,230-digit denominator lcm), all of it in the PRS.
//!
//! A count over `(a, b]` locates `a` and `b` among the isolating
//! intervals; a point inside an isolating interval costs one sign
//! evaluation of `p₀`.  [`isolate_roots_in`](SturmChain::isolate_roots_in)
//! and [`refine_interval`](SturmChain::refine_interval) replay the
//! bisection of the Sturm-chain implementation with these exact counts —
//! the counts a Sturm chain gives, `#{roots in (a, b]}` — so their output is
//! the same interval for interval: half-open [`Interval::left_open`] cells,
//! and the closed singleton [`Interval::point`] for a root that a bisection
//! point hits exactly.

use astro_float::{BigFloat, RoundingMode};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::base::numeric::{bigint_to_bigfloat, ratio_to_bigfloat};

use crate::base::interval::{Interval, IntervalKind};
use crate::base::numeric::Q;
use crate::poly::Poly;
use crate::poly::zpoly::{integer_scaled, powers, z_primitive};

/// The distinct real roots of a polynomial, isolated (see the module
/// documentation), with the counting and isolation queries built on them.
#[derive(Debug, Clone)]
pub(crate) struct SturmChain {
    /// The square-free part `p₀` of the input — the input itself when it
    /// is square-free — exactly as `Poly::square_free_part` gives it.
    p0: Poly,
    /// `p₀` multiplied by a positive integer so that every coefficient is
    /// an integer (ascending degree): the same sign at every point.
    int_p0: Vec<BigInt>,
    /// `int_p0` rounded to [`FILTER_PREC`] bits: the floating-point filter
    /// of [`sign_at`](Self::sign_at).
    float_p0: Vec<BigFloat>,
    /// The distinct real roots of `p₀`, ascending, each isolated.
    roots: Vec<RootLoc>,
}

/// Where one real root of `p₀` is known to lie.
#[derive(Debug, Clone)]
enum RootLoc {
    /// The root is exactly this rational.
    Exact(Q),
    /// The root is the only one of `p₀` in the open interval `(lo, hi)`,
    /// and `p₀` has the sign `after_lo` (`±1`) on `(lo, root)`.
    Open { lo: Q, hi: Q, after_lo: i8 },
}

/// The position of a root relative to a rational point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Below,
    At,
    Above,
}

/// Precision of the floating-point sign filter.
const FILTER_PREC: usize = 128;

/// Primes below `2⁶⁴` for the modular square-freeness certificate
/// ([`certified_square_free`]): `2⁶⁴ − 59`, `2⁶³ − 25`, `2⁶² − 57`, `2⁶¹ − 1`.
const SQUARE_FREE_PRIMES: [u64; 4] = [
    18_446_744_073_709_551_557,
    9_223_372_036_854_775_783,
    4_611_686_018_427_387_847,
    2_305_843_009_213_693_951,
];

#[allow(dead_code)] // Used indirectly via Ex::count_real_roots() bridge; will be exposed publicly later
impl SturmChain {
    /// Isolate the distinct real roots of `p`.
    ///
    /// The input is first made square-free (`p₀ = p / gcd(p, p′)`), so that
    /// every count is of *distinct* roots.  Square-freeness is first
    /// certified modulo word-size primes ([`certified_square_free`]), which
    /// spares the gcd of `p` and `p′` over `ℚ` whenever `p` is square-free.
    pub fn new(p: &Poly) -> Self {
        let p0 =
            if p.degree().unwrap_or(0) == 0 || certified_square_free(&integer_scaled(p.coeffs())) {
                p.clone()
            } else {
                p.square_free_part()
            };
        let int_p0 = integer_scaled(p0.coeffs());
        let float_p0 = int_p0
            .iter()
            .map(|c| bigint_to_bigfloat(c, FILTER_PREC))
            .collect();
        let mut out = SturmChain {
            p0,
            int_p0,
            float_p0,
            roots: Vec::new(),
        };
        out.roots = out.isolate_descartes();
        out
    }

    /// The real roots of `p₀`, ascending (the Descartes method on the
    /// positive and the negative half-line, see the module documentation).
    fn isolate_descartes(&self) -> Vec<RootLoc> {
        if self.int_p0.len() < 2 {
            return Vec::new();
        }
        let mut q = z_primitive(&self.int_p0);
        let mut raw: Vec<RootLoc> = Vec::new();
        // `p₀` is square-free: `0` is at most a simple root.
        if q[0].is_zero() {
            raw.push(RootLoc::Exact(Q::zero()));
            q.remove(0);
        }
        for (lo, hi) in positive_roots(&q, &mut raw, false) {
            raw.push(RootLoc::Open {
                lo,
                hi,
                after_lo: 0,
            });
        }
        let reflected: Vec<BigInt> = q
            .iter()
            .enumerate()
            .map(|(i, c)| if i % 2 == 1 { -c } else { c.clone() })
            .collect();
        for (lo, hi) in positive_roots(&reflected, &mut raw, true) {
            raw.push(RootLoc::Open {
                lo,
                hi,
                after_lo: 0,
            });
        }
        // Cells are disjoint; an exact root equal to the lower end of a
        // cell is below that cell's root.
        raw.sort_by(|a, b| {
            let key = |l: &RootLoc| -> (Q, u8) {
                match l {
                    RootLoc::Exact(r) => (r.clone(), 0),
                    RootLoc::Open { lo, .. } => (lo.clone(), 1),
                }
            };
            key(a).cmp(&key(b))
        });
        for loc in &mut raw {
            if let RootLoc::Open { lo, after_lo, .. } = loc {
                let s = self.sign_at(lo);
                // At a (simple) root `lo`, `p₀` takes the sign of `p₀′`
                // just to its right.
                *after_lo = if s != 0 {
                    s
                } else {
                    self.derivative_sign_at(lo)
                };
            }
        }
        raw
    }

    /// The sign of `p₀` at `x` (`+1`, `-1` or `0`; `0` everywhere for the
    /// zero polynomial).
    ///
    /// Taken from a floating-point Horner evaluation at [`FILTER_PREC`]
    /// bits with a rigorous bound on its rounding error ([`float_sign`]),
    /// and computed exactly in `ℤ` only where that bound does not decide it
    /// (at or next to a root).  The sign is the exact one either way.
    fn sign_at(&self, x: &Q) -> i8 {
        let xf = ratio_to_bigfloat(x, FILTER_PREC, RoundingMode::ToEven);
        if let Some(s) = float_sign(&self.float_p0, &xf) {
            return s;
        }
        let (a, b) = numer_denom(x);
        let b_pows = powers(&b, self.int_p0.len().saturating_sub(1));
        int_sign_at(&self.int_p0, &a, &b, &b_pows)
    }

    /// The sign of `p₀′` at `x`, exactly.
    fn derivative_sign_at(&self, x: &Q) -> i8 {
        let d: Vec<BigInt> = self
            .int_p0
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| c * BigInt::from(i))
            .collect();
        let (a, b) = numer_denom(x);
        let b_pows = powers(&b, d.len().saturating_sub(1));
        int_sign_at(&d, &a, &b, &b_pows)
    }

    /// Is `x` a root of the (square-free part of the) polynomial?
    fn is_root(&self, x: &Q) -> bool {
        self.sign_at(x) == 0
    }

    /// The position of the root at `loc` relative to `c`: one sign
    /// evaluation when `c` lies inside an open cell, none otherwise.
    fn locate(&self, loc: &RootLoc, c: &Q) -> Side {
        match loc {
            RootLoc::Exact(r) => match r.cmp(c) {
                std::cmp::Ordering::Less => Side::Below,
                std::cmp::Ordering::Equal => Side::At,
                std::cmp::Ordering::Greater => Side::Above,
            },
            RootLoc::Open { lo, hi, after_lo } => {
                if c <= lo {
                    Side::Above
                } else if c >= hi {
                    Side::Below
                } else {
                    let s = self.sign_at(c);
                    if s == 0 {
                        Side::At
                    } else if s == *after_lo {
                        // `p₀` keeps its sign from `lo` to `c`.
                        Side::Above
                    } else {
                        Side::Below
                    }
                }
            }
        }
    }

    /// [`locate`](Self::locate), narrowing an open cell that contains `c`
    /// to the side of `c` the root is on.
    fn locate_narrowing(&self, loc: &mut RootLoc, c: &Q) -> Side {
        let side = self.locate(loc, c);
        let inside = matches!(&*loc, RootLoc::Open { lo, hi, .. } if lo < c && c < hi);
        if inside {
            match side {
                Side::At => *loc = RootLoc::Exact(c.clone()),
                Side::Above => {
                    if let RootLoc::Open { lo, .. } = loc {
                        *lo = c.clone();
                    }
                }
                Side::Below => {
                    if let RootLoc::Open { hi, .. } = loc {
                        *hi = c.clone();
                    }
                }
            }
        }
        side
    }

    /// The number of roots in `locs` (ascending) that are `≤ c`, by
    /// binary search, narrowing the cells it looks into.
    fn count_at_most_narrowing(&self, locs: &mut [RootLoc], c: &Q) -> usize {
        let (mut lo, mut hi) = (0usize, locs.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.locate_narrowing(&mut locs[mid], c) == Side::Above {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    /// The number of distinct real roots `≤ c`.
    fn count_at_most(&self, c: &Q) -> usize {
        self.roots
            .partition_point(|loc| self.locate(loc, c) != Side::Above)
    }

    /// Total number of distinct real roots of the polynomial.
    pub fn count_real_roots(&self) -> usize {
        self.roots.len()
    }

    /// Quick check: does the polynomial have zero real roots?
    pub fn has_no_real_roots(&self) -> bool {
        self.count_real_roots() == 0
    }

    /// Count real roots in the half-open interval `(a, b]` (`0` when
    /// `a ≥ b`).
    pub fn count_roots_in(&self, a: &Ratio<BigInt>, b: &Ratio<BigInt>) -> usize {
        if a >= b {
            return 0;
        }
        self.count_at_most(b).saturating_sub(self.count_at_most(a))
    }

    /// Isolate the real roots in `(a, b]` into disjoint half-open
    /// sub-intervals `(lo, hi]` by bisection.
    ///
    /// Every returned interval is [`Interval::left_open`] — the count that
    /// drives the bisection is [`count_roots_in`](Self::count_roots_in),
    /// i.e. over `(lo, hi]` — and contains exactly one root, unless
    /// `max_depth` bisection levels were exhausted first, in which case an
    /// interval may still hold several.  The output is sorted left to right.
    pub fn isolate_roots_in(&self, a: &Q, b: &Q, max_depth: u32) -> Vec<Interval<Q>> {
        // Explicit stack (no recursion); intervals are pushed right-first so
        // that the output comes out sorted left to right.  Each entry
        // carries the index range of the roots in its cell; the isolating
        // cells of `locs` narrow as the bisection points fall into them.
        let two = Ratio::from_integer(BigInt::from(2));
        let mut locs = self.roots.clone();
        let first = self.count_at_most_narrowing(&mut locs, a);
        let end = self.count_at_most_narrowing(&mut locs, b).max(first);
        let mut result = Vec::new();
        let mut stack: Vec<Cell> = vec![Cell {
            lo: a.clone(),
            hi: b.clone(),
            depth: max_depth,
            first,
            end,
        }];
        while let Some(cell) = stack.pop() {
            let n = cell.end - cell.first;
            if n == 0 {
                continue;
            }
            if n == 1 || cell.depth == 0 {
                result.push(Interval::left_open(cell.lo, cell.hi));
                continue;
            }
            let mid = (&cell.lo + &cell.hi) / &two;
            let split =
                cell.first + self.count_at_most_narrowing(&mut locs[cell.first..cell.end], &mid);
            stack.push(Cell {
                lo: mid.clone(),
                hi: cell.hi,
                depth: cell.depth - 1,
                first: split,
                end: cell.end,
            });
            stack.push(Cell {
                lo: cell.lo,
                hi: mid,
                depth: cell.depth - 1,
                first: cell.first,
                end: split,
            });
        }
        result
    }

    /// Isolate **all** distinct real roots of the polynomial into disjoint
    /// rational intervals, each containing exactly one root.
    ///
    /// The search starts from the Cauchy root bound.  Each returned
    /// interval is either
    ///
    /// * [`Interval::left_open`] `(lo, hi]` with `lo < hi` and exactly one
    ///   root (the guarantee of [`isolate_roots_in`](Self::isolate_roots_in)),
    ///   whose upper endpoint is *not* a root, or
    /// * the closed singleton [`Interval::point`] `[r, r]` when the one root
    ///   of a `(lo, r]` cell is `r` itself (a bisection point hit it
    ///   exactly).
    ///
    /// Intervals are sorted.
    pub fn isolate_all_real_roots(&self) -> Vec<Interval<Q>> {
        if self.p0.degree().unwrap_or(0) == 0 {
            return vec![];
        }
        let bound = cauchy_bound(&self.p0) + Ratio::from_integer(BigInt::from(1));
        let neg_bound = -bound.clone();
        let depth = self.isolation_depth(&bound);
        let raw = self.isolate_roots_in(&neg_bound, &bound, depth);
        raw.into_iter()
            .map(|iv| {
                if self.is_root(&iv.upper) {
                    Interval::point(iv.upper)
                } else {
                    iv
                }
            })
            .collect()
    }

    /// Bisection levels that certainly separate the roots in `[−B, B]`:
    /// `log₂(2B)` down to width 1, then `log₂(1/sep)` with Mahler's bound on
    /// the root separation of a square-free integer polynomial of degree
    /// `n`, `sep > √3·n^{−(n+2)/2}·‖p‖₂^{1−n}` (its discriminant is a non-zero
    /// integer), plus a margin.  Before 0.30 the depth was 256 whatever the
    /// bound: the Cauchy bound of `Π (bᵢx − aᵢ)` with 20-digit roots is
    /// `~2⁶⁶⁰`, and 20 roots came back in 2 "isolating" cells.
    fn isolation_depth(&self, bound: &Q) -> u32 {
        let p = &self.int_p0;
        if p.is_empty() {
            return 0;
        }
        let n = p.len().saturating_sub(1) as f64;
        let log2_int = |x: &BigInt| x.bits() as f64;
        let max_coeff = p.iter().map(log2_int).fold(0.0, f64::max);
        let norm = max_coeff + 0.5 * (n + 1.0).log2();
        let sep_bits = (n + 2.0) / 2.0 * n.max(1.0).log2() + (n - 1.0).max(0.0) * norm;
        let bound_bits = log2_int(bound.numer()) - log2_int(bound.denom()) + 2.0;
        let total = bound_bits.max(0.0) + sep_bits + 16.0;
        if total > f64::from(u32::MAX / 2) {
            u32::MAX / 2
        } else {
            (total.ceil() as u32).max(256)
        }
    }

    /// Shrink an isolating interval by bisection until its width is at
    /// most `max_width`.
    ///
    /// `iv` must be an interval as produced by
    /// [`isolate_all_real_roots`](Self::isolate_all_real_roots): either
    /// `(lo, hi]` containing exactly one root, or a point `[r, r]`, which
    /// is returned unchanged.  Each bisection keeps the half whose
    /// [`count_roots_in`](Self::count_roots_in) is one, so the result is
    /// again `(lo, hi]` — or the closed singleton [`Interval::point`] `[r, r]`
    /// when a bisection point lands exactly on the root.
    pub fn refine_interval(&self, iv: &Interval<Q>, max_width: &Q) -> Interval<Q> {
        let two = Ratio::from_integer(BigInt::from(2));
        let mut lo = iv.lower.clone();
        let mut hi = iv.upper.clone();
        let mut kind = iv.kind;
        let mut locs = self.roots.clone();
        // Every halving halves the width: `log₂(width/max_width)` of them
        // reach it (before 0.30 at most 512, short of it for a cell wider
        // than `2⁵¹²·max_width`, which a Cauchy bound of 2⁶⁶⁰ produces).
        let halvings = if max_width.is_positive() && hi > lo {
            let ratio = (&hi - &lo) / max_width;
            let bits = ratio.numer().bits() as i64 - ratio.denom().bits() as i64 + 2;
            usize::try_from(bits).unwrap_or(0).max(512)
        } else {
            512
        };
        for _ in 0..halvings {
            if &hi - &lo <= *max_width || lo == hi {
                break;
            }
            let mid = (&lo + &hi) / &two;
            if self.is_root(&mid) {
                return Interval::point(mid);
            }
            // Exactly one root in `(lo, mid]`?
            let at_mid = self.count_at_most_narrowing(&mut locs, &mid);
            let at_lo = self.count_at_most_narrowing(&mut locs, &lo);
            if at_mid.saturating_sub(at_lo) == 1 {
                hi = mid;
            } else {
                lo = mid;
            }
            // After a bisection the root is located in `(lo, hi]`.
            kind = IntervalKind::LeftOpen;
        }
        Interval {
            lower: lo,
            upper: hi,
            kind,
        }
    }

    /// Count distinct real roots in the **closed** interval `[a, b]`.
    pub fn count_roots_in_closed(&self, a: &Ratio<BigInt>, b: &Ratio<BigInt>) -> usize {
        if a > b {
            return 0;
        }
        let open_right = self.count_roots_in(a, b);
        open_right + usize::from(self.is_root(a))
    }

    /// Return the sign of the leading coefficient of the first (original)
    /// polynomial.  Useful for determining constant sign when there are
    /// no real roots.
    pub fn leading_sign_of_original(&self) -> i8 {
        leading_sign(&self.p0)
    }
}

/// A bisection cell of [`SturmChain::isolate_roots_in`] with the index
/// range `first..end` of the roots in it.
struct Cell {
    lo: Q,
    hi: Q,
    depth: u32,
    first: usize,
    end: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// The Descartes method
// ═══════════════════════════════════════════════════════════════════════════

/// Isolate the positive roots of the square-free integer polynomial `q`
/// (ascending, `q(0) ≠ 0`): the open isolating cells are returned, and a
/// root hit exactly by a bisection point is pushed onto `exact` (negated,
/// with the cells reflected, when `reflect` is set — the roots of `q(−x)`).
///
/// All positive roots lie in `(0, 2^e)` ([`root_bound_exp`]); a cell is
/// `(c, c + 1)·2^{e−k}`, carried as the integer polynomial whose roots in
/// `(0, 1)` correspond to those of `q` in the cell.
///
/// When the root magnitudes are spread over a wide range, `(0, 2^e)` is
/// first cut at powers of two around each cluster of magnitudes
/// ([`magnitude_pieces`]), and every piece is bisected on its own: a root
/// near `1` under a root bound of `2^{16600}` (a quartic with 5,000-digit
/// coefficients) is otherwise reached only after 16,600 halvings, each a
/// Taylor shift of 16,600-bit coefficients (0.3 s per isolation, called
/// four times by one sign query).  The roots found, hence every count and
/// every interval the queries return, are the same.
fn positive_roots(q: &[BigInt], exact: &mut Vec<RootLoc>, reflect: bool) -> Vec<(Q, Q)> {
    let mut cells: Vec<(Q, Q)> = Vec::new();
    let n = q.len().saturating_sub(1);
    if n == 0 {
        return cells;
    }
    let e = root_bound_exp(q);
    // Descartes' rule on `q` itself bounds the positive roots.
    match sign_variations_at_most_2(q.iter()) {
        0 => return cells,
        1 => {
            let hi = dyadic(&BigInt::one(), e, 0);
            cells.push(if reflect {
                (-hi, Q::zero())
            } else {
                (Q::zero(), hi)
            });
            return cells;
        }
        _ => {}
    }
    for (lo_exp, hi_exp) in magnitude_pieces(q, e) {
        let piece = piece_polynomial(q, lo_exp, hi_exp);
        if piece.root_at_lo {
            exact.push(RootLoc::Exact(if reflect {
                -piece.lo.clone()
            } else {
                piece.lo.clone()
            }));
        }
        bisect_piece(&piece, exact, reflect, &mut cells);
    }
    cells
}

/// One piece `(lo, lo + width)` of the positive half-line with the integer
/// polynomial `s` whose roots in `(0, 1)` are those of `q` in the piece
/// (`s(x)` a positive multiple of `q(lo + width·x)`, with any root at an
/// end of the piece divided out).
struct Piece {
    s: Vec<BigInt>,
    lo: Q,
    width: Q,
    /// `Some(b)` for the dyadic piece `(0, 2^b)`.
    dyadic_exp: Option<i64>,
    /// `q(lo) = 0` (recorded by the piece that starts there).
    root_at_lo: bool,
}

/// The [`Piece`] `(2^a, 2^b)`, or `(0, 2^b)` for `lo_exp = None`.
fn piece_polynomial(q: &[BigInt], lo_exp: Option<i64>, hi_exp: i64) -> Piece {
    let n = q.len() - 1;
    // `q(2^t x)` with integer coefficients.
    let scaled = |t: i64| -> Vec<BigInt> {
        if t >= 0 {
            let t = t.unsigned_abs();
            q.iter()
                .enumerate()
                .map(|(i, c)| c << (t * i as u64))
                .collect()
        } else {
            let t = t.unsigned_abs();
            q.iter()
                .enumerate()
                .map(|(i, c)| c << (t * (n - i) as u64))
                .collect()
        }
    };
    let pow2 = |t: i64| dyadic(&BigInt::one(), t, 0);
    let (mut s, lo, width) = match lo_exp {
        None => (scaled(hi_exp), Q::zero(), pow2(hi_exp)),
        Some(a) => {
            // q(2^a(1 + D·x)), D = 2^{b−a} − 1.
            let mut s = scaled(a);
            taylor_shift_1(&mut s);
            let d = (BigInt::one() << (hi_exp - a).unsigned_abs()) - 1;
            let mut dp = BigInt::one();
            for c in s.iter_mut().skip(1) {
                dp *= &d;
                *c *= &dp;
            }
            let width = crate::poly::modgcd::rat_add(&pow2(hi_exp), &-pow2(a));
            (s, pow2(a), width)
        }
    };
    let mut root_at_lo = false;
    if lo_exp.is_some() && s.first().is_some_and(Zero::is_zero) {
        s.remove(0);
        root_at_lo = true;
    }
    if s.iter().sum::<BigInt>().is_zero() {
        s = div_by_x_minus_1(&s);
    }
    strip_power_of_two(&mut s);
    Piece {
        s,
        lo,
        width,
        dyadic_exp: lo_exp.is_none().then_some(hi_exp),
        root_at_lo,
    }
}

/// The Descartes bisection of one [`Piece`]: isolating cells pushed onto
/// `cells`, roots hit by a bisection point onto `exact` (both negated when
/// `reflect` is set).
fn bisect_piece(piece: &Piece, exact: &mut Vec<RootLoc>, reflect: bool, cells: &mut Vec<(Q, Q)>) {
    // The point `lo + width·c/2^k`.
    let point = |c: &BigInt, k: u64| -> Q {
        use crate::poly::modgcd::{rat_add, rat_mul};
        match piece.dyadic_exp {
            Some(b) => dyadic(c, b, k),
            None => rat_add(&piece.lo, &rat_mul(&piece.width, &dyadic(c, 0, k))),
        }
    };
    let emit = |c: &BigInt, k: u64, cells: &mut Vec<(Q, Q)>| {
        let lo = point(c, k);
        let hi = point(&(c + 1), k);
        if reflect {
            cells.push((-hi, -lo));
        } else {
            cells.push((lo, hi));
        }
    };
    if piece.s.len() < 2 {
        return;
    }
    let mut stack: Vec<(Vec<BigInt>, BigInt, u64)> = vec![(piece.s.clone(), BigInt::zero(), 0)];
    while let Some((s, c, k)) = stack.pop() {
        match descartes_bound(&s) {
            0 => {}
            1 => emit(&c, k, cells),
            _ => {
                // Left half: `2^m s(x/2)`; right half: its Taylor shift.
                let m = s.len() - 1;
                let mut left: Vec<BigInt> =
                    s.iter().enumerate().map(|(i, a)| a << (m - i)).collect();
                let c2: BigInt = &c << 1u32;
                let at_mid: BigInt = left.iter().sum();
                if at_mid.is_zero() {
                    let r = point(&(&c2 + 1), k + 1);
                    exact.push(RootLoc::Exact(if reflect { -r } else { r }));
                    left = div_by_x_minus_1(&left);
                }
                let mut right = left.clone();
                taylor_shift_1(&mut right);
                strip_power_of_two(&mut left);
                strip_power_of_two(&mut right);
                stack.push((right, &c2 + 1, k + 1));
                stack.push((left, c2, k + 1));
            }
        }
    }
}

/// The pieces `(2^a, 2^b)` (`a = None` for `0`) that [`positive_roots`]
/// bisects separately: `(0, 2^e)` itself unless the root magnitudes of `q`
/// are spread over more than `4δ + 64` bits, `δ = 2n + 4`; otherwise
/// `(2^{m−δ}, 2^{m+δ})` around each estimated magnitude `2^m` and the gaps
/// between them.  The magnitudes are the slopes of the upper convex hull of
/// the points `(i, bits(qᵢ))` (the Newton polygon: the roots of a hull
/// segment have moduli within a factor polynomial in `n` of `2^m`).  Only
/// the cost depends on these estimates: every piece is searched exactly.
fn magnitude_pieces(q: &[BigInt], e: i64) -> Vec<(Option<i64>, i64)> {
    let whole = vec![(None, e)];
    let n = q.len().saturating_sub(1);
    let delta = 2 * n as i64 + 4;
    let mut hull: Vec<(i64, i64)> = Vec::new();
    for (i, c) in q.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let pt = (i as i64, i64::try_from(c.bits()).unwrap_or(i64::MAX / 4));
        while let [.., a, b] = hull[..] {
            let cross = (b.0 - a.0) as i128 * (pt.1 - a.1) as i128
                - (b.1 - a.1) as i128 * (pt.0 - a.0) as i128;
            if cross >= 0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(pt);
    }
    // Magnitude ranges [floor(m) − δ, ceil(m) + δ] of the hull segments,
    // ascending (the slopes of an upper hull decrease).
    let mut ranges: Vec<(i64, i64)> = Vec::new();
    for w in hull.windows(2) {
        let (dx, dy) = (w[1].0 - w[0].0, w[0].1 - w[1].1);
        let lo = dy.div_euclid(dx) - delta;
        let hi = dy.div_euclid(dx) + i64::from(dy.rem_euclid(dx) != 0) + delta;
        match ranges.last_mut() {
            Some(last) if lo <= last.1 + 8 => last.1 = last.1.max(hi),
            _ => ranges.push((lo, hi)),
        }
    }
    let (Some(first), Some(last)) = (ranges.first(), ranges.last()) else {
        return whole;
    };
    if last.1 - first.0 <= 4 * delta + 64 {
        return whole;
    }
    let mut cuts: Vec<i64> = Vec::new();
    for (lo, hi) in ranges {
        for t in [lo, hi] {
            if t < e && cuts.last().is_none_or(|&l| t > l) {
                cuts.push(t);
            }
        }
    }
    let mut pieces = Vec::with_capacity(cuts.len() + 1);
    let mut prev = None;
    for t in cuts {
        pieces.push((prev, t));
        prev = Some(t);
    }
    pieces.push((prev, e));
    pieces
}

/// `e` with every root of `q` (ascending, `q(0) ≠ 0`, degree `n ≥ 1`) of
/// modulus `< 2^e`: Fujiwara's bound `|z| ≤ 2·maxₖ |a_{n−k}/a_n|^{1/k}`, each
/// ratio bounded by `2^{bits(a_{n−k}) − bits(a_n) + 1}` and each `k`-th root
/// rounded up to a power of two.
fn root_bound_exp(q: &[BigInt]) -> i64 {
    let n = q.len() - 1;
    let bn = i64::try_from(q[n].bits()).unwrap_or(i64::MAX / 4);
    let mut best: Option<i64> = None;
    for k in 1..=n {
        let a = &q[n - k];
        if a.is_zero() {
            continue;
        }
        let ba = i64::try_from(a.bits()).unwrap_or(i64::MAX / 4);
        let d = ba - bn + 1;
        let k = k as i64;
        let t = d.div_euclid(k) + i64::from(d.rem_euclid(k) != 0);
        best = Some(best.map_or(t, |b| b.max(t)));
    }
    1 + best.unwrap_or(0)
}

/// `c · 2^{e−k}` as a rational.
fn dyadic(c: &BigInt, e: i64, k: u64) -> Q {
    let k = i64::try_from(k).unwrap_or(i64::MAX / 4);
    let sh = e - k;
    if sh >= 0 {
        Ratio::from_integer(c << sh.unsigned_abs())
    } else {
        Ratio::new(c.clone(), BigInt::one() << sh.unsigned_abs())
    }
}

/// Descartes' bound on the number of roots of `s` in `(0, 1)`: the sign
/// variations of `(x + 1)ᵐ s(1/(x + 1))`, capped at 2 (all the method needs
/// to know is 0, 1 or more).  The Taylor shift runs in place on the
/// reversed coefficients; after its `i`-th pass the coefficient of `xⁱ` is
/// final, so the count stops as soon as it reaches 2.
fn descartes_bound(s: &[BigInt]) -> u8 {
    let m = s.len().saturating_sub(1);
    if m == 0 {
        return 0;
    }
    let mut a: Vec<BigInt> = s.iter().rev().cloned().collect();
    let mut variations = 0u8;
    let mut last = 0i8;
    let mut see = |x: &BigInt, variations: &mut u8| {
        let sg = sign_of(x);
        if sg != 0 {
            if last != 0 && sg != last {
                *variations += 1;
            }
            last = sg;
        }
    };
    for i in 0..m {
        for j in (i..m).rev() {
            let (low, high) = a.split_at_mut(j + 1);
            low[j] += &high[0];
        }
        see(&a[i], &mut variations);
        if variations >= 2 {
            return 2;
        }
    }
    see(&a[m], &mut variations);
    variations.min(2)
}

/// The sign variations of a coefficient sequence (zeros skipped), capped
/// at 2.
fn sign_variations_at_most_2<'a>(coeffs: impl Iterator<Item = &'a BigInt>) -> u8 {
    let mut variations = 0u8;
    let mut last = 0i8;
    for c in coeffs {
        let sg = sign_of(c);
        if sg != 0 {
            if last != 0 && sg != last {
                variations += 1;
                if variations >= 2 {
                    return 2;
                }
            }
            last = sg;
        }
    }
    variations
}

/// `a(x + 1)` in place (ascending coefficients).
fn taylor_shift_1(a: &mut [BigInt]) {
    let m = a.len().saturating_sub(1);
    for i in 0..m {
        for j in (i..m).rev() {
            let (low, high) = a.split_at_mut(j + 1);
            low[j] += &high[0];
        }
    }
}

/// `a / (x − 1)` for `a(1) = 0` (synthetic division; ascending).
fn div_by_x_minus_1(a: &[BigInt]) -> Vec<BigInt> {
    let m = a.len().saturating_sub(1);
    if m == 0 {
        return Vec::new();
    }
    let mut b = vec![BigInt::zero(); m];
    b[m - 1] = a[m].clone();
    for i in (1..m).rev() {
        b[i - 1] = &a[i] + &b[i];
    }
    b
}

/// Divide out the largest power of two dividing every coefficient.
fn strip_power_of_two(s: &mut [BigInt]) {
    if let Some(t) = s.iter().filter_map(BigInt::trailing_zeros).min()
        && t > 0
    {
        for c in s.iter_mut() {
            *c >>= t;
        }
    }
}

fn sign_of(x: &BigInt) -> i8 {
    match x.sign() {
        num_bigint::Sign::Plus => 1,
        num_bigint::Sign::Minus => -1,
        num_bigint::Sign::NoSign => 0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Square-freeness modulo primes
// ═══════════════════════════════════════════════════════════════════════════

/// Is the integer polynomial `ip` (ascending) certainly square-free?  True
/// when `gcd(ip mod m, ip′ mod m) = 1` over `𝔽_m` for one of the
/// [`SQUARE_FREE_PRIMES`] `m` not dividing the leading coefficient: a
/// common factor `g` of `ip` and `ip′` over `ℚ`, taken primitive in `ℤ[x]`,
/// has a leading coefficient dividing `lc(ip)` (Gauss's lemma), so it
/// would reduce to a common factor of the same positive degree modulo
/// `m`.  `false` means "not certified" (then the caller computes the
/// square-free part exactly).
fn certified_square_free(ip: &[BigInt]) -> bool {
    let n = match ip.iter().rposition(|c| !c.is_zero()) {
        Some(n) => n,
        None => return false,
    };
    if n <= 1 {
        return true;
    }
    SQUARE_FREE_PRIMES.iter().any(|&m| {
        let mb = BigInt::from(m);
        let f: Vec<u64> = ip[..=n]
            .iter()
            .map(|c| c.mod_floor(&mb).to_u64().unwrap_or(0))
            .collect();
        if f[n] == 0 {
            return false;
        }
        let df: Vec<u64> = (1..=n).map(|i| mul_mod(f[i], i as u64 % m, m)).collect();
        coprime_mod(f, df, m)
    })
}

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(m)) as u64
}

/// `a⁻¹ mod m` (extended Euclid), `None` when `gcd(a, m) ≠ 1`.
fn inv_mod(a: u64, m: u64) -> Option<u64> {
    let (mut r0, mut r1) = (i128::from(m), i128::from(a % m));
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    if r0 != 1 {
        return None;
    }
    u64::try_from(t0.rem_euclid(i128::from(m))).ok()
}

/// Are `a` and `b` (ascending coefficients mod `m`) coprime over `ℤ/m`,
/// certified by Euclid's algorithm with every leading coefficient a unit
/// and a unit last remainder?  `false` when that is not established.
fn coprime_mod(mut a: Vec<u64>, mut b: Vec<u64>, m: u64) -> bool {
    let trim = |v: &mut Vec<u64>| {
        while v.last() == Some(&0) {
            v.pop();
        }
    };
    trim(&mut a);
    trim(&mut b);
    loop {
        let Some(&lead) = b.last() else {
            // gcd = a: a unit only when it is a non-zero constant.
            return a.len() == 1 && inv_mod(a[0], m).is_some();
        };
        let Some(inv) = inv_mod(lead, m) else {
            return false;
        };
        if b.len() == 1 {
            return true;
        }
        while a.len() >= b.len() {
            let Some(&top) = a.last() else {
                break;
            };
            let q = mul_mod(top, inv, m);
            let shift = a.len() - b.len();
            for (i, &bi) in b.iter().enumerate() {
                let t = mul_mod(q, bi, m);
                let v = a[shift + i];
                a[shift + i] = if v >= t { v - t } else { v + (m - t) };
            }
            trim(&mut a);
        }
        std::mem::swap(&mut a, &mut b);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// `x = a / b` with `b > 0`.
fn numer_denom(x: &Ratio<BigInt>) -> (BigInt, BigInt) {
    if x.denom().is_negative() {
        (-x.numer(), -x.denom())
    } else {
        (x.numer().clone(), x.denom().clone())
    }
}

/// Sign of the integer polynomial `c` (ascending) at `a / b` with `b > 0`:
/// the sign of `b^n · c(a/b) = Σ cᵢ aⁱ b^{n−i}`, by Horner's rule in `ℤ`.
/// `b_pows[j] = b^j` are precomputed powers (any missing power is computed
/// on the spot).
fn int_sign_at(c: &[BigInt], a: &BigInt, b: &BigInt, b_pows: &[BigInt]) -> i8 {
    let Some((lead, rest)) = c.split_last() else {
        return 0;
    };
    let n = rest.len();
    let mut v = lead.clone();
    for (i, ci) in rest.iter().enumerate().rev() {
        v *= a;
        if !ci.is_zero() {
            match b_pows.get(n - i) {
                Some(bp) => v += ci * bp,
                None => v += ci * b.pow((n - i) as u32),
            }
        }
    }
    sign_of(&v)
}

/// The sign of the integer polynomial whose coefficients rounded to `p =
/// FILTER_PREC` bits are `c` (ascending) at the rational whose rounding is
/// `x`, when floating-point Horner decides it: `None` otherwise.  With the
/// unit roundoff `u = 2^{−p}` and `n = deg`, the coefficients, `x` and each
/// of the `2n` Horner operations contribute a relative error `u` per
/// factor, so the computed value is within `γ_{3n+2}·Σ|cᵢ||x|ⁱ` of the exact
/// one (Higham, *Accuracy and Stability of Numerical Algorithms*, §5.1),
/// `γ_k = k·u/(1 − k·u)`; the sum is itself computed alongside (another
/// `γ_{2n}`), and the sign is taken only when `|v| ≥ 2^{e_v − 1}` exceeds
/// four times the bound.
fn float_sign(c: &[BigFloat], x: &BigFloat) -> Option<i8> {
    let (lead, rest) = c.split_last()?;
    let rm = RoundingMode::ToEven;
    let p = FILTER_PREC;
    let ax = x.abs();
    let mut v = lead.clone();
    let mut s = lead.abs();
    for ci in rest.iter().rev() {
        v = v.mul(x, p, rm).add(ci, p, rm);
        s = s.mul(&ax, p, rm).add(&ci.abs(), p, rm);
    }
    if v.is_zero() || v.is_nan() || v.is_inf() || s.is_nan() || s.is_inf() {
        return None;
    }
    let ops = 3 * rest.len() as i64 + 4;
    let k = 64 - i64::from(ops.leading_zeros());
    let (ev, es) = (i64::from(v.exponent()?), i64::from(s.exponent()?));
    if ev - 1 > es + k + 2 - p as i64 {
        Some(if v.is_negative() { -1 } else { 1 })
    } else {
        None
    }
}

/// Cauchy root bound: every root `z` of `p` satisfies
/// `|z| ≤ 1 + maxᵢ |aᵢ / aₙ|`.
pub(crate) fn cauchy_bound(p: &Poly) -> Ratio<BigInt> {
    let one = Ratio::from_integer(BigInt::from(1));
    let Some(n) = p.degree() else {
        return one;
    };
    if n == 0 {
        return one;
    }
    // max |aᵢ/aₙ| = (max |aᵢ|)/|aₙ|: one division (with Lehmer's gcd)
    // instead of one reduced fraction per coefficient (binary gcds).
    let lc = p.coeff(n).abs();
    let mut max = Ratio::from_integer(BigInt::from(0));
    for i in 0..n {
        let a = p.coeff(i).abs();
        if a > max {
            max = a;
        }
    }
    let ratio = crate::poly::traits::Field::div(&max, &lc);
    crate::poly::modgcd::rat_add(&ratio, &one)
}

/// Sign of the leading coefficient: +1, −1, or 0 (for zero poly).
fn leading_sign(p: &Poly) -> i8 {
    match p.leading_coeff() {
        None => 0,
        Some(c) if c.is_positive() => 1,
        Some(c) if c.is_negative() => -1,
        Some(_) => 0, // zero (shouldn't happen after normalisation)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::zpoly::{pseudo_rem_pos, z_sturm_sequence};
    use num_bigint::BigInt;
    use num_rational::Ratio;
    use num_traits::ToPrimitive;

    fn r(n: i64) -> Ratio<BigInt> {
        Ratio::from_integer(BigInt::from(n))
    }

    fn q(n: i64, d: i64) -> Q {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    // ── The Sturm-chain implementation of 0.30, the reference ───────────────────────────

    /// The Sturm chain of the square-free part in `ℤ[x]` (the subresultant
    /// sequence with fixed signs, [`z_sturm_sequence`]), exact signs, and
    /// the bisections of 0.30 verbatim.
    struct SturmReference {
        p0: Poly,
        chain: Vec<Vec<BigInt>>,
    }

    impl SturmReference {
        fn new(p: &Poly) -> Self {
            let p0 = p.square_free_part();
            let i0 = integer_scaled(p0.coeffs());
            let i1 = integer_scaled(p0.derivative().coeffs());
            let chain = if i1.is_empty() {
                vec![i0]
            } else {
                match z_sturm_sequence(&i0, &i1) {
                    Some(seq) => seq,
                    None => {
                        let mut c = vec![i0, i1];
                        loop {
                            let n = c.len();
                            let rem = pseudo_rem_pos(&c[n - 2], &c[n - 1]);
                            if rem.is_empty() {
                                break;
                            }
                            c.push(z_primitive(
                                &rem.into_iter().map(|x| -x).collect::<Vec<_>>(),
                            ));
                        }
                        c
                    }
                }
            };
            SturmReference { p0, chain }
        }

        fn var(&self, x: &Q) -> usize {
            let (a, b) = numer_denom(x);
            let n = self.chain.iter().map(Vec::len).max().unwrap_or(1);
            let b_pows = powers(&b, n);
            let signs: Vec<i8> = self
                .chain
                .iter()
                .map(|c| int_sign_at(c, &a, &b, &b_pows))
                .filter(|&s| s != 0)
                .collect();
            signs.windows(2).filter(|w| w[0] != w[1]).count()
        }

        fn var_at_infinity(&self, negative: bool) -> usize {
            let signs: Vec<i8> = self
                .chain
                .iter()
                .filter_map(|c| {
                    let lc = sign_of(c.last()?);
                    let odd = c.len() % 2 == 0;
                    Some(if negative && odd { -lc } else { lc })
                })
                .filter(|&s| s != 0)
                .collect();
            signs.windows(2).filter(|w| w[0] != w[1]).count()
        }

        fn count_real_roots(&self) -> usize {
            self.var_at_infinity(true)
                .saturating_sub(self.var_at_infinity(false))
        }

        fn count_roots_in(&self, a: &Q, b: &Q) -> usize {
            self.var(a).saturating_sub(self.var(b))
        }

        fn is_root(&self, x: &Q) -> bool {
            let (a, b) = numer_denom(x);
            let b_pows = powers(&b, self.chain[0].len());
            int_sign_at(&self.chain[0], &a, &b, &b_pows) == 0
        }

        fn isolate_all_real_roots(&self, depth: u32) -> Vec<Interval<Q>> {
            if self.p0.degree().unwrap_or(0) == 0 {
                return vec![];
            }
            let bound = cauchy_bound(&self.p0) + r(1);
            let two = r(2);
            let mut result = Vec::new();
            let mut stack = vec![(-bound.clone(), bound.clone(), depth)];
            while let Some((lo, hi, d)) = stack.pop() {
                let n = self.count_roots_in(&lo, &hi);
                if n == 0 {
                    continue;
                }
                if n == 1 || d == 0 {
                    result.push(Interval::left_open(lo, hi));
                    continue;
                }
                let mid = (&lo + &hi) / &two;
                stack.push((mid.clone(), hi, d - 1));
                stack.push((lo, mid, d - 1));
            }
            result
                .into_iter()
                .map(|iv| {
                    if self.is_root(&iv.upper) {
                        Interval::point(iv.upper)
                    } else {
                        iv
                    }
                })
                .collect()
        }

        fn refine_interval(&self, iv: &Interval<Q>, max_width: &Q) -> Interval<Q> {
            let two = r(2);
            let (mut lo, mut hi, mut kind) = (iv.lower.clone(), iv.upper.clone(), iv.kind);
            let halvings = if max_width.is_positive() && hi > lo {
                let ratio = (&hi - &lo) / max_width;
                let bits = ratio.numer().bits() as i64 - ratio.denom().bits() as i64 + 2;
                usize::try_from(bits).unwrap_or(0).max(512)
            } else {
                512
            };
            for _ in 0..halvings {
                if &hi - &lo <= *max_width || lo == hi {
                    break;
                }
                let mid = (&lo + &hi) / &two;
                if self.is_root(&mid) {
                    return Interval::point(mid);
                }
                if self.count_roots_in(&lo, &mid) == 1 {
                    hi = mid;
                } else {
                    lo = mid;
                }
                kind = IntervalKind::LeftOpen;
            }
            Interval {
                lower: lo,
                upper: hi,
                kind,
            }
        }
    }

    /// Every query of the Descartes implementation against the Sturm-chain
    /// reference on `p`: counts, isolating intervals (identical), refined
    /// intervals (identical), counts over `(a, b]` and `[a, b]` at rational
    /// points around every root.  Returns the number of comparisons.
    fn assert_same_as_sturm(p: &Poly, label: &str) -> usize {
        let new = SturmChain::new(p);
        let old = SturmReference::new(p);
        assert_eq!(new.p0, old.p0, "{label}: square-free part");
        assert_eq!(
            new.count_real_roots(),
            old.count_real_roots(),
            "{label}: count"
        );
        let bound = cauchy_bound(&new.p0) + r(1);
        let depth = new.isolation_depth(&bound);
        let ivs = new.isolate_all_real_roots();
        assert_eq!(ivs, old.isolate_all_real_roots(depth), "{label}: isolation");
        let mut checks = 2;
        let mut xs: Vec<Q> = vec![r(0), r(1), r(-1), q(1, 2), -bound.clone(), bound.clone()];
        for (w_num, w_den) in [(1i64, 1024i64), (1, 1 << 40)] {
            let width = q(w_num, w_den);
            for iv in &ivs {
                let fine = new.refine_interval(iv, &width);
                assert_eq!(
                    fine,
                    old.refine_interval(iv, &width),
                    "{label}: refine {iv:?}"
                );
                checks += 1;
                xs.push(fine.lower.clone());
                xs.push(fine.upper.clone());
                xs.push((&fine.lower + &fine.upper) / r(2));
            }
        }
        for iv in &ivs {
            xs.push(iv.lower.clone());
            xs.push(iv.upper.clone());
        }
        for (i, a) in xs.iter().enumerate() {
            for b in xs.iter().skip(i % 3).step_by(3) {
                assert_eq!(
                    new.count_roots_in(a, b),
                    old.count_roots_in(a, b),
                    "{label}: count in ({a}, {b}]"
                );
                let closed_old = if a > b {
                    0
                } else {
                    old.count_roots_in(a, b) + usize::from(old.is_root(a))
                };
                assert_eq!(
                    new.count_roots_in_closed(a, b),
                    closed_old,
                    "{label}: count in [{a}, {b}]"
                );
                checks += 2;
            }
        }
        checks
    }

    /// A deterministic pseudo-random polynomial with small integer
    /// coefficients (some repeated factors when `square` is set).
    fn pseudo_random_poly(seed: u64, degree: usize, square: bool) -> Poly {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 19) as i64 - 9
        };
        let mut coeffs: Vec<Ratio<BigInt>> = (0..=degree).map(|_| r(next())).collect();
        if coeffs[degree].is_zero() {
            coeffs[degree] = r(1);
        }
        let p = Poly::from_coeffs(coeffs);
        if square {
            let q = Poly::from_coeffs(vec![r(next()), r(next()), r(1)]);
            &(&p * &q) * &q
        } else {
            p
        }
    }

    /// A pseudo-random integer polynomial with coefficients of about
    /// `bits` bits.
    fn big_random_poly(seed: u64, degree: usize, bits: u32) -> Poly {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut word = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let coeffs: Vec<Ratio<BigInt>> = (0..=degree)
            .map(|_| {
                let mut c = BigInt::from(0);
                for _ in 0..bits.div_ceil(64) {
                    c = (c << 64) + BigInt::from(word());
                }
                c >>= (64 * bits.div_ceil(64) - bits) as usize;
                if word() % 2 == 0 {
                    c = -c;
                }
                Ratio::from_integer(c)
            })
            .collect();
        Poly::from_coeffs(coeffs)
    }

    fn linear(a: &Q) -> Poly {
        Poly::from_coeffs(vec![-a.clone(), r(1)])
    }

    /// The Descartes isolation gives exactly the counts and intervals of
    /// the Sturm-chain bisection of 0.30 — the byte identity the
    /// certificates built on root isolation rely on — over small and large
    /// integer coefficients, repeated factors, rational roots hit exactly by
    /// bisection points (including 0 and dyadic roots), clustered roots,
    /// roots of `10¹⁸` next to roots of `10⁻¹⁸`, and Mignotte polynomials.
    #[test]
    fn descartes_isolation_replays_the_sturm_bisection_exactly() {
        let mut checks = 0usize;
        for seed in 1..=16u64 {
            let p = pseudo_random_poly(seed, 3 + (seed as usize * 5) % 12, seed % 3 == 0);
            checks += assert_same_as_sturm(&p, &format!("small {seed}"));
        }
        for seed in 1..=10u64 {
            let degree = 3 + (seed as usize * 7) % 14;
            let bits = [8u32, 40, 120, 400][seed as usize % 4];
            let p = big_random_poly(seed, degree, bits);
            if p.degree().unwrap_or(0) >= 1 {
                checks += assert_same_as_sturm(&p, &format!("big {seed}"));
            }
        }
        // Rational roots on bisection points, a double root, 0.
        let roots = [
            q(0, 1),
            q(1, 2),
            q(-3, 1),
            q(5, 8),
            q(7, 3),
            q(-1, 1024),
            q(1, 1),
        ];
        let mut p = Poly::from_coeffs(vec![r(1)]);
        for (i, a) in roots.iter().enumerate() {
            p = &p * &linear(a);
            checks += assert_same_as_sturm(&p, &format!("rational roots {i}"));
        }
        let p2 = &p * &linear(&q(5, 8));
        checks += assert_same_as_sturm(&p2, "double root");
        // Clustered roots 2^-30 apart, and huge next to tiny.
        let a = q(1_000_003, 999);
        let eps = Ratio::new(BigInt::from(1), BigInt::from(1) << 30);
        let cl = &(&linear(&a) * &linear(&(&a + &eps))) * &linear(&(&a - &eps));
        checks += assert_same_as_sturm(&(&cl + &Poly::from_coeffs(vec![q(1, 1 << 20)])), "cluster");
        let big = Ratio::from_integer(BigInt::from(10).pow(18) + BigInt::from(7));
        let tiny = Ratio::new(BigInt::from(3), BigInt::from(10).pow(18));
        let ht = &(&linear(&big) * &linear(&tiny)) * &(&linear(&-tiny.clone()) * &linear(&r(2)));
        checks += assert_same_as_sturm(&ht, "huge and tiny");
        // Mignotte: x^n - 2(ax - 1)^2 has two roots within ~a^{-(n+2)/2}.
        for (n, a) in [(5usize, 10i64), (8, 30), (11, 7)] {
            let mut xn = vec![r(0); n + 1];
            xn[n] = r(1);
            let lin = Poly::from_coeffs(vec![r(-1), r(a)]);
            let m = &Poly::from_coeffs(xn) - &(&(&lin * &lin) * &Poly::from_coeffs(vec![r(2)]));
            checks += assert_same_as_sturm(&m, &format!("mignotte {n} {a}"));
        }
        // Random 30-digit rational coefficients.
        for seed in 1..=4u64 {
            let num = big_random_poly(seed + 50, 7, 100);
            let den = big_random_poly(seed + 90, 7, 100);
            let coeffs: Vec<Q> = (0..=7)
                .map(|i| {
                    let d = den.coeff(i).numer().abs() + BigInt::from(1);
                    Ratio::new(num.coeff(i).numer().clone(), d)
                })
                .collect();
            checks += assert_same_as_sturm(&Poly::from_coeffs(coeffs), &format!("rat {seed}"));
        }
        assert!(checks > 5000, "{checks}");
    }

    /// The isolating cells themselves: disjoint, sorted, each with exactly
    /// one root (a sign change of the square-free part, or an exact zero).
    #[test]
    fn descartes_cells_isolate() {
        for seed in 1..=12u64 {
            let p = big_random_poly(seed + 7, 6 + seed as usize, 60);
            let chain = SturmChain::new(&p);
            for w in chain.roots.windows(2) {
                let hi0 = match &w[0] {
                    RootLoc::Exact(r) => r.clone(),
                    RootLoc::Open { hi, .. } => hi.clone(),
                };
                let lo1 = match &w[1] {
                    RootLoc::Exact(r) => r.clone(),
                    RootLoc::Open { lo, .. } => lo.clone(),
                };
                assert!(hi0 <= lo1, "seed {seed}: {:?}", chain.roots);
            }
            for loc in &chain.roots {
                match loc {
                    RootLoc::Exact(x) => assert!(chain.p0.eval(x).is_zero()),
                    RootLoc::Open { lo, hi, after_lo } => {
                        let sl = chain.p0.eval(lo);
                        let sh = chain.p0.eval(hi);
                        assert!(sh.is_zero() || sign_of(sh.numer()) == -after_lo);
                        assert!(sl.is_zero() || sign_of(sl.numer()) == *after_lo);
                    }
                }
            }
        }
    }

    /// Deterministic Miller–Rabin for `u64` (bases 2..37).
    fn is_prime_u64(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        let pow = |mut b: u64, mut e: u64| {
            let mut acc = 1u64;
            b %= n;
            while e > 0 {
                if e & 1 == 1 {
                    acc = mul_mod(acc, b, n);
                }
                b = mul_mod(b, b, n);
                e >>= 1;
            }
            acc
        };
        let (mut d, mut s) = (n - 1, 0);
        while d % 2 == 0 {
            d /= 2;
            s += 1;
        }
        [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            .iter()
            .all(|&a| {
                if a % n == 0 {
                    return true;
                }
                let mut x = pow(a, d);
                if x == 1 || x == n - 1 {
                    return true;
                }
                for _ in 1..s {
                    x = mul_mod(x, x, n);
                    if x == n - 1 {
                        return true;
                    }
                }
                false
            })
    }

    /// The modular certificate: the moduli are primes, a square-free
    /// polynomial is certified, one with a repeated factor never is.
    #[test]
    fn square_freeness_certificate() {
        for m in SQUARE_FREE_PRIMES {
            assert!(is_prime_u64(m), "{m}");
        }
        for seed in 1..=10u64 {
            let p = big_random_poly(seed, 5 + seed as usize, 200);
            let ip = integer_scaled(p.coeffs());
            let sqf = p.square_free_part();
            assert_eq!(
                certified_square_free(&ip),
                sqf.degree() == p.degree(),
                "seed {seed}"
            );
            let sq = &p * &linear(&q(seed as i64, 3));
            let sq = &sq * &linear(&q(seed as i64, 3));
            assert!(!certified_square_free(&integer_scaled(sq.coeffs())));
        }
        assert!(!certified_square_free(&[]));
        assert!(certified_square_free(&[BigInt::from(3), BigInt::from(-7)]));
    }

    /// `count_real_roots` of a degree-40 polynomial with 41 random 30-digit
    /// rational coefficients (denominator lcm of ~1,230 digits) took 4.3 s
    /// in a release build, all of it in the Sturm chain's subresultant PRS
    /// (the Descartes method: 0.13 s in a debug build).
    #[test]
    fn huge_denominators_are_fast() {
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        let mut digits = |n: usize| {
            let mut s = String::new();
            for i in 0..n {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let d = (state % 10) as u8;
                s.push(char::from(b'0' + if i == 0 { d % 9 + 1 } else { d }));
            }
            s.parse::<BigInt>().unwrap_or_default()
        };
        let coeffs: Vec<Q> = (0..=40)
            .map(|i| {
                let n = digits(30);
                let d = digits(30);
                Ratio::new(if i % 3 == 1 { -n } else { n }, d)
            })
            .collect();
        let p = Poly::from_coeffs(coeffs);
        let start = std::time::Instant::now();
        let chain = SturmChain::new(&p);
        let ivs = chain.isolate_all_real_roots();
        let width = q(1, 1024);
        for iv in &ivs {
            let fine = chain.refine_interval(iv, &width);
            assert!(fine.width() <= width);
            assert_eq!(chain.count_roots_in(&fine.lower, &fine.upper), 1);
        }
        assert!(start.elapsed().as_secs() < 30, "took {:?}", start.elapsed());
        assert_eq!(chain.count_real_roots(), ivs.len());
        assert_eq!(chain.count_real_roots(), HUGE_DENOMINATORS_ROOTS);
    }

    /// The real roots of the polynomial of [`huge_denominators_are_fast`]:
    /// SymPy 1.14 `Poly(p, x).intervals()` = `[((-2, -1), 1), ((1, 2), 1)]`;
    /// mpmath 1.3.0 `polyroots` at 200 digits has two real roots, −1.0941…
    /// and 1.5147… (target/scratch/rr_oracle.py, rr_oracle_mp.py).
    const HUGE_DENOMINATORS_ROOTS: usize = 2;

    /// `Σ_{j=k}^{n} C(n,j) x^j (1−x)^{n−j} − 1/denom`: the Clopper–Pearson
    /// tail polynomial, whose Cauchy bound is huge (`> 6·10⁷` at `n = 40`)
    /// while all roots lie in `|z| < 1.5`.
    fn binomial_tail(n: usize, k: usize, denom: i64) -> Poly {
        let mut acc = Poly::zero();
        let one_minus_x = Poly::from_coeffs(vec![r(1), r(-1)]);
        let x = Poly::x();
        for j in k..=n {
            let mut c = r(1);
            for i in 0..j {
                c *= Ratio::new(BigInt::from(n - i), BigInt::from(i + 1));
            }
            let mut term = Poly::from_coeffs(vec![c]);
            for _ in 0..j {
                term = &term * &x;
            }
            for _ in 0..(n - j) {
                term = &term * &one_minus_x;
            }
            acc = &acc + &term;
        }
        &acc - &Poly::from_coeffs(vec![Ratio::new(BigInt::from(1), BigInt::from(denom))])
    }

    /// Isolating and refining the roots of the degree-40 tail polynomial
    /// bisects a width-`10⁸` Cauchy interval down to `1/1024` — about 36
    /// halvings per root — and used to take 16 s with rational Horner
    /// evaluation; with integer signs it is milliseconds.
    #[test]
    fn degree_40_tail_isolation_is_fast_and_correct() {
        let f = binomial_tail(40, 12, 40);
        assert_eq!(f.degree(), Some(40));
        let start = std::time::Instant::now();
        let chain = SturmChain::new(&f);
        let ivs = chain.isolate_all_real_roots();
        assert_eq!(ivs.len(), 2, "{ivs:?}");
        let width = Ratio::new(BigInt::from(1), BigInt::from(1024));
        let refined: Vec<Interval<Q>> = ivs
            .iter()
            .map(|iv| chain.refine_interval(iv, &width))
            .collect();
        assert!(start.elapsed().as_secs() < 5, "took {:?}", start.elapsed());
        for (iv, fine) in ivs.iter().zip(&refined) {
            assert!(&fine.upper - &fine.lower <= width);
            assert!(iv.lower <= fine.lower && fine.upper <= iv.upper);
            assert_eq!(chain.count_roots_in(&fine.lower, &fine.upper), 1);
        }
        // The root in (0, 1) is the 2.5% Clopper–Pearson lower bound for
        // 12 successes in 40 trials: 0.16562720439… (scipy.stats.beta.ppf).
        let lo = refined[1].lower.to_f64().unwrap_or(f64::NAN);
        let hi = refined[1].upper.to_f64().unwrap_or(f64::NAN);
        assert!(
            lo <= 0.1656272043932356 && 0.1656272043932356 <= hi,
            "[{lo}, {hi}]"
        );
        assert!(assert_same_as_sturm(&f, "tail") > 0);
    }

    /// Root magnitudes spread over hundreds of bits: the positive half-line
    /// is cut into pieces around each magnitude (`magnitude_pieces`), with
    /// the same roots, counts and intervals as the Sturm reference.
    #[test]
    fn spread_root_magnitudes_match_sturm() {
        let ten = |k: u32| Ratio::from_integer(BigInt::from(10).pow(k));
        let two = |k: u32| Ratio::from_integer(BigInt::from(2).pow(k));
        let lin = |a: Q, b: Q| Poly::from_coeffs(vec![-a, b]);
        let factors = [
            lin(-ten(90), r(1)),
            lin(r(3), r(1)),
            Poly::from_coeffs(vec![r(-2), r(0), r(1)]),
            lin(r(1), ten(60)),
            lin(two(200), r(1)),
            lin(-(r(1) / two(150)), r(1)),
            lin(r(7), r(1)),
        ];
        let p = factors.iter().fold(Poly::from_int(1), |acc, f| &acc * f);
        let q = z_primitive(&integer_scaled(p.coeffs()));
        assert!(magnitude_pieces(&q, root_bound_exp(&q)).len() > 1);
        assert!(assert_same_as_sturm(&p, "spread") > 0);
        // SymPy 1.14: Poly(f).count_roots() = 8.
        assert_eq!(SturmChain::new(&p).count_real_roots(), 8);
    }

    /// A quartic with 5,000-digit coefficients (SymPy 1.14
    /// `Poly(f).count_roots()` = 2): its roots near 1 sit 16,600 bits below
    /// the root bound; the isolation took 0.3 s per call before the
    /// magnitude pieces.
    #[test]
    fn huge_coefficient_quartic_isolates_quickly() {
        let big = |k: u32, c: i64| Ratio::from_integer(BigInt::from(7).pow(k) + BigInt::from(c));
        // x⁴ − a x³ + b x² − c x + d with a, b, c, d ≈ 7^5900 (5,000 digits).
        let p = Poly::from_coeffs(vec![
            big(5900, 5),
            -big(5900, 3),
            big(5900, 1),
            -big(5901, 0),
            r(1),
        ]);
        let start = std::time::Instant::now();
        let chain = SturmChain::new(&p);
        let n = chain.count_real_roots();
        assert!(start.elapsed().as_secs() < 5, "took {:?}", start.elapsed());
        // SymPy: count_roots() = 2, count_roots(0, None) = 2,
        // count_roots(-1, 10**6) = 1 (a root in (0, 1), one near 7^5901).
        assert_eq!(n, 2);
        assert_eq!(chain.count_roots_in(&r(-1), &r(1_000_000)), 1);
        assert_eq!(chain.count_roots_in(&r(0), &r(1)), 1);
        assert_eq!((chain.sign_at(&r(0)), chain.sign_at(&r(1))), (1, -1));
    }

    /// Roots hit exactly by a bisection point: the filter defers to the
    /// exact evaluation there.
    #[test]
    fn filtered_signs_are_exact_at_zeros() {
        // (x − 1/2)(x + 3)(8x − 5)(x² − 2)·(big coefficients)
        let big = BigInt::from(10).pow(60) + BigInt::from(7);
        let lin = |a: i64, b: i64| Poly::from_coeffs(vec![r(-a), r(b)]);
        let p = &(&(&lin(1, 2) * &lin(-3, 1)) * &lin(5, 8))
            * &Poly::from_coeffs(vec![r(-2), r(0), r(1)]);
        let p = p.scale(&Ratio::from_integer(big));
        let chain = SturmChain::new(&p);
        for x in [q(1, 2), r(-3), q(5, 8), r(0), q(1, 3)] {
            let v = p.eval(&x);
            let expected = if v.is_positive() {
                1
            } else if v.is_negative() {
                -1
            } else {
                0
            };
            assert_eq!(chain.sign_at(&x), expected, "{x}");
        }
        assert_eq!(chain.count_real_roots(), 5);
    }

    /// Before 0.30 the bisection stopped at depth 256 whatever the Cauchy
    /// bound, and `Π (bᵢx − aᵢ)` with 20-digit `aᵢ` and 10-digit `bᵢ` (bound
    /// `~2⁶⁶⁰`) came back as 2 "isolating" intervals holding all 20 roots;
    /// the refinement stopped at 512 halvings, short of width 1/1024.
    #[test]
    fn isolation_separates_roots_under_a_huge_cauchy_bound() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        let mut p = Poly::from_coeffs(vec![r(1)]);
        let mut roots: Vec<Q> = Vec::new();
        for _ in 0..20 {
            let a = BigInt::from(next(10_000_000_000)) * BigInt::from(10_000_000_000u64)
                + BigInt::from(next(10_000_000_000));
            let a = if next(2) == 0 { -a } else { a };
            let b = BigInt::from(next(10_000_000_000) + 1);
            roots.push(Ratio::new(a.clone(), b.clone()));
            p = &p * &Poly::from_coeffs(vec![Ratio::from_integer(-a), Ratio::from_integer(b)]);
        }
        roots.sort();
        roots.dedup();
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), roots.len());
        let ivs = chain.isolate_all_real_roots();
        assert_eq!(ivs.len(), roots.len());
        let width = Ratio::new(BigInt::from(1), BigInt::from(1024));
        for (iv, root) in ivs.iter().zip(&roots) {
            let fine = chain.refine_interval(iv, &width);
            assert!(&fine.upper - &fine.lower <= width, "{fine:?}");
            assert!(
                fine.lower <= *root && *root <= fine.upper,
                "{root} not in {fine:?}"
            );
            assert!(iv.lower <= *root && *root <= iv.upper);
        }
    }

    /// The square-free part (`Poly::square_free_part`, or the input when it
    /// is certified square-free) equals `p / gcd(p, p')` with the gcd
    /// computed by Euclid over ℚ.
    #[test]
    fn square_free_part_matches_euclid() {
        fn euclid_square_free_part(p: &Poly) -> Poly {
            let dp = p.derivative();
            if dp.is_zero() {
                return p.clone();
            }
            p.div(&Poly::gcd_euclid(p, &dp))
        }
        for seed in 1..=10u64 {
            let p = pseudo_random_poly(seed, 6 + seed as usize % 4, true);
            assert_eq!(
                SturmChain::new(&p).p0,
                euclid_square_free_part(&p),
                "seed {seed}"
            );
            let q = pseudo_random_poly(seed + 100, 9, false);
            assert_eq!(
                SturmChain::new(&q).p0,
                euclid_square_free_part(&q),
                "seed {seed}"
            );
        }
        let half = Ratio::new(BigInt::from(1), BigInt::from(2));
        let p = Poly::from_coeffs(vec![half.clone(), r(3), half, r(1)]);
        let p2 = &p * &p;
        assert_eq!(SturmChain::new(&p2).p0, euclid_square_free_part(&p2));
    }

    /// x^2 - 1 = (x-1)(x+1) → 2 real roots
    #[test]
    fn sturm_x2_minus_1() {
        let p = Poly::from_coeffs(vec![r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 2);
        assert!(!chain.has_no_real_roots());
    }

    /// x^2 + 1 → 0 real roots
    #[test]
    fn sturm_x2_plus_1() {
        let p = Poly::from_coeffs(vec![r(1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 0);
        assert!(chain.has_no_real_roots());
    }

    /// x^3 - x = x(x-1)(x+1) → 3 real roots at -1, 0, 1
    #[test]
    fn sturm_x3_minus_x() {
        let p = Poly::from_coeffs(vec![r(0), r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 3);
    }

    /// x → 1 real root at 0
    #[test]
    fn sturm_x() {
        let p = Poly::x();
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 1);
    }

    /// count_roots_in for sub-intervals of x^3 - x
    #[test]
    fn count_roots_in_subintervals() {
        let p = Poly::from_coeffs(vec![r(0), r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_roots_in(&r(-2), &r(2)), 3);
        let neg_half = q(-1, 2);
        assert_eq!(chain.count_roots_in(&r(-2), &neg_half), 1);
        let half = q(1, 2);
        assert_eq!(chain.count_roots_in(&neg_half, &half), 1);
        assert_eq!(chain.count_roots_in(&half, &r(2)), 1);
        assert_eq!(chain.count_roots_in(&r(2), &r(10)), 0);
    }

    /// isolate_roots_in for x^3 - x in (-10, 10)
    #[test]
    fn isolate_roots_x3_minus_x() {
        let p = Poly::from_coeffs(vec![r(0), r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        let intervals = chain.isolate_roots_in(&r(-10), &r(10), 50);
        assert_eq!(intervals.len(), 3, "should isolate 3 roots");
        for iv in &intervals {
            assert_eq!(iv.kind, IntervalKind::LeftOpen);
            assert_eq!(chain.count_roots_in(&iv.lower, &iv.upper), 1);
        }
    }

    /// isolate_roots_in for x^2 + 1 → no roots
    #[test]
    fn isolate_roots_no_real() {
        let p = Poly::from_coeffs(vec![r(1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        let intervals = chain.isolate_roots_in(&r(-100), &r(100), 50);
        assert!(intervals.is_empty());
    }

    /// Constant polynomial → no roots
    #[test]
    fn sturm_constant() {
        let p = Poly::from_int(5);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 0);
        assert!(chain.has_no_real_roots());
        assert_eq!(chain.count_roots_in_closed(&r(-1), &r(1)), 0);
    }

    /// Zero polynomial
    #[test]
    fn sturm_zero() {
        let p = Poly::zero();
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 0);
        assert!(chain.isolate_all_real_roots().is_empty());
    }

    /// x^2 - 2 → 2 real roots (irrational)
    #[test]
    fn sturm_x2_minus_2() {
        let p = Poly::from_coeffs(vec![r(-2), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 2);
    }

    /// Repeated roots: (x-1)^2 = x^2 - 2x + 1
    /// square_free_part removes the multiplicity → 1 distinct root
    #[test]
    fn sturm_repeated_root() {
        let p = Poly::from_coeffs(vec![r(1), r(-2), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_real_roots(), 1);
    }

    /// isolate_all_real_roots on x^3 - x with exact rational roots
    #[test]
    fn isolate_all_x3_minus_x() {
        let p = Poly::from_coeffs(vec![r(0), r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        let iv = chain.isolate_all_real_roots();
        assert_eq!(iv.len(), 3);
        for w in iv.windows(2) {
            assert!(w[0].upper <= w[1].lower);
        }
        for i in &iv {
            if i.is_point() {
                assert!(p.eval(&i.lower).is_zero());
            } else {
                assert_eq!(i.kind, IntervalKind::LeftOpen);
                assert_eq!(chain.count_roots_in(&i.lower, &i.upper), 1);
                assert!(!p.eval(&i.upper).is_zero());
            }
        }
    }

    /// isolate_all_real_roots on x^2 - 2: two irrational roots
    #[test]
    fn isolate_all_x2_minus_2() {
        let p = Poly::from_coeffs(vec![r(-2), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        let iv = chain.isolate_all_real_roots();
        assert_eq!(iv.len(), 2);
        assert!(iv[0].upper <= r(0) && iv[1].lower >= r(0));
        let width = q(1, 100);
        let refined = chain.refine_interval(&iv[1], &width);
        assert_eq!(refined.kind, IntervalKind::LeftOpen);
        assert!(refined.width() <= width);
        assert!(refined.lower < q(1415, 1000));
        assert!(refined.upper > q(1414, 1000));
        let zero = Interval::point(r(0));
        assert_eq!(chain.refine_interval(&zero, &width), zero);
    }

    /// count_roots_in_closed includes the left endpoint
    #[test]
    fn closed_interval_count() {
        let p = Poly::from_coeffs(vec![r(0), r(-1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.count_roots_in_closed(&r(-1), &r(1)), 3);
        assert_eq!(chain.count_roots_in(&r(-1), &r(1)), 2);
        assert_eq!(chain.count_roots_in_closed(&r(0), &r(0)), 1);
        assert_eq!(chain.count_roots_in_closed(&r(2), &r(1)), 0);
    }

    /// leading_sign_of_original
    #[test]
    fn leading_sign_positive() {
        let p = Poly::from_coeffs(vec![r(1), r(0), r(1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.leading_sign_of_original(), 1);
    }

    #[test]
    fn leading_sign_negative() {
        let p = Poly::from_coeffs(vec![r(-1), r(0), r(-1)]);
        let chain = SturmChain::new(&p);
        assert_eq!(chain.leading_sign_of_original(), -1);
    }
}
