//! 0.21 track: the two-tier `Matrix`.
//!
//! A `Matrix` whose entries are all rational literals now runs
//! `char_poly_coeffs` (Berkowitz), `matmul`, `trace` and `lu` on the exact
//! [`QMatrix`] tier and interns the results afterwards, and remembers the
//! rationality decision (with the converted entries) across calls.  Every
//! lowering is *output-preserving*: the arena folds any arithmetic on two
//! rational literals into one rational literal, so the symbolic route
//! could only ever have produced the nodes the exact route interns.  The
//! tests here check that claim structurally (`==` on `Ex` is node
//! identity) against two independent symbolic routes computed in this
//! file through the public API:
//!
//! * a **re-implementation of the pre-0.21 symbolic algorithms** over `Ex`
//!   (`ref_*` below: the Berkowitz loop with `expand`, the `Σ aᵢₖbₖⱼ` loop,
//!   the first-non-zero-pivot LU), run on the same rational matrices; and
//! * the **crate's own symbolic route**, forced by replacing one entry with
//!   a symbol `t` (so `as_qmatrix()` is `None`) and substituting the
//!   rational value back into the result.  (No `#[doc(hidden)]` toggle was
//!   needed for that; the only hidden addition is the cache probe
//!   `Matrix::is_exact_cached`, used by the cache-lifetime tests.)

// The reference LU/Berkowitz re-implementations index rows and columns
// deliberately: the loops are the textbook algorithms.
#![allow(clippy::needless_range_loop)]

use std::time::Instant;

use symplex::decompositions::Lu;
use symplex::linprog::{q, qi};
use symplex::num_bigint::BigInt;
use symplex::num_rational::Ratio;
use symplex::num_traits::{One, Zero};
use symplex::prelude::*;
use symplex::stats::multivariate::pca;

type Q = Ratio<BigInt>;

// ═══════════════════════════════════════════════════════════════════════════
// Deterministic inputs
// ═══════════════════════════════════════════════════════════════════════════

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// An integer in `−5..=5`.
    fn small(&mut self) -> i64 {
        (self.next() % 11) as i64 - 5
    }

    /// A fraction with a small numerator and a denominator in `1..=4`
    /// (about one entry in seven is zero).
    fn ratio(&mut self) -> Q {
        let n = self.small();
        let d = (self.next() % 4) as i64 + 1;
        q(n, d)
    }
}

fn int_matrix(ctx: &Context, n: usize, m: usize, seed: u64) -> Matrix {
    let mut g = Lcg(seed);
    Matrix::from_fn(n, m, |_, _| ctx.int(g.small())).unwrap()
}

fn ratio_matrix(ctx: &Context, n: usize, m: usize, seed: u64) -> Matrix {
    let mut g = Lcg(seed);
    Matrix::from_fn(n, m, |_, _| ctx.from_ratio(g.ratio())).unwrap()
}

/// About 40 % zeros, so that LU has to swap rows.
fn sparse_matrix(ctx: &Context, n: usize, seed: u64) -> Matrix {
    let mut g = Lcg(seed);
    Matrix::from_fn(n, n, |_, _| {
        if g.next() % 5 < 2 {
            ctx.zero()
        } else {
            ctx.from_ratio(g.ratio())
        }
    })
    .unwrap()
}

fn ratio_qmatrix(n: usize, m: usize, seed: u64) -> QMatrix {
    let mut g = Lcg(seed);
    QMatrix::from_fn(n, m, |_, _| g.ratio()).unwrap()
}

fn display_rows(m: &Matrix) -> Vec<String> {
    (0..m.nrows())
        .map(|i| {
            m.row(i)
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// `m` with entry `(0, 0)` replaced by the symbol `t`, and the value it
/// replaced — the crate's symbolic route, forced.
fn with_symbol(ctx: &Context, m: &Matrix) -> (Matrix, Ex, Ex) {
    let t = ctx.symbol("t");
    let a00 = m[(0, 0)].clone();
    let mut sym = m.clone();
    sym.set(0, 0, t.clone());
    assert!(!sym.is_exact_cached());
    (sym, t, a00)
}

// ═══════════════════════════════════════════════════════════════════════════
// Reference implementations: the pre-0.21 symbolic algorithms over `Ex`
// ═══════════════════════════════════════════════════════════════════════════

/// Berkowitz over `Ex` exactly as `Matrix::berkowitz_monic` ran it before
/// the exact tier: coefficients of `det(λI − A)`, highest degree first,
/// `expand()` after every accumulation.
fn ref_berkowitz_monic(m: &Matrix) -> Vec<Ex> {
    let ctx = m.context();
    let n = m.nrows();
    let one = ctx.one();
    let zero = ctx.zero();
    let mut vec: Vec<Ex> = vec![one.clone(), -m.get(n - 1, n - 1)];
    for k in 2..=n {
        let s = n - k;
        let a = m.get(s, s);
        let r: Vec<&Ex> = (s + 1..n).map(|j| m.get(s, j)).collect();
        let mut c: Vec<Ex> = (s + 1..n).map(|i| m.get(i, s).clone()).collect();
        let mut diags: Vec<Ex> = vec![one.clone(), -a];
        for step in 0..(k - 1) {
            if step > 0 {
                let next: Vec<Ex> = (s + 1..n)
                    .map(|i| {
                        let mut acc = zero.clone();
                        for (idx, j) in (s + 1..n).enumerate() {
                            acc += m.get(i, j) * &c[idx];
                        }
                        acc.expand()
                    })
                    .collect();
                c = next;
            }
            let mut rc = zero.clone();
            for (ri, ci) in r.iter().zip(c.iter()) {
                rc += *ri * ci;
            }
            diags.push((-rc).expand());
        }
        let mut next_vec: Vec<Ex> = Vec::with_capacity(k + 1);
        for i in 0..=k {
            let mut acc = zero.clone();
            for (j, v) in vec.iter().enumerate().take(k) {
                if j <= i {
                    acc += &diags[i - j] * v;
                }
            }
            next_vec.push(acc.expand());
        }
        vec = next_vec;
    }
    vec
}

/// `Matrix::char_poly_coeffs` as it was: ascending `det(A − λI)`.
fn ref_char_poly_coeffs(m: &Matrix) -> Vec<Ex> {
    let n = m.nrows();
    let monic = ref_berkowitz_monic(m);
    let flip = n % 2 == 1;
    (0..=n)
        .map(|k| {
            let c = monic[n - k].clone();
            if flip { -c } else { c }
        })
        .collect()
}

/// `Matrix::matmul` as it was: `acc = a₀b₀; acc += aₖbₖ`.
fn ref_matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let p = a.ncols();
    Matrix::from_fn(a.nrows(), b.ncols(), |i, j| {
        let mut acc: Ex = a.get(i, 0) * b.get(0, j);
        for k in 1..p {
            acc += a.get(i, k) * b.get(k, j);
        }
        acc
    })
    .unwrap()
}

/// `Matrix::trace` as it was.
fn ref_trace(m: &Matrix) -> Ex {
    let mut acc = m.get(0, 0).clone();
    for i in 1..m.nrows() {
        acc += m.get(i, i);
    }
    acc
}

/// `Matrix::lu` as it was: first structurally non-zero pivot, `Ex`
/// division and subtraction.  `None` when singular.
fn ref_lu(m: &Matrix) -> Option<Lu<Matrix>> {
    let ctx = m.context();
    let n = m.nrows();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut u: Vec<Vec<Ex>> = (0..n).map(|i| m.row(i).to_vec()).collect();
    let mut l: Vec<Vec<Ex>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { ctx.one() } else { ctx.zero() })
                .collect()
        })
        .collect();
    for k in 0..n {
        let pivot_row = (k..n).find(|&i| !u[i][k].is_zero_structural())?;
        if pivot_row != k {
            u.swap(k, pivot_row);
            perm.swap(k, pivot_row);
            for j in 0..k {
                let tmp = l[k][j].clone();
                l[k][j] = l[pivot_row][j].clone();
                l[pivot_row][j] = tmp;
            }
        }
        for i in (k + 1)..n {
            if u[i][k].is_zero_structural() {
                continue;
            }
            let factor = &u[i][k] / &u[k][k];
            l[i][k] = factor.clone();
            u[i][k] = ctx.zero();
            for j in (k + 1)..n {
                let term = &factor * &u[k][j];
                u[i][j] = &u[i][j] - &term;
            }
        }
    }
    Some(Lu {
        l: Matrix::new(l).unwrap(),
        u: Matrix::new(u).unwrap(),
        perm,
    })
}

/// Berkowitz over `Q` (plain `Ratio` arithmetic), ascending `det(A − λI)`
/// — the reference for `QMatrix::char_poly_coeffs`.
fn ref_q_char_poly(a: &QMatrix) -> Vec<Q> {
    let n = a.nrows();
    let mut vec = vec![Q::one(), -a[(n - 1, n - 1)].clone()];
    for k in 2..=n {
        let s = n - k;
        let mut c: Vec<Q> = (s + 1..n).map(|i| a[(i, s)].clone()).collect();
        let mut diags = vec![Q::one(), -a[(s, s)].clone()];
        for step in 0..k - 1 {
            if step > 0 {
                c = (s + 1..n)
                    .map(|i| (s + 1..n).zip(&c).map(|(j, cj)| &a[(i, j)] * cj).sum())
                    .collect();
            }
            let rc: Q = (s + 1..n).zip(&c).map(|(j, cj)| &a[(s, j)] * cj).sum();
            diags.push(-rc);
        }
        vec = (0..=k)
            .map(|i| {
                vec.iter()
                    .enumerate()
                    .take(i + 1)
                    .map(|(j, v)| &diags[i - j] * v)
                    .sum()
            })
            .collect();
    }
    let flip = n % 2 == 1;
    (0..=n)
        .map(|k| {
            let c = vec[n - k].clone();
            if flip { -c } else { c }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// char_poly_coeffs / char_poly
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn char_poly_coeffs_rational_fast_path_is_the_symbolic_result() {
    let ctx = Context::new();
    for (n, seed) in [
        (1usize, 1u64),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
    ] {
        for m in [
            int_matrix(&ctx, n, n, seed),
            ratio_matrix(&ctx, n, n, seed + 100),
        ] {
            let fast = m.char_poly_coeffs().unwrap();
            assert!(m.is_exact_cached(), "{n}×{n}: decision cached");
            let reference = ref_char_poly_coeffs(&m);
            assert_eq!(fast, reference, "{n}×{n} seed {seed}: structural identity");
            let fast_s: Vec<String> = fast.iter().map(ToString::to_string).collect();
            let ref_s: Vec<String> = reference.iter().map(ToString::to_string).collect();
            assert_eq!(fast_s, ref_s, "{n}×{n} seed {seed}: Display identity");
            assert!(fast.iter().all(|c| c.as_rational().is_some()));
            // c₀ = det A, c_n = (−1)ⁿ.
            assert_eq!(fast[0], m.det().unwrap());
            assert_eq!(fast[n], ctx.int(if n % 2 == 1 { -1 } else { 1 }));
        }
    }
}

#[test]
fn char_poly_rational_matches_the_forced_symbolic_route() {
    let ctx = Context::new();
    let lam = ctx.symbol("lambda");
    for seed in 11..15u64 {
        let m = ratio_matrix(&ctx, 5, 5, seed);
        let (sym, t, a00) = with_symbol(&ctx, &m);
        let fast = m.char_poly(&lam).unwrap();
        let forced = sym.char_poly(&lam).unwrap().subs(&t, &a00);
        assert_eq!(fast, forced, "seed {seed}");
        assert_eq!(fast.to_string(), forced.to_string());
        let fast_c = m.char_poly_coeffs().unwrap();
        let forced_c: Vec<Ex> = sym
            .char_poly_coeffs()
            .unwrap()
            .iter()
            .map(|c| c.subs(&t, &a00))
            .collect();
        assert_eq!(fast_c, forced_c, "seed {seed}");
    }
}

#[test]
fn char_poly_display_is_unchanged() {
    let ctx = Context::new();
    let lam = ctx.symbol("lambda");
    // The `Matrix::char_poly` / `char_poly_coeffs` doc examples.
    assert_eq!(
        matrix![ctx, [2, 1], [1, 2]]
            .char_poly(&lam)
            .unwrap()
            .to_string(),
        "lambda^2 - 4*lambda + 3"
    );
    assert_eq!(
        matrix![ctx, [1, 2], [3, 4]].char_poly_coeffs().unwrap(),
        vec![ctx.int(-2), ctx.int(-5), ctx.int(1)]
    );
    // A 4×4 with fractions: Berkowitz on cleared denominators, scaled back.
    let m = Matrix::from_ratio(
        &ctx,
        &[
            vec![q(1, 2), qi(1), qi(0), q(1, 3)],
            vec![qi(2), qi(-1), q(1, 2), qi(0)],
            vec![qi(1), qi(3), qi(0), q(-1, 2)],
            vec![qi(0), q(1, 4), qi(1), qi(1)],
        ],
    )
    .unwrap();
    let p = m.char_poly(&lam).unwrap();
    assert_eq!(
        p,
        ref_char_poly_coeffs(&m)
            .iter()
            .enumerate()
            .fold(ctx.zero(), |acc, (k, c)| acc + c * lam.powi(k as i64))
            .expand()
    );
    // sympy: expand((A - lambda*eye(4)).det()) = lambda**4 - lambda**3/2 - 4*lambda**2 + 65*lambda/16 - 125/32
    assert_eq!(
        p.to_string(),
        "lambda^4 - 1/2*lambda^3 - 4*lambda^2 + 65/16*lambda - 125/32"
    );
    assert_eq!(m.det().unwrap(), ctx.rational(-125, 32));
}

#[test]
fn qmatrix_char_poly_coeffs_matches_a_plain_ratio_berkowitz() {
    for (n, seed) in [(1usize, 1u64), (2, 2), (3, 3), (5, 5), (8, 8)] {
        let a = ratio_qmatrix(n, n, seed);
        let got = a.char_poly_coeffs().unwrap();
        assert_eq!(got, ref_q_char_poly(&a), "{n}×{n}");
        assert_eq!(got[0], a.det().unwrap());
        // Cayley–Hamilton: Σ cₖ Aᵏ = 0.
        let mut power = QMatrix::identity(n).unwrap();
        let mut acc = QMatrix::zeros(n, n).unwrap();
        for c in &got {
            acc = acc.add(&power.scale(c)).unwrap();
            power = power.matmul(&a).unwrap();
        }
        assert!(acc.is_zero(), "{n}×{n}: Cayley–Hamilton");
    }
    // Wide integer entries: the coefficients outgrow every fixed-width
    // cell, so the escalating kernel ends on `BigInt`.
    let big = BigInt::from(1i64 << 50);
    let a = QMatrix::from_fn(6, 6, |i, j| {
        Ratio::from_integer(BigInt::from((i * 7 + j * 3) as i64 % 11 - 5) * &big)
    })
    .unwrap();
    let got = a.char_poly_coeffs().unwrap();
    assert_eq!(got, ref_q_char_poly(&a));
    assert!(
        got[0].numer().bits() > 256,
        "det has {} bits",
        got[0].numer().bits()
    );
    // Errors: not square.
    assert!(
        QMatrix::from_i64(&[&[1, 2, 3]])
            .unwrap()
            .char_poly_coeffs()
            .is_err()
    );
    assert!(
        Matrix::from_i64(&Context::new(), &[&[1, 2, 3]])
            .unwrap()
            .char_poly_coeffs()
            .is_err()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// matmul / trace
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matmul_rational_fast_path_is_the_symbolic_result() {
    let ctx = Context::new();
    for (n, p, m, seed) in [
        (1usize, 1usize, 1usize, 1u64),
        (2, 3, 2, 2),
        (4, 4, 4, 3),
        (6, 7, 5, 4),
        (9, 9, 9, 5),
    ] {
        let a = ratio_matrix(&ctx, n, p, seed);
        let b = int_matrix(&ctx, p, m, seed + 50);
        let fast = a.matmul(&b).unwrap();
        assert!(a.is_exact_cached() && b.is_exact_cached());
        let reference = ref_matmul(&a, &b);
        assert_eq!(fast, reference, "{n}×{p}·{p}×{m}");
        assert_eq!(display_rows(&fast), display_rows(&reference));
        assert_eq!(fast.to_string(), reference.to_string());
        assert_eq!(&a * &b, reference);
        // Forced symbolic route.
        let (sym, t, a00) = with_symbol(&ctx, &a);
        let forced = sym.matmul(&b).unwrap().map(|e| e.subs(&t, &a00));
        assert_eq!(fast, forced);
    }
    // Shape errors are unchanged.
    let a = int_matrix(&ctx, 2, 3, 1);
    assert!(a.matmul(&a).is_err());
    // Mixed rational × symbolic still goes the symbolic way.
    let x = ctx.symbol("x");
    let s = Matrix::new(vec![
        vec![x.clone(), ctx.int(1)],
        vec![ctx.int(0), x.clone()],
    ])
    .unwrap();
    let r = matrix![ctx, [1, 2], [3, 4]];
    assert_eq!(r.matmul(&s).unwrap(), ref_matmul(&r, &s));
    assert_eq!(s.matmul(&r).unwrap(), ref_matmul(&s, &r));
}

#[test]
fn matmul_doc_examples_are_unchanged() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2], [3, 4]];
    let b = matrix![ctx, [0, 1], [1, 0]];
    assert_eq!(&a * &b, matrix![ctx, [2, 1], [4, 3]]);
    assert_eq!((&a * &b).to_string(), "[\n  [2, 1],\n  [4, 3]\n]");
    let v = matrix![ctx, [1], [1]];
    assert_eq!(&a * &v, matrix![ctx, [3], [7]]);
    let half = Matrix::from_ratio(&ctx, &[vec![q(1, 2), qi(0)], vec![qi(0), q(1, 2)]]).unwrap();
    assert_eq!((&a * &half).to_string(), "[\n  [1/2, 1],\n  [3/2, 2]\n]");
}

#[test]
fn trace_rational_fast_path_is_the_symbolic_result() {
    let ctx = Context::new();
    for (n, seed) in [(1usize, 1u64), (3, 2), (8, 3), (20, 4)] {
        let m = ratio_matrix(&ctx, n, n, seed);
        let fast = m.trace().unwrap();
        assert_eq!(fast, ref_trace(&m));
        assert_eq!(fast.to_string(), ref_trace(&m).to_string());
        let (sym, t, a00) = with_symbol(&ctx, &m);
        assert_eq!(sym.trace().unwrap().subs(&t, &a00), fast);
    }
    // Only the diagonal needs to be rational.
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![
        vec![ctx.rational(1, 2), x.clone()],
        vec![x.sin(), ctx.rational(1, 3)],
    ])
    .unwrap();
    assert_eq!(m.trace().unwrap(), ctx.rational(5, 6));
    assert_eq!(m.trace().unwrap(), ref_trace(&m));
    // A symbolic diagonal is summed symbolically.
    let s = Matrix::new(vec![vec![x.clone(), ctx.int(1)], vec![ctx.int(0), &x + 1]]).unwrap();
    assert_eq!(s.trace().unwrap(), &x * 2 + 1);
    assert!(matrix![ctx, [1, 2, 3]].trace().is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// lu
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lu_rational_fast_path_is_the_symbolic_result() {
    let ctx = Context::new();
    let mut with_pivoting = 0;
    for (n, seed) in [
        (1usize, 1u64),
        (2, 2),
        (3, 3),
        (4, 4),
        (6, 5),
        (6, 6),
        (8, 7),
        (10, 8),
    ] {
        for m in [
            int_matrix(&ctx, n, n, seed),
            ratio_matrix(&ctx, n, n, seed + 200),
            sparse_matrix(&ctx, n, seed + 300),
            sparse_matrix(&ctx, n, seed + 400),
        ] {
            let Some(reference) = ref_lu(&m) else {
                assert!(
                    m.lu().is_err(),
                    "{n}×{n} seed {seed}: singular in both routes"
                );
                continue;
            };
            let fast = m.lu().unwrap();
            assert_eq!(
                fast.perm, reference.perm,
                "{n}×{n} seed {seed}: same pivots"
            );
            assert_eq!(fast.l, reference.l, "{n}×{n} seed {seed}: L");
            assert_eq!(fast.u, reference.u, "{n}×{n} seed {seed}: U");
            assert_eq!(display_rows(&fast.l), display_rows(&reference.l));
            assert_eq!(display_rows(&fast.u), display_rows(&reference.u));
            if fast.perm.iter().enumerate().any(|(i, &p)| i != p) {
                with_pivoting += 1;
            }
            // P·A = L·U exactly.
            let pa = Matrix::new(fast.perm.iter().map(|&i| m.row(i).to_vec()).collect()).unwrap();
            assert_eq!(&fast.l * &fast.u, pa);
        }
    }
    assert!(
        with_pivoting >= 4,
        "the cases exercise row swaps ({with_pivoting})"
    );
    // The pivot rule in a case where it matters: a zero leading entry.
    let m = matrix![ctx, [0, 1, 2], [3, 4, 5], [6, 7, 9]];
    let Lu { l, u, perm } = m.lu().unwrap();
    assert_eq!(perm, vec![1, 0, 2]);
    assert_eq!(l, matrix![ctx, [1, 0, 0], [0, 1, 0], [2, -1, 1]]);
    assert_eq!(u, matrix![ctx, [3, 4, 5], [0, 1, 2], [0, 0, 1]]);
    // Doc example and singular input.
    let a = matrix![ctx, [4, 3], [6, 3]];
    let Lu { l, u, perm } = a.lu().unwrap();
    assert_eq!(perm, vec![0, 1]);
    assert_eq!(l.to_string(), "[\n  [  1, 0],\n  [3/2, 1]\n]");
    assert_eq!(u.to_string(), "[\n  [4,    3],\n  [0, -3/2]\n]");
    assert!(matrix![ctx, [1, 2], [2, 4]].lu().is_err());
    assert!(matrix![ctx, [1, 2, 3]].lu().is_err());
}

#[test]
fn lu_rational_matches_the_forced_symbolic_route_without_pivoting() {
    // With a symbol at (0, 0) the symbolic route sees every entry that
    // mixes with it as structurally non-zero, so pivots can only be
    // compared when no row swap happens: leading principal minors ≠ 0.
    let ctx = Context::new();
    let m = Matrix::from_ratio(
        &ctx,
        &[
            vec![qi(2), q(1, 2), qi(1), qi(0)],
            vec![qi(1), qi(3), q(-1, 3), qi(2)],
            vec![qi(0), qi(1), qi(4), q(1, 2)],
            vec![q(1, 2), qi(0), qi(1), qi(5)],
        ],
    )
    .unwrap();
    let fast = m.lu().unwrap();
    assert_eq!(fast.perm, vec![0, 1, 2, 3]);
    let (sym, t, a00) = with_symbol(&ctx, &m);
    let forced = sym.lu().unwrap();
    assert_eq!(forced.perm, fast.perm);
    assert_eq!(forced.l.map(|e| e.subs(&t, &a00).simplify()), fast.l);
    assert_eq!(forced.u.map(|e| e.subs(&t, &a00).simplify()), fast.u);
}

#[test]
fn qmatrix_lu_factors_exactly() {
    for (n, seed) in [(1usize, 1u64), (3, 2), (5, 3), (7, 4), (12, 5)] {
        let a = ratio_qmatrix(n, n, seed);
        match a.lu() {
            Ok(Lu { l, u, perm }) => {
                let pa = QMatrix::new(perm.iter().map(|&i| a.row(i).to_vec()).collect()).unwrap();
                assert_eq!(l.matmul(&u).unwrap(), pa, "{n}×{n}: P·A = L·U");
                for i in 0..n {
                    assert_eq!(l[(i, i)], Q::one());
                    for j in 0..i {
                        assert!(u[(i, j)].is_zero(), "U upper triangular");
                    }
                    for j in i + 1..n {
                        assert!(l[(i, j)].is_zero(), "L lower triangular");
                    }
                }
                let mut sorted = perm.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            }
            Err(_) => assert!(a.det().unwrap().is_zero(), "{n}×{n}: only singular fails"),
        }
    }
    assert!(
        QMatrix::from_i64(&[&[1, 2], &[2, 4]])
            .unwrap()
            .lu()
            .is_err()
    );
    assert!(QMatrix::from_i64(&[&[1, 2, 3]]).unwrap().lu().is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// The eigen-family and pca, which route through char_poly_coeffs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eigen_family_values_are_unchanged() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1], [1, 2]];
    let mut ev = a.eigenvals().unwrap();
    ev.sort_by_key(ToString::to_string);
    assert_eq!(ev, vec![ctx.int(1), ctx.int(3)]);
    let m = matrix![ctx, [4, 1, 2], [1, 3, 1], [2, 1, 5]];
    let ev = m.eigenvals_with_multiplicity().unwrap();
    assert_eq!(ev.len(), 3);
    assert!(
        ev.iter()
            .all(|(v, m)| *m == 1 && v.to_string().starts_with("RootOf"))
    );
    let sum: f64 = ev.iter().map(|(v, _)| v.eval_f64().unwrap()).sum();
    assert!((sum - 12.0).abs() < 1e-9);
    // A defective matrix's Jordan form.
    let j = matrix![ctx, [2, 1], [0, 2]].jordan_form().unwrap();
    assert_eq!(j.j, matrix![ctx, [2, 1], [0, 2]]);
}

#[test]
fn pca_doc_values_are_unchanged() {
    let ctx = Context::new();
    // `pca` doc example and tests/v14/v14_multivariate_order.rs.
    let p = pca(&ctx, &QMatrix::from_i64(&[&[2, 1], &[1, 2]]).unwrap()).unwrap();
    assert_eq!(p.eigenvalues, vec![ctx.int(3), ctx.int(1)]);
    assert_eq!(
        p.explained_variance_ratio,
        vec![ctx.rational(3, 4), ctx.rational(1, 4)]
    );
    assert_eq!(
        p.components[0][0].equals(&(ctx.one() / ctx.int(2).sqrt())),
        Some(true)
    );
    let r2 = ctx.int(2).sqrt() / 2;
    assert_eq!(p.components[1][1].equals(&(-&r2)), Some(true));
    // Surd eigenvalues (5 ∓ √5)/2.
    let p = pca(&ctx, &QMatrix::from_i64(&[&[2, 1], &[1, 3]]).unwrap()).unwrap();
    let r5 = ctx.int(5).sqrt();
    assert_eq!(
        p.eigenvalues[0].equals(&((ctx.int(5) + &r5) / 2)),
        Some(true)
    );
    assert!((p.eigenvalues[0].eval_f64().unwrap() - 3.618033988749895).abs() < 1e-12);
    assert!((p.eigenvalues[1].eval_f64().unwrap() - 1.381966011250105).abs() < 1e-12);
    // Casus irreducibilis 3×3: ordered RootOf eigenvalues.
    let p = pca(
        &ctx,
        &QMatrix::from_i64(&[&[4, 2, 1], &[2, 3, 1], &[1, 1, 2]]).unwrap(),
    )
    .unwrap();
    let want = [6.048917339522303, 1.6431041321077904, 1.3079785283699037];
    for (v, w) in p.eigenvalues.iter().zip(want) {
        assert!((v.eval_f64().unwrap() - w).abs() < 1e-9, "{v} ≠ {w}");
    }
    let ratios = p.explained_variance_ratio_f64().unwrap();
    assert!((ratios[0] - 0.6721019266135893).abs() < 1e-12);
    // Errors are unchanged.
    assert!(pca(&ctx, &QMatrix::from_i64(&[&[1, 2], &[2, 1]]).unwrap()).is_err());
    assert!(pca(&ctx, &QMatrix::from_i64(&[&[0, 0], &[0, 0]]).unwrap()).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// The cache
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rationality_cache_is_decided_once_and_survives_clone() {
    let ctx = Context::new();
    let m = int_matrix(&ctx, 5, 5, 42);
    assert!(!m.is_exact_cached(), "fresh matrix: undecided");
    let d = m.det().unwrap();
    assert!(m.is_exact_cached(), "det decided it");
    // Clone carries the decision along (a shared pointer, not a re-scan).
    let c = m.clone();
    assert!(c.is_exact_cached());
    assert_eq!(c, m);
    assert_eq!(c.det().unwrap(), d);
    // The negative decision is cached too (`char_poly_coeffs` always
    // consults it; the 2×2 `det` formula does not).
    let x = ctx.symbol("x");
    let s = Matrix::new(vec![
        vec![x.clone(), ctx.int(1)],
        vec![ctx.int(0), ctx.int(2)],
    ])
    .unwrap();
    assert!(!s.is_exact_cached());
    assert_eq!(s.det().unwrap(), &x * 2);
    assert!(!s.is_exact_cached(), "the 2×2 formula never asks");
    assert_eq!(s.char_poly_coeffs().unwrap()[0], &x * 2);
    assert!(s.is_exact_cached());
    assert!(s.clone().is_exact_cached());
    // Derived matrices start undecided.
    assert!(!m.transpose().is_exact_cached());
    assert!(!m.map(|e| e.clone()).is_exact_cached());
    assert!(!(&m + &m).is_exact_cached());
}

#[test]
fn rationality_cache_is_dropped_by_mutation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut m = matrix![ctx, [1, 2], [3, 4]];
    assert_eq!(m.det().unwrap(), ctx.int(-2));
    assert_eq!(m.rank(), 2);
    assert!(m.is_exact_cached());
    // set: rational → symbolic.
    m.set(0, 0, x.clone());
    assert!(!m.is_exact_cached(), "set drops the decision");
    assert_eq!(m.det().unwrap(), &x * 4 - 6);
    assert_eq!(m.trace().unwrap(), &x + 4);
    assert_eq!(m.rank(), 2);
    assert!(
        m.is_exact_cached(),
        "…and the new (negative) decision is cached"
    );
    // IndexMut: symbolic → rational.
    m[(0, 0)] = ctx.int(5);
    assert!(!m.is_exact_cached());
    assert_eq!(m.det().unwrap(), ctx.int(14));
    assert_eq!(
        m.char_poly_coeffs().unwrap(),
        vec![ctx.int(14), ctx.int(-9), ctx.int(1)]
    );
    assert!(m.is_exact_cached());
    // get_mut on a clone leaves the original's cache alone.
    let mut c = m.clone();
    *c.get_mut(1, 1) = ctx.int(0);
    assert!(!c.is_exact_cached());
    assert!(m.is_exact_cached());
    assert_eq!(m.det().unwrap(), ctx.int(14));
    assert_eq!(c.det().unwrap(), ctx.int(-6));
    assert_ne!(c, m);
}

#[test]
fn equality_and_debug_ignore_the_cache() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2], [3, 4]];
    let b = matrix![ctx, [1, 2], [3, 4]];
    a.char_poly_coeffs().unwrap();
    assert!(a.is_exact_cached() && !b.is_exact_cached());
    assert_eq!(a, b);
    assert_eq!(b, a);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(format!("{a:?}"), "Matrix(2×2, [[1, 2], [3, 4]])");
    assert_eq!(a.to_string(), b.to_string());
    assert_ne!(a, matrix![ctx, [1, 2], [3, 5]]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Timing (loose: debug builds, shared machines)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rational_paths_are_fast_enough() {
    let ctx = Context::new();
    let budget = std::time::Duration::from_secs(2);

    let cov = QMatrix::from_fn(6, 6, |i, j| qi(if i == j { 3 } else { 1 })).unwrap();
    let t = Instant::now();
    let p = pca(&ctx, &cov).unwrap();
    assert!(t.elapsed() < budget, "pca 6×6 took {:?}", t.elapsed());
    assert_eq!(p.eigenvalues[0], ctx.int(8));
    assert!(p.eigenvalues[1..].iter().all(|v| *v == ctx.int(2)));

    let a8 = int_matrix(&ctx, 8, 8, 1);
    let lam = ctx.symbol("lambda");
    let t = Instant::now();
    let cp = a8.char_poly(&lam).unwrap();
    assert!(t.elapsed() < budget, "char_poly 8×8 took {:?}", t.elapsed());
    assert_eq!(
        cp,
        ref_char_poly_coeffs(&a8)
            .iter()
            .enumerate()
            .fold(ctx.zero(), |acc, (k, c)| acc + c * lam.powi(k as i64))
            .expand()
    );

    let a = int_matrix(&ctx, 20, 20, 2);
    let b = int_matrix(&ctx, 20, 20, 3);
    let t = Instant::now();
    let c = a.matmul(&b).unwrap();
    assert!(t.elapsed() < budget, "matmul 20×20 took {:?}", t.elapsed());
    assert_eq!(c, ref_matmul(&a, &b));

    let a15 = int_matrix(&ctx, 15, 15, 4);
    let t = Instant::now();
    let lu = a15.lu().unwrap();
    assert!(t.elapsed() < budget, "lu 15×15 took {:?}", t.elapsed());
    let reference = ref_lu(&a15).unwrap();
    assert_eq!(lu.perm, reference.perm);
    assert_eq!(lu.l, reference.l);
    assert_eq!(lu.u, reference.u);
}
