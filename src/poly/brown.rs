//! Brown's dense modular gcd in `ℤ[x₁, …, xₙ]`.
//!
//! The heuristic gcd ([`modgcd::heu_gcd`]) evaluates one variable after
//! another at a large integer, so its integers grow with the product of the
//! partial degrees: for two dense inputs in four variables with 21-digit
//! coefficients it worked on million-digit images and took 20 s, 90% of it
//! in integer gcds and divisions.  Brown's algorithm works modulo word
//! primes instead, where every number has 62 bits:
//!
//! 1. reduce the primitive inputs modulo a prime `p`;
//! 2. in `𝔽ₚ[x₀, …, x_{k−1}]`, split off the content in the last variable
//!    `y`, evaluate `y` at points `a ∈ 𝔽ₚ`, and recurse on the images in one
//!    variable fewer (Euclid over `𝔽ₚ` at the bottom); interpolate the gcd in
//!    `y` from enough images, each scaled by the gcd of the leading
//!    coefficients so that all images have the same normalisation;
//! 3. combine the images for several primes by the Chinese remainder
//!    theorem until the result stops changing, then verify the candidate by
//!    trial division over `ℤ`.
//!
//! A returned gcd is therefore always right; what can go wrong is
//! recognised by degrees.  An image whose degree in some variable exceeds a
//! known bound came from an unlucky prime or point and is discarded; one
//! whose degree is *below* the bound shows that the bound was not sharp:
//! the bound is lowered and the prime started over (`Restart`).  Bounds
//! only decrease, so there are finitely many restarts.  The gcd in
//! `ℤ[x₁, …, xₙ]` is unique up to sign once its integer content is fixed, so
//! the result is the polynomial the heuristic finds — only faster.
//!
//! The cost is about one univariate gcd over `𝔽ₚ` per point of the grid
//! `∏_{v ≥ 1} (deg_v h + 1)` per prime: polynomial in the degrees, where the
//! heuristic is exponential in the number of variables through its integer
//! sizes.  [`brown_gcd`] declines (`None`) when that grid exceeds
//! [`MAX_IMAGES`], and the caller falls back to the heuristic.
//!
//! # Representation
//!
//! A polynomial over `𝔽ₚ` is a `Vec<(Exps, u64)>` sorted ascending in the
//! lexicographic order of the exponent vectors, without zero coefficients.
//! In that order the terms that agree in all but the last variable are
//! contiguous ("runs"), so evaluating, taking the content in, or
//! interpolating the last variable are linear scans.  Univariate
//! polynomials are [`PolyIn<Fp64>`].
//!
//! # Provenance
//!
//! The structure follows SymPy's `modgcd_multivariate` and
//! `_modgcd_multivariate_p` (`sympy/polys/modulargcd.py`, BSD-3-Clause),
//! after W. S. Brown, "On Euclid's algorithm and the computation of
//! polynomial greatest common divisors", *J. ACM* 18 (1971), and M.
//! Monagan's course notes on the non-monic gcd problem.  Evaluation points
//! come from a fixed-seed generator, so the work done is deterministic.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::ops::Range;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use super::modgcd::{self, ZMap};
use super::modpoly::{Fp64, PolyIn, RingOps};
use super::multipoly::Exps;
use crate::base::rng::SplitMix64;

/// Largest evaluation grid `∏_{v ≥ 1} (min(deg_v f, deg_v g) + 1)` per
/// prime that [`brown_gcd`] attempts (about one univariate gcd per point).
pub(crate) const MAX_IMAGES: u64 = 1_000_000;

/// Upper bound on the univariate gcds over `𝔽ₚ` one call of [`brown_gcd`]
/// may compute in total, over all primes, restarts and discarded images.
const MAX_WORK: u64 = 20_000_000;

/// A polynomial over `𝔽ₚ`: terms ascending in lex order, no zeros.
type Terms = Vec<(Exps, u64)>;

/// A univariate polynomial over `𝔽ₚ`.
type Up = PolyIn<Fp64>;

/// The degree bounds of the gcd, indexed by variable: `deg[v]` bounds the
/// degree of the gcd in `x_v`, `cont[v]` the degree of its content in
/// `𝔽ₚ[x_v][x₀, …, x_{v−1}]` (unused for `v = 0`).
struct Bounds {
    deg: Vec<u32>,
    cont: Vec<u32>,
}

/// A degree bound was lowered: the images collected so far may be
/// unlucky, start over with the next prime.
struct Restart;

/// The state shared by the recursion for one call of [`brown_gcd`].
struct Ctx {
    field: Fp64,
    bounds: Bounds,
    rng: SplitMix64,
    /// Univariate gcds computed so far (bounded by [`MAX_WORK`]).
    work: u64,
}

/// The gcd of the non-zero integer polynomials `f` and `g` in `nv`
/// variables (lex-ordered [`ZMap`]s) by Brown's modular algorithm, with
/// integer content `gcd(content f, content g)` and positive leading
/// coefficient; `None` when the evaluation grid exceeds [`MAX_IMAGES`], a
/// degree exceeds [`MAX_DENSE_LIST`](crate::base::cast::MAX_DENSE_LIST)
/// (the univariate images are dense), or the work bound is reached (the
/// caller then uses another method).
pub(crate) fn brown_gcd(nv: usize, f: &ZMap, g: &ZMap) -> Option<ZMap> {
    let cf = modgcd::content(f.values());
    let cg = modgcd::content(g.values());
    if cf.is_zero() || cg.is_zero() {
        return None;
    }
    let ch = modgcd::int_gcd(&cf, &cg);
    // The variables that occur, the one with the largest degree bound
    // first: it is the variable of the univariate gcds at the bottom, the
    // others span the evaluation grid.
    let deg_in = |p: &ZMap, v: usize| p.keys().map(|e| e[v]).max().unwrap_or(0);
    let mut present: Vec<usize> = (0..nv)
        .filter(|&v| deg_in(f, v) > 0 || deg_in(g, v) > 0)
        .collect();
    // Every dense image below (`image_degree_bounds`, `run_poly`,
    // `lc_main`) is sized by a degree of `f` or `g`: bounded here, once.
    let dense = |v: &usize| {
        let top = deg_in(f, *v).max(deg_in(g, *v));
        crate::base::cast::bounded_len(top, crate::base::cast::MAX_DENSE_LIST).is_some()
    };
    if !present.iter().all(dense) {
        return None;
    }
    if present.is_empty() {
        return Some(ZMap::from([(Exps::from_elem(0, nv), ch)]));
    }
    let mut rng = SplitMix64::new(0x9e37_79b9_7f4a_7c15);
    let mut bounds: Vec<u32> = present
        .iter()
        .map(|&v| deg_in(f, v).min(deg_in(g, v)))
        .collect();
    image_degree_bounds(f, g, &present, &mut bounds, &mut rng);
    if let Some(main) = (0..present.len()).max_by_key(|&i| (bounds[i], usize::MAX - i)) {
        let v = present.remove(main);
        present.insert(0, v);
        let b = bounds.remove(main);
        bounds.insert(0, b);
    }
    let perm = present;
    let k = perm.len();
    let permute = |p: &ZMap, c: &BigInt| -> ZMap {
        p.iter()
            .map(|(e, x)| (perm.iter().map(|&v| e[v]).collect::<Exps>(), x / c))
            .collect()
    };
    let (f, g) = (permute(f, &cf), permute(g, &cg));

    let deg = bounds;
    let grid = deg[1..]
        .iter()
        .fold(1u64, |acc, &d| acc.saturating_mul(u64::from(d) + 1));
    if grid > MAX_IMAGES {
        return None;
    }
    let lc = |p: &ZMap| p.last_key_value().map(|(_, c)| c.clone());
    let gamma = modgcd::int_gcd(&lc(&f)?, &lc(&g)?);
    // A prime dividing the gcd of the leading coefficients with respect to
    // some variable could lower a degree: skip those.
    let bad: Vec<BigInt> = (0..k)
        .map(|i| modgcd::int_gcd(&lc_main_int(&f, i), &lc_main_int(&g, i)))
        .collect();
    // Enough primes for the coefficients of `γ/lc(h)·h`, generously: the
    // cap only ends a hopeless search, the verification decides.
    let max_bits = |p: &ZMap| p.values().map(BigInt::bits).max().unwrap_or(0);
    let terms_bits = |p: &ZMap| u64::from(usize::BITS - p.len().leading_zeros());
    let h_bits = gamma.bits()
        + max_bits(&f).min(max_bits(&g))
        + deg.iter().map(|&d| u64::from(d)).sum::<u64>()
        + terms_bits(&f).min(terms_bits(&g))
        + 2;
    let cap = 2 * usize::try_from(h_bits / 61).ok()? + 66;

    let mut ctx = Ctx {
        field: Fp64::new(modgcd::prime(0)),
        bounds: Bounds {
            cont: deg.clone(),
            deg,
        },
        rng,
        work: 0,
    };
    // CRT state: the symmetric residues modulo `m` of `γ/lc(h)·h`.
    let mut hm: ZMap = ZMap::new();
    let mut m = BigInt::one();
    let (mut tried, mut i) = (0usize, 0usize);
    while tried < cap {
        let p = modgcd::prime(i);
        i += 1;
        if modgcd::residue(&gamma, p) == 0 || bad.iter().any(|b| modgcd::residue(b, p) == 0) {
            continue;
        }
        tried += 1;
        ctx.field = Fp64::new(p);
        let (fp, gp) = (reduce(&f, p), reduce(&g, p));
        let hp = match gcd_mod_p(&mut ctx, &fp, &gp, k) {
            Err(Restart) => {
                hm.clear();
                m = BigInt::one();
                if ctx.work > MAX_WORK {
                    return None;
                }
                continue;
            }
            Ok(None) => {
                if ctx.work > MAX_WORK {
                    return None;
                }
                continue;
            }
            Ok(Some(hp)) => hp,
        };
        let gp_res = modgcd::residue(&gamma, p);
        let hp: Terms = hp
            .into_iter()
            .map(|(e, c)| (e, ctx.field.mul(&c, &gp_res)))
            .collect();
        if m.is_one() {
            hm = hp.into_iter().map(|(e, c)| (e, symmetric(c, p))).collect();
            m = BigInt::from(p);
            continue;
        }
        let stable = crt_combine(&mut hm, &m, &hp, ctx.field);
        m *= p;
        if stable && let Some(h) = verified(&f, &g, &hm) {
            // Back to the caller's variables, with the common content and
            // positive leading coefficient in the caller's lex order.
            let mut out: ZMap = h
                .into_iter()
                .map(|(e, x)| {
                    let mut full = Exps::from_elem(0, nv);
                    for (j, &v) in perm.iter().enumerate() {
                        full[v] = e[j];
                    }
                    (full, x * &ch)
                })
                .collect();
            if out.last_key_value().is_some_and(|(_, x)| x.is_negative()) {
                out.values_mut().for_each(|x| *x = -std::mem::take(x));
            }
            return Some(out);
        }
    }
    None
}

/// Lower the degree bounds `bounds[j]` of the gcd in the variables
/// `vars[j]` to the degree of a univariate image: the other variables at
/// random points modulo a prime.  The gcd of the images is a multiple of
/// the image of the gcd, whose degree is kept when the leading
/// coefficients of `f` and `g` in that variable do not vanish at the point
/// (checked: their images keep their degrees), so the result is still an
/// upper bound — and usually exact, which spares the recursion one
/// restart per variable whose bound it would otherwise discover.
fn image_degree_bounds(
    f: &ZMap,
    g: &ZMap,
    vars: &[usize],
    bounds: &mut [u32],
    rng: &mut SplitMix64,
) {
    let p = modgcd::prime(0);
    let field = Fp64::new(p);
    let (fp, gp) = (reduce(f, p), reduce(g, p));
    let nv = f.keys().next().map_or(0, Exps::len);
    // The degrees over ℤ: an image of the same degree certifies that the
    // leading coefficient survived both the prime and the point.
    let tops = |t: &ZMap| -> Vec<u32> {
        (0..nv)
            .map(|w| t.keys().map(|e| e[w]).max().unwrap_or(0))
            .collect()
    };
    let (top_f, top_g) = (tops(f), tops(g));
    let top: Vec<u32> = top_f.iter().zip(&top_g).map(|(a, b)| *a.max(b)).collect();
    for (j, &v) in vars.iter().enumerate() {
        for _ in 0..3 {
            // Powers of a random point for every other variable.
            let pows: Vec<Vec<u64>> = (0..nv)
                .map(|w| {
                    let a = rng.next_u64() % p;
                    let mut row = Vec::with_capacity(top[w] as usize + 1);
                    let mut acc = 1u64;
                    for _ in 0..=top[w] {
                        row.push(acc);
                        acc = field.mul(&acc, &a);
                    }
                    row
                })
                .collect();
            let image = |t: &[(Exps, u64)], deg: u32| -> Up {
                let mut c = vec![0u64; deg as usize + 1];
                for (e, x) in t {
                    let mut y = *x;
                    for w in (0..nv).filter(|&w| w != v) {
                        y = field.mul(&y, &pows[w][e[w] as usize]);
                    }
                    let slot = &mut c[e[v] as usize];
                    *slot = field.add(slot, &y);
                }
                PolyIn::from_coeffs(field, c)
            };
            let (uf, ug) = (image(&fp, top_f[v]), image(&gp, top_g[v]));
            if uf.degree() != Some(top_f[v] as usize) || ug.degree() != Some(top_g[v] as usize) {
                continue;
            }
            let d = uf.gcd(&ug).degree().unwrap_or(0) as u32;
            bounds[j] = bounds[j].min(d);
            break;
        }
    }
}

/// `x ≡ hm (mod m)`, `x ≡ hp (mod p)` in place, symmetric modulo `m·p`;
/// `true` if no coefficient changed (the images agree).
fn crt_combine(hm: &mut ZMap, m: &BigInt, hp: &[(Exps, u64)], field: Fp64) -> bool {
    let p = field.modulus();
    let m_inv = field.inv(&modgcd::residue(m, p)).unwrap_or(0);
    let mp = m * BigInt::from(p);
    let half = &mp >> 1u32;
    let mut stable = true;
    let mut update = |h: &mut BigInt, r: u64| {
        let delta = field.mul(&field.sub(&r, &modgcd::residue(h, p)), &m_inv);
        if delta != 0 {
            stable = false;
            *h += m * BigInt::from(delta);
            if *h > half {
                *h -= &mp;
            }
        }
    };
    let (mut seen, old_len) = (0usize, hm.len());
    for (e, r) in hp {
        match hm.get_mut(e) {
            Some(h) => {
                seen += 1;
                update(h, *r);
            }
            None => {
                let mut h = BigInt::zero();
                update(&mut h, *r);
                hm.insert(e.clone(), h);
            }
        }
    }
    // Monomials of `hm` absent from the image: residue 0 there.
    if seen < old_len {
        let image: std::collections::BTreeSet<&Exps> = hp.iter().map(|(e, _)| e).collect();
        for (e, h) in hm.iter_mut() {
            if !image.contains(e) {
                update(h, 0);
            }
        }
    }
    hm.retain(|_, h| !h.is_zero());
    stable
}

/// The primitive part of `hm` with positive leading coefficient, if it
/// divides both `f` and `g` exactly.
fn verified(f: &ZMap, g: &ZMap, hm: &ZMap) -> Option<ZMap> {
    let c = modgcd::content(hm.values());
    if c.is_zero() {
        return None;
    }
    let negative = hm.last_key_value().is_some_and(|(_, x)| x.is_negative());
    let c = if negative { -c } else { c };
    let h: ZMap = hm.iter().map(|(e, x)| (e.clone(), x / &c)).collect();
    modgcd::zmap_div_exact(f, &h)?;
    modgcd::zmap_div_exact(g, &h)?;
    Some(h)
}

/// `r` in `(−p/2, p/2]`.
fn symmetric(r: u64, p: u64) -> BigInt {
    if r > p / 2 {
        BigInt::from(r) - BigInt::from(p)
    } else {
        BigInt::from(r)
    }
}

/// `p` modulo the prime `q`, zero coefficients dropped (order kept).
fn reduce(p: &ZMap, q: u64) -> Terms {
    p.iter()
        .filter_map(|(e, c)| {
            let r = modgcd::residue(c, q);
            (r != 0).then(|| (e.clone(), r))
        })
        .collect()
}

/// The order with `x_i` first and the other variables after it in their
/// order: `(e_i, e_0, …, e_{i−1}, e_{i+1}, …)` lexicographically, over the
/// exponents `e[..n]`.
fn cmp_main(a: &[u32], b: &[u32], i: usize, n: usize) -> Ordering {
    a[i].cmp(&b[i])
        .then_with(|| a[..i].cmp(&b[..i]))
        .then_with(|| a[i + 1..n].cmp(&b[i + 1..n]))
}

/// The integer leading coefficient of `p` when `x_i` is the first variable.
fn lc_main_int(p: &ZMap, i: usize) -> BigInt {
    let mut best: Option<(&Exps, &BigInt)> = None;
    for (e, c) in p {
        let n = e.len();
        if best.is_none_or(|(b, _)| cmp_main(e, b, i, n) == Ordering::Greater) {
            best = Some((e, c));
        }
    }
    best.map(|(_, c)| c.clone()).unwrap_or_default()
}

// ── 𝔽ₚ[x₀, …, x_{k−1}] in lex order, the last variable `y` special ─────────

/// The exponents of all but the last variable.
fn prefix(e: &[u32]) -> &[u32] {
    &e[..e.len().saturating_sub(1)]
}

/// The exponent of the last variable.
fn last(e: &[u32]) -> u32 {
    e.last().copied().unwrap_or(0)
}

/// The runs of `t` that agree in all but the last variable.
fn runs(t: &[(Exps, u64)]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=t.len() {
        if i == t.len() || prefix(&t[i].0) != prefix(&t[start].0) {
            out.push(start..i);
            start = i;
        }
    }
    out
}

/// A run (terms with equal prefix) as a polynomial in the last variable.
fn run_poly(field: Fp64, run: &[(Exps, u64)]) -> Up {
    let top = run.iter().map(|(e, _)| last(e)).max().unwrap_or(0);
    let mut c = vec![0u64; top as usize + 1];
    for (e, x) in run {
        c[last(e) as usize] = *x;
    }
    PolyIn::from_coeffs(field, c)
}

/// Append the terms `prefix · y^j · c_j` of the univariate `u` to `out`.
fn push_run(out: &mut Terms, pre: &[u32], u: &Up) {
    for (j, &c) in u.coeffs().iter().enumerate() {
        if c != 0 {
            let mut e = Exps::with_capacity(pre.len() + 1);
            e.extend_from_slice(pre);
            e.push(j as u32);
            out.push((e, c));
        }
    }
}

/// The degree of `t` in variable `v`.
fn degree_in(t: &[(Exps, u64)], v: usize) -> u32 {
    t.iter().map(|(e, _)| e[v]).max().unwrap_or(0)
}

/// The content of `t` in `𝔽ₚ[y]` (monic gcd of the coefficients of `t` as a
/// polynomial in the other variables).
fn content_last(field: Fp64, t: &[(Exps, u64)]) -> Up {
    let mut c = PolyIn::zero(field);
    for r in runs(t) {
        c = c.gcd(&run_poly(field, &t[r]));
        if c.degree() == Some(0) {
            break;
        }
    }
    c
}

/// Each coefficient (in the other variables) of `t` divided by, or
/// multiplied by, the univariate `u`.
fn map_runs(field: Fp64, t: &[(Exps, u64)], u: &Up, divide: bool) -> Terms {
    let mut out = Vec::with_capacity(t.len());
    for r in runs(t) {
        let pre: Exps = Exps::from_slice(prefix(&t[r.start].0));
        let run = run_poly(field, &t[r]);
        let new = if divide { run.div(u) } else { run.mul(u) };
        push_run(&mut out, &pre, &new);
    }
    out
}

/// The leading coefficient of `t` in `𝔽ₚ[y][x₀, …, x_{k−2}]` (lex) when
/// `x_i` (`i < k − 1`) is the first variable: a polynomial in `y`.
fn lc_main(field: Fp64, t: &[(Exps, u64)], i: usize) -> Up {
    let mut best: Option<&Exps> = None;
    let mut coeffs: Vec<(u32, u64)> = Vec::new();
    for (e, c) in t {
        let n = e.len() - 1;
        let ord = best.map_or(Ordering::Greater, |b| cmp_main(e, b, i, n));
        match ord {
            Ordering::Greater => {
                best = Some(e);
                coeffs.clear();
                coeffs.push((last(e), *c));
            }
            Ordering::Equal => coeffs.push((last(e), *c)),
            Ordering::Less => {}
        }
    }
    let top = coeffs.iter().map(|&(j, _)| j).max().unwrap_or(0);
    let mut c = vec![0u64; top as usize + 1];
    for (j, x) in coeffs {
        c[j as usize] = x;
    }
    PolyIn::from_coeffs(field, c)
}

/// `t` at `y = a`: a polynomial in the other variables (still ascending).
fn eval_last(field: Fp64, t: &[(Exps, u64)], a: u64) -> Terms {
    let top = t.iter().map(|(e, _)| last(e)).max().unwrap_or(0) as usize;
    let table = top <= 4 * t.len() + 16;
    let pows: Vec<u64> = if table {
        let mut v = Vec::with_capacity(top + 1);
        let mut acc = 1u64;
        for _ in 0..=top {
            v.push(acc);
            acc = field.mul(&acc, &a);
        }
        v
    } else {
        Vec::new()
    };
    let pow = |j: u32| {
        if table {
            pows[j as usize]
        } else {
            field.pow(a, u64::from(j))
        }
    };
    let mut out = Vec::new();
    for r in runs(t) {
        let mut s = 0u64;
        for (e, c) in &t[r.clone()] {
            s = field.add(&s, &field.mul(c, &pow(last(e))));
        }
        if s != 0 {
            out.push((Exps::from_slice(prefix(&t[r.start].0)), s));
        }
    }
    out
}

/// In-place inverses of non-zero field elements (Montgomery's trick: one
/// exponentiation for all).
fn batch_inv(field: Fp64, xs: &mut [u64]) {
    let mut prefix = Vec::with_capacity(xs.len());
    let mut acc = 1u64;
    for &x in xs.iter() {
        prefix.push(acc);
        acc = field.mul(&acc, &x);
    }
    let mut inv = field.inv(&acc).unwrap_or(0);
    for (x, pre) in xs.iter_mut().zip(prefix).rev() {
        let xi = field.mul(&inv, &pre);
        inv = field.mul(&inv, x);
        *x = xi;
    }
}

/// The polynomial in one more variable `y` (appended last) whose image at
/// `y = points[j]` is `images[j]`, of degree below `points.len()` in `y`
/// (Newton's divided differences, coefficient by coefficient).
fn interpolate(field: Fp64, points: &[u64], images: &[Terms]) -> Terms {
    let n = points.len();
    let mut values: BTreeMap<&Exps, Vec<u64>> = BTreeMap::new();
    for (j, img) in images.iter().enumerate() {
        for (e, c) in img {
            values.entry(e).or_insert_with(|| vec![0; n])[j] = *c;
        }
    }
    // inv[(j, i)] = 1/(a_i − a_{i−j}) for 1 ≤ j ≤ i < n, row-major by j.
    let mut inv: Vec<u64> = Vec::with_capacity(n * n / 2);
    for j in 1..n {
        for i in j..n {
            inv.push(field.sub(&points[i], &points[i - j]));
        }
    }
    batch_inv(field, &mut inv);
    let row = |j: usize| (j - 1) * n - (j - 1) * j / 2;
    let mut out = Vec::new();
    for (e, mut c) in values {
        for j in 1..n {
            let base = row(j);
            for i in (j..n).rev() {
                let d = field.sub(&c[i], &c[i - 1]);
                c[i] = field.mul(&d, &inv[base + i - j]);
            }
        }
        // Newton form → monomial basis by Horner in `(y − a_j)`.
        let mut poly = vec![c[n - 1]];
        for j in (0..n - 1).rev() {
            let mut next = vec![0u64; poly.len() + 1];
            for (k, &x) in poly.iter().enumerate() {
                next[k + 1] = field.add(&next[k + 1], &x);
                next[k] = field.sub(&next[k], &field.mul(&x, &points[j]));
            }
            next[0] = field.add(&next[0], &c[j]);
            poly = next;
        }
        push_run(&mut out, e, &PolyIn::from_coeffs(field, poly));
    }
    out
}

/// The monic (in lex order) gcd of the non-zero `f`, `g` in
/// `𝔽ₚ[x₀, …, x_{k−1}]`: `Ok(None)` for an unlucky prime or point (an image
/// above a degree bound, or too many such), `Err(Restart)` when a bound was
/// lowered.  The recursion depth is the number of variables.
fn gcd_mod_p(
    ctx: &mut Ctx,
    f: &[(Exps, u64)],
    g: &[(Exps, u64)],
    k: usize,
) -> Result<Option<Terms>, Restart> {
    let field = ctx.field;
    if k == 1 {
        ctx.work += 1;
        let h = run_poly(field, f).gcd(&run_poly(field, g));
        let d = h.degree().unwrap_or(0) as u32;
        if d > ctx.bounds.deg[0] {
            return Ok(None);
        }
        if d < ctx.bounds.deg[0] {
            ctx.bounds.deg[0] = d;
            return Err(Restart);
        }
        let mut out = Vec::new();
        push_run(&mut out, &[], &h);
        return Ok(Some(out));
    }
    let y = k - 1;
    let (dyf, dyg) = (degree_in(f, y), degree_in(g, y));
    let contf = content_last(field, f);
    let contg = content_last(field, g);
    let f = map_runs(field, f, &contf, true);
    let g = map_runs(field, g, &contg, true);
    let conth = contf.gcd(&contg);
    let deg = |u: &Up| u.degree().unwrap_or(0) as u32;
    let dch = deg(&conth);
    if dch > ctx.bounds.cont[y] {
        return Ok(None);
    }
    if dch < ctx.bounds.cont[y] {
        ctx.bounds.cont[y] = dch;
        return Err(Restart);
    }
    let lcf = run_poly(field, runs(&f).last().map_or(&[][..], |r| &f[r.clone()]));
    let lcg = run_poly(field, runs(&g).last().map_or(&[][..], |r| &g[r.clone()]));
    let delta = lcf.gcd(&lcg);
    let mut evaltest = delta.clone();
    for i in 0..y {
        evaltest = evaltest.mul(&lc_main(field, &f, i).gcd(&lc_main(field, &g, i)));
    }
    let needed = i64::from(dyf - deg(&contf))
        .min(i64::from(dyg - deg(&contg)))
        .min(i64::from(ctx.bounds.deg[y]) - i64::from(ctx.bounds.cont[y]) + i64::from(deg(&delta)))
        .max(0)
        + 1;
    let needed = usize::try_from(needed).unwrap_or(usize::MAX);
    let p = field.modulus();
    let mut points: Vec<u64> = Vec::with_capacity(needed);
    let mut images: Vec<Terms> = Vec::with_capacity(needed);
    let mut unlucky = 0usize;
    let mut tries = 0usize;
    while tries < 4 * needed + 16 {
        tries += 1;
        if ctx.work > MAX_WORK {
            return Ok(None);
        }
        let a = ctx.rng.next_u64() % p;
        if points.contains(&a) || evaltest.eval(a) == 0 {
            continue;
        }
        let fa = eval_last(field, &f, a);
        let ga = eval_last(field, &g, a);
        let Some(ha) = gcd_mod_p(ctx, &fa, &ga, y)? else {
            unlucky += 1;
            if unlucky > points.len() {
                return Ok(None);
            }
            continue;
        };
        if ha.len() == 1 && ha[0].0.iter().all(|&x| x == 0) {
            // The primitive parts are coprime: the gcd is the content.
            let mut out = Vec::new();
            push_run(&mut out, &Exps::from_elem(0, y), &conth);
            return Ok(Some(out));
        }
        let da = delta.eval(a);
        points.push(a);
        images.push(
            ha.into_iter()
                .map(|(e, c)| (e, field.mul(&c, &da)))
                .collect(),
        );
        if points.len() == needed {
            let h = interpolate(field, &points, &images);
            let h = map_runs(field, &h, &content_last(field, &h), true);
            let h = map_runs(field, &h, &conth, false);
            let dyh = degree_in(&h, y);
            if dyh > ctx.bounds.deg[y] {
                return Ok(None);
            }
            if dyh < ctx.bounds.deg[y] {
                ctx.bounds.deg[y] = dyh;
                return Err(Restart);
            }
            return Ok(Some(h));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zmap(terms: &[(&[u32], i64)]) -> ZMap {
        terms
            .iter()
            .map(|(e, c)| (Exps::from_slice(e), BigInt::from(*c)))
            .collect()
    }

    fn mul(a: &ZMap, b: &ZMap) -> ZMap {
        let mut out = ZMap::new();
        for (ea, ca) in a {
            for (eb, cb) in b {
                let e: Exps = ea.iter().zip(eb).map(|(x, y)| x + y).collect();
                *out.entry(e).or_default() += ca * cb;
            }
        }
        out.retain(|_, c| !c.is_zero());
        out
    }

    #[test]
    fn bivariate_common_factor() {
        // h = x·y + 3, f = h·(x − y), g = h·(x + y + 1).
        let h = zmap(&[(&[1, 1], 1), (&[0, 0], 3)]);
        let f = mul(&h, &zmap(&[(&[1, 0], 1), (&[0, 1], -1)]));
        let g = mul(&h, &zmap(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], 1)]));
        assert_eq!(brown_gcd(2, &f, &g), Some(h));
    }

    #[test]
    fn content_in_a_variable_and_integer_content() {
        // f = 6·z·(x + 1), g = 4·z²·(x + 1)·(y − 2): gcd 2·z·(x + 1).
        let f = mul(
            &zmap(&[(&[0, 0, 1], 6)]),
            &zmap(&[(&[1, 0, 0], 1), (&[0, 0, 0], 1)]),
        );
        let g = mul(
            &mul(
                &zmap(&[(&[0, 0, 2], 4)]),
                &zmap(&[(&[1, 0, 0], 1), (&[0, 0, 0], 1)]),
            ),
            &zmap(&[(&[0, 1, 0], 1), (&[0, 0, 0], -2)]),
        );
        let want = zmap(&[(&[1, 0, 1], 2), (&[0, 0, 1], 2)]);
        assert_eq!(brown_gcd(3, &f, &g), Some(want));
    }

    /// Brown and the heuristic gcd agree (up to the sign the heuristic does
    /// not normalise) on random products `h·a`, `h·b` in two to four
    /// variables, and the gcd is divisible by `h`.
    #[test]
    fn agrees_with_the_heuristic() {
        let mut rng = SplitMix64::new(42);
        let mut checked = 0;
        for _ in 0..40 {
            let nv = 2 + (rng.next_u64() % 3) as usize;
            let mut random = |terms: usize| -> ZMap {
                let mut p = ZMap::new();
                for _ in 0..terms {
                    let e: Exps = (0..nv).map(|_| (rng.next_u64() % 3) as u32).collect();
                    let c = (rng.next_u64() % 41) as i64 - 20;
                    *p.entry(e).or_default() += BigInt::from(c);
                }
                *p.entry(Exps::from_elem(0, nv)).or_default() += BigInt::from(1);
                p.retain(|_, c| !c.is_zero());
                p
            };
            let (h, a, b) = (random(3), random(3), random(3));
            let (f, g) = (mul(&h, &a), mul(&h, &b));
            let Some(mine) = brown_gcd(nv, &f, &g) else {
                continue;
            };
            let Some(theirs) = modgcd::heu_gcd(nv, &f, &g) else {
                continue;
            };
            let negative = theirs
                .h
                .last_key_value()
                .is_some_and(|(_, c)| c.is_negative());
            let theirs: ZMap = if negative {
                theirs.h.into_iter().map(|(e, c)| (e, -c)).collect()
            } else {
                theirs.h
            };
            assert_eq!(mine, theirs, "gcd of {f:?} and {g:?}");
            assert!(modgcd::zmap_div_exact(&mine, &h).is_some());
            checked += 1;
        }
        assert!(checked >= 30, "only {checked} cases compared");
    }

    #[test]
    fn coprime_inputs() {
        let f = zmap(&[(&[2, 0], 1), (&[0, 1], 1)]);
        let g = zmap(&[(&[1, 1], 1), (&[0, 0], 7)]);
        assert_eq!(brown_gcd(2, &f, &g), Some(zmap(&[(&[0, 0], 1)])));
    }
}
