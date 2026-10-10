//! Multivariate factorization over ℤ by evaluation, Hensel lifting and
//! recombination — the route of
//! [`factor_multivariate`](super::factor_zassenhaus::factor_multivariate)
//! for polynomials whose Kronecker image is too large.
//!
//! For a primitive `f ∈ ℤ[x, y₁, …, yₖ]`:
//!
//! 1. **Content and square-free parts in a main variable `x`.**  The
//!    content of `f` in `x` (a polynomial in the `yⱼ`, the gcd of the
//!    coefficients of the powers of `x`) is factored on its own; the
//!    primitive part is split by Yun's algorithm in `x` (gcds in
//!    `ℤ[x, y]`), so that every part is square-free and primitive in `x`.
//! 2. **Evaluation.**  For a square-free part `g` of degree `n` in `x` with
//!    leading coefficient `l(y)`, an integer point `a` is chosen with
//!    `l(a) ≠ 0` and `g(x, a)` square-free of degree `n`; its factorization
//!    over ℤ (Berlekamp–Zassenhaus) bounds the factorization of `g`: a
//!    factor of `g` keeps its degree in `x` at `a`, so an irreducible image
//!    proves `g` irreducible.  Of a few such points the one with the fewest
//!    factors is kept.
//! 3. **Hensel lifting over ℚ in the ideal `(y − a)`.**  With `y ↦ y + a`
//!    the point is the origin.  The monic factors `uᵢ(x)` of `g(x, 0)/l(0)`
//!    are lifted to `Hᵢ ∈ ℚ[x][[y]]`, monic in `x`, with
//!    `g ≡ l · ∏ Hᵢ` modulo total degree `B` in `y`: at degree `d` the
//!    homogeneous part `E_d` of the error is distributed by the partial
//!    fraction identity `Σ sᵢ ∏_{j≠i} uⱼ = 1` (`Hᵢ += (E_d/l(0)) sᵢ mod uᵢ`).
//!    The monic factorization of `g` over the power series ring is unique,
//!    so a true factor `h` of `g` is `l·h/lc_x(h) = l · ∏_{i∈S} Hᵢ` for some
//!    subset `S`, a polynomial of total degree in `y` below
//!    `B = tdeg(l) + tdeg(g) + 1`.
//! 4. **Recombination.**  Subsets of increasing size are tried: the
//!    truncation of `l · ∏_{i∈S} Hᵢ` is a factor candidate when it divides
//!    `l · g` exactly; its primitive part in `x` is then a factor of `g`.
//!    When no subset of at most half the remaining factors works, the
//!    remaining cofactor is irreducible.
//!
//! This is the classical evaluation/lifting/recombination approach (D. R.
//! Musser, *J. ACM* 22 (1975) 291–308; Geddes, Czapor and Labahn,
//! *Algorithms for Computer Algebra*, ch. 6), lifting monic factors over ℚ
//! and handling the leading coefficient by multiplying each candidate by
//! `l`.  SymPy's `dmp_zz_wang` (Wang's EEZ, 1978) instead predetermines the
//! leading coefficients of the factors and lifts modulo a large prime, which
//! avoids recombination but must restart when the image has extraneous
//! factors; it serves as the oracle here, its code is not followed.
//!
//! Every returned factor divides the input exactly; the caller multiplies
//! back.  `None` means the method gave up (no usable evaluation point, or a
//! work budget was exhausted); it never returns a reducible factor as
//! irreducible unless the recombination budget runs out, which also
//! returns `None`.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

use super::dense::Poly;
use super::factor_zassenhaus::{factor_zassenhaus_checked, factor_zassenhaus_with_content};
use super::multipoly::{MonomialOrd, MultiPoly};

type Q = Ratio<BigInt>;

/// Evaluation points examined per square-free part.
const MAX_EVALUATION_TRIES: usize = 64;

/// Usable evaluation points compared (the one with the fewest univariate
/// factors is lifted).
const GOOD_POINTS: usize = 3;

/// Factor subsets examined during recombination before giving up.
const MAX_SUBSETS: usize = 20_000;

/// Terms of any intermediate product of the lifting before giving up.
const MAX_TERMS: usize = 400_000;

/// Factor a primitive integer polynomial with positive leading coefficient
/// (in the order `O`) involving at least one variable into irreducible
/// factors over ℤ, listed with repetition.  Every factor is primitive with
/// positive leading coefficient, and their product is `f`.
///
/// `None` if the method gives up (see the module documentation).
pub(crate) fn factor_primitive<O: MonomialOrd>(f: &MultiPoly<O>) -> Option<Vec<MultiPoly<O>>> {
    let nv = f.num_vars();
    let mut work: Vec<(MultiPoly<O>, u32)> = vec![(f.clone(), 1)];
    let mut out: Vec<(MultiPoly<O>, u32)> = Vec::new();
    while let Some((g, m)) = work.pop() {
        let present = g.variables_present();
        if present.is_empty() {
            continue;
        }
        // Monomial content.
        let mono = g.monomial_content();
        let g = if mono.iter().any(|&e| e > 0) {
            let mut terms = Vec::with_capacity(g.num_terms());
            for (exp, c) in g.terms() {
                let e: Vec<u32> = exp.iter().zip(&mono).map(|(a, b)| a - b).collect();
                terms.push((e, c.clone()));
            }
            for (i, &e) in mono.iter().enumerate() {
                if e > 0 {
                    out.push((MultiPoly::var(nv, i), m * e));
                }
            }
            MultiPoly::from_terms(nv, terms)?
        } else {
            g
        };
        let present = g.variables_present();
        match present.len() {
            0 => {}
            1 => {
                let v = present[0];
                let (_, fs) = factor_zassenhaus_with_content(&to_uni(&g, v));
                for (h, k) in fs {
                    out.push((from_uni(&h, v, nv), m * k));
                }
            }
            _ => {
                let x = main_variable(&g, &present);
                let split = g.split_content_in(x);
                let (c, p) = (split.content, split.primitive);
                if c.total_degree().unwrap_or(0) > 0 {
                    work.push((normalize(&c), m));
                }
                for (s, k) in square_free_in(&p, x)? {
                    if s.degree_in(x) == 0 {
                        work.push((normalize(&s), m * k));
                        continue;
                    }
                    for h in factor_square_free(&s, x)? {
                        if h.variables_present().len() < 2 {
                            // A univariate piece (the parts are primitive in
                            // `x`, so this is a polynomial in `x` alone):
                            // factor it on the work list.
                            work.push((h, m * k));
                        } else {
                            out.push((h, m * k));
                        }
                    }
                }
            }
        }
    }
    let mut result = Vec::new();
    for (g, m) in out {
        for _ in 0..m {
            result.push(g.clone());
        }
    }
    Some(result)
}

/// Integer primitive part with positive leading coefficient.
fn normalize<O: MonomialOrd>(p: &MultiPoly<O>) -> MultiPoly<O> {
    let q = p.primitive_part_q();
    if q.leading_coeff().is_some_and(|c| c.is_negative()) {
        q.neg()
    } else {
        q
    }
}

/// The main variable: a constant leading coefficient first (no content to
/// carry through the lifting), then the lowest degree.
fn main_variable<O: MonomialOrd>(g: &MultiPoly<O>, present: &[usize]) -> usize {
    let mut best = present[0];
    let mut best_key = (true, u32::MAX);
    for &v in present {
        let d = g.degree_in(v);
        let lc_varies = g.coeff_in(v, d).total_degree().unwrap_or(0) > 0;
        let key = (lc_varies, d);
        if key < best_key {
            best_key = key;
            best = v;
        }
    }
    best
}

/// Yun's square-free decomposition of `p`, primitive in `x`, as a
/// polynomial in `x` over `ℚ(y)` with the gcds taken in `ℤ[x, y]`: parts
/// `(sᵢ, i)` with `p = c · ∏ sᵢ^i`, each `sᵢ` normalized.
fn square_free_in<O: MonomialOrd>(p: &MultiPoly<O>, x: usize) -> Option<Vec<(MultiPoly<O>, u32)>> {
    let dp = p.partial_derivative(x);
    let g = MultiPoly::gcd(p, &dp);
    if g.degree_in(x) == 0 {
        return Some(vec![(normalize(p), 1)]);
    }
    let mut b = p.div_exact(&g)?;
    let c = dp.div_exact(&g)?;
    let mut d = c.sub(&b.partial_derivative(x));
    let mut out = Vec::new();
    let mut i = 1u32;
    while b.degree_in(x) > 0 {
        let a = MultiPoly::gcd(&b, &d);
        if a.degree_in(x) > 0 {
            out.push((normalize(&a), i));
        }
        b = b.div_exact(&a)?;
        let c = d.div_exact(&a)?;
        d = c.sub(&b.partial_derivative(x));
        i = i.checked_add(1)?;
    }
    Some(out)
}

/// A polynomial in the single variable `v` as a dense `Poly`.
fn to_uni<O: MonomialOrd>(p: &MultiPoly<O>, v: usize) -> Poly {
    let deg = p.degree_in(v) as usize;
    let mut coeffs = vec![Q::zero(); deg + 1];
    for (exp, c) in p.terms() {
        coeffs[exp[v] as usize] += c;
    }
    Poly::from_coeffs(coeffs)
}

/// A dense `Poly` as a polynomial in the variable `v` of `nv`.
fn from_uni<O: MonomialOrd>(p: &Poly, v: usize, nv: usize) -> MultiPoly<O> {
    let mut terms = Vec::new();
    for (k, c) in p.coeffs().iter().enumerate() {
        if !c.is_zero() {
            let mut e = vec![0u32; nv];
            e[v] = k as u32;
            terms.push((e, c.clone()));
        }
    }
    MultiPoly::from_terms(nv, terms).unwrap_or_else(|| MultiPoly::zero(nv))
}

/// Total degree of a term in the variables `ys`.
fn y_degree(exp: &[u32], ys: &[usize]) -> u32 {
    ys.iter().map(|&j| exp[j]).sum()
}

/// `p(x, y + a)`: each `yⱼ` (index `ys[j]`) replaced by `yⱼ + aⱼ`.
fn shift<O: MonomialOrd>(p: &MultiPoly<O>, ys: &[usize], a: &[Q]) -> MultiPoly<O> {
    let nv = p.num_vars();
    let mut cur = p.clone();
    for (&j, aj) in ys.iter().zip(a) {
        if aj.is_zero() {
            continue;
        }
        let mut acc: FxHashMap<Vec<u32>, Q> = FxHashMap::default();
        for (exp, c) in cur.terms() {
            let e = exp[j];
            // Σₖ C(e, k) aʲ^{e−k} yⱼ^k
            let mut binom = BigInt::one();
            let mut apow: Vec<Q> = Vec::with_capacity(e as usize + 1);
            let mut t = Q::one();
            for _ in 0..=e {
                apow.push(t.clone());
                t *= aj;
            }
            for k in 0..=e {
                let coef = c * &apow[(e - k) as usize] * Q::from_integer(binom.clone());
                let mut ne = exp.to_vec();
                ne[j] = k;
                let slot = acc.entry(ne).or_insert_with(Q::zero);
                *slot += coef;
                binom = binom * BigInt::from(e - k) / BigInt::from(k + 1);
            }
        }
        // Every exponent vector has `nv` entries: `from_terms` cannot fail.
        cur = MultiPoly::from_terms(nv, acc.into_iter().collect())
            .unwrap_or_else(|| MultiPoly::zero(nv));
    }
    cur
}

/// Terms of `p` of total degree at most `max` in `ys`.
fn truncate<O: MonomialOrd>(p: &MultiPoly<O>, ys: &[usize], max: u32) -> MultiPoly<O> {
    let terms: Vec<(Vec<u32>, Q)> = p
        .terms()
        .filter(|(e, _)| y_degree(e, ys) <= max)
        .map(|(e, c)| (e.to_vec(), c.clone()))
        .collect();
    MultiPoly::from_terms(p.num_vars(), terms).unwrap_or_else(|| MultiPoly::zero(p.num_vars()))
}

/// `a · b` truncated at total degree `max` in `ys`; `None` beyond
/// [`MAX_TERMS`] terms.
fn mul_trunc<O: MonomialOrd>(
    a: &MultiPoly<O>,
    b: &MultiPoly<O>,
    ys: &[usize],
    max: u32,
) -> Option<MultiPoly<O>> {
    let nv = a.num_vars();
    let bt: Vec<(&[u32], &Q, u32)> = b.terms().map(|(e, c)| (e, c, y_degree(e, ys))).collect();
    let mut acc: FxHashMap<Vec<u32>, Q> = FxHashMap::default();
    for (ea, ca) in a.terms() {
        let da = y_degree(ea, ys);
        if da > max {
            continue;
        }
        for &(eb, cb, db) in &bt {
            if da + db > max {
                continue;
            }
            let e: Vec<u32> = ea.iter().zip(eb).map(|(p, q)| p + q).collect();
            let slot = acc.entry(e).or_insert_with(Q::zero);
            *slot += ca * cb;
        }
        if acc.len() > MAX_TERMS {
            return None;
        }
    }
    MultiPoly::from_terms(nv, acc.into_iter().collect())
}

/// `p` with each `yⱼ` (index `ys[j]`) set to `aⱼ`, as a polynomial in `x`.
fn image_in_x<O: MonomialOrd>(p: &MultiPoly<O>, x: usize, ys: &[usize], a: &[Q]) -> Poly {
    let mut cur = p.clone();
    for (&j, aj) in ys.iter().zip(a) {
        cur = cur.eval_var(j, aj);
    }
    to_uni(&cur, x)
}

/// The irreducible factors of `g`, square-free and primitive in `x` of
/// degree at least 1 and involving at least one other variable.
fn factor_square_free<O: MonomialOrd>(g: &MultiPoly<O>, x: usize) -> Option<Vec<MultiPoly<O>>> {
    let nv = g.num_vars();
    let n = g.degree_in(x);
    if n <= 1 {
        return Some(vec![normalize(g)]);
    }
    let ys: Vec<usize> = g
        .variables_present()
        .into_iter()
        .filter(|&v| v != x)
        .collect();
    if ys.is_empty() {
        let (_, fs) = factor_zassenhaus_with_content(&to_uni(g, x));
        return Some(
            fs.into_iter()
                .flat_map(|(h, k)| std::iter::repeat_n(from_uni(&h, x, nv), k as usize))
                .collect(),
        );
    }
    let lc = g.coeff_in(x, n);

    // 2. Evaluation point with the fewest univariate factors.
    let mut rng = crate::base::rng::SplitMix64::new(0x5EED_FAC7_0000 + u64::from(n));
    let mut best: Option<(Vec<Q>, Vec<Poly>)> = None;
    let mut good = 0usize;
    for t in 0..MAX_EVALUATION_TRIES {
        let a: Vec<Q> = if t == 0 {
            vec![Q::zero(); ys.len()]
        } else {
            let r = 1 + (t / 4) as u64;
            ys.iter()
                .map(|_| {
                    Q::from_integer(BigInt::from(
                        (rng.next_u64() % (2 * r + 1)) as i64 - r as i64,
                    ))
                })
                .collect()
        };
        let l0 = image_in_x(&lc, x, &ys, &a);
        if l0.is_zero() {
            continue;
        }
        let u = image_in_x(g, x, &ys, &a);
        if u.degree() != Some(n as usize) || u.is_squarefree() != Some(true) {
            continue;
        }
        let (_, fs, complete) = factor_zassenhaus_checked(&u);
        if !complete {
            continue;
        }
        if fs.len() == 1 && fs[0].1 == 1 {
            // A factor of `g` keeps its degree in `x` at `a`.
            return Some(vec![normalize(g)]);
        }
        if fs.iter().any(|(_, k)| *k != 1) {
            continue;
        }
        let monic: Vec<Poly> = fs.into_iter().map(|(h, _)| h.make_monic()).collect();
        if best.as_ref().is_none_or(|(_, b)| monic.len() < b.len()) {
            best = Some((a, monic));
        }
        good += 1;
        if good >= GOOD_POINTS {
            break;
        }
    }
    let (a, us) = best?;

    // 3. Hensel lifting at the origin of the shifted variables.
    let gs = shift(g, &ys, &a);
    let ls = shift(&lc, &ys, &a);
    let l0 = ls
        .terms()
        .find(|(e, _)| y_degree(e, &ys) == 0)
        .map(|(_, c)| c.clone())?;
    let r = us.len();
    let mut s_coef: Vec<Poly> = Vec::with_capacity(r);
    for (i, ui) in us.iter().enumerate() {
        let mut others = Poly::from_int(1);
        for (j, u) in us.iter().enumerate() {
            if j != i {
                others = &others * u;
            }
        }
        let e = Poly::extended_gcd(&others, ui);
        if e.gcd.degree() != Some(0) {
            return None;
        }
        let inv = Q::one() / e.gcd.coeff(0);
        s_coef.push(e.x.scale(&inv).rem(ui));
    }
    let tdeg_y = |p: &MultiPoly<O>| p.terms().map(|(e, _)| y_degree(e, &ys)).max().unwrap_or(0);
    let bound = tdeg_y(&ls) + tdeg_y(&gs);
    let mut hs: Vec<MultiPoly<O>> = us.iter().map(|u| from_uni(u, x, nv)).collect();
    let inv_l0 = Q::one() / &l0;
    for d in 1..=bound {
        let mut prod = truncate(&ls, &ys, d);
        for h in &hs {
            prod = mul_trunc(&prod, h, &ys, d)?;
        }
        let err = truncate(&gs, &ys, d).sub(&prod);
        if err.is_zero() {
            continue;
        }
        // The parts below degree `d` vanish by construction.
        let mut by_mono: FxHashMap<Vec<u32>, Vec<Q>> = FxHashMap::default();
        for (e, c) in err.terms() {
            if y_degree(e, &ys) != d {
                return None;
            }
            let mut key = e.to_vec();
            let k = key[x] as usize;
            key[x] = 0;
            let coeffs = by_mono.entry(key).or_default();
            if coeffs.len() <= k {
                coeffs.resize(k + 1, Q::zero());
            }
            coeffs[k] = c * &inv_l0;
        }
        for (mono, coeffs) in by_mono {
            let c = Poly::from_coeffs(coeffs);
            for ((h, s), u) in hs.iter_mut().zip(&s_coef).zip(&us) {
                let sigma = (&c * s).rem(u);
                let mut terms = Vec::new();
                for (k, cf) in sigma.coeffs().iter().enumerate() {
                    if !cf.is_zero() {
                        let mut e = mono.clone();
                        e[x] = k as u32;
                        terms.push((e, cf.clone()));
                    }
                }
                *h = h.add(&MultiPoly::from_terms(nv, terms)?);
            }
        }
    }

    // 4. Recombination (in the shifted variables).
    let mut remaining: Vec<usize> = (0..r).collect();
    let mut cur = gs;
    let mut found: Vec<MultiPoly<O>> = Vec::new();
    let mut budget = MAX_SUBSETS;
    let mut size = 1usize;
    'sizes: while 2 * size <= remaining.len() {
        let lcur = cur.coeff_in(x, cur.degree_in(x));
        let target = lcur.mul(&cur);
        let mut idx: Vec<usize> = (0..size).collect();
        loop {
            budget = budget.checked_sub(1)?;
            let mut cand = truncate(&lcur, &ys, bound);
            for &k in &idx {
                cand = mul_trunc(&cand, &hs[remaining[k]], &ys, bound)?;
            }
            if target.div_exact(&cand).is_some() {
                let h = normalize(&cand.split_content_in(x).primitive);
                if h.degree_in(x) > 0
                    && let Some(q) = cur.div_exact(&h)
                {
                    found.push(h);
                    cur = q;
                    for &k in idx.iter().rev() {
                        remaining.remove(k);
                    }
                    continue 'sizes;
                }
            }
            if !next_subset(&mut idx, remaining.len()) {
                break;
            }
        }
        size += 1;
    }
    if cur.degree_in(x) > 0 {
        found.push(normalize(&cur));
    }
    let neg: Vec<Q> = a.iter().map(|v| -v).collect();
    Some(
        found
            .iter()
            .map(|h| normalize(&shift(h, &ys, &neg)))
            .collect(),
    )
}

/// Advance `idx` (strictly increasing indices below `n`) to the next
/// combination of the same size; `false` after the last one.
fn next_subset(idx: &mut [usize], n: usize) -> bool {
    let k = idx.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if idx[i] < n - k + i {
            idx[i] += 1;
            for j in i + 1..k {
                idx[j] = idx[j - 1] + 1;
            }
            return true;
        }
    }
    false
}
