//! Eigenvalue multiplicity via polynomial factoring (part of [`matrix`](super)).

use super::*;

/// Sort `vals` in descending numeric order if *every* entry evaluates to
/// an `f64`; otherwise leave the order unchanged (a partial comparator
/// would not be a total order).
pub(super) fn sort_descending_numeric(vals: &mut [Ex]) {
    let keys: Option<Vec<f64>> = vals.iter().map(|v| v.eval_f64().ok()).collect();
    let Some(keys) = keys else { return };
    let mut order: Vec<usize> = (0..vals.len()).collect();
    order.sort_by(|&a, &b| keys[b].total_cmp(&keys[a]));
    let sorted: Vec<Ex> = order.iter().map(|&i| vals[i].clone()).collect();
    vals.clone_from_slice(&sorted);
}

/// Recursive cofactor expansion (used for the hard-coded 3×3 path).
pub(super) fn det_cofactor_inner(m: &[Vec<Ex>]) -> Ex {
    let n = m.len();
    if n == 1 {
        return m[0][0].clone();
    }
    if n == 2 {
        let ad = &m[0][0] * &m[1][1];
        let bc = &m[0][1] * &m[1][0];
        return ad - bc;
    }
    let mut result: Option<Ex> = None;
    for j in 0..n {
        let minor: Vec<Vec<Ex>> = (1..n)
            .map(|row| {
                (0..n)
                    .filter(|&col| col != j)
                    .map(|col| m[row][col].clone())
                    .collect()
            })
            .collect();
        let cofactor = det_cofactor_inner(&minor);
        let term = &m[0][j] * &cofactor;
        result = Some(match result {
            None => {
                if j % 2 == 0 {
                    term
                } else {
                    -term
                }
            }
            Some(acc) => {
                if j % 2 == 0 {
                    acc + term
                } else {
                    acc - term
                }
            }
        });
    }
    result.unwrap_or_else(|| m[0][0].clone())
}

/// Closed-form roots for `c_0 + c_1 λ + c_2 λ²` (degree ≤ 2), with
/// multiplicities.  Used when the generic solver cannot handle symbolic
/// coefficients.
pub(super) fn low_degree_roots(coeffs: &[Ex]) -> Vec<(Ex, usize)> {
    match coeffs.len() {
        2 => {
            // c0 + c1 λ = 0 → λ = −c0/c1
            let root = (-&coeffs[0] / &coeffs[1]).eval().simplify();
            vec![(root, 1)]
        }
        3 => {
            let (c0, c1, c2) = (&coeffs[0], &coeffs[1], &coeffs[2]);
            let ctx = c0.context();
            let two = ctx.int(2);
            let disc = (&c1.powi(2) - &(&(&ctx.int(4) * c2) * c0)).expand();
            // A constant of ℚ(√…, i) in normal form: its imaginary part is
            // then exactly 0 when it vanishes (`8/(1 + i) + 4i` is `4`), and
            // `√` of a negative real number evaluates (before 0.31 `evalf`
            // refused an eigenvalue whose radicand cancelled only
            // numerically onto the branch cut).
            let disc =
                crate::domains::linalg::algebraic_constant_normal_form(&disc).unwrap_or(disc);
            let c1 = &crate::domains::linalg::algebraic_constant_normal_form(c1)
                .unwrap_or_else(|| c1.clone());
            let denom = &two * c2;
            if ex_is_zero(&disc).is_true() {
                let root = (-c1 / &denom).eval();
                return vec![(root, 2)];
            }
            let sq = quadratic_sqrt(&disc);
            let r1 = (&(-c1 + &sq) / &denom).eval();
            let r2 = (&(-c1 - &sq) / &denom).eval();
            vec![(r1, 1), (r2, 1)]
        }
        _ => Vec::new(),
    }
}

/// `√q` when it is rational.
fn rational_sqrt_q(q: &Q) -> Option<Q> {
    use num_traits::Signed;
    if q.is_negative() {
        return None;
    }
    let n = q.numer().sqrt();
    let d = q.denom().sqrt();
    (&n * &n == *q.numer() && &d * &d == *q.denom()).then(|| Q::new(n, d))
}

/// A square root of `disc` for the quadratic formula.
///
/// Any `r` with `r² = disc` yields the same *set* `{(−b ± r)/2a}`, so we
/// are free to pick the friendliest one — it need not be the principal
/// root.  When `±disc` is a perfect square (`4ω²`, `(a−d)²`, …) forced
/// denesting (every symbol treated as positive) gives `2ω`, and for
/// negative discriminants `2·i·ω` instead of `√(−4ω²)`; the candidate is
/// then checked exactly (`r² − disc = 0`), so forcing is sound here.  The
/// nice form matters downstream: `A − λI` pivots then cancel structurally,
/// which is what makes `eigenvects` / `matrix_exp` of e.g. `[[0, −ω], [ω,
/// 0]]` work without sign assumptions on `ω`.  Falls back to `disc.sqrt()`.
fn quadratic_sqrt(disc: &Ex) -> Ex {
    quadratic_sqrt_with(disc, true)
}

/// [`quadratic_sqrt`], trying [`square_root_by_parts`] for a non-real
/// `disc` when `by_parts` (which calls this without it: depth ≤ 2).
fn quadratic_sqrt_with(disc: &Ex, by_parts: bool) -> Ex {
    use crate::base::node::ExprNode;
    use num_traits::{Signed, Zero};
    let ctx = disc.context();
    // A Gaussian rational `a + bi` with `a² + b² = s²` a rational square:
    // `√((s + a)/2) + sign(b)·i·√((s − a)/2)` (radicals of rationals, and
    // rational for `√(−3 − 4i) = 1 − 2i`).  Before 0.31 it stayed nested
    // and the eigenvalues of `[[−1, 0], [1 + i, i]]` were
    // `√(2i)/2 + i/2 − 1/2` and its conjugate.
    if let Some(crate::domains::linalg::GaussianRational { re: a, im: b }) =
        crate::domains::linalg::gaussian_rational_parts(disc)
        && !b.is_zero()
        && let Some(s) = rational_sqrt_q(&(&a * &a + &b * &b))
    {
        let two = Q::from_integer(BigInt::from(2));
        let re = ctx.from_ratio((&s + &a) / &two).sqrt();
        let im = ctx.from_ratio((&s - &a) / &two).sqrt();
        let im = if b.is_negative() { -im } else { im };
        return (&re + &(&ctx.i_unit() * &im)).eval();
    }
    // A root in ℚ(radicals, i) found by denesting, checked exactly.
    // (`sqrtdenest` may hand back the nested root itself, which passes the
    // check; a non-real radicand is then denested by its parts.  A check
    // that expands to a structural `0` has no generators, so the tower's
    // zero test leaves it undecided: before 0.44 `√(24 + 16√2) = 4 + 2√2`
    // was found and then dropped.)
    let denested = disc.sqrt().sqrtdenest();
    let check = (&denested.powi(2) - disc).expand();
    if (check.is_zero_structural()
        || Truth::from(crate::domains::linalg::algebraic_function_is_zero(&check)).is_true())
        && crate::domains::linalg::algebraic_function_is_zero(&denested).is_some()
    {
        // A radical of a non-real base (`√(2p²·i)` is built as `p·√2·√i`
        // since 0.44, when factors known non-negative come out of a root)
        // is denested by its parts too: `p + p·i`.
        if by_parts
            && (is_nested_radical(&denested) || has_radical_of_nonreal_base(&denested))
            && let Some(r) = square_root_by_parts(disc)
        {
            return r;
        }
        return denested;
    }
    let strip_abs = |e: &Ex| {
        e.replace(|v| match v.node() {
            ExprNode::Abs(inner) => Some(e.wrap(*inner)),
            _ => None,
        })
    };
    for (base, factor) in [(disc.clone(), ctx.one()), ((-disc).eval(), ctx.i_unit())] {
        let root = strip_abs(&base.sqrt().powdenest(true).simplify());
        if has_radical(&root) {
            continue;
        }
        let check = (&root.powi(2).expand() - &base.expand()).expand();
        if check.is_zero_structural() {
            return (&factor * &root).eval();
        }
    }

    square_root_by_factoring(disc)
        .or_else(|| by_parts.then(|| square_root_by_parts(disc)).flatten())
        .unwrap_or_else(|| disc.sqrt())
}

/// `r` with `r² = disc` for a non-real `disc = a + b·i` with real parts
/// `a`, `b` whose modulus `s = √(a² + b²)` has a square root without a
/// nested radical: `√((s + a)/2) ± i·√((s − a)/2)` (each square root by
/// [`quadratic_sqrt`]), checked exactly; `None` when no such `r` is found.
/// Before 0.44 `√(1 + 2√2·i)` (which is `√2 + i`) and `√(2p²·i)` (`p +
/// p·i`) stayed nested, so the eigenvalues of `[[3i + 2√2, −2i − 2√2],
/// [3i + 3√2, −2i − 3√2]]` (`i` and `−√2`) and of a matrix with the
/// eigenvalues `p·i` and `−p` (`p > 0`) came out of the quadratic formula
/// as radicals whose value lies on the negative real axis only up to an
/// imaginary part that no evaluation can prove zero: `matrix_sqrt` and
/// `matrix_log` returned entries that do not evaluate.
fn square_root_by_parts(disc: &Ex) -> Option<Ex> {
    use crate::base::node::ExprNode;
    let (a, b) = disc.as_real_imag();
    if b.is_zero_structural() {
        return None;
    }
    // Unknown parts (`re(x)` of a symbol of unknown realness) give nothing.
    let opaque = |e: &Ex| {
        let inner = e.inner.read();
        crate::base::walk::post_order_ids(&inner.arena, e.raw_id())
            .into_iter()
            .any(|id| {
                matches!(
                    inner.arena.node(id),
                    ExprNode::Re(_) | ExprNode::Im(_) | ExprNode::Conjugate(_) | ExprNode::Abs(_)
                )
            })
    };
    if opaque(&a) || opaque(&b) {
        return None;
    }
    let (a, b) = (a.expand(), b.expand());
    let s2 = (&a.powi(2) + &b.powi(2)).expand();
    let s = quadratic_sqrt_with(&s2, false);

    if is_nested_radical(&s) {
        return None;
    }
    let half = disc.context().rational(1, 2);
    let re = quadratic_sqrt_with(&(&(&s + &a) * &half).expand(), false);
    let im = quadratic_sqrt_with(&(&(&s - &a) * &half).expand(), false);
    if is_nested_radical(&re) || is_nested_radical(&im) {
        return None;
    }
    let i = disc.context().i_unit();
    for sign in [1i64, -1] {
        let r = (&re + &(&(&i * &im) * &disc.context().int(sign))).eval();
        let check = (&r.powi(2) - disc).expand();

        if check.is_zero_structural()
            || Truth::from(crate::domains::linalg::algebraic_function_is_zero(&check)).is_true()
        {
            return Some(r);
        }
    }
    None
}

/// Does `e` contain a radical whose radicand contains `i` (`√i`)?
fn has_radical_of_nonreal_base(e: &Ex) -> bool {
    use crate::base::node::ExprNode;
    let inner = e.inner.read();
    let arena = &inner.arena;
    crate::base::walk::post_order_ids(arena, e.raw_id())
        .into_iter()
        .any(|id| {
            matches!(arena.node(id), ExprNode::Pow(base, x)
                if arena.as_num(*x).is_some_and(|r| !r.is_integer())
                    && crate::base::walk::post_order_ids(arena, *base)
                        .into_iter()
                        .any(|b| matches!(arena.node(b), ExprNode::ImaginaryUnit)))
        })
}

/// Does `e` contain a radical whose radicand contains a radical?
pub(super) fn is_nested_radical(e: &Ex) -> bool {
    use crate::base::node::ExprNode;
    let inner = e.inner.read();
    let arena = &inner.arena;
    let radical = |id| {
        matches!(arena.node(id), ExprNode::Pow(_, x)
            if arena.as_num(*x).is_some_and(|r| !r.is_integer()))
    };
    let mut stack = vec![e.raw_id()];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let ExprNode::Pow(base, _) = arena.node(id)
            && radical(id)
            && crate::base::walk::post_order_ids(arena, *base)
                .into_iter()
                .any(radical)
        {
            return true;
        }
        stack.extend(arena.node(id).children());
    }
    false
}

/// `r` with `r² = disc` for a polynomial `disc` in the symbols that is
/// `±q²·∏ fᵢ^(2kᵢ)` over `ℚ` (factored by `factor_list_all`), checked
/// exactly.  Forced denesting misses such squares when they are expanded:
/// before 0.38 the eigenvalues of `[[2y − x, x − y], [2y − 2x, 2x − y]]`
/// were `(x + y ± √(x² − 2xy + y²))/2` — `x` and `y` region by region —
/// so a 4×4 matrix with the eigenvalue `x` three times had it twice plus
/// once in disguise.
fn square_root_by_factoring(disc: &Ex) -> Option<Ex> {
    use num_traits::Signed;
    if disc.free_symbols().is_empty() || has_radical(disc) {
        return None;
    }
    let ctx = disc.context();
    let (content, factors) = disc.factor_list_all();
    let c = content.as_rational()?;
    if factors.is_empty() || factors.iter().any(|(_, m)| m % 2 != 0) {
        return None;
    }
    let mut root = if c.is_negative() {
        &ctx.from_ratio(rational_sqrt_q(&-c)?) * &ctx.i_unit()
    } else {
        ctx.from_ratio(rational_sqrt_q(&c)?)
    };
    for (f, m) in &factors {
        root = &root * &f.powi(i64::from(m / 2));
    }
    let root = root.expand();
    (&root.powi(2) - disc)
        .expand()
        .is_zero_structural()
        .then_some(root)
}

/// A coefficient of [`parametric_radical_roots`] in normal form: a rational
/// function of the symbols by `ratsimp`, anything else (radicals) expanded.
pub(super) fn parametric_normal(e: &Ex) -> Ex {
    if has_radical(e) || has_root_of(e) {
        e.expand()
    } else {
        e.ratsimp()
    }
}

/// The roots of the cubic or quartic `Σ coeffs[k]·λᵏ` (ascending) whose
/// coefficients have free symbols, in radicals: SymPy's `roots_cubic` and
/// `roots_quartic` (`polys/polyroots.py`, the forms `roots(f, cubics=True,
/// quartics=True)` returns for a symbolic cubic or quartic and which
/// `Matrix.eigenvals` returns).  Every case test is exact (a rational
/// function of the symbols that vanishes identically): the degenerate shapes
/// (`x·(…)`, `p = 0`, `q = 0`, the quasi-symmetric and biquadratic quartics)
/// get their own short forms.
///
/// Cubic `x³ + a·x² + b·x + c`, depressed by `x = y − a/3` to `y³ + p·y + q`:
/// the roots are `ωᵏ·w − ω̄ᵏ·p/(3w) − a/3` (`ω = −1/2 + √3·i/2`) with
/// `w = ∛(−q/2 + √R)` when `q` is known to be negative and `w = −∛(q/2 + √R)`
/// otherwise, `R = q²/4 + p³/27` — SymPy's two branches (its `C =
/// ∛((D₁ + √(D₁² − 4D₀³))/2)` is `−3w`).  For `q` of known sign (with any
/// `p`) `w ≠ 0`, so the formula holds for every value of the symbols; for `q`
/// of unknown sign it is generic, as SymPy's, and fails (`0/0`) only where
/// `p = 0` and `q < 0`.
///
/// Quartic: SymPy's `roots_quartic` — `x·(cubic)`, quasi-symmetric (`(c/a)²
/// = d`), biquadratic and `y·(cubic)` after depressing; Descartes–Euler
/// when the resolvent has a root rational in the symbols
/// ([`resolvent_rational_root`], square roots only); otherwise Ferrari
/// ([`ferrari_roots`]).  Ferrari's roots `(±w − t·√(−(3e + 2y ± 2f/w)))/2
/// − a/4` take square roots of quantities that are exactly negative real
/// numbers wherever the quartic has a pair of complex-conjugate roots at
/// real values of the symbols (`(r − r̄)²`), whose imaginary parts cancel
/// only exactly (the resolvent root `y` comes from Cardano's formula with
/// complex cube roots).  Up to 0.44 they were refused, as they would not
/// evaluate there; since `eval` decides the realness of such constants
/// exactly (`x − x̄ = 0` in `ℚ(ζ)[radicals]`), `√` of a negative one is
/// `i·√(−·)` after a substitution and the roots evaluate.  The four roots
/// are checked at a few rational points of the symbols
/// ([`quartic_roots_verified`]).
///
/// The square roots of the radicands are any square roots (the formulas
/// hold for both signs as long as `w ≠ 0`), so perfect squares are pulled
/// out ([`quadratic_sqrt`]).  `None` when the leading coefficient is not
/// provably nonzero or the degree is not 3 or 4.
pub(super) fn parametric_radical_roots(coeffs: &[Ex]) -> Option<Vec<Ex>> {
    let deg = coeffs.len().checked_sub(1)?;
    if !(3..=4).contains(&deg) || !provably_nonzero(&coeffs[deg]) {
        return None;
    }
    let lead = &coeffs[deg];
    let monic: Vec<Ex> = coeffs[..deg]
        .iter()
        .map(|c| parametric_normal(&(c / lead)))
        .collect();
    if deg == 3 {
        parametric_cubic_roots(&monic[2], &monic[1], &monic[0])
    } else {
        parametric_quartic_roots(&monic[3], &monic[2], &monic[1], &monic[0])
    }
}

/// The cube roots of unity `1, −1/2 + √3·i/2, −1/2 − √3·i/2`.
fn cube_roots_of_unity(ctx: &Context) -> [Ex; 3] {
    let coeff = &(&ctx.int(3).sqrt() * &ctx.i_unit()) / &ctx.int(2);
    let half = ctx.rational(-1, 2);
    [ctx.one(), &half + &coeff, &half - &coeff]
}

/// The two (or one double) roots of `x² + b·x + c` by [`low_degree_roots`],
/// repeated by multiplicity.
fn monic_quadratic_roots(c: &Ex, b: &Ex) -> Vec<Ex> {
    let ctx = c.context();
    low_degree_roots(&[c.clone(), b.clone(), ctx.one()])
        .into_iter()
        .flat_map(|(v, m)| std::iter::repeat_n(v, m))
        .collect()
}

/// [`parametric_radical_roots`] of `x³ + a·x² + b·x + c`.
fn parametric_cubic_roots(a: &Ex, b: &Ex, c: &Ex) -> Option<Vec<Ex>> {
    let ctx = a.context();
    let zero = |e: &Ex| ex_is_zero(e).is_true();
    if zero(c) {
        // x·(x² + a·x + b)
        let mut r = monic_quadratic_roots(b, a);
        if r.len() != 2 {
            return None;
        }
        r.insert(1, ctx.zero());
        return Some(r);
    }
    let q3 = |n: i64, d: i64| ctx.rational(n, d);
    let p = parametric_normal(&(b - &(&a.powi(2) * &q3(1, 3))));
    let q = parametric_normal(&(&(c - &(&(a * b) * &q3(1, 3))) + &(&a.powi(3) * &q3(2, 27))));
    let aon3 = parametric_normal(&(a * &q3(1, 3)));
    let omega = cube_roots_of_unity(&ctx);
    if zero(&p) {
        if zero(&q) {
            return Some(vec![(-&aon3).eval(); 3]);
        }
        // y³ = −q: any cube root of −q times the cube roots of unity.
        let w = if Truth::from(q.is_positive()).is_true() {
            -q.cbrt()
        } else {
            (-&q).eval().cbrt()
        };
        return Some(omega.iter().map(|u| (&(u * &w) - &aon3).eval()).collect());
    }
    if zero(&q) {
        // y·(y² + p)
        let s = quadratic_sqrt(&parametric_normal(&-&p));
        return Some(vec![
            (&s - &aon3).eval(),
            (-&aon3).eval(),
            (&(-&s) - &aon3).eval(),
        ]);
    }
    let r = parametric_normal(&(&(&q.powi(2) * &q3(1, 4)) + &(&p.powi(3) * &q3(1, 27))));
    let s = quadratic_sqrt(&r);
    let half_q = &q * &q3(1, 2);
    let w = if Truth::from(q.is_negative()).is_true() {
        (&s - &half_q).cbrt()
    } else {
        -(&half_q + &s).cbrt()
    };
    let pw = &p / &(&w * &ctx.int(3));
    Some(
        (0..3)
            .map(|k| {
                let conj = &omega[(3 - k) % 3];
                (&(&(&omega[k] * &w) - &(conj * &pw)) - &aon3).eval()
            })
            .collect(),
    )
}

/// [`parametric_radical_roots`] of `x⁴ + a·x³ + b·x² + c·x + d`.
fn parametric_quartic_roots(a: &Ex, b: &Ex, c: &Ex, d: &Ex) -> Option<Vec<Ex>> {
    let ctx = a.context();
    let zero = |e: &Ex| ex_is_zero(e).is_true();
    let qn = |n: i64, dd: i64| ctx.rational(n, dd);
    if zero(d) {
        // x·(x³ + a·x² + b·x + c)
        let mut r = parametric_cubic_roots(a, b, c)?;
        r.insert(0, ctx.zero());
        return Some(r);
    }
    if !zero(a) {
        // Quasi-symmetric: (c/a)² = d.  z = x + m/x with m = c/a solves
        // z² + a·z + b − 2m = 0, then x² − z·x + m = 0.
        let m = parametric_normal(&(c / a));
        if zero(&parametric_normal(&(&m.powi(2) - d))) {
            let zs = monic_quadratic_roots(&parametric_normal(&(b - &(&m * &ctx.int(2)))), a);
            if zs.len() != 2 {
                return None;
            }
            let mut out = Vec::with_capacity(4);
            for z in &zs {
                let r = monic_quadratic_roots(&m, &(-z).eval());
                if r.len() != 2 {
                    return None;
                }
                out.extend(r);
            }
            return Some(out);
        }
    }
    // Depressed y⁴ + e·y² + f·y + g with x = y − a/4.
    let a2 = a.powi(2);
    let e = parametric_normal(&(b - &(&a2 * &qn(3, 8))));
    let f = parametric_normal(&(c + &(a * &(&(&a2 * &qn(1, 8)) - &(b * &qn(1, 2))))));
    let aon4 = parametric_normal(&(a * &qn(1, 4)));
    let g = parametric_normal(
        &(d - &(&aon4 * &(&(a * &(&(&a2 * &qn(3, 64)) - &(b * &qn(1, 4)))) + c))),
    );
    let shift = |y: Ex| (&y - &aon4).eval();
    if zero(&f) {
        // Biquadratic: y² = t for the roots t of t² + e·t + g.
        let ts = monic_quadratic_roots(&g, &e);
        if ts.len() != 2 {
            return None;
        }
        let (y1, y2) = (ts[0].sqrt(), ts[1].sqrt());
        return Some(vec![shift(-&y1), shift(-&y2), shift(y1), shift(y2)]);
    }
    if zero(&g) {
        // y·(y³ + e·y + f)
        let mut ys = parametric_cubic_roots(&ctx.zero(), &e, &f)?;
        ys.insert(0, ctx.zero());
        return Some(ys.into_iter().map(shift).collect());
    }
    let roots = match resolvent_rational_root(&e, &f, &g) {
        // Descartes–Euler (SymPy's `_roots_quartic_euler`): `±√R ∓ √(A ±
        // B)` with `A = −R − e/2`, `B = −f·√R/(4R)` — square roots only.
        // Either sign of `√R` swaps the two pairs, and both signs of the
        // other roots are taken: the set is the same for every branch.
        Some(r) => {
            let c1 = quadratic_sqrt(&r);
            let bb = &(&-&f * &c1) / &(&r * &ctx.int(4));
            let aa = parametric_normal(&(&-&r - &(&e * &qn(1, 2))));
            let c2 = (&aa + &bb).sqrt();
            let c3 = (&aa - &bb).sqrt();
            vec![
                shift(&c1 - &c2),
                shift(&-&c1 - &c3),
                shift(&-&c1 + &c3),
                shift(&c1 + &c2),
            ]
        }
        None => ferrari_roots(&e, &f, &g).into_iter().map(shift).collect(),
    };
    quartic_roots_verified(&[d.clone(), c.clone(), b.clone(), a.clone()], &roots).then_some(roots)
}

/// A root `R ≠ 0` of the Descartes–Euler resolvent `64R³ + 32e·R² + (4e²
/// − 16g)·R − f²` of `y⁴ + e·y² + f·y + g` that is a rational function of
/// the symbols (a linear factor over `ℚ(symbols)`), in normal form; `None`
/// when it has none.  SymPy takes only rational roots, so a symbolic
/// quartic goes to Ferrari there.
fn resolvent_rational_root(e: &Ex, f: &Ex, g: &Ex) -> Option<Ex> {
    let ctx = e.context();
    let r = ctx.symbol("__descartes_euler_r");
    let poly = &(&(&(&r.powi(3) * &ctx.int(64)) + &(&(e * &r.powi(2)) * &ctx.int(32)))
        + &(&(&(&e.powi(2) * &ctx.int(4)) - &(g * &ctx.int(16))) * &r))
        - &f.powi(2);
    let (_, factors) = poly.expand().factor_list_all();
    factors.iter().find_map(|(h, _)| {
        let cs = h.coeffs(&r)?;
        if cs.len() != 2 || !provably_nonzero(&cs[1]) {
            return None;
        }
        let root = parametric_normal(&(-&(&cs[0] / &cs[1])));
        provably_nonzero(&root).then_some(root)
    })
}

/// The roots of the depressed `y⁴ + e·y² + f·y + g` by Ferrari's method as
/// SymPy's `roots_quartic` writes it: a root `y₀ = −5e/6 + z` of the
/// resolvent (`z³ + p·z + q = 0`, `p = −e²/12 − g`, `q = −e³/108 + e·g/3
/// − f²/8`, by Cardano's formula as in [`parametric_cubic_roots`]), `w =
/// √(e + 2y₀)`, and the roots `(s·w − t·√(−(3e + 2y₀ + 2s·f/w)))/2` for
/// `s, t = ±1`.  The formulas hold for every choice of the roots as long as
/// `w ≠ 0` (true wherever `f ≠ 0`: `w` is the sum of two of the roots, and
/// a pairing summing to 0 makes the quartic biquadratic) and the cube root
/// is not 0: generic, as SymPy's, failing (`0/0`) only where `f = 0` or, for
/// `q` of unknown sign, where `p = 0` and `q > 0`.
fn ferrari_roots(e: &Ex, f: &Ex, g: &Ex) -> Vec<Ex> {
    let ctx = e.context();
    let qn = |n: i64, d: i64| ctx.rational(n, d);
    let zero = |x: &Ex| ex_is_zero(x).is_true();
    let ee = e.powi(2);
    let p = parametric_normal(&(&(&ee * &qn(-1, 12)) - g));
    let q = parametric_normal(
        &(&(&(&ee * e) * &qn(-1, 108)) + &(&(&(e * g) * &qn(1, 3)) - &(&f.powi(2) * &qn(1, 8)))),
    );
    let five_e_6 = parametric_normal(&(e * &qn(-5, 6)));
    let z = if zero(&p) {
        // z³ = −q
        if Truth::from(q.is_positive()).is_true() {
            -q.cbrt()
        } else {
            (-&q).eval().cbrt()
        }
    } else {
        let rad = parametric_normal(&(&(&q.powi(2) * &qn(1, 4)) + &(&p.powi(3) * &qn(1, 27))));
        let s = quadratic_sqrt(&rad);
        let half_q = &q * &qn(1, 2);
        let u = if Truth::from(q.is_negative()).is_true() {
            (&s - &half_q).cbrt()
        } else {
            -(&half_q + &s).cbrt()
        };
        &u - &(&p / &(&u * &ctx.int(3)))
    };
    let y = &five_e_6 + &z;
    let two = ctx.int(2);
    let w = (e + &(&y * &two)).sqrt();
    let arg1 = &(e * &ctx.int(3)) + &(&y * &two);
    let arg2 = &(f * &two) / &w;
    let mut out = Vec::with_capacity(4);
    for s in [-1i64, 1] {
        let sw = &w * &ctx.int(s);
        let root = (-(&arg1 + &(&arg2 * &ctx.int(s)))).sqrt();
        for t in [-1i64, 1] {
            out.push(&(&sw - &(&root * &ctx.int(t))) / &two);
        }
    }
    out
}

/// Are `roots` the four roots of the monic `x⁴ + a·x³ + b·x² + c·x + d`
/// (`coeffs = [d, c, b, a]`, symbolic) at one of a few rational points of
/// its symbols: certified complex values, each a root to 10⁻⁸ of the scale
/// of the polynomial there, and pairwise apart (so that together they are
/// all four roots)?  A point where a value does not evaluate (a radicand
/// on its branch cut that no realness proof settles) or the roots meet is
/// skipped.  A guard against a wrong formula, not a proof for every point.
fn quartic_roots_verified(coeffs: &[Ex], roots: &[Ex]) -> bool {
    let ctx = roots[0].context();
    let syms = {
        let mut all: Vec<Ex> = Vec::new();
        for e in coeffs.iter().chain(roots) {
            for s in e.free_symbols() {
                if !all.contains(&s) {
                    all.push(s);
                }
            }
        }
        all
    };
    (0..3i64).any(|k| {
        let vals: Vec<Ex> = (0..syms.len() as i64)
            .map(|i| ctx.rational(31 + 17 * i + 7 * k, 11 + 4 * k + 3 * i))
            .collect();
        let subs: Vec<(&Ex, &Ex)> = syms.iter().zip(vals.iter()).collect();
        let Ok(c) = coeffs
            .iter()
            .map(|e| e.subs_map(&subs).eval_complex64())
            .collect::<Result<Vec<_>, _>>()
        else {
            return false;
        };
        let Ok(z) = roots
            .iter()
            .map(|e| e.subs_map(&subs).eval_complex64())
            .collect::<Result<Vec<_>, _>>()
        else {
            return false;
        };
        let scale = c.iter().map(|x| x.norm()).fold(1.0f64, f64::max);
        let radius = 1.0 + scale;
        let at = |x: num_complex::Complex64| (((x + c[3]) * x + c[2]) * x + c[1]) * x + c[0];
        let roots_ok = z
            .iter()
            .all(|&x| at(x).norm() <= 1e-8 * scale * radius.powi(4).max(1.0));
        let apart = (0..4).all(|i| ((i + 1)..4).all(|j| (z[i] - z[j]).norm() > 1e-6 * radius));
        roots_ok && apart
    })
}
// ═══════════════════════════════════════════════════════════════════════════
// The spectrum: complete, or said not to be
// ═══════════════════════════════════════════════════════════════════════════

/// The eigenvalues found (the roots found of a polynomial), each with its
/// algebraic multiplicity, and whether they are all of them.
///
/// A list of roots is complete or says it is not.  Up to 0.59 the roots of
/// a factor of the characteristic polynomial that was not solved were
/// dropped with a log line, and the short list travelled as a plain `Vec`:
/// `eigenvals` of a 6×6 matrix with the characteristic polynomial `(λ −
/// 1)·(λ⁵ − a·λ − 1)` was `Ok([1])`, `eigenvects` one vector, and
/// `diagonalize` reported `NotDiagonalizable` (B-C11-01).
///
/// The type has no `len`, no iterator and no conversion to a list: a
/// consumer takes [`all`](Self::all) (every eigenvalue, or the error), or
/// matches the variants and says why a part is enough for it.  It is built
/// only here, by [`of`](Self::of), which checks the multiplicities against
/// the degree; `tests/unit/test_matrix_linalg.rs`
/// (`partial_root_lists_are_not_read_as_complete`) counts the places that
/// name a variant outside this file.
#[must_use]
#[derive(Clone, Debug)]
pub(crate) enum Spectrum {
    /// Every eigenvalue: the multiplicities sum to `n`.
    Complete(Vec<(Ex, usize)>),
    /// Some eigenvalues are not known: `missing` of them, counted with
    /// multiplicity, are not in `found` (`found` may be empty; `missing`
    /// is at least 1).
    Partial {
        /// The eigenvalues that were found, each an eigenvalue with at
        /// least the multiplicity listed.
        found: Vec<(Ex, usize)>,
        /// How many eigenvalues, with multiplicity, are missing.
        missing: usize,
    },
}

impl Spectrum {
    /// The spectrum of a polynomial of degree `n` (an `n × n` matrix) from
    /// the roots found for it: complete exactly when their multiplicities
    /// sum to `n`.  Multiplicities beyond `n` are a contradiction (one root
    /// listed in two forms, or a multiplicity too high), so none of them is
    /// claimed.
    pub(super) fn of(pairs: Vec<(Ex, usize)>, n: usize) -> Spectrum {
        let total: usize = pairs.iter().map(|(_, m)| *m).sum();
        match total.cmp(&n) {
            std::cmp::Ordering::Equal => Spectrum::Complete(pairs),
            std::cmp::Ordering::Less => Spectrum::Partial {
                found: pairs,
                missing: n - total,
            },
            std::cmp::Ordering::Greater => {
                debug!(
                    total,
                    n, "Spectrum::of: more roots than the degree — none is claimed"
                );
                Spectrum::Partial {
                    found: Vec::new(),
                    missing: n,
                }
            }
        }
    }

    /// Are these all the eigenvalues?
    pub(crate) fn is_complete(&self) -> bool {
        matches!(self, Spectrum::Complete(_))
    }

    /// The eigenvalues found and the number missing.
    fn into_parts(self) -> (Vec<(Ex, usize)>, usize) {
        match self {
            Spectrum::Complete(pairs) => (pairs, 0),
            Spectrum::Partial { found, missing } => (found, missing),
        }
    }

    /// Every eigenvalue with its multiplicity, or the error that says how
    /// many are missing and names the ones found.
    ///
    /// # Errors
    ///
    /// [`SymplexError::ComputationFailed`] for a partial spectrum.
    pub(crate) fn all(self, operation: &'static str) -> Result<Vec<(Ex, usize)>, SymplexError> {
        match self {
            Spectrum::Complete(pairs) => Ok(pairs),
            Spectrum::Partial { found, missing } => {
                Err(incomplete_spectrum(operation, &found, missing))
            }
        }
    }

    /// The same spectrum with every eigenvalue rewritten by `f` (a
    /// renaming: the values do not change).
    pub(super) fn map(self, f: impl Fn(&Ex) -> Ex) -> Spectrum {
        let rename = |pairs: Vec<(Ex, usize)>| pairs.into_iter().map(|(v, m)| (f(&v), m)).collect();
        match self {
            Spectrum::Complete(pairs) => Spectrum::Complete(rename(pairs)),
            Spectrum::Partial { found, missing } => Spectrum::Partial {
                found: rename(found),
                missing,
            },
        }
    }

    /// The spectrum of `pᵐ` from that of `p`.
    fn scaled(self, m: usize) -> Spectrum {
        let (found, missing) = self.into_parts();
        let found: Vec<(Ex, usize)> = found.into_iter().map(|(v, k)| (v, k * m)).collect();
        match missing {
            0 => Spectrum::Complete(found),
            _ => Spectrum::Partial {
                found,
                missing: missing * m,
            },
        }
    }

    /// The spectrum of a product `p·q` (of a block-triangular matrix) from
    /// those of `p` and `q` (of its diagonal blocks), `self` first: complete
    /// when both are.  A root of `q` that `same` finds among those of `p`
    /// adds its multiplicity there.
    pub(super) fn joined(self, other: Spectrum, same: impl Fn(&Ex, &Ex) -> bool) -> Spectrum {
        let (mut found, missing) = self.into_parts();
        let (more, more_missing) = other.into_parts();
        for (v, m) in more {
            match found.iter_mut().find(|(w, _)| same(w, &v)) {
                Some((_, k)) => *k += m,
                None => found.push((v, m)),
            }
        }
        match missing + more_missing {
            0 => Spectrum::Complete(found),
            missing => Spectrum::Partial { found, missing },
        }
    }

    /// The same spectrum checked against the size `n` of the matrix it was
    /// computed for ([`of`](Self::of) on the eigenvalues found): whatever
    /// the steps before reported, it is complete only when the
    /// multiplicities sum to `n`.
    pub(super) fn checked_for(self, n: usize) -> Spectrum {
        Spectrum::of(self.into_parts().0, n)
    }
}

/// The error of an incomplete spectrum: how many eigenvalues are missing,
/// and the ones found.
fn incomplete_spectrum(
    operation: &'static str,
    found: &[(Ex, usize)],
    missing: usize,
) -> SymplexError {
    const WHY: &str = "it has an irreducible factor with symbolic coefficients of degree above \
                       4 (or a quartic one whose radical roots did not check at sample values \
                       of the symbols)";
    if found.is_empty() {
        return SymplexError::computation_failed(
            operation,
            format!("could not solve the characteristic polynomial: {WHY}"),
        );
    }
    // The eigenvalues found, shortened: a root in radicals can fill a page.
    const SHOWN: usize = 6;
    const WIDTH: usize = 60;
    let mut names: Vec<String> = found
        .iter()
        .take(SHOWN)
        .map(|(v, m)| {
            let text = v.to_string();
            let text = if text.chars().count() > WIDTH {
                let head: String = text.chars().take(WIDTH).collect();
                format!("{head}…")
            } else {
                text
            };
            if *m == 1 {
                text
            } else {
                format!("{text} (×{m})")
            }
        })
        .collect();
    if found.len() > SHOWN {
        names.push("…".into());
    }
    let known: usize = found.iter().map(|(_, m)| *m).sum();
    SymplexError::computation_failed(
        operation,
        format!(
            "could not solve the characteristic polynomial: found {known} of its {} roots \
             ({}), the other {missing} are not known — {WHY}, or a root whose multiplicity \
             was not established",
            known + missing,
            names.join(", "),
        ),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Eigenvalue multiplicity via polynomial factoring
// ═══════════════════════════════════════════════════════════════════════════

/// Extract eigenvalues with algebraic multiplicities from a characteristic
/// polynomial of degree `degree` in `var`.
///
/// **Primary path:** convert to `Poly`, call `factor_over_z()` (Yun's
/// square-free decomposition) for exact `(factor, multiplicity)` pairs;
/// each factor is solved for its roots, which inherit the multiplicity.
///
/// **Fallback:** if the polynomial has non-rational coefficients,
/// `expr_to_poly` returns `None`; then derivative-based multiplicity
/// detection is used on whatever roots the generic solver finds.
///
/// The result says whether the roots are all of them ([`Spectrum`]).
pub(super) fn eigvals_with_multiplicity(char_poly: &Ex, var: &Ex, degree: usize) -> Spectrum {
    let exact = match eigvals_via_poly_factor(char_poly, var).map(|s| s.checked_for(degree)) {
        Some(spectrum) if spectrum.is_complete() => {
            debug!("eigvals_with_multiplicity: used Poly::factor_over_z path");
            return spectrum;
        }
        other => other,
    };
    debug!("eigvals_with_multiplicity: falling back to derivative-based detection");
    let fallback = eigvals_via_derivative(char_poly, var, degree);
    match exact {
        // Neither is complete: the factors solved exactly are the better part.
        Some(Spectrum::Partial { found, missing })
            if !fallback.is_complete() && !found.is_empty() =>
        {
            Spectrum::Partial { found, missing }
        }
        _ => fallback,
    }
}

/// Primary multiplicity path: convert to Poly, factor, solve each factor.
///
/// Every irreducible factor over ℚ has an exact root list: a rational, the
/// quadratic formula, radicals when they are compact, and `RootOf(factor,
/// k)` otherwise — also when the solver does not return as many roots as
/// the degree (an irreducible polynomial has that many distinct ones).  Up
/// to 0.59 the roots of such a factor were dropped.
fn eigvals_via_poly_factor(char_poly: &Ex, var: &Ex) -> Option<Spectrum> {
    let inner = char_poly.inner.read();
    let arena = &inner.arena;
    let poly = crate::poly::polybridge::expr_to_poly(arena, char_poly.raw_id(), var.raw_id())?;
    let (_content, factors) = poly.factor_over_z();
    if factors.is_empty() {
        return None;
    }
    drop(inner);

    let mut eigen_pairs: Vec<(Ex, usize)> = Vec::new();
    let mut total_degree = 0usize;
    for (factor, mult) in &factors {
        let degree = factor.degree().unwrap_or(0);
        if degree == 0 {
            continue;
        }
        total_degree += degree * *mult as usize;
        if degree == 1 {
            let coeffs = factor.coeffs();
            let a = &coeffs[1];
            let b = &coeffs[0];
            let root_val = -(b / a);
            let root_numer: Result<i64, _> = root_val.numer().clone().try_into();
            let root_denom: Result<i64, _> = root_val.denom().clone().try_into();
            if let (Ok(n), Ok(d)) = (root_numer, root_denom) {
                eigen_pairs.push((char_poly.context().rational(n, d), *mult as usize));
                continue;
            }
        }
        // Higher degree (or huge rational root): solve the factor.
        let mut write_inner = char_poly.inner.write();
        let factor_expr =
            crate::poly::polybridge::poly_to_expr(&mut write_inner.arena, factor, var.raw_id());
        drop(write_inner);
        let factor_ex = char_poly.wrap(factor_expr);
        let root_ofs = || -> Vec<Ex> { (0..degree).map(|i| root_of(&factor_ex, var, i)).collect() };
        let roots = if (3..=4).contains(&degree) && !radical_form_is_compact(factor) {
            // Irreducible cubic/quartic without a compact radical form: the
            // Cardano/Ferrari expressions (nested complex cube roots) make
            // every downstream step — eigenvectors, P⁻¹, simplify — swell
            // exponentially.  `RootOf` is exact, small and numerically
            // evaluable, so it is the better closed form here.
            debug!(
                degree,
                "eigvals_via_poly_factor: irreducible factor without compact radicals → RootOf"
            );
            root_ofs()
        } else {
            // The factor is irreducible over ℚ: it has `degree` distinct
            // roots, and a solve that fails or lists another number of
            // them has not listed its roots.
            match factor_ex.solve(var) {
                Ok(roots) if roots.len() == degree => roots,
                unsolved => {
                    debug!(
                        degree,
                        solved = unsolved.is_ok(),
                        "eigvals_via_poly_factor: the solver did not list the roots of a factor"
                    );
                    match degree {
                        1 => {
                            let c = factor.coeffs();
                            vec![char_poly.context().from_ratio(-(&c[0] / &c[1]))]
                        }
                        2 => {
                            let ctx = char_poly.context();
                            let c: Vec<Ex> = factor
                                .coeffs()
                                .iter()
                                .map(|q| ctx.from_ratio(q.clone()))
                                .collect();
                            low_degree_roots(&c).into_iter().map(|(r, _)| r).collect()
                        }
                        _ => root_ofs(),
                    }
                }
            }
        };
        trace!(
            degree,
            root_count = roots.len(),
            mult,
            "eigvals_via_poly_factor: degree-{} factor, {} roots, mult {}",
            degree,
            roots.len(),
            mult
        );
        if roots.len() != degree {
            // Not the root list of this factor: none of it is claimed.
            continue;
        }
        for r in roots {
            eigen_pairs.push((r, *mult as usize));
        }
    }
    Some(Spectrum::of(eigen_pairs, total_degree))
}

/// Eigenvalues of a characteristic polynomial with symbolic coefficients
/// that factors over `ℚ[symbols]`: each factor is solved on its own (the
/// linear and quadratic formulas of [`low_degree_roots`], the generic path
/// above that) and its roots inherit its multiplicity, as SymPy's `roots`
/// does with `factor_list`.  `None` when the polynomial does not split.
///
/// Before 0.38 the whole polynomial went to the generic solver: the
/// rank-one 4×4 matrix `u·vᵀ` with `u = (x, y, x, 2)`, `v = (3, y, 3, 3)`,
/// whose characteristic polynomial is `λ³·(λ − y² − 6x − 6)`, was refused
/// ("could not solve the characteristic polynomial"; SymPy: `{0: 3,
/// y² + 6x + 6: 1}`), and a reducible quadratic factor could come back
/// from the quadratic formula with its roots in disguise.
///
/// The result is the [`Spectrum`] of the product: a factor whose roots are
/// not all found (an irreducible one of degree above 4 with symbolic
/// coefficients) leaves it partial.  Up to 0.59 its roots were left out of
/// a plain list: `(λ − 1)·(λ⁵ − a·λ − 1)` had the roots `[1]`.
pub(super) fn symbolic_factor_roots(cp: &Ex, lam: &Ex) -> Option<Spectrum> {
    let (_, factors) = cp.factor_list_all();
    if factors.len() < 2 && factors.iter().all(|(_, m)| *m < 2) {
        return None;
    }
    let mut out = Spectrum::of(Vec::new(), 0);
    for (f, m) in &factors {
        let coeffs = f.coeffs(lam)?;
        let Some(degree) = coeffs.len().checked_sub(1).filter(|d| *d > 0) else {
            continue;
        };
        let parametric = coeffs.iter().any(|c| !c.free_symbols().is_empty());
        let in_radicals =
            |rs: Vec<Ex>| Spectrum::of(rs.into_iter().map(|r| (r, 1)).collect(), degree);
        let roots = match degree {
            1 | 2 => Spectrum::of(low_degree_roots(&coeffs), degree),
            // An irreducible symbolic cubic or quartic factor: in radicals
            // (before 0.44 its roots were missing from the result).
            3 | 4 if parametric => match parametric_radical_roots(&coeffs) {
                Some(rs) => in_radicals(rs),
                None => eigvals_with_multiplicity(&f.expand(), lam, degree),
            },
            _ => eigvals_with_multiplicity(&f.expand(), lam, degree),
        };
        out = out.joined(roots.scaled(*m as usize), |_, _| false);
    }
    Some(out)
}

/// [`parametric_radical_roots`] of the characteristic polynomial `cp` in
/// `lam` (ascending coefficients `coeffs`) when it is square-free: its
/// discriminant is provably not the zero function (the roots are then
/// distinct for generic values of the symbols, each of multiplicity 1).
pub(super) fn squarefree_parametric_roots(cp: &Ex, coeffs: &[Ex], lam: &Ex) -> Option<Vec<Ex>> {
    let disc = cp.discriminant_symbolic(lam)?;
    if !provably_nonzero(&parametric_normal(&disc)) {
        return None;
    }
    parametric_radical_roots(coeffs)
}

/// `RootOf(poly, var, index)` — the `index`-th root (Aberth ordering: by
/// real part, then imaginary part) of the polynomial `poly` in `var`.
fn root_of(poly: &Ex, var: &Ex, index: usize) -> Ex {
    let mut inner = poly.inner.write();
    let idx = inner.arena.int(index as i64);
    let id = inner.arena.intern(crate::base::node::ExprNode::RootOf(
        poly.raw_id(),
        var.raw_id(),
        idx,
    ));
    drop(inner);
    poly.wrap(id)
}

/// Does an irreducible cubic or quartic over ℤ have a *compact* radical
/// form worth returning instead of `RootOf`?
///
/// After the Tschirnhaus shift `x = y − b/(n·a)` the polynomial is
/// "depressed"; if the remaining odd-degree coefficient vanishes the
/// roots are a rational shift plus a single cube root (cubic `y³ + q`)
/// or nested square roots (biquadratic `y⁴ + p·y² + r`).  Everything else
/// needs the full Cardano/Ferrari formulas, whose casus-irreducibilis
/// complex cube roots are enormous and unsimplifiable.
fn radical_form_is_compact(f: &crate::poly::dense::Poly) -> bool {
    use num_bigint::BigInt;
    let Some(n) = f.degree() else { return true };
    let c = |i: usize| f.coeff(i);
    let k = |v: i64| num_rational::Ratio::from_integer(BigInt::from(v));
    match n {
        // y³ + p y + q with p = (3ac − b²)/(3a²): compact iff p = 0.
        3 => &(&c(3) * &c(1)) * &k(3) == &c(2) * &c(2),
        // Linear coefficient of the depressed quartic ∝ b³ − 4abc + 8a²d.
        4 => {
            let (a, b, cc, d) = (c(4), c(3), c(2), c(1));
            let b3 = &(&b * &b) * &b;
            let abc = &(&(&a * &b) * &cc) * &k(4);
            let aad = &(&(&a * &a) * &d) * &k(8);
            &(&b3 - &abc) + &aad == k(0)
        }
        _ => true,
    }
}

/// Fallback: derivative-based multiplicity detection.
///
/// For each root `r`, finds the smallest `k` such that `p^(k)(r) ≠ 0`.
///
/// The roots are those the solver lists for the whole polynomial; when it
/// refuses (a factor it cannot solve, roots it found "but not necessarily
/// all") none is known.  A multiplicity is counted only as far as the
/// derivatives vanish structurally, so it is never too high, and the
/// [`Spectrum`] is complete only when the multiplicities reach `degree`.
fn eigvals_via_derivative(char_poly: &Ex, var: &Ex, degree: usize) -> Spectrum {
    let Ok(all_roots) = char_poly.solve(var) else {
        return Spectrum::of(Vec::new(), degree);
    };
    if all_roots.is_empty() {
        return Spectrum::of(Vec::new(), degree);
    }
    let mut unique_roots: Vec<Ex> = Vec::new();
    for root in &all_roots {
        if !unique_roots.iter().any(|r| r == root) {
            unique_roots.push(root.clone());
        }
    }

    let mut eigen_pairs: Vec<(Ex, usize)> = Vec::new();
    for root in &unique_roots {
        let mut mult = 0usize;
        let mut current = char_poly.clone();
        for k in 0..20 {
            let val = current.subs(var, root).eval().simplify();
            if !val.is_zero_structural() {
                mult = k;
                break;
            }
            if k < 19 {
                current = current.diff(var);
            }
        }
        let mult = if mult == 0 { 1 } else { mult };
        eigen_pairs.push((root.clone(), mult));
    }
    Spectrum::of(eigen_pairs, degree)
}

// ═══════════════════════════════════════════════════════════════════════════
// Jordan form helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Zero test used inside the eigen-family (pivot selection in
/// `A − λI`): structural, then the simplifying three-valued test, then —
/// for constant expressions such as nested radicals or `RootOf` values
/// that `simplify` cannot collapse — numeric evaluation with a tight
/// tolerance.
///
/// The numeric fallback is sound here because `λ` is an exact eigenvalue:
/// `A − λI` *is* singular, so a pivot candidate that evaluates to ~1e-15
/// is a radical expression for zero, not a genuinely tiny constant.
/// The zero decisions of [`Matrix::rref_core`], told each row operation as
/// it is applied to the expressions (so a shadow copy can follow it).
pub(super) trait PivotOracle {
    /// Is the current entry `(i, j)`, whose expression is `e`, zero?
    /// Three-valued, in the reading of [`pivot_truth`]: `True` is a proof
    /// that it is, `False` a proof that it is not, `Unknown` neither.
    fn is_zero(&mut self, i: usize, j: usize, e: &Ex) -> Truth;
    /// Rows `a` and `b` were swapped.
    fn swap(&mut self, _a: usize, _b: usize) {}
    /// Row `row` was divided by its entry in column `col`.
    fn normalize(&mut self, _row: usize, _col: usize) {}
    /// Row `row` had `entry(row, col)` times row `pivot_row` subtracted.
    fn eliminate(&mut self, _row: usize, _pivot_row: usize, _col: usize) {}
    /// Replace the (zero) entries of the rows below the rank by `0`?
    fn clean_zero_rows(&self) -> bool {
        false
    }
}

/// A zero test on the expression alone; `clean` asks for the zero rows of
/// the result to be written as zeros (the public `rref`).  The verdicts are
/// remembered per expression: one elimination asks about the same entry
/// when it looks for a pivot and when it clears the column.
pub(super) struct ClosureOracle<'a> {
    pub(super) test: &'a dyn Fn(&Ex) -> Truth,
    pub(super) clean: bool,
    pub(super) seen: rustc_hash::FxHashMap<crate::base::node::ExprId, Truth>,
}

impl<'a> ClosureOracle<'a> {
    pub(super) fn new(test: &'a dyn Fn(&Ex) -> Truth, clean: bool) -> Self {
        ClosureOracle {
            test,
            clean,
            seen: rustc_hash::FxHashMap::default(),
        }
    }
}

impl PivotOracle for ClosureOracle<'_> {
    fn is_zero(&mut self, _i: usize, _j: usize, e: &Ex) -> Truth {
        if e.is_zero_structural() {
            return Truth::True;
        }
        if let Some(t) = self.seen.get(&e.raw_id()) {
            return *t;
        }
        let t = (self.test)(e);
        self.seen.insert(e.raw_id(), t);
        t
    }
    fn clean_zero_rows(&self) -> bool {
        self.clean
    }
}

/// `A − tI` over `ℚ[t]/(g)` (`g` irreducible), eliminated in lockstep with
/// the expressions of `A − λI` for a root `λ` of `g`: the two are the same
/// numbers, so the residue decides each zero test exactly.
pub(super) struct FieldShadow {
    pub(super) rows: Vec<Vec<crate::poly::dense::Poly>>,
    g: crate::poly::dense::Poly,
}

impl FieldShadow {
    pub(super) fn new(a: &QMatrix, g: &crate::poly::dense::Poly) -> Self {
        use crate::poly::dense::Poly;
        let rows = a
            .to_rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                row.into_iter()
                    .enumerate()
                    .map(|(j, c)| {
                        let c = Poly::constant(c);
                        if i == j { c.sub(&Poly::x()) } else { c }
                    })
                    .collect()
            })
            .collect();
        FieldShadow { rows, g: g.clone() }
    }

    /// Inverse of a nonzero residue (`g` irreducible, so the extended gcd
    /// is a nonzero constant).
    fn inverse(&self, a: &crate::poly::dense::Poly) -> crate::poly::dense::Poly {
        use crate::poly::dense::Poly;
        let e = Poly::extended_gcd(a, &self.g);
        match e.gcd.leading_coeff() {
            Some(c) if e.gcd.degree() == Some(0) => {
                e.x.scale(&(Q::from_integer(BigInt::from(1)) / c))
                    .rem(&self.g)
            }
            // Unreachable for a nonzero residue and an irreducible `g`;
            // a zero residue is never a pivot.
            _ => Poly::zero(),
        }
    }
}

impl PivotOracle for FieldShadow {
    fn is_zero(&mut self, i: usize, j: usize, _e: &Ex) -> Truth {
        Truth::from_bool(self.rows[i][j].is_zero())
    }
    fn swap(&mut self, a: usize, b: usize) {
        self.rows.swap(a, b);
    }
    fn normalize(&mut self, row: usize, col: usize) {
        let inv = self.inverse(&self.rows[row][col]);
        let g = &self.g;
        for c in &mut self.rows[row] {
            *c = c.mul(&inv).rem(g);
        }
    }
    fn eliminate(&mut self, row: usize, pivot_row: usize, col: usize) {
        let f = self.rows[row][col].clone();
        let pivot = self.rows[pivot_row].clone();
        let g = &self.g;
        for (c, p) in self.rows[row].iter_mut().zip(&pivot) {
            *c = c.sub(&f.mul(p)).rem(g);
        }
    }
}

/// Minimal polynomial over ℚ of an eigenvalue of a rational matrix, when it
/// has degree ≥ 2 and can be certified: the (irreducible) polynomial of a
/// `RootOf`, or the verified minimal polynomial of a radical expression.
/// `None` for rational eigenvalues (their `A − λI` is rational) and for
/// anything else.
pub(super) fn eigenvalue_minimal_polynomial(lam: &Ex) -> Option<crate::poly::dense::Poly> {
    use crate::base::node::ExprNode;
    if lam.expr_type() == ExprType::Number {
        return None;
    }
    let mut inner = lam.inner.write();
    let arena = &mut inner.arena;
    let g = match arena.node(lam.raw_id()) {
        ExprNode::RootOf(p, v, _) => {
            let (p, v) = (*p, *v);
            let g = crate::poly::polybridge::expr_to_poly(arena, p, v)?;
            irreducible_cached(&g).then_some(g)?
        }
        _ => crate::poly::algebraic::minimal_polynomial(arena, lam.raw_id())?,
    };
    (g.degree()? >= 2).then_some(g)
}

/// Zero test of the pivots of [`Matrix::rref`] (and `rank`, `nullspace`,
/// `lu`, `solve`, the eigenvector solves): **three-valued**, in the generic
/// reading of a matrix whose entries are functions of its symbols.
///
/// - `True` — `e` is proved zero for every value of its symbols: it is
///   never a pivot.
/// - `False` — `e` is proved *not identically zero*: a constant proved
///   nonzero, or an expression with symbols that is nonzero for generic
///   values of them.  It may be a pivot.
/// - `Unknown` — neither: the caller refuses, or (where its signature has
///   no refusal) documents the generic reading it takes.
///
/// The tests, cheapest first.  Without a limit: structural; exact in
/// `ℚ[t]/(g)` for the entries of `A − λI` with `λ = RootOf(g, k)`
/// ([`rootof_field_is_zero`]); exact for rational functions of the symbols
/// over `ℚ(radicals, i)` ([`rational_function_is_zero`]); for a constant
/// [`constant_is_zero`]; and what [`Ex::is_zero`] decides — the test is
/// never weaker than the public one.  Charged to the work budget of the
/// matrix call ([`PivotWork`]): for an expression with symbols and
/// functions a certified value at a sample point that respects the
/// symbols' assumptions (nonzero there: not identically zero); then, for
/// an expression of at most [`PROOF_NODES`] nodes, the proofs of a zero —
/// the identities `simplify` can prove
/// ([`vanishes_by_identity`](crate::simplify::identically_zero::vanishes_by_identity),
/// [`ex_is_zero`]), the constants of the expression that are rational
/// numbers in disguise ([`with_recognised_constants`]), the constant
/// coefficients of its numerator ([`vanishes_by_constant_coefficients`]),
/// and for a constant its factors and terms ([`constant_by_parts`]).  When
/// the budget is spent, or the expression is too large, the entry is
/// `Unknown`.  A value that is zero to the precision reached proves
/// nothing.
///
/// Up to 0.60.0 the test was two-valued and stopped after the first three
/// steps ([`pivot_is_zero`]): an entry it did not decide was a pivot.
/// `[[sin(1)² + cos(1)² − 1]]` had rank 1 and an empty null space although
/// `Ex::is_zero` of the entry is `Some(true)`, and so had 15 of 24 zero
/// forms tried (`ln 6 − ln 2 − ln 3`, `sin²x + cos²x − 1`, `e^{2x} −
/// (e^x)²`, …).
pub(crate) fn pivot_truth(e: &Ex) -> Truth {
    if e.is_zero_structural() {
        return Truth::True;
    }
    if e.expr_type() == ExprType::Number {
        return Truth::False;
    }
    let exact = rootof_field_is_zero(e).or_else(|| rational_function_is_zero(e));
    if exact.is_known() {
        return exact;
    }
    let _work = PivotWork::enter();
    let symbol_free = e.free_symbols().is_empty();
    // A `RootOf` binds a variable that `free_symbols` may report: no
    // sample point, and no part of such an entry is read on its own.  It
    // gets the test of 0.60.0's eigenvector solves, at its cost (an entry
    // `y + 2 − RootOf(…)` of `A − λI` was tested three times over).
    if !symbol_free && has_root_of(e) {
        return ex_is_zero(e);
    }
    if symbol_free {
        let known = constant_core(e);
        if known.is_known() {
            return known;
        }
        // The constant with its parts that are rational numbers written
        // as such: equal to `e`, by the proofs of the replacements.
        let plain = with_recognised_constants(e);
        if plain.is_zero_structural() {
            return Truth::True;
        }
        if plain.expr_type() == ExprType::Number {
            return Truth::False;
        }
        if plain != *e {
            let known = constant_core(&plain);
            if known.is_known() {
                return known;
            }
        }
        return constant_by_parts(&plain);
    }
    let sample_size = tree_size_capped(e, SAMPLE_NODES);
    if sample_size <= SAMPLE_NODES
        && PivotWork::charge(SAMPLE_WORK + sample_size as u64)
        && generically_nonzero_at_a_sample(e)
    {
        return Truth::False;
    }
    let public = Truth::from(e.is_zero());
    if public.is_known() {
        return public;
    }
    let size = tree_size_capped(e, PROOF_NODES);
    if size > PROOF_NODES || !PivotWork::charge(PROOF_WORK + PROOF_WEIGHT * size as u64) {
        return Truth::Unknown;
    }
    if vanishes_by_identity(e) {
        return Truth::True;
    }
    // `e` with its constants that are rational numbers written as such:
    // equal to `e`, by the proofs of the replacements.
    let plain = with_recognised_constants(e);
    if plain != *e {
        if plain.is_zero_structural() {
            return Truth::True;
        }
        let exact = rational_function_is_zero(&plain);
        if exact.is_known() {
            return exact;
        }
        if vanishes_by_identity(&plain) {
            return Truth::True;
        }
    }
    if vanishes_by_constant_coefficients(&plain) {
        return Truth::True;
    }
    ex_is_zero(&plain)
}

/// Work units of the certified zero tests of one matrix call (one
/// elimination; `inv`, `ldl`, `hessenberg` and a Gram–Schmidt process are
/// one call each, `eigenvects` is one per eigenvalue).  A unit is about a
/// node of an expression evaluated at a sample point: a sample costs
/// [`SAMPLE_WORK`] plus the nodes of the entry, the proofs of a zero cost
/// [`PROOF_WORK`] plus [`PROOF_WEIGHT`] per node, a constant read as a
/// rational number [`RECOGNITION_WORK`].  The cap keeps the tests of a call
/// to a few hundred milliseconds whatever the entries are (measured: a 4×4
/// matrix of sixteen entries that no test decides, 0.25 s per elimination
/// in a release build); an entry met after the budget is spent is
/// undecided.
const PIVOT_WORK: u64 = 60_000;

/// Most nodes of an expression that is evaluated at a sample point.
const SAMPLE_NODES: usize = 3_000;

/// Work units of one evaluation at the sample points, besides the nodes.
const SAMPLE_WORK: u64 = 300;

/// Most nodes of an expression for which a zero is looked for by
/// `simplify` and the other proofs (their cost is not linear in the size).
const PROOF_NODES: usize = 300;

/// Work units of the proofs of a zero for one entry, besides its nodes
/// (the samples at the precision cap, `simplify`, `together`).
const PROOF_WORK: u64 = 4_000;

/// Work units per node of the proofs of a zero.
const PROOF_WEIGHT: u64 = 20;

thread_local! {
    /// The work units left to the matrix call in progress on this thread;
    /// `None` outside one.
    static PIVOT_WORK_LEFT: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// The work budget of the certified zero tests of one matrix call
/// ([`PIVOT_WORK`] units), opened by the outermost call that enters and
/// shared by what it calls; closed on drop (also when unwinding).  The
/// work is counted in nodes of the expressions tested, so the same call
/// meets the same limit on every machine and in every context.
pub(crate) struct PivotWork {
    outermost: bool,
}

impl PivotWork {
    /// Open the budget unless a call further out has opened one.
    pub(crate) fn enter() -> Self {
        let outermost = PIVOT_WORK_LEFT.with(|left| {
            let free = left.get().is_none();
            if free {
                left.set(Some(PIVOT_WORK));
            }
            free
        });
        PivotWork { outermost }
    }

    /// Take `units` from the budget in progress: `false`, with nothing
    /// taken, when fewer are left (or no budget is open).
    fn charge(units: u64) -> bool {
        PIVOT_WORK_LEFT.with(|left| match left.get() {
            Some(have) if have >= units => {
                left.set(Some(have - units));
                true
            }
            _ => false,
        })
    }
}

impl Drop for PivotWork {
    fn drop(&mut self) {
        if self.outermost {
            PIVOT_WORK_LEFT.with(|left| left.set(None));
        }
    }
}

/// [`vanishes_by_identity`](crate::simplify::identically_zero::vanishes_by_identity)
/// of an expression.
fn vanishes_by_identity(e: &Ex) -> bool {
    let mut inner = e.inner.write();
    crate::simplify::identically_zero::vanishes_by_identity(&mut inner.arena, e.raw_id())
}

/// The zero tests of a constant `e` as a whole: what [`Ex::is_zero`]
/// decides, [`constant_is_zero`], and — charged to the work budget, for a
/// constant of at most [`PROOF_NODES`] nodes — [`ex_is_zero`] (`simplify`
/// reaching `0`).
fn constant_core(e: &Ex) -> Truth {
    // The public test first: it decides a zero in radicals (an entry of
    // `A − λI`) at once, where the minimal polynomial of the unsimplified
    // entry (`constant_is_zero`) takes a hundred times as long.
    let known = rootof_field_is_zero(e)
        .or_else(|| Truth::from(e.is_zero()))
        .or_else(|| constant_is_zero(e));
    if known.is_known() {
        return known;
    }
    let size = tree_size_capped(e, PROOF_NODES);
    if size > PROOF_NODES || !PivotWork::charge(PROOF_WORK + PROOF_WEIGHT * size as u64) {
        return Truth::Unknown;
    }
    ex_is_zero(e)
}

/// A constant decided by its parts, one level down ([`constant_core`] of
/// each part): a product is zero when a factor is proved zero and the
/// others are proved finite, and nonzero when every factor is proved
/// nonzero; a power with a positive exponent is what its base is; a sum is
/// zero when every term is proved zero (`(cos(π/7) + cos(3π/7) + cos(5π/7)
/// − 1/2)·(erf 1 + erfc 1 − 1)`: the zero test of constants proves each
/// factor and not the product).  `Unknown` for any other form.
fn constant_by_parts(e: &Ex) -> Truth {
    use crate::base::node::{ExprId, ExprNode};
    enum Parts {
        Factors(Vec<ExprId>),
        Terms(Vec<ExprId>),
        Base(ExprId),
    }
    let parts = {
        let inner = e.inner.read();
        let arena = &inner.arena;
        match arena.node(e.raw_id()) {
            ExprNode::Mul(ch) => Parts::Factors(ch.to_vec()),
            ExprNode::Add(ch) => Parts::Terms(ch.to_vec()),
            ExprNode::Neg(c) => Parts::Base(*c),
            ExprNode::Pow(b, x)
                if arena
                    .as_num(*x)
                    .is_some_and(num_traits::Signed::is_positive) =>
            {
                Parts::Base(*b)
            }
            _ => return Truth::Unknown,
        }
    };
    match parts {
        Parts::Base(b) => constant_core(&e.wrap(b)),
        Parts::Terms(terms) => {
            if terms.iter().all(|&t| constant_core(&e.wrap(t)).is_true()) {
                Truth::True
            } else {
                Truth::Unknown
            }
        }
        Parts::Factors(factors) => {
            let factors: Vec<Ex> = factors.into_iter().map(|f| e.wrap(f)).collect();
            let verdicts: Vec<Truth> = factors.iter().map(constant_core).collect();
            if verdicts.iter().all(|v| v.is_false()) {
                return Truth::False;
            }
            let zero_and_finite = verdicts.iter().any(|v| v.is_true())
                && factors
                    .iter()
                    .zip(&verdicts)
                    .all(|(f, v)| v.is_true() || Truth::from(f.is_finite()).is_true());
            if zero_and_finite {
                Truth::True
            } else {
                Truth::Unknown
            }
        }
    }
}

/// Largest denominator of a rational number that
/// [`with_recognised_constants`] looks for.
const RECOGNISED_DENOMINATOR: i64 = 1_000;

/// Work units of one attempt of [`with_recognised_constants`] to read a
/// constant as a rational number (an evaluation and a proof).
const RECOGNITION_WORK: u64 = 1_500;

/// `e` with each constant part that is a rational number in disguise
/// replaced by that number, from the inside out: a constant sum, product
/// or integer power, and the constant terms of a sum or the constant
/// factors of a product taken together (`x + 1 + cos(π/7) + cos(3π/7) +
/// cos(5π/7) − 1/2` is `x + 1`, `(erf 1 + erfc 1)·x` is `x`, `2/(erf 1 +
/// erfc 1 − 2) + 2` is `0`).  The number is read off twelve certified
/// digits of the part (a guess) and the replacement is made only when the
/// difference is **proved** zero by the zero test of constants
/// ([`constant_core`]), so the result is equal to `e`.  Function arguments
/// and non-integer powers are not entered; each attempt is charged to the
/// work budget ([`RECOGNITION_WORK`]) and none is made once it is spent.
///
/// An elimination mixes the constants of different entries: a sum of
/// products of two disguised numbers is not proved zero as a whole, and is
/// plain arithmetic once each is written as the number it is.
fn with_recognised_constants(e: &Ex) -> Ex {
    use crate::base::node::{ExprId, ExprNode};
    enum Shape {
        Leaf,
        Sum(Vec<ExprId>),
        Product(Vec<ExprId>),
        Power(ExprId, ExprId),
        Negation(ExprId),
    }
    let shape = |id: ExprId| -> Shape {
        let inner = e.inner.read();
        let arena = &inner.arena;
        match arena.node(id) {
            ExprNode::Add(ch) => Shape::Sum(ch.to_vec()),
            ExprNode::Mul(ch) => Shape::Product(ch.to_vec()),
            ExprNode::Pow(b, x) if arena.as_num(*x).is_some_and(|r| r.is_integer()) => {
                Shape::Power(*b, *x)
            }
            ExprNode::Neg(c) => Shape::Negation(*c),
            _ => Shape::Leaf,
        }
    };
    let symbol_free = |id: ExprId| e.inner.read().arena.is_symbol_free(id);
    // The rational number that the constant `k` is proved to be, or `k`.
    let recognised = |k: ExprId| -> ExprId {
        let constant = e.wrap(k);
        if constant.expr_type() == ExprType::Number
            || !symbol_free(k)
            || !PivotWork::charge(RECOGNITION_WORK)
        {
            return k;
        }
        // Twelve certified digits of the real part (exact, as a rational);
        // a constant that does not evaluate is left alone, and one that
        // is not real fails the proof below.
        let digits = {
            let inner = e.inner.read();
            crate::transforms::evalf::const_value(&inner.arena, k, 12)
                .ok()
                .and_then(|v| v.re_decimal())
                .and_then(|d| d.to_ratio(18))
        };
        let Some(guess) = digits.and_then(|v| rational_near(&v)) else {
            return k;
        };
        let number = e.context().rational(guess.0, guess.1);
        if constant_core(&(&constant - &number)).is_true() {
            number.raw_id()
        } else {
            k
        }
    };
    let mut value: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let mut stack = vec![(e.raw_id(), false)];
    while let Some((id, ready)) = stack.pop() {
        if value.contains_key(&id) {
            continue;
        }
        let mapped = |c: &ExprId| value.get(c).copied().unwrap_or(*c);
        let new = match shape(id) {
            Shape::Leaf => id,
            Shape::Sum(children) | Shape::Product(children) if !ready => {
                stack.push((id, true));
                stack.extend(children.into_iter().map(|c| (c, false)));
                continue;
            }
            Shape::Power(base, _) | Shape::Negation(base) if !ready => {
                stack.push((id, true));
                stack.push((base, false));
                continue;
            }
            Shape::Sum(children) => {
                let (constants, mut parts): (Vec<ExprId>, Vec<ExprId>) =
                    children.iter().map(mapped).partition(|&c| symbol_free(c));
                if !constants.is_empty() {
                    let together = e.inner.write().arena.add(&constants);
                    parts.push(recognised(together));
                }
                e.inner.write().arena.add(&parts)
            }
            Shape::Product(children) => {
                let (constants, mut parts): (Vec<ExprId>, Vec<ExprId>) =
                    children.iter().map(mapped).partition(|&c| symbol_free(c));
                if !constants.is_empty() {
                    let together = e.inner.write().arena.mul(&constants);
                    parts.push(recognised(together));
                }
                e.inner.write().arena.mul(&parts)
            }
            Shape::Power(base, exponent) => {
                let base = mapped(&base);
                let power = e.inner.write().arena.pow(base, exponent);
                recognised(power)
            }
            Shape::Negation(base) => {
                let base = mapped(&base);
                e.inner.write().arena.neg(base)
            }
        };
        value.insert(id, new);
    }
    match value.get(&e.raw_id()) {
        Some(&id) => e.wrap(id),
        None => e.clone(),
    }
}

/// The rational `p/q` with the smallest `0 < q ≤` [`RECOGNISED_DENOMINATOR`]
/// that lies within `10⁻⁹` of `v`, when `|v| < 10⁹`: a guess at what a
/// constant is, to be proved by its caller.  Exact arithmetic.
fn rational_near(v: &Q) -> Option<(i64, i64)> {
    use num_traits::{Signed, ToPrimitive};
    let bound = Q::from_integer(BigInt::from(1_000_000_000));
    if v.abs() >= bound {
        return None;
    }
    let tolerance = bound.recip();
    (1..=RECOGNISED_DENOMINATOR).find_map(|q| {
        let denominator = Q::from_integer(BigInt::from(q));
        let p = (v * &denominator).round();
        ((v - &p / &denominator).abs() <= tolerance)
            .then(|| p.to_integer().to_i64())
            .flatten()
            .map(|p| (p, q))
    })
}

/// Most nodes of a numerator that [`vanishes_by_constant_coefficients`]
/// multiplies out.
const COEFFICIENT_TEST_NODES: usize = 600;

/// Is `e`, an expression with symbols, zero because the constant
/// coefficients of its numerator are?  The numerator is multiplied out and
/// its terms are collected by their factors with symbols; a coefficient is
/// a constant, decided by the zero test of constants (`x·(erf 1 + erfc 1 −
/// 1)`, `y + 2 − (y + 2) + cos(π/7) + cos(3π/7) + cos(5π/7) − 1/2`: the
/// identities of constants are proved for constants only, and the
/// eliminations mix them with the symbols).  The terms whose coefficient is
/// proved zero are dropped; what is left must be nothing, or zero by an
/// identity of its functions.  The denominator must be proved not
/// identically zero (the value exists).  `false` proves nothing.
fn vanishes_by_constant_coefficients(e: &Ex) -> bool {
    use crate::base::node::{ExprId, ExprNode};
    let (num, den) = {
        let mut inner = e.inner.write();
        crate::poly::polybridge::fraction_parts(&mut inner.arena, e.raw_id())
    };
    let (num, den) = (e.wrap(num), e.wrap(den));
    // The value must exist: a denominator proved not identically zero.
    if den == *e
        || tree_size_capped(&num, COEFFICIENT_TEST_NODES) > COEFFICIENT_TEST_NODES
        || !pivot_truth(&den).is_false()
    {
        return false;
    }
    let expanded = num.expand();
    // The terms by their factors with symbols: (those factors' product,
    // the constant factors' products).
    let mut groups: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    {
        let mut inner = expanded.inner.write();
        let arena = &mut inner.arena;
        let terms: Vec<ExprId> = match arena.node(expanded.raw_id()) {
            ExprNode::Add(ch) => ch.to_vec(),
            _ => vec![expanded.raw_id()],
        };
        for term in terms {
            let (term, negated) = match arena.node(term) {
                ExprNode::Neg(c) => (*c, true),
                _ => (term, false),
            };
            let factors: Vec<ExprId> = match arena.node(term) {
                ExprNode::Mul(ch) => ch.to_vec(),
                _ => vec![term],
            };
            let (mut constants, rest): (Vec<ExprId>, Vec<ExprId>) =
                factors.into_iter().partition(|&f| arena.is_symbol_free(f));
            if negated {
                constants.push(arena.int(-1));
            }
            let one = arena.one();
            let product = |arena: &mut crate::base::arena::Arena, fs: &[ExprId]| match fs {
                [] => one,
                fs => arena.mul(fs),
            };
            let monomial = product(arena, &rest);
            let coefficient = product(arena, &constants);
            match groups.iter_mut().find(|(m, _)| *m == monomial) {
                Some((_, cs)) => cs.push(coefficient),
                None => groups.push((monomial, vec![coefficient])),
            }
        }
    }
    let mut dropped = false;
    let mut kept: Vec<Ex> = Vec::new();
    for (monomial, coefficients) in groups {
        let coefficient = {
            let mut inner = expanded.inner.write();
            let id = inner.arena.add(&coefficients);
            drop(inner);
            expanded.wrap(id)
        };
        if pivot_truth(&coefficient).is_true() {
            dropped = true;
        } else {
            kept.push(&coefficient * &expanded.wrap(monomial));
        }
    }
    if !dropped {
        return false;
    }
    if kept.is_empty() {
        return true;
    }
    let rest = Ex::sum_of(&expanded.context(), kept);
    if rest.is_zero_structural() {
        return true;
    }
    vanishes_by_identity(&rest) || ex_is_zero(&rest).is_true()
}

/// Is the expression `e` with free symbols certified nonzero at a sample
/// point that respects the assumptions of its symbols
/// ([`nonzero_at_a_sample`](crate::simplify::identically_zero::nonzero_at_a_sample))?
/// Then it is not identically zero.  Not asked of an expression that may
/// vanish on a region without vanishing identically (`H(x)`, `0^x`,
/// `max(x, 0)`: [`may_vanish_on_a_region`](crate::base::canon::may_vanish_on_a_region)),
/// for which "nonzero for generic values" does not follow from one point.
/// `false` proves nothing.
fn generically_nonzero_at_a_sample(e: &Ex) -> bool {
    let mut inner = e.inner.write();
    let arena = &mut inner.arena;
    !crate::base::canon::may_vanish_on_a_region(arena, e.raw_id())
        && crate::simplify::identically_zero::nonzero_at_a_sample(arena, e.raw_id())
}

/// The zero test of 0.60.0 — structural, exact for rational functions of
/// the symbols over `ℚ(radicals, i)`, and for any other constant
/// [`constant_is_zero`] — two-valued: proved zero.  For the block split of
/// the eigenvalues only, whose result is right either way (an entry not
/// proved zero makes a block larger; two equal eigenvalues of different
/// blocks that are not joined here are joined, or refused, by
/// [`merged_eigen_pairs`]).  It is kept there because the split decides
/// which characteristic polynomials are solved: with [`pivot_truth`] a
/// matrix with a disguised zero splits into smaller blocks, and a cubic
/// block with unsimplified coefficients takes seconds in Cardano's formula
/// where the whole matrix was refused at once (measured: `eigenvals` 0.14 s
/// → 38 s for a 4×4 matrix of the hunt).
pub(super) fn pivot_is_zero(e: &Ex) -> bool {
    if e.is_zero_structural() {
        return true;
    }
    if e.expr_type() == ExprType::Number {
        return false;
    }
    rational_function_is_zero(e)
        .or_else(|| constant_is_zero(e))
        .is_true()
}

/// [`pivot_truth`] read as "proved zero", for an elimination step that is
/// right either way (a row is cleared by an entry that is not proved zero,
/// whatever it is).
pub(super) fn proved_zero(e: &Ex) -> bool {
    pivot_truth(e).is_true()
}

/// Zero test of a constant that is not a rational function over
/// `ℚ(radicals of rationals, i)` — a root of a negative number of index
/// above 2, a nested radical, a `RootOf`, `π`: exactly in its number field
/// for a single `RootOf` ([`rootof_field_is_zero`]), otherwise the certified
/// test (a certified digit is nonzero; a value zero to the precision goes to
/// the minimal polynomial of an algebraic number, then the deep zero
/// search; undecided stays `None`).  `None` for an expression with free
/// symbols.
///
/// Before 0.31 the pivots of `rref` never reached it (only `ex_is_zero`,
/// behind `inv` and the eigenvectors, did): the entry `∛(−2) −
/// ∛2·(1/2 + √3·i/2)`, exactly 0 (`(−2)^(1/3) = 2^(1/3)·e^(iπ/3)`), was a
/// pivot, so `rank` of `[[it, 0], [0, 1]]` was 2, its null space empty and
/// `lu` returned a "factorization" with that zero as a pivot.
fn constant_is_zero(e: &Ex) -> Truth {
    if !e.free_symbols().is_empty() {
        return Truth::Unknown;
    }
    rootof_field_is_zero(e).or_else(|| {
        let mut inner = e.inner.write();
        Truth::from(crate::poly::algebraic::is_zero_checked(
            &mut inner.arena,
            e.raw_id(),
        ))
    })
}

/// The refusal of an elimination that met a pivot candidate `e` neither
/// proved zero nor proved nonzero ([`pivot_truth`]).
pub(crate) fn undecided_pivot(operation: &'static str, e: &Ex) -> SymplexError {
    SymplexError::computation_failed(
        operation,
        format!(
            "cannot decide whether the entry {} is zero (it is neither proved zero nor proved \
             nonzero), so the rank is not known; simplify the entries or substitute values \
             first",
            shortened(e)
        ),
    )
}

/// `e` printed, cut after 120 characters (an entry of an elimination or
/// an eigenvalue in radicals can fill a page).
pub(super) fn shortened(e: &Ex) -> String {
    let shown = e.to_string();
    match shown.char_indices().nth(120) {
        Some((cut, _)) => format!("{}…", &shown[..cut]),
        None => shown,
    }
}

/// Eigenvalues with their multiplicities.
pub(super) type EigenPairs = Vec<(Ex, usize)>;

/// The eigenvalues `pairs` with the ones that are proved equal
/// ([`pivot_truth`] of their difference) merged, their multiplicities
/// added; and the first two eigenvalues of the result that are **not**
/// proved different, if any.  Two eigenvalues are proved different when
/// their difference is proved not identically zero and is meromorphic in
/// the symbols by its form ([`analytic_by_form`]): a difference with a
/// root or another branch function is analytic only piece by piece, and
/// may vanish on one piece while a sample point lies in another.
pub(super) fn merged_eigen_pairs(pairs: EigenPairs) -> (EigenPairs, Option<(Ex, Ex)>) {
    let mut out: Vec<(Ex, usize)> = Vec::new();
    let mut open: Option<(Ex, Ex)> = None;
    for (v, m) in pairs {
        let mut same = None;
        for (idx, (w, _)) in out.iter().enumerate() {
            let difference = w.clone() - v.clone();
            match pivot_truth(&difference) {
                Truth::True => {
                    same = Some(idx);
                    break;
                }
                Truth::False if analytic_by_form(&difference, false) => {}
                _ => {
                    if open.is_none() {
                        open = Some((w.clone(), v.clone()));
                    }
                }
            }
        }
        match same.and_then(|idx| out.get_mut(idx)) {
            Some((_, k)) => *k += m,
            None => out.push((v, m)),
        }
    }
    (out, open)
}

/// An eigenvector entry of a symbolic matrix that is a rational function
/// of its symbols, in `ratsimp` normal form (SymPy's `eigenvects`
/// simplifies too).  The elimination leaves nested quotients: before 0.38
/// the eigenvector of `y² + 6x + 6` of the rank-1 4×4 matrix `u·vᵀ` with
/// `u = (x, y, x, 2)`, `v = (3, y, 3, 3)` was 2 KB instead of
/// `(x/2, y/2, x/2, 1)`, and inverting the eigenvector matrix for
/// `matrix_exp` took 14 s.  Other entries are returned as they are.
pub(super) fn tidy_symbolic_entry(e: &Ex) -> Ex {
    if e.expr_type() == ExprType::Number
        || e.free_symbols().is_empty()
        || has_radical(e)
        || has_root_of(e)
    {
        return e.clone();
    }
    e.ratsimp()
}

/// Is `e` provably not the zero function?  Exactly for rational functions
/// over `ℚ(radicals, i)` and certified for constants; otherwise `e` is
/// evaluated (certified) at a few rational points of its free symbols: a
/// nonzero value there proves `e ≢ 0`.  `false` means "not proven", never
/// "zero".
fn provably_nonzero(e: &Ex) -> bool {
    if e.is_zero_structural() {
        return false;
    }
    if e.expr_type() == ExprType::Number {
        return true;
    }
    let rf = rational_function_is_zero(e);
    if rf.is_known() {
        return rf.is_false();
    }
    let syms = e.free_symbols();
    if syms.is_empty() {
        return constant_is_zero(e).is_false();
    }
    // A `RootOf` binds a variable that `free_symbols` may report.
    if has_root_of(e) {
        return false;
    }
    let ctx = e.context();
    (0..3i64).any(|k| {
        let vals: Vec<Ex> = (0..syms.len() as i64)
            .map(|i| ctx.rational(31 + 17 * i + 7 * k, 11 + 4 * k + 3 * i))
            .collect();
        let subs: Vec<(&Ex, &Ex)> = syms.iter().zip(vals.iter()).collect();
        constant_is_zero(&e.subs_map(&subs)).is_false()
    })
}

/// [`provably_nonzero`] for an expression in Cardano/Ferrari radicals of
/// its symbols, by evaluation only: nonzero digits of `e` at one of a few
/// rational points of its symbols prove `e ≢ 0`.  A value that does not
/// evaluate there proves nothing; the exact fallback of [`constant_is_zero`]
/// (the minimal polynomial of the value) is skipped — for such radicals it
/// is a Zassenhaus factorization of enormous degree (minutes for one entry
/// of an eigenvector of a symbolic 4×4 matrix).
pub(super) fn radical_nonzero_by_value(e: &Ex) -> bool {
    if e.is_zero_structural() {
        return false;
    }
    if e.expr_type() == ExprType::Number {
        return true;
    }
    let rf = rational_function_is_zero(e);
    if rf.is_known() {
        return rf.is_false();
    }
    let syms = e.free_symbols();
    if has_root_of(e) {
        return false;
    }
    let ctx = e.context();
    (0..3i64).any(|k| {
        let vals: Vec<Ex> = (0..syms.len() as i64)
            .map(|i| ctx.rational(31 + 17 * i + 7 * k, 11 + 4 * k + 3 * i))
            .collect();
        let subs: Vec<(&Ex, &Ex)> = syms.iter().zip(vals.iter()).collect();
        // A part with certified digits (up to 0.57: the printout was not `"0"`).
        let at = e.subs_map(&subs).eval();
        let guard = at.inner.read();
        crate::transforms::evalf::const_value(&guard.arena, at.raw_id(), 12)
            .is_ok_and(|v| v.re_sign().is_nonzero().is_true() || v.im_sign().is_nonzero().is_true())
    })
}

/// The eigenvector of a simple eigenvalue `λ` from `m = A − λI`.
///
/// An eigenvalue of algebraic multiplicity 1 has a one-dimensional
/// eigenspace, so `rank(A − λI) = n − 1` is known in advance: Gaussian
/// elimination needs only `n − 1` pivots that are *provably* nonzero
/// ([`provably_nonzero`]; the smallest such entry of a column is taken),
/// and the remaining row is then zero without a zero test.  This is what
/// a symbolic eigenvalue such as `x²/2 + √(x⁴ + 8x² + 8xy)/2` needs: no
/// zero test decides the last pivot `−λ − 2x(x + y)/(x² − λ)`, and
/// before 0.38 `eigenvects` returned no eigenvector for it (SymPy:
/// `[λ/(2x), 1]`).  `None` when fewer than `n − 1` pivots are proven.
pub(super) fn simple_eigenvector(m: &Matrix) -> Option<Matrix> {
    let n = m.ncols;
    let nr = m.nrows;
    if n == 0 || nr + 1 < n {
        return None;
    }
    let mut rows = m.rows.clone();
    let mut pivots: Vec<usize> = Vec::new();
    let mut pr = 0usize;
    for col in 0..n {
        if pr + 1 == n {
            break;
        }
        let Some(best) = (pr..nr)
            .filter(|&i| provably_nonzero(&rows[i][col]))
            .min_by_key(|&i| rows[i][col].count_ops())
        else {
            continue;
        };
        rows.swap(pr, best);
        let piv = rows[pr][col].clone();
        for e in rows[pr].iter_mut() {
            *e = (&*e / &piv).eval();
        }
        let pivot_row = rows[pr].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i == pr || row[col].is_zero_structural() {
                continue;
            }
            let factor = row[col].clone();
            for (e, p) in row.iter_mut().zip(&pivot_row) {
                *e = (&*e - &(&factor * p)).eval();
            }
        }
        pivots.push(col);
        pr += 1;
    }
    if pr + 1 != n {
        return None;
    }
    let free = (0..n).find(|c| !pivots.contains(c))?;
    let mut v = vec![m.ctx_zero(); n];
    v[free] = m.ctx_one();
    for (k, &pc) in pivots.iter().enumerate() {
        v[pc] = (-&rows[k][free]).eval();
    }
    Some(Matrix::col_vector_unchecked(v))
}

/// Are the eigenvectors of `spaces` (all of them: as many as the matrix has
/// rows) a basis for **every** value of the symbols?  Proved when every
/// eigenvalue and every entry of the eigenvectors is finite at every point
/// of the symbols' domain and the determinant of the eigenvector matrix `P`
/// vanishes at none.  `A·P = P·D` is an identity between functions of the
/// symbols, so it then holds at every point with an invertible `P`: the
/// matrix is diagonalizable there, also where two eigenvalues meet (`[[x,
/// y], [y, x]]` at `y = 0`).  `false` proves nothing.
pub(super) fn eigenvectors_independent_everywhere(spaces: &[(Ex, usize, Vec<Matrix>)]) -> bool {
    let cols: Vec<&Matrix> = spaces.iter().flat_map(|(_, _, vs)| vs.iter()).collect();
    if cols.is_empty()
        || !spaces.iter().all(|(l, _, _)| finite_everywhere(l))
        || !cols.iter().all(|v| v.iter().all(finite_everywhere))
    {
        return false;
    }
    let Ok(p) = Matrix::hstack(&cols) else {
        return false;
    };
    p.is_square()
        && p.det().is_ok_and(|d| {
            tree_size_capped(&d, EVERYWHERE_PROOF_NODES) <= EVERYWHERE_PROOF_NODES
                && Truth::from(d.is_nonzero()).is_true()
        })
}

/// Most nodes of an expression of which a proof "for every value of the
/// symbols" asks a query (finite, nonzero): `is_diagonalizable` answers
/// `None` rather than spend seconds on an eigenvector in radicals.
pub(super) const EVERYWHERE_PROOF_NODES: usize = 300;

/// Is `e` finite at every finite value of its symbols?  Proved by its form
/// ([`analytic_by_form`], entire), or — for a small expression — by the
/// assumption system ([`Ex::is_finite`], which reads a symbol without
/// assumptions as possibly infinite).  `false` proves nothing.
fn finite_everywhere(e: &Ex) -> bool {
    // The query is not asked of a large expression (an eigenvector in
    // radicals): it would cost more than the answer is worth here.
    analytic_by_form(e, true)
        || (tree_size_capped(e, EVERYWHERE_PROOF_NODES) <= EVERYWHERE_PROOF_NODES
            && Truth::from(e.is_finite()).is_true())
}

/// Is `e` an analytic function of its symbols by its form: constants (any
/// part without symbols) and symbols (not declared infinite) under sums,
/// products, powers with an integer exponent and the entire functions
/// `sin`, `cos`, `exp`, `sinh`, `cosh` — no root, no logarithm, no other
/// function with a branch cut or a step?  With `entire` the exponents must
/// not be negative (no pole: the value is finite at every finite point; a
/// non-negative rational exponent is allowed there); without it `e` is
/// meromorphic: single-valued on the whole plane of each symbol, so a
/// nonzero value at one point means nonzero except on a set without
/// interior.  `false` proves nothing.
fn analytic_by_form(e: &Ex, entire: bool) -> bool {
    use crate::base::assumptions::Props;
    use crate::base::node::ExprNode;
    let inner = e.inner.read();
    let arena = &inner.arena;
    let mut seen = rustc_hash::FxHashSet::default();
    let mut stack = vec![e.raw_id()];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Num(_) | ExprNode::Pi | ExprNode::E | ExprNode::ImaginaryUnit => {}
            ExprNode::Symbol(sid)
                if !Truth::from(arena.symbol_assumptions(*sid).query(Props::FINITE)).is_false() => {
            }
            // A constant part of a meromorphic function is a number.
            _ if !entire && arena.is_symbol_free(id) => {}
            ExprNode::Add(ch) | ExprNode::Mul(ch) => stack.extend(ch.iter().copied()),
            ExprNode::Neg(c)
            | ExprNode::Sin(c)
            | ExprNode::Cos(c)
            | ExprNode::Exp(c)
            | ExprNode::Sinh(c)
            | ExprNode::Cosh(c) => stack.push(*c),
            ExprNode::Pow(b, x)
                if arena.as_num(*x).is_some_and(|r| {
                    if entire {
                        !num_traits::Signed::is_negative(r)
                    } else {
                        r.is_integer()
                    }
                }) =>
            {
                stack.push(*b);
            }
            _ => return false,
        }
    }
    true
}

/// Compute nullspace of `(a_minus_lambda)^power`.
pub(super) fn jordan_null_power(
    a_minus_lambda: &Matrix,
    power: usize,
) -> Result<Vec<Matrix>, SymplexError> {
    if power == 0 {
        return Ok(Vec::new());
    }
    let mut m = a_minus_lambda.clone();
    for _ in 1..power {
        m = m.matmul(a_minus_lambda)?;
    }
    m.nullspace_decided("jordan_form")
}

/// Compute `(a_minus_lambda)^power * vec` where vec is a column vector.
pub(super) fn matrix_pow_vec(
    a_minus_lambda: &Matrix,
    vec: &Matrix,
    power: usize,
) -> Result<Matrix, SymplexError> {
    let mut result = vec.clone();
    for _ in 0..power {
        result = a_minus_lambda.matmul(&result)?;
    }
    Ok(result)
}

/// Pick a vector from `candidates` that lies outside the span of
/// `exclude` (which may itself be linearly dependent or contain
/// duplicates).  A candidate qualifies when appending it raises the rank.
pub(super) fn pick_independent_vec(
    candidates: &[Matrix],
    exclude: &[&Matrix],
) -> Result<Option<Matrix>, SymplexError> {
    if candidates.is_empty() {
        return Ok(None);
    }
    if exclude.is_empty() {
        return Ok(Some(candidates[0].clone()));
    }
    let base_rank = Matrix::hstack(exclude)?.rank_decided("jordan_form")?;
    for candidate in candidates {
        let mut cols: Vec<&Matrix> = exclude.to_vec();
        cols.push(candidate);
        let combined = Matrix::hstack(&cols)?;
        if combined.rank_decided("jordan_form")? > base_rank {
            return Ok(Some(candidate.clone()));
        }
    }
    Ok(None)
}
