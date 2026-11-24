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
// Eigenvalue multiplicity via polynomial factoring
// ═══════════════════════════════════════════════════════════════════════════

/// Extract eigenvalues with algebraic multiplicities from a characteristic
/// polynomial.
///
/// **Primary path:** convert to `Poly`, call `factor_over_z()` (Yun's
/// square-free decomposition) for exact `(factor, multiplicity)` pairs;
/// each factor is solved for its roots, which inherit the multiplicity.
///
/// **Fallback:** if the polynomial has non-rational coefficients,
/// `expr_to_poly` returns `None`; then derivative-based multiplicity
/// detection is used on whatever roots the generic solver finds.
pub(super) fn eigvals_with_multiplicity(char_poly: &Ex, var: &Ex) -> Vec<(Ex, usize)> {
    if let Some(pairs) = eigvals_via_poly_factor(char_poly, var)
        && !pairs.is_empty()
    {
        debug!(
            count = pairs.len(),
            "eigvals_with_multiplicity: used Poly::factor_over_z path"
        );
        return pairs;
    }
    debug!("eigvals_with_multiplicity: falling back to derivative-based detection");
    eigvals_via_derivative(char_poly, var)
}

/// Primary multiplicity path: convert to Poly, factor, solve each factor.
fn eigvals_via_poly_factor(char_poly: &Ex, var: &Ex) -> Option<Vec<(Ex, usize)>> {
    let inner = char_poly.inner.read();
    let arena = &inner.arena;
    let poly = crate::poly::polybridge::expr_to_poly(arena, char_poly.raw_id(), var.raw_id())?;
    let (_content, factors) = poly.factor_over_z();
    if factors.is_empty() {
        return None;
    }
    drop(inner);

    let mut eigen_pairs: Vec<(Ex, usize)> = Vec::new();
    for (factor, mult) in &factors {
        let degree = factor.degree().unwrap_or(0);
        if degree == 0 {
            continue;
        }
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
            (0..degree).map(|i| root_of(&factor_ex, var, i)).collect()
        } else {
            factor_ex.solve_or_empty(var)
        };
        if roots.is_empty() {
            warn!(
                "eigenvals: irreducible factor of degree {} yielded no roots — \
                 eigenvalues from this factor are missing",
                degree
            );
        }
        trace!(
            degree,
            root_count = roots.len(),
            mult,
            "eigvals_via_poly_factor: degree-{} factor, {} roots, mult {}",
            degree,
            roots.len(),
            mult
        );
        for r in roots {
            eigen_pairs.push((r, *mult as usize));
        }
    }
    Some(eigen_pairs)
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
pub(super) fn symbolic_factor_roots(cp: &Ex, lam: &Ex) -> Option<Vec<(Ex, usize)>> {
    let (_, factors) = cp.factor_list_all();
    if factors.len() < 2 && factors.iter().all(|(_, m)| *m < 2) {
        return None;
    }
    let mut out: Vec<(Ex, usize)> = Vec::new();
    for (f, m) in &factors {
        let coeffs = f.coeffs(lam)?;
        let parametric = coeffs.iter().any(|c| !c.free_symbols().is_empty());
        let roots = match coeffs.len() {
            0 | 1 => continue,
            2 | 3 => low_degree_roots(&coeffs),
            // An irreducible symbolic cubic or quartic factor: in radicals
            // (before 0.44 its roots were missing from the result).
            4 | 5 if parametric => match parametric_radical_roots(&coeffs) {
                Some(rs) => rs.into_iter().map(|r| (r, 1)).collect(),
                None => eigvals_with_multiplicity(&f.expand(), lam),
            },
            _ => eigvals_with_multiplicity(&f.expand(), lam),
        };
        for (r, k) in roots {
            out.push((r, k * *m as usize));
        }
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
fn eigvals_via_derivative(char_poly: &Ex, var: &Ex) -> Vec<(Ex, usize)> {
    let all_roots = char_poly.solve_or_empty(var);
    if all_roots.is_empty() {
        return Vec::new();
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
    eigen_pairs
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
    fn is_zero(&mut self, i: usize, j: usize, e: &Ex) -> bool;
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
/// the result to be written as zeros (the public `rref`).
pub(super) struct ClosureOracle<'a> {
    pub(super) test: &'a dyn Fn(&Ex) -> bool,
    pub(super) clean: bool,
}

impl PivotOracle for ClosureOracle<'_> {
    fn is_zero(&mut self, _i: usize, _j: usize, e: &Ex) -> bool {
        (self.test)(e)
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
    fn is_zero(&mut self, i: usize, j: usize, _e: &Ex) -> bool {
        self.rows[i][j].is_zero()
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
/// `lu`, `solve`, the block split of the eigenvalues): structural, exact for
/// rational functions of the symbols over `ℚ(radicals, i)`, and for any
/// other constant [`constant_is_zero`].  A symbolic entry that is neither
/// is a pivot unless structurally zero (the generic rank).
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

pub(super) fn eigen_zero_test(e: &Ex) -> bool {
    if e.is_zero_structural() {
        return true;
    }
    if e.expr_type() == ExprType::Number {
        return false;
    }
    // Exact for the entries of `A − λI` with rational `A` and `λ =
    // RootOf(g, k)`; certified for other constants (see `ex_is_zero`).
    // Before 0.30 an undecided entry was zero when its `f64` value was
    // below `10⁻¹⁰` in absolute value (see `rootof_field_is_zero`).
    rootof_field_is_zero(e).or_else(|| ex_is_zero(e)).is_true()
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
    Ok(m.nullspace_semantic())
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
    let base_rank = Matrix::hstack(exclude)?.rank_semantic();
    for candidate in candidates {
        let mut cols: Vec<&Matrix> = exclude.to_vec();
        cols.push(candidate);
        let combined = Matrix::hstack(&cols)?;
        if combined.rank_semantic() > base_rank {
            return Ok(Some(candidate.clone()));
        }
    }
    Ok(None)
}
