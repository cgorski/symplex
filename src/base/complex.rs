//! Complex analysis: real/imaginary decomposition, conjugation and argument.
//!
//! This module implements the canonical constructors behind
//! [`Arena::re`], [`Arena::im`], [`Arena::conjugate`] and [`Arena::arg`],
//! plus [`as_real_imag`], which splits any expression into its real and
//! imaginary parts: `expr = re + im·i`.
//!
//! # Design
//!
//! Nothing here *assumes* realness.  A symbol is only treated as real when
//! the assumption system ([`crate::base::assumptions`]) can prove it; every
//! other unknown quantity is represented by explicit
//! [`Re`](ExprNode::Re) / [`Im`](ExprNode::Im) / [`Arg`](ExprNode::Arg) /
//! [`Conjugate`](ExprNode::Conjugate) nodes.  This replaces the pre-0.2
//! behaviour of silently treating every bare symbol as real.
//!
//! Principal-branch semantics are used throughout: `arg(z) ∈ (−π, π]`,
//! `ln z = ln|z| + i·arg z`.  Functions with branch cuts (`ln`, non-integer
//! powers, inverse trig/hyperbolic functions, `LambertW`, `Ei`, `Ci`, `li`)
//! are never pushed through `conjugate`; they stay as unevaluated nodes
//! unless the argument is provably real.
//!
//! All traversals are iterative (post-order over the DAG with a cache);
//! no recursion over expression trees.

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};
use rustc_hash::FxHashMap;

// ═══════════════════════════════════════════════════════════════════════════
// Assumption helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Query a single property of `id` with a fresh assumption cache.
///
/// The cache reads symbol-level assumptions from the arena's symbol table,
/// so results agree with [`Context::query`](crate::api::context::Context::query)
/// for everything except properties that were asserted on *compound*
/// expressions via the context cache.
pub(crate) fn query_prop(arena: &Arena, id: ExprId, prop: Props) -> Option<bool> {
    let mut cache = AssumptionCache::new();
    cache.query(arena, id, prop)
}

/// Three-valued realness test: `Some(true)` if provably real, `Some(false)`
/// if provably not real, `None` if unknown.
pub(crate) fn is_real(arena: &Arena, id: ExprId) -> Option<bool> {
    query_prop(arena, id, Props::REAL)
}

/// Nodes that are real-valued by definition regardless of their argument.
fn is_intrinsically_real(node: &ExprNode) -> bool {
    matches!(
        node,
        ExprNode::Num(_)
            | ExprNode::Pi
            | ExprNode::E
            | ExprNode::EulerGamma
            | ExprNode::Catalan
            | ExprNode::GoldenRatio
            | ExprNode::PhysicalConstant(_, _)
            | ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::Abs(_)
            | ExprNode::Re(_)
            | ExprNode::Im(_)
            | ExprNode::Arg(_)
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Real / imaginary decomposition
// ═══════════════════════════════════════════════════════════════════════════

/// The decomposition of one sub-expression.
///
/// `exact` is `true` when the decomposition contains no opaque
/// `Re`/`Im`/`Arg`/`Conjugate` nodes that were introduced *for this
/// sub-expression* — i.e. the real and imaginary parts are fully determined
/// in terms of real quantities.  Products only distribute over factors that
/// are exact; inexact factors are grouped under a single `Re(∏)`/`Im(∏)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Parts {
    /// Real part.
    pub(crate) re: ExprId,
    /// Imaginary part (a real quantity).
    pub(crate) im: ExprId,
    /// Whether the decomposition is free of opaque complex nodes.
    pub(crate) exact: bool,
}

/// Decompose `expr` into `(real_part, imaginary_part)` such that
/// `expr = real_part + imaginary_part * i`.
///
/// Symbols without a provable `Real` (or `Imaginary`) assumption yield
/// `Re(x)` / `Im(x)` nodes.
pub(crate) fn as_real_imag(arena: &mut Arena, expr: ExprId) -> (ExprId, ExprId) {
    let p = decompose(arena, expr);
    (p.re, p.im)
}

/// Full decomposition including the exactness flag.
pub(crate) fn decompose(arena: &mut Arena, expr: ExprId) -> Parts {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, Parts> = FxHashMap::default();
    let mut assumptions = AssumptionCache::new();

    for &id in &post_order {
        let parts = decompose_node(arena, id, &cache, &mut assumptions);
        cache.insert(id, parts);
    }

    cache.get(&expr).copied().unwrap_or(Parts {
        re: expr,
        im: arena.zero,
        exact: true,
    })
}

/// Look up the cached decomposition of a child; falls back to an opaque
/// decomposition (never happens in a correct post-order walk).
fn child_parts(arena: &mut Arena, cache: &FxHashMap<ExprId, Parts>, id: ExprId) -> Parts {
    if let Some(p) = cache.get(&id) {
        *p
    } else {
        opaque(arena, id)
    }
}

/// The opaque decomposition `(Re(id), Im(id))`.
fn opaque(arena: &mut Arena, id: ExprId) -> Parts {
    let re = arena.intern(ExprNode::Re(id));
    let im = arena.intern(ExprNode::Im(id));
    Parts {
        re,
        im,
        exact: false,
    }
}

/// The trivially real decomposition `(id, 0)`.
fn real_parts(arena: &Arena, id: ExprId) -> Parts {
    Parts {
        re: id,
        im: arena.zero,
        exact: true,
    }
}

/// `exp(x)` with `exp(0) = 1` folded (the arena constructor only interns).
fn fexp(arena: &mut Arena, x: ExprId) -> ExprId {
    if x == arena.zero {
        arena.one
    } else {
        arena.exp(x)
    }
}
/// `sin(x)` with `sin(0) = 0` folded.
fn fsin(arena: &mut Arena, x: ExprId) -> ExprId {
    if x == arena.zero {
        arena.zero
    } else {
        arena.sin(x)
    }
}
/// `cos(x)` with `cos(0) = 1` folded.
fn fcos(arena: &mut Arena, x: ExprId) -> ExprId {
    if x == arena.zero {
        arena.one
    } else {
        arena.cos(x)
    }
}
/// `sinh(x)` with `sinh(0) = 0` folded.
fn fsinh(arena: &mut Arena, x: ExprId) -> ExprId {
    if x == arena.zero {
        arena.zero
    } else {
        arena.sinh(x)
    }
}
/// `cosh(x)` with `cosh(0) = 1` folded.
fn fcosh(arena: &mut Arena, x: ExprId) -> ExprId {
    if x == arena.zero {
        arena.one
    } else {
        arena.cosh(x)
    }
}

/// Complex product `(a + bi)(c + di)`.
fn cmul(arena: &mut Arena, a: ExprId, b: ExprId, c: ExprId, d: ExprId) -> (ExprId, ExprId) {
    let ac = arena.mul(&[a, c]);
    let bd = arena.mul(&[b, d]);
    let ad = arena.mul(&[a, d]);
    let bc = arena.mul(&[b, c]);
    (arena.sub(ac, bd), arena.add(&[ad, bc]))
}

/// Complex reciprocal `1/(a + bi) = (a − bi)/(a² + b²)`.
fn cinv(arena: &mut Arena, a: ExprId, b: ExprId) -> (ExprId, ExprId) {
    let two = arena.int(2);
    let a2 = arena.pow(a, two);
    let b2 = arena.pow(b, two);
    let denom = arena.add(&[a2, b2]);
    let re = arena.div(a, denom);
    let neg_b = arena.neg(b);
    let im = arena.div(neg_b, denom);
    (re, im)
}

/// Extract a small non-zero integer from a numeric node.
fn small_int(arena: &Arena, id: ExprId) -> Option<i64> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    Some(n)
}

/// Largest integer power that is expanded via repeated complex
/// multiplication.  Larger exponents stay as opaque `Re`/`Im` nodes.
const MAX_EXPANDED_POWER: i64 = 32;

fn decompose_node(
    arena: &mut Arena,
    id: ExprId,
    cache: &FxHashMap<ExprId, Parts>,
    assumptions: &mut AssumptionCache,
) -> Parts {
    let node = arena.node(id).clone();

    // ── Intrinsically real nodes and anything the assumption system can
    //    prove real → (id, 0).
    if is_intrinsically_real(&node) || assumptions.query(arena, id, Props::REAL) == Some(true) {
        return real_parts(arena, id);
    }

    match node {
        ExprNode::ImaginaryUnit => Parts {
            re: arena.zero,
            im: arena.one,
            exact: true,
        },

        // Undefined quantities: both parts are NaN.
        ExprNode::ComplexInfinity | ExprNode::NaN => Parts {
            re: arena.nan,
            im: arena.nan,
            exact: true,
        },

        // Boolean and set-valued nodes have no complex structure; treat
        // them as inert.
        ExprNode::BoolTrue
        | ExprNode::BoolFalse
        | ExprNode::Gt(..)
        | ExprNode::Ge(..)
        | ExprNode::Eq_(..)
        | ExprNode::Ne(..)
        | ExprNode::And(_)
        | ExprNode::Or(_)
        | ExprNode::Not(_)
        | ExprNode::EmptySet
        | ExprNode::UniversalSet
        | ExprNode::Interval(..)
        | ExprNode::FiniteSet(_)
        | ExprNode::SetUnion(_)
        | ExprNode::SetIntersection(_)
        | ExprNode::SetComplement(..)
        | ExprNode::ConditionSet(..) => real_parts(arena, id),

        // ── Symbols: pure-imaginary assumption → (0, −i·x); otherwise opaque.
        ExprNode::Symbol(_) => {
            if assumptions.query(arena, id, Props::IMAGINARY) == Some(true) {
                let neg_i = arena.neg(arena.i_unit);
                let im = arena.mul(&[neg_i, id]);
                Parts {
                    re: arena.zero,
                    im,
                    exact: true,
                }
            } else {
                opaque(arena, id)
            }
        }

        // ── Add: ℝ-linear.
        ExprNode::Add(ref children) => {
            let mut res: Vec<ExprId> = Vec::with_capacity(children.len());
            let mut ims: Vec<ExprId> = Vec::with_capacity(children.len());
            let mut exact = true;
            for &c in children.iter() {
                let p = child_parts(arena, cache, c);
                res.push(p.re);
                ims.push(p.im);
                exact &= p.exact;
            }
            Parts {
                re: arena.add(&res),
                im: arena.add(&ims),
                exact,
            }
        }

        // ── Neg.
        ExprNode::Neg(inner) => {
            let p = child_parts(arena, cache, inner);
            Parts {
                re: arena.neg(p.re),
                im: arena.neg(p.im),
                exact: p.exact,
            }
        }

        // ── Mul: distribute over exact factors; group inexact factors.
        ExprNode::Mul(ref children) => {
            let mut acc_re = arena.one;
            let mut acc_im = arena.zero;
            let mut inexact: Vec<ExprId> = Vec::new();
            let mut single_inexact: Option<Parts> = None;
            for &c in children.iter() {
                let p = child_parts(arena, cache, c);
                if p.exact {
                    let (r, i) = cmul(arena, acc_re, acc_im, p.re, p.im);
                    acc_re = r;
                    acc_im = i;
                } else {
                    inexact.push(c);
                    single_inexact = Some(p);
                }
            }
            match inexact.len() {
                0 => Parts {
                    re: acc_re,
                    im: acc_im,
                    exact: true,
                },
                1 => {
                    // Use the factor's own (partially known) decomposition,
                    // e.g. re(2·exp(z)) = 2·exp(re z)·cos(im z).
                    let p = single_inexact.unwrap_or_else(|| opaque(arena, inexact[0]));
                    let (r, i) = cmul(arena, acc_re, acc_im, p.re, p.im);
                    Parts {
                        re: r,
                        im: i,
                        exact: false,
                    }
                }
                _ => {
                    // Several unknown factors: a single Re(∏)/Im(∏) is
                    // simpler than the fully distributed form.
                    let w = arena.mul(&inexact);
                    let p = opaque(arena, w);
                    let (r, i) = cmul(arena, acc_re, acc_im, p.re, p.im);
                    Parts {
                        re: r,
                        im: i,
                        exact: false,
                    }
                }
            }
        }

        // ── Pow.
        ExprNode::Pow(base, exp) => {
            let bp = child_parts(arena, cache, base);
            let ep = child_parts(arena, cache, exp);

            // Integer exponents with an exact base: repeated multiplication.
            if let Some(n) = small_int(arena, exp)
                && bp.exact
                && n.abs() <= MAX_EXPANDED_POWER
            {
                let mut acc_re = arena.one;
                let mut acc_im = arena.zero;
                for _ in 0..n.unsigned_abs() {
                    let (r, i) = cmul(arena, acc_re, acc_im, bp.re, bp.im);
                    acc_re = r;
                    acc_im = i;
                }
                if n < 0 {
                    let (r, i) = cinv(arena, acc_re, acc_im);
                    acc_re = r;
                    acc_im = i;
                }
                return Parts {
                    re: acc_re,
                    im: acc_im,
                    exact: true,
                };
            }

            // Principal square root of an exact numeric complex a + bi, b ≠ 0:
            //   √z = √((|z|+a)/2) + i·sign(b)·√((|z|−a)/2)
            if bp.exact
                && let Some(r) = arena.as_num(exp)
                && *r == Ratio::new(BigInt::from(1), BigInt::from(2))
                && let (Some(a), Some(b)) =
                    (arena.as_num(bp.re).cloned(), arena.as_num(bp.im).cloned())
                && !b.is_zero()
            {
                let a2b2 = &a * &a + &b * &b;
                let nid = arena.intern_num(a2b2);
                let a2b2_id = arena.intern(ExprNode::Num(nid));
                let modulus = arena.sqrt(a2b2_id);
                let a_id = bp.re;
                let half = arena.rational(1, 2);
                let sum = arena.add(&[modulus, a_id]);
                let diff = arena.sub(modulus, a_id);
                let re_sq = arena.mul(&[half, sum]);
                let im_sq = arena.mul(&[half, diff]);
                let re = arena.sqrt(re_sq);
                let im_abs = arena.sqrt(im_sq);
                let im = if b.is_negative() {
                    arena.neg(im_abs)
                } else {
                    im_abs
                };
                return Parts {
                    re,
                    im,
                    exact: true,
                };
            }

            // Positive real base, exact complex exponent c + di:
            //   b^(c+di) = b^c · (cos(d·ln b) + i·sin(d·ln b))
            if ep.exact
                && ep.im != arena.zero
                && assumptions.query(arena, base, Props::POSITIVE) == Some(true)
            {
                let ln_b = arena.ln(base);
                let theta = arena.mul(&[ep.im, ln_b]);
                let mag = arena.pow(base, ep.re);
                let cos_t = fcos(arena, theta);
                let sin_t = fsin(arena, theta);
                return Parts {
                    re: arena.mul(&[mag, cos_t]),
                    im: arena.mul(&[mag, sin_t]),
                    exact: true,
                };
            }

            // Branch cuts (non-integer powers of possibly negative or
            // complex bases): unevaluated.
            opaque(arena, id)
        }

        // ── exp(a + bi) = e^a (cos b + i sin b)
        ExprNode::Exp(inner) => {
            let p = child_parts(arena, cache, inner);
            if p.im == arena.zero {
                return Parts {
                    re: arena.exp(p.re),
                    im: arena.zero,
                    exact: p.exact,
                };
            }
            let ea = fexp(arena, p.re);
            let cb = fcos(arena, p.im);
            let sb = fsin(arena, p.im);
            Parts {
                re: arena.mul(&[ea, cb]),
                im: arena.mul(&[ea, sb]),
                exact: p.exact,
            }
        }

        // ── sin(a + bi) = sin a cosh b + i cos a sinh b
        ExprNode::Sin(inner) => {
            let p = child_parts(arena, cache, inner);
            if p.im == arena.zero {
                return Parts {
                    re: arena.sin(p.re),
                    im: arena.zero,
                    exact: p.exact,
                };
            }
            let sa = fsin(arena, p.re);
            let ca = fcos(arena, p.re);
            let chb = fcosh(arena, p.im);
            let shb = fsinh(arena, p.im);
            Parts {
                re: arena.mul(&[sa, chb]),
                im: arena.mul(&[ca, shb]),
                exact: p.exact,
            }
        }

        // ── cos(a + bi) = cos a cosh b − i sin a sinh b
        ExprNode::Cos(inner) => {
            let p = child_parts(arena, cache, inner);
            if p.im == arena.zero {
                return Parts {
                    re: arena.cos(p.re),
                    im: arena.zero,
                    exact: p.exact,
                };
            }
            let sa = fsin(arena, p.re);
            let ca = fcos(arena, p.re);
            let chb = fcosh(arena, p.im);
            let shb = fsinh(arena, p.im);
            let prod = arena.mul(&[sa, shb]);
            Parts {
                re: arena.mul(&[ca, chb]),
                im: arena.neg(prod),
                exact: p.exact,
            }
        }

        // ── sinh(a + bi) = sinh a cos b + i cosh a sin b
        ExprNode::Sinh(inner) => {
            let p = child_parts(arena, cache, inner);
            if p.im == arena.zero {
                return Parts {
                    re: arena.sinh(p.re),
                    im: arena.zero,
                    exact: p.exact,
                };
            }
            let sha = fsinh(arena, p.re);
            let cha = fcosh(arena, p.re);
            let cb = fcos(arena, p.im);
            let sb = fsin(arena, p.im);
            Parts {
                re: arena.mul(&[sha, cb]),
                im: arena.mul(&[cha, sb]),
                exact: p.exact,
            }
        }

        // ── cosh(a + bi) = cosh a cos b + i sinh a sin b
        ExprNode::Cosh(inner) => {
            let p = child_parts(arena, cache, inner);
            if p.im == arena.zero {
                return Parts {
                    re: arena.cosh(p.re),
                    im: arena.zero,
                    exact: p.exact,
                };
            }
            let sha = fsinh(arena, p.re);
            let cha = fcosh(arena, p.re);
            let cb = fcos(arena, p.im);
            let sb = fsin(arena, p.im);
            Parts {
                re: arena.mul(&[cha, cb]),
                im: arena.mul(&[sha, sb]),
                exact: p.exact,
            }
        }

        // ── tan(a + bi) = (sin 2a + i sinh 2b) / (cos 2a + cosh 2b)
        ExprNode::Tan(inner) => {
            let p = child_parts(arena, cache, inner);
            if p.im == arena.zero {
                return Parts {
                    re: arena.tan(p.re),
                    im: arena.zero,
                    exact: p.exact,
                };
            }
            if !p.exact {
                return opaque(arena, id);
            }
            let two = arena.int(2);
            let two_a = arena.mul(&[two, p.re]);
            let two_b = arena.mul(&[two, p.im]);
            let s2a = fsin(arena, two_a);
            let c2a = fcos(arena, two_a);
            let sh2b = fsinh(arena, two_b);
            let ch2b = fcosh(arena, two_b);
            let denom = arena.add(&[c2a, ch2b]);
            Parts {
                re: arena.div(s2a, denom),
                im: arena.div(sh2b, denom),
                exact: true,
            }
        }

        // ── tanh(a + bi) = (sinh 2a + i sin 2b) / (cosh 2a + cos 2b)
        ExprNode::Tanh(inner) => {
            let p = child_parts(arena, cache, inner);
            if p.im == arena.zero {
                return Parts {
                    re: arena.tanh(p.re),
                    im: arena.zero,
                    exact: p.exact,
                };
            }
            if !p.exact {
                return opaque(arena, id);
            }
            let two = arena.int(2);
            let two_a = arena.mul(&[two, p.re]);
            let two_b = arena.mul(&[two, p.im]);
            let sh2a = fsinh(arena, two_a);
            let ch2a = fcosh(arena, two_a);
            let s2b = fsin(arena, two_b);
            let c2b = fcos(arena, two_b);
            let denom = arena.add(&[ch2a, c2b]);
            Parts {
                re: arena.div(sh2a, denom),
                im: arena.div(s2b, denom),
                exact: true,
            }
        }

        // ── ln z = ln|z| + i·arg z  (principal branch, always valid)
        ExprNode::Ln(inner) => {
            let p = child_parts(arena, cache, inner);
            let arg_z = arg(arena, inner);
            let arg_is_opaque = matches!(arena.node(arg_z), ExprNode::Arg(_));
            let re = if p.exact && p.im != arena.zero {
                // ln|a + bi| = ½ ln(a² + b²)
                let two = arena.int(2);
                let a2 = arena.pow(p.re, two);
                let b2 = arena.pow(p.im, two);
                let sum = arena.add(&[a2, b2]);
                let ln_sum = arena.ln(sum);
                let half = arena.rational(1, 2);
                arena.mul(&[half, ln_sum])
            } else {
                let abs_z = arena.abs(inner);
                arena.ln(abs_z)
            };
            Parts {
                re,
                im: arg_z,
                exact: p.exact && !arg_is_opaque,
            }
        }

        // ── conjugate(w): (re w, −im w)
        ExprNode::Conjugate(inner) => {
            let p = child_parts(arena, cache, inner);
            Parts {
                re: p.re,
                im: arena.neg(p.im),
                exact: p.exact,
            }
        }

        // ── Everything else (functions with branch cuts, formal nodes,
        //    user functions, …) whose realness could not be established.
        _ => opaque(arena, id),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// re / im
// ═══════════════════════════════════════════════════════════════════════════

/// Canonical real part.  See the module docs for the evaluation rules.
pub(crate) fn re(arena: &mut Arena, z: ExprId) -> ExprId {
    decompose(arena, z).re
}

/// Canonical imaginary part (real-valued).
pub(crate) fn im(arena: &mut Arena, z: ExprId) -> ExprId {
    decompose(arena, z).im
}

/// Rewrite `z` as `re(z) + i·im(z)` with symbols expanded to
/// `Re(x) + i·Im(x)` unless they are provably real.
pub(crate) fn expand_complex(arena: &mut Arena, z: ExprId) -> ExprId {
    let p = decompose(arena, z);
    if p.im == arena.zero {
        return p.re;
    }
    let i_im = arena.mul(&[arena.i_unit, p.im]);
    arena.add(&[p.re, i_im])
}

// ═══════════════════════════════════════════════════════════════════════════
// conjugate
// ═══════════════════════════════════════════════════════════════════════════

/// Canonical complex conjugate.
///
/// Rules applied bottom-up:
/// * provably real → `z`; provably imaginary → `−z`; `i` → `−i`;
/// * distributes over `Add`, `Mul`, `Neg`, integer `Pow`;
/// * `b^e` with `b > 0` → `b^conj(e)`;
/// * commutes with real-analytic functions without branch cuts on ℂ∖ℝ:
///   `exp`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `Γ`, `erf`, `erfc`,
///   `ψ`, `ψ⁽ⁿ⁾`, `ζ`, `Si`;
/// * `conjugate(conjugate(z)) = z`; `Abs`/`Re`/`Im`/`Arg` are real;
/// * anything else (`ln`, non-integer powers, inverse functions, `W`,
///   `Ei`, `Ci`, `li`, user functions, formal nodes) → `Conjugate(z)` node.
pub(crate) fn conjugate(arena: &mut Arena, z: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, z);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut assumptions = AssumptionCache::new();

    for &id in &post_order {
        let c = conjugate_node(arena, id, &cache, &mut assumptions);
        cache.insert(id, c);
    }

    cache.get(&z).copied().unwrap_or(z)
}

fn conjugate_node(
    arena: &mut Arena,
    id: ExprId,
    cache: &FxHashMap<ExprId, ExprId>,
    assumptions: &mut AssumptionCache,
) -> ExprId {
    let node = arena.node(id).clone();
    let get = |c: ExprId| cache.get(&c).copied().unwrap_or(c);

    if is_intrinsically_real(&node) || assumptions.query(arena, id, Props::REAL) == Some(true) {
        return id;
    }

    match node {
        ExprNode::ImaginaryUnit => arena.neg(arena.i_unit),
        ExprNode::ComplexInfinity | ExprNode::NaN => id,

        ExprNode::Symbol(_) => {
            if assumptions.query(arena, id, Props::IMAGINARY) == Some(true) {
                arena.neg(id)
            } else {
                arena.intern(ExprNode::Conjugate(id))
            }
        }

        ExprNode::Add(ref children) => {
            let new: Vec<ExprId> = children.iter().map(|&c| get(c)).collect();
            arena.add(&new)
        }
        ExprNode::Mul(ref children) => {
            let new: Vec<ExprId> = children.iter().map(|&c| get(c)).collect();
            arena.mul(&new)
        }
        ExprNode::Neg(inner) => {
            let c = get(inner);
            arena.neg(c)
        }

        ExprNode::Pow(base, exp) => {
            if arena.as_num(exp).is_some_and(|r| r.is_integer()) {
                let cb = get(base);
                arena.pow(cb, exp)
            } else if assumptions.query(arena, base, Props::POSITIVE) == Some(true) {
                let ce = get(exp);
                arena.pow(base, ce)
            } else {
                arena.intern(ExprNode::Conjugate(id))
            }
        }

        // Real-analytic on ℝ, no branch cut in ℂ∖ℝ → Schwarz reflection.
        ExprNode::Exp(inner) => {
            let c = get(inner);
            arena.exp(c)
        }
        ExprNode::Sin(inner) => {
            let c = get(inner);
            arena.sin(c)
        }
        ExprNode::Cos(inner) => {
            let c = get(inner);
            arena.cos(c)
        }
        ExprNode::Tan(inner) => {
            let c = get(inner);
            arena.tan(c)
        }
        ExprNode::Sinh(inner) => {
            let c = get(inner);
            arena.sinh(c)
        }
        ExprNode::Cosh(inner) => {
            let c = get(inner);
            arena.cosh(c)
        }
        ExprNode::Tanh(inner) => {
            let c = get(inner);
            arena.tanh(c)
        }
        ExprNode::Gamma(inner) => {
            let c = get(inner);
            arena.gamma(c)
        }
        ExprNode::Digamma(inner) => {
            let c = get(inner);
            arena.digamma(c)
        }
        ExprNode::Erf(inner) => {
            let c = get(inner);
            arena.erf(c)
        }
        ExprNode::Erfc(inner) => {
            let c = get(inner);
            arena.erfc(c)
        }
        ExprNode::Zeta(inner) => {
            let c = get(inner);
            arena.zeta(c)
        }
        ExprNode::Si(inner) => {
            let c = get(inner);
            arena.si(c)
        }
        ExprNode::Polygamma(n, x) => {
            if assumptions.query(arena, n, Props::REAL) == Some(true) {
                let cx = get(x);
                arena.polygamma(n, cx)
            } else {
                arena.intern(ExprNode::Conjugate(id))
            }
        }

        // Involution.
        ExprNode::Conjugate(inner) => inner,

        // Branch cuts / unknown structure → unevaluated node.
        _ => arena.intern(ExprNode::Conjugate(id)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// arg
// ═══════════════════════════════════════════════════════════════════════════

/// Canonical principal argument `arg(z) ∈ (−π, π]`.
///
/// * provably positive → `0`; provably negative → `π`; zero/`NaN`/`z∞` → `NaN`;
/// * `i` → `π/2`;
/// * positive real factors are dropped: `arg(c·w) = arg(w)` for `c > 0`;
/// * `arg(exp(a + bi))` with `b` a rational multiple of `π` is reduced
///   into `(−π, π]`;
/// * when `z = a + bi` decomposes exactly with `b ≠ 0`, the result is
///   `atan2(b, a)` (which folds for numeric arguments);
/// * otherwise an [`ExprNode::Arg`] node.
pub(crate) fn arg(arena: &mut Arena, z: ExprId) -> ExprId {
    let node = arena.node(z).clone();
    match node {
        ExprNode::NaN | ExprNode::ComplexInfinity => return arena.nan,
        ExprNode::ImaginaryUnit => {
            let half = arena.rational(1, 2);
            return arena.mul(&[half, arena.pi]);
        }
        ExprNode::Num(nid) => {
            let r = arena.num(nid).clone();
            return if r.is_zero() {
                arena.nan
            } else if r.is_positive() {
                arena.zero
            } else {
                arena.pi
            };
        }
        ExprNode::Infinity => return arena.zero,
        ExprNode::NegInfinity => return arena.pi,
        _ => {}
    }

    let mut assumptions = AssumptionCache::new();
    if assumptions.query(arena, z, Props::POSITIVE) == Some(true) {
        return arena.zero;
    }
    if assumptions.query(arena, z, Props::NEGATIVE) == Some(true) {
        return arena.pi;
    }
    if assumptions.query(arena, z, Props::ZERO) == Some(true) {
        return arena.nan;
    }

    // Strip provably positive factors: arg(c·w) = arg(w) for c > 0.
    if let ExprNode::Mul(ref children) = node {
        let mut rest: Vec<ExprId> = Vec::new();
        for &c in children.iter() {
            if assumptions.query(arena, c, Props::POSITIVE) != Some(true) {
                rest.push(c);
            }
        }
        if rest.len() < children.len() {
            let w = arena.mul(&rest);
            return arg(arena, w);
        }
    }

    // arg(exp(a + bi)) = b reduced into (−π, π] when b is a rational
    // multiple of π (including b = 0).
    if let ExprNode::Exp(inner) = node {
        let p = decompose(arena, inner);
        if p.exact
            && let Some(k) = as_pi_multiple(arena, p.im)
        {
            let reduced = reduce_pi_multiple(k);
            let nid = arena.intern_num(reduced);
            let coeff = arena.intern(ExprNode::Num(nid));
            return arena.mul(&[coeff, arena.pi]);
        }
        return arena.intern(ExprNode::Arg(z));
    }

    let p = decompose(arena, z);
    if !p.exact {
        return arena.intern(ExprNode::Arg(z));
    }
    if p.im == arena.zero {
        // Real with unknown sign.
        return match assumptions.query(arena, p.re, Props::POSITIVE) {
            Some(true) => arena.zero,
            _ => match assumptions.query(arena, p.re, Props::NEGATIVE) {
                Some(true) => arena.pi,
                _ => arena.intern(ExprNode::Arg(z)),
            },
        };
    }
    if let Some(folded) = crate::transforms::eval::eval_atan2(arena, p.im, p.re) {
        folded
    } else {
        arena.atan2(p.im, p.re)
    }
}

/// If `id` is `k·π` for a rational `k` (or the number `0`), return `k`.
fn as_pi_multiple(arena: &Arena, id: ExprId) -> Option<Ratio<BigInt>> {
    if id == arena.pi() {
        return Some(Ratio::from_integer(BigInt::from(1)));
    }
    if let Some(r) = arena.as_num(id)
        && r.is_zero()
    {
        return Some(Ratio::zero());
    }
    if let ExprNode::Mul(children) = arena.node(id)
        && children.len() == 2
        && children[1] == arena.pi()
        && let ExprNode::Num(nid) = arena.node(children[0])
    {
        return Some(arena.num(*nid).clone());
    }
    None
}

/// Reduce `k` (in units of π) into `(−1, 1]`.
fn reduce_pi_multiple(k: Ratio<BigInt>) -> Ratio<BigInt> {
    let two = Ratio::from_integer(BigInt::from(2));
    let one = Ratio::from_integer(BigInt::from(1));
    // k mod 2 into [0, 2)
    let mut m = &k - (&k / &two).floor() * &two;
    if m > one {
        m -= &two;
    }
    m
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::assumptions::Assumptions;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn real_sym(a: &mut Arena, name: &str) -> ExprId {
        let id = a.symbol(name);
        if let ExprNode::Symbol(sid) = a.node(id).clone() {
            let mut asm = Assumptions::default();
            asm.assert_true(Props::REAL);
            a.set_symbol_assumptions(sid, asm);
        }
        id
    }
    fn positive_sym(a: &mut Arena, name: &str) -> ExprId {
        let id = a.symbol(name);
        if let ExprNode::Symbol(sid) = a.node(id).clone() {
            let mut asm = Assumptions::default();
            asm.assert_true(Props::POSITIVE);
            a.set_symbol_assumptions(sid, asm);
        }
        id
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn real_number() {
        let mut a = Arena::new();
        let five = a.int(5);
        let (re, im) = as_real_imag(&mut a, five);
        assert_eq!(display(&a, re), "5");
        assert_eq!(display(&a, im), "0");
    }

    #[test]
    fn imaginary_unit() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let (re, im) = as_real_imag(&mut a, i);
        assert_eq!(display(&a, re), "0");
        assert_eq!(display(&a, im), "1");
    }

    #[test]
    fn two_plus_three_i() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);
        let three_i = a.mul(&[three, a.i_unit]);
        let expr = a.add(&[two, three_i]);
        let (re, im) = as_real_imag(&mut a, expr);
        assert_eq!(display(&a, re), "2");
        assert_eq!(display(&a, im), "3");
    }

    #[test]
    fn i_squared() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let two = a.int(2);
        let i_sq = a.pow(i, two); // canonicalizes to -1
        let (re, im) = as_real_imag(&mut a, i_sq);
        assert_eq!(display(&a, re), "-1");
        assert_eq!(display(&a, im), "0");
    }

    #[test]
    fn product_a_plus_bi_times_c_plus_di() {
        let mut a = Arena::new();
        let i = a.i_unit;
        // (1+2i)*(3+4i) = (3-8) + (4+6)i = -5 + 10i
        let one = a.int(1);
        let two = a.int(2);
        let three = a.int(3);
        let four = a.int(4);
        let two_i = a.mul(&[two, i]);
        let z1 = a.add(&[one, two_i]);
        let four_i = a.mul(&[four, i]);
        let z2 = a.add(&[three, four_i]);
        let product = a.mul(&[z1, z2]);
        let (re, im) = as_real_imag(&mut a, product);
        let re = crate::transforms::expand::expand(&mut a, re);
        let im = crate::transforms::expand::expand(&mut a, im);
        assert_eq!(display(&a, re), "-5");
        assert_eq!(display(&a, im), "10");
    }

    #[test]
    fn exp_of_i_x_real() {
        let mut a = Arena::new();
        let x = real_sym(&mut a, "x");
        let i = a.i_unit;
        let ix = a.mul(&[i, x]);
        let expr = a.exp(ix);
        let (re, im) = as_real_imag(&mut a, expr);
        assert_eq!(display(&a, re), "cos(x)");
        assert_eq!(display(&a, im), "sin(x)");
    }

    #[test]
    fn unknown_symbol_is_not_assumed_real() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let (re_x, im_x) = as_real_imag(&mut a, x);
        assert_eq!(display(&a, re_x), "re(x)");
        assert_eq!(display(&a, im_x), "im(x)");
        assert!(matches!(a.node(re_x), ExprNode::Re(_)));
        assert!(matches!(a.node(im_x), ExprNode::Im(_)));
    }

    #[test]
    fn real_symbol_is_real() {
        let mut a = Arena::new();
        let x = real_sym(&mut a, "x");
        assert_eq!(re(&mut a, x), x);
        assert_eq!(im(&mut a, x), a.zero);
        assert_eq!(conjugate(&mut a, x), x);
    }

    #[test]
    fn re_im_linear_over_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = real_sym(&mut a, "y");
        let i = a.i_unit;
        let iy = a.mul(&[i, y]);
        let expr = a.add(&[x, iy, a.one]);
        let r = re(&mut a, expr);
        let m = im(&mut a, expr);
        assert_eq!(display(&a, r), "re(x) + 1");
        assert_eq!(display(&a, m), "y + im(x)");
    }

    #[test]
    fn re_of_i_times_x_is_neg_im() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ix = a.mul(&[a.i_unit, x]);
        let r = re(&mut a, ix);
        let m = im(&mut a, ix);
        assert_eq!(display(&a, r), "-im(x)");
        assert_eq!(display(&a, m), "re(x)");
    }

    #[test]
    fn re_of_scaled_symbol() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let e = a.mul(&[three, x]);
        let r = re(&mut a, e);
        assert_eq!(display(&a, r), "3*re(x)");
    }

    #[test]
    fn product_of_two_unknowns_stays_single_node() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let xy = a.mul(&[x, y]);
        let r = re(&mut a, xy);
        assert_eq!(display(&a, r), "re(x*y)");
    }

    #[test]
    fn idempotence_and_interplay() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let rx = re(&mut a, x);
        let ix = im(&mut a, x);
        let cx = conjugate(&mut a, x);
        assert_eq!(re(&mut a, rx), rx, "re(re z) = re z");
        assert_eq!(im(&mut a, rx), a.zero, "im(re z) = 0");
        assert_eq!(re(&mut a, ix), ix, "re(im z) = im z");
        assert_eq!(re(&mut a, cx), rx, "re(conj z) = re z");
        let neg_ix = a.neg(ix);
        assert_eq!(im(&mut a, cx), neg_ix, "im(conj z) = -im z");
        assert_eq!(conjugate(&mut a, cx), x, "conj(conj z) = z");
    }

    #[test]
    fn conjugate_distributes_and_commutes() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let i = a.i_unit;
        // conj(x + 2i) = conj(x) - 2i
        let two = a.int(2);
        let two_i = a.mul(&[two, i]);
        let z = a.add(&[x, two_i]);
        let cz = conjugate(&mut a, z);
        let cx = a.intern(ExprNode::Conjugate(x));
        let expected = a.sub(cx, two_i);
        assert_eq!(cz, expected);

        // conj(sin(x)) = sin(conj(x))
        let sx = a.sin(x);
        let csx = conjugate(&mut a, sx);
        let expected = a.sin(cx);
        assert_eq!(csx, expected);

        // conj(x^3) = conj(x)^3
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let cx3 = conjugate(&mut a, x3);
        let expected = a.pow(cx, three);
        assert_eq!(cx3, expected);

        // conj(ln(x)) stays unevaluated (branch cut)
        let lx = a.ln(x);
        let clx = conjugate(&mut a, lx);
        assert!(matches!(a.node(clx), ExprNode::Conjugate(_)));

        // conj(sqrt(x)) stays unevaluated
        let sq = a.sqrt(x);
        let csq = conjugate(&mut a, sq);
        assert!(matches!(a.node(csq), ExprNode::Conjugate(_)));
    }

    #[test]
    fn conjugate_of_i_and_numbers() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let neg_i = a.neg(i);
        assert_eq!(conjugate(&mut a, i), neg_i);
        let five = a.int(5);
        assert_eq!(conjugate(&mut a, five), five);
        // conj(3 + 4i) = 3 - 4i
        let three = a.int(3);
        let four = a.int(4);
        let four_i = a.mul(&[four, i]);
        let z = a.add(&[three, four_i]);
        let cz = conjugate(&mut a, z);
        let expected = a.sub(three, four_i);
        assert_eq!(cz, expected);
    }

    #[test]
    fn arg_special_values() {
        let mut a = Arena::new();
        let one = a.int(1);
        assert_eq!(arg(&mut a, one), a.zero);
        let neg_two = a.int(-2);
        assert_eq!(arg(&mut a, neg_two), a.pi);
        let i = a.i_unit;
        let arg_i = arg(&mut a, i);
        assert_eq!(display(&a, arg_i), "1/2*pi");
        let zero = a.zero;
        assert_eq!(arg(&mut a, zero), a.nan);
        // arg(1 + i) = pi/4
        let z = a.add(&[one, i]);
        let az = arg(&mut a, z);
        assert_eq!(display(&a, az), "1/4*pi");
        // arg(-1 + i) = 3pi/4
        let neg_one = a.neg_one;
        let z2 = a.add(&[neg_one, i]);
        let az2 = arg(&mut a, z2);
        assert_eq!(display(&a, az2), "3/4*pi");
    }

    #[test]
    fn arg_with_assumptions() {
        let mut a = Arena::new();
        let p = positive_sym(&mut a, "p");
        assert_eq!(arg(&mut a, p), a.zero);
        let neg_p = a.neg(p);
        assert_eq!(arg(&mut a, neg_p), a.pi);
        // Unknown-sign real → Arg node
        let x = real_sym(&mut a, "x");
        let ax = arg(&mut a, x);
        assert!(matches!(a.node(ax), ExprNode::Arg(_)));
        // Unknown complex → Arg node
        let z = sym(&mut a, "z");
        let az = arg(&mut a, z);
        assert!(matches!(a.node(az), ExprNode::Arg(_)));
        // arg(3*z) = arg(z)
        let three = a.int(3);
        let three_z = a.mul(&[three, z]);
        assert_eq!(arg(&mut a, three_z), az);
        // arg(x + i*y) for real x,y = atan2(y, x)
        let y = real_sym(&mut a, "y");
        let iy = a.mul(&[a.i_unit, y]);
        let w = a.add(&[x, iy]);
        let aw = arg(&mut a, w);
        assert_eq!(display(&a, aw), "atan2(y, x)");
    }

    #[test]
    fn arg_of_exp_reduces_pi_multiple() {
        let mut a = Arena::new();
        // arg(exp(i*5pi/2)) = pi/2
        let k = a.rational(5, 2);
        let kpi = a.mul(&[k, a.pi]);
        let ikpi = a.mul(&[a.i_unit, kpi]);
        let e = a.exp(ikpi);
        let r = arg(&mut a, e);
        assert_eq!(display(&a, r), "1/2*pi");
        // arg(exp(i*pi)) = pi (boundary maps to +pi)
        let ipi = a.mul(&[a.i_unit, a.pi]);
        let e2 = a.exp(ipi);
        let r2 = arg(&mut a, e2);
        assert_eq!(r2, a.pi);
        // arg(exp(-i*pi)) = pi as well
        let neg_ipi = a.neg(ipi);
        let e3 = a.exp(neg_ipi);
        let r3 = arg(&mut a, e3);
        assert_eq!(r3, a.pi);
    }

    #[test]
    fn pow_of_complex_number_expands() {
        let mut a = Arena::new();
        // (1 + i)^2 = 2i
        let one = a.int(1);
        let z = a.add(&[one, a.i_unit]);
        let two = a.int(2);
        let z2 = a.pow(z, two);
        let (re, im) = as_real_imag(&mut a, z2);
        let re = crate::transforms::expand::expand(&mut a, re);
        let im = crate::transforms::expand::expand(&mut a, im);
        assert_eq!(display(&a, re), "0");
        assert_eq!(display(&a, im), "2");
        // 1/(1 + i) = 1/2 - i/2
        let inv = a.pow(z, a.neg_one);
        let (re, im) = as_real_imag(&mut a, inv);
        let re = crate::transforms::eval::eval(&mut a, re);
        let im = crate::transforms::eval::eval(&mut a, im);
        assert_eq!(display(&a, re), "1/2");
        assert_eq!(display(&a, im), "-1/2");
    }

    #[test]
    fn sqrt_of_unknown_symbol_is_opaque() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let s = a.sqrt(x);
        let r = re(&mut a, s);
        assert!(matches!(a.node(r), ExprNode::Re(_)));
        // But sqrt of a positive symbol is real.
        let p = positive_sym(&mut a, "p");
        let sp = a.sqrt(p);
        assert_eq!(re(&mut a, sp), sp);
        assert_eq!(im(&mut a, sp), a.zero);
    }

    #[test]
    fn ln_of_negative_number() {
        let mut a = Arena::new();
        let neg_two = a.int(-2);
        let l = a.ln(neg_two);
        let (re_l, im_l) = as_real_imag(&mut a, l);
        assert_eq!(display(&a, re_l), "ln(2)");
        assert_eq!(im_l, a.pi);
    }

    #[test]
    fn expand_complex_of_symbol() {
        let mut a = Arena::new();
        let z = sym(&mut a, "z");
        let e = expand_complex(&mut a, z);
        assert_eq!(display(&a, e), "im(z)*I + re(z)");
        let x = real_sym(&mut a, "x");
        assert_eq!(expand_complex(&mut a, x), x);
    }

    #[test]
    fn reduce_pi_multiple_ranges() {
        let r = |p: i64, q: i64| Ratio::new(BigInt::from(p), BigInt::from(q));
        assert_eq!(reduce_pi_multiple(r(5, 2)), r(1, 2));
        assert_eq!(reduce_pi_multiple(r(1, 1)), r(1, 1));
        assert_eq!(reduce_pi_multiple(r(-1, 1)), r(1, 1));
        assert_eq!(reduce_pi_multiple(r(3, 1)), r(1, 1));
        assert_eq!(reduce_pi_multiple(r(-3, 2)), r(1, 2));
        assert_eq!(reduce_pi_multiple(r(7, 4)), r(-1, 4));
        assert_eq!(reduce_pi_multiple(r(0, 1)), r(0, 1));
    }
}
