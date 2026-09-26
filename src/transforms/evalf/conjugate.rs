//! Exact realness from conjugate symmetry.
//!
//! A sum or product whose non-real terms pair up into complex conjugates is
//! real, with an imaginary part that is exactly 0 — not 0 to within its
//! error.  Cardano's formula in the irreducible case, which the integrator
//! produces for the roots of a quartic, is the typical source:
//! `cbrt(65/1024 + 21·√87/1024·i) + cbrt(65/1024 − 21·√87/1024·i)` is real,
//! but its computed imaginary part is a rounding residue, and the square
//! root of such a real negative number sits on its branch cut with an
//! undecidable side (see `accuracy.rs`).  Numerics cannot see that the
//! residue stands for an exact 0; the structure can.
//!
//! Every evaluated node gets a signature pair `(S, S̄)`: `S` identifies the
//! expression up to the order of the terms of a sum and the factors of a
//! product (and the representation of a monomial's coefficient and its
//! powers of `i`), and `S̄` is the signature its complex conjugate would
//! have.  `S̄` follows the rules of conjugation:
//!
//! * `conj(a + b) = conj a + conj b`, `conj(c·iᵏ·Π fⱼ) = c·(−1)ᵏ·iᵏ·Π conj fⱼ`
//!   for a rational `c`;
//! * `conj(bⁿ) = (conj b)ⁿ` for an integer `n`, and `conj(f(z)) =
//!   f(conj z)` for `exp`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`;
//! * for `ln`, `√`, other powers and the inverse functions only when the
//!   argument is certainly off the function's branch cut, where the
//!   principal branch breaks the symmetry (`conj √−1 = −i ≠ √(conj −1)`);
//! * a node whose value is exactly real is its own conjugate; any other node
//!   gets a signature no other node can match.
//!
//! Two terms with `S(y) = S̄(x)` and `S(x) = S̄(y)` are then conjugates by
//! construction (up to a 64-bit hash collision, which the computed
//! imaginary part — it must lie within its error of 0 — also guards
//! against).

use std::hash::{Hash, Hasher};

use astro_float::BigFloat;
use rustc_hash::{FxHashMap, FxHasher};

use super::accuracy::{self, Bound};
use crate::base::arena::Arena;
use crate::base::bigcomplex::Complex;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;

/// The signatures of a node (`S`) and of its complex conjugate (`S̄`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Sig {
    /// `S`.
    own: u64,
    /// `S̄`.
    conj: u64,
}

fn sig(own: u64, conj: u64) -> Sig {
    Sig { own, conj }
}

/// A node seen as a monomial `coef·i^k·Π factors` (the factors' signatures).
#[derive(Clone, Debug)]
pub(super) struct Mono {
    coef: Q,
    k: u32,
    factors: Vec<Sig>,
}

/// Signatures and monomial views of the evaluated nodes.
#[derive(Default)]
pub(super) struct Sigs {
    sigs: FxHashMap<ExprId, Sig>,
    monos: FxHashMap<ExprId, Mono>,
}

const MUL: u64 = 0x6d75_6c00;
const ADD: u64 = 0x6164_6400;
const UNIQUE: u64 = 0x756e_6971;

fn hash_of(parts: impl Hash) -> u64 {
    let mut h = FxHasher::default();
    parts.hash(&mut h);
    h.finish()
}

/// The signature of a monomial and of its conjugate.
fn mono_sig(m: &Mono) -> Sig {
    let mut s: Vec<u64> = m.factors.iter().map(|f| f.own).collect();
    let mut c: Vec<u64> = m.factors.iter().map(|f| f.conj).collect();
    s.sort_unstable();
    c.sort_unstable();
    // The coefficient by its (reduced) numerator and denominator: num-rational
    // hashes a `Ratio` through its continued fraction, recursively, which
    // overflows the stack for an 80,000-bit rational (and the decimal
    // strings are quadratic to build).
    let coef_conj = if m.k % 2 == 1 {
        -m.coef.clone()
    } else {
        m.coef.clone()
    };
    sig(
        hash_of((MUL, m.coef.numer(), m.coef.denom(), m.k, s)),
        hash_of((MUL, coef_conj.numer(), coef_conj.denom(), m.k, c)),
    )
}

/// Is the part `x ± 2^e` certainly nonzero?
fn certainly_nonzero(x: &BigFloat, e: accuracy::ErrExp) -> bool {
    !accuracy::part_contains_zero(x, e)
}

/// Is `z ± b` certainly off the cut `(−∞, 0]`?
fn off_negative_axis(z: &Complex, b: Bound) -> bool {
    certainly_nonzero(&z.1, b.im) || (z.0.is_positive() && certainly_nonzero(&z.0, b.re))
}

impl Sigs {
    pub(super) fn new() -> Sigs {
        Sigs::default()
    }

    fn get(&self, id: ExprId) -> Option<Sig> {
        self.sigs.get(&id).copied()
    }

    /// The monomial view of `id` (a plain factor when it is none).
    fn mono_of(&self, id: ExprId) -> Option<Mono> {
        if let Some(m) = self.monos.get(&id) {
            return Some(m.clone());
        }
        Some(Mono {
            coef: Q::from_integer(1.into()),
            k: 0,
            factors: vec![self.get(id)?],
        })
    }

    /// Record the signature of node `id`, evaluated to `value ± bound`, from
    /// its children's (see the module documentation).
    pub(super) fn record(
        &mut self,
        arena: &Arena,
        id: ExprId,
        value: &Complex,
        bound: Bound,
        cache: &FxHashMap<ExprId, Complex>,
        errs: &FxHashMap<ExprId, Bound>,
    ) {
        let real = accuracy::exactly_real(value, bound);
        let unique = || hash_of((UNIQUE, id));
        let child_bound = |c: &ExprId| errs.get(c).copied().unwrap_or(Bound::UNKNOWN);
        let off_cut = |c: &ExprId| {
            cache
                .get(c)
                .is_some_and(|v| off_negative_axis(v, child_bound(c)))
        };
        let across_nonzero = |c: &ExprId, imaginary_axis: bool| {
            cache.get(c).is_some_and(|v| {
                let b = child_bound(c);
                if imaginary_axis {
                    certainly_nonzero(&v.0, b.re)
                } else {
                    certainly_nonzero(&v.1, b.im)
                }
            })
        };
        let node = arena.node(id);
        // Monomials: numbers, i, products, negations.
        let mono: Option<Mono> = match node {
            ExprNode::Num(n) => Some(Mono {
                coef: arena.num(*n).clone(),
                k: 0,
                factors: Vec::new(),
            }),
            ExprNode::ImaginaryUnit => Some(Mono {
                coef: Q::from_integer(1.into()),
                k: 1,
                factors: Vec::new(),
            }),
            ExprNode::Neg(c) => self.mono_of(*c).map(|m| Mono { coef: -m.coef, ..m }),
            ExprNode::Mul(children) => {
                let mut acc = Mono {
                    coef: Q::from_integer(1.into()),
                    k: 0,
                    factors: Vec::new(),
                };
                let mut ok = true;
                for c in children.iter() {
                    match self.mono_of(*c) {
                        Some(m) => {
                            acc.coef *= m.coef;
                            acc.k += m.k;
                            acc.factors.extend(m.factors);
                        }
                        None => ok = false,
                    }
                }
                ok.then_some(acc)
            }
            _ => None,
        };
        let s = if let Some(m) = mono {
            let s = mono_sig(&m);
            self.monos.insert(id, m);
            s
        } else {
            let kids = node.children();
            let child_sigs: Option<Vec<Sig>> = kids.iter().map(|c| self.get(*c)).collect();
            let tag = std::mem::discriminant(node);
            let symmetric = match node {
                ExprNode::Add(_) => true,
                ExprNode::Pow(_, e) if arena.as_num(*e).is_some_and(|q| q.is_integer()) => true,
                ExprNode::Pow(b, _) | ExprNode::Ln(b) => off_cut(b),
                ExprNode::Exp(_)
                | ExprNode::Sin(_)
                | ExprNode::Cos(_)
                | ExprNode::Tan(_)
                | ExprNode::Sinh(_)
                | ExprNode::Cosh(_)
                | ExprNode::Tanh(_) => true,
                ExprNode::Asin(c) | ExprNode::Acos(c) | ExprNode::Atanh(c) | ExprNode::Acosh(c) => {
                    across_nonzero(c, false)
                }
                ExprNode::Atan(c) | ExprNode::Asinh(c) => across_nonzero(c, true),
                _ => false,
            };
            match child_sigs {
                Some(cs) if symmetric => {
                    let mut s: Vec<u64> = cs.iter().map(|c| c.own).collect();
                    let mut c: Vec<u64> = cs.iter().map(|c| c.conj).collect();
                    if matches!(node, ExprNode::Add(_)) {
                        s.sort_unstable();
                        c.sort_unstable();
                        sig(hash_of((ADD, s)), hash_of((ADD, c)))
                    } else {
                        let tag = hash_of(tag);
                        sig(hash_of((tag, s)), hash_of((tag, c)))
                    }
                }
                Some(cs) => {
                    let s: Vec<u64> = cs.iter().map(|c| c.own).collect();
                    let h = hash_of((hash_of(tag), s));
                    sig(h, if real { h } else { unique() })
                }
                None => {
                    let h = unique();
                    sig(h, if real { h } else { hash_of((UNIQUE, id, 1u8)) })
                }
            }
        };
        // An exactly real node is its own conjugate.
        let s = if real { sig(s.own, s.own) } else { s };
        self.sigs.insert(id, s);
    }

    /// A seeded (bound) variable: a leaf.
    pub(super) fn record_leaf(&mut self, id: ExprId, value: &Complex, bound: Bound) {
        let h = hash_of((UNIQUE, id, 2u8));
        let conj = if accuracy::exactly_real(value, bound) {
            h
        } else {
            hash_of((UNIQUE, id, 3u8))
        };
        self.sigs.insert(id, sig(h, conj));
    }

    /// Do the children of a sum or product that are not exactly real pair
    /// up into complex conjugates (see the module documentation)?
    pub(super) fn conjugates_pair_up(
        &self,
        children: &[ExprId],
        cache: &FxHashMap<ExprId, Complex>,
        errs: &FxHashMap<ExprId, Bound>,
    ) -> bool {
        let mut open: Vec<Sig> = Vec::new();
        for c in children {
            let (Some(v), Some(s)) = (cache.get(c), self.get(*c)) else {
                return false;
            };
            let b = errs.get(c).copied().unwrap_or(Bound::UNKNOWN);
            if accuracy::exactly_real(v, b) {
                continue;
            }
            if b.is_unknown() {
                return false;
            }
            if let Some(pos) = open.iter().position(|o| o.own == s.conj && o.conj == s.own) {
                open.swap_remove(pos);
            } else {
                open.push(s);
            }
        }
        open.is_empty()
    }
}
