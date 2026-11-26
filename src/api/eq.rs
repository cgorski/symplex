//! Symbolic equation type.
//!
//! [`Equation`] represents `lhs = rhs` and provides convenience methods
//! for solving, substitution, rearrangement, and display. Internally,
//! solving converts to `lhs - rhs = 0` and delegates to the existing solver.
//!
//! The type is named `Equation` (not `Eq`) to avoid conflict with
//! `std::cmp::Eq`.
//!
//! # Arithmetic
//!
//! Applying an operation to both sides is written with the ordinary
//! operators: `&eq + 1`, `&eq * &x`, `-&eq`, `&eq / 2.0`.  The right-hand
//! operand may be any [`ToEx`] value (integers, `f64` — converted
//! exactly — `BigInt`, `Ratio`, `Ex`, `&Ex`).  Two equations can also be
//! combined side-by-side: `&eq1 + &eq2` is `lhs1 + lhs2 = rhs1 + rhs2`.
//!
//! ```
//! use symplex::prelude::*;
//!
//! let ctx = Context::new();
//! let x = ctx.symbol("x");
//! let eq = Equation::new(&x * 2 + 3, ctx.int(11));   // 2x + 3 = 11
//! let eq = (&eq - 3) / 2;                             // x = 4
//! assert_eq!(format!("{eq}"), "x = 4");
//! assert_eq!(eq.solve(&x).unwrap(), vec![ctx.int(4)]);
//! ```

use std::fmt;
use std::ops;

use crate::api::expr::Ex;
use crate::base::assumptions::Props;
use crate::base::errors::SymplexError;
use crate::base::truth::Truth;

// `ToEx` / `Scalar` live in the (crate-private) `expr_ops` module; the
// `Equation` operators are generic over `ToEx`, so this public module is
// where the traits are reachable from outside the crate.
pub use crate::api::expr_ops::{Scalar, ToEx};

/// A symbolic equation `lhs = rhs`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::eq::Equation;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let eq = Equation::new(&x + 1, ctx.int(5));
/// assert_eq!(format!("{eq}"), "x + 1 = 5");
/// ```
#[derive(Clone)]
pub struct Equation {
    /// Left-hand side of the equation.
    pub lhs: Ex,
    /// Right-hand side of the equation.
    pub rhs: Ex,
}

impl Equation {
    /// Create a new equation `lhs = rhs`.
    ///
    /// # Panics
    ///
    /// Panics if the two sides belong to different contexts.
    pub fn new(lhs: Ex, rhs: Ex) -> Self {
        let _ = lhs.checked_id(&rhs);
        Self { lhs, rhs }
    }

    /// The left-hand side.
    #[must_use]
    pub fn lhs(&self) -> &Ex {
        &self.lhs
    }

    /// The right-hand side.
    #[must_use]
    pub fn rhs(&self) -> &Ex {
        &self.rhs
    }

    /// The equation `rhs = lhs`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let eq = Equation::new(ctx.int(5), &x + 1).swap();
    /// assert_eq!(format!("{eq}"), "x + 1 = 5");
    /// ```
    #[must_use]
    pub fn swap(&self) -> Equation {
        Equation {
            lhs: self.rhs.clone(),
            rhs: self.lhs.clone(),
        }
    }

    /// Apply the same function to both sides.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// // Real x: over ℂ, ln(eˣ) = x only for Im x ∈ (−π, π].
    /// let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    /// let eq = Equation::new(x.exp(), ctx.int(2)).apply(|s| s.ln());
    /// assert_eq!(format!("{}", eq.simplify()), "x = ln(2)");
    /// ```
    #[must_use]
    pub fn apply(&self, f: impl Fn(&Ex) -> Ex) -> Equation {
        Equation {
            lhs: f(&self.lhs),
            rhs: f(&self.rhs),
        }
    }

    /// Move everything to the left: the equation `lhs - rhs = 0`.
    ///
    /// For the bare expression `lhs - rhs` use [`to_expr`](Self::to_expr)
    /// (or the [`ZeroForm`](crate::polysys::ZeroForm) trait, which the
    /// solvers accept).
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let eq = Equation::new(x.powi(2), &x + 2).to_zero_equation();
    /// assert_eq!(format!("{eq}"), "x^2 - x - 2 = 0");
    /// ```
    #[must_use]
    pub fn to_zero_equation(&self) -> Equation {
        Equation {
            lhs: &self.lhs - &self.rhs,
            rhs: self.lhs.context().zero(),
        }
    }

    /// Solve this equation for `var`, returning the solution values.
    ///
    /// Internally computes `lhs - rhs` and solves for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::eq::Equation;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let eq = Equation::new(&x + 1, ctx.int(5));
    /// let roots = eq.solve(&x).unwrap();
    /// assert_eq!(format!("{}", roots[0]), "4");
    /// ```
    pub fn solve(&self, var: &Ex) -> Result<Vec<Ex>, SymplexError> {
        self.to_expr().solve(var)
    }

    /// Solve for `var`, returning each solution as an equation `var = value`.
    ///
    /// This is the form you want when the result feeds further
    /// substitution or display; [`solve`](Self::solve) returns the bare
    /// values.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let sols = Equation::new(x.powi(2), ctx.int(4)).solve_for(&x).unwrap();
    /// let mut shown: Vec<String> = sols.iter().map(|e| format!("{e}")).collect();
    /// shown.sort();
    /// assert_eq!(shown, vec!["x = -2", "x = 2"]);
    /// ```
    pub fn solve_for(&self, var: &Ex) -> Result<Vec<Equation>, SymplexError> {
        Ok(self
            .solve(var)?
            .into_iter()
            .map(|v| Equation::new(var.clone(), v))
            .collect())
    }

    /// Solve this equation, returning empty vec on failure.
    pub fn solve_or_empty(&self, var: &Ex) -> Vec<Ex> {
        self.to_expr().solve_or_empty(var)
    }

    /// Substitute a variable in both sides.
    #[must_use]
    pub fn subs(&self, old: &Ex, new: &Ex) -> Equation {
        Equation {
            lhs: self.lhs.subs(old, new),
            rhs: self.rhs.subs(old, new),
        }
    }

    /// Substitute an integer value in both sides.
    #[must_use]
    pub fn subs_i64(&self, old: &Ex, val: i64) -> Equation {
        Equation {
            lhs: self.lhs.subs_i64(old, val),
            rhs: self.rhs.subs_i64(old, val),
        }
    }

    /// Simplify both sides.
    #[must_use]
    pub fn simplify(&self) -> Equation {
        self.apply(Ex::simplify)
    }

    /// Expand both sides.
    #[must_use]
    pub fn expand(&self) -> Equation {
        self.apply(Ex::expand)
    }

    /// Evaluate both sides.
    #[must_use]
    pub fn eval(&self) -> Equation {
        self.apply(Ex::eval)
    }

    /// Check if the equation is satisfied: are the two sides mathematically
    /// equal?
    ///
    /// Three-valued via [`Ex::equals`]: `Some(true)` when the sides are
    /// proved equal, `Some(false)` when they are proved different wherever
    /// they are defined (`x = x + 1`), `None` otherwise — in particular for
    /// an equation that holds for some values of its symbols and fails for
    /// others (`x = 3`, `x = y`), which
    /// [`is_identity`](Self::is_identity) answers with `Some(false)`.
    #[must_use]
    pub fn is_satisfied(&self) -> Option<bool> {
        self.lhs.equals(&self.rhs)
    }

    /// Does the equation hold for all values of its symbols?
    ///
    /// * `Some(true)` when [`Ex::equals`] proves the two sides equal
    ///   (`lhs − rhs` simplifies to zero).
    /// * `Some(false)` when it proves them different (they differ by a
    ///   nonzero constant), or when the equation fails at a point: values
    ///   of the symbols, each proved to lie in its symbol's declared
    ///   domain, at which both sides have a finite value and the two
    ///   values are certified different (`x = 3` fails at `x = 1/2`).  Six
    ///   fixed points are tried, built from a short list of values: the
    ///   fractions `1/2`, `−2/3`, `3/2`, small integers, the non-real `4i`,
    ///   and `√2`, `π`, `e`, `i` for a symbol whose domain holds no rational.
    /// * `None` when the equation is neither proved nor refuted: it may be
    ///   an identity that the simplifier does not prove, or fail only at
    ///   points that were not tried.
    ///
    /// Up to 0.60.0 only [`Ex::equals`] was asked, and `x = 3` was `None`
    /// while `x = x + 1` was `Some(false)`.
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let ctx = Context::new();
    /// let x = ctx.symbol("x");
    /// let pyth = Equation::new(&x.sin().powi(2) + &x.cos().powi(2), ctx.one());
    /// assert_eq!(pyth.is_identity(), Some(true));
    /// assert_eq!(Equation::new(&x + 1, x.clone()).is_identity(), Some(false));
    /// assert_eq!(Equation::new(x.clone(), ctx.int(3)).is_identity(), Some(false));
    /// // sqrt(x^2) = x fails at x = -2/3 ...
    /// let root = Equation::new(x.powi(2).sqrt(), x.clone());
    /// assert_eq!(root.is_identity(), Some(false));
    /// // ... and holds on the declared domain of a positive symbol.
    /// let p = ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    /// assert_eq!(Equation::new(p.powi(2).sqrt(), p.clone()).is_identity(), Some(true));
    /// // Nothing is known about an undefined function.
    /// let f = ctx.apply("f", &[&x]).unwrap();
    /// assert_eq!(Equation::new(f, ctx.int(3)).is_identity(), None);
    /// ```
    #[must_use]
    pub fn is_identity(&self) -> Option<bool> {
        let known = self.lhs.equals(&self.rhs);
        if known.is_some() {
            return known;
        }
        self.fails_at_a_point().then_some(false)
    }

    /// Is there a point of the symbols' declared domain at which both
    /// sides have a finite value and the values are certified different?
    ///
    /// The points are fixed ([`Self::sample_values`]), so the answer does
    /// not depend on the history of the context; the symbols are taken in
    /// the order of their names.  A value is tried for a symbol only when
    /// it is proved to have every property the symbol is known to have and
    /// to lack every property the symbol is known to lack, so a symbol
    /// whose domain holds none of the values (a non-commutative one, an
    /// infinite one) refutes nothing.  A point at which a side has no
    /// finite value (a pole, `0/0`, an unevaluated form, an undefined
    /// function) is skipped; "different" is [`Ex::equals`] `== Some(false)`
    /// for the two constants, which is a proof.
    fn fails_at_a_point(&self) -> bool {
        /// The values tried for one symbol, and the points tried.
        const VALUES_PER_SYMBOL: usize = 8;
        const POINTS: usize = 6;

        let mut symbols = self.lhs.free_symbols();
        for s in self.rhs.free_symbols() {
            if !symbols.contains(&s) {
                symbols.push(s);
            }
        }
        if symbols.is_empty() {
            // Two constants: `equals` has said what is known.
            return false;
        }
        symbols.sort_by_cached_key(Ex::to_string);

        let values = Self::sample_values(&self.lhs);
        let mut admitted: Vec<Vec<&Ex>> = Vec::with_capacity(symbols.len());
        for s in &symbols {
            // What is known about the symbol, property by property.
            let known: Vec<(Props, Truth)> = Props::all()
                .iter()
                .map(|p| (p, Truth::from(s.query(p))))
                .filter(|(_, t)| t.is_known())
                .collect();
            let in_domain = |v: &Ex| {
                known.iter().all(|&(p, t)| {
                    let of_value = Truth::from(v.query(p));
                    of_value.is_known() && of_value.is_true() == t.is_true()
                })
            };
            let of_symbol: Vec<&Ex> = values
                .iter()
                .filter(|v| in_domain(v))
                .take(VALUES_PER_SYMBOL)
                .collect();
            if of_symbol.is_empty() {
                return false;
            }
            admitted.push(of_symbol);
        }

        let is_value = |e: &Ex| {
            e.free_symbols().is_empty()
                && !e.has_unevaluated()
                && Truth::from(e.is_finite()).is_true()
        };
        for t in 0..POINTS {
            // Symbol `j` takes its value number `t + j·(2t + 1)`: the first
            // point gives different symbols different values.
            let point: Vec<(&Ex, &Ex)> = symbols
                .iter()
                .zip(&admitted)
                .enumerate()
                .map(|(j, (s, of_symbol))| (s, of_symbol[(t + j * (2 * t + 1)) % of_symbol.len()]))
                .collect();
            let lhs = self.lhs.subs_map(&point).eval();
            let rhs = self.rhs.subs_map(&point).eval();
            if !is_value(&lhs) || !is_value(&rhs) {
                continue;
            }
            if Truth::from(lhs.equals(&rhs)).is_false() {
                return true;
            }
        }
        false
    }

    /// The values [`fails_at_a_point`](Self::fails_at_a_point) tries, in
    /// the context of `e`.  The first ones are points in general position
    /// (a fraction of either sign, a number off the real line and beyond
    /// the strip `|Im| ≤ π`), because the special points — `0`, `1`, the
    /// integers — are where two different functions most often agree
    /// (`floor(x) = x`) or have no value (`1/x`, `ln x`); integers,
    /// irrational and further non-real numbers follow for the symbols
    /// whose domain holds only those.
    fn sample_values(e: &Ex) -> Vec<Ex> {
        let ctx = e.context();
        let int = |n: i64| ctx.int(n);
        let frac = |p: i64, q: i64| ctx.rational(p, q);
        let i = ctx.i_unit();
        let root2 = ctx.int(2).sqrt();
        vec![
            frac(1, 2),
            frac(-2, 3),
            int(2),
            &i * 4,
            int(-1),
            frac(3, 2),
            int(0),
            int(1),
            root2.clone(),
            &i + 1,
            int(-2),
            int(3),
            ctx.pi(),
            frac(5, 7),
            frac(-1, 2),
            int(5),
            -&i,
            &i * 2,
            int(4),
            int(7),
            -&root2,
            ctx.e(),
            i.clone(),
            int(9),
            -ctx.pi(),
            int(-3),
            int(-5),
        ]
    }

    /// Convert to a single expression `lhs - rhs`.
    #[must_use]
    pub fn to_expr(&self) -> Ex {
        &self.lhs - &self.rhs
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display / Debug / PartialEq
// ═══════════════════════════════════════════════════════════════════════════

impl fmt::Display for Equation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.lhs, self.rhs)
    }
}

impl fmt::Debug for Equation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Equation({} = {})", self.lhs, self.rhs)
    }
}

/// Structural equality: both sides are the same arena nodes.
///
/// `x + 1 = 5` and `1 + x = 5` are equal (same canonical form);
/// `x + 1 = 5` and `x = 4` are not, even though they are equivalent.
impl PartialEq for Equation {
    fn eq(&self, other: &Self) -> bool {
        self.lhs == other.lhs && self.rhs == other.rhs
    }
}

impl Eq for Equation {}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic on both sides
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! impl_eq_scalar_op {
    ($trait:ident, $method:ident) => {
        impl<T: ToEx> ops::$trait<T> for &Equation {
            type Output = Equation;
            fn $method(self, rhs: T) -> Equation {
                let v = rhs.to_ex(&self.lhs.context());
                Equation {
                    lhs: ops::$trait::$method(&self.lhs, &v),
                    rhs: ops::$trait::$method(&self.rhs, &v),
                }
            }
        }
        impl<T: ToEx> ops::$trait<T> for Equation {
            type Output = Equation;
            fn $method(self, rhs: T) -> Equation {
                ops::$trait::$method(&self, rhs)
            }
        }
        impl ops::$trait<&Equation> for &Equation {
            type Output = Equation;
            fn $method(self, rhs: &Equation) -> Equation {
                Equation {
                    lhs: ops::$trait::$method(&self.lhs, &rhs.lhs),
                    rhs: ops::$trait::$method(&self.rhs, &rhs.rhs),
                }
            }
        }
        impl ops::$trait<Equation> for Equation {
            type Output = Equation;
            fn $method(self, rhs: Equation) -> Equation {
                ops::$trait::$method(&self, &rhs)
            }
        }
    };
}

impl_eq_scalar_op!(Add, add);
impl_eq_scalar_op!(Sub, sub);
impl_eq_scalar_op!(Mul, mul);
impl_eq_scalar_op!(Div, div);

impl ops::Neg for &Equation {
    type Output = Equation;
    fn neg(self) -> Equation {
        Equation {
            lhs: -&self.lhs,
            rhs: -&self.rhs,
        }
    }
}

impl ops::Neg for Equation {
    type Output = Equation;
    fn neg(self) -> Equation {
        -&self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;

    #[test]
    fn equation_display() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(&x + 1, ctx.int(5));
        let s = format!("{eq}");
        assert!(s.contains("=") && s.contains("5"), "got: {s}");
    }

    #[test]
    fn equation_solve_linear() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(&x + 1, ctx.int(5));
        let roots = eq.solve_or_empty(&x);
        assert_eq!(roots.len(), 1);
        assert_eq!(format!("{}", roots[0]), "4");
    }

    #[test]
    fn equation_solve_quadratic() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(x.powi(2), ctx.int(4));
        let roots = eq.solve_or_empty(&x);
        assert_eq!(roots.len(), 2, "x²=4 should have 2 roots");
    }

    #[test]
    fn equation_subs() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(&x + 1, ctx.int(5));
        let substituted = eq.subs_i64(&x, 4);
        assert!(
            substituted.is_satisfied() == Some(true),
            "x=4 should satisfy x+1=5"
        );
        assert_eq!(eq.subs_i64(&x, 3).is_satisfied(), Some(false));
    }

    #[test]
    fn equation_to_expr() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(x.clone(), ctx.int(3));
        let expr = eq.to_expr();
        // x - 3
        let s = format!("{expr}");
        assert!(s.contains("x") && s.contains("3"), "to_expr: {s}");
    }

    #[test]
    fn equation_simplify() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(&x.sin().powi(2) + &x.cos().powi(2), ctx.int(1));
        let simplified = eq.simplify();
        assert_eq!(format!("{}", simplified.lhs), "1");
    }

    #[test]
    fn equation_debug() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(x.clone(), ctx.int(0));
        let s = format!("{eq:?}");
        assert_eq!(s, "Equation(x = 0)");
    }

    #[test]
    fn accessors_and_swap() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(x.clone(), ctx.int(2));
        assert_eq!(eq.lhs(), &x);
        assert_eq!(eq.rhs(), &ctx.int(2));
        let sw = eq.swap();
        assert_eq!(sw.lhs(), &ctx.int(2));
        assert_eq!(sw.rhs(), &x);
        assert_eq!(sw.swap(), eq);
    }

    #[test]
    fn partial_eq_is_structural() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let a = Equation::new(&x + 1, ctx.int(5));
        let b = Equation::new(1 + &x, ctx.int(5));
        let c = Equation::new(x.clone(), ctx.int(4));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, a.swap());
    }

    #[test]
    fn scalar_arithmetic_both_sides() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(&x * 2 + 3, ctx.int(11));
        let step1 = &eq - 3;
        assert_eq!(step1, Equation::new(&x * 2, ctx.int(8)));
        let step2 = &step1 / 2;
        assert_eq!(step2, Equation::new(x.clone(), ctx.int(4)));
        let owned = eq.clone() * 0.5;
        assert_eq!(
            owned,
            Equation::new(&(&x * 2 + 3) * &ctx.rational(1, 2), ctx.rational(11, 2))
        );
        let by_ex = &eq + &x;
        assert_eq!(by_ex, Equation::new(&x * 3 + 3, &x + 11));
        let by_owned_ex = &eq - x.clone();
        assert_eq!(by_owned_ex, Equation::new(&x + 3, 11 - &x));
    }

    #[test]
    fn equation_plus_equation() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let e1 = Equation::new(&x + &y, ctx.int(3));
        let e2 = Equation::new(&x - &y, ctx.int(1));
        let sum = &e1 + &e2;
        assert_eq!(sum, Equation::new(&x * 2, ctx.int(4)));
        let diff = e1.clone() - e2.clone();
        assert_eq!(diff, Equation::new(&y * 2, ctx.int(2)));
        let prod = &e1 * &e2;
        assert_eq!(prod, Equation::new(&(&x + &y) * &(&x - &y), ctx.int(3)));
        let quot = &e1 / &e2;
        assert_eq!(quot, Equation::new(&(&x + &y) / &(&x - &y), ctx.int(3)));
    }

    #[test]
    fn negation() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(x.clone(), ctx.int(2));
        assert_eq!(-&eq, Equation::new(-&x, ctx.int(-2)));
        assert_eq!(-eq.clone(), Equation::new(-&x, ctx.int(-2)));
    }

    #[test]
    fn apply_and_zero_form() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(x.clone(), ctx.int(3));
        let sq = eq.apply(|s| s.powi(2));
        assert_eq!(sq, Equation::new(x.powi(2), ctx.int(9)));
        let z = sq.to_zero_equation();
        assert_eq!(z.rhs(), &ctx.zero());
        assert_eq!(z.lhs(), &(&x.powi(2) - 9));
    }

    #[test]
    fn is_identity_three_valued() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        assert_eq!(
            Equation::new((&x + 1).powi(2), &x.powi(2) + &x * 2 + 1).is_identity(),
            Some(true)
        );
        assert_eq!(Equation::new(x.clone(), &x + 1).is_identity(), Some(false));
        // Fails at x = 1/2 (up to 0.60.0: `None`).
        assert_eq!(
            Equation::new(x.clone(), ctx.int(1)).is_identity(),
            Some(false)
        );
        assert_eq!(Equation::new(x.clone(), ctx.int(1)).is_satisfied(), None);
        assert_eq!(
            Equation::new(ctx.int(2), ctx.int(2)).is_identity(),
            Some(true)
        );
    }

    #[test]
    fn solve_for_returns_equations() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let sols = Equation::new(x.powi(2) - 1, ctx.zero())
            .solve_for(&x)
            .unwrap();
        assert_eq!(sols.len(), 2);
        for s in &sols {
            assert_eq!(s.lhs(), &x);
            assert!(s.rhs().is_constant());
        }
        let vals: Vec<Ex> = sols.into_iter().map(|e| e.rhs).collect();
        assert!(vals.contains(&ctx.int(1)) && vals.contains(&ctx.int(-1)));
    }

    #[test]
    fn f64_operands_are_exact_on_equations() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let eq = Equation::new(x.clone(), ctx.int(1)) * 0.1;
        assert_eq!(eq.rhs(), &ctx.from_f64(0.1).unwrap());
        assert_ne!(eq.rhs(), &ctx.rational(1, 10));
    }

    #[test]
    #[should_panic(expected = "different contexts")]
    fn new_rejects_mixed_contexts() {
        let a = Context::new();
        let b = Context::new();
        let _ = Equation::new(a.symbol("x"), b.int(1));
    }
}
