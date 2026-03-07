//! Symbolic equation type.
//!
//! [`Equation`] represents `lhs = rhs` and provides convenience methods
//! for solving, substitution, and display. Internally, solving converts
//! to `lhs - rhs = 0` and delegates to the existing solver.
//!
//! The type is named `Equation` (not `Eq`) to avoid conflict with
//! `std::cmp::Eq`.

use std::fmt;

/// A symbolic equation `lhs = rhs`.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::eq::Equation;
///
/// let x = symplex::var("x");
/// let eq = Equation::new(&x + 1, symplex::int(5));
/// assert_eq!(format!("{eq}"), "x + 1 = 5");
/// ```
#[derive(Clone)]
pub struct Equation {
    /// Left-hand side of the equation.
    pub lhs: crate::api::expr::Ex,
    /// Right-hand side of the equation.
    pub rhs: crate::api::expr::Ex,
}

impl Equation {
    /// Create a new equation `lhs = rhs`.
    pub fn new(lhs: crate::api::expr::Ex, rhs: crate::api::expr::Ex) -> Self {
        Self { lhs, rhs }
    }

    /// Solve this equation for `var`.
    ///
    /// Internally computes `lhs - rhs` and solves for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::eq::Equation;
    ///
    /// let x = symplex::var("x");
    /// let eq = Equation::new(&x + 1, symplex::int(5));
    /// let roots = eq.solve(&x).unwrap();
    /// assert_eq!(format!("{}", roots[0]), "4");
    /// ```
    pub fn solve(
        &self,
        var: &crate::api::expr::Ex,
    ) -> Result<Vec<crate::api::expr::Ex>, crate::base::errors::SymplexError> {
        let diff = &self.lhs - &self.rhs;
        diff.solve(var)
    }

    /// Solve this equation, returning empty vec on failure.
    pub fn solve_or_empty(&self, var: &crate::api::expr::Ex) -> Vec<crate::api::expr::Ex> {
        let diff = &self.lhs - &self.rhs;
        diff.solve_or_empty(var)
    }

    /// Substitute a variable in both sides.
    pub fn subs(&self, old: &crate::api::expr::Ex, new: &crate::api::expr::Ex) -> Equation {
        Equation {
            lhs: self.lhs.subs(old, new),
            rhs: self.rhs.subs(old, new),
        }
    }

    /// Substitute an integer value in both sides.
    pub fn subs_i64(&self, old: &crate::api::expr::Ex, val: i64) -> Equation {
        Equation {
            lhs: self.lhs.subs_i64(old, val),
            rhs: self.rhs.subs_i64(old, val),
        }
    }

    /// Simplify both sides.
    pub fn simplify(&self) -> Equation {
        Equation {
            lhs: self.lhs.simplify(),
            rhs: self.rhs.simplify(),
        }
    }

    /// Expand both sides.
    pub fn expand(&self) -> Equation {
        Equation {
            lhs: self.lhs.expand(),
            rhs: self.rhs.expand(),
        }
    }

    /// Evaluate both sides.
    pub fn eval(&self) -> Equation {
        Equation {
            lhs: self.lhs.eval(),
            rhs: self.rhs.eval(),
        }
    }

    /// Check if the equation is satisfied (both sides equal).
    pub fn is_satisfied(&self) -> Option<bool> {
        self.lhs.equals(&self.rhs)
    }

    /// Convert to a single expression `lhs - rhs`.
    pub fn to_expr(&self) -> crate::api::expr::Ex {
        &self.lhs - &self.rhs
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equation_display() {
        let x = crate::var("x");
        let eq = Equation::new(&x + 1, crate::int(5));
        let s = format!("{eq}");
        assert!(s.contains("=") && s.contains("5"), "got: {s}");
    }

    #[test]
    fn equation_solve_linear() {
        let x = crate::var("x");
        let eq = Equation::new(&x + 1, crate::int(5));
        let roots = eq.solve_or_empty(&x);
        assert_eq!(roots.len(), 1);
        assert_eq!(format!("{}", roots[0]), "4");
    }

    #[test]
    fn equation_solve_quadratic() {
        let x = crate::var("x");
        let eq = Equation::new(x.powi(2), crate::int(4));
        let roots = eq.solve_or_empty(&x);
        assert_eq!(roots.len(), 2, "x²=4 should have 2 roots");
    }

    #[test]
    fn equation_subs() {
        let x = crate::var("x");
        let eq = Equation::new(&x + 1, crate::int(5));
        let substituted = eq.subs_i64(&x, 4);
        assert!(
            substituted.is_satisfied() == Some(true),
            "x=4 should satisfy x+1=5"
        );
    }

    #[test]
    fn equation_to_expr() {
        let x = crate::var("x");
        let eq = Equation::new(x.clone(), crate::int(3));
        let expr = eq.to_expr();
        // x - 3
        let s = format!("{expr}");
        assert!(s.contains("x") && s.contains("3"), "to_expr: {s}");
    }

    #[test]
    fn equation_simplify() {
        let x = crate::var("x");
        let eq = Equation::new(&x.sin().powi(2) + &x.cos().powi(2), crate::int(1));
        let simplified = eq.simplify();
        assert_eq!(format!("{}", simplified.lhs), "1");
    }

    #[test]
    fn equation_debug() {
        let x = crate::var("x");
        let eq = Equation::new(x.clone(), crate::int(0));
        let s = format!("{eq:?}");
        assert!(s.contains("Equation"), "debug: {s}");
    }
}
