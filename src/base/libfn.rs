//! The registry of library special functions, the nodes
//! [`ExprNode::Lib`](crate::base::node::ExprNode::Lib).
//!
//! Elementary functions (`sin`, `exp`, `gamma`, `erf`, …) are dedicated
//! `ExprNode` variants.  The larger family of special functions — Bessel,
//! orthogonal polynomials, incomplete gamma, elliptic integrals, the integer
//! sequences — is stored as `Lib(LibFn, args)`.  [`LibFn`] is the closed
//! list of those functions, so a consumer that matches on it (an
//! evaluator, a differentiator, a printer) is checked for completeness by
//! the compiler instead of failing at run time on a name it forgot.
//!
//! Nodes are built with `Arena::lib_apply` (or the arity-checked
//! `Arena::func`).  Until 0.49 they were `Apply(SymbolId, args)` nodes
//! resolved by the interned text of the head, so a user function named
//! like a library function took on its semantics.  The name
//! ([`LibFn::name`]) is still the head of the JSON `Apply` form and of the
//! display.
//!
//! The list of functions itself (since 0.49) is the one list of *every*
//! function, node-backed, library or internal, in `base/func.rs`: it
//! generates [`LibFn`] — its variants, names and arities — together with
//! the crate's internal function identity, and this module re-exports it.

use std::fmt;

pub use crate::base::func::LibFn;

/// How many arguments a library function takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arity {
    /// Exactly `n` arguments.
    Fixed(u8),
    /// Between `min` and `max` arguments, inclusive: Lambert W, `lambertw(x)`
    /// or `lambertw(x, k)` (the branch `k`, SymPy's `LambertW(x, k)`).
    Range {
        /// Fewest arguments accepted.
        min: u8,
        /// Most arguments accepted.
        max: u8,
    },
}

impl Arity {
    /// Does a call with `n` arguments have an acceptable shape?
    ///
    /// ```
    /// use symplex::base::libfn::{Arity, LibFn};
    ///
    /// assert!(LibFn::BesselJ.arity().accepts(2));
    /// assert!(!LibFn::BesselJ.arity().accepts(1));
    /// assert_eq!(LibFn::Jacobi.arity(), Arity::Fixed(4));
    /// assert!(LibFn::LambertW.arity().accepts(1) && LibFn::LambertW.arity().accepts(2));
    /// assert!(!LibFn::LambertW.arity().accepts(3));
    /// ```
    pub const fn accepts(self, n: usize) -> bool {
        match self {
            Arity::Fixed(k) => n == k as usize,
            Arity::Range { min, max } => min as usize <= n && n <= max as usize,
        }
    }
}

impl LibFn {
    /// The function whose name equals `s` up to ASCII case (`shi` → [`LibFn::Shi`]).
    ///
    /// The text parser lower-cases function names before matching them.
    pub fn from_name_ignore_ascii_case(s: &str) -> Option<LibFn> {
        LibFn::ALL
            .iter()
            .copied()
            .find(|f| f.name().eq_ignore_ascii_case(s))
    }
}

impl fmt::Display for LibFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
