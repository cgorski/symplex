//! Mathematical property inference engine.
//!
//! This module implements a three-valued logic system for tracking
//! mathematical properties of expressions. Each property can be
//! `Some(true)` (known true), `Some(false)` (known false), or `None`
//! (unknown).
//!
//! The core data structures are:
//!
//! - [`Props`] — a bitflag set naming mathematical properties.
//! - [`Assumptions`] — a pair of `Props` sets (`known_true`, `known_false`)
//!   representing three-valued knowledge about an expression.
//! - [`AssumptionCache`] — a cache mapping `ExprId` → [`Assumptions`],
//!   with methods to query and compute properties.
//!
//! # Inference rules
//!
//! When a property is asserted, [`Assumptions::forward_chain`] applies
//! implication rules until a fixpoint is reached.  For example, asserting
//! `integer = true` immediately deduces `rational = true`, `real = true`,
//! `complex = true`, `finite = true`, `commutative = true`, and
//! `algebraic = true`.
//!
//! The rules are derived from the same mathematical ontology as SymPy's
//! assumption system, but compiled into bitmask operations for speed.
//!
//! # Sign properties and infinities
//!
//! `positive`, `negative`, `nonnegative` and `nonpositive` are *extended
//! real* notions: `oo` is positive and `-oo` is negative (SymPy's
//! `extended_positive` / `extended_negative`).  Consequently a sign
//! property alone implies `extended_real`, `nonzero` (for the strict ones)
//! and the negations of the opposite signs — but **not** `real`, `finite`
//! or `complex`.  Those follow from `extended_real ∧ finite → real`:
//!
//! | value  | positive | extended_real | finite | real  | complex | infinite |
//! |--------|----------|---------------|--------|-------|---------|----------|
//! | `1`    | true     | true          | true   | true  | true    | false    |
//! | `oo`   | true     | true          | false  | false | false   | true     |
//! | `-oo`  | false    | true          | false  | false | false   | true     |
//! | `zoo`  | false    | false         | false  | false | false   | true     |
//! | `I`    | false    | false         | true   | false | true    | false    |
//! | `nan`  | unknown  | unknown       | unknown| unknown | unknown | unknown |
//!
//! A symbol *declared* `Positive` (via `Context::symbol_with`, `sym!` or
//! `Ex::assume`) is nevertheless a finite positive number, as in SymPy:
//! [`Assumptions::normalize_declared`] adds `finite` to declared sets that
//! carry a sign unless finiteness was declared explicitly.  Declaring a
//! contradictory set panics.

use bitflags::bitflags;
use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, SymbolId};

// ═══════════════════════════════════════════════════════════════════════════
// Props — property flags
// ═══════════════════════════════════════════════════════════════════════════

bitflags! {
    /// A set of mathematical properties.
    ///
    /// Used as both a "known true" mask and a "known false" mask inside
    /// [`Assumptions`].
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Props: u32 {
        /// Commutes with all other objects under multiplication.
        const COMMUTATIVE     = 1 << 0;
        /// Element of ℂ.
        const COMPLEX         = 1 << 1;
        /// Element of ℝ.
        const REAL            = 1 << 2;
        /// Element of ℚ.
        const RATIONAL        = 1 << 3;
        /// Element of ℤ.
        const INTEGER         = 1 << 4;
        /// Root of a polynomial with rational coefficients.
        const ALGEBRAIC       = 1 << 5;
        /// Not algebraic (π, e, …).
        const TRANSCENDENTAL  = 1 << 6;
        /// Real but not rational.
        const IRRATIONAL      = 1 << 7;
        /// Pure imaginary (nonzero, real part is zero).
        const IMAGINARY       = 1 << 8;
        /// Strictly greater than zero.
        const POSITIVE        = 1 << 9;
        /// Strictly less than zero.
        const NEGATIVE        = 1 << 10;
        /// Greater than or equal to zero.
        const NONNEGATIVE     = 1 << 11;
        /// Less than or equal to zero.
        const NONPOSITIVE     = 1 << 12;
        /// Equal to zero.
        const ZERO            = 1 << 13;
        /// Not equal to zero.
        const NONZERO         = 1 << 14;
        /// Divisible by 2 (integer).
        const EVEN            = 1 << 15;
        /// Not divisible by 2 (integer).
        const ODD             = 1 << 16;
        /// A prime number.
        const PRIME           = 1 << 17;
        /// A composite number.
        const COMPOSITE       = 1 << 18;
        /// Bounded in absolute value.
        const FINITE          = 1 << 19;
        /// Unbounded (±∞).
        const INFINITE        = 1 << 20;
        /// Equal to its own conjugate transpose.
        const HERMITIAN       = 1 << 21;
        /// Equal to the negation of its conjugate transpose.
        const ANTIHERMITIAN   = 1 << 22;
        /// Element of ℝ ∪ {−∞, +∞} (every real is an extended real; so are `±oo`).
        const EXTENDED_REAL   = 1 << 23;
    }
}

/// `(flag, lowercase name)` for every property, in bit order.
const PROP_NAMES: [(Props, &str); 24] = [
    (Props::COMMUTATIVE, "commutative"),
    (Props::COMPLEX, "complex"),
    (Props::REAL, "real"),
    (Props::RATIONAL, "rational"),
    (Props::INTEGER, "integer"),
    (Props::ALGEBRAIC, "algebraic"),
    (Props::TRANSCENDENTAL, "transcendental"),
    (Props::IRRATIONAL, "irrational"),
    (Props::IMAGINARY, "imaginary"),
    (Props::POSITIVE, "positive"),
    (Props::NEGATIVE, "negative"),
    (Props::NONNEGATIVE, "nonnegative"),
    (Props::NONPOSITIVE, "nonpositive"),
    (Props::ZERO, "zero"),
    (Props::NONZERO, "nonzero"),
    (Props::EVEN, "even"),
    (Props::ODD, "odd"),
    (Props::PRIME, "prime"),
    (Props::COMPOSITE, "composite"),
    (Props::FINITE, "finite"),
    (Props::INFINITE, "infinite"),
    (Props::HERMITIAN, "hermitian"),
    (Props::ANTIHERMITIAN, "antihermitian"),
    (Props::EXTENDED_REAL, "extended_real"),
];

impl std::fmt::Display for Props {
    /// Comma-separated lowercase property names in bit order
    /// (e.g. `positive, real, nonzero`); the empty set prints as `none`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (flag, name) in PROP_NAMES {
            if self.contains(flag) {
                if !first {
                    f.write_str(", ")?;
                }
                f.write_str(name)?;
                first = false;
            }
        }
        if first {
            f.write_str("none")?;
        }
        Ok(())
    }
}

/// A user-facing assumption specifier for the [`sym!`](crate::sym) macro
/// and [`Context::symbol_with`](crate::api::context::Context::symbol_with).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Assumption {
    /// Commutes under multiplication.
    Commutative,
    /// Element of ℂ.
    Complex,
    /// Element of ℝ.
    Real,
    /// Element of ℚ.
    Rational,
    /// Element of ℤ.
    Integer,
    /// Root of a polynomial with rational coefficients.
    Algebraic,
    /// Not algebraic (π, e, …).
    Transcendental,
    /// Real but not rational.
    Irrational,
    /// Pure imaginary (nonzero, real part is zero).
    Imaginary,
    /// Strictly greater than zero.
    Positive,
    /// Strictly less than zero.
    Negative,
    /// Greater than or equal to zero.
    NonNegative,
    /// Less than or equal to zero.
    NonPositive,
    /// Equal to zero.
    Zero,
    /// Not equal to zero.
    NonZero,
    /// Divisible by 2 (integer).
    Even,
    /// Not divisible by 2 (integer).
    Odd,
    /// A prime number.
    Prime,
    /// A composite number.
    Composite,
    /// Bounded in absolute value.
    Finite,
    /// Unbounded (±∞).
    Infinite,
    /// Equal to its own conjugate transpose.
    Hermitian,
    /// Equal to the negation of its conjugate transpose.
    AntiHermitian,
    /// Element of ℝ ∪ {−∞, +∞}.
    ExtendedReal,
    // Negated forms
    /// Assert not real.
    NotReal,
    /// Assert not complex.
    NotComplex,
    /// Assert not integer.
    NotInteger,
    /// Assert not rational.
    NotRational,
    /// Assert not positive.
    NotPositive,
    /// Assert not negative.
    NotNegative,
    /// Assert not zero (alias for `NonZero`-as-negation).
    NotZero,
    /// Assert not finite.
    NotFinite,
    /// Assert not commutative.
    NotCommutative,
    /// Assert not algebraic.
    NotAlgebraic,
    /// Assert not transcendental.
    NotTranscendental,
    /// Assert not irrational.
    NotIrrational,
    /// Assert not (purely) imaginary.
    NotImaginary,
    /// Assert not nonnegative (i.e. negative or not real).
    NotNonNegative,
    /// Assert not nonpositive (i.e. positive or not real).
    NotNonPositive,
    /// Assert not nonzero.
    NotNonZero,
    /// Assert not even.
    NotEven,
    /// Assert not odd.
    NotOdd,
    /// Assert not prime.
    NotPrime,
    /// Assert not composite.
    NotComposite,
    /// Assert not infinite.
    NotInfinite,
    /// Assert not Hermitian.
    NotHermitian,
    /// Assert not anti-Hermitian.
    NotAntiHermitian,
    /// Assert not an extended real.
    NotExtendedReal,
}

impl Assumption {
    /// Convert to a `(Props, bool)` pair — which property flag, and
    /// whether it's being asserted true or false.
    pub fn to_prop_value(self) -> (Props, bool) {
        match self {
            Assumption::Commutative => (Props::COMMUTATIVE, true),
            Assumption::Complex => (Props::COMPLEX, true),
            Assumption::Real => (Props::REAL, true),
            Assumption::Rational => (Props::RATIONAL, true),
            Assumption::Integer => (Props::INTEGER, true),
            Assumption::Algebraic => (Props::ALGEBRAIC, true),
            Assumption::Transcendental => (Props::TRANSCENDENTAL, true),
            Assumption::Irrational => (Props::IRRATIONAL, true),
            Assumption::Imaginary => (Props::IMAGINARY, true),
            Assumption::Positive => (Props::POSITIVE, true),
            Assumption::Negative => (Props::NEGATIVE, true),
            Assumption::NonNegative => (Props::NONNEGATIVE, true),
            Assumption::NonPositive => (Props::NONPOSITIVE, true),
            Assumption::Zero => (Props::ZERO, true),
            Assumption::NonZero => (Props::NONZERO, true),
            Assumption::Even => (Props::EVEN, true),
            Assumption::Odd => (Props::ODD, true),
            Assumption::Prime => (Props::PRIME, true),
            Assumption::Composite => (Props::COMPOSITE, true),
            Assumption::Finite => (Props::FINITE, true),
            Assumption::Infinite => (Props::INFINITE, true),
            Assumption::Hermitian => (Props::HERMITIAN, true),
            Assumption::AntiHermitian => (Props::ANTIHERMITIAN, true),
            Assumption::ExtendedReal => (Props::EXTENDED_REAL, true),
            Assumption::NotReal => (Props::REAL, false),
            Assumption::NotComplex => (Props::COMPLEX, false),
            Assumption::NotInteger => (Props::INTEGER, false),
            Assumption::NotRational => (Props::RATIONAL, false),
            Assumption::NotPositive => (Props::POSITIVE, false),
            Assumption::NotNegative => (Props::NEGATIVE, false),
            Assumption::NotZero => (Props::ZERO, false),
            Assumption::NotFinite => (Props::FINITE, false),
            Assumption::NotCommutative => (Props::COMMUTATIVE, false),
            Assumption::NotAlgebraic => (Props::ALGEBRAIC, false),
            Assumption::NotTranscendental => (Props::TRANSCENDENTAL, false),
            Assumption::NotIrrational => (Props::IRRATIONAL, false),
            Assumption::NotImaginary => (Props::IMAGINARY, false),
            Assumption::NotNonNegative => (Props::NONNEGATIVE, false),
            Assumption::NotNonPositive => (Props::NONPOSITIVE, false),
            Assumption::NotNonZero => (Props::NONZERO, false),
            Assumption::NotEven => (Props::EVEN, false),
            Assumption::NotOdd => (Props::ODD, false),
            Assumption::NotPrime => (Props::PRIME, false),
            Assumption::NotComposite => (Props::COMPOSITE, false),
            Assumption::NotInfinite => (Props::INFINITE, false),
            Assumption::NotHermitian => (Props::HERMITIAN, false),
            Assumption::NotAntiHermitian => (Props::ANTIHERMITIAN, false),
            Assumption::NotExtendedReal => (Props::EXTENDED_REAL, false),
        }
    }

    /// The assumption asserting the opposite truth value of the same
    /// property (`Positive` ↔ `NotPositive`, `Zero` ↔ `NotZero`, …).
    ///
    /// `negate` is an involution: `a.negate().negate() == a`.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// assert_eq!(Assumption::Positive.negate(), Assumption::NotPositive);
    /// assert_eq!(Assumption::NotReal.negate(), Assumption::Real);
    /// assert_eq!(Assumption::Even.negate().negate(), Assumption::Even);
    /// ```
    pub fn negate(self) -> Assumption {
        match self {
            Assumption::Commutative => Assumption::NotCommutative,
            Assumption::Complex => Assumption::NotComplex,
            Assumption::Real => Assumption::NotReal,
            Assumption::Rational => Assumption::NotRational,
            Assumption::Integer => Assumption::NotInteger,
            Assumption::Algebraic => Assumption::NotAlgebraic,
            Assumption::Transcendental => Assumption::NotTranscendental,
            Assumption::Irrational => Assumption::NotIrrational,
            Assumption::Imaginary => Assumption::NotImaginary,
            Assumption::Positive => Assumption::NotPositive,
            Assumption::Negative => Assumption::NotNegative,
            Assumption::NonNegative => Assumption::NotNonNegative,
            Assumption::NonPositive => Assumption::NotNonPositive,
            Assumption::Zero => Assumption::NotZero,
            Assumption::NonZero => Assumption::NotNonZero,
            Assumption::Even => Assumption::NotEven,
            Assumption::Odd => Assumption::NotOdd,
            Assumption::Prime => Assumption::NotPrime,
            Assumption::Composite => Assumption::NotComposite,
            Assumption::Finite => Assumption::NotFinite,
            Assumption::Infinite => Assumption::NotInfinite,
            Assumption::Hermitian => Assumption::NotHermitian,
            Assumption::AntiHermitian => Assumption::NotAntiHermitian,
            Assumption::ExtendedReal => Assumption::NotExtendedReal,
            Assumption::NotReal => Assumption::Real,
            Assumption::NotComplex => Assumption::Complex,
            Assumption::NotInteger => Assumption::Integer,
            Assumption::NotRational => Assumption::Rational,
            Assumption::NotPositive => Assumption::Positive,
            Assumption::NotNegative => Assumption::Negative,
            Assumption::NotZero => Assumption::Zero,
            Assumption::NotFinite => Assumption::Finite,
            Assumption::NotCommutative => Assumption::Commutative,
            Assumption::NotAlgebraic => Assumption::Algebraic,
            Assumption::NotTranscendental => Assumption::Transcendental,
            Assumption::NotIrrational => Assumption::Irrational,
            Assumption::NotImaginary => Assumption::Imaginary,
            Assumption::NotNonNegative => Assumption::NonNegative,
            Assumption::NotNonPositive => Assumption::NonPositive,
            Assumption::NotNonZero => Assumption::NonZero,
            Assumption::NotEven => Assumption::Even,
            Assumption::NotOdd => Assumption::Odd,
            Assumption::NotPrime => Assumption::Prime,
            Assumption::NotComposite => Assumption::Composite,
            Assumption::NotInfinite => Assumption::Infinite,
            Assumption::NotHermitian => Assumption::Hermitian,
            Assumption::NotAntiHermitian => Assumption::AntiHermitian,
            Assumption::NotExtendedReal => Assumption::ExtendedReal,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumptions — three-valued property storage
// ═══════════════════════════════════════════════════════════════════════════

/// Three-valued assumption state for a single expression.
///
/// Each mathematical property is either known true, known false, or
/// unknown.  The `forward_chain` method applies implication rules to
/// derive consequences whenever a new fact is asserted.
///
/// # Representation
///
/// Two [`Props`] bitflags — one for "known true", one for "known false".
/// A property present in neither is unknown.  A property present in both
/// indicates a contradiction (which should not happen if the inference
/// rules are correct).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Assumptions {
    /// Properties known to be **true**.
    pub known_true: Props,
    /// Properties known to be **false**.
    pub known_false: Props,
}

impl Assumptions {
    /// Query a single property.
    ///
    /// Returns `Some(true)` if known true, `Some(false)` if known false,
    /// or `None` if unknown.
    #[inline]
    pub fn query(&self, prop: Props) -> Option<bool> {
        if self.known_true.contains(prop) {
            Some(true)
        } else if self.known_false.contains(prop) {
            Some(false)
        } else {
            None
        }
    }

    /// Assert that a property is true and run forward-chaining inference.
    pub fn assert_true(&mut self, prop: Props) {
        self.known_true.insert(prop);
        self.forward_chain();
    }

    /// Assert that a property is false and run forward-chaining inference.
    pub fn assert_false(&mut self, prop: Props) {
        self.known_false.insert(prop);
        self.forward_chain();
    }

    /// Check whether the assumption set is self-contradictory.
    pub fn is_contradictory(&self) -> bool {
        self.known_true.intersects(self.known_false)
    }

    /// Normalise a set of assumptions *declared on a symbol*.
    ///
    /// The sign properties (`positive`, `negative`, `nonnegative`,
    /// `nonpositive`) are extended-real notions in the lattice — `oo` is
    /// positive — so on their own they do not imply `real`.  A symbol that
    /// a user declares `Positive`, however, is meant to be an ordinary
    /// (finite) positive number, exactly as in SymPy.  This method encodes
    /// that convention: when finiteness was not declared either way and the
    /// set carries a sign property or is known `!real`, `finite` is added,
    /// from which `real` (for signed symbols) follows.
    ///
    /// Symbols declared only `ExtendedReal` stay agnostic about finiteness,
    /// and an explicit `Infinite` / `NotFinite` is always respected
    /// (`[Positive, Infinite]` describes `+oo`).
    ///
    /// The result is forward-chained.  Idempotent.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let mut pos = Assumptions::default();
    /// pos.assert_true(Props::POSITIVE);
    /// assert_eq!(pos.query(Props::REAL), None, "could be +oo");
    /// pos.normalize_declared();
    /// assert_eq!(pos.query(Props::FINITE), Some(true));
    /// assert_eq!(pos.query(Props::REAL), Some(true));
    ///
    /// let mut ext = Assumptions::default();
    /// ext.assert_true(Props::EXTENDED_REAL);
    /// ext.normalize_declared();
    /// assert_eq!(ext.query(Props::FINITE), None);
    /// ```
    pub fn normalize_declared(&mut self) {
        self.forward_chain();
        if self.query(Props::FINITE).is_some() {
            return;
        }
        let signed = self.known_true.intersects(
            Props::POSITIVE | Props::NEGATIVE | Props::NONNEGATIVE | Props::NONPOSITIVE,
        );
        let not_real = self.known_false.contains(Props::REAL);
        if signed || not_real {
            self.assert_true(Props::FINITE);
        }
    }

    /// Does everything known in `other` follow from `self`?
    ///
    /// `self` is forward-chained first, so `implies` sees derived facts:
    /// asserting `positive` implies `{extended_real, nonzero, !negative}`.
    /// A contradictory `self` implies everything.
    ///
    /// # Examples
    ///
    /// ```
    /// use symplex::prelude::*;
    ///
    /// let mut pos = Assumptions::default();
    /// pos.assert_true(Props::POSITIVE);
    /// let mut ext_nonzero = Assumptions::default();
    /// ext_nonzero.known_true = Props::EXTENDED_REAL | Props::NONZERO;
    /// assert!(pos.implies(&ext_nonzero));
    /// assert!(!ext_nonzero.implies(&pos));
    /// ```
    pub fn implies(&self, other: &Assumptions) -> bool {
        let mut me = *self;
        me.forward_chain();
        if me.is_contradictory() {
            return true;
        }
        me.known_true.contains(other.known_true) && me.known_false.contains(other.known_false)
    }
}

impl std::fmt::Display for Assumptions {
    /// Lists the active properties: known-true names, then known-false
    /// names prefixed with `!`, comma-separated (e.g.
    /// `positive, real, nonzero, !negative, !zero`).  Prints `unknown`
    /// when nothing is known.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (flag, name) in PROP_NAMES {
            if self.known_true.contains(flag) {
                if !first {
                    f.write_str(", ")?;
                }
                f.write_str(name)?;
                first = false;
            }
        }
        for (flag, name) in PROP_NAMES {
            if self.known_false.contains(flag) {
                if !first {
                    f.write_str(", ")?;
                }
                f.write_str("!")?;
                f.write_str(name)?;
                first = false;
            }
        }
        if first {
            f.write_str("unknown")?;
        }
        Ok(())
    }
}

impl Assumptions {
    /// Merge knowledge from another assumption set.
    ///
    /// Any property known in `other` but unknown in `self` is copied.
    /// Returns true if anything changed.
    pub fn merge(&mut self, other: &Assumptions) -> bool {
        let old_true = self.known_true;
        let old_false = self.known_false;
        self.known_true |= other.known_true;
        self.known_false |= other.known_false;
        if self.known_true != old_true || self.known_false != old_false {
            self.forward_chain();
            true
        } else {
            false
        }
    }

    /// Apply all implication rules until no new facts are deduced.
    ///
    /// This is the compiled inference engine.  Each iteration applies
    /// alpha rules (single-trigger: `A → B`) and beta rules
    /// (join-trigger: `A ∧ B → C`) as bitmask operations.
    ///
    /// Converges in 2–4 iterations for typical inputs.
    pub fn forward_chain(&mut self) {
        loop {
            let old_true = self.known_true;
            let old_false = self.known_false;

            // ── Alpha rules: single-trigger implications ──────────

            // integer → rational, algebraic, real, complex, finite, commutative, hermitian
            if self.known_true.contains(Props::INTEGER) {
                self.known_true |= Props::RATIONAL
                    | Props::ALGEBRAIC
                    | Props::REAL
                    | Props::COMPLEX
                    | Props::FINITE
                    | Props::COMMUTATIVE
                    | Props::HERMITIAN;
                self.known_false |= Props::INFINITE | Props::IMAGINARY | Props::TRANSCENDENTAL;
            }

            // rational → algebraic, real, complex, finite, commutative, hermitian
            if self.known_true.contains(Props::RATIONAL) {
                self.known_true |= Props::ALGEBRAIC
                    | Props::REAL
                    | Props::COMPLEX
                    | Props::FINITE
                    | Props::COMMUTATIVE
                    | Props::HERMITIAN;
                self.known_false |=
                    Props::INFINITE | Props::IMAGINARY | Props::TRANSCENDENTAL | Props::IRRATIONAL;
            }

            // irrational → real, nonzero, !rational, !integer
            if self.known_true.contains(Props::IRRATIONAL) {
                self.known_true |= Props::REAL
                    | Props::COMPLEX
                    | Props::FINITE
                    | Props::COMMUTATIVE
                    | Props::HERMITIAN
                    | Props::NONZERO;
                self.known_false |= Props::RATIONAL
                    | Props::INTEGER
                    | Props::INFINITE
                    | Props::IMAGINARY
                    | Props::ZERO
                    | Props::EVEN
                    | Props::ODD
                    | Props::PRIME
                    | Props::COMPOSITE;
            }

            // algebraic → complex, finite, commutative
            if self.known_true.contains(Props::ALGEBRAIC) {
                self.known_true |= Props::COMPLEX | Props::FINITE | Props::COMMUTATIVE;
                self.known_false |= Props::INFINITE | Props::TRANSCENDENTAL;
            }

            // transcendental → complex, !algebraic
            if self.known_true.contains(Props::TRANSCENDENTAL) {
                self.known_true |= Props::COMPLEX | Props::FINITE | Props::COMMUTATIVE;
                self.known_false |=
                    Props::ALGEBRAIC | Props::RATIONAL | Props::INTEGER | Props::INFINITE;
            }

            // real → extended_real, complex, finite, commutative, hermitian, !imaginary
            if self.known_true.contains(Props::REAL) {
                self.known_true |= Props::EXTENDED_REAL
                    | Props::COMPLEX
                    | Props::FINITE
                    | Props::COMMUTATIVE
                    | Props::HERMITIAN;
                self.known_false |= Props::IMAGINARY | Props::INFINITE;
            }

            // imaginary → complex, finite, nonzero, commutative, antihermitian, !real
            if self.known_true.contains(Props::IMAGINARY) {
                self.known_true |= Props::COMPLEX
                    | Props::FINITE
                    | Props::NONZERO
                    | Props::COMMUTATIVE
                    | Props::ANTIHERMITIAN;
                self.known_false |= Props::REAL
                    | Props::EXTENDED_REAL
                    | Props::INFINITE
                    | Props::ZERO
                    | Props::POSITIVE
                    | Props::NEGATIVE
                    | Props::NONNEGATIVE
                    | Props::NONPOSITIVE
                    | Props::INTEGER
                    | Props::RATIONAL
                    | Props::IRRATIONAL
                    | Props::EVEN
                    | Props::ODD
                    | Props::PRIME
                    | Props::COMPOSITE;
            }

            // complex → commutative, finite
            if self.known_true.contains(Props::COMPLEX) {
                self.known_true |= Props::COMMUTATIVE | Props::FINITE;
                self.known_false |= Props::INFINITE;
            }

            // extended_real → commutative, !imaginary
            if self.known_true.contains(Props::EXTENDED_REAL) {
                self.known_true |= Props::COMMUTATIVE;
                self.known_false |= Props::IMAGINARY;
            }

            // The four sign properties live in the *extended* reals: `oo` is
            // positive.  They therefore imply `extended_real` but not `real`
            // or `finite`; `real` follows from `extended_real ∧ finite` (beta
            // rule below).

            // positive → nonnegative, nonzero, extended_real, !negative, !zero, !nonpositive
            if self.known_true.contains(Props::POSITIVE) {
                self.known_true |=
                    Props::NONNEGATIVE | Props::NONZERO | Props::EXTENDED_REAL | Props::COMMUTATIVE;
                self.known_false |=
                    Props::NEGATIVE | Props::ZERO | Props::NONPOSITIVE | Props::IMAGINARY;
            }

            // negative → nonpositive, nonzero, extended_real, !positive, !zero, !nonnegative
            if self.known_true.contains(Props::NEGATIVE) {
                self.known_true |=
                    Props::NONPOSITIVE | Props::NONZERO | Props::EXTENDED_REAL | Props::COMMUTATIVE;
                self.known_false |=
                    Props::POSITIVE | Props::ZERO | Props::NONNEGATIVE | Props::IMAGINARY;
            }

            // nonnegative → extended_real, !negative
            if self.known_true.contains(Props::NONNEGATIVE) {
                self.known_true |= Props::EXTENDED_REAL | Props::COMMUTATIVE;
                self.known_false |= Props::NEGATIVE | Props::IMAGINARY;
            }

            // nonpositive → extended_real, !positive
            if self.known_true.contains(Props::NONPOSITIVE) {
                self.known_true |= Props::EXTENDED_REAL | Props::COMMUTATIVE;
                self.known_false |= Props::POSITIVE | Props::IMAGINARY;
            }

            // zero → even, finite, nonneg, nonpos, real, integer, rational, algebraic, complex, !nonzero
            if self.known_true.contains(Props::ZERO) {
                self.known_true |= Props::EVEN
                    | Props::FINITE
                    | Props::NONNEGATIVE
                    | Props::NONPOSITIVE
                    | Props::REAL
                    | Props::INTEGER
                    | Props::RATIONAL
                    | Props::ALGEBRAIC
                    | Props::COMPLEX
                    | Props::COMMUTATIVE
                    | Props::HERMITIAN;
                self.known_false |= Props::NONZERO
                    | Props::POSITIVE
                    | Props::NEGATIVE
                    | Props::ODD
                    | Props::PRIME
                    | Props::COMPOSITE
                    | Props::INFINITE
                    | Props::IMAGINARY
                    | Props::IRRATIONAL
                    | Props::TRANSCENDENTAL;
            }

            // nonzero → !zero
            if self.known_true.contains(Props::NONZERO) {
                self.known_false |= Props::ZERO;
            }

            // even → integer
            if self.known_true.contains(Props::EVEN) {
                self.known_true |= Props::INTEGER
                    | Props::RATIONAL
                    | Props::ALGEBRAIC
                    | Props::REAL
                    | Props::COMPLEX
                    | Props::FINITE
                    | Props::COMMUTATIVE
                    | Props::HERMITIAN;
                self.known_false |= Props::ODD
                    | Props::IMAGINARY
                    | Props::INFINITE
                    | Props::IRRATIONAL
                    | Props::TRANSCENDENTAL;
            }

            // odd → integer, nonzero, !even
            if self.known_true.contains(Props::ODD) {
                self.known_true |= Props::INTEGER
                    | Props::NONZERO
                    | Props::RATIONAL
                    | Props::ALGEBRAIC
                    | Props::REAL
                    | Props::COMPLEX
                    | Props::FINITE
                    | Props::COMMUTATIVE
                    | Props::HERMITIAN;
                self.known_false |= Props::EVEN
                    | Props::ZERO
                    | Props::IMAGINARY
                    | Props::INFINITE
                    | Props::IRRATIONAL
                    | Props::TRANSCENDENTAL;
            }

            // prime → integer, positive, nonzero, !composite, !zero
            if self.known_true.contains(Props::PRIME) {
                self.known_true |= Props::INTEGER
                    | Props::POSITIVE
                    | Props::NONNEGATIVE
                    | Props::NONZERO
                    | Props::RATIONAL
                    | Props::ALGEBRAIC
                    | Props::REAL
                    | Props::COMPLEX
                    | Props::FINITE
                    | Props::COMMUTATIVE
                    | Props::HERMITIAN;
                self.known_false |= Props::COMPOSITE
                    | Props::ZERO
                    | Props::NEGATIVE
                    | Props::NONPOSITIVE
                    | Props::IMAGINARY
                    | Props::INFINITE
                    | Props::IRRATIONAL
                    | Props::TRANSCENDENTAL;
            }

            // composite → integer, positive, !prime
            if self.known_true.contains(Props::COMPOSITE) {
                self.known_true |= Props::INTEGER
                    | Props::POSITIVE
                    | Props::NONNEGATIVE
                    | Props::NONZERO
                    | Props::RATIONAL
                    | Props::ALGEBRAIC
                    | Props::REAL
                    | Props::COMPLEX
                    | Props::FINITE
                    | Props::COMMUTATIVE
                    | Props::HERMITIAN;
                self.known_false |= Props::PRIME
                    | Props::ZERO
                    | Props::NEGATIVE
                    | Props::NONPOSITIVE
                    | Props::IMAGINARY
                    | Props::INFINITE
                    | Props::IRRATIONAL
                    | Props::TRANSCENDENTAL;
            }

            // finite → !infinite
            if self.known_true.contains(Props::FINITE) {
                self.known_false |= Props::INFINITE;
            }

            // infinite → !finite, nonzero, and nothing that lives in ℂ (which
            // is finite by definition): !complex, !real, !imaginary, …
            if self.known_true.contains(Props::INFINITE) {
                self.known_true |= Props::NONZERO;
                self.known_false |= Props::FINITE
                    | Props::ZERO
                    | Props::COMPLEX
                    | Props::REAL
                    | Props::IMAGINARY
                    | Props::INTEGER
                    | Props::RATIONAL
                    | Props::IRRATIONAL
                    | Props::ALGEBRAIC
                    | Props::TRANSCENDENTAL
                    | Props::EVEN
                    | Props::ODD
                    | Props::PRIME
                    | Props::COMPOSITE;
            }

            // ── Contrapositives ────────────────────────────────────

            // !complex → !real, !rational, !integer, !algebraic, !transcendental, !imaginary
            // (the sign properties are *not* excluded: `oo` is !complex yet positive)
            if self.known_false.contains(Props::COMPLEX) {
                self.known_false |= Props::REAL
                    | Props::RATIONAL
                    | Props::INTEGER
                    | Props::ALGEBRAIC
                    | Props::TRANSCENDENTAL
                    | Props::IMAGINARY
                    | Props::IRRATIONAL
                    | Props::ZERO
                    | Props::EVEN
                    | Props::ODD
                    | Props::PRIME
                    | Props::COMPOSITE
                    | Props::HERMITIAN
                    | Props::ANTIHERMITIAN;
            }

            // !real → !rational, !integer, !irrational, !zero, …
            // (again not the sign properties, which are extended-real notions)
            if self.known_false.contains(Props::REAL) {
                self.known_false |= Props::RATIONAL
                    | Props::INTEGER
                    | Props::IRRATIONAL
                    | Props::EVEN
                    | Props::ODD
                    | Props::PRIME
                    | Props::COMPOSITE
                    | Props::ZERO;
            }

            // !extended_real → !positive, !negative, !nonnegative, !nonpositive, !zero
            if self.known_false.contains(Props::EXTENDED_REAL) {
                self.known_false |= Props::POSITIVE
                    | Props::NEGATIVE
                    | Props::NONNEGATIVE
                    | Props::NONPOSITIVE
                    | Props::ZERO;
            }

            // !rational → !integer
            if self.known_false.contains(Props::RATIONAL) {
                self.known_false |= Props::INTEGER
                    | Props::EVEN
                    | Props::ODD
                    | Props::PRIME
                    | Props::COMPOSITE
                    | Props::ZERO;
            }

            // !integer → !even, !odd, !prime, !composite
            if self.known_false.contains(Props::INTEGER) {
                self.known_false |= Props::EVEN | Props::ODD | Props::PRIME | Props::COMPOSITE;
            }

            // !finite → infinite, !complex (ℂ ⊂ finite)
            if self.known_false.contains(Props::FINITE) {
                self.known_true |= Props::INFINITE;
                self.known_false |= Props::COMPLEX;
            }

            // !infinite → finite
            if self.known_false.contains(Props::INFINITE) {
                self.known_true |= Props::FINITE;
            }

            // !extended_real → !real (and hence !rational, … via the !real rule)
            if self.known_false.contains(Props::EXTENDED_REAL) {
                self.known_false |= Props::REAL;
            }

            // ── Beta rules: join conditions ──────────────────────

            // nonnegative ∧ nonzero → positive
            if self
                .known_true
                .contains(Props::NONNEGATIVE | Props::NONZERO)
            {
                self.known_true.insert(Props::POSITIVE);
            }

            // nonpositive ∧ nonzero → negative
            if self
                .known_true
                .contains(Props::NONPOSITIVE | Props::NONZERO)
            {
                self.known_true.insert(Props::NEGATIVE);
            }

            // nonnegative ∧ nonpositive → zero
            if self
                .known_true
                .contains(Props::NONNEGATIVE | Props::NONPOSITIVE)
            {
                self.known_true.insert(Props::ZERO);
            }

            // real ∧ !rational → irrational
            if self.known_true.contains(Props::REAL) && self.known_false.contains(Props::RATIONAL) {
                self.known_true.insert(Props::IRRATIONAL);
            }

            // complex ∧ !algebraic → transcendental
            // (only if also real or if we know it's not imaginary-only)
            if self.known_true.contains(Props::COMPLEX)
                && self.known_false.contains(Props::ALGEBRAIC)
                && (self.known_true.contains(Props::REAL)
                    || self.known_true.contains(Props::IMAGINARY))
            {
                self.known_true.insert(Props::TRANSCENDENTAL);
            }

            // integer ∧ !even → odd
            if self.known_true.contains(Props::INTEGER) && self.known_false.contains(Props::EVEN) {
                self.known_true.insert(Props::ODD);
            }

            // integer ∧ !odd → even
            if self.known_true.contains(Props::INTEGER) && self.known_false.contains(Props::ODD) {
                self.known_true.insert(Props::EVEN);
            }

            // transcendental ∧ real → irrational
            // (complex transcendental numbers exist, so only over ℝ)
            if self
                .known_true
                .contains(Props::TRANSCENDENTAL | Props::REAL)
            {
                self.known_true.insert(Props::IRRATIONAL);
            }

            // extended_real ∧ finite → real
            if self
                .known_true
                .contains(Props::EXTENDED_REAL | Props::FINITE)
            {
                self.known_true.insert(Props::REAL);
            }

            // !real ∧ finite → !extended_real  (contrapositive of the above)
            if self.known_false.contains(Props::REAL) && self.known_true.contains(Props::FINITE) {
                self.known_false.insert(Props::EXTENDED_REAL);
            }

            // extended_real ∧ !real → infinite  (extended_real = real ∨ ±∞)
            if self.known_true.contains(Props::EXTENDED_REAL)
                && self.known_false.contains(Props::REAL)
            {
                self.known_true.insert(Props::INFINITE);
            }

            // Check fixpoint.
            if self.known_true == old_true && self.known_false == old_false {
                break;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumption cache + property handlers
// ═══════════════════════════════════════════════════════════════════════════

/// Cache of computed assumptions, indexed by `ExprId`.
///
/// Lives on the [`Context`](crate::api::context::Context) separately from
/// the arena to avoid lock-ordering issues.
#[derive(Debug, Default)]
pub struct AssumptionCache {
    cache: FxHashMap<ExprId, Assumptions>,
}

impl AssumptionCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Query a property for an expression.
    ///
    /// If the property is already cached, returns immediately.
    /// Otherwise computes it from the expression structure and caches
    /// the result.
    pub fn query(&mut self, arena: &Arena, id: ExprId, prop: Props) -> Option<bool> {
        // Check cache first.
        if let Some(cached) = self.cache.get(&id)
            && let Some(val) = cached.query(prop)
        {
            return Some(val);
        }

        // Compute from the expression structure.
        let assumptions = self.compute(arena, id);

        // Cache and return.
        let entry = self.cache.entry(id).or_default();
        entry.merge(&assumptions);
        entry.query(prop)
    }

    /// Compute assumptions for an expression from its structure.
    ///
    /// This is the equivalent of SymPy's `_eval_is_*` methods, but
    /// dispatched via a single match on the node type.
    fn compute(&mut self, arena: &Arena, id: ExprId) -> Assumptions {
        // Check cache first to avoid recomputation.
        if let Some(cached) = self.cache.get(&id) {
            return *cached;
        }

        let node = arena.node(id).clone();
        let result = match node {
            ExprNode::Num(nid) => self.compute_num(arena, nid),
            ExprNode::Symbol(sid) => compute_symbol(arena, sid),
            ExprNode::Pi => compute_pi(),
            ExprNode::E => compute_e(),
            ExprNode::ImaginaryUnit => compute_imaginary_unit(),
            ExprNode::Infinity => compute_infinity(),
            ExprNode::NegInfinity => compute_neg_infinity(),
            ExprNode::ComplexInfinity => compute_complex_infinity(),
            ExprNode::NaN => compute_nan(),
            ExprNode::Add(ref args) => self.compute_add(arena, args),
            ExprNode::Mul(ref args) => self.compute_mul(arena, args),
            ExprNode::Pow(base, exp) => self.compute_pow(arena, base, exp),
            ExprNode::Neg(inner) => self.compute_neg(arena, inner),
            ExprNode::Sin(inner) | ExprNode::Cos(inner) | ExprNode::Tan(inner) => {
                self.compute_trig(arena, inner)
            }
            ExprNode::Exp(inner) => self.compute_exp(arena, inner),
            ExprNode::Ln(inner) => self.compute_ln(arena, inner),
            ExprNode::Abs(inner) => self.compute_abs(arena, inner),
            ExprNode::Sinh(inner)
            | ExprNode::Tanh(inner)
            | ExprNode::Asinh(inner)
            | ExprNode::Atanh(inner) => self.compute_hyp_odd(arena, inner),
            ExprNode::Cosh(inner) => self.compute_cosh(arena, inner),
            ExprNode::Asin(inner) | ExprNode::Acos(inner) | ExprNode::Atan(inner) => {
                self.compute_inverse_trig(arena, inner)
            }
            ExprNode::Acosh(inner) => self.compute_acosh(arena, inner),
            // ── 0.2 additions: named constants, complex-analysis nodes, specials ──
            ExprNode::EulerGamma | ExprNode::Catalan => compute_positive_real_constant(),
            ExprNode::GoldenRatio => compute_golden_ratio(),
            ExprNode::Re(_) | ExprNode::Im(_) | ExprNode::Arg(_) => compute_real_valued(),
            ExprNode::Conjugate(inner) => self.compute_conjugate(arena, inner),
            ExprNode::KroneckerDelta(..) => compute_kronecker_delta(),
            ExprNode::Floor(inner) | ExprNode::Ceiling(inner) | ExprNode::Sign(inner) => {
                self.compute_real_to_integer(arena, inner)
            }
            ExprNode::Gamma(inner)
            | ExprNode::Digamma(inner)
            | ExprNode::Erf(inner)
            | ExprNode::Erfc(inner)
            | ExprNode::Heaviside(inner)
            | ExprNode::Si(inner)
            | ExprNode::Ei(inner)
            | ExprNode::Zeta(inner) => self.compute_real_to_real(arena, inner),
            ExprNode::LogGamma(inner) | ExprNode::Ci(inner) | ExprNode::Li(inner) => {
                self.compute_positive_to_real(arena, inner)
            }
            ExprNode::Polygamma(_, x) => self.compute_positive_to_real(arena, x),
            ExprNode::Atan2(a, b) => self.compute_all_real_to_real(arena, &[a, b]),
            ExprNode::Min(ref args) | ExprNode::Max(ref args) => {
                self.compute_all_real_to_real(arena, args)
            }
            // ∫ₐᵇ f dx is real when the integrand and both bounds are real
            // (convergence is not asserted here — only the codomain).
            ExprNode::DefiniteIntegral(body, _, lo, hi) => {
                self.compute_all_real_to_real(arena, &[body, lo, hi])
            }
            _ => Assumptions::default(), // Apply, Derivative, Integral, formal nodes, …
        };

        // Close the handler's facts under the implication rules *before*
        // caching, so that a property first asked for later (or first
        // reached through a parent's `compute`) gets the same answer as one
        // asked for directly.
        let mut result = result;
        result.forward_chain();

        // Exact sign of a univariate polynomial with rational coefficients in
        // a real-assumed symbol, when the structural rules left it open
        // (`3u² + 2u + 1` for `u ≥ 0`, `x² - 2x + 2` for real `x`, …).
        if matches!(node, ExprNode::Add(_))
            && result.query(Props::POSITIVE).is_none()
            && result.query(Props::NEGATIVE).is_none()
        {
            let extra = polynomial_sign_facts(arena, id);
            if extra != Assumptions::default() {
                result.merge(&extra);
                result.forward_chain();
            }
        }
        if result.is_contradictory() {
            tracing::debug!(
                node = ?arena.node(id),
                assumptions = %result,
                "assumptions: handler produced a contradictory set"
            );
        }

        // Cache the result.
        self.cache.insert(id, result);
        result
    }

    // ── Atom handlers ──────────────────────────────────────────────

    fn compute_num(&self, arena: &Arena, nid: crate::base::node::NumId) -> Assumptions {
        let val = arena.num(nid);
        let mut a = Assumptions::default();

        // All rationals are rational, real, algebraic, complex, finite, commutative.
        a.known_true |= Props::RATIONAL
            | Props::ALGEBRAIC
            | Props::REAL
            | Props::COMPLEX
            | Props::FINITE
            | Props::COMMUTATIVE
            | Props::HERMITIAN;
        a.known_false |=
            Props::INFINITE | Props::IMAGINARY | Props::TRANSCENDENTAL | Props::IRRATIONAL;

        if val.is_integer() {
            a.known_true |= Props::INTEGER;
        } else {
            a.known_false |=
                Props::INTEGER | Props::EVEN | Props::ODD | Props::PRIME | Props::COMPOSITE;
        }

        if val.is_zero() {
            a.known_true |= Props::ZERO
                | Props::EVEN
                | Props::NONNEGATIVE
                | Props::NONPOSITIVE
                | Props::INTEGER;
            a.known_false |= Props::NONZERO
                | Props::POSITIVE
                | Props::NEGATIVE
                | Props::ODD
                | Props::PRIME
                | Props::COMPOSITE;
        } else {
            a.known_true |= Props::NONZERO;
            a.known_false |= Props::ZERO;

            if val.is_positive() {
                a.known_true |= Props::POSITIVE | Props::NONNEGATIVE;
                a.known_false |= Props::NEGATIVE | Props::NONPOSITIVE;
            } else {
                a.known_true |= Props::NEGATIVE | Props::NONPOSITIVE;
                a.known_false |= Props::POSITIVE | Props::NONNEGATIVE;
            }

            // Even/odd for integers.
            if val.is_integer() {
                let int_val = val.to_integer();
                if (&int_val % BigInt::from(2)).is_zero() {
                    a.known_true.insert(Props::EVEN);
                    a.known_false.insert(Props::ODD);
                } else {
                    a.known_true.insert(Props::ODD);
                    a.known_false.insert(Props::EVEN);
                }

                // Prime check for small positive integers.
                if val.is_positive() {
                    let n: Option<u64> = int_val.try_into().ok();
                    if let Some(n) = n {
                        if is_small_prime(n) {
                            a.known_true.insert(Props::PRIME);
                            a.known_false.insert(Props::COMPOSITE);
                        } else if n > 1 {
                            a.known_false.insert(Props::PRIME);
                            a.known_true.insert(Props::COMPOSITE);
                        }
                    }
                }
            }
        }

        a.forward_chain();
        a
    }

    // ── Composite handlers ─────────────────────────────────────────

    fn compute_add(
        &mut self,
        arena: &Arena,
        args: &smallvec::SmallVec<[ExprId; 6]>,
    ) -> Assumptions {
        let mut a = Assumptions::default();
        if args.is_empty() {
            return a;
        }

        // Collect child properties.
        let mut all_integer = true;
        let mut all_rational = true;
        let mut all_real = true;
        let mut all_complex = true;
        let mut all_finite = true;
        let mut all_commutative = true;

        let mut all_nonneg = true;
        let mut any_positive = false;

        let mut all_nonpos = true;
        let mut any_negative = false;

        for &child in args.iter() {
            let child_a = self.compute(arena, child);

            if child_a.query(Props::INTEGER) != Some(true) {
                all_integer = false;
            }
            if child_a.query(Props::RATIONAL) != Some(true) {
                all_rational = false;
            }
            if child_a.query(Props::REAL) != Some(true) {
                all_real = false;
            }
            if child_a.query(Props::COMPLEX) != Some(true) {
                all_complex = false;
            }
            if child_a.query(Props::FINITE) != Some(true) {
                all_finite = false;
            }
            if child_a.query(Props::COMMUTATIVE) != Some(true) {
                all_commutative = false;
            }

            if child_a.query(Props::POSITIVE) == Some(true) {
                any_positive = true;
            }
            if child_a.query(Props::NONNEGATIVE) != Some(true) {
                all_nonneg = false;
            }
            if child_a.query(Props::NEGATIVE) == Some(true) {
                any_negative = true;
            }
            if child_a.query(Props::NONPOSITIVE) != Some(true) {
                all_nonpos = false;
            }
        }

        if all_integer {
            a.known_true |= Props::INTEGER;
        }
        if all_rational {
            a.known_true |= Props::RATIONAL;
        }
        if all_real {
            a.known_true |= Props::REAL;
        }
        if all_complex {
            a.known_true |= Props::COMPLEX;
        }
        if all_finite {
            a.known_true |= Props::FINITE;
        }
        if all_commutative {
            a.known_true |= Props::COMMUTATIVE;
        }

        // Sign deduction: all nonneg with at least one positive → positive.
        if all_nonneg && any_positive {
            a.known_true |= Props::POSITIVE;
        } else if all_nonneg {
            a.known_true |= Props::NONNEGATIVE;
        }

        if all_nonpos && any_negative {
            a.known_true |= Props::NEGATIVE;
        } else if all_nonpos {
            a.known_true |= Props::NONPOSITIVE;
        }

        a.forward_chain();
        a
    }

    fn compute_mul(
        &mut self,
        arena: &Arena,
        args: &smallvec::SmallVec<[ExprId; 6]>,
    ) -> Assumptions {
        let mut a = Assumptions::default();
        if args.is_empty() {
            return a;
        }

        let mut all_integer = true;
        let mut all_rational = true;
        let mut all_real = true;
        let mut all_complex = true;
        let mut all_finite = true;
        let mut all_commutative = true;
        let mut any_zero = false;
        let mut negative_count = 0u32;
        let mut all_nonzero = true;
        let mut sign_known = true;
        let mut imaginary_count: usize = 0;
        let mut real_count: usize = 0;
        let mut any_even: bool = false;
        let mut all_odd: bool = true;

        for &child in args.iter() {
            let child_a = self.compute(arena, child);

            if child_a.query(Props::INTEGER) != Some(true) {
                all_integer = false;
            }
            if child_a.query(Props::RATIONAL) != Some(true) {
                all_rational = false;
            }
            if child_a.query(Props::REAL) != Some(true) {
                all_real = false;
            }
            if child_a.query(Props::COMPLEX) != Some(true) {
                all_complex = false;
            }
            if child_a.query(Props::FINITE) != Some(true) {
                all_finite = false;
            }
            if child_a.query(Props::COMMUTATIVE) != Some(true) {
                all_commutative = false;
            }

            if child_a.query(Props::ZERO) == Some(true) {
                any_zero = true;
            }
            if child_a.query(Props::NONZERO) != Some(true) {
                all_nonzero = false;
            }

            match child_a.query(Props::NEGATIVE) {
                Some(true) => {
                    negative_count += 1;
                }
                Some(false) => { /* positive or zero — doesn't flip sign */ }
                None => {
                    sign_known = false;
                }
            }

            if child_a.query(Props::EVEN) == Some(true) {
                any_even = true;
                all_odd = false;
            } else if child_a.query(Props::ODD) == Some(true) {
                // all_odd stays true
            } else {
                all_odd = false;
            }

            if child_a.query(Props::IMAGINARY) == Some(true) {
                imaginary_count += 1;
            }
            if child_a.query(Props::REAL) == Some(true) {
                real_count += 1;
            }
        }

        if all_integer {
            a.known_true |= Props::INTEGER;
            if any_even {
                a.known_true |= Props::EVEN;
                a.known_false |= Props::ODD;
            } else if all_odd && !args.is_empty() {
                a.known_true |= Props::ODD;
                a.known_false |= Props::EVEN;
            }
        }
        if all_rational {
            a.known_true |= Props::RATIONAL;
        }
        if all_real {
            a.known_true |= Props::REAL;
        }
        // Handle mix of real and imaginary factors.
        // real * imaginary = imaginary; imaginary * imaginary = real
        if imaginary_count > 0 && real_count + imaginary_count == args.len() {
            if imaginary_count.is_multiple_of(2) {
                a.known_true |= Props::REAL;
            } else {
                a.known_true |= Props::IMAGINARY;
            }
        }
        if all_complex {
            a.known_true |= Props::COMPLEX;
        }
        if all_finite {
            a.known_true |= Props::FINITE;
        }
        if all_commutative {
            a.known_true |= Props::COMMUTATIVE;
        }

        if any_zero && all_finite {
            a.known_true |= Props::ZERO;
        }

        if all_nonzero {
            a.known_true |= Props::NONZERO;
        }

        // Sign: product of reals — negative count determines sign.
        if all_real && sign_known && all_nonzero {
            if negative_count.is_multiple_of(2) {
                a.known_true |= Props::POSITIVE;
            } else {
                a.known_true |= Props::NEGATIVE;
            }
        }

        a.forward_chain();
        a
    }

    fn compute_pow(&mut self, arena: &Arena, base: ExprId, exp: ExprId) -> Assumptions {
        let mut a = Assumptions::default();

        let base_a = self.compute(arena, base);
        let exp_a = self.compute(arena, exp);

        // If base and exp are both complex → result is complex.
        if base_a.query(Props::COMPLEX) == Some(true) && exp_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }

        // real^real → complex (at least).
        if base_a.query(Props::REAL) == Some(true) && exp_a.query(Props::REAL) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }

        // positive_base^real_exp → positive, real
        if base_a.query(Props::POSITIVE) == Some(true) && exp_a.query(Props::REAL) == Some(true) {
            a.known_true |= Props::POSITIVE | Props::REAL;
        }

        // nonneg_base^positive_exp → nonneg
        if base_a.query(Props::NONNEGATIVE) == Some(true)
            && exp_a.query(Props::POSITIVE) == Some(true)
        {
            a.known_true |= Props::NONNEGATIVE | Props::REAL;
        }

        // real_base^even_integer → nonneg
        if base_a.query(Props::REAL) == Some(true) && exp_a.query(Props::EVEN) == Some(true) {
            a.known_true |= Props::NONNEGATIVE | Props::REAL;
        }

        // real_base^integer_exp → real (wherever defined)
        if base_a.query(Props::REAL) == Some(true) && exp_a.query(Props::INTEGER) == Some(true) {
            a.known_true |= Props::REAL;
        }

        // integer^nonneg_integer → integer
        if base_a.query(Props::INTEGER) == Some(true)
            && exp_a.query(Props::INTEGER) == Some(true)
            && exp_a.query(Props::NONNEGATIVE) == Some(true)
        {
            a.known_true |= Props::INTEGER;
        }

        // rational^integer → rational
        if base_a.query(Props::RATIONAL) == Some(true) && exp_a.query(Props::INTEGER) == Some(true)
        {
            a.known_true |= Props::RATIONAL;
        }

        // finite^finite → finite
        if base_a.query(Props::FINITE) == Some(true) && exp_a.query(Props::FINITE) == Some(true) {
            a.known_true |= Props::FINITE;
        }

        a.known_true |= Props::COMMUTATIVE;
        a.forward_chain();
        a
    }

    fn compute_neg(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        let mut a = self.compute(arena, inner);
        // Negation flips sign properties.
        let was_positive = a.query(Props::POSITIVE);
        let was_negative = a.query(Props::NEGATIVE);
        let was_nonneg = a.query(Props::NONNEGATIVE);
        let was_nonpos = a.query(Props::NONPOSITIVE);

        // Clear sign bits and re-set them flipped.
        a.known_true
            .remove(Props::POSITIVE | Props::NEGATIVE | Props::NONNEGATIVE | Props::NONPOSITIVE);
        a.known_false
            .remove(Props::POSITIVE | Props::NEGATIVE | Props::NONNEGATIVE | Props::NONPOSITIVE);

        if was_positive == Some(true) {
            a.known_true |= Props::NEGATIVE;
            a.known_false |= Props::POSITIVE;
        }
        if was_positive == Some(false) {
            a.known_false |= Props::NEGATIVE;
        }
        if was_negative == Some(true) {
            a.known_true |= Props::POSITIVE;
            a.known_false |= Props::NEGATIVE;
        }
        if was_negative == Some(false) {
            a.known_false |= Props::POSITIVE;
        }
        if was_nonneg == Some(true) {
            a.known_true |= Props::NONPOSITIVE;
        }
        if was_nonpos == Some(true) {
            a.known_true |= Props::NONNEGATIVE;
        }

        a.forward_chain();
        a
    }

    fn compute_trig(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        let mut a = Assumptions::default();
        let inner_a = self.compute(arena, inner);

        // sin/cos/tan of real → real.
        if inner_a.query(Props::REAL) == Some(true) {
            a.known_true |= Props::REAL;
        }

        if inner_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }

        if inner_a.query(Props::FINITE) == Some(true) {
            a.known_true |= Props::FINITE;
        }

        a.known_true |= Props::COMMUTATIVE;
        a.forward_chain();
        a
    }

    fn compute_exp(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        let mut a = Assumptions::default();
        let inner_a = self.compute(arena, inner);

        // exp(real) → positive, real
        if inner_a.query(Props::REAL) == Some(true) {
            a.known_true |= Props::POSITIVE | Props::REAL | Props::NONZERO;
        }

        if inner_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX | Props::NONZERO;
        }

        if inner_a.query(Props::FINITE) == Some(true) {
            a.known_true |= Props::FINITE | Props::NONZERO;
        }

        a.known_true |= Props::COMMUTATIVE;
        a.forward_chain();
        a
    }

    fn compute_ln(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        let mut a = Assumptions::default();
        let inner_a = self.compute(arena, inner);

        // ln(positive) → real
        if inner_a.query(Props::POSITIVE) == Some(true) {
            a.known_true |= Props::REAL;
        }

        if inner_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }

        if inner_a.query(Props::FINITE) == Some(true) && inner_a.query(Props::NONZERO) == Some(true)
        {
            a.known_true |= Props::FINITE;
        }

        a.known_true |= Props::COMMUTATIVE;
        a.forward_chain();
        a
    }

    fn compute_abs(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        let mut a = Assumptions::default();
        let inner_a = self.compute(arena, inner);

        // abs(anything) → nonneg, real
        a.known_true |= Props::NONNEGATIVE | Props::REAL;

        if inner_a.query(Props::ZERO) == Some(true) {
            a.known_true |= Props::ZERO;
        }

        if inner_a.query(Props::NONZERO) == Some(true) {
            a.known_true |= Props::POSITIVE;
        }

        if inner_a.query(Props::FINITE) == Some(true) {
            a.known_true |= Props::FINITE;
        }

        if inner_a.query(Props::INTEGER) == Some(true) {
            a.known_true |= Props::INTEGER;
        }

        a.known_true |= Props::COMMUTATIVE;
        a.forward_chain();
        a
    }

    fn compute_hyp_odd(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        // sinh, tanh, asinh, atanh: real → real, complex → complex, odd function
        let inner_a = self.compute(arena, inner);
        let mut a = Assumptions::default();
        a.known_true |= Props::COMMUTATIVE | Props::FINITE;
        if inner_a.query(Props::REAL) == Some(true) {
            a.known_true |= Props::REAL;
        }
        if inner_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }
        a.forward_chain();
        a
    }

    fn compute_cosh(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        // cosh: real → real, positive (cosh(x) >= 1 for real x)
        let inner_a = self.compute(arena, inner);
        let mut a = Assumptions::default();
        a.known_true |= Props::COMMUTATIVE | Props::FINITE;
        if inner_a.query(Props::REAL) == Some(true) {
            a.known_true |= Props::REAL | Props::POSITIVE;
        }
        if inner_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }
        a.forward_chain();
        a
    }

    fn compute_inverse_trig(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        // asin, acos, atan: real → real (for appropriate domain), complex → complex
        let inner_a = self.compute(arena, inner);
        let mut a = Assumptions::default();
        a.known_true |= Props::COMMUTATIVE | Props::FINITE;
        if inner_a.query(Props::REAL) == Some(true) {
            a.known_true |= Props::REAL; // technically only for |x|≤1 for asin/acos, but conservative
        }
        if inner_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }
        a.forward_chain();
        a
    }

    fn compute_acosh(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        // acosh: real >= 1 → real, nonnegative
        let inner_a = self.compute(arena, inner);
        let mut a = Assumptions::default();
        a.known_true |= Props::COMMUTATIVE | Props::FINITE;
        if inner_a.query(Props::REAL) == Some(true) {
            // acosh is real for x >= 1, complex otherwise. Conservative: just say complex.
            a.known_true |= Props::COMPLEX;
            if inner_a.query(Props::POSITIVE) == Some(true) {
                a.known_true |= Props::REAL | Props::NONNEGATIVE;
            }
        }
        if inner_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }
        a.forward_chain();
        a
    }

    /// `conjugate(z)` shares realness/imaginariness/finiteness with `z`
    /// (and is zero/non-zero exactly when `z` is).
    fn compute_conjugate(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        let inner_a = self.compute(arena, inner);
        let mut a = Assumptions::default();
        a.known_true |= Props::COMMUTATIVE;
        for p in [
            Props::REAL,
            Props::COMPLEX,
            Props::IMAGINARY,
            Props::FINITE,
            Props::INFINITE,
            Props::ZERO,
            Props::NONZERO,
            Props::RATIONAL,
            Props::INTEGER,
            Props::ALGEBRAIC,
            Props::POSITIVE,
            Props::NEGATIVE,
            Props::NONNEGATIVE,
            Props::NONPOSITIVE,
        ] {
            match inner_a.query(p) {
                Some(true) => a.known_true |= p,
                Some(false) => a.known_false |= p,
                None => {}
            }
        }
        a.forward_chain();
        a
    }

    /// Functions that map reals to reals (and complex to complex).
    fn compute_real_to_real(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        let inner_a = self.compute(arena, inner);
        let mut a = Assumptions::default();
        a.known_true |= Props::COMMUTATIVE;
        if inner_a.query(Props::REAL) == Some(true) {
            a.known_true |= Props::REAL;
        }
        if inner_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }
        a.forward_chain();
        a
    }

    /// Functions that are real on the positive reals (`ln Γ`, `Ci`, `li`, `ψ⁽ⁿ⁾`).
    fn compute_positive_to_real(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        let inner_a = self.compute(arena, inner);
        let mut a = Assumptions::default();
        a.known_true |= Props::COMMUTATIVE;
        if inner_a.query(Props::POSITIVE) == Some(true) {
            a.known_true |= Props::REAL;
        }
        if inner_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }
        a.forward_chain();
        a
    }

    /// Functions that are integer-valued on the reals (`floor`, `ceiling`, `sign`).
    fn compute_real_to_integer(&mut self, arena: &Arena, inner: ExprId) -> Assumptions {
        let inner_a = self.compute(arena, inner);
        let mut a = Assumptions::default();
        a.known_true |= Props::COMMUTATIVE;
        if inner_a.query(Props::REAL) == Some(true) {
            a.known_true |= Props::INTEGER | Props::REAL | Props::FINITE;
        }
        if inner_a.query(Props::COMPLEX) == Some(true) {
            a.known_true |= Props::COMPLEX;
        }
        a.forward_chain();
        a
    }

    /// Multi-argument functions that are real when every argument is real.
    fn compute_all_real_to_real(&mut self, arena: &Arena, args: &[ExprId]) -> Assumptions {
        let mut all_real = true;
        let mut all_complex = true;
        for &c in args {
            let ca = self.compute(arena, c);
            if ca.query(Props::REAL) != Some(true) {
                all_real = false;
            }
            if ca.query(Props::COMPLEX) != Some(true) {
                all_complex = false;
            }
        }
        let mut a = Assumptions::default();
        a.known_true |= Props::COMMUTATIVE;
        if all_real {
            a.known_true |= Props::REAL;
        }
        if all_complex {
            a.known_true |= Props::COMPLEX;
        }
        a.forward_chain();
        a
    }

    /// Store user-supplied symbol assumptions in the cache.
    ///
    /// The set is normalised with [`Assumptions::normalize_declared`].
    ///
    /// # Panics
    ///
    /// Panics if the declared assumptions are self-contradictory (e.g.
    /// `Positive` together with `Negative`, or `Integer` with
    /// `Irrational`).  Declaring impossible facts about a symbol is a
    /// programming error, on a par with mixing expressions from two
    /// contexts.
    pub fn set_symbol_assumptions(&mut self, id: ExprId, assumptions: Assumptions) {
        let mut assumptions = assumptions;
        assumptions.normalize_declared();
        assert!(
            !assumptions.is_contradictory(),
            "contradictory assumptions declared on symbol {id:?}: {assumptions} \
             (properties {} are both asserted and denied)",
            assumptions.known_true & assumptions.known_false
        );
        self.cache.insert(id, assumptions);
    }
}

/// `γ` and `G`: positive real constants whose (ir)rationality is unproven.
fn compute_positive_real_constant() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::POSITIVE
        | Props::NONNEGATIVE
        | Props::NONZERO
        | Props::REAL
        | Props::COMPLEX
        | Props::FINITE
        | Props::COMMUTATIVE
        | Props::HERMITIAN;
    a.known_false |= Props::NEGATIVE
        | Props::NONPOSITIVE
        | Props::ZERO
        | Props::INTEGER
        | Props::IMAGINARY
        | Props::INFINITE
        | Props::EVEN
        | Props::ODD
        | Props::PRIME
        | Props::COMPOSITE;
    // Deliberately unknown: RATIONAL, IRRATIONAL, ALGEBRAIC, TRANSCENDENTAL.
    a
}

/// `φ = (1+√5)/2`: positive, algebraic, irrational.
fn compute_golden_ratio() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::POSITIVE
        | Props::NONNEGATIVE
        | Props::NONZERO
        | Props::REAL
        | Props::COMPLEX
        | Props::FINITE
        | Props::COMMUTATIVE
        | Props::ALGEBRAIC
        | Props::IRRATIONAL
        | Props::HERMITIAN;
    a.known_false |= Props::NEGATIVE
        | Props::NONPOSITIVE
        | Props::ZERO
        | Props::INTEGER
        | Props::RATIONAL
        | Props::TRANSCENDENTAL
        | Props::IMAGINARY
        | Props::INFINITE
        | Props::EVEN
        | Props::ODD
        | Props::PRIME
        | Props::COMPOSITE;
    a
}

/// `re`, `im`, `arg`: real-valued by definition.
fn compute_real_valued() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::REAL | Props::COMPLEX | Props::COMMUTATIVE | Props::HERMITIAN;
    a.known_false |= Props::IMAGINARY;
    a
}

/// `δᵢⱼ ∈ {0, 1}`: a non-negative integer.
fn compute_kronecker_delta() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::INTEGER | Props::NONNEGATIVE | Props::FINITE | Props::COMMUTATIVE;
    a.known_false |= Props::NEGATIVE;
    a.forward_chain();
    a
}

// ═══════════════════════════════════════════════════════════════════════════
// Constant property handlers (no arena access needed)
// ═══════════════════════════════════════════════════════════════════════════

fn compute_symbol(arena: &Arena, sid: SymbolId) -> Assumptions {
    // Symbols get default assumptions from the symbol table.
    // The base assumption for all symbols is commutative = true (like SymPy).
    let mut a = arena.symbol_assumptions(sid);
    if a.query(Props::COMMUTATIVE).is_none() {
        a.assert_true(Props::COMMUTATIVE);
    }
    a
}

fn compute_pi() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::POSITIVE
        | Props::NONNEGATIVE
        | Props::NONZERO
        | Props::REAL
        | Props::EXTENDED_REAL
        | Props::COMPLEX
        | Props::FINITE
        | Props::COMMUTATIVE
        | Props::TRANSCENDENTAL
        | Props::IRRATIONAL
        | Props::HERMITIAN;
    a.known_false |= Props::NEGATIVE
        | Props::NONPOSITIVE
        | Props::ZERO
        | Props::INTEGER
        | Props::RATIONAL
        | Props::ALGEBRAIC
        | Props::IMAGINARY
        | Props::INFINITE
        | Props::EVEN
        | Props::ODD
        | Props::PRIME
        | Props::COMPOSITE;
    a
}

fn compute_e() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::POSITIVE
        | Props::NONNEGATIVE
        | Props::NONZERO
        | Props::REAL
        | Props::EXTENDED_REAL
        | Props::COMPLEX
        | Props::FINITE
        | Props::COMMUTATIVE
        | Props::TRANSCENDENTAL
        | Props::IRRATIONAL
        | Props::HERMITIAN;
    a.known_false |= Props::NEGATIVE
        | Props::NONPOSITIVE
        | Props::ZERO
        | Props::INTEGER
        | Props::RATIONAL
        | Props::ALGEBRAIC
        | Props::IMAGINARY
        | Props::INFINITE
        | Props::EVEN
        | Props::ODD
        | Props::PRIME
        | Props::COMPOSITE;
    a
}

fn compute_imaginary_unit() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::IMAGINARY
        | Props::ALGEBRAIC
        | Props::COMPLEX
        | Props::FINITE
        | Props::COMMUTATIVE
        | Props::NONZERO
        | Props::ANTIHERMITIAN;
    a.known_false |= Props::REAL
        | Props::EXTENDED_REAL
        | Props::RATIONAL
        | Props::INTEGER
        | Props::POSITIVE
        | Props::NEGATIVE
        | Props::NONNEGATIVE
        | Props::NONPOSITIVE
        | Props::ZERO
        | Props::INFINITE
        | Props::TRANSCENDENTAL
        | Props::IRRATIONAL
        | Props::EVEN
        | Props::ODD
        | Props::PRIME
        | Props::COMPOSITE
        | Props::HERMITIAN;
    a
}

/// `oo`: positive, extended real, infinite — and therefore not finite, not
/// real and not complex (ℂ is finite).  Matches SymPy's model of `oo`
/// (`extended_positive`, `infinite`, `!finite`, `!real`, `!complex`).
fn compute_infinity() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::INFINITE
        | Props::EXTENDED_REAL
        | Props::POSITIVE
        | Props::NONNEGATIVE
        | Props::NONZERO
        | Props::COMMUTATIVE;
    a.known_false |= Props::FINITE
        | Props::REAL
        | Props::COMPLEX
        | Props::NEGATIVE
        | Props::NONPOSITIVE
        | Props::ZERO
        | Props::INTEGER
        | Props::RATIONAL
        | Props::IRRATIONAL
        | Props::ALGEBRAIC
        | Props::TRANSCENDENTAL
        | Props::IMAGINARY
        | Props::EVEN
        | Props::ODD
        | Props::PRIME
        | Props::COMPOSITE;
    a.forward_chain();
    debug_assert!(!a.is_contradictory(), "oo: {a}");
    a
}

/// `-oo`: the mirror image of [`compute_infinity`].
fn compute_neg_infinity() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::INFINITE
        | Props::EXTENDED_REAL
        | Props::NEGATIVE
        | Props::NONPOSITIVE
        | Props::NONZERO
        | Props::COMMUTATIVE;
    a.known_false |= Props::FINITE
        | Props::REAL
        | Props::COMPLEX
        | Props::POSITIVE
        | Props::NONNEGATIVE
        | Props::ZERO
        | Props::INTEGER
        | Props::RATIONAL
        | Props::IRRATIONAL
        | Props::ALGEBRAIC
        | Props::TRANSCENDENTAL
        | Props::IMAGINARY
        | Props::EVEN
        | Props::ODD
        | Props::PRIME
        | Props::COMPOSITE;
    a.forward_chain();
    debug_assert!(!a.is_contradictory(), "-oo: {a}");
    a
}

/// `zoo`: infinite with no direction — not extended real, so no sign.
fn compute_complex_infinity() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::INFINITE | Props::NONZERO | Props::COMMUTATIVE;
    a.known_false |= Props::FINITE
        | Props::EXTENDED_REAL
        | Props::REAL
        | Props::COMPLEX
        | Props::POSITIVE
        | Props::NEGATIVE
        | Props::NONNEGATIVE
        | Props::NONPOSITIVE
        | Props::ZERO
        | Props::INTEGER
        | Props::RATIONAL
        | Props::IRRATIONAL
        | Props::ALGEBRAIC
        | Props::TRANSCENDENTAL
        | Props::IMAGINARY
        | Props::EVEN
        | Props::ODD
        | Props::PRIME
        | Props::COMPOSITE;
    a.forward_chain();
    debug_assert!(!a.is_contradictory(), "zoo: {a}");
    a
}

/// `nan`: nothing numeric can be said about it (it is not even
/// `infinite`); only `commutative` holds.
fn compute_nan() -> Assumptions {
    let mut a = Assumptions::default();
    a.known_true |= Props::COMMUTATIVE;
    a
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Simple primality test for small numbers.
fn is_small_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    let mut i = 5u64;
    while i * i <= n {
        if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
            return false;
        }
        i += 6;
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial sign fallback (Sturm)
// ═══════════════════════════════════════════════════════════════════════════

/// Largest degree for which the Sturm-based sign fallback is attempted.
const POLY_SIGN_MAX_DEGREE: usize = 24;

/// Sign facts for `id` when it is a polynomial with rational coefficients
/// in exactly one symbol whose assumptions place it on a known part of the
/// real line.  Decided exactly with square-free factoring and Sturm
/// sequences; returns an empty set when not applicable or undecidable.
fn polynomial_sign_facts(arena: &Arena, id: ExprId) -> Assumptions {
    use crate::api::expr_poly_ext::poly_sign_on_interval;
    use crate::base::extended::Extended;
    use num_rational::Ratio;

    let mut facts = Assumptions::default();
    let syms = crate::base::walk::free_symbols(arena, id);
    let [var] = syms.as_slice() else {
        return facts;
    };
    let ExprNode::Symbol(sid) = arena.node(*var) else {
        return facts;
    };
    let sym = compute_symbol(arena, *sid);
    if sym.query(Props::REAL) != Some(true) {
        return facts;
    }
    let Some(f) = crate::poly::polybridge::expr_to_poly(arena, id, *var) else {
        return facts;
    };
    let deg = f.degree().unwrap_or(0);
    if deg == 0 || deg > POLY_SIGN_MAX_DEGREE {
        return facts;
    }

    // The symbol's domain as a closed interval plus whether the finite
    // endpoint is excluded (`x > 0` rather than `x ≥ 0`).
    let zero = Ratio::from_integer(BigInt::from(0));
    let (lo, hi, open_at) = if sym.query(Props::POSITIVE) == Some(true) {
        (
            Extended::Finite(zero.clone()),
            Extended::PosInf,
            Some(zero.clone()),
        )
    } else if sym.query(Props::NONNEGATIVE) == Some(true) {
        (Extended::Finite(zero.clone()), Extended::PosInf, None)
    } else if sym.query(Props::NEGATIVE) == Some(true) {
        (
            Extended::NegInf,
            Extended::Finite(zero.clone()),
            Some(zero.clone()),
        )
    } else if sym.query(Props::NONPOSITIVE) == Some(true) {
        (Extended::NegInf, Extended::Finite(zero.clone()), None)
    } else {
        (Extended::NegInf, Extended::PosInf, None)
    };

    // A polynomial that is positive (≥ 0) on the closed domain is so on the
    // open one; for strict positivity on an open-ended domain the excluded
    // endpoint may be a root, so re-count roots away from it.
    let strict_on_domain = |g: &crate::poly::Poly| -> bool {
        if poly_sign_on_interval(g, &lo, &hi, true) {
            return true;
        }
        let Some(a) = &open_at else {
            return false;
        };
        // g ≥ 0 on the closed domain, g(a) = 0, and no other root inside.
        if !poly_sign_on_interval(g, &lo, &hi, false) || !g.eval(a).is_zero() {
            return false;
        }
        let bound = crate::poly::sturm::cauchy_bound(g) + Ratio::from_integer(BigInt::from(1));
        let chain = crate::poly::sturm::SturmChain::new(g);
        let roots_closed = match (&lo, &hi) {
            (Extended::Finite(l), Extended::PosInf) => chain.count_roots_in_closed(l, &bound),
            (Extended::NegInf, Extended::Finite(h)) => chain.count_roots_in_closed(&(-bound), h),
            _ => return false,
        };
        roots_closed == 1
    };

    let neg_f = f.neg();
    if strict_on_domain(&f) {
        facts.assert_true(Props::POSITIVE);
    } else if poly_sign_on_interval(&f, &lo, &hi, false) {
        facts.assert_true(Props::NONNEGATIVE);
    } else if strict_on_domain(&neg_f) {
        facts.assert_true(Props::NEGATIVE);
    } else if poly_sign_on_interval(&neg_f, &lo, &hi, false) {
        facts.assert_true(Props::NONPOSITIVE);
    }
    if facts != Assumptions::default() {
        facts.assert_true(Props::REAL);
    }
    facts
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Assumptions basics ──────────────────────────────────────────

    #[test]
    fn default_assumptions_are_all_unknown() {
        let a = Assumptions::default();
        assert_eq!(a.query(Props::REAL), None);
        assert_eq!(a.query(Props::POSITIVE), None);
        assert_eq!(a.query(Props::INTEGER), None);
    }

    #[test]
    fn assert_true_and_query() {
        let mut a = Assumptions::default();
        a.assert_true(Props::POSITIVE);
        assert_eq!(a.query(Props::POSITIVE), Some(true));
    }

    #[test]
    fn assert_false_and_query() {
        let mut a = Assumptions::default();
        a.assert_false(Props::REAL);
        assert_eq!(a.query(Props::REAL), Some(false));
    }

    #[test]
    fn contradiction_detection() {
        let a = Assumptions {
            known_true: Props::POSITIVE,
            known_false: Props::POSITIVE,
        };
        assert!(a.is_contradictory());
    }

    // ── Forward chaining ────────────────────────────────────────────

    #[test]
    fn integer_implies_rational_real_complex() {
        let mut a = Assumptions::default();
        a.assert_true(Props::INTEGER);
        assert_eq!(a.query(Props::RATIONAL), Some(true));
        assert_eq!(a.query(Props::REAL), Some(true));
        assert_eq!(a.query(Props::COMPLEX), Some(true));
        assert_eq!(a.query(Props::FINITE), Some(true));
        assert_eq!(a.query(Props::COMMUTATIVE), Some(true));
        assert_eq!(a.query(Props::ALGEBRAIC), Some(true));
        assert_eq!(a.query(Props::IMAGINARY), Some(false));
        assert_eq!(a.query(Props::INFINITE), Some(false));
    }

    #[test]
    fn positive_implies_extended_real_nonneg_nonzero() {
        let mut a = Assumptions::default();
        a.assert_true(Props::POSITIVE);
        assert_eq!(a.query(Props::NONNEGATIVE), Some(true));
        assert_eq!(a.query(Props::NONZERO), Some(true));
        assert_eq!(a.query(Props::EXTENDED_REAL), Some(true));
        assert_eq!(a.query(Props::REAL), None, "could be +oo");
        assert_eq!(a.query(Props::FINITE), None, "could be +oo");
        assert_eq!(a.query(Props::NEGATIVE), Some(false));
        assert_eq!(a.query(Props::ZERO), Some(false));
        // With finiteness the real line is recovered.
        a.assert_true(Props::FINITE);
        assert_eq!(a.query(Props::REAL), Some(true));
        assert_eq!(a.query(Props::COMPLEX), Some(true));
        assert!(!a.is_contradictory());
    }

    #[test]
    fn positive_and_infinite_is_consistent() {
        let mut a = Assumptions::default();
        a.assert_true(Props::POSITIVE);
        a.assert_true(Props::INFINITE);
        assert!(!a.is_contradictory(), "{a}");
        assert_eq!(a.query(Props::REAL), Some(false));
        assert_eq!(a.query(Props::COMPLEX), Some(false));
        assert_eq!(a.query(Props::EXTENDED_REAL), Some(true));
        // ¬ real ∧ extended_real → infinite
        let mut b = Assumptions::default();
        b.assert_true(Props::EXTENDED_REAL);
        b.assert_false(Props::REAL);
        assert_eq!(b.query(Props::INFINITE), Some(true));
    }

    #[test]
    fn normalize_declared_makes_signed_symbols_finite() {
        let mut pos = Assumptions::default();
        pos.assert_true(Props::POSITIVE);
        pos.normalize_declared();
        assert_eq!(pos.query(Props::FINITE), Some(true));
        assert_eq!(pos.query(Props::REAL), Some(true));
        pos.normalize_declared();
        assert_eq!(pos.query(Props::REAL), Some(true), "idempotent");

        let mut ext = Assumptions::default();
        ext.assert_true(Props::EXTENDED_REAL);
        ext.normalize_declared();
        assert_eq!(ext.query(Props::FINITE), None);

        let mut pos_inf = Assumptions::default();
        pos_inf.assert_true(Props::POSITIVE);
        pos_inf.assert_true(Props::INFINITE);
        pos_inf.normalize_declared();
        assert_eq!(pos_inf.query(Props::FINITE), Some(false));
        assert!(!pos_inf.is_contradictory());

        let mut not_real = Assumptions::default();
        not_real.assert_false(Props::REAL);
        not_real.normalize_declared();
        assert_eq!(not_real.query(Props::POSITIVE), Some(false));
    }

    #[test]
    fn zero_implies_even_integer_nonneg_nonpos() {
        let mut a = Assumptions::default();
        a.assert_true(Props::ZERO);
        assert_eq!(a.query(Props::EVEN), Some(true));
        assert_eq!(a.query(Props::INTEGER), Some(true));
        assert_eq!(a.query(Props::NONNEGATIVE), Some(true));
        assert_eq!(a.query(Props::NONPOSITIVE), Some(true));
        assert_eq!(a.query(Props::NONZERO), Some(false));
        assert_eq!(a.query(Props::POSITIVE), Some(false));
        assert_eq!(a.query(Props::NEGATIVE), Some(false));
    }

    #[test]
    fn not_complex_implies_not_real_not_integer() {
        let mut a = Assumptions::default();
        a.assert_false(Props::COMPLEX);
        assert_eq!(a.query(Props::REAL), Some(false));
        assert_eq!(a.query(Props::INTEGER), Some(false));
        assert_eq!(a.query(Props::RATIONAL), Some(false));
    }

    #[test]
    fn nonneg_and_nonzero_implies_positive() {
        let mut a = Assumptions::default();
        a.assert_true(Props::NONNEGATIVE);
        a.assert_true(Props::NONZERO);
        assert_eq!(a.query(Props::POSITIVE), Some(true));
    }

    #[test]
    fn nonneg_and_nonpos_implies_zero() {
        let mut a = Assumptions::default();
        a.assert_true(Props::NONNEGATIVE);
        a.assert_true(Props::NONPOSITIVE);
        assert_eq!(a.query(Props::ZERO), Some(true));
    }

    #[test]
    fn integer_and_not_even_implies_odd() {
        let mut a = Assumptions::default();
        a.assert_true(Props::INTEGER);
        a.assert_false(Props::EVEN);
        assert_eq!(a.query(Props::ODD), Some(true));
    }

    #[test]
    fn prime_implies_integer_positive() {
        let mut a = Assumptions::default();
        a.assert_true(Props::PRIME);
        assert_eq!(a.query(Props::INTEGER), Some(true));
        assert_eq!(a.query(Props::POSITIVE), Some(true));
        assert_eq!(a.query(Props::COMPOSITE), Some(false));
        assert_eq!(a.query(Props::RATIONAL), Some(true));
        assert_eq!(a.query(Props::NONZERO), Some(true));
    }

    // ── 0.2 forward-chain rules ─────────────────────────────────────────

    #[test]
    fn even_and_odd_imply_integer() {
        let mut e = Assumptions::default();
        e.assert_true(Props::EVEN);
        assert_eq!(e.query(Props::INTEGER), Some(true));
        assert_eq!(e.query(Props::RATIONAL), Some(true));
        assert_eq!(e.query(Props::ODD), Some(false));
        let mut o = Assumptions::default();
        o.assert_true(Props::ODD);
        assert_eq!(o.query(Props::INTEGER), Some(true));
        assert_eq!(o.query(Props::EVEN), Some(false));
        assert_eq!(o.query(Props::NONZERO), Some(true));
        assert_eq!(o.query(Props::ZERO), Some(false));
    }

    #[test]
    fn zero_full_chain() {
        let mut a = Assumptions::default();
        a.assert_true(Props::ZERO);
        for p in [
            Props::EVEN,
            Props::NONNEGATIVE,
            Props::NONPOSITIVE,
            Props::FINITE,
            Props::ALGEBRAIC,
            Props::RATIONAL,
            Props::INTEGER,
            Props::REAL,
            Props::COMPLEX,
            Props::EXTENDED_REAL,
        ] {
            assert_eq!(a.query(p), Some(true), "zero ⇒ {p}");
        }
        for p in [
            Props::NONZERO,
            Props::POSITIVE,
            Props::NEGATIVE,
            Props::ODD,
            Props::PRIME,
            Props::IRRATIONAL,
            Props::TRANSCENDENTAL,
            Props::IMAGINARY,
            Props::INFINITE,
        ] {
            assert_eq!(a.query(p), Some(false), "zero ⇒ ¬{p}");
        }
    }

    #[test]
    fn integer_rational_algebraic_complex_tower() {
        let mut i = Assumptions::default();
        i.assert_true(Props::INTEGER);
        assert_eq!(i.query(Props::RATIONAL), Some(true));
        let mut r = Assumptions::default();
        r.assert_true(Props::RATIONAL);
        assert_eq!(r.query(Props::ALGEBRAIC), Some(true));
        assert_eq!(r.query(Props::IRRATIONAL), Some(false));
        assert_eq!(r.query(Props::TRANSCENDENTAL), Some(false));
        let mut al = Assumptions::default();
        al.assert_true(Props::ALGEBRAIC);
        assert_eq!(al.query(Props::COMPLEX), Some(true));
        assert_eq!(al.query(Props::TRANSCENDENTAL), Some(false));
        // algebraic does not imply real (i is algebraic)
        assert_eq!(al.query(Props::REAL), None);
    }

    #[test]
    fn irrational_implies_real_nonzero_nonint() {
        let mut a = Assumptions::default();
        a.assert_true(Props::IRRATIONAL);
        assert_eq!(a.query(Props::REAL), Some(true));
        assert_eq!(a.query(Props::NONZERO), Some(true));
        assert_eq!(a.query(Props::ZERO), Some(false));
        assert_eq!(a.query(Props::INTEGER), Some(false));
        assert_eq!(a.query(Props::RATIONAL), Some(false));
        assert_eq!(a.query(Props::EVEN), Some(false));
        assert_eq!(a.query(Props::PRIME), Some(false));
        // √2 is irrational and algebraic: neither algebraic nor transcendental is decided
        assert_eq!(a.query(Props::ALGEBRAIC), None);
        assert_eq!(a.query(Props::TRANSCENDENTAL), None);
    }

    #[test]
    fn transcendental_real_implies_irrational_but_complex_alone_does_not() {
        let mut t = Assumptions::default();
        t.assert_true(Props::TRANSCENDENTAL);
        assert_eq!(
            t.query(Props::IRRATIONAL),
            None,
            "transcendental alone: unknown"
        );
        assert_eq!(t.query(Props::RATIONAL), Some(false));
        t.assert_true(Props::REAL);
        assert_eq!(t.query(Props::IRRATIONAL), Some(true));
        assert_eq!(t.query(Props::NONZERO), Some(true));
        // asserting the opposite order also works
        let mut t2 = Assumptions::default();
        t2.assert_true(Props::REAL);
        t2.assert_true(Props::TRANSCENDENTAL);
        assert_eq!(t2.query(Props::IRRATIONAL), Some(true));
    }

    #[test]
    fn positive_full_chain() {
        let mut a = Assumptions::default();
        a.assert_true(Props::POSITIVE);
        assert_eq!(a.query(Props::NONNEGATIVE), Some(true));
        assert_eq!(a.query(Props::NONZERO), Some(true));
        assert_eq!(a.query(Props::EXTENDED_REAL), Some(true));
        assert_eq!(a.query(Props::COMMUTATIVE), Some(true));
        assert_eq!(a.query(Props::NONPOSITIVE), Some(false));
        assert_eq!(a.query(Props::NEGATIVE), Some(false));
        assert_eq!(a.query(Props::ZERO), Some(false));
        assert_eq!(a.query(Props::IMAGINARY), Some(false));
    }

    #[test]
    fn imaginary_implies_nonzero_complex_not_real() {
        let mut a = Assumptions::default();
        a.assert_true(Props::IMAGINARY);
        assert_eq!(a.query(Props::NONZERO), Some(true));
        assert_eq!(a.query(Props::ZERO), Some(false));
        assert_eq!(a.query(Props::COMPLEX), Some(true));
        assert_eq!(a.query(Props::REAL), Some(false));
        assert_eq!(a.query(Props::EXTENDED_REAL), Some(false));
        assert_eq!(a.query(Props::INTEGER), Some(false));
        assert_eq!(a.query(Props::POSITIVE), Some(false));
        assert_eq!(a.query(Props::EVEN), Some(false));
    }

    #[test]
    fn extended_real_rules() {
        let mut r = Assumptions::default();
        r.assert_true(Props::REAL);
        assert_eq!(r.query(Props::EXTENDED_REAL), Some(true));

        let mut ef = Assumptions::default();
        ef.assert_true(Props::EXTENDED_REAL);
        assert_eq!(ef.query(Props::REAL), None, "could be ±∞");
        ef.assert_true(Props::FINITE);
        assert_eq!(ef.query(Props::REAL), Some(true));

        let mut ne = Assumptions::default();
        ne.assert_false(Props::EXTENDED_REAL);
        assert_eq!(ne.query(Props::REAL), Some(false));
        assert_eq!(ne.query(Props::POSITIVE), Some(false));

        let mut nr = Assumptions::default();
        nr.assert_false(Props::REAL);
        assert_eq!(nr.query(Props::EXTENDED_REAL), None, "could be ±∞");
        nr.assert_true(Props::FINITE);
        assert_eq!(nr.query(Props::EXTENDED_REAL), Some(false));

        let arena = Arena::new();
        let mut cache = AssumptionCache::new();
        assert_eq!(
            cache.query(&arena, arena.infinity, Props::EXTENDED_REAL),
            Some(true)
        );
        assert_eq!(
            cache.query(&arena, arena.neg_infinity, Props::EXTENDED_REAL),
            Some(true)
        );
        assert_eq!(
            cache.query(&arena, arena.complex_infinity, Props::EXTENDED_REAL),
            Some(false)
        );
        assert_eq!(
            cache.query(&arena, arena.i_unit, Props::EXTENDED_REAL),
            Some(false)
        );
        assert_eq!(
            cache.query(&arena, arena.pi, Props::EXTENDED_REAL),
            Some(true)
        );
        assert_eq!(
            cache.query(&arena, arena.one, Props::EXTENDED_REAL),
            Some(true)
        );
    }

    #[test]
    fn contradictions_from_new_chains() {
        let mut a = Assumptions::default();
        a.assert_true(Props::IRRATIONAL);
        a.assert_true(Props::ZERO);
        assert!(a.is_contradictory(), "irrational ∧ zero");

        let mut b = Assumptions::default();
        b.assert_true(Props::IMAGINARY);
        b.assert_true(Props::ZERO);
        assert!(b.is_contradictory(), "imaginary ∧ zero");

        let mut c = Assumptions::default();
        c.assert_true(Props::TRANSCENDENTAL);
        c.assert_true(Props::REAL);
        c.assert_true(Props::RATIONAL);
        assert!(c.is_contradictory(), "transcendental ∧ real ∧ rational");

        let mut d = Assumptions::default();
        d.assert_true(Props::EXTENDED_REAL);
        d.assert_true(Props::FINITE);
        d.assert_false(Props::REAL);
        assert!(d.is_contradictory(), "extended_real ∧ finite ∧ ¬real");

        let mut e = Assumptions::default();
        e.assert_true(Props::EVEN);
        e.assert_false(Props::INTEGER);
        assert!(e.is_contradictory(), "even ∧ ¬integer");

        let mut f = Assumptions::default();
        f.assert_true(Props::PRIME);
        f.assert_true(Props::NEGATIVE);
        assert!(f.is_contradictory(), "prime ∧ negative");

        let mut ok = Assumptions::default();
        ok.assert_true(Props::IRRATIONAL);
        ok.assert_true(Props::POSITIVE);
        assert!(!ok.is_contradictory());
    }

    // ── implies / negate / Display ─────────────────────────────────────────────

    #[test]
    fn implies_uses_derived_facts() {
        let mut pos = Assumptions::default();
        pos.assert_true(Props::POSITIVE);
        let real = Assumptions {
            known_true: Props::REAL,
            known_false: Props::empty(),
        };
        let ext_real = Assumptions {
            known_true: Props::EXTENDED_REAL,
            known_false: Props::empty(),
        };
        let not_neg = Assumptions {
            known_true: Props::empty(),
            known_false: Props::NEGATIVE,
        };
        assert!(pos.implies(&ext_real));
        assert!(pos.implies(&not_neg));
        assert!(!pos.implies(&real), "positive alone could be +oo");
        assert!(!real.implies(&pos));
        assert!(
            pos.implies(&Assumptions::default()),
            "everything implies nothing"
        );
        assert!(pos.implies(&pos));
        // unchained self still works
        let raw = Assumptions {
            known_true: Props::INTEGER,
            known_false: Props::empty(),
        };
        assert!(raw.implies(&real));
        // contradictory implies everything
        let bad = Assumptions {
            known_true: Props::ZERO,
            known_false: Props::ZERO,
        };
        assert!(bad.implies(&pos));
    }

    #[test]
    fn negate_is_an_involution_and_flips_value() {
        let all = [
            Assumption::Commutative,
            Assumption::Complex,
            Assumption::Real,
            Assumption::Rational,
            Assumption::Integer,
            Assumption::Algebraic,
            Assumption::Transcendental,
            Assumption::Irrational,
            Assumption::Imaginary,
            Assumption::Positive,
            Assumption::Negative,
            Assumption::NonNegative,
            Assumption::NonPositive,
            Assumption::Zero,
            Assumption::NonZero,
            Assumption::Even,
            Assumption::Odd,
            Assumption::Prime,
            Assumption::Composite,
            Assumption::Finite,
            Assumption::Infinite,
            Assumption::Hermitian,
            Assumption::AntiHermitian,
            Assumption::ExtendedReal,
        ];
        for a in all {
            let n = a.negate();
            assert_ne!(a, n);
            assert_eq!(n.negate(), a, "{a:?}");
            let (p, v) = a.to_prop_value();
            let (np, nv) = n.to_prop_value();
            assert_eq!(p, np, "{a:?}");
            assert_eq!(v, !nv, "{a:?}");
        }
        assert_eq!(Assumption::NotZero.negate(), Assumption::Zero);
        assert_eq!(
            Assumption::NotExtendedReal.to_prop_value(),
            (Props::EXTENDED_REAL, false)
        );
    }

    #[test]
    fn display_lists_active_props() {
        assert_eq!(Assumptions::default().to_string(), "unknown");
        let a = Assumptions {
            known_true: Props::POSITIVE | Props::REAL,
            known_false: Props::ZERO,
        };
        assert_eq!(a.to_string(), "real, positive, !zero");
        let mut chained = Assumptions::default();
        chained.assert_true(Props::PRIME);
        let s = chained.to_string();
        assert!(
            s.starts_with("commutative, complex, real, rational, integer"),
            "{s}"
        );
        assert!(s.contains("prime") && s.contains("!composite"), "{s}");
        assert_eq!(Props::empty().to_string(), "none");
        assert_eq!((Props::EVEN | Props::INTEGER).to_string(), "integer, even");
    }

    // ── Numeric assumptions ─────────────────────────────────────────

    #[test]
    fn zero_value_assumptions() {
        let arena = Arena::new();
        let mut cache = AssumptionCache::new();
        let zero = arena.zero;
        assert_eq!(cache.query(&arena, zero, Props::ZERO), Some(true));
        assert_eq!(cache.query(&arena, zero, Props::INTEGER), Some(true));
        assert_eq!(cache.query(&arena, zero, Props::EVEN), Some(true));
        assert_eq!(cache.query(&arena, zero, Props::POSITIVE), Some(false));
    }

    #[test]
    fn one_value_assumptions() {
        let arena = Arena::new();
        let mut cache = AssumptionCache::new();
        let one = arena.one;
        assert_eq!(cache.query(&arena, one, Props::POSITIVE), Some(true));
        assert_eq!(cache.query(&arena, one, Props::INTEGER), Some(true));
        assert_eq!(cache.query(&arena, one, Props::ODD), Some(true));
        assert_eq!(cache.query(&arena, one, Props::ZERO), Some(false));
    }

    #[test]
    fn neg_one_value_assumptions() {
        let arena = Arena::new();
        let mut cache = AssumptionCache::new();
        let neg_one = arena.neg_one;
        assert_eq!(cache.query(&arena, neg_one, Props::NEGATIVE), Some(true));
        assert_eq!(cache.query(&arena, neg_one, Props::INTEGER), Some(true));
        assert_eq!(cache.query(&arena, neg_one, Props::ODD), Some(true));
    }

    #[test]
    fn rational_value_assumptions() {
        let mut arena = Arena::new();
        let r = arena.rational(1, 3);
        let mut cache = AssumptionCache::new();
        assert_eq!(cache.query(&arena, r, Props::RATIONAL), Some(true));
        assert_eq!(cache.query(&arena, r, Props::REAL), Some(true));
        assert_eq!(cache.query(&arena, r, Props::POSITIVE), Some(true));
        assert_eq!(cache.query(&arena, r, Props::INTEGER), Some(false));
    }

    #[test]
    fn prime_detection() {
        let mut arena = Arena::new();
        let mut cache = AssumptionCache::new();
        let seven = arena.int(7);
        assert_eq!(cache.query(&arena, seven, Props::PRIME), Some(true));
        let four = arena.int(4);
        assert_eq!(cache.query(&arena, four, Props::PRIME), Some(false));
        assert_eq!(cache.query(&arena, four, Props::COMPOSITE), Some(true));
    }

    // ── Constant assumptions ────────────────────────────────────────

    #[test]
    fn pi_assumptions() {
        let arena = Arena::new();
        let mut cache = AssumptionCache::new();
        assert_eq!(cache.query(&arena, arena.pi, Props::POSITIVE), Some(true));
        assert_eq!(cache.query(&arena, arena.pi, Props::REAL), Some(true));
        assert_eq!(
            cache.query(&arena, arena.pi, Props::TRANSCENDENTAL),
            Some(true)
        );
        assert_eq!(cache.query(&arena, arena.pi, Props::IRRATIONAL), Some(true));
        assert_eq!(cache.query(&arena, arena.pi, Props::RATIONAL), Some(false));
    }

    #[test]
    fn e_assumptions() {
        let arena = Arena::new();
        let mut cache = AssumptionCache::new();
        assert_eq!(
            cache.query(&arena, arena.e_const, Props::POSITIVE),
            Some(true)
        );
        assert_eq!(
            cache.query(&arena, arena.e_const, Props::TRANSCENDENTAL),
            Some(true)
        );
    }

    #[test]
    fn imaginary_unit_assumptions() {
        let arena = Arena::new();
        let mut cache = AssumptionCache::new();
        assert_eq!(
            cache.query(&arena, arena.i_unit, Props::IMAGINARY),
            Some(true)
        );
        assert_eq!(cache.query(&arena, arena.i_unit, Props::REAL), Some(false));
        assert_eq!(
            cache.query(&arena, arena.i_unit, Props::COMPLEX),
            Some(true)
        );
        assert_eq!(
            cache.query(&arena, arena.i_unit, Props::ALGEBRAIC),
            Some(true)
        );
    }

    #[test]
    fn infinity_assumptions() {
        let arena = Arena::new();
        let mut cache = AssumptionCache::new();
        assert_eq!(
            cache.query(&arena, arena.infinity, Props::INFINITE),
            Some(true)
        );
        assert_eq!(
            cache.query(&arena, arena.infinity, Props::POSITIVE),
            Some(true)
        );
        assert_eq!(
            cache.query(&arena, arena.infinity, Props::FINITE),
            Some(false)
        );
    }

    #[test]
    fn neg_infinity_assumptions() {
        let arena = Arena::new();
        let mut cache = AssumptionCache::new();
        assert_eq!(
            cache.query(&arena, arena.neg_infinity, Props::INFINITE),
            Some(true)
        );
        assert_eq!(
            cache.query(&arena, arena.neg_infinity, Props::NEGATIVE),
            Some(true)
        );
    }

    #[test]
    fn nan_assumptions() {
        let arena = Arena::new();
        let mut cache = AssumptionCache::new();
        // NaN: almost everything is unknown.
        assert_eq!(cache.query(&arena, arena.nan, Props::REAL), None);
        assert_eq!(
            cache.query(&arena, arena.nan, Props::COMMUTATIVE),
            Some(true)
        );
    }

    // ── Add propagation ─────────────────────────────────────────────

    #[test]
    fn add_positive_positive_is_positive() {
        let mut arena = Arena::new();
        let mut cache = AssumptionCache::new();

        // Use known-positive constants (1 + 1 = 2).
        let two = arena.add(&[arena.one, arena.one]);
        assert_eq!(cache.query(&arena, two, Props::POSITIVE), Some(true));
        assert_eq!(cache.query(&arena, two, Props::INTEGER), Some(true));
    }

    // ── Mul propagation ─────────────────────────────────────────────

    #[test]
    fn mul_positive_negative_is_negative() {
        let mut arena = Arena::new();
        let mut cache = AssumptionCache::new();

        let one = arena.one;
        let neg_one = arena.neg_one;
        let prod = arena.mul(&[one, neg_one]);
        // 1 * (-1) = -1
        assert_eq!(cache.query(&arena, prod, Props::NEGATIVE), Some(true));
    }

    // ── Pow propagation ─────────────────────────────────────────────

    #[test]
    fn pow_real_even_is_nonneg() {
        let mut arena = Arena::new();
        let mut cache = AssumptionCache::new();

        let neg_one = arena.neg_one;
        let two = arena.int(2);
        let result = arena.pow(neg_one, two);
        // (-1)^2 = 1 — we know the exponent is even, base is real.
        assert_eq!(cache.query(&arena, result, Props::NONNEGATIVE), Some(true));
    }

    // ── is_small_prime ──────────────────────────────────────────────

    #[test]
    fn small_prime_test() {
        assert!(!is_small_prime(0));
        assert!(!is_small_prime(1));
        assert!(is_small_prime(2));
        assert!(is_small_prime(3));
        assert!(!is_small_prime(4));
        assert!(is_small_prime(5));
        assert!(is_small_prime(7));
        assert!(!is_small_prime(9));
        assert!(is_small_prime(11));
        assert!(is_small_prime(97));
        assert!(!is_small_prime(100));
    }

    // ── Hyperbolic / inverse trig propagation ───────────────────────

    #[test]
    fn sinh_of_real_is_real() {
        let mut arena = Arena::new();
        let mut cache = AssumptionCache::new();
        // Use pi as a known-real value.
        let expr = arena.sinh(arena.pi);
        assert_eq!(cache.query(&arena, expr, Props::REAL), Some(true));
    }

    #[test]
    fn cosh_of_real_is_positive() {
        let mut arena = Arena::new();
        let mut cache = AssumptionCache::new();
        let expr = arena.cosh(arena.pi);
        assert_eq!(cache.query(&arena, expr, Props::POSITIVE), Some(true));
    }

    #[test]
    fn tanh_of_real_is_real() {
        let mut arena = Arena::new();
        let mut cache = AssumptionCache::new();
        let expr = arena.tanh(arena.pi);
        assert_eq!(cache.query(&arena, expr, Props::REAL), Some(true));
    }

    #[test]
    fn asin_of_real_is_real() {
        let mut arena = Arena::new();
        let mut cache = AssumptionCache::new();
        let expr = arena.asin(arena.one);
        assert_eq!(cache.query(&arena, expr, Props::REAL), Some(true));
    }

    #[test]
    fn product_with_i_is_imaginary() {
        let mut arena = Arena::new();
        let mut cache = AssumptionCache::new();
        let three = arena.int(3);
        let prod = arena.mul(&[three, arena.i_unit]); // 3*i
        assert_eq!(cache.query(&arena, prod, Props::IMAGINARY), Some(true));
    }

    #[test]
    fn product_i_times_i_is_real() {
        let mut arena = Arena::new();
        let mut cache = AssumptionCache::new();
        // i*i canonicalizes to i^2 = -1 via canon_mul/canon_pow,
        // so check the assumption on that result.
        let prod = arena.mul(&[arena.i_unit, arena.i_unit]);
        assert_eq!(cache.query(&arena, prod, Props::REAL), Some(true));
    }
}
