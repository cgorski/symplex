//! Three-valued decisions, sign sets and the strength of evidence:
//! [`Truth`] for "yes / no / not decided", [`SignSet`] for "which signs can
//! this finite value have", and [`Evidence`] / [`Verdict<T>`] for "how was
//! that established".
//!
//! They replace encodings that could not say what they meant.  Three-valued
//! decisions travelled as `Option<bool>` and were read with `== Some(true)`
//! or `!= Some(true)` — over 500 such comparisons at 0.48 — so wherever the
//! wrong one was written, "not proved true" silently meant "false".  Signs
//! came in five encodings (`Option<Ordering>`, `Option<i8>`, `Option<i32>`,
//! `Option<bool>`, `i8`) and three private sign-class enums, and Kleene
//! AND/OR was written out by hand six times.  Here the undecided case is a
//! variant every `match` must name, signs are one lattice whose arithmetic
//! is tested exhaustively, and the strength of a numeric decision is a value
//! rather than a comment.
//!
//! # Which type
//!
//! | Need | Type |
//! |---|---|
//! | a yes / no / undecided answer inside the crate | [`Truth`] |
//! | the same answer from a public query | `Option<bool>`, for now (CONTRIBUTING.md, "The API Model", Pattern 4); convert at the boundary with `From` or [`Truth::to_option`] |
//! | the signs a finite real or complex value may have; a comparison as `sign(a − b)` | [`SignSet`] |
//! | the sign of something that may be `±∞`, `zoo` or NaN | decide finiteness first (see "Infinities"), then [`SignSet`] |
//! | a value or bound that may be `±∞` | [`Extended<T>`](crate::base::extended::Extended) |
//! | symbolic facts about an expression (integer, prime, finite, …) | [`Assumptions`](crate::base::assumptions::Assumptions) |
//! | how an answer was established | [`Evidence`], carried with the answer as [`Verdict<T>`] |
//! | a `bool` for one branch | [`Truth::is_true`], [`Truth::is_false`] or their negations — the method names the reading |
//!
//! `Truth` is for internal decisions.  It is reachable as
//! `symplex::base::truth::Truth`, like the other foundation types, but no
//! public query returns it yet; that may change at a breaking release.
//!
//! # Reading a `Truth`
//!
//! `Truth` has no `PartialEq<bool>`, no `From<Truth> for bool`, no `Deref`
//! to `bool`, no `unwrap_or` and no ordering.  Each would let a call site
//! turn `Unknown` into an answer without saying which answer it chose — the
//! `x == Some(true)` / `x != Some(false)` habit this type exists to end.  A
//! branch states its reading:
//!
//! | Meaning | Write |
//! |---|---|
//! | proved true | `t.is_true()` |
//! | proved false | `t.is_false()` |
//! | decided either way | `t.is_known()` |
//! | not refuted (true or unknown) | `!t.is_false()` |
//! | not proved (false or unknown) — a policy; say which in a comment | `!t.is_true()` |
//! | the three cases differ | `match t { Truth::True => …, Truth::False => …, Truth::Unknown => … }` |
//!
//! `!t.is_true()` (not proved true) and `t.not().is_true()` (proved false)
//! are different questions.  So are `a == b` (the same *value*: `Unknown ==
//! Unknown` is `true`) and [`a.iff(b)`](Truth::iff) (the biconditional:
//! `Unknown` when either side is).
//!
//! # Corner cases
//!
//! - **Empty iterators.**  [`Truth::all`] of nothing is `True`,
//!   [`Truth::any`] of nothing is `False`, as for `Iterator::all`/`any`.
//! - **`or_else` is not `or`.**  [`Truth::or_else`] asks a second decider the
//!   *same* question when the first said `Unknown` (the cost ladder:
//!   structural test, then exact algebra, then certified numerics).
//!   [`Truth::or`] is the disjunction of two *different* statements.
//! - **The empty `SignSet`** says that no finite value is possible: two
//!   deciders contradict each other, or the value is `0⁻¹`.  Arithmetic
//!   propagates it (`EMPTY` op anything is `EMPTY`), and every three-valued
//!   projection ([`SignSet::is_positive`], …) answers `Unknown` for it —
//!   never `True` or `False`, because a contradiction is a bug or a dead
//!   branch and must not become a claim.  Test [`SignSet::is_empty`] where
//!   that matters.
//! - **Division.**  [`SignSet::recip`] drops the `ZERO` atom, which has no
//!   reciprocal: `{0, >0}.recip()` is `{>0}` and `ZERO.recip()` is `EMPTY`.
//!   Whether a possibly-zero divisor is an error, `zoo`, or excluded by the
//!   branch being taken is the caller's decision; ask
//!   [`SignSet::is_nonzero`] first.  `pow_int` with a negative exponent does
//!   the same (`ZERO.pow_int(-2)` is `EMPTY`).
//! - **`0⁰ = 1`.**  `pow_int(0)` is `POSITIVE` for every non-empty set,
//!   `ZERO` included, as `canon_pow` rewrites `x^0 → 1` for finite `x`.
//! - **A power is not a repeated product.**  `NONREAL.square()` is
//!   `{<0, nonreal}` (`i² = −1`; `z² > 0` needs a real `z`), but
//!   `NONREAL.mul(NONREAL)` is `{<0, >0, nonreal}` (`i·(−i) = 1`): `mul`
//!   treats its operands as independent values, `pow_int` knows they are one
//!   value.  Use `pow_int` for `xⁿ`.
//! - **Two non-real summands** can land anywhere: `NONREAL.add(NONREAL)` is
//!   `UNKNOWN` (`i + (−i) = 0`, `(1+i) + (1−i) = 2`).
//! - **`abs`, `conj`.**  `|z| > 0` for every `z ≠ 0`, so `abs` maps
//!   `NONREAL` to `POSITIVE`; `conj` keeps every atom.
//!
//! # Infinities
//!
//! `SignSet` describes *finite* values only; `±∞`, `zoo` and NaN have no
//! atom.  Do not encode `+∞` as `POSITIVE`: the tables would then be wrong
//! (`∞ + (−∞)`, `0·∞`, `∞⁰`).  A caller that may meet an infinity decides
//! finiteness first and keeps that answer beside the set — a `Truth` for
//! "finite", or an enum with an explicit infinite/undefined case, as
//! `ValueSign::Undefined` in `transforms/inequalities.rs`.
//!
//! # Evidence policy
//!
//! Only [`Evidence::Exact`] or [`Evidence::Certified`] may justify a claim:
//! a returned value, a rewrite, a mathematical outcome such as
//! `Inconsistent`, a branch that drops cases.  [`Evidence::Approximate`] and
//! [`Evidence::Probabilistic`] may only guide heuristics whose result is
//! correct either way — pivot choice, which method to try first, an early
//! rejection that an exact test then confirms.  Read a [`Verdict`] with
//! [`proven`](Verdict::proven) or `truth_at_least(Evidence::Certified)`;
//! `at_least(Evidence::Approximate)` marks a deliberate heuristic, visibly and
//! greppably.
//!
//! # Migrating
//!
//! | Before (`x: Option<bool>`) | After (`t: Truth`) |
//! |---|---|
//! | `x == Some(true)`, `x.unwrap_or(false)` | `t.is_true()` |
//! | `x == Some(false)` | `t.is_false()` |
//! | `x != Some(true)` | `!t.is_true()` |
//! | `x != Some(false)`, `x.unwrap_or(true)` | `!t.is_false()` |
//! | `x.is_some()` / `x.is_none()` | `t.is_known()` / `!t.is_known()` |
//!
//! There is no `From<Option<Ordering>>` for `SignSet`: its `None` meant "real,
//! sign unknown" ([`SignSet::REAL`]) at some sites and "maybe not real"
//! ([`SignSet::UNKNOWN`]) or "unordered" at others.  Spell it out per site,
//! e.g. `o.map_or(SignSet::REAL, SignSet::from_ordering)`.
//!
//! # Before and after
//!
//! At 0.48 `linsolve` asked `is_zero_checked(..) == Some(true)` of a
//! right-hand side `k = Γ(1/3)Γ(2/3) − 2π/√3` (0 by the reflection formula,
//! but not recognised), read "not proved zero" as "nonzero", and reported
//! the system inconsistent:
//!
//! ```
//! use symplex::base::truth::{SignSet, Truth};
//! # fn is_zero_checked() -> Option<bool> { None }
//! // Before: `None` is silently folded into "nonzero" …
//! let rhs_is_zero = is_zero_checked() == Some(true);
//! assert!(!rhs_is_zero); // … so the system is "inconsistent": wrong.
//!
//! // After: the third case has to be written down.
//! let outcome = match Truth::from(is_zero_checked()) {
//!     Truth::True => "consistent",
//!     Truth::False => "inconsistent",
//!     Truth::Unknown => "undecided: refuse",
//! };
//! assert_eq!(outcome, "undecided: refuse");
//!
//! // Signs: x < 0 and y < 0 give x·y + 1 > 0; the question stays three-valued.
//! let s = SignSet::NEGATIVE.mul(SignSet::NEGATIVE).add(SignSet::POSITIVE);
//! assert_eq!(s, SignSet::POSITIVE);
//! assert!(s.is_positive().is_true());
//! assert_eq!(SignSet::REAL.is_positive(), Truth::Unknown);
//! ```

use std::cmp::Ordering;
use std::fmt;
use std::ops::{BitAnd, BitOr, Not};

// ── Truth ──────────────────────────────────────────────────────────────

/// A three-valued (Kleene) truth value: proved true, proved false, or not
/// decided.
///
/// The connectives are Kleene's strong ones — [`and`](Truth::and) is `False`
/// as soon as either side is, [`or`](Truth::or) is `True` as soon as either
/// side is, otherwise an `Unknown` operand makes the result `Unknown`.  They
/// are the boolean operators lifted to the sets of values each side may
/// have (`Unknown` standing for `{false, true}`), which the tests check.
///
/// Deliberately absent: `PartialEq<bool>`, `From<Truth> for bool`,
/// `Deref<Target = bool>`, `unwrap_or` and `PartialOrd` (the truth order
/// `False < Unknown < True` exists, but `t >= Truth::Unknown` hides the
/// reading as surely as `!= Some(false)`).  See the module documentation,
/// "Reading a `Truth`".
///
/// ```
/// use symplex::base::truth::Truth;
/// let t = Truth::True.and(Truth::Unknown);
/// assert_eq!(t, Truth::Unknown);
/// assert!(!t.is_true() && !t.is_false());
/// assert_eq!(Truth::False & Truth::Unknown, Truth::False);
/// assert_eq!(Truth::from(Some(true)), Truth::True);
/// assert_eq!(Truth::Unknown.to_option(), None);
/// ```
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Truth {
    /// Proved false.
    False,
    /// Not decided: no claim either way.
    Unknown,
    /// Proved true.
    True,
}

impl Truth {
    /// `True` for `true`, `False` for `false`.
    pub const fn from_bool(b: bool) -> Truth {
        if b { Truth::True } else { Truth::False }
    }

    /// Proved true.  `!t.is_true()` means "not proved": `False` *or*
    /// `Unknown`.
    #[must_use]
    pub const fn is_true(self) -> bool {
        matches!(self, Truth::True)
    }

    /// Proved false.  `!t.is_false()` means "not refuted": `True` *or*
    /// `Unknown`.
    #[must_use]
    pub const fn is_false(self) -> bool {
        matches!(self, Truth::False)
    }

    /// Decided either way: `True` or `False`.
    #[must_use]
    pub const fn is_known(self) -> bool {
        !matches!(self, Truth::Unknown)
    }

    /// Kleene negation: `True` and `False` swap, `Unknown` stays.  Also `!t`.
    pub const fn not(self) -> Truth {
        match self {
            Truth::False => Truth::True,
            Truth::Unknown => Truth::Unknown,
            Truth::True => Truth::False,
        }
    }

    /// Kleene conjunction: `False` if either side is `False`, `True` if both
    /// are `True`, otherwise `Unknown`.  Also `a & b`; both operands are
    /// evaluated — for a lazy chain use [`Truth::all`] over a lazy iterator.
    pub const fn and(self, other: Truth) -> Truth {
        match (self, other) {
            (Truth::False, _) | (_, Truth::False) => Truth::False,
            (Truth::True, Truth::True) => Truth::True,
            _ => Truth::Unknown,
        }
    }

    /// Kleene disjunction: `True` if either side is `True`, `False` if both
    /// are `False`, otherwise `Unknown`.  Also `a | b`; both operands are
    /// evaluated — for a lazy chain use [`Truth::any`] over a lazy iterator.
    pub const fn or(self, other: Truth) -> Truth {
        match (self, other) {
            (Truth::True, _) | (_, Truth::True) => Truth::True,
            (Truth::False, Truth::False) => Truth::False,
            _ => Truth::Unknown,
        }
    }

    /// Kleene implication `¬self ∨ other`: `True` when `self` is `False` or
    /// `other` is `True`, `False` when `self` is `True` and `other` is
    /// `False`, otherwise `Unknown`.
    pub const fn implies(self, other: Truth) -> Truth {
        self.not().or(other)
    }

    /// Kleene biconditional: whether both statements have the same truth
    /// value, `Unknown` if either is undecided.  Not `==`, which compares
    /// the *values*: `Unknown == Unknown` is `true`, while
    /// `Unknown.iff(Unknown)` is `Unknown`.
    pub const fn iff(self, other: Truth) -> Truth {
        match (self, other) {
            (Truth::Unknown, _) | (_, Truth::Unknown) => Truth::Unknown,
            (Truth::True, Truth::True) | (Truth::False, Truth::False) => Truth::True,
            _ => Truth::False,
        }
    }

    /// Kleene conjunction of all items: `False` at the first `False` (later
    /// items are not pulled), otherwise `Unknown` if any item was, otherwise
    /// `True` — so `True` for no items.
    pub fn all(items: impl IntoIterator<Item = Truth>) -> Truth {
        let mut acc = Truth::True;
        for t in items {
            match t {
                Truth::False => return Truth::False,
                Truth::Unknown => acc = Truth::Unknown,
                Truth::True => {}
            }
        }
        acc
    }

    /// Kleene disjunction of all items: `True` at the first `True` (later
    /// items are not pulled), otherwise `Unknown` if any item was, otherwise
    /// `False` — so `False` for no items.
    pub fn any(items: impl IntoIterator<Item = Truth>) -> Truth {
        let mut acc = Truth::False;
        for t in items {
            match t {
                Truth::True => return Truth::True,
                Truth::Unknown => acc = Truth::Unknown,
                Truth::False => {}
            }
        }
        acc
    }

    /// The cost ladder: `self` if decided, otherwise `f()` — the answer of a
    /// second, usually more expensive, decider of the **same** question.
    /// `f` runs only when `self` is `Unknown`.
    ///
    /// This is not the disjunction [`or`](Truth::or): `False.or_else(f)` is
    /// `False` without calling `f`.
    ///
    /// ```
    /// use symplex::base::truth::Truth;
    /// let structural = Truth::Unknown;
    /// let exact_algebra = || Truth::False;
    /// assert_eq!(structural.or_else(exact_algebra), Truth::False);
    /// assert_eq!(Truth::True.or_else(|| unreachable!("not asked")), Truth::True);
    /// ```
    pub fn or_else(self, f: impl FnOnce() -> Truth) -> Truth {
        match self {
            Truth::Unknown => f(),
            known => known,
        }
    }

    /// `Some(true)`, `Some(false)` or `None`: the public-API encoding
    /// (CONTRIBUTING.md, "The API Model", Pattern 4).
    #[must_use]
    pub const fn to_option(self) -> Option<bool> {
        match self {
            Truth::False => Some(false),
            Truth::Unknown => None,
            Truth::True => Some(true),
        }
    }
}

impl Not for Truth {
    type Output = Truth;

    fn not(self) -> Truth {
        Truth::not(self)
    }
}

impl BitAnd for Truth {
    type Output = Truth;

    fn bitand(self, rhs: Truth) -> Truth {
        self.and(rhs)
    }
}

impl BitOr for Truth {
    type Output = Truth;

    fn bitor(self, rhs: Truth) -> Truth {
        self.or(rhs)
    }
}

impl From<bool> for Truth {
    fn from(b: bool) -> Truth {
        Truth::from_bool(b)
    }
}

impl From<Option<bool>> for Truth {
    fn from(o: Option<bool>) -> Truth {
        match o {
            Some(b) => Truth::from_bool(b),
            None => Truth::Unknown,
        }
    }
}

impl From<Truth> for Option<bool> {
    fn from(t: Truth) -> Option<bool> {
        t.to_option()
    }
}

impl fmt::Display for Truth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Truth::False => "false",
            Truth::Unknown => "unknown",
            Truth::True => "true",
        })
    }
}

// ── SignSet ────────────────────────────────────────────────────────────

/// The set of signs a **finite** real or complex value may have: a subset
/// of the atoms [`NEGATIVE`](Self::NEGATIVE) (real, `< 0`),
/// [`ZERO`](Self::ZERO), [`POSITIVE`](Self::POSITIVE) (real, `> 0`) and
/// [`NONREAL`](Self::NONREAL) (imaginary part `≠ 0`).
///
/// Every finite complex number has exactly one atom, so a set says "the
/// value has one of these signs": [`UNKNOWN`](Self::UNKNOWN) (all four)
/// says nothing, a single atom decides every sign question, and
/// [`EMPTY`](Self::EMPTY) is a contradiction.  Fewer atoms is more
/// knowledge: combine two deciders' answers about one value with
/// [`intersect`](Self::intersect), merge the branches of a case split with
/// [`union`](Self::union).
///
/// The arithmetic ([`neg`](Self::neg), [`add`](Self::add),
/// [`mul`](Self::mul), [`recip`](Self::recip), [`pow_int`](Self::pow_int),
/// [`abs`](Self::abs)) returns exactly the signs `x op y` can have for `x`
/// with a sign in `self` and `y` with a sign in the other set: sound (no
/// possible sign is missing) and tight (no impossible one is listed) for
/// every pair of atoms, which the tests check by brute force, and the union
/// over atom pairs for larger sets.  That is exact for independent `x` and
/// `y`; for one value used twice it can be wider (`x·x` with `x` in
/// `{<0, >0}` is positive, `mul` says `{<0, >0}`), so powers have their own
/// [`pow_int`](Self::pow_int).
///
/// There are no operator impls (`+`, `*`, `|`) on purpose: on a set `a + b`
/// could be read as the union or as the sign of a sum.  `SignSet` is
/// `Hash + Eq`, so it is a memo key as it stands; there is no raw-bits
/// accessor (a memo keyed by hand-picked `u8` values is one of the bugs it
/// replaces).  The constants work as `match` patterns.
///
/// Infinities are out of scope; see the module documentation.
///
/// ```
/// use symplex::base::truth::{SignSet, Truth};
/// let x = SignSet::NONNEGATIVE; // x ≥ 0
/// let y = x.add(SignSet::POSITIVE); // x + 1
/// assert_eq!(y, SignSet::POSITIVE);
/// assert_eq!(y.recip().is_positive(), Truth::True);
/// assert_eq!(SignSet::NONREAL.square(), SignSet::NEGATIVE.union(SignSet::NONREAL));
/// assert_eq!(SignSet::REAL.is_zero(), Truth::Unknown);
/// assert_eq!(SignSet::NONPOSITIVE.to_string(), "{<0, 0}");
/// ```
#[must_use]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SignSet(u8);

/// The atoms, in the order of [`ATOM_NAMES`] and of `Display`.
const ATOMS: [SignSet; 4] = [
    SignSet::NEGATIVE,
    SignSet::ZERO,
    SignSet::POSITIVE,
    SignSet::NONREAL,
];

/// How `Display` writes each atom of [`ATOMS`].
const ATOM_NAMES: [&str; 4] = ["<0", "0", ">0", "nonreal"];

impl SignSet {
    /// No sign at all: no finite value is possible (a contradiction, or a
    /// value such as `0⁻¹`).
    pub const EMPTY: SignSet = SignSet(0b0000);
    /// A real number `< 0`.
    pub const NEGATIVE: SignSet = SignSet(0b0001);
    /// Exactly `0`.
    pub const ZERO: SignSet = SignSet(0b0010);
    /// A real number `> 0`.
    pub const POSITIVE: SignSet = SignSet(0b0100);
    /// A complex number with a non-zero imaginary part.
    pub const NONREAL: SignSet = SignSet(0b1000);
    /// Real: `< 0`, `0` or `> 0`.
    pub const REAL: SignSet = Self::NEGATIVE.union(Self::ZERO).union(Self::POSITIVE);
    /// Real and `≥ 0`.
    pub const NONNEGATIVE: SignSet = Self::ZERO.union(Self::POSITIVE);
    /// Real and `≤ 0`.
    pub const NONPOSITIVE: SignSet = Self::NEGATIVE.union(Self::ZERO);
    /// Not `0`: `< 0`, `> 0` or non-real.
    pub const NONZERO: SignSet = Self::NEGATIVE.union(Self::POSITIVE).union(Self::NONREAL);
    /// Real and not `0`.
    pub const NONZERO_REAL: SignSet = Self::NEGATIVE.union(Self::POSITIVE);
    /// Any finite value: all four atoms, nothing known.
    pub const UNKNOWN: SignSet = Self::REAL.union(Self::NONREAL);

    // ── Set operations ──

    /// Whether every atom of `other` is in `self` (`other ⊆ self`);
    /// `contains(EMPTY)` is `true`.
    #[must_use]
    pub const fn contains(self, other: SignSet) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether every atom of `self` is in `other` (`self ⊆ other`).  A
    /// statement about the sets; [`is_in`](Self::is_in) asks the
    /// three-valued question about the value.
    #[must_use]
    pub const fn is_subset(self, other: SignSet) -> bool {
        other.contains(self)
    }

    /// `self ∪ other`: the signs of a value from either set — the merge of
    /// a case split.  (The lattice join of the subset order; named after the
    /// set, because "join" and "meet" swap meaning between the subset order
    /// and the order by information.)
    pub const fn union(self, other: SignSet) -> SignSet {
        SignSet(self.0 | other.0)
    }

    /// `self ∩ other`: what two sound deciders of the same value allow
    /// together; `EMPTY` if they contradict each other.  (The lattice meet
    /// of the subset order.)
    pub const fn intersect(self, other: SignSet) -> SignSet {
        SignSet(self.0 & other.0)
    }

    /// No atom: a contradiction (see the module documentation).
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Exactly one atom: the sign is decided.
    #[must_use]
    pub const fn is_single(self) -> bool {
        self.0.count_ones() == 1
    }

    /// The atoms of `self`, in the order `NEGATIVE`, `ZERO`, `POSITIVE`,
    /// `NONREAL`.
    pub fn atoms(self) -> impl Iterator<Item = SignSet> {
        ATOMS.into_iter().filter(move |&a| self.contains(a))
    }

    // ── Arithmetic ──

    /// The signs of `−x`: `NEGATIVE` and `POSITIVE` swap, `0` and non-real
    /// values stay.
    pub const fn neg(self) -> SignSet {
        lift1(Unary::Neg, self)
    }

    /// The signs of `x + y`.  Opposite real signs give `REAL`; a real plus a
    /// non-real is non-real; two non-reals can sum to anything.
    pub const fn add(self, other: SignSet) -> SignSet {
        lift2(Binary::Add, self, other)
    }

    /// The signs of `x·y` for independent `x` and `y`.  `0` absorbs; two
    /// non-reals multiply to anything but `0` (`i·i = −1`, `i·(−i) = 1`,
    /// `(1+i)² = 2i`).
    pub const fn mul(self, other: SignSet) -> SignSet {
        lift2(Binary::Mul, self, other)
    }

    /// The signs of `1/x` over the non-zero `x`: every non-zero atom maps
    /// to itself (`1/z = z̄/|z|²`), and `ZERO`, which has no reciprocal, is
    /// dropped — `{0, >0}.recip()` is `{>0}`, `ZERO.recip()` is `EMPTY`.  The
    /// caller decides what dividing by a possibly-zero value means; ask
    /// [`is_nonzero`](Self::is_nonzero) first.
    pub const fn recip(self) -> SignSet {
        lift1(Unary::Recip, self)
    }

    /// The signs of `xⁿ` for one value `x` (unlike `n`-fold
    /// [`mul`](Self::mul)).
    ///
    /// - `n = 0`: `POSITIVE` for every non-empty set, `ZERO` included
    ///   (`0⁰ = 1`, as `canon_pow`).
    /// - `NEGATIVE`: negative for odd `n`, positive for even `n`.
    /// - `ZERO`: `0` for `n > 0`; for `n < 0`, `0ⁿ` is not finite and the
    ///   atom is dropped, as in [`recip`](Self::recip).
    /// - `NONREAL`: `z = r·e^{iθ}` with `θ ∉ πℤ` has `zⁿ = rⁿ·e^{inθ}`, never
    ///   0 and non-real for small `θ`; negative only when `|n| ≥ 2`
    ///   (`θ = π/|n|`: `i² = −1`) and positive only when `|n| ≥ 3`
    ///   (`θ = 2π/|n|`; for `|n| = 2` that would need `θ = π`).
    pub const fn pow_int(self, n: i64) -> SignSet {
        lift1(Unary::Pow(n), self)
    }

    /// The signs of `x²`: [`pow_int(2)`](Self::pow_int).
    pub const fn square(self) -> SignSet {
        self.pow_int(2)
    }

    /// The signs of `|x|`: `0` for `0`, positive otherwise, real or not.
    pub const fn abs(self) -> SignSet {
        lift1(Unary::Abs, self)
    }

    /// The signs of the complex conjugate `x̄`: the same set (a real value
    /// is its own conjugate, a non-real one stays non-real).
    pub const fn conj(self) -> SignSet {
        self
    }

    // ── Three-valued projections ──

    /// Is the value in `property`?  `True` when `self` is a non-empty subset
    /// of `property`, `False` when the two are disjoint, `Unknown` otherwise
    /// — and `Unknown` for the empty `self`, since a contradiction must not
    /// become a claim.  The named projections below are this with a fixed
    /// `property`.
    pub const fn is_in(self, property: SignSet) -> Truth {
        if self.is_empty() {
            return Truth::Unknown;
        }
        if self.is_subset(property) {
            Truth::True
        } else if self.intersect(property).is_empty() {
            Truth::False
        } else {
            Truth::Unknown
        }
    }

    /// Is the value real and `> 0`?
    pub const fn is_positive(self) -> Truth {
        self.is_in(Self::POSITIVE)
    }

    /// Is the value real and `< 0`?
    pub const fn is_negative(self) -> Truth {
        self.is_in(Self::NEGATIVE)
    }

    /// Is the value `0`?
    pub const fn is_zero(self) -> Truth {
        self.is_in(Self::ZERO)
    }

    /// Is the value `≠ 0` (real or not)?
    pub const fn is_nonzero(self) -> Truth {
        self.is_in(Self::NONZERO)
    }

    /// Is the value real and `≥ 0`?  A non-real value is not nonnegative
    /// (as with `Props::NONNEGATIVE`, which implies `EXTENDED_REAL`).
    pub const fn is_nonnegative(self) -> Truth {
        self.is_in(Self::NONNEGATIVE)
    }

    /// Is the value real and `≤ 0`?  A non-real value is not nonpositive.
    pub const fn is_nonpositive(self) -> Truth {
        self.is_in(Self::NONPOSITIVE)
    }

    /// Is the imaginary part `0`?
    pub const fn is_real(self) -> Truth {
        self.is_in(Self::REAL)
    }

    /// Is the imaginary part `≠ 0`?
    pub const fn is_nonreal(self) -> Truth {
        self.is_in(Self::NONREAL)
    }

    // ── Conversions ──

    /// The sign of a real number given as its `Ordering` against 0
    /// (`x.cmp(&0)`).
    pub const fn from_ordering(o: Ordering) -> SignSet {
        match o {
            Ordering::Less => Self::NEGATIVE,
            Ordering::Equal => Self::ZERO,
            Ordering::Greater => Self::POSITIVE,
        }
    }

    /// The sign of a real number given as a signum: any negative `s` is
    /// `NEGATIVE`, `0` is `ZERO`, any positive `s` is `POSITIVE`.
    pub const fn from_sign_i8(s: i8) -> SignSet {
        match s {
            i8::MIN..=-1 => Self::NEGATIVE,
            0 => Self::ZERO,
            1..=i8::MAX => Self::POSITIVE,
        }
    }

    /// The `Ordering` against 0 of a single real atom; `None` for
    /// `NONREAL` (unordered), for several atoms and for `EMPTY`.
    #[must_use]
    pub const fn to_ordering(self) -> Option<Ordering> {
        match self {
            Self::NEGATIVE => Some(Ordering::Less),
            Self::ZERO => Some(Ordering::Equal),
            Self::POSITIVE => Some(Ordering::Greater),
            _ => None,
        }
    }
}

impl fmt::Display for SignSet {
    /// The atoms in braces: `{}`, `{<0, 0}`, `{<0, 0, >0, nonreal}`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        let mut first = true;
        for (atom, name) in ATOMS.into_iter().zip(ATOM_NAMES) {
            if self.contains(atom) {
                if !first {
                    f.write_str(", ")?;
                }
                f.write_str(name)?;
                first = false;
            }
        }
        f.write_str("}")
    }
}

impl fmt::Debug for SignSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// A one-argument operation on signs, applied atom by atom by [`lift1`].
#[derive(Clone, Copy)]
enum Unary {
    Neg,
    Recip,
    Abs,
    Pow(i64),
}

/// A two-argument operation on signs, applied atom by atom by [`lift2`].
#[derive(Clone, Copy)]
enum Binary {
    Add,
    Mul,
}

/// The union of `op(a)` over the atoms `a` of `s`.
const fn lift1(op: Unary, s: SignSet) -> SignSet {
    let mut out = SignSet::EMPTY;
    let mut i = 0;
    while i < ATOMS.len() {
        if s.contains(ATOMS[i]) {
            out = out.union(unary_atom(op, ATOMS[i]));
        }
        i += 1;
    }
    out
}

/// The union of `op(a, b)` over the atoms `a` of `x` and `b` of `y`.
const fn lift2(op: Binary, x: SignSet, y: SignSet) -> SignSet {
    let mut out = SignSet::EMPTY;
    let mut i = 0;
    while i < ATOMS.len() {
        let mut j = 0;
        while j < ATOMS.len() {
            if x.contains(ATOMS[i]) && y.contains(ATOMS[j]) {
                out = out.union(binary_atom(op, ATOMS[i], ATOMS[j]));
            }
            j += 1;
        }
        i += 1;
    }
    out
}

// The per-atom rules.  Each `match` ends in `_ => SignSet::UNKNOWN` for
// arguments that are not single atoms: the lifting loops never pass one, and
// the answer that claims nothing is the safe one if that ever changes.

/// The signs of `op(x)` for every `x` of the single atom `a`.
const fn unary_atom(op: Unary, a: SignSet) -> SignSet {
    match op {
        Unary::Neg => match a {
            SignSet::NEGATIVE => SignSet::POSITIVE,
            SignSet::POSITIVE => SignSet::NEGATIVE,
            SignSet::ZERO | SignSet::NONREAL => a,
            _ => SignSet::UNKNOWN,
        },
        Unary::Recip => match a {
            SignSet::ZERO => SignSet::EMPTY,
            SignSet::NEGATIVE | SignSet::POSITIVE | SignSet::NONREAL => a,
            _ => SignSet::UNKNOWN,
        },
        Unary::Abs => match a {
            SignSet::ZERO => SignSet::ZERO,
            SignSet::NEGATIVE | SignSet::POSITIVE | SignSet::NONREAL => SignSet::POSITIVE,
            _ => SignSet::UNKNOWN,
        },
        Unary::Pow(n) => pow_atom(a, n),
    }
}

/// The signs of `xⁿ` for every `x` of the single atom `a`; the cases are
/// derived on [`SignSet::pow_int`].
const fn pow_atom(a: SignSet, n: i64) -> SignSet {
    if n == 0 {
        return SignSet::POSITIVE;
    }
    let odd = n % 2 != 0;
    match a {
        SignSet::NEGATIVE if odd => SignSet::NEGATIVE,
        SignSet::NEGATIVE => SignSet::POSITIVE,
        SignSet::ZERO if n > 0 => SignSet::ZERO,
        SignSet::ZERO => SignSet::EMPTY,
        SignSet::POSITIVE => SignSet::POSITIVE,
        SignSet::NONREAL => match n.unsigned_abs() {
            1 => SignSet::NONREAL,
            2 => SignSet::NEGATIVE.union(SignSet::NONREAL),
            _ => SignSet::NONZERO,
        },
        _ => SignSet::UNKNOWN,
    }
}

/// The signs of `op(x, y)` for every `x` of atom `a` and `y` of atom `b`.
const fn binary_atom(op: Binary, a: SignSet, b: SignSet) -> SignSet {
    match op {
        Binary::Add => add_atoms(a, b),
        Binary::Mul => mul_atoms(a, b),
    }
}

/// The signs of `x + y` for every `x` of atom `a` and `y` of atom `b`.
const fn add_atoms(a: SignSet, b: SignSet) -> SignSet {
    match (a, b) {
        // i + (−i) = 0, (1+i) + (1−i) = 2, (−1+i) + (−1−i) = −2, i + i = 2i.
        (SignSet::NONREAL, SignSet::NONREAL) => SignSet::UNKNOWN,
        // A real summand leaves the imaginary part alone.
        (SignSet::NONREAL, SignSet::NEGATIVE | SignSet::ZERO | SignSet::POSITIVE)
        | (SignSet::NEGATIVE | SignSet::ZERO | SignSet::POSITIVE, SignSet::NONREAL) => {
            SignSet::NONREAL
        }
        (SignSet::ZERO, SignSet::NEGATIVE | SignSet::ZERO | SignSet::POSITIVE) => b,
        (SignSet::NEGATIVE | SignSet::POSITIVE, SignSet::ZERO) => a,
        (SignSet::NEGATIVE, SignSet::NEGATIVE) => SignSet::NEGATIVE,
        (SignSet::POSITIVE, SignSet::POSITIVE) => SignSet::POSITIVE,
        // −3 + 1 < 0, −1 + 1 = 0, −1 + 3 > 0.
        (SignSet::NEGATIVE, SignSet::POSITIVE) | (SignSet::POSITIVE, SignSet::NEGATIVE) => {
            SignSet::REAL
        }
        _ => SignSet::UNKNOWN,
    }
}

/// The signs of `x·y` for every `x` of atom `a` and `y` of atom `b`.
const fn mul_atoms(a: SignSet, b: SignSet) -> SignSet {
    match (a, b) {
        // 0 absorbs every finite factor.
        (
            SignSet::ZERO,
            SignSet::NEGATIVE | SignSet::ZERO | SignSet::POSITIVE | SignSet::NONREAL,
        )
        | (SignSet::NEGATIVE | SignSet::POSITIVE | SignSet::NONREAL, SignSet::ZERO) => {
            SignSet::ZERO
        }
        // Arguments add: i·i = −1, i·(−i) = 1, (1+i)·(1+i) = 2i; never 0.
        (SignSet::NONREAL, SignSet::NONREAL) => SignSet::NONZERO,
        // A non-zero real factor scales z, or turns it by π: still off the axis.
        (SignSet::NONREAL, SignSet::NEGATIVE | SignSet::POSITIVE)
        | (SignSet::NEGATIVE | SignSet::POSITIVE, SignSet::NONREAL) => SignSet::NONREAL,
        (SignSet::NEGATIVE, SignSet::NEGATIVE) | (SignSet::POSITIVE, SignSet::POSITIVE) => {
            SignSet::POSITIVE
        }
        (SignSet::NEGATIVE, SignSet::POSITIVE) | (SignSet::POSITIVE, SignSet::NEGATIVE) => {
            SignSet::NEGATIVE
        }
        _ => SignSet::UNKNOWN,
    }
}

// ── Evidence ───────────────────────────────────────────────────────────

/// How a decision was established.  From strongest to weakest: `Exact`,
/// `Certified`, `Approximate`, `Probabilistic`; only the first two are proofs
/// (module documentation, "Evidence policy").
///
/// There is deliberately no `PartialOrd`/`Ord`: with a derived order the
/// declaration order decides whether `e >= Evidence::Certified` means "at
/// least as strong" or "at most as strong", and reordering the variants
/// would silently invert every comparison.  Compare with
/// [`at_least`](Self::at_least) and [`stronger_than`](Self::stronger_than),
/// whose direction is in the name.
///
/// The order between the two non-proofs is a convention (kept from the
/// design survey): neither may justify a claim, so it only matters to a
/// heuristic that accepts one and not the other.  A sampling test that
/// finds a non-zero value *with exact arithmetic* has proved "non-zero":
/// record that as `Exact`; only the "zero at every sample" side is
/// `Probabilistic`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Evidence {
    /// A structural or algebraic proof: exact arithmetic, a canonical form,
    /// a minimal polynomial, a theorem.
    Exact,
    /// Rigorous numerics with an error bound: an enclosure that excludes 0,
    /// a certified ball.
    Certified,
    /// Unrigorous floating point: an `f64` evaluation, a value that is zero
    /// only to the working precision.
    Approximate,
    /// Random sampling: values at random points (Schwartz–Zippel), right
    /// with high probability.
    Probabilistic,
}

impl Evidence {
    /// Strength rank, larger is stronger.  Private, so the only orderings
    /// are the named methods.
    const fn rank(self) -> u8 {
        match self {
            Evidence::Exact => 3,
            Evidence::Certified => 2,
            Evidence::Approximate => 1,
            Evidence::Probabilistic => 0,
        }
    }

    /// `Exact` or `Certified`: evidence that may justify a claim.
    #[must_use]
    pub const fn is_proof(self) -> bool {
        matches!(self, Evidence::Exact | Evidence::Certified)
    }

    /// Whether `self` is `min` or stronger.
    #[must_use]
    pub const fn at_least(self, min: Evidence) -> bool {
        self.rank() >= min.rank()
    }

    /// Whether `self` is strictly stronger than `other`.
    #[must_use]
    pub const fn stronger_than(self, other: Evidence) -> bool {
        self.rank() > other.rank()
    }

    /// The weaker of the two: the evidence for a conclusion drawn from both.
    #[must_use]
    pub const fn weaker(self, other: Evidence) -> Evidence {
        if other.rank() < self.rank() {
            other
        } else {
            self
        }
    }
}

/// An answer together with the [`Evidence`] behind it.
///
/// Read it at the strength the use requires: [`proven`](Self::proven) for a
/// claim, [`at_least`](Self::at_least) with an explicit level for a
/// heuristic.  [`truth_at_least`](Verdict::truth_at_least) and
/// [`signs_at_least`](Verdict::signs_at_least) turn a too-weak answer into
/// the no-information value (`Truth::Unknown`, `SignSet::UNKNOWN`) instead
/// of an `Option`.
///
/// ```
/// use symplex::base::truth::{Evidence, SignSet, Truth, Verdict};
/// // `exp(1e-150) − 1 − 1e-150` is 0 to working precision: no proof.
/// let zero = Verdict::new(Truth::True, Evidence::Approximate);
/// assert_eq!(zero.proven(), None);
/// assert_eq!(zero.truth_at_least(Evidence::Certified), Truth::Unknown);
/// assert_eq!(zero.truth_at_least(Evidence::Approximate), Truth::True); // a heuristic may
/// let sign = Verdict::new(SignSet::POSITIVE, Evidence::Certified);
/// assert_eq!(sign.signs_at_least(Evidence::Certified), SignSet::POSITIVE);
/// assert_eq!(sign.signs_at_least(Evidence::Exact), SignSet::UNKNOWN);
/// ```
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Verdict<T> {
    /// The answer.
    pub value: T,
    /// How it was established.
    pub evidence: Evidence,
}

impl<T> Verdict<T> {
    /// `value`, established by `evidence`.
    pub const fn new(value: T, evidence: Evidence) -> Self {
        Verdict { value, evidence }
    }

    /// The value if the evidence is a proof (`Exact` or `Certified`).
    #[must_use]
    pub fn proven(self) -> Option<T> {
        self.evidence.is_proof().then_some(self.value)
    }

    /// The value if the evidence is `min` or stronger.
    #[must_use]
    pub fn at_least(self, min: Evidence) -> Option<T> {
        self.evidence.at_least(min).then_some(self.value)
    }

    /// The same evidence for `f(value)`.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Verdict<U> {
        Verdict {
            value: f(self.value),
            evidence: self.evidence,
        }
    }
}

impl Verdict<Truth> {
    /// The value if the evidence is `min` or stronger, `Truth::Unknown`
    /// otherwise.
    pub const fn truth_at_least(self, min: Evidence) -> Truth {
        if self.evidence.at_least(min) {
            self.value
        } else {
            Truth::Unknown
        }
    }
}

impl Verdict<SignSet> {
    /// The value if the evidence is `min` or stronger, `SignSet::UNKNOWN`
    /// otherwise.
    pub const fn signs_at_least(self, min: Evidence) -> SignSet {
        if self.evidence.at_least(min) {
            self.value
        } else {
            SignSet::UNKNOWN
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::collections::HashSet;

    const TRUTHS: [Truth; 3] = [Truth::False, Truth::Unknown, Truth::True];

    const STRONGEST_FIRST: [Evidence; 4] = [
        Evidence::Exact,
        Evidence::Certified,
        Evidence::Approximate,
        Evidence::Probabilistic,
    ];

    // ── Truth ──

    /// `Truth` as the set of booleans it may stand for: the independent
    /// encoding the connectives are checked against.
    fn possible(t: Truth) -> &'static [bool] {
        match t {
            Truth::False => &[false],
            Truth::Unknown => &[false, true],
            Truth::True => &[true],
        }
    }

    /// The `Truth` standing for a non-empty set of booleans.
    fn from_possible(values: &[bool]) -> Truth {
        match (values.contains(&false), values.contains(&true)) {
            (true, false) => Truth::False,
            (false, true) => Truth::True,
            (true, true) => Truth::Unknown,
            (false, false) => panic!("no possible value"),
        }
    }

    fn lifted1(a: Truth, op: fn(bool) -> bool) -> Truth {
        let values: Vec<bool> = possible(a).iter().map(|&x| op(x)).collect();
        from_possible(&values)
    }

    fn lifted2(a: Truth, b: Truth, op: fn(bool, bool) -> bool) -> Truth {
        let values: Vec<bool> = possible(a)
            .iter()
            .flat_map(|&x| possible(b).iter().map(move |&y| op(x, y)))
            .collect();
        from_possible(&values)
    }

    #[test]
    fn kleene_connectives_are_the_set_lifted_boolean_ones() {
        let mut cases = 0;
        for a in TRUTHS {
            assert_eq!(a.not(), lifted1(a, |x| !x), "not {a}");
            assert_eq!(!a, a.not(), "!{a}");
            for b in TRUTHS {
                assert_eq!(a.and(b), lifted2(a, b, |x, y| x && y), "{a} and {b}");
                assert_eq!(a.or(b), lifted2(a, b, |x, y| x || y), "{a} or {b}");
                assert_eq!(
                    a.implies(b),
                    lifted2(a, b, |x, y| !x || y),
                    "{a} implies {b}"
                );
                assert_eq!(a.iff(b), lifted2(a, b, |x, y| x == y), "{a} iff {b}");
                assert_eq!(a & b, a.and(b), "{a} & {b}");
                assert_eq!(a | b, a.or(b), "{a} | {b}");
                cases += 1;
            }
        }
        assert_eq!(cases, 9);
        // The cases a hand-written copy most often gets wrong.
        assert_eq!(Truth::Unknown.and(Truth::False), Truth::False);
        assert_eq!(Truth::Unknown.or(Truth::True), Truth::True);
        assert_eq!(Truth::False.implies(Truth::Unknown), Truth::True);
        assert_eq!(Truth::Unknown.iff(Truth::Unknown), Truth::Unknown);
    }

    #[test]
    fn truth_conversions_and_readings() {
        for b in [false, true] {
            assert_eq!(Truth::from(b), Truth::from_bool(b));
            assert_eq!(Truth::from(Some(b)), Truth::from_bool(b));
            assert_eq!(Truth::from_bool(b).to_option(), Some(b));
        }
        assert_eq!(Truth::from(None::<bool>), Truth::Unknown);
        for t in TRUTHS {
            assert_eq!(Truth::from(t.to_option()), t);
            assert_eq!(Option::<bool>::from(t), t.to_option());
            // "Not proved" and "refuted" differ exactly at `Unknown`.
            let not_proved = !t.is_true();
            let refuted = t.not().is_true();
            assert_eq!(not_proved == refuted, t.is_known(), "{t}");
        }
        let readings = |t: Truth| [t.is_true(), t.is_false(), t.is_known()];
        assert_eq!(readings(Truth::True), [true, false, true]);
        assert_eq!(readings(Truth::False), [false, true, true]);
        assert_eq!(readings(Truth::Unknown), [false, false, false]);
        assert_eq!(Truth::True.to_string(), "true");
        assert_eq!(Truth::False.to_string(), "false");
        assert_eq!(Truth::Unknown.to_string(), "unknown");
    }

    /// Every sequence over `TRUTHS` of length `0..=max_len`.
    fn sequences(max_len: usize) -> Vec<Vec<Truth>> {
        let mut all = vec![vec![]];
        let mut frontier: Vec<Vec<Truth>> = vec![vec![]];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for s in &frontier {
                for t in TRUTHS {
                    let mut longer = s.clone();
                    longer.push(t);
                    next.push(longer);
                }
            }
            all.extend(next.iter().cloned());
            frontier = next;
        }
        all
    }

    #[test]
    fn all_and_any_agree_with_folds_and_short_circuit() {
        let seqs = sequences(4);
        assert_eq!(seqs.len(), 1 + 3 + 9 + 27 + 81);
        for s in &seqs {
            let and_fold = s.iter().fold(Truth::True, |acc, &t| acc.and(t));
            let or_fold = s.iter().fold(Truth::False, |acc, &t| acc.or(t));
            assert_eq!(Truth::all(s.iter().copied()), and_fold, "all {s:?}");
            assert_eq!(Truth::any(s.iter().copied()), or_fold, "any {s:?}");

            // Nothing after the first decisive item (`False` for `all`,
            // `True` for `any`) is pulled.
            for decisive in [Truth::False, Truth::True] {
                let pulled = Cell::new(0);
                let lazy = s.iter().inspect(|_| pulled.set(pulled.get() + 1)).copied();
                let _ = if decisive.is_false() {
                    Truth::all(lazy)
                } else {
                    Truth::any(lazy)
                };
                let expected = s
                    .iter()
                    .position(|&t| t == decisive)
                    .map_or(s.len(), |i| i + 1);
                assert_eq!(
                    pulled.get(),
                    expected,
                    "short circuit on {decisive} in {s:?}"
                );
            }
        }
        assert_eq!(Truth::all(std::iter::empty::<Truth>()), Truth::True);
        assert_eq!(Truth::any(std::iter::empty::<Truth>()), Truth::False);
    }

    #[test]
    fn or_else_asks_only_when_unknown() {
        for first in TRUTHS {
            for second in TRUTHS {
                let calls = Cell::new(0);
                let got = first.or_else(|| {
                    calls.set(calls.get() + 1);
                    second
                });
                if first.is_known() {
                    assert_eq!((got, calls.get()), (first, 0), "{first} or_else {second}");
                } else {
                    assert_eq!((got, calls.get()), (second, 1), "{first} or_else {second}");
                }
            }
        }
        // Not `or`: a decided `False` is final.
        assert_eq!(Truth::False.or_else(|| Truth::True), Truth::False);
        assert_eq!(Truth::False.or(Truth::True), Truth::True);
    }

    // The core operations are usable in `const` items.
    const CONST_TRUTH: Truth = Truth::True.and(Truth::Unknown).not().or(Truth::False);
    const CONST_SIGNS: SignSet = SignSet::NEGATIVE
        .mul(SignSet::NONREAL)
        .add(SignSet::ZERO)
        .pow_int(2);

    #[test]
    fn core_operations_are_const() {
        assert_eq!(CONST_TRUTH, Truth::Unknown);
        assert_eq!(CONST_SIGNS, SignSet::NEGATIVE.union(SignSet::NONREAL));
    }

    // ── SignSet: sample values ──

    /// A finite complex sample in Cartesian form, for `+` (and `−x`,
    /// `conj`, `|x|`).  Each part of a sum is one IEEE addition, whose
    /// result has the sign of the exact sum and is 0 exactly when the exact
    /// sum is (rounding is monotone, `fl(−x) = −fl(x)`, and gradual underflow
    /// makes a tiny sum exact), so the sign class of a computed sum is exact
    /// whatever the sample values.
    #[derive(Clone, Copy, Debug)]
    struct Cart {
        re: f64,
        im: f64,
    }

    impl Cart {
        fn class(self) -> SignSet {
            if self.im != 0.0 {
                return SignSet::NONREAL;
            }
            match self.re.partial_cmp(&0.0) {
                Some(Ordering::Less) => SignSet::NEGATIVE,
                Some(Ordering::Equal) => SignSet::ZERO,
                Some(Ordering::Greater) => SignSet::POSITIVE,
                None => panic!("NaN sample"),
            }
        }

        fn add(self, other: Cart) -> Cart {
            Cart {
                re: self.re + other.re,
                im: self.im + other.im,
            }
        }

        fn neg(self) -> Cart {
            Cart {
                re: -self.re,
                im: -self.im,
            }
        }

        fn conj(self) -> Cart {
            Cart {
                re: self.re,
                im: -self.im,
            }
        }

        fn abs(self) -> Cart {
            Cart {
                re: self.re.hypot(self.im),
                im: 0.0,
            }
        }
    }

    fn cartesian_samples() -> Vec<Cart> {
        let h = 3f64.sqrt() / 2.0; // e^{±iπ/3}, e^{±2iπ/3}
        let s = std::f64::consts::FRAC_1_SQRT_2; // e^{±iπ/4}, e^{±3iπ/4}
        let mut out = Vec::new();
        for r in [3.0, 1.0, 0.25, 1e-9] {
            out.push(Cart { re: -r, im: 0.0 });
            out.push(Cart { re: r, im: 0.0 });
        }
        out.push(Cart { re: 0.0, im: 0.0 });
        out.push(Cart { re: -0.0, im: 0.0 });
        for (re, im) in [
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
            (-2.0, 0.5),
            (0.3, -4.0),
            (0.5, h),
            (0.5, -h),
            (-0.5, h),
            (-0.5, -h),
            (s, s),
            (s, -s),
            (-s, s),
            (-s, -s),
            (0.0, 1e-9),
            (3.0, -1e-9),
        ] {
            out.push(Cart { re, im });
        }
        out
    }

    /// A finite complex sample in polar form: `0`, or `r·e^{iπ·p/q}` with
    /// `r > 0`, `q > 0`.  Products, reciprocals, powers, negation,
    /// conjugation and `|x|` act on the angle by integer arithmetic, and the
    /// sign class depends on the angle alone (`p/q ≡ 0 (mod 2)`: positive,
    /// `≡ 1`: negative, otherwise non-real), so classification is exact with
    /// no floating point at all.  `r` affects no class and is not stored.
    #[derive(Clone, Copy, Debug)]
    enum Polar {
        Zero,
        Ray { p: i64, q: i64 },
    }

    impl Polar {
        fn class(self) -> SignSet {
            match self {
                Polar::Zero => SignSet::ZERO,
                Polar::Ray { p, q } => {
                    let m = p.rem_euclid(2 * q);
                    if m == 0 {
                        SignSet::POSITIVE
                    } else if m == q {
                        SignSet::NEGATIVE
                    } else {
                        SignSet::NONREAL
                    }
                }
            }
        }

        fn mul(self, other: Polar) -> Polar {
            match (self, other) {
                (Polar::Ray { p: a, q: b }, Polar::Ray { p: c, q: d }) => Polar::Ray {
                    p: a * d + c * b,
                    q: b * d,
                },
                _ => Polar::Zero,
            }
        }

        /// `None`: `1/0` is not finite.
        fn recip(self) -> Option<Polar> {
            match self {
                Polar::Zero => None,
                Polar::Ray { p, q } => Some(Polar::Ray { p: -p, q }),
            }
        }

        /// `None`: `0ⁿ` for `n < 0` is not finite.  `0⁰ = 1`, the crate's
        /// convention (`canon_pow`).
        fn pow(self, n: i64) -> Option<Polar> {
            match self {
                Polar::Ray { p, q } => Some(Polar::Ray { p: n * p, q }),
                Polar::Zero => match n.cmp(&0) {
                    Ordering::Greater => Some(Polar::Zero),
                    Ordering::Equal => Some(Polar::Ray { p: 0, q: 1 }),
                    Ordering::Less => None,
                },
            }
        }

        fn neg(self) -> Polar {
            match self {
                Polar::Zero => Polar::Zero,
                Polar::Ray { p, q } => Polar::Ray { p: p + q, q },
            }
        }

        fn conj(self) -> Polar {
            match self {
                Polar::Zero => Polar::Zero,
                Polar::Ray { p, q } => Polar::Ray { p: -p, q },
            }
        }

        fn abs(self) -> Polar {
            match self {
                Polar::Zero => Polar::Zero,
                Polar::Ray { .. } => Polar::Ray { p: 0, q: 1 },
            }
        }
    }

    fn polar_samples() -> Vec<Polar> {
        let mut out = vec![Polar::Zero];
        for (p, q) in [
            // Negative reals.
            (1, 1),
            (-1, 1),
            (3, 1),
            // Positive reals.
            (0, 1),
            (2, 1),
            (-4, 2),
            // Non-real: every angle π·p/q with q ≤ 7 that the tightness of
            // the power tables needs (π/2, π/|n|, 2π/|n|), and others.
            (1, 2),
            (-1, 2),
            (1, 3),
            (-1, 3),
            (2, 3),
            (-2, 3),
            (1, 4),
            (-1, 4),
            (3, 4),
            (-3, 4),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (-1, 5),
            (-2, 5),
            (1, 6),
            (5, 6),
            (-5, 6),
            (1, 7),
            (-3, 7),
        ] {
            out.push(Polar::Ray { p, q });
        }
        out
    }

    fn all_sets() -> impl Iterator<Item = SignSet> {
        (0u8..16).map(SignSet)
    }

    fn atom_index(a: SignSet) -> usize {
        ATOMS.iter().position(|&b| b == a).expect("a single atom")
    }

    type AtomTable = [[SignSet; 4]; 4];

    /// `seen[i][j]`: the classes of `op(x, y)` over all samples `x` of atom
    /// `ATOMS[i]` and `y` of atom `ATOMS[j]` for which `op` is defined.
    fn observe2<S: Copy>(
        samples: &[S],
        class: fn(S) -> SignSet,
        op: impl Fn(S, S) -> Option<S>,
    ) -> AtomTable {
        let mut seen = [[SignSet::EMPTY; 4]; 4];
        for &x in samples {
            for &y in samples {
                if let Some(z) = op(x, y) {
                    let (i, j) = (atom_index(class(x)), atom_index(class(y)));
                    seen[i][j] = seen[i][j].union(class(z));
                }
            }
        }
        seen
    }

    /// `seen[i]`: the classes of `op(x)` over all samples `x` of atom
    /// `ATOMS[i]` for which `op` is defined.
    fn observe1<S: Copy>(
        samples: &[S],
        class: fn(S) -> SignSet,
        op: impl Fn(S) -> Option<S>,
    ) -> [SignSet; 4] {
        let mut seen = [SignSet::EMPTY; 4];
        for &x in samples {
            if let Some(z) = op(x) {
                let i = atom_index(class(x));
                seen[i] = seen[i].union(class(z));
            }
        }
        seen
    }

    /// Soundness: every observed sign is in the table.  Tightness: every
    /// sign in the table is observed.
    fn assert_sound_and_tight(what: &str, table: SignSet, observed: SignSet) {
        assert!(
            observed.is_subset(table),
            "{what}: unsound — observed {observed}, table says {table}"
        );
        assert!(
            table.is_subset(observed),
            "{what}: not tight — table says {table}, observed only {observed}"
        );
    }

    /// Checks `op` on atom pairs against `seen`, then on all 16×16 sets
    /// against the union of `seen` over their atom pairs.
    fn check_binary(name: &str, op: fn(SignSet, SignSet) -> SignSet, seen: &AtomTable) {
        for (i, &a) in ATOMS.iter().enumerate() {
            for (j, &b) in ATOMS.iter().enumerate() {
                assert_sound_and_tight(&format!("{a} {name} {b}"), op(a, b), seen[i][j]);
            }
        }
        let mut pairs = 0;
        for a in all_sets() {
            for b in all_sets() {
                let mut union = SignSet::EMPTY;
                for (i, &x) in ATOMS.iter().enumerate() {
                    for (j, &y) in ATOMS.iter().enumerate() {
                        if a.contains(x) && b.contains(y) {
                            union = union.union(seen[i][j]);
                        }
                    }
                }
                assert_eq!(op(a, b), union, "{a} {name} {b}");
                pairs += 1;
            }
        }
        assert_eq!(pairs, 256);
    }

    /// Checks `op` on atoms against `seen`, then on all 16 sets against the
    /// union of `seen` over their atoms.
    fn check_unary(name: &str, op: impl Fn(SignSet) -> SignSet, seen: &[SignSet; 4]) {
        for (i, &a) in ATOMS.iter().enumerate() {
            assert_sound_and_tight(&format!("{name} for x in {a}"), op(a), seen[i]);
        }
        for s in all_sets() {
            let union = ATOMS
                .iter()
                .zip(seen)
                .filter(|(a, _)| s.contains(**a))
                .fold(SignSet::EMPTY, |acc, (_, &r)| acc.union(r));
            assert_eq!(op(s), union, "{name} for x in {s}");
        }
    }

    #[test]
    fn samples_cover_every_atom() {
        let cart = cartesian_samples();
        let polar = polar_samples();
        for atom in ATOMS {
            let c = cart.iter().filter(|z| z.class() == atom).count();
            let p = polar.iter().filter(|z| z.class() == atom).count();
            assert!(c >= 1 && p >= 1, "{atom}: {c} Cartesian, {p} polar samples");
        }
    }

    #[test]
    fn add_is_sound_and_tight() {
        let seen = observe2(&cartesian_samples(), Cart::class, |x, y| Some(x.add(y)));
        check_binary("+", SignSet::add, &seen);
    }

    #[test]
    fn mul_is_sound_and_tight() {
        let seen = observe2(&polar_samples(), Polar::class, |x, y| Some(x.mul(y)));
        check_binary("·", SignSet::mul, &seen);
    }

    #[test]
    fn unary_ops_are_sound_and_tight() {
        let cart = cartesian_samples();
        let polar = polar_samples();
        let c = |op: fn(Cart) -> Cart| observe1(&cart, Cart::class, move |x| Some(op(x)));
        let p = |op: fn(Polar) -> Polar| observe1(&polar, Polar::class, move |x| Some(op(x)));
        check_unary("−x", SignSet::neg, &c(Cart::neg));
        check_unary("−x", SignSet::neg, &p(Polar::neg));
        check_unary("conj x", SignSet::conj, &c(Cart::conj));
        check_unary("conj x", SignSet::conj, &p(Polar::conj));
        check_unary("|x|", SignSet::abs, &c(Cart::abs));
        check_unary("|x|", SignSet::abs, &p(Polar::abs));
        check_unary(
            "1/x",
            SignSet::recip,
            &observe1(&polar, Polar::class, Polar::recip),
        );
        let squares = observe1(&polar, Polar::class, |x| x.pow(2));
        check_unary("x²", SignSet::square, &squares);
    }

    #[test]
    fn pow_int_is_sound_and_tight_for_small_exponents() {
        let polar = polar_samples();
        for n in -5..=5 {
            let seen = observe1(&polar, Polar::class, |x| x.pow(n));
            check_unary(&format!("x^{n}"), |s| s.pow_int(n), &seen);
        }
    }

    #[test]
    fn pow_int_relations() {
        for s in all_sets() {
            assert_eq!(s.pow_int(1), s, "{s}");
            assert_eq!(s.pow_int(-1), s.recip(), "{s}");
            assert_eq!(s.pow_int(2), s.square(), "{s}");
            let one = if s.is_empty() {
                SignSet::EMPTY
            } else {
                SignSet::POSITIVE
            };
            assert_eq!(s.pow_int(0), one, "{s}");

            // A power is never wider than the repeated independent product.
            let inv = s.recip();
            let (mut product, mut inv_product) = (s, inv);
            for n in 2..=6 {
                product = product.mul(s);
                inv_product = inv_product.mul(inv);
                assert!(s.pow_int(n).is_subset(product), "{s}^{n}");
                assert!(s.pow_int(-n).is_subset(inv_product), "{s}^-{n}");
            }

            // Only the sign of n, its parity and min(|n|, 3) matter.
            for n in [
                7,
                8,
                1000,
                1001,
                i64::MAX,
                -7,
                -8,
                -1000,
                -1001,
                i64::MIN + 1,
                i64::MIN,
            ] {
                let m = if n % 2 == 0 { 4 } else { 3 };
                let representative = if n > 0 { m } else { -m };
                assert_eq!(s.pow_int(n), s.pow_int(representative), "{s}^{n}");
            }

            // Without the ZERO atom, x⁻ⁿ and xⁿ have the same signs.
            if !s.contains(SignSet::ZERO) {
                for n in 1..=6 {
                    assert_eq!(s.pow_int(-n), s.pow_int(n), "{s}^±{n}");
                }
            }
        }
    }

    /// A projection, the property set it tests, and the property itself on
    /// a Cartesian sample.
    struct Property {
        name: &'static str,
        query: fn(SignSet) -> Truth,
        set: SignSet,
        holds: fn(Cart) -> bool,
    }

    #[test]
    fn projections_on_all_sets() {
        let props = [
            Property {
                name: "positive",
                query: SignSet::is_positive,
                set: SignSet::POSITIVE,
                holds: |z| z.im == 0.0 && z.re > 0.0,
            },
            Property {
                name: "negative",
                query: SignSet::is_negative,
                set: SignSet::NEGATIVE,
                holds: |z| z.im == 0.0 && z.re < 0.0,
            },
            Property {
                name: "zero",
                query: SignSet::is_zero,
                set: SignSet::ZERO,
                holds: |z| z.im == 0.0 && z.re == 0.0,
            },
            Property {
                name: "nonzero",
                query: SignSet::is_nonzero,
                set: SignSet::NONZERO,
                holds: |z| z.im != 0.0 || z.re != 0.0,
            },
            Property {
                name: "nonnegative",
                query: SignSet::is_nonnegative,
                set: SignSet::NONNEGATIVE,
                holds: |z| z.im == 0.0 && z.re >= 0.0,
            },
            Property {
                name: "nonpositive",
                query: SignSet::is_nonpositive,
                set: SignSet::NONPOSITIVE,
                holds: |z| z.im == 0.0 && z.re <= 0.0,
            },
            Property {
                name: "real",
                query: SignSet::is_real,
                set: SignSet::REAL,
                holds: |z| z.im == 0.0,
            },
            Property {
                name: "nonreal",
                query: SignSet::is_nonreal,
                set: SignSet::NONREAL,
                holds: |z| z.im != 0.0,
            },
        ];
        let samples = cartesian_samples();
        let mut checked = 0;
        for s in all_sets() {
            let members: Vec<Cart> = samples
                .iter()
                .copied()
                .filter(|z| s.contains(z.class()))
                .collect();
            for p in &props {
                let all = members.iter().all(|&z| (p.holds)(z));
                let any = members.iter().any(|&z| (p.holds)(z));
                // The empty set is a contradiction: it answers nothing.
                let expected = match (members.is_empty(), all, any) {
                    (true, _, _) => Truth::Unknown,
                    (false, true, _) => Truth::True,
                    (false, false, false) => Truth::False,
                    (false, false, true) => Truth::Unknown,
                };
                assert_eq!((p.query)(s), expected, "{s}.is_{}", p.name);
                assert_eq!(s.is_in(p.set), expected, "{s}.is_in({})", p.set);
                checked += 1;
            }
            if !s.is_empty() {
                assert_eq!(s.is_nonzero(), s.is_zero().not(), "{s}");
                assert_eq!(s.is_nonreal(), s.is_real().not(), "{s}");
            }
        }
        assert_eq!(checked, 16 * 8);
    }

    /// Membership by raw bit, independent of the set methods.
    fn model(s: SignSet) -> [bool; 4] {
        [0, 1, 2, 3].map(|i| (s.0 & (1 << i)) != 0)
    }

    #[test]
    fn set_operations_match_a_membership_model() {
        for (i, a) in ATOMS.iter().enumerate() {
            assert_eq!(a.0, 1 << i, "{a}");
        }
        for a in all_sets() {
            let ma = model(a);
            assert_eq!(a.is_empty(), ma == [false; 4], "{a}");
            assert_eq!(a.is_single(), ma.iter().filter(|&&m| m).count() == 1, "{a}");
            let atoms: Vec<SignSet> = ATOMS
                .iter()
                .zip(ma)
                .filter(|(_, m)| *m)
                .map(|(&x, _)| x)
                .collect();
            assert_eq!(a.atoms().collect::<Vec<_>>(), atoms, "{a}");
            for b in all_sets() {
                let mb = model(b);
                let subset = (0..4).all(|i| !ma[i] || mb[i]);
                assert_eq!(a.is_subset(b), subset, "{a} ⊆ {b}");
                assert_eq!(b.contains(a), subset, "{b} ⊇ {a}");
                assert_eq!(
                    model(a.union(b)),
                    [0, 1, 2, 3].map(|i| ma[i] || mb[i]),
                    "{a} ∪ {b}"
                );
                assert_eq!(
                    model(a.intersect(b)),
                    [0, 1, 2, 3].map(|i| ma[i] && mb[i]),
                    "{a} ∩ {b}"
                );
            }
        }
        // Members in the order <0, 0, >0, nonreal.
        let named = [
            (SignSet::EMPTY, [false, false, false, false]),
            (SignSet::NEGATIVE, [true, false, false, false]),
            (SignSet::ZERO, [false, true, false, false]),
            (SignSet::POSITIVE, [false, false, true, false]),
            (SignSet::NONREAL, [false, false, false, true]),
            (SignSet::REAL, [true, true, true, false]),
            (SignSet::NONNEGATIVE, [false, true, true, false]),
            (SignSet::NONPOSITIVE, [true, true, false, false]),
            (SignSet::NONZERO, [true, false, true, true]),
            (SignSet::NONZERO_REAL, [true, false, true, false]),
            (SignSet::UNKNOWN, [true, true, true, true]),
        ];
        for (s, members) in named {
            assert_eq!(model(s), members, "{s}");
        }
    }

    #[test]
    fn ordering_and_i8_conversions() {
        let pairs = [
            (Ordering::Less, SignSet::NEGATIVE),
            (Ordering::Equal, SignSet::ZERO),
            (Ordering::Greater, SignSet::POSITIVE),
        ];
        for (o, s) in pairs {
            assert_eq!(SignSet::from_ordering(o), s);
            assert_eq!(s.to_ordering(), Some(o));
        }
        for v in i8::MIN..=i8::MAX {
            assert_eq!(
                SignSet::from_sign_i8(v),
                SignSet::from_ordering(v.cmp(&0)),
                "{v}"
            );
        }
        for s in all_sets() {
            let expected = pairs.iter().find(|&&(_, t)| t == s).map(|&(o, _)| o);
            assert_eq!(s.to_ordering(), expected, "{s}");
        }
    }

    #[test]
    fn display_and_debug_list_the_atoms() {
        assert_eq!(SignSet::EMPTY.to_string(), "{}");
        assert_eq!(SignSet::NONPOSITIVE.to_string(), "{<0, 0}");
        assert_eq!(SignSet::NONZERO.to_string(), "{<0, >0, nonreal}");
        assert_eq!(SignSet::UNKNOWN.to_string(), "{<0, 0, >0, nonreal}");
        assert_eq!(format!("{:?}", SignSet::ZERO), "{0}");
        let distinct: HashSet<String> = all_sets().map(|s| s.to_string()).collect();
        assert_eq!(distinct.len(), 16);
    }

    #[test]
    fn evidence_order_is_explicit() {
        for (i, a) in STRONGEST_FIRST.into_iter().enumerate() {
            assert_eq!(a.is_proof(), i < 2, "{a:?}");
            for (j, b) in STRONGEST_FIRST.into_iter().enumerate() {
                assert_eq!(a.at_least(b), i <= j, "{a:?} at least {b:?}");
                assert_eq!(a.stronger_than(b), i < j, "{a:?} stronger than {b:?}");
                assert_eq!(
                    a.weaker(b),
                    STRONGEST_FIRST[i.max(j)],
                    "weaker of {a:?}, {b:?}"
                );
            }
        }
    }

    #[test]
    fn verdicts_hide_weak_answers() {
        for (i, e) in STRONGEST_FIRST.into_iter().enumerate() {
            let t = Verdict::new(Truth::True, e);
            let s = Verdict::new(SignSet::NEGATIVE, e);
            assert_eq!(t.proven(), (i < 2).then_some(Truth::True), "{e:?}");
            assert_eq!(s.proven(), (i < 2).then_some(SignSet::NEGATIVE), "{e:?}");
            for (j, min) in STRONGEST_FIRST.into_iter().enumerate() {
                let enough = i <= j;
                assert_eq!(
                    t.at_least(min),
                    enough.then_some(Truth::True),
                    "{e:?} vs {min:?}"
                );
                assert_eq!(
                    s.at_least(min),
                    enough.then_some(SignSet::NEGATIVE),
                    "{e:?} vs {min:?}"
                );
                let truth = if enough { Truth::True } else { Truth::Unknown };
                assert_eq!(t.truth_at_least(min), truth, "{e:?} vs {min:?}");
                let signs = if enough {
                    SignSet::NEGATIVE
                } else {
                    SignSet::UNKNOWN
                };
                assert_eq!(s.signs_at_least(min), signs, "{e:?} vs {min:?}");
            }
            assert_eq!(s.map(SignSet::is_negative), Verdict::new(Truth::True, e));
        }
    }
}
