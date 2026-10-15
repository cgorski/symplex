//! One identity for every mathematical function: [`Func`], with its
//! declarative data in a [`FuncSpec`], generated from **one list** together
//! with the public [`LibFn`].
//!
//! A function reaches the arena in one of three representations:
//!
//! - a dedicated [`ExprNode`] variant (`Sin(x)`, `Beta(a, b)`, `Min(…)`:
//!   the 43 *node-backed* functions);
//! - an `ExprNode::Lib(f, args)` node of a library function (the 50
//!   [`LibFn`]s: Bessel, orthogonal polynomials, `erfi`, …), identified by
//!   `f` (until 0.49 an `Apply` resolved by the text of its head);
//! - an `ExprNode::Internal(f, args)` node of one of the 8 internal
//!   pseudo-functions of Gruntz's asymptotic rewrite ([`InternalFn`]:
//!   `__ei_asym`, …; they exist only inside a limit computation, and no
//!   name builds them — until 0.49 an `Apply` with a reserved head, which
//!   `Context::from_tree` built from the name).
//!
//! Until 0.49 nothing named "the function" across the three: code that had
//! to treat a family uniformly matched node variants *and* compared `Apply`
//! names, and code that identified a node by its shape confused functions —
//! `evalf`'s conjugate pairing signed every `Apply` alike, so `Shi(z̄)` and
//! `erfi(z)` "paired" and an imaginary part of 10⁻⁴⁵ became exactly 0, a
//! value off by 2πi after a logarithm.  [`Func`] is that identity: one
//! variant per function whatever its representation, `Copy`, hashable, and
//! exhaustive, so a `match f` that forgets a function does not compile.
//! User functions (`Context::apply("f", …)`) are not functions in this
//! sense: nothing is known about them, and they have no `Func`.
//!
//! # The list
//!
//! Every function is one entry of the `funcs!` invocation below: its
//! variant name and a [`FuncSpec`] struct literal written out in full
//! (`name` first, no `..`).  Adding a field to `FuncSpec` makes every entry
//! that does not set it a compile error (E0063), so a new function cannot
//! forget a facet and a new facet cannot forget a function.  The library
//! entries also generate the public [`LibFn`] — same variants, order, names
//! and arities as before the list existed — and `From<LibFn> for Func`.
//!
//! Only facets that are *data*, uniform across the representations, live in
//! the spec: the name, the parser's further names, the argument count, the
//! sort-key sub-rank and the LaTeX layout, and six facets with semantics,
//! each read by the code that used to keep a list of its own:
//!
//! | Facet | Says | Read by |
//! |---|---|---|
//! | [`Conj`] | where `conj f(z…) = f(conj z…)` | `conjugate` (`base/complex.rs`), `evalf`'s exact-realness signatures |
//! | [`RealOn`] | on which arguments `f` is real | the assumption system (`compute_func`) |
//! | [`Parity`] | `f(−z) = ±f(z)`, and how the sign is extracted today | a test against `canon`/`eval` |
//! | [`Poles`] | where `f` is infinite at finite arguments | the assumption system (no claim at a known pole) |
//! | [`Constants`] | whether its special values at constants are folded | the assumption system (no claim at constants otherwise) |
//! | [`Runtime`] (`rt`) | which numeric back ends evaluate `f` in `f64`, with which runtime helper | `compile`, `to_rust_fn`, `to_c_fn`, `to_python` (availability, refusals) |
//!
//! Up to 0.48 symbolic conjugation knew 14 functions and `evalf` 35, in
//! two lists that disagreed, and the assumption system knew no library
//! function at all: `conjugate(erfi(z))` stayed unevaluated and
//! `erfi(x)` for a real `x` was not known to be real.  The values come
//! from DLMF and SymPy; an entry says why where it is not obvious, and
//! each rule is checked numerically (mpmath at two complex points off
//! the cut).  Algorithms (evaluation, derivatives, series, integration)
//! stay in their modules as exhaustive `match f: Func` arms.
//!
//! # Which entry point
//!
//! | Need | Use |
//! |---|---|
//! | which function a node applies, and to what | [`ExprNode::func`] (`None` for non-functions and user functions) |
//! | build `f(args)` as today's constructors do (canonical folds included) | [`Arena::func`] (arity-checked) |
//! | a function's data | [`Func::spec`] |
//! | the public library-function type | [`LibFn`] (`Func::from`, [`Func::as_lib`]) |
//!
//! # Corner cases
//!
//! - **Lambert W.**  The principal branch `W(x)` is the node
//!   `ExprNode::LambertW`, [`Func::LambertW0`]; the library function
//!   `lambertw(x, k)` ([`LibFn::LambertW`], [`Func::LambertW`]) carries the
//!   other branches.  `Arena::func(Func::LambertW, [x])` and `[x, 0]` build
//!   the node, as `lambertw(x)` and `lambertw(x, 0)` parse.  Both print as
//!   `W`; the library entry's `name` is its interned head `lambertw`.
//! - **Names.**  `name` is the head `Display` prints in call notation (and
//!   the head of the JSON `Apply` form of a library function, the interned
//!   head symbol of an internal one).  Two exceptions, both
//!   tested: `Factorial` prints postfix (`n!`) and `lambertw(x, k)` prints
//!   as `W(x, k)`.  The parser compares names ignoring ASCII case, except
//!   Heaviside's display name `H`, which it reads only as written (and only
//!   while no undefined function `H` exists); `parse` lists the names it
//!   reads besides `name`, at the function's own arity (`log(x, b)` is a
//!   quotient, not `ln`).
//! - **Wrong argument counts.**  A library function's node always has an
//!   argument count the function accepts: [`Arena::func`] refuses any
//!   other, and so do [`Context::apply`] (`ctx.apply("Shi", [x, 3])` is an
//!   error; until 0.49 it built a node every consumer left unevaluated),
//!   the parser and `from_json`.  An internal `Apply` of another count is
//!   not a call of that function ([`ExprNode::func`] returns `None`).
//! - **Built-in names.**  [`Context::apply`] with the name of a built-in
//!   function builds that function (`apply("sin", [x])` is `sin(x)`,
//!   `apply("bell", [3])` the Bell number); `Apply` means a user function
//!   (or an internal head), never a library function.
//! - **Sort keys.**  [`SortSub`] reproduces the hand-assigned bytes of
//!   `sort_key.rs` exactly.  `Binomial` and `Asin` share byte 8 (since
//!   0.13; byte identity keeps it), the one allowlisted duplicate.
//! - **`Min`/`Max`** take any positive number of arguments
//!   ([`ArgCount::AtLeast`]); one argument folds to itself.  The parser asks
//!   for two.
//! - **Internal functions have no name.**  An internal function is a node
//!   of its own ([`ExprNode::Internal`]); its head is printed, but no name
//!   converts back to it, so a user expression never takes on the
//!   asymptotic semantics of `__ei_asym` (up to 0.48 `apply("__ei_asym",
//!   [x])` had limit 0 at `∞`, and up to 0.49 so had `from_tree` of an
//!   `Apply` named `__ei_asym`).  The heads stay reserved:
//!   [`Context::apply`] refuses them, the parser does not read them and
//!   `from_json` refuses them; `Context::from_tree` (infallible) builds a
//!   user function of that name, about which nothing is known.
//! - **Poles and the generic point.**  A claim of the table ("real",
//!   "finite complex") is withheld only where an argument is *known* to be
//!   at a pole ([`Poles`]); a symbol is a generic point, so `Γ(x)` for a
//!   real `x` is real, as it was before the table.  A *constant* is not a
//!   generic point: a function whose special values at constants are not
//!   all folded at construction ([`Constants::Evaluated`]: the library
//!   functions, `B`) gets no claim at constant arguments — it is a number
//!   for evaluation to decide, and an unfolded zero (`subfactorial(1)`,
//!   `chebyshev_t(3, 0)`) called "real" would make `√2/subfactorial(1)`
//!   real.
//!
//! [`Context::apply`]: crate::api::context::Context::apply

use std::fmt;

use smallvec::SmallVec;
#[cfg(test)]
use smallvec::smallvec;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::libfn::Arity;
use crate::base::node::{ExprId, ExprNode};

// ---------------------------------------------------------------------------
// The facets
// ---------------------------------------------------------------------------

/// How many arguments a [`Func`] takes.
///
/// The public [`Arity`] of a [`LibFn`] cannot say "any number": `Min` and
/// `Max` need [`AtLeast`](Self::AtLeast).  [`LibFn::arity`] is this, read
/// back as an `Arity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ArgCount {
    /// Exactly `n` arguments.
    Exactly(u8),
    /// Between `min` and `max` arguments, inclusive (`lambertw(x)` or
    /// `lambertw(x, k)`).
    Between {
        /// Fewest arguments accepted.
        min: u8,
        /// Most arguments accepted.
        max: u8,
    },
    /// `min` arguments or more (`min(a, b, …)`).
    AtLeast(u8),
}

impl ArgCount {
    /// Does a call with `n` arguments have an acceptable shape?
    pub(crate) const fn accepts(self, n: usize) -> bool {
        match self {
            ArgCount::Exactly(k) => n == k as usize,
            ArgCount::Between { min, max } => min as usize <= n && n <= max as usize,
            ArgCount::AtLeast(min) => min as usize <= n,
        }
    }

    /// The public [`Arity`] of a library function.  No library function
    /// is variadic (tested); `AtLeast` would read as a range up to 255.
    const fn lib_arity(self) -> Arity {
        match self {
            ArgCount::Exactly(n) => Arity::Fixed(n),
            ArgCount::Between { min, max } => Arity::Range { min, max },
            ArgCount::AtLeast(min) => Arity::Range { min, max: u8::MAX },
        }
    }
}

impl fmt::Display for ArgCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ArgCount::Exactly(1) => f.write_str("1 argument"),
            ArgCount::Exactly(n) => write!(f, "{n} arguments"),
            ArgCount::Between { min, max } => write!(f, "{min} to {max} arguments"),
            ArgCount::AtLeast(1) => f.write_str("at least 1 argument"),
            ArgCount::AtLeast(n) => write!(f, "at least {n} arguments"),
        }
    }
}

/// Where a function's nodes sort: the leading bytes of their sort key,
/// before the arguments' keys (see `sort_key.rs`, whose bytes these
/// reproduce; a test compares them for every [`Func`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SortSub {
    /// The function rank (100), then this byte.
    Function(u8),
    /// A rank of its own (`Min` 130, `Max` 132).
    Rank(u8),
    /// The function rank (100), the `Apply` byte (17), the name and a 0
    /// byte: library and internal functions sort by name among themselves
    /// and with user functions (a library function's `Lib` node has the
    /// key its `Apply` node had until 0.49).
    Named,
}

/// The LaTeX layout of a call, with each argument rendered in place: the
/// notation of SymPy's `latex()` where it has one (`T_{n}\left(x\right)`,
/// `F_{n}`, `{x}^{\left(n\right)}` for the rising and
/// `{\left(x\right)}_{n}` for the falling factorial, the combinatorial
/// convention).  The LaTeX printer renders the library functions from it;
/// a test checks the node-backed ones against their own printer arms.  Up to
/// 0.49 library functions without a notation printed their names, and an
/// underscore in one was a subscript (`rising_factorial\left(x, y\right)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Tex {
    /// `head\left(a, b, …\right)`.
    Call(&'static str),
    /// Literal pieces and argument slots, in output order
    /// (`\binom{n}{k}`, `\operatorname{E}_{n}\left(x\right)`).
    Layout(&'static [TexPiece]),
    /// `a!` for an atom `a`, `\left(a\right)!` otherwise.
    Postfix(&'static str),
}

/// One piece of a [`Tex::Layout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TexPiece {
    /// Literal LaTeX.
    Lit(&'static str),
    /// The argument at this position.
    Arg(u8),
    /// The argument at this position, in `\left(…\right)` unless it is an
    /// atom (the base of `{x}^{\left(n\right)}`, the operand of `!n`).
    Group(u8),
}

/// A branch cut: where, in the values of one argument, a function's
/// principal branch is discontinuous (DLMF §4.2(i), §4.23(vii), §6.2,
/// §10.2(ii), §8.2(i), §25.12).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Cut {
    /// The position of the argument whose values are cut.
    pub(crate) arg: u8,
    /// The cut.
    pub(crate) line: CutLine,
}

/// The point sets of the complex plane that the cuts of the table are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CutLine {
    /// `(−∞, 0]`: `ln`, `ln Γ`, `Ci`, `Ei`, `Chi`, `E_n`, the incomplete
    /// gamma functions, the Bessel functions of non-integer order.
    NonPositiveReals,
    /// `(−∞, 1]`: `acosh`, `li`.
    BelowOne,
    /// `[1, ∞)`: `polylog`, the complete elliptic integrals.
    AtLeastOne,
    /// `(−∞, −1] ∪ [1, ∞)`: `asin`, `acos`, `atanh`.
    OutsideUnitInterval,
    /// `(−i∞, −i] ∪ [i, i∞)`: `atan`, `asinh`.
    ImaginaryOutsideUnit,
}

/// Complex conjugation: where `conj f(z₁, …, zₙ) = f(conj z₁, …, conj
/// zₙ)` (every argument conjugated) holds.  `conjugate` (`base/complex.rs`)
/// pushes the conjugate inside where the rule holds by the assumptions, and
/// `evalf`'s exact-realness signatures (`evalf/conjugate.rs`) pair `f(z̄)`
/// with `f(z)` where it holds for the computed values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Conj {
    /// Everywhere: the function is entire or meromorphic in every
    /// argument and real on the real axis (Schwarz reflection, DLMF
    /// §1.10(ii)).
    Entire,
    /// Everywhere off the cut; on it the principal branch breaks the
    /// symmetry (`conj √−1 = −i ≠ √(conj −1)`).
    OffCut(Cut),
    /// Everywhere when the first argument (an order) is an integer
    /// (`nonnegative`: an integer `≥ 0`), where the function is entire or
    /// meromorphic in the others; off the cut otherwise (`J_ν`, `I_ν`;
    /// `ψ⁽ᵐ⁾`, which is `ln Γ`, cut, at `m = −1`).
    IntegerOrderOffCut {
        /// The cut for the other orders.
        cut: Cut,
        /// Whether the order must also be non-negative.
        nonnegative: bool,
    },
    /// Everywhere when argument `degree` is a non-negative integer: the
    /// function is then a polynomial in the others with real coefficients.
    /// No rule otherwise.
    Polynomial {
        /// The position of the degree.
        degree: u8,
    },
    /// No rule: none holds, or one holds off a cut or for a definition
    /// the consumers cannot test (the entry says which).
    No,
}

/// Where a function is real: the argument domain on which the assumption
/// system claims `f(args)` real.  The handler of the table
/// (`AssumptionCache::compute_func`) reads the variants up to
/// [`NonNegativeIntegers`](Self::NonNegativeIntegers); the functions with
/// the later variants have handlers of their own, which a test checks
/// against the variant.
///
/// Claims are made away from the poles only in the sense of [`Poles`]: an
/// argument known to be at a pole withholds every claim; one merely real
/// does not (`Γ(x)` for a real `x` is real).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RealOn {
    /// Real when every argument is real; finite complex when every
    /// argument is (`erfi`, `Γ`, `B(a, b)`, `min`).
    Reals,
    /// Real when every argument is real; defined there only, so no complex
    /// claim (the real-analytic continuations of counting sequences that
    /// `evalf` evaluates at real arguments: `fibonacci`, `bernoulli`).
    RealLine,
    /// Real when the last argument is positive (and finite) and the others
    /// are real; finite complex when the last argument is (`ln Γ`, `Chi`,
    /// `Y_ν`, `ψ⁽ᵐ⁾`: the earlier arguments are parameters).
    PositiveReals,
    /// Real when the last argument is non-negative (and finite) and the
    /// others are real; finite complex as for
    /// [`PositiveReals`](Self::PositiveReals) (`W₀`, `H_x`).
    NonNegativeReals,
    /// Real when the last argument is a real below 1 (known non-positive,
    /// or a rational literal `< 1`) and the others are real; finite complex
    /// as for [`PositiveReals`](Self::PositiveReals) (`K(m)`, `F(φ | m)`).
    BelowOne,
    /// Real for a real argument known to lie in `[−1, 1]` (`closed`) or
    /// `(−1, 1)` (a rational literal, `sin`/`cos`/`tanh` of a real): `asin`,
    /// `acos`, `atanh`, `erfinv`.  No complex claim by the table.
    UnitInterval {
        /// Whether `±1` belong to the domain.
        closed: bool,
    },
    /// Real when argument `degree` is a non-negative integer and the
    /// others are real; finite complex when it is one and the others are
    /// complex (the orthogonal polynomials, `(x)ₙ`).
    Polynomial {
        /// The position of the degree.
        degree: u8,
    },
    /// Real when the order (first argument) is an integer and the second
    /// argument real, or the order is real and the second argument
    /// positive; finite complex when both are complex (`J_ν`, `I_ν`).
    BesselOrder,
    /// Real when every argument is a non-negative integer (the counting
    /// sequences, defined at the integers only).  No complex claim.
    NonNegativeIntegers,
    /// Real on `[1, ∞)` (`acosh`).
    AtLeastOne,
    /// Integer-valued for a real argument (`floor`, `ceiling`, `sign`).
    IntegerValued,
    /// Real for every argument (`abs`, `re`, `im`, `arg`).
    Everywhere,
    /// Decided by a handler of its own from the arguments' facts
    /// (`conjugate`, `KroneckerDelta`).
    Custom,
    /// No claim: the function may be real somewhere (the entry says
    /// where), but the assumption system cannot express where.
    Unclaimed,
}

/// Symmetry under `z ↦ −z` (of the only argument), and how today's
/// canonical forms and `eval` extract the sign.  The extraction is
/// written in `canon.rs` and `eval.rs`; a test checks it against this
/// facet for every function, so a change to one is a test failure until
/// the other follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Parity {
    /// `f(−z) = −f(z)`.
    Odd(AutoSign),
    /// `f(−z) = f(z)`.
    Even(AutoSign),
    /// Neither (and every function of more than one argument).
    Neither,
}

/// Which negated arguments the sign is pulled out of automatically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AutoSign {
    /// At construction, from any argument with a negative coefficient
    /// (`Si(−2x)` is `−Si(2x)`).
    Canonical,
    /// By `eval`, from any argument with a negative coefficient
    /// (`erfi(−2x)` → `−erfi(2x)`).
    Coefficient,
    /// By `eval`, from a negated term or a negative number (`sin(−x)` →
    /// `−sin(x)`; `sin(−2x)` stays).
    Negation,
    /// By `eval`, from a symbol-free argument only (`erf(−2)` → `−erf(2)`;
    /// `erf(−x)` stays).
    SymbolFree,
    /// Never.
    Never,
}

/// Whether the table's claims extend to constant arguments.
///
/// The assumption system treats a real value as a generic point (`1/x` is
/// real for a real `x`).  A constant is a specific number: if it is an
/// unfolded exact zero, "real" makes `1/c` real and `im(1/c)` 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Constants {
    /// The constructors fold the function's special values at constant
    /// arguments (`Γ(−2) = zoo`, `ζ(−2) = 0`, `Si(0) = 0`): claims hold for
    /// constants as for symbols.
    Folded,
    /// Some special values at constant arguments stay unfolded
    /// (`subfactorial(1)`, `chebyshev_t(3, 0)`, `B(1/2, −1/2)` are 0): claims
    /// are made for arguments with a symbol only, and a constant is left to
    /// evaluation.
    Evaluated,
}

/// Where a function is infinite (a pole or a logarithmic singularity) at
/// finite arguments.  An argument *known* to be there withholds every
/// claim of the table's handler (`Γ(n)` for an integer `n ≤ 0` is
/// `zoo`); claims for arguments merely real or complex are made (the
/// convention of the assumption system: a symbol is a generic point).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Poles {
    /// Finite at every finite argument.
    None,
    /// Where any of these arguments is an integer `≤ 0`.
    NonPositiveIntegers(&'static [u8]),
    /// Where this argument is 0.
    Zero(u8),
    /// Where this argument is 1.
    One(u8),
    /// Elsewhere, at points the facet does not describe (`tan` at
    /// `π/2 + kπ`): the table then claims no complex value, and claims
    /// realness only on a domain clear of them.
    Other,
}

/// One numeric back end: a consumer of the [`Runtime`] facet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Backend {
    /// The stack VM behind `Ex::compile`.
    Vm,
    /// The Rust source emitter (`to_rust_fn`).
    Rust,
    /// The C99 source emitter (`to_c_fn`).
    C,
    /// The Python source emitter (`to_python`, the `math` module only).
    Python,
}

impl Backend {
    /// The bit of this back end in a [`Backends`] set.
    const fn bit(self) -> u8 {
        match self {
            Backend::Vm => 1,
            Backend::Rust => 2,
            Backend::C => 4,
            Backend::Python => 8,
        }
    }

    /// The entry point, as error messages name it.
    pub(crate) const fn entry_point(self) -> &'static str {
        match self {
            Backend::Vm => "compile",
            Backend::Rust => "to_rust_fn",
            Backend::C => "to_c_fn",
            Backend::Python => "to_python",
        }
    }
}

/// A set of [`Backend`]s.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Backends(u8);

impl Backends {
    /// No back end.
    pub(crate) const NONE: Backends = Backends(0);
    /// The VM and the Rust and C emitters: every back end that shares the
    /// `f64` runtime (`numeric_rt` and its C port).
    pub(crate) const VM_RUST_C: Backends = Backends(1 | 2 | 4);
    /// Every back end.
    pub(crate) const ALL: Backends = Backends(1 | 2 | 4 | 8);

    /// Does the set contain `b`?
    pub(crate) const fn contains(self, b: Backend) -> bool {
        self.0 & b.bit() != 0
    }
}

impl fmt::Debug for Backends {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let all = [Backend::Vm, Backend::Rust, Backend::C, Backend::Python];
        f.debug_set()
            .entries(all.iter().filter(|b| self.contains(**b)))
            .finish()
    }
}

/// The `f64` routine of a function, the same in every back end that
/// implements it ([`Runtime::backends`]).  A helper `h` is the function
/// `numeric_rt::h` the VM calls, emitted as `symplex_rt::h` into generated
/// Rust (the runtime source embedded verbatim) and as the C port
/// `symplex_h` into generated C.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    /// Each back end lowers the function in an arm of its own: an
    /// operator, an `f64` method or a `<math.h>` function, or a runtime
    /// helper it picks (`sin`, `|x|`, `Γ`: Rust `symplex_rt::gamma`, C
    /// `tgamma`, Python `math.gamma`).
    Own,
    /// `h(x)`.
    Unary(&'static str),
    /// `h(a, b)`.
    Binary(&'static str),
    /// `h(n, x)` with an order or degree `n` that must be a constant
    /// integer.
    Ordered(&'static str),
    /// One helper per branch that is real somewhere on the real axis, the
    /// branch chosen by a constant argument: `lambertw(x, k)` is `W₀` or
    /// `W₋₁` (`codegen::lambert_rt_helper`).
    Branches(&'static [&'static str]),
    /// No `f64` routine: the function is complex-valued on the reals
    /// (`conjugate`, `arg`), an exact count (`bell`), or not implemented
    /// in `f64` yet.
    None,
}

impl Kernel {
    /// The runtime helpers the kernel calls (none for [`Own`](Self::Own)
    /// and [`None`](Self::None)).
    #[cfg(test)]
    pub(crate) fn helpers(self) -> SmallVec<[&'static str; 2]> {
        match self {
            Kernel::Unary(h) | Kernel::Binary(h) | Kernel::Ordered(h) => smallvec![h],
            Kernel::Branches(hs) => hs.iter().copied().collect(),
            Kernel::Own | Kernel::None => SmallVec::new(),
        }
    }
}

/// The `rt` facet: which numeric back ends evaluate a function in `f64`,
/// and with which routine.
///
/// Read by `compile` (the VM), `to_rust_fn`, `to_c_fn` and `to_python`:
/// a back end not in [`backends`](Self::backends) refuses the function,
/// and the refusal says where the function is available instead (another
/// back end, or the SciPy routine for Python).  The tests check every
/// claim against the emitters, and that each helper a claim names exists
/// in the embedded Rust runtime and in the C helper library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Runtime {
    /// The routine.
    pub(crate) kernel: Kernel,
    /// The back ends implementing it.
    pub(crate) backends: Backends,
    /// The `scipy.special` routine with the function's value, written as
    /// a call over the function's own argument names (SciPy 1.13); named
    /// in the refusals of the Python, NumPy and Julia printers, which emit
    /// `math`/`numpy`/base Julia only.  `None` for the functions Python
    /// spells, and where SciPy has no routine (the integer sequences,
    /// `li`, `polylog`, `dirichlet_eta`, `elliptic_pi`) or only a
    /// differently shaped one (`bernoulli`/`euler` return the whole
    /// sequence; `lambertw` is complex-valued).
    pub(crate) scipy: Option<&'static str>,
}

impl Runtime {
    /// Lowered by each back end in `on` in an arm of its own.
    const fn own(on: Backends) -> Runtime {
        Runtime {
            kernel: Kernel::Own,
            backends: on,
            scipy: None,
        }
    }

    /// The helper kernel `kernel` in the back ends `on`.
    const fn helper(kernel: Kernel, on: Backends) -> Runtime {
        Runtime {
            kernel,
            backends: on,
            scipy: None,
        }
    }

    /// No `f64` routine in any back end.
    const NONE: Runtime = Runtime {
        kernel: Kernel::None,
        backends: Backends::NONE,
        scipy: None,
    };

    /// The same, with the SciPy routine `call`.
    const fn scipy(self, call: &'static str) -> Runtime {
        Runtime {
            scipy: Some(call),
            ..self
        }
    }

    /// Does `b` implement the function?
    pub(crate) const fn on(self, b: Backend) -> bool {
        self.backends.contains(b)
    }

    /// The end of `b`'s refusal message: the other back ends that do
    /// implement the function (` (available in to_rust_fn, to_c_fn)`), or
    /// nothing when none does.
    pub(crate) fn elsewhere(self, b: Backend) -> String {
        let others: Vec<&str> = [Backend::Vm, Backend::Rust, Backend::C, Backend::Python]
            .into_iter()
            .filter(|&o| o != b && self.on(o))
            .map(Backend::entry_point)
            .collect();
        if others.is_empty() {
            String::new()
        } else {
            format!(" (available in {})", others.join(", "))
        }
    }
}

/// The data of one function.  Every entry of the list writes every field
/// (a struct literal without `..`), so a new field is a compile error
/// until each function says what it is.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "`parse`, `sort` and `parity` are read by the tests over `Func::ALL`; the \
                  parser, sort keys and sign extraction migrate to them later"
    )
)]
pub(crate) struct FuncSpec {
    /// The name: the head `Display` prints in call notation (`sin`,
    /// `Gamma`, `W`, `C`), for a library function the head of its JSON
    /// `Apply` form (`besselj`, `lambertw`), and for an internal function
    /// the head its node prints with (`__ei_asym`).  See the module
    /// documentation for the two printing exceptions.
    pub(crate) name: &'static str,
    /// Further names the parser reads as this function at its arity
    /// (`arcsin`, `ceil`, `log`), compared ignoring ASCII case.
    pub(crate) parse: &'static [&'static str],
    /// The accepted argument counts.
    pub(crate) arity: ArgCount,
    /// The sort-key sub-rank.
    pub(crate) sort: SortSub,
    /// The LaTeX layout.
    pub(crate) tex: Tex,
    /// Where complex conjugation commutes with the function.
    pub(crate) conj: Conj,
    /// Where the function is real.
    pub(crate) real_on: RealOn,
    /// Symmetry under negation and today's automatic sign extraction.
    pub(crate) parity: Parity,
    /// Where the function is infinite.
    pub(crate) poles: Poles,
    /// Whether the claims extend to constant arguments.
    pub(crate) constants: Constants,
    /// Which numeric back ends evaluate the function in `f64`, and how.
    pub(crate) rt: Runtime,
}

/// Which representation a [`Func`] has in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Family {
    /// A dedicated [`ExprNode`] variant.
    Node,
    /// An `ExprNode::Lib` node of a [`LibFn`].
    Lib,
    /// An `ExprNode::Internal` node of an [`InternalFn`] (never parsed).
    Internal,
}

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------

/// Generates [`Func`], its spec table, the public [`LibFn`] and the
/// conversions from the three groups of entries.  Each entry is
/// `Variant => FuncSpec { name: "…", <every other field> };`; the `name`
/// literal is matched separately so that `LibFn::name`/`from_name` stay
/// plain `match`es on string literals.
macro_rules! funcs {
    (
        node {
            $( $(#[$ndoc:meta])* $N:ident => FuncSpec {
                name: $nname:literal, $( $nfield:ident : $nvalue:expr ),+ $(,)?
            }; )*
        }
        lib {
            $( $(#[$ldoc:meta])* $L:ident => FuncSpec {
                name: $lname:literal, $( $lfield:ident : $lvalue:expr ),+ $(,)?
            }; )*
        }
        internal {
            $( $(#[$idoc:meta])* $I:ident => FuncSpec {
                name: $iname:literal, $( $ifield:ident : $ivalue:expr ),+ $(,)?
            }; )*
        }
    ) => {
        /// A library special function: the identity of its expression node
        /// (an `Apply(name, args)` resolved by name until 0.49).
        ///
        /// The variant order is the declaration order of the historical
        /// `FN_*` name constants and is the order of [`LibFn::ALL`].  The
        /// type is generated, together with the crate's internal function
        /// identity, from the one function list in `base/func.rs`.
        ///
        /// ```
        /// use symplex::base::libfn::LibFn;
        ///
        /// assert_eq!(LibFn::UpperGamma.name(), "uppergamma");
        /// assert_eq!(LibFn::from_name("besselj"), Some(LibFn::BesselJ));
        /// assert_eq!(LibFn::from_name("sin"), None);
        /// assert_eq!(LibFn::ALL.len(), 50);
        /// assert_eq!(LibFn::Shi.to_string(), "Shi");
        /// ```
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum LibFn {
            $( $(#[$ldoc])* $L, )*
        }

        impl LibFn {
            /// Every library function, in declaration order.
            pub const ALL: &[LibFn] = &[ $( LibFn::$L, )* ];

            /// The name: what `Display` prints (`lambertw(x, k)` prints as
            /// `W(x, k)`) and the head of the JSON `Apply` form.
            pub const fn name(self) -> &'static str {
                match self { $( LibFn::$L => $lname, )* }
            }

            /// The function whose interned name is exactly `s`.
            pub fn from_name(s: &str) -> Option<LibFn> {
                match s { $( $lname => Some(LibFn::$L), )* _ => None }
            }

            /// The number of arguments the function takes; a call of any
            /// other length is left unevaluated by every consumer.
            pub const fn arity(self) -> Arity {
                self.func().spec().arity.lib_arity()
            }

            /// The function identity of this library function.
            pub(crate) const fn func(self) -> Func {
                match self { $( LibFn::$L => Func::$L, )* }
            }
        }

        /// An internal pseudo-function of the limit engine: the identity of
        /// an [`ExprNode::Internal`] node.
        ///
        /// The type is `pub` only because `ExprNode` is reachable from the
        /// macro support module; it is not nameable outside the crate, so
        /// no user code constructs one, and no name converts to one (there
        /// is no `from_name`): an internal function is built only by
        /// [`Arena::func`].  Generated from the `internal` entries of the
        /// function list.
        // The variants are `Func`'s, whose internal entries are all
        // asymptotic heads (`EiAsym`, …).
        #[allow(clippy::enum_variant_names)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum InternalFn {
            $( $(#[$idoc])* $I, )*
        }

        impl InternalFn {
            /// The head it prints with (`__ei_asym`): its [`FuncSpec::name`].
            pub(crate) const fn name(self) -> &'static str {
                match self { $( InternalFn::$I => $iname, )* }
            }

            /// The function identity of this internal function.
            pub(crate) const fn func(self) -> Func {
                match self { $( InternalFn::$I => Func::$I, )* }
            }
        }

        /// One mathematical function, whatever its representation in the
        /// arena (see the [module documentation](self)).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(crate) enum Func {
            $( $(#[$ndoc])* $N, )*
            $( $(#[$ldoc])* $L, )*
            $( $(#[$idoc])* $I, )*
        }

        impl Func {
            /// Every function: the node-backed ones in `ExprNode` order, the
            /// library functions in [`LibFn::ALL`] order, then the internal
            /// ones.
            #[cfg_attr(
                not(test),
                expect(dead_code, reason = "iterated by the tests; by the table consumers in W6")
            )]
            pub(crate) const ALL: &[Func] = &[
                $( Func::$N, )* $( Func::$L, )* $( Func::$I, )*
            ];

            /// The specs, indexed by discriminant (the order of `ALL`).
            const SPECS: &[FuncSpec] = &[
                $( FuncSpec { name: $nname, $( $nfield: $nvalue ),+ }, )*
                $( FuncSpec { name: $lname, $( $lfield: $lvalue ),+ }, )*
                $( FuncSpec { name: $iname, $( $ifield: $ivalue ),+ }, )*
            ];

            /// The function's data.
            pub(crate) const fn spec(self) -> &'static FuncSpec {
                &Self::SPECS[self as usize]
            }

            /// The representation of the function's nodes.
            pub(crate) const fn family(self) -> Family {
                match self {
                    $( Func::$N => Family::Node, )*
                    $( Func::$L => Family::Lib, )*
                    $( Func::$I => Family::Internal, )*
                }
            }

            /// The public library function, for the 50 that are one.
            pub(crate) const fn as_lib(self) -> Option<LibFn> {
                match self {
                    $( Func::$N => None, )*
                    $( Func::$L => Some(LibFn::$L), )*
                    $( Func::$I => None, )*
                }
            }

            /// The internal function, for the 8 that are one.
            pub(crate) const fn as_internal(self) -> Option<InternalFn> {
                match self {
                    $( Func::$N => None, )*
                    $( Func::$L => None, )*
                    $( Func::$I => Some(InternalFn::$I), )*
                }
            }

            /// Is `s` exactly the head of an internal function?
            fn is_internal_head(s: &str) -> bool {
                matches!(s, $( $iname )|*)
            }
        }

        /// The pattern matching every function that is not node-backed.
        macro_rules! non_node_funcs {
            () => { $( Func::$L )|* | $( Func::$I )|* };
        }
    };
}

impl From<LibFn> for Func {
    fn from(f: LibFn) -> Func {
        f.func()
    }
}

impl From<InternalFn> for Func {
    fn from(f: InternalFn) -> Func {
        f.func()
    }
}

use ArgCount::{AtLeast, Between, Exactly};
use AutoSign::{Canonical, Coefficient, Negation, SymbolFree};
use Parity::{Even, Neither, Odd};
use SortSub::{Function, Named, Rank};
use TexPiece::{Arg, Group, Lit};

/// The cut `(−∞, 0]` of the first argument.
const NEG0: Cut = Cut {
    arg: 0,
    line: CutLine::NonPositiveReals,
};
/// The cut `(−∞, 0]` of the second argument.
const NEG1: Cut = Cut {
    arg: 1,
    line: CutLine::NonPositiveReals,
};

funcs! {
    node {
        /// Minimum `min(a, b, …)`.
        Min => FuncSpec {
            name: "min", parse: &[], arity: AtLeast(1), sort: Rank(130), tex: Tex::Call(r"\min"),
            // Of real arguments only (the order of ℝ).
            conj: Conj::No, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Maximum `max(a, b, …)`.
        Max => FuncSpec {
            name: "max", parse: &[], arity: AtLeast(1), sort: Rank(132), tex: Tex::Call(r"\max"),
            conj: Conj::No, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Floor `⌊x⌋`.
        Floor => FuncSpec {
            name: "floor", parse: &[], arity: Exactly(1), sort: Function(20),
            tex: Tex::Layout(&[Lit(r"\lfloor "), Arg(0), Lit(r"\rfloor")]),
            // SymPy's `floor(x + iy) = floor(x) + i·floor(y)` does not reflect:
            // `floor(−y) ≠ −floor(y)`.
            conj: Conj::No, real_on: RealOn::IntegerValued, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Ceiling `⌈x⌉`.
        Ceiling => FuncSpec {
            name: "ceiling", parse: &["ceil"], arity: Exactly(1), sort: Function(21),
            tex: Tex::Layout(&[Lit(r"\lceil "), Arg(0), Lit(r"\rceil")]),
            conj: Conj::No, real_on: RealOn::IntegerValued, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Sine.
        Sin => FuncSpec {
            name: "sin", parse: &[], arity: Exactly(1), sort: Function(0), tex: Tex::Call(r"\sin"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Odd(Negation), poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Cosine.
        Cos => FuncSpec {
            name: "cos", parse: &[], arity: Exactly(1), sort: Function(1), tex: Tex::Call(r"\cos"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Even(Negation), poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Tangent.
        Tan => FuncSpec {
            name: "tan", parse: &[], arity: Exactly(1), sort: Function(2), tex: Tex::Call(r"\tan"),
            // Poles at π/2 + kπ; real and finite for a real argument by the
            // generic convention of its own handler (`compute_trig`).
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Odd(Negation), poles: Poles::Other,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Exponential `eˣ`.
        Exp => FuncSpec {
            name: "exp", parse: &[], arity: Exactly(1), sort: Function(3), tex: Tex::Call(r"\exp"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Natural logarithm (`log(x)` with one argument).
        Ln => FuncSpec {
            name: "ln", parse: &["log"], arity: Exactly(1), sort: Function(4), tex: Tex::Call(r"\ln"),
            conj: Conj::OffCut(NEG0), real_on: RealOn::PositiveReals, parity: Neither,
            poles: Poles::Zero(0),
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Absolute value (complex modulus).
        Abs => FuncSpec {
            name: "abs", parse: &[], arity: Exactly(1), sort: Function(6),
            tex: Tex::Layout(&[Lit(r"\left|"), Arg(0), Lit(r"\right|")]),
            conj: Conj::Entire, real_on: RealOn::Everywhere, parity: Even(Canonical),
            poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Inverse sine.
        Asin => FuncSpec {
            name: "asin", parse: &["arcsin"], arity: Exactly(1), sort: Function(8),
            tex: Tex::Call(r"\arcsin"),
            conj: Conj::OffCut(Cut { arg: 0, line: CutLine::OutsideUnitInterval }),
            real_on: RealOn::UnitInterval { closed: true }, parity: Odd(SymbolFree),
            poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Inverse cosine.
        Acos => FuncSpec {
            name: "acos", parse: &["arccos"], arity: Exactly(1), sort: Function(9),
            tex: Tex::Call(r"\arccos"),
            conj: Conj::OffCut(Cut { arg: 0, line: CutLine::OutsideUnitInterval }),
            real_on: RealOn::UnitInterval { closed: true }, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Inverse tangent.
        Atan => FuncSpec {
            name: "atan", parse: &["arctan"], arity: Exactly(1), sort: Function(10),
            tex: Tex::Call(r"\arctan"),
            // Poles at ±i.
            conj: Conj::OffCut(Cut { arg: 0, line: CutLine::ImaginaryOutsideUnit }),
            real_on: RealOn::Reals, parity: Odd(SymbolFree), poles: Poles::Other,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Two-argument arctangent `atan2(y, x)`.
        Atan2 => FuncSpec {
            name: "atan2", parse: &[], arity: Exactly(2), sort: Function(19),
            tex: Tex::Call(r"\operatorname{atan2}"),
            // Of real arguments only.
            conj: Conj::No, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Hyperbolic sine.
        Sinh => FuncSpec {
            name: "sinh", parse: &[], arity: Exactly(1), sort: Function(11), tex: Tex::Call(r"\sinh"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Odd(Negation), poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Hyperbolic cosine.
        Cosh => FuncSpec {
            name: "cosh", parse: &[], arity: Exactly(1), sort: Function(12), tex: Tex::Call(r"\cosh"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Even(Negation), poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Hyperbolic tangent.
        Tanh => FuncSpec {
            name: "tanh", parse: &[], arity: Exactly(1), sort: Function(13), tex: Tex::Call(r"\tanh"),
            // Poles at i(π/2 + kπ).
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Odd(Negation), poles: Poles::Other,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Inverse hyperbolic sine.
        Asinh => FuncSpec {
            name: "asinh", parse: &["arcsinh"], arity: Exactly(1), sort: Function(14),
            tex: Tex::Call(r"\operatorname{asinh}"),
            conj: Conj::OffCut(Cut { arg: 0, line: CutLine::ImaginaryOutsideUnit }),
            real_on: RealOn::Reals, parity: Odd(SymbolFree), poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Inverse hyperbolic cosine.
        Acosh => FuncSpec {
            name: "acosh", parse: &["arccosh"], arity: Exactly(1), sort: Function(15),
            tex: Tex::Call(r"\operatorname{acosh}"),
            conj: Conj::OffCut(Cut { arg: 0, line: CutLine::BelowOne }),
            real_on: RealOn::AtLeastOne, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Inverse hyperbolic tangent.
        Atanh => FuncSpec {
            name: "atanh", parse: &["arctanh"], arity: Exactly(1), sort: Function(16),
            tex: Tex::Call(r"\operatorname{atanh}"),
            // Poles at ±1.
            conj: Conj::OffCut(Cut { arg: 0, line: CutLine::OutsideUnitInterval }),
            real_on: RealOn::UnitInterval { closed: false }, parity: Odd(SymbolFree),
            poles: Poles::Other,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Sign function.
        Sign => FuncSpec {
            name: "sign", parse: &["sgn"], arity: Exactly(1), sort: Function(18),
            tex: Tex::Call(r"\operatorname{sgn}"),
            // `sign z = z/|z|` (SymPy's `sign._eval_conjugate`).
            conj: Conj::Entire, real_on: RealOn::IntegerValued, parity: Odd(Canonical),
            poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Heaviside step `H(x)` (the parser reads `H` only as written).
        Heaviside => FuncSpec {
            name: "H", parse: &["heaviside"], arity: Exactly(1), sort: Function(28),
            tex: Tex::Call(r"\operatorname{H}"),
            // Of real arguments.
            conj: Conj::No, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Dirac delta `δ(x)`.
        DiracDelta => FuncSpec {
            name: "DiracDelta", parse: &["dirac_delta"], arity: Exactly(1), sort: Function(29),
            tex: Tex::Call(r"\delta"),
            // A distribution: 0 off the origin, no value at it.
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Even(AutoSign::Never),
            poles: Poles::Other,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Real part.
        Re => FuncSpec {
            name: "re", parse: &[], arity: Exactly(1), sort: Function(31), tex: Tex::Call(r"\Re"),
            conj: Conj::Entire, real_on: RealOn::Everywhere, parity: Odd(Canonical),
            poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Imaginary part.
        Im => FuncSpec {
            name: "im", parse: &[], arity: Exactly(1), sort: Function(32), tex: Tex::Call(r"\Im"),
            // `im z̄ = −im z`.
            conj: Conj::No, real_on: RealOn::Everywhere, parity: Odd(Canonical), poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Complex conjugate.
        Conjugate => FuncSpec {
            name: "conjugate", parse: &["conj"], arity: Exactly(1), sort: Function(33),
            tex: Tex::Layout(&[Lit(r"\overline{"), Arg(0), Lit("}")]),
            conj: Conj::Entire, real_on: RealOn::Custom, parity: Odd(Canonical), poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::NONE,
        };
        /// Principal argument `arg(z) ∈ (−π, π]`.
        Arg => FuncSpec {
            name: "arg", parse: &[], arity: Exactly(1), sort: Function(34), tex: Tex::Call(r"\arg"),
            // `arg z̄ = −arg z` off the cut.
            conj: Conj::No, real_on: RealOn::Everywhere, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::NONE,
        };
        /// Gamma function.
        Gamma => FuncSpec {
            name: "Gamma", parse: &[], arity: Exactly(1), sort: Function(22), tex: Tex::Call(r"\Gamma"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither,
            poles: Poles::NonPositiveIntegers(&[0]),
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Log-gamma function.
        LogGamma => FuncSpec {
            name: "LogGamma", parse: &[], arity: Exactly(1), sort: Function(23),
            tex: Tex::Call(r"\ln \Gamma"),
            conj: Conj::OffCut(NEG0), real_on: RealOn::PositiveReals, parity: Neither,
            poles: Poles::NonPositiveIntegers(&[0]),
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Digamma `ψ(x)`.
        Digamma => FuncSpec {
            name: "Digamma", parse: &[], arity: Exactly(1), sort: Function(24), tex: Tex::Call(r"\psi"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither,
            poles: Poles::NonPositiveIntegers(&[0]),
            constants: Constants::Folded,
            rt: Runtime::own(Backends::VM_RUST_C).scipy("scipy.special.psi(x)"),
        };
        /// Error function.
        Erf => FuncSpec {
            name: "erf", parse: &[], arity: Exactly(1), sort: Function(25),
            tex: Tex::Call(r"\operatorname{erf}"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Odd(SymbolFree), poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Complementary error function.
        Erfc => FuncSpec {
            name: "erfc", parse: &[], arity: Exactly(1), sort: Function(26),
            tex: Tex::Call(r"\operatorname{erfc}"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Principal branch `W₀(x)` of Lambert W (the node `ExprNode::LambertW`;
        /// the other branches are [`Func::LambertW`]).
        LambertW0 => FuncSpec {
            name: "W", parse: &["lambertw"], arity: Exactly(1), sort: Function(30),
            tex: Tex::Call(r"\operatorname{W}"),
            // `W₀(z̄) = conj W₀(z)` off the cut `(−∞, −1/e]`, which no consumer
            // tests; real on `[−1/e, ∞)` (DLMF §4.13), claimed on `[0, ∞)`.
            conj: Conj::No, real_on: RealOn::NonNegativeReals, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::VM_RUST_C),
        };
        /// Beta function `B(a, b)`.
        Beta => FuncSpec {
            name: "B", parse: &["beta"], arity: Exactly(2), sort: Function(27),
            tex: Tex::Call(r"\mathrm{B}"),
            // `Γ(a)Γ(b)/Γ(a + b)`; real for real arguments as SymPy's `beta`.
            // Zero where `a + b` is a pole and `a`, `b` are not: `B(1/2, −1/2)`
            // stays unfolded.
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither,
            poles: Poles::NonPositiveIntegers(&[0, 1]),
            constants: Constants::Evaluated,
            rt: Runtime::own(Backends::VM_RUST_C).scipy("scipy.special.beta(a, b)"),
        };
        /// Sine integral.
        Si => FuncSpec {
            name: "Si", parse: &[], arity: Exactly(1), sort: Function(35),
            tex: Tex::Call(r"\operatorname{Si}"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Odd(Canonical), poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::helper(Kernel::Unary("sin_integral"), Backends::VM_RUST_C)
                .scipy("scipy.special.sici(x)[0]"),
        };
        /// Cosine integral.
        Ci => FuncSpec {
            name: "Ci", parse: &[], arity: Exactly(1), sort: Function(36),
            tex: Tex::Call(r"\operatorname{Ci}"),
            // `Ci(−x) = Ci(x) + iπ` for `x > 0` (DLMF §6.4).
            conj: Conj::OffCut(NEG0), real_on: RealOn::PositiveReals, parity: Neither,
            poles: Poles::Zero(0),
            constants: Constants::Folded,
            rt: Runtime::helper(Kernel::Unary("cos_integral"), Backends::VM_RUST_C)
                .scipy("scipy.special.sici(x)[1]"),
        };
        /// Exponential integral.
        Ei => FuncSpec {
            name: "Ei", parse: &[], arity: Exactly(1), sort: Function(37),
            tex: Tex::Call(r"\operatorname{Ei}"),
            // Real on the whole real line (the principal value on the cut,
            // where the reflection holds too; tested off it only).
            conj: Conj::OffCut(NEG0), real_on: RealOn::Reals, parity: Neither, poles: Poles::Zero(0),
            constants: Constants::Folded,
            rt: Runtime::helper(Kernel::Unary("exp_integral_ei"), Backends::VM_RUST_C)
                .scipy("scipy.special.expi(x)"),
        };
        /// Logarithmic integral.
        Li => FuncSpec {
            name: "li", parse: &[], arity: Exactly(1), sort: Function(38),
            tex: Tex::Call(r"\operatorname{li}"),
            // `li z = Ei(ln z)` is cut along `(−∞, 0]` only (real on `(0, 1)`);
            // `(−∞, 1]` is the sufficient test `evalf` has used since 0.45.
            conj: Conj::OffCut(Cut { arg: 0, line: CutLine::BelowOne }),
            real_on: RealOn::PositiveReals, parity: Neither, poles: Poles::One(0),
            constants: Constants::Folded,
            rt: Runtime::helper(Kernel::Unary("log_integral"), Backends::VM_RUST_C),
        };
        /// Riemann zeta.
        Zeta => FuncSpec {
            name: "zeta", parse: &[], arity: Exactly(1), sort: Function(39), tex: Tex::Call(r"\zeta"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither, poles: Poles::One(0),
            constants: Constants::Folded,
            rt: Runtime::NONE.scipy("scipy.special.zeta(s)"),
        };
        /// Polygamma `ψ⁽ⁿ⁾(x)`, `polygamma(n, x)`.
        Polygamma => FuncSpec {
            name: "polygamma", parse: &[], arity: Exactly(2), sort: Function(40),
            tex: Tex::Layout(&[Lit(r"\psi^{("), Arg(0), Lit(r")}\left("), Arg(1), Lit(r"\right)")]),
            // Meromorphic in `x` for an order `m ≥ 0`; for other real orders
            // (`ζ(m + 1, x)`-like, real for `x > 0` in mpmath's and SymPy's
            // definitions) cut along `x ≤ 0`.
            conj: Conj::IntegerOrderOffCut { cut: NEG1, nonnegative: true },
            real_on: RealOn::PositiveReals, parity: Neither,
            poles: Poles::NonPositiveIntegers(&[1]),
            constants: Constants::Folded,
            rt: Runtime::NONE.scipy("scipy.special.polygamma(n, x)"),
        };
        /// Kronecker delta `δᵢⱼ`.
        KroneckerDelta => FuncSpec {
            name: "KroneckerDelta", parse: &["kronecker_delta"], arity: Exactly(2),
            sort: Function(41),
            tex: Tex::Layout(&[Lit(r"\delta_{"), Arg(0), Lit(" "), Arg(1), Lit("}")]),
            // 0 or 1: `δ(ī, j̄) = δ(i, j)`.
            conj: Conj::Entire, real_on: RealOn::Custom, parity: Neither, poles: Poles::None,
            constants: Constants::Folded,
            rt: Runtime::NONE,
        };
        /// Factorial `n!` (printed postfix).
        Factorial => FuncSpec {
            name: "factorial", parse: &[], arity: Exactly(1), sort: Function(7),
            tex: Tex::Postfix("!"),
            // `Γ(x + 1)`: poles at the negative integers, so claimed real on
            // `[0, ∞)` only (SymPy: real for a non-negative or non-integer `x`).
            conj: Conj::Entire, real_on: RealOn::NonNegativeReals, parity: Neither,
            poles: Poles::Other,
            constants: Constants::Folded,
            rt: Runtime::own(Backends::ALL),
        };
        /// Binomial coefficient `C(n, k)`.  Its sort byte 8 is `Asin`'s too.
        Binomial => FuncSpec {
            name: "C", parse: &["binomial"], arity: Exactly(2), sort: Function(8),
            tex: Tex::Layout(&[Lit(r"\binom{"), Arg(0), Lit("}{"), Arg(1), Lit("}")]),
            // `Γ(n + 1)/(Γ(k + 1)Γ(n − k + 1))`, meromorphic; an integer for
            // non-negative integers.
            conj: Conj::Entire, real_on: RealOn::NonNegativeIntegers, parity: Neither,
            poles: Poles::Other,
            constants: Constants::Folded,
            // Python spells it with the GKP convention (`scipy.special.binom`
            // is NaN at `C(−2, −4)` and `C(−3, 2)`, 0 and 6 here).
            rt: Runtime::own(Backends::ALL),
        };
    }
    lib {
        // ── Integer sequences and combinatorial counts ────────────────────────
        /// Double factorial `n!!`.
        Factorial2 => FuncSpec {
            name: "factorial2", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Postfix("!!"),
            // Integers (SymPy's extension to ℂ is not evaluated here); `(−2k)!!`
            // is infinite.
            conj: Conj::No, real_on: RealOn::NonNegativeIntegers, parity: Neither,
            poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("factorial2"), Backends::VM_RUST_C)
                .scipy("scipy.special.factorial2(n)"),
        };
        /// Subfactorial (derangement count) `!n`.
        Subfactorial => FuncSpec {
            name: "subfactorial", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Layout(&[Lit("!"), Group(0)]),
            // SymPy's extension `Γ(n + 1, −1)/e` sits on the cut of `Γ(a, ·)`.
            conj: Conj::No, real_on: RealOn::NonNegativeIntegers, parity: Neither,
            poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// Rising factorial (Pochhammer symbol) `(x)_n = x (x+1) ⋯ (x+n−1)`.
        RisingFactorial => FuncSpec {
            name: "rising_factorial", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit("{"), Group(0), Lit(r"}^{\left("), Arg(1), Lit(r"\right)}")]),
            // `Γ(x + n)/Γ(x)` (DLMF §5.2(iii)): a polynomial in `x` for an
            // integer `n ≥ 0`.
            conj: Conj::Entire, real_on: RealOn::Polynomial { degree: 1 }, parity: Neither,
            poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Binary("rising_factorial"), Backends::VM_RUST_C)
                .scipy("scipy.special.poch(x, n)"),
        };
        /// Falling factorial `x^(n) = x (x−1) ⋯ (x−n+1)`.
        FallingFactorial => FuncSpec {
            name: "falling_factorial", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit(r"{\left("), Arg(0), Lit(r"\right)}_{"), Arg(1), Lit("}")]),
            // `Γ(x + 1)/Γ(x − n + 1)`.
            conj: Conj::Entire, real_on: RealOn::Polynomial { degree: 1 }, parity: Neither,
            poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Binary("falling_factorial"), Backends::VM_RUST_C)
                .scipy("scipy.special.poch(x - n + 1, n)"),
        };
        /// Fibonacci number `F_n`.
        Fibonacci => FuncSpec {
            name: "fibonacci", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Layout(&[Lit("F_{"), Arg(0), Lit("}")]),
            // Binet's `(φˣ − cos(πx)φ⁻ˣ)/√5` at real `x` (`evalf`).
            conj: Conj::No, real_on: RealOn::RealLine, parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("fibonacci"), Backends::VM_RUST_C),
        };
        /// Lucas number `L_n`.
        Lucas => FuncSpec {
            name: "lucas", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Layout(&[Lit("L_{"), Arg(0), Lit("}")]),
            conj: Conj::No, real_on: RealOn::RealLine, parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("lucas"), Backends::VM_RUST_C),
        };
        /// Bernoulli number `B_n`.
        Bernoulli => FuncSpec {
            name: "bernoulli", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Layout(&[Lit("B_{"), Arg(0), Lit("}")]),
            // `−s·ζ(1 − s)` at real `s` (`evalf`), entire and real on ℝ.  The
            // exact `bernoulli(1)` is its value `+1/2` (SymPy ≥ 1.12; until
            // 0.49 it was `−1/2`, a jump of the continuous extension, which
            // is why the claim stopped at the integers:
            // `limₓ→₀ li(bernoulli(x + 1))` came out `li(−1/2)`).  The claim
            // is kept at the non-negative integers; `RealLine` would now be
            // correct too.
            conj: Conj::No, real_on: RealOn::NonNegativeIntegers, parity: Neither,
            poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// Harmonic number `H_n`.
        Harmonic => FuncSpec {
            name: "harmonic", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Layout(&[Lit("H_{"), Arg(0), Lit("}")]),
            // `ψ(x + 1) + γ`: poles at the negative integers.
            conj: Conj::No, real_on: RealOn::NonNegativeReals, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("harmonic"), Backends::VM_RUST_C),
        };
        /// Catalan number `C_n`.
        Catalan => FuncSpec {
            name: "catalan", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Layout(&[Lit("C_{"), Arg(0), Lit("}")]),
            // `Γ(2x + 1)/(Γ(x + 1)Γ(x + 2))`: poles at `x = −1/2, −3/2, …`.
            conj: Conj::No, real_on: RealOn::NonNegativeReals, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// Bell number `B_n`.
        Bell => FuncSpec {
            name: "bell", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Layout(&[Lit("B_{"), Arg(0), Lit("}")]),
            conj: Conj::No, real_on: RealOn::NonNegativeIntegers, parity: Neither,
            poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// Euler number `E_n`.
        EulerNumber => FuncSpec {
            name: "euler_number", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Layout(&[Lit("E_{"), Arg(0), Lit("}")]),
            // By Hurwitz zeta functions at real `s` (`evalf`), entire.
            conj: Conj::No, real_on: RealOn::RealLine, parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// Signed Stirling number of the first kind `s(n, k)`.
        Stirling1 => FuncSpec {
            name: "stirling1", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit(r"\left[{"), Arg(0), Lit(r" \atop "), Arg(1), Lit(r"}\right]")]),
            conj: Conj::No, real_on: RealOn::NonNegativeIntegers, parity: Neither,
            poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// Stirling number of the second kind `S(n, k)`.
        Stirling2 => FuncSpec {
            name: "stirling2", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit(r"\left\{{"), Arg(0), Lit(r" \atop "), Arg(1), Lit(r"}\right\}")]),
            conj: Conj::No, real_on: RealOn::NonNegativeIntegers, parity: Neither,
            poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.stirling2(n, k)"),
        };
        /// Integer partition count `p(n)`.
        PartitionCount => FuncSpec {
            name: "partition_count", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("p"),
            conj: Conj::No, real_on: RealOn::NonNegativeIntegers, parity: Neither,
            poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// Lambert W, `lambertw(x, k)`: the branch `W_k` (`k` an integer) of the
        /// inverse of `w·eʷ`, as SymPy's `LambertW(x, k)`.  The principal branch
        /// `k = 0` is the dedicated `ExprNode::LambertW(x)` node, which
        /// `Arena::lambertw_branch` and `eval` build for `lambertw(x)` and
        /// `lambertw(x, 0)`; a `Lib` node carries only the other branches.
        LambertW => FuncSpec {
            name: "lambertw", parse: &["w"], arity: Between { min: 1, max: 2 }, sort: Named,
            tex: Tex::Layout(&[
                Lit(r"\operatorname{W}_{"), Arg(1), Lit(r"}\left("), Arg(0), Lit(r"\right)"),
            ]),
            // `W_k(z̄) = conj W_{−k}(z)`, another branch; `W_{−1}` is real on
            // `[−1/e, 0)`; `W_k(0) = −∞` for `k ≠ 0` (DLMF §4.13).
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Branches(&["lambert_w0", "lambert_wm1"]), Backends::VM_RUST_C),
        };

        // ── Bessel functions `(order, x)` ─────────────────────────────────────
        /// Bessel function of the first kind `J_ν(x)`.
        BesselJ => FuncSpec {
            name: "besselj", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit("J_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)")]),
            // Entire in `x` for an integer order, else `(x/2)^ν·(entire)`
            // (DLMF §10.2(ii)); infinite at 0 for some orders.
            conj: Conj::IntegerOrderOffCut { cut: NEG1, nonnegative: false },
            real_on: RealOn::BesselOrder, parity: Neither, poles: Poles::Zero(1),
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Ordered("bessel_j"), Backends::VM_RUST_C)
                .scipy("scipy.special.jv(order, x)"),
        };
        /// Bessel function of the second kind `Y_ν(x)`.
        BesselY => FuncSpec {
            name: "bessely", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit("Y_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)")]),
            conj: Conj::OffCut(NEG1), real_on: RealOn::PositiveReals, parity: Neither,
            poles: Poles::Zero(1),
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Ordered("bessel_y"), Backends::VM_RUST_C)
                .scipy("scipy.special.yv(order, x)"),
        };
        /// Modified Bessel function of the first kind `I_ν(x)`.
        BesselI => FuncSpec {
            name: "besseli", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit("I_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)")]),
            conj: Conj::IntegerOrderOffCut { cut: NEG1, nonnegative: false },
            real_on: RealOn::BesselOrder, parity: Neither, poles: Poles::Zero(1),
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Ordered("bessel_i"), Backends::VM_RUST_C)
                .scipy("scipy.special.iv(order, x)"),
        };
        /// Modified Bessel function of the second kind `K_ν(x)`.
        BesselK => FuncSpec {
            name: "besselk", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit("K_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)")]),
            conj: Conj::OffCut(NEG1), real_on: RealOn::PositiveReals, parity: Neither,
            poles: Poles::Zero(1),
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Ordered("bessel_k"), Backends::VM_RUST_C)
                .scipy("scipy.special.kv(order, x)"),
        };

        // ── Classical orthogonal polynomials `(n, x)` ─────────────────────────
        /// Legendre polynomial `P_n(x)`.
        Legendre => FuncSpec {
            name: "legendre", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit("P_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)")]),
            // A polynomial for an integer degree `n ≥ 0`; `₂F₁(−ν, ν + 1; 1;
            // (1 − x)/2)`, infinite at `x = −1`, otherwise.
            conj: Conj::Polynomial { degree: 0 }, real_on: RealOn::Polynomial { degree: 0 },
            parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Ordered("legendre_p"), Backends::VM_RUST_C)
                .scipy("scipy.special.eval_legendre(n, x)"),
        };
        /// Chebyshev polynomial of the first kind `T_n(x)`.
        ChebyshevT => FuncSpec {
            name: "chebyshev_t", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit("T_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)")]),
            conj: Conj::Polynomial { degree: 0 }, real_on: RealOn::Polynomial { degree: 0 },
            parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Ordered("chebyshev_t"), Backends::VM_RUST_C)
                .scipy("scipy.special.eval_chebyt(n, x)"),
        };
        /// Chebyshev polynomial of the second kind `U_n(x)`.
        ChebyshevU => FuncSpec {
            name: "chebyshev_u", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit("U_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)")]),
            conj: Conj::Polynomial { degree: 0 }, real_on: RealOn::Polynomial { degree: 0 },
            parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Ordered("chebyshev_u"), Backends::VM_RUST_C)
                .scipy("scipy.special.eval_chebyu(n, x)"),
        };
        /// Physicists' Hermite polynomial `H_n(x)`.
        Hermite => FuncSpec {
            name: "hermite", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit("H_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)")]),
            // Entire in `x` for every degree (DLMF §12.7, §18.5).
            conj: Conj::Polynomial { degree: 0 }, real_on: RealOn::Polynomial { degree: 0 },
            parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Ordered("hermite_h"), Backends::VM_RUST_C)
                .scipy("scipy.special.eval_hermite(n, x)"),
        };
        /// Laguerre polynomial `L_n(x)`.
        Laguerre => FuncSpec {
            name: "laguerre", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit("L_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)")]),
            // `₁F₁(−ν; 1; x)`, entire in `x` for every degree.
            conj: Conj::Polynomial { degree: 0 }, real_on: RealOn::Polynomial { degree: 0 },
            parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Ordered("laguerre_l"), Backends::VM_RUST_C)
                .scipy("scipy.special.eval_laguerre(n, x)"),
        };

        // ── Error-function family and integrals (0.9) ─────────────────────────
        /// Imaginary error function `erfi(x)`.
        Erfi => FuncSpec {
            name: "erfi", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{erfi}"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Odd(Coefficient),
            poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("erfi"), Backends::VM_RUST_C)
                .scipy("scipy.special.erfi(x)"),
        };
        /// Inverse error function `erf⁻¹(y)`.
        ErfInv => FuncSpec {
            name: "erfinv", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{erf}^{-1}"),
            // Defined on `(−1, 1)` here (the continuation of the inverse of
            // `erf` is many-valued); `±∞` at `±1`.
            conj: Conj::No, real_on: RealOn::UnitInterval { closed: false },
            parity: Odd(Coefficient), poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("erfinv"), Backends::VM_RUST_C)
                .scipy("scipy.special.erfinv(y)"),
        };
        /// Inverse complementary error function `erfc⁻¹(y)`.
        ErfcInv => FuncSpec {
            name: "erfcinv", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{erfc}^{-1}"),
            // Real on `(0, 2)`, which the assumption system cannot express;
            // `±∞` at 0 and 2.
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("erfcinv"), Backends::VM_RUST_C)
                .scipy("scipy.special.erfcinv(y)"),
        };
        /// Generalised exponential integral `E_n(x)`, as `expint(n, x)`.
        ExpInt => FuncSpec {
            name: "expint", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[
                Lit(r"\operatorname{E}_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)"),
            ]),
            // `E_n(0)` is infinite for `n ≤ 1` (DLMF §8.19).
            conj: Conj::OffCut(NEG1), real_on: RealOn::PositiveReals, parity: Neither,
            poles: Poles::Zero(1),
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.expn(n, x)"),
        };
        /// Hyperbolic sine integral `Shi(x)`.
        Shi => FuncSpec {
            name: "Shi", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Shi}"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Odd(Coefficient),
            poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("sinh_integral"), Backends::VM_RUST_C)
                .scipy("scipy.special.shichi(x)[0]"),
        };
        /// Hyperbolic cosine integral `Chi(x)`.
        Chi => FuncSpec {
            name: "Chi", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Chi}"),
            // `Chi(−x) = Chi(x) + iπ` for `x > 0` (DLMF §6.2).
            conj: Conj::OffCut(NEG0), real_on: RealOn::PositiveReals, parity: Neither,
            poles: Poles::Zero(0),
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("cosh_integral"), Backends::VM_RUST_C)
                .scipy("scipy.special.shichi(x)[1]"),
        };
        /// Fresnel sine integral `S(x)`.
        FresnelS => FuncSpec {
            name: "fresnels", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("S"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Odd(Coefficient),
            poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("fresnel_s"), Backends::VM_RUST_C)
                .scipy("scipy.special.fresnel(x)[0]"),
        };
        /// Fresnel cosine integral `C(x)`.
        FresnelC => FuncSpec {
            name: "fresnelc", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("C"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Odd(Coefficient),
            poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("fresnel_c"), Backends::VM_RUST_C)
                .scipy("scipy.special.fresnel(x)[1]"),
        };
        /// Lower incomplete gamma `γ(s, x)`.
        LowerGamma => FuncSpec {
            name: "lowergamma", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call(r"\gamma"),
            // `γ(s, x) = xˢ·(entire)`, poles at `s = 0, −1, …` (DLMF §8.2).
            conj: Conj::OffCut(NEG1), real_on: RealOn::PositiveReals, parity: Neither,
            poles: Poles::NonPositiveIntegers(&[0]),
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.gamma(s) * scipy.special.gammainc(s, x)"),
        };
        /// Upper incomplete gamma `Γ(s, x)`.
        UpperGamma => FuncSpec {
            name: "uppergamma", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call(r"\Gamma"),
            // Entire in `s` for `x ≠ 0`; `Γ(s, 0) = Γ(s)`.
            conj: Conj::OffCut(NEG1), real_on: RealOn::PositiveReals, parity: Neither,
            poles: Poles::Zero(1),
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.gamma(s) * scipy.special.gammaincc(s, x)"),
        };
        /// Polylogarithm `Li_s(z)`, as `polylog(s, z)`.
        PolyLog => FuncSpec {
            name: "polylog", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[
                Lit(r"\operatorname{Li}_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)"),
            ]),
            // Cut along `z ∈ [1, ∞)` (DLMF §25.12); `Li_s(1) = ζ(s)`.
            conj: Conj::OffCut(Cut { arg: 1, line: CutLine::AtLeastOne }),
            real_on: RealOn::BelowOne, parity: Neither, poles: Poles::One(1),
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// Dirichlet eta `η(s)`.
        DirichletEta => FuncSpec {
            name: "dirichlet_eta", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\eta"),
            // `(1 − 2^{1−s})·ζ(s)`, entire.
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };

        // ── Airy functions ────────────────────────────────────────────────────
        /// Airy function `Ai(x)`.
        AiryAi => FuncSpec {
            name: "airyai", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Ai}"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.airy(x)[0]"),
        };
        /// Airy function `Bi(x)`.
        AiryBi => FuncSpec {
            name: "airybi", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Bi}"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.airy(x)[2]"),
        };
        /// Derivative `Ai′(x)`.
        AiryAiPrime => FuncSpec {
            name: "airyaiprime", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Ai}^\prime"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.airy(x)[1]"),
        };
        /// Derivative `Bi′(x)`.
        AiryBiPrime => FuncSpec {
            name: "airybiprime", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Bi}^\prime"),
            conj: Conj::Entire, real_on: RealOn::Reals, parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.airy(x)[3]"),
        };

        // ── Elliptic integrals (parameter `m = k²`, as in SymPy) ──────────────
        /// Complete elliptic integral of the first kind `K(m)`.
        EllipticK => FuncSpec {
            name: "elliptic_k", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("K"),
            // Cut along `m ∈ [1, ∞)`, `K(1) = ∞` (DLMF §19.2(i)).
            conj: Conj::OffCut(Cut { arg: 0, line: CutLine::AtLeastOne }),
            real_on: RealOn::BelowOne, parity: Neither, poles: Poles::One(0),
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("elliptic_k"), Backends::VM_RUST_C)
                .scipy("scipy.special.ellipk(m)"),
        };
        /// Complete elliptic integral of the second kind `E(m)`.
        EllipticE => FuncSpec {
            name: "elliptic_e", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("E"),
            conj: Conj::OffCut(Cut { arg: 0, line: CutLine::AtLeastOne }),
            real_on: RealOn::BelowOne, parity: Neither, poles: Poles::None,
            constants: Constants::Evaluated,
            rt: Runtime::helper(Kernel::Unary("elliptic_e"), Backends::VM_RUST_C)
                .scipy("scipy.special.ellipe(m)"),
        };
        /// Incomplete elliptic integral of the first kind `F(φ | m)`.
        EllipticF => FuncSpec {
            name: "elliptic_f", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit(r"F\left("), Arg(0), Lit(r"\middle| "), Arg(1), Lit(r"\right)")]),
            // The cuts in `φ` are not tabulated; real for real `φ` and `m < 1`.
            conj: Conj::No, real_on: RealOn::BelowOne, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.ellipkinc(phi, m)"),
        };
        /// Complete elliptic integral of the third kind `Π(n | m)`.
        EllipticPi => FuncSpec {
            name: "elliptic_pi", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit(r"\Pi\left("), Arg(0), Lit(r"\middle| "), Arg(1), Lit(r"\right)")]),
            // Cut in both arguments; real for `n, m < 1` (two bounds).
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };

        // ── Orthogonal polynomials with parameters ────────────────────────────
        /// Gegenbauer polynomial `C_n^{(α)}(x)`, as `gegenbauer(n, α, x)`.
        Gegenbauer => FuncSpec {
            name: "gegenbauer", parse: &[], arity: Exactly(3), sort: Named,
            tex: Tex::Layout(&[
                Lit("C_{"), Arg(0), Lit(r"}^{\left("), Arg(1), Lit(r"\right)}\left("), Arg(2),
                Lit(r"\right)"),
            ]),
            // A polynomial in `α` and `x` for an integer `n ≥ 0` (DLMF §18.5);
            // not entire for `n < 0` (mpmath: a `₂F₁` cut at `x ≤ −1`).
            conj: Conj::Polynomial { degree: 0 }, real_on: RealOn::Polynomial { degree: 0 },
            parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.eval_gegenbauer(n, alpha, x)"),
        };
        /// Jacobi polynomial `P_n^{(α,β)}(x)`, as `jacobi(n, α, β, x)`.
        Jacobi => FuncSpec {
            name: "jacobi", parse: &[], arity: Exactly(4), sort: Named,
            tex: Tex::Layout(&[
                Lit("P_{"), Arg(0), Lit(r"}^{\left("), Arg(1), Lit(","), Arg(2),
                Lit(r"\right)}\left("), Arg(3), Lit(r"\right)"),
            ]),
            conj: Conj::Polynomial { degree: 0 }, real_on: RealOn::Polynomial { degree: 0 },
            parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.eval_jacobi(n, alpha, beta, x)"),
        };
        /// Associated Legendre function `P_n^m(x)`, as `assoc_legendre(n, m, x)`.
        AssocLegendre => FuncSpec {
            name: "assoc_legendre", parse: &[], arity: Exactly(3), sort: Named,
            tex: Tex::Layout(&[
                Lit("P_{"), Arg(0), Lit(r"}^{\left("), Arg(1), Lit(r"\right)}\left("), Arg(2),
                Lit(r"\right)"),
            ]),
            // The factor `(1 − x²)^{m/2}`: for an odd `m` real on `[−1, 1]` only.
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.lpmv(m, n, x)"),
        };
        /// Generalised Laguerre polynomial `L_n^{(α)}(x)`, as `assoc_laguerre(n, α, x)`.
        AssocLaguerre => FuncSpec {
            name: "assoc_laguerre", parse: &[], arity: Exactly(3), sort: Named,
            tex: Tex::Layout(&[
                Lit("L_{"), Arg(0), Lit(r"}^{\left("), Arg(1), Lit(r"\right)}\left("), Arg(2),
                Lit(r"\right)"),
            ]),
            conj: Conj::Polynomial { degree: 0 }, real_on: RealOn::Polynomial { degree: 0 },
            parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE.scipy("scipy.special.eval_genlaguerre(n, alpha, x)"),
        };

        // ── Generalised incomplete beta (0.12), SymPy order `(a, b, x1, x2)` ──
        /// Incomplete beta `B_{(x₁, x₂)}(a, b)`.
        BetaInc => FuncSpec {
            name: "betainc", parse: &[], arity: Exactly(4), sort: Named,
            tex: Tex::Layout(&[
                Lit(r"\operatorname{B}_{("), Arg(2), Lit(", "), Arg(3), Lit(r")}\left("), Arg(0),
                Lit(", "), Arg(1), Lit(r"\right)"),
            ]),
            // `∫ t^{a−1}(1 − t)^{b−1} dt` from `x₁` to `x₂`: real for `x₁, x₂ ∈ [0, 1]`.
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE
                .scipy("scipy.special.beta(a, b) * (scipy.special.betainc(a, b, x2) - scipy.special.betainc(a, b, x1))"),
        };
        /// Regularised incomplete beta `I_{(x₁, x₂)}(a, b)`.
        BetaIncRegularized => FuncSpec {
            name: "betainc_regularized", parse: &[], arity: Exactly(4), sort: Named,
            tex: Tex::Layout(&[
                Lit(r"\operatorname{I}_{("), Arg(2), Lit(", "), Arg(3), Lit(r")}\left("), Arg(0),
                Lit(", "), Arg(1), Lit(r"\right)"),
            ]),
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE
                .scipy("scipy.special.betainc(a, b, x2) - scipy.special.betainc(a, b, x1)"),
        };
    }
    internal {
        // Heads of Gruntz's asymptotic rewrite (`calculus/series.rs` reads
        // its `*_ASYMPTOTIC` names from here); each exists only inside a
        // limit computation, where the limit engine knows them: the table
        // claims nothing about them.
        /// `erfcx(z) = e^{z²} erfc(z)` for `z → +∞` (SymPy's `_erfs`).
        ErfcxAsym => FuncSpec {
            name: "__erfcx_asym", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{\_\_erfcx\_asym}"),
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// `e^{−z} Ei(z)` for `z → ±∞` (SymPy's `_eis`).
        EiAsym => FuncSpec {
            name: "__ei_asym", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{\_\_ei\_asym}"),
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// `√v·e^{−v} erfi(√v)`, a function of `v = z²`.
        Erfix2Asym => FuncSpec {
            name: "__erfix2_asym", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{\_\_erfix2\_asym}"),
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// `√v·e^{v} erfc(√v)`, a function of `v = z²`.
        Erfcx2Asym => FuncSpec {
            name: "__erfcx2_asym", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{\_\_erfcx2\_asym}"),
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// `√(2πz) e^{−z} Iν(z)`, arguments `(ν, z)`.
        BesselIAsym => FuncSpec {
            name: "__besseli_asym", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call(r"\operatorname{\_\_besseli\_asym}"),
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// `√(2z/π) e^{z} Kν(z)`, arguments `(ν, z)`.
        BesselKAsym => FuncSpec {
            name: "__besselk_asym", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call(r"\operatorname{\_\_besselk\_asym}"),
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// `z^{1−s} e^{z} Γ(s, z)`, arguments `(s, z)`.
        UpperGammaAsym => FuncSpec {
            name: "__uppergamma_asym", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call(r"\operatorname{\_\_uppergamma\_asym}"),
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
        /// `2^z (ζ(z) − 1)`.
        ZetaAsym => FuncSpec {
            name: "__zeta_asym", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{\_\_zeta\_asym}"),
            conj: Conj::No, real_on: RealOn::Unclaimed, parity: Neither, poles: Poles::Other,
            constants: Constants::Evaluated,
            rt: Runtime::NONE,
        };
    }
}

// ---------------------------------------------------------------------------
// The view and the constructor
// ---------------------------------------------------------------------------

impl ExprNode {
    /// The function this node applies, without its arguments: the head of
    /// [`ExprNode::func`] (`None` for a node that is not a function
    /// application and for a user function), for the code that only asks
    /// which function a node is (`node.func_head().is_some_and(pred)`).
    ///
    /// The match is exhaustive over `ExprNode`: a new variant must say here
    /// whether it is a function.
    pub(crate) const fn func_head(&self) -> Option<Func> {
        match self {
            ExprNode::Min(_) => Some(Func::Min),
            ExprNode::Max(_) => Some(Func::Max),
            ExprNode::Floor(_) => Some(Func::Floor),
            ExprNode::Ceiling(_) => Some(Func::Ceiling),
            ExprNode::Sin(_) => Some(Func::Sin),
            ExprNode::Cos(_) => Some(Func::Cos),
            ExprNode::Tan(_) => Some(Func::Tan),
            ExprNode::Exp(_) => Some(Func::Exp),
            ExprNode::Ln(_) => Some(Func::Ln),
            ExprNode::Abs(_) => Some(Func::Abs),
            ExprNode::Asin(_) => Some(Func::Asin),
            ExprNode::Acos(_) => Some(Func::Acos),
            ExprNode::Atan(_) => Some(Func::Atan),
            ExprNode::Atan2(..) => Some(Func::Atan2),
            ExprNode::Sinh(_) => Some(Func::Sinh),
            ExprNode::Cosh(_) => Some(Func::Cosh),
            ExprNode::Tanh(_) => Some(Func::Tanh),
            ExprNode::Asinh(_) => Some(Func::Asinh),
            ExprNode::Acosh(_) => Some(Func::Acosh),
            ExprNode::Atanh(_) => Some(Func::Atanh),
            ExprNode::Sign(_) => Some(Func::Sign),
            ExprNode::Heaviside(_) => Some(Func::Heaviside),
            ExprNode::DiracDelta(_) => Some(Func::DiracDelta),
            ExprNode::Re(_) => Some(Func::Re),
            ExprNode::Im(_) => Some(Func::Im),
            ExprNode::Conjugate(_) => Some(Func::Conjugate),
            ExprNode::Arg(_) => Some(Func::Arg),
            ExprNode::Gamma(_) => Some(Func::Gamma),
            ExprNode::LogGamma(_) => Some(Func::LogGamma),
            ExprNode::Digamma(_) => Some(Func::Digamma),
            ExprNode::Erf(_) => Some(Func::Erf),
            ExprNode::Erfc(_) => Some(Func::Erfc),
            ExprNode::LambertW(_) => Some(Func::LambertW0),
            ExprNode::Beta(..) => Some(Func::Beta),
            ExprNode::Si(_) => Some(Func::Si),
            ExprNode::Ci(_) => Some(Func::Ci),
            ExprNode::Ei(_) => Some(Func::Ei),
            ExprNode::Li(_) => Some(Func::Li),
            ExprNode::Zeta(_) => Some(Func::Zeta),
            ExprNode::Polygamma(..) => Some(Func::Polygamma),
            ExprNode::KroneckerDelta(..) => Some(Func::KroneckerDelta),
            ExprNode::Factorial(_) => Some(Func::Factorial),
            ExprNode::Binomial(..) => Some(Func::Binomial),
            ExprNode::Lib(lib, _) => Some(lib.func()),
            ExprNode::Internal(g, _) => Some(g.func()),
            ExprNode::Apply(..)
            | ExprNode::Num(_)
            | ExprNode::Symbol(_)
            | ExprNode::Pi
            | ExprNode::E
            | ExprNode::ImaginaryUnit
            | ExprNode::EulerGamma
            | ExprNode::Catalan
            | ExprNode::GoldenRatio
            | ExprNode::PhysicalConstant(..)
            | ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN
            | ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Pow(..)
            | ExprNode::Neg(_)
            | ExprNode::BoolTrue
            | ExprNode::BoolFalse
            | ExprNode::Gt(..)
            | ExprNode::Ge(..)
            | ExprNode::Eq_(..)
            | ExprNode::Ne(..)
            | ExprNode::And(_)
            | ExprNode::Or(_)
            | ExprNode::Not(_)
            | ExprNode::Piecewise(_)
            | ExprNode::Derivative(..)
            | ExprNode::Integral(..)
            | ExprNode::DefiniteIntegral(..)
            | ExprNode::Sum(..)
            | ExprNode::Product_(..)
            | ExprNode::Limit(..)
            | ExprNode::Series(..)
            | ExprNode::LaplaceTransform(..)
            | ExprNode::InverseLaplaceTransform(..)
            | ExprNode::Residue(..)
            | ExprNode::RootOf(..)
            | ExprNode::RootSum(..)
            | ExprNode::DSolve(..)
            | ExprNode::ConditionSet(..)
            | ExprNode::EmptySet
            | ExprNode::UniversalSet
            | ExprNode::Interval(..)
            | ExprNode::FiniteSet(_)
            | ExprNode::SetUnion(_)
            | ExprNode::SetIntersection(_)
            | ExprNode::SetComplement(..)
            | ExprNode::Subs(..)
            | ExprNode::ImageSet(..) => None,
        }
    }

    /// The argument of a one-argument function `f` for which `pred(f)`
    /// holds (`sin(a)` → `a` for a predicate selecting `sin`), else `None`.
    pub(crate) fn unary_arg_if(&self, pred: impl Fn(Func) -> bool) -> Option<ExprId> {
        let f = self.func_head()?;
        if !pred(f) {
            return None;
        }
        match self.children().as_slice() {
            &[a] => Some(a),
            _ => None,
        }
    }

    /// The function this node applies and its arguments, in node order;
    /// `None` for a node that is not a function application (a number, a
    /// sum, a relation, …) and for a user function (`Apply`, whatever its
    /// head).  The arguments are the node's children.
    ///
    /// [`Arena::func`] of the result builds this node again (for a
    /// canonical node).
    pub(crate) fn func(&self, _arena: &Arena) -> Option<(Func, SmallVec<[ExprId; 4]>)> {
        let f = self.func_head()?;
        let args: SmallVec<[ExprId; 4]> = self.children().into_iter().collect();
        // A library or internal node has an argument count its function
        // accepts (`Arena::func` checks it); the check keeps the view total.
        (f.family() == Family::Node || f.spec().arity.accepts(args.len())).then_some((f, args))
    }
}

impl Func {
    /// Is `name` the head of an internal function (`__ei_asym`, …)?  Such
    /// heads are reserved: [`Context::apply`], the parser and `from_json`
    /// refuse them, so that no user function prints like an internal one.
    /// (No name builds an internal function: it is a node of its own.)
    ///
    /// [`Context::apply`]: crate::api::context::Context::apply
    pub(crate) fn is_internal_name(name: &str) -> bool {
        Func::is_internal_head(name)
    }

    /// The family of an elementary transcendental function: `Some` for the
    /// 14 one-argument functions `sin`, `cos`, `tan`, their inverses, the
    /// hyperbolic functions and their inverses, `exp` and `ln`; `None` for
    /// every other function.
    ///
    /// Code that treats these uniformly asks this instead of listing the
    /// variants (until 0.49 some 20 hand-written or-patterns did); the match
    /// is exhaustive, so a new function must say whether it is one.
    pub(crate) const fn elementary(self) -> Option<Elementary> {
        match self {
            Func::Sin | Func::Cos | Func::Tan => Some(Elementary::Trig),
            Func::Asin | Func::Acos | Func::Atan => Some(Elementary::InverseTrig),
            Func::Sinh | Func::Cosh | Func::Tanh => Some(Elementary::Hyperbolic),
            Func::Asinh | Func::Acosh | Func::Atanh => Some(Elementary::InverseHyperbolic),
            Func::Exp => Some(Elementary::Exp),
            Func::Ln => Some(Elementary::Ln),
            Func::Min
            | Func::Max
            | Func::Floor
            | Func::Ceiling
            | Func::Abs
            | Func::Atan2
            | Func::Sign
            | Func::Heaviside
            | Func::DiracDelta
            | Func::Re
            | Func::Im
            | Func::Conjugate
            | Func::Arg
            | Func::Gamma
            | Func::LogGamma
            | Func::Digamma
            | Func::Erf
            | Func::Erfc
            | Func::LambertW0
            | Func::Beta
            | Func::Si
            | Func::Ci
            | Func::Ei
            | Func::Li
            | Func::Zeta
            | Func::Polygamma
            | Func::KroneckerDelta
            | Func::Factorial
            | Func::Binomial
            | non_node_funcs!() => None,
        }
    }

    /// Is this one of the 15 node-backed special functions: `Γ`, `ln Γ`,
    /// `ψ`, `ψ⁽ⁿ⁾`, `ζ`, `B`, `n!`, `C(n, k)`, `erf`, `erfc`, `Ei`, `li`,
    /// `Si`, `Ci`, `W`?  (The library functions are special functions too;
    /// [`Func::family`] tells them apart.)  Exhaustive, as
    /// [`Func::elementary`].
    pub(crate) const fn is_special(self) -> bool {
        match self {
            Func::Gamma
            | Func::LogGamma
            | Func::Digamma
            | Func::Polygamma
            | Func::Zeta
            | Func::Beta
            | Func::Factorial
            | Func::Binomial
            | Func::Erf
            | Func::Erfc
            | Func::Ei
            | Func::Li
            | Func::Si
            | Func::Ci
            | Func::LambertW0 => true,
            Func::Min
            | Func::Max
            | Func::Floor
            | Func::Ceiling
            | Func::Sin
            | Func::Cos
            | Func::Tan
            | Func::Exp
            | Func::Ln
            | Func::Abs
            | Func::Asin
            | Func::Acos
            | Func::Atan
            | Func::Atan2
            | Func::Sinh
            | Func::Cosh
            | Func::Tanh
            | Func::Asinh
            | Func::Acosh
            | Func::Atanh
            | Func::Sign
            | Func::Heaviside
            | Func::DiracDelta
            | Func::Re
            | Func::Im
            | Func::Conjugate
            | Func::Arg
            | Func::KroneckerDelta
            | non_node_funcs!() => false,
        }
    }
}

/// The family of an elementary transcendental function ([`Func::elementary`]).
/// The algorithms keep their own finer private kinds (`TrigKind`, …) where
/// they dispatch on the individual function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Elementary {
    /// `sin`, `cos`, `tan`.
    Trig,
    /// `asin`, `acos`, `atan` (one argument).
    InverseTrig,
    /// `sinh`, `cosh`, `tanh`.
    Hyperbolic,
    /// `asinh`, `acosh`, `atanh`.
    InverseHyperbolic,
    /// `exp`.
    Exp,
    /// `ln`.
    Ln,
}

/// The error for a call of `f` with `n` arguments.
fn arity_error(f: Func, n: usize) -> SymplexError {
    let spec = f.spec();
    SymplexError::invalid_argument(
        "Arena::func",
        format!("{} takes {}, got {n}", spec.name, spec.arity),
    )
}

/// The `N` arguments of a call of `f` with exactly `N`.
fn fixed<const N: usize>(f: Func, args: &[ExprId]) -> Result<[ExprId; N], SymplexError> {
    <[ExprId; N]>::try_from(args).map_err(|_| arity_error(f, args.len()))
}

impl Arena {
    /// Builds `f(args)` exactly as today's constructor for `f` does — the
    /// canonical folds included (`Si(0)` is `0`, `abs(−2·x)` is `2·abs(x)`,
    /// `KroneckerDelta` orders its arguments, `lambertw(x, 0)` is the
    /// principal-branch node).
    ///
    /// # Errors
    ///
    /// [`SymplexError::InvalidArgument`] when `f` does not take
    /// `args.len()` arguments.
    pub(crate) fn func(&mut self, f: Func, args: &[ExprId]) -> Result<ExprId, SymplexError> {
        if !f.spec().arity.accepts(args.len()) {
            return Err(arity_error(f, args.len()));
        }
        match f.family() {
            Family::Node => self.node_func(f, args),
            Family::Lib => {
                let Some(lib) = f.as_lib() else {
                    return Err(arity_error(f, args.len()));
                };
                Ok(match (lib, args) {
                    (LibFn::LambertW, [x]) => self.lambertw(*x),
                    (LibFn::LambertW, [x, k]) => self.lambertw_branch(*x, *k),
                    _ => self.lib_apply(lib, args),
                })
            }
            Family::Internal => {
                let Some(g) = f.as_internal() else {
                    return Err(arity_error(f, args.len()));
                };
                Ok(self.intern(ExprNode::Internal(g, args.iter().copied().collect())))
            }
        }
    }

    /// [`Arena::func`] of a node-backed function, arity already checked.
    fn node_func(&mut self, f: Func, args: &[ExprId]) -> Result<ExprId, SymplexError> {
        let one = |args: &[ExprId]| fixed::<1>(f, args).map(|[x]| x);
        Ok(match f {
            Func::Min => self.intern(ExprNode::Min(args.iter().copied().collect())),
            Func::Max => self.intern(ExprNode::Max(args.iter().copied().collect())),
            Func::Floor => self.floor(one(args)?),
            Func::Ceiling => self.ceiling(one(args)?),
            Func::Sin => self.sin(one(args)?),
            Func::Cos => self.cos(one(args)?),
            Func::Tan => self.tan(one(args)?),
            Func::Exp => self.exp(one(args)?),
            Func::Ln => self.ln(one(args)?),
            Func::Abs => self.abs(one(args)?),
            Func::Asin => self.asin(one(args)?),
            Func::Acos => self.acos(one(args)?),
            Func::Atan => self.atan(one(args)?),
            Func::Atan2 => {
                let [y, x] = fixed(f, args)?;
                self.atan2(y, x)
            }
            Func::Sinh => self.sinh(one(args)?),
            Func::Cosh => self.cosh(one(args)?),
            Func::Tanh => self.tanh(one(args)?),
            Func::Asinh => self.asinh(one(args)?),
            Func::Acosh => self.acosh(one(args)?),
            Func::Atanh => self.atanh(one(args)?),
            Func::Sign => self.sign(one(args)?),
            Func::Heaviside => self.heaviside(one(args)?),
            Func::DiracDelta => self.dirac_delta(one(args)?),
            Func::Re => self.re(one(args)?),
            Func::Im => self.im(one(args)?),
            Func::Conjugate => self.conjugate(one(args)?),
            Func::Arg => self.arg(one(args)?),
            Func::Gamma => self.gamma(one(args)?),
            Func::LogGamma => self.log_gamma(one(args)?),
            Func::Digamma => self.digamma(one(args)?),
            Func::Erf => self.erf(one(args)?),
            Func::Erfc => self.erfc(one(args)?),
            Func::LambertW0 => self.lambertw(one(args)?),
            Func::Beta => {
                let [a, b] = fixed(f, args)?;
                self.beta(a, b)
            }
            Func::Si => self.si(one(args)?),
            Func::Ci => self.ci(one(args)?),
            Func::Ei => self.ei(one(args)?),
            Func::Li => self.li(one(args)?),
            Func::Zeta => self.zeta(one(args)?),
            Func::Polygamma => {
                let [n, x] = fixed(f, args)?;
                self.polygamma(n, x)
            }
            Func::KroneckerDelta => {
                let [i, j] = fixed(f, args)?;
                self.kronecker_delta(i, j)
            }
            Func::Factorial => self.factorial(one(args)?),
            Func::Binomial => {
                let [n, k] = fixed(f, args)?;
                self.binomial(n, k)
            }
            non_node_funcs!() => {
                return Err(SymplexError::computation_failed(
                    "Arena::func",
                    format!("{} is not a node-backed function", f.spec().name),
                ));
            }
        })
    }
}

/// The functions a predicate selects, as a set: the tests that a list
/// rewritten as a predicate selects exactly its old set.
#[cfg(test)]
pub(crate) fn selected(pred: impl Fn(Func) -> bool) -> std::collections::HashSet<Func> {
    Func::ALL.iter().copied().filter(|&f| pred(f)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;
    use crate::api::expr::Ex;
    use crate::base::assumptions::{Assumption, Props};

    /// Symbols for argument slots, in sort-key order.
    const SLOTS: [&str; 4] = ["t", "x", "y", "z"];

    /// A typical argument count for `f` (two for the variadic `Min`/`Max`,
    /// which fold a single argument to itself; two for `lambertw(x, k)`,
    /// whose one-argument call is the principal-branch node).
    fn sample_count(f: Func) -> usize {
        match f.spec().arity {
            ArgCount::Exactly(n) => usize::from(n),
            ArgCount::Between { max, .. } => usize::from(max),
            ArgCount::AtLeast(n) => usize::from(n).max(2),
        }
    }

    /// A context, its slot symbols, and `f` of the first `n` of them built
    /// by [`Arena::func`].
    fn built(f: Func, n: usize) -> (Context, Vec<Ex>, Ex) {
        let ctx = Context::new();
        let syms: Vec<Ex> = SLOTS[..n].iter().map(|s| ctx.symbol(s)).collect();
        let ids: Vec<ExprId> = syms.iter().map(Ex::raw_id).collect();
        let id = ctx
            .with_arena_mut(|a| a.func(f, &ids))
            .unwrap_or_else(|e| panic!("{f:?}: {e}"));
        let e = syms[0].wrap(id);
        (ctx, syms, e)
    }

    fn render_tex(tex: Tex, args: &[String], atom: bool) -> String {
        match tex {
            Tex::Call(head) => format!(r"{head}\left({}\right)", args.join(", ")),
            Tex::Layout(pieces) => pieces
                .iter()
                .map(|p| match *p {
                    TexPiece::Lit(s) => s.to_owned(),
                    TexPiece::Arg(i) => args[usize::from(i)].clone(),
                    TexPiece::Group(i) if atom => args[usize::from(i)].clone(),
                    TexPiece::Group(i) => format!(r"\left({}\right)", args[usize::from(i)]),
                })
                .collect(),
            Tex::Postfix(op) if atom => format!("{}{op}", args[0]),
            Tex::Postfix(op) => format!(r"\left({}\right){op}", args[0]),
        }
    }

    #[test]
    fn the_list_has_every_function_once_in_discriminant_order() {
        let count = |fam: Family| Func::ALL.iter().filter(|f| f.family() == fam).count();
        assert_eq!(
            (
                count(Family::Node),
                count(Family::Lib),
                count(Family::Internal)
            ),
            (43, 50, 8)
        );
        assert_eq!(Func::ALL.len(), Func::SPECS.len());
        for (i, f) in Func::ALL.iter().enumerate() {
            assert_eq!(*f as usize, i, "{f:?}");
        }
        let mut names: Vec<&str> = Func::ALL.iter().map(|f| f.spec().name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Func::ALL.len(), "two entries with one name");
    }

    #[test]
    fn lib_fn_is_the_library_part_of_func() {
        let libs: Vec<Func> = Func::ALL
            .iter()
            .copied()
            .filter(|f| f.family() == Family::Lib)
            .collect();
        assert_eq!(libs.len(), LibFn::ALL.len());
        for (lib, f) in LibFn::ALL.iter().zip(&libs) {
            assert_eq!(Func::from(*lib), *f);
            assert_eq!(f.as_lib(), Some(*lib));
            assert_eq!(lib.name(), f.spec().name);
            assert_eq!(LibFn::from_name(lib.name()), Some(*lib));
            assert!(!matches!(f.spec().arity, ArgCount::AtLeast(_)), "{lib:?}");
            for n in 0..6 {
                assert_eq!(
                    lib.arity().accepts(n),
                    f.spec().arity.accepts(n),
                    "{lib:?} {n}"
                );
            }
        }
        for f in Func::ALL.iter().filter(|f| f.family() != Family::Lib) {
            assert_eq!(f.as_lib(), None);
            assert_eq!(LibFn::from_name(f.spec().name), None, "{f:?}");
        }
    }

    /// The internal functions are the `Internal` nodes, whose heads are
    /// reserved names but which no name builds: until 0.49 they were
    /// `Apply` nodes resolved by the text of the head, and
    /// `Context::from_tree` of `Apply{"__ei_asym", [x]}` built the internal
    /// function, with limit 0 at `∞` (an unknown function has none).
    #[test]
    fn internal_functions_are_internal_nodes_that_no_name_builds() {
        for &f in Func::ALL {
            let Some(g) = f.as_internal() else {
                assert_ne!(f.family(), Family::Internal, "{f:?}");
                assert!(!Func::is_internal_name(f.spec().name), "{f:?}");
                continue;
            };
            assert_eq!(f.family(), Family::Internal);
            assert_eq!(Func::from(g), f);
            assert_eq!(g.name(), f.spec().name);
            assert!(Func::is_internal_name(g.name()));
            let n = sample_count(f);
            let (ctx, syms, e) = built(f, n);
            ctx.with_arena_mut(|a| {
                assert!(matches!(a.node(e.raw_id()), ExprNode::Internal(h, _) if *h == g));
            });
            // The JSON form names the head; reading it back never gives the
            // internal node: `from_json` refuses, `from_tree` builds a user
            // function (an `Apply`), `parse` and `apply` refuse.
            let json = e.to_json().unwrap_or_else(|err| panic!("{f:?}: {err}"));
            assert!(
                json.contains(&format!("\"name\":\"{}\"", g.name())),
                "{json}"
            );
            let back = ctx.from_tree(&e.to_tree());
            assert_ne!(back.raw_id(), e.raw_id(), "{f:?}");
            ctx.with_arena_mut(|a| {
                assert!(
                    matches!(a.node(back.raw_id()), ExprNode::Apply(..)),
                    "{f:?}"
                );
                assert_eq!(a.node(back.raw_id()).func(a), None);
            });
            assert!(ctx.from_json(&json).is_err(), "{f:?}");
            assert!(ctx.parse(&e.to_string()).is_err(), "{f:?}");
            assert!(ctx.apply(g.name(), &syms).is_err(), "{f:?}");
        }
    }

    /// (a) The node `Arena::func` builds has exactly the sort key that the
    /// spec's sub-rank predicts, followed by its arguments' keys: the bytes
    /// of `sort_key.rs` by construction, for every function.
    #[test]
    fn sort_keys_follow_the_spec() {
        for &f in Func::ALL {
            let (ctx, syms, e) = built(f, sample_count(f));
            let mut expected: Vec<u8> = match f.spec().sort {
                SortSub::Function(b) => vec![100, b],
                SortSub::Rank(r) => vec![r],
                SortSub::Named => {
                    let mut k = vec![100, 17];
                    k.extend_from_slice(f.spec().name.as_bytes());
                    k.push(0);
                    k
                }
            };
            ctx.with_arena_mut(|a| {
                for s in &syms {
                    expected.extend_from_slice(a.sort_key(s.raw_id()));
                }
                assert_eq!(a.sort_key(e.raw_id()), &expected[..], "{f:?}");
            });
        }
    }

    /// The function sub-rank bytes are distinct, except the one historical
    /// duplicate.
    #[test]
    fn sort_bytes_are_unique_but_for_binomial_and_asin() {
        let mut seen: Vec<(u8, Func)> = Vec::new();
        for &f in Func::ALL {
            if let SortSub::Function(b) = f.spec().sort {
                for &(c, g) in &seen {
                    if c == b {
                        assert_eq!((g, f), (Func::Asin, Func::Binomial), "byte {b}");
                    }
                }
                seen.push((b, f));
            }
        }
        assert_ne!(seen.len(), 0);
    }

    /// (b) `Display` prints `name(args)`, but for the two documented
    /// exceptions.
    #[test]
    fn display_uses_the_spec_name() {
        for &f in Func::ALL {
            let n = sample_count(f);
            let (_ctx, _syms, e) = built(f, n);
            let expected = match f {
                Func::Factorial => "t!".to_owned(),
                Func::LambertW => "W(t, x)".to_owned(),
                _ => format!("{}({})", f.spec().name, SLOTS[..n].join(", ")),
            };
            assert_eq!(e.to_string(), expected, "{f:?}");
        }
    }

    /// (c) Parsing the display builds the same node; so do the name in
    /// lower case and every further parse name.  Internal functions are not
    /// parseable (in a context that has no such node: the parser reads a
    /// call of a head some `Apply` already has as that function).
    #[test]
    fn parse_of_display_and_of_every_name_builds_the_same_node() {
        for &f in Func::ALL {
            let n = sample_count(f);
            let (ctx, _syms, e) = built(f, n);
            if f.family() == Family::Internal {
                assert!(Context::new().parse(&e.to_string()).is_err(), "{f:?}");
                continue;
            }
            let back = ctx
                .parse(&e.to_string())
                .unwrap_or_else(|err| panic!("{f:?}: {err}"));
            assert_eq!(back.raw_id(), e.raw_id(), "{f:?}: {e}");
            let spec = f.spec();
            let mut names: Vec<String> = spec.parse.iter().map(|s| (*s).to_owned()).collect();
            names.push(spec.name.to_owned());
            if spec.name != "H" {
                names.push(spec.name.to_ascii_lowercase());
            }
            for name in names {
                let src = format!("{name}({})", SLOTS[..n].join(", "));
                let back = ctx.parse(&src).unwrap_or_else(|err| panic!("{src}: {err}"));
                assert_eq!(back.raw_id(), e.raw_id(), "{src}");
            }
        }
    }

    /// A library function's node is `ExprNode::Lib`, identified by the
    /// function; an `Apply` is never one, whatever the text of its head
    /// (until 0.49 library functions were `Apply` nodes resolved by name, so
    /// `Context::apply("bell", [3])` evaluated to 5 as an "undefined"
    /// function).
    #[test]
    fn library_functions_are_lib_nodes_and_apply_is_never_one() {
        for &f in Func::ALL.iter().filter(|f| f.family() == Family::Lib) {
            let n = sample_count(f);
            let (ctx, _syms, e) = built(f, n);
            ctx.with_arena_mut(|a| {
                let node = a.node(e.raw_id()).clone();
                assert!(
                    matches!(node, ExprNode::Lib(lib, ref args)
                        if Some(lib) == f.as_lib() && args.len() == n),
                    "{f:?}: {node:?}"
                );
                assert_eq!(node.func(a).map(|(g, _)| g), Some(f), "{f:?}");
                let head = a.symbols.intern(f.spec().name);
                let apply = ExprNode::Apply(head, node.children().into_iter().collect());
                assert_eq!(apply.func(a), None, "{f:?}");
                // The sort key is the one the `Apply` form had (byte identity).
                let old = a.intern(apply);
                assert_ne!(old, e.raw_id());
                assert_eq!(a.sort_key(old), a.sort_key(e.raw_id()), "{f:?}");
            });
        }
    }

    /// The 14 elementary functions, the or-pattern of
    /// `calculus_util::like_arguments_unified` until 0.49 (W6c: each list
    /// rewritten through a family selects exactly its old set).
    #[test]
    fn elementary_is_the_old_list_of_fourteen() {
        use Func::*;
        let old = [
            Ln, Exp, Sin, Cos, Tan, Sinh, Cosh, Tanh, Asin, Acos, Atan, Asinh, Acosh, Atanh,
        ];
        assert_eq!(
            selected(|f| f.elementary().is_some()),
            old.into_iter().collect()
        );
        assert_eq!(Sin.elementary(), Some(Elementary::Trig));
        assert_eq!(Acosh.elementary(), Some(Elementary::InverseHyperbolic));
    }

    /// `heurisch::collect_components` and `integrate::u_sub_candidates`:
    /// the elementary functions and `|·|` are their old 15-variant lists.
    #[test]
    fn elementary_and_abs_are_the_old_component_lists() {
        use Func::*;
        let old = [
            Sin, Cos, Tan, Exp, Ln, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Abs,
        ];
        assert_eq!(
            selected(|f| f.elementary().is_some() || f == Abs),
            old.into_iter().collect()
        );
    }

    /// `canon::is_costly_node`: the node-backed special functions are its
    /// old 15-variant list.
    #[test]
    fn is_special_is_the_old_costly_list() {
        use Func::*;
        let old = [
            Gamma, LogGamma, Digamma, Polygamma, Zeta, Beta, Factorial, Binomial, Erf, Erfc, Ei,
            Li, Si, Ci, LambertW0,
        ];
        assert_eq!(selected(Func::is_special), old.into_iter().collect());
    }

    /// `gruntz::apply_unary` rebuilds every one-argument function through
    /// `Arena::func`: each of the 24 its old arms built is one, and
    /// `Arena::func` builds it with that constructor (`view_inverts_the_constructor`).
    #[test]
    fn apply_unary_covers_its_old_arms() {
        use Func::*;
        let old = [
            Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Erf, Erfc,
            Gamma, LogGamma, Digamma, LambertW0, Factorial, Abs, Sign, Heaviside, Exp, Ln,
        ];
        for f in old {
            assert_eq!(f.spec().arity, Exactly(1), "{f:?}");
            let (ctx, syms, e) = built(f, 1);
            ctx.with_arena_mut(|a| {
                let node = a.node(e.raw_id()).clone();
                assert_eq!(
                    node.func(a),
                    Some((f, smallvec![syms[0].raw_id()])),
                    "{f:?}"
                );
            });
        }
    }

    /// The internal heads print as valid TeX: until 0.49 `__ei_asym(t)`
    /// printed `__ei_asym\left(t\right)`, a double subscript outside math
    /// mode that TeX rejects.
    #[test]
    fn internal_heads_print_as_valid_tex() {
        for &f in Func::ALL.iter().filter(|f| f.family() == Family::Internal) {
            let (_ctx, _syms, e) = built(f, sample_count(f));
            let tex = e.to_latex();
            assert!(tex.starts_with(r"\operatorname{\_\_"), "{f:?}: {tex}");
            assert!(!tex.replace(r"\_", "").contains('_'), "{f:?}: {tex}");
        }
    }

    /// (d) The LaTeX of every function is its spec's layout.
    #[test]
    fn latex_is_the_spec_layout() {
        for &f in Func::ALL {
            let n = sample_count(f);
            let (_ctx, syms, e) = built(f, n);
            let args: Vec<String> = syms.iter().map(Ex::to_latex).collect();
            assert_eq!(e.to_latex(), render_tex(f.spec().tex, &args, true), "{f:?}");
        }
        // The postfix layout of a compound argument.
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let x1 = &x + 1;
        let id = ctx.with_arena_mut(|a| a.func(Func::Factorial, &[x1.raw_id()]));
        let fact = x.wrap(id.unwrap_or_else(|e| panic!("{e}")));
        assert_eq!(
            fact.to_latex(),
            render_tex(Func::Factorial.spec().tex, &[x1.to_latex()], false)
        );
    }

    /// (e) Arena::func accepts exactly the spec's argument counts.
    #[test]
    fn arity_errors_for_wrong_argument_counts() {
        for &f in Func::ALL {
            let ctx = Context::new();
            let syms: Vec<ExprId> = ["t", "x", "y", "z", "u", "v"]
                .iter()
                .map(|s| ctx.symbol(s).raw_id())
                .collect();
            for n in 0..=syms.len() {
                let r = ctx.with_arena_mut(|a| a.func(f, &syms[..n]));
                match r {
                    Ok(_) => assert!(f.spec().arity.accepts(n), "{f:?} accepted {n}"),
                    Err(SymplexError::InvalidArgument { reason, .. }) => {
                        assert!(!f.spec().arity.accepts(n), "{f:?} refused {n}: {reason}");
                        assert!(reason.starts_with(f.spec().name), "{reason}");
                    }
                    Err(e) => panic!("{f:?} {n}: {e}"),
                }
            }
        }
    }

    /// The view names the function `Arena::func` built, with its arguments.
    #[test]
    fn view_inverts_the_constructor() {
        for &f in Func::ALL {
            let n = sample_count(f);
            let (ctx, syms, e) = built(f, n);
            let ids: SmallVec<[ExprId; 4]> = syms.iter().map(Ex::raw_id).collect();
            ctx.with_arena_mut(|a| {
                let node = a.node(e.raw_id()).clone();
                assert_eq!(node.func(a), Some((f, ids.clone())), "{f:?}");
            });
        }
    }

    /// `f(args)` built by [`Arena::func`], or `None` when the constructor
    /// folded it to another node (`acosh(cosh x)`, `li(1)`).
    fn build_f(ctx: &Context, f: Func, args: &[Ex]) -> Option<Ex> {
        let ids: Vec<ExprId> = args.iter().map(Ex::raw_id).collect();
        let id = ctx
            .with_arena_mut(|a| a.func(f, &ids))
            .unwrap_or_else(|e| panic!("{f:?}: {e}"));
        let e = ctx.zero().wrap(id);
        let same = ctx.with_arena_mut(|a| {
            let node = a.node(id).clone();
            node.func(a).is_some_and(|(g, _)| g == f)
        });
        same.then_some(e)
    }

    fn declared(ctx: &Context, name: &str, facts: &[Assumption]) -> Ex {
        ctx.symbol_with(name, facts)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    /// The parity facet is today's sign extraction — at construction
    /// (`canon.rs`) or by `eval` (`eval.rs`) — for every function of one
    /// argument.
    #[test]
    fn parity_is_todays_sign_extraction() {
        for &f in Func::ALL {
            if f.spec().arity != ArgCount::Exactly(1) {
                assert_eq!(f.spec().parity, Parity::Neither, "{f:?}");
                continue;
            }
            let ctx = Context::new();
            let z = ctx.symbol("z");
            let two = ctx.int(2);
            let two_z = &two * &z;
            let one = |arg: &Ex| {
                let id = ctx
                    .with_arena_mut(|a| a.func(f, &[arg.raw_id()]))
                    .unwrap_or_else(|e| panic!("{f:?}: {e}"));
                z.wrap(id)
            };
            let (f_z, f_neg_z) = (one(&z), one(&-&z));
            let (f_2z, f_neg_2z) = (one(&two_z), one(&-&two_z));
            let (f_2, f_neg_2) = (one(&two), one(&-&two));
            let (odd, sign) = match f.spec().parity {
                Parity::Odd(s) => (true, s),
                Parity::Even(s) => (false, s),
                Parity::Neither => {
                    assert_eq!(f_neg_z.eval(), f_neg_z, "{f:?}");
                    continue;
                }
            };
            let signed = |e: &Ex| if odd { -e } else { e.clone() };
            match sign {
                AutoSign::Canonical => assert_eq!(f_neg_z, signed(&f_z), "{f:?}"),
                AutoSign::Coefficient => {
                    assert_ne!(f_neg_z, signed(&f_z), "{f:?} at construction");
                    assert_eq!(f_neg_2z.eval(), signed(&f_2z), "{f:?}");
                }
                AutoSign::Negation => {
                    assert_eq!(f_neg_z.eval(), signed(&f_z), "{f:?}");
                    assert_eq!(f_neg_2z.eval(), f_neg_2z, "{f:?}");
                    assert_eq!(f_neg_2.eval(), signed(&f_2), "{f:?}");
                }
                AutoSign::SymbolFree => {
                    assert_eq!(f_neg_z.eval(), f_neg_z, "{f:?}");
                    assert_eq!(f_neg_2.eval(), signed(&f_2.eval()), "{f:?}");
                }
                AutoSign::Never => assert_eq!(f_neg_z.eval(), f_neg_z, "{f:?}"),
            }
        }
    }

    /// Every function is real on its `real_on` domain — by the table's
    /// handler or by its own — and no function of complex arguments but the
    /// real-valued ones is claimed real.
    #[test]
    fn real_on_is_what_the_assumption_system_claims() {
        use Assumption::{Integer, Negative, NonNegative, Positive, Real};
        for &f in Func::ALL {
            let real_on = f.spec().real_on;
            if real_on == RealOn::Custom {
                continue;
            }
            let n = sample_count(f);
            let ctx = Context::new();
            let mut args: Vec<Ex> = (0..n)
                .map(|i| declared(&ctx, &format!("r{i}"), &[Real]))
                .collect();
            let last = n - 1;
            let x = declared(&ctx, "x", &[Real]);
            match real_on {
                RealOn::PositiveReals => args[last] = declared(&ctx, "p", &[Positive]),
                RealOn::NonNegativeReals => args[last] = declared(&ctx, "u", &[Real, NonNegative]),
                RealOn::BelowOne => args[last] = declared(&ctx, "q", &[Negative]),
                RealOn::UnitInterval { closed: true } => args[last] = x.sin(),
                RealOn::UnitInterval { closed: false } => args[last] = x.tanh(),
                RealOn::AtLeastOne => args[last] = x.cosh(),
                RealOn::Polynomial { degree } => {
                    args[usize::from(degree)] = declared(&ctx, "k", &[Integer, NonNegative]);
                }
                RealOn::BesselOrder => args[0] = declared(&ctx, "m", &[Integer]),
                RealOn::NonNegativeIntegers => {
                    for (i, a) in args.iter_mut().enumerate() {
                        *a = declared(&ctx, &format!("k{i}"), &[Integer, NonNegative]);
                    }
                }
                _ => {}
            }
            if let Some(e) = build_f(&ctx, f, &args) {
                let real = ctx.query(&e, Props::REAL);
                if real_on == RealOn::Unclaimed {
                    assert_ne!(real, Some(true), "{f:?}: {e}");
                } else {
                    assert_eq!(real, Some(true), "{f:?}: {e}");
                }
                if real_on == RealOn::IntegerValued {
                    assert_eq!(ctx.query(&e, Props::INTEGER), Some(true), "{f:?}: {e}");
                }
            }
            let plain: Vec<Ex> = (0..n).map(|i| ctx.symbol(&format!("c{i}"))).collect();
            if real_on != RealOn::Everywhere
                && let Some(e) = build_f(&ctx, f, &plain)
            {
                assert_ne!(ctx.query(&e, Props::REAL), Some(true), "{f:?}: {e}");
            }
        }
    }

    /// A function whose special values at constants are not all folded
    /// ([`Constants::Evaluated`]) gets no claim at constant arguments —
    /// `B(1/2, −1/2)`, `subfactorial(1)`, `chebyshev_t(3, 0)` are unfolded
    /// zeros — while a folded one keeps its claims there, and both claim for
    /// symbols.
    #[test]
    fn constants_facet_decides_claims_at_constant_arguments() {
        use crate::base::assumptions::AssumptionCache;
        let ctx = Context::new();
        ctx.symbol_with("x", &[Assumption::Real])
            .unwrap_or_else(|e| panic!("{e}"));
        for (src, want) in [
            ("beta(1/2, -1/2)", None),
            ("subfactorial(1)", None),
            ("chebyshev_t(3, 0)", None),
            ("erfi(2)", None),
            ("beta(x, -1/2)", Some(true)),
            ("erfi(x)", Some(true)),
            ("Gamma(1/3)", Some(true)),
            ("lambertw(2)", Some(true)),
            ("factorial(5/2)", Some(true)),
        ] {
            let e = ctx.parse(src).unwrap_or_else(|err| panic!("{src}: {err}"));
            let got =
                ctx.with_arena_mut(|a| AssumptionCache::new().query(a, e.raw_id(), Props::REAL));
            assert_eq!(got, want, "{src}: {e}");
        }
    }

    /// An argument known to be at a pole withholds the claims "real" and
    /// "finite complex".
    #[test]
    fn no_claim_at_a_known_pole() {
        use Assumption::{Integer, NonPositive, Real, Zero};
        for &f in Func::ALL {
            let n = sample_count(f);
            let ctx = Context::new();
            let mut args: Vec<Ex> = (0..n)
                .map(|i| declared(&ctx, &format!("r{i}"), &[Real]))
                .collect();
            match f.spec().poles {
                Poles::NonPositiveIntegers(which) => {
                    for &i in which {
                        args[usize::from(i)] = declared(&ctx, "np", &[Integer, NonPositive]);
                    }
                }
                Poles::Zero(i) => args[usize::from(i)] = declared(&ctx, "o", &[Zero]),
                Poles::One(i) => args[usize::from(i)] = ctx.int(1),
                Poles::None | Poles::Other => continue,
            }
            if let Some(e) = build_f(&ctx, f, &args) {
                assert_ne!(ctx.query(&e, Props::REAL), Some(true), "{f:?}: {e}");
                assert_ne!(ctx.query(&e, Props::COMPLEX), Some(true), "{f:?}: {e}");
            }
        }
    }

    /// `conjugate` pushes the conjugate into `f(args)` exactly where the
    /// table's rule holds by the assumptions: always for `Entire`, for an
    /// imaginary (finite, non-real) cut argument for `OffCut`, for an
    /// integer order or degree; never for `No`.
    #[test]
    fn conjugate_follows_the_reflection_facet() {
        use Assumption::{Imaginary, Integer, NonNegative, Real};
        for &f in Func::ALL {
            let n = sample_count(f);
            let ctx = Context::new();
            let plain: Vec<Ex> = (0..n).map(|i| ctx.symbol(&format!("c{i}"))).collect();
            let with = |i: u8, e: Ex| {
                let mut args = plain.clone();
                args[usize::from(i)] = e;
                args
            };
            let off_cut = |cut: Cut| {
                if cut.line == CutLine::ImaginaryOutsideUnit {
                    declared(&ctx, "r", &[Real])
                } else {
                    declared(&ctx, "w", &[Imaginary])
                }
            };
            let integer = |nonnegative: bool| {
                if nonnegative {
                    declared(&ctx, "k", &[Integer, NonNegative])
                } else {
                    declared(&ctx, "m", &[Integer])
                }
            };
            // (arguments, whether the rule holds for them by the assumptions)
            let cases: Vec<(Vec<Ex>, bool)> = match f.spec().conj {
                Conj::Entire => vec![(plain.clone(), true)],
                Conj::OffCut(cut) => {
                    vec![(plain.clone(), false), (with(cut.arg, off_cut(cut)), true)]
                }
                Conj::IntegerOrderOffCut { cut, nonnegative } => vec![
                    (plain.clone(), false),
                    (with(cut.arg, off_cut(cut)), true),
                    (with(0, integer(nonnegative)), true),
                ],
                Conj::Polynomial { degree } => {
                    vec![(plain.clone(), false), (with(degree, integer(true)), true)]
                }
                Conj::No => vec![(plain.clone(), false)],
            };
            for (args, holds) in cases {
                let Some(e) = build_f(&ctx, f, &args) else {
                    continue;
                };
                if ctx.query(&e, Props::REAL) == Some(true) {
                    continue;
                }
                let ids: Vec<ExprId> = if holds {
                    args.iter().map(|a| a.conjugate().raw_id()).collect()
                } else {
                    Vec::new()
                };
                let want = ctx.with_arena_mut(|a| {
                    if holds {
                        a.func(f, &ids).unwrap_or_else(|err| panic!("{f:?}: {err}"))
                    } else {
                        a.intern(ExprNode::Conjugate(e.raw_id()))
                    }
                });
                assert_eq!(e.conjugate(), e.wrap(want), "{f:?}: conjugate({e})");
            }
        }
    }

    /// User functions, wrong-arity internal calls and non-functions have
    /// no `Func` (`Context::apply` refuses a library name with the wrong
    /// arity since 0.50; `Arena::func` too).
    #[test]
    fn view_of_non_functions_is_none() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let three = ctx.int(3);
        let user = ctx.apply("f", std::slice::from_ref(&x)).unwrap();
        assert!(ctx.apply("Shi", &[x.clone(), three.clone()]).is_err());
        let sum = &x + 1;
        ctx.with_arena_mut(|a| {
            assert!(a.func(Func::Shi, &[x.raw_id(), three.raw_id()]).is_err());
            let head = a.symbols.intern(Func::EiAsym.spec().name);
            let internal2 = a.intern(ExprNode::Apply(head, smallvec![x.raw_id(), three.raw_id()]));
            for id in [user.raw_id(), internal2, sum.raw_id(), x.raw_id()] {
                let node = a.node(id).clone();
                assert_eq!(node.func(a), None, "{node:?}");
            }
        });
    }
}
