//! One identity for every mathematical function: [`Func`], with its
//! declarative data in a [`FuncSpec`], generated from **one list** together
//! with the public [`LibFn`].
//!
//! A function reaches the arena in one of three representations:
//!
//! - a dedicated [`ExprNode`] variant (`Sin(x)`, `Beta(a, b)`, `Min(…)`:
//!   the 43 *node-backed* functions);
//! - an `Apply(name, args)` whose head names a library function (the 50
//!   [`LibFn`]s: Bessel, orthogonal polynomials, `erfi`, …);
//! - an `Apply` headed by one of the 8 internal pseudo-functions of Gruntz's
//!   asymptotic rewrite (`__ei_asym`, …; they exist only inside a limit
//!   computation and are never parsed).
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
//! sort-key sub-rank and the LaTeX layout.  Algorithms (evaluation,
//! derivatives, series, integration) stay in their modules as exhaustive
//! `match f: Func` arms.
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
//!   the interned head symbol of an `Apply` family).  Two exceptions, both
//!   tested: `Factorial` prints postfix (`n!`) and `lambertw(x, k)` prints
//!   as `W(x, k)`.  The parser compares names ignoring ASCII case, except
//!   Heaviside's display name `H`, which it reads only as written (and only
//!   while no undefined function `H` exists); `parse` lists the names it
//!   reads besides `name`, at the function's own arity (`log(x, b)` is a
//!   quotient, not `ln`).
//! - **Wrong argument counts.**  An `Apply` whose head names a library or
//!   internal function but whose argument count the function does not
//!   accept (`ctx.apply("Shi", [x, 3])`) is not a call of that function:
//!   every consumer leaves it unevaluated, and [`ExprNode::func`] returns
//!   `None`.
//! - **Sort keys.**  [`SortSub`] reproduces the hand-assigned bytes of
//!   `sort_key.rs` exactly.  `Binomial` and `Asin` share byte 8 (since
//!   0.13; byte identity keeps it), the one allowlisted duplicate.
//! - **`Min`/`Max`** take any positive number of arguments
//!   ([`ArgCount::AtLeast`]); one argument folds to itself.  The parser asks
//!   for two.

use std::fmt;

use smallvec::{SmallVec, smallvec};

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::libfn::Arity;
use crate::base::node::{ExprId, ExprNode, SymbolId};

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
    /// The function rank (100), the `Apply` byte (17), the interned name and
    /// a 0 byte: library and internal functions sort by name among
    /// themselves and with user functions.
    Named,
}

/// The LaTeX layout of a call, with each argument rendered in place.
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
}

/// The data of one function.  Every entry of the list writes every field
/// (a struct literal without `..`), so a new field is a compile error
/// until each function says what it is.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "`parse`, `sort` and `tex` are read by the tests over `Func::ALL`; the printers, \
                  parser and sort keys migrate to them in wave 2 (W6)"
    )
)]
pub(crate) struct FuncSpec {
    /// The name: the head `Display` prints in call notation (`sin`,
    /// `Gamma`, `W`, `C`), and for a library or internal function the
    /// interned head of its `Apply` node (`besselj`, `lambertw`,
    /// `__ei_asym`).  See the module documentation for the two printing
    /// exceptions.
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
}

/// Which representation a [`Func`] has in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Family {
    /// A dedicated [`ExprNode`] variant.
    Node,
    /// `Apply` headed by a [`LibFn`] name.
    Lib,
    /// `Apply` headed by an internal `__…` name (never parsed).
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
        /// A library special function carried as `Apply(name, args)`.
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

            /// The interned name: what the `Apply` node's symbol holds and
            /// what `Display` prints (`lambertw(x, k)` prints as `W(x, k)`).
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

            /// The internal function whose head is exactly `s`.
            fn internal_named(s: &str) -> Option<Func> {
                match s { $( $iname => Some(Func::$I), )* _ => None }
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

use ArgCount::{AtLeast, Between, Exactly};
use SortSub::{Function, Named, Rank};
use TexPiece::{Arg, Lit};

funcs! {
    node {
        /// Minimum `min(a, b, …)`.
        Min => FuncSpec {
            name: "min", parse: &[], arity: AtLeast(1), sort: Rank(130), tex: Tex::Call(r"\min"),
        };
        /// Maximum `max(a, b, …)`.
        Max => FuncSpec {
            name: "max", parse: &[], arity: AtLeast(1), sort: Rank(132), tex: Tex::Call(r"\max"),
        };
        /// Floor `⌊x⌋`.
        Floor => FuncSpec {
            name: "floor", parse: &[], arity: Exactly(1), sort: Function(20),
            tex: Tex::Layout(&[Lit(r"\lfloor "), Arg(0), Lit(r"\rfloor")]),
        };
        /// Ceiling `⌈x⌉`.
        Ceiling => FuncSpec {
            name: "ceiling", parse: &["ceil"], arity: Exactly(1), sort: Function(21),
            tex: Tex::Layout(&[Lit(r"\lceil "), Arg(0), Lit(r"\rceil")]),
        };
        /// Sine.
        Sin => FuncSpec {
            name: "sin", parse: &[], arity: Exactly(1), sort: Function(0), tex: Tex::Call(r"\sin"),
        };
        /// Cosine.
        Cos => FuncSpec {
            name: "cos", parse: &[], arity: Exactly(1), sort: Function(1), tex: Tex::Call(r"\cos"),
        };
        /// Tangent.
        Tan => FuncSpec {
            name: "tan", parse: &[], arity: Exactly(1), sort: Function(2), tex: Tex::Call(r"\tan"),
        };
        /// Exponential `eˣ`.
        Exp => FuncSpec {
            name: "exp", parse: &[], arity: Exactly(1), sort: Function(3), tex: Tex::Call(r"\exp"),
        };
        /// Natural logarithm (`log(x)` with one argument).
        Ln => FuncSpec {
            name: "ln", parse: &["log"], arity: Exactly(1), sort: Function(4), tex: Tex::Call(r"\ln"),
        };
        /// Absolute value (complex modulus).
        Abs => FuncSpec {
            name: "abs", parse: &[], arity: Exactly(1), sort: Function(6),
            tex: Tex::Layout(&[Lit(r"\left|"), Arg(0), Lit(r"\right|")]),
        };
        /// Inverse sine.
        Asin => FuncSpec {
            name: "asin", parse: &["arcsin"], arity: Exactly(1), sort: Function(8),
            tex: Tex::Call(r"\arcsin"),
        };
        /// Inverse cosine.
        Acos => FuncSpec {
            name: "acos", parse: &["arccos"], arity: Exactly(1), sort: Function(9),
            tex: Tex::Call(r"\arccos"),
        };
        /// Inverse tangent.
        Atan => FuncSpec {
            name: "atan", parse: &["arctan"], arity: Exactly(1), sort: Function(10),
            tex: Tex::Call(r"\arctan"),
        };
        /// Two-argument arctangent `atan2(y, x)`.
        Atan2 => FuncSpec {
            name: "atan2", parse: &[], arity: Exactly(2), sort: Function(19),
            tex: Tex::Call(r"\operatorname{atan2}"),
        };
        /// Hyperbolic sine.
        Sinh => FuncSpec {
            name: "sinh", parse: &[], arity: Exactly(1), sort: Function(11), tex: Tex::Call(r"\sinh"),
        };
        /// Hyperbolic cosine.
        Cosh => FuncSpec {
            name: "cosh", parse: &[], arity: Exactly(1), sort: Function(12), tex: Tex::Call(r"\cosh"),
        };
        /// Hyperbolic tangent.
        Tanh => FuncSpec {
            name: "tanh", parse: &[], arity: Exactly(1), sort: Function(13), tex: Tex::Call(r"\tanh"),
        };
        /// Inverse hyperbolic sine.
        Asinh => FuncSpec {
            name: "asinh", parse: &["arcsinh"], arity: Exactly(1), sort: Function(14),
            tex: Tex::Call(r"\operatorname{asinh}"),
        };
        /// Inverse hyperbolic cosine.
        Acosh => FuncSpec {
            name: "acosh", parse: &["arccosh"], arity: Exactly(1), sort: Function(15),
            tex: Tex::Call(r"\operatorname{acosh}"),
        };
        /// Inverse hyperbolic tangent.
        Atanh => FuncSpec {
            name: "atanh", parse: &["arctanh"], arity: Exactly(1), sort: Function(16),
            tex: Tex::Call(r"\operatorname{atanh}"),
        };
        /// Sign function.
        Sign => FuncSpec {
            name: "sign", parse: &["sgn"], arity: Exactly(1), sort: Function(18),
            tex: Tex::Call(r"\operatorname{sgn}"),
        };
        /// Heaviside step `H(x)` (the parser reads `H` only as written).
        Heaviside => FuncSpec {
            name: "H", parse: &["heaviside"], arity: Exactly(1), sort: Function(28),
            tex: Tex::Call(r"\operatorname{H}"),
        };
        /// Dirac delta `δ(x)`.
        DiracDelta => FuncSpec {
            name: "DiracDelta", parse: &["dirac_delta"], arity: Exactly(1), sort: Function(29),
            tex: Tex::Call(r"\delta"),
        };
        /// Real part.
        Re => FuncSpec {
            name: "re", parse: &[], arity: Exactly(1), sort: Function(31), tex: Tex::Call(r"\Re"),
        };
        /// Imaginary part.
        Im => FuncSpec {
            name: "im", parse: &[], arity: Exactly(1), sort: Function(32), tex: Tex::Call(r"\Im"),
        };
        /// Complex conjugate.
        Conjugate => FuncSpec {
            name: "conjugate", parse: &["conj"], arity: Exactly(1), sort: Function(33),
            tex: Tex::Layout(&[Lit(r"\overline{"), Arg(0), Lit("}")]),
        };
        /// Principal argument `arg(z) ∈ (−π, π]`.
        Arg => FuncSpec {
            name: "arg", parse: &[], arity: Exactly(1), sort: Function(34), tex: Tex::Call(r"\arg"),
        };
        /// Gamma function.
        Gamma => FuncSpec {
            name: "Gamma", parse: &[], arity: Exactly(1), sort: Function(22), tex: Tex::Call(r"\Gamma"),
        };
        /// Log-gamma function.
        LogGamma => FuncSpec {
            name: "LogGamma", parse: &[], arity: Exactly(1), sort: Function(23),
            tex: Tex::Call(r"\ln \Gamma"),
        };
        /// Digamma `ψ(x)`.
        Digamma => FuncSpec {
            name: "Digamma", parse: &[], arity: Exactly(1), sort: Function(24), tex: Tex::Call(r"\psi"),
        };
        /// Error function.
        Erf => FuncSpec {
            name: "erf", parse: &[], arity: Exactly(1), sort: Function(25),
            tex: Tex::Call(r"\operatorname{erf}"),
        };
        /// Complementary error function.
        Erfc => FuncSpec {
            name: "erfc", parse: &[], arity: Exactly(1), sort: Function(26),
            tex: Tex::Call(r"\operatorname{erfc}"),
        };
        /// Principal branch `W₀(x)` of Lambert W (the node `ExprNode::LambertW`;
        /// the other branches are [`Func::LambertW`]).
        LambertW0 => FuncSpec {
            name: "W", parse: &["lambertw"], arity: Exactly(1), sort: Function(30),
            tex: Tex::Call(r"\operatorname{W}"),
        };
        /// Beta function `B(a, b)`.
        Beta => FuncSpec {
            name: "B", parse: &["beta"], arity: Exactly(2), sort: Function(27),
            tex: Tex::Call(r"\mathrm{B}"),
        };
        /// Sine integral.
        Si => FuncSpec {
            name: "Si", parse: &[], arity: Exactly(1), sort: Function(35),
            tex: Tex::Call(r"\operatorname{Si}"),
        };
        /// Cosine integral.
        Ci => FuncSpec {
            name: "Ci", parse: &[], arity: Exactly(1), sort: Function(36),
            tex: Tex::Call(r"\operatorname{Ci}"),
        };
        /// Exponential integral.
        Ei => FuncSpec {
            name: "Ei", parse: &[], arity: Exactly(1), sort: Function(37),
            tex: Tex::Call(r"\operatorname{Ei}"),
        };
        /// Logarithmic integral.
        Li => FuncSpec {
            name: "li", parse: &[], arity: Exactly(1), sort: Function(38),
            tex: Tex::Call(r"\operatorname{li}"),
        };
        /// Riemann zeta.
        Zeta => FuncSpec {
            name: "zeta", parse: &[], arity: Exactly(1), sort: Function(39), tex: Tex::Call(r"\zeta"),
        };
        /// Polygamma `ψ⁽ⁿ⁾(x)`, `polygamma(n, x)`.
        Polygamma => FuncSpec {
            name: "polygamma", parse: &[], arity: Exactly(2), sort: Function(40),
            tex: Tex::Layout(&[Lit(r"\psi^{("), Arg(0), Lit(r")}\left("), Arg(1), Lit(r"\right)")]),
        };
        /// Kronecker delta `δᵢⱼ`.
        KroneckerDelta => FuncSpec {
            name: "KroneckerDelta", parse: &["kronecker_delta"], arity: Exactly(2),
            sort: Function(41),
            tex: Tex::Layout(&[Lit(r"\delta_{"), Arg(0), Lit(" "), Arg(1), Lit("}")]),
        };
        /// Factorial `n!` (printed postfix).
        Factorial => FuncSpec {
            name: "factorial", parse: &[], arity: Exactly(1), sort: Function(7),
            tex: Tex::Postfix("!"),
        };
        /// Binomial coefficient `C(n, k)`.  Its sort byte 8 is `Asin`'s too.
        Binomial => FuncSpec {
            name: "C", parse: &["binomial"], arity: Exactly(2), sort: Function(8),
            tex: Tex::Layout(&[Lit(r"\binom{"), Arg(0), Lit("}{"), Arg(1), Lit("}")]),
        };
    }
    lib {
        // ── Integer sequences and combinatorial counts ────────────────────────
        /// Double factorial `n!!`.
        Factorial2 => FuncSpec {
            name: "factorial2", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("factorial2"),
        };
        /// Subfactorial (derangement count) `!n`.
        Subfactorial => FuncSpec {
            name: "subfactorial", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("subfactorial"),
        };
        /// Rising factorial (Pochhammer symbol) `(x)_n = x (x+1) ⋯ (x+n−1)`.
        RisingFactorial => FuncSpec {
            name: "rising_factorial", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call("rising_factorial"),
        };
        /// Falling factorial `x^(n) = x (x−1) ⋯ (x−n+1)`.
        FallingFactorial => FuncSpec {
            name: "falling_factorial", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call("falling_factorial"),
        };
        /// Fibonacci number `F_n`.
        Fibonacci => FuncSpec {
            name: "fibonacci", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("fibonacci"),
        };
        /// Lucas number `L_n`.
        Lucas => FuncSpec {
            name: "lucas", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("lucas"),
        };
        /// Bernoulli number `B_n`.
        Bernoulli => FuncSpec {
            name: "bernoulli", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("bernoulli"),
        };
        /// Harmonic number `H_n`.
        Harmonic => FuncSpec {
            name: "harmonic", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("harmonic"),
        };
        /// Catalan number `C_n`.
        Catalan => FuncSpec {
            name: "catalan", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("catalan"),
        };
        /// Bell number `B_n`.
        Bell => FuncSpec {
            name: "bell", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("bell"),
        };
        /// Euler number `E_n`.
        EulerNumber => FuncSpec {
            name: "euler_number", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("euler_number"),
        };
        /// Signed Stirling number of the first kind `s(n, k)`.
        Stirling1 => FuncSpec {
            name: "stirling1", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call("stirling1"),
        };
        /// Stirling number of the second kind `S(n, k)`.
        Stirling2 => FuncSpec {
            name: "stirling2", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call("stirling2"),
        };
        /// Integer partition count `p(n)`.
        PartitionCount => FuncSpec {
            name: "partition_count", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("partition_count"),
        };
        /// Lambert W, `lambertw(x, k)`: the branch `W_k` (`k` an integer) of the
        /// inverse of `w·eʷ`, as SymPy's `LambertW(x, k)`.  The principal branch
        /// `k = 0` is the dedicated `ExprNode::LambertW(x)` node, which
        /// `Arena::lambertw_branch` and `eval` build for `lambertw(x)` and
        /// `lambertw(x, 0)`; an `Apply` carries only the other branches.
        LambertW => FuncSpec {
            name: "lambertw", parse: &["w"], arity: Between { min: 1, max: 2 }, sort: Named,
            tex: Tex::Layout(&[
                Lit(r"\operatorname{W}_{"), Arg(1), Lit(r"}\left("), Arg(0), Lit(r"\right)"),
            ]),
        };

        // ── Bessel functions `(order, x)` ─────────────────────────────────────
        /// Bessel function of the first kind `J_ν(x)`.
        BesselJ => FuncSpec {
            name: "besselj", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call("besselj"),
        };
        /// Bessel function of the second kind `Y_ν(x)`.
        BesselY => FuncSpec {
            name: "bessely", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call("bessely"),
        };
        /// Modified Bessel function of the first kind `I_ν(x)`.
        BesselI => FuncSpec {
            name: "besseli", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call("besseli"),
        };
        /// Modified Bessel function of the second kind `K_ν(x)`.
        BesselK => FuncSpec {
            name: "besselk", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call("besselk"),
        };

        // ── Classical orthogonal polynomials `(n, x)` ─────────────────────────
        /// Legendre polynomial `P_n(x)`.
        Legendre => FuncSpec {
            name: "legendre", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call("legendre"),
        };
        /// Chebyshev polynomial of the first kind `T_n(x)`.
        ChebyshevT => FuncSpec {
            name: "chebyshev_t", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call("chebyshev_t"),
        };
        /// Chebyshev polynomial of the second kind `U_n(x)`.
        ChebyshevU => FuncSpec {
            name: "chebyshev_u", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call("chebyshev_u"),
        };
        /// Physicists' Hermite polynomial `H_n(x)`.
        Hermite => FuncSpec {
            name: "hermite", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call("hermite"),
        };
        /// Laguerre polynomial `L_n(x)`.
        Laguerre => FuncSpec {
            name: "laguerre", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call("laguerre"),
        };

        // ── Error-function family and integrals (0.9) ─────────────────────────
        /// Imaginary error function `erfi(x)`.
        Erfi => FuncSpec {
            name: "erfi", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{erfi}"),
        };
        /// Inverse error function `erf⁻¹(y)`.
        ErfInv => FuncSpec {
            name: "erfinv", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{erf}^{-1}"),
        };
        /// Inverse complementary error function `erfc⁻¹(y)`.
        ErfcInv => FuncSpec {
            name: "erfcinv", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{erfc}^{-1}"),
        };
        /// Generalised exponential integral `E_n(x)`, as `expint(n, x)`.
        ExpInt => FuncSpec {
            name: "expint", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[
                Lit(r"\operatorname{E}_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)"),
            ]),
        };
        /// Hyperbolic sine integral `Shi(x)`.
        Shi => FuncSpec {
            name: "Shi", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Shi}"),
        };
        /// Hyperbolic cosine integral `Chi(x)`.
        Chi => FuncSpec {
            name: "Chi", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Chi}"),
        };
        /// Fresnel sine integral `S(x)`.
        FresnelS => FuncSpec {
            name: "fresnels", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("S"),
        };
        /// Fresnel cosine integral `C(x)`.
        FresnelC => FuncSpec {
            name: "fresnelc", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("C"),
        };
        /// Lower incomplete gamma `γ(s, x)`.
        LowerGamma => FuncSpec {
            name: "lowergamma", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call(r"\gamma"),
        };
        /// Upper incomplete gamma `Γ(s, x)`.
        UpperGamma => FuncSpec {
            name: "uppergamma", parse: &[], arity: Exactly(2), sort: Named, tex: Tex::Call(r"\Gamma"),
        };
        /// Polylogarithm `Li_s(z)`, as `polylog(s, z)`.
        PolyLog => FuncSpec {
            name: "polylog", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[
                Lit(r"\operatorname{Li}_{"), Arg(0), Lit(r"}\left("), Arg(1), Lit(r"\right)"),
            ]),
        };
        /// Dirichlet eta `η(s)`.
        DirichletEta => FuncSpec {
            name: "dirichlet_eta", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\eta"),
        };

        // ── Airy functions ────────────────────────────────────────────────────
        /// Airy function `Ai(x)`.
        AiryAi => FuncSpec {
            name: "airyai", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Ai}"),
        };
        /// Airy function `Bi(x)`.
        AiryBi => FuncSpec {
            name: "airybi", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Bi}"),
        };
        /// Derivative `Ai′(x)`.
        AiryAiPrime => FuncSpec {
            name: "airyaiprime", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Ai}^\prime"),
        };
        /// Derivative `Bi′(x)`.
        AiryBiPrime => FuncSpec {
            name: "airybiprime", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call(r"\operatorname{Bi}^\prime"),
        };

        // ── Elliptic integrals (parameter `m = k²`, as in SymPy) ──────────────
        /// Complete elliptic integral of the first kind `K(m)`.
        EllipticK => FuncSpec {
            name: "elliptic_k", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("K"),
        };
        /// Complete elliptic integral of the second kind `E(m)`.
        EllipticE => FuncSpec {
            name: "elliptic_e", parse: &[], arity: Exactly(1), sort: Named, tex: Tex::Call("E"),
        };
        /// Incomplete elliptic integral of the first kind `F(φ | m)`.
        EllipticF => FuncSpec {
            name: "elliptic_f", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit(r"F\left("), Arg(0), Lit(r"\middle| "), Arg(1), Lit(r"\right)")]),
        };
        /// Complete elliptic integral of the third kind `Π(n | m)`.
        EllipticPi => FuncSpec {
            name: "elliptic_pi", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Layout(&[Lit(r"\Pi\left("), Arg(0), Lit(r"\middle| "), Arg(1), Lit(r"\right)")]),
        };

        // ── Orthogonal polynomials with parameters ────────────────────────────
        /// Gegenbauer polynomial `C_n^{(α)}(x)`, as `gegenbauer(n, α, x)`.
        Gegenbauer => FuncSpec {
            name: "gegenbauer", parse: &[], arity: Exactly(3), sort: Named,
            tex: Tex::Layout(&[
                Lit("C_{"), Arg(0), Lit(r"}^{\left("), Arg(1), Lit(r"\right)}\left("), Arg(2),
                Lit(r"\right)"),
            ]),
        };
        /// Jacobi polynomial `P_n^{(α,β)}(x)`, as `jacobi(n, α, β, x)`.
        Jacobi => FuncSpec {
            name: "jacobi", parse: &[], arity: Exactly(4), sort: Named,
            tex: Tex::Layout(&[
                Lit("P_{"), Arg(0), Lit(r"}^{\left("), Arg(1), Lit(","), Arg(2),
                Lit(r"\right)}\left("), Arg(3), Lit(r"\right)"),
            ]),
        };
        /// Associated Legendre function `P_n^m(x)`, as `assoc_legendre(n, m, x)`.
        AssocLegendre => FuncSpec {
            name: "assoc_legendre", parse: &[], arity: Exactly(3), sort: Named,
            tex: Tex::Layout(&[
                Lit("P_{"), Arg(0), Lit(r"}^{\left("), Arg(1), Lit(r"\right)}\left("), Arg(2),
                Lit(r"\right)"),
            ]),
        };
        /// Generalised Laguerre polynomial `L_n^{(α)}(x)`, as `assoc_laguerre(n, α, x)`.
        AssocLaguerre => FuncSpec {
            name: "assoc_laguerre", parse: &[], arity: Exactly(3), sort: Named,
            tex: Tex::Layout(&[
                Lit("L_{"), Arg(0), Lit(r"}^{\left("), Arg(1), Lit(r"\right)}\left("), Arg(2),
                Lit(r"\right)"),
            ]),
        };

        // ── Generalised incomplete beta (0.12), SymPy order `(a, b, x1, x2)` ──
        /// Incomplete beta `B_{(x₁, x₂)}(a, b)`.
        BetaInc => FuncSpec {
            name: "betainc", parse: &[], arity: Exactly(4), sort: Named,
            tex: Tex::Layout(&[
                Lit(r"\operatorname{B}_{("), Arg(2), Lit(", "), Arg(3), Lit(r")}\left("), Arg(0),
                Lit(", "), Arg(1), Lit(r"\right)"),
            ]),
        };
        /// Regularised incomplete beta `I_{(x₁, x₂)}(a, b)`.
        BetaIncRegularized => FuncSpec {
            name: "betainc_regularized", parse: &[], arity: Exactly(4), sort: Named,
            tex: Tex::Layout(&[
                Lit(r"\operatorname{I}_{("), Arg(2), Lit(", "), Arg(3), Lit(r")}\left("), Arg(0),
                Lit(", "), Arg(1), Lit(r"\right)"),
            ]),
        };
    }
    internal {
        // Heads of Gruntz's asymptotic rewrite (`calculus/series.rs`, whose
        // `*_ASYMPTOTIC` constants spell the same names); each exists only
        // inside a limit computation.
        /// `erfcx(z) = e^{z²} erfc(z)` for `z → +∞` (SymPy's `_erfs`).
        ErfcxAsym => FuncSpec {
            name: "__erfcx_asym", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("__erfcx_asym"),
        };
        /// `e^{−z} Ei(z)` for `z → ±∞` (SymPy's `_eis`).
        EiAsym => FuncSpec {
            name: "__ei_asym", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("__ei_asym"),
        };
        /// `√v·e^{−v} erfi(√v)`, a function of `v = z²`.
        Erfix2Asym => FuncSpec {
            name: "__erfix2_asym", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("__erfix2_asym"),
        };
        /// `√v·e^{v} erfc(√v)`, a function of `v = z²`.
        Erfcx2Asym => FuncSpec {
            name: "__erfcx2_asym", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("__erfcx2_asym"),
        };
        /// `√(2πz) e^{−z} Iν(z)`, arguments `(ν, z)`.
        BesselIAsym => FuncSpec {
            name: "__besseli_asym", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call("__besseli_asym"),
        };
        /// `√(2z/π) e^{z} Kν(z)`, arguments `(ν, z)`.
        BesselKAsym => FuncSpec {
            name: "__besselk_asym", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call("__besselk_asym"),
        };
        /// `z^{1−s} e^{z} Γ(s, z)`, arguments `(s, z)`.
        UpperGammaAsym => FuncSpec {
            name: "__uppergamma_asym", parse: &[], arity: Exactly(2), sort: Named,
            tex: Tex::Call("__uppergamma_asym"),
        };
        /// `2^z (ζ(z) − 1)`.
        ZetaAsym => FuncSpec {
            name: "__zeta_asym", parse: &[], arity: Exactly(1), sort: Named,
            tex: Tex::Call("__zeta_asym"),
        };
    }
}

// ---------------------------------------------------------------------------
// The view and the constructor
// ---------------------------------------------------------------------------

impl ExprNode {
    /// The function this node applies and its arguments, in node order;
    /// `None` for a node that is not a function application (a number, a
    /// sum, a relation, …), for a user function, and for an `Apply` whose
    /// head names a library or internal function with an argument count it
    /// does not accept.
    ///
    /// [`Arena::func`] of the result builds this node again (for a
    /// canonical node).  The match is exhaustive over `ExprNode`: a new
    /// variant must say here whether it is a function.
    pub(crate) fn func(&self, arena: &Arena) -> Option<(Func, SmallVec<[ExprId; 4]>)> {
        let one = |f: Func, x: &ExprId| Some((f, smallvec![*x]));
        let two = |f: Func, a: &ExprId, b: &ExprId| Some((f, smallvec![*a, *b]));
        match self {
            ExprNode::Min(xs) => Some((Func::Min, xs.clone())),
            ExprNode::Max(xs) => Some((Func::Max, xs.clone())),
            ExprNode::Floor(x) => one(Func::Floor, x),
            ExprNode::Ceiling(x) => one(Func::Ceiling, x),
            ExprNode::Sin(x) => one(Func::Sin, x),
            ExprNode::Cos(x) => one(Func::Cos, x),
            ExprNode::Tan(x) => one(Func::Tan, x),
            ExprNode::Exp(x) => one(Func::Exp, x),
            ExprNode::Ln(x) => one(Func::Ln, x),
            ExprNode::Abs(x) => one(Func::Abs, x),
            ExprNode::Asin(x) => one(Func::Asin, x),
            ExprNode::Acos(x) => one(Func::Acos, x),
            ExprNode::Atan(x) => one(Func::Atan, x),
            ExprNode::Atan2(y, x) => two(Func::Atan2, y, x),
            ExprNode::Sinh(x) => one(Func::Sinh, x),
            ExprNode::Cosh(x) => one(Func::Cosh, x),
            ExprNode::Tanh(x) => one(Func::Tanh, x),
            ExprNode::Asinh(x) => one(Func::Asinh, x),
            ExprNode::Acosh(x) => one(Func::Acosh, x),
            ExprNode::Atanh(x) => one(Func::Atanh, x),
            ExprNode::Sign(x) => one(Func::Sign, x),
            ExprNode::Heaviside(x) => one(Func::Heaviside, x),
            ExprNode::DiracDelta(x) => one(Func::DiracDelta, x),
            ExprNode::Re(x) => one(Func::Re, x),
            ExprNode::Im(x) => one(Func::Im, x),
            ExprNode::Conjugate(x) => one(Func::Conjugate, x),
            ExprNode::Arg(x) => one(Func::Arg, x),
            ExprNode::Gamma(x) => one(Func::Gamma, x),
            ExprNode::LogGamma(x) => one(Func::LogGamma, x),
            ExprNode::Digamma(x) => one(Func::Digamma, x),
            ExprNode::Erf(x) => one(Func::Erf, x),
            ExprNode::Erfc(x) => one(Func::Erfc, x),
            ExprNode::LambertW(x) => one(Func::LambertW0, x),
            ExprNode::Beta(a, b) => two(Func::Beta, a, b),
            ExprNode::Si(x) => one(Func::Si, x),
            ExprNode::Ci(x) => one(Func::Ci, x),
            ExprNode::Ei(x) => one(Func::Ei, x),
            ExprNode::Li(x) => one(Func::Li, x),
            ExprNode::Zeta(x) => one(Func::Zeta, x),
            ExprNode::Polygamma(n, x) => two(Func::Polygamma, n, x),
            ExprNode::KroneckerDelta(i, j) => two(Func::KroneckerDelta, i, j),
            ExprNode::Factorial(x) => one(Func::Factorial, x),
            ExprNode::Binomial(n, k) => two(Func::Binomial, n, k),
            ExprNode::Apply(head, args) => {
                let f = Func::of_head(arena, *head)?;
                f.spec()
                    .arity
                    .accepts(args.len())
                    .then(|| (f, args.iter().copied().collect()))
            }
            ExprNode::Num(_)
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
}

impl Func {
    /// The library or internal function an `Apply` head names (by its
    /// interned text, as [`Arena::lib_fn`] resolves library names).
    fn of_head(arena: &Arena, head: SymbolId) -> Option<Func> {
        let name = arena.symbol_name(head);
        match LibFn::from_name(name) {
            Some(lib) => Some(Func::from(lib)),
            None => Func::internal_named(name),
        }
    }
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
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "used by the tests over `Func::ALL`; callers migrate to it in wave 2 (W6)"
        )
    )]
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
                let head = self.symbols.intern(f.spec().name);
                Ok(self.intern(ExprNode::Apply(head, args.iter().copied().collect())))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;
    use crate::api::expr::Ex;

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
                    expected.extend_from_slice(a.sort_key(s.raw_id()).as_bytes());
                }
                assert_eq!(a.sort_key(e.raw_id()).as_bytes(), &expected[..], "{f:?}");
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

    /// User functions, wrong-arity library calls and non-functions have no
    /// `Func`.
    #[test]
    fn view_of_non_functions_is_none() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let three = ctx.int(3);
        let user = ctx.apply("f", std::slice::from_ref(&x)).unwrap();
        let shi2 = ctx.apply("Shi", &[x.clone(), three]).unwrap();
        let sum = &x + 1;
        ctx.with_arena_mut(|a| {
            for e in [&user, &shi2, &sum, &x] {
                let node = a.node(e.raw_id()).clone();
                assert_eq!(node.func(a), None, "{node:?}");
            }
        });
    }
}
