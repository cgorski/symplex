//! Core node types for the symplex symbolic expression graph.
//!
//! Every expression in symplex is represented as a node in a directed acyclic
//! graph (DAG).  Nodes are identified by [`ExprId`] handles that index into an
//! arena held by the expression context.  The actual payload of each node is
//! described by the [`ExprNode`] enum.
//!
//! Numeric literals and symbolic names are stored in separate side‐tables and
//! referenced via [`NumId`] and [`SymbolId`] respectively, keeping the core
//! node type small and cheap to clone.
//!
//! # Shape
//!
//! Which operands of a node are children, how many there are, how to
//! rebuild it with new children ([`ExprNode::map_children`]), how to move it
//! to another arena ([`ExprNode::remap_ids`]) and which operand a binder
//! binds are generated from the enum declaration by
//! `#[derive(NodeShape)]` (symplex-macros): an `ExprId` field is a child,
//! a `SmallVec` of ids (or of id pairs) is a list of children, a
//! `#[payload] ExprId` is an id that is not a child, and the
//! `#[binder(..)]` / `#[var_slot(..)]` attributes are the binder tables of
//! `walk::binder` and `walk::var_slot`.  A new variant therefore touches
//! the declaration and its semantic code only.  The sort key
//! (`sort_key.rs`) is still written by hand: its bytes are byte-identity
//! critical.

use smallvec::SmallVec;
use std::fmt;
use symplex_macros::NodeShape;

use crate::base::func::InternalFn;
use crate::base::interval::IntervalKind;
use crate::base::libfn::LibFn;
use crate::base::walk::Binder;

// ---------------------------------------------------------------------------
// Id newtypes
// ---------------------------------------------------------------------------

/// Opaque handle that identifies an expression node inside an expression
/// context.
///
/// Ids have no order: their numbers are allocation order, which depends on
/// the context's history.  Order nodes canonically with
/// `Arena::canonical_cmp`; where an arbitrary but fixed order is wanted
/// (dedup, set identity) and never reaches output, use `ArenaOrder`.  The
/// number is private to `base`; elsewhere [`index`](Self::index) gives it
/// for hashing and memo keys only (the `test_id_order` ratchet).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(pub(in crate::base) u32);

impl ExprId {
    /// The allocation number, for hashing, memo keys and side tables
    /// indexed by id — never for ordering: it depends on everything the
    /// context built before.
    pub(crate) const fn index(self) -> u32 {
        self.0
    }

    /// The id with allocation number `i` (which the caller checks against
    /// the arena's `node_count`).
    pub(crate) const fn from_index(i: u32) -> ExprId {
        ExprId(i)
    }
}

impl fmt::Debug for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

/// Index into the numeric literal side‐table of an expression context.
///
/// No order (allocation order); compare the values instead.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NumId(pub(in crate::base) u32);

impl fmt::Debug for NumId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Index into the symbol name table of an expression context.
///
/// No order (creation order); compare the names instead.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub(in crate::base) u32);

impl fmt::Debug for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// Identifies an expression context (arena).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CtxId(pub u32);

impl fmt::Debug for CtxId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ctx{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ExprNode
// ---------------------------------------------------------------------------

/// The payload of a single node in the expression DAG.
///
/// Variants fall into the following categories:
///
/// * **Atoms** – leaves that carry no child expressions: `Num`, `Symbol`,
///   the mathematical constants (`Pi`, `E`, `ImaginaryUnit`, `EulerGamma`,
///   `Catalan`, `GoldenRatio`, `PhysicalConstant`), the special values
///   (`Infinity`, `NegInfinity`, `ComplexInfinity`, `NaN`), the boolean
///   atoms (`BoolTrue`, `BoolFalse`) and the set atoms (`EmptySet`,
///   `UniversalSet`).
/// * **N-ary arithmetic** – `Add`, `Mul`, `Min`, `Max`.
/// * **Binary arithmetic** – `Pow`.
/// * **Elementary functions** – `Neg`, `Floor`, `Ceiling`, trigonometric,
///   hyperbolic and their inverses, `Exp`, `Ln`, `Abs`, `Sign`,
///   `Heaviside`, `DiracDelta`.
/// * **Complex analysis** – `Re`, `Im`, `Conjugate`, `Arg`.
/// * **Special functions** – `Gamma`, `LogGamma`, `Digamma`, `Polygamma`,
///   `Erf`, `Erfc`, `LambertW`, `Beta`, `Si`, `Ci`, `Ei`, `Li`, `Zeta`,
///   `KroneckerDelta`.
/// * **Combinatorial** – `Factorial`, `Binomial`.
/// * **Boolean / relational / logical** – `Gt`, `Ge`, `Eq_`, `Ne`, `And`,
///   `Or`, `Not`, `Piecewise`.
/// * **Function application** – `Apply` (user-defined functions,
///   identified by name), `Lib` (library functions) and `Internal` (the
///   limit engine's internal functions).
/// * **Calculus / formal (unevaluated) forms** – `Derivative`, `Integral`,
///   `Sum`, `Product_`, `Limit`, `Series`, `LaplaceTransform`,
///   `InverseLaplaceTransform`, `Residue`, `DSolve`, `ConditionSet`, `Subs`.
/// * **Algebraic answers** – `RootOf`, `RootSum` (complete closed-form
///   descriptions of polynomial roots; *not* unevaluated).
/// * **Sets** – `Interval`, `FiniteSet`, `SetUnion`, `SetIntersection`,
///   `SetComplement`, `ImageSet` (the image of the integers under a
///   function: periodic solution families).
///
/// The shape methods ([`children`](Self::children),
/// [`for_each_child`](Self::for_each_child),
/// [`child_count`](Self::child_count), [`is_atom`](Self::is_atom), and the
/// crate-internal rebuild, remap and binder views) are generated from this
/// declaration (see the module documentation).
#[derive(Clone, PartialEq, Eq, Hash, NodeShape)]
pub enum ExprNode {
    // -- atoms ---------------------------------------------------------------
    /// A numeric literal.  The actual value lives in the number side‐table
    /// and is looked up via the contained `NumId`.
    Num(NumId),

    /// A symbolic name (variable, constant, or function name).  The actual
    /// string is stored in the symbol table and looked up via `SymbolId`.
    Symbol(SymbolId),

    /// The mathematical constant π ≈ 3.14159…
    Pi,

    /// Euler's number *e* ≈ 2.71828…
    E,

    /// The imaginary unit *i*, satisfying *i*² = −1.
    ImaginaryUnit,

    /// The Euler–Mascheroni constant γ ≈ 0.57721 56649…
    ///
    /// Whether γ is rational or irrational is an open problem, so the
    /// assumption system deliberately leaves those properties unknown.
    EulerGamma,

    /// Catalan's constant G = Σ (−1)ⁿ/(2n+1)² ≈ 0.91596 55941…
    ///
    /// Irrationality of G is unproven; only positivity, realness and
    /// finiteness are asserted.
    Catalan,

    /// The golden ratio φ = (1 + √5)/2 ≈ 1.61803 39887…
    ///
    /// φ is algebraic (root of x² − x − 1) and irrational.
    GoldenRatio,

    /// A named physical constant with a known exact value.
    /// Displays as name, evaluates to value. E.g., speed of light, Planck's constant.
    /// An atom: the value is a payload, not a child.
    PhysicalConstant(SymbolId, #[payload] ExprId),

    /// Positive infinity (+∞).
    Infinity,

    /// Negative infinity (−∞).
    NegInfinity,

    /// Complex infinity (∞̃) — magnitude is infinite, direction is undefined.
    ComplexInfinity,

    /// Not‐a‐number, representing an indeterminate or undefined result.
    NaN,

    // -- n‐ary operators -----------------------------------------------------
    /// An n‐ary sum: `a + b + c + …`
    Add(SmallVec<[ExprId; 6]>),

    /// An n‐ary product: `a · b · c · …`
    Mul(SmallVec<[ExprId; 6]>),

    // -- n-ary min/max -------------------------------------------------------
    /// N-ary minimum: min(a, b, c, ...).
    Min(SmallVec<[ExprId; 4]>),

    /// N-ary maximum: max(a, b, c, ...).
    Max(SmallVec<[ExprId; 4]>),

    // -- binary operators ----------------------------------------------------
    /// Exponentiation: `base ^ exponent`.
    Pow(ExprId, ExprId),

    // -- unary operators -----------------------------------------------------
    /// Unary arithmetic negation: `−x`.
    Neg(ExprId),

    /// Floor function: ⌊x⌋ (greatest integer ≤ x).
    Floor(ExprId),

    /// Ceiling function: ⌈x⌉ (least integer ≥ x).
    Ceiling(ExprId),

    /// Sine function: `sin(x)`.
    Sin(ExprId),

    /// Cosine function: `cos(x)`.
    Cos(ExprId),

    /// Tangent function: `tan(x)`.
    Tan(ExprId),

    /// Natural exponential function: `eˣ`.
    Exp(ExprId),

    /// Natural logarithm: `ln(x)`.
    Ln(ExprId),

    /// Absolute value (or complex modulus): `|x|`.
    Abs(ExprId),

    /// Inverse sine: `asin(x)` (arcsin).
    Asin(ExprId),

    /// Inverse cosine: `acos(x)` (arccos).
    Acos(ExprId),

    /// Inverse tangent: `atan(x)` (arctan).
    Atan(ExprId),

    /// Two-argument arctangent: `atan2(y, x)` — gives the angle in (-π, π].
    /// Correctly handles all four quadrants, unlike `atan(y/x)`.
    Atan2(ExprId, ExprId),

    /// Hyperbolic sine: `sinh(x)`.
    Sinh(ExprId),

    /// Hyperbolic cosine: `cosh(x)`.
    Cosh(ExprId),

    /// Hyperbolic tangent: `tanh(x)`.
    Tanh(ExprId),

    /// Inverse hyperbolic sine: `asinh(x)`.
    Asinh(ExprId),

    /// Inverse hyperbolic cosine: `acosh(x)`.
    Acosh(ExprId),

    /// Inverse hyperbolic tangent: `atanh(x)`.
    Atanh(ExprId),

    /// Sign function: `sign(x)` = 1 if x > 0, -1 if x < 0, 0 if x = 0.
    Sign(ExprId),

    /// Heaviside step function: H(x) = 0 for x<0, 1/2 for x=0, 1 for x>0.
    Heaviside(ExprId),

    /// Dirac delta distribution: δ(x) = 0 for x≠0, symbolic at x=0.
    DiracDelta(ExprId),

    // -- complex analysis ----------------------------------------------------
    /// Real part of a complex quantity: `re(z)`.
    ///
    /// Always real-valued. Only constructed when the real part cannot be
    /// determined from the structure of `z` and the assumption system.
    Re(ExprId),

    /// Imaginary part of a complex quantity: `im(z)`, a real number such
    /// that `z = re(z) + i·im(z)`.
    Im(ExprId),

    /// Complex conjugate: `conjugate(z)`.
    ///
    /// Distributes over `Add`/`Mul`/integer `Pow` and commutes with
    /// real-analytic functions at construction; the node itself is only
    /// produced for arguments whose realness is unknown.
    Conjugate(ExprId),

    /// Principal complex argument `arg(z) ∈ (−π, π]`.
    Arg(ExprId),

    // -- special functions ---------------------------------------------------
    /// Gamma function: Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt.
    Gamma(ExprId),

    /// Log-gamma function: ln(Γ(x)).
    LogGamma(ExprId),

    /// Digamma function: ψ(x) = Γ'(x)/Γ(x).
    Digamma(ExprId),

    /// Error function: erf(x) = 2/√π ∫₀ˣ e^(-t²) dt.
    Erf(ExprId),

    /// Complementary error function: erfc(x) = 1 - erf(x).
    Erfc(ExprId),

    /// Lambert W function (principal branch): W(x)·exp(W(x)) = x.
    LambertW(ExprId),

    /// Beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b).
    Beta(ExprId, ExprId),

    /// Sine integral: Si(x) = ∫₀ˣ sin(t)/t dt.
    Si(ExprId),

    /// Cosine integral: Ci(x) = γ + ln(x) + ∫₀ˣ (cos(t) − 1)/t dt.
    Ci(ExprId),

    /// Exponential integral (Cauchy principal value):
    /// Ei(x) = −∫_{−x}^{∞} e^{−t}/t dt = γ + ln|x| + Σ_{k≥1} xᵏ/(k·k!).
    Ei(ExprId),

    /// Logarithmic integral: li(x) = ∫₀ˣ dt/ln(t) = Ei(ln x).
    Li(ExprId),

    /// Riemann zeta function ζ(s).
    Zeta(ExprId),

    /// Polygamma function ψ⁽ⁿ⁾(x) = dⁿ⁺¹/dxⁿ⁺¹ ln Γ(x): `Polygamma(n, x)`.
    ///
    /// `Polygamma(0, x)` is canonicalised to [`Digamma`](ExprNode::Digamma).
    Polygamma(ExprId, ExprId),

    /// Kronecker delta δᵢⱼ = 1 if i = j, else 0: `KroneckerDelta(i, j)`.
    ///
    /// Arguments are stored in canonical (sorted) order since δ is symmetric.
    KroneckerDelta(ExprId, ExprId),

    // -- combinatorial -------------------------------------------------------
    /// Factorial: `n!`
    Factorial(ExprId),

    /// Binomial coefficient: `C(n, k)` = n! / (k! * (n-k)!)
    Binomial(ExprId, ExprId),

    // -- boolean atoms -------------------------------------------------------
    /// Boolean true.
    BoolTrue,
    /// Boolean false.
    BoolFalse,

    // -- relational operators ------------------------------------------------
    /// Greater than: `a > b`.
    Gt(ExprId, ExprId),
    /// Greater than or equal: `a >= b`.
    Ge(ExprId, ExprId),
    /// Mathematical equality test: `a == b` (boolean-valued).
    Eq_(ExprId, ExprId),
    /// Not equal: `a != b`.
    Ne(ExprId, ExprId),

    // -- logical connectives -------------------------------------------------
    /// Logical conjunction (n-ary): `a && b && c`.
    And(SmallVec<[ExprId; 6]>),
    /// Logical disjunction (n-ary): `a || b || c`.
    Or(SmallVec<[ExprId; 6]>),
    /// Logical negation: `!a`.
    Not(ExprId),

    // -- piecewise -----------------------------------------------------------
    /// Piecewise function: each pair is (value, condition).
    Piecewise(SmallVec<[(ExprId, ExprId); 3]>),

    // -- composite forms -----------------------------------------------------
    /// Application of a user-defined function, identified by its name, to a
    /// list of argument expressions (`Context::apply("f", …)`; nothing is
    /// known about it).  The only other heads are the reserved `__…` names
    /// of the series engine's internal asymptotic functions, which no user
    /// entry point accepts.
    Apply(SymbolId, SmallVec<[ExprId; 2]>),

    /// Application of a library function ([`LibFn`]: Bessel functions,
    /// orthogonal polynomials, `erfi`, …) to its arguments, identified by
    /// the function, not by a name.  Built only by `Arena::lib_apply` and
    /// `Arena::func`, so the argument count is one the function accepts.
    /// Until 0.49 library functions were `Apply` nodes resolved by their
    /// name, so a user function named `bell` evaluated as the Bell numbers.
    Lib(LibFn, SmallVec<[ExprId; 2]>),

    /// Formal derivative: d/d(var) of an expression.
    ///
    /// `Derivative(body, var)` represents ∂/∂`var` `body`.
    #[var_slot(var = 1, scoped(0))]
    Derivative(ExprId, ExprId),

    /// Formal indefinite integral: ∫ expr d(var).
    ///
    /// `Integral(body, var)` represents ∫ `body` d`var`.
    #[var_slot(var = 1, scoped(0))]
    Integral(ExprId, ExprId),

    /// Formal definite integral: ∫ₗₒʰⁱ expr d(var).
    ///
    /// `DefiniteIntegral(body, var, lo, hi)` represents `∫_lo^hi body dvar`.
    /// It is the unevaluated form returned by definite integration when no
    /// closed form can be established, so that the bounds are preserved for
    /// display, differentiation (Leibniz rule) and numeric quadrature.
    /// `var` is bound inside `body` only; `lo` and `hi` are in the outer
    /// scope.
    #[binder(var = 1, scoped(0), outer(2, 3))]
    DefiniteIntegral(ExprId, ExprId, ExprId, ExprId),

    /// Symbolic summation: Sum(body, var, lower, upper).
    #[binder(var = 1, scoped(0), outer(2, 3))]
    Sum(ExprId, ExprId, ExprId, ExprId),

    /// Symbolic product: Product(body, var, lower, upper).
    #[binder(var = 1, scoped(0), outer(2, 3))]
    Product_(ExprId, ExprId, ExprId, ExprId),

    /// Formal limit: lim_{var → point} body.
    #[binder(var = 1, scoped(0), outer(2))]
    Limit(ExprId, ExprId, ExprId),

    /// Formal series expansion: Series(body, var, point, order).
    #[var_slot(var = 1, scoped(0), outer(2, 3))]
    Series(ExprId, ExprId, ExprId, ExprId),

    /// Formal Laplace transform: ℒ{body}(t → s).
    #[binder(var = 1, scoped(0), outer(2))]
    LaplaceTransform(ExprId, ExprId, ExprId),

    /// Formal inverse Laplace transform: ℒ⁻¹{body}(s → t).
    #[binder(var = 1, scoped(0), outer(2))]
    InverseLaplaceTransform(ExprId, ExprId, ExprId),

    /// Formal residue: Res_{var=point} body.
    #[binder(var = 1, scoped(0), outer(2))]
    Residue(ExprId, ExprId, ExprId),

    /// Root of a polynomial: the index-th root of poly.
    ///
    /// `RootOf(poly, var, index)` where:
    /// - `poly` is a polynomial in `var` (it may have other symbols as
    ///   parameters: `RootOf(x⁵ − a·x + 1, x, 0)` is a function of `a`);
    /// - `var` is the bound variable symbol, as for `RootSum` (SymPy's
    ///   `CRootOf` likewise keeps its polynomial's generator);
    /// - `index` selects the root (Aberth ordering: by real part, then
    ///   imaginary part).
    #[binder(var = 1, scoped(0), outer(2))]
    RootOf(ExprId, ExprId, ExprId),

    /// Sum over roots of a polynomial: Σ_{α: poly(α)=0} body(α, x).
    ///
    /// Used by Rothstein-Trager when the resultant has irreducible factors
    /// of degree ≥ 5 whose roots cannot be expressed in radicals.
    ///
    /// `RootSum(poly, body, sumvar)` where:
    /// - `poly` is the polynomial expression whose roots are summed over
    /// - `body` is the expression evaluated at each root (contains `sumvar`)
    /// - `sumvar` is the bound summation variable symbol
    #[binder(var = 2, scoped(0, 1))]
    RootSum(ExprId, ExprId, ExprId),

    /// Formal ODE solution: DSolve(expr=0, func, var).
    #[var_slot(var = 2, scoped(0, 1))]
    DSolve(ExprId, ExprId, ExprId),

    /// Condition set: {var | condition}.
    #[binder(var = 0, scoped(1))]
    ConditionSet(ExprId, ExprId),

    // -- set atoms --------------------------------------------------------
    /// The empty set ∅.
    EmptySet,
    /// The universal set (all values).
    UniversalSet,

    // -- set constructors -------------------------------------------------
    /// An interval between two endpoints, with the kind saying which
    /// endpoints belong to it.  Its sort-key byte (and the `flags` of the
    /// JSON `ExprTree::Interval`) is the kind's flag byte of symplex ≤ 0.48
    /// (`IntervalKind::flag_bits`: bit 0 = lower end open, bit 1 = upper
    /// end open).
    Interval(ExprId, ExprId, IntervalKind),

    /// A finite set of elements {a, b, c, ...}, sorted and deduplicated.
    FiniteSet(SmallVec<[ExprId; 4]>),

    /// Union of sets: A ∪ B ∪ C ∪ ...
    SetUnion(SmallVec<[ExprId; 4]>),

    /// Intersection of sets: A ∩ B ∩ C ∩ ...
    SetIntersection(SmallVec<[ExprId; 4]>),

    /// Set complement (relative): A \ B.
    SetComplement(ExprId, ExprId),

    // -- evaluation at a point (declared last: appending keeps the
    //    discriminants, hence the hashes, of every older variant) ----------
    /// Unevaluated substitution: `body` with `var` set to `point`.
    ///
    /// `Subs(body, var, point)` binds `var` in `body`; `point` is in the
    /// enclosing scope.  It is the value at a point of a function of `var`
    /// that cannot be substituted into directly: `f′(0)` for an undefined
    /// `f` is `Subs(Derivative(f(x), x), x, 0)`, as in SymPy.  Substitution
    /// builds one only around a `Derivative`, `Integral`, `Series` or
    /// `DSolve` whose variable is replaced by something that is not a fresh
    /// symbol, and `diff` for a partial derivative of an undefined function
    /// in an argument that is not a lone symbol (`∂f/∂u` at `u = x²`).
    #[binder(var = 1, scoped(0), outer(2))]
    Subs(ExprId, ExprId, ExprId),

    // -- image of the integers (declared after `Subs`: appending keeps the
    //    discriminants, hence the hashes, of every older variant) ----------
    /// Image set over the integers: `ImageSet(var, body)` is
    /// `{body(var) : var ∈ ℤ}`, SymPy's `ImageSet(Lambda(var, body),
    /// Integers)`.
    ///
    /// `var` (a `Symbol`) is bound in `body`.  The periodic solution
    /// families of an equation are image sets: `sin x = 0` has the solution
    /// set `{n·π : n ∈ ℤ}`.  A complete answer, not an unevaluated form;
    /// `transforms::sets::image_set` builds the canonical form.
    #[binder(var = 0, scoped(1))]
    ImageSet(ExprId, ExprId),

    // -- internal functions (declared last: appending keeps the
    //    discriminants, hence the hashes, of every older variant) ----------
    /// Application of one of the limit engine's internal asymptotic
    /// functions (`InternalFn`: `__ei_asym`, `__besselk_asym`, …, the
    /// heads of Gruntz's rewrite of `erfc`, `Ei`, `Kν`, … at infinity) to
    /// its arguments.  It exists only inside a limit computation and is
    /// built only by `Arena::func`: no name converts to it.  Until 0.49
    /// these were `Apply` nodes with reserved `__…` heads, and
    /// `Context::from_tree` of an `Apply` named `__ei_asym` built the
    /// internal function.  It prints, sorts and compares as that `Apply`
    /// did.
    Internal(InternalFn, SmallVec<[ExprId; 2]>),
}

impl IntervalKind {
    /// The kind's flag byte in the node encoding of symplex ≤ 0.48: bit 0
    /// set when the lower end is open, bit 1 when the upper end is open
    /// (`[a, b]` = 0, `(a, b]` = 1, `[a, b)` = 2, `(a, b)` = 3).  It is the
    /// byte an `Interval` node contributes to its sort key and the `flags`
    /// of `ExprTree::Interval` (the stable JSON form).
    pub(crate) const fn flag_bits(self) -> u8 {
        match self {
            IntervalKind::Closed => 0,
            IntervalKind::LeftOpen => 1,
            IntervalKind::RightOpen => 2,
            IntervalKind::Open => 3,
        }
    }

    /// The kind with the given flag byte ([`flag_bits`](Self::flag_bits));
    /// bits other than the two low ones are ignored.
    pub(crate) const fn from_flag_bits(flags: u8) -> Self {
        IntervalKind::from_open_ends(flags & 1 != 0, flags & 2 != 0)
    }
}

/// Maps the ids a node refers to — its children, a physical constant's
/// value, numbers and symbols — into another arena's id spaces: the
/// callbacks of [`ExprNode::remap_ids`] (compaction).
pub(crate) trait IdRemap {
    /// The id, in the destination, of the expression `id`.
    fn expr(&mut self, id: ExprId) -> ExprId;
    /// The id, in the destination, of the number `id`.
    fn num(&mut self, id: NumId) -> NumId;
    /// The id, in the destination, of the symbol name `id`.
    fn symbol(&mut self, id: SymbolId) -> SymbolId;
}

impl ExprNode {
    /// Returns `true` if this node is a set-valued expression.
    pub fn is_set_node(&self) -> bool {
        matches!(
            self,
            ExprNode::EmptySet
                | ExprNode::UniversalSet
                | ExprNode::Interval(..)
                | ExprNode::FiniteSet(_)
                | ExprNode::SetUnion(_)
                | ExprNode::SetIntersection(_)
                | ExprNode::SetComplement(..)
                | ExprNode::ImageSet(..)
        )
    }
}

impl fmt::Debug for ExprNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprNode::Num(id) => write!(f, "Num({id:?})"),
            ExprNode::Symbol(id) => write!(f, "Symbol({id:?})"),
            ExprNode::Pi => write!(f, "Pi"),
            ExprNode::E => write!(f, "E"),
            ExprNode::ImaginaryUnit => write!(f, "ImaginaryUnit"),
            ExprNode::EulerGamma => write!(f, "EulerGamma"),
            ExprNode::Catalan => write!(f, "Catalan"),
            ExprNode::GoldenRatio => write!(f, "GoldenRatio"),
            ExprNode::PhysicalConstant(name, val) => {
                write!(f, "PhysicalConstant({name:?}, {val:?})")
            }
            ExprNode::Infinity => write!(f, "Infinity"),
            ExprNode::NegInfinity => write!(f, "NegInfinity"),
            ExprNode::ComplexInfinity => write!(f, "ComplexInfinity"),
            ExprNode::NaN => write!(f, "NaN"),
            ExprNode::Add(ids) => f.debug_tuple("Add").field(ids).finish(),
            ExprNode::Mul(ids) => f.debug_tuple("Mul").field(ids).finish(),
            ExprNode::Pow(base, exp) => f.debug_tuple("Pow").field(base).field(exp).finish(),
            ExprNode::Neg(x) => f.debug_tuple("Neg").field(x).finish(),
            ExprNode::Sin(x) => f.debug_tuple("Sin").field(x).finish(),
            ExprNode::Cos(x) => f.debug_tuple("Cos").field(x).finish(),
            ExprNode::Tan(x) => f.debug_tuple("Tan").field(x).finish(),
            ExprNode::Exp(x) => f.debug_tuple("Exp").field(x).finish(),
            ExprNode::Ln(x) => f.debug_tuple("Ln").field(x).finish(),
            ExprNode::Abs(x) => f.debug_tuple("Abs").field(x).finish(),
            ExprNode::Asin(x) => f.debug_tuple("Asin").field(x).finish(),
            ExprNode::Acos(x) => f.debug_tuple("Acos").field(x).finish(),
            ExprNode::Atan(x) => f.debug_tuple("Atan").field(x).finish(),
            ExprNode::Atan2(y, x) => f.debug_tuple("Atan2").field(y).field(x).finish(),
            ExprNode::Sinh(x) => f.debug_tuple("Sinh").field(x).finish(),
            ExprNode::Cosh(x) => f.debug_tuple("Cosh").field(x).finish(),
            ExprNode::Tanh(x) => f.debug_tuple("Tanh").field(x).finish(),
            ExprNode::Asinh(x) => f.debug_tuple("Asinh").field(x).finish(),
            ExprNode::Acosh(x) => f.debug_tuple("Acosh").field(x).finish(),
            ExprNode::Atanh(x) => f.debug_tuple("Atanh").field(x).finish(),
            ExprNode::Floor(x) => f.debug_tuple("Floor").field(x).finish(),
            ExprNode::Ceiling(x) => f.debug_tuple("Ceiling").field(x).finish(),
            ExprNode::Min(ids) => f.debug_tuple("Min").field(ids).finish(),
            ExprNode::Max(ids) => f.debug_tuple("Max").field(ids).finish(),
            ExprNode::Sign(id) => write!(f, "Sign({id:?})"),
            ExprNode::Heaviside(x) => f.debug_tuple("Heaviside").field(x).finish(),
            ExprNode::DiracDelta(x) => f.debug_tuple("DiracDelta").field(x).finish(),
            ExprNode::Re(x) => f.debug_tuple("Re").field(x).finish(),
            ExprNode::Im(x) => f.debug_tuple("Im").field(x).finish(),
            ExprNode::Conjugate(x) => f.debug_tuple("Conjugate").field(x).finish(),
            ExprNode::Arg(x) => f.debug_tuple("Arg").field(x).finish(),
            ExprNode::Gamma(x) => f.debug_tuple("Gamma").field(x).finish(),
            ExprNode::LogGamma(x) => f.debug_tuple("LogGamma").field(x).finish(),
            ExprNode::Digamma(x) => f.debug_tuple("Digamma").field(x).finish(),
            ExprNode::Erf(x) => f.debug_tuple("Erf").field(x).finish(),
            ExprNode::Erfc(x) => f.debug_tuple("Erfc").field(x).finish(),
            ExprNode::LambertW(x) => f.debug_tuple("LambertW").field(x).finish(),
            ExprNode::Beta(a, b) => f.debug_tuple("Beta").field(a).field(b).finish(),
            ExprNode::Si(x) => f.debug_tuple("Si").field(x).finish(),
            ExprNode::Ci(x) => f.debug_tuple("Ci").field(x).finish(),
            ExprNode::Ei(x) => f.debug_tuple("Ei").field(x).finish(),
            ExprNode::Li(x) => f.debug_tuple("Li").field(x).finish(),
            ExprNode::Zeta(x) => f.debug_tuple("Zeta").field(x).finish(),
            ExprNode::Polygamma(n, x) => f.debug_tuple("Polygamma").field(n).field(x).finish(),
            ExprNode::KroneckerDelta(i, j) => {
                f.debug_tuple("KroneckerDelta").field(i).field(j).finish()
            }
            ExprNode::Factorial(id) => write!(f, "Factorial({id:?})"),
            ExprNode::Binomial(n, k) => write!(f, "Binomial({n:?}, {k:?})"),
            ExprNode::BoolTrue => write!(f, "BoolTrue"),
            ExprNode::BoolFalse => write!(f, "BoolFalse"),
            ExprNode::Gt(a, b) => f.debug_tuple("Gt").field(a).field(b).finish(),
            ExprNode::Ge(a, b) => f.debug_tuple("Ge").field(a).field(b).finish(),
            ExprNode::Eq_(a, b) => f.debug_tuple("Eq_").field(a).field(b).finish(),
            ExprNode::Ne(a, b) => f.debug_tuple("Ne").field(a).field(b).finish(),
            ExprNode::And(ids) => f.debug_tuple("And").field(ids).finish(),
            ExprNode::Or(ids) => f.debug_tuple("Or").field(ids).finish(),
            ExprNode::Not(x) => f.debug_tuple("Not").field(x).finish(),
            ExprNode::Piecewise(pairs) => {
                let mut d = f.debug_tuple("Piecewise");
                for pair in pairs {
                    d.field(pair);
                }
                d.finish()
            }
            ExprNode::Apply(sym, args) => f.debug_tuple("Apply").field(sym).field(args).finish(),
            ExprNode::Lib(lib, args) => f.debug_tuple("Lib").field(lib).field(args).finish(),
            ExprNode::Internal(g, args) => f.debug_tuple("Internal").field(g).field(args).finish(),
            ExprNode::Derivative(body, var) => {
                f.debug_tuple("Derivative").field(body).field(var).finish()
            }
            ExprNode::Integral(body, var) => {
                f.debug_tuple("Integral").field(body).field(var).finish()
            }
            ExprNode::DefiniteIntegral(body, var, lo, hi) => f
                .debug_tuple("DefiniteIntegral")
                .field(body)
                .field(var)
                .field(lo)
                .field(hi)
                .finish(),
            ExprNode::Sum(body, var, lo, hi) => f
                .debug_tuple("Sum")
                .field(body)
                .field(var)
                .field(lo)
                .field(hi)
                .finish(),
            ExprNode::Product_(body, var, lo, hi) => f
                .debug_tuple("Product_")
                .field(body)
                .field(var)
                .field(lo)
                .field(hi)
                .finish(),
            ExprNode::EmptySet => write!(f, "EmptySet"),
            ExprNode::UniversalSet => write!(f, "UniversalSet"),
            ExprNode::Interval(a, b, kind) => f
                .debug_tuple("Interval")
                .field(a)
                .field(b)
                .field(&kind.flag_bits())
                .finish(),
            ExprNode::FiniteSet(ids) => f.debug_tuple("FiniteSet").field(ids).finish(),
            ExprNode::SetUnion(ids) => f.debug_tuple("SetUnion").field(ids).finish(),
            ExprNode::SetIntersection(ids) => f.debug_tuple("SetIntersection").field(ids).finish(),
            ExprNode::SetComplement(a, b) => {
                f.debug_tuple("SetComplement").field(a).field(b).finish()
            }
            ExprNode::Limit(body, var, point) => f
                .debug_tuple("Limit")
                .field(body)
                .field(var)
                .field(point)
                .finish(),
            ExprNode::Series(body, var, point, order) => f
                .debug_tuple("Series")
                .field(body)
                .field(var)
                .field(point)
                .field(order)
                .finish(),
            ExprNode::LaplaceTransform(body, t, s) => f
                .debug_tuple("LaplaceTransform")
                .field(body)
                .field(t)
                .field(s)
                .finish(),
            ExprNode::InverseLaplaceTransform(body, s, t) => f
                .debug_tuple("InverseLaplaceTransform")
                .field(body)
                .field(s)
                .field(t)
                .finish(),
            ExprNode::Residue(body, var, point) => f
                .debug_tuple("Residue")
                .field(body)
                .field(var)
                .field(point)
                .finish(),
            ExprNode::RootOf(poly, var, idx) => f
                .debug_tuple("RootOf")
                .field(poly)
                .field(var)
                .field(idx)
                .finish(),
            ExprNode::DSolve(expr, func, var) => f
                .debug_tuple("DSolve")
                .field(expr)
                .field(func)
                .field(var)
                .finish(),
            ExprNode::RootSum(poly, body, sumvar) => f
                .debug_tuple("RootSum")
                .field(poly)
                .field(body)
                .field(sumvar)
                .finish(),
            ExprNode::ConditionSet(var, cond) => f
                .debug_tuple("ConditionSet")
                .field(var)
                .field(cond)
                .finish(),
            ExprNode::Subs(body, var, point) => f
                .debug_tuple("Subs")
                .field(body)
                .field(var)
                .field(point)
                .finish(),
            ExprNode::ImageSet(var, body) => {
                f.debug_tuple("ImageSet").field(var).field(body).finish()
            }
        }
    }
}

/// One node of every variant (several for the variable-arity ones and for
/// each [`IntervalKind`]), over the distinct children `c`: the samples of
/// the shape tests.  The tests check that every variant name occurs.
#[cfg(test)]
pub(crate) fn shape_samples(
    c: [ExprId; 8],
    num: NumId,
    sym: SymbolId,
    value: ExprId,
) -> Vec<ExprNode> {
    use ExprNode as N;
    use smallvec::smallvec;
    let [a, b, d, e, g, h, k, m] = c;
    vec![
        N::Num(num),
        N::Symbol(sym),
        N::Pi,
        N::E,
        N::ImaginaryUnit,
        N::EulerGamma,
        N::Catalan,
        N::GoldenRatio,
        N::PhysicalConstant(sym, value),
        N::Infinity,
        N::NegInfinity,
        N::ComplexInfinity,
        N::NaN,
        N::Add(smallvec![a, b]),
        N::Add(smallvec![a, b, d, e, g, h, k]),
        N::Mul(smallvec![a, b, d]),
        N::Mul(smallvec![a, b, d, e, g, h, k]),
        N::Min(smallvec![a, b]),
        N::Max(smallvec![a, b, d, e, g]),
        N::Pow(a, b),
        N::Neg(a),
        N::Floor(a),
        N::Ceiling(a),
        N::Sin(a),
        N::Cos(a),
        N::Tan(a),
        N::Exp(a),
        N::Ln(a),
        N::Abs(a),
        N::Asin(a),
        N::Acos(a),
        N::Atan(a),
        N::Atan2(a, b),
        N::Sinh(a),
        N::Cosh(a),
        N::Tanh(a),
        N::Asinh(a),
        N::Acosh(a),
        N::Atanh(a),
        N::Sign(a),
        N::Heaviside(a),
        N::DiracDelta(a),
        N::Re(a),
        N::Im(a),
        N::Conjugate(a),
        N::Arg(a),
        N::Gamma(a),
        N::LogGamma(a),
        N::Digamma(a),
        N::Erf(a),
        N::Erfc(a),
        N::LambertW(a),
        N::Beta(a, b),
        N::Si(a),
        N::Ci(a),
        N::Ei(a),
        N::Li(a),
        N::Zeta(a),
        N::Polygamma(a, b),
        N::KroneckerDelta(a, b),
        N::Factorial(a),
        N::Binomial(a, b),
        N::BoolTrue,
        N::BoolFalse,
        N::Gt(a, b),
        N::Ge(a, b),
        N::Eq_(a, b),
        N::Ne(a, b),
        N::And(smallvec![a, b]),
        N::Or(smallvec![a, b, d]),
        N::Not(a),
        N::Piecewise(smallvec![(a, b)]),
        N::Piecewise(smallvec![(a, b), (d, e), (g, h), (k, m)]),
        N::Apply(sym, smallvec![]),
        N::Apply(sym, smallvec![a]),
        N::Apply(sym, smallvec![a, b, d]),
        N::Lib(LibFn::Erfi, smallvec![a]),
        N::Lib(LibFn::BesselJ, smallvec![a, b]),
        N::Lib(LibFn::Jacobi, smallvec![a, b, d, e]),
        N::Internal(InternalFn::EiAsym, smallvec![a]),
        N::Internal(InternalFn::BesselKAsym, smallvec![a, b]),
        N::Derivative(a, b),
        N::Integral(a, b),
        N::DefiniteIntegral(a, b, d, e),
        N::Sum(a, b, d, e),
        N::Product_(a, b, d, e),
        N::Limit(a, b, d),
        N::Series(a, b, d, e),
        N::LaplaceTransform(a, b, d),
        N::InverseLaplaceTransform(a, b, d),
        N::Residue(a, b, d),
        N::RootOf(a, b, d),
        N::RootSum(a, b, d),
        N::DSolve(a, b, d),
        N::ConditionSet(a, b),
        N::EmptySet,
        N::UniversalSet,
        N::Interval(a, b, IntervalKind::Closed),
        N::Interval(a, b, IntervalKind::LeftOpen),
        N::Interval(a, b, IntervalKind::RightOpen),
        N::Interval(a, b, IntervalKind::Open),
        N::FiniteSet(smallvec![a]),
        N::FiniteSet(smallvec![a, b, d, e, g]),
        N::SetUnion(smallvec![a, b]),
        N::SetIntersection(smallvec![a, b, d]),
        N::SetComplement(a, b),
        N::Subs(a, b, d),
        N::ImageSet(a, b),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    fn ids8(base: u32) -> [ExprId; 8] {
        std::array::from_fn(|i| ExprId(base + i as u32))
    }

    fn samples() -> Vec<ExprNode> {
        shape_samples(ids8(10), NumId(3), SymbolId(4), ExprId(99))
    }

    /// The variants without children (the hand-written `is_atom` list of
    /// 0.48).
    const ATOMS: &[&str] = &[
        "Num",
        "Symbol",
        "Pi",
        "E",
        "ImaginaryUnit",
        "EulerGamma",
        "Catalan",
        "GoldenRatio",
        "PhysicalConstant",
        "Infinity",
        "NegInfinity",
        "ComplexInfinity",
        "NaN",
        "BoolTrue",
        "BoolFalse",
        "EmptySet",
        "UniversalSet",
    ];

    /// The binder table of `walk::binder` (0.48, hand-written): variant,
    /// position of the variable, positions it is bound in, outer positions.
    const BINDERS: &[(&str, usize, &[usize], &[usize])] = &[
        ("Sum", 1, &[0], &[2, 3]),
        ("Product_", 1, &[0], &[2, 3]),
        ("DefiniteIntegral", 1, &[0], &[2, 3]),
        ("RootSum", 2, &[0, 1], &[]),
        ("ConditionSet", 0, &[1], &[]),
        ("ImageSet", 0, &[1], &[]),
        ("RootOf", 1, &[0], &[2]),
        ("Limit", 1, &[0], &[2]),
        ("Residue", 1, &[0], &[2]),
        ("Subs", 1, &[0], &[2]),
        ("LaplaceTransform", 1, &[0], &[2]),
        ("InverseLaplaceTransform", 1, &[0], &[2]),
    ];

    /// The table of `walk::var_slot` (0.48, hand-written).
    const VAR_SLOTS: &[(&str, usize, &[usize], &[usize])] = &[
        ("Derivative", 1, &[0], &[]),
        ("Integral", 1, &[0], &[]),
        ("Series", 1, &[0], &[2, 3]),
        ("DSolve", 2, &[0, 1], &[]),
    ];

    #[test]
    fn shape_samples_cover_every_variant() {
        let names: std::collections::BTreeSet<&str> =
            samples().iter().map(ExprNode::variant_name).collect();
        let all: std::collections::BTreeSet<&str> =
            ExprNode::VARIANT_NAMES.iter().copied().collect();
        assert_eq!(names, all);
        assert_eq!(all.len(), ExprNode::VARIANT_NAMES.len());
        for n in samples() {
            let debug = format!("{n:?}");
            let head = debug.split('(').next().unwrap_or_default();
            assert_eq!(n.variant_name(), head, "{debug}");
        }
    }

    #[test]
    fn generated_children_count_and_atoms_agree() {
        for n in samples() {
            let mut seen: SmallVec<[ExprId; 6]> = SmallVec::new();
            n.for_each_child(|c| seen.push(c));
            assert_eq!(seen, n.children(), "{n:?}");
            assert_eq!(n.child_count(), seen.len(), "{n:?}");
            assert_eq!(n.is_atom(), ATOMS.contains(&n.variant_name()), "{n:?}");
            if n.is_atom() {
                assert!(seen.is_empty(), "{n:?}");
            }
        }
        // A physical constant's value is a payload, not a child.
        let pc = ExprNode::PhysicalConstant(SymbolId(1), ExprId(5));
        assert!(pc.children().is_empty() && pc.is_atom());
        // Piecewise pairs are flattened value, condition, value, ...
        let pw = ExprNode::Piecewise(smallvec![(ExprId(1), ExprId(2)), (ExprId(3), ExprId(4))]);
        assert_eq!(
            pw.children().as_slice(),
            &[ExprId(1), ExprId(2), ExprId(3), ExprId(4)]
        );
    }

    #[test]
    fn map_children_maps_each_child_in_order_and_keeps_payload() {
        let shift = |c: ExprId| ExprId(c.0 + 100);
        for n in samples() {
            assert_eq!(n.map_children(|c| c), None, "{n:?}");
            let mut order = Vec::new();
            let mapped = n.map_children(|c| {
                order.push(c);
                shift(c)
            });
            assert_eq!(order.as_slice(), n.children().as_slice(), "{n:?}");
            match mapped {
                None => assert!(n.child_count() == 0, "{n:?}"),
                Some(m) => {
                    assert_eq!(m.variant_name(), n.variant_name());
                    let want: SmallVec<[ExprId; 6]> = n.children().into_iter().map(shift).collect();
                    assert_eq!(m.children(), want, "{n:?}");
                    // Payloads (interval kind, function name) survive.
                    assert_eq!(m.map_children(|c| ExprId(c.0 - 100)), Some(n.clone()));
                }
            }
        }
    }

    struct Offset(i64);

    impl IdRemap for Offset {
        fn expr(&mut self, id: ExprId) -> ExprId {
            ExprId((i64::from(id.0) + self.0) as u32)
        }
        fn num(&mut self, id: NumId) -> NumId {
            NumId((i64::from(id.0) + self.0) as u32)
        }
        fn symbol(&mut self, id: SymbolId) -> SymbolId {
            SymbolId((i64::from(id.0) + self.0) as u32)
        }
    }

    #[test]
    fn remap_ids_maps_every_id_and_nothing_else() {
        for n in samples() {
            let m = n.remap_ids(&mut Offset(1000));
            assert_eq!(m.variant_name(), n.variant_name());
            let want: SmallVec<[ExprId; 6]> = n
                .children()
                .into_iter()
                .map(|c| ExprId(c.0 + 1000))
                .collect();
            assert_eq!(m.children(), want, "{n:?}");
            assert_eq!(m.remap_ids(&mut Offset(-1000)), n, "{n:?}");
        }
        let pc = ExprNode::PhysicalConstant(SymbolId(1), ExprId(5));
        assert_eq!(
            pc.remap_ids(&mut Offset(7)),
            ExprNode::PhysicalConstant(SymbolId(8), ExprId(12))
        );
        assert_eq!(
            ExprNode::Num(NumId(2)).remap_ids(&mut Offset(7)),
            ExprNode::Num(NumId(9))
        );
        let iv = ExprNode::Interval(ExprId(1), ExprId(2), IntervalKind::LeftOpen);
        assert_eq!(
            iv.remap_ids(&mut Offset(1)),
            ExprNode::Interval(ExprId(2), ExprId(3), IntervalKind::LeftOpen)
        );
    }

    fn check_layouts(
        table: &[(&str, usize, &[usize], &[usize])],
        view: impl Fn(&ExprNode) -> Option<Binder>,
        rebuild: impl Fn(&ExprNode, ExprId, &[ExprId], &[ExprId]) -> Option<ExprNode>,
    ) {
        for n in samples() {
            let kids = n.children();
            let row = table.iter().find(|r| r.0 == n.variant_name());
            match (view(&n), row) {
                (None, None) => {
                    assert_eq!(
                        rebuild(&n, kids.first().copied().unwrap_or(ExprId(0)), &[], &[]),
                        None
                    );
                }
                (Some(b), Some(&(_, var, scoped, outer))) => {
                    assert_eq!(b.var, kids[var], "{n:?}");
                    let s: Vec<ExprId> = scoped.iter().map(|&i| kids[i]).collect();
                    let o: Vec<ExprId> = outer.iter().map(|&i| kids[i]).collect();
                    assert_eq!(b.scoped.as_slice(), s.as_slice(), "{n:?}");
                    assert_eq!(b.outer.as_slice(), o.as_slice(), "{n:?}");
                    assert_eq!(n.var_operand(), Some(b.var), "{n:?}");
                    assert_eq!(rebuild(&n, b.var, &b.scoped, &b.outer), Some(n.clone()));
                    // A new variable lands in the variable's position.
                    let renamed = rebuild(&n, ExprId(77), &b.scoped, &b.outer);
                    assert_eq!(renamed.and_then(|r| r.var_operand()), Some(ExprId(77)));
                    // Wrong operand counts are refused.
                    assert_eq!(
                        rebuild(&n, b.var, &b.outer, &b.scoped).is_some(),
                        scoped.len() == outer.len()
                    );
                }
                (got, want) => panic!("{n:?}: layout {got:?}, table {want:?}"),
            }
        }
    }

    #[test]
    fn binder_and_var_slot_layouts_are_the_048_tables() {
        check_layouts(
            BINDERS,
            ExprNode::binder_operands,
            ExprNode::with_binder_operands,
        );
        check_layouts(
            VAR_SLOTS,
            ExprNode::var_slot_operands,
            ExprNode::with_var_slot_operands,
        );
        for n in samples() {
            let both = n.binder_operands().is_some() && n.var_slot_operands().is_some();
            assert!(!both, "{n:?}");
            if n.binder_operands().is_none() && n.var_slot_operands().is_none() {
                assert_eq!(n.var_operand(), None, "{n:?}");
            }
        }
    }

    #[test]
    fn interval_kind_flag_bits_are_the_048_flags() {
        use IntervalKind::*;
        for (kind, bits) in [(Closed, 0u8), (LeftOpen, 1), (RightOpen, 2), (Open, 3)] {
            assert_eq!(kind.flag_bits(), bits);
            assert_eq!(IntervalKind::from_flag_bits(bits), kind);
            assert_eq!(kind.lower_open(), bits & 1 != 0);
            assert_eq!(kind.upper_open(), bits & 2 != 0);
        }
        // Only the two low bits mean anything.
        assert_eq!(IntervalKind::from_flag_bits(4), Closed);
        assert_eq!(IntervalKind::from_flag_bits(0xff), Open);
        assert_eq!(
            format!("{:?}", ExprNode::Interval(ExprId(1), ExprId(2), RightOpen)),
            "Interval(e1, e2, 2)"
        );
    }

    #[test]
    fn debug_expr_id() {
        let id = ExprId(42);
        assert_eq!(format!("{id:?}"), "e42");
    }

    #[test]
    fn debug_num_id() {
        let id = NumId(7);
        assert_eq!(format!("{id:?}"), "n7");
    }

    #[test]
    fn debug_symbol_id() {
        let id = SymbolId(3);
        assert_eq!(format!("{id:?}"), "s3");
    }

    #[test]
    fn debug_ctx_id() {
        let id = CtxId(0);
        assert_eq!(format!("{id:?}"), "ctx0");
    }

    #[test]
    fn atom_has_no_children() {
        let node = ExprNode::Pi;
        assert!(node.is_atom());
        assert!(node.children().is_empty());
    }

    #[test]
    fn num_is_atom() {
        let node = ExprNode::Num(NumId(0));
        assert!(node.is_atom());
        assert!(node.children().is_empty());
    }

    #[test]
    fn symbol_is_atom() {
        let node = ExprNode::Symbol(SymbolId(1));
        assert!(node.is_atom());
    }

    #[test]
    fn special_values_are_atoms() {
        for node in [
            ExprNode::Infinity,
            ExprNode::NegInfinity,
            ExprNode::ComplexInfinity,
            ExprNode::NaN,
            ExprNode::ImaginaryUnit,
            ExprNode::E,
        ] {
            assert!(node.is_atom(), "{node:?} should be an atom");
            assert!(node.children().is_empty());
        }
    }

    #[test]
    fn add_children() {
        let ids: SmallVec<[ExprId; 6]> = smallvec![ExprId(1), ExprId(2), ExprId(3)];
        let node = ExprNode::Add(ids.clone());
        assert!(!node.is_atom());
        assert_eq!(node.children(), ids);
    }

    #[test]
    fn mul_children() {
        let ids: SmallVec<[ExprId; 6]> = smallvec![ExprId(4), ExprId(5)];
        let node = ExprNode::Mul(ids.clone());
        assert!(!node.is_atom());
        assert_eq!(node.children(), ids);
    }

    #[test]
    fn pow_children() {
        let node = ExprNode::Pow(ExprId(10), ExprId(20));
        assert!(!node.is_atom());
        let kids = node.children();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], ExprId(10));
        assert_eq!(kids[1], ExprId(20));
    }

    #[test]
    fn unary_children() {
        for node in [
            ExprNode::Neg(ExprId(1)),
            ExprNode::Sin(ExprId(1)),
            ExprNode::Cos(ExprId(1)),
            ExprNode::Tan(ExprId(1)),
            ExprNode::Exp(ExprId(1)),
            ExprNode::Ln(ExprId(1)),
            ExprNode::Abs(ExprId(1)),
        ] {
            assert!(!node.is_atom());
            let kids = node.children();
            assert_eq!(kids.len(), 1, "{node:?} should have exactly 1 child");
            assert_eq!(kids[0], ExprId(1));
        }
    }

    #[test]
    fn apply_children() {
        let args: SmallVec<[ExprId; 2]> = smallvec![ExprId(5), ExprId(6)];
        let node = ExprNode::Apply(SymbolId(0), args);
        assert!(!node.is_atom());
        let kids = node.children();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], ExprId(5));
        assert_eq!(kids[1], ExprId(6));
    }

    #[test]
    fn derivative_children() {
        let node = ExprNode::Derivative(ExprId(3), ExprId(7));
        assert!(!node.is_atom());
        let kids = node.children();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], ExprId(3));
        assert_eq!(kids[1], ExprId(7));
    }

    #[test]
    fn integral_children() {
        let node = ExprNode::Integral(ExprId(8), ExprId(9));
        assert!(!node.is_atom());
        let kids = node.children();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], ExprId(8));
        assert_eq!(kids[1], ExprId(9));
    }

    #[test]
    fn named_constants_are_atoms() {
        for node in [
            ExprNode::EulerGamma,
            ExprNode::Catalan,
            ExprNode::GoldenRatio,
        ] {
            assert!(node.is_atom(), "{node:?} should be an atom");
            assert!(node.children().is_empty());
            assert_eq!(node.child_count(), 0);
            let mut count = 0;
            node.for_each_child(|_| count += 1);
            assert_eq!(count, 0);
        }
        assert_eq!(format!("{:?}", ExprNode::EulerGamma), "EulerGamma");
        assert_eq!(format!("{:?}", ExprNode::Catalan), "Catalan");
        assert_eq!(format!("{:?}", ExprNode::GoldenRatio), "GoldenRatio");
    }

    #[test]
    fn complex_and_special_unary_children() {
        for node in [
            ExprNode::Re(ExprId(3)),
            ExprNode::Im(ExprId(3)),
            ExprNode::Conjugate(ExprId(3)),
            ExprNode::Arg(ExprId(3)),
            ExprNode::Si(ExprId(3)),
            ExprNode::Ci(ExprId(3)),
            ExprNode::Ei(ExprId(3)),
            ExprNode::Li(ExprId(3)),
            ExprNode::Zeta(ExprId(3)),
        ] {
            assert!(!node.is_atom());
            assert_eq!(node.child_count(), 1);
            let kids = node.children();
            assert_eq!(kids.len(), 1, "{node:?} should have exactly 1 child");
            assert_eq!(kids[0], ExprId(3));
            let mut seen = Vec::new();
            node.for_each_child(|c| seen.push(c));
            assert_eq!(seen, vec![ExprId(3)]);
        }
    }

    #[test]
    fn polygamma_and_kronecker_children() {
        for node in [
            ExprNode::Polygamma(ExprId(1), ExprId(2)),
            ExprNode::KroneckerDelta(ExprId(1), ExprId(2)),
        ] {
            assert!(!node.is_atom());
            assert_eq!(node.child_count(), 2);
            let kids = node.children();
            assert_eq!(kids.len(), 2);
            assert_eq!(kids[0], ExprId(1));
            assert_eq!(kids[1], ExprId(2));
            let mut seen = Vec::new();
            node.for_each_child(|c| seen.push(c));
            assert_eq!(seen, vec![ExprId(1), ExprId(2)]);
        }
        assert_eq!(
            format!("{:?}", ExprNode::Polygamma(ExprId(1), ExprId(2))),
            "Polygamma(e1, e2)"
        );
        assert_eq!(
            format!("{:?}", ExprNode::KroneckerDelta(ExprId(1), ExprId(2))),
            "KroneckerDelta(e1, e2)"
        );
    }
}
