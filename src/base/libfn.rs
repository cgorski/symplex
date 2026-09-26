//! The registry of library special functions carried as
//! [`ExprNode::Apply`](crate::base::node::ExprNode::Apply).
//!
//! Elementary functions (`sin`, `exp`, `gamma`, `erf`, …) are dedicated
//! `ExprNode` variants.  The larger family of special functions — Bessel,
//! orthogonal polynomials, incomplete gamma, elliptic integrals, the integer
//! sequences — is stored as `Apply(SymbolId, args)` where the symbol interns
//! the function's name.  [`LibFn`] is the closed list of those names, so a
//! consumer that matches on it (an evaluator, a differentiator, a printer)
//! is checked for completeness by the compiler instead of failing at run
//! time on a name it forgot.
//!
//! Names are resolved through `Arena::lib_fn`
//! (a lookup on the interned text; no symbol is pre-interned, so `SymbolId`
//! numbering is unaffected) and nodes are built with
//! `Arena::lib_apply`.
//!
//! This module depends on `std` only.

use std::fmt;

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

macro_rules! lib_fns {
    (@arity $n:literal) => {
        Arity::Fixed($n)
    };
    (@arity $min:literal, $max:literal) => {
        Arity::Range {
            min: $min,
            max: $max,
        }
    };
    ($( $(#[$doc:meta])* $variant:ident = $name:literal / $arity:literal $(..= $max:literal)? ; )*) => {
        /// A library special function carried as `Apply(name, args)`.
        ///
        /// The variant order is the declaration order of the historical
        /// `FN_*` name constants and is the order of [`LibFn::ALL`].
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
            $( $(#[$doc])* $variant, )*
        }

        impl LibFn {
            /// Every library function, in declaration order.
            pub const ALL: &[LibFn] = &[ $( LibFn::$variant, )* ];

            /// The interned name: what `Display` prints and what the
            /// `Apply` node's symbol holds.
            pub const fn name(self) -> &'static str {
                match self { $( LibFn::$variant => $name, )* }
            }

            /// The function whose interned name is exactly `s`.
            pub fn from_name(s: &str) -> Option<LibFn> {
                match s { $( $name => Some(LibFn::$variant), )* _ => None }
            }

            /// The number of arguments the function takes; a call of any
            /// other length is left unevaluated by every consumer.
            pub const fn arity(self) -> Arity {
                match self { $( LibFn::$variant => lib_fns!(@arity $arity $(, $max)?), )* }
            }
        }
    };
}

lib_fns! {
    // ── Integer sequences and combinatorial counts ────────────────────────
    /// Double factorial `n!!`.
    Factorial2 = "factorial2" / 1;
    /// Subfactorial (derangement count) `!n`.
    Subfactorial = "subfactorial" / 1;
    /// Rising factorial (Pochhammer symbol) `(x)_n = x (x+1) ⋯ (x+n−1)`.
    RisingFactorial = "rising_factorial" / 2;
    /// Falling factorial `x^(n) = x (x−1) ⋯ (x−n+1)`.
    FallingFactorial = "falling_factorial" / 2;
    /// Fibonacci number `F_n`.
    Fibonacci = "fibonacci" / 1;
    /// Lucas number `L_n`.
    Lucas = "lucas" / 1;
    /// Bernoulli number `B_n`.
    Bernoulli = "bernoulli" / 1;
    /// Harmonic number `H_n`.
    Harmonic = "harmonic" / 1;
    /// Catalan number `C_n`.
    Catalan = "catalan" / 1;
    /// Bell number `B_n`.
    Bell = "bell" / 1;
    /// Euler number `E_n`.
    EulerNumber = "euler_number" / 1;
    /// Signed Stirling number of the first kind `s(n, k)`.
    Stirling1 = "stirling1" / 2;
    /// Stirling number of the second kind `S(n, k)`.
    Stirling2 = "stirling2" / 2;
    /// Integer partition count `p(n)`.
    PartitionCount = "partition_count" / 1;
    /// Lambert W, `lambertw(x, k)`: the branch `W_k` (`k` an integer) of the
    /// inverse of `w·eʷ`, as SymPy's `LambertW(x, k)`.  The principal branch
    /// `k = 0` is the dedicated `ExprNode::LambertW(x)` node, which
    /// `Arena::lambertw_branch` and `eval` build for `lambertw(x)` and
    /// `lambertw(x, 0)`; an `Apply` carries only the other branches.
    LambertW = "lambertw" / 1 ..= 2;

    // ── Bessel functions `(order, x)` ─────────────────────────────────────
    /// Bessel function of the first kind `J_ν(x)`.
    BesselJ = "besselj" / 2;
    /// Bessel function of the second kind `Y_ν(x)`.
    BesselY = "bessely" / 2;
    /// Modified Bessel function of the first kind `I_ν(x)`.
    BesselI = "besseli" / 2;
    /// Modified Bessel function of the second kind `K_ν(x)`.
    BesselK = "besselk" / 2;

    // ── Classical orthogonal polynomials `(n, x)` ─────────────────────────
    /// Legendre polynomial `P_n(x)`.
    Legendre = "legendre" / 2;
    /// Chebyshev polynomial of the first kind `T_n(x)`.
    ChebyshevT = "chebyshev_t" / 2;
    /// Chebyshev polynomial of the second kind `U_n(x)`.
    ChebyshevU = "chebyshev_u" / 2;
    /// Physicists' Hermite polynomial `H_n(x)`.
    Hermite = "hermite" / 2;
    /// Laguerre polynomial `L_n(x)`.
    Laguerre = "laguerre" / 2;

    // ── Error-function family and integrals (0.9) ─────────────────────────
    /// Imaginary error function `erfi(x)`.
    Erfi = "erfi" / 1;
    /// Inverse error function `erf⁻¹(y)`.
    ErfInv = "erfinv" / 1;
    /// Inverse complementary error function `erfc⁻¹(y)`.
    ErfcInv = "erfcinv" / 1;
    /// Generalised exponential integral `E_n(x)`, as `expint(n, x)`.
    ExpInt = "expint" / 2;
    /// Hyperbolic sine integral `Shi(x)`.
    Shi = "Shi" / 1;
    /// Hyperbolic cosine integral `Chi(x)`.
    Chi = "Chi" / 1;
    /// Fresnel sine integral `S(x)`.
    FresnelS = "fresnels" / 1;
    /// Fresnel cosine integral `C(x)`.
    FresnelC = "fresnelc" / 1;
    /// Lower incomplete gamma `γ(s, x)`.
    LowerGamma = "lowergamma" / 2;
    /// Upper incomplete gamma `Γ(s, x)`.
    UpperGamma = "uppergamma" / 2;
    /// Polylogarithm `Li_s(z)`, as `polylog(s, z)`.
    PolyLog = "polylog" / 2;
    /// Dirichlet eta `η(s)`.
    DirichletEta = "dirichlet_eta" / 1;

    // ── Airy functions ────────────────────────────────────────────────────
    /// Airy function `Ai(x)`.
    AiryAi = "airyai" / 1;
    /// Airy function `Bi(x)`.
    AiryBi = "airybi" / 1;
    /// Derivative `Ai′(x)`.
    AiryAiPrime = "airyaiprime" / 1;
    /// Derivative `Bi′(x)`.
    AiryBiPrime = "airybiprime" / 1;

    // ── Elliptic integrals (parameter `m = k²`, as in SymPy) ──────────────
    /// Complete elliptic integral of the first kind `K(m)`.
    EllipticK = "elliptic_k" / 1;
    /// Complete elliptic integral of the second kind `E(m)`.
    EllipticE = "elliptic_e" / 1;
    /// Incomplete elliptic integral of the first kind `F(φ | m)`.
    EllipticF = "elliptic_f" / 2;
    /// Complete elliptic integral of the third kind `Π(n | m)`.
    EllipticPi = "elliptic_pi" / 2;

    // ── Orthogonal polynomials with parameters ────────────────────────────
    /// Gegenbauer polynomial `C_n^{(α)}(x)`, as `gegenbauer(n, α, x)`.
    Gegenbauer = "gegenbauer" / 3;
    /// Jacobi polynomial `P_n^{(α,β)}(x)`, as `jacobi(n, α, β, x)`.
    Jacobi = "jacobi" / 4;
    /// Associated Legendre function `P_n^m(x)`, as `assoc_legendre(n, m, x)`.
    AssocLegendre = "assoc_legendre" / 3;
    /// Generalised Laguerre polynomial `L_n^{(α)}(x)`, as `assoc_laguerre(n, α, x)`.
    AssocLaguerre = "assoc_laguerre" / 3;

    // ── Generalised incomplete beta (0.12), SymPy order `(a, b, x1, x2)` ──
    /// Incomplete beta `B_{(x₁, x₂)}(a, b)`.
    BetaInc = "betainc" / 4;
    /// Regularised incomplete beta `I_{(x₁, x₂)}(a, b)`.
    BetaIncRegularized = "betainc_regularized" / 4;
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
