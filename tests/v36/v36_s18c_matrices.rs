//! Tests of session 18. Lane C of the 0.60.1 block: matrices (pivots that are hidden zeros, `diagonalize`,
//! `is_diagonalizable`, `is_controllable`).
//! Every test says what was wrong before and cites its oracle.
//!
//! The oracle calls are in `oracle_s18c.py` of the lane's scratch (SymPy 1.14, mpmath through
//! `sympy.N` at 50 and 100 digits); their output is quoted here.  "Before" is 0.60.0.

use std::time::Duration;

use crate::perf_assert::assert_fast;
use symplex::matrix_decomp::gram_schmidt;
use symplex::prelude::*;

/// A context with `x` and `y` plain and `p` positive.
fn context() -> Context {
    let ctx = Context::new();
    ctx.symbol_with("p", &[Assumption::Positive]).unwrap();
    ctx
}

/// A matrix from the sources of its entries.
fn mat(ctx: &Context, rows: &[&[&str]]) -> Matrix {
    Matrix::new(
        rows.iter()
            .map(|r| r.iter().map(|s| ctx.parse(s).unwrap()).collect())
            .collect(),
    )
    .unwrap()
}

/// Is `r` the refusal of an undecided pivot (not a verdict such as "singular")?
fn undecided<T>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::ComputationFailed { reason, .. }) if reason.contains("cannot decide"))
}

/// Is `r` the verdict "singular"?
fn singular<T>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::ComputationFailed { reason, .. }) if reason.contains("singular"))
}

/// Largest `|entry|` of `m` at `p = 7/5`, `x = 7/5`.
fn size_at(ctx: &Context, m: &Matrix) -> f64 {
    let at = ctx.rational(7, 5);
    m.subs(&ctx.symbol("p"), &at)
        .subs(&ctx.symbol("x"), &at)
        .iter()
        .map(|e| e.eval_complex64().unwrap().norm())
        .fold(0.0, f64::max)
}

/// The zero forms that the pivot test proves (20 of the 24 of the findings; `p` is positive).
const ZERO_FORMS: [&str; 20] = [
    "sin(p)^2 + cos(p)^2 - 1",
    "cos(2*p) - 2*cos(p)^2 + 1",
    "exp(2*p) - exp(p)^2",
    "cosh(p)^2 - sinh(p)^2 - 1",
    "sinh(p) - (exp(p) - exp(-p))/2",
    "tan(p) - sin(p)/cos(p)",
    "log(exp(p)) - p",
    "sqrt(p^2) - abs(p)",
    "sin(2*p) - 2*sin(p)*cos(p)",
    "(p + 1)^2 - p^2 - 2*p - 1",
    "zeta(2) - pi^2/6",
    "sin(1)^2 + cos(1)^2 - 1",
    "erf(1) + erfc(1) - 1",
    "gamma(1/2)^2 - pi",
    "log(6) - log(2) - log(3)",
    "sqrt(2)*sqrt(3) - sqrt(6)",
    "exp(I*pi/5)^5 + 1",
    "cos(pi/7) + cos(3*pi/7) + cos(5*pi/7) - 1/2",
    "atan(1/2) + atan(1/3) - pi/4",
    "polylog(2, 1) - pi^2/6",
];

/// W02.  The pivot test stopped at the exact test for rational functions and, for constants,
/// at a certified value: an entry it did not decide was a pivot.  Before, 11 of these 20
/// entries (`sin(p)^2 + cos(p)^2 - 1`, `cos(2*p) - …`, `exp(2*p) - exp(p)^2`, `cosh² − sinh² −
/// 1`, `sinh(p) - …`, `tan(p) - …`, `sin(2*p) - …`, `sin(1)^2 + cos(1)^2 - 1`, `log(6) - log(2) -
/// log(3)`, the cosine sum, the arctangent sum) made `[[e]]` a matrix of rank 1 with an empty
/// null space, and `[[1, 1], [1, 1 + e]]` one of rank 2.  Now the test is three-valued and
/// goes on to the zero test of expressions: rank 0 and 1.
///
/// Oracle: each form is 0 — SymPy `simplify(e)` is `0` for 17 of them; the three it leaves
/// are identities (`erf + erfc = 1` by definition; `atan(1/2) + atan(1/3) = atan(1) = π/4` by
/// the addition theorem; `cos(π/7) + cos(3π/7) + cos(5π/7) = 1/2`), and all 20 are `0.0` at 50
/// and at 100 digits (`sympy.N` at `p = 7/5`).  SymPy `Matrix([[0]]).rank()` = 0,
/// `Matrix([[1, 1], [1, 1]]).rank()` = 1.
#[test]
fn a_zero_form_is_not_a_pivot() {
    for zero in ZERO_FORMS {
        let ctx = context();
        let e = ctx.parse(zero).unwrap();
        let single = Matrix::new(vec![vec![e.clone()]]).unwrap();
        assert_eq!(single.rank(), 0, "[[{zero}]]");
        assert_eq!(single.nullspace().len(), 1, "[[{zero}]]");
        assert!(singular(&single.inv()), "[[{zero}]].inv()");
        assert!(singular(&single.lu()), "[[{zero}]].lu()");
        let one = ctx.int(1);
        let two = Matrix::new(vec![
            vec![one.clone(), one.clone()],
            vec![one.clone(), &one + &e],
        ])
        .unwrap();
        assert_eq!(two.rank(), 1, "[[1, 1], [1, 1 + {zero}]]");
        assert_eq!(two.nullspace().len(), 1, "[[1, 1], [1, 1 + {zero}]]");
        assert_eq!(two.rref().1, vec![0], "[[1, 1], [1, 1 + {zero}]]");
    }
}

/// W02, the other side: a zero test that takes a value "zero to the precision reached" for
/// zero would pass the test above.  `exp(-60)` is not zero, alone or beside a zero form.
/// (Right before as well; pinned so that it stays right.)
///
/// Oracle: mpmath `exp(-60)` = `8.7565107626965203385e-27`.
#[test]
fn a_tiny_nonzero_entry_is_a_pivot() {
    let ctx = context();
    for tiny in [
        "exp(-60)",
        "sin(1)^2 + cos(1)^2 - 1 + exp(-60)",
        "sin(p)^2 + cos(p)^2 - 1 + exp(-60)",
    ] {
        assert_eq!(mat(&ctx, &[&[tiny]]).rank(), 1, "[[{tiny}]]");
        let m = mat(&ctx, &[&["1", "1"], &["1", &format!("1 + {tiny}")]]);
        assert_eq!(m.rank(), 2, "[[1, 1], [1, 1 + {tiny}]]");
        assert!(m.nullspace().is_empty());
        assert!(m.inv().is_ok(), "[[1, 1], [1, 1 + {tiny}]].inv()");
        assert!(m.lu().is_ok());
    }
}

/// W02.  `[[1, 1], [1, 1 + sin(x)^2 + cos(x)^2 - 1]]` is `[[1, 1], [1, 1]]`.  Before: `rref()`
/// had the pivots `[0, 1]` and a matrix of four `nan` (the second row was divided by the zero),
/// `lu()` returned factors with that zero as the second pivot, and `solve` an answer.  Now:
/// one pivot, the second row `0 0`, and `lu`, `solve` and `inv` say "singular".
///
/// Oracle: `sin² + cos² = 1`; SymPy `Matrix([[1, 1], [1, 1]]).rank()` = 1.
#[test]
fn elimination_does_not_divide_by_a_zero_form() {
    let ctx = context();
    let m = mat(&ctx, &[&["1", "1"], &["1", "1 + sin(x)^2 + cos(x)^2 - 1"]]);
    let (r, pivots) = m.rref();
    assert_eq!(pivots, vec![0]);
    assert_eq!(r, mat(&ctx, &[&["1", "1"], &["0", "0"]]));
    assert!(singular(&m.lu()));
    assert!(singular(&m.inv()));
    assert!(singular(&m.solve(&mat(&ctx, &[&["1"], &["2"]]))));
    assert_eq!(m.columnspace().len(), 1);
    assert_eq!(m.rowspace().len(), 1);
    assert_eq!(m.left_nullspace().len(), 1);
}

/// W02, the undecided case.  `g = gamma(p + 1) - p*gamma(p)` is 0, by an identity that no zero
/// test of the library knows: it is neither proved zero nor proved nonzero.  Before, `g` was
/// a pivot everywhere: `[[g]].inv()` was `[[1/g]]`, `lu()` returned `U = [[g]]`, `solve` divided
/// by it.  Now every function that returns a `Result` refuses, naming the entry; `rank`,
/// which cannot refuse, keeps the reading its doc states (the entry counts as nonzero).
///
/// Oracle: SymPy `simplify(gamma(x + 1) - x*gamma(x))` = 0; `0.0` at 50 and 100 digits.  So
/// the matrix is singular and no inverse exists: the answers of 0.60.0 were wrong, and the
/// right one ("singular") is not proved — a refusal.
#[test]
fn an_undecided_pivot_is_refused_where_the_function_can_refuse() {
    let ctx = context();
    let m = mat(&ctx, &[&["gamma(p + 1) - p*gamma(p)"]]);
    assert!(undecided(&m.inv()), "{:?}", m.inv());
    assert!(undecided(&m.lu()));
    assert!(undecided(&m.solve(&mat(&ctx, &[&["1"]]))));
    assert!(undecided(&m.pinv()));
    assert!(undecided(&m.rank_decomposition()));
    assert!(undecided(&m.qr()));
    assert!(undecided(&m.condition_number()));
    assert!(undecided(&m.solve_least_squares(&mat(&ctx, &[&["1"]]))));
    // The documented reading of a function that cannot refuse.
    assert_eq!(m.rank(), 1);
    let Err(SymplexError::ComputationFailed { reason, .. }) = m.lu() else {
        panic!("lu");
    };
    assert!(reason.contains("Gamma(p + 1)"), "{reason}");
}

/// W02.  An undecided entry is a pivot only when its column has no entry proved nonzero:
/// `[[g, 2], [3, 4]]` (`g` as above) has the pivot 3, and what `g` is does not matter.  Before,
/// `g` was the pivot: `lu()` had `L = [[1, 0], [3/g, 1]]`, a division by zero.  Now `perm =
/// [1, 0]` and `L·U = P·A`.
///
/// Oracle: SymPy `Matrix([[0, 2], [3, 4]])`: rank 2, det −6, `LUsolve(Matrix([1, 2]))` =
/// `[0, 1/2]`.
#[test]
fn an_undecided_entry_beside_a_proved_pivot_needs_no_decision() {
    let ctx = context();
    let m = mat(&ctx, &[&["gamma(p + 1) - p*gamma(p)", "2"], &["3", "4"]]);
    assert_eq!(m.rank(), 2);
    let Lu { l, u, perm } = m.lu().unwrap();
    assert_eq!(perm, vec![1, 0]);
    let pa = Matrix::new(perm.iter().map(|&i| m.row(i).to_vec()).collect()).unwrap();
    assert!(size_at(&ctx, &l.matmul(&u).unwrap().sub(&pa).unwrap()) < 1e-12);
    let s = m.solve(&mat(&ctx, &[&["1"], &["2"]])).unwrap();
    let want = mat(&ctx, &[&["0"], &["1/2"]]);
    assert!(size_at(&ctx, &s.sub(&want).unwrap()) < 1e-12, "{s}");
    assert!(m.inv().is_ok());
}

/// W18.  `[[1, g], [0, 1]]` with `g = gamma(p + 1) - p*gamma(p)` is the identity matrix.
/// Before: `eigenvects()` returned one eigenvector for the eigenvalue 1 of multiplicity 2,
/// `diagonalize()` was `Err(NotDiagonalizable)` and `jordan_form()` a Jordan block.  Now the
/// three are refused (`ComputationFailed`, naming `g`) and `is_diagonalizable()` is `None`:
/// `NotDiagonalizable` is reported only when every zero decision of the eigenvector solves
/// was a proof.
///
/// Oracle: SymPy `simplify(gamma(x + 1) - x*gamma(x))` = 0; `eye(2).is_diagonalizable()` =
/// `True`.
#[test]
fn an_undecided_entry_of_an_eigenvector_solve_is_refused() {
    let ctx = context();
    let m = mat(&ctx, &[&["1", "gamma(p + 1) - p*gamma(p)"], &["0", "1"]]);
    assert!(undecided(&m.eigenvects()), "{:?}", m.eigenvects());
    assert!(undecided(&m.diagonalize()));
    assert!(undecided(&m.jordan_form()));
    assert_eq!(m.is_diagonalizable(), None);
    // The eigenvalues need no zero test of the entry.
    assert_eq!(m.eigenvals().unwrap(), vec![ctx.int(1), ctx.int(1)]);
}

/// W18.  With a zero form that the pivot test proves, the same matrix is the identity for
/// every function (right before for this form, which `simplify` reduces; pinned), and a
/// Jordan block with a function on the diagonal is "not diagonalizable" for the query too
/// (before: `None` for `[[exp(x), 1], [0, exp(x)]]`, since the query answered only for
/// rational functions of the symbols).
///
/// Oracle: `eye(2).is_diagonalizable()` = `True`; SymPy `Matrix([[exp(u), 1], [0,
/// exp(u)]]).is_diagonalizable()` = `False` (`A − e^u·I = [[0, 1], [0, 0]]` has rank 1).
#[test]
fn diagonalize_and_the_query_agree() {
    let ctx = context();
    let identity = mat(&ctx, &[&["1", "sin(x)^2 + cos(x)^2 - 1"], &["0", "1"]]);
    let spaces = identity.eigenvects().unwrap();
    assert_eq!(spaces.len(), 1);
    assert_eq!((spaces[0].1, spaces[0].2.len()), (2, 2));
    assert_eq!(identity.is_diagonalizable(), Some(true));
    assert!(identity.diagonalize().is_ok());

    let block = mat(&ctx, &[&["exp(x)", "1"], &["0", "exp(x)"]]);
    assert_eq!(block.is_diagonalizable(), Some(false));
    assert!(matches!(
        block.diagonalize(),
        Err(SymplexError::NotDiagonalizable { .. })
    ));
    let JordanForm { j, .. } = block.jordan_form().unwrap();
    assert_eq!(j, block);
}

/// W08.  `is_diagonalizable()` answered for generic values of the symbols: `[[1, 1], [0, x]]`
/// was `Some(true)`, and it is a Jordan block at `x = 1`.  Now `Some(true)` is a statement
/// about every value: here `None`, and the two specialisations are decided.
///
/// Oracle: SymPy `Matrix([[1, 1], [0, 1]]).is_diagonalizable()` = `False`, `Matrix([[1, 1],
/// [0, 2]]).is_diagonalizable()` = `True`.
#[test]
fn is_diagonalizable_is_true_only_for_every_value_of_the_symbols() {
    let ctx = context();
    let x = ctx.symbol("x");
    let m = mat(&ctx, &[&["1", "1"], &["0", "x"]]);
    assert_eq!(m.is_diagonalizable(), None);
    assert_eq!(m.subs(&x, &ctx.int(1)).is_diagonalizable(), Some(false));
    assert_eq!(m.subs(&x, &ctx.int(2)).is_diagonalizable(), Some(true));
    // The generic diagonalization is still returned (it divides by x − 1).
    assert!(m.diagonalize().is_ok());
}

/// W08, the proofs that keep `Some(true)`.  (a) The eigenvector matrix is finite and
/// invertible for every value: `[[x, y], [y, x]]` has the eigenvectors `(1, 1)` and `(−1, 1)`
/// whatever `x` and `y` are — also at `y = 0`, where its two eigenvalues meet — and a diagonal
/// matrix has the unit vectors.  (b) The eigenvalues differ for every value: for a positive
/// `p`, `[[1, 1], [0, -p]]` has the eigenvalues `1` and `−p`, and the discriminant `(p + 1)²`
/// of its characteristic polynomial is never 0.  (All three `Some(true)` before as well.)
///
/// Oracle: SymPy `Matrix([[u, y], [y, u]]).eigenvects()` = `[(u − y, 1, [(−1, 1)]), (u + y, 1,
/// [(1, 1)])]`; `Matrix([[1, 1], [0, -x]]).eigenvals()` = `{−x: 1, 1: 1}` for a positive `x`.
#[test]
fn is_diagonalizable_proves_what_holds_for_every_value() {
    let ctx = context();
    assert_eq!(
        mat(&ctx, &[&["x", "y"], &["y", "x"]]).is_diagonalizable(),
        Some(true)
    );
    assert_eq!(
        mat(&ctx, &[&["1", "0"], &["0", "x"]]).is_diagonalizable(),
        Some(true)
    );
    assert_eq!(
        mat(&ctx, &[&["1", "1"], &["0", "-p"]]).is_diagonalizable(),
        Some(true)
    );
    // Defective for every x: Some(false), as before.
    assert_eq!(
        mat(&ctx, &[&["x", "1"], &["0", "x"]]).is_diagonalizable(),
        Some(false)
    );
}

/// W12.  `is_controllable` and `is_observable` compared the rank of W02 with `n`.  Before: a
/// system whose input matrix is zero — `B = (0, log(6) - log(2) - log(3))ᵀ` — was
/// controllable, and one whose output matrix is zero — `C = (sin(1)^2 + cos(1)^2 - 1, 0)` —
/// observable.  Now `false`; with an input matrix that is not zero the system stays
/// controllable, and an entry that is neither proved zero nor proved nonzero gives `false`
/// (`true` only when proved) and makes `ackermann` an error.
///
/// Oracle: SymPy, `A = [[0, 1], [0, 0]]`: rank `[B, AB]` = 2 for `B = (0, 1)ᵀ` and 0 for `B =
/// 0`; rank `[C; CA]` = 2 for `C = (1, 0)` and 0 for `C = 0`.
#[test]
fn controllability_and_observability_are_true_only_when_proved() {
    let ctx = context();
    let a = || mat(&ctx, &[&["0", "1"], &["0", "0"]]);
    let d = || mat(&ctx, &[&["0"]]);
    let system = |b: &str, c: &str| {
        StateSpace::new(
            a(),
            mat(&ctx, &[&["0"], &[b]]),
            mat(&ctx, &[&[c, "0"]]),
            d(),
        )
        .unwrap()
    };
    let plain = system("1", "1");
    assert!(plain.is_controllable() && plain.is_observable());
    let disguised = system(
        "1 + log(6) - log(2) - log(3)",
        "1 + sin(1)^2 + cos(1)^2 - 1",
    );
    assert!(disguised.is_controllable() && disguised.is_observable());
    let zero_input = system("log(6) - log(2) - log(3)", "1");
    assert!(!zero_input.is_controllable());
    assert!(zero_input.is_observable());
    let zero_output = system("1", "sin(1)^2 + cos(1)^2 - 1");
    assert!(zero_output.is_controllable());
    assert!(!zero_output.is_observable());
    // Neither proved zero nor proved nonzero: not proved controllable, not proved observable.
    let open = system("gamma(p + 1) - p*gamma(p)", "gamma(p + 1) - p*gamma(p)");
    assert!(!open.is_controllable());
    assert!(!open.is_observable());
    assert!(open.ackermann(&[ctx.int(-1), ctx.int(-2)]).is_err());
}

/// W12, the reading for a system with symbols: the generic rank, as `Matrix::rank` (and as
/// before).  `A = diag(1, x)`, `B = (1, 1)ᵀ` is controllable for every `x` but 1.
///
/// Oracle: SymPy `det [B, AB]` = `x − 1`.
#[test]
fn controllability_of_a_system_with_symbols_is_generic() {
    let ctx = context();
    let x = ctx.symbol("x");
    let system = |a22: &Ex| {
        StateSpace::new(
            Matrix::new(vec![
                vec![ctx.int(1), ctx.int(0)],
                vec![ctx.int(0), a22.clone()],
            ])
            .unwrap(),
            mat(&ctx, &[&["1"], &["1"]]),
            mat(&ctx, &[&["1", "1"]]),
            mat(&ctx, &[&["0"]]),
        )
        .unwrap()
    };
    assert!(system(&x).is_controllable());
    assert!(!system(&ctx.int(1)).is_controllable());
    assert!(system(&ctx.int(2)).is_controllable());
}

/// The other eliminations that divided by an undecided entry (`g = gamma(p + 1) -
/// p*gamma(p)`, which is 0).  Before: `ldl` of `[[g, 1], [1, 1]]` returned `L = [[1, 0], [1/g,
/// 1]]`; `gram_schmidt` of `(1, 0)`, `(1, g)` returned the second vector `(0, g)` as a basis
/// vector (it is the zero vector); `hessenberg` took `g` for the pivot of a column with two
/// candidates and divided by it.  Now each is refused.  Where nothing is divided by the
/// entry, there is nothing to refuse: a column with `g` alone below the diagonal.
///
/// Oracle: SymPy `simplify(gamma(x + 1) - x*gamma(x))` = 0.
#[test]
fn ldl_gram_schmidt_and_hessenberg_refuse_an_undecided_pivot() {
    let ctx = context();
    let g = "gamma(p + 1) - p*gamma(p)";
    assert!(undecided(&mat(&ctx, &[&[g, "1"], &["1", "1"]]).ldl()));
    let v = |a: &str, b: &str| {
        Matrix::col_vector(vec![ctx.parse(a).unwrap(), ctx.parse(b).unwrap()]).unwrap()
    };
    assert!(undecided(&gram_schmidt(&[v("1", "0"), v("1", g)], false)));
    assert!(gram_schmidt(&[v("1", "0"), v("1", "1")], false).is_ok());
    let two = mat(&ctx, &[&["1", "2", "3"], &[g, "1", "1"], &[g, "2", "5"]]);
    assert!(undecided(&two.hessenberg()));
    let one = mat(&ctx, &[&["1", "2", "3"], &[g, "1", "1"], &["0", "2", "5"]]);
    let Hessenberg { h, p } = one.hessenberg().unwrap();
    assert_eq!(h, one);
    assert_eq!(p, Matrix::identity(&ctx, 3).unwrap());
}

/// The pivot test is hot: an elimination asks it for every pivot candidate.  The certified
/// test of an entry with functions is one evaluation at a sample point when the entry is not
/// zero, so a 4×4 matrix of trigonometric entries is eliminated in milliseconds (0.08 s for
/// the four calls, alone, in a debug build; the budgets are for a loaded machine).
#[test]
fn the_pivot_test_stays_cheap_for_entries_with_functions() {
    let ctx = context();
    let m = mat(
        &ctx,
        &[
            &["cos(x)", "-sin(x)", "0", "y"],
            &["sin(x)", "cos(x)", "0", "exp(y)"],
            &["0", "0", "1", "x*y"],
            &["sin(2*x)", "cos(2*x)", "tan(x)", "1"],
        ],
    );
    let rank = assert_fast!(Duration::from_secs(5), m.rank());
    assert_eq!(rank, 4);
    assert!(assert_fast!(Duration::from_secs(5), m.nullspace()).is_empty());
    assert!(assert_fast!(Duration::from_secs(5), m.lu()).is_ok());
    assert!(assert_fast!(Duration::from_secs(10), m.inv()).is_ok());
}

/// W02, constants that only their parts decide.  The zero test of constants proves `cos(π/7) +
/// cos(3π/7) + cos(5π/7) − 1/2 = 0` and `erf(1) + erfc(1) − 1 = 0`, each alone; it does not
/// prove their product zero, nor `2/(erf(1) + erfc(1) − 2) + 2`, and an elimination multiplies
/// such constants of different entries.  The pivot test decides a product by its factors and
/// reads a constant part that is a rational number in disguise as that number (a guess from
/// twelve certified digits, replaced only when the difference is proved zero).  Before:
/// `[[2/(erf(1) + erfc(1) - 2) + 2]]` had rank 1 and the 2×3 matrix below rank 2 (the product
/// of the two zero constants alone was decided before as well; pinned).
///
/// Oracle: the two identities (see `a_zero_form_is_not_a_pivot`); `2/(1 − 2) + 2 = 0`; SymPy
/// `Matrix([[-(x + 1), -1, -1], [x + 1, 1, 1]]).rank()` = 1 (the second row is minus the
/// first; case 271 of the hunt, exact rank over `ℚ(x)`).
#[test]
fn a_constant_decided_by_its_parts_is_not_a_pivot() {
    let ctx = context();
    for zero in [
        "(cos(pi/7) + cos(3*pi/7) + cos(5*pi/7) - 1/2)*(erf(1) + erfc(1) - 1)",
        "2/(erf(1) + erfc(1) - 2) + 2",
    ] {
        let m = mat(&ctx, &[&[zero]]);
        assert_eq!(m.rank(), 0, "[[{zero}]]");
        assert!(singular(&m.inv()), "[[{zero}]]");
    }
    let m = mat(
        &ctx,
        &[
            &["-(p + 1) + (erf(1) + erfc(1) - 1)", "-1", "-1"],
            &[
                "p + 1",
                "1",
                "1 + cos(pi/7) + cos(3*pi/7) + cos(5*pi/7) - 1/2",
            ],
        ],
    );
    assert_eq!(m.rank(), 1);
    assert_eq!(m.nullspace().len(), 2);
    // Not a rational number: nothing is replaced, and the entry is a pivot.
    assert_eq!(
        mat(&ctx, &[&["erf(1) + erfc(1) - 1 + pi - 355/113"]]).rank(),
        1
    );
}

/// The zero tests of one matrix call have a work budget, counted in nodes of the entries
/// tested (not in time): when it is spent, or an entry is too large for the proofs, the
/// entry is undecided.  A 4×4 matrix of sixteen different entries that no test decides
/// (`gamma(p + k + 1) − (p + k)·gamma(p + k)`, each 0): every call returns — `rank` with the
/// documented reading, `lu` and `solve` with the refusal — within the budget (2.7 s for the
/// three calls, alone, in a debug build; without the budget every entry of every step of
/// the elimination went through every proof).  The budgets are for a loaded machine.
///
/// Oracle: SymPy `simplify(gamma(x + 1) - x*gamma(x))` = 0, so the matrix is the zero matrix
/// and no `Ok` of `lu` or `solve` would be right.
#[test]
fn the_zero_tests_of_a_matrix_call_are_bounded() {
    let ctx = context();
    let g = |k: usize| format!("gamma(p + {}) - (p + {k})*gamma(p + {k})", k + 1);
    let rows: Vec<Vec<String>> = (0..4)
        .map(|i| (0..4).map(|j| g(4 * i + j + 1)).collect())
        .collect();
    let m = Matrix::new(
        rows.iter()
            .map(|r| r.iter().map(|s| ctx.parse(s).unwrap()).collect())
            .collect(),
    )
    .unwrap();
    // (Its value is the documented reading of undecided entries, not asserted.)
    let _ = assert_fast!(Duration::from_secs(20), m.rank());
    assert!(undecided(&assert_fast!(Duration::from_secs(20), m.lu())));
    let b = mat(&ctx, &[&["1"], &["2"], &["3"], &["4"]]);
    assert!(undecided(&assert_fast!(
        Duration::from_secs(20),
        m.solve(&b)
    )));
}
