# Cases of the rule set `paths` (review A §3; the table is tables/path_moves.tsv, from api_fate.tsv).
from support import case

CASES = [
    # ---- written paths
    case("paths.written",
         "fn t() {\n    let m = symplex::matrix::Matrix::zeros(&ctx, 2, 2);\n    let e: symplex::expr::Ex = x;\n}\n",
         "fn t() {\n    let m = symplex::linalg::Matrix::zeros(&ctx, 2, 2);\n    let e: symplex::Ex = x;\n}\n"),
    case("paths.written",  # the rational helpers go to `num`, the LP types to `lp`
         "fn t() {\n    let h = symplex::linprog::q(1, 2);\n    let p = symplex::linprog::LpProblem::new();\n}\n",
         "fn t() {\n    let h = symplex::num::q(1, 2);\n    let p = symplex::lp::LpProblem::new();\n}\n"),
    case("paths.written",  # a macro path, and the module of the same name
         "fn t() {\n    let d = symplex::dim!(m / s);\n    let m = symplex::matrix![[1, 2], [3, 4]];\n}\n",
         "fn t() {\n    let d = symplex::experimental::units::dim!(m / s);\n    let m = symplex::matrix![[1, 2], [3, 4]];\n}\n"),
    case("paths.written",  # units move under `experimental` (Part 4, C-1)
         "fn t(f: symplex::units::Force) {}\n",
         "fn t(f: symplex::experimental::units::Force) {}\n"),
    case("paths.written",
         "fn t() { let s: symplex::linprog::LpSolution = solve(); }\n", manual="is renamed to"),
    case("paths.written",
         "fn t() { let e = symplex::parse::parse(&ctx, \"x\"); }\n", manual="is removed at 1.0"),
    case("paths.written",  # not in strings, comments or other crates' paths
         "// symplex::matrix::Matrix\nfn t() { let s = \"symplex::matrix::Matrix\"; let o = other::symplex::matrix::X; }\n",
         silent=True),
    case("paths.written",  # unchanged paths are not sites
         "fn t() { let p = symplex::ntheory::is_prime(7); let c = symplex::prelude::Context::new(); }\n", silent=True),
    # ---- `use` items
    case("paths.use",
         "use symplex::matrix::{Matrix, QMatrix};\nuse symplex::linprog::{q, qi, LpProblem};\nfn t() {}\n",
         "use symplex::linalg::{Matrix, QMatrix};\nuse symplex::num::{q, qi};\nuse symplex::lp::LpProblem;\nfn t() {}\n"),
    case("paths.use",  # a nested tree is flattened to one line per 1.0 module
         "mod m {\n    use symplex::{expr::Ex, matrix::Matrix, prelude::*};\n}\n",
         "mod m {\n    use symplex::Ex;\n    use symplex::linalg::Matrix;\n    use symplex::prelude::*;\n}\n"),
    case("paths.use",  # an alias is kept
         "use symplex::matrix::Matrix as M;\n",
         "use symplex::linalg::Matrix as M;\n"),
    case("paths.use",
         "use symplex::linprog::*;\n", manual="more than one 1.0 module"),
    case("paths.use",
         "use symplex::matrix;\nfn t() { let m = matrix::Matrix::zeros(&ctx, 1, 1); }\n", manual="imported by name"),
    case("paths.use",
         "use symplex::matrix::{jacobian, Matrix};\n",
         "use symplex::matrix::jacobian;\nuse symplex::linalg::Matrix;\n", manual="cannot be imported"),
    case("paths.use", "use symplex::prelude::*;\nuse symplex::ntheory::{is_prime, totient};\n", silent=True),
    # ---- names that leave the prelude
    case("paths.prelude-names",
         "use symplex::prelude::*;\n\nfn t() {\n    let p: Poly = make();\n    let m = QMatrix::zeros(2, 2);\n    let ok = Matrix::identity(&ctx, 2);\n}\n",
         "use symplex::prelude::*;\nuse symplex::linalg::QMatrix;\nuse symplex::poly::Poly;\n\nfn t() {\n    let p: Poly = make();\n"
         "    let m = QMatrix::zeros(2, 2);\n    let ok = Matrix::identity(&ctx, 2);\n}\n"),
    case("paths.prelude-names",  # already imported, or not from the prelude at all: nothing to add
         "use symplex::prelude::*;\nuse symplex::poly::Poly;\nfn t(p: Poly) {}\n", silent=True),
    case("paths.prelude-names",
         "fn t(p: Poly) {}\n", silent=True),
    case("paths.prelude-names",
         "use symplex::prelude::*;\nfn t(s: LpSolution) {}\n", manual="is renamed to"),
]

CASES += [
    # the name is imported explicitly under its OLD path: rule paths.use moves that line, nothing is added
    case("paths.use",
         "use symplex::prelude::*;\nuse symplex::poly_ex::Poly;\nfn t(p: Poly) {}\n",
         "use symplex::prelude::*;\nuse symplex::poly::Poly;\nfn t(p: Poly) {}\n"),
]

CASES += [
    # an unknown name under a module that moves is not guessed at
    case("paths.use", "use symplex::matrix::CodegenOpts;\n", manual="is not an item of `symplex::matrix`"),
    case("paths.written", "fn t() { let x = symplex::matrix::brand_new(1); }\n", manual="is not an item of"),
]

CASES += [
    # a struct pattern after `let` names the type; a string that holds the word does not
    case("paths.prelude-names",
         "use symplex::prelude::*;\nfn t() {\n    let Qr { q, r } = m.qr().unwrap();\n    let s = \"Step 1: Interval\";\n}\n",
         "use symplex::prelude::*;\nuse symplex::linalg::Qr;\nfn t() {\n    let Qr { q, r } = m.qr().unwrap();\n    let s = \"Step 1: Interval\";\n}\n"),
    case("paths.prelude-names",  # the re-exported foreign type and a macro of the old prelude
         "use symplex::prelude::*;\nfn t() {\n    let z = Complex64::new(0.0, 1.0);\n    let r = rule!(a => b);\n}\n",
         "use symplex::prelude::*;\nuse symplex::num::Complex64;\nuse symplex::rules::rule;\nfn t() {\n"
         "    let z = Complex64::new(0.0, 1.0);\n    let r = rule!(a => b);\n}\n"),
]

CASES += [
    # a module imported by name moves with its parent when it keeps its own name
    case("paths.use",
         "use symplex::stats::numdist::{gamma, norm, betainc_regularized_f64};\nfn t() { let q = norm::ppf(0.5); }\n",
         "use symplex::experimental::stats::numdist::{gamma, norm, betainc_regularized_f64};\nfn t() { let q = norm::ppf(0.5); }\n"),
]
