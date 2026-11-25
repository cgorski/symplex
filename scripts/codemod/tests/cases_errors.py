# Cases of the rule set `errors` (review C §3.2).
from support import case

P = "use symplex::prelude::*;\n"
K = "use symplex::prelude::*;\nuse symplex::errors::ErrorKind;\n"  # the import the tool adds (Part 4, C-12)


def f(body, head=P):
    return head + "fn t() {\n" + body + "}\n"


CASES = [
    # ---- C item 1: matches!
    case("errors.matches",
         f("    assert!(matches!(r, Err(SymplexError::InvalidArgument { .. })), \"{r:?}\");\n"),
         f("    assert!(matches!(r, Err(e) if e.kind() == ErrorKind::InvalidArgument), \"{r:?}\");\n", K)),
    case("errors.matches",  # a tuple variant; NotImplemented -> Unsupported
         f("    assert!(matches!(x.try_limit(&y, &a), Err(SymplexError::NotImplemented(_))));\n"),
         f("    assert!(matches!(x.try_limit(&y, &a), Err(e) if e.kind() == ErrorKind::Unsupported));\n", K)),
    case("errors.matches",  # the value itself is matched: a comparison of kinds
         f("    assert!(matches!(err, SymplexError::NoSolution { .. }));\n"),
         f("    assert!(err.kind() == ErrorKind::NoSolution);\n", K)),
    case("errors.matches",
         f("    if !matches!(&e, SymplexError::Divergent { .. }) { fail(); }\n"),
         f("    if e.kind() != ErrorKind::Divergent { fail(); }\n", K)),
    case("errors.matches",  # two variants
         f("    let b = matches!(r, Err(SymplexError::NoSolution { .. } | SymplexError::InfiniteSolutions { .. }));\n"),
         f("    let b = matches!(r, Err(e) if matches!(e.kind(), ErrorKind::NoSolution | ErrorKind::InfiniteSolutions));\n",
           K)),
    case("errors.matches",  # the two catch-alls become Undetermined, and the line is marked
         f("    assert!(matches!(r, Err(SymplexError::ComputationFailed { .. })));\n"),
         f("    // codemod(errors): was `ComputationFailed` -> `Undetermined` by default; the raise site may get "
           "another kind (E2b-E2d)\n"
           "    assert!(matches!(r, Err(e) if e.kind() == ErrorKind::Undetermined));\n", K)),
    case("errors.matches",  # ExpressionSwell merges into BudgetExhausted; the path may be written out
         f("    let b = matches!(r, Err(symplex::errors::SymplexError::ExpressionSwell { .. }));\n"),
         f("    let b = matches!(r, Err(e) if e.kind() == ErrorKind::BudgetExhausted);\n", K)),
    case("errors.matches",
         f("    assert!(matches!(&d, Err(SymplexError::Infinite { .. }) if cond));\n"), manual="guard"),
    # ---- C item 1: arms
    case("errors.match-arm",
         f("    match r {\n        Err(SymplexError::Divergent { .. }) => {}\n        other => panic!(\"{other:?}\"),\n    }\n"),
         f("    match r {\n        Err(e) if e.kind() == ErrorKind::Divergent => {}\n        other => panic!(\"{other:?}\"),\n    }\n",
           K)),
    case("errors.match-arm",  # `e` is used in the arm: another name is taken
         f("    match e.try_limit(&x, &a) {\n        Err(SymplexError::Divergent { .. }) => println!(\"{e}\"),\n"
           "        _ => {}\n    }\n"),
         f("    match e.try_limit(&x, &a) {\n        Err(err) if err.kind() == ErrorKind::Divergent => println!(\"{e}\"),\n"
           "        _ => {}\n    }\n", K)),
    case("errors.match-arm",
         f("    if let Err(SymplexError::NoSolution { .. }) = eq.solve(&x) {\n        return;\n    }\n"),
         f("    if matches!(eq.solve(&x), Err(e) if e.kind() == ErrorKind::NoSolution) {\n        return;\n    }\n", K)),
    case("errors.match-arm",
         f("    match r {\n        Err(SymplexError::Divergent { .. }) if strict => {}\n        _ => {}\n    }\n"),
         manual="has a guard already"),
    # ---- patterns that bind: by hand (C §3.3)
    case("errors.other",
         f("    match r {\n        Err(SymplexError::InvalidArgument { reason, .. }) => assert!(reason.contains(\"x\")),\n"
           "        _ => {}\n    }\n"), manual="binds or tests a field"),
    case("errors.other",
         f("    assert!(matches!(r, Err(SymplexError::DegenerateData { kind: Degeneracy::Collinear, .. })));\n"),
         manual="binds or tests a field"),
    case("errors.other",
         f("    let e = SymplexError::InvalidArgument { operation: \"f\", reason: \"r\".into() };\n"),
         manual="constructs a variant"),
    case("errors.other",
         f("    match err {\n        SymplexError::Divergent { .. } => 1,\n        _ => 0,\n    };\n"),
         manual="outside `Err(..)`"),
    # ---- C item 2: the try_ twin takes the plain name where the Result is read at the call
    case("errors.try-rename",
         P + "use symplex::ntheory::try_fibonacci;\nfn t() {\n    assert!(try_fibonacci(1 << 40).is_err());\n}\n",
         P + "use symplex::ntheory::fibonacci;\nfn t() {\n    assert!(fibonacci(1 << 40).is_err());\n}\n"),
    case("errors.try-rename",  # both forms were imported: one leaf is left
         P + "use symplex::ntheory::{fibonacci, try_fibonacci, lucas};\nfn t() {\n    let n = try_fibonacci(9)?;\n}\n",
         P + "use symplex::ntheory::{fibonacci, lucas};\nfn t() {\n    let n = fibonacci(9)?;\n}\n"),
    case("errors.try-rename",
         f("    let rv = RandomVariable::try_new(&ctx, \"X\", d)?;\n    let p = Poly::try_new(&e, &[&x]).unwrap();\n"),
         f("    let rv = RandomVariable::new(&ctx, \"X\", d)?;\n    let p = Poly::new(&e, &[&x]).unwrap();\n")),
    case("errors.try-rename",  # bound, and read as a Result further down
         f("    let r = MultiPoly::<Lex>::try_var(2, 5);\n    assert!(r.is_err());\n"),
         f("    let r = MultiPoly::<Lex>::var(2, 5);\n    assert!(r.is_err());\n")),
    case("errors.try-rename",  # a method whose plain name the tool never touches: no reading needed
         f("    let s = a.try_add(&b);\n    take(s);\n"),
         f("    let s = a.add(&b);\n    take(s);\n")),
    case("errors.try-rename",  # passed on unread: a later run could not tell it from the old plain call
         f("    check(Rule::try_new(\"r\", &a, &b));\n"), manual="not read next to the call"),
    case("errors.try-rename",  # class a pairs stay
         f("    let l = e.try_limit(&x, &a)?;\n    let d = e.try_diff(&x)?;\n"), silent=True),
    # ---- C item 3: the plain call now returns Result
    case("errors.plain-unwrap",
         f("    let x = MultiPoly::<Lex>::var(2, 0);\n    let rv = RandomVariable::new(&ctx, \"X\", d);\n"),
         f("    let x = MultiPoly::<Lex>::var(2, 0).unwrap();\n    let rv = RandomVariable::new(&ctx, \"X\", d).unwrap();\n")),
    case("errors.plain-unwrap",  # `?` where the function returns a Result that can hold the error
         P + "fn main() -> Result<(), SymplexError> {\n    let r = Rule::new(\"r\", &a, &b);\n    Ok(())\n}\n",
         P + "fn main() -> Result<(), SymplexError> {\n    let r = Rule::new(\"r\", &a, &b)?;\n    Ok(())\n}\n",
         corpus="examples"),
    case("errors.plain-unwrap",  # ... but not inside a closure, and not when the error type is a String
         P + "fn main() -> Result<(), SymplexError> {\n    let v: Vec<_> = (0..3).map(|i| MultiPoly::<Lex>::var(3, i)).collect();\n"
             "    Ok(())\n}\n",
         P + "fn main() -> Result<(), SymplexError> {\n    let v: Vec<_> = (0..3).map(|i| MultiPoly::<Lex>::var(3, i).unwrap()).collect();\n"
             "    Ok(())\n}\n", corpus="examples"),
    case("errors.plain-unwrap",
         P + "fn run() -> Result<(), String> {\n    let r = Rule::new(\"r\", &a, &b);\n    Ok(())\n}\n",
         P + "fn run() -> Result<(), String> {\n    let r = Rule::new(\"r\", &a, &b).unwrap();\n    Ok(())\n}\n"),
    case("errors.plain-unwrap",  # a free function is taken only when a `use` line names it
         P + "use symplex::ntheory::{fibonacci, lucas};\nfn t() {\n    assert_eq!(fibonacci(10), BigInt::from(55));\n}\n",
         P + "use symplex::ntheory::{fibonacci, lucas};\nfn t() {\n    assert_eq!(fibonacci(10).unwrap(), BigInt::from(55));\n}\n"),
    case("errors.plain-unwrap",
         f("    assert_eq!(fibonacci(10), BigInt::from(55));\n"), manual="no `use` line of this file names it"),
    case("errors.plain-unwrap",
         P + "use symplex::ntheory::fibonacci;\nfn fibonacci(n: u64) -> u64 { n }\nfn t() { let a = fibonacci(3); }\n",
         manual="defines a function named `fibonacci`"),
    case("errors.plain-unwrap",  # `Poly::new` is an Option today: only the readers change
         f("    assert!(Poly::new(&e, &[&x]).is_none());\n    let p = Poly::new(&f, &[&x]).unwrap();\n"),
         f("    assert!(Poly::new(&e, &[&x]).is_err());\n    let p = Poly::new(&f, &[&x]).unwrap();\n")),
    # method names that other types share are never touched
    case("errors.plain-unwrap", f("    let s = a.add(&b);\n    let p = x.pow(2);\n    let g = u.gcd(&v);\n"), silent=True),
    # the Distribution constructors belong to `domains` (Part 4, C-22)
    case("errors.plain-unwrap", f("    let d = Distribution::normal(m, s);\n"), silent=True),
    # ---- C item 4: doc links
    case("errors.doc-link",
         "/// Fails with [`SymplexError::NotImplemented`] or [`SymplexError::InvalidArgument`].\nfn t() {}\n",
         "/// Fails with [`ErrorKind::Unsupported`] or [`ErrorKind::InvalidArgument`].\nfn t() {}\n", corpus="src"),
    case("errors.doc-link",  # not in a string, not in a plain comment
         "// see [`SymplexError::Divergent`]\nfn t() { let s = \"[`SymplexError::Divergent`]\"; }\n", silent=True,
         corpus="src"),
    # ---- C item 5: constructors, in library code only (--src-code)
    case("errors.construct",
         "fn f() -> SymplexError {\n    SymplexError::InvalidArgument { operation: \"f\", reason: format!(\"{n} < 0\") }\n}\n",
         "fn f() -> SymplexError {\n    SymplexError::invalid_argument(\"f\", format!(\"{n} < 0\"))\n}\n",
         corpus="src", src_code=True),
    case("errors.construct",  # shorthand fields, any order
         "fn f() -> SymplexError {\n    SymplexError::NoSolution { reason, operation }\n}\n",
         "fn f() -> SymplexError {\n    SymplexError::no_solution(operation, reason)\n}\n", corpus="src", src_code=True),
    case("errors.construct",
         "fn f() -> SymplexError {\n    SymplexError::Unevaluable { reason: r }\n}\n",
         manual="no constructor is named", corpus="src", src_code=True),
]

CASES += [
    # a doctest in the very file that defines the library function: the definition is not "a local function"
    case("errors.plain-unwrap",
         "/// ```\n/// use symplex::ntheory::fibonacci;\n/// assert_eq!(fibonacci(10), 55.into());\n/// ```\n"
         "pub fn fibonacci(n: u64) -> BigInt { todo!() }\n",
         "/// ```\n/// use symplex::ntheory::fibonacci;\n/// assert_eq!(fibonacci(10).unwrap(), 55.into());\n/// ```\n"
         "pub fn fibonacci(n: u64) -> BigInt { todo!() }\n", corpus="src"),
]

CASES += [
    # a module or a type reached through an alias of the file
    case("errors.try-rename",
         P + "use symplex::ntheory as nt;\nfn t() { let r = nt::try_sqrt_mod_all(0, n).unwrap(); let f = nt::fibonacci(9); }\n",
         P + "use symplex::ntheory as nt;\nfn t() { let r = nt::sqrt_mod_all(0, n).unwrap(); let f = nt::fibonacci(9).unwrap(); }\n"),
    case("errors.plain-unwrap",
         P + "fn t() {\n    type P = MultiPoly<GrevLex>;\n    let x = P::var(2, 0);\n}\n",
         P + "fn t() {\n    type P = MultiPoly<GrevLex>;\n    let x = P::var(2, 0).unwrap();\n}\n"),
    # a `try_` function the pairs table does not have on that type is listed, not skipped
    case("errors.try-rename",
         f("    assert!(OrderStatistic::try_new(u.clone(), 3, 4).is_err());\n"), manual="is not a pair of the table"),
]
