# Cases of the rule set `truth` (review B §3.2).  The files import the prelude, as 155 of the
# 160 test files with sites do; the import cases are in test_views.py.
from support import case

P = "use symplex::prelude::*;\n"


def f(body):
    return P + "fn t() {\n" + body + "}\n"


CASES = [
    # ---- B shape 1-3: assert_eq! with a value
    case("truth.assert-eq-true",
         f("    assert_eq!(x.is_positive(), Some(true));\n"),
         f("    assert_eq!(x.is_positive(), Truth::True);\n")),
    case("truth.assert-eq-true",  # either operand order, a message, a receiver chain with arguments
         f('    assert_eq!(Some(true), a.is_subset(&ctx.universal_set()), "{a} ⊆ U");\n'),
         f('    assert_eq!(Truth::True, a.is_subset(&ctx.universal_set()), "{a} ⊆ U");\n')),
    case("truth.assert-eq-true",  # spread over lines, as rustfmt writes the long ones
         f("    assert_eq!(\n        (&m - &m.transpose()).is_zero(),\n        Some(true),\n        \"sym\"\n    );\n"),
         f("    assert_eq!(\n        (&m - &m.transpose()).is_zero(),\n        Truth::True,\n        \"sym\"\n    );\n")),
    case("truth.assert-eq-true",  # proptest's macro has the same two leading arguments
         f("    prop_assert_eq!(expr.query(Props::INTEGER), Some(true));\n"),
         f("    prop_assert_eq!(expr.query(Props::INTEGER), Truth::True);\n")),
    case("truth.assert-eq-false",
         f("    assert_eq!(reals.contains(&i), Some(false));\n"),
         f("    assert_eq!(reals.contains(&i), Truth::False);\n")),
    case("truth.assert-eq-none",
         f("    assert_eq!(a.contains(&x), None);\n"),
         f("    assert_eq!(a.contains(&x), Truth::Unknown);\n")),
    # the receiver is not a call of a query: nothing says that the value is a query's answer
    case("truth.assert-eq-true", f("    assert_eq!(flag, Some(true));\n"), silent=True),
    case("truth.assert-eq-true", f("    assert_eq!(stack.pop(), Some(true));\n"), silent=True),
    case("truth.assert-eq-true", f("    assert_eq!(v.get(0).copied(), Some(true));\n"), silent=True),
    # a query that does not return Truth at 1.0 (Part 4): listed with the reason
    case("truth.assert-eq-true",
         f("    assert_eq!(p.poly_is_positive_on(&x, &a, &b), Some(true));\n"),
         manual="moves from `Ex` to `Poly`"),
    case("truth.assert-eq-false",
         f("    assert_eq!(e.is_prime_value(), Some(false));\n"), manual="removed at 1.0"),
    case("truth.assert-eq-true",
         f("    assert_eq!(m.is_hermitian(), Some(true));\n"), manual="two fates"),
    # the alternative style of the task brief
    case("truth.assert-eq-true",
         f('    assert_eq!(x.is_real(), Some(true), "{x}");\n'),
         f('    assert!(x.is_real().is_true(), "{x}");\n'), opts={"truth.assert": "is-true"}),
    case("truth.assert-eq-none",
         f("    assert_eq!(None, x.is_real());\n"),
         f("    assert!(x.is_real().is_unknown());\n"), opts={"truth.assert": "is-true"}),
    # ---- B shape 4: assert_ne!
    case("truth.assert-ne",
         f("    assert_ne!(x.is_zero(), Some(true));\n"),
         f("    assert_ne!(x.is_zero(), Truth::True);\n")),
    case("truth.assert-ne",
         f("    assert_ne!(x.is_zero(), Some(false));\n"),
         f("    assert!(!x.is_zero().is_false());\n"), opts={"truth.assert": "is-true"}),
    # ---- B shape 5: == / != with a value
    case("truth.cmp",
         f("    let ok = q.check_solution(&x, r) == Some(true);\n"),
         f("    let ok = q.check_solution(&x, r).is_true();\n")),
    case("truth.cmp",
         f("    if a.is_zero() == Some(false) && b.is_real() == None {}\n"),
         f("    if a.is_zero().is_false() && b.is_real().is_unknown() {}\n")),
    case("truth.cmp",  # the value first
         f("    assert!(Some(true) == a.equals(&b));\n"),
         f("    assert!(a.equals(&b).is_true());\n")),
    case("truth.cmp",  # `!=` is a policy reading: rewritten, and marked for review
         f("    assert!(r.equals(&y) != Some(true), \"msg\");\n"),
         f("    // codemod(truth): `!= Some(true)` read as \"not proved\" (not as \"proved false\"): check the reading\n"
           "    assert!(!r.equals(&y).is_true(), \"msg\");\n")),
    case("truth.cmp",
         f("    let known = e.is_integer() != None;\n"),
         f("    let known = e.is_integer().is_known();\n")),
    case("truth.cmp", f("    let b = m == Some(false);\n"), silent=True),
    case("truth.cmp", f("    let b = raw.get(\"k\").and_then(|v| v.as_bool()) == Some(true);\n"), silent=True),
    # ---- B shape 6: unwrap_or
    case("truth.unwrap-or",
         f("    if e.is_zero().unwrap_or(false) { return; }\n"),
         f("    if e.is_zero().is_true() { return; }\n")),
    case("truth.unwrap-or",
         f("    let maybe = e.is_zero().unwrap_or(true);\n"),
         f("    // codemod(truth): `.unwrap_or(true)` read as \"not refuted\": check the reading\n"
           "    let maybe = !e.is_zero().is_false();\n")),
    case("truth.unwrap-or",  # `!q.unwrap_or(true)` is exactly "proved false"
         f("    if !e.is_zero().unwrap_or(true) {}\n"),
         f("    if e.is_zero().is_false() {}\n")),
    case("truth.unwrap-or", f("    let n = count.checked_add(1).unwrap_or(false);\n"), silent=True),
    # ---- B shape 7: is_none / is_some
    case("truth.is-none",
         f("    assert!(m.is_none() || x.is_positive().is_none());\n"),
         f("    assert!(m.is_none() || x.is_positive().is_unknown());\n")),
    case("truth.is-none",
         f("    let known = e.is_real().is_some();\n"),
         f("    let known = e.is_real().is_known();\n")),
    # ---- match arms
    case("truth.match-arms",
         f("    match sys.is_stable() {\n        Some(true) => yes(),\n        Some(false) => {\n            no()\n        }\n"
           "        None => unknown(),\n    }\n"),
         f("    match sys.is_stable() {\n        Truth::True => yes(),\n        Truth::False => {\n            no()\n        }\n"
           "        Truth::Unknown => unknown(),\n    }\n")),
    case("truth.match-arms",
         f("    let s = match e.is_zero() { Some(true) => 1, Some(false) | None => 0 };\n"),
         f("    let s = match e.is_zero() { Truth::True => 1, Truth::False | Truth::Unknown => 0 };\n")),
    case("truth.match-arms",
         f("    match e.is_zero() { Some(b) => use_it(b), None => 0 };\n"), manual="an arm binds the value"),
    # ---- matches!
    case("truth.matches",
         f("    assert!(matches!(big.is_positive(), None | Some(true)));\n"),
         f("    assert!(matches!(big.is_positive(), Truth::Unknown | Truth::True));\n")),
    case("truth.matches",
         f("    assert!(matches!(e.is_zero(), Some(b) if b));\n"), manual="binds or has a guard"),
    # `contains` of a string inside a guard is not a query (review B's regular expression counted 15 of these)
    case("truth.matches",
         f("    assert!(matches!(&d, Err(E::U { reason }) if reason.contains(\"underflows\")));\n"), silent=True),
    # ---- queries that return bool today and Truth at 1.0
    case("truth.bool-query",
         f("    assert!(ode.check_ode_solution(&sol, &y, &x), \"{sol}\");\n"),
         f("    assert!(ode.check_ode_solution(&sol, &y, &x).is_true(), \"{sol}\");\n")),
    case("truth.bool-query",
         f("    if ss.is_controllable() && ss.is_observable() { go(); }\n"),
         f("    if ss.is_controllable().is_true() && ss.is_observable().is_true() { go(); }\n")),
    case("truth.bool-query",
         f("    assert!(!ode.check_ode_solution(&wrong, &y, &x));\n"), manual="negated assertion"),
    case("truth.bool-query",
         f("    let ok = ode.check_ode_solution(&s, &y, &x);\n"), manual="bound, returned or passed on"),
    # ---- other readings: found, never rewritten
    case("truth.other-reading",
         f("    if let Some(b) = e.is_zero() { use_it(b); }\n"), manual="`let` pattern"),
    case("truth.other-reading",
         f("    let v = e.is_strictly_increasing(&x, &r);\n"), manual="bound to a variable"),
    case("truth.other-reading",
         f("    assert!(e.is_zero().unwrap());\n"), manual="`.unwrap()` on the answer"),
    case("truth.other-reading",
         f("    assert_eq!(e.is_zero(), expected);\n"), manual="not a query"),
    # ---- renames
    case("truth.rename",
         f("    assert!(!s.is_constant());\n    let b = p.and(&q).satisfiable();\n"),
         f("    assert!(!s.is_symbol_free());\n    let b = p.and(&q).is_satisfiable();\n")),
    case("truth.rename",  # the rename and the reading are two passes of one run
         f("    assert_eq!(eq.is_satisfied(), Some(true));\n"),
         f("    assert_eq!(eq.is_identity(), Truth::True);\n")),
    case("truth.rename",
         f("    let g: GenPoly<Q> = make();\n    assert!(g.is_constant());\n"), manual="names `GenPoly`"),
    # ---- expected-output comments
    case("truth.printed-comment",
         "```rust\n" + f('    println!("{:?} {:?}", b.is_open(), a.is_closed());   // Some(false) Some(true)\n') + "```\n",
         "```rust\n" + f('    println!("{:?} {:?}", b.is_open(), a.is_closed());   // False True\n') + "```\n", kind="md",
         corpus="book"),
    case("truth.printed-comment",
         "```rust\na.contains(&x);                                       // None\n```\n",
         "```rust\na.contains(&x);                                       // Unknown\n```\n", kind="md", corpus="readme"),
    case("truth.printed-comment",  # `None` here is what `to_bigint_rows()` printed, not a query's answer
         "```rust\n" + f('    println!("{:?} {:?}", f.is_integer_matrix(), f.to_bigint_rows());   // Some(false) None\n')
         + "```\n", manual="2 value(s) for 1 query", kind="md", corpus="book"),
    # ---- mentions in prose
    case("truth.mention",
         f("    // (`[0, 2].contains(e)` was `Some(true)` before)\n    let y = 1;\n"), info="comment or prose"),
    case("truth.mention",  # `contains` of a string, in prose without an `Option<bool>` value: not a mention
         f("    // the name list: `names.contains(x)` decides\n    let y = 1;\n"), silent=True),
    case("truth.other-reading",
         f('    println!("{:?}", e.is_convergent(&k));\n'), info="printed"),
]
