# Cases of the rule set `numbers` (review D §3, "D-R2 — outside src/").
from support import case

P = "use symplex::prelude::*;\n"

CASES = [
    # ---- D item 1: BigInt -> Integer, and its import (Part 4, C-12: the new import line is written)
    case("numbers.bigint",
         P + "use num_bigint::BigInt;\n\nfn t() {\n    let n: BigInt = BigInt::from(7) + BigInt::one();\n    let v: Vec<BigInt> = vec![];\n}\n",
         P + "use symplex::num::Integer;\n\nfn t() {\n    let n: Integer = Integer::from(7) + Integer::one();\n    let v: Vec<Integer> = vec![];\n}\n"),
    case("numbers.bigint",  # through symplex's re-export, and written as a path
         P + "use symplex::num_bigint::BigInt;\nfn t() {\n    let n = num_bigint::BigInt::from(1);\n    let m = BigInt::zero();\n}\n",
         P + "use symplex::num::Integer;\nfn t() {\n    let n = Integer::from(1);\n    let m = Integer::zero();\n}\n"),
    case("numbers.bigint",  # a string, a comment and a method named like the type are not the type
         P + "use symplex::num::Integer;\nfn t() {\n    let s = \"BigInt::from(1)\"; // BigInt\n    let k = x.BigInt;\n}\n",
         silent=True),
    case("numbers.bigint",  # a file with its own BigInt is not touched
         "struct BigInt(i64);\nfn t() -> BigInt { BigInt(1) }\n", silent=True),
    # ---- D item 2: the rational type under its three names
    case("numbers.rational",
         P + "use num_bigint::BigInt;\nuse num_rational::Ratio;\nfn half() -> Ratio<BigInt> {\n"
             "    Ratio::from_integer(BigInt::from(1)) / Ratio::from_integer(BigInt::from(2))\n}\n",
         P + "use symplex::num::{Integer, Rational};\nfn half() -> Rational {\n"
             "    Rational::from(Integer::from(1)) / Rational::from(Integer::from(2))\n}\n"),
    case("numbers.rational",
         P + "use num_rational::BigRational;\nfn t(a: &[BigRational]) -> BigRational { BigRational::zero() }\n",
         P + "use symplex::num::Rational;\nfn t(a: &[Rational]) -> Rational { Rational::zero() }\n"),
    case("numbers.rational",  # Q, where the file takes it from symplex
         "use symplex::linprog::{q, qi, Q};\nfn t(v: &[Q]) -> Q {\n    let z = Q::from_integer(3.into());\n    q(1, 2) + qi(3) + Q::zero()\n}\n",
         "use symplex::linprog::{q, qi};\nuse symplex::num::Rational;\nfn t(v: &[Rational]) -> Rational {\n"
         "    let z = Rational::from(3.into());\n    q(1, 2) + qi(3) + Rational::zero()\n}\n"),
    case("numbers.rational",  # Q of the file's own: a generic parameter, a matrix named Q
         P + "fn orth<Q: Clone>(m: Q) -> Q { m }\n", silent=True),
    case("numbers.rational",
         "fn t() {\n    let Q = qr.q;\n    let R = qr.r;\n    show(Q);\n}\n", silent=True),
    case("numbers.rational",  # Ratio over machine integers is another type
         "use num_rational::Ratio;\nfn t() -> Ratio<i64> { Ratio::new(1, 2) }\n", manual="another integer type"),
    case("numbers.rational",
         "fn t(v: Vec<Q>) {}\n", manual="not imported from symplex"),
    # ---- D item 3: trait imports
    case("numbers.trait-imports",
         P + "use num_traits::{One, Signed, ToPrimitive, Zero};\nuse std::fmt;\nfn t() {}\n",
         P + "use std::fmt;\nfn t() {}\n"),
    case("numbers.trait-imports",  # a trait D does not name stays, and is listed
         P + "use num_traits::{Pow, Zero};\nfn t() {}\n",
         P + "use num_traits::Pow;\nfn t() {}\n"),
    case("numbers.trait-imports",  # the trait `Integer` of num-integer would collide with the type
         P + "use num_bigint::BigInt;\nuse num_integer::Integer;\nfn t() -> BigInt { BigInt::from(6).gcd(&BigInt::from(4)) }\n",
         P + "use symplex::num::Integer;\nfn t() -> Integer { Integer::from(6).gcd(&Integer::from(4)) }\n"),
    # ---- D item 4: Complex64
    case("numbers.complex-import",
         P + "use num_complex::Complex64;\nfn t() -> Complex64 { Complex64::new(0.0, 1.0) }\n",
         P + "use symplex::num::Complex64;\nfn t() -> Complex64 { Complex64::new(0.0, 1.0) }\n"),
    # ---- listed for a person
    case("numbers.ratio-new",
         P + "use num_rational::Ratio;\nuse num_bigint::BigInt;\nfn t() { let h = Ratio::new(BigInt::from(1), BigInt::from(2)); }\n",
         P + "use symplex::num::{Integer, Rational};\nfn t() { let h = Rational::new(Integer::from(1), Integer::from(2)); }\n",
         manual="returns `Result` at 1.0"),
    case("numbers.ratio-new",  # once a person has added the reading it is no longer listed
         P + "use symplex::num::Rational;\nfn t() { let h = Rational::new(a, b).unwrap(); let g = Rational::new(c, d)?; }\n",
         silent=True),
    case("numbers.by-hand",
         P + "use num_bigint::BigInt;\nfn t() { let n = BigInt::parse_bytes(b\"12\", 10).unwrap(); }\n",
         P + "use symplex::num::Integer;\nfn t() { let n = Integer::parse_bytes(b\"12\", 10).unwrap(); }\n",
         manual="parse_bytes"),
    case("numbers.by-hand",
         P + "use num_bigint::{BigInt, BigUint, Sign};\nfn t(n: BigInt) {}\n",
         P + "use num_bigint::{BigUint, Sign};\nuse symplex::num::Integer;\nfn t(n: Integer) {}\n",
         manual="no 1.0 counterpart"),
    # ---- D-R2x renames
    case("numbers.renames",
         P + "fn t() {\n    let a = ctx.from_bigint(n);\n    let b = ctx.from_ratio(r);\n    let c = e.as_bigint();\n    let d = ctx.from_u64(7);\n}\n",
         P + "fn t() {\n    let a = ctx.from_integer(n);\n    let b = ctx.from_rational(r);\n    let c = e.as_integer();\n    let d = ctx.from_integer(7);\n}\n"),
    case("numbers.renames",
         P + "fn t() { let m = Matrix::from_bigint(&ctx, rows); }\n", manual="QMatrix"),
]

CASES += [
    # the file's own alias `type Q = Ratio<BigInt>;` keeps its name; the calls through it still change
    case("numbers.rational",
         "use num_bigint::BigInt;\nuse num_rational::Ratio;\ntype Q = Ratio<BigInt>;\nfn one() -> Q { Q::from_integer(BigInt::from(1)) }\n",
         "use symplex::num::{Integer, Rational};\ntype Q = Rational;\nfn one() -> Q { Q::from(Integer::from(1)) }\n"),
    case("numbers.ratio-new",
         "use num_bigint::BigInt;\nuse num_rational::Ratio;\ntype Q = Ratio<BigInt>;\nfn h() -> Q { Q::new(BigInt::from(1), BigInt::from(2)) }\n",
         "use symplex::num::{Integer, Rational};\ntype Q = Rational;\nfn h() -> Q { Q::new(Integer::from(1), Integer::from(2)) }\n",
         manual="returns `Result` at 1.0"),
]
