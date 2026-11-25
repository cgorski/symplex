"""Views (doctests in doc comments, Markdown blocks), imports, and the engine's guarantees."""
import os
import re
import tempfile
import unittest

import support
from support import run
from cm import engine
from cm.engine import MANUAL
from cm.views import FileUnit, parse_use, render_use
from cm.lexer import Lexed


def out(ruleset, src, **kw):
    return run(ruleset, src, **kw)[0]


def read(path):
    with open(path) as f:
        return f.read()


class DocViews(unittest.TestCase):
    SRC = (
        "//! Crate docs.\n"
        "//!\n"
        "//! ```\n"
        "//! use symplex::prelude::*;\n"
        "//! assert_eq!(x.is_zero(), Some(true));\n"
        "//! ```\n"
        "\n"
        "/// Text with `x.is_zero() == Some(true)` in prose.\n"
        "///\n"
        "/// ```text\n"
        "/// assert_eq!(y.is_zero(), Some(true));\n"
        "/// ```\n"
        "///\n"
        "/// ```no_run\n"
        "/// # use symplex::prelude::*;\n"
        "/// # fn main() {\n"
        "/// let ok = e.is_zero()\n"
        "///     == Some(false);\n"
        "/// # }\n"
        "/// ```\n"
        "pub fn f() { let internal = z.is_zero() == Some(true); }\n"
    )

    def test_blocks_found(self):
        u = FileUnit("/x/src/a.rs", "src/a.rs", self.SRC, "src")
        views = u.views(["doc"])
        self.assertEqual([v.info for v in views], ["", "no_run"])  # the `text` block is not Rust
        self.assertEqual(views[0].text, "use symplex::prelude::*;\nassert_eq!(x.is_zero(), Some(true));\n")
        # hidden lines are code: the marker is blanked, the offsets stay
        self.assertEqual(views[1].text.splitlines()[0], "  use symplex::prelude::*;")
        self.assertEqual(views[1].line0, 15)

    def test_rewrites_only_the_rust_blocks_of_the_library(self):
        got = out("truth", self.SRC, corpus="src")
        want = (self.SRC
                .replace("//! assert_eq!(x.is_zero(), Some(true));", "//! assert_eq!(x.is_zero(), Truth::True);")
                .replace("/// let ok = e.is_zero()\n///     == Some(false);", "/// let ok = e.is_zero().is_false();"))
        self.assertEqual(got, want)  # prose, the `text` block and the library's own code are untouched

    def test_library_code_only_on_request(self):
        src = "use num_bigint::BigInt;\n/// ```\n/// let n = BigInt::from(1);\n/// ```\npub fn f(n: BigInt) {}\n"
        self.assertEqual(out("numbers", src, corpus="src"),
                         "use num_bigint::BigInt;\n/// ```\n/// # use symplex::num::Integer;\n/// let n = Integer::from(1);\n"
                         "/// ```\npub fn f(n: BigInt) {}\n")
        self.assertEqual(out("numbers", src, corpus="src", src_code=True),
                         "use symplex::num::Integer;\n/// ```\n/// # use symplex::num::Integer;\n/// let n = Integer::from(1);\n"
                         "/// ```\npub fn f(n: Integer) {}\n")

    def test_multi_line_replacement_keeps_the_comment_marker(self):
        src = ("    /// ```\n    /// use symplex::matrix::{Matrix, jacobian};\n    /// use symplex::linprog::{q, LpProblem};\n"
               "    /// ```\n    fn f() {}\n")
        self.assertEqual(out("paths", src, corpus="src"),
                         "    /// ```\n    /// use symplex::linalg::Matrix;\n    /// use symplex::matrix::jacobian;\n"
                         "    /// use symplex::num::q;\n    /// use symplex::lp::LpProblem;\n    /// ```\n    fn f() {}\n")

    def test_import_into_a_doctest(self):
        # a doctest whose `use` lines are hidden gets a hidden import, placed after them
        src = ("/// ```\n/// # use symplex::prelude::*;\n/// # let r = f();\n"
               "/// assert!(matches!(r, Err(SymplexError::Divergent { .. })));\n/// ```\nfn f() {}\n")
        self.assertEqual(out("errors", src, corpus="src"),
                         "/// ```\n/// # use symplex::prelude::*;\n/// # use symplex::errors::ErrorKind;\n/// # let r = f();\n"
                         "/// assert!(matches!(r, Err(e) if e.kind() == ErrorKind::Divergent));\n/// ```\nfn f() {}\n")
        # visible `use` lines: a visible import; an empty `///` line after them keeps its bare marker
        src = ("/// ```\n/// use symplex::prelude::*;\n///\n/// let b = matches!(r, Err(SymplexError::Divergent { .. }));\n"
               "/// ```\nfn f() {}\n")
        self.assertEqual(out("errors", src, corpus="src"),
                         "/// ```\n/// use symplex::prelude::*;\n/// use symplex::errors::ErrorKind;\n///\n"
                         "/// let b = matches!(r, Err(e) if e.kind() == ErrorKind::Divergent);\n/// ```\nfn f() {}\n")

    def test_removed_import_line_in_a_doctest(self):
        # the removed line goes with its own marker; the import lands before the bare `///` line
        src = ("/// ```\n/// use symplex::combinatorics::factorial;\n/// use num_bigint::BigInt;\n///\n"
               "/// assert_eq!(factorial(5), BigInt::from(120));\n/// ```\nfn f() {}\n")
        self.assertEqual(out("numbers", src, corpus="src"),
                         "/// ```\n/// use symplex::combinatorics::factorial;\n/// use symplex::num::Integer;\n///\n"
                         "/// assert_eq!(factorial(5), Integer::from(120));\n/// ```\nfn f() {}\n")
        src = "    /// ```\n    /// use num_traits::Zero;\n    /// let z = x.is_zero();\n    /// ```\n    fn f() {}\n"
        self.assertEqual(out("numbers", src, corpus="src"),
                         "    /// ```\n    /// let z = x.is_zero();\n    /// ```\n    fn f() {}\n")

    def test_compile_fail_blocks_are_listed_not_rewritten(self):
        src = "/// ```compile_fail\n/// use symplex::prelude::*;\n/// let b = x.is_zero() == Some(true);\n/// ```\nfn f() {}\n"
        text, rw, man, _ = run("truth", src, corpus="src")
        self.assertEqual(text, src)
        self.assertTrue(any("compile_fail" in s.reason for s in man))


class MarkdownViews(unittest.TestCase):
    def test_only_rust_fences(self):
        src = ("Prose: `e.is_zero() == Some(true)`.\n\n```rust\nlet a = e.is_zero() == Some(true);\n```\n\n"
               "```python\ne.is_zero() == Some(true)\n```\n\n```\nlet b = e.is_zero() == Some(true);\n```\n\n"
               "```rust,ignore\n# use symplex::prelude::*;\nlet c = e.is_zero() == Some(false);\n```\n")
        want = (src.replace("let a = e.is_zero() == Some(true);", "let a = e.is_zero().is_true();")
                .replace("let c = e.is_zero() == Some(false);", "let c = e.is_zero().is_false();"))
        self.assertEqual(out("truth", src, kind="md", corpus="book"), want)

    def test_fragment_gets_no_import(self):
        # a snippet without any `use` line is not a program: no import is invented for it
        src = "```rust\nlet b = matches!(r, Err(SymplexError::Divergent { .. }));\n```\n"
        self.assertEqual(out("errors", src, kind="md", corpus="book"),
                         "```rust\nlet b = matches!(r, Err(e) if e.kind() == ErrorKind::Divergent);\n```\n")
        src = "```rust\nuse symplex::prelude::*;\nlet b = matches!(r, Err(SymplexError::Divergent { .. }));\n```\n"
        self.assertEqual(out("errors", src, kind="md", corpus="book"),
                         "```rust\nuse symplex::prelude::*;\nuse symplex::errors::ErrorKind;\n"
                         "let b = matches!(r, Err(e) if e.kind() == ErrorKind::Divergent);\n```\n")

    def test_block_that_does_not_tokenize_is_reported(self):
        src = "```rust\nlet a = e.is_zero() == Some(true);\nmatch r { Ok(v) => …, Err(SymplexError::X { .. } => \n```\n"
        text, rw, man, notes = run("truth", src, kind="md", corpus="book")
        self.assertEqual(text, src)
        self.assertTrue(notes and "does not tokenize" in notes[0])


class Imports(unittest.TestCase):
    def test_use_trees(self):
        lx = Lexed("use a::{b::{c, d as e}, f::*, g::{self, h}};")
        leaves, semi = parse_use(lx, 0)
        self.assertEqual([("::".join(s), al) for s, al, _a, _b in leaves],
                         [("a::b::c", None), ("a::b::d", "e"), ("a::f::*", None), ("a::g", None), ("a::g::h", None)])
        self.assertEqual(render_use([(["a", "b", "c"], None), (["a", "x"], None), (["a", "b", "d"], "e")], "  "),
                         "use a::b::{c, d as e};\n  use a::x;")

    def test_truth_needs_an_import_only_without_the_prelude(self):
        body = "fn t() {\n    assert_eq!(x.is_zero(), Some(true));\n}\n"
        self.assertEqual(out("truth", "use symplex::prelude::*;\n" + body),
                         "use symplex::prelude::*;\n" + body.replace("Some(true)", "Truth::True"))
        self.assertEqual(out("truth", "use symplex::Context;\n\n" + body),
                         "use symplex::Context;\nuse symplex::Truth;\n\n" + body.replace("Some(true)", "Truth::True"))

    def test_import_in_a_file_without_use_lines(self):
        src = "//! Module docs.\n#![allow(dead_code)]\n\n/// About t.\n#[test]\nfn t() {\n    assert_eq!(x.is_zero(), None);\n}\n"
        self.assertEqual(out("truth", src),
                         "//! Module docs.\n#![allow(dead_code)]\n\nuse symplex::Truth;\n\n/// About t.\n#[test]\nfn t() {\n"
                         "    assert_eq!(x.is_zero(), Truth::Unknown);\n}\n")

    def test_import_goes_into_the_module_that_needs_it(self):
        src = ("use symplex::Context;\n\nmod inner {\n    use symplex::Context;\n\n    fn t() {\n"
               "        assert_eq!(x.is_zero(), Some(false));\n    }\n}\n\nmod sees_parent {\n    use super::*;\n"
               "    fn u() -> bool { y.is_zero() == Some(true) }\n}\n")
        got = out("truth", src)
        self.assertIn("mod inner {\n    use symplex::Context;\n    use symplex::Truth;\n", got)
        self.assertIn("Truth::False", got)
        self.assertEqual(got.count("use symplex::Truth;"), 1)  # `==` needs no import; `inner` got its own

    def test_name_clash_writes_the_full_path(self):
        src = "use std::io::ErrorKind;\nfn t() { let b = matches!(r, Err(SymplexError::Divergent { .. })); }\n"
        self.assertEqual(out("errors", src),
                         "use std::io::ErrorKind;\nfn t() { let b = matches!(r, Err(e) if e.kind() == "
                         "symplex::errors::ErrorKind::Divergent); }\n")

    def test_imports_are_grouped(self):
        src = ("use num_bigint::BigInt;\nuse num_rational::Ratio;\nuse std::fmt;\n"
               "fn t(a: BigInt, b: Ratio<BigInt>) {}\n")
        self.assertEqual(out("numbers", src), "use std::fmt;\nuse symplex::num::{Integer, Rational};\nfn t(a: Integer, b: Rational) {}\n")


class Engine(unittest.TestCase):
    CANON = {
        "truth": "assert_eq!(x.is_zero(), Some(true)); let b = e.is_real() == Some(false);",
        "errors": "matches!(r, Err(SymplexError::Divergent { .. })); let v = MultiPoly::<Lex>::var(2, 0);",
        "numbers": "let n: BigInt = BigInt::from(1); let r: Ratio<BigInt> = make();",
        "core-renames": "let a = e.piecewise_simplify(); let b = e.powdenest(true); let c = Ex::symbolic_sum(&b, &k, &l, &h);",
        "subs": "let y = f.subs_i64(&x, 3);",
        "paths": "let m = symplex::matrix::Matrix::zeros(&ctx, 1, 1);",
        "domains": "let d = Distribution::normal(m, s); let e = m.eigenvals(); let p = r.p_value_f64();",
    }

    def test_canonical_inputs_are_rewritten(self):
        for name, code in self.CANON.items():
            src = "use symplex::prelude::*;\nfn t() { %s }\n" % code
            self.assertNotEqual(out(name, src), src, name)

    def test_never_inside_strings_or_comments(self):
        for name, code in self.CANON.items():
            for wrap in ('let s = "%s";', 'let s = r#"%s"#;', "// %s", "/* %s */", "/* a /* %s */ b */",
                         'let s = b"%s";'):
                src = "use symplex::prelude::*;\nfn t() {\n    %s\n}\n" % (wrap % code)
                text, rw, man, _ = run(name, src)
                self.assertEqual(text, src, "%s inside %s" % (name, wrap))
                self.assertEqual(rw, [])

    def test_every_rule_set_is_idempotent_on_its_canonical_input(self):
        for name, code in self.CANON.items():
            src = "use symplex::prelude::*;\nfn t() { %s }\n" % code
            once = out(name, src)
            self.assertEqual(out(name, once), once, name)

    def test_nested_sites_reach_a_fixpoint(self):
        # the outer rewrite moves text that holds another site: two passes, one result
        src = "use symplex::prelude::*;\nfn t() { let s = Ex::symbolic_sum(&t.subs_map_with(&m).trig_combine(), &k, &a, &b); }\n"
        once = out("core-renames", src)
        self.assertEqual(once, "use symplex::prelude::*;\nfn t() { let s = t.subs_map_with(&m).combine_trig()"
                               ".formal_summation(&k, &a, &b); }\n")
        self.assertEqual(out("core-renames", once), once)

    def test_skip_marker(self):
        src = ("use symplex::prelude::*;\nfn t() {\n    let a = f.subs_i64(&x, 1); // codemod:skip (a trait method of ours)\n"
               "    // codemod:skip\n    let b = f.subs_i64(&x, 2);\n    let c = f.subs_i64(&x, 3);\n}\n")
        text, rw, man, _ = run("subs", src)
        self.assertEqual(text, src.replace("let c = f.subs_i64", "let c = f.subs"))
        self.assertEqual(len([s for s in man if "codemod:skip" in s.reason]), 2)

    def test_ui_tests_with_recorded_output_are_frozen(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "tests", "ui"))
        rs = os.path.join(d, "tests", "ui", "bad.rs")
        with open(rs, "w") as f:
            f.write("fn main() { let y = f.subs_i64(&x, 3); }\n")
        with open(rs[:-3] + ".stderr", "w") as f:
            f.write("error[E0599]\n")
        res = engine.run(d, ["tests"], support.RULESETS["subs"], engine.Cx())
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].new, res[0].old)
        self.assertTrue(res[0].manual and "recorded compiler output" in res[0].manual[0].reason)

    def test_dry_run_writes_nothing_and_apply_is_a_fixpoint(self):
        import codemod
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "tests"))
        os.makedirs(os.path.join(d, "src"))
        open(os.path.join(d, "Cargo.toml"), "w").close()
        p = os.path.join(d, "tests", "a.rs")
        src = "use symplex::prelude::*;\nfn t() { assert_eq!(x.is_zero(), Some(true)); }\n"
        with open(p, "w") as f:
            f.write(src)
        diff = os.path.join(d, "out.diff")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            codemod.main(["truth", "--repo", d, "--diff", diff, "--quiet"])
        self.assertEqual(read(p), src)
        self.assertIn("+fn t() { assert_eq!(x.is_zero(), Truth::True); }", read(diff))
        with contextlib.redirect_stdout(io.StringIO()):
            codemod.main(["truth", "--repo", d, "--apply", "--quiet"])
        once = read(p)
        self.assertIn("Truth::True", once)
        with contextlib.redirect_stdout(io.StringIO()):
            codemod.main(["truth", "--repo", d, "--apply", "--quiet"])
        self.assertEqual(read(p), once)

    def test_triggers_cover_their_tables(self):
        # a rule's cheap prefilter must match every name the rule keys on (a hand-written
        # pattern once missed `poly_is_positive_on`)
        cx = engine.Cx()
        truth = [r for r in support.RULESETS["truth"].rules if r.id == "truth.cmp"][0]
        for row in cx.table("truth_queries"):
            if row["fate"] != "not-a-query":
                self.assertTrue(truth.trigger.search("x.%s()" % row["name"]), row["name"])
        plain = [r for r in support.RULESETS["errors"].rules if r.id == "errors.plain-unwrap"][0]
        tryr = [r for r in support.RULESETS["errors"].rules if r.id == "errors.try-rename"][0]
        for row in cx.table("try_pairs"):
            if row["class"] in ("b", "cV", "cP") and row["container"] != "Distribution" \
                    and row["try"] not in ("try_symbol", "try_replace"):
                self.assertTrue(plain.trigger.search("%s(1)" % row["plain"]), row["plain"])
                self.assertTrue(tryr.trigger.search("%s(1)" % row["try"]), row["try"])
        for name in ("core-renames", "subs", "domains"):
            for r in support.RULESETS[name].rules:
                old = getattr(r, "old", None)
                if old:
                    self.assertTrue(r.trigger.search("a.%s(b)" % old), r.id)
        dist = [r for r in support.RULESETS["domains"].rules if r.id == "domains.dist-ctor"][0]
        for row in cx.table("dist_ctors"):
            self.assertTrue(dist.trigger.search("Distribution::try_%s(a)" % row["ctor"]), row["ctor"])

    def test_tables_are_consistent(self):
        cx = engine.Cx()
        kinds = set("Divergent Infinite NoSolution InfiniteSolutions Singular NotDiagonalizable DoesNotExist "
                    "DegenerateData NonElementary Undetermined Unsupported NotConverged SizeLimit PrecisionExhausted "
                    "Underflow Overflow BudgetExhausted DeadlineExceeded Cancelled InvalidArgument FreeSymbol Syntax "
                    "ContradictoryAssumptions Internal".split())  # API_1.0.md §2.3: the 24 kinds
        self.assertEqual(len(kinds), 24)
        rows = cx.table("error_kinds")
        self.assertEqual(len(rows), 15)  # today's 15 variants
        for r in rows:
            self.assertIn(r["kind"], kinds, r["variant"])
        same = [r["variant"] for r in rows if r["variant"] == r["kind"]]
        self.assertEqual(len(same), 11)  # C §3.2: "V' is V for 11 variants"
        self.assertEqual(len([r for r in cx.table("dist_ctors") if r["rewrite"] == "yes"]), 22)
        for name in ("core_renames", "domain_renames"):
            ids = [r["id"] for r in cx.table(name)]
            self.assertEqual(len(ids), len(set(ids)), name)


if __name__ == "__main__":
    unittest.main()
