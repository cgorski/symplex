"""The matching core: tokenizer, bracket structure, argument splitting, postfix chains."""
import unittest

import support  # noqa: F401  (puts the tool on sys.path)
from cm.lexer import Lexed, ID, STR, CHAR, LIFE, NUM, is_boundary_before, is_boundary_after
from cm.chain import chain_start, chain_end, call_at, enclosing_macro, enclosing_fn, fn_returns_result, path_before


def toks(src):
    lx = Lexed(src)
    return [(k, t) for k, t in zip(lx.kind, lx.text)]


def texts(src):
    return Lexed(src).text


class Tokenizer(unittest.TestCase):
    def test_line_and_block_comments(self):
        lx = Lexed("a // b == Some(true)\nc /* d /* nested */ still */ e")
        self.assertEqual(lx.text, ["a", "c", "e"])
        self.assertEqual(len(lx.comments), 2)
        self.assertTrue(lx.ok)

    def test_doc_comment_kinds(self):
        lx = Lexed("/// outer\n//! inner\n// plain\n//// not doc\n/** outer block */\n/*! inner block */\n/**/ x")
        self.assertEqual([c.doc for c in lx.comments], ["outer", "inner", "", "", "outer", "inner", ""])

    def test_strings(self):
        self.assertEqual(texts(r'f("a \" b == Some(true)", x)'), ["f", "(", r'"a \" b == Some(true)"', ",", "x", ")"])
        lx = Lexed('let s = "line one\n  // not a comment\n";')
        self.assertEqual(lx.comments, [])
        self.assertEqual(lx.kind[3], STR)

    def test_raw_strings_with_any_number_of_hashes(self):
        for lit in ('r"a\\"', 'r#"a "quoted" b"#', 'r##"a "# b"##', 'br#"bytes " here"#', 'b"x\\"y"', 'c"c string"',
                    'cr#"raw c " string"#', 'r###"deep "## still"###'):
            lx = Lexed("let s = %s; done" % lit)
            self.assertTrue(lx.ok, lit)
            self.assertEqual(lx.text, ["let", "s", "=", lit, ";", "done"], lit)
            self.assertEqual(lx.kind[3], STR)

    def test_raw_identifier_is_not_a_raw_string(self):
        self.assertEqual(toks("r#type + r")[0], (ID, "r#type"))

    def test_chars_and_lifetimes(self):
        lx = Lexed("fn f<'a>(x: &'a str, c: char) -> bool { c == 'a' || c == '\\'' || c == '(' || c == b'x' }")
        self.assertTrue(lx.ok)
        self.assertEqual([t for k, t in zip(lx.kind, lx.text) if k == LIFE], ["'a", "'a"])
        self.assertEqual([t for k, t in zip(lx.kind, lx.text) if k == CHAR], ["'a'", "'\\''", "'('", "b'x'"])

    def test_bracket_in_a_char_or_string_does_not_unbalance(self):
        self.assertTrue(Lexed("let a = ['(', ')', '{']; let s = \"}}}\"; let l = 'outer: loop { break 'outer; };").ok)

    def test_unicode_escape_and_static(self):
        lx = Lexed("const C: &'static str = \"x\"; let u = '\\u{1F600}'; let r = 'ℝ';")
        self.assertTrue(lx.ok)
        self.assertEqual([t for k, t in zip(lx.kind, lx.text) if k == CHAR], ["'\\u{1F600}'", "'ℝ'"])

    def test_numbers_are_opaque(self):
        self.assertEqual(texts("1_000u64 + 0x1F + 1.5e-3 + 1..2 + x.0.max(2) + 2.pow(3)"),
                         ["1_000u64", "+", "0x1F", "+", "1.5e-3", "+", "1", "..", "2", "+", "x", ".", "0", ".", "max",
                          "(", "2", ")", "+", "2", ".", "pow", "(", "3", ")"])

    def test_operators(self):
        self.assertEqual(texts("a == b != c => d -> e :: f ..= g && h || i <= j >= k"),
                         ["a", "==", "b", "!=", "c", "=>", "d", "->", "e", "::", "f", "..=", "g", "&&", "h", "||", "i",
                          "<=", "j", ">=", "k"])
        # shifts are never joined, so that `Vec<Vec<T>>` closes twice
        self.assertEqual(texts("Vec<Vec<T>>"), ["Vec", "<", "Vec", "<", "T", ">", ">"])

    def test_brackets_and_parents(self):
        lx = Lexed("f(a, [b, {c}], d)")
        self.assertEqual(lx.mate[1], len(lx) - 1)
        c = lx.text.index("c")
        self.assertEqual(lx.text[lx.up[c]], "{")
        self.assertEqual(lx.text[lx.up[lx.up[c]]], "[")
        self.assertEqual(lx.up[0], -1)

    def test_text_that_does_not_lex(self):
        for bad in ("f(a, b", "f(a]", 'let s = "unterminated;', "a /* never closed", "x)"):
            self.assertFalse(Lexed(bad).ok, bad)
        self.assertIn("line 2", Lexed("ok();\nf(a]").error)

    def test_line_numbers(self):
        lx = Lexed("a\n\nb\n  c")
        self.assertEqual([lx.line_of(s) for s in lx.start], [1, 3, 4])
        self.assertEqual(lx.indent_at(lx.start[2]), "  ")


class Arguments(unittest.TestCase):
    def args(self, src):
        lx = Lexed(src)
        a = lx.split_args(lx.text.index("("))
        return None if a is None else [lx.span(x, y) for x, y in a]

    def test_plain(self):
        self.assertEqual(self.args("f(a, b + 1, g(c, d), [e, f], \"x, y\")"), ["a", "b + 1", "g(c, d)", "[e, f]", '"x, y"'])
        self.assertEqual(self.args("f()"), [])
        self.assertEqual(self.args("f(a, b,)"), ["a", "b"])
        self.assertEqual(self.args("f(a,\n  // a comment, with commas\n  b)"), ["a", "b"])

    def test_turbofish_commas(self):
        self.assertEqual(self.args("assert_eq!(v.iter().collect::<HashMap<_, _>>(), w)"),
                         ["v.iter().collect::<HashMap<_, _>>()", "w"])
        self.assertEqual(self.args("f(MultiPoly::<Lex, Q>::var(2, 0), 1)"), ["MultiPoly::<Lex, Q>::var(2, 0)", "1"])

    def test_closure_parameters(self):
        self.assertEqual(self.args("f(|a, b| a + b, c)"), ["|a, b| a + b", "c"])
        self.assertEqual(self.args("f(x, move |a, b| g(a, b))"), ["x", "move |a, b| g(a, b)"])
        self.assertEqual(self.args("f(a | b, c || d)"), ["a | b", "c || d"])

    def test_comparison_is_not_generics(self):
        self.assertEqual(self.args("f(a < b, c > d)"), ["a < b", "c > d"])

    def test_gives_up_when_unsure(self):
        self.assertIsNone(self.args("f(x as HashMap<K, V>, y)"))
        self.assertIsNone(self.args("f(<T as Trait<A, B>>::g(), y)"))


class Chains(unittest.TestCase):
    def start(self, src, last=None):
        lx = Lexed(src)
        i = len(lx) - 1 if last is None else lx.text.index(last)
        s = chain_start(lx, i)
        return None if s < 0 else lx.span(s, i)

    def test_receivers(self):
        for src, want in [
            ("x == a.b(1).c()", "a.b(1).c()"),
            ("!a.is_zero()", "a.is_zero()"),
            ("&ctx.int(3).is_in(&s)", "ctx.int(3).is_in(&s)"),
            ("y + (&m - &m.transpose()).is_zero()", "(&m - &m.transpose()).is_zero()"),
            ("return self.data[i][j].get(0)?.is_zero()", "self.data[i][j].get(0)?.is_zero()"),
            ("f(Matrix::identity(&ctx, 2).is_identity()", "Matrix::identity(&ctx, 2).is_identity()"),
            ("let v = vec![a, b].len()", "vec![a, b].len()"),
            ("x = expr!(ctx, x^2).is_positive()", "expr!(ctx, x^2).is_positive()"),
            ("x = a.collect::<Vec<_>>().len()", "a.collect::<Vec<_>>().len()"),
            ("x = MultiPoly::<Lex>::var(2, 0).degree()", "MultiPoly::<Lex>::var(2, 0).degree()"),
            ("x = \"abc\".len()", "\"abc\".len()"),
            ("x = t.0.is_zero()", "t.0.is_zero()"),
            ("if a > (b).c()", "(b).c()"),
            ("x = $e.is_zero()", "$e.is_zero()"),
            ("x = fut.await.is_zero()", "fut.await.is_zero()"),
            ("x = ::symplex::f(1)", "::symplex::f(1)"),
        ]:
            self.assertEqual(self.start(src), want, src)

    def test_keywords_are_not_callees(self):
        self.assertEqual(self.start("return (a + b).c()"), "(a + b).c()")
        self.assertEqual(self.start("x in (a).b()"), "(a).b()")

    def test_refuses_blocks_and_qualified_paths(self):
        self.assertIsNone(self.start("x = match a { _ => b }.is_zero()"))
        self.assertIsNone(self.start("x = Foo { a }.is_zero()"))
        self.assertIsNone(self.start("x = <T as Trait>::f().is_zero()"))

    def test_chain_end(self):
        for src, want in [
            ("a.b(1).c()? + d", "a.b(1).c()?"),
            ("Type::<T>::new(1).x[0] == y", "Type::<T>::new(1).x[0]"),
            ("vec![1, 2].len() , z", "vec![1, 2].len()"),
            ("(a + b).c * d", "(a + b).c"),
            ("x.y::<T>(1), z", "x.y::<T>(1)"),
        ]:
            lx = Lexed(src)
            e = chain_end(lx, 0)
            self.assertEqual(lx.span(0, e), want, src)
        self.assertEqual(chain_end(Lexed("if x {"), 0), -1)

    def test_calls(self):
        lx = Lexed("fn f(a: T) { g(1); a.h::<U>(2); T::k(3); }")
        found = [(lx.text[i], call_at(lx, i).style) for i in range(len(lx)) if lx.kind[i] == ID and call_at(lx, i)]
        self.assertEqual(found, [("g", "free"), ("h", "method"), ("k", "path")])  # `fn f(` is a definition
        lx = Lexed("symplex::stats::Distribution::normal(m, s)")
        self.assertEqual(path_before(lx, lx.text.index("normal")), ["symplex", "stats", "Distribution"])

    def test_enclosing_macro(self):
        lx = Lexed("assert_eq!(a.b(), Some(true), \"m\")")
        m = enclosing_macro(lx, lx.text.index("a"))
        self.assertEqual((m.name, [lx.span(x, y) for x, y in m.args]), ("assert_eq", ["a.b()", "Some(true)", '"m"']))
        self.assertIsNone(enclosing_macro(Lexed("f(a)"), 2))

    def test_enclosing_fn_and_closures(self):
        src = ("fn a() -> Result<(), SymplexError> {\n    let v = xs.iter().map(|x| inner(x)).collect();\n"
               "    if c { direct(1); }\n    let w = (|| { boxed(2) })();\n    Ok(())\n}\nfn b() { plain(3); }\n")
        lx = Lexed(src)

        def at(name):
            f = enclosing_fn(lx, lx.text.index(name))
            return (lx.text[f[0] + 1], f[2], bool(fn_returns_result(lx, f[0], f[1])))

        self.assertEqual(at("inner"), ("a", True, True))  # inside a closure: `?` would not leave `a`
        self.assertEqual(at("direct"), ("a", False, True))
        self.assertEqual(at("boxed"), ("a", True, True))
        self.assertEqual(at("plain"), ("b", False, False))

    def test_boundaries(self):
        lx = Lexed("if a == b && c { d }")
        self.assertTrue(is_boundary_before(lx, lx.text.index("a")))
        self.assertFalse(is_boundary_before(lx, lx.text.index("b")))
        self.assertTrue(is_boundary_after(lx, lx.text.index("b")))
        self.assertTrue(is_boundary_after(lx, lx.text.index("c")))
        self.assertFalse(is_boundary_after(lx, lx.text.index("a")))


if __name__ == "__main__":
    unittest.main()
