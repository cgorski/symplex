# Rule set `truth` (package T2): reads of the yes/no queries, `Option<bool>` -> `Truth`.
#
# Review B, REPORT §3.2 "M1", and its locator scripts/03_callsites.py.  The type of an
# expression is not known to a text tool, so every rule here keys on the NAME of the
# method that is called (tables/truth_queries.tsv, from B's tables and the merged fate
# table) and fires only when the value that is read is, visibly, the result of a call
# of such a method:
#
#     recv.is_zero() == Some(true)          a site
#     flag == Some(true)                    not a site (a variable: nothing says what it holds)
#     stack.pop() == Some(true)             not a site (`pop` is not a query)
#
# B's seven fixed shapes are the first seven rules; the others are the shapes B lists
# as "other readings", which the tool finds and, where B says a script can, rewrites.
import re

from .lexer import ID, PUNCT, OPEN, CLOSE, is_boundary_before, is_boundary_after
from .chain import chain_start, call_at, enclosing_macro, whole_arg, toks_are, stmt_start
from .engine import Rule, RuleSet, Site, REWRITE, MANUAL, INFO

TRUTH_PATH = "symplex::Truth"
VALUES = {("Some", "(", "true", ")"): "True", ("Some", "(", "false", ")"): "False", ("None",): "Unknown"}
ASSERT_EQ = {"assert_eq": "eq", "debug_assert_eq": "eq", "assert_ne": "ne", "debug_assert_ne": "ne",
             # proptest's forms take the same two leading arguments
             "prop_assert_eq": "eq", "prop_assert_ne": "ne"}
PRINT_MACROS = frozenset("println print eprintln eprint format write writeln dbg panic".split())
COMBINATORS = frozenset(
    "map and_then is_some_and is_none_or map_or filter or or_else ok_or ok_or_else unwrap_or_else into copied zip "
    "unwrap_or_default".split()
)
TRUTH_METHODS = frozenset(
    "is_true is_false is_unknown is_known known not and or implies iff if_unknown to_option or_else".split()
)
RENAMES = {
    # B §3.1 "renames" and API_1.0.md §3.4 "Queries (B)"
    "is_constant": ("is_symbol_free", "GenPoly"),
    "satisfiable": ("is_satisfiable", ""),
    "is_satisfied": ("is_identity", ""),
}



def _trigger():
    # built from the table itself: a hand-written pattern (`\bis_`) silently missed `poly_is_positive_on`
    from .rules_renames import _load
    names = sorted(set(r["name"] for r in _load("truth_queries") if r["fate"] != "not-a-query"), key=len, reverse=True)
    return re.compile(r"\b(?:%s)\b" % "|".join(names))


_TRIGGER = _trigger()


def _value_at(lx, a, b):
    """"True" / "False" / "Unknown" when tokens a..b spell Some(true) / Some(false) / None."""
    return VALUES.get(tuple(lx.text[a:b + 1]))


def _value_after(lx, i):
    """A value that starts at token i: (name, last_token) or (None, -1)."""
    if lx.is_id(i, "None"):
        return "Unknown", i
    if lx.is_id(i, "Some") and lx.is_open(i + 1, "(") and lx.mate[i + 1] == i + 3:
        v = _value_at(lx, i, i + 3)
        if v:
            return v, i + 3
    return None, -1


def _value_before(lx, i):
    """A value that ends at token i: (name, first_token) or (None, -1)."""
    if lx.is_id(i, "None"):
        return "Unknown", i
    if lx.is_close(i, ")") and i - 3 >= 0 and lx.mate[i] == i - 2:
        v = _value_at(lx, i - 3, i)
        if v:
            return v, i - 3
    return None, -1


def _is_truth_path(lx, a, b=None):
    """Do the tokens from a (to b) spell a `Truth` value: `Truth::True`, `symplex::Truth::Unknown`, or a
    pattern made of them?  Such a reading is already migrated and is not a site."""
    if lx.is_p(a, "|"):
        a += 1
    if lx.is_id(a, "symplex") and lx.is_p(a + 1, "::"):
        a += 2
    return lx.is_id(a, "Truth") and lx.is_p(a + 1, "::") and lx.text[a + 2:a + 3] in (["True"], ["False"], ["Unknown"])


def _truth_path_ends(lx, i):
    """Does a `Truth::X` value end at token i?"""
    return (lx.is_id(i) and lx.text[i] in ("True", "False", "Unknown") and lx.is_p(i - 1, "::")
            and lx.is_id(i - 2, "Truth"))


class _Names(object):
    def __init__(self, cx):
        self.opt = {}
        self.boolq = {}
        every = cx.opt("truth.names") == "B"
        for r in cx.table("truth_queries"):
            if r["fate"] == "not-a-query":
                continue
            if r["today"] == "bool":
                if r["rewrite"] == "yes":
                    self.boolq[r["name"]] = r
                continue
            row = dict(r)
            if every:
                # reproduce review B's count: every `Option<bool>` query is taken to return `Truth`
                row["rewrite"], row["fate"] = "yes", "truth"
            self.opt[r["name"]] = row
        self.all = set(self.opt) | set(self.boolq)


def _names(cx):
    n = getattr(cx, "_truth_names", None)
    if n is None:
        n = cx._truth_names = _Names(cx)
    return n


def _why_not(row):
    fate = row["fate"]
    if fate == "removed":
        return "`%s` is removed at 1.0: the call itself changes (B §3.3)" % row["name"]
    if fate == "moves-to-Poly":
        return "`%s` moves from `Ex` to `Poly` (F PE-MOVE): receiver and result change, by hand" % row["name"]
    if fate == "result-bool":
        return "`%s` returns `Result<bool>` at 1.0, not `Truth` (Part 4, C-17 / B §2.2)" % row["name"]
    if fate == "result-truth":
        return "`%s` returns `Result<Truth>` at 1.0: unwrap the `Result` first" % row["name"]
    if fate.startswith("mixed"):
        return ("`%s` has two fates (%s) and the text does not say which type the receiver has"
                % (row["name"], fate[6:]))
    return "`%s`: fate %s" % (row["name"], fate)


def _marker(view, tok_i, text):
    """An edit that puts a full-line comment above the line of token tok_i."""
    lx = view.lx
    off = lx.start[tok_i]
    ls = lx.line_start(off)
    return (ls, ls, "%s// codemod(truth): %s\n" % (lx.indent_at(off), text))


def classify(view, cx):
    """Every read of a query in the view, as Sites of the truth.* rules."""
    cached = getattr(view, "_truth_sites", None)
    if cached is not None:
        return cached
    lx = view.lx
    names = _names(cx)
    style = cx.opt("truth.assert", "truth-eq")
    sites = []
    out = sites.append
    for i in range(len(lx.kind)):
        if lx.kind[i] != ID or lx.text[i] not in names.all:
            continue
        c = call_at(lx, i)
        if c is None:
            continue
        name = lx.text[i]
        e = c.close_i
        s = chain_start(lx, e)
        if name in names.boolq and name not in names.opt:
            if s >= 0:
                st = _bool_query(view, cx, name, s, e)
                if st is not None:
                    out(st)
            continue
        row = names.opt[name]
        ambiguous = row["ambiguous"] == "yes"
        ok = row["rewrite"] == "yes"
        if s < 0:
            if not ambiguous:
                out(_site("truth.other-reading", view, i, MANUAL,
                          "the receiver is not a plain chain of calls (a block or a cast)", cls="unclassified",
                          name=name))
            continue
        nxt = e + 1
        # ---------------------------------------------------------- postfix readings
        if lx.is_p(nxt, ".") and lx.is_id(nxt + 1):
            m = lx.text[nxt + 1]
            mo = nxt + 2
            mc = lx.mate[mo] if lx.is_open(mo, "(") else -1
            if m == "unwrap_or" and mc == mo + 2 and lx.text[mo + 1] in ("true", "false"):
                lit = lx.text[mo + 1]
                if not ok:
                    out(_site("truth.unwrap-or", view, i, MANUAL, _why_not(row), cls="unwrap_or_" + lit, name=name))
                elif lit == "false":
                    out(_site("truth.unwrap-or", view, i, REWRITE,
                              edits=[(lx.start[nxt + 1], lx.end[mc], "is_true()")], cls="unwrap_or_false", name=name))
                else:
                    out(_unwrap_or_true(view, i, name, s, nxt, mc))
                continue
            if m in ("is_none", "is_some") and mc == mo + 1:
                if not ok:
                    out(_site("truth.is-none", view, i, MANUAL, _why_not(row), cls=m, name=name))
                else:
                    new = "is_unknown" if m == "is_none" else "is_known"
                    out(_site("truth.is-none", view, i, REWRITE,
                              edits=[(lx.start[nxt + 1], lx.end[nxt + 1], new)], cls=m, name=name))
                continue
            if m in ("unwrap", "expect") and mc >= 0:
                out(_site("truth.other-reading", view, i, MANUAL,
                          "`.%s()` on the answer: decide what `Unknown` should do here (B §3.3)" % m,
                          cls="unwrap", name=name))
                continue
            if m in COMBINATORS and mc >= 0:
                out(_site("truth.other-reading", view, i, MANUAL,
                          "`.%s(..)` on the `Option`: rewrite with `known()` or a `match`" % m,
                          cls="combinator", name=name))
                continue
            continue  # some other method on the result: not an `Option<bool>` reading
        if lx.is_p(nxt, "?"):
            out(_site("truth.other-reading", view, i, MANUAL, "`?` on the answer", cls="question_mark", name=name))
            continue
        # ----------------------------------------------------------- `==` / `!=`
        if lx.is_p(nxt, "==") or lx.is_p(nxt, "!="):
            val, last = _value_after(lx, nxt + 1)
            if val and is_boundary_after(lx, last):
                out(_cmp(view, i, name, row, s, e, lx.text[nxt], val, (lx.end[e], lx.end[last]), None))
                continue
            if _is_truth_path(lx, nxt + 1):
                continue  # already compared with a `Truth`
        if lx.is_p(s - 1, "==") or lx.is_p(s - 1, "!="):
            val, first = _value_before(lx, s - 2)
            if val and is_boundary_before(lx, first) and is_boundary_after(lx, e):
                out(_cmp(view, i, name, row, s, e, lx.text[s - 1], val, None, (lx.start[first], lx.start[s])))
                continue
            if _truth_path_ends(lx, s - 2):
                continue
        # ------------------------------------------------------- enclosing macro
        mac = enclosing_macro(lx, s)
        if mac is not None and mac.args is not None:
            idx = whole_arg(mac.args, s, e)
            if idx >= 0 and mac.name in ASSERT_EQ and idx in (0, 1) and len(mac.args) >= 2:
                oa, ob = mac.args[1 - idx]
                val = _value_at(lx, oa, ob)
                ne = ASSERT_EQ[mac.name] == "ne"
                if val is None and _is_truth_path(lx, oa):
                    continue  # already compared with a `Truth`
                if val is None:
                    cls = "assert_ne_other" if ne else "assert_eq_other"
                    if _is_query_chain(lx, names, oa, ob):
                        out(_site("truth.other-reading", view, i, INFO,
                                  "two queries compared with each other: compiles unchanged", cls=cls, name=name))
                    else:
                        out(_site("truth.other-reading", view, i, MANUAL,
                                  "compared with an expression that is not a query: an `Option<bool>` value "
                                  "must become a `Truth`", cls=cls, name=name))
                    continue
                if ne:
                    rule, cls = "truth.assert-ne", "assert_ne_some_" + val.lower() if val != "Unknown" else "assert_ne_none"
                else:
                    rule = {"True": "truth.assert-eq-true", "False": "truth.assert-eq-false",
                            "Unknown": "truth.assert-eq-none"}[val]
                    cls = "assert_eq_none" if val == "Unknown" else "assert_eq_some_" + val.lower()
                if not ok:
                    out(_site(rule, view, i, MANUAL, _why_not(row), cls=cls, name=name))
                    continue
                if style == "is-true":
                    out(_assert_is(view, rule, cls, i, name, mac, idx, s, e, val, ne))
                else:
                    t = view.use_name(oa, TRUTH_PATH)
                    out(_site(rule, view, i, REWRITE,
                              edits=[(lx.start[oa], lx.end[ob], "%s::%s" % (t, val))], cls=cls, name=name))
                continue
            if idx == 0 and mac.name == "matches" and len(mac.args) == 2:
                if not _is_truth_path(lx, mac.args[1][0]):
                    out(_matches(view, i, name, row, mac))
                continue
            if idx >= 0 and mac.name in PRINT_MACROS:
                eol = view.text.find("\n", lx.end[mac.close_i])
                tail = view.text[lx.end[mac.close_i]:eol if eol >= 0 else len(view.text)]
                if not ambiguous or re.search(r"//.*(Some\((true|false)\)|\bNone\b)", tail):
                    out(_site("truth.other-reading", view, i, INFO,
                              "printed: compiles unchanged, the printed text changes (`Some(true)` -> `True`)",
                              cls="printed", name=name))
                continue
        # ---------------------------------------------------- statement context
        if lx.is_id(s - 1, "match") and lx.is_open(nxt, "{"):
            if not _is_truth_path(lx, nxt + 1):
                out(_match(view, i, name, row, nxt))
            continue
        if lx.is_p(s - 1, "="):
            let = _let_before(lx, s - 1)
            if let >= 0:
                pat_some = any(lx.text[k] in ("Some", "None") for k in range(let + 1, s - 1))
                if pat_some:
                    out(_site("truth.other-reading", view, i, MANUAL,
                              "`let` pattern on the `Option`: write a `match` on `Truth` or use `known()`",
                              cls="if_let", name=name))
                    continue
                if lx.is_p(nxt, ";") and not ambiguous:
                    out(_site("truth.other-reading", view, i, MANUAL,
                              "bound to a variable: the compiler finds its readers", cls="bound", name=name))
                    continue
        if lx.is_p(nxt, ";") and is_boundary_before(lx, s) and not lx.is_p(s - 1, "=") \
                and not lx.is_id(s - 1, "return"):
            # an expression statement, as the README writes them: `a.contains(&x);   // Some(false)`
            eol = view.text.find("\n", lx.end[nxt])
            tail = view.text[lx.end[nxt]:eol if eol >= 0 else len(view.text)]
            if not ambiguous or re.search(r"//.*(Some\((true|false)\)|\bNone\b)", tail):
                out(_site("truth.other-reading", view, i, INFO,
                          "the value is not read (a documentation line): compiles unchanged",
                          cls="statement", name=name))
            continue
        if not ambiguous:
            out(_site("truth.other-reading", view, i, MANUAL,
                      "a reading the tool does not know (argument, return value, larger expression)",
                      cls="unclassified", name=name))
    view._truth_sites = sites
    return sites


def _let_before(lx, eq):
    """`eq` is an `=`.  Index of the `let` whose pattern ends just before it, or -1."""
    j = eq - 1
    n = 0
    while j >= 0 and n < 40:
        n += 1
        k, t = lx.kind[j], lx.text[j]
        if k == CLOSE:
            j = lx.mate[j] - 1
            continue
        if k == OPEN or (k == PUNCT and t in (";", "=", "=>", ",")):
            return -1
        if k == ID and t == "let":
            return j
        j -= 1
    return -1


def _is_query_chain(lx, names, a, b):
    """Do tokens a..b end in a call of a query that returns `Truth` at 1.0?"""
    if not lx.is_close(b, ")"):
        return False
    o = lx.mate[b]
    if o < 1 or o - 1 < a or not lx.is_id(o - 1):
        return False
    row = names.opt.get(lx.text[o - 1])
    return row is not None and row["rewrite"] == "yes" and chain_start(lx, b) == a


def _site(rule, view, tok_i, status, reason="", edits=(), cls="", name=""):
    return Site(rule, view, view.lx.start[tok_i], status, reason, edits, cls, name)


def _cmp(view, i, name, row, s, e, op, val, after, before):
    """`q == Some(true)` (after = span from the end of q to the end of the value) or
    `Some(true) == q` (before = span from the start of the value to the start of q)."""
    lx = view.lx
    cls = ("eq_" if op == "==" else "ne_") + ("none" if val == "Unknown" else "some_" + val.lower())
    if row["rewrite"] != "yes":
        return _site("truth.cmp", view, i, MANUAL, _why_not(row), cls=cls, name=name)
    method = {"True": "is_true", "False": "is_false", "Unknown": "is_unknown"}[val]
    edits = []
    if op == "!=":
        if val == "Unknown":
            method = "is_known"
        else:
            edits.append((lx.start[s], lx.start[s], "!"))
            edits.append(_marker(view, s, "`!= Some(%s)` read as \"not %s\" (not as \"%s\"): check the reading"
                                 % (val.lower(), "proved" if val == "True" else "refuted",
                                    "proved false" if val == "True" else "proved true")))
    if after is not None:
        edits.append((after[0], after[1], ".%s()" % method))
    else:
        edits.append((before[0], before[1], ""))
        edits.append((lx.end[e], lx.end[e], ".%s()" % method))
    return _site("truth.cmp", view, i, REWRITE, edits=edits, cls=cls, name=name)


def _unwrap_or_true(view, i, name, s, dot, mc):
    lx = view.lx
    edits = [(lx.start[dot + 1], lx.end[mc], "is_false()")]
    note = "`.unwrap_or(true)` read as \"not refuted\": check the reading"
    if lx.is_p(s - 1, "!") and not lx.is_p(mc + 1, ".") and not lx.is_p(mc + 1, "?"):
        # `!q.unwrap_or(true)` is exactly `q.is_false()`
        edits.append((lx.start[s - 1], lx.start[s], ""))
    elif is_boundary_before(lx, s) and not lx.is_p(mc + 1, ".") and not lx.is_p(mc + 1, "?"):
        edits.append((lx.start[s], lx.start[s], "!"))
        edits.append(_marker(view, s, note))
    else:
        edits.append((lx.start[s], lx.start[s], "(!"))
        edits.append((lx.end[mc], lx.end[mc], ")"))
        edits.append(_marker(view, s, note))
    return _site("truth.unwrap-or", view, i, REWRITE, edits=edits, cls="unwrap_or_true", name=name)


def _assert_is(view, rule, cls, i, name, mac, idx, s, e, val, ne):
    """The alternative style: assert_eq!(q, Some(true), ..) -> assert!(q.is_true(), ..)."""
    lx = view.lx
    method = {"True": "is_true", "False": "is_false", "Unknown": "is_unknown"}[val]
    new_macro = lx.text[mac.name_i].replace("_eq", "").replace("_ne", "")  # assert / debug_assert / prop_assert
    edits = [(lx.start[mac.name_i], lx.end[mac.name_i], new_macro)]
    oa, ob = mac.args[1 - idx]
    if idx == 0:
        edits.append((lx.end[e], lx.end[ob], ".%s()" % method))
    else:
        edits.append((lx.start[oa], lx.start[s], ""))
        edits.append((lx.end[e], lx.end[e], ".%s()" % method))
    if ne:
        edits.append((lx.start[s], lx.start[s], "!"))
    return _site(rule, view, i, REWRITE, edits=edits, cls=cls, name=name)


def _pattern_edits(view, a, b, anchor):
    """Edits that turn the pattern in tokens a..b (alternatives of Some(true) / Some(false) /
    None / Some(_) / _) into a pattern on `Truth`; None when an alternative is anything else."""
    lx = view.lx
    edits = []
    j = a
    if lx.is_p(j, "|"):
        j += 1
    t = None
    while j <= b:
        val, last = _value_after(lx, j)
        if val is None:
            if lx.is_id(j, "_"):
                last = j
            elif lx.is_id(j, "Some") and lx.is_open(j + 1, "(") and lx.mate[j + 1] == j + 3 and lx.is_id(j + 2, "_"):
                if t is None:
                    t = view.use_name(anchor, TRUTH_PATH)
                edits.append((lx.start[j], lx.end[j + 3], "%s::True | %s::False" % (t, t)))
                last = j + 3
            else:
                return None
        else:
            if t is None:
                t = view.use_name(anchor, TRUTH_PATH)
            edits.append((lx.start[j], lx.end[last], "%s::%s" % (t, val)))
        j = last + 1
        if j <= b:
            if not lx.is_p(j, "|"):
                return None
            j += 1
    return edits


def _matches(view, i, name, row, mac):
    lx = view.lx
    a, b = mac.args[1]
    if row["rewrite"] != "yes":
        return _site("truth.matches", view, i, MANUAL, _why_not(row), cls="matches", name=name)
    edits = _pattern_edits(view, a, b, a)
    if not edits:
        view.take_pending_imports()
        return _site("truth.matches", view, i, MANUAL,
                     "`matches!` with a pattern that binds or has a guard: `%s`" % lx.span(a, b)[:60],
                     cls="matches", name=name)
    return _site("truth.matches", view, i, REWRITE, edits=edits, cls="matches", name=name)


def match_arms(lx, brace):
    """The arms of the `match` block that opens at `brace`: [(pattern_first, pattern_last)],
    the pattern without its guard; None when the block is not understood."""
    close = lx.mate[brace]
    arms = []
    j = brace + 1
    while j < close:
        a = j
        arrow = -1
        guard = -1
        while j < close:
            if lx.kind[j] == OPEN:
                j = lx.mate[j]
            elif lx.is_p(j, "=>"):
                arrow = j
                break
            elif lx.is_id(j, "if") and guard < 0:
                guard = j
            j += 1
        if arrow < 0:
            return None
        arms.append((a, (guard if guard >= 0 else arrow) - 1, guard >= 0))
        j = arrow + 1
        if lx.is_open(j, "{"):
            j = lx.mate[j] + 1
            if lx.is_p(j, ","):
                j += 1
            continue
        while j < close and not lx.is_p(j, ","):
            if lx.kind[j] == OPEN:
                j = lx.mate[j]
            j += 1
        j += 1
    return arms


def _match(view, i, name, row, brace):
    lx = view.lx
    if row["rewrite"] != "yes":
        return _site("truth.match-arms", view, i, MANUAL, _why_not(row), cls="match", name=name)
    arms = match_arms(lx, brace)
    if not arms:
        return _site("truth.match-arms", view, i, MANUAL, "the arms of this `match` were not understood",
                     cls="match", name=name)
    edits = []
    for (a, b, _guard) in arms:
        if a > b:
            return _site("truth.match-arms", view, i, MANUAL, "an arm without a pattern", cls="match", name=name)
        pe = _pattern_edits(view, a, b, a)
        if pe is None:
            view.take_pending_imports()
            return _site("truth.match-arms", view, i, MANUAL,
                         "an arm binds the value or is not one of Some(true) / Some(false) / None / _: `%s`"
                         % lx.span(a, b)[:50], cls="match", name=name)
        edits.extend(pe)
    if not edits:
        return _site("truth.match-arms", view, i, MANUAL, "only wildcard arms", cls="match", name=name)
    return _site("truth.match-arms", view, i, REWRITE, edits=edits, cls="match", name=name)


def _bool_query(view, cx, name, s, e):
    """A query that returns a plain `bool` today and `Truth` at 1.0 (B2:DEFECT:bool):
    `assert!(q)` -> `assert!(q.is_true())`."""
    lx = view.lx
    i = s
    nxt = e + 1
    append = [(lx.end[e], lx.end[e], ".is_true()")]
    if lx.is_p(nxt, "."):
        if lx.is_id(nxt + 1) and lx.text[nxt + 1] in TRUTH_METHODS:
            return None  # already reads a `Truth`
        return _site("truth.bool-query", view, i, MANUAL, "a method is called on the `bool`", cls="method", name=name)
    if lx.is_p(nxt, "?"):
        return None
    mac = enclosing_macro(lx, s)
    if mac is not None and mac.args is not None:
        idx = whole_arg(mac.args, s, e)
        if mac.name in ("assert", "debug_assert") and idx == 0:
            return _site("truth.bool-query", view, i, REWRITE, edits=append, cls="assert", name=name)
        if mac.name in ("assert", "debug_assert") and mac.args and mac.args[0] == (s - 1, e) and lx.is_p(s - 1, "!"):
            return _site("truth.bool-query", view, i, MANUAL,
                         "negated assertion: \"not proved\" (`!q.is_true()`) or \"proved false\" (`q.is_false()`)? "
                         "(B §3.3)", cls="assert_not", name=name)
        if mac.name in ASSERT_EQ and idx in (0, 1) and len(mac.args) >= 2:
            oa, ob = mac.args[1 - idx]
            if oa == ob and lx.text[oa] in ("true", "false"):
                return _site("truth.bool-query", view, i, REWRITE, edits=append, cls="assert_eq_bool", name=name)
            if lx.kind[oa] == ID and lx.text[oa] in ("Truth", "symplex"):
                return None  # already compared with a `Truth`
            return _site("truth.bool-query", view, i, MANUAL, "compared with another expression",
                         cls="assert_eq_other", name=name)
        if mac.name in PRINT_MACROS and idx >= 0:
            return _site("truth.bool-query", view, i, INFO,
                         "printed: compiles unchanged, the printed text changes (`true` -> `True`)",
                         cls="printed", name=name)
    before_ok = lx.is_id(s - 1, "if") or lx.is_id(s - 1, "while") or lx.is_p(s - 1, "&&") or lx.is_p(s - 1, "||")
    after_ok = lx.is_open(nxt, "{") or lx.is_p(nxt, "&&") or lx.is_p(nxt, "||")
    if (before_ok and (after_ok or is_boundary_after(lx, e))) or (after_ok and is_boundary_before(lx, s)):
        return _site("truth.bool-query", view, i, REWRITE, edits=append, cls="condition", name=name)
    if lx.is_p(s - 1, "!"):
        return _site("truth.bool-query", view, i, MANUAL,
                     "negated: \"not proved\" (`!q.is_true()`) or \"proved false\" (`q.is_false()`)?",
                     cls="negated", name=name)
    if lx.is_p(nxt, "==") or lx.is_p(nxt, "!=") or lx.is_p(s - 1, "==") or lx.is_p(s - 1, "!="):
        return _site("truth.bool-query", view, i, MANUAL, "compared with `==`", cls="cmp", name=name)
    return _site("truth.bool-query", view, i, MANUAL,
                 "the `bool` is bound, returned or passed on: the compiler finds its readers", cls="bound", name=name)


class _Reading(Rule):
    trigger = _TRIGGER

    def find(self, view, cx):
        return [s for s in classify(view, cx) if s.rule == self.id]


def _reading(rule_id, text):
    return type("R_" + rule_id.replace(".", "_").replace("-", "_"), (_Reading,), {"id": rule_id, "description": text})()


class Rename(Rule):
    id = "truth.rename"
    description = ("B's renamed queries: `.is_constant()` -> `.is_symbol_free()`, `.satisfiable()` -> "
                   "`.is_satisfiable()`, `.is_satisfied()` -> `.is_identity()`")
    trigger = re.compile(r"\b(?:%s)\b" % "|".join(RENAMES))

    def find(self, view, cx):
        lx = view.lx
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or lx.text[i] not in RENAMES:
                continue
            c = call_at(lx, i)
            if c is None or c.style == "free":
                continue
            new, clash = RENAMES[lx.text[i]]
            near = view.unit.text if view.kind == "code" else view.text
            if clash and re.search(r"\b%s\b" % clash, near):
                yield self.manual(view, lx.start[i],
                                  "this file names `%s`, whose own `%s` keeps its name" % (clash, lx.text[i]),
                                  name=lx.text[i])
                continue
            yield self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], new)], name=lx.text[i])


class PrintedComment(Rule):
    id = "truth.printed-comment"
    description = ("the expected-output comment after a printed query: `// Some(true)` -> `// True` "
                   "(only when the line has as many such values as queries)")
    trigger = re.compile(r"//.*(?:Some\((?:true|false)\)|\bNone\b)")
    _val = re.compile(r"Some\((true|false)\)|\bNone\b")

    def find(self, view, cx):
        lx = view.lx
        names = _names(cx)
        text = view.text
        # trailing line comments of the view, by line start
        if view.kind == "code":
            comments = [(c.start, c.end) for c in lx.comments if text.startswith("//", c.start) and not c.doc]
        else:
            comments = [(c.start, c.end) for c in lx.comments if text.startswith("//", c.start)]
        for (cs, ce) in comments:
            body = text[cs:ce]
            vals = list(self._val.finditer(body))
            if not vals:
                continue
            ls = lx.line_start(cs)
            if text[ls:cs].strip() == "":
                continue  # a comment on a line of its own: prose
            a = lx.tok_index_at(ls)
            b = lx.tok_index_at(cs)
            queries = [k for k in range(a, b)
                       if lx.kind[k] == ID and lx.text[k] in names.opt and call_at(lx, k) is not None]
            # a query whose result is read by a method (`.unwrap()`, `.is_true()`) prints something else
            queries = [k for k in queries if not _read_by_method(lx, k)]
            if not queries:
                continue
            bad = [k for k in queries if names.opt[lx.text[k]]["rewrite"] != "yes"]
            if bad:
                yield self.manual(view, cs, _why_not(names.opt[lx.text[bad[0]]]), name=lx.text[bad[0]])
                continue
            if len(vals) != len(queries):
                yield self.manual(view, cs, "the comment has %d value(s) for %d quer%s on the line: which is which?"
                                  % (len(vals), len(queries), "y" if len(queries) == 1 else "ies"),
                                  name=lx.text[queries[0]])
                continue
            edits = []
            for m in vals:
                new = {"true": "True", "false": "False", None: "Unknown"}[m.group(1)]
                edits.append((cs + m.start(), cs + m.end(), new))
            yield self.rewrite(view, cs, edits, name=lx.text[queries[0]])


def _read_by_method(lx, k):
    c = call_at(lx, k)
    return c is not None and lx.is_p(c.close_i + 1, ".")


class Mention(Rule):
    id = "truth.mention"
    description = ("a query named in a comment or in prose (B's class \"mention\"): text for a person; never "
                   "rewritten")
    targets = ("text",)
    trigger = _TRIGGER

    def find_text(self, tv, cx):
        names = _names(cx)
        unit = tv.unit
        rx = getattr(cx, "_truth_mention_rx", None)
        if rx is None:
            rx = cx._truth_mention_rx = re.compile(
                r"(?<![A-Za-z0-9_])(%s)\(" % "|".join(sorted(names.opt, key=len, reverse=True)))
        val = re.compile(r"Some\((?:true|false)\)|[=!]=\s*None\b")
        if unit.is_rust:
            if unit.corpus == "src":
                spans = [(c.start, c.end) for c in unit.lx.comments if c.doc]
            else:
                spans = [(c.start, c.end) for c in unit.lx.comments]
            fenced = _fenced_spans(unit, "doc")
        else:
            spans = [(0, len(unit.text))]
            fenced = _fenced_spans(unit, "md")
        for (a, b) in spans:
            for m in rx.finditer(unit.text, a, b):
                if any(x <= m.start() < y for (x, y) in fenced):
                    continue  # inside a Rust code block: the token rules see it
                ls = unit.text.rfind("\n", 0, m.start()) + 1
                le = unit.text.find("\n", m.start())
                line = unit.text[ls:le if le >= 0 else len(unit.text)]
                # a name that other types share (`contains`, `is_empty`) is a mention only where the
                # line shows an `Option<bool>` value beside it; the others always are
                if names.opt[m.group(1)]["ambiguous"] == "yes" and not val.search(line):
                    continue
                yield Site(self.id, tv, m.start(), INFO, "comment or prose: reword by hand", (), "mention", m.group(1))


def _fenced_spans(unit, kind):
    out = []
    for v in unit.views([kind]):
        if v.segs is None:
            out.append((v.base, v.base + len(v.text)))
        else:
            out.append((v.segs[0][1], v.segs[-1][1] + v.segs[-1][2]))
    return out


RULES = [
    _reading("truth.assert-eq-true", "B shape 1: `assert_eq!(q, Some(true))` -> `assert_eq!(q, Truth::True)` (either "
                                     "operand order)"),
    _reading("truth.assert-eq-false", "B shape 2: `assert_eq!(q, Some(false))` -> `assert_eq!(q, Truth::False)`"),
    _reading("truth.assert-eq-none", "B shape 3: `assert_eq!(q, None)` -> `assert_eq!(q, Truth::Unknown)`"),
    _reading("truth.assert-ne", "B shape 4: `assert_ne!(q, Some(..) | None)` -> `assert_ne!(q, Truth::..)`"),
    _reading("truth.cmp", "B shape 5: `q == Some(true)` -> `q.is_true()`, `== Some(false)` -> `.is_false()`, "
                          "`== None` -> `.is_unknown()`, `!= Some(true)` -> `!q.is_true()` with a review marker"),
    _reading("truth.unwrap-or", "B shape 6: `q.unwrap_or(false)` -> `q.is_true()`; `q.unwrap_or(true)` -> "
                                "`!q.is_false()` with a review marker"),
    _reading("truth.is-none", "B shape 7: `q.is_none()` -> `q.is_unknown()`, `q.is_some()` -> `q.is_known()`"),
    _reading("truth.match-arms", "`match q { Some(true) => .., None => .. }`: the arms become `Truth::True => ..` "
                                 "(only when no arm binds the value)"),
    _reading("truth.matches", "`matches!(q, Some(true) | None)` -> `matches!(q, Truth::True | Truth::Unknown)`"),
    _reading("truth.bool-query", "queries that return `bool` today and `Truth` at 1.0 (`check_ode_solution`, "
                                 "`is_controllable`, `is_observable`): `assert!(q)` -> `assert!(q.is_true())`"),
    _reading("truth.other-reading", "other readings of a query (`if let`, `.unwrap()`, bound to a variable, "
                                    "compared with an expression, printed): found, never rewritten"),
    Rename(),
    PrintedComment(),
    Mention(),
]

RULESET = RuleSet(
    "truth",
    "reads of yes/no queries: Option<bool> -> Truth (review B, M1)",
    RULES,
    needs=("tables/truth_queries.tsv — the query method names with their merged fate (B's optbool.tsv and "
           "fates_B.tsv joined with api_fate.tsv).  Regenerate it when the listing is regenerated from 0.60 "
           "(WP-00) and whenever T2 changes which queries return `Truth`.  `Truth` must be exported at "
           "`symplex::Truth` and in the prelude (package T1) before the output compiles."),
    package="T2",
)
