# Rule set `errors` (packages E1, E2a, R1): review C, REPORT §3.2 — five rewrites.
#
#   1  errors.matches      matches!(r, Err(SymplexError::V { .. }))  -> matches!(r, Err(e) if e.kind() == ErrorKind::V')
#                          matches!(e, SymplexError::V { .. })       -> e.kind() == ErrorKind::V'
#      errors.match-arm    Err(SymplexError::V { .. }) => ..          -> Err(e) if e.kind() == ErrorKind::V' => ..
#                          if let Err(SymplexError::V { .. }) = r {   -> if matches!(r, Err(e) if e.kind() == ..) {
#   2  errors.try-rename   x::try_name(..) / .try_name(..)            -> x::name(..) / .name(..)   (classes b, cV, cP)
#   3  errors.plain-unwrap name(..) that now returns Result           -> name(..).unwrap()  or  name(..)?
#   4  errors.doc-link     [`SymplexError::V`] in doc comments        -> [`ErrorKind::V'`]
#   5  errors.construct    SymplexError::V { operation: a, reason: b } -> SymplexError::v(a, b)   (library code)
#
# Only patterns that TEST a variant are rewritten; a pattern that binds a field
# (`{ reason, .. }`, `(msg)`) is listed: its body reads something that moves to an
# accessor or to `to_string()`, and a person decides which (C §3.3: 145 sites).
#
# Rules 2 and 3 undo each other's evidence: after `try_fibonacci(n)` has become
# `fibonacci(n)`, nothing in the text says that this call was never the old, plain
# `fibonacci`.  So that a second run of the tool cannot append `.unwrap()` to it, the
# tool renames a `try_` call only where the `Result` is read right at the call (or the
# bound variable is read as a `Result` in the same block), and rule 3 leaves every call
# that is read as a `Result` alone.  A `try_` call without such a reading is listed.
import re

from .lexer import ID, PUNCT, OPEN, CLOSE, is_boundary_before, is_boundary_after
from .chain import (call_at, chain_start, chain_end, path_before, path_start, enclosing_macro, macro_at,
                    whole_arg, enclosing_fn, fn_returns_result)
from .engine import Rule, RuleSet, Site, REWRITE, MANUAL, INFO
from .rules_truth import match_arms
from .rules_renames import _defines_fn
from .views import use_items, use_item_edit

KIND_PATH = "symplex::errors::ErrorKind"
RESULT_READERS = frozenset(
    "unwrap expect unwrap_err expect_err is_ok is_err ok err map_err unwrap_or unwrap_or_else unwrap_or_default "
    "map and_then or_else is_ok_and is_err_and as_ref".split()
)
FRESH = ("e", "err", "error", "symplex_err")


def _kinds(cx):
    k = getattr(cx, "_err_kinds", None)
    if k is None:
        k = cx._err_kinds = {r["variant"]: r for r in cx.table("error_kinds")}
    return k


class _Occ(object):
    """One `SymplexError::Variant [payload]` in the text."""

    __slots__ = ("first", "var", "last", "variant", "payload", "binds")

    def __init__(self, lx, v, kinds):
        self.var = v
        self.variant = lx.text[v]
        self.first = path_start(lx, v)
        self.last = v
        self.payload = ""
        self.binds = False
        n = v + 1
        if lx.is_open(n, "{") or lx.is_open(n, "("):
            c = lx.mate[n]
            self.last = c
            self.payload = lx.text[n]
            inner = lx.text[n + 1:c]
            if lx.text[n] == "{":
                self.binds = inner not in ([".."], [])
            else:
                self.binds = inner not in (["_"], [".."])


def _occurrences(lx, kinds):
    for v in range(2, len(lx.kind)):
        if lx.kind[v] == ID and lx.text[v] in kinds and lx.is_p(v - 1, "::") and lx.is_id(v - 2, "SymplexError"):
            yield _Occ(lx, v, kinds)


def _alternatives(lx, a, b, kinds):
    """Tokens a..b as `SymplexError::V {..} | SymplexError::W(_)`: the occurrences, or None."""
    out = []
    j = a
    if lx.is_p(j, "|"):
        j += 1
    while j <= b:
        # a path that ends in SymplexError::V
        k = j
        while lx.is_id(k) and lx.is_p(k + 1, "::"):
            k += 2
        if not (lx.is_id(k) and lx.text[k] in kinds and lx.is_id(k - 2, "SymplexError") and k - 2 >= j):
            return None
        o = _Occ(lx, k, kinds)
        if o.first != j:
            return None
        out.append(o)
        j = o.last + 1
        if j <= b:
            if not lx.is_p(j, "|"):
                return None
            j += 1
    return out or None


def _guard(view, anchor, occs, kinds, name):
    kn = view.use_name(anchor, KIND_PATH)
    ks = []
    for o in occs:
        k = kinds[o.variant]["kind"]
        if k not in ks:
            ks.append(k)
    if len(ks) == 1:
        return "%s.kind() == %s::%s" % (name, kn, ks[0])
    return "matches!(%s.kind(), %s)" % (name, " | ".join("%s::%s" % (kn, k) for k in ks))


def _marks(view, tok_i, occs, kinds):
    old = sorted(set(o.variant for o in occs if kinds[o.variant]["mark"] == "yes"))
    if not old:
        return []
    lx = view.lx
    off = lx.start[tok_i]
    ls = lx.line_start(off)
    return [(ls, ls, "%s// codemod(errors): was %s -> `Undetermined` by default; the raise site may get another "
                     "kind (E2b-E2d)\n" % (lx.indent_at(off), ", ".join("`%s`" % v for v in old)))]


def _fresh(lx, a, b):
    """A binding name that nothing in tokens a..b uses: no identifier, and no string literal either,
    because a format string captures variables by name (`println!("{e}")`)."""
    used = set(lx.text[k] for k in range(a, b + 1) if lx.kind[k] == ID)
    strings = " ".join(lx.text[k] for k in range(a, b + 1) if lx.kind[k] == "str")
    for n in FRESH:
        if n not in used and not re.search(r"\b%s\b" % n, strings):
            return n
    return None


def _arm_extent(lx, arrow):
    """Last token of the arm whose `=>` is at `arrow`."""
    j = arrow + 1
    g = lx.up[arrow]
    close = lx.mate[g] if g >= 0 else len(lx.kind)
    if lx.is_open(j, "{"):
        return lx.mate[j]
    while j < close and not lx.is_p(j, ","):
        if lx.kind[j] == OPEN:
            j = lx.mate[j]
        j += 1
    return j - 1


def classify(view, cx):
    cached = getattr(view, "_err_sites", None)
    if cached is not None:
        return cached
    lx = view.lx
    kinds = _kinds(cx)
    sites = []
    done = set()
    library = view.unit.corpus == "src" and view.kind == "code"
    for o in _occurrences(lx, kinds):
        if o.var in done:
            continue
        pos = lx.start[o.first]
        name = o.variant
        g = lx.up[o.first]
        wrapped = g >= 0 and lx.is_open(g, "(") and lx.is_id(g - 1, "Err") and not lx.is_p(g - 2, ".")
        if wrapped:
            close = lx.mate[g]
            occs = _alternatives(lx, g + 1, close - 1, kinds)
            if occs is None:
                sites.append(Site("errors.other", view, pos, MANUAL,
                                  "`Err(..)` holds more than variants of `SymplexError`", (), "pattern", name))
                continue
            for x in occs:
                done.add(x.var)
            ps, pe = g - 1, close
            # `Err(A) | Err(B)`: take the whole alternation
            ok = True
            while lx.is_p(pe + 1, "|") and lx.is_id(pe + 2, "Err") and lx.is_open(pe + 3, "("):
                more = _alternatives(lx, pe + 4, lx.mate[pe + 3] - 1, kinds)
                if more is None:
                    ok = False
                    break
                for x in more:
                    done.add(x.var)
                occs = occs + more
                pe = lx.mate[pe + 3]
            if not ok or lx.is_p(ps - 1, "|") or lx.is_p(pe + 1, "|"):
                sites.append(Site("errors.other", view, pos, MANUAL,
                                  "an alternation that mixes `SymplexError` variants with other patterns", (),
                                  "pattern", name))
                continue
            binders = [x for x in occs if x.binds]
            if binders:
                sites.append(Site("errors.other", view, pos, MANUAL,
                                  "the pattern binds or tests a field (`%s`): read it with an accessor "
                                  "(`degeneracy()`, `symbol()`) or `to_string()` (C §3.3)"
                                  % lx.span(binders[0].var, binders[0].last)[:48], (), "pattern-binds", name))
                continue
            nxt = pe + 1
            mac = macro_at(lx, lx.up[ps]) if lx.up[ps] >= 0 else None
            if mac is not None and mac.name in ("matches", "assert_matches") and mac.args is not None \
                    and whole_arg(mac.args, ps, pe) == 1 and len(mac.args) == 2:
                guard = _guard(view, o.first, occs, kinds, "e")
                edits = [(lx.start[ps], lx.end[pe], "Err(e) if %s" % guard)] + _marks(view, mac.name_i, occs, kinds)
                sites.append(Site("errors.matches", view, pos, REWRITE, "", edits, "matches-err", name))
                continue
            if mac is not None and mac.name in ("matches", "assert_matches"):
                sites.append(Site("errors.matches", view, pos, MANUAL,
                                  "`matches!` with a guard or more than two arguments: join the conditions by hand",
                                  (), "matches-guard", name))
                continue
            if lx.is_p(nxt, "=>"):
                last = _arm_extent(lx, nxt)
                bind = _fresh(lx, nxt, last)
                if bind is None:
                    sites.append(Site("errors.match-arm", view, pos, MANUAL, "no free name for the binding", (),
                                      "arm", name))
                    continue
                guard = _guard(view, o.first, occs, kinds, bind)
                edits = [(lx.start[ps], lx.end[pe], "Err(%s) if %s" % (bind, guard))] + _marks(view, ps, occs, kinds)
                sites.append(Site("errors.match-arm", view, pos, REWRITE, "", edits, "arm", name))
                continue
            if lx.is_id(nxt, "if"):
                sites.append(Site("errors.match-arm", view, pos, MANUAL,
                                  "the arm has a guard already: join the two conditions by hand", (), "arm-guard",
                                  name))
                continue
            if lx.is_p(nxt, "=") and lx.is_id(ps - 1, "let") and lx.is_id(ps - 2, "if"):
                site = _if_let(view, kinds, o, occs, ps, pe, name)
                sites.append(site)
                continue
            sites.append(Site("errors.other", view, pos, MANUAL,
                              "a pattern in a place the tool does not rewrite (`let .. else`, `while let`, a "
                              "function argument)", (), "pattern", name))
            continue
        # ---------------------------------------------------------- not inside Err(..)
        mac = macro_at(lx, g) if g >= 0 else None
        if mac is not None and mac.name in ("matches", "assert_matches") and mac.args is not None \
                and len(mac.args) == 2:
            a1, b1 = mac.args[1]
            occs = _alternatives(lx, a1, b1, kinds)
            if occs is not None and a1 <= o.first <= b1:
                for x in occs:
                    done.add(x.var)
                if any(x.binds for x in occs):
                    sites.append(Site("errors.other", view, pos, MANUAL,
                                      "the pattern binds or tests a field: read it with an accessor or "
                                      "`to_string()` (C §3.3)", (), "pattern-binds", name))
                    continue
                sites.append(_bare_matches(view, kinds, o, occs, mac, name))
                continue
        if _looks_like_pattern(lx, o):
            sites.append(Site("errors.other", view, pos, MANUAL,
                              "a variant pattern outside `Err(..)` (a `match` on the error itself): write "
                              "`e if e.kind() == ErrorKind::..` after checking that the value may be bound",
                              (), "pattern-bare", name))
            continue
        # a construction
        row = kinds[o.variant]
        if library:
            sites.append(_construct(view, o, row))
        else:
            sites.append(Site("errors.other", view, pos, MANUAL,
                              "constructs a variant: at 1.0 use a constructor (`SymplexError::new(kind, operation, "
                              "reason)`, `invalid_argument`, ..)", (), "construction", name))
    view._err_sites = sites
    return sites


def _looks_like_pattern(lx, o):
    """Is the occurrence a pattern (rather than an expression that builds the variant)?"""
    inner = lx.text[o.var + 2:o.last] if o.payload else []
    if ".." in inner or inner == ["_"]:
        return True
    if lx.is_p(o.last + 1, "=>") or lx.is_p(o.last + 1, "|") or lx.is_p(o.first - 1, "|"):
        return True
    if lx.is_p(o.last + 1, "=") and not lx.is_p(o.last + 1, "=="):
        return True  # `let SymplexError::V { a, b } = e`
    g = lx.up[o.first]
    if lx.is_id(o.last + 1, "if") and g >= 0 and lx.is_open(g, "{") and _is_match_block(lx, g):
        return True
    return False


def _is_match_block(lx, brace):
    j = brace - 1
    n = 0
    while j >= 0 and n < 60:
        n += 1
        if lx.is_id(j, "match"):
            return True
        if lx.is_p(j, ";") or lx.is_open(j, "{") or lx.is_close(j, "}"):
            return False
        if lx.kind[j] == CLOSE:
            j = lx.mate[j]
        j -= 1
    return False


def _if_let(view, kinds, o, occs, ps, pe, name):
    """`if let Err(SymplexError::V { .. }) = EXPR {` -> `if matches!(EXPR, Err(e) if ..) {`"""
    lx = view.lx
    pos = lx.start[o.first]
    eq = pe + 1
    j = eq + 1
    g = lx.up[eq]
    close = lx.mate[g] if g >= 0 else len(lx.kind)
    while j < close and not lx.is_open(j, "{"):
        if lx.is_p(j, "&&") or lx.is_p(j, "||") or lx.is_p(j, ";"):
            return Site("errors.match-arm", view, pos, MANUAL, "`if let` in a chain of conditions", (), "if-let",
                        name)
        if lx.kind[j] == OPEN:
            j = lx.mate[j]
        j += 1
    if j >= close or j == eq + 1:
        return Site("errors.match-arm", view, pos, MANUAL, "the `if let` was not understood", (), "if-let", name)
    guard = _guard(view, o.first, occs, kinds, "e")
    edits = [(lx.start[ps - 1], lx.start[eq + 1], "matches!("),
             (lx.end[j - 1], lx.end[j - 1], ", Err(e) if %s)" % guard)] + _marks(view, ps - 2, occs, kinds)
    return Site("errors.match-arm", view, pos, REWRITE, "", edits, "if-let", name)


def _bare_matches(view, kinds, o, occs, mac, name):
    """`matches!(e, SymplexError::V { .. })` -> `e.kind() == ErrorKind::V'`"""
    lx = view.lx
    pos = lx.start[o.first]
    a0, b0 = mac.args[0]
    inner = a0 + 1 if (lx.is_p(a0, "&") or lx.is_p(a0, "*")) else a0
    if inner > b0 or chain_end(lx, inner) != b0:
        return Site("errors.matches", view, pos, MANUAL,
                    "the value that is matched is not a plain chain (`%s`)" % lx.span(a0, b0)[:40], (),
                    "matches-bare", name)
    recv = lx.span(inner, b0)
    guard = _guard(view, o.first, occs, kinds, recv)
    start_i, end_i = mac.name_i, mac.close_i
    single = guard.startswith(recv + ".kind() ==")
    edits = []
    if lx.is_p(start_i - 1, "!") and single:
        guard = guard.replace(".kind() ==", ".kind() !=", 1)
        edits.append((lx.start[start_i - 1], lx.start[start_i], ""))
    elif single and not (is_boundary_before(lx, start_i) and is_boundary_after(lx, end_i)):
        guard = "(" + guard + ")"
    edits.append((lx.start[start_i], lx.end[end_i], guard))
    edits += _marks(view, start_i, occs, kinds)
    return Site("errors.matches", view, pos, REWRITE, "", edits, "matches-bare", name)


def _construct(view, o, row):
    """Library code: `SymplexError::V { operation: a, reason: b }` -> `SymplexError::v(a, b)`"""
    lx = view.lx
    pos = lx.start[o.first]
    name = o.variant
    if not row["ctor"]:
        return Site("errors.construct", view, pos, MANUAL,
                    "no constructor is named for `%s` in tables/error_kinds.tsv (package E1 fills it in)" % name,
                    (), "construction", name)
    if o.payload != "{":
        return Site("errors.construct", view, pos, MANUAL, "not a struct literal", (), "construction", name)
    want = row["ctor_fields"].split(",")
    args = lx.split_args(o.var + 1)
    if args is None:
        return Site("errors.construct", view, pos, MANUAL, "the fields were not understood", (), "construction", name)
    got = {}
    for (a, b) in args:
        if lx.is_p(a, ".."):
            return Site("errors.construct", view, pos, MANUAL, "struct update syntax (`..`)", (), "construction", name)
        if not lx.is_id(a):
            return Site("errors.construct", view, pos, MANUAL, "the fields were not understood", (), "construction",
                        name)
        if a == b:
            got[lx.text[a]] = lx.text[a]
        elif lx.is_p(a + 1, ":"):
            got[lx.text[a]] = lx.span(a + 2, b)
        else:
            return Site("errors.construct", view, pos, MANUAL, "the fields were not understood", (), "construction",
                        name)
    if sorted(got) != sorted(want):
        return Site("errors.construct", view, pos, MANUAL,
                    "fields %s, expected %s" % (sorted(got), sorted(want)), (), "construction", name)
    text = "%s(%s)" % (row["ctor"], ", ".join(got[f] for f in want))
    return Site("errors.construct", view, pos, REWRITE, "", [(lx.start[o.var], lx.end[o.last], text)],
                "construction", name)


class _Pattern(Rule):
    trigger = re.compile(r"\bSymplexError\s*::\s*[A-Z]")

    def find(self, view, cx):
        return [s for s in classify(view, cx) if s.rule == self.id]


def _pattern(rule_id, text, library=False):
    return type("E_" + rule_id.replace(".", "_").replace("-", "_"), (_Pattern,),
                {"id": rule_id, "description": text, "library_code": library})()


class DocLink(Rule):
    id = "errors.doc-link"
    description = "C item 4: the intra-doc link [`SymplexError::V`] in a doc comment -> [`ErrorKind::V'`]"
    targets = ("text",)
    trigger = re.compile(r"\[`(?:\w+::)*SymplexError::[A-Z]\w+`\]")
    library_code = True

    def find_text(self, tv, cx):
        kinds = _kinds(cx)
        unit = tv.unit
        if not unit.is_rust:
            return
        rx = re.compile(r"\[`((?:\w+::)*SymplexError)::([A-Z]\w+)`\]")
        for c in unit.lx.comments:
            if not c.doc:
                continue
            for m in rx.finditer(unit.text, c.start, c.end):
                v = m.group(2)
                if v not in kinds:
                    continue
                # `[`SymplexError::V`](path)` carries its own target: leave the target alone
                new = "[`ErrorKind::%s`]" % kinds[v]["kind"]
                yield Site(self.id, tv, m.start(), REWRITE, "", [(m.start(), m.end(), new)], "doc-link", v)


# ------------------------------------------------------------------ the pairs
def reads_result(view, s, e, cx):
    """Is the value of the chain s..e read as a `Result`, visibly?"""
    lx = view.lx
    n = e + 1
    if lx.is_p(n, "?"):
        return True
    if lx.is_p(n, ".") and lx.is_id(n + 1) and lx.text[n + 1] in RESULT_READERS:
        return True
    if lx.is_id(s - 1, "match") and lx.is_open(n, "{"):
        return lx.text[n + 1:n + 2] in (["Ok"], ["Err"])
    mac = enclosing_macro(lx, s)
    if mac is not None and mac.args is not None and len(mac.args) >= 2 and whole_arg(mac.args, s, e) == 0 \
            and mac.name in ("matches", "assert_matches"):
        return lx.text[mac.args[1][0]] in ("Ok", "Err")
    if lx.is_p(s - 1, "="):
        j = s - 2
        k = 0
        while j >= 0 and k < 30 and not lx.is_p(j, ";") and not lx.is_open(j, "{"):
            if lx.is_id(j, "let"):
                pat = lx.text[j + 1:s - 1]
                if "Ok" in pat or "Err" in pat or "Result" in pat:
                    return True
                if len(pat) >= 1 and lx.is_id(j + 1) and lx.is_p(n, ";"):
                    var = pat[1] if pat[0] == "mut" and len(pat) > 1 else pat[0]
                    return _var_read_as_result(lx, n, var)
                return False
            j -= 1
            k += 1
        return False
    # the tail expression, or a `return`, of a function that returns a Result
    tail = lx.is_close(n, "}") and (is_boundary_before(lx, s) or lx.is_id(s - 1, "return"))
    if tail or (lx.is_id(s - 1, "return") and lx.is_p(n, ";")):
        f = enclosing_fn(lx, s)
        if f is not None and not f[2] and fn_returns_result(lx, f[0], f[1]):
            return lx.is_id(s - 1, "return") or lx.up[s] == f[1]
    return False


def _var_read_as_result(lx, semi, var):
    """After `let var = CALL;` (the `;` at `semi`): is `var` read as a Result further down the block?"""
    g = lx.up[semi]
    close = lx.mate[g] if g >= 0 else len(lx.kind)
    for j in range(semi + 1, close):
        if lx.kind[j] == ID and lx.text[j] == var and not lx.is_p(j - 1, ".") and not lx.is_p(j - 1, "::"):
            if lx.is_p(j + 1, "?"):
                return True
            if lx.is_p(j + 1, ".") and lx.is_id(j + 2) and lx.text[j + 2] in RESULT_READERS:
                return True
            if lx.is_id(j - 1, "match") and lx.is_open(j + 1, "{") and lx.text[j + 2:j + 3] in (["Ok"], ["Err"]):
                return True
            m = enclosing_macro(lx, j)
            if m is not None and m.args is not None and len(m.args) >= 2 and m.name in ("matches", "assert_matches") \
                    and lx.text[m.args[1][0]] in ("Ok", "Err"):
                return True
            if lx.is_p(j + 1, "=") or lx.is_id(j - 1, "let"):
                return False  # rebound
    return False


class _Pairs(object):
    def __init__(self, cx):
        self.by_try = {}
        self.by_plain = {}
        for r in cx.table("try_pairs"):
            if r["class"] not in ("b", "cV", "cP"):
                continue
            if r["container"] == "Distribution":
                continue  # Part 4, C-22: `Distribution::new(<Record> { .. })` — rule domains.dist-ctor
            if r["try"] in ("try_symbol", "try_replace"):
                continue  # C-16: `parse_symbol` (core-renames); C-15: `replace` / `try_replace` stay a pair
            self.by_try.setdefault(r["try"], []).append(r)
            self.by_plain.setdefault(r["plain"], []).append(r)
        # plain METHOD names whose every public owner is in the table: a call is then
        # attributed by its name alone.  Names that std or other types share are never certain.
        self.certain_methods = set()
        for name, rows in self.by_plain.items():
            if name in STD_NAMES or not all(r["call"] == "method" for r in rows):
                continue
            mine = set()
            for r in rows:
                mine.update(r["container"].split("+"))
            owners = set(x.strip() for r in rows for x in r["owners_of_plain_name"].split(",") if x.strip())
            if owners and owners <= mine:
                self.certain_methods.add(name)


def _pairs(cx):
    p = getattr(cx, "_err_pairs", None)
    if p is None:
        p = cx._err_pairs = _Pairs(cx)
    return p


# method names that std or everyday Rust types have: a call is never attributed by such a name
STD_NAMES = frozenset("add sub mul div pow new get get_mut row col gcd lcm call".split())
OPTION_TODAY = frozenset([("Poly", "new")])


def _resolve(view, seg):
    """The item a path segment stands for: itself, or what a `use .. as seg` / `type seg = X<..>;`
    of the file makes it an alias of."""
    cache = view.__dict__.setdefault("_aliases", None)
    if cache is None:
        cache = view._aliases = {}
        for group in view._use_table().values():
            for (segs, alias, _u, _s) in group:
                if alias:
                    cache[alias] = segs[-1]
        text = view.unit.text if view.kind == "code" else view.text
        for m in re.finditer(r"\btype\s+(\w+)\s*=\s*(?:\w+::)*(\w+)", text):
            cache.setdefault(m.group(1), m.group(2))
    return cache.get(seg, seg)


def _row_for(lx, c, rows, view):
    """The pairs-table row that a call certainly belongs to, or (None, reason)."""
    i = c.name_i
    name = lx.text[i]
    if c.style == "path":
        segs = path_before(lx, i)
        if segs is None:
            return None, "the path before `%s` was not understood" % name
        owner = _resolve(view, segs[-1]) if segs else ""
        for r in rows:
            if r["call"] in ("path", "free") and owner == r["container"]:
                return r, ""
        if name.startswith("try_") and segs and segs[-1] != "Self":
            return None, ("`%s::%s` is not a pair of the table (tables/try_pairs.tsv knows `%s` on %s)"
                          % (segs[-1], name, name, ", ".join(sorted(set(r["container"] for r in rows)))))
        return None, ""
    if c.style == "free":
        for r in rows:
            if r["call"] != "free":
                continue
            if _defines_fn(view, name):
                return None, "this file defines a function named `%s`" % name
            text = view.unit.text if view.kind == "code" else view.text
            if re.search(r"\buse\s+[^;]*\b%s\b" % re.escape(name), text) or \
                    re.search(r"\buse\s+[^;]*\b%s::\*" % re.escape(r["container"]), text):
                return r, ""
            return None, "`%s` is called without a path and no `use` line of this file names it" % name
        return None, ""
    for r in rows:
        if r["call"] == "method":
            return r, ""
    return None, ""


def _plain_names():
    from .rules_renames import _load
    return set(r["plain"] for r in _load("try_pairs") if r["class"] in ("b", "cV", "cP")
               and r["container"] != "Distribution" and r["try"] not in ("try_symbol", "try_replace"))


_PLAIN_NAMES = _plain_names()


class TryRename(Rule):
    id = "errors.try-rename"
    description = ("C item 2: `x::try_name(..)` -> `x::name(..)`, `.try_name(..)` -> `.name(..)` for the pairs of "
                   "classes b, cV, cP (tables/try_pairs.tsv), where the `Result` is read at the call")
    trigger = re.compile(r"\btry_[a-z]")

    def find(self, view, cx):
        lx = view.lx
        pairs = _pairs(cx)
        sites = []
        blocked = set()
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or lx.text[i] not in pairs.by_try:
                continue
            c = call_at(lx, i)
            name = lx.text[i]
            if c is None:
                continue
            row, why = _row_for(lx, c, pairs.by_try[name], view)
            if row is None:
                if why:
                    sites.append(self.manual(view, lx.start[i], why, name=name))
                    blocked.add(name)
                continue
            s = chain_start(lx, c.close_i) if c.style == "method" else path_start(lx, i)
            # the reading must be visible only where rule 3 would otherwise take the renamed
            # call for the old plain form
            guarded = c.style != "method" or row["plain"] in pairs.certain_methods
            if not guarded or (s >= 0 and reads_result(view, s, c.close_i, cx)):
                sites.append(self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], row["plain"])],
                                          cls=row["class"], name=name))
            else:
                blocked.add(name)
                sites.append(self.manual(view, lx.start[i],
                                         "the `Result` is not read next to the call (passed on, returned from a "
                                         "closure): rename by hand, and mark the line `codemod:skip` before any "
                                         "later run", cls=row["class"], name=name))
        # the `use` lines that name a free `try_` function follow, once no call of it is left for a person
        for use_i, semi, leaves in use_items(lx):
            new_leaves = []
            changed = False
            for (segs, alias, _a, _b) in leaves:
                last = segs[-1]
                rows = [r for r in pairs.by_try.get(last, ()) if r["call"] == "free"]
                if rows and alias is None and last not in blocked and len(segs) >= 2 \
                        and segs[-2] == rows[0]["container"]:
                    changed = True
                    segs = segs[:-1] + [rows[0]["plain"]]
                if (segs, alias) not in new_leaves:
                    new_leaves.append((segs, alias))
            if changed:
                edit = use_item_edit(view, use_i, semi, new_leaves)
                if edit is not None:
                    sites.append(self.rewrite(view, lx.start[use_i], [edit], cls="import"))
        return sites


class PlainUnwrap(Rule):
    id = "errors.plain-unwrap"
    description = ("C item 3: a call of a function that returns `Result` at 1.0 gets `.unwrap()` (or `?` where the "
                   "enclosing function returns a `Result`); only calls that are certainly the listed function "
                   "(`Type::name(..)`, or a free function named in a `use` line)")
    trigger = re.compile(r"\b(?:%s)\s*(?:::\s*<[^;(]*>\s*)?\(" % "|".join(sorted(_PLAIN_NAMES, key=len, reverse=True)))

    def find(self, view, cx):
        lx = view.lx
        pairs = _pairs(cx)
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or lx.text[i] not in pairs.by_plain:
                continue
            c = call_at(lx, i)
            if c is None:
                continue
            name = lx.text[i]
            rows = pairs.by_plain[name]
            if c.style == "method":
                if name not in pairs.certain_methods or _defines_fn(view, name):
                    continue  # `.add(`, `.pow(`, `.gcd(`: the name does not say whose method it is
                row = rows[0]
            else:
                row, why = _row_for(lx, c, rows, view)
                if row is None:
                    if why and c.style == "free":
                        yield self.manual(view, lx.start[i], why, name=name)
                    continue
            s = chain_start(lx, c.close_i) if c.style == "method" else path_start(lx, i)
            if s < 0:
                continue
            e = c.close_i
            if (row["container"], row["plain"]) in OPTION_TODAY:
                n = e + 1
                if lx.is_p(n, ".") and lx.is_id(n + 1) and lx.text[n + 1] in ("is_none", "is_some"):
                    new = "is_err" if lx.text[n + 1] == "is_none" else "is_ok"
                    yield self.rewrite(view, lx.start[i], [(lx.start[n + 1], lx.end[n + 1], new)],
                                       cls="option-reader", name=name)
                elif not reads_result(view, s, e, cx) and not (lx.is_p(n, ".") or lx.is_p(n, "?")):
                    yield self.manual(view, lx.start[i], "`Option` today, `Result` at 1.0, and the reading is not "
                                                         "`.unwrap()` / `.is_none()`", cls="option", name=name)
                continue
            if reads_result(view, s, e, cx):
                continue  # already reads a Result: migrated, or renamed from the `try_` form
            if lx.is_p(e + 1, "."):
                suffix = ".unwrap()"
            else:
                suffix = cx.unwrap_or_question(view, i)
            yield self.rewrite(view, lx.start[i], [(lx.end[e], lx.end[e], suffix)], cls=row["class"], name=name)


RULES = [
    _pattern("errors.matches", "C item 1: `matches!(r, Err(SymplexError::V { .. }))` -> `matches!(r, Err(e) if "
                               "e.kind() == ErrorKind::V')`; `matches!(e, SymplexError::V { .. })` -> `e.kind() == "
                               "ErrorKind::V'`"),
    _pattern("errors.match-arm", "C item 1: the arm `Err(SymplexError::V { .. }) =>` -> `Err(e) if e.kind() == "
                                 "ErrorKind::V' =>`; `if let Err(SymplexError::V { .. }) = r {` -> `if matches!(r, "
                                 "Err(e) if ..) {`"),
    TryRename(),
    PlainUnwrap(),
    DocLink(),
    _pattern("errors.construct", "C item 5 (library code, with --src-code): `SymplexError::V { operation: a, "
                                 "reason: b }` -> `SymplexError::v(a, b)`", library=True),
    _pattern("errors.other", "uses of a variant that are found and never rewritten: patterns that bind a field, "
                             "constructions on the user side, bare-variant `match` arms"),
]

RULESET = RuleSet(
    "errors",
    "error patterns, the plain/try_ pairs, doc links, constructors (review C §3.2)",
    RULES,
    needs=("tables/error_kinds.tsv — variant -> kind, and for rule 5 the constructor per variant (E1 must fill in "
           "the `ctor` column: the design names only six constructors); tables/try_pairs.tsv — C's pairs "
           "(regenerate from try_pairs_census.py on the 0.60 listing).  E1 uses errors.construct (--src-code); "
           "E2a uses errors.matches, errors.match-arm, errors.doc-link; R1 uses errors.try-rename and "
           "errors.plain-unwrap (in ONE run: see the comment at the top of rules_errors.py)."),
    package="E1, E2a, R1",
)
