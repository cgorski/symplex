# Table-driven renames: the rule sets `core-renames`, `subs` (review E) and the rename
# part of `domains` (review F).  One rule per row of tables/core_renames.tsv and
# tables/domain_renames.tsv; the shapes are described at the top of those files.
#
# A rename keys on the OLD name, which is what makes it safe: `subs_i64`,
# `piecewise_simplify`, `eigenvals` are names nothing else in Rust has.  Where the public
# API itself has the name on a second type that keeps it (the 31 unit copies of
# `log_combine`; `GenPoly`), the rule refuses in files that use that type.
import re

from .lexer import ID, is_boundary_before
from .chain import call_at, chain_start, chain_end, path_before, path_start
from .engine import Rule, RuleSet

_UNITS = re.compile(r"\bunits::|\bdim!\s*\(|\bQty\s*<|\bconst_assert_dim!")
_FAMILY = re.compile(r"\bimpl\s+(?:\w+::)*Family\s+for\b")


def _defines_fn(view, name):
    """Does the code around a site define a function of this name (so that a call may be to it)?

    For a test or an example that is the whole file.  For a doctest or a Markdown block it is
    the block alone: a doctest is compiled as a crate of its own and cannot see the private
    functions of the file whose doc comment it sits in — and in `src/` that file DEFINES the
    library function the doctest calls."""
    scope = view.unit if view.kind == "code" else view
    cache = scope.__dict__.setdefault("_defines_fn", {})
    if name not in cache:
        cache[name] = re.search(r"\bfn\s+%s\b" % re.escape(name), scope.text) is not None
    return cache[name]


def _in_use_item(lx, i):
    """Is token i inside a `use ...;` item?"""
    j = i
    n = 0
    while j >= 0 and n < 200:
        n += 1
        if lx.is_p(j, ";"):
            return False
        if lx.is_id(j, "use"):
            return True
        if lx.is_open(j) and lx.text[j] != "{":
            return False
        j -= 1
    return False


class TableRename(Rule):
    def __init__(self, row):
        self.row = row
        self.id = row["id"]
        self.shape = row["shape"]
        self.old = row["old"]
        self.new = row["new"]
        self.extra = row.get("extra", "") or ""
        self.units = row.get("units", "-") or "-"
        self.note = row.get("note", "") or ""
        self.arity = None
        if self.shape.startswith("method/"):
            self.arity = int(self.shape.split("/")[1])
            self.shape = "method"
        self.types = ()
        if self.shape.startswith("path:"):
            self.types = tuple(self.shape[5:].split("|"))
            self.shape = "path"
        if self.shape.startswith("in:"):
            self.types = tuple(self.shape[3:].split("|"))
            self.shape = "in"
        self.trigger = re.compile(r"\b%s\b" % re.escape(self.old))
        self.description = self._describe()

    def _describe(self):
        o, n, x = self.old, self.new, self.extra
        d = {
            "method": "`.%s(` -> `.%s(`" % (o, n),
            "path": "`%s::%s(` -> `%s::%s(`" % ("|".join(self.types), o, "|".join(self.types), n),
            "in": "`<%s>::%s(` -> `::%s(`" % ("|".join(self.types), o, n),
            "flag": "`.%s(true)` -> `.%s()`, `.%s(false)` -> `.%s()`" % (o, x, o, n),
            "recv-first": "`%s::%s(a, ..)` -> `a.%s(..)`" % (x, o, n),
            "ctx-first": "`%s::%s(&ctx, ..)` -> `ctx.%s(..)`" % (x, o, n),
            "append-arg": "`.%s(a)` -> `.%s(a, %s)`" % (o, n, x.split("|")[0]),
            "ineq": "`.%s(v)` -> `.%s(0).%s(v)`" % (o, x, n),
            "to-field": "`.%s(..)` -> `.%s.%s(..)`" % (o, x, n),
            "free": "`%s(` -> `%s(` (a free function, also in `use` lists)" % (o, n),
            "type": "the type name `%s` -> `%s`" % (o, n),
        }.get(self.shape)
        if d is None:
            d = "`%s`: listed for a person, never rewritten" % o
        if self.arity is not None:
            d += " (calls with %d argument%s)" % (self.arity, "" if self.arity == 1 else "s")
        return d + (" — " + self.note if self.note else "")

    # ------------------------------------------------------------------ find
    def find(self, view, cx):
        lx = view.lx
        unit = view.unit
        old = self.old
        shape = self.shape
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or lx.text[i] != old:
                continue
            pos = lx.start[i]
            if shape in ("type", "manual-type"):
                if lx.is_p(i - 1, ".") or lx.is_id(i - 1, "fn") or lx.is_id(i - 1, "struct") \
                        or lx.is_id(i - 1, "enum") or lx.is_id(i - 1, "type"):
                    continue
                if shape == "manual-type":
                    yield self.manual(view, pos, self.note or "by hand", name=old)
                elif view._local_def(old):
                    yield self.manual(view, pos, "this file defines its own `%s`" % old, name=old)
                else:
                    yield self.rewrite(view, pos, [(pos, lx.end[i], self.new)], name=old)
                continue
            c = call_at(lx, i)
            if shape == "free" and c is None and _in_use_item(lx, i) and not lx.is_id(i + 1, "as"):
                yield self.rewrite(view, pos, [(pos, lx.end[i], self.new)], cls="import", name=old)
                continue
            if c is None:
                if lx.is_p(i - 1, "::") and not lx.is_id(i - 2, "self") and shape not in ("manual-free", "free"):
                    segs = path_before(lx, i)
                    if segs and segs[-1][:1].isupper() and not _in_use_item(lx, i):
                        yield self.manual(view, pos, "named as a function value (`%s::%s` without a call)"
                                          % (segs[-1], old), cls="reference", name=old)
                continue
            if shape in ("manual", "manual-free"):
                if shape == "manual" and c.style == "free":
                    continue
                if shape == "manual-free" and c.style == "method":
                    continue
                yield self.manual(view, pos, self.note or "by hand", name=old)
                continue
            near = unit.text if view.kind == "code" else view.text  # a doctest is a crate of its own
            if old in ("density", "characteristic_function") and _FAMILY.search(near):
                yield self.manual(view, pos, "this file implements `Family`, whose own method keeps the name",
                                  name=old)
                continue
            if _defines_fn(view, old):
                yield self.manual(view, pos, "this file defines a function named `%s`: the call may be to it" % old,
                                  name=old)
                continue
            if self.units == "keep" and c.style == "method" and _UNITS.search(near):
                yield self.manual(view, pos, "this file uses unit quantities, whose copy of `%s` keeps its name at "
                                             "1.0: the receiver's type is not in the text" % old, name=old)
                continue
            site = getattr(self, "_" + shape.replace("-", "_"))(view, cx, i, c)
            if site is not None:
                yield site

    # ---------------------------------------------------------------- shapes
    def _method(self, view, cx, i, c):
        lx = view.lx
        if c.style == "free":
            return None
        if self.arity is not None:
            args = lx.split_args(c.open_i)
            if args is None:
                return self.manual(view, lx.start[i], "the argument list was not understood", name=self.old)
            if len(args) != self.arity:
                return None  # another method of the same name (or the already renamed one)
        return self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], self.new)], name=self.old)

    def _path(self, view, cx, i, c):
        lx = view.lx
        segs = path_before(lx, i)
        if c.style != "path" or not segs or segs[-1] not in self.types:
            return None
        return self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], self.new)], name=self.old)

    _in = _path

    def _free(self, view, cx, i, c):
        lx = view.lx
        if c.style == "method":
            return None
        return self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], self.new)], name=self.old)

    def _flag(self, view, cx, i, c):
        lx = view.lx
        if c.style == "free":
            return None
        args = lx.split_args(c.open_i)
        if args is None:
            return self.manual(view, lx.start[i], "the argument list was not understood", name=self.old)
        if len(args) == 0:
            if self.old == self.new:
                return None  # already the flag-less form
            return self.manual(view, lx.start[i], "called without the flag", name=self.old)
        a, b = args[0]
        if len(args) != 1 or a != b or lx.text[a] not in ("true", "false"):
            return self.manual(view, lx.start[i], "the `force` flag is not a literal `true` / `false`", name=self.old)
        new = self.extra if lx.text[a] == "true" else self.new
        return self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], new),
                                                (lx.end[c.open_i], lx.start[c.close_i], "")],
                            cls=lx.text[a], name=self.old)

    def _recv_first(self, view, cx, i, c):
        lx = view.lx
        segs = path_before(lx, i)
        if c.style != "path" or not segs or segs[-1] not in (self.extra, "Expr", "Self"):
            return None
        if segs[-1] == "Self":
            return self.manual(view, lx.start[i], "`Self::%s`: inside an impl" % self.old, name=self.old)
        args = lx.split_args(c.open_i)
        if not args:
            return self.manual(view, lx.start[i], "the argument list was not understood", name=self.old)
        a, b = args[0]
        inner = a + 1 if lx.is_p(a, "&") else a
        if inner <= b and chain_end(lx, inner) == b:
            recv = lx.span(inner, b)
        else:
            recv = "(" + lx.span(a, b) + ")"
        ps = path_start(lx, i)
        if ps < 0:
            return None
        edits = [(lx.start[ps], lx.end[c.open_i], "%s.%s(" % (recv, self.new))]
        if len(args) > 1:
            edits.append((lx.start[a], lx.start[args[1][0]], ""))
        else:
            edits.append((lx.start[a], lx.start[c.close_i], ""))
        return self.rewrite(view, lx.start[i], edits, name=self.old)

    _ctx_first = _recv_first

    def _append_arg(self, view, cx, i, c):
        lx = view.lx
        if c.style == "free":
            return None
        args = lx.split_args(c.open_i)
        if not args:
            return self.manual(view, lx.start[i], "the argument list was not understood", name=self.old)
        text, _, imp = self.extra.partition("|")
        if imp:
            short = imp.split("::")[-1]
            text = text.replace(short, view.use_name(i, imp))
        last = args[-1][1]
        return self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], self.new),
                                                (lx.end[last], lx.end[last], ", " + text)], name=self.old)

    def _ineq(self, view, cx, i, c):
        lx = view.lx
        if c.style != "method":
            return None
        return self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], "%s(0).%s" % (self.extra, self.new))],
                            name=self.old)

    def _to_field(self, view, cx, i, c):
        lx = view.lx
        if c.style != "method":
            return None
        field = self.extra
        r = i - 2  # last token of the receiver
        if lx.is_id(r, field) and lx.is_p(r - 1, "."):
            # `x.p_value.p_value_f64()`: the receiver is the field already
            return self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], self.new)], cls="on-field",
                                name=self.old)
        s = chain_start(lx, r)
        recv = lx.span(s, r) if s >= 0 else ""
        if s < 0 or re.search(r"\b%ss\b" % field, recv):
            # an element of `p_values`, a closure's parameter: a bare value of the field's type
            return self.manual(view, lx.start[i],
                               "the receiver may be a `%s` value itself (then `.%s()`), not a result that has the "
                               "field" % (field, self.new), name=self.old)
        if s == r and _pvalue_bound(view.unit, lx.text[r], field):
            return self.manual(view, lx.start[i],
                               "`%s` is bound from a `.%s` field or typed `PValue` in this file: then `.%s()`, "
                               "not `.%s.%s()`" % (lx.text[r], field, self.new, field, self.new), name=self.old)
        return self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], "%s.%s" % (field, self.new))],
                            name=self.old)


def _pvalue_bound(unit, name, field):
    """Does the file bind `name` to a bare value of the field's type: from a `.<field>` field, with
    the type `PValue`, or as the loop / closure variable over a `<field>s` list?"""
    cache = unit.__dict__.setdefault("_pvalue_bound", {})
    key = (name, field)
    if key not in cache:
        n = re.escape(name)
        rx = (r"\blet\s+(?:mut\s+)?%s\b[^;=]*=[^;]*\.%s\s*(?:;|\.clone\(\)|\))"
              r"|\b%s\s*:\s*&?\s*(?:[\w:]+::)?PValue\b"
              r"|\b%ss\b[^;{]*\|\s*&?\(?%s\b[^|]*\|"
              r"|\bfor\s+\(?[^{;]*\b%s\b[^{;]*\bin\b[^{;]*\b%ss\b" % (n, field, n, field, n, n, field))
        cache[key] = re.search(rx, unit.text) is not None
    return cache[key]


def rules_from(cx_rows, prefix=None):
    out = []
    for row in cx_rows:
        if prefix and not row["id"].startswith(prefix):
            continue
        out.append(TableRename(row))
    return out


def _load(name):
    import csv
    import os
    from .engine import TABLES
    with open(os.path.join(TABLES, name + ".tsv"), newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
    return list(csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE))


CORE = RuleSet(
    "core-renames",
    "the core method renames (review E §2.3.3; API_1.0.md §3.4)",
    rules_from(_load("core_renames"), "core."),
    needs=("tables/core_renames.tsv — review E's rename table; tables/homonyms.tsv says which other public "
           "types have a method of the same name.  Run it LAST in its block (package NM1: renames on the "
           "survivors), after re-generating homonyms.tsv from the listing of that day.  `as_ex` -> `to_ex` must "
           "be applied before tests call the new `Symbol::as_ex`."),
    package="NM1 (and K2 for expr_type, SV1 for the inequality solvers)",
)

SUBS = RuleSet(
    "subs",
    "`subs_i64(&x, 3)` -> `subs(&x, 3)` and the two `subs_map` twins (review E N-1)",
    rules_from(_load("core_renames"), "subs."),
    needs=("nothing but the three names.  `subs` must accept `impl ToEx` first (package V2b), otherwise the "
           "renamed calls do not compile."),
    package="V2b",
)
