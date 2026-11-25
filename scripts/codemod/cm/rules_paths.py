# Rule set `paths` (package S1): review A §3 — the written paths that move, and the names
# that leave the prelude.
#
#   paths.written        symplex::matrix::Matrix::zeros(..)   -> symplex::linalg::Matrix::zeros(..)
#   paths.use            use symplex::{matrix::Matrix, linprog::{q, qi}};
#                                     -> use symplex::linalg::Matrix;  use symplex::num::{q, qi};
#   paths.prelude-names  a file that has `use symplex::prelude::*` and names `Poly`, `QMatrix`, `linsolve`, ..
#                                     -> gains `use symplex::poly::Poly;` ..   (one site per name and scope)
#
# The table is tables/path_moves.tsv: every public non-method path of today (and the
# second paths review A retires) with its home at 1.0, from api_fate.tsv — that is A's
# module_fates.tsv after the tier decisions of Part 4 (units, polytope and ten statistics
# modules under `experimental`).  A path is rewritten only when the item itself is the
# same item at 1.0 (change = move, retype or keep); a renamed, merged or removed item is
# listed, because its other mentions in the file do not move with the path.
import re

from .lexer import ID
from .engine import Rule, RuleSet, Site, REWRITE, MANUAL
from .views import use_items, use_item_edit, PRELUDE_1_0

MOVABLE = ("move", "retype", "keep")


class _Table(object):
    def __init__(self, cx):
        self.items = {}
        self.macros = {}
        for r in cx.table("path_moves"):
            (self.macros if r["kind"] == "macro" else self.items)[r["old_path"]] = r
        # a module is "split" when a child of it lands outside the module's own new home:
        # a glob import of it cannot be rewritten
        self.split = set()
        for old, r in self.items.items():
            parent = old.rsplit("::", 1)[0]
            pr = self.items.get(parent)
            if pr is not None and pr["kind"] == "module" and r["tier"] not in ("removed", "internal"):
                if not r["new_path"].startswith(pr["new_path"] + "::"):
                    self.split.add(parent)
        self.prelude = {}
        for r in cx.table("prelude_moves"):
            self.prelude[r["name"]] = r

    def lookup(self, segs, macro=False):
        """Longest known prefix of the path: (row, number of segments matched) or (None, 0)."""
        for n in range(len(segs), 1, -1):
            key = "::".join(segs[:n])
            if macro and n == len(segs) and key in self.macros:
                return self.macros[key], n
            if key in self.items:
                return self.items[key], n
        return None, 0


def _table(cx):
    t = getattr(cx, "_path_table", None)
    if t is None:
        t = cx._path_table = _Table(cx)
    return t


def _verdict(row):
    """("ok", new_path) or ("manual", reason)."""
    new = row["new_path"]
    old = row["old_path"]
    if row["tier"] == "removed" or row["change"] in ("remove", "merge"):
        hint = (" (see `%s`)" % new) if new not in ("-", "") else ""
        return "manual", "`%s` is removed at 1.0%s" % (old, hint)
    if new.startswith("pub(") or new in ("-", "") or row["tier"] == "internal" and "__internal" not in new:
        return "manual", "`%s` is internal at 1.0 (no public path)" % old
    if row["change"] == "rename":
        return "manual", "`%s` is renamed to `%s`: its other mentions in the file change too" % (old, new)
    if row["change"] not in MOVABLE:
        return "manual", "`%s`: change `%s`" % (old, row["change"])
    return "ok", new


_UNKNOWN = ("`%s` is not an item of `%s` in the table (new, renamed or private): the module moves, "
            "but where this name lives is not known")


def _unknown_child(row, segs, k, macro):
    """The longest known prefix is a MODULE and more segments follow: the next one is not a public
    item of today's listing, so the module's new path says nothing about it."""
    return row["kind"] == "module" and k < len(segs)


def _is_type_seg(seg):
    return seg[:1].isupper()


class Written(Rule):
    id = "paths.written"
    description = ("a written path `symplex::<old>::..` outside `use` items -> its 1.0 path (tables/path_moves.tsv); "
                   "renamed, removed and internal items are listed")
    trigger = re.compile(r"\bsymplex\s*::")

    def find(self, view, cx):
        lx = view.lx
        table = _table(cx)
        in_use = set()
        for use_i, semi, _leaves in use_items(lx):
            in_use.update(range(use_i, semi + 1))
        i = 0
        n = len(lx.kind)
        while i < n:
            if not (lx.kind[i] == ID and lx.text[i] == "symplex" and lx.is_p(i + 1, "::")) or i in in_use \
                    or lx.is_p(i - 1, ".") or (lx.is_p(i - 1, "::") and lx.is_id(i - 2)):
                i += 1
                continue
            segs = ["symplex"]
            toks = [i]
            j = i
            while lx.is_p(j + 1, "::") and lx.is_id(j + 2):
                j += 2
                segs.append(lx.text[j])
                toks.append(j)
            macro = lx.is_p(j + 1, "!")
            row, k = table.lookup(segs, macro)
            nxt = j + 1
            if row is None:
                i = nxt
                continue
            verdict, new = _verdict(row)
            pos = lx.start[i]
            if verdict == "ok" and _unknown_child(row, segs, k, macro):
                if new != row["old_path"]:
                    yield self.manual(view, pos, _UNKNOWN % (segs[k], row["old_path"]), name=row["old_path"])
            elif verdict == "manual":
                yield self.manual(view, pos, new, name=row["old_path"])
            elif new != row["old_path"]:
                yield self.rewrite(view, pos, [(lx.start[i], lx.end[toks[k - 1]], new)], name=row["old_path"])
            i = nxt


class Use(Rule):
    id = "paths.use"
    description = ("a `use symplex::..` item whose leaves move -> regenerated with the 1.0 paths (one line per "
                   "parent module); a glob of a module that is split, a renamed or removed item, and a module "
                   "imported by name are listed")
    trigger = re.compile(r"\buse\s+(?:::\s*)?symplex\b")

    def find(self, view, cx):
        lx = view.lx
        table = _table(cx)
        for use_i, semi, leaves in use_items(lx):
            if not leaves or leaves[0][0][:1] != ["symplex"]:
                continue
            new_leaves = []
            changed = False
            moved = 0
            problems = []
            for (segs, alias, _a, _b) in leaves:
                if segs[-1] == "*":
                    mod = segs[:-1]
                    row, k = table.lookup(mod)
                    if row is None or k != len(mod):
                        new_leaves.append((segs, alias))
                        continue
                    key = "::".join(mod)
                    verdict, new = _verdict(row)
                    if verdict == "manual":
                        problems.append(new)
                        new_leaves.append((segs, alias))
                    elif key in table.split:
                        problems.append("`%s::*`: the module's items go to more than one 1.0 module" % key)
                        new_leaves.append((segs, alias))
                    else:
                        if new != key:
                            changed = True
                            moved += 1
                        new_leaves.append((new.split("::") + ["*"], alias))
                    continue
                row, k = table.lookup(segs)
                if row is None:
                    new_leaves.append((segs, alias))
                    continue
                verdict, new = _verdict(row)
                if verdict == "manual":
                    problems.append(new)
                    new_leaves.append((segs, alias))
                    continue
                if new == row["old_path"]:
                    new_leaves.append((segs, alias))
                    continue
                if _unknown_child(row, segs, k, False):
                    problems.append(_UNKNOWN % (segs[k], row["old_path"]))
                    new_leaves.append((segs, alias))
                    continue
                if k == len(segs) and row["kind"] == "module" and alias is None \
                        and new.split("::")[-1] != segs[-1]:
                    # (a module that keeps its name under a new parent can simply be imported from there)
                    problems.append("the module `%s` is imported by name: paths written through it (`%s::..`) "
                                    "would not follow" % (row["old_path"], segs[-1]))
                    new_leaves.append((segs, alias))
                    continue
                new_segs = new.split("::") + segs[k:]
                if len(new_segs) >= 2 and _is_type_seg(new_segs[-2]) and not _is_type_seg(new_segs[-1]):
                    problems.append("`%s` becomes the associated function `%s`: it cannot be imported"
                                    % ("::".join(segs), "::".join(new_segs)))
                    new_leaves.append((segs, alias))
                    continue
                changed = True
                moved += 1
                new_leaves.append((new_segs, alias))
            pos = lx.start[use_i]
            if changed:
                edit = use_item_edit(view, use_i, semi, new_leaves)
                if edit is None:
                    yield self.manual(view, pos, "a `pub use` that would become several lines")
                else:
                    # the site is one `use` item; `name` says how many of its paths move
                    yield self.rewrite(view, pos, [edit], cls="use", name="%d" % moved)
            for p in problems:
                yield self.manual(view, pos, p, cls="use")


class PreludeNames(Rule):
    id = "paths.prelude-names"
    description = ("a name that leaves the prelude (88 -> 15 names), used bare in a file that glob-imports the "
                   "prelude -> a `use` line for it (tables/prelude_moves.tsv); one site per name and scope")
    trigger = re.compile(r"\bprelude\s*::\s*\*")

    def find(self, view, cx):
        lx = view.lx
        table = _table(cx)
        globbed = any(u[0][-2:] == ["prelude", "*"] and u[0][0] == "symplex"
                      for g in view._use_table().values() for u in g)
        if not globbed:
            return
        in_use = set()
        explicit = set()  # names this view imports from symplex by an explicit `use` (rule paths.use moves those)
        for use_i, semi, leaves in use_items(lx):
            in_use.update(range(use_i, semi + 1))
            for (segs, alias, _a, _b) in leaves:
                if segs[0] in ("symplex", "crate") and segs[-1] != "*":
                    explicit.add(alias or segs[-1])
        seen = set()
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or i in in_use:
                continue
            name = lx.text[i]
            row = table.prelude.get(name)
            if row is None or name in PRELUDE_1_0 or name in explicit:
                continue
            if lx.is_p(i - 1, "::") or lx.is_p(i - 1, ".") or lx.is_id(i - 1, "fn") or lx.is_id(i - 1, "struct") \
                    or lx.is_id(i - 1, "enum") or lx.is_id(i - 1, "trait") or lx.is_id(i - 1, "type") \
                    or lx.is_id(i - 1, "mod"):
                continue
            if (lx.is_id(i - 1, "let") or lx.is_id(i - 1, "mut")) and not (
                    lx.is_open(i + 1, "{") or lx.is_open(i + 1, "(") or lx.is_p(i + 1, "::")):
                continue  # `let Name = ..` binds a variable; `let Qr { q, r } = ..` names the type
            if lx.is_p(i + 1, ":") and not lx.is_p(i - 1, ":") and not lx.is_p(i - 1, "<") \
                    and not lx.is_p(i - 1, "&"):
                continue  # `Name: value` — a struct field or a binding, not the type
            if row["kind"] == "macro":
                if not lx.is_p(i + 1, "!"):
                    continue
            elif lx.is_p(i + 1, "!"):
                continue
            elif name[:1].islower() and not lx.is_open(i + 1, "("):
                continue  # a function of the prelude is a site only where it is called
            scope = self._scope(lx, i)
            if (name, scope) in seen:
                continue
            seen.add((name, scope))
            pos = lx.start[i]
            if name == "Q":
                continue  # the rational alias: rule set `numbers`
            verdict, new = _verdict(dict(row, old_path=name))
            if verdict == "manual" or row["tier"] == "internal":
                yield self.manual(view, pos, new if verdict == "manual" else "`%s` is internal at 1.0" % name,
                                  name=name)
                continue
            if view._local_def(name):
                continue
            written = view.use_name(i, new)
            if view._pending_imports:
                yield Site(self.id, view, pos, REWRITE, "", (), row["tier"], name)
            elif written != name:
                view.take_pending_imports()
                yield self.manual(view, pos, "another item named `%s` is imported here" % name, name=name)

    @staticmethod
    def _scope(lx, i):
        g = lx.up[i]
        while g >= 0:
            if lx.is_open(g, "{") and lx.is_id(g - 1) and lx.is_id(g - 2, "mod"):
                return g
            g = lx.up[g]
        return -1


RULESET = RuleSet(
    "paths",
    "written paths and `use` lines that move; names that leave the prelude (review A §3)",
    [Written(), Use(), PreludeNames()],
    needs=("tables/path_moves.tsv and tables/prelude_moves.tsv — regenerate both with tables/build_tables.py from "
           "the api_fate.tsv of the day (WP-00 regenerates the listing from 0.60).  The facade (S1) must exist "
           "with the new paths before the output compiles; run with the old paths still exported, then remove "
           "them (A §3 \"Order\")."),
    package="S1",
)
