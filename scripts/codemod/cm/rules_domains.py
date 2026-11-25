# Rule set `domains` (review F §6): the two table-driven rewrites, plus the domain renames
# of tables/domain_renames.tsv.
#
#   domains.dist-ctor    Distribution::normal(m, s)      -> Distribution::new(Normal { mean: m, std: s })?
#                        Distribution::try_normal(m, s)  -> Distribution::new(Normal { mean: m, std: s })
#   domains.matrix-get   m.get(i, j)                     -> &m[(i, j)]        (m.get(i, j).f() -> m[(i, j)].f())
#   domains.matrix-try   m.try_get(i, j).is_none()       -> m.get(i, j).is_none()
import re

from .lexer import ID, is_boundary_before
from .chain import call_at, chain_start, path_before, enclosing_macro, whole_arg
from .engine import Rule, RuleSet
from .rules_renames import rules_from, _load, _defines_fn

OPTION_READERS = frozenset(
    "is_none is_some unwrap expect map and_then copied cloned unwrap_or unwrap_or_else unwrap_or_default ok_or "
    "ok_or_else is_some_and filter".split()
)
# evidence that a file works with matrices: a matrix type or the macro is named, or a method
# is called that only the matrix types have
_MATRIXY = re.compile(r"\b(?:Matrix|QMatrix|ZMatrix|ExactMatrix)\b|\bmatrix!\s*[\[(]"
                      r"|\.(?:transpose|matmul|det|nrows|ncols|to_rotation_matrix|mass_matrix|inertia_matrix|"
                      r"jacobian|char_poly|eigenvals)\(")
_OTHER_GET2 = re.compile(r"\b(?:RatingTable|DimMap)\b")


def reads_option(lx, s, e):
    """Is the value of the chain s..e read, right there, as an `Option`?"""
    n = e + 1
    if lx.is_p(n, ".") and lx.is_id(n + 1) and lx.text[n + 1] in OPTION_READERS:
        return True
    if lx.is_p(n, "?"):
        return True
    if (lx.is_p(n, "==") or lx.is_p(n, "!=")) and lx.text[n + 1:n + 2] in (["None"], ["Some"]):
        return True
    if lx.is_p(s - 1, "="):
        j = s - 2
        k = 0
        while j >= 0 and k < 30 and not lx.is_p(j, ";") and not lx.is_open(j, "{"):
            if lx.is_id(j, "let"):
                return any(lx.text[x] in ("Some", "None") for x in range(j + 1, s - 1))
            j -= 1
            k += 1
    if lx.is_id(s - 1, "match") and lx.is_open(n, "{"):
        return lx.text[n + 1:n + 2] in (["Some"], ["None"])
    mac = enclosing_macro(lx, s)
    if mac is not None and mac.args is not None and len(mac.args) >= 2:
        idx = whole_arg(mac.args, s, e)
        if idx in (0, 1) and mac.name in ("assert_eq", "assert_ne", "matches", "debug_assert_eq"):
            oa = mac.args[1 - idx][0]
            return lx.text[oa] in ("Some", "None")
    return False


class DistCtor(Rule):
    id = "domains.dist-ctor"
    description = ("`Distribution::<family>(a, b)` -> `Distribution::new(<Record> { f1: a, f2: b })?` (`.unwrap()` "
                   "where `?` cannot be used); `Distribution::try_<family>(a, b)` -> `Distribution::new(<Record> "
                   "{ .. })`; table: tables/dist_ctors.tsv")
    trigger = re.compile(r"\bDistribution\s*::\s*(?:try_)?[a-z_0-9]+")

    def find(self, view, cx):
        lx = view.lx
        table = getattr(cx, "_dist", None)
        if table is None:
            table = cx._dist = {r["ctor"]: r for r in cx.table("dist_ctors")}
            cx._dist_extra = ("finite",)
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or not lx.is_p(i - 1, "::") or not lx.is_id(i - 2, "Distribution"):
                continue
            name = lx.text[i]
            twin = name.startswith("try_")
            fam = name[4:] if twin else name
            row = table.get(fam)
            if row is None:
                if fam in cx._dist_extra:
                    if call_at(lx, i) is not None:
                        yield self.manual(view, lx.start[i], "`Finite` has private fields: its record form is not "
                                                             "in the design; by hand", name=name)
                continue
            c = call_at(lx, i)
            if c is None:
                if not lx.is_open(i + 1):
                    yield self.manual(view, lx.start[i], "the constructor is passed as a function value", name=name)
                continue
            if row["rewrite"] != "yes":
                yield self.manual(view, lx.start[i], row["note"] or "not a plain record", name=name)
                continue
            fields = row["fields_in_argument_order"].split(",")
            args = lx.split_args(c.open_i)
            if args is None or len(args) != len(fields):
                yield self.manual(view, lx.start[i], "expected %d argument(s) for `%s`, found %s"
                                  % (len(fields), fam, "an argument list the tool cannot split" if args is None
                                     else len(args)), name=name)
                continue
            rec = view.use_name(i, "symplex::stats::" + row["record"])
            edits = [(lx.start[i], lx.end[i], "new")]
            for n, ((a, b), f) in enumerate(zip(args, fields)):
                lead = "%s { " % rec if n == 0 else ""
                label = "" if (a == b and lx.text[a] == f) else "%s: " % f
                edits.append((lx.start[a], lx.start[a], lead + label))
            last = args[-1][1]
            edits.append((lx.end[last], lx.end[last], " }"))
            if not twin:
                edits.append((lx.end[c.close_i], lx.end[c.close_i], cx.unwrap_or_question(view, i)))
            yield self.rewrite(view, lx.start[i], edits, cls="try" if twin else "plain", name=name)


class MatrixGet(Rule):
    id = "domains.matrix-get"
    description = ("`m.get(i, j)` -> `&m[(i, j)]` (type-identical); `m.get(i, j).f()` -> `m[(i, j)].f()`; only in "
                   "files that name a matrix type, and not where the result is already read as an `Option`")
    trigger = re.compile(r"\.\s*get\s*\(")

    def find(self, view, cx):
        lx = view.lx
        unit = view.unit
        matrixy = None
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or lx.text[i] != "get" or not lx.is_p(i - 1, "."):
                continue
            c = call_at(lx, i)
            if c is None or c.open_i != i + 1:
                continue
            args = lx.split_args(c.open_i)
            if args is None or len(args) != 2:
                continue
            if matrixy is None:
                matrixy = _MATRIXY.search(unit.text) is not None
            pos = lx.start[i]
            s = chain_start(lx, c.close_i)
            if s < 0:
                if matrixy:
                    yield self.manual(view, pos, "the receiver is not a plain chain of calls", name="get")
                continue
            if reads_option(lx, s, c.close_i):
                continue  # already the 1.0 `get -> Option`, or another type's `get`
            if not matrixy:
                yield self.manual(view, pos, "nothing in this file says that the receiver is a matrix (no matrix "
                                             "type, no `matrix!`, no matrix method): if it is, write `m[(i, j)]`",
                                  name="get")
                continue
            if _OTHER_GET2.search(unit.text) or _defines_fn(view, "get"):
                yield self.manual(view, pos, "this file also has another type with a two-argument `get`", name="get")
                continue
            dot = i - 1
            edits = [(lx.start[dot], lx.end[c.open_i], "[("), (lx.start[c.close_i], lx.end[c.close_i], ")]")]
            n = c.close_i + 1
            cls = "postfix"
            if not (lx.is_p(n, ".") or lx.is_p(n, "?") or lx.is_open(n, "[") or lx.is_open(n, "(")):
                if lx.is_p(s - 1, "*"):
                    edits.append((lx.start[s - 1], lx.start[s], ""))  # `*m.get(i, j)` -> `m[(i, j)]`
                    cls = "deref"
                else:
                    edits.append((lx.start[s], lx.start[s], "&"))
                    cls = "borrow"
            yield self.rewrite(view, pos, edits, cls=cls, name="get")


class MatrixTry(Rule):
    id = "domains.matrix-try"
    description = ("`m.try_get(i, j)`, `try_get_mut`, `try_row`, `try_col` -> `get`, `get_mut`, `row`, `col` where "
                   "the result is read as an `Option` right at the call (so that a later run still knows it)")
    trigger = re.compile(r"\btry_(?:get|get_mut|row|col)\b")
    NAMES = {"try_get": "get", "try_get_mut": "get_mut", "try_row": "row", "try_col": "col"}

    def find(self, view, cx):
        lx = view.lx
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or lx.text[i] not in self.NAMES or not lx.is_p(i - 1, "."):
                continue
            c = call_at(lx, i)
            if c is None:
                continue
            s = chain_start(lx, c.close_i)
            if s >= 0 and reads_option(lx, s, c.close_i):
                yield self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], self.NAMES[lx.text[i]])],
                                   name=lx.text[i])
            else:
                yield self.manual(view, lx.start[i],
                                  "the `Option` is not read next to the call: rename by hand (a later run of "
                                  "domains.matrix-get must not take the renamed call for the old panicking `get`)",
                                  name=lx.text[i])


RULESET = RuleSet(
    "domains",
    "the domain rewrites of review F: Distribution constructors, Matrix::get, and the domain renames",
    [DistCtor(), MatrixGet(), MatrixTry()] + rules_from(_load("domain_renames")),
    needs=("tables/dist_ctors.tsv — constructor -> record and field order, regenerated from the source of the "
           "day by tables/build_tables.py (ST1 must run it after any change of a family struct); "
           "tables/domain_renames.tsv.  One package uses only its own rules: ST1 dist-ctor, density, "
           "characteristic_function; ST2 the p_value rows; LA1 matrix-get, matrix-try, the eigen rows; LA2 the "
           "matrix_* rows; PL1 the polynomial rows (select with --opt rules=<id prefix>,...)."),
    package="ST1, ST2, LA1, LA2, PL1",
)
