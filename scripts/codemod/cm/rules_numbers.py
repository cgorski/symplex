# Rule set `numbers` (packages N2, N3): review D §3 "D-R2" — four rewrites, and the
# sites D lists for a person.
#
#   1  numbers.bigint          BigInt -> Integer; `use num_bigint::BigInt;` -> `use symplex::num::Integer;`
#   2  numbers.rational        Ratio<BigInt>, BigRational, Q -> Rational; X::from_integer(n) -> Rational::from(n)
#   3  numbers.trait-imports   use num_traits::{Zero, One, Signed, ToPrimitive}; -> removed
#   4  numbers.complex-import  use num_complex::Complex64; -> use symplex::num::Complex64;
#      numbers.ratio-new       Ratio::new(a, b) / Q::new / BigRational::new: listed (the constructor returns Result)
#      numbers.by-hand         parse_bytes, to_integer, modpow, extended_gcd, magnitude, to_biguint, Sign::, ..
#      numbers.renames         D-R2x: ctx.from_bigint / from_i128 / from_u64 -> from_integer, from_ratio ->
#                              from_rational, as_bigint -> as_integer
#
# Part 4, C-12: the number types are not in the prelude, so the tool writes the import
# line (`use symplex::num::{Integer, Rational};`) instead of only deleting the old one.
import re

from .lexer import ID
from .chain import call_at, path_before, path_start
from .engine import Rule, RuleSet
from .views import use_items, use_item_edit

NUM_CRATES = ("num_bigint", "num_rational", "num_traits", "num_integer", "num_complex")
INTEGER = "symplex::num::Integer"
RATIONAL = "symplex::num::Rational"
COMPLEX = "symplex::num::Complex64"
DROP_TRAITS = ("Zero", "One", "Signed", "ToPrimitive")
_OTHER_RATIO = re.compile(r"\bRatio\s*<\s*(?!BigInt\b|Integer\b|num_bigint::BigInt\b)\w|\bRational64\b|\bRational32\b")
_Q_LOCAL = re.compile(
    r"\b(?:fn|impl|struct|enum|trait|type)\b[^{;(=]*<[^>{;]*\bQ\b|\b(?:let|const|static|struct|type|enum)\s+(?:mut\s+)?Q\b"
    r"|\|\s*Q\s*[|,:]")


_Q_ALIAS = re.compile(r"\btype\s+Q\s*=\s*(?:\w+::)*(?:Ratio\s*<\s*(?:\w+::)*(?:BigInt|Integer)\s*>|BigRational|Rational)\s*;")


def _crate_of(segs):
    """("num_bigint", rest) for a leaf of one of the num crates (directly or through
    symplex's re-export), else (None, segs)."""
    if segs and segs[0] in NUM_CRATES:
        return segs[0], segs[1:]
    if len(segs) >= 2 and segs[0] in ("symplex", "crate") and segs[1] in NUM_CRATES:
        return segs[1], segs[2:]
    return None, segs


def _q_alias(view):
    """Is `Q` symplex's rational alias in this view?  "yes" / "no" / "alias" (the file defines
    `type Q = Ratio<BigInt>;` itself) / "local" (the file gives the name to something else)."""
    cached = getattr(view, "_q_alias", None)
    if cached is not None:
        return cached
    if _Q_ALIAS.search(view.unit.text):
        verdict = "alias"  # `type Q = Ratio<BigInt>;` of the file's own: the name stays, the type moves
    elif _Q_LOCAL.search(view.unit.text):
        verdict = "local"
    else:
        verdict = "no"
        for group in view._use_table().values():
            for (segs, alias, _u, _s) in group:
                if alias:
                    continue
                if segs[-1] == "Q" and segs[0] in ("symplex", "crate", "super"):
                    verdict = "yes"
                if segs[-1] == "*" and segs[0] == "symplex" and segs[-2] in ("prelude", "linprog", "numeric", "data"):
                    verdict = "yes"
    view._q_alias = verdict
    return verdict


DROPPED = {
    "num_bigint": ("BigInt",),
    "num_rational": ("Ratio", "BigRational"),
    "num_traits": DROP_TRAITS,
    "num_integer": ("Integer",),
    "num_complex": ("Complex64",),
}


def _dropped(segs, alias):
    """Is this `use` leaf one that the import rule removes?  (Then it does not clash with the
    import of the new type that is written in the same pass.)"""
    crate, rest = _crate_of(segs)
    return crate is not None and alias is None and len(rest) == 1 and rest[0] in DROPPED[crate]


class Imports(Rule):
    """One site per `use` item that names a num crate.  The rule id says which of D's four
    items the import belongs to (the first that applies)."""

    library_code = True
    trigger = re.compile(r"\bnum_(?:bigint|rational|traits|integer|complex)\b|\buse\b[^;]*\bQ\b")
    ids = ("numbers.bigint", "numbers.rational", "numbers.trait-imports", "numbers.complex-import")

    def __init__(self, rule_id, description):
        self.id = rule_id
        self.description = description

    def find(self, view, cx):
        cached = getattr(view, "_num_import_sites", None)
        if cached is None:
            cached = view._num_import_sites = list(self._all(view))
        return [s for s in cached if s.rule == self.id]

    def _all(self, view):
        lx = view.lx
        q = None
        for use_i, semi, leaves in use_items(lx):
            keep = []
            needs = []
            cats = []
            problems = []
            for (segs, alias, _a, _b) in leaves:
                crate, rest = _crate_of(segs)
                if crate is None:
                    if segs[-1] == "Q" and segs[0] in ("symplex", "crate") and alias is None:
                        if q is None:
                            q = _q_alias(view)
                        if q == "yes":
                            needs.append(RATIONAL)
                            cats.append("numbers.rational")
                            continue
                    keep.append((segs, alias))
                    continue
                name = rest[-1] if rest else ""
                if alias is not None or name in ("*", "") or len(rest) != 1:
                    keep.append((segs, alias))
                    problems.append("`%s`%s" % ("::".join(segs), " as %s" % alias if alias else ""))
                    continue
                if crate == "num_bigint" and name == "BigInt":
                    needs.append(INTEGER)
                    cats.append("numbers.bigint")
                elif crate == "num_rational" and name in ("Ratio", "BigRational"):
                    if name == "Ratio" and _OTHER_RATIO.search(view.unit.text):
                        keep.append((segs, alias))
                        problems.append("`Ratio` is also used with another integer type in this file")
                    else:
                        needs.append(RATIONAL)
                        cats.append("numbers.rational")
                elif crate == "num_traits" and name in DROP_TRAITS:
                    cats.append("numbers.trait-imports")
                elif crate == "num_integer" and name == "Integer":
                    # the trait: its name would collide with the new type, its methods are inherent
                    cats.append("numbers.trait-imports")
                elif crate == "num_complex" and name == "Complex64":
                    needs.append(COMPLEX)
                    cats.append("numbers.complex-import")
                else:
                    keep.append((segs, alias))
                    problems.append("`%s`" % "::".join(segs))
            pos = lx.start[use_i]
            if not cats:
                if problems:
                    rid = "numbers.by-hand"
                    yield _mk(rid, view, pos, None,
                              "imports %s: no 1.0 counterpart is named in D's table (D §3: BigUint, Sign, "
                              "Pow, the free functions of num-integer, ..)" % ", ".join(problems))
                continue
            rid = [r for r in self.ids if r in cats][0]
            edit = use_item_edit(view, use_i, semi, keep)
            if edit is None:
                yield _mk(rid, view, pos, None, "a `pub use` that would become several lines")
                continue
            for p in needs:
                view.use_name(use_i, p, ignore=_dropped)
            # the new import must not land on the line that is being removed: register it
            # for the scope; the engine places it after the last remaining `use`
            yield _mk(rid, view, pos, [edit], "", cls="import")
            if problems:
                yield _mk("numbers.by-hand", view, pos, None,
                          "the import keeps %s: no 1.0 counterpart is named in D's table" % ", ".join(problems))


def _mk(rule_id, view, pos, edits, reason, cls="", name=""):
    from .engine import Site, REWRITE, MANUAL
    if edits is None:
        return Site(rule_id, view, pos, MANUAL, reason, (), cls, name)
    return Site(rule_id, view, pos, REWRITE, "", edits, cls, name)


class TypeNames(Rule):
    """The type names in code (outside `use` items)."""

    library_code = True
    trigger = re.compile(r"\b(?:BigInt|BigRational|Ratio|Q)\b")
    ids = ("numbers.bigint", "numbers.rational")

    def __init__(self, rule_id, description):
        self.id = rule_id
        self.description = description

    def find(self, view, cx):
        cached = getattr(view, "_num_type_sites", None)
        if cached is None:
            cached = view._num_type_sites = list(self._all(view))
        return [s for s in cached if s.rule == self.id]

    def _all(self, view):
        lx = view.lx
        unit = view.unit
        in_use = set()
        for use_i, semi, _leaves in use_items(lx):
            in_use.update(range(use_i, semi + 1))
        other_ratio = None
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or i in in_use:
                continue
            t = lx.text[i]
            if t not in ("BigInt", "BigRational", "Ratio", "Q") or lx.is_p(i - 1, "."):
                continue
            pos = lx.start[i]
            if lx.is_id(i - 1, "struct") or lx.is_id(i - 1, "enum") or lx.is_id(i - 1, "type") \
                    or lx.is_id(i - 1, "fn") or lx.is_id(i - 1, "mod"):
                continue
            first = i
            if lx.is_p(i - 1, "::"):
                segs = path_before(lx, i) or []
                crate, _rest = _crate_of(segs + [t])
                if crate is None:
                    continue  # `other::BigInt`: someone else's
                first = path_start(lx, i)
            if t == "BigInt":
                if view._local_def("BigInt"):
                    continue
                # inside `Ratio<BigInt>` the rational rule takes the whole type
                if lx.is_p(first - 1, "<") and lx.is_id(first - 2, "Ratio") and lx.is_p(i + 1, ">"):
                    continue
                yield _mk("numbers.bigint", view, pos,
                          [(lx.start[first], lx.end[i], view.use_name(i, INTEGER, ignore=_dropped))], "", name=t)
                continue
            if t == "BigRational":
                if view._local_def("BigRational"):
                    continue
                last, tail = self._from_integer(lx, i)
                yield _mk("numbers.rational", view, pos,
                          [(lx.start[first], lx.end[i], view.use_name(i, RATIONAL, ignore=_dropped))] + tail, "", name=t)
                continue
            if t == "Ratio":
                if view._local_def("Ratio"):
                    continue
                end = i
                generic = False
                if lx.is_p(i + 1, "::") and lx.is_p(i + 2, "<"):
                    g = lx.turbofish_end(i + 2)
                    inner = lx.text[i + 3:g] if g > 0 else None
                    generic = True
                    end = g
                elif lx.is_p(i + 1, "<"):
                    g = lx.turbofish_end(i + 1)
                    inner = lx.text[i + 2:g] if g > 0 else None
                    generic = True
                    end = g
                if generic:
                    if inner is None or inner[-1:] not in (["BigInt"], ["Integer"]) or \
                            any(x not in ("BigInt", "Integer", "num_bigint", "symplex", "num", "::") for x in inner):
                        yield _mk("numbers.rational", view, pos, None,
                                  "`Ratio` over another integer type: not `Rational`", name=t)
                        continue
                else:
                    if other_ratio is None:
                        other_ratio = _OTHER_RATIO.search(unit.text) is not None
                    if other_ratio:
                        yield _mk("numbers.rational", view, pos, None,
                                  "this file also uses `Ratio` over another integer type: which one is meant here?",
                                  name=t)
                        continue
                last, tail = self._from_integer(lx, end)
                yield _mk("numbers.rational", view, pos,
                          [(lx.start[first], lx.end[end], view.use_name(i, RATIONAL, ignore=_dropped))] + tail, "", name=t)
                continue
            # Q
            verdict = _q_alias(view)
            if verdict == "local":
                continue
            if verdict == "alias":
                last, tail = self._from_integer(lx, i)
                if tail:
                    yield _mk("numbers.rational", view, pos, tail, "", cls="alias", name=t)
                continue
            if lx.is_p(i + 1, ":") and not lx.is_p(i + 1, "::"):
                continue  # a field or a binding named Q
            if verdict != "yes":
                yield _mk("numbers.rational", view, pos, None,
                          "`Q` is not imported from symplex in this file: is it the rational alias?", name=t)
                continue
            last, tail = self._from_integer(lx, i)
            yield _mk("numbers.rational", view, pos,
                      [(pos, lx.end[i], view.use_name(i, RATIONAL, ignore=_dropped))] + tail, "", name=t)

    @staticmethod
    def _from_integer(lx, i):
        """`X::from_integer(` after the type that ends at token i: the edit that makes it `::from(`."""
        if lx.is_p(i + 1, "::") and lx.is_id(i + 2, "from_integer") and lx.is_open(i + 3, "("):
            return i + 2, [(lx.start[i + 2], lx.end[i + 2], "from")]
        return i, []


class RatioNew(Rule):
    id = "numbers.ratio-new"
    description = ("`Ratio::new(a, b)`, `Q::new`, `BigRational::new`, `Rational::new`: listed for a person (the "
                   "constructor returns `Result` at 1.0: add `?` / `.unwrap()`, or write `q(a, b)` for literals)")
    library_code = False
    trigger = re.compile(r"\b(?:Ratio|Q|BigRational|Rational)\s*(?:::\s*<[^>]*>\s*)?::\s*new(?:_raw)?\s*\(")

    def find(self, view, cx):
        lx = view.lx
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or lx.text[i] not in ("new", "new_raw") or not lx.is_p(i - 1, "::"):
                continue
            segs = path_before(lx, i)
            if not segs or segs[-1] not in ("Ratio", "Q", "BigRational", "Rational"):
                continue
            c = call_at(lx, i)
            if c is None:
                continue
            if segs[-1] == "Q" and _q_alias(view) not in ("yes", "alias"):
                continue
            n = c.close_i + 1
            if lx.is_p(n, "?") or (lx.is_p(n, ".") and lx.is_id(n + 1) and lx.text[n + 1] in ("unwrap", "expect")):
                continue  # already reads the Result
            if lx.text[i] == "new_raw":
                yield self.manual(view, lx.start[i], "`new_raw` has no public counterpart on `Rational` (D §2.4)",
                                  name="new_raw")
            else:
                yield self.manual(view, lx.start[i], "`Rational::new` returns `Result` at 1.0 (zero denominator): "
                                                     "`?`, `.unwrap()`, or `q(a, b)` for literals",
                                  name="new")  # one name before and after the type is renamed


class ByHand(Rule):
    id = "numbers.by-hand"
    description = ("what D lists for a person: `BigInt::parse_bytes`, `.to_integer()` on a ratio, `.modpow`, "
                   "`.extended_gcd`, `.magnitude`, `.to_biguint`, `Sign::`, `BigUint`, `as_ratio_parts`, and imports "
                   "from the num crates that have no counterpart")
    library_code = False
    METHODS = {
        "parse_bytes": "`parse_bytes` -> `\"..\".parse()` (D §3)",
        "to_integer": "`.to_integer()` returns `Option` at 1.0 (D §3) — if the receiver is a ratio",
        "modpow": "not offered by `Integer` under this name (D §3)",
        "extended_gcd": "returns the own `ExtendedGcd { gcd, x, y }` (D §2.4, Part 4 C-9)",
        "magnitude": "`BigUint` is not public at 1.0 (D §3)",
        "to_biguint": "`BigUint` is not public at 1.0 (D §3)",
        "as_ratio_parts": "removed (D-R2x)",
    }
    trigger = re.compile(r"\b(?:parse_bytes|to_integer|modpow|extended_gcd|magnitude|to_biguint|as_ratio_parts|"
                         r"BigUint|Sign\s*::)")

    def find(self, view, cx):
        lx = view.lx
        numeric = None
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID:
                continue
            t = lx.text[i]
            if t in self.METHODS and call_at(lx, i) is not None and (lx.is_p(i - 1, ".") or lx.is_p(i - 1, "::")):
                if numeric is None:
                    numeric = re.search(r"\b(?:BigInt|Ratio|BigRational|num_bigint|num_rational|num_integer|Integer|"
                                        r"Rational|Q)\b", view.unit.text) is not None
                if numeric:
                    yield self.manual(view, lx.start[i], self.METHODS[t], name=t)
            elif t == "BigUint" and not lx.is_p(i - 1, ".") and not _in_use(lx, i):
                yield self.manual(view, lx.start[i], "`BigUint` is not public at 1.0 (D §3)", name=t)
            elif t == "Sign" and lx.is_p(i + 1, "::") and lx.is_id(i + 2) and lx.text[i + 2] in (
                    "Plus", "Minus", "NoSign"):
                yield self.manual(view, lx.start[i], "`num_bigint::Sign` is not public at 1.0 (D §3)", name=t)
        # imports from the num crates that D's table does not map
        for s in Imports("numbers.by-hand", "").find(view, cx):
            yield s


def _in_use(lx, i):
    j = i
    n = 0
    while j >= 0 and n < 80:
        n += 1
        if lx.is_p(j, ";"):
            return False
        if lx.is_id(j, "use"):
            return True
        j -= 1
    return False


class Renames(Rule):
    id = "numbers.renames"
    description = ("D-R2x: `ctx.from_bigint(` / `.from_i128(` / `.from_u64(` -> `.from_integer(`, `.from_ratio(` -> "
                   "`.from_rational(`, `.as_bigint(` -> `.as_integer(` (method calls only: `Matrix::from_bigint` "
                   "is another function)")
    library_code = False
    MAP = {"from_bigint": "from_integer", "from_i128": "from_integer", "from_u64": "from_integer",
           "from_ratio": "from_rational", "as_bigint": "as_integer"}
    trigger = re.compile(r"\b(?:from_bigint|from_i128|from_u64|from_ratio|as_bigint)\b")

    def find(self, view, cx):
        lx = view.lx
        for i in range(len(lx.kind)):
            if lx.kind[i] != ID or lx.text[i] not in self.MAP:
                continue
            c = call_at(lx, i)
            if c is None:
                continue
            t = lx.text[i]
            if c.style == "method":
                yield self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], self.MAP[t])], name=t)
            elif c.style == "path":
                segs = path_before(lx, i) or []
                if segs and segs[-1] == "Context":
                    yield self.rewrite(view, lx.start[i], [(lx.start[i], lx.end[i], self.MAP[t])], name=t)
                elif segs and segs[-1] in ("Matrix", "QMatrix", "ZMatrix"):
                    yield self.manual(view, lx.start[i], "`%s::%s`: becomes a `QMatrix` / `ZMatrix` conversion "
                                                         "(F M-DUP)" % (segs[-1], t), name=t)


_D1 = ("D item 1: `BigInt` -> `Integer` (`BigInt::from`, `one`, `zero` keep their shape); `use num_bigint::BigInt;` "
       "-> `use symplex::num::Integer;`")
_D2 = ("D item 2: `Ratio<BigInt>`, `BigRational`, `Q` -> `Rational`; `X::from_integer(n)` -> `Rational::from(n)`; the "
       "imports of `Ratio`, `BigRational`, `Q` -> `use symplex::num::Rational;`")
_D3 = "D item 3: `use num_traits::{Zero, One, Signed, ToPrimitive};` and the trait `num_integer::Integer` -> removed"
_D4 = "D item 4: `use num_complex::Complex64;` -> `use symplex::num::Complex64;`"


class _Both(Rule):
    """A rule id that is fed by the import scan and by the type-name scan."""

    library_code = True

    def __init__(self, rule_id, description, parts):
        self.id = rule_id
        self.description = description
        self.parts = parts
        self.trigger = re.compile("|".join("(?:%s)" % p.trigger.pattern for p in parts))

    def find(self, view, cx):
        out = []
        for p in self.parts:
            out.extend(p.find(view, cx))
        return out


RULES = [
    _Both("numbers.bigint", _D1, [Imports("numbers.bigint", ""), TypeNames("numbers.bigint", "")]),
    _Both("numbers.rational", _D2, [Imports("numbers.rational", ""), TypeNames("numbers.rational", "")]),
    Imports("numbers.trait-imports", _D3),
    Imports("numbers.complex-import", _D4),
    RatioNew(),
    ByHand(),
    Renames(),
]

RULESET = RuleSet(
    "numbers",
    "the number types: BigInt -> Integer, Ratio<BigInt> / BigRational / Q -> Rational, imports (review D §3)",
    RULES,
    needs=("nothing from a table: the type names are fixed.  `symplex::num::{Integer, Rational, Complex64}` must "
           "exist (package N1).  With --src-code the same rules run on the library's own code (N2); there the "
           "compiler is the list for the four thought-points of D §3."),
    package="N2 (src, with --src-code), N3 (user side)",
)
