# The engine: rules, sites, the transformation of one file to a fixpoint, the tree walk.
#
# A rule looks at one view and yields Sites.  A Site is either REWRITE (it carries
# edits) or MANUAL (it carries the reason why the tool will not touch it).  The engine
# applies the edits of the accepted sites, and runs the rule set again on the result
# until nothing changes: so the output of the tool is a fixpoint of the tool, and
# applying a rule set twice gives what applying it once gives.
import csv
import difflib
import os
import re

from .views import FileUnit

# REWRITE: the tool edits the site.  MANUAL: found, and a person must change it (the reason says
# why the tool does not).  INFO: found, and nothing has to be edited for it to compile (a printed
# value, a mention in prose): counted, so that the totals can be compared with the reviewers'.
REWRITE, MANUAL, INFO = "rewrite", "manual", "info"
MAX_PASSES = 6
SKIP_MARK = "codemod:skip"

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.normpath(os.path.join(HERE, "..", "tables"))


class Site(object):
    __slots__ = ("rule", "kind", "line", "status", "reason", "edits", "imports", "snippet", "cls", "name",
                 "later_pass", "rel", "_view")

    def __init__(self, rule, view, pos, status, reason="", edits=(), cls="", name=""):
        self.rule = rule
        self.kind = view.kind
        self.line = view.real_line(pos)
        self.status = status
        self.reason = reason
        # edits in REAL file coordinates
        self.edits = [view.map_edit(a, b, t) for (a, b, t) in edits]
        self.imports = view.take_pending_imports() if status == REWRITE else {}
        if status != REWRITE:
            view.take_pending_imports()
        self.snippet = view.snippet(pos)
        self.cls = cls  # a finer class inside the rule (the reviewers' "reading")
        self.name = name  # the method / item name the rule keyed on
        self.later_pass = False
        self.rel = view.unit.rel
        self._view = None

    def key(self):
        # what a site IS, without where it is or what its line looks like: a rewrite of
        # another site on the same line must not make this one look new in a later pass
        return (self.rule, self.reason, self.cls, self.name)


class Rule(object):
    """Base class.  A rule sets `id`, `description`, `targets` (view kinds) and `trigger`
    (a regular expression that must match the view's text for the rule to look at it at
    all: a cheap prefilter, never the match itself)."""

    id = ""
    description = ""
    targets = ("code", "doc", "md")
    trigger = None
    # user-side rules do not look at the library's own code unless told to
    library_code = False

    def find(self, view, cx):  # pragma: no cover - interface
        return ()

    def find_text(self, unit, cx):
        """Sites found in the raw text of a file (comments, prose); edits are given in
        real coordinates through `text_site`."""
        return ()

    def rewrite(self, view, pos, edits, cls="", name=""):
        return Site(self.id, view, pos, REWRITE, "", edits, cls, name)

    def manual(self, view, pos, reason, cls="", name=""):
        return Site(self.id, view, pos, MANUAL, reason, (), cls, name)

    def info(self, view, pos, reason, cls="", name=""):
        return Site(self.id, view, pos, INFO, reason, (), cls, name)


class _TextView(object):
    """Adapter that lets a rule build a Site on raw file text (identity mapping)."""

    kind = "text"

    def __init__(self, unit):
        self.unit = unit
        self.text = unit.text

    def real_line(self, pos):
        return self.unit.line_of(pos)

    def map_edit(self, a, b, t):
        return (a, b, t)

    def take_pending_imports(self):
        return {}

    def snippet(self, v, width=110):
        s = self.text
        ls = s.rfind("\n", 0, v) + 1
        le = s.find("\n", v)
        if le < 0:
            le = len(s)
        line = s[ls:le].strip()
        return line if len(line) <= width else line[: width - 1] + "…"


class RuleSet(object):
    def __init__(self, name, description, rules, needs="", package=""):
        self.name = name
        self.description = description
        self.rules = rules
        self.needs = needs  # what the rule set needs as input when its package runs
        self.package = package


class Cx(object):
    """What a rule may ask about its surroundings."""

    def __init__(self, options=None, tables_dir=TABLES):
        self.options = options or {}
        self.tables_dir = tables_dir
        self._tables = {}
        self.unit = None

    def opt(self, key, default=None):
        return self.options.get(key, default)

    def table(self, name):
        """Rows of tables/<name>.tsv as dicts (lines starting with # are comments)."""
        if name not in self._tables:
            path = os.path.join(self.tables_dir, name + ".tsv")
            with open(path, newline="") as f:
                lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
            self._tables[name] = list(csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE))
        return self._tables[name]

    @property
    def corpus(self):
        return self.unit.corpus if self.unit is not None else ""

    def unwrap_or_question(self, view, tok_i):
        """How a call that now returns `Result<_, SymplexError>` is unwrapped at token tok_i:
        `?` when the enclosing function returns a `Result` whose error can hold a `SymplexError`
        and no closure lies in between; `.unwrap()` everywhere else (it is always type-correct)."""
        from .chain import enclosing_fn, fn_returns_result
        f = enclosing_fn(view.lx, tok_i)
        if f is None or f[2]:
            return ".unwrap()"
        ty = fn_returns_result(view.lx, f[0], f[1]).replace(" ", "")
        if not ty:
            return ".unwrap()"
        if "SymplexError" in ty or "Box<dyn" in ty or ty.startswith("symplex::Result<") or ty.startswith("anyhow::"):
            return "?"
        return ".unwrap()"


# ------------------------------------------------------------------ one file
def kinds_for(unit, src_code=False, only=None):
    if unit.is_md:
        kinds = ["md"]
    elif unit.corpus == "src":
        kinds = ["code", "doc"] if src_code else ["doc"]
    else:
        kinds = ["code", "doc"]
    if only:
        kinds = [k for k in kinds if k in only]
    return kinds


def _one_pass(unit, ruleset, cx, kinds):
    cx.unit = unit
    sites = []
    unlexed = []
    library = unit.corpus == "src"
    for view in unit.views(kinds):
        for rule in ruleset.rules:
            if view.kind not in rule.targets:
                continue
            if library and view.kind == "code" and not rule.library_code:
                continue
            if rule.trigger is not None and not rule.trigger.search(view.text):
                continue
            if not view.lx.ok:
                unlexed.append((view.kind, view.line0, view.lx.error))
                break
            view.take_pending_imports()
            for s in rule.find(view, cx):
                if s.status == REWRITE and view.compile_fail:
                    s.status = MANUAL
                    s.reason = "inside a `compile_fail` doctest: the example is meant not to compile; edit by hand"
                    s.edits = []
                    s.imports = {}
                s._view = view
                sites.append(s)
    tv = None
    for rule in ruleset.rules:
        if "text" not in rule.targets:
            continue
        if rule.trigger is not None and not rule.trigger.search(unit.text):
            continue
        if tv is None:
            tv = _TextView(unit)
        for s in rule.find_text(tv, cx):
            s._view = None
            sites.append(s)
    return sites, unlexed


def _overlaps(accepted, a, b):
    for (x, y) in accepted:
        if a == b:
            if x < a < y:
                return True
        elif x == y:
            if a < x < b:
                return True
        elif a < y and x < b:
            return True
    return False


def transform(text, path, rel, corpus, ruleset, cx, src_code=False, only=None, frozen=""):
    """Run `ruleset` on one file's text to a fixpoint.

    Returns (new_text, rewritten_sites, manual_sites, notes).  `frozen` is a reason for which
    nothing in this file may be rewritten (every REWRITE site becomes MANUAL with it)."""
    orig = text
    rewritten = []
    manual = []
    notes = []
    first_manual_keys = None
    for n in range(MAX_PASSES + 1):
        unit = FileUnit(path, rel, text, corpus)
        kinds = kinds_for(unit, src_code, only)
        sites, unlexed = _one_pass(unit, ruleset, cx, kinds)
        if n == 0:
            for (kind, line, err) in unlexed:
                notes.append("%s:%d: %s block not scanned: it does not tokenize (%s)" % (rel, line, kind, err))
        todo = []
        this_manual = []
        lines = None
        for s in sites:
            if s.status == REWRITE and frozen:
                s.status, s.reason, s.edits = MANUAL, frozen, []
            if s.status == REWRITE:
                if lines is None:
                    lines = text.split("\n")
                ln = s.line
                near = lines[ln - 1] if 0 < ln <= len(lines) else ""
                above = lines[ln - 2] if ln >= 2 else ""
                if SKIP_MARK in near or (SKIP_MARK in above and above.lstrip().startswith("//")):
                    s.status, s.reason, s.edits = MANUAL, "marked `%s`" % SKIP_MARK, []
            if s.status == REWRITE and (s.edits or s.imports):
                todo.append(s)
            elif s.status == REWRITE:
                s.status, s.reason = MANUAL, "internal: a rewrite without edits"
                this_manual.append(s)
            else:
                this_manual.append(s)
        if n == 0:
            manual = this_manual
            first_manual_keys = {}
            for s in manual:
                first_manual_keys.setdefault(s.key(), []).append(s)
        elif not todo:
            # the fixpoint.  What is left for a person is what the LAST pass lists; a site that
            # was listed at the start and has been rewritten since (a rename exposed its reading)
            # is no longer one.  A site that was there from the start keeps the line number it
            # has in the untouched file; one that a rewrite exposed is flagged.
            # (Sites that need no edit are counted as they stood in the untouched file.)
            final = [s for s in manual if s.status == INFO]
            for s in this_manual:
                if s.status == INFO:
                    continue
                olds = first_manual_keys.get(s.key())
                if olds:
                    final.append(olds.pop(0))
                else:
                    s.later_pass = True
                    final.append(s)
            manual = final
        if not todo:
            break
        if n == MAX_PASSES:
            notes.append("%s: no fixpoint after %d passes; the last pass's rewrites were not applied" % (rel, n))
            break
        todo.sort(key=lambda s: min([e[0] for e in s.edits] or [0]))
        accepted = []
        edits = []
        pending = {}
        for s in todo:
            if any(_overlaps(accepted, a, b) for (a, b, _t) in s.edits):
                continue  # deferred: found again by the next pass, in the rewritten text
            for (a, b, t) in s.edits:
                accepted.append((a, b))
                edits.append((a, b, t))
            s.later_pass = n > 0
            rewritten.append(s)
            v = getattr(s, "_view", None)
            if v is not None and s.imports:
                bucket = pending.setdefault(id(v), (v, {}))[1]
                for scope, paths in s.imports.items():
                    bucket.setdefault(scope, set()).update(paths)
        for (v, per_scope) in pending.values():
            for (a, b, t) in v.import_edits(per_scope):
                ra, rb, rt = v.map_edit(a, b, t)
                if not _overlaps(accepted, ra, rb):
                    edits.append((ra, rb, rt))
        # apply back to front; among edits at the same offset, keep their original order
        order = sorted(range(len(edits)), key=lambda i: (edits[i][0], edits[i][1], i))
        out = []
        pos = 0
        for i in order:
            a, b, t = edits[i]
            if a < pos:
                continue  # cannot happen after the overlap test; never write a corrupt file
            out.append(text[pos:a])
            out.append(t)
            pos = b
        out.append(text[pos:])
        new_text = "".join(out)
        if new_text == text:
            break
        text = new_text
    for s in rewritten + manual:
        s._view = None
    return text, rewritten, manual, notes


# ------------------------------------------------------------------ the tree
SKIP_DIRS = frozenset(["target", ".git", "node_modules", "pkg", "corpus", "artifacts", "book_out", ".venv",
                       "results", "sandbox_out"])

# root (relative to the repository) -> corpus name, as the reviewers named them
DEFAULT_ROOTS = [
    ("tests", "tests"),
    ("examples", "examples"),
    ("benches", "benches"),
    ("book/src", "book"),
    ("README.md", "readme"),
    ("src", "src"),
    ("fuzz", "fuzz"),
    ("symplex-wasm", "wasm"),
    ("symplex-build", "build"),
    ("symplex-macros", "macros"),
    ("rubi-harness", "rubi"),
    ("probes", "probes"),
]


def corpus_of(rel):
    rel = rel.replace(os.sep, "/")
    best = ("", "other")
    for root, name in DEFAULT_ROOTS:
        if rel == root or rel.startswith(root.rstrip("/") + "/"):
            if len(root) > len(best[0]):
                best = (root, name)
    return best[1]


def walk(repo, roots):
    """Yield (absolute path, path relative to repo) of every .rs and .md file under roots."""
    for root in roots:
        base = root if os.path.isabs(root) else os.path.join(repo, root)
        if os.path.isfile(base):
            yield base, os.path.relpath(base, repo)
            continue
        for dp, dn, fn in os.walk(base):
            dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
            for f in sorted(fn):
                if f.endswith(".rs") or f.endswith(".md"):
                    p = os.path.join(dp, f)
                    yield p, os.path.relpath(p, repo)


# pages that describe the API of an earlier release: found, listed, never rewritten
HISTORY = re.compile(r"(?:^|/)migrating-[0-9][^/]*\.md$|(?:^|/)CHANGELOG\.md$")


class FileResult(object):
    __slots__ = ("rel", "corpus", "old", "new", "rewritten", "manual", "notes")

    def __init__(self, rel, corpus, old, new, rewritten, manual, notes):
        self.rel = rel
        self.corpus = corpus
        self.old = old
        self.new = new
        self.rewritten = rewritten
        self.manual = manual
        self.notes = notes


def run(repo, roots, ruleset, cx, src_code=False, only=None, corpus_override=None):
    """Transform every file under `roots` in memory.  Nothing is written."""
    prefilter = None
    pats = [r.trigger.pattern for r in ruleset.rules if r.trigger is not None]
    if pats and len(pats) == len(ruleset.rules):
        prefilter = re.compile("|".join("(?:%s)" % p for p in pats))
    results = []
    for path, rel in walk(repo, roots):
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError) as e:
            results.append(FileResult(rel, "", "", "", [], [], ["%s: not read (%s)" % (rel, e)]))
            continue
        if prefilter is not None and not prefilter.search(text):
            continue
        corpus = corpus_override or corpus_of(rel)
        frozen = ""
        if path.endswith(".rs") and os.path.isfile(path[:-3] + ".stderr"):
            frozen = "UI test with a recorded compiler output (.stderr): edit by hand and re-bless"
        elif HISTORY.search(rel.replace(os.sep, "/")):
            frozen = "a migration guide of an earlier release: it shows the API of that time on purpose"
        new, rewritten, manual, notes = transform(text, path, rel, corpus, ruleset, cx, src_code, only, frozen)
        if rewritten or manual or notes:
            results.append(FileResult(rel, corpus, text, new, rewritten, manual, notes))
    return results


def unified_diff(results):
    out = []
    for r in results:
        if r.new != r.old:
            out.extend(difflib.unified_diff(r.old.splitlines(True), r.new.splitlines(True),
                                            "a/" + r.rel, "b/" + r.rel, n=2))
    return "".join(out)
