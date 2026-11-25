# Views: the pieces of a file that hold Rust code.
#
#   kind "code"  a .rs file's own tokens (comments and strings are never touched)
#   kind "doc"   one fenced Rust block inside `///` / `//!` doc comments (a doctest)
#   kind "md"    one fenced `rust` block of a Markdown file
#
# A doc or md view has a *virtual* text: the code lines with the comment marker (and the
# hidden-line marker `# `) taken away, so that the tokenizer sees plain Rust.  Offsets
# in the virtual text map back to the file; an edit computed on the virtual text is
# applied to the file, with the comment marker re-inserted after every newline that
# the replacement contains.
import bisect
import re

from .lexer import Lexed, ID, PUNCT, OPEN

_RUSTDOC_TAGS = re.compile(
    r"^(rust|ignore|ignore-.*|should_panic|no_run|compile_fail|edition20\d\d|standalone|standalone_crate|"
    r"test_harness|E\d{4}|\{.*\})$"
)
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([^`]*)$")
_HIDDEN = re.compile(r"^(\s*)#(?= |$)")

# names that `use symplex::prelude::*` brings into scope at 1.0 (API_1.0.md §1.3)
PRELUDE_1_0 = frozenset(
    "Context Ex BoolEx SetEx Expr Symbol Equation Truth SymplexError Assumption Matrix".split()
)


class View(object):
    def __init__(self, unit, kind, text, base=0, segs=None, prefix="", line0=1, info=""):
        self.unit = unit  # the FileUnit
        self.kind = kind
        self.text = text
        self.base = base  # code/md: real offset of virtual offset 0
        self.segs = segs  # doc: [(virt_start, real_start, length)], one per line
        self._vs = [s[0] for s in segs] if segs else None
        self.prefix = prefix  # doc: what to put after a newline inside a replacement
        self.line0 = line0  # real line number of the first virtual line
        self.info = info  # the fence's info string
        self.compile_fail = "compile_fail" in info
        self.hidden_uses = False
        self._lx = None
        self._uses = None
        self._pending_imports = {}

    @property
    def lx(self):
        if self._lx is None:
            self._lx = Lexed(self.text)
        return self._lx

    # ------------------------------------------------------------- positions
    def real(self, v):
        """Real file offset of virtual offset v."""
        if self.segs is None:
            return self.base + v
        k = bisect.bisect_right(self._vs, v) - 1
        if k < 0:
            k = 0
        vs, rs, ln = self.segs[k][:3]
        return rs + min(v - vs, ln)

    def real_line(self, v):
        return self.unit.line_of(self.real(v))

    def map_edit(self, vs, ve, repl):
        rs, re_ = self.real(vs), self.real(ve)
        if self.segs is not None and not repl and ve > vs:
            # whole lines removed: take each line WITH its own comment marker (the markers of
            # neighbouring lines may differ: `/// ` and a bare `///`)
            k1 = bisect.bisect_left(self._vs, vs)
            k2 = bisect.bisect_left(self._vs, ve)
            if k1 < len(self.segs) and self._vs[k1] == vs and (k2 == len(self.segs) or self._vs[k2] == ve):
                total = self.segs[-1][0] + self.segs[-1][2]
                if k2 < len(self.segs) or ve == total:
                    rs = self.segs[k1][1] - len(self.segs[k1][3])
                    if k2 < len(self.segs):
                        re_ = self.segs[k2][1] - len(self.segs[k2][3])
                    else:
                        re_ = self.segs[-1][1] + self.segs[-1][2]
            return rs, re_, repl
        if self.segs is not None and repl:
            k = max(bisect.bisect_right(self._vs, vs) - 1, 0)
            line_vs, _rs, _ln, own = self.segs[k]
            at_line_start = vs == line_vs
            canon = self.prefix
            if "\n" in repl:
                body = repl.rstrip("\n")
                tail = repl[len(body):]
                # every newline inside the replacement is followed by the comment marker; after a
                # final newline comes the marker the following text had (it may be a bare `///`)
                after = own if (at_line_start and vs == ve) else canon
                repl = body.replace("\n", "\n" + canon) + tail.replace("\n", "\n" + after)
            if at_line_start and vs == ve and canon.startswith(own) and len(canon) > len(own):
                # inserted in front of a line whose marker has no space after it (`///`)
                repl = canon[len(own):] + repl
        return rs, re_, repl

    def snippet(self, v, width=110):
        s = self.text
        ls = s.rfind("\n", 0, v) + 1
        le = s.find("\n", v)
        if le < 0:
            le = len(s)
        line = s[ls:le].strip()
        return line if len(line) <= width else line[: width - 1] + "…"

    # ---------------------------------------------------------------- imports
    def _use_table(self):
        """{group open index (-1 = top level): [(segments, alias, use_token, semicolon_token)]}"""
        if self._uses is not None:
            return self._uses
        lx = self.lx
        table = {}
        for i in range(len(lx.kind)):
            if lx.kind[i] == ID and lx.text[i] == "use" and _item_start(lx, i):
                leaves, semi = parse_use(lx, i)
                if leaves is None:
                    continue
                for segs, alias, _a, _b in leaves:
                    table.setdefault(lx.up[i], []).append((segs, alias, i, semi))
        self._uses = table
        return table

    def _local_def(self, name):
        lx = self.lx
        for i in range(len(lx.kind) - 1):
            if lx.kind[i] == ID and lx.text[i] in ("struct", "enum", "type", "trait", "mod", "union"):
                if lx.kind[i + 1] == ID and lx.text[i + 1] == name:
                    return True
        return False

    def use_name(self, tok_i, path, ignore=None):
        """The spelling under which the item `path` (e.g. "symplex::errors::ErrorKind") can be
        written at token tok_i, registering the `use` line the engine must add for it.

        Returns the last segment when the name is, or can be made, visible there; the full
        path when another item of that name is already in scope (no import is added then).
        `ignore(segments, alias)` names `use` leaves that the calling rule removes in the same
        pass: they neither provide the name nor clash with it.
        """
        target = path.split("::")
        name = target[-1]
        table = self._use_table()
        lx = self.lx
        if self.kind != "code":
            leaves = [u for g in table.values() for u in g]
            verdict = _visible(leaves, target, ignore)
            if verdict == "yes":
                return name
            if verdict == "clash" or self._local_def(name):
                return path
            if self.kind == "md" and not leaves:
                # a fragment without any `use` line: it is not a complete program, and
                # an import would be the only one in it
                return name
            self._pending_imports.setdefault("block", set()).add(path)
            return name
        g = lx.up[tok_i]
        scope = -1
        while True:
            verdict = _visible(table.get(g, ()), target, ignore)
            if verdict == "yes":
                return name
            if verdict == "clash":
                return path
            if g == -1:
                break
            if _is_mod_body(lx, g):
                if not any(u[0] == ["super", "*"] for u in table.get(g, ())):
                    scope = g
                    break
            g = lx.up[g]
        if self._local_def(name):
            return path
        self._pending_imports.setdefault(scope, set()).add(path)
        return name

    def take_pending_imports(self):
        p = self._pending_imports
        self._pending_imports = {}
        return p

    def import_edits(self, pending):
        """Edits (virtual coordinates) that add the `use` lines in `pending`
        ({scope: set(paths)}), grouped by parent module."""
        edits = []
        lx = self.lx
        table = self._use_table()
        for scope, paths in sorted(pending.items(), key=lambda kv: str(kv[0])):
            lines = _group_imports(paths)
            if self.kind != "code":
                leaves = [u for g in table.values() for u in g]
                if leaves:
                    last = max(u[3] for u in leaves)
                    pos = _next_line_start(self.text, lx.end[last])
                    indent = lx.indent_at(lx.start[max(u[2] for u in leaves)])
                    hidden = self.hidden_uses
                else:
                    # a doctest without a `use` line: keep what the reader sees unchanged
                    pos = 0
                    indent = ""
                    hidden = True
                if hidden:
                    # the hidden-line marker `# ` shows as two spaces in the virtual text
                    lead = (indent[:-2] if len(indent) >= 2 else "") + "# "
                else:
                    lead = indent
                edits.append((pos, pos, "".join("%s%s\n" % (lead, ln) for ln in lines)))
                continue
            here = table.get(scope, ())
            if here:
                last_semi = max(u[3] for u in here)
                last_use = max(u[2] for u in here)
                pos = _next_line_start(self.text, lx.end[last_semi])
                indent = lx.indent_at(lx.start[last_use])
                text = "".join("%s%s\n" % (indent, ln) for ln in lines)
            elif scope == -1:
                pos = _file_import_pos(lx)
                text = "".join("%s\n" % ln for ln in lines) + "\n"
            else:
                pos = _next_line_start(self.text, lx.end[scope])
                inner = lx.indent_at(lx.start[scope + 1]) if scope + 1 < len(lx.kind) else "    "
                text = "".join("%s%s\n" % (inner, ln) for ln in lines)
            edits.append((pos, pos, text))
        return edits


def _next_line_start(text, offset):
    nl = text.find("\n", offset)
    return len(text) if nl < 0 else nl + 1


def _group_imports(paths):
    by_parent = {}
    for p in sorted(paths):
        parent, _, name = p.rpartition("::")
        by_parent.setdefault(parent, []).append(name)
    out = []
    for parent in sorted(by_parent):
        names = by_parent[parent]
        if len(names) == 1:
            out.append("use %s::%s;" % (parent, names[0]))
        else:
            out.append("use %s::{%s};" % (parent, ", ".join(names)))
    return out


def _file_import_pos(lx):
    """Offset at which a first `use` line can be put in a file that has none: before the
    first item, above the comments and attributes that belong to it, below `//!` and `#![..]`."""
    i = 0
    n = len(lx.kind)
    while i + 2 < n and lx.is_p(i, "#") and lx.is_p(i + 1, "!") and lx.is_open(i + 2, "["):
        i = lx.mate[i + 2] + 1
    if i >= n:
        return len(lx.src)
    pos = lx.line_start(lx.start[i])
    # climb over comments that sit directly above (they describe the item)
    changed = True
    while changed:
        changed = False
        for c in reversed(lx.comments):
            if c.end <= pos and lx.src[c.end:pos].strip() == "" and lx.src.count("\n", c.end, pos) <= 1:
                if c.doc == "inner":
                    return pos
                ls = lx.line_start(c.start)
                if lx.src[ls:c.start].strip() == "":
                    pos = ls
                    changed = True
                break
            if c.end < pos - 400:
                break
    return pos


def _item_start(lx, i):
    """Is the `use` at token i the start of a `use` item (not `r#use`, not inside a path)?"""
    p = i - 1
    if p < 0:
        return True
    # `use` is a keyword: it starts an item unless it is a path segment of a macro's input
    return not (lx.kind[p] == PUNCT and lx.text[p] in ("::", ".", "$"))


def _is_mod_body(lx, g):
    return lx.is_open(g, "{") and lx.is_id(g - 1) and lx.is_id(g - 2, "mod")


def parse_use(lx, use_i):
    """Parse the `use` item at token use_i.

    Returns (leaves, semicolon_token) where a leaf is (segments, alias, first_tok, last_tok);
    a glob has "*" as its last segment.  (None, -1) when the item is not understood."""
    out = []
    try:
        j = _use_tree(lx, use_i + 1, [], out)
    except (ValueError, IndexError):
        return None, -1
    if not lx.is_p(j, ";"):
        return None, -1
    return out, j


def _use_tree(lx, i, prefix, out):
    segs = list(prefix)
    first = i
    if lx.is_p(i, "::"):
        i += 1
    while True:
        if lx.is_open(i, "{"):
            c = lx.mate[i]
            if c < 0:
                raise ValueError
            j = i + 1
            while j < c:
                j = _use_tree(lx, j, segs, out)
                if lx.is_p(j, ","):
                    j += 1
                elif j < c:
                    raise ValueError
            return c + 1
        if lx.is_p(i, "*"):
            out.append((segs + ["*"], None, first, i))
            return i + 1
        if lx.kind[i] == ID:
            segs.append(lx.text[i])
            last = i
            i += 1
            if lx.is_p(i, "::"):
                i += 1
                continue
            alias = None
            if lx.is_id(i, "as"):
                alias = lx.text[i + 1]
                last = i + 1
                i += 2
            if segs[-1] == "self" and len(segs) > 1:
                segs = segs[:-1]
            out.append((segs, alias, first, last))
            return i
        raise ValueError


def _visible(leaves, target, ignore=None):
    """"yes" when one of the `use` leaves brings `target` into scope under its own name,
    "clash" when a different item is imported under that name, else "no"."""
    name = target[-1]
    verdict = "no"
    for u in leaves:
        segs, alias = u[0], u[1]
        if ignore is not None and ignore(segs, alias):
            continue
        if segs[-1] == "*":
            head = segs[:-1]
            if head == target[:-1]:
                return "yes"
            if head[-1:] == ["prelude"] and head[:1] == target[:1] and name in PRELUDE_1_0 and len(target) == 2:
                return "yes"
            continue
        bound = alias or segs[-1]
        if bound != name:
            continue
        if alias is None and segs == target:
            return "yes"
        verdict = "clash"
    return verdict


# ----------------------------------------------------------------- file units
class FileUnit(object):
    """One file of the tree, with its views."""

    def __init__(self, path, rel, text, corpus):
        self.path = path
        self.rel = rel
        self.text = text
        self.corpus = corpus
        self.is_rust = path.endswith(".rs")
        self.is_md = path.endswith(".md")
        self._nl = None
        self._lx = None
        self._views = None
        self.notes = []

    def line_of(self, offset):
        if self._nl is None:
            self._nl = [m.start() for m in re.finditer("\n", self.text)]
        return bisect.bisect_left(self._nl, offset) + 1

    @property
    def lx(self):
        """Tokens of the whole file (Rust files only)."""
        if self._lx is None:
            self._lx = Lexed(self.text) if self.is_rust else Lexed("")
        return self._lx

    def views(self, kinds):
        if self._views is None:
            self._views = {}
        out = []
        for k in kinds:
            if k not in self._views:
                self._views[k] = self._build(k)
            out.extend(self._views[k])
        return out

    def _build(self, kind):
        if kind == "code":
            if not self.is_rust:
                return []
            v = View(self, "code", self.text)
            v._lx = self.lx
            return [v]
        if kind == "doc":
            return doc_views(self) if self.is_rust else []
        if kind == "md":
            return md_views(self) if self.is_md else []
        return []


def _is_rust_fence(info, markdown):
    tags = [t for t in re.split(r"[,\s]+", info.strip()) if t]
    if markdown:
        return bool(tags) and tags[0] in ("rust", "rs")
    return all(_RUSTDOC_TAGS.match(t) for t in tags)


def _hide(content):
    m = _HIDDEN.match(content)
    if m:
        k = m.end(1)
        return content[:k] + " " + content[k + 1:], True
    return content, False


def doc_views(unit):
    """One view per fenced Rust block in the line doc comments of a Rust file."""
    lx = unit.lx
    src = unit.text
    runs = []
    cur = []
    for c in lx.comments:
        if c.doc and src.startswith("//", c.start):
            marker = src[c.start:c.start + 3]
            if cur and cur[-1][2] == marker and src[cur[-1][1]:c.start].strip() == "" \
                    and src.count("\n", cur[-1][1], c.start) == 1:
                cur.append((c.start, c.end, marker))
                continue
            if cur:
                runs.append(cur)
            cur = [(c.start, c.end, marker)]
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    views = []
    for run in runs:
        fence = None
        lines = []
        info = ""
        for (cs, ce, marker) in run:
            body = cs + 3
            if src.startswith(" ", body):
                body += 1
            if body > ce:
                body = ce
            content = src[body:ce]
            stripped = content.strip()
            if fence is None:
                m = _FENCE.match(content)
                if m:
                    fence = m.group(2)
                    info = m.group(3)
                    lines = []
                continue
            if stripped.startswith(fence[0] * len(fence)) and stripped.strip(fence[0]) == "":
                if _is_rust_fence(info, False) and lines:
                    views.append(_make_doc_view(unit, lines, info))
                fence = None
                continue
            ls = lx.line_start(cs)
            lines.append((body, content, src[ls:body]))
    return views


def _make_doc_view(unit, lines, info):
    parts = []
    segs = []
    v = 0
    hidden_use = 0
    shown_use = 0
    for (body, content, _prefix) in lines:
        shown, was_hidden = _hide(content)
        if re.match(r"\s*use\b", shown):
            if was_hidden:
                hidden_use += 1
            else:
                shown_use += 1
        parts.append(shown + "\n")
        segs.append((v, body, len(content) + 1, _prefix))
        v += len(content) + 1
    prefix = lines[0][2]
    for ln in lines:
        if ln[2].endswith(" "):
            prefix = ln[2]
            break
    view = View(unit, "doc", "".join(parts), segs=segs, prefix=prefix,
                line0=unit.line_of(lines[0][0]), info=info)
    view.hidden_uses = hidden_use > 0 and shown_use == 0
    return view


def md_views(unit):
    """One view per fenced `rust` block of a Markdown file."""
    text = unit.text
    views = []
    pos = 0
    fence = None
    info = ""
    start = 0
    for line in text.splitlines(True):
        content = line.rstrip("\n")
        if fence is None:
            m = _FENCE.match(content)
            if m:
                fence = m.group(2)
                info = m.group(3)
                start = pos + len(line)
        else:
            s = content.strip()
            if s.startswith(fence[0] * len(fence)) and s.strip(fence[0]) == "":
                if _is_rust_fence(info, True) and pos > start:
                    raw = text[start:pos]
                    parts = []
                    hidden_use = shown_use = 0
                    for ln in raw.splitlines(True):
                        shown, was_hidden = _hide(ln.rstrip("\n"))
                        if re.match(r"\s*use\b", shown):
                            if was_hidden:
                                hidden_use += 1
                            else:
                                shown_use += 1
                        parts.append(shown + ("\n" if ln.endswith("\n") else ""))
                    view = View(unit, "md", "".join(parts), base=start, line0=unit.line_of(start), info=info)
                    view.hidden_uses = hidden_use > 0 and shown_use == 0
                    views.append(view)
                fence = None
        pos += len(line)
    return views


# ------------------------------------------------------------- `use` items
def use_items(lx):
    """Every `use` item of the token stream: (use_token, semicolon_token, leaves)."""
    for i in range(len(lx.kind)):
        if lx.kind[i] == ID and lx.text[i] == "use" and _item_start(lx, i):
            leaves, semi = parse_use(lx, i)
            if leaves is not None:
                yield i, semi, leaves


def render_use(leaves, indent=""):
    """`use` lines for leaves [(segments, alias)], one line per parent path, in first-seen order."""
    order = []
    groups = {}
    for segs, alias in leaves:
        parent = tuple(segs[:-1])
        name = segs[-1] + (" as %s" % alias if alias else "")
        if parent not in groups:
            groups[parent] = []
            order.append(parent)
        if name not in groups[parent]:
            groups[parent].append(name)
    lines = []
    for parent in order:
        names = groups[parent]
        head = "::".join(parent)
        if len(names) == 1:
            lines.append("use %s%s;" % (head + "::" if head else "", names[0]))
        else:
            lines.append("use %s::{%s};" % (head, ", ".join(names)))
    return ("\n" + indent).join(lines)


def use_item_edit(view, use_i, semi, new_leaves):
    """The edit (virtual coordinates) that replaces the `use` item by the given leaves, or
    removes its line(s) when no leaf is left.  None when the item carries `pub` and would
    become several lines (by hand)."""
    lx = view.lx
    text = view.text
    start = lx.start[use_i]
    end = lx.end[semi]
    if new_leaves:
        out = render_use(new_leaves, lx.indent_at(start))
        if "\n" in out and lx.is_id(use_i - 1, "pub"):
            return None
        return (start, end, out)
    # remove the item, an attribute that belongs to it, and the line if nothing else is on it
    first = use_i
    if lx.is_id(first - 1, "pub"):
        first -= 1
    while lx.is_close(first - 1, "]") and lx.mate[first - 1] >= 1 and lx.is_p(lx.mate[first - 1] - 1, "#"):
        first = lx.mate[first - 1] - 1
    start = lx.start[first]
    ls = text.rfind("\n", 0, start) + 1
    le = text.find("\n", end)
    le = len(text) if le < 0 else le + 1
    if text[ls:start].strip() == "" and text[end:le].strip() == "":
        return (ls, le, "")
    return (start, end, "")
