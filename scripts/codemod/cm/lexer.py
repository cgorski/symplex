"""A tokenizer for Rust source text, and the bracket structure on top of it.

This is NOT a parser.  It knows exactly enough of Rust's lexical grammar that a
rewrite rule can never fire inside a comment or a string literal, and that an
argument list can be matched to its closing bracket:

* line comments and (nested) block comments, with the doc-comment forms told apart;
* string literals: normal, byte (`b".."`), C (`c".."`), raw with any number of `#`
  (`r".."`, `r#".."#`, `br##".."##`);
* char and byte-char literals versus lifetimes and loop labels (`'a'` / `'a`);
* raw identifiers (`r#type`), numbers (opaque: `1_000u64`, `0x1F`, `1.5e-3`);
* brackets `( [ {` with their partners, and for every token its enclosing bracket.

`<` and `>` are NOT brackets here (they are also comparison operators); the two
places that need generic arguments (`turbofish_end`, `split_args`) handle them
explicitly and give up when unsure.

A file that does not lex (an unterminated literal, unbalanced brackets) gets
`ok = False`, and the engine rewrites nothing in it.
"""
import bisect
import re

ID, NUM, STR, CHAR, LIFE, PUNCT, OPEN, CLOSE = "id", "num", "str", "chr", "life", "p", "open", "close"

_MASTER = re.compile(
    r"""
  (?P<ws>\s+)
 |(?P<lc>//[^\n]*)
 |(?P<bc>/\*)
 |(?P<rstr>(?:br|cr|r)(?P<h>\#*)"[\s\S]*?"(?P=h))
 |(?P<str>[bc]?"(?:[^"\\]|\\[\s\S])*")
 |(?P<chr>b?'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]{1,8}\}|[\s\S])|[^\\'\n])')
 |(?P<life>'(?:r\#)?[^\W\d]\w*)
 |(?P<id>(?:r\#)?[^\W\d]\w*)
 |(?P<num>\d\w*(?:\.\d\w*)?(?:(?<=[eE])[+-]\d\w*)?)
 |(?P<p>\.\.=|\.\.\.|<<=|>>=|::|->|=>|==|!=|<=|>=|&&|\|\||\.\.|[-+*/%^&|]=|[-+*/%^&|!=<>@.,;:\#$?~\\])
 |(?P<open>[(\[{])
 |(?P<close>[)\]}])
""",
    re.X,
)
_BLOCK = re.compile(r"/\*|\*/")
_PARTNER = {")": "(", "]": "[", "}": "{"}

# identifiers that are keywords and therefore never the name of a callee or the
# head of a postfix chain (`return (a).b()` is not a call of `return`)
KEYWORDS = frozenset(
    """as async await break const continue dyn else enum extern fn for if impl in let loop match mod move mut
    pub ref return static struct trait type unsafe use where while yield""".split()
)


class Comment(object):
    __slots__ = ("start", "end", "doc")

    def __init__(self, start, end, doc):
        self.start = start
        self.end = end
        # doc: "" (plain), "outer" (/// or /** */), "inner" (//! or /*! */)
        self.doc = doc


def _block_comment_end(src, pos):
    depth = 0
    for m in _BLOCK.finditer(src, pos):
        if m.group() == "/*":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.end()
    return -1


class Lexed(object):
    """The significant tokens of a text (no whitespace, no comments) as parallel lists.

    kind[i], start[i], end[i], text[i]   the token
    mate[i]                              index of the partner bracket, or -1
    up[i]                                index of the innermost enclosing OPEN bracket, or -1
    comments                             every comment, in order
    ok / error                           False and a message when the text does not lex
    """

    def __init__(self, src):
        self.src = src
        self.kind = []
        self.start = []
        self.end = []
        self.text = []
        self.mate = []
        self.up = []
        self.comments = []
        self.ok = True
        self.error = ""
        self._nl = None
        self._lex()

    # ------------------------------------------------------------------ lexing
    def _fail(self, pos, why):
        if self.ok:
            self.ok = False
            self.error = "%s at line %d" % (why, self.src.count("\n", 0, pos) + 1)

    def _lex(self):
        src = self.src
        n = len(src)
        kind, start, end, text, mate, up = self.kind, self.start, self.end, self.text, self.mate, self.up
        stack = []
        pos = 0
        match = _MASTER.match
        while pos < n:
            m = match(src, pos)
            if m is None:
                # an unterminated literal, or a character outside the grammar
                self._fail(pos, "cannot tokenize %r" % src[pos:pos + 12])
                pos += 1
                continue
            g = m.lastgroup
            e = m.end()
            if g == "ws":
                pos = e
                continue
            if g == "lc":
                t = src[pos:e]
                doc = ""
                if t.startswith("///") and not t.startswith("////"):
                    doc = "outer"
                elif t.startswith("//!"):
                    doc = "inner"
                self.comments.append(Comment(pos, e, doc))
                pos = e
                continue
            if g == "bc":
                e = _block_comment_end(src, pos)
                if e < 0:
                    self._fail(pos, "unterminated block comment")
                    e = n
                t = src[pos:e]
                doc = ""
                if t.startswith("/**") and not t.startswith("/***") and len(t) > 4:
                    doc = "outer"
                elif t.startswith("/*!"):
                    doc = "inner"
                self.comments.append(Comment(pos, e, doc))
                pos = e
                continue
            i = len(kind)
            if g == "rstr":
                g = STR
            kind.append(g)
            start.append(pos)
            end.append(e)
            text.append(src[pos:e])
            mate.append(-1)
            if g == OPEN:
                up.append(stack[-1] if stack else -1)
                stack.append(i)
            elif g == CLOSE:
                if stack and text[stack[-1]] == _PARTNER[text[i]]:
                    j = stack.pop()
                    mate[i] = j
                    mate[j] = i
                    up.append(up[j])
                else:
                    self._fail(pos, "unbalanced %r" % text[i])
                    up.append(stack[-1] if stack else -1)
            else:
                up.append(stack[-1] if stack else -1)
            pos = e
        if stack:
            self._fail(start[stack[-1]], "unclosed %r" % text[stack[-1]])

    # --------------------------------------------------------------- positions
    def __len__(self):
        return len(self.kind)

    def line_of(self, offset):
        """1-based line number of a character offset."""
        if self._nl is None:
            self._nl = [m.start() for m in re.finditer("\n", self.src)]
        return bisect.bisect_left(self._nl, offset) + 1

    def line_start(self, offset):
        return self.src.rfind("\n", 0, offset) + 1

    def indent_at(self, offset):
        ls = self.line_start(offset)
        m = re.match(r"[ \t]*", self.src[ls:])
        return m.group()

    def tok_index_at(self, offset):
        """Index of the first token that starts at or after `offset` (len(self) if none)."""
        return bisect.bisect_left(self.start, offset)

    def span(self, i, j):
        """Source text from the start of token i to the end of token j."""
        return self.src[self.start[i]:self.end[j]]

    # ------------------------------------------------------------ small tests
    def is_p(self, i, s):
        return 0 <= i < len(self.kind) and self.kind[i] == PUNCT and self.text[i] == s

    def is_id(self, i, s=None):
        if not (0 <= i < len(self.kind)) or self.kind[i] != ID:
            return False
        return s is None or self.text[i] == s

    def is_open(self, i, s=None):
        if not (0 <= i < len(self.kind)) or self.kind[i] != OPEN:
            return False
        return s is None or self.text[i] == s

    def is_close(self, i, s=None):
        if not (0 <= i < len(self.kind)) or self.kind[i] != CLOSE:
            return False
        return s is None or self.text[i] == s

    def tx(self, i):
        return self.text[i] if 0 <= i < len(self.kind) else ""

    # ------------------------------------------------------------- generics
    def turbofish_end(self, i):
        """`i` is a `<` that follows `::`.  Return the index of the matching `>`, or -1.

        Only type-like content is accepted (identifiers, paths, `&`, lifetimes, `_`,
        commas, nested `<..>` and bracket groups); anything else means that this `<`
        is not a generic-argument list and the caller must give up.
        """
        kind, text, mate = self.kind, self.text, self.mate
        n = len(kind)
        depth = 0
        j = i
        while j < n:
            k, t = kind[j], text[j]
            if k == PUNCT:
                if t == "<":
                    depth += 1
                elif t == ">":
                    depth -= 1
                    if depth == 0:
                        return j
                elif t not in ("::", ",", "&", "*", "->", "=", "!", "+", "?", ":", ";"):
                    return -1
            elif k == OPEN:
                if mate[j] < 0:
                    return -1
                j = mate[j]
            elif k == CLOSE:
                return -1
            elif k not in (ID, LIFE, NUM):
                return -1
            j += 1
        return -1

    def turbofish_start(self, i):
        """`i` is a `>`.  Return the index of the `<` that opens the generic-argument list it
        closes, provided that `<` follows `::` (a turbofish); otherwise -1."""
        kind, text, mate = self.kind, self.text, self.mate
        depth = 0
        j = i
        while j >= 0:
            k, t = kind[j], text[j]
            if k == PUNCT:
                if t == ">":
                    depth += 1
                elif t == "<":
                    depth -= 1
                    if depth == 0:
                        return j if self.is_p(j - 1, "::") else -1
                elif t not in ("::", ",", "&", "*", "->", "=", "!", "+", "?", ":", ";"):
                    return -1
            elif k == CLOSE:
                if mate[j] < 0:
                    return -1
                j = mate[j]
            elif k == OPEN:
                return -1
            elif k not in (ID, LIFE, NUM):
                return -1
            j -= 1
        return -1

    # ------------------------------------------------------------ arguments
    def split_args(self, open_i):
        """The top-level comma-separated arguments of the bracket group that opens at `open_i`.

        Returns a list of (first_token, last_token) index pairs, or None when the split
        is not certain (a `<` that may or may not open generic arguments).  A trailing
        comma does not produce an empty argument.  Commas inside nested brackets, inside
        a turbofish (`collect::<Vec<_>>()`), and between the bars of a closure's
        parameter list (`|a, b| ..`) do not split.
        """
        kind, text, mate = self.kind, self.text, self.mate
        close_i = mate[open_i]
        if close_i < 0:
            return None
        args = []
        first = open_i + 1
        j = first
        while j < close_i:
            k, t = kind[j], text[j]
            if k == OPEN:
                j = mate[j]
                if j < 0:
                    return None
            elif k == PUNCT:
                if t == ",":
                    if j > first:
                        args.append((first, j - 1))
                    else:
                        return None  # an empty argument: not an expression list
                    first = j + 1
                elif t == "<":
                    if self.is_p(j - 1, "::"):
                        e = self.turbofish_end(j)
                        if e < 0 or e >= close_i:
                            return None
                        j = e
                    elif self._type_position(j, first):
                        return None
                elif t == "|" and self._closure_bar(j, first):
                    # skip the closure's parameter list to its closing bar
                    e = j + 1
                    while e < close_i and not (kind[e] == PUNCT and text[e] == "|"):
                        if kind[e] == OPEN:
                            e = mate[e]
                            if e < 0:
                                return None
                        e += 1
                    if e >= close_i:
                        return None
                    j = e
            j += 1
        if first < close_i:
            args.append((first, close_i - 1))
        return args

    def _closure_bar(self, j, first):
        """Is the `|` at j the opening bar of a closure's parameters (and not a binary or)?"""
        if j == first:
            return True
        p = j - 1
        if self.kind[p] == ID:
            return self.text[p] in ("move", "return", "in", "else", "async")
        if self.kind[p] == PUNCT:
            return self.text[p] in ("=", ",", "=>", "&&", "||", ";", ":", "==", "!=", "+", "-", "*", "/", "!")
        return self.kind[p] == OPEN

    def _type_position(self, j, first):
        """A `<` that is not after `::`: does it look like generics (`x as Vec<A, B>`, `<T as U>::f`)?
        If it may be, the argument split is uncertain."""
        if j == first:
            return True  # `<T as Trait>::f(..)`
        p = j - 1
        if self.kind[p] != ID:
            return False
        # `Vec<..>` where the identifier starts with an upper-case letter is a type in
        # practice; a comparison `a < b` has a lower-case left operand in practice.
        # Being wrong here only makes the caller give up.
        name = self.text[p]
        if not name[:1].isupper():
            return False
        # only a problem if a comma follows before the group closes at this depth
        e = self.turbofish_end(j)
        if e < 0:
            return False
        return any(self.kind[x] == PUNCT and self.text[x] == "," for x in range(j, e))


def is_boundary_before(lx, i):
    """Can a new expression start at token i?  True when the token before it is one after
    which an expression begins (so nothing to the left binds to the expression at i)."""
    p = i - 1
    if p < 0:
        return True
    k, t = lx.kind[p], lx.text[p]
    if k == OPEN:
        return True
    if k == CLOSE:
        return t == "}"  # a statement after a block
    if k == PUNCT:
        return t in (",", ";", "=", "=>", "&&", "||", ":", "+=", "-=", "*=", "/=", "|")
    if k == ID:
        return t in ("return", "if", "while", "match", "else", "in", "break", "yield", "move")
    return False


def is_boundary_after(lx, i):
    """Does the expression that ends at token i end there?  True when the next token cannot
    continue it (a closing bracket, a separator, a lazy boolean operator, a block)."""
    q = i + 1
    if q >= len(lx.kind):
        return True
    k, t = lx.kind[q], lx.text[q]
    if k == CLOSE:
        return True
    if k == OPEN:
        return t == "{"  # the block of `if cond {`
    if k == PUNCT:
        return t in (",", ";", "&&", "||", "=>")
    return False
