# Postfix chains, calls and macro calls on top of the tokenizer.
#
# A "postfix chain" is what a method call's receiver can be without parentheses:
#
#     primary ( .name | .name(args) | .name::<T>(args) | .0 | (args) | [index] | ? )*
#
# where primary is a path (`a`, `a::b`, `Vec::<T>::new`), a literal, a parenthesised
# group, an array, or a macro call (`vec![..]`, `expr!(..)`).  Prefix operators
# (`&`, `*`, `-`, `!`) bind more loosely than every postfix operator, so they are
# never part of the chain: in `!a.b()` the receiver of `.b()` is `a`.
#
# Every function returns -1 / None when the shape is not one it is sure about; the
# rules then list the site for a person instead of rewriting it.
from .lexer import ID, NUM, STR, CHAR, PUNCT, OPEN, CLOSE, KEYWORDS


def chain_start(lx, i):
    """`i` is the LAST token of a postfix chain.  Return the index of its first token, or -1."""
    kind, text, mate = lx.kind, lx.text, lx.mate
    j = i
    steps = 0
    while 0 <= j:
        steps += 1
        if steps > 4000:
            return -1
        k, t = kind[j], text[j]
        if k == CLOSE:
            if t == "}":
                return -1  # a block, a `match`, a struct literal: not a simple chain
            o = mate[j]
            if o < 0:
                return -1
            p = o - 1
            if p < 0:
                return o
            pk, pt = kind[p], text[p]
            if pk == ID and pt not in KEYWORDS:
                j = p  # callee name / indexed name
                continue
            if pk == CLOSE and pt in (")", "]"):
                j = p  # a call or an index applied to a call's result
                continue
            if pk == PUNCT and pt == "?":
                j = p
                continue
            if pk == PUNCT and pt == "!":
                if p - 1 >= 0 and kind[p - 1] == ID:
                    j = p - 1  # macro call: name!(..) / name![..]
                    continue
                return o  # `!(a)` : the group is the primary
            if pk == PUNCT and pt == ">" and t == ")":
                s = lx.turbofish_start(p)
                if s < 0:
                    return o  # `a > (b)`: a comparison, the group is the primary
                if s - 2 >= 0 and kind[s - 2] == ID:
                    j = s - 2  # name::<T>(..)
                    continue
                return -1
            return o  # a parenthesised expression, a tuple, an array literal
        if k == ID or k == NUM:
            p = j - 1
            if p >= 0 and kind[p] == PUNCT:
                pt = text[p]
                if pt == ".":
                    j = p - 1
                    continue
                if pt == "::":
                    q = p - 1
                    if q >= 0 and kind[q] == ID:
                        j = q
                        continue
                    if q >= 0 and kind[q] == PUNCT and text[q] == ">":
                        s = lx.turbofish_start(q)
                        if s >= 0 and s - 2 >= 0 and kind[s - 2] == ID:
                            j = s - 2  # Type::<T>::name
                            continue
                        return -1  # `<T as Trait>::name`
                    return p  # `::path`
                if pt == "$":
                    return p  # a macro variable `$x`
            if k == ID and text[j] in KEYWORDS:
                return -1
            return j
        if k == STR or k == CHAR:
            return j
        if k == PUNCT and t == "?":
            j -= 1
            continue
        return -1
    return -1


def chain_end(lx, i):
    """`i` is the FIRST token of a postfix chain.  Return the index of its last token, or -1."""
    kind, text, mate = lx.kind, lx.text, lx.mate
    n = len(kind)
    if i >= n:
        return -1
    j = i
    if kind[j] == PUNCT and text[j] == "$":
        j += 1
        if j >= n:
            return -1
    k, t = kind[j], text[j]
    if k == OPEN and t in ("(", "["):
        j = mate[j]
        if j < 0:
            return -1
    elif k in (STR, CHAR, NUM):
        pass
    elif k == ID:
        if t in KEYWORDS:
            return -1
        while True:
            if lx.is_p(j + 1, "::"):
                if lx.is_id(j + 2):
                    j += 2
                    continue
                if lx.is_p(j + 2, "<"):
                    e = lx.turbofish_end(j + 2)
                    if e < 0:
                        return -1
                    j = e
                    continue
                return -1
            break
        if lx.is_p(j + 1, "!") and lx.is_open(j + 2):
            j = mate[j + 2]
            if j < 0:
                return -1
    else:
        return -1
    while True:
        q = j + 1
        if q >= n:
            break
        qk, qt = kind[q], text[q]
        if qk == PUNCT and qt == ".":
            r = q + 1
            if r < n and kind[r] == ID:
                j = r
                if lx.is_p(j + 1, "::") and lx.is_p(j + 2, "<"):
                    e = lx.turbofish_end(j + 2)
                    if e < 0:
                        return -1
                    j = e
                continue
            if r < n and kind[r] == NUM:
                j = r
                continue
            return -1
        if qk == OPEN and qt in ("(", "["):
            j = mate[q]
            if j < 0:
                return -1
            continue
        if qk == PUNCT and qt == "?":
            j = q
            continue
        break
    return j


class Call(object):
    """A call `name(args)`: `name` at token `name_i`, the parentheses at `open_i` / `close_i`.

    style:  "method"  recv.name(..)      "path"  Type::name(..)      "free"  name(..)
    """

    __slots__ = ("name_i", "open_i", "close_i", "style")

    def __init__(self, name_i, open_i, close_i, style):
        self.name_i = name_i
        self.open_i = open_i
        self.close_i = close_i
        self.style = style


def call_at(lx, i):
    """If token i is an identifier that is called (`name(`, `name::<T>(`), return the Call."""
    if lx.kind[i] != ID:
        return None
    j = i + 1
    if lx.is_p(j, "::") and lx.is_p(j + 1, "<"):
        e = lx.turbofish_end(j + 1)
        if e < 0:
            return None
        j = e + 1
    if not lx.is_open(j, "("):
        return None
    c = lx.mate[j]
    if c < 0:
        return None
    if lx.is_id(i - 1, "fn"):
        return None  # a definition, not a call
    if lx.is_p(i - 1, "."):
        style = "method"
    elif lx.is_p(i - 1, "::"):
        style = "path"
    else:
        style = "free"
    return Call(i, j, c, style)


def calls_named(lx, names):
    """Every call whose callee's last path segment is in `names` (a set or dict)."""
    kind, text = lx.kind, lx.text
    for i in range(len(kind)):
        if kind[i] == ID and text[i] in names:
            c = call_at(lx, i)
            if c is not None:
                yield c


def path_before(lx, name_i):
    """The path segments written before `::name` at name_i, innermost last:
    `symplex::stats::Distribution::normal` -> ["symplex", "stats", "Distribution"].
    Generic arguments (`MultiPoly::<Lex>::var`) are skipped.  [] when there is no path."""
    segs = []
    j = name_i
    while lx.is_p(j - 1, "::"):
        q = j - 2
        if lx.is_p(q, ">"):
            s = lx.turbofish_start(q)
            if s < 0:
                return None
            q = s - 2
        if not lx.is_id(q):
            break
        segs.append(lx.text[q])
        j = q
    segs.reverse()
    return segs


def path_start(lx, name_i):
    """Token index where the path that ends at name_i begins (`a::b::name` -> index of `a`)."""
    j = name_i
    while lx.is_p(j - 1, "::"):
        q = j - 2
        if lx.is_p(q, ">"):
            s = lx.turbofish_start(q)
            if s < 0:
                return -1
            q = s - 2
        if not lx.is_id(q):
            return j - 1  # a leading `::`
        j = q
    return j


class MacroCall(object):
    __slots__ = ("name", "name_i", "open_i", "close_i", "args")

    def __init__(self, name, name_i, open_i, close_i, args):
        self.name = name
        self.name_i = name_i
        self.open_i = open_i
        self.close_i = close_i
        self.args = args


def macro_at(lx, open_i):
    """If the bracket at open_i is the delimiter of a macro call `name!(..)`, return it
    (args is None when the argument split is uncertain)."""
    if open_i < 2 or not lx.is_p(open_i - 1, "!") or not lx.is_id(open_i - 2):
        return None
    c = lx.mate[open_i]
    if c < 0:
        return None
    return MacroCall(lx.text[open_i - 2], open_i - 2, open_i, c, lx.split_args(open_i))


def enclosing_macro(lx, i):
    """The macro call whose delimiters directly enclose token i, or None."""
    g = lx.up[i]
    if g < 0:
        return None
    return macro_at(lx, g)


def whole_arg(args, first, last):
    """Index of the argument that is exactly the token range first..last, or -1."""
    if args is None:
        return -1
    for n, (a, b) in enumerate(args):
        if a == first and b == last:
            return n
    return -1


def toks_are(lx, a, b, texts):
    """Do tokens a..b spell exactly the sequence `texts`?"""
    if b - a + 1 != len(texts):
        return False
    return all(lx.text[a + n] == s for n, s in enumerate(texts))


def stmt_start(lx, i):
    """First token of the statement (or argument-less group content) that contains token i:
    the token after the previous `;` at the same bracket depth, or after the enclosing
    bracket's opener."""
    g = lx.up[i]
    j = i
    while j - 1 > g:
        p = j - 1
        if lx.kind[p] == CLOSE:
            m = lx.mate[p]
            if m < 0:
                return j
            # a `}` that ends an item or a block statement also ends the statement,
            # except where an expression continues after it (`else`, a method call, ..)
            if lx.text[p] == "}" and not _continues_after_block(lx, j):
                return j
            j = m
            continue
        if lx.kind[p] == PUNCT and lx.text[p] == ";":
            return j
        j = p
    return j


def _continues_after_block(lx, j):
    if j >= len(lx.kind):
        return False
    k, t = lx.kind[j], lx.text[j]
    if k == ID:
        return t == "else"
    if k == PUNCT:
        return t in (".", "?", ",", ";", "=>", "==", "!=", "&&", "||", "+", "-", "*", "/")
    return k == CLOSE


def enclosing_fn(lx, i):
    """The innermost `fn` item whose body contains token i.

    Returns (fn_token, body_open, in_closure) or None.  `in_closure` is True when a closure
    body lies between the function's body and token i (then `?` at i would return from the
    closure, not from the function)."""
    in_closure = False
    g = lx.up[i]
    child = i
    while g >= 0:
        if lx.text[g] == "{":
            # is this brace the body of a fn?  scan back over the signature
            f = _fn_of_body(lx, g)
            if f >= 0:
                return (f, g, in_closure)
            if _is_closure_body(lx, g):
                in_closure = True
        # a closure without braces: `|x| expr` somewhere to the left inside this group
        if _closure_bar_before(lx, g, child):
            in_closure = True
        child = g
        g = lx.up[g]
    return None


def _fn_of_body(lx, brace):
    """If `brace` opens the body of a `fn`, return the index of the `fn` token, else -1."""
    j = brace - 1
    lim = 0
    while j >= 0 and lim < 400:
        lim += 1
        k, t = lx.kind[j], lx.text[j]
        if k == CLOSE:
            if t == "}":
                return -1
            j = lx.mate[j] - 1
            continue
        if k == OPEN:
            return -1
        if k == PUNCT and t in (";", "=", "=>", "|", "||", ","):
            # `,` is fine inside generics/where clauses, but those are rare in tests
            if t == ",":
                j -= 1
                continue
            return -1
        if k == ID and t == "fn":
            # `fn name ( .. ) [-> T] [where ..] {`
            if lx.is_id(j + 1) and (lx.is_open(j + 2, "(") or lx.is_p(j + 2, "<")):
                return j
            return -1
        if k == ID and t in ("if", "else", "match", "while", "for", "loop", "unsafe", "move", "impl", "mod",
                             "struct", "enum", "trait", "async", "in", "let", "return"):
            # `impl X for Y {`, `mod m {` and control flow are not function bodies
            if t in ("unsafe", "async"):
                j -= 1
                continue
            return -1
        j -= 1
    return -1


def _is_closure_body(lx, brace):
    p = brace - 1
    if lx.is_p(p, "|") or lx.is_p(p, "||"):
        return True
    # `|x| -> T {`
    j = p
    lim = 0
    while j >= 0 and lim < 12:
        lim += 1
        if lx.is_p(j, "->"):
            return lx.is_p(j - 1, "|") or lx.is_p(j - 1, "||")
        if lx.kind[j] not in (ID, PUNCT) or lx.text[j] in (";", "=", "{", "}"):
            return False
        j -= 1
    return False


def _closure_bar_before(lx, g, child):
    """Inside group g, between its opener and token `child`, in the same comma-separated part,
    is there a closure head (`|..|` or `||`)?"""
    j = child - 1
    while j > g:
        k, t = lx.kind[j], lx.text[j]
        if k == CLOSE:
            j = lx.mate[j] - 1
            continue
        if k == PUNCT:
            if t in (",", ";"):
                return False
            if t == "||":
                # `a || b` is a boolean or when something expression-like precedes it
                if j - 1 <= g or lx.text[j - 1] in ("=", "(", ",", "move", "return"):
                    return True
            if t == "|":
                return True
        j -= 1
    return False


def fn_returns_result(lx, fn_i, body_open):
    """Text of the return type of the fn at fn_i when it is a `Result` (else "")."""
    j = fn_i
    while j < body_open:
        if lx.is_p(j, "->"):
            ty = lx.span(j + 1, body_open - 1)
            head = ty.replace(" ", "")
            for pre in ("Result<", "symplex::Result<", "std::result::Result<", "anyhow::Result<",
                        "core::result::Result<"):
                if head.startswith(pre):
                    return ty
            return ""
        if lx.kind[j] == OPEN:
            j = lx.mate[j]
        j += 1
    return ""
