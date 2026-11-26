#!/usr/bin/env python3
"""Check a hunt.sh result file against an independent computation (mpmath).

    oracle.py RESULTS.tsv [--out VERDICTS.tsv] [--ops evald30,evalf64,diff]
              [--timeout 20] [--mem-mb 2048] [--jobs 8] [--ulps 2] [--f64-ulps 4] [-n 20]
    oracle.py --table BASE.verdicts.tsv NEW.verdicts.tsv

One verdict per (case, op) line of RESULTS.tsv that has an oracle:

    agree            the oracle's value and symplex's agree numerically
    DISAGREE         the oracle has a value and symplex's is another one
    oracle-failed    the oracle raised, ran out of time or memory, or its two
                     precisions do not agree with each other: nothing is known
    not-comparable   nothing to compare: symplex refused (ERR) or did not
                     return (PANIC, TIMEOUT, ...), or the oracle cannot read
                     the expression (a function missing from FUNCTIONS, a free
                     symbol, a HOOK that is not written yet)

A refusal where the oracle does have a value is `not-comparable` with a
detail starting `refused; oracle value` and is counted in its own column
(`refused`): the count to watch when a fix turns answers into refusals.

How a case is checked:

* Every case runs in ONE FORKED CHILD with a time limit (--timeout, wall
  seconds, then SIGKILL) and a memory limit (--mem-mb, RLIMIT_AS).  The ops of
  the case run one after the other in that child; what was finished before
  the limit keeps its verdict, the rest is `oracle-failed`.
* Numeric ops (`evald30`, `evalf64`: the input at x = 7/10, y = -13/10, as
  the harness evaluates it) are computed by mpmath at TWO precisions
  (digits + 10 and 2*digits + 20).  The oracle counts only if the two agree
  to digits + 2 places (or both vanish below 10^-(digits+5): the value is
  zero to that precision); while they do not, both are doubled, twice at
  most, and then the verdict is `oracle-failed`.
* The comparison is numeric, never by string: symplex's printed value is
  read back as a number and must lie within --ulps units of the last of the
  30 digits (`evald30`), or within --f64-ulps units in the last place of an
  f64 (`evalf64`), of the oracle's value.  Complex values are compared by
  the modulus of the difference.
* Symbolic ops are HOOKS (see `SYMBOLIC` below): `diff` is written out as
  the worked example (d/dx of the input, numerically, against the result at
  random points); `integrate`, `limit0`, `limitoo` are marked stubs.

Output: VERDICTS.tsv (default RESULTS.tsv + `.verdicts.tsv`) with the columns
case_id, op, status, verdict, detail, result, input; a table per op and the
DISAGREE lines on stdout.  Exit status 1 iff there is a DISAGREE.

`--table` prints "wrong before -> wrong after" from two verdict files.

Needs mpmath (SymPy is not imported; use it inside a hook if you want a
second opinion, never as the ground truth).  The expression is never passed
to `eval`: it is parsed with `ast` and walked.
"""

import argparse
import ast
import json
import os
import random
import re
import resource
import select
import signal
import sys
import time

import mpmath as mp

VERDICTS = ["agree", "DISAGREE", "oracle-failed", "not-comparable"]
REFUSED = "refused; oracle value"

# ---------------------------------------------------------------------------
# The expression evaluator: symplex's input syntax -> mpmath values.
# ---------------------------------------------------------------------------


class NoOracle(Exception):
    """The oracle cannot read this expression: verdict `not-comparable`."""


CONSTANTS = {
    "pi": lambda: +mp.pi,
    "E": lambda: +mp.e,
    "I": lambda: mp.mpc(0, 1),
    "EulerGamma": lambda: +mp.euler,
    "GoldenRatio": lambda: +mp.phi,
    "Catalan": lambda: +mp.catalan,
    "oo": lambda: mp.inf,
}

# (name, number of arguments) -> mpmath function.  Only functions whose
# conventions (argument order, branch cuts, normalisation) were checked are
# listed: add yours here, with the convention in a comment.  A function that
# is not listed gives `not-comparable`, never a guess.
FUNCTIONS = {
    (name, 1): getattr(mp, name)
    for name in (
        "sin cos tan sec csc cot sinh cosh tanh sech csch coth "
        "asin acos atan acot asec acsc asinh acosh atanh acoth asech acsch "
        "exp sqrt gamma loggamma digamma erf erfc erfi erfinv zeta factorial "
        "lambertw airyai airybi floor sign re im arg"
    ).split()
}
FUNCTIONS.update(
    {
        ("ln", 1): mp.log,
        ("log", 1): mp.log,
        ("log", 2): lambda z, b: mp.log(z, b),  # log(z, b): base second
        ("abs", 1): abs,
        ("conjugate", 1): mp.conj,

        ("beta", 2): mp.beta,
        ("binomial", 2): mp.binomial,
        ("Si", 1): mp.si,
        ("Ci", 1): mp.ci,
        ("Ei", 1): mp.ei,
        ("li", 1): mp.li,
        ("E1", 1): mp.e1,
        ("Shi", 1): mp.shi,
        ("Chi", 1): mp.chi,
        ("besselj", 2): mp.besselj,  # (order, argument), also y, i, k
        ("bessely", 2): mp.bessely,
        ("besseli", 2): mp.besseli,
        ("besselk", 2): mp.besselk,
        ("polylog", 2): mp.polylog,  # (s, z)
        ("dirichlet_eta", 1): mp.altzeta,
        ("fibonacci", 1): mp.fib,
        ("harmonic", 1): mp.harmonic,
        ("legendre", 2): mp.legendre,  # (degree, argument)
        ("chebyshev_t", 2): mp.chebyt,
        ("chebyshev_u", 2): mp.chebyu,
        ("hermite", 2): mp.hermite,
        ("atan2", 2): mp.atan2,  # (y, x)
        # Names symplex PRINTS (results of `diff`, ...) where they differ from
        # the names it reads.
        ("Gamma", 1): mp.gamma,
        ("Digamma", 1): mp.digamma,
        ("W", 1): mp.lambertw,
        ("airyaiprime", 1): lambda z: mp.airyai(z, derivative=1),
        ("airybiprime", 1): lambda z: mp.airybi(z, derivative=1),
    }
)


def _polygamma(n, z):
    # mpmath truncates the order to an integer without a word.
    if mp.im(n) != 0 or n != mp.floor(n) or n < 0:
        raise NoOracle("no oracle for polygamma of an order that is not a non-negative integer")
    return mp.polygamma(int(n), z)


def _real_only(f, name):
    # symplex refuses floor and ceiling of a non-real number; mpmath rounds the two parts.
    def g(z):
        if mp.im(z) != 0:
            raise NoOracle(f"no oracle for {name} of a non-real argument")
        return f(mp.re(z))

    return g


FUNCTIONS[("polygamma", 2)] = _polygamma
FUNCTIONS[("floor", 1)] = _real_only(mp.floor, "floor")
FUNCTIONS[("ceiling", 1)] = _real_only(mp.ceil, "ceiling")


def parse(text):
    """symplex syntax -> (ast node, the source it was parsed from)."""
    src = text.replace("^", "**")
    try:
        return ast.parse(src, mode="eval").body, src
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise NoOracle("the oracle cannot parse this expression") from None


def evaluate(node, env, src):
    """The value of `node` at the current mpmath precision; `env` maps symbols to values."""
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise NoOracle("a literal that is not a number")
        if isinstance(v, int):
            return mp.mpf(v)
        return mp.mpf(ast.get_source_segment(src, node))  # the decimal text, not the float
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]()
        raise NoOracle(f"free symbol or unknown name '{node.id}'")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = evaluate(node.operand, env, src)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp):
        a, b = evaluate(node.left, env, src), evaluate(node.right, env, src)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        if isinstance(node.op, ast.Pow):
            return mp.power(a, b)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        f = FUNCTIONS.get((node.func.id, len(node.args)))
        if f is None:
            raise NoOracle(f"no oracle for {node.func.id}/{len(node.args)} (add it to FUNCTIONS)")
        return f(*[evaluate(a, env, src) for a in node.args])
    raise NoOracle(f"syntax the oracle does not know ({type(node).__name__})")


_NUM = r"(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
# A real part is followed by the end or by the sign of the imaginary part:
# without that lookahead `0.25*i` also matches as 0.2 + 5*i.
_VALUE = re.compile(rf"(?:(?P<re>[+-]?{_NUM})(?=$|[+-]))?(?:(?P<sign>[+-]?)(?:(?P<im>{_NUM})\*)?[iI])?")


def read_value(text):
    """symplex's printed number (`1.5`, `2e-7`, `i`, `-0.6 + 2.5*i`) as mpf/mpc, or None."""
    t = text.replace(" ", "")
    m = _VALUE.fullmatch(t)
    if not t or not m:
        return None
    real = mp.mpf(m["re"]) if m["re"] else mp.mpf(0)
    if t[-1] not in "iI":
        return real
    imag = mp.mpf(m["im"]) if m["im"] else mp.mpf(1)
    return mp.mpc(real, -imag if m["sign"] == "-" else imag)


# ---------------------------------------------------------------------------
# Numeric ops: fully implemented.
# ---------------------------------------------------------------------------

# op -> (kind, significant digits of the result)
NUMERIC = {"evald30": ("decimal", 30), "evalf64": ("f64", 17)}


def at_harness_point(tree, src, dps):
    """The input at x = 7/10, y = -13/10 (probes/golden/src/ops.rs: at_point)."""
    mp.mp.dps = dps
    env = {"x": mp.mpf(7) / 10, "y": -mp.mpf(13) / 10}
    return mp.mpmathify(evaluate(tree, env, src))


def judge_numeric(op, case_id, status, result, tree, src, args):
    kind, digits = NUMERIC[op]
    d1, d2 = digits + 10, 2 * digits + 20
    oracle, why, zero = None, "", False
    try:
        for _ in range(3):
            v1 = at_harness_point(tree, src, d1)
            v2 = at_harness_point(tree, src, d2)
            mp.mp.dps = d2
            floor = mp.mpf(10) ** -(digits + 5)
            if not (mp.isfinite(v1) and mp.isfinite(v2)):
                why = f"oracle value is not finite ({mp.nstr(v2, 8)})"
                break
            if abs(v1 - v2) <= abs(v2) * mp.mpf(10) ** -(digits + 2):
                oracle = v2
                break
            if abs(v1) < floor and abs(v2) < floor:
                oracle, zero = mp.mpf(0), True
                break
            why = f"oracle unstable: {mp.nstr(v1, 12)} at {d1} digits, {mp.nstr(v2, 12)} at {d2}"
            d1, d2 = d2, 2 * d2
    except (NoOracle, MemoryError):
        raise
    except Exception as e:  # noqa: BLE001  (a pole, a domain error, no convergence, ...)
        why = f"oracle raised {type(e).__name__}: {str(e)[:80]}"
    mp.mp.dps = max(d2, 2 * digits + 20)
    if oracle is None:
        allowed = None
    elif zero:
        allowed = mp.mpf(10) ** -(digits + 5)
    elif kind == "decimal":
        allowed = args.ulps * mp.mpf(10) ** (mp.floor(mp.log10(abs(oracle))) - digits + 1)
    else:  # an f64 has no unit below 2^-1074: a value that underflows compares with 0
        allowed = args.f64_ulps * mp.mpf(2) ** max(mp.floor(mp.log(abs(oracle), 2)) - 52, -1074)
    shown = "0 (to %s)" % mp.nstr(allowed, 1) if zero else (mp.nstr(oracle, digits + 4) if oracle is not None else "")

    if status == "ERR":
        if oracle is None:
            return "not-comparable", f"refused, and no oracle value: {why}"
        if kind == "f64" and abs(mp.im(oracle)) > allowed:
            return "not-comparable", f"refused, and the oracle value is not real: {shown}"
        return "not-comparable", f"{REFUSED} {shown}"
    if status != "OK":
        return "not-comparable", f"no value from symplex ({status})"
    s = read_value(result)
    if s is None:
        return "not-comparable", f"cannot read the value '{result[:40]}'"
    if oracle is None:
        return "oracle-failed", why
    err = abs(s - oracle)
    detail = f"oracle {shown}; |difference| {mp.nstr(err, 3)}, allowed {mp.nstr(allowed, 3)}"
    return ("agree" if err <= allowed else "DISAGREE"), detail


# ---------------------------------------------------------------------------
# HOOKS: symbolic ops.  One function per op, same signature as judge_numeric,
# returning (verdict, detail); raise NoOracle for `not-comparable`.  Compare
# by numeric substitution at random points (seed the generator with the case
# id, so a verdict is reproducible), at two precisions, and skip points
# where either side cannot be evaluated.
# ---------------------------------------------------------------------------

SYM_POINTS = 3  # points that must agree
SYM_TRIES = 12  # points tried (poles and domain errors are skipped)
SYM_RTOL = 1e-10  # relative agreement asked at each point


def judge_diff(op, case_id, status, result, tree, src, args):
    """WORKED EXAMPLE.  `diff` is d/dx of the input: mpmath differentiates
    the INPUT numerically at a random point and the RESULT is evaluated there.
    """
    if status != "OK":
        return "not-comparable", f"no value from symplex ({status})"
    if re.fullmatch(r"len=[0-9]+ fnv=[0-9a-f]+", result):
        raise NoOracle("the result is a hash (a golden TSV): run the case through hunt.sh")
    rtree, rsrc = parse(result)  # NoOracle if the result is not plain (Derivative(...), ...)
    rng = random.Random(case_id)
    good, skipped = 0, []
    for _ in range(SYM_TRIES):
        if good == SYM_POINTS:
            break
        px, py = rng.randint(200, 1800), rng.randint(200, 1800)
        try:
            vals = []
            for dps in (25, 50):
                mp.mp.dps = dps
                x0, y0 = mp.mpf(px) / 1000, mp.mpf(py) / 1000
                want = mp.diff(lambda t, y0=y0: evaluate(tree, {"x": t, "y": y0}, src), x0)
                got = evaluate(rtree, {"x": x0, "y": y0}, rsrc)
                vals.append((mp.mpmathify(want), mp.mpmathify(got)))
        except (NoOracle, MemoryError):
            raise
        except Exception as e:  # noqa: BLE001
            skipped.append(type(e).__name__)
            continue
        (w1, g1), (w2, g2) = vals
        scale = max(abs(w2), abs(g2), mp.mpf(10) ** -15)
        if not (mp.isfinite(w2) and mp.isfinite(g2)) or abs(w1 - w2) > scale * SYM_RTOL / 100:
            skipped.append("unstable")
            continue
        if abs(w2 - g2) > scale * SYM_RTOL:
            return "DISAGREE", f"at x={px}/1000, y={py}/1000: numeric d/dx {mp.nstr(w2, 15)}, result {mp.nstr(g2, 15)}"
        good += 1
    if good >= 2:
        return "agree", f"{good} random points to {SYM_RTOL:g} relative"
    return "oracle-failed", f"only {good} usable points ({', '.join(skipped[:4])})"


def judge_integrate(op, case_id, status, result, tree, src, args):
    """HOOK, NOT WRITTEN.  Same pattern with the roles swapped: differentiate
    the RESULT numerically (`mp.diff`) and compare with the INPUT at random
    points; `not-comparable` when the result still contains an integral.
    Mind branch cuts: a correct antiderivative may be piecewise constant off
    the points you draw, so draw them in one interval.
    """
    raise NoOracle("HOOK not written: integrate (see judge_integrate in oracle.py)")


def judge_limit(op, case_id, status, result, tree, src, args):
    """HOOK, NOT WRITTEN.  `limit0`: evaluate the input at x = +-10^-k for
    k = 5, 10, 20 (precision well above k digits) and compare the sequence
    with the result (two-sided: both signs must tend to it).  `limitoo`: the
    same at x = 10^k.  A result with `oo` or `zoo` needs its own rule.
    """
    raise NoOracle(f"HOOK not written: {op} (see judge_limit in oracle.py)")


SYMBOLIC = {
    "diff": judge_diff,
    "integrate": judge_integrate,
    "limit0": judge_limit,
    "limitoo": judge_limit,
}
JUDGES = {**dict.fromkeys(NUMERIC, judge_numeric), **SYMBOLIC}

# ---------------------------------------------------------------------------
# One forked child per case, with a time limit and a memory limit.
# ---------------------------------------------------------------------------


def child(case_id, expr, rows, wfd, args):
    """Runs in the forked child: one JSON line per op into the pipe, then exits."""
    code = 0
    try:
        resource.setrlimit(resource.RLIMIT_AS, (args.mem_mb << 20, args.mem_mb << 20))
        try:
            parsed, bad = parse(expr), None
        except NoOracle as e:
            parsed, bad = None, str(e)
        for op, status, result in rows:
            try:
                if parsed is None:
                    raise NoOracle(bad)
                verdict, detail = JUDGES[op](op, case_id, status, result, parsed[0], parsed[1], args)
            except NoOracle as e:
                verdict, detail = "not-comparable", str(e)
            except MemoryError:
                verdict, detail = "oracle-failed", f"memory limit ({args.mem_mb} MiB)"
            except RecursionError:
                verdict, detail = "oracle-failed", "recursion limit"
            except Exception as e:  # noqa: BLE001
                verdict, detail = "oracle-failed", f"{type(e).__name__}: {str(e)[:80]}"
            os.write(wfd, (json.dumps([op, verdict, detail]) + "\n").encode())
    except BaseException:  # noqa: BLE001  (never return into the parent's code)
        code = 1
    finally:
        os._exit(code)


def run_cases(cases, args):
    """{case_id: {op: (verdict, detail)}}; `cases` is [(case_id, expr, rows)]."""
    todo = list(reversed(cases))
    running, done = {}, {}

    def finish(fd, reason):
        pid, case_id, rows, _, buf = running.pop(fd)
        os.close(fd)
        got = {}
        for line in buf.decode("utf-8", "replace").splitlines():
            try:
                op, verdict, detail = json.loads(line)
                got[op] = (verdict, detail)
            except ValueError:
                pass
        for op, _, _ in rows:
            got.setdefault(op, ("oracle-failed", reason))
        done[case_id] = got

    while todo or running:
        while todo and len(running) < args.jobs:
            case_id, expr, rows = todo.pop()
            rfd, wfd = os.pipe()
            sys.stdout.flush()
            pid = os.fork()
            if pid == 0:
                os.close(rfd)
                child(case_id, expr, rows, wfd, args)
            os.close(wfd)
            running[rfd] = (pid, case_id, rows, time.monotonic() + args.timeout, b"")
        now = time.monotonic()
        wait = max(0.0, min(v[3] for v in running.values()) - now)
        ready, _, _ = select.select(list(running), [], [], min(wait, 0.5))
        for fd in ready:
            data = os.read(fd, 1 << 16)
            pid, case_id, rows, deadline, buf = running[fd]
            if data:
                running[fd] = (pid, case_id, rows, deadline, buf + data)
                continue
            _, st = os.waitpid(pid, 0)
            sig = os.WTERMSIG(st) if os.WIFSIGNALED(st) else 0
            finish(fd, f"the oracle process died (signal {sig})" if sig else "the oracle process exited early")
        now = time.monotonic()
        for fd in [fd for fd, v in running.items() if v[3] <= now]:
            pid = running[fd][0]
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            finish(fd, f"time limit ({args.timeout:g} s)")
    return done


# ---------------------------------------------------------------------------
# Files and tables.
# ---------------------------------------------------------------------------


def read_results(path, only):
    """[(case_id, expr, [(op, status, result)])] for the ops that have an oracle."""
    cases, index = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            case_id, op = int(p[0]), p[1]
            if case_id not in index:
                index[case_id] = len(cases)
                cases.append([case_id, "", []])
            if op == "expr":
                cases[index[case_id]][1] = p[3]
            elif op in JUDGES and (only is None or op in only):
                cases[index[case_id]][2].append((op, p[2], p[3]))
    return [tuple(c) for c in cases if c[2]]


def counts_of(rows):
    """{op: {verdict..., 'refused': n, 'pairs': n}} from (op, verdict, detail) rows."""
    out = {}
    for op, verdict, detail in rows:
        c = out.setdefault(op, {**dict.fromkeys(VERDICTS, 0), "refused": 0, "pairs": 0})
        c[verdict] += 1
        c["pairs"] += 1
        if detail.startswith(REFUSED):
            c["refused"] += 1
    return out


def check(args):
    only = set(args.ops.split(",")) if args.ops else None
    cases = read_results(args.results, only)
    if not cases:
        sys.exit(f"{args.results}: no line of an op with an oracle ({', '.join(JUDGES)})")
    t0 = time.monotonic()
    done = run_cases(cases, args)
    out_path = args.out or args.results + ".verdicts.tsv"
    flat = []
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("case_id\top\tstatus\tverdict\tdetail\tresult\tinput\n")
        for case_id, expr, rows in cases:
            for op, status, result in rows:
                verdict, detail = done[case_id][op]
                detail = detail.replace("\t", " ").replace("\n", " ")
                f.write(f"{case_id}\t{op}\t{status}\t{verdict}\t{detail}\t{result}\t{expr}\n")
                flat.append((case_id, op, status, verdict, detail, result, expr))
    counts = counts_of([(r[1], r[3], r[4]) for r in flat])
    print(f"{'op':<12}{'pairs':>7}{'agree':>7}{'DISAGREE':>10}{'oracle-failed':>15}{'not-comparable':>16}{'(refused)':>11}")
    for op, c in counts.items():
        print(f"{op:<12}{c['pairs']:>7}{c['agree']:>7}{c['DISAGREE']:>10}{c['oracle-failed']:>15}{c['not-comparable']:>16}{c['refused']:>11}")
    print(f"{len(cases)} cases in {time.monotonic() - t0:.1f} s, mpmath {mp.__version__}; verdicts in {out_path}")
    bad = [r for r in flat if r[3] == "DISAGREE"]
    for case_id, op, _, _, detail, result, expr in bad[: args.n]:
        print(f"DISAGREE case {case_id} {op}: input {expr}\n    symplex {result}\n    {detail}")
    if len(bad) > args.n:
        print(f"... and {len(bad) - args.n} more DISAGREE lines in {out_path}")
    sys.exit(1 if bad else 0)


def read_verdicts(path):
    rows = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 7:
                rows[(int(p[0]), p[1])] = p[2:7]
    return rows


def table(base_path, new_path):
    base, new = read_verdicts(base_path), read_verdicts(new_path)
    cb = counts_of([(k[1], v[1], v[2]) for k, v in base.items()])
    cn = counts_of([(k[1], v[1], v[2]) for k, v in new.items()])
    cols = [("wrong", "DISAGREE"), ("refused", "refused"), ("agree", "agree"), ("oracle-failed", "oracle-failed"), ("not-comparable", "not-comparable")]
    print(f"{'before -> after':<12}" + "".join(f"{name:>17}" for name, _ in cols))
    zero = {**dict.fromkeys(VERDICTS, 0), "refused": 0}
    for op in list(cb) + [o for o in cn if o not in cb]:
        b, n = cb.get(op, zero), cn.get(op, zero)
        print(f"{op:<12}" + "".join(f"{f'{b[key]} -> {n[key]}':>17}" for _, key in cols))
    moved = [(k, base[k], new[k]) for k in base if k in new and (base[k][1] != new[k][1] or base[k][2].startswith(REFUSED) != new[k][2].startswith(REFUSED))]
    print(f"{len(moved)} pairs changed their verdict" + (":" if moved else " (`refused` is part of `not-comparable`)"))
    for (case_id, op), b, n in moved:
        print(f"  case {case_id} {op}: {b[1]} -> {n[1]}   input {n[4]}\n    before {b[0]} {b[3]}  [{b[2]}]\n    after  {n[0]} {n[3]}  [{n[2]}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", help="a hunt.sh output")
    ap.add_argument("--table", nargs=2, metavar=("BASE", "NEW"), help="two verdict files: wrong before -> after")
    ap.add_argument("--out", help="verdict file (default RESULTS.tsv.verdicts.tsv)")
    ap.add_argument("--ops", help="only these ops (default: every op with an oracle)")
    ap.add_argument("--timeout", type=float, default=20.0, help="wall seconds per case (default 20)")
    ap.add_argument("--mem-mb", type=int, default=2048, help="address space per case in MiB (default 2048)")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="children at a time (default 8)")
    ap.add_argument("--ulps", type=float, default=2.0, help="evald30: units of the 30th digit allowed (default 2)")
    ap.add_argument("--f64-ulps", type=float, default=4.0, help="evalf64: f64 units in the last place allowed (default 4)")
    ap.add_argument("-n", type=int, default=20, help="DISAGREE lines to print (default 20)")
    args = ap.parse_args()
    if args.table:
        table(*args.table)
    elif args.results:
        check(args)
    else:
        ap.error("give RESULTS.tsv or --table BASE NEW")


if __name__ == "__main__":
    main()
