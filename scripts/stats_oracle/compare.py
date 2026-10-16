#!/usr/bin/env python3
"""Compare symplex's value dump with scipy / statsmodels and mpmath (scripts/stats_oracle/README.md).

    python3 scripts/stats_oracle/compare.py --cases scripts/stats_oracle/cases \
        --dump target/stats_oracle/dump.tsv --allowlist scripts/stats_oracle/allowlist.tsv \
        --report target/stats_oracle/report.md [--jobs 4]

Every value symplex claims is checked against the truth for its own input:
certified decimals (`D:`) to one unit of their last digit, `f64` routes to
their documented tolerance, exact rationals exactly.  The truth is mpmath at
two precisions (40 and 60 digits) from the closed forms in refs.py, trusted
only where the two agree; where mpmath has no formula, scipy / statsmodels
(double precision) is the reference within 1e-10.  scipy / statsmodels are
judged the same way against mpmath at *their* (rounded) input: off by more
than 1e-9 relative is a scipy error (allowlist it with the mpmath value),
between 1e-12 and 1e-9 an inaccuracy (counted only).

Exit status 1 when a failure is not on the allowlist (classes below).
"""
import argparse
import fnmatch
import math
import multiprocessing
import os
import signal
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refs  # noqa: E402

# Classes.  OK: AGREE (checked against mpmath or exactly), AGREE_SCIPY (scipy only, no mpmath
# formula), NONFINITE (both say ±∞/NaN or refuse a nonfinite value), UNCHECKED (no reference),
# SCIPY_INACCURATE (symplex right, scipy off by 1e-12..1e-9).  Failures unless allowlisted:
FAILURES = {"SYMPLEX_WRONG", "SCIPY_WRONG", "UNRESOLVED", "REFUSED", "PANIC", "HANG", "CRASH", "BADROW",
            "MISSING"}
DPS = (40, 60)
ORACLE_TIMEOUT = int(os.environ.get("STATS_ORACLE_MP_TIMEOUT", "120"))


@dataclass
class Row:
    id: str
    kind: str
    target: str
    params: str
    method: str
    arg: str


@dataclass
class Verdict:
    id: str
    output: str
    cls: str
    sym: str
    scipy: str = ""
    mp: str = ""
    note: str = ""


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


# ───────────────────────────── reading ─────────────────────────────

def read_cases(path):
    files = sorted(path.glob("*.tsv")) if path.is_dir() else [path]
    rows = {}
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            c = line.split("\t")
            if len(c) != 6:
                rows[c[0]] = Row(c[0], "malformed", "", "", "", "")
                continue
            rows[c[0]] = Row(*c)
    return rows


def read_dump(path):
    out = defaultdict(dict)
    ms = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) < 3:
            continue
        out[c[0]][c[1]] = c[2]
        if len(c) > 3 and c[3].isdigit():
            ms[c[0]] = int(c[3])
    return out, ms


def read_allowlist(path):
    entries = []
    if not path or not path.exists():
        return entries
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) < 5 or not c[4].strip():
            sys.exit(f"{path}:{n}: an entry needs id, output, class, mpmath value and a reason")
        entries.append({"id": c[0], "output": c[1], "cls": c[2], "mp": c[3], "reason": c[4], "line": n,
                        "used": 0})
    return entries


# ───────────────────────────── values ─────────────────────────────

def parse_complex_decimal(s):
    """'a+b*i', 'a-b*i', 'b*i', 'a' -> (re, im) as Fractions (None for a missing part)."""
    s = s.replace("I", "i")
    if not s.endswith("i"):
        return F(s), None
    body = s[:-1].rstrip("*")
    # split at the last sign that is not part of an exponent
    for k in range(len(body) - 1, 0, -1):
        if body[k] in "+-" and body[k - 1] not in "eE":
            return F(body[:k]), F(body[k:] if body[k:] not in ("+", "-") else body[k:] + "1")
    return None, F(body if body not in ("", "+", "-") else body + "1")


def sym_kind(v):
    for k in ("D:", "F:", "C:", "Q:"):
        if v.startswith(k):
            return k[0]
    if v in ("INF", "-INF", "NAN", "NONE", "HANG"):
        return v
    return v.split(":", 1)[0]


def digits_tol(value):
    """One unit in the 30th significant digit of a certified decimal."""
    if value == 0:
        return mp.mpf(0)
    e = mp.floor(mp.log10(abs(value)))
    return mp.mpf(10) ** (e - 29)


def to_mp(x):
    if isinstance(x, F):
        return refs.M(x)
    return x


def fmt(v, n=20):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, F):
        return mp.nstr(refs.M(v), n) if v.denominator != 1 else str(v)
    if isinstance(v, complex):
        return f"{v.real!r}{v.imag:+}j"
    if isinstance(v, float):
        return repr(v)
    return mp.nstr(v, n)


def memo_dps(fn):
    """fn, evaluated once per working precision (the outputs of a row share their truth)."""
    if fn is None:
        return None
    cache = {}

    def wrapped():
        key = mp.mp.prec
        if key not in cache:
            cache[key] = fn()
        return cache[key]
    return wrapped


def mp_twice(fn):
    """fn() at two precisions; (value, agreeing digits).  Strings ("inf", ...) pass through."""
    vals = []
    for dps in DPS:
        with mp.workdps(dps):
            v = fn()
        if v is None or isinstance(v, str):
            return v, 999
        vals.append(v)
    a, b = vals
    with mp.workdps(DPS[1]):
        a, b = to_mp(a), to_mp(b)
        if a == b:
            return b, 999
        scale = max(abs(a), abs(b))
        if scale == 0:
            return b, 999
        d = abs(a - b) / scale
        return b, (999 if d == 0 else int(-mp.log10(d)))


def close(x, y, rel, absol=0):
    x, y = to_mp(x), to_mp(y)
    return abs(x - y) <= rel * max(abs(x), abs(y)) + absol


# ───────────────────────────── judging ─────────────────────────────

def judge_scalar(row, output, sym, sym_tol, scipy_val, mp_exact, mp_at_float=None):
    """sym: an mpf/Fraction; sym_tol(value) -> absolute tolerance of symplex's claim; scipy_val: float or
    None; mp_exact: a function giving the truth at symplex's input (or None); mp_at_float: the truth at
    scipy's input (None: the inputs are the same)."""
    truth, agree = (None, 0)
    if mp_exact is not None:
        try:
            truth, agree = mp_twice(mp_exact)
        except (Timeout, ZeroDivisionError, ValueError, OverflowError, mp.libmp.NoConvergence):
            truth, agree = None, 0
    sv = fmt(sym)
    if isinstance(truth, str):
        return Verdict(row.id, output, "SYMPLEX_WRONG", sv, fmt(scipy_val), truth, "the value is nonfinite")
    if truth is not None and agree >= 33:
        with mp.workdps(DPS[1]):
            ok = abs(to_mp(sym) - truth) <= sym_tol(truth)
            if not ok:
                return Verdict(row.id, output, "SYMPLEX_WRONG", sv, fmt(scipy_val), fmt(truth, 35),
                               f"error {mp.nstr(abs(to_mp(sym) - truth) / abs(truth) if truth else 0, 3)} rel")
            if scipy_val is None or not math.isfinite(scipy_val):
                if scipy_val is not None and not math.isfinite(scipy_val):
                    return Verdict(row.id, output, "SCIPY_WRONG", sv, fmt(scipy_val), fmt(truth, 25),
                                   "scipy nonfinite")
                return Verdict(row.id, output, "AGREE", sv, "", fmt(truth, 25))
            sci_truth = truth
            rel_err = abs(mp.mpf(scipy_val) - truth) / abs(truth) if truth != 0 else abs(mp.mpf(scipy_val))
            if rel_err > mp.mpf("1e-12") and mp_at_float is not None:
                try:
                    with mp.workdps(30):
                        st = mp_at_float()
                    if st is not None and not isinstance(st, str):
                        sci_truth = to_mp(st)
                        rel_err = (abs(mp.mpf(scipy_val) - sci_truth) / abs(sci_truth) if sci_truth != 0
                                   else abs(mp.mpf(scipy_val)))
                except (Timeout, ZeroDivisionError, ValueError, OverflowError):
                    pass
            if rel_err <= mp.mpf("1e-12") or abs(mp.mpf(scipy_val) - sci_truth) <= mp.mpf("1e-300"):
                return Verdict(row.id, output, "AGREE", sv, fmt(scipy_val), fmt(truth, 25))
            cls = "SCIPY_WRONG" if rel_err > mp.mpf("1e-9") else "SCIPY_INACCURATE"
            return Verdict(row.id, output, cls, sv, fmt(scipy_val), fmt(sci_truth, 25),
                           f"scipy error {mp.nstr(rel_err, 3)} rel")
    if scipy_val is not None and math.isfinite(scipy_val):
        if close(sym, F(scipy_val) if scipy_val == scipy_val else 0, mp.mpf("1e-10"), mp.mpf("1e-300")):
            return Verdict(row.id, output, "AGREE_SCIPY", sv, fmt(scipy_val), "",
                           "" if mp_exact is None else f"mpmath unsure ({agree} digits)")
        return Verdict(row.id, output, "UNRESOLVED", sv, fmt(scipy_val), fmt(truth),
                       "no mpmath reference" if mp_exact is None else f"mpmath unsure ({agree} digits)")
    return Verdict(row.id, output, "UNCHECKED", sv, fmt(scipy_val), fmt(truth))


def judge_nonfinite(row, output, val, scipy_val, mp_exact):
    """symplex says INF / -INF / NAN."""
    want = {"INF": "inf", "-INF": "-inf", "NAN": "nan"}[val]
    truth = None
    if mp_exact is not None:
        try:
            with mp.workdps(DPS[0]):
                truth = mp_exact()
        except (Timeout, ZeroDivisionError, ValueError, OverflowError):
            truth = None
    sci = None if scipy_val is None else ("nan" if math.isnan(scipy_val) else "inf" if scipy_val == math.inf
                                          else "-inf" if scipy_val == -math.inf else "finite")
    if isinstance(truth, str):
        if truth != want:
            return Verdict(row.id, output, "SYMPLEX_WRONG", val, fmt(scipy_val), truth)
        if sci is not None and sci != want:
            return Verdict(row.id, output, "SCIPY_WRONG", val, fmt(scipy_val), truth, "nonfinite kind")
        return Verdict(row.id, output, "NONFINITE", val, fmt(scipy_val), truth)
    if truth is not None:
        return Verdict(row.id, output, "SYMPLEX_WRONG", val, fmt(scipy_val), fmt(truth, 25))
    if sci == want:
        return Verdict(row.id, output, "NONFINITE", val, fmt(scipy_val))
    return Verdict(row.id, output, "UNRESOLVED", val, fmt(scipy_val), "", "no mpmath reference")


def judge_refusal(row, output, val, scipy_val, mp_exact):
    """symplex returned ERR / NONE: fine when the value is nonfinite or undefined, else REFUSED."""
    truth = None
    if mp_exact is not None:
        try:
            with mp.workdps(DPS[0]):
                truth = mp_exact()
        except (Timeout, ZeroDivisionError, ValueError, OverflowError):
            truth = None
    if isinstance(truth, str) or (truth is None and (scipy_val is None or not math.isfinite(scipy_val))):
        return Verdict(row.id, output, "NONFINITE", val[:120], fmt(scipy_val), truth or "")
    if truth is not None:
        t = abs(to_mp(truth))
        # eval_decimal refuses a value beyond the arbitrary-precision exponent range (documented).
        if t != 0 and abs(mp.log(t, 2)) > 2.1e9:
            return Verdict(row.id, output, "OUT_OF_RANGE", val[:120], fmt(scipy_val), fmt(truth, 20))
    return Verdict(row.id, output, "REFUSED", val[:160], fmt(scipy_val), fmt(truth, 20))


def parse_sym(val):
    """(kind, value): D -> Fraction or (re, im); F -> float; C -> complex; Q -> Fraction."""
    k = sym_kind(val)
    if k == "D":
        body = val[2:]
        if "i" in body or "I" in body:
            return "DC", parse_complex_decimal(body)
        return "D", F(body)
    if k == "F":
        return "F", float(val[2:])
    if k == "C":
        re_, im = val[2:].split(",")
        return "C", complex(float(re_), float(im))
    if k == "Q":
        n, d = val[2:].split("/")
        return "Q", F(int(n), int(d))
    return k, None


def tol_decimal(_sym):
    return digits_tol


def tol_rel(r, floor_abs=mp.mpf("1e-320")):
    return lambda truth: r * abs(truth) + floor_abs


def tol_kernel(truth):
    """The f64 kernels: ≈1e-15, up to 3e-14 for a tail ≥ 1e-10, ≈8ε|ln P| below (numdist's table)."""
    t = abs(truth)
    lg = abs(mp.log(t)) if t > 0 else mp.mpf(0)
    return max(mp.mpf("5e-14"), 10 * mp.mpf(2) ** -52 * lg) * t + mp.mpf("1e-320")


def judge_value(row, output, val, scipy_val, mp_exact, mp_float=None, f64_tol=None):
    kind, v = parse_sym(val)
    if kind in ("D", "Q"):
        tol = digits_tol if kind == "D" else (lambda truth: mp.mpf(0))
        return judge_scalar(row, output, v, tol, scipy_val, mp_exact, mp_float)
    if kind == "F":
        if math.isnan(v):
            return judge_nonfinite(row, output, "NAN", scipy_val, mp_exact)
        if math.isinf(v):
            # ±∞ is the correctly rounded double of a finite value beyond f64::MAX.
            truth = None
            if mp_exact is not None:
                try:
                    with mp.workdps(DPS[0]):
                        truth = mp_exact()
                except (Timeout, ZeroDivisionError, ValueError, OverflowError):
                    truth = None
            if truth is not None and not isinstance(truth, str):
                t = to_mp(truth)
                ok = abs(t) > mp.mpf("1.7976931348623157e308") and (t > 0) == (v > 0)
                return Verdict(row.id, output, "AGREE" if ok else "SYMPLEX_WRONG", val, fmt(scipy_val),
                               fmt(t, 20), "rounds to an infinite double" if ok else "")
            return judge_nonfinite(row, output, "INF" if v > 0 else "-INF", scipy_val, mp_exact)
        tol = f64_tol or tol_rel(mp.mpf("1e-15"))
        verdict = judge_scalar(row, output, F(v), tol, scipy_val, mp_exact, mp_float)
        # A double below the smallest subnormal (or rounding to it) is right within one subnormal.
        if verdict.cls == "SYMPLEX_WRONG" and verdict.mp and abs(v) <= 1e-300:
            try:
                tv = mp.mpf(verdict.mp)
                if abs(mp.mpf(v) - tv) <= mp.mpf("5e-324") * 2:
                    verdict.cls = "AGREE"
            except (ValueError, TypeError):
                pass
        return verdict
    if kind in ("INF", "-INF", "NAN"):
        return judge_nonfinite(row, output, kind, scipy_val, mp_exact)
    if kind in ("NONE", "ERR"):
        return judge_refusal(row, output, val, scipy_val, mp_exact)
    if kind == "OFF":
        return Verdict(row.id, output, "OFFSUPPORT", val)
    if kind in ("DC", "C"):
        return judge_complex(row, output, kind, v, mp_exact)
    return Verdict(row.id, output, kind if kind in FAILURES else "BADROW", val[:160])


def judge_complex(row, output, kind, v, mp_exact):
    if mp_exact is None:
        return Verdict(row.id, output, "UNCHECKED", str(v))
    try:
        truth, agree = mp_twice(mp_exact)
    except (Timeout, ZeroDivisionError, ValueError, OverflowError):
        truth, agree = None, 0
    if isinstance(truth, str) or truth is None or agree < 33:
        return Verdict(row.id, output, "UNCHECKED" if truth is None else "UNRESOLVED", str(v), "", str(truth))
    with mp.workdps(DPS[1]):
        truth = mp.mpc(truth)
        mag = abs(truth)
        if kind == "DC":
            re_, im = v
            parts = [(re_, truth.real), (im, truth.imag)]
            ok = True
            for s, t in parts:
                if s is None:
                    ok &= abs(t) <= mag * mp.mpf(10) ** -29
                else:
                    ok &= abs(refs.M(s) - t) <= digits_tol(t) + mag * mp.mpf(10) ** -29
        else:
            z = mp.mpc(v.real, v.imag)
            ok = abs(z - truth) <= mp.mpf("1e-15") * mag + mp.mpf("1e-300")
            for s, t in ((v.real, truth.real), (v.imag, truth.imag)):
                ok &= abs(mp.mpf(s) - t) <= mp.mpf("1e-15") * max(abs(t), mag * mp.mpf("1e-16")) + mp.mpf("1e-300")
    cls = "AGREE" if ok else "SYMPLEX_WRONG"
    return Verdict(row.id, output, cls, str(v), "", mp.nstr(truth, 25))


# ───────────────────────────── per kind ─────────────────────────────

def float_params(fam, P):
    """The parameters scipy is given: each rounded to a double (integers stay exact)."""
    if fam == "finite":
        return [(F(float(v)), F(float(p))) for v, p in P]
    return [F(float(v)) if v.denominator != 1 else v for v in P]


def dist_refs(row):
    """(scipy value, mp truth at the exact input, mp truth at scipy's input (rounded parameters and
    argument; None when nothing rounds), f64 tolerance) per output."""
    fam, P0, m = row.target, refs.params_of(row), row.method
    disc = fam in refs.DISC
    try:
        sci = refs.scipy_dist_ref(row, None)
    except Exception:
        sci = None
    exact_arg = refs.frac(row.arg) if m in ("pdf", "cdf", "sf", "quantile", "mgf", "cf") else None
    float_arg = F(float(exact_arg)) if exact_arg is not None else None
    Pf = float_params(fam, P0)
    rounds = Pf != P0 or float_arg != exact_arg

    def at(arg, P=P0):
        if m == "pdf":
            if disc:
                if arg.denominator != 1:
                    return lambda: refs.M(F(0))
                return lambda: to_mp(refs.disc_pmf(fam, P, int(arg)))
            return lambda: refs.cont_pdf(fam, P, arg)
        if m in ("cdf", "sf"):
            i = 0 if m == "cdf" else 1
            if disc:
                return lambda: to_mp(refs.disc_cdf_sf(fam, P, arg)[i])
            return lambda: refs.cont_cdf_sf(fam, P, arg)[i]
        if m == "quantile":
            return lambda: refs.closed_quantile(fam, P, arg)
        return None

    if m in ("pdf", "cdf", "sf", "quantile"):
        return sci, at(exact_arg), (at(float_arg, Pf) if rounds else None), None
    if m == "median":
        return (sci, (lambda: refs.closed_quantile(fam, P0, F(1, 2))),
                (lambda: refs.closed_quantile(fam, Pf, F(1, 2))) if rounds else None, None)
    if m in ("mean", "variance", "skewness", "kurtosis"):
        i = ("mean", "variance", "skewness", "kurtosis").index(m)
        return (sci, (lambda: refs.moments(fam, P0)[i]),
                (lambda: refs.moments(fam, Pf)[i]) if rounds else None, None)
    if m == "entropy":
        return sci, (lambda: refs.entropy(fam, P0)), (lambda: refs.entropy(fam, Pf)) if rounds else None, None
    P = P0
    if m == "mgf":
        return None, (lambda: refs.mgf(fam, P, refs.M(exact_arg))), None, None
    if m == "cf":
        return None, (lambda: refs.mgf(fam, P, mp.mpc(0, refs.M(exact_arg)))), None, None
    return sci, None, None, None


def quantile_verdict(row, output, val):
    """quantile_f64 / isf_f64: the float level is exact; check the answer against the tail itself."""
    fam, P = row.target, refs.params_of(row)
    upper = row.method == "isf_f64"
    lev = F(float(row.arg))
    try:
        sci = refs.scipy_dist_ref(row, None)
    except Exception:
        sci = None
    kind, v = parse_sym(val)
    if kind != "F":
        return judge_refusal(row, output, val, sci, None)
    x = F(v) if math.isfinite(v) else None
    disc = fam in refs.DISC
    with mp.workdps(DPS[0]):
        if disc:
            if x is None:
                return Verdict(row.id, output, "SYMPLEX_WRONG", val, fmt(sci), "", "nonfinite lattice quantile")
            def tails(k):
                return refs.disc_cdf_sf(fam, P, k) if fam != "finite" else refs.disc_cdf_sf(fam, P, k)
            # previous atom (a listed value is returned as its nearest double)
            if fam == "finite":
                atoms = sorted(a for a, _ in P)
                near = min(atoms, key=lambda a: abs(a - x))
                if float(near) == v:
                    x = near
                prev = max((a for a in atoms if a < x), default=None)
            else:
                prev = x - 1
            c, s = (to_mp(t) for t in tails(x))
            levm = refs.M(lev)
            if upper:
                ok = s <= levm and (prev is None or to_mp(tails(prev)[1]) > levm)
            else:
                ok = c >= levm and (prev is None or to_mp(tails(prev)[0]) < levm)
            scipy_ok = sci is not None and sci == v
            cls = "AGREE" if ok and (sci is None or scipy_ok or not math.isfinite(sci)) else (
                "SYMPLEX_WRONG" if not ok else "SCIPY_WRONG")
            if ok and sci is not None and math.isfinite(sci) and not scipy_ok:
                # is scipy's answer also a valid quantile (it cannot be, the quantile is unique)?
                cls = "SCIPY_WRONG"
            return Verdict(row.id, output, cls, val, fmt(sci), "", "lattice condition" + ("" if ok else " fails"))
        # continuous: the root of the tail at the exact level, near symplex's answer
        tail = (lambda u: refs.cont_cdf_sf(fam, P, u)[1]) if upper else (lambda u: refs.cont_cdf_sf(fam, P, u)[0])
        if x is None or x == 0 or v == 5e-324:
            t = tail(F(5e-324)) if v == 5e-324 else None
            if v == 5e-324 and t is not None and ((upper and t <= refs.M(lev)) or (not upper and t >= refs.M(lev))):
                return Verdict(row.id, output, "AGREE", val, fmt(sci), "", "smallest positive double")
            return Verdict(row.id, output, "UNCHECKED", val, fmt(sci))
        try:
            root = find_root(tail, refs.M(lev), refs.M(x), increasing=not upper)
            if root is not None:
                with mp.workdps(DPS[1]):
                    root2 = find_root(tail, refs.M(lev), root, increasing=not upper)
                if root2 is None or abs(root2 - root) > abs(root2) * mp.mpf(10) ** -25:
                    root = None
        except (Timeout, ZeroDivisionError, ValueError, OverflowError, mp.libmp.NoConvergence):
            root = None
        if root is None:
            return Verdict(row.id, output, "UNRESOLVED" if sci is None or abs(sci - v) > 1e-10 * abs(v) else
                           "AGREE_SCIPY", val, fmt(sci), "", "no root found")
        # condition: |d ln x / d ln P| = P / (x f(x))
        try:
            dens = refs.cont_pdf(fam, P, F(mp.nstr(root, 30)))
            kappa = abs(tail(F(mp.nstr(root, 30))) / (root * dens)) if dens else mp.mpf(1)
        except (ZeroDivisionError, ValueError):
            kappa = mp.mpf(1)
        tol = mp.mpf("1e-13") * max(mp.mpf(1), kappa)
        rel = abs(refs.M(x) - root) / abs(root)
        if rel > tol:
            return Verdict(row.id, output, "SYMPLEX_WRONG", val, fmt(sci), mp.nstr(root, 20),
                           f"error {mp.nstr(rel, 3)} rel (tolerance {mp.nstr(tol, 2)})")
        if sci is not None and math.isfinite(sci) and sci != 0:
            srel = abs(mp.mpf(sci) - root) / abs(root)
            if srel > mp.mpf("1e-9") and srel > 100 * tol:
                return Verdict(row.id, output, "SCIPY_WRONG", val, fmt(sci), mp.nstr(root, 20),
                               f"scipy error {mp.nstr(srel, 3)} rel")
        elif sci is not None and not math.isfinite(sci):
            return Verdict(row.id, output, "SCIPY_WRONG", val, fmt(sci), mp.nstr(root, 20), "scipy nonfinite")
        return Verdict(row.id, output, "AGREE", val, fmt(sci), mp.nstr(root, 20))


def find_root(tail, level, x0, increasing):
    """The u with tail(u) = level (tail monotone), by bracketing ln tail − ln level around x0."""
    def g(u):
        t = tail(F(mp.nstr(u, mp.mp.dps + 5)))
        if t <= 0:
            return -mp.inf if increasing else mp.inf
        return mp.log(t) - mp.log(level)
    # Secant steps on g from x0 (the answer under test, usually right to ~1e-15), then a bracket
    # check around the result; bisection only when the steps do not settle.
    u0, u1 = x0, x0 * (1 + mp.mpf("1e-12"))
    g0, g1 = g(u0), g(u1)
    for _ in range(30):
        if not (mp.isfinite(g0) and mp.isfinite(g1)) or g1 == g0:
            break
        u2 = u1 - g1 * (u1 - u0) / (g1 - g0)
        u0, g0, u1 = u1, g1, u2
        g1 = g(u1)
        if g1 == 0 or abs(u1 - u0) <= abs(u1) * mp.mpf(10) ** (-mp.mp.dps + 6):
            d = abs(u1) * mp.mpf(10) ** (-mp.mp.dps + 10)
            ga, gb = g(u1 - d), g(u1 + d)
            if (ga <= 0 <= gb) or (gb <= 0 <= ga):
                return u1
            break
    lo, hi = x0 * (1 - mp.mpf("1e-9")), x0 * (1 + mp.mpf("1e-9"))
    if lo > hi:
        lo, hi = hi, lo
    for _ in range(80):
        glo, ghi = g(lo), g(hi)
        if glo == 0:
            return lo
        if ghi == 0:
            return hi
        if (glo < 0) != (ghi < 0):
            break
        w = hi - lo
        lo, hi = lo - w * 4, hi + w * 4
    else:
        return None
    for _ in range(400):
        mid = (lo + hi) / 2
        gm = g(mid)
        if gm == 0 or hi - lo <= abs(mid) * mp.mpf(10) ** (-mp.mp.dps + 8):
            return mid
        if (gm < 0) == (glo < 0):
            lo, glo = mid, gm
        else:
            hi = mid
    return (lo + hi) / 2


def judge_dist(row, outs):
    verdicts = []
    shared = None
    for output, val in outs.items():
        if row.method in ("quantile_f64", "isf_f64"):
            verdicts.append(quantile_verdict(row, output, val))
            continue
        if shared is None:
            sci, mp_exact, mp_float, _ = dist_refs(row)
            shared = (sci, memo_dps(mp_exact), memo_dps(mp_float))
        sci, mp_exact, mp_float = shared
        if output == "c64" or (output == "d30" and row.method == "cf"):
            verdicts.append(judge_value(row, output, val, None, mp_exact))
            continue
        verdicts.append(judge_value(row, output, val, sci, mp_exact, mp_float))
    return verdicts


def kernel_mp(row):
    a = [F(float(s)) for s in row.params.split(",")]
    t = row.target
    if t == "betainc_regularized": return lambda: refs.ibeta(a[0], a[1], a[2])
    if t == "gammainc_lower_regularized": return lambda: refs.gammp(a[0], a[1])
    if t == "gammainc_upper_regularized": return lambda: refs.gammq(a[0], a[1])
    fam, fn = t.split(".")
    famname = {"norm": "normal", "t": "student_t", "gamma": "gamma", "chi2": "chi_squared", "beta": "beta",
               "f": "f", "binom": "binomial", "poisson": "poisson"}[fam]
    P = {"norm": [F(0), F(1)], "t": a[1:2], "gamma": a[1:3], "chi2": a[1:2], "beta": a[1:3], "f": a[1:3],
         "binom": a[1:3], "poisson": a[1:2]}[fam]
    return famname, P, fn, a[0]


def judge_kernel(row, outs):
    val = outs.get("f64", "")
    try:
        sci = refs.scipy_kernel_ref(row)
    except Exception:
        sci = None
    k = kernel_mp(row)
    if callable(k):
        return [judge_value(row, "f64", val, sci, k, None, tol_kernel)]
    fam, P, fn, x = k
    if fn in ("ppf", "isf"):
        fake = Row(row.id, "dist", fam, ",".join(str(p) for p in P), "quantile_f64" if fn == "ppf" else "isf_f64",
                   repr(float(x)))
        if fam == "binomial":
            fake.params = f"{int(P[0])},{P[1]}"
        v = quantile_verdict(fake, "f64", val)
        v.id = row.id
        return [v]
    if fam in refs.DISC:
        i = {"cdf": 0, "sf": 1}.get(fn)
        if i is None:
            return [Verdict(row.id, "f64", "UNCHECKED", val)]
        if fam == "binomial":
            P = [F(int(P[0])), P[1]]
        mp_exact = lambda: to_mp(refs.disc_cdf_sf(fam, P, x)[i])
    elif fn == "pdf":
        mp_exact = lambda: refs.cont_pdf(fam, P, x)
    else:
        i = 0 if fn == "cdf" else 1
        mp_exact = lambda: refs.cont_cdf_sf(fam, P, x)[i]
    return [judge_value(row, "f64", val, sci, mp_exact, None, tol_kernel)]


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [F(0)] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = F(i + j + 2, 2)
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def tie_sum(xs):
    c = Counter(xs)
    return sum(t ** 3 - t for t in c.values())


def pearson_r(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return refs.M(sxy) / mp.sqrt(refs.M(sxx * syy))


def var_unbiased(x):
    n = len(x)
    m = sum(x) / n
    return sum((a - m) ** 2 for a in x) / (n - 1)


def hyper_two_sided(probs, k):
    pk = probs[k]
    return min(F(1), sum((p for p in probs.values() if p <= pk), F(0)))


def test_mp(row):
    """{output: function giving the mpmath / exact value} for a test row."""
    f = refs.fields(row.params)
    alt = f.get("alt", "two-sided")
    t = row.target
    xs = lambda k: refs.qlist(f[k])
    out = {}
    if t in ("t_test_one_sample", "t_test_paired"):
        if t == "t_test_one_sample":
            d, mu0 = xs("x"), refs.frac(f["mu0"])
        else:
            d, mu0 = [a - b for a, b in zip(xs("x"), xs("y"))], F(0)
        n = len(d)
        stat = lambda: refs.M(sum(d) / n - mu0) / mp.sqrt(refs.M(var_unbiased(d) / n))
        out["statistic"] = stat
        out["df"] = lambda: F(n - 1)
        out["pvalue"] = lambda: refs.p_tail_t(stat(), n - 1, alt)
    elif t == "t_test_two_sample":
        x, y = xs("x"), xs("y")
        n1, n2 = len(x), len(y)
        v1, v2 = var_unbiased(x), var_unbiased(y)
        diff = sum(x) / n1 - sum(y) / n2
        if f["equal_var"] == "true":
            df = F(n1 + n2 - 2)
            sp = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
            se2 = sp * (F(1, n1) + F(1, n2))
        else:
            se2 = v1 / n1 + v2 / n2
            df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        stat = lambda: refs.M(diff) / mp.sqrt(refs.M(se2))
        out["statistic"] = stat
        out["df"] = lambda: df
        out["pvalue"] = lambda: refs.p_tail_t(stat(), refs.M(df), alt)
    elif t == "binomial_test":
        k, n, p0 = int(f["k"]), int(f["n"]), refs.frac(f["p0"])
        def pv():
            probs = {i: F(math.comb(n, i)) * p0 ** i * (1 - p0) ** (n - i) for i in range(n + 1)}
            if alt == "less": return sum((probs[i] for i in range(k + 1)), F(0))
            if alt == "greater": return sum((probs[i] for i in range(k, n + 1)), F(0))
            return hyper_two_sided(probs, k)
        out["statistic"] = lambda: F(k, n)
        if n <= 1000:
            out["pvalue"] = pv
    elif t == "fisher_exact":
        tab = [[int(c) for c in rr.split(",")] for rr in f["table"].split(":")]
        (a, b), (c, d) = tab
        r1, c1, N = a + b, a + c, a + b + c + d
        lo, hi = max(0, r1 + c1 - N), min(r1, c1)
        out["statistic"] = (lambda: "inf") if b * c == 0 and a * d > 0 else (
            (lambda: "nan") if b * c == 0 else (lambda: F(a * d, b * c)))
        if hi - lo + 1 <= 2000:
            def pv():
                tot = math.comb(N, c1)
                probs = {i: F(math.comb(r1, i) * math.comb(N - r1, c1 - i), tot) for i in range(lo, hi + 1)}
                if alt == "less": return sum((probs[i] for i in range(lo, a + 1)), F(0))
                if alt == "greater": return sum((probs[i] for i in range(a, hi + 1)), F(0))
                return hyper_two_sided(probs, a)
            out["pvalue"] = pv
    elif t in ("chi_square_independence", "chi_square_goodness_of_fit"):
        if t == "chi_square_goodness_of_fit":
            obs = xs("observed")
            e = sum(obs) / len(obs)
            stat = sum((o - e) ** 2 / e for o in obs)
            df = len(obs) - 1
        else:
            tab = [[refs.frac(c) for c in rr.split(",")] for rr in f["table"].split(":")]
            rs = [sum(r) for r in tab]
            cs = [sum(col) for col in zip(*tab)]
            N = sum(rs)
            df = (len(rs) - 1) * (len(cs) - 1)
            stat = F(0)
            for i, r in enumerate(tab):
                for j, o in enumerate(r):
                    e = rs[i] * cs[j] / N
                    dev = abs(o - e)
                    if f["correction"] == "true" and df == 1:
                        dev = max(F(0), dev - F(1, 2))
                    stat += dev ** 2 / e
        out["statistic"] = lambda: stat
        out["df"] = lambda: F(df)
        out["pvalue"] = lambda: refs.gammq(F(df, 2), stat / 2)
    elif t == "z_test_proportion":
        k, n, p0 = int(f["k"]), int(f["n"]), refs.frac(f["p0"])
        stat = lambda: refs.M(F(k, n) - p0) / mp.sqrt(refs.M(p0 * (1 - p0) / n))
        out["statistic"] = stat
        out["pvalue"] = lambda: refs.p_tail_z(stat(), alt)
    elif t in ("pearson_test", "spearman_test"):
        x, y = xs("x"), xs("y")
        if t == "spearman_test":
            x, y = ranks(x), ranks(y)
        n = len(x)
        r = lambda: pearson_r(x, y)
        out["statistic"] = r
        def pv():
            rr = r()
            if abs(rr) >= 1:
                return mp.mpf(0) if alt == "two-sided" else None
            tt = rr * mp.sqrt((n - 2) / (1 - rr * rr))
            return refs.p_tail_t(tt, n - 2, alt)
        out["pvalue"] = pv
    elif t == "mann_whitney_u":
        x, y = xs("x"), xs("y")
        n1, n2 = len(x), len(y)
        rk = ranks(x + y)
        u1 = sum(rk[:n1]) - F(n1 * (n1 + 1), 2)
        out["statistic"] = lambda: u1
        if f["method"] != "exact":
            n = n1 + n2
            var = F(n1 * n2, 12) * ((n + 1) - F(tie_sum(x + y), n * (n - 1)))
            def pv():
                num = u1 - F(n1 * n2, 2)
                if f["method"] == "asymptotic_cc":
                    if alt == "two-sided":
                        num = num - F(1, 2) if num > 0 else (num + F(1, 2) if num < 0 else num)
                        if (u1 - F(n1 * n2, 2)) * num < 0:
                            num = F(0)
                    elif alt == "greater":
                        num -= F(1, 2)
                    else:
                        num += F(1, 2)
                if var <= 0:
                    return None
                return refs.p_tail_z(refs.M(num) / mp.sqrt(refs.M(var)), alt)
            out["pvalue"] = pv
    elif t == "wilcoxon_signed_rank":
        d = [v for v in xs("x") if v != 0]
        n = len(d)
        if n:
            rk = ranks([abs(v) for v in d])
            tp = sum(r for r, v in zip(rk, d) if v > 0)
            tm = sum(r for r, v in zip(rk, d) if v < 0)
            out["statistic"] = (lambda: min(tp, tm)) if alt == "two-sided" else (lambda: tp)
            var = (F(n * (n + 1) * (2 * n + 1)) - F(tie_sum([abs(v) for v in d]), 2)) / 24
            def pv():
                num = tp - F(n * (n + 1), 4)
                if f["method"] == "asymptotic_cc":
                    num = num - F(1, 2) * (1 if num > 0 else -1 if num < 0 else 0)
                if var <= 0:
                    return None
                return refs.p_tail_z(refs.M(num) / mp.sqrt(refs.M(var)), alt)
            out["pvalue"] = pv
    elif t == "kruskal_wallis":
        groups = [refs.qlist(g) for g in f["groups"].split(":")]
        allv = [v for g in groups for v in g]
        N = len(allv)
        rk = ranks(allv)
        pos, h = 0, F(0)
        for g in groups:
            rs = sum(rk[pos:pos + len(g)])
            h += rs * rs / len(g)
            pos += len(g)
        h = F(12, N * (N + 1)) * h - 3 * (N + 1)
        corr = 1 - F(tie_sum(allv), N ** 3 - N)
        if corr > 0:
            stat = h / corr
            out["statistic"] = lambda: stat
            out["pvalue"] = lambda: refs.gammq(F(len(groups) - 1, 2), stat / 2)
    elif t == "anova_one_way":
        groups = [refs.qlist(g) for g in f["groups"].split(":")]
        allv = [v for g in groups for v in g]
        N, k = len(allv), len(groups)
        gm = sum(allv) / N
        ssb = sum(len(g) * (sum(g) / len(g) - gm) ** 2 for g in groups)
        ssw = sum(sum((v - sum(g) / len(g)) ** 2 for v in g) for g in groups)
        if ssw > 0 and N > k:
            fstat = (ssb / (k - 1)) / (ssw / (N - k))
            d1, d2 = k - 1, N - k
            out["statistic"] = lambda: fstat
            out["pvalue"] = lambda: refs.ibeta(F(d2, 2), F(d1, 2), F(d2) / (d2 + d1 * fstat))
    return out


def fisher_numeric(row):
    f = refs.fields(row.params)
    tab = [[int(c) for c in rr.split(",")] for rr in f["table"].split(":")]
    (a, b), (c, d) = tab
    r1, c1, N = a + b, a + c, a + b + c + d
    return min(r1, c1) - max(0, r1 + c1 - N) + 1 > 2000


def judge_test(row, outs):
    try:
        sci = refs.scipy_test_ref(row)
    except Exception as e:
        sci = {}
    mps = test_mp(row)
    # scipy / statsmodels get the null proportion as a double.
    f = refs.fields(row.params)
    mps_float = {}
    if "p0" in f and F(float(refs.frac(f["p0"]))) != refs.frac(f["p0"]):
        frow = Row(row.id, row.kind, row.target,
                   row.params.replace(f"p0={f['p0']}", f"p0={refs.q_of(F(float(refs.frac(f['p0']))))}"),
                   row.method, row.arg)
        mps_float = test_mp(frow)
    verdicts = []
    for output, val in outs.items():
        ref_out = "pvalue" if output == "pvalue_f64" else output
        s = sci.get(ref_out)
        s = None if s is None else float(s)
        mp_fn = mps.get(ref_out)
        f64_tol = None
        if output == "pvalue_f64":
            f64_tol = tol_rel(mp.mpf("1e-14"))
        if row.target == "fisher_exact" and output == "pvalue" and fisher_numeric(row):
            # Above 2,000 support points the p-value is an f64 walk (documented), not exact.
            val = "F:" + val[2:] if val.startswith("D:") else val
            f64_tol = tol_rel(mp.mpf("1e-12"))
        verdicts.append(judge_value(row, output, val, s, mp_fn, mps_float.get(ref_out), f64_tol))
    return verdicts


def solve_exact(X, y):
    """β = (XᵀX)⁻¹Xᵀy and (XᵀX)⁻¹ in exact rationals."""
    p = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(p)] for i in range(p)]
    b = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(p)]
    aug = [A[i] + [b[i]] + [F(int(i == j)) for j in range(p)] for i in range(p)]
    for col in range(p):
        piv = next((r for r in range(col, p) if aug[r][col] != 0), None)
        if piv is None:
            return None, None
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(p):
            if r != col and aug[r][col] != 0:
                fac = aug[r][col]
                aug[r] = [v - fac * w for v, w in zip(aug[r], aug[col])]
    beta = [aug[i][p] for i in range(p)]
    inv = [aug[i][p + 1:] for i in range(p)]
    return beta, inv


def model_mp(row):
    y, X = refs.model_design(row)
    beta, inv = solve_exact(X, y)
    if beta is None:
        return {}
    n, p = len(y), len(beta)
    fitted = [sum(b * v for b, v in zip(beta, r)) for r in X]
    ssr = sum((a - b) ** 2 for a, b in zip(y, fitted))
    ybar = sum(y) / n
    tss = sum((a - ybar) ** 2 for a in y)
    dfr = n - p
    s2 = ssr / dfr
    out = {}
    for i in range(p):
        out[f"coef.{i}"] = (lambda i=i: beta[i])
        out[f"se.{i}"] = (lambda i=i: mp.sqrt(refs.M(s2 * inv[i][i])))
        tval = (lambda i=i: refs.M(beta[i]) / mp.sqrt(refs.M(s2 * inv[i][i])))
        out[f"t.{i}"] = tval
        out[f"p.{i}"] = (lambda tval=tval: refs.p_tail_t(tval(), dfr, "two-sided"))
    r2 = 1 - ssr / tss
    out["rsquared"] = lambda: r2
    out["rsquared_adj"] = lambda: 1 - F(n - 1, dfr) * (1 - r2)
    dfm = p - 1
    if dfm > 0 and ssr > 0:
        fval = ((tss - ssr) / dfm) / s2
        out["fvalue"] = lambda: fval
        out["f_pvalue"] = lambda: refs.ibeta(F(dfr, 2), F(dfm, 2), F(dfr) / (dfr + dfm * fval))
    llf = lambda: -refs.M(F(n, 2)) * (mp.log(2 * mp.pi) + mp.log(refs.M(ssr / n)) + 1)
    out["llf"] = llf
    out["aic"] = lambda: -2 * llf() + 2 * p
    out["bic"] = lambda: -2 * llf() + p * mp.log(n)
    return out


def judge_model(row, outs):
    try:
        sci = refs.statsmodels_ref(row)
    except Exception:
        sci = {}
    mps = model_mp(row)
    return [judge_value(row, o, v, sci.get(o), mps.get(o)) for o, v in outs.items()]


def judge_row(item):
    t0 = time.time()
    verdicts = judge_row_inner(item)
    return item[0].id, time.time() - t0, verdicts


def judge_row_inner(item):
    row, outs = item
    mp.mp.dps = 50
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(ORACLE_TIMEOUT)
    try:
        if not outs:
            return [Verdict(row.id, "*", "MISSING", "")]
        star = outs.get("*")
        if star is not None:
            kind = sym_kind(star)
            return [Verdict(row.id, "*", kind if kind in FAILURES else "BADROW", star[:200])]
        if row.kind == "dist":
            return judge_dist(row, outs)
        if row.kind == "kernel":
            return judge_kernel(row, outs)
        if row.kind == "test":
            return judge_test(row, outs)
        if row.kind == "model":
            return judge_model(row, outs)
        return [Verdict(row.id, "*", "BADROW", f"unknown kind {row.kind}")]
    except Timeout:
        return [Verdict(row.id, o, "UNRESOLVED", v[:120], "", "", "oracle timeout") for o, v in outs.items()]
    except Exception as e:  # an oracle bug must not hide the other rows
        return [Verdict(row.id, o, "UNRESOLVED", v[:120], "", "", f"oracle error {e!r}"[:200])
                for o, v in outs.items()]
    finally:
        signal.alarm(0)


def _worker(tasks, done):
    while True:
        item = tasks.get()
        if item is None:
            return
        done.put(("start", os.getpid(), item[0].id))
        done.put(("result", os.getpid(), judge_row(item)))


def run_pool(items, jobs, hard_limit):
    """judge_row over items in `jobs` processes.  SIGALRM bounds a row's Python code; a row stuck in one
    C-level operation (a huge integer power) is killed with its process after `hard_limit` seconds and
    recorded as an oracle timeout, and a fresh process takes over."""
    ctx = multiprocessing.get_context("fork")
    tasks, done = ctx.Queue(), ctx.Queue()
    by_id = {row.id: (row, outs) for row, outs in items}
    for item in items:
        tasks.put(item)
    workers = {}

    def spawn():
        p = ctx.Process(target=_worker, args=(tasks, done), daemon=True)
        p.start()
        workers[p.pid] = [p, None, 0.0]

    for _ in range(max(1, min(jobs, len(items)))):
        spawn()
    results, pending = [], len(items)
    import queue
    while pending:
        try:
            kind, pid, payload = done.get(timeout=5)
        except queue.Empty:
            kind = None
        if kind == "start" and pid in workers:
            workers[pid][1], workers[pid][2] = payload, time.time()
        elif kind == "result":
            results.append(payload)
            pending -= 1
            if pid in workers:
                workers[pid][1] = None
        now = time.time()
        for pid, (p, rid, t0) in list(workers.items()):
            if rid is not None and now - t0 > hard_limit:
                p.kill()
                p.join()
                del workers[pid]
                row, outs = by_id[rid]
                results.append((rid, now - t0, [Verdict(rid, o, "UNRESOLVED", v[:120], "", "", "oracle timeout")
                                              for o, v in outs.items()]))
                pending -= 1
                spawn()
            elif not p.is_alive() and rid is not None:
                del workers[pid]
                row, outs = by_id[rid]
                results.append((rid, now - t0, [Verdict(rid, o, "UNRESOLVED", v[:120], "", "", "oracle crashed")
                                              for o, v in outs.items()]))
                pending -= 1
                spawn()
    for _ in workers:
        tasks.put(None)
    for p, _, _ in workers.values():
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    return results


# ───────────────────────────── report ─────────────────────────────

def allowed(v, entries, rows):
    for e in entries:
        if e["cls"] == v.cls and fnmatch.fnmatchcase(v.id, e["id"]) and fnmatch.fnmatchcase(v.output, e["output"]):
            e["used"] += 1
            return e
    return None


def describe(row):
    if row.kind == "dist":
        a = "" if row.arg == "-" else f"({row.arg})"
        return f"{row.target}({row.params}).{row.method}{a}"
    return f"{row.target}({row.params})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--allowlist", type=Path)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, help="write every verdict as TSV")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    t0 = time.time()
    rows = read_cases(args.cases)
    dump, ms = read_dump(args.dump)
    entries = read_allowlist(args.allowlist)
    items = [(rows[i], dump.get(i, {})) for i in rows]
    results = run_pool(items, args.jobs, ORACLE_TIMEOUT + 30)
    order = {i: n for n, i in enumerate(rows)}
    results.sort(key=lambda r: order[r[0]])
    verdicts = [v for _, _, vs in results for v in vs]
    oracle_time = sorted(((t, i) for i, t, _ in results), reverse=True)
    extra = sorted(set(dump) - set(rows))
    counts = Counter()
    by_kind = defaultdict(Counter)
    failures, known, new_scipy, inaccurate = [], [], [], []
    for v in verdicts:
        row = rows[v.id]
        counts[v.cls] += 1
        by_kind[row.kind][v.cls] += 1
        if v.cls in FAILURES:
            e = allowed(v, entries, rows)
            if e:
                known.append((v, e))
            else:
                failures.append(v)
        elif v.cls == "SCIPY_INACCURATE":
            inaccurate.append(v)
    stale = [e for e in entries if e["used"] == 0]
    lines = ["# Statistics oracle", ""]
    status = "FAILED" if failures else "passed"
    lines.append(f"**{status}**: {len(rows)} cases, {len(verdicts)} values, {len(failures)} unexplained "
                 f"failures, {len(known)} allowlisted, {len(stale)} stale allowlist entries "
                 f"({time.time() - t0:.0f} s).")
    lines += ["", "| kind | " + " | ".join(sorted(counts)) + " |", "|---|" + "---|" * len(counts)]
    for kind in sorted(by_kind):
        lines.append(f"| {kind} | " + " | ".join(str(by_kind[kind][c]) for c in sorted(counts)) + " |")

    def table(title, vs, with_reason=False):
        if not vs:
            return
        lines.extend(["", f"## {title} ({len(vs)})", "",
                      "| class | id | output | case | symplex | scipy | mpmath | note |" + (" reason |" if with_reason else ""),
                      "|---|---|---|---|---|---|---|---|" + ("---|" if with_reason else "")])
        for item in vs[:300]:
            v, e = item if isinstance(item, tuple) else (item, None)
            cells = [v.cls, v.id, v.output, describe(rows[v.id])[:120], v.sym[:60], v.scipy[:30], v.mp[:40], v.note[:80]]
            if with_reason:
                cells.append(e["reason"][:160])
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        if len(vs) > 300:
            lines.append(f"| … {len(vs) - 300} more | | | | | | | |")

    table("Failures", sorted(failures, key=lambda v: (v.cls, v.id)))
    table("Allowlisted", sorted(known, key=lambda t: (t[0].cls, t[0].id)), with_reason=True)
    table("scipy inaccuracies (1e-12 .. 1e-9, not failures)", inaccurate)
    if stale:
        lines += ["", f"## Stale allowlist entries ({len(stale)})", ""]
        lines += [f"- line {e['line']}: `{e['id']}` `{e['output']}` {e['cls']} — {e['reason']}" for e in stale]
    if extra:
        lines += ["", f"## Dump rows without a case ({len(extra)})", ""] + [f"- `{i}`" for i in extra[:50]]
    slow = sorted(ms.items(), key=lambda kv: -kv[1])[:10]
    lines += ["", "## Slowest rows (symplex, ms)", ""] + [f"- {t} ms `{i}` {describe(rows[i])[:100]}"
                                                         for i, t in slow if i in rows]
    lines += ["", f"## Slowest rows (references, s; {sum(t for t, _ in oracle_time):.0f} s in all)", ""]
    lines += [f"- {t:.1f} s `{i}` {describe(rows[i])[:100]}" for t, i in oracle_time[:10]]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.verdicts:
        with open(args.verdicts, "w", encoding="utf-8") as fh:
            for v in verdicts:
                fh.write("\t".join([v.cls, v.id, v.output, v.sym, v.scipy, v.mp, v.note]) + "\n")
    print("\n".join(lines[:4 + len(by_kind) + 3]))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
