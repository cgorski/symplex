#!/usr/bin/env python3
"""Generate the symplex 0.3 SymPy cross-validation fixtures.

Output: ``tests/fixtures/v03_cross_validation.json`` (path derived from the
location of this script), consumed by ``tests/v03_oracle_*.rs`` — one test
file per area, one ``#[test]`` per ``(category, subcategory)``:

    poly        as_dict, degree, LC, all_coeffs, eval, nroots,
                degree_symbolic, coeff_symbolic          → v03_oracle_poly.rs
    ratsimp     cancel                                   → v03_oracle_poly.rs
    linprog     lpmax, lpmin, infeasible, unbounded,
                feasible_nonneg                          → v03_oracle_linprog.rs
    normalforms hnf, hnf_row, snf, nullspace_rank,
                lattice_det, igcd_ilcm                   → v03_oracle_normalforms.rs
    optimize    polyfit_exact, brent_root                → v03_oracle_poly.rs

Design rules (see tests/README.md, identical to generate_v02_fixtures.py):

* **Deterministic.**  Fixed seed, no timestamps, ``sort_keys=True``; running
  the script twice yields byte-identical output.
* **Bounded.**  Every SymPy computation runs under a per-fixture wall-clock
  timeout.  A fixture whose oracle computation times out is *kept* with
  ``"sympy_timeout": true``; one whose computation raises is kept with
  ``"sympy_error"``.  Nothing is silently dropped.
* **Exact where the library is exact.**  Rationals are stored as strings
  (``"-3/4"``), integers as strings, so the Rust consumers can compare with
  ``Ratio<BigInt>`` / ``BigInt`` equality rather than ``f64`` tolerances.
  Symbolic coefficients are stored as ``sstr`` plus numeric values at fixed
  parameter points.
* **Every fixture has a stable ``key``** (unique within category:subcategory)
  so the Rust consumers can attach ``// BUG:`` annotations that survive
  regeneration.

Oracle conventions that differ from symplex and are handled by the consumers:

* SymPy's ``hermite_normal_form`` is column-style and drops zero columns;
  symplex's ``column_hermite_normal_form`` keeps them.  The ``hnf_row``
  reference is derived from SymPy's column HNF by the reverse/transpose
  identity documented in ``symplex::normalforms``.
* LP optima may be attained at several vertices: the objective value is the
  reference, the returned point is checked for feasibility and optimality.
* The LP reference is ``lpmin``/``lpmax`` with explicit relational
  constraints.  SymPy 1.14's ``linprog`` mishandles non-default ``bounds``
  (see ``sympy_linprog``) and needs a (possibly trivial) ``A``/``b`` when
  ``A_eq`` is given; it is only used as a secondary check with default
  bounds, where a ``0·x ≤ 1`` row is passed if there are no ``≤`` rows.

Usage::

    /Users/chris.gorski/repos/math/symplex/.venv/bin/python \
        scripts/generate_v03_fixtures.py            # writes the JSON
    ... --check                                     # exit 1 if the file would change
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import signal
import sys
from collections import Counter

import sympy as sp
from sympy import Matrix, Rational, S, oo, symbols, zoo
from sympy.matrices.normalforms import hermite_normal_form, smith_normal_form
from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError, linprog, lpmax, lpmin

SEED = 20260918
PER_FIXTURE_TIMEOUT = 25.0  # seconds of wall clock per SymPy computation
N_DIGITS = 30

x, y, z, a, b = symbols("x y z a b")
SYMS = {"x": x, "y": y, "z": z, "a": a, "b": b}

RNG = random.Random(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# Infrastructure
# ═══════════════════════════════════════════════════════════════════════════


class FixtureTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise FixtureTimeout()


def run_bounded(fn, seconds=PER_FIXTURE_TIMEOUT):
    """Run ``fn()`` with a wall-clock limit.  Returns (result, status) where
    status ∈ {"ok", "timeout", "error"}; on error the result is the message."""
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(), "ok"
    except FixtureTimeout:
        return None, "timeout"
    except Exception as e:  # noqa: BLE001 — recorded in the fixture
        return f"{type(e).__name__}: {e}", "error"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def numval(expr):
    """{"re", "im"} for a finite number, {"special": ...} for extended values,
    or None if SymPy cannot produce a number."""
    if expr is None:
        return None
    if expr == oo:
        return {"special": "oo"}
    if expr == -oo:
        return {"special": "-oo"}
    if expr == zoo:
        return {"special": "zoo"}
    if expr is S.NaN:
        return {"special": "nan"}
    try:
        v = complex(sp.N(expr, N_DIGITS))
    except Exception:  # noqa: BLE001
        return None
    re_, im_ = v.real, v.imag
    if not (math.isfinite(re_) and math.isfinite(im_)):
        return None
    if abs(im_) < 1e-18 * max(1.0, abs(re_)):
        im_ = 0.0
    if abs(re_) < 1e-18 * max(1.0, abs(im_)):
        re_ = 0.0
    return {"re": re_, "im": im_}


def subs_json(d):
    return {str(kk): float(v) for kk, v in d.items()}


def qstr(r):
    """Exact rational as a string SymPy/Rust can both read: "-3/4", "5"."""
    r = sp.Rational(r)
    return str(r)


def is_finite_rational(v):
    return v.is_Rational and v.is_finite


class Collector:
    def __init__(self):
        self.fixtures = []
        self.keys = set()
        self.timeouts = 0
        self.errors = 0

    def add(self, category, subcategory, key, fields, compute):
        """Register one fixture.  ``compute()`` returns a dict of oracle
        fields; it runs under the per-fixture timeout."""
        ident = (category, subcategory, key)
        if ident in self.keys:
            raise ValueError(f"duplicate fixture key {ident}")
        self.keys.add(ident)
        fx = {"category": category, "subcategory": subcategory, "key": key}
        fx.update(fields)
        result, status = run_bounded(compute)
        if status == "timeout":
            fx["sympy_timeout"] = True
            self.timeouts += 1
            print(f"  TIMEOUT {category}:{subcategory} {key}", file=sys.stderr)
        elif status == "error":
            fx["sympy_error"] = result
            self.errors += 1
            print(f"  ERROR   {category}:{subcategory} {key}: {result}", file=sys.stderr)
        else:
            fx.update(result)
        self.fixtures.append(fx)


C = Collector()

# ── Random data helpers ────────────────────────────────────────────────────


def rint(lo, hi, nonzero=False):
    while True:
        v = RNG.randint(lo, hi)
        if v != 0 or not nonzero:
            return v


def rrat(max_num=9, max_den=6, nonzero=False):
    while True:
        v = Rational(RNG.randint(-max_num, max_num), RNG.randint(1, max_den))
        if v != 0 or not nonzero:
            return v


def rmonomial(gens, max_deg):
    """Random monomial of total degree ≤ max_deg."""
    while True:
        exps = [RNG.randint(0, max_deg) for _ in gens]
        if sum(exps) <= max_deg:
            break
    m = S.One
    for g, e in zip(gens, exps):
        m *= g**e
    return m


def rpoly(gens, max_deg, n_terms, coeff=lambda: rint(-7, 7, nonzero=True)):
    """Random non-zero polynomial with the given coefficient sampler."""
    while True:
        p = sum(coeff() * rmonomial(gens, max_deg) for _ in range(n_terms))
        p = sp.expand(p)
        if p != 0:
            return p


def rpoly_rat(gens, max_deg, n_terms):
    return rpoly(gens, max_deg, n_terms, coeff=lambda: rrat(nonzero=True))


# Parameter points for symbolic coefficients.  Every symbol that can occur
# in a coefficient (parameters a, b as well as non-generator x, y, z) gets a
# non-zero value so that 1/a-style coefficients are finite; substituting a
# symbol that does not occur is a no-op on both sides.
r_, f_, j_ = symbols("r f j")
PARAM_POINTS = [
    {a: Rational(1, 2), b: Rational(-3, 2), x: Rational(5, 3), y: Rational(-2, 7), z: Rational(11, 4), r_: Rational(3, 5), f_: Rational(1, 2), j_: Rational(-3, 2)},
    {a: Rational(7, 10), b: Rational(3, 10), x: Rational(-4, 9), y: Rational(13, 6), z: Rational(1, 8), r_: Rational(-7, 3), f_: Rational(7, 10), j_: Rational(3, 10)},
    {a: Rational(-5, 4), b: Rational(9, 4), x: Rational(2, 5), y: Rational(6, 11), z: Rational(-9, 2), r_: Rational(4, 7), f_: Rational(-5, 4), j_: Rational(9, 4)},
]


def coeff_record(c):
    """{"coeff": sstr, "rational": bool, "values": [numval at PARAM_POINTS]}."""
    c = sp.sympify(c)
    rec = {"coeff": sp.sstr(c), "rational": bool(c.is_Rational)}
    if not c.is_Rational:
        rec["values"] = [numval(c.subs(p)) for p in PARAM_POINTS]
    return rec


def mat_json(rows):
    return [[str(sp.sympify(e)) for e in row] for row in rows]


def mat_label(rows):
    return "[" + ";".join(",".join(str(e) for e in row) for row in rows) + "]"


def rand_int_matrix(m, n, lo=-9, hi=9):
    return [[rint(lo, hi) for _ in range(n)] for _ in range(m)]


def rand_low_rank_matrix(m, n, r, lo=-4, hi=4):
    """m×n integer matrix of rank ≤ r as a product of m×r and r×n factors."""
    L = Matrix(rand_int_matrix(m, r, lo, hi))
    R = Matrix(rand_int_matrix(r, n, lo, hi))
    return (L * R).tolist()


# ═══════════════════════════════════════════════════════════════════════════
# poly: Poly view (as_dict, degree, LC, all_coeffs, eval, nroots, symbolic)
# ═══════════════════════════════════════════════════════════════════════════


def poly_as_dict_record(p):
    """Terms as [{"exps": [...], **coeff_record}] in descending lex order."""
    terms = []
    for exps, c in sorted(p.as_dict().items(), reverse=True):
        rec = {"exps": [int(e) for e in exps]}
        rec.update(coeff_record(c))
        terms.append(rec)
    return terms


def gen_poly():
    # ── as_dict ──
    # Hand-written cases: symbolic coefficients (domain EX / ZZ(a,b)),
    # constants, cancellation of terms, negative-power parameters.
    handmade = [
        ((a + 1) * x**2 * y - b * x * y**2 + 3 * a * b, [x, y]),
        (a * x**2 + (a + 1) * x + 3, [x]),
        ((x + y) ** 3 - x**3 - y**3, [x, y]),
        ((x + a) * (x - a), [x]),
        ((x + a) * (x - a), [x, a]),
        (x * y * z + x * y + x + 1, [x, y, z]),
        (x * y * z + x * y + x + 1, [z]),
        (Rational(1, 2) * x**2 - Rational(3, 4) * x * y + Rational(5, 6), [x, y]),
        (x**2 / a + x * b / 2 + 1 / (a * b), [x]),
        ((a * x + b * y) ** 2, [x, y]),
        (sp.sin(a) * x + sp.cos(b), [x]),
        (7, [x, y]),
        (x - x + y, [x, y]),
        ((x + 1) * (x + 2) * (x + 3) - x**3, [x]),
        (z**4 * y**2 * x - z * y**5 + x**3 * y * z**2, [x, y, z]),
    ]
    for e, gens in handmade:
        e = sp.sympify(e)

        def compute(e=e, gens=gens):
            p = sp.Poly(e, *gens)
            return {
                "terms": poly_as_dict_record(p),
                "num_terms": len(p.as_dict()),
                "total_degree": int(p.total_degree()) if not p.is_zero else None,
                "domain": str(p.domain),
                "sympy_result": sp.sstr(p.as_expr()),
            }

        key = f"{sp.sstr(e)} | {','.join(map(str, gens))}"
        C.add(
            "poly",
            "as_dict",
            key,
            {"input": sp.sstr(e), "gens": [str(g) for g in gens], "param_points": [subs_json(p) for p in PARAM_POINTS]},
            compute,
        )
    # Random multivariate polynomials: rational coefficients, then mixed
    # symbolic coefficients in a, b.
    for i in range(10):
        gens = [x, y] if i % 2 == 0 else [x, y, z]
        e = rpoly_rat(gens, 4, 5)

        def compute(e=e, gens=gens):
            p = sp.Poly(e, *gens)
            return {
                "terms": poly_as_dict_record(p),
                "num_terms": len(p.as_dict()),
                "total_degree": int(p.total_degree()),
                "domain": str(p.domain),
                "sympy_result": sp.sstr(p.as_expr()),
            }

        C.add(
            "poly",
            "as_dict",
            f"rand_rat_{i}",
            {"input": sp.sstr(e), "gens": [str(g) for g in gens], "param_points": [subs_json(p) for p in PARAM_POINTS]},
            compute,
        )
    for i in range(10):
        gens = [x, y]
        e = rpoly(gens, 3, 4, coeff=lambda: RNG.choice([a, b, a + 1, a * b, 2 * a - b, Rational(1, 2), 3, -a, b**2]))
        # Multiply through a factor so that like terms need collecting.
        e = sp.expand(e * (x + a) if i % 3 == 0 else e)

        def compute(e=e, gens=gens):
            p = sp.Poly(e, *gens)
            return {
                "terms": poly_as_dict_record(p),
                "num_terms": len(p.as_dict()),
                "total_degree": int(p.total_degree()),
                "domain": str(p.domain),
                "sympy_result": sp.sstr(p.as_expr()),
            }

        C.add(
            "poly",
            "as_dict",
            f"rand_sym_{i}",
            {"input": sp.sstr(e), "gens": [str(g) for g in gens], "param_points": [subs_json(p) for p in PARAM_POINTS]},
            compute,
        )

    # ── degree: total degree and per-generator degrees ──
    degree_cases = [
        (x**3 * y + y**2, [x, y]),
        (x * y * z + x**2, [x, y, z]),
        ((x + y + z) ** 4, [x, y, z]),
        (a * x**5 + b * y**2 * x, [x, y]),
        (a * x**5 + b * y**2 * x, [x]),
        (5, [x]),
        ((x**2 + 1) ** 3 * (y - 2), [x, y]),
        ((x - y) ** 2 * (x + y) ** 2, [x, y]),
    ]
    for i in range(6):
        gens = RNG.choice([[x], [x, y], [x, y, z]])
        degree_cases.append((rpoly(gens, RNG.randint(1, 6), RNG.randint(2, 5)), gens))
    for i, (e, gens) in enumerate(degree_cases):
        e = sp.sympify(e)

        def compute(e=e, gens=gens):
            p = sp.Poly(e, *gens)
            return {
                "total_degree": int(p.total_degree()),
                "degree_list": [int(d) for d in p.degree_list()],
                "is_homogeneous": bool(p.is_homogeneous),
            }

        C.add("poly", "degree", f"{i}:{sp.sstr(e)} | {','.join(map(str, gens))}", {"input": sp.sstr(e), "gens": [str(g) for g in gens]}, compute)

    # ── LC: leading coefficient / monomial in lex order ──
    lc_cases = [
        (a * x**2 + (a + 1) * x + 3, [x]),
        (x**2 * y + x * y**2 + y**3, [x, y]),
        (y**3 + x * y**2 + x**2 * y, [y, x]),
        (3 * x**2 * y - Rational(7, 2) * x**3 + y**5, [x, y]),
        ((a + b) * x * y**2 - a * x**2 * y + b, [x, y]),
        (-x + 1, [x]),
        (Rational(-2, 3) * x**4 * z + y**6, [x, y, z]),
        (b * z**2 + a * y * z + x, [z, y, x]),
    ]
    for i in range(6):
        gens = RNG.choice([[x], [x, y], [x, y, z]])
        lc_cases.append((rpoly_rat(gens, 4, 4), gens))
    for i, (e, gens) in enumerate(lc_cases):
        e = sp.sympify(e)

        def compute(e=e, gens=gens):
            p = sp.Poly(e, *gens)
            rec = coeff_record(p.LC())
            return {"LC": rec, "LM": [int(k) for k in p.monoms()[0]]}

        C.add(
            "poly",
            "LC",
            f"{i}:{sp.sstr(e)} | {','.join(map(str, gens))}",
            {"input": sp.sstr(e), "gens": [str(g) for g in gens], "param_points": [subs_json(p) for p in PARAM_POINTS]},
            compute,
        )

    # ── all_coeffs: dense univariate coefficient list, symbolic allowed ──
    ac_cases = [
        (x**3 * 2 - 5, x),
        (a * x**2 + (a + 1) * x + 3, x),
        ((a + 1) * x**4 + b, x),
        ((x + a) ** 3, x),
        (Rational(1, 3) * x**5 - x, x),
        (x**2 * y + x * y**2 + y**3, x),
        (x**2 * y + x * y**2 + y**3, y),
        (b * x / 2 - a / b, x),
        (x**4 * (a - b) ** 2, x),
        (7, x),
    ]
    for i in range(6):
        ac_cases.append((rpoly_rat([x], RNG.randint(1, 6), RNG.randint(2, 4)), x))
    for i, (e, v) in enumerate(ac_cases):
        e = sp.sympify(e)

        def compute(e=e, v=v):
            p = sp.Poly(e, v)
            return {"all_coeffs": [coeff_record(c) for c in p.all_coeffs()], "degree": int(p.degree())}

        C.add(
            "poly",
            "all_coeffs",
            f"{i}:{sp.sstr(e)} | {v}",
            {"input": sp.sstr(e), "var": str(v), "param_points": [subs_json(p) for p in PARAM_POINTS]},
            compute,
        )

    # ── eval: exact rational evaluation of multivariate polynomials ──
    for i in range(10):
        gens = RNG.choice([[x], [x, y], [x, y, z]])
        e = rpoly_rat(gens, 4, RNG.randint(3, 6))
        pts = [[rrat(max_num=7, max_den=5) for _ in gens] for _ in range(4)]

        def compute(e=e, gens=gens, pts=pts):
            out = []
            for vals in pts:
                v = e.subs(dict(zip(gens, vals)))
                out.append({"values": [qstr(t) for t in vals], "result": qstr(v)})
            return {"points": out}

        C.add("poly", "eval", f"rand_{i}", {"input": sp.sstr(e), "gens": [str(g) for g in gens]}, compute)

    # ── nroots: all complex roots (multiset), including repeated roots ──
    nroots_cases = [
        x**2 - 2,
        x**3 - x,
        3 * x**2 + 1,
        (x - 1) ** 3 * (x + 2) * (x**2 + 1),
        x**4 - 1,
        x**5 - x + 1,
        (2 * x - 3) ** 2 * (x**2 + x + 1),
        x**6 + x**3 + 1,
        Rational(1, 2) * x**3 - Rational(1, 3) * x + Rational(1, 7),
    ]
    for _ in range(5):
        nroots_cases.append(rpoly([x], RNG.randint(2, 6), RNG.randint(3, 5)))
    for _ in range(3):
        f1 = rpoly([x], 1, 2)
        f2 = rpoly([x], 2, 3)
        nroots_cases.append(sp.expand(f1**2 * f2))
    for i, f in enumerate(nroots_cases):
        f = sp.expand(sp.sympify(f))

        def compute(f=f):
            p = sp.Poly(f, x)
            try:
                roots = p.nroots(n=30, maxsteps=500)
                method = "nroots"
            except Exception:  # noqa: BLE001 — mpmath does not converge on repeated roots
                roots = p.all_roots(radicals=False)
                method = "all_roots"
            vals = [numval(r) for r in roots]
            vals.sort(key=lambda v: (round(v["re"], 9), round(v["im"], 9)))
            return {"roots": vals, "degree": int(p.degree()), "oracle": method}

        C.add("poly", "nroots", f"{i}:{sp.sstr(f)}", {"input": sp.sstr(f), "var": "x"}, compute)

    # ── degree_symbolic: Ex::degree with symbolic coefficients ──
    r, f, j = symbols("r f j")
    ds_cases = [
        (j * r**2 + (j + 1) * r * f + 3, r),
        (j * r**2 + (j + 1) * r * f + 3, f),
        (j * r**2 + (j + 1) * r * f + 3, j),
        (a * x**3 + b * x**3 - (a + b) * x**3 + x, x),  # leading terms cancel → degree 1
        ((a + 1) * x**2 + x / a, x),
        (a * b * x**5 - x**5 * a * b + 2, x),  # everything but the constant cancels → degree 0
        (sp.sin(a) * x**4 + sp.exp(b) * x, x),
        ((x + a) ** 4, x),
        ((x + a) ** 4, a),
        (a * x * y**2 + b * y, y),
        (a * x * y**2 + b * y, x),
        (a, x),
    ]
    for i, (e, v) in enumerate(ds_cases):
        e = sp.sympify(e)

        def compute(e=e, v=v):
            d = sp.degree(e, v)
            return {"degree": int(d)}

        C.add("poly", "degree_symbolic", f"{i}:{sp.sstr(e)} | {v}", {"input": sp.sstr(e), "var": str(v)}, compute)

    # ── coeff_symbolic: Ex::coeff(var, n) with symbolic coefficients ──
    cs_cases = [
        (j * r**2 + (j + 1) * r * f + 3, r, 1),
        (j * r**2 + (j + 1) * r * f + 3, r, 2),
        (j * r**2 + (j + 1) * r * f + 3, r, 0),
        (j * r**2 + (j + 1) * r * f + 3, r, 3),
        ((x + a) ** 3, x, 1),
        ((x + a) ** 3, x, 2),
        ((a * x + b) * (b * x - a), x, 1),
        ((a * x + b) * (b * x - a), x, 0),
        (x**2 / a + x * b / 2 + 1 / (a * b), x, 2),
        (x**2 / a + x * b / 2 + 1 / (a * b), x, 0),
        (sp.sin(a) * x**4 + sp.exp(b) * x, x, 4),
        (a * x * y**2 + b * y + (a - b) * y**2, y, 2),
    ]
    for i, (e, v, n) in enumerate(cs_cases):
        e = sp.sympify(e)

        def compute(e=e, v=v, n=n):
            p = sp.Poly(e, v)
            c = p.coeff_monomial(v**n) if n > 0 else p.coeff_monomial(1)
            return {"coeff": coeff_record(c)}

        C.add(
            "poly",
            "coeff_symbolic",
            f"{i}:{sp.sstr(e)} | {v}^{n}",
            {"input": sp.sstr(e), "var": str(v), "power": n, "param_points": [subs_json(pt) for pt in PARAM_POINTS]},
            compute,
        )


# ═══════════════════════════════════════════════════════════════════════════
# ratsimp: rational-function normal form vs sympy.cancel
# ═══════════════════════════════════════════════════════════════════════════


def rational_eval_points(e, gens, want=5, tries=60):
    """Exact rational sample points where the *original* expression is a
    finite rational (so any correct normal form must agree)."""
    pts = []
    for _ in range(tries):
        if len(pts) >= want:
            break
        sub = {g: rrat(max_num=9, max_den=7, nonzero=True) for g in gens}
        try:
            v = sp.nsimplify(e.subs(sub))
        except Exception:  # noqa: BLE001
            continue
        if v is S.NaN or not is_finite_rational(v):
            continue
        pts.append({"subs": {str(g): qstr(t) for g, t in sub.items()}, "value": qstr(v)})
    return pts


def gen_ratsimp():
    handmade = [
        (1 / (x + 1 / y) + 1 / (1 / x + y), [x, y]),
        ((x**2 - y**2) / (x - y), [x, y]),
        (1 / x + 1 / y, [x, y]),
        (1 / x + 1 / y + 1 / z, [x, y, z]),
        (1 / (1 + 1 / (1 + 1 / x)), [x]),
        ((x**2 - 1) / (x**2 + 2 * x + 1), [x]),
        ((x * (x + 1) - 2) / (x - 1) + 3 / (x + 2), [x]),
        ((x**3 - y**3) / (x - y) - (x**2 + x * y + y**2), [x, y]),
        (x / (x - 1) - 1 / (x - 1) - 1, [x]),
        ((a * x + b) / (x + 1) - (a * x + a) / (x + 1), [x, a, b]),
        ((x + y) / (x - y) + (x - y) / (x + y), [x, y]),
        (1 / (x * y) - 1 / (x * (y + 1)), [x, y]),
        ((x**2 + 2 * x * y + y**2) / (x + y) ** 3, [x, y]),
        (x / (y * z) + y / (x * z) + z / (x * y), [x, y, z]),
        ((2 * x + 3) / (4 * x + 6), [x]),
        (Rational(3, 4) + x / 2, [x]),
        ((x**4 - 1) / (x**2 - 1) / (x**2 + 1), [x]),
        (1 / (x - 1) + 1 / (x + 1) - 2 * x / (x**2 - 1), [x]),  # ≡ 0
        ((x + 1 / x) ** 2 - (x - 1 / x) ** 2, [x]),  # ≡ 4
        (1 / (a * x + b) - 1 / (a * x - b), [x, a, b]),
        ((y + 1 / x) / (x + 1 / y), [x, y]),
        ((x**2 * y - x * y**2) / (x * y * (x - y)), [x, y]),
        # solve-style: differences of linear fractions and of quadratics
        ((3 * x + 2) / (x - 4) - (x + 5) / (2 * x + 1), [x]),
        ((x**2 + 3 * x + 2) / (x**2 - 4) - (x + 1) / (x - 2), [x]),
        ((a * x + b) / (b * x + a) - 1, [x, a, b]),
        (1 / (x - y) - 1 / (x + y) - 2 * y / (x**2 - y**2), [x, y]),  # ≡ 0
        ((x + y + z) ** 2 / (x + y + z) - (x + y), [x, y, z]),
    ]
    for i, (e, gens) in enumerate(handmade):
        e = sp.sympify(e)

        def compute(e=e, gens=gens):
            c = sp.cancel(e)
            n, d = sp.fraction(c)
            nd = int(sp.Poly(n, *gens).total_degree()) if n != 0 else 0
            dd = int(sp.Poly(d, *gens).total_degree())
            return {
                "sympy_result": sp.sstr(c),
                "numer": sp.sstr(n),
                "denom": sp.sstr(d),
                "numer_degree": nd,
                "denom_degree": dd,
                "is_zero": bool(c == 0),
                "eval_points": rational_eval_points(e, gens),
            }

        C.add("ratsimp", "cancel", f"h{i}:{sp.sstr(e)}", {"input": sp.sstr(e), "symbols": [str(g) for g in gens]}, compute)

    # Random rational expressions: sums of two or three fractions with
    # small-degree numerators/denominators, some sharing factors so that
    # cancellation actually happens, and nested continued-fraction shapes.
    for i in range(22):
        gens = RNG.choice([[x], [x, y], [x, y], [x, y, z]])
        shape = i % 4
        if shape == 0:
            # p1/q1 + p2/q2
            e = rpoly(gens, 2, 3) / rpoly(gens, 2, 2) + rpoly(gens, 1, 2) / rpoly(gens, 2, 2)
        elif shape == 1:
            # (p·g)/(q·g) + r/q  — g cancels
            g = rpoly(gens, 1, 2)
            q = rpoly(gens, 1, 2)
            e = sp.expand(rpoly(gens, 2, 2) * g) / sp.expand(q * g) + rpoly(gens, 1, 2) / q
        elif shape == 2:
            # nested: 1/(p + 1/q) + q/(1 + p/q)
            p = rpoly(gens, 1, 2)
            q = rpoly(gens, 1, 2)
            e = 1 / (p + 1 / q) + q / (1 + p / q)
        else:
            # three fractions with a common factor in two denominators
            g = rpoly(gens, 1, 2)
            e = rpoly(gens, 1, 2) / g + rpoly(gens, 1, 2) / sp.expand(g * rpoly(gens, 1, 2)) - rpoly(gens, 2, 2) / rpoly(gens, 1, 2)

        def compute(e=e, gens=gens):
            c = sp.cancel(e)
            n, d = sp.fraction(c)
            nd = int(sp.Poly(n, *gens).total_degree()) if n != 0 else 0
            dd = int(sp.Poly(d, *gens).total_degree())
            return {
                "sympy_result": sp.sstr(c),
                "numer": sp.sstr(n),
                "denom": sp.sstr(d),
                "numer_degree": nd,
                "denom_degree": dd,
                "is_zero": bool(c == 0),
                "eval_points": rational_eval_points(e, gens),
            }

        C.add("ratsimp", "cancel", f"r{i}", {"input": sp.sstr(e), "symbols": [str(g) for g in gens]}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# linprog: exact LP vs sympy.solvers.simplex.linprog
# ═══════════════════════════════════════════════════════════════════════════


def lp_fields(c, A_ub, b_ub, A_eq, b_eq, bounds):
    def bnd(t):
        lo, hi = t
        return [None if lo is None else qstr(lo), None if hi is None else qstr(hi)]

    return {
        "c": [qstr(v) for v in c],
        "A_ub": [[qstr(v) for v in row] for row in A_ub],
        "b_ub": [qstr(v) for v in b_ub],
        "A_eq": [[qstr(v) for v in row] for row in A_eq],
        "b_eq": [qstr(v) for v in b_eq],
        "bounds": [bnd(t) for t in bounds] if bounds else [],
    }


def sympy_linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, sense):
    """SymPy oracle: {"status", "objective", "x"}; ``sense`` ∈ {min, max}.

    The reference is ``lpmin`` / ``lpmax`` with every row, equality and
    bound written as an explicit relational constraint.  SymPy 1.14's
    ``linprog`` mishandles non-default ``bounds`` (a negative lower bound or
    a free variable is still implicitly forced to be ≥ 0 by its
    ``_handle_bounds`` substitution, e.g. ``linprog([1], zeros(1,1), [1],
    bounds=[(-2, 3)])`` returns 0 instead of −2), so it is only recorded as
    a secondary verdict (``linprog_status``) for default-bound problems.
    """
    n = len(c)
    xs = symbols(f"v0:{n}")
    constr = []
    infeasible_constant = False

    def push(rel):
        nonlocal infeasible_constant
        if rel is S.true:
            return
        if rel is S.false:
            infeasible_constant = True
            return
        constr.append(rel)

    for row, rhs in zip(A_ub, b_ub):
        push(sum(r * v for r, v in zip(row, xs)) <= rhs)
    for row, rhs in zip(A_eq, b_eq):
        push(sp.Eq(sum(r * v for r, v in zip(row, xs)), rhs))
    bnds = bounds if bounds else [(0, None)] * n
    for v, (lo, hi) in zip(xs, bnds):
        if lo is not None:
            push(v >= lo)
        if hi is not None:
            push(v <= hi)
    obj = sum(cv * v for cv, v in zip(c, xs))
    result = {}
    if infeasible_constant:
        result["status"] = "infeasible"
    else:
        try:
            o, point = (lpmax if sense == "max" else lpmin)(obj, constr)
            result = {"status": "optimal", "objective": qstr(o), "x": [qstr(point[v]) for v in xs]}
        except InfeasibleLPError:
            result["status"] = "infeasible"
        except UnboundedLPError:
            result["status"] = "unbounded"
    # Secondary verdict from linprog (default bounds only, see docstring).
    if not bounds:
        cc = [(-v if sense == "max" else v) for v in c]
        A = Matrix(A_ub) if A_ub else sp.zeros(1, n)
        bb = list(b_ub) if A_ub else [S.One]
        Aeq = Matrix(A_eq) if A_eq else None
        beq = list(b_eq) if A_eq else None
        try:
            o2, _ = linprog(cc, A, bb, Aeq, beq, None)
            result["linprog_status"] = "optimal"
            result["linprog_objective"] = qstr(-o2 if sense == "max" else o2)
        except InfeasibleLPError:
            result["linprog_status"] = "infeasible"
        except UnboundedLPError:
            result["linprog_status"] = "unbounded"
        if result["linprog_status"] != result["status"] or result.get("linprog_objective") != result.get("objective"):
            raise ValueError(f"lpmin/lpmax and linprog disagree: {result}")
    return result


def gen_linprog():
    # ── lpmax: bounded feasible maximisations (x ≥ 0) ──
    textbook_max = [
        ([3, 2], [[1, 1], [1, 3]], [4, 6], [], [], None),
        ([5, 4, 3], [[2, 3, 1], [4, 1, 2], [3, 4, 2]], [5, 11, 8], [], [], None),
        ([1, 1], [[1, 2], [3, 1]], [4, 6], [], [], None),
        ([Rational(1, 2), Rational(1, 3)], [[Rational(1, 4), 1], [1, Rational(1, 5)]], [Rational(3, 2), Rational(7, 3)], [], [], None),
        ([2, 3, 1], [[1, 1, 1]], [10], [[1, -1, 0]], [2], None),
        ([1, 2], [[1, 1]], [5], [], [], [(0, 2), (0, 4)]),
    ]
    for i, (c, A, bb, Aeq, beq, bnds) in enumerate(textbook_max):
        C.add(
            "linprog",
            "lpmax",
            f"t{i}",
            {**lp_fields(c, A, bb, Aeq, beq, bnds), "sense": "max"},
            lambda c=c, A=A, bb=bb, Aeq=Aeq, beq=beq, bnds=bnds: sympy_linprog(c, A, bb, Aeq, beq, bnds, "max"),
        )
    for i in range(8):
        n = RNG.randint(2, 4)
        m = RNG.randint(1, 3)
        c = [rrat(max_num=6, max_den=3) for _ in range(n)]
        A = [[rrat(max_num=5, max_den=2) for _ in range(n)] for _ in range(m)]
        x0 = [rrat(max_num=4, max_den=2).__abs__() for _ in range(n)]
        bb = [sum(A[r][j] * x0[j] for j in range(n)) + Rational(RNG.randint(0, 3), RNG.randint(1, 2)) for r in range(m)]
        # A box row keeps the maximisation bounded.
        A.append([S.One] * n)
        bb.append(sum(x0) + RNG.randint(1, 5))
        Aeq, beq = [], []
        if i % 3 == 0:
            row = [rrat(max_num=3, max_den=2, nonzero=True) for _ in range(n)]
            Aeq = [row]
            beq = [sum(row[j] * x0[j] for j in range(n))]
        C.add(
            "linprog",
            "lpmax",
            f"r{i}",
            {**lp_fields(c, A, bb, Aeq, beq, None), "sense": "max"},
            lambda c=c, A=A, bb=bb, Aeq=Aeq, beq=beq: sympy_linprog(c, A, bb, Aeq, beq, None, "max"),
        )

    # ── lpmin: minimisations, some with explicit / free bounds ──
    textbook_min = [
        ([1, 1], [[-1, -2], [-3, -1]], [-1, -1], [], [], None),
        ([-1, -1], [[1, 2], [3, 1]], [4, 6], [], [], None),
        ([2, 3, 4], [[-3, -2, -1], [-2, -5, -3]], [-10, -15], [], [], None),
        ([1, -1], [[1, 1]], [4], [], [], [(-2, 3), (None, 5)]),
        ([1, 1], [[1, 1]], [4], [], [], [(None, None), (0, None)]),
        ([-1, 2, -3], [[1, 1, 1]], [7], [[1, -1, 0]], [1], [(0, None), (0, None), (0, 2)]),
        ([1, 1, 1, 1], [[1, 2, 3, 4], [4, 3, 2, 1]], [5, 5], [], [], None),  # SymPy doc example shape
        ([-1, 5, 1, 4], [[-1, 5, 2, 5]], [5], [[0, 3, 0, 1], [-1, 0, 1, 2]], [2, 1], None),  # SymPy doc example
        ([Rational(1, 3), Rational(1, 7)], [[-1, -1]], [-Rational(5, 2)], [], [], None),
        ([1, 1], [], [], [[Rational(1, 3), Rational(1, 7)], [1, -1]], [1, 0], None),
    ]
    for i, (c, A, bb, Aeq, beq, bnds) in enumerate(textbook_min):
        C.add(
            "linprog",
            "lpmin",
            f"t{i}",
            {**lp_fields(c, A, bb, Aeq, beq, bnds), "sense": "min"},
            lambda c=c, A=A, bb=bb, Aeq=Aeq, beq=beq, bnds=bnds: sympy_linprog(c, A, bb, Aeq, beq, bnds, "min"),
        )
    for i in range(8):
        n = RNG.randint(2, 4)
        m = RNG.randint(1, 3)
        # Non-negative costs keep the default-bounds minimisation bounded.
        c = [abs(rrat(max_num=6, max_den=3)) for _ in range(n)]
        A = [[rrat(max_num=5, max_den=2) for _ in range(n)] for _ in range(m)]
        x0 = [abs(rrat(max_num=4, max_den=2)) for _ in range(n)]
        # ≥ constraints written as ≤ with negated rows.
        bb = [-(sum(A[r][j] * x0[j] for j in range(n)) - Rational(RNG.randint(0, 3), RNG.randint(1, 2))) for r in range(m)]
        A = [[-v for v in row] for row in A]
        bnds = None
        if i % 2 == 1:
            bnds = [(Rational(RNG.randint(-3, 0), 2), RNG.choice([None, Rational(RNG.randint(3, 8), 2)])) for _ in range(n)]
        Aeq, beq = [], []
        if i % 4 == 2:
            row = [rrat(max_num=3, max_den=2, nonzero=True) for _ in range(n)]
            Aeq = [row]
            beq = [sum(row[j] * x0[j] for j in range(n))]
        C.add(
            "linprog",
            "lpmin",
            f"r{i}",
            {**lp_fields(c, A, bb, Aeq, beq, bnds), "sense": "min"},
            lambda c=c, A=A, bb=bb, Aeq=Aeq, beq=beq, bnds=bnds: sympy_linprog(c, A, bb, Aeq, beq, bnds, "min"),
        )

    # ── infeasible: contradictory constraints (default bounds x ≥ 0) ──
    infeasible = [
        ([1, 1], [[1, 1], [-1, -1]], [1, -3], [], []),
        ([1, 2], [[1, 1]], [-1], [], []),
        ([1, 1, 1], [[1, 1, 1]], [1], [[1, 1, 1]], [2]),
        ([3, -1], [[-1, 0]], [-5], [[1, 0]], [2]),
        ([1, 1], [[1, -1], [-1, 1]], [-1, -1], [], []),
        ([0, 0], [], [], [[1, 1], [1, 1]], [1, 2]),
        ([1, 1], [[2, 3]], [6], [[4, 6]], [13]),
    ]
    for i in range(5):
        n = RNG.randint(2, 3)
        row = [abs(rrat(max_num=5, max_den=2, nonzero=True)) for _ in range(n)]
        # Σ positive·x ≤ negative is impossible with x ≥ 0.
        infeasible.append(([rint(-3, 3) for _ in range(n)], [row], [-Rational(RNG.randint(1, 5), RNG.randint(1, 3))], [], []))
    for i, (c, A, bb, Aeq, beq) in enumerate(infeasible):
        C.add(
            "linprog",
            "infeasible",
            f"{i}",
            {**lp_fields(c, A, bb, Aeq, beq, None), "sense": "min"},
            lambda c=c, A=A, bb=bb, Aeq=Aeq, beq=beq: sympy_linprog(c, A, bb, Aeq, beq, None, "min"),
        )

    # ── unbounded ──
    unbounded = [
        ([-1, -1], [[-1, 0]], [1], [], [], None),
        ([-1, 0], [[0, 1]], [3], [], [], None),
        ([1, 1], [[1, 1]], [4], [], [], [(None, None), (None, None)]),
        ([-2, -3, -1], [[1, -1, 0]], [2], [], [], None),
        ([-1, 1], [[-1, 1]], [1], [], [], None),
        ([0, -1], [], [], [[1, -1]], [0], None),
        ([1, -1], [[1, 0]], [5], [], [], [(0, None), (None, None)]),
    ]
    for i in range(3):
        n = RNG.randint(2, 3)
        # Objective wants x_k → ∞ and no row has a positive coefficient on x_k.
        k = RNG.randrange(n)
        c = [rint(-3, 3) for _ in range(n)]
        c[k] = -RNG.randint(1, 4)
        row = [rint(-3, 3) for _ in range(n)]
        row[k] = -RNG.randint(0, 2)
        unbounded.append((c, [row], [RNG.randint(1, 6)], [], [], None))
    for i, (c, A, bb, Aeq, beq, bnds) in enumerate(unbounded):
        C.add(
            "linprog",
            "unbounded",
            f"{i}",
            {**lp_fields(c, A, bb, Aeq, beq, bnds), "sense": "min"},
            lambda c=c, A=A, bb=bb, Aeq=Aeq, beq=beq, bnds=bnds: sympy_linprog(c, A, bb, Aeq, beq, bnds, "min"),
        )

    # ── feasible_nonneg: A x = b, x ≥ 0 with fractional data ──
    feas_cases = [
        ([[Rational(1, 3), Rational(1, 7)], [1, -1]], [1, 0]),
        ([[1, 1]], [-1]),
        ([[1, 2, 3]], [Rational(5, 2)]),
        ([[1, -1]], [Rational(1, 2)]),
        ([[-1, -1, 0], [0, 1, -1]], [Rational(1, 2), 0]),
        ([[Rational(1, 2), Rational(1, 3), Rational(1, 4)], [1, 1, 1]], [1, 3]),  # infeasible: max of first row at sum=3 is 3/2 ≥ 1 fine, min 3/4 — feasible actually
    ]
    for i in range(10):
        n = RNG.randint(3, 5)
        m = RNG.randint(1, n - 1)
        A = [[rrat(max_num=5, max_den=3) for _ in range(n)] for _ in range(m)]
        if i % 2 == 0:
            x0 = [abs(rrat(max_num=4, max_den=3)) for _ in range(n)]
            bb = [sum(A[r][j] * x0[j] for j in range(n)) for r in range(m)]
        else:
            bb = [rrat(max_num=6, max_den=3) for _ in range(m)]
        feas_cases.append((A, bb))
    for i, (A, bb) in enumerate(feas_cases):
        n = len(A[0])

        def compute(A=A, bb=bb, n=n):
            r = sympy_linprog([S.Zero] * n, [], [], A, bb, None, "min")
            return {"feasible": r["status"] == "optimal", "x": r.get("x")}

        C.add("linprog", "feasible_nonneg", f"{i}", {"A_eq": [[qstr(v) for v in row] for row in A], "b_eq": [qstr(v) for v in bb]}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# normalforms: HNF / SNF / integer kernel / lattice determinant / igcd, ilcm
# ═══════════════════════════════════════════════════════════════════════════


def sympy_row_hnf(rows):
    """Row-style HNF (symplex convention) derived from SymPy's column HNF:
    R(B) = T·P·Q·Cc(P·T·B) with T = transpose, P = reverse rows,
    Q = reverse columns, Cc = SymPy column HNF (zero columns dropped), then
    padded with zero rows to the shape of B."""
    B = Matrix(rows)
    m, n = B.shape
    Pt = Matrix(list(reversed(B.T.tolist())))  # P·T·B  (n×m)
    X = hermite_normal_form(Pt)  # n×r
    R = X.T  # r×n
    R = [list(reversed(r)) for r in reversed(R.tolist())]  # P then Q
    while len(R) < m:
        R.append([0] * n)
    return R


def gcd_of_maximal_minors(rows):
    """gcd of all m×m minors of an m×n matrix with m ≤ n (0 if all vanish)."""
    M = Matrix(rows)
    m, n = M.shape
    g = 0
    for cols in itertools.combinations(range(n), m):
        g = sp.igcd(g, int(M[:, list(cols)].det()))
    return int(abs(g))


def gen_normalforms():
    fixed = [
        [[2, 4, 4], [-6, 6, 12], [10, -4, -16]],
        [[12, 6, 4], [3, 9, 6], [2, 16, 14]],
        [[3, 1], [1, 2]],
        [[1, 2], [2, 4]],
        [[1, 0], [0, 1], [0, 0]],
        [[0, 0], [0, 0]],
        [[5]],
        [[0, 3, 0], [0, 0, 0], [2, 0, 0]],
        [[1, 2, 3], [2, 4, 6], [1, 1, 1]],
        [[1, 0, 2, -1], [0, 1, 1, 1], [1, 1, 3, 0]],
        [[2, 4], [4, 8], [6, 12]],
        [[-7, 3], [2, -5]],
        [[6, 4, 2], [4, 6, 2]],
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[100, 35], [-42, 7]],
    ]
    matrices = list(fixed)
    shapes = [(2, 2), (3, 3), (3, 3), (4, 4), (2, 3), (3, 2), (3, 4), (4, 3), (2, 4)]
    for m, n in shapes:
        matrices.append(rand_int_matrix(m, n))
    for m, n, r in [(3, 3, 1), (3, 3, 2), (4, 4, 2), (3, 4, 2), (4, 3, 1), (4, 2, 1), (2, 4, 1)]:
        matrices.append(rand_low_rank_matrix(m, n, r))

    for rows in matrices:
        label = mat_label(rows)
        M = Matrix(rows)

        def compute_hnf(M=M):
            H = hermite_normal_form(M)
            return {"hnf": mat_json(H.tolist()) if H.cols > 0 else [[] for _ in range(H.rows)], "hnf_shape": [int(H.rows), int(H.cols)], "rank": int(M.rank())}

        C.add("normalforms", "hnf", label, {"matrix": mat_json(rows)}, compute_hnf)

        def compute_row(rows=rows, M=M):
            R = sympy_row_hnf(rows)
            return {"hnf_row": mat_json(R), "rank": int(M.rank())}

        C.add("normalforms", "hnf_row", label, {"matrix": mat_json(rows)}, compute_row)

        def compute_snf(M=M):
            Sm = smith_normal_form(M, domain=sp.ZZ)
            diag = [abs(int(Sm[i, i])) for i in range(min(Sm.shape))]
            diag = sorted(d for d in diag if d != 0)
            return {"snf_diag": [str(d) for d in diag], "rank": int(M.rank()), "sympy_result": str(Sm.tolist())}

        C.add("normalforms", "snf", label, {"matrix": mat_json(rows)}, compute_snf)

        def compute_ns(M=M):
            ns = M.nullspace()
            return {"rank": int(M.rank()), "nullity": len(ns), "ncols": int(M.cols)}

        C.add("normalforms", "nullspace_rank", label, {"matrix": mat_json(rows)}, compute_ns)

        def compute_ld(M=M, rows=rows):
            m, n = M.shape
            full_row_rank = M.rank() == m
            return {
                "full_row_rank": bool(full_row_rank),
                "lattice_det": str(gcd_of_maximal_minors(rows)) if full_row_rank and m <= n else None,
            }

        C.add("normalforms", "lattice_det", label, {"matrix": mat_json(rows)}, compute_ld)

    # ── igcd / ilcm on lists ──
    gcd_lists = [
        [12, 18],
        [12, 18, 30],
        [-4, 6, 0],
        [0, 0],
        [0, 7],
        [1, 2, 3, 4, 5],
        [2**61 - 1, 2**31 - 1],
        [2**64 + 1, 274177],
        [360, 1001, 2**20 - 1],
        [17, 17, 17],
        [-15],
        [6, 10, 15],
        [2 * 3 * 5 * 7 * 11 * 13, 3 * 5 * 7 * 11 * 13 * 17, 5 * 7 * 11 * 13 * 17 * 19],
        [10**18 + 9, 10**18 + 3],
    ]
    for _ in range(6):
        k = RNG.randint(2, 5)
        g = RNG.randint(1, 60)
        gcd_lists.append([g * rint(-50, 50) for _ in range(k)])
    for i, vals in enumerate(gcd_lists):

        def compute(vals=vals):
            return {"gcd": str(sp.igcd(*vals) if len(vals) > 1 else abs(vals[0])), "lcm": str(sp.ilcm(*vals) if len(vals) > 1 else abs(vals[0]))}

        C.add("normalforms", "igcd_ilcm", f"{i}:{vals}", {"values": [str(v) for v in vals]}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# optimize: exact polynomial least squares, Brent root finding
# ═══════════════════════════════════════════════════════════════════════════


def gen_optimize():
    # ── polyfit_exact ──
    fit_cases = [
        ([(0, Rational(1, 7)), (1, Rational(-1, 42)), (2, Rational(10, 21)), (3, Rational(23, 14)), (4, Rational(73, 21))], 2),
        ([(0, 1), (1, 2), (2, 5)], 2),  # interpolation x²+1
        ([(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)], 1),  # overdetermined line through a parabola
        ([(-1, 1), (0, 0), (1, 1), (2, 4)], 2),
        ([(Rational(1, 2), 1), (Rational(3, 2), 2), (Rational(5, 2), 0), (Rational(7, 2), 1)], 3),
        ([(0, 1), (1, 1), (2, 1), (3, 1)], 0),
        ([(1, 2), (2, 3), (3, 5), (4, 7), (5, 11)], 3),
    ]
    for i in range(7):
        npts = RNG.randint(3, 7)
        deg = RNG.randint(0, npts - 1)
        xs = RNG.sample(range(-6, 7), npts)
        pts = [(Rational(xv, RNG.choice([1, 1, 2, 3])), rrat(max_num=9, max_den=4)) for xv in xs]
        # distinct abscissae after scaling? re-sample until distinct
        while len({p[0] for p in pts}) < npts:
            pts = [(Rational(xv, RNG.choice([1, 1, 2, 3])), rrat(max_num=9, max_den=4)) for xv in xs]
        fit_cases.append((pts, deg))
    for i, (pts, deg) in enumerate(fit_cases):
        pts = [(Rational(px), Rational(py)) for px, py in pts]

        def compute(pts=pts, deg=deg):
            V = Matrix([[px**k for k in range(deg + 1)] for px, _ in pts])
            yv = Matrix([py for _, py in pts])
            coeffs = (V.T * V).solve(V.T * yv)
            residual = (V * coeffs - yv).T * (V * coeffs - yv)
            return {"coeffs": [qstr(c) for c in coeffs], "residual_sq": qstr(residual[0])}

        C.add("optimize", "polyfit_exact", f"{i}", {"points": [[qstr(px), qstr(py)] for px, py in pts], "degree": deg}, compute)

    # ── brent_root: transcendental equations with a sign-changing bracket ──
    root_cases = [
        (sp.cos(x) - x, 0, 1),
        (sp.exp(x) - 3, 0, 2),
        (x * sp.exp(x) - 1, 0, 1),
        (sp.sin(x) - x / 2, 1, 3),
        (sp.log(x) - 1, 1, 4),
        (x**3 - 2 * x - 5, 2, 3),
        (sp.tan(x) - 2 * x, Rational(1, 2), Rational(3, 2)),
        (sp.exp(-x) - x, 0, 1),
        (sp.atan(x) - x / 2, 1, 3),
        (x**2 - sp.sqrt(x) - 1, 1, 2),
        (sp.sinh(x) - 1, 0, 1),
        (x * sp.log(x) - 1, 1, 2),
        (sp.exp(x) + x - 5, 0, 2),
        (sp.cos(x) ** 2 - x**3, 0, 1),
        (sp.erf(x) - Rational(1, 2), 0, 1),
        (1 / x - sp.exp(-x) - Rational(1, 2), Rational(1, 2), 2),
        (sp.log(x + 1) - x**2, Rational(1, 4), 1),
    ]
    for i, (f, lo, hi) in enumerate(root_cases):
        f = sp.sympify(f)

        def compute(f=f, lo=lo, hi=hi):
            fa, fb = sp.N(f.subs(x, lo), 30), sp.N(f.subs(x, hi), 30)
            if fa * fb >= 0:
                raise ValueError(f"not a bracket: f({lo}) = {fa}, f({hi}) = {fb}")
            r = sp.nsolve(f, x, (lo, hi), solver="bisect", prec=30)
            return {"root": str(r), "root_f64": float(r), "f_at_root": float(sp.N(f.subs(x, r), 30))}

        C.add("optimize", "brent_root", f"{i}:{sp.sstr(f)}", {"input": sp.sstr(f), "var": "x", "bracket": [float(lo), float(hi)]}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════


def build():
    for gen in (gen_poly, gen_ratsimp, gen_linprog, gen_normalforms, gen_optimize):
        print(f"== {gen.__name__}", file=sys.stderr)
        gen()

    fixtures = sorted(C.fixtures, key=lambda f: (f["category"], f["subcategory"]))  # stable
    for i, f in enumerate(fixtures, 1):
        f["id"] = i
    counts = Counter(f"{f['category']}:{f['subcategory']}" for f in fixtures)
    return {
        "generated_by": f"generate_v03_fixtures.py (sympy {sp.__version__})",
        "generator": "scripts/generate_v03_fixtures.py",
        "seed": SEED,
        "fixture_count": len(fixtures),
        "categories": dict(sorted(counts.items())),
        "fixtures": fixtures,
    }


def main():
    check = "--check" in sys.argv
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests",
        "fixtures",
        "v03_cross_validation.json",
    )
    output = build()
    text = json.dumps(output, indent=1, sort_keys=True) + "\n"
    print(
        f"\nGenerated {output['fixture_count']} fixtures ({C.timeouts} sympy timeouts, {C.errors} sympy errors)",
        file=sys.stderr,
    )
    for cat, cnt in output["categories"].items():
        print(f"  {cat:40s} {cnt:4d}", file=sys.stderr)
    lp_status = Counter(
        (f["subcategory"], f.get("status", "feasible" if f.get("feasible") else ("infeasible" if "feasible" in f else "?")))
        for f in output["fixtures"]
        if f["category"] == "linprog"
    )
    for (sub, st), cnt in sorted(lp_status.items()):
        print(f"  linprog:{sub:16s} {st:12s} {cnt:3d}", file=sys.stderr)
    if check:
        with open(out_path) as fh:
            if fh.read() != text:
                print("MISMATCH: fixture file differs from generator output", file=sys.stderr)
                sys.exit(1)
        print("OK: fixture file is up to date", file=sys.stderr)
        return
    with open(out_path, "w") as fh:
        fh.write(text)
    print(f"Wrote {out_path} ({len(text) / 1024:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
