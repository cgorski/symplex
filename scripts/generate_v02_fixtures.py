#!/usr/bin/env python3
"""Generate the symplex 0.2 SymPy cross-validation fixtures.

Output: ``tests/fixtures/v02_cross_validation.json`` (path derived from the
location of this script), consumed by ``tests/v02_oracle_*.rs`` — one test
file per area, one ``#[test]`` per ``(category, subcategory)``.

Design rules (see tests/README.md):

* **Deterministic.**  Fixed seed, no timestamps, ``sort_keys=True``; running
  the script twice yields byte-identical output.
* **Bounded.**  Every SymPy computation runs under a per-fixture wall-clock
  timeout (``PER_FIXTURE_TIMEOUT``).  A fixture whose oracle computation times
  out is *kept* with ``"sympy_timeout": true`` so the consumer can report it
  as SKIPPED_ORACLE instead of silently dropping the case.  A fixture whose
  oracle computation raises is kept with ``"sympy_error"``; one whose result is
  an unevaluated SymPy object is kept with ``"sympy_unevaluated": true``.
* **Reference values are as independent as possible.**  Where a brute-force
  computation exists (finite sums/products, recurrences, truth tables, set
  membership) it is used as the reference rather than SymPy's closed form.
* **Every fixture has a stable ``key``** (unique within category:subcategory)
  so the Rust consumers can attach ``// BUG:`` annotations that survive
  regeneration.

Usage::

    /Users/chris.gorski/repos/math/symplex/.venv/bin/python \
        scripts/generate_v02_fixtures.py            # writes the JSON
    ... --check                                     # exit 1 if the file would change
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import sys
import traceback
from collections import Counter
from fractions import Fraction

import sympy as sp
from sympy import (
    Abs,
    And,
    Complement,
    Eq,
    FiniteSet,
    Function,
    Heaviside,
    I,
    ImageSet,
    Integers,
    Intersection,
    Interval,
    Lambda,
    Matrix,
    Not,
    Or,
    Rational,
    S,
    SymmetricDifference,
    Union,
    Xor,
    cos,
    exp,
    log,
    oo,
    pi,
    sin,
    sqrt,
    symbols,
    zoo,
)

SEED = 20260918
PER_FIXTURE_TIMEOUT = 25.0  # seconds of wall clock per SymPy computation
N_DIGITS = 30

x, y, z, t, s, k, n, w, a, b = symbols("x y z t s k n w a b")
xr, yr = symbols("xr yr", real=True)
ap = symbols("a", positive=True)

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
    """{"re", "im"} for a finite number, {"special": "oo"|"-oo"|"zoo"|"nan"}
    for extended values, or None if SymPy cannot produce a number."""
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


def eval_points(expr, pts):
    out = []
    for p in pts:
        v = numval(expr.subs(p))
        if v is not None:
            out.append({"subs": subs_json(p), "value": v})
    return out


def has_unevaluated(expr, *classes):
    return any(expr.has(c) for c in classes)


class Collector:
    def __init__(self):
        self.fixtures = []
        self.keys = set()
        self.timeouts = 0
        self.errors = 0
        self.unevaluated = 0

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
            if result.get("sympy_unevaluated"):
                self.unevaluated += 1
                print(f"  UNEVAL  {category}:{subcategory} {key}", file=sys.stderr)
            fx.update(result)
        self.fixtures.append(fx)


C = Collector()

# Standard evaluation points.
PTS_X = [{x: Rational(3, 2)}, {x: Rational(7, 10)}, {x: Rational(-11, 5)}, {x: Rational(23, 10)}]
PTS_X_POS = [{x: Rational(3, 2)}, {x: Rational(7, 10)}, {x: Rational(23, 10)}, {x: Rational(2, 7)}]
PTS_XY = [
    {x: Rational(3, 2), y: Rational(1, 2)},
    {x: Rational(7, 10), y: Rational(-3, 10)},
    {x: Rational(-5, 4), y: Rational(9, 4)},
]
PTS_XY_POS = [
    {x: Rational(3, 2), y: Rational(1, 2)},
    {x: Rational(7, 10), y: Rational(3, 10)},
    {x: Rational(5, 4), y: Rational(9, 4)},
]


def bound_str(bnd):
    if bnd == oo:
        return "oo"
    if bnd == -oo:
        return "-oo"
    return str(bnd)


# ═══════════════════════════════════════════════════════════════════════════
# calculus: definite integrals
# ═══════════════════════════════════════════════════════════════════════════


def gen_definite_integrals():
    cases = {
        "proper": [
            ("x**3 - 2*x + 1", 0, 2),
            ("sin(x)**2", 0, pi),
            ("x*exp(x)", 0, 1),
            ("1/(1 + x**2)", 0, 1),
            ("sqrt(1 - x**2)", 0, 1),
            ("x**2*cos(x)", 0, pi / 2),
            ("log(x)/x", 1, sp.E),
            ("exp(-x)*sin(x)", 0, pi),
            ("1/(x**2 + 3*x + 2)", 0, 1),
            ("tan(x)", 0, pi / 4),
            ("x/sqrt(x**2 + 1)", 0, 2),
            ("cosh(x)", -1, 1),
            ("abs(x - 1)", 0, 3),
            ("x**Rational(3, 2)", 0, 4),
            ("sin(x)*cos(2*x)", 0, pi / 2),
            ("1/(x*log(x))", sp.E, sp.E**2),
            ("exp(2*x)*cos(x)", 0, pi),
            ("x**2/(1 + x**3)", 0, 1),
        ],
        "improper": [
            ("log(x)", 0, 1),
            ("1/sqrt(x)", 0, 1),
            ("1/sqrt(1 - x**2)", -1, 1),
            ("log(x)**2", 0, 1),
            ("x*log(x)", 0, 1),
            ("1/x**Rational(1, 3)", 0, 8),
            ("1/sqrt(4 - x)", 0, 4),
            ("log(x)/sqrt(x)", 0, 1),
            ("1/(x - 1)**Rational(1, 2)", 1, 5),
        ],
        "infinite": [
            ("exp(-x)*x**3", 0, oo),
            ("x*exp(-x**2)", 0, oo),
            ("1/(1 + x**2)", -oo, oo),
            ("sin(x)/x", 0, oo),
            ("log(x)/(1 + x**2)", 0, oo),
            ("x**2*exp(-x**2)", -oo, oo),
            ("1/(x**2 + 4*x + 5)", -oo, oo),
            ("exp(-2*x)*cos(3*x)", 0, oo),
            ("1/(x**2 - 1)", 2, oo),
            ("sqrt(x)*exp(-x)", 0, oo),
            ("1/(1 + x**4)", 0, oo),
            ("1/cosh(x)", -oo, oo),
            ("x/(exp(x) - 1)", 0, oo),
            ("exp(-x**2)*cos(x)", -oo, oo),
            ("exp(-x)/sqrt(x)", 0, oo),
            ("1/(x**2 + 1)**2", -oo, oo),
            ("x**3/(exp(x) - 1)", 0, oo),
            ("exp(-3*x)*sin(2*x)", 0, oo),
            ("1/(x*(x + 1))", 1, oo),
            ("atan(x)/x**2", 1, oo),
        ],
        "symmetric": [
            ("x**3*cos(x)", -2, 2),
            ("x**2*abs(x)", -1, 1),
            ("sin(x)**3", -pi, pi),
            ("x*sin(x)", -pi, pi),
            ("exp(-abs(x))", -3, 3),
            ("x**4 - x**2", -1, 1),
            ("x*exp(-x**2)", -oo, oo),
            ("cos(x)/(1 + x**2)", -oo, oo),
        ],
    }
    for sub, lst in cases.items():
        for f_str, lo, hi in lst:
            f = sp.sympify(f_str)

            def compute(f=f, lo=lo, hi=hi):
                r = sp.integrate(f, (x, lo, hi))
                out = {"sympy_result": str(r)}
                if r.has(sp.Integral) or r.has(sp.Piecewise):
                    val = sp.Integral(f, (x, lo, hi)).evalf(N_DIGITS)
                    out["sympy_method"] = "numeric"
                    out["value"] = numval(val)
                else:
                    out["sympy_method"] = "symbolic"
                    out["value"] = numval(r)
                if out["value"] is None:
                    out["sympy_unevaluated"] = True
                return out

            C.add(
                "definite_integral",
                sub,
                f"{f_str} | [{bound_str(lo)}, {bound_str(hi)}]",
                {"input": str(f), "variable": "x", "lower": bound_str(lo), "upper": bound_str(hi)},
                compute,
            )

    # Parametric: integrand mentions a positive parameter `a`.
    param_cases = [
        ("exp(-a*x**2)", -oo, oo),
        ("exp(-a*x)", 0, oo),
        ("x*exp(-a*x)", 0, oo),
        ("1/(x**2 + a**2)", -oo, oo),
        ("exp(-a*x)*cos(x)", 0, oo),
        ("x**2*exp(-a*x)", 0, oo),
        ("1/(a**2 + x**2)", 0, a),
        ("sin(a*x)/x", 0, oo),
    ]
    a_vals = [{ap: Rational(1, 2)}, {ap: 2}, {ap: Rational(7, 3)}]
    for f_str, lo, hi in param_cases:
        f = sp.sympify(f_str).subs(a, ap)
        lo_ = sp.sympify(lo).subs(a, ap) if lo not in (oo, -oo) else lo
        hi_ = sp.sympify(hi).subs(a, ap) if hi not in (oo, -oo) else hi

        def compute(f=f, lo=lo_, hi=hi_):
            r = sp.integrate(f, (x, lo, hi))
            out = {"sympy_result": str(r)}
            if r.has(sp.Integral):
                out["sympy_unevaluated"] = True
                return out
            out["eval_points"] = eval_points(r, a_vals)
            return out

        C.add(
            "definite_integral",
            "parametric",
            f"{f_str} | [{bound_str(lo)}, {bound_str(hi)}]",
            {
                "input": f_str,
                "variable": "x",
                "lower": bound_str(lo),
                "upper": bound_str(hi),
                "assumptions": {"a": "positive"},
            },
            compute,
        )

    # Numeric quadrature: symplex integrate_numeric vs SymPy Integral.evalf.
    numeric_cases = [
        ("exp(-x)*sin(x)", 0, oo),
        ("sin(x**2)", 0, 3),
        ("exp(-x**2)*log(1 + x**2)", 0, oo),
        ("1/(1 + x**6)", -oo, oo),
        ("sqrt(x)*cos(x)", 0, 5),
        ("exp(sin(x))", 0, 2 * pi),
        ("x**x", 0, 1),
        ("1/sqrt(1 + x**3)", 0, 4),
        ("cos(x)*exp(-x/3)", 0, oo),
        ("log(1 + x)/x", 0, 1),
        ("atan(x)**2/(1 + x**2)", 0, 1),
        ("exp(-x**2/2)*x**4", -oo, oo),
    ]
    for f_str, lo, hi in numeric_cases:
        f = sp.sympify(f_str)

        def compute(f=f, lo=lo, hi=hi):
            val = sp.Integral(f, (x, lo, hi)).evalf(N_DIGITS)
            return {"value": numval(val), "sympy_result": str(val)}

        C.add(
            "definite_integral",
            "numeric",
            f"{f_str} | [{bound_str(lo)}, {bound_str(hi)}]",
            {"input": str(f), "variable": "x", "lower": bound_str(lo), "upper": bound_str(hi)},
            compute,
        )


# ═══════════════════════════════════════════════════════════════════════════
# calculus: sums and products
# ═══════════════════════════════════════════════════════════════════════════


def brute_sum(f, lo, hi):
    return sum((f.subs(k, i) for i in range(lo, hi + 1)), S.Zero)


def brute_prod(f, lo, hi):
    r = S.One
    for i in range(lo, hi + 1):
        r *= f.subs(k, i)
    return r


def gen_sums_products():
    finite = [
        "k**3",
        "k**4",
        "k*(k + 1)",
        "1/(k*(k + 1))",
        "1/(k*(k + 2))",
        "k*2**k",
        "k**2*2**k",
        "1/k",
        "k*factorial(k)",
        "(2*k - 1)**2",
        "3**k",
        "k/2**k",
        "1/(4*k**2 - 1)",
        "binomial(n, k)",
        "k*binomial(n, k)",
        "(-1)**k*k",
        "1/k**2",
    ]
    for f_str in finite:
        f = sp.sympify(f_str)
        n_vals = list(range(1, 7))

        def compute(f=f):
            closed = sp.summation(f, (k, 1, n))
            out = {"sympy_result": str(closed)}
            pts = []
            for nv in n_vals:
                exact = brute_sum(f.subs(n, nv), 1, nv)
                pts.append({"subs": {"n": float(nv)}, "value": numval(exact)})
            out["eval_points"] = pts
            if closed.has(sp.Sum):
                out["sympy_closed_form"] = False
            else:
                out["sympy_closed_form"] = True
                # Sanity: the closed form must agree with brute force.
                for nv in n_vals:
                    diff = sp.N(closed.subs(n, nv) - brute_sum(f.subs(n, nv), 1, nv))
                    assert abs(complex(diff)) < 1e-9, (f_str, nv, diff)
            return out

        C.add(
            "summation",
            "finite_symbolic",
            f_str,
            {"input": f_str, "variable": "k", "lower": "1", "upper": "n"},
            compute,
        )

    infinite = [
        "1/k**2",
        "1/k**4",
        "1/k**6",
        "1/(k*(k + 1))",
        "1/2**k",
        "k/3**k",
        "1/factorial(k)",
        "(-1)**k/k",
        "1/((2*k - 1)*(2*k + 1))",
        "(-1)**(k + 1)/k**2",
        "k**2/2**k",
        "1/(4*k**2 - 1)",
        "1/k**3",
        "(-1)**k/(2*k + 1)",
        "1/(k**2 + k)",
        "1/(k*(k + 1)*(k + 2))",
        "1/(k**2 + 1)",
        "1/k**2*(1/2)**k",
        "k/(k + 1)**3",
        "(2/3)**k",
    ]
    for f_str in infinite:
        f = sp.sympify(f_str)
        lo = 0 if f_str == "(-1)**k/(2*k + 1)" else 1

        def compute(f=f, lo=lo):
            r = sp.summation(f, (k, lo, oo))
            out = {"sympy_result": str(r)}
            if r.has(sp.Sum):
                val = sp.Sum(f, (k, lo, oo)).evalf(N_DIGITS)
                out["sympy_method"] = "numeric"
                out["value"] = numval(val)
            else:
                out["sympy_method"] = "symbolic"
                out["value"] = numval(r)
            if out["value"] is None:
                out["sympy_unevaluated"] = True
            return out

        C.add(
            "summation",
            "infinite",
            f_str,
            {"input": f_str, "variable": "k", "lower": str(lo), "upper": "oo"},
            compute,
        )

    divergent = ["1/k", "1/sqrt(k)", "k", "(k + 1)/k"]
    for f_str in divergent:
        f = sp.sympify(f_str)

        def compute(f=f):
            r = sp.summation(f, (k, 1, oo))
            return {"sympy_result": str(r), "value": numval(r)}

        C.add(
            "summation",
            "divergent",
            f_str,
            {"input": f_str, "variable": "k", "lower": "1", "upper": "oo"},
            compute,
        )

    prod_finite = ["k", "1 + 1/k", "1 - 1/k**2", "2", "k/(k + 1)", "(k + 1)/k", "2*k", "k**2"]
    for f_str in prod_finite:
        f = sp.sympify(f_str)
        lo = 2 if f_str == "1 - 1/k**2" else 1

        def compute(f=f, lo=lo):
            closed = sp.product(f, (k, lo, n))
            pts = []
            for nv in range(lo, lo + 6):
                pts.append({"subs": {"n": float(nv)}, "value": numval(brute_prod(f, lo, nv))})
            return {"sympy_result": str(closed), "eval_points": pts}

        C.add(
            "product",
            "finite_symbolic",
            f_str,
            {"input": f_str, "variable": "k", "lower": str(lo), "upper": "n"},
            compute,
        )

    prod_infinite = [("1 - 1/k**2", 2), ("1 + 1/k**2", 1), ("1 - 1/(4*k**2)", 1), ("(k**2 - 1)/(k**2 + 1)", 2)]
    for f_str, lo in prod_infinite:
        f = sp.sympify(f_str)

        def compute(f=f, lo=lo):
            r = sp.product(f, (k, lo, oo))
            out = {"sympy_result": str(r)}
            if r.has(sp.Product):
                val = sp.Product(f, (k, lo, oo)).evalf(N_DIGITS)
                out["sympy_method"] = "numeric"
                out["value"] = numval(val)
            else:
                out["sympy_method"] = "symbolic"
                out["value"] = numval(r)
            if out["value"] is None:
                out["sympy_unevaluated"] = True
            return out

        C.add(
            "product",
            "infinite",
            f_str,
            {"input": f_str, "variable": "k", "lower": str(lo), "upper": "oo"},
            compute,
        )

    convergence = [
        ("1/k**2", True),
        ("1/k", False),
        ("(-1)**k/k", True),
        ("1/(k*log(k)**2)", True),
        ("k/2**k", True),
        ("1/sqrt(k)", False),
        ("factorial(k)/k**k", True),
        ("1/(k**2 + 1)", True),
        ("k/(k**2 + 1)", False),
        ("1/(k*log(k))", False),
        ("(k/(k + 1))**k**2", True),
        ("2**k/factorial(k)", True),
        ("1/k**Rational(3, 2)", True),
        ("(-1)**k", False),
        ("log(k)/k**2", True),
        ("k**2/(k**4 + 1)", True),
    ]
    for f_str, expected in convergence:
        f = sp.sympify(f_str)

        def compute(f=f, expected=expected):
            r = sp.Sum(f, (k, 2, oo)).is_convergent()
            out = {"sympy_result": str(r)}
            if r in (S.true, S.false):
                assert bool(r) == expected, (f_str, r, expected)
                out["value_bool"] = bool(r)
            else:
                out["value_bool"] = expected
                out["sympy_note"] = "SymPy undecided; expected value from analysis"
            return out

        C.add("summation", "convergence", f_str, {"input": str(f), "variable": "k"}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# calculus: one-sided limits, series at infinity, residues
# ═══════════════════════════════════════════════════════════════════════════


def gen_limits_series_residues():
    onesided = [
        ("1/x", "0"),
        ("abs(x)/x", "0"),
        ("floor(x)", "2"),
        ("exp(1/x)", "0"),
        ("1/(x - 1)**2", "1"),
        ("tan(x)", "pi/2"),
        ("(x**2 - 1)/abs(x - 1)", "1"),
        ("1/(1 + exp(1/x))", "0"),
        ("atan(1/x)", "0"),
        ("ceiling(x)", "1"),
        ("x/abs(x) + x", "0"),
        ("sqrt(x**2)/x", "0"),
        ("log(x)", "0"),
        ("1/(x**2 - 4)", "2"),
        ("(x - 3)/abs(x - 3)*sin(x)", "3"),
        ("x*floor(1/x)", "0"),
        ("sign(x)*(1 + x)", "0"),
        ("exp(-1/x**2)", "0"),
        ("1/(exp(1/x) - 1)", "0"),
        ("floor(x) + floor(-x)", "1"),
        ("(1 - cos(x))/x", "0"),
        ("1/x**3", "0"),
    ]
    for f_str, pt in onesided:
        f = sp.sympify(f_str)
        p = sp.sympify(pt)
        for direction, sub in (("+", "right"), ("-", "left")):

            def compute(f=f, p=p, direction=direction):
                r = sp.limit(f, x, p, direction)
                out = {"sympy_result": str(r), "value": numval(r)}
                if out["value"] is None or r.has(sp.Limit):
                    out["sympy_unevaluated"] = True
                return out

            C.add(
                "limit",
                sub,
                f"{f_str} @ {pt}",
                {"input": f_str, "variable": "x", "point": pt, "direction": sub},
                compute,
            )

    series_inf = [
        "sqrt(x**2 + 1)",
        "x/(x + 1)",
        "exp(1/x)*x",
        "atan(x)",
        "log(1 + 1/x)",
        "x*sin(1/x)",
        "(x**2 + 3*x + 1)/(x - 2)",
        "x**2/(x**2 + 1)",
        "sqrt(x**2 + x) - x",
        "x*(exp(1/x) - 1)",
        "1/(x**2 - x)",
        "(x + 1)**3/x**2",
        "x**2*cos(1/x)",
    ]
    for f_str in series_inf:
        f = sp.sympify(f_str)

        def compute(f=f):
            ser = sp.series(f, x, oo, 8).removeO()
            coeffs = []
            for term in sp.Add.make_args(sp.expand(ser)):
                c, pw = term.as_independent(x, as_Add=False)
                if pw == 1:
                    e = 0
                elif pw == x:
                    e = 1
                elif pw.is_Pow and pw.base == x:
                    e = pw.exp
                else:
                    raise ValueError(f"unexpected series term {term}")
                coeffs.append({"exp": float(e), "value": numval(c)})
            coeffs.sort(key=lambda d: -d["exp"])
            return {"sympy_result": str(ser), "coefficients": coeffs}

        C.add(
            "series_at_infinity",
            "laurent",
            f_str,
            {"input": f_str, "variable": "x", "n_terms": 6},
            compute,
        )

    residues = [
        ("1/z", "0"),
        ("exp(z)/z**2", "0"),
        ("exp(z)/(z**2 + 1)", "I"),
        ("1/(z**2 + 1)**2", "I"),
        ("cos(z)/sin(z)", "0"),
        ("1/(z*sin(z))", "0"),
        ("z**2*exp(1/z)", "0"),
        ("1/(z**4 + 1)", "exp(I*pi/4)"),
        ("z/((z - 1)*(z - 2)**2)", "2"),
        ("z/((z - 1)*(z - 2)**2)", "1"),
        ("1/(z**3 - z)", "1"),
        ("exp(2*z)/(z - 1)**3", "1"),
        ("1/(z**2 + 4)", "2*I"),
        ("z*exp(z)/(z**2 - 2*z + 5)", "1 + 2*I"),
        ("1/sin(z)", "pi"),
        ("1/(z**2*(z - 3))", "0"),
        ("cot(z)/z**2", "0"),
        ("1/(exp(z) - 1)", "0"),
        ("z**3/(z**2 + 1)", "oo"),
        ("1/z", "oo"),
        ("(z**2 + 1)/(z**3 - z)", "oo"),
        ("exp(1/z)", "oo"),
        ("1/(z - 1)**2", "oo"),
        ("z", "oo"),
    ]
    for f_str, pt in residues:
        f = sp.sympify(f_str)
        sub = "infinity" if pt == "oo" else "finite"

        def compute(f=f, pt=pt):
            if pt == "oo":
                # Res_{z=∞} f = −Res_{t=0} f(1/t)/t²
                tt = symbols("tt")
                r = -sp.residue(f.subs(z, 1 / tt) / tt**2, tt, 0)
            else:
                r = sp.residue(f, z, sp.sympify(pt))
            return {"sympy_result": str(r), "value": numval(r)}

        C.add(
            "residue",
            sub,
            f"{f_str} @ {pt}",
            {"input": f_str, "variable": "z", "point": pt},
            compute,
        )


# ═══════════════════════════════════════════════════════════════════════════
# transforms: Laplace, Fourier (ordinary convention), Mellin
# ═══════════════════════════════════════════════════════════════════════════


def gen_transforms():
    S_PTS = [{s: Rational(3, 2)}, {s: 2}, {s: Rational(37, 10)}, {s: 5}]
    laplace = [
        "1",
        "t",
        "t**3",
        "exp(2*t)",
        "exp(-3*t)*t**2",
        "sin(2*t)",
        "cos(3*t)",
        "sinh(t)",
        "cosh(2*t)",
        "t*sin(t)",
        "t*cos(2*t)",
        "exp(-t)*sin(3*t)",
        "exp(2*t)*cos(t)",
        "t**2*exp(t)",
        "sin(t)**2",
        "Heaviside(t - 2)",
        "exp(-t)*Heaviside(t - 1)",
        "DiracDelta(t - 1)",
        "sqrt(t)",
        "1/sqrt(t)",
        "t*exp(-2*t)*sin(t)",
        "sin(t)*cosh(t)",
        "t**4*exp(-t)",
        "cos(t)**2",
        "(1 - exp(-t))/t",
        "exp(-t**2)",
    ]
    for f_str in laplace:
        f = sp.sympify(f_str)

        def compute(f=f):
            r = sp.laplace_transform(f, t, s, noconds=True)
            out = {"sympy_result": str(r)}
            if r.has(sp.LaplaceTransform):
                out["sympy_unevaluated"] = True
                return out
            out["eval_points"] = eval_points(r, S_PTS)
            return out

        C.add("laplace", "forward", f_str, {"input": f_str, "time_var": "t", "freq_var": "s"}, compute)

    K_PTS = [{k: Rational(1, 4)}, {k: Rational(7, 10)}, {k: Rational(-13, 10)}, {k: 2}]
    fourier = [
        "exp(-x**2)",
        "exp(-abs(x))",
        "Heaviside(x)*exp(-2*x)",
        "Heaviside(x + 1) - Heaviside(x - 1)",
        "x*exp(-x**2)",
        "exp(-3*abs(x))",
        "exp(-pi*x**2)",
        "exp(-x**2/2)",
        "Heaviside(x)*x*exp(-x)",
        "exp(-2*x**2 + 3*x)",
        "exp(-abs(x))*cos(x)",
        "Heaviside(x)*exp(-x)*cos(2*x)",
    ]
    for f_str in fourier:
        f = sp.sympify(f_str)

        def compute(f=f):
            r = sp.fourier_transform(f, x, k)
            out = {"sympy_result": str(r)}
            if r.has(sp.FourierTransform):
                out["sympy_unevaluated"] = True
                return out
            out["eval_points"] = eval_points(r, K_PTS)
            return out

        C.add(
            "fourier",
            "ordinary",
            f_str,
            {"input": f_str, "time_var": "x", "freq_var": "k", "convention": "ordinary"},
            compute,
        )

    mellin = [
        "exp(-x)",
        "exp(-x**2)",
        "1/(1 + x)",
        "1/(1 + x)**3",
        "x**2*exp(-3*x)",
        "exp(-2*x)",
        "1/(1 + x)**2",
        "Heaviside(1 - x)",
        "Heaviside(1 - x)*(1 - x)**2",
        "x*exp(-x)",
        "log(1 + x)",
        "1/(1 + x**2)",
        "exp(-x)*x**Rational(1, 2)",
        "Heaviside(1 - x)*x**3",
    ]
    for f_str in mellin:
        f = sp.sympify(f_str)

        def compute(f=f):
            F, (lo, hi), cond = sp.mellin_transform(f, x, s)
            out = {"sympy_result": str(F), "strip": [bound_str(lo), bound_str(hi)]}
            if F.has(sp.MellinTransform):
                out["sympy_unevaluated"] = True
                return out
            # Sample s strictly inside the fundamental strip.
            lo_f = -4 if lo == -oo else float(lo)
            hi_f = lo_f + 4 if hi == oo else float(hi)
            pts = []
            for frac in (Rational(1, 3), Rational(1, 2), Rational(3, 4)):
                sv = sp.nsimplify(lo_f) + (sp.nsimplify(hi_f) - sp.nsimplify(lo_f)) * frac
                pts.append({s: sv})
            out["eval_points"] = eval_points(F, pts)
            return out

        C.add("mellin", "forward", f_str, {"input": str(f), "space_var": "x", "freq_var": "s"}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# solve: general (periodic) solutions, linsolve, nonlinear systems, ODE IVPs,
# recurrences
# ═══════════════════════════════════════════════════════════════════════════


def enumerate_real_solutions(sol_set, window, n_range=range(-8, 9)):
    """Enumerate a solveset result (FiniteSet / ImageSet / Union) inside
    [-window, window]."""
    vals = []

    def walk(sset):
        if isinstance(sset, FiniteSet):
            for e in sset:
                vals.append(float(sp.N(e, 30)))
        elif isinstance(sset, ImageSet):
            lam = sset.lamda
            for nv in n_range:
                vals.append(float(sp.N(lam(nv), 30)))
        elif isinstance(sset, Union):
            for part in sset.args:
                walk(part)
        elif sset == S.EmptySet:
            pass
        else:
            raise ValueError(f"cannot enumerate {sset}")

    walk(sol_set)
    vals = sorted(v for v in vals if abs(v) <= window + 1e-9)
    out = []
    for v in vals:
        if not out or abs(v - out[-1]) > 1e-9:
            out.append(v)
    return out


def gen_solve():
    general = [
        "sin(x) - 1/2",
        "cos(x) - 1/2",
        "tan(x) - 1",
        "cos(2*x) - 1/2",
        "sin(3*x)",
        "sin(x)**2 - sin(x)",
        "2*cos(x)**2 - 1",
        "sin(x) + cos(x)",
        "tan(x)**2 - 3",
        "sin(2*x) - cos(x)",
        "cos(x)**2 - 3/4",
        "sin(x - pi/3) - sqrt(3)/2",
    ]
    WINDOW = 10.0
    for eq_str in general:
        f = sp.sympify(eq_str)

        def compute(f=f):
            sol = sp.solveset(f, x, S.Reals)
            vals = enumerate_real_solutions(sol, WINDOW)
            return {"sympy_result": str(sol), "solutions_in_window": vals}

        C.add(
            "solve",
            "general_periodic",
            eq_str,
            {"input": eq_str, "variable": "x", "window": WINDOW},
            compute,
        )

    # linsolve: unique numeric
    lin_unique = [
        (["x + y + z - 6", "2*x - y + z - 3", "x + 2*y - z - 2"], ["x", "y", "z"]),
        (["3*x + 2*y - 7", "x - y - 1"], ["x", "y"]),
        (["x + 2*y + 3*z + 4*w - 10", "2*x - y + z - w - 1", "x + y - z + 2*w - 5", "3*x + y + 2*z - w - 8"], ["x", "y", "z", "w"]),
        (["x/2 + y/3 - 1", "x/4 - y/5 - 2"], ["x", "y"]),
        (["2*x + 3*y - z - 5", "4*x - y + 2*z - 6", "-x + 5*y + 3*z - 7"], ["x", "y", "z"]),
        (["x + y - 2", "x - y", "2*x + 2*y - 4"], ["x", "y"]),  # over-determined, consistent
        (["7*x - 3*y + 2*z - 4", "x + y + z - 1", "x - 2*y"], ["x", "y", "z"]),
        (["x + y + z + w - 1", "x - y + z - w - 2", "x + y - z - w - 3", "x - y - z + w - 4"], ["x", "y", "z", "w"]),
    ]
    for eqs, vars_ in lin_unique:
        syms = [sp.Symbol(v) for v in vars_]
        eq_exprs = [sp.sympify(e) for e in eqs]

        def compute(eq_exprs=eq_exprs, syms=syms):
            sol = sp.linsolve(eq_exprs, syms)
            assert len(sol) == 1, sol
            tup = list(sol)[0]
            values = {str(v): numval(val) for v, val in zip(syms, tup)}
            return {"sympy_result": str(sol), "eval_points": [{"subs": {}, "values": values}]}

        C.add(
            "linsolve",
            "unique",
            " ; ".join(eqs),
            {"equations": eqs, "variables": vars_},
            compute,
        )

    # linsolve: symbolic coefficients
    lin_sym = [
        (["a*x + y - 1", "x - y - b"], ["x", "y"], ["a", "b"]),
        (["a*x + b*y - 1", "b*x - a*y"], ["x", "y"], ["a", "b"]),
        (["x + a*y - 2", "a*x + y - 3"], ["x", "y"], ["a"]),
        (["x + y + z - a", "x - y - b", "y + z - 1"], ["x", "y", "z"], ["a", "b"]),
    ]
    param_pts = [
        {a: Rational(3, 2), b: Rational(-1, 2)},
        {a: 3, b: 2},
        {a: Rational(-7, 5), b: Rational(2, 3)},
    ]
    for eqs, vars_, params in lin_sym:
        syms = [sp.Symbol(v) for v in vars_]
        eq_exprs = [sp.sympify(e) for e in eqs]

        def compute(eq_exprs=eq_exprs, syms=syms):
            sol = sp.linsolve(eq_exprs, syms)
            assert len(sol) == 1, sol
            tup = list(sol)[0]
            pts = []
            for p in param_pts:
                values = {str(v): numval(val.subs(p)) for v, val in zip(syms, tup)}
                pts.append({"subs": subs_json({kk: v for kk, v in p.items() if str(kk) in params}), "values": values})
            return {"sympy_result": str(sol), "eval_points": pts}

        C.add(
            "linsolve",
            "symbolic",
            " ; ".join(eqs),
            {"equations": eqs, "variables": vars_, "parameters": params},
            compute,
        )

    # linsolve: parametric (under-determined) and inconsistent
    lin_param = [
        (["x + y + z - 1", "x - y + 2*z"], ["x", "y", "z"]),
        (["x + 2*y - 3*z + w - 4", "2*x + 4*y - 6*z + 2*w - 8"], ["x", "y", "z", "w"]),
        (["x + y - 3"], ["x", "y"]),
        (["x + y + z", "2*x + 2*y + 2*z"], ["x", "y", "z"]),
        (["x - 2*y + z - w", "y + z + w - 1"], ["x", "y", "z", "w"]),
    ]
    for eqs, vars_ in lin_param:
        syms = [sp.Symbol(v) for v in vars_]
        eq_exprs = [sp.sympify(e) for e in eqs]

        def compute(eq_exprs=eq_exprs, syms=syms):
            sol = sp.linsolve(eq_exprs, syms)
            tup = list(sol)[0]
            free = set()
            for val in tup:
                free |= val.free_symbols & set(syms)
            return {"sympy_result": str(sol), "free_count": len(free)}

        C.add(
            "linsolve",
            "parametric",
            " ; ".join(eqs),
            {"equations": eqs, "variables": vars_},
            compute,
        )
    lin_incons = [
        (["x + y - 1", "x + y - 2"], ["x", "y"]),
        (["x + y + z - 1", "2*x + 2*y + 2*z - 3"], ["x", "y", "z"]),
        (["x - y", "x + y - 2", "x - 5"], ["x", "y"]),
    ]
    for eqs, vars_ in lin_incons:
        syms = [sp.Symbol(v) for v in vars_]
        eq_exprs = [sp.sympify(e) for e in eqs]

        def compute(eq_exprs=eq_exprs, syms=syms):
            sol = sp.linsolve(eq_exprs, syms)
            assert sol == S.EmptySet, sol
            return {"sympy_result": str(sol)}

        C.add(
            "linsolve",
            "inconsistent",
            " ; ".join(eqs),
            {"equations": eqs, "variables": vars_},
            compute,
        )

    # nonlinear polynomial systems
    nonlinear = [
        (["x**2 + y**2 - 5", "x*y - 2"], ["x", "y"]),
        (["x**2 + y**2 - 1", "y - x"], ["x", "y"]),
        (["x**2 - y", "y**2 - x"], ["x", "y"]),
        (["x + y - 3", "x*y - 2"], ["x", "y"]),
        (["x**2 + y - 3", "x - y**2 + 1"], ["x", "y"]),
        (["x**2 + y**2 + z**2 - 3", "x + y + z - 3", "x - y"], ["x", "y", "z"]),
        (["x**2 - 2*x*y + y**2 - 1", "x + y - 1"], ["x", "y"]),
        (["x*y - 1", "x**2 - y**2 - 3"], ["x", "y"]),
        (["x**3 - y", "x - y + 6"], ["x", "y"]),
        (["x**2 + y**2 - 4", "x**2 - y - 2"], ["x", "y"]),
    ]
    for eqs, vars_ in nonlinear:
        syms = [sp.Symbol(v) for v in vars_]
        eq_exprs = [sp.sympify(e) for e in eqs]

        def compute(eq_exprs=eq_exprs, syms=syms):
            sols = sp.solve(eq_exprs, syms, dict=True)
            tuples = []
            for d in sols:
                tuples.append([numval(d[v]) for v in syms])
            tuples.sort(key=lambda tup: [(v["re"], v["im"]) for v in tup])
            return {"sympy_result": str(sols), "solutions": tuples}

        C.add(
            "nonlinear_system",
            "polynomial",
            " ; ".join(eqs),
            {"equations": eqs, "variables": vars_},
            compute,
        )

    # ODE initial-value problems
    yf = Function("y")
    X_PTS = [{x: Rational(1, 2)}, {x: 1}, {x: Rational(-3, 4)}, {x: 2}]
    odes = [
        # kind linear: coeffs[i] multiplies y^{(i)}; rhs on the right-hand side
        ({"kind": "linear", "coeffs": ["0", "1"], "rhs": "x*y"}, [[0, "0", "2"]], "first_order_separable"),
        ({"kind": "linear", "coeffs": ["2", "3", "1"], "rhs": "0"}, [[0, "0", "1"], [1, "0", "0"]], "second_order_const"),
        ({"kind": "linear", "coeffs": ["1", "0", "1"], "rhs": "0"}, [[0, "0", "0"], [1, "0", "1"]], "harmonic"),
        ({"kind": "linear", "coeffs": ["4", "4", "1"], "rhs": "0"}, [[0, "0", "1"], [1, "0", "1"]], "repeated_root"),
        ({"kind": "linear", "coeffs": ["-1", "1"], "rhs": "exp(x)"}, [[0, "0", "1"]], "first_order_linear_forced"),
        ({"kind": "linear", "coeffs": ["1", "0", "1"], "rhs": "x"}, [[0, "0", "1"], [1, "0", "0"]], "harmonic_forced"),
        ({"kind": "linear", "coeffs": ["-2", "-1", "1"], "rhs": "0"}, [[0, "0", "3"], [1, "0", "0"]], "second_order_distinct"),
        ({"kind": "linear", "coeffs": ["2", "1"], "rhs": "sin(x)"}, [[0, "0", "0"]], "first_order_forced_sin"),
        ({"kind": "linear", "coeffs": ["5", "2", "1"], "rhs": "0"}, [[0, "0", "1"], [1, "0", "1"]], "complex_roots"),
        ({"kind": "linear", "coeffs": ["0", "1", "1"], "rhs": "1"}, [[0, "0", "0"], [1, "0", "0"]], "second_order_forced_const"),
        ({"kind": "linear", "coeffs": ["1", "1"], "rhs": "x**2"}, [[0, "0", "1"]], "first_order_poly_forced"),
        ({"kind": "linear", "coeffs": ["-1", "0", "0", "1"], "rhs": "0"}, [[0, "0", "1"], [1, "0", "0"], [2, "0", "0"]], "third_order"),
    ]
    for ode, ics, label in odes:
        coeffs = [sp.sympify(c) for c in ode["coeffs"]]
        rhs = sp.sympify(ode["rhs"])

        def compute(coeffs=coeffs, rhs=rhs, ics=ics):
            lhs = sum((c * sp.diff(yf(x), x, i) if i > 0 else c * yf(x)) for i, c in enumerate(coeffs))
            rhs_f = rhs.subs(y, yf(x))
            eq = Eq(lhs, rhs_f)
            ics_d = {}
            for order, x0, val in ics:
                x0e = sp.sympify(x0)
                lhs_ic = yf(x).diff(x, order).subs(x, x0e) if order > 0 else yf(x0e)
                ics_d[lhs_ic] = sp.sympify(val)
            sol = sp.dsolve(eq, yf(x), ics=ics_d)
            expr = sol.rhs
            # Verify SymPy's own solution.
            resid = sp.simplify((lhs - rhs_f).subs(yf(x), expr).doit())
            assert resid == 0, (label, resid)
            return {"sympy_result": str(expr), "eval_points": eval_points(expr, X_PTS)}

        C.add(
            "ode_ivp",
            "linear",
            label,
            {"ode": ode, "ics": ics, "func": "y", "variable": "x"},
            compute,
        )

    # Linear recurrences: reference values by direct iteration.
    recurrences = [
        # coeffs c0..ck : c0 a(n) + c1 a(n+1) + ... + ck a(n+k) = forcing(n)
        (["-1", "-1", "1"], None, ["0", "1"], "fibonacci"),
        (["-2", "1"], "1", ["0"], "hanoi"),
        (["-3", "1"], None, ["2"], "geometric_3"),
        (["2", "-3", "1"], None, ["1", "2"], "roots_1_2"),
        (["1", "-2", "1"], None, ["1", "3"], "double_root_1"),
        (["-6", "11", "-6", "1"], None, ["1", "0", "0"], "third_order_int_roots"),
        (["-6", "5", "-1", "1"], None, ["1", "0", "0"], "third_order_irrational_roots"),
        (["-2", "1"], "n", ["1"], "forced_poly"),
        (["-1", "1"], "2**n", ["0"], "forced_exp"),
        (["1", "0", "1"], None, ["1", "0"], "complex_roots_period4"),
        (["-4", "0", "1"], None, ["1", "2"], "roots_pm2"),
        (["-1", "-1", "1"], None, ["2", "1"], "lucas"),
        (["-1", "1"], "n**2", ["0"], "sum_of_squares"),
    ]
    for coeffs, forcing, ics, label in recurrences:

        def compute(coeffs=coeffs, forcing=forcing, ics=ics):
            cs = [Fraction(c) for c in coeffs]
            order = len(cs) - 1
            seq = [Fraction(v) for v in ics]
            f_expr = sp.sympify(forcing) if forcing else None
            for m in range(order, 12):
                # c_k a(m) = f(m-k) - Σ_{i<k} c_i a(m-k+i)
                base = m - order
                rhs_val = Fraction(0)
                if f_expr is not None:
                    fv = f_expr.subs(n, base)
                    rhs_val = Fraction(int(fv.p), int(fv.q))
                acc = rhs_val
                for i in range(order):
                    acc -= cs[i] * seq[base + i]
                seq.append(acc / cs[order])
            return {"values": [numval(Rational(v.numerator, v.denominator)) for v in seq]}

        C.add(
            "rsolve",
            "linear",
            label,
            {"coeffs": coeffs, "forcing": forcing, "ics": ics, "variable": "n"},
            compute,
        )


# ═══════════════════════════════════════════════════════════════════════════
# polynomials
# ═══════════════════════════════════════════════════════════════════════════


def gen_poly():
    multivariate = [
        "x**2 - y**2",
        "x**2*y - y**3 + 2*x**2 - 2*y**2",
        "x**3 + y**3",
        "x**4 - y**4",
        "x**2 + 2*x*y + y**2 - 1",
        "x**3*y - x*y**3",
        "2*x**2*y**2 - 8",
        "x**2*y**2 - x**2 - y**2 + 1",
        "x**3 - 3*x**2*y + 3*x*y**2 - y**3",
        "x**4 + 2*x**2*y**2 + y**4",
        "6*x**2*y + 9*x*y**2 - 4*x - 6*y",
        "x**2*z - y**2*z + x**2 - y**2",
    ]
    for f_str in multivariate:
        f = sp.sympify(f_str)
        pts = PTS_XY if z not in f.free_symbols else [
            {x: Rational(3, 2), y: Rational(1, 2), z: Rational(-2, 3)},
            {x: Rational(7, 10), y: Rational(-3, 10), z: 2},
        ]

        def compute(f=f, pts=pts):
            content, factors = sp.factor_list(f)
            return {
                "sympy_result": str(sp.factor(f)),
                "content": str(content),
                "factors": [[str(fac), int(m)] for fac, m in factors],
                "eval_points": eval_points(f, pts),
            }

        C.add("factor", "multivariate", f_str, {"input": f_str}, compute)

    univariate = [
        "2*x**3 - 2*x**2 - 2*x + 2",
        "x**4 - 1",
        "x**6 - 1",
        "x**4 + 4",
        "x**5 - x**4 - x + 1",
        "12*x**2 - 7*x - 12",
        "x**4 - 5*x**2 + 4",
        "x**8 - 1",
        "x**3 - 6*x**2 + 11*x - 6",
        "3*x**4 + 6*x**3 - 3*x**2 - 6*x",
    ]
    for f_str in univariate:
        f = sp.sympify(f_str)

        def compute(f=f):
            content, factors = sp.factor_list(f, x)
            return {
                "sympy_result": str(sp.factor(f)),
                "content": str(content),
                "factors": [[str(fac), int(m)] for fac, m in factors],
                "eval_points": eval_points(f, PTS_X),
            }

        C.add("factor", "univariate_list", f_str, {"input": f_str, "variable": "x"}, compute)

    resultants = [
        ("x**2 - 1", "x - 1"),
        ("x**2 - 1", "x - 3"),
        ("x**2 + x + 1", "x**2 - x + 1"),
        ("x**3 - 2", "x**2 - 3"),
        ("2*x**2 + 3*x - 5", "x**3 + x + 1"),
        ("x**4 + 1", "x**2 + 2"),
        ("x**2 - 2*x + 1", "x**2 - 1"),
        ("x**5 - x - 1", "x**2 - 2"),
    ]
    for f_str, g_str in resultants:
        f, g = sp.sympify(f_str), sp.sympify(g_str)

        def compute(f=f, g=g):
            r = sp.resultant(f, g, x)
            return {"sympy_result": str(r), "value": numval(r)}

        C.add("resultant", "integer", f"{f_str} , {g_str}", {"input_f": f_str, "input_g": g_str, "variable": "x"}, compute)

    res_sym = [("x**2 + a*x + b", "x - 1"), ("x**2 + a", "x**2 + b*x + 1"), ("a*x**2 + b*x + 1", "x**3 - a")]
    ab_pts = [{a: Rational(3, 2), b: Rational(-1, 2)}, {a: 2, b: 3}, {a: Rational(-7, 5), b: Rational(2, 3)}]
    for f_str, g_str in res_sym:
        f, g = sp.sympify(f_str), sp.sympify(g_str)

        def compute(f=f, g=g):
            r = sp.resultant(f, g, x)
            return {"sympy_result": str(r), "eval_points": eval_points(r, ab_pts)}

        C.add("resultant", "symbolic", f"{f_str} , {g_str}", {"input_f": f_str, "input_g": g_str, "variable": "x"}, compute)

    discs = [
        "x**2 + 3*x + 1",
        "x**2 - 2*x + 1",
        "x**3 - x",
        "x**3 + x + 1",
        "x**4 - 1",
        "2*x**3 - 3*x**2 + 5",
        "x**5 - x - 1",
        "x**4 + x**2 + 1",
    ]
    for f_str in discs:
        f = sp.sympify(f_str)

        def compute(f=f):
            r = sp.discriminant(f, x)
            return {"sympy_result": str(r), "value": numval(r)}

        C.add("discriminant", "integer", f_str, {"input": f_str, "variable": "x"}, compute)
    disc_sym = ["a*x**2 + b*x + 1", "x**3 + a*x + b", "x**2 + a*x + b"]
    for f_str in disc_sym:
        f = sp.sympify(f_str)

        def compute(f=f):
            r = sp.discriminant(f, x)
            return {"sympy_result": str(r), "eval_points": eval_points(r, ab_pts)}

        C.add("discriminant", "symbolic", f_str, {"input": f_str, "variable": "x"}, compute)

    sqf = [
        "x**5 - x**4 - x + 1",
        "x**6 - 2*x**5 + x**4",
        "(x - 1)**3*(x + 2)**2*(x**2 + 1)",
        "x**4 - 2*x**2 + 1",
        "2*x**5 + 4*x**4 + 2*x**3",
        "x**7 - 3*x**6 + 3*x**5 - x**4",
        "x**3 - 3*x + 2",
        "x**8 - 2*x**4 + 1",
    ]
    for f_str in sqf:
        f = sp.expand(sp.sympify(f_str))

        def compute(f=f):
            content, parts = sp.sqf_list(f, x)
            return {
                "sympy_result": str(sp.sqf(f)),
                "content": str(content),
                "parts": [[str(p), int(m), int(sp.degree(p, x))] for p, m in parts],
                "eval_points": eval_points(f, PTS_X),
            }

        C.add("sqf_list", "univariate", f_str, {"input": str(f), "variable": "x"}, compute)

    nroots = [
        "x**2 + 1",
        "x**5 - x - 1",
        "x**3 - 2*x - 5",
        "x**4 - 10*x**2 + 1",
        "x**6 + x**3 + 1",
        "x**3 + x**2 - 2*x - 1",
        "2*x**4 - 3*x**3 + x - 7",
        "x**7 - 7*x + 3",
        "x**4 + 4",
        "x**2 - 2*x + 1",
        "(x**2 + 1)**2*(x - 3)",
        "x**8 - 1",
    ]
    for f_str in nroots:
        f = sp.expand(sp.sympify(f_str))

        def compute(f=f):
            roots = sp.Poly(f, x).all_roots(radicals=False)  # exact CRootOf, multiplicities kept
            vals = [numval(r) for r in roots]
            vals.sort(key=lambda v: (round(v["re"], 9), round(v["im"], 9)))
            return {"roots": vals}

        C.add("nroots", "complex", f_str, {"input": str(f), "variable": "x"}, compute)

    gcds = [
        ("x**4 - 1", "x**6 - 1"),
        ("x**2 - 1", "x**2 - x"),
        ("x**3 - 3*x**2 + 3*x - 1", "x**2 - 2*x + 1"),
        ("2*x**2 + 2*x", "4*x**3 + 4*x**2"),
        ("x**5 - x", "x**3 - x"),
        ("x**2 + 2*x + 1", "x**2 - 1"),
        ("x**4 + 2*x**2 + 1", "x**3 + x"),
        ("6*x**3 - 6*x", "9*x**2 - 9"),
    ]
    for f_str, g_str in gcds:
        f, g = sp.sympify(f_str), sp.sympify(g_str)
        for op in ("gcd", "lcm"):

            def compute(f=f, g=g, op=op):
                r = sp.gcd(f, g) if op == "gcd" else sp.lcm(f, g)
                return {"sympy_result": str(r), "degree": int(sp.degree(r, x)), "eval_points": eval_points(r, PTS_X)}

            C.add("poly_gcd_lcm", op, f"{f_str} , {g_str}", {"input_f": f_str, "input_g": g_str, "variable": "x"}, compute)

    aparts = [
        "(x**2 + 1)/(x*(x - 1)**2)",
        "1/(x**2 - 1)",
        "(3*x + 5)/(x**2 + 3*x + 2)",
        "x**2/(x**2 - 4)",
        "1/(x*(x**2 + 1))",
        "(2*x**3 + x)/(x**2 - 1)",
        "1/((x - 1)*(x - 2)*(x - 3))",
        "(x + 1)/(x**2*(x - 1))",
        "1/(x**2 + 1)**2",
        "(x**2 + 2)/((x**2 + 1)*(x + 3))",
    ]
    for f_str in aparts:
        f = sp.sympify(f_str)

        def compute(f=f):
            r = sp.apart(f, x)
            return {"sympy_result": str(r), "eval_points": eval_points(r, PTS_X)}

        C.add("apart", "univariate", f_str, {"input": f_str, "variable": "x"}, compute)

    divs = [
        ("x**4 + 3*x**2 + 1", "x**2 + x + 1"),
        ("x**5 - 1", "x - 1"),
        ("2*x**3 + x - 4", "x**2 - 2"),
        ("x**6", "x**2 + 1"),
        ("3*x**4 - 2*x**3 + x", "2*x**2 + 1"),
        ("x**3 + 2*x**2 + 3*x + 4", "x + 1"),
    ]
    for f_str, g_str in divs:
        f, g = sp.sympify(f_str), sp.sympify(g_str)

        def compute(f=f, g=g):
            q, r = sp.div(f, g, x)
            return {
                "quotient": str(q),
                "remainder": str(r),
                "quotient_points": eval_points(q, PTS_X),
                "remainder_points": eval_points(r, PTS_X),
            }

        C.add("poly_div", "univariate", f"{f_str} , {g_str}", {"input_f": f_str, "input_g": g_str, "variable": "x"}, compute)

    decomps = ["x**4 + 2*x**2 + 1", "x**4 - 2*x**3 + x**2", "x**6 + 3*x**3 + 1", "(x**2 + 1)**3", "x**8 + 4*x**4 + 4", "x**4 + 4*x**3 + 6*x**2 + 4*x + 5"]
    for f_str in decomps:
        f = sp.expand(sp.sympify(f_str))

        def compute(f=f):
            parts = sp.decompose(f, x)
            return {"sympy_result": [str(p) for p in parts], "component_count": len(parts), "eval_points": eval_points(f, PTS_X)}

        C.add("decompose", "univariate", f_str, {"input": str(f), "variable": "x"}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# complex analysis & special-function numerics
# ═══════════════════════════════════════════════════════════════════════════


def gen_complex_special():
    points = ["1 + 2*I", "-3 + I/2", "2 - I", "-1 - I", "I", "3/2"]
    exprs = ["exp(z)", "z**2 + 1/z", "sin(z)", "log(z)", "sqrt(z)", "z*exp(I*z)", "cosh(z)", "z**3 - 2*z", "1/(z**2 + 1)", "exp(z)*sin(z)"]
    ops = ["re", "im", "conjugate", "arg", "abs"]
    i = 0
    for e_str in exprs:
        e = sp.sympify(e_str)
        for op in ops:
            pt = points[i % len(points)]
            i += 1
            pte = sp.sympify(pt)
            if e_str == "1/(z**2 + 1)" and pte == I:
                pt, pte = "2 - I", sp.sympify("2 - I")

            def compute(e=e, op=op, pte=pte):
                val = e.subs(z, pte)
                fn = {"re": sp.re, "im": sp.im, "conjugate": sp.conjugate, "arg": sp.arg, "abs": sp.Abs}[op]
                r = fn(val)
                return {"sympy_result": str(sp.N(r, 20)), "value": numval(r)}

            C.add("complex", op, f"{op}({e_str}) @ z={pt}", {"input": e_str, "variable": "z", "point": pt, "operation": op}, compute)

    expand_cx = [
        "exp(x + I*y)",
        "(x + I*y)**3",
        "sin(x + I*y)",
        "1/(x + I*y)",
        "cosh(x + I*y)",
        "(x + I*y)*exp(x + I*y)",
        "log(x + I*y)",
        "(x + I*y)**2*(x - I*y)",
    ]
    XY_C = [{xr: Rational(3, 2), yr: Rational(1, 2)}, {xr: Rational(-7, 10), yr: Rational(4, 5)}, {xr: 2, yr: -1}]
    for e_str in expand_cx:
        e = sp.sympify(e_str).subs({x: xr, y: yr})

        def compute(e=e):
            r = sp.expand_complex(e)
            pts = []
            for p in XY_C:
                pts.append({"subs": subs_json({x: p[xr], y: p[yr]}), "value": numval(e.subs(p))})
            return {"sympy_result": str(r), "eval_points": pts}

        C.add("complex", "expand_complex", e_str, {"input": e_str, "real_symbols": ["x", "y"]}, compute)

    specials = [
        "zeta(2)",
        "zeta(3)",
        "zeta(1/2)",
        "zeta(-3)",
        "zeta(7)",
        "zeta(0)",
        "zeta(-1)",
        "zeta(5/2)",
        "polygamma(1, 1)",
        "polygamma(1, 1/2)",
        "polygamma(2, 1/3)",
        "polygamma(3, 5/2)",
        "polygamma(1, 7)",
        "Si(2)",
        "Si(1/2)",
        "Si(10)",
        "Ci(3/2)",
        "Ci(1/10)",
        "Ci(20)",
        "Ei(2)",
        "Ei(-1)",
        "Ei(1/3)",
        "li(10)",
        "li(2)",
        "li(1000)",
        "EulerGamma",
        "Catalan",
        "GoldenRatio",
        "GoldenRatio**2 - GoldenRatio",
        "EulerGamma + Catalan*pi",
        "gamma(1/3)",
        "erf(1/2)",
        "digamma(1/3)",
        "loggamma(10)",
        "LambertW(1)",
        "zeta(2)*Catalan",
        "Si(pi)",
        "Ci(-3/2)",
        "polygamma(4, 1)",
        "Ei(5)",
    ]
    for e_str in specials:
        e = sp.sympify(e_str)

        def compute(e=e):
            v = sp.N(e, N_DIGITS)
            return {"sympy_result": str(v), "value": numval(e), "digits30": str(v)}

        C.add("special_func", "numeric", e_str, {"input": e_str}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# simplification: sqrtdenest, nsimplify, trigsimp, power/log rewriting
# ═══════════════════════════════════════════════════════════════════════════


def sqrt_depth(e):
    if e.is_Pow and e.exp == Rational(1, 2):
        return 1 + sqrt_depth(e.base)
    if not e.args:
        return 0
    return max(sqrt_depth(arg) for arg in e.args)


def gen_simplify():
    denest = [
        "sqrt(5 + 2*sqrt(6))",
        "sqrt(3 + 2*sqrt(2))",
        "sqrt(7 - 4*sqrt(3))",
        "sqrt(11 + 6*sqrt(2))",
        "sqrt(2 + sqrt(3))",
        "sqrt(9 - 4*sqrt(5))",
        "sqrt(17 + 12*sqrt(2))",
        "sqrt(6 + 2*sqrt(5)) + sqrt(6 - 2*sqrt(5))",
        "sqrt(10 + 2*sqrt(21))",
        "sqrt(4 + 2*sqrt(3))",
    ]
    for e_str in denest:
        e = sp.sympify(e_str)

        def compute(e=e):
            r = sp.sqrtdenest(e)
            return {"sympy_result": str(r), "value": numval(e), "sympy_sqrt_depth": sqrt_depth(r), "input_sqrt_depth": sqrt_depth(e)}

        C.add("simplify", "sqrtdenest", e_str, {"input": e_str}, compute)

    nsimp = [
        ("0.333333333333333", []),
        ("0.75", []),
        ("1.4142135623730951", ["sqrt(2)"]),
        ("3.141592653589793", ["pi"]),
        ("2.718281828459045", ["E"]),
        ("0.6180339887498949", ["GoldenRatio"]),
        ("1.7320508075688772", ["sqrt(3)"]),
        ("0.14285714285714285", []),
        ("6.283185307179586", ["pi"]),
        ("2.6457513110645907", ["sqrt(7)"]),
        ("0.8660254037844386", ["sqrt(3)"]),
        ("1.5707963267948966", ["pi"]),
    ]
    for f_str, consts in nsimp:
        v = sp.Float(f_str, 20)

        def compute(v=v, consts=consts):
            r = sp.nsimplify(v, [sp.sympify(c) for c in consts], tolerance=1e-10, rational=not consts)
            assert not r.has(sp.Float), r
            return {"sympy_result": str(r), "value": numval(r)}

        C.add("simplify", "nsimplify", f_str, {"input": f_str, "constants": consts}, compute)

    trig = [
        "sin(x)**2 + cos(x)**2",
        "sin(x)**4 - cos(x)**4",
        "sin(x)/cos(x)",
        "1 - 2*sin(x)**2",
        "2*sin(x)*cos(x)",
        "cos(x)**2 - sin(x)**2",
        "sin(x)**2*cos(x)**2",
        "(1 - cos(2*x))/2",
        "tan(x)*cos(x)",
        "sin(x + y) - sin(x)*cos(y) - cos(x)*sin(y)",
        "cosh(x)**2 - sinh(x)**2",
        "sin(2*x)/sin(x)",
        "1/cos(x)**2 - tan(x)**2",
        "sin(x)**3 + sin(x)*cos(x)**2",
        "cos(x)**4 - sin(x)**4 - cos(2*x)",
        "(sin(x) + cos(x))**2",
        "sin(x)*cos(y) + cos(x)*sin(y)",
        "cos(3*x) - 4*cos(x)**3 + 3*cos(x)",
        "tan(x) + 1/tan(x)",
        "sinh(2*x) - 2*sinh(x)*cosh(x)",
    ]
    for e_str in trig:
        e = sp.sympify(e_str)
        pts = PTS_XY if y in e.free_symbols else PTS_X

        def compute(e=e, pts=pts):
            r = sp.trigsimp(e)
            return {"sympy_result": str(r), "sympy_ops": int(sp.count_ops(r)), "input_ops": int(sp.count_ops(e)), "eval_points": eval_points(e, pts)}

        C.add("simplify", "trigsimp", e_str, {"input": e_str}, compute)

    power_log = [
        ("powsimp", "x**2*x**3"),
        ("powsimp", "x**a*x**b"),
        ("powsimp", "x**2*y**2"),
        ("powsimp", "(x*y)**2/x**2"),
        ("powsimp", "2**x*3**x"),
        ("powsimp", "x**(1/2)*x**(1/3)"),
        ("powsimp", "exp(x)*exp(y)"),
        ("powsimp", "x**y/x**2"),
        ("powdenest", "(x**2)**(1/2)"),
        ("powdenest", "(x**a)**b"),
        ("powdenest", "sqrt(x**4)"),
        ("powdenest", "(x**(1/2))**4"),
        ("powdenest", "((x**2)**3)**(1/6)"),
        ("logcombine", "log(x) + log(y)"),
        ("logcombine", "2*log(x) - log(y)"),
        ("logcombine", "log(x) - 3*log(y) + log(2)"),
        ("logcombine", "a*log(x) + b*log(y)"),
        ("logcombine", "log(x)/2 + log(y)/2"),
        ("expand_log", "log(x*y)"),
        ("expand_log", "log(x**3)"),
        ("expand_log", "log(x**2*y**3)"),
        ("expand_log", "log(x/y)"),
        ("expand_log", "log(sqrt(x)*y)"),
        ("expand_log", "log(2*x**a*y)"),
    ]
    AB_POS = {a: Rational(3, 2), b: Rational(-1, 3)}
    for op, e_str in power_log:
        e = sp.sympify(e_str)
        pts = [{**p, **AB_POS} for p in PTS_XY_POS] if e.free_symbols & {a, b} else (PTS_XY_POS if y in e.free_symbols else PTS_X_POS)

        def compute(e=e, op=op, pts=pts):
            if op == "powsimp":
                r = sp.powsimp(e, force=True)
            elif op == "powdenest":
                r = sp.powdenest(e, force=True)
            elif op == "logcombine":
                r = sp.logcombine(e, force=True)
            else:
                r = sp.expand_log(e, force=True)
            return {"sympy_result": str(r), "eval_points": eval_points(e, pts)}

        C.add("simplify", op, e_str, {"input": e_str, "positive_symbols": ["x", "y"]}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# sets, inequalities, boolean logic
# ═══════════════════════════════════════════════════════════════════════════


def iv(lo, hi, lo_open=False, hi_open=False):
    return {"interval": [bound_str(sp.sympify(lo)), bound_str(sp.sympify(hi)), lo_open, hi_open]}


def fs(*elems):
    return {"finite": [str(sp.sympify(e)) for e in elems]}


def op(name, *args):
    return {"op": name, "args": list(args)}


def build_sympy_set(tree):
    if "interval" in tree:
        lo, hi, lo_open, hi_open = tree["interval"]
        return Interval(sp.sympify(lo), sp.sympify(hi), lo_open, hi_open)
    if "finite" in tree:
        return FiniteSet(*[sp.sympify(e) for e in tree["finite"]])
    if "reals" in tree:
        return S.Reals
    if "empty" in tree:
        return S.EmptySet
    args = [build_sympy_set(a) for a in tree["args"]]
    name = tree["op"]
    if name == "union":
        return Union(*args)
    if name == "intersection":
        return Intersection(*args)
    if name == "complement":  # args[0] \ args[1]
        return Complement(args[0], args[1])
    if name == "symmetric_difference":
        return SymmetricDifference(args[0], args[1])
    raise ValueError(name)


SAMPLE_PTS = [
    "-5", "-3", "-5/2", "-2", "-3/2", "-1", "-1/2", "-1/10", "0", "1/10", "1/4", "1/3", "1/2",
    "1", "3/2", "2", "5/2", "3", "4", "5", "7", "10",
]


def gen_sets_logic():
    sets = [
        ("union_overlap", op("union", iv(0, 2), iv(1, 3, True, False))),
        ("union_disjoint", op("union", iv(0, 1, False, True), iv(2, 3))),
        ("union_touching", op("union", iv(0, 1, False, True), iv(1, 2))),
        ("union_touching_open", op("union", iv(0, 1, False, True), iv(1, 2, True, False))),
        ("intersection_overlap", op("intersection", iv(0, 2, True, False), iv(1, 3))),
        ("intersection_disjoint", op("intersection", iv(0, 1), iv(2, 3))),
        ("intersection_point", op("intersection", iv(0, 1), iv(1, 2))),
        ("complement_inner", op("complement", iv(0, 3), iv(1, 2, True, True))),
        ("complement_point", op("complement", iv(0, 5), fs(2))),
        ("complement_reals", op("complement", {"reals": True}, iv(0, 2))),
        ("complement_reals_open", op("complement", {"reals": True}, iv(-1, 1, True, True))),
        ("symdiff", op("symmetric_difference", iv(0, 3), iv(1, 5))),
        ("nested", op("intersection", op("union", iv(-2, 0), iv(1, 4)), op("complement", {"reals": True}, fs(2, 3)))),
        ("finite_union_interval", op("union", fs(-1, 5), iv(0, 1))),
        ("finite_minus_interval", op("complement", fs(0, 1, 2, 3), iv(1, 2, False, False))),
        ("half_lines", op("intersection", iv("-oo", 2, True, True), iv(-1, "oo", False, True))),
        ("union_three", op("union", iv(-5, -2), iv(-1, 1), iv(2, 5, True, True))),
        ("unbounded_union", op("union", iv("-oo", -1, True, False), iv(1, "oo", False, True))),
    ]
    for label, tree in sets:

        def compute(tree=tree):
            S_ = build_sympy_set(tree)
            membership = [bool(sp.sympify(p) in S_) for p in SAMPLE_PTS]
            return {"sympy_result": str(S_), "sample_points": SAMPLE_PTS, "membership": membership}

        C.add("sets", "interval_ops", label, {"set": tree}, compute)

    ineq = [
        ("quadratic_and_linear", [("x**2 - 4", ">", "0"), ("x", "<", "5")]),
        ("cubic_le", [("(x - 1)*(x - 2)*(x - 3)", "<=", "0")]),
        ("reciprocal_gt", [("1/x", ">", "2")]),
        ("abs_lt", [("abs(x - 1)", "<", "2")]),
        ("abs_ge", [("abs(x)", ">=", "3")]),
        ("rational_ge", [("(x - 1)/(x + 1)", ">=", "0")]),
        ("quadratic_no_solution", [("x**2 + 1", "<", "0")]),
        ("quadratic_all", [("x**2 + 1", ">", "0")]),
        ("quadratic_le", [("x**2 - 3*x + 2", "<=", "0")]),
        ("two_sided", [("x", ">=", "-1"), ("x", "<", "2")]),
        ("ne_and_lt", [("x", "!=", "1"), ("x**2", "<", "4")]),
        ("quartic", [("x**4 - 5*x**2 + 4", ">", "0")]),
        ("rational_lt", [("x/(x - 2)", "<", "1")]),
        ("product_ge", [("x*(x + 1)*(x - 3)", ">=", "0")]),
        ("exp_gt", [("exp(x)", ">", "1")]),
        ("sqrt_lt", [("sqrt(x)", "<", "2")]),
    ]
    sym_ops = {">": sp.Gt, ">=": sp.Ge, "<": sp.Lt, "<=": sp.Le, "=": sp.Eq, "!=": sp.Ne}
    for label, conds in ineq:

        def compute(conds=conds):
            rels = [sym_ops[o](sp.sympify(lhs), sp.sympify(rhs)) for lhs, o, rhs in conds]
            sol = sp.reduce_inequalities(rels, x)
            membership = []
            for p in SAMPLE_PTS:
                pe = sp.sympify(p)
                ok = True
                for lhs, o, rhs in conds:
                    lv = sp.sympify(lhs).subs(x, pe)
                    rv = sp.sympify(rhs).subs(x, pe)
                    if lv.has(zoo) or lv.has(S.NaN) or lv == zoo or not lv.is_real:
                        ok = False
                        break
                    val = sym_ops[o](lv, rv)
                    if val not in (S.true, S.false):
                        val = sp.simplify(val)
                    if val is S.true:
                        continue
                    if val is S.false:
                        ok = False
                        break
                    raise ValueError(f"undecided {val}")
                membership.append(ok)
            return {"sympy_result": str(sol), "sample_points": SAMPLE_PTS, "membership": membership}

        C.add(
            "inequalities",
            "reduce",
            label,
            {"conditions": [{"lhs": lhs, "op": o, "rhs": rhs} for lhs, o, rhs in conds], "variable": "x"},
            compute,
        )

    # Boolean formulas over 4 variables: fixed hand-written + seeded random.
    VARS = ["p", "q", "r", "s"]
    psym = {v: sp.Symbol(v) for v in VARS}

    def rand_formula(depth):
        if depth == 0 or RNG.random() < 0.25:
            leaf = {"var": RNG.choice(VARS)}
            return {"op": "not", "args": [leaf]} if RNG.random() < 0.3 else leaf
        name = RNG.choice(["and", "or", "and", "or", "xor", "implies", "equivalent", "not"])
        if name == "not":
            return {"op": "not", "args": [rand_formula(depth - 1)]}
        arity = 2 if name in ("xor", "implies", "equivalent") else RNG.choice([2, 2, 3])
        return {"op": name, "args": [rand_formula(depth - 1) for _ in range(arity)]}

    def build_bool(tree):
        if "var" in tree:
            return psym[tree["var"]]
        args = [build_bool(a) for a in tree["args"]]
        name = tree["op"]
        if name == "not":
            return Not(args[0])
        if name == "and":
            return And(*args)
        if name == "or":
            return Or(*args)
        if name == "xor":
            return Xor(*args)
        if name == "implies":
            return sp.Implies(args[0], args[1])
        if name == "equivalent":
            return sp.Equivalent(args[0], args[1])
        raise ValueError(name)

    hand = [
        ("xor_pq", op("xor", {"var": "p"}, {"var": "q"})),
        ("majority3", op("or", op("and", {"var": "p"}, {"var": "q"}), op("and", {"var": "p"}, {"var": "r"}), op("and", {"var": "q"}, {"var": "r"}))),
        ("tautology", op("or", {"var": "p"}, op("not", {"var": "p"}))),
        ("contradiction", op("and", {"var": "p"}, op("not", {"var": "p"}))),
        ("absorption", op("or", {"var": "p"}, op("and", {"var": "p"}, {"var": "q"}))),
        ("demorgan", op("not", op("and", {"var": "p"}, {"var": "q"}, {"var": "r"}))),
        ("implies_chain", op("implies", op("implies", {"var": "p"}, {"var": "q"}), op("implies", op("not", {"var": "q"}), op("not", {"var": "p"})))),
        ("equiv_xor", op("equivalent", {"var": "p"}, op("xor", {"var": "q"}, {"var": "r"}))),
        ("full_adder_sum", op("xor", op("xor", {"var": "p"}, {"var": "q"}), {"var": "r"})),
        ("full_adder_carry", op("or", op("and", {"var": "p"}, {"var": "q"}), op("and", {"var": "r"}, op("xor", {"var": "p"}, {"var": "q"})))),
    ]
    formulas = list(hand)
    for i in range(14):
        formulas.append((f"random_{i:02d}", rand_formula(3)))
    for label, tree in formulas:

        def compute(tree=tree):
            f = build_bool(tree)
            table = []
            for bits in range(16):
                assign = {psym[v]: bool((bits >> (3 - i)) & 1) for i, v in enumerate(VARS)}
                val = f.subs(assign)
                table.append(bool(val))
            simp = sp.simplify_logic(f)
            cnf = sp.to_cnf(f, simplify=True, force=True)
            sat = sp.satisfiable(f) is not False
            assert sat == any(table)
            return {
                "sympy_simplified": str(simp),
                "sympy_cnf": str(cnf),
                "truth_table": table,
                "satisfiable": sat,
                "tautology": all(table),
            }

        C.add("logic", "boolean", label, {"formula": tree, "variables": VARS}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# matrices
# ═══════════════════════════════════════════════════════════════════════════


def mat_json(rows):
    return [[str(sp.sympify(e)) for e in row] for row in rows]


def mat_label(rows):
    return "[" + ";".join(",".join(str(e) for e in row) for row in rows) + "]"


def gen_matrix():
    eig = [
        [[2, 1], [1, 2]],
        [[0, -1], [1, 0]],
        [[4, 1, 0], [1, 4, 1], [0, 1, 4]],
        [[1, 2, 0], [3, 4, 1], [0, 1, 5]],
        [[2, 0, 0], [0, 2, 0], [0, 0, 3]],
        [[1, 1], [0, 1]],
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
        [[Rational(1, 2), Rational(1, 3)], [Rational(1, 4), Rational(1, 5)]],
        [[6, -2, 1], [-2, 5, 0], [1, 0, 4]],
        [[1, 2, 3], [0, 4, 5], [0, 0, 6]],
    ]
    for rows in eig:
        M = Matrix(rows)

        def compute(M=M):
            ev = M.eigenvals()
            vals = []
            for val, mult in ev.items():
                for _ in range(mult):
                    vals.append(numval(val))
            vals.sort(key=lambda v: (round(v["re"], 9), round(v["im"], 9)))
            return {"eigenvalues": vals, "sympy_result": str(ev)}

        C.add("matrix", "eigenvals", mat_label(rows), {"matrix": mat_json(rows)}, compute)

    qr = [
        [[4, 2], [2, 3]],
        [[1, 2], [3, 4]],
        [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
        [[1, 1, 0], [1, 0, 1], [0, 1, 1]],
        [[1, 2], [3, 4], [5, 6]],
        [[3, 1, 2], [0, 4, 1], [2, 1, 5]],
    ]
    for rows in qr:
        M = Matrix(rows)

        def compute(M=M):
            Q, R = M.QRdecomposition()
            return {"r_diag_abs": [float(abs(sp.N(R[i, i], 20))) for i in range(min(R.shape))], "sympy_result": str(R)}

        C.add("matrix", "qr", mat_label(rows), {"matrix": mat_json(rows)}, compute)

    chol = [
        [[4, 2], [2, 3]],
        [[25, 15, -5], [15, 18, 0], [-5, 0, 11]],
        [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
        [[9, 3, 3], [3, 5, 1], [3, 1, 6]],
        [[1, 0, 0, 0], [0, 4, 0, 0], [0, 0, 9, 0], [0, 0, 0, 16]],
        [[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]],
    ]
    for rows in chol:
        M = Matrix(rows)

        def compute(M=M):
            L = M.cholesky()
            return {"result": [[numval(L[i, j]) for j in range(L.cols)] for i in range(L.rows)], "sympy_result": str(L)}

        C.add("matrix", "cholesky", mat_label(rows), {"matrix": mat_json(rows)}, compute)

    jordan = [
        [[2, 1, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]],
        [[5, 4, 2, 1], [0, 1, -1, -1], [-1, -1, 3, 0], [1, 1, -1, 2]],
        [[1, 1], [0, 1]],
        [[2, 0], [0, 2]],
        [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        [[3, 1, 0], [0, 3, 1], [0, 0, 3]],
        [[1, 2, 0], [3, 4, 1], [0, 1, 5]],
        [[4, 1, 0, 0], [0, 4, 0, 0], [0, 0, 4, 1], [0, 0, 0, 4]],
    ]
    for rows in jordan:
        M = Matrix(rows)

        def compute(M=M):
            P, J = M.jordan_form()
            diag = [numval(J[i, i]) for i in range(J.rows)]
            diag.sort(key=lambda v: (round(v["re"], 9), round(v["im"], 9)))
            ones = sum(1 for i in range(J.rows - 1) if J[i, i + 1] != 0)
            return {"jordan_diag": diag, "superdiag_ones": ones, "sympy_result": str(J)}

        C.add("matrix", "jordan_form", mat_label(rows), {"matrix": mat_json(rows)}, compute)

    mexp = [
        [[1, 1], [0, 2]],
        [[0, -1], [1, 0]],
        [[0, 1], [0, 0]],
        [[1, 2], [3, 4]],
        [[2, 0, 0], [0, 3, 0], [0, 0, -1]],
        [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[-1, 1], [1, -1]],
    ]
    for rows in mexp:
        M = Matrix(rows)

        def compute(M=M):
            E = M.exp()
            return {"result": [[numval(E[i, j]) for j in range(E.cols)] for i in range(E.rows)], "sympy_result": str(E)}

        C.add("matrix", "exp", mat_label(rows), {"matrix": mat_json(rows)}, compute)

    pinv = [
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2, 3], [4, 5, 6]],
        [[1, 0], [0, 0]],
        [[1, 2], [2, 4]],
        [[1, 2], [3, 4]],
        [[1, 1, 1], [1, 1, 1]],
        [[2, 0, 0], [0, 0, 0], [0, 0, 3]],
    ]
    for rows in pinv:
        M = Matrix(rows)

        def compute(M=M):
            Pm = M.pinv()
            return {"result": [[numval(Pm[i, j]) for j in range(Pm.cols)] for i in range(Pm.rows)], "sympy_result": str(Pm)}

        C.add("matrix", "pinv", mat_label(rows), {"matrix": mat_json(rows)}, compute)

    rank_null = [
        [[1, 2, 3], [2, 4, 6], [1, 1, 1]],
        [[1, 2], [3, 4]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 0], [0, 0]],
        [[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]],
        [[1, 0, 2, -1], [0, 1, 1, 1], [1, 1, 3, 0]],
        [[2, 4, 1], [1, 2, 3], [3, 6, 4]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ]
    for rows in rank_null:
        M = Matrix(rows)

        def compute(M=M):
            return {"rank": int(M.rank()), "nullity": len(M.nullspace())}

        C.add("matrix", "rank_nullspace", mat_label(rows), {"matrix": mat_json(rows)}, compute)

    charpoly = [
        [[1, 2], [3, 4]],
        [[2, 1], [1, 2]],
        [[1, 2, 0], [3, 4, 1], [0, 1, 5]],
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0]],
        [[6, -2, 1], [-2, 5, 0], [1, 0, 4]],
        [[Rational(1, 2), 1], [1, Rational(1, 3)]],
    ]
    for rows in charpoly:
        M = Matrix(rows)

        def compute(M=M):
            lam = sp.Symbol("lambda")
            cp = M.charpoly(lam)
            coeffs = cp.all_coeffs()  # det(λI − A), highest degree first
            return {"coeffs": [numval(c) for c in coeffs], "sympy_result": str(cp.as_expr())}

        C.add("matrix", "charpoly", mat_label(rows), {"matrix": mat_json(rows)}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# number theory, combinatorics, diophantine
# ═══════════════════════════════════════════════════════════════════════════

CARMICHAEL = [561, 1105, 1729, 2465, 2821, 6601, 8911, 41041, 62745, 63973, 101101, 252601, 410041, 1024651, 4335241, 9890881]
SPSP2 = [2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141, 52633, 65281, 74665, 80581, 85489, 88357, 90751, 3215031751]


def gen_ntheory():
    from sympy import ntheory as nt

    # factorint: small, powers, 40-bit semiprimes, 62-bit semiprime, primes
    prime_pool_20 = [nt.prime(RNG.randint(60000, 82000)) for _ in range(6)]  # ~20-bit primes
    semiprimes = [prime_pool_20[i] * prime_pool_20[i + 1] for i in range(0, 6, 2)]
    factorint_inputs = [
        360,
        1001,
        2**20 - 1,
        3**15 * 5**4,
        999999999989,  # prime
        600851475143,
        1099511627791,  # 2^40 + 15, prime
        *semiprimes,
        1000000016000000063,  # (10^9+7)(10^9+9)
        2**61 - 1,
        18446744073709551617,  # 2^64 + 1 = 274177 * 67280421310721
        12345678987654321,
    ]
    for nn in factorint_inputs:

        def compute(nn=nn):
            f = nt.factorint(nn)
            return {"factors": [[str(p), int(e)] for p, e in sorted(f.items())]}

        C.add("ntheory", "factorint", str(nn), {"input": str(nn)}, compute)

    isprime_inputs = [2, 3, 4, 97, 7919, 1000003, 2**31 - 1, 2**61 - 1, 2**89 - 1, 2**127 - 1, 2**67 - 1, 10**18 + 9, 10**18 + 3, 999999999989, 999999999999, 4294967297, 18446744073709551557, 170141183460469231731687303715884105727 * 3 + 2]
    isprime_inputs += CARMICHAEL + SPSP2
    for nn in isprime_inputs:

        def compute(nn=nn):
            return {"value_bool": bool(nt.isprime(nn))}

        C.add("ntheory", "isprime", str(nn), {"input": str(nn)}, compute)

    sqrt_mod_cases = [(2, 7), (4, 7), (3, 7), (10, 13), (5, 11), (2, 17), (9, 101), (1234, 100003), (7, 8), (4, 9), (2, 3), (0, 5), (1, 2), (3, 25), (58, 101), (7, 29 * 31)]
    for aa, mm in sqrt_mod_cases:

        def compute(aa=aa, mm=mm):
            roots = nt.sqrt_mod(aa, mm, all_roots=True)
            return {"roots": [str(r) for r in roots] if roots else []}

        C.add("ntheory", "sqrt_mod", f"{aa} mod {mm}", {"a": str(aa), "modulus": str(mm)}, compute)

    dlog_cases = [(3, 13, 17), (2, 5, 11), (5, 8, 23), (7, 1, 13), (2, 3, 7), (3, 4, 7), (2, 64, 1000003), (5, 12345 % 1000003, 1000003), (3, 3, 101), (10, 7, 19), (2, 2**20 % 65537, 65537)]
    for base, target, mod in dlog_cases:

        def compute(base=base, target=target, mod=mod):
            try:
                v = nt.discrete_log(mod, target, base)
                return {"value": str(v), "solvable": True}
            except ValueError:
                return {"value": None, "solvable": False}

        C.add("ntheory", "discrete_log", f"{base}^x = {target} mod {mod}", {"base": str(base), "target": str(target), "modulus": str(mod)}, compute)

    prim_root = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 23, 25, 27, 31, 41, 49, 50, 97, 101, 1009, 65537, 1000003]
    for nn in prim_root:

        def compute(nn=nn):
            r = nt.primitive_root(nn)
            out = {"value": str(r) if r is not None else None}
            if r is not None and nn < 200:
                allr = [g for g in range(1, nn) if math.gcd(g, nn) == 1 and nt.is_primitive_root(g, nn)]
                out["all_roots"] = [str(g) for g in allr]
            return out

        C.add("ntheory", "primitive_root", str(nn), {"input": str(nn)}, compute)

    jacobi = [(1001, 9907), (19, 45), (8, 21), (5, 21), (2, 15), (30, 59), (7, 143), (1, 1), (0, 3), (-1, 7), (-1, 5), (12345, 331), (2, 2**61 - 1), (3, 10**9 + 7), (11, 35)]
    for aa, nn in jacobi:

        def compute(aa=aa, nn=nn):
            return {"value_int": int(nt.jacobi_symbol(aa, nn))}

        C.add("ntheory", "jacobi_symbol", f"({aa}/{nn})", {"a": str(aa), "n": str(nn)}, compute)

    cf = [(415, 93), (355, 113), (1, 3), (7, 1), (-17, 5), (103993, 33102), (1234567, 89), (0, 1), (13, 8), (6, 4)]
    for p_, q_ in cf:

        def compute(p_=p_, q_=q_):
            return {"value_list": [int(v) for v in sp.continued_fraction(Rational(p_, q_))]}

        C.add("ntheory", "continued_fraction", f"{p_}/{q_}", {"numerator": str(p_), "denominator": str(q_)}, compute)

    unary = {
        "primepi": ([1, 2, 10, 100, 1000, 10**4, 10**5, 10**6, 2**20, 5 * 10**6], nt.primepi),
        "totient": ([1, 2, 12, 36, 97, 1000, 65536, 999999, 2**32 - 1, 10**9 + 7], nt.totient),
        "mobius": ([1, 2, 4, 6, 30, 210, 2310, 12, 49, 1001, 999999, 1024], nt.mobius),
        "partition": ([0, 1, 5, 10, 50, 100, 200, 500, 1000], nt.npartitions),
        "bell": ([0, 1, 2, 5, 10, 15, 20, 30], sp.bell),
        "catalan": ([0, 1, 2, 5, 10, 20, 30, 50], sp.catalan),
        "fibonacci": ([0, 1, 2, 10, 50, 100, 200, 300, -7, -8], sp.fibonacci),
        "lucas": ([0, 1, 2, 10, 50, 100, 200, -5, -6], sp.lucas),
    }
    for name, (inputs, fn) in unary.items():
        for nn in inputs:

            def compute(nn=nn, fn=fn):
                return {"value": str(int(fn(nn)))}

            C.add("ntheory", name, str(nn), {"input": str(nn)}, compute)

    for nn in [0, 1, 2, 4, 6, 8, 10, 12, 20, 30, 50, 60]:

        def compute(nn=nn):
            v = sp.bernoulli(nn)
            out = {"value": str(v)}
            if nn == 1:
                # SymPy >= 1.12 uses B_1 = +1/2; the classical convention is -1/2.
                out["convention_note"] = "B1 sign is convention-dependent; either sign accepted"
            return out

        C.add("ntheory", "bernoulli", str(nn), {"input": str(nn)}, compute)

    for nn, kk in [(12, 0), (12, 1), (12, 2), (100, 1), (100, 3), (1, 1), (7, 2), (360, 0), (360, 1), (999983, 1)]:

        def compute(nn=nn, kk=kk):
            return {"value": str(int(nt.divisor_sigma(nn, kk)))}

        C.add("ntheory", "divisor_sigma", f"sigma_{kk}({nn})", {"input": str(nn), "k": kk}, compute)

    for nn, kk in [(10, 3), (5, 2), (8, 1), (8, 8), (12, 6), (15, 4), (20, 10), (6, 0), (0, 0), (7, 9)]:
        for kind in (1, 2):

            def compute(nn=nn, kk=kk, kind=kind):
                v = sp.functions.combinatorial.numbers.stirling(nn, kk, kind=kind, signed=(kind == 1))
                return {"value": str(int(v))}

            C.add("ntheory", f"stirling{kind}", f"S{kind}({nn},{kk})", {"n": str(nn), "k": str(kk)}, compute)

    dioph_lin = [(3, 5, 1), (4, 6, 7), (4, 6, 10), (12, 18, 30), (7, 11, 1), (25, 35, 5), (-3, 7, 2), (6, 9, 21), (1, 1, 0), (10, 4, 6)]
    for aa, bb, cc in dioph_lin:

        def compute(aa=aa, bb=bb, cc=cc):
            from sympy.solvers.diophantine import diophantine

            sol = diophantine(aa * x + bb * y - cc)
            g = math.gcd(aa, bb)
            return {"solvable": bool(sol), "sympy_result": str(sol), "gcd": g}

        C.add("diophantine", "linear", f"{aa}x + {bb}y = {cc}", {"a": str(aa), "b": str(bb), "c": str(cc)}, compute)

    pells = [2, 3, 5, 6, 7, 10, 13, 61, 109, 181, 277, 421, 991]
    for d in pells:

        def compute(d=d):
            from sympy.solvers.diophantine.diophantine import diop_DN

            sol = diop_DN(d, 1)
            xx, yy = min(sol, key=lambda p: p[0])
            return {"x": str(xx), "y": str(yy)}

        C.add("diophantine", "pell", str(d), {"d": str(d)}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# codegen / compile
# ═══════════════════════════════════════════════════════════════════════════


def gen_codegen():
    exprs = [
        "gamma(x) + x**2",
        "erf(x)*exp(-x)",
        "sin(x)**2 + atan(x)",
        "digamma(x) + loggamma(x)",
        "LambertW(x)*x",
        "Si(x) + Ci(x)",
        "Ei(x) - li(x)",
        "zeta(x)",
        "polygamma(1, x)",
        "abs(x - 1)*sign(x) + floor(x)",
        "sqrt(x**2 + 1)/(1 + exp(-x))",
        "besselj(0, x)",
        "erfc(x) + gamma(x/2)",
        "log(x)*x**(1/3)",
        "(x**3 - 2*x + 1)/(x**2 + 1)",
        "sinh(x)*tanh(x) + cosh(x)",
    ]
    for e_str in exprs:
        e = sp.sympify(e_str)
        pts = []
        for _ in range(4):
            v = RNG.uniform(0.3, 4.0)
            pts.append({x: sp.Rational(round(v * 1000), 1000)})

        def compute(e=e, pts=pts):
            return {"eval_points": eval_points(e, pts)}

        C.add("codegen", "compile", e_str, {"input": e_str, "variables": ["x"]}, compute)


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════


def build():
    for gen in (
        gen_definite_integrals,
        gen_sums_products,
        gen_limits_series_residues,
        gen_transforms,
        gen_solve,
        gen_poly,
        gen_complex_special,
        gen_simplify,
        gen_sets_logic,
        gen_matrix,
        gen_ntheory,
        gen_codegen,
    ):
        print(f"== {gen.__name__}", file=sys.stderr)
        gen()

    fixtures = sorted(C.fixtures, key=lambda f: (f["category"], f["subcategory"]))  # stable
    for i, f in enumerate(fixtures, 1):
        f["id"] = i
    counts = Counter(f"{f['category']}:{f['subcategory']}" for f in fixtures)
    return {
        "generated_by": f"SymPy {sp.__version__}",
        "generator": "scripts/generate_v02_fixtures.py",
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
        "v02_cross_validation.json",
    )
    output = build()
    text = json.dumps(output, indent=1, sort_keys=True) + "\n"
    print(
        f"\nGenerated {output['fixture_count']} fixtures "
        f"({C.timeouts} sympy timeouts, {C.errors} sympy errors, {C.unevaluated} sympy unevaluated)",
        file=sys.stderr,
    )
    for cat, cnt in output["categories"].items():
        print(f"  {cat:40s} {cnt:4d}", file=sys.stderr)
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
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
