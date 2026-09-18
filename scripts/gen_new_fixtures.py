#!/usr/bin/env python3
"""Generate SymPy cross-validation fixtures for the 0.1→0.2 "new features"
(eigenvectors, Jordan form, LambertW, Bessel functions, refine, inverse
hyperbolic integrals, erf special values, matrix exponential).

Output: ``tests/fixtures/new_features_cross_validation.json``, consumed by
``tests/v02_oracle_new_features.rs``.

Deterministic (no timestamps, sorted keys) and bounded: every fixture's
SymPy computation runs under ``PER_FIXTURE_TIMEOUT``; a timeout / error is
recorded as ``sympy_timeout`` / ``sympy_error`` instead of dropping the case.

Usage:
    /Users/chris.gorski/repos/math/symplex/.venv/bin/python scripts/gen_new_fixtures.py
    ... --check     # exit 1 if the committed file would change
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
from collections import Counter

import sympy
from sympy import (
    Abs,
    E,
    Integer,
    LambertW,
    Matrix,
    Rational,
    Symbol,
    acosh,
    asinh,
    atanh,
    besselj,
    bessely,
    ceiling,
    erf,
    erfc,
    eye,
    floor,
    integrate,
    log,
    oo,
    refine,
    sign,
    sqrt,
    symbols,
    zeros,
)

x, y = symbols("x y")
PER_FIXTURE_TIMEOUT = 20.0


class _FixtureTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _FixtureTimeout()


fixtures = []


def add(category, subcategory, fields, compute, key=None):
    """Register one fixture.  ``key`` must be unique within the category
    (defaults to the subcategory, or to ``fields["input"]`` when present)."""
    if key is None:
        key = fields["input"] if isinstance(fields.get("input"), str) else subcategory
    fx = {"category": category, "subcategory": subcategory, "key": key}
    fx.update(fields)
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, PER_FIXTURE_TIMEOUT)
    try:
        fx.update(compute())
    except _FixtureTimeout:
        fx["sympy_timeout"] = True
        print(f"  TIMEOUT {category}:{subcategory}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — recorded in the fixture
        fx["sympy_error"] = f"{type(e).__name__}: {e}"
        print(f"  ERROR   {category}:{subcategory}: {e}", file=sys.stderr)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    fixtures.append(fx)


def numval(expr):
    v = complex(sympy.N(expr, 30))
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    return {"re": v.real, "im": 0.0 if abs(v.imag) < 1e-18 else v.imag}


def mat_str(m):
    return [[str(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


def mat_num(m):
    return [[numval(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


# ── eigenvects ─────────────────────────────────────────────────────────
eig_cases = [
    ("2x2_distinct", Matrix([[2, 1], [0, 3]])),
    ("2x2_defective", Matrix([[1, 1], [0, 1]])),
    ("3x3_diagonal", Matrix([[5, 0, 0], [0, -3, 0], [0, 0, 7]])),
    ("2x2_symmetric", Matrix([[4, 2], [2, 1]])),
    ("3x3_upper_triangular", Matrix([[1, 2, 3], [0, 4, 5], [0, 0, 6]])),
    ("3x3_repeated_full_geometric", Matrix([[2, 0, 0], [0, 2, 0], [0, 0, 5]])),
    ("2x2_rotation_complex", Matrix([[0, -1], [1, 0]])),
]
for label, A in eig_cases:

    def compute(A=A):
        out = []
        for val, mult, vecs in A.eigenvects():
            out.append({"eigenvalue": numval(val), "algebraic_multiplicity": int(mult), "geometric_multiplicity": len(vecs)})
        out.sort(key=lambda d: (round(d["eigenvalue"]["re"], 9), round(d["eigenvalue"]["im"], 9)))
        return {"eigenspaces": out}

    add("eigenvects", label, {"input_matrix": mat_str(A)}, compute, key=label)

# ── jordan_form ────────────────────────────────────────────────────────
jordan_cases = [
    ("2x2_defective", Matrix([[1, 1], [0, 1]])),
    ("3x3_diagonal", Matrix([[5, 0, 0], [0, -3, 0], [0, 0, 7]])),
    ("4x4_mixed", Matrix([[2, 1, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])),
    ("2x2_identity", eye(2)),
    ("2x2_nilpotent", Matrix([[0, 1], [0, 0]])),
]
for label, A in jordan_cases:

    def compute(A=A):
        _, J = A.jordan_form()
        diag = sorted((numval(J[i, i]) for i in range(J.rows)), key=lambda d: (d["re"], d["im"]))
        ones = sum(1 for i in range(J.rows - 1) if J[i, i + 1] != 0)
        return {"jordan_matrix": mat_str(J), "jordan_diag": diag, "superdiag_ones": ones}

    add("jordan_form", label, {"input_matrix": mat_str(A)}, compute, key=label)

# ── LambertW ───────────────────────────────────────────────────────────
for inp in ["0", "E", "-1/E", "-log(2)/2", "2*log(2)", "3*log(3)"]:

    def compute(inp=inp):
        r = LambertW(sympy.sympify(inp))
        return {"expected": str(r), "value": numval(r)}

    add("lambertw", "eval_symbolic", {"input": inp}, compute)
for val in [Rational(1, 2), Integer(5), Rational(1, 10), Integer(10), Integer(100), Rational(-1, 4)]:

    def compute(val=val):
        return {"value": numval(LambertW(val))}

    add("lambertw", "eval_numerical", {"input": str(val)}, compute)

# ── Bessel ─────────────────────────────────────────────────────────────
bessel_cases = [
    ("J", 0, "0"), ("J", 0, "1"), ("J", 0, "5"), ("J", 0, "10"), ("J", 1, "0"), ("J", 1, "1"), ("J", 1, "5"),
    ("J", 2, "3"), ("J", 3, "7/2"), ("J", 5, "10"), ("J", 0, "12"), ("J", 0, "15"), ("J", 3, "15"),
    ("J", 0, "20"), ("J", 1, "20"), ("J", 2, "20"), ("J", 5, "20"), ("J", 0, "30"), ("J", 1, "30"), ("J", 0, "50"),
    ("Y", 0, "1"), ("Y", 0, "5"), ("Y", 1, "1"), ("Y", 1, "2"), ("Y", 2, "3"), ("Y", 3, "15/2"), ("Y", 5, "10"),
    ("Y", 0, "12"), ("Y", 0, "15"), ("Y", 0, "20"), ("Y", 5, "20"), ("Y", 0, "30"), ("Y", 1, "50"),
]
for kind, order, xval in bessel_cases:

    def compute(kind=kind, order=order, xval=xval):
        xe = sympy.sympify(xval)
        r = besselj(order, xe) if kind == "J" else bessely(order, xe)
        return {"value": numval(r)}

    add("bessel", f"{kind}{order}", {"kind": kind, "order": order, "x": xval}, compute, key=f"{kind}{order}({xval})")

# ── refine ─────────────────────────────────────────────────────────────
refine_cases = [
    ("abs_positive", "Abs(x)", "positive"),
    ("abs_negative", "Abs(x)", "negative"),
    ("abs_nonnegative", "Abs(x)", "nonnegative"),
    ("sign_positive", "sign(x)", "positive"),
    ("sign_negative", "sign(x)", "negative"),
    ("sign_zero", "sign(x)", "zero"),
    ("sqrt_x2_positive", "sqrt(x**2)", "positive"),
    ("sqrt_x2_real", "sqrt(x**2)", "real"),
    ("floor_integer", "floor(x)", "integer"),
    ("ceiling_integer", "ceiling(x)", "integer"),
    ("abs_x2_real", "Abs(x**2)", "real"),
    ("sqrt_x2_negative", "sqrt(x**2)", "negative"),
]
SAMPLE = {"positive": ["1/2", "3", "7"], "negative": ["-1/2", "-3", "-7"], "nonnegative": ["0", "2", "5"],
          "zero": ["0"], "real": ["-3", "0", "5/2"], "integer": ["-4", "0", "9"]}
for label, input_str, assumption in refine_cases:

    def compute(input_str=input_str, assumption=assumption):
        xsym = Symbol("x", **{assumption: True})
        expr = eval(input_str, {"x": xsym, "Abs": Abs, "sign": sign, "sqrt": sqrt, "floor": floor, "ceiling": ceiling})
        result = refine(expr)
        pts = []
        for p in SAMPLE[assumption]:
            pe = sympy.sympify(p)
            pts.append({"subs": {"x": float(pe)}, "value": numval(result.subs(xsym, pe))})
        return {"sympy_result": str(result), "eval_points": pts, "sympy_ops": int(sympy.count_ops(result))}

    add("refine", label, {"input": input_str, "assumptions": assumption}, compute, key=label)

# ── inverse hyperbolic antiderivatives ─────────────────────────────────
for label, f, pts in [
    ("asinh", asinh(x), [Rational(1, 2), Rational(3, 2), 3]),
    ("acosh", acosh(x), [Rational(3, 2), 2, 3]),
    ("atanh", atanh(x), [Rational(1, 4), Rational(1, 2), Rational(3, 4)]),
    ("x_asinh", x * asinh(x), [Rational(1, 2), Rational(3, 2), 3]),
]:

    def compute(f=f, pts=pts):
        F = integrate(f, x)
        out = {"sympy_result": str(F)}
        if F.has(sympy.Integral):
            out["sympy_unevaluated"] = True
            return out
        out["eval_points"] = [{"subs": {"x": float(p)}, "value": numval(F.subs(x, p))} for p in pts]
        return out

    add("integrate", label, {"input": str(f), "variable": "x"}, compute, key=label)

# ── erf / erfc special values ──────────────────────────────────────────
for label, expr in [
    ("erf_zero", erf(0)), ("erf_inf", erf(oo)), ("erf_neg_inf", erf(-oo)), ("erfc_zero", erfc(0)), ("erfc_inf", erfc(oo)),
    ("erfc_neg_inf", erfc(-oo)), ("erf_one", erf(1)), ("erfc_two", erfc(2)), ("erf_minus_half", erf(Rational(-1, 2))),
]:

    def compute(expr=expr):
        return {"expected": str(expr), "value": numval(expr)}

    inp = str(expr) if not expr.is_number or expr.free_symbols else label
    add("eval", label, {"input": {"erf_zero": "erf(0)", "erf_inf": "erf(oo)", "erf_neg_inf": "erf(-oo)", "erfc_zero": "erfc(0)",
                                  "erfc_inf": "erfc(oo)", "erfc_neg_inf": "erfc(-oo)", "erf_one": "erf(1)", "erfc_two": "erfc(2)",
                                  "erf_minus_half": "erf(-1/2)"}[label]}, compute)

# ── matrix exponential ─────────────────────────────────────────────────
for label, M in [
    ("zero_matrix", zeros(2)),
    ("rotation_generator", Matrix([[0, 1], [-1, 0]])),
    ("nilpotent", Matrix([[0, 1], [0, 0]])),
    ("diagonal", Matrix([[2, 0], [0, 3]])),
    ("identity", eye(2)),
    ("shear_scaled", Matrix([[1, 2], [0, 1]])),
]:

    def compute(M=M):
        Em = M.exp()
        return {"result": mat_str(Em), "result_numeric": mat_num(Em)}

    add("matrix_exp", label, {"input_matrix": mat_str(M)}, compute, key=label)

# ── output ─────────────────────────────────────────────────────────────
fixtures.sort(key=lambda f: (f["category"], f["subcategory"], f["key"]))
seen = set()
for f in fixtures:
    ident = (f["category"], f["key"])
    assert ident not in seen, f"duplicate key {ident}"
    seen.add(ident)
for i, f in enumerate(fixtures, 1):
    f["id"] = i
output = {
    "generated_by": f"SymPy {sympy.__version__}",
    "generator": "scripts/gen_new_fixtures.py",
    "description": "Cross-validation fixtures for new symplex features",
    "fixture_count": len(fixtures),
    "categories": dict(sorted(Counter(f["category"] for f in fixtures).items())),
    "fixtures": fixtures,
}
text = json.dumps(output, indent=2, sort_keys=True) + "\n"
out_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures", "new_features_cross_validation.json"
)
if "--check" in sys.argv:
    with open(out_path) as fh:
        if fh.read() != text:
            print("MISMATCH: fixture file differs from generator output", file=sys.stderr)
            sys.exit(1)
    print("OK: fixture file is up to date", file=sys.stderr)
else:
    with open(out_path, "w") as fh:
        fh.write(text)
    print(f"Wrote {out_path}", file=sys.stderr)
print(f"Generated {len(fixtures)} fixtures:", file=sys.stderr)
for cat, count in output["categories"].items():
    print(f"  {cat}: {count}", file=sys.stderr)
