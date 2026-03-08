#!/usr/bin/env python3
"""Generate SymPy cross-validation fixtures for new symplex features.

Run from the symplex project root with the .venv that has sympy installed:
    .venv/bin/python3 scripts/gen_new_fixtures.py

Outputs JSON to stdout. Redirect to merge into the fixture file.
"""

import json
import sys
import sympy
from sympy import *

fixtures = []
x, y = symbols('x y')


def eval_at(expr, subs_dict):
    """Evaluate an expression at given substitution points, returning complex value."""
    try:
        result = expr.evalf(subs=subs_dict)
        # Try to convert to complex; if it fails (symbolic residue), skip
        re_part = float(re(result).evalf())
        im_part = float(im(result).evalf())
        return {
            'subs': {str(k): float(v) for k, v in subs_dict.items()},
            'value': {'re': re_part, 'im': im_part}
        }
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Eigenvector fixtures
# ═══════════════════════════════════════════════════════════════════════════

# 2x2 distinct eigenvalues
A = Matrix([[2, 1], [0, 3]])
for val, mult, vecs in A.eigenvects():
    fixtures.append({
        'category': 'eigenvects',
        'subcategory': '2x2_distinct',
        'input_matrix': [[str(A[i, j]) for j in range(2)] for i in range(2)],
        'eigenvalue': str(val),
        'multiplicity': mult,
        'eigenvector': [str(v) for v in vecs[0]],
    })

# 2x2 defective (repeated eigenvalue, geometric mult < algebraic mult)
B = Matrix([[1, 1], [0, 1]])
for val, mult, vecs in B.eigenvects():
    fixtures.append({
        'category': 'eigenvects',
        'subcategory': '2x2_defective',
        'input_matrix': [[str(B[i, j]) for j in range(2)] for i in range(2)],
        'eigenvalue': str(val),
        'multiplicity': mult,
        'eigenvector_count': len(vecs),
    })

# 3x3 diagonal
C = Matrix([[5, 0, 0], [0, -3, 0], [0, 0, 7]])
for val, mult, vecs in C.eigenvects():
    fixtures.append({
        'category': 'eigenvects',
        'subcategory': '3x3_diagonal',
        'input_matrix': [[str(C[i, j]) for j in range(3)] for i in range(3)],
        'eigenvalue': str(val),
        'multiplicity': mult,
    })

# 2x2 symmetric (real eigenvalues guaranteed)
S = Matrix([[4, 2], [2, 1]])
for val, mult, vecs in S.eigenvects():
    fixtures.append({
        'category': 'eigenvects',
        'subcategory': '2x2_symmetric',
        'input_matrix': [[str(S[i, j]) for j in range(2)] for i in range(2)],
        'eigenvalue': str(val),
        'multiplicity': mult,
        'eigenvector': [str(v) for v in vecs[0]],
    })

# 3x3 upper triangular
T = Matrix([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
for val, mult, vecs in T.eigenvects():
    fixtures.append({
        'category': 'eigenvects',
        'subcategory': '3x3_upper_triangular',
        'input_matrix': [[str(T[i, j]) for j in range(3)] for i in range(3)],
        'eigenvalue': str(val),
        'multiplicity': mult,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Jordan form fixtures
# ═══════════════════════════════════════════════════════════════════════════

# 2x2 defective
P2, J2 = B.jordan_form()
fixtures.append({
    'category': 'jordan_form',
    'subcategory': '2x2_defective',
    'input_matrix': [[str(B[i, j]) for j in range(2)] for i in range(2)],
    'jordan_matrix': [[str(J2[i, j]) for j in range(2)] for i in range(2)],
})

# 3x3 diagonal (trivial Jordan = diagonal)
_, J3 = C.jordan_form()
fixtures.append({
    'category': 'jordan_form',
    'subcategory': '3x3_diagonal',
    'input_matrix': [[str(C[i, j]) for j in range(3)] for i in range(3)],
    'jordan_matrix': [[str(J3[i, j]) for j in range(3)] for i in range(3)],
})

# 4x4 mixed blocks: eigenvalue 2 with 2x2 block, eigenvalues 3 and 4 with 1x1
D = Matrix([[2, 1, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])
_, JD = D.jordan_form()
fixtures.append({
    'category': 'jordan_form',
    'subcategory': '4x4_mixed',
    'input_matrix': [[str(D[i, j]) for j in range(4)] for i in range(4)],
    'jordan_matrix': [[str(JD[i, j]) for j in range(4)] for i in range(4)],
})

# Identity matrix (trivial)
I2 = eye(2)
_, JI = I2.jordan_form()
fixtures.append({
    'category': 'jordan_form',
    'subcategory': '2x2_identity',
    'input_matrix': [[str(I2[i, j]) for j in range(2)] for i in range(2)],
    'jordan_matrix': [[str(JI[i, j]) for j in range(2)] for i in range(2)],
})

# Nilpotent
N = Matrix([[0, 1], [0, 0]])
_, JN = N.jordan_form()
fixtures.append({
    'category': 'jordan_form',
    'subcategory': '2x2_nilpotent',
    'input_matrix': [[str(N[i, j]) for j in range(2)] for i in range(2)],
    'jordan_matrix': [[str(JN[i, j]) for j in range(2)] for i in range(2)],
})


# ═══════════════════════════════════════════════════════════════════════════
# LambertW fixtures
# ═══════════════════════════════════════════════════════════════════════════

fixtures.append({
    'category': 'lambertw', 'subcategory': 'eval_symbolic',
    'input': '0', 'expected': '0',
})
fixtures.append({
    'category': 'lambertw', 'subcategory': 'eval_symbolic',
    'input': 'E', 'expected': '1',
})
fixtures.append({
    'category': 'lambertw', 'subcategory': 'eval_symbolic',
    'input': '-1/E', 'expected': '-1',
})
fixtures.append({
    'category': 'lambertw', 'subcategory': 'eval_symbolic',
    'input': '-log(2)/2', 'expected': str(LambertW(-log(2) / 2)),
})

# Numerical evaluations
for val in [Rational(1, 2), Integer(5), Rational(1, 10), Integer(10)]:
    w = float(LambertW(val).evalf())
    fixtures.append({
        'category': 'lambertw', 'subcategory': 'eval_numerical',
        'input': str(val), 'expected_float': w,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Bessel fixtures
# ═══════════════════════════════════════════════════════════════════════════

bessel_cases = [
    ('J', 0, 0.0),
    ('J', 0, 1.0),
    ('J', 0, 5.0),
    ('J', 0, 10.0),
    ('J', 1, 0.0),
    ('J', 1, 1.0),
    ('J', 1, 5.0),
    ('J', 2, 3.0),
    ('Y', 0, 1.0),
    ('Y', 0, 5.0),
    ('Y', 1, 1.0),
    ('Y', 1, 2.0),
    ('Y', 2, 3.0),
]

for kind, order, xval in bessel_cases:
    if kind == 'J':
        val = float(besselj(order, xval).evalf()) if xval != 0 else (1.0 if order == 0 else 0.0)
    else:
        if xval == 0:
            continue  # Y is singular at 0
        val = float(bessely(order, xval).evalf())
    fixtures.append({
        'category': 'bessel',
        'subcategory': f'{kind}{order}',
        'x': xval,
        'expected': val,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Refine fixtures
# ═══════════════════════════════════════════════════════════════════════════

refine_cases = [
    ('abs_positive', 'Abs(x)', 'positive', 'x'),
    ('abs_negative', 'Abs(x)', 'negative', '-x'),
    ('abs_nonnegative', 'Abs(x)', 'nonnegative', 'x'),
    ('sign_positive', 'sign(x)', 'positive', '1'),
    ('sign_negative', 'sign(x)', 'negative', '-1'),
    ('sign_zero', 'sign(x)', 'zero', '0'),
    ('sqrt_x2_positive', 'sqrt(x**2)', 'positive', 'x'),
    ('sqrt_x2_real', 'sqrt(x**2)', 'real', 'Abs(x)'),
    ('floor_integer', 'floor(x)', 'integer', 'x'),
    ('ceiling_integer', 'ceiling(x)', 'integer', 'x'),
]

# Verify against SymPy's refine
for subcat, input_str, assumption, expected_str in refine_cases:
    xsym = Symbol('x', **{assumption: True})
    expr = eval(input_str, {'x': xsym, 'Abs': Abs, 'sign': sign,
                            'sqrt': sqrt, 'floor': floor, 'ceiling': ceiling})
    result = refine(expr)
    fixtures.append({
        'category': 'refine',
        'subcategory': subcat,
        'input': input_str.replace('x', 'x'),
        'assumptions': assumption,
        'expected': expected_str,
        'sympy_result': str(result),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Integration fixtures (inverse hyperbolic)
# ═══════════════════════════════════════════════════════════════════════════

anti_asinh = integrate(asinh(x), x)
anti_acosh = integrate(acosh(x), x)
anti_atanh = integrate(atanh(x), x)

asinh_pts = [eval_at(anti_asinh, {x: Rational(1, 2)}),
             eval_at(anti_asinh, {x: Rational(3, 2)}),
             eval_at(anti_asinh, {x: Integer(3)})]
fixtures.append({
    'category': 'integrate', 'subcategory': 'asinh',
    'input': 'asinh(x)', 'variable': 'x',
    'sympy_result': str(anti_asinh),
    'eval_points': [p for p in asinh_pts if p is not None],
})

acosh_pts = [eval_at(anti_acosh, {x: Rational(3, 2)}),
             eval_at(anti_acosh, {x: Integer(2)}),
             eval_at(anti_acosh, {x: Integer(5)})]
fixtures.append({
    'category': 'integrate', 'subcategory': 'acosh',
    'input': 'acosh(x)', 'variable': 'x',
    'sympy_result': str(anti_acosh),
    'eval_points': [p for p in acosh_pts if p is not None],
})

atanh_pts = [eval_at(anti_atanh, {x: Rational(1, 4)}),
             eval_at(anti_atanh, {x: Rational(1, 2)}),
             eval_at(anti_atanh, {x: Rational(3, 4)})]
fixtures.append({
    'category': 'integrate', 'subcategory': 'atanh',
    'input': 'atanh(x)', 'variable': 'x',
    'sympy_result': str(anti_atanh),
    'eval_points': [p for p in atanh_pts if p is not None],
})


# ═══════════════════════════════════════════════════════════════════════════
# erf/erfc fixtures
# ═══════════════════════════════════════════════════════════════════════════

fixtures.append({
    'category': 'eval', 'subcategory': 'erf_zero',
    'input': 'erf(0)', 'expected': '0',
})
fixtures.append({
    'category': 'eval', 'subcategory': 'erf_inf',
    'input': 'erf(oo)', 'expected': '1',
})
fixtures.append({
    'category': 'eval', 'subcategory': 'erf_neg_inf',
    'input': 'erf(-oo)', 'expected': '-1',
})
fixtures.append({
    'category': 'eval', 'subcategory': 'erfc_zero',
    'input': 'erfc(0)', 'expected': '1',
})
fixtures.append({
    'category': 'eval', 'subcategory': 'erfc_inf',
    'input': 'erfc(oo)', 'expected': '0',
})


# ═══════════════════════════════════════════════════════════════════════════
# Matrix exponential fixtures
# ═══════════════════════════════════════════════════════════════════════════

# Zero matrix → identity
M0 = zeros(2)
expM0 = M0.exp()
fixtures.append({
    'category': 'matrix_exp', 'subcategory': 'zero_matrix',
    'input_matrix': [['0', '0'], ['0', '0']],
    'result': [[str(expM0[i, j]) for j in range(2)] for i in range(2)],
})

# Rotation generator [[0,1],[-1,0]] → [[cos(1), sin(1)], [-sin(1), cos(1)]]
MR = Matrix([[0, 1], [-1, 0]])
expMR = MR.exp()
fixtures.append({
    'category': 'matrix_exp', 'subcategory': 'rotation_generator',
    'input_matrix': [['0', '1'], ['-1', '0']],
    'result': [[str(expMR[i, j]) for j in range(2)] for i in range(2)],
})

# Nilpotent [[0,1],[0,0]] → [[1,1],[0,1]]
MN = Matrix([[0, 1], [0, 0]])
expMN = MN.exp()
fixtures.append({
    'category': 'matrix_exp', 'subcategory': 'nilpotent',
    'input_matrix': [['0', '1'], ['0', '0']],
    'result': [[str(expMN[i, j]) for j in range(2)] for i in range(2)],
})

# Diagonal [[2,0],[0,3]]
MD = Matrix([[2, 0], [0, 3]])
expMD = MD.exp()
fixtures.append({
    'category': 'matrix_exp', 'subcategory': 'diagonal',
    'input_matrix': [['2', '0'], ['0', '3']],
    'result': [[str(expMD[i, j]) for j in range(2)] for i in range(2)],
})

# Identity
MI = eye(2)
expMI = MI.exp()
fixtures.append({
    'category': 'matrix_exp', 'subcategory': 'identity',
    'input_matrix': [['1', '0'], ['0', '1']],
    'result': [[str(expMI[i, j]) for j in range(2)] for i in range(2)],
})


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════

output = {
    'generated_by': f'SymPy {sympy.__version__}',
    'description': 'Cross-validation fixtures for new symplex features',
    'fixture_count': len(fixtures),
    'categories': {},
    'fixtures': fixtures,
}

# Count by category
for f in fixtures:
    cat = f['category']
    output['categories'][cat] = output['categories'].get(cat, 0) + 1

json.dump(output, sys.stdout, indent=2)
print()  # trailing newline

print(f"\nGenerated {len(fixtures)} fixtures:", file=sys.stderr)
for cat, count in sorted(output['categories'].items()):
    print(f"  {cat}: {count}", file=sys.stderr)
