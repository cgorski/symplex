# Third-party notices

symplex is an independent project.  It is **not affiliated with, sponsored
by or endorsed by** the SymPy project, the mpmath, SciPy or statsmodels
projects, or any of their contributors; their names appear in this
repository only to identify them.

## How other projects are used

**As test oracles (no code).**  SymPy, mpmath, SciPy and statsmodels are
*run* — from `scripts/` and by hand, in a local virtualenv — to produce
reference values that tests compare against, each cited next to the value
with the call that produced it.  These are numbers and expressions
(mathematical facts), not code; no part of those projects is included in, or
linked into, symplex.  All four are BSD-3-Clause licensed.

**As references for conventions.**  Where a mathematical choice is
conventional — a branch cut, the ordering of a polynomial's terms, the
parametrisation of a distribution, which identities `expand_log` applies —
symplex usually makes SymPy's choice, so that the two agree and SymPy can
serve as an oracle.  Behaviour is not copyrightable, but the choice is
acknowledged in the doc comment where it is made.

**Implementations that follow SymPy's.**  A few modules follow the
structure of SymPy's implementation of a published algorithm, and say so in
their documentation:

| symplex | follows | algorithm published in |
|---|---|---|
| `src/calculus/gruntz.rs` (`SubsSet`, the sign and MRV routines) | `sympy/series/gruntz.py` | D. Gruntz, *On Computing Limits in a Symbolic Manipulation System*, PhD thesis, ETH Zürich, 1996 |
| `src/calculus/risch/hermite.rs` | `hermite_reduce` in `sympy/integrals/risch.py` | M. Bronstein, *Symbolic Integration I*, §2.3 |
| `src/domains/exact_matrix.rs` (LLL reduction order and rounding) | `DomainMatrix.lll` / `_ddm_lll` | A. K. Lenstra, H. W. Lenstra, L. Lovász, *Math. Ann.* 261 (1982) |
| `src/plotting/textplot.rs` | `sympy/plotting/textplot.py` | — |
| `src/transforms/pattern.rs` (`condition_pow_pow`) | the branch condition of `Pow._eval_power` | the principal branch of `z^a` |

To the extent any of these is a derivative of SymPy, SymPy's licence
applies to that part, and its notice is reproduced here as clause (a)
requires:

```text
Copyright (c) 2006-2023 SymPy Development Team

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of SymPy nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
```

Everything else in symplex is original work under the crate's licence
(MIT OR Apache-2.0, see `LICENSE-MIT` and `LICENSE-APACHE`), implemented
from the mathematical literature cited in the source.
