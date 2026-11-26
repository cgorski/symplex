# hunt — a fix shown to hold beyond its reproducer

A bug fix comes with a small hunt: some hundreds of expressions of the same
class, evaluated on the tree before the fix and on the tree with it, and
checked against an independent computation.  This kit is that hunt, ready
made, on top of the golden harness (`probes/golden`):

| File | Purpose |
|------|---------|
| `hunt.sh` | builds `probes/golden` of the tree it stands in and runs it on **your** list of expressions: one line per (case, op) with the value or the error, and the CPU time |
| `hunt_diff.sh` (`hunt_diff.py`) | two such outputs: the changed pairs by op and by kind (value → value, value → refusal, refusal → value, …, slower by more than a factor) |
| `oracle.py` | one output against mpmath: a verdict per line (`agree`, `DISAGREE`, `oracle-failed`, `not-comparable`); `--table` gives "wrong before → wrong after" |
| `example_cases.txt` | the twenty expressions of the worked example below |

Nothing here is built by `cargo` in the symplex package; the scripts need
`sh`, Python 3 and, for `oracle.py`, mpmath.

## The cases file

One expression per line in symplex's input syntax; the case id is the line
number; blank lines and lines starting with `#` are skipped.  Generate it
however you like (a ten-line Python script per bug class is the usual way).
The ops are those of the golden harness (`probes/golden/README.md`,
"Operations"): `x` is the variable, and the numeric ops `evald30` and
`evalf64` evaluate at x = 7/10, y = −13/10.

## hunt.sh

```sh
probes/hunt/hunt.sh <cases-file> <out.tsv> [--ops a,b,c] [--jobs N] [--timeout S] [--mem-mb M]
```

`<out.tsv>` has the golden columns plus the CPU time of the op's thread in
milliseconds (`-` for the `expr` line and for a TIMEOUT), and results are
never replaced by their hash:

```
case_id <TAB> op <TAB> status <TAB> result <TAB> cpu_ms
```

`status` is `OK` (a value), `ERR` (a refusal; the result is the error text),
`PANIC`, `TIMEOUT` (over `--timeout` seconds, default 20), `OOM` or `CRASH`.
`<out.tsv>.summary.txt` has the statuses per op, `.panics.txt` the panic
locations.  Defaults: all ops, 8 processes.

`hunt.sh` builds the harness of **its own tree**.  For the "before" side,
copy the kit and the harness into the frozen base copy first — the same
harness sources on both sides, as for a golden diff (commands below).

## hunt_diff.sh

```sh
probes/hunt/hunt_diff.sh <base.tsv> <new.tsv> [--factor 2] [--min-ms 50] [-n 10] [--ops a,b] [--changed-out FILE]
```

A pair is `slower` when its CPU time grew by more than `--factor` and is at
least `--min-ms` (timer noise stays out); the last two columns are the CPU
seconds per op on each side.  Exit status 0 iff no value or status changed.

## oracle.py

```sh
python probes/hunt/oracle.py <out.tsv> [--ops evald30,evalf64,diff] [--timeout 20] [--mem-mb 2048] [--jobs 8]
python probes/hunt/oracle.py --table <base.tsv.verdicts.tsv> <new.tsv.verdicts.tsv>
```

* One **forked child per case**, killed after `--timeout` seconds and limited
  to `--mem-mb` of address space; a case that hits a limit is
  `oracle-failed`, the run goes on.
* `evald30` and `evalf64` are implemented: mpmath evaluates the input at
  **two precisions**, and the oracle counts only if the two agree (they are
  doubled up to twice when they do not).  The comparison is **numeric**:
  symplex's printed value is read back as a number and must lie within
  `--ulps` (default 2) units of the 30th digit, or `--f64-ulps` (default 4)
  units in the last place of an f64, of the oracle's value.
* `diff` is the worked example of a **symbolic hook** (`judge_diff`): mpmath
  differentiates the input numerically at three random points (seeded with
  the case id) and the result is evaluated there.  `integrate`, `limit0` and
  `limitoo` are marked stubs (`judge_integrate`, `judge_limit`) that say what
  to write; until then their lines are `not-comparable`.
* A function the oracle has no checked convention for is `not-comparable`
  (`no oracle for elliptic_k/1 (add it to FUNCTIONS)`), never a guess: add it
  to the `FUNCTIONS` table with its argument order.
* A refusal (`ERR`) is `not-comparable`; where the oracle does have a value
  it is counted in the `refused` column — the number to state when a fix
  turns answers into refusals.

Verdicts go to `<out.tsv>.verdicts.tsv` (case, op, status, verdict, detail,
result, input).  Exit status 1 iff there is a `DISAGREE`.

## Worked example

Twenty expressions, three ops, on the frozen base copy (`~/base_wt_a6`,
0.60.1) and on the tree with the change (`~/wt_a6`), as typed on the build
host; the output is that of 2026-11-26 on m6.

```sh
export PYTHON=~/venv/bin/python
BASE=~/base_wt_a6 TREE=~/wt_a6 OUT=~/scratch/wt_a6/hunt; mkdir -p $OUT

# 0. the base copy gets the same harness sources and the kit (not the build)
rsync -a --exclude target $TREE/probes/golden/ $BASE/probes/golden/
rsync -a $TREE/probes/hunt/ $BASE/probes/hunt/

# 1. before and after (the first run of each tree builds the harness: ~70 s)
cd $BASE && probes/hunt/hunt.sh probes/hunt/example_cases.txt $OUT/base.tsv --ops evald30,evalf64,diff
cd $TREE && probes/hunt/hunt.sh probes/hunt/example_cases.txt $OUT/new.tsv  --ops evald30,evalf64,diff
```

```
20 cases, 80 lines, 4 ops, wall 0 s, 8 jobs x 3-case slices, timeout 20000 ms, 0 restarts
op                   OK      ERR    PANIC  TIMEOUT      OOM    CRASH
expr                 20        0        0        0        0        0
diff                 20        0        0        0        0        0
evald30              18        2        0        0        0        0
evalf64              17        3        0        0        0        0
TOTAL                75        5        0        0        0        0
```

Three cases of `new.tsv` (`grep -P '^(1|18|19)\t' $OUT/new.tsv`):

```
1	expr	OK	sin(x)^2 + cos(2*y)	-
1	diff	OK	2*sin(x)*cos(x)	0.237
1	evald30	OK	-0.441872324819067703106076169247	0.383
1	evalf64	OK	-0.4418723248190677	0.283
18	expr	OK	sqrt(y) + log(y)	-
18	diff	OK	0	0.025
18	evald30	OK	0.262364264467491052035495986881 + 4.28176807868893121759869240885*i	0.508
18	evalf64	ERR	eval_f64 could not be computed: expression has nonzero imaginary part (im=4.281768078688931); use eval_complex64() for complex results	0.446
19	expr	OK	gamma(-2)	-
19	diff	OK	0	0.104
19	evald30	ERR	cannot evaluate: cannot evaluate infinity to finite precision	0.143
19	evalf64	ERR	cannot evaluate: cannot evaluate infinity to finite precision	0.080
```

```sh
# 2. what changed between the two trees
probes/hunt/hunt_diff.sh $OUT/base.tsv $OUT/new.tsv
```

```
op             pairs   same  val->val  val->ref  ref->val  ref->ref   ->PANIC   PANIC-> ->TIMEOUT TIMEOUT->  slower  cpu s base  cpu s new
diff              20     20         0         0         0         0         0         0         0         0       0        0.00       0.00
evald30           20     20         0         0         0         0         0         0         0         0       0        0.02       0.02
evalf64           20     20         0         0         0         0         0         0         0         0       0        0.01       0.01
TOTAL             60     60         0         0         0         0         0         0         0         0       0        0.03       0.03

RESULT: no changes
```

(This tree changes no library code, so the two sides are equal.)

```sh
# 3. each side against the oracle, then the table for the notes
$PYTHON probes/hunt/oracle.py $OUT/base.tsv
$PYTHON probes/hunt/oracle.py $OUT/new.tsv
$PYTHON probes/hunt/oracle.py --table $OUT/base.tsv.verdicts.tsv $OUT/new.tsv.verdicts.tsv
```

```
op            pairs  agree  DISAGREE  oracle-failed  not-comparable  (refused)
diff             20     19         0              1               0          0
evald30          20     18         0              0               2          0
evalf64          20     17         0              0               3          0
20 cases in 0.6 s, mpmath 1.3.0; verdicts in /home/ubuntu/scratch/wt_a6/hunt/new.tsv.verdicts.tsv
```

```
before -> after            wrong          refused            agree    oracle-failed   not-comparable
diff                   0 -> 0           0 -> 0         19 -> 19           1 -> 1           0 -> 0
evald30                0 -> 0           0 -> 0         18 -> 18           0 -> 0           2 -> 2
evalf64                0 -> 0           0 -> 0         17 -> 17           0 -> 0           3 -> 3
0 pairs changed their verdict (`refused` is part of `not-comparable`)
```

The detail column says how each verdict was reached
(`grep -P '^(9|14|18|19)\t' $OUT/new.tsv.verdicts.tsv | cut -f1-5`):

```
9	diff	OK	agree	3 random points to 1e-10 relative
9	evald30	OK	agree	oracle 0 (to 1.0e-35); |difference| 0.0, allowed 1.0e-35
9	evalf64	OK	agree	oracle 0 (to 1.0e-22); |difference| 0.0, allowed 1.0e-22
14	diff	OK	agree	3 random points to 1e-10 relative
14	evald30	OK	agree	oracle 100000000000000000000.0; |difference| 3.33e-21, allowed 2.0e-10
14	evalf64	OK	agree	oracle 100000000000000000000.0; |difference| 3.33e-21, allowed 6.55e+4
18	diff	OK	agree	3 random points to 1e-10 relative
18	evald30	OK	agree	oracle (0.2623642644674910520354959868809544 + 4.281768078688931217598692408846257j); |difference| 3.74e-30, allowed 2.0e-29
18	evalf64	ERR	not-comparable	refused, and the oracle value is not real: (0.262364264467491052035 + 4.2817680786889312176j)
19	diff	OK	oracle-failed	only 0 usable points (ValueError, ValueError, ValueError, ValueError)
19	evald30	ERR	not-comparable	refused, and no oracle value: oracle raised ValueError: gamma function pole
19	evalf64	ERR	not-comparable	refused, and no oracle value: oracle raised ValueError: gamma function pole
```

Case 9 is `pi^2/6 - zeta(2)`: both precisions of the oracle vanish, so the
value is zero to 35 places and symplex's `0` agrees.  Case 14 is
`tan(pi/2 - 1/10^20)`: the first two precisions disagreed and the doubled
ones settled it.  Case 19 is `gamma(-2)`: symplex refuses, mpmath raises —
nothing to compare — and the derivative `0` of a pole cannot be checked.

### What a wrong answer looks like

No tree at hand has one in these twenty, so one digit of case 1 is changed
by hand (the 20th of 30):

```sh
sed 's/^1\tevald30\tOK\t-0.441872324819067703106/1\tevald30\tOK\t-0.441872324819067703116/' $OUT/new.tsv > $OUT/demo_wrong.tsv
$PYTHON probes/hunt/oracle.py $OUT/demo_wrong.tsv          # exit status 1
$PYTHON probes/hunt/oracle.py --table $OUT/base.tsv.verdicts.tsv $OUT/demo_wrong.tsv.verdicts.tsv
```

```
op            pairs  agree  DISAGREE  oracle-failed  not-comparable  (refused)
diff             20     19         0              1               0          0
evald30          20     17         1              0               2          0
evalf64          20     17         0              0               3          0
20 cases in 0.6 s, mpmath 1.3.0; verdicts in /home/ubuntu/scratch/wt_a6/hunt/demo_wrong.tsv.verdicts.tsv
DISAGREE case 1 evald30: input sin(x)^2 + cos(2*y)
    symplex -0.441872324819067703116076169247
    oracle -0.441872324819067703106076169247026; |difference| 1.0e-20, allowed 2.0e-30
```

```
before -> after            wrong          refused            agree    oracle-failed   not-comparable
diff                   0 -> 0           0 -> 0         19 -> 19           1 -> 1           0 -> 0
evald30                0 -> 1           0 -> 0         18 -> 17           0 -> 0           2 -> 2
evalf64                0 -> 0           0 -> 0         17 -> 17           0 -> 0           3 -> 3
1 pairs changed their verdict:
  case 1 evald30: agree -> DISAGREE   input sin(x)^2 + cos(2*y)
    before OK -0.441872324819067703106076169247  [oracle -0.441872324819067703106076169247026; |difference| 2.6e-32, allowed 2.0e-30]
    after  OK -0.441872324819067703116076169247  [oracle -0.441872324819067703106076169247026; |difference| 1.0e-20, allowed 2.0e-30]
```

and `hunt_diff.sh $OUT/base.tsv $OUT/demo_wrong.tsv` names the same pair as
`value -> value` (exit status 1):

```
== evald30: value -> value: 1 ==
  case 1  input: sin(x)^2 + cos(2*y)
    base OK    -0.441872324819067703106076169247
    new  OK    -0.441872324819067703116076169247

RESULT: 1 changed pairs, 0 slower
```

## At hunt size

The first 3,000 generated cases of the golden base run of 0.60.1
(`~/golden/base_s18.tsv`, which `oracle.py` reads as well) take 7.3 s with
`--jobs 32` on m6:

```
op            pairs  agree  DISAGREE  oracle-failed  not-comparable  (refused)
diff           3000   2040         0             56             904          0
integrate      3000      0         0              0            3000          0
limit0         3000      0         0              0            3000          0
limitoo        3000      0         0              0            3000          0
evald30        3000   2127         0              2             871          7
evalf64        3000   1917         0              2            1081          5
```

(`integrate` and the limits are the unwritten hooks; most other
`not-comparable` lines are functions missing from the table, and for `diff`
the hashed long results of a golden TSV.)

## Limits

* The oracle is as good as its table: 77 function forms, argument order
  as in mpmath's documentation.  Branch-cut conventions were not compared
  one by one (no `DISAGREE` in the 3,000 cases above) — a `DISAGREE` on a
  branch cut is a question to settle, not yet a bug.
* The oracle computes in floating point.  At an exact singularity it sees a
  huge finite number, not a pole (`sign(cot(pi))` is ±1 for it), and it
  follows mpmath's conventions (`arg(0)` is 0): read a `refused` line before
  counting it as a refusal of something that has a value.
* The oracle's `evalf64` rule assumes a correctly rounded result up to
  `--f64-ulps`; a result that is exactly zero by cancellation is accepted
  only below 10⁻²².
* The numeric ops are tied to the harness's point (7/10, −13/10); to
  evaluate elsewhere, substitute the numbers in the cases file.
* The CPU column exists on 64-bit Linux only (`probes/golden/src/cpu.rs`);
  elsewhere it is `-` and `hunt_diff.sh` reports no `slower`.
* The time limit of `oracle.py` is per case (all its ops), in wall seconds.
