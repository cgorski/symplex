# golden — golden-output harness

A deterministic "golden output" harness: it runs a fixed corpus of
expressions through ~30 symplex operations and writes one TSV line per
`(case, op)`.  Run it on two trees and diff the TSVs to prove that a refactor
changed no output, or to list exactly which outputs an intended change
altered.

Unlike the single-file probes in `probes/` (copied into `examples/` to run),
this is its own crate, like `rubi-harness`: it needs a driver, a custom global
allocator and worker threads, and it must build against *any* tree.
`Cargo.toml` depends on `symplex = { path = "../.." }` and declares an empty
`[workspace]`, so the directory `probes/golden/` copied into another checkout
builds against that checkout.  `Cargo.lock` is kept (the local `.gitignore`
re-includes it) so both trees resolve the same dependency versions.  Only the
public API is used (`symplex::prelude::*`, plus `symplex::base::libfn::LibFn`
for the function-table check).

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI, per-op worker thread + timeout, panic capture, TSV output |
| `src/ops.rs` | the operations (one fresh `Context` per `(case, op)`) |
| `src/cases.rs` | case generator (own SplitMix64, frozen function table) |
| `src/alloc.rs` | counting global allocator, aborts above `--mem-mb` |
| `driver.py` | parallel driver: slices, restarts, merged TSV, summary, meta |
| `run_golden.sh` | `cargo build --release` + `driver.py` |
| `golden_diff.py` | compare two TSVs; optional single-case reruns |
| `fixed.txt` | a small fixed corpus for `--file` mode |

## Running it

On a big host (all numbers below are m6: 64 cores, 123 GB).  Never build
against a tree another process is editing: freeze a copy first.

```sh
# 1. a frozen tree (here: 0.48.0) with the harness next to it
git -C symplex archive v0.48.0 | ssh m6 'mkdir -p ~/g48 && tar -x -C ~/g48'
rsync -a --exclude target symplex/probes/golden/ m6:g48/probes/golden/

# 2. baseline: build + 200,000 cases on 48 processes (~2.5 min)
ssh m6 'source ~/.cargo/env && cd ~/g48/probes/golden && \
        PYTHON=~/venv/bin/python ./run_golden.sh ~/golden/base_048.tsv'

# 3. the tree under test: same harness directory, same command
git -C symplex archive <rev> | ssh m6 'mkdir -p ~/gnew && tar -x -C ~/gnew'
#    (or, for uncommitted work: rsync -a --delete --exclude target --exclude .git symplex/ m6:gnew/)
rsync -a --exclude target symplex/probes/golden/ m6:gnew/probes/golden/
ssh m6 'source ~/.cargo/env && cd ~/gnew/probes/golden && \
        PYTHON=~/venv/bin/python ./run_golden.sh ~/golden/new.tsv'

# 4. compare; re-run the changed pairs singly on both binaries
ssh m6 'cd ~/gnew/probes/golden && ~/venv/bin/python golden_diff.py \
        ~/golden/base_048.tsv ~/golden/new.tsv -n 10 --changed-out ~/golden/changed.tsv \
        --rerun-base ~/g48/probes/golden/target/release/golden \
        --rerun-new  ~/gnew/probes/golden/target/release/golden'
```

`golden_diff.py` exits 0 iff nothing changed.  Use the *same harness
version* on both trees (`<out>.meta.json` records `gen_version` and a
fingerprint of the harness sources; the diff warns when they differ, and an
`expr` difference means the inputs themselves differ).

Driver options: `--seeds A..B` (default `0..200000`), `--ops a,b|all`,
`--timeout-ms` (default 5000), `--mem-mb` (default 4096), `-j` (default 48),
`--slice` (cases per process, default 100), `--file PATH` (fixed list; case
id = line number), `--stall-s`.  The binary alone:

```sh
golden --seeds 1000..1010 --ops simplify,integrate --timeout-ms 5000
golden --file fixed.txt --seeds 3..4          # one line of the fixed corpus
golden --list-ops | --list-fns | --info
```

A single case is reproduced with `golden --seeds C..C+1 --ops OP`; the input
string is the case's `expr` line.

## Output

`case_id \t op \t status \t result`, flushed per line, sorted by case id and
then the canonical op order (`golden --list-ops`).

| status | meaning |
|--------|---------|
| `OK` | the op returned; `result` is its text |
| `ERR` | the API returned an error; `result` is its `Display` (`parse: …` if the input did not parse) |
| `PANIC` | caught by `catch_unwind`; `result` is the panic message (the location goes to stderr and `<out>.panics.txt`) |
| `TIMEOUT` | the op did not finish within `--timeout-ms` (`>5000ms`), or the driver killed a silent process (`driver-kill`) |
| `OOM` | the process exceeded `--mem-mb` live heap (recorded by the driver) |
| `CRASH` | any other abnormal exit — signal, stack overflow (recorded by the driver) |

Results have tab/newline/CR/control characters escaped (`\t`, `\n`, `\r`,
`\xHH`; backslashes are left alone, so LaTeX stays readable).  A result longer
than 400 characters (after escaping) is replaced by `len=<chars> fnv=<FNV-1a
64 of its UTF-8, hex>`.  The pseudo-op `expr` (the input string) is never
hashed.

## Cases

`src/cases.rs` maps a case id to an expression string with its own SplitMix64
seeded from the id — no dependency on symplex, so a case id means the same
input on every tree.  Families by `id % 20`:

| `id % 20` | family |
|-----------|--------|
| 0–5 | random tree in `x`, `y`, depth ≤ 3, `+ − · / ^` and any function |
| 6–7 | the same without symbols (numeric evaluation paths) |
| 8–10 | polynomials / rational functions in `x` (optionally parameter `y`): expanded and factored forms, shared factors, sums of partial fractions, algebraic coefficients (`√c`, `i`) — `factor`/`cancel`/`apart`/gcd paths |
| 11–12 | trigonometric / hyperbolic identities with varied arguments |
| 13–14 | equations, written `lhs - (rhs)` (polynomial, exponential, log, trig, radical, rational, Lambert-W, no-closed-form, random) |
| 15–19 | one function from the table on shallow arguments, cycling through the whole table every 448 cases |

Leaves: `x`, `y`, integers −3…5, rationals, `pi`, `E`, `I`, `oo`,
`EulerGamma`, `GoldenRatio`, `Catalan`, `sqrt(2)`.  Exponents are small
integers/rationals, a symbol, or a power-free subtree (no towers).

The function table is **frozen** in the harness: every dedicated parser form
(`sin` … `acsch`, `exp`, `log`/`ln`/`log(x, b)`, `sqrt`, `cbrt`, `abs`, `sign`,
`floor`, `ceiling`, `re`, `im`, `conjugate`, `arg`, `gamma`, `loggamma`,
`digamma`, `polygamma`, `erf`, `erfc`, `lambertw` (1 and 2 args), `beta`,
`Si`, `Ci`, `Ei`, `li`, `E1`, `zeta`, `factorial`, `binomial`, `Heaviside`,
`DiracDelta`, `KroneckerDelta`, `Min`/`Max` (2 and 3 args), `atan2`) — 62
forms — and all 50 entries of `LibFn::ALL` at 0.48.0 with their
`LibFn::arity()`.  It is frozen rather than read from `LibFn::ALL` at run
time because a tree that adds a library function would otherwise reshuffle
every random case.  `golden --list-fns` compares the table with the tree's
`LibFn::ALL` (`.name()`, `.arity()`) and prints `UNCOVERED` / `ARITY` /
`GONE` lines; the driver repeats them as warnings.  Extending the table (or
any other generator change) must bump `GEN_VERSION` and needs a new baseline.

## Operations

Each op re-parses the input in its own fresh `Context`; `x` is the variable.

| op | call |
|----|------|
| `display` | `e.to_string()` |
| `reparse` | `ctx.parse(&e.to_string())` → `eq` / `ne: <reparsed>` |
| `simplify`, `expand`, `together`, `eval` | `e.simplify()`, `e.expand()`, `e.together()`, `e.eval()` |
| `factor`, `factor_all` | `e.factor(&x)`, `e.factor_all()` |
| `cancel`, `apart` | `e.cancel(&x)`, `e.partial_fractions(&x)` (symplex's `apart`) |
| `gcd` | `(n, d) = e.as_numer_denom()`, `n.poly_gcd(&d, &x)` |
| `diff`, `integrate` | `e.diff(&x)`, `e.integrate(&x)` |
| `series` | `e.series(&x, &0, 4)` |
| `limit0`, `limitoo` | `e.limit(&x, &0)`, `e.limit(&x, &ctx.infinity())` |
| `solve`, `solve_set` | `e.solve(&x)` (principal; `Err` → ERR), `e.solve_as_set(&x)` |
| `evald30`, `evalf64` | `e` at x = 7/10, y = −13/10 (`subs`; no-op if symbol-free), then `eval_decimal(30)` / `eval_f64()` (`{:?}`) |
| `compile` | `e.compile(&["x", "y"])?.try_call(&[0.7, -1.3])` (`{:?}`) |
| `latex`, `mathml` | `e.to_latex()`, `e.to_mathml()` |
| `json`, `json_rt` | `e.to_json()`; `ctx.from_json(&json)` → `eq` / `ne: …` |
| `is_positive`, `is_real`, `is_zero` | `Option<bool>` as `{:?}` |
| `free_symbols` | `e.free_symbols()` (in symplex's order) |
| `trigsimp` | `e.simplify_trig()` |

There is no separate `evalf`: `eval` is the exact evaluation, `evald30` the
numeric one.

## Isolation and determinism

* **Fresh `Context` per `(case, op)`**, with `x`, `y`, `z` created in that
  order before parsing.  Symbol ids, hash-consing order, assumption caches
  and per-context budgets therefore never depend on earlier cases or ops,
  and running one op alone (`--ops simplify`, or a single-case rerun) gives
  the same line as running all of them.
* **Fresh thread per op.**  symplex keeps `thread_local!` caches and depth
  counters (`integrate.rs`, `solve.rs`, `series.rs`, `evalf/*`, `trig_integ.rs`);
  each op starts with them empty.
* **Wall-clock dependence.**  The numerical quadrature in
  `transforms/evalf/quad.rs` gives up after 20 s (2.5 s for its "light"
  first try); `base/budget.rs` deadlines serve linprog / certificates.  None
  of these ops reach them on this corpus (no result mentions a time limit).
  The per-op `--timeout-ms` is the only time dependence, and pairs involving
  TIMEOUT/OOM are excluded from diffs.
* **`std` `HashMap`/`HashSet`** (randomly seeded per process) appear in
  `solve.rs`, `risch/tower.rs`, `groebner.rs` only as visited-sets and
  lookup caches, never iterated.  symplex spawns no threads and uses no
  entropy.

Check at 0.48.0: the default range was run three times — 48 processes ×
100-case slices, 40 × 37-case slices, and the baseline via `run_golden.sh`
(48 × 100) — plus 3 ops alone (`simplify,integrate,evald30`, 250-case
slices): **all diffs empty**; the only differing pairs were TIMEOUTs (164,
157 and 165 of 6.2 M pairs; the 157 are common to all).  Re-run singly with
120 s, 143 of the 164 finish (4–120 s, mostly `series` of nested special
functions), so they are slow, not hung, and borderline cases flip between
OK and TIMEOUT with machine load — which is why the diff ignores them.  Raising `--timeout-ms` turns some into OK lines but costs time;
5000 ms keeps the full run at about 2.5 minutes.

## Timeouts, memory, crashes

Each op runs on a worker thread (512 MiB stack); the main thread waits with
`recv_timeout`.  A thread cannot be killed, so after a TIMEOUT the binary
prints the line and exits with code 3; the driver restarts it with
`--after <op>`, i.e. at the next op of the same case.  The global allocator
counts live requested bytes and aborts above `--mem-mb` (printing
`golden: OOM`); the driver records `OOM` for the running pair — the
successor of the last complete line in the canonical op order — and resumes
after it.  Other abnormal exits are `CRASH` (`stack overflow`, `SIGSEGV:
…`), and a process silent for `2·timeout + 30 s` is killed and recorded as
`TIMEOUT driver-kill`.  The restart paths are exercised with
`GOLDEN_FAULT=kind:case:op` (`panic`, `abort`, `hang`, `oom`, `stall`).

## Runtime at 0.48.0 (m6, 48 processes)

200,000 cases × 31 lines = 6.2 M lines in 131 s of driver wall time (135 s
with the no-op `cargo build`; 314 MB TSV, 52 MB gzipped); 400,000 cases took
256 s.  Baseline `~/golden/base_048.tsv`:

| op | OK | ERR | TIMEOUT |
|----|---:|----:|--------:|
| `solve` | 71,171 | 128,829 | 0 |
| `evald30` | 178,051 | 21,949 | 0 |
| `evalf64` | 161,278 | 38,722 | 0 |
| `compile` | 163,314 | 36,686 | 0 |
| `series` | 199,845 | 0 | 155 |
| `limit0` | 199,991 | 0 | 9 |
| `limitoo` | 199,999 | 0 | 1 |
| all 24 other ops | 200,000 | 0 | 0 |

No PANIC, OOM or CRASH in 200,000 cases (nor in 400,000); no `reparse` or
`json_rt` mismatch.
