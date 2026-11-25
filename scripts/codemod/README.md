# codemod — the rewrite toolkit of the symplex 1.0 API work (WP-05)

One tool, `codemod.py`: Python 3, standard library only. It rewrites the *user side* of the library — tests,
examples, benches, the book, the README, the side crates and the doctests in `src/` — for the mechanical parts
of the API redesign (`API_1.0.md` Parts 3 and 8). It works on **text**: it has a tokenizer for Rust, not a parser
and not a type checker. What it is not sure about it does not rewrite: it lists the site, with file, line and the
reason, for a person.

**Dry run is the default.** Nothing is written unless `--apply` is given.

```sh
python3 codemod.py list                                   # the rule sets, their rules, what each needs
python3 codemod.py truth                                  # dry run on the default roots of the repository
python3 codemod.py truth --diff truth.diff --sites truth.tsv   # ... with the unified diff and every site as TSV
python3 codemod.py truth tests/unit/test_sets.rs README.md     # only these files or directories
python3 codemod.py domains --rules domains.dist-ctor,domains.density   # only some rules of a set
python3 codemod.py truth --apply                          # rewrite the files in place
python3 -m unittest discover -s tests                     # the unit tests (54 tests, 338 rule cases)
```

The repository is found from `--repo`, `$SYMPLEX_REPO`, or the checkout that contains the script. Default roots:
`tests`, `examples`, `benches`, `book/src`, `README.md`, `src`, `fuzz`, `symplex-wasm`, `symplex-build`,
`symplex-macros`, `rubi-harness`, `probes`. In `src` only the doctests are touched; `--src-code` lets the rules
that are made for the library's own code (`numbers.*`, `errors.construct`, `errors.doc-link`) run there too.

The dry run of 2026-11-24 on commit `7632d49`, with the comparison against the reviewers' counts, is `DRYRUN.md`.

## The rule sets

| rule set | package | what it does | needs when the package runs |
|---|---|---|---|
| `truth` | T2 | reads of yes/no queries: `assert_eq!(q, Some(true))` → `assert_eq!(q, Truth::True)`, `q == Some(true)` → `q.is_true()`, `.unwrap_or(false)`, `.is_none()`, `match` arms, `matches!`, expected-output comments; `assert!(check_ode_solution(..))` → `.is_true()`; three renames | `tables/truth_queries.tsv` from the listing and `api_fate.tsv` of the day; `Truth` exported and in the prelude (T1) |
| `errors` | E1, E2a, R1 | `matches!(r, Err(SymplexError::V { .. }))` → `matches!(r, Err(e) if e.kind() == ErrorKind::V′)`; arms; `if let`; `try_x` → `x`; `.unwrap()` / `?` on calls that now return `Result`; doc links; constructors (library) | `tables/error_kinds.tsv` (E1 fills in the constructor column), `tables/try_pairs.tsv` |
| `numbers` | N3 (N2 with `--src-code`) | `BigInt` → `Integer`; `Ratio<BigInt>`, `BigRational`, `Q` → `Rational`; the `use` lines of the num crates; lists `Ratio::new` and D's other hand sites | `symplex::num` exists (N1) |
| `subs` | V2b | `subs_i64` → `subs`, `subs_map_i64` / `subs_map_with` → `subs_map` | `subs` takes `impl ToEx` |
| `paths` | S1 | written `symplex::…` paths and `use` items → the 1.0 paths; `use` lines for names that leave the prelude | `tables/path_moves.tsv`, `tables/prelude_moves.tsv` from the `api_fate.tsv` of the day |
| `domains` | ST1, ST2, LA1, LA2, PL1 | `Distribution::normal(m, s)` → `Distribution::new(Normal { mean: m, std: s })?`; `m.get(i, j)` → `&m[(i, j)]`; `try_get` → `get`; the domain renames | `tables/dist_ctors.tsv` regenerated from the source; `tables/domain_renames.tsv` |
| `core-renames` | NM1 (K2, SV1) | review E's rename table: renames, the `force` flags, `Ex::symbolic_sum(&b, ..)` → `b.formal_summation(..)`, the inequality solvers | `tables/core_renames.tsv`, `tables/homonyms.tsv` |

`python3 codemod.py list` prints every rule with its description. Each package uses only its own rules
(`--rules`), e.g. ST1: `--rules domains.dist-ctor,domains.density,domains.characteristic-function`.

## How a package uses the tool

1. **Regenerate the tables** the rule set reads (`python3 tables/build_tables.py --api-dir … --repo …`) and read
   the diff of the tables: a query that changed fate, a constructor whose fields changed, shows there.
2. **Dry run.** Read the per-rule table, then the list "NOT rewritten — for a person": every entry has a reason.
   Read the diff (`--diff`), at least one hunk of every shape. The tool cannot compile; you can.
3. **Change the library** (the package's real work), then `--apply`, then `cargo fmt` on the build machine (the
   tool keeps line breaks where it can, it does not format) and compile.
4. **Fix by hand** what the compiler and the "not rewritten" list name. Where a hand fix would look like an
   unmigrated site to the tool (see "Pairs" below), put `// codemod:skip` on the line or on a comment line
   above it: the tool then lists the site and leaves it.
5. **After rebasing** (a script package merges last in its block, `API_1.0.md` §8.1): do not resolve conflicts
   in rewritten files. Take the files from the new base, run `--apply` again, re-apply the hand fixes. The run
   is deterministic, and applying a rule set to its own output changes nothing.

## What the report says

Every site is in one of three states:

* **rewrite** — the tool edits it;
* **manual** — found, not edited; the reason says why ("an arm binds the value", "`is_hermitian` has two fates",
  "this file defines a function named `fibonacci`", "inside a `compile_fail` doctest", …);
* **unchanged** — found, and nothing has to be edited for it to compile: a printed query (its *output* changes
  from `Some(true)` to `True`), two queries compared with each other, a mention in prose. Counted so that the
  totals can be compared with the reviewers' counts.

`--sites FILE` writes one line per site (status, rule, class, name, corpus, file, line, reason, source line). A
`~` after the rule id in the listing marks a site that only appeared after another rewrite (its line number is
that of the rewritten text).

Never rewritten, always listed: a file with a `.stderr` beside it (a UI test: its recorded compiler output
depends on line and column), a `compile_fail` doctest, a page `migrating-<version>.md` of the book (it shows an
older API on purpose). A block of Markdown or of a doc comment that does not tokenize (`…` placeholders,
unbalanced brackets) is skipped and named under "notes".

## How the rules decide

* **By name, never by type.** A rule fires on the *name* of the method or path, taken from the reviewers'
  tables, and only when the value being read is visibly the result of a call of that name:
  `x.is_zero() == Some(true)` is a site, `flag == Some(true)` is not, `stack.pop() == Some(true)` is not.
* **Homonyms are refused.** Where the public API has the old name on a second type that keeps it (the 31 unit
  copies of `log_combine`; `GenPoly::is_constant`; a `Family` implementation's `density`), the rule refuses in a
  file — or a doctest — that uses that type. Where std has the name (`to_vec`, `downcast_ref`, `.add(`, `.pow(`)
  the rule only lists, or stays silent for names too common to list (`.row(`, `.col(`).
* **A file that defines a function of the name** (`fn fibonacci`, a trait method `fn get`) is not rewritten for
  that name: the call may be to its own function.
* **Imports.** `Truth` comes with the prelude; `ErrorKind`, `Integer`, `Rational`, the parameter records do not
  (Part 4, C-12). The tool adds `use` lines per scope (file, `mod`, doctest), grouped by module, after the last
  `use` of the scope; hidden (`# use …`) in a doctest that hides its imports; none in a Markdown fragment that
  has no `use` line at all. If another item of the same name is in scope it writes the full path instead.
* **Strings and comments are never rewritten**, with two exceptions that are rules of their own: the
  expected-output comment after a printed query (`// Some(true)` → `// True`, only when the comment has as many
  values as the line has queries) and the doc link ``[`SymplexError::V`]``.
* **Doctests are user code.** Fenced Rust blocks in `///` and `//!` comments and ```` ```rust ```` blocks in
  Markdown are tokenized on their own (the comment marker and the hidden-line marker `# ` taken away) and edited
  in place, with the marker re-inserted on every line the tool writes.

### Pairs: `try_x` and `x`

For the pairs of review C (and `try_get` / `get`), the `try_` form takes the plain name while the plain form
changes its type. After `try_fibonacci(n)` has become `fibonacci(n)`, nothing in the text says that this call was
never the old plain one. So:

* a `try_` call is renamed only where its `Result` (or `Option`) is read right at the call — `?`, `.unwrap()`,
  `.is_err()`, a `match` on `Ok` / `Err`, `matches!` — or the variable it is bound to is read so in the same
  block; any other `try_` call is listed;
* a plain call that is read as a `Result` is left alone (it is migrated already); every other plain call gets
  `.unwrap()`, or `?` where the enclosing function returns a `Result` that can hold a `SymplexError` and no
  closure lies in between.

Consequence: **after renaming a listed `try_` call by hand, mark the line `codemod:skip`** (or make the `Result`
reading visible at the call) before any later run of `errors.plain-unwrap` or `domains.matrix-get`.

### Options

| option | effect |
|---|---|
| `--opt truth.assert=is-true` | `assert_eq!(q, Some(true), "m")` → `assert!(q.is_true(), "m")` instead of the default `assert_eq!(q, Truth::True, "m")`. The default is the form of `API_1.0.md` §2.1 and review B §3.2 ("kept as `assert_eq!` so that a failure prints the actual value") |
| `--opt truth.names=B` | take every `Option<bool>` query to return `Truth`, as review B counted (for comparing counts only) |
| `--rules a,b` | only the rules whose id starts with one of the prefixes |
| `--src-code` | also the library's own code in `src/`, for the rules made for it |
| `--only code,doc,md` | only these kinds of text |
| `--corpus NAME` | treat every file as this corpus (files outside a repository) |

## Limits of the approach

1. **No types.** The receiver of a method call is a piece of text. The guards above (names from the tables, a
   visible reading, homonym refusal, local-definition refusal) make a wrong rewrite unlikely and, in almost
   every case, a compile error rather than a silent change — but not in every case in principle: a test's own
   type with a method named like a listed one, reached through a trait the file imports rather than defines,
   would be rewritten. Read the diff.
2. **No compiler, no formatter.** Nothing the tool writes has been compiled (the targets do not exist yet). It
   checks only that its output still tokenizes and is a fixpoint. `macro_rules!` bodies and macro arguments are
   token sequences to it, not expressions; code generated by macros, and `cfg`-ed variants, it sees as written.
   Run `cargo fmt` and compile after every `--apply`.
3. **Order and re-running.** Each rule set is idempotent on its own output (tested on every changed file of the
   tree). The rule sets are meant to run in the order of the blocks, each on a tree where the previous
   packages' *hand* work is done; a hand-renamed `try_` call needs the `codemod:skip` mark. Re-running after a
   rebase means running on the new base, not on a half-migrated tree.
4. **Meaning.** `!= Some(true)` → `!q.is_true()` and `.unwrap_or(true)` → `!q.is_false()` preserve behaviour
   exactly and get a `// codemod(truth): … check the reading` line, because the *intent* may have been the other
   one. `ComputationFailed` / `Unevaluable` → `Undetermined` is the design's default and is marked; the raise
   site may get another kind later. `is_satisfied` → `is_identity` follows review B, and the meaning of
   `is_identity` changes at 1.0.
5. **Not built** (listed by the tool or left to the compiler): review B's M2 and M3 (`Option` → `Result`,
   `.into_found()`); the function-to-method moves of review A §3 (`cross(&a, &b)` → `a.cross(&b)`,
   `parse::parse(&ctx, s)` → `ctx.parse(s)`); the compiler-guided `Symbol` pass of review E §3.1; everything the
   reviewers mark "by hand". `DRYRUN.md` §8 has the list.

## Layout

```
codemod.py              the command line
cm/lexer.py             tokenizer: comments (nested), strings (raw with any #, byte, C), chars vs lifetimes,
                        brackets with partners and parents, argument splitting, turbofish
cm/chain.py             postfix chains (the receiver of a method call), calls, macro calls, enclosing fn
cm/views.py             code / doctest / Markdown views with offset mapping; `use` trees; import insertion
cm/engine.py            sites, the fixpoint loop, overlap handling, the tree walk, the diff
cm/rules_*.py           one module per rule set; cm/rulesets.py registers them
tables/                 *.tsv read by the rules; build_tables.py regenerates the generated ones
tests/                  test_core.py, test_engine.py, test_rules.py + cases_*.py (input / expected output)
tools/                  compare_*.py (the comparisons of DRYRUN.md), sandbox.py, fixpoint_check.py
sandbox/                copies of eight files of the repository, applied to once and twice
out/                    the dry run of DRYRUN.md: reports, diffs, site lists, comparisons
```

Hand-written tables (the rules themselves; edit them): `core_renames.tsv`, `domain_renames.tsv`,
`error_kinds.tsv`. Generated: `truth_queries.tsv`, `try_pairs.tsv`, `path_moves.tsv`, `prelude_moves.tsv`,
`dist_ctors.tsv`, `homonyms.tsv` — each starts with a line naming its sources.

## Adding a rule

A rename is a row in `core_renames.tsv` or `domain_renames.tsv` (the shapes are described at the top of the
file) plus nothing else: `tests/cases_renames.py` generates its test from a template per shape. Any other rule is
a class with `id`, `description`, `trigger` (a cheap text prefilter — build it from the same table the rule
reads, a hand-written one once missed a name) and `find(view, cx)` that yields `self.rewrite(view, pos, edits)`,
`self.manual(view, pos, reason)` or `self.info(..)`; edits are `(start, end, text)` in the view's own offsets.
`tests/test_rules.py` fails until the rule has a case, and runs every case twice.
