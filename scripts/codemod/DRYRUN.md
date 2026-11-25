# WP-05 — dry run of the codemod toolkit on today's tree

Tree: `symplex` at `7632d49` (0.60.0 plus documentation), read only; `git status` of the repository is empty
before and after. Tool: `codemod.py` in this directory. Every number below was produced on 2026-11-24 by

```sh
export SYMPLEX_REPO=<the checkout>
python3 codemod.py <ruleset> --diff out/<ruleset>.diff --sites out/<ruleset>.sites.tsv > out/<ruleset>.report.txt
python3 tools/compare_B.py out/truth.sites.tsv ../review_B_truth/data/callsites.tsv   # -> out/compare_B.txt
python3 tools/compare_all.py                                                           # -> out/compare_all.txt
python3 tools/compare_regex.py <name> <reviewer's regex> out/<ruleset>.sites.tsv <rule> # -> out/compare_regex.txt
```

`out/<ruleset>.report.txt` holds the per-rule table, the per-corpus table and **every site left for a person with
file:line and the reason**; `out/<ruleset>.diff` is the unified diff of what `--apply` would write.

**Nothing was compiled.** The rewrites' targets (`Truth`-returning queries, `ErrorKind`, `symplex::num`, the new
paths) do not exist yet, so no output of the tool can be compiled today. What was checked instead is in §9.

## 1. The table

Roots: `tests`, `examples`, `benches`, `book/src`, `README.md`, `fuzz`, `symplex-wasm`, `symplex-build`,
`symplex-macros`, `rubi-harness`, `probes`, and the doctests in the doc comments of `src` (not the library's own
code). *found = rewrite + manual + unchanged*; "manual" is what a person does by hand; "unchanged" is counted but
needs no edit to compile (a printed value, a mention in prose).

| rule set (rules) | found | would rewrite | left for a person | unchanged | the reviewers' figure | agreement |
|---|---:|---:|---:|---:|---|---|
| `truth` (14) | 1,951 | 1,518 | 221 | 212 | B: 1,591 user-side sites, 1,352 in seven shapes; 183 in `src` doc comments | user side, B's locator scope: **1,602 found**; seven shapes **1,348 found, 1,241 rewritable** (1,348 with B's assumption, §2) |
| `errors` (7) | 2,515 | 2,318 | 197 | — | C: 524 patterns by script, 145 by hand; 434 `try_` and 3,245 plain calls; 825 doc mentions | patterns: **510 sites hold all 524 but 3**; **145 = 145** by hand; pairs: see §3 |
| `numbers` (7) | 2,562 | 2,371 | 191 | — | D: 227 files; 88 `Ratio::new`; 7 `parse_bytes`; "about 662" `Q` | `Ratio::new` **88 = 88** (+ 4 `new_raw`); `parse_bytes` 7 = 7; `Q` 398 tokens (§4) |
| `core-renames` (50) | 1,300 | 926 | 374 | — | E: 2,106 user call sites, 1,300 "pure text" | per row within a few sites of E's regex (§5) |
| `subs` (3) | 712 | 712 | 0 | — | E: 708 + 3 + 10 = 721 | **712**; the 9 others are in strings and comments |
| `paths` (3) | 2,227 | 1,790 | 437 | — | A: 743 written paths + 456 in doc comments; 417 imports in 247 test files | A's 417 reproduced exactly with A's method: 235 need the import, 182 do not (§6) |
| `domains` (47) | 3,195 | 2,553 | 642 | — | F: 938 constructors, 543 `.get(i, j)`, 385 `p_value_*`, 191 `.density(` | 930 / 539 / 388 / 189 found (§7) |

131 rules in all. The count that WP-05's row asks for — "counts equal the reports' counts" — holds in this sense:
for every reviewer figure that could be re-derived (their site list, or their regular expression re-run on the
same lines), the tool's count differs by fewer than ten sites, or the difference is listed below site by site
and comes from one of four causes:

1. **the regular expression matched inside a comment or a string literal** (the tokenizer does not);
2. **the regular expression matched a homonym** (`str::contains` in a guard, `RatingTable::get`);
3. **the merged design (Part 4) changed the fate of the item after the review** (queries that no longer return
   `Truth`; units and ten statistics modules under `experimental`);
4. **the tool lists instead of rewriting**, because the shape is not one it is sure about.

## 2. `truth` — against review B (`review_B_truth/data/callsites.tsv`, matched by file, line and query name)

User side (tests + examples + book + README), B's locator scope (the rules `truth.rename`, `truth.bool-query`,
`truth.printed-comment` are outside it and counted separately):

| B's reading | B | tool: found | tool: rewrite | note |
|---|---:|---:|---:|---|
| `assert_eq!(q, Some(true))` | 766 | 766 | 714 | |
| `assert_eq!(q, Some(false))` | 380 | 380 | 338 | |
| `assert_eq!(q, None)` | 139 | 139 | 128 | |
| `assert_ne!(q, Some(..))` | 43 | 43 | 42 | |
| `==` / `!=` with `Some(true)`, `Some(false)`, `None` | 16 | 12 | 11 | B's other 4 are in comments (below) |
| `q.unwrap_or(false)` | 5 | 5 | 5 | |
| `q.is_none()` | 3 | 3 | 3 | |
| **the seven shapes** | **1,352** | **1,348** | **1,241** | 107 listed: the query does not return `Truth` at 1.0 |
| other readings (`match`, `matches!`, `if let`, `unwrap`, bound, compared, unclassified) | 93 | 57 | 6 | 21 of B's are homonyms (below) |
| printed, statement, mention in a comment or prose | 146 | 197 | — | the tool also counts ambiguous names beside a value |
| **total** | **1,591** | **1,602** | | |

In the doc comments of `src`: B 183, tool 184 (166 in the seven shapes, 145 rewritable).

**1,241, not 1,352.** B's count takes every one of the 152 `Option<bool>` functions to return `Truth`. The merged
design does not: `poly_is_nonnegative_on`, `poly_is_positive_on`, `is_irreducible`, `is_squarefree` move to
`Poly` (F PE-MOVE; Part 4 C-17 gives two of them `Result<bool>`), `Expr::is_hermitian` is removed while
`Matrix::is_hermitian` stays (C-27: two fates under one name), `is_prime_value` and the unit `try_` twins are
removed, the three `vector` queries return `Result<Truth>`. The tool reads the merged fate per name from
`tables/truth_queries.tsv` (generated from `api_fate.tsv`) and lists those 107 sites with the reason. With
`--opt truth.names=B` (B's assumption) it rewrites 1,348 of 1,348: `out/truth_B.report.txt`.

**Sites only one side has** (after pairing, `out/compare_B.txt`): 44 only in B's list, 48 only in the tool's.
Sampled, with who is right:

| site | B | tool | right |
|---|---|---|---|
| `tests/v18/v18_cox.rs:1049` `Err(SymplexError::InvalidArgument { reason, .. }) if reason.contains("x[2][0]")` | `matches_other` (query `contains`) | no site | tool: `str::contains` in a guard (15 such) |
| `tests/v36/v36_s15p1_evalf.rs:19` `matches!(&d, Err(..Unevaluable { reason }) if reason.contains("underflows"))` | `matches_other` | no site | tool (same) |
| `tests/perf_assert/mod.rs:62` `let Some(text) = value.map(str::trim).filter(\|t\| !t.is_empty()) else {` | `if_let` (query `is_empty`) | no site | tool: `str::is_empty` inside a closure (6 such) |
| `tests/v03/v03_exact_matrix.rs:64` `let Some(p) = (pr..n).find(\|&i\| !rows[i][c].is_zero()) else {` | `if_let` | no site | tool: a rational's `is_zero`, in a closure |
| `tests/unit/test_foundations.rs:657` `//     assert!(at_4.is_satisfied() == Some(true));` | `eq_some_true` | mention | tool: commented-out code (4 of B's 16 comparisons are comments or prose) |
| `tests/v36/v36_w6_facets.rs:67` ``/// … were all `is_real() == None`, and`` | `eq_none` | mention | tool: prose |
| `examples/factoring_and_ntheory.rs:158` `"Ex: 97.is_prime_value() = {:?}",` | `unclassified` | no site here; the call on line 159 is "printed" | tool: line 158 is a string literal |
| `src/base/truth.rs:151` `assert_eq!(SignSet::REAL.is_positive(), Truth::Unknown);` | `assert_eq_other` | no site | tool: already a `Truth` (and `SignSet` is another type) |
| `tests/proptests/proptest_canon.rs:481` `prop_assert_eq!(expr.query(Props::INTEGER), Some(true));` | `assert_eq_some_true` | same, rewritten | both: B's regex matched the suffix of `prop_assert_eq`; the tool handles the macro by name (16 sites) |
| `tests/unit/test_equation.rs:132` `evaled.is_satisfied(), Some(true)` | `assert_eq_some_true` (query `is_satisfied`) | renamed to `is_identity`, then rewritten | both (7 + 1 + 1 sites appear under the new name) |
| `tests/v36/v36_s17b_sets.rs:560` ``// (`[0, 2].contains(e)` was `Some(true)`, …`` | no site (ambiguous name) | mention | a comment the T2 engineer will want to see; nothing is rewritten (20 + 7 + 3 such) |
| `README.md:411` `a.contains(&x);                                       // None` | no site | statement; its comment becomes `// Unknown` | tool: B's locator skips ambiguous names that are not read |

Other rules of the set, user side: `truth.match-arms` 5 of 8 `match`es rewritten (B: "the arms of 9 `match`es"; 3
bind the value); `truth.bool-query` 51 rewritten, 25 listed (B: 82 + 11 sites, 4 negated assertions by hand; the
tool lists the 4 and every binding to a variable); `truth.rename` 36 = 15 + 12 + 9 (B's regular expression:
16 + 13 + 11); `truth.printed-comment` 28 comments rewritten, 12 listed (9 for a query that no longer returns
`Truth`, 3 where the comment has more values than the line has queries).

**Needs when T2 runs:** `tables/truth_queries.tsv`, regenerated by `tables/build_tables.py` from the listing of
the day and the then-current `api_fate.tsv`: the tool keys on the method NAMES in it and never fires on
`x == Some(true)` unless `x` is a call of one of them. `Truth` must be exported at `symplex::Truth` and by the
prelude (T1). B expected five test files to need `use symplex::Truth;`. The tool adds the import in no test
file: each of the 129 test files in which it writes `Truth::…` has the prelude in scope at the site (B's five
are files whose sites are of other kinds). It adds it once, in a doctest of `src/poly/factor_zassenhaus.rs`.

## 3. `errors` — against review C (`review_C_refusals/data/error_users.tsv`, `try_pairs_060.tsv`)

Patterns, per corpus (C counts occurrences of a variant; a site of the tool is one pattern, which may name
several variants):

| corpus | C: tests a variant | C: binds a field | tool: rewrite | tool: by hand (binds) | tool: by hand (other) |
|---|---:|---:|---:|---:|---:|
| tests | 459 | 130 | 458 | 130 | 6 patterns outside `Err(..)` or in a place the tool does not rewrite, 3 constructions |
| examples | 6 | 4 | 6 | 5 | 3 constructions |
| book | 24 | 10 | 22 | 10 | |
| README | 1 | 0 | 1 | 0 | |
| fuzz | 7 | 0 | 4 | 0 | |
| `src` doc comments | 27 | 1 | 19 | 1 | |
| **total** | **524** | **145** | **510** | **146** | 12 |

Line by line, 3 of C's 524 have no site of the tool within two lines: two are in the block of
`book/src/reference/migrating-0.2.md:38` that does not tokenize (`=> …,` placeholders; reported in the notes of
the report) and one is the fourth alternative of a pattern that starts three lines above
(`fuzz/fuzz_targets/fuzz_evalf.rs:744`). The binding patterns agree corpus by corpus (130, 10, 1) except in `examples`, where the tool
lists 5 against C's 4: one that C's classification has as a construction.

108 marker lines `// codemod(errors): was ComputationFailed -> Undetermined by default; …` are written, one per
rewritten pattern on one of the two catch-alls (C: 150 test patterns on them, counting the binding ones too).

Pairs (C counts every call of the name by regular expression; user side):

| | C | tool: found | tool: rewrite | why they differ |
|---|---:|---:|---:|---|
| `try_` calls, classes b, cV (not `Distribution`), cP | 217 (201 by the name pattern on tests, examples, book, README, `src` docs today) | 200 calls + 16 `use` items | 163 calls + 16 `use` items | 4 of the 201 are Markdown prose or a twin the table does not have; 37 listed: 35 where the `Result` is not read at the call (bound and passed on), 1 `OrderStatistic::try_new` (no pair in C's table), 1 in a migration guide |
| plain calls of those functions | 1,762 | 859 | 857 | 642 of C's are calls of a method name other types share (`.add(`, `.mul(`, `.pow(`, `.call(`, `.gcd(`, `.coefficients(`): never attributed by name. Of `Poly::new(` (209 by pattern) only the 25 `.is_none()` / `.is_some()` readers change: the rest is `.unwrap()` and compiles unchanged. Not examined further |
| `try_` / plain calls of class a, cT, cI, `Distribution`, `try_symbol`, `try_replace` | 670 / 1,483 | — | — | class a stays a pair; cT is `truth`; cI and `Distribution` are `domains`; `try_symbol` is `core-renames` |
| doc mentions of a variant | 825 | 772 | 772 | the tool takes only the link form ``[`SymplexError::V`]``; the other 53 are prose |

`errors.construct` (C item 5) finds nothing in a default run: it is for the library's own code and runs with
`--src-code`. Its table (`tables/error_kinds.tsv`, column `ctor`) has constructors for six variants only — the
design names no more — so E1 must fill it in.

**Needs:** `tables/error_kinds.tsv`; `tables/try_pairs.tsv` regenerated from C's `try_pairs_census.py` on the
0.60 listing (`StateSpace::try_poles` is in it already). `ErrorKind` is not in the prelude (Part 4, C-12): the
tool adds `use symplex::errors::ErrorKind;` per scope (in doctests as a hidden line where the block hides its
imports), or writes the full path where another `ErrorKind` (`std::io`) is imported.

## 4. `numbers` — against review D (`review_D_numbers/data/num_usage.txt`; D's counts are of text)

| D's figure (tests, examples, benches, book, README, fuzz) | D | tool | note |
|---|---:|---:|---|
| files that name a num type or `Q` | 227 | 136 with a site | D's pattern also counts every file that names `Complex64` (no edit under D's rule; an import under A's, rule `paths.prelude-names`) and every file with a `Q` of its own. The 91 other files were not examined one by one |
| `Ratio::new` / `Q::new` / `BigRational::new` | 88 | 88 listed (+ 4 `new_raw`) | |
| `BigInt::parse_bytes` | 7 | 7 | |
| `.to_integer()` | 4 | 5 | all five in `tests`; the name is listed in any file that names a number type |
| `modpow`, `extended_gcd`, `magnitude`, `to_biguint` | 8 | 9 | |
| `as_ratio_parts` | 4 | 3 | D's fourth was not looked for |
| `Q` as a type name | about 662 mentions in 84 files | 398 tokens rewritten | D counted `\bQ\b` on every line: comments, `use` lines, prose, and the files where `Q` is the file's own name (a generic parameter, `let Q = qr.q`, or the file's own `type Q = Ratio<BigInt>;`, where the name stays and only `Q::from_integer` changes) |
| `use` items of the num crates | 92 + 54 + 53 + 15 + 6 files | 247 items rewritten | |

All 890 `BigInt` and 685 of 687 `Ratio<BigInt>` / `BigRational` / `Q` sites on the user side are rewritable; the
two listed are `Ratio` in `book/src/guide/matrices.md`, which also uses `Ratio<i64>`. With `--src-code` the same rules run on the
library (N2); that run was not part of this dry run's table (it is the compiler's job there, D §3).

**Needs:** nothing from a table. `symplex::num::{Integer, Rational, Complex64}` must exist (N1). The tool writes
`use symplex::num::{Integer, Rational};` in place of the removed imports (Part 4, C-12).

## 5. `core-renames` and `subs` — against review E (`review_E_core/data/renames.tsv`)

Per rule, E's kind of count (the regular expression `.old(` / `::old(` on the lines of tests, examples, book,
README and `src` doc comments) and the tool's sites: `out/compare_all.txt`, section "E core renames". Totals over
the 53 rows the tool has: regex 2,039, tool 1,967 found (1,601 rewritable, 366 listed). E's 2,106 also has rows
the tool does not carry (`free_symbols` 131, which compiles unchanged; `replace` closures 35; `sym!` 15).

| | E | tool: found | tool: rewrite | the difference |
|---|---:|---:|---:|---|
| `subs_i64` | 708 | 700 | 700 | the 8 others: 4 string literals (`.expect("df.subs_i64(&x, pt_val)…")`), 3 comments, 1 Markdown table |
| `subs_map_i64`, `subs_map_with` | 3, 10 | 2, 10 | 12 | |
| the five `force` flags | 63 | 59 | 59 | every flag is a literal, as E found (17 `true`, 42 `false`) |
| `solve_gt` … `try_solve_le` | 166 | 161 | 161 | |
| `Ex::symbolic_sum`, `symbolic_product` | 71 | 71 | 71 | the receiver moves in front (`Ex::symbolic_sum(&b, &k, ..)` → `b.formal_summation(&k, ..)`) |
| `log_combine`, `trig_combine` | 46 | 45 | 41 | 4 listed: files that use unit quantities, whose copies keep the name (F:U-MIRROR@1.x) |
| `expr_type()`, `ExprType`, `solve_for`, `solve_numeric`, `assume`, `args`, `rewrite_traced` | 84 + 98 + 52 + 23 + 52 + 42 + 6 | 82 + 95 + 50 + 21 + 49 + 58 + 6 | 0 | listed only, as E says ("a person") |

**Needs:** `tables/core_renames.tsv` (the rule table itself) and `tables/homonyms.tsv`, regenerated from the
listing of the day: it says which other public types have a method of an old name. NM1 runs last in block 8.

## 6. `paths` — against review A (`path_uses.py`, `retired_paths.py`)

A's two counts are in units the tool does not use, so both were re-derived:

* **Prelude: A's 417 imports in 247 test files reproduced exactly** by re-running A's scan
  (`tools/compare_A_prelude.py`: a dropped name appears bare outside `use` lines and comments). Of the 417
  (file, name) pairs **the tool adds an import for 235** (in 163 files) and for none that A does not have;
  45 are `Q` (rule set `numbers` renames it); **112 need nothing, because the file already imports the name
  on a `use` line** (A's scan did not look there; rule `paths.use` moves that line); the last 25 are of the same
  kind — nine that were read are names on the continuation lines of a multi-line `use symplex::…::{ … }` item —
  or a word in a string (`"Step 1: …"`).
* **Written paths: A's 743 + 456** count every occurrence of `symplex::<top>` and every name in a `use symplex::{…}`
  group whose top-level module moves in A's table. The tool counts a written path and a `use` *item* as one site
  each, and uses the **merged** table (`api_fate.tsv`): outside `src` 435 written paths (321 rewritable) and
  453 rewritable `use` items holding 977 moved paths; in `src` doc comments 31 + 523 items (694 paths). The
  tool's total is larger than A's because Part 4 moved units, polytope and ten statistics modules under
  `experimental` after A counted (`symplex::stats` alone is written 275 times in tests).

437 sites are listed: 193 name an item that is removed, 135 one that is renamed (`LpSolution`, `CodegenOptions`,
`Q`, …), 66 import a function that becomes an associated function (`matrix::jacobian`), 28 are globs of a module
whose items go to more than one 1.0 module (`symplex::linprog::*`), 8 are UI tests with recorded compiler
output, 2 import by name a module whose own name changes (`use symplex::matrix;`), 5 others.

**Needs:** `tables/path_moves.tsv` and `tables/prelude_moves.tsv`, both regenerated from the `api_fate.tsv` of
the day (WP-00 regenerates the listing from 0.60; `make_fate.py` is re-run).

## 7. `domains` — against review F (`review_F_domains/scripts/callsite_patterns.tsv`, re-run on the same lines)

| F's pattern | regex | tool: found | tool: rewrite | the difference (`out/compare_regex.txt`) |
|---|---:|---:|---:|---|
| `Distribution::<family>(` and `try_<family>(` | 937 | 930 | 861 | 7 regex-only: prose in Markdown and doc text; 69 listed: `finite` (the record has private fields) and `die` (not a record) — 39 and 33 over all roots |
| `.get(i, j)` | 543 | 539 | 434 | 5 regex-only: `RatingTable::get` compared with `Some(..)` / `None`; 1 tool-only: `.get(w[0], w[1])`; 105 listed: 82 in three files that also have another type with a two-argument `get` (75 of them in `tests/legacy/round2_domains.rs`), 24 in files with no sign of a matrix (all roots: 106) |
| `.p_value_f64/ln/log10/decimal/ex(` | 385 | 388 | 381 | 3 tool-only: `<Mauchly as PValue>::p_value_ex(m)`; 7 listed (`p_value_ex`) |
| `.density(` | 191 | 189 | 187 | 2 regex-only: comments |
| `eigenvals`, `eigenvals_with_multiplicity`, `eigenvects` | 124 | 113 | 109 | 13 regex-only: SymPy code in comments; 4 listed: the migration guide of 0.2 |

The other rows are in `out/compare_all.txt` ("F domain renames"): regex 1,849, tool 1,610 found. The large
single differences there are that script's generic pattern, not F's: with F's own pattern `ppf` is 142 against
111 (the 31 others are doc-comment prose in tests such as ``/// `t::ppf(1e-200, 1)`: …``), and `downcast_ref::<`
is 53 against 52.

**Needs:** `tables/dist_ctors.tsv` — regenerated from the source of the day by `tables/build_tables.py`, which
reads each constructor's body (22 of 24 fill a record from their parameters one to one); `tables/domain_renames.tsv`.

## 8. What in the reviewers' migration sketches did not turn out to be scriptable

* **B.** The 107 user-side reads (and 21 in `src` docs) of queries whose fate Part 4 changed. `if let`, `.unwrap()`,
  values bound to variables: as B says. B's M2 (`Option` → `Result`) and M3 (`.into_found()`) are not part of
  WP-05's row and are not built; M3 is a one-shape rule on this core, M2 needs the same "who reads it" evidence
  as the pairs below.
* **C item 3** (append `.unwrap()` to the plain call) for **methods whose name other types share**: 642 of C's
  1,762 calls. A text tool cannot tell `MultiPoly::add` from `Add::add`; the compiler lists them.
* **C items 2 + 3 together are not idempotent as sketched**: after `try_fibonacci(n)` → `fibonacci(n)` nothing
  says the call was never the old plain one, and a second run would append `.unwrap()` to it. The tool renames
  a `try_` call only where the `Result` reading is at the call or on the bound variable (163 of 200) and lists
  the other 37. The same holds for `try_get` → `get` beside `get(i, j)` → `[(i, j)]`.
* **C item 5**: constructor names for nine of the fifteen variants are not in the design.
* **D item 3** (delete the trait imports) is scripted as D wrote it, but a file that calls a `num_traits` method
  on a *primitive* (`n.to_f64()` on an `i64`) loses the trait it needs; the compiler says so, the text cannot.
* **E §3.1** (the compiler-guided pass for `Symbol`: read `cargo check --message-format=json`, append
  `.into_ex()`) needs a build machine and is not a text rewrite: not built. `expr_type`, `solve_for`,
  `solve_numeric`, `assume`, `args`: listed.
* **A §3**: the function-to-method rewrites (`cross(&a, &b)` → `a.cross(&b)`, `parse::parse(&ctx, s)` →
  `ctx.parse(s)`, `ctx.solve_system` → `linsolve`, `normalforms::f(&m)` → `m.f()`, `p.to_multipoly()`) are not
  built; `paths` lists their sites ("is removed at 1.0 (see …)", "becomes the associated function"). A's
  "`use` groups by a small `syn` pass" is done by the tool's own `use`-tree reader.
* **F**: `Distribution::finite` and `die`; `.to_vec()` and `.downcast_ref` (std has both names); `kurtosis`;
  everything F itself marks "by hand" (`LpOutcome`, the `Poly` moves, the rank family). Plain `.row(i)` /
  `.col(j)` (they return `Option` at 1.0) are not even listed: the names are too common to attribute.

## 9. What was verified, and how

| check | command | result |
|---|---|---|
| unit tests: tokenizer, chains, views, imports, engine; 338 input/output cases covering each of the 131 rules | `python3 -m unittest discover -s tests` | 54 tests, OK |
| every case is a fixpoint (the tool run on its own output changes nothing) | same (in `test_rules.py`) | OK |
| nothing fires inside a string, raw string, byte string, line or nested block comment, for all seven rule sets | same (`test_engine.py`) | OK |
| idempotence on copies of 8 real files: each rule set applied once and twice, and all seven in block order | `python3 tools/sandbox.py` | byte-identical, second run 0 rewrites (8 of 8 files changed) |
| whole tree, in memory: every changed file is a fixpoint; every changed Rust file and block still tokenizes with balanced brackets | `python3 tools/fixpoint_check.py` | 0 / 0 for all seven rule sets (1,483 file changes over the seven sets) |
| the repository is untouched | `git status --short` | empty |

Not verified: that the output compiles, and that it means the same. Three bugs found while reading the diffs are
recorded because they show what kind of error to expect: a hand-written prefilter (`\bis_`) silently skipped
files whose only query was `poly_is_positive_on` (now built from the table, with a test); the binding introduced
by `Err(e) if e.kind() == …` captured an `e` that the arm printed with `"{e}"` (the tool now also reads format
strings for the name, with a test); a module-prefix fallback rewrote `symplex::matrix::<unknown>` to
`symplex::linalg::<unknown>` (now listed). The first was found by comparing with B's site list, the second by a
unit test written for the hazard, the third by the sandbox run of all rule sets in sequence.
