# The API snapshot

`api/*.txt` lists the public API of the crate — every module, type, field,
variant, function, method, trait, impl, constant, alias, macro and re-export a
program can name — one item per line, sorted. The files are generated from the
code and checked in, so a change to the public API is a change to a text file
that the reviewer of the pull request sees. CI (job **API snapshot**)
regenerates them and fails if they differ from what is checked in.

Today there is one file, `api/public.txt`. [Tiers](#tiers) says how it splits
into `stable.txt`, `experimental.txt` and `internal.txt`.

| file | what it is |
|---|---|
| `api_snapshot.sh` | builds rustdoc's JSON for the library and runs `api_snapshot.py` |
| `api_snapshot.py` | walks the JSON from the crate root and writes `api/<tier>.txt` |
| `tiers.txt` | which file an item goes to, by path prefix |
| `api_listing.py` | the flat tab-separated listing used by the 1.0 planning tables ([below](#the-flat-listing)) |

Python 3 (standard library only) and a nightly toolchain; nothing else.

## Regenerating

rustdoc emits JSON only on nightly, and the JSON's layout changes between
nightlies, so one dated nightly is pinned as `API_TOOLCHAIN` in
`.github/workflows/ci.yml`. Install it once (the name is in that file):

```sh
rustup toolchain install nightly-YYYY-MM-DD
```

Then, from anywhere in the repository:

```sh
scripts/api_snapshot/api_snapshot.sh            # rewrite api/*.txt
scripts/api_snapshot/api_snapshot.sh --check    # what CI runs: change nothing, exit 1 and print the diff
```

Commit the changed files under `api/` in the same commit as the code change.
The script reads the pin from `ci.yml`; `API_TOOLCHAIN=nightly scripts/...`
uses another toolchain (the script stops if that toolchain's JSON format is
not one it knows). Two runs on the same tree give identical files, whatever
the state of the build directory; and since no public item of the crate is
behind a platform `cfg`, it does not matter where they are generated.

## The lines

```text
mod      symplex::ntheory
struct   symplex::Budget  #[non_exhaustive]
field    symplex::Budget::deadline: Option<Instant>
struct   symplex::expr::Expr<S: Sort>  [private fields]
struct   symplex::SplitMix64(_)
enum     symplex::BudgetHit  #[non_exhaustive]
variant  symplex::BudgetHit::Deadline
trait    symplex::eq::Scalar: symplex::api::expr_ops::private::Sealed + Clone  [sealed]
fn       symplex::ntheory::isprime(n: impl Into<BigInt>) -> bool
method   symplex::expr::Expr<Numeric>::diff(&self, var: &Ex) -> Ex  #[must_use]
method   <T: PartialOrd> symplex::Bounds<T>::contains(&self, x: &T) -> bool
method   symplex::units::DimName::dim_name() -> &'static str  [required]
const    symplex::config::EXPRESSION_BUDGET: usize
type     symplex::expr::Ex = Expr<Numeric>
macro    symplex::syms! ($ctx:expr;$($name:ident),+$(,)?)=>{...};
use      symplex::prelude::Matrix = symplex::matrix::Matrix
use      symplex::expr! = ::symplex_macros::expr!  [extern macro]
impl     Mul<&Matrix> for &symplex::expr::Ex { type Output = Matrix }
impl     Mul<Expr<Numeric>> for i64 { type Output = Expr<Numeric> }
impl     <T: Clone> Clone for symplex::Bounds<T>
impl     !UnwindSafe for symplex::lambdify::CompiledFn
impls    symplex::Budget: Clone + Debug + Default + Eq + PartialEq + RefUnwindSafe + Send + Sync + Unpin + UnwindSafe
```

A line is a kind, the item with its **canonical path** and full signature, and
after two spaces its markers.

- **Canonical path.** The crate is walked from its root through every `pub mod`
  and `pub use`, so paths are the ones a user writes, not the files the code
  lives in. An item reachable under several paths is listed under the shortest
  (a path through `prelude` last, then alphabetically); each other path is a
  `use` line, `second path = canonical path`. When a whole module has a second
  path only the module gets a `use` line, not every item in it. A re-export of
  another crate's item reads `= ::crate::path  [extern <kind>]`.
- **Types** are followed by their parts. A `struct` line says `[unit]`, shows
  the fields of a tuple struct in place (`_` is a private field), or is followed
  by one `field` line per public field; `[private fields]` means there are
  others, so the struct cannot be built or matched exhaustively outside the
  crate. An `enum` is followed by one `variant` line per variant with its
  payload. `#[non_exhaustive]` is on the line of the type or variant that
  carries it.
- **Methods.** `method` lines are inherent methods and the methods a trait
  declares (`[required]` or `[provided]`). A leading `<...>` is the generic
  parameter list of the `impl` block the method is written in, and the type
  after it is that block's type (`Expr<Numeric>::diff` is not available on
  `Expr<Boolean>`). `const` and `type` lines under a type or trait are its
  associated constants and types.
- **Traits.** The line carries the supertraits and `[unsafe]`, `[auto]`,
  `[dyn-compatible]`, `[sealed]`. `[sealed]` is printed when a supertrait has
  no public path, the one way of sealing the script can see; a trait without
  the marker may still be sealed by other means.
- **Impls.** Every implementation of a trait is recorded, but not the methods
  it has to provide (the trait says what they are). An implementation with
  something to say — generic parameters, bounds, trait arguments, associated
  types, a `!` — has an `impl` line of its own, sorted with the type it is
  for (an impl for a foreign type, `Mul<Expr<Numeric>> for i64`, with the first
  type of this crate it names). Plain implementations of traits from other crates (`Clone`, `Debug`,
  `Send`, ...) are collected into the one `impls` line of the type. Of the auto
  traits `Send`, `Sync`, `Unpin`, `UnwindSafe` and `RefUnwindSafe` are recorded
  for every type: a missing one shows as `impl !Trait for T`.
- **Markers.** `#[must_use]`, `#[non_exhaustive]`, `#[deprecated]`,
  `#[doc(hidden)]` are the attributes; `[const]`, `[async]`, `[unsafe]`,
  `[extern "C"]` qualify a function.
- **Names inside signatures** are the names of the items referred to (found
  through rustdoc's ids, not copied from the source), so `crate::api::expr::Ex`,
  `super::Ex` and `Ex` all print `Ex`. When two different items of the same
  name are mentioned anywhere in the snapshot, both are written with their
  full path everywhere (`symplex::stats::anova::Observation`); `Result`,
  `Option`, `Vec`, `String` and `Box` alone always mean the standard ones. A
  full path starting with a private module (`symplex::api::...`) is an item
  that has no public path at all. Type aliases are kept as the source wrote
  them (`Ex`, not `Expr<Numeric>`).
- **Order.** Lines are sorted by path, so a module is followed by what it
  contains and a type by its fields or variants, associated items, methods (by
  name) and impls.

`#[doc(hidden)]` items are listed too (rustdoc is run with
`--document-hidden-items`): the compiler lets anyone call them, and the macros
expand to them.

## Reading a diff

Every `-` line is something a user of the old version could rely on; every `+`
line is something new. The questions to ask of a pull request:

| the diff shows | it means |
|---|---|
| only `+` lines for new items (`fn`, `method`, `impl`, `use`, `mod`, a type with its parts) | an addition; compatible |
| a `-` line with no `+` counterpart | something was removed or made private: breaking |
| a `-`/`+` pair of one `fn` or `method` | its signature changed: parameter types, return type, generics, bounds, `&self`/`&mut self`, a lost `[const]`: breaking unless the new one accepts everything the old one did. A changed parameter *name* also shows; it is harmless |
| a `-`/`+` pair differing only in the path | the item moved. Compatible only if a `use` line keeps the old path |
| `+field` on a struct without `[private fields]` and without `#[non_exhaustive]` | breaking: users can build that struct with a literal |
| `+variant` on an enum without `#[non_exhaustive]` | breaking: users can match it exhaustively |
| `+method ... [required]` on a trait without `[sealed]` | breaking for every implementor outside the crate |
| a trait dropped from an `impls` line, a `-impl`, or a new `impl !Send for ...` | breaking (this is how a lost `UnwindSafe` shows) |
| a marker changed (`#[non_exhaustive]` added, `[sealed]` added, `[dyn-compatible]` lost) | breaking in the direction that restricts the user |
| many lines changing one short name into a full path | a second item of that name is now mentioned somewhere; the lines mean the same as before |

The Semver job (`cargo semver-checks`) and this diff overlap only in part.
Measured on this crate (cargo-semver-checks 0.51.0, four hand-made changes to
`Budget`): it reports a removed method (`inherent_method_missing`, exit 100),
and reports nothing — exit 0 — for a changed parameter type, a changed return
type, and an added method. For signatures this diff is the only check, which
is why the lines carry them in full.

What the snapshot does not record: the values of constants; the bodies behind
`{...}` of a macro (its matchers are recorded); the order of fields and
variants; documentation; and anything about behaviour.

## Tiers

`tiers.txt` assigns every line to a file by path prefix. Today it has one
rule (`public *`), so everything is in `api/public.txt`. When the crate gets
its module facade (stable modules, `symplex::experimental`, and the
`#[doc(hidden)]` gate `symplex::__internal`), the four rules that are commented
out at the end of `tiers.txt` replace it:

```text
internal      doc(hidden)
internal      symplex::__internal
experimental  symplex::experimental
stable        *
```

Run `api_snapshot.sh` once: it writes `api/stable.txt`, `api/experimental.txt`
and `api/internal.txt` and removes `api/public.txt`; nothing else has to
change, and `--check` then also fails on a tier file that is missing or left
over. The longest matching prefix decides; `doc(hidden)` takes every item
that is hidden or reachable only through something hidden. An `impl` goes to
the least public tier among the items its header names (an implementation of
an experimental trait for a stable type is experimental), and an item with
paths in two tiers is listed in the more public one, its other path as a `use`
line in the other file. The split only distributes lines: the three files
together hold exactly the lines one file would. Try a rule set without
touching `api/`:

```sh
python3 scripts/api_snapshot/api_snapshot.py --tiers my_rules.txt --out /tmp/api_try
```

(after one run of `api_snapshot.sh`, which leaves the JSON in
`target/doc/symplex.json`).

## Moving the pinned nightly

The pin exists so that the job fails only when the API changes. To move it:

1. Set `API_TOOLCHAIN` in `.github/workflows/ci.yml` to the new dated nightly
   and install it.
2. Run `api_snapshot.sh`. If it stops with "format N is not one this script
   knows", read what changed between the two format versions in
   `src/rustdoc-json-types/lib.rs` of rust-lang/rust, adapt `api_snapshot.py`,
   and add N to `KNOWN_FORMATS`. Every JSON construct the script does not know
   is an error, not a guess.
3. The diff of `api/` must be empty. If lines changed although the code did
   not, the new rustdoc describes something differently (a new compiler-internal
   marker trait, say — see `IGNORED_CORE_TRAITS`): make the script print the old
   line again rather than committing the new one, so that the snapshot keeps
   meaning the same thing.
4. Commit the pin, the script and (if step 3 could not be met) the regenerated
   files together, saying why.

## The flat listing

`api_listing.py --out FILE` writes the older, tab-separated form of the same
walk: `kind, path, container, trait, signature, file, has_docs, flags`, one row
per item including one row per method of every trait implementation. The
tables that plan the 1.0 API (`api_fate.tsv`) are keyed on its rows, and it is
kept byte-compatible with the listing made for 0.59 so that two releases can be
compared with `diff`; its header comment lists how it differs from the
snapshot. It is a planning aid and is not checked in CI.
