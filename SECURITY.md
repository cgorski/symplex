# Security policy

## Reporting

Report a vulnerability privately through GitHub: **Security → Report a
vulnerability** on <https://github.com/cgorski/symplex> (a private security
advisory).  Please do not open a public issue for a problem that lets
untrusted input crash, hang or exhaust the memory of a process.  You will get
an answer within a week; a fix is released as a patch version of the newest
minor release, and the advisory is published with it.

Only the newest release is supported.

## What counts

symplex has no `unsafe` code (`#![forbid(unsafe_code)]` in all four crates;
test builds of the main crate are exempt, for the test suite's CPU clock),
and the library crate opens no files, sockets or processes (`symplex-build`
writes the generated source file a build script asks for).  The realistic
risks are **denial of service** and **wrong answers**:

- **Untrusted input.**  These entry points are meant to take text or data from
  outside the program, and a panic, an abort, a stack overflow, unbounded
  memory growth or a call that does not return on *any* input to them is a
  security bug:
  `Context::parse` and the other parsers, `Context::from_json` /
  `from_json_any` / `from_tree`, `decimal_str`, and the `Session` of
  `symplex-wasm`.
- **Everything else** (calculus, solving, simplification, statistics on
  expressions you built from untrusted text) is *computation on* untrusted
  input.  The library bounds its own work (`Budget`, size caps with `try_`
  variants that report a refusal), and a hang or an abort there is a bug we
  fix — report it as an ordinary issue unless it is trivially reachable from
  the parser with a short input.
- **A silently wrong answer** is the most serious kind of bug in this project,
  but it is not a vulnerability: open a normal issue with the input, the
  answer you got and the oracle behind the expected one.

## Running symplex on untrusted input

- Run each request under a time limit and a memory limit of your own (a
  worker process or thread you can abandon).  The library's budgets bound the
  common paths, not every path; the known slow inputs are listed in
  `CHANGELOG.md` under *Measured* and in the roadmap.
- Prefer the `try_` variants: they return an error where the plain function
  returns an unevaluated form or a documented empty result.
- Bound the size of what you accept: the parser refuses decimal exponents
  beyond `±10000` and nesting deeper than 128 levels, but a megabyte of valid
  input is still a megabyte of work.
- Generated code (`compile`, the Rust/C emitters, `symplex-build`) is as
  trustworthy as the expression it came from: do not compile code generated
  from an expression an attacker wrote without the same care as any other
  generated code.
