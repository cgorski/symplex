---
name: Wrong answer, crash or hang
about: A result that is wrong, a panic, an abort, or a call that does not return
title: ''
labels: bug
---

**Input** (a complete snippet that runs)

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // …
}
```

**What symplex returns** (the value, the panic message, or how long it ran)

**What is right, and how you know** — the oracle call and its output, e.g.
`mpmath.quad(lambda t: …, [0, 1])` at `mp.dps = 30`, a SymPy expression with
its version, an exact identity, or a substitution that shows the answer does
not satisfy the equation.  A library disagreeing with symplex is a reason to
look, not yet a proof: an independent check settles it.

**Version** — `symplex = "…"`, `rustc --version`, target.
