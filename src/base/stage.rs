//! Stage tracing: where does a slow or memory-hungry computation spend its
//! time?
//!
//! [`stage!`] wraps one step of an algorithm (an integration strategy, a
//! Risch phase) in a `DEBUG` span named after the step.  While the span is
//! enabled, the stage
//!
//! - logs `enter` (`TRACE`) with the expression it works on, clipped to
//!   [`EXPR_CHARS`] characters (formatting stops at the limit, so a huge
//!   expression costs no more to log than a small one);
//! - logs `exit` (`TRACE`) with the elapsed time and the number of arena
//!   nodes created, or a `DEBUG` "hot stage" line instead once either
//!   passes [`HOT_MS`] or [`HOT_NODES`];
//! - gives every event inside it the chain of stages that led there
//!   (`integrate:risch_rational:log_to_real`).
//!
//! Every event uses the target `symplex::stage`:
//!
//! ```text
//! RUST_LOG=symplex::stage=debug   # the hot stages only: a coarse profile
//! RUST_LOG=symplex::stage=trace   # every stage entered and left
//! ```
//!
//! A computation that never finishes (killed by a timeout or a memory cap)
//! never logs its `exit`; the last `enter` lines name the stage it was in.
//!
//! **Cost.**  With no subscriber, or with the span filtered out, a stage
//! is one disabled-callsite check (the same as any `tracing` macro): the
//! clock, the node count and the expression text are only computed for an
//! enabled span.  A binary that wants even that gone enables `tracing`'s
//! `release_max_level_info` feature in its own `Cargo.toml` (`tracing = {
//! version = "0.1", features = ["release_max_level_info"] }`): stages are
//! `DEBUG`/`TRACE`, so its release builds compile every stage out, along
//! with symplex's other debug logging, and keep warnings and errors.

use std::fmt;
use std::time::{Duration, Instant};

use crate::base::arena::Arena;
use crate::base::node::ExprId;

/// The longest expression text an `enter` event shows.
pub(crate) const EXPR_CHARS: usize = 200;

/// A stage taking at least this long is logged as a hot stage.
pub(crate) const HOT_MS: u64 = 250;

/// A stage creating at least this many arena nodes is logged as a hot stage.
pub(crate) const HOT_NODES: usize = 200_000;

/// Run `$body` as the stage `$name` working on `$expr`; evaluates to the
/// value of `$body`.  `$arena` is the `Arena` (or `&mut Arena`) that
/// `$body` uses; it is only read, before and after `$body`.
macro_rules! stage {
    ($arena:ident, $name:literal, $expr:expr, $body:expr) => {{
        let span = ::tracing::debug_span!(target: "symplex::stage", $name);
        let guard = if span.is_disabled() {
            None
        } else {
            Some($crate::base::stage::Stage::enter(
                span, $name, &$arena, $expr,
            ))
        };
        let out = $body;
        if let Some(guard) = guard {
            guard.exit($arena.node_count());
        }
        out
    }};
}
pub(crate) use stage;

/// An entered stage: its span, and the clock and node count at entry.
pub(crate) struct Stage {
    _span: tracing::span::EnteredSpan,
    name: &'static str,
    start: Instant,
    nodes: usize,
}

impl Stage {
    pub(crate) fn enter(
        span: tracing::Span,
        name: &'static str,
        arena: &Arena,
        expr: ExprId,
    ) -> Self {
        let span = span.entered();
        let nodes = arena.node_count();
        tracing::trace!(
            target: "symplex::stage",
            nodes,
            expr = %Clipped(arena.display(expr)),
            "enter"
        );
        Stage {
            _span: span,
            name,
            start: Instant::now(),
            nodes,
        }
    }

    pub(crate) fn exit(self, nodes_now: usize) {
        let elapsed = self.start.elapsed();
        let new_nodes = nodes_now.saturating_sub(self.nodes);
        if elapsed >= Duration::from_millis(HOT_MS) || new_nodes >= HOT_NODES {
            tracing::debug!(
                target: "symplex::stage",
                stage = self.name,
                ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
                new_nodes,
                "hot stage"
            );
        } else {
            tracing::trace!(
                target: "symplex::stage",
                us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
                new_nodes,
                "exit"
            );
        }
    }
}

/// Displays at most [`EXPR_CHARS`] characters of the wrapped value, then
/// `…`.  Formatting is abandoned at the limit rather than run to the end.
struct Clipped<D>(D);

impl<D: fmt::Display> fmt::Display for Clipped<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Limit {
            buf: String,
            left: usize,
        }
        impl fmt::Write for Limit {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                for c in s.chars() {
                    if self.left == 0 {
                        return Err(fmt::Error);
                    }
                    self.buf.push(c);
                    self.left -= 1;
                }
                Ok(())
            }
        }
        let mut w = Limit {
            buf: String::new(),
            left: EXPR_CHARS,
        };
        let clipped = fmt::write(&mut w, format_args!("{}", self.0)).is_err();
        f.write_str(&w.buf)?;
        if clipped {
            f.write_str("…")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Clipped, EXPR_CHARS};
    use crate::base::arena::Arena;

    #[test]
    fn stage_evaluates_to_its_body() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let y = stage!(arena, "test_stage", x, arena.int(7));
        assert_eq!(arena.as_num(y).map(ToString::to_string), Some("7".into()));
    }

    #[test]
    fn clipped_stops_at_the_limit() {
        let long = "x".repeat(EXPR_CHARS + 50);
        let shown = Clipped(&long).to_string();
        assert_eq!(shown.chars().count(), EXPR_CHARS + 1);
        assert!(shown.ends_with('…'));
        assert_eq!(Clipped("short").to_string(), "short");
    }

    #[tracing_test::traced_test]
    #[test]
    fn enabled_stage_logs_enter_and_exit() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let y = stage!(arena, "probe_stage", x, arena.int(3));
        assert_eq!(arena.as_num(y).map(ToString::to_string), Some("3".into()));
        assert!(logs_contain("probe_stage"));
        assert!(logs_contain("enter"));
        assert!(logs_contain("expr=x"));
        assert!(logs_contain("new_nodes="));
    }
}
