//! Convenience macros for creating symbolic variables, plus the public
//! rewrite-engine and simplification option types.
//!
//! These macros reduce boilerplate when declaring multiple symbols or
//! symbols with assumptions.
//!
//! The type re-exports at the bottom of this module ([`Rule`],
//! [`RuleSet`], [`Bindings`], [`RewriteOpts`], [`RewriteStrategy`],
//! [`Step`], [`ExpandOpts`]) make the rewrite-rule engine reachable from
//! a stable public path; the prelude re-exports them as well.

// ── Rewrite-engine / simplification option types ───────────────────────

pub use crate::api::expr_rules_ext::{Bindings, RewriteOpts, RewriteStrategy, Rule, RuleSet, Step};
pub use crate::transforms::expand::{EXPAND_TERM_LIMIT, ExpandOpts};

/// Declare multiple symbolic variables at once.
///
/// Each identifier becomes a `let` binding of type [`Ex`](crate::api::expr::Ex)
/// in the current scope.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::syms;
///
/// let ctx = Context::new();
/// syms!(ctx; x, y, z);
/// let expr = &x + &y + &z;
/// assert_eq!(format!("{expr}"), "x + y + z");
/// ```
#[macro_export]
macro_rules! syms {
    ($ctx:expr; $($name:ident),+ $(,)?) => {
        $(
            let $name = $ctx.symbol(stringify!($name));
        )+
    };
}

/// Declare multiple symbolic variables at once (alias of [`syms!`]).
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vars;
///
/// let ctx = Context::new();
/// vars!(ctx; a, b);
/// assert_eq!(format!("{}", &a * &b), "a*b");
/// ```
#[macro_export]
macro_rules! vars {
    ($ctx:expr; $($name:ident),+ $(,)?) => {
        $(
            let $name = $ctx.symbol(stringify!($name));
        )+
    };
}

/// Declare a symbol with mathematical assumptions.
///
/// The first argument is the context, the second is the symbol name,
/// and any additional identifiers are assumption variants applied to
/// the symbol.
///
/// The declaration goes through [`Context::symbol_with`] and propagates
/// its error with `?` (contradictory assumptions such as
/// `sym!(ctx; t, Positive, Negative)`), so use the macro in a function
/// returning `Result<_, E>` with `E: From<SymplexError>`.
///
/// [`Context::symbol_with`]: crate::api::context::Context::symbol_with
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::sym;
///
/// let ctx = Context::new();
/// sym!(ctx; t, Positive, Real);
/// assert_eq!(ctx.query(&t, Props::POSITIVE), Some(true));
/// assert_eq!(ctx.query(&t, Props::REAL), Some(true));
/// // Inferred:
/// assert_eq!(ctx.query(&t, Props::COMPLEX), Some(true));
/// # Ok::<(), SymplexError>(())
/// ```
#[macro_export]
macro_rules! sym {
    ($ctx:expr; $name:ident $(, $prop:ident)*) => {
        let $name = $ctx.symbol_with(stringify!($name), &[
            $( $crate::base::assumptions::Assumption::$prop, )*
        ])?;
    };
}
