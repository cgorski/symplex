//! Convenience macros for creating symbolic variables.
//!
//! These macros reduce boilerplate when declaring multiple symbols or
//! symbols with assumptions.

/// Declare multiple symbolic variables at once.
///
/// Each identifier becomes a `let` binding of type [`Ex`](crate::expr::Ex)
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

/// Declare a symbol with mathematical assumptions.
///
/// The first argument is the context, the second is the symbol name,
/// and any additional identifiers are assumption variants applied to
/// the symbol.
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
/// ```
#[macro_export]
macro_rules! sym {
    ($ctx:expr; $name:ident $(, $prop:ident)*) => {
        let $name = $ctx.symbol_with(stringify!($name), &[
            $( $crate::assumptions::Assumption::$prop, )*
        ]);
    };
}

/// Declare multiple symbolic variables using the global default context.
///
/// This is a convenience version of [`syms!`] that doesn't require
/// passing a `Context` — it uses [`default_context()`](crate::default_context).
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::vars;
///
/// vars!(x, y, z);
/// let expr = &x + &y + &z;
/// assert_eq!(format!("{expr}"), "x + y + z");
/// ```
#[macro_export]
macro_rules! vars {
    ($($name:ident),+ $(,)?) => {
        $(
            let $name = $crate::var(stringify!($name));
        )+
    };
}
