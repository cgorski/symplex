//! Convenience macros for creating symbolic variables.
//!
//! These macros reduce boilerplate when declaring multiple symbols or
//! symbols with assumptions (future stage).

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

/// Declare a symbol with assumptions (placeholder for Stage 4).
///
/// Currently creates a plain symbol — assumption support will be added
/// when the assumption engine is implemented.
///
/// # Examples
///
/// ```
/// use symplex::prelude::*;
/// use symplex::sym;
///
/// let ctx = Context::new();
/// sym!(ctx; t, positive, real);
/// // For now, `t` is a plain symbol. Assumptions will be applied in Stage 4.
/// assert_eq!(format!("{t}"), "t");
/// ```
#[macro_export]
macro_rules! sym {
    ($ctx:expr; $name:ident $(, $prop:ident)*) => {
        let $name = $ctx.symbol(stringify!($name));
        // TODO(stage4): apply assumptions [$( $prop ),*] to the symbol
    };
}
