//! Compile-time dimensional analysis for physical quantities.
//!
//! This module provides type-safe physical quantity tracking using phantom types
//! and typenum type-level integers. Named newtypes like [`Force`], [`Voltage`],
//! and [`Energy`] produce clear compiler error messages ("expected Force, found Mass")
//! while the generic [`Qty`] type handles exotic dimension combinations.
//!
//! # Architecture
//!
//! - **Named newtypes** (30 types): `Force`, `Voltage`, `Energy`, etc. — best error messages
//! - **Generic `Qty<D>`**: Fallback for intermediate/exotic dimensions — uses typenum Dim
//! - **Blanket `Mul`/`Div`**: Any `Qty<D1> * Qty<D2>` computes output dimension via typenum
//! - **Named `Mul`/`Div` table**: Specific pairs like `Mass × Acceleration → Force`
//! - **`assert_dim!`**: Compile-time checkpoint assertions
//! - **`const_assert_dim!`**: Compile-time formula verification with custom error messages

pub mod dim;
pub mod qty;
pub mod si;
pub mod mul_table;
pub mod conversions;
pub mod assert_macros;
pub mod calculus;

// Re-export all public items for `use symplex::units::*`
pub use dim::*;
pub use qty::*;
pub use si::*;
pub use assert_macros::*;
// mul_table and conversions add impls, no new public types to re-export
// calculus re-exports its own items
pub use calculus::*;
