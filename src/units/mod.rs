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

/// Dimension vectors: phantom-typed compile-time SI dimension tracking.
pub mod dim;
/// Generic dimensioned quantity wrapper and arithmetic operators.
pub mod qty;
/// Named SI newtypes (e.g. `Force`, `Voltage`, `Energy`) with constructor helpers.
pub mod si;
/// Named multiplication and division rules between physical quantity types.
pub mod mul_table;
/// Unit-conversion helpers between SI prefixes and common non-SI units.
pub mod conversions;
/// Compile-time dimension-assertion macros (`assert_dim!`, `const_assert_dim!`).
pub mod assert_macros;
/// Dimensional calculus: differentiation and integration that track dimensions.
pub mod calculus;
/// Runtime dimension inference for expression trees.
pub mod inference;

// Re-export all public items for `use symplex::units::*`
pub use dim::*;
pub use qty::*;
pub use si::*;

// mul_table and conversions add impls, no new public types to re-export
// calculus re-exports its own items
pub use calculus::*;
pub use inference::*;
