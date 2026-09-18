//! Feature tests written alongside the 0.3 API (`tests/v03/*.rs`, one module per
//! former `tests/v03_*.rs` file). Run one module with e.g.
//! `cargo test --test v03 v03_linprog::`.

#[path = "v03/v03_linprog.rs"]
mod v03_linprog;
#[path = "v03/v03_matrix_ergonomics.rs"]
mod v03_matrix_ergonomics;
#[path = "v03/v03_normalforms.rs"]
mod v03_normalforms;
#[path = "v03/v03_optimize.rs"]
mod v03_optimize;
#[path = "v03/v03_poly_symbolic_coeffs.rs"]
mod v03_poly_symbolic_coeffs;
#[path = "v03/v03_poly_view.rs"]
mod v03_poly_view;
#[path = "v03/v03_ratsimp.rs"]
mod v03_ratsimp;
#[path = "v03/v03_user_notes.rs"]
mod v03_user_notes;
