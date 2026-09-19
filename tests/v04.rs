//! Feature tests written alongside the 0.4–0.6 APIs (`tests/v04/*.rs`, one
//! module per feature). Run one module with e.g.
//! `cargo test --test v04 v04_polyhedron::`.

#[path = "v04/v04_polyhedron.rs"]
mod v04_polyhedron;
#[path = "v04/v04_polytope.rs"]
mod v04_polytope;
#[path = "v04/v04_sos.rs"]
mod v04_sos;
