//! SymPy oracle for the 0.3 API (`tests/v03_oracle/*.rs`, one module per former
//! `tests/v03_oracle_*.rs` file). The shared runner lives in
//! `tests/v02_oracle_common/mod.rs` and is included here once as `oracle_common`.

#[path = "v02_oracle_common/mod.rs"]
mod oracle_common;

#[path = "v03_oracle/v03_oracle_linprog.rs"]
mod v03_oracle_linprog;
#[path = "v03_oracle/v03_oracle_meta.rs"]
mod v03_oracle_meta;
#[path = "v03_oracle/v03_oracle_normalforms.rs"]
mod v03_oracle_normalforms;
#[path = "v03_oracle/v03_oracle_poly.rs"]
mod v03_oracle_poly;
