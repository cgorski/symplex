//! SymPy oracle for the 0.2 API (`tests/v02_oracle/*.rs`, one module per former
//! `tests/v02_oracle_*.rs` file). The shared runner
//! (`tests/v02_oracle_common/mod.rs`) is included here once.

mod v02_oracle_common;

#[path = "v02_oracle/v02_oracle_calculus.rs"]
mod v02_oracle_calculus;
#[path = "v02_oracle/v02_oracle_codegen.rs"]
mod v02_oracle_codegen;
#[path = "v02_oracle/v02_oracle_matrix.rs"]
mod v02_oracle_matrix;
#[path = "v02_oracle/v02_oracle_meta.rs"]
mod v02_oracle_meta;
#[path = "v02_oracle/v02_oracle_new_features.rs"]
mod v02_oracle_new_features;
#[path = "v02_oracle/v02_oracle_ntheory.rs"]
mod v02_oracle_ntheory;
#[path = "v02_oracle/v02_oracle_poly.rs"]
mod v02_oracle_poly;
#[path = "v02_oracle/v02_oracle_sets_logic.rs"]
mod v02_oracle_sets_logic;
#[path = "v02_oracle/v02_oracle_simplify.rs"]
mod v02_oracle_simplify;
#[path = "v02_oracle/v02_oracle_solve.rs"]
mod v02_oracle_solve;
#[path = "v02_oracle/v02_oracle_transforms.rs"]
mod v02_oracle_transforms;
