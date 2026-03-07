use symplex::prelude::*;
use symplex::units::*;

fn takes_ex(_e: &Ex) {}

fn main() {
    let f = Force::symbol("f");
    // AsRef<Ex> should NOT auto-coerce &Force to &Ex
    takes_ex(&f);
}
