use symplex::prelude::*;
use symplex::units::*;

fn takes_ex(_e: &Ex) {}

fn main() {
    let ctx = Context::new();
    let f = Force::symbol(&ctx, "f");
    // AsRef<Ex> should NOT auto-coerce &Force to &Ex
    takes_ex(&f);
}
