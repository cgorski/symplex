use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let f = Force::symbol("f");
    let v = Voltage::symbol("v");
    let _bad = &f + &v;
}
