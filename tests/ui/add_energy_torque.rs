use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let e = Energy::symbol("e");
    let t = Torque::symbol("t");
    let _bad = &e + &t;
}
