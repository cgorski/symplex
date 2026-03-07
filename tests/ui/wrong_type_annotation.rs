use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let m = Mass::symbol("m");
    let a = Acceleration::symbol("a");
    let _bad: Velocity = symplex::dim!(Force: m * a);
}
