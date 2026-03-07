use symplex::prelude::*;
use symplex::units::*;

fn main() {
    let l = Length::symbol("l");
    let _bad = l.sin();
}
