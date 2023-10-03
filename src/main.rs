use n::{parse_tensor, Rule, Tensor, N};
use pest::{iterators::Pair, Parser};
use std::ops::{Add, Mul};

fn main() {
    println!(
        "{:?}",
        parse_tensor::<f64>("[[128, 323, 121], [801, 42, 10]]")
    );
}
