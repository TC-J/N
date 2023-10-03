use n::{parse_tensor, Rule, Tensor, N};
use pest::{iterators::Pair, Parser};
use std::ops::{Add, Mul};

pub struct Field<T: Add + Mul> {
    pub inner: Vec<T>,
}

impl<T: Add + Mul> Field<T> {
    pub fn new() -> Self {
        Field { inner: Vec::new() }
    }
}

fn main() {
    println!("{:?}", parse_tensor("[[128, 323, 121], [801, 42, 10]]"));
}
