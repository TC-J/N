#![allow(dead_code)]
use pest::{iterators::Pair, Parser};
use pest_derive::Parser;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

#[derive(Parser)]
#[grammar = "n.pest"]
pub struct N;

pub trait Field: Add + Mul + Sub + Div + FromStr + Debug {}
impl<T> Field for T
where
    T: Add + Mul + Sub + Div + FromStr + Debug,
    <T as FromStr>::Err: Debug,
{
}

#[derive(Debug)]
pub struct Tensor<T: Field> {
    rank: usize,
    shape: Vec<usize>,
    dimensions: usize,
    value: Vec<T>,
}

impl<T: Field> FromStr for Tensor<T> {
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(parse_tensor(s))
    }

    type Err = String;
}

impl<T: Field> From<&str> for Tensor<T> {
    fn from(value: &str) -> Self {
        Self::from_str(value).unwrap()
    }
}

fn parse_tensor<T: Field>(s: &str) -> Tensor<T> {
    let mut n = N::parse(Rule::tensor, s).unwrap();
    let mut shape: Vec<usize> = Vec::new();
    let mut value: Vec<T> = Vec::new();

    fn parse_value<U: Field>(
        pair: Pair<Rule>,
        depth: isize,
        shape: &mut Vec<usize>,
        value: &mut Vec<U>,
    ) {
        match pair.as_rule() {
            Rule::tensor => {
                if depth > 0 {
                    if shape.get((depth as usize) - 1) == None {
                        shape.push(1);
                    } else {
                        shape[(depth as usize) - 1] += 1;
                    }
                }

                for p in pair.into_inner() {
                    parse_value(p, depth + 1, shape, value);
                }
            }

            Rule::scalar => unsafe {
                value.push(pair.as_str().parse().unwrap_unchecked());
            },

            _ => {}
        }
    }
    parse_value(n.next().unwrap(), 0, &mut shape, &mut value);
    let dimensions;

    if shape.len() != 0 {
        dimensions = value.len() / shape.last().unwrap();
        for i in 0..shape.len() {
            for j in (0..i).rev() {
                shape[i] = shape[i] / shape[j];
            }
        }
    } else {
        dimensions = value.len();
    }

    shape.push(dimensions);

    let rank = shape.len();

    Tensor {
        rank,
        shape,
        dimensions,
        value,
    }
}

#[macro_export]
macro_rules! tensor {
    ($v:expr) => {
        Tensor::<f64>::from(stringify!($v))
    };

    ($t:ty, $v:expr) => {
        Tensor::<$t>::from(stringify!($v))
    };
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_parse_tensor_scalar() {
        let t = Tensor::<i32>::from("[123]");
        assert_eq!(t.dimensions, 1);
        assert_eq!(t.shape, [1]);
        assert_eq!(t.rank, 0);
        assert_eq!(t.value.len(), 1);
        assert_eq!(t.value[0], 123);
    }

    #[test]
    fn test_parse_tensor_higher_rank() {
        let t = Tensor::<f64>::from("[[1,2,3], [3,4,5], [6,7,8]]");
        assert_eq!(t.dimensions, 3);
        assert_eq!(t.shape, [3, 3]);
        assert_eq!(t.rank, 1);
        assert_eq!(t.value, [1., 2., 3., 3., 4., 5., 6., 7., 8.]);
    }
}
