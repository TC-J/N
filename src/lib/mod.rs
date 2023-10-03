use pest::{iterators::Pair, Parser};
use pest_derive::Parser;
use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::str::FromStr;

#[derive(Parser)]
#[grammar = "n.pest"]
pub struct N;

#[derive(Debug)]
pub struct Tensor<T: Add + Mul + FromStr> {
    rank: usize,
    shape: Vec<usize>,
    dimensions: usize,
    value: Vec<T>,
}

pub fn parse_tensor<T: Add + Mul + FromStr + Debug>(s: &'static str) -> Tensor<T>
where
    <T as FromStr>::Err: Debug,
{
    let mut n = N::parse(Rule::tensor, s).unwrap();
    let mut shape: Vec<usize> = Vec::new();
    let mut value: Vec<T> = Vec::new();

    fn parse_value<U: Add + Mul + std::str::FromStr + Debug>(
        pair: Pair<Rule>,
        depth: isize,
        shape: &mut Vec<usize>,
        value: &mut Vec<U>,
    ) where
        <U as FromStr>::Err: Debug,
    {
        match pair.as_rule() {
            Rule::tensor => {
                println!("depth: {}", depth);
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

            Rule::scalar => {
                value.push(pair.as_str().parse().unwrap());
            }

            _ => {}
        }
    }
    parse_value(n.next().unwrap(), 0, &mut shape, &mut value);
    let dimensions;

    if shape.len() != 0 {
        for i in 0..shape.len() {
            for j in (0..i).rev() {
                shape[i] = shape[i] / shape[j];
            }
        }
        dimensions = value.len() / shape.last().unwrap();
    } else {
        dimensions = value.len();
    }

    shape.push(dimensions);

    let rank = shape.len() - 1;

    Tensor {
        rank,
        shape,
        dimensions,
        value,
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_parse_tensor_scalar() {
        let t = parse_tensor::<i32>("[123]");
        assert_eq!(t.dimensions, 1);
        assert_eq!(t.shape, [1]);
        assert_eq!(t.rank, 0);
        assert_eq!(t.value.len(), 1);
        assert_eq!(t.value[0], 123);
    }

    #[test]
    fn test_parse_tensor_higher_rank() {
        let t = parse_tensor::<f64>("[[1,2,3], [3,4,5], [6,7,8]]");
        assert_eq!(t.dimensions, 3);
        assert_eq!(t.shape, [3, 3]);
        assert_eq!(t.rank, 1);
    }
}
