use crate::{fundamental::Field, tensor::Tensor};

use pest::{iterators::Pair, Parser};
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "n.pest"]
pub struct N;

pub(crate) fn parse_tensor<T: Field>(s: &str) -> Tensor<T> {
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
