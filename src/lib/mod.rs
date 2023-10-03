use pest::{iterators::Pair, Parser};
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "n.pest"]
pub struct N;

#[derive(Debug)]
pub struct Tensor {
    rank: usize,
    shape: Vec<usize>,
    dimensions: usize,
    value: Vec<String>,
}

pub fn parse_tensor(s: &'static str) -> Tensor {
    let mut n = N::parse(Rule::tensor, s).unwrap();
    let mut shape: Vec<usize> = Vec::new();
    let mut value: Vec<String> = Vec::new();
    println!("{:?}", n);

    fn parse_value(
        pair: Pair<Rule>,
        depth: isize,
        shape: &mut Vec<usize>,
        value: &mut Vec<String>,
    ) {
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
                println!("A");
                value.push(pair.as_str().to_string());
                println!("B");
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

    println!("shape-pre: {:?}", shape);
    let rank = shape.len() - 1;
    println!("rank: {}", rank);
    println!("shape: {:?}", shape);
    println!("value: {:?}", value);

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
        let t = parse_tensor("[123]");
        assert_eq!(t.dimensions, 1);
        assert_eq!(t.shape, [1]);
        assert_eq!(t.rank, 0);
        assert_eq!(t.value.len(), 1);
    }

    #[test]
    fn test_parse_tensor_higher_rank() {
        let t = parse_tensor("[[1,2,3], [3,4,5], [6,7,8]]");
        assert_eq!(t.dimensions, 3);
        assert_eq!(t.shape, [3, 3]);
        assert_eq!(t.rank, 1);
    }
}
