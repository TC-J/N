use crate::{fundamental::Field, tensor::Tensor};

use pest::{iterators::Pair, Parser};
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "n.pest"]
pub struct N;

pub fn parse_tensor<T: Field>(s: &str) -> Tensor<T> {
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
    let mut dimensions;

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

    let mut rank = shape.len();

    if dimensions == 1 || (rank == 2 && shape[0] == 1) {
        rank -= 1;
    }

    let shape_len = shape.len();

    if shape.len() > 1 {
        if shape[shape_len - 1] == 1 || shape[shape_len - 2] == 1 {
            let row = shape[shape_len - 1];
            let col = shape[shape_len - 2];
            dimensions = std::cmp::max(row, col);
        }
    }

    Tensor {
        rank,
        shape,
        dimensions,
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_tensor;

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
    fn test_parse_row_vector() {
        let v = parse_tensor::<f64>("[[1, 2, 3]]");
        assert_eq!(v.dimensions, 3);
        assert_eq!(v.shape, [1, 3]);
        assert_eq!(v.rank, 1);
        assert_eq!(v.value.len(), 3);
    }

    #[test]
    fn test_parse_column_vector() {
        let col_v = parse_tensor::<i64>("[[1], [2], [3]]");
        assert_eq!(col_v.dimensions, 3);
        assert_eq!(col_v.rank, 1);
        assert_eq!(col_v.shape, [3, 1]);
        assert_eq!(col_v.value.len(), 3);
    }

    #[test]
    fn test_parse_tensor_higher_rank() {
        let t = parse_tensor::<f64>("[[1,2,3], [3,4,5], [6,7,8]]");
        assert_eq!(t.dimensions, 3);
        assert_eq!(t.shape, [3, 3]);
        assert_eq!(t.rank, 2);
        assert_eq!(t.value, [1., 2., 3., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn test_tensor_with_several_indices() {
        let t = parse_tensor::<f32>(
            "[
            [
                [
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ],
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ]
                ],
                [
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ],
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ]
                ]
            ],
            [
                [
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ],
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ]
                ],
                [
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ],
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ]
                ]
            ]
        ]",
        ); // shape (2, 2, 2, 2, 4, 4)
        assert_eq!(t.rank, 6);
        assert_eq!(t.dimensions, 4);
        assert_eq!(t.shape, [2, 2, 2, 2, 4, 4]);
        assert_eq!(t.value.len(), 2 * 2 * 2 * 2 * 4 * 4);
    }
}
