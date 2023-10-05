use crate::fundamental::Field;
use crate::n_lang::parse_tensor;

use std::fmt::Debug;
use std::str::FromStr;

#[derive(Debug)]
pub struct Tensor<T: Field> {
    pub(crate) rank: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) dimensions: usize,
    pub(crate) value: Vec<T>,
}

impl<T: Field> FromStr for Tensor<T> {
    // FIXME: error handling
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s
            .replace(" ", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace("\r", "");
        Ok(parse_tensor(&*s))
    }

    type Err = String;
}

impl<T: Field> From<&str> for Tensor<T> {
    fn from(value: &str) -> Self {
        Self::from_str(value).unwrap()
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

pub use tensor;

#[cfg(test)]
mod tests {
    use super::{tensor, Tensor};

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
    fn test_parse_row_vector() {
        let v = tensor!([[1, 2, 3]]);
        assert_eq!(v.dimensions, 3);
        assert_eq!(v.shape, [1, 3]);
        assert_eq!(v.rank, 1);
        assert_eq!(v.value.len(), 3);
    }

    #[test]
    fn test_parse_column_vector_hint_is_excused_error() {
        let col_v = tensor!([[1], [2], [3]]);
        assert_eq!(col_v.dimensions, 1);
        assert_eq!(col_v.rank, 1);
        assert_eq!(col_v.shape, [3, 1]);
        assert_eq!(col_v.value.len(), 3);
    }

    #[test]
    fn test_parse_tensor_higher_rank() {
        let t = Tensor::<f64>::from("[[1,2,3], [3,4,5], [6,7,8]]");
        assert_eq!(t.dimensions, 3);
        assert_eq!(t.shape, [3, 3]);
        assert_eq!(t.rank, 2);
        assert_eq!(t.value, [1., 2., 3., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn test_tensor_with_several_indices() {
        let t = tensor!([
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
        ]); // shape (2, 2, 2, 2, 4, 4)
        assert_eq!(t.rank, 6);
        assert_eq!(t.dimensions, 4);
        assert_eq!(t.shape, [2, 2, 2, 2, 4, 4]);
        assert_eq!(t.value.len(), 2 * 2 * 2 * 2 * 4 * 4);
    }
}
