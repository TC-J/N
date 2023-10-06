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

impl<T: Field> Tensor<T> {
    pub fn shape(&self) -> &Vec<usize> {
        &(self.shape)
    }
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

impl<T: Field> Tensor<T> {
    pub fn flat_index(&self, index: usize) -> T {
        self.value[index]
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

#[macro_export]
macro_rules! index {
    ($t:ident, $($i:tt)*) => {
        {
            let indices = [$( $i )*];
            let shape = $t.shape();
            let size_disp = shape.len() - indices.len();
            let mut index = 0;
            println!("indices {:?} shape {:?} lengths: i {} s {}", indices, shape, indices.len(), shape.len());
            for i in 0..indices.len() {
                print!("indices[{}] {}", i, indices[i]);
                let mut tmp = indices[i];
                for j in i..shape.len() {
                    print!(", shape[{}] {}", j, shape[j]);
                    if i != j {
                        tmp += tmp * (shape[j] - 1);
                    }
                }
                println!(" tmp {} new index {}", tmp, index + tmp);
                index += tmp;
            }

            $t.flat_index(index)
        }
    };
}
pub use index;

#[cfg(test)]
mod tests {
    use super::{index, tensor, Tensor};

    #[test]
    fn test_scalar_rank_0_tensor() {
        let t = tensor!(i32, [123]);
        assert_eq!(t.dimensions, 1);
        assert_eq!(t.shape, [1]);
        assert_eq!(t.rank, 0);
        assert_eq!(t.value.len(), 1);
        assert_eq!(t.value[0], 123);
    }

    #[test]
    fn test_row_vector_rank_1_tensor() {
        let v = tensor!([[1, 2, 3]]);
        assert_eq!(v.dimensions, 3);
        assert_eq!(v.shape, [1, 3]);
        assert_eq!(v.rank, 1);
        assert_eq!(v.value.len(), 3);
    }

    #[test]
    fn test_column_vector_rank_1_tensor() {
        let col_v = tensor!([[1], [2], [3], [4]]);
        assert_eq!(col_v.dimensions, 4);
        assert_eq!(col_v.rank, 1);
        assert_eq!(col_v.shape, [4, 1]);
        assert_eq!(col_v.value.len(), 4);
    }

    #[test]
    fn test_matrix_rank_2_tensor() {
        let t = Tensor::<f64>::from("[[1,2,3], [3,4,5], [6,7,8]]");
        assert_eq!(t.dimensions, 3);
        assert_eq!(t.shape, [3, 3]);
        assert_eq!(t.rank, 2);
        assert_eq!(t.value, [1., 2., 3., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn test_rank_6_tensor() {
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

    #[test]
    fn test_rank_4_indices() {
        let t = tensor!(
            i64,
            [
                [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12],]],
                [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24],]]
            ]
        );

        let shape = t.shape();
        assert_eq!(shape, &[2, 2, 2, 3]);

        let mut n = 1;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for l in 0..shape[3] {
                        let value = index!(t, i, j, k, l);
                        assert_eq!(value, n);
                        n += 1;
                    }
                }
            }
        }
    }
}
