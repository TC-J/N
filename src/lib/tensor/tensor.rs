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
        Ok(parse_tensor(s))
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
