use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

pub trait Field: Add + Mul + Sub + Div + FromStr + Debug + Copy {}
impl<T> Field for T
where
    T: Add + Mul + Sub + Div + FromStr + Debug + Copy,
    <T as FromStr>::Err: Debug,
{
}
