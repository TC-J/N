#![allow(dead_code)]

pub mod n_lang;

pub mod fundamental;

pub mod tensor;

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

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
