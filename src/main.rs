use n::{tensor::index, tensor::tensor, tensor::Tensor};

fn main() {
    let t0 = tensor!(
        i64,
        [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12],]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24],]]
        ]
    );
    println!("{:?}", t0);
    let v = index!(t0, 0, 1, 0, 2);
    println!("{}", v);
}
