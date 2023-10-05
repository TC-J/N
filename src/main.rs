use n::{tensor::tensor, tensor::Tensor};

fn main() {
    let t0 = tensor!(i64, [[[[1], [2]], [[3], [4]]]]);
    println!("{:?}", t0);
}
