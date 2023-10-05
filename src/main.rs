use n::{tensor, Tensor};

pub fn to_str(v: [u32; 2]) {
    println!("{}",);
}

fn main() {
    let t0 = tensor!(i64, [[[[1], [2]], [[3, 4]]]]);
    let a = [[1, 2, 3], [4, 5, 6]];
    let b = a[1][2];
    println!("{:?}", t0);
    to_str([1, 2]);
}
