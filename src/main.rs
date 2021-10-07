use machlearn::matrix::Matrix;
use std::ops::IndexMut;

#[derive(Clone, PartialEq, Eq)]
struct TestNumber {

}

fn main() {

    let mut vec = vec![1];
    vec[0] = 1;

    let mut a = Matrix::<f64>::new(0., 3, 2);
    let mut b = Matrix::new(0., 2, 2);
    let mut c = Matrix::new(0., 2, 2);
    println!("{:?}", a);
    println!("{} {} {}", a == a, a == b, b == c);

    a[0][0] = 1.;
    println!("{:?}", a);

    // let mut d = Matrix::new(TestNumber{}, 2, 2);
    // let mut e = Matrix::new(TestNumber{}, 2, 2);
    // println!("{}", d == e);


}
