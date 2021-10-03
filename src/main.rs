use machlearn::matrix::Matrix;

#[derive(Clone, PartialEq, Eq)]
struct TestNumber {

}

fn main() {

    let a = Matrix::<f64>::new(0., 3, 2);
    let b = Matrix::new(0., 2, 2);
    let c = Matrix::new(0., 2, 2);
    println!("{:?}", a);
    println!("{} {} {}", a == a, a == b, b == c);

    let d = Matrix::new(TestNumber{}, 2, 2);
    let e = Matrix::new(TestNumber{}, 2, 2);
    println!("{}", d == e);
}
