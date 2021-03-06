use itertools::Itertools;
use machlearn::matrix::{Matrix, MatrixDimensions, MatrixT, StrideIter, TransposedMatrixView};

#[derive(Clone, PartialEq, Eq)]
struct TestNumber {

}

fn main() {

    let mut vec = vec![1];
    vec[0] = 1;

    let mut a = Matrix::<f64>::new( 3, 2);
    let mut b = Matrix::new( 2, 2);
    let mut c = Matrix::new( 2, 2);
    println!("{:?}", a);
    println!("{} {} {}", a == a, a == b, b == c);

    a[(0,0)] = 1.;
    println!("{:?}", a);

    // let mut d = Matrix::new(TestNumber{}, 2, 2);
    // let mut e = Matrix::new(TestNumber{}, 2, 2);
    // println!("{}", d == e);

    a[(0,0)] = 1.;
    a[(2,0)] = 2.;
    a[(0,1)] = 3.;
    println!("{}", a);

    // println!("{}", -a);

    // (0..2).into_iter().collect();
    println!("{}", a.col_iter(0).format(""));

    let dim = MatrixDimensions {rows:1, columns: 2};
    dim.transpose();
    let dimref = &dim;
    dimref.transpose();

    let b: TransposedMatrixView<_, _> = a.as_transpose();

    let ar = &a;
    // let ard: &dyn MatrixT<f64, ColIter=StrideIter<f64>, RowIter=StrideIter<f64>> = ar;
}
