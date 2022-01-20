use machlearn::matrix::Matrix;

#[test]
fn equals() {
    let mut a = Matrix::new(3, 2);
    let mut b = Matrix::new(3, 2);
    let mut c = Matrix::new(2, 3);
    let mut d = Matrix::new(3, 2);
    d[(0,0)] = 1.;
    let mut e = Matrix::new(3, 2);
    e[(0,0)] = 1.;

    assert_eq!(a, a); // same instance
    assert_eq!(a, b); // equal
    assert_ne!(a, c); // different dimensions
    assert_ne!(a, d); // different values
    assert_eq!(d, e); // same values
}