use crate::vector::*;
use std::ops::Mul;
use crate::matrix::{MatrixDimensions, MatrixT};


#[test]
fn equals() {
    let mut a = Vector::new(3);
    let mut b = Vector::new(3);
    let mut c = Vector::new(2);
    let mut d = Vector::new(3);
    d[0] = 1.;
    let mut e = Vector::new(3);
    e[0] = 1.;

    assert_eq!(a, a); // same instance
    assert_eq!(a, b); // equal
    assert_ne!(a, c); // different dimensions
    assert_ne!(a, d); // different values
    assert_eq!(d, e); // same values
}

#[test]
fn index() {
    let mut a = Vector::new(3);
    a[0] = 1.1;
    a[1] = 2.1;

    assert_eq!(1.1, a[0]);
    assert_eq!(2.1, a[1]);
}

#[test]
fn neg() {
    let mut a = Vector::new( 3);
    a[0] = 1.1;
    a[1] = 2.1;

    let a = -a;

    assert_eq!(-1.1, a[0]);
    assert_eq!(-2.1, a[1]);
    assert_eq!(0., a[2]);

}

#[test]
fn add() {
    let mut a = Vector::new( 3);
    a[0] = 1.1;
    a[1] = 2.1;
    let mut b = Vector::new(3);
    b[0] = 10.;
    b[1] = 20.;

    let a = a + b;

    assert_eq!(1.1 + 10., a[0]);
    assert_eq!(2.1 + 20., a[1]);
    assert_eq!(0., a[2]);

}

#[test]
fn add_vector() {
    let mut a = Vector::new( 3);
    a[0] = 1.1;
    a[1] = 2.1;
    let mut b = Vector::new(3);
    b[0] = 10.;
    b[1] = 20.;

    let a = a.add_vector(&b);

    assert_eq!(1.1 + 10., a[0]);
    assert_eq!(2.1 + 20., a[1]);
    assert_eq!(0., a[2]);

}


#[test]
fn add_assign() {
    let mut a = Vector::new( 3);
    a[0] = 1.1;
    a[1] = 2.1;
    let mut b = Vector::new(3);
    b[0] = 10.;
    b[1] = 20.;

    a += b;

    assert_eq!(1.1 + 10., a[0]);
    assert_eq!(2.1 + 20., a[1]);
    assert_eq!(0., a[2]);

}

#[test]
fn add_vector_assign() {
    let mut a = Vector::new( 3);
    a[0] = 1.1;
    a[1] = 2.1;
    let mut b = Vector::new(3);
    b[0] = 10.;
    b[1] = 20.;

    a.add_vector_assign(&b);

    assert_eq!(1.1 + 10., a[0]);
    assert_eq!(2.1 + 20., a[1]);
    assert_eq!(0., a[2]);

}


#[test]
fn sub() {
    let mut a = Vector::new( 3);
    a[0] = 1.1;
    a[1] = 2.1;
    let mut b = Vector::new(3);
    b[0] = 10.;
    b[1] = 20.;

    let a = a - b;

    assert_eq!(1.1 - 10., a[0]);
    assert_eq!(2.1 - 20., a[1]);
    assert_eq!(0., a[2]);

}

#[test]
fn sub_vector() {
    let mut a = Vector::new( 3);
    a[0] = 1.1;
    a[1] = 2.1;
    let mut b = Vector::new(3);
    b[0] = 10.;
    b[1] = 20.;

    let a = a.sub_vector(&b);

    assert_eq!(1.1 - 10., a[0]);
    assert_eq!(2.1 - 20., a[1]);
    assert_eq!(0., a[2]);

}

#[test]
fn sub_assign() {
    let mut a = Vector::new( 3);
    a[0] = 1.1;
    a[1] = 2.1;
    let mut b = Vector::new(3);
    b[0] = 10.;
    b[1] = 20.;

    a -= b;

    assert_eq!(1.1 - 10., a[0]);
    assert_eq!(2.1 - 20., a[1]);
    assert_eq!(0., a[2]);
}

#[test]
fn sub_vector_assign() {
    let mut a = Vector::new( 3);
    a[0] = 1.1;
    a[1] = 2.1;
    let mut b = Vector::new(3);
    b[0] = 10.;
    b[1] = 20.;

    a.sub_vector_assign(&b);

    assert_eq!(1.1 - 10., a[0]);
    assert_eq!(2.1 - 20., a[1]);
    assert_eq!(0., a[2]);
}

#[test]
fn iter() {
    let mut a = Vector::new( 2);
    a[0] = 1.1;
    a[1] = 2.1;

    let mut iter = a.iter();

    assert_eq!(Some(&1.1), iter.next());
    assert_eq!(Some(&2.1), iter.next());
    assert_eq!(None, iter.next());
}

// #[test]
// fn mul1() {
//     let mut a = Vector::new( 2);
//     a[0] = 1.1;
//     a[1] = 2.1;
//
//     let mut b = Vector::new( 2);
//     b[0] = 2.;
//     b[1] = 3.;
//
//     let x: &dyn VectorT<f64> = &a;
//     let y: &dyn VectorT<f64> = &b;
//     assert_eq!(8.5, x * y);
// }


#[test]
fn vec_prod() {
    let mut a = Vector::new( 2);
    a[0] = 1.1;
    a[1] = 2.1;

    let mut b = Vector::new( 2);
    b[0] = 2.;
    b[1] = 3.;

    assert_eq!(8.5, a.scalar_prod(&b));
}

#[test]
fn to_matrix() {
    let mut a = Vector::new( 2);
    a[0] = 1.1;
    a[1] = 2.1;

    let m = a.to_matrix();

    assert_eq!(MatrixDimensions {rows: 2, columns: 1}, m.dimensions());
    assert_eq!(1.1, *m.elm(0, 0));
    assert_eq!(2.1, *m.elm(1, 0));
}

#[test]
fn as_matrix_mut() {
    let mut a = Vector::new( 2);
    a[0] = 1.1;
    a[1] = 2.1;

    let mut m = a.as_matrix_mut();

    assert_eq!(MatrixDimensions {rows: 2, columns: 1}, m.dimensions());
    assert_eq!(1.1, *m.elm(0, 0));
    assert_eq!(2.1, *m.elm(1, 0));

    *m.elm_mut(0, 0) = 3.1;
    *m.elm_mut(1, 0) = 4.1;
    assert_eq!(3.1, a[0]);
    assert_eq!(4.1, a[1]);
}

#[test]
fn as_matrix() {
    let mut a = Vector::new( 2);
    a[0] = 1.1;
    a[1] = 2.1;

    let m = a.as_matrix();

    assert_eq!(MatrixDimensions {rows: 2, columns: 1}, m.dimensions());
    assert_eq!(1.1, *m.elm(0, 0));
    assert_eq!(2.1, *m.elm(1, 0));
}

#[test]
fn mul_scalar() {
    let mut a = Vector::new( 2);
    a[0] = 1.1;
    a[1] = 2.1;

    let mut result = Vector::new( 2);
    result[0] = 2.2;
    result[1] = 4.2;

    let b = a.clone();
    a *= 2.;

    assert_eq!(result, a);
    assert_eq!(result, 2. * b);

}

#[test]
fn mul_comp() {
    let mut a = Vector::new( 2);
    a[0] = 1;
    a[1] = 2;

    let mut b = Vector::new( 2);
    b[0] = 2;
    b[1] = 3;

    let mut result = Vector::new( 2);
    result[0] = 2;
    result[1] = 6;

    assert_eq!(result, a.mul_comp(&b));

}

#[test]
fn apply() {
    let mut a = Vector::new(2);
    a[0] = 1;
    a[1] = 2;

    a = a.apply(|x| 2 * x);

    assert_eq!(2, a[0]);
    assert_eq!(4, a[1]);

    let mut c = 0;
    a = a.apply(|x| { c += 1; c * x});

    assert_eq!(2, a[0]);
    assert_eq!(8, a[1]);

    fn d(x: i32) -> i32 {
        x + 1
    }
    a = a.apply(d);

    assert_eq!(3, a[0]);
    assert_eq!(9, a[1]);
}

#[test]
fn apply_ref() {
    let mut a = Vector::new(2);
    a[0] = 1;
    a[1] = 2;

    a.apply_ref(|x| 2 * x);

    assert_eq!(2, a[0]);
    assert_eq!(4, a[1]);
}