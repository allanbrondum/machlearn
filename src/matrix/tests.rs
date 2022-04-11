use crate::matrix::*;
use itertools::Itertools;
use std::borrow::Borrow;
use crate::vector::Vector;

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

#[test]
fn index() {
    let mut a = Matrix::new( 3, 2);
    a[(0,0)] = 1.1;
    a[(1,0)] = 2.1;
    a[(0,1)] = 3.1;
    a[(2,1)] = 4.1;

    assert_eq!(1.1, a[(0,0)]);
    assert_eq!(2.1, a[(1,0)]);
    assert_eq!(3.1, a[(0,1)]);
    assert_eq!(4.1, a[(2,1)]);

}

#[test]
fn matrix_with_col_stride() {
    let mut a = Matrix::new_from_elements(
        MatrixLinearIndex::new_col_stride(MatrixDimensions{rows: 3, columns: 2}, 3),
        vec!(1.1, 2.1, 0.0, 3.1, 0.0, 4.1));

    a[(0,0)] = 1.1;
    a[(1,0)] = 2.1;
    a[(0,1)] = 3.1;
    a[(2,1)] = 4.1;

    assert_eq!(1.1, a[(0,0)]);
    assert_eq!(2.1, a[(1,0)]);
    assert_eq!(0.0, a[(2,0)]);
    assert_eq!(3.1, a[(0,1)]);
    assert_eq!(0.0, a[(1,1)]);
    assert_eq!(4.1, a[(2,1)]);
    assert_eq!(1.1, *a.elm(0,0));
    assert_eq!(2.1, *a.elm(1,0));
    assert_eq!(0.0, *a.elm(2,0));
    assert_eq!(3.1, *a.elm(0,1));
    assert_eq!(0.0, *a.elm(1,1));
    assert_eq!(4.1, *a.elm(2,1));

    let mut row_iter = a.row_iter(0);
    assert_eq!(Some(&1.1), row_iter.next());
    assert_eq!(Some(&3.1), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(1);
    assert_eq!(Some(&2.1), row_iter.next());
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(2);
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(Some(&4.1), row_iter.next());
    assert_eq!(None, row_iter.next());

    let mut col_iter = a.col_iter(0);
    assert_eq!(Some(&1.1), col_iter.next());
    assert_eq!(Some(&2.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(None, col_iter.next());
    let mut col_iter = a.col_iter(1);
    assert_eq!(Some(&3.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(Some(&4.1), col_iter.next());
    assert_eq!(None, col_iter.next());

}

#[test]
fn matrix_with_row_stride() {
    let mut a = Matrix::new_from_elements(
        MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 3, columns: 2}, 2),
        vec!(1.1, 3.1, 2.1, 0.0, 0.0, 4.1));

    a[(0,0)] = 1.1;
    a[(1,0)] = 2.1;
    a[(0,1)] = 3.1;
    a[(2,1)] = 4.1;

    assert_eq!(1.1, a[(0,0)]);
    assert_eq!(2.1, a[(1,0)]);
    assert_eq!(0.0, a[(2,0)]);
    assert_eq!(3.1, a[(0,1)]);
    assert_eq!(0.0, a[(1,1)]);
    assert_eq!(4.1, a[(2,1)]);
    assert_eq!(1.1, *a.elm(0,0));
    assert_eq!(2.1, *a.elm(1,0));
    assert_eq!(0.0, *a.elm(2,0));
    assert_eq!(3.1, *a.elm(0,1));
    assert_eq!(0.0, *a.elm(1,1));
    assert_eq!(4.1, *a.elm(2,1));

    let mut row_iter = a.row_iter(0);
    assert_eq!(Some(&1.1), row_iter.next());
    assert_eq!(Some(&3.1), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(1);
    assert_eq!(Some(&2.1), row_iter.next());
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(2);
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(Some(&4.1), row_iter.next());
    assert_eq!(None, row_iter.next());

    let mut col_iter = a.col_iter(0);
    assert_eq!(Some(&1.1), col_iter.next());
    assert_eq!(Some(&2.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(None, col_iter.next());
    let mut col_iter = a.col_iter(1);
    assert_eq!(Some(&3.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(Some(&4.1), col_iter.next());
    assert_eq!(None, col_iter.next());

}

#[test]
fn matrix_mut_slice_view_with_col_stride() {
    let mut vec = vec!(1.1, 2.1, 0.0, 3.1, 0.0, 4.1);
    let mut a = MutSliceView::new_col_stride(
        3, 2,
        &mut vec,
        3);

    *a.elm_mut(0,0) = 1.1;
    *a.elm_mut(1,0) = 2.1;
    *a.elm_mut(0,1) = 3.1;
    *a.elm_mut(2,1) = 4.1;

    assert_eq!(1.1, *a.elm(0,0));
    assert_eq!(2.1, *a.elm(1,0));
    assert_eq!(0.0, *a.elm(2,0));
    assert_eq!(3.1, *a.elm(0,1));
    assert_eq!(0.0, *a.elm(1,1));
    assert_eq!(4.1, *a.elm(2,1));

    let mut row_iter = a.row_iter(0);
    assert_eq!(Some(&1.1), row_iter.next());
    assert_eq!(Some(&3.1), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(1);
    assert_eq!(Some(&2.1), row_iter.next());
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(2);
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(Some(&4.1), row_iter.next());
    assert_eq!(None, row_iter.next());

    let mut col_iter = a.col_iter(0);
    assert_eq!(Some(&1.1), col_iter.next());
    assert_eq!(Some(&2.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(None, col_iter.next());
    let mut col_iter = a.col_iter(1);
    assert_eq!(Some(&3.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(Some(&4.1), col_iter.next());
    assert_eq!(None, col_iter.next());

}

#[test]
fn matrix_mut_slice_view_with_row_stride() {
    let mut vec = vec!(1.1, 3.1, 2.1, 0.0, 0.0, 4.1);
    let mut a = MutSliceView::new_row_stride(
        3, 2,
        &mut vec,
        2);

    *a.elm_mut(0,0) = 1.1;
    *a.elm_mut(1,0) = 2.1;
    *a.elm_mut(0,1) = 3.1;
    *a.elm_mut(2,1) = 4.1;

    assert_eq!(1.1, *a.elm(0,0));
    assert_eq!(2.1, *a.elm(1,0));
    assert_eq!(0.0, *a.elm(2,0));
    assert_eq!(3.1, *a.elm(0,1));
    assert_eq!(0.0, *a.elm(1,1));
    assert_eq!(4.1, *a.elm(2,1));

    let mut row_iter = a.row_iter(0);
    assert_eq!(Some(&1.1), row_iter.next());
    assert_eq!(Some(&3.1), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(1);
    assert_eq!(Some(&2.1), row_iter.next());
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(2);
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(Some(&4.1), row_iter.next());
    assert_eq!(None, row_iter.next());

    let mut col_iter = a.col_iter(0);
    assert_eq!(Some(&1.1), col_iter.next());
    assert_eq!(Some(&2.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(None, col_iter.next());
    let mut col_iter = a.col_iter(1);
    assert_eq!(Some(&3.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(Some(&4.1), col_iter.next());
    assert_eq!(None, col_iter.next());

}

#[test]
fn matrix_slice_view_with_col_stride() {
    let vec = vec!(1.1, 2.1, 0.0, 3.1, 0.0, 4.1);
    let a = SliceView::new_col_stride(
        3, 2,
        &vec,
        3);

    assert_eq!(1.1, *a.elm(0,0));
    assert_eq!(2.1, *a.elm(1,0));
    assert_eq!(0.0, *a.elm(2,0));
    assert_eq!(3.1, *a.elm(0,1));
    assert_eq!(0.0, *a.elm(1,1));
    assert_eq!(4.1, *a.elm(2,1));

    let mut row_iter = a.row_iter(0);
    assert_eq!(Some(&1.1), row_iter.next());
    assert_eq!(Some(&3.1), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(1);
    assert_eq!(Some(&2.1), row_iter.next());
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(2);
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(Some(&4.1), row_iter.next());
    assert_eq!(None, row_iter.next());

    let mut col_iter = a.col_iter(0);
    assert_eq!(Some(&1.1), col_iter.next());
    assert_eq!(Some(&2.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(None, col_iter.next());
    let mut col_iter = a.col_iter(1);
    assert_eq!(Some(&3.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(Some(&4.1), col_iter.next());
    assert_eq!(None, col_iter.next());

}

#[test]
fn matrix_slice_view_with_row_stride() {
    let vec = vec!(1.1, 3.1, 2.1, 0.0, 0.0, 4.1);
    let a = SliceView::new_row_stride(
        3, 2,
        &vec,
        2);

    assert_eq!(1.1, *a.elm(0,0));
    assert_eq!(2.1, *a.elm(1,0));
    assert_eq!(0.0, *a.elm(2,0));
    assert_eq!(3.1, *a.elm(0,1));
    assert_eq!(0.0, *a.elm(1,1));
    assert_eq!(4.1, *a.elm(2,1));

    let mut row_iter = a.row_iter(0);
    assert_eq!(Some(&1.1), row_iter.next());
    assert_eq!(Some(&3.1), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(1);
    assert_eq!(Some(&2.1), row_iter.next());
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(None, row_iter.next());
    let mut row_iter = a.row_iter(2);
    assert_eq!(Some(&0.0), row_iter.next());
    assert_eq!(Some(&4.1), row_iter.next());
    assert_eq!(None, row_iter.next());

    let mut col_iter = a.col_iter(0);
    assert_eq!(Some(&1.1), col_iter.next());
    assert_eq!(Some(&2.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(None, col_iter.next());
    let mut col_iter = a.col_iter(1);
    assert_eq!(Some(&3.1), col_iter.next());
    assert_eq!(Some(&0.0), col_iter.next());
    assert_eq!(Some(&4.1), col_iter.next());
    assert_eq!(None, col_iter.next());

}

#[test]
fn neg() {
    let mut a = Matrix::new( 3, 2);
    a[(0,0)] = 1.1;
    a[(1,0)] = 2.1;
    a[(0,1)] = 3.1;
    a[(2,1)] = 4.1;

    let a = -a;

    assert_eq!(-1.1, a[(0,0)]);
    assert_eq!(-2.1, a[(1,0)]);
    assert_eq!(-3.1, a[(0,1)]);
    assert_eq!(-4.1, a[(2,1)]);

}

#[test]
fn add() {
    let mut a = Matrix::new( 3, 2);
    a[(0,0)] = 1.1;
    a[(1,0)] = 2.1;
    a[(0,1)] = 3.1;
    a[(2,1)] = 4.1;
    let mut b = Matrix::new( 3, 2);
    b[(0,0)] = 10.;
    b[(1,0)] = 20.;
    b[(0,1)] = 30.;
    b[(2,1)] = 40.;

    let a = a + b;

    assert_eq!(1.1 + 10., a[(0,0)]);
    assert_eq!(2.1 + 20., a[(1,0)]);
    assert_eq!(3.1 + 30., a[(0,1)]);
    assert_eq!(4.1 + 40., a[(2,1)]);
    assert_eq!(0., a[(1,1)]);

}

#[test]
fn add_assign() {
    let mut a = Matrix::new( 3, 2);
    a[(0,0)] = 1.1;
    a[(1,0)] = 2.1;
    a[(0,1)] = 3.1;
    a[(2,1)] = 4.1;
    let mut b = Matrix::new( 3, 2);
    b[(0,0)] = 10.;
    b[(1,0)] = 20.;
    b[(0,1)] = 30.;
    b[(2,1)] = 40.;

    a += b;

    assert_eq!(1.1 + 10., a[(0,0)]);
    assert_eq!(2.1 + 20., a[(1,0)]);
    assert_eq!(3.1 + 30., a[(0,1)]);
    assert_eq!(4.1 + 40., a[(2,1)]);
    assert_eq!(0., a[(1,1)]);

}

#[test]
fn sub() {
    let mut a = Matrix::new( 3, 2);
    a[(0,0)] = 1.1;
    a[(1,0)] = 2.1;
    a[(0,1)] = 3.1;
    a[(2,1)] = 4.1;
    let mut b = Matrix::new( 3, 2);
    b[(0,0)] = 10.;
    b[(1,0)] = 20.;
    b[(0,1)] = 30.;
    b[(2,1)] = 40.;

    let a = a - b;

    assert_eq!(1.1 - 10., a[(0,0)]);
    assert_eq!(2.1 - 20., a[(1,0)]);
    assert_eq!(3.1 - 30., a[(0,1)]);
    assert_eq!(4.1 - 40., a[(2,1)]);
    assert_eq!(0., a[(1,1)]);

}

#[test]
fn sub_assign() {
    let mut a = Matrix::new( 3, 2);
    a[(0,0)] = 1.1;
    a[(1,0)] = 2.1;
    a[(0,1)] = 3.1;
    a[(2,1)] = 4.1;
    let mut b = Matrix::new( 3, 2);
    b[(0,0)] = 10.;
    b[(1,0)] = 20.;
    b[(0,1)] = 30.;
    b[(2,1)] = 40.;

    a -= b;

    assert_eq!(1.1 - 10., a[(0,0)]);
    assert_eq!(2.1 - 20., a[(1,0)]);
    assert_eq!(3.1 - 30., a[(0,1)]);
    assert_eq!(4.1 - 40., a[(2,1)]);
    assert_eq!(0., a[(1,1)]);
}

#[test]
fn row_iter() {
    let mut a = Matrix::new( 3, 2);
    a[(0,0)] = 1.1;
    a[(0,1)] = 2.1;
    a[(1,0)] = 3.1;
    a[(1,1)] = 4.1;

    let mut row_iter = a.row_iter(0);

    assert_eq!(Some(&1.1), row_iter.next());
    assert_eq!(Some(&2.1), row_iter.next());
    assert_eq!(None, row_iter.next());
    assert_eq!(None, row_iter.next());

    let mut row_iter = a.row_iter(1);

    assert_eq!(Some(&3.1), row_iter.next());
    assert_eq!(Some(&4.1), row_iter.next());
    assert_eq!(None, row_iter.next());
    assert_eq!(None, row_iter.next());
}

#[test]
fn all_elements_iter() {
    let mut a = Matrix::new(3, 2);
    a[(0, 0)] = 1.1;
    a[(0, 1)] = 2.1;
    a[(1, 0)] = 3.1;
    a[(1, 1)] = 4.1;

    let vecres: Vec<_> = a.iter().collect();

    assert_eq!(vec!(&1.1, &2.1, &3.1, &4.1, &0.0, &0.0), vecres);
}

#[test]
fn col_iter() {
    let mut a = Matrix::new( 3, 2);
    a[(0,0)] = 1.1;
    a[(0,1)] = 2.1;
    a[(1,0)] = 3.1;
    a[(1,1)] = 4.1;

    let mut col_iter = a.col_iter(0);

    assert_eq!(Some(&1.1), col_iter.next());
    assert_eq!(Some(&3.1), col_iter.next());
    assert_eq!(Some(&0.), col_iter.next());
    assert_eq!(None, col_iter.next());
    assert_eq!(None, col_iter.next());

    let mut col_iter = a.col_iter(1);

    assert_eq!(Some(&2.1), col_iter.next());
    assert_eq!(Some(&4.1), col_iter.next());
    assert_eq!(Some(&0.), col_iter.next());
    assert_eq!(None, col_iter.next());
    assert_eq!(None, col_iter.next());

    let mut col_iter = a.col_iter(1);
    // col_iter.take_while_ref()

    let r: &dyn Iterator<Item=&f64> = &col_iter;
    let r2 = col_iter.by_ref();
    let r: &dyn Iterator<Item=&f64> = &r2;
    // r2.take_while()
    // vec!(1).iter().as_ref().split()
    // let r: &dyn Iterator<Item=&f64> = &
}

#[test]
fn multiply() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let mut b = Matrix::new( 4, 2);
    b[(0,0)] = 1;
    b[(0,1)] = 2;
    b[(1,0)] = 3;
    b[(1,1)] = 4;

    println!("{} {}", a, b);

    let mut product = Matrix::new( 3, 2);
    product[(0,0)] = 7;
    product[(0,1)] = 10;
    product[(1,0)] = 15;
    product[(1,1)] = 22;

    assert_eq!(product, a * b);
}

#[test]
fn multiply2() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let mut b = Matrix::new( 4, 2);
    b[(0,0)] = 1;
    b[(0,1)] = 2;
    b[(1,0)] = 3;
    b[(1,1)] = 4;

    println!("{} {}", a, b);

    let mut product = Matrix::new( 3, 2);
    product[(0,0)] = 7;
    product[(0,1)] = 10;
    product[(1,0)] = 15;
    product[(1,1)] = 22;

    assert_eq!(product, a.mul_mat(&b));
}

#[test]
fn multiply_refs() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let mut b = Matrix::new( 4, 2);
    b[(0,0)] = 1;
    b[(0,1)] = 2;
    b[(1,0)] = 3;
    b[(1,1)] = 4;

    println!("{} {}", a, b);

    let mut product = Matrix::new( 3, 2);
    product[(0,0)] = 7;
    product[(0,1)] = 10;
    product[(1,0)] = 15;
    product[(1,1)] = 22;

    assert_eq!(product, &a * &b);
}

#[test]
fn multiply_vector() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let mut b = Vector::new( 4);
    b[0] = 1;
    b[1] = 2;

    println!("{} {}", a, b);

    let mut product = Vector::new(3);
    product[0] = 5;
    product[1] = 11;
    product[2] = 0;

    assert_eq!(product, a.clone() * b.clone());
    assert_eq!(product, a.mul_vector(&b));
}

#[test]
fn multiply_vector_refs() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let mut b = Vector::new( 4);
    b[0] = 1;
    b[1] = 2;

    println!("{} {}", a, b);

    let mut product = Vector::new(3);
    product[0] = 5;
    product[1] = 11;
    product[2] = 0;

    assert_eq!(product, &a * &b);
}

#[test]
fn multiply_vector_lhs() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let mut b = Vector::new( 3);
    b[0] = 1;
    b[1] = 2;

    println!("{} {}", a, b);

    let mut product = Vector::new(4);
    product[0] = 7;
    product[1] = 10;
    product[2] = 0;
    product[3] = 0;

    assert_eq!(product, b.clone() * a.clone());
    assert_eq!(product, a.mul_vector_lhs(&b));
}

#[test]
fn multiply_vector_lhs_refs() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let mut b = Vector::new( 3);
    b[0] = 1;
    b[1] = 2;

    println!("{} {}", a, b);

    let mut product = Vector::new(4);
    product[0] = 7;
    product[1] = 10;
    product[2] = 0;
    product[3] = 0;

    assert_eq!(product, &b * &a);
}

// #[test]
// fn col_vector() {
//     let mut a = Matrix::new( 3, 4);
//     a[(0,0)] = 1;
//     a[(0,1)] = 2;
//     a[(1,0)] = 3;
//     a[(1,1)] = 4;
//
//     let b = a.col(1);
//
//     assert_eq!(3, b.len());
//     assert_eq!(2, b[0]);
//     assert_eq!(4, b[1]);
//     assert_eq!(0, b[2]);
// }
//
// #[test]
// fn row_vector() {
//     let mut a = Matrix::new( 3, 4);
//     a[(0,0)] = 1;
//     a[(0,1)] = 2;
//     a[(1,0)] = 3;
//     a[(1,1)] = 4;
//
//     let b = a.row(1);
//
//     assert_eq!(4, b.len());
//     assert_eq!(3, b[0]);
//     assert_eq!(4, b[1]);
//     assert_eq!(0, b[2]);
//     assert_eq!(0, b[3]);
// }

#[test]
fn elm() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    assert_eq!(1, *a.elm(0, 0));
    assert_eq!(2, *a.elm(0, 1));
    assert_eq!(3, *a.elm(1, 0));
    assert_eq!(4, *a.elm(1, 1));

}

#[test]
fn transpose() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let t = a.transpose();

    assert_eq!(MatrixDimensions{rows: 4, columns: 3}, t.dimensions());
    assert_eq!(1, *t.elm(0, 0));
    assert_eq!(2, *t.elm(1, 0));
    assert_eq!(3, *t.elm(0, 1));
    assert_eq!(4, *t.elm(1, 1));
    assert_eq!(1, t[(0,0)]);
    assert_eq!(2, t[(1,0)]);
    assert_eq!(3, t[(0,1)]);
    assert_eq!(4, t[(1,1)]);

    let t2 = t.transpose();

    assert_eq!(MatrixDimensions{rows: 3, columns: 4}, t2.dimensions());
    assert_eq!(1, *t2.elm(0, 0));
    assert_eq!(2, *t2.elm(0, 1));
    assert_eq!(3, *t2.elm(1, 0));
    assert_eq!(4, *t2.elm(1, 1));
    assert_eq!(1, t2[(0,0)]);
    assert_eq!(2, t2[(0,1)]);
    assert_eq!(3, t2[(1,0)]);
    assert_eq!(4, t2[(1,1)]);

}

#[test]
fn as_transpose() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let mut t = a.as_transpose();

    assert_eq!(MatrixDimensions{rows: 4, columns: 3}, t.dimensions());
    assert_eq!(1, *t.elm(0, 0));
    assert_eq!(2, *t.elm(1, 0));
    assert_eq!(3, *t.elm(0, 1));
    assert_eq!(4, *t.elm(1, 1));
    let col0: Vec<_> = t.col_iter(0).copied().collect();
    assert_eq!(vec!(1, 2, 0, 0), col0);
    let row0: Vec<_> = t.row_iter(0).copied().collect();
    assert_eq!(vec!(1, 3, 0), row0);

    let t2 = t.as_transpose();

    assert_eq!(MatrixDimensions{rows: 3, columns: 4}, t2.dimensions());
    assert_eq!(1, *t2.elm(0, 0));
    assert_eq!(2, *t2.elm(0, 1));
    assert_eq!(3, *t2.elm(1, 0));
    assert_eq!(4, *t2.elm(1, 1));
    let col0: Vec<_> = t2.col_iter(0).copied().collect();
    assert_eq!(vec!(1, 3, 0), col0);
    let row0: Vec<_> = t2.row_iter(0).copied().collect();
    assert_eq!(vec!(1, 2, 0, 0), row0);
}

#[test]
fn multiply_scalar() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let mut result = Matrix::new( 3, 4);
    result[(0,0)] = 2;
    result[(0,1)] = 4;
    result[(1,0)] = 6;
    result[(1,1)] = 8;

    a *= 2;

    assert_eq!(result, a);
}

#[test]
fn multiply_scalar2() {
    let mut a = Matrix::new( 3, 4);
    a[(0,0)] = 1.;
    a[(0,1)] = 2.;
    a[(1,0)] = 3.;
    a[(1,1)] = 4.;

    let mut result = Matrix::new( 3, 4);
    result[(0,0)] = 2.;
    result[(0,1)] = 4.;
    result[(1,0)] = 6.;
    result[(1,1)] = 8.;

    assert_eq!(result, 2. * a);
}

#[test]
fn scalar_product() {
    let mut a = Matrix::new( 2, 3);
    a[(0,0)] = 1;
    a[(0,1)] = 2;
    a[(1,0)] = 3;
    a[(1,1)] = 4;

    let mut b = Matrix::new( 2, 3);
    b[(0,0)] = 3;
    b[(0,1)] = 2;
    b[(1,0)] = 2;
    b[(1,1)] = 2;

    let result = a.scalar_prod(&b);
    assert_eq!(result, 3 + 4 + 6 + 8);
}

#[test]
fn apply() {
    let mut a = Matrix::new(2, 1);
    a[(0,0)] = 1;
    a[(1,0)] = 2;

    a = a.apply(|x| 2 * x);

    assert_eq!(2, a[(0,0)]);
    assert_eq!(4, a[(1,0)]);

    let mut c = 0;
    a = a.apply(|x| { c += 1; c * x});

    assert_eq!(2, a[(0,0)]);
    assert_eq!(8, a[(1,0)]);

    fn d(x: i32) -> i32 {
        x + 1
    }
    a = a.apply(d);

    assert_eq!(3, a[(0,0)]);
    assert_eq!(9, a[(1,0)]);

    a.apply_ref(|x| -x);

    assert_eq!(-3, a[(0,0)]);
    assert_eq!(-9, a[(1,0)]);
}
