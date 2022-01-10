//! Vector type and arithmetic operations on the Vector.

use std::fmt::{Display, Formatter, Write};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Deref, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use std::slice::Iter;

use itertools::Itertools;

use crate::matrix::{Matrix, MatrixElement, MatrixDimensions};
use crate::neuralnetwork::ampl;

pub mod arit;

/// Vector with arithmetic operations.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Vector<T>
    where T: MatrixElement
{
    elements: Vec<T>
}

impl<T> Vector<T>
    where T: MatrixElement
{
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn vec_prod(&self, rhs: &Vector<T>) -> T {
        let v1 = self;
        let v2 = rhs;
        if v1.len() != v2.len() {
            panic!("Vector 1 length {} not equal to vector 2 length {}", v1.len(), v2.len())
        }
        self.iter().zip(rhs.iter())
            .map(|pair| *pair.0 * *pair.1)
            .sum()
    }

    pub fn iter(&self) -> impl Iterator<Item=&T> {
        self.elements.iter()
    }

    pub fn to_matrix(self) -> Matrix<T> {
        let len = self.len();
        Matrix::new_from_elements(len, 1, self.elements,
                                  1, len)
    }

    pub fn as_matrix(&self) -> Matrix<T> {
        todo!("implement")
    }
}

impl<T> Vector<T>
    where T: MatrixElement
{
    pub fn new(len: usize) -> Vector<T> {
        Vector {
            elements: vec![Default::default(); len]
        }
    }
}

impl<T> Vector<T>
    where T: MatrixElement
{

    pub fn apply(self, mut func: impl FnMut(T) -> T) -> Self {
        let mut ret = self;
        for elm in &mut ret.elements {
            *elm = func(*elm);
        }
        ret
    }
}

impl<T> Index<usize> for Vector<T>
    where T: MatrixElement
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.elements[index]
    }
}

impl<T> IndexMut<usize> for Vector<T>
    where T: MatrixElement
{

    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.elements[index]
    }
}

impl<T> Display for Vector<T>
    where T: MatrixElement
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        write!(f, "{}", self.iter().format(" "))?;
        f.write_str("]")?;
        std::fmt::Result::Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::*;


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
    fn mul2() {
        let mut a = Vector::new( 2);
        a[0] = 1.1;
        a[1] = 2.1;

        let mut b = Vector::new( 2);
        b[0] = 2.;
        b[1] = 3.;

        let x: &Vector<f64> = &a;
        let y: &Vector<f64> = &b;
        assert_eq!(8.5, x * y);
        assert_eq!(8.5, &a * &b);
        assert_eq!(8.5, (&a).mul(&b));
        // assert_eq!(8.5, (&a).mul(&b));
        assert_eq!(8.5, <&Vector<f64> as Mul>::mul(&a, &b));
        assert_eq!(8.5, Mul::mul(&a, &b));
    }

    #[test]
    fn vec_prod() {
        let mut a = Vector::new( 2);
        a[0] = 1.1;
        a[1] = 2.1;

        let mut b = Vector::new( 2);
        b[0] = 2.;
        b[1] = 3.;

        assert_eq!(8.5, a.vec_prod(&b));
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
    fn mul_scalar() {
        let mut a = Vector::new( 2);
        a[0] = 1.1;
        a[1] = 2.1;

        let mut result = Vector::new( 2);
        result[0] = 2.2;
        result[1] = 4.2;

        a *= 2.;

        assert_eq!(result, a);

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
}