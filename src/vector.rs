//! Vector type and arithmetic operations on the Vector.

use std::fmt::{Display, Formatter, Write};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Deref, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use std::slice::Iter;

use itertools::Itertools;

use crate::matrix::Matrix;
use crate::neuralnetwork::ampl;

pub mod arit;

pub type vdim = usize;

pub trait VectorT<T> :
Index<vdim, Output=T>
{
    fn len(&self) -> vdim;

}

/// Vector with arithmetic operations.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Vector<T>
    where T: Clone + PartialEq
{
    elements: Vec<T>
}

impl<T> Vector<T>
    where T: Clone + PartialEq + Default
{
    pub fn new(len: vdim) -> Vector<T> {
        Vector {
            elements: vec![Default::default(); len]
        }
    }
}

impl<T> VectorT<T> for Vector<T>
    where T: Clone + PartialEq
{
    fn len(&self) -> vdim {
        self.elements.len()
    }


}

impl<T> Vector<T>
    where T: Clone + PartialEq
{

    pub fn iter(&self) -> impl Iterator<Item=&T> {
        self.elements.iter()
    }

}

impl<T> Vector<T>
    where T: Clone + PartialEq + Copy
{

    pub fn apply(self, func: fn(T) -> T) -> Self {
        let mut ret = self;
        for elm in &mut ret.elements {
            *elm = func(*elm);
        }
        ret
    }
}

impl<T> Index<vdim> for Vector<T>
    where T: Clone + PartialEq
{
    type Output = T;

    fn index(&self, index: vdim) -> &T {
        &self.elements[index]
    }
}

impl<T> IndexMut<vdim> for Vector<T>
    where T: Clone + PartialEq
{

    fn index_mut(&mut self, index: vdim) -> &mut T {
        &mut self.elements[index]
    }
}

impl<T> Display for Vector<T>
    where T: Clone + PartialEq + Display
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        write!(f, "{}", self.iter().format(" "))?;
        f.write_str("]")?;
        std::fmt::Result::Ok(())
    }
}



impl<'a, T> AsRef<dyn VectorT<T> + 'a> for Vector<T>
    where T: Clone + PartialEq + 'a {

    fn as_ref(&self) -> &(dyn VectorT<T> + 'a) {
        self
    }
}



#[cfg(test)]
mod tests {
    use crate::vector::*;
    use crate::vector::arit::VectorProduct;

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
    fn mul3() {
        let mut a = Vector::new( 2);
        a[0] = 1.1;
        a[1] = 2.1;

        let mut b = Vector::new( 2);
        b[0] = 2.;
        b[1] = 3.;

        assert_eq!(8.5, a.vec_prod(&b));
        assert_eq!(8.5, (&a).vec_prod(&b));
        assert_eq!(8.5, <&Vector<f64> as VectorProduct<f64>>::vec_prod(&a, &b));
        assert_eq!(8.5, VectorProduct::vec_prod(&a, &b));
        assert_eq!(8.5, VectorProduct::<f64>::vec_prod(&a, &b));
    }

}