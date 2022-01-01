//! Vector type and arithmetic operations on the Vector.

use std::fmt::{Display, Formatter, Write};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Deref, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use std::slice::Iter;

use itertools::Itertools;

use crate::neuralnetwork::ampl;
use crate::matrix::Matrix;

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

impl<T> Neg for Vector<T>
    where T: Copy + PartialEq + Neg<Output = T>
{
    type Output = Vector<T>;

    fn neg(mut self) -> Self::Output {
        for elm in self.elements.iter_mut() {
            *elm = elm.neg();
        }
        self
    }
}

impl<T> Add for Vector<T>
    where T: Copy + PartialEq + AddAssign
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T> AddAssign for Vector<T>
    where T: Copy + PartialEq + AddAssign
{
    fn add_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 += *elm_pair.1;
        }
    }
}

impl<T> Sub for Vector<T>
    where T: Copy + PartialEq + SubAssign
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T> SubAssign for Vector<T>
    where T: Copy + PartialEq + SubAssign
{
    fn sub_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 -= *elm_pair.1;
        }
    }
}

impl<T> Mul for &dyn VectorT<T>
    where T: Copy + PartialEq + AddAssign + Mul<Output=T> + Default {
    type Output = T;

    fn mul(self, rhs: Self) -> T {
        if self.len() != rhs.len() {
            panic!("Vector 1 length {} not equal to vector 2 length length {}", self.len(), rhs.len())
        }
        let mut sum = T::default();
        for i in 0..self.len() {
            sum += self[i] * rhs[i];
        }
        sum
    }
}

impl<T> Mul for &Vector<T>
    where T: Copy + PartialEq + AddAssign + Mul<Output=T> + Default {
    type Output = T;

    fn mul(self, rhs: Self) -> T {
        if self.len() != rhs.len() {
            panic!("Vector 1 length {} not equal to vector 2 length length {}", self.len(), rhs.len())
        }
        let mut sum = T::default();
        for i in 0..self.len() {
            sum += self[i] * rhs[i];
        }
        sum
    }
}

impl<'a, T> AsRef<dyn VectorT<T> + 'a> for Vector<T>
    where T: Clone + PartialEq {

    fn as_ref(&self) -> &(dyn VectorT<T> + 'a) {
        self
    }
}

pub trait Mult<T, Rhs = Self> {

    type Output;

    fn mult(self, rhs: Rhs) -> Self::Output;
}


impl<T, R: AsRef<dyn VectorT<T>>> Mult<T> for R
    where T: Copy + PartialEq + AddAssign + Mul<Output=T> + Default
{
    type Output = T;

    fn mult(self, rhs: Self) -> Self::Output {
        let v1 = self.as_ref();
        let v2 = rhs.as_ref();
        if v1.len() != v2.len() {
            panic!("Vector 1 length {} not equal to vector 2 length length {}", v1.len(), v2.len())
        }
        let mut sum = T::default();
        for i in 0..v1.len() {
            sum += v1[i] * v2[i];
        }
        sum
    }
}

// impl<T, V: VectorT<T>, R: AsRef<V>> Mul for R
//  {
//      type Output = T;
//
//
//      fn mul(self, rhs: &Self) -> Self::Output {
//         todo!()
//     }
//
//
//     // fn mul(self, rhs: Self) -> T {
//     //     if self.len() != rhs.len() {
//     //         panic!("Vector 1 length {} not equal to vector 2 length length {}", self.len(), rhs.len())
//     //     }
//     //     let mut sum = T::default();
//     //     for i in 0..self.len() {
//     //         sum += self[i] * rhs[i];
//     //     }
//     //     sum
//     // }
// }

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

    #[test]
    fn product() {
        let mut a = Vector::new( 2);
        a[0] = 1.1;
        a[1] = 2.1;

        let mut b = Vector::new( 2);
        b[0] = 2.;
        b[1] = 3.;

        let x: &dyn VectorT<f64> = &a;
        let y: &dyn VectorT<f64> = &b;
        assert_eq!(8.5, x * y);
    }

}