use std::ops::{Neg, AddAssign, Sub, SubAssign, Add, Mul, MulAssign};
use crate::vector::{Vector};
use crate::matrix::{Matrix, MatrixElement};



impl<T> Neg for Vector<T>
    where T: MatrixElement
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
    where T: MatrixElement
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T> AddAssign for Vector<T>
    where T: MatrixElement
{
    fn add_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 += *elm_pair.1;
        }
    }
}

impl<T> Sub for Vector<T>
    where T: MatrixElement
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<T> Sub<&Vector<T>> for Vector<T>
    where T: MatrixElement
{
    type Output = Self;

    fn sub(mut self, rhs: &Vector<T>) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T> SubAssign for Vector<T>
    where T: MatrixElement
{
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign(&other);
    }
}

impl<T> SubAssign<&Vector<T>> for Vector<T>
    where T: MatrixElement
{
    fn sub_assign(&mut self, other: &Vector<T>) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 -= *elm_pair.1;
        }
    }
}

impl<T> MulAssign<T> for Vector<T>
    where T: MatrixElement {

    fn mul_assign(&mut self, rhs: T) {
        for elm in &mut self.elements {
            *elm = rhs * *elm;
        }
    }
}

impl<T> Mul for &Vector<T>
    where T: MatrixElement {
    type Output = T;

    fn mul(self, rhs: Self) -> T {
       self.scalar_prod(rhs)
    }
}

impl<T> Mul<Vector<T>> for Matrix<T>
    where T: MatrixElement
{
    type Output = Vector<T>;

    fn mul(self, rhs: Vector<T>) -> Vector<T> {
        &self * &rhs
    }
}

impl<T> Mul<&Vector<T>> for &Matrix<T>
    where T: MatrixElement
{
    type Output = Vector<T>;

    fn mul(self, rhs: &Vector<T>) -> Vector<T> {
        self.mul_vector(rhs)
    }
}

impl<T> Mul<Matrix<T>> for Vector<T>
    where T: MatrixElement
{
    type Output = Vector<T>;

    fn mul(self, rhs: Matrix<T>) -> Vector<T> {
        &self * &rhs
    }
}

impl<T> Mul<&Matrix<T>> for &Vector<T>
    where T: MatrixElement
{
    type Output = Vector<T>;

    fn mul(self, rhs: &Matrix<T>) -> Vector<T> {
        rhs.mul_vector_lhs(&self)
    }
}