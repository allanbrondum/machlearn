use std::ops::{Neg, Add, AddAssign, Sub, SubAssign, Mul, MulAssign};
use crate::matrix::{Matrix, MatrixElement};
use std::iter::Sum;

impl<T> Neg for Matrix<T>
    where T: MatrixElement
{
    type Output = Matrix<T>;

    fn neg(mut self) -> Self::Output {
        for elm in self.elements.iter_mut() {
            *elm = elm.neg();
        }
        self
    }
}

impl<T> Add for Matrix<T>
    where T: MatrixElement
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T> AddAssign for Matrix<T>
    where T: MatrixElement
{
    fn add_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 += *elm_pair.1;
        }
    }
}

impl<T> Sub for Matrix<T>
    where T: MatrixElement
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T> SubAssign for Matrix<T>
    where T: MatrixElement
{
    fn sub_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 -= *elm_pair.1;
        }
    }
}

impl<T> MulAssign<T> for Matrix<T>
    where T: MatrixElement {

    fn mul_assign(&mut self, rhs: T) {
        let mut ret = self;
        for elm in &mut ret.elements {
            *elm = rhs * *elm;
        }
    }
}

impl<T> Mul for Matrix<T>
    where T: MatrixElement
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.mat_mul(&rhs)
    }
}

impl<T> Mul for &Matrix<T>
    where T: MatrixElement
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Self) -> Matrix<T> {
        self.mat_mul(rhs)
    }
}

