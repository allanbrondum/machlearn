
use std::ops::{Neg, AddAssign, Sub, SubAssign, Add, Mul, MulAssign};
use crate::vector::{Vector};
use crate::matrix::{Matrix, MatrixElement, MatrixT};



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
        self.add_vector_assign(&other);
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
        self.sub_vector_assign(other);
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

impl Mul<Vector<f64>> for f64
{
    type Output = Vector<f64>;

    fn mul(self, mut rhs: Vector<f64>) -> Vector<f64> {
        rhs *= self;
        rhs
    }
}

// impl<T> Mul<Vector<T>> for T
//     where T: MatrixElement {
//     type Output = Vector<T>;
//
//     fn mul(self, mut rhs: Vector<T>) -> Vector<T> {
//         rhs *= self;
//         rhs
//     }
// }

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