use std::ops::{Neg, AddAssign, Sub, SubAssign, Add, Mul, MulAssign};
use crate::vector::{Vector, VectorT};
use crate::matrix::{Matrix, MatrixT, MatrixElement};
use std::iter::Sum;


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

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T> SubAssign for Vector<T>
    where T: MatrixElement
{
    fn sub_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 -= *elm_pair.1;
        }
    }
}

impl<T> MulAssign<T> for Vector<T>
    where T: MatrixElement {

    fn mul_assign(&mut self, rhs: T) {
        let mut ret = self;
        for elm in &mut ret.elements {
            *elm = rhs * *elm;
        }
    }
}

impl<T> Mul for &Vector<T>
    where T: MatrixElement {
    type Output = T;

    fn mul(self, rhs: Self) -> T {
       self.vec_prod(rhs)
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
        if self.column_count() != rhs.len() {
            panic!("Cannot multiply matrix {} with vector {} because of dimensions", self.dimensions(), rhs.len());
        }
        let mut result = Vector::<T>::new(self.row_count());
        for row in 0..self.row_count() {
            result[row] = self.row_iter(row).zip(rhs.iter())
                .map(|pair| *pair.0 * *pair.1)
                .sum();
        }
        result
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
        if self.len() != rhs.row_count() {
            panic!("Cannot multiply vector {} with matrix {} because of dimensions", self.len(), rhs.dimensions());
        }
        let mut result = Vector::<T>::new(rhs.column_count());
        for column in 0..rhs.column_count() {
            result[column] = rhs.col_iter(column).zip(self.iter())
                .map(|pair| *pair.0 * *pair.1)
                .sum();
        }
        result
    }
}