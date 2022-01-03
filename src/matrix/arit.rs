use std::ops::{Neg, Add, AddAssign, Sub, SubAssign, Mul};
use crate::matrix::{Matrix, MatrixT};
use std::iter::Sum;

impl<T> Neg for Matrix<T>
    where T: Copy + PartialEq + Neg<Output = T>
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
    where T: Copy + PartialEq + AddAssign
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T> AddAssign for Matrix<T>
    where T: Copy + PartialEq + AddAssign
{
    fn add_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 += *elm_pair.1;
        }
    }
}

impl<T> Sub for Matrix<T>
    where T: Copy + PartialEq + SubAssign
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T> SubAssign for Matrix<T>
    where T: Copy + PartialEq + SubAssign
{
    fn sub_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 -= *elm_pair.1;
        }
    }
}

impl<T> Mul for Matrix<T>
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + AddAssign + Sum + 'static
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        &self * &rhs
    }
}

impl<T> Mul for &Matrix<T>
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + AddAssign + Sum + 'static
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Self) -> Matrix<T> {
        MatrixMultiplication::mat_mul(self, rhs)
    }
}

pub trait MatrixMultiplication<T>
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + Sum + 'static {

    fn mat_mul(self, rhs: Self) -> Matrix<T>;
}

impl<T, R: AsRef<dyn MatrixT<T>>> MatrixMultiplication<T> for &R
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + AddAssign + Sum + 'static
{
    fn mat_mul(self, rhs: Self) -> Matrix<T> {
        let m1 = self.as_ref();
        let m2 = rhs.as_ref();
        if m1.dimensions().columns != m2.dimensions().rows {
            panic!("Cannot multiply matrices {} and {} because of dimensions", m1.dimensions(), m2.dimensions());
        }
        let row_count = m1.dimensions().rows;
        let col_count = m2.dimensions().columns;
        let mut result = Matrix::<T>::new(row_count, col_count);
        for row in 0..row_count {
            for col in 0..col_count {
                // result[row][col] = m1.row_iter(row).zip(m2.col_iter(col))
                //     .map(|pair| *pair.0 * *pair.1)
                //     .sum();
                let mut sum = T::default();
                for i in 0..m1.dimensions().columns {
                    sum += *m1.elm(row, i) * *m2.elm(i, col);
                }
                result[row][col] = sum;
            }
        }
        result
    }
}