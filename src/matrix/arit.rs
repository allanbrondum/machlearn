use std::ops::{Neg, Add, AddAssign, Sub, SubAssign, Mul};
use crate::matrix::Matrix;
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
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + Sum
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        &self * &rhs
    }
}

impl<T> Mul for &Matrix<T>
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + Sum
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Self) -> Matrix<T> {
        if self.column_count() != rhs.row_count() {
            panic!("Cannot multiply matrices {} and {} because of dimensions", self.dimensions(), rhs.dimensions());
        }
        let row_count = self.row_count();
        let col_count = rhs.column_count();
        let mut result = Matrix::<T>::new(row_count, col_count);
        for row in 0..row_count {
            for col in 0..col_count {
                result[row][col] = self.row_iter(row).zip(rhs.col_iter(col))
                    .map(|pair| *pair.0 * *pair.1)
                    .sum();
            }
        }
        result
    }
}
