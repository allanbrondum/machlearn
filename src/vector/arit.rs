use std::ops::{Neg, AddAssign, Sub, SubAssign, Add, Mul};
use crate::vector::{Vector, VectorT};
use crate::matrix::Matrix;
use std::iter::Sum;


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

impl<T> Mul for &Vector<T>
    where T: Copy + PartialEq + AddAssign + Mul<Output=T> + Default + 'static {
    type Output = T;

    fn mul(self, rhs: Self) -> T {
       VectorProduct::vec_prod(self, rhs)
    }
}

pub trait VectorProduct<T>
    where T: Copy + PartialEq + AddAssign + Mul<Output=T> + Default + 'static {

    fn vec_prod(self, rhs: Self) -> T;
}

impl<T, R: AsRef<dyn VectorT<T>>> VectorProduct<T> for &R
    where T: Copy + PartialEq + AddAssign + Mul<Output=T> + Default + 'static
{

    fn vec_prod(self, rhs: Self) -> T {
        let v1 = self.as_ref();
        let v2 = rhs.as_ref();
        if v1.len() != v2.len() {
            panic!("Vector 1 length {} not equal to vector 2 length {}", v1.len(), v2.len())
        }
        let mut sum = T::default();
        for i in 0..v1.len() {
            sum += v1[i] * v2[i];
        }
        sum
    }
}

impl<T> Mul<Vector<T>> for Matrix<T>
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + Sum
{
    type Output = Vector<T>;

    fn mul(self, rhs: Vector<T>) -> Vector<T> {
        &self * &rhs
    }
}

impl<T> Mul<&Vector<T>> for &Matrix<T>
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + Sum
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
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + Sum
{
    type Output = Vector<T>;

    fn mul(self, rhs: Matrix<T>) -> Vector<T> {
        &self * &rhs
    }
}

// impl<T, V: VectorT<T>> Mul<&Matrix<T>> for &V
//     where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + Sum
// {
//     type Output = Vector<T>;
//
//     fn mul(self, rhs: &Matrix<T>) -> Vector<T> {
//         if self.len() != rhs.row_count() {
//             panic!("Cannot multiply vector {} with matrix {} because of dimensions", self.len(), rhs.dimensions());
//         }
//         let mut result = Vector::<T>::new(rhs.column_count());
//         for column in 0..rhs.column_count() {
//             result[column] = &rhs.col(column) * self;
//         }
//         result
//     }
// }

impl<T> Mul<&Matrix<T>> for &Vector<T>
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + Sum
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