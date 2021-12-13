//! Matrix type and arithmetic operations on the Matrix.

use std::ops::{Index, IndexMut, Neg, Add, AddAssign, SubAssign, Sub, Mul, MulAssign};
use std::fmt::{Display, Formatter, Write};
use std::slice::Iter;
use std::iter::Sum;
use crate::vector::Vector;

pub type mdim = usize;

/// Matrix with arithmetic operations.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Matrix<T>
    where T: Clone + PartialEq
{
    rows: mdim,
    columns: mdim,
    elements: Vec<T>
}

struct ColIter<'a, T>
    where T: Clone + PartialEq
{
    matrix: &'a Matrix<T>,
    column: mdim,
    row: mdim
}

impl<T> ColIter<'_, T>
    where T: Clone + PartialEq
{
    fn new(matrix: &Matrix<T>, column: mdim) -> ColIter<T> {
        ColIter {
            matrix,
            column,
            row: 0
        }
    }
}

impl<'a, T> Iterator for ColIter<'a, T>
    where T: Clone + PartialEq
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if (self.row == self.matrix.rows) {
            None
        } else {
            let val = &self.matrix.elements[self.row * self.matrix.columns + self.column];
            self.row += 1;
            Some(val)
        }
    }
}

impl<T> Matrix<T>
    where T: Clone + PartialEq + Default
{

    pub fn new(rows: mdim, columns: mdim) -> Matrix<T> {
        Matrix {
            rows,
            columns,
            elements: vec![Default::default(); rows * columns]
        }
    }

    pub fn column_count(&self) -> mdim {
        self.columns
    }

    pub fn row_count(&self) -> mdim {
        self.rows
    }

    pub fn row_iter(&self, row: mdim) -> impl Iterator<Item = &T> {
        self[row].iter()
    }

    pub fn col_iter(&self, col: mdim) -> impl Iterator<Item = &T> {
        ColIter::new(self, col)
    }
}

impl<T> Index<mdim> for Matrix<T>
    where T: Clone + PartialEq + Default
{
    type Output = [T];

    fn index(&self, row_index: mdim) -> &[T] {
        &self.elements[row_index * self.columns..(row_index + 1) * self.columns]
    }
}

impl<T> IndexMut<mdim> for Matrix<T>
    where T: Clone + PartialEq + Default
{

    fn index_mut(&mut self, row_index: mdim) -> &mut [T] {
        &mut self.elements[row_index * self.columns..(row_index + 1) * self.columns]
    }
}

impl<T> Display for Matrix<T>
    where T: Clone + PartialEq + Default + Display
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.row_count() {
            f.write_str("|")?;
            for col in 0..self.column_count() {
                write!(f, "{}", self[row][col])?;
                if col != self.column_count() - 1 {
                    f.write_str(" ")?;
                }
            }
            f.write_str("|\n")?;
        }
        std::fmt::Result::Ok(())
    }
}

// impl<T> Neg for & Matrix<T>
//     where T: Copy + PartialEq + Neg<Output = T>
// {
//     type Output = Matrix<T>;
//
//     fn neg(self) -> Self::Output {
//         let mut clone = self.clone();
//         for elm in clone.elements.iter_mut() {
//             *elm = elm.neg();
//         }
//         clone
//     }
// }


impl<T> Neg for Matrix<T>
    where T: Copy + PartialEq + Default + Neg<Output = T>
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
    where T: Copy + PartialEq + Default + AddAssign
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T> AddAssign for Matrix<T>
    where T: Copy + PartialEq + Default + AddAssign
{
    fn add_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 += *elm_pair.1;
        }
    }
}

impl<T> Sub for Matrix<T>
    where T: Copy + PartialEq + Default + SubAssign
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T> SubAssign for Matrix<T>
    where T: Copy + PartialEq + Default + SubAssign
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
        if self.column_count() != rhs.row_count() {
            panic!("Cannot multiply matrices because of dimensions");
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

impl<T> Mul<Vector<T>> for Matrix<T>
    where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + Sum
{
    type Output = Vector<T>;

    fn mul(self, rhs: Vector<T>) -> Vector<T> {
        if self.column_count() != rhs.len() {
            panic!("Cannot multiply matrix with vector because of dimensions");
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

#[cfg(test)]
mod tests {
    use crate::matrix::*;

    #[test]
    fn equals() {
        let mut a = Matrix::new(3, 2);
        let mut b = Matrix::new(3, 2);
        let mut c = Matrix::new(2, 3);
        let mut d = Matrix::new(3, 2);
        d[0][0] = 1.;
        let mut e = Matrix::new(3, 2);
        e[0][0] = 1.;

        assert_eq!(a, a); // same instance
        assert_eq!(a, b); // equal
        assert_ne!(a, c); // different dimensions
        assert_ne!(a, d); // different values
        assert_eq!(d, e); // same values
    }

    #[test]
    fn index() {
        let mut a = Matrix::new( 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;

        assert_eq!(1.1, a[0][0]);
        assert_eq!(2.1, a[1][0]);
        assert_eq!(3.1, a[0][1]);
        assert_eq!(4.1, a[2][1]);

    }

    #[test]
    fn neg() {
        let mut a = Matrix::new( 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;

        let a = -a;

        assert_eq!(-1.1, a[0][0]);
        assert_eq!(-2.1, a[1][0]);
        assert_eq!(-3.1, a[0][1]);
        assert_eq!(-4.1, a[2][1]);

    }

    #[test]
    fn add() {
        let mut a = Matrix::new( 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;
        let mut b = Matrix::new( 3, 2);
        b[0][0] = 10.;
        b[1][0] = 20.;
        b[0][1] = 30.;
        b[2][1] = 40.;

        let a = a + b;

        assert_eq!(1.1 + 10., a[0][0]);
        assert_eq!(2.1 + 20., a[1][0]);
        assert_eq!(3.1 + 30., a[0][1]);
        assert_eq!(4.1 + 40., a[2][1]);
        assert_eq!(0., a[1][1]);

    }

    #[test]
    fn add_assign() {
        let mut a = Matrix::new( 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;
        let mut b = Matrix::new( 3, 2);
        b[0][0] = 10.;
        b[1][0] = 20.;
        b[0][1] = 30.;
        b[2][1] = 40.;

        a += b;

        assert_eq!(1.1 + 10., a[0][0]);
        assert_eq!(2.1 + 20., a[1][0]);
        assert_eq!(3.1 + 30., a[0][1]);
        assert_eq!(4.1 + 40., a[2][1]);
        assert_eq!(0., a[1][1]);

    }

    #[test]
    fn sub() {
        let mut a = Matrix::new( 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;
        let mut b = Matrix::new( 3, 2);
        b[0][0] = 10.;
        b[1][0] = 20.;
        b[0][1] = 30.;
        b[2][1] = 40.;

        let a = a - b;

        assert_eq!(1.1 - 10., a[0][0]);
        assert_eq!(2.1 - 20., a[1][0]);
        assert_eq!(3.1 - 30., a[0][1]);
        assert_eq!(4.1 - 40., a[2][1]);
        assert_eq!(0., a[1][1]);

    }

    #[test]
    fn sub_assign() {
        let mut a = Matrix::new( 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;
        let mut b = Matrix::new( 3, 2);
        b[0][0] = 10.;
        b[1][0] = 20.;
        b[0][1] = 30.;
        b[2][1] = 40.;

        a -= b;

        assert_eq!(1.1 - 10., a[0][0]);
        assert_eq!(2.1 - 20., a[1][0]);
        assert_eq!(3.1 - 30., a[0][1]);
        assert_eq!(4.1 - 40., a[2][1]);
        assert_eq!(0., a[1][1]);
    }

    #[test]
    fn row_iter() {
        let mut a = Matrix::new( 3, 2);
        a[0][0] = 1.1;
        a[0][1] = 2.1;
        a[1][0] = 3.1;
        a[1][1] = 4.1;

        let mut row_iter = a.row_iter(0);

        assert_eq!(Some(&1.1), row_iter.next());
        assert_eq!(Some(&2.1), row_iter.next());
        assert_eq!(None, row_iter.next());
        assert_eq!(None, row_iter.next());

        let mut row_iter = a.row_iter(1);

        assert_eq!(Some(&3.1), row_iter.next());
        assert_eq!(Some(&4.1), row_iter.next());
        assert_eq!(None, row_iter.next());
        assert_eq!(None, row_iter.next());
    }

    #[test]
    fn col_iter() {
        let mut a = Matrix::new( 3, 2);
        a[0][0] = 1.1;
        a[0][1] = 2.1;
        a[1][0] = 3.1;
        a[1][1] = 4.1;

        let mut col_iter = a.col_iter(0);

        assert_eq!(Some(&1.1), col_iter.next());
        assert_eq!(Some(&3.1), col_iter.next());
        assert_eq!(Some(&0.), col_iter.next());
        assert_eq!(None, col_iter.next());
        assert_eq!(None, col_iter.next());

        let mut col_iter = a.col_iter(1);

        assert_eq!(Some(&2.1), col_iter.next());
        assert_eq!(Some(&4.1), col_iter.next());
        assert_eq!(Some(&0.), col_iter.next());
        assert_eq!(None, col_iter.next());
        assert_eq!(None, col_iter.next());
    }

    #[test]
    fn multiply() {
        let mut a = Matrix::new( 3, 4);
        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;

        let mut b = Matrix::new( 4, 2);
        b[0][0] = 1;
        b[0][1] = 2;
        b[1][0] = 3;
        b[1][1] = 4;

        println!("{} {}", a, b);

        let mut product = Matrix::new( 3, 2);
        product[0][0] = 7;
        product[0][1] = 10;
        product[1][0] = 15;
        product[1][1] = 22;

        assert_eq!(product, a * b);
    }

    #[test]
    fn multiply_vector() {
        let mut a = Matrix::new( 3, 4);
        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;

        let mut b = Vector::new( 4);
        b[0] = 1;
        b[1] = 2;

        println!("{} {}", a, b);

        let mut product = Vector::new(3);
        product[0] = 5;
        product[1] = 11;
        product[2] = 0;

        assert_eq!(product, a * b);
    }

}