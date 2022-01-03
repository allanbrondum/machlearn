//! Matrix type and arithmetic operations on the Matrix.

use std::ops::{Index, IndexMut, Neg, Add, AddAssign, SubAssign, Sub, Mul, MulAssign};
use std::fmt::{Display, Formatter, Write};
use std::slice::Iter;
use std::iter::Sum;
use crate::vector::{Vector, VectorT};

pub mod arit;

pub trait MatrixElement: Copy + PartialEq + AddAssign + Add<Output=Self> + Mul<Output=Self> + Default + Display + Neg<Output=Self> + SubAssign + Sub<Output=Self> + Sum + 'static {

}

impl MatrixElement for f64 {
}

impl MatrixElement for f32 {
}

impl MatrixElement for i32 {
}

impl MatrixElement for i64 {
}

pub trait MatrixT<T>
    where T: MatrixElement
{
    fn dimensions(&self) -> MatrixDimensions;

    fn elm(&self, row: usize, col:usize) -> &T;

    fn mat_mul(&self, rhs: &dyn AsRef<dyn MatrixT<T>>) -> Matrix<T> {
        let m1 = self;
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

/// Matrix with arithmetic operations.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Matrix<T>
    where T: MatrixElement
{
    dimensions: MatrixDimensions,
    elements: Vec<T>
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct MatrixDimensions {
    rows: usize,
    columns: usize
}

struct ColIter<'a, T>
    where T: MatrixElement
{
    matrix: &'a Matrix<T>,
    column: usize,
    row: usize
}

impl<T> ColIter<'_, T>
    where T: MatrixElement
{
    fn new(matrix: &Matrix<T>, column: usize) -> ColIter<T> {
        ColIter {
            matrix,
            column,
            row: 0
        }
    }
}

impl<'a, T> Iterator for ColIter<'a, T>
    where T: MatrixElement
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row == self.matrix.row_count() {
            None
        } else {
            let val = &self.matrix.elements[self.row * self.matrix.column_count() + self.column];
            self.row += 1;
            Some(val)
        }
    }
}

impl<T> Matrix<T>
    where T: MatrixElement
{
    pub fn new(rows: usize, columns: usize) -> Matrix<T> {
        Matrix {
            dimensions: MatrixDimensions { rows, columns },
            elements: vec![Default::default(); rows * columns]
        }
    }
}


impl<T> MatrixT<T> for Matrix<T>
    where T: MatrixElement
{
    fn dimensions(&self) -> MatrixDimensions {
        self.dimensions
    }

    fn elm(&self, row: usize, col: usize) -> &T {
        &self[row][col]
    }
}

impl<T> Matrix<T>
    where T: MatrixElement
{
    pub fn column_count(&self) -> usize {
        self.dimensions.columns
    }

    pub fn row_count(&self) -> usize {
        self.dimensions.rows
    }

    pub fn row_iter(&self, row: usize) -> impl Iterator<Item = &T> {
        self[row].iter()
    }

    pub fn col_iter(&self, col: usize) -> impl Iterator<Item = &T> {
        ColIter::new(self, col)
    }

    pub fn row<'a>(&'a self, row: usize) -> impl VectorT<T> + 'a {
        RowVector {
            matrix: &self,
            row
        }
    }

    pub fn col<'a>(&'a self, col: usize) -> impl VectorT<T> + 'a {
        ColVector {
            matrix: &self,
            col
        }
    }
}

struct ColVector<'a, T>
    where T: MatrixElement
{
    matrix: &'a Matrix<T>,
    col: usize
}

impl<'a, T> Index<usize> for ColVector<'a, T>
    where T: MatrixElement
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.matrix[index][self.col]
    }
}

impl<'a, T> VectorT<T> for ColVector<'a, T>
    where T: MatrixElement
{
    fn len(&self) -> usize {
        self.matrix.row_count()
    }
}

struct RowVector<'a, T>
    where T: MatrixElement
{
    matrix: &'a Matrix<T>,
    row: usize
}

impl<'a, T> Index<usize> for RowVector<'a, T>
    where T: MatrixElement
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.matrix[self.row][index]
    }
}

impl<'a, T> VectorT<T> for RowVector<'a, T>
    where T: MatrixElement
{
    fn len(&self) -> usize {
        self.matrix.column_count()
    }
}

impl<T> Index<usize> for Matrix<T>
    where T: MatrixElement
{
    type Output = [T];

    fn index(&self, row_index: usize) -> &[T] {
        &self.elements[row_index * self.column_count()..(row_index + 1) * self.column_count()]
    }
}

impl<T> IndexMut<usize> for Matrix<T>
    where T: MatrixElement
{
    fn index_mut(&mut self, row_index: usize) -> &mut [T] {
        let col_count = self.column_count();
        &mut self.elements[row_index * col_count..(row_index + 1) * col_count]
    }
}

impl<T> Display for Matrix<T>
    where T: MatrixElement
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

impl Display for MatrixDimensions
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.rows, self.columns)?;
        std::fmt::Result::Ok(())
    }
}

impl<T> Matrix<T>
    where T: MatrixElement
{

    pub fn apply(self, func: fn(T) -> T) -> Self {
        let mut ret = self;
        for elm in &mut ret.elements {
            *elm = func(*elm);
        }
        ret
    }
}

impl<'a, T> AsRef<dyn MatrixT<T> + 'a> for Matrix<T>
    where T: MatrixElement
{

    fn as_ref(&self) -> &(dyn MatrixT<T> + 'a) {
        self
    }
}

struct TransposedMatrix<'a, T> {
    matrix: &'a dyn MatrixT<T>
}

impl<'a, T> MatrixT<T> for TransposedMatrix<'a, T>
    where T: MatrixElement
{
    fn dimensions(&self) -> MatrixDimensions {
        MatrixDimensions {
            rows: self.dimensions().columns,
            columns: self.dimensions().rows
        }
    }

    fn elm(&self, row: usize, col: usize) -> &T {
        self.elm(col, row)
    }
}

// impl<T, R: AsRef<dyn MatrixT<T>>> MatrixMultiplication<T> for &R
//     where T: Copy + PartialEq + Default + Mul<Output = T> + Add<Output = T> + AddAssign + Sum + 'static
// {
//
// }

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
    fn multiply2() {
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

        assert_eq!(product, a.mat_mul(&b));
    }

    #[test]
    fn multiply_refs() {
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

        assert_eq!(product, &a * &b);
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

    #[test]
    fn multiply_vector_refs() {
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

        assert_eq!(product, &a * &b);
    }

    #[test]
    fn multiply_vector_lhs() {
        let mut a = Matrix::new( 3, 4);
        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;

        let mut b = Vector::new( 3);
        b[0] = 1;
        b[1] = 2;

        println!("{} {}", a, b);

        let mut product = Vector::new(4);
        product[0] = 7;
        product[1] = 10;
        product[2] = 0;
        product[3] = 0;

        assert_eq!(product, b * a);
    }

    #[test]
    fn multiply_vector_lhs_refs() {
        let mut a = Matrix::new( 3, 4);
        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;

        let mut b = Vector::new( 3);
        b[0] = 1;
        b[1] = 2;

        println!("{} {}", a, b);

        let mut product = Vector::new(4);
        product[0] = 7;
        product[1] = 10;
        product[2] = 0;
        product[3] = 0;

        assert_eq!(product, &b * &a);
    }

    #[test]
    fn col_vector() {
        let mut a = Matrix::new( 3, 4);
        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;

        let b = a.col(1);

        assert_eq!(3, b.len());
        assert_eq!(2, b[0]);
        assert_eq!(4, b[1]);
        assert_eq!(0, b[2]);
    }

    #[test]
    fn row_vector() {
        let mut a = Matrix::new( 3, 4);
        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;

        let b = a.row(1);

        assert_eq!(4, b.len());
        assert_eq!(3, b[0]);
        assert_eq!(4, b[1]);
        assert_eq!(0, b[2]);
        assert_eq!(0, b[3]);
    }

    #[test]
    fn elm() {
        let mut a = Matrix::new( 3, 4);
        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;

        assert_eq!(1, *a.elm(0, 0));
        assert_eq!(2, *a.elm(0, 1));
        assert_eq!(3, *a.elm(1, 0));
        assert_eq!(4, *a.elm(1, 1));

    }


}