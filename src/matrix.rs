//! Matrix type and arithmetic operations on the Matrix.

use std::ops::{Index, IndexMut, Neg, Add, AddAssign, SubAssign, Sub, Mul, MulAssign};
use std::fmt::{Display, Formatter, Write};
use std::slice::Iter;
use std::iter::Sum;
use crate::vector::{Vector, VectorT, vdim};

pub mod arit;

pub type mdim = usize;

/// Matrix with arithmetic operations.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Matrix<T>
    where T: Clone + PartialEq
{
    dimensions: MatrixDimensions,
    elements: Vec<T>
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct MatrixDimensions {
    rows: mdim,
    columns: mdim
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
    where T: Clone + PartialEq + Default
{
    pub fn new(rows: mdim, columns: mdim) -> Matrix<T> {
        Matrix {
            dimensions: MatrixDimensions { rows, columns },
            elements: vec![Default::default(); rows * columns]
        }
    }
}

impl<T> Matrix<T>
    where T: Clone + PartialEq
{
    pub fn column_count(&self) -> mdim {
        self.dimensions.columns
    }

    pub fn row_count(&self) -> mdim {
        self.dimensions.rows
    }

    pub fn dimensions(&self) -> MatrixDimensions {
        self.dimensions
    }

    pub fn row_iter(&self, row: mdim) -> impl Iterator<Item = &T> {
        self[row].iter()
    }

    pub fn col_iter(&self, col: mdim) -> impl Iterator<Item = &T> {
        ColIter::new(self, col)
    }

    pub fn row<'a>(&'a self, row: mdim) -> impl VectorT<T> + 'a {
        RowVector {
            matrix: &self,
            row
        }
    }

    pub fn col<'a>(&'a self, col: mdim) -> impl VectorT<T> + 'a {
        ColVector {
            matrix: &self,
            col
        }
    }
}

struct ColVector<'a, T>
    where T: Clone + PartialEq
{
    matrix: &'a Matrix<T>,
    col: mdim
}

impl<'a, T> Index<vdim> for ColVector<'a, T>
    where T: Clone + PartialEq
{
    type Output = T;

    fn index(&self, index: vdim) -> &T {
        &self.matrix[index][self.col]
    }
}

impl<'a, T> VectorT<T> for ColVector<'a, T>
    where T: Clone + PartialEq
{
    fn len(&self) -> vdim {
        self.matrix.row_count()
    }
}

struct RowVector<'a, T>
    where T: Clone + PartialEq
{
    matrix: &'a Matrix<T>,
    row: mdim
}

impl<'a, T> Index<vdim> for RowVector<'a, T>
    where T: Clone + PartialEq
{
    type Output = T;

    fn index(&self, index: vdim) -> &T {
        &self.matrix[self.row][index]
    }
}

impl<'a, T> VectorT<T> for RowVector<'a, T>
    where T: Clone + PartialEq
{
    fn len(&self) -> vdim {
        self.matrix.column_count()
    }
}

impl<T> Index<mdim> for Matrix<T>
    where T: Clone + PartialEq
{
    type Output = [T];

    fn index(&self, row_index: mdim) -> &[T] {
        &self.elements[row_index * self.column_count()..(row_index + 1) * self.column_count()]
    }
}

impl<T> IndexMut<mdim> for Matrix<T>
    where T: Clone + PartialEq
{
    fn index_mut(&mut self, row_index: mdim) -> &mut [T] {
        let col_count = self.column_count();
        &mut self.elements[row_index * col_count..(row_index + 1) * col_count]
    }
}

impl<T> Display for Matrix<T>
    where T: Clone + PartialEq + Display
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
}