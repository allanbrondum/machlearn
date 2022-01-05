//! Matrix type and arithmetic operations on the Matrix.

use std::ops::{Index, IndexMut, Neg, Add, AddAssign, SubAssign, Sub, Mul, MulAssign};
use std::fmt::{Display, Formatter, Write};
use std::slice::Iter;
use std::iter::Sum;
use crate::vector::Vector;

pub mod arit;

pub trait MatrixElement:
Copy +
PartialEq +
AddAssign +
Add<Output=Self> +
Mul<Output=Self> +
Default +
Display +
Neg<Output=Self> +
SubAssign +
Sub<Output=Self> +
Sum +
'static {

}

impl MatrixElement for f64 {
}

impl MatrixElement for f32 {
}

impl MatrixElement for i32 {
}

impl MatrixElement for i64 {
}

/// Matrix with arithmetic operations.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Matrix<T>
    where T: MatrixElement
{
    dimensions: MatrixDimensions,
    elements: Vec<T>,
    row_stride: usize,
    col_stride: usize
}

impl<T> Matrix<T>
    where T: MatrixElement
{
    pub fn dimensions(&self) -> MatrixDimensions {
        self.dimensions
    }

    fn lin_index(&self, row: usize, col:usize) -> usize {
        row * self.row_stride + col * self.col_stride
    }

    pub fn elm(&self, row: usize, col:usize) -> &T {
        &self.elements[self.lin_index(row, col)]
    }

    pub fn elm_mut(&mut self, row: usize, col:usize) -> &mut T {
        let index = self.lin_index(row, col);
        &mut self.elements[index]
    }

    pub fn column_count(&self) -> usize {
        self.dimensions.columns
    }

    pub fn row_count(&self) -> usize {
        self.dimensions.rows
    }

    pub fn row_iter(&self, row: usize) -> impl Iterator<Item = &T> {
        todo!("implement");
        vec!().iter()
    }

    pub fn col_iter(&self, col: usize) -> impl Iterator<Item = &T> {
        todo!("implement");
        vec!().iter()
    }

    pub fn mat_mul(&self, rhs: &Matrix<T>) -> Matrix<T> {
        let m1 = self;
        let m2 = rhs;
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
                result[(row,col)] = sum;
            }
        }
        result
    }

    pub fn transpose(self) -> Matrix<T>
        where Self: Sized
    {
        Matrix {
            dimensions: MatrixDimensions {
                columns: self.dimensions.rows,
                rows: self.dimensions.columns
            },
            elements: self.elements,
            row_stride: self.col_stride,
            col_stride: self.row_stride
        }
    }

    pub fn as_transpose(self) -> Matrix<T>
        where Self: Sized
    {
        todo!("implement")
    }

    //
    // fn row<'a>(&'a self, row: usize) -> RowVector<T>
    //     where Self: Sized {
    //     RowVector {
    //         matrix: self,
    //         row
    //     }
    // }
    //
    // fn col<'a>(&'a self, col: usize) -> ColVector<T>
    //     where Self: Sized {
    //     ColVector {
    //         matrix: self,
    //         col
    //     }
    // }
}



#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct MatrixDimensions {
    pub rows: usize,
    pub columns: usize
}
//
// struct ColIter<'a, T>
//     where T: MatrixElement
// {
//     matrix: &'a Matrix<T>,
//     column: usize,
//     row: usize
// }
//
// impl<T> ColIter<'_, T>
//     where T: MatrixElement
// {
//     fn new(matrix: &Matrix<T>, column: usize) -> ColIter<T> {
//         ColIter {
//             matrix,
//             column,
//             row: 0
//         }
//     }
// }
//
// impl<'a, T> Iterator for ColIter<'a, T>
//     where T: MatrixElement
// {
//     type Item = &'a T;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.row == self.matrix.row_count() {
//             None
//         } else {
//             let val = &self.matrix.elements[self.row * self.matrix.column_count() + self.column];
//             self.row += 1;
//             Some(val)
//         }
//     }
// }

impl<T> Matrix<T>
    where T: MatrixElement
{
    pub fn new(rows: usize, columns: usize) -> Matrix<T> {
        Matrix {
            dimensions: MatrixDimensions { rows, columns },
            elements: vec![Default::default(); rows * columns],
            row_stride: columns,
            col_stride: 1
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
    where T: MatrixElement
{
    type Output = T;

    fn index(&self, row_col_index: (usize, usize)) -> &T {
        let lin_index = self.lin_index(row_col_index.0, row_col_index.1);
        &self.elements[lin_index]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
    where T: MatrixElement
{
    fn index_mut(&mut self, row_col_index: (usize, usize)) -> &mut T {
        let lin_index = self.lin_index(row_col_index.0, row_col_index.1);
        & mut self.elements[lin_index]
    }
}

impl<T> Display for Matrix<T>
    where T: MatrixElement
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.row_count() {
            f.write_str("|")?;
            for col in 0..self.column_count() {
                write!(f, "{}", self[(row,col)])?;
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

// pub struct StrideSlice<'a, T> {
//     slice: &'a [T],
//     stride: usize
// }
//
// pub struct StrideSliceMut<'a, T> {
//     slice: &'a mut [T],
//     stride: usize
// }
//
// impl<'a, T> Index<usize> for StrideSlice<'a, T>
//     where T: MatrixElement
// {
//     type Output = T;
//
//     fn index(&self, index: usize) -> &T {
//         &self.slice[index * self.stride]
//     }
// }
//
// impl<'a, T> Index<usize> for StrideSliceMut<'a, T>
//     where T: MatrixElement
// {
//     type Output = T;
//
//     fn index(&self, index: usize) -> &T {
//         &self.slice[index * self.stride]
//     }
// }
//
// impl<'a, T> IndexMut<usize> for StrideSliceMut<'a, T>
//     where T: MatrixElement
// {
//     fn index_mut(&mut self, index: usize) -> &mut T {
//         let lin_index = index * self.stride;
//         &mut self.slice[lin_index]
//     }
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
        d[(0,0)] = 1.;
        let mut e = Matrix::new(3, 2);
        e[(0,0)] = 1.;

        assert_eq!(a, a); // same instance
        assert_eq!(a, b); // equal
        assert_ne!(a, c); // different dimensions
        assert_ne!(a, d); // different values
        assert_eq!(d, e); // same values
    }

    #[test]
    fn index() {
        let mut a = Matrix::new( 3, 2);
        a[(0,0)] = 1.1;
        a[(1,0)] = 2.1;
        a[(0,1)] = 3.1;
        a[(2,1)] = 4.1;

        assert_eq!(1.1, a[(0,0)]);
        assert_eq!(2.1, a[(1,0)]);
        assert_eq!(3.1, a[(0,1)]);
        assert_eq!(4.1, a[(2,1)]);

    }

    #[test]
    fn matrix_with_col_stride() {
        let mut a = Matrix {
            dimensions: MatrixDimensions { rows: 3, columns: 2},
            elements: vec!(1.1, 2.1, 0.0, 3.1, 0.0, 4.1),
            row_stride: 1,
            col_stride: 3
        };
        a[(0,0)] = 1.1;
        a[(1,0)] = 2.1;
        a[(0,1)] = 3.1;
        a[(2,1)] = 4.1;

        assert_eq!(1.1, a[(0,0)]);
        assert_eq!(2.1, a[(1,0)]);
        assert_eq!(0.0, a[(2,0)]);
        assert_eq!(3.1, a[(0,1)]);
        assert_eq!(0.0, a[(1,1)]);
        assert_eq!(4.1, a[(2,1)]);
        assert_eq!(1.1, *a.elm(0,0));
        assert_eq!(2.1, *a.elm(1,0));
        assert_eq!(0.0, *a.elm(2,0));
        assert_eq!(3.1, *a.elm(0,1));
        assert_eq!(0.0, *a.elm(1,1));
        assert_eq!(4.1, *a.elm(2,1));

    }

    #[test]
    fn matrix_with_row_stride() {
        let mut a = Matrix {
            dimensions: MatrixDimensions { rows: 3, columns: 2},
            elements: vec!(1.1, 3.1, 2.1, 0.0, 0.0, 4.1),
            row_stride: 2,
            col_stride: 1
        };
        a[(0,0)] = 1.1;
        a[(1,0)] = 2.1;
        a[(0,1)] = 3.1;
        a[(2,1)] = 4.1;

        assert_eq!(1.1, a[(0,0)]);
        assert_eq!(2.1, a[(1,0)]);
        assert_eq!(0.0, a[(2,0)]);
        assert_eq!(3.1, a[(0,1)]);
        assert_eq!(0.0, a[(1,1)]);
        assert_eq!(4.1, a[(2,1)]);
        assert_eq!(1.1, *a.elm(0,0));
        assert_eq!(2.1, *a.elm(1,0));
        assert_eq!(0.0, *a.elm(2,0));
        assert_eq!(3.1, *a.elm(0,1));
        assert_eq!(0.0, *a.elm(1,1));
        assert_eq!(4.1, *a.elm(2,1));

    }

    #[test]
    fn neg() {
        let mut a = Matrix::new( 3, 2);
        a[(0,0)] = 1.1;
        a[(1,0)] = 2.1;
        a[(0,1)] = 3.1;
        a[(2,1)] = 4.1;

        let a = -a;

        assert_eq!(-1.1, a[(0,0)]);
        assert_eq!(-2.1, a[(1,0)]);
        assert_eq!(-3.1, a[(0,1)]);
        assert_eq!(-4.1, a[(2,1)]);

    }

    #[test]
    fn add() {
        let mut a = Matrix::new( 3, 2);
        a[(0,0)] = 1.1;
        a[(1,0)] = 2.1;
        a[(0,1)] = 3.1;
        a[(2,1)] = 4.1;
        let mut b = Matrix::new( 3, 2);
        b[(0,0)] = 10.;
        b[(1,0)] = 20.;
        b[(0,1)] = 30.;
        b[(2,1)] = 40.;

        let a = a + b;

        assert_eq!(1.1 + 10., a[(0,0)]);
        assert_eq!(2.1 + 20., a[(1,0)]);
        assert_eq!(3.1 + 30., a[(0,1)]);
        assert_eq!(4.1 + 40., a[(2,1)]);
        assert_eq!(0., a[(1,1)]);

    }

    #[test]
    fn add_assign() {
        let mut a = Matrix::new( 3, 2);
        a[(0,0)] = 1.1;
        a[(1,0)] = 2.1;
        a[(0,1)] = 3.1;
        a[(2,1)] = 4.1;
        let mut b = Matrix::new( 3, 2);
        b[(0,0)] = 10.;
        b[(1,0)] = 20.;
        b[(0,1)] = 30.;
        b[(2,1)] = 40.;

        a += b;

        assert_eq!(1.1 + 10., a[(0,0)]);
        assert_eq!(2.1 + 20., a[(1,0)]);
        assert_eq!(3.1 + 30., a[(0,1)]);
        assert_eq!(4.1 + 40., a[(2,1)]);
        assert_eq!(0., a[(1,1)]);

    }

    #[test]
    fn sub() {
        let mut a = Matrix::new( 3, 2);
        a[(0,0)] = 1.1;
        a[(1,0)] = 2.1;
        a[(0,1)] = 3.1;
        a[(2,1)] = 4.1;
        let mut b = Matrix::new( 3, 2);
        b[(0,0)] = 10.;
        b[(1,0)] = 20.;
        b[(0,1)] = 30.;
        b[(2,1)] = 40.;

        let a = a - b;

        assert_eq!(1.1 - 10., a[(0,0)]);
        assert_eq!(2.1 - 20., a[(1,0)]);
        assert_eq!(3.1 - 30., a[(0,1)]);
        assert_eq!(4.1 - 40., a[(2,1)]);
        assert_eq!(0., a[(1,1)]);

    }

    #[test]
    fn sub_assign() {
        let mut a = Matrix::new( 3, 2);
        a[(0,0)] = 1.1;
        a[(1,0)] = 2.1;
        a[(0,1)] = 3.1;
        a[(2,1)] = 4.1;
        let mut b = Matrix::new( 3, 2);
        b[(0,0)] = 10.;
        b[(1,0)] = 20.;
        b[(0,1)] = 30.;
        b[(2,1)] = 40.;

        a -= b;

        assert_eq!(1.1 - 10., a[(0,0)]);
        assert_eq!(2.1 - 20., a[(1,0)]);
        assert_eq!(3.1 - 30., a[(0,1)]);
        assert_eq!(4.1 - 40., a[(2,1)]);
        assert_eq!(0., a[(1,1)]);
    }

    #[test]
    fn row_iter() {
        let mut a = Matrix::new( 3, 2);
        a[(0,0)] = 1.1;
        a[(0,1)] = 2.1;
        a[(1,0)] = 3.1;
        a[(1,1)] = 4.1;

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
        a[(0,0)] = 1.1;
        a[(0,1)] = 2.1;
        a[(1,0)] = 3.1;
        a[(1,1)] = 4.1;

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
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

        let mut b = Matrix::new( 4, 2);
        b[(0,0)] = 1;
        b[(0,1)] = 2;
        b[(1,0)] = 3;
        b[(1,1)] = 4;

        println!("{} {}", a, b);

        let mut product = Matrix::new( 3, 2);
        product[(0,0)] = 7;
        product[(0,1)] = 10;
        product[(1,0)] = 15;
        product[(1,1)] = 22;

        assert_eq!(product, a * b);
    }

    #[test]
    fn multiply2() {
        let mut a = Matrix::new( 3, 4);
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

        let mut b = Matrix::new( 4, 2);
        b[(0,0)] = 1;
        b[(0,1)] = 2;
        b[(1,0)] = 3;
        b[(1,1)] = 4;

        println!("{} {}", a, b);

        let mut product = Matrix::new( 3, 2);
        product[(0,0)] = 7;
        product[(0,1)] = 10;
        product[(1,0)] = 15;
        product[(1,1)] = 22;

        assert_eq!(product, a.mat_mul(&b));
    }

    #[test]
    fn multiply_refs() {
        let mut a = Matrix::new( 3, 4);
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

        let mut b = Matrix::new( 4, 2);
        b[(0,0)] = 1;
        b[(0,1)] = 2;
        b[(1,0)] = 3;
        b[(1,1)] = 4;

        println!("{} {}", a, b);

        let mut product = Matrix::new( 3, 2);
        product[(0,0)] = 7;
        product[(0,1)] = 10;
        product[(1,0)] = 15;
        product[(1,1)] = 22;

        assert_eq!(product, &a * &b);
    }

    #[test]
    fn multiply_vector() {
        let mut a = Matrix::new( 3, 4);
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

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
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

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
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

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
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

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

    // #[test]
    // fn col_vector() {
    //     let mut a = Matrix::new( 3, 4);
    //     a[(0,0)] = 1;
    //     a[(0,1)] = 2;
    //     a[(1,0)] = 3;
    //     a[(1,1)] = 4;
    //
    //     let b = a.col(1);
    //
    //     assert_eq!(3, b.len());
    //     assert_eq!(2, b[0]);
    //     assert_eq!(4, b[1]);
    //     assert_eq!(0, b[2]);
    // }
    //
    // #[test]
    // fn row_vector() {
    //     let mut a = Matrix::new( 3, 4);
    //     a[(0,0)] = 1;
    //     a[(0,1)] = 2;
    //     a[(1,0)] = 3;
    //     a[(1,1)] = 4;
    //
    //     let b = a.row(1);
    //
    //     assert_eq!(4, b.len());
    //     assert_eq!(3, b[0]);
    //     assert_eq!(4, b[1]);
    //     assert_eq!(0, b[2]);
    //     assert_eq!(0, b[3]);
    // }

    #[test]
    fn elm() {
        let mut a = Matrix::new( 3, 4);
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

        assert_eq!(1, *a.elm(0, 0));
        assert_eq!(2, *a.elm(0, 1));
        assert_eq!(3, *a.elm(1, 0));
        assert_eq!(4, *a.elm(1, 1));

    }

    #[test]
    fn transpose() {
        let mut a = Matrix::new( 3, 4);
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

        let t = a.transpose();

        assert_eq!(MatrixDimensions{rows: 4, columns: 3}, t.dimensions());
        assert_eq!(1, *t.elm(0, 0));
        assert_eq!(2, *t.elm(1, 0));
        assert_eq!(3, *t.elm(0, 1));
        assert_eq!(4, *t.elm(1, 1));
        assert_eq!(1, t[(0,0)]);
        assert_eq!(2, t[(1,0)]);
        assert_eq!(3, t[(0,1)]);
        assert_eq!(4, t[(1,1)]);

        let t2 = t.transpose();

        assert_eq!(MatrixDimensions{rows: 3, columns: 4}, t2.dimensions());
        assert_eq!(1, *t2.elm(0, 0));
        assert_eq!(2, *t2.elm(0, 1));
        assert_eq!(3, *t2.elm(1, 0));
        assert_eq!(4, *t2.elm(1, 1));
        assert_eq!(1, t2[(0,0)]);
        assert_eq!(2, t2[(0,1)]);
        assert_eq!(3, t2[(1,0)]);
        assert_eq!(4, t2[(1,1)]);

    }

    #[test]
    fn multiply_scalar() {
        let mut a = Matrix::new( 3, 4);
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

        let mut result = Matrix::new( 3, 4);
        result[(0,0)] = 2;
        result[(0,1)] = 4;
        result[(1,0)] = 6;
        result[(1,1)] = 8;

        a *= 2;

        assert_eq!(result, a);
    }
}