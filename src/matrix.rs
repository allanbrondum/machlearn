//! Matrix type and arithmetic operations on the Matrix.

use std::ops::{Index, IndexMut, Neg, Add, AddAssign, SubAssign, Sub, Mul};
use std::fmt::{Display, Formatter};
use serde::{Serialize, Deserialize};
use std::iter::Sum;
use std::marker::PhantomData;
pub use crate::matrix::transposedview::TransposedMatrixView;
pub use crate::matrix::sliceview::SliceView;
use crate::vector::Vector;

/// Operator implementations for matrix
mod arit;
mod transposedview;
mod sliceview;

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

pub trait MatrixT<'a, T: MatrixElement> {
    type ColIter : Iterator<Item = &'a T> + 'a;
    type RowIter : Iterator<Item = &'a T> + 'a;

    fn dimensions(&self) -> MatrixDimensions;

    fn elm(&self, row: usize, col:usize) -> &T;

    fn elm_mut(&mut self, row: usize, col:usize) -> &mut T;

    fn column_count(&self) -> usize {
        self.dimensions().columns
    }

    fn row_count(&self) -> usize {
        self.dimensions().rows
    }

    fn row_iter(&'a self, row: usize) -> Self::RowIter;

    fn col_iter(&'a self, col: usize) -> Self::ColIter;

    fn iter(&'a self) -> AllElementsIter<'a, T, Self> where Self: Sized {
        AllElementsIter::new(self)
    }

    // fn iter_mut(&'a mut self) -> AllElementsIter<'a, T, Self> where Self: Sized {
    //     AllElementsMutIter::new(self)
    // }

    /// Matrix multiplication with another matrix
    fn mul_mat(&'a self, rhs: &'a impl MatrixT<'a, T>) -> Matrix<T> {
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
                result[(row, col)] = m1.row_iter(row).zip(m2.col_iter(col))
                    .map(|pair| *pair.0 * *pair.1)
                    .sum();
            }
        }
        result
    }

    /// Sum of component-wise multiplication (like vector dot product)
    fn scalar_prod(&'a self, rhs: &'a impl MatrixT<'a, T>) -> T {
        if self.dimensions() != rhs.dimensions() {
            panic!("Cannot make scalar product of matrices {} and {} because of dimensions", self.dimensions(), rhs.dimensions());
        }
        let mut result = T::default();
        for row in 0..self.row_count() {
            for col in 0..self.column_count() {
                result += *self.elm(row, col) * *rhs.elm(row, col);
            }
        }
        result
    }

    /// Multiply matrix with vector (from right hand side). Same as matrix multiplication considering
    /// the given vector as a matrix with a single column.
    fn mul_vector(&'a self, rhs: &Vector<T>) -> Vector<T> {
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

    /// Multiply matrix with vector from left hand side. Same as matrix multiplication considering
    /// the given vector as a matrix with a single row.
    fn mul_vector_lhs(&'a self, lhs: &Vector<T>) -> Vector<T> {
        if lhs.len() != self.row_count() {
            panic!("Cannot multiply vector {} with matrix {} because of dimensions", lhs.len(), self.dimensions());
        }
        let mut result = Vector::<T>::new(self.column_count());
        for column in 0..self.column_count() {
            result[column] = self.col_iter(column).zip(lhs.iter())
                .map(|pair| *pair.0 * *pair.1)
                .sum();
        }
        result
    }

    /// Transposed view of matrix
    fn as_transpose(&'a mut self) -> TransposedMatrixView<T, Self>
        where Self: Sized
    {
        TransposedMatrixView::new(self)
    }
}

pub struct AllElementsIter<'a, T: MatrixElement, M: MatrixT<'a, T>> {
    inner: &'a M,
    current_row_iter: M::RowIter,
    current_row: usize,
}

impl<'a, T: MatrixElement, M: MatrixT<'a, T>> AllElementsIter<'a, T, M> {
    fn new(inner: &'a M) -> AllElementsIter<'a, T, M> {
        AllElementsIter {
            inner,
            current_row_iter: inner.row_iter(0),
            current_row: 0,
        }
    }
}

impl<'a, T: MatrixElement, M: MatrixT<'a, T>> Iterator for AllElementsIter<'a, T, M> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(elm) = self.current_row_iter.next() {
            Some(elm)
        } else if self.current_row + 1 < self.inner.row_count() {
            self.current_row += 1;
            self.current_row_iter = self.inner.row_iter(self.current_row);
            self.current_row_iter.next()
        } else {
            None
        }
    }
}

/// Matrix with arithmetic operations.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Matrix<T>
    where T: MatrixElement
{
    linear_index: MatrixLinearIndex,
    elements: Vec<T>,
}

impl<'a, T> MatrixT<'a, T> for Matrix<T>
    where T: MatrixElement
{
    type ColIter = StrideIter<'a, T>;
    type RowIter = StrideIter<'a, T>;

    fn dimensions(&self) -> MatrixDimensions {
        self.linear_index.dimensions
    }

    fn elm(&self, row: usize, col:usize) -> &T {
        &self.elements[self.linear_index.lin_index(row, col)]
    }

    fn elm_mut(&mut self, row: usize, col:usize) -> &mut T {
        &mut self.elements[self.linear_index.lin_index(row, col)]
    }

    fn row_iter(&'a self, row: usize) -> Self::RowIter {
        let offset = self.linear_index.lin_index(row, 0);
        StrideIter::new(self.elements.as_slice(), offset, self.linear_index.col_stride, self.linear_index.dimensions.columns)
    }

    fn col_iter(&'a self, col: usize) -> Self::ColIter {
        let offset = self.linear_index.lin_index(0, col);
        StrideIter::new(self.elements.as_slice(), offset, self.linear_index.row_stride, self.linear_index.dimensions.rows)
    }

}

impl<T> Matrix<T>
    where T: MatrixElement
{

    pub fn transpose(self) -> Matrix<T>
        where Self: Sized
    {
        Matrix {
            linear_index: self.linear_index.transpose(),
            elements: self.elements,
        }
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
#[derive(Serialize, Deserialize)]
pub struct MatrixDimensions {
    pub rows: usize,
    pub columns: usize
}

impl MatrixDimensions {
    pub fn transpose(self) -> MatrixDimensions {
        MatrixDimensions {rows: self.columns, columns: self.rows}
    }
}

impl<T> Matrix<T>
    where T: MatrixElement
{
    pub fn new(rows: usize, columns: usize) -> Matrix<T> {
        Matrix {
            linear_index: MatrixLinearIndex::new_row_stride(
                MatrixDimensions { rows, columns },
                columns,
            ),
            elements: vec![Default::default(); rows * columns],
        }
    }

    pub fn new_from_elements(linear_index:  MatrixLinearIndex, elements: Vec<T>) -> Matrix<T> {
        assert!(linear_index.required_length() <= elements.len(), "Required length {}, elements length {}", linear_index.required_length(), elements.len());
        Matrix {
            linear_index: linear_index,
            elements,
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
    where T: MatrixElement
{
    type Output = T;

    fn index(&self, (row, column): (usize, usize)) -> &T {
        self.elm(row, column)
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
    where T: MatrixElement
{
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut T {
        self.elm_mut(row, column)
    }
}

impl<T> Display for Matrix<T>
    where T: MatrixElement
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.row_count() {
            f.write_str("|")?;
            for col in 0..self.column_count() {
                write!(f, "{:6.2}", self[(row,col)])?;
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

    /// Apply function to each component
    pub fn apply(self, func: impl FnMut(T) -> T) -> Self {
        let mut ret = self;
        ret.apply_ref(func);
        ret
    }

    /// Apply function to each component
    pub fn apply_ref(&mut self, mut func: impl FnMut(T) -> T) {
        for elm in &mut self.elements {
            *elm = func(*elm);
        }
    }
}

pub struct StrideIter<'a, T> {
    slice: &'a [T],
    offset: usize,
    len: usize,
    stride: usize,
    index: usize
}

impl<T> StrideIter<'_, T> {
    fn new(slice: &[T], offset: usize, stride: usize, len: usize) -> StrideIter<T> {
        StrideIter {
            slice,
            offset,
            len,
            stride,
            index: 0
        }
    }
}

impl<'a, T> Iterator for StrideIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let lin_index = self.offset + self.index * self.stride;
            self.index = self.index + 1;
            Some(&self.slice[lin_index])
        } else {
            None
        }
    }
}


#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[derive(Serialize, Deserialize)]
pub struct MatrixLinearIndex {
    pub dimensions: MatrixDimensions,
    pub row_stride: usize,
    pub col_stride: usize,
    pub offset: usize,
}

impl MatrixLinearIndex {
    pub fn lin_index(&self, row: usize, col:usize) -> usize {
        assert!(row < self.dimensions.rows, "Row index {} out of bounds for number of rows {}", row, self.dimensions.rows);
        assert!(col < self.dimensions.columns, "Column index {} out of bounds for number of columns {}", col, self.dimensions.columns);
        row * self.row_stride + col * self.col_stride + self.offset
    }

    pub fn required_length(&self) -> usize {
        (self.dimensions.rows - 1) * self.row_stride + (self.dimensions.columns - 1) * self.col_stride + self.offset + 1
    }

    pub fn transpose(self) -> MatrixLinearIndex {
        MatrixLinearIndex {dimensions: self.dimensions.transpose(), offset: self.offset,
            row_stride: self.col_stride, col_stride: self.row_stride}
    }

    pub fn new_row_stride(
        dimensions: MatrixDimensions,
        row_stride: usize) -> MatrixLinearIndex {
        MatrixLinearIndex {dimensions, row_stride, col_stride: 1, offset: 0}
    }

    pub fn new_col_stride(
        dimensions: MatrixDimensions,
        col_stride: usize) -> MatrixLinearIndex {
        MatrixLinearIndex {dimensions, row_stride: 1, col_stride, offset: 0}
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::*;
    use itertools::Itertools;
    use std::borrow::Borrow;
    use crate::vector::Vector;

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
        let mut a = Matrix::new_from_elements(
            MatrixLinearIndex::new_col_stride(MatrixDimensions{rows: 3, columns: 2}, 3),
            vec!(1.1, 2.1, 0.0, 3.1, 0.0, 4.1));

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

        let mut row_iter = a.row_iter(0);
        assert_eq!(Some(&1.1), row_iter.next());
        assert_eq!(Some(&3.1), row_iter.next());
        assert_eq!(None, row_iter.next());
        let mut row_iter = a.row_iter(1);
        assert_eq!(Some(&2.1), row_iter.next());
        assert_eq!(Some(&0.0), row_iter.next());
        assert_eq!(None, row_iter.next());
        let mut row_iter = a.row_iter(2);
        assert_eq!(Some(&0.0), row_iter.next());
        assert_eq!(Some(&4.1), row_iter.next());
        assert_eq!(None, row_iter.next());

        let mut col_iter = a.col_iter(0);
        assert_eq!(Some(&1.1), col_iter.next());
        assert_eq!(Some(&2.1), col_iter.next());
        assert_eq!(Some(&0.0), col_iter.next());
        assert_eq!(None, col_iter.next());
        let mut col_iter = a.col_iter(1);
        assert_eq!(Some(&3.1), col_iter.next());
        assert_eq!(Some(&0.0), col_iter.next());
        assert_eq!(Some(&4.1), col_iter.next());
        assert_eq!(None, col_iter.next());

    }

    #[test]
    fn matrix_with_row_stride() {
        let mut a = Matrix::new_from_elements(
            MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 3, columns: 2}, 2),
            vec!(1.1, 3.1, 2.1, 0.0, 0.0, 4.1));

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

        let mut row_iter = a.row_iter(0);
        assert_eq!(Some(&1.1), row_iter.next());
        assert_eq!(Some(&3.1), row_iter.next());
        assert_eq!(None, row_iter.next());
        let mut row_iter = a.row_iter(1);
        assert_eq!(Some(&2.1), row_iter.next());
        assert_eq!(Some(&0.0), row_iter.next());
        assert_eq!(None, row_iter.next());
        let mut row_iter = a.row_iter(2);
        assert_eq!(Some(&0.0), row_iter.next());
        assert_eq!(Some(&4.1), row_iter.next());
        assert_eq!(None, row_iter.next());

        let mut col_iter = a.col_iter(0);
        assert_eq!(Some(&1.1), col_iter.next());
        assert_eq!(Some(&2.1), col_iter.next());
        assert_eq!(Some(&0.0), col_iter.next());
        assert_eq!(None, col_iter.next());
        let mut col_iter = a.col_iter(1);
        assert_eq!(Some(&3.1), col_iter.next());
        assert_eq!(Some(&0.0), col_iter.next());
        assert_eq!(Some(&4.1), col_iter.next());
        assert_eq!(None, col_iter.next());

    }

    #[test]
    fn matrix_vector_view_with_col_stride() {
        let mut vec = vec!(1.1, 2.1, 0.0, 3.1, 0.0, 4.1);
        let mut a = SliceView::new_col_stride(
            3, 2,
            &mut vec,
            3);

        *a.elm_mut(0,0) = 1.1;
        *a.elm_mut(1,0) = 2.1;
        *a.elm_mut(0,1) = 3.1;
        *a.elm_mut(2,1) = 4.1;

        assert_eq!(1.1, *a.elm(0,0));
        assert_eq!(2.1, *a.elm(1,0));
        assert_eq!(0.0, *a.elm(2,0));
        assert_eq!(3.1, *a.elm(0,1));
        assert_eq!(0.0, *a.elm(1,1));
        assert_eq!(4.1, *a.elm(2,1));

        let mut row_iter = a.row_iter(0);
        assert_eq!(Some(&1.1), row_iter.next());
        assert_eq!(Some(&3.1), row_iter.next());
        assert_eq!(None, row_iter.next());
        let mut row_iter = a.row_iter(1);
        assert_eq!(Some(&2.1), row_iter.next());
        assert_eq!(Some(&0.0), row_iter.next());
        assert_eq!(None, row_iter.next());
        let mut row_iter = a.row_iter(2);
        assert_eq!(Some(&0.0), row_iter.next());
        assert_eq!(Some(&4.1), row_iter.next());
        assert_eq!(None, row_iter.next());

        let mut col_iter = a.col_iter(0);
        assert_eq!(Some(&1.1), col_iter.next());
        assert_eq!(Some(&2.1), col_iter.next());
        assert_eq!(Some(&0.0), col_iter.next());
        assert_eq!(None, col_iter.next());
        let mut col_iter = a.col_iter(1);
        assert_eq!(Some(&3.1), col_iter.next());
        assert_eq!(Some(&0.0), col_iter.next());
        assert_eq!(Some(&4.1), col_iter.next());
        assert_eq!(None, col_iter.next());

    }

    #[test]
    fn matrix_vector_view_with_row_stride() {
        let mut vec = vec!(1.1, 3.1, 2.1, 0.0, 0.0, 4.1);
        let mut a = SliceView::new_row_stride(
            3, 2,
            &mut vec,
            2);

        *a.elm_mut(0,0) = 1.1;
        *a.elm_mut(1,0) = 2.1;
        *a.elm_mut(0,1) = 3.1;
        *a.elm_mut(2,1) = 4.1;

        assert_eq!(1.1, *a.elm(0,0));
        assert_eq!(2.1, *a.elm(1,0));
        assert_eq!(0.0, *a.elm(2,0));
        assert_eq!(3.1, *a.elm(0,1));
        assert_eq!(0.0, *a.elm(1,1));
        assert_eq!(4.1, *a.elm(2,1));

        let mut row_iter = a.row_iter(0);
        assert_eq!(Some(&1.1), row_iter.next());
        assert_eq!(Some(&3.1), row_iter.next());
        assert_eq!(None, row_iter.next());
        let mut row_iter = a.row_iter(1);
        assert_eq!(Some(&2.1), row_iter.next());
        assert_eq!(Some(&0.0), row_iter.next());
        assert_eq!(None, row_iter.next());
        let mut row_iter = a.row_iter(2);
        assert_eq!(Some(&0.0), row_iter.next());
        assert_eq!(Some(&4.1), row_iter.next());
        assert_eq!(None, row_iter.next());

        let mut col_iter = a.col_iter(0);
        assert_eq!(Some(&1.1), col_iter.next());
        assert_eq!(Some(&2.1), col_iter.next());
        assert_eq!(Some(&0.0), col_iter.next());
        assert_eq!(None, col_iter.next());
        let mut col_iter = a.col_iter(1);
        assert_eq!(Some(&3.1), col_iter.next());
        assert_eq!(Some(&0.0), col_iter.next());
        assert_eq!(Some(&4.1), col_iter.next());
        assert_eq!(None, col_iter.next());

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
    fn all_elements_iter() {
        let mut a = Matrix::new(3, 2);
        a[(0, 0)] = 1.1;
        a[(0, 1)] = 2.1;
        a[(1, 0)] = 3.1;
        a[(1, 1)] = 4.1;

        let vecres: Vec<_> = a.iter().collect();

        assert_eq!(vec!(&1.1, &2.1, &3.1, &4.1, &0.0, &0.0), vecres);
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

        let mut col_iter = a.col_iter(1);
        // col_iter.take_while_ref()

        let r: &dyn Iterator<Item=&f64> = &col_iter;
        let r2 = col_iter.by_ref();
        let r: &dyn Iterator<Item=&f64> = &r2;
        // r2.take_while()
        // vec!(1).iter().as_ref().split()
        // let r: &dyn Iterator<Item=&f64> = &
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

        assert_eq!(product, a.mul_mat(&b));
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

        assert_eq!(product, a.clone() * b.clone());
        assert_eq!(product, a.mul_vector(&b));
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

        assert_eq!(product, b.clone() * a.clone());
        assert_eq!(product, a.mul_vector_lhs(&b));
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
    fn as_transpose() {
        let mut a = Matrix::new( 3, 4);
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

        let mut t = a.as_transpose();

        assert_eq!(MatrixDimensions{rows: 4, columns: 3}, t.dimensions());
        assert_eq!(1, *t.elm(0, 0));
        assert_eq!(2, *t.elm(1, 0));
        assert_eq!(3, *t.elm(0, 1));
        assert_eq!(4, *t.elm(1, 1));
        let col0: Vec<_> = t.col_iter(0).copied().collect();
        assert_eq!(vec!(1, 2, 0, 0), col0);
        let row0: Vec<_> = t.row_iter(0).copied().collect();
        assert_eq!(vec!(1, 3, 0), row0);

        let t2 = t.as_transpose();

        assert_eq!(MatrixDimensions{rows: 3, columns: 4}, t2.dimensions());
        assert_eq!(1, *t2.elm(0, 0));
        assert_eq!(2, *t2.elm(0, 1));
        assert_eq!(3, *t2.elm(1, 0));
        assert_eq!(4, *t2.elm(1, 1));
        let col0: Vec<_> = t2.col_iter(0).copied().collect();
        assert_eq!(vec!(1, 3, 0), col0);
        let row0: Vec<_> = t2.row_iter(0).copied().collect();
        assert_eq!(vec!(1, 2, 0, 0), row0);
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

    #[test]
    fn multiply_scalar2() {
        let mut a = Matrix::new( 3, 4);
        a[(0,0)] = 1.;
        a[(0,1)] = 2.;
        a[(1,0)] = 3.;
        a[(1,1)] = 4.;

        let mut result = Matrix::new( 3, 4);
        result[(0,0)] = 2.;
        result[(0,1)] = 4.;
        result[(1,0)] = 6.;
        result[(1,1)] = 8.;

        assert_eq!(result, 2. * a);
    }

    #[test]
    fn scalar_product() {
        let mut a = Matrix::new( 2, 3);
        a[(0,0)] = 1;
        a[(0,1)] = 2;
        a[(1,0)] = 3;
        a[(1,1)] = 4;

        let mut b = Matrix::new( 2, 3);
        b[(0,0)] = 3;
        b[(0,1)] = 2;
        b[(1,0)] = 2;
        b[(1,1)] = 2;

        let result = a.scalar_prod(&b);
        assert_eq!(result, 3 + 4 + 6 + 8);
    }

    #[test]
    fn apply() {
        let mut a = Matrix::new(2, 1);
        a[(0,0)] = 1;
        a[(1,0)] = 2;

        a = a.apply(|x| 2 * x);

        assert_eq!(2, a[(0,0)]);
        assert_eq!(4, a[(1,0)]);

        let mut c = 0;
        a = a.apply(|x| { c += 1; c * x});

        assert_eq!(2, a[(0,0)]);
        assert_eq!(8, a[(1,0)]);

        fn d(x: i32) -> i32 {
            x + 1
        }
        a = a.apply(d);

        assert_eq!(3, a[(0,0)]);
        assert_eq!(9, a[(1,0)]);

        a.apply_ref(|x| -x);

        assert_eq!(-3, a[(0,0)]);
        assert_eq!(-9, a[(1,0)]);
    }

}

