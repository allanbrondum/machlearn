//! Matrix type and arithmetic operations on the Matrix.

use std::ops::{Index, IndexMut, Neg, Add, AddAssign, SubAssign, Sub, Mul};
use std::fmt::{Display, Formatter};
use serde::{Serialize, Deserialize};
use std::iter::{Map, Sum};
use std::marker::PhantomData;
pub use crate::matrix::transposedview::TransposedMatrixView;
pub use crate::matrix::sliceview::MutSliceView;
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

    fn iter_enum(&'a self) -> AllElementsEnummeratedIter<'a, T, Self> where Self: Sized {
        AllElementsEnummeratedIter::new(self)
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
            self.next()
        } else {
            None
        }
    }
}

pub struct AllElementsEnummeratedIter<'a, T: MatrixElement, M: MatrixT<'a, T>> {
    inner: &'a M,
    current_row_iter: M::RowIter,
    current_row: usize,
    current_col: usize,
}

impl<'a, T: MatrixElement, M: MatrixT<'a, T>> AllElementsEnummeratedIter<'a, T, M> {
    fn new(inner: &'a M) -> AllElementsEnummeratedIter<'a, T, M> {
        AllElementsEnummeratedIter {
            inner,
            current_row_iter: inner.row_iter(0),
            current_row: 0,
            current_col: 0,
        }
    }
}

impl<'a, T: MatrixElement, M: MatrixT<'a, T>> Iterator for AllElementsEnummeratedIter<'a, T, M> {
    type Item = (MatrixIndex, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(elm) = self.current_row_iter.next() {
            let enum_elm = (MatrixIndex(self.current_row, self.current_col), elm);
            self.current_col += 1;
            Some(enum_elm)
        } else if self.current_row + 1 < self.inner.row_count() {
            self.current_row += 1;
            self.current_col = 0;
            self.current_row_iter = self.inner.row_iter(self.current_row);
            self.next()
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

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[derive(Serialize, Deserialize)]
pub struct MatrixIndex(pub usize, pub usize);

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

impl Display for MatrixIndex
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.0, self.1)?;
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

    pub const fn new_row_stride(
        dimensions: MatrixDimensions,
        row_stride: usize) -> MatrixLinearIndex {
        MatrixLinearIndex {dimensions, row_stride, col_stride: 1, offset: 0}
    }

    pub const fn new_col_stride(
        dimensions: MatrixDimensions,
        col_stride: usize) -> MatrixLinearIndex {
        MatrixLinearIndex {dimensions, row_stride: 1, col_stride, offset: 0}
    }
}

#[cfg(test)]
mod tests;

