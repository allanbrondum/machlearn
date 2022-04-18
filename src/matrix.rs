//! Matrix type and arithmetic operations on the Matrix.

use std::ops::{Index, IndexMut, Neg, Add, AddAssign, SubAssign, Sub, Mul, MulAssign};
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
MulAssign +
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

    fn iter_enum(&'a self) -> AllElementsEnumeratedIter<'a, T, Self> where Self: Sized {
        AllElementsEnumeratedIter::new(self)
    }

    fn iter_mut_enum(&'a mut self) -> AllElementsEnumeratedMutIter<'a, T, Self> where Self: Sized {
        AllElementsEnumeratedMutIter::new(self)
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

    /// Add given matrix to the matrix (component wise addition).
    fn add_matrix_assign(&'a mut self, rhs: &'a impl MatrixT<'a, T>) where Self: Sized {
        if rhs.dimensions() != self.dimensions() {
            panic!("Cannot add matrix {} to matrix {} because of dimensions", rhs.dimensions(), self.dimensions());
        }
        for (index, elm) in self.iter_mut_enum() {
            *elm += *rhs.elm(index.0, index.1)
        }
    }

    /// Add given matrix to the matrix (component wise addition).
    fn mul_scalar_assign(&'a mut self, scalar: T) where Self: Sized {
        for (_, elm) in self.iter_mut_enum() {
            *elm *= scalar
        }
    }

    /// Transposed view of matrix
    fn as_transpose(&'a mut self) -> TransposedMatrixView<T, Self>
        where Self: Sized
    {
        TransposedMatrixView::new(self)
    }

    fn copy_to_matrix(&'a self) -> Matrix<T> where Self: Sized {
        let mut matrix = Matrix::new_with_dimension(self.dimensions());
        for (index, elm) in self.iter_enum() {
            matrix[(index.0, index.1)] = *elm;
        }
        matrix
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

pub struct AllElementsEnumeratedIter<'a, T: MatrixElement, M: MatrixT<'a, T>> {
    inner: &'a M,
    current_row_iter: M::RowIter,
    current_row: usize,
    current_col: usize,
}

impl<'a, T: MatrixElement, M: MatrixT<'a, T>> AllElementsEnumeratedIter<'a, T, M> {
    fn new(inner: &'a M) -> AllElementsEnumeratedIter<'a, T, M> {
        AllElementsEnumeratedIter {
            inner,
            current_row_iter: inner.row_iter(0),
            current_row: 0,
            current_col: 0,
        }
    }
}

impl<'a, T: MatrixElement, M: MatrixT<'a, T>> Iterator for AllElementsEnumeratedIter<'a, T, M> {
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

pub struct AllElementsEnumeratedMutIter<'a, T: MatrixElement + 'a, M: MatrixT<'a, T>> {
    inner: &'a mut M,
    current_row: usize,
    current_col: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: MatrixElement, M: MatrixT<'a, T>> AllElementsEnumeratedMutIter<'a, T, M> {
    fn new(inner: &'a mut M) -> AllElementsEnumeratedMutIter<'a, T, M> {
        Self {
            inner,
            current_row: 0,
            current_col: 0,
            _phantom: PhantomData
        }
    }
}

impl<'a, T: MatrixElement + 'a, M: MatrixT<'a, T>> Iterator for AllElementsEnumeratedMutIter<'a, T, M> {
    type Item = (MatrixIndex, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row == self.inner.row_count() {
            return None
        }
        let matrix_index = MatrixIndex(self.current_row, self.current_col);

        self.current_col += 1;
        if self.current_col == self.inner.column_count() {
            self.current_col = 0;
            self.current_row += 1;
        }

        let element_pointer: *mut T = self.inner.elm_mut(matrix_index.0, matrix_index.1);
        unsafe {
            // If we only hand out at most one reference to each element during the lifetime of the iterator
            // then the code should be safe. Seems like this the idiomatic way to implement mutable iterators (// https://stackoverflow.com/questions/63437935/in-rust-how-do-i-create-a-mutable-iterator)
            Some((matrix_index, &mut *element_pointer))
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
        &self.elements[self.linear_index.linear_index(row, col)]
    }

    fn elm_mut(&mut self, row: usize, col:usize) -> &mut T {
        &mut self.elements[self.linear_index.linear_index(row, col)]
    }

    fn row_iter(&'a self, row: usize) -> Self::RowIter {
        let offset = self.linear_index.linear_index(row, 0);
        StrideIter::new(self.elements.as_slice(), offset, self.linear_index.col_stride, self.linear_index.dimensions.columns)
    }

    fn col_iter(&'a self, col: usize) -> Self::ColIter {
        let offset = self.linear_index.linear_index(0, col);
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
    pub fn cell_count(&self) -> usize {
        self.rows * self.columns
    }

    pub const fn new(rows: usize, columns: usize) -> MatrixDimensions {
        MatrixDimensions{rows, columns}
    }
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
        Self::new_with_dimension(MatrixDimensions{rows, columns})
    }

    pub fn new_with_indexing(linear_index: MatrixLinearIndex) -> Matrix<T> {
        Matrix {
            linear_index: linear_index,
            elements: vec![T::default(); linear_index.required_linear_array_length()]
        }
    }

    pub fn new_with_dimension(dimension: MatrixDimensions) -> Matrix<T> {
        Matrix {
            linear_index: MatrixLinearIndex::new_row_stride(dimension),
            elements: vec![T::default(); dimension.cell_count()]
        }
    }

    pub fn new_from_elements(linear_index:  MatrixLinearIndex, elements: Vec<T>) -> Matrix<T> {
        assert!(linear_index.required_linear_array_length() <= elements.len(), "Required length {}, elements length {}", linear_index.required_linear_array_length(), elements.len());
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

    pub fn into_elements(self) -> Vec<T> {
        self.elements
    }

    pub fn as_slice(&self) -> &[T] {
        self.elements.as_slice()
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
    pub fn linear_index(&self, row: usize, col:usize) -> usize {
        assert!(row < self.dimensions.rows, "Row index {} out of bounds for number of rows {}", row, self.dimensions.rows);
        assert!(col < self.dimensions.columns, "Column index {} out of bounds for number of columns {}", col, self.dimensions.columns);
        self.linear_index_internal(row, col)
    }

    fn linear_index_internal(&self, row: usize, col: usize) -> usize {
        row * self.row_stride + col * self.col_stride + self.offset
    }

    /// Linear dimension length plus offset
    pub fn required_linear_array_length(&self) -> usize {
        self.linear_dimension_length() + self.offset
    }

    /// Linear dimension length
    pub fn linear_dimension_length(&self) -> usize {
        (self.dimensions.rows - 1) * self.row_stride + (self.dimensions.columns - 1) * self.col_stride + 1
    }

    pub fn transpose(self) -> MatrixLinearIndex {
        MatrixLinearIndex {dimensions: self.dimensions.transpose(), offset: self.offset,
            row_stride: self.col_stride, col_stride: self.row_stride}
    }

    pub fn add_slice_offset(self, offset_delta: usize) -> MatrixLinearIndex {
        MatrixLinearIndex {offset: self.offset + offset_delta, ..self}
    }

    pub fn add_row_col_offset(self, row_delta: usize, col_delta: usize) -> MatrixLinearIndex {
        MatrixLinearIndex {offset: self.linear_index_internal(row_delta, col_delta), ..self}
    }

    pub fn with_dimensions(self, new_dimensions: MatrixDimensions) -> MatrixLinearIndex {
        MatrixLinearIndex {dimensions: new_dimensions, ..self}
    }

    pub const fn new_row_stride(
        dimensions: MatrixDimensions) -> MatrixLinearIndex {
        MatrixLinearIndex {dimensions, row_stride: dimensions.columns, col_stride: 1, offset: 0}
    }

    pub const fn new_col_stride(
        dimensions: MatrixDimensions) -> MatrixLinearIndex {
        MatrixLinearIndex {dimensions, row_stride: 1, col_stride: dimensions.rows, offset: 0}
    }
}

#[cfg(test)]
mod tests;

