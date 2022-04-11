use std::marker::PhantomData;
use crate::matrix::{MatrixDimensions, MatrixElement, MatrixLinearIndex, MatrixT, StrideIter};
use serde::{Serialize, Deserialize};

pub struct MutSliceView<'a, T: MatrixElement> {
    inner: &'a mut [T],
    linear_index: MatrixLinearIndex,
}

impl<'a, T: MatrixElement> MutSliceView<'a, T> {
    pub fn new_row_stride(rows: usize,
               columns: usize,
               inner: &'a mut [T],
               row_stride: usize) -> MutSliceView<'a, T> {
        let linear_index = MatrixLinearIndex::new_row_stride(MatrixDimensions {rows, columns}, row_stride);
        Self::new(linear_index, inner)
    }

    pub fn new_col_stride(rows: usize,
                          columns: usize,
                          inner: &'a mut [T],
                          col_stride: usize) -> MutSliceView<'a, T> {

        let linear_index = MatrixLinearIndex::new_col_stride(MatrixDimensions {rows, columns}, col_stride);
        Self::new(linear_index, inner)
    }

    pub fn new(linear_index: MatrixLinearIndex, inner: &'a mut [T]) -> MutSliceView<'a, T> {
        assert_eq!(linear_index.required_length(), inner.len());
        MutSliceView {linear_index, inner}
    }
}

impl<'a, T: MatrixElement> MatrixT<'a, T> for MutSliceView<'a, T> {
    type ColIter = StrideIter<'a, T>;
    type RowIter = StrideIter<'a, T>;

    fn dimensions(&self) -> MatrixDimensions {
        self.linear_index.dimensions
    }

    fn elm(&self, row: usize, col:usize) -> &T {
        &self.inner[self.linear_index.lin_index(row, col)]
    }

    fn elm_mut(&mut self, row: usize, col:usize) -> &mut T {
        let index = self.linear_index.lin_index(row, col);
        &mut self.inner[index]
    }

    fn row_iter(&'a self, row: usize) -> Self::RowIter {
        let offset = self.linear_index.lin_index(row, 0);
        StrideIter::new(self.inner, offset, self.linear_index.col_stride, self.linear_index.dimensions.columns)
    }

    fn col_iter(&'a self, col: usize) -> Self::ColIter {
        let offset = self.linear_index.lin_index(0, col);
        StrideIter::new(self.inner, offset, self.linear_index.row_stride, self.linear_index.dimensions.rows)
    }
}

pub struct SliceView<'a, T: MatrixElement> {
    inner: &'a [T],
    linear_index: MatrixLinearIndex,
}

impl<'a, T: MatrixElement> SliceView<'a, T> {
    pub fn new_row_stride(rows: usize,
                          columns: usize,
                          inner: &'a [T],
                          row_stride: usize) -> SliceView<'a, T> {
        let linear_index = MatrixLinearIndex::new_row_stride(MatrixDimensions {rows, columns}, row_stride);
        Self::new(linear_index, inner)
    }

    pub fn new_col_stride(rows: usize,
                          columns: usize,
                          inner: &'a [T],
                          col_stride: usize) -> SliceView<'a, T> {

        let linear_index = MatrixLinearIndex::new_col_stride(MatrixDimensions {rows, columns}, col_stride);
        Self::new(linear_index, inner)
    }

    pub fn new(linear_index: MatrixLinearIndex, inner: &'a [T]) -> SliceView<'a, T> {
        assert!(linear_index.required_length() <= inner.len());
        SliceView {linear_index, inner}
    }
}

impl<'a, T: MatrixElement> MatrixT<'a, T> for SliceView<'a, T> {
    type ColIter = StrideIter<'a, T>;
    type RowIter = StrideIter<'a, T>;

    fn dimensions(&self) -> MatrixDimensions {
        self.linear_index.dimensions
    }

    fn elm(&self, row: usize, col:usize) -> &T {
        &self.inner[self.linear_index.lin_index(row, col)]
    }

    fn elm_mut(&mut self, row: usize, col:usize) -> &mut T {
        panic!("Not mutable");
    }

    fn row_iter(&'a self, row: usize) -> Self::RowIter {
        let offset = self.linear_index.lin_index(row, 0);
        StrideIter::new(self.inner, offset, self.linear_index.col_stride, self.linear_index.dimensions.columns)
    }

    fn col_iter(&'a self, col: usize) -> Self::ColIter {
        let offset = self.linear_index.lin_index(0, col);
        StrideIter::new(self.inner, offset, self.linear_index.row_stride, self.linear_index.dimensions.rows)
    }
}
