use std::marker::PhantomData;
use crate::matrix::{MatrixDimensions, MatrixElement, MatrixT, StrideIter};

pub struct SliceView<'a, T: MatrixElement> {
    inner: &'a mut [T],
    dimensions: MatrixDimensions,
    row_stride: usize,
    col_stride: usize,
}

impl<'a, T: MatrixElement> SliceView<'a, T> {
    pub fn new(rows: usize,
               columns: usize,
               inner: &'a mut [T],
               row_stride: usize,
               col_stride: usize) -> SliceView<'a, T> {
        assert_eq!((rows - 1) * row_stride + (columns - 1) * col_stride + 1, inner.len());
        SliceView {inner, dimensions: MatrixDimensions {rows, columns}, row_stride, col_stride}
    }
}

impl<'a, T: MatrixElement> MatrixT<'a, T> for SliceView<'a, T> {
    type ColIter = StrideIter<'a, T>;
    type RowIter = StrideIter<'a, T>;

    fn dimensions(&self) -> MatrixDimensions {
        self.dimensions
    }

    fn elm(&self, row: usize, col:usize) -> &T {
        &self.inner[self.lin_index(row, col)]
    }

    fn elm_mut(&mut self, row: usize, col:usize) -> &mut T {
        let index = self.lin_index(row, col);
        &mut self.inner[index]
    }

    fn row_iter(&'a self, row: usize) -> Self::RowIter {
        let offset = self.lin_index(row, 0);
        StrideIter::new(self.inner, offset, self.col_stride, self.dimensions.columns)
    }

    fn col_iter(&'a self, col: usize) -> Self::ColIter {
        let offset = self.lin_index(0, col);
        StrideIter::new(self.inner, offset, self.row_stride, self.dimensions.rows)
    }
}

impl<T> SliceView<'_, T>
    where T: MatrixElement
{
    fn lin_index(&self, row: usize, col:usize) -> usize {
        row * self.row_stride + col * self.col_stride
    }

}

