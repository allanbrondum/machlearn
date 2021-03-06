use crate::matrix::{MatrixDimensions, MatrixElement, MatrixLinearIndex, MatrixT, StrideIter};

/// Mutable matrix view of slice
pub struct MutSliceView<'a, T: MatrixElement> {
    inner: &'a mut [T],
    linear_index: MatrixLinearIndex,
}

impl<'a, T: MatrixElement> MutSliceView<'a, T> {
    pub fn new_row_stride(rows: usize,
               columns: usize,
               inner: &'a mut [T]) -> MutSliceView<'a, T> {
        let linear_index = MatrixLinearIndex::new_row_stride(MatrixDimensions {rows, columns});
        Self::new(linear_index, inner)
    }

    pub fn new_col_stride(rows: usize,
                          columns: usize,
                          inner: &'a mut [T]) -> MutSliceView<'a, T> {

        let linear_index = MatrixLinearIndex::new_col_stride(MatrixDimensions {rows, columns});
        Self::new(linear_index, inner)
    }

    pub fn new(linear_index: MatrixLinearIndex, inner: &'a mut [T]) -> MutSliceView<'a, T> {
        assert!(linear_index.required_linear_array_length() <= inner.len());
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
        &self.inner[self.linear_index.linear_index(row, col)]
    }

    fn elm_mut(&mut self, row: usize, col:usize) -> &mut T {
        let index = self.linear_index.linear_index(row, col);
        &mut self.inner[index]
    }

    fn row_iter(&'a self, row: usize) -> Self::RowIter {
        let offset = self.linear_index.linear_index(row, 0);
        StrideIter::new(self.inner, offset, self.linear_index.col_stride, self.linear_index.dimensions.columns)
    }

    fn col_iter(&'a self, col: usize) -> Self::ColIter {
        let offset = self.linear_index.linear_index(0, col);
        StrideIter::new(self.inner, offset, self.linear_index.row_stride, self.linear_index.dimensions.rows)
    }

}

/// Immutable matrix view of slice
pub struct SliceView<'a, T: MatrixElement> {
    inner: &'a [T],
    linear_index: MatrixLinearIndex,
}

impl<'a, T: MatrixElement> SliceView<'a, T> {
    pub fn new_row_stride(rows: usize,
                          columns: usize,
                          inner: &'a [T]) -> SliceView<'a, T> {
        let linear_index = MatrixLinearIndex::new_row_stride(MatrixDimensions {rows, columns});
        Self::new(linear_index, inner)
    }

    pub fn new_col_stride(rows: usize,
                          columns: usize,
                          inner: &'a [T]) -> SliceView<'a, T> {

        let linear_index = MatrixLinearIndex::new_col_stride(MatrixDimensions {rows, columns});
        Self::new(linear_index, inner)
    }

    pub fn new(linear_index: MatrixLinearIndex, inner: &'a [T]) -> SliceView<'a, T> {
        assert!(linear_index.required_linear_array_length() <= inner.len());
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
        &self.inner[self.linear_index.linear_index(row, col)]
    }

    fn elm_mut(&mut self, row: usize, col:usize) -> &mut T {
        panic!("Not mutable");
    }

    fn row_iter(&self, row: usize) -> Self::RowIter {
        let offset = self.linear_index.linear_index(row, 0);
        StrideIter::new(self.inner, offset, self.linear_index.col_stride, self.linear_index.dimensions.columns)
    }

    fn col_iter(&'a self, col: usize) -> Self::ColIter {
        let offset = self.linear_index.linear_index(0, col);
        StrideIter::new(self.inner, offset, self.linear_index.row_stride, self.linear_index.dimensions.rows)
    }

}
