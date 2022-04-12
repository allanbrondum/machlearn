use std::marker::PhantomData;
use crate::matrix::{MatrixDimensions, MatrixElement, MatrixT};

pub struct TransposedMatrixView<'a, T: MatrixElement, M: MatrixT<'a, T>> {
    inner: &'a mut M,
    _phantom: PhantomData<T>,
}

impl<'a, T: MatrixElement, M: MatrixT<'a, T>> TransposedMatrixView<'a, T, M> {
    pub(super) fn new(inner: &'a mut M) -> TransposedMatrixView<'a, T, M> {
        TransposedMatrixView{inner, _phantom: PhantomData}
    }
}

impl<'a, T: MatrixElement, M: MatrixT<'a, T>> MatrixT<'a, T> for TransposedMatrixView<'a, T, M> {
    type ColIter = M::RowIter;
    type RowIter = M::ColIter;

    fn dimensions(&self) -> MatrixDimensions {
        self.inner.dimensions().transpose()
    }

    fn elm(&self, row: usize, col: usize) -> &T {
        self.inner.elm(col, row)
    }

    fn elm_mut(&mut self, row: usize, col: usize) -> &mut T {
        self.inner.elm_mut(col, row)
    }

    fn row_iter(&'a self, row: usize) -> Self::RowIter {
        self.inner.col_iter(row)
    }

    fn col_iter(&'a self, col: usize) -> Self::ColIter {
        self.inner.row_iter(col)
    }

}
