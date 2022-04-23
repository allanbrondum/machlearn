//! Vector type and arithmetic operations on the Vector.

use std::fmt::{Display, Formatter};
use serde::{Serialize, Deserialize};
use std::ops::{Index, IndexMut, Deref, DerefMut};
use itertools::Itertools;

use crate::matrix::{Matrix, MatrixDimensions, MatrixElement, MatrixLinearIndex, MatrixT, MutSliceView, SliceView};

/// Operator implementations for vector
mod arit;

/// Vector with arithmetic operations.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Vector<T>
    where T: MatrixElement
{
    elements: Vec<T>
}

impl<T> Vector<T>
    where T: MatrixElement
{
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn scalar_prod(&self, rhs: &Vector<T>) -> T {
        let v1 = self;
        let v2 = rhs;
        if v1.len() != v2.len() {
            panic!("Vector 1 length {} not equal to vector 2 length {}", v1.len(), v2.len())
        }
        v1.iter().zip(v2.iter())
            .map(|pair| *pair.0 * *pair.1)
            .sum()
    }

    /// Component wise multiplication
    pub fn mul_comp(&self, rhs: &Vector<T>) -> Vector<T> {
        let v1 = self;
        let v2 = rhs;
        if v1.len() != v2.len() {
            panic!("Vector 1 length {} not equal to vector 2 length {}", v1.len(), v2.len())
        }
        let vec: Vec<T> = v1.iter().zip(v2.iter())
            .map(|pair| *pair.0 * *pair.1)
            .collect();
        Vector::from_vec(vec)
    }

    pub fn add_vector_assign(&mut self, other: &Vector<T>) {
        if self.len() != other.len() {
            panic!("Vector lengths not equal: {} and {}", self.len(), other.len());
        }
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 += *elm_pair.1;
        }
    }

    pub fn add_vector(mut self, other: &Vector<T>) -> Vector<T> {
        self.add_vector_assign(other);
        self
    }

    pub fn sub_vector_assign(&mut self, other: &Vector<T>) {
        if self.len() != other.len() {
            panic!("Vector lengths not equal: {} and {}", self.len(), other.len());
        }
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 -= *elm_pair.1;
        }
    }

    pub fn sub_vector(mut self, other: &Vector<T>) -> Vector<T> {
        self.sub_vector_assign(other);
        self
    }

    pub fn iter(&self) -> impl Iterator<Item=&T> {
        self.elements.iter()
    }

    pub fn to_matrix(self) -> Matrix<T> {
        Matrix::new_from_elements(
            self.matrix_linear_index(),
            self.elements)
    }

    // pub fn as_matrix_mut(&mut self) -> impl MatrixT<T> { // TODO
    pub fn as_matrix_mut(&mut self) -> MutSliceView<'_, T> {
        MutSliceView::new(self.matrix_linear_index(), &mut self.elements)
    }

    // pub fn as_matrix<'a>(&'a self) -> impl MatrixT<'a, T> { // TODO
    pub fn as_matrix(&self) -> SliceView<'_, T> {
        SliceView::new(self.matrix_linear_index(), &self.elements)
    }

    fn matrix_linear_index(&self) -> MatrixLinearIndex {
        MatrixLinearIndex::new_col_stride(MatrixDimensions { rows: self.len(), columns: 1 })
    }

    pub fn push(&mut self, elm: T) {
        self.elements.push(elm);
    }

    pub fn pop(&mut self) -> T {
        self.elements.pop().unwrap()
    }

    pub fn last(&mut self) -> &mut T {
        self.elements.last_mut().unwrap()
    }

    pub fn as_slice(&self) -> &[T] {
        self.elements.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.elements.as_mut_slice()
    }

}

impl<T> Deref for Vector<T>
    where T: MatrixElement
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.elements.deref()
    }
}

impl<T> DerefMut for Vector<T>
    where T: MatrixElement
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.elements.deref_mut()
    }
}

impl<T> Vector<T>
    where T: MatrixElement
{
    pub fn new(len: usize) -> Vector<T> {
        Vector {
            elements: vec![Default::default(); len]
        }
    }

    pub fn from_vec(vec: Vec<T>) -> Vector<T> {
        Vector {
            elements: vec
        }
    }
}

impl<T> Vector<T>
    where T: MatrixElement
{
    /// Apply function to each component
    pub fn apply(self, mut func: impl FnMut(T) -> T) -> Self {
        let mut ret = self;
        ret.apply_ref(func);
        ret
    }

    pub fn apply_ref(&mut self, mut func: impl FnMut(T) -> T) {
        for elm in &mut self.elements {
            *elm = func(*elm);
        }
    }
}

impl<T> Index<usize> for Vector<T>
    where T: MatrixElement
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.elements[index]
    }
}

impl<T> IndexMut<usize> for Vector<T>
    where T: MatrixElement
{

    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.elements[index]
    }
}

impl<T> Display for Vector<T>
    where T: MatrixElement
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        write!(f,
               "{}",
               self.iter().format_with(
                   " ",
                   |elt, f| f(&format_args!("{:6.2}", elt))))?;
        f.write_str("]")?;
        std::fmt::Result::Ok(())
    }
}

#[cfg(test)]
mod tests;
