use std::ops::{Index, IndexMut};

type mdim = usize;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Matrix<T>
    where T: Clone + PartialEq
{
    rows: mdim,
    columns: mdim,
    elements: Vec<T>
}

impl<T> Matrix<T>
    where T: Clone + PartialEq
{
    pub fn new(val: T, rows: mdim, columns: mdim) -> Matrix<T> {
        Matrix {
            rows,
            columns,
            elements: vec![val; rows * columns]
        }
    }

    pub fn columns(&self) -> mdim {
        self.columns
    }

    pub fn rows(&self) -> mdim {
        self.rows
    }
}

impl<T> Index<mdim> for Matrix<T>
    where T: Clone + PartialEq
{
    type Output = [T];

    fn index(&self, row_index: mdim) -> &[T] {
        &self.elements[row_index * self.columns..(row_index + 1) * self.columns]
    }
}

impl<T> IndexMut<mdim> for Matrix<T>
    where T: Clone + PartialEq
{

    fn index_mut(&mut self, row_index: mdim) -> &mut [T] {
        &mut self.elements[row_index * self.columns..(row_index + 1) * self.columns]
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::*;

    #[test]
    fn equals() {
        let mut a = Matrix::new(0., 3, 2);
        let mut b = Matrix::new(0., 3, 2);
        let mut c = Matrix::new(0., 2, 3);
        let mut d = Matrix::new(0., 3, 2);
        d[0][0] = 1.;
        let mut e = Matrix::new(0., 3, 2);
        e[0][0] = 1.;

        assert_eq!(a, a); // same instance
        assert_eq!(a, b); // equal
        assert_ne!(a, c); // different dimensions
        assert_ne!(a, d); // different values
        assert_eq!(d, e); // same values
    }

    #[test]
    fn index() {
        let mut a = Matrix::new(0., 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;

        assert_eq!(1.1, a[0][0]);
        assert_eq!(2.1, a[1][0]);
        assert_eq!(3.1, a[0][1]);
        assert_eq!(4.1, a[2][1]);

    }
}