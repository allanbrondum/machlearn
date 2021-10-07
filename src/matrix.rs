use std::ops::{Index, IndexMut, Neg, Add, AddAssign, SubAssign, Sub};
use std::fmt::{Display, Formatter, Write};

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

    pub fn column_count(&self) -> mdim {
        self.columns
    }

    pub fn row_count(&self) -> mdim {
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

impl<T> Display for Matrix<T>
    where T: Clone + PartialEq + Display
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.row_count() {
        f.write_str("|")?;
            for col in 0..self.column_count() {
                write!(f, "{}", self[row][col])?;
                if col != self.column_count() - 1 {
                    f.write_str(" ")?;
                }
            }
            f.write_str("|\n")?;
        }
        std::fmt::Result::Ok(())
    }
}

// impl<T> Neg for & Matrix<T>
//     where T: Copy + PartialEq + Neg<Output = T>
// {
//     type Output = Matrix<T>;
//
//     fn neg(self) -> Self::Output {
//         let mut clone = self.clone();
//         for elm in clone.elements.iter_mut() {
//             *elm = elm.neg();
//         }
//         clone
//     }
// }


impl<T> Neg for Matrix<T>
    where T: Copy + PartialEq + Neg<Output = T>
{
    type Output = Matrix<T>;

    fn neg(mut self) -> Self::Output {
        for elm in self.elements.iter_mut() {
            *elm = elm.neg();
        }
        self
    }
}

impl<T> Add for Matrix<T>
    where T: Copy + PartialEq + Add<Output = T>
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for elm_pair in self.elements.iter_mut().zip(rhs.elements.iter()) {
            *elm_pair.0 = *elm_pair.0 + *elm_pair.1;
        }
        self
    }
}

impl<T> AddAssign for Matrix<T>
    where T: Copy + PartialEq + AddAssign
{
    fn add_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 += *elm_pair.1;
        }
    }
}

impl<T> Sub for Matrix<T>
    where T: Copy + PartialEq + Sub<Output = T>
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for elm_pair in self.elements.iter_mut().zip(rhs.elements.iter()) {
            *elm_pair.0 = *elm_pair.0 - *elm_pair.1;
        }
        self
    }
}

impl<T> SubAssign for Matrix<T>
    where T: Copy + PartialEq + SubAssign
{
    fn sub_assign(&mut self, other: Self) {
        for elm_pair in self.elements.iter_mut().zip(other.elements.iter()) {
            *elm_pair.0 -= *elm_pair.1;
        }
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

    #[test]
    fn neg() {
        let mut a = Matrix::new(0., 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;

        let a = -a;

        assert_eq!(-1.1, a[0][0]);
        assert_eq!(-2.1, a[1][0]);
        assert_eq!(-3.1, a[0][1]);
        assert_eq!(-4.1, a[2][1]);

    }

    #[test]
    fn add() {
        let mut a = Matrix::new(0., 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;
        let mut b = Matrix::new(0., 3, 2);
        b[0][0] = 10.;
        b[1][0] = 20.;
        b[0][1] = 30.;
        b[2][1] = 40.;

        let a = a + b;

        assert_eq!(1.1 + 10., a[0][0]);
        assert_eq!(2.1 + 20., a[1][0]);
        assert_eq!(3.1 + 30., a[0][1]);
        assert_eq!(4.1 + 40., a[2][1]);
        assert_eq!(0., a[1][1]);

    }

    #[test]
    fn add_assign() {
        let mut a = Matrix::new(0., 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;
        let mut b = Matrix::new(0., 3, 2);
        b[0][0] = 10.;
        b[1][0] = 20.;
        b[0][1] = 30.;
        b[2][1] = 40.;

        a += b;

        assert_eq!(1.1 + 10., a[0][0]);
        assert_eq!(2.1 + 20., a[1][0]);
        assert_eq!(3.1 + 30., a[0][1]);
        assert_eq!(4.1 + 40., a[2][1]);
        assert_eq!(0., a[1][1]);

    }

    #[test]
    fn sub() {
        let mut a = Matrix::new(0., 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;
        let mut b = Matrix::new(0., 3, 2);
        b[0][0] = 10.;
        b[1][0] = 20.;
        b[0][1] = 30.;
        b[2][1] = 40.;

        let a = a - b;

        assert_eq!(1.1 - 10., a[0][0]);
        assert_eq!(2.1 - 20., a[1][0]);
        assert_eq!(3.1 - 30., a[0][1]);
        assert_eq!(4.1 - 40., a[2][1]);
        assert_eq!(0., a[1][1]);

    }

    #[test]
    fn sub_assign() {
        let mut a = Matrix::new(0., 3, 2);
        a[0][0] = 1.1;
        a[1][0] = 2.1;
        a[0][1] = 3.1;
        a[2][1] = 4.1;
        let mut b = Matrix::new(0., 3, 2);
        b[0][0] = 10.;
        b[1][0] = 20.;
        b[0][1] = 30.;
        b[2][1] = 40.;

        a -= b;

        assert_eq!(1.1 - 10., a[0][0]);
        assert_eq!(2.1 - 20., a[1][0]);
        assert_eq!(3.1 - 30., a[0][1]);
        assert_eq!(4.1 - 40., a[2][1]);
        assert_eq!(0., a[1][1]);

    }


}