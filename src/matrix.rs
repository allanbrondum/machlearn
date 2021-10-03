
type mdim = usize;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Matrix<T>
    where T: Eq + Clone
{
    width: mdim,
    height: mdim,
    elements: Vec<T>
}

impl<T> Matrix<T>
    where T: Eq + Clone
{
    pub fn new(val: T, width: mdim, height: mdim) -> Matrix<T> {
        Matrix {
            width,
            height,
            elements: vec![val; width * height]
        }
    }

    pub fn width(&self) -> mdim {
        self.width
    }

    pub fn height(&self) -> mdim {
        self.height
    }
}