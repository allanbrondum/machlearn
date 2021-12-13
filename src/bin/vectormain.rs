use machlearn::matrix::Matrix;
use std::ops::IndexMut;
use std::borrow::Borrow;
use machlearn::vector::Vector;

fn main() {

    let mut vec = Vector::new(5);
    vec[0] = 1;
    let b = vec[0];
}
