use std::ops::{Index, IndexMut, Neg, Add, AddAssign, SubAssign, Sub, Mul, MulAssign};
use std::fmt::{Display, Formatter, Write};
use std::slice::Iter;
use std::iter::Sum;
use crate::vector::Vector;
use crate::matrix::Matrix;
use crate::matrix::mdim;

pub type ampl = f64;

#[derive(Debug, Clone)]
pub struct Layer
{
    stateVector: Vector<ampl>
}

#[derive(Debug, Clone)]
pub struct Network
{
    layers: Vec<Layer>,
    weights: Vec<Matrix<ampl>>
}

struct TwoLayers<'a> {
    layer1: &'a mut Layer,
    layer2: &'a mut Layer,
    weights: &'a mut Matrix<ampl>
}

struct TwoLayersIter<'a>
{
    matrix: &'a mut Lay
    column: mdim,
    row: mdim
}

impl<T> ColIter<'_, T>
    where T: Clone + PartialEq
{
    fn new(matrix: &Matrix<T>, column: mdim) -> ColIter<T> {
        ColIter {
            matrix,
            column,
            row: 0
        }
    }
}

impl<'a, T> Iterator for ColIter<'a, T>
    where T: Clone + PartialEq
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if (self.row == self.matrix.rows) {
            None
        } else {
            let val = &self.matrix.elements[self.row * self.matrix.columns + self.column];
            self.row += 1;
            Some(val)
        }
    }
}

impl Network {

    fn two_layers(&mut self) -> impl Iterator<Item = &TwoLayers> {

    }

    pub fn evaluate_input_state(&mut self, p0: Vec<ampl>) -> Vec<ampl> {

    }
}

impl Network {
    pub fn new(dimensions: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        let mut weights = Vec::new();
        for dim in dimensions.iter() {
            layers.push(Layer{stateVector: Vector::new(*dim)});
        }
        for i in 0..dimensions.len() - 1 {
            weights.push(Matrix::new(dimensions[i], dimensions[i + 1]));
        }
        Network {
            layers,
            weights
        }
    }

    pub fn weights(&mut self, index: usize) -> &mut Matrix<ampl> {
        &mut self.weights[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::neuralnetwork::*;

    #[test]
    fn test() {

    }
}