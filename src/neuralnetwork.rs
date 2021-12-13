//! Simple multiplayer neural network using backpropagation for learning.

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

impl Network {

    pub fn evaluate_input_state(&mut self, input: Vector<ampl>) -> & Vector<ampl> {
        self.layers[0].stateVector = input;
        for i in 0..self.layers.len() - 1 {
            // self.layers[i + 1].stateVector = self.weights[i] * self.layers[i].stateVector;
        }
        &self.layers.last().unwrap().stateVector
    }
}

impl Network {
    pub fn new(dimensions: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        let mut weights = Vec::new();
        for dim in &dimensions {
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