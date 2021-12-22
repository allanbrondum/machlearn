//! Simple multilayer fully connected neural network using backpropagation for learning.

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

impl Layer {
    pub fn get_state(&self) -> &Vector<ampl> {
        &self.stateVector
    }
}


#[derive(Debug, Clone)]
pub struct Network
{
    layers: Vec<Layer>,
    weights: Vec<Matrix<ampl>>
}

impl Network {

    pub fn sigmoid(input: ampl) -> ampl {
        1. / (1. + (-input).exp())
    }

    pub fn evaluate_input_state(&mut self, input: Vector<ampl>) -> & Vector<ampl> {
        if input.len() != self.layers[0].stateVector.len() {
            panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers[0].stateVector.len())
        }
        self.layers[0].stateVector = input;
        for i in 0..self.layers.len() - 1 {
            self.layers[i + 1].stateVector = (&self.weights[i] * &self.layers[i].stateVector).apply(Network::sigmoid);
        }
        &self.layers.last().unwrap().stateVector
    }
}

impl Network {
    pub fn new(dimensions: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        let mut weights = Vec::new();
        for i in 0..dimensions.len() {
            layers.push(Layer{stateVector: Vector::new(dimensions[i])});
        }
        for i in 0..dimensions.len() - 1 {
            weights.push(Matrix::new(dimensions[i + 1], dimensions[i]));
        }
        Network {
            layers,
            weights
        }
    }

    pub fn get_weights(&self, index: usize) -> &Matrix<ampl> {
        &self.weights[index]
    }

    pub fn set_weights(&mut self, index: usize, weights: Matrix<ampl>) {
        if self.weights[index].dimensions() != weights.dimensions() {
            panic!("Dimensions of weights {} not as required by network {}", weights.dimensions(), self.weights[index].dimensions());
        }
        self.weights[index] = weights;
    }

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }
}

impl Display for Network
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Layers dimensions:\n");
        for layer in &self.layers {
            write!(f, "{}\n", layer.stateVector.len())?;
            // f.write_fmt(format_args!("{}\n", layer.stateVector.len()));
        }

        f.write_str("Weight dimensions:\n");
        for weight in &self.weights {
            write!(f, "{}\n", weight.dimensions())?;
        }

        std::fmt::Result::Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::neuralnetwork::*;

    #[test]
    fn test() {

    }
}