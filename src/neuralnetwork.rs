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

struct NeighbourLayers<'a> {
    layer1: &'a mut Layer,
    layer2: &'a mut Layer,
    weights: &'a mut Matrix<ampl>
}

struct NeighbourLayersIter<'a>
{
    network: &'a mut Network,
    index: usize
}

impl NeighbourLayersIter<'_>
{
    fn new(network: &mut Network) -> NeighbourLayersIter {
        NeighbourLayersIter {
            network,
            index: 0
        }
    }
}

impl<'a> Iterator for NeighbourLayersIter<'a>
{
    type Item = NeighbourLayers<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.network.layers.len() - 1 {
            None
        } else {
            let val = NeighbourLayers{
                layer1: &mut self.network.layers[self.index],
                layer2: &mut self.network.layers[self.index + 1],
                weights: &mut self.network.weights[self.index]
            };
            self.index += 1;
            Some(val)
        }
    }
}

impl Network {

    fn neighbour_layers_iter(&mut self) -> impl Iterator<Item = NeighbourLayers> {
        NeighbourLayersIter::new(self)
    }

    pub fn evaluate_input_state(&mut self, input: Vector<ampl>) -> Vector<ampl> {
        Vector::new(0)
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