//! Simple multilayer fully connected neural network using backpropagation of errors (gradient descent) for learning.

use std::ops::{Index, IndexMut, Neg, Add, AddAssign, SubAssign, Sub, Mul, MulAssign};
use std::fmt::{Display, Formatter, Write};
use std::slice::Iter;
use std::iter::Sum;
use crate::vector::{Vector};
use crate::matrix::{Matrix};

pub type ampl = f64;

#[derive(Debug, Clone)]
pub struct Layer
{
    state: Vector<ampl>
}

#[derive(Debug, Clone)]
pub struct Connector
{
    weights: Matrix<ampl>,
    back_propagation_delta: Vector<ampl>
}

impl Layer {

    pub fn new(dimension: usize) -> Layer {
        Layer {
            state: Vector::new(dimension)
        }
    }

    pub fn get_state(&self) -> &Vector<ampl> {
        &self.state
    }
}

impl Connector {
    pub fn new(layer1_dimension: usize, layer2_dimension: usize) -> Connector {
        Connector {
            weights: Matrix::new(layer2_dimension, layer1_dimension),
            back_propagation_delta: Vector::new(layer2_dimension)
        }
    }

    pub fn get_weights(&self) -> &Matrix<ampl> {
        &self.weights
    }

    pub fn get_back_propagation_delta(&self) -> &Vector<ampl> {
        &self.back_propagation_delta
    }
}

#[derive(Debug, Clone)]
pub struct Network
{
    layers: Vec<Layer>,
    connectors: Vec<Connector>,
    sigmoid: fn(ampl) -> ampl,
    sigmoid_derived: fn(ampl) -> ampl
}

impl Network {
    pub fn get_sigmoid(&self) -> fn(ampl) -> ampl {
        self.sigmoid
    }

    pub fn get_sigmoid_derived(&self) -> fn(ampl) -> ampl {
        self.sigmoid_derived
    }
}

const ny: f64 = 0.1;

pub fn sigmoid_logistic(input: ampl) -> ampl {
    1. / (1. + (-input).exp())
}

pub fn sigmoid_logistic_derived(input: ampl) -> ampl {
    (-input).exp() / (1. + (-input).exp()).powf(2.)
}

impl Network {

    pub fn evaluate_input_state(&mut self, input: Vector<ampl>) {
        if input.len() != self.layers.first().unwrap().state.len() {
            panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers.first().unwrap().state.len())
        }
        self.layers[0].state = input;
        for i in 0..self.layers.len() - 1 {
            self.layers[i + 1].state = (&self.connectors[i].weights * &self.layers[i].state).apply(self.sigmoid);
        }
    }

    pub fn evaluate_input_no_state_change(&self, mut input: Vector<ampl>) -> Vector<ampl> {
        if input.len() != self.layers.first().unwrap().state.len() {
            panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers.first().unwrap().state.len())
        }
        for i in 0..self.layers.len() - 1 {
            input = (&self.connectors[i].weights * &input).apply(self.sigmoid);
        }
        input
    }

    pub fn backpropagate(&mut self, input: Vector<ampl>, output: &Vector<ampl>) {
        if input.len() != self.layers.first().unwrap().state.len() {
            panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers.first().unwrap().state.len())
        }
        if output.len() != self.layers.last().unwrap().state.len() {
            panic!("Output state length {} not equals to last layer state vector length {}", output.len(), self.layers.last().unwrap().state.len())
        }

        self.evaluate_input_state(input);

        // last connector
        {
            let layer1 = &self.layers[self.layers.len() - 2];
            let layer2 = self.layers.last().unwrap();
            let last_connector = self.connectors.last_mut().unwrap();
            let tmp = &last_connector.weights * &layer1.state;
            for i in 0..last_connector.back_propagation_delta.len() {
                last_connector.back_propagation_delta[i] = -2. * (self.sigmoid_derived)(tmp[i]) * (output[i] - layer2.state[i]);
            }
        }

        // the other connectors
        for connector_index in (0..self.connectors.len() - 1).rev() {
            let next_connector = &self.connectors[connector_index + 1];
            let tmp2 = &next_connector.back_propagation_delta * &next_connector.weights;
            let layer1 = &self.layers[connector_index];
            let connector = &mut self.connectors[connector_index];
            let tmp1 = &connector.weights * &layer1.state;
            for i in 0..connector.back_propagation_delta.len() {
                connector.back_propagation_delta[i] = (self.sigmoid_derived)(tmp1[i]) * tmp2[i];
            }
        }

        for connector_index in 0..self.connectors.len() {
            let connector = &mut self.connectors[connector_index];
            let layer1 = &self.layers[connector_index];

            let deltam = connector.back_propagation_delta.clone().to_matrix();
            let statem = layer1.state.clone().to_matrix();
            let statemt = statem.transpose();
            let mut tmp = deltam.mat_mul(&statemt);
            tmp *= -ny;
            connector.weights += tmp;

            // for i in 0..connector.weights.row_count() {
            //     for j in 0..connector.weights.column_count() {
            //         connector.weights[i][j] += - ny * layer1.state[j] * connector.back_propagation_delta[i];
            //     }
            // }
        }
    }
}

impl Network {
    pub fn new_logistic_sigmoid(dimensions: Vec<usize>) -> Self {
        Network::new(dimensions, sigmoid_logistic, sigmoid_logistic_derived)
    }

    pub fn new(dimensions: Vec<usize>, sigmoid: fn(ampl) -> ampl, sigmoid_derived: fn(ampl) -> ampl) -> Self {
        let mut layers = Vec::new();
        let mut connectors = Vec::new();
        for i in 0..dimensions.len() {
            layers.push(Layer::new(dimensions[i]));
        }
        for i in 0..dimensions.len() - 1 {
            connectors.push(Connector::new(dimensions[i], dimensions[i + 1]));
        }
        Network {
            layers,
            connectors,
            sigmoid,
            sigmoid_derived
        }
    }

    pub fn get_weights(&self, index: usize) -> &Matrix<ampl> {
        &self.connectors[index].weights
    }

    pub fn set_weights(&mut self, index: usize, weights: Matrix<ampl>) {
        if self.connectors[index].weights.dimensions() != weights.dimensions() {
            panic!("Dimensions of weights {} not as required by network {}", weights.dimensions(), self.connectors[index].weights.dimensions());
        }
        self.connectors[index].weights = weights;
    }

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    pub fn get_output(&self) -> &Vector<ampl> {
        self.layers.last().unwrap().get_state()
    }
}

impl Display for Network
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // f.write_str("Layers dimensions:\n");
        // for layer in &self.layers {
        //     write!(f, "{}\n", layer.stateVector.len())?;
        //     // f.write_fmt(format_args!("{}\n", layer.stateVector.len()));
        // }
        //
        // f.write_str("Weight dimensions:\n");
        // for weight in &self.weights {
        //     write!(f, "{}\n", weight.dimensions())?;
        // }

        f.write_str("Layers:\n");
        for layer in &self.layers {
            write!(f, "{}\n", layer.state.len())?;
            write!(f, "state: {}\n\n", layer.get_state());
        }

        f.write_str("Connectors:\n");
        for connector in &self.connectors {
            write!(f, "{}\n", connector.weights.dimensions())?;
            write!(f, "weights: {}", connector.weights);
            write!(f, "backprop delta: {}\n\n", connector.back_propagation_delta);
        }

        std::fmt::Result::Ok(())
    }
}

pub struct Sample(pub Vector<ampl>, pub Vector<ampl>);

pub fn run_learning_iterations(network: &mut Network, samples: impl Iterator<Item=Sample>) {
    for sample in samples {
        network.backpropagate(sample.0, &sample.1);
    }
}

pub fn run_test_iterations(network: &Network, samples: impl Iterator<Item=Sample>) -> ampl {
    let mut errsqr = 0.;
    let mut samples_count = 0;
    for sample in samples {
        let output = network.evaluate_input_no_state_change(sample.0);
        let diff= output - sample.1;
        errsqr += &diff * &diff;
        samples_count += 1;
    }
    errsqr / samples_count as ampl
}

pub fn run_learning_and_test_iterations(
    network: &mut Network,
    learning_samples: impl Iterator<Item=Sample>,
    test_samples: impl Iterator<Item=Sample>) -> ampl
{
    run_learning_iterations(network, learning_samples);
    run_test_iterations(network, test_samples)
}

#[cfg(test)]
mod tests {
    use crate::neuralnetwork::*;

    #[test]
    fn test() {

    }
}