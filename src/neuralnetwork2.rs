//! Simple multilayer fully connected neural network using backpropagation of errors (gradient descent) for learning.

use std::fmt::{Display, Formatter};
use crate::vector::{Vector};
use crate::matrix::{Matrix};
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use std::path::{PathBuf, Path};
use std::fs;
use std::io::Write;
use std::io::BufRead;
use std::io::Read;
use std::ops::Mul;

pub type Ampl = f64;

#[derive(Debug, Clone)]
pub struct Layer
{
    backpropagation_gamma: Vector<Ampl>,
    weights: Matrix<Ampl>,
    output_state: Vector<Ampl>,
}

impl Layer {

    pub fn new(input_dimension: usize, output_dimension: usize) -> Layer {
        Layer {
            backpropagation_gamma: Vector::new((input_dimension)),
            weights: Matrix::new(input_dimension, output_dimension),
            output_state: Vector::new(output_dimension),
        }
    }

    pub fn get_state(&self) -> &Vector<Ampl> {
        &self.output_state
    }

    pub fn get_weights(&self) -> &Matrix<Ampl> {
        &self.weights
    }

    pub fn set_random_weights(&mut self) {
        let mut rng = rand::thread_rng();
        self.weights.apply_ref(|_| rng.gen_range(-1.0..1.0));
    }

    fn evaluate_input(&mut self, input: &Vector<Ampl>, sigmoid: fn(Ampl) -> Ampl) {
        self.output_state = self.evaluate_input_no_state_change(input, sigmoid);
    }

    fn evaluate_input_no_state_change(&self, input: &Vector<Ampl>, sigmoid: fn(Ampl) -> Ampl) -> Vector<Ampl> {
        if input.len() != self.get_input_dimension() {
            panic!("Input state length {} not equals to weights column count {}", input.len(), self.weights.dimensions().columns);
        }
        self.weights.mul_vector(input).apply(sigmoid)
    }

    fn get_input_dimension(&self) -> usize {
        self.weights.dimensions().columns
    }

    fn get_output_dimension(&self) -> usize {
        self.weights.dimensions().rows
    }

}

#[derive(Debug, Clone)]
pub struct Network
{
    input_state: Vector<Ampl>,
    layers: Vec<Layer>,
    sigmoid: fn(Ampl) -> Ampl,
    sigmoid_derived: fn(Ampl) -> Ampl,
    biases: bool,
}

impl Network {
    pub fn set_random_weights(&mut self) {
        for layer in &mut self.layers {
            layer.set_random_weights();
        }
    }

    pub fn copy_all_weights(&self) -> Vec<Matrix<Ampl>> {
        self.layers.iter().map(|layer| layer.get_weights().clone()).collect()
    }
}

impl Network {
    pub fn get_sigmoid(&self) -> fn(Ampl) -> Ampl {
        self.sigmoid
    }

    pub fn get_sigmoid_derived(&self) -> fn(Ampl) -> Ampl {
        self.sigmoid_derived
    }
}

pub fn sigmoid_logistic(input: Ampl) -> Ampl {
    1. / (1. + (-input).exp())
}

pub fn sigmoid_logistic_derived(input: Ampl) -> Ampl {
    (-input).exp() / (1. + (-input).exp()).powf(2.)
}

impl Network {

    pub fn evaluate_input_state(&mut self, input: Vector<Ampl>) {
        if input.len() != self.layers.first().unwrap().get_input_dimension() {
            panic!("Input state length {} not equals to first layer input length {}", input.len(), self.layers.first().unwrap().get_input_dimension())
        }
        self.input_state = input;
        // if self.biases {
        //     *self.layers[0].state.last() = 1.0;
        // }
        let mut state_iter = &self.input_state;
        for layer in &mut self.layers {
            layer.evaluate_input(state_iter, self.sigmoid);
            state_iter = layer.get_state();
            // if self.biases {
            //     *self.layers[i + 1].state.last() = 1.0;
            // }
        }

    }

    pub fn evaluate_input_no_state_change(&self, input: &Vector<Ampl>) -> Vector<Ampl> {
        if input.len() != self.layers.first().unwrap().get_input_dimension() {
            panic!("Input state length {} not equals to first layer input length {}", input.len(), self.layers.first().unwrap().get_input_dimension())
        }
        let mut state= input.clone();
        // if self.biases {
        //     *state.last() = 1.0;
        // }
        for layer in &self.layers {
            state = layer.evaluate_input_no_state_change(&state, self.sigmoid);
            // if self.biases {
            //     *state.last() = 1.0;
            // }
        }
        state
    }

    // pub fn backpropagate(&mut self, input: Vector<Ampl>, output: &Vector<Ampl>, ny: Ampl, print: bool) {
    //     if input.len() != self.layers.first().unwrap().state.len() {
    //         panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers.first().unwrap().state.len())
    //     }
    //     if output.len() != self.layers.last().unwrap().state.len() {
    //         panic!("Output state length {} not equals to last layer state vector length {}", output.len(), self.layers.last().unwrap().state.len())
    //     }
    //
    //     self.evaluate_input_state(input);
    //
    //     let mut output = output.clone();
    //     if self.biases {
    //         *output.last() = 1.0;
    //     }
    //
    //     if print {
    //         let diff = output.clone() - self.get_output();
    //         let errsqr = &diff * &diff;
    //         println!("errsqr: {:.4}", errsqr);
    //     }
    //
    //     // last connector
    //     let mut normalize = false;
    //     {
    //         let layer1 = &self.layers[self.layers.len() - 2];
    //         let layer2 = self.layers.last().unwrap();
    //         let last_connector = self.connectors.last_mut().unwrap();
    //         let tmp = &last_connector.weights * &layer1.state;
    //         if print {
    //             println!("");
    //         }
    //         for i in 0..last_connector.back_propagation_delta.len() {
    //             normalize |= (output[i] - layer2.state[i]).abs() > 0.5;
    //
    //             last_connector.back_propagation_delta[i] = -2. * (self.sigmoid_derived)(tmp[i]) * (output[i] - layer2.state[i]);
    //             if print {
    //                 println!("sigder {}: {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
    //                          i,
    //                          last_connector.back_propagation_delta[i],
    //                          output[i] - layer2.state[i],
    //                          (self.sigmoid_derived)(tmp[i]),
    //                          tmp[i],
    //                          output[i],
    //                          layer2.state[i]);
    //             }
    //         }
    //     }
    //     if print {
    //         println!("normalize: {}", normalize);
    //     }
    //
    //     // the other connectors
    //     for connector_index in (0..self.connectors.len() - 1).rev() {
    //         let next_connector = &self.connectors[connector_index + 1];
    //         let tmp2 = &next_connector.back_propagation_delta * &next_connector.weights;
    //         let layer1 = &self.layers[connector_index];
    //         let connector = &mut self.connectors[connector_index];
    //         let tmp1 = &connector.weights * &layer1.state;
    //         for i in 0..connector.back_propagation_delta.len() {
    //             connector.back_propagation_delta[i] = (self.sigmoid_derived)(tmp1[i]) * tmp2[i];
    //         }
    //     }
    //
    //     for connector_index in 0..self.connectors.len() {
    //         let connector = &mut self.connectors[connector_index];
    //         let layer1 = &self.layers[connector_index];
    //
    //         // if print {
    //         //     println!("backpropagation: {}", connector.back_propagation_delta);
    //         // }
    //
    //         let deltam = connector.back_propagation_delta.clone().to_matrix();
    //         let statem = layer1.state.clone().to_matrix();
    //         let statemt = statem.transpose();
    //         let mut tmp = deltam.mat_mul(&statemt);
    //         if normalize {
    //             let normsqr = tmp.scalar_prod(&tmp);
    //             if print {
    //                 println!("normsq: {:.4}", normsqr);
    //             }
    //             if normsqr == 0.0 {
    //                 tmp *= 1. / normsqr.sqrt();
    //             }
    //         }
    //         tmp *= -ny;
    //         connector.weights += tmp;
    //     }
    // }
}

impl Network {
    pub fn new_logistic_sigmoid(dimensions: Vec<usize>) -> Self {
        Network::new(dimensions, sigmoid_logistic, sigmoid_logistic_derived, false)
    }

    pub fn new_logistic_sigmoid_biases(dimensions: Vec<usize>) -> Self {
        Network::new(dimensions, sigmoid_logistic, sigmoid_logistic_derived, true)
    }

    pub fn new(
        dimensions: Vec<usize>,
        sigmoid: fn(Ampl) -> Ampl,
        sigmoid_derived: fn(Ampl) -> Ampl,
        biases: bool) -> Self
    {
        if dimensions.len() < 2 {
            panic!("Must have at least two dimensions, was {}", dimensions.len());
        }
        let mut layers = Vec::new();
        for i in 1..dimensions.len() {
            layers.push(Layer::new(dimensions[i - 1], dimensions[i]));
        }
        Network {
            input_state: Vector::new(dimensions[0]),
            layers,
            sigmoid,
            sigmoid_derived,
            biases
        }
    }

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    pub fn get_output(&self) -> &Vector<Ampl> {
        self.layers.last().unwrap().get_state()
    }

    pub fn get_layer_count(&self) -> usize {
        self.layers.len()
    }
}

impl Display for Network
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {

        write!(f, "input state: {}\n\n", self.input_state)?;

        for layer in self.layers.iter().enumerate() {
            write!(f, "layer {}\n{}", layer.0, layer.1)?;
        }

        std::fmt::Result::Ok(())
    }
}

impl Display for Layer
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "backpropagation\n{}", self.backpropagation_gamma)?;
        write!(f, "weights\n{}", self.weights)?;
        write!(f, "output state: {}\n", self.output_state)?;

        std::fmt::Result::Ok(())
    }
}
