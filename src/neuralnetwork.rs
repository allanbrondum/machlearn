//! Simple multilayer fully connected neural network using backpropagation of errors (gradient descent) for learning.

use std::alloc::System;
use std::any::Any;
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use crate::vector::{Vector};
use crate::matrix::{Matrix, MatrixDimensions, MatrixLinearIndex, MatrixT, MutSliceView, SliceView};
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use std::path::{PathBuf, Path};
use std::fs;
use std::io::Write;
use std::io::BufRead;
use std::io::Read;
use std::ops::{Deref, Mul};
use itertools::{Itertools, MinMaxResult};
use rand::rngs::ThreadRng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
pub use fullyconnected::FullyConnectedLayer;
pub use convolutional::ConvolutionalLayer;

pub type Ampl = f64;

#[cfg(test)]
mod tests;

mod convolutional;
mod fullyconnected;

/// Sample tuple, .0: input, .1: output
pub struct Sample(pub Vector<Ampl>, pub Vector<Ampl>);

/// Feed forward neural network layer.
pub trait Layer : Display + Debug + Sync {
    fn get_weights(&self) -> Vec<&Matrix<Ampl>>;

    fn set_weights(&mut self, new_weights: Vec<Matrix<Ampl>>);

    /// Dimension of input state vector.
    fn get_input_dimension(&self) -> usize;

    /// Dimension of output state vector.
    fn get_output_dimension(&self) -> usize;

    fn set_random_weights(&mut self);

    fn set_random_weights_seed(&mut self, seed: u64);

    /// Evaluates the given input and returns the output of the layer without applying the activation function
    fn evaluate_input_without_activation(&self, input: &Vector<Ampl>) -> Vector<Ampl>;

    /// Use highest gradient back propagation to adjust weights moderated by the factor `ny`.
    /// The vector `delta_output` is the partial
    /// derivatives of error squared evaluated at the given `input` with respect to the layer
    /// output state coordinates before applying activation function
    /// (the dimension of `delta_output` is thus [`get_output_dimension`]). The method should return
    /// the partial derivatives of the error squared evaluated at the given `input` with respect to the input
    /// state coordinates (the dimension of the returned vector is thus [`get_input_dimension`]).
    fn back_propagate_without_activation(&mut self, input: &Vector<Ampl>, delta_output: Vector<Ampl>, ny: Ampl) -> Vector<Ampl>;

    fn as_any(&self) -> &dyn Any;
}

#[derive(Debug, Copy, Clone)]
pub struct ActivationFunction {
    activation_function: fn(Ampl) -> Ampl,
    activation_function_derived: fn(Ampl) -> Ampl,
}

impl ActivationFunction {
    pub fn sigmoid() -> Self {
        ActivationFunction {
            activation_function: sigmoid_logistic,
            activation_function_derived: sigmoid_logistic_derived,
        }
    }

    pub fn relu() -> Self {
        ActivationFunction {
            activation_function: relu,
            activation_function_derived: relu_derived,
        }
    }
}

#[derive(Debug)]
pub struct LayerContainer {
    layer: Box<dyn Layer>,
    activation_function: ActivationFunction,
}

impl LayerContainer {
    pub fn new(layer: Box<dyn Layer>, activation_function: ActivationFunction) -> Self {
        LayerContainer {layer, activation_function}
    }
}


#[derive(Debug)]
pub struct Network
{
    layers: Vec<LayerContainer>,
    biases: bool,
}

impl Network {
    pub fn set_random_weights(&mut self) {
        for layer in &mut self.layers {
            layer.layer.set_random_weights();
        }
    }

    pub fn set_random_weights_seed(&mut self, seed: u64) {
        for layer in &mut self.layers {
            layer.layer.set_random_weights_seed(seed);
        }
    }

    pub fn get_all_weights(&self) -> Vec<Vec<&Matrix<Ampl>>> {
        self.layers.iter().map(|layer| layer.layer.get_weights()).collect()
    }

    pub fn set_all_weights(&mut self, weights: Vec<Vec<Matrix<Ampl>>>) {
        if self.layers.len() != weights.len() {
            panic!("Number of layers {} does not equals weights length {}", self.layers.len(), weights.len());
        }
        for layer_weight in self.layers.iter_mut().zip(weights.into_iter()) {
            layer_weight.0.layer.set_weights(layer_weight.1);
        }
    }
}


impl Network {

    pub fn evaluate_input(&self, input: &Vector<Ampl>) -> Vector<Ampl> {
        if input.len() != self.layers.first().unwrap().layer.get_input_dimension() {
            panic!("Input state length {} not equals to first layer input length {}", input.len(), self.layers.first().unwrap().layer.get_input_dimension())
        }
        // evaluate states feed forward through layers
        let mut state= input.clone();
        if self.biases {
            *state.last() = 1.0;
        }
        for layer in &self.layers {
            state = layer.evaluate_input(&state);
            if self.biases {
                *state.last() = 1.0;
            }
        }
        state
    }

    pub fn back_propagate(&mut self, input: &Vector<Ampl>, expected_output: &Vector<Ampl>, ny: Ampl, print: bool) -> Ampl {
        if input.len() != self.layers.first().unwrap().layer.get_input_dimension() {
            panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers.first().unwrap().layer.get_input_dimension())
        }
        if expected_output.len() != self.layers.last().unwrap().layer.get_output_dimension() {
            panic!("Output state length {} not equals to last layer state vector length {}", expected_output.len(), self.layers.last().unwrap().layer.get_output_dimension())
        }

        // first evaluate states using feed forward
        let mut layer_input_states = Vec::new();
        let mut state= input.clone();
        if self.biases {
            *state.last() = 1.0;
        }
        for layer in &self.layers {
            let mut output = layer.evaluate_input(&state);
            if self.biases {
                *output.last() = 1.0;
            }
            layer_input_states.push(state);
            state = output;
        }

        // backpropagation
        let diff = (state.clone() - expected_output);
        let err_sqr = diff.scalar_prod(&diff);
        let mut gamma = 2. * (state - expected_output);
        for layer_input in self.layers.iter_mut().rev().zip(layer_input_states.iter().rev()) {
            let layer = layer_input.0;
            let input = layer_input.1;

            gamma = layer.back_propagate(input, gamma, ny);
        }
        err_sqr
    }
}

impl LayerContainer {

    /// Evaluates the given input and returns the output of the layer
    fn evaluate_input(&self, input: &Vector<Ampl>) -> Vector<Ampl> {
        if input.len() != self.layer.get_input_dimension() {
            panic!("Input state length {} not equals to weights column count {}", input.len(), self.layer.get_input_dimension());
        }
        self.layer.evaluate_input_without_activation(input).apply(self.activation_function.activation_function)
    }

    /// Use highest gradient back propagation to adjust weights moderated by the factor `ny`.
    /// The vector `gamma_output` is the partial
    /// derivatives of error squared evaluated at the given `input` with respect to the output state coordinates
    /// (the dimension of `gamma_output` is thus [`get_output_dimension`]). The method should return
    /// the partial derivatives of the error squared evaluated at the given `input` with respect to the input
    /// state coordinates (the dimension of the returned vector is thus [`get_input_dimension`]).
    fn back_propagate(&mut self, input: &Vector<Ampl>, gamma_output: Vector<Ampl>, ny: Ampl) -> Vector<Ampl> {
        // the delta vector is the partial derivative of error squared with respect to the layer output before the sigmoid function is applied
        let delta_output = self.layer.evaluate_input_without_activation(input).apply(self.activation_function.activation_function_derived).mul_comp(&gamma_output);

        self.layer.back_propagate_without_activation(&input, delta_output, ny)
    }
}

impl Network {
    pub fn new_fully_connected(dimensions: Vec<usize>) -> Self {
        Self::new_fully_connected_int(dimensions, false)
    }

    pub fn new_fully_connected_biases(dimensions: Vec<usize>) -> Self {
        Self::new_fully_connected_int(dimensions, true)
    }

    fn new_fully_connected_int(
        dimensions: Vec<usize>,
        biases: bool) -> Self
    {
        if dimensions.len() < 2 {
            panic!("Must have at least two dimensions, was {}", dimensions.len());
        }
        let mut layers = Vec::new();
        for i in 1..dimensions.len() {
            let boxed: Box<dyn Layer> = Box::new(FullyConnectedLayer::new(dimensions[i - 1], dimensions[i]));
            layers.push(LayerContainer::new(boxed, ActivationFunction::sigmoid()));
        }
        Self::new(layers, biases)
    }

    pub fn new(
        layers: Vec<LayerContainer>,
        biases: bool) -> Self
    {
        if layers.is_empty() {
            panic!("Must have at least one layer");
        }

        for (layer1, layer2) in layers.iter().tuple_windows() {
            if layer1.layer.get_output_dimension() != layer2.layer.get_input_dimension() {
                panic!("Layer output dimension {} and next layer input dimension {} does not match", layer1.layer.get_output_dimension(), layer2.layer.get_input_dimension());
            }
        }

        Network {
            layers,
            biases,
        }
    }

    pub fn layers(&self) -> impl Iterator<Item=&Box<dyn Layer>> {
        self.layers.iter().map(|it| &it.layer)
    }

    fn layers_mut(&mut self) -> impl Iterator<Item=&mut Box<dyn Layer>> {
        self.layers.iter_mut().map(|it| &mut it.layer)
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

impl Display for Network
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {

        for layer in self.layers.iter().enumerate() {
            write!(f, "layer {}\n{}", layer.0, layer.1.layer)?;
        }

        std::fmt::Result::Ok(())
    }
}

struct NetworkBackPropagateSampler {
    err_sqr: Ampl,
    count: usize,
    total_count: usize,
    axis_max: Option<Ampl>,
}

impl NetworkBackPropagateSampler {
    fn new() -> NetworkBackPropagateSampler {
        NetworkBackPropagateSampler{err_sqr:0.0, count:0, axis_max: None, total_count: 0}
    }

    fn single_iteration(&mut self, err_sqr: Ampl) {
        self.total_count += 1;
        self.count += 1;
        self.err_sqr += err_sqr;


    }

    fn clear_sample_batch(&mut self) {
        self.count = 0;
        self.err_sqr = 0.0;
    }

    fn mean_err_sqr_for_batch(&self) -> Ampl {
        self.err_sqr / self.count as Ampl
    }
}

pub fn run_learning_iterations(network: &mut Network, samples: impl Iterator<Item=Sample>, ny: Ampl, print: bool) {
    let start = Instant::now();
    println!("learning");

    const COUNT_SAMPLE: usize = 1000;
    let mut back_prop_sampler = NetworkBackPropagateSampler::new();
    for sample in samples {
        let err_sqr = network.back_propagate(&sample.0, &sample.1, ny, print);
        back_prop_sampler.single_iteration(err_sqr);

        if back_prop_sampler.count == COUNT_SAMPLE {
            print_learning_stats(network, &mut back_prop_sampler);
            back_prop_sampler.clear_sample_batch();
        }
    }

    let duration = start.elapsed();
    println!("duration {:.2?}", duration);
}

fn print_learning_stats(network: &Network, sampler: &mut NetworkBackPropagateSampler) {
    let err_sqr_mean = sampler.err_sqr / sampler.count as Ampl;
    if sampler.axis_max.is_none() {
        sampler.axis_max.replace(err_sqr_mean);
    }

    const AXIS_CHARS: usize = 100;
    let x = (AXIS_CHARS - 1).min((err_sqr_mean / sampler.axis_max.unwrap() * AXIS_CHARS as Ampl) as usize);

    let layer_weights = network.get_all_weights().into_iter()
        .map(|v| v.iter().flat_map(|m| m.iter()).copied().minmax_by(cmp_ampl_ref))
        .enumerate()
        .filter_map(|(index, minmaxres)| match minmaxres { MinMaxResult::MinMax(min, max) => Some((index, min, max)), _ => None})
        // .map(|(index, min, max)| format_args!("l{} weights min/max {:.3}/{:.3}", index, min, max))
        .format_with(", ", |(index, min, max), f| f(&format_args!("l{} weights min/max {:>7.3}/{:>7.3}", index + 1, min, max)));
    println!("|{0:>1$}{2:>3$}| errsqr: {4:>.3}, samples: {5:>6}, {6}", "*", x + 1, "", AXIS_CHARS - x - 1, err_sqr_mean, sampler.total_count, layer_weights);


}

pub fn cmp_ampl(x: Ampl, y: Ampl) -> Ordering {
    if x > y { Ordering::Greater } else { Ordering::Less }
}

pub fn cmp_ampl_ref(x: &Ampl, y: &Ampl) -> Ordering {
    cmp_ampl(*x, *y)
}

pub fn run_test_iterations(network: &Network, samples: impl Iterator<Item=Sample>) -> Ampl {
    let start = Instant::now();
    println!("testing");

    let mut errsqr_sum = 0.;
    let mut samples_count = 0;
    for sample in samples {
        let errsqr = get_err_sqr(network, &sample);
        errsqr_sum += errsqr;
        samples_count += 1;
    }

    let duration = start.elapsed();
    println!("duration {:.2?}", duration);

    // println!("errsqr {} samples_count {}", errsqr_sum, samples_count);
    errsqr_sum / samples_count as Ampl
}

pub fn run_test_iterations_parallel(network: &Network, samples: impl ParallelIterator<Item=Sample>) -> Ampl {
    let start = Instant::now();
    println!("testing");

    let result: (i32, Ampl) = samples.map(|sample| {
        (1, get_err_sqr(network, &sample))
    }).reduce(|| (0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1));

    let duration = start.elapsed();
    println!("duration {:.2?}", duration);

    result.1 / result.0 as Ampl
}

fn get_err_sqr(network: &Network, sample: &Sample) -> Ampl {
    // println!("{:?}", std::thread::current());
    let output = network.evaluate_input(&sample.0);
    let diff = output - &sample.1;
    let err_sqr = diff.scalar_prod(&diff);
    // if (errsqr > 0.1) {
    //     println!("errsqr {} input {} output sample {} output network {}", errsqr, sample.0, sample.1, output);
    // }
    err_sqr
}

pub fn write_network_to_file(network: &Network, filepath: impl AsRef<Path>) {
    let json = serde_json::to_string_pretty( &network.get_all_weights()).expect("error serializing");
    let mut file = fs::File::create(&filepath).expect("error creating file");
    file.write_all(json.as_bytes()).expect("error writing");
    file.flush().unwrap();
    println!("File written {}", fs::canonicalize(&filepath).unwrap().to_str().unwrap());
}

pub fn read_network_from_file(network : &mut Network, filepath: impl AsRef<Path>) {
    let mut file = fs::File::open(filepath).expect("error opening file");
    let mut json = String::new();
    file.read_to_string(&mut json);
    let weights : Vec<Vec<Matrix<Ampl>>> = serde_json::from_str(&json).expect("error parsing json");
    network.set_all_weights(weights);
}

fn sigmoid_logistic(input: Ampl) -> Ampl {
    sigmoid_logistic_raw(input)
}

fn sigmoid_logistic_raw(input: Ampl) -> f64 {
    1. / (1. + (-input).exp())
}

fn sigmoid_logistic_derived(input: Ampl) -> Ampl {
    sigmoid_logistic_derived_raw(input)
}

fn sigmoid_logistic_derived_raw(input: Ampl) -> f64 {
    (-input).exp() / (1. + (-input).exp()).powf(2.)
}

fn relu(input: Ampl) -> Ampl {
    if input >= 0.0 {input} else {0.0}
}

fn relu_derived(input: Ampl) -> Ampl {
    if input >= 0.0 {1.0} else {0.0}
}


