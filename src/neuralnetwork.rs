//! Simple multilayer fully connected neural network using backpropagation of errors (gradient descent) for learning.

use std::any::Any;
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::{fmt, fs};
use std::io::{BufRead, Write};
use std::io::Read;
use std::io;
use std::mem::{replace, swap};
use std::ops::{Deref, Mul};
use std::path::Path;
use std::time::Instant;

use itertools::Itertools;
use rand::Rng;
use rayon::prelude::*;

pub use convolutional::ConvolutionalLayer;
pub use dense::FullyConnectedLayer;
pub use pool::PoolLayer;
pub use activation_function::ActivationFunction;

use crate::matrix::{Matrix, MatrixT};
use crate::vector::Vector;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeStruct;


pub type Ampl = f64;

#[cfg(test)]
mod tests;

mod convolutional;
mod dense;
mod activation_function;
mod pool;

/// Sample tuple, .0: input, .1: output
pub struct Sample(pub Vector<Ampl>, pub Vector<Ampl>);

/// Feed forward neural network layer.
#[typetag::serde(tag = "type")]
pub trait Layer : Display + Debug + Sync {
    fn get_weights(&self) -> Vec<&Matrix<Ampl>>;

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


#[derive(Debug, Serialize, Deserialize)]
pub struct LayerContainer {
    layer: Box<dyn Layer>,
    activation_function: ActivationFunction,
}

impl LayerContainer {
    pub fn new(layer: Box<dyn Layer>, activation_function: ActivationFunction) -> Self {
        LayerContainer {layer, activation_function}
    }
}


#[derive(Debug, Serialize, Deserialize)]
pub struct Network
{
    layers: Vec<LayerContainer>,
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

    // pub fn set_all_weights(&mut self, weights: Vec<Vec<Matrix<Ampl>>>) {
    //     if self.layers.len() != weights.len() {
    //         panic!("Number of layers {} does not equals weights length {}", self.layers.len(), weights.len());
    //     }
    //     for layer_weight in self.layers.iter_mut().zip(weights.into_iter()) {
    //         layer_weight.0.layer.set_weights(layer_weight.1);
    //     }
    // }
}


impl Network {

    pub fn evaluate_input(&self, input: &Vector<Ampl>) -> Vector<Ampl> {
        if input.len() != self.layers.first().unwrap().layer.get_input_dimension() {
            panic!("Input state length {} not equals to first layer input length {}", input.len(), self.layers.first().unwrap().layer.get_input_dimension())
        }
        // evaluate states feed forward through layers
        let mut state= input.clone();
        for layer in &self.layers {
            state = layer.evaluate_input(&state);
        }
        state
    }

    pub fn back_propagate(&mut self, input: &Vector<Ampl>, expected_output: &Vector<Ampl>, ny: Ampl,
                          print: bool, sampler: &mut NetworkBackPropagateSampler) {
        if input.len() != self.layers.first().unwrap().layer.get_input_dimension() {
            panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers.first().unwrap().layer.get_input_dimension())
        }
        if expected_output.len() != self.layers.last().unwrap().layer.get_output_dimension() {
            panic!("Output state length {} not equals to last layer state vector length {}", expected_output.len(), self.layers.last().unwrap().layer.get_output_dimension())
        }

        // first evaluate states using feed forward
        let mut layer_input_states = Vec::new();
        let mut state= input.clone();
        for layer in &self.layers {
            let mut output = layer.evaluate_input(&state);
            layer_input_states.push(state);
            state = output;
        }

        // backpropagation
        let diff = (state.clone() - expected_output);
        let err_sqr = diff.scalar_prod(&diff);
        sampler.sample_iteration(err_sqr);
        let mut gamma = 2. * (state - expected_output);
        let layer_count = self.layer_count();
        let mut ny_with_factor = ny;
        for (index, (layer, input)) in self.layers.iter_mut().rev().zip(layer_input_states.iter().rev()).enumerate() {
            let layer_index = layer_count - 1 - index;
            gamma = layer.back_propagate(input, gamma, ny_with_factor, sampler.layers_samples.get_mut(layer_index).unwrap());

            ny_with_factor *= ((layer.layer.get_input_dimension() / layer.layer.get_output_dimension()) as Ampl).sqrt(); // try to give higher adjustment to lower layers to counter vanishing gradients
        }
    }
}

impl LayerContainer {

    /// Evaluates the given input and returns the output of the layer
    fn evaluate_input(&self, input: &Vector<Ampl>) -> Vector<Ampl> {
        if input.len() != self.layer.get_input_dimension() {
            panic!("Input state length {} not equals to weights column count {}", input.len(), self.layer.get_input_dimension());
        }
        let state_no_activation = self.layer.evaluate_input_without_activation(input);
        let a = 0;
        state_no_activation.apply(self.activation_function.activation_function)
    }

    /// Use highest gradient back propagation to adjust weights moderated by the factor `ny`.
    /// The vector `gamma_output` is the partial
    /// derivatives of error squared evaluated at the given `input` with respect to the output state coordinates
    /// (the dimension of `gamma_output` is thus [`get_output_dimension`]). The method should return
    /// the partial derivatives of the error squared evaluated at the given `input` with respect to the input
    /// state coordinates (the dimension of the returned vector is thus [`get_input_dimension`]).
    fn back_propagate(&mut self, input: &Vector<Ampl>, gamma_output: Vector<Ampl>, ny: Ampl, sampler: &mut NetworkBackPropagateLayerSampler) -> Vector<Ampl> {
        // the delta vector is the partial derivative of error squared with respect to the layer output before the sigmoid function is applied
        sampler.input.sample(input.iter().copied());
        let output_no_activation = self.layer.evaluate_input_without_activation(input);
        sampler.output_without_activation.sample(output_no_activation.iter().copied());
        let delta_output = output_no_activation.apply(self.activation_function.activation_function_derived).mul_comp(&gamma_output);
        sampler.delta_output.sample(delta_output.iter().copied());

        self.layer.back_propagate_without_activation(&input, delta_output, ny)
    }
}

impl Network {
    pub fn new_dense(dimensions: Vec<usize>) -> Self {
        if dimensions.len() < 2 {
            panic!("Must have at least two dimensions, was {}", dimensions.len());
        }
        let mut layers = Vec::new();
        for i in 1..dimensions.len() {
            let boxed: Box<dyn Layer> = Box::new(FullyConnectedLayer::new(dimensions[i - 1], dimensions[i]));
            layers.push(LayerContainer::new(boxed, ActivationFunction::sigmoid()));
        }
        Self::new(layers)
    }

    pub fn new(layers: Vec<LayerContainer>) -> Self
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

struct MinMaxSum {
    min: Option<Ampl>,
    max: Option<Ampl>,
    sum: Ampl,
    count: usize,
}

impl Display for MinMaxSum {

    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:>7.3}/{:>7.3}/{:>7.3}", self.min.unwrap_or(Ampl::NAN), self.sum / self.count as Ampl, self.max.unwrap_or(Ampl::NAN))
    }
}

impl MinMaxSum {
    fn new() -> Self {
        MinMaxSum {
            min: None,
            max: None,
            sum: 0.0,
            count: 0,
        }
    }

    fn new_from_sample(iter: impl Iterator<Item=Ampl>) -> Self {
        let mut min_max_sum = Self::new();
        min_max_sum.sample(iter);
        min_max_sum
    }

    fn reset(&mut self) {
        self.min = None;
        self.max = None;
        self.sum = 0.0;
        self.count = 0;
    }

    fn sample(&mut self, iter: impl Iterator<Item=Ampl>) {
        iter.for_each(|value| {
            if self.min.is_none() {
                self.min = Some(value);
            }
            if self.max.is_none() {
                self.max = Some(value);
            }
            if value < self.min.unwrap() {
                self.min.replace(value);
            }
            if value > self.max.unwrap() {
                self.max.replace(value);
            }
            self.sum += value;
            self.count += 1;
        });
    }
}

/// Used to provide statistics on back propagation
struct NetworkBackPropagateLayerSampler {
    delta_weights: MinMaxSum,
    delta_output: MinMaxSum,
    input: MinMaxSum,
    output_without_activation: MinMaxSum,
}

impl NetworkBackPropagateLayerSampler {
    fn new() -> NetworkBackPropagateLayerSampler {
        NetworkBackPropagateLayerSampler {
            delta_weights: MinMaxSum::new(),
            delta_output: MinMaxSum::new(),
            input: MinMaxSum::new(),
            output_without_activation: MinMaxSum::new(),
        }
    }

    fn reset_sampling(&mut self) {
        self.delta_weights.reset();
        self.delta_output.reset();
        self.input.reset();
        self.output_without_activation.reset();
    }
}

/// Used to provide statistics on back propagation
pub struct NetworkBackPropagateSampler {
    err_sqr: Ampl,
    count: usize,
    total_count: usize,
    err_sqr_axis_max: Option<Ampl>,
    layers_samples: Vec<NetworkBackPropagateLayerSampler>,
}

impl NetworkBackPropagateSampler {
    fn new(layer_count: usize) -> NetworkBackPropagateSampler {
        NetworkBackPropagateSampler {
            err_sqr: 0.0,
            count: 0,
            err_sqr_axis_max: None,
            total_count: 0,
            layers_samples: (0..layer_count).map(|_| NetworkBackPropagateLayerSampler::new()).collect(),
        }
    }

    fn sample_iteration(&mut self, err_sqr: Ampl) {
        self.total_count += 1;
        self.count += 1;
        self.err_sqr += err_sqr;
    }

    fn reset_sampling(&mut self) {
        self.count = 0;
        self.err_sqr = 0.0;
        for layer_sample in &mut self.layers_samples {
            layer_sample.reset_sampling();
        }
    }

    fn mean_err_sqr_for_batch(&self) -> Ampl {
        self.err_sqr / self.count as Ampl
    }
}

pub fn run_learning_iterations(network: &mut Network, samples: impl Iterator<Item=Sample>, ny: Ampl, print: bool, sample_count: usize) {
    let start = Instant::now();
    println!("learning");

    let mut back_prop_sampler = NetworkBackPropagateSampler::new(network.layer_count());
    for sample in samples {
        network.back_propagate(&sample.0, &sample.1, ny, print, &mut back_prop_sampler);

        if back_prop_sampler.count == sample_count {
            print_learning_stats(network, &mut back_prop_sampler);
            back_prop_sampler.reset_sampling();
        }
    }

    let duration = start.elapsed();
    println!("duration {:.2?}", duration);
}

fn print_learning_stats(network: &Network, sampler: &mut NetworkBackPropagateSampler) {
    let err_sqr_mean = sampler.err_sqr / sampler.count as Ampl;
    if sampler.err_sqr_axis_max.is_none() {
        sampler.err_sqr_axis_max.replace(err_sqr_mean);
    }

    const AXIS_CHARS: usize = 50;
    let x = (AXIS_CHARS - 1).min((err_sqr_mean / sampler.err_sqr_axis_max.unwrap() * AXIS_CHARS as Ampl) as usize);

    let layer_weights_display = network.get_all_weights().into_iter()
        .map(|v|  MinMaxSum::new_from_sample(v.iter().flat_map(|m| m.iter()).copied()))
        .enumerate()
        .format_with(", ", |(index, min_max_sum), f| f(&format_args!("l{} weights {}", index + 1, min_max_sum)));
    let layer_back_prop_stats_display = sampler.layers_samples.iter()
        .enumerate()
        .format_with(", ", |(index, sampler), f| f(&format_args!("l{} in {} out {} out der {}", index + 1, sampler.input, sampler.output_without_activation, sampler.delta_output)));
    println!("|{0:>1$}{2:>3$}| errsqr: {4:>.3}, samples: {5:>6}, {6} {7}", "*", x + 1, "", AXIS_CHARS - x - 1, err_sqr_mean, sampler.total_count, layer_weights_display, layer_back_prop_stats_display);


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
    let json = serde_json::to_string_pretty( &network).expect("error serializing");
    let mut file = fs::File::create(&filepath).expect("error creating file");
    file.write_all(json.as_bytes()).expect("error writing");
    file.flush().unwrap();
    println!("File written {}", fs::canonicalize(&filepath).unwrap().to_str().unwrap());
}

pub fn read_network_from_file(network : &mut Network, filepath: impl AsRef<Path>) {
    let mut file = fs::File::open(filepath).expect("error opening file");
    let mut json = String::new();
    file.read_to_string(&mut json);
    let network_des: Network = serde_json::from_str(&json).expect("error parsing json");
    replace(network, network_des);
}


