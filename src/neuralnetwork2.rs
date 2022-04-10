//! Simple multilayer fully connected neural network using backpropagation of errors (gradient descent) for learning.

use std::fmt::{Debug, Display, Formatter};
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
use rand::rngs::ThreadRng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use crate::neuralnetwork::{Ampl, Sample};

#[derive(Debug, Clone)]
pub struct FullyConnectedLayer
{
    weights: Matrix<Ampl>,
    biases: bool,
}

pub trait Layer : Display + Debug + Sync {
    fn get_weights(&self) -> Vec<&Matrix<Ampl>>;

    fn set_weights(&mut self, new_weights: Vec<Matrix<Ampl>>);

    fn get_input_dimension(&self) -> usize;

    fn get_output_dimension(&self) -> usize;

    fn set_random_weights(&mut self);

    fn set_random_weights_seed(&mut self, seed: u64);

    fn evaluate_input(&self, input: &Vector<Ampl>, sigmoid: fn(Ampl) -> Ampl) -> Vector<Ampl>;

    fn backpropagate(&mut self, input: &Vector<Ampl>, gamma_output: &Vector<Ampl>, sigmoid_derived: fn(Ampl) -> Ampl, ny: Ampl) -> Vector<Ampl>;
}

impl FullyConnectedLayer {
    pub fn new(input_dimension: usize, output_dimension: usize, biases: bool) -> FullyConnectedLayer {
        FullyConnectedLayer {
            weights: Matrix::new(output_dimension, input_dimension),
            biases,
        }
    }
}

impl Layer for FullyConnectedLayer {

    fn get_weights(&self) -> Vec<&Matrix<Ampl>> {
        vec!(&self.weights)
    }

    fn set_weights(&mut self, new_weights_vec: Vec<Matrix<Ampl>>) {
        assert_eq!(1, new_weights_vec.len(), "Should only have one weight matrix, has {}", new_weights_vec.len());
        let new_weights = new_weights_vec.into_iter().next().unwrap();
        if self.weights.dimensions() != new_weights.dimensions() {
            panic!("Weight dimensions for layer {} does not equals dimension of weights to set {}", self.weights.dimensions(), new_weights.dimensions());
        }
        self.weights = new_weights;
    }

    fn get_input_dimension(&self) -> usize {
        self.weights.dimensions().columns
    }

    fn get_output_dimension(&self) -> usize {
        self.weights.dimensions().rows
    }

    fn set_random_weights(&mut self) {
        let mut rng = rand::thread_rng();
        self.weights.apply_ref(|_| rng.gen_range(-1.0..1.0));
    }

    fn set_random_weights_seed(&mut self, seed: u64) {
        let mut rng: Pcg64 = Seeder::from(0).make_rng();
        self.weights.apply_ref(|_| rng.gen_range(-1.0..1.0));
    }

    fn evaluate_input(&self, input: &Vector<Ampl>, sigmoid: fn(Ampl) -> Ampl) -> Vector<Ampl> {
        if input.len() != self.get_input_dimension() {
            panic!("Input state length {} not equals to weights column count {}", input.len(), self.weights.dimensions().columns);
        }
        self.weights.mul_vector(input).apply(sigmoid)
    }

    fn backpropagate(&mut self, input: &Vector<Ampl>, gamma_output: &Vector<Ampl>, sigmoid_derived: fn(Ampl) -> Ampl, ny: Ampl) -> Vector<Ampl> {
        let delta_output = self.weights.mul_vector(input).apply(sigmoid_derived).mul_comp(gamma_output);
        let gamma_input = self.weights.mul_vector_lhs(&delta_output);

        // adjust weights
        self.weights -= ny * delta_output.to_matrix().mul_mat(&input.clone().to_matrix().transpose());

        gamma_input
    }
}

#[derive(Debug)]
pub struct Network
{
    layers: Vec<Box<dyn Layer>>,
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

    pub fn set_random_weights_seed(&mut self, seed: u64) {
        for layer in &mut self.layers {
            layer.set_random_weights_seed(seed);
        }
    }

    pub fn get_all_weights(&self) -> Vec<Vec<&Matrix<Ampl>>> {
        self.layers.iter().map(|layer| layer.get_weights()).collect()
    }

    pub fn set_all_weights(&mut self, weights: Vec<Vec<Matrix<Ampl>>>) {
        if self.layers.len() != weights.len() {
            panic!("Number of layers {} does not equals weights length {}", self.layers.len(), weights.len());
        }
        for layer_weight in self.layers.iter_mut().zip(weights.into_iter()) {
            layer_weight.0.set_weights(layer_weight.1);
        }
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

    pub fn evaluate_input(&self, input: &Vector<Ampl>) -> Vector<Ampl> {
        if input.len() != self.layers.first().unwrap().get_input_dimension() {
            panic!("Input state length {} not equals to first layer input length {}", input.len(), self.layers.first().unwrap().get_input_dimension())
        }
        // evaluate states feed forward through layers
        let mut state= input.clone();
        if self.biases {
            *state.last() = 1.0;
        }
        for layer in &self.layers {
            state = layer.evaluate_input(&state, self.sigmoid);
            if self.biases {
                *state.last() = 1.0;
            }
        }
        state
    }

    pub fn backpropagate(&mut self, input: &Vector<Ampl>, expected_output: &Vector<Ampl>, ny: Ampl, print: bool) {
        if input.len() != self.layers.first().unwrap().get_input_dimension() {
            panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers.first().unwrap().get_input_dimension())
        }
        if expected_output.len() != self.layers.last().unwrap().get_output_dimension() {
            panic!("Output state length {} not equals to last layer state vector length {}", expected_output.len(), self.layers.last().unwrap().get_output_dimension())
        }

        // first evaluate states using feed forward
        let mut layer_input_states = Vec::new();
        let mut state= input.clone();        if self.biases {
            *state.last() = 1.0;
        }
        for layer in &mut self.layers {
            let mut output = layer.evaluate_input(&state, self.sigmoid);
            if self.biases {
                *output.last() = 1.0;
            }
            layer_input_states.push(state);
            state = output;
        }

        // backpropagation
        let mut gamma = 2. * (state - expected_output);
        for layer_input in self.layers.iter_mut().rev().zip(layer_input_states.iter().rev()) {
            let layer = layer_input.0;
            let input = layer_input.1;

            gamma = layer.backpropagate(input, &gamma, self.sigmoid_derived, ny);
        }

    }
}

impl Network {
    pub fn new_logistic_sigmoid(dimensions: Vec<usize>) -> Self {
        Self::new_fully_connected(dimensions, sigmoid_logistic, sigmoid_logistic_derived, false)
    }

    pub fn new_logistic_sigmoid_biases(dimensions: Vec<usize>) -> Self {
        Self::new_fully_connected(dimensions, sigmoid_logistic, sigmoid_logistic_derived, true)
    }

    pub fn new_fully_connected(
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
            let boxed: Box<dyn Layer> = Box::new(FullyConnectedLayer::new(dimensions[i - 1], dimensions[i], biases));
            layers.push(boxed);
        }
        Self::new(layers, sigmoid, sigmoid_derived, biases)
    }

    pub fn new(
        layers: Vec<Box<dyn Layer>>,
        sigmoid: fn(Ampl) -> Ampl,
        sigmoid_derived: fn(Ampl) -> Ampl,
        biases: bool) -> Self
    {
        if layers.is_empty() {
            panic!("Must have at least one layer");
        }

        Network {
            layers,
            sigmoid,
            sigmoid_derived,
            biases
        }
    }

    pub fn get_layers(&self) -> &Vec<Box<dyn Layer>> {
        &self.layers
    }

    pub fn get_layer_count(&self) -> usize {
        self.layers.len()
    }
}

impl Display for Network
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {

        // write!(f, "input state: {}\n\n", self.input_state)?;

        for layer in self.layers.iter().enumerate() {
            write!(f, "layer {}\n{}", layer.0, layer.1)?;
        }

        std::fmt::Result::Ok(())
    }
}

impl Display for FullyConnectedLayer
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // write!(f, "backpropagation\n{}", self.backpropagation_gamma)?;
        write!(f, "weights\n{}", self.weights)?;
        // write!(f, "output state: {}\n", self.output_state)?;

        std::fmt::Result::Ok(())
    }
}


pub fn run_and_print_learning_iterations(network: &mut Network, samples: impl Iterator<Item=Sample>, ny: Ampl) {
    run_learning_iterations_impl(network, samples, ny, true);
}

pub fn run_learning_iterations(network: &mut Network, samples: impl Iterator<Item=Sample>, ny: Ampl) {
    run_learning_iterations_impl(network, samples, ny, false);
}

fn run_learning_iterations_impl(network: &mut Network, samples: impl Iterator<Item=Sample>, ny: Ampl, print: bool) {
    let start = Instant::now();
    println!("learning");

    for sample in samples {
        network.backpropagate(&sample.0, &sample.1, ny, print);
        // println!("network {}", network);
    }

    let duration = start.elapsed();
    println!("duration {:.2?}", duration);
}


pub fn run_test_iterations(network: &Network, samples: impl Iterator<Item=Sample>) -> Ampl {
    let start = Instant::now();
    println!("testing");

    let mut errsqr_sum = 0.;
    let mut samples_count = 0;
    for sample in samples {
        let errsqr = get_errsqr(network, &sample);
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
        (1, get_errsqr(network, &sample))
    }).reduce(|| (0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1));

    let duration = start.elapsed();
    println!("duration {:.2?}", duration);

    result.1 / result.0 as Ampl
}

fn get_errsqr(network: &Network, sample: &Sample) -> Ampl {
    // println!("{:?}", std::thread::current());
    let output = network.evaluate_input(&sample.0);
    let diff = output - &sample.1;
    let errsqr = diff.scalar_prod(&diff);
    // if (errsqr > 0.1) {
    //     println!("errsqr {} input {} output sample {} output network {}", errsqr, sample.0, sample.1, output);
    // }
    errsqr
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
