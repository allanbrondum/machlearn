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
use rand::prelude::ThreadRng;

pub type Ampl = f64;

#[derive(Debug, Clone)]
pub struct Layer
{
    state: Vector<Ampl>
}

#[derive(Debug, Clone)]
pub struct Connector
{
    weights: Matrix<Ampl>,
    back_propagation_delta: Vector<Ampl>
}

impl Layer {

    pub fn new(dimension: usize) -> Layer {
        Layer {
            state: Vector::new(dimension)
        }
    }

    pub fn get_state(&self) -> &Vector<Ampl> {
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

    pub fn get_weights(&self) -> &Matrix<Ampl> {
        &self.weights
    }

    pub fn get_back_propagation_delta(&self) -> &Vector<Ampl> {
        &self.back_propagation_delta
    }
}

#[derive(Debug, Clone)]
pub struct Network
{
    layers: Vec<Layer>,
    connectors: Vec<Connector>,
    sigmoid: fn(Ampl) -> Ampl,
    sigmoid_derived: fn(Ampl) -> Ampl,
    biases: bool,
}

impl Network {
    pub fn set_random_weights(&mut self) {
        let mut rng = rand::thread_rng();
        self.set_random_weights_rng(rng);
    }

    pub fn set_random_weights_rng(&mut self, mut rng: impl Rng) {
        for connector in &mut self.connectors {
            connector.weights.apply_ref(|_| rng.gen_range(-1.0..1.0));
        }
    }

    pub fn copy_all_weights(&self) -> Vec<Matrix<Ampl>> {
        self.connectors.iter().map(|con| con.weights.clone()).collect()
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
        if input.len() != self.layers.first().unwrap().state.len() {
            panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers.first().unwrap().state.len())
        }
        self.layers[0].state = input;
        if self.biases {
            *self.layers[0].state.last() = 1.0;
        }
        for i in 0..self.layers.len() - 1 {
            self.layers[i + 1].state = (&self.connectors[i].weights * &self.layers[i].state).apply(self.sigmoid);
            if self.biases {
                *self.layers[i + 1].state.last() = 1.0;
            }
        }
    }

    pub fn evaluate_input_no_state_change(&self, input: &Vector<Ampl>) -> Vector<Ampl> {
        if input.len() != self.layers.first().unwrap().state.len() {
            panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers.first().unwrap().state.len())
        }
        let mut state= input.clone();
        if self.biases {
            *state.last() = 1.0;
        }
        for i in 0..self.layers.len() - 1 {
            state = (&self.connectors[i].weights * &state).apply(self.sigmoid);
            if self.biases {
                *state.last() = 1.0;
            }
        }
        state
    }

    pub fn backpropagate(&mut self, input: Vector<Ampl>, output: &Vector<Ampl>, ny: Ampl, print: bool) {
        if input.len() != self.layers.first().unwrap().state.len() {
            panic!("Input state length {} not equals to first layer state vector length {}", input.len(), self.layers.first().unwrap().state.len())
        }
        if output.len() != self.layers.last().unwrap().state.len() {
            panic!("Output state length {} not equals to last layer state vector length {}", output.len(), self.layers.last().unwrap().state.len())
        }

        self.evaluate_input_state(input);

        let mut output = output.clone();
        if self.biases {
            *output.last() = 1.0;
        }

        if print {
            let diff = output.clone() - self.get_output();
            let errsqr = diff.scalar_prod(&diff);
            println!("errsqr: {:.4}", errsqr);
        }

        // last connector
        let mut normalize = false;
        {
            let layer1 = &self.layers[self.layers.len() - 2];
            let layer2 = self.layers.last().unwrap();
            let last_connector = self.connectors.last_mut().unwrap();
            let tmp = &last_connector.weights * &layer1.state;
            if print {
                println!("");
            }
            for i in 0..last_connector.back_propagation_delta.len() {
                normalize |= (output[i] - layer2.state[i]).abs() > 0.5;

                last_connector.back_propagation_delta[i] = -2. * (self.sigmoid_derived)(tmp[i]) * (output[i] - layer2.state[i]);
                if print {
                    println!("sigder {}: {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
                             i,
                             last_connector.back_propagation_delta[i],
                             output[i] - layer2.state[i],
                             (self.sigmoid_derived)(tmp[i]),
                             tmp[i],
                             output[i],
                             layer2.state[i]);
                }
            }
        }
        if print {
            println!("normalize: {}", normalize);
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

            // if print {
            //     println!("backpropagation: {}", connector.back_propagation_delta);
            // }

            let deltam = connector.back_propagation_delta.clone().to_matrix();
            let statem = layer1.state.clone().to_matrix();
            let statemt = statem.transpose();
            let mut tmp = deltam.mul_mat(&statemt);
            if normalize {
                let normsqr = tmp.scalar_prod(&tmp);
                if print {
                    println!("normsq: {:.4}", normsqr);
                }
                if normsqr == 0.0 {
                    tmp *= 1. / normsqr.sqrt();
                }
            }
            tmp *= -ny;
            connector.weights += tmp;
        }
    }
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
            sigmoid_derived,
            biases
        }
    }

    pub fn get_weights(&self, index: usize) -> &Matrix<Ampl> {
        &self.connectors[index].weights
    }

    pub fn set_weights(&mut self, index: usize, weights: Matrix<Ampl>) {
        if self.connectors[index].weights.dimensions() != weights.dimensions() {
            panic!("Dimensions of weights {} not as required by network {}", weights.dimensions(), self.connectors[index].weights.dimensions());
        }
        self.connectors[index].weights = weights;
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

        f.write_str("Layers:\n")?;
        for layer in &self.layers {
            write!(f, "{}\n", layer.state.len())?;
            write!(f, "state: {}\n\n", layer.get_state())?;
        }

        f.write_str("Connectors:\n")?;
        for connector in &self.connectors {
            write!(f, "{}\n", connector.weights.dimensions())?;
            write!(f, "weights:\n{}", connector.weights)?;
            write!(f, "backprop delta: {}\n\n", connector.back_propagation_delta)?;
        }

        std::fmt::Result::Ok(())
    }
}

/// Sample tuple, .0: input, .1: output
pub struct Sample(pub Vector<Ampl>, pub Vector<Ampl>);

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
        network.backpropagate(sample.0, &sample.1, ny, print);
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
    let output = network.evaluate_input_no_state_change(&sample.0);
    let diff = output - &sample.1;
    let errsqr = diff.scalar_prod(&diff);
    // if (errsqr > 0.1) {
    //     println!("errsqr {} input {} output sample {} output network {}", errsqr, sample.0, sample.1, output);
    // }
    errsqr
}

pub fn write_network_to_file(network: &Network, filepath: impl AsRef<Path>) {
    let json = serde_json::to_string(&network.copy_all_weights()).expect("error serializing");
    let mut file = fs::File::create(&filepath).expect("error creating file");
    file.write_all(json.as_bytes()).expect("error writing");
    file.flush().unwrap();
    println!("File written {}", fs::canonicalize(&filepath).unwrap().to_str().unwrap());
}

pub fn read_network_from_file(network : &mut Network, filepath: impl AsRef<Path>) {
    let mut file = fs::File::open(filepath).expect("error opening file");
    let mut json = String::new();
    file.read_to_string(&mut json);
    let weights : Vec<Matrix<Ampl>> = serde_json::from_str(&json).expect("error parsing json");
    for weightenum in weights.into_iter().enumerate() {
        network.set_weights(weightenum.0, weightenum.1);
    }
}

#[cfg(test)]
mod tests {
    use crate::neuralnetwork::*;

    #[test]
    fn test() {

    }
}