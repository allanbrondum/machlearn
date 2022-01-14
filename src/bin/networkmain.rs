use std::borrow::Borrow;
use std::ops::IndexMut;

use machlearn::matrix::Matrix;
use machlearn::neuralnetwork::{Network, ampl, sigmoid_logistic, Sample, run_learning_iterations};
use machlearn::vector::Vector;
use rand::{Rng, random};
use std::iter;
use machlearn::neuralnetwork;

fn main() {
    let mut network = Network::new_logistic_sigmoid(vec!(8, 6, 8));

    // print_sigmoid_values(&network);

    let mut network2 = network.clone();

    let mut rng = rand::thread_rng();

    network.set_random_weights();
    network2.set_random_weights();

    let mut samples = iter::from_fn(
        move || {
            let input = Vector::new(8).apply(|_| rng.gen());
            let output = network.evaluate_input_no_state_change(input.clone());
            Some(Sample(input, output))
        });

    let learning_samples = samples.clone().take(1000);
    let test_samples = samples.clone().take(100);

    neuralnetwork::run_learning_iterations(&mut network2, learning_samples);
    let errsqr = neuralnetwork::run_test_iterations(&network2, test_samples);

    println!("error squared: {}", errsqr);
}

fn print_sigmoid_values(network: &Network) {
    let sigmoid = (network.get_sigmoid());
    println!("sigmoid 0.0: {}", sigmoid(0.0));
    println!("sigmoid 0.2: {}", sigmoid(0.2));
    println!("sigmoid 0.4: {}", sigmoid(0.4));
    println!("sigmoid 0.6: {}", sigmoid(0.6));
    println!("sigmoid 0.8: {}", sigmoid(0.8));

    let sigmoid_derived = (network.get_sigmoid_derived());
    println!("sigmoid_derived 0.0: {}", sigmoid_derived(0.0));
    println!("sigmoid_derived 0.2: {}", sigmoid_derived(0.2));
    println!("sigmoid_derived 0.4: {}", sigmoid_derived(0.4));
    println!("sigmoid_derived 0.6: {}", sigmoid_derived(0.6));
    println!("sigmoid_derived 0.8: {}", sigmoid_derived(0.8));
}
