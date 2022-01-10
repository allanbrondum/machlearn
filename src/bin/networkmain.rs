use std::borrow::Borrow;
use std::ops::IndexMut;

use machlearn::matrix::Matrix;
use machlearn::neuralnetwork::{Network, ampl, sigmoid_logistic};
use machlearn::vector::Vector;
use rand::{Rng, random};

fn main() {
    let mut rng = rand::thread_rng();

    let mut network = Network::new_logistic_sigmoid(vec!(8, 6, 8));

    // print_sigmoid_values(&network);

    let mut network2 = network.clone();

    let mut rng = rand::thread_rng();

    network.set_weights(0, Matrix::new(6, 8).apply(|_| rng.gen()));
    network.set_weights(1, Matrix::new(8, 6).apply(|_| rng.gen()));

    network2.set_weights(0, Matrix::new(6, 8).apply(|_| rng.gen()));
    network2.set_weights(1, Matrix::new(8, 6).apply(|_| rng.gen()));

    let mut input = Vector::new(8);
    input[0] = 1.;
    input[1] = 1.;
    input[2] = 1.;
    input[3] = 1.;
    input[4] = 1.;
    network.evaluate_input_state(input.clone());
    let output = network.get_output();

    // for layer in network.get_layers() {
    //     println!("layer state: {}", layer.get_state());
    // }

    // learning set
    let learning_iterations = 1000;
    for i in 0..learning_iterations {
        let mut input = Vector::new(8);
        for j in 0..8 {
            input[j] = rng.gen();
        }

        network.evaluate_input_state(input.clone());
        let output = network.get_output();

        network2.backpropagate(input.clone(), output);
    }

    println!("network:\n{}", network);

    println!("network:\n{}", network2);

    // test set
    let test_iterations = 100;
    let mut errsqr = 0.;
    for i in 0..test_iterations {
        let mut input = Vector::new(8);
        for j in 0..8 {
            input[j] = rng.gen();
        }

        network.evaluate_input_state(input.clone());
        let output = network.get_output();

        network2.evaluate_input_state(input.clone());
        let output2 = network2.get_output();

        let diff = output.clone() - output2.clone();
        errsqr += &diff * &diff;

        // println!("output1: {}", output);
        // println!("output2: {}", output2);
    }
    println!("error squared: {}", errsqr / test_iterations as f64);
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
