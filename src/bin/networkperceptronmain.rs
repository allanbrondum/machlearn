


use machlearn::neuralnetwork::{Sample};
use machlearn::vector::Vector;
use rand::{Rng};
use std::iter;
use machlearn::{neuralnetwork};
use machlearn::neuralnetwork::Network;

fn main() {
    let mut network = Network::new_fully_connected(vec!(8, 1));
    network.set_random_weights();

    let mut rng = rand::thread_rng();

    let mut perceptron = Vector::new(8);
    perceptron[0] = 8.;
    perceptron[1] = -6.;
    perceptron[2] = 4.;
    perceptron[3] = 10.;
    perceptron[4] = 12.;
    perceptron[5] = -5.;
    perceptron[6] = 0.;
    perceptron[7] = 1.;

    let samples = iter::from_fn(
        move || {
            let input = Vector::new(8).apply(|_| if rng.gen_bool(0.5) {1.0} else {-1.0} * rng.gen_range(0.5..1.0));
            // let mut input = Vector::new(8);
            // input[rng.gen_range(0..8)] = rng.gen();
            // let output = if perceptron.vec_prod(&input) > 0. {1.0} else {0.0};
            // println!("input {}", input);
            let output = match perceptron.scalar_prod(&input) {
                x if x > 0. => 1.0,
                _ => 0.0,
            };
            // println!("output {}", output);
            let mut output_vector = Vector::new(1);
            output_vector[0] = output;
            Some(Sample(input, output_vector))
        });

    let learning_samples = samples.clone().take(100000);
    let test_samples = samples.clone().take(1000);

    neuralnetwork::run_learning_iterations(&mut network, learning_samples, 0.5, false, 1000);
    let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);

    println!("error squared: {}", errsqr);
    println!("network:\n{}", network);

    // network.set_weights(0, network.get_weights(0).clone().apply(|x| 100. * x));
    // let test_samples = samples.clone().take(1000);
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    // println!("error squared: {}", errsqr);
    // println!("network connector: {}", network.get_weights(0));
}

