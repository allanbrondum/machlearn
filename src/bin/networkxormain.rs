use std::iter;
use std::time::Instant;

use rand::{ Rng, SeedableRng};
use rayon::iter::ParallelBridge;


use machlearn::neuralnetwork::{Network,  Sample};
use machlearn::neuralnetwork;
use machlearn::vector::Vector;

fn main() {
    let mut network = Network::new_logistic_sigmoid_biases(vec!(3, 3, 2));
    network.set_random_weights();

    // let mut rng = rand::rngs::thread_rng();
    let mut rng = rand::rngs::StdRng::from_entropy();

    let samples = iter::from_fn(
        move || {
            let input = Vector::new(3).apply(|_| if rng.gen_bool(0.5) {1.0} else {0.0});
            // let input = Vector::new(2).apply(|_| if rng.gen_bool(0.5) {1.0} else {0.0});
            let output = if input[0] == input[1] {0.0} else {1.0};
            // println!("input {} output {}", input, output);
            let mut output_vector = Vector::new(2);
            output_vector[0] = output;
            output_vector[1] = 1.0;
            Some(Sample(input, output_vector))
        });

    let learning_samples = samples.clone().take(100000);
    let test_samples = samples.clone().take(1000);

    let start = Instant::now();

    neuralnetwork::run_learning_iterations(&mut network, learning_samples, 0.5);
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork::run_test_iterations_parallel(&network, test_samples.par_bridge());

    let duration = start.elapsed();
    println!("duration {:?}", duration);

    println!("error squared: {}", errsqr);
    for i in 0..network.get_layer_count() - 1 {
        println!("network connector {}:\n{}", i ,network.get_weights(i));
    }

    // println!("network:\n{}", network);
}

