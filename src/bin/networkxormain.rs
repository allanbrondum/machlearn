use std::iter;
use std::time::Instant;

use rand::{ Rng, SeedableRng};
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use rayon::iter::ParallelBridge;


use machlearn::neuralnetwork::{ActivationFunction, FullyConnectedLayer, LayerContainer, Sample};
use machlearn::neuralnetwork;
use machlearn::neuralnetwork::Network;
use machlearn::vector::Vector;

fn main() {
    let layer1 = FullyConnectedLayer::new(3, 4);
    let layer2 = FullyConnectedLayer::new(4, 1);
    let mut network = Network::new(
        vec!(LayerContainer::new(Box::new(layer1), ActivationFunction::relu()),
             LayerContainer::new(Box::new(layer2), ActivationFunction::sigmoid()),
        ),
        true);

    network.set_random_weights_seed(0);

    // let mut rng = rand::rngs::thread_rng();
    let mut rng = rand::rngs::StdRng::from_entropy();

    let samples = iter::from_fn(
        move || {
            let input = Vector::new(3).apply(|_| if rng.gen_bool(0.5) {1.0} else {0.0});
            // let input = Vector::new(2).apply(|_| if rng.gen_bool(0.5) {1.0} else {0.0});
            let output = if input[0] == input[1] {0.0} else {1.0};
            // println!("input {} output {}", input, output);
            let mut output_vector = Vector::new(1);
            output_vector[0] = output;
            Some(Sample(input, output_vector))
        });

    let learning_samples = samples.clone().take(100000);
    let test_samples = samples.clone().take(1000);

    let start = Instant::now();

    neuralnetwork::run_learning_iterations(&mut network, learning_samples, 0.5, false, 1000);
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork::run_test_iterations_parallel(&network, test_samples.par_bridge());

    let duration = start.elapsed();
    println!("duration {:?}", duration);

    println!("error squared: {}", errsqr);
    println!("network:\n{}", network);

    // println!("network:\n{}", network);
}

