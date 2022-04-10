use std::{fs, io, iter};
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Bytes, Read, Write};
use std::iter::{FromFn, Take};
use std::ops::Deref;
use std::rc::Rc;
use std::time::Instant;

use itertools::{Chunk, Itertools};
use rand::Rng;
use rayon::iter::{ParallelBridge, ParallelIterator};

use machlearn::neuralnetwork::{Ampl, Network, Sample};
use machlearn::neuralnetwork;
use machlearn::vector::Vector;
use machlearn::matrix::Matrix;
use std::path::{PathBuf, Path};
use std::str::FromStr;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use machlearn::mnistdigits;

fn main() {
    let mut network = Network::new_logistic_sigmoid(vec!(mnistdigits::IMAGE_PIXEL_COUNT, 10)); // single layer
    // let mut network = Network::new_logistic_sigmoid(vec!(IMAGE_PIXEL_COUNT, IMAGE_PIXEL_COUNT, 10));
    mnistdigits::print_data_examples();

    let read_from_file = false;
    if !read_from_file {
        // let rng = rand::thread_rng();
        let rng: Pcg64 = Seeder::from(0).make_rng();
        network.set_random_weights_rng(rng);

        const LEARNING_SAMPLES: usize = 1_000;
        // const LEARNING_SAMPLES: usize = 10_000;
        neuralnetwork::run_learning_iterations(&mut network, mnistdigits::get_learning_samples().take(LEARNING_SAMPLES), 0.3);
    } else {
        neuralnetwork::read_network_from_file(&mut network, "mnist_twolayer_weights.json");
        // read_network_from_file(&mut network, "mnist_singlelayer_weights.json");
        // println!("network \n{}", network);
    }

    if false {
        neuralnetwork::run_and_print_learning_iterations(&mut network, mnistdigits::get_learning_samples().take(20), 0.3);
    }

    const TEST_SAMPLES: usize = 1000;
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork::run_test_iterations_parallel(&network, mnistdigits::get_test_samples().take(TEST_SAMPLES).par_bridge());

    test_correct_percentage(&network, mnistdigits::get_test_samples().take(20).par_bridge(), true);
    let pct_correct = test_correct_percentage(&network, mnistdigits::get_test_samples().take(TEST_SAMPLES).par_bridge(), false);

    println!("error squared: {:.5}", errsqr);
    println!("% correct: {:.2}", pct_correct);

    // println!("network: {}", network);

    neuralnetwork::write_network_to_file(&network,"mnist_tmp_weights.json");
}

pub fn test_correct_percentage(network: &Network, samples: impl ParallelIterator<Item=Sample>, print: bool) -> f64 {
    let result: (usize, usize) = samples.map(|sample| {
        let output = network.evaluate_input_no_state_change(&sample.0);
        let guess = index_of_max(&output);
        let correct = index_of_max(&sample.1);
        if print {
            println!("Output {}, guess {}, correct {}", output, guess, correct);
        }
        let is_correct = if guess == correct {1} else {0};
        (1, is_correct)
    }).reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1 + y.1));
    100. * result.1 as f64 / result.0 as f64

}

fn index_of_max(vector: &Vector<Ampl>) -> usize {
    vector.iter().enumerate().max_by(|x, y| if x.1 > y.1 { Ordering::Greater } else { Ordering::Less }).unwrap().0
}


