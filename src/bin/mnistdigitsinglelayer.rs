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

use machlearn::neuralnetwork::{Ampl, Sample};
use machlearn::neuralnetwork2;
use machlearn::vector::Vector;
use machlearn::matrix::Matrix;
use std::path::{PathBuf, Path};
use std::str::FromStr;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use machlearn::mnistdigits;
use machlearn::neuralnetwork2::Network;

fn main() {
    let mut network = Network::new_logistic_sigmoid(vec!(mnistdigits::IMAGE_PIXEL_COUNT, 10)); // single layer
    mnistdigits::print_data_examples();

    let read_from_file = false;
    if !read_from_file {
        // let rng = rand::thread_rng();
        let mut rng: Pcg64 = Seeder::from(0).make_rng();
        network.set_random_weights_rng(&mut rng);

        const LEARNING_SAMPLES: usize = 1_000;
        // const LEARNING_SAMPLES: usize = 10_000;
        neuralnetwork2::run_learning_iterations(&mut network, mnistdigits::get_learning_samples().take(LEARNING_SAMPLES), 0.3);
    } else {
        neuralnetwork2::read_network_from_file(&mut network, "mnist_twolayer_weights.json");
        // read_network_from_file(&mut network, "mnist_singlelayer_weights.json");
        // println!("network \n{}", network);
    }

    if false {
        neuralnetwork2::run_and_print_learning_iterations(&mut network, mnistdigits::get_learning_samples().take(20), 0.3);
    }

    const TEST_SAMPLES: usize = 1000;
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork2::run_test_iterations_parallel(&network, mnistdigits::get_test_samples().take(TEST_SAMPLES).par_bridge());

    mnistdigits::test_correct_percentage(&network, mnistdigits::get_test_samples().take(20).par_bridge(), true);
    let pct_correct = mnistdigits::test_correct_percentage(&network, mnistdigits::get_test_samples().take(TEST_SAMPLES).par_bridge(), false);

    println!("error squared: {:.5}", errsqr);
    println!("% correct: {:.2}", pct_correct);

    // println!("network: {}", network);

    neuralnetwork2::write_network_to_file(&network,"mnist_tmp_weights.json");
}

