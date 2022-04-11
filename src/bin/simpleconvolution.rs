use std::{fs, io, iter};
use std::any::Any;
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

use machlearn::neuralnetwork::{Ampl, FullyConnectedLayer, Layer, Sample};
use machlearn::neuralnetwork;
use machlearn::vector::Vector;
use machlearn::matrix::{Matrix, MatrixT, MutSliceView};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use machlearn::datasets::{convolutiontest, imagedatasets, mnistdigits};
use machlearn::neuralnetwork::Network;

fn main() {
    let mut network = Network::new_logistic_sigmoid(vec!(mnistdigits::IMAGE_PIXEL_COUNT, 10)); // single layer
    convolutiontest::print_data_examples();
    if true {
        return;
    }

    network.set_random_weights_seed(0);

    const LEARNING_SAMPLES: usize = 10_000;
    neuralnetwork::run_learning_iterations(&mut network, convolutiontest::get_learning_samples().take(LEARNING_SAMPLES), 0.3);

    if false {
        neuralnetwork::run_and_print_learning_iterations(&mut network, convolutiontest::get_learning_samples().take(20), 0.3);
    }

    const TEST_SAMPLES: usize = 1000;
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork::run_test_iterations_parallel(&network, convolutiontest::get_test_samples().take(TEST_SAMPLES).par_bridge());

    convolutiontest::test_correct_percentage(&network, mnistdigits::get_test_samples().take(20).par_bridge(), true);
    let pct_correct = convolutiontest::test_correct_percentage(&network, mnistdigits::get_test_samples().take(TEST_SAMPLES).par_bridge(), false);

    println!("error squared: {:.5}", errsqr);
    println!("% correct: {:.2}", pct_correct);

    if false { // print kernels
        let weights = &network.get_all_weights()[0];
        for (i, &kernel) in weights.iter().enumerate() {
            println!("kernel {}:\n", i);
            imagedatasets::print_matrix(kernel);
        }
    }

}

