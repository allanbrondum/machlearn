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

use machlearn::neuralnetwork::{ActivationFunction, Ampl, ConvolutionalLayer, FullyConnectedLayer, Layer, LayerContainer, Sample};
use machlearn::neuralnetwork;
use machlearn::vector::Vector;
use machlearn::matrix::{Matrix, MatrixT, MutSliceView};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use machlearn::datasets::{convolutiontest, imagedatasets};
use machlearn::datasets::convolutiontest::KERNEL_INDEX;
use machlearn::neuralnetwork::Network;

const SYMBOLS: usize = 2;
const NY: Ampl = 0.03;

fn main() {
    let layer = ConvolutionalLayer::new(convolutiontest::INPUT_INDEX, KERNEL_INDEX.dimensions, SYMBOLS);
    let mut network = Network::new(
        vec!(LayerContainer::new(Box::new(layer), ActivationFunction::sigmoid())),
        false);
    convolutiontest::print_data_examples(SYMBOLS);

    network.set_random_weights_seed(1);

    const LEARNING_SAMPLES: usize = 1_000;
    // neuralnetwork::run_learning_iterations(&mut network, convolutiontest::get_learning_samples(SYMBOLS).take(LEARNING_SAMPLES), NY, false);

    if true {
        neuralnetwork::run_learning_iterations(&mut network, convolutiontest::get_learning_samples(SYMBOLS).take(20), NY, true);
    }

    const TEST_SAMPLES: usize = 1000;
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork::run_test_iterations_parallel(&network, convolutiontest::get_test_samples(SYMBOLS).take(TEST_SAMPLES).par_bridge());

    convolutiontest::test_correct_percentage(SYMBOLS, &network, convolutiontest::get_test_samples(SYMBOLS).take(20), true);
    let pct_correct = convolutiontest::test_correct_percentage(SYMBOLS, &network, convolutiontest::get_test_samples(SYMBOLS).take(TEST_SAMPLES), false);

    println!("error squared: {:.5}", errsqr);
    println!("% correct: {:.2}", pct_correct);

    if true { // print kernels
        let weights = &network.get_all_weights()[0];
        for (i, &kernel) in weights.iter().enumerate() {
            println!("kernel {}:\n", i);
            imagedatasets::print_matrix(kernel);
        }
    }

}

