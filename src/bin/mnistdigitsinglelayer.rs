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
use machlearn::datasets::{imagedatasets, mnistdigits};
use machlearn::neuralnetwork::Network;

fn main() {
    let mut network = Network::new_logistic_sigmoid(vec!(mnistdigits::IMAGE_PIXEL_COUNT, 10)); // single layer
    mnistdigits::print_data_examples();

    let read_from_file = false;
    if !read_from_file {
        network.set_random_weights_seed(0);

        const LEARNING_SAMPLES: usize = 60_000;
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

    mnistdigits::test_correct_percentage(&network, mnistdigits::get_test_samples().take(20).par_bridge(), true);
    let pct_correct = mnistdigits::test_correct_percentage(&network, mnistdigits::get_test_samples().take(TEST_SAMPLES).par_bridge(), false);

    println!("error squared: {:.5}", errsqr);
    println!("% correct: {:.2}", pct_correct);

    if false { // print kernels
        let weights = network.get_all_weights()[0][0];
        for row in 0..weights.row_count() {
            let mut row_elms: Vec<_> = weights.row_iter(row).copied().collect();
            let kernel = MutSliceView::new_row_stride(mnistdigits::IMAGE_WIDTH_HEIGHT, mnistdigits::IMAGE_WIDTH_HEIGHT,
                                                      &mut row_elms,
                                                      mnistdigits::IMAGE_WIDTH_HEIGHT);
            println!("kernel {}:\n", row);
            imagedatasets::print_matrix(&kernel);
        }
    }

    neuralnetwork::write_network_to_file(&network, "mnist_tmp_weights.json");
}

