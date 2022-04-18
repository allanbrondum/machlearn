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

use machlearn::neuralnetwork::{ActivationFunction, Ampl, FullyConnectedLayer, Layer, LayerContainer, Sample};
use machlearn::neuralnetwork;
use machlearn::vector::Vector;
use machlearn::matrix::{Matrix, MatrixT, MutSliceView, SliceView};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use machlearn::datasets::{imagedatasets, mnistdigits};
use machlearn::neuralnetwork::Network;

fn main() {

    let layer = FullyConnectedLayer::new(mnistdigits::IMAGE_PIXEL_COUNT, 10);
    let mut network = Network::new(
        vec!(LayerContainer::new(Box::new(layer), ActivationFunction::sigmoid())),
        false);

    if true {
        mnistdigits::print_data_examples();
    }

    const NY: Ampl = 0.01;

    let read_from_file = true;
    if !read_from_file {
        // learn weights
        network.set_random_weights_seed(0);

        const LEARNING_SAMPLES: usize = 50_000;
        neuralnetwork::run_learning_iterations(&mut network, mnistdigits::get_learning_samples().take(LEARNING_SAMPLES), NY, false, 1000);
    } else {
        // read weights from file
        neuralnetwork::read_network_from_file(&mut network, "mnist_singlelayer_weights.json");
        // println!("network \n{}", network);
    }

    if false {
        neuralnetwork::run_learning_iterations(&mut network, mnistdigits::get_learning_samples().take(20), NY, true, 1000);
    }

    const TEST_SAMPLES: usize = 1000;
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork::run_test_iterations_parallel(&network, mnistdigits::get_test_samples().take(TEST_SAMPLES).par_bridge());

    mnistdigits::test_correct_percentage(&network, mnistdigits::get_test_samples().take(10), true);
    let pct_correct = mnistdigits::test_correct_percentage(&network, mnistdigits::get_test_samples().take(TEST_SAMPLES), false);

    println!("error squared: {:.5}", errsqr);
    println!("% correct: {:.2}", pct_correct);

    println!("network:\n{}", network);
    let weights = network.get_all_weights()[0][0];
    for i in 0..10 {
        let indexing = mnistdigits::INPUT_INDEX.add_slice_offset(i * mnistdigits::IMAGE_PIXEL_COUNT);
        let view = SliceView::new(indexing, weights.as_slice());
        println!("output {}\n", i);
        imagedatasets::print_matrix(&view);
    }

    neuralnetwork::write_network_to_file(&network, "mnist_tmp_weights.json");
}

