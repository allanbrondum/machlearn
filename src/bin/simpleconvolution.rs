use rayon::iter::ParallelBridge;

use machlearn::datasets::{convolutiontest, imagedatasets};
use machlearn::datasets::convolutiontest::KERNEL_INDEX;
use machlearn::matrix::MatrixT;
use machlearn::neuralnetwork::{ActivationFunction, Ampl, ConvolutionalLayer, Layer, LayerContainer};
use machlearn::neuralnetwork;
use machlearn::neuralnetwork::Network;

const SYMBOLS: usize = 2;
const NY: Ampl = 0.001;

fn main() {
    let layer = ConvolutionalLayer::new(convolutiontest::INPUT_INDEX, KERNEL_INDEX.dimensions, SYMBOLS);
    let mut network = Network::new(vec!(LayerContainer::new(Box::new(layer), ActivationFunction::relu01())));

    if false {
        convolutiontest::print_data_examples(SYMBOLS);
    }

    network.set_random_weights_seed(1);

    const LEARNING_SAMPLES: usize = 10000;
    neuralnetwork::run_learning_iterations(&mut network, convolutiontest::get_learning_samples(SYMBOLS).take(LEARNING_SAMPLES), NY, false, 100);

    if false {
        neuralnetwork::run_learning_iterations(&mut network, convolutiontest::get_learning_samples(SYMBOLS).take(20), NY, true, 100);
    }

    const TEST_SAMPLES: usize = 1000;
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork::run_test_iterations_parallel(&network, convolutiontest::get_test_samples(SYMBOLS).take(TEST_SAMPLES).par_bridge());

    if true {
        convolutiontest::test_correct_percentage(SYMBOLS, &network, convolutiontest::get_test_samples(SYMBOLS).take(10), true);
    }
    let pct_correct = convolutiontest::test_correct_percentage(SYMBOLS, &network, convolutiontest::get_test_samples(SYMBOLS).take(TEST_SAMPLES), false);

    println!("error squared: {:.5}", errsqr);
    println!("% correct: {:.2}", pct_correct);

    if true { // print kernels
        println!("network:\n {}", network);
    }

}

