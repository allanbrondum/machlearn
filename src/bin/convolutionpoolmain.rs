use rayon::iter::ParallelBridge;

use machlearn::datasets::{convolutiontest, imagedatasets};
use machlearn::datasets::convolutiontest::KERNEL_INDEX;
use machlearn::matrix::{Matrix, MatrixDimensions, MatrixLinearIndex, MatrixT};
use machlearn::neuralnetwork::{ActivationFunction, Ampl, ConvolutionalLayer, Layer, LayerContainer, PoolLayer, Sample};
use machlearn::neuralnetwork;
use machlearn::neuralnetwork::Network;
use machlearn::vector::Vector;

const SYMBOLS: usize = 2;
const NY: Ampl = 0.001;
const WINDOW_DIMENSION: MatrixDimensions = MatrixDimensions::new(3, 3);

fn main() {
    let layer1 = ConvolutionalLayer::new(convolutiontest::INPUT_INDEX, KERNEL_INDEX.dimensions, SYMBOLS);
    let layer2 = PoolLayer::new_max(layer1.get_single_kernel_output_indexing(), SYMBOLS, WINDOW_DIMENSION);
    let output_indexing = layer2.get_output_indexing();
    let mut network = Network::new(vec!(
        LayerContainer::new(Box::new(layer1), ActivationFunction::relu()),
        LayerContainer::new(Box::new(layer2), ActivationFunction::identity())));

    let output_indexes: Vec<_> = (0..SYMBOLS).into_iter().map(|i| output_indexing.add_matrix_offset(i)).collect();
    imagedatasets::print_samples(get_learning_samples().take(10), convolutiontest::INPUT_INDEX, &output_indexes);

    network.set_random_weights_seed(1);

    const LEARNING_SAMPLES: usize = 10_000;
    neuralnetwork::run_learning_iterations(&mut network, get_learning_samples().take(LEARNING_SAMPLES), NY, false, 100);

    if false {
        neuralnetwork::run_learning_iterations(&mut network, get_learning_samples().take(20), NY, true, 100);
    }

    const TEST_SAMPLES: usize = 1000;
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork::run_test_iterations_parallel(&network, get_test_samples().take(TEST_SAMPLES).par_bridge());

    if true {
        imagedatasets::test_correct_percentage(&network, get_test_samples().take(10),  convolutiontest::INPUT_INDEX, &output_indexes, true);
    }
    let pct_correct = imagedatasets::test_correct_percentage(&network, get_test_samples().take(TEST_SAMPLES), convolutiontest::INPUT_INDEX, &output_indexes, false);

    println!("error squared: {:.5}", errsqr);
    println!("% correct: {:.2}", pct_correct);

    if true { // print kernels
        println!("network:\n {}", network);
    }

}

fn get_test_samples() -> impl Iterator<Item=Sample> {
    pool_output(convolutiontest::get_test_samples(SYMBOLS), convolutiontest::OUTPUT_INDEX, SYMBOLS, WINDOW_DIMENSION)
}

fn get_learning_samples() -> impl Iterator<Item=Sample> {
    pool_output(convolutiontest::get_learning_samples(SYMBOLS), convolutiontest::OUTPUT_INDEX, SYMBOLS, WINDOW_DIMENSION)
}

fn pool_output(samples: impl Iterator<Item=Sample>, output_indexing: MatrixLinearIndex, output_matrix_count: usize, window_dimension: MatrixDimensions) -> impl Iterator<Item=Sample> {
    let pool_layer = PoolLayer::new_max(output_indexing, output_matrix_count, window_dimension);
    samples.map(move |Sample(input, output)|
        Sample(input, pool_layer.evaluate_input_without_activation(&output)))
}


