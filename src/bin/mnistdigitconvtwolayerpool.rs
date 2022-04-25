use rayon::iter::ParallelBridge;

use machlearn::datasets::{imagedatasets, mnistdigits};
use machlearn::matrix::{MatrixDimensions, MatrixT, SliceView};
use machlearn::neuralnetwork::{ActivationFunction, Ampl, ConvolutionalLayer, FullyConnectedLayer, Layer, LayerContainer, PoolLayer, Sample};
use machlearn::neuralnetwork;
use machlearn::neuralnetwork::Network;

const KERNELS: usize = 4;
const KERNEL_DIMENSION: MatrixDimensions = MatrixDimensions::new(5, 5);

fn main() {

    let mut layer1 = ConvolutionalLayer::new(
        mnistdigits::INPUT_INDEX,
        KERNEL_DIMENSION,
        KERNELS);
    let kernel_indexing = layer1.get_single_kernel_output_indexing();
    layer1.set_kernel_weights(imagedatasets::create_kernel_patterns(KERNEL_DIMENSION, KERNELS));
    // layer1.set_random_weights_seed(0);
    let mut layer2 = PoolLayer::new_mean(layer1.get_single_kernel_output_indexing(), KERNELS, MatrixDimensions::new(3, 3));
    let mut layer3 = FullyConnectedLayer::new(layer2.get_output_dimension(), 10);
    layer3.set_random_weights_seed(0);

    if false {
        print_conv_layer_output(&layer1, mnistdigits::get_learning_samples().take(10));
    }

    let mut network = Network::new(vec!(
        LayerContainer::new(Box::new(layer1), ActivationFunction::relu()),
        LayerContainer::new(Box::new(layer2), ActivationFunction::identity()),
        LayerContainer::new(Box::new(layer3), ActivationFunction::sigmoid()),
    ));

    if false {
        mnistdigits::print_data_examples();
    }

    const NY: Ampl = 0.001;

    let read_from_file = false;
    if !read_from_file {
        println!("network:\n{}", network);

        const LEARNING_SAMPLES: usize = 20_000;
        neuralnetwork::run_learning_iterations(&mut network, mnistdigits::get_learning_samples().take(LEARNING_SAMPLES), NY, false, 10);
    } else {
        // read weights from file
        neuralnetwork::read_network_from_file(&mut network, "mnist_convtwolayer_weights.json");
        // println!("network \n{}", network);
    }

    if false {
        neuralnetwork::run_learning_iterations(&mut network, mnistdigits::get_learning_samples().take(20), NY, true, 1);
    }

    const TEST_SAMPLES: usize = 1000;
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork::run_test_iterations_parallel(&network, mnistdigits::get_test_samples().take(TEST_SAMPLES).par_bridge());

    mnistdigits::test_correct_percentage(&network, mnistdigits::get_test_samples().take(10), true);
    let pct_correct = mnistdigits::test_correct_percentage(&network, mnistdigits::get_test_samples().take(TEST_SAMPLES), false);

    println!("error squared: {:.5}", errsqr);
    println!("% correct: {:.2}", pct_correct);

    println!("network:\n{}", network);
    let weights = network.get_all_weights()[1][0];
    for output_index in 0..10 {
        for kernel_index in 0..KERNELS {
            let indexing = kernel_indexing.add_slice_offset(kernel_index * kernel_indexing.linear_dimension_length()
                + output_index * KERNELS * kernel_indexing.linear_dimension_length());
            let view = SliceView::new(indexing, weights.as_slice());
            println!("output {} kernel {}\n", output_index, kernel_index);
            imagedatasets::print_matrix(&view);
        }
    }

    neuralnetwork::write_network_to_file(&network, "mnist_tmp_weights.json");
}

fn print_conv_layer_output(conv_layer: &ConvolutionalLayer, samples: impl Iterator<Item=Sample>) {
    for (index, sample) in samples.enumerate() {
        let correct_matrix = SliceView::new (mnistdigits::OUTPUT_INDEX, &sample.1);
        println!("Correct output {}", index);
        imagedatasets::print_matrix(&correct_matrix);

        let output = conv_layer.evaluate_input_without_activation(&sample.0);
        let output_kernel_indexing = conv_layer.get_single_kernel_output_indexing();
        for i in 0..KERNELS {
            let output_matrix = SliceView::new(output_kernel_indexing.add_slice_offset(i * output_kernel_indexing.linear_dimension_length()), output.as_slice());
            println!("Output {} kernel {}", index, i);
            imagedatasets::print_matrix(&output_matrix);
        }
    }
}

