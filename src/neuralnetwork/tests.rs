use std::ops::DerefMut;
use crate::matrix::{Matrix, MatrixDimensions, MatrixIndex, MatrixLinearIndex, MatrixT, MutSliceView, SliceView};
use crate::{datasets, neuralnetwork};
use crate::datasets::imagedatasets;
use crate::neuralnetwork::{Ampl, ConvolutionalLayer, Layer};
use crate::vector::Vector;

#[test]
fn convolutional_kernel_output_dim() {
    let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
    let layer = ConvolutionalLayer::new(input_indexing, MatrixDimensions{rows: 4, columns: 5}, 5);

    let expected_kernel_output_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 9, columns: 6});
    assert_eq!(expected_kernel_output_indexing, layer.get_kernel_output_indexing());
    assert_eq!(5 * (9 * 6), layer.get_output_dimension());
}

#[test]
fn convolutional_evaluate_input() {
    let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
    let mut layer = ConvolutionalLayer::new(input_indexing, MatrixDimensions{rows: 3, columns: 3}, 2);

    let mut weights: Vec<_> = layer.get_weights().into_iter().cloned().collect();
    weights[0][(1, 1)] = 1.0;
    weights[1][(1, 0)] = 1.0;
    weights[1][(1, 2)] = 1.0;
    weights[1][(0, 1)] = 1.0;
    weights[1][(2, 1)] = 1.0;
    layer.set_weights(weights);

    // set input
    let mut input: Vector<Ampl> = Vector::new(input_indexing.required_linear_array_length());
    let mut input_matrix = MutSliceView::new(input_indexing, input.deref_mut());
    *input_matrix.elm_mut(3, 6) = 1.0;

    // calculate output
    let output = layer.evaluate_input(&input, neuralnetwork::sigmoid_logistic);
    let output_indexing = layer.get_kernel_output_indexing();
    let output_matrix1 = SliceView::new(output_indexing, output.as_slice());
    let output_matrix2 = SliceView::new(output_indexing.add_slice_offset(output_indexing.linear_dimension_length()), output.as_slice());

    imagedatasets::print_matrix(&output_matrix1);
    imagedatasets::print_matrix(&output_matrix2);

    let mut expected_output_matrix1 = Matrix::new_with_indexing(output_indexing);
    expected_output_matrix1[(2, 5)] = 1.0;
    assert_eq!(expected_output_matrix1, output_matrix1.copy_to_matrix());
    let mut expected_output_matrix2 = Matrix::new_with_indexing(output_indexing);
    expected_output_matrix2[(2, 4)] = 1.0;
    expected_output_matrix2[(2, 6)] = 1.0;
    expected_output_matrix2[(1, 5)] = 1.0;
    expected_output_matrix2[(3, 5)] = 1.0;
    assert_eq!(expected_output_matrix2, output_matrix2.copy_to_matrix());
}