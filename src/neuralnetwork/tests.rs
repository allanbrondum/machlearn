use crate::matrix::{MatrixDimensions, MatrixIndex, MatrixLinearIndex};
use crate::neuralnetwork::{ConvolutionalLayer, Layer};

#[test]
fn convolutional_kernel_output_dim() {
    let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
    let layer = ConvolutionalLayer::new(input_indexing, MatrixDimensions{rows: 4, columns: 5}, 5);

    let expected_kernel_output_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 9, columns: 6});
    assert_eq!(expected_kernel_output_indexing, layer.get_kernel_output_indexing());
    assert_eq!(5 * (9 * 6), layer.get_output_dimension());
}