use std::any::Any;
use std::fmt::{Display, Formatter};
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use crate::datasets::imagedatasets;
use crate::matrix::{Matrix, MatrixDimensions, MatrixLinearIndex, MatrixT, MutSliceView, SliceView};
use crate::neuralnetwork::{Ampl, Layer};
use crate::vector::Vector;

#[derive(Debug, Clone)]
pub struct ConvolutionalLayer
{
    input_matrix_index: MatrixLinearIndex,
    kernel_dimension: MatrixDimensions,
    kernels: Vec<Matrix<Ampl>>,
}

impl ConvolutionalLayer {
    pub fn new(input_matrix_index: MatrixLinearIndex, kernel_dimension: MatrixDimensions, kernels: usize) -> ConvolutionalLayer {
        ConvolutionalLayer {
            input_matrix_index,
            kernel_dimension,
            kernels: (0..kernels).map(|_| Matrix::new_with_dimension(kernel_dimension)).collect(),
        }
    }
}

impl Layer for ConvolutionalLayer {

    fn get_weights(&self) -> Vec<&Matrix<Ampl>> {
        self.kernels.iter().collect()
    }

    fn set_weights(&mut self, new_weights: Vec<Matrix<Ampl>>) {
        assert_eq!(self.kernels.len(), new_weights.len(), "Should have {} kernels, was {}", self.kernels.len(), new_weights.len());
        self.kernels.clear();
        for kernel in new_weights {
            if kernel.dimensions() != self.kernel_dimension {
                panic!("Kernel dimensions for layer {} does not equals dimension of weights to set {}", self.kernel_dimension, kernel.dimensions());
            }
            self.kernels.push(kernel);
        }
    }

    fn get_input_dimension(&self) -> usize {
        self.input_matrix_index.linear_dimension_length()
    }

    fn get_output_dimension(&self) -> usize {
        self.get_single_kernel_output_indexing().linear_dimension_length() * self.kernels.len()
    }

    fn set_random_weights(&mut self) {
        let mut rng = rand::thread_rng();
        self.set_random_weights_impl(rng);
    }

    fn set_random_weights_seed(&mut self, seed: u64) {
        let rng: Pcg64 = Seeder::from(0).make_rng();
        self.set_random_weights_impl(rng);
    }

    fn evaluate_input_without_activation(&self, input: &Vector<Ampl>) -> Vector<Ampl> {
        let mut output = Vector::<Ampl>::new(self.get_output_dimension());
        let single_kernel_output_indexing = self.get_single_kernel_output_indexing();

        let input_kernel_indexing = self.input_matrix_index.with_dimensions(self.kernel_dimension);

        for (kernel_index, kernel) in self.kernels.iter().enumerate() {
            let mut kernel_output_matrix = MutSliceView::new(
                single_kernel_output_indexing.add_slice_offset(kernel_index * single_kernel_output_indexing.linear_dimension_length()),
                output.as_mut_slice());

            for (index, elm) in kernel_output_matrix.iter_mut_enum() {
                let input_kernel_view = SliceView::new(
                    input_kernel_indexing.add_row_col_offset(index.0, index.1),
                    input.as_slice());
                *elm = kernel.scalar_prod(&input_kernel_view);
            }
        }
        output
    }


    fn back_propagate_without_activation(&mut self, input: &Vector<Ampl>, delta_output: Vector<Ampl>, ny: Ampl) -> Vector<Ampl> {
        // calculate the return value: partial derivative of error squared with respect to input state coordinates
        let mut gamma_input = Vector::<Ampl>::new(self.get_input_dimension());

        let single_kernel_output_indexing = self.get_single_kernel_output_indexing();
        let input_kernel_indexing_base = self.input_matrix_index.with_dimensions(self.kernel_dimension);

        for (kernel_index, kernel) in self.kernels.iter_mut().enumerate() {
            let mut kernel_delta_output_matrix = SliceView::new(
                single_kernel_output_indexing.add_slice_offset(kernel_index * single_kernel_output_indexing.linear_dimension_length()),
                delta_output.as_slice());

            let mut kernel_diff = Matrix::new_with_dimension(self.kernel_dimension);
            for (index, elm) in kernel_delta_output_matrix.iter_enum() {
                let input_kernel_indexing = input_kernel_indexing_base.add_row_col_offset(index.0, index.1);
                let mut gamma_input_kernel_view = MutSliceView::new(
                    input_kernel_indexing,
                    gamma_input.as_mut_slice());
                let mut kernel_copy = kernel.clone();
                kernel_copy.mul_scalar_assign(*elm);
                gamma_input_kernel_view.add_matrix_assign(&kernel_copy);

                let mut input_kernel_view = SliceView::new(input_kernel_indexing, input.as_slice()).copy_to_matrix();
                input_kernel_view.mul_scalar_assign(*elm);
                kernel_diff.add_matrix_assign(&input_kernel_view);
            }

            // adjust kernel
            kernel_diff.mul_scalar_assign(-ny);
            kernel.add_matrix_assign(&kernel_diff);
        }

        gamma_input
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl ConvolutionalLayer {

    /// Indexing for output for a single kernel
    pub fn get_single_kernel_output_indexing(&self) -> MatrixLinearIndex {
        MatrixLinearIndex::new_row_stride(MatrixDimensions {
            rows: self.input_matrix_index.dimensions.rows - self.kernel_dimension.rows + 1,
            columns: self.input_matrix_index.dimensions.columns - self.kernel_dimension.columns + 1,
        })
    }

    fn set_random_weights_impl<R: Rng>(&mut self, mut rng: R) {
        let range = 1.0 / (self.kernel_dimension.cell_count()) as Ampl;
        self.kernels.iter_mut().for_each(|kern| kern.apply_ref(|_| rng.gen_range(0.0..range)));
    }
}

impl Display for ConvolutionalLayer
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {

        for (index, kernel) in self.kernels.iter().enumerate() {
            write!(f, "kernel {}\n", index)?;
            imagedatasets::print_matrix(kernel);
        }

        std::fmt::Result::Ok(())
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(expected_kernel_output_indexing, layer.get_single_kernel_output_indexing());
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
        let output = layer.evaluate_input_without_activation(&input);
        let output_indexing = layer.get_single_kernel_output_indexing();
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
}