use std::any::Any;
use std::fmt::{Display, Formatter};
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use crate::datasets::imagedatasets;
use crate::matrix::{Matrix, MatrixDimensions, MatrixLinearIndex, MatrixT, MutSliceView, SliceView};
use crate::neuralnetwork::{Ampl, Layer};
use crate::vector::Vector;
use serde::{Deserialize, Serialize};

/// Layer that interprets input state as a two-dimensional matrix and "scans" the input with
/// each of its kernels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvolutionalLayer
{
    input_matrix_index: MatrixLinearIndex,
    kernel_dimension: MatrixDimensions,
    kernels: Vec<Matrix<Ampl>>,
    biases: Vector<Ampl>,
}

impl ConvolutionalLayer {
    pub fn new(input_matrix_index: MatrixLinearIndex, kernel_dimension: MatrixDimensions, kernels: usize) -> ConvolutionalLayer {
        ConvolutionalLayer {
            input_matrix_index,
            kernel_dimension,
            kernels: (0..kernels).map(|_| Matrix::new_with_dimension(kernel_dimension)).collect(),
            biases: Vector::new(kernels),
        }
    }
}

#[typetag::serde]
impl Layer for ConvolutionalLayer {

    fn get_weights(&self) -> Vec<&Matrix<Ampl>> {
        self.kernels.iter().collect()
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
                single_kernel_output_indexing.add_matrix_offset(kernel_index),
                output.as_mut_slice());

            for (index, elm) in kernel_output_matrix.iter_mut_enum() {
                let input_kernel_view = SliceView::new(
                    input_kernel_indexing.add_row_col_offset(index.0, index.1),
                    input.as_slice());
                *elm = kernel.scalar_prod(&input_kernel_view) + self.biases[kernel_index];
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
                single_kernel_output_indexing.add_matrix_offset(kernel_index),
                delta_output.as_slice());

            let mut kernel_diff = Matrix::new_with_dimension(self.kernel_dimension);
            let mut bias_diff = 0.0;
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
                bias_diff += *elm;
            }

            // adjust kernel
            kernel_diff.mul_scalar_assign(-ny);
            kernel.add_matrix_assign(&kernel_diff);
            self.biases[kernel_index] -= ny * bias_diff;
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

    pub fn set_kernel_weights(&mut self, new_weights: Vec<Matrix<Ampl>>) {
        assert_eq!(self.kernels.len(), new_weights.len(), "Should have {} kernels, was {}", self.kernels.len(), new_weights.len());
        self.kernels.clear();
        for kernel in new_weights {
            if kernel.dimensions() != self.kernel_dimension {
                panic!("Kernel dimensions for layer {} does not equals dimension of weights to set {}", self.kernel_dimension, kernel.dimensions());
            }
            self.kernels.push(kernel);
        }
    }

    pub fn set_biases(&mut self, new_biases: Vector<Ampl>) {
        assert_eq!(self.biases.len(), new_biases.len(), "Bias dimension {} does not equals dimension of biases to set {}", self.biases.len(), new_biases.len());
        self.biases = new_biases;
    }

    pub fn get_biases(&self) -> &Vector<Ampl> {
        &self.biases
    }
}

impl Display for ConvolutionalLayer
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {

        for (index, kernel) in self.kernels.iter().enumerate() {
            write!(f, "kernel {}\n", index)?;
            imagedatasets::print_matrix(kernel);
            write!(f, "bias: {:.5}\n", self.biases[index])?;
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
    fn input_output_dim() {
        let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
        let layer = ConvolutionalLayer::new(input_indexing, MatrixDimensions{rows: 4, columns: 5}, 5);

        // assert input dimension
        assert_eq!(12 * 10, layer.get_input_dimension());

        // assert output indexing and output dimension
        let expected_kernel_output_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 9, columns: 6});
        assert_eq!(expected_kernel_output_indexing, layer.get_single_kernel_output_indexing());
        assert_eq!(5 * (9 * 6), layer.get_output_dimension());
    }

    #[test]
    fn evaluate_input() {
        let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
        let mut layer = ConvolutionalLayer::new(input_indexing, MatrixDimensions{rows: 3, columns: 3}, 2);

        let mut weights: Vec<_> = layer.get_weights().into_iter().cloned().collect();
        weights[0][(1, 1)] = 1.0;
        weights[1][(1, 0)] = 1.0;
        weights[1][(1, 2)] = 1.0;
        weights[1][(0, 1)] = 1.0;
        weights[1][(2, 1)] = 1.0;
        layer.set_kernel_weights(weights);
        let mut bias = Vector::new(2);
        bias[0] = 0.1;
        bias[1] = 0.2;
        layer.set_biases(bias);

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
        expected_output_matrix1.iter_mut_enum().for_each(|(_, elm)| *elm = 0.1);
        expected_output_matrix1[(2, 5)] += 1.0;
        assert_eq!(expected_output_matrix1, output_matrix1.copy_to_matrix());
        let mut expected_output_matrix2 = Matrix::new_with_indexing(output_indexing);
        expected_output_matrix2.iter_mut_enum().for_each(|(_, elm)| *elm = 0.2);
        expected_output_matrix2[(2, 4)] += 1.0;
        expected_output_matrix2[(2, 6)] += 1.0;
        expected_output_matrix2[(1, 5)] += 1.0;
        expected_output_matrix2[(3, 5)] += 1.0;
        assert_eq!(expected_output_matrix2, output_matrix2.copy_to_matrix());
    }

    #[test]
    fn back_propagate() {
        let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
        let mut layer = ConvolutionalLayer::new(input_indexing, MatrixDimensions{rows: 3, columns: 3}, 2);

        let mut weights: Vec<_> = layer.get_weights().into_iter().cloned().collect();
        weights[0][(1, 1)] = 1.0;
        weights[1][(1, 0)] = 1.0;
        weights[1][(1, 2)] = 1.0;
        weights[1][(0, 1)] = 1.0;
        weights[1][(2, 1)] = 1.0;
        layer.set_kernel_weights(weights.clone());
        let mut bias = Vector::new(2);
        bias[0] = 0.1;
        bias[1] = 0.2;
        layer.set_biases(bias.clone());

        // set input
        let mut input: Vector<Ampl> = Vector::new(input_indexing.required_linear_array_length());
        let mut input_matrix = MutSliceView::new(input_indexing, input.deref_mut());
        *input_matrix.elm_mut(2, 6) = 3.0;
        *input_matrix.elm_mut(7, 5) = 3.0;

        // delta output
        let output_indexing = layer.get_single_kernel_output_indexing();
        let mut delta_output= Vector::new(layer.get_output_dimension());
        let mut delta_output_matrix1 = MutSliceView::new(output_indexing, delta_output.as_mut_slice());
        *delta_output_matrix1.elm_mut(2, 5) = 0.5;
        let mut delta_output_matrix2 = MutSliceView::new(output_indexing.add_slice_offset(output_indexing.linear_dimension_length()), delta_output.as_mut_slice());
        *delta_output_matrix2.elm_mut(6, 5) = 2.0;

        // back propagate
        const NY: f64 = 0.1;
        let gamma_input = layer.back_propagate_without_activation(&input, delta_output, NY);
        let gamma_input_matrix = SliceView::new(input_indexing, gamma_input.as_slice());
        imagedatasets::print_matrix(&gamma_input_matrix);

        // assert gamma input
        let mut expected_gamma_input_matrix = Matrix::new_with_indexing(input_indexing);
        *expected_gamma_input_matrix.elm_mut(3, 6) = 0.5;
        *expected_gamma_input_matrix.elm_mut(6, 6) = 2.0;
        *expected_gamma_input_matrix.elm_mut(8, 6) = 2.0;
        *expected_gamma_input_matrix.elm_mut(7, 5) = 2.0;
        *expected_gamma_input_matrix.elm_mut(7, 7) = 2.0;
        assert_eq!(expected_gamma_input_matrix, gamma_input_matrix.copy_to_matrix());

        // assert weight adjustments
        weights[0][(0, 1)] -= 3.0 * 0.5 * NY;
        weights[1][(1, 0)] -= 3.0 * 2.0 * NY;
        let mut actual_weights: Vec<_> = layer.get_weights().into_iter().cloned().collect();
        println!("{}", actual_weights[0]);
        println!("{}", actual_weights[1]);
        assert_eq!(weights[0], actual_weights[0], "kernels 0");
        assert_eq!(weights[1], actual_weights[1], "kernels 0");

        // assert bias adjustments
        bias[0] -= 0.5 * NY;
        bias[1] -= 2.0 * NY;
        let mut actual_biases = layer.get_biases().clone();
        assert_eq!(bias, actual_biases);

    }
}