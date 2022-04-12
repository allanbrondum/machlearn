use std::any::Any;
use std::fmt::{Display, Formatter};
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
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
        self.get_kernel_output_indexing().linear_dimension_length() * self.kernels.len()
    }

    fn set_random_weights(&mut self) {
        let mut rng = rand::thread_rng();
        self.set_random_weights_impl(rng);
    }

    fn set_random_weights_seed(&mut self, seed: u64) {
        let rng: Pcg64 = Seeder::from(0).make_rng();
        self.set_random_weights_impl(rng);
    }

    fn evaluate_input(&self, input: &Vector<Ampl>, sigmoid: fn(Ampl) -> Ampl) -> Vector<Ampl> {
        // let input_matrix = SliceView::new(self.input_matrix_index, input.as_slice());

        let mut output = Vector::<Ampl>::new(self.get_output_dimension());
        let kernel_output_indexing = self.get_kernel_output_indexing();

        let input_kernel_indexing = self.input_matrix_index.with_dimensions(self.kernel_dimension);

        for (kernel_index, kernel) in self.kernels.iter().enumerate() {
            let mut kernel_output_matrix = MutSliceView::new(
                kernel_output_indexing.add_slice_offset(kernel_index * kernel_output_indexing.linear_dimension_length()),
                output.as_mut_slice());

            for (index, elm) in kernel_output_matrix.iter_mut_enum() {
                let input_kernel_view = SliceView::new(
                    input_kernel_indexing.add_row_col_offset(index.0, index.1),
                    input.as_slice());
                *elm = kernel.scalar_prod(&input_kernel_view);
            }
        }

        output.apply(sigmoid)
    }

    fn back_propagate(&mut self, input: &Vector<Ampl>, gamma_output: &Vector<Ampl>, sigmoid_derived: fn(Ampl) -> Ampl, ny: Ampl) -> Vector<Ampl> {
        // let delta_output = self.weights.mul_vector(input).apply(sigmoid_derived).mul_comp(gamma_output);
        // let gamma_input = self.weights.mul_vector_lhs(&delta_output);
        //
        // // adjust weights
        // self.weights -= ny * delta_output.to_matrix().mul_mat(&input.clone().to_matrix().transpose());
        //
        // gamma_input

        Vector::new(0)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl ConvolutionalLayer {
    fn set_random_weights_impl<R: Rng>(&mut self, mut rng: R) {
        self.kernels.iter_mut().for_each(|kern| kern.apply_ref(|_| rng.gen_range(-1.0..1.0)));
    }

    fn get_kernel_output_indexing(&self) -> MatrixLinearIndex {
        MatrixLinearIndex::new_row_stride(MatrixDimensions {
            rows: self.input_matrix_index.dimensions.rows - self.kernel_dimension.rows + 1,
            columns: self.input_matrix_index.dimensions.columns - self.kernel_dimension.columns + 1,
        })
    }
}

impl Display for ConvolutionalLayer
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (index, kernel) in self.kernels.iter().enumerate() {
            write!(f, "kernel {}\n{}", index, kernel)?;
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
}