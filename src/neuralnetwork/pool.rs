use std::any::Any;
use std::fmt::{Display, Formatter};
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use crate::datasets::imagedatasets;
use crate::matrix::{Matrix, MatrixDimensions, MatrixLinearIndex, MatrixT, MutSliceView, SliceView};
use crate::neuralnetwork::{Ampl, cmp_ampl, cmp_ampl_ref, Layer};
use crate::vector::Vector;
use serde::{Deserialize, Serialize};

/// Layer that interprets input state as a series of two-dimensional matrices and reduces
/// the dimensions of the input matrices by "scanning" the input taking max value in windows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolLayer
{
    input_matrix_index: MatrixLinearIndex,
    input_matrix_count: usize,
    window_dimension: MatrixDimensions,
    pool_mode: PoolMode,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum PoolMode
{
    Max,
    Mean,
}

impl PoolLayer {
    pub fn new_max(input_matrix_index: MatrixLinearIndex, input_matrix_count: usize, window_dimension: MatrixDimensions) -> PoolLayer {
        PoolLayer {
            input_matrix_index,
            input_matrix_count,
            window_dimension,
            pool_mode: PoolMode::Max
        }
    }

    pub fn new_mean(input_matrix_index: MatrixLinearIndex, input_matrix_count: usize, window_dimension: MatrixDimensions) -> PoolLayer {
        PoolLayer {
            input_matrix_index,
            input_matrix_count,
            window_dimension,
            pool_mode: PoolMode::Mean
        }
    }
}

#[typetag::serde]
impl Layer for PoolLayer {

    fn get_weights(&self) -> Vec<&Matrix<Ampl>> {
        Vec::with_capacity(0)
    }

    fn get_input_dimension(&self) -> usize {
        self.input_matrix_count * self.input_matrix_index.linear_dimension_length()
    }

    fn get_output_dimension(&self) -> usize {
        self.input_matrix_count * self.get_output_indexing().linear_dimension_length()
    }

    fn set_random_weights(&mut self) {
    }

    fn set_random_weights_seed(&mut self, seed: u64) {
    }

    fn evaluate_input_without_activation(&self, input: &Vector<Ampl>) -> Vector<Ampl> {
        let mut output = Vector::<Ampl>::new(self.get_output_dimension());
        let output_indexing = self.get_output_indexing();

        for i in 0..self.input_matrix_count {
            let input_matrix_indexing = self.input_matrix_index.add_matrix_offset(i);

            let mut output_matrix = MutSliceView::new(
                output_indexing.add_matrix_offset(i),
                output.as_mut_slice());


            for (index, elm) in output_matrix.iter_mut_enum() {
                let window_indexing = input_matrix_indexing
                    .add_row_col_offset(index.0 * self.window_dimension.rows, index.1 * self.window_dimension.columns)
                    .with_dimensions(self.window_dimension);

                let input_window_matrix = SliceView::new(
                    window_indexing,
                    input.as_slice());

                *elm = match self.pool_mode {
                    PoolMode::Max =>
                        input_window_matrix.iter().copied().max_by(cmp_ampl_ref).unwrap(),
                    PoolMode::Mean =>
                        input_window_matrix.iter().copied().sum::<Ampl>() / input_window_matrix.cell_count() as Ampl,
                }
            }
        }
        output
    }


    fn back_propagate_without_activation(&mut self, input: &Vector<Ampl>, delta_output: Vector<Ampl>, ny: Ampl) -> Vector<Ampl> {
        let mut gamma_input = Vector::<Ampl>::new(self.get_input_dimension());
        let output_indexing = self.get_output_indexing();

        for i in 0..self.input_matrix_count {
            let input_matrix_indexing = self.input_matrix_index.add_matrix_offset(i);

            let delta_output_matrix = SliceView::new(
                output_indexing.add_matrix_offset(i),
                delta_output.as_slice());


            for (output_index, gamma_output_elm) in delta_output_matrix.iter_enum() {
                let window_indexing = input_matrix_indexing
                    .add_row_col_offset(output_index.0 * self.window_dimension.rows, output_index.1 * self.window_dimension.columns)
                    .with_dimensions(self.window_dimension);

                let input_window_matrix = SliceView::new(
                    window_indexing,
                    input.as_slice());

                let mut gamma_input_window_matrix = MutSliceView::new(
                    window_indexing,
                    gamma_input.as_mut_slice());
                //
                // let max = input_window_matrix.iter().copied()
                //     .max_by(cmp_ampl_ref).unwrap();
                // input_window_matrix.iter_enum()
                //     .for_each(|(index, elm)| {
                //         if *elm == max {
                //             *gamma_input_window_matrix.elm_mut(index.0, index.1) = *gamma_output_elm;
                //         }
                //     });

                match self.pool_mode {
                    PoolMode::Max => {
                        let max_index = input_window_matrix.iter_enum()
                            .max_by(|it1, it2| cmp_ampl(*it1.1, *it2.1))
                            .unwrap().0;

                        *gamma_input_window_matrix.elm_mut(max_index.0, max_index.1) = *gamma_output_elm;
                    },
                    PoolMode::Mean => {
                        gamma_input_window_matrix.iter_mut_enum().for_each(|(index, elm)| {
                            *elm = *gamma_output_elm / self.window_dimension.cell_count() as Ampl;
                        });

                    }
                }

            }
        }

        gamma_input
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl PoolLayer {

    /// Indexing for output for a single kernel
    pub fn get_output_indexing(&self) -> MatrixLinearIndex {
        MatrixLinearIndex::new_row_stride(MatrixDimensions {
            rows: self.input_matrix_index.dimensions.rows / self.window_dimension.rows,
            columns: self.input_matrix_index.dimensions.columns / self.window_dimension.columns,
        })
    }

}

impl Display for PoolLayer
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Result::Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ops::DerefMut;
    use crate::matrix::{Matrix, MatrixDimensions, MatrixIndex, MatrixLinearIndex, MatrixT, MutSliceView, SliceView};
    use crate::{datasets, neuralnetwork};
    use crate::datasets::imagedatasets;
    use crate::neuralnetwork::{Ampl, PoolLayer, Layer};
    use crate::vector::Vector;

    #[test]
    fn input_output_dim() {
        let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
        let layer = PoolLayer::new_max(input_indexing, 3, MatrixDimensions{rows: 2, columns: 3});

        // input
        assert_eq!(3 * (12 * 10), layer.get_input_dimension());

        // output
        let expected_output_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 6, columns: 3});
        assert_eq!(expected_output_indexing, layer.get_output_indexing());
        assert_eq!(3 * (6 * 3), layer.get_output_dimension());
    }


    #[test]
    fn evaluate_input() {
        let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
        let mut layer = PoolLayer::new_max(input_indexing, 3, MatrixDimensions{rows: 2, columns: 3});

        // set input
        let mut input: Vector<Ampl> = Vector::new(layer.get_input_dimension());
        let mut input_matrix1 = MutSliceView::new(input_indexing, input.deref_mut());
        *input_matrix1.elm_mut(1, 6) = 1.0;
        *input_matrix1.elm_mut(2, 6) = 1.0;
        *input_matrix1.elm_mut(3, 6) = 2.0;
        let mut input_matrix2 = MutSliceView::new(input_indexing.add_slice_offset(input_indexing.linear_dimension_length()), input.deref_mut());
        *input_matrix2.elm_mut(0, 0) = 0.5;
        *input_matrix2.elm_mut(1, 0) = -0.5;

        // calculate output
        let output = layer.evaluate_input_without_activation(&input);
        let output_indexing = layer.get_output_indexing();
        let output_matrix1 = SliceView::new(output_indexing, output.as_slice());
        let output_matrix2 = SliceView::new(output_indexing.add_slice_offset(output_indexing.linear_dimension_length()), output.as_slice());

        imagedatasets::print_matrix(&output_matrix1);
        imagedatasets::print_matrix(&output_matrix2);

        let mut expected_output_matrix1 = Matrix::new_with_indexing(output_indexing);
        expected_output_matrix1[(0, 2)] = 1.0;
        expected_output_matrix1[(1, 2)] = 2.0;
        assert_eq!(expected_output_matrix1, output_matrix1.copy_to_matrix());
        let mut expected_output_matrix2 = Matrix::new_with_indexing(output_indexing);
        expected_output_matrix2[(0, 0)] = 0.5;
        assert_eq!(expected_output_matrix2, output_matrix2.copy_to_matrix());
    }

    #[test]
    fn back_propagate() {
        let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
        let mut layer = PoolLayer::new_max(input_indexing, 3, MatrixDimensions{rows: 2, columns: 3});

        // set input
        let mut input: Vector<Ampl> = Vector::new(layer.get_input_dimension());
        let mut input_matrix1 = MutSliceView::new(input_indexing, input.deref_mut());
        *input_matrix1.elm_mut(1, 6) = 1.0;
        *input_matrix1.elm_mut(2, 6) = 1.0;
        *input_matrix1.elm_mut(3, 6) = 2.0;
        let mut input_matrix2 = MutSliceView::new(input_indexing.add_matrix_offset(1), input.deref_mut());
        *input_matrix2.elm_mut(0, 0) = 0.5;
        *input_matrix2.elm_mut(1, 0) = -0.5;

        // delta output
        let mut delta_output = Vector::new(layer.get_output_dimension());
        let delta_output_indexing = layer.get_output_indexing();
        let mut delta_output_matrix1 = MutSliceView::new(delta_output_indexing, delta_output.as_mut_slice());
        *delta_output_matrix1.elm_mut(0, 2) = 3.0;
        *delta_output_matrix1.elm_mut(1, 2) = 2.5;
        let mut delta_output_matrix2 = MutSliceView::new(delta_output_indexing.add_matrix_offset(1), delta_output.as_mut_slice());
        *delta_output_matrix2.elm_mut(0, 0) = 2.0;
        *delta_output_matrix2.elm_mut(1, 0) = 4.0;

        // back propagate
        const NY: f64 = 0.1;
        let gamma_input = layer.back_propagate_without_activation(&input, delta_output, NY);
        let gamma_input_matrix1 = SliceView::new(input_indexing, gamma_input.as_slice());
        imagedatasets::print_matrix(&gamma_input_matrix1);
        let gamma_input_matrix2 = SliceView::new(input_indexing.add_matrix_offset(1), gamma_input.as_slice());
        imagedatasets::print_matrix(&gamma_input_matrix2);

        // assert gamma input
        let mut expected_gamma_input_matrix1 = Matrix::new_with_indexing(input_indexing);
        *expected_gamma_input_matrix1.elm_mut(1, 6) = 3.0;
        *expected_gamma_input_matrix1.elm_mut(3, 6) = 2.5;
        assert_eq!(expected_gamma_input_matrix1, gamma_input_matrix1.copy_to_matrix());
        let mut expected_gamma_input_matrix2 = Matrix::new_with_indexing(input_indexing);
        *expected_gamma_input_matrix2.elm_mut(0, 0) = 2.0;
        *expected_gamma_input_matrix2.elm_mut(3, 2) = 4.0;
        assert_eq!(expected_gamma_input_matrix2, gamma_input_matrix2.copy_to_matrix());

    }

    #[test]
    fn evaluate_input_mean() {
        let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
        let mut layer = PoolLayer::new_mean(input_indexing, 3, MatrixDimensions{rows: 2, columns: 3});

        // set input
        let mut input: Vector<Ampl> = Vector::new(layer.get_input_dimension());
        let mut input_matrix1 = MutSliceView::new(input_indexing, input.deref_mut());
        *input_matrix1.elm_mut(1, 6) = 1.0;
        *input_matrix1.elm_mut(2, 6) = 1.0;
        *input_matrix1.elm_mut(3, 6) = 2.0;
        let mut input_matrix2 = MutSliceView::new(input_indexing.add_slice_offset(input_indexing.linear_dimension_length()), input.deref_mut());
        *input_matrix2.elm_mut(0, 0) = 0.5;
        *input_matrix2.elm_mut(1, 0) = -1.5;

        // calculate output
        let output = layer.evaluate_input_without_activation(&input);
        let output_indexing = layer.get_output_indexing();
        let output_matrix1 = SliceView::new(output_indexing, output.as_slice());
        let output_matrix2 = SliceView::new(output_indexing.add_slice_offset(output_indexing.linear_dimension_length()), output.as_slice());

        imagedatasets::print_matrix(&output_matrix1);
        imagedatasets::print_matrix(&output_matrix2);

        let mut expected_output_matrix1 = Matrix::new_with_indexing(output_indexing);
        expected_output_matrix1[(0, 2)] = 1.0 / 6.0;
        expected_output_matrix1[(1, 2)] = (1.0 + 2.0) / 6.0;
        assert_eq!(expected_output_matrix1, output_matrix1.copy_to_matrix());
        let mut expected_output_matrix2 = Matrix::new_with_indexing(output_indexing);
        expected_output_matrix2[(0, 0)] = -1.0 / 6.0;
        assert_eq!(expected_output_matrix2, output_matrix2.copy_to_matrix());
    }

    #[test]
    fn back_propagate_mean() {
        let input_indexing = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows: 12, columns: 10});
        let mut layer = PoolLayer::new_mean(input_indexing, 3, MatrixDimensions{rows: 2, columns: 3});

        // set input
        let mut input: Vector<Ampl> = Vector::new(layer.get_input_dimension());
        let mut input_matrix1 = MutSliceView::new(input_indexing, input.deref_mut());
        *input_matrix1.elm_mut(1, 6) = 1.0;
        *input_matrix1.elm_mut(2, 6) = 1.0;
        *input_matrix1.elm_mut(3, 6) = 2.0;
        let mut input_matrix2 = MutSliceView::new(input_indexing.add_matrix_offset(1), input.deref_mut());
        *input_matrix2.elm_mut(0, 0) = 0.5;
        *input_matrix2.elm_mut(1, 0) = -1.5;

        // delta output
        let mut delta_output = Vector::new(layer.get_output_dimension());
        let delta_output_indexing = layer.get_output_indexing();
        let mut delta_output_matrix1 = MutSliceView::new(delta_output_indexing, delta_output.as_mut_slice());
        *delta_output_matrix1.elm_mut(0, 2) = 3.0;
        *delta_output_matrix1.elm_mut(1, 2) = 2.5;
        let mut delta_output_matrix2 = MutSliceView::new(delta_output_indexing.add_matrix_offset(1), delta_output.as_mut_slice());
        *delta_output_matrix2.elm_mut(0, 0) = 2.0;
        *delta_output_matrix2.elm_mut(1, 0) = 4.0;

        // back propagate
        const NY: f64 = 0.1;
        let gamma_input = layer.back_propagate_without_activation(&input, delta_output, NY);
        let gamma_input_matrix1 = SliceView::new(input_indexing, gamma_input.as_slice());
        imagedatasets::print_matrix(&gamma_input_matrix1);
        let gamma_input_matrix2 = SliceView::new(input_indexing.add_matrix_offset(1), gamma_input.as_slice());
        imagedatasets::print_matrix(&gamma_input_matrix2);

        // assert gamma input
        let mut expected_gamma_input_matrix1 = Matrix::new_with_indexing(input_indexing);
        *expected_gamma_input_matrix1.elm_mut(0, 6) = 3.0 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(1, 6) = 3.0 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(0, 7) = 3.0 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(1, 7) = 3.0 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(0, 8) = 3.0 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(1, 8) = 3.0 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(2, 6) = 2.5 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(3, 6) = 2.5 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(2, 7) = 2.5 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(3, 7) = 2.5 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(2, 8) = 2.5 / 6.0;
        *expected_gamma_input_matrix1.elm_mut(3, 8) = 2.5 / 6.0;
        assert_eq!(expected_gamma_input_matrix1, gamma_input_matrix1.copy_to_matrix());
        let mut expected_gamma_input_matrix2 = Matrix::new_with_indexing(input_indexing);
        *expected_gamma_input_matrix2.elm_mut(0, 0) = 2.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(1, 0) = 2.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(0, 1) = 2.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(1, 1) = 2.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(0, 2) = 2.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(1, 2) = 2.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(2, 0) = 4.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(3, 0) = 4.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(2, 1) = 4.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(3, 1) = 4.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(2, 2) = 4.0 / 6.0;
        *expected_gamma_input_matrix2.elm_mut(3, 2) = 4.0 / 6.0;
        assert_eq!(expected_gamma_input_matrix2, gamma_input_matrix2.copy_to_matrix());

    }
}