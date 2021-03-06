use std::any::Any;
use std::fmt::{Display, Formatter};
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use crate::datasets::imagedatasets;
use crate::matrix::{Matrix, MatrixT};
use crate::neuralnetwork::{Ampl, Layer};
use crate::vector::Vector;
use serde::{Deserialize, Serialize};

/// Layer where all input and output states are connected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullyConnectedLayer
{
    weights: Matrix<Ampl>,
    bias: Vector<Ampl>,
}

impl FullyConnectedLayer {
    pub fn new(input_dimension: usize, output_dimension: usize) -> FullyConnectedLayer {
        FullyConnectedLayer {
            weights: Matrix::new(output_dimension, input_dimension),
            bias: Vector::new(output_dimension),
        }
    }
}

#[typetag::serde]
impl Layer for FullyConnectedLayer {

    fn get_weights(&self) -> Vec<&Matrix<Ampl>> {
        vec!(&self.weights)
    }

    fn get_input_dimension(&self) -> usize {
        self.weights.dimensions().columns
    }

    fn get_output_dimension(&self) -> usize {
        self.weights.dimensions().rows
    }

    fn set_random_weights(&mut self) {
        let mut rng = rand::thread_rng();
        self.set_random_weights_impl(rng);
    }

    fn set_random_weights_seed(&mut self, seed: u64) {
        let mut rng: Pcg64 = Seeder::from(0).make_rng();
        self.set_random_weights_impl(rng);
    }

    fn evaluate_input_without_activation(&self, input: &Vector<Ampl>) -> Vector<Ampl> {
        self.weights.mul_vector(input).add_vector(&self.bias)
    }

    fn back_propagate_without_activation(&mut self, input: &Vector<Ampl>, delta_output: Vector<Ampl>, ny: Ampl) -> Vector<Ampl> {
        // calculate the return value: partial derivative of error squared with respect to input state coordinates
        let gamma_input = self.weights.mul_vector_lhs(&delta_output);

        // adjust weights
        let delta_output_scaled = ny * delta_output;
        let weight_delta = delta_output_scaled.as_matrix().mul_mat(&input.as_matrix().as_transpose());
        self.weights -= weight_delta;
        self.bias -= delta_output_scaled;

        gamma_input
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl FullyConnectedLayer {

    fn set_random_weights_impl<R: Rng>(&mut self, mut rng: R) {
        let range = 1.0 / (self.weights.column_count()) as Ampl;
        self.weights.apply_ref(|_| rng.gen_range(0.0..range));
        self.bias.apply_ref(|_| rng.gen_range(0.0..range));
    }

    pub fn set_weights(&mut self, new_weights: Matrix<Ampl>) {
        if self.weights.dimensions() != new_weights.dimensions() {
            panic!("Weight dimensions for layer {} does not equals dimension of weights to set {}", self.weights.dimensions(), new_weights.dimensions());
        }
        self.weights = new_weights;
    }
}

impl Display for FullyConnectedLayer
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "weights\n")?;
        imagedatasets::print_matrix(&self.weights);
        write!(f, "bias\n")?;
        let bias_matrix = self.bias.as_matrix();
        imagedatasets::print_matrix(&bias_matrix);
        // print_matrix3(&bias_matrix);  // TODO

        std::fmt::Result::Ok(())
    }
}

// fn print_matrix2<'a, 'b, M: MatrixT<'a, Ampl>>(matrix: &'b M) where 'b: 'a {
//
// }
//
// fn print_matrix3<'a, M: MatrixT<'a, Ampl>>(matrix: &'a M)  {
//
// }