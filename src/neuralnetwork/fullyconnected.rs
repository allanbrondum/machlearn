use std::any::Any;
use std::fmt::{Display, Formatter};
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use crate::datasets::imagedatasets;
use crate::matrix::{Matrix, MatrixT};
use crate::neuralnetwork::{Ampl, Layer};
use crate::vector::Vector;

#[derive(Debug, Clone)]
pub struct FullyConnectedLayer
{
    weights: Matrix<Ampl>,
}

impl FullyConnectedLayer {
    pub fn new(input_dimension: usize, output_dimension: usize) -> FullyConnectedLayer {
        FullyConnectedLayer {
            weights: Matrix::new(output_dimension, input_dimension),
        }
    }
}

impl Layer for FullyConnectedLayer {

    fn get_weights(&self) -> Vec<&Matrix<Ampl>> {
        vec!(&self.weights)
    }

    fn set_weights(&mut self, new_weights_vec: Vec<Matrix<Ampl>>) {
        assert_eq!(1, new_weights_vec.len(), "Should only have one weight matrix, has {}", new_weights_vec.len());
        let new_weights = new_weights_vec.into_iter().next().unwrap();
        if self.weights.dimensions() != new_weights.dimensions() {
            panic!("Weight dimensions for layer {} does not equals dimension of weights to set {}", self.weights.dimensions(), new_weights.dimensions());
        }
        self.weights = new_weights;
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
        self.weights.mul_vector(input)
    }

    fn back_propagate_without_activation(&mut self, input: &Vector<Ampl>, delta_output: Vector<Ampl>, ny: Ampl) -> Vector<Ampl> {
        // calculate the return value: partial derivative of error squared with respect to input state coordinates
        let gamma_input = self.weights.mul_vector_lhs(&delta_output);

        // adjust weights
        let weight_delta = ny * delta_output.to_matrix().mul_mat(&input.clone().to_matrix().transpose());
        self.weights -= weight_delta;

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
    }
}

impl Display for FullyConnectedLayer
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "weights\n")?;
        imagedatasets::print_matrix(&self.weights);

        std::fmt::Result::Ok(())
    }
}