use std::any::Any;
use std::fmt::{Display, Formatter};
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use crate::matrix::{Matrix, MatrixT};
use crate::neuralnetwork::{Ampl, Layer};
use crate::vector::Vector;

#[derive(Debug, Clone)]
pub struct FullyConnectedLayer
{
    weights: Matrix<Ampl>,
    biases: bool,
}

impl FullyConnectedLayer {
    pub fn new(input_dimension: usize, output_dimension: usize, biases: bool) -> FullyConnectedLayer {
        FullyConnectedLayer {
            weights: Matrix::new(output_dimension, input_dimension),
            biases,
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

    fn evaluate_input(&self, input: &Vector<Ampl>, sigmoid: fn(Ampl) -> Ampl) -> Vector<Ampl> {
        if input.len() != self.get_input_dimension() {
            panic!("Input state length {} not equals to weights column count {}", input.len(), self.weights.dimensions().columns);
        }
        self.weights.mul_vector(input).apply(sigmoid)
    }

    fn back_propagate(&mut self, input: &Vector<Ampl>, gamma_output: &Vector<Ampl>, sigmoid_derived: fn(Ampl) -> Ampl, ny: Ampl) -> Vector<Ampl> {
        // the delta vector is the partial derivative of error squared with respect to the layer output before the sigmoid function is applied
        let delta_output = self.weights.mul_vector(input).apply(sigmoid_derived).mul_comp(gamma_output);

        // calculate the return value: partial derivative of error squared with respect to input state coordinates
        let gamma_input = self.weights.mul_vector_lhs(&delta_output);

        // adjust weights
        self.weights -= ny * delta_output.to_matrix().mul_mat(&input.clone().to_matrix().transpose());

        gamma_input
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl FullyConnectedLayer {
    fn set_random_weights_impl<R: Rng>(&mut self, mut rng: R) {
        self.weights.apply_ref(|_| rng.gen_range(-1.0..1.0));
    }
}

impl Display for FullyConnectedLayer
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "weights\n{}", self.weights)?;

        std::fmt::Result::Ok(())
    }
}