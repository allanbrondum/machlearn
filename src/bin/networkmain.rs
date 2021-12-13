use std::borrow::Borrow;
use std::ops::IndexMut;

use machlearn::matrix::Matrix;
use machlearn::neuralnetwork::Network;
use machlearn::vector::Vector;

fn main() {
    let mut network = Network::new(vec!(8, 6, 8));

    println!("network: {}", network);

    let mut weights1 = Matrix::new(6, 8);
    weights1[0][0] = 1.;
    weights1[1][1] = 1.;
    weights1[2][2] = 1.;
    weights1[3][3] = 1.;
    weights1[4][4] = 1.;
    weights1[5][5] = 1.;
    network.set_weights(0, weights1);

    let mut weights2 = Matrix::new(8, 6);
    weights2[0][0] = 1.;
    weights2[1][1] = 1.;
    weights2[2][2] = 1.;
    weights2[3][3] = 1.;
    weights2[4][4] = 1.;
    weights2[5][5] = 1.;
    network.set_weights(1, weights2);

    println!("network: {}", network);

    let mut input = Vector::new(8);
    input[0] = 1.;
    input[1] = 1.;
    input[2] = 1.;
    input[3] = 1.;
    input[4] = 1.;
    let output = network.evaluate_input_state(input);

    println!("output: {}", output);
}
