use std::borrow::Borrow;
use std::ops::IndexMut;

use machlearn::matrix::Matrix;
use machlearn::neuralnetwork::Network;
use machlearn::vector::Vector;

fn main() {
    let mut network = Network::new(vec!(8, 6, 8));

    let weights1 = network.weights(0);

    weights1[0][0] = 1.;
    weights1[1][1] = 1.;
    weights1[2][2] = 1.;
    weights1[3][3] = 1.;
    weights1[4][4] = 1.;
    weights1[5][5] = 1.;

    let weights2 = network.weights(1);

    weights2[0][0] = 1.;
    weights2[1][1] = 1.;
    weights2[2][2] = 1.;
    weights2[3][3] = 1.;
    weights2[4][4] = 1.;
    weights2[5][5] = 1.;

    let input = Vector::new(5);
    network.evaluate_input_state( input);
}
