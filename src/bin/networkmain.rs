use std::any::Any;
use itertools::Itertools;
use machlearn::matrix::{Matrix, MatrixDimensions, MatrixT, TransposedMatrixView};
use machlearn::neuralnetwork::{FullyConnectedLayer, Layer, Network};


fn main() {

    let mut network = Network::new_fully_connected(vec!(5, 10));

    let layer_a = FullyConnectedLayer::new(1, 2);
    let layer_a_ref = &layer_a;
    let layer_a_any = layer_a_ref as &dyn Any;
    let fullcon_layer_a: &FullyConnectedLayer = layer_a_any.downcast_ref().unwrap();
    println!("{}", fullcon_layer_a.get_input_dimension());


    let layer_ref = network.layers().next().unwrap().as_ref();
    let layer_any = layer_ref.as_any();
    let fullcon_layer: &FullyConnectedLayer = layer_any.downcast_ref().unwrap();
    println!("{}", fullcon_layer.get_input_dimension());

}
