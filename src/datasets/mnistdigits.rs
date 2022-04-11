use crate::neuralnetwork::{Sample, Ampl};
use crate::vector::Vector;
use core::iter;
use std::cmp::Ordering;
use std::io::{BufReader, Read};
use std::fs::File;
use std::iter::Take;
use itertools::Itertools;
use rayon::iter::{IterBridge, ParallelIterator};
use crate::datasets::imagedatasets;
use crate::matrix::{MatrixDimensions, MatrixLinearIndex, MatrixT, MutSliceView};
use crate::neuralnetwork::Network;

pub const IMAGE_WIDTH_HEIGHT: usize = 28;
pub const IMAGE_PIXEL_COUNT: usize = IMAGE_WIDTH_HEIGHT * IMAGE_WIDTH_HEIGHT;
pub type ImageArray = [u8; IMAGE_PIXEL_COUNT];

pub const INPUT_INDEX: MatrixLinearIndex = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows:IMAGE_WIDTH_HEIGHT, columns:IMAGE_WIDTH_HEIGHT}, IMAGE_WIDTH_HEIGHT);
pub const OUTPUT_INDEX: MatrixLinearIndex = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows:1, columns:10}, 10);

/// 60_000 learning samples
pub fn get_learning_samples() -> impl Iterator<Item=Sample> {
    transform_to_samples(get_learning_data())
}

/// 10_000 learning samples
pub fn get_test_samples() -> impl Iterator<Item=Sample> {
    transform_to_samples(get_test_data())
}

fn transform_to_samples(data_sets: impl Iterator<Item=(u8, ImageArray)>) -> impl Iterator<Item=Sample> {
    data_sets.map(|data_set| {
        let input: Vec<Ampl> = data_set.1.iter().map(|val| *val as Ampl / 256.).collect(); // map to float value 0..1
        let output: Vec<Ampl> = (0..10).map(|digit| if digit == data_set.0 {1.} else {0.}).collect();
        Sample(Vector::from_vec(input), Vector::from_vec(output))
    })
}

pub fn print_data_examples() {
    const SAMPLE_SIZE: usize = 10;

    let mut learning_data = get_learning_samples();
    print_samples(&mut learning_data.by_ref().take(SAMPLE_SIZE));
    println!("Count learning data: {}", learning_data.count() + SAMPLE_SIZE);

    let mut test_data = get_test_samples();
    print_samples(&mut test_data.by_ref().take(SAMPLE_SIZE));
    println!("Count test data: {}", test_data.count() + SAMPLE_SIZE);
}

fn print_samples(label_bytes: &mut impl Iterator<Item=Sample>) {
    imagedatasets::print_samples(label_bytes, INPUT_INDEX, OUTPUT_INDEX);
}

fn get_learning_data() -> impl Iterator<Item=(u8, ImageArray)> {
    get_data_sets("mnistdigitdata/train-labels.idx1-ubyte", "mnistdigitdata/train-images.idx3-ubyte")
}

fn get_test_data() -> impl Iterator<Item=(u8, ImageArray)> {
    get_data_sets("mnistdigitdata/t10k-labels.idx1-ubyte", "mnistdigitdata/t10k-images.idx3-ubyte")
}

fn get_data_sets(label_file_path: &str, image_file_path: &str) -> impl Iterator<Item=(u8, ImageArray)> {
    let mut labels_read = BufReader::new(File::open(label_file_path).unwrap());
    labels_read.by_ref().bytes().take(2 * 4).count(); // skip header
    let mut label_bytes = labels_read.bytes();

    let mut images_read = BufReader::new(File::open(image_file_path).unwrap());
    images_read.by_ref().bytes().take(4 * 4).count(); // skip header

    iter::from_fn(move || {
        if let Some(label_result) = label_bytes.next() {
            let mut image = [0u8; IMAGE_PIXEL_COUNT];
            images_read.read_exact(&mut image).unwrap();

            Some((label_result.unwrap(), image))
        } else {
            None
        }
    })
}


pub fn test_correct_percentage(network: &Network, samples: impl ParallelIterator<Item=Sample>, print: bool) -> f64 {
    imagedatasets::test_correct_percentage(network, samples, INPUT_INDEX, OUTPUT_INDEX, print)
}