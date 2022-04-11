use crate::neuralnetwork::{Sample, Ampl};
use crate::vector::Vector;
use core::iter;
use std::cmp::Ordering;
use std::io::{BufReader, Read};
use std::fs::File;
use std::iter::Take;
use itertools::Itertools;
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use rayon::iter::{IterBridge, ParallelIterator};
use crate::datasets::imagedatasets;
use crate::matrix::{Matrix, MatrixDimensions, MatrixLinearIndex, MatrixT, MutSliceView};
use crate::neuralnetwork::Network;

pub const IMAGE_WIDTH_HEIGHT: usize = 28;
pub const IMAGE_PIXEL_COUNT: usize = IMAGE_WIDTH_HEIGHT * IMAGE_WIDTH_HEIGHT;
pub type ImageArray = [u8; IMAGE_PIXEL_COUNT];

pub const INPUT_INDEX: MatrixLinearIndex = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows:IMAGE_WIDTH_HEIGHT, columns:IMAGE_WIDTH_HEIGHT}, IMAGE_WIDTH_HEIGHT);
pub const OUTPUT_INDEX: MatrixLinearIndex = MatrixLinearIndex::new_row_stride(MatrixDimensions{rows:IMAGE_WIDTH_HEIGHT, columns:IMAGE_WIDTH_HEIGHT}, IMAGE_WIDTH_HEIGHT);

const CROSS_PATTERN: &[(i32, i32)] = &[(0,0), (1,1), (2,2), (-1,-1), (-2,-2), (-1,1), (-2,2), (1,-1), (2,-2)];

pub fn get_learning_samples() -> impl Iterator<Item=Sample> {
    let mut rng: Pcg64 = Seeder::from(0).make_rng();
    get_samples(rng)
}

pub fn get_test_samples() -> impl Iterator<Item=Sample> {
    let mut rng: Pcg64 = Seeder::from(1).make_rng();
    get_samples(rng)
}

fn get_samples(mut rng: impl Rng) -> impl Iterator<Item=Sample> {
    iter::from_fn(move || {
        let x = rng.gen_range(0..(IMAGE_WIDTH_HEIGHT as i32));
        let y = rng.gen_range(0..(IMAGE_WIDTH_HEIGHT as i32));

        let mut input_image = Matrix::new_with_indexing(INPUT_INDEX);
        set_pattern(&mut input_image, (x, y), CROSS_PATTERN);
        let mut output_image = Matrix::new_with_indexing(OUTPUT_INDEX);
        let matrix_index = x_y_to_row_col(&output_image, (x, y)).unwrap();
        output_image[matrix_index] = 1.0;

        Some(Sample(Vector::from_vec(input_image.into_elements()), Vector::from_vec(output_image.into_elements())))
    })
}

fn set_pattern(matrix: &mut Matrix<Ampl>, center: (i32, i32), pattern: &[(i32, i32)]) {
    for delta in pattern {
        let point = (center.0 + delta.0, center.1 + delta.1);
        if let Some((row, col)) = x_y_to_row_col(matrix, point) {
            matrix[(row,col)] = 1.0;
        }
    }
}

fn x_y_to_row_col(matrix: &Matrix<Ampl>, (x, y): (i32, i32)) -> Option<(usize, usize)> {
    let row_count_i32: i32 = matrix.row_count().try_into().unwrap();
    let col_count_i32: i32 = matrix.column_count().try_into().unwrap();
    let (row, col) = (row_count_i32 - 1 - y, x);
    if 0 <= row && row < row_count_i32 && 0 <= col && col < col_count_i32 {
        Some((row.try_into().unwrap(), col.try_into().unwrap()))
    } else {
        None
    }
}

pub fn print_data_examples() {
    const SAMPLE_SIZE: usize = 10;

    let mut learning_data = get_learning_samples();
    print_samples(&mut learning_data.by_ref().take(SAMPLE_SIZE));

    let mut test_data = get_test_samples();
    print_samples(&mut test_data.by_ref().take(SAMPLE_SIZE));
}

fn print_samples(label_bytes: &mut impl Iterator<Item=Sample>) {
    imagedatasets::print_samples(label_bytes, INPUT_INDEX, OUTPUT_INDEX);
}

pub fn test_correct_percentage(network: &Network, test_samples: impl ParallelIterator<Item=Sample>, print: bool) -> f64 {
    imagedatasets::test_correct_percentage(network, test_samples, INPUT_INDEX, OUTPUT_INDEX, print)
}