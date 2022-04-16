use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use itertools::Itertools;
use rayon::iter::{ParallelBridge, ParallelIterator};
use crate::matrix::{MatrixIndex, MatrixLinearIndex, MatrixT, SliceView};
use crate::neuralnetwork::{Ampl, cmp_ampl, cmp_ampl_ref, Network, Sample};

pub fn print_sample(sample: &Sample, input_matrix_index: MatrixLinearIndex, output_matrix_indexes: &Vec<MatrixLinearIndex>) {
    println!("Input:");
    print_matrix(&get_sample_input_matrix(&sample, input_matrix_index));
    println!("Output:");
    for output_matrix_index in output_matrix_indexes {
        let output_matrix = &get_sample_output_matrix(&sample, *output_matrix_index);
        print_matrix_max(output_matrix);
        print_matrix(output_matrix);
    }
}

fn get_sample_output_matrix(sample: &Sample, output_matrix_index: MatrixLinearIndex) -> SliceView<Ampl> {
    SliceView::new(output_matrix_index, sample.1.as_slice())
}

fn get_sample_input_matrix(sample: &Sample, input_matrix_index: MatrixLinearIndex) -> SliceView<Ampl> {
    SliceView::new(input_matrix_index, sample.0.as_slice())
}

pub fn print_samples(samples: &mut impl Iterator<Item=Sample>, input_matrix_index: MatrixLinearIndex, output_matrix_indexes: &Vec<MatrixLinearIndex>) {
    for data_set in samples {
        print_sample(&data_set, input_matrix_index, output_matrix_indexes);
    }
}

pub fn print_matrix_max<'a, M: MatrixT<'a, Ampl>>(matrix: &'a M) {
    let index = index_of_max(matrix);
    println!("Max entry ({},{})", index.0, index.1);
}

pub fn print_matrix<'a, M: MatrixT<'a, Ampl>>(matrix: &'a M) {
    let min = matrix.iter().copied().min_by(cmp_ampl_ref).unwrap().min(0.0);
    let max = matrix.iter().copied().max_by(cmp_ampl_ref).unwrap();
    let line: String = std::iter::from_fn(|| Some('-')).take(matrix.column_count() + 2).collect();
    println!("{} min: {} max: {}", line, min, max);
    for row in 0..matrix.row_count() {
        println!("|{}|", matrix.row_iter(row)
            .map(|val| match (256. * (val - min) / (max - min)) as u8 {
                0..=50 => ' ',
                51..=150 => '.',
                151..=200 => '+',
                201..=255 => '*',
                _ => panic!("Unhandled value {}", val) })
            .format(""));
    }
    println!("{}", line);
}

fn index_of_max<'a, M: MatrixT<'a, Ampl>>(matrix: &'a M) -> MatrixIndex {
    matrix.iter_enum().max_by(|x, y| cmp_ampl(*x.1, *y.1)).unwrap().0
}

fn test_correct_percentage_parallel(network: &Network, test_samples: impl ParallelIterator<Item=Sample>, output_matrix_indexes: &Vec<MatrixLinearIndex>) -> f64 {
    let result: (usize, usize) = test_samples.map(|sample| {
        let is_correct = test_sample(network, &sample, output_matrix_indexes, false);
        (1, if is_correct { 1 } else { 0 })
    }).reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1 + y.1));
    100. * result.1 as f64 / result.0 as f64
}

fn test_correct_percentage_sequential(network: &Network, test_samples: impl Iterator<Item=Sample>, output_matrix_indexes: &Vec<MatrixLinearIndex>, print: bool) -> f64 {
    let result: (usize, usize) = test_samples.map(|sample| {
        let is_correct = test_sample(network, &sample, output_matrix_indexes, print);
        (1, if is_correct { 1 } else { 0 })
    }).reduce(|x, y| (x.0 + y.0, x.1 + y.1)).unwrap();
    100. * result.1 as f64 / result.0 as f64
}

pub fn test_correct_percentage(network: &Network, test_samples: impl Iterator<Item=Sample> + Send, output_matrix_indexes: &Vec<MatrixLinearIndex>, print: bool) -> f64 {
    if print {
        test_correct_percentage_sequential(network, test_samples, output_matrix_indexes, print)
    } else {
        test_correct_percentage_parallel(network, test_samples.par_bridge(), output_matrix_indexes)
    }
}

fn test_sample(network: &Network, sample: &Sample, output_matrix_indexes: &Vec<MatrixLinearIndex>, print: bool) -> bool {
    let output = network.evaluate_input(&sample.0);
    let mut guess_indexes = vec!();
    let mut correct_indexes = vec!();
    for (index, &output_indexing) in output_matrix_indexes.iter().enumerate() {
        let output_matrix = SliceView::new(output_indexing, output.as_slice());
        let correct_matrix = get_sample_output_matrix(&sample, output_indexing);

        if print {
            println!("Calculated output {}", index);
            print_matrix(&output_matrix);
            println!("Correct output {}", index);
            print_matrix(&correct_matrix);
        }

        let guess_index = index_of_max(&output_matrix);
        guess_indexes.push(guess_index);
        let correct_index = index_of_max(&correct_matrix);
        correct_indexes.push(correct_index);
    }

    if print {
        println!("Guess {}, correct {}", guess_indexes.iter().format(","), correct_indexes.iter().format(","));
    }
    guess_indexes == correct_indexes
}
