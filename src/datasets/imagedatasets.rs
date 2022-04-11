use std::cmp::Ordering;
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use crate::matrix::{MatrixIndex, MatrixLinearIndex, MatrixT, SliceView};
use crate::neuralnetwork::{Ampl, Network, Sample};

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
    let min = matrix.iter().copied().min_by(cmp_ampl_ref).unwrap();
    let max = matrix.iter().copied().max_by(cmp_ampl_ref).unwrap();
    let line: String = std::iter::from_fn(|| Some('-')).take(matrix.column_count() + 2).collect();
    println!("{}", line);
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

fn cmp_ampl(x: Ampl, y: Ampl) -> Ordering {
    if x > y { Ordering::Greater } else { Ordering::Less }
}

fn cmp_ampl_ref(x: &Ampl, y: &Ampl) -> Ordering {
    cmp_ampl(*x, *y)
}

fn index_of_max<'a, M: MatrixT<'a, Ampl>>(matrix: &'a M) -> MatrixIndex {
    matrix.iter_enum().max_by(|x, y| cmp_ampl(*x.1, *y.1)).unwrap().0
}

pub fn test_correct_percentage(network: &Network, test_samples: impl ParallelIterator<Item=Sample>, output_matrix_index: MatrixLinearIndex, print: bool) -> f64 {
    let result: (usize, usize) = test_samples.map(|sample| {
        let output = network.evaluate_input(&sample.0);
        let output_matrix = SliceView::new(output_matrix_index, output.as_slice());
        let guess_index = index_of_max(&output_matrix);
        let correct_matrix = get_sample_output_matrix(&sample, output_matrix_index);
        let correct_index = index_of_max(&correct_matrix);
        if print {
            println!("Output {}, guess {}, correct {}", output, guess_index, correct_index);
        }
        let is_correct = if guess_index == correct_index {1} else {0};
        (1, is_correct)
    }).reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1 + y.1));
    100. * result.1 as f64 / result.0 as f64
}
