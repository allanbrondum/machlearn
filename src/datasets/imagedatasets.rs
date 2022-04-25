use std::fmt::Display;
use std::io;

use itertools::Itertools;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::matrix::{Matrix, MatrixDimensions, MatrixIndex, MatrixLinearIndex, MatrixT, SliceView};
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

pub fn print_samples(samples: impl Iterator<Item=Sample>, input_matrix_index: MatrixLinearIndex, output_matrix_indexes: &Vec<MatrixLinearIndex>) {
    for data_set in samples {
        print_sample(&data_set, input_matrix_index, output_matrix_indexes);
    }
}

pub fn print_matrix_max<'a, M: MatrixT<'a, Ampl>>(matrix: &'a M) {
    let index = index_of_max(matrix);
    println!("Max entry ({},{})", index.0, index.1);
}

pub fn print_matrix<'a, M: MatrixT<'a, Ampl>>(matrix: &'a M) {
    print_matrix_write(&mut std::io::stdout(), matrix);
}

pub fn print_matrix_write<'a, M: MatrixT<'a, Ampl>>(write: &mut impl io::Write, matrix: &'a M) {
    let min = matrix.iter().copied().min_by(cmp_ampl_ref).unwrap().min(0.0);
    let max = matrix.iter().copied().max_by(cmp_ampl_ref).unwrap();
    let line: String = std::iter::from_fn(|| Some('-')).take(matrix.column_count() + 2).collect();
    write!(write, "{} min: {:.5} max: {:.5}\n", line, min, max);
    for row in 0..matrix.row_count() {
        write!(write, "|{}|\n", matrix.row_iter(row)
            .map(|val| match (256. * (val - min) / (max - min)) as u8 {
                0..=50 => ' ',
                51..=150 => '.',
                151..=200 => '+',
                201..=255 => '*',
                _ => panic!("Unhandled value {}", val) })
            .format(""));
    }
    write!(write, "{}\n", line);
}

fn index_of_max<'a, M: MatrixT<'a, Ampl>>(matrix: &'a M) -> MatrixIndex {
    matrix.iter_enum().max_by(|x, y| cmp_ampl(*x.1, *y.1)).unwrap().0
}

fn test_correct_percentage_parallel(network: &Network, test_samples: impl ParallelIterator<Item=Sample>, input_matrix_index: MatrixLinearIndex, output_matrix_indexes: &Vec<MatrixLinearIndex>) -> f64 {
    let result: (usize, usize) = test_samples.map(|sample| {
        let is_correct = test_sample(network, &sample, input_matrix_index, output_matrix_indexes, false);
        (1, if is_correct { 1 } else { 0 })
    }).reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1 + y.1));
    100. * result.1 as f64 / result.0 as f64
}

fn test_correct_percentage_sequential(network: &Network, test_samples: impl Iterator<Item=Sample>, intput_matrix_index: MatrixLinearIndex, output_matrix_indexes: &Vec<MatrixLinearIndex>, print: bool) -> f64 {
    let result: (usize, usize) = test_samples.map(|sample| {
        let is_correct = test_sample(network, &sample, intput_matrix_index, output_matrix_indexes, print);
        (1, if is_correct { 1 } else { 0 })
    }).reduce(|x, y| (x.0 + y.0, x.1 + y.1)).unwrap();
    100. * result.1 as f64 / result.0 as f64
}

pub fn test_correct_percentage(network: &Network, test_samples: impl Iterator<Item=Sample> + Send, intput_matrix_index: MatrixLinearIndex, output_matrix_indexes: &Vec<MatrixLinearIndex>, print: bool) -> f64 {
    if print {
        test_correct_percentage_sequential(network, test_samples, intput_matrix_index, output_matrix_indexes, print)
    } else {
        test_correct_percentage_parallel(network, test_samples.par_bridge(), intput_matrix_index, output_matrix_indexes)
    }
}

fn test_sample(network: &Network, sample: &Sample, input_matrix_index: MatrixLinearIndex, output_matrix_indexes: &Vec<MatrixLinearIndex>, print: bool) -> bool {
    let output = network.evaluate_input(&sample.0);
    if print {
        println!("Input");
        print_matrix(&SliceView::new(input_matrix_index, sample.0.as_slice()));
    }
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

pub fn create_kernel_patterns(kernel_dimension: MatrixDimensions, kernel_count: usize) -> Vec<Matrix<Ampl>> {
    let mut vec = Vec::new();
    let mut rng: Pcg64 = Seeder::from(0).make_rng();
    for j in 0..kernel_count {
        let mut m = Matrix::new_with_dimension(kernel_dimension);

        // let range = 1.0 / 10.0 as Ampl;
        // m.apply_ref(|_| rng.gen_range(0.0..range));

        set_kernel(&mut m, j);
        vec.push(m);
    }
    vec
}

fn set_kernel(kernel: &mut Matrix<Ampl>, kernel_number: usize) {
    const VALUE: Ampl = 1.0;
    match kernel_number {
        0 => {
            let row_middle = kernel.row_count() / 2;
            for i in 0..kernel.column_count() {
                kernel[(row_middle, i)] = VALUE;
            }
        },
        1 => {
            let col_middle = kernel.column_count() / 2;
            for i in 0..kernel.row_count() {
                kernel[(i, col_middle)] = VALUE;
            }
        },
        2 => {
            for i in 0..kernel.column_count().min(kernel.row_count()) {
                kernel[(i, i)] = VALUE;
            }
        },
        3 => {
            let row_count = kernel.row_count();
            for i in 0..kernel.column_count().min(kernel.row_count()) {
                kernel[(row_count - i - 1, i)] = VALUE;
            }
        },
        _ => panic!("Unhandled kernel number {}", kernel_number),
    }
}