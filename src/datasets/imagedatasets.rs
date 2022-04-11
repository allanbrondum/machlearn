use std::cmp::Ordering;
use itertools::Itertools;
use crate::matrix::{MatrixLinearIndex, MatrixT, SliceView};
use crate::neuralnetwork::{Ampl, Sample};

fn print_sample(sample: &Sample, input_matrix_index: MatrixLinearIndex, output_matrix_index: MatrixLinearIndex) {
    println!("Input:\n");
    print_matrix(&SliceView::new(input_matrix_index, sample.0.as_slice()));
    println!("Output:\n");
    print_matrix(&SliceView::new(output_matrix_index, sample.1.as_slice()));
}

pub fn print_matrix<'a, M: MatrixT<'a, Ampl>>(matrix: &'a M) {
    let min = matrix.iter().copied().min_by(cmp_ampl_ref).unwrap();
    let max = matrix.iter().copied().max_by(cmp_ampl_ref).unwrap();
    for row in 0..matrix.row_count() {
        println!("{}", matrix.row_iter(row)
            .map(|val| match (256. * (val - min) / (max - min)) as u8 {
                0..=50 => ' ',
                51..=150 => '.',
                151..=200 => '+',
                201..=255 => '*',
                _ => panic!("Unhandled value {}", val) })
            .format(""));

    }
}

fn cmp_ampl(x: Ampl, y: Ampl) -> Ordering {
    if x > y { Ordering::Greater } else { Ordering::Less }
}

fn cmp_ampl_ref(x: &Ampl, y: &Ampl) -> Ordering {
    cmp_ampl(*x, *y)
}
