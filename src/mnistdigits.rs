use crate::neuralnetwork::{Sample, Ampl};
use crate::vector::Vector;
use core::iter;
use std::cmp::Ordering;
use std::io::{BufReader, Read};
use std::fs::File;
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use crate::neuralnetwork2::Network;

pub const IMAGE_WIDTH_HEIGHT: usize = 28;
pub const IMAGE_PIXEL_COUNT: usize = IMAGE_WIDTH_HEIGHT * IMAGE_WIDTH_HEIGHT;
pub type ImageArray = [u8; IMAGE_PIXEL_COUNT];

pub fn get_learning_samples() -> impl Iterator<Item=Sample> {
    transform_to_samples(get_learning_data())
}

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
    let mut learning_data = get_learning_data();

    const SAMPLE_SIZE: usize = 10;

    print_data_series(&mut learning_data.by_ref().take(SAMPLE_SIZE));

    println!("Count learning data: {}", learning_data.count() + SAMPLE_SIZE);

    let mut test_data = get_test_data();
    print_data_series(&mut test_data.by_ref().take(SAMPLE_SIZE));

    println!("Count test data: {}", test_data.count() + SAMPLE_SIZE);
}

fn print_data_series(label_bytes: &mut impl Iterator<Item=(u8, ImageArray)>) {
    for data_set in label_bytes {
        println!("Label: {0:} {0:08b}", data_set.0);
        println!("Image:"); // print as ascii art
        // images are u8 grayscale
        print_image(data_set.1.iter().copied());
    }
}

fn print_image(image_bytes: impl Iterator<Item=u8>) {
    for line in &image_bytes.chunks(IMAGE_WIDTH_HEIGHT) {
        println!("{}", line
            .map(|val| match val {
                0..=50 => ' ',
                51..=150 => '.',
                151..=200 => '+',
                201..=255 => '*',
                _ => panic!("Unhandled value {}", val) })
            .format(""));
    }
}

// fn print_samples(samples: &mut impl Iterator<Item=Sample>) {
//     for sample in samples {
//         println!("Output: {}", sample.1);
//         println!("Image:"); // print as ascii art
//         // samples are 0..1 grayscale
//         print_image(sample.0.iter().map(|val| (val * 256.) as u8));
//     }
// }

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
    let result: (usize, usize) = samples.map(|sample| {
        let output = network.evaluate_input(&sample.0);
        let guess = index_of_max(&output);
        let correct = index_of_max(&sample.1);
        if print {
            println!("Output {}, guess {}, correct {}", output, guess, correct);
        }
        let is_correct = if guess == correct {1} else {0};
        (1, is_correct)
    }).reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1 + y.1));
    100. * result.1 as f64 / result.0 as f64

}

fn index_of_max(vector: &Vector<Ampl>) -> usize {
    vector.iter().enumerate().max_by(|x, y| if x.1 > y.1 { Ordering::Greater } else { Ordering::Less }).unwrap().0
}


