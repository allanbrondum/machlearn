use std::{fs, io, iter};
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Bytes, Read, Write};
use std::iter::{FromFn, Take};
use std::ops::Deref;
use std::rc::Rc;
use std::time::Instant;

use itertools::{Chunk, Itertools};
use rand::Rng;
use rayon::iter::{ParallelBridge, ParallelIterator};

use machlearn::neuralnetwork::{Ampl, Network, Sample};
use machlearn::neuralnetwork;
use machlearn::vector::Vector;

const IMAGE_WIDTH_HEIGHT: usize = 28;
const IMAGE_PIXEL_COUNT: usize = IMAGE_WIDTH_HEIGHT * IMAGE_WIDTH_HEIGHT;
type ImageArray = [u8; IMAGE_PIXEL_COUNT];

// http://yann.lecun.com/exdb/mnist/
// https://www.kaggle.com/sylvia23/mnist-data-for-digit-recognation

fn main() {
    // let mut network = Network::new_logistic_sigmoid(vec!(IMAGE_PIXEL_COUNT, 10)); // single layer
    let mut network = Network::new_logistic_sigmoid(vec!(IMAGE_PIXEL_COUNT, IMAGE_PIXEL_COUNT, 10));
    network.set_random_weights();

    print_data_examples();

    // print_samples(&mut get_learning_samples().take(1000));

    let start = Instant::now();
    println!("learning");

    const LEARNING_SAMPLES: usize = 60_000;
    neuralnetwork::run_learning_iterations(&mut network, get_learning_samples().take(LEARNING_SAMPLES), 0.3);

    let duration = start.elapsed();
    println!("duration {:.2?}", duration);

    let start = Instant::now();
    println!("testing");

    const TEST_SAMPLES: usize = 1000;
    // let errsqr = neuralnetwork::run_test_iterations(&network, test_samples);
    let errsqr = neuralnetwork::run_test_iterations_parallel(&network, get_test_samples().take(TEST_SAMPLES).par_bridge());

    test_correct_percentage(&network, get_test_samples().take(20).par_bridge(), true);
    let pct_correct = test_correct_percentage(&network, get_test_samples().take(TEST_SAMPLES).par_bridge(), false);

    let duration = start.elapsed();
    println!("duration {:.2?}", duration);

    println!("error squared: {:.5}", errsqr);
    println!("% correct: {:.2}", pct_correct);

    // println!("network: {}", network);

    write_network_to_file(&network);
}

fn write_network_to_file(network: &Network) {
    let json = serde_json::to_string(&network.copy_all_weights()).expect("error serializing");
    let mut file = fs::File::create("networkweights.json").expect("error creating file");
    file.write_all(json.as_bytes()).expect("error writing");
    file.flush().unwrap();
}

pub fn test_correct_percentage(network: &Network, samples: impl ParallelIterator<Item=Sample>, print: bool) -> f64 {
    let result: (usize, usize) = samples.map(|sample| {
        let output = network.evaluate_input_no_state_change(&sample.0);
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

fn get_learning_samples() -> impl Iterator<Item=Sample> {
    transform_to_samples(get_learning_data())
}

fn get_test_samples() -> impl Iterator<Item=Sample> {
    transform_to_samples(get_test_data())
}

fn transform_to_samples(data_sets: impl Iterator<Item=(u8, ImageArray)>) -> impl Iterator<Item=Sample> {
    data_sets.map(|data_set| {
        let input: Vec<Ampl> = data_set.1.iter().map(|val| *val as Ampl / 256.).collect(); // map to float value 0..1
        let output: Vec<Ampl> = (0..10).map(|digit| if digit == data_set.0 {1.} else {0.}).collect();
        Sample(Vector::from_vec(input), Vector::from_vec(output))
    })
}

fn print_data_examples() {
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
        for line in data_set.1.chunks(IMAGE_WIDTH_HEIGHT) {
            println!("{}", line.iter()
                .map(|val| match val {
                    0..=50 => ' ',
                    51..=150 => '.',
                    151..=200 => '+',
                    201..=255 => '*',
                    _ => panic!("Unhandled value {}", val) })
                .format(""));
        }
    }
}

fn print_samples(samples: &mut impl Iterator<Item=Sample>) {
    for sample in samples {
        println!("Output: {}", sample.1);
        println!("Image:"); // print as ascii art
        // images are u8 grayscale
        for line in sample.0.chunks(IMAGE_WIDTH_HEIGHT) {
            println!("{}", line.iter()
                .map(|val | match val * 256. {
                    0.0..=50. => ' ',
                    51.0..=150. => '.',
                    151.0..=200. => '+',
                    201.0..=255. => '*',
                    _ => panic!("Unhandled value {}", val) })
                .format(""));
        }
    }
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

