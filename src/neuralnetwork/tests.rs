use crate::neuralnetwork::ActivationFunction;

#[test]
fn relu() {
    let act = ActivationFunction::relu();

    assert_eq!(0.0, (act.activation_function)(-2.0));
    assert_eq!(0.0, (act.activation_function)(-1.0));
    assert_eq!(0.0, (act.activation_function)(0.0));
    assert_eq!(1.0, (act.activation_function)(1.0));
    assert_eq!(2.0, (act.activation_function)(2.0));

    assert_eq!(0.0, (act.activation_function_derived)(-1.0));
    assert_eq!(1.0, (act.activation_function_derived)(0.1));
    assert_eq!(1.0, (act.activation_function_derived)(1.0));
}

#[test]
fn relu01() {
    let act = ActivationFunction::relu01();

    assert_eq!(0.0, (act.activation_function)(-2.0));
    assert_eq!(0.0, (act.activation_function)(-1.0));
    assert_eq!(0.0, (act.activation_function)(0.0));
    assert_eq!(1.0, (act.activation_function)(1.0));
    assert_eq!(1.0, (act.activation_function)(2.0));

    assert_eq!(0.0, (act.activation_function_derived)(-1.0));
    assert_eq!(1.0, (act.activation_function_derived)(0.1));
    assert_eq!(1.0, (act.activation_function_derived)(0.9));
    assert_eq!(0.0, (act.activation_function_derived)(1.1));
}

#[test]
fn sigmoid_logistic() {
    let act = ActivationFunction::sigmoid();

    assert!((act.activation_function)(-1.0) > 0.0);
    assert!((act.activation_function)(-1.0) < 0.5);
    assert_eq!(0.5, (act.activation_function)(0.0));
    assert!((act.activation_function)(1.0) < 1.0);
    assert!((act.activation_function)(1.0) > 0.5);
    assert_eq!(((act.activation_function)(-1.0) + (act.activation_function)(1.0)) as f32, 1.0);

    assert!((act.activation_function_derived)(-1.0) > 0.0);
    assert!((act.activation_function_derived)(-1.0) < 0.25);
    assert_eq!(0.25, (act.activation_function_derived)(0.0));
    assert!((act.activation_function_derived)(1.0) < 0.25);
    assert!((act.activation_function_derived)(1.0) > 0.0);
    assert_eq!((act.activation_function_derived)(-1.0) as f32, (act.activation_function_derived)(1.0) as f32);
}