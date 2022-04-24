
use std::fmt;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeStruct;
use crate::neuralnetwork::Ampl;

#[derive(Debug, Copy, Clone)]
pub struct ActivationFunction {
    pub activation_function: fn(Ampl) -> Ampl,
    pub activation_function_derived: fn(Ampl) -> Ampl,
    activation_function_enum: ActivationFunctionEnum,
}

impl Serialize for ActivationFunction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        let mut s = serializer.serialize_struct("ActivationFunction", 1)?;
        s.serialize_field("activation_function_enum", &self.activation_function_enum)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for ActivationFunction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field { ActivationFunctionEnum }

        struct ActivationFunctionVisitor;

        impl<'de> Visitor<'de> for ActivationFunctionVisitor {
            type Value = ActivationFunction;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Duration")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ActivationFunction, V::Error>
                where
                    V: MapAccess<'de>,
            {
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::ActivationFunctionEnum => {
                            let enum_value: ActivationFunctionEnum = map.next_value()?;
                            match enum_value {
                                ActivationFunctionEnum::Relu01 =>
                                    return Ok(ActivationFunction::relu01()),
                                ActivationFunctionEnum::Relu =>
                                    return Ok(ActivationFunction::relu()),
                                ActivationFunctionEnum::Sigmoid =>
                                    return Ok(ActivationFunction::sigmoid()),
                                ActivationFunctionEnum::Identity =>
                                    return Ok(ActivationFunction::identity()),
                            };
                        },
                    }
                }
                Err(de::Error::missing_field("activation_function_enum"))
            }


        }

        const FIELDS: &[&str] = &["activation_function_enum"];
        deserializer.deserialize_struct("ActivationFunction", FIELDS, ActivationFunctionVisitor)
    }
}

impl ActivationFunction {
    fn apply(&self, input: Ampl) -> Ampl {
        (self.activation_function)(input)
    }

    fn apply_derived(&self, input: Ampl) -> Ampl {
        (self.activation_function_derived)(input)
    }

}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFunctionEnum {
    Sigmoid,
    Relu,
    Relu01,
    Identity,
}

impl ActivationFunction {
    pub fn sigmoid() -> Self {
        ActivationFunction {
            activation_function: sigmoid_logistic,
            activation_function_derived: sigmoid_logistic_derived,
            activation_function_enum: ActivationFunctionEnum::Sigmoid,
        }
    }

    pub fn relu() -> Self {
        ActivationFunction {
            activation_function: relu,
            activation_function_derived: relu_derived,
            activation_function_enum: ActivationFunctionEnum::Relu,
        }
    }

    pub fn relu01() -> Self {
        ActivationFunction {
            activation_function: relu01,
            activation_function_derived: relu01_derived,
            activation_function_enum: ActivationFunctionEnum::Relu01,
        }
    }

    pub fn identity() -> Self {
        ActivationFunction {
            activation_function: identity,
            activation_function_derived: identity_derived,
            activation_function_enum: ActivationFunctionEnum::Identity,
        }
    }
}


fn sigmoid_logistic(input: Ampl) -> Ampl {
    sigmoid_logistic_raw(input)
}

fn sigmoid_logistic_raw(input: Ampl) -> f64 {
    1. / (1. + (-input).exp())
}

fn sigmoid_logistic_derived(input: Ampl) -> Ampl {
    sigmoid_logistic_derived_raw(input)
}

fn sigmoid_logistic_derived_raw(input: Ampl) -> f64 {
    (-input).exp() / (1. + (-input).exp()).powf(2.)
}

fn relu(input: Ampl) -> Ampl {
    if input >= 0.0 {input} else {0.0}
}

fn relu_derived(input: Ampl) -> Ampl {
    if input >= 0.0 {1.0} else {0.0}
}

fn relu01(input: Ampl) -> Ampl {
    if input >= 1.0 {1.0} else if input >= 0.0 {input} else {0.0}
}

fn relu01_derived(input: Ampl) -> Ampl {
    if input >= 1.0 {0.0} else if input >= 0.0 {1.0} else {0.0}
}

fn identity(input: Ampl) -> Ampl {
    input
}

fn identity_derived(_input: Ampl) -> Ampl {
    1.0
}
