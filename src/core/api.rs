use std::{any::Any, fmt::Debug};

use super::{AnyValue, Result, TrainingRow};

pub trait Learner<T> {
    fn fit(&self, training_data: &[TrainingRow<T>]) -> Result<Box<dyn Model<T>>>
    where
        Self: Sized;
}

pub trait Model<T>: Debug {
    fn transform(&self, inputs: &[Vec<AnyValue>]) -> Result<Box<dyn Prediction<T>>>;

    fn loss(&self) -> Option<f64> {
        None
    }
}

pub trait Prediction<T>: Debug {
    fn expected(&self) -> Vec<T>;

    fn uncertainty(&self) -> Option<Vec<T>> {
        None
    }
}
