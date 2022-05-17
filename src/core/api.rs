use rand::Rng;

use super::{row::FeatureRow, Result, TrainingRow};

pub trait Learner<T> {
    fn fit(&self, data: &[TrainingRow<T>], rng: &mut impl Rng) -> Result<Box<dyn Model<T>>>
    where
        Self: Sized;
}

pub trait Model<T> {
    fn transform(&self, inputs: &[FeatureRow]) -> Result<Box<dyn Prediction<T>>>;

    fn loss(&self) -> Option<f64> {
        None
    }
}

pub trait Prediction<T> {
    fn expected(&self) -> Vec<T>;

    fn uncertainty(&self) -> Option<Vec<T>> {
        None
    }
}
