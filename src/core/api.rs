use std::fmt::Debug;

use super::{FeatureRow, Label, TrainingRow};

pub trait Learner<L: Label>: Debug {
    fn fit(&mut self, data: &[TrainingRow<L>]) -> Box<dyn Model<L>>;
}

pub trait Model<L: Label>: Debug {
    fn transform(&self, inputs: &[FeatureRow]) -> Box<dyn Prediction<L>>;

    fn loss(&self) -> Option<f64> {
        None
    }
}

pub trait Prediction<L: Label>: Debug {
    fn expected(&self) -> Vec<L>;

    fn uncertainty(&self, observational: bool) -> Option<Vec<L>> {
        None
    }
}
