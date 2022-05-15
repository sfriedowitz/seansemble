use std::fmt::Debug;

use super::{ModelingError, TrainingRow};

pub trait Learner<D, P> {
    fn fit(training_data: &D, parameters: &P) -> Result<Self, ModelingError>
    where
        Self: Sized;
}

// pub trait Model<L: Label>: Debug {
//     fn transform(&self, inputs: &[FeatureRow]) -> Box<dyn Prediction<L>>;

//     fn loss(&self) -> Option<f64> {
//         None
//     }
// }

// pub trait Prediction<L: Label>: Debug {
//     fn expected(&self) -> Vec<L>;

//     fn uncertainty(&self, observational: bool) -> Option<Vec<L>> {
//         None
//     }
// }
