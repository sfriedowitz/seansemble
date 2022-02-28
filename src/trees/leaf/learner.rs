use crate::{
    core::{Learner, Model, TrainingRow},
    linear::{GuessTheMeanLearner, LinearRegressionLearner},
};

#[derive(Clone, Debug)]
pub enum LeafLearner {
    GuessTheMean { learner: GuessTheMeanLearner },
    LinearRegression { learner: LinearRegressionLearner },
}

impl LeafLearner {
    pub fn mean(learner: GuessTheMeanLearner) -> Self {
        LeafLearner::GuessTheMean { learner }
    }

    pub fn linreg(learner: LinearRegressionLearner) -> Self {
        LeafLearner::LinearRegression { learner }
    }

    pub fn train_leaf(&mut self, data: &[TrainingRow<f64>]) -> Box<dyn Model<f64>> {
        match self {
            LeafLearner::GuessTheMean { learner, .. } => learner.fit(data),
            LeafLearner::LinearRegression { learner, .. } => learner.fit(data),
        }
    }
}
