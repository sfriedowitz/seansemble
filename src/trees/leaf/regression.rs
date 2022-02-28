use crate::{
    core::{Learner, Model, TrainingRow},
    learners::{GuessTheMeanLearner, LinearRegressionLearner},
};

#[derive(Clone, Debug)]
pub enum RegressionLeafLearner {
    GuessTheMean { learner: GuessTheMeanLearner },
    LinearRegression { learner: LinearRegressionLearner },
}

impl RegressionLeafLearner {
    pub fn mean(learner: GuessTheMeanLearner) -> Self {
        Self::GuessTheMean { learner }
    }

    pub fn linreg(learner: LinearRegressionLearner) -> Self {
        Self::LinearRegression { learner }
    }
}

impl Learner<f64> for RegressionLeafLearner {
    fn fit(&mut self, data: &[TrainingRow<f64>]) -> Box<dyn Model<f64>> {
        match self {
            Self::GuessTheMean { learner } => learner.fit(data),
            Self::LinearRegression { learner } => learner.fit(data),
        }
    }
}
