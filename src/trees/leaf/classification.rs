use crate::{
    core::{Learner, Model, TrainingRow},
    learners::GuessTheMeanLearner,
};

#[derive(Clone, Debug)]
pub enum ClassificationLeafLearner {
    GuessTheMean { learner: GuessTheMeanLearner },
}

impl ClassificationLeafLearner {
    pub fn mean(learner: GuessTheMeanLearner) -> Self {
        Self::GuessTheMean { learner }
    }
}

impl Learner<usize> for ClassificationLeafLearner {
    fn fit(&mut self, data: &[TrainingRow<usize>]) -> Box<dyn Model<usize>> {
        match self {
            Self::GuessTheMean { learner } => learner.fit(data),
        }
    }
}
