use rand::prelude::StdRng;
use rand::SeedableRng;

use crate::core::{Learner, Model, TrainingRow};
use crate::trees::splits::{Split, Splitter};

use super::{DecisionTreeParameters, LeafLearner, ModelNode, TrainingNode};

#[derive(Debug)]
pub struct RegressionTreeLearner {
    splitter: Box<dyn Splitter<f64>>,
    learner: LeafLearner,
    params: DecisionTreeParameters,
    rng: StdRng,
}

impl RegressionTreeLearner {
    pub fn new(
        splitter: Box<dyn Splitter<f64>>,
        learner: LeafLearner,
        params: DecisionTreeParameters,
        seed_rng: Option<&mut StdRng>,
    ) -> Self {
        let rng = match seed_rng {
            Some(r) => SeedableRng::from_rng(r).expect("Seeding RNG failed."),
            None => SeedableRng::from_entropy(),
        };
        Self { splitter, learner, params, rng }
    }

    fn split_internal(
        &mut self,
        data: Vec<TrainingRow<f64>>,
        split: Split,
        delta_impurity: f64,
        num_features: usize,
        remaining_depth: usize,
    ) -> TrainingNode<f64> {
        let (left_data, right_data): (Vec<_>, Vec<_>) =
            data.into_iter().partition(|row| split.turn_left(&row.features));

        let left_child = self.build_child(left_data, num_features, remaining_depth);
        let right_child = self.build_child(right_data, num_features, remaining_depth);

        TrainingNode::internal(
            split,
            Box::new(left_child),
            Box::new(right_child),
            delta_impurity,
            self.params.max_depth - remaining_depth,
        )
    }

    fn build_child(
        &mut self,
        data: Vec<TrainingRow<f64>>,
        num_features: usize,
        remaining_depth: usize,
    ) -> TrainingNode<f64> {
        let splitter = &mut self.splitter;
        let current_depth = self.params.max_depth - remaining_depth;
        let min_instances = self.params.min_leaf_instances;

        if data.len() >= 2 * min_instances && remaining_depth > 0 {
            let (split, delta) = splitter.find_best_split(&data, num_features, min_instances);
            if split != Split::None && delta > self.params.min_purity_increase {
                self.split_internal(data, split, delta, num_features, remaining_depth - 1)
            } else {
                TrainingNode::leaf(data, current_depth)
            }
        } else {
            TrainingNode::leaf(data, current_depth)
        }
    }
}

impl Learner<f64> for RegressionTreeLearner {
    fn fit(&mut self, data: &[TrainingRow<f64>]) -> Box<dyn Model<f64>> {
        let params = self.params;
        let my_data: Vec<TrainingRow<f64>> = data.to_vec();

        let root_node = if params.max_depth == 0 {
            TrainingNode::leaf(my_data, 0)
        } else {
            let data_nf = data[0].features.num_features();
            let actual_nf = data_nf.min(params.num_features);
            let min_count = params.min_leaf_instances;
            let max_depth = params.max_depth;

            let (split, delta) = self.splitter.find_best_split(&my_data, actual_nf, min_count);

            match split {
                Split::None => TrainingNode::leaf(my_data, 0),
                _split if delta < params.min_purity_increase => TrainingNode::leaf(my_data, 0),
                split => self.split_internal(my_data, split, delta, actual_nf, max_depth - 1),
            }
        };

        todo!()
    }
}

pub struct RegressionTreeModel {
    training_node: TrainingNode<f64>,
    model_node: ModelNode<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::linear_training_data;
    use crate::{trees::leaf::GuessTheMeanLearner, trees::splits::RegressionSplitter};

    #[test]
    fn test_regression_tree() {
        let data = linear_training_data(100, &[0.0, 1.0, 2.0, 3.0], 5.0);

        let params: DecisionTreeParameters = Default::default();
        let splitter = RegressionSplitter::new(true, None);
        let leaf_learner = LeafLearner::mean(GuessTheMeanLearner::new(None));

        let mut tree = RegressionTreeLearner::new(Box::new(splitter), leaf_learner, params, None);

        let tic = std::time::SystemTime::now();
        (0..10).for_each(|_| {
            let _ = tree.fit(&data);
        });
        let toc = std::time::SystemTime::now();
        let since = toc.duration_since(tic).expect("");

        println!("{}", since.as_secs_f64() / 10.0);
    }
}
